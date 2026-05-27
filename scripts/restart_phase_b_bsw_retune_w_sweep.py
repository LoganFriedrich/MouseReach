"""v8.0.x experiment: BSW retune w-sweep (Phase 1 of BSW retune workstream).

Re-sweep boundary sample weight at b=1 on the current corpus + pipeline.
Rationale: BSW b=1/w=0.8 was tuned 2026-05-04 on OLD DLC, mg=2, no
postprocess, symmetric matcher. Since then we've changed: DLC (model 3.1),
merge_gap (0), added leading-trim/trailing-trim/apex-split, matcher
asymmetric (-2/+5), topology classifier matcher-aware. The optimum
(b, w) may have shifted.

Phase 1 grid: b=1 fixed, w in {0.6, 0.7, 0.8, 0.9}. Per cell:
  - 16-fold LOOCV on cal train_pool (cal score)
  - 1 full-corpus train -> infer on 19 holdout videos (hol score)
  - Apply full v8.0.4 postprocess at inference
  - Score with asymmetric matcher + matcher-aware topology

Pre-experiment checklist (per pre_experiment_checklist.md):

1. Cumulative-stacking check (verified 2026-05-26 evening, master 2a59937):
   - Production v8.0.4 stack: BSW b=1/w=0.8 + mg=0 + leading-trim
     T=0.60/N=3 + trailing-trim T=0.60/N=3 + apex-split
     prom=0.12/depth=0.5/peak2<0.85.
   - Asymmetric strict matcher -2/+5, matcher-aware topology classifier.
   - DLC scorer model 3.1, train_pool corpus 2026-05-21_model_3_1_inventory.
   - Live GT (post 2026-05-26 edits).
   - Comparison baseline = v8.0.4 production model bundle. The w=0.8
     cell in this sweep IS a fresh retrain on the new corpus, isolating
     the corpus-change effect.

2. Existing-code-modification check: NO. Inlines train_one_fold logic
   (same pattern as original BSW sweep). mousereach.reach.v8.train,
   mousereach.reach.v8.postprocess UNTOUCHED.

3. Unverified hypotheses (called out):
   - That the optimum has shifted from w=0.8 due to corpus + postprocess
     changes since 2026-05-06.
   - That LOOCV improvements generalize to holdout.
   - That trim and apex-split interact with BSW in non-degenerate ways
     (rather than canceling).

4. FN-direction-reporting: lead each cell's row with both deltas
   (vs v8.0.4 baseline AND vs pure v8.0.0 mg=2 baseline). Topology
   paired with legacy. Cardinal Rule abs_medians both axes both corpora.

5. Framework check: outputs to
   Improvement_Snapshots/reach_detection/v8.0.4_dev_bsw_retune_w_sweep/

6. Branch + tag:
   - Pre-experiment tag: pre-bsw-retune-2026-05-26 at master 2a59937
   - Feature branch: feature/v8-bsw-retune off 2a59937

7. Decision rule (applied per cell):
   ACCEPT if (vs v8.0.4 baseline):
     - Cardinal Rule: start_delta abs_median = 0 AND span_delta abs_median = 0
       on BOTH corpora
     - FN strictly decreases on at least one corpus, non-increasing on the other
     - TP non-decreasing on BOTH corpora
     - MERGED, FRAGMENTED non-increasing on BOTH corpora
   REJECT if any criterion fails.

This script runs Phase 1 only. Phase 2 (2D b/w sweep around any winner
that isn't w=0.8) is in a separate runner. Phase 3 (ship) is gated on
Logan returning to review the results.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
    STRICT_START_TOL_EARLY, STRICT_START_TOL_LATE,
)
from mousereach.reach.v8 import (
    DEFAULT_MERGE_GAP, DEFAULT_MIN_SPAN,
    DEFAULT_TRIM_LK_THRESHOLD, DEFAULT_TRIM_SUSTAIN_N,
    DEFAULT_TRAILING_TRIM_LK_THRESHOLD, DEFAULT_TRAILING_TRIM_SUSTAIN_N,
    DEFAULT_APEX_SPLIT_PROMINENCE, DEFAULT_APEX_SPLIT_DEPTH_MIN,
    DEFAULT_APEX_SPLIT_PEAK2_REL_MAX, DEFAULT_APEX_SPLIT_MIN_DISTANCE,
)
from mousereach.reach.v8.features import (
    feature_columns, extract_features, load_dlc_h5,
)
from mousereach.reach.v8.postprocess import (
    ReachSpan, probabilities_to_reaches,
    trim_leading_sustained_lk, trim_trailing_sustained_lk,
    apex_split_at_trough, compute_paw_mean_lk,
    compute_hand_to_boxl_norm_pos,
)


# ---------- Paths ----------
CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-05-21_model_3_1_inventory"
)
HOLDOUT_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\dlc"
)
GEN_GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\gt"
)
CAL_GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\validation_runs\DLC_2026_03_27\gt"
)
OUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.4_dev_bsw_retune_w_sweep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"

# ---------- Sweep params ----------
BOUNDARY_BUFFER = 1
# Phase 1 grid: 4 cells at b=1. Reordered to put w=0.8 first as a
# verification cell (matches current production BSW), then the other
# three. If w=0.8 produces sane numbers, the rest are guaranteed to
# also run end-to-end.
W_CONFIGS = [0.8, 0.7, 0.9, 0.6]

# Production v8.0.4 postprocess + matcher (frozen)
THRESHOLD = 0.5
MERGE_GAP = 0
MIN_SPAN = 3
TRIM_LEADING_THRESHOLD = DEFAULT_TRIM_LK_THRESHOLD
TRIM_LEADING_SUSTAIN_N = DEFAULT_TRIM_SUSTAIN_N
TRIM_TRAILING_THRESHOLD = DEFAULT_TRAILING_TRIM_LK_THRESHOLD
TRIM_TRAILING_SUSTAIN_N = DEFAULT_TRAILING_TRIM_SUSTAIN_N
APEX_PROMINENCE = DEFAULT_APEX_SPLIT_PROMINENCE
APEX_DEPTH_MIN = DEFAULT_APEX_SPLIT_DEPTH_MIN
APEX_PEAK2_REL_MAX = DEFAULT_APEX_SPLIT_PEAK2_REL_MAX
APEX_MIN_DISTANCE = DEFAULT_APEX_SPLIT_MIN_DISTANCE
SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5

# GBM hyperparams (matching production)
GBM_MAX_ITER = 200
GBM_LEARNING_RATE = 0.05
GBM_MAX_DEPTH = 6
GBM_RANDOM_STATE = 42


# ---------- BSW training (inlined replication of train_one_fold) ----------

def compute_boundary_weights(train_df, n_buffer=1, boundary_weight=0.8):
    sorted_df = train_df.sort_values(["video_id", "frame"])
    rid = sorted_df["reach_id"].to_numpy()
    vid = sorted_df["video_id"].to_numpy()
    n = len(sorted_df)
    transitions = np.zeros(n, dtype=bool)
    if n >= 2:
        same_video = vid[1:] == vid[:-1]
        rid_change = rid[1:] != rid[:-1]
        boundary_pairs = same_video & rid_change
        transitions[1:] |= boundary_pairs
        transitions[:-1] |= boundary_pairs
    dilated = transitions.copy()
    for d in range(1, n_buffer + 1):
        dilated[d:] |= transitions[:-d]
        dilated[:-d] |= transitions[d:]
    weights_sorted = np.ones(n, dtype=np.float32)
    weights_sorted[dilated] = boundary_weight
    weights_series = pd.Series(weights_sorted, index=sorted_df.index)
    return weights_series.reindex(train_df.index).to_numpy()


def train_gbm(train_df, feat_cols, b, w):
    """Train one HGBM with BSW at (b, w)."""
    X = train_df[feat_cols].to_numpy(dtype=np.float32)
    y = train_df["label"].to_numpy(dtype=np.int8)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos > 0 and n_neg > 0:
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)
        class_w = np.where(y == 1, w_pos, w_neg).astype(np.float32)
    else:
        class_w = np.ones(n, dtype=np.float32)
    boundary_w = compute_boundary_weights(
        train_df, n_buffer=b, boundary_weight=w)
    sample_weight = (class_w * boundary_w).astype(np.float32)
    clf = HistGradientBoostingClassifier(
        max_iter=GBM_MAX_ITER, learning_rate=GBM_LEARNING_RATE,
        max_depth=GBM_MAX_DEPTH, random_state=GBM_RANDOM_STATE,
        early_stopping=False,
    )
    clf.fit(X, y, sample_weight=sample_weight)
    return clf


# ---------- v8.0.4 postprocess at inference ----------

def proba_to_algos(proba, paw_lk, norm_pos):
    spans = probabilities_to_reaches(
        proba, threshold=THRESHOLD, merge_gap=MERGE_GAP, min_span=MIN_SPAN)
    spans = trim_leading_sustained_lk(
        spans, paw_lk,
        threshold=TRIM_LEADING_THRESHOLD,
        sustain_n=TRIM_LEADING_SUSTAIN_N,
        min_span=MIN_SPAN)
    spans = trim_trailing_sustained_lk(
        spans, paw_lk,
        threshold=TRIM_TRAILING_THRESHOLD,
        sustain_n=TRIM_TRAILING_SUSTAIN_N,
        min_span=MIN_SPAN)
    spans = apex_split_at_trough(
        spans, norm_pos,
        prominence=APEX_PROMINENCE,
        depth_min=APEX_DEPTH_MIN,
        peak2_rel_max=APEX_PEAK2_REL_MAX,
        min_distance=APEX_MIN_DISTANCE,
        min_span=MIN_SPAN)
    return sorted({(int(r.start_frame), int(r.end_frame)) for r in spans})


# ---------- Matcher + topology (matcher-aware) ----------

def overlap_exists(a_s, a_e, b_s, b_e):
    return not (a_e < b_s or a_s > b_e)


def greedy_match(algos, gts):
    candidates = []
    for ai, (a_s, a_e) in enumerate(algos):
        algo_span = a_e - a_s + 1
        for gi, (g_s, g_e) in enumerate(gts):
            gt_span = g_e - g_s + 1
            sd = a_s - g_s
            pd_ = algo_span - gt_span
            sp_tol = max(SPAN_TOL_FRAC * gt_span, SPAN_TOL_MIN)
            if (-STRICT_START_TOL_EARLY <= sd <= STRICT_START_TOL_LATE
                    and abs(pd_) <= sp_tol):
                candidates.append((abs(sd), ai, gi, sd, pd_))
    candidates.sort()
    matched = set()
    used_a, used_g = set(), set()
    tp_sd, tp_pd = [], []
    for _, ai, gi, sd, pd_ in candidates:
        if ai in used_a or gi in used_g:
            continue
        used_a.add(ai); used_g.add(gi)
        matched.add((ai, gi))
        tp_sd.append(sd); tp_pd.append(pd_)
    return matched, tp_sd, tp_pd


def classify_matcher_aware(algos, gts, matched):
    parent = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry: parent[rx] = ry
    for i in range(len(algos)): parent[('a', i)] = ('a', i)
    for j in range(len(gts)): parent[('g', j)] = ('g', j)
    for i, a in enumerate(algos):
        for j, g in enumerate(gts):
            if overlap_exists(*a, *g):
                union(('a', i), ('g', j))
    by_root = defaultdict(list)
    for n in parent: by_root[find(n)].append(n)
    counts = defaultdict(int)
    tol_pair_events = 0
    for nodes in by_root.values():
        a_idx = {i for k, i in nodes if k == 'a'}
        g_idx = {j for k, j in nodes if k == 'g'}
        na, ng = len(a_idx), len(g_idx)
        if na == 1 and ng == 0:
            counts['FALSE_POSITIVE'] += 1
        elif na == 0 and ng == 1:
            counts['FALSE_NEGATIVE'] += 1
        elif na == 1 and ng == 1:
            i = next(iter(a_idx)); j = next(iter(g_idx))
            if (i, j) in matched:
                counts['TP'] += 1
            else:
                tol_pair_events += 2
        elif na == 1 and ng >= 2:
            counts['MERGED'] += 1
        elif na >= 2 and ng == 1:
            counts['FRAGMENTED'] += 1
        elif na >= 2 and ng >= 2:
            matched_in = [(ai, gj) for (ai, gj) in matched
                          if ai in a_idx and gj in g_idx]
            for _ in matched_in:
                counts['TP'] += 1
            unmatched_a = a_idx - {ai for ai, _ in matched_in}
            unmatched_g = g_idx - {gj for _, gj in matched_in}
            soft_paired = set()
            for ai in sorted(unmatched_a):
                best_gj = None; best_ol = 0
                for gj in sorted(unmatched_g - soft_paired):
                    a_s, a_e = algos[ai]
                    g_s, g_e = gts[gj]
                    s = max(a_s, g_s); e = min(a_e, g_e)
                    ol = max(0, e - s + 1)
                    if ol > best_ol:
                        best_ol = ol; best_gj = gj
                if best_gj is not None and best_ol > 0:
                    tol_pair_events += 2
                    soft_paired.add(best_gj)
                else:
                    counts['FALSE_POSITIVE'] += 1
            for gj in sorted(unmatched_g - soft_paired):
                counts['FALSE_NEGATIVE'] += 1
    counts['TOLERANCE_ERROR_pairs'] = tol_pair_events // 2
    return dict(counts)


# ---------- Data loaders ----------

def load_live_gt(corpus_label, vid):
    if corpus_label == "cal":
        gt_path = CAL_GT_DIR / f"{vid}_unified_ground_truth.json"
    else:
        gt_path = GEN_GT_DIR / f"{vid}_unified_ground_truth.json"
    if not gt_path.exists():
        return []
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    rlist = data.get("reaches", {}).get("reaches", [])
    return sorted(set(
        (int(r["start_frame"]), int(r["end_frame"]))
        for r in rlist if not r.get("exclude_from_analysis")
    ))


PARQUET_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]


def cal_per_video_aux(df):
    """For cal videos: extract paw_lk and norm_pos from the parquet directly,
    so we don't have to load DLC h5s for cal."""
    aux = {}
    for vid, g in df.groupby("video_id", sort=False):
        g = g.sort_values("frame").reset_index(drop=True)
        max_frame = int(g["frame"].max())
        paw_lk = np.full(max_frame + 1, np.nan, dtype=np.float32)
        paw_lk_local = g[PARQUET_LK_COLS].to_numpy(dtype=np.float32).mean(axis=1)
        paw_lk[g["frame"].to_numpy()] = paw_lk_local
        # norm_pos from BoxL/Hand positions
        keypoints = ("RightHand", "RHLeft", "RHOut", "RHRight", "BOXL", "BOXR")
        dlc_like = pd.DataFrame({
            f"{kp}_x": g[f"{kp}_x"].to_numpy()
            for kp in keypoints
        } | {
            f"{kp}_y": g[f"{kp}_y"].to_numpy()
            for kp in keypoints
        })
        np_local = compute_hand_to_boxl_norm_pos(dlc_like)
        norm_pos = np.full(max_frame + 1, np.nan, dtype=np.float32)
        norm_pos[g["frame"].to_numpy()] = np_local
        aux[vid] = {"paw_lk": paw_lk, "norm_pos": norm_pos}
    return aux


# ---------- Cell evaluation ----------

def evaluate_cell(b, w, df, feat_cols, train_pool_ids, cal_aux,
                  holdout_data):
    """For one (b, w): run 16-fold LOOCV on cal, train full-corpus for hol."""
    print(f"\n=== Cell b={b}, w={w} ===", flush=True)
    t_start = time.time()

    # --- Cal LOOCV ---
    cal_totals = {"tp": 0, "fp": 0, "fn": 0}
    cal_topo = defaultdict(int)
    cal_start_deltas = []
    cal_span_deltas = []
    cal_per_video = {}
    for fold_idx, val_vid in enumerate(train_pool_ids):
        train_ids = [v for v in train_pool_ids if v != val_vid]
        train_mask = df["video_id"].isin(train_ids) & df["exhaustive"]
        train_df = df.loc[train_mask]
        val_df = df.loc[df["video_id"] == val_vid].sort_values("frame")
        t0 = time.time()
        clf = train_gbm(train_df, feat_cols, b, w)
        Xv = val_df[feat_cols].to_numpy(dtype=np.float32)
        proba = clf.predict_proba(Xv)[:, 1]
        # Reconstruct full-frame proba array (val_df.frame might not be 0..N)
        max_frame = int(val_df["frame"].max())
        proba_arr = np.zeros(max_frame + 1, dtype=np.float32)
        proba_arr[val_df["frame"].to_numpy()] = proba
        paw_lk = cal_aux[val_vid]["paw_lk"]
        norm_pos = cal_aux[val_vid]["norm_pos"]
        # Align lengths
        L = min(len(proba_arr), len(paw_lk), len(norm_pos))
        algos = proba_to_algos(proba_arr[:L], paw_lk[:L], norm_pos[:L])
        gts = load_live_gt("cal", val_vid)
        matched, tp_sd, tp_pd = greedy_match(algos, gts)
        tp = len(matched); fp = len(algos) - tp; fn = len(gts) - tp
        cal_totals["tp"] += tp; cal_totals["fp"] += fp; cal_totals["fn"] += fn
        cal_start_deltas.extend(tp_sd)
        cal_span_deltas.extend(tp_pd)
        tc = classify_matcher_aware(algos, gts, matched)
        for k, v in tc.items():
            cal_topo[k] += v
        cal_per_video[val_vid] = {"tp": tp, "fp": fp, "fn": fn,
                                    "n_algo": len(algos), "n_gt": len(gts)}
        print(f"  cal fold {fold_idx+1}/{len(train_pool_ids)}: "
              f"{val_vid}  TP={tp} FP={fp} FN={fn}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    # --- Full-corpus train for holdout ---
    print(f"  cal LOOCV done ({time.time()-t_start:.1f}s); "
          f"training full-corpus for holdout...", flush=True)
    t0 = time.time()
    full_train_df = df.loc[df["exhaustive"]]
    clf_full = train_gbm(full_train_df, feat_cols, b, w)
    print(f"  full-corpus train: {time.time()-t0:.1f}s", flush=True)

    # --- Holdout scoring ---
    hol_totals = {"tp": 0, "fp": 0, "fn": 0}
    hol_topo = defaultdict(int)
    hol_start_deltas = []
    hol_span_deltas = []
    hol_per_video = {}
    for vid, vd in holdout_data.items():
        Xh = vd["X"]
        proba = clf_full.predict_proba(Xh)[:, 1]
        algos = proba_to_algos(proba, vd["paw_lk"], vd["norm_pos"])
        gts = vd["gts"]
        matched, tp_sd, tp_pd = greedy_match(algos, gts)
        tp = len(matched); fp = len(algos) - tp; fn = len(gts) - tp
        hol_totals["tp"] += tp; hol_totals["fp"] += fp; hol_totals["fn"] += fn
        hol_start_deltas.extend(tp_sd)
        hol_span_deltas.extend(tp_pd)
        tc = classify_matcher_aware(algos, gts, matched)
        for k, v in tc.items():
            hol_topo[k] += v
        hol_per_video[vid] = {"tp": tp, "fp": fp, "fn": fn,
                                "n_algo": len(algos), "n_gt": len(gts)}

    cal_sd_med = int(np.median([abs(d) for d in cal_start_deltas])) if cal_start_deltas else None
    cal_pd_med = int(np.median([abs(d) for d in cal_span_deltas])) if cal_span_deltas else None
    hol_sd_med = int(np.median([abs(d) for d in hol_start_deltas])) if hol_start_deltas else None
    hol_pd_med = int(np.median([abs(d) for d in hol_span_deltas])) if hol_span_deltas else None

    print(f"\n  Cal: TP={cal_totals['tp']} FP={cal_totals['fp']} FN={cal_totals['fn']}  "
          f"abs_med start={cal_sd_med} span={cal_pd_med}")
    print(f"  Cal topology: {dict(cal_topo)}")
    print(f"  Hol: TP={hol_totals['tp']} FP={hol_totals['fp']} FN={hol_totals['fn']}  "
          f"abs_med start={hol_sd_med} span={hol_pd_med}")
    print(f"  Hol topology: {dict(hol_topo)}")
    print(f"  Cell total time: {time.time()-t_start:.1f}s", flush=True)

    return {
        "b": b, "w": w,
        "cal": {"totals": cal_totals, "topology": dict(cal_topo),
                "start_delta_abs_median": cal_sd_med,
                "span_delta_abs_median": cal_pd_med,
                "per_video": cal_per_video},
        "hol": {"totals": hol_totals, "topology": dict(hol_topo),
                "start_delta_abs_median": hol_sd_med,
                "span_delta_abs_median": hol_pd_med,
                "per_video": hol_per_video},
        "elapsed_seconds": time.time() - t_start,
    }


# ---------- Main ----------

def main():
    print("=" * 70)
    print(f"BSW RETUNE PHASE 1 -- w sweep at b={BOUNDARY_BUFFER}")
    print(f"Grid: w in {W_CONFIGS}")
    print("=" * 70)
    print()

    print(f"Loading cal corpus from {CORPUS_DIR}...", flush=True)
    df = pd.read_parquet(CORPUS_DIR / "phase_b_dataset" / "train_pool.parquet")
    folds_def = json.loads((CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))
    train_pool_ids = folds_def["train_pool"]["video_ids"]
    feat_cols = feature_columns()
    exh_set = set(df.loc[df["exhaustive"], "video_id"].unique().tolist())
    eligible_val = [v for v in train_pool_ids if v in exh_set]
    print(f"  Train pool: {len(train_pool_ids)} videos, "
          f"{len(eligible_val)} exhaustive (LOOCV folds)", flush=True)
    print(f"  Features: {len(feat_cols)}", flush=True)
    print(f"Computing cal per-video aux (paw_lk, norm_pos)...", flush=True)
    cal_aux = cal_per_video_aux(df)

    print(f"Loading holdout 19 (DLC h5s)...", flush=True)
    holdout_data = {}
    for dlc_path in sorted(HOLDOUT_DLC_DIR.glob(f"*{DLC_SUFFIX}.h5")):
        vid = dlc_path.stem.replace(DLC_SUFFIX, "")
        dlc = load_dlc_h5(dlc_path)
        feats = extract_features(dlc)
        X = feats[feat_cols].to_numpy(dtype="float32")
        paw_lk = compute_paw_mean_lk(dlc)
        norm_pos = compute_hand_to_boxl_norm_pos(dlc)
        gts = load_live_gt("hol", vid)
        holdout_data[vid] = {"X": X, "paw_lk": paw_lk, "norm_pos": norm_pos, "gts": gts}
        print(f"  {vid}: X={X.shape}, gts={len(gts)}", flush=True)
    print()

    results = {"cells": {}, "run_timestamp_start": datetime.utcnow().isoformat() + "Z"}
    for w in W_CONFIGS:
        cell_result = evaluate_cell(BOUNDARY_BUFFER, w, df, feat_cols,
                                      eligible_val, cal_aux, holdout_data)
        results["cells"][f"b{BOUNDARY_BUFFER}_w{w}"] = cell_result
        # Save after each cell (so we don't lose all work if interrupted)
        results["run_timestamp_last_cell"] = datetime.utcnow().isoformat() + "Z"
        (OUT_DIR / "metrics" / "sweep_results.json").write_text(
            json.dumps(results, indent=2, default=int), encoding="utf-8")

    # Summary
    print()
    print("=" * 140)
    print(f"SUMMARY (Phase 1 w sweep at b={BOUNDARY_BUFFER})")
    print("=" * 140)
    print(f"{'w':<5} | {'cal_TP':>6} {'cal_FP':>6} {'cal_FN':>6} {'cal_sd':>6} {'cal_pd':>6} "
          f"| {'hol_TP':>6} {'hol_FP':>6} {'hol_FN':>6} {'hol_sd':>6} {'hol_pd':>6} | {'time':>5}s")
    print("-" * 140)
    for w in W_CONFIGS:
        r = results["cells"][f"b{BOUNDARY_BUFFER}_w{w}"]
        ct = r["cal"]["totals"]; ht = r["hol"]["totals"]
        print(f"{w:<5} | {ct['tp']:>6} {ct['fp']:>6} {ct['fn']:>6} "
              f"{r['cal']['start_delta_abs_median']:>6} "
              f"{r['cal']['span_delta_abs_median']:>6} | "
              f"{ht['tp']:>6} {ht['fp']:>6} {ht['fn']:>6} "
              f"{r['hol']['start_delta_abs_median']:>6} "
              f"{r['hol']['span_delta_abs_median']:>6} | "
              f"{r['elapsed_seconds']:.0f}")
    print()
    print(f"Wrote: {OUT_DIR / 'metrics' / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
