"""v8.0.x experiment: GBM max_iter sweep (Phase 1 of hyperparam retune).

Sweep GBM max_iter at production BSW (b=1, w=0.8), learning_rate=0.05,
max_depth=6. Phase 1 grid: max_iter in {150, 200, 300, 400, 500}. Phase 2
(max_depth) is a separate runner triggered only if Phase 1 shows movement
away from max_iter=200. Phase 3 (learning_rate) likewise.

Pre-experiment checklist (per pre_experiment_checklist.md):

1. Cumulative-stacking check (verified 2026-05-27, master 4af1e33):
   - Production v8.0.4 stack: BSW b=1/w=0.8 (production model bundle)
     + mg=0 + leading-trim T=0.60/N=3 + trailing-trim T=0.60/N=3 +
     apex-split prom=0.12/depth=0.5/peak2<0.85.
   - Asymmetric strict matcher -2/+5, matcher-aware topology classifier.
   - DLC model 3.1, NEW corpus 2026-05-21_model_3_1_inventory.
   - Live GT (post 2026-05-26 edits).
   - BSW retune (yesterday) confirmed w=0.8 is still Pareto-optimal
     within-sweep on current corpus.
   - Comparison baselines: report TWO deltas per cell --
       (a) vs v8.0.4 production joblib (will include the ~19 hol TP
           corpus-gap from yesterday's BSW retune)
       (b) vs within-sweep production-hyperparam cell (max_iter=200)
           -- the clean comparison

2. Existing-code-modification check: NO. Inline train_gbm replication
   with hyperparameter overrides. train.py, postprocess.py, features.py
   UNTOUCHED. If accepted, would integrate into train.py defaults.

3. Unverified hypotheses (called out):
   - That increasing max_iter recovers some SUB_MIN_SPAN FNs by giving
     the model more chances to learn weak signals.
   - That production max_iter=200 is at or near the elbow of the
     bias-variance curve on this dataset (in which case sweep returns
     no winner -- expected outcome).
   - The ~19 TP corpus-gap from yesterday's BSW retune will persist
     across hyperparam variants (not addressable by hyperparams alone).

4. FN-direction-reporting: lead each cell with TWO FN deltas
   (vs v8.0.4 production AND vs within-sweep max_iter=200 baseline).
   Topology paired with legacy. Cardinal Rule abs_medians both axes
   both corpora.

5. Framework check: outputs to
   Improvement_Snapshots/reach_detection/v8.0.4_dev_gbm_max_iter_sweep/

6. Branch + tag:
   - Pre-experiment tag: pre-gbm-hyperparam-2026-05-27 at master 4af1e33
   - Feature branch: feature/v8-gbm-hyperparam off 4af1e33

7. Decision rule (per cell, applied to WITHIN-SWEEP max_iter=200 baseline):
   ACCEPT if:
     - Cardinal Rule: start_delta abs_median = 0 AND span_delta abs_median = 0
       on BOTH corpora
     - FN strictly decreases on at least one corpus, non-increasing on the other
     - TP non-decreasing on BOTH corpora
     - MERGED, FRAGMENTED non-increasing on BOTH corpora
     - NOT materially worse than v8.0.4 production (no >10 TP regression
       on holdout vs joblib)
   REJECT if any criterion fails.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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
    r"\reach_detection\v8.0.4_dev_gbm_max_iter_sweep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"

# ---------- Sweep params ----------
# Phase 1: max_iter sweep. Reordered to put production value (200) first
# as the within-sweep baseline cell.
MAX_ITER_CONFIGS = [200, 150, 300, 400, 500]

# Fixed (production) hyperparams
LEARNING_RATE = 0.05
MAX_DEPTH = 6
RANDOM_STATE = 42
# Fixed BSW (yesterday's retune confirmed w=0.8 still optimal)
BOUNDARY_BUFFER = 1
BOUNDARY_WEIGHT = 0.8

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


# ---------- BSW training (inlined replication) ----------

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


def train_gbm(train_df, feat_cols, max_iter, learning_rate, max_depth):
    """Train one HGBM with production BSW + given hyperparams."""
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
        train_df, n_buffer=BOUNDARY_BUFFER, boundary_weight=BOUNDARY_WEIGHT)
    sample_weight = (class_w * boundary_w).astype(np.float32)
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=learning_rate,
        max_depth=max_depth, random_state=RANDOM_STATE,
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
    aux = {}
    for vid, g in df.groupby("video_id", sort=False):
        g = g.sort_values("frame").reset_index(drop=True)
        max_frame = int(g["frame"].max())
        paw_lk = np.full(max_frame + 1, np.nan, dtype=np.float32)
        paw_lk_local = g[PARQUET_LK_COLS].to_numpy(dtype=np.float32).mean(axis=1)
        paw_lk[g["frame"].to_numpy()] = paw_lk_local
        keypoints = ("RightHand", "RHLeft", "RHOut", "RHRight", "BOXL", "BOXR")
        dlc_like = pd.DataFrame({
            f"{kp}_x": g[f"{kp}_x"].to_numpy() for kp in keypoints
        } | {
            f"{kp}_y": g[f"{kp}_y"].to_numpy() for kp in keypoints
        })
        np_local = compute_hand_to_boxl_norm_pos(dlc_like)
        norm_pos = np.full(max_frame + 1, np.nan, dtype=np.float32)
        norm_pos[g["frame"].to_numpy()] = np_local
        aux[vid] = {"paw_lk": paw_lk, "norm_pos": norm_pos}
    return aux


# ---------- Cell evaluation ----------

def evaluate_cell(max_iter, df, feat_cols, train_pool_ids, cal_aux,
                  holdout_data):
    """For one max_iter: run 16-fold LOOCV on cal, train full-corpus for hol."""
    print(f"\n=== Cell max_iter={max_iter} (lr={LEARNING_RATE}, depth={MAX_DEPTH}) ===",
          flush=True)
    t_start = time.time()

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
        clf = train_gbm(train_df, feat_cols,
                        max_iter=max_iter,
                        learning_rate=LEARNING_RATE,
                        max_depth=MAX_DEPTH)
        Xv = val_df[feat_cols].to_numpy(dtype=np.float32)
        proba = clf.predict_proba(Xv)[:, 1]
        max_frame = int(val_df["frame"].max())
        proba_arr = np.zeros(max_frame + 1, dtype=np.float32)
        proba_arr[val_df["frame"].to_numpy()] = proba
        paw_lk = cal_aux[val_vid]["paw_lk"]
        norm_pos = cal_aux[val_vid]["norm_pos"]
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

    print(f"  cal LOOCV done ({time.time()-t_start:.1f}s); "
          f"training full-corpus for holdout...", flush=True)
    t0 = time.time()
    full_train_df = df.loc[df["exhaustive"]]
    clf_full = train_gbm(full_train_df, feat_cols,
                          max_iter=max_iter,
                          learning_rate=LEARNING_RATE,
                          max_depth=MAX_DEPTH)
    print(f"  full-corpus train: {time.time()-t0:.1f}s", flush=True)

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
        "max_iter": max_iter,
        "learning_rate": LEARNING_RATE,
        "max_depth": MAX_DEPTH,
        "bsw_b": BOUNDARY_BUFFER, "bsw_w": BOUNDARY_WEIGHT,
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


def main():
    print("=" * 70)
    print(f"GBM HYPERPARAM PHASE 1 -- max_iter sweep at lr={LEARNING_RATE} depth={MAX_DEPTH}")
    print(f"Grid: max_iter in {MAX_ITER_CONFIGS}")
    print(f"BSW fixed at b={BOUNDARY_BUFFER}, w={BOUNDARY_WEIGHT} (yesterday's retune confirmed optimum)")
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
    for max_iter in MAX_ITER_CONFIGS:
        cell_result = evaluate_cell(max_iter, df, feat_cols, eligible_val,
                                      cal_aux, holdout_data)
        results["cells"][f"max_iter_{max_iter}"] = cell_result
        results["run_timestamp_last_cell"] = datetime.utcnow().isoformat() + "Z"
        (OUT_DIR / "metrics" / "sweep_results.json").write_text(
            json.dumps(results, indent=2, default=int), encoding="utf-8")

    print()
    print("=" * 140)
    print(f"SUMMARY (Phase 1 max_iter sweep at lr={LEARNING_RATE} depth={MAX_DEPTH})")
    print("=" * 140)
    print(f"{'max_iter':<10} | {'cal_TP':>6} {'cal_FP':>6} {'cal_FN':>6} {'cal_sd':>6} {'cal_pd':>6} "
          f"| {'hol_TP':>6} {'hol_FP':>6} {'hol_FN':>6} {'hol_sd':>6} {'hol_pd':>6} | {'time':>5}s")
    print("-" * 140)
    for max_iter in MAX_ITER_CONFIGS:
        r = results["cells"][f"max_iter_{max_iter}"]
        ct = r["cal"]["totals"]; ht = r["hol"]["totals"]
        print(f"{max_iter:<10} | {ct['tp']:>6} {ct['fp']:>6} {ct['fn']:>6} "
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
