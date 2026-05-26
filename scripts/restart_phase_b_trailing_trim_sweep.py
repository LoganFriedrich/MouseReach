"""
v8.0.x experiment: trailing-trim postprocess sweep.

Symmetric to the v8.0.2 leading-trim. Walk inward from each reach's END
frame; if sustain_n consecutive trailing frames have paw_mean_lk below
threshold, trim them. Targets the "hold-during-extension" mechanism where
the mouse extends, grasps the box / holds the paw stationary for ~5-20
frames with partial visibility (paw_lk 0.3-0.6), then retracts. GBM
keeps emitting through the hold -> algo span overshoots GT_end ->
TOLERANCE_ERROR (SPAN-LONG) or FRAGMENTED (if algo splits at the hold).

This is the same mechanism we found in:
  - HOL CNT0303_P2 cid=135 (FRAGMENTED with 12f gap during hold)
  - HOL CNT0407_P3 cid=259 (FRAGMENTED with 19f gap during hold)
  - HOL CNT0316_P3 x6 (TOLERANCE_ERROR SPAN-LONG, all algos overshoot
    GT_end by 13-21 frames into hold phase)

Pre-experiment checklist (per pre_experiment_checklist.md):

1. Cumulative-stacking check (verified 2026-05-26):
   - Production v8.0.3 reach detector + matcher-aware topology classifier.
   - Asymmetric strict matcher -2/+5 on live GT.
   - Comparison baseline: v8.0.3 with no trailing-trim.
       Cal filtered (post 2026-05-26 GT): TP=2309 TOL=21 MERGED=10 FRAG=5 FP=10 FN=34
       Hol filtered:                        TP=3659 TOL=11 MERGED=3  FRAG=8 FP=12 FN=27

2. Existing-code-modification check: NO. Trailing-trim implemented inline
   in this runner. If accepted, would integrate into mousereach.reach.v8
   alongside the leading-trim.

3. Unverified hypotheses:
   - That the trailing-trim with paw_lk threshold in [0.60-0.75] converts
     SPAN-LONG TOLERANCE_ERROR events to TPs (CNT0316_P3 cluster).
   - That it reduces some FRAGMENTED events by trimming the algo span
     before the hold, so the second algo piece either becomes a stranded
     FP or gets absorbed (need to inspect).
   - That sustain_n=3 protects clean TPs from over-trimming (mirroring
     the leading-trim's protection).

4. FN-direction-reporting:
   - Lead with FN delta vs cumulative best (post 2026-05-26 GT, no
     trailing-trim).
   - Both legacy (matcher counts) and topology paired.
   - Particular focus on TOLERANCE_ERROR_pair and FRAGMENTED deltas
     (these are the target classes).

5. Framework check:
   - Output to v8.0.3_dev_trailing_trim_sweep/.
   - sweep_results.json + RESULTS.md.

6. Branch + tag:
   - feature/v8-trailing-trim
   - Tag: pre-trailing-trim-2026-05-26

7. Decision rule (per threshold):
   ACCEPT if:
     - TP rises OR holds (no material drop) on BOTH corpora
     - TOLERANCE_ERROR_pairs drops on BOTH corpora
     - MERGED non-increasing both corpora
     - FRAGMENTED non-increasing both corpora
     - start_delta abs_median holds at 0 (we only trim from the END,
       so start should be unchanged unless the trim shortens the span
       below MIN_SPAN and we drop the reach -- which could affect TP
       count but not start_delta of remaining TPs)
   REJECT if:
     - TP drops materially (>5 events) on either corpus
     - Any other class regresses materially

   Best config = highest TOL+FRAG reduction with stable TP.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
    STRICT_START_TOL_EARLY, STRICT_START_TOL_LATE,
)
from mousereach.reach.v8.postprocess import (
    ReachSpan, trim_leading_sustained_lk, compute_paw_mean_lk,
    apex_split_at_trough, compute_hand_to_boxl_norm_pos,
)


CAL_LOOCV_SOURCE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.1_model_3_1_baseline_loocv\metrics\loocv_aggregate.json"
)
CAL_PARQUET = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-05-21_model_3_1_inventory\phase_b_dataset\train_pool.parquet"
)
HOLDOUT_ALGO_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_holdout_generalization_merge_gap_0\algo_outputs_v8.0.0_mg0"
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
    r"\reach_detection\v8.0.3_dev_trailing_trim_sweep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"
PARQUET_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]
PARQUET_POS_COLS = ([f"{kp}_{ax}" for kp in
                      ("RightHand", "RHLeft", "RHOut", "RHRight",
                        "BOXL", "BOXR") for ax in ("x", "y")])

# v8.0.2 production trim
TRIM_LEADING_THRESHOLD = 0.60
TRIM_LEADING_SUSTAIN_N = 3
TRIM_MIN_SPAN = 3

# v8.0.3 production apex split
APEX_PROMINENCE = 0.12
APEX_DEPTH_MIN = 0.5
APEX_PEAK2_REL_MAX = 0.85
APEX_MIN_DISTANCE = 4
APEX_MIN_SPAN = 3

# Sweep: trailing-trim threshold (paw_lk), sustain N = 3 (mirror leading)
TRAILING_TRIM_SUSTAIN_N = 3
TRAILING_THRESHOLDS = [0.60, 0.65, 0.70, 0.75]

SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5


def smooth(x, w=5):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)


def compute_norm_pos_from_df(df):
    hand_x = smooth(np.mean([df[f"{kp}_x"].to_numpy() for kp in
                              ("RightHand", "RHLeft", "RHOut", "RHRight")], axis=0))
    hand_y = smooth(np.mean([df[f"{kp}_y"].to_numpy() for kp in
                              ("RightHand", "RHLeft", "RHOut", "RHRight")], axis=0))
    boxl_x = smooth(df["BOXL_x"].to_numpy())
    boxl_y = smooth(df["BOXL_y"].to_numpy())
    boxr_x = smooth(df["BOXR_x"].to_numpy())
    boxr_y = smooth(df["BOXR_y"].to_numpy())
    apparatus = np.sqrt((boxr_x - boxl_x) ** 2 + (boxr_y - boxl_y) ** 2)
    dist_boxl = np.sqrt((hand_x - boxl_x) ** 2 + (hand_y - boxl_y) ** 2)
    return dist_boxl / np.maximum(apparatus, 1e-3)


def load_dlc_h5(path):
    df = pd.read_hdf(path)
    df.columns = ['_'.join(col[1:]) for col in df.columns]
    return df


# ---- Trailing-trim postprocess (symmetric to leading) ----

def trim_trailing_sustained_lk(reaches, paw_lk, threshold, sustain_n=3, min_span=3):
    """Walk inward from each reach's end. Trim trailing frames where
    paw_mean_lk is below threshold for sustain_n consecutive frames.
    Reaches reduced below min_span by trimming are dropped.

    Mirror of trim_leading_sustained_lk.
    """
    if not reaches:
        return []
    n_frames = len(paw_lk)
    out = []
    for r in reaches:
        s, e = r.start_frame, r.end_frame
        new_e = e
        while new_e >= s:
            window_start = new_e - sustain_n + 1
            if window_start < s:
                # Not enough frames left for the sustain check
                break
            if window_start < 0 or new_e >= n_frames:
                break
            window = paw_lk[window_start:new_e + 1]
            if np.any(np.isnan(window)):
                break
            if np.any(window >= threshold):
                # At least one confident frame in window; stop trimming
                break
            new_e -= 1
        if new_e - s + 1 >= min_span:
            out.append(ReachSpan(start_frame=s, end_frame=new_e))
    return out


# ---- Matcher + topology (same as the matcher-aware-topology snapshot) ----

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
    """Locked rules from 2026-05-22 topology refactor."""
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
    tol_pair_events = 0  # raw event count for TOL; divide by 2 for pairs
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
                tol_pair_events += 2  # 1 FP event + 1 FN event = 1 pair
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


# ---- Data loading ----

def load_live_gt(corpus_label, video_id):
    if corpus_label == "calibration_loocv":
        gt_path = CAL_GT_DIR / f"{video_id}_unified_ground_truth.json"
    else:
        gt_path = GEN_GT_DIR / f"{video_id}_unified_ground_truth.json"
    if not gt_path.exists():
        return []
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    rlist = data.get("reaches", {}).get("reaches", [])
    return sorted(set(
        (int(r["start_frame"]), int(r["end_frame"]))
        for r in rlist if not r.get("exclude_from_analysis")
    ))


def load_calibration():
    print("Loading calibration LOOCV + parquet...", flush=True)
    data = json.loads(CAL_LOOCV_SOURCE.read_text(encoding="utf-8"))
    raw = data["raw_results"]
    algos = defaultdict(set)
    for r in raw:
        if r["algo_start_frame"] >= 0:
            algos[r["video_id"]].add((int(r["algo_start_frame"]),
                                       int(r["algo_end_frame"])))
    df = pd.read_parquet(CAL_PARQUET,
                          columns=["video_id", "frame"] + PARQUET_LK_COLS + PARQUET_POS_COLS)
    out = {}
    for vid, grp in df.groupby("video_id", sort=False):
        g = grp.sort_values("frame").reset_index(drop=True)
        paw_lk_matrix = g[PARQUET_LK_COLS].to_numpy(dtype=np.float32)
        paw_mean_lk = paw_lk_matrix.mean(axis=1)
        mx_frame = int(g["frame"].max())
        lk_arr = np.full(mx_frame + 1, np.nan, dtype=np.float32)
        lk_arr[g["frame"].to_numpy()] = paw_mean_lk
        norm_pos = compute_norm_pos_from_df(g)
        np_arr = np.full(mx_frame + 1, np.nan, dtype=np.float32)
        np_arr[g["frame"].to_numpy()] = norm_pos
        out[vid] = {
            "algos_v801": sorted(algos.get(vid, set())),
            "gts": load_live_gt("calibration_loocv", vid),
            "paw_lk": lk_arr,
            "norm_pos": np_arr,
        }
    print(f"  {len(out)} calibration videos loaded")
    return out


def load_holdout():
    print("Loading holdout outputs + DLC + GT...", flush=True)
    out = {}
    for algo_path in sorted(HOLDOUT_ALGO_DIR.glob("*_reaches.json")):
        vid = algo_path.stem.replace("_reaches", "")
        dlc_path = HOLDOUT_DLC_DIR / f"{vid}{DLC_SUFFIX}.h5"
        if not dlc_path.exists():
            continue
        adata = json.loads(algo_path.read_text(encoding="utf-8"))
        algos_v801 = sorted(set(
            (int(r["start_frame"]), int(r["end_frame"]))
            for r in adata.get("reaches", [])
        ))
        dlc = load_dlc_h5(dlc_path)
        paw_lk = compute_paw_mean_lk(dlc)
        norm_pos = compute_hand_to_boxl_norm_pos(dlc)
        out[vid] = {
            "algos_v801": algos_v801,
            "gts": load_live_gt("holdout_2026_05_11", vid),
            "paw_lk": paw_lk,
            "norm_pos": norm_pos,
        }
    print(f"  {len(out)} holdout videos loaded")
    return out


# ---- Pipeline ----

def apply_pipeline(algos_v801, paw_lk, norm_pos, trailing_threshold):
    """v8.0.1 algos -> leading trim -> trailing trim (NEW) -> apex split."""
    spans = [ReachSpan(start_frame=s, end_frame=e) for s, e in algos_v801]
    # Leading trim (v8.0.2 production)
    spans = trim_leading_sustained_lk(
        spans, paw_lk,
        threshold=TRIM_LEADING_THRESHOLD,
        sustain_n=TRIM_LEADING_SUSTAIN_N,
        min_span=TRIM_MIN_SPAN)
    # NEW: trailing trim (this experiment)
    if trailing_threshold > 0:
        spans = trim_trailing_sustained_lk(
            spans, paw_lk,
            threshold=trailing_threshold,
            sustain_n=TRAILING_TRIM_SUSTAIN_N,
            min_span=TRIM_MIN_SPAN)
    # Apex split (v8.0.3 production)
    spans = apex_split_at_trough(
        spans, norm_pos,
        prominence=APEX_PROMINENCE,
        depth_min=APEX_DEPTH_MIN,
        peak2_rel_max=APEX_PEAK2_REL_MAX,
        min_distance=APEX_MIN_DISTANCE,
        min_span=APEX_MIN_SPAN)
    return sorted({(int(r.start_frame), int(r.end_frame)) for r in spans})


def score_corpus(corpus_data, trailing_threshold):
    totals = {"tp": 0, "fp": 0, "fn": 0, "n_algo": 0}
    topo = defaultdict(int)
    start_deltas = []
    span_deltas = []
    for vid, vd in corpus_data.items():
        algos = apply_pipeline(vd["algos_v801"], vd["paw_lk"], vd["norm_pos"], trailing_threshold)
        gts = vd["gts"]
        totals["n_algo"] += len(algos)
        matched, tp_sd, tp_pd = greedy_match(algos, gts)
        totals["tp"] += len(matched)
        totals["fp"] += len(algos) - len(matched)
        totals["fn"] += len(gts) - len(matched)
        start_deltas.extend(tp_sd)
        span_deltas.extend(tp_pd)
        tc = classify_matcher_aware(algos, gts, matched)
        for k, v in tc.items():
            topo[k] += v
    s_abs = int(np.median([abs(d) for d in start_deltas])) if start_deltas else None
    p_abs = int(np.median([abs(d) for d in span_deltas])) if span_deltas else None
    return {
        "totals": totals,
        "topology": dict(topo),
        "start_delta_abs_median": s_abs,
        "span_delta_abs_median": p_abs,
    }


def main():
    print("=" * 70)
    print("TRAILING-TRIM SWEEP (paw_lk threshold, sustain_n=3)")
    print(f"Thresholds: {TRAILING_THRESHOLDS} (plus baseline=0 = no trailing trim)")
    print("=" * 70)
    print()

    cal_data = load_calibration()
    print()
    hol_data = load_holdout()
    print()

    # Baseline: no trailing trim (production v8.0.3)
    print("Computing baseline (no trailing trim)...", flush=True)
    cal_base = score_corpus(cal_data, trailing_threshold=0)
    hol_base = score_corpus(hol_data, trailing_threshold=0)
    print(f"  Cal baseline: TP={cal_base['totals']['tp']} FP={cal_base['totals']['fp']} FN={cal_base['totals']['fn']}")
    print(f"  Hol baseline: TP={hol_base['totals']['tp']} FP={hol_base['totals']['fp']} FN={hol_base['totals']['fn']}")
    print(f"  Cal topology: {cal_base['topology']}")
    print(f"  Hol topology: {hol_base['topology']}")
    print()

    results = {"baseline": {"cal": cal_base, "hol": hol_base}, "configs": {}}
    for thresh in TRAILING_THRESHOLDS:
        print(f"--- trailing_threshold={thresh} ---", flush=True)
        cal_r = score_corpus(cal_data, trailing_threshold=thresh)
        hol_r = score_corpus(hol_data, trailing_threshold=thresh)
        results["configs"][str(thresh)] = {"cal": cal_r, "hol": hol_r}
        cb = cal_base["totals"]; ct = cal_r["totals"]
        hb = hol_base["totals"]; ht = hol_r["totals"]
        print(f"  Cal: TP={ct['tp']} ({ct['tp']-cb['tp']:+}) FP={ct['fp']} ({ct['fp']-cb['fp']:+}) "
              f"FN={ct['fn']} ({ct['fn']-cb['fn']:+})  abs_med start={cal_r['start_delta_abs_median']} span={cal_r['span_delta_abs_median']}")
        print(f"  Hol: TP={ht['tp']} ({ht['tp']-hb['tp']:+}) FP={ht['fp']} ({ht['fp']-hb['fp']:+}) "
              f"FN={ht['fn']} ({ht['fn']-hb['fn']:+})  abs_med start={hol_r['start_delta_abs_median']} span={hol_r['span_delta_abs_median']}")
        # Topology deltas
        for label, base, cur in [("CAL", cal_base, cal_r), ("HOL", hol_base, hol_r)]:
            b_topo = base["topology"]; t_topo = cur["topology"]
            deltas = []
            for k in ("TP","TOLERANCE_ERROR_pairs","MERGED","FRAGMENTED","FALSE_POSITIVE","FALSE_NEGATIVE"):
                d = t_topo.get(k, 0) - b_topo.get(k, 0)
                deltas.append(f"{k}={t_topo.get(k,0)}({d:+})")
            print(f"    {label} topology: {' '.join(deltas)}")
        print()

    # Save
    (OUT_DIR / "metrics" / "sweep_results.json").write_text(json.dumps({
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "thresholds": TRAILING_THRESHOLDS,
        "results": results,
    }, indent=2, default=int), encoding="utf-8")

    # Summary table
    print()
    print("=" * 130)
    print("SUMMARY (legacy matcher counts; topology counts in detail above)")
    print("=" * 130)
    print(f"{'thresh':>7}  {'Cal_TP':>6} {'dTP':>4} {'FP':>4} {'dFP':>4} {'FN':>4} {'dFN':>4}  |  "
          f"{'Hol_TP':>6} {'dTP':>4} {'FP':>4} {'dFP':>4} {'FN':>4} {'dFN':>4}")
    print("-" * 130)
    cb = cal_base["totals"]; hb = hol_base["totals"]
    print(f"{'BASE':>7}  {cb['tp']:>6} {'':>4} {cb['fp']:>4} {'':>4} {cb['fn']:>4} {'':>4}  |  "
          f"{hb['tp']:>6} {'':>4} {hb['fp']:>4} {'':>4} {hb['fn']:>4} {'':>4}")
    for thresh in TRAILING_THRESHOLDS:
        r = results["configs"][str(thresh)]
        ct = r["cal"]["totals"]; ht = r["hol"]["totals"]
        print(f"{thresh:>7.2f}  {ct['tp']:>6} {ct['tp']-cb['tp']:>+4} {ct['fp']:>4} {ct['fp']-cb['fp']:>+4} "
              f"{ct['fn']:>4} {ct['fn']-cb['fn']:>+4}  |  "
              f"{ht['tp']:>6} {ht['tp']-hb['tp']:>+4} {ht['fp']:>4} {ht['fp']-hb['fp']:>+4} "
              f"{ht['fn']:>4} {ht['fn']-hb['fn']:>+4}")
    print()
    print(f"Wrote: {OUT_DIR / 'metrics' / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
