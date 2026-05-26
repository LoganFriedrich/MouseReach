"""
v8.0.3 experiment: per-frame proba threshold sweep on holdout.

Hypothesis: stranded FNs in the v8.0.3 manifests are dominated by SHORT
reaches (median 5f on holdout vs median 13f for TPs). The GBM produces
proba per frame; the 0.5 threshold may be too strict for short bursts.
Lowering threshold might recover short reaches without inflating FP/MERGED.

Sweep on HOLDOUT ONLY (production model on the 19-video generalization
holdout). Calibration LOOCV would need 16-fold model re-training to be
clean; saved for a follow-up if this sweep looks promising.

Thresholds: 0.30, 0.35, 0.40, 0.45, 0.50 (production), 0.55.

Pre-experiment checklist (per pre_experiment_checklist.md):

1. Cumulative-stacking check (verified 2026-05-26):
   - Production v8.0.3 reach detector (model bundle v8.0.0_bsw_w0.8.joblib,
     BSW + mg=0 + trim T=0.60/N=3 + apex-split prom=0.12/depth=0.5/peak2<0.85).
   - Asymmetric matcher -2/+5 on live GT.
   - Matcher-aware topology classifier (no COMPLEX class).
   - Baseline = production v8.0.3 at threshold=0.5 on the same holdout
     videos with today's live GT.
       Hol unfiltered: TP=3680, FP=62, FN=70 (legacy matcher counts)
       Hol topology:  TP=3673, TOL=14, MERGED=3, FRAG=9, FP=34, FN=48
       Hol filtered:  TP=3659, FP=34, FN=46

2. Existing-code-modification check: NO. All sweep logic inline.

3. Unverified hypotheses:
   - That lowering threshold recovers short-reach FNs more than it adds
     phantom FPs.
   - That paw_lk-based leading-trim AND apex-split still work at lower
     thresholds (don't introduce new failure modes).

4. FN-direction-reporting: lead with FN delta vs cumulative best
   (threshold=0.5 same model + live GT). Topology paired with legacy.

5. Framework: output to v8.0.3_dev_threshold_sweep_holdout/.

6. Branch + tag: feature/v8-threshold-sweep, tag pre-threshold-sweep-2026-05-26.

7. Decision rule:
   ACCEPT a threshold if:
     - FN drops materially (>= 5 events) AND
     - FP does not exceed 2x baseline (i.e., <= 124 holdout FP) AND
     - MERGED does not increase (otherwise lowering threshold is just
       smearing the boundary).
   REJECT if FP doubles or MERGED rises.
   Best = lowest FN with FP within bound and MERGED non-increasing.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
    STRICT_START_TOL_EARLY, STRICT_START_TOL_LATE,
)
from mousereach.reach.v8 import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MERGE_GAP, DEFAULT_MIN_SPAN,
    DEFAULT_TRIM_LK_THRESHOLD, DEFAULT_TRIM_SUSTAIN_N,
    DEFAULT_APEX_SPLIT_PROMINENCE, DEFAULT_APEX_SPLIT_DEPTH_MIN,
    DEFAULT_APEX_SPLIT_PEAK2_REL_MAX, DEFAULT_APEX_SPLIT_MIN_DISTANCE,
)
from mousereach.reach.v8.features import extract_features, load_dlc_h5
from mousereach.reach.v8.postprocess import (
    probabilities_to_reaches, trim_leading_sustained_lk,
    apex_split_at_trough, compute_paw_mean_lk,
    compute_hand_to_boxl_norm_pos,
)


HOLDOUT_GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\dlc"
)
GEN_GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\gt"
)
OUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.3_dev_threshold_sweep_holdout"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5


def load_live_gt(video_id):
    gt_path = GEN_GT_DIR / f"{video_id}_unified_ground_truth.json"
    if not gt_path.exists():
        return []
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    rlist = data.get("reaches", {}).get("reaches", [])
    return sorted(set(
        (int(r["start_frame"]), int(r["end_frame"]))
        for r in rlist if not r.get("exclude_from_analysis")
    ))


def overlap_exists(a_s, a_e, g_s, g_e):
    return not (a_e < g_s or a_s > g_e)


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
    tp_sd = []
    for _, ai, gi, sd, _ in candidates:
        if ai in used_a or gi in used_g:
            continue
        used_a.add(ai); used_g.add(gi)
        matched.add((ai, gi))
        tp_sd.append(sd)
    return matched, tp_sd


def classify_components_matcher_aware(algos, gts, matched):
    """Apply the locked matcher-aware topology rules."""
    # Build overlap graph
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
                counts['TOLERANCE_ERROR'] += 1
        elif na == 1 and ng >= 2:
            counts['MERGED'] += 1
        elif na >= 2 and ng == 1:
            counts['FRAGMENTED'] += 1
        elif na >= 2 and ng >= 2:
            # Decompose
            matched_in = [(ai, gj) for (ai, gj) in matched
                          if ai in a_idx and gj in g_idx]
            for _ in matched_in:
                counts['TP'] += 1
            unmatched_a = a_idx - {ai for ai, _ in matched_in}
            unmatched_g = g_idx - {gj for _, gj in matched_in}
            soft_paired = set()
            for ai in sorted(unmatched_a):
                best_gi = None; best_ol = 0
                for gj in sorted(unmatched_g - soft_paired):
                    a_s, a_e = algos[ai]
                    g_s, g_e = gts[gj]
                    s = max(a_s, g_s); e = min(a_e, g_e)
                    ol = max(0, e - s + 1)
                    if ol > best_ol:
                        best_ol = ol; best_gi = gj
                if best_gi is not None and best_ol > 0:
                    counts['TOLERANCE_ERROR'] += 1
                    soft_paired.add(best_gi)
                else:
                    counts['FALSE_POSITIVE'] += 1
            for gj in sorted(unmatched_g - soft_paired):
                counts['FALSE_NEGATIVE'] += 1
    return dict(counts)


def main():
    print("=" * 70)
    print(f"PER-FRAME PROBA THRESHOLD SWEEP -- HOLDOUT 19")
    print(f"Thresholds: {THRESHOLDS}")
    print("=" * 70)
    print()

    print(f"Loading production model from {DEFAULT_MODEL_PATH}...", flush=True)
    bundle = joblib.load(DEFAULT_MODEL_PATH)
    model = bundle["model"]
    feat_cols = bundle["feature_columns"]
    print(f"  Model loaded. Features: {len(feat_cols)}")
    print()

    # First pass: get per-video proba + side data
    print("Computing per-frame proba for each holdout video...", flush=True)
    video_data = {}
    for dlc_path in sorted(HOLDOUT_GT_DIR.glob(f"*{DLC_SUFFIX}.h5")):
        vid = dlc_path.stem.replace(DLC_SUFFIX, "")
        dlc = load_dlc_h5(dlc_path)
        feats = extract_features(dlc)
        X = feats[feat_cols].to_numpy(dtype="float32")
        proba = model.predict_proba(X)[:, 1]
        paw_lk = compute_paw_mean_lk(dlc)
        norm_pos = compute_hand_to_boxl_norm_pos(dlc)
        gts = load_live_gt(vid)
        video_data[vid] = {
            "proba": proba,
            "paw_lk": paw_lk,
            "norm_pos": norm_pos,
            "gts": gts,
        }
        print(f"  {vid}: {len(proba)} frames, {len(gts)} GT reaches", flush=True)
    print()

    # Sweep
    results = {}
    for thresh in THRESHOLDS:
        print(f"--- threshold = {thresh} ---", flush=True)
        totals = {"tp": 0, "fp": 0, "fn": 0, "n_algo_total": 0}
        topo_counts = defaultdict(int)
        tp_start_deltas = []
        per_video = {}
        for vid, vd in video_data.items():
            spans = probabilities_to_reaches(
                vd["proba"], threshold=thresh,
                merge_gap=DEFAULT_MERGE_GAP, min_span=DEFAULT_MIN_SPAN)
            trimmed = trim_leading_sustained_lk(
                spans, vd["paw_lk"],
                threshold=DEFAULT_TRIM_LK_THRESHOLD,
                sustain_n=DEFAULT_TRIM_SUSTAIN_N,
                min_span=DEFAULT_MIN_SPAN)
            split = apex_split_at_trough(
                trimmed, vd["norm_pos"],
                prominence=DEFAULT_APEX_SPLIT_PROMINENCE,
                depth_min=DEFAULT_APEX_SPLIT_DEPTH_MIN,
                peak2_rel_max=DEFAULT_APEX_SPLIT_PEAK2_REL_MAX,
                min_distance=DEFAULT_APEX_SPLIT_MIN_DISTANCE,
                min_span=DEFAULT_MIN_SPAN)
            algos = sorted({(int(r.start_frame), int(r.end_frame)) for r in split})
            totals["n_algo_total"] += len(algos)
            matched, tp_sd = greedy_match(algos, vd["gts"])
            totals["tp"] += len(matched)
            totals["fp"] += len(algos) - len(matched)
            totals["fn"] += len(vd["gts"]) - len(matched)
            tp_start_deltas.extend(tp_sd)
            tc = classify_components_matcher_aware(algos, vd["gts"], matched)
            for k, v in tc.items():
                topo_counts[k] += v
            per_video[vid] = {
                "tp": len(matched), "fp": len(algos) - len(matched),
                "fn": len(vd["gts"]) - len(matched), "n_algo": len(algos),
                "topology": tc,
            }
        abs_med = (int(np.median([abs(d) for d in tp_start_deltas]))
                   if tp_start_deltas else None)
        results[str(thresh)] = {
            "threshold": thresh,
            "totals": totals,
            "topology": dict(topo_counts),
            "start_delta_abs_median": abs_med,
            "per_video": per_video,
        }
        print(f"  Legacy: TP={totals['tp']} FP={totals['fp']} FN={totals['fn']}  "
              f"n_algo={totals['n_algo_total']}  abs_med={abs_med}")
        print(f"  Topology: " + "  ".join(
            f"{k}={dict(topo_counts).get(k,0)}"
            for k in ("TP","TOLERANCE_ERROR","MERGED","FRAGMENTED",
                       "FALSE_POSITIVE","FALSE_NEGATIVE")))
        print()

    # Save
    (OUT_DIR / "metrics" / "sweep_results.json").write_text(json.dumps({
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "thresholds": THRESHOLDS,
        "configs": results,
    }, indent=2, default=int), encoding="utf-8")

    # Summary
    print("=" * 120)
    print("SUMMARY (holdout 19, live GT, v8.0.3 pipeline at varying threshold)")
    print("=" * 120)
    base = results["0.5"]["totals"]
    bb = results["0.5"]["topology"]
    print(f"{'thresh':>7}  {'TP':>5} {'dTP':>5} {'FP':>4} {'dFP':>5} {'FN':>4} {'dFN':>5}  "
          f"{'TP_topo':>7} {'TOL':>4} {'MGD':>4} {'FRG':>4} {'FP_topo':>7} {'FN_topo':>7}  {'abs_med':>7}")
    print("-" * 120)
    for thresh in THRESHOLDS:
        r = results[str(thresh)]
        t = r["totals"]; tc = r["topology"]
        marker = "  <-- baseline" if thresh == 0.5 else ""
        print(f"{thresh:>7.2f}  {t['tp']:>5} {t['tp']-base['tp']:>+5} "
              f"{t['fp']:>4} {t['fp']-base['fp']:>+5} {t['fn']:>4} {t['fn']-base['fn']:>+5}  "
              f"{tc.get('TP',0):>7} {tc.get('TOLERANCE_ERROR',0):>4} {tc.get('MERGED',0):>4} "
              f"{tc.get('FRAGMENTED',0):>4} {tc.get('FALSE_POSITIVE',0):>7} {tc.get('FALSE_NEGATIVE',0):>7}  "
              f"{r['start_delta_abs_median']:>7}{marker}")
    print()
    print(f"Wrote: {OUT_DIR / 'metrics' / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
