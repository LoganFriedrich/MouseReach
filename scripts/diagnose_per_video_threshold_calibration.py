"""Diagnostic: per-video adaptive proba threshold calibration.

Question
--------
Today the v8 detector uses a single per-frame proba threshold (0.5) across
all videos. The recent corpus-wide sweep (REJECT 2026-05-26 morning,
c79da34) showed lowering this threshold rescues only ~17% of stranded FNs
because the rest have proba that never elevates near 0.5.

Per-video threshold could in principle do better than the corpus sweep IF
some videos have systematically shifted proba distributions where their
FN peaks sit at 0.30-0.45 (recoverable) while other videos are fine at
0.5. The corpus sweep had to use one threshold for all videos and got
the worst of both worlds (FP blow-up on easy videos, partial recovery
on hard videos).

This diagnostic answers two things:
1. Do FN-heavy videos have systematically shifted proba distributions?
2. Does any unsupervised per-video calibration mechanism produce
   thresholds that match the "ideal" per-video threshold derived from GT?

Approach
--------
Run production GBM on holdout 19 videos. For each video:
  - Get per-frame proba.
  - Pair algo reaches to GT (live GT, asymmetric matcher).
  - For each TP: compute peak proba inside the GT span.
  - For each filtered FN (no algo overlap): compute peak proba inside
    the GT span.
  - Estimate noise floor as p95 of frames with proba < 0.20 (rest frames).

Three calibration mechanisms, all clamped to [0.30, 0.50]:
  A) Relative-to-TP-peak: threshold = 0.5 * median(TP peak probas in video)
  B) Top-N strong-peak median: threshold = 0.5 * median(top10 TP peaks)
  C) Noise-floor-based: threshold = max(0.30, noise_floor * 3)

For each mechanism: apply per-video threshold, re-run probabilities_to_reaches
+ trim + apex-split (production v8.0.4 stack), compare FN/FP to
baseline (threshold=0.5).

NOT a ship experiment. NOT a sweep with decision rule. Just a diagnostic
to characterize whether per-video calibration is viable.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
    STRICT_START_TOL_EARLY, STRICT_START_TOL_LATE,
)
from mousereach.reach.v8 import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MERGE_GAP, DEFAULT_MIN_SPAN,
    DEFAULT_TRIM_LK_THRESHOLD, DEFAULT_TRIM_SUSTAIN_N,
    DEFAULT_TRAILING_TRIM_LK_THRESHOLD, DEFAULT_TRAILING_TRIM_SUSTAIN_N,
    DEFAULT_APEX_SPLIT_PROMINENCE, DEFAULT_APEX_SPLIT_DEPTH_MIN,
    DEFAULT_APEX_SPLIT_PEAK2_REL_MAX, DEFAULT_APEX_SPLIT_MIN_DISTANCE,
)
from mousereach.reach.v8.features import extract_features, load_dlc_h5
from mousereach.reach.v8.postprocess import (
    probabilities_to_reaches, trim_leading_sustained_lk,
    trim_trailing_sustained_lk,
    apex_split_at_trough, compute_paw_mean_lk,
    compute_hand_to_boxl_norm_pos,
)


HOLDOUT_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\dlc"
)
GEN_GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\gt"
)
OUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.4_dev_per_video_threshold_diagnostic"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"

# Production v8.0.4 postprocess (frozen)
SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5

# Calibration clamps
THRESHOLD_FLOOR = 0.30
THRESHOLD_CEILING = 0.50  # never go above 0.50 (production)


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
                candidates.append((abs(sd), ai, gi))
    candidates.sort()
    matched = set()
    used_a, used_g = set(), set()
    for _, ai, gi in candidates:
        if ai in used_a or gi in used_g:
            continue
        used_a.add(ai); used_g.add(gi)
        matched.add((ai, gi))
    return matched


def apply_pipeline(proba, paw_lk, norm_pos, threshold):
    """Production v8.0.4 postprocess with a configurable per-frame threshold."""
    spans = probabilities_to_reaches(
        proba, threshold=threshold, merge_gap=DEFAULT_MERGE_GAP,
        min_span=DEFAULT_MIN_SPAN)
    spans = trim_leading_sustained_lk(
        spans, paw_lk,
        threshold=DEFAULT_TRIM_LK_THRESHOLD,
        sustain_n=DEFAULT_TRIM_SUSTAIN_N,
        min_span=DEFAULT_MIN_SPAN)
    spans = trim_trailing_sustained_lk(
        spans, paw_lk,
        threshold=DEFAULT_TRAILING_TRIM_LK_THRESHOLD,
        sustain_n=DEFAULT_TRAILING_TRIM_SUSTAIN_N,
        min_span=DEFAULT_MIN_SPAN)
    spans = apex_split_at_trough(
        spans, norm_pos,
        prominence=DEFAULT_APEX_SPLIT_PROMINENCE,
        depth_min=DEFAULT_APEX_SPLIT_DEPTH_MIN,
        peak2_rel_max=DEFAULT_APEX_SPLIT_PEAK2_REL_MAX,
        min_distance=DEFAULT_APEX_SPLIT_MIN_DISTANCE,
        min_span=DEFAULT_MIN_SPAN)
    return sorted({(int(r.start_frame), int(r.end_frame)) for r in spans})


def peak_in_span(proba, start, end):
    end_use = min(end, len(proba) - 1)
    if end_use < start:
        return None
    return float(np.max(proba[start:end_use + 1]))


def main():
    print("=" * 70)
    print("PER-VIDEO THRESHOLD CALIBRATION DIAGNOSTIC (HOLDOUT 19)")
    print("=" * 70)
    print()

    print(f"Loading production model from {DEFAULT_MODEL_PATH}...", flush=True)
    bundle = joblib.load(DEFAULT_MODEL_PATH)
    model = bundle["model"]
    feat_cols = bundle["feature_columns"]

    print("Computing per-frame proba + GT pairing for each holdout video...",
          flush=True)
    video_data = {}
    for dlc_path in sorted(HOLDOUT_DLC_DIR.glob(f"*{DLC_SUFFIX}.h5")):
        vid = dlc_path.stem.replace(DLC_SUFFIX, "")
        dlc = load_dlc_h5(dlc_path)
        feats = extract_features(dlc)
        X = feats[feat_cols].to_numpy(dtype="float32")
        proba = model.predict_proba(X)[:, 1]
        paw_lk = compute_paw_mean_lk(dlc)
        norm_pos = compute_hand_to_boxl_norm_pos(dlc)
        gts = load_live_gt(vid)
        # Baseline pipeline at threshold=0.5
        algos_base = apply_pipeline(proba, paw_lk, norm_pos, 0.5)
        matched_base = greedy_match(algos_base, gts)
        # TP peak probas (per GT, peak inside GT span when matched)
        tp_peaks = []
        for ai, gi in matched_base:
            g_s, g_e = gts[gi]
            p = peak_in_span(proba, g_s, g_e)
            if p is not None:
                tp_peaks.append(p)
        # FN categorization: STRANDED (no algo overlap with this GT) vs
        # TOLERANCE (algo overlap exists but matcher rejected).
        # Only STRANDED FNs are candidates for threshold lowering --
        # tolerance FNs already have an emitted algo, lowering threshold
        # would just shift boundaries further off.
        matched_gt_idx = {gi for _, gi in matched_base}
        stranded_fn_peaks = []
        tolerance_fn_peaks = []
        for gi, (g_s, g_e) in enumerate(gts):
            if gi in matched_gt_idx:
                continue
            # Check if any algo span overlaps this GT
            has_overlap = any(overlap_exists(a_s, a_e, g_s, g_e)
                              for a_s, a_e in algos_base)
            p = peak_in_span(proba, g_s, g_e)
            if p is None:
                continue
            if has_overlap:
                tolerance_fn_peaks.append(p)
            else:
                stranded_fn_peaks.append(p)
        # All FN peaks union for back-compat reporting
        fn_peaks = stranded_fn_peaks + tolerance_fn_peaks
        # Noise floor: p95 of frames with proba < 0.2
        rest_frames = proba[proba < 0.20]
        noise_floor = float(np.percentile(rest_frames, 95)) if rest_frames.size else 0.0
        video_data[vid] = {
            "proba": proba, "paw_lk": paw_lk, "norm_pos": norm_pos,
            "gts": gts,
            "algos_base": algos_base, "matched_base": matched_base,
            "tp_peaks": tp_peaks,
            "fn_peaks": fn_peaks,
            "stranded_fn_peaks": stranded_fn_peaks,
            "tolerance_fn_peaks": tolerance_fn_peaks,
            "noise_floor": noise_floor,
        }
        print(f"  {vid}: {len(proba)} frames  GTs={len(gts)}  "
              f"baseline TP={len(matched_base)} FP={len(algos_base)-len(matched_base)} "
              f"FN={len(gts)-len(matched_base)}  noise_floor={noise_floor:.3f}",
              flush=True)
    print()

    # =====================================================================
    # Per-video proba distribution summary
    # =====================================================================
    print("=" * 70)
    print("PER-VIDEO PROBA DISTRIBUTIONS")
    print("=" * 70)
    print(f"{'video':<22} {'noise':>6} {'TP_p50':>7} {'TP_top10':>9} "
          f"{'tFN_n':>5} {'sFN_n':>5} {'sFN_p50':>8} {'sFN_max':>8} "
          f"{'sFN>=.45':>9} {'sFN>=.40':>9} {'sFN>=.30':>9} {'sFN<.30':>8}")
    print("-" * 130)
    for vid, vd in video_data.items():
        tp_p50 = float(np.median(vd["tp_peaks"])) if vd["tp_peaks"] else float("nan")
        top_n = sorted(vd["tp_peaks"], reverse=True)[:10]
        tp_top10 = float(np.median(top_n)) if top_n else float("nan")
        tfn_n = len(vd["tolerance_fn_peaks"])
        sfn = vd["stranded_fn_peaks"]
        sfn_n = len(sfn)
        sfn_p50 = float(np.median(sfn)) if sfn else float("nan")
        sfn_max = float(np.max(sfn)) if sfn else float("nan")
        sfn_45 = sum(1 for p in sfn if p >= 0.45)
        sfn_40 = sum(1 for p in sfn if p >= 0.40)
        sfn_30 = sum(1 for p in sfn if p >= 0.30)
        sfn_lt30 = sum(1 for p in sfn if p < 0.30)
        print(f"{vid:<22} {vd['noise_floor']:>6.3f} {tp_p50:>7.3f} {tp_top10:>9.3f} "
              f"{tfn_n:>5} {sfn_n:>5} {sfn_p50:>8.3f} {sfn_max:>8.3f} "
              f"{sfn_45:>9} {sfn_40:>9} {sfn_30:>9} {sfn_lt30:>8}")
    print()
    # Aggregate STRANDED-FN peak distribution (the relevant population
    # for threshold lowering -- tolerance FNs aren't threshold-bound).
    all_stranded = [p for vd in video_data.values() for p in vd["stranded_fn_peaks"]]
    all_tolerance = [p for vd in video_data.values() for p in vd["tolerance_fn_peaks"]]
    print(f"Corpus-wide FN peak distribution:")
    print(f"  Stranded FNs (no algo overlap, candidates for threshold lower): N={len(all_stranded)}")
    print(f"  Tolerance FNs (algo overlap but matcher rejected):              N={len(all_tolerance)}")
    print()
    if all_stranded:
        print("  STRANDED FN peak proba percentiles:")
        for q in (0, 10, 25, 50, 75, 90, 100):
            print(f"    p{q:>3}: {np.percentile(all_stranded, q):.3f}")
        print("  STRANDED FN proba threshold tally:")
        for t in (0.20, 0.30, 0.35, 0.40, 0.45):
            n_ge = sum(1 for p in all_stranded if p >= t)
            print(f"    FN peaks >= {t}: {n_ge}/{len(all_stranded)} "
                  f"({100.0*n_ge/len(all_stranded):.1f}%)")
    print()

    # =====================================================================
    # Compute per-video thresholds for each mechanism
    # =====================================================================
    def clamp(t):
        return max(THRESHOLD_FLOOR, min(THRESHOLD_CEILING, t))

    mech_thresholds = {"A": {}, "B": {}, "C": {}}
    for vid, vd in video_data.items():
        # A: relative-to-TP-median
        if vd["tp_peaks"]:
            t_a = 0.5 * np.median(vd["tp_peaks"])
        else:
            t_a = 0.50
        mech_thresholds["A"][vid] = clamp(t_a)
        # B: relative-to-top10-TP-median
        top_n = sorted(vd["tp_peaks"], reverse=True)[:10]
        if top_n:
            t_b = 0.5 * np.median(top_n)
        else:
            t_b = 0.50
        mech_thresholds["B"][vid] = clamp(t_b)
        # C: noise-floor based
        t_c = vd["noise_floor"] * 3.0
        mech_thresholds["C"][vid] = clamp(t_c)

    print("=" * 70)
    print("PER-VIDEO THRESHOLDS BY MECHANISM")
    print("=" * 70)
    print(f"{'video':<22} {'A_TPp50':>9} {'B_top10':>9} {'C_noise':>9}")
    print("-" * 60)
    for vid in video_data.keys():
        print(f"{vid:<22} {mech_thresholds['A'][vid]:>9.3f} "
              f"{mech_thresholds['B'][vid]:>9.3f} {mech_thresholds['C'][vid]:>9.3f}")
    print()

    # =====================================================================
    # Apply each mechanism and compare to baseline
    # =====================================================================
    print("=" * 70)
    print("MECHANISM RESULTS (vs threshold=0.5 baseline)")
    print("=" * 70)
    base_tp = sum(len(vd["matched_base"]) for vd in video_data.values())
    base_fp = sum(len(vd["algos_base"]) - len(vd["matched_base"])
                  for vd in video_data.values())
    base_fn = sum(len(vd["gts"]) - len(vd["matched_base"])
                  for vd in video_data.values())
    print(f"Baseline (threshold=0.5): TP={base_tp} FP={base_fp} FN={base_fn}")
    print()

    mech_results = {}
    for mech in ("A", "B", "C"):
        m_tp = m_fp = m_fn = 0
        per_video_deltas = {}
        for vid, vd in video_data.items():
            thresh = mech_thresholds[mech][vid]
            algos = apply_pipeline(vd["proba"], vd["paw_lk"], vd["norm_pos"], thresh)
            matched = greedy_match(algos, vd["gts"])
            tp = len(matched)
            fp = len(algos) - tp
            fn = len(vd["gts"]) - tp
            m_tp += tp; m_fp += fp; m_fn += fn
            base_tp_v = len(vd["matched_base"])
            base_fp_v = len(vd["algos_base"]) - base_tp_v
            base_fn_v = len(vd["gts"]) - base_tp_v
            per_video_deltas[vid] = {
                "threshold": float(thresh),
                "tp": tp, "fp": fp, "fn": fn,
                "dtp": tp - base_tp_v,
                "dfp": fp - base_fp_v,
                "dfn": fn - base_fn_v,
            }
        mech_results[mech] = {
            "totals": {"tp": m_tp, "fp": m_fp, "fn": m_fn},
            "deltas": {"tp": m_tp - base_tp, "fp": m_fp - base_fp,
                        "fn": m_fn - base_fn},
            "per_video": per_video_deltas,
        }
        d = mech_results[mech]["deltas"]
        print(f"Mechanism {mech}: TP={m_tp} ({d['tp']:+d})  "
              f"FP={m_fp} ({d['fp']:+d})  FN={m_fn} ({d['fn']:+d})")
        # Per-video that moved
        moved = [(vid, dd) for vid, dd in per_video_deltas.items()
                 if dd["dtp"] != 0 or dd["dfp"] != 0 or dd["dfn"] != 0]
        if moved:
            print(f"  Videos that moved ({len(moved)}):")
            for vid, dd in moved:
                print(f"    {vid:<22} thresh={dd['threshold']:.3f}  "
                      f"dTP={dd['dtp']:+}  dFP={dd['dfp']:+}  dFN={dd['dfn']:+}")
        print()

    # =====================================================================
    # Save
    # =====================================================================
    out = {
        "baseline": {"tp": base_tp, "fp": base_fp, "fn": base_fn},
        "mechanisms": mech_results,
        "per_video_thresholds": {
            m: {v: float(t) for v, t in d.items()}
            for m, d in mech_thresholds.items()
        },
        "per_video_summary": {
            vid: {
                "noise_floor": vd["noise_floor"],
                "tp_peaks": vd["tp_peaks"],
                "stranded_fn_peaks": vd["stranded_fn_peaks"],
                "tolerance_fn_peaks": vd["tolerance_fn_peaks"],
                "baseline_tp": len(vd["matched_base"]),
                "baseline_fp": len(vd["algos_base"]) - len(vd["matched_base"]),
                "baseline_fn": len(vd["gts"]) - len(vd["matched_base"]),
            } for vid, vd in video_data.items()
        },
    }
    (OUT_DIR / "metrics" / "diagnostic.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"Wrote: {OUT_DIR / 'metrics' / 'diagnostic.json'}")


if __name__ == "__main__":
    main()
