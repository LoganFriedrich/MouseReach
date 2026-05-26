"""Per-video threshold via Otsu's method on the proba histogram.

Extends the per-video threshold diagnostic with two more mechanisms:

  D) Otsu (clamped): per-video threshold = threshold_otsu(proba),
     clamped to [0.30, 0.50].
  E) Otsu (raw): unclamped, just to see what Otsu suggests on its own.

Otsu's method finds the threshold that maximizes between-class variance
in a bimodal distribution. For per-frame proba it should find the valley
between the "rest" mode (near 0) and the "reach" mode (near 1).

Why this might give per-video variation that A/B/C did not:
- Mechanisms A/B failed because TP peaks saturate at 1.0 uniformly.
- Mechanism C failed because noise floors are uniformly tiny (0.000-0.035).
- Otsu uses the FULL histogram, including the intermediate band (proba
  in [0.20, 0.80]). The size of that intermediate band can vary per
  video -- videos with tight bimodal distributions push Otsu high, videos
  with messier intermediate regions push Otsu low.

NOT a ship experiment. Diagnostic only.
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
    r"\reach_detection\v8.0.4_dev_per_video_threshold_otsu"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"

SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5

THRESHOLD_FLOOR = 0.30
THRESHOLD_CEILING = 0.50


def threshold_otsu_1d(values: np.ndarray, n_bins: int = 256) -> float:
    """Otsu's method for a 1-D array of floats in [0, 1]."""
    hist, bin_edges = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    if hist.sum() == 0:
        return 0.5
    # Cumulative sums
    w = hist.cumsum()
    total = w[-1]
    # Bin midpoints
    midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mu = (hist * midpoints).cumsum()
    mu_total = mu[-1]
    # Between-class variance
    # Avoid div-by-zero: skip bins where w == 0 or w == total
    valid = (w > 0) & (w < total)
    if not np.any(valid):
        return 0.5
    sigma_b_sq = np.zeros_like(w)
    w_safe = np.where(valid, w, 1)
    sigma_b_sq[valid] = ((mu_total * w[valid] - mu[valid]) ** 2) / (
        w_safe[valid] * (total - w_safe[valid]))
    idx = int(np.argmax(sigma_b_sq))
    return float(midpoints[idx])


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


def main():
    print("=" * 70)
    print("PER-VIDEO OTSU THRESHOLD DIAGNOSTIC (HOLDOUT 19)")
    print("=" * 70)
    print()

    print(f"Loading production model...", flush=True)
    bundle = joblib.load(DEFAULT_MODEL_PATH)
    model = bundle["model"]
    feat_cols = bundle["feature_columns"]

    print("Computing per-frame proba for each holdout video...", flush=True)
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
        algos_base = apply_pipeline(proba, paw_lk, norm_pos, 0.5)
        matched_base = greedy_match(algos_base, gts)
        # Compute Otsu raw threshold from the full proba distribution
        otsu_raw = threshold_otsu_1d(proba)
        # Also compute on log-spaced bins emphasizing the intermediate band
        # Use a finer histogram restricted to the [0.05, 0.95] range, which
        # excludes the dominant zero-spike that can pull Otsu artificially low
        intermediate = proba[(proba >= 0.05) & (proba <= 0.95)]
        if intermediate.size >= 20:
            otsu_int = threshold_otsu_1d(intermediate)
        else:
            otsu_int = otsu_raw
        video_data[vid] = {
            "proba": proba, "paw_lk": paw_lk, "norm_pos": norm_pos,
            "gts": gts,
            "algos_base": algos_base, "matched_base": matched_base,
            "otsu_raw": float(otsu_raw),
            "otsu_intermediate": float(otsu_int),
        }
        print(f"  {vid}: otsu_raw={otsu_raw:.3f} otsu_int={otsu_int:.3f} "
              f"baseline TP={len(matched_base)} FN={len(gts)-len(matched_base)}",
              flush=True)
    print()

    # Print per-video thresholds for each mechanism
    print("=" * 70)
    print("PER-VIDEO OTSU THRESHOLDS")
    print("=" * 70)
    print(f"{'video':<22} {'otsu_raw':>10} {'D_clamped':>12} {'otsu_int':>10} {'F_int_clamped':>15}")
    print("-" * 80)
    def clamp(t):
        return max(THRESHOLD_FLOOR, min(THRESHOLD_CEILING, t))
    for vid, vd in video_data.items():
        d_clamped = clamp(vd["otsu_raw"])
        f_clamped = clamp(vd["otsu_intermediate"])
        print(f"{vid:<22} {vd['otsu_raw']:>10.3f} {d_clamped:>12.3f} "
              f"{vd['otsu_intermediate']:>10.3f} {f_clamped:>15.3f}")
    print()

    # Apply each mechanism
    print("=" * 70)
    print("MECHANISM RESULTS")
    print("=" * 70)
    base_tp = sum(len(vd["matched_base"]) for vd in video_data.values())
    base_fp = sum(len(vd["algos_base"]) - len(vd["matched_base"])
                  for vd in video_data.values())
    base_fn = sum(len(vd["gts"]) - len(vd["matched_base"])
                  for vd in video_data.values())
    print(f"Baseline (threshold=0.5):  TP={base_tp}  FP={base_fp}  FN={base_fn}")
    print()

    def evaluate(mech_label, threshold_fn):
        m_tp = m_fp = m_fn = 0
        per_video = {}
        for vid, vd in video_data.items():
            thresh = threshold_fn(vd)
            algos = apply_pipeline(vd["proba"], vd["paw_lk"], vd["norm_pos"], thresh)
            matched = greedy_match(algos, vd["gts"])
            tp = len(matched)
            fp = len(algos) - tp
            fn = len(vd["gts"]) - tp
            m_tp += tp; m_fp += fp; m_fn += fn
            btp = len(vd["matched_base"])
            bfp = len(vd["algos_base"]) - btp
            bfn = len(vd["gts"]) - btp
            per_video[vid] = {
                "threshold": float(thresh),
                "tp": tp, "fp": fp, "fn": fn,
                "dtp": tp - btp, "dfp": fp - bfp, "dfn": fn - bfn,
            }
        print(f"Mechanism {mech_label}: TP={m_tp} ({m_tp-base_tp:+d})  "
              f"FP={m_fp} ({m_fp-base_fp:+d})  FN={m_fn} ({m_fn-base_fn:+d})")
        moved = [(vid, dd) for vid, dd in per_video.items()
                 if dd["dtp"] != 0 or dd["dfp"] != 0 or dd["dfn"] != 0]
        if moved:
            print(f"  Videos that moved ({len(moved)}):")
            for vid, dd in moved:
                print(f"    {vid:<22} thresh={dd['threshold']:.3f}  "
                      f"dTP={dd['dtp']:+}  dFP={dd['dfp']:+}  dFN={dd['dfn']:+}")
        print()
        return {"totals": {"tp": m_tp, "fp": m_fp, "fn": m_fn},
                "per_video": per_video}

    results = {
        "D_otsu_clamped":  evaluate("D (otsu clamped [.30,.50])",
                                     lambda vd: clamp(vd["otsu_raw"])),
        "E_otsu_raw":      evaluate("E (otsu raw, no ceiling)",
                                     lambda vd: vd["otsu_raw"]),
        "F_otsu_int":      evaluate("F (otsu on intermediate band, clamped)",
                                     lambda vd: clamp(vd["otsu_intermediate"])),
    }

    out = {
        "baseline": {"tp": base_tp, "fp": base_fp, "fn": base_fn},
        "mechanisms": results,
        "per_video_otsu": {
            vid: {"otsu_raw": vd["otsu_raw"],
                  "otsu_intermediate": vd["otsu_intermediate"]}
            for vid, vd in video_data.items()
        },
    }
    (OUT_DIR / "metrics" / "diagnostic.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"Wrote: {OUT_DIR / 'metrics' / 'diagnostic.json'}")


if __name__ == "__main__":
    main()
