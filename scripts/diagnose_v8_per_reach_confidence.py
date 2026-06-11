"""
Diagnostic: per-reach confidence distribution on v8.0.0 (generalization corpus).

For each detected reach on the 19-video generalization corpus, compute
confidence summary statistics from the per-frame GBM probability series:
mean, median, peak, min, std. Tag each reach as TP or FP under strict
and permissive matching rules using existing GT files. Output per-reach
records + per-video shape summaries + overall histograms broken by
TP/FP status under each criterion.

The question this answers: are per-reach confidences bimodal in a way
that would justify a confidence-based postfilter (real reaches cluster
at high mean_proba; FPs cluster at low)? If yes, a per-video Otsu-style
threshold could separate them without GT contact. If no, this direction
doesn't help.

================================================================
INPUTS
================================================================

  DLC h5 files:
    iterations/generalization_test_2026-05-11/algo_outputs_current/
      <video_id>DLC_resnet50_MPSAOct27shuffle1_100000.h5

  GT files:
    iterations/generalization_test_2026-05-11/gt/
      <video_id>_unified_ground_truth.json

  v8.0.0 production model (bundled in package):
    src/mousereach/reach/v8/models/v8.0.0_bsw_w0.8.joblib

================================================================
OUTPUTS
================================================================

  Improvement_Snapshots/reach_detection/v8.0.0_dev_per_reach_confidence_distribution/
    metrics/
      per_reach_records.json   # every detected reach with confidence stats + status
      summary.json             # aggregates + per-video shape metrics + Otsu thresholds

================================================================
WHAT THIS IS NOT
================================================================

Not an experiment. No model changes, no algorithm modifications. Runs
production v8.0.0 inference on existing videos with the existing model.
Read-only on all inputs. Output is pure analysis.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.core.geometry import load_dlc
from mousereach.reach.v8 import DEFAULT_MODEL_PATH
from mousereach.reach.v8.features import extract_features
from mousereach.reach.v8.postprocess import probabilities_to_reaches
from mousereach.improvement.reach_detection.metrics import (
    Reach, match_reaches,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ITER_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11"
)
DLC_DIR = ITER_DIR / "algo_outputs_current"
GT_DIR = ITER_DIR / "gt"

OUTPUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_per_reach_confidence_distribution"
)


# Matching / postprocess params
THRESHOLD = 0.5
MERGE_GAP = 2
MIN_SPAN = 3
PERMISSIVE_WINDOW = 10
STRICT_START_TOL = 2
STRICT_SPAN_TOL_REL = 0.5
STRICT_SPAN_TOL_ABS = 5


# ---------------------------------------------------------------------------
# Otsu's method (inline; classical statistical thresholding)
# ---------------------------------------------------------------------------

def otsu_threshold(values: List[float], n_bins: int = 50
                   ) -> Optional[float]:
    """Find the threshold that maximizes between-class variance.

    Returns the threshold value (in the same units as `values`) where
    the histogram splits into two classes whose between-class variance
    is maximized. Returns None if there isn't enough data.
    """
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)])
    if len(arr) < 4:
        return None
    counts, edges = np.histogram(arr, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    total = counts.sum()
    if total == 0:
        return None
    cumsum = counts.cumsum().astype(np.float64)
    cum_means = (counts * centers).cumsum() / np.maximum(cumsum, 1)
    grand_mean = (counts * centers).sum() / total
    omega = cumsum / total
    omega_inv = 1.0 - omega
    valid = (omega > 0) & (omega_inv > 0)
    if not valid.any():
        return None
    var_between = omega * omega_inv * (cum_means - grand_mean) ** 2
    var_between = var_between * valid
    best_idx = int(np.argmax(var_between))
    return float(centers[best_idx])


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_gt_with_apex(video_id: str) -> Optional[List[Dict[str, Any]]]:
    unified = GT_DIR / f"{video_id}_unified_ground_truth.json"
    if not unified.exists():
        return None
    data = json.loads(unified.read_text(encoding="utf-8"))
    rd = data.get("reaches", {})
    if not isinstance(rd, dict) or not rd.get("exhaustive"):
        return None
    raw = rd.get("reaches", [])
    filt = [r for r in raw if not r.get("exclude_from_analysis", False)]
    filt.sort(key=lambda r: r["start_frame"])
    return filt


def _find_dlc(video_id: str) -> Optional[Path]:
    candidates = list(DLC_DIR.glob(f"{video_id}DLC_*.h5"))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Per-reach confidence extraction + matching
# ---------------------------------------------------------------------------

def _reach_confidence_stats(proba: np.ndarray, start: int, end: int
                            ) -> Dict[str, float]:
    """Confidence summary stats for one reach window [start, end] inclusive."""
    if end < start or start < 0 or end >= len(proba):
        return {"mean": None, "median": None, "peak": None,
                "min": None, "std": None}
    window = proba[start:end + 1]
    valid = window[~np.isnan(window)]
    if len(valid) == 0:
        return {"mean": None, "median": None, "peak": None,
                "min": None, "std": None}
    return {
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "peak": float(np.max(valid)),
        "min": float(np.min(valid)),
        "std": float(np.std(valid)),
    }


def _strict_match_status(algo_start: int, algo_end: int,
                         gt_objs: List[Reach]) -> str:
    """Return 'TP' if this algo reach matches some GT under strict; else 'FP'.
    Uses a single greedy match (the closest GT by start within strict_tol
    that also passes span check)."""
    best_d = None
    best_g = None
    a_span = algo_end - algo_start + 1
    for g in gt_objs:
        d = abs(algo_start - g.start_frame)
        if d > STRICT_START_TOL:
            continue
        g_span = g.end_frame - g.start_frame + 1
        tol = max(STRICT_SPAN_TOL_REL * g_span, STRICT_SPAN_TOL_ABS)
        if abs(a_span - g_span) > tol:
            continue
        if best_d is None or d < best_d:
            best_d = d
            best_g = g
    return "TP" if best_g is not None else "FP"


def _permissive_match_status(algo_start: int, gt_objs: List[Reach]) -> str:
    """TP iff some GT has |gt_start - algo_start| <= PERMISSIVE_WINDOW."""
    for g in gt_objs:
        if abs(algo_start - g.start_frame) <= PERMISSIVE_WINDOW:
            return "TP"
    return "FP"


def process_video(video_id: str, model: Any, feat_cols: List[str]
                  ) -> List[Dict[str, Any]]:
    """Run v8.0.0 inference, compute per-reach confidence + match status."""
    dlc_path = _find_dlc(video_id)
    if dlc_path is None:
        return []
    gt_dicts = _load_gt_with_apex(video_id)
    if gt_dicts is None:
        return []

    dlc = load_dlc(dlc_path)
    feats = extract_features(dlc)
    X = feats[feat_cols].to_numpy(dtype=np.float32)
    proba = model.predict_proba(X)[:, 1]

    raw_reaches = probabilities_to_reaches(
        proba, threshold=THRESHOLD, merge_gap=MERGE_GAP, min_span=MIN_SPAN)

    gt_objs = [Reach(start_frame=int(r["start_frame"]),
                     end_frame=int(r["end_frame"]),
                     index=i)
               for i, r in enumerate(gt_dicts)]

    records = []
    for i, r in enumerate(raw_reaches):
        stats = _reach_confidence_stats(proba, r.start_frame, r.end_frame)
        status_strict = _strict_match_status(r.start_frame, r.end_frame, gt_objs)
        status_permissive = _permissive_match_status(r.start_frame, gt_objs)
        records.append({
            "video_id": video_id,
            "reach_idx": i,
            "start_frame": int(r.start_frame),
            "end_frame": int(r.end_frame),
            "duration_frames": int(r.end_frame - r.start_frame + 1),
            "status_strict": status_strict,
            "status_permissive": status_permissive,
            "confidence": stats,
        })
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

CONF_BUCKETS = [
    (0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75),
    (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.001),
]


def _bucket_label(lo: float, hi: float) -> str:
    return f"[{lo:.2f},{hi:.2f})"


def histogram_by_status(records: List[Dict[str, Any]],
                        stat_key: str,
                        status_field: str) -> Dict[str, Dict[str, int]]:
    """Bucketed histogram of records by confidence stat, broken by status."""
    out: Dict[str, Dict[str, int]] = {
        _bucket_label(lo, hi): {"TP": 0, "FP": 0}
        for lo, hi in CONF_BUCKETS
    }
    for r in records:
        v = r["confidence"].get(stat_key)
        if v is None:
            continue
        for lo, hi in CONF_BUCKETS:
            if lo <= v < hi:
                out[_bucket_label(lo, hi)][r[status_field]] += 1
                break
    return out


def per_video_shape_summary(records: List[Dict[str, Any]]
                            ) -> Dict[str, Any]:
    """Per-video distribution shape: TP/FP counts + confidence percentiles +
    Otsu thresholds (overall and split-by-status)."""
    by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_video[r["video_id"]].append(r)

    out = {}
    for vid, recs in by_video.items():
        n = len(recs)
        n_tp_s = sum(1 for r in recs if r["status_strict"] == "TP")
        n_fp_s = n - n_tp_s
        n_tp_p = sum(1 for r in recs if r["status_permissive"] == "TP")
        n_fp_p = n - n_tp_p

        all_mean = [r["confidence"]["mean"] for r in recs
                    if r["confidence"]["mean"] is not None]
        tp_strict_mean = [r["confidence"]["mean"] for r in recs
                          if r["status_strict"] == "TP" and r["confidence"]["mean"] is not None]
        fp_strict_mean = [r["confidence"]["mean"] for r in recs
                          if r["status_strict"] == "FP" and r["confidence"]["mean"] is not None]

        out[vid] = {
            "n_reaches": n,
            "n_tp_strict": n_tp_s,
            "n_fp_strict": n_fp_s,
            "n_tp_permissive": n_tp_p,
            "n_fp_permissive": n_fp_p,
            "mean_proba_all_p25": float(np.percentile(all_mean, 25)) if all_mean else None,
            "mean_proba_all_p50": float(np.percentile(all_mean, 50)) if all_mean else None,
            "mean_proba_all_p75": float(np.percentile(all_mean, 75)) if all_mean else None,
            "mean_proba_tp_strict_median": float(np.median(tp_strict_mean)) if tp_strict_mean else None,
            "mean_proba_fp_strict_median": float(np.median(fp_strict_mean)) if fp_strict_mean else None,
            "otsu_threshold_all": otsu_threshold(all_mean),
            "otsu_threshold_tp_vs_fp_actual": otsu_threshold(all_mean) if (tp_strict_mean and fp_strict_mean) else None,
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("PER-REACH CONFIDENCE DISTRIBUTION -- v8.0.0 on 20-video generalization corpus")
    print("=" * 78)
    print()

    print(f"Loading v8.0.0 production model: {DEFAULT_MODEL_PATH}")
    bundle = joblib.load(DEFAULT_MODEL_PATH)
    model = bundle["model"]
    feat_cols = bundle["feature_columns"]
    print(f"  {len(feat_cols)} features")
    print()

    # Discover videos
    algo_files = sorted(DLC_DIR.glob("*DLC_*.h5"))
    video_ids = []
    for f in algo_files:
        name = f.stem
        idx = name.find("DLC_")
        if idx > 0:
            video_ids.append(name[:idx])
    print(f"Candidate videos: {len(video_ids)}")
    print()

    all_records = []
    for vid in video_ids:
        recs = process_video(vid, model, feat_cols)
        if recs:
            n_tp_s = sum(1 for r in recs if r["status_strict"] == "TP")
            n_fp_s = sum(1 for r in recs if r["status_strict"] == "FP")
            n_tp_p = sum(1 for r in recs if r["status_permissive"] == "TP")
            n_fp_p = sum(1 for r in recs if r["status_permissive"] == "FP")
            print(f"  {vid:35} reaches={len(recs):4}  "
                  f"strict TP/FP={n_tp_s:>4}/{n_fp_s:>3}  "
                  f"perm TP/FP={n_tp_p:>4}/{n_fp_p:>3}")
            all_records.extend(recs)
        else:
            print(f"  {vid:35} skipped (no DLC or no exhaustive GT)")
    print(f"\nTotal detected reaches: {len(all_records)}")
    print()

    # Headline histograms
    print("=" * 78)
    print("MEAN_PROBA HISTOGRAM (all reaches), broken by STRICT match status")
    print("=" * 78)
    h_strict = histogram_by_status(all_records, "mean", "status_strict")
    print(f"  {'bucket':<16} {'TP':>6} {'FP':>6} {'total':>6}   TP/(TP+FP)")
    for label, counts in h_strict.items():
        tp = counts["TP"]; fp = counts["FP"]
        total = tp + fp
        ratio = (tp / total) if total else 0.0
        print(f"  {label:<16} {tp:>6} {fp:>6} {total:>6}   {ratio:.3f}")
    print()

    print("=" * 78)
    print("MEAN_PROBA HISTOGRAM, broken by PERMISSIVE match status")
    print("=" * 78)
    h_perm = histogram_by_status(all_records, "mean", "status_permissive")
    print(f"  {'bucket':<16} {'TP':>6} {'FP':>6} {'total':>6}   TP/(TP+FP)")
    for label, counts in h_perm.items():
        tp = counts["TP"]; fp = counts["FP"]
        total = tp + fp
        ratio = (tp / total) if total else 0.0
        print(f"  {label:<16} {tp:>6} {fp:>6} {total:>6}   {ratio:.3f}")
    print()

    # Peak proba histogram (an alternative shape)
    print("=" * 78)
    print("PEAK_PROBA HISTOGRAM, broken by STRICT match status")
    print("=" * 78)
    h_peak = histogram_by_status(all_records, "peak", "status_strict")
    print(f"  {'bucket':<16} {'TP':>6} {'FP':>6} {'total':>6}   TP/(TP+FP)")
    for label, counts in h_peak.items():
        tp = counts["TP"]; fp = counts["FP"]
        total = tp + fp
        ratio = (tp / total) if total else 0.0
        print(f"  {label:<16} {tp:>6} {fp:>6} {total:>6}   {ratio:.3f}")
    print()

    # Cross-video distribution shape
    all_mean = [r["confidence"]["mean"] for r in all_records
                if r["confidence"]["mean"] is not None]
    tp_mean = [r["confidence"]["mean"] for r in all_records
               if r["status_strict"] == "TP" and r["confidence"]["mean"] is not None]
    fp_mean = [r["confidence"]["mean"] for r in all_records
               if r["status_strict"] == "FP" and r["confidence"]["mean"] is not None]
    overall_otsu = otsu_threshold(all_mean)

    print("=" * 78)
    print("OVERALL DISTRIBUTION (cross-video)")
    print("=" * 78)
    print(f"  All reaches mean_proba: n={len(all_mean)}  "
          f"p25={np.percentile(all_mean, 25):.3f}  "
          f"p50={np.percentile(all_mean, 50):.3f}  "
          f"p75={np.percentile(all_mean, 75):.3f}")
    print(f"  TP (strict) mean_proba: n={len(tp_mean)}  "
          f"p25={np.percentile(tp_mean, 25):.3f}  "
          f"p50={np.percentile(tp_mean, 50):.3f}  "
          f"p75={np.percentile(tp_mean, 75):.3f}")
    print(f"  FP (strict) mean_proba: n={len(fp_mean)}  "
          f"p25={np.percentile(fp_mean, 25):.3f}  "
          f"p50={np.percentile(fp_mean, 50):.3f}  "
          f"p75={np.percentile(fp_mean, 75):.3f}")
    print(f"  Otsu threshold (all reaches): {overall_otsu:.3f}" if overall_otsu else "  Otsu: insufficient data")
    print()

    # Per-video summary
    per_video = per_video_shape_summary(all_records)
    print("=" * 78)
    print("PER-VIDEO SHAPE METRICS")
    print("=" * 78)
    print(f"  {'video':<35} {'n_reach':>7} {'TP_s':>5} {'FP_s':>5} "
          f"{'TP_med':>7} {'FP_med':>7} {'Otsu':>6}")
    for vid in sorted(per_video.keys()):
        m = per_video[vid]
        tp_med = m["mean_proba_tp_strict_median"]
        fp_med = m["mean_proba_fp_strict_median"]
        otsu = m["otsu_threshold_all"]
        tp_med_s = f"{tp_med:.3f}" if tp_med is not None else "  n/a"
        fp_med_s = f"{fp_med:.3f}" if fp_med is not None else "  n/a"
        otsu_s = f"{otsu:.3f}" if otsu is not None else "  n/a"
        print(f"  {vid:<35} {m['n_reaches']:>7} {m['n_tp_strict']:>5} "
              f"{m['n_fp_strict']:>5} {tp_med_s:>7} {fp_med_s:>7} {otsu_s:>6}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = OUTPUT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    (metrics_dir / "per_reach_records.json").write_text(
        json.dumps(all_records, indent=2), encoding="utf-8")

    (metrics_dir / "summary.json").write_text(
        json.dumps({
            "n_videos": len(per_video),
            "n_reaches": len(all_records),
            "n_tp_strict": sum(1 for r in all_records if r["status_strict"] == "TP"),
            "n_fp_strict": sum(1 for r in all_records if r["status_strict"] == "FP"),
            "n_tp_permissive": sum(1 for r in all_records if r["status_permissive"] == "TP"),
            "n_fp_permissive": sum(1 for r in all_records if r["status_permissive"] == "FP"),
            "mean_proba_histogram_strict": h_strict,
            "mean_proba_histogram_permissive": h_perm,
            "peak_proba_histogram_strict": h_peak,
            "overall_distribution": {
                "all_mean_p25": float(np.percentile(all_mean, 25)) if all_mean else None,
                "all_mean_p50": float(np.percentile(all_mean, 50)) if all_mean else None,
                "all_mean_p75": float(np.percentile(all_mean, 75)) if all_mean else None,
                "tp_strict_mean_p50": float(np.median(tp_mean)) if tp_mean else None,
                "fp_strict_mean_p50": float(np.median(fp_mean)) if fp_mean else None,
                "otsu_threshold_all": overall_otsu,
            },
            "per_video": per_video,
            "params": {
                "threshold": THRESHOLD,
                "merge_gap": MERGE_GAP,
                "min_span": MIN_SPAN,
                "permissive_window": PERMISSIVE_WINDOW,
                "strict_start_tol": STRICT_START_TOL,
                "strict_span_tol_rel": STRICT_SPAN_TOL_REL,
                "strict_span_tol_abs": STRICT_SPAN_TOL_ABS,
            },
        }, indent=2), encoding="utf-8")

    print(f"Wrote: {metrics_dir / 'per_reach_records.json'}")
    print(f"Wrote: {metrics_dir / 'summary.json'}")
    print()
    print("READING THE OUTPUT:")
    print("  - Mean_proba histograms: if TP/(TP+FP) ratio rises sharply with")
    print("    bucket value, the distribution is separable -- a per-reach")
    print("    confidence postfilter is viable.")
    print("  - Per-video Otsu: if it sits between TP_median and FP_median,")
    print("    bimodality is present at the video level.")
    print("  - If TP and FP medians overlap heavily (e.g., both ~0.85),")
    print("    confidence is not a useful FP separator.")


if __name__ == "__main__":
    main()
