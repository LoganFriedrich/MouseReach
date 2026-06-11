"""
Diagnostic: how much do actual kinematic feature VALUES drift when the
algo window differs from the GT window?

For each permissive-matched (algo, GT) pair on the v8.0.0 generalization
corpus, compute six representative kinematic features TWICE -- once with
the algo window [algo_start, algo_end], once with the GT window
[gt_start, gt_end] -- using the same underlying DLC paw trajectories.
Report the absolute and percentage difference per feature.

This converts the abstract "apex_in 98% / coverage 89%" headline from
the kinematic completeness runner into concrete numbers: e.g.,
"total_path differs by 4.2% on average in the strict-reject subset."

Six features chosen to span the three sensitivity classes documented in
CARDINAL_RULE_NUANCE_2026-05-18.md:

  Class A (apex-anchored, expected robust):
    - extension_past_nose_at_apex      (RightHand y - Nose y at gt_apex)

  Class B (boundary-direct, expected linearly sensitive):
    - duration_frames                  (end - start + 1)
    - paw_width_at_start               (RHLeft <-> RHRight at start frame)
    - paw_width_at_end                 (same at end frame)

  Class C (window-aggregate, expected design-cushioned):
    - total_path                       (sum of frame-to-frame RightHand
                                        distance over the window)
    - peak_speed                       (max frame-to-frame RightHand
                                        distance over the window)

For class C the synthetic anchors are NOT included in this inline
implementation -- production trajectory features use synthetic anchors,
this diagnostic does not. The trend (algo-vs-GT drift) is what's
informative; the absolute values won't match the production export.

================================================================
INPUTS
================================================================

  DLC h5 files (paw + nose positions, per video):
    iterations/generalization_test_2026-05-11/algo_outputs_current/
      <video_id>DLC_resnet50_MPSAOct27shuffle1_100000.h5

  Algo reaches (v8.0.0 inference output, per video):
    Improvement_Snapshots/reach_detection/v8.0.0_generalization_20video/
      algo_outputs_v8.0.0/<video_id>_reaches.json

  GT (with apex frames):
    iterations/generalization_test_2026-05-11/gt/
      <video_id>_unified_ground_truth.json

================================================================
OUTPUT
================================================================

  Improvement_Snapshots/reach_detection/v8.0.0_kinematic_value_drift/
    metrics/
      drift_per_event.json    # one record per permissive-matched pair
      summary.json            # aggregates by strict_ok bucket + drift bucket
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.core.geometry import load_dlc
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

ALGO_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_generalization_20video\algo_outputs_v8.0.0"
)

OUTPUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_kinematic_value_drift"
)


# Matching tolerances
PERMISSIVE_WINDOW = 10
STRICT_START_TOL = 2
STRICT_SPAN_TOL_REL = 0.5
STRICT_SPAN_TOL_ABS = 5


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_v8_reaches(path: Path) -> List[Reach]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Reach(start_frame=int(r["start_frame"]),
                  end_frame=int(r["end_frame"]),
                  index=i)
            for i, r in enumerate(data.get("reaches", []))]


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
# Feature computation (inline; matches data-dictionary semantics for
# the chosen features, but trajectory features omit synthetic anchors)
# ---------------------------------------------------------------------------

def _safe_xy(dlc: pd.DataFrame, bp: str, frame: int
             ) -> Tuple[Optional[float], Optional[float]]:
    if frame < 0 or frame >= len(dlc):
        return None, None
    row = dlc.iloc[frame]
    x = row.get(f"{bp}_x"); y = row.get(f"{bp}_y")
    if x is None or y is None or (isinstance(x, float) and np.isnan(x)):
        return None, None
    return float(x), float(y)


def _pair_dist(dlc: pd.DataFrame, bp_a: str, bp_b: str, frame: int
               ) -> Optional[float]:
    ax, ay = _safe_xy(dlc, bp_a, frame)
    bx, by = _safe_xy(dlc, bp_b, frame)
    if ax is None or bx is None:
        return None
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


def _extension_at_apex(dlc: pd.DataFrame, apex_frame: int) -> Optional[float]:
    """RightHand y - Nose y at the GT apex frame. Same for both windows
    as long as apex is inside both. Returns None if data missing.
    """
    rh_x, rh_y = _safe_xy(dlc, "RightHand", apex_frame)
    n_x, n_y = _safe_xy(dlc, "Nose", apex_frame)
    if rh_y is None or n_y is None:
        return None
    return float(rh_y - n_y)


def _total_path_and_peak_speed(dlc: pd.DataFrame, start: int, end: int
                                ) -> Tuple[Optional[float], Optional[float]]:
    """Frame-to-frame RightHand displacement over [start, end] inclusive."""
    if end < start or end >= len(dlc) or start < 0:
        return None, None
    xs = dlc["RightHand_x"].iloc[start:end + 1].to_numpy(dtype=np.float64)
    ys = dlc["RightHand_y"].iloc[start:end + 1].to_numpy(dtype=np.float64)
    if len(xs) < 2:
        return 0.0, 0.0
    dx = np.diff(xs); dy = np.diff(ys)
    seg = np.sqrt(dx ** 2 + dy ** 2)
    seg_valid = seg[~np.isnan(seg)]
    if len(seg_valid) == 0:
        return None, None
    return float(seg_valid.sum()), float(seg_valid.max())


def compute_features(dlc: pd.DataFrame, start: int, end: int,
                     apex_frame: int) -> Dict[str, Optional[float]]:
    duration = int(end - start + 1)
    extension = _extension_at_apex(dlc, apex_frame)
    paw_w_start = _pair_dist(dlc, "RHLeft", "RHRight", start)
    paw_w_end = _pair_dist(dlc, "RHLeft", "RHRight", end)
    total_path, peak_speed = _total_path_and_peak_speed(dlc, start, end)
    return {
        "duration_frames": float(duration),
        "extension_past_nose": extension,
        "paw_width_at_start": paw_w_start,
        "paw_width_at_end": paw_w_end,
        "total_path": total_path,
        "peak_speed": peak_speed,
    }


FEATURE_NAMES = [
    "duration_frames",
    "extension_past_nose",
    "paw_width_at_start",
    "paw_width_at_end",
    "total_path",
    "peak_speed",
]

FEATURE_CLASS = {
    "duration_frames": "B",
    "extension_past_nose": "A",
    "paw_width_at_start": "B",
    "paw_width_at_end": "B",
    "total_path": "C",
    "peak_speed": "C",
}


# ---------------------------------------------------------------------------
# Per-pair drift
# ---------------------------------------------------------------------------

def _strict_match_pair(gt_start: int, gt_end: int,
                       algo_start: int, algo_end: int) -> bool:
    if abs(algo_start - gt_start) > STRICT_START_TOL:
        return False
    g_span = gt_end - gt_start + 1
    a_span = algo_end - algo_start + 1
    tol = max(STRICT_SPAN_TOL_REL * g_span, STRICT_SPAN_TOL_ABS)
    return abs(a_span - g_span) <= tol


def _diff(algo_val: Optional[float], gt_val: Optional[float]
          ) -> Tuple[Optional[float], Optional[float]]:
    """(absolute_diff, percent_diff_relative_to_gt) or (None, None) if NA."""
    if algo_val is None or gt_val is None:
        return None, None
    abs_d = algo_val - gt_val
    if abs(gt_val) < 1e-9:
        return float(abs_d), None
    pct = 100.0 * abs_d / gt_val
    return float(abs_d), float(pct)


def process_video(video_id: str) -> List[Dict[str, Any]]:
    """Return one record per permissive-matched pair."""
    dlc_path = _find_dlc(video_id)
    if dlc_path is None:
        print(f"  skipping {video_id}: no DLC h5")
        return []
    gt_dicts = _load_gt_with_apex(video_id)
    if gt_dicts is None:
        print(f"  skipping {video_id}: no exhaustive GT")
        return []
    algo_path = ALGO_DIR / f"{video_id}_reaches.json"
    if not algo_path.exists():
        print(f"  skipping {video_id}: no v8 algo output")
        return []

    dlc = load_dlc(dlc_path)
    algo_reaches = _load_v8_reaches(algo_path)
    gt_objs = [Reach(start_frame=int(r["start_frame"]),
                     end_frame=int(r["end_frame"]),
                     index=i)
               for i, r in enumerate(gt_dicts)]

    results = match_reaches(
        algo_reaches, gt_objs,
        window=PERMISSIVE_WINDOW, strict=False)

    records = []
    for r in results:
        if r.status != "matched":
            continue
        gt_idx = r.gt_reach_index
        if gt_idx is None:
            continue
        gt_dict = gt_dicts[gt_idx]
        gt_apex = gt_dict.get("apex_frame")
        if gt_apex is None:
            continue
        gt_apex = int(gt_apex)

        a_start = int(r.algo_start); a_end = int(r.algo_end)
        g_start = int(r.gt_start);   g_end = int(r.gt_end)

        # Apex inside both windows?  Required for the extension feature
        apex_in_algo = a_start <= gt_apex <= a_end
        apex_in_gt = g_start <= gt_apex <= g_end

        feats_algo = compute_features(dlc, a_start, a_end, gt_apex)
        feats_gt = compute_features(dlc, g_start, g_end, gt_apex)

        diffs = {}
        for f in FEATURE_NAMES:
            abs_d, pct_d = _diff(feats_algo[f], feats_gt[f])
            diffs[f] = {
                "algo_value": feats_algo[f],
                "gt_value": feats_gt[f],
                "abs_diff": abs_d,
                "pct_diff": pct_d,
            }

        records.append({
            "video_id": video_id,
            "gt_start": g_start, "gt_end": g_end, "gt_apex": gt_apex,
            "algo_start": a_start, "algo_end": a_end,
            "start_delta": a_start - g_start,
            "end_delta": a_end - g_end,
            "span_delta": (a_end - a_start + 1) - (g_end - g_start + 1),
            "apex_in_algo_window": bool(apex_in_algo),
            "apex_in_gt_window": bool(apex_in_gt),
            "strict_ok": bool(_strict_match_pair(g_start, g_end, a_start, a_end)),
            "feature_diffs": diffs,
        })
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _abs_start_bucket(d: int) -> str:
    a = abs(d)
    if a == 0:
        return "0"
    if a <= 2:
        return "1-2"
    if a <= 5:
        return "3-5"
    if a <= 10:
        return "6-10"
    return "11+"


def aggregate_by_group(records: List[Dict[str, Any]],
                       group_filter) -> Dict[str, Any]:
    """For a filtered subset of records, aggregate per-feature diff stats."""
    items = [r for r in records if group_filter(r)]
    n = len(items)
    if n == 0:
        return {"n": 0}
    out = {"n": n}
    for f in FEATURE_NAMES:
        abs_diffs = [r["feature_diffs"][f]["abs_diff"]
                     for r in items
                     if r["feature_diffs"][f]["abs_diff"] is not None]
        pct_diffs = [r["feature_diffs"][f]["pct_diff"]
                     for r in items
                     if r["feature_diffs"][f]["pct_diff"] is not None]
        if not abs_diffs:
            out[f] = {"n_eval": 0}
            continue
        arr_abs = np.array(abs_diffs)
        arr_pct = np.array(pct_diffs) if pct_diffs else None
        out[f] = {
            "n_eval": len(abs_diffs),
            "abs_diff_mean": float(np.mean(arr_abs)),
            "abs_diff_median": float(np.median(arr_abs)),
            "abs_diff_p10": float(np.percentile(arr_abs, 10)),
            "abs_diff_p90": float(np.percentile(arr_abs, 90)),
            "abs_diff_mean_of_abs": float(np.mean(np.abs(arr_abs))),
            "pct_diff_mean": float(np.mean(arr_pct)) if arr_pct is not None else None,
            "pct_diff_median": float(np.median(arr_pct)) if arr_pct is not None else None,
            "pct_diff_mean_of_abs": float(np.mean(np.abs(arr_pct))) if arr_pct is not None else None,
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("KINEMATIC VALUE DRIFT -- algo window vs GT window (v8.0.0)")
    print("=" * 70)
    print()

    if not DLC_DIR.exists():
        raise FileNotFoundError(f"DLC dir not found: {DLC_DIR}")
    if not ALGO_DIR.exists():
        raise FileNotFoundError(f"Algo dir not found: {ALGO_DIR}")

    # Discover videos
    algo_files = sorted(ALGO_DIR.glob("*_reaches.json"))
    video_ids = [f.stem.replace("_reaches", "") for f in algo_files]
    print(f"Candidate videos: {len(video_ids)}")
    print()

    all_records = []
    for vid in video_ids:
        recs = process_video(vid)
        if recs:
            print(f"  {vid:35} {len(recs)} matched pairs")
        all_records.extend(recs)
    print(f"\nTotal matched pairs: {len(all_records)}")
    print()

    # Aggregate by strict_ok subsets
    overall = aggregate_by_group(all_records, lambda r: True)
    strict_acc = aggregate_by_group(all_records, lambda r: r["strict_ok"])
    strict_rej = aggregate_by_group(all_records, lambda r: not r["strict_ok"])

    # Aggregate by |start_delta| bucket
    by_start = {}
    for b in ["0", "1-2", "3-5", "6-10", "11+"]:
        by_start[b] = aggregate_by_group(
            all_records, lambda r, b=b: _abs_start_bucket(r["start_delta"]) == b)

    # Print headline
    def _print_group(label: str, agg: Dict[str, Any]) -> None:
        print(f"  {label} (n={agg['n']}):")
        if agg["n"] == 0:
            return
        for f in FEATURE_NAMES:
            d = agg.get(f, {})
            if not d or d.get("n_eval", 0) == 0:
                print(f"    [{FEATURE_CLASS[f]}] {f:>22} -- no data")
                continue
            pct = d.get("pct_diff_mean_of_abs")
            pct_str = f"{pct:6.2f}%" if pct is not None else "  n/a "
            abs_med = d.get("abs_diff_median", 0.0)
            print(f"    [{FEATURE_CLASS[f]}] {f:>22} "
                  f"mean|pct_diff|={pct_str}  "
                  f"abs_diff median={abs_med:+.3f}")

    print("=" * 70)
    print("DRIFT BY STRICT SUBSET")
    print("=" * 70)
    _print_group("OVERALL", overall)
    print()
    _print_group("STRICT-ACCEPT", strict_acc)
    print()
    _print_group("STRICT-REJECT (borderline)", strict_rej)
    print()

    print("=" * 70)
    print("DRIFT BY |start_delta| BUCKET")
    print("=" * 70)
    for b in ["0", "1-2", "3-5", "6-10", "11+"]:
        _print_group(f"|start_delta| = {b} frames", by_start[b])
        print()

    # Persist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = OUTPUT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    (metrics_dir / "drift_per_event.json").write_text(
        json.dumps(all_records, indent=2), encoding="utf-8")
    (metrics_dir / "summary.json").write_text(
        json.dumps({
            "feature_class": FEATURE_CLASS,
            "n_events": len(all_records),
            "overall": overall,
            "strict_accept": strict_acc,
            "strict_reject": strict_rej,
            "by_abs_start_delta": by_start,
            "tolerances": {
                "permissive_window": PERMISSIVE_WINDOW,
                "strict_start_tol": STRICT_START_TOL,
                "strict_span_tol_rel": STRICT_SPAN_TOL_REL,
                "strict_span_tol_abs": STRICT_SPAN_TOL_ABS,
            },
        }, indent=2), encoding="utf-8")

    print(f"Wrote: {metrics_dir / 'drift_per_event.json'}")
    print(f"Wrote: {metrics_dir / 'summary.json'}")
    print()
    print("READING THE OUTPUT:")
    print("  [A] features are apex-anchored -- mean|pct_diff| should be 0")
    print("      because the apex is in both windows.")
    print("  [B] features are boundary-direct -- mean|pct_diff| should rise")
    print("      proportionally with |start_delta| / |end_delta|.")
    print("  [C] features are window-aggregate -- mean|pct_diff| should be")
    print("      moderate, cushioned by the limited window overlap.")


if __name__ == "__main__":
    main()
