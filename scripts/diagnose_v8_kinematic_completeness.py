"""
Diagnostic: does boundary imprecision actually corrupt kinematics?

The strict matching rule (start_tol=2, span_tol=max(0.5*gspan, 5)) rejects
an algo reach as FP/FN if its boundaries differ from GT by even a few
frames. The permissive rule (window=10, no span check) accepts the same
reach as TP. The runner-1 criterion reconciliation showed this difference
moves 76% of calibration FPs and 64% of generalization FPs.

The Cardinal Rule says "every frame boundary error contaminates kinematics."
This runner tests that empirically. For each permissive-matched (algo, GT)
pair on the v8.0.0 generalization corpus, it computes:

  - coverage:           fraction of GT frames inside the algo window
  - apex_included:      whether the GT apex frame is inside the algo window
  - anchor_at_start_ok: algo_start <= gt_start - 2 (room for pre-anchor)
  - anchor_at_end_ok:   algo_end   >= gt_end   + 2 (room for post-anchor)

Then breaks results into two groups:

  STRICT-ACCEPT: the algo reach passed start_tol=2 AND span_tol. These are
    the v8 production eval's "TPs". Expected: very high kinematic fidelity.

  STRICT-REJECT (permissive-only): the algo reach was within +/-10f of GT
    start but failed strict criteria. These are the events the production
    eval calls FP/FN -- the ones we'd "fix" if we tried to tighten
    boundaries. The question: how many of these are actually
    kinematically usable as-is?

If most STRICT-REJECT events have apex included, coverage >= 0.8, and both
anchors available, then most "boundary errors" the strict rule penalizes
are NOT kinematically damaging. The Cardinal Rule still binds for the
remainder, but the bulk of the work the strict eval is doing might be
chasing measurement-rule artifacts that don't affect downstream kinematics.

================================================================
INPUTS
================================================================

  Algo reaches (v8.0.0 production, 20-video generalization corpus):
    Improvement_Snapshots/reach_detection/v8.0.0_generalization_20video/
      algo_outputs_v8.0.0/<video_id>_reaches.json
    Schema: {"reaches": [{"start_frame", "end_frame", ...}, ...]}

  GT (apex frames are required; unified schema preferred):
    iterations/generalization_test_2026-05-11/gt/
      <video_id>_unified_ground_truth.json
    Schema: {"reaches": {"reaches": [{"start_frame", "end_frame",
            "apex_frame", "exclude_from_analysis", ...}, ...]}}

================================================================
OUTPUTS
================================================================

  Improvement_Snapshots/reach_detection/v8.0.0_kinematic_completeness_generalization/
    metrics/
      kinematic_completeness.json   # per-reach details + buckets
      summary.json                  # headline aggregates

================================================================
WHAT THIS DOES NOT DO
================================================================

This is NOT an experiment. No algorithm change, no model retraining, no
parameter tuning. Pure analysis on existing v8.0.0 inference outputs +
existing GT files. Read-only on all inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
    Reach, match_reaches,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ALGO_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_generalization_20video\algo_outputs_v8.0.0"
)

GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\gt"
)

OUTPUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_kinematic_completeness_generalization"
)

# Matching params
PERMISSIVE_WINDOW = 10            # +/-10f start tolerance, no span check
STRICT_START_TOL = 2
STRICT_SPAN_TOL_REL = 0.5
STRICT_SPAN_TOL_ABS = 5

# Kinematic-safety thresholds
ANCHOR_FRAMES = 2
COVERAGE_OK_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_v8_reaches(path: Path) -> List[Reach]:
    """Load v8.0.0 per-video reaches JSON written by infer_v8_on_...py."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, r in enumerate(data.get("reaches", [])):
        out.append(Reach(start_frame=int(r["start_frame"]),
                         end_frame=int(r["end_frame"]),
                         index=i))
    return out


def _load_gt_reach_dicts(gt_dir: Path, video_id: str
                         ) -> Optional[List[Dict[str, Any]]]:
    """Load GT reach dicts with apex_frame. Returns None if no GT or not
    exhaustive."""
    unified = gt_dir / f"{video_id}_unified_ground_truth.json"
    split = gt_dir / f"{video_id}_reach_ground_truth.json"
    if unified.exists():
        data = json.loads(unified.read_text(encoding="utf-8"))
        reach_data = data.get("reaches", {})
        if not isinstance(reach_data, dict):
            return None
        if not reach_data.get("exhaustive"):
            return None
        raw = reach_data.get("reaches", [])
    elif split.exists():
        data = json.loads(split.read_text(encoding="utf-8"))
        raw = []
        for seg in data.get("segments", []):
            raw.extend(seg.get("reaches", []))
    else:
        return None
    filtered = [r for r in raw if not r.get("exclude_from_analysis", False)]
    filtered.sort(key=lambda r: r["start_frame"])
    return filtered


# ---------------------------------------------------------------------------
# Per-reach kinematic completeness (inlined for clarity + accessibility of
# all signed deltas needed for bucketing)
# ---------------------------------------------------------------------------

def _strict_match_pair(gt_start: int, gt_end: int,
                       algo_start: int, algo_end: int) -> bool:
    """Would this pair pass the strict criterion?"""
    if abs(algo_start - gt_start) > STRICT_START_TOL:
        return False
    g_span = gt_end - gt_start + 1
    a_span = algo_end - algo_start + 1
    tol = max(STRICT_SPAN_TOL_REL * g_span, STRICT_SPAN_TOL_ABS)
    return abs(a_span - g_span) <= tol


def _coverage(gt_start: int, gt_end: int,
              algo_start: int, algo_end: int) -> float:
    n_gt = gt_end - gt_start + 1
    if n_gt <= 0:
        return 0.0
    ov_start = max(gt_start, algo_start)
    ov_end = min(gt_end, algo_end)
    n_ov = max(0, ov_end - ov_start + 1)
    return n_ov / n_gt


def compute_per_video(gt_dicts: List[Dict[str, Any]],
                      algo_reaches: List[Reach]
                      ) -> List[Dict[str, Any]]:
    """For each permissive-matched (algo, GT) pair, return a record with
    kinematic completeness indicators + strict-accept flag.

    Also surfaces GT-only (no permissive match) and algo-only events for
    completeness, marked as "perm_fn" / "perm_fp" respectively. The
    headline analysis focuses on the matched ones.
    """
    gt_objs = [Reach(start_frame=int(r["start_frame"]),
                     end_frame=int(r["end_frame"]),
                     index=i)
               for i, r in enumerate(gt_dicts)]
    apex_by_gt_idx = {i: (int(r["apex_frame"])
                          if r.get("apex_frame") is not None else None)
                      for i, r in enumerate(gt_dicts)}

    results = match_reaches(
        algo_reaches, gt_objs,
        window=PERMISSIVE_WINDOW, strict=False,
    )

    out = []
    for r in results:
        if r.status == "matched":
            gt_start = r.gt_start; gt_end = r.gt_end
            a_start = r.algo_start; a_end = r.algo_end
            gt_idx = r.gt_reach_index
            gt_apex = apex_by_gt_idx.get(gt_idx) if gt_idx is not None else None
            strict_ok = _strict_match_pair(gt_start, gt_end, a_start, a_end)
            cov = _coverage(gt_start, gt_end, a_start, a_end)
            apex_inside = (gt_apex is not None
                           and a_start <= gt_apex <= a_end)
            anchor_s_ok = a_start <= gt_start - ANCHOR_FRAMES
            anchor_e_ok = a_end >= gt_end + ANCHOR_FRAMES
            safe = (apex_inside and cov >= COVERAGE_OK_THRESHOLD
                    and anchor_s_ok and anchor_e_ok)
            out.append({
                "kind": "matched",
                "gt_start": gt_start, "gt_end": gt_end, "gt_apex": gt_apex,
                "algo_start": a_start, "algo_end": a_end,
                "start_delta": int(a_start - gt_start),
                "end_delta": int(a_end - gt_end),
                "span_delta": int((a_end - a_start + 1) - (gt_end - gt_start + 1)),
                "strict_ok": bool(strict_ok),
                "coverage": round(cov, 4),
                "apex_included": bool(apex_inside) if gt_apex is not None else None,
                "anchor_at_start_ok": bool(anchor_s_ok),
                "anchor_at_end_ok": bool(anchor_e_ok),
                "kinematically_safe": (bool(safe)
                                       if gt_apex is not None else None),
            })
        elif r.status == "fn":
            out.append({
                "kind": "perm_fn",
                "gt_start": r.gt_start, "gt_end": r.gt_end,
                "gt_apex": apex_by_gt_idx.get(r.gt_reach_index),
            })
        elif r.status == "fp":
            out.append({
                "kind": "perm_fp",
                "algo_start": r.algo_start, "algo_end": r.algo_end,
            })
    return out


# ---------------------------------------------------------------------------
# Bucketing + aggregation
# ---------------------------------------------------------------------------

def _bucket_start_delta(d: int) -> str:
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


def _bucket_span_delta(d: int) -> str:
    a = abs(d)
    if a == 0:
        return "0"
    if a <= 2:
        return "1-2"
    if a <= 5:
        return "3-5"
    if a <= 10:
        return "6-10"
    if a <= 20:
        return "11-20"
    return "21+"


def aggregate(per_video: Dict[str, List[Dict[str, Any]]]
              ) -> Dict[str, Any]:
    """Aggregate kinematic completeness across all matched events."""
    all_matched = []
    for vid, recs in per_video.items():
        for r in recs:
            if r.get("kind") == "matched":
                r2 = dict(r); r2["video_id"] = vid
                all_matched.append(r2)

    def _grp_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(items)
        if n == 0:
            return {"n": 0}
        cov = np.array([r["coverage"] for r in items])
        apex = [r["apex_included"] for r in items if r["apex_included"] is not None]
        anc_s = [r["anchor_at_start_ok"] for r in items]
        anc_e = [r["anchor_at_end_ok"] for r in items]
        safe = [r["kinematically_safe"] for r in items if r["kinematically_safe"] is not None]
        return {
            "n": n,
            "median_coverage": float(np.median(cov)),
            "mean_coverage": float(np.mean(cov)),
            "frac_coverage_ok": float((cov >= COVERAGE_OK_THRESHOLD).mean()),
            "frac_coverage_full": float((cov >= 1.0).mean()),
            "n_apex_evaluable": len(apex),
            "frac_apex_included": float(np.mean(apex)) if apex else None,
            "frac_anchor_start_ok": float(np.mean(anc_s)),
            "frac_anchor_end_ok": float(np.mean(anc_e)),
            "frac_both_anchors_ok": float(
                np.mean([a and b for a, b in zip(anc_s, anc_e)])),
            "n_safety_evaluable": len(safe),
            "frac_kinematically_safe": float(np.mean(safe)) if safe else None,
        }

    overall = _grp_stats(all_matched)
    strict_accept = _grp_stats([r for r in all_matched if r["strict_ok"]])
    strict_reject = _grp_stats([r for r in all_matched if not r["strict_ok"]])

    # By |start_delta|
    by_start = {}
    for b in ["0", "1-2", "3-5", "6-10", "11+"]:
        by_start[b] = _grp_stats(
            [r for r in all_matched if _bucket_start_delta(r["start_delta"]) == b])

    # By |span_delta|
    by_span = {}
    for b in ["0", "1-2", "3-5", "6-10", "11-20", "21+"]:
        by_span[b] = _grp_stats(
            [r for r in all_matched if _bucket_span_delta(r["span_delta"]) == b])

    return {
        "overall_permissive_matched": overall,
        "strict_accept_subset": strict_accept,
        "strict_reject_subset_borderline": strict_reject,
        "by_abs_start_delta": by_start,
        "by_abs_span_delta": by_span,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("KINEMATIC COMPLETENESS on v8.0.0 generalization corpus (permissive match)")
    print("=" * 70)
    print()

    if not ALGO_DIR.exists():
        raise FileNotFoundError(f"Algo dir not found: {ALGO_DIR}")
    if not GT_DIR.exists():
        raise FileNotFoundError(f"GT dir not found: {GT_DIR}")

    # Discover videos that have BOTH algo output and exhaustive GT
    algo_files = list(ALGO_DIR.glob("*_reaches.json"))
    eligible = []
    for af in algo_files:
        vid = af.stem.replace("_reaches", "")
        gt_dicts = _load_gt_reach_dicts(GT_DIR, vid)
        if gt_dicts is None:
            print(f"  skipping {vid}: no exhaustive GT")
            continue
        eligible.append(vid)
    print(f"\nEligible videos: {len(eligible)}")
    print()

    per_video = {}
    for vid in eligible:
        algo_path = ALGO_DIR / f"{vid}_reaches.json"
        algo_reaches = _load_v8_reaches(algo_path)
        gt_dicts = _load_gt_reach_dicts(GT_DIR, vid)
        recs = compute_per_video(gt_dicts, algo_reaches)
        per_video[vid] = recs
        n_matched = sum(1 for r in recs if r.get("kind") == "matched")
        n_strict = sum(1 for r in recs if r.get("kind") == "matched" and r.get("strict_ok"))
        n_strict_reject = n_matched - n_strict
        print(f"  {vid:35} matched={n_matched:4}  strict_accept={n_strict:4}  "
              f"strict_reject(borderline)={n_strict_reject:4}")

    print()
    agg = aggregate(per_video)

    def _fmt(stats: Dict[str, Any]) -> str:
        if stats.get("n", 0) == 0:
            return "n=0"
        parts = [f"n={stats['n']}"]
        parts.append(f"cov_med={stats['median_coverage']:.3f}")
        parts.append(f"cov_ok={stats['frac_coverage_ok']:.1%}")
        parts.append(f"cov_full={stats['frac_coverage_full']:.1%}")
        if stats.get('frac_apex_included') is not None:
            parts.append(f"apex_in={stats['frac_apex_included']:.1%}")
        parts.append(f"both_anc={stats['frac_both_anchors_ok']:.1%}")
        if stats.get('frac_kinematically_safe') is not None:
            parts.append(f"safe={stats['frac_kinematically_safe']:.1%}")
        return "  ".join(parts)

    print("=" * 70)
    print("HEADLINE: permissive-matched (algo, GT) pairs")
    print("=" * 70)
    print(f"  OVERALL                {_fmt(agg['overall_permissive_matched'])}")
    print(f"  STRICT-ACCEPT subset   {_fmt(agg['strict_accept_subset'])}")
    print(f"  STRICT-REJECT subset   {_fmt(agg['strict_reject_subset_borderline'])}")
    print()
    print("  (strict-reject = passes permissive window=10 but fails")
    print("   strict_start_tol=2 OR span tolerance. The 'borderline' tail.)")
    print()

    print("=" * 70)
    print("BY |start_delta| (signed start_delta = algo_start - gt_start)")
    print("=" * 70)
    for b, stats in agg["by_abs_start_delta"].items():
        print(f"  |start_delta| {b:>5}f   {_fmt(stats)}")
    print()

    print("=" * 70)
    print("BY |span_delta|")
    print("=" * 70)
    for b, stats in agg["by_abs_span_delta"].items():
        print(f"  |span_delta| {b:>6}f   {_fmt(stats)}")
    print()

    # Persist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = OUTPUT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    (metrics_dir / "summary.json").write_text(
        json.dumps({
            "anchor_frames": ANCHOR_FRAMES,
            "coverage_ok_threshold": COVERAGE_OK_THRESHOLD,
            "permissive_window": PERMISSIVE_WINDOW,
            "strict_start_tol": STRICT_START_TOL,
            "strict_span_tol_rel": STRICT_SPAN_TOL_REL,
            "strict_span_tol_abs": STRICT_SPAN_TOL_ABS,
            "n_videos": len(eligible),
            "aggregates": agg,
        }, indent=2), encoding="utf-8")

    (metrics_dir / "kinematic_completeness.json").write_text(
        json.dumps(per_video, indent=2), encoding="utf-8")

    print(f"Wrote: {metrics_dir / 'summary.json'}")
    print(f"Wrote: {metrics_dir / 'kinematic_completeness.json'}")
    print()
    print("NEXT: human reads the STRICT-REJECT subset row. If most events in")
    print("that subset have apex_included AND coverage >= 0.8 AND both anchors,")
    print("then the strict eval rule is over-penalizing boundary near-misses")
    print("that don't actually corrupt kinematics. The Cardinal Rule binds")
    print("for the remainder (events that are NOT kinematically safe).")


if __name__ == "__main__":
    main()
