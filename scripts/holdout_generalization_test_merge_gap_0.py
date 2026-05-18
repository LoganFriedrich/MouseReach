"""
Holdout generalization test for merge_gap=0 (playbook step 5).

Runs production v8.0.0 inference with `merge_gap=0` on the 20-video
generalization corpus (videos the model never trained on) and compares
to the existing mg=2 results from the same corpus
(`v8.0.0_generalization_20video/` snapshot, produced 2026-05-18).

This is the playbook step-5 gate that must pass before mg=0 can be
merged to production. The same gate was used for BSW w=0.8 acceptance
(4-video holdout) and the same pattern is followed here on a larger
20-video set for additional statistical power.

NOT an experiment in the pre-experiment-checklist sense -- this is a
post-LOOCV verification step. mg=0 was already accepted on the
calibration LOOCV (snapshot `v8.0.0_dev_merge_gap_0_candidate/`); this
run confirms the result generalizes to the held-out corpus.

================================================================
INPUTS
================================================================

  DLC h5 files (paw + nose + landmarks, per video):
    iterations/generalization_test_2026-05-11/algo_outputs_current/
      <video_id>DLC_resnet50_MPSAOct27shuffle1_100000.h5

  GT files:
    iterations/generalization_test_2026-05-11/gt/
      <video_id>_unified_ground_truth.json

  v8.0.0 production model (bundled in package):
    src/mousereach/reach/v8/models/v8.0.0_bsw_w0.8.joblib

  Baseline result for comparison (mg=2 on same corpus):
    Improvement_Snapshots/reach_detection/v8.0.0_generalization_20video/
      metrics/reach_detection_scalars.json
    Aggregate: TP=3356 / FP=317 / FN=368

================================================================
OUTPUT
================================================================

  Improvement_Snapshots/reach_detection/v8.0.0_holdout_generalization_merge_gap_0/
    manifest.json
    algo_outputs_v8.0.0_mg0/
      <video_id>_reaches.json
    metrics/
      reach_detection_scalars.json    # same schema as the mg=2 baseline
      gate_decision.json              # PASS/FAIL + deltas

================================================================
DECISION RULE FOR GATE
================================================================

  PASS if FN drops or TP rises vs mg=2 baseline, AND start_delta
  median/abs_median preserved (no boundary-precision regression).
  FAIL if TP drops AND FN rises.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.core.geometry import load_dlc
from mousereach.reach.v8 import detect_reaches_v8, VERSION as V8_VERSION
from mousereach.improvement.reach_detection.metrics import (
    Reach, match_reaches, _load_gt_reaches, _find_gt_file,
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

OUTPUT_SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_holdout_generalization_merge_gap_0"
)

# Existing mg=2 baseline (for the gate comparison)
BASELINE_TP = 3356
BASELINE_FP = 317
BASELINE_FN = 368

# Strict matching params (same as production eval)
STRICT_START_TOL = 2
STRICT_SPAN_TOL_REL = 0.5
STRICT_SPAN_TOL_ABS = 5

# The change being tested
MERGE_GAP = 0


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_videos() -> List[str]:
    """Find video IDs with both a DLC h5 and a GT file."""
    h5_files = list(DLC_DIR.glob("*DLC_*.h5"))
    h5_video_ids = set()
    for f in h5_files:
        name = f.stem
        idx = name.find("DLC_")
        if idx > 0:
            h5_video_ids.add(name[:idx])

    gt_video_ids = set()
    for f in GT_DIR.glob("*_unified_ground_truth.json"):
        gt_video_ids.add(f.stem.replace("_unified_ground_truth", ""))
    for f in GT_DIR.glob("*_reach_ground_truth.json"):
        gt_video_ids.add(f.stem.replace("_reach_ground_truth", ""))

    return sorted(h5_video_ids & gt_video_ids)


# ---------------------------------------------------------------------------
# Per-video inference + scoring with mg=0
# ---------------------------------------------------------------------------

def infer_and_score(video_id: str) -> Tuple[List[Tuple[int, int]],
                                            List[Tuple[int, int]],
                                            List[Dict[str, Any]]]:
    """Run v8.0.0 inference with mg=0 and match strictly."""
    h5_candidates = list(DLC_DIR.glob(f"{video_id}DLC_*.h5"))
    if not h5_candidates:
        raise FileNotFoundError(f"No DLC h5 for {video_id}")
    h5_path = h5_candidates[0]

    dlc_df = load_dlc(h5_path)
    # Explicit merge_gap=0 -- the change being tested
    algo_tuples = detect_reaches_v8(dlc_df, merge_gap=MERGE_GAP)

    gt_path = _find_gt_file(GT_DIR, video_id)
    if gt_path is None:
        raise FileNotFoundError(f"No GT for {video_id}")
    gt_reach_objs = _load_gt_reaches(gt_path)
    gt_tuples = [(r.start_frame, r.end_frame) for r in gt_reach_objs]

    algo_reaches = [Reach(start_frame=s, end_frame=e, index=i)
                    for i, (s, e) in enumerate(algo_tuples)]

    results = match_reaches(
        algo_reaches, gt_reach_objs,
        strict=True,
        strict_start_tol=STRICT_START_TOL,
        strict_span_tol_rel=STRICT_SPAN_TOL_REL,
        strict_span_tol_abs=STRICT_SPAN_TOL_ABS,
    )

    records = []
    for r in results:
        if r.status == "matched":
            records.append({
                "video_id": video_id, "status": "matched",
                "gt_start": r.gt_start, "gt_end": r.gt_end,
                "algo_start": r.algo_start, "algo_end": r.algo_end,
                "start_delta": r.start_delta,
                "span_delta": (r.algo_end - r.algo_start + 1)
                              - (r.gt_end - r.gt_start + 1),
            })
        elif r.status == "fn":
            records.append({
                "video_id": video_id, "status": "fn",
                "gt_start": r.gt_start, "gt_end": r.gt_end,
                "algo_start": -1, "algo_end": -1,
                "start_delta": None, "span_delta": None,
            })
        elif r.status == "fp":
            records.append({
                "video_id": video_id, "status": "fp",
                "gt_start": -1, "gt_end": -1,
                "algo_start": r.algo_start, "algo_end": r.algo_end,
                "start_delta": None, "span_delta": None,
            })
    return algo_tuples, gt_tuples, records


def save_video_reaches(video_id: str,
                       algo_tuples: List[Tuple[int, int]],
                       out_dir: Path) -> Path:
    out_path = out_dir / f"{video_id}_reaches.json"
    payload = {
        "detector_version": V8_VERSION,
        "merge_gap": MERGE_GAP,
        "video_name": video_id,
        "n_reaches": len(algo_tuples),
        "reaches": [
            {"start_frame": int(s), "end_frame": int(e),
             "duration_frames": int(e - s + 1)}
            for (s, e) in algo_tuples
        ],
        "detected_at": datetime.now().isoformat(),
        "inference_params": {
            "threshold": 0.5,
            "merge_gap": MERGE_GAP,
            "min_span": 3,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print(f"HOLDOUT GENERALIZATION TEST -- v8.0.0 with merge_gap={MERGE_GAP}")
    print(f"Comparison baseline: v8.0.0 with merge_gap=2 (current production)")
    print(f"  baseline: TP={BASELINE_TP}, FP={BASELINE_FP}, FN={BASELINE_FN}")
    print("=" * 78)
    print()

    if not DLC_DIR.exists():
        raise FileNotFoundError(f"DLC dir not found: {DLC_DIR}")
    if not GT_DIR.exists():
        raise FileNotFoundError(f"GT dir not found: {GT_DIR}")

    OUTPUT_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    algo_out_dir = OUTPUT_SNAPSHOT_DIR / "algo_outputs_v8.0.0_mg0"
    algo_out_dir.mkdir(exist_ok=True)
    metrics_dir = OUTPUT_SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    video_ids = discover_videos()
    print(f"Videos with DLC + exhaustive GT: {len(video_ids)}")
    print()

    all_matches = []
    per_video_summary = {}
    total_tp = total_fp = total_fn = 0

    for i, video_id in enumerate(video_ids):
        print(f"  [{i+1}/{len(video_ids)}] {video_id} ...", flush=True)
        try:
            algo_tuples, gt_tuples, records = infer_and_score(video_id)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            continue
        save_video_reaches(video_id, algo_tuples, algo_out_dir)
        v_tp = sum(1 for r in records if r["status"] == "matched")
        v_fp = sum(1 for r in records if r["status"] == "fp")
        v_fn = sum(1 for r in records if r["status"] == "fn")
        per_video_summary[video_id] = {
            "n_algo": len(algo_tuples), "n_gt": len(gt_tuples),
            "n_tp": v_tp, "n_fp": v_fp, "n_fn": v_fn,
        }
        total_tp += v_tp; total_fp += v_fp; total_fn += v_fn
        all_matches.extend(records)
        print(f"      n_algo={len(algo_tuples):4} n_gt={len(gt_tuples):4}  "
              f"TP={v_tp:3} FP={v_fp:3} FN={v_fn:3}")

    print()
    print("=" * 78)
    print(f"AGGREGATE -- v8.0.0 with merge_gap={MERGE_GAP} on generalization corpus")
    print("=" * 78)
    print(f"  Videos processed: {len(per_video_summary)}")
    print(f"  Total TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print()
    print(f"vs mg=2 baseline (same corpus): TP={BASELINE_TP}, FP={BASELINE_FP}, FN={BASELINE_FN}")
    d_tp = total_tp - BASELINE_TP
    d_fp = total_fp - BASELINE_FP
    d_fn = total_fn - BASELINE_FN
    print(f"  TP delta: {d_tp:+d}")
    print(f"  FP delta: {d_fp:+d}")
    print(f"  FN delta: {d_fn:+d}")
    print()

    # Boundary precision check
    matched_records = [r for r in all_matches if r["status"] == "matched"]
    start_deltas = [r["start_delta"] for r in matched_records]
    span_deltas = [r["span_delta"] for r in matched_records]
    if start_deltas:
        sd = np.array(start_deltas)
        sp = np.array(span_deltas)
        print(f"Boundary precision on matched reaches (n={len(start_deltas)}):")
        print(f"  start_delta: median={int(np.median(sd))}f  "
              f"abs_median={int(np.median(np.abs(sd)))}f  "
              f"mean={sd.mean():+.3f}f")
        print(f"  span_delta:  median={int(np.median(sp))}f  "
              f"abs_median={int(np.median(np.abs(sp)))}f  "
              f"mean={sp.mean():+.3f}f")
        sd_median = int(np.median(sd))
        sd_abs_median = int(np.median(np.abs(sd)))
    else:
        sd_median = sd_abs_median = None
    print()

    # Decision rule
    tp_drops = d_tp < 0
    fn_rises = d_fn > 0
    boundary_regressed = (sd_median is not None and (sd_median != 0 or sd_abs_median != 0))

    print("=" * 78)
    print("GATE DECISION")
    print("=" * 78)
    if tp_drops and fn_rises:
        decision = "FAIL"
        reason = "TP drops AND FN rises vs mg=2 baseline"
    elif boundary_regressed:
        decision = "FAIL"
        reason = "Boundary precision regression (start_delta median or abs_median != 0)"
    elif d_fn < 0 or d_tp > 0:
        decision = "PASS"
        reason = "FN drops or TP rises vs mg=2 baseline AND boundary precision preserved"
    else:
        decision = "AMBIGUOUS"
        reason = "Neither clean PASS nor clean FAIL"
    print(f"  {decision}: {reason}")
    print()

    # Persist
    manifest = {
        "snapshot_name": "v8.0.0_holdout_generalization_merge_gap_0",
        "phase": "reach_detection",
        "purpose": ("v8.0.0 inference with merge_gap=0 on the 20-video "
                    "generalization corpus, post-LOOCV holdout gate test."),
        "created_at": datetime.now().isoformat(),
        "detector_version": V8_VERSION,
        "merge_gap": MERGE_GAP,
        "matching_criterion": {
            "strict": True,
            "start_tol": STRICT_START_TOL,
            "span_tol_rel": STRICT_SPAN_TOL_REL,
            "span_tol_abs": STRICT_SPAN_TOL_ABS,
        },
    }
    (OUTPUT_SNAPSHOT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    scalars = {
        "n_videos": len(per_video_summary),
        "n_tp": total_tp, "n_fp": total_fp, "n_fn": total_fn,
        "detector_version": V8_VERSION,
        "merge_gap": MERGE_GAP,
        "matching_criterion": "strict",
        "matches": all_matches,
        "per_video": per_video_summary,
    }
    (metrics_dir / "reach_detection_scalars.json").write_text(
        json.dumps(scalars, indent=2), encoding="utf-8")

    gate = {
        "decision": decision,
        "reason": reason,
        "baseline_mg2": {"tp": BASELINE_TP, "fp": BASELINE_FP, "fn": BASELINE_FN},
        "candidate_mg0": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
        "deltas": {"tp": d_tp, "fp": d_fp, "fn": d_fn},
        "boundary_precision": {
            "start_delta_median": sd_median,
            "start_delta_abs_median": sd_abs_median,
        },
    }
    (metrics_dir / "gate_decision.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8")

    print(f"Wrote: {metrics_dir / 'reach_detection_scalars.json'}")
    print(f"Wrote: {metrics_dir / 'gate_decision.json'}")
    print(f"Wrote: {OUTPUT_SNAPSHOT_DIR / 'manifest.json'}")
    print(f"Wrote: {algo_out_dir}/ ({len(per_video_summary)} per-video reaches JSONs)")


if __name__ == "__main__":
    main()
