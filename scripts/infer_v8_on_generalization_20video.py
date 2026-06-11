"""
Produce v8.0.0 reach outputs on the 20-video generalization corpus and
score them against the same GT files the 2026-05-11 evaluation used.

Replaces the misattributed 2026-05-11 outputs (which were actually
v7.2.0 -- detector_version field in their JSONs confirms this) with a
true v8.0.0 evaluation on the same videos + same GT, so the criterion-
reconciliation and failure-mode-refresh diagnostics can be re-run on
data that actually reflects current production.

================================================================
INPUTS (verified 2026-05-18 by directory listing)
================================================================

  DLC h5 files:
    Y:\\2_Connectome\\Behavior\\MouseReach_Improvement\\iterations\\
      generalization_test_2026-05-11\\algo_outputs_current\\
      <video_id>DLC_resnet50_MPSAOct27shuffle1_100000.h5

  GT files:
    Y:\\2_Connectome\\Behavior\\MouseReach_Improvement\\iterations\\
      generalization_test_2026-05-11\\gt\\
      <video_id>_unified_ground_truth.json  (or _reach_ground_truth.json)

  Video list:
    Auto-discovered: every video that has BOTH a DLC h5 and a GT file
    in the above directories.

================================================================
METHOD
================================================================

For each video:
  1. Load DLC h5 via mousereach.reach.core.geometry.load_dlc.
  2. Run mousereach.reach.v8.detect_reaches_v8(dlc_df) -- the bundled
     production model (v8.0.0_bsw_w0.8.joblib), threshold=0.5,
     merge_gap=2, min_span=3.
  3. Save reaches to <video>_reaches.json in algo_outputs_v8.0.0/.
  4. Load GT reaches via the canonical
     mousereach.improvement.reach_detection.metrics._load_gt_reaches.
  5. Match with strict criterion (start_tol=2,
     span_tol=max(0.5*gspan, 5)).
  6. Append per-event records to a global matches list.

After all videos processed:
  - Write metrics/reach_detection_scalars.json in the same schema the
    existing 2026-05-11 metrics file uses (matches array with video_id,
    status, gt_start/end, algo_start/end, start_delta, span_delta).
  - Write summary counts (TP / FP / FN) + corpus-level signed span
    statistics for sanity-check against runner-1's output.

================================================================
OUTPUTS
================================================================

  Improvement_Snapshots/reach_detection/v8.0.0_generalization_20video/
    manifest.json                          # corpus + model identity
    algo_outputs_v8.0.0/
      <video_id>_reaches.json              # one per video
    metrics/
      reach_detection_scalars.json         # diagnostic-runner-compatible

================================================================
WHAT TO DO AFTER THIS RUNS
================================================================

Update the GEN_SNAPSHOT_JSON path in the two diagnostic runners:
  - scripts/diagnose_v8_criterion_reconciliation.py
  - scripts/diagnose_v8_failure_modes_refreshed.py

to point at the new file:
  Improvement_Snapshots/reach_detection/v8.0.0_generalization_20video/
    metrics/reach_detection_scalars.json

Then re-run both diagnostic runners. They will produce v8.0.0-on-
generalization equivalents of the existing outputs. Compare against
the v7.2.0-on-generalization outputs (the existing 2026-05-11 numbers)
to characterise what shipping v8.0.0 actually changed on this corpus.

================================================================
WHAT THIS IS NOT
================================================================

This is NOT an experiment. There is no parameter tuning, no model
training, no algorithm change. It runs production v8.0.0 inference on
20 videos and scores the result. The pre-experiment checklist does not
apply because nothing in src/mousereach/ is being modified.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

DLC_AND_REACHES_DIR = ITER_DIR / "algo_outputs_current"
GT_DIR = ITER_DIR / "gt"

OUTPUT_SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_generalization_20video"
)


# Strict matching params (same as v8 production eval)
STRICT_START_TOL = 2
STRICT_SPAN_TOL_REL = 0.5
STRICT_SPAN_TOL_ABS = 5


# ---------------------------------------------------------------------------
# Discovery: every video that has BOTH a DLC h5 and a GT file
# ---------------------------------------------------------------------------

def discover_videos() -> List[str]:
    """Find video IDs with both a DLC h5 and a GT file."""
    h5_files = list(DLC_AND_REACHES_DIR.glob("*DLC_*.h5"))
    h5_video_ids = set()
    for f in h5_files:
        # Strip the DLC suffix to get the video_id
        name = f.stem
        idx = name.find("DLC_")
        if idx > 0:
            h5_video_ids.add(name[:idx])

    gt_video_ids = set()
    for f in GT_DIR.glob("*_unified_ground_truth.json"):
        gt_video_ids.add(f.stem.replace("_unified_ground_truth", ""))
    for f in GT_DIR.glob("*_reach_ground_truth.json"):
        gt_video_ids.add(f.stem.replace("_reach_ground_truth", ""))

    eligible = sorted(h5_video_ids & gt_video_ids)
    return eligible


# ---------------------------------------------------------------------------
# Per-video inference + scoring
# ---------------------------------------------------------------------------

def infer_and_score(video_id: str) -> Tuple[List[Tuple[int, int]],
                                            List[Tuple[int, int]],
                                            List[Dict[str, Any]]]:
    """Run v8.0.0 inference on one video, match against GT, return
    (algo_reaches_tuples, gt_reaches_tuples, match_records).
    """
    # Find DLC h5
    h5_candidates = list(DLC_AND_REACHES_DIR.glob(f"{video_id}DLC_*.h5"))
    if not h5_candidates:
        raise FileNotFoundError(f"No DLC h5 for {video_id}")
    h5_path = h5_candidates[0]

    # Run v8.0.0 inference
    dlc_df = load_dlc(h5_path)
    algo_tuples = detect_reaches_v8(dlc_df)
    # detect_reaches_v8 returns inclusive (start, end) tuples

    # Load GT
    gt_path = _find_gt_file(GT_DIR, video_id)
    if gt_path is None:
        raise FileNotFoundError(f"No GT for {video_id}")
    gt_reach_objs = _load_gt_reaches(gt_path)
    gt_tuples = [(r.start_frame, r.end_frame) for r in gt_reach_objs]

    # Build Reach objects for matching
    algo_reaches = [Reach(start_frame=s, end_frame=e, index=i)
                    for i, (s, e) in enumerate(algo_tuples)]
    # gt_reach_objs already has indices

    # Match strict
    results = match_reaches(
        algo_reaches, gt_reach_objs,
        strict=True,
        strict_start_tol=STRICT_START_TOL,
        strict_span_tol_rel=STRICT_SPAN_TOL_REL,
        strict_span_tol_abs=STRICT_SPAN_TOL_ABS,
    )

    # Serialize records in the diagnostic-runner-compatible schema
    records = []
    for r in results:
        if r.status == "matched":
            records.append({
                "video_id": video_id,
                "status": "matched",
                "gt_start": r.gt_start, "gt_end": r.gt_end,
                "algo_start": r.algo_start, "algo_end": r.algo_end,
                "start_delta": r.start_delta,
                "span_delta": (r.algo_end - r.algo_start + 1)
                              - (r.gt_end - r.gt_start + 1),
            })
        elif r.status == "fn":
            records.append({
                "video_id": video_id,
                "status": "fn",
                "gt_start": r.gt_start, "gt_end": r.gt_end,
                "algo_start": -1, "algo_end": -1,
                "start_delta": None, "span_delta": None,
            })
        elif r.status == "fp":
            records.append({
                "video_id": video_id,
                "status": "fp",
                "gt_start": -1, "gt_end": -1,
                "algo_start": r.algo_start, "algo_end": r.algo_end,
                "start_delta": None, "span_delta": None,
            })
    return algo_tuples, gt_tuples, records


def save_video_reaches(video_id: str,
                       algo_tuples: List[Tuple[int, int]],
                       out_dir: Path) -> Path:
    """Write a per-video reaches JSON.  Format is simpler than
    v7.2.0's nested-segments structure since we don't have segments
    knowledge inside v8 -- it operates whole-video."""
    out_path = out_dir / f"{video_id}_reaches.json"
    payload = {
        "detector_version": V8_VERSION,
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
            "merge_gap": 2,
            "min_span": 3,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(f"v8.0.0 INFERENCE on 2026-05-11 generalization corpus")
    print(f"  v8 module VERSION = {V8_VERSION}")
    print("=" * 70)
    print()

    print(f"DLC + reaches dir: {DLC_AND_REACHES_DIR}")
    print(f"GT dir:            {GT_DIR}")
    print()

    video_ids = discover_videos()
    print(f"Discovered {len(video_ids)} videos with both DLC h5 and GT:")
    for v in video_ids:
        print(f"  {v}")
    print()

    OUTPUT_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    algo_out_dir = OUTPUT_SNAPSHOT_DIR / "algo_outputs_v8.0.0"
    algo_out_dir.mkdir(exist_ok=True)
    metrics_dir = OUTPUT_SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # Manifest
    manifest = {
        "snapshot_name": "v8.0.0_generalization_20video",
        "phase": "reach_detection",
        "purpose": ("v8.0.0 inference + strict-criterion scoring on the "
                    "20-video generalization corpus that was misattributed "
                    "in the 2026-05-11 snapshot (those outputs were "
                    "actually v7.2.0)."),
        "created_at": datetime.now().isoformat(),
        "source_dlc_dir": str(DLC_AND_REACHES_DIR),
        "source_gt_dir": str(GT_DIR),
        "detector_version": V8_VERSION,
        "matching_criterion": {
            "strict": True,
            "start_tol": STRICT_START_TOL,
            "span_tol_rel": STRICT_SPAN_TOL_REL,
            "span_tol_abs": STRICT_SPAN_TOL_ABS,
        },
    }
    (OUTPUT_SNAPSHOT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    all_matches = []
    per_video_summary = {}
    total_tp = total_fp = total_fn = 0
    total_t0 = time.time()

    for i, video_id in enumerate(video_ids):
        t0 = time.time()
        print(f"[{i+1}/{len(video_ids)}] {video_id} ... ", end="", flush=True)
        try:
            algo_tuples, gt_tuples, records = infer_and_score(video_id)
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue

        save_video_reaches(video_id, algo_tuples, algo_out_dir)

        v_tp = sum(1 for r in records if r["status"] == "matched")
        v_fp = sum(1 for r in records if r["status"] == "fp")
        v_fn = sum(1 for r in records if r["status"] == "fn")
        per_video_summary[video_id] = {
            "n_algo": len(algo_tuples),
            "n_gt": len(gt_tuples),
            "n_tp": v_tp, "n_fp": v_fp, "n_fn": v_fn,
        }
        total_tp += v_tp; total_fp += v_fp; total_fn += v_fn
        all_matches.extend(records)

        dt = time.time() - t0
        print(f"n_algo={len(algo_tuples):4} n_gt={len(gt_tuples):4}  "
              f"TP={v_tp:3} FP={v_fp:3} FN={v_fn:3}  ({dt:.1f}s)")

    print()
    total_elapsed = time.time() - total_t0
    print("=" * 70)
    print(f"AGGREGATE -- v8.0.0 on generalization corpus")
    print("=" * 70)
    print(f"  Videos processed: {len(per_video_summary)}")
    print(f"  Total TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Elapsed: {total_elapsed:.1f}s "
          f"({total_elapsed/max(1, len(per_video_summary)):.1f}s/video)")
    print()
    print("Compare against the v7.2.0 numbers on the same corpus (from")
    print("the existing 2026-05-11 snapshot under STRICT criterion):")
    print("  v7.2.0: TP=3536  FP=581  FN=186")
    print()

    # Span-on-matched characterization (for sanity-check vs runner 1)
    matched_spans = [(r["algo_end"] - r["algo_start"] + 1)
                     - (r["gt_end"] - r["gt_start"] + 1)
                     for r in all_matches if r["status"] == "matched"]
    if matched_spans:
        arr = np.array(matched_spans)
        print(f"Signed span (algo - gt) on matched reaches:")
        print(f"  median={float(np.median(arr)):+.1f}f  "
              f"mean={float(np.mean(arr)):+.2f}f  "
              f"p10={float(np.percentile(arr, 10)):+.1f}  "
              f"p90={float(np.percentile(arr, 90)):+.1f}")
        print(f"  algo_longer={float((arr > 0).mean()):.1%}  "
              f"algo_shorter={float((arr < 0).mean()):.1%}  "
              f"exact={float((arr == 0).mean()):.1%}")
        print()

    # Write the diagnostic-runner-compatible scalars JSON
    scalars = {
        "n_videos": len(per_video_summary),
        "n_tp": total_tp,
        "n_fp": total_fp,
        "n_fn": total_fn,
        "detector_version": V8_VERSION,
        "matching_criterion": "strict",
        "matches": all_matches,
        "per_video": per_video_summary,
    }
    out_metrics = metrics_dir / "reach_detection_scalars.json"
    out_metrics.write_text(json.dumps(scalars, indent=2), encoding="utf-8")
    print(f"Wrote: {out_metrics}")
    print(f"Wrote: {algo_out_dir}/  ({len(per_video_summary)} per-video reaches JSONs)")
    print(f"Wrote: {OUTPUT_SNAPSHOT_DIR / 'manifest.json'}")
    print()
    print("NEXT STEPS:")
    print(f"  1. Update GEN_SNAPSHOT_JSON in:")
    print(f"     - scripts/diagnose_v8_criterion_reconciliation.py")
    print(f"     - scripts/diagnose_v8_failure_modes_refreshed.py")
    print(f"     to: {out_metrics}")
    print(f"  2. Re-run both diagnostic runners.")
    print(f"  3. Compare v8.0.0 vs v7.2.0 generalization numbers.")


if __name__ == "__main__":
    main()
