"""
Outcome E2E test: v6.0.0 cascade with v8.0.4 algo reaches on the 2026-05-11
generalization corpus.

Purpose
-------
Step-2 baseline pair from the recalibration playbook. The 2026-05-11
generalization baseline used **v7.2.0 reaches + v6.0.0 cascade** and
showed 240/400 correct, 149/400 triaged. v8.0.4 reaches are now in
production. This runner swaps ONLY the reach detector input (v7.2.0 ->
v8.0.4) on the SAME 20-video corpus, holding the outcome cascade
constant, so the delta isolates the reach-detector contribution to
outcome accuracy.

================================================================
Pre-experiment checklist (per memory pre_experiment_checklist.md)
================================================================

1. Cumulative-stacking check (verified 2026-06-01)
   - Reach detector: v8.0.4 production. Includes BSW b=1/w=0.8
     (shipped 2026-05-06), leading-trim sustained-lk T=0.60 N=3
     (v8.0.2, shipped 2026-05-21), apex-split at trough prom=0.12
     depth_min=0.5 peak2_rel_max=0.85 (v8.0.3, shipped 2026-05-22),
     trailing-trim sustained-lk T=0.60 N=3 (v8.0.4, shipped 2026-05-26).
     All four integrated into the production model bundled at
     src/mousereach/reach/v8/models/v8.0.0_bsw_w0.8.joblib + postprocess
     defaults in src/mousereach/reach/v8/__init__.py.
   - Outcome cascade: v6.0.0, unchanged since 2026-05-04. No pending
     integrations.
   - Verification: read snapshot Improvement_Snapshots/reach_detection/
     v8.0.4_progress_report_2026-05-27/RESULTS.md (Calibration LOOCV
     filtered TP=2312, FP=9, FN=34; Holdout 19 filtered TP=3655, FP=12,
     FN=27). Read outcome AGENTS.md and v6_cascade detector module
     header.
   - This experiment is a baseline-pair comparison (swap upstream),
     NOT a tuning experiment, so cumulative-best stacking applies to
     the reach detector input only.

2. Existing-code-modification check: NO. This runner imports from
   src/mousereach/ but does not modify any module. Imports used:
     - mousereach.reach.v8.detect_reaches_v8 (public inference API)
     - mousereach.reach.v8.features.load_dlc_h5 (public DLC loader)
     - mousereach.outcomes.v6_cascade.detector.detect_outcomes_v6_cascade
       (public cascade API)
     - mousereach.improvement.outcome.metrics.compute_outcome_metrics
       (canonical scorer)
     - mousereach.improvement.outcome._run_notebooks (canonical figures
       -- run separately after this script completes)

3. Unverified hypotheses (written upfront, not post-hoc)
   - H1: v8.0.4 reaches will produce fewer triages than v7.2.0 reaches
     on this corpus. Mechanism: better reach boundaries -> fewer
     under-determined cascade decisions.
   - H2: Retrieved class commits will rise from 2/67 baseline.
     Mechanism: retrieval-specific reaches that v7.2.0 missed (or
     boundary-fragmented) may now resolve as clean single reaches under
     v8.0.4's apex-split.
   - H3: No regression in untouched class (currently 100/104 correct).
     Mechanism: untouched segments depend less on reach quality
     because the cascade's untouched-commit logic is paw-position
     based, not reach-based.
   - Verification method: this experiment.

4. FN-direction-reporting (planned RESULTS.md structure)
   Lead line: "Triage count and FN-equivalents went FROM <baseline> TO
   <new>: <direction> by <magnitude> (vs 2026-05-11 baseline with v7.2.0
   reaches). Total correct commits went FROM <baseline> TO <new>:
   <direction>." Two deltas:
     (a) vs cumulative-best (2026-05-11 baseline, same cascade, v7.2.0
         reaches): the operational comparison for "did the reach swap
         help"
     (b) vs pure-baseline (the GT-input upper bound on holdout
         corpus, where v6.0.0 + GT reaches achieves 85.5% raw on the
         10-video test_holdout): the total-progress reference for
         "how close are we to the cascade's ceiling"
   TP/FP/FN counts FIRST, accuracy/precision/recall LAST.

5. Framework-not-adhoc
   - Output snapshot dir: Improvement_Snapshots/outcome/
     v6.0.0_e2e_v804_reaches_2026-06-01/
   - Subdirs: algo_outputs/, metrics/, figures/, RESULTS.md
   - Schema: matches compute_outcome_metrics canonical output
     (scalars.json + per_segment.csv + per_video.csv)
   - Figures via canonical mousereach.improvement.outcome._run_notebooks
     (run_sankey, run_summary_table). NOT hand-rolled.

6. Branch + tag (TO DO BEFORE RUNNING THIS SCRIPT)
   - Pre-experiment tag: outcome-pre-v804-reach-e2e-2026-06-01
   - Feature branch: feature/outcome-v804-reach-e2e
   - Both still pending; Logan to authorize.

7. Decision rule
   - ACCEPT (declare "v8.0.4 reach detector helped outcomes"): IF
       total correct commits >= 240 (baseline) + 5, OR
       retrieved-class correct commits >= 5 (baseline=2, looking for
       statistically meaningful uptick), OR
       triage count drops below 130 (from baseline 149)
     while no class regresses by more than 3 events.
   - REJECT (declare "no measurable outcome benefit from reach
     upgrade"): IF
       total correct commits drops below baseline 240, OR
       any class regresses by more than 3 events with no
       compensating gain elsewhere.
   - NEUTRAL (declare "outcome cascade was insensitive to reach quality
     in this regime"): IF
       all metrics fall within +/-3 events of baseline.
   - Comparison baseline: 2026-05-11 generalization snapshot
     (reach 7.2.0 + outcome 6.0.0 + assignment 1.0.0) per manifest.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Allow direct script execution by extending sys.path to find mousereach
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from mousereach.reach.v8 import detect_reaches_v8
from mousereach.reach.v8.features import load_dlc_h5
from mousereach.outcomes.v6_cascade.detector import detect_outcomes_v6_cascade
from mousereach.improvement.outcome.metrics import compute_outcome_metrics


# ---------------------------------------------------------------------------
# Paths (verified against actual corpus 2026-06-01)
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11"
)
DLC_DIR = CORPUS_DIR / "algo_outputs_current"  # has the 20 video DLC h5s
GT_DIR = CORPUS_DIR / "gt"  # has the 20 unified_ground_truth.json files

SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\outcome\v6.0.0_e2e_v804_reaches_2026-06-01"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_dlc(video_id: str) -> Path:
    """Return DLC h5 for a video_id (single match expected)."""
    matches = sorted(DLC_DIR.glob(f"{video_id}DLC_*.h5"))
    if not matches:
        raise FileNotFoundError(f"No DLC h5 for {video_id} in {DLC_DIR}")
    if len(matches) > 1:
        print(f"  WARNING: multiple DLC h5s for {video_id}; using {matches[0].name}")
    return matches[0]


def load_gt_segments(video_id: str) -> List[Tuple[int, int]]:
    """Parse segment boundaries from the unified GT file.

    Returns a list of (start_frame, end_frame) inclusive tuples, one per
    segment. The GT stores boundary frames; segments are consecutive
    intervals between boundaries.
    """
    gt = json.loads(
        (GT_DIR / f"{video_id}_unified_ground_truth.json").read_text(encoding="utf-8")
    )
    boundary_frames = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
    if len(boundary_frames) < 2:
        return []
    segments = []
    for i in range(len(boundary_frames) - 1):
        s = boundary_frames[i]
        e = boundary_frames[i + 1] - 1  # end is one frame before next boundary
        segments.append((s, e))
    return segments


def save_reaches_segmented(video_id: str, reaches: List[Tuple[int, int]],
                            segments: List[Tuple[int, int]],
                            out_path: Path) -> None:
    """Write reaches in the canonical segmented schema for outcome scoring.

    The compute_outcome_metrics scorer reads reaches.json with the shape:
      {"video_id": ..., "detector": ..., "segments": [
          {"segment_num": int, "reaches": [{"start_frame": ..., "end_frame": ...,
            "reach_id": ..., "reach_num": ...}, ...]},
          ...
      ]}
    Reach ids are 1-indexed globally; reach_nums are 1-indexed within segment.
    """
    video_segments_data = []
    global_reach_id = 0
    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        seg_num = seg_idx + 1
        seg_reaches = []
        reach_num = 0
        for r_start, r_end in reaches:
            if seg_start <= r_start <= seg_end:
                reach_num += 1
                global_reach_id += 1
                seg_reaches.append({
                    "reach_id": global_reach_id,
                    "reach_num": reach_num,
                    "start_frame": int(r_start),
                    "end_frame": int(r_end),
                    "duration_frames": int(r_end - r_start + 1),
                })
        video_segments_data.append({
            "segment_num": seg_num,
            "start_frame": int(seg_start),
            "end_frame": int(seg_end),
            "n_reaches": len(seg_reaches),
            "reaches": seg_reaches,
        })
    data = {
        "video_id": video_id,
        "detector": "v8.0.4_production",
        "n_segments": len(segments),
        "segments": video_segments_data,
    }
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"Outcome E2E with v8.0.4 reaches on 2026-05-11 generalization corpus")
    print(f"Snapshot dir: {SNAPSHOT_DIR}")

    algo_outputs_dir = SNAPSHOT_DIR / "algo_outputs"
    metrics_dir = SNAPSHOT_DIR / "metrics"
    algo_outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Discover videos from GT dir (canonical pattern)
    video_ids = sorted({p.stem.replace("_unified_ground_truth", "")
                         for p in GT_DIR.glob("*_unified_ground_truth.json")})
    print(f"Found {len(video_ids)} videos in corpus")
    if len(video_ids) != 20:
        print(f"  WARNING: expected 20 videos, found {len(video_ids)}")

    # Per video: reach detection -> outcome cascade -> save both JSONs
    for i, vid in enumerate(video_ids, 1):
        print(f"[{i}/{len(video_ids)}] {vid}", flush=True)
        t_vid = time.time()

        # Load DLC
        dlc_path = find_dlc(vid)
        dlc_df = load_dlc_h5(dlc_path)

        # Load segments from GT (we use GT segments since segmentation is upstream
        # of the reach -> outcome pipeline; this experiment isolates the reach
        # detector's effect on outcomes, not the segmenter)
        segments = load_gt_segments(vid)

        # Run v8.0.4 reach detection
        algo_reaches = detect_reaches_v8(dlc_df)
        print(f"   detected {len(algo_reaches)} reaches", flush=True)

        # Save reaches.json in canonical segmented schema
        reaches_path = algo_outputs_dir / f"{vid}_reaches.json"
        save_reaches_segmented(vid, algo_reaches, segments, reaches_path)

        # Run v6 cascade on those reaches
        outcomes = detect_outcomes_v6_cascade(
            dlc_df=dlc_df,
            segments=segments,
            reaches=algo_reaches,
            video_id=vid,
        )
        # Stamp with the reach version we used so downstream is unambiguous
        outcomes["reach_detector_version"] = "v8.0.4_production"

        outcomes_path = algo_outputs_dir / f"{vid}_pellet_outcomes.json"
        outcomes_path.write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
        print(f"   ({time.time() - t_vid:.1f}s)", flush=True)

    # Score using canonical metrics
    print(f"\nScoring with compute_outcome_metrics...", flush=True)
    scalars = compute_outcome_metrics(
        gt_dir=GT_DIR,
        algo_dir=algo_outputs_dir,
        output_dir=metrics_dir,
        video_ids=video_ids,
        reaches_dir=algo_outputs_dir,
    )

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"\nSnapshot complete at: {SNAPSHOT_DIR}")
    print(f"\nNext steps (manual):")
    print(f"  1. Inspect metrics/scalars.json against baseline at:")
    print(f"     Improvement_Snapshots/outcome/generalization_test_2026-05-11/metrics/outcome_scalars.json")
    print(f"  2. Run canonical figures:")
    print(f"     from mousereach.improvement.outcome._run_notebooks import run_sankey, run_summary_table")
    print(f"     run_sankey(SNAPSHOT_DIR); run_summary_table(SNAPSHOT_DIR)")
    print(f"  3. Author RESULTS.md per the checklist (FN-first, two deltas).")


if __name__ == "__main__":
    main()
