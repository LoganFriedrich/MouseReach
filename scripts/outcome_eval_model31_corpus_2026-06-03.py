"""
Outcome cascade evaluation on the 20 model-3.1 corpus videos.

Runs the production reach detector (v8.0.4) + cumulative-best outcome
cascade (v6.0.3 = v6.0.1 vetoes + Fix A Stage 32 + Fix B Stage 31) on
every corpus video that has a confirmed DLC model-3.1 h5 available.

WHY THIS EXISTS
    Model 3.1 DLC was only ever run on the 20 exhaustive-GT videos of the
    47-video outcome corpus (16 of the 37 train-pool + 4 of the 10 frozen
    test-holdout). This script scores the cascade on all 20 so we can
    inventory current errors. Results are split train (in-sample, the
    cascade stages were designed against these via LOOCV trust
    calibration) vs holdout (out-of-sample) so in-sample fit is never
    mistaken for generalization.

INPUTS (verified 2026-06-03)
    - DLC h5: Y: canonical
      validation_runs\\DLC_2026_03_27\\Processing\\updated dlc model 3.1
      (byte-identical to A:\\MouseReach_Pipeline\\All DLC Models\\DLC Model v3.1)
    - GT:     canonical reach corpus folder
      validation_runs\\DLC_2026_03_27\\gt  (reaches + outcomes now read
      from one place). Outcome blocks were reconciled into this folder on
      2026-06-08 from the former walkthrough corpus gt/; the walkthrough
      copy is preserved as the pre-reconciliation original.

NOT AN EXPERIMENT
    No algorithm change. This is an evaluation/error-inventory run.
    Cascade logic is imported verbatim from the accepted Fix B runner
    (outcome_fix_b_retrieved_rescue_2026-06-02.py); no src/ edits.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

# Import the accepted Fix B runner as a module to reuse the EXACT v6.0.3
# cascade build + helpers (build_stages_with_fix_b, run_cascade_on_segments,
# save_reaches_segmented) and the package re-exports it already pulled in
# (detect_reaches_v8, load_dlc_h5, compute_outcome_metrics, SegmentInput).
_fixb_path = SCRIPTS_DIR / "outcome_fix_b_retrieved_rescue_2026-06-02.py"
_spec = importlib.util.spec_from_file_location("_outcome_fix_b", _fixb_path)
fixb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixb)

detect_reaches_v8 = fixb.detect_reaches_v8
load_dlc_h5 = fixb.load_dlc_h5
compute_outcome_metrics = fixb.compute_outcome_metrics
SegmentInput = fixb.SegmentInput
build_stages_with_fix_b = fixb.build_stages_with_fix_b
run_cascade_on_segments = fixb.run_cascade_on_segments
save_reaches_segmented = fixb.save_reaches_segmented


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DLC_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs"
               r"\DLC_2026_03_27\Processing\updated dlc model 3.1")
# Canonical GT: the reach corpus folder (outcomes reconciled in 2026-06-08).
GT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs"
              r"\DLC_2026_03_27\gt")
SNAPSHOT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
                    r"\Improvement_Snapshots\outcome"
                    r"\v6.0.3_eval_model31_corpus_2026-06-03")

# The 20 model-3.1 video IDs, split by corpus role (from
# _corpus/2026-04-30_restart_inventory/cv_folds.json).
TRAIN_IDS = [
    "20250624_CNT0107_P3", "20250627_CNT0105_P1", "20250630_CNT0104_P3",
    "20250701_CNT0111_P1", "20250710_CNT0215_P4", "20250812_CNT0301_P3",
    "20250813_CNT0314_P4", "20250820_CNT0103_P3", "20250821_CNT0110_P4",
    "20250909_CNT0209_P4", "20251009_CNT0310_P2", "20251010_CNT0308_P2",
    "20251022_CNT0413_P4", "20251028_CNT0404_P4", "20251030_CNT0403_P1",
    "20251031_CNT0407_P1",
]
HOLDOUT_IDS = [
    "20250626_CNT0102_P4", "20250708_CNT0210_P3",
    "20250811_CNT0303_P4", "20251024_CNT0402_P4",
]
ROLE = {v: "train" for v in TRAIN_IDS}
ROLE.update({v: "holdout" for v in HOLDOUT_IDS})
VIDEO_IDS = TRAIN_IDS + HOLDOUT_IDS


def find_dlc(vid):
    m = sorted(DLC_DIR.glob(f"{vid}DLC_*.h5"))
    if not m:
        raise FileNotFoundError(f"No model-3.1 DLC for {vid} in {DLC_DIR}")
    return m[0]


def load_gt_segments(vid):
    gt = json.loads((GT_DIR / f"{vid}_unified_ground_truth.json").read_text(encoding="utf-8"))
    bs = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
    if len(bs) < 2:
        return []
    return [(bs[i], bs[i + 1] - 1) for i in range(len(bs) - 1)]


def main():
    t0 = time.time()
    print("=" * 70)
    print("Outcome cascade eval on 20 model-3.1 corpus videos")
    print(f"  reach detector : v8.0.4 production")
    print(f"  outcome cascade: v6.0.3 (Fix B cumulative best)")
    print(f"  DLC source     : {DLC_DIR}")
    print(f"  GT source      : {GT_DIR}")
    print(f"  snapshot       : {SNAPSHOT_DIR}")
    print("=" * 70)

    algo_dir = SNAPSHOT_DIR / "algo_outputs"
    metrics_dir = SNAPSHOT_DIR / "metrics"
    algo_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Verify inputs present.
    runnable = []
    for vid in VIDEO_IDS:
        has_dlc = bool(sorted(DLC_DIR.glob(f"{vid}DLC_*.h5")))
        has_gt = (GT_DIR / f"{vid}_unified_ground_truth.json").exists()
        if has_dlc and has_gt:
            runnable.append(vid)
        else:
            print(f"[SKIP] {vid} (dlc={has_dlc} gt={has_gt})")
    print(f"Runnable: {len(runnable)}/{len(VIDEO_IDS)}")

    stages = build_stages_with_fix_b(video_dir=None)

    for i, vid in enumerate(runnable, 1):
        t_vid = time.time()
        print(f"[{i}/{len(runnable)}] {vid} ({ROLE[vid]})", flush=True)
        dlc = load_dlc_h5(find_dlc(vid))
        segments = load_gt_segments(vid)
        reaches = detect_reaches_v8(dlc)
        print(f"   {len(segments)} segments, {len(reaches)} reaches", flush=True)
        save_reaches_segmented(vid, reaches, segments, algo_dir / f"{vid}_reaches.json")
        seg_inputs = []
        for si_idx, (s, e) in enumerate(segments):
            sn = si_idx + 1
            seg_r = [(r0, r1) for r0, r1 in reaches if s <= r0 <= e]
            seg_inputs.append(SegmentInput(video_id=vid, segment_num=sn,
                                           seg_start=s, seg_end=e,
                                           dlc_df=dlc, reach_windows=seg_r))
        outs = run_cascade_on_segments(seg_inputs, stages)
        if vid in outs:
            (algo_dir / f"{vid}_pellet_outcomes.json").write_text(
                json.dumps({"video_id": vid, "detector": "v6_cascade_fix_b",
                            "detector_version": "6.0.3_fix_b",
                            "reach_detector_version": "v8.0.4_production",
                            "segments": outs[vid]}, indent=2), encoding="utf-8")
        print(f"   ({time.time() - t_vid:.1f}s)", flush=True)

    print("\nScoring (compute_outcome_metrics)...", flush=True)
    scalars = compute_outcome_metrics(gt_dir=GT_DIR, algo_dir=algo_dir,
                                      output_dir=metrics_dir,
                                      video_ids=runnable, reaches_dir=algo_dir)

    # Persist the train/holdout split + runnable list for downstream analysis.
    (metrics_dir / "role_split.json").write_text(json.dumps({
        "train_ids": [v for v in runnable if ROLE[v] == "train"],
        "holdout_ids": [v for v in runnable if ROLE[v] == "holdout"],
    }, indent=2), encoding="utf-8")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Snapshot complete: {SNAPSHOT_DIR}")
    print("Deliverables in metrics/: scalars.json + per-segment table.")


if __name__ == "__main__":
    main()
