"""
Phase E Stage 1 trust calibration.

Runs Stage 1 across the train_pool GT corpus, measures empirical trust
on commits per the user's spec:
    trust = (committed_class matches GT) AND (outcome_known_frame
            within +/-3 frames of GT)

Reports trust at several commit-threshold settings so we can pick the
right operating point for Stage 1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.outcomes.v6_cascade.stage_1_pellet_stable_untouched import (
    Stage1PelletStableUntouched)
from mousereach.outcomes.v6_cascade.stage_base import SegmentInput
from mousereach.outcomes.v6_cascade.trust_calibrator import (
    calibrate_stage, report_calibration)
from mousereach.reach.v8.features import load_dlc_h5


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
QUARANTINE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations"
    r"\2026-04-28_outcome_v4.0.0_dev_walkthrough"
)
DLC_DIR = QUARANTINE / "dlc"
GT_DIR = QUARANTINE / "gt"
ALGO_DIR = QUARANTINE / "algo_outputs"


def find_dlc(video_id: str) -> Path:
    return next(DLC_DIR.glob(f"{video_id}DLC_*.h5"))


def load_segment_bounds(video_id: str) -> Dict[int, tuple]:
    seg_data = json.loads(
        (ALGO_DIR / f"{video_id}_segments.json").read_text(encoding="utf-8"))
    boundaries = seg_data.get("boundaries", []) or []
    return {i + 1: (int(boundaries[i]), int(boundaries[i + 1]) - 1)
            for i in range(len(boundaries) - 1)}


def load_gt(video_id: str) -> dict:
    return json.loads(
        (GT_DIR / f"{video_id}_unified_ground_truth.json").read_text(encoding="utf-8"))


def collapse_outcome(o):
    return "displaced_sa" if o == "displaced_outside" else o


def gt_reaches_for_segment(gt: dict, segment_num: int) -> List[tuple]:
    out = []
    for r in gt.get("reaches", {}).get("reaches", []) or []:
        if r.get("segment_num") == segment_num:
            s = r.get("start_frame")
            e = r.get("end_frame")
            if s is not None and e is not None:
                out.append((int(s), int(e)))
    return out


def build_seg_inputs_and_gt():
    folds = json.loads((CORPUS_DIR / "cv_folds.json").read_text())
    train_pool_ids = folds["train_pool"]["video_ids"]
    seg_inputs = []
    gt_lookup = {}
    for vid in train_pool_ids:
        dlc = load_dlc_h5(find_dlc(vid))
        seg_bounds = load_segment_bounds(vid)
        gt = load_gt(vid)
        gt_outcomes = {s["segment_num"]: s
                       for s in gt.get("outcomes", {}).get("segments", []) or []}
        for sn, (s_start, s_end) in seg_bounds.items():
            seg_input = SegmentInput(
                video_id=vid, segment_num=sn,
                seg_start=s_start, seg_end=s_end,
                dlc_df=dlc,
                reach_windows=gt_reaches_for_segment(gt, sn),
            )
            seg_inputs.append(seg_input)
            gt_seg = gt_outcomes.get(sn, {})
            gt_lookup[(vid, sn)] = {
                "gt_outcome": collapse_outcome(gt_seg.get("outcome")),
                "gt_outcome_known_frame": gt_seg.get("outcome_known_frame"),
                "gt_interaction_frame": gt_seg.get("interaction_frame"),
            }
    return seg_inputs, gt_lookup


def main():
    print("=" * 70)
    print("PHASE E STAGE 1 TRUST CALIBRATION")
    print("=" * 70)
    print()

    print("Loading segments + GT ...", flush=True)
    seg_inputs, gt_lookup = build_seg_inputs_and_gt()
    print(f"  Loaded {len(seg_inputs)} segments")
    print()

    settings = [
        ("default:    frac>=0.95, dist<=1.0r", 0.95, 1.0),
        ("loose:      frac>=0.95, dist<=1.5r", 0.95, 1.5),
        ("loosest:    frac>=0.90, dist<=2.0r", 0.90, 2.0),
    ]

    # Cascade OKF emit = seg_end - 10 (new semantics, see
    # feature_philosophy_event_anchored_walking.md). GT OKF empirically
    # at seg_end - 4 to seg_end + 7 across the corpus. So the natural
    # cascade-vs-GT delta for untouched is roughly -16 to -3. Tolerance
    # of +/-15 captures the bulk; we sweep a few values.
    out_records = []
    for label, cf, cd in settings:
        stage = Stage1PelletStableUntouched(
            commit_frac=cf, commit_distance_radii=cd)
        for okf_tol in [3, 7, 15]:
            cal = calibrate_stage(
                stage=stage, seg_inputs=seg_inputs, gt_lookup=gt_lookup,
                okf_tolerance=okf_tol, ifr_tolerance=okf_tol,
            )
            print(f"\n=== {label} | okf_tol=+/-{okf_tol} ===")
            print(report_calibration(cal))
            out_records.append({
                "label": label, "commit_frac": cf,
                "commit_distance_radii": cd,
                "okf_tolerance": okf_tol,
                "trust_per_class": cal.trust_per_class,
                "n_committed_per_class": cal.n_committed_per_class,
                "n_correct_per_class": cal.n_correct_per_class,
                "yield_per_class": cal.yield_per_class,
                "n_continued": cal.n_continued,
                "residual_gt": cal.residual_gt_distribution,
            })

    out_dir = Path(
        r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
        r"\outcome\v6.0.0_dev_stage1_trust")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trust_calibration.json").write_text(
        json.dumps(out_records, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_dir / 'trust_calibration.json'}")


if __name__ == "__main__":
    main()
