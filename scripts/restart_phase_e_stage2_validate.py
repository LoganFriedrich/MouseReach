"""
Phase E: validate Stage 2 in cascade with Stage 1.

Run Stage 1 first; for each segment that Stage 1 deferred, run Stage 2;
compute per-stage trust + cumulative coverage.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.outcomes.v6_cascade.stage_1_pellet_stable_untouched import (
    Stage1PelletStableUntouched)
from mousereach.outcomes.v6_cascade.stage_2_paw_never_in_pellet_area import (
    Stage2PawNeverInPelletArea)
from mousereach.outcomes.v6_cascade.stage_base import SegmentInput
from mousereach.outcomes.v6_cascade.trust_calibrator import (
    calibrate_stage)
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


def gt_reaches_for_segment(gt: dict, segment_num: int) -> List[tuple]:
    out = []
    for r in gt.get("reaches", {}).get("reaches", []) or []:
        if r.get("segment_num") == segment_num:
            s = r.get("start_frame")
            e = r.get("end_frame")
            if s is not None and e is not None:
                out.append((int(s), int(e)))
    return out


def collapse(o):
    return "displaced_sa" if o == "displaced_outside" else o


def build_seg_inputs_and_gt():
    folds = json.loads((CORPUS_DIR / "cv_folds.json").read_text())
    train_pool_ids = folds["train_pool"]["video_ids"]
    seg_inputs = []
    gt_lookup = {}
    for vid in train_pool_ids:
        dlc = load_dlc_h5(find_dlc(vid))
        seg_bounds = load_segment_bounds(vid)
        gt = load_gt(vid)
        gt_outs = {s["segment_num"]: s
                   for s in gt.get("outcomes", {}).get("segments", []) or []}
        for sn, (s_start, s_end) in seg_bounds.items():
            seg_inputs.append(SegmentInput(
                video_id=vid, segment_num=sn,
                seg_start=s_start, seg_end=s_end,
                dlc_df=dlc,
                reach_windows=gt_reaches_for_segment(gt, sn),
            ))
            seg = gt_outs.get(sn, {})
            gt_lookup[(vid, sn)] = {
                "gt_outcome": collapse(seg.get("outcome")),
                "gt_outcome_known_frame": seg.get("outcome_known_frame"),
                "gt_interaction_frame": seg.get("interaction_frame"),
            }
    return seg_inputs, gt_lookup


def main():
    print("=" * 70)
    print("PHASE E STAGE 2 VALIDATION (in cascade with Stage 1)")
    print("=" * 70)
    print()

    print("Loading segments + GT ...", flush=True)
    seg_inputs, gt_lookup = build_seg_inputs_and_gt()
    print(f"  Loaded {len(seg_inputs)} segments")
    print()

    # --- Stage 1 ---
    stage1 = Stage1PelletStableUntouched(
        commit_frac=0.95, commit_distance_radii=1.5)
    print("Running Stage 1 ...", flush=True)
    cal1 = calibrate_stage(
        stage=stage1, seg_inputs=seg_inputs, gt_lookup=gt_lookup,
        okf_tolerance=3, ifr_tolerance=3, transition_zone_half=5)
    s1_committed = {(c.video_id, c.segment_num)
                    for c in cal1.cases if c.decision == "commit"}
    print(f"  Stage 1: {len(s1_committed)} committed, "
          f"{len(seg_inputs) - len(s1_committed)} deferred")
    print(f"  Stage 1 trust per class: {cal1.trust_per_class}")
    print()

    # --- Stage 2 (only on Stage 1 deferrals) ---
    stage2 = Stage2PawNeverInPelletArea()
    s2_inputs = [s for s in seg_inputs
                 if (s.video_id, s.segment_num) not in s1_committed]
    print(f"Running Stage 2 on {len(s2_inputs)} Stage-1-deferred segments ...",
          flush=True)
    cal2 = calibrate_stage(
        stage=stage2, seg_inputs=s2_inputs, gt_lookup=gt_lookup,
        okf_tolerance=3, ifr_tolerance=3, transition_zone_half=5)
    s2_committed = {(c.video_id, c.segment_num)
                    for c in cal2.cases if c.decision == "commit"}
    print(f"  Stage 2: {len(s2_committed)} committed, "
          f"{len(s2_inputs) - len(s2_committed)} deferred")
    print(f"  Stage 2 trust per class: {cal2.trust_per_class}")
    print()

    # --- Per-class breakdown for Stage 2 commits ---
    s2_commit_by_gt = defaultdict(int)
    s2_commit_by_gt_match = defaultdict(int)
    for c in cal2.cases:
        if c.decision != "commit":
            continue
        s2_commit_by_gt[c.gt_class] += 1
        if c.class_match and c.okf_within_tol:
            s2_commit_by_gt_match[c.gt_class] += 1

    print(f"Stage 2 commits by GT class (precision check):")
    for cls, n in s2_commit_by_gt.items():
        ok = s2_commit_by_gt_match.get(cls, 0)
        print(f"  {cls:>22s}: {ok}/{n} pass trust ({100*ok/n:.1f}%)")
    print()

    # Cumulative residual GT distribution
    s2_deferred = {(c.video_id, c.segment_num)
                   for c in cal2.cases if c.decision == "continue"}
    residual = defaultdict(int)
    for (vid, sn) in s2_deferred:
        gt = gt_lookup.get((vid, sn), {})
        residual[gt.get("gt_outcome")] += 1
    print(f"Cumulative residual after Stage 1 + Stage 2 "
          f"({len(s2_deferred)} segments):")
    for cls, n in sorted(residual.items(), key=lambda x: -x[1]):
        print(f"  {cls:>22s}: {n}")
    print()

    # Cumulative coverage of GT untouched
    n_untouched_gt = sum(
        1 for (vid, sn), g in gt_lookup.items()
        if g.get("gt_outcome") == "untouched")
    s1_untouched_correct = sum(
        1 for c in cal1.cases
        if c.decision == "commit" and c.committed_class == "untouched"
        and c.class_match and c.okf_within_tol)
    s2_untouched_correct = sum(
        1 for c in cal2.cases
        if c.decision == "commit" and c.committed_class == "untouched"
        and c.class_match and c.okf_within_tol)
    cumulative = s1_untouched_correct + s2_untouched_correct
    print(f"Cumulative untouched yield: "
          f"S1={s1_untouched_correct} + S2={s2_untouched_correct} "
          f"= {cumulative} / {n_untouched_gt} GT untouched "
          f"({100*cumulative/n_untouched_gt:.1f}%)")
    print()

    # Save
    out_dir = Path(
        r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
        r"\outcome\v6.0.0_dev_stage2_validate")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage2_results.json").write_text(json.dumps({
        "stage1_n_committed": len(s1_committed),
        "stage2_n_committed": len(s2_committed),
        "stage2_commit_by_gt": dict(s2_commit_by_gt),
        "stage2_commit_by_gt_pass_trust": dict(s2_commit_by_gt_match),
        "cumulative_residual": dict(residual),
        "cumulative_untouched_yield": cumulative,
        "n_untouched_gt": n_untouched_gt,
    }, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir / 'stage2_results.json'}")


if __name__ == "__main__":
    main()
