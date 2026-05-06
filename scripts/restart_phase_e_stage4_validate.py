"""
Phase E: validate Stage 4 in cascade with Stages 1, 2, 3.

Runs Stages 1+2+3+4 in order; reports per-stage trust, per-class
commit breakdown, cumulative residual, and yield on each target class.
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
from mousereach.outcomes.v6_cascade.stage_3_pellet_returns_to_pillar import (
    Stage3PelletReturnsToPillar)
from mousereach.outcomes.v6_cascade.stage_4_pellet_displaced_to_sa import (
    Stage4PelletDisplacedToSA)
from mousereach.outcomes.v6_cascade.stage_base import SegmentInput
from mousereach.outcomes.v6_cascade.trust_calibrator import calibrate_stage
from mousereach.reach.v8.features import load_dlc_h5


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
QUARANTINE = Path(
    r"Y:\2_Connectome\Validation_Runs\DLC_2026_03_27\iterations"
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


def run_stage(stage, inputs, gt_lookup, label):
    cal = calibrate_stage(
        stage=stage, seg_inputs=inputs, gt_lookup=gt_lookup,
        okf_tolerance=3, ifr_tolerance=3, transition_zone_half=5)
    committed = {(c.video_id, c.segment_num)
                 for c in cal.cases if c.decision == "commit"}
    print(f"  {label}: {len(committed)} committed, "
          f"{len(inputs) - len(committed)} deferred")
    print(f"  {label} trust per class: {cal.trust_per_class}")
    return cal, committed


def commit_breakdown(cal, label):
    by_gt = defaultdict(int)
    by_gt_pass = defaultdict(int)
    for c in cal.cases:
        if c.decision != "commit":
            continue
        by_gt[c.gt_class] += 1
        if c.class_match and c.okf_within_tol:
            if cal.cases[0].committed_class == "displaced_sa":
                # For displaced_sa, also check ifr
                if c.ifr_within_tol:
                    by_gt_pass[c.gt_class] += 1
            else:
                by_gt_pass[c.gt_class] += 1
    print(f"  {label} commits by GT class:")
    for cls, n in by_gt.items():
        ok = by_gt_pass.get(cls, 0)
        print(f"    {cls:>22s}: {ok}/{n} pass trust ({100*ok/n:.1f}%)")
    return by_gt, by_gt_pass


def main():
    print("=" * 70)
    print("PHASE E STAGE 4 VALIDATION (in cascade with Stages 1+2+3)")
    print("=" * 70)
    print()

    print("Loading segments + GT ...", flush=True)
    seg_inputs, gt_lookup = build_seg_inputs_and_gt()
    print(f"  Loaded {len(seg_inputs)} segments")
    print()

    # --- Stage 1 ---
    print("Running Stage 1 ...", flush=True)
    stage1 = Stage1PelletStableUntouched(commit_frac=0.95, commit_distance_radii=1.5)
    cal1, s1_committed = run_stage(stage1, seg_inputs, gt_lookup, "Stage 1")
    print()

    # --- Stage 2 ---
    s2_inputs = [s for s in seg_inputs
                 if (s.video_id, s.segment_num) not in s1_committed]
    print(f"Running Stage 2 on {len(s2_inputs)} Stage-1-deferred ...", flush=True)
    stage2 = Stage2PawNeverInPelletArea()
    cal2, s2_committed = run_stage(stage2, s2_inputs, gt_lookup, "Stage 2")
    print()

    # --- Stage 3 ---
    s3_inputs = [s for s in s2_inputs
                 if (s.video_id, s.segment_num) not in s2_committed]
    print(f"Running Stage 3 on {len(s3_inputs)} Stage-2-deferred ...", flush=True)
    stage3 = Stage3PelletReturnsToPillar()
    cal3, s3_committed = run_stage(stage3, s3_inputs, gt_lookup, "Stage 3")
    print()

    # --- Stage 4 ---
    s4_inputs = [s for s in s3_inputs
                 if (s.video_id, s.segment_num) not in s3_committed]
    print(f"Running Stage 4 on {len(s4_inputs)} Stage-3-deferred ...", flush=True)
    stage4 = Stage4PelletDisplacedToSA()
    cal4, s4_committed = run_stage(stage4, s4_inputs, gt_lookup, "Stage 4")
    print()

    # --- Stage 4 commit breakdown ---
    print("Stage 4 commits by GT class (precision check; trust = class_match AND okf_within_tol AND ifr_within_tol for displaced):")
    s4_commit_by_gt = defaultdict(int)
    s4_commit_by_gt_match = defaultdict(int)
    for c in cal4.cases:
        if c.decision != "commit":
            continue
        s4_commit_by_gt[c.gt_class] += 1
        if c.class_match and c.okf_within_tol and c.ifr_within_tol:
            s4_commit_by_gt_match[c.gt_class] += 1
    for cls, n in sorted(s4_commit_by_gt.items()):
        ok = s4_commit_by_gt_match.get(cls, 0)
        print(f"  {cls:>22s}: {ok}/{n} pass trust ({100*ok/n:.1f}%)")
    print()

    # Cumulative residual GT distribution
    s4_deferred = {(c.video_id, c.segment_num)
                   for c in cal4.cases if c.decision == "continue"}
    residual = defaultdict(int)
    for (vid, sn) in s4_deferred:
        gt = gt_lookup.get((vid, sn), {})
        residual[gt.get("gt_outcome")] += 1
    print(f"Cumulative residual after Stages 1+2+3+4 ({len(s4_deferred)} segments):")
    for cls, n in sorted(residual.items(), key=lambda x: -x[1]):
        print(f"  {cls:>22s}: {n}")
    print()

    # Cumulative whole-corpus state
    n_total = len(seg_inputs)
    n_by_class = defaultdict(int)
    for (_, _), g in gt_lookup.items():
        n_by_class[g.get("gt_outcome")] += 1
    print(f"Whole-corpus state after Stages 1+2+3+4:")
    print(f"  Total segments: {n_total}")
    print(f"  Cumulative committed: {n_total - len(s4_deferred)}")
    print(f"  Cumulative deferred:  {len(s4_deferred)}")
    print()

    # Yield-per-target-class
    n_untouched_gt = n_by_class.get("untouched", 0)
    n_displaced_gt = n_by_class.get("displaced_sa", 0)
    s1_un_correct = sum(
        1 for c in cal1.cases if c.decision == "commit" and c.committed_class == "untouched"
        and c.class_match and c.okf_within_tol)
    s2_un_correct = sum(
        1 for c in cal2.cases if c.decision == "commit" and c.committed_class == "untouched"
        and c.class_match and c.okf_within_tol)
    s3_un_correct = sum(
        1 for c in cal3.cases if c.decision == "commit" and c.committed_class == "untouched"
        and c.class_match and c.okf_within_tol)
    s4_disp_correct = sum(
        1 for c in cal4.cases if c.decision == "commit" and c.committed_class == "displaced_sa"
        and c.class_match and c.okf_within_tol and c.ifr_within_tol)
    cum_un = s1_un_correct + s2_un_correct + s3_un_correct
    print(f"Cumulative untouched yield:    "
          f"S1={s1_un_correct} + S2={s2_un_correct} + S3={s3_un_correct} = "
          f"{cum_un} / {n_untouched_gt} GT untouched "
          f"({100*cum_un/max(n_untouched_gt,1):.1f}%)")
    print(f"Cumulative displaced_sa yield: "
          f"S4={s4_disp_correct} / {n_displaced_gt} GT displaced_sa "
          f"({100*s4_disp_correct/max(n_displaced_gt,1):.1f}%)")
    print()

    # Save
    out_dir = Path(
        r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
        r"\outcome\v6.0.0_dev_stage4_validate")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage4_results.json").write_text(json.dumps({
        "stage1_n_committed": len(s1_committed),
        "stage2_n_committed": len(s2_committed),
        "stage3_n_committed": len(s3_committed),
        "stage4_n_committed": len(s4_committed),
        "stage4_commit_by_gt": dict(s4_commit_by_gt),
        "stage4_commit_by_gt_pass_trust": dict(s4_commit_by_gt_match),
        "cumulative_residual": dict(residual),
        "cumulative_untouched_yield": cum_un,
        "n_untouched_gt": n_untouched_gt,
        "cumulative_displaced_sa_yield": s4_disp_correct,
        "n_displaced_sa_gt": n_displaced_gt,
    }, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir / 'stage4_results.json'}")


if __name__ == "__main__":
    main()
