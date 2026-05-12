"""
Phase E (cascade restart): validate Stage 1 of the v6 cascade.

For each segment in train_pool:
  - Build SegmentInput using DLC + GT reach windows (cleanest possible
    inputs for the stage to be evaluated on; v8 reaches can be subbed
    in later for the realistic case).
  - Run Stage 1.
  - Record decision + features.

Then report:
  - Yield: how many cases committed to untouched at this stage
  - Per-class breakdown of committed cases (precision check: GT for
    each committed case)
  - Residual pool composition by GT class (what's left for stage 2+)
  - Threshold sensitivity sweep
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.outcomes.v6_cascade.stage_1_pellet_stable_untouched import (
    Stage1PelletStableUntouched)
from mousereach.outcomes.v6_cascade.stage_base import SegmentInput
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


def collapse_outcome(o):
    return "displaced_sa" if o == "displaced_outside" else o


def run_stage1_on_corpus(stage: Stage1PelletStableUntouched) -> List[dict]:
    """Apply Stage 1 to every segment in train_pool. Returns list of
    dicts: {video_id, segment_num, gt_outcome, decision,
            committed_class, reason, features}.
    """
    folds = json.loads((CORPUS_DIR / "cv_folds.json").read_text())
    train_pool_ids = folds["train_pool"]["video_ids"]

    rows = []
    for vid in train_pool_ids:
        dlc = load_dlc_h5(find_dlc(vid))
        seg_bounds = load_segment_bounds(vid)
        gt = load_gt(vid)
        gt_outcomes = {s["segment_num"]: collapse_outcome(s.get("outcome"))
                       for s in gt.get("outcomes", {}).get("segments", []) or []}

        for sn, (s_start, s_end) in seg_bounds.items():
            gt_outcome = gt_outcomes.get(sn)
            if gt_outcome is None:
                continue
            seg_input = SegmentInput(
                video_id=vid, segment_num=sn,
                seg_start=s_start, seg_end=s_end,
                dlc_df=dlc,
                reach_windows=gt_reaches_for_segment(gt, sn),
            )
            decision = stage.decide(seg_input)
            rows.append({
                "video_id": vid,
                "segment_num": sn,
                "gt_outcome": gt_outcome,
                "decision": decision.decision,
                "committed_class": decision.committed_class,
                "reason": decision.reason,
                **decision.features,
            })
    return rows


def report(rows: List[dict], threshold_label: str = "default"):
    df = pd.DataFrame(rows)
    n_total = len(df)
    n_commit = (df["decision"] == "commit").sum()
    n_continue = (df["decision"] == "continue").sum()
    print(f"\n--- {threshold_label} ---")
    print(f"  Total segments: {n_total}")
    print(f"  Committed (untouched): {n_commit}  ({100*n_commit/n_total:.1f}%)")
    print(f"  Continuing: {n_continue}  ({100*n_continue/n_total:.1f}%)")
    # Per-class breakdown of committed
    com = df[df["decision"] == "commit"]
    print(f"  Committed -- GT distribution:")
    for cls, c in com["gt_outcome"].value_counts().items():
        n_gt_total_for_class = (df["gt_outcome"] == cls).sum()
        print(f"    {cls:>22s}: {c}/{n_gt_total_for_class}  "
              f"(stage1 yield={100*c/n_gt_total_for_class:.1f}%)")
    # Residual pool composition
    res = df[df["decision"] == "continue"]
    print(f"  Residual pool -- GT distribution:")
    for cls, c in res["gt_outcome"].value_counts().items():
        print(f"    {cls:>22s}: {c}")


def main():
    print("=" * 70)
    print("PHASE E STAGE 1 VALIDATION")
    print("=" * 70)

    # Default thresholds first
    stage = Stage1PelletStableUntouched(
        commit_frac=0.95, commit_distance_radii=1.0)
    rows_default = run_stage1_on_corpus(stage)
    report(rows_default, "default (frac>=0.95, dist<=1.0r)")

    # Sensitivity sweep
    for cf, cd in [(0.90, 1.0), (0.95, 1.5), (0.98, 0.5), (0.99, 0.3)]:
        stage = Stage1PelletStableUntouched(
            commit_frac=cf, commit_distance_radii=cd)
        rows = run_stage1_on_corpus(stage)
        report(rows, f"frac>={cf}, dist<={cd}r")

    # Save default decisions
    out_dir = Path(
        r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
        r"\outcome\v6.0.0_dev_stage1_validate")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage1_decisions.json").write_text(
        json.dumps(rows_default, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_dir / 'stage1_decisions.json'}")


if __name__ == "__main__":
    main()
