"""
DIAGNOSTIC (not an experiment): stage-by-stage trace of the 4 Lever-A
displaced_sa yield-gap segments.

These 4 segments have GT=displaced_sa with the pellet tracked by DLC the
whole time (Logan's 2026-06-03 per-error review), yet the cascade declined
on every displaced_sa stage and fell through to Stage 99 triage. This
trace runs EVERY stage's decide() (not stopping at first commit) so we can
see exactly which condition each displaced_sa-target stage failed on.

No algorithm change. Reads the production cascade (v6.0.3 via the Fix B
runner) on Y: data. Output is printed only -- nothing written.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_outcome_fix_b", SCRIPTS_DIR / "outcome_fix_b_retrieved_rescue_2026-06-02.py")
fixb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixb)

detect_reaches_v8 = fixb.detect_reaches_v8
load_dlc_h5 = fixb.load_dlc_h5
SegmentInput = fixb.SegmentInput
build_stages_with_fix_b = fixb.build_stages_with_fix_b

DLC_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs"
               r"\DLC_2026_03_27\Processing\updated dlc model 3.1")
GT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs"
              r"\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough\gt")

# Lever A: pellet tracked throughout, GT=displaced_sa, cascade triaged.
TARGETS = [
    ("20250627_CNT0105_P1", 7),
    ("20250710_CNT0215_P4", 11),
    ("20250710_CNT0215_P4", 13),
    ("20251028_CNT0404_P4", 16),
]

# Stages whose target is displaced_sa (the ones that SHOULD have fired).
import json as _json


def load_gt_segments(vid):
    gt = _json.loads((GT_DIR / f"{vid}_unified_ground_truth.json").read_text(encoding="utf-8"))
    bs = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
    return [(bs[i], bs[i + 1] - 1) for i in range(len(bs) - 1)] if len(bs) >= 2 else []


def find_dlc(vid):
    return sorted(DLC_DIR.glob(f"{vid}DLC_*.h5"))[0]


def main():
    stages = build_stages_with_fix_b(video_dir=None)
    dlc_cache = {}
    for vid, seg_num in TARGETS:
        if vid not in dlc_cache:
            dlc_cache[vid] = (load_dlc_h5(find_dlc(vid)), load_gt_segments(vid))
        dlc, segments = dlc_cache[vid]
        reaches = detect_reaches_v8(dlc)
        s, e = segments[seg_num - 1]
        seg_r = [(r0, r1) for r0, r1 in reaches if s <= r0 <= e]
        seg = SegmentInput(video_id=vid, segment_num=seg_num,
                           seg_start=s, seg_end=e, dlc_df=dlc, reach_windows=seg_r)
        print("=" * 78)
        print(f"{vid} s{seg_num}  frames [{s}, {e}]  n_reaches_in_seg={len(seg_r)}")
        print(f"  reach windows: {seg_r}")
        print("-" * 78)
        first_commit = None
        for label, stage in stages:
            try:
                d = stage.decide(seg)
            except Exception as ex:
                print(f"  {label:48s} ERROR: {ex}")
                continue
            target = getattr(stage, "target_class", "")
            mark = ""
            if d.decision == "commit":
                mark = f" <== COMMIT {d.committed_class}"
                if first_commit is None:
                    first_commit = (label, d.committed_class)
            # Only print stages that did something OR target displaced_sa
            # (the ones we care about for this gap).
            is_disp = (target == "displaced_sa")
            if d.decision != "continue" or is_disp:
                tag = "[disp_sa]" if is_disp else "         "
                reason = (d.reason or "")[:90]
                print(f"  {tag} {label:46s} {d.decision:9s}{mark}")
                if reason:
                    print(f"            reason: {reason}")
        print("-" * 78)
        print(f"  FIRST COMMIT (cascade verdict): {first_commit}")
        print()


if __name__ == "__main__":
    main()
