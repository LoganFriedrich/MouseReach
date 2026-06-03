"""
DIAGNOSTIC (not an experiment): stage-by-stage trace of the 2 remaining
boundary-case displaced_sa segments against the CURRENT v6.0.4 cascade
(Lever A merged).

Targets (Logan's 2026-06-03 per-error notes -- both GT=displaced_sa,
triaged, pellet tracked):
  20250630_CNT0104_P3 s17  "near end of segment"
  20251031_CNT0407_P1 s19  "near start of segment"

Goal: confirm exactly which stage/condition defers each, so we don't
assume both are a Stage 27 buffer case. Prints all stage decisions; no
code change, nothing written.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
# Import the merged Lever A runner to get the v6.0.4 cascade build.
_spec = importlib.util.spec_from_file_location(
    "_leverA", SCRIPTS_DIR / "outcome_leverA_net_displaced_sa_2026-06-03.py")
lva = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lva)

detect_reaches_v8 = lva.detect_reaches_v8
load_dlc_h5 = lva.load_dlc_h5
SegmentInput = lva.SegmentInput
build_stages_with_leverA = lva.build_stages_with_leverA

DLC_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs"
               r"\DLC_2026_03_27\Processing\updated dlc model 3.1")
GT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs"
              r"\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough\gt")

TARGETS = [("20250630_CNT0104_P3", 17), ("20251031_CNT0407_P1", 19)]


def load_gt_segments(vid):
    gt = json.loads((GT_DIR / f"{vid}_unified_ground_truth.json").read_text(encoding="utf-8"))
    bs = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
    return [(bs[i], bs[i + 1] - 1) for i in range(len(bs) - 1)] if len(bs) >= 2 else []


def main():
    stages = build_stages_with_leverA(video_dir=None)
    cache = {}
    for vid, seg_num in TARGETS:
        if vid not in cache:
            cache[vid] = (load_dlc_h5(sorted(DLC_DIR.glob(f"{vid}DLC_*.h5"))[0]),
                          load_gt_segments(vid))
        dlc, segments = cache[vid]
        reaches = detect_reaches_v8(dlc)
        s, e = segments[seg_num - 1]
        seg_r = [(r0, r1) for r0, r1 in reaches if s <= r0 <= e]
        seg = SegmentInput(video_id=vid, segment_num=seg_num, seg_start=s,
                           seg_end=e, dlc_df=dlc, reach_windows=seg_r)
        seg_len = e - s + 1
        print("=" * 78)
        print(f"{vid} s{seg_num}  frames [{s}, {e}]  seg_len={seg_len}  n_reaches={len(seg_r)}")
        if seg_r:
            first_off = seg_r[0][0] - s
            last_off = e - seg_r[-1][1]
            print(f"  first reach starts {first_off}f into segment; "
                  f"last reach ends {last_off}f before segment end")
            print(f"  reach windows (local): {[(a - s, b - s) for a, b in seg_r]}")
        print("-" * 78)
        committed = None
        for label, stage in stages:
            try:
                d = stage.decide(seg)
            except Exception as ex:
                print(f"  {label:48s} ERROR {ex}"); continue
            if d.decision != "continue":
                print(f"  {label:48s} {d.decision.upper()}")
                print(f"        reason: {(d.reason or '')[:140]}")
                if d.decision == "commit" and committed is None:
                    committed = (label, d.committed_class)
                    break
        print("-" * 78)
        print(f"  VERDICT: {committed if committed else 'TRIAGE (fell through)'}")
        print()


if __name__ == "__main__":
    main()
