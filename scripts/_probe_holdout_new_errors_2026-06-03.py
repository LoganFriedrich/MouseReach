"""Read-only: characterize the 4 NEW holdout errors (on the previously-
unscoreable non-exhaustive videos). Reuses the deep-dive characterize() +
gate trace. Prints only."""
import importlib.util
from pathlib import Path

S = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_dd", S / "outcome_remaining_errors_deepdive_2026-06-03.py")
dd = importlib.util.module_from_spec(spec); spec.loader.exec_module(dd)

# all 4 are in the model-3.1 corpus paths (canonical DLC + walkthrough GT)
TARGETS = [("20250711_CNT0210_P2", 11), ("20251031_CNT0413_P2", 8),
           ("20251031_CNT0413_P2", 10), ("20251031_CNT0413_P2", 15)]
stages = dd.build_stages_with_leverA(video_dir=None)
cache = {}
for vid, seg_num in TARGETS:
    if vid not in cache:
        cache[vid] = (dd.load_dlc_h5(sorted(dd.M31_DLC.glob(f"{vid}DLC_*.h5"))[0]),
                      dd.load_gt_segments(dd.M31_GT, vid))
    dlc, segs = cache[vid]
    reaches = dd.detect_reaches_v8(dlc)
    s, e = segs[seg_num - 1]
    seg_r = [(r0, r1) for r0, r1 in reaches if s <= r0 <= e]
    seg = dd.SegmentInput(video_id=vid, segment_num=seg_num, seg_start=s, seg_end=e,
                          dlc_df=dlc, reach_windows=seg_r)
    print("=" * 80)
    print(f"{vid} s{seg_num}  frames[{s},{e}] len={e-s+1} n_reaches={len(seg_r)}")
    dd.characterize(seg, [(a - s, b - s) for a, b in seg_r])
    for label, stage in stages:
        d = stage.decide(seg)
        if d.decision != "continue":
            print(f"  VERDICT: {label} -> {d.decision} :: {(d.reason or '')[:90]}")
            break
    print()
