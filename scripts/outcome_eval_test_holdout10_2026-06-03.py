"""
EVAL (not an experiment): v6.0.4 outcome cascade on the FULL 10-video frozen
test holdout, now that all 10 have model-3.1 DLC (the 6 non-exhaustive ones
were inferred 2026-06-03).

This is the cleanest out-of-sample read for the outcome cascade: the 10
test_holdout videos (cv_folds.json) were frozen and never used in cascade
development (which used the 37 train_pool). Until today only the 4 exhaustive
holdout videos had model-3.1 DLC; now all 10 do.

No algorithm change -- scores the shipped v6.0.4 cascade (build_stages_with_leverA).
DLC: canonical model-3.1 folder. GT: canonical reach corpus gt (DLC_2026_03_27\gt).
Reports overall + split by
exhaustive vs non-exhaustive (the 6 newly-scoreable).
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_leverA", SCRIPTS_DIR / "outcome_leverA_net_displaced_sa_2026-06-03.py")
lva = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lva)
detect_reaches_v8 = lva.detect_reaches_v8
load_dlc_h5 = lva.load_dlc_h5
compute_outcome_metrics = lva.compute_outcome_metrics
SegmentInput = lva.SegmentInput
build_stages_with_leverA = lva.build_stages_with_leverA
run_cascade_on_segments = lva.run_cascade_on_segments
save_reaches_segmented = lva.save_reaches_segmented
load_gt_segments = lva.load_gt_segments

DLC_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs"
               r"\DLC_2026_03_27\Processing\updated dlc model 3.1")
# Canonical GT: the reach corpus folder (outcomes reconciled in 2026-06-08).
GT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs"
              r"\DLC_2026_03_27\gt")
SNAP = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
            r"\Improvement_Snapshots\outcome\v6.0.4_eval_test_holdout10_2026-06-03")

# cv_folds.json test_holdout, with exhaustive flag (from restart_inventory).
HOLDOUT = [
    ("20250626_CNT0102_P4", True), ("20250708_CNT0210_P3", True),
    ("20250711_CNT0210_P2", False), ("20250811_CNT0303_P4", True),
    ("20250820_CNT0104_P2", False), ("20250905_CNT0306_P2", False),
    ("20251007_CNT0314_P3", False), ("20251009_CNT0307_P4", False),
    ("20251024_CNT0402_P4", True), ("20251031_CNT0413_P2", False),
]


def main():
    t0 = time.time()
    algo_dir = SNAP / "algo_outputs"; metrics_dir = SNAP / "metrics"
    algo_dir.mkdir(parents=True, exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True)
    stages = build_stages_with_leverA(video_dir=None)
    ids = [v for v, _ in HOLDOUT]
    for i, vid in enumerate(ids, 1):
        h5 = sorted(DLC_DIR.glob(f"{vid}DLC_*.h5"))
        if not h5:
            print(f"  [skip] {vid} no model-3.1 DLC"); continue
        dlc = load_dlc_h5(h5[0]); segs = load_gt_segments(GT_DIR, vid)
        reaches = detect_reaches_v8(dlc)
        save_reaches_segmented(vid, reaches, segs, algo_dir / f"{vid}_reaches.json")
        seg_inputs = [SegmentInput(video_id=vid, segment_num=si + 1, seg_start=s, seg_end=e,
                                   dlc_df=dlc, reach_windows=[(r0, r1) for r0, r1 in reaches if s <= r0 <= e])
                      for si, (s, e) in enumerate(segs)]
        outs = run_cascade_on_segments(seg_inputs, stages)
        if vid in outs:
            (algo_dir / f"{vid}_pellet_outcomes.json").write_text(
                json.dumps({"video_id": vid, "detector": "v6_cascade_leverA_v6.0.4",
                            "segments": outs[vid]}, indent=2), encoding="utf-8")
        print(f"[{i}/10] {vid} ({'exh' if dict(HOLDOUT)[vid] else 'non-exh'})", flush=True)
    # score overall + per-video
    scalars = compute_outcome_metrics(gt_dir=GT_DIR, algo_dir=algo_dir, output_dir=metrics_dir,
                                      video_ids=ids, reaches_dir=algo_dir)
    ps = scalars["outcome_label_per_segment"]; n = scalars["n_segments_paired"]
    correct = round(ps["strict_accuracy"] * n)
    print("\n" + "=" * 60)
    print(f"FULL 10-video test holdout (v6.0.4): {correct}/{n} = {ps['strict_accuracy']:.4f}")
    print(f"confusion: " + ", ".join(f"{k}={v}" for k, v in
          sorted(ps["confusion_matrix"].items(), key=lambda x: -x[1])))
    # per-video correct from per_video.csv
    import csv
    pv = list(csv.DictReader(open(metrics_dir / "per_video.csv")))
    exhmap = dict(HOLDOUT)
    print("\nper-video (strict_accuracy):")
    for r in pv:
        v = r["video_id"]
        print(f"  {v} ({'exh' if exhmap.get(v) else 'non'})  {r['strict_accuracy']}")
    print(f"\nTotal time: {time.time()-t0:.1f}s\nSnapshot: {SNAP}")


if __name__ == "__main__":
    main()
