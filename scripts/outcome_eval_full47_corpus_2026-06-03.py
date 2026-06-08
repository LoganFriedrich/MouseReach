"""
EVAL (not an experiment): v6.0.4 outcome cascade on the FULL 47-video corpus
(cv_folds.json: 37 train_pool + 10 test_holdout), now that all 47 have
model-3.1 DLC.

Until 2026-06-03 only the 20 exhaustive-GT videos had model-3.1 DLC. The 27
non-exhaustive (21 train + 6 holdout) were inferred + added 2026-06-03, so the
whole corpus can now be scored on model 3.1.

No algorithm change -- scores the shipped v6.0.4 cascade (build_stages_with_leverA).
DLC: canonical model-3.1 folder. GT: canonical reach corpus gt (DLC_2026_03_27\gt).
Reports overall + splits
(train vs holdout; exhaustive vs non-exhaustive).
"""
from __future__ import annotations
import importlib.util, json, time, csv
from pathlib import Path

S = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_lva", S / "outcome_leverA_net_displaced_sa_2026-06-03.py")
lva = importlib.util.module_from_spec(spec); spec.loader.exec_module(lva)

DLC_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\Processing\updated dlc model 3.1")
# Canonical GT: the reach corpus folder. Outcomes were reconciled into this
# folder on 2026-06-08 so reaches and outcomes read from one place.
GT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\gt")
CVF = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\_corpus\2026-04-30_restart_inventory\cv_folds.json")
SNAP = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\outcome\v6.0.4_eval_full47_corpus_2026-06-03")

cv = json.loads(CVF.read_text())
TRAIN = set(cv["train_pool"]["video_ids"]); HOLD = set(cv["test_holdout"]["video_ids"])
ALL = sorted(TRAIN | HOLD)
EXH = set(lva.M31["ids"])  # the 20 exhaustive-GT videos


def score_subset(ids, algo_dir):
    md = SNAP / "metrics_tmp"; md.mkdir(parents=True, exist_ok=True)
    sc = lva.compute_outcome_metrics(gt_dir=GT_DIR, algo_dir=algo_dir, output_dir=md,
                                     video_ids=sorted(ids), reaches_dir=algo_dir)
    ps = sc["outcome_label_per_segment"]; n = sc["n_segments_paired"]
    return round(ps["strict_accuracy"] * n), n, ps["confusion_matrix"]


def main():
    t0 = time.time()
    algo_dir = SNAP / "algo_outputs"; metrics_dir = SNAP / "metrics"
    algo_dir.mkdir(parents=True, exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True)
    stages = lva.build_stages_with_leverA(video_dir=None)
    scored = []
    for i, vid in enumerate(ALL, 1):
        h5 = sorted(DLC_DIR.glob(f"{vid}DLC_*.h5"))
        gt = GT_DIR / f"{vid}_unified_ground_truth.json"
        if not h5 or not gt.exists():
            print(f"  [skip] {vid} (dlc={bool(h5)} gt={gt.exists()})"); continue
        dlc = lva.load_dlc_h5(h5[0]); segs = lva.load_gt_segments(GT_DIR, vid)
        reaches = lva.detect_reaches_v8(dlc)
        lva.save_reaches_segmented(vid, reaches, segs, algo_dir / f"{vid}_reaches.json")
        seg_inputs = [lva.SegmentInput(video_id=vid, segment_num=si+1, seg_start=s, seg_end=e,
                      dlc_df=dlc, reach_windows=[(r0,r1) for r0,r1 in reaches if s<=r0<=e])
                      for si,(s,e) in enumerate(segs)]
        outs = lva.run_cascade_on_segments(seg_inputs, stages)
        if vid in outs:
            (algo_dir / f"{vid}_pellet_outcomes.json").write_text(
                json.dumps({"video_id": vid, "detector": "v6_cascade_leverA_v6.0.4", "segments": outs[vid]},
                           indent=2), encoding="utf-8")
        scored.append(vid)
        if i % 10 == 0: print(f"  [{i}/{len(ALL)}] scored", flush=True)

    full = lva.compute_outcome_metrics(gt_dir=GT_DIR, algo_dir=algo_dir, output_dir=metrics_dir,
                                       video_ids=scored, reaches_dir=algo_dir)
    ps = full["outcome_label_per_segment"]; n = full["n_segments_paired"]
    correct = round(ps["strict_accuracy"] * n)
    print("\n" + "=" * 66)
    print(f"FULL 47-video corpus (v6.0.4): {correct}/{n} = {ps['strict_accuracy']:.4f}")
    print(f"confusion: " + ", ".join(f"{k}={v}" for k, v in sorted(ps['confusion_matrix'].items(), key=lambda x:-x[1])))
    print("\nSplits:")
    for label, subset in (("train_pool (37)", TRAIN), ("test_holdout (10)", HOLD),
                          ("exhaustive (20)", EXH), ("non-exhaustive (27)", set(ALL) - EXH)):
        ids = subset & set(scored)
        c, nn, _ = score_subset(ids, algo_dir)
        print(f"  {label:22s} {c}/{nn} = {c/nn:.4f}")
    print(f"\nTotal time: {time.time()-t0:.1f}s\nSnapshot: {SNAP}")


if __name__ == "__main__":
    main()
