"""
v8 dev experiment: phase B LOOCV with merge_gap=3 (was 2).

Forked from restart_phase_b_loocv.py. Follows the failed merge_gap=4
experiment (2026-05-01) which over-merged adjacent real reaches and
regressed on every metric (TP 1918->1732, FP 337->403, FN 457->643,
span_delta tail 8->20). See
  Improvement_Snapshots/reach_detection/v8.0.0_dev_merge_gap_4/EXPERIMENT_RESULTS.md

Rationale (per RECALIBRATION_PLAYBOOK.md step 3):

  Target failure category:
    Same as merge_gap=4 -- split-reach FPs where v8's per-frame
    in-reach probability dips below threshold for ~3 frames mid-reach,
    leaving a single GT reach detected as two algo reaches.

  Code path responsible:
    Same as merge_gap=4 --
    mousereach.reach.v8.postprocess.probabilities_to_reaches,
    parameter merge_gap.

  Why merge_gap=3, not 4:
    The 2026-05-01 merge_gap=4 experiment showed mouse reach pairs in
    this corpus are frequently separated by <=4 not-in-reach frames,
    causing merge_gap=4 to fuse genuinely-adjacent real reaches into
    one wider algo reach. The fused span fails v8's matching span
    tolerance, breaking the matches for both halves and creating
    simultaneous TP-down + FP-up + FN-up regression.

    merge_gap=3 is the minimum bridge that catches a 3-frame mid-reach
    dip (the smallest gap merge_gap=2 misses) while exposing fewer
    real adjacent reach-pairs to fusion. If the typical inter-reach
    gap distribution has a notch at exactly 3 frames, this still
    over-merges; if 3-frame inter-reach gaps are rare relative to
    3-frame mid-reach dips, this works.

    This is the only step finer than merge_gap=4 we can take while
    still exceeding the merge_gap=2 default. If merge_gap=3 also
    regresses, the post-processing knob is exhausted and the next
    iteration should target the GBM probability output directly
    (smoothing, label construction, feature changes) rather than
    post-processing thresholds.

  Risk and guardrail (same as merge_gap=4):
    Over-merging adjacent real reaches. If n_fn rises materially or
    span_delta tail extends substantially (signature of fused-pair
    matches), revert.

  Expected eval deltas (relative to v8.0.0_dev_initial_loocv):
    - n_fp: small drop (split-FPs recovered for 3-frame dips)
    - n_tp: small rise (the recovered halves now combine into matches)
    - n_fn: small drop or unchanged (some 3-frame-dip splits that
      previously failed both span checks now match)
    - tp_start_delta: unchanged
    - tp_span_delta: small rightward shift on previously fragmented
      reaches; tail should NOT extend past ~10-12 frames (v8 baseline
      tail was [-8, 8])

  Decision rule:
    Reject if TP drops AND FN rises (same criterion as merge_gap=4
    rejection). Accept if FN drops OR (TP rises AND FN unchanged).

Output:
  Improvement_Snapshots/reach_detection/v8.0.0_dev_merge_gap_3/
    metrics/
      loocv_per_fold.json
      loocv_aggregate.json
    figures/
      reach_detection_summary.png
      reach_detection_legend.md

Compare side-by-side against:
  Improvement_Snapshots/reach_detection/v8.0.0_dev_initial_loocv/    (baseline, merge_gap=2)
  Improvement_Snapshots/reach_detection/v8.0.0_dev_merge_gap_4/      (rejected, merge_gap=4)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.train import loocv_evaluate, aggregate_folds
from mousereach.improvement.reach_detection.v8_figures import render_v8_reach_figures


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_merge_gap_3"
)

MERGE_GAP = 3  # the experimental parameter (was 2 in the baseline, 4 in the rejected experiment)


def main():
    print("=" * 70)
    print("PHASE B LOOCV (exhaustive subset) -- merge_gap=3 EXPERIMENT")
    print("=" * 70)
    print()

    print("Loading train_pool.parquet ...", flush=True)
    df = pd.read_parquet(CORPUS_DIR / "phase_b_dataset" / "train_pool.parquet")
    folds_def = json.loads(
        (CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))
    train_pool_ids = folds_def["train_pool"]["video_ids"]
    print(f"  Train pool: {len(train_pool_ids)} videos "
          f"({sum(1 for v in train_pool_ids if df[df['video_id']==v]['exhaustive'].iloc[0])} exhaustive)",
          flush=True)
    print(f"  merge_gap={MERGE_GAP}  (baseline = 2, rejected = 4)")
    print()

    print(f"Running LOOCV (1 fold per exhaustive val video) ...", flush=True)
    folds = loocv_evaluate(
        train_pool_df=df,
        train_video_ids=train_pool_ids,
        threshold=0.5,
        merge_gap=MERGE_GAP,
        min_span=3,
        only_exhaustive_for_train=True,
        only_evaluate_exhaustive=True,
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
    )
    print()

    # Aggregate
    agg = aggregate_folds(folds)
    print("=" * 70)
    print(f"AGGREGATE LOOCV RESULTS (merge_gap={MERGE_GAP})")
    print("=" * 70)
    print(f"  TP={agg['n_tp']}  FP={agg['n_fp']}  FN={agg['n_fn']}")
    print(f"  Start delta on TPs: median={agg['tp_start_delta']['median']}f  "
          f"|median|={agg['tp_start_delta']['abs_median']}f  "
          f"range=[{agg['tp_start_delta']['min']},{agg['tp_start_delta']['max']}]")
    print(f"  Span delta on TPs:  median={agg['tp_span_delta']['median']}f  "
          f"|median|={agg['tp_span_delta']['abs_median']}f  "
          f"range=[{agg['tp_span_delta']['min']},{agg['tp_span_delta']['max']}]")
    print()

    # Save artifacts
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    per_fold = []
    all_results = []
    for f in folds:
        per_fold.append({
            "val_video_ids": f.val_video_ids,
            "summary": f.summary,
        })
        for r in f.raw_results:
            all_results.append({
                "status": r.status,
                "video_id": r.video_id,
                "gt_index": r.gt_index,
                "algo_index": r.algo_index,
                "start_delta": r.start_delta,
                "span_delta": r.span_delta,
            })

    (metrics_dir / "loocv_per_fold.json").write_text(
        json.dumps(per_fold, indent=2), encoding="utf-8")
    (metrics_dir / "loocv_aggregate.json").write_text(
        json.dumps({
            "n_folds": len(folds),
            "summary": agg,
            "raw_results": all_results,
            "merge_gap": MERGE_GAP,
        }, indent=2), encoding="utf-8")

    render_v8_reach_figures(
        snapshot_dir=SNAPSHOT_DIR,
        raw_results=all_results,
        summary=agg,
        title_suffix=f" (LOOCV, {len(folds)} folds, exhaustive only, merge_gap={MERGE_GAP})",
    )

    print(f"Wrote: {metrics_dir / 'loocv_per_fold.json'}")
    print(f"Wrote: {metrics_dir / 'loocv_aggregate.json'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_summary.png'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_legend.md'}")


if __name__ == "__main__":
    main()
