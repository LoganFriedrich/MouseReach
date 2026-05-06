"""
Phase B step 3: full leave-one-video-out CV across 20 exhaustive videos.

Trains one model per fold (val = single held-out exhaustive video,
train = the other 19 exhaustive + can optionally include
non-exhaustive as positive-only). Aggregates results across all folds
into a single canonical Sankey-equivalent reach detection report.

Output:
  Improvement_Snapshots/reach_detection/v8.0.0_dev_initial_loocv/
    metrics/
      loocv_per_fold.json
      loocv_aggregate.json
    figures/
      reach_detection_summary.png
      reach_detection_legend.md
      per_video_breakdown.png
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
    r"\reach_detection\v8.0.0_dev_initial_loocv"
)


def main():
    print("=" * 70)
    print("PHASE B LOOCV (exhaustive subset)")
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
    print()

    print("Running LOOCV (1 fold per exhaustive val video) ...", flush=True)
    folds = loocv_evaluate(
        train_pool_df=df,
        train_video_ids=train_pool_ids,
        threshold=0.5,
        merge_gap=2,
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
    print("AGGREGATE LOOCV RESULTS")
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
        }, indent=2), encoding="utf-8")

    render_v8_reach_figures(
        snapshot_dir=SNAPSHOT_DIR,
        raw_results=all_results,
        summary=agg,
        title_suffix=f" (LOOCV, {len(folds)} folds, exhaustive only)",
    )

    print(f"Wrote: {metrics_dir / 'loocv_per_fold.json'}")
    print(f"Wrote: {metrics_dir / 'loocv_aggregate.json'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_summary.png'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_legend.md'}")


if __name__ == "__main__":
    main()
