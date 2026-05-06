"""
Phase B step 2: initial training pass + sanity-check evaluation.

This is the "does anything work?" smoke test before committing to
full leave-one-video-out CV across 20 exhaustive videos.

Procedure:
  1. Load train_pool.parquet
  2. Pick a small held-out val set (3 exhaustive videos, cohort-spread)
  3. Train one HistGradientBoostingClassifier on the remaining 17
     exhaustive videos
  4. Evaluate on val per the user-mandated metric:
        TP iff algo start within +/- 2f of GT AND span match
        Report TP/FP/FN counts plus start-delta + span-delta distributions
  5. If results look at all reasonable, proceed to full LOOCV. If not,
     iterate features/threshold first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.train import train_one_fold


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)


# Pick 3 exhaustive videos for val that span cohorts. These are NOT
# in the test holdout (those are frozen until Phase E).
INITIAL_VAL_VIDEOS = [
    "20250624_CNT0107_P3",   # CNT_01 exhaustive
    "20251030_CNT0403_P1",   # CNT_04 exhaustive
    "20250812_CNT0301_P3",   # CNT_03 exhaustive
]


def main():
    print("=" * 70)
    print("PHASE B INITIAL TRAINING PASS")
    print("=" * 70)
    print()

    print("Loading train_pool.parquet ...", flush=True)
    df = pd.read_parquet(CORPUS_DIR / "phase_b_dataset" / "train_pool.parquet")
    print(f"  Loaded {len(df):,} frames", flush=True)
    print()

    # Restrict to exhaustive subset only for this initial pass.
    # Train videos = exhaustive minus val.
    exh_videos = sorted(df.loc[df["exhaustive"], "video_id"].unique().tolist())
    print(f"Exhaustive videos in train pool: {len(exh_videos)}")
    for v in exh_videos:
        print(f"  {v}")
    print()

    val_videos = [v for v in INITIAL_VAL_VIDEOS if v in exh_videos]
    train_videos = [v for v in exh_videos if v not in val_videos]
    print(f"Train: {len(train_videos)} exhaustive videos")
    print(f"Val:   {len(val_videos)} exhaustive videos: {val_videos}")
    print()
    if len(val_videos) != len(INITIAL_VAL_VIDEOS):
        missing = set(INITIAL_VAL_VIDEOS) - set(exh_videos)
        print(f"  WARNING: requested val videos not in exhaustive train pool: {missing}")
        print()

    print("Training (HistGradientBoostingClassifier, max_iter=200, max_depth=6) ...", flush=True)
    fold, clf = train_one_fold(
        train_pool_df=df,
        train_video_ids=train_videos,
        val_video_ids=val_videos,
        threshold=0.5,
        merge_gap=2,
        min_span=3,
        only_exhaustive_for_train=True,
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
    )
    print()

    s = fold.summary
    print("=" * 70)
    print("INITIAL VAL RESULTS")
    print("=" * 70)
    print()
    print(f"Reach detection (threshold=0.5, merge_gap=2, min_span=3):")
    print(f"  TP={s['n_tp']}  FP={s['n_fp']}  FN={s['n_fn']}")
    print()
    print(f"  Start delta distribution (algo - gt) over {s['tp_start_delta']['n']} TPs:")
    print(f"    median: {s['tp_start_delta']['median']}f  "
          f"|median|: {s['tp_start_delta']['abs_median']}f")
    print(f"    p10:    {s['tp_start_delta']['p10']}f  p90: {s['tp_start_delta']['p90']}f")
    print(f"    range:  [{s['tp_start_delta']['min']}, {s['tp_start_delta']['max']}]")
    print()
    print(f"  Span delta distribution (algo.span - gt.span):")
    print(f"    median: {s['tp_span_delta']['median']}f  "
          f"|median|: {s['tp_span_delta']['abs_median']}f")
    print(f"    p10:    {s['tp_span_delta']['p10']}f  p90: {s['tp_span_delta']['p90']}f")
    print(f"    range:  [{s['tp_span_delta']['min']}, {s['tp_span_delta']['max']}]")
    print()

    # Per-video breakdown
    print(f"  Per-video TP/FP/FN:")
    for vid in val_videos:
        vid_results = [r for r in fold.raw_results if r.video_id == vid]
        n_tp = sum(1 for r in vid_results if r.status == "tp")
        n_fp = sum(1 for r in vid_results if r.status == "fp")
        n_fn = sum(1 for r in vid_results if r.status == "fn")
        print(f"    {vid}: TP={n_tp:>3} FP={n_fp:>3} FN={n_fn:>3}")
    print()

    # Top-20 feature importances (if available)
    print("Top 20 feature importances (sklearn permutation-style):")
    try:
        from mousereach.reach.v8.features import feature_columns
        feat_cols = feature_columns()
        # HistGradientBoostingClassifier doesn't expose feature_importances_
        # directly; use the inner estimator's split-gain proxy.
        # Fallback: just say "not available" if we can't extract it
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            order = np.argsort(imp)[::-1][:20]
            for i in order:
                print(f"    {feat_cols[i]:>30s}: {imp[i]:.4f}")
        else:
            print("  (HistGradientBoostingClassifier has no feature_importances_;")
            print("   will compute via permutation importance in Phase B-3 if needed)")
    except Exception as e:
        print(f"  ERROR extracting importances: {type(e).__name__}: {e}")
    print()

    # Save fold result for later inspection
    OUT = CORPUS_DIR / "phase_b_initial_fold.json"
    OUT.write_text(json.dumps({
        "train_videos": train_videos,
        "val_videos": val_videos,
        "summary": s,
        "results": [
            {"status": r.status, "video_id": r.video_id,
             "gt_index": r.gt_index, "algo_index": r.algo_index,
             "start_delta": r.start_delta, "span_delta": r.span_delta}
            for r in fold.raw_results
        ],
    }, indent=2), encoding="utf-8")
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
