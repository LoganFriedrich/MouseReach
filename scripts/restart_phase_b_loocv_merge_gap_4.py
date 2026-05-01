"""
v8 dev experiment: phase B LOOCV with merge_gap=4 (was 2).

Forked from restart_phase_b_loocv.py (which produced
v8.0.0_dev_initial_loocv at merge_gap=2). Only change is the merge_gap
parameter passed to loocv_evaluate -- everything else (corpus, fold
definition, threshold=0.5, min_span=3, training hyperparams) is held
fixed so the A/B comparison isolates the merge_gap effect.

Rationale (per RECALIBRATION_PLAYBOOK.md step 3):

  Target failure category:
    Split-reach FPs -- v8 detects a single GT reach as two algo
    reaches because the per-frame in-reach probability dips below
    threshold for ~3-5 frames mid-reach. Under v8's matching rule
    (start_tol=2, span_tol_rel=0.5, span_tol_abs=5), one half matches
    GT and the other becomes a labeled FP.

  Code path responsible:
    mousereach.reach.v8.postprocess.probabilities_to_reaches,
    parameter merge_gap (default 2). Bridges runs of in-reach frames
    separated by <= merge_gap not-in-reach frames. At merge_gap=2 only
    1-2 frame mid-reach dips get bridged. A 3-5 frame dip leaves the
    reach split.

  Why principled, not "tune until eval passes":
    Mouse reaches are physically continuous events lasting ~6-12
    frames. A GBM classifier's per-frame probability is not guaranteed
    monotonic during a continuous physical event -- small mid-reach
    dips below threshold are an expected artifact of frame-by-frame
    classification, not a signal that the reach has actually ended.
    Merging across these dips reflects underlying biology rather than
    classifier per-frame noise. Risk: over-merging adjacent real
    reaches; mitigated by capping merge_gap below typical inter-reach
    gap and watching span_delta + FN count.

  Expected eval deltas (relative to v8.0.0_dev_initial_loocv):
    - n_fp drops, specifically the subset within +/-2f of a GT reach
    - n_tp holds or rises slightly (some currently-FN reaches whose
      split pieces both fell short of any GT match window now combine)
    - tp_start_delta distribution: unchanged (merge_gap doesn't move
      a run's leading edge)
    - tp_span_delta distribution: slight rightward shift on previously
      fragmented reaches; failure mode is rare large outliers from
      fused real-real pairs
    - n_fn: should NOT rise materially -- if it does, real adjacent
      reaches are being fused, and the change should be reverted

Output:
  Improvement_Snapshots/reach_detection/v8.0.0_dev_merge_gap_4/
    metrics/
      loocv_per_fold.json
      loocv_aggregate.json
    figures/
      reach_detection_summary.png
      reach_detection_legend.md

Compare side-by-side against
  Improvement_Snapshots/reach_detection/v8.0.0_dev_initial_loocv/
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
    r"\reach_detection\v8.0.0_dev_merge_gap_4"
)

MERGE_GAP = 4  # the experimental parameter (was 2 in the baseline)


def main():
    print("=" * 70)
    print("PHASE B LOOCV (exhaustive subset) -- merge_gap=4 EXPERIMENT")
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
    print(f"  merge_gap={MERGE_GAP}  (baseline = 2)")
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
