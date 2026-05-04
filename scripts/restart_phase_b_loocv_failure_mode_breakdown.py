"""
v8 dev experiment: phase B LOOCV with extended output schema for
failure-mode characterization.

This is NOT a parameter change. It is a runner-only instrumentation
experiment that re-runs the baseline LOOCV (merge_gap=2, same params
as v8.0.0_dev_initial_loocv) and writes per-event frame positions
(algo_start_frame, algo_end_frame, gt_start_frame, gt_end_frame) into
the loocv_aggregate.json so downstream diagnostic analysis can
characterize FN sources, boundary-error tail, and FP positions.

NO existing module code (eval.py, train.py, postprocess.py, etc.) is
modified. The runner captures the needed frame data by:
  1. Calling train_one_fold per fold to do the expensive training
  2. Re-running inference inline using the returned classifier (cheap)
     to recover the algo and GT reach lists per video, with frame
     positions intact
  3. Looking up frame positions from those local lists when serializing
     each MatchResult to JSON, using gt_index / algo_index

The aggregate TP/FP/FN counts and delta distributions should be
identical to v8.0.0_dev_initial_loocv (within RNG noise of the inline
re-inference, which uses the same trained classifier and so should be
exactly identical), since the algorithm is unchanged. Only the JSON
output is richer.

Output:
  Improvement_Snapshots/reach_detection/v8.0.0_dev_failure_mode_breakdown/
    metrics/
      loocv_per_fold.json          -- existing schema
      loocv_aggregate.json         -- EXTENDED schema with frame positions
    figures/
      reach_detection_summary.png  -- existing figure (sanity-check
                                      that aggregate matches baseline)

Downstream analysis script (NOT run here, run separately after this
LOOCV completes): produces fn_breakdown, boundary_error_tail,
fp_breakdown into the same snapshot dir.

Sanity check after run: TP=1918 FP=337 FN=457 (same as baseline). Any
deviation indicates a bug in the inline re-inference replication.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.train import train_one_fold, aggregate_folds
from mousereach.reach.v8.eval import GTReach, AlgoReach
from mousereach.reach.v8.postprocess import probabilities_to_reaches
from mousereach.reach.v8.features import feature_columns
from mousereach.improvement.reach_detection.v8_figures import render_v8_reach_figures


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_failure_mode_breakdown"
)

THRESHOLD = 0.5
MERGE_GAP = 2  # SAME as baseline v8.0.0_dev_initial_loocv
MIN_SPAN = 3


def reconstruct_per_video_reaches(df, val_vid, clf, feat_cols):
    """Replicate the inline inference + reach construction that
    train_one_fold does internally, so we can capture algo_reaches
    and gt_reaches lists with frame positions for the val video.

    Same logic as the loop in mousereach.reach.v8.train.train_one_fold,
    just lifted out so the resulting lists are accessible to the caller.
    """
    val = df.loc[df["video_id"] == val_vid]
    Xv = val[feat_cols].to_numpy(dtype=np.float32)
    proba = clf.predict_proba(Xv)[:, 1]

    algo_reaches_raw = probabilities_to_reaches(
        proba, threshold=THRESHOLD, merge_gap=MERGE_GAP, min_span=MIN_SPAN)
    algo_reaches = [
        AlgoReach(
            start_frame=r.start_frame,
            end_frame=r.end_frame,
            video_id=val_vid,
            index=i,
        )
        for i, r in enumerate(algo_reaches_raw)
    ]

    sub = val.sort_values("frame")
    rid = sub["reach_id"].to_numpy()
    frames = sub["frame"].to_numpy()
    gt_reaches = []
    unique_rids = sorted(set(rid[rid >= 0].tolist()))
    for ri in unique_rids:
        rmask = rid == ri
        f = frames[rmask]
        gt_reaches.append(GTReach(
            start_frame=int(f.min()),
            end_frame=int(f.max()),
            video_id=val_vid,
            index=ri,
        ))

    return algo_reaches, gt_reaches


def main():
    print("=" * 70)
    print("PHASE B LOOCV (exhaustive subset) -- failure-mode breakdown")
    print("Runner-only instrumentation: extended JSON output schema")
    print(f"merge_gap={MERGE_GAP} (baseline value, NOT a parameter change)")
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

    feat_cols = feature_columns()

    # Identify exhaustive validation videos (mirrors loocv_evaluate logic).
    exh_set = set(df.loc[df["exhaustive"], "video_id"].unique().tolist())
    eligible_val = [v for v in train_pool_ids if v in exh_set]
    print(f"LOOCV: {len(eligible_val)} exhaustive folds")
    print()

    folds = []
    per_video_data = {}  # vid -> (algo_reaches list, gt_reaches list)

    for i, val_vid in enumerate(eligible_val):
        train_ids = [v for v in train_pool_ids if v != val_vid]
        print(f"  fold {i+1}/{len(eligible_val)}: val={val_vid}", flush=True)
        fold, clf = train_one_fold(
            df, train_ids, [val_vid],
            threshold=THRESHOLD,
            merge_gap=MERGE_GAP,
            min_span=MIN_SPAN,
            only_exhaustive_for_train=True,
            max_iter=200,
            learning_rate=0.05,
            max_depth=6,
        )
        s = fold.summary
        print(f"    TP={s['n_tp']:>4} FP={s['n_fp']:>4} FN={s['n_fn']:>4}  "
              f"start_delta median={s['tp_start_delta']['median']} "
              f"abs_median={s['tp_start_delta']['abs_median']}  "
              f"span_delta median={s['tp_span_delta']['median']} "
              f"abs_median={s['tp_span_delta']['abs_median']}",
              flush=True)
        folds.append(fold)

        # Re-run inference inline to capture frame data
        algo_reaches, gt_reaches = reconstruct_per_video_reaches(
            df, val_vid, clf, feat_cols)
        per_video_data[val_vid] = (algo_reaches, gt_reaches)

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
    print("Sanity check: should match v8.0.0_dev_initial_loocv")
    print("  (TP=1918  FP=337  FN=457  start range [-2,2]  span range [-8,8])")
    print()

    # Save artifacts
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    per_fold_out = []
    all_results = []
    for f in folds:
        per_fold_out.append({
            "val_video_ids": f.val_video_ids,
            "summary": f.summary,
        })
        for r in f.raw_results:
            record = {
                "status": r.status,
                "video_id": r.video_id,
                "gt_index": r.gt_index,
                "algo_index": r.algo_index,
                "start_delta": r.start_delta,
                "span_delta": r.span_delta,
            }
            algo_list, gt_list = per_video_data[r.video_id]
            # Augment with frame positions via index lookup
            if r.algo_index >= 0:
                record["algo_start_frame"] = algo_list[r.algo_index].start_frame
                record["algo_end_frame"] = algo_list[r.algo_index].end_frame
            else:
                record["algo_start_frame"] = -1
                record["algo_end_frame"] = -1
            if r.gt_index >= 0:
                # Note: gt_list is ordered by sorted unique reach_id;
                # gt_index in MatchResult is the list position, NOT
                # the reach_id. So gt_list[gt_index] gives the right GT.
                record["gt_start_frame"] = gt_list[r.gt_index].start_frame
                record["gt_end_frame"] = gt_list[r.gt_index].end_frame
            else:
                record["gt_start_frame"] = -1
                record["gt_end_frame"] = -1
            all_results.append(record)

    (metrics_dir / "loocv_per_fold.json").write_text(
        json.dumps(per_fold_out, indent=2), encoding="utf-8")
    (metrics_dir / "loocv_aggregate.json").write_text(
        json.dumps({
            "n_folds": len(folds),
            "summary": agg,
            "raw_results": all_results,
            "merge_gap": MERGE_GAP,
            "schema_version": "extended_with_frame_positions",
        }, indent=2), encoding="utf-8")

    render_v8_reach_figures(
        snapshot_dir=SNAPSHOT_DIR,
        raw_results=all_results,
        summary=agg,
        title_suffix=f" (LOOCV, {len(folds)} folds, exhaustive only, extended-schema baseline)",
    )

    print(f"Wrote: {metrics_dir / 'loocv_per_fold.json'}")
    print(f"Wrote: {metrics_dir / 'loocv_aggregate.json'}  (extended schema)")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_summary.png'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_legend.md'}")


if __name__ == "__main__":
    main()
