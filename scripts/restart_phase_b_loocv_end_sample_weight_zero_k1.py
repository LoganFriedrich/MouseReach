"""
v8 dev experiment: phase B LOOCV with sample-weight = 0 on the last K
frames of each reach, layered on BSW w=0.8.

Differs from the rejected v8.0.0_dev_asymmetric_label_tightening_end1
experiment: instead of FLIPPING the label from 1 to 0 (which created
contradictory training signal at frames where the paw is still
mid-reach), this experiment sets sample_weight = 0 for those frames.
The model isn't trained on them -- it doesn't get a contradictory
signal, just no signal at all. Predictions on those frames will
depend purely on whatever the GBM extrapolates from features.

==============================================================================
PRE-EXPERIMENT CHECKLIST (per pre_experiment_checklist.md, walked 2026-05-05)
==============================================================================

1. Cumulative-stacking check (DYNAMIC, verified at experiment time):
   - Read v8_pending_integrations.md: BSW b=1 w=0.8 listed ACTIVE.
   - Snapshot RESULTS.md (v8.0.0_dev_boundary_sample_weight_b1_w0.8/):
     status ACCEPTED [verified 2026-05-05].
   - git log on master since 529f688: no revert/retract commits
     [verified 2026-05-05].
   - train_one_fold inspection: still pending (uses class-imbalance-only
     sample_weight) [verified 2026-05-05].

   Stacked improvements applied: BSW b=1 w=0.8 + new end-frame
   sample-weight zeroing (K=1).
   Comparison baseline (cumulative best): v8.0.0_dev_boundary_sample_weight_b1_w0.8/
     (TP=1935, FP=330, FN=440, exact_start=84.08%, exact_span=75.09%)
   Reference baseline (pure): v8.0.0_dev_initial_loocv/
     (TP=1918, FP=337, FN=457)

2. Existing-code-modification check:
   Existing module code modified: NO. Runner replicates train_one_fold
   inline. Sample-weight zeroing applied inline as a sample_weight
   multiplier (compose with class-weight + BSW boundary-weight).

3. Assumption check (unverified hypotheses flagged):
   - HYPOTHESIS: setting sample_weight=0 for the last K frames of each
     reach, instead of flipping their label, will avoid the systematic
     -1 shift seen in the prior end-tighten experiment. UNVERIFIED.
     This is the central claim being tested.
   - HYPOTHESIS: with no training signal on those frames, the model's
     prediction there will be determined by features only -- and may
     end up "naturally" lower than 1.0 because the mid-reach feature
     pattern is weakening (paw decelerating, etc.). If true, the
     threshold-crossing might happen near GT_end without forcing the
     model to predict 0 there. UNVERIFIED.
   - HYPOTHESIS: the model's existing inductive bias from clearly-
     in-reach (mid-reach) frames will still cause it to predict close
     to 1 for the dropped frames -- because they look like reach
     frames. If true, this experiment will be approximately equivalent
     to baseline (no useful change). UNVERIFIED.

4. FN-direction-reporting check:
   The results report will lead with FN delta vs BSW w=0.8 (cumulative
   best) AND vs pure baseline.

5. Framework-not-adhoc check: canonical snapshot dir layout.

6. Branch + tag check:
   Pre-experiment tag: v8-pre-end-sample-weight-zero-2026-05-05
   Branch: feature/v8-end-sample-weight-zero-k1

7. Decision-rule check:
   Reject if TP drops AND FN rises vs BSW w=0.8 (cumulative best).
   Reject if exact-frame-match rate drops materially vs BSW w=0.8.
   Accept if FN drops OR TP rises with exact-match held.

==============================================================================
RATIONALE
==============================================================================

Target failure mode (unchanged): within_gt FPs (270/330 = 81.8% of
FPs) + tol_miss_span FNs (115/440 = 26.1% of FNs) share root cause
"algo span extends past GT_end."

Prior end-tightening attempt (label-flipping K=1, rejected 2026-05-04):
the explicit forcing of label=0 on the last frame of each reach
caused the model to systematically shift terminations -1 frame across
ALL reaches, not just the over-extending ones. exact_span crashed
75% -> 60%. The mean span improved (+0.17 -> +0.05) but at the cost
of exact-match precision -- a Cardinal Rule violation.

This experiment uses the SOFTER mechanism: sample_weight = 0 instead
of label = 0. The model isn't told "these frames are NOT reach"; it's
just not trained on them. Its prediction there depends on features
alone -- if the features look reach-like, the model still predicts
~1; if they look transitional, the model may predict lower. No
forcing.

Why this could help OR be neutral:
- Help: the dropped frames near GT_end may have features that are
  "borderline" between reach and not-reach. Without explicit label-1
  forcing, the model's prediction on them naturally lands lower than
  for clear mid-reach frames. Thresholding gives a slightly earlier
  termination on AVERAGE, without the universal -1 shift of the
  label-flip experiment.
- Neutral: the model has so much in-reach signal from the OTHER
  reach frames that it generalizes the same prediction to the
  dropped frames anyway. Net effect ~zero.
- Hurt: less likely, but the GBM could become slightly more
  uncertain near boundaries due to reduced signal density there.

Per the playbook, even a "neutral" result is informative -- it tells
us that label-modification approaches won't help, period (because
the gentlest variant doesn't move the needle). That outcome would
push us toward feature-based or architecture-based fixes.

K=1, MIN_REACH_FOR_TIGHTEN=5 (same as the rejected variant for
direct comparability).

==============================================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.eval import (
    GTReach, AlgoReach, evaluate_reaches, summarize_results,
)
from mousereach.reach.v8.postprocess import probabilities_to_reaches
from mousereach.reach.v8.features import feature_columns
from mousereach.improvement.reach_detection.v8_figures import render_v8_reach_figures


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)

# Stacked accepted improvement: BSW b=1 w=0.8
BOUNDARY_BUFFER = 1
BOUNDARY_WEIGHT = 0.8

# New change: zero sample_weight on last K frames of each reach
END_ZERO_FRAMES = 1
MIN_REACH_FOR_ZEROING = 5  # skip if reach span <= this

THRESHOLD = 0.5
MERGE_GAP = 2
MIN_SPAN = 3

SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_end_sample_weight_zero_k1"
)


def compute_end_sample_weight_mask(df, k=1, min_reach_for_zeroing=5):
    """For each reach (group by video_id, reach_id) with span >
    min_reach_for_zeroing, return a multiplier of 0 for the last k
    frames. Multiplier of 1 elsewhere. Multiplies onto existing
    sample_weight (class_weight * BSW boundary_weight * this).

    Returns
    -------
    mask : np.ndarray of float32, length len(df), aligned to df row order.
        1.0 for frames not in the end-zone, 0.0 for last k frames of
        eligible reaches.
    stats : dict with counts (n_reaches_zeroed, n_frames_zeroed, etc.)
    """
    mask = np.ones(len(df), dtype=np.float32)
    stats = {
        "n_reaches_zeroed": 0,
        "n_reaches_skipped_short": 0,
        "n_frames_zeroed": 0,
    }

    in_reach = df[df["reach_id"] >= 0]
    for (vid, rid), group in in_reach.groupby(["video_id", "reach_id"], sort=False):
        sorted_group = group.sort_values("frame")
        idx_sorted = sorted_group.index.to_numpy()
        span = len(idx_sorted)
        if span <= min_reach_for_zeroing:
            stats["n_reaches_skipped_short"] += 1
            continue
        # Map to df-positional indices then zero
        end_idx = idx_sorted[-k:]
        # Convert end_idx (which are df.index labels) to positions
        df_positions = df.index.get_indexer(end_idx)
        mask[df_positions] = 0.0
        stats["n_reaches_zeroed"] += 1
        stats["n_frames_zeroed"] += k

    return mask, stats


def compute_boundary_weights(train_df, n_buffer=1, boundary_weight=0.5):
    """BSW: per-row sample-weight multiplier; reduced near reach
    boundaries (computed from reach_id transitions, unchanged from
    cumulative-best convention)."""
    sorted_df = train_df.sort_values(["video_id", "frame"])
    rid = sorted_df["reach_id"].to_numpy()
    vid = sorted_df["video_id"].to_numpy()
    n = len(sorted_df)
    transitions = np.zeros(n, dtype=bool)
    if n >= 2:
        same_video = vid[1:] == vid[:-1]
        rid_change = rid[1:] != rid[:-1]
        boundary_pairs = same_video & rid_change
        transitions[1:] |= boundary_pairs
        transitions[:-1] |= boundary_pairs
    dilated = transitions.copy()
    for d in range(1, n_buffer + 1):
        dilated[d:] |= transitions[:-d]
        dilated[:-d] |= transitions[d:]
    weights_sorted = np.ones(n, dtype=np.float32)
    weights_sorted[dilated] = boundary_weight
    weights_series = pd.Series(weights_sorted, index=sorted_df.index)
    return weights_series.reindex(train_df.index).to_numpy()


def train_one_fold_combined(train_pool_df, end_zero_mask, train_video_ids, val_vid, feat_cols):
    """Replicates train_one_fold + BSW + end-frame sample-weight zeroing."""
    train_mask = train_pool_df["video_id"].isin(train_video_ids) & train_pool_df["exhaustive"]
    train_mask &= train_pool_df["video_id"] != val_vid
    train = train_pool_df.loc[train_mask]
    val = train_pool_df.loc[train_pool_df["video_id"] == val_vid]

    # Aligned end-zero mask for the train rows
    end_zero_train = end_zero_mask[train.index.values - train_pool_df.index.values[0]] \
        if (train_pool_df.index == np.arange(len(train_pool_df))).all() \
        else end_zero_mask[train_pool_df.index.get_indexer(train.index)]

    X_train = train[feat_cols].to_numpy(dtype=np.float32)
    y_train = train["label"].to_numpy(dtype=np.int8)

    n = len(y_train)
    n_pos = int(y_train.sum())
    n_neg = n - n_pos
    if n_pos > 0 and n_neg > 0:
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)
        class_w = np.where(y_train == 1, w_pos, w_neg).astype(np.float32)
    else:
        class_w = np.ones(n, dtype=np.float32)

    boundary_w = compute_boundary_weights(train, n_buffer=BOUNDARY_BUFFER, boundary_weight=BOUNDARY_WEIGHT)

    sample_weight = (class_w * boundary_w * end_zero_train).astype(np.float32)
    n_zeroed_in_train = int((end_zero_train == 0).sum())

    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=6,
        random_state=42, early_stopping=False,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    Xv = val[feat_cols].to_numpy(dtype=np.float32)
    proba = clf.predict_proba(Xv)[:, 1]

    algo_reaches_raw = probabilities_to_reaches(
        proba, threshold=THRESHOLD, merge_gap=MERGE_GAP, min_span=MIN_SPAN)
    algo_reaches = [
        AlgoReach(start_frame=r.start_frame, end_frame=r.end_frame, video_id=val_vid, index=i)
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
            start_frame=int(f.min()), end_frame=int(f.max()),
            video_id=val_vid, index=ri))

    results = evaluate_reaches(algo_reaches, gt_reaches, video_id=val_vid)
    summary = summarize_results(results)
    return summary, results, algo_reaches, gt_reaches, n_zeroed_in_train


def main():
    print("=" * 70)
    print("PHASE B LOOCV (exhaustive subset)")
    print(f"  Stacked: BSW b={BOUNDARY_BUFFER} w={BOUNDARY_WEIGHT}")
    print(f"  New: sample_weight=0 on last {END_ZERO_FRAMES} frame(s) per reach")
    print(f"       (skip if reach span <= {MIN_REACH_FOR_ZEROING})")
    print("=" * 70)
    print()

    print("Loading train_pool.parquet ...", flush=True)
    df = pd.read_parquet(CORPUS_DIR / "phase_b_dataset" / "train_pool.parquet")
    folds_def = json.loads((CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))
    train_pool_ids = folds_def["train_pool"]["video_ids"]
    print(f"  Train pool: {len(train_pool_ids)} videos "
          f"({sum(1 for v in train_pool_ids if df[df['video_id']==v]['exhaustive'].iloc[0])} exhaustive)",
          flush=True)
    print()

    print(f"Computing end-zero mask: zeroing last {END_ZERO_FRAMES} frame(s) per reach (skip span <= {MIN_REACH_FOR_ZEROING}) ...",
          flush=True)
    end_zero_mask, ez_stats = compute_end_sample_weight_mask(
        df, k=END_ZERO_FRAMES, min_reach_for_zeroing=MIN_REACH_FOR_ZEROING)
    print(f"  reaches zeroed:           {ez_stats['n_reaches_zeroed']}")
    print(f"  reaches skipped (short):  {ez_stats['n_reaches_skipped_short']}")
    print(f"  total frames zeroed:      {ez_stats['n_frames_zeroed']}")
    print(f"  frames with mask=0:       {int((end_zero_mask == 0).sum())} of {len(end_zero_mask)}")
    print()

    feat_cols = feature_columns()
    exh_set = set(df.loc[df["exhaustive"], "video_id"].unique().tolist())
    eligible_val = [v for v in train_pool_ids if v in exh_set]
    print(f"LOOCV: {len(eligible_val)} exhaustive folds")
    print()

    folds = []
    per_video_data = {}
    all_results_combined = []

    for i, val_vid in enumerate(eligible_val):
        train_ids = [v for v in train_pool_ids if v != val_vid]
        print(f"  fold {i+1}/{len(eligible_val)}: val={val_vid}", flush=True)
        summary, results, algo_reaches, gt_reaches, n_zeroed_in_train = \
            train_one_fold_combined(df, end_zero_mask, train_ids, val_vid, feat_cols)

        s = summary
        sd_mean = s['tp_start_delta']['mean']
        sd_mean_str = f"{sd_mean:.3f}" if sd_mean is not None else "n/a"
        sp_mean = s['tp_span_delta']['mean']
        sp_mean_str = f"{sp_mean:.3f}" if sp_mean is not None else "n/a"
        print(f"    TP={s['n_tp']:>4} FP={s['n_fp']:>4} FN={s['n_fn']:>4}  "
              f"start_delta median={s['tp_start_delta']['median']} mean={sd_mean_str}  "
              f"span_delta median={s['tp_span_delta']['median']} mean={sp_mean_str}  "
              f"n_zeroed_in_train={n_zeroed_in_train}",
              flush=True)
        folds.append({"val_video_ids": [val_vid], "summary": summary})
        per_video_data[val_vid] = (algo_reaches, gt_reaches)
        all_results_combined.extend(results)

    print()
    agg = summarize_results(all_results_combined)
    print("=" * 70)
    print(f"AGGREGATE LOOCV RESULTS")
    print("=" * 70)
    sd_mean_a = agg['tp_start_delta']['mean']
    sp_mean_a = agg['tp_span_delta']['mean']
    sd_mean_a_s = f"{sd_mean_a:.3f}" if sd_mean_a is not None else "n/a"
    sp_mean_a_s = f"{sp_mean_a:.3f}" if sp_mean_a is not None else "n/a"
    print(f"  TP={agg['n_tp']}  FP={agg['n_fp']}  FN={agg['n_fn']}")
    print(f"  Start delta on TPs: median={agg['tp_start_delta']['median']}f  "
          f"|median|={agg['tp_start_delta']['abs_median']}f  "
          f"mean={sd_mean_a_s}  "
          f"range=[{agg['tp_start_delta']['min']},{agg['tp_start_delta']['max']}]")
    print(f"  Span delta on TPs:  median={agg['tp_span_delta']['median']}f  "
          f"|median|={agg['tp_span_delta']['abs_median']}f  "
          f"mean={sp_mean_a_s}  "
          f"range=[{agg['tp_span_delta']['min']},{agg['tp_span_delta']['max']}]")
    print()
    print("Compare against:")
    print("  Pure baseline:               TP=1918  FP=337  FN=457  span mean=0.212")
    print("  BSW w=0.8 (cumul. best):     TP=1935  FP=330  FN=440  span mean=0.170")
    print("  Prior end-flip K=1 (rejected): TP=1910  FP=375  FN=465  exact_span=59.95% (BAD)")
    print()

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    serialized_results = []
    for r in all_results_combined:
        record = {
            "status": r.status, "video_id": r.video_id,
            "gt_index": r.gt_index, "algo_index": r.algo_index,
            "start_delta": r.start_delta, "span_delta": r.span_delta,
        }
        algo_list, gt_list = per_video_data[r.video_id]
        if r.algo_index >= 0:
            record["algo_start_frame"] = algo_list[r.algo_index].start_frame
            record["algo_end_frame"] = algo_list[r.algo_index].end_frame
        else:
            record["algo_start_frame"] = -1
            record["algo_end_frame"] = -1
        if r.gt_index >= 0:
            record["gt_start_frame"] = gt_list[r.gt_index].start_frame
            record["gt_end_frame"] = gt_list[r.gt_index].end_frame
        else:
            record["gt_start_frame"] = -1
            record["gt_end_frame"] = -1
        serialized_results.append(record)

    (metrics_dir / "loocv_per_fold.json").write_text(
        json.dumps(folds, indent=2), encoding="utf-8")
    (metrics_dir / "loocv_aggregate.json").write_text(
        json.dumps({
            "n_folds": len(folds), "summary": agg,
            "raw_results": serialized_results,
            "merge_gap": MERGE_GAP,
            "boundary_buffer": BOUNDARY_BUFFER, "boundary_weight": BOUNDARY_WEIGHT,
            "end_zero_frames": END_ZERO_FRAMES,
            "min_reach_for_zeroing": MIN_REACH_FOR_ZEROING,
            "end_zero_stats": ez_stats,
            "schema_version": "extended_with_frame_positions",
        }, indent=2), encoding="utf-8")

    render_v8_reach_figures(
        snapshot_dir=SNAPSHOT_DIR,
        raw_results=serialized_results,
        summary=agg,
        title_suffix=f" (LOOCV, BSW w={BOUNDARY_WEIGHT} + end-sw-zero K={END_ZERO_FRAMES})",
    )

    print(f"Wrote: {metrics_dir / 'loocv_per_fold.json'}")
    print(f"Wrote: {metrics_dir / 'loocv_aggregate.json'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_summary.png'}")


if __name__ == "__main__":
    main()
