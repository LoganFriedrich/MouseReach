"""
v8 dev experiment: phase B LOOCV with asymmetric label tightening at
the END of each reach, layered on BSW w=0.8.

==============================================================================
PRE-EXPERIMENT CHECKLIST (per pre_experiment_checklist.md, walked 2026-05-04)
==============================================================================

1. Cumulative-stacking check (DYNAMIC, verified at experiment time):
   - Read v8_pending_integrations.md: BSW b=1 w=0.8 listed ACTIVE,
     verified 2026-05-04.
   - Snapshot RESULTS.md (v8.0.0_dev_boundary_sample_weight_b1_w0.8/):
     status ACCEPTED [verified 2026-05-04].
   - git log on master since 529f688 acceptance: no revert / retract
     commits found [verified 2026-05-04].
   - train_one_fold inspection: still uses class-imbalance-only
     sample_weight (BSW not yet integrated into production code, so
     still pending) [verified 2026-05-04].
   - Recent merge commits with "ACCEPTED" tag: only 529f688 (BSW),
     no others [verified 2026-05-04].

   Stacked improvements applied: BSW b=1 w=0.8 + new asymmetric label
   tightening (END only).
   Comparison baseline (cumulative best): v8.0.0_dev_boundary_sample_weight_b1_w0.8/
     (TP=1935, FP=330, FN=440, exact_start=84.08%, exact_span=75.09%)
   Reference baseline (pure): v8.0.0_dev_initial_loocv/
     (TP=1918, FP=337, FN=457)
   Verification method: snapshot RESULTS.md + git log + train_one_fold inspection.

2. Existing-code-modification check:
   Existing module code modified: NO. Runner replicates train_one_fold
   inline. Label modification done inline on the parquet's label column
   (not on the source labels.py module).

3. Assumption check (unverified hypotheses flagged):
   - HYPOTHESIS: training with the last K frames of each reach labeled
     0 (instead of 1) will teach the model to drop probability K
     frames before GT_end. UNVERIFIED -- this is the central claim
     being tested.
   - HYPOTHESIS: K=1 is enough to shift threshold-crossing measurably
     without hurting short reaches. K=2 might be more effective but
     also riskier. Starting with K=1 per "small principled edits"
     playbook rule. UNVERIFIED.
   - HYPOTHESIS: leaving reach_id unchanged (only modifying label)
     means GT extraction at eval time still uses the full original
     reach windows. Verified by reading train_one_fold's GT extraction
     logic -- it builds GT from reach_id and frame, not label. CONFIRMED.
   - HYPOTHESIS: BSW boundary detection from reach_id (unchanged)
     composes correctly with label-modified training. The cumulative
     best uses reach_id-based BSW; applying it identically here.
     UNVERIFIED whether this is the right composition direction.

4. FN-direction-reporting check:
   The results report will lead with FN delta vs BSW w=0.8 (cumulative
   best) AND vs pure baseline. Per protocol updated 2026-05-04.

5. Framework-not-adhoc check:
   Output goes to canonical Improvement_Snapshots layout. Uses extended
   schema (per-event frame positions). render_v8_reach_figures used.

6. Branch + tag check:
   Pre-experiment tag: v8-pre-asymmetric-label-tightening-2026-05-04
   Branch: feature/v8-asymmetric-label-tightening-end1

7. Decision-rule check:
   Reject if TP drops AND FN rises vs BSW w=0.8 (cumulative best).
   Reject if exact-frame-match rate drops materially vs BSW w=0.8.
   Accept if FN drops OR TP rises with exact-match held.

==============================================================================
RATIONALE
==============================================================================

Target failure mode (from v8.0.0_dev_failure_mode_breakdown_on_bsw_w08):
  - within_gt FPs: 270/330 = 81.8% of all FPs
  - tol_miss_span FNs: 115/440 = 26.1% of all FNs
  - tol_miss_both FNs: 172/440 = 39.1% (many of these likely have
    span issues alongside start issues)
  Combined: ~half of all FN+FP events come from algo span not
  matching GT span -- specifically, algo runs extending past GT_end
  through paw retraction motion (Pattern B per the within_gt
  inspection).

The mechanism (verified by inspection): the GBM's per-frame
probability stays high past GT_end because per-frame features look
similar during reach extension and reach retraction. Postprocess
fixes (merge_gap, smoothing, hysteresis) all rejected -- the
probability has no termination signal for them to catch. A single
derivative feature (pellet-dist-diff) also rejected -- redundant with
existing velocities.

This experiment teaches the model to DROP probability K frames before
GT_end by training with tightened labels:
  Original: in_reach=1 for [GT_start, GT_end] (inclusive)
  Modified: in_reach=1 for [GT_start, GT_end - K] (inclusive)
  Frames [GT_end - K + 1, GT_end] now labeled 0.

The model learns "the last K frames of GT are NOT in-reach." At
inference, the threshold-crossing happens K frames earlier than it
would otherwise, terminating the algo run closer to actual GT_end.

Why principled, not "tune until eval passes":
The probability output IS the model's learned function of features.
The model can't know reach has ended unless it's TAUGHT. Postprocess
can't add information that's not in the probability. Features (like
pellet-dist-diff) are redundant with existing velocity signals.
Modifying the LABELS is the only mechanism that gives the model a
direct supervision signal "stop firing before GT_end" -- it's the
information-theoretic missing piece.

Why end-only, not symmetric:
The within_gt FP / tol_miss_span FN problem is specifically about
algo extending PAST GT_end. The start boundary problem (tol_miss_start
= 122 FNs) is a different mechanism (model fires LATE at start). A
symmetric tightening would hurt the start side without addressing it.
End-only targets the dominant failure mode.

K=1 starting choice:
- Gentlest change: 1 frame per reach end gets label-flipped.
- For typical 6-12 frame reaches, removes ~10% of in-reach frames.
- Short reaches (<= 4 frames) are skipped to avoid losing all labels.
- If K=1 helps, follow up with K=2 in a separate experiment.

Risks:
1. Model could just shift its entire probability profile earlier,
   making algo runs systematically end early -> negative span_delta
   tail. Watch for tp_span_delta mean shifting from +0.170 (BSW w=0.8)
   toward -1 or worse.
2. Short reaches that escape the "skip if <= 4 frames" filter could
   end up with label=1 for only 1-2 frames, training the model to
   dismiss them entirely. Watch for FN rising on short reaches.
3. The asymmetry (modify END but not START) might confuse the model;
   it could learn weird patterns. Watch for boundary-precision drop
   on the start axis (sd0 dropping vs cumulative best).

Expected eval deltas (vs BSW w=0.8 cumulative best):
- n_fp drops materially (within_gt FPs whose algo runs now terminate
  near GT_end recover into TPs)
- n_tp rises
- n_fn drops (tol_miss_span FNs become TPs)
- tp_span_delta mean shifts toward 0 (currently +0.170; should
  shrink to near 0 or modestly negative)
- tp_start_delta unchanged (we're only modifying end)
- exact_start: held or rises
- exact_span: rises (algo span better matches GT span on average)

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

# New change: asymmetric end label tightening
END_TIGHTEN_FRAMES = 1   # K -- number of frames to flip from 1 to 0 at reach end
MIN_REACH_FOR_TIGHTEN = 5  # skip tightening if reach span <= this

THRESHOLD = 0.5
MERGE_GAP = 2
MIN_SPAN = 3

SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_asymmetric_label_tightening_end1"
)


def tighten_labels_at_reach_end(df, k=1, min_reach_for_tighten=5):
    """Modify the label column: for each reach (group by video, reach_id),
    flip the LAST k frames from 1 to 0. Skip reaches with span <=
    min_reach_for_tighten to avoid over-tightening short reaches.

    reach_id is NOT modified -- GT extraction at eval time uses reach_id
    and frame, so the original GT windows remain intact.

    Returns a copy of df with modified label column.
    """
    out = df.copy()
    n_reaches_tightened = 0
    n_reaches_skipped_short = 0
    n_frames_flipped = 0

    # Group by (video_id, reach_id), only for reach_id >= 0 (in a reach)
    in_reach = out[out["reach_id"] >= 0]
    for (vid, rid), group in in_reach.groupby(["video_id", "reach_id"], sort=False):
        sorted_group = group.sort_values("frame")
        idx_sorted = sorted_group.index.to_numpy()
        span = len(idx_sorted)
        if span <= min_reach_for_tighten:
            n_reaches_skipped_short += 1
            continue
        # Flip the last k frames' labels from 1 to 0
        end_idx = idx_sorted[-k:]
        out.loc[end_idx, "label"] = np.int8(0)
        n_reaches_tightened += 1
        n_frames_flipped += k

    return out, {
        "n_reaches_tightened": n_reaches_tightened,
        "n_reaches_skipped_short": n_reaches_skipped_short,
        "n_frames_flipped": n_frames_flipped,
    }


def compute_boundary_weights(train_df, n_buffer=1, boundary_weight=0.5):
    """BSW: per-row sample-weight multiplier; reduced near reach boundaries.
    Computed from reach_id (unchanged from cumulative-best convention)."""
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


def train_one_fold_combined(train_pool_df_modified, train_video_ids, val_vid, feat_cols):
    """Replicates train_one_fold + BSW + label-tightening (the labels in
    train_pool_df_modified are already tightened; this function just
    consumes them).
    """
    train_mask = train_pool_df_modified["video_id"].isin(train_video_ids) & train_pool_df_modified["exhaustive"]
    train_mask &= train_pool_df_modified["video_id"] != val_vid
    train = train_pool_df_modified.loc[train_mask]
    val = train_pool_df_modified.loc[train_pool_df_modified["video_id"] == val_vid]

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
    sample_weight = (class_w * boundary_w).astype(np.float32)

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

    # GT reaches built from reach_id (unchanged) and frame -- so even though
    # the label column has been tightened, the GT windows for evaluation
    # are still the FULL original reach windows from the parquet's
    # reach_id annotation.
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
    return summary, results, algo_reaches, gt_reaches


def main():
    print("=" * 70)
    print("PHASE B LOOCV (exhaustive subset)")
    print(f"  Stacked: BSW b={BOUNDARY_BUFFER} w={BOUNDARY_WEIGHT}")
    print(f"  New: asymmetric label tightening end={END_TIGHTEN_FRAMES} frame(s),")
    print(f"       skip if reach span <= {MIN_REACH_FOR_TIGHTEN}")
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

    print(f"Tightening labels: flipping last {END_TIGHTEN_FRAMES} frame(s) per reach to 0 ...",
          flush=True)
    df_modified, stats = tighten_labels_at_reach_end(
        df, k=END_TIGHTEN_FRAMES, min_reach_for_tighten=MIN_REACH_FOR_TIGHTEN)
    print(f"  reaches tightened:        {stats['n_reaches_tightened']}")
    print(f"  reaches skipped (short):  {stats['n_reaches_skipped_short']}")
    print(f"  total frames flipped 1->0: {stats['n_frames_flipped']}")
    pre_n_pos = int(df["label"].sum())
    post_n_pos = int(df_modified["label"].sum())
    print(f"  total in-reach frames:    {pre_n_pos} -> {post_n_pos} "
          f"(delta {post_n_pos - pre_n_pos})")
    print()

    feat_cols = feature_columns()
    exh_set = set(df_modified.loc[df_modified["exhaustive"], "video_id"].unique().tolist())
    eligible_val = [v for v in train_pool_ids if v in exh_set]
    print(f"LOOCV: {len(eligible_val)} exhaustive folds")
    print()

    folds = []
    per_video_data = {}
    all_results_combined = []

    for i, val_vid in enumerate(eligible_val):
        train_ids = [v for v in train_pool_ids if v != val_vid]
        print(f"  fold {i+1}/{len(eligible_val)}: val={val_vid}", flush=True)
        summary, results, algo_reaches, gt_reaches = \
            train_one_fold_combined(df_modified, train_ids, val_vid, feat_cols)

        s = summary
        sd_mean = s['tp_start_delta']['mean']
        sd_mean_str = f"{sd_mean:.3f}" if sd_mean is not None else "n/a"
        sp_mean = s['tp_span_delta']['mean']
        sp_mean_str = f"{sp_mean:.3f}" if sp_mean is not None else "n/a"
        print(f"    TP={s['n_tp']:>4} FP={s['n_fp']:>4} FN={s['n_fn']:>4}  "
              f"start_delta median={s['tp_start_delta']['median']} mean={sd_mean_str}  "
              f"span_delta median={s['tp_span_delta']['median']} mean={sp_mean_str}",
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
            "end_tighten_frames": END_TIGHTEN_FRAMES,
            "min_reach_for_tighten": MIN_REACH_FOR_TIGHTEN,
            "label_tightening_stats": stats,
            "schema_version": "extended_with_frame_positions",
        }, indent=2), encoding="utf-8")

    render_v8_reach_figures(
        snapshot_dir=SNAPSHOT_DIR,
        raw_results=serialized_results,
        summary=agg,
        title_suffix=f" (LOOCV, BSW w={BOUNDARY_WEIGHT} + end-tighten K={END_TIGHTEN_FRAMES})",
    )

    print(f"Wrote: {metrics_dir / 'loocv_per_fold.json'}")
    print(f"Wrote: {metrics_dir / 'loocv_aggregate.json'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_summary.png'}")


if __name__ == "__main__":
    main()
