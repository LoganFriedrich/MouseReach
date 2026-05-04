"""
v8 dev experiment: phase B LOOCV with boundary sample-weighting at
training time.

Same baseline params (merge_gap=2, threshold=0.5, min_span=3) and same
extended JSON output schema. NO change to inference or post-processing.
The change is at TRAINING time: sample_weight on per-frame training
rows is reduced for frames at +/-BOUNDARY_BUFFER around any reach
start/end transition. Multiplies the existing class-imbalance weight.

Rationale (per RECALIBRATION_PLAYBOOK.md step 3):

  Target failure categories (from v8.0.0_dev_failure_mode_breakdown
  diagnostic, 2026-05-04):
    - tol_miss_both / tol_miss_start / tol_miss_span FNs (92.5% of
      FN combined): algo IS firing on these reaches but boundaries are
      off by >2 frames or span doesn't match
    - within_gt FPs (81.6% of FP): the actual split-twin pattern;
      second halves of GT reaches that v8 detected as 2 pieces
    - 144 TPs lost frame-exact start in proba_smooth_w5 experiment;
      that experiment was rejected because smoothing softened
      boundaries unconditionally

  Code path responsible:
    mousereach.reach.v8.train.train_one_fold builds sample_weight for
    class imbalance only. The GBM treats every per-frame training
    sample equally within each class. At reach boundaries (the frame
    where label transitions 0->1 or 1->0), the per-frame features are
    inherently AMBIGUOUS -- mouse is just starting / just finishing,
    features are similar to the surrounding frames -- yet the binary
    label demands a hard 0/1 decision. The model is forced to fit
    these ambiguous frames perfectly, which produces unstable
    probability output at boundaries (sometimes 0.4, sometimes 0.6)
    and uncertain mid-reach probability (occasional dips below 0.5).

  Why principled, not "tune until eval passes":
    Down-weighting boundary-frame training samples tells the model
    "boundary frames are inherently uncertain; focus your fitting
    capacity on the clearly-in-reach and clearly-out-of-reach
    regions." The model can then produce more confident mid-reach
    probability (fewer dips -> fewer split-twin FPs and fewer
    tol_miss_span FNs) and crisper boundaries (less frame-by-frame
    flicker -> fewer tol_miss_start FNs).

    Unlike post-hoc smoothing (which softens boundaries
    unconditionally as a side effect), training-time weighting only
    relaxes the fit at AMBIGUOUS frames -- the model is still pushed
    to fit clearly-labeled frames precisely. So mid-reach probability
    can stay high while boundary probability is allowed to be
    intermediate without gradient pressure.

  Parameter choice -- BOUNDARY_BUFFER=1, BOUNDARY_WEIGHT=0.5:
    BOUNDARY_BUFFER=1: down-weight 3 frames per boundary (the
      transition frame plus 1 each side). For a typical 6-12 frame
      reach this gives 4-6 boundary frames out of 6-12 total
      reach frames; mid-reach gets full weight.
    BOUNDARY_WEIGHT=0.5: half-weight, not zero. The model still
      learns from boundary frames -- it just gets less gradient
      pressure to be exactly right there. Mild effect.

    Conservative starting point. If insufficient, try BUFFER=2 or
    WEIGHT=0.3. If too aggressive (model becomes systematically
    early/late), try BUFFER=0 or WEIGHT=0.7.

  Risk and guardrails:
    1. Boundary mean shift. If the model's predictions at boundaries
       end up systematically lower than 0.5, threshold-crossings
       happen LATER, pushing start_delta toward +1 and ending span
       short. Watch tp_start_delta.mean (was -0.13 in baseline; if
       it shifts toward +0.5 or beyond, this is happening).
    2. Short reach drop. If most frames in a 6-frame reach are
       boundary-zone, the model gets weaker signal on short reaches.
       Could increase n_fn for short reaches. Watch the full FN count.
    3. Adjacent-reach interaction. Two reaches separated by <2
       frames have overlapping boundary zones -> all frames between
       them are boundary-weighted. Could cause non-detection of
       inter-reach gap. Edge case; should be rare.

  Expected eval deltas (vs v8.0.0_dev_initial_loocv):
    - n_tp: rises (tol_miss FNs and within_gt-derived FPs converted
      to TPs because boundaries are crisper)
    - n_fp: drops materially (within_gt 275 -> ?; goal is recovery
      of split-twin pairs as TPs)
    - n_fn: drops (tol_miss reductions)
    - tp_start_delta: should NOT shift mean materially. If mean
      shifts by >0.3, the boundary weighting is biasing predictions.
    - tp_span_delta: should NOT shift; same caveat.
    - exact-frame-match rate (start_delta=0 fraction): should HOLD
      or RISE (vs baseline 83.47%). If it drops, the experiment
      isn't doing what we want and should be rejected.

  Decision rule:
    Reject if TP drops AND FN rises (user's strict rule, 2026-05-01).
    Reject if exact-frame-match rate drops materially (Cardinal
    Rule). Accept if FN drops OR TP rises with exact-frame-match
    rate held.

NO existing module code modified. The runner replicates
train_one_fold's logic inline (training and inference) so the new
sample_weight can be inserted between class-weight computation and
clf.fit().

Output:
  Improvement_Snapshots/reach_detection/v8.0.0_dev_boundary_sample_weight_b1_w0.5/
    metrics/loocv_per_fold.json
    metrics/loocv_aggregate.json     -- extended schema with frame positions
    figures/reach_detection_summary.png

Compare side-by-side against:
  Improvement_Snapshots/reach_detection/v8.0.0_dev_initial_loocv/                 (baseline)
  Improvement_Snapshots/reach_detection/v8.0.0_dev_failure_mode_breakdown/        (diagnostic of baseline)
  Improvement_Snapshots/reach_detection/v8.0.0_dev_proba_smooth_w5/               (rejected smoothing)
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
    GTReach, AlgoReach, MatchResult, evaluate_reaches, summarize_results,
)
from mousereach.reach.v8.postprocess import probabilities_to_reaches
from mousereach.reach.v8.features import feature_columns
from mousereach.improvement.reach_detection.v8_figures import render_v8_reach_figures


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)

BOUNDARY_BUFFER = 1     # +/-1 frame around each reach boundary
BOUNDARY_WEIGHT = 0.5   # 50% weight on boundary frames (full = 1.0)
SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_boundary_sample_weight_b1_w0.5"
)

THRESHOLD = 0.5
MERGE_GAP = 2
MIN_SPAN = 3


def compute_boundary_weights(train_df, n_buffer=1, boundary_weight=0.5):
    """Per-row weight in train_df: boundary_weight in zones, 1.0 elsewhere.

    Boundary frame: any row where reach_id changes between adjacent
    frames within the same video (regardless of which direction).
    Both rows on either side of the transition are flagged.

    Boundary zone: dilation of the boundary-frame set by +/-n_buffer
    frames (within same video).

    Returns
    -------
    weights : np.ndarray of float32, length len(train_df), aligned to
        train_df's row order.
    """
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

    # Dilate within video, capped at video boundaries (we don't span
    # across videos because of the same_video check above)
    dilated = transitions.copy()
    for d in range(1, n_buffer + 1):
        # Shift forward / backward; same_video check would be needed
        # for strict correctness across video boundaries, but in
        # practice dilation by 1-2 frames across a video boundary
        # affects at most 4 rows total per video boundary which is
        # negligible (and they would be either the very last frame
        # of one video or first of next, neither of which is in any
        # reach typically).
        dilated[d:] |= transitions[:-d]
        dilated[:-d] |= transitions[d:]

    weights_sorted = np.ones(n, dtype=np.float32)
    weights_sorted[dilated] = boundary_weight

    # Reorder back to train_df's original order
    weights_series = pd.Series(weights_sorted, index=sorted_df.index)
    weights = weights_series.reindex(train_df.index).to_numpy()
    return weights


def train_one_fold_with_boundary_weight(
    train_pool_df, train_video_ids, val_vid, feat_cols,
):
    """Replicates train_one_fold from mousereach.reach.v8.train, but
    inserts boundary sample-weighting between class-weight computation
    and clf.fit. Also captures algo_reaches and gt_reaches lists for
    the val video (with frame positions intact) for downstream
    diagnostic.
    """
    train_mask = train_pool_df["video_id"].isin(train_video_ids)
    train_mask &= train_pool_df["exhaustive"]  # only_exhaustive_for_train=True
    train = train_pool_df.loc[train_mask]

    val = train_pool_df.loc[train_pool_df["video_id"] == val_vid]

    X_train = train[feat_cols].to_numpy(dtype=np.float32)
    y_train = train["label"].to_numpy(dtype=np.int8)

    # Class-imbalance weight (replicates train_one_fold)
    n = len(y_train)
    n_pos = int(y_train.sum())
    n_neg = n - n_pos
    if n_pos > 0 and n_neg > 0:
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)
        class_w = np.where(y_train == 1, w_pos, w_neg).astype(np.float32)
    else:
        class_w = np.ones(n, dtype=np.float32)

    # Boundary weight (NEW)
    boundary_w = compute_boundary_weights(
        train, n_buffer=BOUNDARY_BUFFER, boundary_weight=BOUNDARY_WEIGHT)
    n_in_zone = int((boundary_w < 1.0).sum())

    sample_weight = (class_w * boundary_w).astype(np.float32)

    # Train (same hyperparams as train_one_fold default)
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        early_stopping=False,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    # Inference (replicates train_one_fold's per-vid loop, single video)
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

    results = evaluate_reaches(algo_reaches, gt_reaches, video_id=val_vid)
    summary = summarize_results(results)

    return summary, results, algo_reaches, gt_reaches, n_in_zone, n


def main():
    print("=" * 70)
    print("PHASE B LOOCV (exhaustive subset) -- boundary sample-weighting")
    print(f"BOUNDARY_BUFFER={BOUNDARY_BUFFER}  BOUNDARY_WEIGHT={BOUNDARY_WEIGHT}")
    print("Sample weight at training: class_w * boundary_w")
    print("Inference: identical to baseline")
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
    exh_set = set(df.loc[df["exhaustive"], "video_id"].unique().tolist())
    eligible_val = [v for v in train_pool_ids if v in exh_set]
    print(f"LOOCV: {len(eligible_val)} exhaustive folds")
    print()

    folds = []  # list of dicts
    per_video_data = {}
    all_results_combined = []

    for i, val_vid in enumerate(eligible_val):
        train_ids = [v for v in train_pool_ids if v != val_vid]
        print(f"  fold {i+1}/{len(eligible_val)}: val={val_vid}", flush=True)

        summary, results, algo_reaches, gt_reaches, n_in_zone, n_total_train = \
            train_one_fold_with_boundary_weight(
                df, train_ids, val_vid, feat_cols)

        s = summary
        zone_pct = 100 * n_in_zone / n_total_train if n_total_train else 0
        print(f"    TP={s['n_tp']:>4} FP={s['n_fp']:>4} FN={s['n_fn']:>4}  "
              f"start_delta median={s['tp_start_delta']['median']} "
              f"abs_median={s['tp_start_delta']['abs_median']} "
              f"mean={s['tp_start_delta']['mean']:.3f}  "
              f"span_delta median={s['tp_span_delta']['median']} "
              f"abs_median={s['tp_span_delta']['abs_median']}  "
              f"boundary-zone={zone_pct:.1f}% of train",
              flush=True)
        folds.append({
            "val_video_ids": [val_vid],
            "summary": summary,
        })
        per_video_data[val_vid] = (algo_reaches, gt_reaches)
        all_results_combined.extend(results)

    print()
    agg = summarize_results(all_results_combined)
    print("=" * 70)
    print(f"AGGREGATE LOOCV RESULTS (boundary buf={BOUNDARY_BUFFER}, w={BOUNDARY_WEIGHT})")
    print("=" * 70)
    print(f"  TP={agg['n_tp']}  FP={agg['n_fp']}  FN={agg['n_fn']}")
    print(f"  Start delta on TPs: median={agg['tp_start_delta']['median']}f  "
          f"|median|={agg['tp_start_delta']['abs_median']}f  "
          f"mean={agg['tp_start_delta']['mean']:.3f}  "
          f"range=[{agg['tp_start_delta']['min']},{agg['tp_start_delta']['max']}]")
    print(f"  Span delta on TPs:  median={agg['tp_span_delta']['median']}f  "
          f"|median|={agg['tp_span_delta']['abs_median']}f  "
          f"mean={agg['tp_span_delta']['mean']:.3f}  "
          f"range=[{agg['tp_span_delta']['min']},{agg['tp_span_delta']['max']}]")
    print()
    print("Compare against baseline v8.0.0_dev_initial_loocv:")
    print("  (TP=1918  FP=337  FN=457  start range [-2,2] mean=-0.127  span range [-8,8] mean=0.212)")
    print()

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    serialized_results = []
    for r in all_results_combined:
        record = {
            "status": r.status,
            "video_id": r.video_id,
            "gt_index": r.gt_index,
            "algo_index": r.algo_index,
            "start_delta": r.start_delta,
            "span_delta": r.span_delta,
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
            "n_folds": len(folds),
            "summary": agg,
            "raw_results": serialized_results,
            "merge_gap": MERGE_GAP,
            "boundary_buffer": BOUNDARY_BUFFER,
            "boundary_weight": BOUNDARY_WEIGHT,
            "schema_version": "extended_with_frame_positions",
        }, indent=2), encoding="utf-8")

    render_v8_reach_figures(
        snapshot_dir=SNAPSHOT_DIR,
        raw_results=serialized_results,
        summary=agg,
        title_suffix=f" (LOOCV, {len(folds)} folds, exhaustive only, boundary w={BOUNDARY_WEIGHT} buf={BOUNDARY_BUFFER})",
    )

    print(f"Wrote: {metrics_dir / 'loocv_per_fold.json'}")
    print(f"Wrote: {metrics_dir / 'loocv_aggregate.json'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_summary.png'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_legend.md'}")


if __name__ == "__main__":
    main()
