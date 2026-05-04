"""
v8 dev experiment: phase B LOOCV with boundary sample-weighting at
training time -- milder variant.

Same approach as restart_phase_b_loocv_boundary_sample_weight.py
(BOUNDARY_BUFFER=1, BOUNDARY_WEIGHT=0.5) but with BOUNDARY_WEIGHT=0.7.
Same set of frames affected (3 frames per boundary), but each frame
gets weighted at 70% instead of 50% -- gentler down-weighting.

Rationale (per RECALIBRATION_PLAYBOOK.md step 3):

  Predecessor outcome (v8.0.0_dev_boundary_sample_weight_b1_w0.5):
    Soft reject. Strict rule passed (TP +9, FP -1, FN -9) but
    Cardinal Rule mildly violated (exact-match start 83.47% -> 81.79%
    = -1.68 pp; span 74.50% -> 73.02% = -1.48 pp). The within_gt FP
    hypothesis was not addressed (FP barely moved).

  This experiment characterizes the trade-off slope:
    - If w=0.7 produces (a) smaller FN reduction AND (b) smaller
      precision loss, the trade-off is roughly linear in weight;
      no value of w produces a Pareto-better outcome than w=0.5.
    - If w=0.7 produces (a) similar FN reduction WITH (b) smaller
      precision loss, then the boundary-weighting is non-linear and
      a milder weighting may be net-better.
    - If w=0.7 produces (a) larger FN reduction WITH (b) smaller
      precision loss, somehow even better than w=0.5, the previous
      experiment was over-aggressive.

  Same target failure modes as predecessor:
    - tol_miss FNs (92.5% of FN combined in the diagnostic baseline)
    - within_gt FPs (81.6% of FP) -- though now suspect this hypothesis
      is wrong since two boundary-related experiments haven't moved it

  Same risks as predecessor (boundary mean shift, short-reach drop,
  adjacent-reach interaction). All milder at w=0.7 because the
  weighting is gentler.

  Decision rule (same as predecessor):
    Reject if TP drops AND FN rises.
    Reject if exact-frame-match rate drops materially (Cardinal Rule).

NO existing module code modified. Replicates train_one_fold's
training+inference logic inline.

Output:
  Improvement_Snapshots/reach_detection/v8.0.0_dev_boundary_sample_weight_b1_w0.7/

Compare against:
  v8.0.0_dev_initial_loocv/                          (baseline)
  v8.0.0_dev_boundary_sample_weight_b1_w0.5/         (the w=0.5 sibling)
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
BOUNDARY_WEIGHT = 0.7   # the milder variant (was 0.5 in the previous experiment)
SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_boundary_sample_weight_b1_w0.7"
)

THRESHOLD = 0.5
MERGE_GAP = 2
MIN_SPAN = 3


def compute_boundary_weights(train_df, n_buffer=1, boundary_weight=0.5):
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
    weights = weights_series.reindex(train_df.index).to_numpy()
    return weights


def train_one_fold_with_boundary_weight(
    train_pool_df, train_video_ids, val_vid, feat_cols,
):
    train_mask = train_pool_df["video_id"].isin(train_video_ids)
    train_mask &= train_pool_df["exhaustive"]
    train = train_pool_df.loc[train_mask]

    val = train_pool_df.loc[train_pool_df["video_id"] == val_vid]

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

    boundary_w = compute_boundary_weights(
        train, n_buffer=BOUNDARY_BUFFER, boundary_weight=BOUNDARY_WEIGHT)
    n_in_zone = int((boundary_w < 1.0).sum())

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
        AlgoReach(start_frame=r.start_frame, end_frame=r.end_frame,
                  video_id=val_vid, index=i)
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

    return summary, results, algo_reaches, gt_reaches, n_in_zone, n


def main():
    print("=" * 70)
    print("PHASE B LOOCV (exhaustive subset) -- boundary sample-weighting (milder)")
    print(f"BOUNDARY_BUFFER={BOUNDARY_BUFFER}  BOUNDARY_WEIGHT={BOUNDARY_WEIGHT}")
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

    folds = []
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
        sd_mean = s['tp_start_delta']['mean']
        sd_mean_str = f"{sd_mean:.3f}" if sd_mean is not None else "n/a"
        print(f"    TP={s['n_tp']:>4} FP={s['n_fp']:>4} FN={s['n_fn']:>4}  "
              f"start_delta median={s['tp_start_delta']['median']} "
              f"abs_median={s['tp_start_delta']['abs_median']} "
              f"mean={sd_mean_str}  "
              f"span_delta median={s['tp_span_delta']['median']} "
              f"abs_median={s['tp_span_delta']['abs_median']}  "
              f"boundary-zone={zone_pct:.1f}% of train",
              flush=True)
        folds.append({"val_video_ids": [val_vid], "summary": summary})
        per_video_data[val_vid] = (algo_reaches, gt_reaches)
        all_results_combined.extend(results)

    print()
    agg = summarize_results(all_results_combined)
    print("=" * 70)
    print(f"AGGREGATE LOOCV RESULTS (boundary buf={BOUNDARY_BUFFER}, w={BOUNDARY_WEIGHT})")
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
    print("  Baseline:       TP=1918  FP=337  FN=457  start mean=-0.127  span mean=0.212")
    print("  BSW b1 w0.5:    TP=1927  FP=336  FN=448  start mean=-0.142  span mean=0.244")
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
            "boundary_buffer": BOUNDARY_BUFFER,
            "boundary_weight": BOUNDARY_WEIGHT,
            "schema_version": "extended_with_frame_positions",
        }, indent=2), encoding="utf-8")

    render_v8_reach_figures(
        snapshot_dir=SNAPSHOT_DIR,
        raw_results=serialized_results,
        summary=agg,
        title_suffix=f" (LOOCV, boundary w={BOUNDARY_WEIGHT} buf={BOUNDARY_BUFFER})",
    )

    print(f"Wrote: {metrics_dir / 'loocv_per_fold.json'}")
    print(f"Wrote: {metrics_dir / 'loocv_aggregate.json'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_summary.png'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_legend.md'}")


if __name__ == "__main__":
    main()
