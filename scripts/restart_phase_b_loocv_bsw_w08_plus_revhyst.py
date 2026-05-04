"""
v8 dev experiment: phase B LOOCV combining BSW w=0.8 + reverse hysteresis.

This is the proper composition test. Per the cumulative-stacking
convention (memory: v8_pending_integrations.md), every new v8
experiment must layer all accepted prior improvements onto the new
change. The reverse-hysteresis experiment that ran against pure
baseline (v8.0.0_dev_reverse_hysteresis_t04_t07) was out-of-protocol.

Stacked improvements applied:
  1. Boundary sample-weighting at training (BSW b=1 w=0.8) -- ACCEPTED
     2026-05-04 in merge 529f688. From v8_pending_integrations.md.
  2. Reverse-hysteresis at postprocess (t_low=0.4, t_high=0.7) -- the
     NEW change being tested.

Comparison baseline: BSW w=0.8 alone (the current cumulative best).
Reject if TP drops AND FN rises VS BSW w=0.8 (NOT vs pure baseline).
Reject if exact-frame-match rate drops materially vs BSW w=0.8.

Rationale recap (from v8.0.0_dev_within_gt_fp_inspection):
  within_gt FPs come from algo runs extending past GT_end. Reverse
  hysteresis terminates runs whenever proba dips below T_HIGH=0.7.
  When composed with BSW w=0.8 (which already produces crisper
  boundaries via training-time weighting), the hysteresis termination
  may be more reliable because the model's probability output is
  already cleaner.

Possible outcomes:
  A) Composition is additive: BSW gain + revhyst gain. Best result.
  B) BSW + revhyst is similar to BSW alone: revhyst's effect is
     redundant with what BSW already accomplished. Reject revhyst.
  C) BSW + revhyst regresses vs BSW alone: the changes interact
     negatively. Reject revhyst (the lever doesn't compose).

NO existing module code modified.

Output:
  Improvement_Snapshots/reach_detection/v8.0.0_dev_bsw_w08_plus_revhyst_t04_t07/
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.eval import (
    GTReach, AlgoReach, evaluate_reaches, summarize_results,
)
from mousereach.reach.v8.features import feature_columns
from mousereach.improvement.reach_detection.v8_figures import render_v8_reach_figures


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)

# Accepted prior improvement: BSW b=1 w=0.8
BOUNDARY_BUFFER = 1
BOUNDARY_WEIGHT = 0.8

# New change: reverse hysteresis t_low=0.4 t_high=0.7
T_LOW = 0.4
T_HIGH = 0.7

MERGE_GAP = 2
MIN_SPAN = 3

SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_bsw_w08_plus_revhyst_t04_t07"
)


@dataclass
class ReachSpan:
    start_frame: int
    end_frame: int


def compute_boundary_weights(train_df, n_buffer=1, boundary_weight=0.5):
    """BSW: per-row sample-weight multiplier; reduced near reach boundaries."""
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


def asymmetric_hysteresis_to_reaches(proba, t_low, t_high, merge_gap, min_span):
    """Postprocess via asymmetric hysteresis instead of fixed threshold."""
    n = len(proba)
    if n == 0:
        return []
    mask = np.zeros(n, dtype=np.int8)
    in_run = False
    for i in range(n):
        p = proba[i]
        if not in_run:
            if p > t_low:
                in_run = True
        else:
            if p < t_high:
                in_run = False
        mask[i] = 1 if in_run else 0

    runs = []
    in_r = False
    s = 0
    for i, v in enumerate(mask):
        if v and not in_r:
            in_r = True
            s = i
        elif not v and in_r:
            in_r = False
            runs.append(ReachSpan(start_frame=s, end_frame=i - 1))
    if in_r:
        runs.append(ReachSpan(start_frame=s, end_frame=n - 1))

    if not runs:
        return []
    merged = [runs[0]]
    for r in runs[1:]:
        gap = r.start_frame - merged[-1].end_frame - 1
        if gap <= merge_gap:
            merged[-1] = ReachSpan(start_frame=merged[-1].start_frame, end_frame=r.end_frame)
        else:
            merged.append(r)
    return [r for r in merged if (r.end_frame - r.start_frame + 1) >= min_span]


def train_one_fold_combined(train_pool_df, train_video_ids, val_vid, feat_cols):
    """Replicates train_one_fold with BSW (training) + reverse hysteresis (inference)."""
    train_mask = train_pool_df["video_id"].isin(train_video_ids) & train_pool_df["exhaustive"]
    train_mask &= train_pool_df["video_id"] != val_vid
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

    # BSW w=0.8 -- accepted prior improvement
    boundary_w = compute_boundary_weights(train, n_buffer=BOUNDARY_BUFFER, boundary_weight=BOUNDARY_WEIGHT)
    sample_weight = (class_w * boundary_w).astype(np.float32)

    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=6,
        random_state=42, early_stopping=False,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    Xv = val[feat_cols].to_numpy(dtype=np.float32)
    proba = clf.predict_proba(Xv)[:, 1]

    # Reverse hysteresis -- the new change being tested
    algo_reaches_raw = asymmetric_hysteresis_to_reaches(
        proba, t_low=T_LOW, t_high=T_HIGH,
        merge_gap=MERGE_GAP, min_span=MIN_SPAN,
    )
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
    return summary, results, algo_reaches, gt_reaches


def main():
    print("=" * 70)
    print("PHASE B LOOCV (exhaustive subset) -- BSW w=0.8 + reverse hysteresis")
    print(f"  BSW: BOUNDARY_BUFFER={BOUNDARY_BUFFER} BOUNDARY_WEIGHT={BOUNDARY_WEIGHT}")
    print(f"  RevHyst: T_LOW={T_LOW} T_HIGH={T_HIGH}")
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
        summary, results, algo_reaches, gt_reaches = \
            train_one_fold_combined(df, train_ids, val_vid, feat_cols)

        s = summary
        sd_mean = s['tp_start_delta']['mean']
        sd_mean_str = f"{sd_mean:.3f}" if sd_mean is not None else "n/a"
        print(f"    TP={s['n_tp']:>4} FP={s['n_fp']:>4} FN={s['n_fn']:>4}  "
              f"start_delta median={s['tp_start_delta']['median']} mean={sd_mean_str}  "
              f"span_delta median={s['tp_span_delta']['median']}",
              flush=True)
        folds.append({"val_video_ids": [val_vid], "summary": summary})
        per_video_data[val_vid] = (algo_reaches, gt_reaches)
        all_results_combined.extend(results)

    print()
    agg = summarize_results(all_results_combined)
    print("=" * 70)
    print(f"AGGREGATE LOOCV RESULTS (BSW w=0.8 + revhyst t_low={T_LOW} t_high={T_HIGH})")
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
    print("  Pure baseline:          TP=1918  FP=337  FN=457")
    print("  BSW w=0.8 alone (=current best):  TP=1935  FP=330  FN=440")
    print("  RevHyst alone (vs pure):           TP=1904  FP=356  FN=471")
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
            "t_low": T_LOW, "t_high": T_HIGH,
            "schema_version": "extended_with_frame_positions",
        }, indent=2), encoding="utf-8")

    render_v8_reach_figures(
        snapshot_dir=SNAPSHOT_DIR,
        raw_results=serialized_results,
        summary=agg,
        title_suffix=f" (LOOCV, BSW w={BOUNDARY_WEIGHT} + revhyst {T_LOW}/{T_HIGH})",
    )

    print(f"Wrote: {metrics_dir / 'loocv_per_fold.json'}")
    print(f"Wrote: {metrics_dir / 'loocv_aggregate.json'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_summary.png'}")


if __name__ == "__main__":
    main()
