"""
v8 dev experiment: phase B LOOCV with reverse-hysteresis thresholding
in post-processing.

Same training pipeline as baseline. Same merge_gap=2, min_span=3.
Only change: instead of `mask = proba > 0.5`, use asymmetric thresholds
where a run STARTS at proba > T_LOW (e.g. 0.4) but continues only
while proba > T_HIGH (e.g. 0.7) -- making runs harder to MAINTAIN
than to start.

Rationale (per RECALIBRATION_PLAYBOOK.md step 3):

  Target failure mode (from v8.0.0_dev_within_gt_fp_inspection,
  2026-05-04):
    The within_gt FPs (81.6% of v8 FPs) are NOT split-twins. The
    dominant pattern is "algo span extends past GT_end" -- the GBM's
    probability stays high through post-reach paw retraction motion
    that the human GT-labeler did NOT consider part of the reach. The
    same root cause produces the within_gt FPs and the tol_miss_span
    FNs (26.7% of FNs).

    In the inspected cases (e.g. 20251010_CNT0308_P2 gt_idx=27), the
    probability often has a small dip near GT_end (1-2 frames going
    down to ~0.2-0.7) but it gets bridged by merge_gap=2 so the run
    never terminates. The single algo run extends 14+ frames past
    GT_end.

  Mechanism of the proposed fix:
    Reverse hysteresis lowers the entry threshold (catches reaches
    that are about to start, including currently-tol-miss-late-start
    cases) but RAISES the exit threshold (so any dip below T_HIGH
    terminates the run, even if it doesn't go all the way to 0).

    For Pattern B (figure 2 in inspection): a small dip near GT_end
    that's currently bridged would now terminate the run, ending it
    much closer to GT_end. Span would shrink. Span_tol satisfied ->
    GT becomes TP. FP disappears.

    For Pattern A (figure 1, late-start): T_LOW=0.4 catches the run
    1-2 frames earlier than T_LOW=0.5 would, possibly squeezing the
    start_delta back inside +/-2.

    Risk: real long reaches with small mid-reach dips would now
    terminate prematurely. This was previously handled by merge_gap=2
    bridging dips. Reverse hysteresis essentially overrides merge_gap
    on the down side: once the run drops below T_HIGH, it ends, even
    if it would come back. May increase n_fn for legitimately long
    reaches.

  Why principled, not "tune until eval passes":
    The within_gt FP inspection showed concretely that merge_gap is
    bridging dips that should TERMINATE runs (because the dip
    represents the actual transition from reach to retraction).
    Reverse hysteresis is the symmetric mechanism that respects those
    dips as termination signals while allowing them to start runs at
    a lower bar (when the model is just getting confident).

  Parameter choice -- T_LOW=0.4, T_HIGH=0.7:
    - T_LOW=0.4: catches runs slightly earlier than baseline 0.5.
      Modest sensitivity bump.
    - T_HIGH=0.7: any dip below 0.7 terminates the run. High-but-not-
      perfect probability sustained needed.
    - Gap of 0.3 between thresholds: enough margin that small noise
      in the mid-0.5s won't terminate clean runs.

  Risk and guardrails:
    - n_fn rising: real long reaches getting prematurely terminated.
      Watch n_fn vs baseline (was 457 at baseline, 440 at BSW w=0.8).
    - Span tail compression to too-short: runs ending too early would
      show negative span_delta on TPs.
    - Boundary precision: T_LOW lower than 0.5 could shift start
      frames slightly earlier (negative start_delta).

  Expected eval deltas (vs v8.0.0_dev_initial_loocv baseline):
    - n_fp drops materially (within_gt FPs whose runs now terminate
      near GT_end recover into TPs)
    - n_tp rises (split-attributable cases recovered)
    - n_fn drops (some tol_miss_span FNs become TPs)
    - tp_start_delta: small leftward shift if T_LOW catches earlier
    - tp_span_delta: should center near 0 -- the experiment removes
      the systematic positive-span bias
    - exact_start, exact_span: should HOLD or RISE

  Decision rule (same as predecessors):
    Reject if TP drops AND FN rises (vs baseline).
    Reject if exact-frame-match rate drops materially.

NO existing module code modified. Replicates train_one_fold's
training+inference logic inline; replaces the threshold-and-mask
step with the asymmetric-hysteresis equivalent. merge_gap and
min_span filters applied after, same as before.

Output:
  Improvement_Snapshots/reach_detection/v8.0.0_dev_reverse_hysteresis_t04_t07/
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

T_LOW = 0.4   # entry threshold (lower than baseline 0.5; easier to start)
T_HIGH = 0.7  # exit threshold (higher than baseline 0.5; runs terminate sooner)
MERGE_GAP = 2
MIN_SPAN = 3
SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_reverse_hysteresis_t04_t07"
)


@dataclass
class ReachSpan:
    start_frame: int
    end_frame: int


def asymmetric_hysteresis_to_reaches(proba, t_low, t_high, merge_gap, min_span):
    """Convert per-frame probability to reach windows via asymmetric
    hysteresis.

    Run starts when proba > t_low (entered from below).
    Run continues while proba > t_high (else terminates).

    After mask construction, apply merge_gap (bridge close runs) and
    min_span (drop short runs) -- same filters as baseline.
    """
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

    # Find runs of 1s
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

    # Merge runs separated by <= merge_gap not-in-run frames
    if not runs:
        return []
    merged = [runs[0]]
    for r in runs[1:]:
        gap = r.start_frame - merged[-1].end_frame - 1
        if gap <= merge_gap:
            merged[-1] = ReachSpan(start_frame=merged[-1].start_frame, end_frame=r.end_frame)
        else:
            merged.append(r)

    # Drop short runs
    out = [r for r in merged if (r.end_frame - r.start_frame + 1) >= min_span]
    return out


def train_one_fold_inline(train_pool_df, train_video_ids, val_vid, feat_cols):
    """Replicates train_one_fold from mousereach.reach.v8.train, but
    overrides the inference postprocess step with asymmetric hysteresis.
    Returns (summary, results, algo_reaches, gt_reaches).
    """
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
        sample_weight = np.where(y_train == 1, w_pos, w_neg).astype(np.float32)
    else:
        sample_weight = None

    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=6,
        random_state=42, early_stopping=False,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    Xv = val[feat_cols].to_numpy(dtype=np.float32)
    proba = clf.predict_proba(Xv)[:, 1]

    algo_reaches_raw = asymmetric_hysteresis_to_reaches(
        proba, t_low=T_LOW, t_high=T_HIGH,
        merge_gap=MERGE_GAP, min_span=MIN_SPAN,
    )
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
    return summary, results, algo_reaches, gt_reaches


def main():
    print("=" * 70)
    print(f"PHASE B LOOCV (exhaustive subset) -- reverse hysteresis t_low={T_LOW} t_high={T_HIGH}")
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
            train_one_fold_inline(df, train_ids, val_vid, feat_cols)

        s = summary
        sd_mean = s['tp_start_delta']['mean']
        sd_mean_str = f"{sd_mean:.3f}" if sd_mean is not None else "n/a"
        print(f"    TP={s['n_tp']:>4} FP={s['n_fp']:>4} FN={s['n_fn']:>4}  "
              f"start_delta median={s['tp_start_delta']['median']} "
              f"abs_median={s['tp_start_delta']['abs_median']} "
              f"mean={sd_mean_str}  "
              f"span_delta median={s['tp_span_delta']['median']} "
              f"abs_median={s['tp_span_delta']['abs_median']}",
              flush=True)
        folds.append({"val_video_ids": [val_vid], "summary": summary})
        per_video_data[val_vid] = (algo_reaches, gt_reaches)
        all_results_combined.extend(results)

    print()
    agg = summarize_results(all_results_combined)
    print("=" * 70)
    print(f"AGGREGATE LOOCV RESULTS (t_low={T_LOW}, t_high={T_HIGH})")
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
    print("  Baseline:    TP=1918  FP=337  FN=457  exact_start=83.47%  span mean=0.212")
    print("  BSW w=0.8:   TP=1935  FP=330  FN=440  exact_start=84.08%  span mean=0.170")
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
            "t_low": T_LOW, "t_high": T_HIGH,
            "schema_version": "extended_with_frame_positions",
        }, indent=2), encoding="utf-8")

    render_v8_reach_figures(
        snapshot_dir=SNAPSHOT_DIR,
        raw_results=serialized_results,
        summary=agg,
        title_suffix=f" (LOOCV, reverse hysteresis t_low={T_LOW} t_high={T_HIGH})",
    )

    print(f"Wrote: {metrics_dir / 'loocv_per_fold.json'}")
    print(f"Wrote: {metrics_dir / 'loocv_aggregate.json'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_summary.png'}")


if __name__ == "__main__":
    main()
