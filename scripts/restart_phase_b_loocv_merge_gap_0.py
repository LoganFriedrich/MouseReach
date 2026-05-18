"""
v8 dev experiment: merge_gap=0 postprocess change.

Follows up the 2026-05-18 merge_gap sweep (which accepted mg=1 as the
primary candidate). mg=0 produced strictly better aggregate numbers in
that sweep but was a diagnostic REFERENCE -- not pre-declared as the
candidate, so it couldn't be accepted directly without violating the
no-tuning-on-eval rule.

This runner re-tests mg=0 as a properly pre-declared candidate with a
written rationale that is grounded in the evidence the sweep produced
(not derived from it post-hoc). If accepted, mg=0 supersedes mg=1 as
the cumulative best.

================================================================
PRE-EXPERIMENT CHECKLIST (per pre_experiment_checklist.md)
================================================================

1. Cumulative-stacking check (verified 2026-05-18):
   - Production v8.0.0 currently ships with BSW b=1, w=0.8 +
     merge_gap=2 (production default in
     `src/mousereach/reach/v8/__init__.py: DEFAULT_MERGE_GAP = 2`).
   - merge_gap=1 was ACCEPTED 2026-05-18 by the merge_gap sweep
     (snapshot `v8.0.0_dev_merge_gap_1/RESULTS.md` documents the
     decision). It has NOT been merged into production yet (paused
     before merge per Logan's instruction).
   - Cumulative-best comparison baseline for THIS experiment:
     BSW w=0.8 + merge_gap=1 LOOCV: TP=2049 / FP=300 / FN=326.
   - Pure-baseline reference: BSW w=0.8 + merge_gap=2 (current
     production): TP=1935 / FP=330 / FN=440.
   - Stacked improvements applied: BSW b=1 w=0.8 inline. merge_gap is
     a single postprocess parameter, so mg=0 is a REPLACEMENT for the
     current value (2), not a composition on top of mg=1.

2. Existing-module-modification check:
   - Existing module code modified: NO. `probabilities_to_reaches`
     accepts `merge_gap` as a parameter; this runner passes
     `merge_gap=0` at the call site. No changes under src/mousereach/.
   - On ship: `DEFAULT_MERGE_GAP = 2 -> 0` in
     `src/mousereach/reach/v8/__init__.py` (deferred until acceptance
     + holdout generalization).

3. Assumption check (unverified hypotheses):
   - HYP (PRINCIPLED RATIONALE for mg=0 as candidate):
     v8.0.0's per-frame proba is saturated near 1.0 inside real reaches
     (per `v8.0.0_dev_per_reach_confidence_distribution`: median TP
     mean_proba = 0.999, 85% of detected reaches have mean_proba >=
     0.95). Therefore intra-reach proba dips below threshold are
     empirically rare. With dips rare, the merge_gap parameter does
     NOT need to be > 0 to "protect against intra-reach noise dips"
     -- there are essentially no such dips to protect against. mg=0
     is the strictest postprocess: any sub-threshold frame ends a
     run. This is principled if the underlying probability landscape
     has the saturated-near-1 character described.
   - HYP: mg=0 will recover the additional +28 TP / -28 FN that the
     sweep's mg=0 reference produced over mg=1.
     PROBABLE OUTCOME: same numbers since GBM training is
     deterministic (random_state=42). The point of this run is the
     pre-experiment commitment under the updated rationale, not new
     numbers.
   - HYP: mg=0 will preserve boundary precision (start_delta median
     and abs_median both 0, span_delta median 0). The sweep already
     showed this; this run confirms.
   - HYP: mg=0 does not introduce a new failure mode beyond the
     CNT0301_P3 FP regression observed in the sweep (+11 FPs on that
     specific chronic-overdetection video).

4. FN-direction-reporting check:
   - Planned RESULTS.md first line:
     "FN vs cumulative best (BSW w=0.8 + mg=1): [direction + magnitude];
      FN vs pure baseline (BSW w=0.8 + mg=2, current production):
      [direction + magnitude]."
   - Two-delta surfacing BEFORE any metric table.

5. Framework-not-adhoc check:
   - Output: `Improvement_Snapshots/reach_detection/v8.0.0_dev_merge_gap_0_candidate/`
     (distinct from the sweep's `v8.0.0_dev_merge_gap_0/` which was
     produced under a different pre-experiment commitment).
   - Canonical metrics layout + canonical figure runner.

6. Branch + tag check (deferred to user before run):
   - Tag: `v8-pre-merge-gap-0-2026-05-18`
   - Branch: `feature/v8-merge-gap-0`

7. Decision rule check (vs cumulative best = BSW w=0.8 + mg=1):
   - REJECT if TP drops AND FN rises (vs cumulative best mg=1:
     TP=2049, FN=326).
   - REJECT if exact-frame-start match rate drops > 0.3 pp (vs mg=1).
   - ACCEPT if FN drops or TP rises with exact_start preserved.
   - DO NOT retune merge_gap or any other parameter in response to
     this run's results. If mg=0 doesn't pass, that's the result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

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


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)

SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_merge_gap_0_candidate"
)


# ---------------------------------------------------------------------------
# Stacked-improvement params (BSW b=1 w=0.8) + the change
# ---------------------------------------------------------------------------

BOUNDARY_BUFFER = 1
BOUNDARY_WEIGHT = 0.8

THRESHOLD = 0.5
MERGE_GAP = 0                # <-- THE CHANGE (was 2 in production, 1 just accepted)
MIN_SPAN = 3


# ---------------------------------------------------------------------------
# Cumulative-stacking: BSW boundary weights (copied verbatim from the BSW runner)
# ---------------------------------------------------------------------------

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
    return weights_series.reindex(train_df.index).to_numpy()


# ---------------------------------------------------------------------------
# Per-fold: train BSW, predict, evaluate at mg=0
# ---------------------------------------------------------------------------

def train_predict_and_eval(
    train_pool_df: pd.DataFrame,
    train_video_ids: List[str],
    val_vid: str,
    feat_cols: List[str],
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
    for ri in sorted(set(rid[rid >= 0].tolist())):
        m = rid == ri
        f = frames[m]
        gt_reaches.append(GTReach(
            start_frame=int(f.min()), end_frame=int(f.max()),
            video_id=val_vid, index=ri))

    results = evaluate_reaches(algo_reaches, gt_reaches, video_id=val_vid)
    summary = summarize_results(results)
    return summary, results, algo_reaches, gt_reaches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print(f"PHASE B LOOCV -- merge_gap=0 candidate on top of BSW b=1 w=0.8")
    print(f"Comparison baseline: BSW w=0.8 + mg=1 (cumulative best, accepted 2026-05-18)")
    print("=" * 78)
    print()

    print("Loading train_pool.parquet ...", flush=True)
    df = pd.read_parquet(CORPUS_DIR / "phase_b_dataset" / "train_pool.parquet")
    folds_def = json.loads(
        (CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))
    train_pool_ids = folds_def["train_pool"]["video_ids"]
    print(f"  Train pool: {len(train_pool_ids)} videos", flush=True)
    print()

    feat_cols = feature_columns()
    exh_set = set(df.loc[df["exhaustive"], "video_id"].unique().tolist())
    eligible_val = [v for v in train_pool_ids if v in exh_set]
    print(f"LOOCV exhaustive folds: {len(eligible_val)}")
    print()

    folds = []
    per_video_data = {}
    all_results_combined = []

    for i, val_vid in enumerate(eligible_val):
        train_ids = [v for v in train_pool_ids if v != val_vid]
        print(f"  fold {i+1}/{len(eligible_val)}: val={val_vid}", flush=True)
        summary, results, algo_reaches, gt_reaches = \
            train_predict_and_eval(df, train_ids, val_vid, feat_cols)
        s = summary
        sd_mean = s['tp_start_delta']['mean']
        sd_mean_str = f"{sd_mean:.3f}" if sd_mean is not None else "n/a"
        print(f"    TP={s['n_tp']:>4} FP={s['n_fp']:>4} FN={s['n_fn']:>4}  "
              f"start_delta med={s['tp_start_delta']['median']} "
              f"abs_med={s['tp_start_delta']['abs_median']} "
              f"mean={sd_mean_str}  "
              f"span_delta med={s['tp_span_delta']['median']}",
              flush=True)
        folds.append({"val_video_ids": [val_vid], "summary": summary})
        per_video_data[val_vid] = (algo_reaches, gt_reaches)
        all_results_combined.extend(results)

    print()
    agg = summarize_results(all_results_combined)
    print("=" * 78)
    print(f"AGGREGATE LOOCV (BSW b=1 w=0.8 + merge_gap=0)")
    print("=" * 78)
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
    print("  Cumulative best (BSW + mg=1, accepted 2026-05-18):  TP=2049  FP=300  FN=326")
    print("  Pure baseline (BSW + mg=2, current production):      TP=1935  FP=330  FN=440")
    print()

    # Decision rule analysis
    cum_best_tp = 2049; cum_best_fp = 300; cum_best_fn = 326
    pure_tp = 1935; pure_fp = 330; pure_fn = 440
    d_tp_cum = agg['n_tp'] - cum_best_tp
    d_fp_cum = agg['n_fp'] - cum_best_fp
    d_fn_cum = agg['n_fn'] - cum_best_fn
    d_tp_pure = agg['n_tp'] - pure_tp
    d_fp_pure = agg['n_fp'] - pure_fp
    d_fn_pure = agg['n_fn'] - pure_fn

    print("=" * 78)
    print("DELTAS")
    print("=" * 78)
    print(f"  vs cumulative best (BSW + mg=1):  TP {d_tp_cum:+d}  FP {d_fp_cum:+d}  FN {d_fn_cum:+d}")
    print(f"  vs pure baseline (BSW + mg=2):     TP {d_tp_pure:+d}  FP {d_fp_pure:+d}  FN {d_fn_pure:+d}")
    print()
    tp_drops = d_tp_cum < 0
    fn_rises = d_fn_cum > 0
    if tp_drops and fn_rises:
        print("DECISION RULE: REJECT (TP drops AND FN rises vs cumulative best)")
    elif d_fn_cum < 0 or d_tp_cum > 0:
        print("DECISION RULE: candidate for ACCEPT (FN drops or TP rises vs cumulative best)")
        print("              -- verify exact_start preserved before final decision")
    else:
        print("DECISION RULE: ambiguous (neither REJECT nor clean ACCEPT)")
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
        title_suffix=" (LOOCV, BSW w=0.8 + merge_gap=0 candidate)",
    )

    print(f"Wrote: {metrics_dir / 'loocv_per_fold.json'}")
    print(f"Wrote: {metrics_dir / 'loocv_aggregate.json'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_summary.png'}")
    print()
    print("REMINDER: write RESULTS.md leading with FN delta vs cumulative best AND")
    print("vs pure baseline, BEFORE any metric table.")


if __name__ == "__main__":
    main()
