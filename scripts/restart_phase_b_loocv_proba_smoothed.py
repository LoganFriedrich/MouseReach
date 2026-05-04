"""
v8 dev experiment: phase B LOOCV with per-frame probability smoothing
applied BEFORE thresholding.

Same baseline params (merge_gap=2, threshold=0.5, min_span=3) and same
extended JSON output schema as the failure-mode-breakdown LOOCV. Only
change: a centered rolling-mean filter of width PROBA_SMOOTH_WINDOW
is applied to the per-frame probability array between
clf.predict_proba(...) and probabilities_to_reaches(...).

Rationale (per RECALIBRATION_PLAYBOOK.md step 3):

  Target failure categories (from v8.0.0_dev_failure_mode_breakdown
  diagnostic, 2026-05-04):
    - tol_miss_both (180/457 FN = 39.4%)
    - tol_miss_start (123/457 FN = 26.9%)
    - tol_miss_span  (122/457 FN = 26.7%)
    - within_gt FPs  (275/337 FP = 81.6%)  -- the actual split-twin
                     pattern

  Code path responsible:
    The GBM (mousereach.reach.v8.train.train_one_fold internal loop)
    produces a per-frame in-reach probability array. That array is
    passed directly to probabilities_to_reaches() in
    mousereach.reach.v8.postprocess. Mid-reach dips below threshold
    cause splits (within_gt FPs); slow boundary ramps cause
    tol_miss_start / tol_miss_span FNs.

  Why principled, not "tune until eval passes":
    Mouse reaches are physically continuous events lasting ~6-12
    frames. The GBM predicts independently per-frame, so its
    probability series can fluctuate frame-to-frame even when the
    underlying reach is contiguous. A small-window rolling-mean
    smoothing IS the temporal regularization the per-frame model
    lacks -- it imposes the prior that "probability shouldn't change
    abruptly within ~5 frames" which matches the physical timescale
    of reach motion.

    Unlike merge_gap (which only bridges below-threshold gaps in a
    binary mask), proba smoothing operates on the continuous
    probability series so it: (a) bridges dips that don't quite reach
    full below-threshold, (b) smooths boundary ramps so the threshold
    crossing happens at a more stable frame.

  Window choice -- start at PROBA_SMOOTH_WINDOW=5 (centered, 5-frame
  rolling mean):
    - bridges 1-2 frame mid-reach dips reliably
    - moderate effect on boundary precision (averaged over ~2 frames
      either side)
    - unlikely to wash out 6-12-frame real reaches entirely
    - leaves room for follow-up at W=7 if W=5 helps but partially

  Risk and guardrail:
    1. Boundary error tail extension. Smoothing the probability ramp
       can shift the threshold crossing 1-2 frames earlier (boundary
       gets "soft"), pushing start_delta out of +/-2 tolerance and
       converting clean TPs into tol_miss_start FNs. Watch the
       start_delta and span_delta distributions on TPs.
    2. Reach fusion. If two adjacent real reaches are separated by
       <5 frames, smoothing could fuse them. Span_delta tail watch
       (was [-8, 8] in baseline; if it extends materially, fusion
       happening). This is the same risk as merge_gap=4 experienced.
    3. Short-reach drop. A 3-frame reach with proba=0.9 throughout
       gets smoothed to ~0.54 (averaged with surrounding zeros at
       boundaries). Should still exceed threshold=0.5 but the margin
       is narrow. Watch n_fn for short GT reaches.

  Expected eval deltas (vs v8.0.0_dev_initial_loocv baseline):
    - n_tp: rises (within_gt FPs become TPs because mid-reach dips
      no longer split; tol_miss_span FNs become TPs because the
      smoothed reach span better matches GT)
    - n_fp: drops materially (the 275 within_gt FPs are the target;
      if a substantial fraction are recovered as TPs, FP count drops
      proportionally)
    - n_fn: drops (tol_miss FNs recovered as TPs)
    - tp_start_delta: tail may EXTEND slightly (boundary softening
      from smoothing); watch for asymmetric shift indicating
      systematic early-/late-fire
    - tp_span_delta: tail may extend slightly; watch for shift toward
      positive (smoothing extends spans both ways equally, so
      span_delta should remain centered on 0)
    - n_perfect_TPs (start_delta=0 AND span_delta=0): may drop
      slightly because boundary precision gets softer

  Decision rule:
    Reject if TP drops AND FN rises (same criterion as merge_gap
    experiments). Accept if TP rises OR FN drops, AND boundary
    precision (|start_delta| <=2 fraction, |span_delta| <=2 fraction)
    on the matched-TP set doesn't drop materially.

Output:
  Improvement_Snapshots/reach_detection/v8.0.0_dev_proba_smooth_w5/
    metrics/
      loocv_per_fold.json
      loocv_aggregate.json     -- extended schema with frame positions
    figures/
      reach_detection_summary.png

If this works, follow-up to test PROBA_SMOOTH_WINDOW=7 in a separate
snapshot. If neither works, the smoothing lever is exhausted and the
next iteration should target labels (mousereach.reach.v8.labels) or
features (mousereach.reach.v8.features).

Compare side-by-side against:
  Improvement_Snapshots/reach_detection/v8.0.0_dev_initial_loocv/                (baseline, mg=2)
  Improvement_Snapshots/reach_detection/v8.0.0_dev_failure_mode_breakdown/       (same baseline, with breakdown)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.train import train_one_fold, aggregate_folds
from mousereach.reach.v8.eval import GTReach, AlgoReach, evaluate_reaches, summarize_results, MatchResult
from mousereach.reach.v8.postprocess import probabilities_to_reaches
from mousereach.reach.v8.features import feature_columns
from mousereach.improvement.reach_detection.v8_figures import render_v8_reach_figures


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_proba_smooth_w5"
)

THRESHOLD = 0.5
MERGE_GAP = 2          # baseline value
MIN_SPAN = 3
PROBA_SMOOTH_WINDOW = 5  # the experimental parameter (centered rolling mean)


def smooth_proba(proba: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling-mean smoothing of a 1D probability array.

    Edges: padded by reflection so the smoothed array has the same
    length as the input. Reflection avoids dragging boundary values
    toward 0 (which simple zero-padding would do).
    """
    if window <= 1:
        return proba
    pad = window // 2
    padded = np.pad(proba, pad_width=pad, mode="reflect")
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(proba.dtype)


def loocv_with_smoothing_and_frame_capture(
    df, train_pool_ids, feat_cols, exhaustive_val_ids,
):
    """Per-fold loop replicating train_one_fold's training, then
    overriding the inference step to apply proba smoothing before
    probabilities_to_reaches. Captures algo_reaches and gt_reaches
    per video for downstream frame-augmented JSON serialization.

    Reuses train_one_fold to do the expensive training (it returns
    the trained classifier), then redoes inference inline with the
    smoothing step inserted.
    """
    folds = []
    per_video_data = {}

    for i, val_vid in enumerate(exhaustive_val_ids):
        train_ids = [v for v in train_pool_ids if v != val_vid]
        print(f"  fold {i+1}/{len(exhaustive_val_ids)}: val={val_vid}",
              flush=True)

        # Use train_one_fold to handle training + return the classifier.
        # Its internal val-eval will be IGNORED -- we recompute below
        # with the smoothing step inserted.
        _fold_no_smooth, clf = train_one_fold(
            df, train_ids, [val_vid],
            threshold=THRESHOLD,
            merge_gap=MERGE_GAP,
            min_span=MIN_SPAN,
            only_exhaustive_for_train=True,
            max_iter=200,
            learning_rate=0.05,
            max_depth=6,
        )

        # Inference with smoothing inserted
        val = df.loc[df["video_id"] == val_vid]
        Xv = val[feat_cols].to_numpy(dtype=np.float32)
        proba_raw = clf.predict_proba(Xv)[:, 1]
        proba_smoothed = smooth_proba(proba_raw, PROBA_SMOOTH_WINDOW)

        algo_reaches_raw = probabilities_to_reaches(
            proba_smoothed,
            threshold=THRESHOLD,
            merge_gap=MERGE_GAP,
            min_span=MIN_SPAN,
        )
        algo_reaches = [
            AlgoReach(
                start_frame=r.start_frame,
                end_frame=r.end_frame,
                video_id=val_vid,
                index=j,
            )
            for j, r in enumerate(algo_reaches_raw)
        ]

        # Build GT reaches identically to train_one_fold's logic
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

        # Match using the existing evaluator (unchanged)
        results = evaluate_reaches(algo_reaches, gt_reaches, video_id=val_vid)
        summary = summarize_results(results)

        # Print per-fold summary
        s = summary
        print(f"    TP={s['n_tp']:>4} FP={s['n_fp']:>4} FN={s['n_fn']:>4}  "
              f"start_delta median={s['tp_start_delta']['median']} "
              f"abs_median={s['tp_start_delta']['abs_median']}  "
              f"span_delta median={s['tp_span_delta']['median']} "
              f"abs_median={s['tp_span_delta']['abs_median']}",
              flush=True)

        # Mimic FoldResult shape (without importing the dataclass)
        class _FR:
            pass
        fold = _FR()
        fold.val_video_ids = [val_vid]
        fold.threshold = THRESHOLD
        fold.summary = summary
        fold.raw_results = results
        folds.append(fold)
        per_video_data[val_vid] = (algo_reaches, gt_reaches)

    return folds, per_video_data


def main():
    print("=" * 70)
    print("PHASE B LOOCV (exhaustive subset) -- proba smoothing W=5")
    print("Probability series smoothed by centered rolling mean of 5 frames")
    print("BEFORE thresholding. All other params held fixed.")
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
    print(f"  PROBA_SMOOTH_WINDOW={PROBA_SMOOTH_WINDOW}")
    print()

    feat_cols = feature_columns()
    exh_set = set(df.loc[df["exhaustive"], "video_id"].unique().tolist())
    eligible_val = [v for v in train_pool_ids if v in exh_set]
    print(f"LOOCV: {len(eligible_val)} exhaustive folds")
    print()

    folds, per_video_data = loocv_with_smoothing_and_frame_capture(
        df, train_pool_ids, feat_cols, eligible_val)
    print()

    # Aggregate
    all_results = []
    for f in folds:
        all_results.extend(f.raw_results)
    agg = summarize_results(all_results)
    print("=" * 70)
    print(f"AGGREGATE LOOCV RESULTS (proba smoothing W={PROBA_SMOOTH_WINDOW})")
    print("=" * 70)
    print(f"  TP={agg['n_tp']}  FP={agg['n_fp']}  FN={agg['n_fn']}")
    print(f"  Start delta on TPs: median={agg['tp_start_delta']['median']}f  "
          f"|median|={agg['tp_start_delta']['abs_median']}f  "
          f"range=[{agg['tp_start_delta']['min']},{agg['tp_start_delta']['max']}]")
    print(f"  Span delta on TPs:  median={agg['tp_span_delta']['median']}f  "
          f"|median|={agg['tp_span_delta']['abs_median']}f  "
          f"range=[{agg['tp_span_delta']['min']},{agg['tp_span_delta']['max']}]")
    print()
    print("Compare against baseline v8.0.0_dev_initial_loocv:")
    print("  (TP=1918  FP=337  FN=457  start range [-2,2]  span range [-8,8])")
    print()

    # Save artifacts
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    per_fold_out = []
    serialized_results = []
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
        json.dumps(per_fold_out, indent=2), encoding="utf-8")
    (metrics_dir / "loocv_aggregate.json").write_text(
        json.dumps({
            "n_folds": len(folds),
            "summary": agg,
            "raw_results": serialized_results,
            "merge_gap": MERGE_GAP,
            "proba_smooth_window": PROBA_SMOOTH_WINDOW,
            "schema_version": "extended_with_frame_positions",
        }, indent=2), encoding="utf-8")

    render_v8_reach_figures(
        snapshot_dir=SNAPSHOT_DIR,
        raw_results=serialized_results,
        summary=agg,
        title_suffix=f" (LOOCV, {len(folds)} folds, exhaustive only, proba smoothing W={PROBA_SMOOTH_WINDOW})",
    )

    print(f"Wrote: {metrics_dir / 'loocv_per_fold.json'}")
    print(f"Wrote: {metrics_dir / 'loocv_aggregate.json'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_summary.png'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_legend.md'}")


if __name__ == "__main__":
    main()
