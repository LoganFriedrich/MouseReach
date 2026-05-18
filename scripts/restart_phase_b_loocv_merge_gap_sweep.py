"""
v8 dev experiment: merge_gap sweep (low side) on top of BSW b=1 w=0.8.

Tests `merge_gap = 0` and `merge_gap = 1` against the cumulative best
(`merge_gap = 2`), all with BSW b=1/w=0.8 boundary weighting at training
time. The expensive step (GBM training per fold) is shared across the
three merge_gap values; the cheap part (postprocess + match) is repeated
per merge_gap.

Motivated by the 2026-05-18 single-video diagnostic
(`diagnose_v8_merge_dips_single_video.py` on `20250718_CNT0214_P1`),
which found that 96% of merge events on that video had a sub-threshold
proba dip (median dip width = 2 frames, median `gap_min_proba` = 0.004)
that the current production `merge_gap = 2` bridges. The model "sees"
the boundary between two real reaches but the postprocess erases it.

The candidate is **merge_gap = 1**, justified from the diagnostic's
median dip-width = 2 frames: bridges 1-frame intra-reach noise dips,
splits 2+ frame inter-reach gaps. merge_gap = 0 is included as a
diagnostic reference (more aggressive; risk of over-fragmentation on
brief intra-reach DLC noise).

================================================================
PRE-EXPERIMENT CHECKLIST (per pre_experiment_checklist.md)
================================================================

1. Cumulative-stacking check (verified 2026-05-18):
   - v8.0.0 production already integrates BSW b=1, w=0.8 (commit
     79f217f). Comparison baseline = production v8.0.0 LOOCV with
     merge_gap=2: TP=1935 / FP=330 / FN=440 / exact_start=84.08%.
   - Stacked improvements applied: BSW b=1 w=0.8 inline, identical to
     `restart_phase_b_loocv_boundary_sample_weight_w08.py`.
   - The merge_gap = 2 result this runner produces should reproduce
     the cumulative best within RNG noise (sanity check).

2. Existing-module-modification check:
   - Existing module code modified: NO. `probabilities_to_reaches`
     accepts `merge_gap` as a parameter; we just pass different values
     at the call site. No changes under src/mousereach/.

3. Assumption check (unverified hypotheses):
   - HYP: merge_gap = 1 will split most merges that were bridged by
     merge_gap = 2 because the median dip width (per single-video
     diagnostic) is 2 frames. With merge_gap = 1, only 1-frame
     sub-threshold gaps are bridged; 2+ frame gaps split.
   - HYP: merge_gap = 1 will preserve real reaches because intra-reach
     proba dips from DLC noise (if present) are typically 1 frame and
     would still be bridged. NOT VERIFIED PRE-RUN. If real-reach dips
     are commonly 2+ frames, merge_gap=1 will over-fragment.
   - HYP: merge_gap = 0 will be too aggressive on real reaches with
     any 1-frame noise dip. NOT VERIFIED PRE-RUN. Tested as the
     reference extreme.
   - HYP: The single-video finding (96% of merges have splittable dips)
     generalizes to the full LOOCV corpus. NOT VERIFIED -- this run
     produces the answer.

4. FN-direction-reporting check:
   - Planned RESULTS.md first line:
     "FN vs cumulative best (merge_gap=2): [direction + magnitude];
      FN vs pure baseline (no BSW, merge_gap=2): [direction + magnitude]."
   - Two-delta surfacing BEFORE any metric table, separately for each
     tested merge_gap value.

5. Framework-not-adhoc check:
   - Output: three snapshot dirs under
     `Improvement_Snapshots/reach_detection/`:
       v8.0.0_dev_merge_gap_0/    (candidate, aggressive)
       v8.0.0_dev_merge_gap_1/    (candidate, principled)
       v8.0.0_dev_merge_gap_2_reproduce/   (sanity check, should match cum-best)
   - Each has canonical `loocv_aggregate.json` (extended schema),
     `loocv_per_fold.json`, and `reach_detection_summary.png`.

6. Branch + tag check (deferred to user before run):
   - Tag: `v8-pre-merge-gap-sweep-low-2026-05-18`
   - Branch: `feature/v8-merge-gap-sweep-low`

7. Decision rule check (vs cumulative best = BSW w=0.8 LOOCV at merge_gap=2):
   - The PRIMARY CANDIDATE is merge_gap = 1.
     * REJECT if TP drops AND FN rises (vs cumulative best).
     * REJECT if exact-frame-start match rate drops > 0.3 pp.
     * ACCEPT if FN drops or TP rises with exact_start preserved.
   - merge_gap = 0 is a DIAGNOSTIC reference; same rule applies but
     it's not the candidate we'd ship even if it passes -- splitting
     on every single-frame dip is too aggressive a default.
   - merge_gap = 2 result is a SANITY CHECK only (should reproduce
     the cumulative best within RNG noise -- if it doesn't, something
     is wrong in the pipeline).
   - DO NOT pick merge_gap to maximise eval score. The principled
     value is 1 (per the dip-width data). The sweep is for context.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

SNAPSHOT_ROOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection"
)

SNAPSHOT_DIRS = {
    0: SNAPSHOT_ROOT / "v8.0.0_dev_merge_gap_0",
    1: SNAPSHOT_ROOT / "v8.0.0_dev_merge_gap_1",
    2: SNAPSHOT_ROOT / "v8.0.0_dev_merge_gap_2_reproduce",
}


# ---------------------------------------------------------------------------
# Stacked-improvement params (BSW b=1 w=0.8)
# ---------------------------------------------------------------------------

BOUNDARY_BUFFER = 1
BOUNDARY_WEIGHT = 0.8

THRESHOLD = 0.5
MIN_SPAN = 3
MERGE_GAPS = [0, 1, 2]
PRIMARY_CANDIDATE_MERGE_GAP = 1


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
# Per-fold: train BSW once, evaluate at multiple merge_gap values
# ---------------------------------------------------------------------------

def train_predict_for_fold(
    train_pool_df: pd.DataFrame,
    train_video_ids: List[str],
    val_vid: str,
    feat_cols: List[str],
):
    """Train GBM with BSW b=1 w=0.8, return (proba, val_df_sorted)."""
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

    return proba, val


def eval_at_merge_gap(proba: np.ndarray, val: pd.DataFrame,
                     val_vid: str, merge_gap: int):
    """Postprocess + match for one merge_gap value. Returns (summary,
    results, algo_reaches, gt_reaches)."""
    algo_reaches_raw = probabilities_to_reaches(
        proba, threshold=THRESHOLD, merge_gap=merge_gap, min_span=MIN_SPAN)
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
# Output writers
# ---------------------------------------------------------------------------

def write_snapshot(merge_gap: int,
                   folds: List[Dict],
                   all_results: List[MatchResult],
                   per_video_data: Dict[str, Tuple[List[AlgoReach], List[GTReach]]]
                   ) -> Dict:
    """Write canonical snapshot files for one merge_gap value."""
    snap_dir = SNAPSHOT_DIRS[merge_gap]
    snap_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = snap_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    agg = summarize_results(all_results)

    serialized = []
    for r in all_results:
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
        serialized.append(record)

    (metrics_dir / "loocv_per_fold.json").write_text(
        json.dumps(folds, indent=2), encoding="utf-8")
    (metrics_dir / "loocv_aggregate.json").write_text(
        json.dumps({
            "n_folds": len(folds), "summary": agg,
            "raw_results": serialized,
            "merge_gap": merge_gap,
            "boundary_buffer": BOUNDARY_BUFFER,
            "boundary_weight": BOUNDARY_WEIGHT,
            "schema_version": "extended_with_frame_positions",
        }, indent=2), encoding="utf-8")

    render_v8_reach_figures(
        snapshot_dir=snap_dir,
        raw_results=serialized,
        summary=agg,
        title_suffix=f" (LOOCV, BSW w=0.8 + merge_gap={merge_gap})",
    )
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("PHASE B LOOCV -- MERGE_GAP SWEEP (low side) on top of BSW b=1 w=0.8")
    print(f"Testing merge_gap = {MERGE_GAPS}")
    print(f"Primary candidate: merge_gap = {PRIMARY_CANDIDATE_MERGE_GAP}")
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

    # Storage: per merge_gap, per fold
    folds_by_mg: Dict[int, List[Dict]] = {mg: [] for mg in MERGE_GAPS}
    results_by_mg: Dict[int, List] = {mg: [] for mg in MERGE_GAPS}
    per_video_data_by_mg: Dict[int, Dict[str, Tuple]] = {mg: {} for mg in MERGE_GAPS}

    for i, val_vid in enumerate(eligible_val):
        train_ids = [v for v in train_pool_ids if v != val_vid]
        print(f"  fold {i+1}/{len(eligible_val)}: val={val_vid}", flush=True)

        # Train ONCE
        proba, val = train_predict_for_fold(df, train_ids, val_vid, feat_cols)

        # Evaluate at each merge_gap
        for mg in MERGE_GAPS:
            summary, results, algo_reaches, gt_reaches = \
                eval_at_merge_gap(proba, val, val_vid, mg)
            folds_by_mg[mg].append({"val_video_ids": [val_vid], "summary": summary})
            results_by_mg[mg].extend(results)
            per_video_data_by_mg[mg][val_vid] = (algo_reaches, gt_reaches)
            s = summary
            sd_mean = s['tp_start_delta']['mean']
            sd_mean_str = f"{sd_mean:.3f}" if sd_mean is not None else "n/a"
            print(f"    mg={mg}: TP={s['n_tp']:>4} FP={s['n_fp']:>4} FN={s['n_fn']:>4}  "
                  f"start_delta med={s['tp_start_delta']['median']} "
                  f"abs_med={s['tp_start_delta']['abs_median']} "
                  f"mean={sd_mean_str}  "
                  f"span_delta med={s['tp_span_delta']['median']}",
                  flush=True)

    print()

    # Write snapshots and collect aggregates
    aggregates = {}
    for mg in MERGE_GAPS:
        agg = write_snapshot(mg, folds_by_mg[mg], results_by_mg[mg],
                             per_video_data_by_mg[mg])
        aggregates[mg] = agg

    # Side-by-side comparison
    print("=" * 78)
    print("SIDE-BY-SIDE AGGREGATE RESULTS")
    print("=" * 78)
    print(f"  {'merge_gap':>9}  {'TP':>5} {'FP':>5} {'FN':>5}  "
          f"{'start_med':>9} {'start_amed':>10} {'start_mean':>10}  "
          f"{'span_med':>8} {'span_amed':>9}")
    for mg in MERGE_GAPS:
        a = aggregates[mg]
        sd = a['tp_start_delta']; sp = a['tp_span_delta']
        sd_mean = sd['mean']; sd_mean_s = f"{sd_mean:.3f}" if sd_mean is not None else "n/a"
        print(f"  {mg:>9}  {a['n_tp']:>5} {a['n_fp']:>5} {a['n_fn']:>5}  "
              f"{sd['median']:>9} {sd['abs_median']:>10} {sd_mean_s:>10}  "
              f"{sp['median']:>8} {sp['abs_median']:>9}")
    print()
    print("Compare against:")
    print("  Pure baseline (no BSW, merge_gap=2):  TP=1918  FP=337  FN=457  exact_start=83.47%")
    print("  Cumulative best (BSW w=0.8, merge_gap=2):  TP=1935  FP=330  FN=440  exact_start=84.08%")
    print()

    # Deltas vs cumulative best for each merge_gap
    cum_best_tp = 1935; cum_best_fp = 330; cum_best_fn = 440
    pure_tp = 1918; pure_fp = 337; pure_fn = 457
    print("=" * 78)
    print("DELTAS vs cumulative best (and pure baseline) for each merge_gap")
    print("=" * 78)
    for mg in MERGE_GAPS:
        a = aggregates[mg]
        d_tp_cum = a['n_tp'] - cum_best_tp
        d_fp_cum = a['n_fp'] - cum_best_fp
        d_fn_cum = a['n_fn'] - cum_best_fn
        d_tp_pure = a['n_tp'] - pure_tp
        d_fp_pure = a['n_fp'] - pure_fp
        d_fn_pure = a['n_fn'] - pure_fn
        primary_marker = " <-- PRIMARY CANDIDATE" if mg == PRIMARY_CANDIDATE_MERGE_GAP else ""
        ref_marker = " (sanity check)" if mg == 2 else ""
        print(f"  merge_gap={mg}{primary_marker}{ref_marker}")
        print(f"    vs cumulative best (BSW w=0.8, mg=2):  "
              f"TP {d_tp_cum:+d}  FP {d_fp_cum:+d}  FN {d_fn_cum:+d}")
        print(f"    vs pure baseline (no BSW, mg=2):       "
              f"TP {d_tp_pure:+d}  FP {d_fp_pure:+d}  FN {d_fn_pure:+d}")
        # Decision rule check
        if mg != 2:
            tp_drops = d_tp_cum < 0
            fn_rises = d_fn_cum > 0
            if tp_drops and fn_rises:
                print(f"    DECISION RULE: REJECT (TP drops AND FN rises vs cum-best)")
            elif d_fn_cum < 0 or d_tp_cum > 0:
                print(f"    DECISION RULE: candidate for ACCEPT (FN drops or TP rises vs cum-best)")
            else:
                print(f"    DECISION RULE: ambiguous (neither REJECT nor clean ACCEPT)")
        print()

    print()
    print("REMINDER: write RESULTS.md in each snapshot dir leading with FN delta")
    print("vs cumulative best AND vs pure baseline, BEFORE any metric table.")
    print()
    print("Snapshot dirs:")
    for mg in MERGE_GAPS:
        print(f"  merge_gap={mg}: {SNAPSHOT_DIRS[mg]}")


if __name__ == "__main__":
    main()
