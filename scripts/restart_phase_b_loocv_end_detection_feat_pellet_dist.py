"""
v8 dev experiment: phase B LOOCV with one new end-detection feature
added to the per-frame feature matrix, layered on BSW w=0.8.

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

   Stacked improvements applied: BSW b=1 w=0.8 + new end-detection feature.
   Comparison baseline: v8.0.0_dev_boundary_sample_weight_b1_w0.8/
     (TP=1935, FP=330, FN=440, exact_start=84.08%, exact_span=75.09%)
   Verification method: snapshot RESULTS.md + git log + train_one_fold inspection.

2. Existing-code-modification check:
   Existing module code modified: NO. Runner replicates train_one_fold
   inline. New feature is computed inline from existing parquet columns
   (dist__Pellet__RightHand) -- no edit to mousereach/reach/v8/features.py.

3. Assumption check (unverified hypotheses flagged):
   - HYPOTHESIS: a centered 5-frame difference of the paw-to-pellet
     distance carries a useful end-detection signal that the existing
     features don't already encode. Existing features include
     RightHand_vy (per-bodypart velocity) and dist__Pellet__RightHand
     (per-frame distance), but NOT the temporal derivative of the
     pairwise distance. This experiment is the verification of whether
     that specific derived signal helps.
   - HYPOTHESIS: 5-frame window is the right time scale. Mice complete
     reaches in ~6-12 frames; a 5-frame centered difference samples
     across roughly half a reach. Could be wrong. If the experiment
     fails, a follow-up at 3 frames or 10 frames would test scale.
   - HYPOTHESIS: RightHand is the relevant paw bodypart. There are
     also RHLeft, RHOut, RHRight as periphery bodyparts. RightHand is
     described as the central paw and is what the existing detector
     uses elsewhere. Could be that one of the periphery is more
     informative for retraction, but starting with the central is the
     principled first step.

4. FN-direction-reporting check:
   The results report will lead with FN delta vs BSW w=0.8 (the
   cumulative-best baseline), explicitly stated as rising/falling/flat
   with magnitude. Will not be buried in a metrics table.

5. Framework-not-adhoc check:
   Output goes to canonical Improvement_Snapshots layout. Uses extended
   schema (per-event frame positions). render_v8_reach_figures used
   for canonical figures. No one-off output to Validation_Runs/reports/.

6. Branch + tag check:
   Pre-experiment tag: v8-pre-end-detection-feat-2026-05-04
   Branch: feature/v8-end-detection-feat-pellet-dist-change

7. Decision-rule check:
   Reject if TP drops AND FN rises vs BSW w=0.8 (NOT vs pure baseline).
   Reject if exact-frame-match rate drops materially vs BSW w=0.8.
   Accept if FN drops OR TP rises with exact-match held.

==============================================================================
RATIONALE
==============================================================================

Target failure mode (from v8.0.0_dev_within_gt_fp_inspection): within_gt
FPs come from algo runs extending past GT_end because GBM probability
stays high through post-reach paw retraction motion. The model's
per-frame features (raw position, velocity, acceleration, smoothed
versions, pairwise distances) encode "what's the paw doing at this
frame" but don't directly encode "is the paw approaching or retreating
from the pellet?" The pairwise distance dist__Pellet__RightHand is
present but its temporal derivative is not. That derivative is the
signed signal "paw approaching pellet" (negative = approaching = reach
extension; positive = retreating = retraction).

Why principled, not "tune until eval passes":
The within_gt FP analysis showed Pattern B's defining feature is
sustained high probability past GT_end during retraction. The model's
input features include velocity of paw and velocity of pellet
SEPARATELY but not the velocity of their distance. A centered finite
difference of dist__Pellet__RightHand is the specific signal that
distinguishes extension (distance shrinking, dist_change < 0) from
retraction (distance growing, dist_change > 0). If the GBM learns to
use this, it should produce LOWER probability during retraction --
which would cause the run to terminate at the right place, addressing
both the within_gt FPs AND the corresponding tol_miss_span FNs.

Risk and guardrails:
- The new feature might be redundant with existing velocity features
  (model already has RightHand_vy and Pellet_vy separately). If so,
  feature importance for the new column would be low and net effect
  near zero (similar to BSW w=0.9 -- no harm, but no benefit). Detect
  via TP/FP/FN movement near zero.
- The new feature might let the model learn to terminate runs too
  aggressively (e.g., if the paw briefly slows mid-reach). Detect
  via FN rising on long reaches.
- Adding any feature can cause regularization shifts in the GBM that
  degrade other behaviors (model trades off learning capacity).
  Detect via boundary-precision metrics dropping.

Expected eval deltas (vs BSW w=0.8 cumulative best):
- n_fp drops materially (within_gt FPs whose runs now terminate near
  GT_end recover into TPs)
- n_tp rises (some span-attributable FNs recover)
- n_fn drops (tol_miss_span FNs recover)
- exact-frame-match rate: should hold or rise (the model's
  termination signal is more accurate)
- start_delta unchanged (the new feature is specifically about end
  detection, shouldn't shift starts)

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

# Stacked accepted improvement: BSW b=1 w=0.8 (per v8_pending_integrations.md)
BOUNDARY_BUFFER = 1
BOUNDARY_WEIGHT = 0.8

# New change being tested: pellet-distance derivative feature
NEW_FEATURE_NAME = "dist__Pellet__RightHand__centered_diff_5f"
NEW_FEATURE_HALF_WIDTH = 2  # centered difference: f[t+2] - f[t-2]
SOURCE_DIST_COL = "dist__Pellet__RightHand"

THRESHOLD = 0.5
MERGE_GAP = 2
MIN_SPAN = 3

SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_end_detection_feat_pellet_dist_5f"
)


def add_pellet_distance_derivative(df, source_col, new_col_name, half_width):
    """Add a centered finite-difference of the paw-to-pellet distance,
    grouped by video. Edge frames (first/last half_width per video)
    get 0 (no shift / no derivative info).
    """
    out = df.copy()
    out[new_col_name] = 0.0

    for vid, group in out.groupby("video_id", sort=False):
        idx = group.index.to_numpy()
        # group must be sorted by frame for the shift to be meaningful
        sub = group.sort_values("frame")
        sub_idx = sub.index.to_numpy()
        d = sub[source_col].to_numpy(dtype=np.float32)
        n = len(d)
        diff = np.zeros(n, dtype=np.float32)
        if n > 2 * half_width:
            diff[half_width:n - half_width] = d[2 * half_width:] - d[:n - 2 * half_width]
        # write back into out at the original (sub_idx) positions
        out.loc[sub_idx, new_col_name] = diff

    return out


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


def train_one_fold_combined(train_pool_df, train_video_ids, val_vid, feat_cols_with_new):
    """Replicates train_one_fold + BSW + augmented feature set."""
    train_mask = train_pool_df["video_id"].isin(train_video_ids) & train_pool_df["exhaustive"]
    train_mask &= train_pool_df["video_id"] != val_vid
    train = train_pool_df.loc[train_mask]
    val = train_pool_df.loc[train_pool_df["video_id"] == val_vid]

    X_train = train[feat_cols_with_new].to_numpy(dtype=np.float32)
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

    Xv = val[feat_cols_with_new].to_numpy(dtype=np.float32)
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

    # Capture feature importance for the new feature
    new_feat_idx = feat_cols_with_new.index(NEW_FEATURE_NAME)
    # HistGradientBoostingClassifier doesn't expose feature_importances_
    # by default in older sklearn; use mean-impurity if available
    try:
        importance = float(clf._predictors[0][0].compute_features_importance(np.array([new_feat_idx])))
    except Exception:
        importance = None

    return summary, results, algo_reaches, gt_reaches, importance


def main():
    print("=" * 70)
    print("PHASE B LOOCV (exhaustive subset)")
    print(f"  Stacked: BSW b={BOUNDARY_BUFFER} w={BOUNDARY_WEIGHT} (accepted prior)")
    print(f"  New: feature {NEW_FEATURE_NAME} (centered-diff half-width={NEW_FEATURE_HALF_WIDTH})")
    print(f"  Source: {SOURCE_DIST_COL}")
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

    print(f"Computing new feature: {NEW_FEATURE_NAME} ...", flush=True)
    df = add_pellet_distance_derivative(
        df, SOURCE_DIST_COL, NEW_FEATURE_NAME, NEW_FEATURE_HALF_WIDTH)
    print(f"  done. Range: [{df[NEW_FEATURE_NAME].min():.3f}, {df[NEW_FEATURE_NAME].max():.3f}]")
    print(f"  mean: {df[NEW_FEATURE_NAME].mean():.3f}, std: {df[NEW_FEATURE_NAME].std():.3f}")
    print()

    base_feat_cols = feature_columns()
    feat_cols_with_new = list(base_feat_cols) + [NEW_FEATURE_NAME]
    print(f"Total features: {len(feat_cols_with_new)} ({len(base_feat_cols)} baseline + 1 new)")
    print()

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
        summary, results, algo_reaches, gt_reaches, importance = \
            train_one_fold_combined(df, train_ids, val_vid, feat_cols_with_new)

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
    print("Compare against (cumulative best):")
    print("  BSW w=0.8 alone:  TP=1935  FP=330  FN=440  exact_start=84.08%  span mean=0.170")
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
            "new_feature_name": NEW_FEATURE_NAME,
            "new_feature_half_width": NEW_FEATURE_HALF_WIDTH,
            "schema_version": "extended_with_frame_positions",
        }, indent=2), encoding="utf-8")

    render_v8_reach_figures(
        snapshot_dir=SNAPSHOT_DIR,
        raw_results=serialized_results,
        summary=agg,
        title_suffix=f" (LOOCV, BSW w={BOUNDARY_WEIGHT} + pellet-dist-diff)",
    )

    print(f"Wrote: {metrics_dir / 'loocv_per_fold.json'}")
    print(f"Wrote: {metrics_dir / 'loocv_aggregate.json'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_summary.png'}")


if __name__ == "__main__":
    main()
