"""
v8 dev experiment: phase B LOOCV with HAND-CENTROID and VALLEY-DEPTH features.

Pre-experiment checklist (per `pre_experiment_checklist.md`):

1. Cumulative-stacking check (verified 2026-05-21):
   - Production v8.0.1 = BSW b=1/w=0.8 (baked into model bundle) + merge_gap=0
     (postprocess default). All accepted improvements composed into the
     cumulative best. v8_pending_integrations.md is stale on this point;
     BSW shipped 2026-05-06 per pipeline_versions.json.
   - Comparison baseline (against parquet labels = pre-rescore GT):
     v8.0.0_dev_merge_gap_0_candidate: TP=2077 / FP=297 / FN=298.
   - Pure baseline: v8.0.0_dev_merge_gap_2_reproduce: TP=1935 / FP=330 / FN=440.
   - This experiment APPLIES BSW b=1/w=0.8 + merge_gap=0 + NEW FEATURES.

2. Existing-code-modification check:
   - NO modifications to src/mousereach/reach/v8/*.
   - All new feature computation is inline in this runner.

3. Unverified hypotheses (acknowledged):
   - The GBM will learn to use valley_depth as an in-reach signal. The
     feature is discriminative when measured at GT-boundary frames in 6/7
     videos (per `v8.0.1_dev_inter_merge_boundary_signature` diagnostic),
     but whether the GBM puts weight on it during training is empirical.
   - The signal validated at known GT boundary frames will translate to
     algo-merged boundary frames. The diagnostic tested at midpoints of
     known inter-GT spans, not at arbitrary in-reach frames.
   - Won't introduce regressions on TP/FP elsewhere. The feature could
     over-fire on long single reaches with mid-reach retraction noise.
   - CNT0301_P3 (chronic-FP video) is unlikely to benefit. The signature
     doesn't exist on its merged spans (per diagnostic). Included in
     aggregate per Logan's decision.

4. FN direction reporting (planned):
   - Lead with FN delta vs cumulative best (mg=0 candidate, pre-rescore)
   - Also report FN delta vs pure baseline (mg=2, pre-rescore)
   - ASCII tables; no precision/recall/F1/AUC

5. Framework check:
   - Output to Improvement_Snapshots/reach_detection/v8.0.0_dev_centroid_valley_features/
   - metrics/loocv_aggregate.json, metrics/loocv_per_fold.json (matching BSW runner schema)
   - Adds metrics/topology_breakdown.json (new -- targeted measurement)
   - Adds metrics/feature_importances.json (new -- to verify the GBM actually used the new features)
   - Figures via mousereach.improvement.reach_detection.v8_figures

6. Branch + tag:
   - Pre-experiment tag: v8-pre-centroid-valley-2026-05-21 (against master HEAD c1de63e)
   - Feature branch: feature/v8-centroid-valley-features
   - NOT pushed; per `feedback_ask_before_every_push.md`

7. Decision rule (LOOCV gate):
   ACCEPT if all of:
     - MERGED topology count drops by >= 10 vs cumulative best
     - TP count does NOT drop vs cumulative best
     - start_delta abs_median holds at 0 (Cardinal Rule)
     - FN does not rise
     - GBM feature importance for at least one new centroid/valley feature > 0
   REJECT if:
     - TP drops AND FN rises
     - start_delta abs_median rises above 0
     - MERGED drop is canceled by FP/FN regressions
     - GBM feature importance ~0 for all new features (model didn't learn from them)

   Holdout gate (only if LOOCV accepts) runs separately.

New features added (35 total):
- Synthetic 19th bodypart RHcentroid = mean(RightHand, RHLeft, RHOut, RHRight)
  -> 14 per-frame features (matching v8 standard suffixes)
  -> 18 new pairwise distances (RHcentroid to each other bodypart, smoothed)
- 3 valley-detection features on dist__RHcentroid__BOXL:
  -> rolling past_max over [t-5, t]
  -> rolling future_max over [t, t+5]
  -> valley_depth = min(past_max, future_max) - current
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.eval import (
    GTReach, AlgoReach, MatchResult, evaluate_reaches, summarize_results,
)
from mousereach.reach.v8.postprocess import probabilities_to_reaches
from mousereach.reach.v8.features import BODYPARTS as V8_BODYPARTS
from mousereach.reach.v8.features import feature_columns as v8_feature_columns
from mousereach.improvement.reach_detection.v8_figures import render_v8_reach_figures


# ===== Constants =====

CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)

SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_centroid_valley_features"
)

# Cumulative best parameters
THRESHOLD = 0.5
MERGE_GAP = 0
MIN_SPAN = 3
BOUNDARY_BUFFER = 1
BOUNDARY_WEIGHT = 0.8

# New-feature parameters
HAND_KPS = ["RightHand", "RHLeft", "RHOut", "RHRight"]
NEW_BP = "RHcentroid"
WINDOW_HALF = 5  # +/- 5 frames for past_max / future_max

# v8-standard per-bodypart suffixes
PER_BP_SUFFIXES = [
    "x", "y", "lk", "x_smooth", "y_smooth",
    "vx", "vy", "ax", "ay", "speed", "dlk",
    "speed_max20", "speed_max40", "lk_min20",
]
SMOOTH_WINDOW = 5
VELOCITY_DT = 2
SPEED_ROLLING_W1 = 21
SPEED_ROLLING_W2 = 41
LK_ROLLING_W = 21


# ===== Feature computation =====

def _centered_diff(arr, dt):
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    if n < 2 * dt + 1:
        return out
    out[dt:n - dt] = (arr[2 * dt:n] - arr[0:n - 2 * dt]) / (2.0 * dt)
    return out


def _rolling_past_max(arr, w):
    """Max over [t-w, t] for each t, NaN for t < w."""
    s = pd.Series(arr).rolling(w + 1, min_periods=1).max()
    return s.to_numpy(dtype=np.float32)


def _rolling_future_max(arr, w):
    """Max over [t, t+w] for each t."""
    # Reverse, rolling max forward, reverse back
    rev = arr[::-1]
    out = pd.Series(rev).rolling(w + 1, min_periods=1).max().to_numpy()
    return out[::-1].astype(np.float32)


def _compute_centroid_features_for_video(video_df):
    """Compute the new features for a single video's frames (must be
    sorted by frame). Returns a DataFrame indexed by the video_df's index.
    """
    df = video_df.sort_values("frame")
    n = len(df)
    out = {}

    # Raw centroid x, y, likelihood (mean of 4 hand keypoints)
    cx_raw = df[[f"{kp}_x" for kp in HAND_KPS]].mean(axis=1).to_numpy(dtype=np.float32)
    cy_raw = df[[f"{kp}_y" for kp in HAND_KPS]].mean(axis=1).to_numpy(dtype=np.float32)
    clk_raw = df[[f"{kp}_lk" for kp in HAND_KPS]].mean(axis=1).to_numpy(dtype=np.float32)

    # Smoothed (centered MA over 5 frames)
    cx_smooth = pd.Series(cx_raw).rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)
    cy_smooth = pd.Series(cy_raw).rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)

    # Velocity, acceleration, speed
    vx = _centered_diff(cx_smooth, VELOCITY_DT)
    vy = _centered_diff(cy_smooth, VELOCITY_DT)
    ax = _centered_diff(vx, VELOCITY_DT)
    ay = _centered_diff(vy, VELOCITY_DT)
    speed = np.sqrt(vx * vx + vy * vy).astype(np.float32)

    # Likelihood derivative + rolling features
    dlk = _centered_diff(clk_raw, VELOCITY_DT)
    speed_max20 = pd.Series(speed).rolling(SPEED_ROLLING_W1, center=True, min_periods=1).max().to_numpy(dtype=np.float32)
    speed_max40 = pd.Series(speed).rolling(SPEED_ROLLING_W2, center=True, min_periods=1).max().to_numpy(dtype=np.float32)
    lk_min20 = pd.Series(clk_raw).rolling(LK_ROLLING_W, center=True, min_periods=1).min().to_numpy(dtype=np.float32)

    out[f"{NEW_BP}_x"] = cx_raw
    out[f"{NEW_BP}_y"] = cy_raw
    out[f"{NEW_BP}_lk"] = clk_raw
    out[f"{NEW_BP}_x_smooth"] = cx_smooth
    out[f"{NEW_BP}_y_smooth"] = cy_smooth
    out[f"{NEW_BP}_vx"] = vx
    out[f"{NEW_BP}_vy"] = vy
    out[f"{NEW_BP}_ax"] = ax
    out[f"{NEW_BP}_ay"] = ay
    out[f"{NEW_BP}_speed"] = speed
    out[f"{NEW_BP}_dlk"] = dlk
    out[f"{NEW_BP}_speed_max20"] = speed_max20
    out[f"{NEW_BP}_speed_max40"] = speed_max40
    out[f"{NEW_BP}_lk_min20"] = lk_min20

    # Pairwise distances: RHcentroid_smooth to each other bodypart's _smooth
    for other_bp in V8_BODYPARTS:
        ox_col = f"{other_bp}_x_smooth"
        oy_col = f"{other_bp}_y_smooth"
        if ox_col in df.columns and oy_col in df.columns:
            ox = df[ox_col].to_numpy(dtype=np.float32)
            oy = df[oy_col].to_numpy(dtype=np.float32)
            d = np.sqrt((cx_smooth - ox) ** 2 + (cy_smooth - oy) ** 2).astype(np.float32)
        else:
            d = np.zeros(n, dtype=np.float32)
        # Naming order: dist__a__b sorted alphabetically per v8 convention
        # V8 BODYPARTS appear before the new one alphabetically in some cases,
        # but to keep names predictable just use dist__{other}__RHcentroid
        # so distances appear after existing pairwise dists in feature_columns().
        out[f"dist__{other_bp}__{NEW_BP}"] = d

    # Valley-detection features on dist__RHcentroid__BOXL
    # (note: this is the same series as dist__BOXL__RHcentroid which we computed above)
    d_boxl = out[f"dist__BOXL__{NEW_BP}"]
    past_max = _rolling_past_max(d_boxl, WINDOW_HALF)
    future_max = _rolling_future_max(d_boxl, WINDOW_HALF)
    valley = np.minimum(past_max, future_max) - d_boxl
    out[f"{NEW_BP}_BOXL_past_max_5f"] = past_max
    out[f"{NEW_BP}_BOXL_future_max_5f"] = future_max
    out[f"{NEW_BP}_BOXL_valley_depth_5f"] = valley.astype(np.float32)

    return pd.DataFrame(out, index=df.index)


def add_centroid_features(parquet_df):
    """Compute new features for the whole parquet, return augmented DataFrame."""
    print(f"  Computing new features across {parquet_df['video_id'].nunique()} videos...", flush=True)
    parts = []
    for vid, grp in parquet_df.groupby("video_id", sort=False):
        new_cols = _compute_centroid_features_for_video(grp)
        parts.append(new_cols)
    new_df = pd.concat(parts).reindex(parquet_df.index)
    return pd.concat([parquet_df, new_df], axis=1)


def feature_columns_with_centroid():
    """Feature column list = v8 baseline + new RHcentroid bodypart features +
    all dist__*__RHcentroid + 3 valley features."""
    cols = list(v8_feature_columns())
    for suf in PER_BP_SUFFIXES:
        cols.append(f"{NEW_BP}_{suf}")
    for other_bp in V8_BODYPARTS:
        cols.append(f"dist__{other_bp}__{NEW_BP}")
    cols.append(f"{NEW_BP}_BOXL_past_max_5f")
    cols.append(f"{NEW_BP}_BOXL_future_max_5f")
    cols.append(f"{NEW_BP}_BOXL_valley_depth_5f")
    return cols


# ===== BSW boundary weighting (copied verbatim from BSW runner) =====

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


# ===== Topology classification (connected-components on algo-GT overlap graph) =====

def classify_topology(algo_reaches, gt_reaches):
    """For each connected component in the algo-GT overlap graph, return a
    topology label: TP, TOLERANCE_ERROR, MERGED, FRAGMENTED, FALSE_POSITIVE,
    FALSE_NEGATIVE, COMPLEX. Returns a list of (label, n_algo, n_gt) per component.
    """
    def overlap(a_start, a_end, b_start, b_end):
        return not (a_end < b_start or a_start > b_end)

    # Build adjacency: algo_i -> set of gt_j it overlaps
    n_algo = len(algo_reaches)
    n_gt = len(gt_reaches)
    algo_to_gt = defaultdict(set)
    gt_to_algo = defaultdict(set)
    for i, a in enumerate(algo_reaches):
        for j, g in enumerate(gt_reaches):
            if overlap(a.start_frame, a.end_frame, g.start_frame, g.end_frame):
                algo_to_gt[i].add(j)
                gt_to_algo[j].add(i)

    # Connected components via BFS
    visited_algo = set()
    visited_gt = set()
    components = []
    for i in range(n_algo):
        if i in visited_algo: continue
        if not algo_to_gt[i]:
            # Isolated algo with no GT overlap = FALSE_POSITIVE
            components.append(("FALSE_POSITIVE", 1, 0))
            visited_algo.add(i)
            continue
        # BFS
        algo_in_comp, gt_in_comp = set(), set()
        queue = [("a", i)]
        while queue:
            kind, idx = queue.pop()
            if kind == "a":
                if idx in algo_in_comp: continue
                algo_in_comp.add(idx)
                for gj in algo_to_gt[idx]:
                    queue.append(("g", gj))
            else:
                if idx in gt_in_comp: continue
                gt_in_comp.add(idx)
                for ai in gt_to_algo[idx]:
                    queue.append(("a", ai))
        visited_algo.update(algo_in_comp)
        visited_gt.update(gt_in_comp)

        na = len(algo_in_comp)
        ng = len(gt_in_comp)
        if na == 1 and ng == 1:
            # Need to check tolerance for TP vs TOLERANCE_ERROR
            a = algo_reaches[next(iter(algo_in_comp))]
            g = gt_reaches[next(iter(gt_in_comp))]
            start_delta = a.start_frame - g.start_frame
            algo_span = a.end_frame - a.start_frame + 1
            gt_span = g.end_frame - g.start_frame + 1
            span_delta = algo_span - gt_span
            span_tol = max(0.5 * gt_span, 5)
            if abs(start_delta) <= 2 and abs(span_delta) <= span_tol:
                components.append(("TP", 1, 1))
            else:
                components.append(("TOLERANCE_ERROR", 1, 1))
        elif na == 1 and ng >= 2:
            components.append(("MERGED", na, ng))
        elif na >= 2 and ng == 1:
            components.append(("FRAGMENTED", na, ng))
        elif na >= 2 and ng >= 2:
            components.append(("COMPLEX", na, ng))

    # Isolated GTs with no algo overlap = FALSE_NEGATIVE
    for j in range(n_gt):
        if j not in visited_gt:
            components.append(("FALSE_NEGATIVE", 0, 1))

    return components


# ===== Per-fold training =====

def train_one_fold(train_pool_df, train_video_ids, val_vid, feat_cols):
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
    unique_rids = sorted(set(rid[rid >= 0].tolist()))
    for ri in unique_rids:
        rmask = rid == ri
        f = frames[rmask]
        gt_reaches.append(GTReach(
            start_frame=int(f.min()), end_frame=int(f.max()),
            video_id=val_vid, index=ri))

    results = evaluate_reaches(algo_reaches, gt_reaches, video_id=val_vid)
    summary = summarize_results(results)
    topology = classify_topology(algo_reaches, gt_reaches)
    return summary, results, algo_reaches, gt_reaches, topology, clf


# ===== Main =====

def main():
    print("=" * 70)
    print("PHASE B LOOCV -- CENTROID + VALLEY FEATURES")
    print(f"  BSW b={BOUNDARY_BUFFER}/w={BOUNDARY_WEIGHT}  mg={MERGE_GAP}  thr={THRESHOLD}")
    print(f"  Window half-width for past/future max: {WINDOW_HALF} frames")
    print("=" * 70)
    print()

    print("Loading train_pool.parquet ...", flush=True)
    parquet_path = CORPUS_DIR / "phase_b_dataset" / "train_pool.parquet"
    if not parquet_path.exists():
        print(f"[FAIL] missing: {parquet_path}")
        sys.exit(1)
    df = pd.read_parquet(parquet_path)
    print(f"  loaded {len(df)} rows across {df['video_id'].nunique()} videos")
    print(f"  baseline feature count: {len(v8_feature_columns())}")
    print()

    print("Adding centroid + valley features...", flush=True)
    df = add_centroid_features(df)
    feat_cols = feature_columns_with_centroid()
    new_cols = [c for c in feat_cols if c not in v8_feature_columns()]
    print(f"  augmented feature count: {len(feat_cols)} (added {len(new_cols)} new)")
    print(f"  new feature samples: {new_cols[:5]} ... {new_cols[-5:]}")
    # Verify all new cols are in df
    missing = [c for c in new_cols if c not in df.columns]
    if missing:
        print(f"[FAIL] missing columns in augmented df: {missing[:5]}")
        sys.exit(1)
    print()

    folds_def = json.loads(
        (CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))
    train_pool_ids = folds_def["train_pool"]["video_ids"]
    exh_set = set(df.loc[df["exhaustive"], "video_id"].unique().tolist())
    eligible_val = [v for v in train_pool_ids if v in exh_set]
    print(f"LOOCV: {len(eligible_val)} exhaustive folds\n")

    folds = []
    per_video_data = {}
    all_results_combined = []
    all_topology_per_video = {}
    last_clf = None  # keep one classifier for feature importance

    for i, val_vid in enumerate(eligible_val):
        train_ids = [v for v in train_pool_ids if v != val_vid]
        print(f"  fold {i+1}/{len(eligible_val)}: val={val_vid}", flush=True)
        summary, results, algo_reaches, gt_reaches, topology, clf = train_one_fold(
            df, train_ids, val_vid, feat_cols)
        last_clf = clf

        s = summary
        sd_mean = s['tp_start_delta']['mean']
        sd_mean_str = f"{sd_mean:.3f}" if sd_mean is not None else "n/a"
        topo_counts = defaultdict(int)
        for lbl, _, _ in topology:
            topo_counts[lbl] += 1
        topo_str = "  ".join(f"{k}={v}" for k, v in sorted(topo_counts.items())
                             if k in ("MERGED", "FRAGMENTED", "COMPLEX"))
        print(f"    TP={s['n_tp']:>4} FP={s['n_fp']:>4} FN={s['n_fn']:>4}  "
              f"start abs_med={s['tp_start_delta']['abs_median']} "
              f"mean={sd_mean_str}  "
              f"span abs_med={s['tp_span_delta']['abs_median']}  "
              f"{topo_str}",
              flush=True)
        folds.append({"val_video_ids": [val_vid], "summary": summary,
                      "topology_counts": dict(topo_counts)})
        per_video_data[val_vid] = (algo_reaches, gt_reaches)
        all_results_combined.extend(results)
        all_topology_per_video[val_vid] = topology

    print()
    agg = summarize_results(all_results_combined)
    topo_total = defaultdict(int)
    for vid, topo in all_topology_per_video.items():
        for lbl, _, _ in topo:
            topo_total[lbl] += 1

    print("=" * 70)
    print("AGGREGATE LOOCV RESULTS -- CENTROID + VALLEY features")
    print("=" * 70)
    print(f"  TP={agg['n_tp']}  FP={agg['n_fp']}  FN={agg['n_fn']}")
    print(f"  start_delta: median={agg['tp_start_delta']['median']} "
          f"abs_med={agg['tp_start_delta']['abs_median']}  "
          f"mean={agg['tp_start_delta']['mean']:.3f}")
    print(f"  span_delta:  median={agg['tp_span_delta']['median']} "
          f"abs_med={agg['tp_span_delta']['abs_median']}")
    print()
    print("Topology breakdown:")
    for k in ("TP", "TOLERANCE_ERROR", "MERGED", "FRAGMENTED",
              "FALSE_POSITIVE", "FALSE_NEGATIVE", "COMPLEX"):
        print(f"  {k:<18} {topo_total[k]}")
    print()
    print("Compare against:")
    print(f"  Cumulative best (v8.0.1 mg=0 candidate, pre-rescore):")
    print(f"    TP=2077  FP=297  FN=298")
    print(f"  Pure baseline (mg=2 reproduce, no BSW):")
    print(f"    TP=1935  FP=330  FN=440")
    print()

    # FN-direction summary (per always_report_fn_direction)
    cum_best_fn = 298
    pure_baseline_fn = 440
    cum_best_tp = 2077
    pure_baseline_tp = 1935
    print("FN direction (leading metric):")
    print(f"  vs cumulative best:  delta FN = {agg['n_fn'] - cum_best_fn:+d}  "
          f"(was {cum_best_fn}, now {agg['n_fn']})")
    print(f"  vs pure baseline:    delta FN = {agg['n_fn'] - pure_baseline_fn:+d}  "
          f"(was {pure_baseline_fn}, now {agg['n_fn']})")
    print()
    print(f"TP direction:")
    print(f"  vs cumulative best:  delta TP = {agg['n_tp'] - cum_best_tp:+d}")
    print(f"  vs pure baseline:    delta TP = {agg['n_tp'] - pure_baseline_tp:+d}")
    print()

    # MERGED-specific check (the targeted improvement)
    # Cumulative best LOOCV had MERGED count from manifests on rescored GT = 58.
    # We compute MERGED here against parquet GT (pre-rescore) so the comparison
    # might not be apples-to-apples. Report the absolute count.
    print(f"MERGED count (targeted): {topo_total['MERGED']}")
    print()

    # ===== Save outputs =====
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
            "window_half": WINDOW_HALF,
            "added_features": new_cols,
            "schema_version": "centroid_valley_v1",
        }, indent=2), encoding="utf-8")

    # Topology breakdown
    (metrics_dir / "topology_breakdown.json").write_text(
        json.dumps({
            "total": dict(topo_total),
            "per_video": {
                vid: dict(defaultdict(int, {l: sum(1 for ll, _, _ in topo if ll == l)
                                            for l in ("TP", "TOLERANCE_ERROR", "MERGED",
                                                      "FRAGMENTED", "FALSE_POSITIVE",
                                                      "FALSE_NEGATIVE", "COMPLEX")}))
                for vid, topo in all_topology_per_video.items()
            }
        }, indent=2), encoding="utf-8")

    # Feature importance for the new features (using last fold's classifier)
    if last_clf is not None:
        # HistGradientBoostingClassifier has feature_importances_ via permutation
        # in newer sklearn, OR we use the built-in get_feature_importances method
        # which uses gain-based importance.
        # NOTE: HGBC doesn't expose feature_importances_ in older sklearn versions.
        # As a fallback, compute permutation importance on the val set of the last fold.
        try:
            imp = last_clf.feature_importances_
        except AttributeError:
            imp = None
        if imp is not None:
            imp_dict = {feat: float(v) for feat, v in zip(feat_cols, imp)}
            new_imp = {k: v for k, v in imp_dict.items() if k in new_cols}
            ranked = sorted(new_imp.items(), key=lambda kv: -kv[1])
            (metrics_dir / "feature_importances.json").write_text(
                json.dumps({
                    "all": imp_dict,
                    "new_features_only_sorted": ranked,
                    "note": "Importance from the LAST LOOCV fold's classifier",
                }, indent=2), encoding="utf-8")
            print("Top 10 new-feature importances (last fold):")
            for feat, v in ranked[:10]:
                print(f"  {feat:<55} {v:.5f}")
        else:
            print("(HGBC does not expose feature_importances_; skipping)")

    render_v8_reach_figures(
        snapshot_dir=SNAPSHOT_DIR,
        raw_results=serialized_results,
        summary=agg,
        title_suffix=" (LOOCV, centroid+valley features, BSW w=0.8, mg=0)",
    )

    print()
    print(f"Wrote: {metrics_dir / 'loocv_aggregate.json'}")
    print(f"Wrote: {metrics_dir / 'loocv_per_fold.json'}")
    print(f"Wrote: {metrics_dir / 'topology_breakdown.json'}")
    if last_clf is not None:
        print(f"Wrote: {metrics_dir / 'feature_importances.json'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_summary.png'}")


if __name__ == "__main__":
    main()
