"""
v8 baseline LOOCV on model 3.1 DLC.

Re-establishes the cumulative best (v8 features only + BSW b=1/w=0.8 + mg=0)
against the model 3.1 DLC parquet. Mirrors restart_phase_b_loocv_boundary_sample_weight_w08.py
verbatim except for paths (new corpus + new snapshot dir).

Pre-experiment checklist applied (per pre_experiment_checklist.md):

1. Cumulative-stacking check:
   - Replicates current production v8.0.1 exactly: v8 baseline features (405)
     + BSW b=1/w=0.8 + mg=0 + min_span=3.
   - Old-DLC cumulative best was TP=2077 / FP=297 / FN=298 (parquet GT, pre-rescore).
   - New-DLC baseline will be the NEW cumulative best for all subsequent experiments
     on this corpus.

2. Existing-code-modification check: NO modifications to src/mousereach/reach/v8/*.

3. Unverified hypothesis:
   - The model 3.1 DLC will not catastrophically change v8 behavior.
   - Plausible outcomes: similar TP/FP/FN (DLC consistency intact); better (new
     DLC produces cleaner keypoints); worse (some videos may be affected).

4. FN-direction-reporting check: leads with delta FN vs old-DLC cumulative best.

5. Framework check: outputs to canonical snapshot dir
   `Improvement_Snapshots/reach_detection/v8.0.1_model_3_1_baseline_loocv/`.

6. Branch: feature/v8-centroid-valley-features (current branch; this baseline
   is the comparison anchor for any retrain experiment). Same branch is fine
   since both runs need new-DLC parquet.

7. Decision rule:
   - This is a BASELINE establishment, not a comparison. Accept whatever
     numbers come out; they become the new cumulative best.
   - Sanity flag: if new-DLC baseline is dramatically worse than old-DLC
     baseline (e.g., TP drops > 50, FN rises > 50), surface for investigation
     before treating as baseline.
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
from mousereach.reach.v8.features import feature_columns
from mousereach.improvement.reach_detection.v8_figures import render_v8_reach_figures


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-05-21_model_3_1_inventory"
)

SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.1_model_3_1_baseline_loocv"
)

# Production v8.0.1 parameters
THRESHOLD = 0.5
MERGE_GAP = 0
MIN_SPAN = 3
BOUNDARY_BUFFER = 1
BOUNDARY_WEIGHT = 0.8


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


def classify_topology(algo_reaches, gt_reaches):
    """Connected-components topology classifier (same as in centroid_valley
    runner). Returns list of (label, n_algo, n_gt) per component."""
    def overlap(a_s, a_e, b_s, b_e):
        return not (a_e < b_s or a_s > b_e)

    n_algo, n_gt = len(algo_reaches), len(gt_reaches)
    algo_to_gt, gt_to_algo = defaultdict(set), defaultdict(set)
    for i, a in enumerate(algo_reaches):
        for j, g in enumerate(gt_reaches):
            if overlap(a.start_frame, a.end_frame, g.start_frame, g.end_frame):
                algo_to_gt[i].add(j)
                gt_to_algo[j].add(i)

    visited_a, visited_g = set(), set()
    components = []
    for i in range(n_algo):
        if i in visited_a: continue
        if not algo_to_gt[i]:
            components.append(("FALSE_POSITIVE", 1, 0))
            visited_a.add(i)
            continue
        algo_in, gt_in = set(), set()
        queue = [("a", i)]
        while queue:
            kind, idx = queue.pop()
            if kind == "a":
                if idx in algo_in: continue
                algo_in.add(idx)
                for gj in algo_to_gt[idx]: queue.append(("g", gj))
            else:
                if idx in gt_in: continue
                gt_in.add(idx)
                for ai in gt_to_algo[idx]: queue.append(("a", ai))
        visited_a.update(algo_in)
        visited_g.update(gt_in)

        na, ng = len(algo_in), len(gt_in)
        if na == 1 and ng == 1:
            a = algo_reaches[next(iter(algo_in))]
            g = gt_reaches[next(iter(gt_in))]
            start_d = a.start_frame - g.start_frame
            a_span = a.end_frame - a.start_frame + 1
            g_span = g.end_frame - g.start_frame + 1
            span_d = a_span - g_span
            span_tol = max(0.5 * g_span, 5)
            if abs(start_d) <= 2 and abs(span_d) <= span_tol:
                components.append(("TP", 1, 1))
            else:
                components.append(("TOLERANCE_ERROR", 1, 1))
        elif na == 1 and ng >= 2:
            components.append(("MERGED", na, ng))
        elif na >= 2 and ng == 1:
            components.append(("FRAGMENTED", na, ng))
        elif na >= 2 and ng >= 2:
            components.append(("COMPLEX", na, ng))

    for j in range(n_gt):
        if j not in visited_g:
            components.append(("FALSE_NEGATIVE", 0, 1))

    return components


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

    algo_raw = probabilities_to_reaches(
        proba, threshold=THRESHOLD, merge_gap=MERGE_GAP, min_span=MIN_SPAN)
    algo_reaches = [
        AlgoReach(start_frame=r.start_frame, end_frame=r.end_frame,
                  video_id=val_vid, index=i) for i, r in enumerate(algo_raw)
    ]

    sub = val.sort_values("frame")
    rid = sub["reach_id"].to_numpy()
    frames = sub["frame"].to_numpy()
    gt_reaches = []
    for ri in sorted(set(rid[rid >= 0].tolist())):
        f = frames[rid == ri]
        gt_reaches.append(GTReach(
            start_frame=int(f.min()), end_frame=int(f.max()),
            video_id=val_vid, index=ri))

    results = evaluate_reaches(algo_reaches, gt_reaches, video_id=val_vid)
    summary = summarize_results(results)
    topology = classify_topology(algo_reaches, gt_reaches)
    return summary, results, algo_reaches, gt_reaches, topology


def main():
    print("=" * 70)
    print("PHASE B LOOCV BASELINE -- MODEL 3.1 DLC")
    print(f"  v8 baseline features only (405); BSW b={BOUNDARY_BUFFER}/w={BOUNDARY_WEIGHT}; mg={MERGE_GAP}; thr={THRESHOLD}")
    print("=" * 70)
    print()

    print("Loading model 3.1 parquet...", flush=True)
    parquet_path = CORPUS_DIR / "phase_b_dataset" / "train_pool.parquet"
    if not parquet_path.exists():
        print(f"[FAIL] missing: {parquet_path}")
        sys.exit(1)
    df = pd.read_parquet(parquet_path)
    print(f"  loaded {len(df)} rows across {df['video_id'].nunique()} videos")
    print()

    folds_def = json.loads(
        (CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))
    train_pool_ids = folds_def["train_pool"]["video_ids"]
    exh_set = set(df.loc[df["exhaustive"], "video_id"].unique().tolist())
    eligible_val = [v for v in train_pool_ids if v in exh_set]
    print(f"LOOCV: {len(eligible_val)} exhaustive folds (videos present in parquet)")
    print()

    feat_cols = feature_columns()
    print(f"Feature count: {len(feat_cols)}")
    print()

    folds = []
    per_video_data = {}
    all_results_combined = []
    all_topology = {}

    for i, val_vid in enumerate(eligible_val):
        train_ids = [v for v in train_pool_ids if v != val_vid]
        print(f"  fold {i+1}/{len(eligible_val)}: val={val_vid}", flush=True)
        summary, results, algo_reaches, gt_reaches, topology = train_one_fold(
            df, train_ids, val_vid, feat_cols)

        s = summary
        topo_counts = defaultdict(int)
        for lbl, _, _ in topology: topo_counts[lbl] += 1
        topo_str = "  ".join(f"{k}={v}" for k, v in sorted(topo_counts.items())
                             if k in ("MERGED", "FRAGMENTED", "COMPLEX"))
        sd_mean = s['tp_start_delta']['mean']
        sd_mean_str = f"{sd_mean:.3f}" if sd_mean is not None else "n/a"
        print(f"    TP={s['n_tp']:>4} FP={s['n_fp']:>4} FN={s['n_fn']:>4}  "
              f"start abs_med={s['tp_start_delta']['abs_median']} mean={sd_mean_str}  "
              f"span abs_med={s['tp_span_delta']['abs_median']}  {topo_str}",
              flush=True)
        folds.append({"val_video_ids": [val_vid], "summary": summary,
                      "topology_counts": dict(topo_counts)})
        per_video_data[val_vid] = (algo_reaches, gt_reaches)
        all_results_combined.extend(results)
        all_topology[val_vid] = topology

    print()
    agg = summarize_results(all_results_combined)
    topo_total = defaultdict(int)
    for vid, topo in all_topology.items():
        for lbl, _, _ in topo: topo_total[lbl] += 1

    print("=" * 70)
    print("AGGREGATE LOOCV BASELINE -- MODEL 3.1 DLC")
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

    # Compare to OLD-DLC cumulative best
    old_dlc_tp, old_dlc_fp, old_dlc_fn = 2077, 297, 298
    print("Compare against OLD-DLC cumulative best (v8.0.1 mg=0, pre-rescore parquet GT):")
    print(f"  Old DLC:  TP={old_dlc_tp}  FP={old_dlc_fp}  FN={old_dlc_fn}")
    print(f"  New DLC:  TP={agg['n_tp']}  FP={agg['n_fp']}  FN={agg['n_fn']}")
    print()
    print("FN direction:")
    print(f"  delta FN vs old-DLC cumulative best: {agg['n_fn'] - old_dlc_fn:+d}")
    print(f"  delta TP vs old-DLC cumulative best: {agg['n_tp'] - old_dlc_tp:+d}")
    print(f"  delta FP vs old-DLC cumulative best: {agg['n_fp'] - old_dlc_fp:+d}")
    print()

    # Save outputs
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    serialized = []
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
        serialized.append(record)

    (metrics_dir / "loocv_per_fold.json").write_text(
        json.dumps(folds, indent=2), encoding="utf-8")
    (metrics_dir / "loocv_aggregate.json").write_text(
        json.dumps({
            "n_folds": len(folds), "summary": agg, "raw_results": serialized,
            "merge_gap": MERGE_GAP, "boundary_buffer": BOUNDARY_BUFFER,
            "boundary_weight": BOUNDARY_WEIGHT,
            "dlc_source": "model 3.1 (canonical new DLC)",
            "schema_version": "v8_baseline_model_3_1",
        }, indent=2), encoding="utf-8")

    (metrics_dir / "topology_breakdown.json").write_text(
        json.dumps({
            "total": dict(topo_total),
            "per_video": {
                vid: {l: sum(1 for ll, _, _ in topo if ll == l)
                      for l in ("TP", "TOLERANCE_ERROR", "MERGED", "FRAGMENTED",
                                "FALSE_POSITIVE", "FALSE_NEGATIVE", "COMPLEX")}
                for vid, topo in all_topology.items()
            }
        }, indent=2), encoding="utf-8")

    render_v8_reach_figures(
        snapshot_dir=SNAPSHOT_DIR, raw_results=serialized, summary=agg,
        title_suffix=" (LOOCV, model 3.1 DLC, BSW w=0.8, mg=0)")

    print(f"Wrote: {metrics_dir / 'loocv_aggregate.json'}")
    print(f"Wrote: {metrics_dir / 'loocv_per_fold.json'}")
    print(f"Wrote: {metrics_dir / 'topology_breakdown.json'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_summary.png'}")


if __name__ == "__main__":
    main()
