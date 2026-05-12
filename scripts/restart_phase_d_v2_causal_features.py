"""
Phase D v2: rebuild Phase D dataset with the new causal-attribution
features layered on top of the existing baseline features. Re-run
LOOCV with the combined feature set. Compare directional confusion
to the v1 baseline.

Reuses the existing per-reach rows + baseline features from
phase_d_dataset/, just adds the new causal-attribution features per
reach computed from DLC + the segment context.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.assignment.v1.features import (
    extract_reach_causal_features, causal_feature_columns,
    feature_columns as baseline_feature_columns)
from mousereach.assignment.v1.train import loocv as v1_loocv, FoldResult
from mousereach.lib.causal_attribution import (
    classify_end_state, find_off_pillar_transition_frame,
    find_displaced_signature_runs)
from mousereach.reach.v8.features import load_dlc_h5


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
QUARANTINE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations"
    r"\2026-04-28_outcome_v4.0.0_dev_walkthrough"
)
DLC_DIR = QUARANTINE / "dlc"
ALGO_DIR = QUARANTINE / "algo_outputs"
OUT_PARQUET_DIR = CORPUS_DIR / "phase_d_dataset_v2_causal"
SNAPSHOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\assignment\v1.1.0_dev_causal_features_loocv"
)


def find_dlc(video_id: str) -> Path:
    return next(DLC_DIR.glob(f"{video_id}DLC_*.h5"))


def load_segments_by_num(video_id: str) -> dict:
    seg_data = json.loads(
        (ALGO_DIR / f"{video_id}_segments.json").read_text(encoding="utf-8"))
    boundaries = seg_data.get("boundaries", []) or []
    return {i + 1: (int(boundaries[i]), int(boundaries[i + 1]) - 1)
            for i in range(len(boundaries) - 1)}


def main():
    OUT_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.mkdir(parents=True, exist_ok=True)

    # Load existing Phase D dataset
    base_train = pd.read_parquet(
        CORPUS_DIR / "phase_d_dataset" / "train_pool.parquet")
    print(f"Baseline train: {len(base_train)} reaches across "
          f"{base_train['video_id'].nunique()} videos")

    # Compute causal features per reach
    causal_rows = []
    causal_cols = causal_feature_columns()
    print(f"Causal feature count: {len(causal_cols)}")

    for vid in sorted(base_train["video_id"].unique()):
        vid_reaches = base_train[base_train["video_id"] == vid]
        dlc = load_dlc_h5(find_dlc(vid))
        seg_bounds = load_segments_by_num(vid)

        # Cache segment-level computations
        seg_caches = {}
        for sn in vid_reaches["segment_num"].unique():
            sn = int(sn)
            seg_start, seg_end = seg_bounds.get(sn, (None, None))
            if seg_start is None:
                continue
            es = classify_end_state(dlc, seg_end)
            transition_frame = find_off_pillar_transition_frame(
                dlc, seg_start, seg_end, es)
            displaced = find_displaced_signature_runs(dlc, seg_start, seg_end)
            seg_caches[sn] = {
                "end_state": es,
                "transition": {"frame": transition_frame},
                "displaced": displaced,
            }

        for _, r in vid_reaches.iterrows():
            sn = int(r["segment_num"])
            seg_start, seg_end = seg_bounds.get(sn, (None, None))
            if seg_start is None:
                continue
            cache = seg_caches.get(sn, {})
            feats = extract_reach_causal_features(
                dlc_df=dlc,
                reach_start=int(r["reach_start_frame"]),
                reach_end=int(r["reach_end_frame"]),
                seg_start=seg_start, seg_end=seg_end,
                end_state_cache=cache.get("end_state"),
                transition_cache=cache.get("transition"),
                displaced_cache=cache.get("displaced"),
            )
            causal_row = {
                "video_id": vid,
                "segment_num": sn,
                "reach_id": int(r["reach_id"]),
            }
            causal_row.update(feats)
            causal_rows.append(causal_row)
        print(f"  {vid}: {len(vid_reaches)} reaches", flush=True)

    causal_df = pd.DataFrame(causal_rows)
    print(f"Causal features computed: {len(causal_df)} rows")

    # Merge baseline + causal features by (video_id, segment_num, reach_id)
    merged = base_train.merge(
        causal_df, on=["video_id", "segment_num", "reach_id"], how="left",
        suffixes=("", "__causal_dup"))
    # Drop dup columns if any
    dup_cols = [c for c in merged.columns if c.endswith("__causal_dup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    print(f"Merged dataset: {len(merged)} rows, {len(merged.columns)} cols")
    n_nan = int(merged[causal_cols].isna().sum().sum())
    n_inf = int(np.isinf(merged[causal_cols].to_numpy()).sum())
    print(f"NaN in causal cols: {n_nan}  Inf: {n_inf}")
    if n_nan > 0:
        # Fill NaN with 0 (unmatched segments etc.)
        for c in causal_cols:
            merged[c] = merged[c].fillna(0.0)
    merged.to_parquet(OUT_PARQUET_DIR / "train_pool.parquet",
                      index=False, engine="pyarrow")

    # Run LOOCV with extended feature set: baseline + causal
    extended_cols = baseline_feature_columns() + causal_cols
    print(f"\nRunning LOOCV with extended features ({len(extended_cols)}) ...")

    # Monkey-patch the feature_columns so loocv uses extended set
    import mousereach.assignment.v1.train as v1_train
    import mousereach.assignment.v1.features as v1_features
    original_fc = v1_features.feature_columns
    v1_features.feature_columns = lambda: extended_cols
    try:
        train_pool_ids = sorted(merged["video_id"].unique().tolist())
        folds = v1_loocv(merged, train_pool_ids,
                         max_iter=200, learning_rate=0.05, max_depth=4)
    finally:
        v1_features.feature_columns = original_fc

    # Aggregate per-segment causal-attribution accuracy
    rows = []
    for f in folds:
        rows.extend(f.rows)

    seg_results = {}
    for r in rows:
        key = (r["video_id"], r["segment_num"])
        if key not in seg_results:
            seg_results[key] = {"outcome": r["segment_outcome"],
                                "predicted_causal": None, "gt_causal": None}
        if r["pred_causal"] == 1:
            seg_results[key]["predicted_causal"] = (r["reach_id"],
                                                    r["reach_start_frame"])
        if r["gt_causal"] == 1:
            seg_results[key]["gt_causal"] = (r["reach_id"],
                                             r["reach_start_frame"])

    by_outcome = defaultdict(lambda: {"n_segments": 0, "causal_correct": 0})
    for k, s in seg_results.items():
        out = s["outcome"]
        by_outcome[out]["n_segments"] += 1
        if (s["predicted_causal"] is not None and
                s["gt_causal"] is not None and
                s["predicted_causal"] == s["gt_causal"]):
            by_outcome[out]["causal_correct"] += 1

    print()
    print("=" * 80)
    print("CAUSAL ATTRIBUTION (extended features)")
    print("=" * 80)
    for outcome in ("retrieved", "displaced_sa"):
        n = by_outcome[outcome]["n_segments"]
        c = by_outcome[outcome]["causal_correct"]
        print(f"  {outcome:<20}: {c}/{n} ({100*c/n:.1f}%)" if n else
              f"  {outcome:<20}: 0/0")

    metrics_dir = SNAPSHOT / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    (metrics_dir / "loocv_results.json").write_text(json.dumps({
        "n_folds": len(folds),
        "by_outcome": dict(by_outcome),
        "rows": rows,
        "feature_count": len(extended_cols),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {metrics_dir / 'loocv_results.json'}")


if __name__ == "__main__":
    main()
