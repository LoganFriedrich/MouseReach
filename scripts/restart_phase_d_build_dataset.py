"""
Phase D step 1: build the per-reach training dataset for the v1
assignment classifier.

For each video, for each TOUCHED GT segment (retrieved or
displaced_sa), for each GT reach in that segment:
  - Compute per-reach features (mousereach.assignment.v1.features)
  - Label = 1 if reach span contains GT interaction_frame, else 0

Untouched and abnormal_exception segments are not included in
training (no causal reach to predict).

Output:
  Improvement_Snapshots/_corpus/.../phase_d_dataset/train_pool.parquet
  Improvement_Snapshots/_corpus/.../phase_d_dataset/test_holdout.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.assignment.v1.features import (
    extract_reach_features, feature_columns)
from mousereach.reach.v8.features import load_dlc_h5


QUARANTINE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations"
    r"\2026-04-28_outcome_v4.0.0_dev_walkthrough"
)
DLC_DIR = QUARANTINE / "dlc"
GT_DIR = QUARANTINE / "gt"
ALGO_DIR = QUARANTINE / "algo_outputs"

CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
OUTPUT_DIR = CORPUS_DIR / "phase_d_dataset"

TOUCHED = {"retrieved", "displaced_sa", "displaced_outside"}


def cohort_from_video(video_id: str) -> str:
    parts = video_id.split("_")
    if len(parts) >= 2 and parts[1].startswith("CNT"):
        return f"CNT_{parts[1][3:5]}"
    return "UNKNOWN"


def find_dlc(video_id: str) -> Path:
    return next(DLC_DIR.glob(f"{video_id}DLC_*.h5"))


def load_segments_by_num(video_id: str) -> dict:
    seg_data = json.loads(
        (ALGO_DIR / f"{video_id}_segments.json").read_text(encoding="utf-8"))
    boundaries = seg_data.get("boundaries", []) or []
    out = {}
    for i in range(len(boundaries) - 1):
        out[i + 1] = (int(boundaries[i]), int(boundaries[i + 1]) - 1)
    return out


def load_gt(video_id: str) -> dict:
    return json.loads(
        (GT_DIR / f"{video_id}_unified_ground_truth.json").read_text(encoding="utf-8"))


def build_split(video_ids: list, split_name: str) -> pd.DataFrame:
    rows = []
    for vid in video_ids:
        dlc = load_dlc_h5(find_dlc(vid))
        gt = load_gt(vid)

        gt_outcomes = {
            s["segment_num"]: s for s in gt.get("outcomes", {}).get("segments", []) or []
        }
        gt_reaches = gt.get("reaches", {}).get("reaches", []) or []
        seg_bounds = load_segments_by_num(vid)

        # Group GT reaches by segment_num
        reaches_by_seg: dict = {}
        for r in gt_reaches:
            sn = r.get("segment_num")
            if sn is not None:
                reaches_by_seg.setdefault(int(sn), []).append(r)

        n_reaches_video = 0
        for sn, seg_reaches in reaches_by_seg.items():
            seg_meta = gt_outcomes.get(sn)
            if seg_meta is None:
                continue
            outcome = seg_meta.get("outcome")
            if outcome == "displaced_outside":
                outcome = "displaced_sa"
            if outcome not in TOUCHED - {"displaced_outside"}:
                continue  # Only train on touched segments

            interaction_frame = seg_meta.get("interaction_frame")
            if interaction_frame is None:
                continue

            seg_start, seg_end = seg_bounds.get(sn, (None, None))
            if seg_start is None:
                continue

            # Sort reaches by start_frame so reach_order is deterministic
            ordered = sorted(seg_reaches, key=lambda r: r.get("start_frame", 0))
            n_reaches = len(ordered)

            for order, r in enumerate(ordered):
                rs = r.get("start_frame")
                re = r.get("end_frame")
                if rs is None or re is None:
                    continue

                feats = extract_reach_features(
                    dlc_df=dlc,
                    reach_start=int(rs),
                    reach_end=int(re),
                    seg_start=int(seg_start),
                    seg_end=int(seg_end),
                    reach_order=order,
                    n_reaches_in_segment=n_reaches,
                )

                # Causal label: this reach contains interaction_frame?
                causal = 1 if (rs <= interaction_frame <= re) else 0

                row = {
                    "video_id": vid,
                    "segment_num": sn,
                    "reach_id": int(r.get("reach_id", order)),
                    "reach_start_frame": int(rs),
                    "reach_end_frame": int(re),
                    "segment_outcome": outcome,
                    "interaction_frame": int(interaction_frame),
                    "exhaustive": bool(gt.get("outcomes", {}).get("exhaustive", False)),
                    "cohort": cohort_from_video(vid),
                    "causal": causal,
                }
                row.update(feats)
                rows.append(row)
                n_reaches_video += 1

        print(f"  [{split_name}] {vid}: {n_reaches_video} GT reaches in touched segments",
              flush=True)
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = json.loads(
        (CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))
    train_ids = folds["train_pool"]["video_ids"]
    test_ids = folds["test_holdout"]["video_ids"]

    print("=" * 70)
    print("PHASE D ASSIGNMENT DATASET BUILD")
    print("=" * 70)
    print()

    print("Building train pool ...")
    train_df = build_split(train_ids, "train")
    print()
    print("Building test holdout ...")
    test_df = build_split(test_ids, "test")
    print()

    feat_cols = feature_columns()
    print(f"Per-reach feature count: {len(feat_cols)}")
    print(f"Train rows: {len(train_df)}  causal={int(train_df['causal'].sum())}")
    print(f"Test rows:  {len(test_df)}   causal={int(test_df['causal'].sum())}")

    # Sanity
    n_nan = int(train_df[feat_cols].isna().sum().sum())
    n_inf = int(np.isinf(train_df[feat_cols].to_numpy()).sum())
    print(f"NaN: {n_nan}, Inf: {n_inf}")
    print()
    print("Causal-vs-miss balance per outcome (train):")
    for out in ("retrieved", "displaced_sa"):
        sub = train_df[train_df["segment_outcome"] == out]
        print(f"  {out}: n_reaches={len(sub)}, causal={int(sub['causal'].sum())}, "
              f"miss={len(sub) - int(sub['causal'].sum())}")
    print()

    print("Writing parquets ...")
    train_df.to_parquet(OUTPUT_DIR / "train_pool.parquet", index=False, engine="pyarrow")
    test_df.to_parquet(OUTPUT_DIR / "test_holdout.parquet", index=False, engine="pyarrow")
    print(f"  Train: {(OUTPUT_DIR / 'train_pool.parquet').stat().st_size / 1024**2:.1f} MB")
    print(f"  Test:  {(OUTPUT_DIR / 'test_holdout.parquet').stat().st_size / 1024**2:.1f} MB")
    print()
    print("=" * 70)
    print("PHASE D DATASET READY")
    print("=" * 70)


if __name__ == "__main__":
    main()
