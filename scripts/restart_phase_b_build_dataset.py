"""
Phase B step 1: build the per-frame training dataset for reach detection v8.

For each video in the train_pool (37 videos, per the locked CV folds):
  - Load DLC h5
  - Extract v1 feature matrix (mousereach.reach.v8.features.extract_features)
  - Load GT reaches and per-frame in-reach labels
  - Concatenate metadata columns (video_id, frame, exhaustive, reach_id)

Output: a single parquet file with all per-frame rows, ready for the
trainer. Test-holdout videos are NOT included; they're saved
separately with the same schema for Phase E.

Sanity-prints:
  - shape per split
  - per-class label balance, separately for exhaustive and supplementary
  - missing-feature scan (should be zero NaN/inf after extraction)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.features import extract_features, load_dlc_h5, feature_columns
from mousereach.reach.v8.labels import load_gt_reaches, per_frame_labels, reach_id_per_frame


QUARANTINE = Path(
    r"Y:\2_Connectome\Validation_Runs\DLC_2026_03_27\iterations"
    r"\2026-04-28_outcome_v4.0.0_dev_walkthrough"
)
DLC_DIR = QUARANTINE / "dlc"
GT_DIR = QUARANTINE / "gt"

CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
OUTPUT_DIR = CORPUS_DIR / "phase_b_dataset"


def find_dlc(video_id: str) -> Path:
    files = list(DLC_DIR.glob(f"{video_id}DLC_*.h5"))
    if not files:
        raise FileNotFoundError(f"No DLC h5 for {video_id}")
    return files[0]


def find_gt(video_id: str) -> Path:
    return GT_DIR / f"{video_id}_unified_ground_truth.json"


def build_split(video_ids: list, split_name: str, inventory_lookup: dict) -> pd.DataFrame:
    """Build the per-frame dataframe for a list of videos."""
    frames_per_video = []
    for vid in video_ids:
        dlc_path = find_dlc(vid)
        gt_path = find_gt(vid)

        dlc = load_dlc_h5(dlc_path)
        feats = extract_features(dlc)
        n_frames = len(feats)

        reaches, exhaustive = load_gt_reaches(gt_path)
        labels = per_frame_labels(n_frames, reaches)
        reach_ids = reach_id_per_frame(n_frames, reaches)

        meta = pd.DataFrame({
            "video_id": vid,
            "frame": np.arange(n_frames, dtype=np.int32),
            "label": labels,
            "exhaustive": exhaustive,
            "reach_id": reach_ids,
            "cohort": inventory_lookup[vid]["cohort"],
        })
        # Concat horizontally; meta columns first
        full = pd.concat([meta.reset_index(drop=True),
                          feats.reset_index(drop=True)], axis=1)
        frames_per_video.append(full)

        print(f"  [{split_name}] {vid}: n_frames={n_frames}  "
              f"n_in_reach={int(labels.sum())}  "
              f"n_reaches={len(reaches)}  exhaustive={exhaustive}")

    return pd.concat(frames_per_video, axis=0, ignore_index=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(
        (CORPUS_DIR / "inventory.json").read_text(encoding="utf-8"))
    folds = json.loads(
        (CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))

    inventory_lookup = {r["video_id"]: r for r in inventory["videos"]}
    train_ids = folds["train_pool"]["video_ids"]
    test_ids = folds["test_holdout"]["video_ids"]

    print("=" * 70)
    print("PHASE B DATASET BUILD")
    print("=" * 70)
    print(f"Train pool: {len(train_ids)} videos")
    print(f"Test holdout: {len(test_ids)} videos (frozen until Phase E)")
    print()

    print("Building train pool...")
    train_df = build_split(train_ids, "train", inventory_lookup)
    print()
    print("Building test holdout...")
    test_df = build_split(test_ids, "test", inventory_lookup)
    print()

    # Sanity checks
    feat_cols = feature_columns()
    n_features = len(feat_cols)
    print(f"Feature count: {n_features}")
    print(f"Train frames:  {len(train_df):>10,}  ({train_df['label'].sum():>7,} in-reach, "
          f"{train_df['label'].mean()*100:.2f}%)")
    print(f"Test frames:   {len(test_df):>10,}  ({test_df['label'].sum():>7,} in-reach, "
          f"{test_df['label'].mean()*100:.2f}%)")
    print()

    # Per-stratum breakdown
    print("Train pool by exhaustive flag:")
    for exh in [True, False]:
        sub = train_df[train_df["exhaustive"] == exh]
        if len(sub) > 0:
            tag = "exhaustive (gold)" if exh else "supplementary"
            print(f"  {tag:>22s}: n_frames={len(sub):>10,}  "
                  f"in_reach={int(sub['label'].sum()):>7,}  "
                  f"({sub['label'].mean()*100:.2f}%)")
    print()

    # Per-cohort breakdown (train only)
    print("Train pool by cohort:")
    for cohort in sorted(train_df["cohort"].unique()):
        sub = train_df[train_df["cohort"] == cohort]
        print(f"  {cohort}: n_frames={len(sub):>10,}  "
              f"in_reach={int(sub['label'].sum()):>7,}  "
              f"({sub['label'].mean()*100:.2f}%)")
    print()

    # NaN/inf scan
    feat_view = train_df[feat_cols]
    n_nan = int(feat_view.isna().sum().sum())
    n_inf = int(np.isinf(feat_view.to_numpy()).sum())
    print(f"NaN scan in features: {n_nan} (any non-zero needs investigation)")
    print(f"Inf scan in features: {n_inf}")
    print()

    # Save
    train_path = OUTPUT_DIR / "train_pool.parquet"
    test_path = OUTPUT_DIR / "test_holdout.parquet"
    print(f"Writing {train_path} ...")
    train_df.to_parquet(train_path, index=False, engine="pyarrow")
    print(f"Writing {test_path} ...")
    test_df.to_parquet(test_path, index=False, engine="pyarrow")
    print()

    print(f"Train file size: {train_path.stat().st_size / 1024**2:.1f} MB")
    print(f"Test file size:  {test_path.stat().st_size / 1024**2:.1f} MB")
    print()
    print("=" * 70)
    print("PHASE B DATASET READY")
    print("=" * 70)


if __name__ == "__main__":
    main()
