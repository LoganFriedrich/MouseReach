"""
Phase B dataset build -- MODEL 3.1 DLC variant.

Mirrors `restart_phase_b_build_dataset.py` but sources DLC h5s from the
new model 3.1 inference directory. Output goes to a new corpus dir so
the existing 2026-04-30 corpus is untouched.

Per dlc_model_3_1_corpus_state_2026-05-21 memory:
- `updated dlc model 3.1\` (Mar 28 mtime) and `generalization_test_2026-05-11/dlc\`
  (Apr 27 mtime) are the SAME DLC model weights, just two inference batches.
- All 16 exhaustive train_pool videos are covered by model 3.1.
- All 4 exhaustive cv_folds test_holdout videos are covered.
- Non-exhaustive train_pool videos (21) are NOT covered. Build skips them.

GT files are unchanged (GT is independent of DLC).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.features import (extract_features, load_dlc_h5,
                                          feature_columns)
from mousereach.reach.v8.labels import (load_gt_reaches, per_frame_labels,
                                        reach_id_per_frame)


# === Paths ===

NEW_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs"
    r"\DLC_2026_03_27\Processing\updated dlc model 3.1"
)

# GT lives in the calibration validation_runs dir, independent of DLC vintage
QUARANTINE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs"
    r"\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough"
)
GT_DIR = QUARANTINE / "gt"

# Reference corpus for inventory + cv_folds
SOURCE_CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)

# New corpus output
NEW_CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-05-21_model_3_1_inventory"
)
OUTPUT_DIR = NEW_CORPUS_DIR / "phase_b_dataset"


def find_dlc(video_id: str) -> Path:
    files = list(NEW_DLC_DIR.glob(f"{video_id}DLC_*.h5"))
    if not files:
        raise FileNotFoundError(f"No DLC h5 in model 3.1 dir for {video_id}")
    return files[0]


def find_gt(video_id: str) -> Path:
    return GT_DIR / f"{video_id}_unified_ground_truth.json"


def build_split(video_ids: list, split_name: str, inventory_lookup: dict,
                available_video_ids: set) -> pd.DataFrame:
    """Build per-frame dataframe. Skip videos not in available_video_ids."""
    frames_per_video = []
    skipped = []
    for vid in video_ids:
        if vid not in available_video_ids:
            skipped.append(vid)
            continue
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
        full = pd.concat([meta.reset_index(drop=True),
                          feats.reset_index(drop=True)], axis=1)
        frames_per_video.append(full)

        print(f"  [{split_name}] {vid}: n_frames={n_frames}  "
              f"n_in_reach={int(labels.sum())}  "
              f"n_reaches={len(reaches)}  exhaustive={exhaustive}",
              flush=True)

    if skipped:
        print(f"  [{split_name}] SKIPPED {len(skipped)} videos missing from model 3.1 DLC:")
        for v in skipped:
            print(f"    - {v}", flush=True)

    if not frames_per_video:
        return pd.DataFrame()
    return pd.concat(frames_per_video, axis=0, ignore_index=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE B DATASET BUILD -- MODEL 3.1 DLC")
    print("=" * 70)
    print(f"  New DLC source: {NEW_DLC_DIR}")
    print(f"  GT source:      {GT_DIR}")
    print(f"  Output:         {OUTPUT_DIR}")
    print()

    # Available videos in new DLC
    available = set(
        f.name.replace("DLC_resnet50_MPSAOct27shuffle1_100000.h5", "")
        for f in NEW_DLC_DIR.iterdir()
        if f.suffix == ".h5"
    )
    print(f"Available videos in model 3.1 DLC dir: {len(available)}")
    print()

    # Load source corpus inventory + folds
    inventory = json.loads(
        (SOURCE_CORPUS_DIR / "inventory.json").read_text(encoding="utf-8"))
    folds = json.loads(
        (SOURCE_CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))

    inventory_lookup = {r["video_id"]: r for r in inventory["videos"]}
    train_ids = folds["train_pool"]["video_ids"]
    test_ids = folds["test_holdout"]["video_ids"]

    print(f"Source train_pool: {len(train_ids)} videos")
    print(f"Source test_holdout: {len(test_ids)} videos")
    print()

    print("Building train pool...")
    train_df = build_split(train_ids, "train", inventory_lookup, available)
    print()
    print("Building test holdout...")
    test_df = build_split(test_ids, "test", inventory_lookup, available)
    print()

    feat_cols = feature_columns()
    n_features = len(feat_cols)
    print(f"Feature count: {n_features}")
    if len(train_df):
        print(f"Train frames: {len(train_df):>10,}  "
              f"({int(train_df['label'].sum()):>7,} in-reach, "
              f"{train_df['label'].mean()*100:.2f}%)")
    if len(test_df):
        print(f"Test frames:  {len(test_df):>10,}  "
              f"({int(test_df['label'].sum()):>7,} in-reach, "
              f"{test_df['label'].mean()*100:.2f}%)")
    print()

    # Per-stratum breakdown
    if len(train_df):
        print("Train pool by exhaustive flag:")
        for exh in [True, False]:
            sub = train_df[train_df["exhaustive"] == exh]
            if len(sub) > 0:
                tag = "exhaustive (gold)" if exh else "supplementary"
                print(f"  {tag:>22s}: n_videos={sub['video_id'].nunique():>3}  "
                      f"n_frames={len(sub):>10,}  "
                      f"in_reach={int(sub['label'].sum()):>7,}")
        print()

    # NaN/inf scan
    if len(train_df):
        feat_view = train_df[feat_cols]
        n_nan = int(feat_view.isna().sum().sum())
        n_inf = int(np.isinf(feat_view.to_numpy()).sum())
        print(f"NaN scan in features: {n_nan} (any non-zero needs investigation)")
        print(f"Inf scan in features: {n_inf}")
        print()

    # Save
    train_path = OUTPUT_DIR / "train_pool.parquet"
    test_path = OUTPUT_DIR / "test_holdout.parquet"

    if len(train_df):
        print(f"Writing {train_path} ...", flush=True)
        train_df.to_parquet(train_path, index=False, engine="pyarrow")
        print(f"  size: {train_path.stat().st_size / 1024**2:.1f} MB")
    if len(test_df):
        print(f"Writing {test_path} ...", flush=True)
        test_df.to_parquet(test_path, index=False, engine="pyarrow")
        print(f"  size: {test_path.stat().st_size / 1024**2:.1f} MB")

    # Copy cv_folds.json + write inventory for self-containedness
    target_folds = NEW_CORPUS_DIR / "cv_folds.json"
    shutil.copy(SOURCE_CORPUS_DIR / "cv_folds.json", target_folds)
    print(f"Copied cv_folds.json from source corpus.")

    # Write a marker file documenting the DLC source
    marker = NEW_CORPUS_DIR / "DLC_SOURCE.txt"
    marker.write_text(
        "DLC h5 source: " + str(NEW_DLC_DIR) + "\n"
        "GT source: " + str(GT_DIR) + "\n"
        "Source corpus referenced for inventory + folds: " + str(SOURCE_CORPUS_DIR) + "\n"
        "Build date: 2026-05-21\n"
        "Model: model 3.1 (canonical new DLC; same weights as 2026-05-11 generalization holdout)\n",
        encoding="utf-8",
    )
    print(f"Wrote: {marker}")

    print()
    print("=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
