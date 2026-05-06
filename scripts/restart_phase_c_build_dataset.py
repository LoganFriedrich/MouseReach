"""
Phase C step 1: build the per-segment training dataset for the v5
outcome classifier.

For each video in the train_pool + test_holdout, and for each segment
within that video:
  1. Load DLC h5 once
  2. Compute per-frame features once (v8 features)
  3. For each segment from algo_outputs/{video}_segments.json:
       - Aggregate per-frame features over the segment + post-pad
         window into a per-segment feature vector
       - Pull GT outcome label from gt/{video}_unified_ground_truth.json
       - Save row

Output:
  Improvement_Snapshots/_corpus/.../phase_c_dataset/train_pool.parquet
  Improvement_Snapshots/_corpus/.../phase_c_dataset/test_holdout.parquet

Schema:
  meta: video_id, segment_num, outcome_label, interaction_frame,
        outcome_known_frame, exhaustive, cohort, segment_start_frame,
        segment_end_frame
  features: ~1219 per-segment aggregated features
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.outcomes.v5.features import extract_segment_features, feature_columns as seg_feature_columns
from mousereach.outcomes.v5.labels import load_segment_labels
from mousereach.reach.v8.features import extract_features as extract_per_frame_features
from mousereach.reach.v8.features import load_dlc_h5


QUARANTINE = Path(
    r"Y:\2_Connectome\Validation_Runs\DLC_2026_03_27\iterations"
    r"\2026-04-28_outcome_v4.0.0_dev_walkthrough"
)
DLC_DIR = QUARANTINE / "dlc"
GT_DIR = QUARANTINE / "gt"
ALGO_DIR = QUARANTINE / "algo_outputs"

CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
OUTPUT_DIR = CORPUS_DIR / "phase_c_dataset"


def cohort_from_video(video_id: str) -> str:
    parts = video_id.split("_")
    if len(parts) >= 2 and parts[1].startswith("CNT"):
        cnt = parts[1]
        if len(cnt) == 7:
            return f"CNT_{cnt[3:5]}"
    return "UNKNOWN"


def find_dlc(video_id: str) -> Path:
    return next(DLC_DIR.glob(f"{video_id}DLC_*.h5"))


def load_segments(video_id: str) -> list:
    """Load algo segment boundaries from algo_outputs/_segments.json.

    Returns list of dicts with segment_num, start_frame, end_frame.
    """
    path = ALGO_DIR / f"{video_id}_segments.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    boundaries = data.get("boundaries", []) or []
    if boundaries:
        # Convert pairs of consecutive boundaries into segments
        segs = []
        for i in range(len(boundaries) - 1):
            segs.append({
                "segment_num": i + 1,
                "start_frame": int(boundaries[i]),
                "end_frame": int(boundaries[i + 1]) - 1,
            })
        return segs

    # Fallback: explicit segments list
    raw_segs = data.get("segments", []) or []
    return [
        {"segment_num": int(s.get("segment_num", i + 1)),
         "start_frame": int(s["start_frame"]),
         "end_frame": int(s["end_frame"])}
        for i, s in enumerate(raw_segs)
    ]


def build_split(video_ids: list, split_name: str, inventory_lookup: dict) -> pd.DataFrame:
    rows = []
    for vid in video_ids:
        dlc = load_dlc_h5(find_dlc(vid))
        per_frame = extract_per_frame_features(dlc)
        segs = load_segments(vid)
        gt_segs = {r["segment_num"]: r for r in load_segment_labels(GT_DIR / f"{vid}_unified_ground_truth.json")}

        for s in segs:
            sn = s["segment_num"]
            gt = gt_segs.get(sn)
            if gt is None:
                continue  # GT and algo seg numbers should match; skip if not
            feats = extract_segment_features(
                dlc_df=dlc, seg_start=s["start_frame"], seg_end=s["end_frame"],
                per_frame_feats=per_frame)
            row = {
                "video_id": vid,
                "segment_num": sn,
                "segment_start_frame": s["start_frame"],
                "segment_end_frame": s["end_frame"],
                "outcome_label": gt["outcome_label"],
                "interaction_frame": gt["interaction_frame"],
                "outcome_known_frame": gt["outcome_known_frame"],
                "exhaustive": gt["exhaustive"],
                "cohort": inventory_lookup[vid]["cohort"],
            }
            row.update(feats)
            rows.append(row)
        print(f"  [{split_name}] {vid}: {len(segs)} segments", flush=True)
    return pd.DataFrame(rows)


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
    print("PHASE C OUTCOME DATASET BUILD")
    print("=" * 70)
    print(f"Train pool:   {len(train_ids)} videos")
    print(f"Test holdout: {len(test_ids)} videos")
    print()

    print("Building train pool ...")
    train_df = build_split(train_ids, "train", inventory_lookup)
    print()
    print("Building test holdout ...")
    test_df = build_split(test_ids, "test", inventory_lookup)
    print()

    print(f"Total train segments: {len(train_df)}")
    print(f"Total test segments:  {len(test_df)}")
    print()
    print("Outcome label distribution (train):")
    print(train_df["outcome_label"].value_counts())
    print()
    print("Outcome label distribution (test):")
    print(test_df["outcome_label"].value_counts())
    print()

    feat_cols = seg_feature_columns()
    print(f"Per-segment feature count: {len(feat_cols)}")
    n_nan = int(train_df[feat_cols].isna().sum().sum())
    n_inf = int(np.isinf(train_df[feat_cols].to_numpy()).sum())
    print(f"NaN in train features: {n_nan}")
    print(f"Inf in train features: {n_inf}")
    print()

    # Save
    print("Writing parquets ...")
    train_df.to_parquet(OUTPUT_DIR / "train_pool.parquet", index=False, engine="pyarrow")
    test_df.to_parquet(OUTPUT_DIR / "test_holdout.parquet", index=False, engine="pyarrow")
    print(f"  Train: {(OUTPUT_DIR / 'train_pool.parquet').stat().st_size / 1024**2:.1f} MB")
    print(f"  Test:  {(OUTPUT_DIR / 'test_holdout.parquet').stat().st_size / 1024**2:.1f} MB")
    print()
    print("=" * 70)
    print("PHASE C DATASET READY")
    print("=" * 70)


if __name__ == "__main__":
    main()
