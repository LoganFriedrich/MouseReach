#!/usr/bin/env python3
"""
Debug reach fragmentation - examine why long GT reaches are being split.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


def load_dlc_data(h5_path: Path) -> pd.DataFrame:
    """Load DLC tracking data."""
    df = pd.read_hdf(h5_path)

    # Flatten columns
    new_cols = []
    for col in df.columns:
        bodypart = col[1]
        coord = col[2]
        new_cols.append(f'{bodypart}_{coord}')

    df.columns = new_cols
    return df


def analyze_fragmented_reach(dlc_df: pd.DataFrame, gt_start: int, gt_end: int):
    """Analyze why a GT reach might be fragmented."""
    print(f"\n## Analyzing GT reach frames {gt_start}-{gt_end} (duration: {gt_end - gt_start + 1})")

    # Get slit center
    boxl_x = dlc_df.iloc[gt_start:gt_end+1]['BOXL_x'].median()
    boxr_x = dlc_df.iloc[gt_start:gt_end+1]['BOXR_x'].median()
    boxl_y = dlc_df.iloc[gt_start:gt_end+1]['BOXL_y'].median()
    boxr_y = dlc_df.iloc[gt_start:gt_end+1]['BOXR_y'].median()
    slit_x = (boxl_x + boxr_x) / 2
    slit_y = (boxl_y + boxr_y) / 2

    print(f"  Slit center: ({slit_x:.1f}, {slit_y:.1f})")

    hand_points = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']
    threshold = 0.3

    # Sample frames throughout the reach
    sample_frames = list(range(gt_start, min(gt_end + 1, gt_start + 50)))
    if gt_end - gt_start > 50:
        sample_frames += list(range(gt_start + 50, gt_end + 1, 10))

    print(f"\n  Frame-by-frame analysis:")
    print(f"  {'Frame':>8} | {'NoseD':>6} | {'NoseL':>5} | {'Hand':>5} | {'HandL':>6} | {'Issue':>20}")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*20}")

    # Track issues
    nose_disengage_count = 0
    hand_invisible_count = 0
    consecutive_hand_invisible = 0
    max_consecutive_invisible = 0

    for frame in sample_frames:
        if frame >= len(dlc_df):
            continue

        row = dlc_df.iloc[frame]

        # Nose engagement
        nose_x = row.get('Nose_x', np.nan)
        nose_y = row.get('Nose_y', np.nan)
        nose_l = row.get('Nose_likelihood', 0)
        nose_dist = np.sqrt((nose_x - slit_x)**2 + (nose_y - slit_y)**2) if not np.isnan(nose_x) else np.nan
        nose_engaged = nose_dist < 20 if not np.isnan(nose_dist) else False

        # Hand visibility
        any_hand_visible = False
        best_hand_l = 0
        for p in hand_points:
            l = row.get(f'{p}_likelihood', 0)
            if l >= threshold:
                any_hand_visible = True
            if l > best_hand_l:
                best_hand_l = l

        # Track issues
        issue = ""
        if not nose_engaged and nose_l > 0.3:
            issue = "NOSE DISENGAGED"
            nose_disengage_count += 1
        if not any_hand_visible:
            issue = "HAND INVISIBLE"
            hand_invisible_count += 1
            consecutive_hand_invisible += 1
            max_consecutive_invisible = max(max_consecutive_invisible, consecutive_hand_invisible)
        else:
            consecutive_hand_invisible = 0

        if issue or frame < gt_start + 5 or frame > gt_end - 5 or frame % 20 == 0:
            print(f"  {frame:>8} | {nose_dist:>6.1f} | {nose_l:>5.2f} | {any_hand_visible!s:>5} | {best_hand_l:>6.2f} | {issue:>20}")

    print(f"\n  Summary:")
    print(f"    Nose disengage events: {nose_disengage_count}")
    print(f"    Hand invisible events: {hand_invisible_count}")
    print(f"    Max consecutive hand invisible: {max_consecutive_invisible}")

    if max_consecutive_invisible >= 3:
        print(f"    ** LIKELY CAUSE: Hand invisible for {max_consecutive_invisible} consecutive frames **")


def main():
    from mousereach.config import Paths
    processing_dir = Paths.PROCESSING
    video_id = "20251021_CNT0405_P4"

    dlc_path = processing_dir / f"{video_id}DLC_resnet50_MPSAOct27shuffle1_100000.h5"
    gt_path = processing_dir / f"{video_id}_reach_ground_truth.json"

    print("Loading DLC data...")
    dlc_df = load_dlc_data(dlc_path)

    # Load GT
    with open(gt_path) as f:
        gt_data = json.load(f)

    gt_reaches = []
    for segment in gt_data.get('segments', []):
        for reach in segment.get('reaches', []):
            gt_reaches.append(reach)

    # Find long reaches that were likely fragmented
    long_reaches = sorted(
        [r for r in gt_reaches if r['end_frame'] - r['start_frame'] > 50],
        key=lambda r: r['end_frame'] - r['start_frame'],
        reverse=True
    )

    print(f"\nAnalyzing top 3 longest GT reaches:")
    for reach in long_reaches[:3]:
        analyze_fragmented_reach(dlc_df, reach['start_frame'], reach['end_frame'])


if __name__ == "__main__":
    main()
