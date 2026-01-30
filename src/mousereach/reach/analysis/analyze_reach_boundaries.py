#!/usr/bin/env python3
"""
Analyze what distinguishes reach boundaries in the human annotations.

Key question: How does the human decide where one reach ends and another begins?
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict


def load_dlc_data(h5_path: Path) -> pd.DataFrame:
    """Load DLC tracking data."""
    df = pd.read_hdf(h5_path)
    new_cols = []
    for col in df.columns:
        bodypart = col[1]
        coord = col[2]
        new_cols.append(f'{bodypart}_{coord}')
    df.columns = new_cols
    return df


def load_human_touched_gt(gt_path: Path) -> List[Dict]:
    """Load only human-touched ground truth reaches."""
    with open(gt_path) as f:
        data = json.load(f)

    reaches = []
    for segment in data.get('segments', []):
        for reach in segment.get('reaches', []):
            if reach.get('source') == 'human_added' or reach.get('human_corrected', False):
                reaches.append({
                    'reach_id': reach.get('reach_id', len(reaches) + 1),
                    'start_frame': reach['start_frame'],
                    'end_frame': reach['end_frame'],
                    'segment_num': segment.get('segment_num', 0)
                })
    return sorted(reaches, key=lambda r: r['start_frame'])


def analyze_inter_reach_gaps(gt_reaches: List[Dict], dlc_df: pd.DataFrame):
    """Analyze what happens between consecutive reaches."""
    print("=" * 70)
    print("INTER-REACH GAP ANALYSIS (Consecutive Human-Touched Reaches)")
    print("=" * 70)

    hand_points = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']
    threshold = 0.3

    # Get median slit center
    slit_x = (dlc_df['BOXL_x'].median() + dlc_df['BOXR_x'].median()) / 2
    slit_y = (dlc_df['BOXL_y'].median() + dlc_df['BOXR_y'].median()) / 2

    gaps = []

    for i in range(len(gt_reaches) - 1):
        r1 = gt_reaches[i]
        r2 = gt_reaches[i + 1]

        gap = r2['start_frame'] - r1['end_frame'] - 1

        # Focus on SHORT gaps (where algorithm might merge)
        if gap <= 10 and gap >= 0:
            # Analyze the gap frames
            gap_data = {
                'r1_end': r1['end_frame'],
                'r2_start': r2['start_frame'],
                'gap': gap,
                'r1_id': r1['reach_id'],
                'r2_id': r2['reach_id'],
                'frames': []
            }

            # Analyze frames from r1 end to r2 start
            for frame in range(r1['end_frame'] - 2, r2['start_frame'] + 3):
                if frame < 0 or frame >= len(dlc_df):
                    continue
                row = dlc_df.iloc[frame]

                # Nose
                nose_x = row.get('Nose_x', np.nan)
                nose_y = row.get('Nose_y', np.nan)
                nose_l = row.get('Nose_likelihood', 0)
                nose_dist = np.sqrt((nose_x - slit_x)**2 + (nose_y - slit_y)**2) if not np.isnan(nose_x) else np.nan

                # Hand likelihoods
                hand_ls = {p: row.get(f'{p}_likelihood', 0) for p in hand_points}
                best_l = max(hand_ls.values())
                any_visible = best_l >= threshold

                gap_data['frames'].append({
                    'frame': frame,
                    'nose_dist': nose_dist,
                    'best_hand_l': best_l,
                    'any_visible': any_visible,
                    'in_r1': frame <= r1['end_frame'],
                    'in_r2': frame >= r2['start_frame'],
                    'in_gap': r1['end_frame'] < frame < r2['start_frame']
                })

            gaps.append(gap_data)

    print(f"\nFound {len(gaps)} consecutive reach pairs with gap <= 10 frames")

    # Analyze gap patterns
    gap_with_invisible = 0
    gap_all_visible = 0
    gap_drop_then_rise = 0

    for g in gaps:
        gap_frames = [f for f in g['frames'] if f['in_gap']]
        if not gap_frames:
            continue

        min_l_in_gap = min(f['best_hand_l'] for f in gap_frames)
        any_invisible_in_gap = any(not f['any_visible'] for f in gap_frames)

        if any_invisible_in_gap:
            gap_with_invisible += 1
        else:
            gap_all_visible += 1

        # Check for drop then rise
        r1_end_frames = [f for f in g['frames'] if f['in_r1']]
        r2_start_frames = [f for f in g['frames'] if f['in_r2']]
        if r1_end_frames and r2_start_frames:
            l_at_r1_end = r1_end_frames[-1]['best_hand_l']
            l_at_r2_start = r2_start_frames[0]['best_hand_l']
            if l_at_r1_end > 0.5 and min_l_in_gap < 0.5 and l_at_r2_start > 0.5:
                gap_drop_then_rise += 1

    print(f"\nGap characteristics:")
    print(f"  Hand goes invisible (<0.3) in gap: {gap_with_invisible}")
    print(f"  Hand stays visible throughout gap: {gap_all_visible}")
    print(f"  Confidence drops below 0.5 in gap: {gap_drop_then_rise}")

    # Print detailed examples
    print("\n" + "-" * 70)
    print("EXAMPLE SHORT GAPS (showing confidence trajectory)")
    print("-" * 70)

    for g in gaps[:15]:
        print(f"\n## Gap between reach #{g['r1_id']} (ends {g['r1_end']}) and #{g['r2_id']} (starts {g['r2_start']})")
        print(f"   Gap size: {g['gap']} frames")

        for f in g['frames']:
            marker = ""
            if f['frame'] == g['r1_end']:
                marker = " <-- R1 END"
            elif f['frame'] == g['r2_start']:
                marker = " <-- R2 START"
            elif f['in_gap']:
                marker = " (GAP)"

            visible_str = "VIS" if f['any_visible'] else "   "
            print(f"   {f['frame']:>6}: hand_l={f['best_hand_l']:.2f} {visible_str} nose_d={f['nose_dist']:>5.1f}{marker}")


def analyze_reach_end_confidence(gt_reaches: List[Dict], dlc_df: pd.DataFrame):
    """Analyze hand confidence at reach ends."""
    print("\n" + "=" * 70)
    print("CONFIDENCE AT REACH END")
    print("=" * 70)

    hand_points = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']

    confidences_at_end = []
    confidences_after_end = []

    for r in gt_reaches:
        end_frame = r['end_frame']
        if end_frame >= len(dlc_df) or end_frame + 1 >= len(dlc_df):
            continue

        row_end = dlc_df.iloc[end_frame]
        row_after = dlc_df.iloc[end_frame + 1]

        best_l_end = max(row_end.get(f'{p}_likelihood', 0) for p in hand_points)
        best_l_after = max(row_after.get(f'{p}_likelihood', 0) for p in hand_points)

        confidences_at_end.append(best_l_end)
        confidences_after_end.append(best_l_after)

    print(f"\nHand confidence AT reach end:")
    print(f"  Mean: {np.mean(confidences_at_end):.2f}")
    print(f"  Median: {np.median(confidences_at_end):.2f}")
    print(f"  Min: {np.min(confidences_at_end):.2f}")
    print(f"  % >= 0.5: {100 * np.mean(np.array(confidences_at_end) >= 0.5):.1f}%")
    print(f"  % >= 0.7: {100 * np.mean(np.array(confidences_at_end) >= 0.7):.1f}%")

    print(f"\nHand confidence AFTER reach end (1 frame later):")
    print(f"  Mean: {np.mean(confidences_after_end):.2f}")
    print(f"  Median: {np.median(confidences_after_end):.2f}")
    print(f"  % < 0.3: {100 * np.mean(np.array(confidences_after_end) < 0.3):.1f}%")
    print(f"  % < 0.5: {100 * np.mean(np.array(confidences_after_end) < 0.5):.1f}%")

    # Confidence drop
    drops = np.array(confidences_at_end) - np.array(confidences_after_end)
    print(f"\nConfidence DROP at reach end:")
    print(f"  Mean drop: {np.mean(drops):.2f}")
    print(f"  Median drop: {np.median(drops):.2f}")
    print(f"  % with drop >= 0.3: {100 * np.mean(drops >= 0.3):.1f}%")


def main():
    from mousereach.config import Paths
    processing_dir = Paths.PROCESSING
    video_id = "20251021_CNT0405_P4"

    dlc_path = processing_dir / f"{video_id}DLC_resnet50_MPSAOct27shuffle1_100000.h5"
    gt_path = processing_dir / f"{video_id}_reach_ground_truth.json"

    print("Loading data...")
    dlc_df = load_dlc_data(dlc_path)
    gt_reaches = load_human_touched_gt(gt_path)

    print(f"Human-touched GT reaches: {len(gt_reaches)}")

    analyze_inter_reach_gaps(gt_reaches, dlc_df)
    analyze_reach_end_confidence(gt_reaches, dlc_df)


if __name__ == "__main__":
    main()
