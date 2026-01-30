#!/usr/bin/env python3
"""
Analyze boundary errors between algorithm detections and ground truth.

Focus on cases where there IS overlap but IoU < threshold.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


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


def compute_overlap(start1: int, end1: int, start2: int, end2: int) -> Tuple[float, int, int]:
    """Compute IoU and overlap details."""
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    if intersection_end < intersection_start:
        return 0.0, 0, 0

    intersection = intersection_end - intersection_start + 1
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection

    iou = intersection / union if union > 0 else 0.0
    return iou, intersection, union


def load_human_touched_gt(gt_path: Path) -> List[Dict]:
    """Load only human-touched ground truth reaches."""
    with open(gt_path) as f:
        data = json.load(f)

    reaches = []
    for segment in data.get('segments', []):
        for reach in segment.get('reaches', []):
            # Only human-touched
            if reach.get('source') == 'human_added' or reach.get('human_corrected', False):
                reaches.append({
                    'reach_id': reach.get('reach_id', len(reaches) + 1),
                    'start_frame': reach['start_frame'],
                    'end_frame': reach['end_frame'],
                    'source': reach.get('source', 'unknown'),
                    'human_corrected': reach.get('human_corrected', False),
                    'segment_num': segment.get('segment_num', 0)
                })
    return reaches


def load_detected(det_path: Path) -> List[Dict]:
    """Load algorithm-detected reaches."""
    with open(det_path) as f:
        data = json.load(f)

    reaches = []
    for segment in data.get('segments', []):
        for reach in segment.get('reaches', []):
            reaches.append({
                'reach_id': reach['reach_id'],
                'start_frame': reach['start_frame'],
                'end_frame': reach['end_frame'],
                'segment_num': segment.get('segment_num', 0)
            })
    return reaches


def analyze_boundary_mismatches(
    gt_reaches: List[Dict],
    det_reaches: List[Dict],
    dlc_df: pd.DataFrame,
    iou_threshold: float = 0.3
):
    """
    Detailed analysis of cases with overlap but IoU < threshold.
    """
    print("=" * 70)
    print("BOUNDARY MISMATCH ANALYSIS")
    print("=" * 70)

    mismatches = []

    for gt in gt_reaches:
        gt_start = gt['start_frame']
        gt_end = gt['end_frame']
        gt_dur = gt_end - gt_start + 1

        # Find all detections that overlap with this GT
        overlapping = []
        for det in det_reaches:
            iou, intersection, union = compute_overlap(
                gt_start, gt_end,
                det['start_frame'], det['end_frame']
            )
            if intersection > 0:
                overlapping.append({
                    'det': det,
                    'iou': iou,
                    'intersection': intersection,
                    'start_diff': det['start_frame'] - gt_start,
                    'end_diff': det['end_frame'] - gt_end
                })

        # Check if any match with IoU >= threshold
        has_match = any(o['iou'] >= iou_threshold for o in overlapping)

        if not has_match and overlapping:
            # This is a boundary mismatch - overlap exists but IoU too low
            mismatches.append({
                'gt': gt,
                'overlapping': overlapping
            })

    print(f"\nFound {len(mismatches)} boundary mismatches (overlap exists but IoU < {iou_threshold})")

    if not mismatches:
        return

    # Categorize mismatches
    start_too_late = 0
    start_too_early = 0
    end_too_late = 0
    end_too_early = 0
    fragmented = 0
    merged = 0

    print("\n" + "-" * 70)
    print("DETAILED MISMATCH ANALYSIS")
    print("-" * 70)

    for mm in mismatches:
        gt = mm['gt']
        overlapping = mm['overlapping']

        print(f"\n## GT #{gt['reach_id']} (seg {gt['segment_num']}): frames {gt['start_frame']}-{gt['end_frame']} (dur={gt['end_frame']-gt['start_frame']+1})")
        print(f"   Source: {gt['source']}, Human corrected: {gt['human_corrected']}")

        if len(overlapping) > 1:
            print(f"   FRAGMENTED: {len(overlapping)} detections overlap this GT")
            fragmented += 1
        elif len(overlapping) == 1:
            o = overlapping[0]
            det = o['det']
            print(f"   Detection: {det['start_frame']}-{det['end_frame']} (dur={det['end_frame']-det['start_frame']+1})")
            print(f"   IoU: {o['iou']:.3f}, Intersection: {o['intersection']} frames")
            print(f"   Start diff: {o['start_diff']:+d}, End diff: {o['end_diff']:+d}")

            if o['start_diff'] > 3:
                start_too_late += 1
                print(f"   -> Algorithm START too LATE by {o['start_diff']} frames")
            elif o['start_diff'] < -3:
                start_too_early += 1
                print(f"   -> Algorithm START too EARLY by {-o['start_diff']} frames")

            if o['end_diff'] > 3:
                end_too_late += 1
                print(f"   -> Algorithm END too LATE by {o['end_diff']} frames")
            elif o['end_diff'] < -3:
                end_too_early += 1
                print(f"   -> Algorithm END too EARLY by {-o['end_diff']} frames")

        # Check multiple overlapping detections for each
        for o in overlapping[:3]:
            det = o['det']
            print(f"     - Det #{det['reach_id']}: {det['start_frame']}-{det['end_frame']} (IoU={o['iou']:.3f}, start_diff={o['start_diff']:+d}, end_diff={o['end_diff']:+d})")

    print("\n" + "=" * 70)
    print("SUMMARY OF ERROR TYPES")
    print("=" * 70)
    print(f"Start too late (>3 frames):   {start_too_late}")
    print(f"Start too early (>3 frames):  {start_too_early}")
    print(f"End too late (>3 frames):     {end_too_late}")
    print(f"End too early (>3 frames):    {end_too_early}")
    print(f"Fragmented (multiple dets):   {fragmented}")

    return mismatches


def analyze_dlc_at_boundaries(
    mismatches: List[Dict],
    dlc_df: pd.DataFrame
):
    """
    Look at DLC tracking data at the boundaries to understand errors.
    """
    print("\n" + "=" * 70)
    print("DLC TRACKING AT ERROR BOUNDARIES")
    print("=" * 70)

    hand_points = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']
    threshold = 0.3

    # Get median slit center (approximate)
    slit_x = (dlc_df['BOXL_x'].median() + dlc_df['BOXR_x'].median()) / 2
    slit_y = (dlc_df['BOXL_y'].median() + dlc_df['BOXR_y'].median()) / 2

    for mm in mismatches[:10]:  # First 10
        gt = mm['gt']
        print(f"\n## GT #{gt['reach_id']}: frames {gt['start_frame']}-{gt['end_frame']}")

        # Check frames around GT start
        print(f"\n   Around GT START ({gt['start_frame']}):")
        for frame in range(max(0, gt['start_frame'] - 3), min(len(dlc_df), gt['start_frame'] + 4)):
            row = dlc_df.iloc[frame]

            # Nose
            nose_x = row.get('Nose_x', np.nan)
            nose_y = row.get('Nose_y', np.nan)
            nose_l = row.get('Nose_likelihood', 0)
            nose_dist = np.sqrt((nose_x - slit_x)**2 + (nose_y - slit_y)**2) if not np.isnan(nose_x) else np.nan

            # Hand visibility
            any_hand_visible = False
            best_hand_l = 0
            for p in hand_points:
                l = row.get(f'{p}_likelihood', 0)
                if l >= threshold:
                    any_hand_visible = True
                if l > best_hand_l:
                    best_hand_l = l

            marker = "<<<" if frame == gt['start_frame'] else ""
            print(f"   {frame:>6}: nose_d={nose_dist:>5.1f}, nose_l={nose_l:.2f}, hand={any_hand_visible!s:>5}, best_l={best_hand_l:.2f} {marker}")

        # Check frames around GT end
        print(f"\n   Around GT END ({gt['end_frame']}):")
        for frame in range(max(0, gt['end_frame'] - 3), min(len(dlc_df), gt['end_frame'] + 4)):
            row = dlc_df.iloc[frame]

            nose_x = row.get('Nose_x', np.nan)
            nose_y = row.get('Nose_y', np.nan)
            nose_l = row.get('Nose_likelihood', 0)
            nose_dist = np.sqrt((nose_x - slit_x)**2 + (nose_y - slit_y)**2) if not np.isnan(nose_x) else np.nan

            any_hand_visible = False
            best_hand_l = 0
            for p in hand_points:
                l = row.get(f'{p}_likelihood', 0)
                if l >= threshold:
                    any_hand_visible = True
                if l > best_hand_l:
                    best_hand_l = l

            marker = "<<<" if frame == gt['end_frame'] else ""
            print(f"   {frame:>6}: nose_d={nose_dist:>5.1f}, nose_l={nose_l:.2f}, hand={any_hand_visible!s:>5}, best_l={best_hand_l:.2f} {marker}")


def main():
    from mousereach.config import Paths
    processing_dir = Paths.PROCESSING
    video_id = "20251021_CNT0405_P4"

    dlc_path = processing_dir / f"{video_id}DLC_resnet50_MPSAOct27shuffle1_100000.h5"
    gt_path = processing_dir / f"{video_id}_reach_ground_truth.json"
    det_path = processing_dir / f"{video_id}_reaches.json"

    print("Loading data...")
    dlc_df = load_dlc_data(dlc_path)
    gt_reaches = load_human_touched_gt(gt_path)
    det_reaches = load_detected(det_path)

    print(f"Human-touched GT reaches: {len(gt_reaches)}")
    print(f"Algorithm detections: {len(det_reaches)}")

    mismatches = analyze_boundary_mismatches(gt_reaches, det_reaches, dlc_df, iou_threshold=0.3)

    if mismatches:
        analyze_dlc_at_boundaries(mismatches, dlc_df)


if __name__ == "__main__":
    main()
