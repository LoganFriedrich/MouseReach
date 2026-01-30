"""
annotate_discrepancies.py - Add review notes to reaches for discrepancy review

Compares algorithm detections against ground truth and adds review_note
fields to reaches JSON for flagged cases that need human review.

Notes appear in the review widget as [!] markers with yellow highlighting.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DiscrepancyConfig:
    """Configuration for discrepancy detection."""
    iou_threshold: float = 0.3          # IoU threshold for matching
    boundary_diff_threshold: int = 5    # Frames difference to flag
    min_confidence_threshold: float = 0.6  # Below this = low confidence
    nose_distance_threshold: int = 25   # Pixels from slit


def compute_iou(start1: int, end1: int, start2: int, end2: int) -> float:
    """Compute IoU between two frame ranges."""
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start + 1)

    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection
    return intersection / union if union > 0 else 0


def get_hand_confidence(dlc_df: pd.DataFrame, start: int, end: int) -> float:
    """Get mean hand confidence for a reach."""
    # Handle both tuple columns and string columns
    hand_cols = []
    for c in dlc_df.columns:
        col_str = str(c)
        if 'RH' in col_str and 'likelihood' in col_str:
            hand_cols.append(c)

    if not hand_cols:
        return 0.0

    segment = dlc_df.iloc[start:end+1]
    max_likelihoods = segment[hand_cols].max(axis=1)
    return float(max_likelihoods.mean())


def get_nose_distance(dlc_df: pd.DataFrame, start: int, end: int) -> float:
    """Get mean nose distance from slit center."""
    try:
        segment = dlc_df.iloc[start:end+1]

        # Handle column naming (could be tuple or flat)
        def get_col(name, coord):
            if (name, coord) in segment.columns:
                return segment[(name, coord)]
            elif f'{name}_{coord}' in segment.columns:
                return segment[f'{name}_{coord}']
            return None

        # Get BOXL and BOXR x positions to find slit center
        boxl_x = get_col('BOXL', 'x')
        boxr_x = get_col('BOXR', 'x')
        nose_x = get_col('Nose', 'x')  # Note: DLC uses 'Nose' not 'NOSE'

        if boxl_x is None or boxr_x is None or nose_x is None:
            return 999

        slit_x = (boxl_x.mean() + boxr_x.mean()) / 2
        return abs(nose_x.mean() - slit_x)
    except (AttributeError, ValueError, TypeError) as e:
        return 999  # Unknown - DLC data unavailable or invalid


def annotate_reaches(
    reaches_path: Path,
    gt_path: Path,
    dlc_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[DiscrepancyConfig] = None
) -> Dict:
    """
    Add review notes to reaches based on GT comparison.

    Returns dict with annotation statistics.
    """
    if config is None:
        config = DiscrepancyConfig()

    if output_path is None:
        output_path = reaches_path

    # Load data
    with open(reaches_path) as f:
        reaches_data = json.load(f)

    with open(gt_path) as f:
        gt_data = json.load(f)

    # Load DLC for confidence analysis
    dlc_df = pd.read_hdf(dlc_path)
    if isinstance(dlc_df.columns, pd.MultiIndex):
        dlc_df.columns = dlc_df.columns.droplevel(0)

    # Build flat lists of reaches
    algo_reaches = []
    for seg in reaches_data['segments']:
        for r in seg['reaches']:
            algo_reaches.append({
                'seg_num': seg['segment_num'],
                'reach': r,
                'start': r['start_frame'],
                'end': r['end_frame']
            })

    gt_reaches = []
    for seg in gt_data['segments']:
        for r in seg['reaches']:
            gt_reaches.append({
                'seg_num': seg['segment_num'],
                'start': r['start_frame'],
                'end': r['end_frame'],
                'source': r.get('source', 'algorithm'),
                'corrected': r.get('human_corrected', False)
            })

    # Match algorithm to GT
    matched_algo = set()
    matched_gt = set()
    matches = []

    for i, a in enumerate(algo_reaches):
        best_iou = 0
        best_j = -1
        for j, g in enumerate(gt_reaches):
            if j in matched_gt:
                continue
            iou = compute_iou(a['start'], a['end'], g['start'], g['end'])
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= config.iou_threshold and best_j >= 0:
            matched_algo.add(i)
            matched_gt.add(best_j)
            matches.append((i, best_j, best_iou))

    stats = {
        'total_algo': len(algo_reaches),
        'total_gt': len(gt_reaches),
        'matched': len(matches),
        'false_positives': len(algo_reaches) - len(matches),
        'false_negatives': len(gt_reaches) - len(matches),
        'notes_added': 0
    }

    # Clear existing notes
    for seg in reaches_data['segments']:
        for r in seg['reaches']:
            r['review_note'] = None

    # Annotate boundary mismatches
    for algo_idx, gt_idx, iou in matches:
        a = algo_reaches[algo_idx]
        g = gt_reaches[gt_idx]

        start_diff = a['start'] - g['start']
        end_diff = a['end'] - g['end']

        notes = []

        if abs(start_diff) > config.boundary_diff_threshold:
            direction = "early" if start_diff < 0 else "late"
            notes.append(f"Start {abs(start_diff)}f {direction}")

        if abs(end_diff) > config.boundary_diff_threshold:
            direction = "early" if end_diff < 0 else "late"
            notes.append(f"End {abs(end_diff)}f {direction}")

        if notes:
            a['reach']['review_note'] = "BOUNDARY: " + ", ".join(notes)
            stats['notes_added'] += 1

    # Annotate false positives (algo found, no GT match)
    for i, a in enumerate(algo_reaches):
        if i in matched_algo:
            continue

        hand_conf = get_hand_confidence(dlc_df, a['start'], a['end'])
        nose_dist = get_nose_distance(dlc_df, a['start'], a['end'])
        duration = a['end'] - a['start'] + 1

        # Check for partial overlap with GT
        best_overlap = 0
        for g in gt_reaches:
            iou = compute_iou(a['start'], a['end'], g['start'], g['end'])
            best_overlap = max(best_overlap, iou)

        notes = []

        if best_overlap > 0.1:
            notes.append(f"Partial GT overlap IoU={best_overlap:.2f}")
        elif hand_conf >= 0.7:
            notes.append(f"Strong tracking ({hand_conf:.0%}), no GT - human missed?")
        elif hand_conf < config.min_confidence_threshold:
            notes.append(f"Low confidence ({hand_conf:.0%})")

        if nose_dist > config.nose_distance_threshold:
            notes.append(f"Nose far ({nose_dist:.0f}px)")

        if duration <= 2:
            notes.append("Very short (<=2f)")

        if notes:
            a['reach']['review_note'] = "FP: " + "; ".join(notes)
            stats['notes_added'] += 1
        else:
            a['reach']['review_note'] = "FP: No GT match"
            stats['notes_added'] += 1

    # Save updated reaches
    with open(output_path, 'w') as f:
        json.dump(reaches_data, f, indent=2)

    return stats


def main():
    """Annotate discrepancies for the 0405 video."""
    from mousereach.config import Paths
    processing = Paths.PROCESSING
    video_id = '20251021_CNT0405_P4'

    reaches_path = processing / f'{video_id}_reaches.json'
    gt_path = processing / f'{video_id}_reach_ground_truth.json'
    dlc_path = processing / f'{video_id}DLC_resnet50_MPSAOct27shuffle1_100000.h5'

    print(f"Annotating discrepancies for {video_id}...")
    print(f"  Reaches: {reaches_path}")
    print(f"  GT: {gt_path}")
    print(f"  DLC: {dlc_path}")

    stats = annotate_reaches(
        reaches_path=reaches_path,
        gt_path=gt_path,
        dlc_path=dlc_path,
        output_path=reaches_path  # Update in place
    )

    print(f"\nResults:")
    print(f"  Algorithm detections: {stats['total_algo']}")
    print(f"  Ground truth reaches: {stats['total_gt']}")
    print(f"  Matched: {stats['matched']}")
    print(f"  False positives: {stats['false_positives']}")
    print(f"  False negatives: {stats['false_negatives']}")
    print(f"  Review notes added: {stats['notes_added']}")
    print(f"\nUpdated: {reaches_path}")


if __name__ == '__main__':
    main()
