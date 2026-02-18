#!/usr/bin/env python3
"""
Generate a human-readable discrepancy report between algorithm and ground truth.
Flags statistically unusual cases for manual review.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def compute_iou(start1: int, end1: int, start2: int, end2: int) -> float:
    """Compute Intersection over Union."""
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    if inter_end < inter_start:
        return 0.0
    intersection = inter_end - inter_start + 1
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection
    return intersection / union if union > 0 else 0.0


def load_dlc_data(h5_path: Path) -> pd.DataFrame:
    """Load DLC tracking data."""
    df = pd.read_hdf(h5_path)
    new_cols = [f'{col[1]}_{col[2]}' for col in df.columns]
    df.columns = new_cols
    return df


def load_gt(gt_path: Path) -> List[Dict]:
    """Load all GT reaches."""
    with open(gt_path) as f:
        data = json.load(f)
    reaches = []
    for seg in data.get('segments', []):
        for r in seg.get('reaches', []):
            reaches.append({
                'start': r['start_frame'],
                'end': r['end_frame'],
                'duration': r['end_frame'] - r['start_frame'] + 1,
                'source': r.get('source', 'unknown'),
                'human_corrected': r.get('human_corrected', False),
                'segment': seg.get('segment_num', 0)
            })
    return reaches


def load_det(det_path: Path) -> List[Dict]:
    """Load algorithm detections."""
    with open(det_path) as f:
        data = json.load(f)
    reaches = []
    for seg in data.get('segments', []):
        for r in seg.get('reaches', []):
            reaches.append({
                'id': r['reach_id'],
                'start': r['start_frame'],
                'end': r['end_frame'],
                'duration': r['end_frame'] - r['start_frame'] + 1,
                'segment': seg.get('segment_num', 0)
            })
    return reaches


def match_reaches(gt_list: List[Dict], det_list: List[Dict], threshold: float = 0.3):
    """Match GT to detections, return matched pairs, FPs, and FNs."""
    used_det = set()
    matched = []

    for i, gt in enumerate(gt_list):
        best_j = None
        best_iou = 0
        for j, det in enumerate(det_list):
            if j in used_det:
                continue
            iou = compute_iou(gt['start'], gt['end'], det['start'], det['end'])
            if iou >= threshold and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j is not None:
            used_det.add(best_j)
            matched.append({
                'gt_idx': i,
                'det_idx': best_j,
                'gt': gt,
                'det': det_list[best_j],
                'iou': best_iou,
                'start_diff': det_list[best_j]['start'] - gt['start'],
                'end_diff': det_list[best_j]['end'] - gt['end']
            })

    fps = [det_list[j] for j in range(len(det_list)) if j not in used_det]
    fns = [gt_list[i] for i in range(len(gt_list)) if i not in [m['gt_idx'] for m in matched]]

    return matched, fps, fns


def analyze_tracking_at_reach(dlc_df: pd.DataFrame, start: int, end: int) -> Dict:
    """Get tracking quality info for a reach."""
    hand_points = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']

    # Get slit center (approximate)
    slit_x = (dlc_df['BOXL_x'].median() + dlc_df['BOXR_x'].median()) / 2
    slit_y = (dlc_df['BOXL_y'].median() + dlc_df['BOXR_y'].median()) / 2

    hand_likelihoods = []
    nose_distances = []

    for frame in range(max(0, start), min(len(dlc_df), end + 1)):
        row = dlc_df.iloc[frame]
        best_l = max(row.get(f'{p}_likelihood', 0) for p in hand_points)
        hand_likelihoods.append(best_l)

        nose_x = row.get('Nose_x', np.nan)
        nose_y = row.get('Nose_y', np.nan)
        if not np.isnan(nose_x):
            nose_distances.append(np.sqrt((nose_x - slit_x)**2 + (nose_y - slit_y)**2))

    return {
        'mean_hand_likelihood': np.mean(hand_likelihoods) if hand_likelihoods else 0,
        'min_hand_likelihood': np.min(hand_likelihoods) if hand_likelihoods else 0,
        'mean_nose_distance': np.mean(nose_distances) if nose_distances else 0,
        'max_nose_distance': np.max(nose_distances) if nose_distances else 0,
        'hand_visible_pct': 100 * np.mean(np.array(hand_likelihoods) >= 0.3) if hand_likelihoods else 0
    }


def main():
    from mousereach.config import Paths
    processing_dir = Paths.PROCESSING
    video_id = "20251021_CNT0405_P4"

    gt_path = processing_dir / f"{video_id}_reach_ground_truth.json"
    det_path = processing_dir / f"{video_id}_reaches.json"
    dlc_path = processing_dir / f"{video_id}DLC_resnet50_MPSAOct27shuffle1_100000.h5"

    print("Loading data...")
    gt_list = load_gt(gt_path)
    det_list = load_det(det_path)
    dlc_df = load_dlc_data(dlc_path)

    matched, fps, fns = match_reaches(gt_list, det_list, threshold=0.3)

    # Compute duration stats for context
    gt_durations = [r['duration'] for r in gt_list]
    det_durations = [r['duration'] for r in det_list]

    print("\n" + "=" * 80)
    print("ALGORITHM vs HUMAN GROUND TRUTH - DISCREPANCY REPORT")
    print("=" * 80)

    print(f"\n## OVERALL STATS")
    print(f"   Human labeled:     {len(gt_list)} reaches")
    print(f"   Algorithm found:   {len(det_list)} reaches")
    print(f"   Correctly matched: {len(matched)} ({100*len(matched)/len(gt_list):.1f}% of human labels)")
    print(f"")
    print(f"   False Negatives:   {len(fns)} (human marked, algo missed)")
    print(f"   False Positives:   {len(fps)} (algo found, human didn't mark)")

    precision = len(matched) / len(det_list) if det_list else 0
    recall = len(matched) / len(gt_list) if gt_list else 0

    print(f"\n   Precision: {precision:.1%}  (of algo detections, how many are correct)")
    print(f"   Recall:    {recall:.1%}  (of human labels, how many algo found)")

    # Frame accuracy for matched
    if matched:
        start_diffs = [m['start_diff'] for m in matched]
        end_diffs = [m['end_diff'] for m in matched]
        print(f"\n## FRAME ACCURACY (for {len(matched)} matched reaches)")
        print(f"   Start frame error: mean {np.mean(start_diffs):+.1f}, median {np.median(start_diffs):+.0f}")
        print(f"   End frame error:   mean {np.mean(end_diffs):+.1f}, median {np.median(end_diffs):+.0f}")
        print(f"   Start within +/-2 frames: {sum(1 for d in start_diffs if abs(d) <= 2)}/{len(matched)} ({100*sum(1 for d in start_diffs if abs(d) <= 2)/len(matched):.0f}%)")
        print(f"   End within +/-2 frames:   {sum(1 for d in end_diffs if abs(d) <= 2)}/{len(matched)} ({100*sum(1 for d in end_diffs if abs(d) <= 2)/len(matched):.0f}%)")

    # Duration stats
    print(f"\n## DURATION STATS")
    print(f"   Human GT:  median {np.median(gt_durations):.0f} frames, mean {np.mean(gt_durations):.1f}, range {min(gt_durations)}-{max(gt_durations)}")
    print(f"   Algorithm: median {np.median(det_durations):.0f} frames, mean {np.mean(det_durations):.1f}, range {min(det_durations)}-{max(det_durations)}")

    # =========================================================================
    # FALSE NEGATIVES - Human marked, algorithm missed
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"FALSE NEGATIVES: {len(fns)} reaches human marked but algorithm missed")
    print("=" * 80)

    if fns:
        # Categorize FNs
        fn_categories = defaultdict(list)

        for fn in fns:
            # Check for overlapping detections
            overlaps = []
            for det in det_list:
                iou = compute_iou(fn['start'], fn['end'], det['start'], det['end'])
                if iou > 0:
                    overlaps.append({'det': det, 'iou': iou})

            tracking = analyze_tracking_at_reach(dlc_df, fn['start'], fn['end'])

            fn['overlaps'] = overlaps
            fn['tracking'] = tracking

            # Categorize
            if fn['duration'] > 100:
                fn_categories['UNUSUALLY LONG (>100 frames) - likely annotation artifact'].append(fn)
            elif fn['duration'] <= 2:
                fn_categories['VERY SHORT (<=2 frames) - may be noise'].append(fn)
            elif tracking['mean_nose_distance'] > 25:
                fn_categories['NOSE TOO FAR (>25px) - outside engagement zone'].append(fn)
            elif tracking['hand_visible_pct'] < 50:
                fn_categories['LOW HAND VISIBILITY (<50%) - tracking issue?'].append(fn)
            elif overlaps and max(o['iou'] for o in overlaps) >= 0.15:
                fn_categories['BOUNDARY MISMATCH (IoU 0.15-0.3) - timing disagreement'].append(fn)
            elif overlaps:
                fn_categories['OVER-MERGED (IoU <0.15) - algo merged multiple reaches'].append(fn)
            else:
                fn_categories['COMPLETELY MISSED - no overlap with any detection'].append(fn)

        for category, items in sorted(fn_categories.items()):
            print(f"\n### {category}: {len(items)} cases")
            for fn in items[:5]:  # Show first 5 of each category
                dur = fn['duration']
                source = fn['source']
                if fn['human_corrected']:
                    source += '+corrected'
                track = fn['tracking']

                overlap_str = ""
                if fn['overlaps']:
                    best = max(fn['overlaps'], key=lambda x: x['iou'])
                    overlap_str = f" | best overlap: det {best['det']['start']}-{best['det']['end']} IoU={best['iou']:.2f}"

                print(f"    {fn['start']:>5}-{fn['end']:<5} dur={dur:>4} seg={fn['segment']:>2} | {source:<20} | hand={track['mean_hand_likelihood']:.2f} nose_d={track['mean_nose_distance']:.0f}px{overlap_str}")
            if len(items) > 5:
                print(f"    ... and {len(items)-5} more")

    # =========================================================================
    # FALSE POSITIVES - Algorithm found, human didn't mark
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"FALSE POSITIVES: {len(fps)} reaches algorithm found but human didn't mark")
    print("=" * 80)

    if fps:
        # Categorize FPs
        fp_categories = defaultdict(list)

        for fp in fps:
            tracking = analyze_tracking_at_reach(dlc_df, fp['start'], fp['end'])
            fp['tracking'] = tracking

            # Check if it overlaps with any GT (might be duplicate detection)
            overlaps = []
            for gt in gt_list:
                iou = compute_iou(fp['start'], fp['end'], gt['start'], gt['end'])
                if iou > 0:
                    overlaps.append({'gt': gt, 'iou': iou})
            fp['overlaps'] = overlaps

            # Categorize
            if fp['duration'] <= 2:
                fp_categories['VERY SHORT (<=2 frames) - may be noise'].append(fp)
            elif tracking['mean_hand_likelihood'] < 0.5:
                fp_categories['LOW CONFIDENCE (<0.5) - weak tracking'].append(fp)
            elif tracking['mean_nose_distance'] > 20:
                fp_categories['NOSE FAR FROM SLIT (>20px) - edge of engagement'].append(fp)
            elif overlaps:
                fp_categories['PARTIAL OVERLAP WITH GT - possible split/boundary issue'].append(fp)
            else:
                fp_categories['HUMAN MAY HAVE MISSED - strong tracking, no GT overlap'].append(fp)

        for category, items in sorted(fp_categories.items()):
            print(f"\n### {category}: {len(items)} cases")
            for fp in items[:8]:  # Show first 8 of each category
                dur = fp['duration']
                track = fp['tracking']

                overlap_str = ""
                if fp['overlaps']:
                    best = max(fp['overlaps'], key=lambda x: x['iou'])
                    overlap_str = f" | overlaps GT {best['gt']['start']}-{best['gt']['end']} IoU={best['iou']:.2f}"

                print(f"    {fp['start']:>5}-{fp['end']:<5} dur={dur:>4} seg={fp['segment']:>2} | hand={track['mean_hand_likelihood']:.2f} nose_d={track['mean_nose_distance']:.0f}px vis={track['hand_visible_pct']:.0f}%{overlap_str}")
            if len(items) > 8:
                print(f"    ... and {len(items)-8} more")

    # =========================================================================
    # FLAGGED FOR REVIEW - statistically unusual cases
    # =========================================================================
    print("\n" + "=" * 80)
    print("FLAGGED FOR MANUAL REVIEW")
    print("=" * 80)

    flagged = []

    # Flag FPs with strong tracking that human may have missed
    strong_fps = [fp for fp in fps
                  if fp['tracking']['mean_hand_likelihood'] >= 0.7
                  and fp['tracking']['mean_nose_distance'] <= 20
                  and fp['duration'] >= 5
                  and not fp['overlaps']]
    if strong_fps:
        flagged.append(('LIKELY MISSED BY HUMAN', strong_fps))

    # Flag unusually long GT reaches
    long_gt = [fn for fn in fns if fn['duration'] > 50]
    if long_gt:
        flagged.append(('UNUSUALLY LONG GT REACHES (>50 frames)', long_gt))

    # Flag cases where algorithm and human boundaries differ significantly
    big_boundary_diff = [m for m in matched if abs(m['start_diff']) > 5 or abs(m['end_diff']) > 5]
    if big_boundary_diff:
        flagged.append(('LARGE BOUNDARY DIFFERENCES (>5 frames)', [{'gt': m['gt'], 'det': m['det'], 'start_diff': m['start_diff'], 'end_diff': m['end_diff']} for m in big_boundary_diff]))

    for flag_reason, items in flagged:
        print(f"\n### {flag_reason}: {len(items)} cases")
        for item in items[:10]:
            if 'det' in item and 'gt' in item:
                # Boundary difference case
                print(f"    GT {item['gt']['start']}-{item['gt']['end']} vs Det {item['det']['start']}-{item['det']['end']} | start_diff={item['start_diff']:+d} end_diff={item['end_diff']:+d}")
            elif 'tracking' in item:
                # FP or FN case
                track = item['tracking']
                print(f"    {item['start']:>5}-{item['end']:<5} dur={item['duration']:>4} | hand={track['mean_hand_likelihood']:.2f} nose_d={track['mean_nose_distance']:.0f}px")
            else:
                print(f"    {item}")
        if len(items) > 10:
            print(f"    ... and {len(items)-10} more")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
