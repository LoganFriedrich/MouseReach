#!/usr/bin/env python3
"""
Evaluate algorithm against complete ground truth.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def compute_overlap(start1: int, end1: int, start2: int, end2: int) -> float:
    """Compute IoU."""
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    if intersection_end < intersection_start:
        return 0.0

    intersection = intersection_end - intersection_start + 1
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection

    return intersection / union if union > 0 else 0.0


def load_all_gt(gt_path: Path) -> List[Dict]:
    """Load ALL ground truth reaches."""
    with open(gt_path) as f:
        data = json.load(f)

    reaches = []
    reach_id = 0
    for segment in data.get('segments', []):
        for reach in segment.get('reaches', []):
            reach_id += 1
            reaches.append({
                'reach_id': reach_id,
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


def match_reaches(
    gt_reaches: List[Dict],
    det_reaches: List[Dict],
    overlap_threshold: float = 0.3
) -> Tuple[List, List, List]:
    """Match detected reaches to GT using IoU."""
    matched = []
    used_gt = set()
    used_det = set()

    gt_sorted = sorted(gt_reaches, key=lambda r: r['start_frame'])
    det_sorted = sorted(det_reaches, key=lambda r: r['start_frame'])

    for gt in gt_sorted:
        gt_id = gt['reach_id']
        if gt_id in used_gt:
            continue

        best_match = None
        best_iou = 0

        for det in det_sorted:
            det_id = det['reach_id']
            if det_id in used_det:
                continue

            iou = compute_overlap(
                gt['start_frame'], gt['end_frame'],
                det['start_frame'], det['end_frame']
            )

            if iou >= overlap_threshold and iou > best_iou:
                best_iou = iou
                best_match = det

        if best_match:
            used_gt.add(gt_id)
            used_det.add(best_match['reach_id'])
            matched.append({
                'gt': gt,
                'det': best_match,
                'iou': best_iou,
                'start_diff': best_match['start_frame'] - gt['start_frame'],
                'end_diff': best_match['end_frame'] - gt['end_frame']
            })

    false_positives = [d for d in det_reaches if d['reach_id'] not in used_det]
    false_negatives = [g for g in gt_reaches if g['reach_id'] not in used_gt]

    return matched, false_positives, false_negatives


def evaluate(gt_path: Path, det_path: Path, overlap_threshold: float = 0.3):
    """Run evaluation and print report."""
    print("=" * 60)
    print("EVALUATION VS COMPLETE GROUND TRUTH")
    print("=" * 60)

    gt_reaches = load_all_gt(gt_path)
    det_reaches = load_detected(det_path)

    # Count human-touched
    human_added = sum(1 for r in gt_reaches if r['source'] == 'human_added')
    human_corrected = sum(1 for r in gt_reaches if r['human_corrected'])
    algorithm_only = sum(1 for r in gt_reaches if r['source'] == 'algorithm' and not r['human_corrected'])

    print(f"\nGT reaches: {len(gt_reaches)}")
    print(f"  - human_added: {human_added}")
    print(f"  - human_corrected: {human_corrected}")
    print(f"  - algorithm (unchanged): {algorithm_only}")
    print(f"Algorithm detections: {len(det_reaches)}")
    print(f"IoU threshold: {overlap_threshold}")

    matched, fps, fns = match_reaches(gt_reaches, det_reaches, overlap_threshold)

    precision = len(matched) / len(det_reaches) if det_reaches else 0
    recall = len(matched) / len(gt_reaches) if gt_reaches else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n## Matching Results")
    print(f"  Matched: {len(matched)} / {len(gt_reaches)} GT reaches")
    print(f"  False Positives: {len(fps)}")
    print(f"  False Negatives: {len(fns)}")

    print(f"\n## Metrics")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall: {recall:.1%}")
    print(f"  F1 Score: {f1:.1%}")

    if matched:
        start_errors = [m['start_diff'] for m in matched]
        end_errors = [m['end_diff'] for m in matched]
        ious = [m['iou'] for m in matched]

        print(f"\n## Frame Accuracy (matched)")
        print(f"  Start error: mean={np.mean(start_errors):.1f}, median={np.median(start_errors):.1f}")
        print(f"  End error: mean={np.mean(end_errors):.1f}, median={np.median(end_errors):.1f}")
        print(f"  Start within +/-2: {sum(1 for e in start_errors if abs(e) <= 2)}/{len(matched)} ({100*sum(1 for e in start_errors if abs(e) <= 2)/len(matched):.0f}%)")
        print(f"  End within +/-2: {sum(1 for e in end_errors if abs(e) <= 2)}/{len(matched)} ({100*sum(1 for e in end_errors if abs(e) <= 2)/len(matched):.0f}%)")
        print(f"  Mean IoU: {np.mean(ious):.2f}")

    # Analyze false negatives by source
    if fns:
        fn_human_added = [fn for fn in fns if fn['source'] == 'human_added']
        fn_corrected = [fn for fn in fns if fn['human_corrected']]
        fn_algo_only = [fn for fn in fns if fn['source'] == 'algorithm' and not fn['human_corrected']]

        print(f"\n## False Negatives by source")
        print(f"  human_added: {len(fn_human_added)}")
        print(f"  human_corrected: {len(fn_corrected)}")
        print(f"  algorithm (unchanged): {len(fn_algo_only)}")

        print(f"\n## False Negatives (first 15)")
        for fn in fns[:15]:
            dur = fn['end_frame'] - fn['start_frame'] + 1
            # Check for overlapping detections
            overlaps = []
            for det in det_reaches:
                iou = compute_overlap(fn['start_frame'], fn['end_frame'],
                                       det['start_frame'], det['end_frame'])
                if iou > 0:
                    overlaps.append((det['reach_id'], round(iou, 3)))
            overlap_str = f" overlaps:{overlaps[:2]}" if overlaps else " NO OVERLAP"
            source = fn['source']
            if fn['human_corrected']:
                source += "+corrected"
            print(f"  {fn['start_frame']}-{fn['end_frame']} (dur={dur}, {source}){overlap_str}")

    # Duration distribution comparison
    gt_durs = [r['end_frame'] - r['start_frame'] + 1 for r in gt_reaches]
    det_durs = [r['end_frame'] - r['start_frame'] + 1 for r in det_reaches]

    print(f"\n## Duration Distribution")
    print(f"  GT: median={np.median(gt_durs):.0f}, mean={np.mean(gt_durs):.1f}")
    print(f"  Det: median={np.median(det_durs):.0f}, mean={np.mean(det_durs):.1f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matched': len(matched),
        'fps': len(fps),
        'fns': len(fns)
    }


def main():
    from mousereach.config import Paths
    processing_dir = Paths.PROCESSING
    video_id = "20251021_CNT0405_P4"

    gt_path = processing_dir / f"{video_id}_reach_ground_truth.json"
    det_path = processing_dir / f"{video_id}_reaches.json"

    evaluate(gt_path, det_path)


if __name__ == "__main__":
    main()
