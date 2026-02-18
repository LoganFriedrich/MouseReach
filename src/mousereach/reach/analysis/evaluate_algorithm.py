#!/usr/bin/env python3
"""
Evaluate reach detection algorithm against ground truth.

Compares algorithm detections to human-labeled reaches and reports:
- Precision: What % of algorithm detections are correct?
- Recall: What % of ground truth reaches were detected?
- Frame-level accuracy: How close are start/end frames?
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result of matching a detected reach to ground truth."""
    gt_id: int
    detected_id: int
    gt_start: int
    gt_end: int
    det_start: int
    det_end: int
    start_error: int  # detected - gt
    end_error: int    # detected - gt
    overlap_ratio: float  # IoU


def load_ground_truth(gt_path: Path) -> List[Dict]:
    """Load ground truth reaches."""
    with open(gt_path, 'r') as f:
        data = json.load(f)

    reaches = []
    for segment in data.get('segments', []):
        for reach in segment.get('reaches', []):
            reaches.append({
                'reach_id': reach.get('reach_id', len(reaches) + 1),
                'start_frame': reach['start_frame'],
                'end_frame': reach['end_frame'],
                'segment_num': segment.get('segment_num', 0)
            })

    return reaches


def load_detected(det_path: Path) -> List[Dict]:
    """Load algorithm-detected reaches."""
    with open(det_path, 'r') as f:
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


def compute_overlap(start1: int, end1: int, start2: int, end2: int) -> float:
    """Compute IoU (Intersection over Union) of two frame ranges."""
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    if intersection_end < intersection_start:
        return 0.0

    intersection = intersection_end - intersection_start + 1
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection

    return intersection / union if union > 0 else 0.0


def match_reaches(
    gt_reaches: List[Dict],
    det_reaches: List[Dict],
    overlap_threshold: float = 0.3
) -> Tuple[List[MatchResult], List[Dict], List[Dict]]:
    """
    Match detected reaches to ground truth using IoU.

    Returns:
        - matched: List of MatchResult for successful matches
        - false_positives: Detected reaches with no GT match
        - false_negatives: GT reaches with no detection match
    """
    matched = []
    used_gt = set()
    used_det = set()

    # Sort both by start frame
    gt_sorted = sorted(gt_reaches, key=lambda r: r['start_frame'])
    det_sorted = sorted(det_reaches, key=lambda r: r['start_frame'])

    # For each GT reach, find best matching detection
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

            matched.append(MatchResult(
                gt_id=gt_id,
                detected_id=best_match['reach_id'],
                gt_start=gt['start_frame'],
                gt_end=gt['end_frame'],
                det_start=best_match['start_frame'],
                det_end=best_match['end_frame'],
                start_error=best_match['start_frame'] - gt['start_frame'],
                end_error=best_match['end_frame'] - gt['end_frame'],
                overlap_ratio=best_iou
            ))

    # False positives: detections without GT match
    false_positives = [d for d in det_reaches if d['reach_id'] not in used_det]

    # False negatives: GT without detection match
    false_negatives = [g for g in gt_reaches if g['reach_id'] not in used_gt]

    return matched, false_positives, false_negatives


def analyze_duration_distribution(gt_reaches: List[Dict], det_reaches: List[Dict]):
    """Compare duration distributions between GT and detected."""
    print("\n" + "=" * 60)
    print("DURATION ANALYSIS")
    print("=" * 60)

    gt_durations = [r['end_frame'] - r['start_frame'] + 1 for r in gt_reaches]
    det_durations = [r['end_frame'] - r['start_frame'] + 1 for r in det_reaches]

    print(f"\n## Ground Truth Duration")
    print(f"  Mean: {np.mean(gt_durations):.1f} frames")
    print(f"  Median: {np.median(gt_durations):.1f} frames")
    print(f"  Min: {np.min(gt_durations)}, Max: {np.max(gt_durations)}")
    print(f"  <= 5 frames: {sum(1 for d in gt_durations if d <= 5)} ({100*sum(1 for d in gt_durations if d <= 5)/len(gt_durations):.1f}%)")
    print(f"  <= 10 frames: {sum(1 for d in gt_durations if d <= 10)} ({100*sum(1 for d in gt_durations if d <= 10)/len(gt_durations):.1f}%)")
    print(f"  > 50 frames: {sum(1 for d in gt_durations if d > 50)} ({100*sum(1 for d in gt_durations if d > 50)/len(gt_durations):.1f}%)")

    print(f"\n## Detected Duration")
    print(f"  Mean: {np.mean(det_durations):.1f} frames")
    print(f"  Median: {np.median(det_durations):.1f} frames")
    print(f"  Min: {np.min(det_durations)}, Max: {np.max(det_durations)}")
    print(f"  <= 5 frames: {sum(1 for d in det_durations if d <= 5)} ({100*sum(1 for d in det_durations if d <= 5)/len(det_durations):.1f}%)")
    print(f"  <= 10 frames: {sum(1 for d in det_durations if d <= 10)} ({100*sum(1 for d in det_durations if d <= 10)/len(det_durations):.1f}%)")
    print(f"  > 50 frames: {sum(1 for d in det_durations if d > 50)} ({100*sum(1 for d in det_durations if d > 50)/len(det_durations):.1f}%)")


def analyze_false_positives(fps: List[Dict], gt_reaches: List[Dict]):
    """Analyze patterns in false positives."""
    print("\n" + "=" * 60)
    print("FALSE POSITIVE ANALYSIS")
    print("=" * 60)

    if not fps:
        print("No false positives!")
        return

    fp_durations = [r['end_frame'] - r['start_frame'] + 1 for r in fps]

    print(f"\n## FP Duration Distribution")
    print(f"  Mean: {np.mean(fp_durations):.1f} frames")
    print(f"  Median: {np.median(fp_durations):.1f} frames")
    print(f"  <= 3 frames: {sum(1 for d in fp_durations if d <= 3)} ({100*sum(1 for d in fp_durations if d <= 3)/len(fps):.1f}%)")
    print(f"  <= 5 frames: {sum(1 for d in fp_durations if d <= 5)} ({100*sum(1 for d in fp_durations if d <= 5)/len(fps):.1f}%)")
    print(f"  <= 10 frames: {sum(1 for d in fp_durations if d <= 10)} ({100*sum(1 for d in fp_durations if d <= 10)/len(fps):.1f}%)")

    # Check if FPs fall within GT reaches (fragmentation)
    fps_in_gt = 0
    for fp in fps:
        for gt in gt_reaches:
            if fp['start_frame'] >= gt['start_frame'] and fp['end_frame'] <= gt['end_frame']:
                fps_in_gt += 1
                break

    print(f"\n## FP Location")
    print(f"  FPs that fall within a GT reach (fragmentation): {fps_in_gt} ({100*fps_in_gt/len(fps):.1f}%)")
    print(f"  FPs outside any GT reach (spurious): {len(fps) - fps_in_gt} ({100*(len(fps)-fps_in_gt)/len(fps):.1f}%)")


def analyze_false_negatives(fns: List[Dict], det_reaches: List[Dict]):
    """Analyze patterns in false negatives."""
    print("\n" + "=" * 60)
    print("FALSE NEGATIVE ANALYSIS")
    print("=" * 60)

    if not fns:
        print("No false negatives!")
        return

    fn_durations = [r['end_frame'] - r['start_frame'] + 1 for r in fns]

    print(f"\n## FN Duration Distribution")
    print(f"  Mean: {np.mean(fn_durations):.1f} frames")
    print(f"  Median: {np.median(fn_durations):.1f} frames")
    print(f"  Min: {np.min(fn_durations)}, Max: {np.max(fn_durations)}")
    print(f"  > 50 frames: {sum(1 for d in fn_durations if d > 50)} ({100*sum(1 for d in fn_durations if d > 50)/len(fns):.1f}%)")
    print(f"  > 100 frames: {sum(1 for d in fn_durations if d > 100)} ({100*sum(1 for d in fn_durations if d > 100)/len(fns):.1f}%)")

    # Check how many detections overlap each FN
    print(f"\n## Detections overlapping missed GT reaches:")
    for fn in fns[:5]:  # First 5
        overlapping = []
        for det in det_reaches:
            iou = compute_overlap(fn['start_frame'], fn['end_frame'],
                                   det['start_frame'], det['end_frame'])
            if iou > 0:
                overlapping.append((det['reach_id'], det['start_frame'], det['end_frame'], iou))

        print(f"  GT #{fn['reach_id']} (frames {fn['start_frame']}-{fn['end_frame']}, dur={fn['end_frame']-fn['start_frame']+1}):")
        if overlapping:
            for det_id, s, e, iou in overlapping[:3]:
                print(f"    -> Det #{det_id}: {s}-{e} (IoU={iou:.2f})")
        else:
            print(f"    -> No overlapping detections")


def evaluate(gt_path: Path, det_path: Path, overlap_threshold: float = 0.3):
    """Run full evaluation and print report."""
    print("=" * 60)
    print("REACH DETECTION EVALUATION")
    print("=" * 60)

    gt_reaches = load_ground_truth(gt_path)
    det_reaches = load_detected(det_path)

    print(f"\nGround truth: {len(gt_reaches)} reaches")
    print(f"Detected: {len(det_reaches)} reaches")
    print(f"Overlap threshold: {overlap_threshold}")

    # Duration analysis first
    analyze_duration_distribution(gt_reaches, det_reaches)

    matched, fps, fns = match_reaches(gt_reaches, det_reaches, overlap_threshold)

    # Compute metrics
    precision = len(matched) / len(det_reaches) if det_reaches else 0
    recall = len(matched) / len(gt_reaches) if gt_reaches else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n## Overall Metrics")
    print(f"  Matched: {len(matched)}")
    print(f"  False Positives (extra detections): {len(fps)}")
    print(f"  False Negatives (missed GT): {len(fns)}")
    print(f"\n  Precision: {precision:.1%}")
    print(f"  Recall: {recall:.1%}")
    print(f"  F1 Score: {f1:.1%}")

    if matched:
        start_errors = [m.start_error for m in matched]
        end_errors = [m.end_error for m in matched]
        ious = [m.overlap_ratio for m in matched]

        print(f"\n## Frame Accuracy (matched reaches)")
        print(f"  Start frame error (det - GT):")
        print(f"    Mean: {np.mean(start_errors):.1f} frames")
        print(f"    Median: {np.median(start_errors):.1f} frames")
        print(f"    Std: {np.std(start_errors):.1f} frames")
        print(f"    Within +/-2 frames: {sum(1 for e in start_errors if abs(e) <= 2)}/{len(start_errors)}")
        print(f"    Within +/-5 frames: {sum(1 for e in start_errors if abs(e) <= 5)}/{len(start_errors)}")

        print(f"\n  End frame error (det - GT):")
        print(f"    Mean: {np.mean(end_errors):.1f} frames")
        print(f"    Median: {np.median(end_errors):.1f} frames")
        print(f"    Std: {np.std(end_errors):.1f} frames")
        print(f"    Within +/-2 frames: {sum(1 for e in end_errors if abs(e) <= 2)}/{len(end_errors)}")
        print(f"    Within +/-5 frames: {sum(1 for e in end_errors if abs(e) <= 5)}/{len(end_errors)}")

        print(f"\n  Overlap (IoU):")
        print(f"    Mean: {np.mean(ious):.2f}")
        print(f"    Median: {np.median(ious):.2f}")
        print(f"    Min: {np.min(ious):.2f}")

    if fps:
        print(f"\n## False Positives (first 10)")
        for fp in fps[:10]:
            print(f"  Det #{fp['reach_id']}: frames {fp['start_frame']}-{fp['end_frame']}")

    if fns:
        print(f"\n## False Negatives (first 10)")
        for fn in fns[:10]:
            print(f"  GT #{fn['reach_id']}: frames {fn['start_frame']}-{fn['end_frame']}")

    # Detailed analysis
    analyze_false_positives(fps, gt_reaches)
    analyze_false_negatives(fns, det_reaches)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matched': len(matched),
        'false_positives': len(fps),
        'false_negatives': len(fns),
        'mean_start_error': np.mean(start_errors) if matched else None,
        'mean_end_error': np.mean(end_errors) if matched else None,
    }


def main():
    """Run evaluation on default test video."""
    from mousereach.config import Paths
    processing_dir = Paths.PROCESSING
    video_id = "20251021_CNT0405_P4"

    gt_path = processing_dir / f"{video_id}_reach_ground_truth.json"
    det_path = processing_dir / f"{video_id}_reaches.json"

    if not gt_path.exists():
        print(f"Ground truth not found: {gt_path}")
        return

    if not det_path.exists():
        print(f"Detections not found: {det_path}")
        print("Run reach detection first: mousereach-detect-reaches")
        return

    evaluate(gt_path, det_path)


if __name__ == "__main__":
    main()
