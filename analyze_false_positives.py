"""
Detailed analysis of false positive reaches in MouseReach algorithm.
Compares ground truth (v2) with algorithm output to characterize FP patterns.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

# Target videos with highest FP rates
WORST_VIDEOS = ["CNT0307_P4", "CNT0311_P2", "CNT0110_P2", "CNT0309_P1", "CNT0413_P2", "CNT0312_P2"]

GT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_0_0")

APEX_TOLERANCE = 5  # frames


def load_gt_reaches(video_name):
    """Load fully determined GT reaches from v2 unified GT file."""
    gt_file = GT_DIR / f"{video_name}_unified_ground_truth.json"
    if not gt_file.exists():
        return []

    with open(gt_file, 'r') as f:
        data = json.load(f)

    gt_reaches = []
    for reach in data['reaches']['reaches']:
        # Only include fully determined reaches
        if reach.get('start_determined') and reach.get('end_determined'):
            gt_reaches.append({
                'start_frame': reach['start_frame'],
                'end_frame': reach['end_frame'],
                'apex_frame': reach.get('apex_frame'),
                'max_extent': reach.get('max_extent_pixels')
            })

    return gt_reaches


def load_algo_reaches(video_name):
    """Load algorithm reaches from Pipeline_0_0."""
    algo_file = ALGO_DIR / f"{video_name}_reaches.json"
    if not algo_file.exists():
        return []

    with open(algo_file, 'r') as f:
        data = json.load(f)

    algo_reaches = []
    for segment in data['segments']:
        for reach in segment['reaches']:
            algo_reaches.append({
                'start_frame': reach['start_frame'],
                'end_frame': reach['end_frame'],
                'apex_frame': reach['apex_frame'],
                'max_extent': reach['max_extent_pixels'],
                'segment_start': segment['start_frame'],
                'segment_end': segment['end_frame']
            })

    return algo_reaches


def match_reaches(gt_reaches, algo_reaches):
    """Match algo reaches to GT by apex frame. Return TP and FP lists."""
    tp_algo = []
    fp_algo = []

    for algo in algo_reaches:
        matched = False
        for gt in gt_reaches:
            # Match by apex frame with tolerance
            if gt['apex_frame'] is not None and algo['apex_frame'] is not None:
                if abs(gt['apex_frame'] - algo['apex_frame']) <= APEX_TOLERANCE:
                    matched = True
                    break

        if matched:
            tp_algo.append(algo)
        else:
            fp_algo.append(algo)

    return tp_algo, fp_algo


def analyze_segment_position(fp_reaches):
    """Analyze where in segment FPs occur (early, middle, late)."""
    positions = []
    for reach in fp_reaches:
        seg_start = reach['segment_start']
        seg_end = reach['segment_end']
        apex = reach['apex_frame']

        # Normalize position: 0 = segment start, 1 = segment end
        seg_duration = seg_end - seg_start
        if seg_duration > 0:
            position = (apex - seg_start) / seg_duration
            positions.append(position)

    return positions


def find_fp_clusters(fp_reaches):
    """Find overlapping FPs (clusters vs isolated)."""
    # Sort by start frame
    sorted_fps = sorted(fp_reaches, key=lambda r: r['start_frame'])

    clusters = []
    isolated = []

    i = 0
    while i < len(sorted_fps):
        cluster = [sorted_fps[i]]
        j = i + 1

        # Find all FPs that overlap with current cluster
        while j < len(sorted_fps):
            # Check if FP j overlaps with any reach in cluster
            overlaps = False
            for reach in cluster:
                if not (sorted_fps[j]['start_frame'] > reach['end_frame'] or
                       sorted_fps[j]['end_frame'] < reach['start_frame']):
                    overlaps = True
                    break

            if overlaps:
                cluster.append(sorted_fps[j])
                j += 1
            else:
                break

        if len(cluster) > 1:
            clusters.append(cluster)
        else:
            isolated.append(cluster[0])

        i = j if j > i + 1 else i + 1

    return clusters, isolated


def print_distribution_stats(values, name, decimals=1):
    """Print statistics for a distribution."""
    if not values:
        print(f"  {name}: No data")
        return

    arr = np.array(values)
    print(f"  {name}:")
    print(f"    Count: {len(arr)}")
    print(f"    Mean: {np.mean(arr):.{decimals}f}")
    print(f"    Median: {np.median(arr):.{decimals}f}")
    print(f"    Std: {np.std(arr):.{decimals}f}")
    print(f"    Min: {np.min(arr):.{decimals}f}")
    print(f"    Max: {np.max(arr):.{decimals}f}")
    print(f"    25th percentile: {np.percentile(arr, 25):.{decimals}f}")
    print(f"    75th percentile: {np.percentile(arr, 75):.{decimals}f}")


def main():
    print("=" * 80)
    print("FALSE POSITIVE REACH ANALYSIS")
    print("=" * 80)
    print()

    all_tp = []
    all_fp = []

    video_stats = {}

    # Process each video
    for video in WORST_VIDEOS:
        print(f"\nProcessing {video}...")

        gt_reaches = load_gt_reaches(video)
        algo_reaches = load_algo_reaches(video)

        if not algo_reaches:
            print(f"  No algorithm reaches found for {video}")
            continue

        tp, fp = match_reaches(gt_reaches, algo_reaches)

        all_tp.extend(tp)
        all_fp.extend(fp)

        video_stats[video] = {
            'gt_count': len(gt_reaches),
            'algo_count': len(algo_reaches),
            'tp_count': len(tp),
            'fp_count': len(fp)
        }

        print(f"  GT reaches: {len(gt_reaches)}")
        print(f"  Algo reaches: {len(algo_reaches)}")
        print(f"  True positives: {len(tp)}")
        print(f"  False positives: {len(fp)}")
        if len(algo_reaches) > 0:
            print(f"  FP rate: {len(fp) / len(algo_reaches) * 100:.1f}%")

    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"\nTotal GT reaches: {sum(s['gt_count'] for s in video_stats.values())}")
    print(f"Total algo reaches: {sum(s['algo_count'] for s in video_stats.values())}")
    print(f"Total TPs: {len(all_tp)}")
    print(f"Total FPs: {len(all_fp)}")
    if all_tp or all_fp:
        total_algo = len(all_tp) + len(all_fp)
        print(f"Overall FP rate: {len(all_fp) / total_algo * 100:.1f}%")

    if not all_fp:
        print("\nNo false positives found!")
        return

    # Extract FP metrics
    fp_extents = [r['max_extent'] for r in all_fp]
    fp_durations = [r['end_frame'] - r['start_frame'] for r in all_fp]
    fp_positions = analyze_segment_position(all_fp)
    fp_apex_frames = [r['apex_frame'] for r in all_fp]

    # Extract TP metrics for comparison
    tp_extents = [r['max_extent'] for r in all_tp]
    tp_durations = [r['end_frame'] - r['start_frame'] for r in all_tp]
    tp_positions = analyze_segment_position(all_tp)
    tp_apex_frames = [r['apex_frame'] for r in all_tp]

    print("\n" + "=" * 80)
    print("FALSE POSITIVE CHARACTERISTICS")
    print("=" * 80)

    print("\n--- (a) Max Extent Distribution (pixels) ---")
    print_distribution_stats(fp_extents, "FP max extent", decimals=1)

    print("\n--- (b) Duration Distribution (frames) ---")
    print_distribution_stats(fp_durations, "FP duration", decimals=1)

    print("\n--- (c) Segment Position Distribution (0=start, 1=end) ---")
    print_distribution_stats(fp_positions, "FP position in segment", decimals=3)
    if fp_positions:
        early = sum(1 for p in fp_positions if p < 0.33)
        middle = sum(1 for p in fp_positions if 0.33 <= p < 0.67)
        late = sum(1 for p in fp_positions if p >= 0.67)
        print(f"  Early (0-0.33): {early} ({early/len(fp_positions)*100:.1f}%)")
        print(f"  Middle (0.33-0.67): {middle} ({middle/len(fp_positions)*100:.1f}%)")
        print(f"  Late (0.67-1.0): {late} ({late/len(fp_positions)*100:.1f}%)")

    print("\n--- (d) FP Clustering Analysis ---")
    clusters, isolated = find_fp_clusters(all_fp)
    print(f"  Isolated FPs: {len(isolated)}")
    print(f"  FP clusters: {len(clusters)}")
    if clusters:
        cluster_sizes = [len(c) for c in clusters]
        print(f"  Total FPs in clusters: {sum(cluster_sizes)}")
        print(f"  Mean cluster size: {np.mean(cluster_sizes):.1f}")
        print(f"  Max cluster size: {max(cluster_sizes)}")
        print(f"  Cluster size distribution: {dict(zip(*np.unique(cluster_sizes, return_counts=True)))}")

    print("\n--- (e) Apex Frame Distribution ---")
    print_distribution_stats(fp_apex_frames, "FP apex frames", decimals=0)

    print("\n" + "=" * 80)
    print("COMPARISON: TRUE POSITIVES vs FALSE POSITIVES")
    print("=" * 80)

    print("\n--- Max Extent Comparison ---")
    print("TRUE POSITIVES:")
    print_distribution_stats(tp_extents, "TP max extent", decimals=1)
    print("\nFALSE POSITIVES:")
    print_distribution_stats(fp_extents, "FP max extent", decimals=1)
    if tp_extents and fp_extents:
        print(f"\nRatio (FP/TP mean): {np.mean(fp_extents) / np.mean(tp_extents):.2f}")

    print("\n--- Duration Comparison ---")
    print("TRUE POSITIVES:")
    print_distribution_stats(tp_durations, "TP duration", decimals=1)
    print("\nFALSE POSITIVES:")
    print_distribution_stats(fp_durations, "FP duration", decimals=1)
    if tp_durations and fp_durations:
        print(f"\nRatio (FP/TP mean): {np.mean(fp_durations) / np.mean(tp_durations):.2f}")

    print("\n--- Segment Position Comparison ---")
    print("TRUE POSITIVES:")
    print_distribution_stats(tp_positions, "TP position in segment", decimals=3)
    print("\nFALSE POSITIVES:")
    print_distribution_stats(fp_positions, "FP position in segment", decimals=3)

    print("\n" + "=" * 80)
    print("PER-VIDEO BREAKDOWN")
    print("=" * 80)

    for video in WORST_VIDEOS:
        print(f"\n{video}:")
        gt = load_gt_reaches(video)
        algo = load_algo_reaches(video)
        tp, fp = match_reaches(gt, algo)

        if fp:
            fp_ext = [r['max_extent'] for r in fp]
            fp_dur = [r['end_frame'] - r['start_frame'] for r in fp]

            print(f"  FP count: {len(fp)}")
            print(f"  FP extent: mean={np.mean(fp_ext):.1f}, median={np.median(fp_ext):.1f}")
            print(f"  FP duration: mean={np.mean(fp_dur):.1f}, median={np.median(fp_dur):.1f}")

            clusters, isolated = find_fp_clusters(fp)
            print(f"  Isolated: {len(isolated)}, Clustered: {len(clusters)} clusters")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
