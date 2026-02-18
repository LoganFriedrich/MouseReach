#!/usr/bin/env python
"""
Analyze characteristics of false positive vs true positive reaches.

Compares algorithmic reaches that match ground truth (TP) vs those that don't (FP)
across all GT videos to understand what distinguishes false detections.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# Constants
REACH_TOLERANCE = 10  # frames - same as collect_results.py


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def matches_gt(algo_reach, gt_reaches, tolerance=REACH_TOLERANCE):
    """
    Check if algo reach matches any GT reach within tolerance.

    Returns True if match found, False otherwise.
    """
    algo_start = algo_reach['start_frame']
    algo_end = algo_reach['end_frame']

    for gt in gt_reaches:
        gt_start = gt['start_frame']
        gt_end = gt['end_frame']

        start_match = abs(algo_start - gt_start) <= tolerance
        end_match = abs(algo_end - gt_end) <= tolerance

        if start_match and end_match:
            return True

    return False


def extract_features(reach):
    """Extract features from an algo reach."""
    return {
        'duration_frames': reach.get('duration_frames'),
        'max_extent_pixels': reach.get('max_extent_pixels'),
        'max_extent_ruler': reach.get('max_extent_ruler'),
        'confidence': reach.get('confidence'),
        'start_confidence': reach.get('start_confidence'),
        'end_confidence': reach.get('end_confidence'),
    }


def compute_stats(values):
    """Compute distribution statistics for a list of values."""
    values = [v for v in values if v is not None]
    if not values:
        return {
            'n': 0,
            'mean': None,
            'median': None,
            'p5': None,
            'p25': None,
            'p75': None,
            'p95': None,
        }

    return {
        'n': len(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'p5': np.percentile(values, 5),
        'p25': np.percentile(values, 25),
        'p75': np.percentile(values, 75),
        'p95': np.percentile(values, 95),
    }


def print_stats_comparison(feature_name, tp_stats, fp_stats):
    """Print side-by-side comparison of TP vs FP stats."""
    print(f"\n{'='*80}", flush=True)
    print(f"{feature_name.upper()}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Metric':<15} {'TP':>15} {'FP':>15} {'Difference':>15}", flush=True)
    print(f"{'-'*80}", flush=True)

    print(f"{'N':<15} {tp_stats['n']:>15} {fp_stats['n']:>15} {'':<15}", flush=True)

    for metric in ['mean', 'median', 'p5', 'p25', 'p75', 'p95']:
        tp_val = tp_stats[metric]
        fp_val = fp_stats[metric]

        if tp_val is not None and fp_val is not None:
            diff = fp_val - tp_val
            print(f"{metric.upper():<15} {tp_val:>15.2f} {fp_val:>15.2f} {diff:>+15.2f}", flush=True)
        else:
            tp_str = f"{tp_val:.2f}" if tp_val is not None else "N/A"
            fp_str = f"{fp_val:.2f}" if fp_val is not None else "N/A"
            print(f"{metric.upper():<15} {tp_str:>15} {fp_str:>15} {'N/A':<15}", flush=True)


def create_histogram(values, n_bins=20, max_width=60):
    """Create text-based histogram."""
    values = [v for v in values if v is not None]
    if not values:
        return "No data"

    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return f"All values = {min_val:.2f}"

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    hist, _ = np.histogram(values, bins=bin_edges)

    max_count = max(hist)
    if max_count == 0:
        return "No data"

    lines = []
    for i, count in enumerate(hist):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        bar_width = int((count / max_count) * max_width)
        bar = '#' * bar_width
        lines.append(f"{bin_start:>8.1f} - {bin_end:>8.1f} | {bar} {count}")

    return '\n'.join(lines)


def main():
    print("="*80, flush=True)
    print("FALSE POSITIVE vs TRUE POSITIVE REACH ANALYSIS", flush=True)
    print("="*80, flush=True)

    # Get processing directory
    from mousereach.config import require_processing_root
    processing_dir = require_processing_root() / "Processing"
    print(f"\nProcessing directory: {processing_dir}", flush=True)

    # Find all GT files
    gt_files = sorted(processing_dir.glob("*_unified_ground_truth.json"))
    print(f"\nFound {len(gt_files)} ground truth files", flush=True)

    if not gt_files:
        print("ERROR: No ground truth files found!", flush=True)
        return

    # Collect features for TP and FP
    tp_features = defaultdict(list)
    fp_features = defaultdict(list)
    fp_per_video = defaultdict(int)
    tp_per_video = defaultdict(int)

    total_algo_reaches = 0
    total_gt_reaches = 0
    videos_processed = 0

    print("\nProcessing videos:", flush=True)
    print("-"*80, flush=True)

    for gt_file in gt_files:
        video_name = gt_file.stem.replace('_unified_ground_truth', '')

        # Load GT
        gt_data = load_json(gt_file)
        gt_reaches = gt_data.get('reaches', {}).get('reaches', [])
        total_gt_reaches += len(gt_reaches)

        # Load algo reaches
        algo_file = gt_file.parent / f"{video_name}_reaches.json"
        if not algo_file.exists():
            print(f"  {video_name}: SKIP (no algo file)", flush=True)
            continue

        algo_data = load_json(algo_file)

        # Extract all algo reaches from all segments
        algo_reaches = []
        for segment in algo_data.get('segments', []):
            algo_reaches.extend(segment.get('reaches', []))

        total_algo_reaches += len(algo_reaches)

        # Classify each algo reach as TP or FP
        n_tp = 0
        n_fp = 0

        for reach in algo_reaches:
            is_tp = matches_gt(reach, gt_reaches)
            features = extract_features(reach)

            if is_tp:
                n_tp += 1
                for key, value in features.items():
                    tp_features[key].append(value)
            else:
                n_fp += 1
                for key, value in features.items():
                    fp_features[key].append(value)

        fp_per_video[video_name] = n_fp
        tp_per_video[video_name] = n_tp
        videos_processed += 1

        print(f"  {video_name}: {len(algo_reaches)} algo ({n_tp} TP, {n_fp} FP) | {len(gt_reaches)} GT", flush=True)

    print("\n" + "="*80, flush=True)
    print("SUMMARY", flush=True)
    print("="*80, flush=True)
    print(f"Videos processed: {videos_processed}", flush=True)
    print(f"Total GT reaches: {total_gt_reaches}", flush=True)
    print(f"Total algo reaches: {total_algo_reaches}", flush=True)
    print(f"  True positives: {sum(tp_per_video.values())}", flush=True)
    print(f"  False positives: {sum(fp_per_video.values())}", flush=True)

    # Print per-video FP breakdown
    print("\n" + "="*80, flush=True)
    print("FALSE POSITIVES PER VIDEO (sorted by count)", flush=True)
    print("="*80, flush=True)
    print(f"{'Video':<50} {'FP':>10} {'TP':>10} {'FP Rate':>10}", flush=True)
    print("-"*80, flush=True)

    for video_name in sorted(fp_per_video.keys(), key=lambda v: fp_per_video[v], reverse=True):
        n_fp = fp_per_video[video_name]
        n_tp = tp_per_video[video_name]
        total = n_fp + n_tp
        fp_rate = (n_fp / total * 100) if total > 0 else 0
        print(f"{video_name:<50} {n_fp:>10} {n_tp:>10} {fp_rate:>9.1f}%", flush=True)

    # Compare distributions
    print("\n" + "="*80, flush=True)
    print("FEATURE DISTRIBUTIONS: TP vs FP", flush=True)
    print("="*80, flush=True)

    for feature_name in ['duration_frames', 'max_extent_pixels', 'max_extent_ruler',
                         'confidence', 'start_confidence', 'end_confidence']:
        tp_stats = compute_stats(tp_features[feature_name])
        fp_stats = compute_stats(fp_features[feature_name])
        print_stats_comparison(feature_name, tp_stats, fp_stats)

    # Duration histograms
    print("\n" + "="*80, flush=True)
    print("DURATION DISTRIBUTION (frames)", flush=True)
    print("="*80, flush=True)

    print("\nTRUE POSITIVES:", flush=True)
    print(create_histogram(tp_features['duration_frames']), flush=True)

    print("\nFALSE POSITIVES:", flush=True)
    print(create_histogram(fp_features['duration_frames']), flush=True)

    # Max extent histograms
    print("\n" + "="*80, flush=True)
    print("MAX EXTENT DISTRIBUTION (pixels)", flush=True)
    print("="*80, flush=True)

    print("\nTRUE POSITIVES:", flush=True)
    print(create_histogram(tp_features['max_extent_pixels']), flush=True)

    print("\nFALSE POSITIVES:", flush=True)
    print(create_histogram(fp_features['max_extent_pixels']), flush=True)

    print("\n" + "="*80, flush=True)
    print("ANALYSIS COMPLETE", flush=True)
    print("="*80, flush=True)


if __name__ == '__main__':
    main()
