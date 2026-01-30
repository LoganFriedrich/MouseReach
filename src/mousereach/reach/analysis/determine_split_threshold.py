#!/usr/bin/env python3
"""
Determine the statistical threshold for when to consider splitting a reach.
"""

import json
import numpy as np
from pathlib import Path


def main():
    from mousereach.config import Paths
    processing_dir = Paths.PROCESSING
    video_id = "20251021_CNT0405_P4"

    gt_path = processing_dir / f"{video_id}_reach_ground_truth.json"

    with open(gt_path) as f:
        data = json.load(f)

    # Get human-touched reaches only
    durations = []
    for segment in data.get('segments', []):
        for reach in segment.get('reaches', []):
            if reach.get('source') == 'human_added' or reach.get('human_corrected', False):
                dur = reach['end_frame'] - reach['start_frame'] + 1
                durations.append(dur)

    durations = np.array(durations)

    print("Human-touched GT reach duration statistics:")
    print(f"  Count: {len(durations)}")
    print(f"  Mean: {np.mean(durations):.1f} frames")
    print(f"  Median: {np.median(durations):.1f} frames")
    print(f"  Std: {np.std(durations):.1f} frames")
    print(f"  Min: {np.min(durations)}, Max: {np.max(durations)}")

    print(f"\nPercentiles:")
    for p in [50, 75, 90, 95, 97.5, 99]:
        print(f"  {p}th percentile: {np.percentile(durations, p):.0f} frames")

    print(f"\nDuration distribution:")
    print(f"  <= 10 frames: {np.sum(durations <= 10)} ({100*np.mean(durations <= 10):.1f}%)")
    print(f"  11-20 frames: {np.sum((durations > 10) & (durations <= 20))} ({100*np.mean((durations > 10) & (durations <= 20)):.1f}%)")
    print(f"  21-30 frames: {np.sum((durations > 20) & (durations <= 30))} ({100*np.mean((durations > 20) & (durations <= 30)):.1f}%)")
    print(f"  31-50 frames: {np.sum((durations > 30) & (durations <= 50))} ({100*np.mean((durations > 30) & (durations <= 50)):.1f}%)")
    print(f"  > 50 frames: {np.sum(durations > 50)} ({100*np.mean(durations > 50):.1f}%)")


if __name__ == "__main__":
    main()
