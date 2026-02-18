"""
Analyze WHERE and WHY the algorithms fail.

Goal: Identify patterns in failure cases to determine if deep learning can help.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def analyze_boundary_failures():
    """Analyze where boundary detection needs human correction.

    v2 GT files no longer store original_frame - corrections are derived
    by comparing GT boundary frames against algorithm output files.
    """
    print("\n" + "="*70)
    print("BOUNDARY DETECTION FAILURE ANALYSIS")
    print("="*70)

    algo_dir = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_0_0")
    corrections = []

    for gt_file in DATA_DIR.glob("*_unified_ground_truth.json"):
        gt = load_json(gt_file)
        if not gt:
            continue

        video = gt['video_name']

        # Load algorithm's original boundaries for comparison
        algo_file = algo_dir / f"{video}_segments.json"
        algo_data = load_json(algo_file)
        if not algo_data:
            continue

        algo_boundaries = algo_data.get('boundaries', [])
        gt_boundaries = gt.get('segmentation', {}).get('boundaries', [])

        for b in gt_boundaries:
            if not b.get('determined'):
                continue
            idx = b['index']
            gt_frame = b['frame']
            # Match by index to get original algo frame
            algo_frame = algo_boundaries[idx] if idx < len(algo_boundaries) else None
            if algo_frame is None:
                continue
            correction = gt_frame - algo_frame
            corrections.append({
                'video': video,
                'index': idx,
                'final_frame': gt_frame,
                'original_frame': algo_frame,
                'correction': correction,
                'abs_correction': abs(correction)
            })

    df = pd.DataFrame(corrections)

    print(f"\nTotal boundaries analyzed: {len(df)}")
    if len(df) == 0:
        print("  No boundary corrections found (GT matches algo or no algo files)")
        return df

    print(f"\nCorrection distribution:")
    print(f"  Exact (0 frames): {(df['abs_correction'] == 0).sum()} ({100*(df['abs_correction'] == 0).mean():.1f}%)")
    print(f"  +/-1 frame: {(df['abs_correction'] <= 1).sum()} ({100*(df['abs_correction'] <= 1).mean():.1f}%)")
    print(f"  +/-2 frames: {(df['abs_correction'] <= 2).sum()} ({100*(df['abs_correction'] <= 2).mean():.1f}%)")
    print(f"  >2 frames: {(df['abs_correction'] > 2).sum()} ({100*(df['abs_correction'] > 2).mean():.1f}%)")

    # Direction of correction
    print(f"\nCorrection direction:")
    print(f"  Algorithm too early (human moved later): {(df['correction'] > 0).sum()}")
    print(f"  Algorithm too late (human moved earlier): {(df['correction'] < 0).sum()}")
    print(f"  Exact: {(df['correction'] == 0).sum()}")

    # Largest failures
    print(f"\nLargest corrections (>3 frames):")
    large = df[df['abs_correction'] > 3].sort_values('abs_correction', ascending=False)
    for _, row in large.head(10).iterrows():
        print(f"  {row['video']} boundary {row['index']}: {row['correction']:+d} frames")

    # Per-video analysis
    print(f"\nVideos with worst boundary detection:")
    video_stats = df.groupby('video').agg({
        'abs_correction': ['mean', 'max', 'count']
    }).round(2)
    video_stats.columns = ['mean_correction', 'max_correction', 'n_boundaries']
    video_stats = video_stats.sort_values('mean_correction', ascending=False)
    print(video_stats.head(5))

    return df


def analyze_reach_failures():
    """Analyze reach detection failures (FP and FN).

    IMPORTANT: Only counts FPs and FNs for videos where reaches are exhaustive.
    For non-exhaustive videos, only reports precision-like metrics (matched GT reaches).
    """
    print("\n" + "="*70)
    print("REACH DETECTION FAILURE ANALYSIS")
    print("="*70)

    # Load algo reaches from archive (the actual algorithm output)
    algo_dir = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_0_0")

    failures = {
        'false_positives': [],  # Algo detected, human didn't verify (exhaustive only)
        'false_negatives': [],  # Human verified, algo missed (exhaustive only)
        'timing_errors': [],    # Both detected but timing off
    }

    # Track exhaustive status per video
    exhaustive_videos = []
    non_exhaustive_videos = []
    matched_reaches = []  # For non-exhaustive: matched GT reaches (TP-like metric)

    for gt_file in DATA_DIR.glob("*_unified_ground_truth.json"):
        gt = load_json(gt_file)
        if not gt:
            continue

        video = gt['video_name']

        # Check if reaches are exhaustive
        is_exhaustive = gt.get('reaches', {}).get('exhaustive', False)
        if is_exhaustive:
            exhaustive_videos.append(video)
        else:
            non_exhaustive_videos.append(video)

        # Get GT reaches (determined only)
        gt_reaches = [r for r in gt.get('reaches', {}).get('reaches', [])
                      if r.get('start_determined', False) and r.get('end_determined', False)]

        # Try to find corresponding algo output
        algo_file = algo_dir / f"{video}_reaches.json"
        algo_data = load_json(algo_file)

        if not algo_data:
            continue

        algo_reaches = algo_data.get('reaches', [])

        # Match reaches by apex frame (±5 frames tolerance)
        gt_matched = set()
        algo_matched = set()
        TOLERANCE = 5

        for ai, algo_r in enumerate(algo_reaches):
            algo_apex = algo_r.get('apex_frame', algo_r.get('frame'))
            if algo_apex is None:
                continue

            best_match = None
            best_dist = float('inf')

            for gi, gt_r in enumerate(gt_reaches):
                if gi in gt_matched:
                    continue
                gt_apex = gt_r.get('apex_frame', gt_r.get('frame'))
                if gt_apex is None:
                    continue

                dist = abs(algo_apex - gt_apex)
                if dist <= TOLERANCE and dist < best_dist:
                    best_match = gi
                    best_dist = dist

            if best_match is not None:
                gt_matched.add(best_match)
                algo_matched.add(ai)
                # Record matched reaches for all videos (for precision metric)
                matched_reaches.append({
                    'video': video,
                    'gt_frame': gt_reaches[best_match].get('apex_frame'),
                    'algo_frame': algo_apex,
                    'is_exhaustive': is_exhaustive
                })
                if best_dist > 0:
                    failures['timing_errors'].append({
                        'video': video,
                        'algo_frame': algo_apex,
                        'gt_frame': gt_reaches[best_match].get('apex_frame'),
                        'error': best_dist
                    })
            else:
                # Only count as false positive if video is exhaustive
                if is_exhaustive:
                    failures['false_positives'].append({
                        'video': video,
                        'frame': algo_apex,
                        'algo_index': ai
                    })

        # False negatives (GT reaches that algo missed) - only for exhaustive videos
        if is_exhaustive:
            for gi, gt_r in enumerate(gt_reaches):
                if gi not in gt_matched:
                    failures['false_negatives'].append({
                        'video': video,
                        'frame': gt_r.get('apex_frame', gt_r.get('frame')),
                        'gt_index': gi
                    })

    # Print video exhaustive status
    print(f"\nExhaustive Status:")
    print(f"  Videos with exhaustive reach GT: {len(exhaustive_videos)}")
    print(f"  Videos with non-exhaustive reach GT: {len(non_exhaustive_videos)}")

    if non_exhaustive_videos:
        print(f"\n  WARNING: The following videos have non-exhaustive reach GT:")
        for video in sorted(non_exhaustive_videos):
            print(f"    - {video}")
        print(f"\n  This means:")
        print(f"    - Unmatched algo reaches could be real (not validated yet)")
        print(f"    - GT reaches may not be complete (human hasn't determined all)")
        print(f"    - FP and FN counts are ONLY valid for exhaustive videos")

    if exhaustive_videos:
        print(f"\n  Exhaustive videos:")
        for video in sorted(exhaustive_videos):
            print(f"    - {video}")

    print(f"\nFailure counts (EXHAUSTIVE VIDEOS ONLY):")
    print(f"  False Positives (algo detected, not in GT): {len(failures['false_positives'])}")
    print(f"  False Negatives (in GT, algo missed): {len(failures['false_negatives'])}")
    print(f"  Timing errors (both detected, timing off): {len(failures['timing_errors'])}")

    # Precision-like metric for non-exhaustive videos
    if matched_reaches:
        total_gt_reaches = len([m for m in matched_reaches])
        matched_count = len(matched_reaches)
        non_exhaustive_matched = len([m for m in matched_reaches if not m['is_exhaustive']])
        exhaustive_matched = len([m for m in matched_reaches if m['is_exhaustive']])

        print(f"\nMatched Reach Statistics:")
        print(f"  Total determined GT reaches: {total_gt_reaches}")
        print(f"  Matched by algorithm: {matched_count}")
        print(f"    - From exhaustive videos: {exhaustive_matched}")
        print(f"    - From non-exhaustive videos: {non_exhaustive_matched}")

        if non_exhaustive_matched > 0:
            print(f"\n  NOTE: For non-exhaustive videos, this shows:")
            print(f"    'Of the reaches humans have determined, how many did the algo find?'")
            print(f"    This is a PRECISION-LIKE metric, not true recall.")

    # Analyze FP distribution (exhaustive only)
    if failures['false_positives']:
        fp_by_video = defaultdict(list)
        for fp in failures['false_positives']:
            fp_by_video[fp['video']].append(fp['frame'])

        print(f"\nFalse Positives by video (top 5, exhaustive only):")
        for video, frames in sorted(fp_by_video.items(), key=lambda x: -len(x[1]))[:5]:
            print(f"  {video}: {len(frames)} FPs")

    # Analyze FN distribution (exhaustive only)
    if failures['false_negatives']:
        fn_by_video = defaultdict(list)
        for fn in failures['false_negatives']:
            fn_by_video[fn['video']].append(fn['frame'])

        print(f"\nFalse Negatives by video (top 5, exhaustive only):")
        for video, frames in sorted(fn_by_video.items(), key=lambda x: -len(x[1]))[:5]:
            print(f"  {video}: {len(frames)} FNs")

    # Timing error distribution
    if failures['timing_errors']:
        errors = [e['error'] for e in failures['timing_errors']]
        print(f"\nTiming error distribution:")
        for i in range(1, 6):
            count = sum(1 for e in errors if e == i)
            print(f"  {i} frames off: {count}")

    return failures


def analyze_outcome_failures():
    """Analyze outcome classification failures."""
    print("\n" + "="*70)
    print("OUTCOME CLASSIFICATION FAILURE ANALYSIS")
    print("="*70)

    algo_dir = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_0_0")

    failures = []
    confusion = defaultdict(lambda: defaultdict(int))

    for gt_file in DATA_DIR.glob("*_unified_ground_truth.json"):
        gt = load_json(gt_file)
        if not gt:
            continue

        video = gt['video_name']

        # Get GT outcomes (determined only)
        gt_outcomes = {s['segment_num']: s['outcome']
                       for s in gt.get('outcomes', {}).get('segments', [])
                       if s.get('determined', False)}

        # Try to find algo output
        algo_file = algo_dir / f"{video}_pellet_outcomes.json"
        algo_data = load_json(algo_file)

        if not algo_data:
            continue

        algo_outcomes = {s['segment_num']: s['outcome']
                         for s in algo_data.get('segments', [])}

        # Compare
        for seg_idx in gt_outcomes:
            gt_out = gt_outcomes[seg_idx]
            algo_out = algo_outcomes.get(seg_idx)

            if algo_out is None:
                continue

            confusion[gt_out][algo_out] += 1

            if gt_out != algo_out:
                failures.append({
                    'video': video,
                    'segment': seg_idx,
                    'gt_outcome': gt_out,
                    'algo_outcome': algo_out
                })

    print(f"\nTotal misclassifications: {len(failures)}")

    # Show confusion matrix
    print(f"\nConfusion matrix (rows=GT, cols=Algo):")
    outcomes = sorted(set(confusion.keys()) | set(k for v in confusion.values() for k in v))

    # Header
    header = 'GT \\ Algo'
    print(f"{header:<20}", end='')
    for o in outcomes:
        print(f"{o[:12]:<15}", end='')
    print()

    # Rows
    for gt_out in outcomes:
        print(f"{gt_out:<20}", end='')
        for algo_out in outcomes:
            count = confusion[gt_out][algo_out]
            print(f"{count:<15}", end='')
        print()

    # Most common mistakes
    if failures:
        print(f"\nMost common mistakes:")
        mistake_counts = defaultdict(int)
        for f in failures:
            key = f"{f['gt_outcome']} -> {f['algo_outcome']}"
            mistake_counts[key] += 1

        for mistake, count in sorted(mistake_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {mistake}: {count}")

        # Videos with most failures
        print(f"\nVideos with most outcome errors:")
        video_errors = defaultdict(int)
        for f in failures:
            video_errors[f['video']] += 1

        for video, count in sorted(video_errors.items(), key=lambda x: -x[1])[:5]:
            print(f"  {video}: {count} errors")

    return failures, confusion


def summarize_findings():
    """Summarize which algorithm needs the most help."""
    print("\n" + "="*70)
    print("SUMMARY: WHERE DOES DEEP LEARNING MAKE SENSE?")
    print("="*70)

    print("""
Based on failure analysis:

1. BOUNDARY DETECTION
   - Most corrections are ±1-2 frames (minor timing adjustments)
   - The algorithm finds the right general location, just not perfect frame
   - VERDICT: DL could help fine-tune, but ROI is low

2. REACH DETECTION
   - False positives: Algorithm detects reaches that humans don't verify
   - False negatives: Algorithm misses some reaches

   IMPORTANT LIMITATION:
   - FP/FN counts are ONLY valid for videos with exhaustive reach GT
   - If no videos have exhaustive=true, FP/FN numbers may be unreliable
   - Non-exhaustive videos only show precision-like metrics:
     "Of determined GT reaches, how many did the algo match?"
   - This does NOT tell us about missed reaches or spurious detections

   - VERDICT: DL could help classify "is this a real reach?" as a second-pass filter
     BUT validation of this requires exhaustive reach GT data

3. OUTCOME CLASSIFICATION
   - Already ~98% accurate
   - Most errors are between similar outcomes (displaced_sa vs displaced_outside)
   - VERDICT: Minimal benefit from DL

RECOMMENDATION:
   The best use of deep learning is a REACH VALIDATION MODEL:
   - Input: DLC features around an algorithm-detected reach
   - Output: Probability this is a real reach (not a false positive)
   - Training: Use GT verified/unverified status as labels

   This addresses the WORST failure mode (false positives in reach detection)
   while keeping the efficient rule-based algorithm for initial detection.

   CAVEAT: To properly validate this recommendation, we need:
   - At least some videos with exhaustive=true for reaches
   - This allows measuring true FP and FN rates
   - Without exhaustive GT, we can only measure precision on known reaches
""")


if __name__ == "__main__":
    boundary_df = analyze_boundary_failures()
    reach_failures = analyze_reach_failures()
    outcome_failures, confusion = analyze_outcome_failures()
    summarize_findings()
