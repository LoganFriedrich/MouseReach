"""
Diagnostic script for analyzing reach start frame detection errors.

This script identifies why the algorithm's start frames differ from ground truth
and categorizes the types of errors to guide algorithm improvements.

Target: Increase start-within-2-frames accuracy from 91.4% to 99%.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import h5py

# Constants
RH_POINTS = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']
HAND_THRESHOLD = 0.5
NOSE_THRESHOLD = 25  # pixels
MAX_MATCH_DIST = 30
START_ERROR_THRESHOLD = 2

# Paths
PROCESSING_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_RESULTS_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_5_0")


def load_ground_truth(processing_dir: Path) -> Dict[str, List[Dict]]:
    """Load all ground truth files."""
    gt_data = {}
    for gt_file in processing_dir.glob("*_unified_ground_truth.json"):
        video_name = gt_file.stem.replace("_unified_ground_truth", "")
        with open(gt_file, 'r') as f:
            data = json.load(f)
            video_name = data.get('video_name', video_name)
            # Filter to valid reaches - reaches are nested under reaches.reaches
            all_reaches = data.get('reaches', {}).get('reaches', [])
            valid_reaches = [
                r for r in all_reaches
                if r.get('start_determined', False)
                and r.get('end_determined', False)
                and not r.get('exclude_from_analysis', False)
            ]
            gt_data[video_name] = valid_reaches
    return gt_data


def load_algo_results(algo_dir: Path) -> Dict[str, List[Dict]]:
    """Load all algorithm results and flatten segments."""
    algo_data = {}
    for algo_file in algo_dir.glob("*_reaches.json"):
        video_name = algo_file.stem.replace("_reaches", "")
        with open(algo_file, 'r') as f:
            data = json.load(f)
            # Flatten from segments[].reaches[]
            all_reaches = []
            for segment in data.get('segments', []):
                for reach in segment.get('reaches', []):
                    # Store segment info with reach
                    reach['segment_id'] = segment.get('segment_id')
                    all_reaches.append(reach)
            algo_data[video_name] = all_reaches
    return algo_data


def load_dlc_data(video_name: str, processing_dir: Path) -> Optional[pd.DataFrame]:
    """Load DLC h5 file for a video."""
    dlc_files = list(processing_dir.glob(f"{video_name}DLC*.h5"))
    if not dlc_files:
        print(f"Warning: No DLC file found for {video_name}")
        return None

    dlc_file = dlc_files[0]
    df = pd.read_hdf(dlc_file)

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        scorer = df.columns.get_level_values(0)[0]
        df = df[scorer]
        # Rename columns to bp_coord format
        new_columns = []
        for col in df.columns:
            if isinstance(col, tuple):
                bp, coord = col
                new_columns.append(f"{bp}_{coord}")
            else:
                new_columns.append(col)
        df.columns = new_columns

    return df


def load_segments(video_name: str, processing_dir: Path) -> Optional[Dict]:
    """Load segment data for a video."""
    seg_file = processing_dir / f"{video_name}_segments.json"
    if not seg_file.exists():
        print(f"Warning: No segments file found for {video_name}")
        return None

    with open(seg_file, 'r') as f:
        return json.load(f)


def get_slit_center(segment_id: int, segments_data: Dict) -> Optional[float]:
    """Calculate slit center from segment's median BOXL and BOXR."""
    for segment in segments_data.get('segments', []):
        if segment.get('segment_id') == segment_id:
            boxl_x = segment.get('BOXL_x_median')
            boxr_x = segment.get('BOXR_x_median')
            if boxl_x is not None and boxr_x is not None:
                return (boxl_x + boxr_x) / 2
    return None


def match_reaches(gt_reaches: List[Dict], algo_reaches: List[Dict]) -> List[Tuple[Dict, Dict, int]]:
    """Match GT and algo reaches using nearest start_frame (greedy)."""
    matches = []
    used_algo = set()

    for gt_reach in gt_reaches:
        gt_start = gt_reach['start_frame']
        best_match = None
        best_dist = MAX_MATCH_DIST + 1

        for i, algo_reach in enumerate(algo_reaches):
            if i in used_algo:
                continue
            algo_start = algo_reach['start_frame']
            dist = abs(algo_start - gt_start)
            if dist < best_dist:
                best_dist = dist
                best_match = i

        if best_match is not None and best_dist <= MAX_MATCH_DIST:
            offset = algo_reaches[best_match]['start_frame'] - gt_start
            matches.append((gt_reach, algo_reaches[best_match], offset))
            used_algo.add(best_match)

    return matches


def is_hand_visible(dlc_df: pd.DataFrame, frame: int, threshold: float = HAND_THRESHOLD) -> Tuple[bool, float]:
    """Check if any hand point is visible at frame with given threshold."""
    if frame >= len(dlc_df):
        return False, 0.0

    best_likelihood = 0.0
    for bp in RH_POINTS:
        likelihood_col = f"{bp}_likelihood"
        if likelihood_col in dlc_df.columns:
            likelihood = dlc_df.iloc[frame][likelihood_col]
            best_likelihood = max(best_likelihood, likelihood)
            if likelihood >= threshold:
                return True, best_likelihood

    return False, best_likelihood


def is_nose_engaged(dlc_df: pd.DataFrame, frame: int, slit_center: float, threshold: float = NOSE_THRESHOLD) -> Tuple[bool, float]:
    """Check if nose is engaged (within threshold pixels of slit)."""
    if frame >= len(dlc_df) or slit_center is None:
        return False, float('inf')

    nose_x_col = "Nose_x"
    if nose_x_col not in dlc_df.columns:
        return False, float('inf')

    nose_x = dlc_df.iloc[frame][nose_x_col]
    distance = abs(nose_x - slit_center)

    return distance <= threshold, distance


def analyze_frame_range(dlc_df: pd.DataFrame, start_frame: int, end_frame: int,
                        slit_center: float) -> List[Dict]:
    """Analyze hand visibility and nose engagement for a frame range."""
    analysis = []
    for frame in range(start_frame, min(end_frame + 1, len(dlc_df))):
        hand_vis, hand_lik = is_hand_visible(dlc_df, frame)
        nose_eng, nose_dist = is_nose_engaged(dlc_df, frame, slit_center)
        analysis.append({
            'frame': frame,
            'hand_visible': hand_vis,
            'hand_likelihood': hand_lik,
            'nose_engaged': nose_eng,
            'nose_distance': nose_dist
        })
    return analysis


def categorize_error(gt_reach: Dict, algo_reach: Dict, offset: int,
                     dlc_df: pd.DataFrame, slit_center: float,
                     algo_reaches: List[Dict]) -> Tuple[str, Dict]:
    """Categorize the type of start detection error."""
    gt_start = gt_reach['start_frame']
    algo_start = algo_reach['start_frame']

    details = {}

    # Check hand visibility at GT start
    hand_vis_gt, hand_lik_gt = is_hand_visible(dlc_df, gt_start)
    nose_eng_gt, nose_dist_gt = is_nose_engaged(dlc_df, gt_start, slit_center)

    details['hand_visible_at_gt'] = hand_vis_gt
    details['hand_likelihood_at_gt'] = hand_lik_gt
    details['nose_engaged_at_gt'] = nose_eng_gt
    details['nose_distance_at_gt'] = nose_dist_gt

    # Check hand visibility at algo start
    hand_vis_algo, hand_lik_algo = is_hand_visible(dlc_df, algo_start)
    nose_eng_algo, nose_dist_algo = is_nose_engaged(dlc_df, algo_start, slit_center)

    details['hand_visible_at_algo'] = hand_vis_algo
    details['hand_likelihood_at_algo'] = hand_lik_algo
    details['nose_engaged_at_algo'] = nose_eng_algo
    details['nose_distance_at_algo'] = nose_dist_algo

    if offset > START_ERROR_THRESHOLD:  # Algo late
        # Hand not visible
        if not hand_vis_gt:
            if hand_lik_gt >= 0.3:
                details['category'] = 'HAND_LOW_CONFIDENCE'
                return 'HAND_LOW_CONFIDENCE', details
            else:
                details['category'] = 'HAND_NOT_VISIBLE'
                return 'HAND_NOT_VISIBLE', details

        # Hand visible but nose not engaged
        if hand_vis_gt and not nose_eng_gt:
            details['category'] = 'NOSE_NOT_ENGAGED'
            return 'NOSE_NOT_ENGAGED', details

    elif offset < -START_ERROR_THRESHOLD:  # Algo early
        # Check for merge artifact (previous reach ending nearby)
        for other_reach in algo_reaches:
            if other_reach is algo_reach:
                continue
            other_end = other_reach['end_frame']
            if 0 < (algo_start - other_end) <= 5:
                details['category'] = 'MERGE_ARTIFACT'
                details['previous_reach_end'] = other_end
                return 'MERGE_ARTIFACT', details

    details['category'] = 'OTHER'
    return 'OTHER', details


def print_example_cases(error_cases: List[Dict], category: str, max_examples: int = 10):
    """Print detailed examples of error cases."""
    category_cases = [c for c in error_cases if c['category'] == category]

    if not category_cases:
        return

    print(f"\n{'='*80}")
    print(f"DETAILED EXAMPLES: {category} (showing first {max_examples})")
    print(f"{'='*80}\n")

    for i, case in enumerate(category_cases[:max_examples], 1):
        print(f"Example {i}/{min(len(category_cases), max_examples)}:")
        print(f"  Video: {case['video']}")
        print(f"  GT start: {case['gt_start']}, Algo start: {case['algo_start']}, Offset: {case['offset']}")
        print(f"  At GT start:")
        print(f"    Hand visible: {case['details']['hand_visible_at_gt']}, Likelihood: {case['details']['hand_likelihood_at_gt']:.3f}")
        print(f"    Nose engaged: {case['details']['nose_engaged_at_gt']}, Distance: {case['details']['nose_distance_at_gt']:.1f}px")
        print(f"  At Algo start:")
        print(f"    Hand visible: {case['details']['hand_visible_at_algo']}, Likelihood: {case['details']['hand_likelihood_at_algo']:.3f}")
        print(f"    Nose engaged: {case['details']['nose_engaged_at_algo']}, Distance: {case['details']['nose_distance_at_algo']:.1f}px")

        # Frame-by-frame for late cases
        if case['offset'] > START_ERROR_THRESHOLD and 'frame_analysis' in case:
            print(f"  Frame-by-frame (GT start → Algo start):")
            for frame_info in case['frame_analysis']:
                print(f"    Frame {frame_info['frame']}: Hand={'Y' if frame_info['hand_visible'] else 'N'}({frame_info['hand_likelihood']:.2f}), "
                      f"Nose={'Y' if frame_info['nose_engaged'] else 'N'}({frame_info['nose_distance']:.1f}px)")

        if 'previous_reach_end' in case['details']:
            print(f"  Previous reach ended at frame {case['details']['previous_reach_end']}")

        print()


def main():
    print("="*80)
    print("REACH START FRAME ERROR DIAGNOSTIC")
    print("="*80)
    print(f"\nTarget: Increase start-within-{START_ERROR_THRESHOLD}-frames accuracy from 91.4% to 99%")
    print(f"Analyzing all cases with |offset| > {START_ERROR_THRESHOLD}\n")

    # Load all data
    print("Loading ground truth...")
    gt_data = load_ground_truth(PROCESSING_DIR)
    total_gt_reaches = sum(len(reaches) for reaches in gt_data.values())
    print(f"  Loaded {len(gt_data)} videos, {total_gt_reaches} valid GT reaches")

    print("Loading algorithm results...")
    algo_data = load_algo_results(ALGO_RESULTS_DIR)
    total_algo_reaches = sum(len(reaches) for reaches in algo_data.values())
    print(f"  Loaded {len(algo_data)} videos, {total_algo_reaches} algo reaches")

    # Match reaches and analyze errors
    all_offsets = []
    error_cases = []
    category_counts = Counter()

    print("\nMatching reaches and analyzing errors...")

    for video_name in sorted(gt_data.keys()):
        if video_name not in algo_data:
            print(f"  Skipping {video_name}: no algo results")
            continue

        # Load DLC and segments
        dlc_df = load_dlc_data(video_name, PROCESSING_DIR)
        segments_data = load_segments(video_name, PROCESSING_DIR)

        if dlc_df is None or segments_data is None:
            print(f"  Skipping {video_name}: missing DLC or segments data")
            continue

        # Match reaches
        matches = match_reaches(gt_data[video_name], algo_data[video_name])
        print(f"  {video_name}: {len(matches)} matched reaches")

        for gt_reach, algo_reach, offset in matches:
            all_offsets.append(offset)

            # Analyze errors beyond threshold
            if abs(offset) > START_ERROR_THRESHOLD:
                gt_start = gt_reach['start_frame']
                algo_start = algo_reach['start_frame']

                # Get slit center
                segment_id = algo_reach.get('segment_id')
                slit_center = get_slit_center(segment_id, segments_data) if segment_id else None

                if slit_center is None:
                    print(f"    Warning: No slit center for reach at frame {gt_start}")
                    continue

                # Categorize error
                category, details = categorize_error(
                    gt_reach, algo_reach, offset, dlc_df, slit_center, algo_data[video_name]
                )
                category_counts[category] += 1

                error_case = {
                    'video': video_name,
                    'gt_start': gt_start,
                    'algo_start': algo_start,
                    'offset': offset,
                    'category': category,
                    'details': details
                }

                # Add frame-by-frame analysis for late cases (first 10 only to save time)
                if offset > START_ERROR_THRESHOLD and len([c for c in error_cases if c['offset'] > START_ERROR_THRESHOLD]) < 10:
                    frame_analysis = analyze_frame_range(dlc_df, gt_start, algo_start, slit_center)
                    error_case['frame_analysis'] = frame_analysis

                error_cases.append(error_case)

    # =====================
    # A. OFFSET DISTRIBUTION
    # =====================
    print("\n" + "="*80)
    print("A. START OFFSET DISTRIBUTION")
    print("="*80)

    offsets_array = np.array(all_offsets)
    within_threshold = np.sum(np.abs(offsets_array) <= START_ERROR_THRESHOLD)
    total_matches = len(all_offsets)
    accuracy = 100 * within_threshold / total_matches if total_matches > 0 else 0

    print(f"\nTotal matched reaches: {total_matches}")
    print(f"Within {START_ERROR_THRESHOLD} frames: {within_threshold} ({accuracy:.1f}%)")
    print(f"Errors (|offset| > {START_ERROR_THRESHOLD}): {len(error_cases)} ({100 - accuracy:.1f}%)")

    algo_early = np.sum(offsets_array < -START_ERROR_THRESHOLD)
    algo_late = np.sum(offsets_array > START_ERROR_THRESHOLD)

    print(f"\nError breakdown:")
    print(f"  Algo EARLY (offset < -{START_ERROR_THRESHOLD}): {algo_early} ({100*algo_early/total_matches:.1f}%)")
    print(f"  Algo LATE (offset > {START_ERROR_THRESHOLD}): {algo_late} ({100*algo_late/total_matches:.1f}%)")

    print(f"\nOffset statistics:")
    print(f"  Mean: {np.mean(offsets_array):.2f} frames")
    print(f"  Median: {np.median(offsets_array):.2f} frames")
    print(f"  Std: {np.std(offsets_array):.2f} frames")
    print(f"  Min: {np.min(offsets_array)}, Max: {np.max(offsets_array)}")
    print(f"  25th percentile: {np.percentile(offsets_array, 25):.1f}")
    print(f"  75th percentile: {np.percentile(offsets_array, 75):.1f}")
    print(f"  95th percentile: {np.percentile(offsets_array, 95):.1f}")
    print(f"  99th percentile: {np.percentile(offsets_array, 99):.1f}")

    print("\nOffset histogram (bins of 1 frame, range -20 to +20):")
    hist, bin_edges = np.histogram(offsets_array, bins=np.arange(-20, 22, 1))
    for i, count in enumerate(hist):
        bin_start = int(bin_edges[i])
        bar = '#' * int(50 * count / max(hist)) if max(hist) > 0 else ''
        print(f"  {bin_start:3d} to {bin_start+1:3d}: {count:4d} {bar}")

    # =====================
    # D. CATEGORIZATION
    # =====================
    print("\n" + "="*80)
    print("D. ERROR CATEGORIZATION")
    print("="*80)

    print(f"\nTotal errors analyzed: {len(error_cases)}")
    for category, count in category_counts.most_common():
        pct = 100 * count / len(error_cases) if error_cases else 0
        print(f"  {category}: {count} ({pct:.1f}%)")

    # =====================
    # B. ALGO LATE ANALYSIS
    # =====================
    print("\n" + "="*80)
    print("B. ALGO LATE CASES (offset > 2)")
    print("="*80)

    late_cases = [c for c in error_cases if c['offset'] > START_ERROR_THRESHOLD]
    print(f"\nTotal late cases: {len(late_cases)}")

    if late_cases:
        # Hand visibility stats
        hand_not_vis_count = sum(1 for c in late_cases if not c['details']['hand_visible_at_gt'])
        hand_low_conf_count = sum(1 for c in late_cases if 0.3 <= c['details']['hand_likelihood_at_gt'] < HAND_THRESHOLD)

        print(f"\nHand visibility at GT start:")
        print(f"  Not visible (< 0.5): {hand_not_vis_count} ({100*hand_not_vis_count/len(late_cases):.1f}%)")
        print(f"  Low confidence (0.3-0.5): {hand_low_conf_count} ({100*hand_low_conf_count/len(late_cases):.1f}%)")

        hand_likelihoods = [c['details']['hand_likelihood_at_gt'] for c in late_cases]
        print(f"  Mean likelihood: {np.mean(hand_likelihoods):.3f}")
        print(f"  Median likelihood: {np.median(hand_likelihoods):.3f}")
        print(f"  25th percentile: {np.percentile(hand_likelihoods, 25):.3f}")
        print(f"  5th percentile: {np.percentile(hand_likelihoods, 5):.3f}")

        # Nose engagement stats
        nose_not_eng_count = sum(1 for c in late_cases if not c['details']['nose_engaged_at_gt'])
        print(f"\nNose engagement at GT start:")
        print(f"  Not engaged (> {NOSE_THRESHOLD}px): {nose_not_eng_count} ({100*nose_not_eng_count/len(late_cases):.1f}%)")

        nose_distances = [c['details']['nose_distance_at_gt'] for c in late_cases
                         if not c['details']['nose_engaged_at_gt']]
        if nose_distances:
            print(f"  Mean distance: {np.mean(nose_distances):.1f}px")
            print(f"  Median distance: {np.median(nose_distances):.1f}px")
            print(f"  75th percentile: {np.percentile(nose_distances, 75):.1f}px")
            print(f"  95th percentile: {np.percentile(nose_distances, 95):.1f}px")

    # =====================
    # C. ALGO EARLY ANALYSIS
    # =====================
    print("\n" + "="*80)
    print("C. ALGO EARLY CASES (offset < -2)")
    print("="*80)

    early_cases = [c for c in error_cases if c['offset'] < -START_ERROR_THRESHOLD]
    print(f"\nTotal early cases: {len(early_cases)}")

    if early_cases:
        merge_artifacts = sum(1 for c in early_cases if c['category'] == 'MERGE_ARTIFACT')
        print(f"  Merge artifacts: {merge_artifacts} ({100*merge_artifacts/len(early_cases):.1f}%)")

        hand_vis_count = sum(1 for c in early_cases if c['details']['hand_visible_at_algo'])
        nose_eng_count = sum(1 for c in early_cases if c['details']['nose_engaged_at_algo'])

        print(f"  Hand visible at algo start: {hand_vis_count} ({100*hand_vis_count/len(early_cases):.1f}%)")
        print(f"  Nose engaged at algo start: {nose_eng_count} ({100*nose_eng_count/len(early_cases):.1f}%)")

    # =====================
    # E. NOSE NOT ENGAGED DETAILS
    # =====================
    print("\n" + "="*80)
    print("E. NOSE NOT ENGAGED CASES")
    print("="*80)

    nose_cases = [c for c in error_cases if c['category'] == 'NOSE_NOT_ENGAGED']
    print(f"\nTotal nose not engaged: {len(nose_cases)}")

    if nose_cases:
        distances = [c['details']['nose_distance_at_gt'] for c in nose_cases]
        print(f"\nNose distance distribution at GT start:")
        print(f"  Mean: {np.mean(distances):.1f}px")
        print(f"  Median: {np.median(distances):.1f}px")
        print(f"  25th percentile: {np.percentile(distances, 25):.1f}px")
        print(f"  75th percentile: {np.percentile(distances, 75):.1f}px")
        print(f"  95th percentile: {np.percentile(distances, 95):.1f}px")
        print(f"  99th percentile: {np.percentile(distances, 99):.1f}px")

        # What threshold would capture 95% / 99%?
        threshold_95 = np.percentile(distances, 95)
        threshold_99 = np.percentile(distances, 99)
        print(f"\nSuggested nose engagement thresholds:")
        print(f"  For 95% capture: {threshold_95:.1f}px (current: {NOSE_THRESHOLD}px)")
        print(f"  For 99% capture: {threshold_99:.1f}px (current: {NOSE_THRESHOLD}px)")

    # =====================
    # F. HAND VISIBILITY DETAILS
    # =====================
    print("\n" + "="*80)
    print("F. HAND VISIBILITY CASES")
    print("="*80)

    hand_cases = [c for c in error_cases
                  if c['category'] in ['HAND_NOT_VISIBLE', 'HAND_LOW_CONFIDENCE']]
    print(f"\nTotal hand visibility issues: {len(hand_cases)}")

    if hand_cases:
        likelihoods = [c['details']['hand_likelihood_at_gt'] for c in hand_cases]
        print(f"\nHand likelihood distribution at GT start:")
        print(f"  Mean: {np.mean(likelihoods):.3f}")
        print(f"  Median: {np.median(likelihoods):.3f}")
        print(f"  25th percentile: {np.percentile(likelihoods, 25):.3f}")
        print(f"  75th percentile: {np.percentile(likelihoods, 75):.3f}")
        print(f"  95th percentile: {np.percentile(likelihoods, 95):.3f}")
        print(f"  99th percentile: {np.percentile(likelihoods, 99):.3f}")

        # What threshold would capture 95% / 99%?
        # Note: Lower threshold means MORE inclusive, so we want low percentiles
        threshold_95 = np.percentile(likelihoods, 5)  # 5th percentile to capture 95%
        threshold_99 = np.percentile(likelihoods, 1)  # 1st percentile to capture 99%
        print(f"\nSuggested hand likelihood thresholds:")
        print(f"  For 95% capture: {threshold_95:.3f} (current: {HAND_THRESHOLD})")
        print(f"  For 99% capture: {threshold_99:.3f} (current: {HAND_THRESHOLD})")

    # =====================
    # DETAILED EXAMPLES
    # =====================
    print_example_cases(error_cases, 'NOSE_NOT_ENGAGED', max_examples=10)
    print_example_cases(error_cases, 'HAND_NOT_VISIBLE', max_examples=10)
    print_example_cases(error_cases, 'HAND_LOW_CONFIDENCE', max_examples=10)
    print_example_cases(error_cases, 'MERGE_ARTIFACT', max_examples=10)
    print_example_cases(error_cases, 'OTHER', max_examples=10)

    # =====================
    # SUMMARY & RECOMMENDATIONS
    # =====================
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)

    print(f"\nCurrent accuracy: {accuracy:.1f}%")
    print(f"Target accuracy: 99.0%")
    print(f"Gap to close: {99.0 - accuracy:.1f} percentage points")
    print(f"Errors to eliminate: {len(error_cases) - int(0.01 * total_matches)} reaches")

    print("\nTop error categories to address:")
    for category, count in category_counts.most_common(3):
        pct_of_errors = 100 * count / len(error_cases) if error_cases else 0
        pct_of_total = 100 * count / total_matches if total_matches > 0 else 0
        print(f"  {category}: {count} errors ({pct_of_errors:.1f}% of errors, {pct_of_total:.2f}% of total)")

        if category == 'NOSE_NOT_ENGAGED' and nose_cases:
            distances = [c['details']['nose_distance_at_gt'] for c in nose_cases]
            threshold_99 = np.percentile(distances, 99)
            print(f"    → Increase nose threshold from {NOSE_THRESHOLD}px to {threshold_99:.1f}px")

        elif category in ['HAND_NOT_VISIBLE', 'HAND_LOW_CONFIDENCE'] and hand_cases:
            likelihoods = [c['details']['hand_likelihood_at_gt'] for c in hand_cases]
            threshold_99 = np.percentile(likelihoods, 1)
            print(f"    → Lower hand threshold from {HAND_THRESHOLD} to {threshold_99:.3f}")

        elif category == 'MERGE_ARTIFACT':
            print(f"    → Improve retraction confirmation to end reaches earlier")

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
