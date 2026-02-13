"""
Analyze FALSE POSITIVE reaches in the worst-performing videos.

Goal: Understand what DLC features distinguish REAL reaches from FALSE POSITIVE
detections. This directly informs algorithm improvements.

Worst videos:
- CNT0311_P2: 34% precision (143 FP, 75 TP)
- CNT0110_P2: 41% precision (83 FP, 57 TP)
- CNT0307_P4: 50% precision (126 FP, 124 TP)
- CNT0312_P2: 52% precision (118 FP, 126 TP)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_0_0")


def load_algo_reaches(video: str) -> list:
    """Load algorithm-detected reaches."""
    reach_file = ALGO_DIR / f"{video}_reaches.json"
    if not reach_file.exists():
        # Try finding in Processing
        reach_file = DATA_DIR / f"{video}_reaches.json"
    if not reach_file.exists():
        return []
    with open(reach_file) as f:
        data = json.load(f)
    # Flatten reaches from all segments
    all_reaches = []
    for seg in data.get('segments', []):
        for r in seg.get('reaches', []):
            r['segment_num'] = seg['segment_num']
            all_reaches.append(r)
    return all_reaches


def load_gt_reaches(video: str) -> tuple[list, bool]:
    """Load ground truth reaches and exhaustive flag.

    Returns:
        tuple: (list of determined reaches, exhaustive flag)
    """
    gt_file = DATA_DIR / f"{video}_unified_ground_truth.json"
    if not gt_file.exists():
        return [], False
    with open(gt_file) as f:
        gt = json.load(f)

    reaches_section = gt.get('reaches', {})
    determined_reaches = [r for r in reaches_section.get('reaches', [])
                          if r.get('start_determined', False) and r.get('end_determined', False)]
    exhaustive = reaches_section.get('exhaustive', False)

    return determined_reaches, exhaustive


def match_reaches(algo_reaches: list, gt_reaches: list, tolerance: int = 5) -> dict:
    """Match algo reaches to GT reaches by apex frame."""
    gt_matched = set()
    algo_matched = set()
    matches = []

    for ai, algo_r in enumerate(algo_reaches):
        algo_apex = algo_r.get('apex_frame')
        if algo_apex is None:
            continue

        best_gi = None
        best_dist = float('inf')

        for gi, gt_r in enumerate(gt_reaches):
            if gi in gt_matched:
                continue
            gt_apex = gt_r.get('apex_frame')
            if gt_apex is None:
                continue

            dist = abs(algo_apex - gt_apex)
            if dist <= tolerance and dist < best_dist:
                best_gi = gi
                best_dist = dist

        if best_gi is not None:
            gt_matched.add(best_gi)
            algo_matched.add(ai)
            matches.append((ai, best_gi, best_dist))

    true_positives = [algo_reaches[ai] for ai, _, _ in matches]
    false_positives = [r for i, r in enumerate(algo_reaches) if i not in algo_matched]
    false_negatives = [r for i, r in enumerate(gt_reaches) if i not in gt_matched]

    return {
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives,
        'matches': matches
    }


def analyze_fp_features(video: str, false_positives: list, true_positives: list, exhaustive: bool):
    """Compare DLC features of FP vs TP reaches.

    Args:
        video: Video identifier
        false_positives: List of false positive reaches (only valid if exhaustive=True)
        true_positives: List of true positive reaches
        exhaustive: Whether GT is exhaustive (if False, FP list is invalid)
    """
    dlc_files = list(DATA_DIR.glob(f"*{video.split('_', 1)[1]}*DLC*.h5"))
    if not dlc_files:
        # Try different name patterns
        dlc_files = list(DATA_DIR.glob(f"*{video}*DLC*.h5"))
    if not dlc_files:
        print(f"  No DLC file found for {video}")
        return None

    df = pd.read_hdf(dlc_files[0])
    scorer = df.columns.get_level_values(0)[0]

    def get_reach_features(reach: dict) -> dict:
        """Extract features for a single reach."""
        start = reach.get('start_frame', 0)
        end = reach.get('end_frame', start + 1)
        apex = reach.get('apex_frame', start)
        duration = end - start + 1

        if start >= len(df) or end >= len(df):
            return None

        # Hand trajectory
        hand_parts = ['RightHand', 'RHLeft', 'RHRight', 'RHOut']
        hand_x_vals = []
        hand_likelihoods = []
        for frame in range(start, min(end + 1, len(df))):
            best_x = None
            best_l = 0
            for part in hand_parts:
                try:
                    l = df[(scorer, part, 'likelihood')].iloc[frame]
                    if l > best_l:
                        best_l = l
                        best_x = df[(scorer, part, 'x')].iloc[frame]
                except (KeyError, IndexError):
                    continue
            hand_x_vals.append(best_x if best_x else np.nan)
            hand_likelihoods.append(best_l)

        hand_x = np.array(hand_x_vals, dtype=float)
        hand_like = np.array(hand_likelihoods, dtype=float)

        # Remove NaN for calculations
        valid = ~np.isnan(hand_x)
        if np.sum(valid) < 2:
            return None

        hand_x_valid = hand_x[valid]
        hand_like_valid = hand_like[valid]

        # Feature extraction
        features = {
            'duration': duration,
            'max_extent': reach.get('max_extent_pixels', 0),
            # Movement features
            'total_x_travel': np.sum(np.abs(np.diff(hand_x_valid))),
            'net_x_displacement': hand_x_valid[-1] - hand_x_valid[0] if len(hand_x_valid) > 1 else 0,
            'max_x_velocity': np.max(np.abs(np.diff(hand_x_valid))) if len(hand_x_valid) > 1 else 0,
            # Likelihood features
            'mean_likelihood': np.mean(hand_like_valid),
            'max_likelihood': np.max(hand_like_valid),
            'min_likelihood': np.min(hand_like_valid),
            'likelihood_range': np.max(hand_like_valid) - np.min(hand_like_valid),
            # Trajectory smoothness
            'n_direction_changes': 0,
            # Ballistic score: how much does movement resemble out-and-back?
            'ballistic_score': 0,
        }

        # Count direction changes (non-smooth = flickering, smooth = real reach)
        if len(hand_x_valid) > 2:
            dx = np.diff(hand_x_valid)
            sign_changes = np.sum(np.abs(np.diff(np.sign(dx))) > 0)
            features['n_direction_changes'] = sign_changes

        # Ballistic score: ratio of net displacement to total travel
        # Real reach: goes out and comes back = net ~0 but total travel high
        # Flicker: random movement = both low
        if features['total_x_travel'] > 0:
            features['ballistic_score'] = features['net_x_displacement'] / features['total_x_travel']

        return features

    # Extract features for TP reaches (always valid)
    tp_features = []
    for r in true_positives:
        f = get_reach_features(r)
        if f:
            tp_features.append(f)

    # Extract features for FP reaches (only if exhaustive GT)
    fp_features = []
    if exhaustive:
        for r in false_positives:
            f = get_reach_features(r)
            if f:
                fp_features.append(f)
    else:
        # Non-exhaustive: unmatched algo reaches may be real reaches not yet determined
        fp_features = []

    return {'fp': fp_features, 'tp': tp_features, 'exhaustive': exhaustive}


def print_comparison(video: str, feature_data: dict):
    """Print feature comparison between TP and FP."""
    fp = feature_data['fp']
    tp = feature_data['tp']
    exhaustive = feature_data.get('exhaustive', True)

    if not tp:
        return

    # If non-exhaustive, we can only report TP features
    if not exhaustive and video != "ALL WORST VIDEOS":
        print(f"\n{'='*70}")
        print(f"FEATURE ANALYSIS: {video} (NON-EXHAUSTIVE GT)")
        print(f"  True Positives: {len(tp)}")
        print(f"  False Positives: SKIPPED (GT not exhaustive)")
        print(f"{'='*70}")
        print("  TP-only analysis available, but FP comparison requires exhaustive GT")
        return

    if not fp:
        return

    print(f"\n{'='*70}")
    print(f"FEATURE COMPARISON: {video}")
    print(f"  True Positives: {len(tp)}")
    print(f"  False Positives: {len(fp)}")
    print(f"{'='*70}")

    features_to_compare = [
        'duration', 'max_extent', 'total_x_travel', 'net_x_displacement',
        'max_x_velocity', 'mean_likelihood', 'max_likelihood',
        'likelihood_range', 'n_direction_changes', 'ballistic_score'
    ]

    header = f"{'Feature':<25} {'TP Mean':>10} {'TP Std':>10} {'FP Mean':>10} {'FP Std':>10} {'Separable?':>12}"
    print(header)
    print("-" * len(header))

    for feat in features_to_compare:
        tp_vals = [f[feat] for f in tp if feat in f]
        fp_vals = [f[feat] for f in fp if feat in f]

        if not tp_vals or not fp_vals:
            continue

        tp_mean = np.mean(tp_vals)
        tp_std = np.std(tp_vals)
        fp_mean = np.mean(fp_vals)
        fp_std = np.std(fp_vals)

        # Check separability: is the difference > 1 combined std?
        combined_std = (tp_std + fp_std) / 2
        separation = abs(tp_mean - fp_mean) / max(combined_std, 0.01)
        separable = "YES" if separation > 1.5 else ("maybe" if separation > 0.8 else "no")

        print(f"{feat:<25} {tp_mean:>10.2f} {tp_std:>10.2f} {fp_mean:>10.2f} {fp_std:>10.2f} {separable:>12}")


def main():
    """Analyze false positives across worst videos."""
    print("REACH FALSE POSITIVE ANALYSIS")
    print("=" * 70)

    # Find the worst videos - need to match GT file names to video IDs
    worst_videos = []
    for gt_file in DATA_DIR.glob("*_unified_ground_truth.json"):
        video = gt_file.stem.replace("_unified_ground_truth", "")
        # Check if this is one of the problematic ones
        for bad in ['CNT0311_P2', 'CNT0110_P2', 'CNT0307_P4', 'CNT0312_P2',
                     'CNT0413_P2', 'CNT0309_P1']:
            if bad in video:
                worst_videos.append(video)

    print(f"Worst videos found: {len(worst_videos)}")
    for v in worst_videos:
        print(f"  - {v}")

    all_fp_features = []
    all_tp_features = []
    exhaustive_videos = []
    non_exhaustive_videos = []

    for video in worst_videos:
        print(f"\nProcessing {video}...")
        algo_reaches = load_algo_reaches(video)
        gt_reaches, exhaustive = load_gt_reaches(video)

        if not algo_reaches or not gt_reaches:
            print(f"  Skipping - algo={len(algo_reaches)}, gt={len(gt_reaches)}")
            continue

        result = match_reaches(algo_reaches, gt_reaches)

        if exhaustive:
            print(f"  TP={len(result['tp'])}, FP={len(result['fp'])}, FN={len(result['fn'])} (exhaustive GT)")
            exhaustive_videos.append(video)
        else:
            print(f"  TP={len(result['tp'])}, FP=SKIPPED, FN={len(result['fn'])} (NON-EXHAUSTIVE GT)")
            print(f"    WARNING: Reaches not exhaustive - FP analysis skipped for this video")
            non_exhaustive_videos.append(video)

        feature_data = analyze_fp_features(video, result['fp'], result['tp'], exhaustive)
        if feature_data:
            all_fp_features.extend(feature_data['fp'])
            all_tp_features.extend(feature_data['tp'])
            print_comparison(video, feature_data)

    # Aggregate analysis
    print(f"\n\n{'='*70}")
    print("AGGREGATE ANALYSIS ACROSS ALL WORST VIDEOS")
    print(f"{'='*70}")

    # Show exhaustive status
    print(f"\nVideos with EXHAUSTIVE GT: {len(exhaustive_videos)}")
    for v in exhaustive_videos:
        print(f"  - {v}")
    print(f"\nVideos with NON-EXHAUSTIVE GT: {len(non_exhaustive_videos)}")
    for v in non_exhaustive_videos:
        print(f"  - {v}")

    # Warning if no exhaustive videos
    if not exhaustive_videos:
        print(f"\n{'='*70}")
        print("WARNING: No videos have exhaustive reach ground truth.")
        print("FP analysis requires exhaustive GT (human has determined ALL reaches).")
        print("Without exhaustive GT, unmatched algo reaches may be real reaches")
        print("that the human hasn't determined yet.")
        print("")
        print("Only precision-like metrics (TP matching) are reported below.")
        print("Mark videos as 'exhaustive' in the GT widget to enable FP analysis.")
        print(f"{'='*70}\n")

    if all_fp_features and all_tp_features:
        print_comparison("ALL WORST VIDEOS", {'fp': all_fp_features, 'tp': all_tp_features, 'exhaustive': True})
    elif all_tp_features:
        print(f"\nTP-only analysis available ({len(all_tp_features)} true positive reaches)")
        print("FP comparison not possible without exhaustive GT")

        # TP-only statistics (no FP data available)
        print(f"\n\nTP FEATURE PROFILE (what matched reaches look like):")
        print("-" * 50)

        tp_extents = [f['max_extent'] for f in all_tp_features]
        tp_durations = [f['duration'] for f in all_tp_features]
        tp_velocities = [f['max_x_velocity'] for f in all_tp_features]

        tp_below_0 = sum(1 for e in tp_extents if e < 0) / len(tp_extents) * 100
        tp_short = sum(1 for d in tp_durations if d <= 6) / len(tp_durations) * 100
        tp_low_vel = sum(1 for v in tp_velocities if v < 3) / len(tp_velocities) * 100

        print(f"\n  TP with extent < 0px: {tp_below_0:.0f}%")
        print(f"  TP with duration <= 6 frames: {tp_short:.0f}%")
        print(f"  TP with max_velocity < 3 px/frame: {tp_low_vel:.0f}%")
        print(f"  TP mean duration: {np.mean(tp_durations):.1f} frames")
        print(f"  TP mean extent: {np.mean(tp_extents):.1f} px")

        print(f"\n  NOTE: FP vs TP comparison requires exhaustive reach GT.")
        print(f"  Mark videos as 'exhaustive' in the GT widget to enable comparison.")


if __name__ == "__main__":
    main()
