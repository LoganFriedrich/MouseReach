"""
Learn from Algorithm Mistakes - WITHOUT Deep Learning

Analyze DLC features around cases where the algorithm got it wrong to understand
WHY it fails. This can directly inform rule improvements.

Key questions:
1. When algo calls "retrieved" as "displaced", what DLC pattern exists?
2. When algo calls "retrieved" as "untouched", what's different?
3. What distinguishes successful retrieval in DLC data?
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_0_0")


def load_dlc_features(video: str, frame: int, window: int = 30) -> dict:
    """Extract key DLC features around a frame."""
    dlc_files = list(DATA_DIR.glob(f"{video}DLC*.h5"))
    if not dlc_files:
        return None

    df = pd.read_hdf(dlc_files[0])

    start = max(0, frame - window)
    end = min(len(df) - 1, frame + window)

    if start >= end:
        return None

    features = {}

    try:
        # Get scorer name (first level of multi-index columns)
        scorer = df.columns.get_level_values(0)[0]

        # Hand position (averaged across markers)
        hand_parts = ['RightHand', 'RHLeft', 'RHRight', 'RHOut']
        hand_x, hand_y, hand_like = [], [], []
        for part in hand_parts:
            try:
                hand_x.append(df[(scorer, part, 'x')].values[start:end+1])
                hand_y.append(df[(scorer, part, 'y')].values[start:end+1])
                hand_like.append(df[(scorer, part, 'likelihood')].values[start:end+1])
            except KeyError:
                continue

        if not hand_x:
            return None

        # Simple mean (could weight by likelihood)
        hand_x = np.array(hand_x)
        hand_y = np.array(hand_y)
        hand_like = np.array(hand_like)

        features['hand_x'] = np.mean(hand_x, axis=0)
        features['hand_y'] = np.mean(hand_y, axis=0)
        features['hand_likelihood'] = np.mean(hand_like, axis=0)

        # Pellet position
        features['pellet_x'] = df[(scorer, 'Pellet', 'x')].values[start:end+1]
        features['pellet_y'] = df[(scorer, 'Pellet', 'y')].values[start:end+1]
        features['pellet_likelihood'] = df[(scorer, 'Pellet', 'likelihood')].values[start:end+1]

        # Nose position (proxy for head/attention)
        features['nose_x'] = df[(scorer, 'Nose', 'x')].values[start:end+1]
        features['nose_y'] = df[(scorer, 'Nose', 'y')].values[start:end+1]

        # Derived features
        # Hand-pellet distance
        features['hand_pellet_dist'] = np.sqrt(
            (features['hand_x'] - features['pellet_x'])**2 +
            (features['hand_y'] - features['pellet_y'])**2
        )

        # Hand velocity (dx/dt)
        features['hand_velocity'] = np.sqrt(
            np.diff(features['hand_x'])**2 + np.diff(features['hand_y'])**2
        )

        # Pellet movement (key for retrieval vs displacement)
        features['pellet_movement'] = np.sqrt(
            np.diff(features['pellet_x'])**2 + np.diff(features['pellet_y'])**2
        )

        # Summary stats
        mid = window  # Index of target frame in window

        features['min_hand_pellet_dist'] = np.min(features['hand_pellet_dist'])
        features['hand_pellet_dist_at_frame'] = features['hand_pellet_dist'][mid] if mid < len(features['hand_pellet_dist']) else np.nan
        features['max_hand_velocity'] = np.max(features['hand_velocity']) if len(features['hand_velocity']) > 0 else 0
        features['total_pellet_movement'] = np.sum(features['pellet_movement']) if len(features['pellet_movement']) > 0 else 0
        features['pellet_visible'] = np.mean(features['pellet_likelihood']) > 0.5

        # Direction of pellet movement (towards mouse or away?)
        if len(features['pellet_x']) > 1:
            pellet_start = (features['pellet_x'][0], features['pellet_y'][0])
            pellet_end = (features['pellet_x'][-1], features['pellet_y'][-1])
            nose_pos = (np.mean(features['nose_x']), np.mean(features['nose_y']))

            dist_start = np.sqrt((pellet_start[0] - nose_pos[0])**2 + (pellet_start[1] - nose_pos[1])**2)
            dist_end = np.sqrt((pellet_end[0] - nose_pos[0])**2 + (pellet_end[1] - nose_pos[1])**2)

            features['pellet_moved_toward_mouse'] = dist_end < dist_start

    except (KeyError, IndexError) as e:
        return None

    return features


def analyze_outcome_mistakes():
    """Analyze what distinguishes correct vs incorrect outcome calls."""
    print("\n" + "="*70)
    print("LEARNING FROM OUTCOME MISTAKES")
    print("="*70)

    # Collect feature stats for each outcome pattern
    stats = defaultdict(lambda: defaultdict(list))

    for gt_file in DATA_DIR.glob("*_unified_ground_truth.json"):
        video = gt_file.stem.replace("_unified_ground_truth", "")

        with open(gt_file) as f:
            gt = json.load(f)

        algo_file = ALGO_DIR / f"{video}_pellet_outcomes.json"
        if not algo_file.exists():
            continue
        with open(algo_file) as f:
            algo = json.load(f)

        # Get boundaries for frame estimation
        boundaries = [b['frame'] for b in gt.get('segmentation', {}).get('boundaries', [])]

        gt_segs = {s['segment_num']: s for s in gt.get('outcomes', {}).get('segments', [])
                   if s.get('determined', False)}
        algo_segs = {s['segment_num']: s for s in algo.get('segments', [])}

        for seg_num in gt_segs:
            gt_seg = gt_segs[seg_num]
            algo_seg = algo_segs.get(seg_num)
            if algo_seg is None:
                continue

            gt_outcome = gt_seg['outcome']
            algo_outcome = algo_seg['outcome']

            # Get frame for feature extraction
            frame = gt_seg.get('interaction_frame') or algo_seg.get('interaction_frame')
            if frame is None:
                # Use segment midpoint
                if 1 <= seg_num <= len(boundaries):
                    start = boundaries[seg_num - 1] if seg_num > 1 else 0
                    end = boundaries[seg_num] if seg_num < len(boundaries) else start + 1000
                    frame = (start + end) // 2
                else:
                    continue

            features = load_dlc_features(video, frame)
            if features is None:
                continue

            # Categorize by mistake pattern
            if gt_outcome == algo_outcome:
                key = f"CORRECT: {gt_outcome}"
            else:
                key = f"MISTAKE: {algo_outcome} -> {gt_outcome}"

            # Store key features
            stats[key]['min_hand_pellet_dist'].append(features['min_hand_pellet_dist'])
            stats[key]['hand_pellet_dist_at_frame'].append(features['hand_pellet_dist_at_frame'])
            stats[key]['max_hand_velocity'].append(features['max_hand_velocity'])
            stats[key]['total_pellet_movement'].append(features['total_pellet_movement'])
            stats[key]['pellet_visible'].append(features['pellet_visible'])
            stats[key]['pellet_moved_toward_mouse'].append(features.get('pellet_moved_toward_mouse', False))

    # Print analysis
    print("\nFeature comparison by outcome pattern:")
    print("-" * 70)

    for key in sorted(stats.keys()):
        n = len(stats[key]['min_hand_pellet_dist'])
        print(f"\n{key} (n={n}):")

        for feature in ['min_hand_pellet_dist', 'max_hand_velocity', 'total_pellet_movement']:
            values = [v for v in stats[key][feature] if not np.isnan(v)]
            if values:
                print(f"  {feature}: mean={np.mean(values):.1f}, std={np.std(values):.1f}")

        # Pellet visibility
        visible_pct = 100 * np.mean(stats[key]['pellet_visible'])
        print(f"  pellet_visible: {visible_pct:.0f}%")

        # Pellet movement direction
        toward_pct = 100 * np.mean(stats[key]['pellet_moved_toward_mouse'])
        print(f"  pellet_moved_toward_mouse: {toward_pct:.0f}%")

    return stats


def generate_rule_improvements():
    """Based on patterns, suggest rule improvements."""
    print("\n" + "="*70)
    print("SUGGESTED RULE IMPROVEMENTS")
    print("="*70)

    print("""
Based on the failure analysis, here are potential rule improvements:

1. RETRIEVED vs DISPLACED DISAMBIGUATION
   Current issue: Algorithm calls "retrieved" as "displaced_sa"

   Potential fix:
   - Check if pellet moves TOWARD the mouse (nose/mouth region)
   - Retrieved: pellet disappears or moves toward mouth
   - Displaced: pellet moves laterally or away

   Rule: if pellet_moved_toward_mouse AND pellet_visibility_drops:
         outcome = "retrieved"

2. RETRIEVED vs UNTOUCHED DISAMBIGUATION
   Current issue: Algorithm calls "retrieved" as "untouched"

   Potential fix:
   - Check minimum hand-pellet distance during segment
   - If hand got very close (<X pixels) but pellet is now gone
   - That's likely a retrieval the algorithm missed

   Rule: if min_hand_pellet_dist < THRESHOLD AND pellet_not_visible_at_end:
         outcome = "retrieved"

3. BOUNDARY TIMING
   Current issue: Algorithm boundaries are 2 frames early on average

   Potential fix:
   - Shift algorithm output by +2 frames
   - Or adjust detection threshold to be slightly later

4. REACH END DETECTION
   Current issue: End frames need human correction more than start frames

   Potential fix:
   - Use hand velocity to detect reach completion
   - End = when hand velocity drops below threshold after apex
   - Or when hand starts moving away from pellet
""")


def main():
    stats = analyze_outcome_mistakes()
    generate_rule_improvements()


if __name__ == "__main__":
    main()
