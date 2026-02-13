"""
Frame-level feature analysis at human-determined reach boundaries.

Treats reach detection as a SEGMENTATION problem:
  - At EVERY frame, what is the DLC state?
  - What distinguishes the exact frame a human calls "reach start"?
  - What distinguishes the exact frame a human calls "reach end"?
  - What's happening in the frames where algo keeps going but human stopped?

For each matched reach (GT + algo), we extract DLC features at:
  1. GT start frame (human says reach begins)
  2. GT end frame (human says reach ends)
  3. Algo end frame (algo says reach ends - often much later)
  4. The "late tail" between GT end and algo end
  5. Pre-reach context (5 frames before GT start)
  6. Post-reach context (5 frames after GT end)

This reveals the DECISION BOUNDARY the human uses.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_0_0")

RH_POINTS = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']
HAND_LIKELIHOOD_THRESHOLD = 0.5
NOSE_ENGAGEMENT_THRESHOLD = 25


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_slit_center(dlc_df, scorer):
    try:
        boxl_x = dlc_df[(scorer, 'BOXL', 'x')].median()
        boxr_x = dlc_df[(scorer, 'BOXR', 'x')].median()
        boxl_y = dlc_df[(scorer, 'BOXL', 'y')].median()
        boxr_y = dlc_df[(scorer, 'BOXR', 'y')].median()
        return (boxl_x + boxr_x) / 2, (boxl_y + boxr_y) / 2
    except KeyError:
        return None, None


def extract_frame_features(dlc_df, scorer, frame, slit_x, slit_y,
                           reach_max_x=None, prev_hand_x=None):
    """Extract all relevant DLC features at a single frame.

    Returns a dict of features, or None if frame is out of range.
    """
    if frame < 0 or frame >= len(dlc_df):
        return None

    row = dlc_df.iloc[frame]

    # Hand: best position and likelihood
    best_hand_x = None
    best_hand_y = None
    best_hand_like = 0
    max_hand_like = 0
    any_hand_visible = False

    for part in RH_POINTS:
        try:
            x = dlc_df[(scorer, part, 'x')].iloc[frame]
            y = dlc_df[(scorer, part, 'y')].iloc[frame]
            like = dlc_df[(scorer, part, 'likelihood')].iloc[frame]
        except (KeyError, IndexError):
            continue
        max_hand_like = max(max_hand_like, like)
        if like >= HAND_LIKELIHOOD_THRESHOLD:
            any_hand_visible = True
            if like > best_hand_like:
                best_hand_x = x
                best_hand_y = y
                best_hand_like = like

    # Hand offset from slit
    hand_offset = (best_hand_x - slit_x) if best_hand_x is not None else None

    # Hand velocity (requires previous position)
    hand_velocity = None
    if best_hand_x is not None and prev_hand_x is not None:
        hand_velocity = best_hand_x - prev_hand_x  # positive = extending

    # Retraction from max extension
    retraction_px = None
    retraction_pct = None
    if best_hand_x is not None and reach_max_x is not None:
        retraction_px = reach_max_x - best_hand_x
        extension = reach_max_x - slit_x
        if extension > 0:
            retraction_pct = retraction_px / extension * 100

    # Nose
    try:
        nose_x = dlc_df[(scorer, 'Nose', 'x')].iloc[frame]
        nose_y = dlc_df[(scorer, 'Nose', 'y')].iloc[frame]
        nose_like = dlc_df[(scorer, 'Nose', 'likelihood')].iloc[frame]
    except (KeyError, IndexError):
        nose_x, nose_y, nose_like = np.nan, np.nan, 0

    nose_dist = None
    nose_engaged = False
    if nose_like >= 0.3 and not np.isnan(nose_x):
        nose_dist = np.sqrt((nose_x - slit_x)**2 + (nose_y - slit_y)**2)
        nose_engaged = nose_dist < NOSE_ENGAGEMENT_THRESHOLD

    return {
        'hand_x': best_hand_x,
        'hand_y': best_hand_y,
        'hand_like': best_hand_like,
        'max_hand_like': max_hand_like,
        'hand_visible': any_hand_visible,
        'hand_offset': hand_offset,
        'hand_velocity': hand_velocity,
        'retraction_px': retraction_px,
        'retraction_pct': retraction_pct,
        'nose_x': nose_x,
        'nose_y': nose_y,
        'nose_like': nose_like,
        'nose_dist': nose_dist,
        'nose_engaged': nose_engaged,
    }


def match_reaches(gt_reaches, algo_reaches, max_dist=30):
    """Match GT reaches to algo reaches by start frame proximity (1:1 greedy)."""
    candidates = []
    for gi, gr in enumerate(gt_reaches):
        gt_start = gr['start_frame']
        for ai, ar in enumerate(algo_reaches):
            a_start = ar.get('start_frame', 0)
            dist = abs(gt_start - a_start)
            if dist <= max_dist:
                candidates.append((dist, gi, ai))

    candidates.sort()
    gt_used = set()
    algo_used = set()
    matches = []

    for dist, gi, ai in candidates:
        if gi not in gt_used and ai not in algo_used:
            gt_used.add(gi)
            algo_used.add(ai)
            matches.append((gi, ai, dist))

    return matches


def main():
    print("FRAME-LEVEL FEATURE ANALYSIS AT REACH BOUNDARIES")
    print("=" * 70)
    print()
    print("What does the DLC look like at the EXACT frames where")
    print("humans say 'reach starts here' and 'reach ends here'?")
    print()

    # Feature collection bins
    features_at = {
        'gt_start': [],         # Frame where human says reach begins
        'gt_end': [],           # Frame where human says reach ends
        'algo_end': [],         # Frame where algo says reach ends (when late)
        'mid_reach': [],        # Middle of the reach
        'pre_reach': [],        # 3 frames before GT start
        'post_gt_end': [],      # 3 frames after GT end (but still in algo reach)
        'late_tail_early': [],  # First 3 frames of late tail (GT end+1 to GT end+3)
        'late_tail_late': [],   # Last 3 frames of late tail (algo end-2 to algo end)
    }

    # Track reach trajectories for pattern analysis
    reach_trajectories = []

    n_matched = 0
    n_start_ok_end_late = 0

    for gt_file in sorted(DATA_DIR.glob("*_unified_ground_truth.json")):
        gt = load_json(gt_file)
        if not gt:
            continue

        video = gt['video_name']
        gt_reaches = [r for r in gt.get('reaches', {}).get('reaches', [])
                      if r.get('start_determined', False) and r.get('end_determined', False)
                      and not r.get('exclude_from_analysis', False)]
        if not gt_reaches:
            continue

        algo_file = ALGO_DIR / f"{video}_reaches.json"
        algo_data = load_json(algo_file)
        if not algo_data:
            continue

        algo_reaches = []
        for seg in algo_data.get('segments', []):
            for r in seg.get('reaches', []):
                algo_reaches.append(r)
        if not algo_reaches:
            continue

        # Load DLC
        dlc_files = list(DATA_DIR.glob(f"{video}DLC*.h5"))
        if not dlc_files:
            dlc_files = list(DATA_DIR.glob(f"*{video.split('_', 1)[1]}*DLC*.h5"))
        if not dlc_files:
            continue

        dlc_df = pd.read_hdf(dlc_files[0])
        scorer = dlc_df.columns.get_level_values(0)[0]
        slit_x, slit_y = get_slit_center(dlc_df, scorer)
        if slit_x is None:
            continue

        matches = match_reaches(gt_reaches, algo_reaches)

        for gi, ai, start_dist in matches:
            gr = gt_reaches[gi]
            ar = algo_reaches[ai]

            gt_start = gr['start_frame']
            gt_end = gr['end_frame']
            algo_start = ar.get('start_frame', 0)
            algo_end = ar.get('end_frame', 0)
            start_offset = algo_start - gt_start
            end_offset = algo_end - gt_end

            n_matched += 1

            # Compute max extension during the GT reach
            reach_max_x = 0
            for f in range(gt_start, min(gt_end + 1, len(dlc_df))):
                hx = None
                best_l = 0
                for part in RH_POINTS:
                    try:
                        x = dlc_df[(scorer, part, 'x')].iloc[f]
                        l = dlc_df[(scorer, part, 'likelihood')].iloc[f]
                        if l > best_l and l >= HAND_LIKELIHOOD_THRESHOLD:
                            hx = x
                            best_l = l
                    except (KeyError, IndexError):
                        continue
                if hx is not None and hx > reach_max_x:
                    reach_max_x = hx

            # Get previous hand position for velocity at each key frame
            def get_prev_hand_x(frame):
                if frame <= 0 or frame >= len(dlc_df):
                    return None
                best_x = None
                best_l = 0
                for part in RH_POINTS:
                    try:
                        x = dlc_df[(scorer, part, 'x')].iloc[frame - 1]
                        l = dlc_df[(scorer, part, 'likelihood')].iloc[frame - 1]
                        if l > best_l and l >= HAND_LIKELIHOOD_THRESHOLD:
                            best_x = x
                            best_l = l
                    except (KeyError, IndexError):
                        continue
                return best_x

            # Extract features at key frames
            # 1. GT start
            feat = extract_frame_features(
                dlc_df, scorer, gt_start, slit_x, slit_y,
                reach_max_x=None, prev_hand_x=get_prev_hand_x(gt_start))
            if feat:
                features_at['gt_start'].append(feat)

            # 2. GT end
            feat = extract_frame_features(
                dlc_df, scorer, gt_end, slit_x, slit_y,
                reach_max_x=reach_max_x, prev_hand_x=get_prev_hand_x(gt_end))
            if feat:
                features_at['gt_end'].append(feat)

            # 3. Mid-reach (middle frame)
            mid = (gt_start + gt_end) // 2
            feat = extract_frame_features(
                dlc_df, scorer, mid, slit_x, slit_y,
                reach_max_x=reach_max_x, prev_hand_x=get_prev_hand_x(mid))
            if feat:
                features_at['mid_reach'].append(feat)

            # 4. Pre-reach context (3 frames before)
            for offset in range(-3, 0):
                feat = extract_frame_features(
                    dlc_df, scorer, gt_start + offset, slit_x, slit_y,
                    reach_max_x=None, prev_hand_x=get_prev_hand_x(gt_start + offset))
                if feat:
                    features_at['pre_reach'].append(feat)

            # 5. Focus on START_OK_END_LATE cases
            if abs(start_offset) <= 2 and end_offset > 2:
                n_start_ok_end_late += 1

                # Algo end frame
                feat = extract_frame_features(
                    dlc_df, scorer, algo_end, slit_x, slit_y,
                    reach_max_x=reach_max_x, prev_hand_x=get_prev_hand_x(algo_end))
                if feat:
                    features_at['algo_end'].append(feat)

                # Post GT-end (3 frames after GT end, in the late tail)
                for offset in range(1, min(4, end_offset + 1)):
                    f = gt_end + offset
                    feat = extract_frame_features(
                        dlc_df, scorer, f, slit_x, slit_y,
                        reach_max_x=reach_max_x, prev_hand_x=get_prev_hand_x(f))
                    if feat:
                        features_at['post_gt_end'].append(feat)
                        features_at['late_tail_early'].append(feat)

                # Late tail end (last 3 frames before algo end)
                for offset in range(-2, 1):
                    f = algo_end + offset
                    if f > gt_end:  # Only if in the late tail
                        feat = extract_frame_features(
                            dlc_df, scorer, f, slit_x, slit_y,
                            reach_max_x=reach_max_x, prev_hand_x=get_prev_hand_x(f))
                        if feat:
                            features_at['late_tail_late'].append(feat)

                # Full trajectory for this reach (for pattern viz)
                trajectory = {
                    'video': video,
                    'gt_start': gt_start, 'gt_end': gt_end,
                    'algo_start': algo_start, 'algo_end': algo_end,
                    'end_offset': end_offset,
                    'frames': [],
                }
                # Extract hand x for every frame from gt_start-3 to algo_end+3
                prev_hx = None
                for f in range(max(0, gt_start - 3), min(algo_end + 4, len(dlc_df))):
                    feat = extract_frame_features(
                        dlc_df, scorer, f, slit_x, slit_y,
                        reach_max_x=reach_max_x, prev_hand_x=prev_hx)
                    if feat:
                        trajectory['frames'].append({
                            'frame': f,
                            'hand_offset': feat['hand_offset'],
                            'hand_visible': feat['hand_visible'],
                            'hand_velocity': feat['hand_velocity'],
                            'retraction_pct': feat['retraction_pct'],
                            'nose_engaged': feat['nose_engaged'],
                        })
                        if feat['hand_x'] is not None:
                            prev_hx = feat['hand_x']

                if len(trajectory['frames']) > 3:
                    reach_trajectories.append(trajectory)

    # ================================================================
    # REPORT
    # ================================================================

    print(f"Total matched reaches analyzed: {n_matched}")
    print(f"Start-OK End-late reaches:      {n_start_ok_end_late}")
    print()

    # ----------------------------------------------------------------
    # SECTION 1: WHAT DOES A HUMAN SEE AT REACH START?
    # ----------------------------------------------------------------
    print("=" * 70)
    print("REACH START: What the human sees at the first frame of a reach")
    print("=" * 70)
    print()

    print_feature_comparison(
        'Pre-reach (3 frames before)', features_at['pre_reach'],
        'GT start frame', features_at['gt_start'],
        'Mid-reach', features_at['mid_reach'],
    )

    # ----------------------------------------------------------------
    # SECTION 2: WHAT DOES A HUMAN SEE AT REACH END?
    # ----------------------------------------------------------------
    print()
    print("=" * 70)
    print("REACH END: What the human sees at the last frame of a reach")
    print("  vs what the algo sees when IT ends the reach (later)")
    print("=" * 70)
    print()

    print_feature_comparison(
        'Mid-reach', features_at['mid_reach'],
        'GT end frame (human stops)', features_at['gt_end'],
        'Post GT-end (human says done)', features_at['post_gt_end'],
        extra_label='Algo end frame (algo stops)',
        extra_features=features_at['algo_end'],
    )

    # ----------------------------------------------------------------
    # SECTION 3: THE LATE TAIL - What's happening when algo is wrong?
    # ----------------------------------------------------------------
    print()
    print("=" * 70)
    print("THE LATE TAIL: Frames between 'human says done' and 'algo says done'")
    print("=" * 70)
    print()

    if features_at['late_tail_early'] and features_at['late_tail_late']:
        print("  These are frames the algo includes but the human doesn't.")
        print("  What does the hand look like here?")
        print()
        print_feature_comparison(
            'GT end frame', features_at['gt_end'],
            'Early late tail (+1 to +3)', features_at['late_tail_early'],
            'Late late tail (algo end-2 to end)', features_at['late_tail_late'],
        )

    # ----------------------------------------------------------------
    # SECTION 4: TRAJECTORY PATTERNS
    # ----------------------------------------------------------------
    print()
    print("=" * 70)
    print("TRAJECTORY PATTERNS: Hand position through entire reach + tail")
    print("=" * 70)
    print()

    if reach_trajectories:
        analyze_trajectories(reach_trajectories)

    # ----------------------------------------------------------------
    # SECTION 5: DECISION BOUNDARY SUMMARY
    # ----------------------------------------------------------------
    print()
    print("=" * 70)
    print("DECISION RULES: What distinguishes human's end frame from algo's")
    print("=" * 70)
    print()

    summarize_decision_rules(features_at)


def print_feature_comparison(label_a, feats_a, label_b, feats_b, label_c, feats_c,
                              extra_label=None, extra_features=None):
    """Print side-by-side feature comparison for multiple frame types."""

    groups = [(label_a, feats_a), (label_b, feats_b), (label_c, feats_c)]
    if extra_label and extra_features:
        groups.append((extra_label, extra_features))

    # Header
    col_width = 22
    header = f"  {'Feature':<25}"
    for label, _ in groups:
        # Truncate label to fit
        short = label[:col_width]
        header += f"{short:>{col_width}}"
    print(header)
    print(f"  {'-' * (25 + col_width * len(groups))}")

    def stat(feats, key, fmt='.1f'):
        vals = [f[key] for f in feats if f.get(key) is not None and not _is_nan(f.get(key))]
        if not vals:
            return 'n/a'
        return f"{np.median(vals):{fmt}}"

    def pct(feats, key):
        vals = [f[key] for f in feats if f.get(key) is not None]
        if not vals:
            return 'n/a'
        return f"{sum(1 for v in vals if v) / len(vals) * 100:.0f}%"

    features_to_show = [
        ('hand_visible', 'Hand visible', pct),
        ('max_hand_like', 'Hand likelihood (med)', stat),
        ('hand_offset', 'Hand offset px (med)', stat),
        ('hand_velocity', 'Hand velocity px/f (med)', lambda f, k: stat(f, k, '+.2f')),
        ('retraction_pct', 'Retraction % (med)', stat),
        ('retraction_px', 'Retraction px (med)', stat),
        ('nose_engaged', 'Nose engaged', pct),
        ('nose_dist', 'Nose dist px (med)', stat),
    ]

    for key, label, fn in features_to_show:
        row = f"  {label:<25}"
        for _, feats in groups:
            if feats:
                val = fn(feats, key)
                row += f"{val:>{col_width}}"
            else:
                row += f"{'(no data)':>{col_width}}"
        print(row)

    # Count
    row = f"  {'n (frames)' :<25}"
    for _, feats in groups:
        row += f"{len(feats):>{col_width}}"
    print(row)


def _is_nan(val):
    try:
        return np.isnan(val)
    except (TypeError, ValueError):
        return False


def analyze_trajectories(trajectories):
    """Analyze hand position trajectory patterns through reach + late tail."""

    print(f"  Analyzing {len(trajectories)} reach trajectories (start-OK, end-late)")
    print()

    # For each trajectory, compute: at what point does hand start retracting?
    # Relative to GT end vs algo end
    retraction_at_gt_end = []
    retraction_at_algo_end = []
    hand_visible_at_gt_end = []
    hand_visible_in_tail = []

    # Classify trajectory shapes in the late tail
    tail_patterns = {
        'hand_retracting': 0,     # Hand moving back toward slit
        'hand_extended': 0,       # Hand still extended but not moving
        'hand_disappeared': 0,    # Hand invisible
        'hand_re_extending': 0,   # Hand extending again (new reach?)
        'hand_oscillating': 0,    # Hand bouncing
    }

    for traj in trajectories:
        gt_end = traj['gt_end']
        algo_end = traj['algo_end']
        frames = traj['frames']

        if not frames:
            continue

        # Find features at GT end
        gt_end_feat = None
        for f in frames:
            if f['frame'] == gt_end:
                gt_end_feat = f
                break

        if gt_end_feat:
            if gt_end_feat['retraction_pct'] is not None:
                retraction_at_gt_end.append(gt_end_feat['retraction_pct'])
            hand_visible_at_gt_end.append(gt_end_feat['hand_visible'])

        # Analyze the late tail (GT end to algo end)
        tail_frames = [f for f in frames if gt_end < f['frame'] <= algo_end]
        if tail_frames:
            # Hand visibility in tail
            vis_count = sum(1 for f in tail_frames if f['hand_visible'])
            vis_pct = vis_count / len(tail_frames)
            hand_visible_in_tail.append(vis_pct)

            # Classify tail pattern by velocity and visibility
            velocities = [f['hand_velocity'] for f in tail_frames
                          if f['hand_velocity'] is not None]
            offsets = [f['hand_offset'] for f in tail_frames
                       if f['hand_offset'] is not None]

            if vis_pct < 0.3:
                tail_patterns['hand_disappeared'] += 1
            elif velocities:
                mean_vel = np.mean(velocities)
                vel_changes = sum(1 for i in range(1, len(velocities))
                                  if (velocities[i] > 0) != (velocities[i-1] > 0))
                if vel_changes > len(velocities) * 0.4:
                    tail_patterns['hand_oscillating'] += 1
                elif mean_vel < -0.5:
                    tail_patterns['hand_retracting'] += 1
                elif mean_vel > 0.5:
                    tail_patterns['hand_re_extending'] += 1
                else:
                    tail_patterns['hand_extended'] += 1
            else:
                tail_patterns['hand_extended'] += 1

    # Report
    if retraction_at_gt_end:
        print(f"  Retraction at GT end frame:")
        print(f"    Mean: {np.mean(retraction_at_gt_end):.1f}%")
        print(f"    Median: {np.median(retraction_at_gt_end):.1f}%")
        for pct in [10, 20, 30, 50, 70, 90]:
            count = sum(1 for r in retraction_at_gt_end if r >= pct)
            print(f"    >= {pct}%: {count}/{len(retraction_at_gt_end)} "
                  f"({count/len(retraction_at_gt_end)*100:.0f}%)")

    if hand_visible_at_gt_end:
        vis = sum(1 for v in hand_visible_at_gt_end if v)
        print(f"\n  Hand visible at GT end: {vis}/{len(hand_visible_at_gt_end)} "
              f"({vis/len(hand_visible_at_gt_end)*100:.1f}%)")

    if hand_visible_in_tail:
        print(f"\n  Hand visibility in late tail (GT end to algo end):")
        print(f"    Mean % visible: {np.mean(hand_visible_in_tail)*100:.1f}%")
        mostly_vis = sum(1 for v in hand_visible_in_tail if v > 0.5)
        mostly_invis = sum(1 for v in hand_visible_in_tail if v <= 0.5)
        print(f"    Mostly visible (>50%): {mostly_vis}")
        print(f"    Mostly invisible (<=50%): {mostly_invis}")

    print(f"\n  Late tail trajectory patterns:")
    total_tails = sum(tail_patterns.values())
    for pattern, count in sorted(tail_patterns.items(), key=lambda x: -x[1]):
        print(f"    {pattern:<25} {count:>4} ({count/max(total_tails,1)*100:.1f}%)")


def summarize_decision_rules(features_at):
    """Synthesize what distinguishes the human's end frame from surrounding frames."""

    gt_end = features_at['gt_end']
    post = features_at['post_gt_end']
    mid = features_at['mid_reach']
    algo_end = features_at['algo_end']

    if not gt_end:
        print("  No GT end frame data to analyze.")
        return

    def median_of(feats, key):
        vals = [f[key] for f in feats if f.get(key) is not None and not _is_nan(f.get(key))]
        return np.median(vals) if vals else None

    def pct_true(feats, key):
        vals = [f[key] for f in feats if f.get(key) is not None]
        return sum(1 for v in vals if v) / max(len(vals), 1) * 100

    print("  WHAT THE HUMAN SEES AT REACH END (vs mid-reach):\n")

    # Key contrasts
    contrasts = []

    mid_vis = pct_true(mid, 'hand_visible')
    end_vis = pct_true(gt_end, 'hand_visible')
    post_vis = pct_true(post, 'hand_visible') if post else None
    contrasts.append(('Hand visible',
                      f"Mid: {mid_vis:.0f}%",
                      f"GT end: {end_vis:.0f}%",
                      f"Post: {post_vis:.0f}%" if post_vis is not None else "Post: n/a"))

    mid_off = median_of(mid, 'hand_offset')
    end_off = median_of(gt_end, 'hand_offset')
    post_off = median_of(post, 'hand_offset')
    contrasts.append(('Hand offset (px)',
                      f"Mid: {mid_off:.1f}" if mid_off else "Mid: n/a",
                      f"GT end: {end_off:.1f}" if end_off else "GT end: n/a",
                      f"Post: {post_off:.1f}" if post_off else "Post: n/a"))

    mid_vel = median_of(mid, 'hand_velocity')
    end_vel = median_of(gt_end, 'hand_velocity')
    post_vel = median_of(post, 'hand_velocity')
    contrasts.append(('Hand velocity (px/f)',
                      f"Mid: {mid_vel:+.2f}" if mid_vel else "Mid: n/a",
                      f"GT end: {end_vel:+.2f}" if end_vel else "GT end: n/a",
                      f"Post: {post_vel:+.2f}" if post_vel else "Post: n/a"))

    mid_ret = median_of(mid, 'retraction_pct')
    end_ret = median_of(gt_end, 'retraction_pct')
    post_ret = median_of(post, 'retraction_pct')
    contrasts.append(('Retraction %',
                      f"Mid: {mid_ret:.0f}%" if mid_ret is not None else "Mid: n/a",
                      f"GT end: {end_ret:.0f}%" if end_ret is not None else "GT end: n/a",
                      f"Post: {post_ret:.0f}%" if post_ret is not None else "Post: n/a"))

    for label, a, b, c in contrasts:
        print(f"    {label:<25} {a:<20} {b:<20} {c}")

    # The key question: what CHANGED between mid-reach and end?
    print(f"\n\n  KEY TRANSITIONS (what changes from mid-reach to GT end):\n")

    if mid_vis > end_vis + 10:
        print(f"    - Hand DISAPPEARS: visible drops from {mid_vis:.0f}% to {end_vis:.0f}%")
        print(f"      This is the strongest signal for reach end.")

    if mid_off is not None and end_off is not None and mid_off > end_off + 2:
        print(f"    - Hand RETRACTS: offset drops from {mid_off:.1f}px to {end_off:.1f}px")
        print(f"      The hand moves back toward the slit.")

    if mid_vel is not None and end_vel is not None:
        if mid_vel > 0 and end_vel < 0:
            print(f"    - VELOCITY REVERSAL: from +{mid_vel:.2f} to {end_vel:.2f} px/frame")
            print(f"      The hand switches from extending to retracting.")
        elif mid_vel > 0 and end_vel >= 0 and end_vel < mid_vel:
            print(f"    - DECELERATION: velocity drops from {mid_vel:.2f} to {end_vel:.2f}")
            print(f"      The hand slows down before the human calls end.")

    if end_ret is not None and end_ret > 20:
        print(f"    - SIGNIFICANT RETRACTION at end: {end_ret:.0f}% of max extension")

    # The algo's failure
    if algo_end:
        algo_vis = pct_true(algo_end, 'hand_visible')
        algo_off = median_of(algo_end, 'hand_offset')
        algo_ret = median_of(algo_end, 'retraction_pct')

        print(f"\n\n  WHY THE ALGO ENDS LATER:\n")
        print(f"    At GT end:   hand visible={end_vis:.0f}%, "
              f"offset={end_off:.1f}px, "
              f"retract={end_ret:.0f}%" if end_ret is not None else "")
        print(f"    At algo end: hand visible={algo_vis:.0f}%, "
              f"offset={algo_off:.1f}px, "
              f"retract={algo_ret:.0f}%" if algo_ret is not None else "")
        print()
        print(f"    The human ends the reach when the hand starts retracting.")
        print(f"    The algo waits until a STRONGER signal (hand disappears or")
        print(f"    retracts past its 40% threshold).")
        print(f"    The gap between '{end_ret:.0f}% retraction' and '40% threshold'")
        print(f"    is where all the late-end error comes from." if end_ret is not None else "")


if __name__ == "__main__":
    main()
