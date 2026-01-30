#!/usr/bin/env python3
"""
Ground Truth Pattern Analysis for Reach Detection

This script analyzes human-labeled ground truth reaches to discover
patterns that can be used to derive reach detection rules.

Questions to answer:
1. Where is nose (relative to BOXL/BOXR) at reach start/end?
2. Which hand points are visible at reach start/end?
3. What happens to hand points at reach end?
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ReachAnalysis:
    """Analysis results for a single reach."""
    reach_id: int
    start_frame: int
    end_frame: int
    duration: int

    # Nose position at start/end (relative to slit)
    nose_x_start: float
    nose_y_start: float
    nose_x_end: float
    nose_y_end: float
    nose_dist_from_slit_start: float
    nose_dist_from_slit_end: float

    # Hand visibility at start
    hand_points_visible_start: List[str]
    hand_likelihood_start: Dict[str, float]

    # Hand visibility at end
    hand_points_visible_end: List[str]
    hand_likelihood_end: Dict[str, float]

    # Hand position change
    any_hand_visible_start: bool
    any_hand_visible_end: bool


def load_dlc_data(h5_path: Path) -> Tuple[pd.DataFrame, str]:
    """Load DLC tracking data from h5 file.

    Returns:
        Tuple of (DataFrame with flattened columns, scorer name)
    """
    df = pd.read_hdf(h5_path)

    # Get scorer name from first column
    scorer = df.columns[0][0]

    # Flatten multi-index columns: (scorer, bodypart, coord) -> bodypart_coord
    new_cols = []
    for col in df.columns:
        # col is tuple like ('DLC_resnet50...', 'Nose', 'x')
        bodypart = col[1]
        coord = col[2]
        new_cols.append(f'{bodypart}_{coord}')

    df.columns = new_cols
    return df, scorer


def load_ground_truth(json_path: Path) -> Dict:
    """Load ground truth reaches."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_slit_center(dlc_df: pd.DataFrame, frame: int) -> Tuple[float, float]:
    """Get the center of the slit (midpoint of BOXL and BOXR)."""
    boxl_x = dlc_df.iloc[frame].get('BOXL_x', np.nan)
    boxl_y = dlc_df.iloc[frame].get('BOXL_y', np.nan)
    boxr_x = dlc_df.iloc[frame].get('BOXR_x', np.nan)
    boxr_y = dlc_df.iloc[frame].get('BOXR_y', np.nan)

    center_x = (boxl_x + boxr_x) / 2
    center_y = (boxl_y + boxr_y) / 2

    return center_x, center_y


def get_nose_position(dlc_df: pd.DataFrame, frame: int) -> Tuple[float, float, float]:
    """Get nose position and likelihood at a frame."""
    nose_x = dlc_df.iloc[frame].get('Nose_x', np.nan)
    nose_y = dlc_df.iloc[frame].get('Nose_y', np.nan)
    nose_l = dlc_df.iloc[frame].get('Nose_likelihood', 0)
    return nose_x, nose_y, nose_l


def get_hand_visibility(dlc_df: pd.DataFrame, frame: int, threshold: float = 0.3) -> Dict:
    """Get hand point visibility at a frame."""
    hand_points = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']
    result = {}

    for point in hand_points:
        likelihood = dlc_df.iloc[frame].get(f'{point}_likelihood', 0)
        x = dlc_df.iloc[frame].get(f'{point}_x', np.nan)
        y = dlc_df.iloc[frame].get(f'{point}_y', np.nan)
        result[point] = {
            'likelihood': likelihood,
            'visible': likelihood >= threshold,
            'x': x,
            'y': y
        }

    return result


def analyze_reach(dlc_df: pd.DataFrame, reach: Dict) -> ReachAnalysis:
    """Analyze a single reach."""
    start_frame = reach['start_frame']
    end_frame = reach['end_frame']

    # Clamp to valid frame range
    start_frame = max(0, min(start_frame, len(dlc_df) - 1))
    end_frame = max(0, min(end_frame, len(dlc_df) - 1))

    # Get slit center (average over reach for stability)
    slit_centers = []
    for f in range(start_frame, min(end_frame + 1, len(dlc_df))):
        cx, cy = get_slit_center(dlc_df, f)
        if not np.isnan(cx):
            slit_centers.append((cx, cy))

    if slit_centers:
        slit_x = np.mean([c[0] for c in slit_centers])
        slit_y = np.mean([c[1] for c in slit_centers])
    else:
        slit_x, slit_y = np.nan, np.nan

    # Nose at start
    nose_x_start, nose_y_start, nose_l_start = get_nose_position(dlc_df, start_frame)
    nose_dist_start = np.sqrt((nose_x_start - slit_x)**2 + (nose_y_start - slit_y)**2) if not np.isnan(slit_x) else np.nan

    # Nose at end
    nose_x_end, nose_y_end, nose_l_end = get_nose_position(dlc_df, end_frame)
    nose_dist_end = np.sqrt((nose_x_end - slit_x)**2 + (nose_y_end - slit_y)**2) if not np.isnan(slit_x) else np.nan

    # Hand at start
    hand_start = get_hand_visibility(dlc_df, start_frame)
    visible_start = [p for p, v in hand_start.items() if v['visible']]
    likelihoods_start = {p: v['likelihood'] for p, v in hand_start.items()}

    # Hand at end
    hand_end = get_hand_visibility(dlc_df, end_frame)
    visible_end = [p for p, v in hand_end.items() if v['visible']]
    likelihoods_end = {p: v['likelihood'] for p, v in hand_end.items()}

    return ReachAnalysis(
        reach_id=reach.get('reach_id', 0),
        start_frame=start_frame,
        end_frame=end_frame,
        duration=end_frame - start_frame + 1,
        nose_x_start=nose_x_start,
        nose_y_start=nose_y_start,
        nose_x_end=nose_x_end,
        nose_y_end=nose_y_end,
        nose_dist_from_slit_start=nose_dist_start,
        nose_dist_from_slit_end=nose_dist_end,
        hand_points_visible_start=visible_start,
        hand_likelihood_start=likelihoods_start,
        hand_points_visible_end=visible_end,
        hand_likelihood_end=likelihoods_end,
        any_hand_visible_start=len(visible_start) > 0,
        any_hand_visible_end=len(visible_end) > 0
    )


def run_analysis(
    dlc_path: Path,
    gt_path: Path,
    output_path: Path = None
) -> List[ReachAnalysis]:
    """Run full analysis on a video."""

    print(f"Loading DLC data from {dlc_path.name}...")
    dlc_df, scorer = load_dlc_data(dlc_path)

    print(f"  Scorer: {scorer}")
    print(f"  Frames: {len(dlc_df)}")
    print(f"  Columns: {list(dlc_df.columns[:10])}...")

    print(f"\nLoading ground truth from {gt_path.name}...")
    gt_data = load_ground_truth(gt_path)

    # Extract all reaches from all segments
    all_reaches = []
    for segment in gt_data.get('segments', []):
        for reach in segment.get('reaches', []):
            all_reaches.append(reach)

    print(f"  Total reaches: {len(all_reaches)}")

    # Analyze each reach
    print(f"\nAnalyzing reaches...")
    results = []
    for reach in all_reaches:
        try:
            analysis = analyze_reach(dlc_df, reach)
            results.append(analysis)
        except Exception as e:
            print(f"  Error analyzing reach {reach.get('reach_id')}: {e}")

    print(f"  Analyzed: {len(results)} reaches")

    return results, dlc_df, scorer


def analyze_hand_positions(results: List[ReachAnalysis], dlc_df: pd.DataFrame):
    """Analyze hand position changes during reaches."""
    print("\n" + "=" * 60)
    print("HAND POSITION ANALYSIS")
    print("=" * 60)

    hand_points = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']

    # For each reach, get hand position at start vs end
    x_changes = []  # end_x - start_x
    y_changes = []  # end_y - start_y

    for r in results:
        for point in hand_points:
            start_x = dlc_df.iloc[r.start_frame].get(f'{point}_x', np.nan)
            start_y = dlc_df.iloc[r.start_frame].get(f'{point}_y', np.nan)
            end_x = dlc_df.iloc[r.end_frame].get(f'{point}_x', np.nan)
            end_y = dlc_df.iloc[r.end_frame].get(f'{point}_y', np.nan)
            start_l = dlc_df.iloc[r.start_frame].get(f'{point}_likelihood', 0)
            end_l = dlc_df.iloc[r.end_frame].get(f'{point}_likelihood', 0)

            # Only count if both positions are confident
            if start_l > 0.3 and end_l > 0.3:
                if not np.isnan(start_x) and not np.isnan(end_x):
                    x_changes.append(end_x - start_x)
                    y_changes.append(end_y - start_y)

    if x_changes:
        print(f"\n## Hand X position change (end - start)")
        print(f"  Mean: {np.mean(x_changes):.1f} pixels")
        print(f"  Median: {np.median(x_changes):.1f} pixels")
        print(f"  Std: {np.std(x_changes):.1f}")
        print(f"  Positive (moved right): {sum(1 for x in x_changes if x > 5)}/{len(x_changes)} ({100*sum(1 for x in x_changes if x > 5)/len(x_changes):.1f}%)")
        print(f"  Negative (moved left): {sum(1 for x in x_changes if x < -5)}/{len(x_changes)} ({100*sum(1 for x in x_changes if x < -5)/len(x_changes):.1f}%)")

        print(f"\n## Hand Y position change (end - start)")
        print(f"  Mean: {np.mean(y_changes):.1f} pixels")
        print(f"  Median: {np.median(y_changes):.1f} pixels")
        print(f"  Positive (moved down): {sum(1 for y in y_changes if y > 5)}/{len(y_changes)} ({100*sum(1 for y in y_changes if y > 5)/len(y_changes):.1f}%)")
        print(f"  Negative (moved up): {sum(1 for y in y_changes if y < -5)}/{len(y_changes)} ({100*sum(1 for y in y_changes if y < -5)/len(y_changes):.1f}%)")


def analyze_velocity_at_boundaries(results: List[ReachAnalysis], dlc_df: pd.DataFrame):
    """Analyze hand velocity at reach start and end."""
    print("\n" + "=" * 60)
    print("VELOCITY AT REACH BOUNDARIES")
    print("=" * 60)

    # Calculate velocity as position change between frames
    def get_velocity(frame: int, point: str = 'RightHand') -> Tuple[float, float]:
        """Get X and Y velocity at a frame (difference from previous frame)."""
        if frame < 1 or frame >= len(dlc_df):
            return np.nan, np.nan

        x_curr = dlc_df.iloc[frame].get(f'{point}_x', np.nan)
        y_curr = dlc_df.iloc[frame].get(f'{point}_y', np.nan)
        x_prev = dlc_df.iloc[frame - 1].get(f'{point}_x', np.nan)
        y_prev = dlc_df.iloc[frame - 1].get(f'{point}_y', np.nan)

        l_curr = dlc_df.iloc[frame].get(f'{point}_likelihood', 0)
        l_prev = dlc_df.iloc[frame - 1].get(f'{point}_likelihood', 0)

        if l_curr < 0.3 or l_prev < 0.3:
            return np.nan, np.nan

        return x_curr - x_prev, y_curr - y_prev

    # Velocities at reach start
    vx_start = []
    vy_start = []
    for r in results:
        vx, vy = get_velocity(r.start_frame)
        if not np.isnan(vx):
            vx_start.append(vx)
            vy_start.append(vy)

    # Velocities at reach end
    vx_end = []
    vy_end = []
    for r in results:
        vx, vy = get_velocity(r.end_frame)
        if not np.isnan(vx):
            vx_end.append(vx)
            vy_end.append(vy)

    # Velocities 1 frame AFTER reach end (is there a direction change?)
    vx_post = []
    vy_post = []
    for r in results:
        vx, vy = get_velocity(r.end_frame + 1)
        if not np.isnan(vx):
            vx_post.append(vx)
            vy_post.append(vy)

    if vx_start:
        print(f"\n## X Velocity (positive=moving right)")
        print(f"  At reach START: Mean={np.mean(vx_start):.2f}, Median={np.median(vx_start):.2f}")
        print(f"  At reach END: Mean={np.mean(vx_end):.2f}, Median={np.median(vx_end):.2f}")
        if vx_post:
            print(f"  POST reach end (+1 frame): Mean={np.mean(vx_post):.2f}, Median={np.median(vx_post):.2f}")

        print(f"\n## Y Velocity (positive=moving down)")
        print(f"  At reach START: Mean={np.mean(vy_start):.2f}, Median={np.median(vy_start):.2f}")
        print(f"  At reach END: Mean={np.mean(vy_end):.2f}, Median={np.median(vy_end):.2f}")
        if vy_post:
            print(f"  POST reach end (+1 frame): Mean={np.mean(vy_post):.2f}, Median={np.median(vy_post):.2f}")

    # Check for velocity sign changes at reach end (direction reversal)
    direction_changes = 0
    velocity_drops = 0  # Significant slowing
    for r in results:
        vx_at_end, _ = get_velocity(r.end_frame)
        vx_after, _ = get_velocity(r.end_frame + 1)

        if not np.isnan(vx_at_end) and not np.isnan(vx_after):
            # Direction change (sign flip)
            if vx_at_end * vx_after < 0:
                direction_changes += 1
            # Significant velocity drop (moving -> stopped or slowed)
            if abs(vx_at_end) > 1 and abs(vx_after) < abs(vx_at_end) * 0.5:
                velocity_drops += 1

    print(f"\n## Direction/speed changes at reach end")
    print(f"  X direction reversal (end->post): {direction_changes}")
    print(f"  Significant velocity drop (>50%): {velocity_drops}")


def analyze_inter_reach_gaps(results: List[ReachAnalysis], dlc_df: pd.DataFrame):
    """Analyze gaps between consecutive reaches."""
    print("\n" + "=" * 60)
    print("INTER-REACH GAP ANALYSIS")
    print("=" * 60)

    # Sort by start frame
    sorted_results = sorted(results, key=lambda r: r.start_frame)

    gaps = []
    hand_visible_in_gap = []  # Is hand visible during gap?

    for i in range(len(sorted_results) - 1):
        current = sorted_results[i]
        next_reach = sorted_results[i + 1]

        gap = next_reach.start_frame - current.end_frame

        if gap > 0:
            gaps.append(gap)

            # Check if hand is visible during the gap
            hand_points = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']
            any_visible_in_gap = False

            for frame in range(current.end_frame + 1, next_reach.start_frame):
                for point in hand_points:
                    likelihood = dlc_df.iloc[frame].get(f'{point}_likelihood', 0)
                    if likelihood >= 0.3:
                        any_visible_in_gap = True
                        break
                if any_visible_in_gap:
                    break

            hand_visible_in_gap.append(any_visible_in_gap)

    if gaps:
        print(f"\n## Gap between consecutive reaches (frames)")
        print(f"  Mean: {np.mean(gaps):.1f}")
        print(f"  Median: {np.median(gaps):.1f}")
        print(f"  Min: {np.min(gaps)}")
        print(f"  Max: {np.max(gaps)}")
        print(f"  Gaps <= 5 frames: {sum(1 for g in gaps if g <= 5)}/{len(gaps)}")
        print(f"  Gaps <= 10 frames: {sum(1 for g in gaps if g <= 10)}/{len(gaps)}")

        if hand_visible_in_gap:
            visible_count = sum(hand_visible_in_gap)
            print(f"\n## Hand visibility in gaps")
            print(f"  Gaps where hand is visible: {visible_count}/{len(hand_visible_in_gap)} ({100*visible_count/len(hand_visible_in_gap):.1f}%)")


def analyze_short_gap_details(results: List[ReachAnalysis], dlc_df: pd.DataFrame):
    """Deep dive into short gaps where hand stays visible - how are reaches distinguished?"""
    print("\n" + "=" * 60)
    print("SHORT GAP ANALYSIS (gaps <= 10 frames with hand visible)")
    print("=" * 60)

    sorted_results = sorted(results, key=lambda r: r.start_frame)
    hand_points = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']

    short_visible_gaps = []

    for i in range(len(sorted_results) - 1):
        current = sorted_results[i]
        next_reach = sorted_results[i + 1]

        gap = next_reach.start_frame - current.end_frame

        if gap > 0 and gap <= 10:
            # Check if hand visible in gap
            any_visible = False
            for frame in range(current.end_frame + 1, next_reach.start_frame):
                for point in hand_points:
                    if dlc_df.iloc[frame].get(f'{point}_likelihood', 0) >= 0.3:
                        any_visible = True
                        break
                if any_visible:
                    break

            if any_visible:
                # Get hand X position at end of current reach vs start of next
                end_x = dlc_df.iloc[current.end_frame].get('RightHand_x', np.nan)
                start_x = dlc_df.iloc[next_reach.start_frame].get('RightHand_x', np.nan)

                # Get slit center
                slit_x, _ = get_slit_center(dlc_df, current.end_frame)

                short_visible_gaps.append({
                    'gap': gap,
                    'end_frame': current.end_frame,
                    'next_start_frame': next_reach.start_frame,
                    'hand_x_at_end': end_x - slit_x if not np.isnan(slit_x) and not np.isnan(end_x) else np.nan,
                    'hand_x_at_next_start': start_x - slit_x if not np.isnan(slit_x) and not np.isnan(start_x) else np.nan,
                })

    if short_visible_gaps:
        print(f"\n## Found {len(short_visible_gaps)} short gaps with hand visible")

        # Hand position comparison
        x_at_end = [g['hand_x_at_end'] for g in short_visible_gaps if not np.isnan(g['hand_x_at_end'])]
        x_at_next = [g['hand_x_at_next_start'] for g in short_visible_gaps if not np.isnan(g['hand_x_at_next_start'])]

        if x_at_end:
            print(f"\n## Hand X position relative to slit")
            print(f"  At current reach END: Mean={np.mean(x_at_end):.1f}, Median={np.median(x_at_end):.1f}")
            print(f"  At next reach START: Mean={np.mean(x_at_next):.1f}, Median={np.median(x_at_next):.1f}")

            # Did hand move back left between reaches?
            moved_left = sum(1 for i in range(len(x_at_end)) if x_at_next[i] < x_at_end[i] - 2)
            print(f"\n  Hand moved LEFT between reaches: {moved_left}/{len(x_at_end)} ({100*moved_left/len(x_at_end):.1f}%)")

        # Print a few examples
        print(f"\n## Sample short gaps (first 5):")
        for g in short_visible_gaps[:5]:
            print(f"  Gap={g['gap']} frames, end_frame={g['end_frame']}, hand_x: {g['hand_x_at_end']:.1f} -> {g['hand_x_at_next_start']:.1f}")
    else:
        print(f"\n  No short gaps with hand visible found")


def analyze_post_reach(results: List[ReachAnalysis], dlc_df: pd.DataFrame, look_ahead: int = 10):
    """Analyze what happens AFTER reach ends."""
    print("\n" + "=" * 60)
    print("POST-REACH ANALYSIS (what happens after reach end?)")
    print("=" * 60)

    hand_points = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']

    # How many frames until hand disappears after reach end?
    frames_until_disappear = []
    hand_disappears_within = {5: 0, 10: 0, 20: 0}

    for r in results:
        end_frame = r.end_frame

        # Look at frames after reach end
        for offset in range(1, min(look_ahead + 1, len(dlc_df) - end_frame)):
            frame = end_frame + offset
            any_visible = False

            for point in hand_points:
                likelihood = dlc_df.iloc[frame].get(f'{point}_likelihood', 0)
                if likelihood >= 0.3:
                    any_visible = True
                    break

            if not any_visible:
                frames_until_disappear.append(offset)
                for threshold in hand_disappears_within:
                    if offset <= threshold:
                        hand_disappears_within[threshold] += 1
                break

    total = len(results)
    print(f"\n## Hand disappearance after reach end")
    print(f"  Disappears within 5 frames: {hand_disappears_within[5]}/{total} ({100*hand_disappears_within[5]/total:.1f}%)")
    print(f"  Disappears within 10 frames: {hand_disappears_within[10]}/{total} ({100*hand_disappears_within[10]/total:.1f}%)")
    print(f"  Disappears within 20 frames: {hand_disappears_within[20]}/{total} ({100*hand_disappears_within[20]/total:.1f}%)")

    if frames_until_disappear:
        print(f"\n  Of reaches where hand disappears within {look_ahead} frames:")
        print(f"    Mean frames until disappear: {np.mean(frames_until_disappear):.1f}")
        print(f"    Median: {np.median(frames_until_disappear):.1f}")


def analyze_hand_relative_to_slit(results: List[ReachAnalysis], dlc_df: pd.DataFrame):
    """Analyze hand position relative to slit center."""
    print("\n" + "=" * 60)
    print("HAND POSITION RELATIVE TO SLIT")
    print("=" * 60)

    hand_x_rel_start = []  # hand_x - slit_x at start
    hand_y_rel_start = []  # hand_y - slit_y at start
    hand_x_rel_end = []
    hand_y_rel_end = []

    for r in results:
        # Get slit center
        slit_x, slit_y = get_slit_center(dlc_df, r.start_frame)
        if np.isnan(slit_x):
            continue

        # Get RightHand position at start
        start_x = dlc_df.iloc[r.start_frame].get('RightHand_x', np.nan)
        start_y = dlc_df.iloc[r.start_frame].get('RightHand_y', np.nan)
        start_l = dlc_df.iloc[r.start_frame].get('RightHand_likelihood', 0)

        if start_l > 0.3 and not np.isnan(start_x):
            hand_x_rel_start.append(start_x - slit_x)
            hand_y_rel_start.append(start_y - slit_y)

        # Get RightHand position at end
        slit_x_end, slit_y_end = get_slit_center(dlc_df, r.end_frame)
        end_x = dlc_df.iloc[r.end_frame].get('RightHand_x', np.nan)
        end_y = dlc_df.iloc[r.end_frame].get('RightHand_y', np.nan)
        end_l = dlc_df.iloc[r.end_frame].get('RightHand_likelihood', 0)

        if end_l > 0.3 and not np.isnan(end_x) and not np.isnan(slit_x_end):
            hand_x_rel_end.append(end_x - slit_x_end)
            hand_y_rel_end.append(end_y - slit_y_end)

    if hand_x_rel_start:
        print(f"\n## RightHand relative to slit at REACH START")
        print(f"  X offset (positive=right of slit): Mean={np.mean(hand_x_rel_start):.1f}, Median={np.median(hand_x_rel_start):.1f}")
        print(f"  Y offset (positive=below slit): Mean={np.mean(hand_y_rel_start):.1f}, Median={np.median(hand_y_rel_start):.1f}")

        print(f"\n## RightHand relative to slit at REACH END")
        print(f"  X offset (positive=right of slit): Mean={np.mean(hand_x_rel_end):.1f}, Median={np.median(hand_x_rel_end):.1f}")
        print(f"  Y offset (positive=below slit): Mean={np.mean(hand_y_rel_end):.1f}, Median={np.median(hand_y_rel_end):.1f}")

        # Movement direction
        if hand_x_rel_start and hand_x_rel_end and len(hand_x_rel_start) == len(hand_x_rel_end):
            x_movement = [e - s for s, e in zip(hand_x_rel_start, hand_x_rel_end)]
            print(f"\n## Movement direction (reach start -> reach end)")
            print(f"  Moved right: {sum(1 for x in x_movement if x > 10)} ({100*sum(1 for x in x_movement if x > 10)/len(x_movement):.1f}%)")
            print(f"  Moved left: {sum(1 for x in x_movement if x < -10)} ({100*sum(1 for x in x_movement if x < -10)/len(x_movement):.1f}%)")


def print_summary(results: List[ReachAnalysis]):
    """Print summary statistics."""

    print("\n" + "=" * 60)
    print("GROUND TRUTH REACH PATTERN ANALYSIS")
    print("=" * 60)

    # Duration stats
    durations = [r.duration for r in results]
    print(f"\n## Duration (frames)")
    print(f"  Mean: {np.mean(durations):.1f}")
    print(f"  Median: {np.median(durations):.1f}")
    print(f"  Min: {np.min(durations)}")
    print(f"  Max: {np.max(durations)}")

    # Nose distance from slit at start
    nose_dists_start = [r.nose_dist_from_slit_start for r in results if not np.isnan(r.nose_dist_from_slit_start)]
    print(f"\n## Nose distance from slit at REACH START (pixels)")
    print(f"  Mean: {np.mean(nose_dists_start):.1f}")
    print(f"  Median: {np.median(nose_dists_start):.1f}")
    print(f"  Std: {np.std(nose_dists_start):.1f}")
    print(f"  90th percentile: {np.percentile(nose_dists_start, 90):.1f}")
    print(f"  95th percentile: {np.percentile(nose_dists_start, 95):.1f}")

    # Nose distance from slit at end
    nose_dists_end = [r.nose_dist_from_slit_end for r in results if not np.isnan(r.nose_dist_from_slit_end)]
    print(f"\n## Nose distance from slit at REACH END (pixels)")
    print(f"  Mean: {np.mean(nose_dists_end):.1f}")
    print(f"  Median: {np.median(nose_dists_end):.1f}")
    print(f"  Std: {np.std(nose_dists_end):.1f}")
    print(f"  90th percentile: {np.percentile(nose_dists_end, 90):.1f}")

    # Hand visibility at start
    visible_count_start = sum(1 for r in results if r.any_hand_visible_start)
    print(f"\n## Hand visibility at REACH START")
    print(f"  Any hand visible: {visible_count_start}/{len(results)} ({100*visible_count_start/len(results):.1f}%)")

    # Which points visible at start
    point_counts_start = {'RightHand': 0, 'RHLeft': 0, 'RHOut': 0, 'RHRight': 0}
    for r in results:
        for point in r.hand_points_visible_start:
            point_counts_start[point] += 1

    print(f"  Point visibility breakdown:")
    for point, count in sorted(point_counts_start.items(), key=lambda x: -x[1]):
        print(f"    {point}: {count} ({100*count/len(results):.1f}%)")

    # Hand visibility at end
    visible_count_end = sum(1 for r in results if r.any_hand_visible_end)
    print(f"\n## Hand visibility at REACH END")
    print(f"  Any hand visible: {visible_count_end}/{len(results)} ({100*visible_count_end/len(results):.1f}%)")

    # Which points visible at end
    point_counts_end = {'RightHand': 0, 'RHLeft': 0, 'RHOut': 0, 'RHRight': 0}
    for r in results:
        for point in r.hand_points_visible_end:
            point_counts_end[point] += 1

    print(f"  Point visibility breakdown:")
    for point, count in sorted(point_counts_end.items(), key=lambda x: -x[1]):
        print(f"    {point}: {count} ({100*count/len(results):.1f}%)")

    # Transition patterns
    print(f"\n## Reach start/end transitions")
    start_visible_end_not = sum(1 for r in results if r.any_hand_visible_start and not r.any_hand_visible_end)
    start_visible_end_visible = sum(1 for r in results if r.any_hand_visible_start and r.any_hand_visible_end)
    start_not_end_visible = sum(1 for r in results if not r.any_hand_visible_start and r.any_hand_visible_end)
    start_not_end_not = sum(1 for r in results if not r.any_hand_visible_start and not r.any_hand_visible_end)

    print(f"  Hand visible at start, gone at end: {start_visible_end_not} ({100*start_visible_end_not/len(results):.1f}%)")
    print(f"  Hand visible at both start and end: {start_visible_end_visible} ({100*start_visible_end_visible/len(results):.1f}%)")
    print(f"  Hand not visible at start, visible at end: {start_not_end_visible} ({100*start_not_end_visible/len(results):.1f}%)")
    print(f"  Hand not visible at start or end: {start_not_end_not} ({100*start_not_end_not/len(results):.1f}%)")


def main():
    """Main entry point."""
    from mousereach.config import Paths
    processing_dir = Paths.PROCESSING
    video_id = "20251021_CNT0405_P4"

    dlc_path = processing_dir / f"{video_id}DLC_resnet50_MPSAOct27shuffle1_100000.h5"
    gt_path = processing_dir / f"{video_id}_reach_ground_truth.json"

    if not dlc_path.exists():
        print(f"DLC file not found: {dlc_path}")
        return

    if not gt_path.exists():
        print(f"Ground truth file not found: {gt_path}")
        return

    results, dlc_df, scorer = run_analysis(dlc_path, gt_path)
    print_summary(results)
    analyze_hand_positions(results, dlc_df)
    analyze_hand_relative_to_slit(results, dlc_df)
    analyze_velocity_at_boundaries(results, dlc_df)
    analyze_inter_reach_gaps(results, dlc_df)
    analyze_short_gap_details(results, dlc_df)
    analyze_post_reach(results, dlc_df, look_ahead=20)

    return results, dlc_df, scorer


if __name__ == "__main__":
    main()
