"""
Probe the DLC signals around early-end cases to find what distinguishes
real retraction from oscillation/bodypart switching.

For each early-end case, examine:
1. Which bodypart is "best" at each frame (identity switching)
2. Hand velocity (position delta)
3. How many of 4 hand points are visible
4. Position variance over recent frames
5. Multi-point position agreement
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_5_0")

RH_POINTS = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']
HAND_THRESHOLD = 0.5


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_dlc(video):
    dlc_files = list(DATA_DIR.glob(f"{video}DLC*.h5"))
    if not dlc_files:
        return None
    df = pd.read_hdf(dlc_files[0])
    if isinstance(df.columns, pd.MultiIndex):
        scorer = df.columns.get_level_values(0)[0]
        df = df[scorer]
        df.columns = [f"{bp}_{coord}" for bp, coord in df.columns]
    return df


def match_reaches(gt_reaches, algo_reaches, max_dist=30):
    candidates = []
    for gi, gr in enumerate(gt_reaches):
        for ai, ar in enumerate(algo_reaches):
            dist = abs(gr['start_frame'] - ar['start_frame'])
            if dist <= max_dist:
                candidates.append((dist, gi, ai))
    candidates.sort()
    gt_used, algo_used = set(), set()
    matches = []
    for dist, gi, ai in candidates:
        if gi not in gt_used and ai not in algo_used:
            gt_used.add(gi)
            algo_used.add(ai)
            matches.append((gt_reaches[gi], algo_reaches[ai]))
    return matches


def frame_signals(df, frame):
    """Get all hand signals at a single frame."""
    if frame < 0 or frame >= len(df):
        return None
    row = df.iloc[frame]

    points = {}
    best_bp = None
    best_l = 0
    n_visible = 0
    visible_xs = []

    for bp in RH_POINTS:
        l = row.get(f'{bp}_likelihood', 0)
        x = row.get(f'{bp}_x', np.nan)
        y = row.get(f'{bp}_y', np.nan)
        points[bp] = {'x': x, 'y': y, 'l': l}
        if l >= HAND_THRESHOLD:
            n_visible += 1
            if not np.isnan(x):
                visible_xs.append(x)
            if l > best_l:
                best_l = l
                best_bp = bp

    best_x = points[best_bp]['x'] if best_bp else None

    return {
        'best_bp': best_bp,
        'best_l': best_l,
        'best_x': best_x,
        'n_visible': n_visible,
        'visible_xs': visible_xs,
        'x_spread': max(visible_xs) - min(visible_xs) if len(visible_xs) >= 2 else 0,
        'points': points,
    }


def main():
    print("MULTI-SIGNAL ANALYSIS OF EARLY-END CASES")
    print("=" * 70)

    # Collect stats across all early-end cases
    bp_switch_count = 0
    no_bp_switch_count = 0
    velocity_toward_count = 0
    velocity_away_count = 0
    multi_point_agree_retract = 0
    multi_point_disagree = 0
    single_point_cases = 0
    total_cases = 0

    # Per-case details for first 15
    detailed_cases = []

    for gt_file in sorted(DATA_DIR.glob("*_unified_ground_truth.json")):
        if 'archive' in str(gt_file):
            continue
        gt = load_json(gt_file)
        video = gt['video_name']

        gt_reaches = [r for r in gt.get('reaches', {}).get('reaches', [])
                      if r.get('start_determined') and r.get('end_determined')
                      and not r.get('exclude_from_analysis', False)]

        algo_file = ALGO_DIR / f"{video}_reaches.json"
        if not algo_file.exists():
            continue
        algo_data = load_json(algo_file)
        algo_reaches = [r for seg in algo_data['segments'] for r in seg['reaches']]

        df = load_dlc(video)
        if df is None:
            continue

        matches = match_reaches(gt_reaches, algo_reaches)

        for gt_r, algo_r in matches:
            end_offset = algo_r['end_frame'] - gt_r['end_frame']
            if end_offset >= -2:  # Not an early-end case
                continue

            total_cases += 1
            algo_end = algo_r['end_frame']

            # The retraction/return check fired at algo_end+1
            # (because end_frame = frame - 1)
            trigger_frame = algo_end + 1

            # Get signals at trigger frame and surrounding frames
            sig_before = frame_signals(df, trigger_frame - 1)  # algo_end
            sig_trigger = frame_signals(df, trigger_frame)      # where check fired
            sig_after1 = frame_signals(df, trigger_frame + 1)
            sig_after2 = frame_signals(df, trigger_frame + 2)

            if not sig_before or not sig_trigger:
                continue

            # Signal 1: Did best bodypart identity change?
            bp_switched = (sig_before['best_bp'] != sig_trigger['best_bp']
                          and sig_before['best_bp'] is not None
                          and sig_trigger['best_bp'] is not None)
            if bp_switched:
                bp_switch_count += 1
            else:
                no_bp_switch_count += 1

            # Signal 2: Hand velocity (position delta)
            if sig_before['best_x'] is not None and sig_trigger['best_x'] is not None:
                velocity = sig_trigger['best_x'] - sig_before['best_x']
                if velocity > 0:
                    velocity_toward_count += 1  # Moving toward pellet (right)
                else:
                    velocity_away_count += 1    # Moving away (left = retracting)
            else:
                velocity = None

            # Signal 3: Multi-point agreement on retraction
            if sig_trigger['n_visible'] >= 2:
                # Check if ALL visible points moved left
                all_retracted = True
                for bp in RH_POINTS:
                    if (sig_before['points'][bp]['l'] >= HAND_THRESHOLD and
                        sig_trigger['points'][bp]['l'] >= HAND_THRESHOLD):
                        bx = sig_before['points'][bp]['x']
                        tx = sig_trigger['points'][bp]['x']
                        if not np.isnan(bx) and not np.isnan(tx):
                            if tx >= bx:  # This point didn't retract
                                all_retracted = False
                                break
                if all_retracted:
                    multi_point_agree_retract += 1
                else:
                    multi_point_disagree += 1
            elif sig_trigger['n_visible'] == 1:
                single_point_cases += 1

            # Collect detailed case
            if len(detailed_cases) < 15:
                # Compute velocity over 3 frames before trigger
                velocities = []
                for f in range(trigger_frame - 3, trigger_frame + 1):
                    s1 = frame_signals(df, f)
                    s2 = frame_signals(df, f + 1)
                    if s1 and s2 and s1['best_x'] and s2['best_x']:
                        velocities.append(s2['best_x'] - s1['best_x'])

                detailed_cases.append({
                    'video': video,
                    'algo_end': algo_end,
                    'gt_end': gt_r['end_frame'],
                    'offset': end_offset,
                    'bp_switched': bp_switched,
                    'bp_before': sig_before['best_bp'],
                    'bp_trigger': sig_trigger['best_bp'],
                    'n_vis_before': sig_before['n_visible'],
                    'n_vis_trigger': sig_trigger['n_visible'],
                    'velocity': velocity,
                    'velocities': velocities,
                    'x_spread_trigger': sig_trigger['x_spread'],
                    'best_x_before': sig_before['best_x'],
                    'best_x_trigger': sig_trigger['best_x'],
                })

    # Summary
    print(f"\nTotal early-end cases analyzed: {total_cases}")
    print(f"\n--- Signal 1: Bodypart Identity Switch ---")
    print(f"  BP switched at trigger: {bp_switch_count}/{total_cases} ({bp_switch_count/total_cases*100:.1f}%)")
    print(f"  BP stable at trigger:   {no_bp_switch_count}/{total_cases} ({no_bp_switch_count/total_cases*100:.1f}%)")

    print(f"\n--- Signal 2: Hand Velocity at Trigger ---")
    print(f"  Toward pellet (positive): {velocity_toward_count}/{total_cases} ({velocity_toward_count/total_cases*100:.1f}%)")
    print(f"  Away from pellet (negative): {velocity_away_count}/{total_cases} ({velocity_away_count/total_cases*100:.1f}%)")

    print(f"\n--- Signal 3: Multi-point Agreement ---")
    print(f"  Multiple points agree retract: {multi_point_agree_retract}/{total_cases} ({multi_point_agree_retract/total_cases*100:.1f}%)")
    print(f"  Multiple points DISAGREE:      {multi_point_disagree}/{total_cases} ({multi_point_disagree/total_cases*100:.1f}%)")
    print(f"  Only 1 point visible:          {single_point_cases}/{total_cases} ({single_point_cases/total_cases*100:.1f}%)")

    # Combination: how many cases could a decision tree catch?
    catchable = 0
    for i in range(total_cases):
        # Placeholder - need to check per case
        pass

    print(f"\n--- Detailed Cases ---")
    for i, c in enumerate(detailed_cases):
        print(f"\n  [{i}] {c['video']} algo_end={c['algo_end']} gt_end={c['gt_end']} offset={c['offset']}")
        print(f"      BP: {c['bp_before']} -> {c['bp_trigger']} {'SWITCH!' if c['bp_switched'] else '(stable)'}")
        print(f"      Visible points: {c['n_vis_before']} -> {c['n_vis_trigger']}")
        print(f"      Best X: {c['best_x_before']:.1f} -> {c['best_x_trigger']:.1f}" if c['best_x_before'] and c['best_x_trigger'] else f"      Best X: ? -> ?")
        if c['velocity'] is not None:
            print(f"      Velocity at trigger: {c['velocity']:+.1f}px {'(TOWARD pellet)' if c['velocity'] > 0 else '(AWAY from pellet)'}")
        print(f"      Recent velocities: {[f'{v:+.1f}' for v in c['velocities']]}")
        print(f"      X spread at trigger: {c['x_spread_trigger']:.1f}px")


if __name__ == "__main__":
    main()
