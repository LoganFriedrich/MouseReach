"""
Analyze reach SPLITTING accuracy separately from reach DETECTION accuracy.

Key insight from investigate_misses.py:
  - 730/743 (98.3%) of algo misses are GT reaches ABSORBED into larger algo reaches
  - The algo IS detecting reaching activity, but merging ~5 distinct reaches into 1
  - Detection is excellent; SPLITTING is the problem

This script:
  1. Separates detection accuracy from splitting accuracy in stats
  2. Analyzes DLC features at GT reach boundaries within long algo reaches
     (what does the hand do at the moment humans say "this reach ends"?)
  3. Provides evidence for whether splitting can be rule-based or needs DL

The algo's reach-end conditions (reach_detector.py):
  - _detect_hand_retraction(): retraction > 40% of extension AND > 5px
  - _hand_returned_to_start(): hand_offset < HAND_RETURN_THRESHOLD (5px)
  - _hand_will_disappear(): hand invisible for DISAPPEAR_THRESHOLD consecutive frames
  - hand not visible (likelihood < 0.5)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_0_0")

RH_POINTS = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']

# Algorithm thresholds (must match reach_detector.py)
HAND_LIKELIHOOD_THRESHOLD = 0.5
NOSE_ENGAGEMENT_THRESHOLD = 25  # px
HAND_RETURN_THRESHOLD = 5.0  # px
RETRACTION_FRACTION = 0.40
RETRACTION_MIN_PX = 5.0
MIN_EXTENSION = 5.0  # px minimum extension to consider retraction


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_slit_center(dlc_df, scorer):
    """Get slit center from BOXL/BOXR median positions."""
    try:
        boxl_x = dlc_df[(scorer, 'BOXL', 'x')].median()
        boxr_x = dlc_df[(scorer, 'BOXR', 'x')].median()
        boxl_y = dlc_df[(scorer, 'BOXL', 'y')].median()
        boxr_y = dlc_df[(scorer, 'BOXR', 'y')].median()
        return (boxl_x + boxr_x) / 2, (boxl_y + boxr_y) / 2
    except KeyError:
        return None, None


def get_best_hand_x(dlc_df, scorer, frame):
    """Get best hand x-position at a frame (highest likelihood point)."""
    best_x = None
    best_like = 0
    for part in RH_POINTS:
        try:
            x = dlc_df[(scorer, part, 'x')].iloc[frame]
            like = dlc_df[(scorer, part, 'likelihood')].iloc[frame]
            if like > best_like and like >= HAND_LIKELIHOOD_THRESHOLD:
                best_x = x
                best_like = like
        except (KeyError, IndexError):
            continue
    return best_x, best_like


def max_hand_likelihood(dlc_df, scorer, frame):
    """Get max hand likelihood at a frame."""
    max_like = 0
    for part in RH_POINTS:
        try:
            like = dlc_df[(scorer, part, 'likelihood')].iloc[frame]
            max_like = max(max_like, like)
        except (KeyError, IndexError):
            continue
    return max_like


# ============================================================================
# PART 1: Separate detection accuracy from splitting accuracy
# ============================================================================

def compute_separated_stats():
    """Compute detection and splitting accuracy separately.

    Detection accuracy: Does the algo detect reaching ACTIVITY at this location?
    A GT reach is "detected" if it falls within ANY algo reach (even a merged one).

    Splitting accuracy: Given the algo detected reaching activity, does it
    correctly identify the individual reach boundaries?
    """
    print("=" * 70)
    print("PART 1: DETECTION vs SPLITTING ACCURACY")
    print("=" * 70)
    print()
    print("Detection = 'does the algo find reaching activity here?'")
    print("Splitting = 'does the algo correctly separate individual reaches?'")
    print()

    TOLERANCE = 5

    total_gt = 0
    total_detected = 0       # GT reach overlaps with any algo reach
    total_matched_1to1 = 0   # GT reach matched 1:1 to an algo reach
    total_absorbed = 0       # GT reach inside a LARGER algo reach (merge)
    total_undetected = 0     # No algo reach overlapping at all

    # For splitting analysis: track merged algo reaches
    merge_groups = []  # Each: {'algo_reach': {...}, 'gt_reaches': [...], 'video': str}

    per_video = defaultdict(lambda: {
        'gt': 0, 'detected': 0, 'matched_1to1': 0,
        'absorbed': 0, 'undetected': 0
    })

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

        # First: standard 1:1 matching (same as analyze_algo_agreement.py)
        algo_used = set()
        gt_1to1_matched = set()

        for gi, gr in enumerate(gt_reaches):
            gt_start = gr['start_frame']
            gt_apex = gr.get('apex_frame')
            best_ai = None
            best_dist = float('inf')

            for ai, ar in enumerate(algo_reaches):
                if ai in algo_used:
                    continue
                dist = abs(gt_start - ar.get('start_frame', 0))
                if dist <= TOLERANCE and dist < best_dist:
                    best_ai = ai
                    best_dist = dist

            if gt_apex is not None:
                for ai, ar in enumerate(algo_reaches):
                    if ai in algo_used:
                        continue
                    algo_apex = ar.get('apex_frame')
                    if algo_apex is None:
                        continue
                    dist = abs(gt_apex - algo_apex)
                    if dist <= TOLERANCE and dist < best_dist:
                        best_ai = ai
                        best_dist = dist

            if best_ai is not None:
                algo_used.add(best_ai)
                gt_1to1_matched.add(gi)

        # Second: for unmatched GT reaches, check containment/overlap
        # Track which algo reaches contain which GT reaches
        algo_to_gt = defaultdict(list)  # algo index -> list of GT reaches it contains

        for gi, gr in enumerate(gt_reaches):
            gt_start = gr['start_frame']
            gt_end = gr['end_frame']

            if gi in gt_1to1_matched:
                total_matched_1to1 += 1
                per_video[video]['matched_1to1'] += 1
                total_detected += 1
                per_video[video]['detected'] += 1
            else:
                # Check if GT reach is CONTAINED in or OVERLAPS any algo reach
                found_containing = False
                for ai, ar in enumerate(algo_reaches):
                    a_start = ar.get('start_frame', 0)
                    a_end = ar.get('end_frame', 0)
                    # Check overlap
                    overlap_start = max(gt_start, a_start)
                    overlap_end = min(gt_end, a_end)
                    if overlap_end >= overlap_start:
                        # There IS overlap - algo detected activity here
                        found_containing = True
                        algo_to_gt[ai].append(gr)
                        break

                if found_containing:
                    total_absorbed += 1
                    per_video[video]['absorbed'] += 1
                    total_detected += 1
                    per_video[video]['detected'] += 1
                else:
                    total_undetected += 1
                    per_video[video]['undetected'] += 1

            total_gt += 1
            per_video[video]['gt'] += 1

        # Also count 1:1 matched algo reaches in the merge tracker
        for ai in algo_used:
            algo_to_gt[ai].append(None)  # placeholder for 1:1 match

        # Build merge groups for later DLC analysis
        for ai, ar in enumerate(algo_reaches):
            # Find ALL GT reaches that overlap this algo reach
            overlapping_gt = []
            a_start = ar.get('start_frame', 0)
            a_end = ar.get('end_frame', 0)
            for gr in gt_reaches:
                g_start = gr['start_frame']
                g_end = gr['end_frame']
                overlap_s = max(g_start, a_start)
                overlap_e = min(g_end, a_end)
                if overlap_e >= overlap_s:
                    overlapping_gt.append(gr)

            if len(overlapping_gt) >= 2:
                merge_groups.append({
                    'algo_reach': ar,
                    'gt_reaches': sorted(overlapping_gt, key=lambda r: r['start_frame']),
                    'video': video,
                })

    # Print results
    det_rate = total_detected / max(total_gt, 1) * 100
    match_rate = total_matched_1to1 / max(total_gt, 1) * 100
    absorb_rate = total_absorbed / max(total_gt, 1) * 100
    undet_rate = total_undetected / max(total_gt, 1) * 100

    print(f"Total human-determined reaches: {total_gt}")
    print()
    print(f"  DETECTION ACCURACY (does algo find activity here?):")
    print(f"    Detected:   {total_detected}/{total_gt} ({det_rate:.1f}%)")
    print(f"    Undetected: {total_undetected}/{total_gt} ({undet_rate:.1f}%)")
    print()
    print(f"  SPLITTING ACCURACY (among detected, is it a 1:1 match?):")
    print(f"    Correctly split (1:1 match):  {total_matched_1to1}/{total_detected} "
          f"({total_matched_1to1/max(total_detected,1)*100:.1f}%)")
    print(f"    Absorbed (merged, no split):  {total_absorbed}/{total_detected} "
          f"({total_absorbed/max(total_detected,1)*100:.1f}%)")
    print()
    print(f"  COMBINED VIEW:")
    print(f"    1:1 matched:    {total_matched_1to1} ({match_rate:.1f}%) -- Detection OK, Splitting OK")
    print(f"    Absorbed/merged:{total_absorbed} ({absorb_rate:.1f}%) -- Detection OK, Splitting FAILED")
    print(f"    Undetected:     {total_undetected} ({undet_rate:.1f}%) -- Detection FAILED")

    # Per-video breakdown
    print(f"\n  Per-video breakdown:")
    print(f"  {'Video':<35} {'GT':>4} {'Det%':>5} {'1:1%':>5} {'Abs%':>5} {'Und%':>5}")
    print(f"  {'-'*60}")
    for video in sorted(per_video.keys(),
                        key=lambda v: per_video[v]['detected'] / max(per_video[v]['gt'], 1)):
        v = per_video[video]
        g = max(v['gt'], 1)
        print(f"  {video:<35} {v['gt']:>4} "
              f"{v['detected']/g*100:>4.0f}% "
              f"{v['matched_1to1']/g*100:>4.0f}% "
              f"{v['absorbed']/g*100:>4.0f}% "
              f"{v['undetected']/g*100:>4.0f}%")

    # Merge group statistics
    if merge_groups:
        gt_per_merge = [len(m['gt_reaches']) for m in merge_groups]
        algo_durs = [m['algo_reach'].get('end_frame', 0) - m['algo_reach'].get('start_frame', 0) + 1
                     for m in merge_groups]
        print(f"\n  MERGE GROUP STATISTICS:")
        print(f"    Algo reaches containing 2+ GT reaches: {len(merge_groups)}")
        print(f"    GT reaches per merged algo reach: "
              f"mean={np.mean(gt_per_merge):.1f}, "
              f"median={np.median(gt_per_merge):.0f}, "
              f"max={np.max(gt_per_merge)}")
        print(f"    Merged algo reach duration: "
              f"mean={np.mean(algo_durs):.0f}, "
              f"median={np.median(algo_durs):.0f}")

        # Distribution of GT reaches per merge
        dist = Counter(gt_per_merge)
        print(f"\n    Distribution of GT reaches per merged algo reach:")
        for n_gt in sorted(dist.keys()):
            print(f"      {n_gt} GT reaches: {dist[n_gt]} algo reaches")

    return merge_groups


# ============================================================================
# PART 2: Analyze DLC features at GT reach boundaries within long algo reaches
# ============================================================================

def analyze_split_points(merge_groups):
    """At frames where humans say one reach ends and the next begins
    (within a single algo reach), what do the DLC features look like?

    This tells us: is there a detectable signal at the split point that
    the algo's current thresholds miss?
    """
    print(f"\n\n{'=' * 70}")
    print("PART 2: DLC FEATURES AT SPLIT POINTS")
    print("=" * 70)
    print()
    print("At each point where humans say 'reach N ends, reach N+1 begins'")
    print("(within a single long algo reach), what is the hand doing?")
    print()

    if not merge_groups:
        print("No merge groups found - nothing to analyze.")
        return []

    # Group by video for efficient DLC loading
    by_video = defaultdict(list)
    for mg in merge_groups:
        by_video[mg['video']].append(mg)

    split_features = []
    video_count = 0
    total_splits = sum(len(mg['gt_reaches']) - 1 for mg in merge_groups)
    print(f"Total split points to analyze: {total_splits}")
    print(f"(From {len(merge_groups)} merged algo reaches across {len(by_video)} videos)")
    print()

    for video, groups in sorted(by_video.items()):
        # Load DLC data
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

        video_count += 1

        for mg in groups:
            gt_reaches = mg['gt_reaches']
            algo_start = mg['algo_reach'].get('start_frame', 0)
            algo_end = mg['algo_reach'].get('end_frame', 0)

            # For each consecutive pair of GT reaches
            for i in range(len(gt_reaches) - 1):
                r_end = gt_reaches[i]
                r_next = gt_reaches[i + 1]

                end_frame = r_end['end_frame']
                next_start = r_next['start_frame']
                gap = next_start - end_frame  # Gap between end of reach i and start of reach i+1

                if end_frame >= len(dlc_df) or next_start >= len(dlc_df):
                    continue

                # --- Analyze DLC features around the split point ---
                # We look at: the end of reach i and start of reach i+1

                # 1. Hand position at reach i END
                hand_x_end, hand_like_end = get_best_hand_x(dlc_df, scorer, end_frame)

                # 2. Hand position at reach i+1 START
                hand_x_next, hand_like_next = get_best_hand_x(dlc_df, scorer, next_start)

                # 3. Max extension during reach i (approximate from frames)
                reach_max_x = 0
                for f in range(r_end['start_frame'], min(end_frame + 1, len(dlc_df))):
                    hx, _ = get_best_hand_x(dlc_df, scorer, f)
                    if hx is not None and hx > reach_max_x:
                        reach_max_x = hx

                # 4. Extension = max_x - slit_x
                extension = reach_max_x - slit_x if reach_max_x else 0

                # 5. Retraction at end frame
                retraction = (reach_max_x - hand_x_end) if (reach_max_x and hand_x_end) else 0
                retraction_pct = retraction / max(extension, 0.1) * 100

                # 6. Hand offset from slit at end frame
                hand_offset_end = (hand_x_end - slit_x) if hand_x_end else None

                # 7. Hand offset from slit at next start
                hand_offset_next = (hand_x_next - slit_x) if hand_x_next else None

                # 8. Was hand visible in the gap?
                gap_hand_visible = 0
                gap_hand_invisible = 0
                gap_min_like = 1.0
                for f in range(end_frame, min(next_start + 1, len(dlc_df))):
                    ml = max_hand_likelihood(dlc_df, scorer, f)
                    if ml >= HAND_LIKELIHOOD_THRESHOLD:
                        gap_hand_visible += 1
                    else:
                        gap_hand_invisible += 1
                    gap_min_like = min(gap_min_like, ml)

                # 9. Nose engaged at split point?
                try:
                    nx = dlc_df[(scorer, 'Nose', 'x')].iloc[end_frame]
                    ny = dlc_df[(scorer, 'Nose', 'y')].iloc[end_frame]
                    nl = dlc_df[(scorer, 'Nose', 'likelihood')].iloc[end_frame]
                    if nl >= 0.3:
                        nose_dist = np.sqrt((nx - slit_x)**2 + (ny - slit_y)**2)
                    else:
                        nose_dist = np.nan
                except (KeyError, IndexError):
                    nose_dist = np.nan

                # 10. Hand velocity (x-direction) near split point
                # Average over 3 frames before and after
                velocities_before = []
                for f in range(max(end_frame - 3, r_end['start_frame']), end_frame):
                    hx1, _ = get_best_hand_x(dlc_df, scorer, f)
                    hx2, _ = get_best_hand_x(dlc_df, scorer, f + 1)
                    if hx1 is not None and hx2 is not None:
                        velocities_before.append(hx2 - hx1)

                velocities_after = []
                for f in range(next_start, min(next_start + 3, r_next['end_frame'])):
                    hx1, _ = get_best_hand_x(dlc_df, scorer, f)
                    hx2, _ = get_best_hand_x(dlc_df, scorer, f + 1)
                    if hx1 is not None and hx2 is not None:
                        velocities_after.append(hx2 - hx1)

                mean_vel_before = np.mean(velocities_before) if velocities_before else np.nan
                mean_vel_after = np.mean(velocities_after) if velocities_after else np.nan

                # Would the algo's current thresholds trigger?
                retraction_would_trigger = (
                    extension >= MIN_EXTENSION
                    and retraction > extension * RETRACTION_FRACTION
                    and retraction > RETRACTION_MIN_PX
                )
                return_would_trigger = (
                    hand_offset_end is not None
                    and hand_offset_end < HAND_RETURN_THRESHOLD
                    and extension >= MIN_EXTENSION
                )
                hand_disappeared = gap_min_like < HAND_LIKELIHOOD_THRESHOLD

                feature = {
                    'video': video,
                    'reach_end_frame': end_frame,
                    'next_start_frame': next_start,
                    'gap_frames': gap,
                    'reach_duration': r_end['end_frame'] - r_end['start_frame'] + 1,
                    'extension_px': extension,
                    'retraction_px': retraction,
                    'retraction_pct': retraction_pct,
                    'hand_offset_at_end': hand_offset_end,
                    'hand_offset_at_next_start': hand_offset_next,
                    'hand_like_at_end': hand_like_end,
                    'hand_like_at_next_start': hand_like_next,
                    'gap_hand_visible_frames': gap_hand_visible,
                    'gap_hand_invisible_frames': gap_hand_invisible,
                    'gap_min_hand_like': gap_min_like,
                    'nose_dist_at_split': nose_dist,
                    'mean_vel_before': mean_vel_before,
                    'mean_vel_after': mean_vel_after,
                    # Would algo thresholds trigger?
                    'retraction_trigger': retraction_would_trigger,
                    'return_trigger': return_would_trigger,
                    'hand_disappear': hand_disappeared,
                    'any_trigger': retraction_would_trigger or return_would_trigger or hand_disappeared,
                }
                split_features.append(feature)

    print(f"Analyzed {len(split_features)} split points from {video_count} videos\n")

    if not split_features:
        print("No split points could be analyzed (DLC data not found).")
        return []

    # --- Summarize split point features ---
    df = pd.DataFrame(split_features)

    print("SPLIT POINT FEATURE SUMMARY:")
    print(f"  Gap between reaches (frames):  "
          f"mean={df['gap_frames'].mean():.1f}, "
          f"median={df['gap_frames'].median():.0f}, "
          f"min={df['gap_frames'].min()}, max={df['gap_frames'].max()}")
    print(f"  Extension (px):                "
          f"mean={df['extension_px'].mean():.1f}, "
          f"median={df['extension_px'].median():.1f}")
    print(f"  Retraction at end (px):        "
          f"mean={df['retraction_px'].mean():.1f}, "
          f"median={df['retraction_px'].median():.1f}")
    print(f"  Retraction at end (%):         "
          f"mean={df['retraction_pct'].mean():.1f}%, "
          f"median={df['retraction_pct'].median():.1f}%")

    valid_offsets = df['hand_offset_at_end'].dropna()
    if len(valid_offsets) > 0:
        print(f"  Hand offset from slit at end:  "
              f"mean={valid_offsets.mean():.1f}px, "
              f"median={valid_offsets.median():.1f}px")

    valid_vel_b = df['mean_vel_before'].dropna()
    valid_vel_a = df['mean_vel_after'].dropna()
    if len(valid_vel_b) > 0:
        print(f"  Hand velocity before split:    "
              f"mean={valid_vel_b.mean():.2f}px/frame "
              f"({'retracting' if valid_vel_b.mean() < 0 else 'extending'})")
    if len(valid_vel_a) > 0:
        print(f"  Hand velocity after split:     "
              f"mean={valid_vel_a.mean():.2f}px/frame "
              f"({'retracting' if valid_vel_a.mean() < 0 else 'extending'})")

    # Hand visibility in gap
    print(f"\n  Hand visibility during gap:")
    hand_vis = df['gap_min_hand_like'] >= HAND_LIKELIHOOD_THRESHOLD
    print(f"    Hand stays visible throughout: {hand_vis.sum()} ({hand_vis.mean()*100:.1f}%)")
    print(f"    Hand briefly disappears:       {(~hand_vis).sum()} ({(~hand_vis).mean()*100:.1f}%)")

    # --- Would current algo thresholds trigger at these split points? ---
    print(f"\n  ALGO THRESHOLD CHECK AT SPLIT POINTS:")
    print(f"  (Would the current algo end the reach at this point?)")
    n_retract = df['retraction_trigger'].sum()
    n_return = df['return_trigger'].sum()
    n_disappear = df['hand_disappear'].sum()
    n_any = df['any_trigger'].sum()
    n = len(df)
    print(f"    Retraction trigger (40%+5px):  {n_retract}/{n} ({n_retract/n*100:.1f}%)")
    print(f"    Return-to-start trigger (5px): {n_return}/{n} ({n_return/n*100:.1f}%)")
    print(f"    Hand disappears:               {n_disappear}/{n} ({n_disappear/n*100:.1f}%)")
    print(f"    ANY trigger would fire:        {n_any}/{n} ({n_any/n*100:.1f}%)")
    print(f"    NO trigger fires:              {n - n_any}/{n} ({(n-n_any)/n*100:.1f}%)")

    # --- Deeper look at the "no trigger" cases ---
    no_trigger = df[~df['any_trigger']]
    if len(no_trigger) > 0:
        print(f"\n  PROFILE OF SPLITS WHERE NO TRIGGER FIRES (n={len(no_trigger)}):")
        print(f"    These are the splits the algo CANNOT detect with current logic.")
        print(f"    Retraction %:  mean={no_trigger['retraction_pct'].mean():.1f}%, "
              f"median={no_trigger['retraction_pct'].median():.1f}%")
        nt_offsets = no_trigger['hand_offset_at_end'].dropna()
        if len(nt_offsets) > 0:
            print(f"    Hand offset:   mean={nt_offsets.mean():.1f}px, "
                  f"median={nt_offsets.median():.1f}px")
        print(f"    Gap (frames):  mean={no_trigger['gap_frames'].mean():.1f}, "
              f"median={no_trigger['gap_frames'].median():.0f}")
        nt_vel = no_trigger['mean_vel_before'].dropna()
        if len(nt_vel) > 0:
            print(f"    Velocity before: mean={nt_vel.mean():.2f}px/frame")

    # --- Retraction distribution ---
    print(f"\n  RETRACTION % DISTRIBUTION AT SPLIT POINTS:")
    for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        count = (df['retraction_pct'] >= threshold).sum()
        print(f"    >= {threshold}%: {count}/{n} ({count/n*100:.1f}%)")

    # --- Hand offset distribution ---
    print(f"\n  HAND OFFSET FROM SLIT AT SPLIT POINTS:")
    for threshold in [0, 2, 5, 10, 15, 20, 30, 50]:
        valid = df['hand_offset_at_end'].dropna()
        count = (valid < threshold).sum()
        print(f"    < {threshold}px from slit: {count}/{len(valid)} "
              f"({count/len(valid)*100:.1f}%)")

    return split_features


# ============================================================================
# PART 3: Can splitting be rule-based or does it need DL?
# ============================================================================

def assess_splitting_approach(split_features):
    """Based on the DLC feature analysis, assess whether simple threshold
    adjustments could fix splitting, or if the problem needs DL.

    Key question: Is there a CONSISTENT, DETECTABLE signal at human-identified
    split points that a rule could exploit?
    """
    print(f"\n\n{'=' * 70}")
    print("PART 3: RULE-BASED vs DEEP LEARNING FOR SPLITTING")
    print("=" * 70)
    print()

    if not split_features:
        print("No split features to analyze.")
        return

    df = pd.DataFrame(split_features)
    n = len(df)

    # Test various threshold relaxations
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("Testing what fraction of split points would be caught by relaxed thresholds:\n")

    # Retraction threshold sweep
    print("  A) RETRACTION THRESHOLD (current: 40% of extension AND >5px):")
    for pct in [10, 15, 20, 25, 30, 35, 40]:
        for min_px in [2, 3, 5]:
            caught = 0
            for _, row in df.iterrows():
                if row['extension_px'] >= MIN_EXTENSION:
                    if (row['retraction_pct'] >= pct and row['retraction_px'] >= min_px):
                        caught += 1
            if pct == 40 and min_px == 5:
                label = " <-- CURRENT"
            else:
                label = ""
            print(f"    {pct}% + {min_px}px: {caught}/{n} ({caught/n*100:.1f}%){label}")

    # Return-to-start threshold sweep
    print(f"\n  B) RETURN-TO-START THRESHOLD (current: hand offset < 5px from slit):")
    for threshold in [3, 5, 8, 10, 15, 20, 25, 30]:
        valid_offsets = df['hand_offset_at_end'].dropna()
        caught = (valid_offsets < threshold).sum()
        total_valid = len(valid_offsets)
        if threshold == 5:
            label = " <-- CURRENT"
        else:
            label = ""
        print(f"    < {threshold}px: {caught}/{total_valid} ({caught/max(total_valid,1)*100:.1f}%){label}")

    # Velocity-based detection
    print(f"\n  C) VELOCITY-BASED DETECTION (not in current algo):")
    valid_vel = df['mean_vel_before'].dropna()
    if len(valid_vel) > 0:
        for threshold in [-0.5, -1.0, -1.5, -2.0, -3.0, -5.0]:
            caught = (valid_vel < threshold).sum()
            print(f"    vel < {threshold}px/frame: {caught}/{len(valid_vel)} "
                  f"({caught/len(valid_vel)*100:.1f}%)")

    # Combined rule: what if we OR multiple relaxed conditions?
    print(f"\n  D) COMBINED RELAXED RULES (OR of multiple conditions):")

    combos = [
        ("Current algo",
         lambda r: r['retraction_trigger'] or r['return_trigger'] or r['hand_disappear']),
        ("Retract 25%+3px OR offset<10px OR disappear",
         lambda r: (r['extension_px'] >= MIN_EXTENSION and r['retraction_pct'] >= 25 and r['retraction_px'] >= 3)
         or (pd.notna(r['hand_offset_at_end']) and r['hand_offset_at_end'] < 10)
         or r['hand_disappear']),
        ("Retract 20%+2px OR offset<15px OR disappear",
         lambda r: (r['extension_px'] >= MIN_EXTENSION and r['retraction_pct'] >= 20 and r['retraction_px'] >= 2)
         or (pd.notna(r['hand_offset_at_end']) and r['hand_offset_at_end'] < 15)
         or r['hand_disappear']),
        ("Retract 15%+2px OR offset<20px OR disappear OR vel<-1.5",
         lambda r: (r['extension_px'] >= MIN_EXTENSION and r['retraction_pct'] >= 15 and r['retraction_px'] >= 2)
         or (pd.notna(r['hand_offset_at_end']) and r['hand_offset_at_end'] < 20)
         or r['hand_disappear']
         or (pd.notna(r['mean_vel_before']) and r['mean_vel_before'] < -1.5)),
    ]

    for name, rule in combos:
        caught = sum(1 for _, row in df.iterrows() if rule(row))
        print(f"    {name}:")
        print(f"      {caught}/{n} ({caught/n*100:.1f}%)")

    # --- Final assessment ---
    print(f"\n\n{'=' * 70}")
    print("ASSESSMENT: RULE-BASED vs DEEP LEARNING")
    print(f"{'=' * 70}")

    current_catch = df['any_trigger'].sum()
    current_pct = current_catch / n * 100

    # Best rule combo catch rate
    best_rule_caught = 0
    for _, row in df.iterrows():
        if ((row['extension_px'] >= MIN_EXTENSION and row['retraction_pct'] >= 15
             and row['retraction_px'] >= 2)
            or (pd.notna(row['hand_offset_at_end']) and row['hand_offset_at_end'] < 20)
            or row['hand_disappear']
            or (pd.notna(row['mean_vel_before']) and row['mean_vel_before'] < -1.5)):
            best_rule_caught += 1
    best_rule_pct = best_rule_caught / n * 100

    uncatchable = n - best_rule_caught
    uncatchable_pct = uncatchable / n * 100

    print(f"""
CURRENT STATE:
  Current algo thresholds catch {current_catch}/{n} ({current_pct:.1f}%) of split points
  This means the algo would correctly end the reach at only {current_pct:.1f}%
  of places where humans identify a reach boundary.

WITH RELAXED RULES:
  Best combination of relaxed thresholds catches {best_rule_caught}/{n} ({best_rule_pct:.1f}%)
  Remaining uncatchable by rules: {uncatchable}/{n} ({uncatchable_pct:.1f}%)

INTERPRETATION:""")

    if best_rule_pct >= 80:
        print(f"""
  STRONG EVIDENCE FOR RULE-BASED APPROACH.
  Relaxing thresholds alone catches {best_rule_pct:.0f}% of split points.
  The hand DOES retract or return to slit at most split points.
  The current thresholds are simply too conservative.

  RECOMMENDED:
  1. Lower retraction threshold from 40% to ~15-20%
  2. Increase return-to-start threshold from 5px to ~15-20px
  3. Add velocity-based detection (hand moving backward)
  4. Test on exhaustive GT to measure false-split rate
""")
    elif best_rule_pct >= 50:
        print(f"""
  MIXED EVIDENCE - Rules help but have limits.
  Relaxed thresholds catch {best_rule_pct:.0f}% but miss {uncatchable_pct:.0f}%.
  There IS a detectable signal at many split points, but not all.

  RECOMMENDED HYBRID APPROACH:
  1. Start with relaxed rule-based thresholds (quick win: +{best_rule_caught-current_catch} splits)
  2. For remaining {uncatchable_pct:.0f}%, consider a lightweight DL classifier:
     - Input: hand trajectory features in a window around the candidate split
     - Output: split/no-split binary classification
     - Training data: human-determined reach boundaries within merged algo reaches
  3. The rule-based step filters obvious splits; DL handles ambiguous cases
""")
    else:
        print(f"""
  STRONG EVIDENCE FOR DEEP LEARNING.
  Even aggressively relaxed rules only catch {best_rule_pct:.0f}% of split points.
  The hand does NOT consistently retract or return at split boundaries.
  Humans are using more complex cues (trajectory shape, timing, context).

  RECOMMENDED:
  1. Train a DL model on the full trajectory window around split points
  2. Features: hand x/y trajectory, velocity, acceleration, likelihood
  3. Positive examples: human-determined reach boundaries within merged reaches
  4. Negative examples: mid-reach frames that are NOT boundaries
  5. Architecture: 1D CNN or small transformer over ~20-frame windows
""")

    # Additional insight: what do uncatchable splits look like?
    if uncatchable > 0:
        uncatchable_df = df[~df.apply(
            lambda r: ((r['extension_px'] >= MIN_EXTENSION and r['retraction_pct'] >= 15
                        and r['retraction_px'] >= 2)
                       or (pd.notna(r['hand_offset_at_end']) and r['hand_offset_at_end'] < 20)
                       or r['hand_disappear']
                       or (pd.notna(r['mean_vel_before']) and r['mean_vel_before'] < -1.5)),
            axis=1
        )]

        print(f"\n  PROFILE OF UNCATCHABLE SPLITS (n={len(uncatchable_df)}):")
        print(f"    (These splits have no detectable retraction, return, disappearance, or velocity)")
        print(f"    Retraction %:   mean={uncatchable_df['retraction_pct'].mean():.1f}%, "
              f"median={uncatchable_df['retraction_pct'].median():.1f}%")
        uc_offsets = uncatchable_df['hand_offset_at_end'].dropna()
        if len(uc_offsets) > 0:
            print(f"    Hand offset:    mean={uc_offsets.mean():.1f}px, "
                  f"median={uc_offsets.median():.1f}px")
        print(f"    Gap frames:     mean={uncatchable_df['gap_frames'].mean():.1f}, "
              f"median={uncatchable_df['gap_frames'].median():.0f}")
        print(f"    Hand visible:   {(uncatchable_df['gap_min_hand_like'] >= 0.5).mean()*100:.0f}% "
              f"have hand visible throughout gap")
        uc_vel = uncatchable_df['mean_vel_before'].dropna()
        if len(uc_vel) > 0:
            print(f"    Velocity:       mean={uc_vel.mean():.2f}px/frame "
                  f"({'still extending!' if uc_vel.mean() > 0 else 'slightly retracting'})")
        print(f"\n    These likely require multi-frame trajectory pattern recognition -")
        print(f"    the hand doesn't clearly retract but the TRAJECTORY SHAPE changes")
        print(f"    (e.g. direction reversal, brief pause, or subtle position change).")


def main():
    print("REACH DETECTION vs SPLITTING ACCURACY ANALYSIS")
    print("=" * 70)
    print()

    # Part 1: Separate detection from splitting in stats
    merge_groups = compute_separated_stats()

    # Part 2: Analyze DLC features at split points
    split_features = analyze_split_points(merge_groups)

    # Part 3: Rule-based vs DL assessment
    assess_splitting_approach(split_features)

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
