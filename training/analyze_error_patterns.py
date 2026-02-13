"""
Detailed error pattern analysis for each of the 4 reach accuracy elements.

For each error case, extract the DLC features to understand WHY the algo
got it wrong, and what specific fix would address it.

Elements:
  1. Existence (45.9% unmatched) - downstream of 3
  2. Start frame (11.4% wrong) - 93 early, 9 late
  3. End frame (53.9% wrong) - 469 late, 13 early
  4. Linking - downstream of 1+3
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

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


def get_best_hand(dlc_df, scorer, frame):
    """Get best hand x, y, likelihood at a frame."""
    best_x, best_y, best_like = None, None, 0
    max_like = 0
    for part in RH_POINTS:
        try:
            x = dlc_df[(scorer, part, 'x')].iloc[frame]
            y = dlc_df[(scorer, part, 'y')].iloc[frame]
            l = dlc_df[(scorer, part, 'likelihood')].iloc[frame]
        except (KeyError, IndexError):
            continue
        max_like = max(max_like, l)
        if l > best_like and l >= HAND_LIKELIHOOD_THRESHOLD:
            best_x, best_y, best_like = x, y, l
    return best_x, best_y, best_like, max_like


def match_reaches(gt_reaches, algo_reaches, max_dist=30):
    """Match GT reaches to algo reaches by start frame proximity (1:1 greedy)."""
    candidates = []
    for gi, gr in enumerate(gt_reaches):
        for ai, ar in enumerate(algo_reaches):
            dist = abs(gr['start_frame'] - ar.get('start_frame', 0))
            if dist <= max_dist:
                candidates.append((dist, gi, ai))
    candidates.sort()
    gt_used, algo_used = set(), set()
    matches = []
    for dist, gi, ai in candidates:
        if gi not in gt_used and ai not in algo_used:
            gt_used.add(gi)
            algo_used.add(ai)
            matches.append((gi, ai, dist))
    return matches, set(range(len(gt_reaches))) - gt_used


# ============================================================================
# ELEMENT 3: END FRAME ERRORS (highest priority)
# ============================================================================

def analyze_end_frame_errors():
    """Deep dive into why algo ends reaches late.

    For each algo-late end, examine:
    - What is the hand doing at GT end frame?
    - Why didn't the algo's end conditions trigger?
    - What specific threshold change would fix it?
    """
    print("=" * 70)
    print("ELEMENT 3: END FRAME ERROR ANALYSIS")
    print("=" * 70)
    print()
    print("For each reach where algo ends late, what is happening")
    print("at the GT end frame that the algo misses?")
    print()

    # Categorize end errors by root cause
    error_causes = Counter()
    cause_details = defaultdict(list)

    n_analyzed = 0
    n_exact = 0
    n_late = 0
    n_early = 0

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

        matches, _ = match_reaches(gt_reaches, algo_reaches)

        for gi, ai, start_dist in matches:
            gr = gt_reaches[gi]
            ar = algo_reaches[ai]

            gt_end = gr['end_frame']
            algo_end = ar.get('end_frame', 0)
            end_offset = algo_end - gt_end

            n_analyzed += 1
            if end_offset == 0:
                n_exact += 1
                continue
            elif end_offset < 0:
                n_early += 1
                # We'll focus on late for now
                continue

            n_late += 1

            # === Analyze what's happening at GT end frame ===
            gt_start = gr['start_frame']

            # Max extension during reach
            reach_max_x = 0
            for f in range(gt_start, min(gt_end + 1, len(dlc_df))):
                hx, _, _, _ = get_best_hand(dlc_df, scorer, f)
                if hx is not None and hx > reach_max_x:
                    reach_max_x = hx

            extension = reach_max_x - slit_x

            # Hand at GT end
            hand_x_end, _, hand_like_end, max_like_end = get_best_hand(dlc_df, scorer, gt_end)
            hand_offset_end = (hand_x_end - slit_x) if hand_x_end else None
            retraction_px = (reach_max_x - hand_x_end) if (reach_max_x and hand_x_end) else 0
            retraction_pct = retraction_px / max(extension, 0.1) * 100

            # Hand at GT end + 1 (first frame human says "not reach")
            if gt_end + 1 < len(dlc_df):
                hand_x_post, _, hand_like_post, max_like_post = get_best_hand(dlc_df, scorer, gt_end + 1)
            else:
                hand_x_post, hand_like_post, max_like_post = None, 0, 0

            # Hand velocity at GT end (hand_x[end] - hand_x[end-1])
            if gt_end > 0:
                hand_x_prev, _, _, _ = get_best_hand(dlc_df, scorer, gt_end - 1)
                velocity_at_end = (hand_x_end - hand_x_prev) if (hand_x_end and hand_x_prev) else None
            else:
                velocity_at_end = None

            # Check: does hand disappear within 3 frames of GT end?
            hand_disappears_soon = False
            frames_to_disappear = None
            for f in range(gt_end + 1, min(gt_end + 6, len(dlc_df))):
                _, _, _, ml = get_best_hand(dlc_df, scorer, f)
                if ml < HAND_LIKELIHOOD_THRESHOLD:
                    hand_disappears_soon = True
                    frames_to_disappear = f - gt_end
                    break

            # Check: does hand reappear after disappearing?
            hand_reappears = False
            if hand_disappears_soon and frames_to_disappear:
                disappear_frame = gt_end + frames_to_disappear
                for f in range(disappear_frame + 1, min(disappear_frame + 15, len(dlc_df))):
                    _, _, _, ml = get_best_hand(dlc_df, scorer, f)
                    if ml >= HAND_LIKELIHOOD_THRESHOLD:
                        hand_reappears = True
                        break

            # Nose at GT end
            try:
                nx = dlc_df[(scorer, 'Nose', 'x')].iloc[gt_end]
                ny = dlc_df[(scorer, 'Nose', 'y')].iloc[gt_end]
                nl = dlc_df[(scorer, 'Nose', 'likelihood')].iloc[gt_end]
                nose_dist = np.sqrt((nx - slit_x)**2 + (ny - slit_y)**2) if nl >= 0.3 else None
            except (KeyError, IndexError):
                nose_dist = None

            # === Classify the cause ===
            # Why didn't the algo end here?

            hand_visible_at_end = max_like_end >= HAND_LIKELIHOOD_THRESHOLD
            hand_invisible_post = max_like_post < HAND_LIKELIHOOD_THRESHOLD if max_like_post is not None else True

            if not hand_visible_at_end:
                cause = 'HAND_ALREADY_GONE'
                # Hand invisible at GT end but algo hasn't noticed yet
            elif hand_invisible_post and not hand_reappears:
                cause = 'DISAPPEAR_DELAY'
                # Hand disappears right after GT end, algo has a delay
            elif hand_disappears_soon and hand_reappears:
                cause = 'DISAPPEAR_REAPPEAR'
                # Hand disappears then comes back - algo re-extends the reach
            elif retraction_pct >= 40 and retraction_px >= 5:
                cause = 'RETRACTION_MET_BUT_MISSED'
                # Algo's own threshold IS met but somehow didn't trigger
            elif retraction_pct >= 20:
                cause = 'RETRACTION_PARTIAL'
                # Hand partially retracted (20-40%) - below algo's 40% threshold
            elif velocity_at_end is not None and velocity_at_end < -1.0:
                cause = 'VELOCITY_REVERSAL'
                # Hand is moving backward but retraction not enough for threshold
            elif hand_offset_end is not None and hand_offset_end < 5:
                cause = 'HAND_NEAR_SLIT'
                # Hand close to slit but algo's 5px return threshold not met
            elif hand_offset_end is not None and hand_offset_end < 15:
                cause = 'HAND_PARTIALLY_RETURNED'
                # Hand somewhat close to slit (5-15px)
            else:
                cause = 'HAND_STILL_EXTENDED'
                # Hand still far from slit, still visible - hardest case

            error_causes[cause] += 1
            cause_details[cause].append({
                'video': video,
                'gt_end': gt_end,
                'algo_end': algo_end,
                'end_offset': end_offset,
                'extension_px': extension,
                'retraction_pct': retraction_pct,
                'retraction_px': retraction_px,
                'hand_offset_end': hand_offset_end,
                'hand_like_end': hand_like_end,
                'max_like_end': max_like_end,
                'velocity_at_end': velocity_at_end,
                'hand_disappears_soon': hand_disappears_soon,
                'frames_to_disappear': frames_to_disappear,
                'hand_reappears': hand_reappears,
                'nose_dist': nose_dist,
            })

    # Report
    print(f"Analyzed: {n_analyzed} matched reaches")
    print(f"  Exact end:  {n_exact} ({n_exact/max(n_analyzed,1)*100:.1f}%)")
    print(f"  Algo early: {n_early} ({n_early/max(n_analyzed,1)*100:.1f}%)")
    print(f"  Algo late:  {n_late} ({n_late/max(n_analyzed,1)*100:.1f}%)")
    print()

    print(f"ALGO-LATE END FRAME CAUSES (n={n_late}):")
    print(f"  {'Cause':<30} {'Count':>5} {'Pct':>6}  Fix")
    print(f"  {'-'*80}")

    fix_map = {
        'DISAPPEAR_REAPPEAR': 'New reach on reappearance, not continuation',
        'RETRACTION_PARTIAL': 'Lower retraction threshold from 40% to ~20%',
        'HAND_PARTIALLY_RETURNED': 'Increase return threshold from 5px to ~15px',
        'VELOCITY_REVERSAL': 'Add velocity-based end detection',
        'HAND_NEAR_SLIT': 'Increase return threshold from 5px to ~10px',
        'DISAPPEAR_DELAY': 'Reduce DISAPPEAR_THRESHOLD from 2 to 1 frame',
        'RETRACTION_MET_BUT_MISSED': 'Bug: algo threshold met but not triggered - investigate',
        'HAND_ALREADY_GONE': 'Algo end-detection has latency - reduce it',
        'HAND_STILL_EXTENDED': 'Hardest case - may need trajectory/context analysis',
    }

    for cause, count in error_causes.most_common():
        pct = count / max(n_late, 1) * 100
        fix = fix_map.get(cause, '?')
        print(f"  {cause:<30} {count:>5} {pct:>5.1f}%  {fix}")

    # Detailed stats per cause
    print(f"\n\nDETAILED PROFILES PER CAUSE:")
    for cause in [c for c, _ in error_causes.most_common()]:
        details = cause_details[cause]
        if not details:
            continue

        offsets = [d['end_offset'] for d in details]
        retr_pcts = [d['retraction_pct'] for d in details]
        hand_offs = [d['hand_offset_end'] for d in details if d['hand_offset_end'] is not None]
        vels = [d['velocity_at_end'] for d in details if d['velocity_at_end'] is not None]

        print(f"\n  {cause} (n={len(details)}):")
        print(f"    End offset:    mean={np.mean(offsets):+.1f}, median={np.median(offsets):+.0f}")
        print(f"    Retraction %:  mean={np.mean(retr_pcts):.1f}%, median={np.median(retr_pcts):.1f}%")
        if hand_offs:
            print(f"    Hand offset:   mean={np.mean(hand_offs):.1f}px, median={np.median(hand_offs):.1f}px")
        if vels:
            print(f"    Velocity:      mean={np.mean(vels):.2f}px/f")

    return error_causes, cause_details


# ============================================================================
# ELEMENT 2: START FRAME ERRORS
# ============================================================================

def analyze_start_frame_errors():
    """Deep dive into why algo starts reaches wrong.

    For the 93 algo-early and 9 algo-late cases:
    - What's at the algo start frame that's NOT at the GT start?
    - What's at the GT start frame that the algo missed?
    """
    print(f"\n\n{'=' * 70}")
    print("ELEMENT 2: START FRAME ERROR ANALYSIS")
    print(f"{'=' * 70}")
    print()

    early_cases = []
    late_cases = []
    n_analyzed = 0

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

        matches, _ = match_reaches(gt_reaches, algo_reaches)

        for gi, ai, start_dist in matches:
            gr = gt_reaches[gi]
            ar = algo_reaches[ai]
            gt_start = gr['start_frame']
            algo_start = ar.get('start_frame', 0)
            offset = algo_start - gt_start  # negative = algo early

            n_analyzed += 1
            if offset == 0:
                continue

            # What's happening at both frames?
            def frame_state(frame):
                if frame < 0 or frame >= len(dlc_df):
                    return None
                hx, hy, hl, ml = get_best_hand(dlc_df, scorer, frame)
                try:
                    nx = dlc_df[(scorer, 'Nose', 'x')].iloc[frame]
                    ny = dlc_df[(scorer, 'Nose', 'y')].iloc[frame]
                    nl = dlc_df[(scorer, 'Nose', 'likelihood')].iloc[frame]
                    nd = np.sqrt((nx - slit_x)**2 + (ny - slit_y)**2) if nl >= 0.3 else None
                except (KeyError, IndexError):
                    nd = None
                return {
                    'hand_x': hx, 'hand_like': hl, 'max_like': ml,
                    'hand_offset': (hx - slit_x) if hx else None,
                    'hand_visible': ml >= HAND_LIKELIHOOD_THRESHOLD,
                    'nose_dist': nd,
                    'nose_engaged': nd is not None and nd < NOSE_ENGAGEMENT_THRESHOLD,
                }

            algo_state = frame_state(algo_start)
            gt_state = frame_state(gt_start)

            # Also check the frame BEFORE gt_start (should NOT be reach)
            pre_gt_state = frame_state(gt_start - 1)

            info = {
                'video': video,
                'gt_start': gt_start,
                'algo_start': algo_start,
                'offset': offset,
                'algo_state': algo_state,
                'gt_state': gt_state,
                'pre_gt_state': pre_gt_state,
            }

            if offset < 0:
                early_cases.append(info)
            else:
                late_cases.append(info)

    # Report early cases
    print(f"Analyzed: {n_analyzed} matched reaches")
    print(f"  Algo-early starts: {len(early_cases)}")
    print(f"  Algo-late starts:  {len(late_cases)}")

    if early_cases:
        print(f"\n  ALGO-EARLY STARTS (algo triggers before human says reach starts):")
        offsets = [c['offset'] for c in early_cases]
        print(f"    Offset: mean={np.mean(offsets):.1f}, median={np.median(offsets):.0f}")
        print(f"    Distribution:")
        dist = Counter(offsets)
        for o in sorted(dist.keys()):
            print(f"      {o:+3d} frames: {dist[o]}")

        # What's at the algo start frame?
        print(f"\n    At ALGO start frame (where algo triggers too early):")
        hand_vis = [c['algo_state']['hand_visible'] for c in early_cases if c['algo_state']]
        nose_eng = [c['algo_state']['nose_engaged'] for c in early_cases if c['algo_state']]
        print(f"      Hand visible: {sum(hand_vis)}/{len(hand_vis)} ({sum(hand_vis)/max(len(hand_vis),1)*100:.0f}%)")
        print(f"      Nose engaged: {sum(nose_eng)}/{len(nose_eng)} ({sum(nose_eng)/max(len(nose_eng),1)*100:.0f}%)")

        hand_likes = [c['algo_state']['max_like'] for c in early_cases
                      if c['algo_state'] and c['algo_state']['max_like'] is not None]
        if hand_likes:
            print(f"      Hand likelihood: mean={np.mean(hand_likes):.2f}, median={np.median(hand_likes):.2f}")

        # What's at the frame BEFORE GT start? (should NOT meet conditions)
        print(f"\n    At frame BEFORE GT start (should NOT be reach):")
        pre_vis = [c['pre_gt_state']['hand_visible'] for c in early_cases if c['pre_gt_state']]
        pre_nose = [c['pre_gt_state']['nose_engaged'] for c in early_cases if c['pre_gt_state']]
        print(f"      Hand visible: {sum(pre_vis)}/{len(pre_vis)} ({sum(pre_vis)/max(len(pre_vis),1)*100:.0f}%)")
        print(f"      Nose engaged: {sum(pre_nose)}/{len(pre_nose)} ({sum(pre_nose)/max(len(pre_nose),1)*100:.0f}%)")

        # Classify early-start causes
        causes = Counter()
        for c in early_cases:
            a = c['algo_state']
            p = c['pre_gt_state']
            if a and a['hand_visible'] and p and p['hand_visible']:
                # Hand was visible BEFORE the human says reach starts
                # The algo caught it but the human says it's not a reach yet
                causes['HAND_VISIBLE_PRE_REACH'] += 1
            elif a and a['hand_visible'] and (not p or not p['hand_visible']):
                # Hand appeared at algo_start, human says reach starts later
                causes['EARLY_HAND_APPEARANCE'] += 1
            elif a and not a['hand_visible']:
                causes['ALGO_FALSE_START'] += 1
            else:
                causes['OTHER'] += 1

        print(f"\n    Early-start causes:")
        for cause, count in causes.most_common():
            print(f"      {cause}: {count}")

        if causes.get('HAND_VISIBLE_PRE_REACH', 0) > 0:
            print(f"\n    HAND_VISIBLE_PRE_REACH: Hand is tracked before human says reach starts.")
            print(f"    This means the hand was near/at the slit but the human decided")
            print(f"    the actual reach hadn't begun yet. Possible causes:")
            print(f"      - Hand hovering/resting near slit (not reaching)")
            print(f"      - Previous reach's retraction being counted as new reach start")
            print(f"      - DLC tracking a different body part as hand")

    if late_cases:
        print(f"\n  ALGO-LATE STARTS (algo triggers after human says reach starts):")
        offsets = [c['offset'] for c in late_cases]
        print(f"    Offset: mean={np.mean(offsets):.1f}, median={np.median(offsets):.0f}")

        # What's at GT start that algo doesn't see?
        print(f"\n    At GT start frame (where human sees reach but algo doesn't yet):")
        gt_vis = [c['gt_state']['hand_visible'] for c in late_cases if c['gt_state']]
        gt_nose = [c['gt_state']['nose_engaged'] for c in late_cases if c['gt_state']]
        print(f"      Hand visible: {sum(gt_vis)}/{len(gt_vis)}")
        print(f"      Nose engaged: {sum(gt_nose)}/{len(gt_nose)}")
        if gt_vis:
            not_vis = sum(1 for v in gt_vis if not v)
            print(f"      Hand NOT visible: {not_vis} - algo can't start without hand")


# ============================================================================
# ELEMENT 1+4: PROJECTED IMPROVEMENT
# ============================================================================

def project_improvement():
    """If we fix end frames, how much do existence and linking improve?

    Simulate: for each unmatched GT reach, if the PREVIOUS algo reach
    had ended at the correct GT end frame, would this GT reach now match?
    """
    print(f"\n\n{'=' * 70}")
    print("PROJECTED IMPROVEMENT: If end frames are fixed")
    print(f"{'=' * 70}")
    print()
    print("If the algo ended each reach at the human-determined end frame,")
    print("how many currently-unmatched GT reaches would become matchable?")
    print()

    total_gt = 0
    currently_matched = 0
    would_match_with_fix = 0
    still_unmatched = 0

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

        # Current matching
        matches, unmatched_gi = match_reaches(gt_reaches, algo_reaches)
        total_gt += len(gt_reaches)
        currently_matched += len(matches)

        # For unmatched GT reaches: would they match if previous algo reach
        # ended correctly? Simulate by checking if an algo reach's START
        # is close to this GT reach's start IF the algo reach weren't merged.

        # Approach: check if this GT reach overlaps with an algo reach
        # that ALSO overlaps with a previous GT reach. If so, splitting
        # that algo reach would free up a match.
        for gi in unmatched_gi:
            gr = gt_reaches[gi]
            g_start = gr['start_frame']
            g_end = gr['end_frame']

            # Is this GT reach inside any algo reach?
            inside_algo = False
            for ar in algo_reaches:
                a_start = ar.get('start_frame', 0)
                a_end = ar.get('end_frame', 0)
                if a_start <= g_start and g_end <= a_end:
                    inside_algo = True
                    break

            if inside_algo:
                # If the algo had correctly split, this reach would match
                would_match_with_fix += 1
            else:
                still_unmatched += 1

    projected_matched = currently_matched + would_match_with_fix
    print(f"Total GT reaches:          {total_gt}")
    print(f"Currently matched:         {currently_matched} ({currently_matched/max(total_gt,1)*100:.1f}%)")
    print(f"Would match with fix:      {would_match_with_fix}")
    print(f"Still unmatched:           {still_unmatched}")
    print(f"PROJECTED matched:         {projected_matched} ({projected_matched/max(total_gt,1)*100:.1f}%)")
    print(f"\nImprovement: {currently_matched/max(total_gt,1)*100:.1f}% -> {projected_matched/max(total_gt,1)*100:.1f}%")
    print(f"  Existence error would drop from {(total_gt-currently_matched)/max(total_gt,1)*100:.1f}% to {still_unmatched/max(total_gt,1)*100:.1f}%")


def main():
    # Element 3 first (highest priority)
    end_causes, end_details = analyze_end_frame_errors()

    # Element 2
    analyze_start_frame_errors()

    # Elements 1+4: projected improvement
    project_improvement()

    # Final summary
    print(f"\n\n{'=' * 70}")
    print("ACTION PLAN SUMMARY")
    print(f"{'=' * 70}")

    total_late = sum(end_causes.values())
    print(f"""
ELEMENT 3 (END FRAME) - {total_late} algo-late errors to fix:
""")

    # Group fixes by type
    threshold_fixes = ['RETRACTION_PARTIAL', 'HAND_NEAR_SLIT', 'HAND_PARTIALLY_RETURNED']
    reappear_fix = ['DISAPPEAR_REAPPEAR']
    delay_fix = ['DISAPPEAR_DELAY', 'HAND_ALREADY_GONE']
    hard_cases = ['HAND_STILL_EXTENDED', 'VELOCITY_REVERSAL', 'RETRACTION_MET_BUT_MISSED']

    threshold_n = sum(end_causes.get(c, 0) for c in threshold_fixes)
    reappear_n = sum(end_causes.get(c, 0) for c in reappear_fix)
    delay_n = sum(end_causes.get(c, 0) for c in delay_fix)
    hard_n = sum(end_causes.get(c, 0) for c in hard_cases)

    print(f"  FIX 1 - Threshold relaxation ({threshold_n} cases, {threshold_n/max(total_late,1)*100:.0f}%):")
    print(f"    Lower retraction threshold: 40% -> ~20%")
    print(f"    Increase return-to-start: 5px -> ~15px")
    print(f"    These are pure parameter changes in reach_detector.py")
    print()
    print(f"  FIX 2 - Disappear-reappear logic ({reappear_n} cases, {reappear_n/max(total_late,1)*100:.0f}%):")
    print(f"    When hand disappears then reappears: START NEW REACH")
    print(f"    Currently algo re-extends the same reach")
    print(f"    This is a state machine logic change")
    print()
    print(f"  FIX 3 - Disappear delay ({delay_n} cases, {delay_n/max(total_late,1)*100:.0f}%):")
    print(f"    Reduce latency in detecting hand disappearance")
    print()
    print(f"  HARD CASES ({hard_n} cases, {hard_n/max(total_late,1)*100:.0f}%):")
    print(f"    Hand still extended and visible - may need velocity or")
    print(f"    trajectory-based end detection")


if __name__ == "__main__":
    main()
