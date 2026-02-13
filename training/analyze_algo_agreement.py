"""
Analyze algorithm AGREEMENT with human determinations.

Key question: "When a human says X, does the algorithm also say X?"

This does NOT require exhaustive GT. Every determined GT item is a human
asserting "this happened." We check whether the algorithm agrees.

What this CAN measure (valid without exhaustive):
  - Algo miss rate: Human says "reach here" but algo has nothing
  - Algo agreement: Human says "reach here" and algo has a match
  - Feature profiles of matched vs missed GT reaches
  - Outcome agreement: Human says "retrieved" and algo says...?

What this CANNOT measure (requires exhaustive):
  - False positives: "Algo detected something with no human determination nearby"
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_0_0")

RH_POINTS = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def analyze_reach_agreement():
    """For each human-determined reach, does the algorithm have a match?"""
    print("REACH DETECTION: HUMAN vs ALGORITHM AGREEMENT")
    print("=" * 70)
    print("Question: When a human says 'this is a reach', does the algo agree?\n")

    TOLERANCE = 5  # frames

    matched_reaches = []    # Human says reach, algo agrees
    missed_reaches = []     # Human says reach, algo has nothing
    per_video = defaultdict(lambda: {'matched': 0, 'missed': 0, 'gt_total': 0})

    for gt_file in sorted(DATA_DIR.glob("*_unified_ground_truth.json")):
        gt = load_json(gt_file)
        if not gt:
            continue

        video = gt['video_name']

        # Human-determined reaches
        gt_reaches = [r for r in gt.get('reaches', {}).get('reaches', [])
                      if r.get('start_determined', False) and r.get('end_determined', False)
                      and not r.get('exclude_from_analysis', False)]

        if not gt_reaches:
            continue

        # Algorithm reaches
        algo_file = ALGO_DIR / f"{video}_reaches.json"
        algo_data = load_json(algo_file)
        if not algo_data:
            continue

        # Flatten algo reaches from segments
        algo_reaches = []
        for seg in algo_data.get('segments', []):
            for r in seg.get('reaches', []):
                r['_seg_num'] = seg['segment_num']
                algo_reaches.append(r)

        # Match each GT reach to nearest algo reach
        algo_used = set()
        for gi, gr in enumerate(gt_reaches):
            gt_start = gr['start_frame']
            gt_end = gr['end_frame']
            gt_apex = gr.get('apex_frame')
            gt_duration = gt_end - gt_start + 1

            # Try matching by start frame proximity
            best_ai = None
            best_dist = float('inf')

            for ai, ar in enumerate(algo_reaches):
                if ai in algo_used:
                    continue
                algo_start = ar.get('start_frame', 0)
                # Match by start frame
                dist = abs(gt_start - algo_start)
                if dist <= TOLERANCE and dist < best_dist:
                    best_ai = ai
                    best_dist = dist

            # Also try matching by apex if available
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

            info = {
                'video': video,
                'gt_start': gt_start,
                'gt_end': gt_end,
                'gt_apex': gt_apex,
                'gt_duration': gt_duration,
                'gt_segment': gr.get('segment_num'),
            }

            if best_ai is not None:
                algo_used.add(best_ai)
                ar = algo_reaches[best_ai]
                info['algo_start'] = ar.get('start_frame', 0)
                info['algo_end'] = ar.get('end_frame', 0)
                info['algo_apex'] = ar.get('apex_frame')
                info['algo_duration'] = ar.get('end_frame', 0) - ar.get('start_frame', 0) + 1
                info['algo_extent'] = ar.get('max_extent_pixels', 0)
                info['start_offset'] = gt_start - ar.get('start_frame', 0)
                info['end_offset'] = gt_end - ar.get('end_frame', 0)
                info['match_dist'] = best_dist
                matched_reaches.append(info)
                per_video[video]['matched'] += 1
            else:
                missed_reaches.append(info)
                per_video[video]['missed'] += 1

            per_video[video]['gt_total'] += 1

    total = len(matched_reaches) + len(missed_reaches)
    match_rate = len(matched_reaches) / max(total, 1) * 100
    miss_rate = len(missed_reaches) / max(total, 1) * 100

    print(f"Total human-determined reaches: {total}")
    print(f"  Algorithm AGREES (has a match):  {len(matched_reaches)} ({match_rate:.1f}%)")
    print(f"  Algorithm MISSES (no match):     {len(missed_reaches)} ({miss_rate:.1f}%)")

    # Per-video breakdown
    print(f"\nPer-video agreement rates:")
    print(f"  {'Video':<35} {'GT':>5} {'Match':>6} {'Miss':>6} {'Rate':>7}")
    print(f"  {'-'*60}")
    for video in sorted(per_video.keys(),
                        key=lambda v: per_video[v]['matched'] / max(per_video[v]['gt_total'], 1)):
        v = per_video[video]
        rate = v['matched'] / max(v['gt_total'], 1) * 100
        print(f"  {video:<35} {v['gt_total']:>5} {v['matched']:>6} {v['missed']:>6} {rate:>6.0f}%")

    return matched_reaches, missed_reaches


def analyze_miss_characteristics(missed_reaches, matched_reaches):
    """What do algo-missed reaches look like vs algo-matched reaches?"""
    print(f"\n\n{'=' * 70}")
    print("WHAT DISTINGUISHES MATCHED vs MISSED REACHES?")
    print(f"{'=' * 70}")
    print("(Feature comparison requires DLC data loading)\n")

    # Duration comparison (available from GT data alone)
    matched_durations = [r['gt_duration'] for r in matched_reaches]
    missed_durations = [r['gt_duration'] for r in missed_reaches]

    print(f"DURATION (from GT annotations):")
    print(f"  Matched reaches:  mean={np.mean(matched_durations):.1f}, "
          f"median={np.median(matched_durations):.0f}, "
          f"min={np.min(matched_durations)}, max={np.max(matched_durations)}")
    print(f"  Missed reaches:   mean={np.mean(missed_durations):.1f}, "
          f"median={np.median(missed_durations):.0f}, "
          f"min={np.min(missed_durations)}, max={np.max(missed_durations)}")

    # Duration distribution
    print(f"\n  Duration distribution:")
    for threshold in [5, 10, 15, 20, 30, 50]:
        matched_pct = sum(1 for d in matched_durations if d <= threshold) / len(matched_durations) * 100
        missed_pct = sum(1 for d in missed_durations if d <= threshold) / len(missed_durations) * 100
        print(f"    <= {threshold:>2} frames:  matched={matched_pct:>5.1f}%  missed={missed_pct:>5.1f}%")

    # Timing offset for matched reaches
    if matched_reaches:
        start_offsets = [r['start_offset'] for r in matched_reaches]
        end_offsets = [r['end_offset'] for r in matched_reaches]

        print(f"\nTIMING OFFSETS (GT frame - Algo frame, for matched reaches):")
        print(f"  Start frame offset:  mean={np.mean(start_offsets):+.1f}, "
              f"median={np.median(start_offsets):+.0f}")
        print(f"  End frame offset:    mean={np.mean(end_offsets):+.1f}, "
              f"median={np.median(end_offsets):+.0f}")
        print(f"  (Positive = algo is early, Negative = algo is late)")

        # Which direction is the algo wrong?
        early_starts = sum(1 for o in start_offsets if o > 0)
        late_starts = sum(1 for o in start_offsets if o < 0)
        exact_starts = sum(1 for o in start_offsets if o == 0)
        print(f"\n  Start timing: {exact_starts} exact, "
              f"{early_starts} algo-early, {late_starts} algo-late")

        early_ends = sum(1 for o in end_offsets if o > 0)
        late_ends = sum(1 for o in end_offsets if o < 0)
        exact_ends = sum(1 for o in end_offsets if o == 0)
        print(f"  End timing:   {exact_ends} exact, "
              f"{early_ends} algo-early, {late_ends} algo-late")

    return matched_durations, missed_durations


def analyze_dlc_at_misses(missed_reaches):
    """Load DLC data at missed reach locations to understand WHY the algo missed.

    Checks the EXACT conditions the algorithm uses:
    1. Nose within 25px of slit center (POSITION, not just likelihood)
    2. Any hand point likelihood >= 0.5
    3. Both conditions met SIMULTANEOUSLY on the same frame
    """
    print(f"\n\n{'=' * 70}")
    print("DLC FEATURES AT ALGO-MISSED REACHES")
    print(f"{'=' * 70}")
    print("Simulating what the algorithm would see at each missed reach.\n")

    NOSE_ENGAGEMENT_THRESHOLD = 25  # pixels - must match algo
    HAND_LIKELIHOOD_THRESHOLD = 0.5

    # Group misses by video for efficient DLC loading
    by_video = defaultdict(list)
    for r in missed_reaches:
        by_video[r['video']].append(r)

    # Also load algo segments to check if reach falls within a segment
    miss_reasons = Counter()
    miss_details = []

    for video, reaches in sorted(by_video.items()):
        # Load DLC data
        dlc_files = list(DATA_DIR.glob(f"{video}DLC*.h5"))
        if not dlc_files:
            dlc_files = list(DATA_DIR.glob(f"*{video.split('_', 1)[1]}*DLC*.h5"))
        if not dlc_files:
            continue

        dlc_df = pd.read_hdf(dlc_files[0])
        scorer = dlc_df.columns.get_level_values(0)[0]

        # Load algo segments to check boundaries
        algo_seg_file = ALGO_DIR / f"{video}_segments.json"
        algo_seg_data = load_json(algo_seg_file)
        algo_segments = []
        if algo_seg_data:
            boundaries = algo_seg_data.get('boundaries', [])
            for i in range(len(boundaries) - 1):
                algo_segments.append((boundaries[i], boundaries[i + 1]))

        # Get slit center (same method as algorithm: median of BOXL, BOXR)
        try:
            boxl_x = dlc_df[(scorer, 'BOXL', 'x')].median()
            boxr_x = dlc_df[(scorer, 'BOXR', 'x')].median()
            boxl_y = dlc_df[(scorer, 'BOXL', 'y')].median()
            boxr_y = dlc_df[(scorer, 'BOXR', 'y')].median()
            slit_x = (boxl_x + boxr_x) / 2
            slit_y = (boxl_y + boxr_y) / 2
        except KeyError:
            continue

        for r in reaches:
            start = r['gt_start']
            end = r['gt_end']
            if start >= len(dlc_df) or end >= len(dlc_df):
                continue

            duration = end - start + 1

            # Per-frame analysis: simulate exactly what the algo checks
            n_hand_visible = 0
            n_nose_engaged = 0
            n_both = 0  # Key: frames where BOTH conditions met simultaneously
            nose_dists = []
            max_hand_likes = []

            for frame in range(start, min(end + 1, len(dlc_df))):
                # Hand check
                frame_max_hand = 0
                for part in RH_POINTS:
                    try:
                        l = dlc_df[(scorer, part, 'likelihood')].iloc[frame]
                        frame_max_hand = max(frame_max_hand, l)
                    except (KeyError, IndexError):
                        continue
                max_hand_likes.append(frame_max_hand)
                hand_ok = frame_max_hand >= HAND_LIKELIHOOD_THRESHOLD

                # Nose position check (same as algo's _is_nose_engaged)
                try:
                    nx = dlc_df[(scorer, 'Nose', 'x')].iloc[frame]
                    ny = dlc_df[(scorer, 'Nose', 'y')].iloc[frame]
                    nl = dlc_df[(scorer, 'Nose', 'likelihood')].iloc[frame]
                except (KeyError, IndexError):
                    nx, ny, nl = np.nan, np.nan, 0

                if nl >= 0.3 and not np.isnan(nx):
                    nose_dist = np.sqrt((nx - slit_x)**2 + (ny - slit_y)**2)
                    nose_ok = nose_dist < NOSE_ENGAGEMENT_THRESHOLD
                else:
                    nose_dist = np.nan
                    nose_ok = False

                nose_dists.append(nose_dist)

                if hand_ok:
                    n_hand_visible += 1
                if nose_ok:
                    n_nose_engaged += 1
                if hand_ok and nose_ok:
                    n_both += 1

            max_hand_likes = np.array(max_hand_likes)
            valid_nose_dists = [d for d in nose_dists if not np.isnan(d)]

            pct_hand = n_hand_visible / max(duration, 1) * 100
            pct_nose = n_nose_engaged / max(duration, 1) * 100
            pct_both = n_both / max(duration, 1) * 100
            mean_nose_dist = np.mean(valid_nose_dists) if valid_nose_dists else np.nan
            min_nose_dist = np.min(valid_nose_dists) if valid_nose_dists else np.nan
            peak_hand = np.max(max_hand_likes) if len(max_hand_likes) > 0 else 0

            # Check if reach falls within an algo segment
            in_segment = any(seg_s <= start and end <= seg_e
                             for seg_s, seg_e in algo_segments)

            detail = {
                'video': video,
                'start': start,
                'end': end,
                'duration': duration,
                'peak_hand_like': peak_hand,
                'mean_hand_like': np.mean(max_hand_likes) if len(max_hand_likes) > 0 else 0,
                'pct_hand_visible': pct_hand,
                'pct_nose_engaged': pct_nose,
                'pct_both_met': pct_both,
                'mean_nose_dist': mean_nose_dist,
                'min_nose_dist': min_nose_dist,
                'in_algo_segment': in_segment,
            }

            # Classify miss reason with FULL algo condition simulation
            if peak_hand < HAND_LIKELIHOOD_THRESHOLD:
                reason = 'HAND_INVISIBLE'
            elif pct_hand < 30:
                reason = 'HAND_BRIEFLY_VISIBLE'
            elif not in_segment:
                reason = 'OUTSIDE_ALGO_SEGMENT'
            elif pct_nose < 10:
                reason = 'NOSE_NOT_ENGAGED'  # Nose too far from slit
            elif pct_both < 10:
                reason = 'NO_SIMULTANEOUS_CONDITIONS'  # Hand and nose never overlap
            elif pct_both < 50:
                reason = 'BRIEF_OVERLAP'  # Conditions briefly overlap
            elif duration <= 4:
                reason = 'TOO_SHORT_FOR_ALGO'
            else:
                reason = 'ALGO_SHOULD_DETECT'  # Conditions look met - true logic gap

            detail['reason'] = reason
            miss_reasons[reason] += 1
            miss_details.append(detail)

    total_analyzed = len(miss_details)
    print(f"Analyzed {total_analyzed} missed reaches with DLC data\n")

    print(f"MISS REASON BREAKDOWN:")
    print(f"  {'Reason':<25} {'Count':>6} {'Pct':>7}")
    print(f"  {'-'*40}")
    for reason, count in miss_reasons.most_common():
        pct = count / max(total_analyzed, 1) * 100
        print(f"  {reason:<25} {count:>6} {pct:>6.1f}%")

    # Feature profiles per reason
    print(f"\nFEATURE PROFILES BY MISS REASON:")
    all_reasons = ['NOSE_NOT_ENGAGED', 'NO_SIMULTANEOUS_CONDITIONS', 'BRIEF_OVERLAP',
                   'OUTSIDE_ALGO_SEGMENT', 'HAND_INVISIBLE', 'HAND_BRIEFLY_VISIBLE',
                   'TOO_SHORT_FOR_ALGO', 'ALGO_SHOULD_DETECT']
    for reason in all_reasons:
        details = [d for d in miss_details if d['reason'] == reason]
        if not details:
            continue

        durations = [d['duration'] for d in details]
        pct_hand = [d['pct_hand_visible'] for d in details]
        pct_nose = [d['pct_nose_engaged'] for d in details]
        pct_both = [d['pct_both_met'] for d in details]
        nose_dists = [d['mean_nose_dist'] for d in details if not np.isnan(d['mean_nose_dist'])]

        print(f"\n  {reason} (n={len(details)}):")
        print(f"    Duration:          mean={np.mean(durations):.1f}, median={np.median(durations):.0f}")
        print(f"    %frames hand>=0.5: mean={np.mean(pct_hand):.1f}%")
        print(f"    %frames nose<25px: mean={np.mean(pct_nose):.1f}%")
        print(f"    %frames BOTH met:  mean={np.mean(pct_both):.1f}%")
        if nose_dists:
            print(f"    Mean nose-slit dist: {np.mean(nose_dists):.1f}px")

    # Actionable recommendations
    print(f"\n\n{'=' * 70}")
    print("ACTIONABLE INSIGHTS (from human-perspective analysis)")
    print(f"{'=' * 70}")

    nose_ne = miss_reasons.get('NOSE_NOT_ENGAGED', 0)
    no_simul = miss_reasons.get('NO_SIMULTANEOUS_CONDITIONS', 0)
    brief_ov = miss_reasons.get('BRIEF_OVERLAP', 0)
    outside = miss_reasons.get('OUTSIDE_ALGO_SEGMENT', 0)
    hand_invisible_n = miss_reasons.get('HAND_INVISIBLE', 0)
    hand_brief_n = miss_reasons.get('HAND_BRIEFLY_VISIBLE', 0)
    short_n = miss_reasons.get('TOO_SHORT_FOR_ALGO', 0)
    should_detect = miss_reasons.get('ALGO_SHOULD_DETECT', 0)

    t = max(total_analyzed, 1)
    print(f"""
1. NOSE_NOT_ENGAGED ({nose_ne} misses, {nose_ne/t*100:.0f}%):
   Nose is too far from slit center (>25px) during the reach.
   The mouse may not be in "reaching position" despite hand being visible.
   FIX: Increase NOSE_ENGAGEMENT_THRESHOLD or use looser engagement check.

2. NO_SIMULTANEOUS_CONDITIONS ({no_simul} misses, {no_simul/t*100:.0f}%):
   Hand is visible AND nose is engaged at different times, but never the same frame.
   The algo requires both on the SAME frame to trigger.
   FIX: Allow brief temporal gap between conditions (e.g. nose engaged
   within last N frames + hand visible now).

3. BRIEF_OVERLAP ({brief_ov} misses, {brief_ov/t*100:.0f}%):
   Conditions overlap on <50% of frames. Algo may start reach but end it
   too quickly, or the overlap is too brief for MIN_REACH_DURATION.
   FIX: More lenient state transitions; once reaching starts, don't
   require continuous nose engagement.

4. OUTSIDE_ALGO_SEGMENT ({outside} misses, {outside/t*100:.0f}%):
   Reach falls outside algorithm's detected segment boundaries.
   The algo only looks for reaches within segments.
   FIX: Ensure segment boundaries encompass all reach activity.

5. HAND_INVISIBLE ({hand_invisible_n} misses, {hand_invisible_n/t*100:.0f}%):
   DLC never tracks the hand above 0.5 likelihood.
   This is a DLC tracking failure, not an algorithm issue.

6. HAND_BRIEFLY_VISIBLE ({hand_brief_n} misses, {hand_brief_n/t*100:.0f}%):
   Hand crosses threshold briefly but not enough for algo.
   FIX: Better gap-bridging for flickering hand tracking.

7. TOO_SHORT_FOR_ALGO ({short_n} misses, {short_n/t*100:.0f}%):
   Reach shorter than MIN_REACH_DURATION (4 frames).

8. ALGO_SHOULD_DETECT ({should_detect} misses, {should_detect/t*100:.0f}%):
   All conditions appear met (hand visible, nose engaged, >50% overlap).
   These are TRUE logic gaps - the algo should find these but doesn't.
   May be reach-end detection cutting reaches short, or post-processing
   filtering (extent threshold, split logic).
   HIGHEST PRIORITY for investigation.
""")

    return miss_details


def analyze_outcome_agreement():
    """For each human-determined outcome, does the algorithm agree?"""
    print(f"\n\n{'=' * 70}")
    print("OUTCOME CLASSIFICATION: HUMAN vs ALGORITHM AGREEMENT")
    print(f"{'=' * 70}")
    print("Question: When a human says outcome=X, does the algo also say X?\n")

    agreements = []
    disagreements = []
    confusion = defaultdict(lambda: defaultdict(int))

    for gt_file in sorted(DATA_DIR.glob("*_unified_ground_truth.json")):
        gt = load_json(gt_file)
        if not gt:
            continue
        video = gt['video_name']

        algo_file = ALGO_DIR / f"{video}_pellet_outcomes.json"
        algo_data = load_json(algo_file)
        if not algo_data:
            continue

        gt_outcomes = {s['segment_num']: s for s in gt.get('outcomes', {}).get('segments', [])
                       if s.get('determined', False)}
        algo_outcomes = {s['segment_num']: s for s in algo_data.get('segments', [])}

        for seg_num, gt_seg in gt_outcomes.items():
            algo_seg = algo_outcomes.get(seg_num)
            if algo_seg is None:
                continue

            gt_out = gt_seg['outcome']
            algo_out = algo_seg['outcome']
            confusion[gt_out][algo_out] += 1

            info = {
                'video': video,
                'segment': seg_num,
                'gt_outcome': gt_out,
                'algo_outcome': algo_out,
            }

            if gt_out == algo_out:
                agreements.append(info)
            else:
                disagreements.append(info)

    total = len(agreements) + len(disagreements)
    agree_rate = len(agreements) / max(total, 1) * 100

    print(f"Total human-determined outcomes: {total}")
    print(f"  Algorithm AGREES: {len(agreements)} ({agree_rate:.1f}%)")
    print(f"  Algorithm DISAGREES: {len(disagreements)} ({100-agree_rate:.1f}%)")

    # Confusion matrix
    all_outcomes = sorted(set(confusion.keys()) | set(k for v in confusion.values() for k in v))

    print(f"\nConfusion matrix (rows = human says, cols = algo says):")
    header = f"  {'Human says':<18}"
    for o in all_outcomes:
        header += f"{o[:14]:<16}"
    print(header)
    print(f"  {'-' * (18 + 16 * len(all_outcomes))}")

    for gt_out in all_outcomes:
        row = f"  {gt_out:<18}"
        for algo_out in all_outcomes:
            count = confusion[gt_out][algo_out]
            if gt_out == algo_out:
                row += f"[{count}]{'':>{14-len(str(count))}}"
            else:
                row += f"{count:<16}"
        print(row)

    # Disagreement patterns
    if disagreements:
        print(f"\nDisagreement patterns (human -> algo):")
        patterns = Counter((d['gt_outcome'], d['algo_outcome']) for d in disagreements)
        for (gt, algo), count in patterns.most_common():
            print(f"  Human says '{gt}' → Algo says '{algo}': {count} times")

    return agreements, disagreements


def main():
    print("ALGORITHM AGREEMENT ANALYSIS")
    print("=" * 70)
    print("Perspective: Starting from what HUMANS determined, checking algo agreement.")
    print("This is valid for ALL determined GT items (exhaustive flag not required).\n")

    # Reach agreement
    matched, missed = analyze_reach_agreement()

    # Characteristics of matches vs misses
    analyze_miss_characteristics(missed, matched)

    # DLC features at misses
    analyze_dlc_at_misses(missed)

    # Outcome agreement
    analyze_outcome_agreement()

    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    total_reaches = len(matched) + len(missed)
    print(f"""
Reach Detection:
  {len(matched)}/{total_reaches} human-determined reaches matched by algo ({len(matched)/max(total_reaches,1)*100:.1f}%)
  {len(missed)}/{total_reaches} human-determined reaches MISSED by algo ({len(missed)/max(total_reaches,1)*100:.1f}%)

Key insight: The algo's miss rate tells us about recall-like performance
WITHOUT needing exhaustive GT. These are confirmed real reaches that the
algorithm failed to detect.

The OPEN QUESTION (still requires exhaustive GT):
  How many of the algo's OTHER detections are false positives?
  Until exhaustive GT exists, we only know about misses, not false alarms.
""")


if __name__ == "__main__":
    main()
