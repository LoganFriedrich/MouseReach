"""
Comprehensive analysis of remaining v4.2 errors.

Analyzes ALL error types:
1. Early-end mechanism (retraction vs return-to-start)
2. Missed GT reaches (35)
3. False positive algo reaches (566)
4. Late-end reaches

Uses DLC data to understand what the algo sees vs what humans decide.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_5_0")

RH_POINTS = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']
HAND_THRESHOLD = 0.5


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def match_reaches(gt_reaches, algo_reaches, max_dist=30):
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
    return matches, set(range(len(gt_reaches))) - gt_used, set(range(len(algo_reaches))) - algo_used


def get_hand_info(df, frame):
    """Get hand visibility and position at a specific frame."""
    if frame < 0 or frame >= len(df):
        return {'visible': False, 'x': None, 'likelihood': 0}
    row = df.iloc[frame]
    best_x, best_l = None, 0
    for p in RH_POINTS:
        l = row.get(f'{p}_likelihood', 0)
        if l > best_l:
            best_l = l
            best_x = row.get(f'{p}_x', None)
    return {'visible': best_l >= HAND_THRESHOLD, 'x': best_x, 'likelihood': best_l}


def get_slit_center(df, seg_start, seg_end):
    segment_df = df.iloc[seg_start:seg_end]
    boxl_x = segment_df['BOXL_x'].median()
    boxr_x = segment_df['BOXR_x'].median()
    return (boxl_x + boxr_x) / 2


def main():
    print("COMPREHENSIVE v4.2 ERROR ANALYSIS")
    print("=" * 70)

    early_ends = []
    late_ends = []
    missed_reaches = []
    false_positives = []
    correct_matches = []

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
            continue
        df = pd.read_hdf(dlc_files[0])
        if isinstance(df.columns, pd.MultiIndex):
            scorer = df.columns.get_level_values(0)[0]
            df = df[scorer]
            df.columns = [f"{bp}_{coord}" for bp, coord in df.columns]

        # Get segments for slit position
        seg_data = load_json(DATA_DIR / f"{video}_segments.json")
        segments = seg_data.get('segments', []) if seg_data else []

        matches, unmatched_gt, unmatched_algo = match_reaches(gt_reaches, algo_reaches)

        # Process matches
        for gi, ai, dist in matches:
            gr = gt_reaches[gi]
            ar = algo_reaches[ai]
            end_offset = ar.get('end_frame', 0) - gr['end_frame']
            start_offset = ar.get('start_frame', 0) - gr['start_frame']

            # Find the segment this reach belongs to
            seg_start = 0
            seg_end = len(df)
            for seg in segments:
                if seg.get('start_frame', 0) <= gr['start_frame'] <= seg.get('end_frame', len(df)):
                    seg_start = seg['start_frame']
                    seg_end = seg['end_frame']
                    break

            slit_x = get_slit_center(df, seg_start, seg_end)

            info = {
                'video': video,
                'gt_start': gr['start_frame'],
                'gt_end': gr['end_frame'],
                'gt_duration': gr['end_frame'] - gr['start_frame'],
                'algo_start': ar.get('start_frame', 0),
                'algo_end': ar.get('end_frame', 0),
                'start_offset': start_offset,
                'end_offset': end_offset,
                'slit_x': slit_x,
            }

            if end_offset < -2:
                # Early end - analyze mechanism
                algo_end = ar.get('end_frame', 0)
                algo_end_info = get_hand_info(df, algo_end)
                next_frame_info = get_hand_info(df, algo_end + 1)

                # Get max extension during algo reach
                max_x = 0
                for f in range(ar.get('start_frame', 0), algo_end + 1):
                    hi = get_hand_info(df, f)
                    if hi['x'] and hi['x'] > max_x:
                        max_x = hi['x']

                # Check retraction at algo_end
                if algo_end_info['x']:
                    retraction = max_x - algo_end_info['x']
                    extension = max_x - slit_x
                    retraction_pct = retraction / max(extension, 1)
                    hand_offset = algo_end_info['x'] - slit_x
                else:
                    retraction = 0
                    retraction_pct = 0
                    hand_offset = 999

                info['algo_end_visible'] = algo_end_info['visible']
                info['next_frame_visible'] = next_frame_info['visible']
                info['max_x'] = max_x
                info['retraction'] = retraction
                info['retraction_pct'] = retraction_pct
                info['hand_offset_from_slit'] = hand_offset
                info['extension'] = extension

                # Categorize the mechanism
                if hand_offset < 5:
                    info['mechanism'] = 'RETURN_TO_START'
                elif retraction_pct > 0.4 and retraction > 5:
                    info['mechanism'] = 'RETRACTION'
                elif not algo_end_info['visible']:
                    info['mechanism'] = 'DISAPPEARANCE'
                else:
                    info['mechanism'] = 'UNKNOWN'

                early_ends.append(info)

            elif end_offset > 2:
                late_ends.append(info)
            else:
                correct_matches.append(info)

        # Missed GT reaches
        for gi in unmatched_gt:
            gr = gt_reaches[gi]
            # Check if any algo reach overlaps
            gt_start, gt_end = gr['start_frame'], gr['end_frame']
            overlapping = []
            for ai, ar in enumerate(algo_reaches):
                a_start = ar.get('start_frame', 0)
                a_end = ar.get('end_frame', 0)
                overlap = max(0, min(gt_end, a_end) - max(gt_start, a_start))
                if overlap > 0:
                    overlapping.append((ai, a_start, a_end, overlap))

            missed_reaches.append({
                'video': video,
                'gt_start': gt_start,
                'gt_end': gt_end,
                'gt_duration': gt_end - gt_start,
                'overlapping_algo': overlapping,
                'n_overlapping': len(overlapping),
            })

        # False positive algo reaches
        for ai in unmatched_algo:
            ar = algo_reaches[ai]
            a_start = ar.get('start_frame', 0)
            a_end = ar.get('end_frame', 0)
            # Check duration
            duration = a_end - a_start + 1
            # Check if overlaps with any GT
            gt_overlaps = []
            for gi, gr in enumerate(gt_reaches):
                overlap = max(0, min(gr['end_frame'], a_end) - max(gr['start_frame'], a_start))
                if overlap > 0:
                    gt_overlaps.append((gi, gr['start_frame'], gr['end_frame'], overlap))

            false_positives.append({
                'video': video,
                'algo_start': a_start,
                'algo_end': a_end,
                'duration': duration,
                'gt_overlaps': gt_overlaps,
                'n_gt_overlaps': len(gt_overlaps),
            })

    # ===================================================================
    # SECTION 1: Early End Analysis
    # ===================================================================
    print(f"\n{'=' * 70}")
    print(f"SECTION 1: EARLY-END ANALYSIS ({len(early_ends)} cases)")
    print(f"{'=' * 70}")

    mechanisms = Counter(e['mechanism'] for e in early_ends)
    print(f"\n  End mechanism:")
    for mech, cnt in mechanisms.most_common():
        print(f"    {mech:<25} {cnt:>5} ({cnt/len(early_ends)*100:.1f}%)")

    # Retraction cases
    retraction_cases = [e for e in early_ends if e['mechanism'] == 'RETRACTION']
    if retraction_cases:
        ret_pcts = [e['retraction_pct'] for e in retraction_cases]
        ret_offsets = [e['end_offset'] for e in retraction_cases]
        print(f"\n  RETRACTION cases ({len(retraction_cases)}):")
        print(f"    Retraction % range: {min(ret_pcts):.1%} - {max(ret_pcts):.1%}")
        print(f"    Mean retraction %: {np.mean(ret_pcts):.1%}")
        print(f"    End offset range: {min(ret_offsets)} to {max(ret_offsets)}")
        print(f"    Mean end offset: {np.mean(ret_offsets):.1f}")

    # Return-to-start cases
    return_cases = [e for e in early_ends if e['mechanism'] == 'RETURN_TO_START']
    if return_cases:
        offsets_ret = [e['hand_offset_from_slit'] for e in return_cases]
        offsets_end = [e['end_offset'] for e in return_cases]
        print(f"\n  RETURN_TO_START cases ({len(return_cases)}):")
        print(f"    Hand offset range: {min(offsets_ret):.1f} - {max(offsets_ret):.1f}")
        print(f"    End offset range: {min(offsets_end)} to {max(offsets_end)}")

    # Unknown cases
    unknown_cases = [e for e in early_ends if e['mechanism'] == 'UNKNOWN']
    if unknown_cases:
        print(f"\n  UNKNOWN cases ({len(unknown_cases)}):")
        for i, u in enumerate(unknown_cases[:10]):
            print(f"    [{i}] video={u['video']} algo_end={u['algo_end']} "
                  f"offset={u['end_offset']} retract={u['retraction_pct']:.1%} "
                  f"hand_offset={u['hand_offset_from_slit']:.1f} "
                  f"visible={u['algo_end_visible']} next_visible={u['next_frame_visible']}")

    # What threshold changes would fix retraction cases?
    if retraction_cases:
        print(f"\n  PROJECTED: Raising retraction threshold:")
        for new_pct in [0.45, 0.50, 0.55, 0.60, 0.70, 0.80]:
            would_fix = sum(1 for e in retraction_cases if e['retraction_pct'] < new_pct)
            print(f"    {new_pct:.0%}: would fix {would_fix}/{len(retraction_cases)}")

    # ===================================================================
    # SECTION 2: Missed GT Reaches
    # ===================================================================
    print(f"\n{'=' * 70}")
    print(f"SECTION 2: MISSED GT REACHES ({len(missed_reaches)} cases)")
    print(f"{'=' * 70}")

    has_overlap = sum(1 for m in missed_reaches if m['n_overlapping'] > 0)
    no_overlap = sum(1 for m in missed_reaches if m['n_overlapping'] == 0)
    print(f"\n  With overlapping algo reach: {has_overlap}")
    print(f"  No overlapping algo reach:  {no_overlap}")

    durations = [m['gt_duration'] for m in missed_reaches]
    if durations:
        print(f"\n  GT duration stats:")
        print(f"    Mean: {np.mean(durations):.1f} frames")
        print(f"    Median: {np.median(durations):.0f} frames")
        print(f"    Min: {min(durations)}, Max: {max(durations)}")

    for m in missed_reaches[:15]:
        overlap_str = f"overlaps: {m['n_overlapping']}" if m['n_overlapping'] > 0 else "NO OVERLAP"
        print(f"    {m['video']}: GT [{m['gt_start']}-{m['gt_end']}] "
              f"(dur={m['gt_duration']}) {overlap_str}")

    # ===================================================================
    # SECTION 3: False Positives
    # ===================================================================
    print(f"\n{'=' * 70}")
    print(f"SECTION 3: FALSE POSITIVES ({len(false_positives)} cases)")
    print(f"{'=' * 70}")

    fp_durations = [f['duration'] for f in false_positives]
    fp_overlaps = [f['n_gt_overlaps'] for f in false_positives]

    has_gt_overlap = sum(1 for f in false_positives if f['n_gt_overlaps'] > 0)
    no_gt_overlap = sum(1 for f in false_positives if f['n_gt_overlaps'] == 0)

    print(f"\n  Overlaps with GT reach:     {has_gt_overlap} "
          f"({has_gt_overlap/len(false_positives)*100:.1f}%) - likely split fragments")
    print(f"  No GT overlap:              {no_gt_overlap} "
          f"({no_gt_overlap/len(false_positives)*100:.1f}%) - pure false positives")

    print(f"\n  Duration distribution:")
    for lo, hi, label in [(1, 3, '1-3'), (4, 5, '4-5'), (6, 10, '6-10'),
                          (11, 20, '11-20'), (21, 50, '21-50'), (51, 999, '>50')]:
        cnt = sum(1 for d in fp_durations if lo <= d <= hi)
        print(f"    {label:>6} frames: {cnt:>5} ({cnt/len(false_positives)*100:.1f}%)")

    # Split fragments: FPs that overlap with a GT reach (algo split one GT reach)
    split_fragments = [f for f in false_positives if f['n_gt_overlaps'] > 0]
    pure_fps = [f for f in false_positives if f['n_gt_overlaps'] == 0]

    if split_fragments:
        sf_durations = [f['duration'] for f in split_fragments]
        print(f"\n  Split fragments ({len(split_fragments)}):")
        print(f"    Mean duration: {np.mean(sf_durations):.1f}")
        print(f"    These are second halves of reaches split by retraction/return-to-start")

    if pure_fps:
        pf_durations = [f['duration'] for f in pure_fps]
        print(f"\n  Pure false positives ({len(pure_fps)}):")
        print(f"    Mean duration: {np.mean(pf_durations):.1f}")
        print(f"    Duration distribution:")
        for lo, hi, label in [(1, 3, '1-3'), (4, 5, '4-5'), (6, 10, '6-10'),
                              (11, 20, '11-20'), (21, 999, '>20')]:
            cnt = sum(1 for d in pf_durations if lo <= d <= hi)
            print(f"      {label:>6} frames: {cnt:>5} ({cnt/len(pure_fps)*100:.1f}%)")

    # ===================================================================
    # SECTION 4: Late-End Analysis
    # ===================================================================
    print(f"\n{'=' * 70}")
    print(f"SECTION 4: LATE-END ANALYSIS ({len(late_ends)} cases)")
    print(f"{'=' * 70}")

    if late_ends:
        late_offsets = [e['end_offset'] for e in late_ends]
        print(f"  Mean offset: +{np.mean(late_offsets):.1f}")
        print(f"  Median: +{np.median(late_offsets):.0f}")
        for lo, hi, label in [(3, 5, '+3..+5'), (6, 10, '+6..+10'),
                              (11, 20, '+11..+20'), (21, 999, '>+20')]:
            cnt = sum(1 for o in late_offsets if lo <= o <= hi)
            print(f"    {label:>8}: {cnt:>5} ({cnt/len(late_ends)*100:.1f}%)")

    # ===================================================================
    # SECTION 5: Summary & Decision Tree Recommendations
    # ===================================================================
    total_gt = len(correct_matches) + len(early_ends) + len(late_ends) + len(missed_reaches)
    print(f"\n{'=' * 70}")
    print(f"SECTION 5: ERROR BUDGET (total GT: {total_gt})")
    print(f"{'=' * 70}")

    n_correct = len(correct_matches)
    n_early = len(early_ends)
    n_late = len(late_ends)
    n_missed = len(missed_reaches)
    n_fp = len(false_positives)
    n_split_fp = len(split_fragments) if 'split_fragments' in dir() else 0

    print(f"\n  Correct (both within 2):  {n_correct} ({n_correct/total_gt*100:.1f}%)")
    print(f"  Early end (>2 early):     {n_early} ({n_early/total_gt*100:.1f}%)")
    print(f"  Late end (>2 late):       {n_late} ({n_late/total_gt*100:.1f}%)")
    print(f"  Missed entirely:          {n_missed} ({n_missed/total_gt*100:.1f}%)")
    print(f"  False positives:          {n_fp}")
    print(f"    - Split fragments:      {has_gt_overlap}")
    print(f"    - Pure FP:              {no_gt_overlap}")

    # Estimate improvement from fixing each category
    print(f"\n  IF WE COULD FIX...")
    fixable_early = sum(1 for e in early_ends if e['mechanism'] in ('RETRACTION', 'RETURN_TO_START'))
    print(f"  All retraction/return-to-start early ends: +{fixable_early} -> "
          f"{(n_correct + fixable_early)/total_gt*100:.1f}% both-within-2")
    print(f"  All missed reaches: +{n_missed} -> "
          f"{(n_correct + fixable_early + n_missed)/total_gt*100:.1f}% both-within-2")
    print(f"  All late ends: +{n_late} -> "
          f"{(n_correct + fixable_early + n_missed + n_late)/total_gt*100:.1f}%")


if __name__ == "__main__":
    main()
