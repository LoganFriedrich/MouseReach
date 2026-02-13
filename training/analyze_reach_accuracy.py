"""
Rigorous reach-by-reach accuracy analysis.

For each human-determined reach:
  1. Does the algo have a corresponding reach AT ALL? (existence match)
  2. If yes, does the START frame match? (exact + within-N)
  3. If yes, does the END frame match? (exact + within-N)

Matching is done by frame proximity, NOT by reach index/ID, to handle
FP-induced index shifts. We match each GT reach to the nearest algo
reach by start frame, then separately check end frame accuracy.

"Intermediate miscalls excluded": if the algo has an FP between GT reach
5 and GT reach 6, that doesn't prevent GT reach 6 from matching algo
reach 7 - we match on frames, not sequential IDs.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_0_0")


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def match_reaches_rigorously(gt_reaches, algo_reaches, max_start_dist=30):
    """Match GT reaches to algo reaches by frame proximity.

    For each GT reach, find the algo reach whose start frame is closest.
    If within max_start_dist, it's a candidate match.

    We match greedily in order of best distance first, to avoid conflicts.
    Each algo reach can only match one GT reach (1:1).

    Returns list of match records.
    """
    # Build all candidate pairs with distances
    candidates = []
    for gi, gr in enumerate(gt_reaches):
        gt_start = gr['start_frame']
        gt_end = gr['end_frame']
        for ai, ar in enumerate(algo_reaches):
            a_start = ar.get('start_frame', 0)
            a_end = ar.get('end_frame', 0)
            start_dist = abs(gt_start - a_start)
            if start_dist <= max_start_dist:
                candidates.append({
                    'gi': gi, 'ai': ai,
                    'gt_start': gt_start, 'gt_end': gt_end,
                    'algo_start': a_start, 'algo_end': a_end,
                    'start_dist': start_dist,
                    'end_dist': abs(gt_end - a_end),
                    'start_offset': a_start - gt_start,  # positive = algo late
                    'end_offset': a_end - gt_end,         # positive = algo late
                })

    # Sort by start_dist (best matches first)
    candidates.sort(key=lambda c: c['start_dist'])

    # Greedy 1:1 matching
    gt_used = set()
    algo_used = set()
    matches = []

    for c in candidates:
        if c['gi'] not in gt_used and c['ai'] not in algo_used:
            gt_used.add(c['gi'])
            algo_used.add(c['ai'])
            matches.append(c)

    # Unmatched GT reaches
    unmatched_gi = set(range(len(gt_reaches))) - gt_used

    return matches, unmatched_gi


def main():
    print("RIGOROUS REACH-BY-REACH ACCURACY ANALYSIS")
    print("=" * 70)
    print()
    print("For each human-determined reach, find the BEST matching algo reach")
    print("(by frame proximity, not by index). Then separately report:")
    print("  - Did a matching reach EXIST at all?")
    print("  - Was the START frame correct?")
    print("  - Was the END frame correct?")
    print()

    # Collect all matches across videos
    all_matches = []
    all_unmatched = []
    per_video = defaultdict(lambda: {
        'gt_total': 0, 'matched': 0, 'unmatched': 0,
        'start_exact': 0, 'end_exact': 0,
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

        matches, unmatched_gi = match_reaches_rigorously(gt_reaches, algo_reaches)

        for m in matches:
            m['video'] = video
            all_matches.append(m)

        for gi in unmatched_gi:
            gr = gt_reaches[gi]
            all_unmatched.append({
                'video': video,
                'gt_start': gr['start_frame'],
                'gt_end': gr['end_frame'],
                'gt_duration': gr['end_frame'] - gr['start_frame'] + 1,
            })

        v = per_video[video]
        v['gt_total'] = len(gt_reaches)
        v['matched'] = len(matches)
        v['unmatched'] = len(unmatched_gi)
        v['start_exact'] = sum(1 for m in matches if m['start_offset'] == 0)
        v['end_exact'] = sum(1 for m in matches if m['end_offset'] == 0)

    total_gt = len(all_matches) + len(all_unmatched)
    total_matched = len(all_matches)

    # ================================================================
    # SECTION 1: EXISTENCE - Does a matching algo reach exist?
    # ================================================================
    print("=" * 70)
    print("1. EXISTENCE: Does the algo have a corresponding reach?")
    print("   (Matched by closest start frame within 30 frames, 1:1 greedy)")
    print("=" * 70)
    print()
    print(f"Total human-determined reaches: {total_gt}")
    print(f"  Matched to an algo reach:     {total_matched} ({total_matched/max(total_gt,1)*100:.1f}%)")
    print(f"  No algo match within 30 frames: {len(all_unmatched)} ({len(all_unmatched)/max(total_gt,1)*100:.1f}%)")

    # How close were the matches?
    if all_matches:
        start_dists = [m['start_dist'] for m in all_matches]
        print(f"\n  Match proximity (start frame distance):")
        for threshold in [0, 1, 2, 3, 5, 10, 15, 20, 30]:
            count = sum(1 for d in start_dists if d <= threshold)
            print(f"    Within {threshold:>2} frames: {count}/{total_matched} "
                  f"({count/total_matched*100:.1f}%)")

    # Per-video
    print(f"\n  Per-video existence rate:")
    print(f"  {'Video':<35} {'GT':>4} {'Match':>5} {'Rate':>5}")
    print(f"  {'-'*55}")
    for video in sorted(per_video.keys(),
                        key=lambda v: per_video[v]['matched'] / max(per_video[v]['gt_total'], 1)):
        v = per_video[video]
        rate = v['matched'] / max(v['gt_total'], 1) * 100
        print(f"  {video:<35} {v['gt_total']:>4} {v['matched']:>5} {rate:>4.0f}%")

    # ================================================================
    # SECTION 2: START FRAME ACCURACY
    # ================================================================
    print(f"\n\n{'=' * 70}")
    print("2. START FRAME ACCURACY")
    print("   For each matched reach, how accurate is the algo's start frame?")
    print("=" * 70)

    if not all_matches:
        print("  No matches to analyze.")
        return

    start_offsets = [m['start_offset'] for m in all_matches]
    start_abs = [abs(o) for o in start_offsets]

    print(f"\n  Total matched reaches: {total_matched}")
    print(f"\n  Start frame offset (algo - human):")
    print(f"    Mean:   {np.mean(start_offsets):+.2f} frames")
    print(f"    Median: {np.median(start_offsets):+.0f} frames")
    print(f"    Std:    {np.std(start_offsets):.2f} frames")
    print(f"    Min:    {np.min(start_offsets):+d}, Max: {np.max(start_offsets):+d}")

    print(f"\n  Start frame accuracy:")
    for threshold in [0, 1, 2, 3, 5, 10]:
        count = sum(1 for d in start_abs if d <= threshold)
        print(f"    Exact or within {threshold:>2} frames: "
              f"{count}/{total_matched} ({count/total_matched*100:.1f}%)")

    # Direction of error
    exact = sum(1 for o in start_offsets if o == 0)
    algo_early = sum(1 for o in start_offsets if o < 0)
    algo_late = sum(1 for o in start_offsets if o > 0)
    print(f"\n  Start frame direction:")
    print(f"    Algo exact:  {exact} ({exact/total_matched*100:.1f}%)")
    print(f"    Algo early:  {algo_early} ({algo_early/total_matched*100:.1f}%)")
    print(f"    Algo late:   {algo_late} ({algo_late/total_matched*100:.1f}%)")

    # Distribution histogram
    print(f"\n  Start offset distribution:")
    offset_counts = Counter(start_offsets)
    for offset in sorted(offset_counts.keys()):
        count = offset_counts[offset]
        bar = '#' * min(count, 80)
        if abs(offset) <= 10:
            print(f"    {offset:+3d}: {count:>4} {bar}")

    # ================================================================
    # SECTION 3: END FRAME ACCURACY
    # ================================================================
    print(f"\n\n{'=' * 70}")
    print("3. END FRAME ACCURACY")
    print("   For each matched reach, how accurate is the algo's end frame?")
    print("=" * 70)

    end_offsets = [m['end_offset'] for m in all_matches]
    end_abs = [abs(o) for o in end_offsets]

    print(f"\n  End frame offset (algo - human):")
    print(f"    Mean:   {np.mean(end_offsets):+.2f} frames")
    print(f"    Median: {np.median(end_offsets):+.0f} frames")
    print(f"    Std:    {np.std(end_offsets):.2f} frames")
    print(f"    Min:    {np.min(end_offsets):+d}, Max: {np.max(end_offsets):+d}")

    print(f"\n  End frame accuracy:")
    for threshold in [0, 1, 2, 3, 5, 10, 15, 20]:
        count = sum(1 for d in end_abs if d <= threshold)
        print(f"    Exact or within {threshold:>2} frames: "
              f"{count}/{total_matched} ({count/total_matched*100:.1f}%)")

    # Direction of error
    exact = sum(1 for o in end_offsets if o == 0)
    algo_early = sum(1 for o in end_offsets if o < 0)
    algo_late = sum(1 for o in end_offsets if o > 0)
    print(f"\n  End frame direction:")
    print(f"    Algo exact:     {exact} ({exact/total_matched*100:.1f}%)")
    print(f"    Algo too early: {algo_early} ({algo_early/total_matched*100:.1f}%)")
    print(f"    Algo too late:  {algo_late} ({algo_late/total_matched*100:.1f}%)")

    # Distribution histogram
    print(f"\n  End offset distribution (showing -20 to +20):")
    offset_counts = Counter(end_offsets)
    for offset in range(-20, 21):
        count = offset_counts.get(offset, 0)
        if count > 0:
            bar = '#' * min(count, 80)
            print(f"    {offset:+3d}: {count:>4} {bar}")

    # How many have large end offsets?
    print(f"\n  Large end offsets (>10 frames):")
    large_end = [m for m in all_matches if abs(m['end_offset']) > 10]
    print(f"    Count: {len(large_end)} ({len(large_end)/total_matched*100:.1f}%)")
    if large_end:
        large_offsets = [m['end_offset'] for m in large_end]
        print(f"    Mean offset: {np.mean(large_offsets):+.1f}")
        print(f"    These are mostly algo-{'late' if np.mean(large_offsets) > 0 else 'early'}")

    # ================================================================
    # SECTION 4: COMBINED - Both start AND end correct?
    # ================================================================
    print(f"\n\n{'=' * 70}")
    print("4. COMBINED: Both start AND end frame correct?")
    print("=" * 70)

    for tol in [0, 1, 2, 3, 5]:
        both_ok = sum(1 for m in all_matches
                      if abs(m['start_offset']) <= tol and abs(m['end_offset']) <= tol)
        start_ok = sum(1 for m in all_matches if abs(m['start_offset']) <= tol)
        end_ok = sum(1 for m in all_matches if abs(m['end_offset']) <= tol)
        print(f"\n  Tolerance: {tol} frames")
        print(f"    Start correct:      {start_ok}/{total_matched} ({start_ok/total_matched*100:.1f}%)")
        print(f"    End correct:        {end_ok}/{total_matched} ({end_ok/total_matched*100:.1f}%)")
        print(f"    BOTH correct:       {both_ok}/{total_matched} ({both_ok/total_matched*100:.1f}%)")

    # ================================================================
    # SECTION 5: WHERE IS THE PROBLEM?
    # ================================================================
    print(f"\n\n{'=' * 70}")
    print("5. PROBLEM DIAGNOSIS: What exactly is wrong?")
    print("=" * 70)

    # Categorize each matched reach
    categories = Counter()
    for m in all_matches:
        s_ok = abs(m['start_offset']) <= 2
        e_ok = abs(m['end_offset']) <= 2
        if s_ok and e_ok:
            cat = 'BOTH_CORRECT'
        elif s_ok and not e_ok:
            cat = 'START_OK_END_WRONG'
        elif not s_ok and e_ok:
            cat = 'START_WRONG_END_OK'
        else:
            cat = 'BOTH_WRONG'
        categories[cat] += 1

    print(f"\n  Categorization (within 2 frames = 'correct'):")
    for cat, count in categories.most_common():
        print(f"    {cat:<25} {count:>5} ({count/total_matched*100:.1f}%)")

    # For START_OK_END_WRONG: what does the end offset look like?
    start_ok_end_wrong = [m for m in all_matches
                          if abs(m['start_offset']) <= 2 and abs(m['end_offset']) > 2]
    if start_ok_end_wrong:
        end_offs = [m['end_offset'] for m in start_ok_end_wrong]
        print(f"\n  START_OK_END_WRONG profile (n={len(start_ok_end_wrong)}):")
        print(f"    End offset: mean={np.mean(end_offs):+.1f}, median={np.median(end_offs):+.0f}")
        print(f"    Algo end is {'too late' if np.mean(end_offs) > 0 else 'too early'} on average")
        print(f"    Distribution:")
        late = sum(1 for o in end_offs if o > 0)
        early = sum(1 for o in end_offs if o < 0)
        print(f"      Algo ends too late:  {late} ({late/len(end_offs)*100:.0f}%)")
        print(f"      Algo ends too early: {early} ({early/len(end_offs)*100:.0f}%)")

    # ================================================================
    # SECTION 6: UNMATCHED GT REACHES - Are they really undetected?
    # ================================================================
    print(f"\n\n{'=' * 70}")
    print("6. UNMATCHED GT REACHES - Truly undetected?")
    print("=" * 70)

    if not all_unmatched:
        print(f"\n  All GT reaches were matched! No unmatched reaches.")
    else:
        print(f"\n  {len(all_unmatched)} GT reaches had no algo match within 30 frames.")
        print(f"  These are possible causes:")
        print(f"    a) GT reach is inside a longer algo reach (merge/split issue)")
        print(f"    b) Algo genuinely missed it (detection failure)")
        print(f"    c) GT reach is very short or at segment boundary")

        # Check if unmatched GT reaches overlap with ANY algo reach
        # Reload data to check overlap
        overlap_count = 0
        no_overlap_count = 0

        for gt_file in sorted(DATA_DIR.glob("*_unified_ground_truth.json")):
            gt_data = load_json(gt_file)
            if not gt_data:
                continue
            video = gt_data['video_name']

            algo_file = ALGO_DIR / f"{video}_reaches.json"
            algo_data = load_json(algo_file)
            if not algo_data:
                continue

            algo_reaches = []
            for seg in algo_data.get('segments', []):
                for r in seg.get('reaches', []):
                    algo_reaches.append(r)

            # Check each unmatched GT reach from this video
            for um in all_unmatched:
                if um['video'] != video:
                    continue

                has_overlap = False
                for ar in algo_reaches:
                    a_start = ar.get('start_frame', 0)
                    a_end = ar.get('end_frame', 0)
                    overlap_s = max(um['gt_start'], a_start)
                    overlap_e = min(um['gt_end'], a_end)
                    if overlap_e >= overlap_s:
                        has_overlap = True
                        break

                if has_overlap:
                    overlap_count += 1
                else:
                    no_overlap_count += 1

        n_um = len(all_unmatched)
        print(f"\n  Of {n_um} unmatched GT reaches:")
        print(f"    Overlap with an algo reach (merge issue): {overlap_count} "
              f"({overlap_count/max(n_um,1)*100:.1f}%)")
        print(f"    No overlap at all (detection failure):    {no_overlap_count} "
              f"({no_overlap_count/max(n_um,1)*100:.1f}%)")

        um_durations = [u['gt_duration'] for u in all_unmatched]
        print(f"\n  Unmatched GT reach durations:")
        print(f"    Mean: {np.mean(um_durations):.1f}, Median: {np.median(um_durations):.0f}")

    # ================================================================
    # SECTION 7: SUMMARY
    # ================================================================
    print(f"\n\n{'=' * 70}")
    print("7. SUMMARY")
    print("=" * 70)

    # Overall rates at tolerance=2
    exist_rate = total_matched / max(total_gt, 1) * 100
    start_2 = sum(1 for m in all_matches if abs(m['start_offset']) <= 2)
    end_2 = sum(1 for m in all_matches if abs(m['end_offset']) <= 2)
    both_2 = sum(1 for m in all_matches if abs(m['start_offset']) <= 2 and abs(m['end_offset']) <= 2)

    print(f"""
  Of {total_gt} human-determined reaches:

  EXISTENCE (does algo have a reach with start frame within 30 frames?):
    {total_matched}/{total_gt} = {exist_rate:.1f}%

  START FRAME (within 2 frames, among matched):
    {start_2}/{total_matched} = {start_2/max(total_matched,1)*100:.1f}%

  END FRAME (within 2 frames, among matched):
    {end_2}/{total_matched} = {end_2/max(total_matched,1)*100:.1f}%

  BOTH START+END (within 2 frames, among matched):
    {both_2}/{total_matched} = {both_2/max(total_matched,1)*100:.1f}%

  OVERALL (existence AND both boundaries within 2 frames, of all GT):
    {both_2}/{total_gt} = {both_2/max(total_gt,1)*100:.1f}%
""")

    # What's the bottleneck?
    if exist_rate < 80:
        print("  BOTTLENECK: Reach existence. Many GT reaches have no algo match.")
        print("  The algo is not detecting reaches that humans see.")
    elif start_2 / max(total_matched, 1) < 0.8:
        print("  BOTTLENECK: Start frame accuracy. Algo finds reaches but at wrong start.")
    elif end_2 / max(total_matched, 1) < 0.8:
        print("  BOTTLENECK: End frame accuracy. Algo finds reaches, starts are good,")
        print("  but end frames are wrong. This is the splitting problem.")
    else:
        print("  The algorithm is performing well on all three metrics.")


if __name__ == "__main__":
    main()
