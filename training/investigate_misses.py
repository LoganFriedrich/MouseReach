"""
Investigate WHY the algorithm misses 701 reaches where conditions ARE met.

Hypothesis: The algo IS detecting something at these frames, but either:
A) The algo absorbs multiple GT reaches into one long reach (merge problem)
B) The algo detects a reach but at a different frame offset (matching tolerance)
C) Post-processing removes the reach (extent filter, duration filter)
D) The algo reach ends too early/late, so start frames don't align

This script checks each hypothesis by looking at algo reach proximity
to each missed GT reach.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_0_0")


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    print("INVESTIGATING ALGO_SHOULD_DETECT MISSES")
    print("=" * 70)
    print()

    TOLERANCE = 5  # Current matching tolerance

    miss_explanations = Counter()
    all_details = []

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

        # Flatten algo reaches
        algo_reaches = []
        for seg in algo_data.get('segments', []):
            for r in seg.get('reaches', []):
                algo_reaches.append(r)

        if not algo_reaches:
            continue

        # First pass: match GT to algo (same as agreement analysis)
        algo_used = set()
        gt_matched = set()

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
                gt_matched.add(gi)

        # For each unmatched GT reach, investigate WHY
        for gi, gr in enumerate(gt_reaches):
            if gi in gt_matched:
                continue

            gt_start = gr['start_frame']
            gt_end = gr['end_frame']
            gt_apex = gr.get('apex_frame')
            gt_dur = gt_end - gt_start + 1

            # Find the CLOSEST algo reach (even if beyond tolerance)
            closest_by_start = None
            closest_start_dist = float('inf')
            closest_by_overlap = None
            best_overlap = 0

            for ai, ar in enumerate(algo_reaches):
                a_start = ar.get('start_frame', 0)
                a_end = ar.get('end_frame', 0)

                # Closest by start frame
                d = abs(gt_start - a_start)
                if d < closest_start_dist:
                    closest_start_dist = d
                    closest_by_start = ar

                # Closest by frame overlap
                overlap_start = max(gt_start, a_start)
                overlap_end = min(gt_end, a_end)
                overlap = max(0, overlap_end - overlap_start + 1)
                if overlap > best_overlap:
                    best_overlap = overlap
                    closest_by_overlap = ar

            # Check if GT reach is CONTAINED within a larger algo reach
            containing_algo = None
            for ar in algo_reaches:
                a_start = ar.get('start_frame', 0)
                a_end = ar.get('end_frame', 0)
                if a_start <= gt_start and a_end >= gt_end:
                    containing_algo = ar
                    break

            # Check if GT reach OVERLAPS with any algo reach
            overlapping_algos = []
            for ar in algo_reaches:
                a_start = ar.get('start_frame', 0)
                a_end = ar.get('end_frame', 0)
                overlap_start = max(gt_start, a_start)
                overlap_end = min(gt_end, a_end)
                if overlap_end >= overlap_start:
                    overlapping_algos.append(ar)

            detail = {
                'video': video,
                'gt_start': gt_start,
                'gt_end': gt_end,
                'gt_duration': gt_dur,
                'closest_start_dist': closest_start_dist,
            }

            # Classify the miss explanation
            if containing_algo is not None:
                # GT reach is fully inside a larger algo reach
                # This is the MERGE problem: algo sees one big reach, humans see multiple
                a_dur = containing_algo.get('end_frame', 0) - containing_algo.get('start_frame', 0) + 1
                explanation = 'ABSORBED_IN_LARGER_REACH'
                detail['algo_reach_duration'] = a_dur
                detail['algo_start'] = containing_algo.get('start_frame', 0)
                detail['algo_end'] = containing_algo.get('end_frame', 0)

            elif closest_start_dist <= 15:
                # Close but beyond matching tolerance
                explanation = 'NEAR_MISS_OFFSET'
                detail['offset'] = closest_start_dist

            elif best_overlap > 0:
                # There IS overlap, but not enough for matching
                explanation = 'PARTIAL_OVERLAP'
                detail['overlap_frames'] = best_overlap
                detail['overlap_pct'] = best_overlap / gt_dur * 100

            elif closest_start_dist <= 50:
                # Moderately close
                explanation = 'MODERATE_OFFSET'
                detail['offset'] = closest_start_dist

            else:
                # No algo reach anywhere near this GT reach
                explanation = 'NO_ALGO_REACH_NEARBY'
                detail['nearest_dist'] = closest_start_dist

            detail['explanation'] = explanation
            miss_explanations[explanation] += 1
            all_details.append(detail)

    total = len(all_details)
    print(f"Total unmatched GT reaches investigated: {total}\n")

    print("MISS EXPLANATION BREAKDOWN:")
    print(f"  {'Explanation':<30} {'Count':>6} {'Pct':>7}")
    print(f"  {'-'*45}")
    for expl, count in miss_explanations.most_common():
        pct = count / max(total, 1) * 100
        print(f"  {expl:<30} {count:>6} {pct:>6.1f}%")

    # Details per category
    print(f"\n\nDETAILED PROFILES:")

    # A) Absorbed into larger reaches
    absorbed = [d for d in all_details if d['explanation'] == 'ABSORBED_IN_LARGER_REACH']
    if absorbed:
        algo_durs = [d['algo_reach_duration'] for d in absorbed]
        gt_durs = [d['gt_duration'] for d in absorbed]
        print(f"\n  ABSORBED_IN_LARGER_REACH (n={len(absorbed)}):")
        print(f"    GT reach duration:   mean={np.mean(gt_durs):.1f}, median={np.median(gt_durs):.0f}")
        print(f"    Algo reach duration: mean={np.mean(algo_durs):.1f}, median={np.median(algo_durs):.0f}")
        print(f"    Algo/GT duration ratio: {np.mean(algo_durs)/np.mean(gt_durs):.1f}x")
        print(f"    The algo is merging what humans see as distinct reaches into")
        print(f"    single long reaches. The algo's reach-splitting logic needs work.")

    # B) Near misses
    near = [d for d in all_details if d['explanation'] == 'NEAR_MISS_OFFSET']
    if near:
        offsets = [d['offset'] for d in near]
        print(f"\n  NEAR_MISS_OFFSET (n={len(near)}):")
        print(f"    Start frame offset: mean={np.mean(offsets):.1f}, median={np.median(offsets):.0f}")
        print(f"    These would match with a larger tolerance (current: {TOLERANCE} frames)")

    # C) Partial overlap
    partial = [d for d in all_details if d['explanation'] == 'PARTIAL_OVERLAP']
    if partial:
        overlaps = [d['overlap_pct'] for d in partial]
        print(f"\n  PARTIAL_OVERLAP (n={len(partial)}):")
        print(f"    Overlap %: mean={np.mean(overlaps):.1f}%, median={np.median(overlaps):.0f}%")

    # D) No algo reach nearby
    none_nearby = [d for d in all_details if d['explanation'] == 'NO_ALGO_REACH_NEARBY']
    if none_nearby:
        dists = [d['nearest_dist'] for d in none_nearby]
        gt_durs = [d['gt_duration'] for d in none_nearby]
        print(f"\n  NO_ALGO_REACH_NEARBY (n={len(none_nearby)}):")
        print(f"    Nearest algo reach: mean={np.mean(dists):.0f} frames away")
        print(f"    GT reach duration:  mean={np.mean(gt_durs):.1f}")
        print(f"    These are genuine detection failures - algo has nothing near here.")

    # Summary
    print(f"\n\n{'=' * 70}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'=' * 70}")

    absorbed_n = miss_explanations.get('ABSORBED_IN_LARGER_REACH', 0)
    near_n = miss_explanations.get('NEAR_MISS_OFFSET', 0)
    partial_n = miss_explanations.get('PARTIAL_OVERLAP', 0)
    moderate_n = miss_explanations.get('MODERATE_OFFSET', 0)
    none_n = miss_explanations.get('NO_ALGO_REACH_NEARBY', 0)
    t = max(total, 1)

    print(f"""
The 743 missed GT reaches break down as:

1. ABSORBED ({absorbed_n}, {absorbed_n/t*100:.0f}%): GT reach is INSIDE a larger algo reach.
   The algo detects a reach here, but it's merged with neighboring reaches.
   FIX: Better reach-splitting in the algorithm. The algo needs to recognize
   when a long detected reach should be broken into multiple shorter ones.

2. NEAR MISS ({near_n}, {near_n/t*100:.0f}%): Algo has a reach within 6-15 frames.
   Just outside current matching tolerance of {TOLERANCE} frames.
   FIX: Consider wider matching tolerance, or algo timing adjustments.

3. PARTIAL OVERLAP ({partial_n}, {partial_n/t*100:.0f}%): Algo and GT reaches overlap
   but start/end frames don't align enough for matching.

4. MODERATE OFFSET ({moderate_n}, {moderate_n/t*100:.0f}%): Algo reach within 50 frames.

5. NO ALGO REACH ({none_n}, {none_n/t*100:.0f}%): Nothing from algo near this GT reach.
   These are genuine detection failures.

BIGGEST WIN: If {absorbed_n} ABSORBED misses are due to merge problems,
fixing the reach-splitting logic alone could recover {absorbed_n/t*100:.0f}% of misses.
""")


if __name__ == "__main__":
    main()
