"""
Diagnose the DLC gap structure for v4.2 early-end (UNKNOWN) cases.

The 643 UNKNOWN cases are disappearance-ended reaches where:
- algo_end = last visible frame (hand visible, extended)
- Hand then invisible for 3+ frames (hits DISAPPEAR_THRESHOLD)
- Hand reappears and GT says reach should continue

We need to know: how many consecutive invisible frames are there in the gap?
If most gaps are 3-5 frames, DISAPPEAR_THRESHOLD=6 would fix them.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_4_2")

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
    return matches


def best_hand_likelihood(row):
    best = 0
    for p in RH_POINTS:
        l = row.get(f'{p}_likelihood', 0)
        if l > best:
            best = l
    return best


def main():
    print("GAP STRUCTURE ANALYSIS FOR EARLY-END CASES")
    print("=" * 70)

    gap_lengths = []         # Length of first consecutive invisible run after algo_end
    max_gap_in_range = []    # Maximum consecutive invisible run between algo_end and gt_end
    total_invisible_in_range = []
    reappear_frame_offsets = []  # How many frames until hand first reappears at >=0.5
    early_end_count = 0

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

        matches = match_reaches(gt_reaches, algo_reaches)
        for gi, ai, dist in matches:
            gr = gt_reaches[gi]
            ar = algo_reaches[ai]
            end_offset = ar.get('end_frame', 0) - gr['end_frame']

            if end_offset >= -2:
                continue  # Not an early-end case

            algo_end = ar.get('end_frame', 0)
            gt_end = gr['end_frame']
            early_end_count += 1

            # Scan frames after algo_end to find gap structure
            # First: how many consecutive invisible frames starting at algo_end+1?
            first_gap = 0
            for f in range(algo_end + 1, min(gt_end + 10, len(df))):
                l = best_hand_likelihood(df.iloc[f])
                if l < HAND_THRESHOLD:
                    first_gap += 1
                else:
                    break
            gap_lengths.append(first_gap)

            # How many frames until hand first reappears at >=0.5 after algo_end?
            reappear_offset = None
            for f in range(algo_end + 1, min(gt_end + 10, len(df))):
                l = best_hand_likelihood(df.iloc[f])
                if l >= HAND_THRESHOLD:
                    reappear_offset = f - algo_end
                    break
            if reappear_offset is not None:
                reappear_frame_offsets.append(reappear_offset)

            # Max consecutive invisible run between algo_end and gt_end
            max_run = 0
            current_run = 0
            total_invisible = 0
            for f in range(algo_end + 1, min(gt_end + 1, len(df))):
                l = best_hand_likelihood(df.iloc[f])
                if l < HAND_THRESHOLD:
                    current_run += 1
                    total_invisible += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            max_gap_in_range.append(max_run)
            total_invisible_in_range.append(total_invisible)

    print(f"\nTotal early-end cases: {early_end_count}")

    # Gap immediately after algo_end
    print(f"\n--- FIRST GAP (consecutive invisible frames after algo_end) ---")
    for thresh in [0, 1, 2, 3, 4, 5, 6, 8, 10, 15]:
        cnt = sum(1 for g in gap_lengths if g <= thresh)
        print(f"  Gap <= {thresh:>2} frames: {cnt:>5} ({cnt/early_end_count*100:.1f}%)")

    print(f"\n  Distribution:")
    for val in range(0, 16):
        cnt = sum(1 for g in gap_lengths if g == val)
        if cnt > 0:
            bar = '#' * (cnt * 40 // early_end_count)
            print(f"    {val:>2} frames: {cnt:>5} ({cnt/early_end_count*100:.1f}%) {bar}")

    # How soon does hand reappear?
    print(f"\n--- REAPPEARANCE (first frame with hand visible after algo_end) ---")
    for thresh in [1, 2, 3, 4, 5, 6, 8, 10]:
        cnt = sum(1 for r in reappear_frame_offsets if r <= thresh)
        print(f"  Within {thresh:>2} frames: {cnt:>5}/{len(reappear_frame_offsets)} "
              f"({cnt/max(len(reappear_frame_offsets),1)*100:.1f}%)")

    # Max gap anywhere between algo_end and gt_end
    print(f"\n--- MAX CONSECUTIVE INVISIBLE RUN (algo_end to gt_end) ---")
    for thresh in [0, 1, 2, 3, 4, 5, 6, 8, 10]:
        cnt = sum(1 for g in max_gap_in_range if g <= thresh)
        print(f"  Max gap <= {thresh:>2}: {cnt:>5} ({cnt/early_end_count*100:.1f}%)")

    # PROJECTED FIX: What DISAPPEAR_THRESHOLD would bridge the first gap?
    print(f"\n--- PROJECTED FIX: INCREASING DISAPPEAR_THRESHOLD ---")
    print(f"  (Current threshold=3, first gap must be bridged)")
    for new_thresh in [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        # Would fix cases where first gap < new_thresh
        # (threshold=N means we end after N consecutive invisible frames,
        #  so gaps of N-1 or fewer are bridged)
        would_fix = sum(1 for g in gap_lengths if g < new_thresh)
        print(f"  DISAPPEAR_THRESHOLD={new_thresh:>2}: would bridge {would_fix:>5}/{early_end_count} "
              f"({would_fix/early_end_count*100:.1f}%) early-end cases")


if __name__ == "__main__":
    main()
