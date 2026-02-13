"""
Diagnose why v4.1 reaches end ~6 frames too early.

For each matched reach where algo ends early (offset < -2), examine DLC data
between algo_end and gt_end to understand the mechanism:
- Is the hand invisible (DLC can't track) at those frames?
- Does the hand flicker back (reappear after disappearing)?
- Is there tracking at lower confidence thresholds?
- What is the nose engagement status?
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


def any_hand_visible(row, threshold=HAND_THRESHOLD):
    for p in RH_POINTS:
        if row.get(f'{p}_likelihood', 0) >= threshold:
            return True
    return False


def best_hand_likelihood(row):
    best = 0
    for p in RH_POINTS:
        l = row.get(f'{p}_likelihood', 0)
        if l > best:
            best = l
    return best


def main():
    print("DIAGNOSING v4.1 EARLY-END ERRORS")
    print("=" * 70)

    early_end_cases = []  # Cases where algo ends > 2 frames before GT

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

            if end_offset < -2:  # Algo ends early
                algo_end = ar.get('end_frame', 0)
                gt_end = gr['end_frame']
                gap_frames = gt_end - algo_end

                # Analyze DLC in the gap between algo_end and gt_end
                gap_analysis = {
                    'video': video,
                    'gt_start': gr['start_frame'],
                    'gt_end': gt_end,
                    'algo_end': algo_end,
                    'offset': end_offset,
                    'gap_frames': gap_frames,
                }

                # Check each frame in the gap
                visible_at_05 = 0  # visible at 0.5
                visible_at_04 = 0  # visible at 0.4
                visible_at_03 = 0  # visible at 0.3
                max_likelihood_in_gap = 0
                likelihoods_in_gap = []
                hand_reappears = False
                frames_until_reappear = None

                for f in range(algo_end + 1, min(gt_end + 1, len(df))):
                    row = df.iloc[f]
                    bl = best_hand_likelihood(row)
                    likelihoods_in_gap.append(bl)
                    max_likelihood_in_gap = max(max_likelihood_in_gap, bl)

                    if bl >= 0.5:
                        visible_at_05 += 1
                        if not hand_reappears:
                            hand_reappears = True
                            frames_until_reappear = f - algo_end
                    if bl >= 0.4:
                        visible_at_04 += 1
                    if bl >= 0.3:
                        visible_at_03 += 1

                # Also check at GT end frame specifically
                if gt_end < len(df):
                    gt_end_row = df.iloc[gt_end]
                    gap_analysis['gt_end_likelihood'] = best_hand_likelihood(gt_end_row)
                    gap_analysis['gt_end_visible_05'] = any_hand_visible(gt_end_row, 0.5)
                    gap_analysis['gt_end_visible_03'] = any_hand_visible(gt_end_row, 0.3)
                else:
                    gap_analysis['gt_end_likelihood'] = 0
                    gap_analysis['gt_end_visible_05'] = False
                    gap_analysis['gt_end_visible_03'] = False

                # Check at algo end frame
                if algo_end < len(df):
                    algo_end_row = df.iloc[algo_end]
                    gap_analysis['algo_end_likelihood'] = best_hand_likelihood(algo_end_row)
                else:
                    gap_analysis['algo_end_likelihood'] = 0

                # Also check 3 frames before algo_end (what the algo last saw)
                pre_end_likelihoods = []
                for f in range(max(0, algo_end - 2), algo_end + 1):
                    if f < len(df):
                        pre_end_likelihoods.append(best_hand_likelihood(df.iloc[f]))
                gap_analysis['pre_end_likelihoods'] = pre_end_likelihoods

                gap_analysis['visible_at_05'] = visible_at_05
                gap_analysis['visible_at_04'] = visible_at_04
                gap_analysis['visible_at_03'] = visible_at_03
                gap_analysis['max_likelihood_in_gap'] = max_likelihood_in_gap
                gap_analysis['hand_reappears'] = hand_reappears
                gap_analysis['frames_until_reappear'] = frames_until_reappear
                gap_analysis['likelihoods_in_gap'] = likelihoods_in_gap

                early_end_cases.append(gap_analysis)

    print(f"\nTotal early-end cases (offset < -2): {len(early_end_cases)}")

    if not early_end_cases:
        return

    # ===== SECTION 1: Gap Size Distribution =====
    offsets = [c['offset'] for c in early_end_cases]
    print(f"\n--- GAP SIZE DISTRIBUTION ---")
    for lo, hi, label in [(-3, -3, '-3'), (-5, -4, '-4..-5'), (-8, -6, '-6..-8'),
                          (-12, -9, '-9..-12'), (-20, -13, '-13..-20'), (-999, -21, '<-20')]:
        cnt = sum(1 for o in offsets if lo <= o <= hi)
        print(f"  {label:>8}: {cnt:>5} ({cnt/len(early_end_cases)*100:.1f}%)")

    # ===== SECTION 2: What does DLC see in the gap? =====
    print(f"\n--- DLC VISIBILITY IN GAP (algo_end+1 to gt_end) ---")

    # Does the hand reappear at 0.5 threshold?
    reappear_count = sum(1 for c in early_end_cases if c['hand_reappears'])
    print(f"  Hand reappears (>=0.5) in gap: {reappear_count}/{len(early_end_cases)} "
          f"({reappear_count/len(early_end_cases)*100:.1f}%)")

    # At 0.5 threshold
    any_visible_05 = sum(1 for c in early_end_cases if c['visible_at_05'] > 0)
    all_invisible_05 = sum(1 for c in early_end_cases if c['visible_at_05'] == 0)
    print(f"  Any frame visible at 0.5: {any_visible_05} ({any_visible_05/len(early_end_cases)*100:.1f}%)")
    print(f"  All frames invisible at 0.5: {all_invisible_05} ({all_invisible_05/len(early_end_cases)*100:.1f}%)")

    # At 0.3 threshold
    any_visible_03 = sum(1 for c in early_end_cases if c['visible_at_03'] > 0)
    print(f"  Any frame visible at 0.3: {any_visible_03} ({any_visible_03/len(early_end_cases)*100:.1f}%)")

    # Max likelihood in gap
    max_ls = [c['max_likelihood_in_gap'] for c in early_end_cases]
    print(f"\n  Max likelihood in gap:")
    for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        cnt = sum(1 for l in max_ls if l >= thresh)
        print(f"    >= {thresh:.1f}: {cnt:>5} ({cnt/len(early_end_cases)*100:.1f}%)")

    # ===== SECTION 3: What does DLC see at GT end frame? =====
    print(f"\n--- DLC AT GT END FRAME ---")
    gt_visible_05 = sum(1 for c in early_end_cases if c['gt_end_visible_05'])
    gt_visible_03 = sum(1 for c in early_end_cases if c['gt_end_visible_03'])
    gt_likelihoods = [c['gt_end_likelihood'] for c in early_end_cases]
    print(f"  Hand visible (>=0.5) at GT end: {gt_visible_05} ({gt_visible_05/len(early_end_cases)*100:.1f}%)")
    print(f"  Hand visible (>=0.3) at GT end: {gt_visible_03} ({gt_visible_03/len(early_end_cases)*100:.1f}%)")
    print(f"  Mean likelihood at GT end: {np.mean(gt_likelihoods):.3f}")
    print(f"  Median likelihood at GT end: {np.median(gt_likelihoods):.3f}")

    # ===== SECTION 4: Categorize the mechanism =====
    print(f"\n--- CATEGORIZATION ---")

    categories = Counter()
    for c in early_end_cases:
        if c['hand_reappears'] and c['frames_until_reappear'] and c['frames_until_reappear'] <= 5:
            categories['DLC_FLICKER_REAPPEAR'] += 1  # Hand flickers back within 5 frames
        elif c['visible_at_03'] > 0 and c['visible_at_05'] == 0:
            categories['LOW_CONFIDENCE_VISIBLE'] += 1  # Visible at 0.3 but not 0.5
        elif c['visible_at_05'] == 0 and c['visible_at_03'] == 0:
            categories['TRULY_INVISIBLE'] += 1  # DLC sees nothing
        elif c['hand_reappears'] and c['frames_until_reappear'] and c['frames_until_reappear'] > 5:
            categories['LATE_REAPPEAR'] += 1  # Reappears but after >5 frames
        else:
            categories['OTHER'] += 1

    for cat, cnt in categories.most_common():
        pct = cnt / len(early_end_cases) * 100
        print(f"  {cat:<30} {cnt:>5} ({pct:.1f}%)")

    # ===== SECTION 5: Reappearance timing =====
    reappear_cases = [c for c in early_end_cases if c['hand_reappears']]
    if reappear_cases:
        print(f"\n--- REAPPEARANCE TIMING (among {len(reappear_cases)} cases) ---")
        timings = [c['frames_until_reappear'] for c in reappear_cases]
        for thresh in [1, 2, 3, 4, 5, 6, 8, 10, 15]:
            cnt = sum(1 for t in timings if t <= thresh)
            print(f"  Reappears within {thresh:>2} frames: {cnt:>5} ({cnt/len(reappear_cases)*100:.1f}%)")

    # ===== SECTION 6: Actionable recommendations =====
    print(f"\n--- RECOMMENDATIONS ---")

    n_total = len(early_end_cases)
    n_flicker = categories.get('DLC_FLICKER_REAPPEAR', 0)
    n_low_conf = categories.get('LOW_CONFIDENCE_VISIBLE', 0)
    n_invisible = categories.get('TRULY_INVISIBLE', 0)
    n_late = categories.get('LATE_REAPPEAR', 0)

    if n_flicker > n_total * 0.3:
        print(f"  [HIGH IMPACT] Increase DISAPPEAR_THRESHOLD: {n_flicker} cases ({n_flicker/n_total*100:.0f}%) "
              f"are DLC flicker where hand reappears within 5 frames")
        if reappear_cases:
            timings = [c['frames_until_reappear'] for c in reappear_cases if c['frames_until_reappear'] <= 10]
            if timings:
                p90 = np.percentile(timings, 90)
                print(f"    Suggested threshold: {int(p90)+1} (covers 90% of reappearances)")

    if n_late > n_total * 0.1:
        print(f"  [MODERATE] Late reappear: {n_late} cases ({n_late/n_total*100:.0f}%) - "
              f"hand comes back after >5 frames")

    if n_low_conf > n_total * 0.1:
        print(f"  [CONSIDER] Lower threshold: {n_low_conf} cases ({n_low_conf/n_total*100:.0f}%) "
              f"have hand visible at 0.3 but not 0.5")

    if n_invisible > n_total * 0.3:
        print(f"  [HARD TO FIX] Truly invisible: {n_invisible} cases ({n_invisible/n_total*100:.0f}%) - "
              f"DLC cannot track hand at GT end, human uses video context")

    # ===== SECTION 7: What if we increase DISAPPEAR_THRESHOLD? =====
    print(f"\n--- PROJECTED IMPROVEMENT: INCREASING DISAPPEAR_THRESHOLD ---")
    # For each reappear case, the hand comes back after N frames.
    # If DISAPPEAR_THRESHOLD >= N, the reach would continue.
    for thresh in [3, 4, 5, 6, 8, 10, 12, 15]:
        would_fix = sum(1 for c in early_end_cases
                        if c['hand_reappears']
                        and c['frames_until_reappear'] is not None
                        and c['frames_until_reappear'] <= thresh)
        print(f"  DISAPPEAR_THRESHOLD={thresh:>2}: would fix {would_fix:>5}/{n_total} "
              f"({would_fix/n_total*100:.1f}%) early-end cases")


if __name__ == "__main__":
    main()
