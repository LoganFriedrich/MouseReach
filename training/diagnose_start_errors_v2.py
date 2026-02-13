"""
Diagnose why start frame accuracy is only ~91% instead of target 99%.
Uses same patterns as the working diagnose_early_ends.py.
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
NOSE_THRESHOLD = 25


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


def nose_distance_to_slit(row, slit_x, slit_y):
    nose_x = row.get('Nose_x', np.nan)
    nose_y = row.get('Nose_y', np.nan)
    nose_l = row.get('Nose_likelihood', 0)
    if nose_l < 0.3 or np.isnan(nose_x):
        return None
    return np.sqrt((nose_x - slit_x)**2 + (nose_y - slit_y)**2)


def main():
    print("START FRAME ERROR DIAGNOSTIC")
    print("=" * 70)

    late_cases = []     # algo starts AFTER GT (offset > 2)
    early_cases = []    # algo starts BEFORE GT (offset < -2)
    all_offsets = []

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

        # Get slit center for each segment
        seg_data = load_json(DATA_DIR / f"{video}_segments.json")
        segments = seg_data.get('segments', []) if seg_data else []

        def get_slit_for_frame(frame):
            for seg in segments:
                if seg.get('start_frame', 0) <= frame <= seg.get('end_frame', len(df)):
                    seg_df = df.iloc[seg['start_frame']:seg['end_frame']]
                    bx = seg_df['BOXL_x'].median()
                    by = seg_df['BOXL_y'].median()
                    rx = seg_df['BOXR_x'].median()
                    ry = seg_df['BOXR_y'].median()
                    return (bx + rx) / 2, (by + ry) / 2
            return None, None

        matches = match_reaches(gt_reaches, algo_reaches)
        for gi, ai, dist in matches:
            gr = gt_reaches[gi]
            ar = algo_reaches[ai]
            offset = ar.get('start_frame', 0) - gr['start_frame']
            all_offsets.append(offset)

            if abs(offset) <= 2:
                continue

            gt_start = gr['start_frame']
            algo_start = ar.get('start_frame', 0)
            slit_x, slit_y = get_slit_for_frame(gt_start)

            case = {
                'video': video,
                'gt_start': gt_start,
                'gt_end': gr['end_frame'],
                'algo_start': algo_start,
                'algo_end': ar.get('end_frame', 0),
                'offset': offset,
            }

            # Check DLC at GT start
            if gt_start < len(df):
                row_gt = df.iloc[gt_start]
                case['hand_lik_gt'] = best_hand_likelihood(row_gt)
                case['hand_vis_gt'] = case['hand_lik_gt'] >= HAND_THRESHOLD
                if slit_x is not None:
                    nd = nose_distance_to_slit(row_gt, slit_x, slit_y)
                    case['nose_dist_gt'] = nd
                    case['nose_engaged_gt'] = nd is not None and nd < NOSE_THRESHOLD

            # Check DLC at algo start
            if algo_start < len(df):
                row_algo = df.iloc[algo_start]
                case['hand_lik_algo'] = best_hand_likelihood(row_algo)
                if slit_x is not None:
                    nd = nose_distance_to_slit(row_algo, slit_x, slit_y)
                    case['nose_dist_algo'] = nd
                    case['nose_engaged_algo'] = nd is not None and nd < NOSE_THRESHOLD

            # For late cases: check each frame from gt_start to algo_start
            if offset > 2 and slit_x is not None:
                frame_details = []
                for f in range(gt_start, min(algo_start + 1, len(df))):
                    row = df.iloc[f]
                    hl = best_hand_likelihood(row)
                    nd = nose_distance_to_slit(row, slit_x, slit_y)
                    frame_details.append({
                        'frame': f,
                        'hand_lik': hl,
                        'hand_vis': hl >= HAND_THRESHOLD,
                        'nose_dist': nd,
                        'nose_engaged': nd is not None and nd < NOSE_THRESHOLD,
                    })
                case['frame_details'] = frame_details

            # For early cases: check if previous algo reach overlaps
            if offset < -2:
                # Find algo reach ending closest before this algo_start
                prev_ends = [r.get('end_frame', 0) for r in algo_reaches
                             if r.get('end_frame', 0) < algo_start]
                if prev_ends:
                    closest_prev_end = max(prev_ends)
                    case['prev_reach_gap'] = algo_start - closest_prev_end
                else:
                    case['prev_reach_gap'] = None

            if offset > 2:
                late_cases.append(case)
            else:
                early_cases.append(case)

    total_matched = len(all_offsets)
    within_2 = sum(1 for o in all_offsets if abs(o) <= 2)
    print(f"\nTotal matched: {total_matched}")
    print(f"Within 2: {within_2} ({within_2/total_matched*100:.1f}%)")
    print(f"Algo LATE (>2): {len(late_cases)}")
    print(f"Algo EARLY (<-2): {len(early_cases)}")

    # ===== LATE CASES (algo starts after GT) =====
    print(f"\n{'=' * 70}")
    print(f"ALGO LATE CASES: {len(late_cases)} (algo starts AFTER GT)")
    print(f"{'=' * 70}")

    if late_cases:
        # Why is algo late? Is hand visible at GT start?
        hand_vis_at_gt = sum(1 for c in late_cases if c.get('hand_vis_gt', False))
        hand_invis_at_gt = len(late_cases) - hand_vis_at_gt
        print(f"\n  Hand visible (>=0.5) at GT start: {hand_vis_at_gt}/{len(late_cases)}")
        print(f"  Hand NOT visible at GT start: {hand_invis_at_gt}/{len(late_cases)}")

        # What's the hand likelihood at GT start for invisible cases?
        invis_liks = [c['hand_lik_gt'] for c in late_cases if not c.get('hand_vis_gt', False)]
        if invis_liks:
            print(f"\n  Hand likelihood at GT start (when invisible):")
            print(f"    Mean: {np.mean(invis_liks):.3f}")
            print(f"    Median: {np.median(invis_liks):.3f}")
            for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
                cnt = sum(1 for l in invis_liks if l >= thresh)
                print(f"    >= {thresh}: {cnt}/{len(invis_liks)}")

        # For cases where hand IS visible: is nose not engaged?
        hand_vis_cases = [c for c in late_cases if c.get('hand_vis_gt', False)]
        if hand_vis_cases:
            nose_engaged = sum(1 for c in hand_vis_cases if c.get('nose_engaged_gt', False))
            nose_not = len(hand_vis_cases) - nose_engaged
            print(f"\n  Among {len(hand_vis_cases)} cases with hand visible at GT start:")
            print(f"    Nose engaged: {nose_engaged}")
            print(f"    Nose NOT engaged: {nose_not}")

            nose_dists = [c['nose_dist_gt'] for c in hand_vis_cases
                          if c.get('nose_dist_gt') is not None and not c.get('nose_engaged_gt', False)]
            if nose_dists:
                print(f"\n    Nose distance when not engaged:")
                print(f"      Mean: {np.mean(nose_dists):.1f}px")
                print(f"      Median: {np.median(nose_dists):.1f}px")
                for thresh in [25, 30, 35, 40, 50, 60, 80, 100]:
                    cnt = sum(1 for d in nose_dists if d <= thresh)
                    print(f"      <= {thresh}px: {cnt}/{len(nose_dists)}")

        # Offset distribution for late cases
        late_offsets = [c['offset'] for c in late_cases]
        print(f"\n  Offset distribution:")
        for lo, hi, label in [(3, 5, '3-5'), (6, 10, '6-10'), (11, 15, '11-15'),
                               (16, 20, '16-20'), (21, 30, '21-30')]:
            cnt = sum(1 for o in late_offsets if lo <= o <= hi)
            print(f"    {label}: {cnt}")

        # Frame-by-frame examples
        print(f"\n  First 10 late cases (frame-by-frame):")
        for i, c in enumerate(late_cases[:10]):
            print(f"\n  [{i}] {c['video']}: GT start={c['gt_start']}, Algo start={c['algo_start']}, offset=+{c['offset']}")
            print(f"      Hand at GT start: lik={c.get('hand_lik_gt', '?'):.2f}, nose_engaged={c.get('nose_engaged_gt', '?')}")
            if 'frame_details' in c:
                for fd in c['frame_details'][:15]:
                    marker = " <-- GT" if fd['frame'] == c['gt_start'] else ""
                    marker = " <-- ALGO" if fd['frame'] == c['algo_start'] else marker
                    print(f"      f{fd['frame']}: hand={fd['hand_lik']:.2f}{'*' if fd['hand_vis'] else ' '} "
                          f"nose={'Y' if fd['nose_engaged'] else 'N'}({fd['nose_dist']:.0f}px){marker}" if fd['nose_dist'] is not None else
                          f"      f{fd['frame']}: hand={fd['hand_lik']:.2f}{'*' if fd['hand_vis'] else ' '} "
                          f"nose=?{marker}")

    # ===== EARLY CASES (algo starts before GT) =====
    print(f"\n{'=' * 70}")
    print(f"ALGO EARLY CASES: {len(early_cases)} (algo starts BEFORE GT)")
    print(f"{'=' * 70}")

    if early_cases:
        # Why is algo early? Common causes:
        # 1. Previous reach absorbed (retraction not confirmed → reach runs longer)
        # 2. Brief hand visibility before the real reach
        # 3. Noise detection

        # Check if previous reach gap is small (merge artifact)
        merge_artifacts = sum(1 for c in early_cases
                              if c.get('prev_reach_gap') is not None and c['prev_reach_gap'] <= 3)
        print(f"\n  Previous algo reach ends within 3 frames: {merge_artifacts}/{len(early_cases)} (merge artifact)")

        # Hand likelihood at algo start
        algo_liks = [c.get('hand_lik_algo', 0) for c in early_cases]
        print(f"\n  Hand likelihood at algo start:")
        print(f"    Mean: {np.mean(algo_liks):.3f}")
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            cnt = sum(1 for l in algo_liks if l >= thresh)
            print(f"    >= {thresh}: {cnt}/{len(early_cases)}")

        # Offset distribution
        early_offsets = [c['offset'] for c in early_cases]
        print(f"\n  Offset distribution:")
        for lo, hi, label in [(-5, -3, '-3..-5'), (-10, -6, '-6..-10'), (-15, -11, '-11..-15'),
                               (-20, -16, '-16..-20'), (-30, -21, '-21..-30')]:
            cnt = sum(1 for o in early_offsets if lo <= o <= hi)
            print(f"    {label}: {cnt}")

        # Check if algo_start is WITHIN a different GT reach
        # (would indicate the algo failed to split here)
        print(f"\n  First 10 early cases:")
        for i, c in enumerate(early_cases[:10]):
            print(f"  [{i}] {c['video']}: GT start={c['gt_start']}, Algo start={c['algo_start']}, offset={c['offset']}")
            print(f"      Hand at algo_start: lik={c.get('hand_lik_algo', '?'):.2f}")
            if c.get('prev_reach_gap') is not None:
                print(f"      Gap from previous reach: {c['prev_reach_gap']} frames")

    # ===== CATEGORIZATION =====
    print(f"\n{'=' * 70}")
    print(f"CATEGORIZATION SUMMARY")
    print(f"{'=' * 70}")

    categories = Counter()
    for c in late_cases:
        if not c.get('hand_vis_gt', False):
            if c.get('hand_lik_gt', 0) >= 0.3:
                categories['LATE: hand_low_confidence'] += 1
            else:
                categories['LATE: hand_invisible'] += 1
        elif not c.get('nose_engaged_gt', False):
            categories['LATE: nose_not_engaged'] += 1
        else:
            categories['LATE: hand_vis+nose_engaged (mystery)'] += 1

    for c in early_cases:
        if c.get('prev_reach_gap') is not None and c['prev_reach_gap'] <= 3:
            categories['EARLY: merge_artifact'] += 1
        elif abs(c['offset']) <= 5:
            categories['EARLY: small_offset (3-5)'] += 1
        else:
            categories['EARLY: large_offset (>5)'] += 1

    for cat, cnt in categories.most_common():
        pct = cnt / (len(late_cases) + len(early_cases)) * 100
        print(f"  {cat:<45} {cnt:>5} ({pct:.1f}%)")

    # ===== RECOMMENDATIONS =====
    print(f"\n{'=' * 70}")
    print(f"PROJECTED IMPROVEMENTS")
    print(f"{'=' * 70}")

    # What if we lower nose threshold?
    fixable_nose = sum(1 for c in late_cases
                       if c.get('hand_vis_gt', False)
                       and not c.get('nose_engaged_gt', False)
                       and c.get('nose_dist_gt') is not None)
    for nose_thresh in [25, 30, 35, 40, 50, 60, 80]:
        fixed = sum(1 for c in late_cases
                    if c.get('hand_vis_gt', False)
                    and c.get('nose_dist_gt') is not None
                    and c['nose_dist_gt'] <= nose_thresh)
        new_within2 = within_2 + fixed
        print(f"  NOSE_THRESHOLD={nose_thresh}px: +{fixed} late cases fixed → {new_within2}/{total_matched} ({new_within2/total_matched*100:.1f}%)")


if __name__ == "__main__":
    main()
