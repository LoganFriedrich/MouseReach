"""
Detailed error analysis for v5.2 reach detection.
Identifies remaining failures and categorizes them for targeted improvement.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_5_3")


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


def main():
    print("V5.2 DETAILED ERROR ANALYSIS")
    print("=" * 80)

    all_matches = []
    all_missed = []
    all_fps = []
    video_stats = {}

    gt_files = sorted(DATA_DIR.glob("*_unified_ground_truth.json"))
    print(f"Found {len(gt_files)} GT files\n")

    for gt_file in gt_files:
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

        matches, missed_gi, fp_ai = match_reaches(gt_reaches, algo_reaches)

        v_stats = {'n_gt': len(gt_reaches), 'n_algo': len(algo_reaches),
                   'n_matched': len(matches), 'n_missed': len(missed_gi),
                   'n_fp': len(fp_ai)}

        start_errors = 0
        end_errors = 0
        both_w2 = 0

        for gi, ai, dist in matches:
            gr = gt_reaches[gi]
            ar = algo_reaches[ai]
            s_off = ar.get('start_frame', 0) - gr['start_frame']
            e_off = ar.get('end_frame', 0) - gr['end_frame']

            m = {
                'video': video,
                'gt_start': gr['start_frame'], 'gt_end': gr['end_frame'],
                'algo_start': ar.get('start_frame', 0), 'algo_end': ar.get('end_frame', 0),
                'start_offset': s_off, 'end_offset': e_off,
                'gt_duration': gr['end_frame'] - gr['start_frame'] + 1,
                'algo_duration': ar.get('end_frame', 0) - ar.get('start_frame', 0) + 1,
            }
            all_matches.append(m)

            s_ok = abs(s_off) <= 2
            e_ok = abs(e_off) <= 2
            if not s_ok:
                start_errors += 1
            if not e_ok:
                end_errors += 1
            if s_ok and e_ok:
                both_w2 += 1

        for gi in missed_gi:
            gr = gt_reaches[gi]
            # Find nearest algo reach
            nearest_dist = 9999
            nearest_ar = None
            for ar in algo_reaches:
                d = abs(gr['start_frame'] - ar.get('start_frame', 0))
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_ar = ar
            all_missed.append({
                'video': video,
                'gt_start': gr['start_frame'], 'gt_end': gr['end_frame'],
                'gt_duration': gr['end_frame'] - gr['start_frame'] + 1,
                'nearest_algo_dist': nearest_dist,
                'nearest_algo_start': nearest_ar.get('start_frame', 0) if nearest_ar else None,
            })

        for ai in fp_ai:
            ar = algo_reaches[ai]
            all_fps.append({
                'video': video,
                'algo_start': ar.get('start_frame', 0),
                'algo_end': ar.get('end_frame', 0),
                'algo_duration': ar.get('end_frame', 0) - ar.get('start_frame', 0) + 1,
            })

        v_stats['start_errors'] = start_errors
        v_stats['end_errors'] = end_errors
        v_stats['both_w2'] = both_w2
        video_stats[video] = v_stats

    total_gt = len(all_matches) + len(all_missed)
    n_matched = len(all_matches)

    # Overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL: {total_gt} GT reaches, {n_matched} matched, "
          f"{len(all_missed)} missed, {len(all_fps)} FPs")
    print(f"{'='*80}")

    start_offsets = [m['start_offset'] for m in all_matches]
    end_offsets = [m['end_offset'] for m in all_matches]

    s_exact = sum(1 for o in start_offsets if o == 0)
    s_w2 = sum(1 for o in start_offsets if abs(o) <= 2)
    s_w5 = sum(1 for o in start_offsets if abs(o) <= 5)
    e_exact = sum(1 for o in end_offsets if o == 0)
    e_w2 = sum(1 for o in end_offsets if abs(o) <= 2)
    e_w5 = sum(1 for o in end_offsets if abs(o) <= 5)
    both_w2 = sum(1 for m in all_matches if abs(m['start_offset']) <= 2 and abs(m['end_offset']) <= 2)
    both_w5 = sum(1 for m in all_matches if abs(m['start_offset']) <= 5 and abs(m['end_offset']) <= 5)

    print(f"\n  Start exact:    {s_exact}/{n_matched} ({s_exact/n_matched*100:.1f}%)")
    print(f"  Start within 2: {s_w2}/{n_matched} ({s_w2/n_matched*100:.1f}%)")
    print(f"  End exact:      {e_exact}/{n_matched} ({e_exact/n_matched*100:.1f}%)")
    print(f"  End within 2:   {e_w2}/{n_matched} ({e_w2/n_matched*100:.1f}%)")
    print(f"  Both within 2:  {both_w2}/{n_matched} ({both_w2/n_matched*100:.1f}%)")
    print(f"  Both within 5:  {both_w5}/{n_matched} ({both_w5/n_matched*100:.1f}%)")

    # ====================================
    # START FRAME ERRORS (|offset| > 2)
    # ====================================
    start_errors = [m for m in all_matches if abs(m['start_offset']) > 2]
    print(f"\n{'='*80}")
    print(f"START FRAME ERRORS: {len(start_errors)} reaches with |start_offset| > 2")
    print(f"{'='*80}")

    if start_errors:
        print(f"\n  By offset value:")
        offset_counts = defaultdict(int)
        for m in start_errors:
            offset_counts[m['start_offset']] += 1
        for off in sorted(offset_counts.keys()):
            print(f"    offset={off:+3d}: {offset_counts[off]}")

        print(f"\n  By video:")
        video_counts = defaultdict(list)
        for m in start_errors:
            video_counts[m['video']].append(m['start_offset'])
        for v in sorted(video_counts.keys()):
            offs = video_counts[v]
            print(f"    {v}: {len(offs)} errors, offsets={sorted(offs)}")

        print(f"\n  Individual start errors:")
        for m in sorted(start_errors, key=lambda x: abs(x['start_offset']), reverse=True)[:30]:
            print(f"    {m['video']}: GT={m['gt_start']}, algo={m['algo_start']}, "
                  f"offset={m['start_offset']:+d}, gt_dur={m['gt_duration']}")

    # ====================================
    # END FRAME ERRORS (|offset| > 2)
    # ====================================
    end_errors = [m for m in all_matches if abs(m['end_offset']) > 2]
    print(f"\n{'='*80}")
    print(f"END FRAME ERRORS: {len(end_errors)} reaches with |end_offset| > 2")
    print(f"{'='*80}")

    if end_errors:
        print(f"\n  By offset value:")
        offset_counts = defaultdict(int)
        for m in end_errors:
            offset_counts[m['end_offset']] += 1
        for off in sorted(offset_counts.keys()):
            print(f"    offset={off:+3d}: {offset_counts[off]}")

        print(f"\n  By video:")
        video_counts = defaultdict(list)
        for m in end_errors:
            video_counts[m['video']].append(m['end_offset'])
        for v in sorted(video_counts.keys()):
            offs = video_counts[v]
            print(f"    {v}: {len(offs)} errors, offsets={sorted(offs)}")

        print(f"\n  By direction:")
        early = [m for m in end_errors if m['end_offset'] < -2]
        late = [m for m in end_errors if m['end_offset'] > 2]
        print(f"    Algo ends early (offset < -2): {len(early)}")
        print(f"    Algo ends late  (offset > +2): {len(late)}")

        print(f"\n  Individual end errors (top 30 by magnitude):")
        for m in sorted(end_errors, key=lambda x: abs(x['end_offset']), reverse=True)[:30]:
            print(f"    {m['video']}: GT_end={m['gt_end']}, algo_end={m['algo_end']}, "
                  f"offset={m['end_offset']:+d}, gt_dur={m['gt_duration']}, algo_dur={m['algo_duration']}")

    # ====================================
    # MISSED GT REACHES
    # ====================================
    print(f"\n{'='*80}")
    print(f"MISSED GT REACHES: {len(all_missed)}")
    print(f"{'='*80}")

    if all_missed:
        print(f"\n  By video:")
        video_counts = defaultdict(list)
        for m in all_missed:
            video_counts[m['video']].append(m)
        for v in sorted(video_counts.keys()):
            misses = video_counts[v]
            print(f"    {v}: {len(misses)} missed")
            for mi in misses[:5]:
                print(f"      GT frames {mi['gt_start']}-{mi['gt_end']} (dur={mi['gt_duration']}), "
                      f"nearest algo dist={mi['nearest_algo_dist']}")

        print(f"\n  By nearest algo distance:")
        for threshold in [5, 10, 20, 30, 50, 100, 999]:
            n = sum(1 for m in all_missed if m['nearest_algo_dist'] <= threshold)
            print(f"    Within {threshold} frames of algo reach: {n}")

        print(f"\n  Duration distribution of missed reaches:")
        durs = [m['gt_duration'] for m in all_missed]
        print(f"    Mean: {np.mean(durs):.1f}, Median: {np.median(durs):.0f}, "
              f"Min: {min(durs)}, Max: {max(durs)}")
        for bucket in [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 9999)]:
            n = sum(1 for d in durs if bucket[0] <= d <= bucket[1])
            print(f"    {bucket[0]}-{bucket[1]:>5} frames: {n}")

    # ====================================
    # FALSE POSITIVES
    # ====================================
    print(f"\n{'='*80}")
    print(f"FALSE POSITIVES: {len(all_fps)}")
    print(f"{'='*80}")

    if all_fps:
        print(f"\n  By video:")
        video_counts = defaultdict(int)
        for fp in all_fps:
            video_counts[fp['video']] += 1
        for v in sorted(video_counts.keys(), key=lambda x: video_counts[x], reverse=True):
            print(f"    {v}: {video_counts[v]} FPs")

        print(f"\n  Duration distribution of FP reaches:")
        durs = [fp['algo_duration'] for fp in all_fps]
        print(f"    Mean: {np.mean(durs):.1f}, Median: {np.median(durs):.0f}, "
              f"Min: {min(durs)}, Max: {max(durs)}")
        for bucket in [(1, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 9999)]:
            n = sum(1 for d in durs if bucket[0] <= d <= bucket[1])
            print(f"    {bucket[0]}-{bucket[1]:>5} frames: {n}")

    # ====================================
    # PER-VIDEO SUMMARY
    # ====================================
    print(f"\n{'='*80}")
    print(f"PER-VIDEO SUMMARY")
    print(f"{'='*80}")
    print(f"\n  {'Video':<35} {'GT':>4} {'Match':>5} {'Miss':>4} {'FP':>4} "
          f"{'S_err':>5} {'E_err':>5} {'Bw2':>6}")
    print(f"  {'-'*80}")

    for v in sorted(video_stats.keys()):
        s = video_stats[v]
        bw2_pct = s['both_w2'] / max(s['n_matched'], 1) * 100
        print(f"  {v:<35} {s['n_gt']:>4} {s['n_matched']:>5} {s['n_missed']:>4} "
              f"{s['n_fp']:>4} {s['start_errors']:>5} {s['end_errors']:>5} "
              f"{s['both_w2']:>3}/{s['n_matched']:<3} ({bw2_pct:>4.0f}%)")

    # ====================================
    # ERROR CATEGORIES
    # ====================================
    print(f"\n{'='*80}")
    print(f"ERROR CATEGORIES SUMMARY")
    print(f"{'='*80}")

    # Categorize start errors
    s_err_early = sum(1 for m in start_errors if m['start_offset'] < -2)
    s_err_late = sum(1 for m in start_errors if m['start_offset'] > 2)
    s_err_small = sum(1 for m in start_errors if 3 <= abs(m['start_offset']) <= 5)
    s_err_large = sum(1 for m in start_errors if abs(m['start_offset']) > 5)

    print(f"\n  Start errors ({len(start_errors)} total):")
    print(f"    Algo starts early (offset < -2): {s_err_early}")
    print(f"    Algo starts late  (offset > +2): {s_err_late}")
    print(f"    Small errors (3-5 frames):       {s_err_small}")
    print(f"    Large errors (>5 frames):        {s_err_large}")

    # Categorize end errors
    e_err_early = sum(1 for m in end_errors if m['end_offset'] < -2)
    e_err_late = sum(1 for m in end_errors if m['end_offset'] > 2)
    e_err_small = sum(1 for m in end_errors if 3 <= abs(m['end_offset']) <= 5)
    e_err_large = sum(1 for m in end_errors if abs(m['end_offset']) > 5)

    print(f"\n  End errors ({len(end_errors)} total):")
    print(f"    Algo ends early (offset < -2): {e_err_early}")
    print(f"    Algo ends late  (offset > +2): {e_err_late}")
    print(f"    Small errors (3-5 frames):     {e_err_small}")
    print(f"    Large errors (>5 frames):      {e_err_large}")

    # Overlap: cases with BOTH start and end errors
    both_errors = [m for m in all_matches if abs(m['start_offset']) > 2 and abs(m['end_offset']) > 2]
    start_only = [m for m in all_matches if abs(m['start_offset']) > 2 and abs(m['end_offset']) <= 2]
    end_only = [m for m in all_matches if abs(m['start_offset']) <= 2 and abs(m['end_offset']) > 2]

    print(f"\n  Error overlap:")
    print(f"    Start error only:  {len(start_only)}")
    print(f"    End error only:    {len(end_only)}")
    print(f"    Both start + end:  {len(both_errors)}")
    print(f"    Neither (correct): {both_w2}")

    # What would perfect start/end correction get us?
    print(f"\n  Ceiling analysis (of {total_gt} GT reaches):")
    print(f"    If all matches had perfect boundaries: {n_matched}/{total_gt} = {n_matched/total_gt*100:.1f}%")
    print(f"    Current both-w2 (of all GT):           {both_w2}/{total_gt} = {both_w2/total_gt*100:.1f}%")
    print(f"    Gap from existence:                    {n_matched - both_w2} reaches")
    print(f"    Gap from missed:                       {len(all_missed)} reaches")


if __name__ == "__main__":
    main()
