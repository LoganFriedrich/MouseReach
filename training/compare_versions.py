"""
Compare reach accuracy across pipeline versions.
Uses the same rigorous matching as analyze_reach_accuracy.py.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ARCHIVE_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive")

VERSIONS = [
    ("v3.5", ARCHIVE_DIR / "Pipeline_0_0"),
    ("v4.2", ARCHIVE_DIR / "Pipeline_4_2"),
    ("v5.0", ARCHIVE_DIR / "Pipeline_5_0"),
    ("v5.1", ARCHIVE_DIR / "Pipeline_5_1"),
    ("v5.2", ARCHIVE_DIR / "Pipeline_5_2"),
    ("v5.3", ARCHIVE_DIR / "Pipeline_5_3"),
    ("v5.4", ARCHIVE_DIR / "Pipeline_5_4"),
]


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
    return matches, set(range(len(gt_reaches))) - gt_used


def collect_video_data(algo_dir):
    """Collect per-video GT and algo data for a given algo directory."""
    video_data = {}
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

        algo_file = algo_dir / f"{video}_reaches.json"
        algo_data = load_json(algo_file)
        if not algo_data:
            continue

        algo_reaches = []
        for seg in algo_data.get('segments', []):
            for r in seg.get('reaches', []):
                algo_reaches.append(r)

        video_data[video] = {
            'gt_reaches': gt_reaches,
            'algo_reaches': algo_reaches,
        }
    return video_data


def analyze_data(video_data, label):
    """Analyze matches from pre-collected video data."""
    all_matches = []
    all_unmatched = []
    total_algo_reaches = 0

    for video in sorted(video_data.keys()):
        vd = video_data[video]
        gt_reaches = vd['gt_reaches']
        algo_reaches = vd['algo_reaches']
        total_algo_reaches += len(algo_reaches)

        matches, unmatched_gi = match_reaches(gt_reaches, algo_reaches)
        for gi, ai, dist in matches:
            gr = gt_reaches[gi]
            ar = algo_reaches[ai]
            all_matches.append({
                'video': video,
                'gt_start': gr['start_frame'],
                'gt_end': gr['end_frame'],
                'algo_start': ar.get('start_frame', 0),
                'algo_end': ar.get('end_frame', 0),
                'start_offset': ar.get('start_frame', 0) - gr['start_frame'],
                'end_offset': ar.get('end_frame', 0) - gr['end_frame'],
            })

        for gi in unmatched_gi:
            gr = gt_reaches[gi]
            all_unmatched.append({
                'video': video,
                'gt_start': gr['start_frame'],
                'gt_end': gr['end_frame'],
            })

    total_gt = len(all_matches) + len(all_unmatched)
    n_matched = len(all_matches)

    print(f"\n  {label}")
    print(f"  {'-' * 60}")

    # Existence
    exist_rate = n_matched / max(total_gt, 1) * 100
    print(f"  EXISTENCE: {n_matched}/{total_gt} ({exist_rate:.1f}%)")
    print(f"  Unmatched GT reaches: {len(all_unmatched)}")
    print(f"  Total algo reaches: {total_algo_reaches}")
    print(f"  False positives (algo - matched): {total_algo_reaches - n_matched}")

    if not all_matches:
        return {}

    # Start frame
    start_offsets = [m['start_offset'] for m in all_matches]
    start_exact = sum(1 for o in start_offsets if o == 0)
    start_2 = sum(1 for o in start_offsets if abs(o) <= 2)
    start_5 = sum(1 for o in start_offsets if abs(o) <= 5)
    print(f"\n  START FRAME (among {n_matched} matched):")
    print(f"    Exact:      {start_exact} ({start_exact/n_matched*100:.1f}%)")
    print(f"    Within 2:   {start_2} ({start_2/n_matched*100:.1f}%)")
    print(f"    Within 5:   {start_5} ({start_5/n_matched*100:.1f}%)")
    print(f"    Mean offset: {np.mean(start_offsets):+.2f}")

    # End frame
    end_offsets = [m['end_offset'] for m in all_matches]
    end_exact = sum(1 for o in end_offsets if o == 0)
    end_2 = sum(1 for o in end_offsets if abs(o) <= 2)
    end_5 = sum(1 for o in end_offsets if abs(o) <= 5)
    end_10 = sum(1 for o in end_offsets if abs(o) <= 10)
    algo_late = sum(1 for o in end_offsets if o > 2)
    algo_early = sum(1 for o in end_offsets if o < -2)
    print(f"\n  END FRAME (among {n_matched} matched):")
    print(f"    Exact:      {end_exact} ({end_exact/n_matched*100:.1f}%)")
    print(f"    Within 2:   {end_2} ({end_2/n_matched*100:.1f}%)")
    print(f"    Within 5:   {end_5} ({end_5/n_matched*100:.1f}%)")
    print(f"    Within 10:  {end_10} ({end_10/n_matched*100:.1f}%)")
    print(f"    Mean offset: {np.mean(end_offsets):+.2f}")
    print(f"    Median:     {np.median(end_offsets):+.0f}")
    print(f"    Algo late (>2):  {algo_late} ({algo_late/n_matched*100:.1f}%)")
    print(f"    Algo early(<-2): {algo_early} ({algo_early/n_matched*100:.1f}%)")

    # End offset distribution
    buckets = [(-999, -10), (-10, -5), (-5, -2), (-2, 3), (3, 6), (6, 11), (11, 999)]
    bucket_labels = ['<-10', '-10..-6', '-5..-3', '-2..+2', '+3..+5', '+6..+10', '>+10']
    print(f"\n  END OFFSET DISTRIBUTION:")
    for (lo, hi), bl in zip(buckets, bucket_labels):
        cnt = sum(1 for o in end_offsets if lo <= o < hi) if hi != 999 else sum(1 for o in end_offsets if o >= lo)
        if hi == 999:
            cnt = sum(1 for o in end_offsets if o > hi - 989)  # >10
        elif lo == -999:
            cnt = sum(1 for o in end_offsets if o < lo + 989)  # <-10
        else:
            cnt = sum(1 for o in end_offsets if lo <= o <= hi - 1)
        # Simpler
        pass
    # Just do it directly
    b_lt_neg10 = sum(1 for o in end_offsets if o < -10)
    b_neg10_6 = sum(1 for o in end_offsets if -10 <= o < -5)
    b_neg5_3 = sum(1 for o in end_offsets if -5 <= o < -3)
    b_ok = sum(1 for o in end_offsets if -2 <= o <= 2)
    b_3_5 = sum(1 for o in end_offsets if 3 <= o <= 5)
    b_6_10 = sum(1 for o in end_offsets if 6 <= o <= 10)
    b_gt10 = sum(1 for o in end_offsets if o > 10)
    for lbl, cnt in [('<-10', b_lt_neg10), ('-10..-6', b_neg10_6), ('-5..-3', b_neg5_3),
                     ('-2..+2', b_ok), ('+3..+5', b_3_5), ('+6..+10', b_6_10), ('>+10', b_gt10)]:
        bar = '#' * (cnt * 40 // n_matched)
        print(f"    {lbl:>8}: {cnt:>5} ({cnt/n_matched*100:>5.1f}%) {bar}")

    # Combined
    both_2 = sum(1 for m in all_matches
                 if abs(m['start_offset']) <= 2 and abs(m['end_offset']) <= 2)
    both_5 = sum(1 for m in all_matches
                 if abs(m['start_offset']) <= 5 and abs(m['end_offset']) <= 5)
    print(f"\n  BOTH START+END:")
    print(f"    Both within 2:  {both_2}/{n_matched} ({both_2/n_matched*100:.1f}%)")
    print(f"    Both within 5:  {both_5}/{n_matched} ({both_5/n_matched*100:.1f}%)")

    # Overall
    overall_2 = both_2 / max(total_gt, 1) * 100
    overall_5 = both_5 / max(total_gt, 1) * 100
    print(f"\n  OVERALL (of all {total_gt} GT reaches):")
    print(f"    Exist + both within 2: {both_2}/{total_gt} ({overall_2:.1f}%)")
    print(f"    Exist + both within 5: {both_5}/{total_gt} ({overall_5:.1f}%)")

    return {
        'total_gt': total_gt, 'matched': n_matched,
        'start_exact': start_exact, 'start_2': start_2,
        'end_exact': end_exact, 'end_2': end_2, 'end_5': end_5,
        'both_2': both_2, 'both_5': both_5,
        'end_mean': np.mean(end_offsets), 'end_median': np.median(end_offsets),
        'algo_reaches': total_algo_reaches,
    }


def main():
    version_labels = [label for label, _ in VERSIONS]
    print(f"REACH DETECTION: {' vs '.join(version_labels)} COMPARISON")
    print("=" * 70)

    # Collect data for all versions
    all_data = {}
    for label, path in VERSIONS:
        data = collect_video_data(path)
        all_data[label] = data
        print(f"  {label}: {len(data)} videos")

    # Find intersection of ALL versions
    video_sets = [set(d.keys()) for d in all_data.values()]
    common_videos = video_sets[0]
    for vs in video_sets[1:]:
        common_videos = common_videos & vs
    print(f"\n  Common videos across all versions: {len(common_videos)}")

    # Show per-version analysis on common videos
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON ON {len(common_videos)} COMMON VIDEOS")
    print(f"{'=' * 70}")

    results = {}
    for label, _ in VERSIONS:
        common_data = {v: all_data[label][v] for v in common_videos}
        results[label] = analyze_data(common_data, f"{label}")

    # Comparison summary table
    if all(results.values()):
        tg = results[version_labels[0]]['total_gt']
        print(f"\n{'=' * 70}")
        print(f"COMPARISON SUMMARY ({len(common_videos)} common videos, {tg} GT reaches)")
        print(f"{'=' * 70}")

        # Header
        header = f"  {'Metric':<28}"
        for label in version_labels:
            header += f" {label:>14}"
        print(f"\n{header}")
        print(f"  {'-' * (28 + 15 * len(version_labels))}")

        metrics = [
            ('Existence', 'matched'),
            ('Start exact', 'start_exact'),
            ('Start within 2', 'start_2'),
            ('End exact', 'end_exact'),
            ('End within 2', 'end_2'),
            ('End within 5', 'end_5'),
            ('Both within 2', 'both_2'),
            ('Both within 5', 'both_5'),
        ]

        for label, key in metrics:
            row = f"  {label:<28}"
            for vlabel in version_labels:
                val = results[vlabel][key]
                pct = val / max(tg, 1) * 100
                row += f" {val:>5} ({pct:>4.1f}%)"
            print(row)

        # End offset summary
        print()
        row_mean = f"  {'End offset mean':<28}"
        row_med = f"  {'End offset median':<28}"
        row_fp = f"  {'Algo reaches (total)':<28}"
        for vlabel in version_labels:
            row_mean += f" {results[vlabel]['end_mean']:>+13.1f}"
            row_med += f" {results[vlabel]['end_median']:>+13.0f}"
            row_fp += f" {results[vlabel]['algo_reaches']:>14}"
        print(row_mean)
        print(row_med)
        print(row_fp)

    # Show latest version on ALL its videos
    latest_label, latest_path = VERSIONS[-1]
    latest_data = all_data[latest_label]
    if len(latest_data) > len(common_videos):
        extra = len(latest_data) - len(common_videos)
        print(f"\n\n{'=' * 70}")
        print(f"  {latest_label} ON ALL {len(latest_data)} VIDEOS (+{extra} not in all versions)")
        print(f"{'=' * 70}")
        analyze_data(latest_data, f"{latest_label} (all videos)")


if __name__ == "__main__":
    main()
