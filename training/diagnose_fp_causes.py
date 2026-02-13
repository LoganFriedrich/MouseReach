"""
Diagnose ROOT CAUSES of false positive reach detections.

Cross-references the algorithm's detection logic with DLC data at each FP
to classify WHY the algorithm triggered a false detection.

The algorithm triggers a reach when:
  1. Nose is engaged (within 25px of slit center)
  2. ANY hand point has likelihood >= 0.5

So every FP is a case where both conditions were met but humans say
it's not a real reach. This script classifies the failure mode:

  A) FLICKERING: Hand likelihood briefly crosses 0.5 threshold then drops
     (DLC tracking noise, not actual hand movement)
  B) GROOMING/POSITIONING: Hand is genuinely visible but mouse is grooming
     or repositioning, not reaching (nose near slit, hand near slit)
  C) MICRO-REACH: Very brief hand extension that doesn't constitute a
     meaningful reach attempt (duration 4-6 frames)
  D) SPLIT ARTIFACT: Part of a longer reach that got split by the algorithm
     (FP apex is near a TP reach)
  E) LOW-QUALITY TRACKING: Hand points have borderline likelihood (0.5-0.6)
     throughout - DLC isn't confident this is really a hand
  F) GENUINE AMBIGUITY: Reasonable-looking reach that humans excluded
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_0_0")

WORST_VIDEOS = ['CNT0307_P4', 'CNT0311_P2', 'CNT0110_P2',
                'CNT0309_P1', 'CNT0413_P2', 'CNT0312_P2']

RH_POINTS = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']


def load_gt_reaches(video):
    """Load GT reaches and exhaustive status.

    Returns:
        tuple: (reaches_list, is_exhaustive)
    """
    gt_file = DATA_DIR / f"{video}_unified_ground_truth.json"
    if not gt_file.exists():
        return [], False
    with open(gt_file) as f:
        gt = json.load(f)
    reaches_data = gt.get('reaches', {})
    reaches = [r for r in reaches_data.get('reaches', [])
               if r.get('start_determined', False) and r.get('end_determined', False)]
    exhaustive = reaches_data.get('exhaustive', False)
    return reaches, exhaustive


def load_algo_reaches(video):
    reach_file = ALGO_DIR / f"{video}_reaches.json"
    if not reach_file.exists():
        return [], []
    with open(reach_file) as f:
        data = json.load(f)
    all_reaches = []
    segments = []
    for seg in data.get('segments', []):
        seg_info = {
            'segment_num': seg['segment_num'],
            'start_frame': seg['start_frame'],
            'end_frame': seg['end_frame'],
        }
        segments.append(seg_info)
        for r in seg.get('reaches', []):
            r['_seg_start'] = seg['start_frame']
            r['_seg_end'] = seg['end_frame']
            all_reaches.append(r)
    return all_reaches, segments


def match_reaches(algo_reaches, gt_reaches, tolerance=5):
    gt_matched = set()
    algo_matched = set()
    for ai, ar in enumerate(algo_reaches):
        apex = ar.get('apex_frame')
        if apex is None:
            continue
        best_gi, best_dist = None, float('inf')
        for gi, gr in enumerate(gt_reaches):
            if gi in gt_matched:
                continue
            ga = gr.get('apex_frame')
            if ga is None:
                continue
            d = abs(apex - ga)
            if d <= tolerance and d < best_dist:
                best_gi, best_dist = gi, d
        if best_gi is not None:
            gt_matched.add(best_gi)
            algo_matched.add(ai)
    fp = [r for i, r in enumerate(algo_reaches) if i not in algo_matched]
    tp = [r for i, r in enumerate(algo_reaches) if i in algo_matched]
    return tp, fp


def classify_fp(fp_reach, tp_reaches, dlc_df, scorer):
    """Classify a single FP reach by its likely failure mode."""
    start = fp_reach.get('start_frame', 0)
    end = fp_reach.get('end_frame', start + 1)
    apex = fp_reach.get('apex_frame', start)
    duration = end - start + 1
    extent = fp_reach.get('max_extent_pixels', 0)

    if start >= len(dlc_df) or end >= len(dlc_df):
        return 'OUT_OF_RANGE', {}

    # Get hand likelihood profile during the reach
    hand_likelihoods = []
    max_hand_likelihoods = []
    for frame in range(start, min(end + 1, len(dlc_df))):
        frame_likes = []
        for part in RH_POINTS:
            try:
                l = dlc_df[(scorer, part, 'likelihood')].iloc[frame]
                frame_likes.append(l)
            except (KeyError, IndexError):
                continue
        if frame_likes:
            hand_likelihoods.append(np.mean(frame_likes))
            max_hand_likelihoods.append(np.max(frame_likes))
        else:
            hand_likelihoods.append(0)
            max_hand_likelihoods.append(0)

    hand_likes = np.array(hand_likelihoods)
    max_likes = np.array(max_hand_likelihoods)

    # Compute diagnostic features
    diag = {
        'duration': duration,
        'extent': extent,
        'mean_max_likelihood': np.mean(max_likes),
        'peak_max_likelihood': np.max(max_likes) if len(max_likes) > 0 else 0,
        'pct_above_0.6': np.mean(max_likes >= 0.6) * 100 if len(max_likes) > 0 else 0,
        'pct_above_0.8': np.mean(max_likes >= 0.8) * 100 if len(max_likes) > 0 else 0,
        'near_tp': False,
    }

    # Check proximity to TP reaches (split artifact detection)
    for tp in tp_reaches:
        tp_start = tp.get('start_frame', 0)
        tp_end = tp.get('end_frame', 0)
        # FP overlaps or is very close to a TP
        if (start <= tp_end + 10 and end >= tp_start - 10):
            diag['near_tp'] = True
            break

    # Classification logic
    # A) FLICKERING: Very brief high likelihood that drops quickly
    if duration <= 6 and diag['pct_above_0.8'] < 50:
        return 'FLICKERING', diag

    # B) SPLIT ARTIFACT: FP is very close to a TP reach
    if diag['near_tp'] and duration <= 15:
        return 'SPLIT_ARTIFACT', diag

    # C) MICRO-REACH: Short duration, some hand visibility
    if duration <= 8:
        return 'MICRO_REACH', diag

    # D) LOW-QUALITY TRACKING: Hand has borderline likelihood throughout
    if diag['mean_max_likelihood'] < 0.65:
        return 'LOW_QUALITY_TRACKING', diag

    # E) GROOMING/POSITIONING: Large extent but negative or very large
    #    (hand visible behind slit or tracking jumping around)
    if extent < 0 or extent > 30:
        return 'POSITION_ARTIFACT', diag

    # F) GENUINE AMBIGUITY: Looks like a real reach but human excluded it
    if diag['peak_max_likelihood'] >= 0.9 and duration >= 10:
        return 'GENUINE_AMBIGUITY', diag

    # G) If nothing else matched - moderate case
    if diag['mean_max_likelihood'] < 0.75:
        return 'LOW_QUALITY_TRACKING', diag

    return 'GENUINE_AMBIGUITY', diag


def main():
    print("FALSE POSITIVE ROOT CAUSE DIAGNOSIS")
    print("=" * 70)
    print()
    print("NOTE: False positive identification requires exhaustive reach ground truth.")
    print("Only videos where a human has determined ALL reaches (exhaustive=true)")
    print("can have valid FP analysis. For non-exhaustive videos, unmatched algo")
    print("reaches may be real reaches the human hasn't gotten to yet.")
    print()

    all_classifications = []
    all_diags = defaultdict(list)
    exhaustive_videos = []
    non_exhaustive_videos = []

    for gt_file in DATA_DIR.glob("*_unified_ground_truth.json"):
        video = gt_file.stem.replace("_unified_ground_truth", "")

        # Only process worst videos
        if not any(bad in video for bad in WORST_VIDEOS):
            continue

        gt_reaches, is_exhaustive = load_gt_reaches(video)

        # Check exhaustive flag - only proceed if reaches are exhaustive
        if not is_exhaustive:
            non_exhaustive_videos.append(video)
            print(f"{video}: Reaches not exhaustive - cannot identify false positives")
            continue

        exhaustive_videos.append(video)

        algo_reaches, segments = load_algo_reaches(video)
        if not algo_reaches:
            continue

        tp, fp = match_reaches(algo_reaches, gt_reaches)
        if not fp:
            continue

        # Load DLC data
        dlc_files = list(DATA_DIR.glob(f"*{video.split('_', 1)[1]}*DLC*.h5"))
        if not dlc_files:
            dlc_files = list(DATA_DIR.glob(f"*{video}*DLC*.h5"))
        if not dlc_files:
            print(f"  No DLC file for {video}")
            continue

        dlc_df = pd.read_hdf(dlc_files[0])
        scorer = dlc_df.columns.get_level_values(0)[0]

        print(f"\n{video}: {len(tp)} TP, {len(fp)} FP")

        video_classifications = []
        for fp_r in fp:
            cause, diag = classify_fp(fp_r, tp, dlc_df, scorer)
            video_classifications.append(cause)
            all_classifications.append(cause)
            all_diags[cause].append(diag)

        # Per-video breakdown
        counts = Counter(video_classifications)
        for cause, count in counts.most_common():
            pct = 100 * count / len(fp)
            print(f"  {cause:<25} {count:>4} ({pct:>5.1f}%)")

    # Summary
    print(f"\n\n{'=' * 70}")
    print("ANALYSIS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Videos with exhaustive GT:     {len(exhaustive_videos)}")
    print(f"Videos with non-exhaustive GT: {len(non_exhaustive_videos)}")

    # If no exhaustive videos, print warning and exit
    if not exhaustive_videos:
        print(f"\n{'=' * 70}")
        print("NO VALID FP ANALYSIS POSSIBLE")
        print(f"{'=' * 70}")
        print("None of the analyzed videos have exhaustive reach ground truth.")
        print("All 'false positives' in previous analyses were UNVALIDATED.")
        print()
        print("To enable FP analysis:")
        print("1. Open the Ground Truth Widget for a video")
        print("2. Review ALL reaches in the video (confirm or add missing ones)")
        print("3. Click 'Mark Exhaustive' for the reaches component")
        print("4. Re-run this analysis")
        print()
        print("Until then, only precision-like metrics (TP matching rate) are valid.")
        return

    # Aggregate
    print(f"\n\n{'=' * 70}")
    print("AGGREGATE ROOT CAUSE BREAKDOWN")
    print(f"{'=' * 70}")
    print(f"Total FP analyzed: {len(all_classifications)}")

    total = len(all_classifications)
    if total == 0:
        print("\nNo false positives found in exhaustive videos.")
        return

    counts = Counter(all_classifications)
    for cause, count in counts.most_common():
        pct = 100 * count / total
        print(f"\n  {cause}: {count} ({pct:.1f}%)")
        diags = all_diags[cause]
        if diags:
            durations = [d['duration'] for d in diags]
            extents = [d['extent'] for d in diags]
            mean_likes = [d['mean_max_likelihood'] for d in diags]
            print(f"    Duration: mean={np.mean(durations):.1f}, median={np.median(durations):.0f}")
            print(f"    Extent:   mean={np.mean(extents):.1f}, median={np.median(extents):.1f}")
            print(f"    MaxLike:  mean={np.mean(mean_likes):.3f}")

    # Actionable summary (only if we have valid FPs to analyze)
    print(f"\n\n{'=' * 70}")
    print("ACTIONABLE RECOMMENDATIONS")
    print(f"{'=' * 70}")

    flicker_pct = 100 * counts.get('FLICKERING', 0) / total
    micro_pct = 100 * counts.get('MICRO_REACH', 0) / total
    split_pct = 100 * counts.get('SPLIT_ARTIFACT', 0) / total
    lowq_pct = 100 * counts.get('LOW_QUALITY_TRACKING', 0) / total
    position_pct = 100 * counts.get('POSITION_ARTIFACT', 0) / total
    ambig_pct = 100 * counts.get('GENUINE_AMBIGUITY', 0) / total

    print(f"""
1. INCREASE MIN_REACH_DURATION from 4 to 7 frames
   Would eliminate: {flicker_pct + micro_pct:.0f}% of FPs (FLICKERING + MICRO_REACH)
   Current: {counts.get('FLICKERING', 0) + counts.get('MICRO_REACH', 0)} FPs
   Risk: May miss genuine very-brief reach attempts

2. INCREASE HAND_LIKELIHOOD_THRESHOLD from 0.5 to 0.6 or 0.7
   Would reduce: {lowq_pct:.0f}% of FPs (LOW_QUALITY_TRACKING)
   Current: {counts.get('LOW_QUALITY_TRACKING', 0)} FPs
   Risk: May miss reaches in poor-quality tracking segments

3. IMPROVE SPLIT/MERGE POST-PROCESSING
   Would eliminate: {split_pct:.0f}% of FPs (SPLIT_ARTIFACT)
   Current: {counts.get('SPLIT_ARTIFACT', 0)} FPs
   These are fragments near real reaches - better merge logic could absorb them

4. TIGHTEN EXTENT FILTER
   Would eliminate: {position_pct:.0f}% of FPs (POSITION_ARTIFACT)
   Current: {counts.get('POSITION_ARTIFACT', 0)} FPs
   Large positive extent (>30px) or negative extent = not real reaches

5. GENUINE AMBIGUITY (hard to fix algorithmically)
   {ambig_pct:.0f}% of FPs ({counts.get('GENUINE_AMBIGUITY', 0)}) look like real reaches
   These may need DL-based classification or human judgment
""")


if __name__ == "__main__":
    main()
