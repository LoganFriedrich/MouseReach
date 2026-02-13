"""
Diagnose why v5.0 multi-signal splitter produces same results as v4.2.

For each video, run the splitter on reaches > 25 frames and report:
- How many reaches are split candidates
- How many confidence dips are found
- What scores they get
- Where boundaries are placed vs old splitter
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(r"Y:\2_Connectome\Behavior\MouseReach\src")))

from mousereach.reach.core.boundary_refiner import (
    compute_frame_signals, _find_confidence_dips, _score_candidate,
    _find_precise_boundary, split_reach_boundaries
)

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
# v4.3 had splitter DISABLED - these are raw state machine reaches (pre-split)
ALGO_PRESPLIT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_4_3")

RH_POINTS = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_slit_center(df, seg_start, seg_end):
    segment_df = df.iloc[seg_start:seg_end]
    boxl_x = segment_df['BOXL_x'].median()
    boxr_x = segment_df['BOXR_x'].median()
    return (boxl_x + boxr_x) / 2


def main():
    print("SPLITTER v5.0 DIAGNOSTIC")
    print("=" * 70)

    total_long_reaches = 0
    total_with_dips = 0
    total_candidates = 0
    score_distribution = []
    score_above_threshold = 0
    boundary_differences = []
    position_data_availability = []

    # Process a few videos to understand patterns
    for gt_file in sorted(DATA_DIR.glob("*_unified_ground_truth.json"))[:5]:
        gt = load_json(gt_file)
        if not gt:
            continue
        video = gt['video_name']

        # Load v4.3 reaches (splitter DISABLED = raw state machine output)
        presplit_file = ALGO_PRESPLIT_DIR / f"{video}_reaches.json"
        presplit = load_json(presplit_file)
        if not presplit:
            continue

        v42_reaches = []
        for seg in presplit.get('segments', []):
            for r in seg.get('reaches', []):
                v42_reaches.append(r)

        # Load DLC
        dlc_files = list(DATA_DIR.glob(f"{video}DLC*.h5"))
        if not dlc_files:
            continue
        df = pd.read_hdf(dlc_files[0])
        if isinstance(df.columns, pd.MultiIndex):
            scorer = df.columns.get_level_values(0)[0]
            df = df[scorer]
            df.columns = [f"{bp}_{coord}" for bp, coord in df.columns]

        # Load segments for slit position
        seg_data = load_json(DATA_DIR / f"{video}_segments.json")
        segments = seg_data.get('segments', []) if seg_data else []

        print(f"\n--- {video} ---")

        # For each v4.2 reach that's long enough to split
        for ar in v42_reaches:
            start = ar.get('start_frame', 0)
            end = ar.get('end_frame', 0)
            duration = end - start + 1

            if duration <= 25:
                continue

            total_long_reaches += 1

            # Find the segment this reach belongs to (for slit position)
            slit_x = None
            for seg in segments:
                if seg.get('start_frame', 0) <= start <= seg.get('end_frame', len(df)):
                    slit_x = get_slit_center(df, seg['start_frame'], seg['end_frame'])
                    break
            if slit_x is None:
                slit_x = get_slit_center(df, 0, len(df))

            # Compute signals
            signals = compute_frame_signals(df, start, end, slit_x, RH_POINTS)
            if len(signals) < 2:
                continue

            # Find confidence dips
            candidates = _find_confidence_dips(signals, 0.5, 0.35)

            if candidates:
                total_with_dips += 1

            for c in candidates:
                total_candidates += 1
                score = _score_candidate(c, slit_x)
                c.score = score
                score_distribution.append(score)

                if score >= 0.5:
                    score_above_threshold += 1

                # Check position data availability in dip region
                region = signals[c.drop_idx:c.rise_idx + 1]
                n_with_position = sum(1 for s in region if s.hand_x is not None)
                n_total = len(region)
                position_data_availability.append(n_with_position / max(n_total, 1))

                # Compute boundary differences
                end_first, start_second = _find_precise_boundary(c, signals)
                boundary_differences.append({
                    'video': video,
                    'reach_start': start,
                    'reach_end': end,
                    'drop_frame': c.drop_frame,
                    'rise_frame': c.rise_frame,
                    'min_conf': c.min_conf,
                    'pre_max_x': c.pre_max_x,
                    'min_hand_x': c.min_hand_x,
                    'score': score,
                    'new_boundary': end_first,
                    'has_vel_rev': c.has_velocity_reversal,
                    'position_avail': n_with_position / max(n_total, 1),
                })

            if total_long_reaches <= 3 and candidates:
                print(f"  Reach [{start}-{end}] dur={duration}")
                for c in candidates:
                    print(f"    Dip: frames {c.drop_frame}-{c.rise_frame}, "
                          f"min_conf={c.min_conf:.2f}, score={c.score:.2f}")
                    print(f"      pre_max_x={c.pre_max_x:.1f}, "
                          f"min_hand_x={c.min_hand_x:.1f if c.min_hand_x else 'None'}, "
                          f"vel_rev={c.has_velocity_reversal}")
                    end_first, _ = _find_precise_boundary(c, signals)
                    print(f"      New boundary: {end_first} (drop was at {c.drop_frame})")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Long reaches (>25 frames): {total_long_reaches}")
    print(f"With confidence dips: {total_with_dips}")
    print(f"Total dip candidates: {total_candidates}")
    print(f"Score >= 0.5 (would split): {score_above_threshold}")
    print(f"Score < 0.5 (rejected): {total_candidates - score_above_threshold}")

    if score_distribution:
        print(f"\nScore distribution:")
        for lo, hi, label in [(0, 0.1, '0.0-0.1'), (0.1, 0.2, '0.1-0.2'),
                               (0.2, 0.3, '0.2-0.3'), (0.3, 0.4, '0.3-0.4'),
                               (0.4, 0.5, '0.4-0.5'), (0.5, 0.6, '0.5-0.6'),
                               (0.6, 0.7, '0.6-0.7'), (0.7, 0.8, '0.7-0.8'),
                               (0.8, 0.9, '0.8-0.9'), (0.9, 1.01, '0.9-1.0')]:
            cnt = sum(1 for s in score_distribution if lo <= s < hi)
            bar = '#' * (cnt * 40 // len(score_distribution)) if score_distribution else ''
            print(f"  {label}: {cnt:>5} ({cnt/len(score_distribution)*100:>5.1f}%) {bar}")

    if position_data_availability:
        print(f"\nPosition data availability in dip regions:")
        print(f"  Mean: {np.mean(position_data_availability):.1%}")
        print(f"  Median: {np.median(position_data_availability):.1%}")
        print(f"  =0%: {sum(1 for p in position_data_availability if p == 0)}")
        print(f"  >50%: {sum(1 for p in position_data_availability if p > 0.5)}")
        print(f"  100%: {sum(1 for p in position_data_availability if p == 1.0)}")

    # Show sample boundaries
    if boundary_differences:
        print(f"\nSample split candidates:")
        for bd in boundary_differences[:10]:
            print(f"  {bd['video']}: reach [{bd['reach_start']}-{bd['reach_end']}]")
            print(f"    Dip: {bd['drop_frame']}-{bd['rise_frame']}, "
                  f"min_conf={bd['min_conf']:.2f}, score={bd['score']:.2f}")
            print(f"    pre_max_x={bd['pre_max_x']:.1f}, "
                  f"min_hand_x={bd['min_hand_x']:.1f if bd['min_hand_x'] else 'None'}")
            print(f"    New boundary: {bd['new_boundary']}, "
                  f"vel_rev={bd['has_vel_rev']}, "
                  f"pos_avail={bd['position_avail']:.0%}")


if __name__ == "__main__":
    main()
