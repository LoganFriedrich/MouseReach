"""
Probe specific late-start cases to understand WHY the algo starts late.
Check nose engagement, hand visibility, and slit center at each frame.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_5_0")

RH_POINTS = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']


def load_dlc(video):
    dlc_files = list(DATA_DIR.glob(f"{video}DLC*.h5"))
    if not dlc_files:
        return None
    df = pd.read_hdf(dlc_files[0])
    if isinstance(df.columns, pd.MultiIndex):
        scorer = df.columns.get_level_values(0)[0]
        df = df[scorer]
        df.columns = [f"{bp}_{coord}" for bp, coord in df.columns]
    return df


def get_slit_center_from_algo(video):
    """Get slit center from algo results (the algo computes it per-segment)."""
    algo_file = ALGO_DIR / f"{video}_reaches.json"
    if not algo_file.exists():
        return {}
    with open(algo_file) as f:
        data = json.load(f)
    # Return segment boundaries for finding which segment a frame belongs to
    segments = {}
    for seg in data.get('segments', []):
        seg_num = seg['segment_num']
        segments[seg_num] = {
            'start': seg['start_frame'],
            'end': seg['end_frame'],
        }
    return segments


def compute_slit_center(df, seg_start, seg_end):
    """Compute slit center same way as the algorithm does."""
    segment_df = df.iloc[seg_start:seg_end]
    boxl_x = segment_df['BOXL_x'].median()
    boxl_y = segment_df['BOXL_y'].median()
    boxr_x = segment_df['BOXR_x'].median()
    boxr_y = segment_df['BOXR_y'].median()
    return (boxl_x + boxr_x) / 2, (boxl_y + boxr_y) / 2


def best_hand_info(row):
    best_l = 0
    best_x = None
    for p in RH_POINTS:
        l = row.get(f'{p}_likelihood', 0)
        if l > best_l:
            best_l = l
            best_x = row.get(f'{p}_x', np.nan)
    return best_l, best_x


def main():
    # Cases from the diagnostic: video, gt_start, algo_start
    cases = [
        ("20250701_CNT0110_P2", 9066, 9071),
        ("20250701_CNT0110_P2", 22883, 22891),
        ("20250701_CNT0110_P2", 15208, 15223),
        ("20250701_CNT0110_P2", 3373, 3400),
        ("20250711_CNT0210_P2", 8107, 8111),
        ("20250711_CNT0210_P2", 7630, 7636),
        ("20250711_CNT0210_P2", 22219, 22225),
        ("20250711_CNT0210_P2", 28710, 28716),
        ("20250711_CNT0210_P2", 1304, 1311),
        ("20250711_CNT0210_P2", 5767, 5774),
        # Also check some early cases
        ("20250701_CNT0110_P2", 15037, 15034),  # early by 3
        ("20250701_CNT0110_P2", 4095, 4091),    # early by 4
    ]

    # Load DLC data for each unique video
    dlc_cache = {}
    for video, _, _ in cases:
        if video not in dlc_cache:
            dlc_cache[video] = load_dlc(video)

    for video, gt_start, algo_start in cases:
        df = dlc_cache[video]
        if df is None:
            print(f"  No DLC data for {video}")
            continue

        offset = algo_start - gt_start
        direction = "LATE" if offset > 0 else "EARLY"

        # Get segment boundaries from algo
        segments = get_slit_center_from_algo(video)

        # Find which segment contains this reach
        seg_start = seg_end = None
        for seg_num, seg_info in segments.items():
            if seg_info['start'] <= gt_start <= seg_info['end']:
                seg_start = seg_info['start']
                seg_end = seg_info['end']
                break

        if seg_start is None:
            # Try algo_start
            for seg_num, seg_info in segments.items():
                if seg_info['start'] <= algo_start <= seg_info['end']:
                    seg_start = seg_info['start']
                    seg_end = seg_info['end']
                    break

        slit_x = slit_y = None
        if seg_start is not None:
            slit_x, slit_y = compute_slit_center(df, seg_start, seg_end)

        print(f"\n{'='*70}")
        print(f"{direction} +{abs(offset)}: {video} GT={gt_start} ALGO={algo_start}")
        if slit_x is not None:
            print(f"  Slit center: ({slit_x:.1f}, {slit_y:.1f})")
        else:
            print(f"  WARNING: No segment found for frame {gt_start}")
            # Show available segments
            for sn, si in sorted(segments.items()):
                print(f"    Segment {sn}: {si['start']}-{si['end']}")
            continue

        # Frame-by-frame analysis
        check_start = min(gt_start, algo_start) - 3
        check_end = max(gt_start, algo_start) + 3
        print(f"  Frame-by-frame ({check_start} to {check_end}):")

        for f in range(check_start, min(check_end + 1, len(df))):
            row = df.iloc[f]

            # Hand
            hand_l, hand_x = best_hand_info(row)
            hand_vis = hand_l >= 0.5

            # Nose
            nose_x = row.get('Nose_x', np.nan)
            nose_y = row.get('Nose_y', np.nan)
            nose_l = row.get('Nose_likelihood', 0)
            if nose_l >= 0.3 and not np.isnan(nose_x):
                nose_dist = np.sqrt((nose_x - slit_x)**2 + (nose_y - slit_y)**2)
                nose_engaged = nose_dist < 25
            else:
                nose_dist = None
                nose_engaged = False

            marker = ""
            if f == gt_start:
                marker = " <-- GT START"
            elif f == algo_start:
                marker = " <-- ALGO START"

            # Would both conditions be met?
            would_start = hand_vis and nose_engaged

            nose_str = f"nose={nose_dist:.1f}px {'Y' if nose_engaged else 'N'}" if nose_dist else "nose=? (low lik)"
            print(f"    f{f}: hand={hand_l:.2f}{'*' if hand_vis else ' '} "
                  f"{nose_str} "
                  f"{'START' if would_start else '     '}{marker}")


if __name__ == "__main__":
    main()
