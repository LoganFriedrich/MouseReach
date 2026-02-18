#!/usr/bin/env python
"""
Create spaghetti plots of false positive vs true positive reaches.

Shows hand X position over time for all FP and TP reaches overlaid,
to visualize the difference in trajectory patterns.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Constants
REACH_TOLERANCE = 10  # frames - same as other analysis scripts


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def matches_gt(algo_reach, gt_reaches, tolerance=REACH_TOLERANCE):
    """
    Check if algo reach matches any GT reach within tolerance.

    Returns True if match found, False otherwise.
    """
    algo_start = algo_reach['start_frame']
    algo_end = algo_reach['end_frame']

    for gt in gt_reaches:
        gt_start = gt['start_frame']
        gt_end = gt['end_frame']

        start_match = abs(algo_start - gt_start) <= tolerance
        end_match = abs(algo_end - gt_end) <= tolerance

        if start_match and end_match:
            return True

    return False


def load_dlc_csv(csv_path):
    """
    Load DLC CSV with multi-level header and flatten to simple column names.

    Returns DataFrame with columns like 'RightHand_x', 'RightHand_y', 'RightHand_likelihood'.
    """
    # Read with multi-level header
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    # Flatten column names
    new_cols = []
    for col in df.columns:
        scorer, bodypart, coord = col
        new_cols.append(f"{bodypart}_{coord}")

    df.columns = new_cols

    return df


def compute_hand_xy(dlc_df, frame_idx, slit_x, slit_y, likelihood_threshold=0.5):
    """
    Compute hand X and Y position for a given frame, relative to slit center.

    Uses ONLY RightHand (single point) to avoid phantom trends from
    different hand points appearing/disappearing across frames.

    Returns (x_offset, y_offset) or (NaN, NaN) if not visible.
    """
    if frame_idx >= len(dlc_df):
        return np.nan, np.nan

    x_col = 'RightHand_x'
    y_col = 'RightHand_y'
    lik_col = 'RightHand_likelihood'

    if x_col not in dlc_df.columns or lik_col not in dlc_df.columns:
        return np.nan, np.nan

    lik_val = dlc_df.loc[frame_idx, lik_col]
    if lik_val < likelihood_threshold:
        return np.nan, np.nan

    x_val = dlc_df.loc[frame_idx, x_col]
    y_val = dlc_df.loc[frame_idx, y_col]
    return x_val - slit_x, y_val - slit_y


def compute_slit_geometry(dlc_df):
    """
    Compute slit center (x, y) and boxr_x from DLC data.

    slit_x = midpoint of box left and right X edges
    slit_y = midpoint of box left and right Y edges
    boxr_x = right edge of slit
    """
    boxl_x = dlc_df['BOXL_x'].median()
    boxr_x = dlc_df['BOXR_x'].median()
    boxl_y = dlc_df['BOXL_y'].median()
    boxr_y = dlc_df['BOXR_y'].median()

    slit_x = (boxl_x + boxr_x) / 2
    slit_y = (boxl_y + boxr_y) / 2

    return slit_x, slit_y, boxr_x


def _smooth(arr, window=3):
    """NaN-aware centered moving average."""
    if window <= 1 or len(arr) < window:
        return arr
    out = np.copy(arr)
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        valid = arr[lo:hi]
        valid = valid[~np.isnan(valid)]
        out[i] = np.mean(valid) if len(valid) > 0 else np.nan
    return out


def extract_reach_trajectory_xy(dlc_df, reach, slit_x, slit_y, smooth_window=3):
    """
    Extract hand X/Y trajectory for a single reach.

    Uses single point (RightHand) with 3-frame moving average smoothing.
    Positions are relative to slit center.

    Returns:
        x_offsets: array of hand X offsets from slit (positive = toward pellet)
        y_offsets: array of hand Y offsets from slit (positive = downward in image coords)
    """
    start = reach['start_frame']
    end = reach['end_frame']

    xs = []
    ys = []

    for frame_abs in range(start, end + 1):
        x_off, y_off = compute_hand_xy(dlc_df, frame_abs, slit_x, slit_y)
        xs.append(x_off)
        ys.append(y_off)

    xs = _smooth(np.array(xs), smooth_window)
    ys = _smooth(np.array(ys), smooth_window)

    return xs, ys


def main():
    print("="*80, flush=True)
    print("SPAGHETTI PLOT: X vs Y spatial paths (FP vs TP)", flush=True)
    print("="*80, flush=True)

    from mousereach.config import require_processing_root
    processing_dir = require_processing_root() / "Processing"
    print(f"\nProcessing directory: {processing_dir}", flush=True)

    gt_files = sorted(processing_dir.glob("*_unified_ground_truth.json"))
    print(f"\nFound {len(gt_files)} ground truth files", flush=True)

    if not gt_files:
        print("ERROR: No ground truth files found!", flush=True)
        return

    # Collect XY trajectories: list of (x_offsets, y_offsets)
    fp_trajectories = []
    tp_trajectories = []

    print("\nProcessing videos:", flush=True)
    print("-"*80, flush=True)

    for gt_file in gt_files:
        video_name = gt_file.stem.replace('_unified_ground_truth', '')
        print(f"  {video_name}...", end=" ", flush=True)

        gt_data = load_json(gt_file)
        gt_reaches = gt_data.get('reaches', {}).get('reaches', [])

        algo_file = gt_file.parent / f"{video_name}_reaches.json"
        if not algo_file.exists():
            print("SKIP (no algo file)", flush=True)
            continue

        algo_data = load_json(algo_file)
        algo_reaches = []
        for segment in algo_data.get('segments', []):
            algo_reaches.extend(segment.get('reaches', []))

        dlc_pattern = f"{video_name}DLC*.csv"
        dlc_files = list(processing_dir.glob(dlc_pattern))
        if not dlc_files:
            print(f"SKIP (no DLC CSV)", flush=True)
            continue

        try:
            dlc_df = load_dlc_csv(dlc_files[0])
        except Exception as e:
            print(f"SKIP (DLC load error: {e})", flush=True)
            continue

        slit_x, slit_y, boxr_x = compute_slit_geometry(dlc_df)

        n_tp = 0
        n_fp = 0

        for reach in algo_reaches:
            is_tp = matches_gt(reach, gt_reaches)
            try:
                xs, ys = extract_reach_trajectory_xy(dlc_df, reach, slit_x, slit_y)
                if is_tp:
                    tp_trajectories.append((xs, ys))
                    n_tp += 1
                else:
                    fp_trajectories.append((xs, ys))
                    n_fp += 1
            except Exception as e:
                print(f"\n    WARNING: {e}", flush=True)
                continue

        print(f"{n_tp} TP, {n_fp} FP", flush=True)

    print("\n" + "="*80, flush=True)
    print(f"Total: {len(tp_trajectories)} TP, {len(fp_trajectories)} FP", flush=True)
    print("="*80, flush=True)

    if not fp_trajectories and not tp_trajectories:
        print("\nERROR: No trajectories collected!", flush=True)
        return

    # --- Create plot: X vs Y spatial paths ---
    print("\nCreating plot...", flush=True)

    plt.style.use('dark_background')
    fig, (ax_fp, ax_tp) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot FP spatial paths
    print(f"  Plotting {len(fp_trajectories)} FP paths...", flush=True)
    for xs, ys in fp_trajectories:
        ax_fp.plot(xs, ys, color='#FF6B6B', alpha=0.08, linewidth=0.8)

    # Plot TP spatial paths (sample if needed)
    import random
    if len(tp_trajectories) > 500:
        print(f"  Sampling 500 of {len(tp_trajectories)} TP paths...", flush=True)
        tp_sample = random.sample(tp_trajectories, 500)
    else:
        tp_sample = tp_trajectories

    print(f"  Plotting {len(tp_sample)} TP paths...", flush=True)
    for xs, ys in tp_sample:
        ax_tp.plot(xs, ys, color='#4ECDC4', alpha=0.08, linewidth=0.8)

    # Reference lines: slit center crosshair
    for ax in [ax_fp, ax_tp]:
        ax.axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.4, label='Slit X center')
        ax.axhline(y=0, color='white', linestyle=':', linewidth=0.8, alpha=0.3, label='Slit Y center')
        ax.set_xlabel('Hand X offset from slit (px)\n← toward box | toward pellet →', fontsize=11)
        ax.set_ylabel('Hand Y offset from slit (px)\n← up | down →', fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15)

    ax_fp.set_title(f'False Positives (N={len(fp_trajectories)})', fontsize=14, fontweight='bold')
    tp_suffix = f" (sampled from {len(tp_trajectories)})" if len(tp_sample) < len(tp_trajectories) else ""
    ax_tp.set_title(f'True Positives (N={len(tp_sample)}){tp_suffix}', fontsize=14, fontweight='bold')

    # Match axes
    for ax in [ax_fp, ax_tp]:
        ax.set_xlim(-30, 50)
        ax.set_ylim(-40, 40)

    plt.tight_layout()

    # Save
    output_dir = require_processing_root() / "eval_reports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "fp_spaghetti_xy.png"

    print(f"\nSaving to {output_path}...", flush=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    print("Saved!", flush=True)

    print(f"\nOpening plot...", flush=True)
    os.startfile(str(output_path))

    print("\nCOMPLETE", flush=True)


if __name__ == '__main__':
    main()
