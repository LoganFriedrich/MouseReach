#!/usr/bin/env python3
"""Compute REAL kinematics from DLC tracking data"""

import json
from pathlib import Path
import csv
import numpy as np
import pandas as pd

from mousereach.config import Paths

FPS = 30
RULER_MM = 9.0  # ruler = 9mm


def compute_kinematics(x, y, ruler_px, fps=30):
    """Compute kinematic features from x,y trajectory."""
    n = len(x)
    if n < 3:
        return {}

    # Convert to mm
    px_to_mm = RULER_MM / ruler_px if ruler_px > 0 else 1.0
    x_mm = x * px_to_mm
    y_mm = y * px_to_mm

    # Frame-to-frame displacements
    dx = np.diff(x_mm)
    dy = np.diff(y_mm)

    # Instantaneous speed (mm/frame -> mm/sec)
    inst_speed = np.sqrt(dx**2 + dy**2) * fps

    # Path length (total distance traveled)
    path_length = np.sum(np.sqrt(dx**2 + dy**2))

    # Direct distance (start to max extent)
    max_x_idx = np.argmax(x_mm)
    direct_dist = np.sqrt((x_mm[max_x_idx] - x_mm[0])**2 + (y_mm[max_x_idx] - y_mm[0])**2)

    # Straightness (1.0 = perfectly straight)
    straightness = direct_dist / path_length if path_length > 0 else 0

    # Velocity components
    vx = dx * fps  # mm/sec
    vy = dy * fps

    # Acceleration (mm/sec^2)
    if len(vx) > 1:
        ax = np.diff(vx) * fps
        ay = np.diff(vy) * fps
        accel_mag = np.sqrt(ax**2 + ay**2)
    else:
        accel_mag = np.array([0])

    # Jerk (rate of acceleration change, mm/sec^3)
    if len(accel_mag) > 1:
        jerk = np.abs(np.diff(accel_mag)) * fps
        mean_jerk = np.mean(jerk)
    else:
        mean_jerk = 0

    # Movement smoothness (normalized jerk - lower = smoother)
    # Using SPARC-like metric: smoothness based on spectral arc length would be better
    # but this is a simple approximation
    duration_sec = n / fps
    if duration_sec > 0 and path_length > 0:
        norm_jerk = mean_jerk * (duration_sec**3) / path_length
    else:
        norm_jerk = 0

    return {
        "path_length_mm": round(path_length, 3),
        "direct_distance_mm": round(direct_dist, 3),
        "straightness": round(straightness, 3),
        "peak_speed_mm_s": round(np.max(inst_speed), 2) if len(inst_speed) > 0 else 0,
        "mean_speed_mm_s": round(np.mean(inst_speed), 2) if len(inst_speed) > 0 else 0,
        "peak_accel_mm_s2": round(np.max(accel_mag), 2) if len(accel_mag) > 0 else 0,
        "mean_accel_mm_s2": round(np.mean(accel_mag), 2) if len(accel_mag) > 0 else 0,
        "mean_jerk_mm_s3": round(mean_jerk, 2),
        "norm_jerk": round(norm_jerk, 4),
    }


def load_dlc_data(dlc_path):
    """Load DLC tracking data from .h5 file."""
    df = pd.read_hdf(dlc_path)
    # Flatten multi-index columns
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df


def get_hand_trajectory(dlc_df, start_frame, end_frame):
    """Extract hand position for reach frames."""
    # Try different hand point names
    hand_cols = [
        ('RightHand_x', 'RightHand_y'),
        ('RHOut_x', 'RHOut_y'),
        ('RHLeft_x', 'RHLeft_y'),
    ]

    for x_col, y_col in hand_cols:
        # Find matching columns (may have scorer prefix)
        x_matches = [c for c in dlc_df.columns if x_col in c and '_likelihood' not in c]
        y_matches = [c for c in dlc_df.columns if y_col in c and '_likelihood' not in c]

        if x_matches and y_matches:
            x = dlc_df[x_matches[0]].iloc[start_frame:end_frame+1].values
            y = dlc_df[y_matches[0]].iloc[start_frame:end_frame+1].values

            # Filter out NaN/invalid values
            valid = ~(np.isnan(x) | np.isnan(y))
            if np.sum(valid) > 3:
                return x[valid], y[valid]

    return None, None


def main():
    PROCESSING = Paths.PROCESSING
    OUTPUT_CSV = PROCESSING.parent / "real_kinematics.csv"

    rows = []

    print("Computing real kinematics from DLC data...")

    for reaches_file in sorted(PROCESSING.glob("*_reaches.json")):
        video_name = reaches_file.stem.replace("_reaches", "")
        print(f"  Processing {video_name}...")

        # Find DLC file
        dlc_files = list(PROCESSING.glob(f"{video_name}*DLC*.h5"))
        if not dlc_files:
            print(f"    No DLC file found, skipping")
            continue

        dlc_df = load_dlc_data(dlc_files[0])

        # Load reaches
        with open(reaches_file) as f:
            reach_data = json.load(f)

        # Load outcomes
        outcome_file = PROCESSING / f"{video_name}_pellet_outcomes.json"
        outcomes_by_seg = {}
        if outcome_file.exists():
            with open(outcome_file) as f:
                for seg in json.load(f).get("segments", []):
                    outcomes_by_seg[seg["segment_num"]] = seg.get("outcome", "unknown")

        # Process each reach
        for seg in reach_data.get("segments", []):
            seg_num = seg["segment_num"]
            outcome = outcomes_by_seg.get(seg_num, "unknown")
            ruler_px = seg.get("ruler_pixels", 34.3)

            for reach in seg.get("reaches", []):
                start = reach.get("start_frame")
                end = reach.get("end_frame")

                if start is None or end is None:
                    continue

                # Get hand trajectory
                x, y = get_hand_trajectory(dlc_df, start, end)

                if x is None:
                    continue

                # Compute kinematics
                kin = compute_kinematics(x, y, ruler_px, FPS)

                row = {
                    "video": video_name,
                    "segment": seg_num,
                    "reach_num": reach.get("reach_num", reach.get("reach_id")),
                    "outcome": outcome,
                    "duration_ms": round((end - start) / FPS * 1000, 1),
                    "n_frames": end - start + 1,
                    **kin
                }
                rows.append(row)

    # Write CSV
    if rows:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nExported {len(rows)} reaches to: {OUTPUT_CSV}")

        # Summary stats
        print(f"\n{'='*60}")
        print("KINEMATIC SUMMARY")
        print(f"{'='*60}")

        for key in ["path_length_mm", "straightness", "peak_speed_mm_s", "mean_speed_mm_s",
                    "peak_accel_mm_s2", "mean_jerk_mm_s3"]:
            vals = [r[key] for r in rows if r.get(key, 0) != 0]
            if vals:
                print(f"{key:25s}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}, range=[{min(vals):.2f}, {max(vals):.2f}]")

        # By outcome
        print(f"\n{'='*60}")
        print("BY OUTCOME (mean values)")
        print(f"{'='*60}")
        from collections import defaultdict
        by_outcome = defaultdict(list)
        for r in rows:
            by_outcome[r["outcome"]].append(r)

        print(f"{'Outcome':20s} {'N':>6s} {'PathLen':>10s} {'Straight':>10s} {'PeakSpd':>10s} {'PeakAcc':>10s}")
        print("-" * 70)
        for outcome in sorted(by_outcome.keys()):
            reaches = by_outcome[outcome]
            n = len(reaches)
            pl = np.mean([r["path_length_mm"] for r in reaches])
            st = np.mean([r["straightness"] for r in reaches])
            ps = np.mean([r["peak_speed_mm_s"] for r in reaches])
            pa = np.mean([r["peak_accel_mm_s2"] for r in reaches])
            print(f"{outcome:20s} {n:6d} {pl:10.2f} {st:10.3f} {ps:10.1f} {pa:10.1f}")
    else:
        print("No reaches with valid tracking data found!")


if __name__ == "__main__":
    main()
