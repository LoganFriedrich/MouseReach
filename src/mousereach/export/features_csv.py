#!/usr/bin/env python3
"""Convert grasp_features.json files to CSV for easy viewing"""

import json
from pathlib import Path
import csv

from mousereach.config import Paths


def main():
    PROCESSING = Paths.PROCESSING
    OUTPUT_CSV = PROCESSING.parent / "grasp_features.csv"

    # Find feature files
    feature_files = list(PROCESSING.glob("*_grasp_features.json"))

    if not feature_files:
        print("No *_grasp_features.json files found!")
        print("Run: mousereach-grasp-analyze -i Processing/")
        return

    rows = []

    for fpath in sorted(feature_files):
        print(f"Processing {fpath.name}...")

        with open(fpath) as f:
            data = json.load(f)

        video_name = data.get("video_name", fpath.stem.replace("_grasp_features", ""))

        for seg in data.get("segments", []):
            seg_num = seg.get("segment_num")
            outcome = seg.get("outcome")

            for reach in seg.get("reaches", []):
                row = {
                    "video": video_name,
                    "segment": seg_num,
                    "reach_id": reach.get("reach_id"),
                    "reach_num": reach.get("reach_num"),
                    "outcome": outcome,
                    "causal_reach": reach.get("causal_reach", False),
                    "is_first_reach": reach.get("is_first_reach", False),
                    "is_last_reach": reach.get("is_last_reach", False),
                    # Temporal
                    "start_frame": reach.get("start_frame"),
                    "apex_frame": reach.get("apex_frame"),
                    "end_frame": reach.get("end_frame"),
                    "duration_frames": reach.get("duration_frames"),
                    # Extent
                    "max_extent_pixels": reach.get("max_extent_pixels"),
                    "max_extent_ruler": reach.get("max_extent_ruler"),
                    "max_extent_mm": reach.get("max_extent_mm"),
                    # Velocity
                    "velocity_at_apex_px_per_frame": reach.get("velocity_at_apex_px_per_frame"),
                    "velocity_at_apex_mm_per_sec": reach.get("velocity_at_apex_mm_per_sec"),
                    "peak_velocity_px_per_frame": reach.get("peak_velocity_px_per_frame"),
                    "mean_velocity_px_per_frame": reach.get("mean_velocity_px_per_frame"),
                    # Trajectory
                    "trajectory_straightness": reach.get("trajectory_straightness"),
                    "trajectory_smoothness": reach.get("trajectory_smoothness"),
                    # Hand orientation
                    "hand_angle_at_apex_deg": reach.get("hand_angle_at_apex_deg"),
                    "hand_rotation_total_deg": reach.get("hand_rotation_total_deg"),
                    # Grasp
                    "grasp_aperture_max_mm": reach.get("grasp_aperture_max_mm"),
                    "grasp_aperture_at_contact_mm": reach.get("grasp_aperture_at_contact_mm"),
                    # Spatial
                    "apex_distance_to_pellet_mm": reach.get("apex_distance_to_pellet_mm"),
                    "lateral_deviation_mm": reach.get("lateral_deviation_mm"),
                    "nose_to_slit_at_apex_mm": reach.get("nose_to_slit_at_apex_mm"),
                    # Quality
                    "mean_likelihood": reach.get("mean_likelihood"),
                    "tracking_quality_score": reach.get("tracking_quality_score"),
                }
                rows.append(row)

    if rows:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nExported {len(rows)} reaches to: {OUTPUT_CSV}")
    else:
        print("No reach data found in feature files!")


if __name__ == "__main__":
    main()
