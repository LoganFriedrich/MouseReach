#!/usr/bin/env python3
"""Export reach-level kinematics to CSV for PI presentation"""

import json
from pathlib import Path
import csv

from mousereach.config import Paths

FPS = 30  # frames per second


def main():
    PROCESSING = Paths.PROCESSING
    OUTPUT_CSV = PROCESSING.parent / "reach_kinematics.csv"

    rows = []

    for reaches_file in sorted(PROCESSING.glob("*_reaches.json")):
        video_name = reaches_file.stem.replace("_reaches", "")

        # Load reaches
        with open(reaches_file) as f:
            reach_data = json.load(f)

        # Load matching outcomes
        outcome_file = PROCESSING / f"{video_name}_pellet_outcomes.json"
        outcomes_by_seg = {}
        if outcome_file.exists():
            with open(outcome_file) as f:
                outcome_data = json.load(f)
                for seg in outcome_data.get("segments", []):
                    outcomes_by_seg[seg["segment_num"]] = seg.get("outcome", "unknown")

        # Extract each reach
        for seg in reach_data.get("segments", []):
            seg_num = seg["segment_num"]
            outcome = outcomes_by_seg.get(seg_num, "unknown")
            ruler_px = seg.get("ruler_pixels", 34.3)  # default if missing

            for reach in seg.get("reaches", []):
                # Basic timing
                duration_frames = reach.get("duration_frames", 0)
                duration_ms = (duration_frames / FPS) * 1000

                # Extent (ruler unit = 9mm)
                extent_ruler = reach.get("max_extent_ruler", 0)
                extent_mm = extent_ruler * 9.0
                extent_px = reach.get("max_extent_pixels", 0)

                rows.append({
                    "video": video_name,
                    "segment": seg_num,
                    "reach_num": reach.get("reach_num", reach.get("reach_id")),
                    "outcome": outcome,
                    "start_frame": reach.get("start_frame"),
                    "apex_frame": reach.get("apex_frame"),
                    "end_frame": reach.get("end_frame"),
                    "duration_frames": duration_frames,
                    "duration_ms": round(duration_ms, 1),
                    "extent_pixels": round(extent_px, 2),
                    "extent_ruler_units": round(extent_ruler, 3),
                    "extent_mm": round(extent_mm, 2),
                    "source": reach.get("source", "algorithm"),
                    "human_corrected": reach.get("human_corrected", False),
                })

    # Write CSV
    if rows:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"Exported {len(rows)} reaches to: {OUTPUT_CSV}")
        print(f"\nColumns: {', '.join(rows[0].keys())}")

        # Quick stats
        print(f"\n=== QUICK STATS ===")
        durations = [r["duration_ms"] for r in rows if r["duration_ms"] > 0]
        extents = [r["extent_mm"] for r in rows if r["extent_mm"] != 0]

        if durations:
            print(f"Mean reach duration: {sum(durations)/len(durations):.1f} ms")
            print(f"Duration range: {min(durations):.0f} - {max(durations):.0f} ms")
        if extents:
            print(f"Mean reach extent: {sum(extents)/len(extents):.2f} mm")
            print(f"Extent range: {min(extents):.2f} - {max(extents):.2f} mm")

        # By outcome
        print(f"\n=== BY OUTCOME ===")
        from collections import defaultdict
        by_outcome = defaultdict(list)
        for r in rows:
            by_outcome[r["outcome"]].append(r)

        for outcome, reaches in sorted(by_outcome.items()):
            dur = [r["duration_ms"] for r in reaches if r["duration_ms"] > 0]
            ext = [r["extent_mm"] for r in reaches if r["extent_mm"] != 0]
            print(f"{outcome:20s}: {len(reaches):4d} reaches", end="")
            if dur:
                print(f" | dur={sum(dur)/len(dur):.0f}ms", end="")
            if ext:
                print(f" | ext={sum(ext)/len(ext):.2f}mm", end="")
            print()
    else:
        print("No reaches found!")


if __name__ == "__main__":
    main()
