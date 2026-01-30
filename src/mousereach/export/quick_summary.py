#!/usr/bin/env python3
"""Quick summary for PI - run this to generate a CSV and stats"""

import json
from pathlib import Path
from collections import Counter
import csv

from mousereach.config import Paths


def main():
    PROCESSING = Paths.PROCESSING
    OUTPUT_CSV = PROCESSING.parent / "summary_for_PI.csv"

    rows = []
    total_reaches = 0
    total_segments = 0
    outcome_counts = Counter()

    print("=" * 60)
    print("MOUSEREACH SUMMARY")
    print("=" * 60)

    # Process each outcome file
    for outcome_file in sorted(PROCESSING.glob("*_pellet_outcomes.json")):
        video_name = outcome_file.stem.replace("_pellet_outcomes", "")

        with open(outcome_file) as f:
            data = json.load(f)

        # Get matching reaches file
        reaches_file = PROCESSING / f"{video_name}_reaches.json"
        n_reaches = 0
        if reaches_file.exists():
            with open(reaches_file) as f:
                reach_data = json.load(f)
                # Count reaches across all segments
                for seg in reach_data.get("segments", []):
                    n_reaches += len(seg.get("reaches", []))

        # Count outcomes
        segments = data.get("segments", [])
        n_segments = len(segments)
        outcomes = [s.get("outcome", "unknown") for s in segments]

        for o in outcomes:
            outcome_counts[o] += 1

        # Success rate for this video
        retrieved = outcomes.count("retrieved")
        success_rate = retrieved / n_segments * 100 if n_segments > 0 else 0

        rows.append({
            "video": video_name,
            "segments": n_segments,
            "reaches": n_reaches,
            "retrieved": retrieved,
            "displaced_sa": outcomes.count("displaced_sa"),
            "displaced_outside": outcomes.count("displaced_outside"),
            "untouched": outcomes.count("untouched"),
            "success_rate": f"{success_rate:.1f}%"
        })

        total_reaches += n_reaches
        total_segments += n_segments

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nProcessed {len(rows)} videos")
    print(f"Total segments (trials): {total_segments}")
    print(f"Total reaches detected: {total_reaches}")
    print(f"\nOUTCOME BREAKDOWN:")
    print("-" * 40)
    for outcome, count in sorted(outcome_counts.items()):
        pct = count / total_segments * 100 if total_segments > 0 else 0
        print(f"  {outcome:20s}: {count:4d} ({pct:5.1f}%)")

    overall_success = outcome_counts.get("retrieved", 0) / total_segments * 100 if total_segments > 0 else 0
    print(f"\nOVERALL SUCCESS RATE: {overall_success:.1f}%")
    print(f"\nCSV saved to: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
