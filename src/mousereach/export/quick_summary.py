#!/usr/bin/env python3
"""Quick summary for PI - run this to generate a CSV and stats.

Reads each video's _features.json in the local Processing folder: the result
with every person's answer applied. A video without a features file (its
kinematics have not run, or were held) is left out and named.
"""

from collections import Counter
import csv

from mousereach.config import Paths


def main():
    # --help answers before any work: generating the CLI reference runs every
    # command with --help, and this one used to run instead (2026-10-22).
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    from mousereach.export.resolved import load_resolved_videos, print_left_out
    PROCESSING = Paths.PROCESSING
    OUTPUT_CSV = PROCESSING.parent / "summary_for_PI.csv"

    rows = []
    total_reaches = 0
    total_segments = 0
    outcome_counts = Counter()

    print("=" * 60)
    print("MOUSEREACH SUMMARY")
    print("=" * 60)

    # WHY the features files: the pellet-outcomes file is the algorithm's call
    # before any person's answer, so a pellet a reviewer corrected was counted
    # with the algorithm's call (export.resolved).
    videos, left_out = load_resolved_videos(PROCESSING, pellets_only=True)
    for data in videos:
        video_name = data.get("video_name")
        segments = data.get("segments", [])
        # Reaches measured, plus those a person took out of the measurements
        # (ignore window or exclusion) -- every reach that happened.
        n_reaches = sum(len(s.get("reaches") or []) + int(s.get("n_reaches_left_out") or 0)
                        for s in segments)

        n_segments = len(segments)
        outcomes = [s.get("outcome") or "unknown" for s in segments]

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

    print_left_out(left_out)
    if not rows:
        print("\nNo resolved results found.")
        return

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nProcessed {len(rows)} videos")
    print(f"Total segments (trials): {total_segments}")
    print(f"Total reaches: {total_reaches}")
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
