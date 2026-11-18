#!/usr/bin/env python3
"""CLI entry points for MouseReach Export.

Both commands read each video's ``_features.json`` -- its result with every
person's answer applied -- and name the videos they left out because they have
no features file yet (export.resolved says why).
"""

import argparse
from pathlib import Path


def main_export():
    """Export data to Excel/CSV"""
    from mousereach.export.core.exporter import load_all_results, export_to_excel, export_to_csv
    from mousereach.export.resolved import print_left_out

    parser = argparse.ArgumentParser(
        description="Export MouseReach results: one row per pellet, from each video's "
                    "_features.json (the result with every person's answer applied). "
                    "A video without a features file is left out and named.")
    parser.add_argument('-i', '--input', type=Path, required=True, help="Input directory")
    parser.add_argument('-o', '--output', type=Path, required=True, help="Output file/directory")
    parser.add_argument('--format', choices=['excel', 'csv'], default='excel', help="Output format")

    args = parser.parse_args()

    left_out = []
    results = load_all_results(args.input, left_out=left_out)
    print(f"Loaded {len(results)} videos")
    print_left_out(left_out)
    if not results:
        print("No resolved results found.")
        return

    if args.format == 'excel':
        output = export_to_excel(results, args.output)
        print(f"Exported to: {output}")
    else:
        outputs = export_to_csv(results, args.output)
        for o in outputs:
            print(f"Exported to: {o}")


def main_summary():
    """Generate summary statistics"""
    from mousereach.export.core.summary import generate_summary
    from mousereach.export.resolved import print_left_out

    parser = argparse.ArgumentParser(
        description="Generate summary statistics from each video's _features.json "
                    "(the result with every person's answer applied). A video without "
                    "a features file is left out and named.")
    parser.add_argument('-i', '--input', type=Path, required=True, help="Input directory")
    parser.add_argument('-o', '--output', type=Path, help="Output JSON file")

    args = parser.parse_args()

    results = generate_summary(args.input, args.output)

    summary = results['summary']
    print_left_out(results['metadata'].get('left_out') or [])
    if not summary['total_segments']:
        print("No resolved results found.")
        return
    print(f"\nSummary Statistics")
    print(f"{'='*40}")
    print(f"Videos analyzed: {summary['n_videos']}")
    print(f"Total segments:  {summary['total_segments']}")
    print(f"Retrieved:       {summary['total_retrieved']} ({summary['total_retrieved']/summary['total_segments']*100:.1f}%)")
    print(f"Displaced:       {summary['total_displaced']} ({summary['total_displaced']/summary['total_segments']*100:.1f}%)")
    print(f"Untouched:       {summary['total_missed']} ({summary['total_missed']/summary['total_segments']*100:.1f}%)")
    print(f"Overall success: {summary['overall_success_rate']*100:.1f}%")

    if args.output:
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main_summary()
