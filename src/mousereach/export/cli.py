#!/usr/bin/env python3
"""CLI entry points for MouseReach Export"""

import argparse
from pathlib import Path


def main_export():
    """Export data to Excel/CSV"""
    from mousereach.export.core.exporter import load_all_results, export_to_excel, export_to_csv
    
    parser = argparse.ArgumentParser(description="Export MouseReach results")
    parser.add_argument('-i', '--input', type=Path, required=True, help="Input directory")
    parser.add_argument('-o', '--output', type=Path, required=True, help="Output file/directory")
    parser.add_argument('--format', choices=['excel', 'csv'], default='excel', help="Output format")
    
    args = parser.parse_args()
    
    results = load_all_results(args.input)
    print(f"Loaded {len(results)} videos")
    
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
    
    parser = argparse.ArgumentParser(description="Generate summary statistics")
    parser.add_argument('-i', '--input', type=Path, required=True, help="Input directory")
    parser.add_argument('-o', '--output', type=Path, help="Output JSON file")
    
    args = parser.parse_args()
    
    results = generate_summary(args.input, args.output)
    
    summary = results['summary']
    print(f"\nSummary Statistics")
    print(f"{'='*40}")
    print(f"Videos analyzed: {summary['n_videos']}")
    print(f"Total segments:  {summary['total_segments']}")
    print(f"Retrieved:       {summary['total_retrieved']} ({summary['total_retrieved']/summary['total_segments']*100:.1f}%)")
    print(f"Displaced:       {summary['total_displaced']} ({summary['total_displaced']/summary['total_segments']*100:.1f}%)")
    print(f"Missed:          {summary['total_missed']} ({summary['total_missed']/summary['total_segments']*100:.1f}%)")
    print(f"Overall success: {summary['overall_success_rate']*100:.1f}%")
    
    if args.output:
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main_summary()
