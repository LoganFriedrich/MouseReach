#!/usr/bin/env python3
"""
CLI entry points for MouseReach Step 5: Feature Extraction

Extracts kinematic features from reaches linked to pellet outcomes.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .core import FeatureExtractor


def main_batch():
    """Batch feature extraction from validated outcomes and reaches."""
    parser = argparse.ArgumentParser(
        description="Extract features from reaches linked to outcomes (Step 5)"
    )
    parser.add_argument(
        '-i', '--input',
        type=Path,
        required=True,
        help='Input directory with validated reaches (Step 3) and outcomes (Step 4)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output directory (default: Step5_Features/)'
    )
    parser.add_argument(
        '-s', '--suffix',
        default='*_pellet_outcomes.json',
        help='File pattern for outcome files'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing feature files'
    )

    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve() if args.output else input_dir

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find outcome files
    outcome_files = sorted(input_dir.glob(args.suffix))

    if not outcome_files:
        print(f"ERROR: No outcome files found matching {args.suffix} in {input_dir}")
        sys.exit(1)

    print(f"Found {len(outcome_files)} outcome files")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    extractor = FeatureExtractor()
    success_count = 0
    error_count = 0

    for outcome_path in outcome_files:
        video_name = outcome_path.stem.replace('_pellet_outcomes', '')

        print(f"\nProcessing: {video_name}")

        # Find corresponding DLC and reaches files
        dlc_files = list(input_dir.glob(f"{video_name}*DLC*.h5"))
        reaches_files = [
            input_dir / f"{video_name}_reaches.json"
        ]

        if not dlc_files:
            print(f"  [WARN] No DLC file found")
            error_count += 1
            continue

        dlc_path = dlc_files[0]

        reaches_path = None
        for rp in reaches_files:
            if rp.exists():
                reaches_path = rp
                break

        if not reaches_path:
            print(f"  [WARN] No reaches file found")
            error_count += 1
            continue

        # Output path
        output_path = output_dir / f"{video_name}_features.json"

        if output_path.exists() and not args.overwrite:
            print(f"  [SKIP] Output exists (use --overwrite)")
            continue

        try:
            # Extract features
            results = extractor.extract(dlc_path, reaches_path, outcome_path)

            # Save results
            with open(output_path, 'w') as f:
                json.dump(results.to_dict(), f, indent=2)

            # Sync to central database
            try:
                from mousereach.sync.database import sync_file_to_database
                sync_file_to_database(output_path)
            except Exception:
                pass  # Don't fail feature extraction if database sync fails

            print(f"  [OK] {results.summary.get('total_reaches', 0)} reaches, "
                  f"{results.summary.get('causal_reaches', 0)} causal")
            success_count += 1

        except Exception as e:
            print(f"  [ERROR] {e}")
            error_count += 1
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"COMPLETE: {success_count} successful, {error_count} errors")


def main_triage():
    """Triage feature extraction results."""
    parser = argparse.ArgumentParser(
        description="Triage feature extraction results"
    )
    parser.add_argument(
        '-i', '--input',
        type=Path,
        required=True,
        help='Directory with feature JSON files'
    )

    args = parser.parse_args()

    feature_files = sorted(args.input.glob('*_features.json'))

    if not feature_files:
        print(f"No feature files found in {args.input}")
        sys.exit(1)

    print("=" * 80)
    print("FEATURE EXTRACTION TRIAGE")
    print("=" * 80)

    for fpath in feature_files:
        with open(fpath) as f:
            data = json.load(f)

        video_name = data['video_name']
        summary = data.get('summary', {})

        total_reaches = summary.get('total_reaches', 0)
        causal_reaches = summary.get('causal_reaches', 0)
        mean_extent = summary.get('mean_extent_mm', 0)

        print(f"\n{video_name}:")
        print(f"  Reaches: {total_reaches} total, {causal_reaches} causal")
        print(f"  Mean extent: {mean_extent:.2f} mm" if mean_extent else "  Mean extent: N/A")
        print(f"  Outcomes: {summary.get('outcome_counts', {})}")


def main_review():
    """Review individual feature extraction file."""
    parser = argparse.ArgumentParser(
        description="Review feature extraction for a video"
    )
    parser.add_argument(
        'feature_file',
        type=Path,
        help='Path to *_features.json file'
    )

    args = parser.parse_args()

    if not args.feature_file.exists():
        print(f"ERROR: File not found: {args.feature_file}")
        sys.exit(1)

    with open(args.feature_file) as f:
        data = json.load(f)

    print("=" * 80)
    print(f"VIDEO: {data['video_name']}")
    print("=" * 80)
    print(f"Extractor version: {data['extractor_version']}")
    print(f"Total frames: {data['total_frames']}")
    print(f"Segments: {data['n_segments']}")
    print(f"\nSUMMARY:")
    for key, value in data.get('summary', {}).items():
        print(f"  {key}: {value}")

    print(f"\n{'='*80}")
    print("CAUSAL REACHES (reaches that caused outcomes):")
    print(f"{'='*80}")

    for seg in data['segments']:
        causal_reaches = [r for r in seg['reaches'] if r['causal_reach']]
        if causal_reaches:
            print(f"\nSegment {seg['segment_num']}: {seg['outcome']}")
            for reach in causal_reaches:
                print(f"  Reach {reach['reach_id']}:")
                print(f"    Extent: {reach.get('max_extent_mm', 'N/A')} mm")
                print(f"    Duration: {reach['duration_frames']} frames")
                print(f"    Peak velocity: {reach.get('peak_velocity_px_per_frame', 'N/A'):.2f} px/frame" 
                      if reach.get('peak_velocity_px_per_frame') else "    Peak velocity: N/A")
                print(f"    Straightness: {reach.get('trajectory_straightness', 'N/A'):.3f}"
                      if reach.get('trajectory_straightness') else "    Straightness: N/A")


if __name__ == "__main__":
    main_batch()
