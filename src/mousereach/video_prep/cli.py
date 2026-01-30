#!/usr/bin/env python3
"""
CLI entry points for MouseReach Video Prep

Commands:
    mousereach-crop      - Crop multi-animal collages
    mousereach-convert   - Convert MKV to MP4
    mousereach-prep      - Full workflow (crop + copy to queue)
"""

import argparse
from pathlib import Path


def main_crop():
    """Crop collages entry point"""
    from mousereach.video_prep.core import crop_all, crop_collage, copy_to_dlc_queue
    from mousereach.config import Paths

    parser = argparse.ArgumentParser(description="Crop 8-camera collages into single-animal videos")
    parser.add_argument('-i', '--input', type=Path,
                        help=f"Input file or directory (default: {Paths.MULTI_ANIMAL_SOURCE})")
    parser.add_argument('-o', '--output', type=Path,
                        help=f"Output directory (default: {Paths.SINGLE_ANIMAL_OUTPUT})")
    parser.add_argument('--queue', action='store_true',
                        help="Also copy outputs to DLC_Queue")
    parser.add_argument('-q', '--quiet', action='store_true')

    args = parser.parse_args()

    # Check if input is a file or directory
    if args.input and args.input.is_file():
        # Single file mode
        output_dir = args.output or args.input.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing single file: {args.input.name}")
        results_list = crop_collage(args.input, output_dir, verbose=not args.quiet)

        success = sum(1 for r in results_list if r.get('status') == 'success')
        skipped = sum(1 for r in results_list if r.get('status') == 'skipped')
        failed = sum(1 for r in results_list if r.get('status') == 'failed')

        results = {
            'total_collages': 1,
            'success': success,
            'skipped': skipped,
            'failed': failed
        }
    else:
        # Directory mode
        results = crop_all(args.input, args.output, verbose=not args.quiet)

    print(f"\nCropping complete:")
    print(f"  Collages processed: {results['total_collages']}")
    print(f"  Videos created: {results['success']}")
    print(f"  Skipped (blank): {results['skipped']}")
    print(f"  Failed: {results['failed']}")

    if args.queue and results['success'] > 0:
        print("\nCopying to DLC queue...")
        output_dir = args.output or Paths.SINGLE_ANIMAL_OUTPUT
        copied = copy_to_dlc_queue(output_dir, verbose=not args.quiet)
        print(f"Copied {copied} files to DLC_Queue")


def main_convert():
    """Convert MKV to MP4"""
    from mousereach.video_prep.core import convert_mkv_to_mp4
    
    parser = argparse.ArgumentParser(description="Convert MKV to MP4")
    parser.add_argument('input', type=Path, nargs='+', help="Input MKV file(s)")
    parser.add_argument('-o', '--output-dir', type=Path)
    
    args = parser.parse_args()
    
    for inp in args.input:
        out = args.output_dir / inp.with_suffix('.mp4').name if args.output_dir else None
        result = convert_mkv_to_mp4(inp, out)
        print(f"Converted: {inp.name} -> {result.name}")


def main_prep():
    """Full video prep workflow"""
    from mousereach.video_prep.core import crop_all, copy_to_dlc_queue, archive_collages
    
    parser = argparse.ArgumentParser(description="Full video prep workflow")
    parser.add_argument('-i', '--input', type=Path,
                        help="Input Multi-Animal directory")
    parser.add_argument('--no-queue', action='store_true',
                        help="Don't copy to DLC queue")
    parser.add_argument('--archive', action='store_true',
                        help="Archive original collages after cropping")
    parser.add_argument('-q', '--quiet', action='store_true')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MouseReach Video Prep Workflow")
    print("=" * 60)
    
    # Step 1: Crop
    print("\n[1/3] Cropping collages...")
    results = crop_all(args.input, verbose=not args.quiet)
    
    print(f"\n  Collages: {results['total_collages']}")
    print(f"  Created:  {results['success']}")
    print(f"  Skipped:  {results['skipped']}")
    
    # Step 2: Copy to queue
    if not args.no_queue and results['success'] > 0:
        print("\n[2/3] Copying to DLC queue...")
        copied = copy_to_dlc_queue(verbose=not args.quiet)
        print(f"  Copied: {copied}")
    else:
        print("\n[2/3] Skipping queue copy")
    
    # Step 3: Archive
    if args.archive:
        print("\n[3/3] Archiving collages...")
        archived = archive_collages(args.input, verbose=not args.quiet)
        print(f"  Archived: {archived}")
    else:
        print("\n[3/3] Skipping archive (use --archive to enable)")
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main_prep()
