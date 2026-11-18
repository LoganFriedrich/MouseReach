#!/usr/bin/env python3
"""
CLI entry points for MouseReach Video Prep

Commands:
    mousereach-crop      - Crop multi-animal collages
    mousereach-convert   - Convert MKV to MP4
    mousereach-prep      - Full workflow (crop + hand the cut videos to the pose step)

Handing cut videos to the pose step (--queue, and mousereach-prep's step 2)
puts each video THIS run cut into the shared singles folder, where a
graphics-card PC's watcher claims and poses it -- one machine per video. It
never copies anything straight into a DLC queue (video_prep.core.cropper.
hand_to_pose says why). --archive moves only collages whose single-animal
videos are all done (the watcher's own retirement rule).
"""

import argparse
from pathlib import Path


def main_crop():
    """Crop collages entry point"""
    from mousereach.video_prep.core import crop_all, crop_collage, hand_to_pose
    from mousereach.config import Paths

    parser = argparse.ArgumentParser(description="Crop 8-camera collages into single-animal videos")
    parser.add_argument('-i', '--input', type=Path,
                        help=f"Input file or directory (default: {Paths.MULTI_ANIMAL_SOURCE})")
    parser.add_argument('-o', '--output', type=Path,
                        help=f"Output directory (default: {Paths.SINGLE_ANIMAL_OUTPUT})")
    parser.add_argument('--queue', action='store_true',
                        help="Also hand the cut videos to the pose step: each video this "
                             "run cut goes into the shared singles folder, where a "
                             "graphics-card PC's watcher claims and poses it (never "
                             "copied straight into a DLC queue)")
    parser.add_argument('-q', '--quiet', action='store_true')

    args = parser.parse_args()

    # Check if input is a file or directory
    if args.input and args.input.is_file():
        # Single file mode
        output_dir = args.output or args.input.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing single file: {args.input.name}")
        # A failed recording is never cut (watcher/recording_issue.py).
        from mousereach.watcher.recording_issue import refuse_if_issue, RecordingIssueError
        try:
            refuse_if_issue(args.input)
        except RecordingIssueError as e:
            print(f"[FAIL] {e}")
            return 1
        results_list = crop_collage(args.input, output_dir, verbose=not args.quiet)

        success = sum(1 for r in results_list if r.get('status') == 'success')
        skipped = sum(1 for r in results_list if r.get('status') == 'skipped')
        failed = sum(1 for r in results_list if r.get('status') == 'failed')

        results = {
            'total_collages': 1,
            'success': success,
            'skipped': skipped,
            'failed': failed,
            'results': results_list,
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
        print("\nHanding the cut videos to the pose step (shared singles folder)...")
        handed = hand_to_pose(_made_by_this_run(results), verbose=not args.quiet)
        _print_handed(handed)


def _made_by_this_run(results: dict) -> list:
    """The videos this run cut (each successful crop result's output_path)."""
    return [r.get('output_path') for r in results.get('results', [])
            if r.get('status') == 'success' and r.get('output_path')]


def _print_handed(handed: dict) -> None:
    print(f"  Put in the shared singles folder: {handed['handed']}")
    print(f"  Already waiting there:            {handed['waiting']}")
    if handed['not_handed']:
        print(f"  [!] Not handed on:                {handed['not_handed']} "
              f"(a video of that name is already there or held for pose)")
    print("  A graphics-card PC's watcher claims and poses each one (press "
          "\"Run the pipeline now\" there when no watcher is running).")


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
    from mousereach.video_prep.core import crop_all, hand_to_pose, archive_collages
    
    parser = argparse.ArgumentParser(description="Full video prep workflow")
    parser.add_argument('-i', '--input', type=Path,
                        help="Input Multi-Animal directory")
    parser.add_argument('--no-queue', action='store_true',
                        help="Don't hand the cut videos to the pose step")
    parser.add_argument('--archive', action='store_true',
                        help="Move collages whose single-animal videos are ALL done "
                             "(filed, current, no review pending) to the archive; "
                             "every other collage stays")
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
    
    # Step 2: hand the cut videos to the pose step (the watcher claims them)
    if not args.no_queue and results['success'] > 0:
        print("\n[2/3] Handing the cut videos to the pose step...")
        _print_handed(hand_to_pose(_made_by_this_run(results), verbose=not args.quiet))
    else:
        print("\n[2/3] Not handing anything to the pose step")
    
    # Step 3: Archive
    if args.archive:
        print("\n[3/3] Archiving collages whose videos are all done...")
        archived = archive_collages(args.input, verbose=not args.quiet)
        print(f"  Archived: {archived}")
    else:
        print("\n[3/3] Skipping archive (use --archive to enable)")
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main_prep()
