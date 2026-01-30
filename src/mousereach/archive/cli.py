#!/usr/bin/env python3
"""
cli.py - Command line interface for mousereach-archive.

Archive fully validated videos from Processing/ to NAS archive.
Videos can only be archived when ALL validation statuses are "validated".

Usage:
    mousereach-archive                    # Archive all ready videos
    mousereach-archive video_id           # Archive specific video
    mousereach-archive --dry-run          # Show what would be archived
    mousereach-archive --list             # List videos ready for archive
"""

import argparse
import sys

from .core import archive_video, archive_all, get_archivable_videos, check_archive_ready


def main():
    """CLI entry point for mousereach-archive."""
    parser = argparse.ArgumentParser(
        description="Archive fully validated videos to NAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    mousereach-archive                    # Archive all ready videos
    mousereach-archive 20250704_CNT0101_P1   # Archive specific video
    mousereach-archive --dry-run          # Preview what would be archived
    mousereach-archive --list             # List ready videos

Requirements:
    A video can only be archived when ALL stages are validated:
    - Segmentation: validated
    - Reach detection: validated
    - Outcome detection: validated
"""
    )

    parser.add_argument(
        "video_id",
        nargs="?",
        help="Video ID to archive (default: all ready videos)"
    )

    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without moving files"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List videos ready for archive"
    )

    parser.add_argument(
        "--status", "-s",
        metavar="VIDEO_ID",
        help="Check archive status for a specific video"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        videos = get_archivable_videos()
        if not videos:
            print("No videos ready for archive.")
            print("Videos need ALL validation statuses to be 'validated'.")
            return

        print(f"Videos ready for archive ({len(videos)}):")
        for vid in sorted(videos):
            print(f"  {vid}")
        return

    # Status mode
    if args.status:
        is_ready, status = check_archive_ready(args.status)
        print(f"Archive status for: {args.status}")
        print("-" * 40)
        for step, val in status.items():
            marker = "OK" if val == "validated" else "PENDING"
            print(f"  {step:10s}: {val:15s} [{marker}]")
        print("-" * 40)
        if is_ready:
            print("READY for archive")
        else:
            not_done = [k for k, v in status.items() if v != "validated"]
            print(f"NOT READY - need: {', '.join(not_done)}")
        return

    verbose = not args.quiet

    print("=" * 60)
    print("MouseReach Archive Tool")
    print("=" * 60)

    # Single video mode
    if args.video_id:
        result = archive_video(
            args.video_id,
            dry_run=args.dry_run,
            verbose=verbose
        )

        if not result["success"] and not args.dry_run:
            sys.exit(1)

    # All ready videos mode
    else:
        results = archive_all(
            dry_run=args.dry_run,
            verbose=verbose
        )

        if results["failed"] > 0 and not args.dry_run:
            sys.exit(1)


if __name__ == "__main__":
    main()
