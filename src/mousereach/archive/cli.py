#!/usr/bin/env python3
"""
cli.py - Command line interface for mousereach-archive.

File finished videos from Processing/ into the Analyzed tree, by hand.

On a machine that runs the watcher, the watcher files every video itself as
soon as it has passed every check, so this command is rarely needed there.
It files a video only when (1) the watcher is not running on this machine
(and that can be told: health.watcher_presence),
(2) this machine's watcher database, if there is one, has the video as
'processed' (passed every check), and (3) the video's features file is there
and newer than its stage outputs. See docs/CLI_REFERENCE.md.

Usage:
    mousereach-archive                    # Archive all ready videos
    mousereach-archive video_id           # Archive specific video
    mousereach-archive --dry-run          # Show what would be archived
    mousereach-archive --list             # List videos ready for archive
"""

import argparse
import sys

from .core import (archive_video, archive_all, get_archivable_videos,
                   check_archive_ready, record_archived, command_filing_refusal,
                   _row_known_here as _row_known)


def watcher_daemon_running() -> bool:
    """True when this machine's watcher is running -- or when that cannot be
    told. WHY refuse then: the watcher files videos itself, and two filers
    moving the same video's files at once can leave it split between
    Processing and the filed folder. "Cannot tell" is treated as running
    because the cost of waiting is nothing and the cost of racing is a split
    video. health.watcher_presence decides; a program of another Windows
    login whose command line cannot be read is "cannot tell" (it used to be
    skipped, so a watcher started by another login looked like none)."""
    return _watcher_running_why()[0]


def _watcher_running_why():
    """``(refuse, why)`` for watcher_daemon_running and its message."""
    try:
        from mousereach.watcher.health import watcher_presence
        running, why = watcher_presence()
    except Exception as e:
        return True, f"whether the watcher runs could not be checked ({e})"
    return (running is not False), why


def main():
    """CLI entry point for mousereach-archive."""
    parser = argparse.ArgumentParser(
        description="File by hand a finished video into its Analyzed cohort folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    mousereach-archive                    # Archive all ready videos
    mousereach-archive 20250704_CNT0101_P1   # Archive specific video
    mousereach-archive --dry-run          # Preview what would be archived
    mousereach-archive --list             # List ready videos

On a machine that runs the watcher you normally never need this command:
the watcher files each video by itself as soon as it has passed every check
(quality triage, the hand-score comparison, review, kinematics). This
command files by hand, and it refuses -- moving nothing, exit code 1 -- when:
    - the watcher is running on this machine, or that cannot be told
      (two programs filing one video at once can split it between two
      folders). The machine-wide one-watcher lock is checked first, so a
      watcher started by another Windows login counts too; a Python program
      of another login whose command line cannot be read counts as "cannot
      tell" unless that lock is known to be free. Stop the watcher first, or
      leave the filing to it;
    - this login's settings cannot name the watcher database (they cannot
      be read, or a processing server's settings have no watcher.db_path);
    - this machine's watcher database is there but cannot be read, or it
      knows the video and does not have it as 'processed'. The message
      names the state (e.g. 'processing' while
      it waits for its hand scores, 'triage', 'failed', 'crystallized');
      deal with it through the watcher and the review tools instead;
    - <video>_features.json is missing, or older than the video's segments,
      reaches or outcomes file (an earlier run's kinematics);
    - a segmentation, reach or outcome file lacks an approved verdict.
A machine whose named watcher database does not exist (the tool used on its
own) is judged on the files alone. After filing, 'archived' is recorded in this machine's
watcher database; if the database knows the video and that cannot be
recorded, the command says so and exits 1. --list shows only videos that
pass these checks; --status VIDEO_ID names the reason one would be refused.
Filing is one step: every file is first copied into the cohort folder
under a temporary name (.<name>.<random>.tmp), and only when all are there
are they renamed into place, the features file last. An attempt that fails
leaves the cohort folder exactly as it was and every file in Processing;
a re-run's earlier results are kept in Analyzed/Archive.
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
        videos = [v for v in get_archivable_videos()
                   if not command_filing_refusal(v)]
        if not videos:
            print("No videos ready for archive.")
            print("Videos need ALL validation statuses to be 'validated', a")
            print("'processed' watcher row (if this machine has a watcher database)")
            print("and an up-to-date features file.")
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
        why = command_filing_refusal(args.status)
        if is_ready and not why:
            print("READY for archive")
        elif not is_ready:
            not_done = [k for k, v in status.items() if v != "validated"]
            print(f"NOT READY - need: {', '.join(not_done)}")
        if why:
            print(f"NOT READY - {why}")
        return

    verbose = not args.quiet

    if not args.dry_run:
        refuse, why = _watcher_running_why()
        if refuse:
            print(f"The watcher is running on this machine, or that cannot be "
                  f"told ({why}). It files finished videos itself; this command "
                  f"does not file beside it. Stop the watcher first, or leave the "
                  f"filing to it. Nothing was moved.")
            sys.exit(1)

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

        # Keep this node's record in step with the disk -- see record_archived.
        # A row that exists but could not be recorded is a failure the person
        # must see: the files are filed and the database says otherwise.
        if result["success"] and not args.dry_run:
            known = _row_known(args.video_id)
            if not record_archived(args.video_id) and known:
                print(f"{args.video_id}: filed, but the watcher database could "
                      f"not record it. Run mousereach-watch-status.")
                sys.exit(1)

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
