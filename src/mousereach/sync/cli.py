#!/usr/bin/env python3
"""
CLI commands for MouseReach Database Sync

Commands:
    mousereach-sync         One-time sync of all features files to reach_data table
    mousereach-sync-watch   Start continuous file watcher
    mousereach-sync-status  Show sync status
"""

import argparse
import sys
from pathlib import Path


def main_sync():
    """
    One-time sync of all _features.json files to reach_data table.

    Usage: mousereach-sync [--force] [--dry-run]
    """
    parser = argparse.ArgumentParser(
        description="Sync MouseReach features to reach_data table in connectome database"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force sync all files, even if unchanged"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be synced without actually syncing"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    args = parser.parse_args()

    from .database import DatabaseSyncer

    print("\nMouseReach Database Sync")
    print("=" * 40)

    syncer = DatabaseSyncer(dry_run=args.dry_run)

    if args.dry_run:
        print("[DRY RUN - no changes will be made]\n")

    # Show status first
    status = syncer.get_status()

    print(f"Database:   {status['database_path']}")
    print(f"Processing: {status['processing_path'] or 'NOT CONFIGURED'}")

    if not status['database_ok']:
        print(f"\nError: {status['database_message']}")
        sys.exit(1)

    print(f"\nDatabase status: OK")
    print(f"  Reaches in DB:   {status.get('total_reaches', 0)}")
    print(f"  Videos in DB:    {status.get('total_videos', 0)}")
    print(f"  Subjects in DB:  {status.get('total_subjects', 0)}")
    print(f"  Causal reaches:  {status.get('causal_reaches', 0)}")
    print(f"\nFeatures files:  {status['syncable_files']}")
    print(f"Pending sync:    {status['pending_files']}")

    if status['pending_files'] == 0 and not args.force:
        print("\nNo files need syncing. Use --force to re-sync all files.")
        return

    # Do the sync
    print(f"\nSyncing{'...' if not args.verbose else ':'}")
    result = syncer.sync_all(force=args.force)

    print(f"\nResults:")
    print(f"  Files synced:    {result.synced}")
    print(f"  Reaches inserted: {result.reaches_inserted}")
    print(f"  Files skipped:   {result.skipped}")
    print(f"  Errors:          {result.errors}")

    if result.errors > 0 and args.verbose:
        print("\nErrors:")
        for msg in result.error_messages:
            print(f"  - {msg}")

    sys.exit(0 if result.errors == 0 else 1)


def main_watch():
    """
    Start continuous file watcher for automatic sync.

    Usage: mousereach-sync-watch [--debounce SECONDS]
    """
    parser = argparse.ArgumentParser(
        description="Watch Processing folder and sync new features files to database"
    )
    parser.add_argument(
        "--debounce", "-d",
        type=float,
        default=2.0,
        help="Seconds to wait after file change before syncing (default: 2.0)"
    )
    args = parser.parse_args()

    try:
        from .watcher import start_watcher
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nThe watcher requires the 'watchdog' package.")
        print("Install with: pip install watchdog")
        sys.exit(1)

    # Initial sync
    from .database import DatabaseSyncer
    syncer = DatabaseSyncer()

    print("\nMouseReach Database Sync Watcher")
    print("=" * 40)

    status = syncer.get_status()
    if not status['database_ok']:
        print(f"\nError: {status['database_message']}")
        sys.exit(1)

    # Do initial sync
    if status['pending_files'] > 0:
        print(f"\nInitial sync of {status['pending_files']} pending files...")
        result = syncer.sync_all()
        print(f"Synced {result.synced} files ({result.reaches_inserted} reaches, {result.errors} errors)")

    # Start watcher (blocking)
    try:
        start_watcher(debounce_seconds=args.debounce, blocking=True)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)


def main_status():
    """
    Show sync status.

    Usage: mousereach-sync-status
    """
    parser = argparse.ArgumentParser(
        description="Show MouseReach database sync status"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    args = parser.parse_args()

    from .database import DatabaseSyncer

    syncer = DatabaseSyncer()
    status = syncer.get_status()

    if args.json:
        import json
        print(json.dumps(status, indent=2))
        return

    print("\nMouseReach Database Sync Status")
    print("=" * 40)

    print(f"\nDatabase:   {status['database_path']}")
    print(f"Processing: {status['processing_path'] or 'NOT CONFIGURED'}")

    if status['database_ok']:
        print(f"\nDatabase Status: OK")
        print(f"  Total reaches:   {status.get('total_reaches', 0)}")
        print(f"  Total videos:    {status.get('total_videos', 0)}")
        print(f"  Total subjects:  {status.get('total_subjects', 0)}")
        print(f"  Causal reaches:  {status.get('causal_reaches', 0)}")
    else:
        print(f"\nDatabase Status: ERROR")
        print(f"  {status['database_message']}")

    print(f"\nFeatures Files:")
    print(f"  Syncable:       {status['syncable_files']}")
    print(f"  Already synced: {status['synced_files']}")
    print(f"  Pending sync:   {status['pending_files']}")

    if status['pending_files'] > 0:
        print(f"\nRun 'mousereach-sync' to sync pending files.")


if __name__ == "__main__":
    main_sync()
