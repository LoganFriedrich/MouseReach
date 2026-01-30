#!/usr/bin/env python3
"""
migrate_to_processing.py - One-time migration to v2.3+ single-folder architecture.

This script migrates files from the old multi-folder pipeline structure to the
new single Processing/ folder architecture where validation status is stored
in JSON metadata rather than determined by folder location.

Usage:
    python -m mousereach.migrate_to_processing --dry-run  # Preview what will happen
    python -m mousereach.migrate_to_processing            # Execute migration
    python -m mousereach.migrate_to_processing --status   # Show current folder status

After running:
    mousereach-index-rebuild   # Rebuild the pipeline index
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Mapping from folder name to validation_status
FOLDER_TO_STATUS = {
    "Seg_AutoReview": "auto_approved",
    "Seg_NeedsReview": "needs_review",
    "Seg_Validated": "validated",
    "Reach_NeedsReview": "needs_review",
    "Reach_Validated": "validated",
    "Outcome_NeedsReview": "needs_review",
    "Outcome_Validated": "validated",
    "Score_NeedsReview": "needs_review",
    "Score_Validated": "validated",
    "DLC_Complete": "needs_review",  # DLC output needs segmentation
}

# Folders that should be migrated (old architecture)
LEGACY_FOLDERS = [
    "DLC_Complete",
    "Seg_AutoReview",
    "Seg_NeedsReview",
    "Seg_Validated",
    "Reach_NeedsReview",
    "Reach_Validated",
    "Outcome_NeedsReview",
    "Outcome_Validated",
    "Score_NeedsReview",
    "Score_Validated",
]

# Folders to keep as-is
KEEP_FOLDERS = ["DLC_Queue", "Failed"]

# Folder to archive (legacy test data)
ARCHIVE_FOLDERS = ["Pipeline_0_0"]


def get_video_id(filename: str) -> str:
    """Extract video_id from filename (strip DLC suffix and extensions)."""
    name = Path(filename).stem

    # Strip DLC suffix (e.g., "20250704_CNT0101_P1DLC_resnet50_..." -> "20250704_CNT0101_P1")
    if "DLC_" in name:
        name = name.split("DLC_")[0]

    # Strip common suffixes
    for suffix in ["_segments", "_reaches", "_pellet_outcomes", "_grasp_features",
                   "_seg_ground_truth", "_reach_ground_truth", "_outcome_ground_truth",
                   "_outcomes_ground_truth", "_seg_validation", "_reaches_validation",
                   "_outcomes_validation"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    return name


def scan_folder(folder: Path) -> Dict[str, List[Path]]:
    """Scan folder and group files by video_id."""
    files_by_video = defaultdict(list)

    if not folder.exists():
        return files_by_video

    for f in folder.iterdir():
        if f.is_file():
            video_id = get_video_id(f.name)
            files_by_video[video_id].append(f)

    return files_by_video


def update_json_validation_status(json_path: Path, status: str, dry_run: bool = False) -> bool:
    """
    Update validation_status in a JSON file.

    Returns True if file was modified, False if skipped (already has status or not JSON).
    """
    if not json_path.suffix == ".json":
        return False

    # Skip ground truth files
    if "ground_truth" in json_path.name:
        return False

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False

    # Skip if already has validation_status
    if "validation_status" in data:
        return False

    # Add validation_status and timestamp
    data["validation_status"] = status
    data["validation_timestamp"] = datetime.now().isoformat()
    data["migrated_from_folder"] = json_path.parent.name

    if not dry_run:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    return True


def show_status(processing_root: Path):
    """Show current folder status."""
    print("=" * 60)
    print("MouseReach Pipeline Folder Status")
    print(f"Root: {processing_root}")
    print("=" * 60)

    total_files = 0

    for folder_name in sorted(processing_root.iterdir()):
        if folder_name.is_dir():
            files = list(folder_name.glob("*"))
            file_count = len([f for f in files if f.is_file()])
            total_files += file_count

            status = ""
            if folder_name.name in LEGACY_FOLDERS:
                status = " [LEGACY - needs migration]"
            elif folder_name.name in KEEP_FOLDERS:
                status = " [KEEP]"
            elif folder_name.name in ARCHIVE_FOLDERS:
                status = " [ARCHIVE]"
            elif folder_name.name == "Processing":
                status = " [v2.3+ target]"
            elif folder_name.name == "Archive":
                status = " [Archive storage]"

            print(f"  {folder_name.name:25s} {file_count:5d} files{status}")

    print("-" * 60)
    print(f"Total: {total_files} files")

    # Check if Processing folder exists
    processing = processing_root / "Processing"
    if not processing.exists():
        print("\n[!] Processing/ folder does NOT exist - migration needed!")
    else:
        print("\n[OK] Processing/ folder exists")


def migrate(processing_root: Path, dry_run: bool = False, verbose: bool = True):
    """
    Execute the migration.

    Args:
        processing_root: Root of pipeline folders (your PROCESSING_ROOT)
        dry_run: If True, only show what would be done without making changes
        verbose: Print progress
    """
    if verbose:
        mode = "DRY RUN" if dry_run else "EXECUTING"
        print("=" * 60)
        print(f"MouseReach Pipeline Migration to v2.3+ [{mode}]")
        print(f"Root: {processing_root}")
        print("=" * 60)

    # Create target folders
    processing_dir = processing_root / "Processing"
    archive_dir = processing_root / "Archive"

    if not dry_run:
        processing_dir.mkdir(exist_ok=True)
        archive_dir.mkdir(exist_ok=True)

    if verbose:
        print(f"\n1. Creating folders:")
        print(f"   Processing/ {'(exists)' if processing_dir.exists() else '(will create)'}")
        print(f"   Archive/    {'(exists)' if archive_dir.exists() else '(will create)'}")

    # Archive legacy test folders
    if verbose:
        print(f"\n2. Archiving legacy folders:")

    for folder_name in ARCHIVE_FOLDERS:
        folder = processing_root / folder_name
        if folder.exists():
            dest = archive_dir / folder_name
            file_count = len(list(folder.glob("*")))
            if verbose:
                print(f"   {folder_name}/ ({file_count} files) -> Archive/{folder_name}/")
            if not dry_run:
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(folder), str(dest))

    # Scan all legacy folders and group by video_id
    if verbose:
        print(f"\n3. Scanning legacy folders...")

    all_videos: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))

    for folder_name in LEGACY_FOLDERS:
        folder = processing_root / folder_name
        if not folder.exists():
            continue

        files_by_video = scan_folder(folder)
        for video_id, files in files_by_video.items():
            for f in files:
                all_videos[video_id][folder_name].append(f)

    if verbose:
        print(f"   Found {len(all_videos)} unique videos across legacy folders")

    # Migrate each video
    if verbose:
        print(f"\n4. Migrating files to Processing/:")

    files_moved = 0
    jsons_updated = 0

    for video_id in sorted(all_videos.keys()):
        folders_data = all_videos[video_id]

        # Determine the "most validated" status for this video
        # Priority: validated > auto_approved > needs_review
        best_status = "needs_review"
        for folder_name in folders_data.keys():
            folder_status = FOLDER_TO_STATUS.get(folder_name, "needs_review")
            if folder_status == "validated":
                best_status = "validated"
                break
            elif folder_status == "auto_approved" and best_status != "validated":
                best_status = "auto_approved"

        # Move all files for this video
        for folder_name, files in folders_data.items():
            folder_status = FOLDER_TO_STATUS.get(folder_name, "needs_review")

            for src_file in files:
                dest_file = processing_dir / src_file.name

                if verbose:
                    print(f"   {src_file.name}")
                    print(f"      {folder_name}/ -> Processing/")

                if not dry_run:
                    # Move file
                    if dest_file.exists():
                        dest_file.unlink()  # Remove existing
                    shutil.move(str(src_file), str(dest_file))

                files_moved += 1

                # Update JSON validation_status
                if src_file.suffix == ".json" and "ground_truth" not in src_file.name:
                    json_path = dest_file if not dry_run else src_file
                    if update_json_validation_status(json_path, folder_status, dry_run):
                        if verbose:
                            print(f"      -> validation_status: {folder_status}")
                        jsons_updated += 1

    # Archive empty legacy folders
    if verbose:
        print(f"\n5. Archiving empty legacy folders:")

    folders_archived = 0
    for folder_name in LEGACY_FOLDERS:
        folder = processing_root / folder_name
        if folder.exists():
            remaining = list(folder.glob("*"))
            if len(remaining) == 0:
                dest = archive_dir / folder_name
                if verbose:
                    print(f"   {folder_name}/ -> Archive/{folder_name}/")
                if not dry_run:
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.move(str(folder), str(dest))
                folders_archived += 1
            else:
                if verbose:
                    print(f"   {folder_name}/ - {len(remaining)} files remaining (not archived)")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Migration Summary:")
        print(f"   Files moved:        {files_moved}")
        print(f"   JSONs updated:      {jsons_updated}")
        print(f"   Folders archived:   {folders_archived}")
        print("=" * 60)

        if dry_run:
            print("\nThis was a DRY RUN. No changes were made.")
            print("Run without --dry-run to execute the migration.")
        else:
            print("\n[OK] Migration complete!")
            print("\nNext step: Run 'mousereach-index-rebuild' to update the pipeline index.")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate MouseReach pipeline to v2.3+ single-folder architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m mousereach.migrate_to_processing --dry-run  # Preview changes
    python -m mousereach.migrate_to_processing            # Execute migration
    python -m mousereach.migrate_to_processing --status   # Show folder status

After migration:
    mousereach-index-rebuild   # Rebuild the pipeline index
"""
    )

    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without making changes"
    )

    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show current folder status"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Override processing root (default: from config)"
    )

    args = parser.parse_args()

    # Get processing root
    if args.root:
        processing_root = args.root
    else:
        try:
            from mousereach.config import Paths
            processing_root = Paths.PROCESSING_ROOT
        except ImportError:
            print("Error: Could not import mousereach.config. Use --root to specify path.")
            sys.exit(1)

    if not processing_root.exists():
        print(f"Error: Processing root does not exist: {processing_root}")
        sys.exit(1)

    if args.status:
        show_status(processing_root)
    else:
        migrate(processing_root, dry_run=args.dry_run, verbose=not args.quiet)


if __name__ == "__main__":
    main()
