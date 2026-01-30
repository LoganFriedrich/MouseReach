#!/usr/bin/env python3
"""
Pipeline Scanner - Full rebuild logic for pipeline index.

This module handles scanning pipeline folders and building/rebuilding the index.
Used by:
- mousereach-index-rebuild: Full rebuild
- mousereach-index-refresh: Refresh specific folder
- PipelineIndex.refresh_folder(): Incremental updates
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .index import PipelineIndex


def scan_folder(index: "PipelineIndex", folder_path: Path, stage: str,
                progress_callback: Callable = None) -> int:
    """Scan a single pipeline folder and update the index.

    Args:
        index: PipelineIndex instance to update
        folder_path: Path to the folder to scan
        stage: Pipeline stage name (e.g., "Processing")
        progress_callback: Optional callback(current, total, message)

    Returns:
        Number of videos found/updated
    """
    from mousereach.config import get_video_id, FilePatterns, parse_tray_type

    if not folder_path.exists():
        return 0

    # Group files by video_id
    video_files: Dict[str, List[str]] = {}

    # List all files in folder (non-recursive for speed)
    try:
        all_files = list(folder_path.iterdir())
    except PermissionError:
        print(f"[Scanner] Permission denied: {folder_path}")
        return 0

    # STEP 1: Find valid video_ids from DLC .h5 files (canonical identifiers)
    # This prevents indexing random files like batch_outcomes_*, README, etc.
    valid_video_ids = set()
    for file_path in all_files:
        if file_path.is_file() and "DLC" in file_path.name and file_path.suffix == ".h5":
            video_id = get_video_id(file_path.name)
            if video_id:
                valid_video_ids.add(video_id)

    if not valid_video_ids:
        return 0

    # STEP 2: Group all files belonging to valid videos
    total = len(all_files)
    for i, file_path in enumerate(all_files):
        if progress_callback and i % 50 == 0:
            progress_callback(i, total, f"Scanning {stage}...")

        if not file_path.is_file():
            continue

        fname = file_path.name
        video_id = get_video_id(fname)

        # Only include files belonging to valid videos (identified by DLC file)
        if not video_id or video_id not in valid_video_ids:
            continue

        if video_id not in video_files:
            video_files[video_id] = []
        video_files[video_id].append(fname)

    # Update index for each video
    for video_id, files in video_files.items():
        # Extract metadata from JSON files if present
        metadata = _extract_metadata_from_files(folder_path, files)

        # Extract tray type from video_id
        tray_info = parse_tray_type(video_id)
        metadata["tray_type"] = tray_info.get("tray_type")
        metadata["tray_name"] = tray_info.get("tray_name", "Unknown")
        metadata["tray_supported"] = tray_info.get("is_supported", False)

        # Record in index
        index.record_video(
            video_id=video_id,
            stage=stage,
            files={stage: files},
            metadata=metadata
        )

    return len(video_files)


def _extract_metadata_from_files(folder_path: Path, files: List[str]) -> Dict:
    """Extract metadata from JSON files in a video's file list.

    New architecture (v2.0+):
        - validation_status stored in each JSON file
        - Falls back to confidence-based triage if not present

    Args:
        folder_path: Path to folder containing files
        files: List of filenames

    Returns:
        Extracted metadata dict
    """
    from mousereach.config import FilePatterns

    metadata = {}

    for fname in files:
        # Segments JSON
        if fname.endswith(FilePatterns.SEGMENTS_SUFFIX):
            try:
                with open(folder_path / fname, 'r') as f:
                    data = json.load(f)
                metadata["seg_boundaries"] = len(data.get("boundaries", []))
                metadata["seg_confidence"] = data.get("mean_confidence", 0)
                metadata["segmenter_version"] = data.get("version", "unknown")

                # Check for explicit validation_status (new architecture)
                if "validation_status" in data:
                    metadata["seg_validation"] = data["validation_status"]
                else:
                    # Fall back to confidence-based triage (legacy)
                    mean_conf = data.get("mean_confidence", 0)
                    cv = data.get("interval_cv", 1.0)
                    if mean_conf >= 0.85 and cv <= 0.10:
                        metadata["seg_validation"] = "auto_approved"
                    else:
                        metadata["seg_validation"] = "needs_review"

                # Extract validation_record if present
                if "validation_record" in data:
                    metadata["seg_validated_at"] = data["validation_record"].get("validated_at")
                    metadata["seg_changes_count"] = data["validation_record"].get("items_changed", 0)

            except (json.JSONDecodeError, OSError):
                pass

        # Reaches JSON
        elif fname.endswith(FilePatterns.REACHES_SUFFIX):
            try:
                with open(folder_path / fname, 'r') as f:
                    data = json.load(f)
                total_reaches = sum(
                    len(seg.get("reaches", []))
                    for seg in data.get("segments", [])
                )
                metadata["reach_count"] = total_reaches
                metadata["reach_version"] = data.get("version", "unknown")

                # Check for explicit validation_status (new architecture)
                if "validation_status" in data:
                    metadata["reach_validation"] = data["validation_status"]
                else:
                    # Confidence-based auto-approval for reaches (v2.4)
                    # Criteria: ALL reaches must have conf >= 0.85 AND no segments flagged
                    segments = data.get("segments", [])
                    all_reaches = []
                    any_flagged = False
                    for seg in segments:
                        all_reaches.extend(seg.get("reaches", []))
                        if seg.get("flagged_for_review", False):
                            any_flagged = True

                    if all_reaches:
                        low_conf = sum(1 for r in all_reaches if r.get("confidence", 0) < 0.85)
                        if low_conf == 0 and not any_flagged:
                            metadata["reach_validation"] = "auto_approved"
                        else:
                            metadata["reach_validation"] = "needs_review"
                    else:
                        # No reaches = nothing to review
                        metadata["reach_validation"] = "auto_approved"

                # Extract validation_record if present
                if "validation_record" in data:
                    metadata["reach_validated_at"] = data["validation_record"].get("validated_at")
                    metadata["reach_changes_count"] = data["validation_record"].get("items_changed", 0)

            except (json.JSONDecodeError, OSError):
                pass

        # Outcomes JSON
        elif fname.endswith(FilePatterns.OUTCOMES_SUFFIX):
            try:
                with open(folder_path / fname, 'r') as f:
                    data = json.load(f)
                outcomes = data.get("outcomes", [])
                metadata["outcome_count"] = len(outcomes)
                metadata["outcomes_version"] = data.get("version", "unknown")
                # Count by type
                outcome_counts = {}
                for outcome in outcomes:
                    otype = outcome.get("outcome", "unknown")
                    outcome_counts[otype] = outcome_counts.get(otype, 0) + 1
                metadata["outcome_breakdown"] = outcome_counts

                # Check for explicit validation_status (new architecture)
                if "validation_status" in data:
                    metadata["outcome_validation"] = data["validation_status"]
                else:
                    # Confidence-based auto-approval for outcomes (v2.4)
                    # Criteria: ALL segments must have conf >= 0.90 AND not flagged AND outcome != "retrieved"
                    segments = data.get("segments", [])
                    if segments:
                        auto_approve = True
                        for seg in segments:
                            conf = seg.get("confidence", 0)
                            flagged = seg.get("flagged_for_review", False)
                            outcome = seg.get("outcome", "").lower()

                            if conf < 0.90 or flagged or outcome == "retrieved":
                                auto_approve = False
                                break

                        metadata["outcome_validation"] = "auto_approved" if auto_approve else "needs_review"
                    else:
                        metadata["outcome_validation"] = "needs_review"

                # Extract validation_record if present
                if "validation_record" in data:
                    metadata["outcome_validated_at"] = data["validation_record"].get("validated_at")
                    metadata["outcome_changes_count"] = data["validation_record"].get("items_changed", 0)

            except (json.JSONDecodeError, OSError):
                pass

        # DLC quality JSON
        elif fname.endswith(FilePatterns.DLC_QUALITY_SUFFIX):
            try:
                with open(folder_path / fname, 'r') as f:
                    data = json.load(f)
                metadata["dlc_quality"] = data.get("overall_quality", "unknown")
                metadata["dlc_mean_confidence"] = data.get("mean_confidence", 0)
            except (json.JSONDecodeError, OSError):
                pass

        # Ground truth files - also read completeness status
        # If gt_complete field exists, use it; otherwise infer from segment data
        elif fname.endswith(FilePatterns.SEG_GROUND_TRUTH_SUFFIX):
            metadata["seg_gt"] = True
            try:
                with open(folder_path / fname, 'r') as f:
                    gt_data = json.load(f)
                if "gt_complete" in gt_data:
                    metadata["seg_gt_complete"] = gt_data["gt_complete"]
                else:
                    # Infer: GT file with type="ground_truth" is complete
                    metadata["seg_gt_complete"] = gt_data.get("type") == "ground_truth"
            except (json.JSONDecodeError, OSError):
                pass

        elif fname.endswith(FilePatterns.REACH_GROUND_TRUTH_SUFFIX):
            metadata["reach_gt"] = True
            try:
                with open(folder_path / fname, 'r') as f:
                    gt_data = json.load(f)
                if "gt_complete" in gt_data:
                    metadata["reach_gt_complete"] = gt_data["gt_complete"]
                else:
                    # Infer: GT file with type="ground_truth" is complete
                    metadata["reach_gt_complete"] = gt_data.get("type") == "ground_truth"
            except (json.JSONDecodeError, OSError):
                pass

        elif fname.endswith(FilePatterns.OUTCOME_GROUND_TRUTH_SUFFIX):
            metadata["outcome_gt"] = True
            try:
                with open(folder_path / fname, 'r') as f:
                    gt_data = json.load(f)
                if "gt_complete" in gt_data:
                    metadata["outcome_gt_complete"] = gt_data["gt_complete"]
                else:
                    # Infer: GT file with type="ground_truth" is complete
                    metadata["outcome_gt_complete"] = gt_data.get("type") == "ground_truth"
            except (json.JSONDecodeError, OSError):
                pass

        # Unified ground truth file (v2.4+)
        # Takes priority over individual GT files
        elif fname.endswith(FilePatterns.UNIFIED_GROUND_TRUTH_SUFFIX):
            metadata["unified_gt"] = True
            try:
                with open(folder_path / fname, 'r') as f:
                    gt_data = json.load(f)
                # Read completion status from the unified GT
                completion = gt_data.get("completion_status", {})
                metadata["seg_gt"] = True
                metadata["reach_gt"] = True
                metadata["outcome_gt"] = True
                metadata["seg_gt_complete"] = completion.get("segments_complete", False)
                metadata["reach_gt_complete"] = completion.get("reaches_complete", False)
                metadata["outcome_gt_complete"] = completion.get("outcomes_complete", False)
                metadata["unified_gt_complete"] = completion.get("all_complete", False)
            except (json.JSONDecodeError, OSError):
                pass

    # Override validation status if complete GT exists
    # Complete GT = human reviewed entire video, no further review needed
    if metadata.get("seg_gt_complete"):
        metadata["seg_validation"] = "validated"
    if metadata.get("reach_gt_complete"):
        metadata["reach_validation"] = "validated"
    if metadata.get("outcome_gt_complete"):
        metadata["outcome_validation"] = "validated"

    return metadata


def rebuild_index(index: "PipelineIndex", progress_callback: Callable = None) -> Dict:
    """Full rebuild of the pipeline index.

    Scans all pipeline folders and rebuilds the index from scratch.

    Args:
        index: PipelineIndex instance to rebuild
        progress_callback: Optional callback(current, total, message)

    Returns:
        Dict with rebuild statistics
    """
    stats = {
        "folders_scanned": 0,
        "videos_found": 0,
        "errors": [],
    }

    # Clear existing data
    index._data = index._empty_index()
    index._loaded = True
    index._dirty = True

    total_stages = len(index.STAGES)

    for i, stage in enumerate(index.STAGES):
        if progress_callback:
            progress_callback(i, total_stages, f"Scanning {stage}...")

        folder_path = index.root / stage
        if not folder_path.exists():
            continue

        try:
            count = scan_folder(index, folder_path, stage, progress_callback)
            stats["videos_found"] += count
            stats["folders_scanned"] += 1

            # Update folder mtime
            index.update_folder_mtime(stage)

        except Exception as e:
            stats["errors"].append(f"{stage}: {e}")

    if progress_callback:
        progress_callback(total_stages, total_stages, "Saving index...")

    # Save the index
    index.save()

    return stats


# =============================================================================
# CLI ENTRY POINTS
# =============================================================================

def cli_rebuild():
    """CLI entry point for mousereach-index-rebuild."""
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Rebuild the MouseReach pipeline index"
    )
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    from .index import PipelineIndex

    print("MouseReach Pipeline Index Rebuild")
    print("=" * 50)

    index = PipelineIndex()
    start_time = time.time()

    def progress(current, total, message):
        if not args.quiet:
            print(f"  [{current}/{total}] {message}")

    stats = rebuild_index(index, progress)

    elapsed = time.time() - start_time

    print()
    print(f"Rebuild complete in {elapsed:.1f}s")
    print(f"  Folders scanned: {stats['folders_scanned']}")
    print(f"  Videos indexed: {stats['videos_found']}")
    if stats['errors']:
        print(f"  Errors: {len(stats['errors'])}")
        for err in stats['errors']:
            print(f"    - {err}")

    print()
    print(f"Index saved to: {index.index_path}")


def cli_status():
    """CLI entry point for mousereach-index-status."""
    from .index import PipelineIndex

    index = PipelineIndex()
    index.load()
    index.print_status()


def cli_refresh():
    """CLI entry point for mousereach-index-refresh."""
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Refresh specific folders in the pipeline index"
    )
    parser.add_argument("folders", nargs="*",
                        help="Folder names to refresh (default: all stale)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Refresh all folders")
    args = parser.parse_args()

    from .index import PipelineIndex

    print("MouseReach Pipeline Index Refresh")
    print("=" * 50)

    index = PipelineIndex()
    index.load()

    # Determine which folders to refresh
    if args.all:
        folders = index.STAGES
    elif args.folders:
        folders = args.folders
    else:
        folders = index.check_stale_folders()
        if not folders:
            print("No stale folders detected. Index is up to date.")
            print("Use --all to force refresh all folders.")
            return

    print(f"Refreshing {len(folders)} folder(s)...")
    start_time = time.time()

    for folder in folders:
        print(f"  Refreshing {folder}...")
        index.refresh_folder(folder)

    index.save()

    elapsed = time.time() - start_time
    print()
    print(f"Refresh complete in {elapsed:.1f}s")
    print(f"Index saved to: {index.index_path}")


def main():
    """Default CLI entry point (same as rebuild)."""
    cli_rebuild()


if __name__ == "__main__":
    main()
