"""
core.py - Archive logic for moving completed videos to NAS.

Videos can only be archived when ALL validation statuses are "validated".
This is the only way files leave the Processing/ folder.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import shutil
from datetime import datetime

from mousereach.config import Paths, get_video_id, AnimalID


def get_archivable_videos() -> List[str]:
    """Get list of videos that are ready to be archived.

    A video is archivable when ALL of:
    - seg_validation == "validated"
    - reach_validation == "validated"
    - outcome_validation == "validated"

    Returns:
        List of video_ids ready for archiving
    """
    from mousereach.index import PipelineIndex

    index = PipelineIndex()
    index.load()
    return index.get_ready_to_archive()


def get_video_files(video_id: str) -> List[Path]:
    """Get all files associated with a video in Processing folder.

    Args:
        video_id: Video identifier (e.g., "20250704_CNT0101_P1")

    Returns:
        List of file paths
    """
    files = []
    processing = Paths.PROCESSING

    if not processing.exists():
        return files

    for f in processing.iterdir():
        if f.is_file() and f.name.startswith(video_id):
            files.append(f)

    return files


def check_archive_ready(video_id: str) -> Tuple[bool, Dict[str, str]]:
    """Check if a video is ready for archiving.

    Args:
        video_id: Video identifier

    Returns:
        (is_ready, status_dict) - status_dict has keys seg, reach, outcome
    """
    from mousereach.index import PipelineIndex

    index = PipelineIndex()
    index.load()

    status = index.get_pipeline_status(video_id)

    accepted = {"validated", "auto_approved"}
    is_ready = (
        status["seg"] in accepted and
        status["reach"] in accepted and
        status["outcome"] in accepted
    )

    return is_ready, status


def get_archive_destination(video_id: str) -> Path:
    """Determine archive destination based on experiment code.

    Args:
        video_id: Video identifier (e.g., "20250704_CNT0101_P1")

    Returns:
        Destination folder path (e.g., {NAS_DRIVE}/Analyzed/Sort/CNT/)
    """
    # Extract animal ID from video_id (format: DATE_ANIMALID_TRAY)
    parts = video_id.split("_")
    if len(parts) >= 2:
        animal_id = parts[1]
        experiment = AnimalID.get_experiment(animal_id)
    else:
        experiment = "UNKNOWN"

    return Paths.ANALYZED_OUTPUT / experiment


def archive_video(
    video_id: str,
    dry_run: bool = False,
    verbose: bool = True,
    skip_ready_check: bool = False,
    source_dir: Path = None,
) -> Dict:
    """Archive a video to NAS.

    Moves all files from Processing/ to NAS archive, organized by experiment.

    Args:
        video_id: Video identifier
        dry_run: If True, only show what would be done
        verbose: Print progress
        skip_ready_check: If True, skip validation status check
        source_dir: If set, discover files from this directory instead of Processing/

    Returns:
        Dict with archive results
    """
    from mousereach.index import PipelineIndex

    result = {
        "video_id": video_id,
        "success": False,
        "files_moved": [],
        "destination": None,
        "error": None,
    }

    # Check if ready
    if not skip_ready_check:
        is_ready, status = check_archive_ready(video_id)

        if not is_ready:
            not_validated = [k for k, v in status.items() if v not in ("validated", "auto_approved")]
            result["error"] = f"Not ready: {', '.join(not_validated)} not validated"
            if verbose:
                print(f"Cannot archive {video_id}: {result['error']}")
            return result

    # Get files
    if source_dir:
        files = [f for f in source_dir.iterdir() if f.is_file() and f.name.startswith(video_id)]
    else:
        files = get_video_files(video_id)
    if not files:
        result["error"] = "No files found in Processing/"
        if verbose:
            print(f"Cannot archive {video_id}: No files found")
        return result

    # Determine destination
    dest = get_archive_destination(video_id)
    result["destination"] = str(dest)

    if verbose:
        print(f"Archive {video_id}:")
        print(f"  Source: {Paths.PROCESSING}")
        print(f"  Destination: {dest}")
        print(f"  Files: {len(files)}")

    if dry_run:
        if verbose:
            print("  [DRY RUN - no files moved]")
            for f in files:
                print(f"    Would move: {f.name}")
        result["success"] = True
        result["files_moved"] = [f.name for f in files]
        result["dry_run"] = True
        return result

    # Create destination if needed
    dest.mkdir(parents=True, exist_ok=True)

    # Move files
    moved = []
    for f in files:
        try:
            dest_path = dest / f.name
            shutil.move(str(f), str(dest_path))
            moved.append(f.name)
            if verbose:
                print(f"    Moved: {f.name}")
        except Exception as e:
            if verbose:
                print(f"    FAILED: {f.name} - {e}")

    result["files_moved"] = moved
    result["success"] = len(moved) == len(files)

    # Update index - remove video
    if result["success"]:
        try:
            index = PipelineIndex()
            index.load()
            index.remove_video(video_id)
            index.save()
            if verbose:
                print("  Index updated")
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to update index: {e}")

    if verbose:
        if result["success"]:
            print(f"  SUCCESS: {len(moved)} files archived")
        else:
            print(f"  PARTIAL: {len(moved)}/{len(files)} files archived")

    return result


def archive_all(
    dry_run: bool = False,
    verbose: bool = True
) -> Dict:
    """Archive all videos that are ready.

    Args:
        dry_run: If True, only show what would be done
        verbose: Print progress

    Returns:
        Summary dict with counts and per-video results
    """
    archivable = get_archivable_videos()

    if verbose:
        print(f"Found {len(archivable)} video(s) ready for archive")
        if dry_run:
            print("[DRY RUN MODE]")
        print("-" * 60)

    results = {
        "total": len(archivable),
        "success": 0,
        "failed": 0,
        "videos": [],
        "archived_at": datetime.now().isoformat(),
    }

    for video_id in archivable:
        result = archive_video(video_id, dry_run=dry_run, verbose=verbose)
        results["videos"].append(result)
        if result["success"]:
            results["success"] += 1
        else:
            results["failed"] += 1

    if verbose:
        print("-" * 60)
        print(f"Archived: {results['success']}/{results['total']}")

    return results
