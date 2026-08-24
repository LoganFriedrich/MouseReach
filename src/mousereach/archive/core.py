"""
core.py - Archive logic for moving completed videos to NAS.

Videos can only be archived when ALL validation statuses are "validated".
This is the only way files leave the Processing/ folder.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
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


# The three stage outputs, and where each records the verdict triage gave it.
_STAGE_FILES = {
    "seg": "_segments.json",
    "reach": "_reaches.json",
    "outcome": "_pellet_outcomes.json",
}

_ACCEPTED = {"validated", "auto_approved"}


def _status_from_files(video_id: str, source_dir: Path) -> Dict[str, str]:
    """Read each stage's validation_status straight out of its output file."""
    status: Dict[str, str] = {}
    for stage, suffix in _STAGE_FILES.items():
        f = Path(source_dir) / f"{video_id}{suffix}"
        try:
            if not f.is_file():
                status[stage] = "not_started"
                continue
            with open(f) as fh:
                status[stage] = json.load(fh).get("validation_status") or "not_started"
        except Exception:
            # Unreadable is not approved. Saying "not_started" here is honest:
            # we cannot show that a human or the triage step cleared this stage.
            status[stage] = "unreadable"
    return status


def check_archive_ready(video_id: str, source_dir: Path = None) -> Tuple[bool, Dict[str, str]]:
    """Check if a video is ready for archiving.

    Reads the verdict out of the stage output files. It used to ask the pipeline
    index -- a cache whose whole purpose is to save a folder scan for the
    dashboard -- and that cache is allowed to be stale. It was, comprehensively:
    on 2026-08-24 this node had 1,084 videos held in 'processed' of which 1,083
    were marked approved by every stage ON DISK, while the index said otherwise
    for the large majority (segmentation reading 'auto_review', a spelling only
    the index ever uses, and reach and outcome missing entirely for 39 of every
    60 sampled). Nothing had archived since February and the retry loop had
    logged 326,235 failures.

    A cache is the wrong authority for an irreversible decision. The files carry
    validation_status; that is the fact. The index stays what it is good at --
    making the dashboard fast.

    Args:
        video_id: Video identifier
        source_dir: Where the outputs live. Defaults to the local Processing dir.

    Returns:
        (is_ready, status_dict) - status_dict has keys seg, reach, outcome
    """
    if source_dir is None:
        source_dir = Paths.PROCESSING
    if source_dir is None:
        return False, {k: "no processing dir configured" for k in _STAGE_FILES}

    status = _status_from_files(video_id, Path(source_dir))
    is_ready = all(status.get(k) in _ACCEPTED for k in _STAGE_FILES)
    return is_ready, status


def get_archive_destination(video_id: str) -> Path:
    """Determine archive destination based on project/cohort.

    Args:
        video_id: Video identifier (e.g., "20250704_CNT0101_P1")

    Returns:
        Destination folder path (e.g., Analyzed/Connectome/CNT01/)
    """
    # Extract animal ID from video_id (format: DATE_ANIMALID_TRAY)
    parts = video_id.split("_")
    if len(parts) >= 2:
        animal_id = parts[1]
        project, cohort = AnimalID.get_project_and_cohort(animal_id)
    else:
        project, cohort = "UNKNOWN", "UNKNOWN"

    return Paths.ANALYZED_OUTPUT / project / cohort


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
        is_ready, status = check_archive_ready(video_id, source_dir=source_dir)

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

    # Anything already at the destination is the PREVIOUS generation's results,
    # and the move below would replace it silently -- shutil.move overwrites. On
    # a reprocess that destroyed the earlier segments, reaches, outcomes,
    # assignments, manifest and kinematics with no copy kept, so there was no way
    # afterwards to see what an earlier model produced or to reproduce a figure
    # made from it.
    #
    # supersede_video_outputs sweeps that generation into the versioned Archive
    # first, checksum-verified, reading the OLD manifest sitting there to decide
    # which model generation and algorithm stack it belonged to. It deliberately
    # leaves the video and any ground-truth or human-review file in place, so a
    # review still travels with its video.
    #
    # Failing to archive must not cost the results: if the sweep reports failures
    # we stop rather than move new files on top of the old ones.
    superseded = None
    if any(dest.glob(f"{video_id}*")):
        try:
            from mousereach.archive.supersede import supersede_video_outputs
            superseded = supersede_video_outputs(video_id, dest)
            if superseded.get("failed"):
                result["error"] = (
                    "refusing to archive: could not preserve the previous "
                    "generation of %s (%s)"
                    % (video_id, ", ".join(superseded["failed"])))
                if verbose:
                    print(f"  {result['error']}")
                return result
            n = len(superseded.get("algo", [])) + len(superseded.get("pose", []))
            if n and verbose:
                print(f"  Superseded {n} earlier file(s) -> "
                      f"{superseded.get('generation')}/{superseded.get('stack')}")
        except Exception as e:
            result["error"] = f"refusing to archive: superseding failed ({e})"
            if verbose:
                print(f"  {result['error']}")
            return result
    result["superseded"] = superseded

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
