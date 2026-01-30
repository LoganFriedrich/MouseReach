"""
advance.py - Move validated files to next pipeline stage

Extracted from 4_advance_validated.py

Paths are now configured via environment variables.
Set MouseReach_PROCESSING_ROOT to customize pipeline location.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from mousereach.config import Paths
from mousereach.utils import get_username


# Default staging directories - derived from configurable environment variables
DEST_VALIDATED = Paths.SEG_VALIDATED


def get_associated_files(validation_path: Path) -> Dict:
    """Find all files associated with a validated video."""
    folder = validation_path.parent
    video_id = validation_path.stem.replace("_seg_validation", "")
    
    associated = {
        "video_id": video_id,
        "validation": validation_path,
        "segments": None,
        "video": None,
        "dlc_h5": None,
        "dlc_csv": None,
        "dlc_pickle": None,
    }
    
    # Find segments file
    segments_path = folder / f"{video_id}_segments.json"
    if segments_path.exists():
        associated["segments"] = segments_path
    
    # Find video
    for ext in [".mp4", ".avi", ".mkv"]:
        video_path = folder / f"{video_id}{ext}"
        if video_path.exists():
            associated["video"] = video_path
            break
    
    # Find DLC files
    for f in folder.glob(f"{video_id}DLC*"):
        if f.suffix == ".h5":
            associated["dlc_h5"] = f
        elif f.suffix == ".csv":
            associated["dlc_csv"] = f
        elif f.suffix == ".pickle":
            associated["dlc_pickle"] = f
    
    return associated


def move_file_bundle(associated_files: Dict, destination: Path, log_file: Path,
                     source_stage: str = None) -> List[str]:
    """Move all associated files to destination folder."""
    video_id = associated_files["video_id"]
    moved_files = []

    for key, src_path in associated_files.items():
        if key == "video_id" or src_path is None:
            continue

        dst_path = destination / src_path.name

        if dst_path.exists():
            continue

        shutil.move(str(src_path), str(dst_path))
        moved_files.append(src_path.name)

    # Log
    timestamp = datetime.now().isoformat()
    user = get_username()
    log_entry = f"{timestamp}\t{user}\t{video_id}\tadvanced_to_validated\t{moved_files}\n"

    with open(log_file, "a") as f:
        f.write(log_entry)

    # Update pipeline index
    if moved_files:
        try:
            from mousereach.index import PipelineIndex
            index = PipelineIndex()
            index.load()
            index.record_files_moved(video_id, source_stage or "unknown", destination.name, moved_files)
            index.record_validation_changed(video_id, "seg", "validated")
            index.save()
        except Exception as e:
            print(f"[Advance] Warning: Could not update index: {e}")

    return moved_files


def advance_videos(
    source_dir: Path,
    dest_dir: Path = None,
    verbose: bool = True
) -> Dict:
    """
    Move validated videos to next pipeline stage.
    
    Only moves videos that have *_seg_validation.json files.
    """
    if dest_dir is None:
        dest_dir = DEST_VALIDATED
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Find validation files
    validation_files = list(source_dir.glob("*_seg_validation.json"))
    
    if not validation_files:
        if verbose:
            print(f"No validated files found in {source_dir}")
            print("Have you run the review tool and saved validations?")
        return {'total': 0, 'advanced': 0}
    
    if verbose:
        print(f"Found {len(validation_files)} validated videos")
        print("-" * 60)
    
    log_file = source_dir.parent / "advance_log.txt"
    advanced = 0
    
    for val_file in sorted(validation_files):
        video_id = val_file.stem.replace("_seg_validation", "")
        
        if verbose:
            # Show validation info
            try:
                with open(val_file) as f:
                    val_data = json.load(f)
                validated_by = val_data.get("validated_by", "unknown")
                changes = val_data.get("changes_made", 0)
                print(f"  {video_id} (by {validated_by}, {changes} changes)...", end=" ")
            except (OSError, json.JSONDecodeError):
                print(f"  {video_id}...", end=" ")
        
        associated = get_associated_files(val_file)
        moved = move_file_bundle(associated, dest_dir, log_file, source_stage=source_dir.name)
        
        if verbose:
            print(f"OK ({len(moved)} files)")
        
        advanced += 1
    
    if verbose:
        print("-" * 60)
        print(f"Advanced {advanced} videos to {dest_dir.name}")
    
    return {'total': len(validation_files), 'advanced': advanced}
