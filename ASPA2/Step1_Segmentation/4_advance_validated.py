#!/usr/bin/env python3
"""
4_advance_validated.py - Move validated files to next stage

WHAT THIS DOES:
    After human review with 3_review_tool.py, files have a *_seg_validation.json
    that records the approval. This script moves approved files from:
    
    Seg_AutoReview/   → Seg_Validated/
    Seg_NeedsReview/  → Seg_Validated/

HOW TO USE:
    python 4_advance_validated.py
    → Select source folder (Seg_AutoReview or Seg_NeedsReview)
    → Script finds files with validation records and moves them

WHAT GETS MOVED:
    Only files that have a *_seg_validation.json are moved.
    The entire bundle (video + DLC + segments + validation) moves together.

NEXT STEP:
    Files in Seg_Validated/ are ready for Step 2 (Reach Detection)
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from tkinter import filedialog, Tk, messagebox


# Default staging directories
DEFAULT_BASE = Path(r"A:\!!! DLC Input")
DEST_VALIDATED = DEFAULT_BASE / "Seg_Validated"


def get_associated_files(validation_path):
    """
    Find all files associated with a validated video.
    """
    folder = validation_path.parent
    
    # Extract video ID
    # e.g., "20250806_CNT0311_P2_seg_validation.json" → "20250806_CNT0311_P2"
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


def move_file_bundle(associated_files, destination, log_file):
    """
    Move all associated files to destination folder.
    """
    video_id = associated_files["video_id"]
    moved_files = []
    
    for key, src_path in associated_files.items():
        if key == "video_id" or src_path is None:
            continue
        
        dst_path = destination / src_path.name
        
        if dst_path.exists():
            print(f"    WARNING: {dst_path.name} already exists, skipping")
            continue
        
        shutil.move(str(src_path), str(dst_path))
        moved_files.append(src_path.name)
    
    # Log the move
    timestamp = datetime.now().isoformat()
    user = os.getlogin()
    log_entry = f"{timestamp}\t{user}\t{video_id}\tadvanced_to_validated\t{moved_files}\n"
    
    with open(log_file, "a") as f:
        f.write(log_entry)
    
    return moved_files


def main():
    print("=" * 60)
    print("ASPA2 Step 1: Advance Validated Files")
    print("=" * 60)
    
    # Check destination exists
    if not DEST_VALIDATED.exists():
        print(f"ERROR: Destination folder does not exist: {DEST_VALIDATED}")
        return
    
    # Hide root window
    root = Tk()
    root.withdraw()
    
    # Select source folder
    source = filedialog.askdirectory(
        title="Select folder to advance FROM (Seg_AutoReview or Seg_NeedsReview)",
        initialdir=str(DEFAULT_BASE)
    )
    if not source:
        print("No folder selected. Exiting.")
        return
    
    source = Path(source)
    
    # Find all validation files
    validation_files = list(source.glob("*_seg_validation.json"))
    
    if not validation_files:
        print(f"No validated files (*_seg_validation.json) found in {source}")
        print("\nHave you run 3_review_tool.py and saved validations?")
        return
    
    print(f"\nFound {len(validation_files)} validated videos")
    print(f"Source: {source}")
    print(f"Destination: {DEST_VALIDATED}")
    
    # Show what we found
    print("\nValidated files:")
    for vf in validation_files:
        video_id = vf.stem.replace("_seg_validation", "")
        
        # Read validation to show who validated
        try:
            with open(vf) as f:
                val_data = json.load(f)
            validated_by = val_data.get("validated_by", "unknown")
            validated_at = val_data.get("validated_at", "unknown")
            changes = val_data.get("changes_made", 0)
            print(f"  {video_id} - by {validated_by}, {changes} changes")
        except:
            print(f"  {video_id}")
    
    # Confirm
    response = messagebox.askyesno(
        "Confirm Advance",
        f"Move {len(validation_files)} validated videos to Seg_Validated?\n\n"
        f"Files will be MOVED (not copied).\n\n"
        f"Continue?"
    )
    if not response:
        print("Cancelled.")
        return
    
    print("\n" + "-" * 60)
    
    # Log file
    log_file = DEFAULT_BASE / "advance_log.txt"
    
    moved_count = 0
    
    for val_file in sorted(validation_files):
        video_id = val_file.stem.replace("_seg_validation", "")
        print(f"\n{video_id}")
        
        # Get associated files
        associated = get_associated_files(val_file)
        
        # Move them
        moved = move_file_bundle(associated, DEST_VALIDATED, log_file)
        print(f"  → Moved to Seg_Validated/ ({len(moved)} files)")
        moved_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("ADVANCE COMPLETE")
    print("=" * 60)
    print(f"Moved: {moved_count} videos to Seg_Validated/")
    print(f"Log file: {log_file}")
    
    print("\n" + "-" * 60)
    print("Files are now ready for Step 2 (Reach Detection)")
    print("(Step 2 not yet implemented)")
    print("-" * 60)


if __name__ == "__main__":
    main()
