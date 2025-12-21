#!/usr/bin/env python3
"""
2_triage_results.py - Move segmented files to appropriate review queues

WHAT THIS DOES:
    Reads the *_segments.json files created by 1_batch_segment.py and moves
    each VIDEO + DLC + SEGMENTS bundle to the appropriate staging folder:
    
    - Seg_AutoReview/    → High confidence, just needs quick visual check
    - Seg_NeedsReview/   → Low confidence or anomalies, needs careful review
    - Failed/            → Could not segment, needs investigation

HOW TO USE:
    python 2_triage_results.py
    → Select the source folder (e.g., DLC_Complete)
    → Script analyzes each segments.json and moves files accordingly

TRIAGE LOGIC:
    GOOD (→ Seg_AutoReview):
        - 21 boundaries detected
        - Interval CV < 0.05
        - No anomalies
    
    WARNING (→ Seg_NeedsReview):
        - 21 boundaries but CV >= 0.05 (possible stuck tray)
        - OR anomalies flagged
        - OR 19-20 boundaries
    
    FAILED (→ Failed):
        - <19 boundaries
        - Segmentation algorithm raised error

NEXT STEP:
    Run 3_review_tool.py to review/correct segments
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from tkinter import filedialog, Tk, messagebox


# Default staging directories (relative to A:\!!! DLC Input\)
DEFAULT_BASE = Path(r"A:\!!! DLC Input")
DEST_AUTO_REVIEW = DEFAULT_BASE / "Seg_AutoReview"
DEST_NEEDS_REVIEW = DEFAULT_BASE / "Seg_NeedsReview"
DEST_FAILED = DEFAULT_BASE / "Failed"


def get_associated_files(segments_path):
    """
    Find all files associated with a video.
    
    Given a segments.json, finds:
    - The source video (.mp4)
    - The DLC tracking files (.h5, .csv, .pickle)
    - The segments file itself
    """
    folder = segments_path.parent
    
    # Extract video ID from segments filename
    # e.g., "20250806_CNT0311_P2_segments.json" → "20250806_CNT0311_P2"
    video_id = segments_path.stem.replace("_segments", "")
    
    associated = {
        "video_id": video_id,
        "segments": segments_path,
        "video": None,
        "dlc_h5": None,
        "dlc_csv": None,
        "dlc_pickle": None,
    }
    
    # Find video
    for ext in [".mp4", ".avi", ".mkv"]:
        video_path = folder / f"{video_id}{ext}"
        if video_path.exists():
            associated["video"] = video_path
            break
    
    # Find DLC files (they have DLC in the name)
    for f in folder.glob(f"{video_id}DLC*"):
        if f.suffix == ".h5":
            associated["dlc_h5"] = f
        elif f.suffix == ".csv":
            associated["dlc_csv"] = f
        elif f.suffix == ".pickle":
            associated["dlc_pickle"] = f
    
    return associated


def classify_segments(segments_path):
    """
    Read segments.json and determine triage category.
    
    Returns: "good", "warning", or "failed"
    """
    try:
        with open(segments_path) as f:
            data = json.load(f)
    except Exception as e:
        return "failed", f"Could not read segments file: {e}"
    
    # Check for required fields
    if "boundaries" not in data or "diagnostics" not in data:
        return "failed", "Missing required fields in segments file"
    
    n_boundaries = len(data["boundaries"])
    diag = data["diagnostics"]
    
    cv = diag.get("interval_cv", 999)
    anomalies = diag.get("anomalies", [])
    
    # Classification logic
    if n_boundaries < 19:
        return "failed", f"Only {n_boundaries} boundaries detected"
    
    if n_boundaries == 21 and cv < 0.05 and len(anomalies) == 0:
        return "good", "High confidence"
    
    # Everything else needs review
    reasons = []
    if n_boundaries != 21:
        reasons.append(f"{n_boundaries} boundaries")
    if cv >= 0.05:
        reasons.append(f"CV={cv:.4f}")
    if anomalies:
        reasons.append(f"anomalies: {anomalies}")
    
    return "warning", "; ".join(reasons)


def move_file_bundle(associated_files, destination, log_file):
    """
    Move all associated files to destination folder.
    Logs the move to log_file.
    """
    video_id = associated_files["video_id"]
    moved_files = []
    
    for key, src_path in associated_files.items():
        if key == "video_id" or src_path is None:
            continue
        
        dst_path = destination / src_path.name
        
        # Don't overwrite existing files
        if dst_path.exists():
            print(f"    WARNING: {dst_path.name} already exists in destination, skipping")
            continue
        
        shutil.move(str(src_path), str(dst_path))
        moved_files.append(src_path.name)
    
    # Log the move
    timestamp = datetime.now().isoformat()
    user = os.getlogin()
    log_entry = f"{timestamp}\t{user}\t{video_id}\t{destination.name}\t{moved_files}\n"
    
    with open(log_file, "a") as f:
        f.write(log_entry)
    
    return moved_files


def main():
    print("=" * 60)
    print("ASPA2 Step 1: Triage Segmentation Results")
    print("=" * 60)
    
    # Check that destination folders exist
    for dest in [DEST_AUTO_REVIEW, DEST_NEEDS_REVIEW, DEST_FAILED]:
        if not dest.exists():
            print(f"ERROR: Destination folder does not exist: {dest}")
            print("Please run the staging folder setup first.")
            return
    
    # Hide root window
    root = Tk()
    root.withdraw()
    
    # Select source folder
    source = filedialog.askdirectory(
        title="Select folder with segmented files (e.g., DLC_Complete)",
        initialdir=str(DEFAULT_BASE)
    )
    if not source:
        print("No folder selected. Exiting.")
        return
    
    source = Path(source)
    
    # Find all segments.json files
    segments_files = list(source.glob("*_segments.json"))
    
    if not segments_files:
        print(f"No *_segments.json files found in {source}")
        return
    
    print(f"\nFound {len(segments_files)} segmented videos")
    print(f"Source: {source}")
    print(f"Destinations:")
    print(f"  Good     → {DEST_AUTO_REVIEW}")
    print(f"  Warnings → {DEST_NEEDS_REVIEW}")
    print(f"  Failed   → {DEST_FAILED}")
    
    # Confirm
    response = messagebox.askyesno(
        "Confirm Triage",
        f"Found {len(segments_files)} videos to triage.\n\n"
        f"Files will be MOVED (not copied) to staging folders.\n\n"
        f"Continue?"
    )
    if not response:
        print("Cancelled.")
        return
    
    print("\n" + "-" * 60)
    
    # Log file for tracking moves
    log_file = DEFAULT_BASE / "triage_log.txt"
    
    counts = {"good": 0, "warning": 0, "failed": 0}
    
    for seg_file in sorted(segments_files):
        video_id = seg_file.stem.replace("_segments", "")
        print(f"\n{video_id}")
        
        # Classify
        category, reason = classify_segments(seg_file)
        counts[category] += 1
        
        # Determine destination
        if category == "good":
            dest = DEST_AUTO_REVIEW
            symbol = "✓"
        elif category == "warning":
            dest = DEST_NEEDS_REVIEW
            symbol = "⚠"
        else:
            dest = DEST_FAILED
            symbol = "✗"
        
        print(f"  {symbol} {category.upper()}: {reason}")
        
        # Find and move associated files
        associated = get_associated_files(seg_file)
        
        # Check we have the essentials
        if associated["dlc_h5"] is None:
            print(f"  WARNING: No DLC .h5 file found, skipping")
            continue
        
        moved = move_file_bundle(associated, dest, log_file)
        print(f"  → Moved to {dest.name}/ ({len(moved)} files)")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRIAGE COMPLETE")
    print("=" * 60)
    print(f"Seg_AutoReview:   {counts['good']}")
    print(f"Seg_NeedsReview:  {counts['warning']}")
    print(f"Failed:           {counts['failed']}")
    print(f"\nLog file: {log_file}")
    
    print("\n" + "-" * 60)
    print("NEXT STEP: Run 3_review_tool.py to review segments")
    print("-" * 60)


if __name__ == "__main__":
    main()
