#!/usr/bin/env python3
"""
1_batch_segment.py - Run segmentation on all DLC files in a folder

WHAT THIS DOES:
    Takes a folder of DLC .h5 files and runs the segmentation algorithm on each.
    Creates a *_segments.json file next to each .h5 file.

HOW TO USE:
    python 1_batch_segment.py
    → A dialog opens, select the folder containing DLC .h5 files
    → Script processes each file and reports results

OUTPUT:
    - *_segments.json files (one per video)
    - Console summary showing GOOD / WARNINGS / FAILED counts

NEXT STEP:
    Run 2_triage_results.py to move files into review queues
"""

import sys
from pathlib import Path

# Add core module to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from tkinter import filedialog, Tk
from core.segmenter_robust import segment_video_robust, save_segmentation, SEGMENTER_VERSION


def main():
    print("=" * 60)
    print("ASPA2 Step 1: Batch Segmentation")
    print(f"Segmenter version: {SEGMENTER_VERSION}")
    print("=" * 60)
    
    # Hide root window
    root = Tk()
    root.withdraw()
    
    # Select folder
    folder = filedialog.askdirectory(
        title="Select folder containing DLC .h5 files"
    )
    if not folder:
        print("No folder selected. Exiting.")
        return
    
    folder = Path(folder)
    h5_files = list(folder.glob("*DLC*.h5"))
    
    if not h5_files:
        print(f"No DLC .h5 files found in {folder}")
        return
    
    print(f"\nFound {len(h5_files)} DLC files")
    print("-" * 60)
    
    results = {"good": [], "warnings": [], "failed": []}
    
    for i, h5_file in enumerate(sorted(h5_files), 1):
        print(f"\n[{i}/{len(h5_files)}] {h5_file.name}")
        
        try:
            # Run segmentation
            boundaries, diag = segment_video_robust(h5_file)
            
            # Determine output path
            # Extract video ID: everything before "DLC"
            video_id = h5_file.stem.split("DLC")[0]
            output_path = h5_file.parent / f"{video_id}_segments.json"
            
            # Save results
            save_segmentation(boundaries, diag, output_path)
            
            # Categorize result
            has_21 = diag.n_primary_candidates >= 21
            low_cv = diag.interval_cv < 0.05
            no_anomalies = len(diag.anomalies) == 0
            
            if has_21 and low_cv and no_anomalies:
                results["good"].append(h5_file.name)
                status = "✓ GOOD"
            elif diag.n_primary_candidates >= 19:
                results["warnings"].append(h5_file.name)
                status = "⚠ WARNINGS"
            else:
                results["failed"].append(h5_file.name)
                status = "✗ NEEDS REVIEW"
            
            print(f"   {status}")
            print(f"   Detections: {diag.n_primary_candidates}, CV: {diag.interval_cv:.4f}")
            if diag.anomalies:
                print(f"   Anomalies: {diag.anomalies}")
            
        except Exception as e:
            results["failed"].append(h5_file.name)
            print(f"   ✗ FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Good (auto-review):     {len(results['good'])}")
    print(f"Warnings (needs review): {len(results['warnings'])}")
    print(f"Failed (investigate):    {len(results['failed'])}")
    
    if results["warnings"]:
        print("\nFiles needing review:")
        for f in results["warnings"]:
            print(f"  - {f}")
    
    if results["failed"]:
        print("\nFiles that failed:")
        for f in results["failed"]:
            print(f"  - {f}")
    
    print("\n" + "-" * 60)
    print("NEXT STEP: Run 2_triage_results.py to move files to review queues")
    print("-" * 60)


if __name__ == "__main__":
    main()
