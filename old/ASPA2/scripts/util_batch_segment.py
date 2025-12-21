#!/usr/bin/env python3
"""
Batch segmentation utility for ASPA2.
Processes all DLC .h5 files in a selected folder.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tkinter import filedialog, Tk
from aspa2_core.segmenter_robust import segment_video_robust, save_segmentation, SEGMENTER_VERSION


def main():
    # Hide root window
    root = Tk()
    root.withdraw()
    
    # Select folder
    folder = filedialog.askdirectory(title="Select folder containing DLC .h5 files")
    if not folder:
        print("No folder selected. Exiting.")
        return
    
    folder = Path(folder)
    h5_files = list(folder.glob("*DLC*.h5"))
    
    if not h5_files:
        print(f"No DLC .h5 files found in {folder}")
        return
    
    print(f"Found {len(h5_files)} DLC files")
    print(f"Using segmenter v{SEGMENTER_VERSION}")
    print("=" * 60)
    
    results = {"good": [], "warnings": [], "failed": []}
    
    for h5_file in h5_files:
        print(f"\nProcessing: {h5_file.name}")
        
        try:
            boundaries, diag = segment_video_robust(h5_file)
            
            # Determine output path
            video_id = h5_file.stem.split("DLC")[0]
            output_path = h5_file.parent / f"{video_id}_segments_v2.json"
            
            # Save results
            save_segmentation(boundaries, diag, output_path)
            
            # Categorize result based on available diagnostics
            # Use interval_cv and n_primary_candidates instead of overall_confidence
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
            
            print(f"  {status} (detections={diag.n_primary_candidates}, CV={diag.interval_cv:.4f}, anomalies={len(diag.anomalies)})")
            
        except Exception as e:
            results["failed"].append(h5_file.name)
            print(f"  ✗ FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Good:     {len(results['good'])}")
    print(f"Warnings: {len(results['warnings'])}")
    print(f"Failed:   {len(results['failed'])}")
    
    if results["warnings"]:
        print("\nWarnings (review recommended):")
        for f in results["warnings"]:
            print(f"  - {f}")
    
    if results["failed"]:
        print("\nFailed (needs investigation):")
        for f in results["failed"]:
            print(f"  - {f}")


if __name__ == "__main__":
    main()