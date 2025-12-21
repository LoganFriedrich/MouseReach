"""
Batch segment all videos using the robust segmenter.

Usage:
    python batch_segment.py                    # Interactive folder select
    python batch_segment.py /path/to/folder    # Process folder
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aspa2_core.segmenter_robust import (
    segment_video_robust, save_segmentation, print_diagnostics,
    SEGMENTER_VERSION, SEGMENTER_ALGORITHM
)


def batch_segment(folder: Path):
    """Process all DLC files in folder."""
    print(f"ASPA2 Batch Segmenter v{SEGMENTER_VERSION}")
    print(f"Algorithm: {SEGMENTER_ALGORITHM}")
    print("=" * 60)
    # Find DLC files
    h5_files = list(folder.glob("*DLC*.h5"))
    csv_files = list(folder.glob("*DLC*.csv"))
    
    # Prefer h5, avoid duplicates
    h5_stems = {f.stem for f in h5_files}
    csv_only = [f for f in csv_files if f.stem not in h5_stems]
    dlc_files = sorted(h5_files + csv_only)
    
    if not dlc_files:
        print(f"No DLC files found in {folder}")
        return
    
    print(f"Found {len(dlc_files)} DLC files")
    print("=" * 60)
    
    results = []
    
    for i, dlc_path in enumerate(dlc_files):
        print(f"\n[{i+1}/{len(dlc_files)}] {dlc_path.name}")
        
        try:
            boundaries, diag = segment_video_robust(dlc_path)
            
            # Save with _segments_v2.json suffix
            video_stem = dlc_path.stem.split("DLC")[0]
            output_path = dlc_path.parent / f"{video_stem}_segments_v2.json"
            save_segmentation(boundaries, diag, output_path)
            
            # Brief summary
            conf = sum(diag.boundary_confidences) / len(diag.boundary_confidences)
            print(f"  → {output_path.name}")
            print(f"     {diag.n_primary_candidates} detections, conf={conf:.2f}, CV={diag.interval_cv:.4f}")
            
            if diag.anomalies:
                for a in diag.anomalies:
                    print(f"     ⚠ {a}")
            
            results.append((dlc_path.name, conf, len(diag.anomalies)))
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((dlc_path.name, 0, -1))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    good = [r for r in results if r[1] >= 0.8 and r[2] == 0]
    warnings = [r for r in results if r[1] >= 0.5 and (r[2] > 0 or r[1] < 0.8)]
    bad = [r for r in results if r[1] < 0.5 or r[2] < 0]
    
    print(f"Good (conf>=0.8, no anomalies): {len(good)}")
    print(f"Needs review (warnings): {len(warnings)}")
    print(f"Failed/low confidence: {len(bad)}")
    
    if warnings:
        print("\nNeeds review:")
        for name, conf, n_anom in warnings:
            print(f"  {name}: conf={conf:.2f}, anomalies={n_anom}")
    
    if bad:
        print("\nFailed:")
        for name, conf, n_anom in bad:
            print(f"  {name}: conf={conf:.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        
        folder = filedialog.askdirectory(title="Select folder with DLC files")
        if not folder:
            print("Cancelled")
            sys.exit(0)
        folder = Path(folder)
    
    batch_segment(folder)
