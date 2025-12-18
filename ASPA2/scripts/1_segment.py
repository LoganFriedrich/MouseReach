"""
ASPA2 Step 1: Segment Videos
============================

Batch process DLC files to find segment boundaries.

Usage:
    python 1_segment.py                    # Interactive - select directory
    python 1_segment.py /path/to/dlc/dir   # Batch process directory
    python 1_segment.py file.h5            # Single file
"""

import sys
from pathlib import Path
import numpy as np

# Add parent to path so we can import aspa2_core
sys.path.insert(0, str(Path(__file__).parent.parent))

from aspa2_core import segment_video
from aspa2_core.segmenter import save_segmentation


def interactive_select():
    """Interactive file/directory selection."""
    import tkinter as tk
    from tkinter import filedialog, messagebox
    
    root = tk.Tk()
    root.withdraw()
    
    result = messagebox.askyesno(
        "ASPA2 Segment Finder",
        "Process a whole directory?\n\n"
        "Yes = Batch process all DLC files\n"
        "No = Select individual file(s)"
    )
    
    if result:
        directory = filedialog.askdirectory(title="Select directory with DLC files")
        if not directory:
            return []
        
        directory = Path(directory)
        h5_files = list(directory.glob("*DLC*.h5"))
        csv_files = list(directory.glob("*DLC*.csv"))
        h5_stems = {f.stem for f in h5_files}
        csv_only = [f for f in csv_files if f.stem not in h5_stems]
        dlc_files = h5_files + csv_only
        
        if not dlc_files:
            messagebox.showerror("Error", "No DLC files found")
            return []
        
        print(f"\nFound {len(dlc_files)} DLC files")
        confirm = messagebox.askyesno("Confirm", f"Process {len(dlc_files)} files?")
        return dlc_files if confirm else []
    
    else:
        files = filedialog.askopenfilenames(
            title="Select DLC file(s)",
            filetypes=[("DLC files", "*.h5"), ("CSV", "*.csv"), ("All", "*.*")]
        )
        return [Path(f) for f in files]


def batch_segment(dlc_files: list, output_dir: Path = None):
    """Process multiple DLC files."""
    results = []
    failed = []
    
    for i, dlc_path in enumerate(dlc_files):
        print(f"\n[{i+1}/{len(dlc_files)}] ", end="")
        try:
            out_path = (output_dir or dlc_path.parent) / f"{dlc_path.stem}_segments.json"
            result = segment_video(dlc_path, output_path=out_path)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append((dlc_path, str(e)))
    
    print("\n" + "=" * 50)
    print(f"DONE: {len(results)} succeeded, {len(failed)} failed")
    
    if failed:
        print("\nFailed:")
        for p, e in failed:
            print(f"  {p.name}: {e}")
    
    if results:
        cvs = [r.confidence for r in results]
        print(f"\nConfidence: min={min(cvs):.2f}, max={max(cvs):.2f}, mean={np.mean(cvs):.2f}")
    
    return results, failed


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.is_dir():
            h5_files = list(path.glob("*DLC*.h5"))
            csv_files = list(path.glob("*DLC*.csv"))
            h5_stems = {f.stem for f in h5_files}
            csv_only = [f for f in csv_files if f.stem not in h5_stems]
            dlc_files = h5_files + csv_only
            if dlc_files:
                print(f"Found {len(dlc_files)} DLC files")
                batch_segment(dlc_files)
            else:
                print(f"No DLC files found in {path}")
        else:
            segment_video(path, output_path=path.parent / f"{path.stem}_segments.json")
    else:
        print("ASPA2 Step 1: Segment Videos")
        print("=" * 40)
        dlc_files = interactive_select()
        if dlc_files:
            if len(dlc_files) == 1:
                p = dlc_files[0]
                segment_video(p, output_path=p.parent / f"{p.stem}_segments.json")
            else:
                batch_segment(dlc_files)
