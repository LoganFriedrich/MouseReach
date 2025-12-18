"""
ASPA2 Segment Finder - Final Version
=====================================

ALGORITHM:
1. Find MOVE periods by detecting motion peaks in smoothed absolute velocity
2. Get the interval from valid consecutive peak pairs (~30.7s)
3. Fit 21 evenly-spaced boundaries to match the pattern

The key insight: we KNOW there are 21 boundaries at ~30.7s intervals.
Find the pattern, fit the model. Noise doesn't matter because the grid
carries us through.

Move periods mark when the conveyor moved. Boundaries are placed:
- Head move (first): boundary at END of move
- Tail move (last): boundary at START of move  
- Middle moves: boundary at MIDPOINT

But since we're fitting a grid, we just place boundaries at the motion peaks.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import json


@dataclass
class SegmentationResult:
    video_name: str
    total_frames: int
    fps: float
    boundaries: List[int]
    segment_durations: List[float]
    interval_frames: float
    n_peaks_detected: int
    confidence: float


def load_dlc_data(filepath: Union[str, Path]) -> pd.DataFrame:
    filepath = Path(filepath)
    if filepath.suffix == '.h5':
        df = pd.read_hdf(filepath)
    else:
        df = pd.read_csv(filepath, header=[0, 1, 2], index_col=0)
    df.columns = ['_'.join([str(c) for c in col[1:]]) for col in df.columns]
    return df


def find_boundaries(df: pd.DataFrame, fps: float = 60.0) -> tuple:
    """
    Find 21 boundaries by detecting motion peaks and fitting a grid.
    
    Returns:
        boundaries: List of 21 frame numbers
        interval: Detected interval in frames
        n_peaks: Number of motion peaks detected
    """
    total_frames = len(df)
    
    # Get SA position signal
    sa = df['SABL_x'].values.copy()
    like = df['SABL_likelihood'].values
    sa[like < 0.5] = np.nan
    sa = pd.Series(sa).interpolate(method='linear', limit_direction='both').ffill().bfill().values
    
    # Motion signal: smoothed absolute velocity
    velocity = np.abs(np.diff(sa))
    velocity = np.concatenate([velocity, [0]])
    motion = uniform_filter1d(velocity, size=int(fps))  # 1-second smoothing
    
    # Find motion peaks (move events)
    # Minimum distance: 20 seconds between peaks
    peaks, _ = find_peaks(motion,
                          distance=int(fps * 20),
                          prominence=np.percentile(motion, 90))
    
    n_peaks = len(peaks)
    
    if n_peaks < 5:
        # Not enough peaks - fall back to even spacing
        interval = total_frames / 22
        boundaries = [int((i + 1) * interval) for i in range(21)]
        return boundaries, interval, n_peaks
    
    # Get interval from consecutive peak pairs
    peaks = sorted(peaks)
    intervals = np.diff(peaks)
    
    # Keep only valid intervals (25-40 seconds)
    valid_intervals = intervals[(intervals > fps * 25) & (intervals < fps * 40)]
    
    if len(valid_intervals) < 3:
        # Not enough valid intervals - use all intervals
        interval = np.median(intervals)
    else:
        interval = np.median(valid_intervals)
    
    # Project grid from last peak backward
    last_peak = peaks[-1]
    n_intervals_to_last = round(last_peak / interval)
    boundary_1 = last_peak - (n_intervals_to_last - 1) * interval
    boundary_1 = max(0, int(boundary_1))
    
    # Generate 21 boundaries
    boundaries = []
    for i in range(21):
        b = int(boundary_1 + i * interval)
        b = max(0, min(total_frames - 1, b))
        boundaries.append(b)
    
    return boundaries, interval, n_peaks


def segment_video(dlc_path: Union[str, Path],
                  fps: float = 60.0,
                  output_path: Path = None) -> SegmentationResult:
    """
    Segment video into 22 segments (garbage + 20 pellets + garbage).
    """
    dlc_path = Path(dlc_path)
    print(f"\nSegmenting: {dlc_path.name}")
    
    df = load_dlc_data(dlc_path)
    total_frames = len(df)
    print(f"  Frames: {total_frames} ({total_frames/fps:.1f}s)")
    
    boundaries, interval, n_peaks = find_boundaries(df, fps)
    print(f"  Motion peaks: {n_peaks}")
    print(f"  Interval: {interval:.0f} frames ({interval/fps:.1f}s)")
    
    # Calculate segment durations
    starts = [0] + boundaries
    ends = boundaries + [total_frames]
    durations = [(e - s) / fps for s, e in zip(starts, ends)]
    
    # Stats on pellet periods (segments 1-20)
    pellet_durations = durations[1:-1]
    
    # Exclude truncated final pellet if needed
    if len(pellet_durations) > 2:
        median_dur = np.median(pellet_durations[:-1])
        if pellet_durations[-1] < median_dur * 0.5:
            cv_durations = pellet_durations[:-1]
        else:
            cv_durations = pellet_durations
    else:
        cv_durations = pellet_durations
    
    mean_dur = np.mean(cv_durations)
    std_dur = np.std(cv_durations)
    cv = std_dur / mean_dur if mean_dur > 0 else 1
    
    print(f"  Pellet durations: {mean_dur:.1f}s ± {std_dur:.1f}s (CV={cv:.3f})")
    
    confidence = max(0, 1 - cv * 10)
    print(f"  Confidence: {confidence:.2f}")
    
    result = SegmentationResult(
        video_name=dlc_path.stem,
        total_frames=total_frames,
        fps=fps,
        boundaries=boundaries,
        segment_durations=durations,
        interval_frames=float(interval),
        n_peaks_detected=n_peaks,
        confidence=float(confidence)
    )
    
    if output_path is None:
        output_path = dlc_path.parent / f"{dlc_path.stem}_segments.json"
    
    with open(output_path, 'w') as f:
        json.dump({
            'video_name': result.video_name,
            'total_frames': result.total_frames,
            'fps': result.fps,
            'boundaries': result.boundaries,
            'segment_durations_seconds': result.segment_durations,
            'interval_frames': result.interval_frames,
            'interval_seconds': result.interval_frames / result.fps,
            'n_peaks_detected': result.n_peaks_detected,
            'confidence': result.confidence,
            'mean_pellet_duration': float(mean_dur),
            'cv': float(cv)
        }, f, indent=2)
    
    print(f"  Saved: {output_path}")
    return result


def interactive_select():
    """
    Interactive file/directory selection.
    Returns list of DLC files to process.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    
    root = tk.Tk()
    root.withdraw()
    
    result = messagebox.askyesno(
        "ASPA2 Segment Finder",
        "Do you want to process a whole directory?\n\n"
        "Yes = Select directory (batch process all DLC files)\n"
        "No = Select individual file(s)"
    )
    
    if result:  # Directory mode
        directory = filedialog.askdirectory(title="Select directory with DLC files")
        if not directory:
            return []
        
        directory = Path(directory)
        
        # Find DLC files - prefer .h5 over .csv
        h5_files = list(directory.glob("*DLC*.h5"))
        csv_files = list(directory.glob("*DLC*.csv"))
        
        # Only use csv if no matching h5 exists
        h5_stems = {f.stem for f in h5_files}
        csv_only = [f for f in csv_files if f.stem not in h5_stems]
        
        dlc_files = h5_files + csv_only
        
        if not dlc_files:
            messagebox.showerror("Error", "No DLC files found in directory")
            return []
        
        print(f"\nFound {len(dlc_files)} DLC files:")
        for f in dlc_files[:10]:
            print(f"  {f.name}")
        if len(dlc_files) > 10:
            print(f"  ... and {len(dlc_files) - 10} more")
        
        confirm = messagebox.askyesno("Confirm", f"Process {len(dlc_files)} files?")
        if not confirm:
            return []
        
        return dlc_files
    
    else:  # Individual files
        files = filedialog.askopenfilenames(
            title="Select DLC file(s)",
            filetypes=[("DLC files", "*.h5"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        return [Path(f) for f in files]


def batch_segment(dlc_files: list, output_dir: Path = None):
    """
    Process multiple DLC files.
    """
    results = []
    failed = []
    
    for i, dlc_path in enumerate(dlc_files):
        print(f"\n[{i+1}/{len(dlc_files)}] ", end="")
        
        try:
            if output_dir:
                out_path = output_dir / f"{dlc_path.stem}_segments.json"
            else:
                out_path = dlc_path.parent / f"{dlc_path.stem}_segments.json"
            
            result = segment_video(dlc_path, output_path=out_path)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append((dlc_path, str(e)))
    
    # Summary
    print("\n" + "=" * 50)
    print(f"DONE: {len(results)} succeeded, {len(failed)} failed")
    
    if failed:
        print("\nFailed files:")
        for path, error in failed:
            print(f"  {path.name}: {error}")
    
    # Show CV distribution
    if results:
        cvs = [r.confidence for r in results]
        print(f"\nConfidence: min={min(cvs):.2f}, max={max(cvs):.2f}, mean={np.mean(cvs):.2f}")
    
    return results, failed


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        path = Path(sys.argv[1])
        
        if path.is_dir():
            # Batch process directory
            dlc_files = list(path.glob("*DLC*.h5")) + list(path.glob("*DLC*.csv"))
            if dlc_files:
                print(f"Found {len(dlc_files)} DLC files in {path}")
                batch_segment(dlc_files)
            else:
                print(f"No DLC files found in {path}")
        else:
            # Single file
            segment_video(path)
    
    else:
        # Interactive mode
        print("ASPA2 Segment Finder")
        print("=" * 40)
        
        dlc_files = interactive_select()
        
        if dlc_files:
            if len(dlc_files) == 1:
                segment_video(dlc_files[0])
            else:
                batch_segment(dlc_files)
