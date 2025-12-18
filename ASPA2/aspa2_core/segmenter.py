"""
ASPA2 Video Segmenter
=====================

Finds 21 segment boundaries in ASPA videos using motion peak detection.

Algorithm:
1. Detect motion peaks in smoothed absolute velocity of SABL position
2. Find the consistent interval between peaks (~30.7 seconds)
3. Fit a 21-boundary grid at that interval
4. Report confidence based on consistency (CV of segment durations)

Usage:
    from aspa2_core import segment_video
    
    result = segment_video('path/to/dlc_file.h5')
    print(result.boundaries)
    print(result.confidence)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union, Tuple, Optional
from dataclasses import dataclass
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import json

from .dlc_utils import load_dlc_data


@dataclass
class SegmentationResult:
    """Result of video segmentation."""
    video_name: str
    total_frames: int
    fps: float
    boundaries: List[int]
    segment_durations: List[float]
    interval_frames: float
    n_peaks_detected: int
    confidence: float


def find_boundaries(df: pd.DataFrame, fps: float = 60.0) -> Tuple[List[int], float, int]:
    """
    Find 21 boundaries by detecting motion peaks and fitting a grid.
    
    Args:
        df: DataFrame with DLC tracking data (must have SABL_x and SABL_likelihood)
        fps: Video frame rate
    
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
    motion = uniform_filter1d(velocity, size=int(fps))
    
    # Find motion peaks (move events)
    # Minimum distance: 20 seconds between peaks
    peaks, _ = find_peaks(motion,
                          distance=int(fps * 20),
                          prominence=np.percentile(motion, 90))
    
    n_peaks = len(peaks)
    
    if n_peaks < 5:
        # Fall back to even spacing
        interval = total_frames / 22
        boundaries = [int((i + 1) * interval) for i in range(21)]
        return boundaries, interval, n_peaks
    
    # Get interval from consecutive peak pairs
    peaks = sorted(peaks)
    intervals = np.diff(peaks)
    
    # Keep only valid intervals (25-40 seconds)
    valid_intervals = intervals[(intervals > fps * 25) & (intervals < fps * 40)]
    
    if len(valid_intervals) < 3:
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
                  output_path: Optional[Path] = None) -> SegmentationResult:
    """
    Segment video into 22 segments (garbage_pre + 20 pellets + garbage_post).
    
    Args:
        dlc_path: Path to DLC output file (.h5 or .csv)
        fps: Video frame rate (default 60)
        output_path: Optional path to save results as JSON
    
    Returns:
        SegmentationResult with boundaries, durations, and confidence
    """
    dlc_path = Path(dlc_path)
    print(f"Segmenting: {dlc_path.name}")
    
    df = load_dlc_data(dlc_path)
    total_frames = len(df)
    
    boundaries, interval, n_peaks = find_boundaries(df, fps)
    
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
    
    confidence = max(0, 1 - cv * 10)
    
    print(f"  Frames: {total_frames}, Peaks: {n_peaks}, CV: {cv:.3f}, Confidence: {confidence:.2f}")
    
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
    
    # Save if output path provided
    if output_path:
        save_segmentation(result, output_path)
    
    return result


def save_segmentation(result: SegmentationResult, output_path: Union[str, Path]):
    """Save segmentation result to JSON file."""
    output_path = Path(output_path)
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
        }, f, indent=2)
    print(f"  Saved: {output_path}")


def load_segmentation(json_path: Union[str, Path]) -> dict:
    """Load segmentation from JSON file."""
    with open(json_path) as f:
        return json.load(f)
