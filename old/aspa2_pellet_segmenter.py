"""
ASPA2 Pellet Segmenter
=====================

PURPOSE:
    Segments a video's DLC tracking data into individual pellet time periods.
    Uses the "weirdness detection" method: finds rightward velocity peaks in 
    the smoothed SA (scoring area) position to identify pellet transitions.

    All thresholds are RULER-RELATIVE, where:
    1 ruler = SABL-SABR distance = 9mm (physical constant)

INPUT:
    - DLC output file (.csv or .h5) containing tracked points
    - Required points: SABL, SABR (scoring area bottom corners)
    - Optional: SATL, SATR, Pellet, Pillar, Reference, BOXL, BOXR

OUTPUT:
    - JSON file with pellet boundaries (frame ranges)
    - Ruler calibration (px-to-mm conversion for this video)
    - Per-video model (box positions, SA geometry)
    - Summary statistics

USAGE:
    Command line:
        python aspa2_pellet_segmenter.py                    # Opens file dialog
        python aspa2_pellet_segmenter.py path/to/file.csv   # Direct path
        python aspa2_pellet_segmenter.py path/to/folder/    # Process all CSVs in folder
    
    As module:
        from aspa2_pellet_segmenter import segment_video
        results = segment_video('path/to/dlc_output.csv')

ALGORITHM:
    1. Calibrate ruler from SABL-SABR distance (establishes pixel scale)
    
    2. Build per-video models:
       - Box model: median positions of Reference, BOXL, BOXR from high-likelihood frames
       - SA geometry: computed from ruler (width=1.0 ruler, height=1.667 ruler)
    
    3. Estimate SA position over time using weighted average of SABL/SABR
    
    4. Apply light smoothing (0.5s window)
    
    5. Find "weirdness" events: peaks in rightward velocity (>0.5 ruler/s)
       - These mark transitions between pellets
       - Expect ~21 events (entry + 19 transitions + exit) for 20 pellets
    
    6. Define pellet boundaries:
       - Pellet 1: frame 0 to weirdness[0]-1
       - Pellet N: weirdness[N-2] to weirdness[N-1]-1
       - Pellet 20: weirdness[18] to weirdness[19]-1 (or end if no exit)

DEPENDENCIES:
    numpy, pandas, scipy, tkinter (for file dialog)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# Import ruler system
try:
    from aspa2_ruler import RulerCalibration, Geometry, RULER_UNITS, THRESHOLDS, TIMING
except ImportError:
    # If not installed, try local import
    import importlib.util
    spec = importlib.util.spec_from_file_location("aspa2_ruler", 
        Path(__file__).parent / "aspa2_ruler.py")
    ruler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ruler_module)
    RulerCalibration = ruler_module.RulerCalibration
    Geometry = ruler_module.Geometry
    RULER_UNITS = ruler_module.RULER_UNITS
    THRESHOLDS = ruler_module.THRESHOLDS
    TIMING = ruler_module.TIMING


@dataclass
class BoxModel:
    """Fixed box reference points (don't move during video)."""
    reference_x: float
    reference_y: float
    boxl_x: float
    boxl_y: float
    boxr_x: float
    boxr_y: float


@dataclass 
class SAModel:
    """Scoring Area rectangle model - now ruler-based."""
    width_px: float       # Measured width in pixels (= ruler_px)
    height_px: float      # Measured or computed height in pixels
    width_ruler: float    # Always 1.0 by definition
    height_ruler: float   # Should be ~1.667


@dataclass
class PelletSegment:
    """One pellet's time segment."""
    pellet_num: int
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_seconds: float


@dataclass
class VideoSegmentation:
    """Complete segmentation results for one video."""
    video_name: str
    total_frames: int
    fps: float
    ruler_calibration: dict  # From RulerCalibration.to_dict()
    box_model: BoxModel
    sa_model: SAModel
    pellet_segments: List[PelletSegment]
    weirdness_frames: List[int]
    weirdness_magnitudes_ruler: List[float]  # Snap sizes in ruler units
    n_pellets: int


def load_dlc_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load DLC output file (.csv or .h5)."""
    filepath = Path(filepath)
    
    if filepath.suffix == '.h5':
        df = pd.read_hdf(filepath)
    elif filepath.suffix in ['.csv', '.gz']:
        df = pd.read_csv(filepath, header=[0, 1, 2], index_col=0)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")
    
    # Flatten column names: (scorer, bodypart, coord) -> bodypart_coord
    df.columns = ['_'.join([str(c) for c in col[1:]]) for col in df.columns]
    
    return df


def build_box_model(df: pd.DataFrame, likelihood_thresh: float = 0.99) -> BoxModel:
    """Build fixed box model from high-likelihood frames."""
    good = (
        (df['Reference_likelihood'] > likelihood_thresh) &
        (df['BOXL_likelihood'] > likelihood_thresh) &
        (df['BOXR_likelihood'] > likelihood_thresh)
    )
    
    if good.sum() < 100:
        # Fall back to lower threshold
        good = (
            (df['Reference_likelihood'] > 0.9) &
            (df['BOXL_likelihood'] > 0.9) &
            (df['BOXR_likelihood'] > 0.9)
        )
    
    return BoxModel(
        reference_x=df.loc[good, 'Reference_x'].median(),
        reference_y=df.loc[good, 'Reference_y'].median(),
        boxl_x=df.loc[good, 'BOXL_x'].median(),
        boxl_y=df.loc[good, 'BOXL_y'].median(),
        boxr_x=df.loc[good, 'BOXR_x'].median(),
        boxr_y=df.loc[good, 'BOXR_y'].median(),
    )


def build_sa_model(df: pd.DataFrame, ruler: RulerCalibration, 
                   likelihood_thresh: float = 0.99) -> SAModel:
    """
    Build SA rectangle model - uses ruler for ground truth dimensions.
    
    Width is ALWAYS 1.0 ruler by definition.
    Height is computed from ruler geometry (1.667 ruler) but validated against SAT if available.
    """
    # Get ruler width in pixels
    width_px = ruler.ruler_px
    width_ruler = 1.0  # By definition
    
    # Compute expected height from geometry
    height_ruler = RULER_UNITS['sabr_satr']  # 1.667
    height_px = height_ruler * ruler.ruler_px
    
    # Optionally validate against SAT points if they're reliable
    good_sat = (
        (df['SABL_likelihood'] > likelihood_thresh) &
        (df['SABR_likelihood'] > likelihood_thresh) &
        (df['SATL_likelihood'] > likelihood_thresh) &
        (df['SATR_likelihood'] > likelihood_thresh)
    )
    
    if good_sat.sum() > 100:
        # Check if measured height matches expected
        height_l = (df.loc[good_sat, 'SABL_y'] - df.loc[good_sat, 'SATL_y']).median()
        height_r = (df.loc[good_sat, 'SABR_y'] - df.loc[good_sat, 'SATR_y']).median()
        measured_height = (height_l + height_r) / 2
        measured_height_ruler = measured_height / ruler.ruler_px
        
        # If within 10% of expected, SAT points are probably good
        if 0.9 * height_ruler < measured_height_ruler < 1.1 * height_ruler:
            height_px = measured_height
            height_ruler = measured_height_ruler
    
    return SAModel(
        width_px=width_px,
        height_px=height_px,
        width_ruler=width_ruler,
        height_ruler=height_ruler
    )


def estimate_sa_position(df: pd.DataFrame, ruler: RulerCalibration,
                         likelihood_thresh: float = 0.9) -> np.ndarray:
    """
    Estimate SA position (as SABL_x) from SABL and SABR.
    Uses weighted average with both bottom corners.
    
    SAT points are NOT used for position - they're unreliable.
    """
    n = len(df)
    sa_x = np.full(n, np.nan)
    
    for i in range(n):
        estimates = []
        weights = []
        
        row = df.iloc[i]
        
        # Bottom corners only - these are reliable
        if row['SABL_likelihood'] > likelihood_thresh:
            estimates.append(row['SABL_x'])
            weights.append(100)
        
        if row['SABR_likelihood'] > likelihood_thresh:
            # Convert SABR to SABL position by subtracting ruler width
            estimates.append(row['SABR_x'] - ruler.ruler_px)
            weights.append(100)
        
        if estimates:
            sa_x[i] = np.average(estimates, weights=weights)
    
    # Interpolate gaps
    sa_x = pd.Series(sa_x).interpolate(method='linear').values
    
    return sa_x


def find_weirdness_events(sa_x: np.ndarray, 
                          ruler_px: float,
                          fps: float = 60.0,
                          smoothing_seconds: float = 0.5,
                          velocity_thresh_ruler: float = 0.5,
                          min_interval_seconds: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find "weirdness" events - peaks in rightward velocity.
    These mark pellet transitions.
    
    All thresholds are now RULER-BASED.
    
    Args:
        sa_x: SA position time series (pixels)
        ruler_px: Ruler size in pixels (SABL-SABR distance)
        fps: Frames per second
        smoothing_seconds: Smoothing window in seconds
        velocity_thresh_ruler: Minimum velocity in ruler units per second
        min_interval_seconds: Minimum time between events
    
    Returns:
        peaks: Frame indices of weirdness events
        magnitudes: Magnitude of each event in ruler units
    """
    # Convert thresholds to pixel/frame units
    smoothing_frames = int(smoothing_seconds * fps)
    min_distance_frames = int(min_interval_seconds * fps)
    velocity_thresh_px = velocity_thresh_ruler * ruler_px  # ruler/s -> px/s
    
    # Smooth
    sa_smooth = uniform_filter1d(sa_x, size=max(3, smoothing_frames))
    
    # Velocity (pixels per second)
    velocity_px_per_frame = np.gradient(sa_smooth)
    velocity_px_per_sec = velocity_px_per_frame * fps
    
    # Find peaks in rightward (positive) velocity
    peaks, properties = find_peaks(velocity_px_per_sec, 
                                   height=velocity_thresh_px, 
                                   distance=min_distance_frames)
    
    # Calculate magnitudes in ruler units
    # Magnitude = displacement around the peak
    magnitudes_ruler = []
    for peak in peaks:
        # Look at displacement in window around peak
        window = 30  # frames
        start = max(0, peak - window)
        end = min(len(sa_x) - 1, peak + window)
        displacement_px = sa_x[end] - sa_x[start]
        displacement_ruler = displacement_px / ruler_px
        magnitudes_ruler.append(displacement_ruler)
    
    return peaks, np.array(magnitudes_ruler)


def create_pellet_segments(weirdness_frames: np.ndarray, 
                           total_frames: int,
                           fps: float = 60.0) -> List[PelletSegment]:
    """
    Create pellet segments from weirdness events.
    
    Pattern: entry + 19 transitions + exit = 21 events = 20 pellets
    But some videos may have fewer events if entry/exit not captured.
    """
    n_events = len(weirdness_frames)
    
    # Determine number of pellets
    if 19 <= n_events <= 21:
        n_pellets = 20
    else:
        n_pellets = max(1, n_events)  # Best guess
    
    segments = []
    
    for i in range(min(n_pellets, n_events)):
        if i == 0:
            start = 0
        else:
            start = weirdness_frames[i - 1]
        
        if i < n_events - 1:
            end = weirdness_frames[i] - 1
        else:
            end = total_frames - 1
        
        duration_frames = end - start + 1
        
        segments.append(PelletSegment(
            pellet_num=i + 1,
            start_frame=int(start),
            end_frame=int(end),
            duration_frames=int(duration_frames),
            duration_seconds=duration_frames / fps
        ))
    
    return segments


def segment_video(filepath: Union[str, Path], 
                  fps: float = 60.0,
                  output_dir: Optional[Path] = None) -> VideoSegmentation:
    """
    Main function: segment a video into pellet periods.
    
    Uses RULER-BASED calibration and thresholds.
    
    Args:
        filepath: Path to DLC output file (.csv or .h5)
        fps: Video frame rate (default 60)
        output_dir: Where to save results (default: same as input)
    
    Returns:
        VideoSegmentation object with all results
    """
    filepath = Path(filepath)
    
    print(f"Processing: {filepath.name}")
    
    # Load data
    df = load_dlc_data(filepath)
    total_frames = len(df)
    print(f"  Loaded {total_frames} frames ({total_frames/fps:.1f}s)")
    
    # CALIBRATE RULER FIRST - this is the foundation
    ruler = RulerCalibration.from_dataframe(df)
    print(f"  Ruler: {ruler.ruler_px:.1f}px = 9mm ({ruler.mm_per_px:.3f} mm/px)")
    print(f"  Ruler quality: {ruler.quality} ({ruler.trusted_frame_count} trusted frames)")
    if ruler.flags:
        print(f"  Ruler flags: {ruler.flags}")
    
    # Build models using ruler
    box_model = build_box_model(df)
    sa_model = build_sa_model(df, ruler)
    print(f"  SA model: {sa_model.width_px:.1f}px × {sa_model.height_px:.1f}px")
    print(f"            ({sa_model.width_ruler:.2f} × {sa_model.height_ruler:.2f} ruler)")
    
    # Estimate SA position
    sa_x = estimate_sa_position(df, ruler)
    
    # Find weirdness events with ruler-based thresholds
    weirdness, magnitudes = find_weirdness_events(
        sa_x, 
        ruler_px=ruler.ruler_px,
        fps=fps
    )
    print(f"  Found {len(weirdness)} weirdness events")
    if len(magnitudes) > 0:
        print(f"  Snap magnitudes: {np.median(magnitudes):.2f} ruler (median)")
    
    # Create segments
    segments = create_pellet_segments(weirdness, total_frames, fps)
    print(f"  Created {len(segments)} pellet segments")
    
    # Build result
    result = VideoSegmentation(
        video_name=filepath.stem,
        total_frames=total_frames,
        fps=fps,
        ruler_calibration=ruler.to_dict(),
        box_model=box_model,
        sa_model=sa_model,
        pellet_segments=segments,
        weirdness_frames=[int(w) for w in weirdness],
        weirdness_magnitudes_ruler=[float(m) for m in magnitudes],
        n_pellets=len(segments)
    )
    
    # Save results
    if output_dir is None:
        output_dir = filepath.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{filepath.stem}_segmentation.json"
    save_results(result, output_file)
    print(f"  Saved: {output_file}")
    
    return result


def save_results(result: VideoSegmentation, filepath: Path):
    """Save segmentation results to JSON."""
    data = {
        'video_name': result.video_name,
        'total_frames': result.total_frames,
        'fps': result.fps,
        'n_pellets': result.n_pellets,
        'ruler_calibration': result.ruler_calibration,
        'box_model': asdict(result.box_model),
        'sa_model': asdict(result.sa_model),
        'weirdness_frames': result.weirdness_frames,
        'weirdness_magnitudes_ruler': result.weirdness_magnitudes_ruler,
        'pellet_segments': [asdict(s) for s in result.pellet_segments]
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_results(filepath: Path) -> VideoSegmentation:
    """Load segmentation results from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return VideoSegmentation(
        video_name=data['video_name'],
        total_frames=data['total_frames'],
        fps=data['fps'],
        box_model=BoxModel(**data['box_model']),
        sa_model=SAModel(**data['sa_model']),
        pellet_segments=[PelletSegment(**s) for s in data['pellet_segments']],
        weirdness_frames=data['weirdness_frames'],
        n_pellets=data['n_pellets']
    )


def select_file_dialog() -> Optional[Path]:
    """Open file dialog to select DLC output file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        
        filepath = filedialog.askopenfilename(
            title="Select DLC Output File",
            filetypes=[
                ("DLC files", "*.csv *.csv.gz *.h5"),
                ("CSV files", "*.csv *.csv.gz"),
                ("H5 files", "*.h5"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        
        return Path(filepath) if filepath else None
        
    except ImportError:
        print("tkinter not available. Please provide filepath as argument.")
        return None


def select_folder_dialog() -> Optional[Path]:
    """Open folder dialog to select directory with DLC files."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        
        folder = filedialog.askdirectory(title="Select Folder with DLC Files")
        
        root.destroy()
        
        return Path(folder) if folder else None
        
    except ImportError:
        print("tkinter not available. Please provide folder path as argument.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Segment DLC tracking data into pellet periods"
    )
    parser.add_argument(
        'path', 
        nargs='?', 
        help="Path to DLC file or folder (opens dialog if not provided)"
    )
    parser.add_argument(
        '--fps', 
        type=float, 
        default=60.0,
        help="Video frame rate (default: 60)"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help="Process all CSV/H5 files in folder"
    )
    
    args = parser.parse_args()
    
    # Get input path
    if args.path:
        input_path = Path(args.path)
    elif args.batch:
        input_path = select_folder_dialog()
    else:
        input_path = select_file_dialog()
    
    if input_path is None:
        print("No file selected. Exiting.")
        return
    
    output_dir = Path(args.output) if args.output else None
    
    # Process
    if input_path.is_dir() or args.batch:
        # Batch mode
        files = list(input_path.glob("*.csv")) + \
                list(input_path.glob("*.csv.gz")) + \
                list(input_path.glob("*.h5"))
        
        print(f"Found {len(files)} files to process")
        
        for f in files:
            try:
                segment_video(f, fps=args.fps, output_dir=output_dir)
            except Exception as e:
                print(f"  ERROR processing {f.name}: {e}")
    else:
        # Single file
        segment_video(input_path, fps=args.fps, output_dir=output_dir)
    
    print("Done!")


if __name__ == "__main__":
    main()
