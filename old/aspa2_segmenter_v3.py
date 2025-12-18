"""
ASPA2 Segmenter v3 - Interval Detection + Snap Validation
==========================================================

GUARANTEE: Every video gets exactly 22 segments.
- Segment 0: Pre-pellet garbage (tray not yet in position)
- Segments 1-20: Pellet periods (SA in active zone)
- Segment 21: Post-pellet garbage (tray has left)

ALGORITHM:
1. Detect ANY confident leftward movements (conveyor moving tray)
2. Find fundamental interval via GCD of pairwise distances
3. Calculate where all 21 movements SHOULD be
4. Find rightward snaps - these are the FIRM boundaries
5. Validate snaps against expected positions
6. Use snaps as segment cut points

The conveyor program is deterministic - timing between movements is constant.
If we detect movements at frames 2000, 3800, 9200:
  - 2000→3800 = 1800, 3800→9200 = 5400, 2000→9200 = 7200
  - GCD-like clustering finds fundamental interval = 1800
  - We can calculate all 21 movement times from there
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict
import json
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, savgol_filter
from collections import Counter

# Import ruler system
try:
    from aspa2_ruler import RulerCalibration, Geometry, RULER_UNITS, THRESHOLDS
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("aspa2_ruler", 
        Path(__file__).parent / "aspa2_ruler.py")
    ruler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ruler_module)
    RulerCalibration = ruler_module.RulerCalibration
    Geometry = ruler_module.Geometry
    RULER_UNITS = ruler_module.RULER_UNITS
    THRESHOLDS = ruler_module.THRESHOLDS


# =============================================================================
# CONSTANTS (Physical truths about the hardware)
# =============================================================================

N_PELLETS = 20    # Physical: tray has 20 pellet positions
N_MOVEMENTS = 21  # Entry + 19 transitions + exit
N_SEGMENTS = 22   # garbage_pre + 20 pellets + garbage_post

# No timing constants - interval is INFERRED from video length / N_SEGMENTS
INTERVAL_TOLERANCE = 0.3  # Allow 30% deviation when clustering


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Boundary:
    """A segment boundary (where one segment ends and next begins)."""
    frame: int
    confidence: float           # 0-1
    method: str                 # 'snap_detected', 'calculated', 'interpolated'
    snap_magnitude_ruler: Optional[float] = None
    expected_frame: Optional[int] = None  # Where we calculated it should be
    deviation_frames: Optional[int] = None  # How far from expected


@dataclass
class Segment:
    """One segment of the video."""
    segment_num: int            # 0-21
    segment_type: str           # 'garbage_pre', 'pellet', 'garbage_post'
    pellet_num: Optional[int]   # 1-20 for pellet segments, None otherwise
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_seconds: float
    confidence: float
    flags: List[str] = field(default_factory=list)


@dataclass 
class SegmentationResult:
    """Complete segmentation for a video."""
    video_name: str
    total_frames: int
    fps: float
    ruler_calibration: dict
    fundamental_interval_frames: float
    fundamental_interval_seconds: float
    segments: List[Segment]
    boundaries: List[Boundary]
    overall_confidence: float
    flags: List[str] = field(default_factory=list)
    
    # Detection stats
    n_leftward_movements_detected: int = 0
    n_rightward_snaps_detected: int = 0
    n_boundaries_from_snaps: int = 0
    n_boundaries_interpolated: int = 0


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dlc_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load DLC output file (.csv or .h5)."""
    filepath = Path(filepath)
    
    if filepath.suffix == '.h5':
        df = pd.read_hdf(filepath)
    elif filepath.suffix in ['.csv', '.gz']:
        df = pd.read_csv(filepath, header=[0, 1, 2], index_col=0)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")
    
    df.columns = ['_'.join([str(c) for c in col[1:]]) for col in df.columns]
    return df


# =============================================================================
# POSITION AND VELOCITY
# =============================================================================

def get_sa_position(df: pd.DataFrame, ruler_px: float,
                    likelihood_thresh: float = 0.5) -> np.ndarray:
    """Get SA position time series (as SABL_x equivalent)."""
    n = len(df)
    sa_x = np.full(n, np.nan)
    
    for i in range(n):
        row = df.iloc[i]
        estimates = []
        
        if row['SABL_likelihood'] > likelihood_thresh:
            estimates.append(row['SABL_x'])
        if row['SABR_likelihood'] > likelihood_thresh:
            estimates.append(row['SABR_x'] - ruler_px)
        
        if estimates:
            sa_x[i] = np.mean(estimates)
    
    # Interpolate and fill
    sa_series = pd.Series(sa_x)
    sa_x = sa_series.interpolate(method='linear', limit_direction='both').values
    if np.any(np.isnan(sa_x)):
        sa_x = np.nan_to_num(sa_x, nan=np.nanmedian(sa_x))
    
    return sa_x


def get_velocity(sa_x: np.ndarray, fps: float, 
                 smooth_seconds: float = 0.3) -> np.ndarray:
    """Get smoothed velocity. Negative = leftward, Positive = rightward."""
    smooth_frames = max(3, int(smooth_seconds * fps))
    
    if len(sa_x) > smooth_frames * 2:
        sa_smooth = savgol_filter(sa_x, smooth_frames, 2)
    else:
        sa_smooth = uniform_filter1d(sa_x, size=3)
    
    velocity = np.gradient(sa_smooth) * fps  # pixels per second
    return velocity


# =============================================================================
# MOVEMENT DETECTION
# =============================================================================

def detect_leftward_movements(velocity: np.ndarray, ruler_px: float, 
                               fps: float) -> List[Tuple[int, float]]:
    """
    Detect leftward movements (conveyor moving tray left).
    Returns list of (frame, duration_seconds).
    
    Leftward movement = sustained negative velocity.
    """
    # Threshold: moving left at >0.1 ruler/second
    threshold = -0.1 * ruler_px
    
    moving_left = velocity < threshold
    
    # Find contiguous regions of leftward movement
    movements = []
    in_movement = False
    start = 0
    
    for i, is_moving in enumerate(moving_left):
        if is_moving and not in_movement:
            in_movement = True
            start = i
        elif not is_moving and in_movement:
            in_movement = False
            duration = i - start
            if duration > fps * 0.5:  # At least 0.5 seconds
                mid_frame = start + duration // 2
                movements.append((mid_frame, duration / fps))
    
    # Handle movement at end
    if in_movement:
        duration = len(velocity) - start
        if duration > fps * 0.5:
            mid_frame = start + duration // 2
            movements.append((mid_frame, duration / fps))
    
    return movements


def detect_rightward_snaps(velocity: np.ndarray, ruler_px: float,
                           fps: float) -> List[Tuple[int, float]]:
    """
    Detect rightward snaps (tray settling after conveyor stops).
    Returns list of (frame, magnitude_ruler).
    
    These are brief, sharp rightward movements - the FIRM boundaries.
    """
    # Look for peaks in rightward (positive) velocity
    # Snap threshold: >0.5 ruler/second peak velocity
    threshold = 0.5 * ruler_px
    
    peaks, properties = find_peaks(velocity, 
                                   height=threshold,
                                   distance=int(5 * fps))  # At least 5s apart
    
    snaps = []
    for peak in peaks:
        # Estimate snap magnitude
        window = int(fps * 0.5)  # 0.5 second window
        start = max(0, peak - window)
        end = min(len(velocity), peak + window)
        
        # Integrate positive velocity to get rightward displacement
        v_window = velocity[start:end]
        displacement = np.sum(v_window[v_window > 0]) / fps
        magnitude_ruler = displacement / ruler_px
        
        if magnitude_ruler > 0.3:  # At least 0.3 ruler units
            snaps.append((int(peak), float(magnitude_ruler)))
    
    return snaps


# =============================================================================
# INTERVAL DETECTION (THE KEY INSIGHT)
# =============================================================================

def find_fundamental_interval(detections: List[int], 
                              expected_interval: float,
                              tolerance: float = 0.3) -> Optional[float]:
    """
    Find the fundamental interval from detected movement times.
    
    If we detect movements at frames [2000, 3800, 9200]:
    - Pairwise intervals: 1800, 5400, 7200
    - These should all be multiples of the fundamental interval
    - Cluster to find it
    
    Args:
        detections: List of frame numbers where movements detected
        expected_interval: Expected interval in frames (~1800 at 60fps)
        tolerance: How much deviation to allow when clustering
    
    Returns:
        Fundamental interval in frames, or None if can't determine
    """
    if len(detections) < 2:
        return None
    
    detections = sorted(detections)
    
    # Calculate all pairwise intervals
    intervals = []
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            intervals.append(detections[j] - detections[i])
    
    if not intervals:
        return None
    
    # For each interval, check what fundamental it implies
    # interval / n should give us the fundamental for various n
    candidates = []
    
    for interval in intervals:
        # Try dividing by 1, 2, 3, ... up to reasonable limit
        for n in range(1, 22):  # Can't have more than 21 intervals
            candidate = interval / n
            
            # Is this close to expected?
            if (1 - tolerance) * expected_interval <= candidate <= (1 + tolerance) * expected_interval:
                candidates.append(candidate)
    
    if not candidates:
        # No candidates near expected - return median of raw intervals
        # divided by likely number of steps
        min_interval = min(intervals)
        return min_interval  # Assume smallest is 1 step
    
    # Cluster candidates and find most common
    # Round to nearest 10 frames for clustering
    rounded = [round(c / 10) * 10 for c in candidates]
    counts = Counter(rounded)
    
    if counts:
        most_common = counts.most_common(1)[0][0]
        # Return average of candidates that round to most common
        matching = [c for c in candidates if round(c / 10) * 10 == most_common]
        return np.mean(matching)
    
    return np.median(candidates)


def calculate_expected_boundaries(fundamental_interval: float,
                                  total_frames: int,
                                  anchor_frame: Optional[int] = None,
                                  anchor_index: Optional[int] = None) -> List[int]:
    """
    Calculate where all 21 boundaries should be.
    
    Args:
        fundamental_interval: Frames between movements
        total_frames: Total frames in video
        anchor_frame: A known boundary frame (optional)
        anchor_index: Which boundary (1-21) the anchor is (optional)
    
    Returns:
        List of 21 expected boundary frames
    """
    if anchor_frame is not None and anchor_index is not None:
        # Calculate from anchor
        first_boundary = anchor_frame - (anchor_index - 1) * fundamental_interval
    else:
        # Estimate: first boundary is roughly 1 interval from start
        # (garbage segment 0 is typically short)
        first_boundary = fundamental_interval * 0.5
    
    boundaries = []
    for i in range(N_MOVEMENTS):
        frame = first_boundary + i * fundamental_interval
        boundaries.append(int(round(frame)))
    
    # Clamp to valid range
    boundaries = [max(1, min(total_frames - 2, b)) for b in boundaries]
    
    return boundaries


# =============================================================================
# BOUNDARY MATCHING AND VALIDATION
# =============================================================================

def match_snaps_to_expected(snaps: List[Tuple[int, float]],
                            expected: List[int],
                            tolerance_frames: int) -> List[Boundary]:
    """
    Match detected snaps to expected boundary positions.
    
    Returns list of 21 Boundary objects.
    """
    boundaries = []
    used_snaps = set()
    
    for i, exp_frame in enumerate(expected):
        # Find closest snap within tolerance
        best_snap = None
        best_distance = tolerance_frames + 1
        
        for j, (snap_frame, magnitude) in enumerate(snaps):
            if j in used_snaps:
                continue
            distance = abs(snap_frame - exp_frame)
            if distance < best_distance:
                best_distance = distance
                best_snap = (j, snap_frame, magnitude)
        
        if best_snap is not None and best_distance <= tolerance_frames:
            j, snap_frame, magnitude = best_snap
            used_snaps.add(j)
            
            boundaries.append(Boundary(
                frame=snap_frame,
                confidence=max(0.5, 1.0 - best_distance / tolerance_frames),
                method='snap_detected',
                snap_magnitude_ruler=magnitude,
                expected_frame=exp_frame,
                deviation_frames=snap_frame - exp_frame
            ))
        else:
            # No snap found - use expected position
            boundaries.append(Boundary(
                frame=exp_frame,
                confidence=0.3,
                method='interpolated',
                snap_magnitude_ruler=None,
                expected_frame=exp_frame,
                deviation_frames=0
            ))
    
    return boundaries


# =============================================================================
# SEGMENT CREATION
# =============================================================================

def create_segments(boundaries: List[Boundary],
                    total_frames: int,
                    fps: float,
                    fundamental_interval_frames: float) -> List[Segment]:
    """Create 22 segments from 21 boundaries."""
    segments = []
    
    expected_duration_s = fundamental_interval_frames / fps
    
    for i in range(N_SEGMENTS):
        # Segment type
        if i == 0:
            seg_type = 'garbage_pre'
            pellet_num = None
        elif i == N_SEGMENTS - 1:
            seg_type = 'garbage_post'
            pellet_num = None
        else:
            seg_type = 'pellet'
            pellet_num = i  # 1-20
        
        # Frame range
        start = 0 if i == 0 else boundaries[i - 1].frame
        end = total_frames - 1 if i == N_SEGMENTS - 1 else boundaries[i].frame - 1
        
        duration_frames = max(1, end - start + 1)
        duration_seconds = duration_frames / fps
        
        # Confidence from bounding boundaries
        if i == 0:
            conf = boundaries[0].confidence
        elif i == N_SEGMENTS - 1:
            conf = boundaries[-1].confidence
        else:
            conf = min(boundaries[i - 1].confidence, boundaries[i].confidence)
        
        # Flags - compare to THIS video's interval, not a hardcoded value
        flags = []
        if seg_type == 'pellet':
            if duration_seconds < expected_duration_s * 0.5:
                flags.append('SHORT')
            elif duration_seconds > expected_duration_s * 1.5:
                flags.append('LONG')
        
        segments.append(Segment(
            segment_num=i,
            segment_type=seg_type,
            pellet_num=pellet_num,
            start_frame=int(start),
            end_frame=int(end),
            duration_frames=int(duration_frames),
            duration_seconds=float(duration_seconds),
            confidence=float(conf),
            flags=flags
        ))
    
    return segments


# =============================================================================
# MAIN SEGMENTATION
# =============================================================================

def segment_video(dlc_path: Union[str, Path],
                  fps: float = 60.0,
                  output_path: Optional[Path] = None) -> SegmentationResult:
    """
    Segment video into exactly 22 segments.
    
    ALWAYS returns 22 segments. Confidence scores indicate reliability.
    """
    dlc_path = Path(dlc_path)
    print(f"\nSegmenting: {dlc_path.name}")
    
    # Load
    df = load_dlc_data(dlc_path)
    total_frames = len(df)
    print(f"  Frames: {total_frames} ({total_frames/fps:.1f}s)")
    
    # Ruler calibration
    ruler = RulerCalibration.from_dataframe(df)
    print(f"  Ruler: {ruler.ruler_px:.1f}px ({ruler.quality})")
    
    flags = list(ruler.flags)
    
    # Position and velocity
    sa_x = get_sa_position(df, ruler.ruler_px)
    velocity = get_velocity(sa_x, fps)
    
    # Detect movements (leftward = conveyor moving)
    leftward = detect_leftward_movements(velocity, ruler.ruler_px, fps)
    print(f"  Leftward movements detected: {len(leftward)}")
    
    # Detect snaps (rightward = tray settling, THE BOUNDARIES)
    snaps = detect_rightward_snaps(velocity, ruler.ruler_px, fps)
    print(f"  Rightward snaps detected: {len(snaps)}")
    
    # Find fundamental interval
    # INFER expected interval from video length - no hardcoded timing
    expected_interval = total_frames / N_SEGMENTS
    
    # Use both leftward movements and snaps as timing evidence
    all_events = [m[0] for m in leftward] + [s[0] for s in snaps]
    
    if len(all_events) >= 2:
        interval = find_fundamental_interval(all_events, expected_interval)
    else:
        interval = None
    
    if interval is None:
        interval = expected_interval
        flags.append('INTERVAL_ASSUMED')
        print(f"  Interval: {interval:.0f} frames (ASSUMED from video length)")
    else:
        print(f"  Interval: {interval:.0f} frames ({interval/fps:.1f}s)")
    
    # Calculate expected boundaries
    # Use first snap as anchor if available
    if snaps:
        anchor_frame = snaps[0][0]
        # Estimate which boundary this is based on position in video
        anchor_index = max(1, min(21, round(anchor_frame / interval)))
        expected = calculate_expected_boundaries(interval, total_frames, 
                                                  anchor_frame, anchor_index)
    else:
        expected = calculate_expected_boundaries(interval, total_frames)
    
    # Match snaps to expected positions
    tolerance = int(interval * 0.3)  # 30% tolerance
    boundaries = match_snaps_to_expected(snaps, expected, tolerance)
    
    # Stats
    n_from_snaps = sum(1 for b in boundaries if b.method == 'snap_detected')
    n_interpolated = sum(1 for b in boundaries if b.method == 'interpolated')
    
    print(f"  Boundaries: {n_from_snaps} from snaps, {n_interpolated} interpolated")
    
    if n_from_snaps < 10:
        flags.append('LOW_SNAP_DETECTION')
    
    # Create segments
    segments = create_segments(boundaries, total_frames, fps, interval)
    
    # Check segment durations
    pellet_durations = [s.duration_seconds for s in segments if s.segment_type == 'pellet']
    mean_dur = np.mean(pellet_durations)
    std_dur = np.std(pellet_durations)
    cv = std_dur / mean_dur if mean_dur > 0 else 0
    
    print(f"  Pellet durations: {mean_dur:.1f}s ± {std_dur:.1f}s (CV={cv:.2f})")
    
    if cv > 0.15:
        flags.append('HIGH_DURATION_VARIANCE')
    
    # Overall confidence
    overall_conf = np.mean([b.confidence for b in boundaries])
    print(f"  Overall confidence: {overall_conf:.2f}")
    
    if flags:
        print(f"  Flags: {flags}")
    
    # Result
    result = SegmentationResult(
        video_name=dlc_path.stem,
        total_frames=total_frames,
        fps=fps,
        ruler_calibration=ruler.to_dict(),
        fundamental_interval_frames=float(interval),
        fundamental_interval_seconds=float(interval / fps),
        segments=segments,
        boundaries=boundaries,
        overall_confidence=float(overall_conf),
        flags=flags,
        n_leftward_movements_detected=len(leftward),
        n_rightward_snaps_detected=len(snaps),
        n_boundaries_from_snaps=n_from_snaps,
        n_boundaries_interpolated=n_interpolated
    )
    
    # Save
    if output_path is None:
        output_path = dlc_path.parent / f"{dlc_path.stem}_segmentation.json"
    
    save_segmentation(result, output_path)
    print(f"  Saved: {output_path}")
    
    return result


# =============================================================================
# SAVING
# =============================================================================

def save_segmentation(result: SegmentationResult, filepath: Path):
    """Save segmentation to JSON."""
    
    def to_serializable(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: to_serializable(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    data = to_serializable(result)
    
    # Add summary at top level
    data['summary'] = {
        'n_segments': len(result.segments),
        'n_pellet_segments': sum(1 for s in result.segments if s.segment_type == 'pellet'),
        'interval_seconds': result.fundamental_interval_seconds,
        'snap_detection_rate': result.n_boundaries_from_snaps / N_MOVEMENTS,
        'confidence': result.overall_confidence
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ASPA2 Video Segmenter v3')
    parser.add_argument('input', nargs='?', help='DLC file or folder')
    parser.add_argument('--fps', type=float, default=60.0)
    args = parser.parse_args()
    
    if args.input is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            args.input = filedialog.askopenfilename(
                title='Select DLC output file',
                filetypes=[('DLC files', '*.csv *.h5'), ('All', '*.*')]
            )
            if not args.input:
                print("No file selected")
                return
        except ImportError:
            print("Usage: python aspa2_segmenter_v3.py <dlc_file.csv>")
            return
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        files = list(input_path.glob('*DLC*.csv')) + list(input_path.glob('*DLC*.h5'))
        print(f"Found {len(files)} DLC files")
        for f in sorted(files):
            try:
                segment_video(f, fps=args.fps)
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        segment_video(input_path, fps=args.fps)


if __name__ == '__main__':
    main()
