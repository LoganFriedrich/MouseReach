"""
ASPA2 Per-Pellet Model Builder
==============================

PURPOSE:
    Builds refined rectangle + pellet position models for each pellet segment.
    Uses the "still period" within each pellet (when SA is not moving) to get
    the most accurate model of where the rectangle and pellet ARE for that pellet.

    All positions are RULER-BASED: 
    - Absolute positions in pixels
    - Relative positions in ruler units (1 ruler = SABL-SABR = 9mm)
    - Expected positions computed from geometry

INPUT:
    - DLC output file (.csv or .h5)
    - Segmentation JSON from aspa2_pellet_segmenter.py (includes ruler calibration)

OUTPUT:
    - JSON file with per-pellet models
    - Each pellet gets: 
        - Measured positions (SABL, SABR, pellet, pillar)
        - Computed expected positions (from geometry)
        - Deviations (measured vs expected)

USAGE:
    Command line:
        python aspa2_pellet_model_builder.py                    # Opens file dialog
        python aspa2_pellet_model_builder.py dlc_file.csv segmentation.json
    
    As module:
        from aspa2_pellet_model_builder import build_pellet_models
        models = build_pellet_models(dlc_path, segmentation_path)

ALGORITHM:
    For each pellet segment:
    1. Calculate velocity within the segment
    2. Find "still" frames where |velocity| < threshold (ruler-based)
    3. Find the longest contiguous still period
    4. From high-likelihood frames in still period:
       - Anchor: SABL, SABR median positions
       - Computed pillar/pellet: from geometry
       - Measured pellet: DLC pellet point position
       - Deviation: measured - computed (in ruler units)

WHY THIS MATTERS:
    The pellet position during the still period is the "ground truth" for where
    the pellet started. Later analysis compares hand/reach positions to this
    to determine if pellet was taken, displaced, or missed.

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


@dataclass
class RectangleModel:
    """Full rectangle model for one pellet's still period."""
    # Anchor positions (absolute pixels) - MEASURED
    sabl_x: float
    sabl_y: float
    sabr_x: float
    sabr_y: float
    # Top corners (computed from geometry, NOT from DLC SAT)
    satl_x: float
    satl_y: float
    satr_x: float
    satr_y: float
    # Derived dimensions
    width_px: float
    height_px: float
    width_ruler: float  # Should be 1.0
    height_ruler: float  # Should be ~1.667


@dataclass
class PelletModel:
    """Pellet/pillar position model for one pellet."""
    pellet_num: int
    # Frame range for this pellet
    start_frame: int
    end_frame: int
    # Still period (best frames for model)
    still_start: int
    still_end: int
    still_frames: int
    # Rectangle model
    rectangle: RectangleModel
    
    # COMPUTED pillar position (from geometry) - this is ground truth
    computed_pillar_x: float
    computed_pillar_y: float
    computed_pillar_ruler: Tuple[float, float]  # In ruler coords (should be 0.5, 0.944)
    
    # MEASURED pellet position (from DLC)
    pellet_x: Optional[float]
    pellet_y: Optional[float]
    pellet_ruler: Optional[Tuple[float, float]]  # In ruler coords
    pellet_likelihood: float
    
    # Pellet deviation from expected (measured - computed, in ruler units)
    pellet_deviation_ruler: Optional[float]  # Distance from expected position
    
    # MEASURED pillar position (from DLC) - for validation
    pillar_x: Optional[float]
    pillar_y: Optional[float]
    pillar_ruler: Optional[Tuple[float, float]]
    pillar_likelihood: float
    pillar_deviation_ruler: Optional[float]  # How far off is DLC pillar from computed?
    
    # Quality flags
    model_quality: str  # 'good', 'fair', 'poor'
    n_high_likelihood_frames: int


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


def load_segmentation(filepath: Union[str, Path]) -> dict:
    """Load segmentation results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def find_longest_contiguous_run(indices: List[int]) -> List[int]:
    """Find the longest contiguous run in a list of indices."""
    if not indices:
        return []
    
    runs = []
    current = [indices[0]]
    
    for idx in indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            runs.append(current)
            current = [idx]
    runs.append(current)
    
    return max(runs, key=len) if runs else []


def estimate_sa_position(df: pd.DataFrame, ruler_px: float,
                         likelihood_thresh: float = 0.9) -> np.ndarray:
    """Estimate SA position (as SABL_x) using ruler for SABR offset."""
    n = len(df)
    sa_x = np.full(n, np.nan)
    
    for i in range(n):
        estimates = []
        weights = []
        row = df.iloc[i]
        
        if row['SABL_likelihood'] > likelihood_thresh:
            estimates.append(row['SABL_x'])
            weights.append(100)
        if row['SABR_likelihood'] > likelihood_thresh:
            estimates.append(row['SABR_x'] - ruler_px)
            weights.append(100)
        
        if estimates:
            sa_x[i] = np.average(estimates, weights=weights)
    
    return pd.Series(sa_x).interpolate(method='linear').values


def find_still_period(df: pd.DataFrame, start_frame: int, end_frame: int,
                      ruler_px: float, 
                      velocity_thresh_ruler: float = 0.03,
                      smoothing_seconds: float = 0.5,
                      fps: float = 60.0) -> Tuple[int, int, int]:
    """
    Find the longest still period within a pellet segment.
    
    Uses RULER-BASED velocity threshold.
    
    Args:
        ruler_px: SABL-SABR distance in pixels
        velocity_thresh_ruler: Max velocity in ruler units per second (default 0.03 = ~0.3mm/s)
    
    Returns:
        (still_start, still_end, n_still_frames)
    """
    segment = df.iloc[start_frame:end_frame + 1]
    smoothing_frames = int(smoothing_seconds * fps)
    
    # Estimate SA position and velocity
    sa_x = estimate_sa_position(segment, ruler_px)
    sa_smooth = uniform_filter1d(sa_x, size=min(smoothing_frames, len(sa_x)))
    
    # Velocity in ruler units per second
    velocity_px_per_frame = np.gradient(sa_smooth)
    velocity_ruler_per_sec = (velocity_px_per_frame * fps) / ruler_px
    
    # Find still frames
    still_mask = np.abs(velocity_ruler_per_sec) < velocity_thresh_ruler
    still_indices = segment.index[still_mask].tolist()
    
    if not still_indices:
        # No still frames - use middle portion as fallback
        mid = (start_frame + end_frame) // 2
        quarter = (end_frame - start_frame) // 4
        return mid - quarter, mid + quarter, quarter * 2
    
    # Find longest contiguous still period
    longest = find_longest_contiguous_run(still_indices)
    
    if len(longest) < 10:
        # Very short - use what we have
        return still_indices[0], still_indices[-1], len(still_indices)
    
    return longest[0], longest[-1], len(longest)


def build_rectangle_model(df: pd.DataFrame, still_start: int, still_end: int,
                          ruler_px: float,
                          likelihood_thresh: float = 0.95) -> Tuple[Optional[RectangleModel], int]:
    """
    Build rectangle model from high-likelihood frames in still period.
    
    CRITICAL: SAT corners are COMPUTED from geometry, not measured.
    Only SABL/SABR are used as anchors.
    
    Returns:
        (RectangleModel or None, n_high_likelihood_frames)
    """
    still_df = df.loc[still_start:still_end]
    
    # Filter to high-likelihood SABL/SABR frames only
    good = (
        (still_df['SABL_likelihood'] > likelihood_thresh) &
        (still_df['SABR_likelihood'] > likelihood_thresh)
    )
    
    n_good = good.sum()
    
    if n_good < 10:
        # Try lower threshold
        good = (
            (still_df['SABL_likelihood'] > 0.8) &
            (still_df['SABR_likelihood'] > 0.8)
        )
        n_good = good.sum()
    
    if n_good < 5:
        return None, n_good
    
    model_df = still_df[good]
    
    # Get median anchor positions (SABL, SABR only)
    sabl = np.array([model_df['SABL_x'].median(), model_df['SABL_y'].median()])
    sabr = np.array([model_df['SABR_x'].median(), model_df['SABR_y'].median()])
    
    # Compute SAT positions from geometry (NOT from DLC!)
    rect_corners = Geometry.compute_sa_rectangle(sabl, sabr)
    # rect_corners order: [SABL, SABR, SATR, SATL]
    satl = rect_corners[3]
    satr = rect_corners[2]
    
    # Dimensions
    width_px = Geometry.distance(sabl, sabr)
    height_px = Geometry.distance(sabl, satl)
    width_ruler = width_px / ruler_px
    height_ruler = height_px / ruler_px
    
    return RectangleModel(
        sabl_x=float(sabl[0]), sabl_y=float(sabl[1]),
        sabr_x=float(sabr[0]), sabr_y=float(sabr[1]),
        satl_x=float(satl[0]), satl_y=float(satl[1]),
        satr_x=float(satr[0]), satr_y=float(satr[1]),
        width_px=width_px, height_px=height_px,
        width_ruler=width_ruler, height_ruler=height_ruler
    ), n_good


def get_pellet_position(df: pd.DataFrame, still_start: int, still_end: int,
                        sabl: np.ndarray, sabr: np.ndarray,
                        likelihood_thresh: float = 0.8) -> Dict:
    """
    Get pellet position from still period.
    
    Returns dict with:
        pellet_x, pellet_y: absolute pixel position
        pellet_ruler: (x, y) in ruler coordinates
        likelihood: mean likelihood
        deviation_ruler: distance from expected pillar position (ruler units)
    """
    still_df = df.loc[still_start:still_end]
    good = still_df['Pellet_likelihood'] > likelihood_thresh
    
    result = {
        'pellet_x': None,
        'pellet_y': None,
        'pellet_ruler': None,
        'likelihood': still_df['Pellet_likelihood'].mean() if len(still_df) > 0 else 0.0,
        'deviation_ruler': None
    }
    
    if good.sum() < 5:
        return result
    
    model_df = still_df[good]
    
    pellet_x = model_df['Pellet_x'].median()
    pellet_y = model_df['Pellet_y'].median()
    pellet_pos = np.array([pellet_x, pellet_y])
    
    # Convert to ruler coordinates
    pellet_ruler = Geometry.to_ruler_coords(pellet_pos, sabl, sabr)
    
    # Compute expected pillar position (where pellet should be)
    expected_pillar = Geometry.compute_pillar_center(sabl, sabr)
    
    # Deviation from expected position (in ruler units)
    ruler_px = Geometry.distance(sabl, sabr)
    deviation_px = Geometry.distance(pellet_pos, expected_pillar)
    deviation_ruler = deviation_px / ruler_px
    
    result.update({
        'pellet_x': float(pellet_x),
        'pellet_y': float(pellet_y),
        'pellet_ruler': pellet_ruler,
        'likelihood': float(model_df['Pellet_likelihood'].mean()),
        'deviation_ruler': float(deviation_ruler)
    })
    
    return result


def get_pillar_position(df: pd.DataFrame, still_start: int, still_end: int,
                        sabl: np.ndarray, sabr: np.ndarray,
                        likelihood_thresh: float = 0.8) -> Dict:
    """
    Get DLC pillar position from still period (for validation only).
    
    The COMPUTED pillar position (from geometry) is the ground truth.
    This measured position is compared to validate DLC accuracy.
    
    Returns dict with:
        pillar_x, pillar_y: measured pixel position
        pillar_ruler: (x, y) in ruler coordinates  
        likelihood: mean likelihood
        deviation_ruler: distance from computed position (ruler units)
    """
    still_df = df.loc[still_start:still_end]
    
    result = {
        'pillar_x': None,
        'pillar_y': None,
        'pillar_ruler': None,
        'likelihood': 0.0,
        'deviation_ruler': None
    }
    
    # Check if Pillar columns exist
    if 'Pillar_x' not in df.columns:
        return result
    
    result['likelihood'] = still_df['Pillar_likelihood'].mean() if len(still_df) > 0 else 0.0
    
    good = still_df['Pillar_likelihood'] > likelihood_thresh
    
    if good.sum() < 5:
        return result
    
    model_df = still_df[good]
    
    pillar_x = model_df['Pillar_x'].median()
    pillar_y = model_df['Pillar_y'].median()
    pillar_pos = np.array([pillar_x, pillar_y])
    
    # Convert to ruler coordinates
    pillar_ruler = Geometry.to_ruler_coords(pillar_pos, sabl, sabr)
    
    # Compute expected pillar position (ground truth from geometry)
    computed_pillar = Geometry.compute_pillar_center(sabl, sabr)
    
    # Deviation from computed position (in ruler units)
    ruler_px = Geometry.distance(sabl, sabr)
    deviation_px = Geometry.distance(pillar_pos, computed_pillar)
    deviation_ruler = deviation_px / ruler_px
    
    result.update({
        'pillar_x': float(pillar_x),
        'pillar_y': float(pillar_y),
        'pillar_ruler': pillar_ruler,
        'likelihood': float(model_df['Pillar_likelihood'].mean()),
        'deviation_ruler': float(deviation_ruler)
    })
    
    return result


def build_pellet_model(df: pd.DataFrame, pellet_num: int, 
                       start_frame: int, end_frame: int,
                       ruler_px: float) -> PelletModel:
    """
    Build complete model for one pellet using ruler-based geometry.
    
    The key insight: pillar position is COMPUTED from geometry, not measured.
    DLC pellet/pillar measurements are compared against this ground truth.
    """
    
    # Find still period using ruler-based velocity threshold
    still_start, still_end, still_frames = find_still_period(
        df, start_frame, end_frame, ruler_px
    )
    
    # Build rectangle model (computes SAT from geometry)
    rect_model, n_good_frames = build_rectangle_model(
        df, still_start, still_end, ruler_px
    )
    
    if rect_model is None:
        # Fallback: use global medians from segment
        segment = df.iloc[start_frame:end_frame + 1]
        sabl = np.array([segment['SABL_x'].median(), segment['SABL_y'].median()])
        sabr = np.array([segment['SABR_x'].median(), segment['SABR_y'].median()])
        rect_corners = Geometry.compute_sa_rectangle(sabl, sabr)
        
        rect_model = RectangleModel(
            sabl_x=float(sabl[0]), sabl_y=float(sabl[1]),
            sabr_x=float(sabr[0]), sabr_y=float(sabr[1]),
            satl_x=float(rect_corners[3][0]), satl_y=float(rect_corners[3][1]),
            satr_x=float(rect_corners[2][0]), satr_y=float(rect_corners[2][1]),
            width_px=ruler_px,
            height_px=ruler_px * RULER_UNITS['sabr_satr'],
            width_ruler=1.0,
            height_ruler=RULER_UNITS['sabr_satr']
        )
        n_good_frames = 0
    
    # Get anchor positions
    sabl = np.array([rect_model.sabl_x, rect_model.sabl_y])
    sabr = np.array([rect_model.sabr_x, rect_model.sabr_y])
    
    # COMPUTE pillar position from geometry (ground truth)
    computed_pillar = Geometry.compute_pillar_center(sabl, sabr)
    computed_pillar_ruler = Geometry.to_ruler_coords(computed_pillar, sabl, sabr)
    
    # Get MEASURED pellet position (to compare with expected)
    pellet_info = get_pellet_position(df, still_start, still_end, sabl, sabr)
    
    # Get MEASURED pillar position (for validation)
    pillar_info = get_pillar_position(df, still_start, still_end, sabl, sabr)
    
    # Quality assessment
    if n_good_frames >= 100 and still_frames >= 500:
        quality = 'good'
    elif n_good_frames >= 20 and still_frames >= 100:
        quality = 'fair'
    else:
        quality = 'poor'
    
    return PelletModel(
        pellet_num=pellet_num,
        start_frame=start_frame,
        end_frame=end_frame,
        still_start=still_start,
        still_end=still_end,
        still_frames=still_frames,
        rectangle=rect_model,
        # Computed (ground truth) pillar position
        computed_pillar_x=float(computed_pillar[0]),
        computed_pillar_y=float(computed_pillar[1]),
        computed_pillar_ruler=computed_pillar_ruler,
        # Measured pellet position
        pellet_x=pellet_info['pellet_x'],
        pellet_y=pellet_info['pellet_y'],
        pellet_ruler=pellet_info['pellet_ruler'],
        pellet_likelihood=pellet_info['likelihood'],
        pellet_deviation_ruler=pellet_info['deviation_ruler'],
        # Measured pillar position (validation)
        pillar_x=pillar_info['pillar_x'],
        pillar_y=pillar_info['pillar_y'],
        pillar_ruler=pillar_info['pillar_ruler'],
        pillar_likelihood=pillar_info['likelihood'],
        pillar_deviation_ruler=pillar_info['deviation_ruler'],
        # Quality
        model_quality=quality,
        n_high_likelihood_frames=n_good_frames
    )


def build_pellet_models(dlc_path: Union[str, Path],
                        segmentation_path: Union[str, Path],
                        output_path: Optional[Path] = None) -> List[PelletModel]:
    """
    Build per-pellet models for all pellets in a video.
    
    Uses RULER calibration from segmentation for all thresholds.
    
    Args:
        dlc_path: Path to DLC output file
        segmentation_path: Path to segmentation JSON
        output_path: Where to save results (default: same as DLC file)
    
    Returns:
        List of PelletModel objects
    """
    dlc_path = Path(dlc_path)
    segmentation_path = Path(segmentation_path)
    
    print(f"Building per-pellet models for: {dlc_path.name}")
    
    # Load data
    df = load_dlc_data(dlc_path)
    seg = load_segmentation(segmentation_path)
    
    # Get ruler from segmentation (calibrated per-video)
    ruler_px = seg['ruler_calibration']['ruler_px']
    print(f"  Using ruler: {ruler_px:.1f}px = 9mm")
    
    models = []
    
    # Handle both old format (pellet_segments) and new format (segments with segment_type)
    if 'pellet_segments' in seg:
        # Old format
        pellet_segments = seg['pellet_segments']
    else:
        # New format - filter to pellet segments only
        pellet_segments = [s for s in seg['segments'] if s['segment_type'] == 'pellet']
    
    for pellet_seg in pellet_segments:
        pellet_num = pellet_seg.get('pellet_num') or pellet_seg.get('segment_num')
        start = pellet_seg['start_frame']
        end = pellet_seg['end_frame']
        
        model = build_pellet_model(df, pellet_num, start, end, ruler_px)
        models.append(model)
        
        # Print deviation info if available
        deviation_str = ""
        if model.pellet_deviation_ruler is not None:
            deviation_str = f", pellet_dev={model.pellet_deviation_ruler:.3f}r"
        
        print(f"  Pellet {pellet_num}: still={model.still_frames} frames, "
              f"quality={model.model_quality}, pellet_like={model.pellet_likelihood:.2f}{deviation_str}")
    
    # Save results
    if output_path is None:
        output_path = dlc_path.parent / f"{dlc_path.stem}_pellet_models.json"
    
    save_pellet_models(models, seg['video_name'], seg['ruler_calibration'], output_path)
    print(f"Saved: {output_path}")
    
    return models


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    return obj


def save_pellet_models(models: List[PelletModel], video_name: str, 
                       ruler_calibration: dict, filepath: Path):
    """Save pellet models to JSON."""
    data = {
        'video_name': video_name,
        'ruler_calibration': ruler_calibration,
        'n_pellets': len(models),
        'pellet_models': []
    }
    
    for m in models:
        model_dict = {
            'pellet_num': m.pellet_num,
            'start_frame': m.start_frame,
            'end_frame': m.end_frame,
            'still_start': m.still_start,
            'still_end': m.still_end,
            'still_frames': m.still_frames,
            'rectangle': asdict(m.rectangle),
            # Computed pillar (ground truth from geometry)
            'computed_pillar_x': m.computed_pillar_x,
            'computed_pillar_y': m.computed_pillar_y,
            'computed_pillar_ruler': m.computed_pillar_ruler,
            # Measured pellet position
            'pellet_x': m.pellet_x,
            'pellet_y': m.pellet_y,
            'pellet_ruler': m.pellet_ruler,
            'pellet_likelihood': m.pellet_likelihood,
            'pellet_deviation_ruler': m.pellet_deviation_ruler,
            # Measured pillar position (validation)
            'pillar_x': m.pillar_x,
            'pillar_y': m.pillar_y,
            'pillar_ruler': m.pillar_ruler,
            'pillar_likelihood': m.pillar_likelihood,
            'pillar_deviation_ruler': m.pillar_deviation_ruler,
            # Quality
            'model_quality': m.model_quality,
            'n_high_likelihood_frames': m.n_high_likelihood_frames
        }
        data['pellet_models'].append(model_dict)
    
    # Convert numpy types to native Python types
    data = convert_to_native(data)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def select_files_dialog() -> Tuple[Optional[Path], Optional[Path]]:
    """Open dialogs to select DLC file and segmentation file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        
        # Select DLC file
        dlc_path = filedialog.askopenfilename(
            title="Select DLC Output File",
            filetypes=[
                ("DLC files", "*.csv *.csv.gz *.h5"),
                ("All files", "*.*")
            ]
        )
        
        if not dlc_path:
            root.destroy()
            return None, None
        
        # Select segmentation file
        seg_path = filedialog.askopenfilename(
            title="Select Segmentation JSON",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        
        return Path(dlc_path) if dlc_path else None, Path(seg_path) if seg_path else None
        
    except ImportError:
        print("tkinter not available. Please provide filepaths as arguments.")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Build per-pellet models from segmented DLC data"
    )
    parser.add_argument(
        'dlc_path',
        nargs='?',
        help="Path to DLC file"
    )
    parser.add_argument(
        'segmentation_path',
        nargs='?',
        help="Path to segmentation JSON"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help="Output path for models JSON"
    )
    
    args = parser.parse_args()
    
    if args.dlc_path and args.segmentation_path:
        dlc_path = Path(args.dlc_path)
        seg_path = Path(args.segmentation_path)
    else:
        dlc_path, seg_path = select_files_dialog()
    
    if dlc_path is None or seg_path is None:
        print("Files not selected. Exiting.")
        return
    
    output_path = Path(args.output) if args.output else None
    
    build_pellet_models(dlc_path, seg_path, output_path)
    
    print("Done!")


if __name__ == "__main__":
    main()
