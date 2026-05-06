"""
DLC to OpenSim TRC Converter.

Converts DeepLabCut .h5 tracking output to OpenSim .trc (Track Row Column)
format for inverse kinematics and Moco optimal control.

The TRC file contains 3D marker positions over time. Since MouseReach uses
a single camera (2D), we project onto the sagittal plane (Z=0) unless
depth data is provided.

Marker mapping (DLC bodypart -> OpenSim marker):
    Shoulder  -> proximal reference (glenohumeral joint)
    Elbow     -> mid-limb joint
    RightHand -> distal endpoint (paw)

Coordinate system:
    DLC: X=right, Y=down (image convention)
    OpenSim: X=right, Y=up, Z=forward (right-hand convention)
    Transform: flip Y axis, scale pixels -> meters via 9mm ruler

References:
    Gilmer et al. 2025 - Validated with these 3 markers
    OpenSim TRC format spec - simtk.org
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# MouseReach default settings
DEFAULT_FPS = 200.0
RULER_MM = 9.0  # SABL-SABR physical distance

# DLC bodypart -> OpenSim marker name
# NOTE: The current MouseReach DLC model (MPSAOct27) does NOT track shoulder
# or elbow. Only RightHand (paw) is available as a limb marker. Shoulder and
# elbow would need to be added to the DLC model via manual labeling + retrain
# to enable full 3-marker IK with the Gilmer model.
#
# Available bodyparts in current model:
#   Limb: RightHand, RHLeft, RHOut, RHRight (paw only, visible during reaches)
#   Head: Nose, LeftEar, RightEar
#   Body: TailBase, LeftFoot
#   Arena: BOXL, BOXR, SABL, SABR, SATL, SATR, Reference, Pellet, Pillar
DEFAULT_MARKER_MAP = {
    'RightHand': 'Paw',
}

# Future marker map once DLC model includes shoulder/elbow:
FULL_MARKER_MAP = {
    'Shoulder': 'Shoulder',
    'Elbow': 'Elbow',
    'RightHand': 'Paw',
}

# Minimum DLC likelihood to include a point (below -> NaN)
LIKELIHOOD_THRESHOLD = 0.6


def load_dlc_h5(h5_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Load DLC .h5 file and flatten column names.

    Returns:
        Tuple of (DataFrame with flat columns like 'Shoulder_x', scorer name)
    """
    df = pd.read_hdf(h5_path)
    scorer = df.columns.get_level_values(0)[0]
    df.columns = ['_'.join(col[1:]) for col in df.columns]
    return df, scorer


def compute_pixel_scale(
    df: pd.DataFrame,
    ruler_left: str = 'SABL',
    ruler_right: str = 'SABR',
    ruler_mm: float = RULER_MM,
) -> float:
    """
    Compute pixel-to-mm scale factor from ruler markers.

    Args:
        df: DLC DataFrame with flat column names
        ruler_left: Left ruler bodypart name
        ruler_right: Right ruler bodypart name
        ruler_mm: Physical distance between ruler markers in mm

    Returns:
        Scale factor in mm per pixel
    """
    lx_col = f'{ruler_left}_x'
    ly_col = f'{ruler_left}_y'
    rx_col = f'{ruler_right}_x'
    ry_col = f'{ruler_right}_y'
    ll_col = f'{ruler_left}_likelihood'
    rl_col = f'{ruler_right}_likelihood'

    # Use high-confidence frames only
    good = (df[ll_col] > 0.9) & (df[rl_col] > 0.9)
    if not good.any():
        # Fall back to all frames
        good = pd.Series(True, index=df.index)

    dist_px = np.sqrt(
        (df.loc[good, rx_col] - df.loc[good, lx_col]) ** 2 +
        (df.loc[good, ry_col] - df.loc[good, ly_col]) ** 2
    )

    median_dist = dist_px.median()
    if median_dist <= 0 or np.isnan(median_dist):
        raise ValueError("Could not compute ruler distance from DLC data")

    return ruler_mm / median_dist


def dlc_to_trc(
    h5_path: Path,
    output_path: Path,
    fps: float = DEFAULT_FPS,
    marker_map: Optional[Dict[str, str]] = None,
    ruler_mm: float = RULER_MM,
    z_data: Optional[np.ndarray] = None,
    reach_frames: Optional[Tuple[int, int]] = None,
) -> Path:
    """
    Convert DLC .h5 output to OpenSim .trc format.

    Args:
        h5_path: Path to DLC .h5 file
        output_path: Path for output .trc file
        fps: Video frame rate (MouseReach default: 200)
        marker_map: Dict mapping DLC bodypart -> OpenSim marker name.
                   Default: Shoulder->Shoulder, Elbow->Elbow, RightHand->Paw
        ruler_mm: Physical ruler distance in mm (default: 9.0)
        z_data: Optional array of shape (n_frames, n_markers) with Z coords
                in pixels. If None, Z=0 (sagittal plane projection).
        reach_frames: Optional (start, end) to export only a reach segment.
                     If None, exports entire video.

    Returns:
        Path to the written .trc file
    """
    if marker_map is None:
        marker_map = DEFAULT_MARKER_MAP

    # Load DLC data
    df, scorer = load_dlc_h5(h5_path)

    # Compute pixel-to-meter scale
    mm_per_px = compute_pixel_scale(df, ruler_mm=ruler_mm)
    m_per_px = mm_per_px / 1000.0

    # Slice to reach if specified
    if reach_frames is not None:
        start, end = reach_frames
        df = df.iloc[start:end + 1].reset_index(drop=True)
        if z_data is not None:
            z_data = z_data[start:end + 1]

    n_frames = len(df)
    dlc_parts = list(marker_map.keys())
    osim_names = list(marker_map.values())
    n_markers = len(dlc_parts)

    # Build coordinate arrays
    coords = np.full((n_frames, n_markers, 3), np.nan)

    for j, bp in enumerate(dlc_parts):
        x_col = f'{bp}_x'
        y_col = f'{bp}_y'
        l_col = f'{bp}_likelihood'

        if x_col not in df.columns:
            raise ValueError(f"DLC bodypart '{bp}' not found in {h5_path.name}")

        x = df[x_col].values
        y = df[y_col].values
        likelihood = df[l_col].values

        # Filter low-confidence frames
        low_conf = likelihood < LIKELIHOOD_THRESHOLD
        x[low_conf] = np.nan
        y[low_conf] = np.nan

        # Transform: pixels -> meters, flip Y (DLC Y-down -> OpenSim Y-up)
        coords[:, j, 0] = x * m_per_px          # X: right
        coords[:, j, 1] = -y * m_per_px          # Y: up (flipped)

        if z_data is not None:
            coords[:, j, 2] = z_data[:, j] * m_per_px  # Z: forward
        else:
            coords[:, j, 2] = 0.0                # Sagittal plane

        # NaN out Z for low-confidence frames too
        coords[low_conf, j, 2] = np.nan

    # Write TRC file
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        # Header line 1
        f.write(f'PathFileType\t4\t(X/Y/Z)\t{output_path.name}\n')

        # Header line 2
        f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t'
                'OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')

        # Header line 3
        f.write(f'{fps:.0f}\t{fps:.0f}\t{n_frames}\t{n_markers}\tm\t'
                f'{fps:.0f}\t1\t{n_frames}\n')

        # Header line 4: marker names
        name_row = 'Frame#\tTime'
        for name in osim_names:
            name_row += f'\t{name}\t\t'  # 3 columns per marker (X, Y, Z)
        f.write(name_row.rstrip() + '\n')

        # Header line 5: XYZ sub-headers
        xyz_row = '\t'
        for j in range(n_markers):
            xyz_row += f'\tX{j+1}\tY{j+1}\tZ{j+1}'
        f.write(xyz_row + '\n')

        # Blank line (some parsers expect it)
        f.write('\n')

        # Data rows
        for i in range(n_frames):
            time = i / fps
            row = f'{i+1}\t{time:.6f}'
            for j in range(n_markers):
                x, y, z = coords[i, j]
                if np.isnan(x):
                    row += '\t\t\t'  # Empty = gap in tracking
                else:
                    row += f'\t{x:.6f}\t{y:.6f}\t{z:.6f}'
            f.write(row + '\n')

    return output_path


def batch_convert_reaches(
    h5_path: Path,
    reaches_json_path: Path,
    output_dir: Path,
    fps: float = DEFAULT_FPS,
    marker_map: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """
    Convert all reaches in a video to individual .trc files.

    Args:
        h5_path: Path to DLC .h5 file
        reaches_json_path: Path to MouseReach *_reaches.json
        output_dir: Directory for output .trc files
        fps: Video frame rate
        marker_map: DLC->OpenSim marker mapping

    Returns:
        List of paths to written .trc files
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(reaches_json_path) as f:
        reaches_data = json.load(f)

    trc_paths = []
    video_name = h5_path.stem.replace('DLC_', '').split('_resnet')[0]

    for seg in reaches_data.get('segments', []):
        for reach in seg.get('reaches', []):
            reach_id = reach['reach_id']
            start = reach['start_frame']
            end = reach['end_frame']

            trc_name = f'{video_name}_reach{reach_id:04d}.trc'
            trc_path = output_dir / trc_name

            try:
                dlc_to_trc(
                    h5_path, trc_path,
                    fps=fps,
                    marker_map=marker_map,
                    reach_frames=(start, end),
                )
                trc_paths.append(trc_path)
            except Exception as e:
                print(f'[!] Failed to convert reach {reach_id}: {e}')

    return trc_paths
