"""
Skeleton Inference from DLC Keypoints.

Estimates shoulder and elbow positions from available DLC bodyparts
(Nose, LeftEar, RightEar, RightHand) using anatomical priors and
geometric constraints from the Gilmer mouse forelimb model.

The mouse is NOT head-fixed — it freely approaches the slit. Head
orientation is estimated from ear-ear and nose vectors, then shoulder
is placed at a fixed anatomical offset from the head. Elbow is solved
via 2-link inverse kinematics given shoulder, paw, and segment lengths.

All distances are in pixels and converted to mm via the 9mm ruler.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# Gilmer model segment lengths (mm) — from mean inter-marker distances
# These are approximate; can be refined per-animal from pre-injury data
HUMERUS_LENGTH_MM = 10.0    # Shoulder to elbow
FOREARM_LENGTH_MM = 11.0    # Elbow to paw (radius + hand)

# Anatomical offset: shoulder relative to BOXR (slit edge) in mm.
# During reaches the mouse presses against the slit, so the shoulder
# is just inside the cage — a few mm behind the slit opening.
# In image coords: "behind slit" = smaller X (toward cage interior).
# Paw-to-BOXR is ~18mm, total arm is ~21mm, so shoulder is ~3mm behind BOXR.
SHOULDER_BEHIND_SLIT_MM = 3.5     # Behind BOXR (into cage)
SHOULDER_BELOW_SLIT_MM = 2.0      # Below BOXR (ventral offset)


def estimate_head_frame(
    nose_x: float, nose_y: float,
    left_ear_x: float, left_ear_y: float,
    right_ear_x: float, right_ear_y: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute head coordinate frame from DLC keypoints.

    Returns:
        Tuple of (ear_midpoint, forward_unit, right_unit) where:
        - ear_midpoint: (x, y) midpoint of ears
        - forward_unit: unit vector from ear midpoint toward nose
        - right_unit: unit vector perpendicular to forward (rightward)

    Note: In image coordinates, Y increases downward. "Ventral" in the
    mouse corresponds to increasing Y when the mouse faces right.
    """
    ear_mid = np.array([
        (left_ear_x + right_ear_x) / 2.0,
        (left_ear_y + right_ear_y) / 2.0,
    ])

    # Forward direction: ear midpoint -> nose
    forward = np.array([nose_x - ear_mid[0], nose_y - ear_mid[1]])
    forward_len = np.linalg.norm(forward)
    if forward_len < 1e-6:
        # Degenerate — fall back to horizontal
        forward_unit = np.array([1.0, 0.0])
    else:
        forward_unit = forward / forward_len

    # Right perpendicular (90 degrees clockwise in image coords)
    # In image coords: right = rotate forward by +90 degrees
    # rotate (fx, fy) by +90: (fy, -fx) ... but image Y is down
    # For a mouse facing right: forward ~ (1, 0), right ~ (0, 1) = ventral
    right_unit = np.array([forward_unit[1], -forward_unit[0]])

    return ear_mid, forward_unit, right_unit


def estimate_shoulder_from_slit(
    boxr_x: float, boxr_y: float,
    nose_x: float, nose_y: float,
    mm_per_px: float,
    behind_mm: float = SHOULDER_BEHIND_SLIT_MM,
    below_mm: float = SHOULDER_BELOW_SLIT_MM,
) -> np.ndarray:
    """
    Estimate shoulder position from BOXR (slit edge) during a reach.

    The mouse presses against the slit to reach. The shoulder is just
    inside the cage, a few mm behind the slit opening. We use the
    nose->BOXR direction to determine "behind" (into the cage).

    Args:
        boxr_x, boxr_y: Slit right edge position in pixels
        nose_x, nose_y: Nose position in pixels
        mm_per_px: Scale factor from ruler calibration
        behind_mm: How far behind BOXR (into cage) in mm
        below_mm: How far below BOXR (ventral) in mm

    Returns:
        (x, y) estimated shoulder position in pixels
    """
    px_per_mm = 1.0 / mm_per_px
    boxr = np.array([boxr_x, boxr_y])

    # "Behind slit" = direction from BOXR toward nose (into cage)
    toward_cage = np.array([nose_x - boxr_x, nose_y - boxr_y])
    cage_len = np.linalg.norm(toward_cage)
    if cage_len < 1e-6:
        # Degenerate — assume cage is to the left
        cage_unit = np.array([-1.0, 0.0])
    else:
        cage_unit = toward_cage / cage_len

    # "Below" = perpendicular to cage direction, downward in image
    # Rotate cage_unit 90 degrees clockwise: (x,y) -> (y, -x)
    down_unit = np.array([cage_unit[1], -cage_unit[0]])
    # Ensure it points downward (positive Y in image coords)
    if down_unit[1] < 0:
        down_unit = -down_unit

    shoulder = boxr.copy()
    shoulder += cage_unit * (behind_mm * px_per_mm)
    shoulder += down_unit * (below_mm * px_per_mm)

    return shoulder


def solve_elbow_2link(
    shoulder: np.ndarray,
    paw: np.ndarray,
    humerus_px: float,
    forearm_px: float,
) -> Optional[np.ndarray]:
    """
    Solve 2-link IK for elbow position given shoulder and paw.

    Given two segment lengths (humerus: shoulder->elbow, forearm:
    elbow->paw) and the endpoint positions, computes the elbow
    position. Returns the solution where the elbow is BELOW the
    shoulder-paw line (anatomically correct for a reaching mouse).

    Args:
        shoulder: (x, y) in pixels
        paw: (x, y) in pixels
        humerus_px: Humerus length in pixels
        forearm_px: Forearm length in pixels

    Returns:
        (x, y) elbow position in pixels, or None if geometry is
        impossible (reach exceeds total arm length).
    """
    d = np.linalg.norm(paw - shoulder)

    # Check reachability
    if d > humerus_px + forearm_px:
        # Overextended — place elbow at midpoint along the line
        # (best approximation for a fully extended arm)
        ratio = humerus_px / (humerus_px + forearm_px)
        return shoulder + (paw - shoulder) * ratio

    if d < abs(humerus_px - forearm_px):
        # Segments overlap — shouldn't happen in reaching, but handle it
        ratio = humerus_px / (humerus_px + forearm_px)
        return shoulder + (paw - shoulder) * ratio

    # Law of cosines: find angle at shoulder
    cos_angle = (humerus_px**2 + d**2 - forearm_px**2) / (2 * humerus_px * d)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    # Direction from shoulder to paw
    sp_dir = (paw - shoulder) / d

    # Two solutions: elbow above or below the shoulder-paw line
    # For a reaching mouse, the elbow is typically BELOW (higher Y in image)
    # We pick the solution with higher Y (more ventral)

    # Rotate sp_dir by +angle and -angle
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Solution 1: rotate clockwise (elbow below for rightward reach)
    elbow1 = shoulder + humerus_px * np.array([
        sp_dir[0] * cos_a - sp_dir[1] * sin_a,
        sp_dir[0] * sin_a + sp_dir[1] * cos_a,
    ])

    # Solution 2: rotate counter-clockwise
    elbow2 = shoulder + humerus_px * np.array([
        sp_dir[0] * cos_a + sp_dir[1] * sin_a,
        -sp_dir[0] * sin_a + sp_dir[1] * cos_a,
    ])

    # Pick the one with higher Y (more ventral in image coords)
    if elbow1[1] >= elbow2[1]:
        return elbow1
    else:
        return elbow2


def infer_skeleton_frame(
    row: pd.Series,
    mm_per_px: float,
    humerus_mm: float = HUMERUS_LENGTH_MM,
    forearm_mm: float = FOREARM_LENGTH_MM,
    likelihood_threshold: float = 0.5,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Infer full skeleton (shoulder, elbow, paw) for one DLC frame.

    Args:
        row: One row of DLC DataFrame (flat column names)
        mm_per_px: Scale factor from ruler calibration
        humerus_mm: Humerus length in mm
        forearm_mm: Forearm length in mm
        likelihood_threshold: Minimum likelihood for keypoints

    Returns:
        Dict with 'shoulder', 'elbow', 'paw' as (x, y) pixel arrays,
        or None if insufficient data.
    """
    # Check required keypoints: Nose, BOXR (stationary), RightHand
    # BOXR has no likelihood filter — it's a fixed arena point (always 100%)
    required_tracked = {
        'Nose': ('Nose_x', 'Nose_y', 'Nose_likelihood'),
        'RightHand': ('RightHand_x', 'RightHand_y', 'RightHand_likelihood'),
    }
    required_fixed = {
        'BOXR': ('BOXR_x', 'BOXR_y'),
    }

    positions = {}
    for name, (xc, yc, lc) in required_tracked.items():
        if row.get(lc, 0) < likelihood_threshold:
            return None
        x, y = row[xc], row[yc]
        if np.isnan(x) or np.isnan(y):
            return None
        positions[name] = np.array([x, y])

    for name, (xc, yc) in required_fixed.items():
        x, y = row[xc], row[yc]
        if np.isnan(x) or np.isnan(y):
            return None
        positions[name] = np.array([x, y])

    # Estimate shoulder from slit position
    shoulder = estimate_shoulder_from_slit(
        positions['BOXR'][0], positions['BOXR'][1],
        positions['Nose'][0], positions['Nose'][1],
        mm_per_px,
    )

    # Convert segment lengths to pixels
    px_per_mm = 1.0 / mm_per_px
    humerus_px = humerus_mm * px_per_mm
    forearm_px = forearm_mm * px_per_mm

    # Solve for elbow
    paw = positions['RightHand']
    elbow = solve_elbow_2link(shoulder, paw, humerus_px, forearm_px)

    if elbow is None:
        return None

    return {
        'shoulder': shoulder,
        'elbow': elbow,
        'paw': paw,
        'boxr': positions['BOXR'],
        'nose': positions['Nose'],
    }


def infer_skeleton_video(
    df: pd.DataFrame,
    mm_per_px: float,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    humerus_mm: float = HUMERUS_LENGTH_MM,
    forearm_mm: float = FOREARM_LENGTH_MM,
) -> pd.DataFrame:
    """
    Infer skeleton for all frames in a video (or frame range).

    Args:
        df: Full DLC DataFrame (flat column names)
        mm_per_px: Scale factor from ruler
        start_frame: First frame (inclusive). Default: 0.
        end_frame: Last frame (inclusive). Default: last.
        humerus_mm: Humerus length
        forearm_mm: Forearm length

    Returns:
        DataFrame with columns: frame, shoulder_x, shoulder_y,
        elbow_x, elbow_y, paw_x, paw_y. NaN where inference failed.
    """
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = len(df) - 1

    records = []
    for frame_idx in range(start_frame, end_frame + 1):
        row = df.iloc[frame_idx]
        result = infer_skeleton_frame(row, mm_per_px,
                                       humerus_mm=humerus_mm,
                                       forearm_mm=forearm_mm)

        if result is not None:
            records.append({
                'frame': frame_idx,
                'shoulder_x': result['shoulder'][0],
                'shoulder_y': result['shoulder'][1],
                'elbow_x': result['elbow'][0],
                'elbow_y': result['elbow'][1],
                'paw_x': result['paw'][0],
                'paw_y': result['paw'][1],
            })
        else:
            records.append({
                'frame': frame_idx,
                'shoulder_x': np.nan,
                'shoulder_y': np.nan,
                'elbow_x': np.nan,
                'elbow_y': np.nan,
                'paw_x': np.nan,
                'paw_y': np.nan,
            })

    return pd.DataFrame(records)
