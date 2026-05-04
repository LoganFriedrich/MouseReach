"""
Per-video Pellet-on-pillar position calibration.

Computes the median Pellet position across pre-first-bout frames where
the pellet is loaded on the pillar and mouse hasn't yet reached. Used
as the reference "pellet-on-pillar" location for the cascade's
on-pillar / off-pillar classification, replacing the calculated
geometric pillar center.

Why: DLC's pellet bodypart often tracks at a systematic offset from
the calculated pillar center (per-video calibration drift). The
pre-first-bout window is the cleanest reference because the pellet
is in its loaded position and hasn't been manipulated.

Returns None if calibration is not reliable (insufficient confident
pre-first-bout frames). Stages should fall back to calculated pillar
center in that case.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series


PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
PELLET_LK_HIGH = 0.95
PAW_LK_THR = 0.5
MIN_CALIBRATION_FRAMES = 10
TRANSITION_ZONE_HALF = 5


@dataclass
class PelletOnPillarCalibration:
    """Per-segment calibration for the 'on-pillar pellet position'."""
    on_pillar_x: float           # median Pellet x in pre-first-bout frames
    on_pillar_y: float           # median Pellet y in pre-first-bout frames
    n_calibration_frames: int    # how many frames went into the median
    deviation_from_calc_pillar_radii: float  # offset from calculated pillar center
    is_reliable: bool            # True if n_calibration_frames >= threshold


def calibrate_pellet_on_pillar(
    dlc_df: pd.DataFrame,
    seg_start: int,
    seg_end: int,
    transition_zone_half: int = TRANSITION_ZONE_HALF,
    pellet_lk_threshold: float = PELLET_LK_HIGH,
    paw_lk_threshold: float = PAW_LK_THR,
    min_calibration_frames: int = MIN_CALIBRATION_FRAMES,
) -> Optional[PelletOnPillarCalibration]:
    """Compute pellet-on-pillar reference position from pre-first-bout
    frames in the segment.

    Returns None if the segment lacks usable calibration frames (e.g.,
    DLC entirely missed the pellet pre-reach).
    """
    clean_end = seg_end - transition_zone_half
    if clean_end <= seg_start:
        return None

    sub_raw = dlc_df.iloc[seg_start:clean_end + 1]
    n = len(sub_raw)
    if n == 0:
        return None

    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    geom = compute_pillar_geometry_series(sub)
    pillar_cx = geom["pillar_cx"].to_numpy(dtype=float)
    pillar_cy = geom["pillar_cy"].to_numpy(dtype=float)
    pillar_r = geom["pillar_r"].to_numpy(dtype=float)
    slit_y_line = pillar_cy + pillar_r

    pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
    pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
    pellet_y = sub["Pellet_y"].to_numpy(dtype=float)

    # Find first paw-past-slit frame.
    paw_past_y = np.zeros(n, dtype=bool)
    for bp in PAW_BODYPARTS:
        py = sub[f"{bp}_y"].to_numpy(dtype=float)
        pl = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
        paw_past_y |= (py <= slit_y_line) & (pl >= paw_lk_threshold)

    first_paw_past_idx = -1
    for i in range(n):
        if paw_past_y[i]:
            first_paw_past_idx = i
            break

    # Calibration window: from segment start to just before first
    # paw-past-slit (or to clean-zone end if no reaches).
    cal_end = first_paw_past_idx if first_paw_past_idx >= 0 else n
    if cal_end <= 0:
        return None

    # Eligible frames: confident pellet, paw NOT past slit (already
    # guaranteed in cal range, but double-check), high lk.
    eligible = (
        (pellet_lk[:cal_end] >= pellet_lk_threshold)
        & (~paw_past_y[:cal_end])
    )
    n_eligible = int(eligible.sum())
    if n_eligible < min_calibration_frames:
        return None

    # Median position.
    cal_x = float(np.median(pellet_x[:cal_end][eligible]))
    cal_y = float(np.median(pellet_y[:cal_end][eligible]))

    # Compare to calculated pillar center for diagnostics.
    cal_pillar_cx = float(np.median(pillar_cx[:cal_end][eligible]))
    cal_pillar_cy = float(np.median(pillar_cy[:cal_end][eligible]))
    cal_pillar_r = float(np.median(pillar_r[:cal_end][eligible]))
    deviation = ((cal_x - cal_pillar_cx) ** 2
                 + (cal_y - cal_pillar_cy) ** 2) ** 0.5
    deviation_radii = deviation / max(cal_pillar_r, 1e-6)

    return PelletOnPillarCalibration(
        on_pillar_x=cal_x,
        on_pillar_y=cal_y,
        n_calibration_frames=n_eligible,
        deviation_from_calc_pillar_radii=deviation_radii,
        is_reliable=True,
    )
