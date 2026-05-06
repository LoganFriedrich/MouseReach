"""
Pillar geometry primitives -- shared across the pipeline.

These were originally implemented inline in the napari widget at
`outcomes/pillar_geometry_widget.py`. Factored here so any algo
(reach detection, outcome, assignment, evaluators) can use the same
calculated pillar position without duplicating the formula.

The calculated pillar circle is derived per-frame from SABL/SABR
(the front edge of the SA tray):

  ruler          = euclidean(SABL, SABR)
  SA midpoint    = ((SABL_x + SABR_x)/2, (SABL_y + SABR_y)/2)
  pillar center  = SA midpoint - 0.944 * ruler perpendicular ("above" SA)
  pillar radius  = 0.10 * ruler

Because the geometry is derived from the SA points themselves, it is
naturally tray-relative: when the mouse grabs the tray and shifts the
apparatus, both SA and the calculated pillar move together. Use this
for any "pellet position relative to pillar" reasoning -- absolute
camera coordinates do not compose with tray motion.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

# Calibration constants (lifted from the widget's _compute_auto_pillar)
PILLAR_PERPENDICULAR_OFFSET = 0.944    # ruler-units above SA midpoint
PILLAR_RADIUS_RULER_FRACTION = 0.10    # pillar radius as fraction of ruler
DEFAULT_PELLET_LK_THRESHOLD = 0.7      # for "pellet detected" boolean

# Smoothing window for SA positions before computing pillar geometry.
# Per-raw-frame inherits DLC single-frame noise; long-window aggregation
# collapses real apparatus motion. 3-5 frames suppresses jitter while
# preserving the per-frame correspondence to apparatus position.
DEFAULT_SA_SMOOTH_WINDOW = 3


def compute_pillar_geometry_row(row: pd.Series) -> Tuple[float, float, float]:
    """Compute (center_x, center_y, radius) for a single DLC row.

    Uses SABL/SABR positions. If either is missing or NaN, returns
    (nan, nan, nan).
    """
    sabl_x = row.get("SABL_x")
    sabl_y = row.get("SABL_y")
    sabr_x = row.get("SABR_x")
    sabr_y = row.get("SABR_y")
    if any(v is None or (isinstance(v, float) and np.isnan(v))
           for v in (sabl_x, sabl_y, sabr_x, sabr_y)):
        return (float("nan"), float("nan"), float("nan"))

    mid_x = (sabl_x + sabr_x) / 2.0
    mid_y = (sabl_y + sabr_y) / 2.0
    dx = sabr_x - sabl_x
    dy = sabr_y - sabl_y
    ruler = float(np.sqrt(dx * dx + dy * dy))

    # In image coords "above" SA is negative Y (matches the widget formula).
    pillar_x = mid_x
    pillar_y = mid_y - (PILLAR_PERPENDICULAR_OFFSET * ruler)
    radius = PILLAR_RADIUS_RULER_FRACTION * ruler
    return float(pillar_x), float(pillar_y), float(radius)


def compute_pillar_geometry_series(
    dlc_df: pd.DataFrame,
    smooth_window: int = DEFAULT_SA_SMOOTH_WINDOW,
) -> pd.DataFrame:
    """Vectorized version: returns a DataFrame indexed like `dlc_df`
    with columns `pillar_cx`, `pillar_cy`, `pillar_r`, `ruler`.

    SA positions (SABL, SABR) are smoothed over `smooth_window` frames
    (centered moving average) before computing geometry. This suppresses
    single-frame DLC jitter while preserving per-frame correspondence to
    real apparatus motion. Pass `smooth_window=1` to compute on raw
    positions if you specifically need that.
    """
    if smooth_window > 1:
        sabl_x = dlc_df["SABL_x"].rolling(
            smooth_window, center=True, min_periods=1).mean().to_numpy(dtype=float)
        sabl_y = dlc_df["SABL_y"].rolling(
            smooth_window, center=True, min_periods=1).mean().to_numpy(dtype=float)
        sabr_x = dlc_df["SABR_x"].rolling(
            smooth_window, center=True, min_periods=1).mean().to_numpy(dtype=float)
        sabr_y = dlc_df["SABR_y"].rolling(
            smooth_window, center=True, min_periods=1).mean().to_numpy(dtype=float)
    else:
        sabl_x = dlc_df["SABL_x"].to_numpy(dtype=float)
        sabl_y = dlc_df["SABL_y"].to_numpy(dtype=float)
        sabr_x = dlc_df["SABR_x"].to_numpy(dtype=float)
        sabr_y = dlc_df["SABR_y"].to_numpy(dtype=float)

    mid_x = (sabl_x + sabr_x) / 2.0
    mid_y = (sabl_y + sabr_y) / 2.0
    dx = sabr_x - sabl_x
    dy = sabr_y - sabl_y
    ruler = np.sqrt(dx * dx + dy * dy)

    pillar_cx = mid_x
    pillar_cy = mid_y - (PILLAR_PERPENDICULAR_OFFSET * ruler)
    pillar_r = PILLAR_RADIUS_RULER_FRACTION * ruler

    return pd.DataFrame({
        "pillar_cx": pillar_cx,
        "pillar_cy": pillar_cy,
        "pillar_r": pillar_r,
        "ruler": ruler,
    }, index=dlc_df.index)


def compute_pillar_geometry_series_cleaned(
    dlc_df: pd.DataFrame,
    smooth_window: int = DEFAULT_SA_SMOOTH_WINDOW,
    cleaning_kwargs: dict = None,
) -> pd.DataFrame:
    """Like `compute_pillar_geometry_series` but applies the DLC
    impossibility filter to SA bodyparts (`SABL`, `SABR`, `SATL`,
    `SATR`) first, so single-frame DLC failures and whole-frame
    recording artifacts cannot contaminate pillar geometry.

    The original `compute_pillar_geometry_series` is preserved unchanged
    so callers that need the historical (uncleaned) calc -- e.g. Stage 1
    and Stage 2 of the v6 cascade outcome detector, whose validation
    provenance is already locked in -- can continue using it without
    re-validation.

    Parameters
    ----------
    dlc_df
        Raw DLC dataframe.
    smooth_window
        Centered moving-average window applied to SA positions AFTER
        cleaning (to suppress sub-pixel jitter on confident frames).
    cleaning_kwargs
        Optional kwargs forwarded to
        `mousereach.lib.dlc_cleaning.clean_dlc_bodyparts`.

    Returns
    -------
    DataFrame indexed like `dlc_df` with columns `pillar_cx`,
    `pillar_cy`, `pillar_r`, `ruler` -- same shape as the un-cleaned
    function.
    """
    from .dlc_cleaning import clean_dlc_bodyparts

    cleaning_kwargs = cleaning_kwargs or {}
    cleaned = clean_dlc_bodyparts(dlc_df, **cleaning_kwargs)
    return compute_pillar_geometry_series(cleaned, smooth_window=smooth_window)


def pellet_inside_pillar_circle(
    dlc_df: pd.DataFrame,
    pillar_geom: pd.DataFrame = None,
    lk_threshold: float = DEFAULT_PELLET_LK_THRESHOLD,
) -> pd.Series:
    """Per-frame boolean: is the pellet detected (lk >= threshold) AND
    inside the calculated pillar circle?

    Returns a boolean Series indexed like dlc_df.
    """
    if pillar_geom is None:
        pillar_geom = compute_pillar_geometry_series(dlc_df)

    px = dlc_df["Pellet_x"].to_numpy(dtype=float)
    py = dlc_df["Pellet_y"].to_numpy(dtype=float)
    plk = dlc_df["Pellet_likelihood"].to_numpy(dtype=float)
    cx = pillar_geom["pillar_cx"].to_numpy()
    cy = pillar_geom["pillar_cy"].to_numpy()
    r = pillar_geom["pillar_r"].to_numpy()

    dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    inside = (plk >= lk_threshold) & (dist <= r)
    return pd.Series(inside, index=dlc_df.index, name="pellet_inside_pillar_circle")


def pellet_dist_from_pillar_center(
    dlc_df: pd.DataFrame,
    pillar_geom: pd.DataFrame = None,
) -> pd.Series:
    """Per-frame Euclidean distance from pellet to pillar center
    (in pixels). NaN where pellet or SA is missing.
    """
    if pillar_geom is None:
        pillar_geom = compute_pillar_geometry_series(dlc_df)
    px = dlc_df["Pellet_x"].to_numpy(dtype=float)
    py = dlc_df["Pellet_y"].to_numpy(dtype=float)
    cx = pillar_geom["pillar_cx"].to_numpy()
    cy = pillar_geom["pillar_cy"].to_numpy()
    dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    return pd.Series(dist, index=dlc_df.index, name="pellet_dist_from_pillar_center")


def to_tray_relative(
    dlc_df: pd.DataFrame,
    bodyparts: list = None,
) -> pd.DataFrame:
    """Compute tray-relative coordinates for a list of bodyparts.

    Origin: SA midpoint (mean of SABL and SABR positions).
    No basis rotation -- just translation by the SA midpoint per frame.
    For each bodypart `bp`, produces `bp_x_tray` and `bp_y_tray` columns.

    If `bodyparts` is None, defaults to all of: Pellet, Pillar,
    RightHand, RHLeft, RHOut, RHRight, Nose, RightEar, LeftEar,
    LeftFoot, TailBase. (SA points are excluded since the tray is the
    reference itself.)
    """
    if bodyparts is None:
        bodyparts = ["Pellet", "Pillar", "RightHand", "RHLeft", "RHOut",
                     "RHRight", "Nose", "RightEar", "LeftEar", "LeftFoot",
                     "TailBase"]

    # SA midpoint computed from SMOOTHED SABL/SABR (DEFAULT_SA_SMOOTH_WINDOW)
    # to suppress DLC single-frame jitter; bodyparts themselves are NOT
    # smoothed here -- the caller decides if the pre-translated bodypart
    # positions need smoothing.
    sw = DEFAULT_SA_SMOOTH_WINDOW
    sabl_x = dlc_df["SABL_x"].rolling(sw, center=True, min_periods=1).mean().to_numpy(dtype=float)
    sabl_y = dlc_df["SABL_y"].rolling(sw, center=True, min_periods=1).mean().to_numpy(dtype=float)
    sabr_x = dlc_df["SABR_x"].rolling(sw, center=True, min_periods=1).mean().to_numpy(dtype=float)
    sabr_y = dlc_df["SABR_y"].rolling(sw, center=True, min_periods=1).mean().to_numpy(dtype=float)
    mid_x = (sabl_x + sabr_x) / 2.0
    mid_y = (sabl_y + sabr_y) / 2.0

    out = {}
    for bp in bodyparts:
        if f"{bp}_x" not in dlc_df.columns:
            continue
        bx = dlc_df[f"{bp}_x"].to_numpy(dtype=float)
        by = dlc_df[f"{bp}_y"].to_numpy(dtype=float)
        out[f"{bp}_x_tray"] = bx - mid_x
        out[f"{bp}_y_tray"] = by - mid_y
    return pd.DataFrame(out, index=dlc_df.index)
