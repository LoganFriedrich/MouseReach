"""
Per-frame feature extraction from raw DLC tracking, for the v8 reach
detector.

Feature schema (v2 -- with derived geometry)
--------------------------------------------
Per bodypart (18 total):
  {bp}_x, {bp}_y, {bp}_lk           -- raw DLC
  {bp}_x_smooth, {bp}_y_smooth      -- 5-frame centered MA
  {bp}_vx, {bp}_vy                  -- centered finite-diff velocity (DT=2)
  {bp}_ax, {bp}_ay                  -- finite-diff acceleration
  {bp}_speed                        -- sqrt(vx^2 + vy^2)
  {bp}_dlk                          -- centered finite-diff dlk
  {bp}_speed_max20, {bp}_speed_max40 -- rolling max |speed| over ±10/±20f
                                       (captures "did this bodypart move
                                       recently" at wider context than
                                       the centered-DT velocity)
  {bp}_lk_min20                     -- rolling min lk over ±10f (sustained
                                       low-confidence indicator)

Per pair of bodyparts (n*(n-1)/2 = 153 pairs):
  dist__{bp_a}__{bp_b}              -- Euclidean distance, smoothed (x_smooth, y_smooth)

Total: 18 * 14 + 153 = 405 features per frame.

Design notes
------------
- All 18 bodyparts are included. Per `feedback_use_all_dlc_features.md`:
  the model decides what's informative; we do not pre-filter by anatomy.
- Pairwise distances span all (a, b) combinations including constant-pair
  references (e.g. BOXL <-> BOXR). Tree models will assign zero
  importance to constant features at no cost.
- Wider rolling-max speed and rolling-min likelihood encode the "did
  something happen here recently" signal that centered DT=2 velocity
  cannot.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# 18 bodyparts in DLC output for the Connectome MPSA model.
BODYPARTS = [
    "Reference",
    "SATL", "SABL", "SABR", "SATR",
    "BOXL", "BOXR",
    "Pellet", "Pillar",
    "RightHand", "RHLeft", "RHOut", "RHRight",
    "Nose", "RightEar", "LeftEar",
    "LeftFoot", "TailBase",
]

# Smoothing window (frames). Centered moving average for position.
SMOOTH_WINDOW = 5
# Velocity finite-difference half-width (frames). Centered: f(t+h) - f(t-h).
VELOCITY_DT = 2
# Acceleration is finite-difference of velocity; reuses the same DT.
# Wider rolling windows for "did this happen near this frame" features.
SPEED_ROLLING_W1 = 21    # +/- 10 frames
SPEED_ROLLING_W2 = 41    # +/- 20 frames
LK_ROLLING_W = 21        # +/- 10 frames


def load_dlc_h5(dlc_path: Path) -> pd.DataFrame:
    """Load a DLC .h5 file into a flat dataframe with `bodypart_x`,
    `bodypart_y`, `bodypart_likelihood` columns. The existing
    `mousereach.reach.core.geometry.load_dlc` produces this shape.
    """
    from mousereach.reach.core.geometry import load_dlc
    return load_dlc(dlc_path)


def extract_features(dlc_df: pd.DataFrame) -> pd.DataFrame:
    """Build the v2 per-frame feature matrix from a flat DLC dataframe.

    See module docstring for the full schema.
    """
    features = {}
    n = len(dlc_df)
    bp_smooth_xy = {}  # for pairwise distances

    for bp in BODYPARTS:
        x_col = f"{bp}_x"
        y_col = f"{bp}_y"
        lk_col = f"{bp}_likelihood"

        if x_col not in dlc_df.columns:
            for suf in ("x", "y", "lk", "x_smooth", "y_smooth",
                        "vx", "vy", "ax", "ay", "speed", "dlk",
                        "speed_max20", "speed_max40", "lk_min20"):
                features[f"{bp}_{suf}"] = np.zeros(n, dtype=np.float32)
            bp_smooth_xy[bp] = (np.zeros(n, dtype=np.float32),
                                np.zeros(n, dtype=np.float32))
            continue

        x = dlc_df[x_col].to_numpy(dtype=np.float32)
        y = dlc_df[y_col].to_numpy(dtype=np.float32)
        lk = dlc_df[lk_col].to_numpy(dtype=np.float32)

        x_smooth = pd.Series(x).rolling(
            SMOOTH_WINDOW, center=True, min_periods=1).mean().to_numpy()
        y_smooth = pd.Series(y).rolling(
            SMOOTH_WINDOW, center=True, min_periods=1).mean().to_numpy()
        bp_smooth_xy[bp] = (x_smooth.astype(np.float32),
                            y_smooth.astype(np.float32))

        vx = _centered_diff(x_smooth, VELOCITY_DT)
        vy = _centered_diff(y_smooth, VELOCITY_DT)
        ax = _centered_diff(vx, VELOCITY_DT)
        ay = _centered_diff(vy, VELOCITY_DT)
        speed = np.sqrt(vx * vx + vy * vy)
        dlk = _centered_diff(lk, VELOCITY_DT)

        speed_max20 = pd.Series(speed).rolling(
            SPEED_ROLLING_W1, center=True, min_periods=1).max().to_numpy()
        speed_max40 = pd.Series(speed).rolling(
            SPEED_ROLLING_W2, center=True, min_periods=1).max().to_numpy()
        lk_min20 = pd.Series(lk).rolling(
            LK_ROLLING_W, center=True, min_periods=1).min().to_numpy()

        features[f"{bp}_x"] = x
        features[f"{bp}_y"] = y
        features[f"{bp}_lk"] = lk
        features[f"{bp}_x_smooth"] = x_smooth.astype(np.float32)
        features[f"{bp}_y_smooth"] = y_smooth.astype(np.float32)
        features[f"{bp}_vx"] = vx
        features[f"{bp}_vy"] = vy
        features[f"{bp}_ax"] = ax
        features[f"{bp}_ay"] = ay
        features[f"{bp}_speed"] = speed
        features[f"{bp}_dlk"] = dlk
        features[f"{bp}_speed_max20"] = speed_max20.astype(np.float32)
        features[f"{bp}_speed_max40"] = speed_max40.astype(np.float32)
        features[f"{bp}_lk_min20"] = lk_min20.astype(np.float32)

    # Pairwise distances on smoothed coordinates
    for i, bp_a in enumerate(BODYPARTS):
        xa, ya = bp_smooth_xy[bp_a]
        for bp_b in BODYPARTS[i + 1:]:
            xb, yb = bp_smooth_xy[bp_b]
            d = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2).astype(np.float32)
            features[f"dist__{bp_a}__{bp_b}"] = d

    out = pd.DataFrame(features, index=dlc_df.index)
    return out


def _centered_diff(arr: np.ndarray, dt: int) -> np.ndarray:
    """Centered finite difference: (arr[t+dt] - arr[t-dt]) / (2*dt).
    Edge frames are extended (zero-derivative pad).
    """
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    if n < 2 * dt + 1:
        return out
    out[dt:n - dt] = (arr[2 * dt:n] - arr[0:n - 2 * dt]) / (2.0 * dt)
    return out


def feature_columns() -> List[str]:
    """Return the canonical column order for the feature matrix.

    Useful to assert consistency between train and inference time.
    """
    suffixes = ["x", "y", "lk", "x_smooth", "y_smooth",
                "vx", "vy", "ax", "ay", "speed", "dlk",
                "speed_max20", "speed_max40", "lk_min20"]
    cols = []
    for bp in BODYPARTS:
        for suf in suffixes:
            cols.append(f"{bp}_{suf}")
    for i, bp_a in enumerate(BODYPARTS):
        for bp_b in BODYPARTS[i + 1:]:
            cols.append(f"dist__{bp_a}__{bp_b}")
    return cols
