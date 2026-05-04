"""
Conservative pellet DLC gap imputer.

Fills SHORT gaps between high-confidence pellet detections when the pre-
and post-gap positions are similar (= pellet didn't move during the
gap, just DLC dropout). Preserves all high-confidence detections
exactly. Never smooths across position discontinuities (= real
displacement events).

Design constraints (per user 2026-05-03):
  - Don't touch high-conf detections (lk >= confident_threshold)
  - Don't smooth across position discontinuities (gap-pre vs gap-post
    distance must be small)
  - Only fill SHORT gaps (length <= max_gap_frames)
  - Imputed frames get lk = 0.96 (above 0.95 threshold but flagged as
    not-original)
  - Linear interpolation between pre and post positions

Used as a LATE-STAGE pre-processor: feed stabilized DLC to a retry
stage that re-evaluates residual segments. We don't apply stabilization
to the main cascade -- only to cases that fell through.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def stabilize_pellet_dlc(
    dlc_df: pd.DataFrame,
    confident_threshold: float = 0.95,
    max_gap_frames: int = 10,
    position_tolerance_px: float = 15.0,
) -> pd.DataFrame:
    """Return a copy of dlc_df with short pellet-confidence gaps imputed.

    A "gap" is a contiguous run of frames where Pellet_likelihood <
    confident_threshold, bounded by frames at or above the threshold.
    Gaps are imputed only if:
      - gap length <= max_gap_frames
      - distance between bounding-frame pellet positions <=
        position_tolerance_px (so we're not smoothing across a real
        position change)

    Imputed frames get linearly interpolated x/y, and lk = 0.96.
    """
    df = dlc_df.copy()
    if "Pellet_likelihood" not in df.columns:
        return df

    n = len(df)
    pellet_lk = df["Pellet_likelihood"].to_numpy(dtype=float).copy()
    pellet_x = df["Pellet_x"].to_numpy(dtype=float).copy()
    pellet_y = df["Pellet_y"].to_numpy(dtype=float).copy()

    confident = pellet_lk >= confident_threshold
    if not confident.any():
        return df  # nothing to anchor against

    # Walk through and find gaps bounded by confident detections.
    in_gap = False
    gap_start = -1
    n_imputed = 0
    n_imputed_frames = 0
    for i in range(n):
        if confident[i]:
            if in_gap and gap_start > 0:
                # Gap closed at i; gap is [gap_start, i-1].
                gap_len = i - gap_start
                if gap_len <= max_gap_frames:
                    pre_idx = gap_start - 1
                    post_idx = i
                    pre_x, pre_y = pellet_x[pre_idx], pellet_y[pre_idx]
                    post_x, post_y = pellet_x[post_idx], pellet_y[post_idx]
                    d = float(np.sqrt((post_x - pre_x) ** 2
                                      + (post_y - pre_y) ** 2))
                    if d <= position_tolerance_px:
                        # Linear interpolate.
                        for k in range(gap_start, i):
                            t = (k - pre_idx) / (post_idx - pre_idx)
                            pellet_x[k] = pre_x + t * (post_x - pre_x)
                            pellet_y[k] = pre_y + t * (post_y - pre_y)
                            pellet_lk[k] = 0.96
                            n_imputed_frames += 1
                        n_imputed += 1
            in_gap = False
            gap_start = -1
        else:
            if not in_gap and i > 0 and confident[:i].any():
                # Start of a gap (we have at least one prior confident
                # frame to bound against; otherwise no anchor).
                in_gap = True
                gap_start = i
            elif in_gap:
                pass  # still in gap

    df["Pellet_likelihood"] = pellet_lk
    df["Pellet_x"] = pellet_x
    df["Pellet_y"] = pellet_y
    return df
