"""
DLC bodypart impossibility filter -- shared cleaning utility.

Detects three classes of physically-impossible DLC predictions and
replaces them with the rolling median of nearby high-confidence
neighbors. Designed for use upstream of any signal that uses raw
bodypart positions (pillar geometry, paw motion, pellet position, etc.).

The impossibility predicates encode physical/optical realities of the
MouseReach apparatus:

  Tier 1 -- per-bodypart, lk-based.
    DLC declares uncertainty: any bodypart-frame where likelihood is
    below `lk_impossible_threshold` is flagged. These cases are
    typically 1-3 frame DLC failures where lk drops to ~0; the raw
    (x, y) prediction at those frames is meaningless.

  Tier 2 -- per-SA-bodypart rigid-body deviation.
    The 4 SA bodyparts (SABL, SABR, SATL, SATR) form an approximately
    rigid quadrilateral on the cycling tray. When all 4 are tracking
    correctly, each bodypart's position is consistent with the rolling
    median of its own recent confident detections. A single bodypart
    that deviates from its rolling median by more than
    `sa_deviation_threshold` px is the rigid body breaking; that
    bodypart at that frame is impossible regardless of its lk.

  Tier 3 -- per-frame whole-frame artifact.
    BOXL, BOXR, and Reference are mounted on the immovable enclosure
    box; they cannot physically move. Any frame where a static
    landmark moves more than `static_motion_threshold` px from the
    prior frame indicates a recording artifact (camera bump,
    compression glitch, dropped frame). Every bodypart in that frame
    is suspect.

For impossible bodypart-frame combinations, the raw (x, y) is
replaced by the rolling median of the surrounding window's
high-confidence neighbors. Likelihood values are NOT modified, so
downstream code can still apply lk-based filtering if desired.

Default thresholds were derived empirically across the 37 train-pool
videos (1.4M frames) of the v4.0.0_dev_walkthrough quarantine corpus
on 2026-05-01. See `apparatus_motion_physics.md` memory for the
diagnostic motion-pattern model.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


# Tier 1: lk threshold below which the bodypart-frame is impossible.
DEFAULT_LK_IMPOSSIBLE = 0.5

# Tier 1+ helper: lk threshold for "this neighbor is good enough to
# contribute to a rolling median that fills holes."
DEFAULT_LK_TRUSTWORTHY = 0.7

# Tier 2: max px deviation of a single SA bodypart from its own rolling
# median before that bodypart-frame is considered impossible.
# Empirically: stationary p99.9 across SA bodyparts is 4-8 px;
# moving (real tray cycle) max is 5 px; sets the floor for "much more
# than normal jitter."
DEFAULT_SA_DEVIATION_THRESHOLD = 15.0

# Tier 3: per-frame motion threshold for static landmarks. Above this,
# the entire frame is treated as a recording artifact. Empirically:
# BOXL/BOXR/Reference p99.9 frame-to-frame motion is <=3.4 px; the
# distribution is sharply bimodal with the artifact tail jumping into
# 100s of px, so 5 px sits cleanly above the noise floor.
DEFAULT_STATIC_MOTION_THRESHOLD = 5.0

# Rolling-median window used both for impossibility detection and for
# the fill value at impossible frames. 5 frames handles 1-2 consecutive
# bad frames cleanly; longer windows blur real apparatus motion.
DEFAULT_MEDIAN_WINDOW = 5

DEFAULT_SA_BODYPARTS = ("SABL", "SABR", "SATL", "SATR")
DEFAULT_STATIC_LANDMARKS = ("BOXL", "BOXR", "Reference")


def detect_recording_artifact_frames(
    dlc_df: pd.DataFrame,
    static_landmarks: Iterable[str] = DEFAULT_STATIC_LANDMARKS,
    motion_threshold: float = DEFAULT_STATIC_MOTION_THRESHOLD,
    lk_trustworthy: float = DEFAULT_LK_TRUSTWORTHY,
) -> np.ndarray:
    """Per-frame boolean mask: is this frame a whole-frame recording
    artifact (Tier 3)?

    A frame is flagged when any provided static landmark moved more
    than `motion_threshold` px from the prior frame, where both
    adjacent frames had likelihood >= `lk_trustworthy` for that
    landmark. The likelihood gate prevents counting low-lk noise as
    motion.
    """
    n = len(dlc_df)
    if n == 0:
        return np.zeros(0, dtype=bool)
    artifact = np.zeros(n, dtype=bool)
    for bp in static_landmarks:
        if f"{bp}_x" not in dlc_df.columns:
            continue
        x = dlc_df[f"{bp}_x"].to_numpy(dtype=float)
        y = dlc_df[f"{bp}_y"].to_numpy(dtype=float)
        lk = dlc_df[f"{bp}_likelihood"].to_numpy(dtype=float)
        dx = np.zeros(n)
        dy = np.zeros(n)
        dx[1:] = np.diff(x)
        dy[1:] = np.diff(y)
        d = np.sqrt(dx * dx + dy * dy)
        # Both ends of the diff need to be high-lk to count as real motion
        both_lk = np.zeros(n, dtype=bool)
        both_lk[1:] = (lk[:-1] >= lk_trustworthy) & (lk[1:] >= lk_trustworthy)
        artifact |= (d > motion_threshold) & both_lk
    return artifact


def _rolling_median_with_lk_mask(
    x: np.ndarray,
    y: np.ndarray,
    lk: np.ndarray,
    window: int,
    lk_trustworthy: float,
) -> tuple:
    """Return (x_med, y_med) where each is the centered rolling median
    of (x, y) using only frames whose lk meets `lk_trustworthy`. NaN
    is returned for frames where no in-window neighbor passes the lk
    gate.
    """
    x_masked = np.where(lk >= lk_trustworthy, x, np.nan)
    y_masked = np.where(lk >= lk_trustworthy, y, np.nan)
    x_med = pd.Series(x_masked).rolling(
        window, center=True, min_periods=1).median().to_numpy()
    y_med = pd.Series(y_masked).rolling(
        window, center=True, min_periods=1).median().to_numpy()
    return x_med, y_med


def clean_dlc_bodyparts(
    dlc_df: pd.DataFrame,
    sa_bodyparts: Iterable[str] = DEFAULT_SA_BODYPARTS,
    other_bodyparts_to_clean: Optional[Iterable[str]] = None,
    static_landmarks: Iterable[str] = DEFAULT_STATIC_LANDMARKS,
    lk_impossible: float = DEFAULT_LK_IMPOSSIBLE,
    lk_trustworthy: float = DEFAULT_LK_TRUSTWORTHY,
    sa_deviation_threshold: float = DEFAULT_SA_DEVIATION_THRESHOLD,
    static_motion_threshold: float = DEFAULT_STATIC_MOTION_THRESHOLD,
    median_window: int = DEFAULT_MEDIAN_WINDOW,
) -> pd.DataFrame:
    """Return a copy of `dlc_df` with impossible bodypart-frame
    positions replaced by rolling-median fills.

    Parameters
    ----------
    dlc_df
        DLC dataframe with columns `{bp}_x`, `{bp}_y`, `{bp}_likelihood`
        for each bodypart.
    sa_bodyparts
        SA quadrilateral bodyparts to clean with the full 3-tier
        filter (lk, rigid-body deviation, frame artifact).
    other_bodyparts_to_clean
        Additional bodyparts to clean with Tier 1 + Tier 3 only
        (no rigid-body check, since they are not part of the SA
        quadrilateral). If None, nothing else is cleaned.
    static_landmarks
        Bodyparts used to detect Tier 3 (whole-frame artifact).
        These are NOT cleaned by this function -- they are reference
        only. The caller is expected to leave their raw values intact.
    lk_impossible
        Tier 1 threshold: bodypart-frames with likelihood below this
        are impossible.
    lk_trustworthy
        Likelihood threshold for "good enough to contribute to a
        rolling median fill value." Should be >= lk_impossible.
    sa_deviation_threshold
        Tier 2 threshold: SA bodypart deviation from its rolling
        median in px.
    static_motion_threshold
        Tier 3 threshold: static landmark frame-to-frame motion in px.
    median_window
        Centered rolling window length for both impossibility detection
        (Tier 2) and fill value computation.

    Returns
    -------
    Cleaned DataFrame (copy). Likelihood columns are not modified;
    only x/y coordinates of impossible bodypart-frames are replaced.
    """
    out = dlc_df.copy()
    n = len(dlc_df)
    if n == 0:
        return out

    # Tier 3: whole-frame recording artifact mask
    artifact_frame = detect_recording_artifact_frames(
        dlc_df,
        static_landmarks=static_landmarks,
        motion_threshold=static_motion_threshold,
        lk_trustworthy=lk_trustworthy,
    )

    # Helper to clean one bodypart
    def _clean_bp(bp: str, apply_rigid_body_check: bool):
        x = dlc_df[f"{bp}_x"].to_numpy(dtype=float)
        y = dlc_df[f"{bp}_y"].to_numpy(dtype=float)
        lk = dlc_df[f"{bp}_likelihood"].to_numpy(dtype=float)
        x_med, y_med = _rolling_median_with_lk_mask(
            x, y, lk, median_window, lk_trustworthy)

        # Tier 1: low lk
        impossible = lk < lk_impossible

        # Tier 2: deviation from rolling median (only for SA points)
        if apply_rigid_body_check:
            deviation = np.sqrt((x - x_med) ** 2 + (y - y_med) ** 2)
            # If x_med is NaN (no good neighbors in window), skip the
            # deviation check at that frame -- can't reliably detect.
            valid_med = ~np.isnan(x_med)
            tier2 = valid_med & (deviation > sa_deviation_threshold)
            impossible |= tier2

        # Tier 3: whole-frame artifact (applies to all bodyparts)
        impossible |= artifact_frame

        # Fill: use rolling median; if median is NaN, leave raw value
        x_clean = np.where(
            impossible & ~np.isnan(x_med), x_med, x)
        y_clean = np.where(
            impossible & ~np.isnan(y_med), y_med, y)
        return x_clean, y_clean, impossible.sum()

    summary = {}
    for bp in sa_bodyparts:
        if f"{bp}_x" not in dlc_df.columns:
            continue
        x_clean, y_clean, n_replaced = _clean_bp(bp, apply_rigid_body_check=True)
        out[f"{bp}_x"] = x_clean
        out[f"{bp}_y"] = y_clean
        summary[bp] = int(n_replaced)

    if other_bodyparts_to_clean:
        for bp in other_bodyparts_to_clean:
            if f"{bp}_x" not in dlc_df.columns:
                continue
            x_clean, y_clean, n_replaced = _clean_bp(
                bp, apply_rigid_body_check=False)
            out[f"{bp}_x"] = x_clean
            out[f"{bp}_y"] = y_clean
            summary[bp] = int(n_replaced)

    # Stash diagnostics on the dataframe attrs for caller introspection
    out.attrs["dlc_cleaning_summary"] = summary
    out.attrs["dlc_cleaning_artifact_frames"] = int(artifact_frame.sum())
    return out
