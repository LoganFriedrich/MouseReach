"""The ``legacy_swipe`` measures: an earlier swipe-measure definition, kept for continuity.

WHY THIS EXISTS. Before this pipeline, reaching was measured by an earlier program
that called each reach a "swipe" and described it with seven numbers of its own
(breadth, length, area, path length, speed, duration in frames, path over frames).
Analyses built on those numbers can only be continued, or compared with new data,
if the same numbers are computed the same way. These columns reproduce that
definition exactly; they are not a better measurement of the reach, and the
pipeline's own kinematics (the rest of the ``extended`` block) are not replaced by
them. Read them as "what the earlier definition gives for this reach".

WHAT IS THE SAME AS THE EARLIER DEFINITION
  * The per-frame test. A frame of the reach counts only when ALL of these hold:
      - RightHand likelihood > 0.7 (``RIGHTHAND_MIN_LIKELIHOOD``)
      - Nose likelihood > 0.5 (``NOSE_MIN_LIKELIHOOD``)
      - the nose is in the slit zone, set from the Reference point read at the
        pellet window's first frame and truncated to whole pixels:
        Nose_y > ref_y - 80, and ref_x + 70 < Nose_x < ref_x + 100
      - the paw is more than 8 px past the nose: RightHand_y - Nose_y > 8
      - RightHand_y > y_max - 300, where y_max is the largest RightHand_y over the
        pellet window's frames with RightHand likelihood > 0.95 (the earlier
        program's "threshold" line; with no such frame no frame counts)
  * Only the RightHand point, as tracked: no smoothing, no interpolation, no
    anchoring at the nose. That is deliberately unlike every other path measure in
    this extractor (anchored_path.py), because the point is to match the old numbers.
  * The formulas, on the counted frames in frame order (x, y = RightHand):
      - duration_frames  = number of counted frames
      - first_frame, last_frame = the first and last counted frame (provenance:
                           which frames the values below were taken over)
      - breadth_px      = max(x) - min(x)
      - length_px        = max(y) - Nose_y in the frame holding max(y) (first such
                           frame); empty when that nose is beyond the paw
      - area_px2         = shoelace area of the polygon through the (x, y) points
      - path_length_px   = sum of straight-line steps between consecutive counted frames
      - speed_px_div4_per_sec = (path_length_px / 4) / ((last - first frame) / frame rate)
      - path_over_frames_px_per_frame = path_length_px / duration_frames
    Every measure is empty when fewer than two frames count, as in the earlier
    definition; duration_frames is still given (0 or 1), and first_frame and
    last_frame whenever one frame counts.

WHAT IS DIFFERENT, AND WHY
  * The start and end of the reach come from this pipeline's reach detector, not
    from the earlier program's own grouping of frames into swipes. The per-frame
    test above is applied inside the detected reach.
  * Speed divides by the video's own frame rate (frame_rate.py), where the earlier
    program divided by a constant 60. On a 60 frames-per-second video the two are
    identical, number for number; on any other rate this one is the true per-second
    value and the constant would not be.
  * Units are kept as the earlier program kept them: pixels. Its divisor of 4 in the
    speed (a nominal 4 pixels per mm for the video format it was written for) is
    kept so the number is the earlier one, and the column name says so
    (``px_div4``): it is NOT a calibrated millimetre value. A calibrated mm value of
    any measure can be had from the ruler (ruler_pixels on the segment) if needed.

WHAT IS NOT HERE. The earlier definition's per-session "Total Swipes" is a count of
reaches per animal per session. It spans every video of the session, so no single
video's features file can hold it: it is derived where the session is assembled
(downstream, from the per-reach rows), not here.

WHERE THE COLUMNS GO. Into each reach's ``extended`` block, so they are written into
every ``<video>_features.json`` and carried by anything that carries that block.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

PREFIX = "legacy_swipe_"

RIGHTHAND_MIN_LIKELIHOOD = 0.7
NOSE_MIN_LIKELIHOOD = 0.5
# The slit zone, relative to the Reference point (pixels).
NOSE_ABOVE_REF_MAX = 80        # Nose_y > ref_y - 80
NOSE_X_MIN_RIGHT_OF_REF = 70   # Nose_x > ref_x + 70
NOSE_X_MAX_RIGHT_OF_REF = 100  # Nose_x < ref_x + 100
PAW_PAST_NOSE_MIN_PX = 8       # RightHand_y - Nose_y > 8
# The earlier program's threshold line: y_max - 300, y_max from confident frames.
THRESHOLD_BELOW_YMAX_PX = 300
YMAX_MIN_LIKELIHOOD = 0.95
# The earlier program's nominal pixels per mm, kept only inside the speed.
NOMINAL_PX_PER_MM = 4.0

KEYS = (
    PREFIX + "duration_frames",
    PREFIX + "first_frame",
    PREFIX + "last_frame",
    PREFIX + "breadth_px",
    PREFIX + "length_px",
    PREFIX + "area_px2",
    PREFIX + "path_length_px",
    PREFIX + "speed_px_div4_per_sec",
    PREFIX + "path_over_frames_px_per_frame",
)

_NEEDED = ("RightHand_x", "RightHand_y", "RightHand_likelihood",
           "Nose_x", "Nose_y", "Nose_likelihood", "Reference_x", "Reference_y")


def _empty() -> Dict[str, Optional[float]]:
    return {k: None for k in KEYS}


def _num(v) -> Optional[float]:
    """A plain float for JSON, or None for a missing or non-finite value."""
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def legacy_swipe_measures(df: pd.DataFrame, start_frame: int, end_frame: int,
                          window_start: int, window_end: int,
                          frame_rate: Optional[float]) -> Dict[str, Optional[float]]:
    """The legacy_swipe values of one reach (module docstring says what and why).

    ``df`` is the pose table with flattened columns (``RightHand_x`` ...), indexed by
    position = frame. ``start_frame``/``end_frame`` bound the reach,
    ``window_start``/``window_end`` the pellet window (segment) it lies in.

    A pose without a needed point, or a Reference point with no position at the
    window's first frame, gives every value empty (the definition cannot be applied
    there). Anything else that goes wrong is not caught here: it reaches the
    extractor, which holds the video rather than file it with these values silently
    missing.
    """
    out = _empty()
    if any(c not in df.columns for c in _NEEDED):
        return out
    n = len(df)
    if n == 0 or not (0 <= window_start < n):
        return out
    ref_x_raw = df["Reference_x"].iloc[window_start]
    ref_y_raw = df["Reference_y"].iloc[window_start]
    if not (np.isfinite(ref_x_raw) and np.isfinite(ref_y_raw)):
        return out
    # Truncated to whole pixels, as the earlier program did (int()).
    ref_x, ref_y = int(ref_x_raw), int(ref_y_raw)

    rhx = df["RightHand_x"].to_numpy(dtype=float)
    rhy = df["RightHand_y"].to_numpy(dtype=float)
    rhl = df["RightHand_likelihood"].to_numpy(dtype=float)
    nx = df["Nose_x"].to_numpy(dtype=float)
    ny = df["Nose_y"].to_numpy(dtype=float)
    nl = df["Nose_likelihood"].to_numpy(dtype=float)

    w = slice(max(0, window_start), min(n, window_end + 1))
    confident = rhl[w] > YMAX_MIN_LIKELIHOOD
    if confident.any():
        y_max = float(np.nanmax(rhy[w][confident]))
    else:
        y_max = float("nan")  # no frame can pass "> nan": none counts

    f = np.arange(max(0, start_frame), min(n, end_frame + 1))
    with np.errstate(invalid="ignore"):
        ok = ((rhl[f] > RIGHTHAND_MIN_LIKELIHOOD)
              & (nl[f] > NOSE_MIN_LIKELIHOOD)
              & (ny[f] > ref_y - NOSE_ABOVE_REF_MAX)
              & (nx[f] > ref_x + NOSE_X_MIN_RIGHT_OF_REF)
              & (nx[f] < ref_x + NOSE_X_MAX_RIGHT_OF_REF)
              & (rhy[f] - ny[f] > PAW_PAST_NOSE_MIN_PX)
              & (rhy[f] > y_max - THRESHOLD_BELOW_YMAX_PX))
    f = f[ok]

    out[PREFIX + "duration_frames"] = int(len(f))
    if len(f):
        out[PREFIX + "first_frame"] = int(f[0])
        out[PREFIX + "last_frame"] = int(f[-1])
    if len(f) <= 1:
        return out

    x, y = rhx[f], rhy[f]
    my = y.max()
    nose_at = ny[f][int(np.flatnonzero(y == my)[0])]
    if not nose_at > my:
        out[PREFIX + "length_px"] = _num(my - nose_at)
    out[PREFIX + "breadth_px"] = _num(x.max() - x.min())
    out[PREFIX + "area_px2"] = _num(
        0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    path = float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)))
    out[PREFIX + "path_length_px"] = _num(path)
    out[PREFIX + "path_over_frames_px_per_frame"] = _num(path / len(f))
    if frame_rate:
        seconds = (f[-1] - f[0]) / float(frame_rate)
        if seconds > 0:
            out[PREFIX + "speed_px_div4_per_sec"] = _num((path / NOMINAL_PX_PER_MM) / seconds)
    return out
