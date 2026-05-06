"""
Causal attribution primitives -- end-state classification, off-pillar
transition back-walk, displaced-signature search.

These follow the decision rules in
`feature_philosophy_event_anchored_walking.md`:
- Displaced signature is segment-scoped (no off-paw filter)
- Retrieved signature is "displaced never fires + pellet absent + pillar
  jointly revealed"
- Causal reach is found by BACK-WALKING from a validated end state, not
  by forward scanning

All primitives use per-frame smoothed-SA pillar geometry from
`mousereach.lib.pillar_geometry`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from mousereach.lib.pillar_geometry import (
    compute_pillar_geometry_series, pellet_inside_pillar_circle,
    pellet_dist_from_pillar_center,
)

PELLET_HIGH_LK = 0.95
PELLET_VISIBLE_LK = 0.7
PELLET_MISSING_LK = 0.3
PILLAR_VISIBLE_LK = 0.7
SMOOTH_TRAJECTORY_PX = 3.0  # frame-to-frame jump tolerance for "smooth"
DISPLACED_MIN_RUN = 3
RETRIEVED_MIN_MISSING_RUN = 30
END_STATE_WINDOW = 30
APEX_PAW_BODYPART = "RightHand"  # peak nose-to-paw distance


@dataclass
class EndStateResult:
    classification: str   # "on_pillar" | "off_pillar_visible" | "missing" | "ambiguous"
    last_30f_frame_range: Tuple[int, int]
    pellet_inside_pillar_count: int
    pellet_off_pillar_visible_count: int
    pellet_missing_count: int
    pillar_revealed_count: int


def compute_reach_apex(
    reach_start: int,
    reach_end: int,
    dlc_df: pd.DataFrame,
    paw_bodypart: str = APEX_PAW_BODYPART,
) -> int:
    """Return the apex frame within [reach_start, reach_end].

    Apex = argmax over reach window of cartesian distance(nose, paw).
    The user-mandated definition: it is the moment of peak paw extension
    away from the head, regardless of where the pellet is.
    """
    if reach_end < reach_start:
        return reach_start
    sub = dlc_df.iloc[reach_start:reach_end + 1]
    nose_x = sub["Nose_x"].to_numpy(dtype=float)
    nose_y = sub["Nose_y"].to_numpy(dtype=float)
    paw_x = sub[f"{paw_bodypart}_x"].to_numpy(dtype=float)
    paw_y = sub[f"{paw_bodypart}_y"].to_numpy(dtype=float)
    d = np.sqrt((paw_x - nose_x) ** 2 + (paw_y - nose_y) ** 2)
    if not len(d):
        return reach_start
    apex_idx_local = int(np.nanargmax(d))
    return reach_start + apex_idx_local


def classify_end_state(
    dlc_df: pd.DataFrame,
    segment_end: int,
    window: int = END_STATE_WINDOW,
) -> EndStateResult:
    """Classify pellet end state in last `window` frames before segment_end.

    Per-frame booleans aggregated:
    - pellet_inside_pillar_circle (lk >= 0.7 AND inside circle)
    - pellet_off_pillar_visible (lk >= 0.95 AND outside circle)
    - pellet_missing (lk < 0.3)
    - pillar_revealed (lk >= 0.7 -- proxy for joint pillar quality;
      true joint check could compare to expected position too)

    Classification heuristic on these counts:
    - on_pillar  if pellet_inside_pillar_count is the dominant signal
    - off_pillar_visible if pellet_off_pillar_visible_count is dominant
    - missing  if pellet_missing_count is dominant + pillar_revealed
    - ambiguous otherwise
    """
    s = max(0, segment_end - window + 1)
    e = segment_end + 1
    sub = dlc_df.iloc[s:e]
    if len(sub) == 0:
        return EndStateResult("ambiguous", (s, e - 1), 0, 0, 0, 0)

    geom = compute_pillar_geometry_series(sub)
    inside = pellet_inside_pillar_circle(sub, pillar_geom=geom).to_numpy()
    plk = sub["Pellet_likelihood"].to_numpy(dtype=float)
    pillar_lk = sub["Pillar_likelihood"].to_numpy(dtype=float)
    dist = pellet_dist_from_pillar_center(sub, pillar_geom=geom).to_numpy()
    radius = geom["pillar_r"].to_numpy()

    inside_count = int(inside.sum())
    off_pillar_visible = (plk >= PELLET_HIGH_LK) & (dist > radius)
    off_pillar_visible_count = int(off_pillar_visible.sum())
    missing = plk < PELLET_MISSING_LK
    missing_count = int(missing.sum())
    pillar_revealed_count = int((pillar_lk >= PILLAR_VISIBLE_LK).sum())

    n = len(sub)
    if inside_count >= 0.5 * n:
        cls = "on_pillar"
    elif off_pillar_visible_count >= 0.3 * n:
        cls = "off_pillar_visible"
    elif missing_count >= 0.5 * n and pillar_revealed_count >= 0.5 * n:
        cls = "missing"
    else:
        cls = "ambiguous"

    return EndStateResult(
        classification=cls,
        last_30f_frame_range=(s, e - 1),
        pellet_inside_pillar_count=inside_count,
        pellet_off_pillar_visible_count=off_pillar_visible_count,
        pellet_missing_count=missing_count,
        pillar_revealed_count=pillar_revealed_count,
    )


def find_displaced_signature_runs(
    dlc_df: pd.DataFrame,
    segment_start: int,
    segment_end: int,
    min_run: int = DISPLACED_MIN_RUN,
    high_lk: float = PELLET_HIGH_LK,
    smooth_px: float = SMOOTH_TRAJECTORY_PX,
) -> dict:
    """Search the segment for displaced-signature runs.

    A frame qualifies if:
    - pellet at lk >= high_lk
    - pellet position OUTSIDE the calculated pillar circle
    - frame-to-frame position jump <= smooth_px from previous qualifying
      frame (smooth trajectory)

    Note: NO off-paw filter. The paw can occlude the displaced pellet
    later in the segment; that does not invalidate earlier displaced
    detections.

    Returns dict with:
    - n_qualifying_frames
    - longest_run_length
    - first_run_start_frame (or None)
    """
    s = max(0, segment_start)
    e = min(len(dlc_df) - 1, segment_end)
    if e < s:
        return {"n_qualifying_frames": 0, "longest_run_length": 0,
                "first_run_start_frame": None}
    sub = dlc_df.iloc[s:e + 1]
    geom = compute_pillar_geometry_series(sub)
    plk = sub["Pellet_likelihood"].to_numpy(dtype=float)
    px = sub["Pellet_x"].to_numpy(dtype=float)
    py = sub["Pellet_y"].to_numpy(dtype=float)
    cx = geom["pillar_cx"].to_numpy()
    cy = geom["pillar_cy"].to_numpy()
    r = geom["pillar_r"].to_numpy()
    pellet_dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    visible_off_pillar = (plk >= high_lk) & (pellet_dist > r) & np.isfinite(pellet_dist)

    # Walk through and find runs that maintain smooth trajectory
    n_qualifying = 0
    longest = 0
    first_start = None
    current_run = 0
    last_x, last_y = None, None
    for i in range(len(sub)):
        if visible_off_pillar[i]:
            if last_x is not None and last_y is not None:
                jump = np.sqrt((px[i] - last_x) ** 2 + (py[i] - last_y) ** 2)
                if jump <= smooth_px:
                    current_run += 1
                else:
                    # Trajectory broken -- restart
                    current_run = 1
            else:
                current_run = 1
            last_x, last_y = px[i], py[i]
            n_qualifying += 1
            if current_run >= min_run:
                if first_start is None:
                    first_start = s + i - current_run + 1
                longest = max(longest, current_run)
        else:
            current_run = 0
            last_x, last_y = None, None

    return {
        "n_qualifying_frames": n_qualifying,
        "longest_run_length": longest,
        "first_run_start_frame": first_start,
    }


def find_off_pillar_transition_frame(
    dlc_df: pd.DataFrame,
    segment_start: int,
    segment_end: int,
    end_state: EndStateResult,
) -> Optional[int]:
    """Back-walk from validated end state to find when pellet transitioned
    from on-pillar to not-on-pillar.

    If end state is on_pillar -> no transition; return None.
    Otherwise: walk backward from segment_end, find the most recent frame
    where pellet was confidently inside the pillar circle. Return the
    NEXT frame (the first off-pillar frame that started the transition).

    If pellet was never confidently on pillar in the segment, return None
    (means we cannot attribute a transition frame -- either pellet
    wasn't there to start with, or DLC failed).
    """
    if end_state.classification == "on_pillar":
        return None
    if end_state.classification == "ambiguous":
        return None

    s = max(0, segment_start)
    e = min(len(dlc_df) - 1, segment_end)
    if e < s:
        return None
    sub = dlc_df.iloc[s:e + 1]
    inside = pellet_inside_pillar_circle(sub).to_numpy()
    # Walk backward
    last_inside_local = -1
    for i in range(len(sub) - 1, -1, -1):
        if inside[i]:
            last_inside_local = i
            break
    if last_inside_local < 0:
        return None
    # Transition is the next frame after the last confirmed on-pillar frame
    transition_local = last_inside_local + 1
    if transition_local >= len(sub):
        return None
    return s + transition_local


def reach_contains_or_precedes_transition(
    apex_frame: int,
    reach_start: int,
    reach_end: int,
    transition_frame: Optional[int],
    pre_apex_pad: int = 5,
) -> bool:
    """True iff the transition occurs within this reach's apex window
    (apex - pre_apex_pad to reach_end + small offset).

    None transition -> False.
    """
    if transition_frame is None:
        return False
    lo = apex_frame - pre_apex_pad
    hi = reach_end + pre_apex_pad
    return lo <= transition_frame <= hi


# ---------------------------------------------------------------------------
# Tray-motion-onset detection -- gives the natural outcome_known_frame for
# the "untouched" hypothesis. Per `untouched_outcome_known_frame_derivation`:
# the last stable frame before SA bodyparts start moving (tray cycling).
# ---------------------------------------------------------------------------

SA_BODYPARTS = ("SABL", "SABR", "SATL", "SATR")
TRAY_MOTION_PX_THRESHOLD = 5.0
TRAY_MOTION_LOOKBACK_FRAMES = 5


def detect_tray_motion_onset(
    dlc_df: pd.DataFrame,
    segment_end: int,
    walk_back: int = 60,
) -> Optional[int]:
    """Walk backward from segment_end up to `walk_back` frames; return
    the frame at which SA bodyparts first stop moving (going backward
    from segment_end in time, this is the last frame that's still
    "jumpy" -- one frame later is the first stable one looking forward,
    or equivalently one frame earlier is the last-stable from the
    segment-end perspective).

    Implementation: at each candidate frame f, compare SA positions to
    SA positions at f - LOOKBACK. If max displacement across the four
    SA points is > THRESHOLD, the tray is moving at f. Walk backward
    until that condition becomes false; that's the last stable frame.

    Returns None if SA tracking is unreliable in the lookback range.
    """
    s = max(TRAY_MOTION_LOOKBACK_FRAMES,
            segment_end - walk_back + 1)
    if s > segment_end:
        return None

    # We walk f from segment_end going backward.
    last_unstable = None
    for f in range(segment_end, s - 1, -1):
        base = f - TRAY_MOTION_LOOKBACK_FRAMES
        if base < 0:
            break
        max_disp = 0.0
        ok = True
        for bp in SA_BODYPARTS:
            x_col = f"{bp}_x"
            y_col = f"{bp}_y"
            if x_col not in dlc_df.columns:
                ok = False
                break
            dx = dlc_df[x_col].iloc[f] - dlc_df[x_col].iloc[base]
            dy = dlc_df[y_col].iloc[f] - dlc_df[y_col].iloc[base]
            d = float(np.sqrt(dx * dx + dy * dy))
            if d > max_disp:
                max_disp = d
        if not ok:
            continue
        if max_disp > TRAY_MOTION_PX_THRESHOLD:
            last_unstable = f
        else:
            # Stable. If we previously saw any unstable frame, the
            # transition (motion onset working forward in time) was
            # at last_unstable. The "last stable frame before motion"
            # is f.
            if last_unstable is not None:
                return f
            else:
                # No motion observed up to this point yet -- keep looking
                continue

    # Edge cases:
    # - If we never observed instability, the tray doesn't seem to be
    #   moving in this window; return segment_end as the safe answer.
    # - If we observed instability throughout the walk, return the
    #   earliest frame we examined (s) as our best stable candidate
    #   (might be wrong but it's the best within the walk_back budget).
    if last_unstable is None:
        return segment_end
    return s

