"""The path of a hand point through a reach, with both ends placed where the hand is.

THE RULE. A reach starts and ends at the nose. Before the first frame a reach is
detected, the hand is on its way out from beside the nose, hidden behind the box
wall, so tracking cannot see it. Just after the paw disappears it is back there,
hidden again. So every calculation that cares where the paw is must treat these
two frames as part of the reach:

    frame before detection (start_frame - 1):
        every hand point is at the START POINT, between the left slit corner
        (BOXL) and the nose.
    frame after the paw disappears (end_frame + 1):
        every hand point is at the END POINT, between the right slit corner
        (BOXR) and the nose.

WHY IT MATTERS. Measured only over detected frames, a path begins wherever the
tracker first noticed the paw and stops wherever it last saw it. Those places vary
with the camera, the tracking and the boundary frame, so a reach would look shorter
and straighter than it was, by different amounts in different animals. The two ends
are physical facts, so they are stated here rather than left to the tracker.

WHERE THE POINTS SIT. ``ANCHOR_FRACTION_FROM_NOSE`` is the one place that says how
far along the line from nose to slit corner each point sits: one half, the midpoint.

WHICH MEASURES LEAVE THE ANCHOR STEPS OUT. The path, its length and its straightness
run nose to nose, anchors included. Peak speed and speed at the apex do not: they
read ``tracked_steps``, which drops the first and last step. Those two steps are a
hand nobody saw, moved between a calculated point and the first or last detected
position and charged to a single frame however long the trip behind the wall really
took. Their length is a statement about geometry, not a speed. A peak taken over them
reports that jump on about half of all reaches, and a speed "at the apex" taken from
one says nothing about the apex. So a value that claims to be how fast the hand was
moving comes only from steps between two detected frames, and needs no anchor at all.

REPEATED FRAMES. A recording program can write frames faster than its cameras take
pictures; it then repeats the last picture until a new one arrives. The video's frame
rate is how often a frame is shown, not how often the hand was seen. On a repeated
frame the tracked hand has not moved, and on the next new frame it has moved for two
(or more) frame times at once. A speed read step by step therefore swings between
nothing and double: the peak is the double, the speed at the apex is one or the other
depending on which frame the apex fell on, and jerk and the agreement between the paw
points' speeds measure the repetition rather than the hand. ``fill_repeated_frames``
puts each repeated frame where the hand passed between the two new pictures around it,
so a speed is the distance between two pictures over the frames between them. It
assumes no fixed pattern (one repeat, then two, then none are all handled), but a
recorder repeats a picture for a frame or two, not longer: a point that stays put for
more frames than ``MAX_REPEATED_FRAMES`` is a hand that is really still, and is left
as tracked, so a pause followed by a move keeps the move's own speed. The one change
on a video with no repeats: a point that moves less than ``REPEATED_FRAME_MAX_PX`` for
one or two frames has those frames' movement evened out with the step that follows.
Only the speed measures (peak, apex, jerk, velocity correlation) read it; positions,
the path, its length and straightness read the frames as tracked.

WHEN A LANDMARK CANNOT BE TRUSTED. An anchor is never invented silently. Each reach
records in ``start_source`` and ``end_source`` how its anchor was placed:

    "anchor_frame"   nose and slit corner both trusted in that very frame.
    "nearest_frame"  a landmark was not trusted there, so its nearest trusted
                     position within ANCHOR_SEARCH_FRAMES frames was used.
    "video_edge"     the anchor frame is outside the video (a reach starting on
                     frame 0, or ending on the last frame), so the nearest frame
                     inside the video stands in for it.
    "none"           no trusted landmark in range. There is no anchor, ``ok`` is
                     False, and every path measurement of that reach is empty
                     rather than measured from a guess.

A NEIGHBOURING REACH IN THE ANCHOR FRAME. Two detected reaches can touch, so the paw
is visible in what would be the anchor frame. Only the nose and slit are read from
that frame, never the hand, and the anchor stands, because the rule is about every
reach. ``start_in_neighbour`` and ``end_in_neighbour`` record it, so these reaches
can be set aside in an analysis.

WHAT NOT TO USE THIS FOR. Distances between hand points (paw width, outline, hand
angle) ask about the shape of the paw, which the anchors say nothing about: every
point at one place would read as a paw of zero width. Likelihoods, reach detection
and pellet outcomes decide what happened from what was actually seen, and must not
be fed positions nobody tracked.
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# What "between" means: how far from the nose towards the slit corner. The midpoint.
ANCHOR_FRACTION_FROM_NOSE = 0.5
# Below this a landmark's position is not used for an anchor. It is the threshold
# the rest of the extractor uses for trusting a point.
ANCHOR_MIN_LIKELIHOOD = 0.7
# How far from the anchor frame a landmark may be taken from, in frames: half a
# second at 60 frames per second. The nose moves little in that time compared with
# the length of a reach.
ANCHOR_SEARCH_FRAMES = 30
HAND_POINTS = ("RightHand", "RHLeft", "RHRight", "RHOut")


@dataclass
class AnchoredPath:
    """One hand point through one reach: detected frames plus the two anchors.

    ``x`` and ``y`` have two more values than the reach has frames. ``[0]`` is the
    start anchor and ``[-1]`` the end anchor, in image coordinates. ``nose_x`` and
    ``nose_y`` line up with them, so ``x - nose_x`` is nose-relative. The path covers
    (end_frame - start_frame + 2) frame steps.
    """
    x: np.ndarray
    y: np.ndarray
    nose_x: np.ndarray
    nose_y: np.ndarray
    start_source: str
    end_source: str
    start_in_neighbour: bool
    end_in_neighbour: bool

    @property
    def ok(self) -> bool:
        """Both anchors exist. Without them no path measurement is made."""
        return self.start_source != "none" and self.end_source != "none"

    def nose_relative(self) -> Tuple[np.ndarray, np.ndarray]:
        """The path with the nose subtracted frame by frame, so head movement does
        not count as paw movement."""
        return self.x - self.nose_x, self.y - self.nose_y


# A frame whose point sits within this many pixels of the last new position is a
# repeat of that picture, not a new one. Tracking the same picture twice moves a point
# by a few hundredths of a pixel; a hand that is really moving leaves this distance
# within a frame or two, and a hand that creeps slower than this is merged into a
# straight, steady path between the pictures, where its speed is the same.
REPEATED_FRAME_MAX_PX = 0.5
# The most frames in a row a recorder repeats one picture. A recorder writing about
# twice as many frames as its cameras deliver repeats a picture once, and now and then
# twice when a picture arrives late; longer still runs are a hand that is not moving.
MAX_REPEATED_FRAMES = 2


def fill_repeated_frames(x, y, max_px: float = REPEATED_FRAME_MAX_PX,
                         max_repeats: int = MAX_REPEATED_FRAMES):
    """``(x, y)`` of an anchored path with each repeated detected frame moved to where
    the hand passed between the two new positions around it (module docstring: REPEATED
    FRAMES). Returns new arrays; the inputs are not changed.

    Only the detected frames (``[1:-1]``) are considered: the anchors are calculated
    points, never pictures. A detected frame is a new position when it lies at least
    ``max_px`` from the last new position. The frames between two new positions are
    spread evenly along the straight line between them, so each step between them is
    the distance over the frames it took. A run of more than ``max_repeats`` frames
    (a hand that is really still), frames after the last new position, and any run
    that holds an untracked (NaN) frame are left as they are.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    if n < 4:
        return x, y
    fresh = []
    last = None
    for k in range(1, n - 1):
        if not (np.isfinite(x[k]) and np.isfinite(y[k])):
            last = None          # an untracked frame ends the run; the next tracked one is new
            continue
        if last is None or np.hypot(x[k] - x[last], y[k] - y[last]) >= max_px:
            fresh.append(k)
            last = k
    out_x, out_y = x.copy(), y.copy()
    for a, b in zip(fresh[:-1], fresh[1:]):
        if b - a < 2 or b - a - 1 > max_repeats:
            continue
        between = slice(a + 1, b)
        if not (np.all(np.isfinite(x[between])) and np.all(np.isfinite(y[between]))):
            continue
        t = (np.arange(a + 1, b) - a) / float(b - a)
        out_x[between] = x[a] + (x[b] - x[a]) * t
        out_y[between] = y[a] + (y[b] - y[a]) * t
    return out_x, out_y


def tracked_steps(steps: np.ndarray) -> np.ndarray:
    """The frame-to-frame steps of an anchored path that join two detected frames.

    ``steps`` has one value per step of the path, so its first value leaves the start
    anchor and its last arrives at the end anchor; both are dropped (the module
    docstring says why). What remains lines up with the detected frames:
    ``tracked_steps(steps)[i]`` is the step leaving detected frame i, and a reach of n
    detected frames has n - 1 of them. The last detected frame has none, so a speed
    leaving it does not exist rather than being read from the anchor.
    """
    return steps[1:-1]


def _trusted_position(df: pd.DataFrame, point: str, frame: int) -> Optional[Tuple[float, float, bool]]:
    """(x, y, was_it_this_frame) for the trusted position of ``point`` nearest
    ``frame``, or None if there is none within ANCHOR_SEARCH_FRAMES."""
    if f"{point}_x" not in df.columns or f"{point}_y" not in df.columns:
        return None
    x, y = df[f"{point}_x"].values, df[f"{point}_y"].values
    lk_col = f"{point}_likelihood"
    lk = df[lk_col].values if lk_col in df.columns else np.ones(len(df))
    for step in range(ANCHOR_SEARCH_FRAMES + 1):
        for f in ((frame,) if step == 0 else (frame - step, frame + step)):
            if 0 <= f < len(df) and lk[f] >= ANCHOR_MIN_LIKELIHOOD \
                    and not (np.isnan(x[f]) or np.isnan(y[f])):
                return float(x[f]), float(y[f]), step == 0
    return None


def _anchor(df: pd.DataFrame, slit_point: str, frame: int):
    """(anchor_x, anchor_y, nose_x, nose_y, source) for one end of a reach."""
    inside = max(0, min(frame, len(df) - 1))
    nose = _trusted_position(df, "Nose", inside)
    slit = _trusted_position(df, slit_point, inside)
    if nose is None or slit is None:
        return np.nan, np.nan, np.nan, np.nan, "none"
    if inside != frame:
        source = "video_edge"
    else:
        source = "anchor_frame" if nose[2] and slit[2] else "nearest_frame"
    x = nose[0] + ANCHOR_FRACTION_FROM_NOSE * (slit[0] - nose[0])
    y = nose[1] + ANCHOR_FRACTION_FROM_NOSE * (slit[1] - nose[1])
    return x, y, nose[0], nose[1], source


def anchored_hand_path(
    df: pd.DataFrame,
    hand_point: str,
    start_frame: int,
    end_frame: int,
    other_reaches: Optional[Iterable[Tuple[int, int]]] = None,
) -> AnchoredPath:
    """The path of ``hand_point`` for the reach start_frame..end_frame, with the two
    anchors added. Every paw-position measurement goes through here: the module
    docstring says why.

    ``df`` has flat column names (``RightHand_x``). ``other_reaches`` are the
    (start_frame, end_frame) of the video's other reaches, used only to record
    whether an anchor frame falls inside one. Nothing here reads a hand position
    outside the detected frames.
    """
    real = df.iloc[start_frame:end_frame + 1]
    sx, sy, snx, sny, s_source = _anchor(df, "BOXL", start_frame - 1)
    ex, ey, enx, eny, e_source = _anchor(df, "BOXR", end_frame + 1)
    others = list(other_reaches or [])

    def _occupied(frame):
        return any(a <= frame <= b for a, b in others)

    return AnchoredPath(
        x=np.concatenate([[sx], real[f"{hand_point}_x"].values.astype(float), [ex]]),
        y=np.concatenate([[sy], real[f"{hand_point}_y"].values.astype(float), [ey]]),
        nose_x=np.concatenate([[snx], real["Nose_x"].values.astype(float), [enx]]),
        nose_y=np.concatenate([[sny], real["Nose_y"].values.astype(float), [eny]]),
        start_source=s_source,
        end_source=e_source,
        start_in_neighbour=_occupied(start_frame - 1),
        end_in_neighbour=_occupied(end_frame + 1),
    )


def anchored_reach_frames(df: pd.DataFrame, start_frame: int, end_frame: int) -> Optional[pd.DataFrame]:
    """The reach's frames as a table, with one frame added at each end whose hand
    points all sit at the anchor. None if the reach has no anchor.

    For tools that take whole frames rather than one point's path, such as a
    biomechanics export: the limb then starts and ends where it really does. The
    added frames say the hand was trusted (likelihood 1), because such a tool would
    otherwise drop them for being unseen. Every other point in them is what the
    tracker reported. Do not use this table for visibility or likelihood measures.
    """
    path = anchored_hand_path(df, HAND_POINTS[0], start_frame, end_frame)
    if not path.ok:
        return None
    last = len(df) - 1
    frames = [max(start_frame - 1, 0)] + list(range(start_frame, end_frame + 1)) + [min(end_frame + 1, last)]
    out = df.iloc[frames].reset_index(drop=True)
    for point in HAND_POINTS:
        if f"{point}_x" not in out.columns:
            continue
        for row, x, y in ((0, path.x[0], path.y[0]), (len(out) - 1, path.x[-1], path.y[-1])):
            out.loc[row, [f"{point}_x", f"{point}_y", f"{point}_likelihood"]] = [x, y, 1.0]
    return out
