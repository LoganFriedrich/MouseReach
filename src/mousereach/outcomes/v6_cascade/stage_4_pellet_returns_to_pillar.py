"""
Stage 4: Pellet returns to pillar after reach (4.0 recalibration).

Question this stage answers:
    "After the detected reaches in the segment, is the pellet ever
    observed sustained back on the pillar?"

4.0 recalibration note:
    The original Stage 4 used paw-past-y-line bouts to define the
    search-after window. On 4.0, paw detection past the slit line is
    much more frequent (approaching paw detected at moderate lk),
    so the bout-based window is unreliable. This replacement uses
    detected reaches (algo 2) to anchor the search window.

    Additionally includes a vanish guard: a retrieved pellet is gone
    for a long stretch, so reject a late on-pillar blip if the pellet
    has a sustained absence (lk<0.5 for >= VANISH_RUN frames).

Cascade emit on commit:
- committed_class: "untouched"
- whens["outcome_known_frame"]: clean_end (last clean-zone frame)
- whens["interaction_frame"]: None
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .guards import lrun
from .stage_base import SegmentInput, Stage, StageDecision

PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5

PILLAR_BUFFER_FACTOR = 1.2
PELLET_LK_THR = 0.7
PAW_LK_THR = 0.5
POST_BOUT_SETTLING_FRAMES = 15
CONSECUTIVE_REQUIRED = 3
PILLAR_LK_DISPLACEMENT_THRESHOLD = 0.5
VANISH_RUN = 60


class Stage4PelletReturnsToPillar(Stage):
    name = "stage_4_pellet_returns_to_pillar"
    target_class = "untouched"

    def __init__(
        self,
        pillar_buffer_factor: float = PILLAR_BUFFER_FACTOR,
        post_bout_settling_frames: int = POST_BOUT_SETTLING_FRAMES,
        consecutive_required: int = CONSECUTIVE_REQUIRED,
        pellet_lk_threshold: float = PELLET_LK_THR,
        paw_lk_threshold: float = PAW_LK_THR,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
        pillar_lk_displacement_threshold: float = PILLAR_LK_DISPLACEMENT_THRESHOLD,
    ):
        self.pillar_buffer_factor = pillar_buffer_factor
        self.post_bout_settling_frames = post_bout_settling_frames
        self.consecutive_required = consecutive_required
        self.pellet_lk_threshold = pellet_lk_threshold
        self.paw_lk_threshold = paw_lk_threshold
        self.transition_zone_half = transition_zone_half
        self.pillar_lk_displacement_threshold = pillar_lk_displacement_threshold

    def decide(self, seg: SegmentInput) -> StageDecision:
        ce = seg.seg_end - self.transition_zone_half
        if ce <= seg.seg_start:
            return StageDecision(
                decision="continue",
                reason="too_short")
        sub_raw = seg.dlc_df.iloc[seg.seg_start:ce + 1]
        n = len(sub_raw)
        if n == 0:
            return StageDecision(decision="continue", reason="empty")
        sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
        g = compute_pillar_geometry_series(sub)
        cx = g["pillar_cx"].to_numpy(float)
        cy = g["pillar_cy"].to_numpy(float)
        r = g["pillar_r"].to_numpy(float)
        slit = cy + r
        paw = np.zeros(n, bool)
        for bp in PAW_BODYPARTS:
            yy = sub[f"{bp}_y"].to_numpy(float)
            ll = sub[f"{bp}_likelihood"].to_numpy(float)
            paw |= (yy <= slit) & (ll >= self.paw_lk_threshold)
        # Search-after index: use detected reaches (algo 2) to anchor
        # the search window. Start looking after the last reach end +
        # settling buffer.
        ends = [min(b - seg.seg_start, n - 1)
                for (a, b) in (seg.reach_windows or [])
                if b - seg.seg_start >= 0]
        ss = (max(ends) + 1 + self.post_bout_settling_frames) if ends else 0
        px = sub["Pellet_x"].to_numpy(float)
        py = sub["Pellet_y"].to_numpy(float)
        plk = sub_raw["Pellet_likelihood"].to_numpy(float)
        dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        thr = r * self.pillar_buffer_factor
        onp = (plk >= self.pellet_lk_threshold) & (dist <= thr) & (~paw)
        plk2 = sub_raw["Pillar_likelihood"].to_numpy(float)
        gmax = 0.0
        if ss < n:
            el = ~paw[ss:n]
            if el.any():
                gmax = float(plk2[ss:n][el].max())
        if gmax >= self.pillar_lk_displacement_threshold:
            return StageDecision(decision="continue", reason="pillar_rose")
        found = -1
        if ss < n:
            rl = 0
            for i in range(ss, n):
                if onp[i]:
                    rl += 1
                    if rl >= self.consecutive_required:
                        found = i - self.consecutive_required + 1
                        break
                else:
                    rl = 0
        vrun = lrun(plk < 0.5)
        if found >= 0 and vrun < VANISH_RUN:
            return StageDecision(
                decision="commit", committed_class="untouched",
                whens={"outcome_known_frame": int(ce),
                       "interaction_frame": None},
                reason=(
                    f"reach_window_on_pillar (vanish={vrun})"))
        return StageDecision(
            decision="continue",
            reason=(
                f"no_return_or_pellet_vanished (vanish={vrun})"))
