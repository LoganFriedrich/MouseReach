"""
Stage 3: Pellet observed back on pillar after the last reach attempt.

Semantic question this stage answers:
    "After the last reach attempt in the segment, is the pellet ever
    observed sustained back on the pillar?"

Physical reasoning:
    A pellet that was actually displaced cannot return to the pillar
    (per the existing `pellet_cannot_return_to_pillar.md` memory rule
    -- once truly off, it stays off). So if at any point AFTER the
    last reach attempt (with a settling buffer), the pellet is
    confidently observed back on the pillar (sustained for at least
    3 consecutive frames, inside the pillar circle with a small
    pellet-size buffer), then no reach in this segment displaced the
    pellet.

If YES (pellet sustained back on pillar after last reach) AND the
    raw Pillar bodypart never rose above the displacement-likelihood
    threshold in the post-settling window -> commit untouched. The
    last reach was a non-contact attempt, and any earlier reaches
    also could not have displaced (or the pellet couldn't be back on
    pillar now).

If NO -> continue. Either the pellet is not observed back on pillar
    after the last reach (it might have been displaced, or off-pillar
    from segment start) OR the raw Pillar bodypart rose to confident
    detection in the post-settling window (which only happens when
    the pellet is actually gone and the now-exposed pillar becomes
    visible to DLC -- a positive signal of displacement that we use
    to reject the pellet-on-pillar evidence as a possible label
    switch). Deferred to later stages.

Pillar-bodypart-rises-after-displacement signal:
    DLC sometimes label-switches the Pillar as "Pellet" at high lk
    after a real displacement, when the pellet is gone and the
    now-visible pillar gets re-labeled. Without correction, this
    signature would falsely satisfy Stage 3's "pellet on pillar"
    criterion. The corrective signal: when the pellet is actually
    gone, the Pillar BODYPART (separate from the relabeled Pellet
    bodypart) rises to confident lk -- empirically 100% of GT
    displaced/retrieved segments show sustained Pillar_lk >= 0.5 in
    the post-event window, vs. <10% of GT untouched segments. So
    requiring max(Pillar_lk) < 0.5 across the eligible window blocks
    the false commits without sacrificing meaningful yield.

Cascade OKF emit on commit: seg_end - TRANSITION_ZONE_HALF (last
clean frame of current segment), parallel to Stages 1 and 2.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import SegmentInput, Stage, StageDecision

PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5

# "Pellet on pillar" predicate: pellet inside the pillar circle with
# a small buffer to account for the pellet's own image-space size and
# detection edge noise (a pellet sitting on the pillar surface can
# have its image-projected center slightly outside the pillar circle
# due to its own radius and slight wobble).
PILLAR_BUFFER_FACTOR = 1.2

# Likelihood thresholds.
PELLET_LK_THR = 0.7    # "pellet detected" = lk >= this
PAW_LK_THR = 0.5       # paw-past-y-line counts only at lk >= this

# Post-bout settling buffer: after the paw retracts past the y-line,
# the pellet may be wobbling/jittering at the pillar surface before
# settling back to a stable position. This buffer ensures we only
# look for sustained-on-pillar evidence once the pellet has had time
# to come to rest.
POST_BOUT_SETTLING_FRAMES = 15

# Min consecutive frames of confident on-pillar detection required to
# count as "pellet returned to pillar."
CONSECUTIVE_REQUIRED = 3

# Pillar-bodypart-rises rejection threshold: any frame in the post-
# settling eligible window where the raw Pillar bodypart's likelihood
# meets or exceeds this value is treated as evidence that the pellet
# was actually displaced (the now-exposed pillar is visible to DLC).
# Empirical (2026-05-01, 37 train-pool videos): 100% of GT displaced/
# retrieved segments show sustained Pillar_lk >= 0.5 in the post-event
# window; <10% of GT untouched do.
PILLAR_LK_DISPLACEMENT_THRESHOLD = 0.5


def _find_paw_past_y_line_bouts(
    paw_past_y: np.ndarray,
) -> List[Tuple[int, int]]:
    """Return list of (start, end_inclusive) bouts where paw_past_y
    is True for at least one frame.
    """
    n = len(paw_past_y)
    bouts: List[Tuple[int, int]] = []
    run_start = -1
    for i in range(n):
        if paw_past_y[i]:
            if run_start < 0:
                run_start = i
        else:
            if run_start >= 0:
                bouts.append((run_start, i - 1))
                run_start = -1
    if run_start >= 0:
        bouts.append((run_start, n - 1))
    return bouts


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
        clean_end = seg.seg_end - self.transition_zone_half
        if clean_end <= seg.seg_start:
            return StageDecision(
                decision="continue",
                reason="segment_too_short_to_have_clean_zone")

        sub_raw = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
        n = len(sub_raw)
        if n == 0:
            return StageDecision(decision="continue", reason="empty_segment")

        # Apply impossibility filter to SA + Pellet bodyparts so the
        # pillar geometry and pellet position are clean.
        sub = clean_dlc_bodyparts(
            sub_raw, other_bodyparts_to_clean=("Pellet",))

        # Pillar geometry from cleaned SA bodyparts.
        geom = compute_pillar_geometry_series(sub)
        pillar_cx = geom["pillar_cx"].to_numpy()
        pillar_cy = geom["pillar_cy"].to_numpy()
        pillar_r = geom["pillar_r"].to_numpy()
        slit_y_line = pillar_cy + pillar_r

        # Per-frame paw-past-y-line predicate (any of the 4 paw
        # bodyparts at confident lk past the slit-closest pillar y).
        paw_past_y = np.zeros(n, dtype=bool)
        for bp in PAW_BODYPARTS:
            paw_y = sub[f"{bp}_y"].to_numpy(dtype=float)
            paw_lk = sub[f"{bp}_likelihood"].to_numpy(dtype=float)
            paw_past_y |= (paw_y <= slit_y_line) & (paw_lk >= self.paw_lk_threshold)

        bouts = _find_paw_past_y_line_bouts(paw_past_y)

        # Pellet position + on-pillar predicate. Use the original
        # likelihood (not modified by cleaning) so we still respect
        # DLC's confidence reports.
        pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
        pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
        pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
        dist = np.sqrt((pellet_x - pillar_cx) ** 2 +
                       (pellet_y - pillar_cy) ** 2)
        on_pillar_threshold = pillar_r * self.pillar_buffer_factor
        pellet_on_pillar = (
            (pellet_lk >= self.pellet_lk_threshold)
            & (dist <= on_pillar_threshold)
            & (~paw_past_y)
        )

        # Determine the "look-after" window: starts after the last
        # paw-past-y-line bout's end + settling buffer. If there are
        # no bouts, look across the whole clean zone (rare for
        # Stage-3 input -- Stage 2 should already have committed
        # those -- but handle gracefully).
        if bouts:
            last_bout_end = bouts[-1][1]
            search_start = last_bout_end + 1 + self.post_bout_settling_frames
        else:
            search_start = 0

        # Raw Pillar bodypart likelihood across the clean zone (use the
        # ORIGINAL DLC dataframe -- we have not cleaned the Pillar
        # bodypart, and we wouldn't want to: low Pillar lk during
        # untouched segments is the expected occluded state, not a
        # tracking failure to fill in.)
        pillar_lk = sub_raw["Pillar_likelihood"].to_numpy(dtype=float)

        feats = {
            "n_clean_zone_frames": int(n),
            "n_paw_past_y_line_bouts": int(len(bouts)),
            "last_bout_end_idx": int(bouts[-1][1]) if bouts else -1,
            "search_start_idx": int(search_start),
            "search_window_frames": max(0, int(n - search_start)),
        }

        # Pillar-bodypart-rises rejection: if the raw Pillar bodypart
        # ever rises to the displacement-likelihood threshold in the
        # post-settling, paw-not-past-y-line window, the pellet was
        # actually displaced (now-exposed pillar visible to DLC).
        # Defer regardless of any pellet-on-pillar evidence.
        pillar_lk_eligible_max = 0.0
        if search_start < n:
            window_pillar_lk = pillar_lk[search_start:n]
            window_paw_past = paw_past_y[search_start:n]
            eligible = ~window_paw_past
            if eligible.any():
                pillar_lk_eligible_max = float(window_pillar_lk[eligible].max())
        feats["pillar_lk_eligible_max"] = pillar_lk_eligible_max
        feats["pillar_lk_displacement_threshold"] = float(
            self.pillar_lk_displacement_threshold)

        if pillar_lk_eligible_max >= self.pillar_lk_displacement_threshold:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pillar_bodypart_rose_to_displacement_lk "
                    f"(max Pillar_lk = {pillar_lk_eligible_max:.3f} "
                    f">= {self.pillar_lk_displacement_threshold} "
                    f"in post-settling eligible window -- pellet "
                    f"likely displaced)"
                ),
                features=feats,
            )

        # Look for any 3+ consecutive on-pillar frames in the search
        # window.
        found_run_idx = -1
        if search_start < n:
            run_len = 0
            for i in range(search_start, n):
                if pellet_on_pillar[i]:
                    run_len += 1
                    if run_len >= self.consecutive_required:
                        found_run_idx = i - self.consecutive_required + 1
                        break
                else:
                    run_len = 0

        feats["pellet_returned_to_pillar_at_idx"] = int(found_run_idx)

        if found_run_idx >= 0:
            okf = clean_end
            feats["outcome_known_frame_emitted"] = int(okf)
            return StageDecision(
                decision="commit",
                committed_class="untouched",
                whens={"outcome_known_frame": int(okf),
                       "interaction_frame": None},
                reason=(
                    f"pellet_observed_back_on_pillar_after_last_reach "
                    f"(sustained {self.consecutive_required}+ "
                    f"frames inside circle with "
                    f"{self.pillar_buffer_factor}x buffer, "
                    f"{self.post_bout_settling_frames}f settling, "
                    f"Pillar_lk stayed below "
                    f"{self.pillar_lk_displacement_threshold})"
                ),
                features=feats,
            )

        return StageDecision(
            decision="continue",
            reason=(
                f"pellet_not_observed_back_on_pillar_after_last_reach "
                f"(no {self.consecutive_required}+ consecutive on-pillar "
                f"frames found in {feats['search_window_frames']}-frame "
                f"post-settling window)"
            ),
            features=feats,
        )
