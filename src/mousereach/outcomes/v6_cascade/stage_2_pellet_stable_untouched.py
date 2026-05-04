"""
Stage 1: Pellet-stable-untouched.

Question this stage answers:
    "Is the pellet essentially in the same spot inside the pillar circle
    for the whole segment except during reach occlusions?"

If yes -> COMMIT untouched (high confidence). If not -> CONTINUE.

Inputs:
- pellet_inside_pillar_circle_frac_excluding_during_reach
  (per-frame inside-circle boolean, averaged over non-during-reach
   frames, using smoothed-SA pillar geometry)
- pellet_position_start_end_distance_in_radii
  (median pellet position in first 30f vs last 30f of segment, in
   units of pillar radius)

Commit rule:
    frac >= COMMIT_FRAC_THRESHOLD AND
    start_end_distance <= COMMIT_DISTANCE_THRESHOLD_RADII

Defaults are conservative -- a stage 1 false-commit on a touched case
is unrecoverable. Tune against per-stage validation, not aggregate.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from mousereach.lib.pillar_geometry import (
    compute_pillar_geometry_series, pellet_inside_pillar_circle,
)
from mousereach.lib.causal_attribution import detect_tray_motion_onset
from .stage_base import SegmentInput, Stage, StageDecision


COMMIT_FRAC_THRESHOLD = 0.95
COMMIT_DISTANCE_THRESHOLD_RADII = 1.0
START_END_WINDOW = 30  # frames at start / end to compute median position
# Transition zone around segmenter's seg_end. Per the 2026-05-01 user
# reframing: GT outcome_known_frame marks "new segment is solidly started";
# segmenter's seg_end marks "old segment is clearly ending"; between them
# is a transitional zone where apparatus is in flux and mouse can mess
# with things. Discard +/- 5 frames around seg_end. Effective last-clean-
# frame for current segment = seg_end - 5; cascade analysis anchor =
# seg_end - 10 (with +/- 5-frame buffer either side for stability check).
TRANSITION_ZONE_HALF = 5
ANCHOR_BACK_OFFSET = 10
ANCHOR_HALF_WINDOW = 5


def _during_reach_mask(seg_start: int, seg_end: int,
                       reach_windows: List[Tuple[int, int]]) -> np.ndarray:
    """Boolean array length seg_end-seg_start+1; True at frames inside
    any of the reach windows.
    """
    n = seg_end - seg_start + 1
    mask = np.zeros(n, dtype=bool)
    for rs, re in reach_windows:
        # Clip to segment
        s = max(seg_start, int(rs))
        e = min(seg_end, int(re))
        if e < s:
            continue
        mask[s - seg_start:e - seg_start + 1] = True
    return mask


class Stage2PelletStableUntouched(Stage):
    name = "stage_2_pellet_stable_untouched"
    target_class = "untouched"

    def __init__(
        self,
        commit_frac: float = COMMIT_FRAC_THRESHOLD,
        commit_distance_radii: float = COMMIT_DISTANCE_THRESHOLD_RADII,
        start_end_window: int = START_END_WINDOW,
    ):
        self.commit_frac = commit_frac
        self.commit_distance_radii = commit_distance_radii
        self.start_end_window = start_end_window

    def decide(self, seg: SegmentInput) -> StageDecision:
        # CLEAN current-segment range -- exclude the +5 transition zone
        # at segment end (per 2026-05-01 user reframing). Frames
        # seg_end - 5 to seg_end + 5 are no-man's-land.
        clean_end = seg.seg_end - TRANSITION_ZONE_HALF
        if clean_end <= seg.seg_start:
            return StageDecision(decision="continue",
                                 reason="segment_too_short_to_have_clean_zone")
        sub = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
        if len(sub) == 0:
            return StageDecision(decision="continue", reason="empty_segment")

        # Per-frame inside-circle boolean (smoothed-SA geometry)
        geom = compute_pillar_geometry_series(sub)
        inside = pellet_inside_pillar_circle(sub, pillar_geom=geom).to_numpy()

        # Mask out during-reach frames within the clean zone
        during = _during_reach_mask(
            seg.seg_start, clean_end, seg.reach_windows)
        non_during = ~during
        n_non_during = int(non_during.sum())
        if n_non_during == 0:
            return StageDecision(
                decision="continue",
                reason="all_clean_frames_during_reach",
                features={
                    "n_non_during_reach_frames": 0,
                    "pellet_inside_pillar_circle_frac_excluding_during_reach": None,
                    "pellet_position_start_end_distance_in_radii": None,
                },
            )

        frac_inside = float(inside[non_during].mean())

        plk = sub["Pellet_likelihood"].to_numpy(dtype=float)
        px = sub["Pellet_x"].to_numpy(dtype=float)
        py = sub["Pellet_y"].to_numpy(dtype=float)
        n = len(sub)

        median_radius = float(np.nanmedian(geom["pillar_r"].to_numpy()))
        if median_radius <= 0 or not np.isfinite(median_radius):
            median_radius = 1.0

        # Cascade OKF anchor: seg_end - 5, the LAST clean-zone frame
        # (the right edge of the clean range, just before the +/- 5
        # transition zone). The analysis window is still
        # (seg_end - 15, seg_end - 5) -- the +/- 5 around seg_end - 10
        # pre-anchor -- but the EMITTED OKF is the latest clean frame
        # so trust comparison against (walked-back) GT OKF stays in
        # clean territory. See `cascade_trust_framework.md`.
        anchor_frame = seg.seg_end - TRANSITION_ZONE_HALF
        analysis_window_size = 2 * ANCHOR_HALF_WINDOW + 1
        aw_start_in_sub = max(0, n - analysis_window_size)
        aw_inside = inside[aw_start_in_sub:]
        aw_lk = plk[aw_start_in_sub:]

        # Per the user's 2026-05-01 design: Stage 1 is for the cleanest
        # cases only. ANY frame in the analysis window where the pellet
        # is obscured (lk < threshold), not inside the pillar circle,
        # or paw is close to the pellet -> defer to Stage 2 which
        # handles paw-near cases.
        # No "walk back past during-reach" -- if the analysis window
        # has reach activity / paw nearby / obscuration, it's not a
        # Stage 1 commit case.
        rh_x = sub["RightHand_x"].to_numpy(dtype=float)[aw_start_in_sub:]
        rh_y = sub["RightHand_y"].to_numpy(dtype=float)[aw_start_in_sub:]
        rh_lk = sub["RightHand_likelihood"].to_numpy(dtype=float)[aw_start_in_sub:]
        # Paw-pellet distance in radii units. Only "near" if paw lk is
        # high enough to trust the position.
        aw_px = px[aw_start_in_sub:]
        aw_py = py[aw_start_in_sub:]
        paw_pellet_dist = np.sqrt((rh_x - aw_px) ** 2 + (rh_y - aw_py) ** 2)
        paw_near = (rh_lk >= 0.7) & (paw_pellet_dist < 2.0 * median_radius)

        aw_pellet_visible = aw_lk >= 0.7
        aw_pellet_inside = aw_inside.astype(bool)

        # Per-frame "clean": pellet visible AND inside circle AND no paw close
        aw_clean = aw_pellet_visible & aw_pellet_inside & ~paw_near
        analysis_window_all_clean = bool(aw_clean.all()) and len(aw_clean) > 0
        analysis_window_clean_frac = (
            float(aw_clean.mean()) if len(aw_clean) > 0 else 0.0)

        feats = {
            "n_non_during_reach_frames": n_non_during,
            "pellet_inside_pillar_circle_frac_excluding_during_reach": frac_inside,
            "median_pillar_radius_px": median_radius,
            "anchor_frame": int(anchor_frame),
            "analysis_window_size": int(len(aw_clean)),
            "analysis_window_all_clean": analysis_window_all_clean,
            "analysis_window_clean_frac": analysis_window_clean_frac,
            "analysis_window_paw_near_count": int(paw_near.sum()),
            "analysis_window_pellet_obscured_count": int((~aw_pellet_visible).sum()),
            "analysis_window_pellet_outside_circle_count": int((~aw_pellet_inside).sum()),
        }

        # Commit rule: clean across whole segment AND analysis window
        # is unambiguously clean (no paw near, no obscuration, all
        # inside-circle). Anything else -> defer to Stage 2.
        if (frac_inside >= self.commit_frac and analysis_window_all_clean):
            # Cascade OKF = anchor frame (seg_end - 10), the center
            # of our analysis window. This is the cascade's last clean
            # frame from this segment, semantically distinct from
            # GT-er's outcome_known_frame (which marks "new segment is
            # solidly started"). See `feature_philosophy_event_anchored
            # _walking.md` and `untouched_outcome_known_frame_derivation
            # .md`.
            okf = anchor_frame  # seg.seg_end - ANCHOR_BACK_OFFSET (=10)
            feats["outcome_known_frame_emitted"] = int(okf)
            return StageDecision(
                decision="commit",
                committed_class="untouched",
                whens={"outcome_known_frame": int(okf),
                       "interaction_frame": None},
                reason=(
                    f"pellet_stable_inside_pillar_circle "
                    f"(frac={frac_inside:.3f}, analysis_window_clean=1.0, "
                    f"anchor={anchor_frame})"
                ),
                features=feats,
            )

        return StageDecision(
            decision="continue",
            reason=(
                f"not_unambiguously_clean (frac_inside={frac_inside:.3f}, "
                f"analysis_window_clean_frac={analysis_window_clean_frac:.2f}, "
                f"paw_near_count={int(paw_near.sum())}, "
                f"obscured_count={int((~aw_pellet_visible).sum())}, "
                f"outside_count={int((~aw_pellet_inside).sum())})"
            ),
            features=feats,
        )
