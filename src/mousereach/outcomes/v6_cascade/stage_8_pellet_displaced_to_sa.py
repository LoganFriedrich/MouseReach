"""
Stage 5: Pellet visibly displaced from the pillar into the SA.

Semantic question this stage answers:
    "Does the pellet ever come to rest at a sustained off-pillar
    position somewhere in the SA for >=40 frames at any point in
    the segment?"

Physical reasoning:
    A pellet that was actually displaced from the pillar by a reach
    in this segment will land somewhere in the SA and rest there for
    the remainder of the segment. So if the pellet is observed
    sustained at a stable off-pillar position for at least 40
    consecutive frames anywhere in the segment, it was displaced.

    This is intentionally simple. We do NOT try to identify which
    reach caused the displacement at the per-frame level -- the
    sustained-rest signature is enough to confirm the outcome class
    is displaced_sa. IFR and OKF are emitted from where the rest
    period begins (OKF) and from the most recent reach before that
    (IFR).

Stage ordering:
    Stage 4 (pellet-was-off-pillar-at-segment-start) runs before
    Stage 5 and picks off segments where the pellet entered already
    displaced. So Stage 5's input has the precondition that the
    pellet was on the pillar at segment start (otherwise Stage 4
    would have committed it as untouched). This stage explicitly
    re-verifies the on-pillar-at-segment-start precondition for
    safety -- segments that don't satisfy it defer rather than
    commit.

Co-detection triage (pre-check):
    Same as Stage 4 -- if Pellet AND Pillar are both detected at
    high lk within 1 pillar-radius for sustained frames anywhere in
    the clean zone, the segment goes to TRIAGE (DLC artifact, often
    plexiglass over the SA).

Cascade emit on commit:
    - committed_class: "displaced_sa"
    - whens["interaction_frame"]: middle of the LAST paw-past-y-line
      bout that ended before the sustained-rest period starts
      (with empirical p50 IFR position within bout = 0.4).
    - whens["outcome_known_frame"]: the frame at which the
      sustained-rest period STARTS (i.e., the pellet first reaches
      its final resting position).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import SegmentInput, Stage, StageDecision

PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5

# Likelihood thresholds.
# Pellet lk threshold raised aggressively (from 0.7 default) on
# 2026-05-02 to filter DLC label-switch cases. Real displaced
# pellets at rest show very high lk (close to 1.0); label-switch
# artifacts in problem videos like `20250716_CNT0213_P3` track
# non-pellet features at moderately high lk (0.7-0.9 range).
PELLET_LK_THR = 0.95   # raised from 0.7 to filter DLC label-switch
                       # cases at moderate lk; 0.95 catches abnormal
                       # cases but `20250716_CNT0213_P3` cases pass
PILLAR_LK_THR = 0.7
PAW_LK_THR = 0.5

# Co-detection triage (same as Stage 5). Both lk thresholds are very
# high (per 2026-05-02 user refinement) and distance very tight, so
# triage fires only for true DLC label-switch (bodyparts physically
# overlapping with confident detection on both). Edge-proximity cases
# where pellet sits near pillar tip do NOT trigger.
CODETECTION_PELLET_LK_THR = 0.95
CODETECTION_PILLAR_LK_THR = 0.95
CODETECTION_DISTANCE_RADII = 0.5
CODETECTION_SUSTAINED_FRAMES = 3

# On-pillar-at-segment-start precondition.
PRE_REACH_WINDOW_FRAMES = 30
ON_PILLAR_RADII = 1.2
PRE_REACH_ON_PILLAR_SUSTAINED_FRAMES = 5

# Sustained-rest-in-SA: the core Stage 5 signal. Looks at the
# pellet's median position (across confident, paw-not-past-y-line
# frames in clean zone). The pellet must be:
#   (1) clearly outside the pillar circle (> 1 pillar_r from calc'd
#       pillar center) -- pellet is not on the pillar
#   (2) WITHIN the SA quadrilateral bounds (between SABL/SABR and
#       SATL/SATR) -- pellet is sitting on the SA tray surface, where
#       displaced pellets actually land in 2D top-down view (per user
#       2026-05-02: "gravity pulls in the z plane that we can't see
#       in. we observe the behavior top down")
#   (3) sustained near the median position for >= REST_FRAMES_TOTAL
#       accumulated frames
PELLET_OFF_PILLAR_RADII = 1.0          # just outside pillar circle
REST_FRAMES_TOTAL = 40                  # total accumulated frames near
                                        # median (NOT consecutive --
                                        # follow-up reaches can break
                                        # contiguity but the pellet
                                        # is still resting between)
NEAR_MEDIAN_TOLERANCE_RADII = 1.5      # pellet within this radii of
                                        # the median = "at rest at
                                        # this location"
SA_QUAD_Y_BUFFER_PX = 0                 # pellet must have y within
                                        # [SATL_y - buffer, SABL_y +
                                        # buffer]. Buffer = 0 = strict
                                        # SA quadrilateral boundary.

# IFR position within causal bout (empirical p50 = 0.4 across 351 GT
# displaced_sa segments, 2026-05-01).
IFR_POSITION_IN_BOUT = 0.4

# Late-segment observability: real displaced_sa segments have the
# pellet sustained near its rest position throughout the LATER part
# of the segment (since the pellet can't return to the pillar). For
# retrieved cases, the pellet briefly appears off-pillar then
# disappears -- low late-segment observability. Empirical analysis
# (2026-05-02): displaced p5 = 324 frames, retrieved p50 = 39, in
# the last 25% of clean zone. Threshold of 250 cleanly separates.
LATE_SEGMENT_FRACTION = 0.25
LATE_SEGMENT_OBSERVABLE_MIN_FRAMES = 250


def _find_paw_past_y_line_bouts(
    paw_past_y: np.ndarray,
) -> List[Tuple[int, int]]:
    """List of (start, end_inclusive) bouts."""
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


def _detect_codetection_triage(
    pellet_lk, pillar_lk, pellet_x, pellet_y, pillar_x, pillar_y,
    pillar_r,
    pellet_lk_thr, pillar_lk_thr, distance_radii_thr, sustained_frames,
) -> bool:
    pp_dist = np.sqrt((pellet_x - pillar_x) ** 2 + (pellet_y - pillar_y) ** 2)
    pp_dist_radii = pp_dist / np.maximum(pillar_r, 1e-6)
    co_detected = (
        (pellet_lk >= pellet_lk_thr)
        & (pillar_lk >= pillar_lk_thr)
        & (pp_dist_radii <= distance_radii_thr)
    )
    run_len = 0
    for i in range(len(co_detected)):
        if co_detected[i]:
            run_len += 1
            if run_len >= sustained_frames:
                return True
        else:
            run_len = 0
    return False


def _find_first_near_median(
    eligible: np.ndarray,
    pellet_x: np.ndarray,
    pellet_y: np.ndarray,
    median_x: float,
    median_y: float,
    pillar_r: np.ndarray,
    tolerance_radii: float,
) -> int:
    """Find the FIRST frame index where eligible AND pellet position
    is within `tolerance_radii * pillar_r` of (median_x, median_y).
    Returns -1 if not found.
    """
    n = len(eligible)
    for i in range(n):
        if not eligible[i]:
            continue
        d = ((pellet_x[i] - median_x) ** 2
             + (pellet_y[i] - median_y) ** 2) ** 0.5
        if d <= tolerance_radii * max(pillar_r[i], 1e-6):
            return i
    return -1


class Stage8PelletDisplacedToSA(Stage):
    name = "stage_8_pellet_displaced_to_sa"
    target_class = "displaced_sa"

    def __init__(
        self,
        codetection_distance_radii: float = CODETECTION_DISTANCE_RADII,
        codetection_sustained_frames: int = CODETECTION_SUSTAINED_FRAMES,
        pre_reach_window_frames: int = PRE_REACH_WINDOW_FRAMES,
        on_pillar_radii: float = ON_PILLAR_RADII,
        pre_reach_on_pillar_sustained_frames: int = PRE_REACH_ON_PILLAR_SUSTAINED_FRAMES,
        pellet_off_pillar_radii: float = PELLET_OFF_PILLAR_RADII,
        rest_frames_total: int = REST_FRAMES_TOTAL,
        near_median_tolerance_radii: float = NEAR_MEDIAN_TOLERANCE_RADII,
        late_segment_fraction: float = LATE_SEGMENT_FRACTION,
        late_segment_observable_min_frames: int = LATE_SEGMENT_OBSERVABLE_MIN_FRAMES,
        ifr_position_in_bout: float = IFR_POSITION_IN_BOUT,
        pellet_lk_threshold: float = PELLET_LK_THR,
        pillar_lk_threshold: float = PILLAR_LK_THR,
        paw_lk_threshold: float = PAW_LK_THR,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.codetection_distance_radii = codetection_distance_radii
        self.codetection_sustained_frames = codetection_sustained_frames
        self.pre_reach_window_frames = pre_reach_window_frames
        self.on_pillar_radii = on_pillar_radii
        self.pre_reach_on_pillar_sustained_frames = pre_reach_on_pillar_sustained_frames
        self.pellet_off_pillar_radii = pellet_off_pillar_radii
        self.rest_frames_total = rest_frames_total
        self.near_median_tolerance_radii = near_median_tolerance_radii
        self.late_segment_fraction = late_segment_fraction
        self.late_segment_observable_min_frames = late_segment_observable_min_frames
        self.ifr_position_in_bout = ifr_position_in_bout
        self.pellet_lk_threshold = pellet_lk_threshold
        self.pillar_lk_threshold = pillar_lk_threshold
        self.paw_lk_threshold = paw_lk_threshold
        self.transition_zone_half = transition_zone_half

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

        sub = clean_dlc_bodyparts(
            sub_raw, other_bodyparts_to_clean=("Pellet",))

        geom = compute_pillar_geometry_series(sub)
        pillar_cx = geom["pillar_cx"].to_numpy()
        pillar_cy = geom["pillar_cy"].to_numpy()
        pillar_r = geom["pillar_r"].to_numpy()
        slit_y_line = pillar_cy + pillar_r

        pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
        pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
        pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
        pellet_dist_radii = (
            np.sqrt((pellet_x - pillar_cx) ** 2 +
                    (pellet_y - pillar_cy) ** 2)
            / np.maximum(pillar_r, 1e-6)
        )

        pillar_x_raw = sub_raw["Pillar_x"].to_numpy(dtype=float)
        pillar_y_raw = sub_raw["Pillar_y"].to_numpy(dtype=float)
        pillar_lk_raw = sub_raw["Pillar_likelihood"].to_numpy(dtype=float)

        # ----- Co-detection triage (FIRST priority).
        if _detect_codetection_triage(
            pellet_lk, pillar_lk_raw,
            pellet_x, pellet_y,
            pillar_x_raw, pillar_y_raw,
            pillar_r,
            CODETECTION_PELLET_LK_THR, CODETECTION_PILLAR_LK_THR,
            self.codetection_distance_radii,
            self.codetection_sustained_frames,
        ):
            # Per 2026-05-02 user reframing: codetection signature
            # alone doesn't prove triage is needed. Defer instead so
            # the segment can fall through to retrieved / abnormal /
            # dedicated-codetection stages later.
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_pillar_codetection_observed_defer "
                    f"(both bodyparts at high lk within "
                    f"{self.codetection_distance_radii} radii sustained "
                    f"{self.codetection_sustained_frames}+ frames -- "
                    f"not Stage 6's case to triage; defer)"
                ),
                features={"codetection_observed": True},
            )

        # Per-frame paw-past-y-line.
        paw_past_y = np.zeros(n, dtype=bool)
        for bp in PAW_BODYPARTS:
            paw_y = sub[f"{bp}_y"].to_numpy(dtype=float)
            paw_lk = sub[f"{bp}_likelihood"].to_numpy(dtype=float)
            paw_past_y |= (paw_y <= slit_y_line) & (paw_lk >= self.paw_lk_threshold)

        bouts = _find_paw_past_y_line_bouts(paw_past_y)
        if not bouts:
            return StageDecision(
                decision="continue",
                reason="no_paw_past_y_line_bouts_in_clean_zone")

        feats = {
            "n_clean_zone_frames": int(n),
            "n_paw_past_y_line_bouts": int(len(bouts)),
        }

        # ----- On-pillar-at-segment-start precondition. The pellet
        # must be observed sustained on-pillar in the small window
        # immediately before the first reach. Otherwise this could
        # be a started-displaced case (Stage 4's domain) or some
        # other pattern.
        first_bout_start = bouts[0][0]
        pre_reach_start = max(
            0, first_bout_start - self.pre_reach_window_frames)
        feats["first_bout_start_idx"] = int(first_bout_start)

        on_pillar_pre_reach_run = 0
        on_pillar_pre_reach_satisfied = False
        for i in range(pre_reach_start, first_bout_start):
            if (not paw_past_y[i]
                and pellet_lk[i] >= self.pellet_lk_threshold
                and pellet_dist_radii[i] <= self.on_pillar_radii):
                on_pillar_pre_reach_run += 1
                if on_pillar_pre_reach_run >= self.pre_reach_on_pillar_sustained_frames:
                    on_pillar_pre_reach_satisfied = True
                    break
            else:
                on_pillar_pre_reach_run = 0
        feats["on_pillar_pre_reach_satisfied"] = bool(on_pillar_pre_reach_satisfied)

        if not on_pillar_pre_reach_satisfied:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_not_observed_on_pillar_before_first_reach "
                    f"(needed {self.pre_reach_on_pillar_sustained_frames}+ "
                    f"sustained on-pillar frames in the "
                    f"{self.pre_reach_window_frames}-frame window before "
                    f"the first reach -- not Stage 5's case)"
                ),
                features=feats,
            )

        # ----- Core Stage 5 check: count confident, paw-not-past-
        # y-line, OFF-PILLAR frames across the whole clean zone. If
        # the pellet is observed off-pillar at enough total frames,
        # it landed in the SA. Then verify those off-pillar frames
        # cluster around a single rest position (similar location).
        off_pillar_eligible = (
            (pellet_lk >= self.pellet_lk_threshold)
            & (pellet_dist_radii > self.pellet_off_pillar_radii)
            & (~paw_past_y)
        )
        off_pillar_count = int(off_pillar_eligible.sum())
        feats["off_pillar_frame_count"] = off_pillar_count

        if off_pillar_count < self.rest_frames_total:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_off_pillar_count_too_low "
                    f"(only {off_pillar_count} confident off-pillar frames "
                    f"in clean zone, needed {self.rest_frames_total} -- "
                    f"pellet was not observed to settle in SA, defer)"
                ),
                features=feats,
            )

        # Compute the median position OF THE OFF-PILLAR FRAMES (not
        # all confident frames -- otherwise pre-displacement on-
        # pillar frames pull the median back toward the pillar).
        median_x = float(np.median(pellet_x[off_pillar_eligible]))
        median_y = float(np.median(pellet_y[off_pillar_eligible]))
        feats["off_pillar_median_x"] = median_x
        feats["off_pillar_median_y"] = median_y

        # ----- Within-SA-quadrilateral check: median pellet position
        # must be inside the SA bounds (front edge SATL/SATR, back
        # edge SABL/SABR). Real displaced pellets land on the SA tray
        # surface; positions outside this quadrilateral are
        # physically implausible (pellet between pillar and SA, or
        # below SA back edge -- both geometrically unreachable for a
        # passively-displaced pellet).
        # Use median SA bounds across off-pillar frames.
        sa_top_y = float(np.median(
            (sub["SATL_y"].to_numpy() + sub["SATR_y"].to_numpy())[off_pillar_eligible] / 2.0))
        sa_bot_y = float(np.median(
            (sub["SABL_y"].to_numpy() + sub["SABR_y"].to_numpy())[off_pillar_eligible] / 2.0))
        sa_left_x = float(np.median(
            np.minimum(sub["SABL_x"].to_numpy(), sub["SATL_x"].to_numpy())[off_pillar_eligible]))
        sa_right_x = float(np.median(
            np.maximum(sub["SABR_x"].to_numpy(), sub["SATR_x"].to_numpy())[off_pillar_eligible]))
        feats.update({
            "sa_top_y": sa_top_y, "sa_bot_y": sa_bot_y,
            "sa_left_x": sa_left_x, "sa_right_x": sa_right_x,
        })
        in_sa_y = sa_top_y <= median_y <= sa_bot_y
        in_sa_x = sa_left_x <= median_x <= sa_right_x
        feats["median_in_sa_quadrilateral"] = bool(in_sa_y and in_sa_x)
        if not (in_sa_y and in_sa_x):
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_median_outside_sa_quadrilateral "
                    f"(median ({median_x:.1f}, {median_y:.1f}); SA bounds "
                    f"y=[{sa_top_y:.1f}, {sa_bot_y:.1f}], "
                    f"x=[{sa_left_x:.1f}, {sa_right_x:.1f}]; "
                    f"in_y={in_sa_y}, in_x={in_sa_x}) -- physically "
                    f"implausible position for a displaced pellet; defer)"
                ),
                features=feats,
            )

        # Verify "similar location": >= rest_frames_total of those
        # off-pillar frames are within tolerance of the median.
        deviation = np.sqrt((pellet_x - median_x) ** 2
                            + (pellet_y - median_y) ** 2)
        deviation_radii = deviation / np.maximum(pillar_r, 1e-6)
        near_median_eligible = (
            off_pillar_eligible
            & (deviation_radii <= self.near_median_tolerance_radii)
        )
        near_median_count = int(near_median_eligible.sum())
        feats["near_median_frame_count"] = near_median_count

        if near_median_count < self.rest_frames_total:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_did_not_rest_at_similar_location "
                    f"(only {near_median_count} off-pillar frames near "
                    f"median, needed {self.rest_frames_total} -- pellet "
                    f"was bouncing/moving rather than resting; defer)"
                ),
                features=feats,
            )

        # ----- Late-segment observability: filter retrieved cases.
        # A real displaced_sa pellet stays in the SA throughout the
        # rest of the segment (can't be retrieved from SA per the
        # apparatus model). A retrieved pellet is briefly visible
        # then disappears. Require sustained off-pillar observability
        # in the last LATE_SEGMENT_FRACTION of clean zone.
        late_start_idx = int(n * (1 - self.late_segment_fraction))
        late_off_pillar_eligible = off_pillar_eligible.copy()
        late_off_pillar_eligible[:late_start_idx] = False
        late_observable_count = int(late_off_pillar_eligible.sum())
        feats["late_observable_count"] = late_observable_count
        feats["late_segment_window_start_idx"] = int(late_start_idx)

        if late_observable_count < self.late_segment_observable_min_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_not_sustained_visible_late_in_segment "
                    f"(only {late_observable_count} confident off-pillar "
                    f"frames in last {int(100*self.late_segment_fraction)}% "
                    f"of clean zone, needed "
                    f"{self.late_segment_observable_min_frames} -- "
                    f"pellet appears retrieved/anomalous, not displaced "
                    f"to SA; defer to retrieved stage)"
                ),
                features=feats,
            )

        # Find FIRST eligible frame near the off-pillar median --
        # that's when the pellet first arrives at its resting place.
        first_near_median_idx = _find_first_near_median(
            off_pillar_eligible, pellet_x, pellet_y,
            median_x, median_y, pillar_r,
            self.near_median_tolerance_radii,
        )
        feats["first_near_median_idx"] = int(first_near_median_idx)
        if first_near_median_idx < 0:
            return StageDecision(
                decision="continue",
                reason="no_eligible_frame_near_off_pillar_median_position",
                features=feats,
            )

        median_dist_radii = float(
            np.median(pellet_dist_radii[off_pillar_eligible]))
        feats["off_pillar_median_dist_radii"] = median_dist_radii

        # ----- Causal bout: walk back from the first off-pillar-
        # near-median frame to find the most recent bout that ended
        # before it. Empirically this picks the same bout as GT in
        # 92.5% of class-matched commits.
        # The on->off transition test was tried (2026-05-02) but
        # produced same trust % with lower yield -- specifically
        # didn't fix the `20250716_CNT0213_P3` problem cases where
        # DLC label-switches the pellet to a sustained off-pillar
        # position throughout the segment, defeating the "stays
        # off" check.
        rest_start_idx = first_near_median_idx
        causal_bout_idx = -1
        for bidx in range(len(bouts) - 1, -1, -1):
            if bouts[bidx][1] < rest_start_idx:
                causal_bout_idx = bidx
                break
        if causal_bout_idx < 0:
            # No bout before the rest run started -- pellet was
            # already in SA from before any reach. This is a
            # started-displaced case that should have been caught
            # by Stage 4 (or the pre-reach precondition). Defer.
            return StageDecision(
                decision="continue",
                reason=(
                    f"no_paw_past_y_line_bout_before_sustained_rest_run "
                    f"(pellet in SA before any reach in clean zone -- "
                    f"started-displaced case; not Stage 5's domain)"
                ),
                features=feats,
            )
        causal_bout_start, causal_bout_end = bouts[causal_bout_idx]

        # ----- Precision gate: pellet must be sustained ON-pillar
        # in the window IMMEDIATELY before the chosen causal bout.
        # If pellet was already off-pillar before this bout, then the
        # actual displacement happened earlier and our chosen bout
        # is wrong -- defer rather than emit a wrong-bout commit.
        # Pre-causal-bout window: from previous bout's end (or
        # segment start if first bout) to this bout's start.
        PRE_CAUSAL_ON_PILLAR_SUSTAINED = 3
        pre_causal_start = (bouts[causal_bout_idx - 1][1] + 1
                            if causal_bout_idx > 0 else 0)
        pre_causal_end = causal_bout_start
        on_pillar_eligible_inner = (
            (pellet_lk >= self.pellet_lk_threshold)
            & (pellet_dist_radii <= self.on_pillar_radii)
            & (~paw_past_y)
        )
        run_len = 0
        pre_causal_on_pillar_satisfied = False
        for i in range(pre_causal_start, pre_causal_end):
            if on_pillar_eligible_inner[i]:
                run_len += 1
                if run_len >= PRE_CAUSAL_ON_PILLAR_SUSTAINED:
                    pre_causal_on_pillar_satisfied = True
                    break
            else:
                run_len = 0
        feats["pre_causal_on_pillar_satisfied"] = bool(pre_causal_on_pillar_satisfied)
        feats["pre_causal_window_start"] = int(pre_causal_start)
        feats["pre_causal_window_end"] = int(pre_causal_end)

        if not pre_causal_on_pillar_satisfied:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_not_sustained_on_pillar_before_chosen_causal_bout "
                    f"(pre-causal window [{pre_causal_start}, "
                    f"{pre_causal_end}) did not have "
                    f"{PRE_CAUSAL_ON_PILLAR_SUSTAINED}+ consecutive "
                    f"sustained on-pillar frames -- chosen causal bout "
                    f"is suspect, defer rather than emit a wrong-bout "
                    f"commit)"
                ),
                features=feats,
            )

        # ----- Timing check: pellet should NEVER have been observed
        # at the off-pillar rest position BEFORE the causal bout
        # (even briefly). If it was, the displacement happened
        # earlier (or the apparent off-pillar position is a DLC
        # label-switch that predates any reach) -- defer.
        # ANY single frame at rest in the pre-causal window is
        # disqualifying: real displaced segments have the pellet
        # firmly on the pillar before any reach, never momentarily
        # at the off-pillar rest position.
        deviation_pre = np.sqrt((pellet_x - median_x) ** 2
                                + (pellet_y - median_y) ** 2)
        deviation_radii_pre = deviation_pre / np.maximum(pillar_r, 1e-6)
        pre_causal_at_rest_eligible = (
            (np.arange(n) < causal_bout_start)
            & (~paw_past_y)
            & (pellet_lk >= self.pellet_lk_threshold)
            & (deviation_radii_pre <= self.near_median_tolerance_radii)
        )
        pre_causal_at_rest_count = int(pre_causal_at_rest_eligible.sum())
        feats["pre_causal_at_rest_count"] = pre_causal_at_rest_count
        if pre_causal_at_rest_count > 0:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_observed_at_rest_position_before_causal_bout "
                    f"({pre_causal_at_rest_count} frames -- ANY frame "
                    f"at off-pillar rest position before the causal bout "
                    f"is disqualifying; displacement appears to have "
                    f"happened earlier OR the reading is a DLC label-"
                    f"switch that predates any reach -- defer to triage/"
                    f"next stage)"
                ),
                features=feats,
            )

        bout_length = causal_bout_end - causal_bout_start + 1
        interaction_idx = int(causal_bout_start
                              + round(self.ifr_position_in_bout * bout_length))
        # Constrain interaction_idx to be within the bout (in case
        # rounding takes it out).
        interaction_idx = max(causal_bout_start,
                              min(causal_bout_end, interaction_idx))

        # ----- OKF emit: empirical fixed offset. The pellet first
        # appears at its rest position at first_near_median_idx, and
        # GT marks OKF ~6 frames later when "pellet has settled" --
        # so we add 6. Note: per-segment velocity-based settling
        # detection was tried and produced slightly worse results
        # than this fixed offset, so we keep it simple.
        OKF_SETTLE_OFFSET = 6
        okf_idx = int(rest_start_idx) + OKF_SETTLE_OFFSET
        okf_idx = min(okf_idx, n - 1)

        interaction_frame_video = int(seg.seg_start + interaction_idx)
        okf_video = int(seg.seg_start + okf_idx)
        feats.update({
            "causal_bout_idx": int(causal_bout_idx),
            "causal_bout_start_idx": int(causal_bout_start),
            "causal_bout_end_idx": int(causal_bout_end),
            "interaction_frame_video": interaction_frame_video,
            "okf_video": okf_video,
        })
        return StageDecision(
            decision="commit",
            committed_class="displaced_sa",
            whens={
                "outcome_known_frame": okf_video,
                "interaction_frame": interaction_frame_video,
            },
            reason=(
                f"pellet_settled_in_sa_at_median_position "
                f"({near_median_count} accumulated frames near median "
                f"({median_x:.1f}, {median_y:.1f}) at "
                f"{median_dist_radii:.2f} radii off pillar; first arrival "
                f"at clean-zone idx {rest_start_idx}; causal bout "
                f"{causal_bout_idx} ended at idx {causal_bout_end})"
            ),
            features=feats,
        )
