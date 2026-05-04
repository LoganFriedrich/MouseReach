"""
Stage 4: Pellet was off-pillar from segment start and remained
observable in the SA (was never retrieved).

Semantic question this stage answers:
    "Was the pellet sustained off the pillar from the start of the
    segment, AND remained observable somewhere in the SA throughout
    the segment (i.e., never got retrieved)?"

Physical reasoning (and definition of `untouched`):
    A segment is GT untouched if no reach IN THAT SEGMENT caused
    interaction. A pellet that came in already displaced (sitting
    somewhere in the SA at the start of the segment) is a valid
    untouched case as long as nothing in this segment retrieves the
    pellet. The mouse can absolutely move the pellet around within
    the SA -- that does not make the segment displaced or anything
    else; the pellet is already off-pillar. What is physically
    IMPOSSIBLE is for the mouse to retrieve a pellet from the SA
    (the SA geometry does not permit it). So if the pellet was off-
    pillar at segment start and is observed to disappear sustained
    afterward (looks retrieved), that is a tracking anomaly -- not
    a real retrieval -- and the segment goes to TRIAGE for human
    review rather than being committed by this stage.

    Operationally:
    - Pre-bout window (frames before the first paw-past-y-line bout):
      pellet sustained at confident lk and off the pillar for at
      least N consecutive eligible frames.
    - Post-pre-bout window: pellet must be observed at confident lk
      sustained for at least M consecutive eligible (paw-not-past-
      y-line) frames somewhere later in the clean zone -- if not, the
      pellet appears to have been retrieved, which is impossible from
      a starting off-pillar state -> triage.

Co-detection triage (pre-check):
    A frame where Pellet AND Pillar are both detected at high lk
    within 1 pillar-radius of each other is physically impossible
    -- the pellet sits ON the pillar and 3D-occludes it. Sustained
    co-detection indicates a DLC tracking artifact (e.g. plexiglass
    over the SA causing reflection/refraction issues). Such
    segments are TRIAGED, not committed by this stage. See memory
    `pellet_pillar_cooccurrence_is_artifact.md`.

Cascade order:
    Runs after Stages 1, 2, 3 (which together commit the
    pellet-on-pillar-throughout, no-reach-could-have-touched, and
    pellet-returned-to-pillar untouched cases). What reaches Stage
    4 is segments where Stage 3 didn't see the pellet return to
    the pillar after the last reach AND none of the earlier stages
    fired -- which is precisely the regime where "pellet was off
    pillar from segment start and never moved" is the relevant
    pattern.

Cascade emit on commit:
    - committed_class: "untouched"
    - whens["outcome_known_frame"]: seg_end - TRANSITION_ZONE_HALF
      (parallel to Stages 1, 2, 3)
    - whens["interaction_frame"]: None (no interaction)

Cascade emit on triage:
    - decision: "triage"
    - reason: co-detection signature observed
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

# Likelihood thresholds.
# Codetection triage uses VERY HIGH likelihood thresholds for BOTH
# pellet and pillar AND very tight distance, per 2026-05-02 user
# refinement. Rationale: the codetection triage should only fire when
# the algo is confident the bodyparts are physically overlapping
# (true DLC label-switch). Loose thresholds were over-triaging
# legitimate edge-proximity cases (pellet sitting near pillar tip).
PELLET_LK_THR = 0.7
CODETECTION_PELLET_LK_THR = 0.95
CODETECTION_PILLAR_LK_THR = 0.95
PILLAR_LK_THR = 0.7
PAW_LK_THR = 0.5

# Co-detection triage threshold: Pellet AND Pillar both at high lk
# within this many pillar-radii of each other for sustained frames
# = DLC tracking artifact -> triage.
CODETECTION_DISTANCE_RADII = 0.5
CODETECTION_SUSTAINED_FRAMES = 3

# "Off pillar" requires the pellet centroid to be clearly in the
# scoring area (SA), not just past the pillar edge. The geometry:
# pillar center is at ~9.44 * pillar_r above SA midpoint (from the
# pillar-geometry formula `pillar_cy = SA_mid_y - 0.944 * ruler`).
# Empirically, GT displaced_sa pellets sit at p10=3.60, p50=4.49,
# p90=6.31 radii from the calc'd pillar center after the pellet has
# settled in the SA (37 train-pool videos, 337 segments, 2026-05-02).
# Threshold of 3.0 sits below the bulk of the displaced distribution
# (catches ~95% of true displaced_sa) while staying clear of pellet-
# at-pillar-edge calibration artifacts (which sit at 1-2 radii).
PELLET_OFF_PILLAR_RADII = 3.0

# Segment-start window: how many frames at the very start of the
# clean zone to inspect for off-pillar evidence. We're looking for
# "as we enter the segment, the pellet is already off the pillar"
# -- the goal is to characterize the pellet's INCOMING state, not
# anything that happens in the middle/end of the segment.
SEGMENT_START_WINDOW_FRAMES = 30

# Within the segment-start window, the pellet must be observed
# sustained at confident lk AND off-pillar for at least this many
# consecutive eligible frames.
SEGMENT_START_OFF_PILLAR_SUSTAINED_FRAMES = 5

# "Never returns to pillar": the predicate "pellet sustained at
# confident lk AND ON pillar (within `1.0 * pillar_r` -- using the
# tight on-pillar definition, complement of the off-pillar 1.5
# threshold)" must fire NOWHERE in the rest of the clean zone for
# this many consecutive frames. If the pellet IS observed sustained
# on-pillar later in the segment, either our segment-start detection
# was wrong or the pellet "returned" to the pillar (impossible per
# `pellet_cannot_return_to_pillar.md`). Either way Stage 4's case
# doesn't apply -- defer.
RETURN_TO_PILLAR_SUSTAINED_FRAMES = 5
ON_PILLAR_THRESHOLD_RADII = 1.2  # buffered: matches Stage 3's
                                  # on-pillar definition (pillar edge
                                  # plus pellet-size buffer)

# SA-stability precondition: tray motion can be in progress at the
# very start of a segment (segmenter places the boundary near a tray
# event but the tray takes some frames to fully settle). We must wait
# for the SA bodyparts to be stable before evaluating the pellet's
# "off-pillar at start" check, otherwise the pellet appears at varying
# positions during settling and post-settling DLC tracking can have
# small (~2 px) offsets that fall just outside the off-pillar
# threshold even when the pellet is physically still on the pillar.
SA_CENTROID_VELOCITY_STABLE_THR_PX = 2.0    # frame-to-frame velocity
                                             # below this = stable
SA_STABLE_SUSTAINED_FRAMES = 5               # this many consecutive
                                             # stable frames define
                                             # the settled point
MAX_SETTLING_SEARCH_FRAMES = 200             # don't search forever
                                             # for settling

# Post-pre-bout reappearance evidence: somewhere AFTER the pre-bout
# off-pillar evidence (in eligible / paw-not-past-y-line frames),
# the pellet must be observed at confident lk sustained for at least
# this many consecutive frames -- otherwise the pellet appears to
# have been retrieved, which is impossible from a starting off-
# pillar state, and the segment is triaged.
POST_PRE_BOUT_REAPPEARANCE_SUSTAINED_FRAMES = 5


def _find_paw_past_y_line_bouts(
    paw_past_y: np.ndarray,
) -> List[Tuple[int, int]]:
    """List of (start, end_inclusive) bouts where paw_past_y is True."""
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
    pellet_lk: np.ndarray,
    pillar_lk: np.ndarray,
    pellet_x: np.ndarray,
    pellet_y: np.ndarray,
    pillar_x: np.ndarray,
    pillar_y: np.ndarray,
    pillar_r: np.ndarray,
    pellet_lk_thr: float,
    pillar_lk_thr: float,
    distance_radii_thr: float,
    sustained_frames: int,
) -> bool:
    """Return True if a sustained run of frames satisfies the
    co-detection triage predicate (Pellet AND Pillar both at high lk
    within `distance_radii_thr` of each other).
    """
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


class Stage5PelletOffPillarThroughout(Stage):
    name = "stage_5_pellet_off_pillar_throughout"
    target_class = "untouched"

    def __init__(
        self,
        codetection_distance_radii: float = CODETECTION_DISTANCE_RADII,
        codetection_sustained_frames: int = CODETECTION_SUSTAINED_FRAMES,
        pellet_off_pillar_radii: float = PELLET_OFF_PILLAR_RADII,
        on_pillar_threshold_radii: float = ON_PILLAR_THRESHOLD_RADII,
        segment_start_window_frames: int = SEGMENT_START_WINDOW_FRAMES,
        segment_start_off_pillar_sustained_frames: int = SEGMENT_START_OFF_PILLAR_SUSTAINED_FRAMES,
        return_to_pillar_sustained_frames: int = RETURN_TO_PILLAR_SUSTAINED_FRAMES,
        post_pre_bout_reappearance_sustained_frames: int = POST_PRE_BOUT_REAPPEARANCE_SUSTAINED_FRAMES,
        sa_centroid_velocity_stable_thr_px: float = SA_CENTROID_VELOCITY_STABLE_THR_PX,
        sa_stable_sustained_frames: int = SA_STABLE_SUSTAINED_FRAMES,
        max_settling_search_frames: int = MAX_SETTLING_SEARCH_FRAMES,
        pellet_lk_threshold: float = PELLET_LK_THR,
        pillar_lk_threshold: float = PILLAR_LK_THR,
        paw_lk_threshold: float = PAW_LK_THR,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.codetection_distance_radii = codetection_distance_radii
        self.codetection_sustained_frames = codetection_sustained_frames
        self.pellet_off_pillar_radii = pellet_off_pillar_radii
        self.on_pillar_threshold_radii = on_pillar_threshold_radii
        self.segment_start_window_frames = segment_start_window_frames
        self.segment_start_off_pillar_sustained_frames = segment_start_off_pillar_sustained_frames
        self.return_to_pillar_sustained_frames = return_to_pillar_sustained_frames
        self.post_pre_bout_reappearance_sustained_frames = post_pre_bout_reappearance_sustained_frames
        self.sa_centroid_velocity_stable_thr_px = sa_centroid_velocity_stable_thr_px
        self.sa_stable_sustained_frames = sa_stable_sustained_frames
        self.max_settling_search_frames = max_settling_search_frames
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

        # Cleaning: SA + Pellet (Pillar bodypart NOT cleaned -- low lk
        # is the expected occluded state and we use it as a real signal).
        sub = clean_dlc_bodyparts(
            sub_raw, other_bodyparts_to_clean=("Pellet",))

        # Pillar geometry from cleaned SA.
        geom = compute_pillar_geometry_series(sub)
        pillar_cx = geom["pillar_cx"].to_numpy()
        pillar_cy = geom["pillar_cy"].to_numpy()
        pillar_r = geom["pillar_r"].to_numpy()
        slit_y_line = pillar_cy + pillar_r

        # Pellet position (cleaned for x/y; original lk).
        pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
        pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
        pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
        pellet_dist = np.sqrt((pellet_x - pillar_cx) ** 2 +
                              (pellet_y - pillar_cy) ** 2)
        pellet_dist_radii = pellet_dist / np.maximum(pillar_r, 1e-6)

        # Pillar bodypart (raw).
        pillar_x_raw = sub_raw["Pillar_x"].to_numpy(dtype=float)
        pillar_y_raw = sub_raw["Pillar_y"].to_numpy(dtype=float)
        pillar_lk_raw = sub_raw["Pillar_likelihood"].to_numpy(dtype=float)

        # Co-detection triage check (FIRST priority). Uses dedicated
        # codetection lk thresholds (much higher than the stage's
        # general pellet/pillar lk thresholds) so triage fires only
        # when both bodyparts are confidently at the same location --
        # i.e., a true DLC label-switch artifact -- not when the
        # pellet just happens to sit near the pillar edge with one
        # bodypart at moderate lk.
        codetection = _detect_codetection_triage(
            pellet_lk, pillar_lk_raw,
            pellet_x, pellet_y,
            pillar_x_raw, pillar_y_raw,
            pillar_r,
            CODETECTION_PELLET_LK_THR, CODETECTION_PILLAR_LK_THR,
            self.codetection_distance_radii,
            self.codetection_sustained_frames,
        )
        if codetection:
            # Per 2026-05-02 user reframing: the codetection signature
            # at THIS stage doesn't prove triage is needed; it just
            # proves the pellet wasn't untouched. Defer rather than
            # triage; downstream stages (or a future dedicated
            # codetection-triage stage) can decide what to do.
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_pillar_codetection_observed_defer "
                    f"(both bodyparts at high lk within "
                    f"{self.codetection_distance_radii} pillar-radii "
                    f"of each other for {self.codetection_sustained_frames}+ "
                    f"sustained frames -- not Stage 5's case to triage; defer)"
                ),
                features={"codetection_observed": True},
            )

        # Per-frame paw-past-y-line (used for masking eligibility).
        paw_past_y = np.zeros(n, dtype=bool)
        for bp in PAW_BODYPARTS:
            paw_y = sub[f"{bp}_y"].to_numpy(dtype=float)
            paw_lk = sub[f"{bp}_likelihood"].to_numpy(dtype=float)
            paw_past_y |= (paw_y <= slit_y_line) & (paw_lk >= self.paw_lk_threshold)

        bouts = _find_paw_past_y_line_bouts(paw_past_y)

        feats = {
            "n_clean_zone_frames": int(n),
            "n_paw_past_y_line_bouts": int(len(bouts)),
        }

        # ----- SA-stability precondition: tray motion can be in
        # progress at the start of the segment (segmenter places the
        # boundary near a tray event but the tray takes time to
        # settle). Find when the SA bodyparts have been stable
        # (centroid velocity < threshold) for sustained_frames
        # consecutive frames -- that is the "settled" point. We only
        # evaluate the pellet's "off-pillar at start" check AFTER
        # this settled point.
        sa_centroid_x = (sub["SABL_x"].to_numpy(dtype=float)
                         + sub["SABR_x"].to_numpy(dtype=float)
                         + sub["SATL_x"].to_numpy(dtype=float)
                         + sub["SATR_x"].to_numpy(dtype=float)) / 4.0
        sa_centroid_y = (sub["SABL_y"].to_numpy(dtype=float)
                         + sub["SABR_y"].to_numpy(dtype=float)
                         + sub["SATL_y"].to_numpy(dtype=float)
                         + sub["SATR_y"].to_numpy(dtype=float)) / 4.0
        sa_velocity = np.zeros(n)
        if n > 1:
            sa_velocity[1:] = np.sqrt(
                np.diff(sa_centroid_x) ** 2 + np.diff(sa_centroid_y) ** 2)
        sa_stable = sa_velocity < self.sa_centroid_velocity_stable_thr_px
        # Find first index where sa_stable has been True for
        # `sa_stable_sustained_frames` consecutive frames.
        settled_idx = -1
        run_len = 0
        search_end = min(n, self.max_settling_search_frames)
        for i in range(search_end):
            if sa_stable[i]:
                run_len += 1
                if run_len >= self.sa_stable_sustained_frames:
                    # The "settled" point is the start of this
                    # sustained-stable run.
                    settled_idx = i - self.sa_stable_sustained_frames + 1
                    break
            else:
                run_len = 0
        feats["sa_settled_idx"] = int(settled_idx)

        if settled_idx < 0:
            return StageDecision(
                decision="continue",
                reason=(
                    f"sa_never_settled_within_{self.max_settling_search_frames}_frames "
                    f"(SA centroid velocity stayed above "
                    f"{self.sa_centroid_velocity_stable_thr_px} px/frame; "
                    f"tray motion ongoing -- not a Stage 4 case)"
                ),
                features=feats,
            )

        # ----- Check 1: pellet was OFF the pillar as we entered the
        # segment. Look in the segment-start window (starting from
        # the settled point) but CAPPED at the first paw-past-y-line
        # bout. The cap is critical: if a reach happens within the
        # window, post-reach frames could read as off-pillar because
        # the pellet IS displaced -- those don't count as evidence of
        # "started off-pillar." We only want pre-reach off-pillar
        # evidence here.
        first_bout_start = bouts[0][0] if bouts else n
        segment_start_begin = settled_idx
        segment_start_end = min(
            n,
            settled_idx + self.segment_start_window_frames,
            first_bout_start,
        )
        feats["first_bout_start_idx"] = int(first_bout_start)
        feats["segment_start_window_begin"] = int(segment_start_begin)
        feats["segment_start_window_capped_at"] = int(segment_start_end)
        segment_start_eligible = (
            (np.arange(n) >= segment_start_begin)
            & (np.arange(n) < segment_start_end)
            & (~paw_past_y)
            & (pellet_lk >= self.pellet_lk_threshold)
            & (pellet_dist_radii > self.pellet_off_pillar_radii)
        )
        run_len = 0
        seg_start_satisfied = False
        seg_start_run_end_idx = -1
        for i in range(segment_start_begin, segment_start_end):
            if segment_start_eligible[i]:
                run_len += 1
                if run_len >= self.segment_start_off_pillar_sustained_frames:
                    seg_start_satisfied = True
                    seg_start_run_end_idx = i
                    break
            else:
                run_len = 0
        feats["segment_start_window_end"] = int(segment_start_end)
        feats["segment_start_off_pillar_satisfied"] = bool(seg_start_satisfied)

        if not seg_start_satisfied:
            return StageDecision(
                decision="continue",
                reason=(
                    f"no_sustained_off_pillar_pellet_in_segment_start_window "
                    f"(needed {self.segment_start_off_pillar_sustained_frames}+ "
                    f"consecutive eligible frames with pellet_lk >= "
                    f"{self.pellet_lk_threshold} AND pellet_dist > "
                    f"{self.pellet_off_pillar_radii} radii in the first "
                    f"{self.segment_start_window_frames} clean-zone frames)"
                ),
                features=feats,
            )

        # ----- Check 2: pellet NEVER returns to the pillar in the
        # rest of the clean zone. If pellet is observed sustained
        # ON the pillar later in the segment, either (a) our
        # segment-start detection was wrong (pellet was actually on
        # pillar), or (b) the pellet "returned" to the pillar
        # (physically impossible per `pellet_cannot_return_to_pillar`).
        # Either way Stage 4's case doesn't apply.
        on_pillar_eligible = (
            (~paw_past_y)
            & (pellet_lk >= self.pellet_lk_threshold)
            & (pellet_dist_radii <= self.on_pillar_threshold_radii)
        )
        run_len = 0
        return_to_pillar_observed = False
        return_to_pillar_idx = -1
        for i in range(seg_start_run_end_idx + 1, n):
            if on_pillar_eligible[i]:
                run_len += 1
                if run_len >= self.return_to_pillar_sustained_frames:
                    return_to_pillar_observed = True
                    return_to_pillar_idx = i
                    break
            else:
                run_len = 0
        feats["return_to_pillar_observed"] = bool(return_to_pillar_observed)
        if return_to_pillar_idx >= 0:
            feats["return_to_pillar_idx"] = int(return_to_pillar_idx)

        if return_to_pillar_observed:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_observed_back_on_pillar_after_segment_start "
                    f"(sustained {self.return_to_pillar_sustained_frames}+ "
                    f"on-pillar frames at idx {return_to_pillar_idx} -- "
                    f"either pellet was on-pillar from start (segment-start "
                    f"detection was a transient artifact) or the pellet "
                    f"impossibly returned to pillar; either way Stage 4's "
                    f"case does not apply)"
                ),
                features=feats,
            )

        # ----- Check 3: pellet remains observable somewhere in the
        # clean zone (didn't apparent-retrieval). If pellet is never
        # confidently detected after the segment-start window,
        # something anomalous happened from a starting off-pillar
        # state -- triage for human review.
        observable_eligible = (
            (~paw_past_y)
            & (pellet_lk >= self.pellet_lk_threshold)
        )
        run_len = 0
        observable_satisfied = False
        for i in range(seg_start_run_end_idx + 1, n):
            if observable_eligible[i]:
                run_len += 1
                if run_len >= self.post_pre_bout_reappearance_sustained_frames:
                    observable_satisfied = True
                    break
            else:
                run_len = 0
        feats["pellet_remains_observable"] = bool(observable_satisfied)

        if not observable_satisfied:
            return StageDecision(
                decision="triage",
                reason=(
                    f"pellet_appears_retrieved_from_off_pillar_state "
                    f"(no sustained {self.post_pre_bout_reappearance_sustained_frames}+ "
                    f"frame run of confident pellet detection after the "
                    f"segment-start off-pillar evidence -- physically "
                    f"impossible from a starting off-pillar state, anomaly "
                    f"for human review)"
                ),
                features=feats,
            )

        # All three checks passed: pellet was off-pillar at segment
        # start, never returned to the pillar, and remained
        # observable in the SA (mouse may have moved it but did not
        # retrieve it). Commit untouched.
        okf = clean_end
        feats["outcome_known_frame_emitted"] = int(okf)
        return StageDecision(
            decision="commit",
            committed_class="untouched",
            whens={"outcome_known_frame": int(okf),
                   "interaction_frame": None},
            reason=(
                f"pellet_off_pillar_at_segment_start_and_stayed_off_throughout "
                f"(segment-start sustained off-pillar evidence + pellet never "
                f"returned to pillar + pellet remained observable -- mouse may "
                f"have moved pellet within SA but did not retrieve it)"
            ),
            features=feats,
        )
