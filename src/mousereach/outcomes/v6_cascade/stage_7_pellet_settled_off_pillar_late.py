"""
Stage 7: Pellet settled off-pillar in the SA at the end of the segment.

Question this stage answers:
    "In the late portion of the clean zone, is the pellet sustained
    OFF the pillar at a stable, well-detected location inside the SA
    quadrilateral?"

If yes -> COMMIT displaced_sa. The pellet was demonstrably off the
pillar and at rest in the SA at the end of the segment, regardless of
what happened pre-reach. Mirror of Stage 1's "position-never-changed"
applied to the END of the segment (the user's 2026-05-02 framing).

Why this stage exists:
    Stage 8 (formerly the only displaced_sa stage) requires the pellet
    to have been confidently on-pillar BEFORE the causal reach AND
    sustained off-pillar after AND a clean causal-bout walk-back. Real
    displaced cases often fail the pre-reach precondition because the
    paw approaching the pellet obscures the pellet bodypart -- DLC
    can't confidently detect the pellet for the few frames before the
    reach. Stage 7 ignores pre-reach evidence and commits based purely
    on the strong post-reach off-pillar SA evidence.

Defenses against false-commit:
- Use pillar-relative coords for the position-stability test, so
  apparatus tray drift across the segment doesn't inflate std
- Pellet position must lie inside the SA quadrilateral (rules out
  geometrically implausible "displaced" cases)
- Pellet must be detected at high lk (>= 0.95) for ALL frames in the
  test window (filters DLC label-switch at moderate lk)
- Required late-segment off-pillar count is large (>= 100 frames),
  matching real-pellet-at-rest evidence patterns

Cascade emit on commit:
- committed_class: "displaced_sa"
- whens["interaction_frame"]: middle of the most recent paw-past-y-
  line bout that ended before the rest period began (empirical
  p50 = 0.4 of bout length within bout)
- whens["outcome_known_frame"]: rest_start + small settle offset
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import SegmentInput, Stage, StageDecision


PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5

# Late-segment window: focus on the last LATE_FRAC fraction of the
# clean zone. Real displaced pellets settle and stay there; the late
# segment has the cleanest "settled" evidence.
LATE_SEGMENT_FRACTION = 0.50  # last 50% of clean zone

# Confidence and physical-plausibility thresholds.
PELLET_LK_THR = 0.95           # high lk filters DLC label-switch
ANCHOR_PELLET_LK_THR = 0.7     # more permissive for the off-pillar-
                               # between-reaches anchor (DLC less
                               # confident immediately post-reach but
                               # the pellet is still observably
                               # displaced)
PAW_LK_THR = 0.5
PELLET_OFF_PILLAR_RADII = 1.0  # outside the pillar circle
ANCHOR_OFF_PILLAR_RADII = 0.85 # slightly looser for transition anchor
SA_QUAD_Y_BUFFER_PX = 0        # no buffer; strict SA quadrilateral

# Sustained near-median test (matches Stage 8's analogous check).
# In SA-normalized (u, v) coords. Pellet at rest on the SA tray
# gives a tight (u, v) cluster; this tolerance defines the cluster
# radius. 0.075 ~= 1.5 pillar radii (since pillar_r/SA_width ~= 0.05).
NEAR_MEDIAN_TOLERANCE_UV = 0.075
NEAR_MEDIAN_TOLERANCE_RADII = 1.5
MIN_NEAR_MEDIAN_FRAMES = 100   # >= 100 sustained frames near median

# On-pillar starting-state precondition.
ON_PILLAR_RADII = 1.0           # within 1 radius = on pillar
MIN_ON_PILLAR_FRAMES = 5        # need at least this many on-pillar frames
                                # somewhere in clean zone

# Causal bout selection (user 2026-05-02 algorithm):
#   "For a reach to have displaced a pellet from the pillar, the
#    pellet has to have been on the pillar BEFORE that reach and off
#    AFTER it."
# Per-bout test:
#   - PRE_BOUT_WINDOW frames before bout start: pellet observed
#     confidently on-pillar (>= MIN_PRE_BOUT_ON_PILLAR_FRAMES)
#   - POST_BOUT_WINDOW frames after bout end: pellet observed
#     confidently off-pillar (>= MIN_POST_BOUT_OFF_PILLAR_FRAMES)
# First bout (in segment order) matching both = causal bout.
PRE_BOUT_WINDOW = 2  # user 2026-05-03: just the immediate frame(s)
                     # before reach start. Pellet either was on pillar
                     # right before the reach or it wasn't -- a wide
                     # window adds noise from intervening events.
POST_BOUT_WINDOW = 30
MIN_PRE_BOUT_ON_PILLAR_FRAMES = 1   # 1 of 2 immediate-pre frames
                                     # showing on-pillar
MIN_POST_BOUT_OFF_PILLAR_FRAMES = 5

# Immediate-after post-off check (2026-05-03): the chosen reach
# itself must put the pellet off-pillar IMMEDIATELY after the reach
# ends, not somewhere in the next 30 frames. The 30-frame post
# window can be polluted by displacement from a SUBSEQUENT reach
# (when reaches are close together), causing wrong-pick reaches to
# falsely satisfy post-off. Tight immediate-after window is the
# fix. Used in addition to the 30-frame post-off check for sanity.
POST_BOUT_IMMEDIATE_WINDOW = 10
MIN_POST_BOUT_IMMEDIATE_OFF_FRAMES = 3

# Causal-reach selection by pellet displacement across the reach.
# Empirical 2026-05-02: across the corpus's GT-causal reaches, pellet
# position median (30f pre vs 30f post) shifts by >= 1.7 radii on
# every GT reach. Non-causal reaches almost always show < 0.3 radii
# (p95 < 3 radii). Threshold 1.5 separates with a clean margin.
REACH_DISPLACEMENT_WINDOW = 30
REACH_DISPLACEMENT_THRESHOLD_RADII = 1.5
# Pre-reach on-pillar evidence (100-frame window) required when the
# displacement test is satisfied. Empirical: correct GT reach has
# p25=38, wrong GT reach p25=33; non-GT reaches in correct segments
# p25=0. Threshold of 20 cleanly separates.
MIN_PRE_ON_PILLAR_W100 = 20

# IFR / OKF emit.
IFR_POSITION_IN_BOUT = 0.4
OKF_SETTLE_OFFSET = 6


def _find_paw_past_y_line_bouts(paw_past_y: np.ndarray) -> List[Tuple[int, int]]:
    n = len(paw_past_y)
    bouts: List[Tuple[int, int]] = []
    rs = -1
    for i in range(n):
        if paw_past_y[i]:
            if rs < 0:
                rs = i
        else:
            if rs >= 0:
                bouts.append((rs, i - 1))
                rs = -1
    if rs >= 0:
        bouts.append((rs, n - 1))
    return bouts


class Stage7PelletSettledOffPillarLate(Stage):
    name = "stage_7_pellet_settled_off_pillar_late"
    target_class = "displaced_sa"

    def __init__(
        self,
        late_segment_fraction: float = LATE_SEGMENT_FRACTION,
        pellet_lk_threshold: float = PELLET_LK_THR,
        anchor_pellet_lk_threshold: float = ANCHOR_PELLET_LK_THR,
        paw_lk_threshold: float = PAW_LK_THR,
        pellet_off_pillar_radii: float = PELLET_OFF_PILLAR_RADII,
        anchor_off_pillar_radii: float = ANCHOR_OFF_PILLAR_RADII,
        on_pillar_radii: float = ON_PILLAR_RADII,
        min_on_pillar_frames: int = MIN_ON_PILLAR_FRAMES,
        near_median_tolerance_radii: float = NEAR_MEDIAN_TOLERANCE_RADII,
        near_median_tolerance_uv: float = NEAR_MEDIAN_TOLERANCE_UV,
        min_near_median_frames: int = MIN_NEAR_MEDIAN_FRAMES,
        pre_bout_window: int = PRE_BOUT_WINDOW,
        post_bout_window: int = POST_BOUT_WINDOW,
        min_pre_bout_on_pillar_frames: int = MIN_PRE_BOUT_ON_PILLAR_FRAMES,
        min_post_bout_off_pillar_frames: int = MIN_POST_BOUT_OFF_PILLAR_FRAMES,
        post_bout_immediate_window: int = POST_BOUT_IMMEDIATE_WINDOW,
        min_post_bout_immediate_off_frames: int = MIN_POST_BOUT_IMMEDIATE_OFF_FRAMES,
        reach_displacement_window: int = REACH_DISPLACEMENT_WINDOW,
        reach_displacement_threshold_radii: float = REACH_DISPLACEMENT_THRESHOLD_RADII,
        min_pre_on_pillar_w100: int = MIN_PRE_ON_PILLAR_W100,
        ifr_position_in_bout: float = IFR_POSITION_IN_BOUT,
        okf_settle_offset: int = OKF_SETTLE_OFFSET,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.late_segment_fraction = late_segment_fraction
        self.pellet_lk_threshold = pellet_lk_threshold
        self.anchor_pellet_lk_threshold = anchor_pellet_lk_threshold
        self.paw_lk_threshold = paw_lk_threshold
        self.pellet_off_pillar_radii = pellet_off_pillar_radii
        self.anchor_off_pillar_radii = anchor_off_pillar_radii
        self.on_pillar_radii = on_pillar_radii
        self.min_on_pillar_frames = min_on_pillar_frames
        self.near_median_tolerance_radii = near_median_tolerance_radii
        self.near_median_tolerance_uv = near_median_tolerance_uv
        self.min_near_median_frames = min_near_median_frames
        self.pre_bout_window = pre_bout_window
        self.post_bout_window = post_bout_window
        self.min_pre_bout_on_pillar_frames = min_pre_bout_on_pillar_frames
        self.min_post_bout_off_pillar_frames = min_post_bout_off_pillar_frames
        self.post_bout_immediate_window = post_bout_immediate_window
        self.min_post_bout_immediate_off_frames = min_post_bout_immediate_off_frames
        self.reach_displacement_window = reach_displacement_window
        self.reach_displacement_threshold_radii = reach_displacement_threshold_radii
        self.min_pre_on_pillar_w100 = min_pre_on_pillar_w100
        self.ifr_position_in_bout = ifr_position_in_bout
        self.okf_settle_offset = okf_settle_offset
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

        sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
        geom = compute_pillar_geometry_series(sub)
        pillar_cx = geom["pillar_cx"].to_numpy(dtype=float)
        pillar_cy = geom["pillar_cy"].to_numpy(dtype=float)
        pillar_r = geom["pillar_r"].to_numpy(dtype=float)
        slit_y_line = pillar_cy + pillar_r

        pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
        pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
        pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
        pellet_dist_radii = (
            np.sqrt((pellet_x - pillar_cx) ** 2
                    + (pellet_y - pillar_cy) ** 2)
            / np.maximum(pillar_r, 1e-6)
        )

        # SA-rectangle-normalized pellet coordinates. The SA forms a
        # quadrilateral with corners SATL/SATR (top) and SABL/SABR
        # (bottom). Define an affine frame using SABL as origin,
        # (SABR - SABL) as the x-basis, (SATL - SABL) as the y-basis.
        # Pellet position in (u, v) coords cancels apparatus
        # translation, rotation, AND scale -- much more robust for
        # rest-position consistency than pillar-relative (single-point
        # translation only).
        sabl_x = sub["SABL_x"].to_numpy(dtype=float)
        sabl_y = sub["SABL_y"].to_numpy(dtype=float)
        sabr_x = sub["SABR_x"].to_numpy(dtype=float)
        sabr_y = sub["SABR_y"].to_numpy(dtype=float)
        satl_x = sub["SATL_x"].to_numpy(dtype=float)
        satl_y = sub["SATL_y"].to_numpy(dtype=float)
        ex_x = sabr_x - sabl_x
        ex_y = sabr_y - sabl_y
        ey_x = satl_x - sabl_x
        ey_y = satl_y - sabl_y
        # Determinant for inverse matrix
        det = ex_x * ey_y - ex_y * ey_x
        det_safe = np.where(np.abs(det) > 1e-6, det, 1e-6)
        rel_x = pellet_x - sabl_x
        rel_y = pellet_y - sabl_y
        # Pellet in SA-normalized coords (u, v):
        pellet_u = (rel_x * ey_y - rel_y * ey_x) / det_safe
        pellet_v = (-rel_x * ex_y + rel_y * ex_x) / det_safe

        # Per-frame paw-past-y-line.
        paw_past_y = np.zeros(n, dtype=bool)
        for bp in PAW_BODYPARTS:
            paw_y = sub[f"{bp}_y"].to_numpy(dtype=float)
            paw_lk = sub[f"{bp}_likelihood"].to_numpy(dtype=float)
            paw_past_y |= (paw_y <= slit_y_line) & (paw_lk >= self.paw_lk_threshold)

        # On-pillar evidence (used both as a precondition and to
        # anchor the on->off transition test for causal-bout
        # selection). On-pillar frame = pellet high-lk + within 1
        # radius + no paw past y-line.
        on_pillar = (
            (pellet_lk >= self.pellet_lk_threshold)
            & (pellet_dist_radii <= self.on_pillar_radii)
            & (~paw_past_y)
        )
        n_on_pillar = int(on_pillar.sum())
        if n_on_pillar < self.min_on_pillar_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_never_observed_on_pillar "
                    f"({n_on_pillar} on-pillar frames in clean zone "
                    f"< {self.min_on_pillar_frames}; can't establish "
                    f"a starting state -- not Stage 7's case)"
                ),
                features={"n_on_pillar": n_on_pillar},
            )

        # Late-segment window definition.
        late_start_idx = int(n * (1 - self.late_segment_fraction))
        late_mask = np.zeros(n, dtype=bool)
        late_mask[late_start_idx:] = True

        # Eligible frames in the late window: confident pellet, off-
        # pillar, no paw past y-line.
        eligible = (
            late_mask
            & (pellet_lk >= self.pellet_lk_threshold)
            & (pellet_dist_radii > self.pellet_off_pillar_radii)
            & (~paw_past_y)
        )
        n_eligible = int(eligible.sum())

        feats = {
            "n_clean_zone_frames": n,
            "late_segment_window_start_idx": int(late_start_idx),
            "n_late_eligible_frames": n_eligible,
        }

        if n_eligible < self.min_near_median_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_late_segment_off_pillar_evidence "
                    f"({n_eligible} eligible frames < "
                    f"{self.min_near_median_frames}; pellet not "
                    f"sufficiently sustained off-pillar in late segment)"
                ),
                features=feats,
            )

        median_x = float(np.median(pellet_x[eligible]))
        median_y = float(np.median(pellet_y[eligible]))
        median_dist_radii = float(np.median(pellet_dist_radii[eligible]))
        feats.update({
            "off_pillar_median_x": median_x,
            "off_pillar_median_y": median_y,
            "off_pillar_median_dist_radii": median_dist_radii,
        })

        # Within-SA-quadrilateral check (matches Stage 8's logic).
        sa_top_y = float(np.median(
            (sub["SATL_y"].to_numpy() + sub["SATR_y"].to_numpy())[eligible] / 2.0))
        sa_bot_y = float(np.median(
            (sub["SABL_y"].to_numpy() + sub["SABR_y"].to_numpy())[eligible] / 2.0))
        sa_left_x = float(np.median(
            np.minimum(sub["SABL_x"].to_numpy(), sub["SATL_x"].to_numpy())[eligible]))
        sa_right_x = float(np.median(
            np.maximum(sub["SABR_x"].to_numpy(), sub["SATR_x"].to_numpy())[eligible]))
        in_sa_y = sa_top_y <= median_y <= sa_bot_y
        in_sa_x = sa_left_x <= median_x <= sa_right_x
        feats.update({
            "sa_top_y": sa_top_y, "sa_bot_y": sa_bot_y,
            "sa_left_x": sa_left_x, "sa_right_x": sa_right_x,
            "median_in_sa_quadrilateral": bool(in_sa_y and in_sa_x),
        })

        if not (in_sa_y and in_sa_x):
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_median_outside_sa_quadrilateral "
                    f"(median ({median_x:.1f},{median_y:.1f}); SA "
                    f"y=[{sa_top_y:.1f},{sa_bot_y:.1f}], "
                    f"x=[{sa_left_x:.1f},{sa_right_x:.1f}]); physically "
                    f"implausible position for displaced pellet"
                ),
                features=feats,
            )

        # Sustained near-median test in SA-NORMALIZED coordinates.
        # Pellet (u, v) cancels apparatus translation/rotation/scale,
        # so a pellet truly at rest gives a tight (u, v) cluster
        # regardless of tray motion. Tolerance is in normalized units
        # (fraction of SA width/height).
        median_u = float(np.median(pellet_u[eligible]))
        median_v = float(np.median(pellet_v[eligible]))
        feats["off_pillar_median_u"] = median_u
        feats["off_pillar_median_v"] = median_v
        deviation_uv = np.sqrt((pellet_u - median_u) ** 2
                               + (pellet_v - median_v) ** 2)
        # Convert near_median_tolerance_radii (px-radii basis) to a
        # comparable normalized tolerance. Pillar_r ~= 0.05 of ruler
        # = 0.05 of SA width. So 1.5 radii ~= 0.075 SA-width units.
        # We pick a comparable normalized tolerance below.
        near_median_late = (
            eligible
            & (deviation_uv <= self.near_median_tolerance_uv)
        )
        n_near_median_late = int(near_median_late.sum())
        feats["n_near_median_in_late_window"] = n_near_median_late

        if n_near_median_late < self.min_near_median_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_late_window_not_sustained_at_single_location "
                    f"({n_near_median_late} near-median < "
                    f"{self.min_near_median_frames}; pellet was off-"
                    f"pillar but moving rather than settled)"
                ),
                features=feats,
            )

        # CAUSAL BOUT (user 2026-05-02): "for a reach to have displaced
        # the pellet, the pellet has to have been on the pillar BEFORE
        # the reach and off AFTER it." For each bout, check pre-bout
        # on-pillar AND post-bout off-pillar windows. First bout (in
        # segment order) satisfying BOTH = causal.
        bouts = _find_paw_past_y_line_bouts(paw_past_y)
        if not bouts:
            return StageDecision(
                decision="continue",
                reason=(
                    f"no_paw_past_y_line_bouts_in_clean_zone "
                    f"(can't attribute causal reach without paw activity)"
                ),
                features=feats,
            )

        confident = pellet_lk >= self.pellet_lk_threshold
        off_pillar_confident = (
            confident
            & (pellet_dist_radii > self.pellet_off_pillar_radii)
            & (~paw_past_y)
        )

        # CAUSAL REACH (empirical 2026-05-02): the pellet actually MOVED
        # across the causal reach. Compute median pellet position in
        # 30-frame pre and 30-frame post windows for each reach; the
        # reach where pellet position changes by >= REACH_DISPLACEMENT_
        # THRESHOLD_RADII is the causal reach. Walk forward; first
        # match wins.
        # Empirical (corpus 2026-05-02): GT reaches all have
        # displacement >= 1.7 radii; non-GT reaches p95 = ~3 but p50
        # is ~0.1 radii. Threshold 1.5 cleanly separates with margin.
        reaches_local: List[Tuple[int, int]] = []
        for rs, re in seg.reach_windows:
            ls = max(0, int(rs) - seg.seg_start)
            le = min(n - 1, int(re) - seg.seg_start)
            if le < ls:
                continue
            reaches_local.append((ls, le))
        reaches_local.sort()
        feats["n_reaches"] = len(reaches_local)
        if not reaches_local:
            return StageDecision(
                decision="continue",
                reason=f"no_reaches_in_segment",
                features=feats,
            )

        confident_pellet = pellet_lk >= self.pellet_lk_threshold
        px_conf = np.where(confident_pellet, pellet_x, np.nan)
        py_conf = np.where(confident_pellet, pellet_y, np.nan)

        # CAUSAL REACH (user 2026-05-03): the right anchor is the FIRST
        # frame where the pellet is observably off-pillar AND no paw
        # is past the y-line. If pellet is off-pillar between reaches,
        # the displacement has happened -- walk back to the most
        # recent reach before that frame.
        # This is more robust than walk-back-from-rest because:
        #   - rest position (median of late frames) requires the
        #     pellet to settle, which can take many frame durations
        #   - in cases with DLC visibility issues the pellet is only
        #     briefly observable; rest-based anchoring picks up later
        #     reaches where DLC happens to catch the pellet again
        #   - first off-pillar between-reaches observation locks down
        #     when the displacement was first SEEN, not where it ended
        # Require 3+ sustained frames to filter DLC noise blips.
        # Use a MORE PERMISSIVE lk threshold (0.7) for this anchor
        # detection -- between reaches the pellet is often at moderate
        # lk (DLC less confident due to recent paw activity / motion
        # blur / partial occlusion), but if dist > 1 radius and no paw
        # past y-line, it's clear off-pillar evidence.
        # Anchor: sustained off-pillar AND near the eventual rest
        # position. The "near rest" check filters DLC noise frames at
        # segment start that are off-pillar but at a different
        # location than the actual displacement endpoint.
        anchor_off_pillar = (
            (pellet_lk >= self.anchor_pellet_lk_threshold)
            & (pellet_dist_radii > self.anchor_off_pillar_radii)
            & (~paw_past_y)
            & (deviation_uv <= self.near_median_tolerance_uv)
        )
        sustained_off_run_start = -1
        run = 0
        for i in range(n):
            if anchor_off_pillar[i]:
                run += 1
                if run >= 3 and sustained_off_run_start < 0:
                    sustained_off_run_start = i - 2  # start of the 3-frame run
                    break
            else:
                run = 0
        first_near_median_idx = sustained_off_run_start
        if first_near_median_idx < 0:
            return StageDecision(
                decision="continue",
                reason="no_sustained_off_pillar_between_reaches",
                features=feats,
            )
        feats["first_off_pillar_between_reaches_idx"] = int(first_near_median_idx)

        causal_reach_idx = -1
        causal_bout_start = -1
        causal_bout_end = -1
        for ri in range(len(reaches_local) - 1, -1, -1):
            rs_local, re_local = reaches_local[ri]
            if re_local >= first_near_median_idx:
                continue
            pre_start = max(0, rs_local - self.pre_bout_window)
            pre_on_count = int(on_pillar[pre_start:rs_local].sum())
            if pre_on_count < self.min_pre_bout_on_pillar_frames:
                continue
            post_start = re_local + 1
            post_end_window = min(n, re_local + 1 + self.post_bout_window)
            post_off_count = int(off_pillar_confident[post_start:post_end_window].sum())
            if post_off_count < self.min_post_bout_off_pillar_frames:
                continue
            # Immediate-after check: pellet must be off-pillar in the
            # very next few frames after this reach ends. Filters
            # cases where post-off is satisfied only because a LATER
            # reach displaced the pellet within the post window.
            post_imm_end = min(n, re_local + 1 + self.post_bout_immediate_window)
            post_imm_off = int(off_pillar_confident[post_start:post_imm_end].sum())
            if post_imm_off < self.min_post_bout_immediate_off_frames:
                continue
            causal_reach_idx = ri
            causal_bout_start = rs_local
            causal_bout_end = re_local
            break

        if causal_reach_idx < 0:
            return StageDecision(
                decision="continue",
                reason=(
                    f"no_causal_reach_via_walkback_with_pre_post_check "
                    f"(walked back from first_near_median_idx="
                    f"{first_near_median_idx}; no reach satisfied "
                    f"pre-on AND post-off transition)"
                ),
                features=feats,
            )

        causal_bout_idx = causal_reach_idx
        # Compute displacement for every reach in segment.
        all_disps = []
        for ri, (rs_local, re_local) in enumerate(reaches_local):
            pre_s = max(0, rs_local - self.reach_displacement_window)
            post_e = min(n, re_local + 1 + self.reach_displacement_window)
            pre_x = px_conf[pre_s:rs_local]
            pre_y = py_conf[pre_s:rs_local]
            post_x = px_conf[re_local + 1:post_e]
            post_y = py_conf[re_local + 1:post_e]
            if np.all(np.isnan(pre_x)) or np.all(np.isnan(post_x)):
                all_disps.append(0.0)
                continue
            pre_mx = float(np.nanmedian(pre_x))
            pre_my = float(np.nanmedian(pre_y))
            post_mx = float(np.nanmedian(post_x))
            post_my = float(np.nanmedian(post_y))
            d_px = float(np.sqrt((post_mx - pre_mx) ** 2
                                 + (post_my - pre_my) ** 2))
            med_pillar_r = float(np.nanmedian(pillar_r[pre_s:post_e]))
            all_disps.append(d_px / max(med_pillar_r, 1e-6))
        chosen_disp_radii = all_disps[causal_reach_idx]
        max_disp = max(all_disps)
        max_disp_reach_idx = int(np.argmax(all_disps))
        feats["chosen_reach_displacement_radii"] = chosen_disp_radii
        feats["max_reach_displacement_radii"] = max_disp
        feats["max_disp_reach_idx"] = max_disp_reach_idx

        # Gate 1: chosen reach must have meaningful displacement.
        if chosen_disp_radii < self.reach_displacement_threshold_radii:
            return StageDecision(
                decision="continue",
                reason=(
                    f"chosen_reach_displacement_too_low "
                    f"({chosen_disp_radii:.2f} radii < "
                    f"{self.reach_displacement_threshold_radii})"
                ),
                features=feats,
            )

        # Gate 2: pellet must NEVER have been observed at the off-
        # pillar rest position BEFORE the chosen reach. If it was,
        # the displacement happened earlier (or DLC label-switch
        # predates the chosen reach) -- defer.
        pre_at_rest_mask = (
            off_pillar_confident[:causal_bout_start]
            & (deviation_uv[:causal_bout_start] <= self.near_median_tolerance_uv)
        )
        pre_causal_at_rest_count = int(pre_at_rest_mask.sum())
        feats["pre_causal_at_rest_count"] = pre_causal_at_rest_count
        if pre_causal_at_rest_count > 0:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_observed_at_rest_position_before_causal_reach "
                    f"({pre_causal_at_rest_count} frames -- displacement "
                    f"happened earlier than chosen reach -- defer)"
                ),
                features=feats,
            )

        # Gate 3: chosen reach must ALSO be the max-displacement reach.
        if causal_reach_idx != max_disp_reach_idx:
            return StageDecision(
                decision="continue",
                reason=(
                    f"walkback_and_max_displacement_disagree "
                    f"(walk-back chose reach {causal_reach_idx} with "
                    f"displacement {chosen_disp_radii:.2f}, but max "
                    f"displacement {max_disp:.2f} is at reach "
                    f"{max_disp_reach_idx} -- ambiguous reach selection, "
                    f"defer)"
                ),
                features=feats,
            )

        if causal_bout_idx < 0:
            return StageDecision(
                decision="continue",
                reason=(
                    f"no_reach_with_sufficient_pellet_displacement "
                    f"(max displacement {max_disp:.2f} radii < threshold "
                    f"{self.reach_displacement_threshold_radii} -- pellet "
                    f"didn't move enough at any reach)"
                ),
                features=feats,
            )

        # OKF emit: first frame at rest position after causal reach.
        first_near_median_idx = causal_bout_end + 1
        for i in range(causal_bout_end + 1, n):
            if (off_pillar_confident[i]
                    and deviation_uv[i] <= self.near_median_tolerance_uv):
                first_near_median_idx = i
                break

        bout_length = causal_bout_end - causal_bout_start + 1
        interaction_idx = int(causal_bout_start
                              + round(self.ifr_position_in_bout * bout_length))
        interaction_idx = max(causal_bout_start,
                              min(causal_bout_end, interaction_idx))
        okf_idx = min(first_near_median_idx + self.okf_settle_offset, n - 1)
        interaction_frame_video = int(seg.seg_start + interaction_idx)
        okf_video = int(seg.seg_start + okf_idx)
        feats.update({
            "causal_bout_idx": int(causal_bout_idx),
            "causal_bout_start_idx": int(causal_bout_start),
            "causal_bout_end_idx": int(causal_bout_end),
            "first_near_median_idx": int(first_near_median_idx),
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
                f"pellet_settled_off_pillar_in_sa_late "
                f"({n_near_median_late} near-median frames in last "
                f"{int(100*self.late_segment_fraction)}% of clean zone, "
                f"median dist {median_dist_radii:.2f} radii off-pillar; "
                f"causal bout {causal_bout_idx} ended at idx {causal_bout_end})"
            ),
            features=feats,
        )
