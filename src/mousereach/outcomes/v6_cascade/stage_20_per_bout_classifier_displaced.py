"""
Stage 20: Per-bout classifier for displaced_sa using per-segment
pellet-on-pillar calibration.

Question this stage answers:
    "Using the per-segment pellet-on-pillar reference (from pre-first-
    bout frames where pellet is loaded), is there a bout where
    pre-bout pellet was on-pillar (within 1.0 calibrated radius) and
    post-bout pellet is sustained off-pillar in SA? Pick that bout
    as causal -- relocations on-pillar before it don't count."

Whittling logic:
    - Many displaced residuals have multiple bouts where mouse may
      have relocated the pellet on-pillar before finally displacing
      it. The per-bout classifier walks through each bout and finds
      the one that pushed pellet from "on pillar zone" to "off pillar
      in SA".
    - Per-segment calibration uses the actual loaded pellet position
      (from pre-first-bout frames) instead of the calculated pillar
      center. This corrects for systematic per-video DLC offsets.

Defenses against false-commit:
    - Require valid calibration (>= 10 confident pre-first-bout
      pellet observations).
    - Strict 1.0 calibrated radius for "on pillar" (per user's
      wariness about widening this zone -- pellet cannot return to
      pillar once off, so a tight zone protects against masking).
    - Off-pillar threshold strict (>= 2.5 calibrated radii) to
      filter pillar-edge DLC noise.
    - Late-zone confirmation: pellet must be sustained off-pillar
      in late zone (final state confirmation).

Cascade emit on commit:
    - committed_class: "displaced_sa"
    - whens["interaction_frame"]: middle of identified causal bout
    - whens["outcome_known_frame"]: causal bout end + small offset
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .pellet_calibration import calibrate_pellet_on_pillar
from .stage_base import SegmentInput, Stage, StageDecision


PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5

PELLET_LK_HIGH = 0.95
PELLET_LK_OFF_PILLAR = 0.7
PAW_LK_THR = 0.5

ON_PILLAR_RADII = 1.0      # strict: pellet within 1.0 radius of
                           # per-segment calibrated pellet-on-pillar
                           # median (per user 2026-05-03: don't widen
                           # zone -- pellet cannot return to pillar
                           # once off)
OFF_PILLAR_RADII_MIN = 2.5  # strict: pellet at >= 2.5 radii is
                            # clearly off-pillar in SA (filters
                            # pillar-edge DLC noise)

# 2026-05-03: Stage 20 disabled. Per-bout classifier with per-video
# pellet calibration looked promising but hit ~19% trust ceiling.
# Root cause: DLC noise causes transient off-pillar observations
# between early bouts, BEFORE actual displacement, fooling the
# "first on->off transition" pick. Even with tightened sustained
# thresholds (50+ frames), the wrong bout often passes because
# noise can sustain across many frames in problem videos.
MIN_PRE_BOUT_ON_PILLAR = 100000  # disabled
MIN_POST_BOUT_OFF_PILLAR = 50
MIN_LATE_OFF_PILLAR = 50
LATE_FRACTION = 0.3
MIN_SUSTAINED_RUN = 3
OKF_SETTLE_OFFSET = 6
NEAR_MEDIAN_TOLERANCE_RADII = 1.5
MIN_NEAR_MEDIAN_POST_BOUT = 30


def _sustained_run_count(arr, min_run):
    total = 0
    run = 0
    for v in arr:
        if v:
            run += 1
        else:
            if run >= min_run:
                total += run
            run = 0
    if run >= min_run:
        total += run
    return total


class Stage20PerBoutClassifierDisplaced(Stage):
    name = "stage_20_per_bout_classifier_displaced"
    target_class = "displaced_sa"

    def __init__(
        self,
        pellet_lk_high: float = PELLET_LK_HIGH,
        pellet_lk_off_pillar: float = PELLET_LK_OFF_PILLAR,
        paw_lk_threshold: float = PAW_LK_THR,
        on_pillar_radii: float = ON_PILLAR_RADII,
        off_pillar_radii_min: float = OFF_PILLAR_RADII_MIN,
        min_pre_bout_on_pillar: int = MIN_PRE_BOUT_ON_PILLAR,
        min_post_bout_off_pillar: int = MIN_POST_BOUT_OFF_PILLAR,
        min_late_off_pillar: int = MIN_LATE_OFF_PILLAR,
        late_fraction: float = LATE_FRACTION,
        min_sustained_run: int = MIN_SUSTAINED_RUN,
        okf_settle_offset: int = OKF_SETTLE_OFFSET,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.pellet_lk_high = pellet_lk_high
        self.pellet_lk_off_pillar = pellet_lk_off_pillar
        self.paw_lk_threshold = paw_lk_threshold
        self.on_pillar_radii = on_pillar_radii
        self.off_pillar_radii_min = off_pillar_radii_min
        self.min_pre_bout_on_pillar = min_pre_bout_on_pillar
        self.min_post_bout_off_pillar = min_post_bout_off_pillar
        self.min_late_off_pillar = min_late_off_pillar
        self.late_fraction = late_fraction
        self.min_sustained_run = min_sustained_run
        self.okf_settle_offset = okf_settle_offset
        self.transition_zone_half = transition_zone_half

    def decide(self, seg: SegmentInput) -> StageDecision:
        clean_end = seg.seg_end - self.transition_zone_half
        if clean_end <= seg.seg_start:
            return StageDecision(decision="continue",
                                 reason="segment_too_short")

        # Calibration: per-segment pellet-on-pillar reference.
        cal = calibrate_pellet_on_pillar(
            seg.dlc_df, seg.seg_start, seg.seg_end,
            transition_zone_half=self.transition_zone_half,
            pellet_lk_threshold=self.pellet_lk_high,
            paw_lk_threshold=self.paw_lk_threshold,
        )
        if cal is None or not cal.is_reliable:
            return StageDecision(
                decision="continue",
                reason=(
                    "calibration_unreliable "
                    "(insufficient pre-first-bout pellet observations)"
                ),
            )

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

        pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
        pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
        pellet_y = sub["Pellet_y"].to_numpy(dtype=float)

        # Distance from per-segment calibrated pellet-on-pillar median.
        d_to_cal = np.sqrt((pellet_x - cal.on_pillar_x) ** 2
                           + (pellet_y - cal.on_pillar_y) ** 2)
        # Convert to pillar-radii units.
        d_to_cal_radii = d_to_cal / np.maximum(pillar_r, 1e-6)

        # Also compute distance from calc'd pillar center (for
        # off-pillar in-SA gate).
        d_to_calc = np.sqrt((pellet_x - pillar_cx) ** 2
                            + (pellet_y - pillar_cy) ** 2)
        d_to_calc_radii = d_to_calc / np.maximum(pillar_r, 1e-6)

        paw_past_y = np.zeros(n, dtype=bool)
        for bp in PAW_BODYPARTS:
            py = sub[f"{bp}_y"].to_numpy(dtype=float)
            pl = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
            paw_past_y |= (py <= slit_y_line) & (pl >= self.paw_lk_threshold)

        # SA quadrilateral.
        sabl_x = sub["SABL_x"].to_numpy(dtype=float)
        sabl_y = sub["SABL_y"].to_numpy(dtype=float)
        sabr_x = sub["SABR_x"].to_numpy(dtype=float)
        sabr_y = sub["SABR_y"].to_numpy(dtype=float)
        satl_x = sub["SATL_x"].to_numpy(dtype=float)
        satl_y = sub["SATL_y"].to_numpy(dtype=float)
        satr_x = sub["SATR_x"].to_numpy(dtype=float)
        satr_y = sub["SATR_y"].to_numpy(dtype=float)
        sa_top_y = (satl_y + satr_y) / 2.0
        sa_bot_y = (sabl_y + sabr_y) / 2.0
        sa_left_x = np.minimum(sabl_x, satl_x)
        sa_right_x = np.maximum(sabr_x, satr_x)
        in_sa = (
            (pellet_y >= sa_top_y) & (pellet_y <= sa_bot_y)
            & (pellet_x >= sa_left_x) & (pellet_x <= sa_right_x)
        )

        # Per-frame state predicates.
        # On pillar: confident pellet, distance <= on_pillar_radii from
        # calibrated reference, paw not past slit.
        on_pillar = (
            (pellet_lk >= self.pellet_lk_high)
            & (d_to_cal_radii <= self.on_pillar_radii)
            & (~paw_past_y)
        )
        # Off-pillar in SA: confident pellet, distance >= off_pillar_radii_min
        # from calc'd pillar center, in SA quadrilateral, paw not past slit.
        off_pillar_in_sa = (
            (pellet_lk >= self.pellet_lk_off_pillar)
            & (d_to_calc_radii >= self.off_pillar_radii_min)
            & in_sa
            & (~paw_past_y)
        )

        # GT reach windows.
        reach_windows_local: List[Tuple[int, int]] = []
        for rs, re in seg.reach_windows:
            ls = max(0, int(rs) - seg.seg_start)
            le = min(n - 1, int(re) - seg.seg_start)
            if le >= ls:
                reach_windows_local.append((ls, le))
        reach_windows_local.sort()
        feats = {
            "n_clean_zone_frames": int(n),
            "calibration_x": cal.on_pillar_x,
            "calibration_y": cal.on_pillar_y,
            "calibration_n_frames": cal.n_calibration_frames,
            "calibration_deviation_radii": cal.deviation_from_calc_pillar_radii,
            "n_gt_reaches": len(reach_windows_local),
        }
        if not reach_windows_local:
            return StageDecision(
                decision="continue",
                reason="no_gt_reaches",
                features=feats,
            )

        # Walk through bouts, find FIRST bout with pre-on-pillar AND
        # sustained post-off-pillar-in-SA. Relocations on-pillar
        # before this bout don't count -- their post-bout state is
        # also on-pillar.
        causal_idx = -1
        chosen_pre_on = 0
        chosen_post_off = 0
        for ri, (rs_local, re_local) in enumerate(reach_windows_local):
            # Pre-bout window: from previous bout end (or seg start) to
            # this bout start.
            prev_end = (reach_windows_local[ri - 1][1] + 1
                        if ri > 0 else 0)
            pre_on_count = int(on_pillar[prev_end:rs_local].sum())
            if pre_on_count < self.min_pre_bout_on_pillar:
                continue
            # Post-bout window: from this bout end to end of clean zone.
            post_off_count = _sustained_run_count(
                off_pillar_in_sa[re_local + 1:], self.min_sustained_run)
            if post_off_count < self.min_post_bout_off_pillar:
                continue
            causal_idx = ri
            chosen_pre_on = pre_on_count
            chosen_post_off = post_off_count
            break

        if causal_idx < 0:
            return StageDecision(
                decision="continue",
                reason=(
                    f"no_bout_with_on_to_off_pillar_transition "
                    f"(no bout had pre-on-pillar >= "
                    f"{self.min_pre_bout_on_pillar} AND post-off-pillar "
                    f">= {self.min_post_bout_off_pillar})"
                ),
                features=feats,
            )

        # Late-zone off-pillar confirmation.
        late_start_idx = int(n * (1 - self.late_fraction))
        late_off_count = int(off_pillar_in_sa[late_start_idx:].sum())
        feats["late_off_pillar"] = late_off_count
        if late_off_count < self.min_late_off_pillar:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_late_off_pillar "
                    f"({late_off_count} < {self.min_late_off_pillar})"
                ),
                features=feats,
            )

        bs, be = reach_windows_local[causal_idx]
        bout_length = be - bs + 1
        interaction_idx = bs + bout_length // 2
        okf_idx = min(be + self.okf_settle_offset, n - 1)
        interaction_frame_video = int(seg.seg_start + interaction_idx)
        okf_video = int(seg.seg_start + okf_idx)
        feats.update({
            "causal_idx": int(causal_idx),
            "bout_start_idx": int(bs),
            "bout_end_idx": int(be),
            "pre_on_pillar_count": int(chosen_pre_on),
            "post_off_pillar_count": int(chosen_post_off),
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
                f"per_bout_classifier_on_to_off_transition "
                f"(reach {causal_idx}; pre-on {chosen_pre_on}f, "
                f"post-off {chosen_post_off}f; late off "
                f"{late_off_count}f; cal deviation "
                f"{cal.deviation_from_calc_pillar_radii:.2f}r)"
            ),
            features=feats,
        )
