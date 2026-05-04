"""
Stage 15: Multi-reach retrieved via above-slit-vs-in-SA visibility
partition.

Question this stage answers:
    "In a multi-GT-reach segment, is there a reach AFTER WHICH all
    sustained pellet observations are above the slit-y-line (in
    mouse face / in mouth) and zero/few are in the SA (below slit)?"

Whittling logic:
    - Stage 12 only handles single-bout segments. This stage extends
      the above-slit-vs-in-SA discriminator to multi-reach segments.
    - For each reach, compute post-reach pellet observations split
      by location. The FIRST reach where:
        - post-reach in-SA pellet count <= small budget
        - post-reach above-slit pellet count > 0 (positive in-mouth)
      is the retrieval reach.
    - Subsequent reaches are paw-over-empty-pillar (don't matter for
      classification).

Defenses against false-commit:
    - Single GT reach excluded (Stage 12's domain).
    - Reach must have at least 1 sustained run of above-slit pellet
      observation post-reach (positive evidence of in-mouth track).
    - Sustained in-SA pellet observations post-reach must be very
      few (else it's displaced not retrieved).
    - The chosen reach must be the FIRST that satisfies these
      criteria (matches GT's "last paw-over-pellet" semantic for
      retrieval where subsequent reaches are over empty pillar).

Cascade emit on commit:
    - committed_class: "retrieved"
    - whens["interaction_frame"]: middle of the chosen GT reach
    - whens["outcome_known_frame"]: clean zone end
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import SegmentInput, Stage, StageDecision


PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5

PELLET_LK_HIGH = 0.7
PAW_LK_THR = 0.5
ON_PILLAR_RADII = 1.0
SLIT_BUFFER_PX = 5.0

# 2026-05-03: this stage's above-slit-vs-in-SA partition has too many
# false positives in multi-reach segments to converge to 100% trust.
# Setting threshold prohibitively high to effectively disable this
# stage; keeping the file for documentation of an attempted approach
# that didn't pan out.
MIN_POST_ABOVE_SLIT = 10000
MAX_POST_IN_SA = 0
MIN_SUSTAINED_RUN = 3


class Stage15MultiReachRetrievedViaAboveSlitSplit(Stage):
    name = "stage_15_multi_reach_retrieved_via_above_slit_split"
    target_class = "retrieved"

    def __init__(
        self,
        pellet_lk_high: float = PELLET_LK_HIGH,
        paw_lk_threshold: float = PAW_LK_THR,
        on_pillar_radii: float = ON_PILLAR_RADII,
        slit_buffer_px: float = SLIT_BUFFER_PX,
        min_post_above_slit: int = MIN_POST_ABOVE_SLIT,
        max_post_in_sa: int = MAX_POST_IN_SA,
        min_sustained_run: int = MIN_SUSTAINED_RUN,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.pellet_lk_high = pellet_lk_high
        self.paw_lk_threshold = paw_lk_threshold
        self.on_pillar_radii = on_pillar_radii
        self.slit_buffer_px = slit_buffer_px
        self.min_post_above_slit = min_post_above_slit
        self.max_post_in_sa = max_post_in_sa
        self.min_sustained_run = min_sustained_run
        self.transition_zone_half = transition_zone_half

    def decide(self, seg: SegmentInput) -> StageDecision:
        clean_end = seg.seg_end - self.transition_zone_half
        if clean_end <= seg.seg_start:
            return StageDecision(decision="continue",
                                 reason="segment_too_short")

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
        dist_radii = (np.sqrt((pellet_x - pillar_cx) ** 2
                              + (pellet_y - pillar_cy) ** 2)
                      / np.maximum(pillar_r, 1e-6))

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

        confident = (pellet_lk >= self.pellet_lk_high) & (~paw_past_y)
        off_pillar = confident & (dist_radii > self.on_pillar_radii)
        above_slit_raw = off_pillar & (pellet_y < slit_y_line - self.slit_buffer_px)
        in_sa_raw = off_pillar & in_sa & (pellet_y > slit_y_line + self.slit_buffer_px)

        def sustained_mask(arr, min_run):
            out = np.zeros_like(arr, dtype=bool)
            run = 0
            for i in range(len(arr)):
                if arr[i]:
                    run += 1
                else:
                    if run >= min_run:
                        out[i - run:i] = True
                    run = 0
            if run >= min_run:
                out[len(arr) - run:] = True
            return out
        above_slit = sustained_mask(above_slit_raw, self.min_sustained_run)
        in_sa_pellet = sustained_mask(in_sa_raw, self.min_sustained_run)

        # Multi-GT-reach only.
        reach_windows_local: List[Tuple[int, int]] = []
        for rs, re in seg.reach_windows:
            ls = max(0, int(rs) - seg.seg_start)
            le = min(n - 1, int(re) - seg.seg_start)
            if le >= ls:
                reach_windows_local.append((ls, le))
        reach_windows_local.sort()
        feats = {
            "n_clean_zone_frames": int(n),
            "n_gt_reaches": len(reach_windows_local),
        }
        if len(reach_windows_local) < 2:
            return StageDecision(
                decision="continue",
                reason=(
                    f"not_multi_gt_reach "
                    f"({len(reach_windows_local)} reaches; this stage only "
                    f"handles multi-reach segments)"
                ),
                features=feats,
            )

        # For each reach, compute post-reach above-slit and in-SA
        # sustained counts. Pick FIRST reach satisfying criteria.
        causal_idx = -1
        chosen_above = 0
        chosen_in_sa = 0
        for ri, (rs, re) in enumerate(reach_windows_local):
            post_above = int(above_slit[re + 1:].sum())
            post_in_sa = int(in_sa_pellet[re + 1:].sum())
            if (post_above >= self.min_post_above_slit
                    and post_in_sa <= self.max_post_in_sa):
                causal_idx = ri
                chosen_above = post_above
                chosen_in_sa = post_in_sa
                break
        if causal_idx < 0:
            return StageDecision(
                decision="continue",
                reason=(
                    f"no_reach_with_above_slit_in_mouth_signature "
                    f"(no reach has post-reach above-slit "
                    f">= {self.min_post_above_slit} AND in-SA "
                    f"<= {self.max_post_in_sa})"
                ),
                features=feats,
            )

        # Defense: subsequent reaches must NOT have additional in-SA
        # pellet (would suggest contested displaced).
        for ri in range(causal_idx + 1, len(reach_windows_local)):
            rs, re = reach_windows_local[ri]
            post_in_sa = int(in_sa_pellet[re + 1:].sum())
            if post_in_sa > self.max_post_in_sa:
                feats["disqualifying_post_reach_idx"] = ri
                feats["disqualifying_post_in_sa"] = post_in_sa
                return StageDecision(
                    decision="continue",
                    reason=(
                        f"subsequent_reach_has_in_sa_pellet "
                        f"(reach {ri} post in-SA {post_in_sa} > "
                        f"{self.max_post_in_sa}; mouse may be retrieving "
                        f"a displaced pellet -- defer)"
                    ),
                    features=feats,
                )

        bs, be = reach_windows_local[causal_idx]
        bout_length = be - bs + 1
        interaction_idx = bs + bout_length // 2
        okf_idx = n - 1
        interaction_frame_video = int(seg.seg_start + interaction_idx)
        okf_video = int(seg.seg_start + okf_idx)
        feats.update({
            "causal_idx": int(causal_idx),
            "bout_start_idx": int(bs),
            "bout_end_idx": int(be),
            "post_above_slit": int(chosen_above),
            "post_in_sa": int(chosen_in_sa),
            "interaction_frame_video": interaction_frame_video,
            "okf_video": okf_video,
        })
        return StageDecision(
            decision="commit",
            committed_class="retrieved",
            whens={
                "outcome_known_frame": okf_video,
                "interaction_frame": interaction_frame_video,
            },
            reason=(
                f"multi_reach_retrieved_via_above_slit "
                f"(reach {causal_idx}; post above-slit {chosen_above}f, "
                f"post in-SA {chosen_in_sa}f)"
            ),
            features=feats,
        )
