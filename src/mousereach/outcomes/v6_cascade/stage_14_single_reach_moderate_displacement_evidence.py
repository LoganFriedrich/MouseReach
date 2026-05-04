"""
Stage 14: Single-reach displaced with moderate (10-99 frame) evidence.

Question this stage answers:
    "Is this a single-GT-reach segment where the pellet is observed
    at confident off-pillar in-SA position for 10-99 sustained
    frames -- enough to establish displacement but below Stage 7's
    100-frame threshold for sustained near-median rest?"

Whittling logic:
    - Stage 7 requires 100+ frames at sustained near-median position.
    - Stage 10 commits at 5+ frames sustained displacement-evidence
      with tight defenses (only 1 paw bout pre-evidence, no bouts
      after).
    - Stage 14 fills the middle: single-reach segments with 10+
      sustained off-pillar in-SA observations. Single-reach removes
      bout-pick ambiguity. Lower threshold than Stage 7 catches
      cases with weaker DLC tracking.

Defenses against false-commit:
    - Single GT reach only.
    - 10+ sustained off-pillar in-SA pellet observations post-reach.
    - At dist > 2.0 radii (filter pillar-edge noise).
    - Late-zone has at least some off-pillar pellet observation
      (positive evidence pellet stayed in SA -- excludes brief
      retrievals).

Cascade emit on commit:
    - committed_class: "displaced_sa"
    - whens["interaction_frame"]: middle of GT reach
    - whens["outcome_known_frame"]: first sustained off-pillar
      observation post-reach + small offset
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import SegmentInput, Stage, StageDecision


PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5

PELLET_LK_OFF_PILLAR = 0.7
PAW_LK_THR = 0.5
ON_PILLAR_RADII = 1.0
OFF_PILLAR_RADII_MIN = 2.0  # filter pillar-edge noise

MIN_SUSTAINED_DISPLACEMENT_FRAMES = 100000  # disabled
MIN_SUSTAINED_RUN = 3
MIN_LATE_OFF_PILLAR_FRAMES = 30
LATE_FRACTION = 0.3

OKF_SETTLE_OFFSET = 6


class Stage14SingleReachModerateDisplacementEvidence(Stage):
    name = "stage_14_single_reach_moderate_displacement_evidence"
    target_class = "displaced_sa"

    def __init__(
        self,
        pellet_lk_off_pillar: float = PELLET_LK_OFF_PILLAR,
        paw_lk_threshold: float = PAW_LK_THR,
        on_pillar_radii: float = ON_PILLAR_RADII,
        off_pillar_radii_min: float = OFF_PILLAR_RADII_MIN,
        min_sustained_displacement_frames: int = MIN_SUSTAINED_DISPLACEMENT_FRAMES,
        min_sustained_run: int = MIN_SUSTAINED_RUN,
        min_late_off_pillar_frames: int = MIN_LATE_OFF_PILLAR_FRAMES,
        late_fraction: float = LATE_FRACTION,
        okf_settle_offset: int = OKF_SETTLE_OFFSET,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.pellet_lk_off_pillar = pellet_lk_off_pillar
        self.paw_lk_threshold = paw_lk_threshold
        self.on_pillar_radii = on_pillar_radii
        self.off_pillar_radii_min = off_pillar_radii_min
        self.min_sustained_displacement_frames = min_sustained_displacement_frames
        self.min_sustained_run = min_sustained_run
        self.min_late_off_pillar_frames = min_late_off_pillar_frames
        self.late_fraction = late_fraction
        self.okf_settle_offset = okf_settle_offset
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

        # Single GT reach only.
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
        if len(reach_windows_local) != 1:
            return StageDecision(
                decision="continue",
                reason=(
                    f"not_single_gt_reach "
                    f"({len(reach_windows_local)} GT reaches; defer)"
                ),
                features=feats,
            )

        # Hyperactive paw activity defense.
        bouts_paw = []
        rs = -1
        for i in range(n):
            if paw_past_y[i]:
                if rs < 0:
                    rs = i
            else:
                if rs >= 0:
                    bouts_paw.append((rs, i - 1))
                    rs = -1
        if rs >= 0:
            bouts_paw.append((rs, n - 1))
        feats["n_paw_bouts"] = len(bouts_paw)
        MAX_PAW_BOUTS = 10
        if len(bouts_paw) > MAX_PAW_BOUTS:
            return StageDecision(
                decision="continue",
                reason=(
                    f"hyperactive_paw_activity "
                    f"({len(bouts_paw)} > {MAX_PAW_BOUTS}; defer)"
                ),
                features=feats,
            )

        bs, be = reach_windows_local[0]

        # Off-pillar in SA evidence.
        off_pillar_in_sa = (
            (pellet_lk >= self.pellet_lk_off_pillar)
            & (~paw_past_y)
            & (dist_radii > self.off_pillar_radii_min)
            & in_sa
        )

        # Sustained mask helper.
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
        sustained_off_in_sa = sustained_mask(off_pillar_in_sa, self.min_sustained_run)

        post_reach_displacement_count = int(sustained_off_in_sa[be + 1:].sum())
        feats["post_reach_displacement_count"] = post_reach_displacement_count
        if post_reach_displacement_count < self.min_sustained_displacement_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_post_reach_displacement_evidence "
                    f"({post_reach_displacement_count} < "
                    f"{self.min_sustained_displacement_frames})"
                ),
                features=feats,
            )

        # Late-zone positive evidence.
        late_start_idx = int(n * (1 - self.late_fraction))
        late_off_in_sa = int(sustained_off_in_sa[late_start_idx:].sum())
        feats["late_off_in_sa"] = late_off_in_sa
        if late_off_in_sa < self.min_late_off_pillar_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_late_off_pillar_evidence "
                    f"({late_off_in_sa} < "
                    f"{self.min_late_off_pillar_frames}; pellet not in SA "
                    f"late -- could be retrieved)"
                ),
                features=feats,
            )

        # Find first sustained off-pillar in-SA frame post-reach (for OKF).
        first_off_idx = -1
        run = 0
        for i in range(be + 1, n):
            if off_pillar_in_sa[i]:
                run += 1
                if run >= self.min_sustained_run:
                    first_off_idx = i - self.min_sustained_run + 1
                    break
            else:
                run = 0
        if first_off_idx < 0:
            first_off_idx = be + 1

        bout_length = be - bs + 1
        interaction_idx = bs + bout_length // 2
        okf_idx = min(first_off_idx + self.okf_settle_offset, n - 1)
        interaction_frame_video = int(seg.seg_start + interaction_idx)
        okf_video = int(seg.seg_start + okf_idx)
        feats.update({
            "bout_start_idx": int(bs),
            "bout_end_idx": int(be),
            "first_off_pillar_idx": int(first_off_idx),
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
                f"single_reach_moderate_displacement_evidence "
                f"(post-reach displacement {post_reach_displacement_count}f, "
                f"late off-in-SA {late_off_in_sa}f)"
            ),
            features=feats,
        )
