"""
Stage 19: Retrieved via Pillar-lk transition at FIRST GT reach
(multi-reach allowance).

Question this stage answers:
    "Does the FIRST GT reach in this segment cause Pillar_lk to
    transition from sustained-low (pellet on pillar) to sustained-
    high (pillar revealed) AND pellet stays gone (no off-pillar
    observations) for the rest of the segment?"

Whittling logic:
    - Stage 13 required single-GT-reach. Many retrieval cases have
      multiple reaches (mouse keeps trying paw-over-empty-pillar
      after retrieval).
    - For retrieved with multi-reach: GT picks FIRST reach (the one
      where pellet went from on-pillar to gone). Subsequent reaches
      are paw-over-empty-pillar (don't matter for IFR).
    - This stage commits when:
        - The FIRST GT reach has Pillar_lk transition (pre-low,
          post-high)
        - Post-FIRST-reach: pellet never confidently observed
          off-pillar (no displacement signature)
        - Subsequent reaches: also no displacement signature
          (otherwise displaced not retrieved)

Defenses against false-commit:
    - Pillar_lk pre-FIRST-reach must be sustained low.
    - Pillar_lk post-FIRST-reach must be sustained high.
    - Post-FIRST-reach off-pillar pellet observations <= 5 sustained.
    - Post-FIRST-reach late zone pellet observations <= 5 sustained.

Cascade emit on commit:
    - committed_class: "retrieved"
    - whens["interaction_frame"]: middle of FIRST GT reach
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

PILLAR_LK_PRE_LOW = 0.3
PILLAR_LK_POST_HIGH = 0.5
PAW_LK_THR = 0.5
PELLET_LK_HIGH = 0.7
ON_PILLAR_RADII = 1.0

# 2026-05-03: this stage's "first GT reach" causal pick is wrong for
# retrieved cases where mouse continues to reach over empty pillar
# after retrieval (GT picks the actual retrieval reach which may be
# late). Disabled. Keeping for documentation.
PRE_REACH_WINDOW = 30
MIN_PRE_LOW_FRAMES = 100000
MIN_POST_HIGH_FRAMES = 30
MAX_POST_OFF_PILLAR_PELLET_OBS = 5
MIN_SUSTAINED_RUN = 3


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


class Stage19RetrievedViaPillarLkFirstReach(Stage):
    name = "stage_19_retrieved_via_pillar_lk_first_reach"
    target_class = "retrieved"

    def __init__(
        self,
        pillar_lk_pre_low: float = PILLAR_LK_PRE_LOW,
        pillar_lk_post_high: float = PILLAR_LK_POST_HIGH,
        paw_lk_threshold: float = PAW_LK_THR,
        pellet_lk_high: float = PELLET_LK_HIGH,
        on_pillar_radii: float = ON_PILLAR_RADII,
        pre_reach_window: int = PRE_REACH_WINDOW,
        min_pre_low_frames: int = MIN_PRE_LOW_FRAMES,
        min_post_high_frames: int = MIN_POST_HIGH_FRAMES,
        max_post_off_pillar_pellet_obs: int = MAX_POST_OFF_PILLAR_PELLET_OBS,
        min_sustained_run: int = MIN_SUSTAINED_RUN,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.pillar_lk_pre_low = pillar_lk_pre_low
        self.pillar_lk_post_high = pillar_lk_post_high
        self.paw_lk_threshold = paw_lk_threshold
        self.pellet_lk_high = pellet_lk_high
        self.on_pillar_radii = on_pillar_radii
        self.pre_reach_window = pre_reach_window
        self.min_pre_low_frames = min_pre_low_frames
        self.min_post_high_frames = min_post_high_frames
        self.max_post_off_pillar_pellet_obs = max_post_off_pillar_pellet_obs
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

        pillar_lk_raw = sub_raw["Pillar_likelihood"].to_numpy(dtype=float)
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

        # GT reaches (need at least 1).
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
        if not reach_windows_local:
            return StageDecision(
                decision="continue",
                reason="no_gt_reaches",
                features=feats,
            )

        first_bs, first_be = reach_windows_local[0]

        # Pillar lk transition at first reach.
        pillar_low = (pillar_lk_raw <= self.pillar_lk_pre_low) & (~paw_past_y)
        pillar_high = (pillar_lk_raw >= self.pillar_lk_post_high) & (~paw_past_y)

        pre_start = max(0, first_bs - self.pre_reach_window)
        pre_low_count = int(pillar_low[pre_start:first_bs].sum())
        feats["pre_low_count"] = pre_low_count
        if pre_low_count < self.min_pre_low_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_pre_first_reach_pillar_low "
                    f"({pre_low_count} < {self.min_pre_low_frames})"
                ),
                features=feats,
            )

        post_high_count = _sustained_run_count(
            pillar_high[first_be + 1:], self.min_sustained_run)
        feats["post_high_count"] = post_high_count
        if post_high_count < self.min_post_high_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_post_first_reach_pillar_high "
                    f"({post_high_count} < {self.min_post_high_frames})"
                ),
                features=feats,
            )

        # Post-first-reach: pellet must be essentially gone (no
        # sustained off-pillar observations).
        confident_off_pillar = (
            (pellet_lk >= self.pellet_lk_high)
            & (~paw_past_y)
            & (dist_radii > self.on_pillar_radii)
        )
        post_off_pillar_count = _sustained_run_count(
            confident_off_pillar[first_be + 1:], self.min_sustained_run)
        feats["post_off_pillar_pellet"] = post_off_pillar_count
        if post_off_pillar_count > self.max_post_off_pillar_pellet_obs:
            return StageDecision(
                decision="continue",
                reason=(
                    f"sustained_post_off_pillar_pellet "
                    f"({post_off_pillar_count} > "
                    f"{self.max_post_off_pillar_pellet_obs}); pellet still "
                    f"in apparatus -- not retrieved"
                ),
                features=feats,
            )

        bout_length = first_be - first_bs + 1
        interaction_idx = first_bs + bout_length // 2
        okf_idx = n - 1
        interaction_frame_video = int(seg.seg_start + interaction_idx)
        okf_video = int(seg.seg_start + okf_idx)
        feats.update({
            "bout_start_idx": int(first_bs),
            "bout_end_idx": int(first_be),
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
                f"retrieved_via_pillar_lk_first_reach "
                f"(pre-low {pre_low_count}f, post-high {post_high_count}f, "
                f"post-off-pillar pellet {post_off_pillar_count}f)"
            ),
            features=feats,
        )
