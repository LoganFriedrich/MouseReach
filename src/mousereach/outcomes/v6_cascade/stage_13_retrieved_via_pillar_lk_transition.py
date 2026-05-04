"""
Stage 13: Retrieved via Pillar-bodypart visibility transition (pre-low,
post-high) AND pellet completely gone post-reach.

Question this stage answers:
    "Is there exactly ONE reach in clean zone where Pillar_lk
    transitions from sustained-low (pellet on pillar) to sustained-
    high (pillar revealed), AND the pellet is essentially invisible
    after that reach?"

Whittling logic:
    - Stage 10 used Pillar lk transition for displaced_sa (with
      post-reach off-pillar pellet evidence required). This is the
      complementary logic for retrieved: Pillar lk transition AND
      pellet completely gone (no post-reach off-pillar observation
      anywhere).
    - Single-reach restriction avoids bout-pick ambiguity.

Defenses against false-commit:
    - Single GT reach only.
    - Pillar_lk pre-reach sustained low (pellet was occluding it).
    - Pillar_lk post-reach sustained high (pellet has left).
    - Post-reach off-pillar pellet observations <= 5 (sustained
      filter): pellet is gone, not just displaced briefly.
    - Late-zone pellet observability essentially zero.

Cascade emit on commit:
    - committed_class: "retrieved"
    - whens["interaction_frame"]: middle of the GT reach
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

PRE_REACH_WINDOW = 30
MIN_PRE_LOW_FRAMES = 5
MIN_POST_HIGH_FRAMES = 30
MAX_POST_OFF_PILLAR_PELLET_OBS = 5
MAX_LATE_PELLET_VISIBILITY = 10
MIN_SUSTAINED_RUN = 3


def _find_paw_past_y_line_bouts(paw_past_y):
    n = len(paw_past_y)
    bouts = []
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


class Stage13RetrievedViaPillarLkTransition(Stage):
    name = "stage_13_retrieved_via_pillar_lk_transition"
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
        max_late_pellet_visibility: int = MAX_LATE_PELLET_VISIBILITY,
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
        self.max_late_pellet_visibility = max_late_pellet_visibility
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

        # Use GT reach windows; require exactly one.
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
                    f"({len(reach_windows_local)} reaches; multi-reach "
                    f"causes wrong-bout commits -- defer)"
                ),
                features=feats,
            )
        bs, be = reach_windows_local[0]

        # Pillar_lk transition.
        pillar_low = (pillar_lk_raw <= self.pillar_lk_pre_low) & (~paw_past_y)
        pillar_high = (pillar_lk_raw >= self.pillar_lk_post_high) & (~paw_past_y)

        pre_start = max(0, bs - self.pre_reach_window)
        pre_low_count = int(pillar_low[pre_start:bs].sum())
        feats["pre_low_count"] = pre_low_count
        if pre_low_count < self.min_pre_low_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_pre_reach_pillar_low "
                    f"({pre_low_count} < {self.min_pre_low_frames})"
                ),
                features=feats,
            )

        post_high_count = _sustained_run_count(
            pillar_high[be + 1:], self.min_sustained_run)
        feats["post_high_count"] = post_high_count
        if post_high_count < self.min_post_high_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_post_reach_pillar_high "
                    f"({post_high_count} < {self.min_post_high_frames})"
                ),
                features=feats,
            )

        # Post-reach pellet must be essentially gone (off-pillar
        # observations <= budget AND late visibility low).
        confident_off_pillar = (
            (pellet_lk >= self.pellet_lk_high)
            & (~paw_past_y)
            & (dist_radii > self.on_pillar_radii)
        )
        post_off_pillar_count = _sustained_run_count(
            confident_off_pillar[be + 1:], self.min_sustained_run)
        feats["post_off_pillar_pellet"] = post_off_pillar_count
        if post_off_pillar_count > self.max_post_off_pillar_pellet_obs:
            return StageDecision(
                decision="continue",
                reason=(
                    f"sustained_post_off_pillar_pellet "
                    f"({post_off_pillar_count} > "
                    f"{self.max_post_off_pillar_pellet_obs}); pellet still "
                    f"observable -- not retrieved"
                ),
                features=feats,
            )

        # Late zone visibility check (final defense -- pellet shouldn't
        # be visible at any position late).
        late_start_idx = int(n * 0.5)
        any_pellet_late = (pellet_lk >= 0.7) & (~paw_past_y)
        any_pellet_late[:late_start_idx] = False
        late_visibility = _sustained_run_count(any_pellet_late, self.min_sustained_run)
        feats["late_visibility"] = late_visibility
        if late_visibility > self.max_late_pellet_visibility:
            return StageDecision(
                decision="continue",
                reason=(
                    f"late_pellet_visibility_too_high "
                    f"({late_visibility} > {self.max_late_pellet_visibility})"
                ),
                features=feats,
            )

        bout_length = be - bs + 1
        interaction_idx = bs + bout_length // 2
        okf_idx = n - 1
        interaction_frame_video = int(seg.seg_start + interaction_idx)
        okf_video = int(seg.seg_start + okf_idx)
        feats.update({
            "bout_start_idx": int(bs),
            "bout_end_idx": int(be),
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
                f"retrieved_via_pillar_lk_transition "
                f"(pre-low pillar {pre_low_count}f, post-high pillar "
                f"{post_high_count}f, post-off-pillar pellet "
                f"{post_off_pillar_count}f, late visibility "
                f"{late_visibility}f)"
            ),
            features=feats,
        )
