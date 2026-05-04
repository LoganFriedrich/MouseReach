"""
Stage 12: Retrieved with pellet observations above the slit-y-line
(mouse face / mouth area).

Question this stage answers:
    "Stage 9 deferred this segment because off-pillar pellet
    observations exist post-reach (anti-displaced gate). But are
    those observations CLUSTERED ABOVE the slit-y-line (in the mouse
    face area where in-mouth tracks live), rather than BELOW (where
    a displaced pellet would sit on the SA tray)?"

Whittling logic:
    - Stage 9's anti-displaced gate fires when sustained off-pillar
      pellet observations exist post-first-reach (any direction).
    - For real retrieved cases, post-reach pellet observations come
      from DLC briefly tracking the pellet in the mouse's mouth/paws.
      Those observations are at y < slit_y_line (above the slit).
    - For real displaced cases, post-reach pellet observations are
      at y > slit_y_line (in the SA tray below the slit).
    - This stage commits retrieved when ALL post-reach off-pillar
      pellet observations are above the slit-y-line.

Defenses against false-commit:
    - At least one post-reach pellet observation must exist above
      the slit (positive evidence of in-mouth tracking).
    - ZERO sustained off-pillar pellet observations below the slit.
    - Single GT reach OR exactly one paw bout to avoid bout-pick
      ambiguity.
    - Late-zone pellet observations also must be above-slit only.

Cascade emit on commit:
    - committed_class: "retrieved"
    - whens["interaction_frame"]: middle of the GT reach (or paw bout)
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

# Slit-y discriminator: pellet position relative to pillar_cy + pillar_r
# (slit y-line). Above (smaller y) = mouse face. Below (larger y) = SA.
# We require the y SHIFT magnitude beyond the slit to ensure clean
# discrimination -- pellet right at the slit is ambiguous.
SLIT_BUFFER_PX = 5.0  # pellet must be MORE THAN buffer above/below slit

MAX_BELOW_SLIT_FRAMES = 5         # sustained sub-slit pellet observations
MIN_ABOVE_SLIT_FRAMES = 3         # need positive in-mouth evidence
MIN_SUSTAINED_PELLET_RUN = 3


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


class Stage12RetrievedPelletAboveSlit(Stage):
    name = "stage_12_retrieved_pellet_above_slit_y_line"
    target_class = "retrieved"

    def __init__(
        self,
        pellet_lk_high: float = PELLET_LK_HIGH,
        paw_lk_threshold: float = PAW_LK_THR,
        on_pillar_radii: float = ON_PILLAR_RADII,
        slit_buffer_px: float = SLIT_BUFFER_PX,
        max_below_slit_frames: int = MAX_BELOW_SLIT_FRAMES,
        min_above_slit_frames: int = MIN_ABOVE_SLIT_FRAMES,
        min_sustained_pellet_run: int = MIN_SUSTAINED_PELLET_RUN,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.pellet_lk_high = pellet_lk_high
        self.paw_lk_threshold = paw_lk_threshold
        self.on_pillar_radii = on_pillar_radii
        self.slit_buffer_px = slit_buffer_px
        self.max_below_slit_frames = max_below_slit_frames
        self.min_above_slit_frames = min_above_slit_frames
        self.min_sustained_pellet_run = min_sustained_pellet_run
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

        # Use GT reach windows when available.
        reach_windows_local: List[Tuple[int, int]] = []
        for rs, re in seg.reach_windows:
            ls = max(0, int(rs) - seg.seg_start)
            le = min(n - 1, int(re) - seg.seg_start)
            if le >= ls:
                reach_windows_local.append((ls, le))
        reach_windows_local.sort()
        bouts = (reach_windows_local
                 if reach_windows_local
                 else _find_paw_past_y_line_bouts(paw_past_y))
        feats = {
            "n_clean_zone_frames": int(n),
            "n_bouts": len(bouts),
            "bouts_source": ("reach_windows"
                             if reach_windows_local
                             else "paw_past_y"),
        }
        if not bouts:
            return StageDecision(
                decision="continue",
                reason="no_bouts",
                features=feats,
            )

        # First bout's start (we partition pre/post relative to first
        # reach -- matches Stage 9's anti-displaced semantics).
        first_bout_start = bouts[0][0]
        first_bout_end = bouts[0][1]

        # Pellet observations partitioned by slit-y-line position.
        # `above_slit` = y < slit_y_line - buffer (in mouse face area).
        # `below_slit` = y > slit_y_line + buffer (in SA tray area).
        confident = (pellet_lk >= self.pellet_lk_high) & (~paw_past_y)
        off_pillar = confident & (dist_radii > self.on_pillar_radii)
        above_slit_raw = off_pillar & (pellet_y < slit_y_line - self.slit_buffer_px)
        below_slit_raw = off_pillar & (pellet_y > slit_y_line + self.slit_buffer_px)

        # Post-first-reach sustained counts.
        post_above_slit = _sustained_run_count(
            above_slit_raw[first_bout_end + 1:], self.min_sustained_pellet_run)
        post_below_slit = _sustained_run_count(
            below_slit_raw[first_bout_end + 1:], self.min_sustained_pellet_run)
        feats.update({
            "post_above_slit": int(post_above_slit),
            "post_below_slit": int(post_below_slit),
        })

        if post_below_slit > self.max_below_slit_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"sustained_pellet_below_slit_post_first_reach "
                    f"({post_below_slit} > {self.max_below_slit_frames}); "
                    f"pellet appears in SA tray area -- looks displaced"
                ),
                features=feats,
            )

        if post_above_slit < self.min_above_slit_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_above_slit_pellet_observations "
                    f"({post_above_slit} < {self.min_above_slit_frames}); "
                    f"no positive evidence of in-mouth track"
                ),
                features=feats,
            )

        # Restrict to single-GT-reach OR single-paw-bout to avoid
        # bout-pick ambiguity.
        if len(bouts) != 1:
            return StageDecision(
                decision="continue",
                reason=(
                    f"multiple_bouts_in_segment "
                    f"({len(bouts)} bouts; can't pick GT's last paw-over-"
                    f"pellet without disambiguation -- defer)"
                ),
                features=feats,
            )

        # Additional defense: even if there's only 1 GT reach, total
        # paw-past-y-line activity must also be limited. Hyperactive
        # mouse segments (many paw bouts beyond the GT reach) suggest
        # contested-displaced rather than clean retrieval.
        all_paw_bouts = _find_paw_past_y_line_bouts(paw_past_y)
        feats["n_all_paw_bouts"] = len(all_paw_bouts)
        MAX_PAW_BOUTS_FOR_CLEAN_RETRIEVAL = 15
        if len(all_paw_bouts) > MAX_PAW_BOUTS_FOR_CLEAN_RETRIEVAL:
            return StageDecision(
                decision="continue",
                reason=(
                    f"hyperactive_paw_activity "
                    f"({len(all_paw_bouts)} paw bouts > "
                    f"{MAX_PAW_BOUTS_FOR_CLEAN_RETRIEVAL}; mouse likely "
                    f"contested -- not a clean retrieval signature)"
                ),
                features=feats,
            )

        causal_start, causal_end = bouts[0]
        bout_length = causal_end - causal_start + 1
        interaction_idx = causal_start + bout_length // 2
        okf_idx = n - 1
        interaction_frame_video = int(seg.seg_start + interaction_idx)
        okf_video = int(seg.seg_start + okf_idx)
        feats.update({
            "causal_start_idx": int(causal_start),
            "causal_end_idx": int(causal_end),
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
                f"retrieved_pellet_above_slit_y_line "
                f"(post-bout above-slit {post_above_slit}f, below-slit "
                f"{post_below_slit}f)"
            ),
            features=feats,
        )
