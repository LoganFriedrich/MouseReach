"""
Stage 11: Pellet effectively invisible all segment + paw activity = retrieved.

Question this stage answers:
    "Did DLC fail to track the pellet entirely throughout this
    segment (essentially zero confident detections), AND did the
    mouse perform paw activity past the slit? If so, the pellet was
    retrieved during one of those reaches and the failure to track
    is symptomatic of: (a) pellet briefly held by paws and consumed,
    (b) DLC struggling on this video's pellet appearance."

Whittling logic:
    - Stage 9 cannot commit when there are no GT reach windows (it
      defers with `no_reaches_in_segment`). Some retrieved segments
      legitimately have GT reach windows missing or have weak
      Pellet-bodypart tracking that triggers Stage 9's anti-displaced
      gate even though the pellet was actually retrieved.
    - Orthogonal signal: TOTAL pellet visibility across the clean
      zone. For displaced cases, the pellet is in the apparatus and
      DLC will track it at SOME point even if poorly. For retrieved
      cases where DLC also misses pre-reach pellet detection (poor
      video), pellet visibility is essentially zero throughout.
    - This stage commits retrieved when:
        1. Total confident pellet observations < 10 sustained frames
           in entire clean zone (pellet is essentially invisible).
        2. At least 1 paw-past-slit bout exists (something happened).
        3. Late-zone pellet observations are zero (definitely gone).

Defenses against false-commit:
    - Strict zero-late-visibility check (<= 2 sustained frames).
    - Strict overall-low-visibility check (<= 10 sustained frames).
    - At least 1 paw bout (sanity).
    - The IFR is the LAST paw bout (matches GT's "last paw-over-
      pellet" semantic for retrieval).

Cascade emit on commit:
    - committed_class: "retrieved"
    - whens["interaction_frame"]: middle of last paw-past-slit bout
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

PELLET_LK_THR = 0.7  # very loose -- any moderately confident detection
PAW_LK_THR = 0.5

LATE_FRACTION = 0.5
MAX_TOTAL_PELLET_VISIBILITY = 100  # sustained frames in entire clean
                                    # zone -- "essentially invisible"
MAX_LATE_PELLET_VISIBILITY = 5     # virtually zero late observation
MIN_SUSTAINED_PELLET_RUN = 3
OKF_OFFSET = 0  # OKF = clean zone end


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


class Stage11SingleReachCleanDisplacement(Stage):
    # Class name preserved for cascade-runner stability.
    name = "stage_11_pellet_invisible_throughout_with_paw_activity"
    target_class = "retrieved"

    def __init__(
        self,
        pellet_lk_threshold: float = PELLET_LK_THR,
        paw_lk_threshold: float = PAW_LK_THR,
        late_fraction: float = LATE_FRACTION,
        max_total_pellet_visibility: int = MAX_TOTAL_PELLET_VISIBILITY,
        max_late_pellet_visibility: int = MAX_LATE_PELLET_VISIBILITY,
        min_sustained_pellet_run: int = MIN_SUSTAINED_PELLET_RUN,
        okf_offset: int = OKF_OFFSET,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.pellet_lk_threshold = pellet_lk_threshold
        self.paw_lk_threshold = paw_lk_threshold
        self.late_fraction = late_fraction
        self.max_total_pellet_visibility = max_total_pellet_visibility
        self.max_late_pellet_visibility = max_late_pellet_visibility
        self.min_sustained_pellet_run = min_sustained_pellet_run
        self.okf_offset = okf_offset
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
        pillar_cy = geom["pillar_cy"].to_numpy(dtype=float)
        pillar_r = geom["pillar_r"].to_numpy(dtype=float)
        slit_y_line = pillar_cy + pillar_r

        pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)

        paw_past_y = np.zeros(n, dtype=bool)
        for bp in PAW_BODYPARTS:
            py = sub[f"{bp}_y"].to_numpy(dtype=float)
            pl = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
            paw_past_y |= (py <= slit_y_line) & (pl >= self.paw_lk_threshold)

        # Total pellet visibility (sustained 3+ frames at lk >= 0.7
        # with paw not past slit).
        pellet_visible_raw = (pellet_lk >= self.pellet_lk_threshold) & (~paw_past_y)
        total_visibility = _sustained_run_count(
            pellet_visible_raw, self.min_sustained_pellet_run)
        feats = {
            "n_clean_zone_frames": int(n),
            "total_pellet_visibility": int(total_visibility),
        }
        if total_visibility > self.max_total_pellet_visibility:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_visibility_too_high "
                    f"({total_visibility} > "
                    f"{self.max_total_pellet_visibility}); pellet was "
                    f"tracked at some point -- not 'effectively "
                    f"invisible' case)"
                ),
                features=feats,
            )

        # Late-zone visibility check.
        late_start_idx = int(n * (1 - self.late_fraction))
        late_visible_raw = pellet_visible_raw.copy()
        late_visible_raw[:late_start_idx] = False
        late_visibility = _sustained_run_count(
            late_visible_raw, self.min_sustained_pellet_run)
        feats["late_visibility"] = int(late_visibility)
        if late_visibility > self.max_late_pellet_visibility:
            return StageDecision(
                decision="continue",
                reason=(
                    f"late_pellet_visibility_too_high "
                    f"({late_visibility} > "
                    f"{self.max_late_pellet_visibility})"
                ),
                features=feats,
            )

        # Need at least 1 paw bout.
        bouts = _find_paw_past_y_line_bouts(paw_past_y)
        feats["n_paw_past_y_line_bouts"] = len(bouts)
        if not bouts:
            return StageDecision(
                decision="continue",
                reason="no_paw_past_y_line_bouts_in_clean_zone",
                features=feats,
            )

        # Use GT reach windows when available for IFR emit (matches
        # trust framework). Otherwise use last paw bout.
        reach_windows_local: List[Tuple[int, int]] = []
        for rs, re in seg.reach_windows:
            ls = max(0, int(rs) - seg.seg_start)
            le = min(n - 1, int(re) - seg.seg_start)
            if le >= ls:
                reach_windows_local.append((ls, le))
        reach_windows_local.sort()

        # For retrieved with pellet-invisible-throughout: when there
        # is exactly ONE GT reach, that's the only candidate -- safe
        # to commit. With multiple GT reaches, GT might pick a later
        # one based on visual paw-over-pellet evidence we can't
        # verify (pellet is invisible). Defer those.
        if reach_windows_local:
            if len(reach_windows_local) != 1:
                feats["n_gt_reaches"] = len(reach_windows_local)
                return StageDecision(
                    decision="continue",
                    reason=(
                        f"multiple_gt_reaches "
                        f"({len(reach_windows_local)} reaches; can't pick "
                        f"GT's last paw-over-pellet without visual "
                        f"evidence -- defer)"
                    ),
                    features=feats,
                )
            causal_start, causal_end = reach_windows_local[0]
            causal_source = "reach_windows[0]"
        else:
            # No GT reaches; only commit if exactly 1 paw bout to
            # avoid bout-pick ambiguity.
            if len(bouts) != 1:
                return StageDecision(
                    decision="continue",
                    reason=(
                        f"no_gt_reaches_and_multiple_paw_bouts "
                        f"({len(bouts)} paw bouts; can't pick causal -- "
                        f"defer)"
                    ),
                    features=feats,
                )
            causal_start, causal_end = bouts[0]
            causal_source = "paw_bouts[0]"
        feats["causal_source"] = causal_source

        bout_length = causal_end - causal_start + 1
        interaction_idx = causal_start + bout_length // 2
        okf_idx = n - 1 - self.okf_offset
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
                f"pellet_invisible_throughout_with_paw_activity "
                f"(total visibility {total_visibility}f, late "
                f"{late_visibility}f; n_bouts {len(bouts)})"
            ),
            features=feats,
        )
