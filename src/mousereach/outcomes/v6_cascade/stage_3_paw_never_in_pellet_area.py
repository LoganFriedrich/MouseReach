"""
Stage 2: No reach that could have possibly touched the pellet.

Semantic question this stage answers:
    "Did the mouse perform any reach during the segment that could have
    possibly made contact with the pellet?"

Operationalized as: across the entire current-segment clean zone, did
any paw point ever cross past the slit-closest pillar y line into
pellet-reaching territory at sustained tracking confidence (3-frame
rolling mean of per-frame in-zone max-likelihood >= empirical floor)?
A reach that does not satisfy this minimum could not have physically
made contact with the pellet regardless of pellet position.

If NO -> commit untouched (no reach that could have touched the pellet
            ever occurred).
If YES -> continue.

A "paw point in pellet area" requires three simultaneous conditions:
1. paw_y <= pillar_cy + pillar_r (past slit-closest pillar y -- physical
   prerequisite for touching)
2. distance(paw_point, pellet) <= 2 * pillar_radius (within touching
   distance)
3. lk above the empirical contact-reach floor when smoothed across 3
   frames (filters single-frame DLC noise)

Per-frame "in-zone" max-lk is taken across all 4 paw points satisfying
conditions 1 and 2. Three-frame rolling mean of that max-lk threshold
is 0.22 -- the absolute lowest sustained-3-frame max-lk observed across
all 381 GT-confirmed causal contact reaches in the corpus. Below 0.22,
no real contact reach in the corpus ever sustained the paw confidently
in the pellet zone for 3 consecutive frames.

Cascade OKF emit on commit: seg_end - TRANSITION_ZONE_HALF (last clean
frame of current segment).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import SegmentInput, Stage, StageDecision

PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5
# Stage 2's "paw in pellet zone" predicate: ANY paw point past the
# slit-closest pillar y line (paw_y <= pillar_cy + pillar_r) at lk
# above the empirical floor when smoothed across 3 consecutive frames.
# No distance-from-pillar-center bound -- the y-line alone is the
# physical gate. A paw past that line at confident lk is in extension/
# reach territory and could have made contact regardless of x offset.
LK_FLOOR_3F_ROLLING = 0.22    # empirical contact-reach 3-frame floor
ROLLING_WINDOW = 3
CONSECUTIVE_REQUIRED = 3      # min consecutive past-line frames to trigger


class Stage3PawNeverInPelletArea(Stage):
    name = "stage_3_paw_never_in_pellet_area"
    target_class = "untouched"

    def __init__(
        self,
        lk_floor: float = LK_FLOOR_3F_ROLLING,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.lk_floor = lk_floor
        self.transition_zone_half = transition_zone_half

    def decide(self, seg: SegmentInput) -> StageDecision:
        clean_end = seg.seg_end - self.transition_zone_half
        if clean_end <= seg.seg_start:
            return StageDecision(
                decision="continue",
                reason="segment_too_short_to_have_clean_zone")
        sub = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
        n = len(sub)
        if n == 0:
            return StageDecision(decision="continue", reason="empty_segment")

        geom = compute_pillar_geometry_series(sub)
        pillar_cy = geom["pillar_cy"].to_numpy()
        pillar_r = geom["pillar_r"].to_numpy()
        slit_edge_y = pillar_cy + pillar_r

        # The y-line gate is the only physical filter. Any paw point
        # past the slit-closest pillar y line could be in position to
        # touch the pellet -- distance offset doesn't matter because
        # the mouse has actively extended into pellet vertical range.

        # Per-frame: max-lk across paw points past the slit-closest
        # pillar y line.
        per_frame_max_lk = np.zeros(n, dtype=float)
        per_frame_in_zone = np.zeros(n, dtype=bool)
        for bp in PAW_BODYPARTS:
            paw_y = sub[f"{bp}_y"].to_numpy(dtype=float)
            paw_lk = sub[f"{bp}_likelihood"].to_numpy(dtype=float)

            past_edge = paw_y <= slit_edge_y
            in_zone = past_edge
            per_frame_in_zone |= in_zone

            # Update max-lk where this paw is past edge AND has higher lk
            update_mask = in_zone & (paw_lk > per_frame_max_lk)
            per_frame_max_lk = np.where(
                update_mask, paw_lk, per_frame_max_lk)

        # Find consecutive runs of in-zone frames; for each run of >=
        # CONSECUTIVE_REQUIRED frames, compute 3-frame rolling mean of
        # max-lk; check if ANY rolling mean meets/exceeds the floor.
        triggered = False
        max_rolling_mean_observed = 0.0
        max_run_length = 0
        run_start = -1
        for i in range(n):
            if per_frame_in_zone[i]:
                if run_start < 0:
                    run_start = i
            else:
                if run_start >= 0:
                    run_len = i - run_start
                    if run_len > max_run_length:
                        max_run_length = run_len
                    if run_len >= CONSECUTIVE_REQUIRED:
                        run_lks = per_frame_max_lk[run_start:i]
                        for j in range(len(run_lks) - ROLLING_WINDOW + 1):
                            m = float(run_lks[j:j + ROLLING_WINDOW].mean())
                            if m > max_rolling_mean_observed:
                                max_rolling_mean_observed = m
                            if m >= self.lk_floor:
                                triggered = True
                                break
                    run_start = -1
                if triggered:
                    break
        # Tail
        if not triggered and run_start >= 0:
            run_len = n - run_start
            if run_len > max_run_length:
                max_run_length = run_len
            if run_len >= CONSECUTIVE_REQUIRED:
                run_lks = per_frame_max_lk[run_start:n]
                for j in range(len(run_lks) - ROLLING_WINDOW + 1):
                    m = float(run_lks[j:j + ROLLING_WINDOW].mean())
                    if m > max_rolling_mean_observed:
                        max_rolling_mean_observed = m
                    if m >= self.lk_floor:
                        triggered = True
                        break

        feats = {
            "n_frames_in_zone": int(per_frame_in_zone.sum()),
            "max_in_zone_run_length": int(max_run_length),
            "max_3frame_rolling_mean_lk": float(max_rolling_mean_observed),
            "lk_floor_used": float(self.lk_floor),
        }

        if not triggered:
            okf = clean_end
            feats["outcome_known_frame_emitted"] = int(okf)
            return StageDecision(
                decision="commit",
                committed_class="untouched",
                whens={"outcome_known_frame": int(okf),
                       "interaction_frame": None},
                reason=(
                    f"paw_never_confidently_in_pellet_area "
                    f"(max 3-frame rolling mean = "
                    f"{max_rolling_mean_observed:.3f} < floor "
                    f"{self.lk_floor})"
                ),
                features=feats,
            )

        return StageDecision(
            decision="continue",
            reason=(
                f"paw_entered_pellet_area_with_sustained_confidence "
                f"(max 3-frame rolling mean = "
                f"{max_rolling_mean_observed:.3f} >= floor "
                f"{self.lk_floor})"
            ),
            features=feats,
        )
