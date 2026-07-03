"""
Stage 3: Paw never approached the pellet area (4.0 recalibration).

Question this stage answers:
    "Did the paw ever get within APPROACH_RADII pillar-radii of the
    pillar center (sustained 3-frame, while past the slit line at
    lk>=0.5)?"

4.0 recalibration note:
    The 3.1 y-line + lk-floor(0.22) signal is DEAD on 4.0 (paw
    confidently past the reaching line in nearly every segment;
    untouched/touched fully overlap). Replaced with a distance signal:
    'no reach could have touched' = the paw never got within
    APPROACH_RADII pillar-radii of the pillar center (sustained
    3-frame, while past the slit line at lk>=0.5).

    Additionally includes a pellet-displaced guard: the paw signal can
    drop out (poorly tracked paw reads as 'never near'), so if the
    pellet demonstrably moved into the SA (sustained >3r for >=
    SUSTAINED_SA_RUN frames), it WAS touched -- do not commit untouched
    regardless of paw visibility.

Cascade emit on commit:
- committed_class: "untouched"
- whens["outcome_known_frame"]: clean_end (last clean-zone frame)
- whens["interaction_frame"]: None
"""
from __future__ import annotations

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .guards import lrun
from .stage_base import SegmentInput, Stage, StageDecision

PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5
APPROACH_RADII = 2.5
SUSTAINED_SA_RUN = 15


class Stage3PawNeverInPelletArea(Stage):
    name = "stage_3_paw_never_in_pellet_area"
    target_class = "untouched"

    def __init__(
        self,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.transition_zone_half = transition_zone_half

    def decide(self, seg: SegmentInput) -> StageDecision:
        ce = seg.seg_end - self.transition_zone_half
        if ce <= seg.seg_start:
            return StageDecision(
                decision="continue",
                reason="too_short")
        sub_raw = seg.dlc_df.iloc[seg.seg_start:ce + 1]
        n = len(sub_raw)
        if n < 5:
            return StageDecision(decision="continue", reason="short")
        sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
        g = compute_pillar_geometry_series(sub)
        cx = g["pillar_cx"].to_numpy(float)
        cy = g["pillar_cy"].to_numpy(float)
        r = g["pillar_r"].to_numpy(float)
        slit = cy + r
        dpil = np.full(n, np.inf)
        for bp in PAW_BODYPARTS:
            yy = sub[f"{bp}_y"].to_numpy(float)
            xx = sub[f"{bp}_x"].to_numpy(float)
            ll = sub_raw[f"{bp}_likelihood"].to_numpy(float)
            ok = (yy <= slit) & (ll >= 0.5)
            dpil = np.minimum(
                dpil,
                np.where(ok,
                         np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                         / np.maximum(r, 1e-6),
                         np.inf))
        best = np.inf
        for i in range(n - 2):
            w = dpil[i:i + 3]
            if np.all(np.isfinite(w)):
                best = min(best, float(w.mean()))
        # Pellet-displaced guard: the paw signal can drop out (poorly
        # tracked paw reads as 'never near'), so if the pellet
        # demonstrably moved into the SA (sustained >3r), it WAS
        # touched -- do not commit untouched regardless of paw visibility.
        pex = sub["Pellet_x"].to_numpy(float)
        pey = sub["Pellet_y"].to_numpy(float)
        plk = sub_raw["Pellet_likelihood"].to_numpy(float)
        prad = np.sqrt((pex - cx) ** 2 + (pey - cy) ** 2) / np.maximum(r, 1e-6)
        sa_run = lrun((plk >= 0.7) & (prad > 3.0))
        if best > APPROACH_RADII and sa_run < SUSTAINED_SA_RUN:
            return StageDecision(
                decision="commit", committed_class="untouched",
                whens={"outcome_known_frame": int(ce),
                       "interaction_frame": None},
                reason=(
                    f"paw_never_within_{APPROACH_RADII}r_of_pillar "
                    f"(closest={best:.2f}, sa_run={sa_run})"))
        return StageDecision(
            decision="continue",
            reason=(
                f"paw_approached_or_pellet_in_SA "
                f"(closest={best:.2f}, sa_run={sa_run})"))
