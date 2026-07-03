"""
Stage 6b: Pellet on-pillar, never displaced into SA, never vanished.

Question this stage answers:
    "Is the pellet confidently tracked (lk>=0.7 for >=70% of frames),
    present late (>=50% of last 30 frames), median radius < 1.8 when
    confident, never vanished (longest lk<0.7 run <= 10), and never
    sustained in the SA (longest confident >3r run < 15)?"

If yes -> COMMIT untouched. The pellet was present throughout and
never left the pillar area.

Position in cascade:
    Immediately after stage_6 (predominantly on pillar). Catches
    additional untouched segments that stage_6's visibility/frac_inside
    thresholds miss, where the pellet was clearly present and near
    the pillar throughout but with minor tracking noise.

Cascade emit on commit:
- committed_class: "untouched"
- whens["outcome_known_frame"]: seg_end - 5 (last clean-zone frame)
- whens["interaction_frame"]: None
"""
from __future__ import annotations

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .guards import lrun
from .stage_base import SegmentInput, Stage, StageDecision


class Stage6bNeverEnteredSA(Stage):
    name = "stage_6b_never_entered_sa"
    target_class = "untouched"

    def decide(self, seg: SegmentInput) -> StageDecision:
        ce = seg.seg_end - 5
        if ce <= seg.seg_start:
            return StageDecision(decision="continue", reason="too_short")
        sub_raw = seg.dlc_df.iloc[seg.seg_start:ce + 1]
        n = len(sub_raw)
        if n < 20:
            return StageDecision(decision="continue", reason="short")
        sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
        g = compute_pillar_geometry_series(sub)
        cx = g["pillar_cx"].to_numpy(float)
        cy = g["pillar_cy"].to_numpy(float)
        r = g["pillar_r"].to_numpy(float)
        px = sub["Pellet_x"].to_numpy(float)
        py = sub["Pellet_y"].to_numpy(float)
        plk = sub_raw["Pellet_likelihood"].to_numpy(float)
        radii = (np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                 / np.maximum(r, 1e-6))
        conf = plk >= 0.7
        if conf.mean() < 0.7:
            return StageDecision(decision="continue",
                                 reason="poorly_tracked")
        if conf[max(0, n - 30):].mean() < 0.5:
            return StageDecision(decision="continue",
                                 reason="absent_late")
        rc = radii[conf]
        if rc.size == 0 or np.median(rc) >= 1.8:
            return StageDecision(decision="continue",
                                 reason="pellet_rests_off_pillar")
        if lrun(plk < 0.7) > 10:
            return StageDecision(decision="continue",
                                 reason="pellet_vanishes_retrieved")
        if lrun(conf & (radii > 3.0)) >= 15:
            return StageDecision(decision="continue",
                                 reason="sustained_SA")
        return StageDecision(
            decision="commit", committed_class="untouched",
            whens={"outcome_known_frame": int(ce),
                   "interaction_frame": None},
            reason="on_pillar_never_displaced_present_throughout")
