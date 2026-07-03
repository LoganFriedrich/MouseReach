"""
Stage 1: On-pillar physical invariant (4.0 recalibration).

Question this stage answers:
    "In the last 30 frames of the clean zone, is the pellet confidently
    detected on the pillar (within 1.0 pillar-radii at lk>=0.9) for at
    least 50% of frames?"

If yes AND the pellet was not displaced or vanished (shared guard) ->
COMMIT untouched. The pellet is demonstrably still on the pillar at
segment end.

4.0 recalibration note:
    The original Stage 1 (position-never-changed) used per-frame xy
    position stability across the whole segment. On DLC 4.0, tray drift
    and paw-near-pellet DLC wobble inflate position std, causing
    excessive deferrals. This replacement uses a direct physical
    invariant: is the pellet ON the pillar at the end of the segment?
    Combined with the displaced_or_vanished guard, this catches clean
    untouched segments robustly.

Cascade emit on commit:
- committed_class: "untouched"
- whens["outcome_known_frame"]: seg_end - 5 (last clean-zone frame)
- whens["interaction_frame"]: None
"""
from __future__ import annotations

import numpy as np

from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .guards import pellet_displaced_or_vanished
from .stage_base import SegmentInput, Stage, StageDecision


class Stage1PelletPositionNeverChanged(Stage):
    name = "stage_1_pellet_position_never_changed"
    target_class = "untouched"

    def decide(self, seg: SegmentInput) -> StageDecision:
        ce = seg.seg_end - 5
        if ce <= seg.seg_start:
            return StageDecision(decision="continue", reason="too_short")
        sub = seg.dlc_df.iloc[seg.seg_start:ce + 1]
        n = len(sub)
        if n == 0:
            return StageDecision(decision="continue", reason="empty")
        g = compute_pillar_geometry_series(sub)
        cx = g["pillar_cx"].to_numpy(float)
        cy = g["pillar_cy"].to_numpy(float)
        r = g["pillar_r"].to_numpy(float)
        px = sub["Pellet_x"].to_numpy(float)
        py = sub["Pellet_y"].to_numpy(float)
        plk = sub["Pellet_likelihood"].to_numpy(float)
        dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        onp = (plk >= 0.9) & np.isfinite(r) & (dist <= 1.0 * r)
        late = onp[max(0, n - 30):]
        if len(late) < 10:
            return StageDecision(decision="continue", reason="late_short")
        if late.mean() >= 0.5:
            if pellet_displaced_or_vanished(seg.dlc_df, seg.seg_start,
                                            seg.seg_end):
                return StageDecision(
                    decision="continue",
                    reason="pellet_displaced_or_vanished_guard")
            return StageDecision(
                decision="commit", committed_class="untouched",
                whens={"outcome_known_frame": int(ce),
                       "interaction_frame": None},
                reason="on_pillar_late")
        return StageDecision(decision="continue",
                             reason="not_on_pillar_late")
