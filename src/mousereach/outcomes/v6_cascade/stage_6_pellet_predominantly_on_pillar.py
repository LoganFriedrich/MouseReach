"""
Stage 6: Pellet predominantly on pillar (permissive untouched commit).

Question this stage answers:
    "Was the pellet visible AND inside the pillar circle for almost
    every confident, non-during-reach frame in the segment's clean
    zone?"

If yes -> COMMIT untouched. The pellet stayed visible (didn't get
eaten or displaced out of view) AND it stayed within the pillar
circle for ~100% of the time it was visible. By physical causality,
no reach displaced or retrieved the pellet.

This is a more permissive untouched stage that catches segments where:
- Stage 1 (position-never-changed) rejected because the pellet's
  position-stability was just barely over the std threshold (often
  caused by tray drift in y not perfectly cancelled by pillar
  geometry, or by short segments below the 100-frame minimum)
- Stage 2 (stable-on-pillar) rejected because the analysis window had
  brief outside-circle blips (DLC noise at the segment end)
- Stages 3-5 rejected on their specific predicates
- BUT the pellet was clearly on the pillar throughout the segment in
  aggregate (frac_inside >= 0.99 of confident frames)

Empirical calibration (2026-05-02 corpus, train_pool, 740 GT segments):
- Untouched class: visibility_frac p1 = 0.993; frac_inside p25 = 0.994
- Retrieved class: visibility_frac p99 = 0.970 (pellet disappears
  when eaten); frac_inside p75 = 0.983
- Displaced_sa class: most have very low frac_inside (pellet visibly
  off-pillar after displacement)
- Joint envelope test (frac_inside >= 0.99 AND visibility_frac >= 0.99):
  catches 7 of 11 known residual untouched cases with ZERO
  contamination from displaced_sa, retrieved, or abnormal_exception.

Cascade emit on commit:
- committed_class: "untouched"
- whens["outcome_known_frame"]: seg_end - TRANSITION_ZONE_HALF
  (last clean-zone frame, matching other untouched stages)
- whens["interaction_frame"]: None
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import SegmentInput, Stage, StageDecision


PELLET_LK_THR = 0.7
ON_PILLAR_RADII = 1.0
TRANSITION_ZONE_HALF = 5

# Commit thresholds. Both calibrated empirically on the train_pool to
# yield ZERO contamination from non-untouched GT classes. Tightening
# either further only reduces yield without improving safety; loosening
# either introduces displaced_sa contamination.
FRAC_INSIDE_THR = 0.99
VISIBILITY_FRAC_THR = 0.99


def _during_reach_mask(seg_start: int, seg_end: int,
                       reach_windows: List[Tuple[int, int]]) -> np.ndarray:
    n = seg_end - seg_start + 1
    m = np.zeros(n, dtype=bool)
    for rs, re in reach_windows:
        s = max(seg_start, int(rs))
        e = min(seg_end, int(re))
        if e < s:
            continue
        m[s - seg_start:e - seg_start + 1] = True
    return m


class Stage6PelletPredominantlyOnPillar(Stage):
    name = "stage_6_pellet_predominantly_on_pillar"
    target_class = "untouched"

    def __init__(
        self,
        frac_inside_threshold: float = FRAC_INSIDE_THR,
        visibility_frac_threshold: float = VISIBILITY_FRAC_THR,
        pellet_lk_threshold: float = PELLET_LK_THR,
        on_pillar_radii: float = ON_PILLAR_RADII,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.frac_inside_threshold = frac_inside_threshold
        self.visibility_frac_threshold = visibility_frac_threshold
        self.pellet_lk_threshold = pellet_lk_threshold
        self.on_pillar_radii = on_pillar_radii
        self.transition_zone_half = transition_zone_half

    def decide(self, seg: SegmentInput) -> StageDecision:
        clean_end = seg.seg_end - self.transition_zone_half
        if clean_end <= seg.seg_start:
            return StageDecision(
                decision="continue",
                reason="segment_too_short_to_have_clean_zone")

        sub_raw = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
        n = len(sub_raw)
        if n == 0:
            return StageDecision(decision="continue", reason="empty_segment")

        geom = compute_pillar_geometry_series(sub_raw)
        pillar_cx = geom["pillar_cx"].to_numpy(dtype=float)
        pillar_cy = geom["pillar_cy"].to_numpy(dtype=float)
        pillar_r = geom["pillar_r"].to_numpy(dtype=float)

        pellet_x = sub_raw["Pellet_x"].to_numpy(dtype=float)
        pellet_y = sub_raw["Pellet_y"].to_numpy(dtype=float)
        pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
        dist_radii = (np.sqrt((pellet_x - pillar_cx) ** 2
                              + (pellet_y - pillar_cy) ** 2)
                      / np.maximum(pillar_r, 1e-6))

        during = _during_reach_mask(seg.seg_start, clean_end, seg.reach_windows)
        not_during = ~during
        n_not_during = int(not_during.sum())
        if n_not_during == 0:
            return StageDecision(
                decision="continue",
                reason="no_non_during_reach_frames")

        confident = pellet_lk >= self.pellet_lk_threshold
        eligible = not_during & confident
        n_eligible = int(eligible.sum())
        visibility_frac = float(n_eligible / n_not_during)
        feats = {
            "n_clean_zone_frames": n,
            "n_non_during_reach_frames": n_not_during,
            "n_confident_eligible_frames": n_eligible,
            "visibility_frac": visibility_frac,
        }

        if visibility_frac < self.visibility_frac_threshold:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_visibility_too_low "
                    f"(vis_frac={visibility_frac:.3f} < "
                    f"{self.visibility_frac_threshold:.2f}); pellet may "
                    f"have been retrieved/eaten/obscured -- defer)"
                ),
                features=feats,
            )

        if n_eligible == 0:
            return StageDecision(
                decision="continue",
                reason="no_confident_eligible_frames_for_position_check")

        inside = (dist_radii <= self.on_pillar_radii) & eligible
        frac_inside = float(inside.sum() / n_eligible)
        n_off_pillar = int(((dist_radii > self.on_pillar_radii) & eligible).sum())
        feats.update({
            "frac_inside_pillar_circle": frac_inside,
            "n_off_pillar_frames": n_off_pillar,
        })

        if frac_inside < self.frac_inside_threshold:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_not_predominantly_inside_pillar_circle "
                    f"(frac_inside={frac_inside:.3f} < "
                    f"{self.frac_inside_threshold:.2f}, "
                    f"{n_off_pillar} off-pillar frames); pellet had "
                    f"meaningful off-pillar evidence -- defer)"
                ),
                features=feats,
            )

        okf = int(seg.seg_end - self.transition_zone_half)
        feats["outcome_known_frame_emitted"] = okf
        return StageDecision(
            decision="commit",
            committed_class="untouched",
            whens={"outcome_known_frame": okf,
                   "interaction_frame": None},
            reason=(
                f"pellet_predominantly_on_pillar "
                f"(frac_inside={frac_inside:.3f}, "
                f"visibility_frac={visibility_frac:.3f}, "
                f"n_off_pillar={n_off_pillar}/{n_eligible}; pellet "
                f"stayed visible AND inside the pillar circle for "
                f"essentially the entire segment)"
            ),
            features=feats,
        )
