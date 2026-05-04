"""
Stage 25: Retry with stricter pellet-confidence floor (>= 0.85).

Hypothesis: most "off-pillar pellet seen" false positives in residual
cases come from DLC firing the Pellet bodypart at 0.7-0.85 confidence
on label-switch targets (pillar tip, paw, etc.). Real pellet
observations are typically >= 0.95 when the pellet is truly visible.

This stage re-runs the cascade's retrieved/displaced detectors
(Stages 9, 16, 17, 21) with the segment's DLC modified so that any
Pellet observation at lk < 0.85 is treated as if it weren't observed
(lk set to 0.0). The conservative approach: don't fabricate
observations, just suppress unreliable ones.

If this filtering causes Stage 21 (or 9/16/17) to commit cleanly when
they wouldn't on raw DLC, the segment commits here.

Trust framework: same as before. The class call must match GT and
the IFR must fall in the same GT reach.
"""
from __future__ import annotations

from .stage_base import SegmentInput, Stage, StageDecision
from .stage_9_pellet_vanished_after_reach import Stage9PelletVanishedAfterReach
from .stage_16_displaced_via_max_displacement_reach import Stage16DisplacedViaMaxDisplacement
from .stage_17_displaced_via_dominant_max_displacement import Stage17DisplacedViaDominantMaxDisplacement
from .stage_21_causal_reach_via_immediate_on_off_transition import (
    Stage21CausalReachViaImmediateOnOffTransition)


PELLET_CONFIDENCE_FLOOR = 0.85
# Defense: if pellet was seen off-pillar at moderate conf (0.5+) in
# the original DLC at any sustained 5+ frames, the segment might
# actually be displaced with poor DLC tracking. Don't commit retrieved
# in that case -- defer to triage instead.
ANTI_RETRIEVED_OFF_PILLAR_LK = 0.5
ANTI_RETRIEVED_OFF_PILLAR_RADII = 1.5
ANTI_RETRIEVED_MAX_SUSTAINED = 10


class Stage25RetryWithStrictPelletConfidence(Stage):
    name = "stage_25_retry_with_strict_pellet_confidence"
    target_class = None  # commits whatever inner stage commits

    def __init__(self):
        self.retry_stages = [
            ("S21", Stage21CausalReachViaImmediateOnOffTransition()),
            ("S9", Stage9PelletVanishedAfterReach()),
            ("S16", Stage16DisplacedViaMaxDisplacement()),
            ("S17", Stage17DisplacedViaDominantMaxDisplacement()),
        ]

    def decide(self, seg: SegmentInput) -> StageDecision:
        # Suppress Pellet observations below the strict floor by setting
        # their lk to 0. Don't touch x/y -- the position is whatever
        # DLC said; we just don't trust it.
        df = seg.dlc_df.copy()
        if "Pellet_likelihood" in df.columns:
            mask = df["Pellet_likelihood"] < PELLET_CONFIDENCE_FLOOR
            # Use .loc with the boolean mask to avoid SettingWithCopy.
            df.loc[mask, "Pellet_likelihood"] = 0.0
        strict_seg = SegmentInput(
            video_id=seg.video_id,
            segment_num=seg.segment_num,
            seg_start=seg.seg_start,
            seg_end=seg.seg_end,
            dlc_df=df,
            reach_windows=seg.reach_windows,
        )
        # Compute anti-retrieved defense signal: did the ORIGINAL DLC
        # see the pellet sustained off-pillar at moderate confidence?
        # If yes, can't commit retrieved -- might be displaced with
        # poor DLC tracking.
        from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
        from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
        import numpy as np

        clean_end = seg.seg_end - 5
        if clean_end > seg.seg_start:
            sub_raw = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
            sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
            geom = compute_pillar_geometry_series(sub)
            pillar_cx = geom["pillar_cx"].to_numpy(dtype=float)
            pillar_cy = geom["pillar_cy"].to_numpy(dtype=float)
            pillar_r = geom["pillar_r"].to_numpy(dtype=float)
            pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
            pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
            pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
            dist_radii = (np.sqrt((pellet_x - pillar_cx) ** 2
                                  + (pellet_y - pillar_cy) ** 2)
                          / np.maximum(pillar_r, 1e-6))
            off_pillar_moderate = (
                (pellet_lk >= ANTI_RETRIEVED_OFF_PILLAR_LK)
                & (dist_radii > ANTI_RETRIEVED_OFF_PILLAR_RADII)
            )
            run = 0
            max_run = 0
            for v in off_pillar_moderate:
                if v:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
            anti_retrieved_block = max_run > ANTI_RETRIEVED_MAX_SUSTAINED
        else:
            anti_retrieved_block = False

        for label, stage in self.retry_stages:
            decision = stage.decide(strict_seg)
            if decision.decision == "commit":
                # Anti-retrieved defense: if low-conf off-pillar evidence
                # exists in original DLC, don't commit retrieved.
                if (decision.committed_class == "retrieved"
                        and anti_retrieved_block):
                    return StageDecision(
                        decision="continue",
                        reason=(
                            f"strict_retry_via_{label}_would_commit_retrieved "
                            f"but moderate-conf off-pillar evidence "
                            f"present in original DLC -- could be "
                            f"displaced; defer"
                        ),
                    )
                decision.reason = (
                    f"[strict-pellet-conf-retry via {label}] " + decision.reason
                )
                return decision
        return StageDecision(
            decision="continue",
            reason="no_retry_stage_committed_with_strict_pellet_confidence",
        )
