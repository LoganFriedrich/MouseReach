"""
Stage 22: Retry with stabilized DLC pellet data.

Question this stage answers:
    "If we conservatively impute short gaps in pellet DLC tracking
    (only when surrounding frames have stable position), does Stage 21
    (causal-reach via immediate on/off transition) now commit a class
    that it couldn't before?"

Why:
    Per user 2026-05-03: "we are basically doing something really
    smart which is not messing with what we are looking at unless we
    really really need to". The cascade runs Stages 1-21 on raw DLC.
    For segments still in residual after that, this stage re-runs
    Stage 21 logic with conservatively-stabilized pellet data.

Stabilization (see pellet_stabilizer.py):
    - Imputes pellet position only across SHORT gaps (<= 10 frames)
      where bounding high-conf detections are at similar positions
      (<= 15 px apart).
    - Never alters high-conf detections.
    - Never smooths across position discontinuities (= real
      displacement events).

Trust framework: stabilized DLC may give Stage 21 enough continuity to
commit retrieval/displacement cleanly where the original DLC's gaps
caused phantom transitions. Stage 21's defenses (no return to pillar,
late-zone evidence, single-bout, etc.) still apply -- so the commit
is gated by physics, just measured against cleaner DLC.
"""
from __future__ import annotations

from .pellet_stabilizer import stabilize_pellet_dlc
from .stage_9_pellet_vanished_after_reach import Stage9PelletVanishedAfterReach
from .stage_16_displaced_via_max_displacement_reach import Stage16DisplacedViaMaxDisplacement
from .stage_17_displaced_via_dominant_max_displacement import Stage17DisplacedViaDominantMaxDisplacement
from .stage_21_causal_reach_via_immediate_on_off_transition import (
    Stage21CausalReachViaImmediateOnOffTransition)
from .stage_base import SegmentInput, Stage, StageDecision


class Stage22RetryWithStabilizedDlc(Stage):
    name = "stage_22_retry_with_stabilized_dlc"
    target_class = None  # commits whatever inner stage commits

    def __init__(self):
        # Retry stages in order. First commit wins.
        self.retry_stages = [
            ("S21", Stage21CausalReachViaImmediateOnOffTransition()),
            ("S9", Stage9PelletVanishedAfterReach()),
            ("S16", Stage16DisplacedViaMaxDisplacement()),
            ("S17", Stage17DisplacedViaDominantMaxDisplacement()),
        ]

    def decide(self, seg: SegmentInput) -> StageDecision:
        # Build a SegmentInput with stabilized DLC. Conservative
        # parameters: max_gap=5 frames, position_tolerance=10px.
        stabilized_df = stabilize_pellet_dlc(
            seg.dlc_df, max_gap_frames=5, position_tolerance_px=10.0)
        stabilized_seg = SegmentInput(
            video_id=seg.video_id,
            segment_num=seg.segment_num,
            seg_start=seg.seg_start,
            seg_end=seg.seg_end,
            dlc_df=stabilized_df,
            reach_windows=seg.reach_windows,
        )
        for label, stage in self.retry_stages:
            decision = stage.decide(stabilized_seg)
            if decision.decision == "commit":
                decision.reason = (
                    f"[stabilized-dlc-retry via {label}] " + decision.reason
                )
                return decision
        return StageDecision(
            decision="continue",
            reason="no_retry_stage_committed_with_stabilized_dlc",
        )
