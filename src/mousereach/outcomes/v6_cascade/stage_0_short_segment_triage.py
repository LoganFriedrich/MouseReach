"""
Stage 0: Short-segment data-quality triage.

Question this stage answers:
    "Is this segment so abnormally short that it cannot be a normal
    pellet trial?"

If yes -> TRIAGE with reason `abnormally_short_segment`. The segment
is excluded from normal cascade evaluation.

Rationale:
    Typical MouseReach segments span ~1840 frames. End-of-video
    apparatus failure / cycling abort produces truncated segments of
    62-300 frames that aren't real pellet trials -- the apparatus
    didn't successfully present a pellet for the full duration. These
    cases should be flagged for human review, not pushed through
    cascade rules that assume full-length trials.

Position in cascade:
    FIRST. This is a pre-screen: any segment that fails the data-
    quality check is triaged before any committing stage gets a
    chance to evaluate it.

Empirical calibration (2026-05-02 corpus, train_pool, 740 segments):
- Normal segments: 1825-1851 frames (very tight, p1-p100)
- Below 200 frames: 5 untouched + 1 retrieved (all data-quality)
- Below 300 frames: same 6 (no new touched-class additions)
- Below 500 frames: adds 1 displaced_sa (length 316; potentially a
  legitimately truncated real trial -- don't triage)
- Threshold of 300 captures clear data-quality cases without grabbing
  real truncated trials.

Cascade emit on triage:
- decision: "triage"
- reason: explicit abnormally_short_segment
- features: include the length so downstream tooling can confirm
"""
from __future__ import annotations

from .stage_base import SegmentInput, Stage, StageDecision


# Threshold below which a segment is considered "abnormally short" /
# data-quality issue. Calibrated 2026-05-02 against the train_pool
# corpus distribution: normal segments are 1825+ frames; below 300
# only data-quality outliers exist.
MIN_NORMAL_SEGMENT_LENGTH = 300


class Stage0ShortSegmentTriage(Stage):
    name = "stage_0_short_segment_triage"
    target_class = None  # triage only, no commit

    def __init__(
        self,
        min_normal_segment_length: int = MIN_NORMAL_SEGMENT_LENGTH,
    ):
        self.min_normal_segment_length = min_normal_segment_length

    def decide(self, seg: SegmentInput) -> StageDecision:
        # 4.0 recalibration: segment length is algo 1's concern, not
        # algo 3's. Bypass length-based triage entirely so short
        # segments proceed through the cascade for outcome classification.
        return StageDecision(decision="continue", reason="stage0_bypassed")
