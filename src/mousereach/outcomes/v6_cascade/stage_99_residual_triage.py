"""
Stage 99: Residual triage (always last in the cascade).

Question this stage answers:
    "If we got here, no committing stage was confident enough to
    commit a class. What do we do?"

Answer: TRIAGE. Anything reaching this stage is by definition outside
the cascade's reasonable scoring competence -- it must go to human
review. This makes the residual triage explicit rather than implicit
in the cascade ordering.

Position:
    Always last. No committing stage runs after this. By design this
    stage NEVER defers (would be a no-op) and NEVER commits (would
    bypass triage). It always emits decision="triage".

Cascade emit on triage:
- decision: "triage"
- reason: explicit "fell_through_all_committing_stages"
- features: empty (no per-segment computation done; the triage decision
  is structural, not feature-based)
"""
from __future__ import annotations

from .stage_base import SegmentInput, Stage, StageDecision


class Stage99ResidualTriage(Stage):
    name = "stage_99_residual_triage"
    target_class = None  # always triages, never commits

    def decide(self, seg: SegmentInput) -> StageDecision:
        return StageDecision(
            decision="triage",
            reason=(
                "fell_through_all_committing_stages "
                "(no committing stage matched; this segment is outside "
                "the cascade's reasonable scoring competence and goes "
                "to human review)"
            ),
            features={"residual_triage": True},
        )
