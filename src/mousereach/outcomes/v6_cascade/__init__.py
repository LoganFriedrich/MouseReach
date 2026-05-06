"""
Outcome detection v6 -- confidence-cascaded ensemble.

30 sequential stages testing physics-grounded predicates.
Each stage either COMMITS a class (if a high-confidence signal fires),
CONTINUES to the next stage, or TRIAGES (final stages only).
First stage to commit wins. Deterministic -- no model artifact required.

Design notes in `feature_philosophy_event_anchored_walking.md` and the
2026-04-30 conversation.

Production entry point::

    from mousereach.outcomes.v6_cascade import detect_outcomes_v6_cascade, VERSION
"""
VERSION = "6.0.0"

from .detector import detect_outcomes_v6_cascade  # noqa: E402, F401

__all__ = ["VERSION", "detect_outcomes_v6_cascade"]
