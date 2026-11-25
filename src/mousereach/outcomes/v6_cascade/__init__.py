"""
Outcome detection v6 -- confidence-cascaded ensemble.

33 sequential stages (0-29, 6b, 98, 99) testing physics-grounded predicates.
Each stage either COMMITS a class (if a high-confidence signal fires),
CONTINUES to the next stage, or TRIAGES (final stages only).
First stage to commit wins. Deterministic -- no model artifact required.

Design: each stage walks the segment anchored on events (the pellet on
the pillar, a reach, the pellet at rest) rather than on fixed windows.

Production entry point::

    from mousereach.outcomes.v6_cascade import detect_outcomes_v6_cascade, VERSION
"""
VERSION = "6.1.0"

from .detector import detect_outcomes_v6_cascade  # noqa: E402, F401

__all__ = ["VERSION", "detect_outcomes_v6_cascade"]
