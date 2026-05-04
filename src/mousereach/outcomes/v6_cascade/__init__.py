"""
Outcome detection v6 -- confidence-cascaded ensemble.

Stages, each focused on a specific discrimination with a small input
set. Stages either COMMIT a class (if a high-confidence signal fires)
or CONTINUE to the next stage. Only the final stage may TRIAGE.

Design notes in `feature_philosophy_event_anchored_walking.md` and the
2026-04-30 conversation.
"""
VERSION = "6.0.0_dev"
