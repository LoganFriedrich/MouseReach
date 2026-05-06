"""
Reach assignment v1 -- per-reach permanent output table.

Production v1.0.0 is a cascade-trusted JOIN: takes the v8 reach detector
outputs and the v6 cascade outcome detector outputs and stamps a final
outcome label on every reach. Downstream kinematic analysis reads this
table directly and never re-derives outcomes per reach.

The module also retains the v1.0.0_dev learned classifier (features.py
+ train.py + overrides.py) used in development eval; it remains
importable for analysis but is NOT the production path.

Usage:
    from mousereach.assignment.v1 import assign_reaches_v1
    out = assign_reaches_v1(
        reaches=algo_reaches,
        segments_with_outcomes=cascade_segments,
        video_id="20250626_CNT0102_P4",
    )
    # out is the standard reach-assignments dict; write to JSON or
    # serialize as needed.
"""
from .assign import assign_reaches_v1

VERSION = "1.0.0"

__all__ = ["assign_reaches_v1", "VERSION"]
