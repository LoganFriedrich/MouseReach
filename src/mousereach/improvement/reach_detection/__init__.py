"""
Reach detection phase of the MouseReach Improvement Process.

Tracks improvements to the reach detector that identifies individual
reaching attempts within each segment.
"""
from mousereach.improvement.reach_detection.metrics import (
    Reach,
    ReachMatchResult,
    match_reaches,
    compute_reach_detection_metrics,
)

__all__ = [
    "Reach",
    "ReachMatchResult",
    "match_reaches",
    "compute_reach_detection_metrics",
]
