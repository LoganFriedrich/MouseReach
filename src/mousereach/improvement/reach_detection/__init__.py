"""
Reach detection phase of the MouseReach Improvement Process.

Tracks improvements to the reach detector that identifies individual
reaching attempts within each segment.
"""
from mousereach.improvement.reach_detection.metrics import (
    Reach,
    ReachMatchResult,
    KinematicCompletenessResult,
    KinematicCompletenessAggregates,
    match_reaches,
    compute_reach_detection_metrics,
    compute_kinematic_completeness,
)

__all__ = [
    "Reach",
    "ReachMatchResult",
    "KinematicCompletenessResult",
    "KinematicCompletenessAggregates",
    "match_reaches",
    "compute_reach_detection_metrics",
    "compute_kinematic_completeness",
]
