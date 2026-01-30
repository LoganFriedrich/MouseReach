"""
MouseReach Performance Tracking Module

Automatically logs algorithm vs human comparison metrics during validation.
Provides CLI and GUI tools for viewing performance history and generating reports.

Usage:
    # Automatic logging (happens during validation)
    from mousereach.performance import PerformanceLogger
    logger = PerformanceLogger()
    logger.log_reach_detection(video_id, algo_data, human_data, validator)

    # Changelog tracking
    from mousereach.performance import get_changelog
    changelog = get_changelog()
    changelog.log_change(
        version="v3.4.0",
        change_summary="Removed negative extent filter",
        metrics_before={'recall': 0.15},
        metrics_after={'recall': 0.99},
    )

    # CLI commands
    mousereach-perf           # View performance summary
    mousereach-perf-eval      # Run batch evaluation
    mousereach-perf-report    # Generate scientific report
"""

from .logger import PerformanceLogger
from .metrics import (
    compute_segmentation_metrics,
    compute_reach_metrics,
    compute_outcome_metrics,
)
from .changelog import (
    ChangeLog,
    ChangeEntry,
    MetricDelta,
    get_changelog,
)

__all__ = [
    "PerformanceLogger",
    "compute_segmentation_metrics",
    "compute_reach_metrics",
    "compute_outcome_metrics",
    "ChangeLog",
    "ChangeEntry",
    "MetricDelta",
    "get_changelog",
]
