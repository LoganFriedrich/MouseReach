"""
Multi-level analysis framework for MouseReach feature data.

Provides tools for aggregating and analyzing data across:
- Sessions (single video)
- Mice (multiple sessions per mouse)
- Cohorts (groups of mice)
- Timepoints (longitudinal tracking)
- Experimental conditions
"""

from .data_loader import DataLoader, VideoMetadata, SessionData
from .aggregator import FeatureAggregator
from .visualizer import MultiLevelVisualizer

__all__ = [
    'DataLoader',
    'VideoMetadata',
    'SessionData',
    'FeatureAggregator',
    'MultiLevelVisualizer'
]
