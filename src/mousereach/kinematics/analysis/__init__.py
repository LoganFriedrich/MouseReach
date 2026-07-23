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

# aggregator/visualizer were never added to this package; importing them
# unconditionally broke the whole analysis package (and the reach-export /
# real-kinematics CLIs that live in it). Import optionally so the package loads.
try:
    from .aggregator import FeatureAggregator  # type: ignore
except ImportError:
    FeatureAggregator = None  # type: ignore
try:
    from .visualizer import MultiLevelVisualizer  # type: ignore
except ImportError:
    MultiLevelVisualizer = None  # type: ignore

__all__ = [
    'DataLoader',
    'VideoMetadata',
    'SessionData',
    'FeatureAggregator',
    'MultiLevelVisualizer',
]
