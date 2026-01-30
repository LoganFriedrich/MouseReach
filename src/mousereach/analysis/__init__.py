"""
MouseReach Analysis Module

Interactive analysis dashboard for behavioral data exploration,
statistical comparisons, and publication-ready figure generation.

Usage:
    mousereach-analyze              # Launch interactive dashboard
    mousereach-analyze --export     # Export all data to CSV for external analysis

Designed to integrate with:
- Reach kinematics and outcomes from MouseReach pipeline
- External datasets (connectome, histology, etc.) via mouse ID matching
"""

from .data import (
    load_all_data,
    load_data_with_metadata,
    load_tracking_metadata,
    load_all_tracking_metadata,
    merge_with_metadata,
    ReachDataFrame,
    TIMEPOINT_MAPPING,
    # Surgery/mouse-level metadata
    load_surgery_metadata,
    load_all_surgery_metadata,
    # Session and segment context
    add_session_context,
    add_segment_context,
    # Unified "god view" data builder
    build_unified_reach_data,
)
from .stats import compare_groups, run_pca, compute_effect_size

# Plotting functions require seaborn - make import optional
try:
    from .plots import (
        plot_comparison,
        plot_pca_scores,
        plot_learning_curve,
        save_publication_figure
    )
except ImportError:
    # seaborn not installed - plotting functions unavailable
    plot_comparison = None
    plot_pca_scores = None
    plot_learning_curve = None
    save_publication_figure = None

__all__ = [
    # Data loading
    'load_all_data',
    'load_data_with_metadata',
    'load_tracking_metadata',
    'load_all_tracking_metadata',
    'merge_with_metadata',
    'ReachDataFrame',
    'TIMEPOINT_MAPPING',
    # Surgery/mouse-level metadata
    'load_surgery_metadata',
    'load_all_surgery_metadata',
    # Session and segment context
    'add_session_context',
    'add_segment_context',
    # Unified "god view" data builder
    'build_unified_reach_data',
    # Statistics
    'compare_groups',
    'run_pca',
    'compute_effect_size',
    # Plotting
    'plot_comparison',
    'plot_pca_scores',
    'plot_learning_curve',
    'save_publication_figure',
]
