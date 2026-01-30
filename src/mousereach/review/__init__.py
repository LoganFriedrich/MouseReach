"""
MouseReach Review Tools - Two-Tool Architecture
================================================

This module provides tools for reviewing and correcting algorithm outputs.
There are TWO distinct tools with different purposes:

TWO-TOOL ARCHITECTURE
---------------------

1. REVIEW TOOL (GroundTruthWidget with review_mode=True)
   Purpose: Fix algorithm mistakes quickly
   Output:  Edits algorithm JSON files directly (*_segments.json, *_reaches.json, etc.)
   Use when: Algorithm made errors that need correction

2. GROUND TRUTH TOOL (GroundTruthWidget with review_mode=False)
   Purpose: Create comprehensive ground truth for algorithm evaluation
   Output:  Creates separate *_unified_ground_truth.json files
   Use when: Creating evaluation datasets for measuring algorithm accuracy

KEY DIFFERENCE
--------------
- Review Tool: Corrections go INTO the algorithm files (human_corrected=true)
- GT Tool: Corrections go into SEPARATE ground truth files

MODULE STRUCTURE
----------------

    ground_truth_widget.py   - Main widget (supports both modes via review_mode param)
    unified_widget.py      - Legacy tabbed widget (deprecated, kept for compatibility)
    unified_gt.py          - Ground truth data structures and I/O
    base.py                - AlgoGTReviewMixin base class
    comparison_panel.py    - UI component for comparing algo vs GT
    save_panel.py          - Save button UI with mode-aware behavior

ENTRY POINTS
------------
CLI commands (defined in pyproject.toml):
    mousereach-review-tool     - Launch Review Tool (edits algo files)
    mousereach-unified-review  - Launch GT Tool (creates GT files)

GUI access:
    mousereach                 - Both tools available as tabs in launcher

DATA FLOW
---------

Algorithm files (edited by Review Tool):
    *_segments.json        # Boundary frames with human_corrected flags
    *_reaches.json         # Reach timing with source/human_corrected fields
    *_pellet_outcomes.json # Outcomes with human_verified flag

Ground truth files (created by GT Tool):
    *_unified_ground_truth.json  # Complete verified annotations for evaluation

UNIFIED GT FILE FORMAT
----------------------
{
    "video_name": "20250624_CNT0115_P2",
    "type": "unified_ground_truth",
    "schema_version": "1.0",
    "created_by": "username",
    "created_at": "2026-01-16T12:00:00",
    "segmentation": {
        "boundaries": [
            {"index": 0, "frame": 100, "verified": true},
            {"index": 1, "frame": 1937, "verified": true, "original_frame": 1940},
            ...
        ]
    },
    "reaches": {
        "reaches": [
            {"reach_id": 1, "segment_num": 1, "start_frame": 150, "end_frame": 210,
             "start_verified": true, "end_verified": true, "source": "algorithm"},
            ...
        ]
    },
    "outcomes": {
        "segments": [
            {"segment_num": 1, "outcome": "retrieved", "interaction_frame": 180,
             "verified": true, "original_outcome": "displaced_sa"},
            ...
        ]
    },
    "completion_status": {
        "segments_complete": true,
        "reaches_complete": true,
        "outcomes_complete": true,
        "all_complete": true
    }
}

USAGE EXAMPLES
--------------

Launch Review Tool (fix algorithm mistakes):
    >>> from mousereach.review import GroundTruthWidget
    >>> widget = GroundTruthWidget(viewer, review_mode=True)

Launch GT Tool (create ground truth):
    >>> widget = GroundTruthWidget(viewer, review_mode=False)

Load ground truth data:
    >>> from mousereach.review import load_or_create_unified_gt
    >>> gt = load_or_create_unified_gt(video_path)

Save ground truth:
    >>> from mousereach.review import save_unified_gt
    >>> save_unified_gt(gt, video_path)
"""

# =============================================================================
# BASE CLASSES
# =============================================================================
# Mixin class providing common functionality for review widgets

from .base import (
    AlgoGTReviewMixin,  # Base mixin with video loading, navigation, etc.
    DiffSummary         # Dataclass summarizing differences between algo and GT
)

# =============================================================================
# UI COMPONENTS
# =============================================================================
# Reusable UI components for review widgets

from .comparison_panel import (
    ComparisonPanel,            # Panel showing algo vs GT comparison
    ComparisonItem,             # Single item in comparison list
    create_boundary_comparison, # Create comparison for boundary
    create_reach_comparison,    # Create comparison for reach
    create_outcome_comparison,  # Create comparison for outcome
)

# =============================================================================
# WIDGETS
# =============================================================================
# The main review widgets

from .unified_widget import UnifiedReviewWidget      # Legacy tabbed widget (deprecated)
from .ground_truth_widget import GroundTruthWidget # Current widget (both modes)

# =============================================================================
# GROUND TRUTH DATA STRUCTURES
# =============================================================================
# Data classes and I/O for unified ground truth files

from .unified_gt import (
    # Data classes
    UnifiedGroundTruth,  # Container for all GT data
    BoundaryGT,          # Single boundary ground truth
    ReachGT,             # Single reach ground truth
    OutcomeGT,           # Single outcome ground truth

    # I/O functions
    load_unified_gt,          # Load GT from file
    save_unified_gt,          # Save GT to file
    load_or_create_unified_gt, # Load existing or create from algo output
    get_unified_gt_path,      # Get path for GT file given video path
)


__all__ = [
    # Base
    "AlgoGTReviewMixin",
    "DiffSummary",
    # UI Components
    "ComparisonPanel",
    "ComparisonItem",
    "create_boundary_comparison",
    "create_reach_comparison",
    "create_outcome_comparison",
    # Widgets
    "UnifiedReviewWidget",
    "GroundTruthWidget",
    # Ground Truth
    "UnifiedGroundTruth",
    "BoundaryGT",
    "ReachGT",
    "OutcomeGT",
    "load_unified_gt",
    "save_unified_gt",
    "load_or_create_unified_gt",
    "get_unified_gt_path",
]
