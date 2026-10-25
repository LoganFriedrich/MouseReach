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
# EAGER: the light parts
# =============================================================================
# Review-file I/O (stdlib only). The watcher imports this package for these and
# for triage_status; they must stay cheap.

from .causal_review_io import (
    save_causal_review,
    update_corpus_index,
    build_segment_record,
    collect_provenance,
)

# =============================================================================
# LAZY: the GUI parts (PEP 562 -- loaded on first use)
# =============================================================================
# The widgets and their UI pieces load napari, Qt, vispy, pandas, scipy and
# more (about 2,600 modules and several seconds). They used to be imported
# here, so EVERY import of this package loaded them -- including the
# watcher's, which only needs causal_review_io / triage_status: a headless
# daemon loaded a whole GUI stack (measured 4.6 s, 2026-10-24) to call a path
# helper, and a widget that failed to import took the watcher's review paths
# down with it. Now a GUI name is imported the first time it is used;
# ``from mousereach.review import CausalReviewWidget`` works as before.

_LAZY = {
    # Base mixin with video loading, navigation, etc.; differences summary
    "AlgoGTReviewMixin": ".base",
    "DiffSummary": ".base",
    # Reusable UI components for review widgets
    "ComparisonPanel": ".comparison_panel",
    "ComparisonItem": ".comparison_panel",
    "create_boundary_comparison": ".comparison_panel",
    "create_reach_comparison": ".comparison_panel",
    "create_outcome_comparison": ".comparison_panel",
    # The review widgets
    "UnifiedReviewWidget": ".unified_widget",        # legacy tabbed widget (deprecated)
    "GroundTruthWidget": ".ground_truth_widget",     # current widget (both modes)
    "CausalReviewWidget": ".causal_review_widget",   # causal review tool
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    from importlib import import_module
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value      # later lookups skip this function
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


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
    "CausalReviewWidget",
    # Causal Review I/O
    "save_causal_review",
    "update_corpus_index",
    "build_segment_record",
    "collect_provenance",
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
