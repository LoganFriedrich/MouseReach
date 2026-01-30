"""
MouseReach Step 2: Segmentation - Core Module
=============================================

This module detects trial boundaries (segment boundaries) in mouse reaching videos.
A "segment" is the period between two pellet presentations - typically ~1800 frames
at 30fps (1 minute per trial).

WHAT THIS MODULE DOES
---------------------
Input:  DeepLabCut tracking file (.h5) with 18 bodyparts tracked
Output: JSON file (*_segments.json) with 21 boundary frames

The algorithm detects when pellets appear/disappear to identify trial transitions.
A typical video has 20 trials = 21 boundaries (start of trial 1 through end of trial 20).

MODULE STRUCTURE
----------------
This package contains four submodules:

    segmenter_robust.py  - The detection algorithm
    batch.py             - Batch processing multiple videos
    triage.py            - Auto-sort results by confidence
    advance.py           - Move validated videos to next stage

WORKFLOW
--------
1. DETECT: Run segmentation algorithm on DLC files
   >>> from mousereach.segmentation.core import process_batch
   >>> process_batch(Path("Processing/"))

2. TRIAGE: Update validation_status in JSON based on confidence
   >>> from mousereach.segmentation.core import triage_results
   >>> triage_results(Path("Processing/"))

3. REVIEW: Human reviews boundaries in napari (see review_widget.py)
   >>> mousereach-segment-review  # CLI command

4. ADVANCE: Mark videos as validated (sets validation_status in JSON)
   >>> from mousereach.segmentation.core import advance_videos
   >>> advance_videos(Path("Processing/"))

CLI COMMANDS
------------
mousereach-segment          - Run segmentation on DLC files
mousereach-triage           - Triage results by confidence
mousereach-advance          - Move validated files to next stage
mousereach-segment-review   - Launch napari review tool

OUTPUT FORMAT
-------------
The *_segments.json file contains:
{
    "video_name": "20250624_CNT0115_P2",
    "boundaries": [100, 1937, 3774, ...],  # 21 frame numbers
    "n_boundaries": 21,
    "boundary_confidence": [0.95, 0.87, ...],  # Per-boundary confidence
    "overall_confidence": 0.91,
    "validation_status": "needs_review",  # or "validated", "auto_approved"
    "diagnostics": { ... }  # Algorithm debug info
}

CONFIDENCE THRESHOLDS
---------------------
- overall_confidence >= 0.85: Auto-approved (validation_status = "auto_approved")
- overall_confidence < 0.85:  Needs human review (validation_status = "needs_review")
- After human review:         validation_status = "validated"

All files remain in Processing/ folder. The dashboard filters by validation_status
to show which videos need review vs. which are approved.
"""

# =============================================================================
# ALGORITHM: segment_video_robust()
# =============================================================================
# The core detection algorithm. Analyzes pellet visibility patterns to find
# trial boundaries. See segmenter_robust.py for detailed algorithm docs.

try:
    from .segmenter_robust import (
        segment_video_robust,   # Main function: DLC file → boundary frames
        save_segmentation,      # Save results to JSON
        SEGMENTER_VERSION,      # Algorithm version string (e.g., "2.1.0")
        SegmentationDiagnostics # Dataclass with algorithm debug info
    )
except ImportError as e:
    # segmenter_robust.py contains proprietary algorithm - may not be present
    print(f"WARNING: Could not import segmenter_robust: {e}")
    print("Make sure you've copied your segmenter_robust.py to core/")
    SEGMENTER_VERSION = "MISSING"
    segment_video_robust = None
    save_segmentation = None
    SegmentationDiagnostics = None

# =============================================================================
# BATCH PROCESSING: process_batch()
# =============================================================================
# Run segmentation on multiple videos. Handles finding DLC files, running
# algorithm, saving results, and updating pipeline index.

from .batch import (
    find_dlc_files,   # Find all *DLC*.h5 files in a directory
    process_single,   # Process one video: DLC file → segments JSON
    process_batch     # Process all videos in a directory
)

# =============================================================================
# TRIAGE: triage_results()
# =============================================================================
# Set validation_status in JSON files based on confidence scores.
# High-confidence results get "auto_approved", low-confidence get "needs_review".
# All files stay in Processing/ folder - status tracked in JSON metadata.

from .triage import (
    get_associated_files,  # Find all files for a video (h5, mp4, json, etc.)
    classify_segments,     # Determine confidence level
    move_file_bundle,      # Move video + all associated files together
    triage_results,        # Main function: update validation_status for all videos
    DEST_AUTO_REVIEW,      # Legacy constant (not used in v2.3+)
    DEST_NEEDS_REVIEW,     # Legacy constant (not used in v2.3+)
    DEST_FAILED            # Legacy constant (not used in v2.3+)
)

# =============================================================================
# ADVANCE: advance_videos()
# =============================================================================
# Mark videos as validated by setting validation_status = "validated" in JSON.
# Files stay in Processing/ folder. Only updates videos that have been
# human-reviewed.

from .advance import (
    advance_videos,   # Mark videos as validated (updates JSON)
    get_username,     # Get current user for audit trail
    DEST_VALIDATED    # Legacy constant (not used in v2.3+)
)


__all__ = [
    # Algorithm
    'segment_video_robust', 'save_segmentation', 'SEGMENTER_VERSION', 'SegmentationDiagnostics',
    # Batch
    'find_dlc_files', 'process_single', 'process_batch',
    # Triage
    'get_associated_files', 'classify_segments', 'move_file_bundle', 'triage_results',
    'DEST_AUTO_REVIEW', 'DEST_NEEDS_REVIEW', 'DEST_FAILED',
    # Advance
    'advance_videos', 'get_username', 'DEST_VALIDATED',
]
