"""
MouseReach Step 4: Pellet Outcomes - Core Module
================================================

This module classifies what happened to each pellet during a trial.
After each pellet presentation, the mouse may retrieve it, displace it, or leave it.

WHAT THIS MODULE DOES
---------------------
Input:  DeepLabCut tracking (.h5) + segments (*_segments.json) + reaches (*_reaches.json)
Output: JSON file (*_pellet_outcomes.json) with outcome classification per segment

The algorithm tracks pellet position relative to the pillar throughout each trial
and classifies based on visibility changes and displacement magnitude.

OUTCOME CATEGORIES
------------------
    retrieved      (R) - Mouse successfully grasped and ate the pellet
    displaced_sa   (D) - Pellet knocked but still in scoring area
    displaced_outside (O) - Pellet knocked outside scoring area
    untouched      (U) - Pellet not touched, remains on pillar
    no_pellet      (-) - No pellet was presented (rare)
    uncertain      (?) - Tracking too poor for confident classification

MODULE STRUCTURE
----------------
This package contains five submodules:

    pellet_outcome.py  - The classification algorithm (see detailed docs in file)
    geometry.py        - Coordinate transforms and pillar geometry
    batch.py           - Batch processing multiple videos
    triage.py          - Flag low-confidence classifications
    advance.py         - Move validated videos to next stage

WORKFLOW
--------
1. DETECT: Run outcome detection on videos with reach data
   >>> from mousereach.outcomes.core import process_batch
   >>> process_batch(Path("Processing/"))

2. TRIAGE: Flag low-confidence classifications in JSON
   >>> from mousereach.outcomes.core import triage_results
   >>> triage_results(Path("Processing/"))

3. REVIEW: Human reviews outcomes in napari (see review_widget.py)
   >>> mousereach-review-pellet-outcomes  # CLI command

4. ADVANCE: Mark videos as validated (sets validation_status in JSON)
   >>> from mousereach.outcomes.core import advance_videos
   >>> advance_videos(Path("Processing/"))

CLI COMMANDS
------------
mousereach-detect-outcomes         - Run outcome detection
mousereach-triage-outcomes         - Triage results by confidence
mousereach-advance-outcomes        - Move validated files to next stage
mousereach-review-pellet-outcomes  - Launch napari review tool

OUTPUT FORMAT
-------------
The *_pellet_outcomes.json file contains:
{
    "detector_version": "2.4.4",
    "video_name": "20250624_CNT0115_P2",
    "n_segments": 20,
    "segments": [
        {
            "segment_num": 1,
            "outcome": "displaced_sa",        # R, D, O, U, or uncertain
            "interaction_frame": 3774,         # Frame when pellet first touched
            "outcome_known_frame": 3842,       # Frame when outcome is clear
            "pellet_visible_start": true,
            "pellet_visible_end": true,
            "distance_from_pillar_start": 0.15,  # In ruler units
            "distance_from_pillar_end": 0.42,
            "causal_reach_id": 4,              # Which reach caused this outcome
            "confidence": 0.85,
            "human_verified": false,
            "flagged_for_review": true,
            "flag_reason": "Interaction timing uncertain"
        },
        ...
    ],
    "summary": {
        "total_segments": 20,
        "retrieved": 5,
        "displaced_sa": 8,
        "displaced_outside": 2,
        "untouched": 5,
        "flagged": 3,
        "mean_confidence": 0.89
    },
    "validation_status": "needs_review"  # or "validated"
}

CLASSIFICATION ALGORITHM (summary)
----------------------------------
See pellet_outcome.py for full documentation. Key rules:
1. Retrieved: Pellet disappears near pillar (mouse ate it)
2. Displaced: Pellet moves > threshold distance from pillar
3. Untouched: Pellet position unchanged throughout segment
4. Thresholds in ruler units (1 ruler unit = 9mm physical)
"""

# =============================================================================
# ALGORITHM: PelletOutcomeDetector, detect_pellet_outcomes()
# =============================================================================
# The core classification algorithm. Tracks pellet position relative to pillar
# and classifies based on visibility and displacement. See pellet_outcome.py
# for detailed algorithm documentation.

from .pellet_outcome import (
    PelletOutcomeDetector,    # Class encapsulating classification logic
    detect_pellet_outcomes,   # Main function: (DLC, segments, reaches) → outcomes
    VideoOutcomes,            # Dataclass: all outcomes for a video
    PelletOutcome,            # Dataclass: single segment outcome
    VERSION                   # Algorithm version string (e.g., "2.4.4")
)

# =============================================================================
# GEOMETRY: Coordinate transforms and pillar geometry
# =============================================================================
# Convert pixel coordinates to ruler units, calculate ideal pillar position
# from anchor points (SABL/SABR), and load input files.

from .geometry import (
    compute_segment_geometry,  # Calculate ruler scale for a segment
    compute_ideal_pillar,      # Calculate theoretical pillar position
    load_dlc,                  # Load DeepLabCut .h5 file → DataFrame
    load_segments,             # Load *_segments.json → dict
    SegmentGeometry            # Dataclass: calibration for one segment
)

# =============================================================================
# BATCH PROCESSING: process_batch()
# =============================================================================
# Run outcome detection on multiple videos. Finds DLC + segments + reaches
# file sets, runs algorithm, saves results, and updates pipeline index.

from .batch import (
    find_file_sets,   # Find (DLC .h5, segments .json, reaches .json) sets
    process_single,   # Process one video
    process_batch     # Process all videos in a directory
)

# =============================================================================
# TRIAGE: triage_results()
# =============================================================================
# Flag low-confidence classifications for human review. Checks for ambiguous
# outcomes, tracking issues, or unusual patterns.

from .triage import (
    load_all_results,  # Load all *_pellet_outcomes.json in directory
    check_anomalies,   # Check single video for issues
    triage_results     # Main function: flag problematic videos
)

# =============================================================================
# ADVANCE: advance_videos()
# =============================================================================
# Mark videos as validated by setting validation_status = "validated" in JSON.
# Files stay in Processing/ folder. Only updates videos that have been
# human-reviewed.

from .advance import (
    advance_videos,     # Mark videos as validated (updates JSON)
    mark_as_validated,  # Set validation_status in JSON file
    get_username        # Get current user for audit trail
)


__all__ = [
    # Algorithm
    'PelletOutcomeDetector', 'detect_pellet_outcomes', 'VideoOutcomes', 'PelletOutcome', 'VERSION',
    # Geometry
    'compute_segment_geometry', 'compute_ideal_pillar', 'load_dlc', 'load_segments', 'SegmentGeometry',
    # Batch
    'find_file_sets', 'process_single', 'process_batch',
    # Triage
    'load_all_results', 'check_anomalies', 'triage_results',
    # Advance
    'advance_videos', 'mark_as_validated', 'get_username',
]
