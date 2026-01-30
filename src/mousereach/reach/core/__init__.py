"""
MouseReach Step 3: Reach Detection - Core Module
=================================================

This module detects individual reaching movements within each trial segment.
A "reach" is when the mouse extends its paw through the slit toward the pellet.

WHAT THIS MODULE DOES
---------------------
Input:  DeepLabCut tracking (.h5) + segmentation boundaries (*_segments.json)
Output: JSON file (*_reaches.json) with reach timing for each segment

The algorithm identifies reach start, apex (max extension), and end frames
by tracking hand visibility while the nose is engaged at the slit.

MODULE STRUCTURE
----------------
This package contains five submodules:

    reach_detector.py  - The detection algorithm (see detailed docs in file)
    geometry.py        - Coordinate transforms and calibration
    batch.py           - Batch processing multiple videos
    triage.py          - Auto-sort results by anomaly detection
    advance.py         - Move validated videos to next stage

WORKFLOW
--------
1. DETECT: Run reach detection on segmented videos
   >>> from mousereach.reach.core import process_batch
   >>> process_batch(Path("Processing/"))

2. TRIAGE: Flag anomalous results in JSON (unusual reach counts, durations, etc.)
   >>> from mousereach.reach.core import triage_results
   >>> triage_results(Path("Processing/"))

3. REVIEW: Human reviews reaches in napari (see review_widget.py)
   >>> mousereach-review-reaches  # CLI command

4. ADVANCE: Mark videos as validated (sets validation_status in JSON)
   >>> from mousereach.reach.core import advance_videos
   >>> advance_videos(Path("Processing/"))

CLI COMMANDS
------------
mousereach-detect-reaches   - Run reach detection
mousereach-triage-reaches   - Triage results by anomaly detection
mousereach-advance-reaches  - Move validated files to next stage
mousereach-review-reaches   - Launch napari review tool

OUTPUT FORMAT
-------------
The *_reaches.json file contains:
{
    "detector_version": "3.3.0",
    "video_name": "20250624_CNT0115_P2",
    "total_frames": 54009,
    "n_segments": 21,
    "segments": [
        {
            "segment_num": 1,
            "start_frame": 3526,
            "end_frame": 3842,
            "reaches": [
                {
                    "reach_id": 1,
                    "start_frame": 3724,
                    "apex_frame": 3725,
                    "end_frame": 3725,
                    "duration_frames": 2,
                    "source": "algorithm",       # or "human_added"
                    "human_corrected": false,    # true if edited by human
                    "confidence": 0.85
                }
            ]
        },
        ...
    ],
    "summary": {
        "total_reaches": 42,
        "reaches_per_segment_mean": 2.0,
        ...
    },
    "validation_status": "needs_review"  # or "validated"
}

DETECTION ALGORITHM (summary)
-----------------------------
See reach_detector.py for full documentation. Key rules:
1. Nose must be engaged at slit (within threshold distance)
2. Reach starts when any hand point becomes visible (likelihood > 0.5)
3. Reach ends when hand disappears or retracts
4. Minimum duration filter removes tracking noise
5. Apex is the frame with maximum hand extension

GEOMETRY & CALIBRATION
----------------------
See geometry.py for coordinate system details:
- Pixel coordinates → ruler units (9mm ruler = 1.0 ruler unit)
- Pillar position calculated from SABL/SABR anchor points
- Box slit center from BOXL/BOXR midpoint
"""

# =============================================================================
# ALGORITHM: ReachDetector, detect_reaches()
# =============================================================================
# The core detection algorithm. Analyzes hand visibility during nose engagement
# to identify reaching movements. See reach_detector.py for 87 lines of detailed
# algorithm documentation including scientific description and known limitations.

from .reach_detector import (
    ReachDetector,     # Class encapsulating detection logic
    detect_reaches,    # Main function: (DLC, segments) → reaches
    VideoReaches,      # Dataclass: all reaches for a video
    SegmentReaches,    # Dataclass: reaches within one segment
    Reach,             # Dataclass: single reach event
    VERSION            # Algorithm version string (e.g., "3.3.0")
)

# =============================================================================
# GEOMETRY: Coordinate transforms and calibration
# =============================================================================
# Convert pixel coordinates to ruler units, calculate pillar position from
# anchor points, and load input files.

from .geometry import (
    compute_segment_geometry,  # Calculate ruler scale for a segment
    compute_ideal_pillar,      # Calculate theoretical pillar position
    get_boxr_reference,        # Get slit reference point from BOXL/BOXR
    load_dlc,                  # Load DeepLabCut .h5 file → DataFrame
    load_segments,             # Load *_segments.json → dict
    SegmentGeometry,           # Dataclass: calibration for one segment
    PHYSICAL_RULER_MM,         # Constant: 9.0 (ruler is 9mm)
    APEX_ANGLE_DEG             # Constant: reach apex angle for pillar calc
)

# =============================================================================
# BATCH PROCESSING: process_batch()
# =============================================================================
# Run reach detection on multiple videos. Finds DLC + segments file pairs,
# runs algorithm, saves results, and updates pipeline index.

from .batch import (
    find_file_pairs,  # Find (DLC .h5, segments .json) pairs
    process_single,   # Process one video
    process_batch     # Process all videos in a directory
)

# =============================================================================
# TRIAGE: triage_results()
# =============================================================================
# Flag anomalous results for human review. Checks for unusual patterns like
# too many/few reaches, very short/long durations, or tracking issues.

from .triage import (
    load_all_results,  # Load all *_reaches.json in directory
    check_anomalies,   # Check single video for anomalies
    triage_results     # Main function: flag anomalous videos
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
    # Main algorithm
    'ReachDetector', 'detect_reaches', 'VideoReaches', 'SegmentReaches', 'Reach', 'VERSION',
    # Geometry
    'compute_segment_geometry', 'compute_ideal_pillar', 'get_boxr_reference',
    'load_dlc', 'load_segments', 'SegmentGeometry', 'PHYSICAL_RULER_MM', 'APEX_ANGLE_DEG',
    # Batch
    'find_file_pairs', 'process_single', 'process_batch',
    # Triage
    'load_all_results', 'check_anomalies', 'triage_results',
    # Advance
    'advance_videos', 'mark_as_validated', 'get_username',
]
