"""
MouseReach - Automated Single Pellet reaching Analysis v2
=====================================================

A complete pipeline for analyzing mouse skilled reaching behavior videos.

Pipeline Steps:
    0. video_prep   - Crop multi-animal videos into single-animal clips
    1. dlc          - DeepLabCut pose estimation
    2. segmentation - Detect pellet presentation boundaries
    3. reach        - Detect individual reach attempts
    4. outcomes     - Classify pellet outcomes (retrieved/displaced/etc)
    5. kinematics   - Extract grasp kinematics features
    6. export       - Export analysis results

Usage:
    from mousereach.config import Paths, FilePatterns
    from mousereach import segmentation, reach, outcomes
"""

__version__ = "2.3.0"
__author__ = "Logan Friedrich"

# Convenience imports for common config access
from mousereach.config import Paths, FilePatterns, get_video_id
