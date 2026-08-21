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

import os as _os

# OpenBLAS (numpy's math library) starts one thread per logical processor and
# sizes per-thread buffers on the stack. This machine has 104 of them, which
# overruns the stack and kills the process outright -- Windows exception
# 0xc00000fd, no Python traceback, no chance to catch it. It took down a review
# session mid-video on 2026-08-21. Nothing here is limited by how fast BLAS can
# multiply matrices, so cap the pools well below the core count.
#
# This has to happen before numpy is imported, which is why it lives at the top
# of the package rather than in the tool that needs it: importing anything under
# mousereach runs this file first. setdefault, so an explicit setting still wins.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, "8")

__version__ = "2.4.0"
__author__ = "Logan Friedrich"

# Convenience imports for common config access
from mousereach.config import Paths, FilePatterns, get_video_id
