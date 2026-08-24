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

# Keep in step with pyproject.toml -- tests/test_version.py fails if they drift.
# This is the number stamped into every video processing manifest, so it is the
# pipeline's provenance record of which code produced a result. It read 2.4.0
# while pyproject said 2.16.0-dev, so every manifest named a version that had
# not existed since February.
#
# Deliberately NOT importlib.metadata.version("mousereach"): a stale
# mousereach.egg-info sits in src/ in both the Y: and C: trees, left over from
# an old setuptools install, and it shadows the real dist-info -- metadata
# reports 2.3.0 there while pip reports 2.16.0.dev0. A literal cannot be fooled
# by that.
__version__ = "2.16.0-dev"

__author__ = "Logan Friedrich"

# Convenience imports for common config access
from mousereach.config import Paths, FilePatterns, get_video_id
