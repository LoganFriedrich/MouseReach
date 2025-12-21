"""
ASPA2 Step 1 Core - Segmentation Algorithm

This module contains the core segmentation algorithm that detects
the 21 boundary frames dividing a video into 22 pellet segments.
"""

from .segmenter_robust import (
    segment_video_robust,
    save_segmentation,
    print_diagnostics,
    SEGMENTER_VERSION,
    SEGMENTER_ALGORITHM,
)

__version__ = SEGMENTER_VERSION
