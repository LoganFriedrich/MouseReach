"""
ASPA2 Core Library
==================

Core algorithms for Automated Skilled Pellet Assessment.

Modules:
    segmenter: Video segmentation (find 21 boundaries)
    dlc_utils: Load and preprocess DLC files
    calibration: Ruler detection, px-to-mm conversion
    reach_detector: Detect reach events within segments
    scorer: Score reaches (success/fail/etc)
"""

__version__ = "0.1.0"

from .segmenter import segment_video, find_boundaries, SegmentationResult
from .dlc_utils import load_dlc_data
