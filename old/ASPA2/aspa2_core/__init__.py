"""ASPA2 Core - Automated Skilled Pellet Assessment v2"""

from .segmenter_robust import (
    segment_video_robust,
    save_segmentation,
    print_diagnostics,
    SEGMENTER_VERSION,
    SEGMENTER_ALGORITHM,
)

__version__ = "2.1.0"
