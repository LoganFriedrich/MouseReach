"""
Tests for ASPA2 Segmenter
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from aspa2_core import segment_video, load_dlc_data


class TestSegmenter:
    """Test segmentation functionality."""
    
    def test_import(self):
        """Test that we can import the module."""
        from aspa2_core import segmenter
        assert hasattr(segmenter, 'segment_video')
        assert hasattr(segmenter, 'find_boundaries')
    
    # TODO: Add more tests with test data
    # def test_segment_video(self):
    #     result = segment_video('tests/test_data/sample.h5')
    #     assert len(result.boundaries) == 21


class TestDLCUtils:
    """Test DLC utility functions."""
    
    def test_import(self):
        """Test that we can import the module."""
        from aspa2_core import dlc_utils
        assert hasattr(dlc_utils, 'load_dlc_data')
        assert hasattr(dlc_utils, 'list_bodyparts')
