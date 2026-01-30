"""
MouseReach Archive Module - Move completed videos to NAS archive.

This module handles the final step of the pipeline: moving fully validated
videos from Processing/ to the NAS archive.

A video can only be archived if ALL validation statuses are "validated":
- seg_validation: validated
- reach_validation: validated
- outcome_validation: validated
"""

from .core import archive_video, get_archivable_videos

__all__ = ["archive_video", "get_archivable_videos"]
