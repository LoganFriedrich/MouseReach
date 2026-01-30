#!/usr/bin/env python3
"""
MouseReach Pipeline Index Module

Fast pipeline file indexing to eliminate slow folder scanning on network drives.

Usage:
    from mousereach.index import PipelineIndex

    index = PipelineIndex()
    index.load()  # Fast: single file read

    # Get files by stage
    videos = index.get_videos_in_stage("Processing")
    dlc_files = index.get_dlc_files()

    # Get cached metadata (no JSON parsing needed)
    metadata = index.get_video_metadata("20250704_CNT0101_P1")

CLI Commands:
    mousereach-index-rebuild   # Full rebuild of index
    mousereach-index-status    # Show index status
    mousereach-index-refresh   # Refresh specific folder
"""

from .index import PipelineIndex

__all__ = ["PipelineIndex"]
