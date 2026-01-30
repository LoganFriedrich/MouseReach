#!/usr/bin/env python3
"""
MouseReach Database Sync Module

Automatically syncs pipeline results to the central connectome database.

Usage:
    # One-time sync
    from mousereach.sync import DatabaseSyncer
    syncer = DatabaseSyncer()
    syncer.sync_all()

    # File watcher (continuous)
    from mousereach.sync import start_watcher
    start_watcher()  # Blocks, watching for new files

CLI:
    mousereach-sync         # One-time sync of all new files
    mousereach-sync-watch   # Start continuous watcher
    mousereach-sync-status  # Show sync status
"""

from .database import DatabaseSyncer
from .watcher import PipelineWatcher, start_watcher

__all__ = ["DatabaseSyncer", "PipelineWatcher", "start_watcher"]
