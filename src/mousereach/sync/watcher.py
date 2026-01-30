#!/usr/bin/env python3
"""
Pipeline File Watcher for MouseReach

Watches the Processing folder for new/modified _features.json files and
automatically syncs them to the reach_data table in the connectome database.

Uses watchdog for efficient filesystem monitoring.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Callable
from threading import Thread, Event

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    Observer = None
    FileSystemEventHandler = object

from .database import DatabaseSyncer, FEATURES_SUFFIX, parse_subject_id


logger = logging.getLogger(__name__)


class PipelineEventHandler(FileSystemEventHandler if HAS_WATCHDOG else object):
    """
    Handles filesystem events for _features.json files.

    Debounces rapid file changes (e.g., during write) and batches syncs.
    """

    def __init__(self, syncer: DatabaseSyncer, debounce_seconds: float = 2.0):
        super().__init__()
        self.syncer = syncer
        self.debounce_seconds = debounce_seconds

        self._pending: dict = {}  # path -> last_modified_time
        self._stop_event = Event()
        self._sync_thread: Optional[Thread] = None

    def _is_features_file(self, path: str) -> bool:
        """Check if path is a _features.json file."""
        return path.endswith(FEATURES_SUFFIX)

    def on_created(self, event):
        """Handle file creation."""
        if event.is_directory:
            return
        if self._is_features_file(event.src_path):
            self._queue_sync(event.src_path)

    def on_modified(self, event):
        """Handle file modification."""
        if event.is_directory:
            return
        if self._is_features_file(event.src_path):
            self._queue_sync(event.src_path)

    def _queue_sync(self, path: str):
        """Queue a file for sync with debouncing."""
        self._pending[path] = time.time()
        logger.debug(f"Queued for sync: {path}")

        if self._sync_thread is None or not self._sync_thread.is_alive():
            self._sync_thread = Thread(target=self._sync_loop, daemon=True)
            self._sync_thread.start()

    def _sync_loop(self):
        """Background thread that processes pending syncs."""
        while not self._stop_event.is_set():
            now = time.time()
            to_sync = []

            for path, last_mod in list(self._pending.items()):
                if now - last_mod >= self.debounce_seconds:
                    to_sync.append(path)

            for path in to_sync:
                del self._pending[path]
                try:
                    self._sync_file(path)
                except Exception as e:
                    logger.error(f"Failed to sync {path}: {e}")

            if not self._pending:
                break

            time.sleep(0.5)

    def _sync_file(self, path: str):
        """Sync a single features file to reach_data table."""
        file_path = Path(path)
        logger.info(f"Syncing: {file_path.name}")

        # Get subject ID
        video_name = file_path.stem.replace('_features', '')
        subject_id = parse_subject_id(video_name)

        if subject_id is None:
            logger.warning(f"Could not parse subject ID from: {file_path.name}")
            return

        try:
            n_reaches = self.syncer.sync_features_file(file_path, subject_id)
            logger.info(f"Synced: {file_path.name} -> {subject_id} ({n_reaches} reaches)")
        except Exception as e:
            logger.error(f"Sync failed for {file_path.name}: {e}")

    def stop(self):
        """Stop the sync loop."""
        self._stop_event.set()


class PipelineWatcher:
    """
    Watches Processing folder and syncs new _features.json files to database.

    Usage:
        watcher = PipelineWatcher()
        watcher.start()  # Non-blocking, runs in background
        ...
        watcher.stop()

    Or blocking:
        watcher = PipelineWatcher()
        watcher.run()  # Blocks until interrupted
    """

    def __init__(
        self,
        processing_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        debounce_seconds: float = 2.0
    ):
        if not HAS_WATCHDOG:
            raise ImportError(
                "watchdog package is required for file watching.\n"
                "Install with: pip install watchdog"
            )

        self.syncer = DatabaseSyncer(db_path=db_path, processing_path=processing_path)
        self.debounce_seconds = debounce_seconds

        if processing_path:
            self.processing_path = processing_path
        else:
            self.processing_path = self.syncer.processing_path

        if self.processing_path is None:
            raise ValueError("Processing path not configured. Run: mousereach-setup")

        self._observer: Optional[Observer] = None
        self._handler: Optional[PipelineEventHandler] = None

    def start(self):
        """Start watching (non-blocking)."""
        if self._observer is not None:
            return

        self._handler = PipelineEventHandler(
            self.syncer,
            debounce_seconds=self.debounce_seconds
        )

        self._observer = Observer()
        self._observer.schedule(
            self._handler,
            str(self.processing_path),
            recursive=False
        )
        self._observer.start()

        logger.info(f"Watching: {self.processing_path}")

    def stop(self):
        """Stop watching."""
        if self._handler:
            self._handler.stop()

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        logger.info("Watcher stopped")

    def run(self, on_sync: Optional[Callable] = None):
        """Run watcher (blocking)."""
        self.start()

        print(f"\nMouseReach Database Sync Watcher")
        print(f"================================")
        print(f"Watching: {self.processing_path}")
        print(f"Database: {self.syncer.db_path}")
        print(f"Syncing:  _features.json -> reach_data table")
        print(f"\nPress Ctrl+C to stop\n")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()

    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._observer is not None and self._observer.is_alive()


def start_watcher(
    processing_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    blocking: bool = True
) -> Optional[PipelineWatcher]:
    """
    Start the pipeline watcher.

    Args:
        processing_path: Path to Processing folder (default: from config)
        db_path: Path to connectome.db
        blocking: If True, blocks until interrupted. If False, returns watcher.

    Returns:
        PipelineWatcher if non-blocking, None if blocking
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    watcher = PipelineWatcher(
        processing_path=processing_path,
        db_path=db_path
    )

    if blocking:
        watcher.run()
        return None
    else:
        watcher.start()
        return watcher
