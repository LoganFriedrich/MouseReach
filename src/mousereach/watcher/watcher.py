"""
Polling-based file watcher for the MouseReach pipeline.

Monitors NAS directories for new collage and single-animal videos.
Uses polling (not watchdog) because SMB/network drives are unreliable
with filesystem events.
"""

import time
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from mousereach.watcher.state import WatcherStateManager
from mousereach.config import Paths, WatcherConfig

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Results from one scan cycle."""
    new_collages: int = 0
    new_singles: int = 0
    quarantined: int = 0
    stable_ready: int = 0
    scan_time_ms: float = 0.0


class FileWatcher:
    """Polling-based file watcher for network directories."""

    def __init__(self, config: WatcherConfig, state: WatcherStateManager):
        """
        Initialize file watcher.

        Args:
            config: WatcherConfig instance
            state: WatcherStateManager instance
        """
        self.config = config
        self.state = state
        self._scan_count = 0

    def scan(self) -> ScanResult:
        """
        Perform one scan cycle.

        Scans configured directories for new files, checks stability,
        and updates state database.

        Returns:
            ScanResult with counts and timing
        """
        start_time = time.time()
        result = ScanResult()

        try:
            # Discover new collages
            if Paths.MULTI_ANIMAL_SOURCE and Paths.MULTI_ANIMAL_SOURCE.exists():
                new_collages = self.state.discover_new_collages(Paths.MULTI_ANIMAL_SOURCE)
                result.new_collages = len(new_collages)
            else:
                logger.debug("Multi-animal source directory not configured or does not exist")

            # Discover new singles
            if Paths.SINGLE_ANIMAL_OUTPUT and Paths.SINGLE_ANIMAL_OUTPUT.exists():
                new_singles = self.state.discover_new_singles(Paths.SINGLE_ANIMAL_OUTPUT)
                result.new_singles = len(new_singles)
            else:
                logger.debug("Single-animal output directory not configured or does not exist")

            # Check stability of all 'validated' collages
            validated_collages = self.state.db.get_collages_in_state('validated')
            stable_count = 0
            for collage in validated_collages:
                if self.state.check_collage_stability(collage['filename']):
                    stable_count += 1

            result.stable_ready = stable_count

            # Count quarantined items from this scan
            # (We don't track this per-scan in the current implementation,
            # so just report 0. The quarantine count is visible in the DB.)
            result.quarantined = 0

        except Exception as e:
            logger.error(f"Error during scan: {e}", exc_info=True)

        # Record timing
        elapsed_ms = (time.time() - start_time) * 1000
        result.scan_time_ms = elapsed_ms

        return result

    def run_forever(self, shutdown_event: threading.Event):
        """
        Run scan loop until shutdown event is set.

        Catches and logs exceptions per cycle to ensure the watcher
        never crashes - it keeps running until explicitly stopped.

        Args:
            shutdown_event: Threading event signaling shutdown
        """
        self._scan_count = 0

        logger.info("FileWatcher started")

        while not shutdown_event.is_set():
            try:
                self._scan_count += 1
                result = self.scan()

                # Log scan results
                if result.new_collages > 0 or result.new_singles > 0 or result.stable_ready > 0:
                    logger.info(
                        f"Scan #{self._scan_count}: "
                        f"new_collages={result.new_collages}, "
                        f"new_singles={result.new_singles}, "
                        f"stable_ready={result.stable_ready}, "
                        f"time={result.scan_time_ms:.1f}ms"
                    )
                else:
                    logger.debug(
                        f"Scan #{self._scan_count}: no new files, "
                        f"time={result.scan_time_ms:.1f}ms"
                    )

            except Exception as e:
                logger.error(f"Scan cycle #{self._scan_count} failed: {e}", exc_info=True)

            # Wait for next cycle or shutdown
            shutdown_event.wait(timeout=self.config.poll_interval_seconds)

        logger.info("FileWatcher stopped")
