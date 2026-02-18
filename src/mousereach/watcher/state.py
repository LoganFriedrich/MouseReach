"""
State management for the MouseReach watcher.

Provides high-level operations on top of WatcherDB:
- Discovery of new files
- Filename validation + quarantine
- File stability checking
- Priority-based work queue
- Stall detection
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

from mousereach.watcher.db import WatcherDB
from mousereach.watcher.validator import (
    validate_collage_filename, validate_single_filename, quarantine_file
)
from mousereach.watcher.transfer import check_file_stable_quick
from mousereach.config import WatcherConfig

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {'.mkv', '.avi', '.mp4', '.mov', '.wmv'}


@dataclass
class WorkItem:
    """A unit of work for the orchestrator."""
    type: str          # "collage_crop", "single_dlc", "single_process", "single_archive"
    id: str            # video_id or collage filename
    source_path: Path
    tray_type: str     # P, E, F
    priority: int      # Lower = higher priority (0 = highest)


class WatcherStateManager:
    """High-level state management for the watcher pipeline."""

    def __init__(self, db: WatcherDB, config: WatcherConfig):
        """
        Initialize state manager.

        Args:
            db: WatcherDB instance
            config: WatcherConfig instance
        """
        self.db = db
        self.config = config

    def discover_new_collages(self, scan_dir: Path) -> List[str]:
        """
        Scan directory for new collage videos and register them.

        Args:
            scan_dir: Directory to scan for videos

        Returns:
            List of newly registered filenames
        """
        if not scan_dir or not scan_dir.exists():
            logger.debug(f"Scan directory does not exist: {scan_dir}")
            return []

        newly_registered = []

        for file_path in scan_dir.iterdir():
            if not file_path.is_file():
                continue

            # Check extension
            if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            filename = file_path.name

            # Skip if already in database
            if self.db.collage_exists(filename):
                continue

            # Validate filename
            result = validate_collage_filename(filename)

            if result.valid:
                # Register in database
                collage_id = self.db.register_collage(
                    filename=filename,
                    source_path=str(file_path)
                )

                # Update with parsed metadata
                if result.parsed:
                    self.db.update_collage_state(
                        filename=filename,
                        new_state='validated',
                        date=result.parsed['date'],
                        animal_ids=','.join(result.parsed['animal_ids']),
                        tray_suffix=f"{result.parsed['tray_type']}{result.parsed['tray_run']}"
                    )
                else:
                    # No parsed metadata - stay in discovered state
                    pass

                newly_registered.append(filename)
                logger.info(f"Registered new collage: {filename}")

            else:
                # Invalid filename - quarantine
                quarantine_dir = self.config.get_quarantine_dir()
                quarantine_file(
                    file_path,
                    quarantine_dir,
                    reason=result.error
                )

                # Register as quarantined in DB
                collage_id = self.db.register_collage(
                    filename=filename,
                    source_path=str(file_path)
                )
                self.db.mark_quarantined(filename, reason=result.error, is_collage=True)

                logger.warning(f"Quarantined collage: {filename} - {result.error}")

        return newly_registered

    def discover_new_singles(self, scan_dir: Path) -> List[str]:
        """
        Scan directory for new single-animal videos and register them.

        Args:
            scan_dir: Directory to scan for videos

        Returns:
            List of newly registered video_ids
        """
        if not scan_dir or not scan_dir.exists():
            logger.debug(f"Scan directory does not exist: {scan_dir}")
            return []

        newly_registered = []

        for file_path in scan_dir.iterdir():
            if not file_path.is_file():
                continue

            # Only .mp4 for singles
            if file_path.suffix.lower() != '.mp4':
                continue

            filename = file_path.name

            # Generate video_id
            from mousereach.config import get_video_id
            video_id = get_video_id(filename)

            # Skip if already in database
            if self.db.video_exists(video_id):
                continue

            # Validate filename
            result = validate_single_filename(filename)

            if result.valid:
                # Register in database with parsed metadata
                metadata = {}
                if result.parsed:
                    metadata = {
                        'date': result.parsed['date'],
                        'animal_id': result.parsed['animal_id'],
                        'experiment': result.parsed['experiment'],
                        'cohort': result.parsed['cohort'],
                        'subject': result.parsed['subject'],
                        'tray_type': result.parsed['tray_type'],
                    }

                self.db.register_video(
                    video_id=video_id,
                    source_path=str(file_path),
                    **metadata
                )

                # Update to validated state
                self.db.update_state(video_id, 'validated', current_path=str(file_path))

                newly_registered.append(video_id)
                logger.info(f"Registered new single video: {video_id}")

            else:
                # Invalid filename - quarantine
                quarantine_dir = self.config.get_quarantine_dir()
                quarantine_file(
                    file_path,
                    quarantine_dir,
                    reason=result.error
                )

                # Register as quarantined in DB
                self.db.register_video(
                    video_id=video_id,
                    source_path=str(file_path)
                )
                self.db.mark_quarantined(video_id, reason=result.error, is_collage=False)

                logger.warning(f"Quarantined single video: {video_id} - {result.error}")

        return newly_registered

    def discover_dlc_staged(self, staging_dir: Path) -> List[str]:
        """
        Scan DLC staging directory for new DLC-complete video+h5 pairs.

        Used by the Processing Server to discover videos that the DLC PC
        has staged to the NAS after running DLC inference.

        Args:
            staging_dir: Directory to scan (Paths.DLC_STAGING)

        Returns:
            List of newly discovered video_ids
        """
        if not staging_dir or not staging_dir.exists():
            logger.debug(f"Staging directory does not exist: {staging_dir}")
            return []

        from mousereach.config import get_video_id

        newly_registered = []

        # Scan for DLC h5 files (the definitive output of DLC)
        for h5_path in staging_dir.glob("*DLC*.h5"):
            # Extract video_id from DLC filename (e.g. "20250704_CNT0101_P1DLC_..." -> "20250704_CNT0101_P1")
            video_id = h5_path.stem.split("DLC")[0].rstrip('_')

            # Skip if already in database
            if self.db.video_exists(video_id):
                continue

            # Find matching video file
            mp4_path = staging_dir / f"{video_id}.mp4"
            if not mp4_path.exists():
                logger.debug(f"No matching video for DLC output: {h5_path.name}")
                continue

            # Validate filename
            result = validate_single_filename(mp4_path.name)

            if result.valid:
                # Register with parsed metadata, directly at dlc_complete state
                metadata = {}
                if result.parsed:
                    metadata = {
                        'date': result.parsed['date'],
                        'animal_id': result.parsed['animal_id'],
                        'experiment': result.parsed['experiment'],
                        'cohort': result.parsed['cohort'],
                        'subject': result.parsed['subject'],
                        'tray_type': result.parsed['tray_type'],
                    }

                self.db.register_video(
                    video_id=video_id,
                    source_path=str(mp4_path),
                    current_path=str(mp4_path),
                    **metadata
                )

                # Advance directly to dlc_complete (DLC already done)
                self.db.update_state(video_id, 'validated')
                self.db.update_state(
                    video_id, 'dlc_complete',
                    dlc_output_path=str(h5_path),
                    current_path=str(mp4_path)
                )

                newly_registered.append(video_id)
                logger.info(f"Discovered staged DLC output: {video_id}")

            else:
                # Invalid filename - quarantine
                quarantine_dir = self.config.get_quarantine_dir()
                quarantine_file(
                    mp4_path,
                    quarantine_dir,
                    reason=result.error
                )

                self.db.register_video(
                    video_id=video_id,
                    source_path=str(mp4_path)
                )
                self.db.mark_quarantined(video_id, reason=result.error, is_collage=False)

                logger.warning(f"Quarantined staged video: {video_id} - {result.error}")

        return newly_registered

    def check_collage_stability(self, filename: str) -> bool:
        """
        Check if a collage file is stable (not actively being written).

        Updates database with new size/timestamps if size changed.

        Args:
            filename: Collage filename

        Returns:
            True if file is stable, False otherwise
        """
        collage = self.db.get_collage(filename)
        if not collage:
            logger.warning(f"Collage not found in database: {filename}")
            return False

        source_path = Path(collage['source_path'])
        if not source_path.exists():
            logger.warning(f"Collage file does not exist: {source_path}")
            return False

        # Get current size and last change time from DB
        recorded_size = collage['file_size']
        last_change_str = collage['last_size_change_at']

        # Convert timestamp to float
        last_change_time = None
        if last_change_str:
            try:
                dt = datetime.fromisoformat(last_change_str)
                last_change_time = dt.timestamp()
            except Exception:
                pass

        # Check stability
        is_stable, current_size, change_time = check_file_stable_quick(
            source_path,
            recorded_size,
            min_stable_seconds=self.config.stability_wait_seconds,
            last_change_time=last_change_time
        )

        # Update database with new size if changed
        if current_size != recorded_size:
            self.db.update_file_size(filename, current_size)

        # If stable, update state
        if is_stable and collage['state'] == 'validated':
            self.db.update_collage_state(filename, 'stable')
            logger.info(f"Collage now stable: {filename}")
            return True

        return is_stable

    def get_next_work_item(self) -> Optional[WorkItem]:
        """
        Get the next highest-priority work item from the queue.

        Priority order:
        1. Stable collages waiting to crop (priority=0)
        2. Singles waiting for DLC (priority=1)
        3. Singles waiting for post-DLC processing (priority=2)
        4. Singles ready for archive (priority=3)

        Returns:
            WorkItem or None if no work available
        """
        # Priority 0: Collages ready to crop
        collages = self.db.get_collages_in_state('stable')
        if collages:
            c = collages[0]
            return WorkItem(
                type='collage_crop',
                id=c['filename'],
                source_path=Path(c['source_path']),
                tray_type=c['tray_suffix'][0] if c.get('tray_suffix') else 'P',
                priority=0
            )

        # Priority 1: Singles waiting for DLC
        videos = self.db.get_videos_in_state('validated')
        if videos:
            v = videos[0]
            return WorkItem(
                type='single_dlc',
                id=v['video_id'],
                source_path=Path(v['current_path']) if v.get('current_path') else Path(v['source_path']),
                tray_type=v.get('tray_type', 'P'),
                priority=1
            )

        # Priority 2: Singles with DLC complete, waiting for processing
        videos = self.db.get_videos_in_state('dlc_complete')
        if videos:
            v = videos[0]
            return WorkItem(
                type='single_process',
                id=v['video_id'],
                source_path=Path(v['current_path']) if v.get('current_path') else Path(v['source_path']),
                tray_type=v.get('tray_type', 'P'),
                priority=2
            )

        # Priority 3: Singles processed and ready for archive
        videos = self.db.get_videos_in_state('processed')
        if videos:
            v = videos[0]
            return WorkItem(
                type='single_archive',
                id=v['video_id'],
                source_path=Path(v['current_path']) if v.get('current_path') else Path(v['source_path']),
                tray_type=v.get('tray_type', 'P'),
                priority=3
            )

        # No work available
        return None

    def get_stalled_items(self, timeout_minutes: int = 60) -> List[dict]:
        """
        Find items stuck in running states for longer than timeout.

        Args:
            timeout_minutes: How long before considering an item stalled

        Returns:
            List of dicts with id, state, time_stalled (minutes)
        """
        stalled = []
        cutoff = datetime.now() - timedelta(minutes=timeout_minutes)

        # Check collages in running states
        for state in ['cropping']:
            collages = self.db.get_collages_in_state(state)
            for c in collages:
                started_at_str = c.get('crop_started_at')
                if not started_at_str:
                    continue

                try:
                    started_at = datetime.fromisoformat(started_at_str)
                    if started_at < cutoff:
                        stalled_minutes = (datetime.now() - started_at).total_seconds() / 60
                        stalled.append({
                            'id': c['filename'],
                            'type': 'collage',
                            'state': state,
                            'time_stalled_minutes': stalled_minutes
                        })
                except Exception:
                    pass

        # Check videos in running states
        running_states = ['dlc_running', 'processing', 'archiving']
        timestamp_fields = {
            'dlc_running': 'dlc_started_at',
            'processing': 'processing_started_at',
            'archiving': 'archived_at'  # Note: archiving uses archived_at as start
        }

        for state in running_states:
            videos = self.db.get_videos_in_state(state)
            for v in videos:
                ts_field = timestamp_fields.get(state)
                if not ts_field:
                    continue

                started_at_str = v.get(ts_field)
                if not started_at_str:
                    continue

                try:
                    started_at = datetime.fromisoformat(started_at_str)
                    if started_at < cutoff:
                        stalled_minutes = (datetime.now() - started_at).total_seconds() / 60
                        stalled.append({
                            'id': v['video_id'],
                            'type': 'video',
                            'state': state,
                            'time_stalled_minutes': stalled_minutes
                        })
                except Exception:
                    pass

        return stalled
