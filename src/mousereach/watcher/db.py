"""
mousereach.watcher.db - SQLite state tracking for video pipeline.

Tracks every video through the watcher pipeline with full state management,
collage tracking, and processing audit logs.
"""

import sqlite3
import threading
import socket
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Valid state transitions
COLLAGE_TRANSITIONS = {
    'discovered': ['validated', 'quarantined'],
    'quarantined': ['validated'],  # After renaming
    'validated': ['stable'],
    'stable': ['cropping'],
    'cropping': ['cropped', 'failed'],
    'cropped': ['archived'],
    'failed': ['validated'],  # Retry
}

VIDEO_TRANSITIONS = {
    'discovered': ['validated', 'quarantined'],
    'quarantined': ['validated'],
    'validated': ['dlc_queued'],
    'dlc_queued': ['dlc_running'],
    'dlc_running': ['dlc_complete', 'dlc_queued', 'failed'],  # dlc_queued = re-queue on interrupt
    'dlc_complete': ['processing'],
    'processing': ['processed', 'failed'],
    'processed': ['archiving'],
    'archiving': ['archived', 'failed'],
    'failed': ['validated', 'dlc_queued', 'dlc_complete', 'processing', 'processed'],  # Retry from any prior state
}


class WatcherDB:
    """
    SQLite database for tracking video pipeline state.

    Thread-safe with automatic table creation and state validation.
    Detects network drives and uses DELETE journal mode instead of WAL.
    """

    def __init__(self, db_path: Path = None):
        """
        Initialize WatcherDB.

        Args:
            db_path: Path to SQLite database. If None, uses PROCESSING_ROOT/watcher.db
        """
        if db_path is None:
            # Import here to avoid circular dependency
            from mousereach.config import PROCESSING_ROOT
            db_path = PROCESSING_ROOT / "watcher.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._hostname = socket.gethostname()

        # Detect network drive
        self._is_network_drive = self._detect_network_drive(self.db_path)

        # Initialize database
        self._init_db()

        logger.info(f"WatcherDB initialized at {self.db_path} (network={self._is_network_drive})")

    def _detect_network_drive(self, path: Path) -> bool:
        """
        Detect if path is on a network drive.

        Args:
            path: Path to check

        Returns:
            True if network drive, False otherwise
        """
        path_str = str(path.resolve())

        # UNC path
        if path_str.startswith(r'\\'):
            return True

        # Check if drive letter is network mounted (Windows)
        if os.name == 'nt' and len(path_str) >= 2 and path_str[1] == ':':
            import win32file
            try:
                drive = path_str[:2] + '\\'
                drive_type = win32file.GetDriveType(drive)
                # DRIVE_REMOTE = 4
                return drive_type == 4
            except Exception:
                # If win32file not available or error, assume local
                return False

        # Unix/Linux network mount detection
        try:
            import subprocess
            result = subprocess.run(['df', str(path)], capture_output=True, text=True)
            if result.returncode == 0:
                # Check if mount point contains nfs, smb, cifs
                mount_info = result.stdout.lower()
                return any(proto in mount_info for proto in ['nfs', 'smb', 'cifs', '//'])
        except Exception:
            pass

        return False

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get database connection with appropriate settings.

        Returns:
            SQLite connection
        """
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None  # Autocommit mode
        )
        conn.row_factory = sqlite3.Row

        # Use DELETE journal mode for network drives (WAL not safe)
        if self._is_network_drive:
            conn.execute("PRAGMA journal_mode=DELETE")
        else:
            conn.execute("PRAGMA journal_mode=WAL")

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys=ON")

        return conn

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        with self._lock:
            conn = self._get_connection()
            try:
                # Videos table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS videos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_id TEXT UNIQUE NOT NULL,
                        source_path TEXT NOT NULL,
                        collage_id TEXT,
                        date TEXT,
                        animal_id TEXT,
                        experiment TEXT,
                        cohort TEXT,
                        subject TEXT,
                        tray_type TEXT,
                        tray_position INTEGER,
                        state TEXT NOT NULL DEFAULT 'discovered',
                        discovered_at TEXT NOT NULL,
                        validated_at TEXT,
                        crop_started_at TEXT,
                        crop_completed_at TEXT,
                        dlc_started_at TEXT,
                        dlc_completed_at TEXT,
                        processing_started_at TEXT,
                        processing_completed_at TEXT,
                        archived_at TEXT,
                        error_message TEXT,
                        error_count INTEGER DEFAULT 0,
                        last_error_at TEXT,
                        claimed_by TEXT,
                        current_path TEXT,
                        dlc_output_path TEXT,
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                """)

                # Collages table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS collages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT UNIQUE NOT NULL,
                        source_path TEXT NOT NULL,
                        state TEXT NOT NULL DEFAULT 'discovered',
                        date TEXT,
                        animal_ids TEXT,
                        tray_suffix TEXT,
                        file_size INTEGER,
                        first_seen_at TEXT NOT NULL,
                        last_size_change_at TEXT,
                        stable_since TEXT,
                        validation_error TEXT,
                        crop_started_at TEXT,
                        crop_completed_at TEXT,
                        videos_created INTEGER DEFAULT 0,
                        videos_skipped INTEGER DEFAULT 0,
                        archived_at TEXT,
                        archive_path TEXT,
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                """)

                # Processing log table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS processing_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_id TEXT NOT NULL,
                        step TEXT NOT NULL,
                        status TEXT NOT NULL,
                        message TEXT,
                        duration_seconds REAL,
                        machine TEXT,
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                """)

                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_state ON videos(state)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_collage ON videos(collage_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_collages_state ON collages(state)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_log_video ON processing_log(video_id)")

                conn.commit()

            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                raise
            finally:
                conn.close()

    def _validate_transition(self, current_state: str, new_state: str,
                           transitions: Dict[str, List[str]], entity_type: str):
        """
        Validate state transition is legal.

        Args:
            current_state: Current state
            new_state: Requested new state
            transitions: Valid transitions dict
            entity_type: "video" or "collage" (for error message)

        Raises:
            ValueError: If transition is invalid
        """
        if current_state == new_state:
            return  # No-op transition is fine

        valid_next = transitions.get(current_state, [])
        if new_state not in valid_next:
            raise ValueError(
                f"Invalid {entity_type} state transition: {current_state} to {new_state}. "
                f"Valid transitions from {current_state}: {', '.join(valid_next)}"
            )

    def _now(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()

    def register_collage(self, filename: str, source_path: str) -> int:
        """
        Register new collage file.

        Args:
            filename: Collage filename
            source_path: Full path to source file

        Returns:
            Collage ID (database row id)
        """
        with self._lock:
            conn = self._get_connection()
            try:
                # Check if exists
                existing = conn.execute(
                    "SELECT id FROM collages WHERE filename = ?",
                    (filename,)
                ).fetchone()

                if existing:
                    return existing['id']

                # Insert new
                cursor = conn.execute("""
                    INSERT INTO collages (filename, source_path, first_seen_at)
                    VALUES (?, ?, ?)
                """, (filename, source_path, self._now()))

                conn.commit()
                return cursor.lastrowid

            except Exception as e:
                logger.error(f"Failed to register collage {filename}: {e}")
                raise
            finally:
                conn.close()

    def register_video(self, video_id: str, source_path: str,
                      collage_id: str = None, **metadata) -> str:
        """
        Register new video.

        Args:
            video_id: Unique video identifier
            source_path: Full path to source file
            collage_id: Optional collage filename this came from
            **metadata: Additional fields (date, animal_id, experiment, cohort,
                       subject, tray_type, tray_position, etc.)

        Returns:
            video_id (unchanged)
        """
        with self._lock:
            conn = self._get_connection()
            try:
                # Check if exists
                existing = conn.execute(
                    "SELECT video_id FROM videos WHERE video_id = ?",
                    (video_id,)
                ).fetchone()

                if existing:
                    logger.debug(f"Video {video_id} already registered")
                    return video_id

                # Build insert
                fields = ['video_id', 'source_path', 'collage_id', 'discovered_at']
                values = [video_id, source_path, collage_id, self._now()]

                # Add metadata fields
                for key, value in metadata.items():
                    if value is not None:
                        fields.append(key)
                        values.append(value)

                placeholders = ', '.join(['?'] * len(values))
                field_list = ', '.join(fields)

                conn.execute(
                    f"INSERT INTO videos ({field_list}) VALUES ({placeholders})",
                    values
                )

                conn.commit()
                logger.info(f"Registered video {video_id}")
                return video_id

            except Exception as e:
                logger.error(f"Failed to register video {video_id}: {e}")
                raise
            finally:
                conn.close()

    def update_state(self, video_id: str, new_state: str, **kwargs):
        """
        Update video state with validation.

        Args:
            video_id: Video identifier
            new_state: New state to transition to
            **kwargs: Additional fields to update (e.g., dlc_started_at, current_path)

        Raises:
            ValueError: If transition is invalid or video not found
        """
        with self._lock:
            conn = self._get_connection()
            try:
                # Get current state
                row = conn.execute(
                    "SELECT state FROM videos WHERE video_id = ?",
                    (video_id,)
                ).fetchone()

                if not row:
                    raise ValueError(f"Video {video_id} not found")

                current_state = row['state']

                # Validate transition
                self._validate_transition(current_state, new_state, VIDEO_TRANSITIONS, "video")

                # Build update
                fields = ['state = ?', 'updated_at = ?']
                values = [new_state, self._now()]

                # Add timestamp for state-specific fields
                state_timestamp_map = {
                    'validated': 'validated_at',
                    'dlc_running': 'dlc_started_at',
                    'dlc_complete': 'dlc_completed_at',
                    'processing': 'processing_started_at',
                    'processed': 'processing_completed_at',
                    'archived': 'archived_at',
                }

                if new_state in state_timestamp_map:
                    timestamp_field = state_timestamp_map[new_state]
                    fields.append(f'{timestamp_field} = ?')
                    values.append(self._now())

                # Add additional fields
                for key, value in kwargs.items():
                    fields.append(f'{key} = ?')
                    values.append(value)

                # Execute update
                values.append(video_id)
                conn.execute(
                    f"UPDATE videos SET {', '.join(fields)} WHERE video_id = ?",
                    values
                )

                conn.commit()
                logger.info(f"Updated video {video_id}: {current_state} -> {new_state}")

            except Exception as e:
                logger.error(f"Failed to update video {video_id}: {e}")
                raise
            finally:
                conn.close()

    def force_state(self, video_id: str, new_state: str, **kwargs):
        """
        Set video state directly, bypassing transition validation.

        Used for intake scenarios where videos arrive at an intermediate state
        (e.g., DLC already completed by another machine).

        Args:
            video_id: Video identifier
            new_state: State to set
            **kwargs: Additional fields to update
        """
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT state FROM videos WHERE video_id = ?",
                    (video_id,)
                ).fetchone()

                if not row:
                    raise ValueError(f"Video {video_id} not found")

                current_state = row['state']

                # Build update (same as update_state but no validation)
                fields = ['state = ?', 'updated_at = ?']
                values = [new_state, self._now()]

                # Add timestamp for state-specific fields
                state_timestamp_map = {
                    'validated': 'validated_at',
                    'dlc_running': 'dlc_started_at',
                    'dlc_complete': 'dlc_completed_at',
                    'processing': 'processing_started_at',
                    'processed': 'processing_completed_at',
                    'archived': 'archived_at',
                }

                if new_state in state_timestamp_map:
                    timestamp_field = state_timestamp_map[new_state]
                    fields.append(f'{timestamp_field} = ?')
                    values.append(self._now())

                for key, value in kwargs.items():
                    fields.append(f'{key} = ?')
                    values.append(value)

                values.append(video_id)
                conn.execute(
                    f"UPDATE videos SET {', '.join(fields)} WHERE video_id = ?",
                    values
                )

                conn.commit()
                logger.info(f"Force-set video {video_id}: {current_state} -> {new_state}")

            except Exception as e:
                logger.error(f"Failed to force-set video {video_id}: {e}")
                raise
            finally:
                conn.close()

    def update_collage_state(self, filename: str, new_state: str, **kwargs):
        """
        Update collage state with validation.

        Args:
            filename: Collage filename
            new_state: New state to transition to
            **kwargs: Additional fields to update

        Raises:
            ValueError: If transition is invalid or collage not found
        """
        with self._lock:
            conn = self._get_connection()
            try:
                # Get current state
                row = conn.execute(
                    "SELECT state FROM collages WHERE filename = ?",
                    (filename,)
                ).fetchone()

                if not row:
                    raise ValueError(f"Collage {filename} not found")

                current_state = row['state']

                # Validate transition
                self._validate_transition(current_state, new_state, COLLAGE_TRANSITIONS, "collage")

                # Build update
                fields = ['state = ?', 'updated_at = ?']
                values = [new_state, self._now()]

                # Add timestamp for state-specific fields
                state_timestamp_map = {
                    'validated': 'validated_at',
                    'stable': 'stable_since',
                    'cropping': 'crop_started_at',
                    'cropped': 'crop_completed_at',
                    'archived': 'archived_at',
                }

                if new_state in state_timestamp_map:
                    timestamp_field = state_timestamp_map[new_state]
                    if timestamp_field != 'validated_at':  # validated_at might be in kwargs
                        fields.append(f'{timestamp_field} = ?')
                        values.append(self._now())

                # Add additional fields
                for key, value in kwargs.items():
                    fields.append(f'{key} = ?')
                    values.append(value)

                # Execute update
                values.append(filename)
                conn.execute(
                    f"UPDATE collages SET {', '.join(fields)} WHERE filename = ?",
                    values
                )

                conn.commit()
                logger.info(f"Updated collage {filename}: {current_state} -> {new_state}")

            except Exception as e:
                logger.error(f"Failed to update collage {filename}: {e}")
                raise
            finally:
                conn.close()

    def get_videos_in_state(self, state: str) -> List[dict]:
        """
        Get all videos in given state.

        Args:
            state: State to filter by

        Returns:
            List of video dicts
        """
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    "SELECT * FROM videos WHERE state = ? ORDER BY created_at DESC",
                    (state,)
                ).fetchall()

                return [dict(row) for row in rows]

            finally:
                conn.close()

    def get_collages_in_state(self, state: str) -> List[dict]:
        """
        Get all collages in given state.

        Args:
            state: State to filter by

        Returns:
            List of collage dicts
        """
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    "SELECT * FROM collages WHERE state = ? ORDER BY created_at DESC",
                    (state,)
                ).fetchall()

                return [dict(row) for row in rows]

            finally:
                conn.close()

    def log_step(self, video_id: str, step: str, status: str,
                message: str = None, duration: float = None):
        """
        Log processing step to audit trail.

        Args:
            video_id: Video identifier
            step: Processing step name (e.g., "crop", "dlc", "segment")
            status: Status (e.g., "started", "completed", "failed")
            message: Optional message
            duration: Optional duration in seconds
        """
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                    INSERT INTO processing_log
                    (video_id, step, status, message, duration_seconds, machine)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (video_id, step, status, message, duration, self._hostname))

                conn.commit()

            except Exception as e:
                logger.error(f"Failed to log step for {video_id}: {e}")
                # Don't raise - logging failure shouldn't break pipeline
            finally:
                conn.close()

    def get_video(self, video_id: str) -> Optional[dict]:
        """
        Get single video by ID.

        Args:
            video_id: Video identifier

        Returns:
            Video dict or None if not found
        """
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT * FROM videos WHERE video_id = ?",
                    (video_id,)
                ).fetchone()

                return dict(row) if row else None

            finally:
                conn.close()

    def get_collage(self, filename: str) -> Optional[dict]:
        """
        Get single collage by filename.

        Args:
            filename: Collage filename

        Returns:
            Collage dict or None if not found
        """
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT * FROM collages WHERE filename = ?",
                    (filename,)
                ).fetchone()

                return dict(row) if row else None

            finally:
                conn.close()

    def get_pipeline_summary(self) -> dict:
        """
        Get pipeline summary with counts by state.

        Returns:
            Dict with video and collage statistics
        """
        with self._lock:
            conn = self._get_connection()
            try:
                # Video stats
                video_rows = conn.execute("""
                    SELECT state, COUNT(*) as count
                    FROM videos
                    GROUP BY state
                """).fetchall()

                video_stats = {row['state']: row['count'] for row in video_rows}

                # Collage stats
                collage_rows = conn.execute("""
                    SELECT state, COUNT(*) as count
                    FROM collages
                    GROUP BY state
                """).fetchall()

                collage_stats = {row['state']: row['count'] for row in collage_rows}

                # Failed/quarantined counts
                failed_videos = conn.execute(
                    "SELECT COUNT(*) as count FROM videos WHERE state = 'failed'"
                ).fetchone()['count']

                quarantined_videos = conn.execute(
                    "SELECT COUNT(*) as count FROM videos WHERE state = 'quarantined'"
                ).fetchone()['count']

                quarantined_collages = conn.execute(
                    "SELECT COUNT(*) as count FROM collages WHERE state = 'quarantined'"
                ).fetchone()['count']

                # Total counts
                total_videos = conn.execute("SELECT COUNT(*) as count FROM videos").fetchone()['count']
                total_collages = conn.execute("SELECT COUNT(*) as count FROM collages").fetchone()['count']

                return {
                    'videos': {
                        'total': total_videos,
                        'by_state': video_stats,
                        'failed': failed_videos,
                        'quarantined': quarantined_videos,
                    },
                    'collages': {
                        'total': total_collages,
                        'by_state': collage_stats,
                        'quarantined': quarantined_collages,
                    }
                }

            finally:
                conn.close()

    def mark_quarantined(self, identifier: str, reason: str, is_collage: bool = False):
        """
        Mark video or collage as quarantined.

        Args:
            identifier: video_id or filename
            reason: Quarantine reason
            is_collage: True if identifier is a collage filename
        """
        if is_collage:
            self.update_collage_state(identifier, 'quarantined', validation_error=reason)
        else:
            self.update_state(identifier, 'quarantined', error_message=reason)

    def mark_failed(self, video_id: str, error_message: str):
        """
        Mark video as failed with error tracking.

        Args:
            video_id: Video identifier
            error_message: Error description
        """
        with self._lock:
            conn = self._get_connection()
            try:
                # Get current error count
                row = conn.execute(
                    "SELECT error_count FROM videos WHERE video_id = ?",
                    (video_id,)
                ).fetchone()

                if not row:
                    raise ValueError(f"Video {video_id} not found")

                error_count = (row['error_count'] or 0) + 1

                # Update to failed state
                conn.execute("""
                    UPDATE videos
                    SET state = 'failed',
                        error_message = ?,
                        error_count = ?,
                        last_error_at = ?,
                        updated_at = ?
                    WHERE video_id = ?
                """, (error_message, error_count, self._now(), self._now(), video_id))

                conn.commit()
                logger.warning(f"Marked video {video_id} as failed (error #{error_count}): {error_message}")

            except Exception as e:
                logger.error(f"Failed to mark {video_id} as failed: {e}")
                raise
            finally:
                conn.close()

    def reset_failed(self, video_id: str, to_state: str = None):
        """
        Reset a failed video to retry from a previous state.

        Args:
            video_id: Video identifier
            to_state: State to reset to. If None, infers from context:
                     - If has DLC output: reset to 'dlc_complete'
                     - Else if validated: reset to 'validated'
                     - Else: reset to 'discovered'
        """
        with self._lock:
            conn = self._get_connection()
            try:
                # Get video info
                row = conn.execute(
                    "SELECT * FROM videos WHERE video_id = ?",
                    (video_id,)
                ).fetchone()

                if not row:
                    raise ValueError(f"Video {video_id} not found")

                if row['state'] != 'failed':
                    logger.warning(f"Video {video_id} is not in failed state (current: {row['state']})")
                    return

                # Infer state if not provided
                if to_state is None:
                    if row['dlc_output_path']:
                        to_state = 'dlc_complete'
                    elif row['validated_at']:
                        to_state = 'validated'
                    else:
                        to_state = 'discovered'

                # Validate target state exists in transitions
                valid_retry_states = VIDEO_TRANSITIONS.get('failed', [])
                if to_state not in valid_retry_states:
                    raise ValueError(
                        f"Cannot reset to {to_state}. Valid retry states: {', '.join(valid_retry_states)}"
                    )

                # Reset
                conn.execute("""
                    UPDATE videos
                    SET state = ?,
                        error_message = NULL,
                        updated_at = ?
                    WHERE video_id = ?
                """, (to_state, self._now(), video_id))

                conn.commit()
                logger.info(f"Reset video {video_id} from failed to {to_state}")

            except Exception as e:
                logger.error(f"Failed to reset video {video_id}: {e}")
                raise
            finally:
                conn.close()

    def update_file_size(self, filename: str, size: int):
        """
        Update collage file size for stability tracking.

        Args:
            filename: Collage filename
            size: File size in bytes
        """
        with self._lock:
            conn = self._get_connection()
            try:
                # Get current size
                row = conn.execute(
                    "SELECT file_size FROM collages WHERE filename = ?",
                    (filename,)
                ).fetchone()

                if not row:
                    logger.warning(f"Collage {filename} not found for size update")
                    return

                current_size = row['file_size']

                # Update if changed
                if current_size != size:
                    conn.execute("""
                        UPDATE collages
                        SET file_size = ?,
                            last_size_change_at = ?,
                            updated_at = ?
                        WHERE filename = ?
                    """, (size, self._now(), self._now(), filename))

                    conn.commit()
                    logger.debug(f"Updated size for {filename}: {current_size} -> {size}")

            except Exception as e:
                logger.error(f"Failed to update file size for {filename}: {e}")
                # Don't raise - size tracking failure shouldn't break pipeline
            finally:
                conn.close()

    def get_recent_log(self, limit: int = 20) -> List[dict]:
        """
        Get recent processing log entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of log entry dicts
        """
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute("""
                    SELECT * FROM processing_log
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,)).fetchall()

                return [dict(row) for row in rows]

            finally:
                conn.close()

    def collage_exists(self, filename: str) -> bool:
        """
        Check if collage is already registered.

        Args:
            filename: Collage filename

        Returns:
            True if exists, False otherwise
        """
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT 1 FROM collages WHERE filename = ?",
                    (filename,)
                ).fetchone()

                return row is not None

            finally:
                conn.close()

    def video_exists(self, video_id: str) -> bool:
        """
        Check if video is already registered.

        Args:
            video_id: Video identifier

        Returns:
            True if exists, False otherwise
        """
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT 1 FROM videos WHERE video_id = ?",
                    (video_id,)
                ).fetchone()

                return row is not None

            finally:
                conn.close()
