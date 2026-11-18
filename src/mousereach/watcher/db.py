"""
mousereach.watcher.db - SQLite state tracking for video pipeline.

Tracks every video through the watcher pipeline with full state management,
collage tracking, and processing audit logs.
"""

import contextlib
import sqlite3
import threading
import socket
import logging
import os
import time
import functools
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Valid state transitions
# 'recording_issue' (watcher/recording_issue.py): the recording itself failed
# (empty, a fraction of a second, unreadable). It was set aside in
# Processing/Review/_Problematic with a reason file and is never cut. Reached
# from every state before the cut; it leaves only when a person moves the
# recording back into the intake AND takes its reason file away
# ('recording_issue' -> 'validated'; the check then runs again).
COLLAGE_TRANSITIONS = {
    'discovered': ['validated', 'quarantined', 'recording_issue'],
    'quarantined': ['validated', 'recording_issue'],  # After renaming
    'validated': ['stable', 'recording_issue'],
    'stable': ['cropping', 'recording_issue'],
    'cropping': ['cropped', 'failed', 'recording_issue'],
    'cropped': ['archived'],
    'failed': ['validated', 'recording_issue'],  # Retry
    'recording_issue': ['validated'],
}

VIDEO_TRANSITIONS = {
    'discovered': ['validated', 'quarantined'],
    'quarantined': ['validated'],
    'validated': ['dlc_queued', 'unresolvable'],
    'dlc_queued': ['dlc_running', 'unresolvable'],
    'dlc_running': ['dlc_complete', 'dlc_queued', 'failed', 'unresolvable'],  # dlc_queued = re-queue on interrupt
    # 'archived' here means "this NODE is done with it", not "fully analyzed" --
    # the same idiom the collage table uses for 'cropped': ['archived']. A GPU node
    # running with also_process=false finishes DLC, stages the pose to the NAS
    # (_stage_to_nas) and is done; the MouseReach algorithms run on whichever node
    # claims it next. Without this, every GPU-node hand-off raised on the transition
    # and the video was marked failed even though DLC had succeeded.
    'dlc_complete': ['processing', 'archived', 'unresolvable'],
    'processing': ['processed', 'failed', 'triage', 'deep_review', 'unresolvable'],  # triage/deep_review = human-review holds
    'processed': ['archiving', 'unresolvable'],
    # 'archiving' is written BEFORE the files move (orchestrator archive
    # steps). 'processed' = that attempt failed with the files still in
    # Processing (and startup reclaim of an interrupted filing): the archive
    # bucket, which selects only 'processed', tries again after its backoff.
    'archiving': ['archived', 'processed', 'failed', 'unresolvable'],
    'archived': ['outdated', 'crystallized', 'unresolvable'],  # Version-aware reprocessing
    # 'archived' = un-mark: the row compares current again (e.g. after a
    # corrective or compatible-version declaration). Without it 'outdated' was
    # a one-way door and a declaration accident could never be undone.
    'outdated': ['dlc_queued', 'processing', 'failed', 'archived', 'unresolvable'],  # Re-enters pipeline
    'crystallized': [],  # Locked against reprocessing (use force_state to unlock)
    # --- Human-review holds: the video is held OUT of the archive + central database
    # until a human clears it. Kinematics NEVER run on a held video. ---
    # 'triage': the algo left per-element questions (which reach / what outcome).
    #   The triage review tool answers them. When EVERY triaged element is
    #   addressed -> 'processing', resuming just after the step that sent it
    #   (watcher/resume_point.py; every later check runs again and the gate
    #   re-checks). If the reviewer cannot resolve it -> escalate to 'deep_review'.
    'triage': ['outdated', 'processing', 'deep_review', 'failed'],
    # 'deep_review': segmentation FAILED, or a reviewer escalated a triaged video.
    #   The causal/GT deep tools clear the flag -> the video is re-injected just
    #   after segmentation (a person's cuts are kept; everything after them is
    #   remade, watcher/resume_point.py): -> 'outdated'/'processing' or
    #   'dlc_queued' (only if the deep fix changed the pose h5).
    'deep_review': ['outdated', 'processing', 'dlc_queued', 'failed'],
    # Retry from any prior WORK step. Not 'processed': that state means
    # "passed every check, file it", and a failed video put straight there was
    # filed without the checks re-running -- with the features file of an
    # earlier run, if one had been copied down. A failed video re-runs from
    # 'processing' (or earlier) and earns 'processed' again.
    'failed': ['validated', 'dlc_queued', 'dlc_complete', 'processing', 'unresolvable'],
    # --- 'unresolvable': this NODE has no file for this video. ---
    # Not a failure and not a scientific hold: nothing went wrong with the video,
    # it simply is not here. Cross-node recovery reads state out of the shared database,
    # which carries no file paths, so it can learn about videos whose files live
    # on another machine entirely. Registering those as 'dlc_complete' or
    # 'dlc_queued' made the work loop pick them up every cycle and crash on
    # Path(None), or mark them failed -- which retries forever.
    #
    # 'failed' is wrong for them twice over: it is a retry state, and it puts an
    # infrastructure fact into a column people read as a scientific verdict.
    # 'unresolvable' is terminal for the work loop (no handler selects it) and
    # self-healing: _recover_local_dlc_queue re-drives one the moment its file
    # actually appears on this node.
    'unresolvable': ['validated', 'dlc_queued', 'dlc_complete', 'processing',
                     'processed', 'archived'],
}


class CrystallizedError(ValueError):
    """A crystallized (publication-locked) video was asked to move or change
    state. Raised before anything is moved; the message says how to unlock."""


def crystallized_refusal(db, video_id: str, action: str, folder=None,
                         analyzed_root=None):
    """The refusal message when ``video_id`` is crystallized -- on ``db``, or
    by the lock record beside its filed results -- else None.

    A crystallized video is locked for a publication (mousereach-crystallize):
    its filed results must stay exactly as they are until a person unlocks it
    (mousereach-uncrystallize). Every path that would move its files into a
    review queue, back to Processing or to a new pose asks this first and
    refuses. WHY: routing to a queue, the review return and the version
    scan's divert used to force their way through the lock (force_state), so
    a locked video's results could leave the filed folder.
    The row is one machine's record; the lock RECORD
    (``<video>_crystallized.json`` beside the filed results,
    pipeline/publication_lock.py) is the one every machine reads, so it is
    checked too, with or without ``db``: a lock set from another machine is
    honoured here. ``folder`` is the folder holding the video's filed results
    when the caller already knows it; otherwise the folder the archive step
    files into (under ``analyzed_root``, default the configured Analyzed).
    A record that cannot be checked (the share is unreachable) refuses.
    A row that cannot be read is not a lock by itself."""
    if db is not None:
        try:
            row = db.get_video(video_id) or {}
        except Exception:
            row = {}
        if row.get('state') == 'crystallized':
            label = row.get('crystallized_label') or '?'
            return (f"{video_id} is crystallized (locked for '{label}'); not {action}. "
                    f"Unlock it first with mousereach-uncrystallize if this is intended.")
    try:
        from mousereach.pipeline.publication_lock import lock_refusal
        return lock_refusal(video_id, action, analyzed_root=analyzed_root, folder=folder)
    except Exception as e:
        return (f"cannot tell whether {video_id} is locked for a publication ({e}); "
                f"not {action}.")


def _retry_network_errors(method):
    """Retry DB methods on transient network drive errors (readonly, locked, busy)."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self._is_network_drive:
            return method(self, *args, **kwargs)

        max_retries = 5
        base_delay = 0.5
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return method(self, *args, **kwargs)
            except sqlite3.OperationalError as e:
                last_error = e
                err = str(e).lower()
                is_transient = any(x in err for x in [
                    'readonly', 'locked', 'busy', 'disk i/o',
                ])
                if is_transient and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"DB retry {attempt + 1}/{max_retries}: {e} "
                        f"(waiting {delay:.1f}s)"
                    )
                    time.sleep(delay)
                    continue
                raise
        raise last_error
    return wrapper


# The start of the error_message an 'archived' row carries when a re-run of the
# filed video failed for good and the row was put back (its filed results
# stand). clear_errors_on_archived keeps such messages; filing clears them.
FAILED_RERUN_PREFIX = "re-run of a filed video failed"

# The start of the error_message an 'outdated' row carries while its re-run
# could not start (its files could not be copied down) and it waits to try
# again; " (attempt N)" follows it. Cleared when a re-run starts
# (orchestrator.ProcessingOrchestrator._rerun_could_not_start says why).
RERUN_NOT_STARTED_PREFIX = "re-run could not start"

# The start of the error_message of an 'unresolvable' row the processing
# server parked because a posed video's files were not all in the posed folder
# when its intake looked (a GPU node was still handing them over, say). The
# posed-folder scan (state.WatcherStateManager.discover_dlc_staged) puts such
# a row back to 'dlc_complete' -- ready for intake -- once the video and its
# pose are there; the re-pose adoption (repose.adopt_staged_reposes) leaves it
# to that scan. WHY a fixed start: 'unresolvable' has other reasons, and only
# this one says "the files are on their way to the posed folder".
INTAKE_MISS_REASON = "posed files not all in the posed folder when intake looked"

# Written inside an INTAKE_MISS_REASON row's message when the files WERE in the
# posed folder but one of them failed to copy (a full disk, a dropped share, a
# locked file). Such a row is not brought back at once: the posed-folder scan
# waits WatcherStateManager.COPY_RETRY_WAIT_S after the failure, and after
# watcher.max_retries attempts the intake marks the row 'failed' instead
# (orchestrator._intake_from_staging). WHY: the files are all there, so
# "back at once" would copy the whole video over the share and throw it away
# again on every pass, for ever.
INTAKE_COPY_FAILED = "failed to copy"

# The states in which a row's reprocess_scope means something: a re-run that
# is owed ('outdated') or under way ('processing', a 'failed' run waiting for
# its retry, and the 'processed' -> 'archiving' steps of filing it). Entering
# any other state clears the scope, unless the caller sets it. And a scope is
# carried only BETWEEN these states: a row that enters one of them from a state
# outside them starts with no scope unless the caller gives one
# (WatcherDB._end_of_round_defaults).
# WHY: the scope used to outlive its re-run on rows that went on to triage,
# deep review, a quality hold or 'unresolvable'. A later run of such a row --
# a quality-hold release, an intake, a restart -- read a leftover
# 'kinematics' back (orchestrator.kept_rerun_start_stage) and skipped every
# algorithm, the quality triage and the hand-score comparison, on outputs
# nobody had re-checked. A scope written for one round now ends with it.
# WHY the second half: rows written before this rule still carry a scope in
# 'triage', 'deep_review', 'unresolvable' or 'archived'. Such a row going
# triage -> 'failed' or -> 'outdated' would have brought its old 'kinematics'
# into the very states where it is read. Opening the database also clears
# those leftovers (WatcherDB._clear_scopes_outside_reruns), but that is a
# tidy-up a locked database can skip; the transition rule is what holds.
REPROCESS_SCOPE_STATES = frozenset(
    {"outdated", "processing", "failed", "processed", "archiving"})

# The states in which a row's resume_stage and held_at_step mean something: a
# review return's resumed run ('processing', a 'failed' run waiting for its
# retry, and the 'processed' -> 'archiving' steps of filing it). Entering any
# other state clears both, unless the caller sets them -- with one exception:
# held_at_step is also kept on entering a review hold ('triage',
# 'deep_review'; HELD_AT_STEP_STATES). A new hold (review_gate.route_to_queue)
# writes its own; a row set to its queue's state because its bundle was found
# there (the watcher's reconcile paths) keeps the step its route recorded,
# which is the same one the bundle's manifest carries.
# WHY: a resume point is written for ONE round, by the return of a review.
# It used to outlive that round on rows that a person reset (reset_failed),
# that were sent for a new pose, or that went 'unresolvable' and were found
# again. Their next run read the leftover back (orchestrator._run_pipeline ->
# _resume_point) and reused the segmentation, reach and outcome outputs of
# the earlier round -- a pose redone with the same model has the same scorer
# name, so resume_point.current_start cannot tell those outputs came from the
# distrusted pose. A reset, a re-pose or a fresh intake is a fresh run with
# every step and check.
RESUME_POINT_STATES = frozenset(
    {"processing", "failed", "processed", "archiving"})
HELD_AT_STEP_STATES = RESUME_POINT_STATES | {"triage", "deep_review"}


class WatcherDB:
    """
    SQLite database for tracking video pipeline state.

    Thread-safe with automatic table creation and state validation.
    Detects network drives and uses DELETE journal mode instead of WAL.
    Retries transient errors on network drives with exponential backoff.
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

        # DELETE journal mode EVERYWHERE, not just network drives. WAL was
        # used on local drives until 2026-09-01, when Windows shm handling
        # under multi-process access (watcher + external read-only probes +
        # force-killed deploys) twice degraded the database to
        # SQLITE_READONLY for every writer -- the whole node stalled until
        # all holders were stopped and the -shm/-wal sidecars deleted. DELETE
        # mode has no sidecars to poison, is what the shared-NAS databases
        # already use by design, and this db's write volume is tiny (state
        # rows and log lines); brief reader/writer blocking is a non-issue at
        # this scale. Do not switch back to WAL without solving the shm
        # poisoning: the failure is silent until the first write and then
        # total.
        conn.execute("PRAGMA journal_mode=DELETE")

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys=ON")

        return conn

    @_retry_network_errors
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
                        pipeline_versions_hash TEXT,
                        crystallized_at TEXT,
                        crystallized_by TEXT,
                        crystallized_label TEXT,
                        reprocess_scope TEXT,
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

        # Run migrations for existing databases
        self._migrate_db()

    @_retry_network_errors
    def _migrate_db(self):
        """Add columns that may not exist in older databases."""
        new_columns = [
            ('videos', 'pipeline_versions_hash', 'TEXT'),
            ('videos', 'crystallized_at', 'TEXT'),
            ('videos', 'crystallized_by', 'TEXT'),
            ('videos', 'crystallized_label', 'TEXT'),
            ('videos', 'reprocess_scope', 'TEXT'),
            ('collages', 'retry_count', 'INTEGER'),
            # WHY a column and not the log: force_state's reason goes only to
            # the text log, so nothing queryable distinguishes a HAND-MARK
            # ("re-run this pathological video") from a scanner mark. The
            # reprocessor's two-way door must never un-mark a hand-mark, and
            # a mark's reason must survive log rotation to enforce that.
            ('videos', 'mark_reason', 'TEXT'),
            # Which step sent the video to a review queue, written when it is
            # routed (review_gate.route_to_queue), and where its run resumes
            # once a person has answered (review_return._return_to_processing).
            # WHY on the row: a returned video resumes just after the step that
            # sent it, with every later step run (watcher/resume_point.py); a
            # resume point held only in memory would be lost by a retry or a
            # restart, and the run would fall back to segmentation.
            ('videos', 'held_at_step', 'TEXT'),
            ('videos', 'resume_stage', 'TEXT'),
        ]

        with self._lock:
            conn = self._get_connection()
            try:
                for table, column, col_type in new_columns:
                    try:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                        logger.debug(f"Added column {table}.{column}")
                    except Exception as err:
                        # "Already exists" is the normal case on every start after
                        # the first. WHY report the rest: a locked database also
                        # landed here, and then the column was missing with no
                        # sign of why later writes to it failed.
                        if "duplicate column" not in str(err).lower():
                            logger.warning(f"Could not add column {table}.{column}: {err}")
                conn.commit()
                self._known_columns = self._read_video_columns(conn)
            finally:
                conn.close()
        missing = [c for t, c, _ in new_columns
                   if t == 'videos' and c not in self._known_columns]
        if missing:
            # Not fatal here: update_state and force_state try again before
            # they write (_ensure_video_columns), and until a column is there
            # they do not add their own writes to it.
            logger.error(
                "watcher database %s: column(s) %s could not be added (the "
                "database was probably locked by another program). They are "
                "tried again at the next state change.",
                self.db_path, ", ".join(missing))
        self._clear_scopes_outside_reruns()

    # The columns added after the first release that update_state and
    # force_state write BY THEMSELVES (to end a re-run scope or a resume point
    # with its round), with their types. See _ensure_video_columns.
    _ROUND_COLUMNS = (('reprocess_scope', 'TEXT'), ('resume_stage', 'TEXT'),
                      ('held_at_step', 'TEXT'))

    @staticmethod
    def _read_video_columns(conn) -> set:
        return {r[1] for r in conn.execute("PRAGMA table_info(videos)").fetchall()}

    def _ensure_video_columns(self, conn) -> set:
        """The columns the videos table really has, adding any of
        _ROUND_COLUMNS that is still missing. Never raises.

        WHY: _migrate_db adds columns when the database is opened, and it
        only logs an ALTER that fails -- a database locked at that moment (a
        review window, a status view and the watcher all open this file) left
        the column missing for the life of the process. That was harmless
        while only callers who needed a column wrote it. But every state
        change now writes these three columns to end a scope or a resume
        point, so one missed ALTER would have failed EVERY later transition
        of that process -- routing to a queue, filing, archiving -- with "no
        such column". So the missing ones are added again here, just before
        a write; and a column that still cannot be added is left out of the
        writes this class adds by itself (_end_of_round_defaults), so the
        state change itself goes through as it did before these columns
        existed. A column a CALLER names is still written, and still fails
        loudly if it is not there."""
        known = getattr(self, '_known_columns', None)
        if known is not None and all(c in known for c, _ in self._ROUND_COLUMNS):
            return known
        try:
            known = self._read_video_columns(conn)
            for column, col_type in self._ROUND_COLUMNS:
                if column in known:
                    continue
                try:
                    conn.execute(f"ALTER TABLE videos ADD COLUMN {column} {col_type}")
                    logger.warning(f"Added column videos.{column}, which could not "
                                   f"be added when the database was opened")
                except Exception as err:
                    if "duplicate column" not in str(err).lower():
                        logger.warning(f"Could not add column videos.{column}: {err}")
            known = self._read_video_columns(conn)
        except Exception as err:
            logger.warning(f"Could not read the columns of the videos table: {err}")
            known = known or set()
        self._known_columns = known
        return known

    def _end_of_round_defaults(self, conn, current_state: str, new_state: str,
                               kwargs: dict) -> None:
        """Add to ``kwargs`` the writes that end a re-run scope and a review
        return's resume point when the row leaves their round. A value the
        caller passed is never replaced.

          * reprocess_scope -> NULL when the new state is outside
            REPROCESS_SCOPE_STATES, and also when the row ENTERS one of those
            states from outside them (the constant's comment says why);
          * resume_stage -> NULL outside RESUME_POINT_STATES;
          * held_at_step -> NULL outside HELD_AT_STEP_STATES.
        A column the database does not have yet is skipped
        (_ensure_video_columns)."""
        known = self._ensure_video_columns(conn)
        entering = (current_state != new_state
                    and current_state not in REPROCESS_SCOPE_STATES)
        if 'reprocess_scope' in known and (
                new_state not in REPROCESS_SCOPE_STATES or entering):
            kwargs.setdefault('reprocess_scope', None)
        if 'resume_stage' in known and new_state not in RESUME_POINT_STATES:
            kwargs.setdefault('resume_stage', None)
        if 'held_at_step' in known and new_state not in HELD_AT_STEP_STATES:
            kwargs.setdefault('held_at_step', None)

    def _end_of_round_sql(self, conn, current_state: str, new_state: str) -> str:
        """The same rule as _end_of_round_defaults, as "col = NULL, " text
        for the writers that set a state with their own UPDATE (mark_failed,
        mark_unresolvable, reset_failed). '' when nothing ends."""
        ended = {}
        self._end_of_round_defaults(conn, current_state, new_state, ended)
        return "".join(f"{column} = NULL, " for column in ended)

    def _clear_scopes_outside_reruns(self) -> int:
        """Clear reprocess_scope on rows whose state is outside
        REPROCESS_SCOPE_STATES. Run each time the database is opened; returns
        how many rows were cleared. Never raises.

        WHY: before a scope ended with its re-run, rows kept one after they
        went on to a review queue, 'unresolvable' or 'archived'. Nothing reads
        a scope in those states and no transition carries one out of them
        (_end_of_round_defaults), so these are leftovers that only mislead a
        person reading the row. It looks first and writes only when there is
        something to clear, so opening the database stays a read on every
        start after the first. If the database is locked the leftovers stay
        until the next open, which is safe for the reason just given.

        WHY it writes what it is about to clear into the log first: the
        clearing cannot be undone from the database, and it is done by
        whichever program happens to open the database first, with nobody
        watching. So BEFORE the UPDATE the log gets how many rows per state
        and every (video, state, scope) about to be cleared. The rows are read
        and cleared under one write lock, so the rows in the log are exactly
        the rows cleared. Those lines are enough to put each value back by hand
        ONLY if they reach a log file that is kept: they go to the logger of
        whichever program opened the database first. The watcher keeps its
        log, so when the watcher opens it first the record is there; when a
        dashboard or control window opens it first, the lines go to that
        program's logger and may never be written anywhere. Anyone who must be
        able to undo the clearing copies the database before the first start
        of this version, and puts values back from that copy."""
        marks = ",".join("?" * len(REPROCESS_SCOPE_STATES))
        where = f"reprocess_scope IS NOT NULL AND state NOT IN ({marks})"
        states = tuple(sorted(REPROCESS_SCOPE_STATES))
        try:
            with self._lock:
                conn = self._get_connection()
                try:
                    if conn.execute(f"SELECT 1 FROM videos WHERE {where} LIMIT 1",
                                    states).fetchone() is None:
                        return 0
                    # Take the write lock BEFORE reading the rows, so the
                    # rows written to the log are exactly the rows cleared:
                    # no other program can change one in between.
                    conn.execute("BEGIN IMMEDIATE")
                    found = [tuple(r) for r in conn.execute(
                        f"SELECT video_id, state, reprocess_scope FROM videos "
                        f"WHERE {where} ORDER BY state, video_id", states)]
                    per_state = {}
                    for _, state, _ in found:
                        per_state[state] = per_state.get(state, 0) + 1
                    logger.info(
                        "Clearing a leftover re-run scope on %d row(s) that are not "
                        "in a re-run -- per state: %s. The rows, as video (state) = "
                        "scope, follow.", len(found),
                        ", ".join(f"{s}: {n}" for s, n in sorted(per_state.items())))
                    # A few dozen to a line, so no single log line is cut short.
                    for i in range(0, len(found), 40):
                        logger.info("Leftover re-run scope being cleared: %s", "; ".join(
                            f"{vid} ({state}) = {scope}"
                            for vid, state, scope in found[i:i + 40]))
                    # One statement, under the lock taken above: one per row
                    # held the database for many seconds on a table with
                    # thousands of leftovers.
                    cur = conn.execute(
                        f"UPDATE videos SET reprocess_scope = NULL WHERE {where}", states)
                    n = cur.rowcount or 0
                    conn.commit()
                finally:
                    conn.close()
            logger.info(f"Cleared a leftover re-run scope on {n} row(s) that are "
                        f"not in a re-run")
            return n
        except Exception as err:
            logger.warning(f"Could not clear leftover re-run scopes ({err}); "
                           f"tried again the next time the database is opened")
            return 0

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

    @contextlib.contextmanager
    def _state_write(self):
        """One connection holding the database's write lock from the read of a
        row's state to the write of its new one: BEGIN IMMEDIATE, the caller's
        SELECT, check and UPDATE, then COMMIT -- or ROLLBACK and re-raise.

        WHY: the connection is in autocommit mode, so a state change used to
        be a SELECT and an UPDATE as two separate statements. Several programs
        write this database (the watcher, the review windows, route-to-queue,
        repose-videos, the command-line tools); two of them could both read the
        same state, both pass the transition check, and the second UPDATE then
        silently overwrote the first with a state that was never legal from
        the row's real state. Under the write lock the second program waits
        (or, after the busy timeout, raises 'database is locked', which
        _retry_network_errors retries on a network drive) and then checks
        against the state the first one wrote."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    yield conn
                    conn.commit()
                except BaseException:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    raise
            finally:
                conn.close()

    @_retry_network_errors
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

    # Stored in source_path when a video is registered with no file on this node.
    # Deliberately not path-shaped: the previous placeholder was 'recovered',
    # which reads as a relative path and was passed to Path() by four handlers.
    NO_FILE_HERE = '(no file on this node)'

    @_retry_network_errors
    def register_video(self, video_id: str, source_path: str = None,
                      collage_id: str = None, **metadata) -> str:
        """
        Register new video.

        ``source_path`` may be omitted, and the row is still created -- but then
        it is created honestly. Two things happen in that case:

          1. The video's .mp4 is looked for on this node (working folders, NAS
             staging, this video's archive folder). If it is found, that real
             path is recorded and the row is ordinary.
          2. If it is not found anywhere, the row is created in the terminal
             'unresolvable' state with the reason in error_message.

        This exists because the alternative was tried and it broke the pipeline
        twice. Cross-node recovery reads pipeline_videos out of the shared
        database, a table that carries state but no file paths -- every one of
        its rows had source_path NULL, because sync_video_state writes with INSERT OR
        REPLACE and only ever passes the columns it knows about. Passing that
        NULL straight through raised "NOT NULL constraint failed:
        videos.source_path" on every scan; substituting a placeholder instead
        minted rows in working states for videos this node has no files for,
        which the work loop then picked up and crashed on (Path(None)) or marked
        failed, forever. Neither of those is better than saying "not here".

        Args:
            video_id: Unique video identifier
            source_path: Full path to source file. Optional -- see above.
            collage_id: Optional collage filename this came from
            **metadata: Additional fields (date, animal_id, experiment, cohort,
                       subject, tray_type, tray_position, etc.)

        Returns:
            video_id (unchanged)
        """
        unresolved_reason = None
        if not source_path:
            found = None
            try:
                from mousereach.watcher.locate import locate_video_file
                found = locate_video_file(video_id)
            except Exception as e:
                logger.debug(f"Could not search for {video_id} on this node: {e}")
            if found is not None:
                source_path = str(found)
                logger.info(
                    f"Registering {video_id} with no source_path given; found its "
                    f"file on this node at {found}")
            else:
                source_path = self.NO_FILE_HERE
                unresolved_reason = (
                    "registered with no source_path and no file for it anywhere on "
                    "this node -- recorded as unresolvable rather than given a "
                    "working state it cannot act on")
                logger.warning(f"{video_id}: {unresolved_reason}")

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

                if unresolved_reason:
                    conn.execute(
                        "UPDATE videos SET state = 'unresolvable', error_message = ?, "
                        "updated_at = ? WHERE video_id = ?",
                        (unresolved_reason, self._now(), video_id))

                conn.commit()
                logger.info(f"Registered video {video_id}")
                return video_id

            except Exception as e:
                logger.error(f"Failed to register video {video_id}: {e}")
                raise
            finally:
                conn.close()

    @_retry_network_errors
    def clear_errors_on_archived(self) -> int:
        """Clear the error_message of every 'archived' row; return how many.

        Run at watcher start. WHY: until 2026-10-22 reaching 'archived' did not
        clear the previous error, so finished videos kept showing failures from
        runs that later succeeded (e.g. after a share outage). The history is in processing_log.

        Except a message starting with FAILED_RERUN_PREFIX: that is not an old
        error but the CURRENT state of a filed video whose re-run failed (its
        filed results stand, the re-run did not happen); the watcher lists it
        for a person from exactly these rows
        (orchestrator.ProcessingOrchestrator._restore_filed_after_failed_rerun).
        Filing the video again clears it.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "UPDATE videos SET error_message = NULL WHERE state = 'archived' "
                    "AND error_message IS NOT NULL AND error_message NOT LIKE ?",
                    (FAILED_RERUN_PREFIX + "%",))
                conn.commit()
                return cur.rowcount or 0
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
        # Reaching 'archived' ends whatever went wrong before: the row's
        # error_message is the CURRENT error, and a finished video has none.
        # WHY: archived rows still showed failures from runs that later
        # succeeded, and every status view read them as current.
        # The history stays in processing_log.
        if new_state == 'archived':
            kwargs.setdefault('error_message', None)
        try:
            # Read, check and write under one write lock (_state_write).
            with self._state_write() as conn:
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

                # A re-run scope ends with its re-run and a resume point with
                # its round (_end_of_round_defaults).
                self._end_of_round_defaults(conn, current_state, new_state, kwargs)

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
            logger.info(f"Updated video {video_id}: {current_state} -> {new_state}")

        except Exception as e:
            logger.error(f"Failed to update video {video_id}: {e}")
            raise

    @_retry_network_errors
    def force_state(self, video_id: str, new_state: str, reason: str = None,
                    unlock_crystallized: bool = False, **kwargs):
        """
        Set video state directly, bypassing transition validation.

        Used for intake scenarios where videos arrive at an intermediate state
        (e.g., DLC already completed by another machine), for cross-node recovery
        where this node adopts another node's state, and to unlock 'crystallized'.

        This is a deliberate escape hatch from the state machine, so every use is
        logged at WARNING with the transition it bypassed and why. Without that,
        a bypass is indistinguishable from a legal transition after the fact --
        and the state machine's guarantees are only as good as the visibility of
        the things that skip it.

        Args:
            video_id: Video identifier
            new_state: State to set
            reason: Why validation is being bypassed (recorded in the log)
            unlock_crystallized: True only for the unlock command
                (pipeline.versions.uncrystallize). Without it a crystallized
                row is never moved to another state: CrystallizedError.
            **kwargs: Additional fields to update
        """
        # Reaching 'archived' ends whatever went wrong before: the row's
        # error_message is the CURRENT error, and a finished video has none.
        # WHY: archived rows still showed failures from runs that later
        # succeeded, and every status view read them as current.
        # The history stays in processing_log.
        if new_state == 'archived':
            kwargs.setdefault('error_message', None)
        try:
            # Read, check and write under one write lock (_state_write).
            with self._state_write() as conn:
                row = conn.execute(
                    "SELECT state FROM videos WHERE video_id = ?",
                    (video_id,)
                ).fetchone()

                if not row:
                    raise ValueError(f"Video {video_id} not found")

                current_state = row['state']

                # A crystallized video is locked for a publication. The bypass
                # does not reach through the lock: only the unlock command
                # passes unlock_crystallized (crystallized_refusal says why).
                if (current_state == 'crystallized' and new_state != 'crystallized'
                        and not unlock_crystallized):
                    raise CrystallizedError(
                        f"{video_id} is crystallized; force_state to '{new_state}' "
                        f"refused (reason given: {reason or 'none'}). Unlock it "
                        f"with mousereach-uncrystallize first.")

                # Whether this transition would have been legal anyway; the
                # WARNING below says so (after the commit, so a bypass that was
                # rolled back is never logged as done).
                legal = new_state in VIDEO_TRANSITIONS.get(current_state, [])

                # A re-run scope ends with its re-run and a resume point with
                # its round (_end_of_round_defaults).
                self._end_of_round_defaults(conn, current_state, new_state, kwargs)

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
            # Make the bypass visible. If this transition would have been legal
            # anyway, say so -- that distinguishes "force_state used where
            # update_state would have done" from a genuine override.
            logger.warning(
                "force_state: %s %s -> %s (bypassed validation%s)%s",
                video_id, current_state, new_state,
                "; transition was legal anyway" if legal else "",
                f" reason: {reason}" if reason else " reason: not given",
            )

        except Exception as e:
            logger.error(f"Failed to force-set video {video_id}: {e}")
            raise

    @_retry_network_errors
    def force_collage_state(self, filename: str, new_state: str, **kwargs):
        """
        Set collage state directly, bypassing transition validation.

        Used by startup recovery to restore collage states from central DB.

        Args:
            filename: Collage filename
            new_state: State to set
            **kwargs: Additional fields to update
        """
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT state FROM collages WHERE filename = ?",
                    (filename,)
                ).fetchone()

                if not row:
                    raise ValueError(f"Collage {filename} not found")

                current_state = row['state']

                fields = ['state = ?', 'updated_at = ?']
                values = [new_state, self._now()]

                state_timestamp_map = {
                    'validated': 'validated_at',
                    'stable': 'stable_since',
                    'cropping': 'crop_started_at',
                    'cropped': 'crop_completed_at',
                    'archived': 'archived_at',
                }

                if new_state in state_timestamp_map:
                    timestamp_field = state_timestamp_map[new_state]
                    fields.append(f'{timestamp_field} = ?')
                    values.append(self._now())

                for key, value in kwargs.items():
                    fields.append(f'{key} = ?')
                    values.append(value)

                values.append(filename)
                conn.execute(
                    f"UPDATE collages SET {', '.join(fields)} WHERE filename = ?",
                    values
                )

                conn.commit()
                logger.info(f"Force-set collage {filename}: {current_state} -> {new_state}")

            except Exception as e:
                logger.error(f"Failed to force-set collage {filename}: {e}")
                raise
            finally:
                conn.close()

    @_retry_network_errors
    def set_collage_source_path(self, filename: str, source_path: str):
        """Correct where a collage's file actually is, without touching its state.

        Cross-node recovery used to write the OWNING HOSTNAME into this column,
        because pipeline_collages has a hostname and no path. _process_collage
        then did Path('EXAMPLE-HOST-1').exists() -> False and raised
        FileNotFoundError, and because the row existed, discover_new_collages
        skipped the file forever: the collage could never enter the pipeline
        again even with the .mkv sitting in the intake folder.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "UPDATE collages SET source_path = ?, updated_at = ? "
                    "WHERE filename = ?",
                    (source_path, self._now(), filename))
                conn.commit()
                logger.info(f"Collage {filename}: source_path -> {source_path}")
            finally:
                conn.close()

    @_retry_network_errors
    def retry_parked_collage(self, filename: str) -> Optional[int]:
        """Give ONE failed collage its automatic retries back: retry_count
        goes to 0. Returns the old count, or None when nothing was changed
        (no such collage, or it is not 'failed').

        The state is left 'failed' on purpose: the collage intake scan
        (state.discover_new_collages) moves a failed collage on to
        'validated' only while its file is still in the intake folder, and
        counts the retry itself -- so this goes through exactly the retry the
        watcher makes on its own, on the next scan, with no restart.

        WHY: the only remedy for a collage that had used up its retries was
        to raise watcher.max_retries for the whole machine and restart the
        watcher, which gave every failed video on that machine another try
        too. The old count and error are kept in processing_log (step
        'retry')."""
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT state, retry_count, validation_error FROM collages "
                    "WHERE filename = ?", (filename,)).fetchone()
                if not row or row['state'] != 'failed':
                    return None
                old = int(row['retry_count'] or 0)
                cur = conn.execute(
                    "UPDATE collages SET retry_count = 0, updated_at = ? "
                    "WHERE filename = ? AND state = 'failed'",
                    (self._now(), filename))
                conn.commit()
                if cur.rowcount != 1:
                    return None
                error = row['validation_error']
            finally:
                conn.close()
        self.log_step(filename, 'retry', 'started',
                      message=(f"retries given back by a person (it had used "
                               f"{old}); last error: {error or 'none recorded'}"))
        logger.info(f"Collage {filename}: retries given back (was {old}); "
                    f"tried again at the next scan while its file is in the "
                    f"intake folder")
        return old

    @_retry_network_errors
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

    @_retry_network_errors
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

    @_retry_network_errors
    def get_videos_held_by_failed_archive(self) -> List[dict]:
        """Analysed videos ('processed') whose last logged step is a FAILED
        'archive': filing them did not work and the watcher is waiting out a
        back-off before it tries again.

        WHY this is read from the log: the back-off itself lives only in the
        running watcher's memory, so without this the status command showed
        such a video as plain 'processed' -- indistinguishable from one about
        to be filed -- while it could sit there for hours. The newest log row
        is taken by id, not by created_at: the log stamps UTC while other
        columns are local time, and id is the order the rows were written.

        Returns dicts with video_id, message (the archive error) and
        failed_at (the log row's created_at, UTC).
        """
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute("""
                    SELECT v.video_id AS video_id, l.message AS message,
                           l.created_at AS failed_at
                    FROM videos v
                    JOIN processing_log l ON l.id = (
                        SELECT MAX(id) FROM processing_log WHERE video_id = v.video_id)
                    WHERE v.state = 'processed'
                      AND l.step = 'archive' AND l.status = 'failed'
                    ORDER BY l.id
                """).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    @_retry_network_errors
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

    def set_fields(self, video_id: str, **fields) -> None:
        """Update columns on a row WITHOUT changing its state.

        For bookkeeping that belongs to the row's current state -- narrowing
        a reprocess_scope, recording where a fresh pose landed -- where
        update_state would refuse (no transition) and force_state would log a
        validation bypass that never happened. Raises if the row is missing.
        """
        if not fields:
            return
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT state FROM videos WHERE video_id = ?", (video_id,)
                ).fetchone()
                if not row:
                    raise ValueError(f"Video {video_id} not found")
                cols = [f'{k} = ?' for k in fields] + ['updated_at = ?']
                vals = list(fields.values()) + [self._now(), video_id]
                conn.execute(
                    f"UPDATE videos SET {', '.join(cols)} WHERE video_id = ?", vals)
                conn.commit()
                logger.info(f"Updated video {video_id} fields ({row['state']}): "
                            f"{', '.join(fields)}")
            except Exception as e:
                logger.error(f"Failed to update fields on {video_id}: {e}")
                raise
            finally:
                conn.close()

    @_retry_network_errors
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

    @_retry_network_errors
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

    @_retry_network_errors
    def states_of(self, video_ids) -> Dict[str, tuple]:
        """{video_id: (state, updated_at)} for those of ``video_ids`` that have
        a row. One query per 500 ids. WHY: the review-return scan needs the
        state of every queued video each pass, and one get_video per bundle is
        a connection per bundle (hundreds per scan)."""
        ids = list(dict.fromkeys(video_ids))
        out: Dict[str, tuple] = {}
        if not ids:
            return out
        with self._lock:
            conn = self._get_connection()
            try:
                for i in range(0, len(ids), 500):
                    chunk = ids[i:i + 500]
                    marks = ",".join("?" * len(chunk))
                    for row in conn.execute(
                            f"SELECT video_id, state, updated_at FROM videos "
                            f"WHERE video_id IN ({marks})", chunk):
                        out[row["video_id"]] = (row["state"], row["updated_at"])
                return out
            finally:
                conn.close()

    @_retry_network_errors
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

    @_retry_network_errors
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

    @_retry_network_errors
    def get_animal_summary(self) -> list:
        """
        Get per-animal pipeline summary.

        Returns:
            List of dicts, one per animal_id, with state counts.
            Sorted by animal_id.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute("""
                    SELECT
                        animal_id,
                        state,
                        COUNT(*) as count
                    FROM videos
                    WHERE animal_id IS NOT NULL AND animal_id != ''
                    GROUP BY animal_id, state
                    ORDER BY animal_id, state
                """).fetchall()

                # Pivot into per-animal dicts
                animals = {}
                for row in rows:
                    aid = row['animal_id']
                    if aid not in animals:
                        animals[aid] = {'animal_id': aid, 'total': 0, 'states': {}}
                    animals[aid]['states'][row['state']] = row['count']
                    animals[aid]['total'] += row['count']

                return sorted(animals.values(), key=lambda a: a['animal_id'])

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

    @_retry_network_errors
    def mark_failed(self, video_id: str, error_message: str):
        """
        Mark video as failed with error tracking.

        Args:
            video_id: Video identifier
            error_message: Error description

        Raises CrystallizedError for a crystallized row: a publication-locked
        video is never re-labelled failed (the same rule force_state keeps;
        crystallized_refusal says why). Its filed results are untouched by
        whatever went wrong, and 'failed' would put it back in the retry work.
        """
        try:
            # Read and write under one write lock (_state_write).
            with self._state_write() as conn:
                # Get current error count
                row = conn.execute(
                    "SELECT error_count, state FROM videos WHERE video_id = ?",
                    (video_id,)
                ).fetchone()

                if not row:
                    raise ValueError(f"Video {video_id} not found")
                if row['state'] == 'crystallized':
                    raise CrystallizedError(
                        f"{video_id} is crystallized; not marked failed ({error_message}). "
                        f"Unlock it with mousereach-uncrystallize first if this is intended.")

                error_count = (row['error_count'] or 0) + 1

                # Update to failed state. A row that fails from OUTSIDE a
                # re-run (a review queue, say) does not bring a re-run scope
                # into 'failed', where the retry would read it back
                # (_end_of_round_defaults); a run that fails keeps its own.
                clears = self._end_of_round_sql(conn, row['state'], 'failed')
                conn.execute(f"""
                    UPDATE videos
                    SET state = 'failed',
                        {clears}error_message = ?,
                        error_count = ?,
                        last_error_at = ?,
                        updated_at = ?
                    WHERE video_id = ?
                """, (error_message, error_count, self._now(), self._now(), video_id))
            logger.warning(f"Marked video {video_id} as failed (error #{error_count}): {error_message}")

        except Exception as e:
            logger.error(f"Failed to mark {video_id} as failed: {e}")
            raise

    @_retry_network_errors
    def mark_unresolvable(self, video_id: str, reason: str):
        """Record that this NODE has no file for this video, and stop working it.

        Use this, not mark_failed, whenever a handler cannot find the video's
        input. The difference is not cosmetic:

          - mark_failed is a RETRY state. Every reset puts the video back in the
            work loop, where it fails on the same missing file, without end.
          - 'failed' is also read by people as "something went wrong with this
            animal's data". A file that lives on another machine is not that.

        error_count is deliberately NOT incremented here: nothing was attempted
        and nothing failed. (One caller counts on its own: an intake whose copy
        failed sets error_count and last_error_at right after, because there
        the files were here and a copy WAS attempted -- see
        INTAKE_COPY_FAILED.) The row keeps whatever it already knows, and
        _recover_local_dlc_queue puts it straight back into the pipeline if the
        file ever turns up here.

        Raises CrystallizedError for a crystallized row, as mark_failed and
        force_state do (crystallized_refusal says why): a publication-locked
        video keeps its state, and 'unresolvable' can be re-driven into the
        pipeline when a file turns up.
        """
        try:
            # Read and write under one write lock (_state_write).
            with self._state_write() as conn:
                row = conn.execute(
                    "SELECT state FROM videos WHERE video_id = ?", (video_id,)
                ).fetchone()
                if not row:
                    raise ValueError(f"Video {video_id} not found")
                if row['state'] == 'crystallized':
                    raise CrystallizedError(
                        f"{video_id} is crystallized; not marked unresolvable ({reason}). "
                        f"Unlock it with mousereach-uncrystallize first if this is intended.")

                # Outside every re-run and review round: scope and resume
                # point end here (_end_of_round_defaults).
                clears = self._end_of_round_sql(conn, row['state'], 'unresolvable')
                conn.execute(f"""
                    UPDATE videos
                    SET state = 'unresolvable',
                        {clears}error_message = ?,
                        updated_at = ?
                    WHERE video_id = ?
                """, (reason, self._now(), video_id))
            logger.warning(
                f"{video_id}: no file on this node ({row['state']} -> "
                f"unresolvable): {reason}")

        except Exception as e:
            logger.error(f"Failed to mark {video_id} unresolvable: {e}")
            raise

    # The states a person's reset can choose with --from-step: the legal
    # targets from 'failed' that put a video back in front of a work step.
    # 'validated' and 'unresolvable' are left out on purpose (see
    # infer_reset_state for the one case where 'validated' is inferred).
    # 'processed' is not one (see the 'failed' transitions above).
    RESET_STEP_CHOICES = ('dlc_queued', 'dlc_complete', 'processing')

    @staticmethod
    def infer_reset_state(video_id: str, dlc_queue=None, processing_dir=None,
                          error_message=None):
        """Where a reset of a failed video should put it, judged from the
        files THIS node holds: ``(state, {column: value})``, or None when its
        files are not here.

          * the video and a pose in the processing folder -> 'processing';
          * the video in this node's DLC queue -> 'dlc_complete' if a pose from
            the declared model is beside it, else 'dlc_queued'.

        The same reading the automatic retries make (the processing server's
        failed-video retry; a GPU node's restart, _recover_local_dlc_queue),
        so a person's reset and an automatic retry agree. ``dlc_queue`` and
        ``processing_dir`` default to this node's configured folders.

        WHY never 'discovered': it is not a legal state after 'failed', and
        inferring it made every reset of a video with no recorded pose and no
        validation time raise. WHY 'validated' only in one case: on a GPU
        node that state is "waiting in the shared singles folder for a node
        to take it", and a reset must not claim that a file is there. It IS
        there in one case, which is then the answer:

          * the row failed copying the single into a DLC queue (its
            ``error_message`` starts with single_claim.COPY_FAILED_REASON)
            and the video is back at the top of the shared singles folder ->
            'validated', with source_path and current_path set to that file.

        WHY: that failure gives the video back to the shared folder, and the
        row's own message tells a person to run mousereach-watch-reprocess
        once the cause is fixed. Without this case that reset found no file
        on this node and refused. Rows written by older versions say "could
        not copy <file> into the DLC queue; put back in the shared singles
        folder ..." and are read the same way."""
        try:
            from mousereach.config import Paths
            if dlc_queue is None:
                dlc_queue = Paths.DLC_QUEUE
            if processing_dir is None:
                processing_dir = Paths.PROCESSING
        except Exception:
            pass

        def _poses(folder):
            try:
                return sorted(Path(folder).glob(f"{video_id}DLC*.h5"))
            except OSError:
                return []

        def _declared(poses):
            try:
                from mousereach.watcher import repose
                declared = repose.declared_scorer()
                if not declared:
                    return None
                for h5 in poses:
                    if repose.scorer_of(h5) == declared:
                        return h5
            except Exception:
                return None
            return None

        def _is_file(p):
            try:
                return p.is_file()
            except OSError:
                return False

        if processing_dir:
            mp4 = Path(processing_dir) / f"{video_id}.mp4"
            poses = _poses(processing_dir)
            if poses and _is_file(mp4):
                from mousereach.pipeline.manifest import select_pose_file
                h5 = _declared(poses) or select_pose_file(poses)
                return 'processing', {'current_path': str(mp4),
                                      'dlc_output_path': str(h5) if h5 else None}
        if dlc_queue:
            mp4 = Path(dlc_queue) / f"{video_id}.mp4"
            if _is_file(mp4):
                h5 = _declared(_poses(dlc_queue))
                if h5 is not None:
                    return 'dlc_complete', {'current_path': str(mp4),
                                            'source_path': str(mp4),
                                            'dlc_output_path': str(h5)}
                return 'dlc_queued', {'current_path': str(mp4),
                                      'source_path': str(mp4)}
        try:
            from mousereach.watcher import single_claim
            msg = error_message or ''
            legacy = (msg.startswith('could not copy ')
                      and 'into the DLC queue; put back in the shared singles folder' in msg)
            if msg.startswith(single_claim.COPY_FAILED_REASON) or legacy:
                door = single_claim.front_door()
                if door is not None:
                    mp4 = door / f"{video_id}.mp4"
                    if _is_file(mp4):
                        return 'validated', {'current_path': str(mp4),
                                             'source_path': str(mp4)}
        except Exception:
            pass
        return None

    def reset_failed(self, video_id: str, to_state: str = None,
                     dlc_queue=None, processing_dir=None) -> Optional[str]:
        """
        Reset a failed video so it is worked again. Returns the state it was
        set to, or None when the row was NOT changed because it is not
        'failed' (a person's command says so; it never claims a reset it did
        not make).

        Args:
            video_id: Video identifier
            to_state: State to reset to (the person's --from-step). If None,
                it is read from the files this node holds
                (infer_reset_state); if they are not here a ValueError names
                the --from-step choices.

        The reset gives the video its automatic retries back: error_count
        goes to 0. WHY: the count is what the automatic retries compare with
        watcher.max_retries, and a person resets a video once the cause is
        fixed; keeping the old count put the very next failure past the
        limit. The old count and error are kept in processing_log
        (step 'reset').
        """
        inferred_fields = {}
        if to_state is None:
            row = self.get_video(video_id)
            found = self.infer_reset_state(
                video_id, dlc_queue, processing_dir,
                error_message=(row or {}).get('error_message'))
            if found is not None:
                to_state, inferred_fields = found
        return self._reset_failed_row(video_id, to_state, inferred_fields)

    @_retry_network_errors
    def _reset_failed_row(self, video_id: str, to_state, fields: dict) -> Optional[str]:
        """The database half of reset_failed, under one write lock
        (_state_write): the row is changed only if it is still 'failed'."""
        old_count = old_error = None
        try:
            with self._state_write() as conn:
                row = conn.execute(
                    "SELECT * FROM videos WHERE video_id = ?",
                    (video_id,)
                ).fetchone()

                if not row:
                    raise ValueError(f"Video {video_id} not found")

                if row['state'] != 'failed':
                    logger.warning(f"Video {video_id} is not in failed state "
                                   f"(current: {row['state']}); not changed")
                    return None

                if to_state is None:
                    raise ValueError(
                        f"{video_id}: its files are not on this node (not in this "
                        f"node's DLC queue, nor video and pose in its processing "
                        f"folder), so where to restart it cannot be read from them. "
                        f"Name the step: --from-step "
                        f"{' | '.join(self.RESET_STEP_CHOICES)}")

                # Validate target state exists in transitions
                valid_retry_states = VIDEO_TRANSITIONS.get('failed', [])
                if to_state not in valid_retry_states:
                    raise ValueError(
                        f"Cannot reset to {to_state}. Valid retry states: {', '.join(valid_retry_states)}"
                    )

                old_count = row['error_count'] or 0
                old_error = row['error_message']

                # Reset. A re-run scope ends with its re-run
                # (REPROCESS_SCOPE_STATES) and a review return's resume point
                # with its round (RESUME_POINT_STATES): a reset to an earlier
                # state is a fresh run with every step and check.
                clears = self._end_of_round_sql(conn, 'failed', to_state)
                sets = "".join(f"{col} = ?, " for col in fields)
                cur = conn.execute(f"""
                    UPDATE videos
                    SET state = ?,
                        error_message = NULL,
                        error_count = 0,
                        {sets}{clears}updated_at = ?
                    WHERE video_id = ? AND state = 'failed'
                """, (to_state, *fields.values(), self._now(), video_id))
                if cur.rowcount != 1:
                    return None
        except Exception as e:
            logger.error(f"Failed to reset video {video_id}: {e}")
            raise
        logger.info(f"Reset video {video_id} from failed to {to_state}")
        self.log_step(video_id, 'reset', 'completed',
                      message=f"reset from failed to {to_state}; retries given back "
                              f"(error count was {old_count}); last error: {old_error}")
        return to_state

    @_retry_network_errors
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

    @_retry_network_errors
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

    @_retry_network_errors
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

    @_retry_network_errors
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

    # =========================================================================
    # CENTRAL DB EXPORT (cross-node provenance on NAS)
    # =========================================================================

    def export_to_central_db(self, video_id: str, central_db_path: Path = None):
        """
        Export a video's record and processing log to the central DB on NAS.

        Called after successful archive. Creates a unified audit trail across
        all processing nodes at NAS_ROOT/watcher_central.db.

        Args:
            video_id: Video identifier to export
            central_db_path: Path to central DB. If None, uses NAS_ROOT/watcher_central.db
        """
        if central_db_path is None:
            try:
                from mousereach.config import Paths
                if Paths.NAS_ROOT is None:
                    logger.warning("NAS_ROOT not configured, skipping central DB export")
                    return
                central_db_path = Paths.NAS_ROOT / "watcher_central.db"
            except Exception as e:
                logger.warning(f"Could not resolve NAS_ROOT for central DB: {e}")
                return

        central_db_path = Path(central_db_path)

        try:
            # Read local data
            video_row, log_rows = self._read_video_for_export(video_id)
            if video_row is None:
                logger.warning(f"Video {video_id} not found in local DB, skipping export")
                return

            # Write to central DB
            self._write_to_central_db(central_db_path, video_row, log_rows)
            logger.info(
                f"Exported {video_id} to central DB "
                f"({len(log_rows)} log entries, machine={self._hostname})"
            )

        except Exception as e:
            # Never let central DB export failures break the pipeline
            logger.warning(f"Central DB export failed for {video_id}: {e}")

    def _read_video_for_export(self, video_id: str):
        """Read video row and processing log from local DB for export."""
        with self._lock:
            conn = self._get_connection()
            try:
                video_row = conn.execute(
                    "SELECT * FROM videos WHERE video_id = ?",
                    (video_id,)
                ).fetchone()

                if not video_row:
                    return None, []

                log_rows = conn.execute(
                    "SELECT * FROM processing_log WHERE video_id = ? ORDER BY created_at",
                    (video_id,)
                ).fetchall()

                return dict(video_row), [dict(r) for r in log_rows]

            finally:
                conn.close()

    def _write_to_central_db(self, central_db_path: Path, video_row: dict, log_rows: list):
        """Write video record and processing log to central DB on NAS."""
        central_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Detect if central DB is on network drive
        is_network = self._detect_network_drive(central_db_path)

        # Wait up to 30 s for another machine's write to finish (Python's
        # default is 5 s): the share is sometimes slow enough that an ordinary
        # write holds the file longer than that, and a failed export waits for
        # the processing server's next re-send check (up to 30 minutes).
        conn = sqlite3.connect(str(central_db_path), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        if is_network:
            conn.execute("PRAGMA journal_mode=DELETE")
        else:
            conn.execute("PRAGMA journal_mode=WAL")

        try:
            # Create tables if needed (same schema as local + source_machine)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    source_path TEXT,
                    collage_id TEXT,
                    date TEXT,
                    animal_id TEXT,
                    experiment TEXT,
                    cohort TEXT,
                    subject TEXT,
                    tray_type TEXT,
                    tray_position INTEGER,
                    state TEXT,
                    discovered_at TEXT,
                    validated_at TEXT,
                    crop_started_at TEXT,
                    crop_completed_at TEXT,
                    dlc_started_at TEXT,
                    dlc_completed_at TEXT,
                    processing_started_at TEXT,
                    processing_completed_at TEXT,
                    archived_at TEXT,
                    error_message TEXT,
                    error_count INTEGER,
                    last_error_at TEXT,
                    claimed_by TEXT,
                    current_path TEXT,
                    dlc_output_path TEXT,
                    pipeline_versions_hash TEXT,
                    crystallized_at TEXT,
                    crystallized_by TEXT,
                    crystallized_label TEXT,
                    reprocess_scope TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    source_machine TEXT NOT NULL,
                    exported_at TEXT NOT NULL,
                    UNIQUE(video_id, source_machine)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    step TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    duration_seconds REAL,
                    machine TEXT,
                    created_at TEXT,
                    source_machine TEXT NOT NULL,
                    exported_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_central_videos_id
                ON videos(video_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_central_videos_machine
                ON videos(source_machine)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_central_log_video
                ON processing_log(video_id)
            """)

            now = self._now()

            # Upsert video row (replace if same video_id + source_machine)
            conn.execute(
                "DELETE FROM videos WHERE video_id = ? AND source_machine = ?",
                (video_row['video_id'], self._hostname)
            )

            # Build insert from video_row
            columns = [
                'video_id', 'source_path', 'collage_id', 'date', 'animal_id',
                'experiment', 'cohort', 'subject', 'tray_type', 'tray_position',
                'state', 'discovered_at', 'validated_at', 'crop_started_at',
                'crop_completed_at', 'dlc_started_at', 'dlc_completed_at',
                'processing_started_at', 'processing_completed_at', 'archived_at',
                'error_message', 'error_count', 'last_error_at', 'claimed_by',
                'current_path', 'dlc_output_path', 'pipeline_versions_hash',
                'crystallized_at', 'crystallized_by', 'crystallized_label',
                'reprocess_scope', 'created_at', 'updated_at',
            ]
            values = [video_row.get(c) for c in columns]
            columns.extend(['source_machine', 'exported_at'])
            values.extend([self._hostname, now])

            placeholders = ', '.join(['?'] * len(values))
            col_list = ', '.join(columns)
            conn.execute(f"INSERT INTO videos ({col_list}) VALUES ({placeholders})", values)

            # Replace processing log entries for this video + machine
            conn.execute(
                "DELETE FROM processing_log WHERE video_id = ? AND source_machine = ?",
                (video_row['video_id'], self._hostname)
            )

            for log_row in log_rows:
                conn.execute("""
                    INSERT INTO processing_log
                    (video_id, step, status, message, duration_seconds, machine,
                     created_at, source_machine, exported_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    log_row['video_id'], log_row['step'], log_row['status'],
                    log_row.get('message'), log_row.get('duration_seconds'),
                    log_row.get('machine'), log_row.get('created_at'),
                    self._hostname, now,
                ))

            conn.commit()

        finally:
            conn.close()
