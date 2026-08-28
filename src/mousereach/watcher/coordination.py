"""
mousereach.watcher.coordination - Cross-PC pipeline coordination via connectome.db.

Uses MouseReach's own shared database (<NAS root>/watcher_central.db) as a
shared coordination layer. Each DLC PC syncs its pipeline state here so that:

1. On startup, any PC can recover its local watcher.db from shared state
2. At runtime, collage claims prevent duplicate cropping across PCs
3. Video states are visible across all PCs for monitoring

The local watcher.db remains the primary data store for speed. Connectome.db
sync is always best-effort — NAS unavailability never blocks processing.
"""

import shutil
import logging
import socket
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

from mousereach.watcher.locate import locate_video_file

if TYPE_CHECKING:
    from mousereach.watcher.db import WatcherDB

try:
    from sqlalchemy import create_engine, text
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

logger = logging.getLogger(__name__)

# Cross-node coordination lives in MouseReach's OWN shared database on the NAS
# root (watcher_central.db), never inside another tool's database. Until
# 2026-08-28 these tables were created inside the lab database file of a
# separate tool at a hardcoded lab path -- the tool owned tables in a database
# it did not own, and could not run anywhere else.
def _coordination_db_path():
    try:
        from mousereach.config import Paths
        if Paths.NAS_ROOT:
            return Path(Paths.NAS_ROOT) / "watcher_central.db"
    except Exception:
        pass
    return None


CONNECTOME_DB_PATH = None  # kept as a name for old imports; resolved lazily

# State ordering for "only advance, never regress" logic
VIDEO_STATE_ORDER = [
    'discovered', 'quarantined', 'validated', 'dlc_queued', 'dlc_running',
    'dlc_complete', 'processing', 'processed', 'archiving', 'archived',
    'crystallized',
]

# States that mean "this node holds the files and is working on it". Adopting one
# of these from another node's record, for a video whose files are not here, is
# what created the phantom rows: pathless videos sitting in dlc_complete, picked
# up by the work loop every cycle, crashing on Path(None) forever.
#
# 'archived', 'crystallized', 'triage' and 'deep_review' are NOT in this list.
# Those live on the NAS, are visible to every node, and no work-loop handler
# selects them -- adopting them is how a node learns not to redo finished work.
NODE_LOCAL_STATES = {
    'discovered', 'validated', 'dlc_queued', 'dlc_running',
    'dlc_complete', 'processing', 'processed', 'archiving', 'outdated',
}

COLLAGE_STATE_ORDER = [
    'discovered', 'quarantined', 'validated', 'stable', 'cropping', 'cropped', 'archived',
]

CREATE_PIPELINE_VIDEOS_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_videos (
    video_id        TEXT PRIMARY KEY,
    collage_id      TEXT,
    hostname        TEXT NOT NULL,
    state           TEXT NOT NULL,
    source_path     TEXT,
    nas_path        TEXT,
    discovered_at   TEXT,
    dlc_completed_at TEXT,
    processed_at    TEXT,
    staged_at       TEXT,
    updated_at      TEXT NOT NULL,
    error_message   TEXT
);
"""

CREATE_PIPELINE_COLLAGES_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_collages (
    filename        TEXT PRIMARY KEY,
    hostname        TEXT NOT NULL,
    state           TEXT NOT NULL,
    claimed_at      TEXT NOT NULL,
    completed_at    TEXT,
    singles_created INTEGER DEFAULT 0
);
"""


def _now() -> str:
    return datetime.now().isoformat()


def _state_index(state: str, order: list) -> int:
    """Get numeric index for state ordering. Returns -1 for unknown states."""
    try:
        return order.index(state)
    except ValueError:
        return -1



def _collage_source(filename: str):
    """Where this collage's file actually is on this node, or None.

    Only the intake folder counts: that is the one place a collage can be picked
    up from and cropped.
    """
    from mousereach.config import Paths
    src_dir = Paths.MULTI_ANIMAL_SOURCE
    if not src_dir:
        return None
    try:
        candidate = Path(src_dir) / filename
        return candidate if candidate.is_file() else None
    except OSError:
        return None

# =============================================================================
# DB FILE BACKUP / RESTORE
# =============================================================================

def backup_db(db_path: Path, nas_root: Path, hostname: str):
    """Copy local watcher.db to NAS as a fast-restore safety net.

    Writes to {nas_root}/watcher_state/{hostname}/watcher.db using atomic
    copy (write to .tmp, rename) to prevent corruption.

    Args:
        db_path: Local watcher.db path
        nas_root: NAS root (e.g. Y:/.../MouseReach_Pipeline)
        hostname: This PC's hostname
    """
    if not nas_root or not db_path.exists():
        return

    backup_dir = nas_root / "watcher_state" / hostname
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "watcher.db"
    tmp_path = backup_dir / "watcher.db.tmp"

    try:
        shutil.copy2(str(db_path), str(tmp_path))
        tmp_path.replace(backup_path)
        logger.debug(f"DB backed up to {backup_path}")
    except Exception as e:
        # Clean up partial copy
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def restore_db(db_path: Path, nas_root: Path, hostname: str) -> bool:
    """Restore watcher.db from NAS backup if local DB is empty or missing.

    Args:
        db_path: Local watcher.db path
        nas_root: NAS root
        hostname: This PC's hostname

    Returns:
        True if restored, False if no restore needed/available
    """
    if not nas_root:
        return False

    # Check if local DB needs restoring
    if db_path.exists() and db_path.stat().st_size > 4096:
        # DB exists and has content (> 4KB means it has data, not just schema)
        return False

    backup_path = nas_root / "watcher_state" / hostname / "watcher.db"
    if not backup_path.exists():
        logger.info("No NAS backup found, starting fresh")
        return False

    try:
        shutil.copy2(str(backup_path), str(db_path))
        logger.info(f"Restored watcher.db from NAS backup ({backup_path})")
        return True
    except Exception as e:
        logger.warning(f"Failed to restore DB from NAS: {e}")
        return False


# =============================================================================
# PIPELINE COORDINATOR
# =============================================================================

class PipelineCoordinator:
    """Syncs pipeline state to/from connectome.db for cross-PC coordination.

    Uses the same sqlalchemy pattern as mousereach.sync.database.DatabaseSyncer.
    All operations are best-effort — failures are logged but never raised.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _coordination_db_path()
        self._engine = None
        self._tables_ensured = False

    @property
    def engine(self):
        """Lazy sqlalchemy engine initialization."""
        if self._engine is None:
            if not HAS_SQLALCHEMY:
                raise ImportError("sqlalchemy required for pipeline coordination")
            if self.db_path is None:
                raise FileNotFoundError("coordination database: NAS root not configured")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)  # our own file; create it
            self._engine = create_engine(f"sqlite:///{self.db_path}")
        return self._engine

    def ensure_tables(self):
        """Create pipeline_videos and pipeline_collages tables if needed."""
        if self._tables_ensured:
            return
        with self.engine.connect() as conn:
            conn.execute(text(CREATE_PIPELINE_VIDEOS_SQL))
            conn.execute(text(CREATE_PIPELINE_COLLAGES_SQL))
            conn.commit()
        self._tables_ensured = True
        logger.debug("Pipeline coordination tables ensured")

    # -----------------------------------------------------------------
    # Video coordination
    # -----------------------------------------------------------------

    def sync_video_state(self, video_id: str, hostname: str, state: str, **kwargs):
        """Upsert video state to connectome.db.

        Called after each local DB state change. Last writer wins, which is
        correct since only one PC works a video at a time -- but only for the
        columns that writer actually supplies.

        This used to be INSERT OR REPLACE, which is a DELETE followed by an
        INSERT: every column the caller did not name was reset to NULL. Callers
        name state and a timestamp, never source_path, so every sync erased the
        path again. The result was a coordination table where all 2,899 rows had
        source_path NULL -- and cross-node recovery, which reads exactly this
        table, could therefore never give a recovered video a real file to work
        on. UPSERT updates the named columns and leaves the rest alone.
        """
        self.ensure_tables()

        fields = {
            'video_id': video_id,
            'hostname': hostname,
            'state': state,
            'updated_at': _now(),
        }
        fields.update(kwargs)

        columns = ', '.join(fields.keys())
        placeholders = ', '.join(f':{k}' for k in fields.keys())
        updates = ', '.join(f'{k} = excluded.{k}' for k in fields if k != 'video_id')

        with self.engine.connect() as conn:
            conn.execute(text(
                f"INSERT INTO pipeline_videos ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(video_id) DO UPDATE SET {updates}"
            ), fields)
            conn.commit()

    def get_all_video_states(self) -> Dict[str, dict]:
        """Read all pipeline_videos. Returns {video_id: row_dict}."""
        self.ensure_tables()
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM pipeline_videos")).fetchall()
        return {row[0]: dict(row._mapping) for row in rows}

    # -----------------------------------------------------------------
    # Collage coordination
    # -----------------------------------------------------------------

    def try_claim_collage(self, filename: str, hostname: str) -> bool:
        """Attempt to claim a collage for cropping.

        Uses INSERT OR IGNORE — first writer wins (SQLite atomic).
        Returns True if we got the claim, False if another PC already claimed it.
        """
        self.ensure_tables()
        with self.engine.connect() as conn:
            result = conn.execute(text(
                "INSERT OR IGNORE INTO pipeline_collages "
                "(filename, hostname, state, claimed_at) "
                "VALUES (:filename, :hostname, 'cropping', :claimed_at)"
            ), {'filename': filename, 'hostname': hostname, 'claimed_at': _now()})
            conn.commit()

            if result.rowcount > 0:
                return True

            # Row already existed — check who owns it
            row = conn.execute(text(
                "SELECT hostname, state FROM pipeline_collages WHERE filename = :filename"
            ), {'filename': filename}).fetchone()

            if row and row[0] == hostname:
                # We already claimed it (e.g., from a previous run)
                return True

            logger.info(f"Collage {filename} already claimed by {row[0] if row else 'unknown'}")
            return False

    def update_collage_state(self, filename: str, state: str, **kwargs):
        """Update collage state after cropping completes."""
        self.ensure_tables()
        fields = {'state': state}
        if state in ('cropped', 'archived'):
            fields['completed_at'] = _now()
        fields.update(kwargs)

        set_clause = ', '.join(f'{k} = :{k}' for k in fields.keys())
        fields['filename'] = filename

        with self.engine.connect() as conn:
            conn.execute(text(
                f"UPDATE pipeline_collages SET {set_clause} WHERE filename = :filename"
            ), fields)
            conn.commit()

    def get_all_collage_states(self) -> Dict[str, dict]:
        """Read all pipeline_collages. Returns {filename: row_dict}."""
        self.ensure_tables()
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM pipeline_collages")).fetchall()
        return {row[0]: dict(row._mapping) for row in rows}

    # -----------------------------------------------------------------
    # Startup recovery
    # -----------------------------------------------------------------

    def recover_local_db(self, local_db: "WatcherDB", hostname: str) -> dict:
        """Sync connectome.db state into local watcher.db on startup.

        For each video in pipeline_videos:
          - If not in local DB: register + force_state
          - If in local DB but behind: force_state forward
          - Never regress state (only advance)

        For each collage in pipeline_collages:
          - If not in local DB: register + force_collage_state

        Also cross-references reach_data table:
          - Any video_name in reach_data -> definitively fully processed

        Returns:
            Dict with recovery statistics
        """
        stats = {
            'videos_recovered': 0,
            'videos_advanced': 0,
            'videos_elsewhere': 0,
            'collages_recovered': 0,
            'collages_not_here': 0,
            'mousedb_confirmed': 0,
        }

        # --- Recover videos from pipeline_videos ---
        try:
            all_videos = self.get_all_video_states()
        except Exception as e:
            logger.warning(f"Could not read pipeline_videos: {e}")
            all_videos = {}

        for video_id, remote in all_videos.items():
            remote_state = remote.get('state', 'discovered')
            if remote_state == 'failed':
                continue  # Don't import failures from other PCs

            try:
                local_row = local_db.get_video(video_id)
            except Exception:
                local_row = None

            if local_row is None:
                # Not in local DB -- register, then adopt the remote state only
                # as far as this node can actually act on it.
                #
                # pipeline_videos carries state but no usable file path: it is
                # written by sync_video_state, which for most of this pipeline's
                # life was an INSERT OR REPLACE that nulled every column it did
                # not name. So a recovered video's source_path is almost always
                # NULL, and the previous code substituted the string 'recovered'
                # and adopted the remote state anyway. That is how a video whose
                # files sit on another machine ended up in this node's
                # dlc_complete queue, where _stage_to_nas called Path(None),
                # raised TypeError, marked it failed, and met it again next cycle.
                try:
                    # The search costs a handful of network stats per video, so it
                    # is done only where its answer changes something: an in-flight
                    # state. For 'archived' and the review holds -- 2,886 of the
                    # 2,899 rows -- the file lives on the NAS, every node can reach
                    # it, and the row exists only to stop this node redoing
                    # finished work, so searching would add minutes to startup and
                    # decide nothing.
                    #
                    # For a working state the file has to be in a WORKING folder.
                    # A hit in the archive would mean the video is finished, which
                    # is the opposite of what the remote state claims -- adopting
                    # 'dlc_complete' on the strength of an archived copy would send
                    # the stager to move files out of the archive.
                    claims_in_flight = remote_state in NODE_LOCAL_STATES
                    local_file = locate_video_file(
                        video_id, raw=remote.get('source_path'),
                        search_archive=False) if claims_in_flight else None

                    if local_file is None and claims_in_flight:
                        # Another node owns the work and holds the files. Say so
                        # once, in the row, and never pick it up here.
                        local_db.register_video(
                            video_id=video_id,
                            source_path=local_db.NO_FILE_HERE,
                            collage_id=remote.get('collage_id'),
                        )
                        local_db.mark_unresolvable(
                            video_id,
                            "no file on this node; %s has it at state '%s'" % (
                                remote.get('hostname') or 'another node', remote_state))
                        stats['videos_elsewhere'] += 1
                        continue

                    local_db.register_video(
                        video_id=video_id,
                        source_path=(str(local_file) if local_file
                                     else local_db.NO_FILE_HERE),
                        collage_id=remote.get('collage_id'),
                    )
                    if remote_state != 'discovered':
                        local_db.force_state(
                            video_id, remote_state,
                            reason="cross-node recovery from %s" % (
                                remote.get('hostname') or '?'))
                    stats['videos_recovered'] += 1
                    logger.debug(f"Recovered video {video_id} as {remote_state}")
                except Exception as e:
                    logger.debug(f"Could not recover video {video_id}: {e}")
            else:
                # Already in local DB — only advance state
                local_state = local_row['state']
                local_idx = _state_index(local_state, VIDEO_STATE_ORDER)
                remote_idx = _state_index(remote_state, VIDEO_STATE_ORDER)

                if remote_idx > local_idx and local_state != 'failed':
                    try:
                        local_db.force_state(video_id, remote_state)
                        stats['videos_advanced'] += 1
                        logger.debug(f"Advanced video {video_id}: {local_state} -> {remote_state}")
                    except Exception as e:
                        logger.debug(f"Could not advance video {video_id}: {e}")

        # --- Recover collages from pipeline_collages ---
        try:
            all_collages = self.get_all_collage_states()
        except Exception as e:
            logger.warning(f"Could not read pipeline_collages: {e}")
            all_collages = {}

        for filename, remote in all_collages.items():
            remote_state = remote.get('state', 'discovered')
            if not local_db.collage_exists(filename):
                # pipeline_collages records who claimed the collage, not where it
                # is. The previous code put the HOSTNAME in source_path, which
                # _process_collage passes to Path().exists() -- always False, so
                # every recovered collage failed on the first crop attempt. Worse,
                # the row's mere existence made discover_new_collages skip the
                # file, so a collage sitting in the intake folder could never get
                # back in.
                #
                # A collage this node cannot see is not registered at all. The
                # intake scan will register it properly, with a real path, the
                # moment it can see it.
                source = _collage_source(filename)
                if source is None:
                    stats['collages_not_here'] += 1
                    logger.debug(
                        f"Collage {filename} is {remote_state} on "
                        f"{remote.get('hostname', '?')} and is not visible here; "
                        f"leaving it for the intake scan")
                    continue
                try:
                    local_db.register_collage(
                        filename=filename,
                        source_path=str(source),
                    )
                    if remote_state != 'discovered':
                        local_db.force_collage_state(filename, remote_state)
                    stats['collages_recovered'] += 1
                    logger.debug(f"Recovered collage {filename} as {remote_state}")
                except Exception as e:
                    logger.debug(f"Could not recover collage {filename}: {e}")

        # --- Cross-reference reach_data (mousedb) ---
        try:
            mousedb_videos = self._get_mousedb_video_names()
        except Exception as e:
            logger.debug(f"MouseDB cross-reference skipped: {e}")
            mousedb_videos = set()

        for video_name in mousedb_videos:
            try:
                local_row = local_db.get_video(video_name)
            except Exception:
                local_row = None

            if local_row is None:
                # Video in mousedb but not in local DB. Its reach data is in the
                # central database, so it is finished -- the row exists purely to
                # stop this node redoing it. 'archived' is truthful whether or not
                # the file is on this machine, because the archive is on the NAS
                # and every node can reach it; the placeholder path that used to
                # be written here ('recovered_from_mousedb') was not, and four
                # handlers passed it to Path().
                try:
                    # No file search here on purpose. This row's only job is to
                    # say "finished, do not redo"; nothing reads its path, and the
                    # reprocess route finds a video's files by searching the
                    # archive itself. Searching here would cost a few network
                    # stats for each of ~2,600 videos on a cold node's first
                    # start, to fill in a column no code consults.
                    local_db.register_video(
                        video_id=video_name,
                        source_path=local_db.NO_FILE_HERE,
                    )
                    local_db.force_state(
                        video_name, 'archived',
                        reason="reach data already in connectome.db")
                    stats['mousedb_confirmed'] += 1
                except Exception as e:
                    logger.debug(f"Could not register mousedb video {video_name}: {e}")
            else:
                # In local DB — if stuck before processed, advance to archived
                local_state = local_row['state']
                local_idx = _state_index(local_state, VIDEO_STATE_ORDER)
                archived_idx = _state_index('archived', VIDEO_STATE_ORDER)

                if local_idx < archived_idx and local_state != 'failed':
                    try:
                        local_db.force_state(video_name, 'archived')
                        stats['mousedb_confirmed'] += 1
                        logger.debug(f"MouseDB confirmed {video_name}: {local_state} -> archived")
                    except Exception as e:
                        logger.debug(f"Could not advance mousedb video {video_name}: {e}")

        if any(v > 0 for v in stats.values()):
            logger.info(
                f"Startup recovery: {stats['videos_recovered']} videos recovered, "
                f"{stats['videos_advanced']} advanced, "
                f"{stats['videos_elsewhere']} left to the node that holds them, "
                f"{stats['collages_recovered']} collages recovered, "
                f"{stats['mousedb_confirmed']} confirmed from mousedb"
            )

        return stats

    def _get_mousedb_video_names(self) -> set:
        """DISTINCT video_name from an OPTIONAL external central database's
        reach_data table (mousereach.config.central_db_path). Empty set when
        no such database is configured -- recovery then relies on disk alone."""
        try:
            from mousereach.config import central_db_path
            cdb = central_db_path()
        except Exception:
            cdb = None
        if cdb is None or not Path(cdb).exists():
            return set()
        ext = create_engine(f"sqlite:///{cdb}")
        with ext.connect() as conn:
            # Check if reach_data table exists
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='reach_data'"
            ))
            if result.fetchone() is None:
                return set()

            rows = conn.execute(text(
                "SELECT DISTINCT video_name FROM reach_data"
            )).fetchall()
            return {row[0] for row in rows}
