"""
mousereach.watcher.coordination - Cross-PC pipeline coordination via connectome.db.

Uses MouseReach's own shared database (<NAS root>/watcher_central.db) as a
shared coordination layer. Each DLC PC syncs its pipeline state here so that:

1. On startup, any PC can recover its local watcher.db from shared state
2. At runtime, collage claims prevent duplicate cropping across PCs
3. Video states are visible across all PCs for monitoring

The local watcher.db remains the primary data store for speed. State SYNC
(sync_video_state, update_collage_state, recovery) is best-effort for the
CALLER: these methods raise, and the orchestrator logs and carries on, so
NAS unavailability never blocks posing or processing of work a node already
holds.

The collage CLAIM is the exception, and deliberately fails closed:
try_claim_collage raises on any database error instead of guessing, and a
node that cannot obtain a claim does not crop. WHY: a crop is ~14 GPU-minutes
per child and its children are then posed, processed and archived; a node
that crops without a claim duplicates all of that downstream of every other
node that did the same. Skipping one poll costs nothing -- the collage is
still in the intake folder next poll.
"""

import shutil
import logging
import socket
from pathlib import Path
from datetime import datetime, timedelta
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


def _is_live_local_work(local_row) -> bool:
    """A local row in a working state whose recorded file is really on this
    node is this node's LIVE work, not stale bookkeeping to be advanced.

    WHY: every re-posed video was archived once, so its central record (and
    its name in the reach table) says 'archived'. Recovery used to advance
    the local row to match, which turned a video queued for a new pose back
    into 'archived' on the first restart, with its copied mp4 orphaned in
    DLC_Queue (2026-09-12). Disk truth wins, as it does in the archive step.
    """
    try:
        state = (local_row or {}).get('state')
        if state == 'outdated':
            # This node's own verdict about the archive (and any hand-mark on
            # it) -- never a thing another node's record may overwrite.
            return True
        if state not in NODE_LOCAL_STATES:
            return False
        from pathlib import Path
        raw = (local_row or {}).get('current_path')
        if not raw or not Path(raw).is_file():
            return False
        # A file on the shared drive is not this node's working copy.
        from mousereach.config import Paths
        nas = Paths.NAS_ROOT
        if nas:
            try:
                Path(raw).resolve().relative_to(Path(nas).resolve())
                return False
            except ValueError:
                pass
        return True
    except Exception:
        return False


COLLAGE_STATE_ORDER = [
    'discovered', 'quarantined', 'validated', 'stable', 'cropping', 'cropped', 'archived',
]

# A collage claim left in 'cropping' this long by another host is taken over.
# WHY: claims used to be permanent, so a node that claimed a collage and then
# crashed, lost the share, or failed the crop without releasing blocked that
# collage for every node forever. A crop takes minutes to an hour or two, so a
# day of silence means the claimant is not coming back; the margin also swamps
# any clock or timezone difference between nodes (claimed_at is each node's
# local time). Same value, and the same reasoning, as repose.STALE_S for
# re-pose requests nobody heartbeats.
COLLAGE_CLAIM_STALE_S = 24 * 3600

# Collage claim states that mean the crop FINISHED: its children exist and
# have been handed on. Such a claim is never taken over and never released --
# reopening it would crop and pose every child a second time.
_COLLAGE_DONE_STATES = ('cropped', 'archived')

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


def _parse_claimed_at(value) -> Optional[datetime]:
    """claimed_at as written by _now() (naive local ISO 8601), or None when
    the column holds something else. An unreadable timestamp is treated as
    NOT stale by the caller: a claim is only ever taken over on evidence."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


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
    Database and driver errors RAISE from every method; it is the caller that
    decides what a failure means. For state sync the orchestrator logs and
    carries on (best-effort). For the collage claim it must not crop: see the
    module docstring. (recover_local_db is the one method that absorbs errors
    itself, per row, so one bad record cannot stop a node starting.)
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
        """Create pipeline_videos and pipeline_collages tables if needed.

        Raises on any failure (sqlalchemy missing, no shared root configured,
        share unreachable, file not a database) and stays un-ensured, so the
        next call tries again. WHY raise: the orchestrator's startup calls this
        inside a try that leaves ``self.coordinator`` None on failure, and
        every claim-dependent step must be able to see that coordination is
        unavailable -- a swallowed failure here is how crops ran unclaimed.
        """
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

        Returns True when ``hostname`` holds the claim (it just took it, it
        already held it from an earlier run, or it took over a stale one) and
        False when another host holds a live claim.

        Database and driver errors RAISE -- they are never turned into True
        or False. WHY: True on an error is how every node cropped the same
        collage whenever the share blinked, and False would make an outage
        look like another node's work. The caller must treat a raise as "do
        not crop this poll".

        First writer wins through INSERT OR IGNORE (SQLite serialises the
        insert). A claim another host has left in 'cropping' for longer than
        COLLAGE_CLAIM_STALE_S is taken over by a CONDITIONAL update that names
        the old holder and the old state, so when two hosts race for the same
        stale claim exactly one update matches a row. A finished claim
        ('cropped'/'archived') is never taken over.

        Two further rules keep a takeover from re-cropping work that exists:

          * The holder re-using its own 'cropping' claim RESTARTS claimed_at
            (conditionally, so a host that lost the claim in between does not
            get it back). WHY: age is the only staleness evidence. A holder
            that crashed, stayed down a day and then cropped again under its
            old timestamp was cropping behind a claim every other host read as
            abandoned -- a second host took it over and both cropped.
          * A stale claim is NOT taken over while any child of the collage is
            on record in pipeline_videos (by collage_id, or by the child names
            the collage filename implies). WHY: a stale 'cropping' row does
            not mean nothing was cropped. The holder may have queued children
            and then failed, or finished and failed only to record 'cropped'.
            Its children then sit in its DLC queue, in Posed or on the
            processing server, where the taker's finished-work check cannot
            see them, and a takeover would crop and pose them all again
            (~14 GPU-minutes each). Such a collage waits for its holder, or a
            person; a WARNING names it each time a host asks.
        """
        self.ensure_tables()
        # Three passes cover a row that changes between our read and our write
        # (released, taken over by another host); the first pass normally decides.
        for _attempt in range(3):
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "INSERT OR IGNORE INTO pipeline_collages "
                    "(filename, hostname, state, claimed_at) "
                    "VALUES (:filename, :hostname, 'cropping', :claimed_at)"
                ), {'filename': filename, 'hostname': hostname, 'claimed_at': _now()})
                conn.commit()

                if result.rowcount > 0:
                    return True

                # Row already existed -- check who owns it
                row = conn.execute(text(
                    "SELECT hostname, state, claimed_at FROM pipeline_collages "
                    "WHERE filename = :filename"
                ), {'filename': filename}).fetchone()

            if row is None:
                continue                  # released in between; insert again

            holder, state, claimed_at = row[0], row[1], row[2]
            if holder == hostname:
                # We already claimed it (e.g., from a previous run).
                if state != 'cropping':
                    return True
                if self._refresh_own_claim(filename, hostname):
                    return True
                continue                  # taken over in between; look again

            if state == 'cropping':
                when = _parse_claimed_at(claimed_at)
                cutoff = datetime.now() - timedelta(seconds=COLLAGE_CLAIM_STALE_S)
                if when is not None and when < cutoff:
                    on_record = self.collage_children_on_record(filename)
                    if on_record:
                        sample = ", ".join(
                            f"{r['video_id']} ({r.get('state')} on {r.get('hostname')})"
                            for r in on_record[:3])
                        logger.warning(
                            f"Collage {filename}: {holder} has held its crop claim in "
                            f"'cropping' for more than {COLLAGE_CLAIM_STALE_S // 3600} h, "
                            f"but {len(on_record)} of its child video(s) are already on "
                            f"record ({sample}), so the claim is NOT taken over -- a "
                            f"second crop would pose them again. {holder} finishes it "
                            f"when its watcher runs again; if {holder} is gone for good, "
                            f"a person decides what is left to crop.")
                        return False
                    if self._take_over_stale_claim(filename, holder, hostname,
                                                   cutoff=cutoff):
                        return True
                    continue              # released or taken meanwhile; look again

            logger.info(f"Collage {filename} already claimed by {holder} ({state})")
            return False

        logger.info(f"Collage {filename}: claim row kept changing under this host; "
                    f"not claimed this poll")
        return False

    def _refresh_own_claim(self, filename: str, hostname: str) -> bool:
        """Restart the clock on a 'cropping' claim ``hostname`` still holds.
        True only if this update matched the row (the WHERE names the holder,
        so a claim another host took over in between is left alone)."""
        with self.engine.connect() as conn:
            result = conn.execute(text(
                "UPDATE pipeline_collages SET claimed_at = :now "
                "WHERE filename = :filename AND hostname = :hostname "
                "AND state = 'cropping'"
            ), {'now': _now(), 'filename': filename, 'hostname': hostname})
            conn.commit()
        return result.rowcount == 1

    def collage_children_on_record(self, filename: str) -> List[dict]:
        """pipeline_videos rows for this collage's children:
        ``[{video_id, hostname, state}]``, [] when there are none.

        A child is matched by collage_id, or by the child names the collage
        filename implies (the cropper names each child from it), so a row
        synced without its collage_id still counts. Raises on database errors
        -- the caller is deciding whether a claim may be taken over, and an
        unreadable table is no evidence that nothing was cropped.
        """
        self.ensure_tables()
        stems: List[str] = []
        try:
            from mousereach.video_prep.core.collage_provenance import expected_offspring
            stems = [c['offspring_stem'] for c in expected_offspring(filename)
                     if not c.get('blank') and c.get('offspring_stem')]
        except Exception as e:
            logger.debug(f"could not derive the children of {filename}: {e}")
        params = {'filename': filename}
        clause = "collage_id = :filename"
        if stems:
            names = []
            for i, stem in enumerate(stems):
                params[f"s{i}"] = stem
                names.append(f":s{i}")
            clause += f" OR video_id IN ({', '.join(names)})"
        with self.engine.connect() as conn:
            rows = conn.execute(text(
                f"SELECT video_id, hostname, state FROM pipeline_videos WHERE {clause}"
            ), params).fetchall()
        return [{'video_id': r[0], 'hostname': r[1], 'state': r[2]} for r in rows]

    def get_collage_claim(self, filename: str) -> Optional[dict]:
        """The pipeline_collages row for one collage, or None. Raises on errors.
        One row, so a node asking who holds a collage does not read the whole
        table over the share."""
        self.ensure_tables()
        with self.engine.connect() as conn:
            row = conn.execute(text(
                "SELECT * FROM pipeline_collages WHERE filename = :filename"
            ), {'filename': filename}).fetchone()
        return dict(row._mapping) if row is not None else None

    def _take_over_stale_claim(self, filename: str, old_hostname: str,
                               new_hostname: str,
                               cutoff: Optional[datetime] = None) -> bool:
        """Move a stale 'cropping' claim from ``old_hostname`` to
        ``new_hostname``. True only if THIS update changed the row.

        The WHERE clause re-checks everything the decision rested on (holder,
        state, age) inside the single UPDATE statement, so SQLite's write lock
        makes it all-or-nothing: of two hosts that both read the stale row,
        the first update rewrites the holder and the second matches nothing.
        claimed_at is compared as text, which orders correctly because every
        row is written by _now() in the same ISO 8601 form as the cutoff.
        """
        if cutoff is None:
            cutoff = datetime.now() - timedelta(seconds=COLLAGE_CLAIM_STALE_S)
        with self.engine.connect() as conn:
            result = conn.execute(text(
                "UPDATE pipeline_collages SET hostname = :new, claimed_at = :now "
                "WHERE filename = :filename AND hostname = :old "
                "AND state = 'cropping' AND claimed_at < :cutoff"
            ), {'new': new_hostname, 'now': _now(), 'filename': filename,
                'old': old_hostname, 'cutoff': cutoff.isoformat()})
            conn.commit()
        if result.rowcount == 1:
            logger.warning(
                f"Collage {filename}: took over the crop claim {old_hostname} left in "
                f"'cropping' for more than {COLLAGE_CLAIM_STALE_S // 3600} h "
                f"(that crop never finished); {new_hostname} will crop it")
            return True
        return False

    def release_collage_claim(self, filename: str, hostname: str) -> bool:
        """Give up this host's claim on a collage whose crop FAILED, so any
        node (this one included) can claim and crop it again.

        Deletes the row only when ``hostname`` holds it and the crop has not
        finished. True if a row was released. Raises on database errors.

        WHY: a claim that outlives a failed crop blocks the collage for every
        node until the stale takeover a day later. WHY never a finished claim
        ('cropped', or the later 'archived'): its children already exist and
        have been handed on; reopening it would crop and pose them all again.
        WHY only the holder: another host's claim is its live work.
        """
        self.ensure_tables()
        with self.engine.connect() as conn:
            result = conn.execute(text(
                "DELETE FROM pipeline_collages "
                "WHERE filename = :filename AND hostname = :hostname "
                "AND state NOT IN ('cropped', 'archived')"
            ), {'filename': filename, 'hostname': hostname})
            conn.commit()
        released = result.rowcount > 0
        if released:
            logger.info(f"Collage {filename}: claim released by {hostname} after a failed crop")
        return released

    def update_collage_state(self, filename: str, state: str,
                             only_if_held_by: Optional[str] = None, **kwargs) -> bool:
        """Update collage state after cropping completes. True if a row changed.

        ``only_if_held_by``: change the row only while that host holds the
        claim. WHY: a node whose claim was taken over while it cropped must not
        stamp 'cropped' over the new holder's live claim -- the False result is
        how it learns the collage may have been cropped twice.
        """
        self.ensure_tables()
        fields = {'state': state}
        if state in ('cropped', 'archived'):
            fields['completed_at'] = _now()
        fields.update(kwargs)

        set_clause = ', '.join(f'{k} = :{k}' for k in fields.keys())
        fields['filename'] = filename
        where = "filename = :filename"
        if only_if_held_by is not None:
            fields['_holder'] = only_if_held_by
            where += " AND hostname = :_holder"

        with self.engine.connect() as conn:
            result = conn.execute(text(
                f"UPDATE pipeline_collages SET {set_clause} WHERE {where}"
            ), fields)
            conn.commit()
        return result.rowcount > 0

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
                    #
                    # search_staging=False: recovery runs only on GPU nodes (only
                    # DLCOrchestrator builds a coordinator), and a copy in
                    # Processing/Posed is the processing server's intake, not
                    # this node's in-flight file. Counting it made a fresh node
                    # adopt another node's hand-off as its own 'dlc_complete'
                    # row (watcher/locate.py docstring).
                    claims_in_flight = remote_state in NODE_LOCAL_STATES
                    local_file = locate_video_file(
                        video_id, raw=remote.get('source_path'),
                        search_archive=False,
                        search_staging=False) if claims_in_flight else None

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

                if (remote_idx > local_idx and local_state != 'failed'
                        and not _is_live_local_work(local_row)):
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

                if (local_idx < archived_idx and local_state != 'failed'
                        and not _is_live_local_work(local_row)):
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
