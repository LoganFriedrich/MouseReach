"""Videos the pipeline has stopped working on, and that wait for a person.

WHY THIS EXISTS
---------------
A video whose row is 'failed' (a step went wrong and its automatic retries
are used up) or 'unresolvable' (this machine has no file for it) is in no
review queue. The Pipeline Dashboard finds videos by the FOLDERS they sit
in, so a stopped video whose file is in no scanned folder -- or whose file
sits in the working folder looking like any other -- was on no screen with
its reason. Only a terminal command on the machine that holds the row said
why, and only to someone who knew to ask.

This module is the one reader of those rows for every screen:

  * ``read_stopped`` -- every stopped video with its reason and since when,
    read from a COPY of the watcher database (``snapshot_db``), never from
    the live file: the dashboard must not hold a read on the database the
    watcher writes to. Also lists a held video whose re-pose run stopped
    part way (its bundle parked in Repose_Queue/_parked/, no request out,
    its queue empty) -- in no queue either, with a row that says it is.
  * ``rerun`` -- the Re-run action, through the existing reset paths only:
    a failed row through ``WatcherDB.reset_failed`` (what
    mousereach-watch-reprocess does), an unresolvable row through
    ``retry_unresolvable_row`` (what mousereach-watch-unresolvable --retry
    does). It reads the LIVE row first: the watcher retries failed videos
    by itself, so what a table showed may already have moved on.
  * ``counts`` -- how many rows are failed / unresolvable, for the node's
    status report on the shared drive (watcher/node_status.py), so the
    processing server's dashboard can see a graphics-card PC's stopped
    videos without a terminal on that PC.
  * ``merge_row_states`` -- the dashboard's folder view plus the row states,
    so its "Failed" filter shows every failed video, not only the ones the
    folders hold.

Nothing here moves or deletes a file, and nothing writes to the share.
ASCII-only text (Windows consoles cannot print Unicode).
"""
from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

FAILED = "failed"
UNRESOLVABLE = "unresolvable"
PARKED_STOPPED = "parked_repose_stopped"     # a pseudo-state, never written to a row
STOPPED_STATES = (FAILED, UNRESOLVABLE)

# What a person does, by state. Shown beside each video.
WHAT_TO_DO = {
    FAILED: ("Fix the cause named in the reason, then select it and press Re-run "
             "(it restarts at the step its files allow, with its automatic retries "
             "given back)."),
    UNRESOLVABLE: ("This machine has no file for it -- usually another machine holds "
                   "it, or a person set it aside on purpose (the reason says which). "
                   "Re-run takes it back only when its file is on this machine."),
    PARKED_STOPPED: ("Its bundle was taken out of its review queue for a new pose, "
                     "but the re-pose run stopped part way and no request went out. "
                     "Run mousereach-repose-videos again for this video to finish it."),
}


def snapshot_db(db_path) -> Path:
    """A consistent copy of the watcher database in a temporary file (SQLite's
    own online backup, which reads the live file once, briefly, and never
    writes to it). The caller deletes it (``read_stopped`` does)."""
    fd, tmp = tempfile.mkstemp(prefix="watcher_copy_", suffix=".db")
    os.close(fd)
    try:
        src = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True,
                              timeout=30)
        try:
            dst = sqlite3.connect(tmp)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
    except Exception:
        try:
            os.unlink(tmp)                  # never leave a half copy behind
        except OSError:
            pass
        raise
    return Path(tmp)


def _rows_from(conn) -> List[dict]:
    conn.row_factory = sqlite3.Row
    cols = {r[1] for r in conn.execute("PRAGMA table_info(videos)")}
    want = [c for c in ("video_id", "state", "error_message", "error_count",
                        "last_error_at", "updated_at") if c in cols]
    q = ("SELECT %s FROM videos WHERE state IN (%s)"
         % (", ".join(want), ", ".join("?" for _ in STOPPED_STATES)))
    return [dict(r) for r in conn.execute(q, STOPPED_STATES)]


def _held_states(conn) -> Dict[str, str]:
    conn.row_factory = sqlite3.Row
    return {r["video_id"]: r["state"] for r in conn.execute(
        "SELECT video_id, state FROM videos WHERE state IN ('triage', 'deep_review')")}


def _parked_stopped(held: Dict[str, str]) -> List[dict]:
    """Held rows whose bundle is parked for a re-pose with no request out and
    no bundle back in their queue: a re-pose run that stopped part way."""
    out = []
    try:
        from mousereach.config import Paths
        from mousereach.watcher import repose
        from mousereach.watcher.repose_videos import parked_videos
        qdir = repose.repose_queue_dir()
        if not qdir:
            return out
        for vid in parked_videos():
            state = held.get(vid)
            if state is None:
                continue
            if (repose.queue_path(vid, qdir).exists()
                    or repose.inflight_path(vid, qdir).exists()):
                continue
            q = getattr(Paths, "DEEP_REVIEW" if state == "deep_review" else "TRIAGE_REVIEW", None)
            if q and (Path(q) / vid).is_dir():
                continue                    # a leftover beside a live bundle, not stopped
            out.append({"video_id": vid, "state": PARKED_STOPPED, "row_state": state,
                        "error_message": ("row says '%s', but its bundle is parked in %s "
                                          "and no re-pose request is out"
                                          % (state, Path(qdir) / "_parked" / vid)),
                        "error_count": None, "last_error_at": None, "updated_at": None})
    except Exception as e:
        logger.debug("could not read the parked re-pose folders: %s", e)
    return out


def read_stopped(db_path=None, db=None) -> List[dict]:
    """Every stopped video: ``{video_id, state, reason, since, error_count,
    what_to_do}``, failed first, then a parked re-pose that stopped, then
    unresolvable; newest first within each. Read from a copy of the database
    at ``db_path`` (or, given ``db``, from a copy of ``db.db_path``).
    Raises when the database cannot be read -- the caller shows that."""
    path = Path(db_path) if db_path is not None else Path(db.db_path)
    copy = snapshot_db(path)
    try:
        conn = sqlite3.connect(str(copy))
        try:
            rows = _rows_from(conn)
            held = _held_states(conn)
        finally:
            conn.close()
    finally:
        try:
            copy.unlink()
        except OSError:
            pass
    rows += _parked_stopped(held)
    out = []
    for r in rows:
        state = r.get("state")
        out.append({
            "video_id": r.get("video_id"),
            "state": state,
            "row_state": r.get("row_state", state),
            "reason": (r.get("error_message") or "no reason recorded").strip(),
            "since": r.get("last_error_at") or r.get("updated_at") or "",
            "error_count": r.get("error_count"),
            "what_to_do": WHAT_TO_DO.get(state, ""),
        })
    # Stable sorts, last key first: by state group, newest first within it
    # (ISO times sort as text), then by id.
    order = {FAILED: 0, PARKED_STOPPED: 1, UNRESOLVABLE: 2}
    out.sort(key=lambda r: r["video_id"] or "")
    out.sort(key=lambda r: r["since"] or "", reverse=True)
    out.sort(key=lambda r: order.get(r["state"], 9))
    return out


def counts(db) -> Dict[str, int]:
    """``{"failed": n, "unresolvable": m}`` from this machine's database
    (one short COUNT; raises when it cannot be read)."""
    with db._lock:
        conn = db._get_connection()
        try:
            got = dict(conn.execute(
                "SELECT state, COUNT(*) FROM videos WHERE state IN (?, ?) GROUP BY state",
                STOPPED_STATES).fetchall())
        finally:
            conn.close()
    return {s: int(got.get(s, 0)) for s in STOPPED_STATES}


# --------------------------------------------------------------- the Re-run

def filed_results_present(video_id: str) -> bool:
    """True when the video's features file and processing manifest are both
    in its filed folder -- the same test the watcher uses to call a video
    filed (ProcessingOrchestrator._filed_results_present). An unresolvable
    video with filed results is re-run from them (the dashboard's 'outdated'
    mark), not through the reset paths here. Never raises."""
    try:
        from mousereach.archive.core import get_archive_destination
        d = Path(get_archive_destination(video_id))
        return ((d / f"{video_id}_features.json").is_file()
                and (d / f"{video_id}_processing_manifest.json").is_file())
    except Exception:
        return False


def retry_unresolvable_row(db, row: dict, *, server: bool, dry_run: bool) -> Tuple[str, str]:
    """mousereach-watch-unresolvable --retry for ONE row (the command loops
    over this; the dashboard's Re-run calls it for the selected video).
    Returns ``(outcome, message)``: outcome is 'server' (a processing server
    changes nothing: its watcher takes a posed video back by itself),
    'queued' (in this node's DLC queue -> 'dlc_queued'), 'shared' (left at
    the top of the shared singles folder for the intake), 'waiting' (already
    waiting for the intake), 'elsewhere' (a copy this node does not work
    from) or 'none' (no copy found). The command's WHY is in cli.py."""
    from mousereach.config import Paths
    from mousereach.watcher import single_claim
    from mousereach.watcher.locate import locate_video_file

    video_id = row["video_id"]
    if server:
        return "server", (f"{video_id}: this is a processing server; nothing is forced. "
                          f"Its watcher takes the video back by itself once the video "
                          f"and its pose are in the posed folder (Processing/Posed).")
    queue = Path(Paths.DLC_QUEUE) if Paths.DLC_QUEUE else None
    local = queue / f"{video_id}.mp4" if queue else None
    try:
        in_queue = local is not None and local.is_file()
    except OSError:
        in_queue = False
    if in_queue:
        if not dry_run:
            db.force_state(video_id, 'dlc_queued',
                           reason="a person's retry: the file is in this node's DLC queue",
                           source_path=str(local), current_path=str(local),
                           error_message=None)
        return "queued", f"{video_id}: in this node's DLC queue ({local}) -> dlc_queued"
    found = locate_video_file(video_id, search_archive=True, search_staging=True)
    if found is None:
        return "none", f"{video_id}: no copy of its file was found"
    if single_claim.in_front_door(found):
        keep_reasons = ((single_claim.DUPLICATE_OF_CLAIM_REASON,)
                        + single_claim.REDRIVE_REASONS)
        reason = row.get('error_message') or ''
        if reason.startswith(keep_reasons):
            return "waiting", (f"{video_id}: in the shared singles folder; already waiting "
                               f"for the intake ({reason[:60]}) -- left as it is")
        if not dry_run:
            db.mark_unresolvable(
                video_id, f"{single_claim.SHARED_REASON}: put back for the intake "
                          f"by a person's retry ({found})")
        return "shared", (f"{video_id}: in the shared singles folder ({found}) -> left for "
                          f"the intake to claim it")
    return "elsewhere", (f"{video_id}: a copy is at {found}, which is not this node's "
                         f"(filed, posed for the server, or another node's claim) -- "
                         f"left as it is")


def rerun(db, video_id: str, from_step: Optional[str] = None,
          server: Optional[bool] = None) -> Tuple[bool, str]:
    """The Re-run action for one stopped video. ``(changed, message)``.

    Reads the LIVE row first and acts on what it says now. A failed row goes
    through ``reset_failed``: with ``from_step`` None the restart step is
    read from the files this machine holds; when they are not here a
    ValueError names the --from-step choices, and the message says
    "Choose the step" so the caller can ask (WatcherDB.RESET_STEP_CHOICES).
    An unresolvable row goes through ``retry_unresolvable_row``."""
    row = db.get_video(video_id)
    if row is None:
        return False, f"{video_id}: this machine's database has no row for it."
    state = row.get("state")
    if state == FAILED:
        try:
            new = db.reset_failed(video_id, to_state=from_step)
        except ValueError as e:
            return False, f"Choose the step to restart it from. {e}"
        if not new:
            now = (db.get_video(video_id) or row).get("state")
            return False, f"{video_id} was not changed: it is now '{now}', not failed."
        return True, f"{video_id} -> {new} (retries given back); the watcher picks it up."
    if state == UNRESOLVABLE:
        if server is None:
            try:
                from mousereach.config import WatcherConfig
                server = WatcherConfig.load().mode == 'processing_server'
            except Exception as e:
                return False, f"could not read this machine's watcher settings: {e}"
        outcome, msg = retry_unresolvable_row(db, row, server=server, dry_run=False)
        return outcome in ("queued", "shared"), msg
    if state in ("triage", "deep_review"):
        # A stopped re-pose: the row still says triage/deep_review while the
        # bundle sits in the parked folder. There is no reset path for it;
        # the re-pose command is the action, so say that, not "not stopped".
        if _parked_stopped({video_id: state}):
            return False, f"{video_id}: {WHAT_TO_DO[PARKED_STOPPED]}"
    return False, (f"{video_id} is '{state}' now, not stopped; nothing to re-run "
                   f"(the watcher may have retried it already).")


# --------------------------------------------------------- the folder view

def merge_row_states(files: Dict[str, dict], stopped: Iterable[dict]) -> Dict[str, dict]:
    """The dashboard's folder view (``{stem: info}``) with each stopped
    video's row state added as ``row_state``. A FAILED video is shown as
    failed wherever its file sits (the row is the word that counts: it waits
    for a person); a stopped video the folders do not hold at all is added,
    so it can be selected and re-run. An unresolvable video the folders DO
    hold keeps its folder stage (another machine usually has it in hand)."""
    out = dict(files)
    for r in stopped or ():
        vid, state = r.get("video_id"), r.get("state")
        if not vid or state not in STOPPED_STATES + (PARKED_STOPPED,):
            continue
        info = out.get(vid)
        if info is None:
            stage = r.get("row_state") if state == PARKED_STOPPED else state
            out[vid] = {
                "locations": [{"stage": stage,
                               "path": "(no file in a scanned folder; watcher database only)"}],
                "versions": {}, "timestamps": {}, "ground_truths": [],
                "status": "failed" if state != UNRESOLVABLE else "needs_review",
                "current_stage": state, "row_state": state,
                "metadata": {"state": state, "reason": r.get("reason")},
                "dlc_status": "pending", "seg_status": "pending",
                "reach_status": "pending", "outcome_status": "pending",
                "archive_ready": False, "review_status": "none",
                "tray_type": None, "tray_supported": True,
            }
            continue
        info = dict(info, row_state=state)
        info["metadata"] = dict(info.get("metadata") or {}, reason=r.get("reason"))
        if state in (FAILED, PARKED_STOPPED):
            # The row is the word that counts for these: the folders say
            # "working" or "parked, waiting for a pose", which is not true.
            info["current_stage"] = state
            info["status"] = "failed"
        out[vid] = info
    return out


def matches(info: dict, states) -> bool:
    """Whether a dashboard row belongs under a stage filter: by the folder
    stage, or by the row state ``merge_row_states`` added."""
    return info.get("current_stage") in states or info.get("row_state") in states
