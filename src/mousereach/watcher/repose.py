"""Re-pose round trip over shared storage.

WHY THIS EXISTS
---------------
When pipeline_versions.json declares a new DLC model, archived videos with no
pose from that model are marked state='outdated', reprocess_scope='full' by
the version scan. Until 2026-09-12 that was a dead end: the node that runs
the scan usually has no GPU, nothing carried the video to a node that does,
and when a GPU node was fed the video by hand and staged the new pose into
Processing/DLC_Complete, the staged-pose scan ignored it because the video
already had a row. Every such video cost one hand step on each machine.

THE SHAPE: PULL, OVER SHARED STORAGE
------------------------------------
Nodes never talk to each other; shared storage is the only channel, and no
role is assumed to exist. A lab may run one GPU PC and a drive, or a
processing server plus one or more GPU PCs; the same three steps work in
both:

  1. PUBLISH  (any node that runs the version scan, once per scan) -- for
     each outdated video that needs a NEW pose, one small JSON request in
     NAS_ROOT/Processing/Repose_Queue/<video_id>.json. Nothing is moved.
  2. CONSUME  (any GPU node, every poll, a few at a time) -- CLAIM a request
     by renaming it into Repose_Queue/.inflight/ (a rename is atomic, so two
     GPU nodes cannot both take it), copy the archived mp4 into this node's
     own local DLC_Queue, put the local row in 'dlc_queued'. The ordinary
     pose + stage steps take it from there.
  3. ADOPT    (any node that takes work from Processing/DLC_Complete) -- a
     staged pose from the declared model for a video whose row is parked
     'outdated' narrows that row's reprocess scope to 'segmentation' and
     records the staged pose, so the EXISTING reprocess path re-runs every
     post-DLC stage against it -- with the archived results folder (human
     segmentation and reviews included) copied down beside it, as any other
     partial reprocess does. Adoption closes the request.

The request file is the claim and the idempotency key: it exists (queued or
inflight) from publish until the new pose is adopted, or until a GPU node
that also processes archives the result itself. Publish never re-asks while
either file exists; a GPU node heartbeats the inflight file while it holds
the video, and a request nobody has touched for a day goes back to the
queue. The archived mp4 is COPIED, never moved. Request files are queue
bookkeeping and are deleted when the round trip closes, the same way the
review-return claim files are; the processing_log keeps the provenance.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterable, List, Optional, Set

from mousereach.config import Paths
from mousereach.watcher.transfer import safe_copy

logger = logging.getLogger(__name__)

STEP = "repose_request"            # processing_log step name
INFLIGHT = ".inflight"             # subfolder of the queue: claimed requests
STALE_S = 24 * 3600                # inflight with no heartbeat this long -> back to queue
IN_FLIGHT_STATES = ("dlc_queued", "dlc_running", "dlc_complete",
                    "processing", "processed", "archiving")
REDRIVE_STATES = ("archived", "outdated", "unresolvable", "validated", "deep_review")
REFUSE_STATES = ("crystallized", "triage", "quarantined", "discovered")
ADOPT_STATES = ("outdated", "unresolvable")
REASON_PREFIX = "re-pose request"  # mark_reason prefix on rows a GPU node re-drove


# ---------------------------------------------------------------- paths / files

def repose_queue_dir() -> Optional[Path]:
    """The shared request folder, or None when no shared root is configured."""
    d = Paths.REPOSE_QUEUE
    return Path(d) if d else None


def queue_path(video_id: str, repose_dir) -> Path:
    return Path(repose_dir) / f"{video_id}.json"


def inflight_path(video_id: str, repose_dir) -> Path:
    return Path(repose_dir) / INFLIGHT / f"{video_id}.json"


def failed_path(video_id: str, repose_dir) -> Path:
    return Path(repose_dir) / f"{video_id}.failed.json"


def read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json_atomic(p: Path, body: dict) -> None:
    """Write via a sibling temp file and rename, so a reader never sees a
    half-written request."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _unlink(p: Path) -> bool:
    try:
        Path(p).unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        logger.warning(f"could not remove {p} ({e})")
        return False


def _list_ids(folder: Path) -> Dict[str, Path]:
    """{video_id: path} for every request json directly in ``folder``."""
    out: Dict[str, Path] = {}
    try:
        for p in Path(folder).glob("*.json"):
            if p.is_file() and not p.name.endswith(".failed.json"):
                out[p.stem] = p
    except OSError:
        pass
    return out


def _once(latched: Optional[Set[str]], key: str) -> bool:
    """True the first time ``key`` is seen in ``latched`` (a set owned by the
    caller), so a condition is logged once per process, not once per poll."""
    if latched is None:
        return True
    if key in latched:
        return False
    latched.add(key)
    return True


# ---------------------------------------------------------------- pose lookups

def declared_scorer() -> str:
    """The DLC scorer pipeline_versions.json declares current, read fresh
    (the manifest module caches its copy for the life of the process)."""
    try:
        from mousereach.pipeline.versions import get_current_versions
        return (get_current_versions().get("versions") or {}).get("dlc_scorer", "") or ""
    except Exception as e:
        logger.debug(f"declared scorer unavailable: {e}")
        return ""


def scorer_of(h5: Path) -> str:
    from mousereach.pipeline.manifest import extract_dlc_model_info
    return extract_dlc_model_info(Path(h5)).get("dlc_scorer", "") or ""


def video_id_of_h5(h5: Path) -> str:
    return Path(h5).stem.split("DLC")[0].rstrip("_")


def declared_poses_in(folder: Optional[Path], declared: str) -> Dict[str, Path]:
    """{video_id: h5} for every pose file in ``folder`` from the declared
    model (any model when nothing is declared). One glob; *.part ignored."""
    out: Dict[str, Path] = {}
    if not folder:
        return out
    folder = Path(folder)
    if not folder.exists():
        return out
    try:
        for h5 in folder.glob("*DLC*.h5"):
            if not declared or scorer_of(h5) == declared:
                out.setdefault(video_id_of_h5(h5), h5)
    except OSError:
        pass
    return out


def archive_folder(video_id: str) -> Optional[Path]:
    try:
        from mousereach.archive.core import get_archive_destination
        return Path(get_archive_destination(video_id))
    except Exception as e:
        logger.debug(f"{video_id}: archive destination unresolved ({e})")
        return None


def declared_pose_in_archive(video_id: str, declared: str) -> Optional[Path]:
    """A pose from the declared model in this video's archive folder, or None.
    (The version scan looks further -- the whole Analyzed tree -- before it
    calls a video 'full'; this is the cheap re-check a request needs.)"""
    d = archive_folder(video_id)
    if d is None or not d.exists():
        return None
    try:
        for h5 in d.glob(f"{video_id}DLC*.h5"):
            if not declared or scorer_of(h5) == declared:
                return h5
    except OSError:
        pass
    return None


def archived_video(video_id: str) -> Optional[Path]:
    """The archived mp4 for this video, or None."""
    d = archive_folder(video_id)
    if d is not None:
        p = d / f"{video_id}.mp4"
        if p.is_file():
            return p
    try:
        from mousereach.watcher.locate import locate_video_file
        return locate_video_file(video_id, search_archive=True)
    except Exception as e:
        logger.debug(f"{video_id}: locate failed ({e})")
        return None


def resolve_request_video(video_id: str, body: dict) -> Optional[Path]:
    """Where the mp4 named by a request is ON THIS NODE. The request carries
    a path relative to the shared root (drive letters differ between nodes)
    and, as a fallback, the publisher's absolute path; the archive folder
    is tried in between. The file must be named <video_id>.mp4."""
    candidates: List[Path] = []
    rel = body.get("video_rel")
    if rel and Paths.NAS_ROOT:
        candidates.append(Path(Paths.NAS_ROOT) / Path(*PurePosixPath(rel).parts))
    d = archive_folder(video_id)
    if d is not None:
        candidates.append(d / f"{video_id}.mp4")
    raw = body.get("video_path")
    if raw:
        candidates.append(Path(raw))
    for p in candidates:
        try:
            if p.is_file() and p.name == f"{video_id}.mp4":
                return p
        except OSError:
            continue
    try:
        from mousereach.watcher.locate import locate_video_file
        p = locate_video_file(video_id, search_archive=True)
        if p is not None and Path(p).name == f"{video_id}.mp4":
            return Path(p)
    except Exception:
        pass
    return None


def _log(db, video_id: str, status: str, message: str) -> None:
    try:
        db.log_step(video_id, STEP, status, message=message)
    except Exception:
        pass


# ---------------------------------------------------------------- lifecycle

def close_request(video_id: str, repose_dir: Optional[Path] = None) -> bool:
    """The round trip is over for this video: drop its queue, inflight and
    failure files. True if anything was removed."""
    repose_dir = repose_dir if repose_dir is not None else repose_queue_dir()
    if not repose_dir:
        return False
    removed = False
    for p in (queue_path(video_id, repose_dir), inflight_path(video_id, repose_dir),
              failed_path(video_id, repose_dir)):
        removed = _unlink(p) or removed
    return removed


def note_failure(video_id: str, hostname: str, error: str,
                 repose_dir: Optional[Path] = None) -> bool:
    """A GPU node could not pose a requested video: say so in shared
    storage, where the requester can see it. Only for videos that were
    actually requested (an inflight or queued file exists)."""
    repose_dir = repose_dir if repose_dir is not None else repose_queue_dir()
    if not repose_dir:
        return False
    if not (inflight_path(video_id, repose_dir).exists()
            or queue_path(video_id, repose_dir).exists()):
        return False
    try:
        write_json_atomic(failed_path(video_id, repose_dir), {
            "video_id": video_id, "host": hostname, "error": str(error)[:2000],
            "failed_at": datetime.now().isoformat()})
        return True
    except OSError as e:
        logger.debug(f"{video_id}: could not write failure note ({e})")
        return False


# ---------------------------------------------------------------- 1. publish

def publish_pending(db, rows: Iterable[dict], *, hostname: str,
                    staging_dir: Optional[Path] = None,
                    repose_dir: Optional[Path] = None,
                    declared: Optional[str] = None,
                    latched: Optional[Set[str]] = None) -> Dict[str, list]:
    """Ask GPU nodes, over shared storage, for a new pose for every row in
    ``rows`` (this node's outdated videos with reprocess_scope 'full').
    Runs once per version scan. Builds its index of the queue and the
    staging folder once, then touches shared storage only for videos that
    need a request written, rewritten, or withdrawn.

    Also sweeps this node's own bookkeeping: withdraws a queued request whose
    row is no longer outdated/full or whose declared model changed, and
    returns an inflight request nobody has heartbeated for a day to the
    queue. Never raises."""
    out: Dict[str, list] = {"published": [], "republished": [], "rewritten": [],
                            "withdrawn": [], "returned": [], "staged": [],
                            "in_archive": [], "inflight": [], "no_video": [],
                            "failed_remote": []}
    repose_dir = repose_dir if repose_dir is not None else repose_queue_dir()
    rows = list(rows)
    if not repose_dir:
        if rows and _once(latched, "no_queue"):
            logger.warning(
                "%d outdated video(s) need a new DLC pose, but no shared root is "
                "configured on this node, so no re-pose request can be published. "
                "They are held, not lost: %s%s", len(rows),
                ", ".join(r["video_id"] for r in rows[:5]),
                "" if len(rows) <= 5 else " and %d more" % (len(rows) - 5))
        return out
    repose_dir = Path(repose_dir)
    declared = declared_scorer() if declared is None else declared
    staging_dir = staging_dir if staging_dir is not None else Paths.DLC_STAGING
    try:
        repose_dir.mkdir(parents=True, exist_ok=True)
        (repose_dir / INFLIGHT).mkdir(exist_ok=True)
    except OSError as e:
        if _once(latched, "queue_unwritable"):
            logger.warning(f"re-pose queue {repose_dir} is not writable ({e}); "
                           f"requests are held until it is")
        return out

    queued = _list_ids(repose_dir)
    inflight = _list_ids(repose_dir / INFLIGHT)
    staged = declared_poses_in(staging_dir, declared)
    wanted = {r["video_id"]: r for r in rows}
    now = time.time()

    for video_id, row in wanted.items():
        if video_id in staged:
            out["staged"].append(video_id)       # adoption takes it next poll
            continue
        if declared_pose_in_archive(video_id, declared) is not None:
            out["in_archive"].append(video_id)   # next scan will narrow the scope
            continue
        fp = failed_path(video_id, repose_dir)
        if fp.exists() and _once(latched, f"failed_remote:{video_id}"):
            note = read_json(fp) or {}
            logger.warning(f"{video_id}: a GPU node ({note.get('host', '?')}) could not "
                           f"pose it: {note.get('error', '?')}. The request stays "
                           f"out; fix the cause there, or re-mark the video here.")
            _log(db, video_id, "failed_remote", f"{note.get('host', '?')}: {note.get('error', '')}")
            out["failed_remote"].append(video_id)
        if video_id in inflight:
            body = read_json(inflight[video_id]) or {}
            if declared and body.get("declared_scorer") not in ("", None, declared) \
                    and _once(latched, f"inflight_stale:{video_id}"):
                logger.warning(f"{video_id}: a GPU node is posing it for "
                               f"{body.get('declared_scorer')} but the declared model is "
                               f"now {declared}; it will be re-requested after that run")
            out["inflight"].append(video_id)
            continue
        if video_id in queued:
            body = read_json(queued[video_id]) or {}
            if declared and body.get("declared_scorer") not in ("", None, declared):
                body["declared_scorer"] = declared
                body["rewritten_at"] = datetime.now().isoformat()
                try:
                    write_json_atomic(queued[video_id], body)
                    out["rewritten"].append(video_id)
                    _log(db, video_id, "rewritten", f"declared model now {declared}")
                except OSError as e:
                    logger.warning(f"{video_id}: could not rewrite request ({e})")
            continue
        mp4 = archived_video(video_id)
        if mp4 is None:
            if _once(latched, f"no_video:{video_id}"):
                logger.warning(f"{video_id}: needs a new pose but no archived video "
                               f"file was found on this node -- no request written")
                _log(db, video_id, "skipped", "no archived video file to re-pose")
            out["no_video"].append(video_id)
            continue
        body = {
            "video_id": video_id,
            "declared_scorer": declared,
            "video_rel": _relative_to_root(mp4),
            "video_path": str(mp4),
            "requested_by": hostname,
            "requested_at": datetime.now().isoformat(),
            "reason": (f"reprocess scope 'full': no pose from the declared model "
                       f"in the archive (marked {row.get('updated_at') or '?'})"),
        }
        try:
            write_json_atomic(queue_path(video_id, repose_dir), body)
        except OSError as e:
            logger.warning(f"{video_id}: could not write re-pose request ({e})")
            continue
        republish = _once(latched, f"published:{video_id}") is False
        out["republished" if republish else "published"].append(video_id)
        _log(db, video_id, "republished" if republish else "published", f"to {repose_dir}")

    # --- sweep this node's own requests ---
    for video_id, p in queued.items():
        if video_id in wanted:
            continue
        body = read_json(p) or {}
        if body.get("requested_by") != hostname:
            continue                             # another node's request
        if _unlink(p):
            out["withdrawn"].append(video_id)
            _log(db, video_id, "withdrawn", "row is no longer outdated with scope full")
    for video_id, p in inflight.items():
        try:
            age = now - p.stat().st_mtime
        except OSError:
            continue
        if age <= STALE_S:
            continue
        body = read_json(p) or {}
        if body.get("requested_by") != hostname:
            continue
        try:
            os.replace(p, queue_path(video_id, repose_dir))
            out["returned"].append(video_id)
            logger.warning(f"{video_id}: re-pose request was claimed by "
                           f"{body.get('consumed_by', '?')} but not touched for "
                           f"{age / 3600:.0f} h; returned to the queue")
            _log(db, video_id, "returned", f"no heartbeat from {body.get('consumed_by', '?')}")
        except OSError as e:
            logger.debug(f"{video_id}: could not return stale request ({e})")

    if out["published"] or out["republished"] or out["rewritten"]:
        logger.info("Re-pose requests: %d published, %d re-published, %d rewritten "
                    "for a GPU node to pull: %s",
                    len(out["published"]), len(out["republished"]), len(out["rewritten"]),
                    ", ".join((out["published"] + out["republished"])[:5]))
    return out


def _relative_to_root(p: Path) -> Optional[str]:
    root = Paths.NAS_ROOT
    if not root:
        return None
    try:
        return PurePosixPath(*Path(p).resolve().relative_to(Path(root).resolve()).parts).as_posix()
    except (ValueError, OSError):
        try:
            return PurePosixPath(*Path(p).relative_to(Path(root)).parts).as_posix()
        except ValueError:
            return None


# ---------------------------------------------------------------- 2. consume

def _requeue_for_dlc(db, video_id: str, state: Optional[str], reason: str,
                     **fields) -> None:
    """Put a row into 'dlc_queued' by the legal transitions where the state
    machine has them (archived goes through 'outdated', which records WHY),
    force_state (logged as a bypass) only where it does not."""
    from mousereach.watcher.db import VIDEO_TRANSITIONS
    fields = dict(fields, error_message=None, reprocess_scope=None, mark_reason=reason)
    if state == "archived":
        db.update_state(video_id, "outdated", reprocess_scope="full", mark_reason=reason)
        db.update_state(video_id, "dlc_queued", **fields)
        return
    if state and "dlc_queued" in VIDEO_TRANSITIONS.get(state, []):
        db.update_state(video_id, "dlc_queued", **fields)
        return
    db.force_state(video_id, "dlc_queued", reason=reason, **fields)


def _register_single(db, video_id: str, mp4: Path) -> None:
    """Register a video this node has never seen, with parsed metadata so
    the work-priority tiers and the priority animal can see it."""
    metadata = {}
    try:
        from mousereach.watcher.validator import validate_single_filename
        result = validate_single_filename(mp4.name)
        if result.valid and result.parsed:
            metadata = {k: result.parsed[k] for k in
                        ("date", "animal_id", "experiment", "cohort", "subject", "tray_type")
                        if k in result.parsed}
    except Exception:
        pass
    db.register_video(video_id=video_id, source_path=str(mp4),
                      current_path=str(mp4), **metadata)


def _local_file_exists(video_id: str, row: dict, dlc_queue: Path) -> bool:
    try:
        from mousereach.watcher.locate import locate_video_file
        return locate_video_file(video_id, raw=row.get("current_path"),
                                 extra_dirs=[dlc_queue], search_archive=False) is not None
    except Exception:
        return False


def _in_flight_reposes(db) -> int:
    n = 0
    for state in ("dlc_queued", "dlc_running"):
        try:
            for r in db.get_videos_in_state(state):
                if (r.get("mark_reason") or "").startswith(REASON_PREFIX):
                    n += 1
        except Exception:
            pass
    return n


def heartbeat(db, *, hostname: str, repose_dir: Optional[Path] = None) -> int:
    """Touch the inflight file of every request this node consumed whose
    local row is still in flight, so the publisher knows it is alive. Returns
    how many were touched."""
    repose_dir = repose_dir if repose_dir is not None else repose_queue_dir()
    if not repose_dir:
        return 0
    n = 0
    for video_id, p in _list_ids(Path(repose_dir) / INFLIGHT).items():
        body = read_json(p) or {}
        if body.get("consumed_by") != hostname:
            continue
        try:
            row = db.get_video(video_id)
        except Exception:
            row = None
        if row and row.get("state") in IN_FLIGHT_STATES:
            try:
                os.utime(p, None)
                n += 1
            except OSError:
                pass
    return n


def consume_requests(db, *, dlc_queue: Optional[Path], hostname: str,
                     repose_dir: Optional[Path] = None,
                     declared: Optional[str] = None,
                     batch: int = 2, max_retries: int = 3,
                     on_queued: Optional[Callable[[str, Path], None]] = None,
                     latched: Optional[Set[str]] = None) -> Dict[str, int]:
    """GPU side, once per poll: pull up to ``batch`` requests into this
    node's local DLC_Queue and queue them for pose, keeping at most ``batch``
    re-poses in flight on this node. Never raises.

    The local row is inspected BEFORE the request is claimed, so a row this
    node must not touch (locked, in a human queue, failed too often) never
    holds the request away from another GPU node."""
    summary = {"queued": 0, "satisfied": 0, "completed": 0, "in_flight": 0,
               "refused": 0, "held_failed": 0, "no_video": 0, "copy_failed": 0,
               "claimed_elsewhere": 0, "bad_request": 0, "skipped_cap": 0}
    repose_dir = repose_dir if repose_dir is not None else repose_queue_dir()
    if not repose_dir or not dlc_queue:
        return summary
    repose_dir = Path(repose_dir)
    dlc_queue = Path(dlc_queue)
    queued = _list_ids(repose_dir)
    if not queued:
        return summary
    declared = declared_scorer() if declared is None else declared
    slots = max(0, int(batch) - _in_flight_reposes(db))
    if slots <= 0:
        summary["skipped_cap"] = len(queued)
        return summary

    for video_id in sorted(queued):
        if slots <= 0:
            summary["skipped_cap"] += 1
            continue
        req = queued[video_id]
        body = read_json(req)
        if body is None:
            if _once(latched, f"bad_request:{video_id}"):
                logger.warning(f"{req.name}: unreadable re-pose request; left in place")
            summary["bad_request"] += 1
            continue

        # --- look before claiming -------------------------------------
        try:
            row = db.get_video(video_id)
        except Exception:
            row = None
        state = (row or {}).get("state")
        if state in IN_FLIGHT_STATES:
            if _local_file_exists(video_id, row, dlc_queue):
                summary["in_flight"] += 1       # this node already has it going
                continue
            try:                                # a husk: heal it, then re-drive
                db.mark_unresolvable(video_id, "re-pose request found the recorded "
                                               "file missing on this node")
                state = "unresolvable"
            except Exception:
                summary["in_flight"] += 1
                continue
        if state in REFUSE_STATES or (state is not None and state not in REDRIVE_STATES
                                      and state != "failed"):
            if _once(latched, f"refused:{video_id}"):
                logger.warning(f"{video_id}: re-pose requested but the row here is "
                               f"'{state}', which this node will not re-drive; "
                               f"request left for another node or a person")
            summary["refused"] += 1
            continue
        if state == "failed" and int((row or {}).get("error_count") or 0) >= max_retries:
            if _once(latched, f"held_failed:{video_id}"):
                logger.warning(f"{video_id}: re-pose requested but it has failed "
                               f"{row.get('error_count')} time(s) here: "
                               f"{row.get('error_message')}; a person must look first")
                _log(db, video_id, "held", "failed too often on this node")
            summary["held_failed"] += 1
            continue

        # --- claim: an atomic rename, exactly one node wins ----------------
        infl = inflight_path(video_id, repose_dir)
        try:
            infl.parent.mkdir(exist_ok=True)
            os.replace(req, infl)
        except FileNotFoundError:
            summary["claimed_elsewhere"] += 1
            continue
        except OSError as e:
            logger.debug(f"{video_id}: could not claim request ({e})")
            summary["claimed_elsewhere"] += 1
            continue
        body["consumed_by"] = hostname
        body["consumed_at"] = datetime.now().isoformat()
        try:
            write_json_atomic(infl, body)
        except OSError:
            pass

        # --- already satisfied? ---------------------------------------
        local_h5s = list(dlc_queue.glob(f"{video_id}DLC*.h5")) if dlc_queue.exists() else []
        local_declared = [h for h in local_h5s if not declared or scorer_of(h) == declared]
        if declared_pose_in_archive(video_id, declared) is not None:
            close_request(video_id, repose_dir)
            logger.info(f"{video_id}: re-pose request already satisfied (declared pose "
                        f"is in the archive); request closed")
            _log(db, video_id, "satisfied", "declared pose already in archive")
            summary["satisfied"] += 1
            continue

        # --- the video file --------------------------------------------
        src = resolve_request_video(video_id, body)
        if src is None:
            try:
                os.replace(infl, req)           # let another node try
            except OSError:
                pass
            if _once(latched, f"no_video:{video_id}"):
                logger.warning(f"{video_id}: re-pose requested but no video file was "
                               f"found from this node (request names "
                               f"{body.get('video_rel') or body.get('video_path')!r})")
            summary["no_video"] += 1
            continue
        dest = dlc_queue / f"{video_id}.mp4"
        try:
            dlc_queue.mkdir(parents=True, exist_ok=True)
            same = dest.is_file() and dest.stat().st_size == src.stat().st_size
        except OSError:
            same = False
        if not same and not safe_copy(src, dest, verify=True):
            try:
                os.replace(infl, req)
            except OSError:
                pass
            logger.warning(f"{video_id}: could not copy video into DLC_Queue")
            summary["copy_failed"] += 1
            continue

        # --- an old-model pose beside it would be mistaken for the new one
        for h5 in local_h5s:
            if h5 not in local_declared:
                stem = h5.name[:-3]
                for f in dlc_queue.glob(stem + "*"):
                    if _unlink(f):
                        logger.info(f"{video_id}: removed stale {f.name} from DLC_Queue")

        reason = (f"{REASON_PREFIX} from {body.get('requested_by', '?')}: "
                  f"{body.get('reason', '')}").strip()
        try:
            if row is None:
                _register_single(db, video_id, dest)
                row = db.get_video(video_id) or {}
                state = row.get("state")
            if local_declared:
                # DLC already ran for the declared model here; nothing to pose.
                target = "dlc_complete"
                fields = dict(current_path=str(dest), source_path=str(dest),
                              dlc_output_path=str(local_declared[0]),
                              error_message=None, reprocess_scope=None,
                              mark_reason=reason)
                db.force_state(video_id, target, reason=reason + " (declared pose already local)",
                               **fields)
                summary["completed"] += 1
            else:
                _requeue_for_dlc(db, video_id, state, reason,
                                 current_path=str(dest), source_path=str(dest),
                                 dlc_output_path=None)
                summary["queued"] += 1
                slots -= 1
        except Exception as e:
            logger.error(f"{video_id}: could not queue for re-pose ({e})")
            try:
                os.replace(infl, req)
            except OSError:
                pass
            continue
        _log(db, video_id, "consumed", json.dumps(body))
        if on_queued is not None:
            try:
                on_queued(video_id, dest)
            except Exception as e:
                logger.debug(f"{video_id}: on_queued hook failed ({e})")
        logger.info(f"{video_id}: re-pose request consumed; queued for DLC on {hostname}")
    return summary


# ---------------------------------------------------------------- 3. adopt

def adopt_staged_reposes(db, staging_dir: Optional[Path], *,
                         repose_dir: Optional[Path] = None,
                         declared: Optional[str] = None,
                         latched: Optional[Set[str]] = None) -> List[str]:
    """Server side, once per poll. For every pose from the declared model in
    staging whose video has a row here:

      * row 'outdated' or 'unresolvable' -> the row becomes 'outdated' with
        reprocess_scope 'segmentation' and dlc_output_path = the staged h5.
        The existing reprocess handler then copies that pose AND the archived
        results folder down and re-runs every post-DLC stage. No state is
        forced past the state machine, and a hand-mark reason survives.
      * row 'archived' whose archived manifest names another model -> the
        same, via the legal archived -> outdated transition. New pose
        information is worth a re-run whether or not the scan has caught up.
      * row 'failed' -> named once; a person decides.

    Closes the request. Returns the adopted video ids. Never raises."""
    adopted: List[str] = []
    staged = declared_poses_in(staging_dir, declared_scorer() if declared is None else declared)
    if not staged:
        return adopted
    declared = declared_scorer() if declared is None else declared
    repose_dir = repose_dir if repose_dir is not None else repose_queue_dir()
    for video_id, h5 in sorted(staged.items()):
        try:
            row = db.get_video(video_id)
        except Exception:
            row = None
        if not row:
            continue                                # discover_dlc_staged's job
        state = row.get("state")
        if state == "failed":
            if _once(latched, f"adopt_failed:{video_id}"):
                logger.warning(f"{video_id}: a new pose is staged but the row is "
                               f"'failed' ({row.get('error_message')}); reset it "
                               f"(mousereach-watch-reprocess) to take the pose")
            continue
        if state == "archived":
            if declared and _archived_scorer(video_id) == declared:
                continue                            # already current on this pose
        elif state not in ADOPT_STATES:
            continue
        if state == "outdated" and row.get("reprocess_scope") == "segmentation" \
                and row.get("dlc_output_path") == str(h5):
            continue                                # already adopted, waiting its turn
        old_scope = row.get("reprocess_scope")
        reason = (f"declared pose {h5.name} staged for a row parked '{state}'; "
                  f"re-run from segmentation on it")
        try:
            if state == "outdated":
                db.set_fields(video_id, reprocess_scope="segmentation",
                              dlc_output_path=str(h5), error_message=None)
            elif state == "archived":
                db.update_state(video_id, "outdated", reprocess_scope="segmentation",
                                dlc_output_path=str(h5), mark_reason=None)
            else:                                   # unresolvable: legal to archived,
                db.force_state(video_id, "outdated", reason=reason,   # not to outdated
                               reprocess_scope="segmentation",
                               dlc_output_path=str(h5), error_message=None)
        except Exception as e:
            logger.error(f"{video_id}: could not adopt staged pose ({e})")
            continue
        close_request(video_id, repose_dir)
        _log(db, video_id, "adopted",
             f"{h5.name}; scope {old_scope!r} -> segmentation; "
             f"mark_reason={row.get('mark_reason')!r}")
        logger.info(f"{video_id}: adopted staged {h5.name} (row was '{state}'); "
                    f"queued for a post-DLC re-run")
        adopted.append(video_id)
    return adopted


def _archived_scorer(video_id: str) -> str:
    """The DLC scorer the archived manifest records, or ''."""
    d = archive_folder(video_id)
    if d is None:
        return ""
    m = read_json(d / f"{video_id}_processing_manifest.json") or {}
    return ((m.get("dlc_model") or {}).get("dlc_scorer") or "")
