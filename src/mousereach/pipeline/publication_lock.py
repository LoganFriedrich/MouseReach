"""The publication lock record: ``<video>_crystallized.json`` beside the filed results.

WHY A FILE AS WELL AS THE DATABASE ROW
--------------------------------------
``mousereach-crystallize`` locks a filed video for a publication: its filed
results must stay exactly as they are until a person unlocks it. The lock used
to be only a state ('crystallized') in the watcher database of the machine
that ran the command. Every machine keeps its own watcher database, so the
lock was known to that one database only: another analyzing machine's version
scan never saw it, and a lock written into the wrong database (a login whose
settings named an old copy) looked set and protected nothing.

The filed folder is the one place every machine reads. So locking writes a
small record there, beside the video's filed results, and the record IS the
lock that every machine honors:

  * the version scan never marks a video with a record for a re-run (it reads
    the records from the same folder listing it already makes);
  * a re-run of a video with a record is refused when it would start, and the
    row goes back to 'archived';
  * filing into a folder whose record names the video is refused, so locked
    results are never replaced;
  * every path that would move the video into a review queue, back to
    Processing or to a new pose asks watcher.db.crystallized_refusal, which
    reads the row AND the record;
  * the dashboard's Re-run button and mousereach-repose-videos refuse.

``mousereach-uncrystallize`` removes the record (and the row's lock). The
database row is kept as well: it is what this machine's state machine checks
on every state change.

THE RECORD
----------
``<Analyzed>/<project>/<cohort>/<video>_crystallized.json`` -- the folder the
archive step files the video into (archive.core.filed_folder) -- holding the
label, who locked it and when. It is written to a temporary dot-name in the
same folder and renamed into place in one step, so a reader sees either no
record or a whole one. No folder is ever created for it: a video whose filed
folder is not there is not filed, and is refused.

A record that exists but cannot be read still locks (the cautious side): only
a person running mousereach-uncrystallize takes a lock away. A folder that
cannot be checked at all (the share is unreachable) is "cannot tell", and the
callers refuse.
"""
from __future__ import annotations

import json
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

LOCK_SUFFIX = "_crystallized.json"

# A video's filed results: one of these beside it shows the video is filed
# in that folder. Checked before a record is written.
_FILED_MARKERS = ("_processing_manifest.json", "_features.json", ".mp4")


class LockRecordError(RuntimeError):
    """A lock record could not be written or removed; the message says why."""


def _analyzed_root(analyzed_root=None) -> Optional[Path]:
    if analyzed_root is not None:
        return Path(analyzed_root)
    from mousereach.config import Paths
    root = getattr(Paths, "ANALYZED_OUTPUT", None)
    return Path(root) if root else None


def filed_folder_of(video_id: str, analyzed_root=None) -> Optional[Path]:
    """The folder the video is (or would be) filed in, or None when this
    machine has no filed tree configured."""
    root = _analyzed_root(analyzed_root)
    if root is None:
        return None
    from mousereach.archive.core import filed_folder
    return filed_folder(video_id, root)


def record_path(video_id: str, analyzed_root=None, folder=None) -> Optional[Path]:
    """Where the video's lock record is, or None (no filed tree configured)."""
    folder = Path(folder) if folder is not None else filed_folder_of(video_id, analyzed_root)
    if folder is None:
        return None
    return folder / f"{video_id}{LOCK_SUFFIX}"


def read_record(path: Path) -> Dict:
    """The record's contents; a record that cannot be read or parsed is still
    a lock, returned as ``{"label": "?", "unreadable": <why>}``."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {"label": "?", "unreadable": "not a JSON object"}
    except Exception as e:
        return {"label": "?", "unreadable": str(e)}


def lock_state(video_id: str, analyzed_root=None, folder=None) -> Tuple[str, object]:
    """``("locked", record)``, ``("unlocked", None)`` or ``("unknown", why)``.

    No filed tree configured on this machine -> "unlocked": there are no filed
    results here to protect, and every step that would touch them needs that
    tree anyway. A filed tree that IS configured but cannot be reached -- the
    folder holding it is not there either (an unmapped drive, an unreachable
    share) -> "unknown": a missing record there proves nothing. A filed tree
    whose own folder is simply not made yet, beside a folder that is there,
    holds nothing filed -> "unlocked"."""
    path = record_path(video_id, analyzed_root, folder)
    if path is None:
        return "unlocked", None
    root = _analyzed_root(analyzed_root)
    if root is not None:
        try:
            reachable = root.is_dir() or root.parent.is_dir()
        except OSError:
            reachable = False
        if not reachable:
            return "unknown", f"the filed tree {root} cannot be reached"
    try:
        path.stat()
    except FileNotFoundError:
        return "unlocked", None
    except OSError as e:
        return "unknown", f"{path} could not be checked ({e})"
    rec = read_record(path)
    rec.setdefault("record", str(path))
    return "locked", rec


def lock_refusal(video_id: str, action: str, analyzed_root=None, folder=None) -> Optional[str]:
    """The refusal message when the video has a lock record (or whether it has
    one cannot be told), else None."""
    state, info = lock_state(video_id, analyzed_root, folder)
    if state == "unlocked":
        return None
    if state == "unknown":
        return (f"cannot tell whether {video_id} is locked for a publication ({info}); "
                f"not {action}.")
    label = (info or {}).get("label") or "?"
    return (f"{video_id} is crystallized (locked for '{label}', record "
            f"{(info or {}).get('record')}); not {action}. Unlock it first with "
            f"mousereach-uncrystallize if this is intended.")


def write_record(video_id: str, label: str, by: Optional[str] = None,
                 at: Optional[str] = None, analyzed_root=None) -> Path:
    """Write the video's lock record beside its filed results, in one step.
    Raises LockRecordError when the video's filed results are not where the
    archive step files them, or the record cannot be written."""
    folder = filed_folder_of(video_id, analyzed_root)
    if folder is None:
        raise LockRecordError("no filed tree (Analyzed) is configured on this machine")
    if not folder.is_dir() or not any((folder / f"{video_id}{s}").exists()
                                      for s in _FILED_MARKERS):
        raise LockRecordError(f"its filed results are not in {folder}")
    final = folder / f"{video_id}{LOCK_SUFFIX}"
    record = {
        "video_id": video_id,
        "label": label,
        "crystallized_by": by or socket.gethostname(),
        "crystallized_at": at or datetime.now().isoformat(),
        "machine": socket.gethostname(),
        "what": ("Publication lock (mousereach-crystallize). While this file is here "
                 "no machine re-runs, re-files or moves this video's filed results. "
                 "Remove it only with mousereach-uncrystallize."),
    }
    # A dot-name that does not end in .json: no listing of filed results
    # (version scan, census, backfills) can mistake a half-written temp for
    # a record or a result.
    tmp = folder / f".{video_id}_crystallized.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
    except OSError as e:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise LockRecordError(f"{final} could not be written ({e})") from e
    return final


def remove_record(video_id: str, analyzed_root=None, path=None) -> bool:
    """Remove the video's lock record. True when one was removed, False when
    there was none. Raises LockRecordError when it is there and cannot be
    removed (the video stays locked)."""
    p = Path(path) if path is not None else record_path(video_id, analyzed_root)
    if p is None:
        return False
    try:
        p.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        raise LockRecordError(f"{p} could not be removed ({e})") from e


def iter_records(analyzed_root=None) -> Iterator[Tuple[str, Path]]:
    """``(video_id, record path)`` for every lock record in the filed tree,
    from one listing (never inside a superseded folder)."""
    root = _analyzed_root(analyzed_root)
    if root is None:
        return
    from mousereach.pipeline.analyzed_tree import iter_file_entries
    for e in iter_file_entries(root, f"*{LOCK_SUFFIX}"):
        if e.name.startswith("."):
            continue
        yield e.name[: -len(LOCK_SUFFIX)], Path(e.path)


def records_with_label(label: str, analyzed_root=None) -> List[Tuple[str, Path]]:
    """Every lock record whose label is ``label``."""
    return [(vid, p) for vid, p in iter_records(analyzed_root)
            if (read_record(p).get("label") or "") == label]
