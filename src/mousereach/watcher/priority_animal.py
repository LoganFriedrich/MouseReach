"""The priority animal: one animal whose videos the watcher takes first.

WHAT IT IS
----------
A small file, ``<processing_root>/priority_animal.json``, naming one animal.
While it is there, the watcher prefers that animal's videos in every work
queue (cutting, posing, analysis, filing); every other video is still done,
only after them (orchestrator ``_get_priority_animal``). It changes the ORDER
only: every step, check, hold and review queue is exactly as for any video.

WHO SETS IT
-----------
``mousereach-watch-prioritize <animal>`` (``--clear`` to remove it) and the
"Do one animal's videos first" box on the Run Pipeline tab. Both call the
functions below, so the file, its check of the animal id and its wording are
one implementation.

WHY THE TAB HAS IT: running videos by hand from a window must be possible for
anything the watcher can do. The tab's old "Select Specific Files..." ran
chosen pose files through a separate, older copy of the steps with no checks;
putting an animal first is the watcher's own way to have chosen videos done
now, with every check kept.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

FILE_NAME = "priority_animal.json"


def priority_file(root=None) -> Path:
    """Where the file lives: in ``root`` when given (the watcher passes the
    processing root it already resolved), else in this machine's processing
    root (raises when none is configured, with the message saying how to
    configure one)."""
    if root is None:
        from mousereach.config import require_processing_root
        root = require_processing_root()
    return Path(root) / FILE_NAME


def current_priority(root=None) -> Optional[dict]:
    """``{"animal_id": ..., "set_at": ...}`` when an animal is set, else None.
    An unreadable file counts as none (the watcher reads it the same way).
    ``root``: as for :func:`priority_file`."""
    p = priority_file(root)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) and data.get("animal_id") else None


def set_priority_animal(animal_id: str) -> str:
    """Put ``animal_id`` first. Returns the id as stored (upper case).
    Raises ValueError, writing nothing, when it is not a valid animal id."""
    from mousereach.config import AnimalID
    animal_id = (animal_id or "").strip().upper()
    parsed = AnimalID.parse(animal_id)
    if not parsed.get("valid"):
        raise ValueError("Invalid animal ID '%s': %s" % (animal_id, parsed.get("error")))
    data = {"animal_id": animal_id, "set_at": datetime.now().isoformat()}
    # One-step replacement: a cut-off file would read as "no priority".
    from mousereach.pipeline.fsutil import write_json_atomic
    write_json_atomic(priority_file(), data, indent=2)
    return animal_id


def clear_priority() -> bool:
    """Remove the priority animal. True when one was set."""
    p = priority_file()
    if p.exists():
        p.unlink()
        return True
    return False
