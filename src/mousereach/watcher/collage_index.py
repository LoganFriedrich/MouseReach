"""Where every collage rests on the shared drive: one small file, built by the server.

WHY THIS EXISTS
---------------
A recording PC that sends named recordings (watcher/recording_outbox.py) must
know whether a recording is already on the shared drive, and where: a collage
leaves the intake once its videos are filed, so "not in the intake" does not
mean "never sent". Answering that by walking the shared drive's folder trees
from the recording PC means hundreds of folder listings over the network on
every pass. A long recursive walk over a network share from a PC is exactly
what must never happen: when the PC's connection to the share degrades, every
listing takes minutes, the watcher crawls, and the shared coordination database
(which lives on the same share) waits behind it for every machine.

So the walk is done ONCE, by the processing server, and written down as one
file the PCs read whole:

    <nas_root>/Processing/.collage_index.json

The PCs never walk the share. They read this one file and open single named
files (watcher/recording_outbox.py says exactly which).

WHAT IS IN IT
-------------
Every file found directly inside the folders where a collage can rest:

* the intake, ``Unanalyzed/Multi-Animal``;
* ``Unanalyzed/Unsupported_Tray_Type``, ``Processing/Quarantine`` and the
  recording-issue folder ``Processing/Review/_Problematic``, all the way down
  (they keep subfolders);
* every folder named ``Multi-Animal`` up to four levels under ``Analyzed`` and
  under ``Archive`` (where a collage goes once its videos are filed).

For each file: its name (the key, lower case), its path RELATIVE to the shared
root (forward slashes, so a PC that maps the share under another letter reads
the same file) and its size. The file also says when and on which machine it
was built.

WHAT IT IS NOT
--------------
It is a LOCATOR, never proof. A PC never deletes anything because the index
lists a name: it opens the one listed file and compares size and SHA-256 with
its own copy (recording_outbox._match_existing). An index that is out of date
can therefore make a PC wait, never make it delete the wrong thing.

WHY NOT THE COORDINATION DATABASE ALONE
---------------------------------------
``pipeline_collages`` (watcher_central.db) has a row for a collage only once
a machine has claimed it for cutting, and the row holds no path and no
content hash; where a filed collage ended up is kept only in the local
database of the machine that filed it. The database can say "the pipeline
has seen this name" (the outbox asks it that, for one name, when nothing can
be found), but not where the file is, so it cannot confirm a copy.

WHO BUILDS IT, AND HOW OFTEN
----------------------------
The processing server's watcher, during its scan, at most every
``REBUILD_EVERY_S`` (10 minutes), on its own thread so the loop never waits
for it (a build of a few thousand collages takes tens of seconds); by hand
with
``python -m mousereach.watcher.collage_index``. Written whole through a
temporary file swapped in, so a reader sees the old file or the new one,
never half of one.

All text written to a terminal or a log is ASCII.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

INDEX_RELATIVE = ("Processing", ".collage_index.json")
FORMAT = 1
REBUILD_EVERY_S = 600.0
_PROBLEMATIC = "_Problematic"      # recording_issue.FOLDER_NAME (kept in step by a test)


def index_path(nas_root) -> Optional[Path]:
    """<nas_root>/Processing/.collage_index.json, or None without a root."""
    if not nas_root:
        return None
    return Path(nas_root).joinpath(*INDEX_RELATIVE)


# =============================================================================
# BUILDING (processing server only)
# =============================================================================

def resting_places(nas_root: Path, depth: int = 4) -> List[Path]:
    """Every folder on the shared drive where a sent collage can be now.

    SERVER ONLY: this walks folder trees on the share. A recording PC reads
    the index instead (see the module docstring for why).
    """
    nas_root = Path(nas_root)
    out: List[Path] = []

    def add(p: Path):
        if p.is_dir() and p not in out:
            out.append(p)

    add(nas_root / "Unanalyzed" / "Multi-Animal")
    for top in (nas_root / "Unanalyzed" / "Unsupported_Tray_Type",
                nas_root / "Processing" / "Quarantine",
                nas_root / "Processing" / "Review" / _PROBLEMATIC):
        if top.is_dir():
            for d, _dirs, _files in os.walk(top):
                add(Path(d))
    for top in (nas_root / "Analyzed", nas_root / "Archive"):
        frontier = [top] if top.is_dir() else []
        for _ in range(depth):
            nxt = []
            for d in frontier:
                try:
                    with os.scandir(d) as it:
                        for e in it:
                            if e.is_dir(follow_symlinks=False):
                                p = Path(e.path)
                                if e.name == "Multi-Animal":
                                    add(p)
                                else:
                                    nxt.append(p)
                except OSError:
                    continue
            frontier = nxt
    return out


def build(nas_root: Path) -> dict:
    """The index as a dict (not written). SERVER ONLY: walks the share."""
    nas_root = Path(nas_root)
    started = time.time()
    entries: Dict[str, List[dict]] = {}
    for d in resting_places(nas_root):
        try:
            with os.scandir(d) as it:
                for e in it:
                    if not e.is_file(follow_symlinks=False):
                        continue
                    try:
                        size = e.stat(follow_symlinks=False).st_size
                    except OSError:
                        size = None
                    rel = Path(e.path).relative_to(nas_root).as_posix()
                    entries.setdefault(e.name.lower(), []).append({"path": rel, "bytes": size})
        except OSError:
            continue
    # Stamped with the time the walk STARTED: a file that arrived during the
    # walk may be missing, and a reader must not believe the index is newer
    # than it is.
    return {"format": FORMAT,
            "built_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
            "built_at_epoch": started,
            "built_by": socket.gethostname(),
            "seconds_to_build": round(time.time() - started, 1),
            "entries": entries}


def write(nas_root: Path) -> Path:
    """Build the index and put it in place in one step. Returns its path."""
    from mousereach.pipeline.fsutil import write_json_atomic
    data = build(nas_root)
    path = index_path(nas_root)
    write_json_atomic(path, data, indent=0)
    logger.info("Collage index: %d names written to %s in %.1f s",
                len(data["entries"]), path, data["seconds_to_build"])
    return path


class Rebuilder:
    """Rebuilds the index at most every ``every_s`` seconds, on its own thread.

    WHY a thread: on a share of a few thousand collages one build takes
    tens of seconds, and the processing server's loop must not stop for it.
    At most one build runs at a time; a build still running when the next
    one is due is not doubled. ``background=False`` builds in the caller's
    thread (tests, the command line). Never raises.
    """

    def __init__(self, every_s: float = REBUILD_EVERY_S, clock=time.monotonic,
                 background: bool = True):
        self.every_s = every_s
        self._clock = clock
        self._background = background
        self._last = None
        self._thread = None

    @staticmethod
    def _write_logged(nas_root: Path) -> bool:
        try:
            write(nas_root)
            return True
        except Exception as e:
            logger.warning("Could not rewrite the collage index (the recording PCs keep "
                           "using the previous one): %s", e)
            return False

    def maybe_write(self, nas_root) -> bool:
        """True when a build was started (or, in the foreground, succeeded)."""
        if not nas_root:
            return False
        if self._thread is not None and self._thread.is_alive():
            return False
        now = self._clock()
        if self._last is not None and now - self._last < self.every_s:
            return False
        self._last = now
        if not self._background:
            return self._write_logged(Path(nas_root))
        import threading
        self._thread = threading.Thread(target=self._write_logged, args=(Path(nas_root),),
                                        name="collage-index", daemon=True)
        self._thread.start()
        return True


# =============================================================================
# READING (any machine: ONE file)
# =============================================================================

class Index:
    """What a reader got: the entries, when they were built, and the problem
    (None when the file was read). ``status`` is 'ok', 'missing' or 'unreadable'."""

    def __init__(self, entries: Optional[Dict[str, List[dict]]], built_at_epoch: Optional[float],
                 built_at: Optional[str], status: str, problem: Optional[str] = None):
        self.entries = entries
        self.built_at_epoch = built_at_epoch
        self.built_at = built_at
        self.status = status
        self.problem = problem

    @property
    def usable(self) -> bool:
        return self.entries is not None

    def paths(self, nas_root: Path, name: str) -> List[Path]:
        """Every listed path for ``name`` (absolute, on this machine's root)."""
        out = []
        for rec in (self.entries or {}).get(name.lower(), []):
            rel = rec.get("path") if isinstance(rec, dict) else None
            # Relative, inside the root: no drive, no leading slash, no "..".
            if rel and not Path(rel).anchor and ".." not in Path(rel).parts:
                out.append(Path(nas_root) / Path(rel))
        return out

    def describe(self, now_epoch: Optional[float] = None) -> dict:
        d = {"status": self.status, "built_at": self.built_at}
        if self.problem:
            d["problem"] = self.problem
        if self.built_at_epoch is not None:
            now = time.time() if now_epoch is None else now_epoch
            d["age_minutes"] = int(max(0.0, now - self.built_at_epoch) // 60)
        return d


def read(nas_root) -> Index:
    """Read the one index file. Never raises; never lists a folder."""
    path = index_path(nas_root)
    if path is None:
        return Index(None, None, None, "missing", "no shared root is configured")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Index(None, None, None, "missing",
                     "%s does not exist (the processing server writes it)" % path)
    except (OSError, ValueError) as e:
        return Index(None, None, None, "unreadable", "%s: %s" % (path, e))
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        return Index(None, None, None, "unreadable", "%s has no entries" % path)
    epoch = data.get("built_at_epoch")
    return Index(entries, float(epoch) if isinstance(epoch, (int, float)) else None,
                 data.get("built_at"), "ok")


def main(argv=None) -> int:
    """Rebuild the index now (on the processing server)."""
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m mousereach.watcher.collage_index",
        description=("Rebuild the shared drive's collage index (Processing/.collage_index.json) "
                     "now. The processing server's watcher does this on its own every 10 "
                     "minutes; run this on the processing server only, never on a recording "
                     "or graphics-card PC (it walks folder trees on the shared drive)."))
    p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from mousereach.config import Paths
    if not Paths.NAS_ROOT:
        print("[FAIL] nas_root is not configured (mousereach-setup)")
        return 1
    path = write(Path(Paths.NAS_ROOT))
    print("[OK] written: %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
