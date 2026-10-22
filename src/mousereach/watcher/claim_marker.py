"""An exclusive claim on one item of a shared pool, held as a marker file.

WHY THIS EXISTS
---------------
Several watcher nodes read the same shared folders. When two of them see the
same item -- a posed video waiting in ``Processing/Posed``, a cleared bundle in
a review queue -- exactly one may act on it. The old markers did this as
check, write, sleep half a second, read back. That narrows the window in which
two nodes both write a marker and both believe they won; it does not close it
(the single-video claim's docstring, ``single_claim.py``, says so itself).

THE CLAIM IS AN EXCLUSIVE CREATE
--------------------------------
``take`` creates ``<claim dir>/<key><suffix>`` with ``O_CREAT | O_EXCL``. The
file server performs "create only if it does not exist" as one operation, so
when several nodes race, exactly one create succeeds and every other one gets
"file exists". Measured on the lab's share before this was relied on
(2026-10-21): three processes racing for each of 400 keys, every key won
exactly once, none twice, none lost. The marker then records which host holds
it.

Why a marker and not a rename of the item itself (the singles folder's
claim): an item here is several files (a video and its pose files, or a
whole review bundle), and every other part of the pipeline already looks for
them where they are. A marker beside the pool claims the item without moving
any of its files.

WHEN A CLAIM IS STALE -- THE HOLDER'S HEARTBEAT, NOT THE CLAIM'S AGE
--------------------------------------------------------------------
A claim is held for as long as its holder is alive, however long its work
takes: a video can sit in a node's local queue for hours, or in a review
queue for weeks. So a claim is never broken for being old. It is broken only
when its holder has stopped writing its status report
(``node_status``, ``Processing/.node_status/<host>.json``, rewritten on every
turn of the holder's work loop) for ``GONE_AFTER``.

Both times compared are stamped by the FILE SERVER: the holder's report's
modified time, and "now" taken as the newest report anyone has written. The
machines' own clocks are never compared. WHY: the lab's clocks disagree by
about 21 minutes (2026-10-21), and an age measured against a reading node's
clock would expire a live node's claim early or keep a dead one late.

A holder that has never written a report (an older watcher, or a host that no
longer exists) is judged by the marker's own share-stamped age instead. When
nobody's report can be read at all, nothing is broken: with no evidence a
holder is gone, the claim stands.

ASCII-only log text (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# A holder whose status report has not been rewritten for this long, by the
# file server's clock, is gone, and its claims may be broken. WHY three hours:
# a live node rewrites its report once per turn of its loop, and one turn has
# been measured at over 18 minutes (a slow scan) and can include a pose run;
# node_status itself calls a node "not running" after 90 minutes. Twice that
# leaves room for a slow turn on a slow night. Breaking a live node's claim
# costs duplicate work; keeping a dead node's claim three hours costs a delay.
GONE_AFTER_S = 3 * 3600


def _path(claim_dir, key: str, suffix: str) -> Path:
    return Path(claim_dir) / f"{key}{suffix}"


def take(claim_dir, key: str, host: str, suffix: str = ".claimed") -> bool:
    """Claim ``key`` for ``host``. True when ``host`` holds it afterwards.

    True also when ``host`` already held it (a node that crashed between
    claiming and releasing must not lock itself out). False when another host
    holds it, or when the claim could not be made at all -- the claim folder
    cannot be created or written. WHY false then (it used to be true, "single
    node mode"): with several nodes sharing the pool, acting without a claim
    is exactly the double work the claim exists to stop. A node that cannot
    claim waits and tries again.
    """
    p = _path(claim_dir, key, suffix)
    try:
        Path(claim_dir).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("Cannot create claim folder %s (%s); %s not claimed",
                       claim_dir, e, key)
        return False
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return holder(claim_dir, key, suffix) == host
    except OSError as e:
        logger.warning("Could not claim %s (%s)", key, e)
        return False
    try:
        os.write(fd, f"{host}\n{datetime.now().isoformat()}\n".encode("ascii", "replace"))
    finally:
        os.close(fd)
    return True


def holder(claim_dir, key: str, suffix: str = ".claimed") -> Optional[str]:
    """The host holding ``key``; None when unclaimed. An empty or unreadable
    marker returns "" -- claimed, holder unknown (a node stopped between
    creating the marker and writing its name)."""
    p = _path(claim_dir, key, suffix)
    try:
        text = p.read_text(encoding="ascii", errors="replace")
    except FileNotFoundError:
        return None
    except OSError:
        return ""
    return text.split("\n", 1)[0].strip()


def holds(claim_dir, key: str, host: str, suffix: str = ".claimed") -> bool:
    return holder(claim_dir, key, suffix) == host


def release(claim_dir, key: str, host: str, suffix: str = ".claimed") -> bool:
    """Remove ``host``'s claim on ``key``. Another host's claim is never
    removed here (only ``sweep_gone`` breaks those, on evidence). True when
    nothing of ``host``'s remains."""
    h = holder(claim_dir, key, suffix)
    if h is None:
        return True
    if h != host:
        return False
    try:
        _path(claim_dir, key, suffix).unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.debug("Could not release claim %s (%s)", key, e)
        return False
    return True


def _mtime(p: Path) -> Optional[float]:
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def status_times(status_dir) -> dict:
    """``{host: modified time of its status report}``, by the file server's
    clock. Empty when the folder cannot be read."""
    out = {}
    if status_dir is None:
        return out
    try:
        with os.scandir(status_dir) as entries:
            for e in entries:
                if e.name.endswith(".json") and not e.name.startswith("."):
                    try:
                        out[e.name[:-5]] = e.stat().st_mtime
                    except OSError:
                        pass
    except OSError:
        pass
    return out


def holder_gone(host: Optional[str], marker: Path, times: dict,
                gone_after_s: float = GONE_AFTER_S) -> bool:
    """True only on evidence that ``host`` is no longer running.

    ``times`` is ``status_times(...)``; "now" is its newest entry. See the
    module docstring for why only file-server times are compared."""
    if not times:
        return False
    now = max(times.values())
    if host and host in times:
        return now - times[host] > gone_after_s
    # No report from this holder (an older watcher, a vanished host, or a
    # marker whose holder never got written): judge by the marker's own age.
    m = _mtime(marker)
    return m is not None and now - m > gone_after_s


def sweep_gone(claim_dir, me: str, status_dir, suffix: str = ".claimed",
               gone_after_s: float = GONE_AFTER_S) -> List[str]:
    """Break every claim in ``claim_dir`` whose holder is gone. Never ``me``'s.
    Returns the keys whose claims were broken."""
    broken: List[str] = []
    times = status_times(status_dir)
    if not times:
        return broken
    try:
        with os.scandir(claim_dir) as entries:
            markers = [Path(e.path) for e in entries if e.name.endswith(suffix)]
    except OSError:
        return broken
    for m in markers:
        key = m.name[: -len(suffix)]
        h = holder(claim_dir, key, suffix)
        if h is None or h == me:
            continue
        if not holder_gone(h, m, times, gone_after_s):
            continue
        try:
            m.unlink()
            broken.append(key)
            logger.info("Broke claim on %s: holder %s has not reported for over %.0f h",
                        key, h or "(unknown)", gone_after_s / 3600)
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.debug("Could not break claim %s (%s)", key, e)
    return broken
