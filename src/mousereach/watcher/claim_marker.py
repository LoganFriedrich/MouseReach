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
"file exists". Measured on an SMB share before this was relied on: several
processes racing for each of hundreds of keys, every key won exactly once,
none twice, none lost. The marker then records which host holds
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
machines' own clocks are never compared. WHY: machine clocks can disagree by
many minutes, and an age measured against a reading node's
clock would expire a live node's claim early or keep a dead one late.

A holder that has never written a report (an older watcher, or a host that no
longer exists) is judged by the marker's own share-stamped age instead. When
nobody's report can be read at all, nothing is broken: with no evidence a
holder is gone, the claim stands.

ONE PROCESS, NOT ONE MACHINE, WHEN THE CALLER SAYS SO
-----------------------------------------------------
By default a claim belongs to a host: a host that already holds a claim gets
True from ``take`` again. Several processes of ONE machine can work on the
same item, though -- the review queues are moved by the watcher's return
scan, by a review window and by mousereach-route-to-queue, all on the
machine where review happens. A caller that passes ``process``
(``this_process()``: the process id and its start time) gets a claim of that
process only: another live process of the same host is refused, exactly as
another host is. A claim whose process has ended (its id is gone, or now
belongs to a process started at another time) is a crashed process's
leftover and is taken over -- the host's status report cannot tell, because
the host is still running. A marker with no process recorded (written by an
older version) keeps the old meaning: it is the host's.

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


_PROCESS_TAG = "process "


def this_process() -> str:
    """This process's identity for a per-process claim: its id and its start
    time (an id alone can be reused by a later process)."""
    pid = os.getpid()
    try:
        import psutil
        return f"{pid} {psutil.Process(pid).create_time():.3f}"
    except Exception:
        return f"{pid} ?"


def process_alive(identity: Optional[str]) -> bool:
    """True unless there is evidence the process ``identity`` (this_process)
    has ended: no process has its id, or the one that has it started at
    another time. Anything that cannot be checked counts as alive -- a live
    claim must never be taken over."""
    try:
        pid_text, started = str(identity).split(" ", 1)
        pid = int(pid_text)
    except (TypeError, ValueError):
        return True
    try:
        import psutil
    except Exception:
        return True
    try:
        proc = psutil.Process(pid)
        if started == "?":
            return True
        return abs(proc.create_time() - float(started)) < 1.0
    except psutil.NoSuchProcess:
        return False
    except Exception:
        return True


def holder_process(claim_dir, key: str, suffix: str = ".claimed") -> Optional[str]:
    """The process (this_process) recorded in ``key``'s marker, or None when
    the marker records none (or cannot be read)."""
    try:
        text = _path(claim_dir, key, suffix).read_text(encoding="ascii", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(_PROCESS_TAG):
            return line[len(_PROCESS_TAG):].strip()
    return None


def _create(p: Path, host: str, process: Optional[str]) -> bool:
    """Create the marker exclusively. True when this call created it."""
    fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        text = f"{host}\n{datetime.now().isoformat()}\n"
        if process:
            text += f"{_PROCESS_TAG}{process}\n"
        os.write(fd, text.encode("ascii", "replace"))
    finally:
        os.close(fd)
    return True


def take(claim_dir, key: str, host: str, suffix: str = ".claimed",
         process: Optional[str] = None) -> bool:
    """Claim ``key`` for ``host`` (and, given ``process``, for that process of
    it only -- see the module docstring). True when the caller holds it
    afterwards.

    Without ``process``: True also when ``host`` already held it (a node that
    crashed between claiming and releasing must not lock itself out).
    With ``process``: True when this process already held it, or when the
    process that held it has ended (process_alive); False while another live
    process of the same host holds it. A marker of the same host that names
    no process (an older version's) counts as held by another process until
    it is LEGACY_MARKER_TAKEOVER_S old. False when another host holds it, or
    when the claim could not be made at all -- the claim folder cannot be
    created or written. WHY false then (it used to be true, "single node
    mode"): with several nodes sharing the pool, acting without a claim is
    exactly the double work the claim exists to stop. A node that cannot
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
        return _create(p, host, process)
    except FileExistsError:
        pass
    except OSError as e:
        logger.warning("Could not claim %s (%s)", key, e)
        return False
    if holder(claim_dir, key, suffix) != host:
        return False
    if process is None:
        return True
    theirs = holder_process(claim_dir, key, suffix)
    if theirs == process:
        return True              # already ours
    if theirs is None:
        # A marker of this machine that names no process: written by an older
        # version, which held claims per machine. While old and new code run
        # side by side (a review window started before an update), that
        # marker may belong to a live process of this machine, so it is NOT
        # taken as ours -- only once it is older than LEGACY_MARKER_TAKEOVER_S
        # (no move under a claim lasts that long), by this machine's own
        # clock, which wrote it.
        if not _legacy_marker_expired(p):
            return False
        if not _break_ended(p, key, None):
            return False
        logger.info("Took over the claim on %s: its marker named no process and "
                    "is older than %d minutes", key, LEGACY_MARKER_TAKEOVER_S // 60)
        try:
            return _create(p, host, process)
        except OSError:
            return False
    if process_alive(theirs):
        return False             # another live process of this machine
    # The holding process has ended without releasing (a crash): take it over.
    if not _break_ended(p, key, theirs):
        return False
    logger.info("Took over the claim on %s: its process %s has ended", key, theirs)
    try:
        return _create(p, host, process)
    except OSError:
        return False             # another process won the retake


# How old (seconds) a marker of this machine that names no process must be
# before a process-aware caller may take it over (take()).
LEGACY_MARKER_TAKEOVER_S = 3600


def _legacy_marker_expired(p: Path) -> bool:
    """True when the marker ``p`` was written more than
    LEGACY_MARKER_TAKEOVER_S ago, by the time on its second line (written by
    this same machine's clock). Unreadable: False (the safe side)."""
    try:
        lines = p.read_text(encoding="ascii", errors="replace").splitlines()
        written = datetime.fromisoformat(lines[1].strip()).timestamp()
    except (OSError, IndexError, ValueError):
        return False
    import time as _time
    return _time.time() - written > LEGACY_MARKER_TAKEOVER_S


def _break_ended(p: Path, key: str, ended: Optional[str]) -> bool:
    """Remove the marker ``p`` of the ended process ``ended`` (None: a marker
    that names no process) -- only if it is still that marker. True when it is gone (and the caller may
    create its own); False when it now belongs to someone else.

    WHY not simply unlink: two live processes can both read the ended
    holder. The first deletes the marker and creates its own; the second,
    deleting by name, would then delete the FIRST one's new marker and
    create its own, and both would believe they hold the claim. So the
    marker is first renamed to a name no other taker uses (a rename either
    takes the file or finds nothing), and what was taken is read: the ended
    process's marker is deleted; anyone else's is renamed back at once, and
    this caller does not take the claim."""
    import uuid
    grabbed = p.with_name(f"{p.name}.breaking-{uuid.uuid4().hex}")
    try:
        os.rename(str(p), str(grabbed))
    except FileNotFoundError:
        return True              # gone already; the exclusive create decides
    except OSError as e:
        logger.debug("Could not break the ended process's claim on %s (%s)", key, e)
        return False
    owner = None
    try:
        for line in grabbed.read_text(encoding="ascii", errors="replace").splitlines():
            if line.startswith(_PROCESS_TAG):
                owner = line[len(_PROCESS_TAG):].strip()
    except OSError:
        owner = None
    if owner == ended:
        try:
            grabbed.unlink()
        except OSError as e:
            logger.debug("Could not delete the ended claim %s (%s)", grabbed, e)
        return True
    # Another taker's live marker: give it back. If a third process has
    # created a marker in the meantime, theirs stands and this one is
    # dropped -- either way exactly one marker is in place, and it is not ours.
    try:
        os.rename(str(grabbed), str(p))
    except OSError:
        try:
            grabbed.unlink()
        except OSError:
            pass
    return False


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


def release(claim_dir, key: str, host: str, suffix: str = ".claimed",
            process: Optional[str] = None) -> bool:
    """Remove ``host``'s claim on ``key`` (given ``process``: only that
    process's -- a claim another process of the host recorded is left). Another
    host's claim is never removed here (only ``sweep_gone`` breaks those, on
    evidence). True when nothing of the caller's remains."""
    h = holder(claim_dir, key, suffix)
    if h is None:
        return True
    if h != host:
        return False
    if process is not None:
        theirs = holder_process(claim_dir, key, suffix)
        if theirs is not None and theirs != process:
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
