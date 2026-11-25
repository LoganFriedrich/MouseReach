"""The shared pipeline drive (nas_root): is it there, and how to wait when it is not.

WHY THIS EXISTS
---------------
Every watcher works on files that live on one shared drive. When that drive goes
away -- the file server restarts, the network drops, a PC's SMB client wedges --
every call that touches it either fails with a network error or, worse, HANGS: a
listing, a stat, an open can sit in Windows for many minutes before it gives up,
and a cycle makes hundreds of them. In such an outage a watcher can sit inside shared-drive calls for hours, never
reaching the line that reads its stop flag; a PC can read the version declaration
as "not there" and stop posing as if it were misconfigured; and a watcher
restarted while the drive is away can crash at startup and stay down after the
drive comes back.

An outage is a PAUSE, not an error. So:

* ``probe`` asks one cheap question -- "is the pipeline folder a folder?" -- in a
  worker thread with a time limit, so the question itself can never hang the
  caller. The watcher asks it at the top of every cycle and does no shared-drive
  work at all while the answer is no.
* ``call_bounded`` runs any other shared-drive call the same way. A call that
  does not answer in time is abandoned (the thread is left to finish on its own;
  Windows gives no way to cancel it) and the caller is told the drive is
  unavailable. Only ONE abandoned call per purpose is ever outstanding: while it
  is still stuck, the next call for the same purpose is refused at once instead
  of starting another stuck thread.
* ``is_share_down_error`` says whether an error is the drive being away, so a
  video is never marked failed for it.
* ``wait_for_shared_drive`` is what a starting watcher does instead of crashing.
* ``StopWatchdog`` makes the stop flag work even while the watcher is stuck in a
  call the drive will not answer.
* ``maybe_remap`` re-establishes a drive LETTER whose mapping has disappeared.

WHY REMAP THE LETTER AND NOT SWITCH TO THE UNC PATH
---------------------------------------------------
Every recorded path (database rows, claim markers, queue copies, the folders
derived from nas_root at import time) is written with the drive letter, and
several checks compare paths as text. Working on ``\\\\server\\share`` instead
would give one file two names mid-run. So the watcher only ever puts the letter
back, and only when the letter is not mapped AT ALL: a letter that is mapped but
not answering is left to Windows' own reconnect, because deleting a wedged
mapping (``net use /delete``) is itself what hung for half an hour in the
outage this module was written for. The UNC origin is learned, not configured:
Windows' remembered mapping (``HKCU\\Network\\<letter>``) and a local note the
watcher writes while the drive works (``shared_drive_origin.json`` in the
processing folder) -- so a mapping deleted outright can still be put back.

ASCII only in everything logged: these lines reach a Windows console.
"""
from __future__ import annotations

import json
import logging
import os
import re
import stat as _stat
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

# How long the availability question may take before the answer is "no".
PROBE_TIMEOUT_S = 20.0
# How long a small read (a settings file) may take.
READ_TIMEOUT_S = 30.0
# How long the node's status report may take to write.
STATUS_WRITE_TIMEOUT_S = 15.0

# Windows errors that mean the network location is away, not that a file is
# wrong: 53 network path not found, 59 unexpected network error, 64 network
# name no longer available, 67 network name not found, 121 semaphore timeout,
# 1201 connection unavailable (a remembered mapping not connected), 1222 no
# network, 1231 network location unreachable, 1232 host unreachable, 2250
# network connection does not exist.
SHARE_DOWN_WINERRORS = frozenset({53, 59, 64, 67, 121, 1201, 1222, 1231, 1232, 2250})

_WINERROR_TEXT = re.compile(r"\[WinError (\d+)\]")

OK, DOWN, MISSING = "ok", "down", "missing"


class SharedDriveUnavailable(RuntimeError):
    """The shared drive cannot be reached right now.

    Deliberately NOT an OSError: code all over the watcher catches OSError to
    decide something about a FILE (gone, locked, not a pose), and this is not
    about any file. It must travel up to the code that knows to wait.
    """


class ShareTimeout(SharedDriveUnavailable):
    """A call on the shared drive did not answer within its time limit."""


# ---------------------------------------------------------------------------
# Classifying errors
# ---------------------------------------------------------------------------

def _winerror_of(e) -> Optional[int]:
    code = getattr(e, "winerror", None)
    if code is None and isinstance(e, BaseException):
        m = _WINERROR_TEXT.search(str(e))
        if m:
            code = int(m.group(1))
    return code


def is_share_down_error(e) -> bool:
    """True when ``e`` (or what caused it) is the shared drive being away.

    Facts first: our own SharedDriveUnavailable, or an OSError whose Windows
    error code is one of SHARE_DOWN_WINERRORS. The ``[WinError N]`` text is read
    only for an error that wrapped one (a handler that re-raised it as its own
    message) -- the code is gone from such an error but the text is not.
    """
    seen = 0
    while e is not None and seen < 5:
        if isinstance(e, SharedDriveUnavailable):
            return True
        if _winerror_of(e) in SHARE_DOWN_WINERRORS:
            return True
        e = e.__cause__ or e.__context__
        seen += 1
    return False


def message_names_share_down(message) -> bool:
    """True when a failure message carries a share-down Windows error code."""
    m = _WINERROR_TEXT.search(str(message or ""))
    return bool(m) and int(m.group(1)) in SHARE_DOWN_WINERRORS


def describe_error(e) -> str:
    return ("%s: %s" % (type(e).__name__, e)).encode("ascii", "replace").decode("ascii")


# ---------------------------------------------------------------------------
# Bounded calls
# ---------------------------------------------------------------------------

_inflight_lock = threading.Lock()
_inflight: dict = {}        # key -> the abandoned thread still waiting


def call_bounded(fn: Callable, *args, timeout: float, what: str,
                 key: Optional[str] = None, **kwargs):
    """Run ``fn(*args, **kwargs)`` and return its result, or raise ShareTimeout
    if it has not returned after ``timeout`` seconds.

    The call runs in a daemon thread. A thread that times out is abandoned, not
    killed (Windows cannot cancel a pending SMB call); while it is still stuck,
    any further call with the same ``key`` (default: ``what``) is refused at
    once, so a dead drive costs one stuck thread per purpose, not one per cycle.
    An exception from ``fn`` is re-raised unchanged.
    """
    key = key or what
    with _inflight_lock:
        stuck = _inflight.get(key)
        if stuck is not None:
            if stuck.is_alive():
                raise ShareTimeout("%s: an earlier attempt is still waiting on the "
                                   "shared drive" % what)
            del _inflight[key]
    box: dict = {}

    def runner():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as e:          # handed to the caller below
            box["error"] = e

    t = threading.Thread(target=runner, name="shared-drive: %s" % what, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        with _inflight_lock:
            _inflight[key] = t
        raise ShareTimeout("%s did not answer within %.0f s" % (what, timeout))
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _forget_stuck_calls() -> None:
    """Tests only: drop the record of abandoned calls."""
    with _inflight_lock:
        _inflight.clear()


# ---------------------------------------------------------------------------
# The availability question
# ---------------------------------------------------------------------------

def _probe_once(root: str) -> Tuple[str, str]:
    # os.stat, not os.path.isdir: isdir swallows every OSError and answers
    # False, which would make "the network is gone" look like "no such folder".
    try:
        st = os.stat(root)
    except OSError as e:
        if is_share_down_error(e):
            return DOWN, describe_error(e)
        anchor = Path(root).anchor
        if anchor:
            try:
                os.stat(anchor)
            except OSError as e2:
                return DOWN, "%s cannot be reached (%s)" % (anchor, describe_error(e2))
        return MISSING, "%s not found (%s)" % (root, describe_error(e))
    if not _stat.S_ISDIR(st.st_mode):
        return MISSING, "%s is not a folder" % root
    return OK, ""


def probe(nas_root, timeout: Optional[float] = None, key: str = "probe") -> Tuple[str, str]:
    """(state, detail): OK, DOWN (the drive is away or did not answer in time)
    or MISSING (the drive answers but the folder is not there -- a setting, not
    an outage). Never raises, never takes longer than ``timeout``."""
    root = str(nas_root)
    timeout = PROBE_TIMEOUT_S if timeout is None else timeout
    try:
        return call_bounded(_probe_once, root, timeout=timeout,
                            what="shared drive check", key="%s %s" % (key, root))
    except SharedDriveUnavailable as e:
        return DOWN, str(e)
    except Exception as e:                  # never let the question itself break a caller
        return DOWN, describe_error(e)


def read_text_bounded(path, timeout: Optional[float] = None, what: Optional[str] = None
                      ) -> Optional[str]:
    """A small text file on the shared drive: its text, or None when the file is
    not there on a drive that answers. Raises SharedDriveUnavailable when the
    drive is away or does not answer in time -- so "away" can never be mistaken
    for "not there" (which would stop a node's posing on a dead drive)."""
    path = Path(path)
    label = what or ("read %s" % path.name)
    timeout = READ_TIMEOUT_S if timeout is None else timeout

    def _read():
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            # A drive letter whose mapping vanished also says "not found".
            anchor = path.anchor
            if anchor:
                try:
                    os.stat(anchor)
                except OSError as e2:
                    raise SharedDriveUnavailable(
                        "%s cannot be reached (%s)" % (anchor, describe_error(e2))) from e
            return None
        except OSError as e:
            if is_share_down_error(e):
                raise SharedDriveUnavailable(describe_error(e)) from e
            raise

    return call_bounded(_read, timeout=timeout, what=label, key="read %s" % path)


# ---------------------------------------------------------------------------
# Drive-letter mapping
# ---------------------------------------------------------------------------

ORIGIN_FILENAME = "shared_drive_origin.json"
REMAP_EVERY_S = 600.0
_ERROR_NOT_CONNECTED = 2250


def drive_letter(nas_root) -> Optional[str]:
    """'Y:' for a path on a drive letter, None for a UNC or non-Windows path."""
    drive = Path(str(nas_root)).drive
    return drive.upper() if re.fullmatch(r"[A-Za-z]:", drive or "") else None


def mapped_remote(drive: str) -> Tuple[Optional[int], Optional[str]]:
    """(Windows result code, remote name) for a mapped letter, via
    WNetGetConnectionW. Code 0: connected; 1201: remembered, not connected;
    2250: the letter is not mapped at all. (None, None) off Windows or when
    the call itself is unavailable."""
    if sys.platform != "win32":
        return None, None
    try:
        import ctypes
        from ctypes import wintypes
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buf))
        rc = ctypes.windll.mpr.WNetGetConnectionW(drive, buf, ctypes.byref(size))
        return int(rc), (buf.value or None)
    except Exception:
        return None, None


def remembered_remote(drive: str) -> Optional[str]:
    """The UNC Windows remembers for a persistent mapping of ``drive``."""
    if sys.platform != "win32":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Network\%s" % drive[0].upper()) as k:
            value, _ = winreg.QueryValueEx(k, "RemotePath")
            return str(value) or None
    except Exception:
        return None


def _origin_path() -> Optional[Path]:
    try:
        from mousereach.config import require_processing_root
        return require_processing_root() / ORIGIN_FILENAME
    except Exception:
        return None


def learned_remote(drive: str) -> Optional[str]:
    p = _origin_path()
    if p is None:
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if str(data.get("drive", "")).upper() == drive:
            return data.get("remote") or None
    except Exception:
        pass
    return None


def remember_origin(nas_root) -> Optional[str]:
    """While the drive works, note which share the letter points at, locally, so
    the mapping can be put back even if it is deleted outright. Never raises."""
    drive = drive_letter(nas_root)
    if not drive:
        return None
    rc, remote = mapped_remote(drive)
    if rc != 0 or not remote:
        return None
    p = _origin_path()
    if p is None or learned_remote(drive) == remote:
        return remote
    try:
        p.write_text(json.dumps({"drive": drive, "remote": remote,
                                 "noted_at": datetime.now().isoformat(timespec="seconds")},
                                indent=2) + "\n", encoding="ascii")
    except Exception as e:
        logger.debug("could not note the shared drive's origin: %s", e)
    return remote


def letter_absent(drive: str) -> bool:
    """True when Windows has no drive at all under this letter (GetDriveTypeW
    says DRIVE_NO_ROOT_DIR) -- never for a local disk or a mapped share."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        return ctypes.windll.kernel32.GetDriveTypeW(drive + "\\") == 1
    except Exception:
        return False


def maybe_remap(nas_root, run=subprocess.run, mapped=mapped_remote,
                remembered=remembered_remote, learned=learned_remote,
                probe_fn=probe, absent=letter_absent) -> Optional[str]:
    """Put a vanished drive-letter mapping back. Returns a line for the log when
    something was tried, else None. Never raises.

    Only when the letter is not mapped at all (2250) and no drive of any kind
    is under it (a local disk also answers 2250); only to a share that
    answers right now; ``net use`` with no password (the logged-on user's own
    access, no prompt) and a time limit. Rate-limited by the caller."""
    try:
        drive = drive_letter(nas_root)
        if not drive:
            return None
        rc, _ = mapped(drive)
        if rc != _ERROR_NOT_CONNECTED or not absent(drive):
            return None
        remote = learned(drive) or remembered(drive)
        if not remote:
            return ("%s is not mapped and no share is known for it; map it again "
                    "by hand (net use %s \\\\server\\share)" % (drive, drive))
        rest = str(nas_root)[len(drive):].lstrip("\\/")
        target = str(Path(remote) / rest) if rest else remote
        state, detail = probe_fn(target)
        if state != OK:
            return ("%s is not mapped; its share %s does not answer either (%s)"
                    % (drive, remote, detail))
        try:
            r = run(["net", "use", drive, remote, "/persistent:yes"],
                    stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
            ok = r.returncode == 0
            out = (r.stdout or r.stderr or "").strip().splitlines()
            said = out[-1] if out else ""
        except Exception as e:
            ok, said = False, describe_error(e)
        said = said.encode("ascii", "replace").decode("ascii")
        return ("%s was not mapped; mapped it again to %s: %s"
                % (drive, remote, "done" if ok else "failed (%s)" % said))
    except Exception as e:
        return "could not try to map the shared drive again: %s" % describe_error(e)


# ---------------------------------------------------------------------------
# Watching the drive during a run
# ---------------------------------------------------------------------------

class ShareMonitor:
    """Whether the shared drive is there, with one log line on losing it, one
    an hour while it stays away, and one on getting it back -- never one per
    cycle (a line every 30 seconds through an outage buries everything)."""

    REMIND_EVERY_S = 3600.0

    def __init__(self, nas_root, timeout: Optional[float] = None,
                 clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], datetime] = datetime.now,
                 probe_fn=None, remap_fn=None, remember_fn=None):
        self.nas_root = nas_root
        self.timeout = PROBE_TIMEOUT_S if timeout is None else timeout
        self._clock, self._now = clock, now
        # Looked up when used (not bound here) so a test can stand in for them.
        self._probe = probe_fn or (lambda root, timeout: probe(root, timeout))
        self._remap = remap_fn or (lambda root: maybe_remap(root))
        self._remember = remember_fn or (lambda root: remember_origin(root))
        self._lock = threading.Lock()
        self._last_at: Optional[float] = None
        self._last_ok = True
        self.down_since: Optional[datetime] = None
        self.detail = ""
        self._reminded_at: Optional[float] = None
        self._remap_at: Optional[float] = None
        self._origin_noted = False
        self.last_state = OK

    def check(self, max_age: float = 0.0) -> bool:
        """True when the drive answered. ``max_age`` > 0 reuses an answer that
        young (for callers asking many times in a row)."""
        if not self.nas_root:
            return True
        with self._lock:
            t = self._clock()
            if max_age > 0 and self._last_at is not None and t - self._last_at <= max_age:
                return self._last_ok
            state, detail = self._probe(self.nas_root, self.timeout)
            if state != OK and (self._remap_at is None
                                or t - self._remap_at >= REMAP_EVERY_S):
                self._remap_at = t
                said = self._remap(self.nas_root)
                if said:
                    logger.warning("Shared drive: %s", said)
                    state, detail = self._probe(self.nas_root, self.timeout)
            ok = state == OK
            self.last_state = state
            self._note(ok, detail, t)
            if ok and not self._origin_noted:
                self._origin_noted = True
                try:
                    self._remember(self.nas_root)
                except Exception:
                    pass
            self._last_at, self._last_ok = t, ok
            return ok

    def _note(self, ok: bool, detail: str, t: float) -> None:
        if not ok:
            self.detail = detail
            if self.down_since is None:
                self.down_since = self._now()
                self._reminded_at = t
                logger.warning("Shared drive unavailable since %s -- waiting. No "
                               "shared-drive work until it answers; claims and local "
                               "work are kept. (%s: %s)",
                               self.down_since.strftime("%H:%M"), self.nas_root, detail)
            elif t - (self._reminded_at or t) >= self.REMIND_EVERY_S:
                self._reminded_at = t
                logger.warning("Shared drive still unavailable (since %s, %d min) -- "
                               "waiting. (%s)", self.down_since.strftime("%H:%M"),
                               self.minutes_down(), detail)
        elif self.down_since is not None:
            logger.info("Shared drive back after %d min (unavailable since %s); "
                        "resuming.", self.minutes_down(), self.down_since.strftime("%H:%M"))
            self.down_since, self.detail = None, ""

    def minutes_down(self) -> int:
        if self.down_since is None:
            return 0
        return int((self._now() - self.down_since).total_seconds() // 60)

    def reason(self) -> str:
        """For the node's status report."""
        if self.down_since is None:
            return ""
        return ("shared drive unavailable since %s -- waiting (%s)"
                % (self.down_since.strftime("%Y-%m-%d %H:%M"), self.detail))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def wait_for_shared_drive(nas_root, stop_requested: Callable[[], bool],
                          sleep: Callable[[float], None] = time.sleep,
                          monitor: Optional[ShareMonitor] = None,
                          first_wait: float = 15.0, longest_wait: float = 300.0,
                          give_up_after: Optional[float] = None,
                          clock: Callable[[], float] = time.monotonic) -> str:
    """Wait until the shared drive answers. Returns OK, MISSING (the drive
    answers but the folder is not there: a setting to fix, so the caller refuses
    to start, as before), 'stop' (a stop was requested while waiting) or DOWN
    (only when ``give_up_after`` ran out; the watcher passes none).

    WHY wait instead of failing: a watcher started during an outage -- by a
    person, at login, by its scheduled task -- would crash at the first
    look at the drive, and nothing would start it again once the drive was back.
    Waits grow from ``first_wait`` to ``longest_wait`` seconds; the stop flag
    is read before every wait.
    """
    mon = monitor or ShareMonitor(nas_root)
    started = clock()
    pause = first_wait
    while True:
        if stop_requested():
            return "stop"
        if mon.check():
            return OK
        if mon.last_state == MISSING:
            return MISSING
        if give_up_after is not None and clock() - started >= give_up_after:
            return DOWN
        logger.info("Waiting %.0f s for the shared drive (%s).", pause, nas_root)
        # Sleep in short steps so a stop request is seen within seconds.
        waited = 0.0
        while waited < pause:
            if stop_requested():
                return "stop"
            step = min(5.0, pause - waited)
            sleep(step)
            waited += step
        pause = min(longest_wait, pause * 2)


# ---------------------------------------------------------------------------
# The stop flag while stuck
# ---------------------------------------------------------------------------

def _hard_exit(code: int) -> None:
    """End the process at once (no clean-up that could touch the dead drive)."""
    os._exit(code)


class StopWatchdog(threading.Thread):
    """Ends the watcher when a stop was requested and its loop is stuck on an
    unavailable shared drive.

    WHY: the loop reads the stop flag between items, and a call on a dead drive
    can keep it from getting there for an hour or more -- people had to kill
    watchers by hand in exactly that state. Exits only when ALL of these hold:
    the flag is there; the loop has not come round for ``grace_s``; no pose is
    running (a pose is local GPU work that will finish, protected up to
    ``pose_protect_s``); and the drive does not answer right now. Exiting is
    safe then: the local database survives a process exit, files written to the
    drive go through a temporary name first, and startup recovery puts back
    whatever was in hand. The flag is left in place, as for any stop.
    """

    # Ten minutes without the loop coming round: well past any single call on
    # a drive that answers, far short of the hour-plus hangs seen.
    GRACE_S = 600.0
    # A pose (local GPU work, ~15 min) is left to finish for up to this long.
    POSE_PROTECT_S = 2700.0
    POLL_S = 15.0

    def __init__(self, stop_requested: Callable[[], bool],
                 last_progress: Callable[[], Optional[float]],
                 posing_since: Callable[[], Optional[float]],
                 share_ok: Callable[[], bool],
                 grace_s: Optional[float] = None, pose_protect_s: Optional[float] = None,
                 poll_s: Optional[float] = None,
                 exit_fn: Optional[Callable[[int], None]] = None,
                 clock: Callable[[], float] = time.monotonic):
        super().__init__(name="stop watchdog", daemon=True)
        self._stop_requested, self._last_progress = stop_requested, last_progress
        self._posing_since, self._share_ok = posing_since, share_ok
        self.grace_s = self.GRACE_S if grace_s is None else grace_s
        self.pose_protect_s = self.POSE_PROTECT_S if pose_protect_s is None else pose_protect_s
        self.poll_s = self.POLL_S if poll_s is None else poll_s
        self._exit = exit_fn or (lambda code: _hard_exit(code))
        self._clock = clock
        self._halt = threading.Event()

    def halt(self) -> None:
        self._halt.set()

    def run(self) -> None:
        while not self._halt.wait(self.poll_s):
            why = self.reason_to_exit()
            if why:
                self.exit_now(why)
                return

    def reason_to_exit(self) -> Optional[str]:
        try:
            if not self._stop_requested():
                return None
            last = self._last_progress()
            now = self._clock()
            if last is None or now - last < self.grace_s:
                return None
            posing = self._posing_since()
            if posing is not None and now - posing < self.pose_protect_s:
                return None
            if self._share_ok():
                return None
            return ("stop requested, and the loop has been stuck for %d min on a shared "
                    "drive that does not answer" % ((now - last) // 60))
        except Exception as e:
            logger.debug("stop watchdog check failed: %s", e)
            return None

    def exit_now(self, why: str) -> None:
        logger.warning("Exiting now: %s. Whatever was in hand is put back at the next "
                       "start; the stop flag is left in place.", why)
        for h in list(logging.getLogger().handlers):
            try:
                h.flush()
            except Exception:
                pass
        self._exit(0)
