"""
Pause the watcher while a recording program is running.

WHY THIS EXISTS
---------------
A GPU node in a behaviour room does two jobs on one machine: it RECORDS
videos of animals, and when nobody is recording it POSES videos (DeepLabCut,
~14 minutes of GPU per video). The two cannot share the machine: a pose
saturates the GPU, the CPU and the disk, and a recording that drops frames
because of it is lost behaviour that can never be filmed again. A pose can
always be run later. So recording must always win.

The node cannot know a recording schedule -- people record when the animals
are ready, not on a calendar. What it CAN see is whether the recording
program is open. Operators close the recording program when they are not
recording, so "the recording program is running" is the signal to stand
aside, and "it has been closed for a while" is the signal to go back to work.

WHAT IS CONFIGURED
------------------
``watcher.pause_while_running`` in ~/.mousereach/config.json: a list of
program names as the operating system shows them (on Windows, the name in
Task Manager's Details tab, for example ``recorder.exe``). EMPTY BY DEFAULT,
and empty means exactly the behaviour from before this module existed: the
watcher never checks for any program. The list is configuration, not code,
because which program a lab records with is that lab's business.

``watcher.pause_resume_grace_seconds`` (default 120): how long every listed
program must have been closed before work starts again. WHY a grace period:
an operator often closes the recording program and reopens it a moment later
(a crashed camera, a changed setting, the next animal). Starting a 14-minute
pose in that gap would either make the new recording compete with it or throw
the pose away seconds later.

HOW NAMES ARE MATCHED
---------------------
Case-insensitively (Windows does not care about case, and neither does a
person typing a name into a box). Blank entries are ignored. An entry typed
as a full path, or without its ``.exe``, still matches the program -- a
setting a novice typed slightly differently from Task Manager must not
silently fail to protect a recording.

FAIL SAFE
---------
If programs are configured but the check itself cannot run (psutil is not
installed, or listing processes fails), the guard reports a reason and the
watcher stays PAUSED. WHY: the one outcome this module exists to prevent is
posing on top of a recording; a node that sits idle because the check broke
is a nuisance a person sees in the status line, a ruined recording is not
recoverable.

Two entry points:

  * ``RecordingGuard`` -- keeps state across calls (the grace timer, a short
    cache of the process list). The watcher holds one for its whole life.
  * ``pause_reason`` -- one question, "should the watcher be paused, and
    why?", combining the hand-set pause flag with the guard. The status line,
    the CLI and the GUI ask this so they all say the same thing.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)

# The hand-set pause (mousereach-watch-toggle, the GUI Pause button). Named
# here once so every caller reports it the same way.
PAUSE_FLAG_NAME = "watcher_paused.flag"
HAND_PAUSE_REASON = f"paused by hand ({PAUSE_FLAG_NAME})"


def _ascii(text) -> str:
    """Printable ASCII only. WHY: these reasons are logged and printed, and a
    Windows console cannot encode every character a process or an exception
    message may contain -- one would crash the logging call."""
    return str(text).encode("ascii", "replace").decode("ascii")


def _match_key(name) -> str:
    """The form a configured name and a running process are compared in.

    Lower case; only the last path component (someone may paste the full path
    of the program); a trailing ``.exe`` dropped (someone may type the name
    without it). Returns '' for a blank or non-text entry, which is ignored.
    """
    if not isinstance(name, str):
        return ""
    base = name.strip().replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
    if base.endswith(".exe"):
        base = base[:-4]
    return base


def clean_names(names: Optional[Iterable[str]]) -> List[str]:
    """The configured names with blanks removed and duplicates dropped,
    spelled as the person typed them (trimmed). A single string is treated as
    a one-item list, so a hand-edited config that forgot the brackets still
    protects the recording."""
    if not names:
        return []
    if isinstance(names, str):
        names = [names]
    out, seen = [], set()
    for n in names:
        key = _match_key(n)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(n.strip())
    return out


def _windows_process_names() -> Optional[List[str]]:
    """Every running program's name (e.g. "recorder.exe") from ONE Windows
    snapshot, or None when that is not available here (see
    windows_process_table for why not psutil)."""
    table = windows_process_table()
    return [exe for _pid, _ppid, exe in table] if table else None


def windows_process_table():
    """[(pid, parent pid, program name), ...] for every process, from ONE
    Windows snapshot, or None when that is not available here.

    WHY not psutil on Windows: psutil opens each process in turn, and a process
    it may not open (another user's, a service) sends it down a slow path. On a
    busy machine a single listing measured over a minute and a half. The guard
    asks every few seconds, while a pose runs and from the panel's refresh --
    a check that slow would leave a pose running for minutes after recording
    started. The Toolhelp snapshot lists every program name in one call,
    without opening any process, in milliseconds.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _ProcessEntry(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD),
                        ("cntUsage", wintypes.DWORD),
                        ("th32ProcessID", wintypes.DWORD),
                        ("th32DefaultHeapID", ctypes.c_size_t),
                        ("th32ModuleID", wintypes.DWORD),
                        ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD),
                        ("pcPriClassBase", ctypes.c_long),
                        ("dwFlags", wintypes.DWORD),
                        ("szExeFile", ctypes.c_wchar * 260)]

        TH32CS_SNAPPROCESS = 0x00000002
        INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.Process32FirstW.restype = wintypes.BOOL
        kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
        kernel32.Process32NextW.restype = wintypes.BOOL
        kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if not snap or snap == INVALID_HANDLE_VALUE:
            return None
        try:
            entry = _ProcessEntry()
            entry.dwSize = ctypes.sizeof(_ProcessEntry)
            rows = []
            ok = kernel32.Process32FirstW(snap, ctypes.byref(entry))
            while ok:
                rows.append((int(entry.th32ProcessID),
                             int(entry.th32ParentProcessID), entry.szExeFile))
                ok = kernel32.Process32NextW(snap, ctypes.byref(entry))
        finally:
            kernel32.CloseHandle(snap)
        # An empty list cannot be right (this very process is running), so it
        # is treated as "could not look" and psutil gets its turn.
        return rows or None
    except Exception as e:
        logger.debug(f"process snapshot unavailable, using psutil: {e}")
        return None


def running_programs(names: Iterable[str]) -> List[str]:
    """Which of ``names`` are running right now, in the order configured and
    spelled as configured. [] when ``names`` is empty (no process is listed).

    On Windows the list comes from one system snapshot
    (_windows_process_names); elsewhere, or if that fails, from psutil.

    Raises ImportError when psutil is needed but not installed, and whatever
    psutil raises when the process list cannot be read. Deliberately NOT
    swallowed here: the caller must be able to tell "nothing is running" from
    "could not look", because the second one has to keep the watcher paused.
    """
    wanted = {}
    for n in clean_names(names):
        wanted.setdefault(_match_key(n), n)
    if not wanted:
        return []

    found = set()
    snapshot = _windows_process_names()
    if snapshot is not None:
        for name in snapshot:
            key = _match_key(name or "")
            if key in wanted:
                found.add(key)
        return [display for key, display in wanted.items() if key in found]

    import psutil  # noqa: WPS433 -- optional at import time; see docstring

    for proc in psutil.process_iter(["name"]):
        try:
            key = _match_key(proc.info.get("name") or "")
        except Exception:
            # A process that exited or cannot be inspected mid-listing says
            # nothing about the ones we are looking for.
            continue
        if key in wanted:
            found.add(key)
    return [display for key, display in wanted.items() if key in found]


def _join(names: List[str]) -> str:
    return ", ".join(names)


class RecordingGuard:
    """Answers "is a recording program running, or did one close too
    recently?" for the life of one watcher.

    ``reason()`` returns None when work may run, otherwise one ASCII line
    saying why not. The process list is read at most once every
    ``cache_seconds`` (WHY: the watcher and a running pose both ask
    repeatedly, and listing every process on the machine each time is wasted
    work). The grace timer starts at the first check that finds no listed
    program after one was running -- or after a check that could not tell,
    since that may have hidden a recording.

    ``clock`` and ``scan`` are injectable so tests can move time and fake the
    process list without touching the real machine.
    """

    def __init__(self, names, grace_seconds: int = 120,
                 clock: Callable[[], float] = time.monotonic,
                 scan: Callable[[Iterable[str]], List[str]] = running_programs,
                 cache_seconds: float = 5):
        self.names = clean_names(names)
        try:
            self.grace_seconds = max(0, int(grace_seconds))
        except (TypeError, ValueError):
            self.grace_seconds = 120
        self._clock = clock
        self._scan = scan
        self._cache_seconds = max(0.0, float(cache_seconds or 0))
        self._scanned_at: Optional[float] = None
        self._running: List[str] = []
        self._error: Optional[str] = None
        # Names seen running (or None when nothing ever was), and when every
        # one of them was first seen gone.
        self._last_seen: Optional[List[str]] = None
        self._clear_since: Optional[float] = None

    @property
    def enabled(self) -> bool:
        """True when any program is configured. False = never pauses."""
        return bool(self.names)

    def inherit_history(self, previous: Optional["RecordingGuard"]) -> None:
        """Carry an older guard's memory (what was seen running, and when it
        closed) into this one.

        WHY: a running watcher builds a new guard when someone edits the
        program list. Without this, adding a second program during the grace
        period after the first one closed would forget that grace period and
        start a pose at once -- the gap the grace period exists to protect.
        Carrying it over can only make the watcher wait longer, never shorter.
        """
        if previous is None:
            return
        self._last_seen = list(previous._last_seen) if previous._last_seen else None
        self._clear_since = previous._clear_since
        if previous._error is not None and self._last_seen is None:
            self._last_seen = list(self.names)

    def _refresh(self, now: float) -> None:
        if (self._scanned_at is not None
                and now - self._scanned_at < self._cache_seconds):
            return
        self._scanned_at = now
        try:
            running = list(self._scan(self.names))
            self._error = None
        except ImportError:
            running = None
            # The fix is named, because the person reading this line is
            # looking at a PC that has stopped working for no visible reason.
            self._error = ("psutil is not installed (run mousereach-setup, or "
                           "install it with: pip install psutil)")
        except Exception as e:
            running = None
            self._error = f"{type(e).__name__}: {e}"
        if running is None:
            # Could not look. Treat it as possibly running: the grace period
            # starts again only once a check succeeds and finds nothing.
            self._running = []
            self._last_seen = self._last_seen or list(self.names)
            self._clear_since = None
            return
        self._running = running
        if running:
            self._last_seen = running
            self._clear_since = None
        elif self._last_seen is not None and self._clear_since is None:
            self._clear_since = now

    def reason(self) -> Optional[str]:
        """None when nothing blocks work; else why work must wait (ASCII)."""
        if not self.names:
            return None
        now = self._clock()
        self._refresh(now)
        if self._error is not None:
            return _ascii(f"cannot check for recording programs: {self._error}")
        if self._running:
            verb = "is" if len(self._running) == 1 else "are"
            return _ascii(f"{_join(self._running)} {verb} running")
        if self._clear_since is not None:
            waited = max(0.0, now - self._clear_since)
            if waited < self.grace_seconds:
                return _ascii(f"{_join(self._last_seen or self.names)} closed "
                              f"{int(waited)} s ago; resuming after "
                              f"{self.grace_seconds} s")
        return None


def pause_reason(processing_root: Optional[Path] = None,
                 config=None,
                 guard: Optional[RecordingGuard] = None) -> Optional[str]:
    """Why the watcher should be paused right now, or None to work.

    The hand-set flag wins over the guard (WHY: a person pausing on purpose
    is the most specific instruction there is, and the reason shown should be
    the one they can undo). Otherwise the guard's reason.

    ``guard``: the watcher passes its own long-lived guard, so the grace
    timer carries across polls. Without one, a guard is built from ``config``
    (or from ~/.mousereach/config.json when ``config`` is None too). A freshly
    built guard has no history, so it reports a program that is running but
    cannot report a grace period -- fine for a one-off status question.

    Never raises: an unreadable processing root skips the flag; an unreadable
    configuration means no programs are configured.
    """
    root = processing_root
    if root is None:
        try:
            from mousereach.config import require_processing_root
            root = require_processing_root()
        except Exception:
            root = None
    if root is not None:
        try:
            if (Path(root) / PAUSE_FLAG_NAME).exists():
                return HAND_PAUSE_REASON
        except OSError:
            pass
    if guard is None:
        if config is None:
            try:
                from mousereach.config import WatcherConfig
                config = WatcherConfig.load()
            except Exception as e:
                logger.debug(f"pause_reason: watcher config unreadable ({e})")
                config = None
        if config is None:
            return None
        guard = RecordingGuard(
            getattr(config, "pause_while_running", None) or [],
            grace_seconds=getattr(config, "pause_resume_grace_seconds", 120))
    return guard.reason()
