"""The checks a watcher passes before it starts: one watcher per machine, and
no stop request left behind.

ONE WATCHER PER MACHINE
-----------------------
Two watchers working from one machine's state database is never a valid
setup: they pick work from the same rows, claim the same items and move the
same files. A named operating-system lock (a Windows "mutex") stops it. The
first watcher to start creates the lock and keeps it until its process ends
-- however it ends, a forced kill included, because the operating system
removes the lock with the last process holding it. Any later start finds the
lock and refuses.

Every way of starting a watcher takes the SAME lock: the command line
(``mousereach-watch``, cli.main_watch) and the Watcher Control panel's Start
and Run Once buttons. WHY the panel too: its watcher runs inside the panel's
own program, and looking for another watcher only by reading the
command lines of running programs -- which may not show a watcher started by
another Windows login -- could start a second watcher next to one
already running.

"Held" means another watcher exists. "Access denied" is read as held as
well: a lock created by another Windows login can be refused to this one,
and that refusal is itself the proof that the lock exists.

On a system that is not Windows there is no such lock (as before): a start
is allowed and the log says the check is not available.

A STOP REQUEST LEFT BEHIND
--------------------------
A watcher stops when it finds ``watcher_stop.flag`` in the machine's
processing folder. The update
procedure relies on that file keeping a stopped watcher stopped until the
person updating deletes it. A file forgotten there made every later start
exit at once with one line in the log -- a scheduled task even reported
success -- so the machine silently did nothing. ``stop_flag_left`` lets every
start find it and refuse loudly, naming the file and its age. The file is
never deleted here: whether the update it belongs to is finished is the
person's call.

ASCII-only text (Windows consoles cannot print Unicode).
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# The lock's name. Shared by every way of starting a watcher; "Global\" makes
# it one lock for the whole machine, not one per Windows login.
WATCHER_LOCK_NAME = "Global\\mousereach-watcher-singleton"

# Exit code of a start refused because a stop flag is waiting. Not 0: a
# scheduled task must show that the watcher did not run.
EXIT_STOP_FLAG_LEFT = 3

_ERROR_ALREADY_EXISTS = 183
_ERROR_ACCESS_DENIED = 5


class LockUnavailable(Exception):
    """The lock could not be checked at all (not Windows, or the call failed)."""


def _kernel32():
    import ctypes
    return ctypes.WinDLL("kernel32", use_last_error=True)


def acquire_watcher_lock(name: str = WATCHER_LOCK_NAME):
    """Take the one-watcher-per-machine lock.

    Returns a handle to keep for as long as the watcher runs (pass it to
    release_watcher_lock when the watcher ends; a process that simply exits
    lets the operating system release it), or None when another watcher
    already holds the lock. Raises LockUnavailable when the lock cannot be
    checked here."""
    if sys.platform != "win32":
        raise LockUnavailable("no machine-wide lock on this system")
    try:
        import ctypes
        k = _kernel32()
        k.CreateMutexW.restype = ctypes.c_void_p
        handle = k.CreateMutexW(None, False, name)
        err = ctypes.get_last_error()
    except Exception as e:
        raise LockUnavailable(str(e))
    if not handle:
        if err == _ERROR_ACCESS_DENIED:
            return None                   # it exists, made by another login
        raise LockUnavailable(f"CreateMutexW failed (error {err})")
    if err == _ERROR_ALREADY_EXISTS:
        _close(handle)                    # never keep a handle to someone else's lock
        return None
    return handle


_ERROR_FILE_NOT_FOUND = 2
_SYNCHRONIZE = 0x00100000


def watcher_lock_exists(name: str = WATCHER_LOCK_NAME):
    """Is the one-watcher-per-machine lock held right now? True, False, or
    None when that cannot be checked here (not Windows, or the call failed).

    Only LOOKS: the lock is opened, never created, and the handle is closed at
    once, so asking can never itself make a watcher's start refuse. "Access
    denied" means the lock exists (created by another login), so True.
    WHY: a command that must not run beside the watcher (mousereach-archive)
    cannot read another login's program command lines, but every login sees
    this lock (health.watcher_presence)."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        k = _kernel32()
        k.OpenMutexW.restype = ctypes.c_void_p
        k.OpenMutexW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
        handle = k.OpenMutexW(_SYNCHRONIZE, False, name)
        err = ctypes.get_last_error()
    except Exception:
        return None
    if handle:
        _close(handle)
        return True
    if err == _ERROR_ACCESS_DENIED:
        return True
    if err == _ERROR_FILE_NOT_FOUND:
        return False
    return None


def release_watcher_lock(handle) -> None:
    """Give the lock back (the watcher this handle was taken for has ended)."""
    if handle:
        _close(handle)


def _close(handle) -> None:
    try:
        import ctypes
        k = _kernel32()
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        k.CloseHandle(handle)
    except Exception as e:
        logger.debug(f"could not close the watcher lock handle ({e})")


def stop_flag_left(flag: Optional[Path]) -> Optional[Tuple[Path, float]]:
    """(path, age in seconds) when the stop flag ``flag`` exists, else None.
    Never raises; an unreadable folder counts as no flag (the watcher's own
    loop reads it the same way)."""
    if flag is None:
        return None
    try:
        st = Path(flag).stat()
    except OSError:
        return None
    return Path(flag), max(0.0, time.time() - st.st_mtime)


def describe_age(seconds: float) -> str:
    """'45 minutes', '3.5 hours', '2.0 days'."""
    if seconds < 2 * 3600:
        return f"{seconds / 60:.0f} minutes"
    if seconds < 2 * 86400:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


def stop_flag_message(path: Path, age_s: float, what: str = "the watcher") -> str:
    """The refusal a person reads when a start finds a stop flag waiting."""
    return (f"Not started: a stop request is waiting in {path} (left "
            f"{describe_age(age_s)} ago). It makes {what} stop as soon as it "
            f"starts. It is left by a stop or an update; if that is finished, "
            f"delete the file and start again.")
