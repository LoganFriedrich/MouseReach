"""Plain-language pipeline health -- the dashboard's one health line.

WHY: an operator should never need a terminal or a log file to know whether
the auto-processor is alive and whether anything is stuck waiting for a
person. Every line here is a sentence a person can act on; diagnostics stay
inside the code, self-healing where possible, and only the actionable
remainder surfaces.

ASCII-only output (Windows consoles cannot print Unicode).
"""
from __future__ import annotations

import re
import time
from typing import List, Optional

# The watcher itself, started either way: the console script
# (".../mousereach-watch.exe", or a python child whose command line says the
# same) or `python -c "...main_watch()"`. WHY the look-ahead: the other watcher
# commands share the prefix -- mousereach-watch-status, -toggle, -recorders --
# and matching them made a PC with no watcher look RUNNING whenever someone
# checked its status, so the panel refused to start one.
_WATCHER_COMMAND = re.compile(r"\bmain_watch\b|mousereach-watch(?![-\w])")


def is_watcher_command(cmdline: str) -> bool:
    """True when a process command line is the watcher daemon itself."""
    return bool(_WATCHER_COMMAND.search(cmdline or ""))


def _watcher_processes(first_only: bool = False) -> list:
    """psutil.Process objects for every running watcher daemon (at most one
    when ``first_only``). Raises ImportError when psutil is missing.

    WHY names first: reading EVERY process's command line through psutil took
    about 85 seconds per call on a busy server, and the Watcher Control panel
    asks every few seconds on the GUI thread -- the panel froze. One Windows
    snapshot lists every program name in milliseconds, so only Python and
    MouseReach programs (the only ones a watcher can be) have their command
    lines read. Where the snapshot is unavailable, every process is checked as
    before.
    """
    import psutil
    candidates = None
    try:
        from mousereach.watcher.recording_guard import windows_process_table
        table = windows_process_table()
        if table:
            candidates = [pid for pid, _ppid, exe in table
                          if "python" in (exe or "").lower()
                          or "mousereach" in (exe or "").lower()]
    except Exception:
        candidates = None

    found = []
    if candidates is not None:
        for pid in candidates:
            try:
                p = psutil.Process(pid)
                if is_watcher_command(" ".join(p.cmdline() or [])):
                    found.append(p)
                    if first_only:
                        break
            except Exception:
                continue
        return found
    for p in psutil.process_iter(["cmdline"]):
        cmd = " ".join(p.info.get("cmdline") or [])
        if is_watcher_command(cmd):
            found.append(p)
            if first_only:
                break
    return found


def watcher_running() -> Optional[int]:
    """The auto-processor's pid, or None when it is not running."""
    try:
        found = _watcher_processes(first_only=True)
    except Exception:
        return None
    return found[0].pid if found else None


def _current_pause_reason() -> Optional[str]:
    """Why the watcher on this PC is paused, or None. Never raises."""
    try:
        from mousereach.watcher.recording_guard import pause_reason
        return pause_reason()
    except Exception:
        return None


def pause_line(reason: str) -> str:
    """The health line for a paused watcher, with what a person does about it.

    WHY on the dashboard: a paused watcher looks exactly like a stuck one --
    RUNNING, nothing failed, and the counts do not move. The pause may be the
    hand pause, a recording program left open, or a check that cannot run, and
    only this line tells an operator which, before they press Restart.
    """
    from mousereach.watcher.recording_guard import HAND_PAUSE_REASON
    if reason == HAND_PAUSE_REASON:
        return ("It is PAUSED by hand, so it starts no new work. Press Resume in "
                "Watcher Control (or run mousereach-watch-toggle --resume) to "
                "let it work again.")
    if reason.startswith("cannot check"):
        return ("It is PAUSED: %s. It stays paused until that check works, "
                "because recording must always win." % reason)
    return ("It is PAUSED: %s. Close the recording program when you are not "
            "recording; work starts again by itself a little later." % reason)


def health_report(db=None) -> List[str]:
    """Plain-English health lines; the first is always the headline."""
    lines: List[str] = []
    pid = watcher_running()
    if pid:
        lines.append("The auto-processor is RUNNING.")
        reason = _current_pause_reason()
        if reason:
            lines.append(pause_line(reason))
    else:
        lines.append("The auto-processor is NOT RUNNING -- videos will wait "
                     "until it is started (Restart button below).")
    if db is not None:
        try:
            failed = db.get_videos_in_state("failed")
            unres = db.get_videos_in_state("unresolvable")
            if failed:
                names = ", ".join(v["video_id"] for v in failed[:3])
                lines.append(
                    "%d video(s) FAILED and wait for a person (fix the cause, "
                    "then select the row and press Re-run): %s%s"
                    % (len(failed), names, " ..." if len(failed) > 3 else ""))
            if unres:
                lines.append(
                    "%d video(s) are parked as unresolvable -- each carries "
                    "its reason (File Details shows it)." % len(unres))
            if not failed and not unres:
                lines.append("Nothing is stuck: no failed or parked videos.")
        except Exception as e:
            lines.append("Could not read the pipeline database: %s" % e)
    return lines


def restart_watcher(stop_delay: float = 3.0, verify_wait: float = 12.0,
                    graceful_wait: float = 90.0, force: bool = False):
    """Stop the running auto-processor (if any) and start a fresh one.

    Machine-agnostic: launches through this environment's own python with the
    same entry the service wrapper uses, so no machine path is needed; the
    single-instance mutex prevents doubles. Returns (ok, message).

    The stop is a request, not a kill (see below). ``graceful_wait`` is how
    long to wait for the watcher to finish its current item and exit; a caller
    that would rather lose that work than wait can pass ``force=True``, which
    kills whatever is left after that wait. The default never kills, because
    the usual caller is a button in the GUI and the usual item in flight is a
    fourteen-minute pose.
    """
    import subprocess
    import sys
    try:
        import psutil
    except ImportError:
        return False, "psutil is not installed; cannot manage the process."
    # Match BOTH spellings. A watcher started from the console script is
    # 'mousereach-watch.exe' with a python child whose command line says the
    # same; only a watcher started as `python -c "...main_watch()"` contains
    # 'main_watch'. Matching one spelling meant this stopped nothing on half
    # the machines in the lab, and then reported success -- the replacement
    # exited on the singleton mutex and the OLD code kept running (found on a
    # GPU node, 2026-09-13).
    # Same fast lookup as watcher_running (see _watcher_processes for why).
    targets = _watcher_processes()

    # Stopping is a REQUEST, not a kill.
    #
    # There is no polite kill available here. On Windows psutil's terminate()
    # is documented as an alias for kill(), and a watcher started the way this
    # function starts one is detached with no console, so a Ctrl+C event
    # cannot reach it either. A hard kill part-way through a pose throws away
    # about fourteen minutes of GPU, and before startup reclaim existed it
    # stranded the video outright.
    #
    # So: drop watcher_stop.flag, which the loop checks between work items,
    # and let it finish what it is doing. If it is busy, say so and change
    # nothing -- an honest "not now" beats a restart that costs a pose.
    from mousereach.config import require_processing_root
    try:
        stop_flag = require_processing_root() / "watcher_stop.flag"
    except Exception as e:
        return False, f"Cannot find the pipeline folder to signal a stop: {e}"

    stopped = 0
    try:
        if targets:
            stop_flag.write_text("restart requested\n", encoding="utf-8")
            gone, alive = psutil.wait_procs(targets, timeout=graceful_wait)
            stopped = len(gone)
            if alive and not force:
                # Clear the request first, so the watcher carries on normally.
                stop_flag.unlink(missing_ok=True)
                # It may have exited in the instant between giving up and
                # clearing the flag. If it did, start a replacement rather
                # than leave the machine with no watcher at all.
                if [p for p in alive if p.is_running()]:
                    return False, (
                        "The auto-processor is busy, most likely part-way "
                        "through a pose, which takes about fourteen minutes. "
                        "It has not stopped yet, so nothing was killed and no "
                        "work was lost. It is still running normally. Try "
                        "again when it is idle, or stop it from its own "
                        "console with Ctrl+C.")
            for p in alive:
                try:
                    p.kill()
                    stopped += 1
                except Exception:
                    pass
            if alive:
                psutil.wait_procs(alive, timeout=10)
        # The replacement must never find this, or it exits on sight.
        stop_flag.unlink(missing_ok=True)

        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
        subprocess.Popen(
            [sys.executable, "-c",
             "from mousereach.watcher.cli import main_watch; main_watch()"],
            creationflags=flags, close_fds=True)
    finally:
        # Belt and braces: a stop flag left behind would stop the next watcher
        # the moment it starts, which is a worse failure than not restarting.
        try:
            stop_flag.unlink(missing_ok=True)
        except Exception:
            pass
    time.sleep(verify_wait)
    pid = watcher_running()
    if pid:
        return True, ("Auto-processor restarted (stopped %d, new pid %d). "
                      "Its first scan after a restart is slower (cold caches)."
                      % (stopped, pid))
    return False, ("The auto-processor did not come back up -- check the "
                   "watcher log in the pipeline's watcher_logs folder.")
