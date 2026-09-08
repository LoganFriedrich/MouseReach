"""Plain-language pipeline health -- the dashboard's one health line.

WHY: an operator should never need a terminal or a log file to know whether
the auto-processor is alive and whether anything is stuck waiting for a
person. Every line here is a sentence a person can act on; diagnostics stay
inside the code, self-healing where possible, and only the actionable
remainder surfaces.

ASCII-only output (Windows consoles cannot print Unicode).
"""
from __future__ import annotations

import time
from typing import List, Optional


def watcher_running() -> Optional[int]:
    """The auto-processor's pid, or None when it is not running."""
    try:
        import psutil
        for p in psutil.process_iter(["cmdline"]):
            cmd = " ".join(p.info.get("cmdline") or [])
            if "main_watch" in cmd or "mousereach-watch" in cmd:
                return p.pid
    except Exception:
        return None
    return None


def health_report(db=None) -> List[str]:
    """Plain-English health lines; the first is always the headline."""
    lines: List[str] = []
    pid = watcher_running()
    if pid:
        lines.append("The auto-processor is RUNNING.")
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


def restart_watcher(stop_delay: float = 3.0, verify_wait: float = 12.0):
    """Stop the running auto-processor (if any) and start a fresh one.

    Machine-agnostic: launches through this environment's own python with the
    same entry the service wrapper uses, so no machine path is needed; the
    single-instance mutex prevents doubles. Returns (ok, message).
    """
    import subprocess
    import sys
    try:
        import psutil
    except ImportError:
        return False, "psutil is not installed; cannot manage the process."
    stopped = 0
    for p in psutil.process_iter(["cmdline"]):
        cmd = " ".join(p.info.get("cmdline") or [])
        if "main_watch" in cmd:
            try:
                p.kill()
                stopped += 1
            except Exception:
                pass
    time.sleep(stop_delay)
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    subprocess.Popen(
        [sys.executable, "-c",
         "from mousereach.watcher.cli import main_watch; main_watch()"],
        creationflags=flags, close_fds=True)
    time.sleep(verify_wait)
    pid = watcher_running()
    if pid:
        return True, ("Auto-processor restarted (stopped %d, new pid %d). "
                      "Its first scan after a restart is slower (cold caches)."
                      % (stopped, pid))
    return False, ("The auto-processor did not come back up -- check the "
                   "watcher log in the pipeline's watcher_logs folder.")
