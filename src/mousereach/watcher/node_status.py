"""What each node is doing, written where every other machine can read it.

WHY THIS EXISTS
---------------
A node that is paused and a node that is dead look identical from anywhere else: both
simply stop appearing in the shared record. That ambiguity is not theoretical. A node
pauses whenever a recording program is open, and a recording program is routinely left
open after the day's recording is finished -- so a machine being quiet for days is the
NORMAL, correct state, and a machine whose watcher has died looks exactly the same.

The cost of not being able to tell them apart runs both ways. Treat every silence as a
fault and you raise a false alarm most weeks, which teaches everyone to ignore the
alarm. Treat every silence as normal and a genuinely dead watcher sits unnoticed --
which has happened, for about sixty hours.

So each node writes a small file saying what it is doing and why. Silence in the FILE
then means something: the node is not running at all. A pause says so in words, with
the reason, and is not an alarm.

WHY A FILE AND NOT THE SHARED DATABASE
--------------------------------------
This is a heartbeat: frequent, tiny, and worthless once it is superseded. Putting it in
the coordination database would add write traffic and lock contention to the one
resource every node must agree on, to store something no node needs to agree about.
A file per node has no schema to migrate, cannot block another machine's claim, and a
node whose file cannot be written just carries on working -- nothing here is ever
allowed to interrupt processing.

ASCII only, and every failure is swallowed: a status report that can break the thing it
reports on is worse than no status report.
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

DIR_NAME = ".node_status"

# States a node can report. Deliberately few -- this answers "should anyone care?",
# not "what exactly is it doing", which the shared record already says.
WORKING = "working"      # running, taking work
PAUSED = "paused"        # running, deliberately not taking work (with a reason)
STOPPING = "stopping"    # shutting down cleanly

# A file older than this means nobody has written it recently, so the node is not
# running. Generous on purpose: a node polls on its own interval, a shared drive can be
# slow, and clocks between machines disagree -- a threshold that is too tight
# manufactures exactly the false alarm this module exists to prevent.
STALE_AFTER = timedelta(minutes=30)


def status_dir(nas_root) -> Optional[Path]:
    """The folder these reports live in, or None when there is no usable root.

    An unset root must yield None rather than a RELATIVE path: ``Path("") / "x"`` is
    ``x``, so a blank root would silently create a status folder in whatever directory
    the watcher happened to be started from, and the reports would be invisible to
    everyone. Absolute or nothing.
    """
    if nas_root is None:
        return None
    try:
        root = Path(nas_root)
    except (TypeError, ValueError):
        return None
    if not str(root).strip() or not root.is_absolute():
        return None
    return root / "Processing" / DIR_NAME


def write(nas_root, state: str, reason: Optional[str] = None,
          hostname: Optional[str] = None, extra: Optional[dict] = None) -> bool:
    """Record what this node is doing. True if written; never raises.

    Written through a temporary file and renamed, so a reader never sees half a file.
    """
    d = status_dir(nas_root)
    if d is None:
        return False
    host = hostname or socket.gethostname()
    record = {
        "hostname": host,
        "state": state,
        "reason": reason or "",
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        record.update({k: v for k, v in extra.items() if k not in record})
    try:
        d.mkdir(parents=True, exist_ok=True)
        text = json.dumps(record, indent=2, sort_keys=True) + "\n"
        fd, tmp = tempfile.mkstemp(dir=str(d), prefix=".%s." % host, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="ascii", errors="replace") as f:
                f.write(text)
            os.replace(tmp, str(d / ("%s.json" % host)))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return True
    except Exception:
        # Never let a status report stop a node from working.
        return False


def read_all(nas_root) -> List[dict]:
    """Every node's last report, newest first. Never raises.

    Each entry gains ``age_minutes`` and ``running`` (False once the report is older
    than STALE_AFTER, which is what "this node is not running" looks like).
    """
    d = status_dir(nas_root)
    out: List[dict] = []
    if d is None or not d.is_dir():
        return out
    now = datetime.now()
    try:
        files = [p for p in d.iterdir() if p.suffix == ".json"]
    except OSError:
        return out
    for p in files:
        try:
            record = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        try:
            age = (now - datetime.fromisoformat(str(record.get("at")))).total_seconds() / 60.0
        except (TypeError, ValueError):
            age = None
        record["age_minutes"] = round(age, 1) if age is not None else None
        record["running"] = (age is not None and age <= STALE_AFTER.total_seconds() / 60.0)
        record.setdefault("hostname", p.stem)
        out.append(record)
    out.sort(key=lambda r: (r.get("age_minutes") is None, r.get("age_minutes") or 0))
    return out


def describe(nas_root) -> str:
    """One line per node, for a person: what it is doing, or that it is not running."""
    rows = read_all(nas_root)
    if not rows:
        return ("No node has reported yet. Either no watcher has run since this was "
                "added, or the shared folder cannot be read.")
    lines = []
    for r in rows:
        host = str(r.get("hostname"))[:20]
        if not r.get("running"):
            age = r.get("age_minutes")
            when = ("%.0f h ago" % (age / 60.0)) if age and age >= 60 else (
                "%s min ago" % age if age is not None else "unknown")
            lines.append("  %-20s NOT RUNNING   last said '%s' %s"
                         % (host, r.get("state", "?"), when))
        elif r.get("state") == PAUSED:
            lines.append("  %-20s paused        %s" % (host, r.get("reason") or "no reason given"))
        else:
            lines.append("  %-20s %-13s %s" % (host, r.get("state", "?"), r.get("reason") or ""))
    return "\n".join(lines)
