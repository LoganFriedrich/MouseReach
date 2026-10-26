"""Where is each session of a list? One line of state per listed session.

WHY THIS EXISTS
---------------
A lab imports recordings because some question needs particular sessions (one
animal's tray run on one day). Once imported, those sessions spread across the
pipeline -- a recording waiting to be cut, a video being posed on some GPU node,
one waiting for the processing node, one held for a person's review, one
finished -- and nothing said how far a list had got without someone asking
every machine. This answers it from shared storage alone, for any list:

    python -m mousereach.watcher.coverage --sessions-file <list> [--out <json>]

Each listed session is placed in exactly one stage, first match wins:

    analysed            its results are in Analyzed (a processing manifest)
    in review           its bundle is in a review queue (triage or deep review)
    review, parked      its bundle is set aside in a parked review folder
    <watcher state>     the shared record's state for the video (posed, being
                        posed, waiting for analysis, ...)
    waiting to be posed the cut video is at the top of the shared singles folder
    being posed         a node holds it in the shared singles claim folder
    posed, waiting      its pose is in the posed-video folder
    recording waiting / being cut / cut   from the recording's own record
    not imported        no recording holding it is in the intake folder (this
                        includes a session that has no recording at all)

Read-only: it lists folders and reads the shared coordination database
(``watcher_central.db``) without writing. The list format is the one
``mousereach-aspa-import-collages --sessions-file`` takes.

ASCII-only output.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Watcher states (shared record) -> the stage a person would call it.
STATE_STAGE = {
    "discovered": "waiting to be posed", "validated": "waiting to be posed",
    "dlc_queued": "waiting to be posed", "dlc_running": "being posed",
    "dlc_complete": "posed, waiting for analysis",
    "processing": "being analysed", "processed": "being analysed",
    "archiving": "being analysed", "archived": "analysed",
    "triage": "in review", "deep_review": "in review",
    "outdated": "being re-run", "crystallized": "analysed",
    "failed": "needs attention", "unresolvable": "needs attention",
    "quarantined": "needs attention",
}


def _names(folder) -> set:
    try:
        with os.scandir(folder) as it:
            return {e.name for e in it}
    except OSError:
        return set()


def _sessions_in_collage(stem: str) -> List[str]:
    from mousereach.aspa.import_collages import sessions_in
    return sessions_in(stem)


def session_coverage(sessions: List[str], nas_root: Path) -> Dict[str, str]:
    """``{session: stage}`` for every listed session (pipeline naming)."""
    from mousereach.config import Paths
    from mousereach.archive.core import get_archive_destination

    stage: Dict[str, str] = {}
    wanted = list(dict.fromkeys(sessions))

    # 1. finished: a processing manifest in its Analyzed folder (one listing
    #    per folder, not one check per session)
    by_folder: Dict[str, List[str]] = {}
    for s in wanted:
        try:
            by_folder.setdefault(str(get_archive_destination(s)), []).append(s)
        except Exception:
            continue
    for folder, members in by_folder.items():
        names = _names(folder)
        for s in members:
            if f"{s}_processing_manifest.json" in names:
                stage[s] = "analysed"

    # 2. review queues, and the parked review folders
    review = Path(Paths.REVIEW_ROOT) if Paths.REVIEW_ROOT else None
    if review is not None:
        for q in ("triage", "deep_review"):
            for s in _names(review / q) & set(wanted):
                stage.setdefault(s, "in review")
        for parked in (d for d in _names(review) if d.startswith("_deprioritized")):
            for q in _names(review / parked):
                for s in _names(review / parked / q) & set(wanted):
                    stage.setdefault(s, "review, parked")

    # 3. the shared record of every node's videos and recordings
    rows, collages = {}, {}
    db = Path(nas_root) / "watcher_central.db"
    if db.is_file():
        # A once-a-day report: it waits up to 10 minutes for the record to be
        # readable rather than fail -- a write that crawls over a slow share
        # can hold the file for minutes (2026-10-26 05:30, the report failed).
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
        try:
            for vid, state in con.execute("SELECT video_id, state FROM videos"):
                rows[vid] = state
            for vid, state in con.execute("SELECT video_id, state FROM pipeline_videos"):
                rows.setdefault(vid, state)
            for fn, state in con.execute("SELECT filename, state FROM pipeline_collages"):
                collages[Path(fn).stem] = state
        finally:
            con.close()
    for s in wanted:
        if s not in stage and s in rows:
            stage[s] = STATE_STAGE.get(rows[s], rows[s])

    # 4. files on the share for videos the record does not place yet
    door = Path(Paths.SINGLE_ANIMAL_OUTPUT) if Paths.SINGLE_ANIMAL_OUTPUT else None
    if door is not None:
        top = _names(door)
        held = set()
        for h in _names(door / ".inflight"):
            held |= _names(door / ".inflight" / h)
        for s in wanted:
            if s in stage:
                continue
            if f"{s}.mp4" in held:
                stage[s] = "being posed"
            elif f"{s}.mp4" in top:
                stage[s] = "waiting to be posed"
    if Paths.DLC_STAGING:
        posed = _names(Paths.DLC_STAGING)
        for s in wanted:
            if s not in stage and f"{s}.mp4" in posed:
                stage[s] = "posed, waiting for analysis"

    # 5. the recording that holds it
    if Paths.MULTI_ANIMAL_SOURCE:
        holder = {}
        for n in _names(Paths.MULTI_ANIMAL_SOURCE):
            if not n.lower().endswith((".mkv", ".mp4")):
                continue
            for s in _sessions_in_collage(Path(n).stem):
                holder.setdefault(s, Path(n).stem)
        for s in wanted:
            if s in stage:
                continue
            rec = holder.get(s)
            if rec is None:
                stage[s] = "not imported"
            else:
                st = collages.get(rec)
                stage[s] = ("recording waiting to be cut" if st is None else
                            "recording being cut" if st == "cropping" else
                            "cut; on a node, not yet in the shared record")
    for s in wanted:
        stage.setdefault(s, "not imported")
    return stage


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="mousereach-session-coverage",
        description="Where each session of a list is in the pipeline, from shared "
                    "storage only. Read-only.")
    parser.add_argument("--sessions-file", required=True,
                        help="One {date}_{animal}_{tray}{position} per line (lab or "
                             "pipeline naming), '#' comments allowed.")
    parser.add_argument("--out", default=None,
                        help="Write the result as JSON here (default: beside the list, "
                             "<list>.coverage.json).")
    args = parser.parse_args(argv)

    from mousereach.aspa.import_collages import read_sessions_file
    from mousereach.config import Paths
    if not Paths.NAS_ROOT:
        print("[FAIL] No NAS root configured (run mousereach-setup).")
        return 1
    sessions, bad = read_sessions_file(Path(args.sessions_file))
    stage = session_coverage(sessions, Path(Paths.NAS_ROOT))
    counts = Counter(stage.values())
    out = Path(args.out) if args.out else Path(args.sessions_file).with_suffix(".coverage.json")
    out.write_text(json.dumps({
        "sessions_file": str(args.sessions_file),
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "sessions": len(sessions),
        "by_stage": dict(counts.most_common()),
        "stage": stage,
        "not_session_names": bad,
    }, indent=2), encoding="utf-8")
    print(f"{len(sessions)} sessions listed ({args.sessions_file})")
    for name, n in counts.most_common():
        print(f"  {n:6d}  {name}")
    if bad:
        print(f"[!] {len(bad)} line(s) are not session names and were ignored")
    print(f"Written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
