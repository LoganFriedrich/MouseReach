"""What the watcher would do -- without doing any of it (`mousereach-watch --dry-run`).

WHY THIS EXISTS
---------------
The dry run is the step the install guide has every new machine run before its
watcher starts. It used to build the full watcher -- whose start-up runs
recovery -- and then call the real scans, which register new files and move
videos between states, so "dry run" changed the local database and touched the
shared one (found 2026-08-30: 104 recordings advanced on one machine).

This preview only reads. The watcher database is opened read-only, folders are
listed, and the next pick is worked out by the watcher's own selection code on
a stand-in that can only read. Nothing is created, moved or written, and no
single-instance lock is taken, so it can run beside a live watcher.

ASCII-only output.
"""
from __future__ import annotations

import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import List, Optional

VIDEO_EXT = (".mp4", ".mkv", ".avi")


class ReadOnlyDB:
    """The database reads work selection needs, on a read-only connection.

    Any other method -- every one that writes -- does not exist here, so a
    write attempted during a preview fails instead of happening. A missing
    database or table reads as empty (a new machine has none yet).
    """

    def __init__(self, path):
        self.path = Path(path) if path else None

    def _rows(self, sql: str, args=()) -> List[dict]:
        if self.path is None or not self.path.is_file():
            return []
        con = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in con.execute(sql, args)]
        except sqlite3.Error:
            return []
        finally:
            con.close()

    def get_videos_in_state(self, state: str) -> List[dict]:
        return self._rows("SELECT * FROM videos WHERE state = ? ORDER BY created_at DESC", (state,))

    def get_collages_in_state(self, state: str) -> List[dict]:
        return self._rows("SELECT * FROM collages WHERE state = ? ORDER BY created_at DESC", (state,))

    def get_video(self, video_id: str) -> Optional[dict]:
        rows = self._rows("SELECT * FROM videos WHERE video_id = ?", (video_id,))
        return rows[0] if rows else None

    def counts(self, table: str) -> Counter:
        return Counter(r["state"] for r in self._rows(f"SELECT state FROM {table}"))

    def ids(self, table: str, column: str) -> set:
        return {str(r[column]) for r in self._rows(f"SELECT {column} FROM {table}")}


def _names(folder) -> List[str]:
    try:
        with os.scandir(folder) as it:
            return [e.name for e in it if e.is_file()]
    except (OSError, TypeError):
        return []


def _stem(name: str) -> str:
    return os.path.splitext(name)[0]


def _next_pick(config, db: ReadOnlyDB) -> str:
    """The work the watcher would take next, by its own selection code."""
    from mousereach.watcher.orchestrator import DLCOrchestrator, ProcessingOrchestrator
    cls = ProcessingOrchestrator if config.mode == "processing_server" else DLCOrchestrator
    orch = object.__new__(cls)          # no __init__: nothing starts, nothing recovers
    orch.config = config
    orch.db = db
    orch._archive_backoff_active = lambda video_id: False
    try:
        pick = (orch._select_work_item(admit_deferred=False)
                or orch._select_work_item(admit_deferred=True))
    except Exception as e:  # a bucket needing more than a read: say so, never write
        return "could not be worked out without starting the watcher (%s)" % e
    if pick is None:
        return "nothing -- every queue this machine serves is empty"
    return "%s %s" % (pick.get("type"), pick.get("id"))


def preview(config, db_path) -> List[str]:
    """Lines describing what the watcher would do now. Reads only."""
    from mousereach.config import Paths
    db = ReadOnlyDB(db_path)
    lines = ["DRY RUN -- nothing below was done; the database was opened read-only.", ""]

    videos = db.counts("videos")
    lines.append("Videos in this machine's database: %d" % sum(videos.values()))
    for state, n in sorted(videos.items()):
        lines.append("  %-14s %d" % (state, n))

    known_videos = db.ids("videos", "video_id")
    if config.mode == "processing_server":
        posed = [n for n in _names(Paths.DLC_STAGING) if n.lower().endswith(".h5")]
        new = {_stem(n).split("DLC")[0] for n in posed} - known_videos
        lines += ["", "Posed folder: %s" % (Paths.DLC_STAGING or "(not configured)"),
                  "  posed videos it would newly take in: %d" % len(new)]
    else:
        collages = db.counts("collages")
        known_collages = db.ids("collages", "filename")
        intake = [n for n in _names(Paths.MULTI_ANIMAL_SOURCE) if n.lower().endswith(VIDEO_EXT)]
        shared = [n for n in _names(Paths.SINGLE_ANIMAL_OUTPUT) if n.lower().endswith(VIDEO_EXT)]
        queue = [n for n in _names(Paths.DLC_QUEUE) if n.lower().endswith(VIDEO_EXT)]
        lines += ["", "Recordings in this machine's database: %d" % sum(collages.values())]
        for state, n in sorted(collages.items()):
            lines.append("  %-14s %d" % (state, n))
        lines += ["", "Intake folder: %s" % (Paths.MULTI_ANIMAL_SOURCE or "(not configured)"),
                  "  recordings it would newly register: %d"
                  % len([n for n in intake if n not in known_collages]),
                  "Shared cut videos waiting to be posed: %d" % len(shared),
                  "This machine's own pose queue: %d" % len(queue)]

    lines += ["", "Work priority:"]
    try:
        from mousereach.watcher.work_priority import load_lab_policy
        policy = load_lab_policy(getattr(config, "work_priority", None))
        lines += ["  " + l for l in policy.describe()]
        lines += ["  [!] " + c for c in policy.complaints]
    except Exception as e:
        lines.append("  could not be read (%s)" % e)

    lines += ["", "It would take next: " + _next_pick(config, db)]
    return lines
