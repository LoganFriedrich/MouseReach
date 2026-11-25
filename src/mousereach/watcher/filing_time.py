"""A video's filing time means "filed NOW".

THE RULE. ``videos.archived_at`` -- the filing time -- is set when a video
is filed and holds only while the row is in a filed state (db.FILED_STATES:
'archived', 'outdated', 'crystallized'). Every change that takes the row out
of those states -- a route into a review queue, a re-run, a re-pose, a
retirement to 'unresolvable', a failure, a re-cut, a person's command,
anything -- moves the time into ``previously_filed_at`` and clears
``archived_at`` in the same write (WatcherDB._filing_time_defaults and
WatcherDB._filing_time_sql, applied by update_state, force_state,
mark_failed, mark_unresolvable and reset_failed, which every state change
passes through). Filing the video again sets ``archived_at`` again;
``previously_filed_at`` keeps the time of the filing before it. Moving
between filed states (marked 'outdated' for a re-run, un-marked, locked or
unlocked for a publication) is not a new filing and keeps the time.

WHY: archived_at must not be kept for ever once set, or rows keep a filing time
with nothing filed: videos
routed back out of Analyzed into triage, some of them
escalated to deep review since, re-posed videos whose re-run was held in
triage, and retired rows whose files had gone to manual corrections. Every
count or report that read the filing time took them for filed videos.

WHO READS WHICH:
  * "is it filed now?" -- ``archived_at`` (or the state);
  * "has it ever been filed?" -- ``db.last_filed_at(row)``: ``archived_at``,
    else ``previously_filed_at``. The watcher's own decisions that depend on
    a video having been filed (a kinematics-only re-run of a filed video,
    putting a failed re-run back to 'archived', never starting a filed video
    over from a new cut, refusing to finalize over filed results) read this,
    which is exactly what ``archived_at`` alone meant before the rule. So none
    of those decisions changed.

THIS MODULE applies the rule once to rows written before it existed:

    python -m mousereach.watcher.filing_time            # list them, change nothing
    python -m mousereach.watcher.filing_time --apply    # move their filing times

It changes only the watcher database: on each row, archived_at ->
previously_filed_at, and one line in that video's processing log. No file is
moved or read, no review bundle is touched, and the row's updated_at is left
as it was (its state did not change). It takes the database's write lock
the way every state change does (one short transaction), so it is safe with
the watcher running; stopping the watcher first is still the tidy way.
Running it a second time finds nothing to do.

ASCII-only console output (Windows consoles cannot print Unicode).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def run(db, apply: bool) -> List[dict]:
    """List (and with ``apply``, move) the filing times of rows that are not
    filed now. Returns the rows found, as they were before any change."""
    return db.move_stale_filing_times(apply=apply)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m mousereach.watcher.filing_time",
        description="Move the filing time of every watcher row that is not filed now "
                    "into previously_filed_at (dry run unless --apply).")
    p.add_argument("--apply", action="store_true",
                   help="change the rows (without it, only list them)")
    p.add_argument("--db", type=Path, default=None,
                   help="watcher database to use (default: this machine's own)")
    a = p.parse_args(argv)

    if a.db is not None:
        if not a.db.is_file():
            print(f"[FAIL] no watcher database at {a.db}")
            return 1
        from mousereach.watcher.db import WatcherDB
        db = WatcherDB(a.db)
        where = str(a.db)
    else:
        from mousereach.watcher.db_location import open_watcher_db_if_present
        db, why = open_watcher_db_if_present()
        if db is None:
            print(f"[FAIL] {why}")
            return 1
        where = str(db.db_path)

    try:
        rows = run(db, a.apply)
    except Exception as e:
        print(f"[FAIL] {where}: {e}")
        return 1

    by_state = {}
    for r in rows:
        by_state[r["state"]] = by_state.get(r["state"], 0) + 1
    verb = "Moved" if a.apply else "Would move"
    print(f"Watcher database: {where}")
    print(f"{verb} the filing time of {len(rows)} row(s) that are not filed now"
          + (": " + ", ".join(f"{s} {n}" for s, n in sorted(by_state.items()))
             if rows else "."))
    for r in rows:
        print(f"  {r['video_id']}  state={r['state']}  filed_at={r['archived_at']}")
    if rows and not a.apply:
        print("Nothing was changed. Run again with --apply to move them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
