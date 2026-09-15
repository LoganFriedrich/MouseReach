"""mousereach-route-to-queue -- put an archived video into a review queue.

WHY THIS IS A PUBLIC COMMAND
----------------------------
MouseReach decides on its own when a video needs a person (segmentation
failed, an element it could not commit to). But OTHER systems can have
reasons too -- for example a database tool that compares the pipeline's
pellet outcomes with hand-scored bench sheets and finds a disagreement. Such
a tool must not reach into MouseReach's internals or its files; it asks
through this command. MouseReach stays independent (it knows nothing about
who asked or why beyond the reason text it records), and the integrator
gets exactly the same routing the pipeline uses itself.

What it does, in order:
  1. finds the video's results in the configured Analyzed tree,
  2. optionally flags specific segments (flagged_for_review=True with the
     given reason, triage_cleared cleared) in {video}_pellet_outcomes.json --
     that is what the triage review tool walks,
  3. moves the video's bundle into the queue with a routing manifest
     (review_gate.route_to_queue), updating the watcher database.

THE WATCHER DATABASE -- the same file the daemon writes, or nothing is routed
-----------------------------------------------------------------------------
The database is resolved exactly as the daemon resolves it
(db_location.resolve_watcher_db_path: the configured watcher.db_path, else
<processing_root>/watcher.db) and printed as "[watcher db] <path>", like every
other watcher command. This command used to open a bare WatcherDB(), which
ignores the db_path override: every state write went to an unused decoy
database while the bundles moved on disk, and the live database never learned
of any of it.

Routing is REFUSED (exit 1, nothing flagged, nothing moved) when that database
file does not exist, or exists but cannot be opened. WHY refuse rather than
"route on disk only": a video moved into a queue while the database still says
'archived' leaves disk and database disagreeing for that video, and a worklist
does it for every video at once. A missing file is not created either --
WatcherDB's constructor would build an empty database, which is just a new
decoy. Fix the configuration (mousereach-setup) or start the watcher once so
its database exists, then route again.

A VIDEO THE WATCHER IS WORKING ON IS DEFERRED
---------------------------------------------
A video whose watcher state is dlc_queued, dlc_running, dlc_complete,
processing, processed or archiving is not flagged and not moved; it is
reported as "wait" (deferred: true with --json). WHY: this command writes the
daemon's live database. Setting 'triage' underneath a running pipeline makes
the pipeline's own 'processing' -> 'processed' write an illegal transition, and
the daemon marks the video failed with fresh outputs in Processing and old ones
in the queue; a processed or archiving video's Analyzed copy is older than the
outputs the daemon is about to archive, so a reviewer would judge stale data.
A deferral is not a failure: offer the video again on a later run.

Usage:
    mousereach-route-to-queue VIDEO_ID --queue triage --reason "bench disagreement" --flag-segments 3,7
    mousereach-route-to-queue VIDEO_ID --queue deep_review --reason "segmentation wrong"
    mousereach-route-to-queue --worklist worklist.json --queue triage --reason "..."
        worklist.json: [{"video_id": "...", "segment_nums": [3, 7]}, ...]

Exit codes:
    0  every requested video was routed, was already not in Analyzed, or was
       deferred because the watcher is working on it
    1  some video could not be routed, OR routing was refused because the
       watcher database could not be resolved, does not exist, or cannot be
       opened (then no video was touched)
ASCII-only output. With --json, stdout carries only the JSON results; the
"[watcher db]" line and any refusal message go to stderr.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

QUEUES = ("triage", "deep_review")

# Watcher states in which the daemon itself is about to move this video's files
# or write its state. WHY: see "A VIDEO THE WATCHER IS WORKING ON IS DEFERRED"
# in the module docstring -- routing such a video races the daemon over the
# live database. The DLC states are included because the daemon will re-run
# the video through the pipeline and gate, producing newer outputs than the
# Analyzed copy this command would hand a reviewer.
IN_FLIGHT_STATES = ("dlc_queued", "dlc_running", "dlc_complete",
                    "processing", "processed", "archiving")


def _find_outcomes(analyzed: Path, video_id: str) -> Optional[Path]:
    """The video's pellet_outcomes file under the Analyzed tree, or None."""
    from mousereach.pipeline.analyzed_tree import first_file, is_superseded_dir
    name = f"{video_id}_pellet_outcomes.json"
    # Analyzed/Archive/<folder>/<name> is exactly two levels deep, so the fast
    # glob matched superseded outputs too -- and route_video then wrote review
    # flags into the ARCHIVED file and moved an archive folder into a review
    # queue. Skip any hit that sits under a superseded folder.
    for hit in analyzed.glob(f"*/*/{name}"):
        if any(is_superseded_dir(p) for p in hit.relative_to(analyzed).parts[:-1]):
            continue
        return hit
    # Same rule for the full-depth fallback: never enter Analyzed/Archive/.
    return first_file(analyzed, name)


def flag_segments(outcomes_path: Path, segment_nums: Iterable[int], reason: str) -> List[int]:
    """Set flagged_for_review on the given segments; returns those found."""
    wanted = {int(s) for s in segment_nums}
    data = json.loads(outcomes_path.read_text(encoding="utf-8"))
    flagged = []
    for s in data.get("segments", []):
        if s.get("segment_num") in wanted:
            s["flagged_for_review"] = True
            s["flag_reason"] = reason
            s["triage_cleared"] = None
            flagged.append(int(s["segment_num"]))
    if flagged:
        outcomes_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return sorted(flagged)


def route_video(video_id: str, queue: str, reason: str,
                segment_nums: Optional[Iterable[int]] = None, db=None) -> dict:
    """Flag (optional) and route one video. Returns a result dict; never raises."""
    from mousereach.config import Paths
    from mousereach.watcher.review_gate import route_to_queue

    res = {"video_id": video_id, "queue": queue, "flagged": [], "routed": False,
           "deferred": False, "error": None}
    try:
        analyzed = Paths.ANALYZED_OUTPUT
        if not analyzed or not Path(analyzed).exists():
            res["error"] = "Analyzed tree not configured (run mousereach-setup)"
            return res
        outcomes = _find_outcomes(Path(analyzed), video_id)
        if outcomes is None:
            res["error"] = "not found in Analyzed (already routed, or never archived)"
            return res
        if db is not None:
            # Checked BEFORE any flag is written or file moved (IN_FLIGHT_STATES
            # says why). A database that cannot be read raises into the except
            # below and is reported as a failure, never routed blind.
            state = (db.get_video(video_id) or {}).get("state")
            if state in IN_FLIGHT_STATES:
                res["deferred"] = True
                res["error"] = ("in flight on the watcher (state '%s'); not "
                                "routed -- offer it again on a later run" % state)
                return res
        if segment_nums:
            res["flagged"] = flag_segments(outcomes, segment_nums, reason)
        dest_root = Paths.TRIAGE_REVIEW if queue == "triage" else Paths.DEEP_REVIEW
        route_to_queue(video_id, outcomes.parent, dest_root, reason=reason,
                       db=db, db_state=queue)
        res["routed"] = True
    except Exception as e:  # never break the caller; report instead
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def _open_watcher_db(out):
    """Open the daemon's own watcher database, or print why not and return None.

    Never creates the file and never falls back to routing without a database:
    see "THE WATCHER DATABASE" in the module docstring for why.
    ``out`` is where the messages go (stderr in --json mode, so stdout stays
    parseable).
    """
    from mousereach.watcher.db_location import resolve_watcher_db_path
    try:
        path = resolve_watcher_db_path()
    except Exception as e:
        print("[FAIL] cannot resolve the watcher database (%s: %s). Nothing was "
              "routed." % (type(e).__name__, e), file=out)
        return None
    print("[watcher db] %s" % path, file=out)
    if not Path(path).is_file():
        # WHY check before constructing: WatcherDB() creates a missing file as
        # an empty database -- a fresh decoy the daemon never reads.
        print("[FAIL] watcher database not found: %s\n"
              "  Routing is refused: moving videos into a review queue that the "
              "watcher database does not record leaves disk and database "
              "disagreeing for every video. Check watcher.db_path "
              "(mousereach-setup) or start the watcher once so it creates its "
              "database. Nothing was routed." % path, file=out)
        return None
    from mousereach.watcher.db import WatcherDB
    try:
        return WatcherDB(path)
    except Exception as e:
        # Same WHY: without the database, routing on disk only is the
        # disagreement this refusal exists to prevent.
        print("[FAIL] cannot open the watcher database %s (%s: %s). Routing "
              "is refused. Nothing was routed." % (path, type(e).__name__, e),
              file=out)
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_id", nargs="?", help="e.g. 20250624_CNT0115_P2")
    ap.add_argument("--worklist", type=Path,
                    help='JSON list of {"video_id", "segment_nums"} to route in one go')
    ap.add_argument("--queue", choices=QUEUES, required=True)
    ap.add_argument("--reason", required=True, help="Recorded in the routing manifest")
    ap.add_argument("--flag-segments", default="",
                    help="Comma-separated segment numbers to flag (single-video mode)")
    ap.add_argument("--json", action="store_true", help="Machine-readable results")
    args = ap.parse_args(argv)

    if bool(args.video_id) == bool(args.worklist):
        ap.error("give exactly one of VIDEO_ID or --worklist")

    db = _open_watcher_db(sys.stderr if args.json else sys.stdout)
    if db is None:
        return 1

    items = []
    if args.worklist:
        for it in json.loads(Path(args.worklist).read_text(encoding="utf-8")):
            items.append((it["video_id"], it.get("segment_nums") or []))
    else:
        segs = [int(x) for x in args.flag_segments.split(",") if x.strip()]
        items.append((args.video_id, segs))

    results = [route_video(v, args.queue, args.reason, segs, db=db) for v, segs in items]
    if args.json:
        print(json.dumps(results, indent=1))
    else:
        for r in results:
            tag = ("OK  " if r["routed"] else "wait" if r["deferred"]
                   else ("skip" if r["error"] and "not found" in r["error"] else "FAIL"))
            print("%s %s -> %s  flagged=%s  %s" % (tag, r["video_id"], r["queue"],
                                                  r["flagged"] or "-", r["error"] or ""))
    bad = [r for r in results if not r["routed"] and not r["deferred"]
           and not (r["error"] and "not found" in r["error"])]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
