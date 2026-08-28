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
     (review_gate.route_to_queue), updating the local watcher database.

Usage:
    mousereach-route-to-queue VIDEO_ID --queue triage --reason "bench disagreement" --flag-segments 3,7
    mousereach-route-to-queue VIDEO_ID --queue deep_review --reason "segmentation wrong"
    mousereach-route-to-queue --worklist worklist.json --queue triage --reason "..."
        worklist.json: [{"video_id": "...", "segment_nums": [3, 7]}, ...]

Exit code 0 if every requested video was routed (or was already not in
Analyzed), 1 otherwise. ASCII-only output.
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


def _find_outcomes(analyzed: Path, video_id: str) -> Optional[Path]:
    """The video's pellet_outcomes file under the Analyzed tree, or None."""
    name = f"{video_id}_pellet_outcomes.json"
    for hit in analyzed.glob(f"*/*/{name}"):
        return hit
    for hit in analyzed.rglob(name):
        return hit
    return None


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

    res = {"video_id": video_id, "queue": queue, "flagged": [], "routed": False, "error": None}
    try:
        analyzed = Paths.ANALYZED_OUTPUT
        if not analyzed or not Path(analyzed).exists():
            res["error"] = "Analyzed tree not configured (run mousereach-setup)"
            return res
        outcomes = _find_outcomes(Path(analyzed), video_id)
        if outcomes is None:
            res["error"] = "not found in Analyzed (already routed, or never archived)"
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

    from mousereach.watcher.db import WatcherDB
    db = None
    try:
        db = WatcherDB()
    except Exception as e:
        logger.warning("watcher database unavailable (%s); routing on disk only", e)

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
            tag = "OK  " if r["routed"] else ("skip" if r["error"] and "not found" in r["error"] else "FAIL")
            print("%s %s -> %s  flagged=%s  %s" % (tag, r["video_id"], r["queue"],
                                                  r["flagged"] or "-", r["error"] or ""))
    bad = [r for r in results if not r["routed"] and not (r["error"] and "not found" in r["error"])]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
