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
     that is what the triage review tool walks. A worklist item may carry
     "segment_details" ({segment number: words}); each flagged segment's
     reason is then "<reason>: <its words>", and the review tool shows it on
     screen beside that pellet (WHY: all a reviewer sees is the tool, so a
     pellet sent back must say why -- a reviewer met finished videos coming
     back with no reason given, 2026-10-23),
  3. moves the video's bundle into the queue with a routing manifest
     (review_gate.route_to_queue), updating the watcher database.

A SEGMENT A PERSON HAS ANSWERED IS NEVER FLAGGED
------------------------------------------------
Before flagging, the video's saved review is read and every requested segment
a person (or ground truth) has already answered -- matched by the FRAMES the
answer is about, the rule the release check uses -- is dropped from the list
and reported as "answered". If nothing is left, the video is not routed.
WHY: the release check counts such a segment answered, so a routed video went
straight back out with nobody looking, and the next night's scan sent it again
(2026-10). A caller cannot always know what a person answered; this command
reads the review itself, so the rule holds whoever asks.

The one exception is asked for by name: ``--recheck`` ("are you sure?").
Each requested pellet a person answered is then REOPENED rather than dropped
(reopen_answered): the answer is kept in the review file under
``answers.recheck.previous``, the pellet is marked not yet answered, its flag
reason carries the question, and the video stays in triage until a person
answers that pellet again. The review file is copied to
Analyzed/Archive/manual_corrections/ first. Ground truth is never reopened.

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
       deferred because the watcher is working on it, or had
       every requested segment already answered by a person ("done")
    1  some video could not be routed, OR routing was refused because the
       watcher database could not be resolved, does not exist, or cannot be
       opened (then no video was touched)
ASCII-only output. With --json, stdout carries only the JSON results; the
"[watcher db]" line and any refusal message go to stderr.
"""
from __future__ import annotations

import argparse
import time
import json
import logging
import shutil
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


def flag_segments(outcomes_path: Path, segment_nums: Iterable[int], reason: str,
                  details: Optional[dict] = None) -> List[int]:
    """Set flagged_for_review on the given segments; returns those found.
    ``details`` ({segment number: words}) adds each segment's own words to
    its reason."""
    wanted = {int(s) for s in segment_nums}
    details = {int(k): str(v) for k, v in (details or {}).items()}
    data = json.loads(outcomes_path.read_text(encoding="utf-8"))
    flagged = []
    for s in data.get("segments", []):
        if s.get("segment_num") in wanted:
            s["flagged_for_review"] = True
            own = details.get(int(s["segment_num"]))
            s["flag_reason"] = f"{reason}: {own}" if own else reason
            s["triage_cleared"] = None
            flagged.append(int(s["segment_num"]))
    if flagged:
        outcomes_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return sorted(flagged)


def answered_segments_of(video_id: str, folder: Path) -> set:
    """Segments a person or ground truth has answered, matched by frames
    (triage_status.resolved_segments + gt_resolved_segments -- the release
    check's own rule). Raises when the review exists but cannot be read: a
    guess here could route a person's answered segment back to them."""
    from mousereach.review.causal_review_io import resolve_review_path
    from mousereach.review.triage_status import gt_resolved_segments, resolved_segments
    review = None
    rp = resolve_review_path(video_id, primary_dir=folder)
    if rp is not None:
        review = json.loads(Path(rp).read_text(encoding="utf-8"))
    seg_doc = None
    sp = Path(folder) / f"{video_id}_segments.json"
    if sp.is_file():
        seg_doc = json.loads(sp.read_text(encoding="utf-8"))
    answered = set(resolved_segments(review, seg_doc)) if review else set()
    gt_full, gt_segs = gt_resolved_segments(video_id)
    if gt_full:
        return {"all"}
    return answered | set(gt_segs)


def reopen_answered(video_id: str, folder: Path, segment_nums: Iterable[int],
                    questions: Optional[dict] = None) -> List[int]:
    """Ask a person again about pellets a person already answered ("are you
    sure?"). Returns the segments reopened.

    WHY: a person's answer is the truth for a pellet -- until there is reason to
    think a review slipped. Logan, 2026-10-26, on answers of "retrieved" where
    the frames show the pellet lying in the scoring area afterwards: "toss these
    cases back at the review queue and be like - are you sure". So the answer is
    not changed and not dropped: for each requested segment (matched by frames,
    as the results step matches answers) the record keeps its answer under
    ``answers.recheck.previous`` with the question and when it was asked, and is
    marked ``answers.reviewed = False`` -- the established meaning of "not
    answered yet" -- so the video stays in triage until a person answers that
    pellet again. Saving it in the review tool writes a fresh record, which ends
    the re-check. The review file is copied first to
    <Analyzed>/Archive/manual_corrections/<name>.before_recheck_<time>."""
    from datetime import datetime
    from mousereach.config import Paths
    from mousereach.review.causal_review_io import (index_review_by_segment,
                                                    resolve_review_path,
                                                    segmentation_is_human,
                                                    segments_from_segmentation)
    wanted = {int(s) for s in segment_nums}
    questions = {int(k): str(v) for k, v in (questions or {}).items()}
    rp = resolve_review_path(video_id, primary_dir=folder)
    if rp is None or not Path(rp).is_file():
        return []
    rp = Path(rp)
    review = json.loads(rp.read_text(encoding="utf-8"))
    sp = Path(folder) / f"{video_id}_segments.json"
    seg_doc = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else None
    current = segments_from_segmentation(seg_doc) if seg_doc else None
    mapping, _ = index_review_by_segment(
        review, current, human_segmentation=bool(seg_doc) and segmentation_is_human(seg_doc))
    now = datetime.now().isoformat(timespec="seconds")
    reopened = []
    for sn in sorted(wanted):
        rec = mapping.get(sn)
        answers = (rec or {}).get("answers") or {}
        if rec is None or answers.get("reviewed") is False:
            continue
        # Who gave the answer being questioned: the record's own values, or
        # the file's for a review saved before records carried them. The
        # reopened record is nobody's answer, so it carries no reviewer.
        rec["answers"] = {"reviewed": False, "recheck": {
            "asked_at": now, "question": questions.get(sn),
            "previous": {"human": rec.get("human"), "answers": answers,
                         "reviewer": rec.pop("reviewer", None) or review.get("reviewer"),
                         "reviewed_at": (rec.pop("reviewed_at", None)
                                         or review.get("reviewed_at"))}}}
        reopened.append(sn)
    if reopened:
        backup_dir = Path(Paths.ANALYZED_OUTPUT) / "Archive" / "manual_corrections"
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rp, backup_dir / ("%s.before_recheck_%s" % (
            rp.name, now.replace("-", "").replace(":", "").replace("T", "_"))))
        rp.write_text(json.dumps(review, indent=2), encoding="utf-8")
    return reopened


def route_video(video_id: str, queue: str, reason: str,
                segment_nums: Optional[Iterable[int]] = None, db=None,
                segment_details: Optional[dict] = None,
                recheck: bool = False) -> dict:
    """Flag (optional) and route one video. Returns a result dict; never raises."""
    from mousereach.config import Paths
    from mousereach.watcher.review_gate import route_to_queue

    res = {"video_id": video_id, "queue": queue, "flagged": [], "answered": [],
           "routed": False, "deferred": False, "error": None, "reopened": []}
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
            # Never ask a person again what they (or ground truth) already
            # answered -- see "A SEGMENT A PERSON HAS ANSWERED" above.
            wanted = sorted({int(s) for s in segment_nums})
            done = answered_segments_of(video_id, outcomes.parent)
            res["answered"] = wanted if "all" in done else [s for s in wanted if s in done]
            if recheck and res["answered"]:
                # Asked again on purpose ("are you sure?"): reopen the person's
                # answers rather than skip them -- see reopen_answered.
                res["reopened"] = reopen_answered(video_id, outcomes.parent,
                                                  res["answered"], segment_details)
                res["answered"] = [s for s in res["answered"] if s not in res["reopened"]]
            segment_nums = [s for s in wanted if s not in res["answered"]]
            if not segment_nums:
                return res          # nothing left to ask; not routed, not a failure
            res["flagged"] = flag_segments(outcomes, segment_nums, reason,
                                           details=segment_details)
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
                    help='JSON list of {"video_id", "segment_nums", optional '
                         '"segment_details": {segment: words}} to route in one go')
    ap.add_argument("--queue", choices=QUEUES, required=True)
    ap.add_argument("--reason", required=True, help="Recorded in the routing manifest")
    ap.add_argument("--flag-segments", default="",
                    help="Comma-separated segment numbers to flag (single-video mode)")
    ap.add_argument("--recheck", action="store_true",
                    help="Ask again about pellets a person already answered (their answer "
                         "is kept and shown; the pellet stays open until answered again). "
                         "Without it, answered pellets are never sent back.")
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
            items.append((it["video_id"], it.get("segment_nums") or [],
                          it.get("segment_details") or {}))
    else:
        segs = [int(x) for x in args.flag_segments.split(",") if x.strip()]
        items.append((args.video_id, segs, {}))

    # Each result is printed (and flushed) the moment its video is done, with
    # how long it took. WHY: the nightly bench check runs this under a time
    # limit, and printing only at the end meant a run stopped part-way left no
    # record of what it had routed -- nor of where the time went (2026-10-24:
    # stopped after 3 hours with 34 videos routed and not one line written).
    results = []
    for v, segs, det in items:
        started = time.time()
        r = route_video(v, args.queue, args.reason, segs, db=db, segment_details=det,
                        recheck=args.recheck)
        r["seconds"] = round(time.time() - started, 1)
        results.append(r)
        if not args.json:
            tag = ("OK  " if r["routed"] else "wait" if r["deferred"]
                   else "done" if (r["answered"] and not r["error"])
                   else ("skip" if r["error"] and "not found" in r["error"] else "FAIL"))
            print("%s %s -> %s  flagged=%s  %s%s%s  (%.0f s)" % (
                tag, r["video_id"], r["queue"], r["flagged"] or "-",
                ("asked again: %s  " % r["reopened"]) if r.get("reopened") else "",
                ("already answered by a person: %s  " % r["answered"]) if r["answered"] else "",
                r["error"] or "", r["seconds"]), flush=True)
    if args.json:
        print(json.dumps(results, indent=1))
    bad = [r for r in results if not r["routed"] and not r["deferred"]
           and not (r["answered"] and not r["error"])
           and not (r["error"] and "not found" in r["error"])]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
