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
  2. moves the video's bundle into the queue with a routing manifest
     (review_gate.route_to_queue), updating the watcher database -- all or
     nothing (below),
  3. optionally flags specific segments (flagged_for_review=True with the
     given reason, triage_cleared cleared) in the BUNDLE's
     {video}_pellet_outcomes.json -- that is what the triage review tool
     walks. A worklist item may carry "segment_details" ({segment number:
     words}); each flagged segment's reason is then "<reason>: <its words>",
     and the review tool shows it on screen beside that pellet (WHY: all a
     reviewer sees is the tool, so a pellet sent back must say why -- a
     reviewer met finished videos coming back with no reason given).
Steps 2 and 3 are done under the bundle's claim (review_return._claim_return),
so the watcher's return scan -- another process -- cannot return the bundle
between the move and the flag.

THE FILED FOLDER IS NEVER CHANGED BY A ROUTE THAT DOES NOT HAPPEN
-----------------------------------------------------------------
The flags are written only after the move has succeeded, into the bundle's
copy. WHY: they used to be written into the FILED outcomes file first; a
route that then failed (a file held open, so every file went back), or was
refused because the video is held, left the filed results carrying review
flags although nothing had been routed. A failed or refused route now leaves
every filed file byte for byte as it was. If the move succeeds and the flag
cannot be written, the video is in the queue and is reported FAIL "question
not recorded" (exit code 1): route it again to ask.

A SEGMENT A PERSON HAS ANSWERED IS NEVER FLAGGED
------------------------------------------------
Before flagging, the video's saved review is read and every requested segment
a person (or ground truth) has already answered -- matched by the FRAMES the
answer is about, the rule the release check uses -- is dropped from the list
and reported as "answered". If nothing is left, the video is not routed.
WHY: the release check counts such a segment answered, so a routed video went
straight back out with nobody looking, and the next night's scan sent it again.
A caller cannot always know what a person answered; this command
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
<processing_root>/watcher.db on a GPU PC; a processing server whose settings
name no watcher.db_path, or settings that cannot be read, are refused) and
printed as "[watcher db] <path>", like every
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
A deferral is not a failure: offer the video again on a later run. A video
whose bundle another process is moving right now (it holds the bundle's
claim, review_return._claim_return -- the watcher's return scan, say) is
deferred the same way.

A VIDEO IN DEEP REVIEW IS NEVER SENT TO TRIAGE
---------------------------------------------
Deep review supersedes triage: a video with a bundle in the deep-review queue
is not flagged and not routed to triage; it is reported as "deep" (in_deep_review:
true with --json) and stays where it is. Not a failure for the exit code. WHY:
the filed results this command routes from are older than what deep review
holds, and a triage copy beside the deep-review bundle had a reviewer answer
the wrong copy. Routing into deep_review empties the video's triage bundle
once the deep-review bundle is complete, without losing a person's file
(review_gate._drop_triage_copy): a video or pose file the deep-review bundle
has none of is moved in; then what has a current copy elsewhere -- machine
outputs, files the deep-review bundle holds byte for byte, the triage review
when the durable review store already holds it byte for byte -- is
deleted; ground truth, hand-set cuts,
clear markers and a review the store does not hold byte for byte are never carried
into the deep-review bundle and never deleted: they stay where they are,
with a note, listed in the dashboard's "Needs your attention" panel.

A VIDEO ALREADY HELD IN A QUEUE IS NOT MOVED -- ITS QUESTION GOES IN ITS BUNDLE
--------------------------------------------------------------------------------
A video whose watcher state is 'triage' or 'deep_review', or that has a bundle
in either queue (published, or still being built in the queue's .incoming
folder), is not moved and its filed copy is not touched; it is reported as
"held" (held: true with --json). WHY: the held bundle is the video's live copy,
and it can hold a person's answers and cuts. Routing used to take the OLDER
filed results and move them into that bundle, replacing the held run's video,
pose, results -- and a review a person had saved there -- with the filed ones.

A question asked about a held video (--flag-segments, or a worklist's
segment_nums) is not dropped: it is written into the HELD bundle's
pellet_outcomes.json -- the file the triage review tool walks -- exactly as for
a routed video (a segment a person already answered is skipped, --recheck
reopens it). That is the path the hand-score check uses for videos already
waiting in a queue. Exit code:
  * held, and every requested segment flagged in the held bundle (or none was
    requested, or every one was already answered): 0;
  * held, and the question could NOT be recorded: 1, with "question not
    recorded" and why. That happens when the video is held in deep review (its
    results are made again after re-segmentation, so a flag would be thrown
    away), when only the watcher state says it is held and no published triage
    bundle is reachable from here, when the held triage bundle has no outcomes
    file or not the requested segments, when deep review was asked for a video
    held in triage (escalate it from the triage review tool instead), when
    the triage folder is an old copy kept for a person (listed under 'Folders
    left for a person' in the Review Queues tab), or when the bundle is being moved by the return scan right
    now. Ask again later --
    after the video is filed again, a normal route takes the question.

A CRYSTALLIZED VIDEO IS NEVER MOVED
-----------------------------------
A video locked for a publication (mousereach-crystallize; watcher state
'crystallized') is not flagged and not moved; it is reported as "lock"
(locked: true with --json) with a message naming the lock. Its filed results
must stay exactly as published. Unlock it first (mousereach-uncrystallize) if
it really is to be reviewed again. Not a failure for the exit code: the
command did what the lock asks.

Usage:
    mousereach-route-to-queue VIDEO_ID --queue triage --reason "bench disagreement" --flag-segments 3,7
    mousereach-route-to-queue VIDEO_ID --queue deep_review --reason "segmentation wrong"
    mousereach-route-to-queue --worklist worklist.json --queue triage --reason "..."
        worklist.json: [{"video_id": "...", "segment_nums": [3, 7]}, ...]

Exit codes:
    0  every requested video was routed (with its question, if any, written
       into its bundle), was already not in Analyzed, or was
       deferred because the watcher is working on it, or had
       every requested segment already answered by a person ("done"), or is
       crystallized ("lock"), or is in deep review and was asked for triage
       ("deep"), or is already held in a queue with its question (if any)
       written into the held bundle ("held")
    1  some video could not be routed, or a held or routed video's question
       could not be recorded (printed as FAIL "... question not recorded:
       <why>"), OR
       routing was refused because the watcher database could not be resolved,
       does not exist, or cannot be opened (then no video was touched)
ASCII-only output. With --json, stdout carries only the JSON results; the
"[watcher db]" line and any refusal message go to stderr.
"""
from __future__ import annotations

import argparse
import time
import json
import logging
import os
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
    from mousereach.archive.core import filed_folder
    from mousereach.pipeline.analyzed_tree import first_file
    name = f"{video_id}_pellet_outcomes.json"
    # The folder the filing rule names first. It is never a superseded folder,
    # which matters: a superseded copy keeps this name, and a hit there once
    # made route_video write review flags into the ARCHIVED file.
    try:
        hit = filed_folder(video_id, analyzed) / name
        if hit.is_file():
            return hit
    except Exception:
        pass
    # The whole tree otherwise, never entering Analyzed/Archive/.
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
        # One-step replacement, never in place: the outcomes file is a stage
        # output every later step reads, and write_text empties it before
        # writing -- a crash here used to leave it cut off.
        from mousereach.pipeline.fsutil import write_json_atomic
        write_json_atomic(outcomes_path, data, indent=2)
    return sorted(flagged)


def answered_segments_of(video_id: str, folder: Path) -> set:
    """Segments a person or ground truth has answered, matched by frames
    (triage_status.resolved_segments + gt_resolved_segments -- the release
    check's own rule). Raises when the review exists but cannot be read: a
    guess here could route a person's answered segment back to them."""
    from mousereach.review.causal_review_io import effective_review, review_dirs
    from mousereach.review.triage_status import gt_resolved_segments, resolved_segments
    seg_doc = None
    sp = Path(folder) / f"{video_id}_segments.json"
    if sp.is_file():
        seg_doc = json.loads(sp.read_text(encoding="utf-8"))
    # Every copy, merged -- the release check's own reading (one copy picked
    # by file time could call a pellet open that another copy answered).
    # strict: a copy that exists but cannot be read RAISES here instead of
    # being skipped -- it may be the only copy holding the answer, and a
    # skipped answer would be routed back to the person as a new question.
    review = effective_review(video_id, review_dirs(video_id, folder), seg_doc,
                              strict=True)
    answered = set(resolved_segments(review, seg_doc)) if review else set()
    gt_full, gt_segs = gt_resolved_segments(video_id, extra_dirs=(folder,))
    if gt_full:
        return {"all"}
    return answered | set(gt_segs)


def reopen_answered(video_id: str, folder: Path, segment_nums: Iterable[int],
                    questions: Optional[dict] = None, dry_run: bool = False) -> List[int]:
    """Ask a person again about pellets a person already answered ("are you
    sure?"). Returns the segments reopened. ``dry_run`` True writes nothing
    and returns the segments that would be reopened (route_video decides with
    it whether there is anything to ask before it moves the video).

    WHY: a person's answer is the truth for a pellet -- until there is reason to
    think a review slipped -- e.g. answers of "retrieved" where the frames show
    the pellet lying in the scoring area afterwards go back to the review queue
    as "are you sure?". So the answer is
    not changed and not dropped: for each requested segment (matched by frames,
    as the results step matches answers) the record keeps its answer under
    ``answers.recheck.previous`` with the question and when it was asked, and is
    marked ``answers.reviewed = False`` -- the established meaning of "not
    answered yet" -- so the video stays in triage until a person answers that
    pellet again. Saving it in the review tool writes a fresh record, which ends
    the re-check.

    WHICH COPY: the decision is made on EVERY copy of the review merged
    (causal_review_io.effective_review -- the reading the window, the gate and
    the release check use), and the reopened record is written into BOTH the
    copy in ``folder`` (the one that travels with the video; created if
    missing) and the durable copy of record, if it exists. WHY: this used to
    pick ONE copy by file time and write only there, so a pellet answered in
    another copy was not reopened, or the recheck sat in a copy the window
    then out-voted. Each copy written is first backed up to
    <Analyzed>/Archive/manual_corrections/<name>.before_recheck_<time>.<role>,
    <role> being ``travelling`` or ``durable``: two separate files, because
    both copies have the same name and one backup must not replace the other. It is
    written directly, never through save_causal_review, whose never-drop-an-
    answer rule would put the reopened answer straight back.

    Raises, writing nothing, when a copy of the review exists but cannot be
    read (effective_review strict=True)."""
    from datetime import datetime
    from mousereach.config import Paths
    from mousereach.review.causal_review_io import (durable_review_path, effective_review,
                                                    index_review_by_segment, is_answered,
                                                    review_dirs, segmentation_is_human,
                                                    segments_from_segmentation)
    wanted = {int(s) for s in segment_nums}
    questions = {int(k): str(v) for k, v in (questions or {}).items()}
    sp = Path(folder) / f"{video_id}_segments.json"
    seg_doc = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else None
    current = segments_from_segmentation(seg_doc) if seg_doc else None
    human = bool(seg_doc) and segmentation_is_human(seg_doc)
    # strict, as in answered_segments_of: an unreadable copy raises, so a
    # pellet answered only there is never taken for open (or left un-reopened).
    review = effective_review(video_id, review_dirs(video_id, folder), seg_doc,
                              strict=True)
    if not review:
        return []
    mapping, _ = index_review_by_segment(review, current, human_segmentation=human)
    now = datetime.now().isoformat(timespec="seconds")
    reopened = {}
    for sn in sorted(wanted):
        rec = mapping.get(sn)
        answers = (rec or {}).get("answers") or {}
        # Only an answer can be asked again (is_answered, the one definition):
        # a placeholder or a pellet left undecided is already open.
        if not is_answered(rec):
            continue
        # Who gave the answer being questioned: the record's own values, or
        # the file's for a review saved before records carried them. The
        # reopened record is nobody's answer, so it carries no reviewer.
        new = {k: v for k, v in rec.items() if k not in ("reviewer", "reviewed_at")}
        new["segment_num"] = sn
        new["answers"] = {"reviewed": False, "recheck": {
            "asked_at": now, "question": questions.get(sn),
            "previous": {"human": rec.get("human"), "answers": answers,
                         "reviewer": rec.get("reviewer") or review.get("reviewer"),
                         "reviewed_at": (rec.get("reviewed_at")
                                         or review.get("reviewed_at"))}}}
        reopened[sn] = new
    if reopened and not dry_run:
        from mousereach.pipeline.fsutil import write_json_atomic
        backup_dir = Path(Paths.ANALYZED_OUTPUT) / "Archive" / "manual_corrections"
        stamp = now.replace("-", "").replace(":", "").replace("T", "_")
        travelling = Path(folder) / f"{video_id}_causal_review.json"
        # Each target carries its role, which goes into its backup's name:
        # both copies share the basename <video>_causal_review.json and the
        # same stamp, so without the role the durable copy's backup replaced
        # the travelling copy's and that copy's pre-recheck state was lost.
        targets = [(travelling, "travelling")]
        try:
            durable = durable_review_path(video_id)
        except Exception:
            durable = None
        if durable is not None and Path(durable).is_file() and \
                os.path.normcase(str(Path(durable).resolve())) != \
                os.path.normcase(str(travelling.resolve())):
            targets.append((Path(durable), "durable"))
        for target, role in targets:
            if target.is_file():
                backup_dir.mkdir(parents=True, exist_ok=True)
                # The backup first, then a one-step replacement: write_text
                # empties the file before writing, so a crash used to leave a
                # cut-off review, which every reader takes for "no answers".
                shutil.copy2(target, backup_dir / ("%s.before_recheck_%s.%s" % (
                    target.name, stamp, role)))
                doc = json.loads(target.read_text(encoding="utf-8"))
            else:
                doc = {"video_stem": video_id, "segments": []}
            segs = doc.setdefault("segments", [])
            tmap, _ = index_review_by_segment(doc, current, human_segmentation=human)
            for sn, new in reopened.items():
                old = tmap.get(sn)
                at = next((i for i, r in enumerate(segs) if r is old), None)
                if at is None:
                    segs.append(dict(new))
                else:
                    segs[at] = dict(new)
            write_json_atomic(target, doc, indent=2)
    return sorted(reopened)


def _triage_would_ask(video_id: str, outcomes: Path, wanted_open) -> bool:
    """Would a triage bundle of this video hold a question a person can
    answer? True when a requested segment (``wanted_open``: requested and not
    already answered, or reopened on purpose) exists in the outcomes and in the
    segmentation, or when the video already has a flagged segment on a real
    segment (triage_status.bundle_triage, the release check's rule) that
    nobody answered. Reads only."""
    from mousereach.review.triage_status import bundle_triage
    folder = Path(outcomes).parent
    try:
        od = json.loads(Path(outcomes).read_text(encoding="utf-8"))
    except Exception:
        od = {}
    present = {int(x["segment_num"]) for x in (od.get("segments") or [])
               if isinstance(x, dict) and x.get("segment_num") is not None}
    sp = folder / f"{video_id}_segments.json"
    try:
        bounds = (json.loads(sp.read_text(encoding="utf-8")).get("boundaries") or []
                  if sp.is_file() else [])
    except Exception:
        bounds = []
    n_segs = len(bounds) - 1 if bounds else None
    if any(s in present and (n_segs is None or s <= n_segs) for s in wanted_open):
        return True
    kept, _ghost = bundle_triage(folder, video_id)
    if not kept:
        return False
    done = answered_segments_of(video_id, folder)
    return "all" not in done and bool(set(kept) - done)


def route_video(video_id: str, queue: str, reason: str,
                segment_nums: Optional[Iterable[int]] = None, db=None,
                segment_details: Optional[dict] = None,
                recheck: bool = False) -> dict:
    """Flag (optional) and route one video. Returns a result dict; never raises."""
    from mousereach.config import Paths
    from mousereach.watcher.review_gate import AlreadyInQueueError, route_to_queue

    res = {"video_id": video_id, "queue": queue, "flagged": [], "answered": [],
           "routed": False, "deferred": False, "error": None, "reopened": [],
           "locked": False, "in_deep_review": False, "held": False}
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
            if state == "crystallized":
                # Locked for a publication: never moved, never flagged
                # (db.crystallized_refusal says why). Refused, not deferred --
                # offering it again changes nothing until a person unlocks it.
                from mousereach.watcher.db import crystallized_refusal
                res["locked"] = True
                res["error"] = crystallized_refusal(db, video_id, "routed")
                return res
            if state in IN_FLIGHT_STATES:
                res["deferred"] = True
                res["error"] = ("in flight on the watcher (state '%s'); not "
                                "routed -- offer it again on a later run" % state)
                return res
        if queue == "triage":
            from mousereach.watcher.review_routing import in_deep_review
            if in_deep_review(video_id):
                # Deep review supersedes triage -- see the module docstring.
                # Checked before any flag is written.
                res["in_deep_review"] = True
                res["error"] = ("in deep review, which supersedes triage; not "
                                "routed to triage")
                return res
        held = _held_in_a_queue(video_id, db)
        if held:
            # Already held -- see "A VIDEO ALREADY HELD IN A QUEUE" above.
            # Checked before any flag is written or file moved; the question,
            # if any, goes into the held bundle.
            _ask_in_held_bundle(res, video_id, queue, reason, segment_nums,
                                segment_details, recheck, held)
            return res
        wanted = []
        wanted_open = []
        if segment_nums:
            # Never ask a person again what they (or ground truth) already
            # answered -- see "A SEGMENT A PERSON HAS ANSWERED" above. Decided
            # here, reading only: nothing is written before the move.
            wanted = sorted({int(s) for s in segment_nums})
            try:
                done = answered_segments_of(video_id, outcomes.parent)
                answered = wanted if "all" in done else [s for s in wanted if s in done]
                # Asked again on purpose ("are you sure?"): an answer that can
                # be reopened still counts as something to ask (reopen_answered).
                reopenable = (reopen_answered(video_id, outcomes.parent, answered,
                                              segment_details, dry_run=True)
                              if recheck and answered else [])
            except Exception as e:
                # A copy of the review (or the cuts) exists but cannot be
                # read: guessing could ask a person again what they answered
                # (answered_segments_of says why). Nothing has moved yet.
                res["error"] = ("its review or cuts could not be read (%s: %s); not routed"
                                % (type(e).__name__, e))
                return res
            if not [s for s in wanted if s not in answered or s in reopenable]:
                res["answered"] = answered
                return res          # nothing left to ask; not routed, not a failure
            wanted_open = [s for s in wanted if s not in answered or s in reopenable]
        else:
            wanted_open = []
        try:
            would_ask = queue != "triage" or _triage_would_ask(video_id, outcomes,
                                                               wanted_open)
        except Exception as e:
            # Same refusal as above: it reads which flagged segments are
            # answered, from every copy of the review.
            res["error"] = ("its review or cuts could not be read (%s: %s); not routed"
                            % (type(e).__name__, e))
            return res
        if not would_ask:
            # A triage bundle with nothing a person can answer is never
            # released: the return scan finds no question and skips it on
            # every pass. So it is refused before anything moves.
            res["error"] = ("nothing to ask in triage: none of the requested segments "
                            "exists and the video has no open flagged segment; not routed")
            return res
        dest_root = Paths.TRIAGE_REVIEW if queue == "triage" else Paths.DEEP_REVIEW
        from mousereach.watcher.review_return import (_claim_return_nested,
                                                      _release_return_claim)
        claimed, release = _claim_return_nested(video_id)
        if not claimed:
            res["deferred"] = True
            res["error"] = ("its bundle is being moved by another process right now; "
                            "not routed -- offer it again on a later run")
            return res
        try:
            try:
                # All or nothing: the source is the FILED folder, and a filed
                # generation must never be left half moved (a file still there
                # would go on counting as a filed result).
                bundle = route_to_queue(video_id, outcomes.parent, dest_root,
                                        reason=reason, db=db, db_state=queue,
                                        all_or_nothing=True)
            except AlreadyInQueueError as e:
                # A bundle appeared between the check above and the move (or
                # the check could not see it): nothing moved and nothing was
                # written. The question goes into the held bundle.
                _ask_in_held_bundle(res, video_id, queue, reason, wanted,
                                    segment_details, recheck, str(e))
                return res
            res["routed"] = True
            if wanted:
                # The flags go into the bundle's copy now that it is in the
                # queue (THE FILED FOLDER IS NEVER CHANGED ... above).
                _record_question(res, video_id, Path(bundle), reason, wanted,
                                 segment_details, recheck, "routed")
        finally:
            if release:
                _release_return_claim(video_id)
    except Exception as e:  # never break the caller; report instead
        res["error"] = f"{type(e).__name__}: {e}"
    return res


HELD_STATES = ("triage", "deep_review")


def _ask_in_held_bundle(res: dict, video_id: str, queue: str, reason: str,
                        segment_nums, segment_details, recheck: bool,
                        held_why: str) -> None:
    """Record a question about a HELD video inside its held triage bundle (see
    "A VIDEO ALREADY HELD IN A QUEUE" in the module docstring). Fills ``res``:
    held True always; flagged / answered / reopened as for a routed video;
    ``unrecorded`` lists the segments whose question could not be recorded,
    with the reason in ``error`` (a failure for the exit code). Never raises
    (route_video's except reports anything unexpected).

    Done under the bundle's claim, which belongs to one process
    (review_return._claim_return): the outcomes file is never rewritten while
    the watcher's return scan -- another process, on this machine or another --
    is moving the bundle (that would leave a flag in a file already gone from
    the queue)."""
    from mousereach.config import Paths
    res["held"] = True
    res["error"] = held_why
    wanted = sorted({int(s) for s in (segment_nums or [])})
    if not wanted:
        return                              # nothing asked; already held
    bundle = (Path(Paths.TRIAGE_REVIEW) / video_id
              if getattr(Paths, "TRIAGE_REVIEW", None) else None)
    outcomes = bundle / f"{video_id}_pellet_outcomes.json" if bundle else None

    def _not_recorded(why: str) -> None:
        res["unrecorded"] = wanted
        res["error"] = "%s; question not recorded: %s" % (held_why, why)

    from mousereach.watcher.review_routing import in_deep_review
    if in_deep_review(video_id):
        return _not_recorded("it is in deep review, whose results are made again "
                             "after re-segmentation; ask again once it is filed")
    if queue != "triage":
        return _not_recorded(
            "deep review was asked for a video held in triage -- escalate it from "
            "the triage review tool, or route it again once it is filed")
    if bundle is None or not bundle.is_dir():
        return _not_recorded("no published triage bundle of it is reachable from "
                             "here")
    from mousereach.watcher.review_routing import is_queue_bundle
    if not is_queue_bundle(bundle):
        # Not a bundle: an old folder of the video kept for a person
        # (review_routing.clear_leftover). A question written there would
        # never be asked.
        return _not_recorded("its triage folder is an old copy kept for a person "
                             "(listed under 'Folders left for a person' in the "
                             "Review Queues tab); ask again once that folder is "
                             "dealt with")
    if not outcomes.is_file():
        return _not_recorded("its triage bundle has no pellet outcomes file")
    from mousereach.watcher.review_return import (_claim_return_nested,
                                                  _release_return_claim)
    claimed, release = _claim_return_nested(video_id)
    if not claimed:
        return _not_recorded("the return scan is moving its bundle right now")
    try:
        if not bundle.is_dir() or not outcomes.is_file():
            return _not_recorded("its triage bundle left the queue meanwhile")
        _record_question(res, video_id, bundle, reason, wanted, segment_details,
                         recheck, held_why)
    finally:
        if release:
            _release_return_claim(video_id)


def _record_question(res: dict, video_id: str, bundle: Path, reason: str,
                     wanted: List[int], segment_details, recheck: bool,
                     why: str) -> None:
    """Write the question about ``wanted`` into ``bundle``'s outcomes file:
    a segment a person (or ground truth) answered is skipped, or reopened with
    ``recheck``; every other one is flagged. Fills ``res`` (answered,
    reopened, flagged; ``unrecorded`` and ``error`` "<why>; question not
    recorded ..." for what could not be written). The caller holds the
    bundle's claim. Never raises."""
    outcomes = Path(bundle) / f"{video_id}_pellet_outcomes.json"

    def _not_recorded(text: str) -> None:
        res["unrecorded"] = wanted
        res["error"] = "%s; question not recorded: %s" % (why, text)

    if not outcomes.is_file():
        return _not_recorded("its bundle has no pellet outcomes file")
    try:
        done = answered_segments_of(video_id, bundle)
    except Exception as e:
        # A review or cuts file that cannot be read: guessing could ask a
        # person again what they answered (answered_segments_of says why).
        return _not_recorded("its review or cuts could not be read (%s: %s)"
                             % (type(e).__name__, e))
    answered = wanted if "all" in done else [s for s in wanted if s in done]
    try:
        if recheck and answered:
            res["reopened"] = reopen_answered(video_id, bundle, answered,
                                              segment_details)
            answered = [s for s in answered if s not in res["reopened"]]
        res["answered"] = answered
        ask = [s for s in wanted if s not in answered]
        flagged = (flag_segments(outcomes, ask, reason, details=segment_details)
                   if ask else [])
    except Exception as e:
        return _not_recorded("its review or outcomes file could not be updated "
                             "(%s: %s)" % (type(e).__name__, e))
    res["flagged"] = flagged
    missing = [s for s in ask if s not in flagged]
    if missing:
        res["unrecorded"] = missing
        res["error"] = ("%s; question not recorded for segment(s) %s: the "
                        "bundle's outcomes have no such segment"
                        % (why, ", ".join(str(m) for m in missing)))
    else:
        res["error"] = None


def _held_in_a_queue(video_id: str, db) -> str:
    """Why ``video_id`` counts as already held in a review queue, or "".

    Held = the watcher row says 'triage' or 'deep_review', or a bundle of the
    video exists in either queue, published or in the queue's ``.incoming``
    build folder (a route in progress, or one that was interrupted). The
    folders are checked as well as the row because the row is this machine's
    view and a bundle may have been routed from another. A leftover kept for
    a person (a folder holding review_routing.LEFT_NOTE) is not a bundle and
    does not count (review_routing.is_queue_bundle): the route goes on, and
    the gate deals with that folder (review_gate._clear_the_way)."""
    from mousereach.config import Paths
    from mousereach.watcher.review_routing import INCOMING_DIR_NAME, is_queue_bundle
    if db is not None:
        state = (db.get_video(video_id) or {}).get("state")
        if state in HELD_STATES:
            return ("already held for review (watcher state '%s'); not routed -- "
                    "the held bundle is the live copy" % state)
    for name, root in (("triage", getattr(Paths, "TRIAGE_REVIEW", None)),
                       ("deep review", getattr(Paths, "DEEP_REVIEW", None))):
        if not root:
            continue
        for folder in (Path(root) / video_id, Path(root) / INCOMING_DIR_NAME / video_id):
            if is_queue_bundle(folder):
                return ("already has a bundle in the %s queue (%s); not routed -- "
                        "the held bundle is the live copy" % (name, folder))
    return ""


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
    ap.add_argument("video_id", nargs="?", help="e.g. 20990101_ABC9901_P1")
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
    # record of what it had routed -- nor of where the time went (a run was
    # stopped by its time limit with not one line written).
    results = []
    for v, segs, det in items:
        started = time.time()
        r = route_video(v, args.queue, args.reason, segs, db=db, segment_details=det,
                        recheck=args.recheck)
        r["seconds"] = round(time.time() - started, 1)
        results.append(r)
        if not args.json:
            tag = ("FAIL" if r.get("unrecorded")
                   else "OK  " if r["routed"] else "wait" if r["deferred"]
                   else "lock" if r.get("locked")
                   else "deep" if r.get("in_deep_review")
                   else "held" if r.get("held")
                   else "done" if (r["answered"] and not r["error"])
                   else ("skip" if r["error"] and "not found" in r["error"] else "FAIL"))
            print("%s %s -> %s  flagged=%s  %s%s%s  (%.0f s)" % (
                tag, r["video_id"], r["queue"], r["flagged"] or "-",
                ("asked again: %s  " % r["reopened"]) if r.get("reopened") else "",
                ("already answered by a person: %s  " % r["answered"]) if r["answered"] else "",
                r["error"] or "", r["seconds"]), flush=True)
    if args.json:
        print(json.dumps(results, indent=1))
    bad = [r for r in results if r.get("unrecorded")
           or not r["routed"] and not r["deferred"]
           and not r.get("locked") and not r.get("in_deep_review")
           and not (r.get("held") and not r.get("unrecorded"))
           and not (r["answered"] and not r["error"])
           and not (r["error"] and "not found" in r["error"])]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
