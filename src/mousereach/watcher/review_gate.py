"""The pipeline's human-review GATE + routing.

Called AFTER algo-4 (reach assignment), BEFORE kinematics. Decides a freshly
(re)processed video's fate and, when it cannot proceed on its own, MOVES the
whole bundle into the right review queue so kinematics + the central database never
touch it:

    segmentation failed (hard or soft) OR QC-critical -> DEEP_REVIEW
        (needs the causal/GT deep tools; clearing re-injects just after
        segmentation: reaches, outcomes and assignment are remade)
    any triaged element still UNRESOLVED              -> TRIAGE
        (fast per-element review tool; resolving resumes the video just after
        the step that sent it and runs every later step, this gate included)
    clean / every triaged element already resolved    -> proceed to kinematics

Every route records WHICH STEP sent the video (``held_at_step`` on the watcher
row and in the bundle's manifest); the return reads it to decide where the run
resumes (watcher/resume_point.py has the rule and its reasons).

The moved bundle is self-contained (mp4 + pose h5 + all algo JSONs) and carries a
``manifest.json`` with BUNDLE-LOCAL canonical pointers, so the review tool opens
it in place -- see ``CausalReviewWidget.load_from_manifest`` (loads the video
from ``canonical_video_path`` and globs its parent for the pose h5).

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

from ..config import Paths, is_outcome_free_tray
from ..review.triage_status import TriageStatus, triage_status
from .review_routing import move_video_bundle

logger = logging.getLogger(__name__)

DECISION_CLEAN = "clean"
DECISION_TRIAGE = "triage"
DECISION_DEEP = "deep_review"
# Outputs predate the segments file: mid-flight, not reviewable, not routable.
DECISION_STALE = "stale_outputs"

# Key the watcher writes into <video>_reach_assignments.json when reach
# assignment could not run (orchestrator._hold_for_failed_assignment). The file
# then names no causal reach, so every touched pellet is triaged; the gate's
# reason says why, so a person knows what to answer.
ASSIGNMENT_FAILED_KEY = "assignment_failed"

# The quality verdict the watcher passes when the unified quality triage could
# not run at all (it raised). The gate HOLDS such a video in deep review: an
# unchecked video must not pass as clean (it used to -- the verdict stayed
# 'auto_approved' when triage crashed). The reason below is what the review
# tools show as the routed reason.
QC_TRIAGE_FAILED = "triage_failed"
# The quality triage found the POSE unusable (pipeline/pose_check.py). Never
# given to the gate: the watcher sends such a video for a new pose instead
# (orchestrator._pose_is_unusable), because no review queue can fix a pose.
QC_POSE_UNUSABLE = "pose_unusable"
TRIAGE_FAILED_REASON = ("triage_failed: the automatic quality check could not run "
                        "on this video -- a person checks it")


def _gt_certification(video_id: str, processing_dir=None) -> Tuple[bool, set]:
    """``(fully_exhaustive, gt_determined_outcome_segnums)`` for a video's GT.

    ``fully_exhaustive`` is True when GT marks the whole video complete
    (``completion_status.all_complete`` or both reaches+outcomes exhaustive) --
    a human has certified every element, so the whole video is human truth. The
    second value is the set of segment_nums whose outcome GT determined, used to
    clear those specific segments when GT is only partial. Never raises.

    The ground truth beside the video (``processing_dir``) is found first --
    triage_status.gt_resolved_segments says why; it is the one reading of
    ground truth, so the gate and the release check cannot disagree."""
    from ..review.triage_status import gt_resolved_segments
    return gt_resolved_segments(video_id, extra_dirs=(processing_dir,))


def evaluate_gate(
    video_id: str,
    processing_dir: Path,
    qc_verdict: str = "auto_approved",
) -> Tuple[str, str, TriageStatus]:
    """Decide a video's fate. PURE (no moves, no DB) -- returns
    ``(decision, reason, status)``.

    Priority: an exhaustively GT'd video is certified human truth end-to-end (the
    extractor substitutes GT for every element), so its algo triage flags -- and
    even a failed algo segmentation -- are moot -> CLEAN. Otherwise a failed
    segmentation or a critical QC verdict means the whole video is untrustworthy
    -> DEEP_REVIEW. Otherwise, any triaged element the human has not addressed
    (and that GT did not already determine) -> TRIAGE. Otherwise -> CLEAN.
    """
    st = triage_status(processing_dir, video_id)
    gt_full, gt_segs = _gt_certification(video_id, processing_dir)
    if gt_full:
        return DECISION_CLEAN, "ground_truth_exhaustive", st
    # Easy and Flat trays are never held for a person: they are never
    # triaged, their pellet outcomes do not matter (pellets ARE presented, but
    # nothing is concluded from them), they are not segmented, and they need
    # no kinematics. What they need is the session's reaches: no outcome is
    # computed, so there is nothing a reviewer could answer. A machine problem
    # is still recorded in _triage.json; held in a queue the video would only
    # wait for nobody. The stale-output checks below still run: they send a
    # video round again, not to a person.
    outcome_free = is_outcome_free_tray(video_id)
    if st.seg_failed and not outcome_free:
        # Same honour-the-human rule as qc_needs_review below: a completed
        # deep review outranks the machine's self-check. A soft segmentation
        # failure is re-raised identically by every re-run, so without this a
        # human-cleared video ping-pongs between deep review and reprocessing
        # exactly like the QC case did.
        if _deep_review_cleared(video_id, processing_dir):
            logger.info(
                f"Gate: {video_id} segmentation flagged as failed but a "
                f"human's deep-review clear exists -- honoring the clear.")
        else:
            return DECISION_DEEP, "segmentation_failed", st
    # A boundary change voids everything computed from boundaries. When the
    # outcomes file PREDATES the segments file, downstream was computed
    # against cuts that no longer exist: the video is mid-flight, not
    # reviewable. Judging it routed reviewers to data describing the OLD
    # cuts, repeatedly (hand-fixed segmentations re-circulating to triage).
    # A healthy pipeline always writes segments before
    # outcomes, so this never fires on an ordinary run.
    seg_p = processing_dir / f"{video_id}_segments.json"
    out_p = processing_dir / f"{video_id}_pellet_outcomes.json"
    try:
        if seg_p.exists() and out_p.exists() \
                and out_p.stat().st_mtime < seg_p.stat().st_mtime:
            return (DECISION_STALE,
                    "outcomes predate segments -- boundaries changed after "
                    "they were computed", st)
    except OSError:
        pass
    # The mtime test alone is defeated whenever something rewrites all the
    # files after the fact -- the unified-triage validation_status rewrite
    # touches segments, reaches AND outcomes seconds before this gate runs,
    # so outcomes always look newer (found by a reviewer). The
    # IN-FILE dates cannot be defeated that way: a human correction stamp
    # (segments corrected_at) newer than the reach detector's own run stamp
    # (reaches detected_at) means the reaches were computed against
    # boundaries that no longer exist, whatever the file times say.
    reach_p = processing_dir / f"{video_id}_reaches.json"
    try:
        if seg_p.exists() and reach_p.exists():
            corrected = (json.loads(seg_p.read_text(encoding="utf-8"))
                         .get("corrected_at"))
            if corrected:
                detected = (json.loads(reach_p.read_text(encoding="utf-8"))
                            .get("detected_at"))
                if detected and str(detected) < str(corrected):
                    return (DECISION_STALE,
                            "reaches predate the human boundary correction "
                            "(in-file dates)", st)
    except (OSError, ValueError):
        pass

    # NOT routing on the segmenter's needs_human verdict. It was briefly wired
    # up here and is deliberately switched off.
    #
    # The rule fired on about 10% of ordinary videos, and there was never
    # evidence that was the right 10%. Checked against the three videos a human
    # had actually judged mis-segmented, it caught one. Two of the three had
    # textbook segmentation output -- even spacing, every boundary backed by a
    # real detection -- because they were offset rather than malformed, and no
    # measurement taken inside a single video can see that.
    #
    # The established route for this is a person noticing during review and
    # pressing "Flag Session", which works and has been used. A heuristic that
    # fills the deep-review queue with videos nobody has reason to doubt is
    # worse than no heuristic.
    #
    # The segmenter still RECORDS needs_human in its output; nothing acts on it.
    # If it is ever routed on again, validate it against videos a human has
    # judged first, not against how many videos it routes.
    if outcome_free:
        return DECISION_CLEAN, "outcome-free tray (Easy/Flat): never held", st
    if qc_verdict == "needs_review":
        # A human's completed deep review outranks the machine's self-check
        # (if a human completed all that needed to be reviewed, then it moves
        # on). Without this, a deterministic QC
        # flag -- e.g. an unstable landmark ruler, a property of the tracking
        # that every re-run re-raises identically -- ping-ponged cleared
        # videos between deep review and reprocessing forever. The flag still travels with the outputs in
        # _triage.json, so flagged kinematics stay identifiable downstream.
        if _deep_review_cleared(video_id, processing_dir):
            logger.info(
                f"Gate: {video_id} QC verdict is needs_review but a human's "
                f"deep-review clear exists -- honoring the clear.")
        else:
            return DECISION_DEEP, "qc_needs_review", st
    if qc_verdict == QC_TRIAGE_FAILED:
        # Fail closed: the quality checks never ran, so nothing says the video
        # is sound. Held for a person -- unless a person already cleared it in
        # deep review (same honour rule as above: without it a triage that
        # crashes the same way on every run would bounce a cleared video
        # between deep review and reprocessing for ever).
        if _deep_review_cleared(video_id, processing_dir):
            logger.info(
                f"Gate: {video_id} quality triage could not run but a human's "
                f"deep-review clear exists -- honoring the clear.")
        else:
            return DECISION_DEEP, TRIAGE_FAILED_REASON, st
    # GT-determined segments are resolved; so are segments a human addressed in a
    # review saved ANYWHERE the resolver looks (the Pending bundle, or next to the
    # canonical video) -- not just the fresh work dir triage_status read. Honoring
    # it lets a re-run of an already-reviewed video finalize CLEAN instead of
    # re-triaging (and losing the reviewer's resolution).
    review_resolved = _review_resolved_segments(video_id, processing_dir)
    unresolved = set(st.unresolved) - gt_segs - review_resolved
    # A triaged segment too short to be a pellet presentation cannot be answered
    # in triage (answer_problem refuses it there): it is a question about the
    # cuts, and only something a person can fix in the GUI may hold a video
    # -- so deep review, where the cuts are fixed. If a person
    # has already cleared the video there and kept these cuts, there is no pellet
    # in such a segment to ask about.
    short = sorted(s for s in unresolved if s in st.too_short)
    if short:
        if _deep_review_cleared(video_id, processing_dir):
            logger.info("Gate: %s segment(s) %s too short to be a pellet, cuts kept by "
                        "a person's deep-review clear -- nothing to ask there",
                        video_id, short)
            unresolved -= set(short)
        else:
            from mousereach.review.triage_status import short_segment_reason
            return DECISION_DEEP, short_segment_reason(st, short), st
    if unresolved:
        segs = ", ".join(str(s) for s in sorted(unresolved))
        failed = _assignment_failure(video_id, processing_dir)
        if failed:
            return (DECISION_TRIAGE,
                    f"reach assignment could not run ({failed}) -- choose the "
                    f"causal reach for segment(s) {segs}", st)
        return DECISION_TRIAGE, f"{len(unresolved)} triaged segment(s) unresolved: {segs}", st
    return DECISION_CLEAN, "clean", st


def _assignment_failure(video_id: str, processing_dir: Path) -> str:
    """The recorded reason reach assignment could not run, or "". Never raises."""
    try:
        doc = json.loads((Path(processing_dir) / f"{video_id}_reach_assignments.json")
                         .read_text(encoding="utf-8"))
        return str(doc.get(ASSIGNMENT_FAILED_KEY) or "")
    except Exception:
        return ""


def _deep_review_cleared(video_id: str, processing_dir: Path) -> bool:
    """True when a human's deep-review clear marker exists for this video.

    The clear marker (``{video_id}_deep_review_cleared.json``) is written when a
    person clears a video in the deep-review tools, and review-return moves it
    into the processing dir with the rest of the bundle -- so at gate time it
    sits beside the fresh outputs. Never raises."""
    try:
        return (Path(processing_dir) / f"{video_id}_deep_review_cleared.json").is_file()
    except Exception:
        return False


def _review_resolved_segments(video_id: str, processing_dir: Path):
    """Resolved segment numbers from a saved human review found ANYWHERE the
    resolver looks (processing dir, Pending bundle, next to the canonical video).
    Empty set on any error -- never blocks the gate."""
    try:
        import json
        from ..review.causal_review_io import resolve_review_path
        from ..review.triage_status import resolved_segments
        rp = resolve_review_path(video_id, processing_dir)
        if rp is None or not rp.exists():
            return set()
        # Matched by frames against the segmentation these outputs came from,
        # as the results step applies it (causal_review_io.answered_segments).
        seg_path = Path(processing_dir) / f"{video_id}_segments.json"
        seg_doc = json.loads(seg_path.read_text(encoding="utf-8")) if seg_path.is_file() else None
        return resolved_segments(json.loads(rp.read_text(encoding="utf-8")), seg_doc)
    except Exception:
        return set()


def _in_staging(path: Path) -> bool:
    """True when ``path`` sits directly in the pose staging folder
    (``Paths.DLC_STAGING``). Read at call time so it always matches the folder
    the resolver searched. Never raises."""
    staging = getattr(Paths, "DLC_STAGING", None)
    if not staging:
        return False
    try:
        return (os.path.normcase(os.path.normpath(str(Path(path).parent)))
                == os.path.normcase(os.path.normpath(str(staging))))
    except (TypeError, ValueError):
        return False


def _copy_pose_into_bundle(pose: Path, bundle: Path, video_id: str) -> Optional[Path]:
    """Copy a staged pose file into ``bundle`` under its own name; return the
    copy, or None when the copy failed (and nothing is left behind).

    WHY copy to ``.part`` then rename: a half-written file under a pose name
    would be taken for the pose by every resolver that globs the bundle. WHY a
    copy and not a move: the staged pose also belongs to the intake that pulls
    it from staging; taking it away here would strand that side."""
    pose = Path(pose)
    dest = bundle / pose.name
    part = bundle / (pose.name + ".part")
    try:
        shutil.copy2(pose, part)
        if part.stat().st_size != pose.stat().st_size:
            raise OSError("copied size differs from the staged pose")
        os.replace(part, dest)
        return dest
    except OSError as e:
        logger.warning(
            f"{video_id}: could not copy staged pose {pose.name} into the review "
            f"bundle ({e}); the manifest's pose pointer is left empty and the "
            f"return path will resolve the pose when the bundle returns.")
        try:
            part.unlink()
        except OSError:
            pass
        return None


def _write_review_manifest(bundle: Path, video_id: str, reason: str,
                           held_at_step: Optional[str] = None) -> None:
    """Write the ``manifest.json`` the review tool needs to open a self-contained
    bundle -- BUNDLE-LOCAL canonical pointers so the video + pose load in place.

    ``provenance.held_at_step`` records which step sent the video (route_to_queue);
    the return reads it, so it travels with the bundle even where the row
    cannot be written (a review window routing with no watcher database).
    """
    mp4 = bundle / f"{video_id}.mp4"
    h5s = sorted(p for p in bundle.glob(f"{video_id}DLC*.h5") if p.is_file())
    # WHY select_pose_file, not the first name: a bundle can hold poses from two
    # models, and the review tools trust this pointer before choosing for
    # themselves -- the first name sorted is often the OLD model's pose. Same
    # choice (declared scorer first) the return path makes.
    if h5s:
        from mousereach.pipeline.manifest import select_pose_file
        chosen = select_pose_file(h5s)
        pose_path = str(chosen) if chosen is not None else str(h5s[0])
    else:
        pose_path = None
    self_contained = bool(h5s)
    if pose_path is None:
        # No pose in the bundle: record where the canonical pose LIVES, so
        # the return path and the review tools can load it. A manifest
        # claiming self_contained with a null pose pointer put one video into
        # an every-2-minutes return-refusal loop. Reuse the
        # return path's own resolver (Analyzed model trees, then
        # declared-scorer staging) so the two can never disagree.
        try:
            from mousereach.watcher.review_return import _resolve_inputs
            _mp4, _pose = _resolve_inputs(bundle, video_id)
            if _pose is not None and _in_staging(_pose):
                # WHY never point at staging: it is a transient handover folder.
                # Its pose moves on (intake pulls it; a layout change set the old
                # staging folder aside where no resolver reads), and a manifest
                # still naming it makes the bundle refuse to return every cycle
                # and open with no pose in the review tool. Carry a copy IN the
                # bundle instead; if the copy fails, leave the pointer null and
                # let the return path resolve the pose when the bundle returns.
                copied = _copy_pose_into_bundle(_pose, bundle, video_id)
                if copied is not None:
                    pose_path = str(copied)
                    self_contained = True
            elif _pose is not None:
                pose_path = str(_pose)
        except Exception as e:
            logger.debug(f"{video_id}: manifest pose resolve failed: {e}")
    manifest = {
        "type": "causal_review_bundle",
        "schema_version": "1.0",
        "video_stem": video_id,
        "canonical_video_path": str(mp4),
        "canonical_dlc_h5_path": pose_path,
        "provenance": {
            "routed_reason": reason,
            "staged_at": datetime.now().isoformat(),
            # True only when the pose really is IN the bundle.
            "self_contained": self_contained,
            "held_at_step": held_at_step,
        },
    }
    # One-step write retried while the share holds the file (pipeline/fsutil):
    # a plain write_text failed on a passing hold, and a cut-off manifest is
    # unreadable to the review tools.
    from mousereach.pipeline.fsutil import write_json_atomic
    write_json_atomic(bundle / f"{video_id}_manifest.json", manifest, indent=2)


def supersede_deep_review_clear(video_id: str, source_dir: Path) -> Optional[Path]:
    """Set aside an earlier deep-review clear when a reviewer escalates.

    Pressing "Escalate: bad segmentation -> re-segmentation" means the reviewer has
    found a new problem with the cuts, so the button removes any earlier deep
    review clearing. Only the button does this: the automatic routes into
    deep review fire only when no clear exists. Left in place, it travelled with
    the bundle, and the return scan -- which reads "cleared" from the marker
    alone -- released the video from deep review within seconds of the
    escalation, re-ran it on the same cuts and put it back in triage: the
    reviewer's "Escalate: bad segmentation" button did nothing visible.
    The marker is
    renamed, never deleted, so the earlier clear stays on record and goes on
    travelling with the bundle under a name nothing reads as a clear. Done
    BEFORE the move, so no scan can see the old marker in the new queue. A
    rename that fails raises: moving the video with a live clear would only
    bounce it again, and the caller says so to the reviewer.
    """
    marker = Path(source_dir) / f"{video_id}_deep_review_cleared.json"
    if not marker.is_file():
        return None
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    kept = marker.with_name(f"{video_id}_deep_review_cleared_superseded_{stamp}.json")
    marker.rename(kept)
    logger.info("Routing %s to deep review: earlier deep-review clear set aside as %s",
                video_id, kept.name)
    return kept


def escalate_to_deep_review(video_id: str, bundle: Path, reason: str,
                            db=None) -> Path:
    """The triage tool's Escalate button: set aside any earlier deep-review
    clear (see supersede_deep_review_clear), then move the video to deep review
    and record 'deep_review' in the watcher database ``db``.

    WHY the database: the button used to route with no database, so the bundle
    sat in deep review while the row still said 'triage' -- status views read
    the row, the GPU nodes and reconcile read the folder, and the two disagreed
    until the video came back. Pass the daemon's own database
    (db_location.open_watcher_db_if_present). With ``db`` None the move still
    happens (the person's decision is what matters) and the disagreement is
    logged; it heals when the video returns from deep review."""
    # Under the bundle claim, from setting the old clear aside to the end of
    # the move: the return scan acts on the same bundle under that claim, and
    # without it both could work on one folder at once (BundleBusyError).
    if not take_bundle_claim(video_id):
        raise BundleBusyError(f"{video_id}: {BUSY_MESSAGE}")
    try:
        supersede_deep_review_clear(video_id, bundle)
        if db is None:
            logger.warning(f"Escalate {video_id}: no watcher database given; the bundle "
                           f"moves to deep review but the row keeps its state until "
                           f"the video returns")
        return route_to_queue(video_id, Path(bundle), Paths.DEEP_REVIEW,
                              reason=reason, db=db, db_state="deep_review")
    finally:
        release_bundle_claim(video_id)


class AlreadyInQueueError(RuntimeError):
    """Raised by route_to_queue, before anything moves, when the destination
    queue already holds the video's LIVE bundle (the video is held there) and
    the source is somewhere else. See route_to_queue for why nothing is
    merged, and _clear_the_way for the folders that are not a live bundle."""


class BundleBusyError(AlreadyInQueueError):
    """Raised by route_to_queue (and escalate_to_deep_review), before anything
    moves, when the video's queue bundle is being moved by another program
    right now: another process or machine holds the bundle claim the return
    scan takes (review_return._claim_return). A subclass of
    AlreadyInQueueError so every caller that already treats "refused, nothing
    moved, try later" handles it the same way (the watcher waits and runs the
    video again; it never counts as a failure of the video)."""


def is_queue_bundle_folder(folder) -> bool:
    """Is ``folder`` a per-video folder directly inside a review queue?"""
    if not folder:
        return False
    parent = Path(folder).parent
    return any(_same_folder(parent, getattr(Paths, q, None))
               for q in ("TRIAGE_REVIEW", "DEEP_REVIEW"))


def take_bundle_claim(video_id: str) -> bool:
    """Take the video's bundle claim for this thread (re-entrant; see the
    block comment above review_return._claim_return). True when held; every
    True must be matched by one release_bundle_claim. For a review window
    that does several things to one bundle (set aside an old clear, then
    move it) under one claim."""
    from .review_return import _claim_return
    return _claim_return(video_id)


def release_bundle_claim(video_id: str) -> None:
    from .review_return import _release_return_claim
    _release_return_claim(video_id)


BUSY_MESSAGE = ("another program (the watcher's return scan, the route command or "
                "a re-pose) is moving this video's bundle right now; nothing was "
                "moved -- try again in a minute")


# Watcher states in which a bundle in a queue is the video's live copy.
HELD_STATES = ("triage", "deep_review")


def _inside(path, root) -> bool:
    if not path or not root:
        return False
    try:
        r = os.path.normcase(os.path.abspath(str(root))).rstrip("\\/")
        q = os.path.normcase(os.path.abspath(str(path)))
        return q == r or q.startswith(r + os.sep)
    except (TypeError, ValueError):
        return False


def _iso_epoch(text) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(text)).timestamp()
    except (TypeError, ValueError):
        return None


def folder_last_touched(folder: Path, video_id: str) -> Optional[float]:
    """When ``folder`` was last routed or written: the later of its routing
    record's ``routed_at`` and its newest file's modified time. None when
    neither can be read."""
    from .review_routing import read_routing_manifest
    times = []
    t = _iso_epoch((read_routing_manifest(folder, video_id) or {}).get("routed_at"))
    if t is not None:
        times.append(t)
    try:
        for f in Path(folder).iterdir():
            try:
                if f.is_file():
                    times.append(f.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        pass
    return max(times) if times else None


def queue_folder_is_live(video_id: str, existing: Path, processing_dir, row,
                         live_states=HELD_STATES):
    """``(live, why)``: whether ``existing`` = <queue>/<video> is the video's
    LIVE bundle, which a route from ``processing_dir`` must not replace.
    ``row`` is the watcher row (None when there is no database to ask).

    Live when any of these holds:
      * there is no database to ask (the safe side);
      * the row says the video is held in this queue (``live_states``);
      * the folder is a published bundle (it holds the queue's routing record
        or review manifest) or holds a file a person wrote -- UNLESS this
        route is a newer run of the video: the source is a processing folder
        (not a review queue, not the filed results) and the row's
        ``processing_started_at`` is later than the folder was last routed or
        written (folder_last_touched).
    Otherwise (only machine outputs and inputs, no queue records, nothing a
    person wrote) it is an old copy.

    WHY not the row alone: the row is one machine's view, and it can be
    wrong. A bundle escalated from the triage window before the window
    recorded the move kept the row at 'triage' while the video, and the
    reviewer's saved review, sat in deep review. Judged by the row alone,
    the next route of that video into deep review from anywhere else took the
    reviewer's live bundle for an old copy and moved it out of the queue.
    What the folder holds (a routing record, a person's file) and when it was
    last touched are on the shared folder, where every machine sees the same
    thing. Only a pipeline run that started after the folder was last touched
    shows the folder has been superseded."""
    from .review_routing import is_queue_record, person_entries
    if row is None:
        return True, "there is no watcher database to ask"
    state = row.get("state")
    if state in live_states:
        return True, f"its watcher state is '{state}'"
    try:
        names = [f.name for f in Path(existing).iterdir() if f.is_file()]
        person = person_entries(existing, video_id)
    except OSError as e:
        return True, f"the folder could not be read ({e})"
    published = any(is_queue_record(video_id, n) for n in names)
    if not published and not person:
        return False, "it holds no queue record and nothing a person wrote"
    what = ("a person's file (" + ", ".join(person) + ")") if person else \
        "the queue's routing record or review manifest"
    from_a_run = not (_inside(processing_dir, getattr(Paths, "REVIEW_ROOT", None))
                      or _inside(processing_dir, getattr(Paths, "ANALYZED_OUTPUT", None)))
    started = _iso_epoch(row.get("processing_started_at"))
    touched = folder_last_touched(Path(existing), video_id)
    if from_a_run and started is not None and touched is not None and started > touched:
        return False, (f"it holds {what}, but this route is a pipeline run that "
                       f"started after the folder was last routed or written")
    return True, (f"it holds {what} and this route is not a run that started after "
                  f"the folder was last routed or written")


def _clear_the_way(video_id: str, existing: Path, processing_dir, reason: str,
                   db, extra_sources: Optional[Iterable[Path]] = None) -> None:
    """Deal with a folder ``existing`` = <queue>/<video> met by a route from
    somewhere else, BEFORE anything moves. Returns when the route may go on;
    raises AlreadyInQueueError when it may not.

      * empty: removed (a return whose last step failed leaves one; if it
        cannot be removed the route fills it -- there is nothing to replace);
      * the video's LIVE bundle (queue_folder_is_live: held in this queue by
        its watcher state, no database to ask, or a published bundle or a
        person's file that no later pipeline run has superseded):
        AlreadyInQueueError, and the refusal is listed for a person in the
        attention panel (review_routing.note_folder_in_the_way);
      * otherwise a leftover (only queue records; machine outputs and inputs;
        or a folder a later pipeline run of the video has superseded):
        review_routing.clear_leftover, with the files this route brings as
        the live copy -- what has a current copy elsewhere is deleted, the
        rest stays where it is and is listed. When nothing is kept the route
        goes on; when something is kept the folder is still there, so the
        route is refused (AlreadyInQueueError) rather than merged into it.

    WHY: refusing every existing folder would fail later runs of a video for
    good -- an empty residue, a records-only folder or an old bundle of
    machine outputs would make the gate raise on every run. Merging into the
    folder (what routes used to do) replaced a held run's files and a
    person's saved review. The state is matched to THIS queue because a
    video held in triage can have a stale deep-review folder of machine
    outputs (and the other way round); refusing on the other queue's state
    blocked a person's Escalate until someone cleaned up by hand. The folder's
    own contents are read as well as the row (queue_folder_is_live says why)."""
    from .review_routing import (_bundle_files, clear_leftover, is_queue_record,
                                 left_for_a_person, note_folder_in_the_way)
    from .review_return import _claim_return_nested, _release_return_claim

    def _refuse(why: str, what_to_do: str, listed: bool = False):
        says = (f"{video_id} could not be routed to {existing.parent.name} "
                f"({reason}): {why}. Nothing was moved from {processing_dir}. "
                f"{what_to_do}")
        if not listed:
            note_folder_in_the_way(video_id, existing, says)
        raise AlreadyInQueueError(says)

    try:
        entries = list(existing.iterdir())
    except OSError as e:
        _refuse(f"the folder {existing} is there and cannot be read ({e})",
                "Check that folder, then let the video run again.")
    if not entries:
        try:
            existing.rmdir()
        except OSError:
            pass
        return
    records_only = all(e.is_file() and is_queue_record(video_id, e.name)
                       for e in entries)
    if not records_only and not left_for_a_person(existing):
        row = None
        if db is not None:
            try:
                row = db.get_video(video_id) or {}
            except Exception:
                row = {}
        if _same_folder(existing.parent, getattr(Paths, "TRIAGE_REVIEW", None)):
            live_states = ("triage",)
        elif _same_folder(existing.parent, getattr(Paths, "DEEP_REVIEW", None)):
            live_states = ("deep_review",)
        else:
            live_states = HELD_STATES           # a queue not configured here
        live, why = queue_folder_is_live(video_id, existing, processing_dir, row,
                                         live_states)
        if live:
            _refuse(f"the queue already holds its live bundle {existing} ({why})",
                    "A video is in one place at a time, and nothing is merged into "
                    "a live bundle. If the bundle in the queue is the copy to keep, "
                    "finish its review as usual. If the newer run should replace it, "
                    "a person must first take that bundle out of the queue.")
    claimed, release = _claim_return_nested(video_id)
    if not claimed:
        _refuse(f"another program is moving the folder {existing} right now",
                "It is routed on its next try once that move is done.")
    try:
        what = "queue records only" if records_only else "an older copy of the video"
        # The live copy: the files this route brings. A leftover file the route
        # brings byte for byte is deleted; machine outputs are deleted because
        # this run makes its own.
        live: dict = {}
        try:
            for f in _bundle_files(Path(processing_dir), video_id):
                live[f.name] = f
        except OSError:
            pass
        for f in extra_sources or []:
            if Path(f).is_file():
                live.setdefault(Path(f).name, Path(f))
        kept = clear_leftover(
            existing, video_id, live,
            why=(f"{what} was in the queue when a later run of the video was routed "
                 f"there ({reason}); a video is in one place at a time"),
            what_to_do=("look at what is kept; the answers here (review, ground truth, "
                        "cuts) belong to this older copy's cuts. While this folder is "
                        "there, the video cannot be routed into this queue: it waits "
                        "and the route is tried again."))
    finally:
        if release:
            _release_return_claim(video_id)
    if kept:
        _refuse(f"an older folder of the video, {existing}, holds "
                f"{', '.join(kept)}, which has no current copy anywhere else, and it "
                f"is left where it is",
                "It is listed for a person; the route is tried again once the "
                "folder is gone.", listed=True)
    logger.warning(f"Routing {video_id}: {existing} was not the live bundle ({what}); "
                   f"cleared before routing")


def route_to_queue(
    video_id: str,
    processing_dir: Path,
    dest_root: Path,
    reason: str,
    db=None,
    db_state: Optional[str] = None,
    extra_sources: Optional[Iterable[Path]] = None,
    held_at_step: Optional[str] = None,
    all_or_nothing: bool = False,
) -> Path:
    """Move a video bundle into a review queue, write the review manifest, and
    set the DB hold state. Returns the bundle dir.

    ``held_at_step`` is the step that sent the video (resume_point.RESUME_AFTER
    names them). It is written to the row and to the bundle's manifest, so the
    return resumes just after that step. Left None it is the step the queue
    stands for: 'segmentation' for deep review (where cuts are fixed),
    'outcome' for triage (pellet outcomes and causal reaches).

    ``all_or_nothing`` True (routes out of the FILED folder): if any file
    cannot be moved, everything is put back and
    review_routing.RouteIncompleteError is raised, with nothing routed and no
    state changed (review_routing.move_video_bundle says why).

    Raises CrystallizedError, before anything moves, when ``db`` says the
    video is crystallized (publication-locked; db.crystallized_refusal).
    Raises BundleBusyError, before anything moves, when the source is a queue
    bundle and another program holds its bundle claim (see the comment in
    the body).

    DEEP REVIEW SUPERSEDES TRIAGE (review_routing, block comment above
    deep_review_stems):
      * into triage while the video has a deep-review bundle: nothing is moved
        or created; the video stays in deep review, its row is set to
        'deep_review', and the deep-review bundle is returned. Files at the
        source stay where they are (logged). Only a live deep-review bundle
        counts (review_routing.in_deep_review / is_queue_bundle): a deep-
        review folder that is a leftover kept for a person is not one, and
        the route into triage goes on as usual. WHY: taking a leftover for
        the hold relabelled the row, moved nothing, and left the video with
        no live bundle anywhere.
      * into deep review: once the deep-review bundle is complete (no file
        failed to move), the video's triage bundle, if any, is emptied
        without losing a person's file (_drop_triage_copy).

    A VIDEO IS NEVER MERGED INTO A FOLDER IT ALREADY HAS IN THAT QUEUE:
      * the source IS that folder (routed onto itself): nothing moves, only
        the database state is recorded (see below);
      * the source is anywhere else (Processing, the filed folder, the other
        queue): _clear_the_way decides, before anything moves -- a folder that
        is not the live bundle (empty, records only, or an old copy of a video
        that is no longer held) is cleared as a leftover (review_routing.
        clear_leftover) and the route goes on, unless something in it has no
        current copy elsewhere: that stays where it is, listed for a person,
        and the route raises AlreadyInQueueError; so does the live bundle of
        a held video. WHY: the move used to pour the source's files into the
        existing bundle, replacing same-named files -- an older filed
        generation replaced the held run's results, and a review a person had
        saved in the bundle could be replaced by an older one."""
    from .db import CrystallizedError, crystallized_refusal
    locked = crystallized_refusal(db, video_id, "routed to a review queue")
    if locked:
        raise CrystallizedError(locked)
    # A move OUT OF a review queue (Escalate, Flag Session, the route command,
    # the return scan's diverts) happens under the video's bundle claim, the
    # one the return scan takes before it touches a bundle. WHY: the window's
    # buttons moved a queue bundle with no claim, so the scan on another
    # machine (or the watcher on this one) could be returning or diverting
    # the same folder at that moment and the bundle was torn between two
    # places. The claim is re-entrant per thread, so a route started by a
    # holder (the scan, the route command) goes on as before. A route from
    # Processing takes no claim here; _clear_the_way claims before it clears
    # a folder already in the destination queue.
    from_queue = is_queue_bundle_folder(processing_dir)
    if from_queue and not take_bundle_claim(video_id):
        raise BundleBusyError(f"{video_id}: {BUSY_MESSAGE}")
    try:
        return _route_to_queue_claimed(
            video_id, processing_dir, dest_root, reason, db=db,
            db_state=db_state, extra_sources=extra_sources,
            held_at_step=held_at_step, all_or_nothing=all_or_nothing)
    finally:
        if from_queue:
            release_bundle_claim(video_id)


def _route_to_queue_claimed(
    video_id: str,
    processing_dir: Path,
    dest_root: Path,
    reason: str,
    db=None,
    db_state: Optional[str] = None,
    extra_sources: Optional[Iterable[Path]] = None,
    held_at_step: Optional[str] = None,
    all_or_nothing: bool = False,
) -> Path:
    """route_to_queue's work, run under the bundle claim when the source is a
    queue bundle (see route_to_queue)."""
    from .review_routing import in_deep_review
    from .resume_point import QUEUE_STEP

    def _step_for(state: str) -> Optional[str]:
        # Deep review is where cuts are fixed, whatever sent the video there.
        if state == "deep_review":
            return QUEUE_STEP["deep_review"]
        return held_at_step or QUEUE_STEP.get(state or "")

    def _record_state(state: str, why: str) -> None:
        if db is None or not state:
            return
        # The step that sent it, and no resume point yet (one is set when a
        # person's answers return it): a resume point from an earlier round
        # must never outlive a new hold.
        hold = {"held_at_step": _step_for(state), "resume_stage": None}
        try:
            db.update_state(video_id, state, **hold)
        except Exception as e:
            # The bundle has ALREADY been moved into the review queue above, so
            # the filesystem is the ground truth here and the DB has to follow it.
            # Previously this only warned, which left the video sitting in a review
            # folder while the DB reported some other state -- a silent
            # disagreement between disk and database, visible as one log line and
            # nothing else. Reconcile deliberately instead of diverging.
            try:
                db.force_state(
                    video_id, state,
                    reason=f"{why}; normal transition rejected ({e})", **hold)
            except Exception as e2:
                logger.error(
                    f"Gate: {video_id} is in the '{state}' queue ON DISK but its "
                    f"DB state could not be set ({e2}). Disk and DB now disagree.")

    dest_root = Path(dest_root)
    if _same_folder(processing_dir, dest_root / video_id):
        # The video is ALREADY in this queue: the source folder is the bundle
        # itself (e.g. Flag Session pressed in the re-segmentation window).
        # Nothing is moved and neither the routing record nor the review
        # manifest is rewritten -- the bundle is already where the caller
        # wants it. Only the database is brought in line. WHY: moving each
        # file onto itself used to delete it (review_routing._safe_move).
        logger.warning(
            f"Routing {video_id}: already in {dest_root} ({reason}); nothing "
            f"moved")
        _record_state(db_state, "routed onto its own bundle: already in this queue")
        return dest_root / video_id
    if (_same_folder(dest_root, getattr(Paths, "TRIAGE_REVIEW", None))
            and in_deep_review(video_id)):
        deep_bundle = Path(Paths.DEEP_REVIEW) / video_id
        logger.warning(
            f"Routing {video_id}: NOT routed to triage ({reason}) -- the video is "
            f"in deep review, which supersedes triage; it stays there. Nothing was "
            f"moved from {processing_dir}.")
        _record_state("deep_review",
                      f"routing into triage refused: the video is in deep review")
        return deep_bundle

    existing = dest_root / video_id
    if existing.is_dir():
        _clear_the_way(video_id, existing, processing_dir, reason, db,
                       extra_sources=extra_sources)

    is_deep = _same_folder(dest_root, getattr(Paths, "DEEP_REVIEW", None))
    step = _step_for("deep_review" if is_deep else (db_state or "triage"))
    extra = {"db_state": db_state} if db_state else {}
    if step:
        # Kept in the routing record too, so a review manifest the route could
        # not write is rebuilt with it (review_return._repair_missing_manifest)
        # and the return still resumes just after the step that sent it.
        extra["held_at_step"] = step
    bundle, moved = move_video_bundle(
        video_id, processing_dir, dest_root, reason,
        extra_sources=extra_sources,
        extra_manifest=extra or None,
        all_or_nothing=all_or_nothing,
    )
    # The hold is recorded the moment the bundle is in the queue, BEFORE the
    # review manifest is written. WHY this order: a manifest write that failed
    # (a passing hold on the share) used to raise out of here with the files
    # already in the queue and the row never set -- the dispatcher then
    # marked the video failed, and its retry needed an mp4 that had moved.
    _record_state(db_state, f"review gate moved bundle to '{reason}'")
    from .review_routing import clear_folder_in_the_way
    clear_folder_in_the_way(video_id)     # an earlier refusal is resolved
    try:
        _write_review_manifest(bundle, video_id, reason, held_at_step=step)
    except Exception as e:
        # The bundle is published and the row says so; only the file the
        # review tools list a bundle by is missing. The return scan rebuilds
        # it from the routing record (review_return.repair_unfinished_routes),
        # so this is logged, not raised.
        logger.error(
            f"Routing {video_id}: the bundle is in {bundle} and its state is "
            f"recorded, but its review manifest could not be written ({e}); "
            f"the return scan writes it again shortly")
    # If the SOURCE was a per-video bundle dir (a queue-to-queue divert) and
    # the move emptied it, remove it: an empty date-named dir left behind in
    # a queue reads as a seg-failed bundle and re-diverts "(0 files)" forever
    # -- the phantom that re-flipped a parked video. The shared
    # Processing dir is never named like a video stem, so it never matches.
    processing_dir = Path(processing_dir)
    try:
        if processing_dir.name == video_id and not any(processing_dir.iterdir()):
            processing_dir.rmdir()
    except OSError:
        pass
    if _same_folder(dest_root, getattr(Paths, "DEEP_REVIEW", None)):
        _drop_triage_copy(video_id, bundle)
    return bundle


def _same_folder(a, b) -> bool:
    if not a or not b:
        return False
    try:
        return (os.path.normcase(os.path.abspath(str(a)))
                == os.path.normcase(os.path.abspath(str(b))))
    except (TypeError, ValueError):
        return False


# Files of a video that are its INPUTS (the video and the pose, with the pose
# step's side files); everything else of the video in a processing folder is an
# output of the run, made again whenever the video is analysed again.
def _is_input_file(video_id: str, name: str) -> bool:
    return name == f"{video_id}.mp4" or name.startswith(f"{video_id}DLC")


def discard_run_outputs(video_id: str, processing_dir: Path, deep_bundle: Path) -> dict:
    """Take a run's files out of ``processing_dir`` when the gate could not
    route the video to triage because it is in deep review.

    A video is in one place at a time, and the deep-review bundle is its live
    copy. So, for each of the video's files in ``processing_dir`` (only there:
    a file elsewhere, e.g. a filed copy, is never touched):
      * a machine output of this run (segments, reaches, outcomes, assignment,
        manifest, triage record ... -- review_routing.is_machine_output) is
        deleted -- it is made again when the video leaves deep review and is
        analysed again;
      * a file a person wrote (a review, ground truth, a clear marker, cuts
        set by hand -- review_routing.is_person_file) is deleted only when the
        deep-review bundle holds the same file byte for byte; otherwise it
        stays where it is and is listed for a person in the attention panel
        (review_routing.note_for_a_person, kind 'person_file_left'). WHY:
        "not an input, so an output of this run" counted a person's files as
        outputs and deleted them; and a person's work is never moved away
        where nobody looks;
      * an input (the video or a pose file) is deleted only when the
        deep-review bundle holds the same file (same name and content); one
        that could be the only copy is set aside, never deleted, to
        ``<processing root>/_finished_elsewhere/<date>/<folder>/`` -- the set-
        aside folder the watcher already uses for local copies of videos held
        or finished elsewhere.
    Returns ``{"deleted": [...], "set_aside": [...], "kept": [...],
    "failed": [...]}`` and logs what it did. Never raises."""
    import shutil
    from datetime import date as _date
    from mousereach.archive.core import _same_file
    from .review_routing import (_bundle_files, _same_content, is_person_file,
                                 note_for_a_person)
    out = {"deleted": [], "set_aside": [], "kept": [], "failed": []}
    try:
        files = list(_bundle_files(Path(processing_dir), video_id))
    except OSError:
        files = []
    root = getattr(Paths, "PROCESSING_ROOT", None) or Path(processing_dir).parent
    aside = Path(root) / "_finished_elsewhere" / _date.today().isoformat()
    for f in files:
        try:
            twin = Path(deep_bundle) / f.name
            if is_person_file(f, video_id):
                if twin.is_file() and _same_content(f, twin):
                    f.unlink()
                    out["deleted"].append(f.name)
                else:
                    out["kept"].append(f.name)
                    note_for_a_person(
                        "person_file_left", video_id, f,
                        says=(f"{video_id} is in deep review ({deep_bundle}), so this "
                              f"run's files were taken out of {processing_dir}; but "
                              f"{f.name} is a person's file that the deep-review "
                              f"bundle does not hold, so it was left at {f}. Look at "
                              f"it; delete it when it is not needed (this line then "
                              f"goes away by itself). While it is there, the video's "
                              f"return from deep review into this folder waits."))
                continue
            if _is_input_file(video_id, f.name):
                deletable = twin.is_file() and _same_file(f, twin)
            else:
                deletable = True        # a machine output of this run
            if deletable:
                f.unlink()
                out["deleted"].append(f.name)
            else:
                target = aside / f.parent.name
                target.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(target / f.name))
                out["set_aside"].append(str(target / f.name))
        except OSError as e:
            out["failed"].append(f"{f.name} ({e})")
    logger.warning(
        f"{video_id}: in deep review, so this run's files were taken out of "
        f"{processing_dir}: {len(out['deleted'])} deleted (outputs, or files the "
        f"deep-review bundle holds identically), {len(out['set_aside'])} input(s) "
        f"that could be the only copy set aside to {aside}"
        + (f"; a person's file(s) left in place and listed: {', '.join(out['kept'])}"
           if out["kept"] else "")
        + (f"; could not remove: {', '.join(out['failed'])}" if out["failed"] else ""))
    return out


def _drop_triage_copy(video_id: str, deep_bundle: Path) -> None:
    """Empty the triage copy under the video's bundle claim (see
    _empty_triage_copy for what it does). Never raises.

    WHY the claim: emptying the copy is several moves and deletes, and the
    return scan on another machine acts on the same folder under the claim
    (a byte-identical triage leftover of a video in deep review is cleared
    there). Without the claim both could work on one folder at once. When the
    claim is refused, nothing is touched: the return scan meets the leftover
    on its next pass and deals with it under the claim.

    A triage folder that is already a leftover kept for a person (it holds
    review_routing.LEFT_NOTE) is not a triage copy and is left as it is: what
    it holds was already judged to have no current copy elsewhere, and it is
    listed for a person (review_routing.is_queue_bundle)."""
    from .review_return import _claim_return_nested, _release_return_claim
    from .review_routing import is_queue_bundle
    triage_root = getattr(Paths, "TRIAGE_REVIEW", None)
    if not triage_root or not is_queue_bundle(Path(triage_root) / video_id):
        return                          # no triage copy: nothing to claim
    claimed, release = _claim_return_nested(video_id)
    if not claimed:
        logger.warning(f"Routing {video_id}: its triage copy is being moved by "
                       f"another process right now; left for the return scan")
        return
    try:
        _empty_triage_copy(video_id, deep_bundle)
    finally:
        if release:
            _release_return_claim(video_id)


def _empty_triage_copy(video_id: str, deep_bundle: Path) -> None:
    """Empty ``<triage>/<video>`` once the video's deep-review bundle is
    complete (deep review supersedes triage), without losing anything a
    person wrote. Never raises.

    Never touched while the deep-review bundle's routing record lists a file
    that failed to move (those files may still be in the triage folder, and
    may be the video's only copy): it is listed for a person in the attention
    panel, and the return scan empties it once the deep-review bundle is whole
    (review_return.repair_unfinished_routes). Otherwise:
      * the video, or a pose file, when the deep-review bundle has no file of
        that kind at all (review_routing.input_kind), is MOVED into it: it may
        be the video's only copy, and the live bundle needs it;
      * then the folder is a leftover, with the deep-review bundle as the live
        copy (review_routing.clear_leftover): its queue records and machine
        outputs, any file the deep-review bundle holds byte for byte, and a
        review the durable review store already holds byte for byte are
        deleted (nothing is written into the store: these answers were made
        on cuts that are about to be redone); anything else -- ground truth,
        a review the store does not hold byte for byte, a clear marker, hand-set
        cuts, a video or pose that differs from the deep-review bundle's --
        stays where it is, with a note, listed for a person. Every reader
        ignores a triage folder of a video in deep review (deep review
        supersedes triage) and the return scan skips a folder holding the
        note.
    No answer is carried into the deep-review bundle here. WHY: re-segmentation
    voids every answer after it, so no answer file is re-homed into the bundle
    whose cuts are being redone. (Escalate and the return scan's mislabel
    divert are different: they route the triage bundle ITSELF into deep
    review, and its review and ground truth travel with it -- see
    docs/REVIEW_TOOLS.md.) NOT YET DONE: the durable store's copy of the
    review is still read and matched to the new cuts by frame span
    (causal_review_io.index_review_by_segment); voiding it is separate work.
    WHY at all: the whole folder used to be deleted (shutil.rmtree), a
    person's ground truth and clear markers with it; only reviews had a
    durable copy."""
    from .review_routing import (read_routing_manifest, input_kind, clear_leftover,
                                 _safe_move, note_folder_in_the_way)
    triage_root = getattr(Paths, "TRIAGE_REVIEW", None)
    if not triage_root:
        return
    copy = Path(triage_root) / video_id
    try:
        if not copy.is_dir():
            return
    except OSError:
        return
    record = read_routing_manifest(deep_bundle, video_id) or {}
    if record.get("failed_files"):
        says = (f"{video_id} is in deep review, but {len(record['failed_files'])} "
                f"file(s) failed to move into its bundle {deep_bundle}; its triage "
                f"folder {copy} is kept until they are in (the watcher moves them in "
                f"by itself when it can, then empties this folder). If it stays "
                f"listed, see the line that lists the files missing from the "
                f"deep-review bundle.")
        logger.error(f"Routing {video_id}: {says}")
        note_folder_in_the_way(video_id, copy, says)
        return
    deep_bundle = Path(deep_bundle)
    carried = []
    try:
        held = {input_kind(video_id, g.name) for g in deep_bundle.iterdir()
                if g.is_file()}
        for f in sorted(copy.iterdir()):
            kind = input_kind(video_id, f.name) if f.is_file() else None
            if kind is None or kind in held or (deep_bundle / f.name).exists():
                continue
            try:
                _safe_move(f, deep_bundle / f.name)
                carried.append(f.name)
                held.add(kind)
            except OSError as e:
                logger.warning(f"Routing {video_id}: could not move {f.name} from the "
                               f"triage copy into the deep-review bundle ({e}); it "
                               f"stays in {copy}")
        live = {g.name: g for g in deep_bundle.iterdir() if g.is_file()}
    except OSError as e:
        says = (f"{video_id} is in deep review; its triage folder {copy} or the "
                f"deep-review bundle could not be read ({e}), so the triage folder is "
                f"left where it is. The watcher tries again on its next check.")
        logger.error(f"Routing {video_id}: {says}")
        note_folder_in_the_way(video_id, copy, says)
        return
    if carried:
        logger.info(f"Routing {video_id}: carried {', '.join(carried)} from the triage "
                    f"copy into the deep-review bundle {deep_bundle}")
    clear_leftover(
        copy, video_id, live,
        why=(f"the video moved into deep review ({deep_bundle}), which supersedes "
             f"triage"),
        what_to_do=(f"the video's live bundle is {deep_bundle}; its cuts are being "
                    f"redone, so the answer files here (review, ground truth, cuts, "
                    f"clear marker) are kept only as a record and are not moved into "
                    f"it. (The review's copy in the durable review store is still "
                    f"matched to the new cuts by frame span.) Every reader ignores this "
                    f"folder while the video is in deep review."))


def route_deep_review(
    video_id: str, processing_dir: Path, reason: str, db=None,
    extra_sources: Optional[Iterable[Path]] = None,
    all_or_nothing: bool = False,
) -> Path:
    """Route a video to DEEP_REVIEW (segmentation failed / QC-critical / escalated)."""
    return route_to_queue(
        video_id, processing_dir, Paths.DEEP_REVIEW, reason,
        db=db, db_state="deep_review", extra_sources=extra_sources,
        all_or_nothing=all_or_nothing,
    )


def run_gate(
    video_id: str,
    processing_dir: Path,
    db=None,
    *,
    qc_verdict: str = "auto_approved",
    mp4_path: Optional[Path] = None,
) -> str:
    """Evaluate the gate and, if the video cannot proceed, route + set DB state.

    Returns the decision string. On ``DECISION_CLEAN`` nothing is moved -- the
    caller runs kinematics. On ``DECISION_TRIAGE`` / ``DECISION_DEEP`` the bundle
    has been moved out of ``processing_dir`` and the caller MUST NOT run
    kinematics or archive.

    ``mp4_path`` is an optional extra source pulled into the bundle if the mp4 is
    not co-located in ``processing_dir``.
    """
    decision, reason, st = evaluate_gate(video_id, processing_dir, qc_verdict)
    extra = [mp4_path] if mp4_path else None

    if decision == DECISION_DEEP:
        route_to_queue(video_id, processing_dir, Paths.DEEP_REVIEW, reason,
                       db=db, db_state="deep_review", extra_sources=extra)
        logger.info(f"Gate: {video_id} -> DEEP_REVIEW ({reason})")
    elif decision == DECISION_TRIAGE:
        # The step the questions are about: the causal reach when reach
        # assignment could not run (its answer is all that is missing), else
        # the pellet outcomes and causal reaches (resume_point says where each
        # resumes).
        step = "assignment" if _assignment_failure(video_id, processing_dir) else "outcome"
        bundle = route_to_queue(video_id, processing_dir, Paths.TRIAGE_REVIEW, reason,
                                db=db, db_state="triage", extra_sources=extra,
                                held_at_step=step)
        if _same_folder(Path(bundle).parent, getattr(Paths, "DEEP_REVIEW", None)):
            # Refused: the video is in deep review, which supersedes triage.
            # This run's files must not stay here as a second place the video
            # is in (discard_run_outputs).
            logger.info(f"Gate: {video_id} -> stays in DEEP_REVIEW (triage refused: "
                        f"the video is in deep review)")
            discard_run_outputs(video_id, Path(processing_dir), Path(bundle))
        else:
            logger.info(f"Gate: {video_id} -> TRIAGE ({reason})")
    elif decision == DECISION_STALE:
        # Do NOT route -- a reviewer must not judge data describing old cuts
        # -- and do NOT touch state or files. The video stays 'processing'
        # with its files in place, so the work loop re-picks it and a full
        # rerun recomputes downstream from the (possibly hand-corrected)
        # LOCAL segmentation. Deliberately not marked 'outdated': the
        # reprocess path copies from the ARCHIVE, which can hold the old
        # algo segmentation and would overwrite the human fix.
        logger.warning(f"Gate: {video_id} -> STALE ({reason}) -- not routed; "
                       f"left in Processing for a full recompute")
    else:
        logger.info(f"Gate: {video_id} -> CLEAN (proceed to kinematics)")
    return decision
