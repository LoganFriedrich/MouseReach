"""The pipeline's human-review GATE + routing.

Called AFTER algo-4 (reach assignment), BEFORE kinematics. Decides a freshly
(re)processed video's fate and, when it cannot proceed on its own, MOVES the
whole bundle into the right review queue so kinematics + the central database never
touch it:

    segmentation failed (hard or soft) OR QC-critical -> DEEP_REVIEW
        (needs the causal/GT deep tools; clearing re-injects at the pipeline start)
    any triaged element still UNRESOLVED              -> TRIAGE
        (fast per-element review tool; resolving re-runs the video through here)
    clean / every triaged element already resolved    -> proceed to kinematics

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


def _write_review_manifest(bundle: Path, video_id: str, reason: str) -> None:
    """Write the ``manifest.json`` the review tool needs to open a self-contained
    bundle -- BUNDLE-LOCAL canonical pointers so the video + pose load in place.
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
    supersede_deep_review_clear(video_id, bundle)
    if db is None:
        logger.warning(f"Escalate {video_id}: no watcher database given; the bundle "
                       f"moves to deep review but the row keeps its state until "
                       f"the video returns")
    return route_to_queue(video_id, Path(bundle), Paths.DEEP_REVIEW,
                          reason=reason, db=db, db_state="deep_review")


class AlreadyInQueueError(RuntimeError):
    """Raised by route_to_queue, before anything moves, when the destination
    queue already holds the video's LIVE bundle (the video is held there) and
    the source is somewhere else. See route_to_queue for why nothing is
    merged, and _clear_the_way for the folders that are not a live bundle."""


# Watcher states in which a bundle in a queue is the video's live copy.
HELD_STATES = ("triage", "deep_review")


def _clear_the_way(video_id: str, existing: Path, processing_dir, reason: str,
                   db, extra_sources: Optional[Iterable[Path]] = None) -> None:
    """Deal with a folder ``existing`` = <queue>/<video> met by a route from
    somewhere else, BEFORE anything moves. Returns when the route may go on;
    raises AlreadyInQueueError when it may not.

      * empty: removed (a return whose last step failed leaves one; if it
        cannot be removed the route fills it -- there is nothing to replace);
      * only queue records (manifest, routing record): not a bundle -- taken
        out of the queue (review_routing.set_aside_queue_folder);
      * a file of the video, and the video is HELD IN THIS QUEUE (watcher
        state 'triage' for the triage queue, 'deep_review' for the deep-review
        queue, or no database to ask): the folder is the live copy.
        AlreadyInQueueError, and the refusal is listed for a person in the
        attention panel (review_routing.note_folder_in_the_way);
      * a file of the video, and the video is NOT held in this queue (it has
        been analysed again since -- this route is that newer run -- or it is
        held in the OTHER queue): the folder is an old copy. Its review goes to
        the durable review store first; then it is taken out of the queue
        whole -- kept for a person, and listed in the attention panel, when a
        person wrote anything in it or it holds a video or pose file this
        route does not bring (review_routing.inputs_the_live_copy_lacks) --
        and the route goes on.

    WHY: refusing every existing folder would fail later runs of a video for
    good -- an empty residue, a records-only folder, or an old bundle someone
    had written in would make the gate raise on every run. Merging into the
    folder (what routes used to do) replaced a held run's files and a
    person's saved review. The state is matched to THIS queue because a
    video held in triage can have a stale deep-review folder (and the other
    way round); refusing on the other queue's state blocked a person's
    Escalate until someone cleaned up by hand."""
    from .review_routing import (_bundle_files, inputs_the_live_copy_lacks,
                                 is_queue_record, note_folder_in_the_way,
                                 set_aside_queue_folder)
    from .review_return import _claim_return_nested, _release_return_claim

    def _refuse(why: str, what_to_do: str):
        says = (f"{video_id} could not be routed to {existing.parent.name} "
                f"({reason}): {why}. Nothing was moved from {processing_dir}. "
                f"{what_to_do}")
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
    if not records_only:
        state = None
        if db is not None:
            try:
                state = (db.get_video(video_id) or {}).get("state")
            except Exception:
                state = None
        if _same_folder(existing.parent, getattr(Paths, "TRIAGE_REVIEW", None)):
            live_states = ("triage",)
        elif _same_folder(existing.parent, getattr(Paths, "DEEP_REVIEW", None)):
            live_states = ("deep_review",)
        else:
            live_states = HELD_STATES           # a queue not configured here
        if db is None or state in live_states:
            _refuse(f"the queue already holds its live bundle {existing} (watcher "
                    f"state '{state or 'unknown'}')",
                    "A video is in one place at a time: decide which copy is the "
                    "live one, and move or remove the other by hand.")
    claimed, release = _claim_return_nested(video_id)
    if not claimed:
        _refuse(f"another machine is moving the folder {existing} right now",
                "It is routed on its next run once that move is done.")
    try:
        what = "queue records only" if records_only else "an older copy of the video"
        # A review leaving the queue keeps a copy where every reader looks, as
        # in every other set-aside (_retire_stale_bundle, _drop_triage_copy):
        # the new bundle has no review, and without the durable copy the
        # person would be asked again.
        from .review_return import _ensure_durable_review
        _ensure_durable_review(existing, video_id)
        live: dict = {}
        try:
            for f in _bundle_files(Path(processing_dir), video_id):
                live[f.name] = f
        except OSError:
            pass
        for f in extra_sources or []:
            if Path(f).is_file():
                live.setdefault(Path(f).name, Path(f))
        try:
            lacking = inputs_the_live_copy_lacks(existing, video_id, live)
        except OSError:
            lacking = []
        dest = set_aside_queue_folder(
            existing, video_id,
            why=(f"{what} was in the queue when a later run of the video was routed "
                 f"there ({reason}); a video is in one place at a time"
                 + (f". It holds {', '.join(lacking)}, which the new route does not "
                    f"bring" if lacking else "")),
            what_to_do=(f"the live bundle is {existing}. Move into it anything here "
                        f"that a person still needs (a review or ground truth "
                        f"re-attaches to the new cuts by frames; a video or pose file "
                        f"the live bundle lacks)."),
            needs_person=lacking)
        logger.warning(f"Routing {video_id}: {existing} was not the live bundle "
                       f"({what}); set aside to {dest} before routing")
    except OSError as e:
        _refuse(f"the old folder {existing} is in the way and could not be set "
                f"aside ({e})",
                "Close anything that has a file in it open; the video is routed on "
                "its next run.")
    finally:
        if release:
            _release_return_claim(video_id)


def route_to_queue(
    video_id: str,
    processing_dir: Path,
    dest_root: Path,
    reason: str,
    db=None,
    db_state: Optional[str] = None,
    extra_sources: Optional[Iterable[Path]] = None,
    all_or_nothing: bool = False,
) -> Path:
    """Move a video bundle into a review queue, write the review manifest, and
    set the DB hold state. Returns the bundle dir.

    ``all_or_nothing`` True (routes out of the FILED folder): if any file
    cannot be moved, everything is put back and
    review_routing.RouteIncompleteError is raised, with nothing routed and no
    state changed (review_routing.move_video_bundle says why).

    Raises CrystallizedError, before anything moves, when ``db`` says the
    video is crystallized (publication-locked; db.crystallized_refusal).

    DEEP REVIEW SUPERSEDES TRIAGE (review_routing, block comment above
    deep_review_stems):
      * into triage while the video has a deep-review bundle: nothing is moved
        or created; the video stays in deep review, its row is set to
        'deep_review', and the deep-review bundle is returned. Files at the
        source stay where they are (logged).
      * into deep review: once the deep-review bundle is complete (no file
        failed to move), the video's triage bundle, if any, is emptied
        without losing a person's file (_drop_triage_copy).

    A VIDEO IS NEVER MERGED INTO A FOLDER IT ALREADY HAS IN THAT QUEUE:
      * the source IS that folder (routed onto itself): nothing moves, only
        the database state is recorded (see below);
      * the source is anywhere else (Processing, the filed folder, the other
        queue): _clear_the_way decides, before anything moves -- a folder that
        is not the live bundle (empty, records only, or an old copy of a video
        that is no longer held) is taken out of the queue and the route goes
        on; the live bundle of a held video raises AlreadyInQueueError, listed
        for a person. WHY: the move used to pour the source's files into the
        existing bundle, replacing same-named files -- an older filed
        generation replaced the held run's results, and a review a person had
        saved in the bundle could be replaced by an older one."""
    from .db import CrystallizedError, crystallized_refusal
    from .review_routing import in_deep_review
    locked = crystallized_refusal(db, video_id, "routed to a review queue")
    if locked:
        raise CrystallizedError(locked)

    def _record_state(state: str, why: str) -> None:
        if db is None or not state:
            return
        try:
            db.update_state(video_id, state)
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
                    reason=f"{why}; normal transition rejected ({e})")
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

    bundle, moved = move_video_bundle(
        video_id, processing_dir, dest_root, reason,
        extra_sources=extra_sources,
        extra_manifest={"db_state": db_state} if db_state else None,
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
        _write_review_manifest(bundle, video_id, reason)
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
        deep-review bundle holds the same file byte for byte; otherwise it is
        set aside like an input below. WHY: "not an input, so an output of
        this run" counted a person's files as outputs and deleted them;
      * an input (the video or a pose file) is deleted only when the
        deep-review bundle holds the same file (same name and content); one
        that could be the only copy is set aside, never deleted, to
        ``<processing root>/_finished_elsewhere/<date>/<folder>/`` -- the set-
        aside folder the watcher already uses for local copies of videos held
        or finished elsewhere.
    Returns ``{"deleted": [...], "set_aside": [...], "failed": [...]}`` and
    logs what it did. Never raises."""
    import shutil
    from datetime import date as _date
    from mousereach.archive.core import _same_file
    from .review_routing import _bundle_files, _same_content, is_person_file
    out = {"deleted": [], "set_aside": [], "failed": []}
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
                deletable = twin.is_file() and _same_content(f, twin)
            elif _is_input_file(video_id, f.name):
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
        f"{processing_dir}: {len(out['deleted'])} deleted (outputs, or inputs the "
        f"deep-review bundle holds identically), {len(out['set_aside'])} set aside "
        f"to {aside} (inputs or a person's files that could be the only copy)"
        + (f"; could not remove: {', '.join(out['failed'])}" if out["failed"] else ""))
    return out


def _drop_triage_copy(video_id: str, deep_bundle: Path) -> None:
    """Empty ``<triage>/<video>`` once the video's deep-review bundle is
    complete (deep review supersedes triage), without losing anything a
    person wrote, and take what is left out of the triage queue. Never raises.

    Never touched while the deep-review bundle's routing record lists a file
    that failed to move (those files may still be in the triage folder, and
    may be the video's only copy): it is listed for a person in the attention
    panel, and the return scan empties it once the deep-review bundle is whole
    (review_return.repair_unfinished_routes). Otherwise:
      * the triage copy's review goes to the durable review store first;
      * a review or ground-truth file the deep-review bundle does not have is
        MOVED into it (it re-attaches to the new cuts by frames, as an
        escalated review does);
      * the video, or a pose file, when the deep-review bundle has no file of
        that kind at all (review_routing.input_kind), is MOVED into it: it may
        be the video's only copy, and the live bundle needs it;
      * machine outputs, and any file the deep-review bundle holds byte for
        byte, are deleted;
      * anything else -- a different review or ground truth than the
        deep-review bundle's, a clear marker (moved into deep review it would
        release the video at once), a hand-set segmentation, a video or pose
        file that differs from the deep-review bundle's -- stays in the
        folder, and the folder is taken out of the triage queue whole to
        <review root>/_Problematic/_kept_for_a_person/ with a note, listed in
        the attention panel (review_routing.set_aside_queue_folder). A pose of
        an older model, beside the deep-review bundle's own pose, needs no
        person: such a remainder goes to the superseded-folders archive.
    The folder is removed when nothing is left in it.
    WHY: the whole folder used to be deleted (shutil.rmtree), a person's
    ground truth and clear markers with it; only reviews had a durable copy.
    And a remainder left IN the triage queue was a second place the video was
    in: once the video left deep review, every later triage route met it."""
    from .review_routing import (read_routing_manifest, input_kind,
                                 inputs_the_live_copy_lacks, is_person_file,
                                 _same_content, _safe_move,
                                 note_folder_in_the_way, set_aside_queue_folder)
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
                f"listed, move the files named in the bundle's routing record in by "
                f"hand.")
        logger.error(f"Routing {video_id}: {says}")
        note_folder_in_the_way(video_id, copy, says)
        return
    try:
        from .review_return import _ensure_durable_review
        _ensure_durable_review(copy, video_id)
    except Exception as e:
        logger.warning(f"Routing {video_id}: could not copy the triage review to the "
                       f"durable store first: {e}")
    deep_bundle = Path(deep_bundle)
    carried, deleted, kept = [], [], []
    try:
        entries = sorted(copy.iterdir())
    except OSError as e:
        says = (f"{video_id} is in deep review; its triage folder {copy} could not "
                f"be listed ({e}) and is left where it is -- move what a person "
                f"needs into {deep_bundle}, then delete the folder.")
        logger.error(f"Routing {video_id}: {says}")
        note_folder_in_the_way(video_id, copy, says)
        return
    for f in entries:
        try:
            if not f.is_file():
                kept.append(f.name)
                continue
            twin = deep_bundle / f.name
            name = f.name
            if twin.is_file() and _same_content(f, twin):
                f.unlink()
                deleted.append(name)
            elif is_person_file(f, video_id):
                carry = (name == f"{video_id}_causal_review.json"
                         or "ground_truth" in name)
                if carry and not twin.exists():
                    _safe_move(f, twin)
                    carried.append(name)
                else:
                    kept.append(name)
            elif input_kind(video_id, name) is not None:
                kind = input_kind(video_id, name)
                held = {input_kind(video_id, g.name) for g in deep_bundle.iterdir()
                        if g.is_file()}
                if kind not in held and not twin.exists():
                    _safe_move(f, twin)
                    carried.append(name)
                else:
                    kept.append(name)
            else:
                f.unlink()              # a machine output of an older run
                deleted.append(name)
        except OSError as e:
            kept.append(f"{f.name} ({e})")
    if carried:
        logger.info(f"Routing {video_id}: carried {', '.join(carried)} from the triage "
                    f"copy into the deep-review bundle {deep_bundle}")
    try:
        left = any(copy.iterdir())
    except OSError:
        left = True
    if not left:
        try:
            copy.rmdir()
            logger.info(f"Routing {video_id}: triage copy {copy} removed -- the video "
                        f"is in deep review, which supersedes triage")
            return
        except OSError:
            pass                        # set aside below, like a remainder
    try:
        live = {g.name: g for g in deep_bundle.iterdir() if g.is_file()}
        lacking = inputs_the_live_copy_lacks(copy, video_id, live)
    except OSError:
        lacking = [n for n in kept if input_kind(video_id, n) is not None]
    try:
        set_aside_queue_folder(
            copy, video_id,
            why=(f"the video moved into deep review (which supersedes triage) and "
                 f"its triage copy still held {', '.join(kept) or 'a folder'} -- a "
                 f"person's file, or a video or pose file that differs from the "
                 f"deep-review bundle's"),
            what_to_do=(f"the video's live bundle is {deep_bundle}. Move into it "
                        f"anything here a person still needs (a different review, "
                        f"ground truth, video or pose: decide which one is right)."),
            needs_person=lacking)
    except OSError as e:
        says = (f"{video_id} is in deep review; its triage folder {copy} still holds "
                f"{', '.join(kept)} and could not be taken out of the triage queue "
                f"({e}). Move what a person needs into {deep_bundle}, then delete "
                f"the folder.")
        logger.error(f"Routing {video_id}: {says}")
        note_folder_in_the_way(video_id, copy, says)


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
        bundle = route_to_queue(video_id, processing_dir, Paths.TRIAGE_REVIEW, reason,
                                db=db, db_state="triage", extra_sources=extra)
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
