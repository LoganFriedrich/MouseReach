"""
mousereach.pipeline.reprocess_to_current -- bring ONE already-DLC'd video up to the
current algo stack, non-destructively.

The corpus was DLC'd with Model 4.0 (pose landing in ``Analyzed/.../DLC Model 4/``)
but its algo outputs are still the old 3.1 generation. This runs the current algos
(seg -> reach -> outcome[v6] -> assign) on the 4.0 pose, QC-triages, and runs the
human-review GATE. If the video is CLEAN it extracts kinematics (applying any human
review). Old outputs are archived (never overwritten) and the new current outputs
take their place next to the video.

Safety / storage model:
 - The 4.0 pose is read READ-ONLY from ``DLC Model 4/`` (copied to a work dir) --
   never moved/renamed, so the running DLC bulk job is undisturbed.
 - All algo work happens in a work dir. Nothing live is touched unless
   ``finalize=True``.
 - On finalize, the OLD-version outputs beside the video are superseded
   (checksum-moved) to the Archive; GT/review/the video stay put.

``finalize=False`` (default) is a safe DRY run: everything lands in the work dir and
the gate DECISION is reported, but the live tree is not modified.

The same rules as the watcher. WHY: this routine files into
Analyzed and holds videos in the review queues, so it must keep the rules every
other path keeps, or a video it touches ends up in two places, filed unchecked,
or with a person's work lost:
 - A person's cuts and clear marker beside the video are copied into the work
   folder before any step runs; the segmenter keeps a person's cuts, and
   finalize never replaces them with the algorithm's (archive/supersede.py).
 - The quality check fails CLOSED: if it cannot run, nothing is finalized.
 - The hand-score comparison runs before the gate, as in the watcher
   (watcher/hand_scores.py): a disagreeing pellet is flagged so the gate holds
   the video in triage; a session whose hand scores cannot be found is NOT
   finalized, never filed uncompared. A video whose results match another
   tray's hand scores (hand_scores.label_mismatch) is refused the same way:
   not finalized and not routed to triage. Nothing is moved.
 - Every stop is shown to a person (_show_the_stop): it is the summary's
   ``error``, it is printed, and with ``finalize=True`` it puts the video on
   this machine's attention list (<NAS root>/attention/<host>_bring_current_
   held.json, the Pipeline Dashboard's red panel); filing the video, or
   holding it in a review queue, takes it off again. That includes a step
   that raised: it is shown, then raised on. Printed but not listed
   (_NOT_LISTED says why): a crystallized video, one the watcher is working
   on, a machine with no watcher database, and older files left beside a
   generation that was filed.
 - A held video goes into its queue through the watcher's own router
   (review_gate.route_to_queue): deep review beats triage, a bundle already
   there is never overwritten, the move is recorded in this machine's watcher
   database, and a crystallized video is refused.
 - finalize needs this machine's watcher database: it refuses a crystallized
   video and a video the watcher is working on (any state but 'archived'), and
   refuses when the database cannot be read. Run it on the processing server.
 - A clean video is finalized only with its features file, and not when that
   file still holds an undecided pellet or a reach whose measures failed (the
   watcher's own last checks, archive/core.filed_features_refusal); nothing
   is pushed to any database (the database tool pulls filed results itself).
 - Finalize files through the watcher's own filing (archive/core.
   archive_video), so it is one step as far as a reader can tell: the new
   files are staged under temporary names, the files they replace are kept
   as verified copies in the versioned Archive, and renames put the new
   generation in place with the features file last; a failure puts the
   previous generation back as it was. It needs this run's own provenance
   manifest and refuses a video that is not in the folder the filing rules
   name. Any failure is the summary's ``error`` ("not filed: ..."), never a
   success, and is shown as every stop is. A copy left in this run's work
   folder after every file was filed is not a failure: it is recorded under
   ``work_copy_left`` and printed. Nothing retries a failure.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def find_current_pose(video_id: str):
    """The current (Model 4.0) pose h5 for a video, or None. Prefers the
    ``DLC Model 4`` tree."""
    from ..config import Paths
    an = Path(Paths.ANALYZED_OUTPUT) if Paths.ANALYZED_OUTPUT else None
    if not an or not an.exists():
        return None
    best = None
    # WHY iter_files, not rglob: a superseded pose under Analyzed/Archive/ (e.g.
    # Archive/DLC Model 4.0/...) keeps its name AND its path contains
    # "DLC Model 4", so the substring preference below would pick it over the
    # live one. Never entering Archive is the fix; DLC Model <N> stays visible.
    from .analyzed_tree import iter_files
    for h5 in iter_files(an, f"{video_id}*resnet101*shuffle3*.h5"):
        if best is None or "DLC Model 4" in str(h5):
            best = h5
    return best


def find_video_file(video_id: str):
    """The source mp4/mkv for a video (its live home in Analyzed), or None."""
    from ..config import Paths
    an = Path(Paths.ANALYZED_OUTPUT) if Paths.ANALYZED_OUTPUT else None
    if not an or not an.exists():
        return None
    # WHY iter_files/first_file, not rglob: a superseded copy under
    # Analyzed/Archive/ keeps its name; finalize would land current outputs
    # beside that archived video instead of the live one.
    from .analyzed_tree import iter_files, first_file
    for ext in (".mp4", ".mkv"):
        # prefer the cohort dir (not a DLC Model N staging dir)
        hits = [p for p in iter_files(an, f"{video_id}{ext}") if "DLC Model" not in str(p)]
        if hits:
            return hits[0]
        any_hit = first_file(an, f"{video_id}{ext}")
        if any_hit:
            return any_hit
    return None


def _stage_review_bundle(video_id, work_dir, video_file, dlc_h5, decision, reason):
    """HOLD a triaged / deep-review video: move its work-folder files into the
    right review queue through the watcher's own router
    (review_gate.route_to_queue), with the move recorded in this machine's
    watcher database. Does NOT finalize / supersede -- the filed outputs stay
    put until the video clears review and is filed again. Returns a summary.

    WHY the router: this used to build ``<queue>/<video>`` itself -- no
    ``.incoming`` build folder, no check for a deep-review bundle before a
    triage one, no watcher record, and a never-overwrite check of the one
    queue only. route_to_queue keeps every queue rule: deep review beats
    triage, a live bundle is never merged into (AlreadyInQueueError), and a
    crystallized video is refused (CrystallizedError). The filed video is
    never passed along: the review tools find it in the filed folder."""
    from mousereach.config import Paths
    from mousereach.watcher.review_gate import (route_to_queue, AlreadyInQueueError,
                                                DECISION_DEEP)
    from mousereach.watcher.db import CrystallizedError
    from mousereach.watcher.db_location import open_watcher_db_if_present
    dest_root = Paths.DEEP_REVIEW if decision == DECISION_DEEP else Paths.TRIAGE_REVIEW
    if not dest_root:
        return {"held": False, "error": "no review-queue root configured"}
    db, why_no_db = open_watcher_db_if_present()
    report: Dict = {}
    try:
        bundle = route_to_queue(video_id, Path(work_dir), Path(dest_root),
                                f"bring-current: {reason}", db=db, db_state=decision,
                                report=report)
    except CrystallizedError as e:
        return {"held": False, "crystallized": True, "error": f"refused: {e}"}
    except AlreadyInQueueError as e:
        return {"held": True, "note": f"already in a review queue -- left intact ({e})"}
    except Exception as e:
        return {"held": False, "error": f"could not route to {Path(dest_root).name}: {e}"}
    out = {"held": True, "bundle": str(bundle), "queue": Path(dest_root).name,
           "state_recorded": bool(report.get("state_recorded"))}
    if not out["state_recorded"]:
        out["state_note"] = report.get("state_note") or why_no_db
    return out


def _finalize_refusal(video_id: str, action: str = "brought current",
                      tool: str = "bring-current",
                      require_record: bool = False,
                      why_out: Optional[Dict] = None) -> Optional[str]:
    """Why finalize must not touch this video's filed results, or None.
    Reads this machine's watcher database (never creates it).

    ``why_out``, when given, gets ``"not_a_defect"`` set to "crystallized",
    "watcher" or "no_db" for a refusal that is no fault for a person to fix:
    a crystallized video (the lock is deliberate), one the watcher is working
    on (it is the watcher's to finish, and the watcher shows its own stops),
    or no watcher database here (the run belongs on another machine).
    Bring-current prints those with what that means, and does not list them
    for a person (_NOT_LISTED says why).

    Also the review-bundle stager's check (review/staging.stage_video), with
    its own words (``action``, ``tool``) and ``require_record`` True: a
    bundle put into a review queue must have a watcher record to hold its
    queue state, so a video the watcher does not know is refused there."""
    from mousereach.watcher.db import crystallized_refusal
    from mousereach.watcher.db_location import open_watcher_db_if_present
    db, why = open_watcher_db_if_present()
    if db is None:
        if why_out is not None:
            why_out["not_a_defect"] = "no_db"
        return (f"cannot tell whether it is crystallized or being worked ({why}); "
                f"run {tool} on the machine whose watcher database holds it")
    locked = crystallized_refusal(db, video_id, action)
    if locked:
        if why_out is not None:
            why_out["not_a_defect"] = "crystallized"
        return locked
    try:
        row = db.get_video(video_id)
    except Exception as e:
        return f"its watcher record could not be read ({e})"
    if row is None and require_record:
        return ("the watcher database on this machine has no record of it, so its "
                "queue state could not be recorded")
    state = (row or {}).get("state")
    if state not in (None, "archived"):
        if why_out is not None:
            why_out["not_a_defect"] = "watcher"
        return f"the watcher is working on it (state '{state}'); it is not touched"
    return None


# The attention list a bring-current video that needs a person goes on
# (<NAS root>/attention/<host>_bring_current_held.json; watcher/attention.py
# says why it is a list of its own). Re-exported here for callers that named
# it here.
from mousereach.watcher.attention import BRING_CURRENT_HELD_KIND  # noqa: E402,F401


def _attention_note(video_id: str, says: str) -> bool:
    """List ``video_id`` on this machine's bring-current attention list, so a
    person sees it on the Pipeline Dashboard's red panel without reading the
    returned summary. True only when the list was written (note_video says
    False when no NAS root is set or the write failed). Never raises."""
    try:
        import socket
        from mousereach.config import Paths
        from mousereach.watcher.attention import note_video
        return bool(note_video(getattr(Paths, "NAS_ROOT", None),
                               socket.gethostname(),
                               BRING_CURRENT_HELD_KIND, video_id, says))
    except Exception as e:      # note_video never raises; an import might
        logger.warning("could not list %s for attention: %s", video_id, e)
        return False


def _attention_drop(video_id: str) -> None:
    """Take ``video_id`` off the bring-current attention list (it was filed,
    or is held in a review queue where a person sees it). Never raises."""
    try:
        import socket
        from mousereach.config import Paths
        from mousereach.watcher.attention import drop_video
        drop_video(getattr(Paths, "NAS_ROOT", None), socket.gethostname(),
                   BRING_CURRENT_HELD_KIND, video_id)
    except Exception as e:
        logger.warning("could not update the attention list for %s: %s", video_id, e)


# ``notice`` keys the run sets for the one place that shows a stop
# (_show_the_stop): "says" -- the sentence for a person, when the default
# ("<video>: <error>. Fix that, then run bring-current again for it.") is not
# the right instruction; "listed" _NOT_LISTED -- printed only. WHY printed
# only: the list is for what a person must fix, and an item stays on the red
# panel until a filing of the video (archive_video) or a review-queue hold
# clears it. So these are not listed: a crystallized video (the lock is
# deliberate, nothing to fix, and it is never filed, so the item would stay
# for good); a video the watcher is working on (nothing to fix: it is the
# watcher's to finish, and the watcher shows its own stops); a machine with
# no watcher database (nothing to fix about the video: run bring-current on
# the machine that has one; an item here would be on this machine's list,
# which that run never clears); and older files left beside an already-filed
# generation (no later filing clears it: bring-current skips a video whose
# filed results are current; the printed line says what to move).
_NOT_LISTED = False

# What a person is told about a refusal that is not a defect (see above),
# in place of the default "Fix that, then run bring-current again".
_NOT_A_DEFECT_SAYS = {
    "crystallized": "Left as it is: the publication lock is deliberate, and "
                    "bring-current never changes a locked video.",
    "watcher": "Left to the watcher, which is working on it.",
    "no_db": "Nothing is wrong with the video: run bring-current for it on the "
             "machine whose watcher database holds it.",
}


def _show_the_stop(summary: Dict, video_id: str, notice: Dict, finalize: bool) -> None:
    """Every bring-current stop is printed, and with ``finalize`` listed for a
    person unless ``notice`` says it is not to be (see _NOT_LISTED). WHY one
    place: each refusal used to decide for itself, and most only set the
    summary's ``error``, so the video stayed as it was with nobody told unless
    they read the returned dictionary. A dry run touches nothing live, so it
    only prints. ``summary["attention"]`` records what was shown and
    ``summary["listed"]`` whether it went on the list."""
    err = summary.get("error")
    if not err:
        return
    says = notice.get("says") or (f"{video_id}: {err}. Fix that, then run "
                                  f"bring-current again for it.")
    wanted = bool(finalize) and notice.get("listed", True)
    print(f"bring-current: {says}")
    listed = False
    if wanted:
        # WHY the result is read: "listed" must say what was done. A failed
        # write (no NAS root, unwritable list) leaves nobody told on the red
        # panel, so the printed line says so and the summary says False.
        listed = _attention_note(video_id, says)
        if not listed:
            print(f"bring-current: {video_id} could NOT be put on the "
                  f"attention list (no NAS root set, or the list could not "
                  f"be written); this printed line is the only notice.")
    summary["attention"] = says
    summary["listed"] = listed


def reprocess_video_to_current(
    video_id: str,
    dlc_h5,
    video_file=None,
    work_dir=None,
    *,
    finalize: bool = False,
    archive_root=None,
    sync_db: Optional[bool] = None,     # ignored: nothing is pushed (module docstring)
    skip_if_current: bool = True,
) -> Dict:
    """Reprocess one video to the current algo stack. See module docstring.

    ``dlc_h5`` is the current (4.0) pose; ``video_file`` the source mp4 (its dir is
    passed to the v6 cascade's CV stage and is where finalized outputs land). Returns
    a summary with the gate decision, versions, and outputs. Never raises on an
    individual step past setup -- failures are captured in the summary, and
    every one is shown to a person (_show_the_stop)."""
    notice: Dict = {}
    try:
        summary = _reprocess_video_to_current(
            video_id, dlc_h5, video_file, work_dir, finalize=finalize,
            archive_root=archive_root, skip_if_current=skip_if_current, notice=notice)
    except Exception as e:
        # A call that raised (the gate, the filing ...) is a stop like any
        # other: shown to a person, then raised on, so a caller sees what it
        # always saw (reprocess_batch records it in its results).
        _show_the_stop({"video_id": video_id, "error": f"bring-current stopped: {e}"},
                       video_id, {}, finalize)
        raise
    _show_the_stop(summary, video_id, notice, finalize)
    return summary


def _reprocess_video_to_current(video_id, dlc_h5, video_file, work_dir, *,
                                finalize, archive_root, skip_if_current,
                                notice: Dict) -> Dict:
    """The work of reprocess_video_to_current. A stop sets the summary's
    ``error`` and returns; ``notice`` carries how it is to be shown."""
    from mousereach.segmentation.core.batch import process_single as seg_single
    from mousereach.reach.core.batch import process_single as reach_single
    from mousereach.outcomes.core.batch import process_single as outcome_single
    from mousereach.assignment.run import assign_reaches_for_video
    from mousereach.pipeline.manifest import create_processing_manifest
    from mousereach.pipeline.triage import triage_video
    from mousereach.watcher.review_gate import (evaluate_gate, DECISION_CLEAN, DECISION_DEEP,
                                                DECISION_TRIAGE)
    from mousereach.watcher.transfer import safe_copy

    dlc_h5 = Path(dlc_h5)
    video_file = Path(video_file) if video_file else find_video_file(video_id)
    video_dir = video_file.parent if video_file else None

    if work_dir is None:
        import tempfile
        work_dir = Path(tempfile.mkdtemp(prefix=f"reproc_{video_id}_"))
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict = {
        "video_id": video_id, "finalize": finalize, "work_dir": str(work_dir),
        "steps": {}, "decision": None, "versions": {}, "error": None,
    }
    t0 = time.time()

    # Idempotency + the user's rule: if this video's CURRENT Analyzed output is
    # already the current stack, do NOT reprocess it. (The worklist already skips
    # queued/GT videos; this catches anything already finalized-current, e.g. a
    # re-run.)
    if skip_if_current and video_dir is not None:
        try:
            from mousereach.pipeline.versions import (
                get_current_versions, compare_manifest_to_current)
            from mousereach.config import Paths
            man_p = Path(video_dir) / f"{video_id}_processing_manifest.json"
            if man_p.is_file() and man_p.stat().st_size > 0:
                man = json.loads(man_p.read_text(encoding="utf-8"))
                if compare_manifest_to_current(
                        man, get_current_versions(Paths.NAS_ROOT)).get("is_current"):
                    summary["decision"] = "skipped_current"
                    summary["note"] = "already current -- not reprocessed"
                    return summary
        except Exception:
            pass

    # Pose copied read-only into the work dir (never touch DLC Model 4/).
    work_h5 = work_dir / dlc_h5.name
    if not work_h5.exists():
        if not safe_copy(dlc_h5, work_h5, verify=True):
            summary["error"] = "pose copy failed"
            return summary
    # DeepLabCut's record of the video's frame rate goes with the pose, so the
    # kinematics can convert to seconds even when no video file is found
    # (kinematics/core/frame_rate.py; without a rate they are not written).
    meta = dlc_h5.with_name(dlc_h5.stem + "_meta.pickle")
    work_meta = work_dir / meta.name
    meta_copy_failed = None
    if meta.is_file() and not work_meta.exists():
        # Checked, as the pose copy above is. WHY not refused: the video file,
        # when found, also gives the frame rate. But when it is not found the
        # kinematics refuse for "no frame rate", and a person reading that
        # would look for a missing record instead of a copy that failed, so
        # the failure is recorded and named in what the summary reports.
        if not safe_copy(meta, work_meta, verify=True):
            meta_copy_failed = (f"the pose's frame-rate record {meta.name} could not be "
                                f"copied into the work folder")
            summary["pose_record"] = meta_copy_failed
            logger.warning("reprocess %s: %s", video_id, meta_copy_failed)

    # A person's work beside the video goes into the work folder BEFORE any
    # step runs: their cuts (a segments file marked boundary_source "human",
    # which the segmenter keeps instead of re-cutting -- segmentation/core/
    # batch.py) and their clear marker (which the gate reads). WHY: the work
    # folder used to start empty, so the segmenter re-cut over a person's
    # boundaries, the gate asked again for a clear a person had given, and
    # finalize then archived the person's cuts and filed the algorithm's.
    if video_dir is not None:
        from mousereach.archive.supersede import is_human_cuts
        for name in (f"{video_id}_segments.json", f"{video_id}_deep_review_cleared.json"):
            src = Path(video_dir) / name
            if not src.is_file() or (work_dir / name).exists():
                continue
            if name.endswith("_segments.json") and not is_human_cuts(src):
                continue            # the algorithm's own cuts are made again
            if not safe_copy(src, work_dir / name, verify=True):
                summary["error"] = f"could not copy a person's {name} into the work folder"
                return summary

    def _step(name, fn):
        s = time.time()
        try:
            r = fn()
            summary["steps"][name] = {"ok": True, "s": round(time.time() - s, 1)}
            return r
        except Exception as e:
            summary["steps"][name] = {"ok": False, "err": str(e)}
            logger.warning("reprocess %s: %s failed: %s", video_id, name, e)
            raise

    seg_path = work_dir / f"{video_id}_segments.json"
    reach_path = work_dir / f"{video_id}_reaches.json"
    outcome_path = work_dir / f"{video_id}_pellet_outcomes.json"

    try:
        # The filed cuts are what a new segmentation replaces (segmenter_robust.
        # save_segmentation records a re-cut against them), so answers on
        # pellets whose frames move are asked again, never fitted onto the new
        # cuts. The work folder holds no algorithm cuts of its own.
        _filed_cuts = (Path(video_dir) / f"{video_id}_segments.json") if video_dir is not None else None
        if _filed_cuts is not None and not _filed_cuts.is_file():
            _filed_cuts = None
        _step("segmentation", lambda: (
            seg_single(work_h5, output_dir=work_dir, previous_cuts=_filed_cuts)
            if _filed_cuts is not None else seg_single(work_h5, output_dir=work_dir)))
        _step("reach", lambda: reach_single(work_h5, seg_path, output_dir=work_dir))
        _step("outcome", lambda: outcome_single(work_h5, seg_path, reach_path,
                                                output_dir=work_dir, video_dir=video_dir))
        _step("assignment",
              lambda: assign_reaches_for_video(work_dir, video_id, work_h5))
    except Exception as e:
        summary["error"] = f"algo stage failed: {e}"
        return summary

    # provenance manifest (records the versions that just ran)
    try:
        create_processing_manifest(
            video_id=video_id, processing_dir=work_dir, dlc_path=work_h5,
            step_timestamps={"reprocessed_at": datetime.now().isoformat()})
    except Exception as e:
        logger.warning("manifest failed for %s: %s", video_id, e)

    # record the version stamps for reporting
    for label, p, key in (("segmenter", seg_path, "segmenter_version"),
                          ("reach_detector", reach_path, "detector_version"),
                          ("outcome_detector", outcome_path, "detector_version")):
        try:
            summary["versions"][label] = json.loads(p.read_text()).get(key)
        except Exception:
            pass

    # QC triage. Fails CLOSED: a check that could not run is not a pass.
    try:
        tr = triage_video(video_id=video_id, processing_dir=work_dir, h5_path=work_h5)
        qc_verdict = tr.verdict
        tr.save(work_dir / f"{video_id}_triage.json")
    except Exception as e:
        logger.warning("triage failed for %s: %s", video_id, e)
        summary["error"] = f"the quality check could not run ({e}); nothing finalized"
        return summary
    summary["qc_verdict"] = qc_verdict

    # Hand scores, before the gate -- the watcher's order and the watcher's
    # check (watcher/hand_scores.py). A disagreeing pellet is flagged, so the
    # gate below holds the video in triage. A session whose hand scores cannot
    # be found is a defect to fix, never filed uncompared: reported, not
    # finalized.
    from mousereach.watcher import hand_scores
    try:
        verdict = hand_scores.check(video_id, work_dir)
    except hand_scores.PelletNumbersUnknown as e:
        verdict = None
        summary["hand_scores"] = "pellet numbers unknown"
        decision, reason = DECISION_DEEP, f"hand scores: {e}"
    except Exception as e:
        summary["error"] = f"the hand-score comparison could not run ({e}); nothing finalized"
        return summary
    if verdict is not None:
        summary["hand_scores"] = verdict.status
        if verdict.status == hand_scores.WAITING:
            summary["error"] = (f"no hand scores found for this session ({verdict.reason}); "
                                f"held, nothing finalized")
            # R4: hand scores always exist, so a session without them is OUR
            # defect (an import or a sheet-to-video match) and is held
            # VISIBLY: printed, and with finalize on the attention list
            # (Pipeline Dashboard, red panel) -- _show_the_stop. WHY: the hold
            # used to be only in the returned summary, so the video stayed
            # filed with its old results and nobody saw it unless they read
            # the dict.
            notice["says"] = (
                f"{video_id}: {summary['error']}. Its filed results are unchanged. "
                f"Find why its hand scores cannot be found (its rows in the "
                f"tracking sheet, or the hand-scores file this machine reads), "
                f"then run bring-current again for it.")
            return summary
        if verdict.status == hand_scores.LABEL_MISMATCH:
            # The results match another tray's hand scores: the name or the
            # sheet has the trays swapped. No review can fix that, so it is
            # neither routed to triage nor filed (hand_scores.label_mismatch).
            summary["error"] = (f"label mismatch: {verdict.reason}; held, nothing "
                                f"finalized")
            return summary
        if verdict.status in (hand_scores.AGREES, hand_scores.DISAGREES,
                              hand_scores.SUSPENDED, hand_scores.SHEET_ERROR):
            try:
                from mousereach.watcher.route_cli import answered_segments_of
                _answered = set(answered_segments_of(video_id, work_dir))
            except Exception:
                _answered = {"all"}
            hand_scores.clear_bench_flags(work_dir, video_id,
                                          keep=set(verdict.disagreements) | _answered)
        if verdict.status == hand_scores.DISAGREES:
            hand_scores.flag_disagreements(work_dir, video_id, verdict)
        hand_scores.record(work_dir, video_id, verdict, hand_scores.scores_file_stamp())
        # GATE (pure -- decision only, no moves/DB)
        decision, reason, status = evaluate_gate(video_id, work_dir, qc_verdict=qc_verdict)
    summary["decision"] = decision
    summary["gate_reason"] = reason

    # Kinematics only for CLEAN videos (applies any human review).
    if decision == DECISION_CLEAN and reach_path.exists() and outcome_path.exists():
        try:
            from mousereach.kinematics.core.feature_extractor import FeatureExtractor
            from mousereach.review.causal_review_io import resolve_review_path
            review_path = resolve_review_path(video_id, work_dir)
            # The pose was copied alone into the work folder, so the video is named:
            # its frame rate converts frames to seconds (frame_rate.py).
            feats = FeatureExtractor().extract(work_h5, reach_path, outcome_path,
                                               review_path=review_path,
                                               video_path=video_file)
            (work_dir / f"{video_id}_features.json").write_text(
                json.dumps(feats.to_dict(), indent=2))
            # Per-video finalized results CSV -- the reconciled 'ultimate result'
            # that lives with the video (moved into place on finalize).
            try:
                from mousereach.kinematics.analysis.reach_export import write_video_results_csv
                write_video_results_csv(work_dir / f"{video_id}_features.json")
            except Exception as _e:
                logger.warning("results CSV failed for %s: %s", video_id, _e)
            summary["steps"]["kinematics"] = {"ok": True, "n_segments": feats.n_segments}
        except Exception as e:
            err = str(e)
            if meta_copy_failed:
                err = f"{err} ({meta_copy_failed})"
            summary["steps"]["kinematics"] = {"ok": False, "err": err}
            logger.warning("kinematics failed for %s: %s", video_id, err)

    summary["elapsed_s"] = round(time.time() - t0, 1)

    if not finalize:
        summary["note"] = "DRY: nothing live touched; outputs in work_dir"
        return summary

    # --- FINALIZE: obey the gate ---
    # Triaged / deep-review videos must be HELD in the review queue, NOT finalized
    # into Analyzed. Only a CLEAN video supersedes its old outputs + lands next to
    # the video. Neither is done to a crystallized video or one the watcher is
    # working on (_finalize_refusal).
    if decision not in (DECISION_CLEAN, DECISION_DEEP, DECISION_TRIAGE):
        # The gate's "stale outputs" answer (review_gate.DECISION_STALE): the
        # reaches or outcomes were computed against cuts that no longer exist.
        # That is neither a pass nor a review question -- a reviewer must not
        # judge data describing old cuts, and the watcher itself never routes
        # it (review_gate.run_gate). So nothing is filed and nothing is
        # sent to a queue; the summary says why, and the filed results stay.
        summary["error"] = f"not finalized: the gate answered '{decision}' ({reason})"
        return summary
    refusal_kind: Dict = {}
    refusal = _finalize_refusal(video_id, why_out=refusal_kind)
    if refusal:
        summary["error"] = f"not finalized: {refusal}"
        kind = refusal_kind.get("not_a_defect")
        if kind:
            notice["listed"] = _NOT_LISTED
            notice["says"] = f"{video_id}: {summary['error']}. {_NOT_A_DEFECT_SAYS[kind]}"
        return summary
    if decision != DECISION_CLEAN:
        if decision == DECISION_TRIAGE:
            # Pellets whose answer a person's re-cut voided are flagged so the
            # triage window walks them (review_gate.flag_recut_voided says why).
            from mousereach.watcher.review_gate import flag_recut_voided
            summary["flagged_recut_voided"] = flag_recut_voided(video_id, work_dir)
        summary["held"] = _stage_review_bundle(
            video_id, work_dir, video_file, dlc_h5, decision, reason)
        if summary["held"].get("held"):
            _attention_drop(video_id)   # held in a queue, where a person sees it
        else:
            # Neither filed nor in a queue: a stop like any other. WHY: this
            # used to be only inside summary["held"], with no error at all.
            summary["error"] = (f"not held for review ({decision}): "
                                f"{summary['held'].get('error') or 'routing failed'}")
            if summary["held"].get("crystallized"):
                notice["listed"] = _NOT_LISTED
                notice["says"] = (f"{video_id}: {summary['error']}. "
                                  f"{_NOT_A_DEFECT_SAYS['crystallized']}")
        return summary

    if video_dir is None:
        summary["error"] = "cannot finalize: video dir unknown"
        return summary
    if not (work_dir / f"{video_id}_features.json").is_file():
        summary["error"] = "cannot finalize: no kinematics were made for this run"
        if meta_copy_failed:
            summary["error"] += f" ({meta_copy_failed})"
        return summary
    # The watcher's own last checks on the file about to be filed (an
    # undecided pellet; a reach whose measures failed), asked before anything
    # moves. WHY: filing means done for good, and this routine files.
    from mousereach.archive.core import filed_features_refusal
    why = filed_features_refusal(video_id, work_dir)
    if why:
        summary["error"] = f"not finalized: {why} (nothing moved)"
        return summary
    # Filed only with this run's own provenance manifest. WHY: the filing
    # replaces only the files this run brings, so without one the filed
    # folder would keep the previous generation's manifest beside this run's
    # kinematics, and the version scan would read the old versions.
    if not (work_dir / f"{video_id}_processing_manifest.json").is_file():
        summary["error"] = ("cannot finalize: this run's provenance manifest was not "
                            "written (nothing moved)")
        return summary
    from mousereach.archive.core import archive_video, get_archive_destination
    dest = get_archive_destination(video_id)
    try:
        same_folder = Path(dest).resolve() == Path(video_dir).resolve()
    except Exception:
        same_folder = False
    if not same_folder:
        # The watcher's filing (used below) files into the folder the filing
        # rules name; the video found is somewhere else, so filing would leave
        # this run's results in one folder and the video with its old results
        # in another.
        summary["error"] = (f"not finalized: the video is in {video_dir}, not in {dest} "
                            f"where the filing rules file it (nothing moved)")
        return summary

    # FILING IS THE WATCHER'S OWN (archive/core.archive_video): ONE STEP as far
    # as any reader of the filed folder can tell. Every new file is copied
    # into the filed folder under a temporary name first; the files it
    # replaces are copied, checksum-verified, into the versioned Archive; then
    # renames in that one folder put the new files in place, the manifest
    # second to last and the features file last, and a rename that fails is
    # undone so the previous generation stands complete again. It also keeps
    # a person's cuts (refused, nothing moved) and refuses a video with a
    # publication lock record. WHY: bring-current used to move the files
    # itself in name order -- features FIRST -- after archiving the old
    # generation, and dropped a failed move silently, so the filed folder
    # could hold new kinematics without the reaches or segments they were
    # computed from, reported as success.
    #
    # The pose copy stays out of the filing: the current pose lives in its
    # own tree (DLC Model <N>), not beside the video.
    kept_out = work_dir / "_not_filed"
    try:
        kept_out.mkdir(exist_ok=True)
        if work_h5.exists():
            work_h5.replace(kept_out / work_h5.name)
    except OSError as e:
        summary["error"] = f"cannot finalize: could not set the pose copy aside ({e})"
        return summary

    # Files of an older generation beside the video that this run does not
    # replace (an output the current algorithms no longer write). The
    # filing replaces only what this run brings, so these are moved into
    # the Archive after it, under the old manifest's generation and stack.
    from mousereach.archive.supersede import supersede_video_outputs
    from mousereach.pipeline.fsutil import files_of_video
    new_names = {f.name for f in files_of_video(work_dir, video_id)}
    old_manifest = None
    try:
        old_manifest = json.loads((Path(video_dir) / f"{video_id}_processing_manifest.json")
                                  .read_text(encoding="utf-8"))
    except Exception:
        old_manifest = None
    pre = supersede_video_outputs(video_id, video_dir, archive_root,
                                  manifest=old_manifest, dry_run=True)
    leftovers = {n for n in (pre.get("algo", []) + pre.get("pose", []))
                 if n not in new_names}

    filed = archive_video(video_id, verbose=False, skip_ready_check=True,
                          source_dir=work_dir, archive_root=archive_root,
                          update_index=False)
    summary["superseded"] = filed.get("superseded")
    summary["finalized_moved"] = list(filed.get("files_moved") or [])
    if not filed.get("success") and not filed.get("filed"):
        # Not filed: the filing put the previous generation back (or, if it
        # could not, kept a copy of it in the Archive, and its error says so).
        summary["error"] = f"not filed: {filed.get('error') or 'filing failed'}"
        return summary
    if not filed.get("success"):
        # Every new file is in place in the filed folder; only a copy in this
        # run's own work folder could not be removed afterward. That is a
        # filed video, not a failure, and needs nobody: the work folder is
        # bring-current's own. Recorded, printed, and the rest of finalize
        # goes on.
        summary["work_copy_left"] = filed.get("error")
        # files_moved lists the work copies removed, which stopped at the
        # first that could not be; what was filed is every file this run
        # brought.
        summary["finalized_moved"] = sorted(n for n in new_names
                                            if (Path(video_dir) / n).is_file())
        print(f"bring-current: {video_id}: filed; a copy in its work folder "
              f"{work_dir} could not be removed afterward ({filed.get('error')}). "
              f"The filed results are complete.")

    if leftovers:
        sweep = supersede_video_outputs(video_id, video_dir, archive_root,
                                        manifest=old_manifest, only=leftovers)
        if sweep.get("error") or sweep.get("failed"):
            why = sweep.get("error") or ", ".join(sweep["failed"])
            summary["decision_final"] = decision
            summary["error"] = (f"filed, but older files beside it could not be moved "
                                f"into the Archive ({why}); they are not part of this "
                                f"run's results")
            notice["says"] = (f"{video_id}: {summary['error']}. Move them by hand into "
                              f"the versioned Archive.")
            notice["listed"] = _NOT_LISTED
            _attention_drop(video_id)   # filed: an earlier hold is resolved
            return summary
    summary["decision_final"] = decision
    _attention_drop(video_id)
    return summary


def build_reprocess_worklist(limit: Optional[int] = None, *,
                             skip_queued: bool = True, skip_gt: bool = False):
    """``[(video_id, pose_h5, video_file), ...]`` for videos that still need to be
    brought current. Scans the ``DLC Model 4`` tree (read-only), one entry per
    video, and EXCLUDES videos we must not reprocess:

    - already in a review queue (Pending / Deep_Review) -- those were already
      processed with the current stack and are held for human review; reprocessing
      would clobber a reviewer's in-progress work.

    GT'd videos are NOT excluded by default: reprocessing them is safe (the gate
    treats an exhaustively-GT'd video as clean and the truth-resolver makes GT win
    in the kinematics), and building a corpus-wide GT index is slow. Pass
    ``skip_gt=True`` to exclude them anyway.

    (Per-video version-currency is a second guard inside reprocess_video_to_current:
    a video already carrying current Analyzed outputs is skipped there too.)"""
    from ..config import Paths
    an = Path(Paths.ANALYZED_OUTPUT) if Paths.ANALYZED_OUTPUT else None
    if not an or not an.exists():
        return []
    dlc4 = an / "Connectome" / "DLC Model 4"
    scan_root = dlc4 if dlc4.exists() else an

    exclude = set()
    if skip_queued:
        for qroot in (Paths.TRIAGE_REVIEW, Paths.DEEP_REVIEW):
            if qroot and Path(qroot).exists():
                exclude |= {d.name for d in Path(qroot).iterdir() if d.is_dir()}
    if skip_gt:
        try:
            from ..review.causal_review_io import gt_index
            exclude |= set(gt_index().keys())
        except Exception:
            pass

    work = []
    seen = set()
    # WHY iter_files, not rglob: when scan_root falls back to Analyzed itself,
    # rglob would queue superseded poses under Analyzed/Archive/ as work (and
    # "seen" could then shadow the live pose for the same video).
    from .analyzed_tree import iter_files
    for h5 in iter_files(scan_root, "*resnet101*shuffle3*.h5"):
        vid = h5.name.split("DLC")[0]
        if vid in seen or vid in exclude:
            continue
        seen.add(vid)
        work.append((vid, h5, find_video_file(vid)))
        if limit and len(work) >= limit:
            break
    return work


def reprocess_batch(worklist=None, *, finalize: bool = False,
                    limit: Optional[int] = None, progress=None) -> Dict:
    """Reprocess a batch of already-DLC'd videos to the current algo stack.

    ``worklist`` defaults to ``build_reprocess_worklist(limit)``. finalize=False is
    a DRY run (nothing live touched). Returns ``{n, spread, results}`` where spread
    is the gate-decision histogram (clean/triage/deep_review). ``progress(msg)`` is
    called before each video."""
    from collections import Counter
    if worklist is None:
        worklist = build_reprocess_worklist(limit=limit)
    spread: Counter = Counter()
    results = []
    for i, (vid, pose, vf) in enumerate(worklist):
        if progress:
            progress(f"[{i + 1}/{len(worklist)}] {vid}")
        try:
            r = reprocess_video_to_current(vid, pose, video_file=vf, finalize=finalize)
        except Exception as e:
            r = {"video_id": vid, "error": str(e), "decision": None}
        results.append(r)
        spread[r.get("decision")] += 1
    return {"n": len(results), "spread": dict(spread), "results": results}
