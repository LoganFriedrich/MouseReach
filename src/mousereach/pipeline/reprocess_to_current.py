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
   finalized, never filed uncompared. That refusal is only in the returned
   summary's ``error`` (and in reprocess_batch's ``results``): nothing is
   printed or moved, and the video is put on no attention list or queue, so
   whoever runs this must read the summaries. A video whose results match
   another tray's hand scores (hand_scores.label_mismatch) is refused the
   same way: not finalized and not routed to triage.
 - A held video goes into its queue through the watcher's own router
   (review_gate.route_to_queue): deep review beats triage, a bundle already
   there is never overwritten, the move is recorded in this machine's watcher
   database, and a crystallized video is refused.
 - finalize needs this machine's watcher database: it refuses a crystallized
   video and a video the watcher is working on (any state but 'archived'), and
   refuses when the database cannot be read. Run it on the processing server.
 - A clean video is finalized only with its features file; nothing is pushed
   to any database (mousedb pulls filed results itself).
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
        return {"held": False, "error": f"refused: {e}"}
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
                      require_record: bool = False) -> Optional[str]:
    """Why finalize must not touch this video's filed results, or None.
    Reads this machine's watcher database (never creates it).

    Also the review-bundle stager's check (review/staging.stage_video), with
    its own words (``action``, ``tool``) and ``require_record`` True: a
    bundle put into a review queue must have a watcher record to hold its
    queue state, so a video the watcher does not know is refused there."""
    from mousereach.watcher.db import crystallized_refusal
    from mousereach.watcher.db_location import open_watcher_db_if_present
    db, why = open_watcher_db_if_present()
    if db is None:
        return (f"cannot tell whether it is crystallized or being worked ({why}); "
                f"run {tool} on the machine whose watcher database holds it")
    locked = crystallized_refusal(db, video_id, action)
    if locked:
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
        return f"the watcher is working on it (state '{state}'); it is not touched"
    return None


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
    individual step past setup -- failures are captured in the summary."""
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
    if meta.is_file() and not work_meta.exists():
        safe_copy(meta, work_meta, verify=True)

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
            summary["steps"]["kinematics"] = {"ok": False, "err": str(e)}
            logger.warning("kinematics failed for %s: %s", video_id, e)

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
    refusal = _finalize_refusal(video_id)
    if refusal:
        summary["error"] = f"not finalized: {refusal}"
        return summary
    if decision != DECISION_CLEAN:
        if decision == DECISION_TRIAGE:
            # Pellets whose answer a person's re-cut voided are flagged so the
            # triage window walks them (review_gate.flag_recut_voided says why).
            from mousereach.watcher.review_gate import flag_recut_voided
            summary["flagged_recut_voided"] = flag_recut_voided(video_id, work_dir)
        summary["held"] = _stage_review_bundle(
            video_id, work_dir, video_file, dlc_h5, decision, reason)
        return summary

    if video_dir is None:
        summary["error"] = "cannot finalize: video dir unknown"
        return summary
    if not (work_dir / f"{video_id}_features.json").is_file():
        summary["error"] = "cannot finalize: no kinematics were made for this run"
        return summary
    from mousereach.archive.supersede import supersede_video_outputs
    from mousereach.pipeline.fsutil import files_of_video
    new_files = {f.name: f for f in files_of_video(work_dir, video_id)}
    # A person's cuts beside the video would be replaced by this run's
    # algorithm cuts: refused, asked with a DRY RUN before anything moves.
    # WHY first: supersede keeps the person's file but sweeps every other
    # file beside the video into the Archive, so refusing after the real
    # sweep would leave the video with its old results archived and no new ones.
    # (Cannot happen while the person's file is copied into the work folder
    # above; refused rather than overwritten if it ever does.)
    pre = supersede_video_outputs(video_id, video_dir, archive_root,
                                  incoming=new_files, dry_run=True)
    if pre.get("human_cuts_kept") and any(n in new_files for n in pre["human_cuts_kept"]):
        summary["error"] = ("refusing to finalize: a person's cuts beside the video "
                            "would be replaced by the algorithm's (nothing moved)")
        return summary
    sup = supersede_video_outputs(video_id, video_dir, archive_root,
                                  incoming=new_files)
    if sup.get("human_cuts_kept") and any(n in new_files for n in sup["human_cuts_kept"]):
        summary["error"] = ("refusing to finalize: a person's cuts beside the video "
                            "would be replaced by the algorithm's")
        return summary
    summary["superseded"] = {"generation": sup.get("generation"),
                             "stack": sup.get("stack"),
                             "n_pose": len(sup.get("pose", [])),
                             "n_algo": len(sup.get("algo", []))}
    from mousereach.watcher.transfer import safe_move
    moved = []
    # files_of_video: the shared rule (not another video whose stem begins
    # with this one; never an unfinished copy).
    from mousereach.pipeline.fsutil import files_of_video
    for f in sorted(files_of_video(work_dir, video_id)):
        if f.name == work_h5.name:
            continue  # the 4.0 pose stays in DLC Model 4/ until relocation phase
        if safe_move(f, video_dir / f.name):
            moved.append(f.name)
    summary["finalized_moved"] = moved
    summary["decision_final"] = decision

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
