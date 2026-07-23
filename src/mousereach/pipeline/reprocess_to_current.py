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
    for h5 in an.rglob(f"{video_id}*resnet101*shuffle3*.h5"):
        if best is None or "DLC Model 4" in str(h5):
            best = h5
    return best


def find_video_file(video_id: str):
    """The source mp4/mkv for a video (its live home in Analyzed), or None."""
    from ..config import Paths
    an = Path(Paths.ANALYZED_OUTPUT) if Paths.ANALYZED_OUTPUT else None
    if not an or not an.exists():
        return None
    for ext in (".mp4", ".mkv"):
        # prefer the cohort dir (not a DLC Model N staging dir)
        hits = [p for p in an.rglob(f"{video_id}{ext}") if "DLC Model" not in str(p)]
        if hits:
            return hits[0]
        any_hit = next(an.rglob(f"{video_id}{ext}"), None)
        if any_hit:
            return any_hit
    return None


def reprocess_video_to_current(
    video_id: str,
    dlc_h5,
    video_file=None,
    work_dir=None,
    *,
    finalize: bool = False,
    archive_root=None,
    sync_db: Optional[bool] = None,
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
    from mousereach.watcher.review_gate import evaluate_gate, DECISION_CLEAN
    from mousereach.watcher.transfer import safe_copy

    if sync_db is None:
        sync_db = finalize  # live finalize syncs current kinematics to connectome.db
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

    # Pose copied read-only into the work dir (never touch DLC Model 4/).
    work_h5 = work_dir / dlc_h5.name
    if not work_h5.exists():
        if not safe_copy(dlc_h5, work_h5, verify=True):
            summary["error"] = "pose copy failed"
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
        _step("segmentation", lambda: seg_single(work_h5, output_dir=work_dir))
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

    # QC triage
    qc_verdict = "auto_approved"
    try:
        tr = triage_video(video_id=video_id, processing_dir=work_dir, h5_path=work_h5)
        qc_verdict = tr.verdict
        tr.save(work_dir / f"{video_id}_triage.json")
    except Exception as e:
        logger.warning("triage failed for %s: %s", video_id, e)
    summary["qc_verdict"] = qc_verdict

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
            feats = FeatureExtractor().extract(work_h5, reach_path, outcome_path,
                                               review_path=review_path)
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

    # --- FINALIZE: supersede old outputs beside the video, move new into place ---
    if video_dir is None:
        summary["error"] = "cannot finalize: video dir unknown"
        return summary
    from mousereach.archive.supersede import supersede_video_outputs
    sup = supersede_video_outputs(video_id, video_dir, archive_root)
    summary["superseded"] = {"generation": sup.get("generation"),
                             "stack": sup.get("stack"),
                             "n_pose": len(sup.get("pose", [])),
                             "n_algo": len(sup.get("algo", []))}
    from mousereach.watcher.transfer import safe_move
    moved = []
    for f in sorted(work_dir.glob(f"{video_id}*")):
        if f.name == work_h5.name:
            continue  # the 4.0 pose stays in DLC Model 4/ until relocation phase
        if safe_move(f, video_dir / f.name):
            moved.append(f.name)
    summary["finalized_moved"] = moved
    summary["decision_final"] = decision

    # Clean video -> sync the current kinematics into connectome.db. The sync does
    # an atomic per-video replace (DELETE old rows for this video, INSERT new), so
    # re-syncing a reprocessed video swaps its old-version rows for the current
    # ones -- no duplicates, no version blending.
    if sync_db and decision == DECISION_CLEAN and f"{video_id}_features.json" in moved:
        try:
            from mousereach.sync.database import sync_file_to_database
            summary["db_synced"] = bool(
                sync_file_to_database(video_dir / f"{video_id}_features.json"))
        except Exception as e:
            summary["db_synced"] = False
            logger.warning("db sync failed for %s: %s", video_id, e)
    return summary


def build_reprocess_worklist(limit: Optional[int] = None):
    """``[(video_id, pose_h5, video_file), ...]`` for videos that have a current
    (Model 4.0) pose whose algo outputs still need to be brought current. Scans the
    ``DLC Model 4`` tree (read-only). One entry per video."""
    from ..config import Paths
    an = Path(Paths.ANALYZED_OUTPUT) if Paths.ANALYZED_OUTPUT else None
    if not an or not an.exists():
        return []
    dlc4 = an / "Connectome" / "DLC Model 4"
    root = dlc4 if dlc4.exists() else an
    work = []
    seen = set()
    for h5 in root.rglob("*resnet101*shuffle3*.h5"):
        vid = h5.name.split("DLC")[0]
        if vid in seen:
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
