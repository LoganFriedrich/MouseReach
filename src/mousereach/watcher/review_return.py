"""Return path: re-inject a human-cleared held video into the pipeline.

The forward gate (review_gate) MOVES un-committable videos into the TRIAGE or
DEEP_REVIEW queue and holds them out of kinematics + connectome.db. This module
is the other half: it scans those queues for bundles a human has cleared and
moves them back into the local Processing dir, set to ``processing`` state, so
the pipeline re-runs them (Priority-2 'pipeline' work). On the re-run the review
file travels with the bundle, so the gate re-checks and -- now that every
triaged element is resolved -- lets the video through to kinematics.

Clear signals:
  TRIAGE      -- every triaged element resolved (``triage_status.fully_resolved``
                 and segmentation not failed). Works with the existing review
                 tool today: resolve all elements -> next scan re-injects.
                 EXCEPTION: if the reviewer set ``true_segment_num`` anywhere
                 (declared a segment mislabeled), the video is DIVERTED to the
                 DEEP_REVIEW queue for manual re-segmentation instead of being
                 re-injected -- re-running with the wrong boundaries would
                 compute kinematics over the wrong frames.
  DEEP_REVIEW -- an explicit ``{stem}_deep_review_cleared.json`` marker (written
                 by the deep tools when a reviewer clears the flag) OR a
                 co-located ``{stem}_unified_ground_truth.json`` (the deep GT
                 tool produced the answer). Re-injection restarts the pipeline
                 from segmentation.

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..config import Paths
from ..review.triage_status import triage_status
from .review_routing import _safe_move

logger = logging.getLogger(__name__)

# Queue-only metadata that must NOT follow the bundle back into Processing.
_QUEUE_METADATA_SUFFIXES = ("_manifest.json", "_routing.json")
_QUEUE_METADATA_NAMES = ("manifest.json",)


def _resolve_inputs(bundle: Path, stem: str):
    """Find the mp4 and pose file for a bundle that is being returned.

    Bundles are staged NOT self-contained: the mp4 and pose normally stay in
    Analyzed and the bundle carries only the algo JSONs plus a ``_manifest.json``
    naming the canonical paths. Returning a bundle without resolving those means
    the pipeline re-runs the video with no pose at all -- which is how 723
    videos ended up in the deep-review queue as "segmentation_failed" on
    2026-08-19.

    Note this must run BEFORE the bundle is emptied: ``_manifest.json`` is in
    _QUEUE_METADATA_SUFFIXES, so the move loop deletes the very file that says
    where the inputs live.

    Looks in: the bundle itself -> the manifest's canonical paths -> Analyzed,
    by stem. Nothing is copied; the returned paths are used as-is, so a pose
    sitting in Analyzed is read from there rather than duplicated locally.

    Returns:
        (mp4_path_or_None, pose_path_or_None)
    """
    import json
    from mousereach.pipeline.manifest import select_pose_file

    def _first_file(paths):
        for p in paths:
            if p and Path(p).is_file():
                return Path(p)
        return None

    mp4 = _first_file(bundle.glob(f"{stem}.mp4"))
    pose_hits = [p for p in bundle.glob(f"{stem}DLC*.h5") if p.is_file()]
    pose = select_pose_file(pose_hits) if pose_hits else None

    if mp4 is None or pose is None:
        manifest = bundle / f"{stem}_manifest.json"
        if manifest.is_file():
            try:
                d = json.loads(manifest.read_text())
                mp4 = mp4 or _first_file([d.get('canonical_video_path')])
                pose = pose or _first_file([d.get('canonical_dlc_h5_path')])
            except Exception as e:
                logger.debug(f"{stem}: could not read bundle manifest: {e}")

    if (mp4 is None or pose is None) and Paths.ANALYZED_OUTPUT:
        root = Path(Paths.ANALYZED_OUTPUT)
        try:
            if mp4 is None:
                mp4 = _first_file(root.rglob(f"{stem}.mp4"))
            if pose is None:
                hits = [p for p in root.rglob(f"{stem}DLC*.h5") if p.is_file()]
                pose = select_pose_file(hits) if hits else None
        except OSError as e:
            logger.debug(f"{stem}: could not search Analyzed: {e}")

    return mp4, pose


def _bundles(queue_root: Optional[Path]) -> List[Path]:
    # Dot-directories are never bundles (a stray .omc/ in both queues was
    # iterated as a phantom video every scan on 2026-09-01, and -- having no
    # segments file -- read as seg_failed and got divert-retried forever).
    if not queue_root or not Path(queue_root).exists():
        return []
    return [d for d in Path(queue_root).iterdir()
            if d.is_dir() and not d.name.startswith(".")]


def _is_queue_metadata(name: str) -> bool:
    return name in _QUEUE_METADATA_NAMES or any(name.endswith(s) for s in _QUEUE_METADATA_SUFFIXES)


def _deep_review_cleared(bundle: Path, stem: str) -> bool:
    """True if a reviewer has cleared this deep-review bundle."""
    if (bundle / f"{stem}_deep_review_cleared.json").exists():
        return True
    # The deep GT tool produced the answer in place.
    if (bundle / f"{stem}_unified_ground_truth.json").exists():
        return True
    return False


def _ensure_durable_review(bundle: Path, stem: str) -> None:
    """Copy this bundle's human review to the durable store before the bundle is
    emptied. Never raises; a failure is logged and the return still proceeds,
    because refusing to return the video would strand the reviewer's clearance.
    """
    src = Path(bundle) / f"{stem}_causal_review.json"
    if not src.is_file():
        return
    try:
        from mousereach.review.causal_review_io import durable_review_path
        dest = durable_review_path(stem)
        if dest is None:
            logger.warning(
                "Return %s: no durable review store configured; the review is "
                "about to move to this node's local disk only.", stem)
            return
        if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        logger.info(f"Return {stem}: human review copied to the durable store")
    except Exception as e:
        logger.error(
            "Return %s: could NOT make a durable copy of the human review (%s). "
            "After this return its only copy is on this node's local disk.",
            stem, e)


def _return_to_processing(bundle: Path, stem: str, processing_dir: Path, db,
                          reason: str) -> bool:
    """Move a cleared bundle's data files into ``processing_dir`` and set the
    video to ``processing`` so the pipeline re-runs it. Queue-only metadata
    (manifest / routing) is dropped so the bundle disappears from the queue.
    Returns True on success."""
    processing_dir = Path(processing_dir)
    processing_dir.mkdir(parents=True, exist_ok=True)

    # Make sure the DB can hold this video BEFORE touching the queue. A node
    # that has never seen the video (fresh DB, or the review was cleared on a
    # different machine) has no row to update -- and this function used to move
    # the bundle's files out, delete the bundle, and only then discover it could
    # not record the state. The clearance was consumed, the files landed in
    # Processing with nothing referencing them, and the function returned True.
    # 274 bundles went that way in one run before it was caught.
    try:
        if db.get_video(stem) is None:
            # Parse the stem's own metadata into the row. Rows registered here
            # used to carry only the id and path -- no tray_type/date/animal --
            # and tray-aware work selection silently degrades to a random pick
            # on a NULL tray_type. This stops NEW metadata-less rows; rows that
            # already exist with NULLs are handled by the orchestrator's
            # read-side filename fallback -- both halves are required.
            meta = {}
            try:
                from .validator import validate_single_filename
                result = validate_single_filename(f"{stem}.mp4")
                if result.valid and result.parsed:
                    meta = {
                        'date': result.parsed['date'],
                        'animal_id': result.parsed['animal_id'],
                        'experiment': result.parsed['experiment'],
                        'cohort': result.parsed['cohort'],
                        'subject': result.parsed['subject'],
                        'tray_type': result.parsed['tray_type'],
                    }
            except Exception:
                meta = {}
            db.register_video(
                video_id=stem,
                source_path=str(processing_dir / f"{stem}.mp4"),
                **meta,
            )
            logger.info(f"Return {stem}: registered (not previously known to this node)")
    except Exception as e:
        logger.error(
            f"Return {stem}: cannot register video, leaving the bundle in the "
            f"queue for a later attempt: {e}"
        )
        return False

    # Resolve the inputs BEFORE emptying the bundle -- the move loop deletes
    # _manifest.json, which is what names them. Returning a video whose pose
    # cannot be found guarantees the pipeline fails on it, so refuse and leave
    # the clearance in the queue rather than spend it on a doomed run.
    mp4_src, pose_src = _resolve_inputs(bundle, stem)
    if pose_src is None:
        logger.error(
            f"Return {stem}: no pose file found (bundle, manifest, or Analyzed). "
            f"Leaving the bundle in the queue -- returning it would re-run the "
            f"video with no pose."
        )
        return False

    # This function is where a review leaves shared storage: every bundle file is
    # MOVED onto this node's local processing dir and the bundle directory is then
    # removed. That is the moment a reviewer's answers stop being visible to any
    # other machine, and if this node's processing dir is later cleared -- or the
    # video never reaches the canonical results dir -- the review is gone.
    #
    # Kinematics reads the review AFTER this point, so losing it here means the
    # human's outcome and causal reach never reach the data product. Make the
    # durable copy before touching anything.
    _ensure_durable_review(bundle, stem)

    h5_dest: Optional[Path] = pose_src
    moved = 0
    for f in list(bundle.iterdir()):
        if not f.is_file():
            continue
        if _is_queue_metadata(f.name):
            try:
                f.unlink()
            except OSError:
                pass
            continue
        dest = processing_dir / f.name
        try:
            _safe_move(f, dest)
            moved += 1
            if f.name.endswith(".h5"):
                h5_dest = dest
        except Exception as e:
            logger.warning(f"Return {stem}: could not move {f.name}: {e}")
            return False

    if moved == 0:
        return False

    kw = {'dlc_output_path': str(h5_dest)} if h5_dest is not None else {}
    try:
        # NOTE FOR ANYONE REASONING ABOUT THE 'processing' POOL: this is a
        # SECOND ENTRY POINT into 'processing', beside intake. Any guard or
        # invariant of the form "videos in processing got there through
        # intake, so a check at intake is sufficient" is wrong because of
        # this path -- a human-cleared review re-enters here, bypassing
        # intake entirely. (An orchestrator work-selection guard was designed
        # on that assumption and had to be redone; hence this sign.)
        #
        # A cleared review routinely returns a video whose row has no legal
        # path to 'processing' (typically 'archived': the bundle sat in a
        # review queue while the row was closed out). That is exactly what
        # force_state exists for -- go straight there instead of letting
        # update_state log an ERROR first and then forcing anyway. The WARNING
        # force_state writes is the honest audit line; the ERROR was noise
        # that fired on every legitimate return.
        state = None
        try:
            row = db.get_video(stem)
            state = row['state'] if row else None
        except Exception:
            state = None
        from .db import VIDEO_TRANSITIONS
        if state is not None and "processing" not in VIDEO_TRANSITIONS.get(state, []):
            db.force_state(stem, "processing",
                           reason="review cleared; returned to Processing", **kw)
        else:
            db.update_state(stem, "processing", **kw)
    except Exception:
        # The row exists (registered above), so a legal-transition failure is
        # what lands here. force_state is the documented escape hatch and logs
        # the bypass; without it the video would sit in Processing unreferenced.
        try:
            db.force_state(stem, "processing",
                           reason="review cleared; returned to Processing", **kw)
        except Exception as e2:
            # Do NOT delete the bundle -- the files are already in Processing but
            # the queue entry is the only remaining pointer to this work.
            logger.error(
                f"Return {stem}: could not set 'processing' state ({e2}). "
                f"Bundle kept in the queue so the clearance is not lost."
            )
            return False

    # Remove the now-empty bundle dir so it leaves the queue.
    try:
        bundle.rmdir()
    except OSError:
        pass

    logger.info(f"Returned {stem} to Processing ({moved} files) reason={reason}")
    return True


# How many bundles one scan may re-inject before yielding back to the work loop.
# Unbounded, this starves everything else: each return moves 6 files across the
# NAS (~10 s), so a 748-deep queue occupied the loop for ~2 hours while
# 'processing' climbed and not one pipeline ran. Returning is not more important
# than running -- interleave them.
MAX_RETURNS_PER_SCAN = 10


def scan_review_queues(db, processing_dir: Path,
                       limit: int = MAX_RETURNS_PER_SCAN) -> Dict[str, int]:
    """Re-inject human-cleared held bundles, at most ``limit`` per scan.

    The remainder are picked up on subsequent cycles, so a deep queue drains
    steadily instead of blocking the pipeline until it is empty.
    """
    summary = {"triage_returned": 0, "deep_returned": 0, "deferred": 0,
               "diverted_to_deep": 0}

    def _budget_left():
        return (summary["triage_returned"] + summary["deep_returned"]
                + summary["diverted_to_deep"]) < limit

    # TRIAGE: every triaged element resolved (and segmentation sound).
    for bundle in _bundles(Paths.TRIAGE_REVIEW):
        if not _budget_left():
            summary["deferred"] += 1
            continue
        stem = bundle.name
        try:
            st = triage_status(bundle, stem)
        except Exception:
            continue
        if st.seg_failed or st.seg_pending_reseg:
            _dr = getattr(Paths, "DEEP_REVIEW", None)
            if _dr and (Path(_dr) / stem).exists():
                # A same-stem bundle ALREADY sits in deep review (historical
                # escalations split bundles across queues: jsons moved, mp4
                # left behind). Diverting would merge blindly into the
                # reviewer's copy, and retrying the impossible move every
                # scan consumed the whole return budget -- the queue
                # live-locked on 2026-09-01 with human-cleared releases
                # starved behind it. Leave duplicates for deliberate cleanup;
                # one info line, no budget spent, no route attempted.
                logger.info("Return scan: %s exists in BOTH queues; divert "
                            "skipped (duplicate needs deliberate cleanup)", stem)
                continue
        if st.seg_failed:
            # A seg-failed bundle can NEVER satisfy the triage release
            # condition below, and nothing else moves it -- it sat in the
            # triage queue with no route out (40 bundles at the time this was
            # added). A failed segmentation makes the whole video
            # untrustworthy, which is deep review's definition, so send it
            # there. Ordering note: the deep-review release path must honor
            # the human clear marker (review_gate seg_failed branch does,
            # same change) or this merely moves the stall to a queue that
            # does not drain.
            try:
                from .review_gate import route_to_queue
                route_to_queue(
                    stem, bundle, Paths.DEEP_REVIEW,
                    reason="segmentation failed -- triage cannot release this "
                           "bundle; needs deep review",
                    db=db, db_state="deep_review",
                )
                summary["diverted_to_deep"] += 1
                logger.info(
                    "Return scan: %s diverted triage -> deep review "
                    "(segmentation failed; no triage route out)", stem)
            except Exception as e:
                logger.error("Return scan: could not divert %s to deep review: %s",
                             stem, e)
            continue
        if not (st.has_triage and st.fully_resolved and not st.seg_failed):
            continue
        if st.seg_pending_reseg:
            # The reviewer answered everything BUT also declared at least one
            # segment mislabeled (true_segment_num set). The boundaries cannot
            # be trusted, so re-injecting would re-run kinematics over the wrong
            # frame windows and archive the video as finished with the
            # segmentation error intact. Divert to DEEP_REVIEW for manual
            # re-segmentation instead; the review file travels with the bundle
            # and re-attaches by frame span after the boundaries are fixed.
            try:
                from .review_gate import route_to_queue
                route_to_queue(
                    stem, bundle, Paths.DEEP_REVIEW,
                    reason="reviewer declared segment mislabel "
                           f"(true_segment_num on segments {sorted(st.seg_corrected)}) "
                           "-- needs re-segmentation",
                    db=db, db_state="deep_review",
                )
                summary["diverted_to_deep"] += 1
                logger.info(
                    "Return scan: %s diverted triage -> deep review; reviewer "
                    "corrected segment number(s) %s", stem, sorted(st.seg_corrected))
            except Exception as e:
                logger.error("Return scan: could not divert %s to deep review: %s",
                             stem, e)
            continue
        if _return_to_processing(bundle, stem, processing_dir, db, "triage_cleared"):
            summary["triage_returned"] += 1

    # DEEP_REVIEW: an explicit clear marker or in-bundle GT.
    # Own budget, NOT shared with the triage loop's: on 2026-09-01 the triage
    # loop's diverts consumed the whole shared budget every scan and 46
    # human-released deep bundles were starved indefinitely. Releases of
    # finished human work must never queue behind machine-initiated moves.
    for bundle in _bundles(Paths.DEEP_REVIEW):
        if summary["deep_returned"] >= limit:
            summary["deferred"] += 1
            continue
        stem = bundle.name
        if _deep_review_cleared(bundle, stem):
            if _return_to_processing(bundle, stem, processing_dir, db, "deep_review_cleared"):
                summary["deep_returned"] += 1

    if summary["triage_returned"] or summary["deep_returned"] or summary["diverted_to_deep"]:
        logger.info(
            f"Review-return scan: {summary['triage_returned']} triage + "
            f"{summary['deep_returned']} deep-review videos re-injected, "
            f"{summary['diverted_to_deep']} diverted to deep review (segment mislabel)"
            + (f" ({summary['deferred']} left for later cycles)"
               if summary["deferred"] else "")
        )
    return summary
