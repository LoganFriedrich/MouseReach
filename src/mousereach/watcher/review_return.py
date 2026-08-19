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
  DEEP_REVIEW -- an explicit ``{stem}_deep_review_cleared.json`` marker (written
                 by the deep tools when a reviewer clears the flag) OR a
                 co-located ``{stem}_unified_ground_truth.json`` (the deep GT
                 tool produced the answer). Re-injection restarts the pipeline
                 from segmentation.

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

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


def _bundles(queue_root: Optional[Path]) -> List[Path]:
    if not queue_root or not Path(queue_root).exists():
        return []
    return [d for d in Path(queue_root).iterdir() if d.is_dir()]


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
            db.register_video(
                video_id=stem,
                source_path=str(processing_dir / f"{stem}.mp4"),
            )
            logger.info(f"Return {stem}: registered (not previously known to this node)")
    except Exception as e:
        logger.error(
            f"Return {stem}: cannot register video, leaving the bundle in the "
            f"queue for a later attempt: {e}"
        )
        return False

    h5_dest: Optional[Path] = None
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

    try:
        if h5_dest is not None:
            db.update_state(stem, "processing", dlc_output_path=str(h5_dest))
        else:
            db.update_state(stem, "processing")
    except Exception:
        # The row exists (registered above), so a legal-transition failure is
        # what lands here. force_state is the documented escape hatch and logs
        # the bypass; without it the video would sit in Processing unreferenced.
        try:
            kw = {'dlc_output_path': str(h5_dest)} if h5_dest is not None else {}
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


def scan_review_queues(db, processing_dir: Path) -> Dict[str, int]:
    """Re-inject every human-cleared held bundle. Returns a summary dict."""
    summary = {"triage_returned": 0, "deep_returned": 0}

    # TRIAGE: every triaged element resolved (and segmentation sound).
    for bundle in _bundles(Paths.TRIAGE_REVIEW):
        stem = bundle.name
        try:
            st = triage_status(bundle, stem)
        except Exception:
            continue
        if st.has_triage and st.fully_resolved and not st.seg_failed:
            if _return_to_processing(bundle, stem, processing_dir, db, "triage_cleared"):
                summary["triage_returned"] += 1

    # DEEP_REVIEW: an explicit clear marker or in-bundle GT.
    for bundle in _bundles(Paths.DEEP_REVIEW):
        stem = bundle.name
        if _deep_review_cleared(bundle, stem):
            if _return_to_processing(bundle, stem, processing_dir, db, "deep_review_cleared"):
                summary["deep_returned"] += 1

    if summary["triage_returned"] or summary["deep_returned"]:
        logger.info(
            f"Review-return scan: {summary['triage_returned']} triage + "
            f"{summary['deep_returned']} deep-review videos re-injected"
        )
    return summary
