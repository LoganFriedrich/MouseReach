"""The pipeline's human-review GATE + routing.

Called AFTER algo-4 (reach assignment), BEFORE kinematics. Decides a freshly
(re)processed video's fate and, when it cannot proceed on its own, MOVES the
whole bundle into the right review queue so kinematics + connectome.db never
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
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

from ..config import Paths
from ..review.triage_status import TriageStatus, triage_status
from .review_routing import move_video_bundle

logger = logging.getLogger(__name__)

DECISION_CLEAN = "clean"
DECISION_TRIAGE = "triage"
DECISION_DEEP = "deep_review"


def _gt_certification(video_id: str) -> Tuple[bool, set]:
    """``(fully_exhaustive, gt_determined_outcome_segnums)`` for a video's GT.

    ``fully_exhaustive`` is True when GT marks the whole video complete
    (``completion_status.all_complete`` or both reaches+outcomes exhaustive) --
    a human has certified every element, so the whole video is human truth. The
    second value is the set of segment_nums whose outcome GT determined, used to
    clear those specific segments when GT is only partial. Never raises."""
    try:
        from ..review.causal_review_io import find_gt
        p = find_gt(video_id)
        if not p:
            return False, set()
        gt = json.loads(Path(p).read_text(encoding="utf-8"))
        cs = gt.get("completion_status") or {}
        rr = (gt.get("reaches") or {}).get("exhaustive")
        oo = (gt.get("outcomes") or {}).get("exhaustive")
        full = bool(cs.get("all_complete")) or bool(rr and oo)
        segs = set()
        for seg in ((gt.get("outcomes") or {}).get("segments") or []):
            sn = seg.get("segment_num")
            if sn is not None and seg.get("determined") is not False:
                segs.add(int(sn))
        return full, segs
    except Exception:
        return False, set()


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
    gt_full, gt_segs = _gt_certification(video_id)
    if gt_full:
        return DECISION_CLEAN, "ground_truth_exhaustive", st
    if st.seg_failed:
        return DECISION_DEEP, "segmentation_failed", st
    if qc_verdict == "needs_review":
        return DECISION_DEEP, "qc_needs_review", st
    # GT-determined segments are resolved; so are segments a human addressed in a
    # review saved ANYWHERE the resolver looks (the Pending bundle, or next to the
    # canonical video) -- not just the fresh work dir triage_status read. Honoring
    # it lets a re-run of an already-reviewed video finalize CLEAN instead of
    # re-triaging (and losing the reviewer's resolution).
    review_resolved = _review_resolved_segments(video_id, processing_dir)
    unresolved = set(st.unresolved) - gt_segs - review_resolved
    if unresolved:
        segs = ", ".join(str(s) for s in sorted(unresolved))
        return DECISION_TRIAGE, f"{len(unresolved)} triaged segment(s) unresolved: {segs}", st
    return DECISION_CLEAN, "clean", st


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
        return resolved_segments(json.loads(rp.read_text(encoding="utf-8")))
    except Exception:
        return set()


def _write_review_manifest(bundle: Path, video_id: str, reason: str) -> None:
    """Write the ``manifest.json`` the review tool needs to open a self-contained
    bundle -- BUNDLE-LOCAL canonical pointers so the video + pose load in place.
    """
    mp4 = bundle / f"{video_id}.mp4"
    h5s = sorted(bundle.glob(f"{video_id}*.h5"))
    manifest = {
        "type": "causal_review_bundle",
        "schema_version": "1.0",
        "video_stem": video_id,
        "canonical_video_path": str(mp4),
        "canonical_dlc_h5_path": str(h5s[0]) if h5s else None,
        "provenance": {
            "routed_reason": reason,
            "staged_at": datetime.now().isoformat(),
            "self_contained": True,
        },
    }
    (bundle / f"{video_id}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def route_to_queue(
    video_id: str,
    processing_dir: Path,
    dest_root: Path,
    reason: str,
    db=None,
    db_state: Optional[str] = None,
    extra_sources: Optional[Iterable[Path]] = None,
) -> Path:
    """Move a video bundle into a review queue, write the review manifest, and
    set the DB hold state. Returns the bundle dir."""
    bundle, moved = move_video_bundle(
        video_id, processing_dir, dest_root, reason,
        extra_sources=extra_sources,
        extra_manifest={"db_state": db_state} if db_state else None,
    )
    _write_review_manifest(bundle, video_id, reason)
    if db is not None and db_state:
        try:
            db.update_state(video_id, db_state)
        except Exception as e:
            logger.warning(f"Gate: could not set DB state {db_state} for {video_id}: {e}")
    return bundle


def route_deep_review(
    video_id: str, processing_dir: Path, reason: str, db=None,
    extra_sources: Optional[Iterable[Path]] = None,
) -> Path:
    """Route a video to DEEP_REVIEW (segmentation failed / QC-critical / escalated)."""
    return route_to_queue(
        video_id, processing_dir, Paths.DEEP_REVIEW, reason,
        db=db, db_state="deep_review", extra_sources=extra_sources,
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
        route_to_queue(video_id, processing_dir, Paths.TRIAGE_REVIEW, reason,
                       db=db, db_state="triage", extra_sources=extra)
        logger.info(f"Gate: {video_id} -> TRIAGE ({reason})")
    else:
        logger.info(f"Gate: {video_id} -> CLEAN (proceed to kinematics)")
    return decision
