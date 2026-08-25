"""Qt-free triage / resolution status for a video's algo outputs.

Single source of truth for the THREE questions the automatic pipeline gate and
the interactive review tool both ask -- so they can never drift apart:

  1. Did segmentation FAIL?  A hard failure never writes a segments file; a SOFT
     failure (overall_confidence <= 0, or a "reference quality" anomaly -- the
     uniform-fallback case from an over-long recording) writes one but the
     boundaries are junk. Both belong in DEEP review / manual re-seg, not the
     fast per-element triage queue.
  2. Does the video have any TRIAGED element?  Either the cascade left a segment
     ``outcome == "triaged"`` / ``flagged_for_review``, or a touched outcome
     (retrieved / displaced_sa / displaced_outside) has no committed causal
     reach in the assignment.
  3. Which triaged elements remain UNRESOLVED by a human review?  A segment is
     resolved once its ``*_causal_review.json`` record has
     ``answers.reviewed is not False``.

The kinematics gate rule the whole project hangs on:
    kinematics run ONLY when segmentation is sound AND every triaged element has
    been specifically addressed by a human. Anything else is held out of the
    archive + connectome.db.

All core functions take already-loaded JSON docs (dicts) so they do zero I/O and
are trivially testable. ``triage_status`` is the dir-based convenience wrapper.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Touched outcomes: a real reach moved/removed the pellet, so a causal reach
# MUST be committed. A touched segment with no committed causal reach is an
# unresolved assignment -> triage.
_TOUCHED = ("retrieved", "displaced_sa", "displaced_outside")


# ---------------------------------------------------------------------------
# Pure (doc-in) predicates -- no I/O
# ---------------------------------------------------------------------------

def segmentation_failed(seg_doc: Optional[Dict[str, Any]]) -> bool:
    """True if segmentation is unusable even though a segments file exists.

    Mirrors the review tool's ``_segmentation_failed``: overall_confidence <= 0
    (uniform-fallback) or a "reference quality" anomaly. A missing/empty doc is
    treated as failed (there is nothing to trust)."""
    if not seg_doc:
        return True
    c = seg_doc.get("overall_confidence")
    if c is not None and c <= 0.0:
        return True
    anoms = seg_doc.get("anomalies") or []
    return any("reference quality" in str(a).lower() for a in anoms)


def segmentation_corrected(review_doc: Optional[Dict[str, Any]]) -> Set[int]:
    """Segment numbers where a REVIEWER declared the segmentation wrong.

    A review record carries ``true_segment_num`` + ``segmentation_wrong: True``
    when the human said "this stretch is not the pellet presentation the
    segmenter thinks it is" (causal_review_io.make_segment_record). That is a
    fact about the segmentation, not about the reach or outcome -- so a video
    with any such record must be re-segmented, not returned to the pipeline
    with its boundaries intact. Distinct from :func:`segmentation_failed`,
    which is the segmenter's OWN account; this is the human overruling it."""
    if not review_doc:
        return set()
    return {
        int(rec["segment_num"])
        for rec in review_doc.get("segments", [])
        if rec.get("segmentation_wrong") and rec.get("segment_num") is not None
    }


def _committed_causal_segments(assign_doc: Optional[Dict[str, Any]]) -> Set[int]:
    """Segment numbers that have a committed causal reach (is_causal)."""
    if not assign_doc:
        return set()
    return {
        int(r["segment_num"])
        for r in assign_doc.get("reaches", [])
        if r.get("is_causal") and r.get("segment_num") is not None
    }


def triaged_segments(
    outcome_doc: Optional[Dict[str, Any]],
    assign_doc: Optional[Dict[str, Any]] = None,
) -> Set[int]:
    """Segment numbers with a triaged element -- SAME definition the review
    tool's ``_bundle_has_triage`` uses.

    A segment is triaged if the cascade flagged it (``outcome == "triaged"`` or
    ``flagged_for_review``), or it is a touched outcome with no committed causal
    reach in the assignment. If ``assign_doc`` is None the touched-no-causal
    check is skipped (cascade triage still applies)."""
    if not outcome_doc:
        return set()
    committed = _committed_causal_segments(assign_doc)
    out: Set[int] = set()
    for s in outcome_doc.get("segments", []):
        sn = s.get("segment_num")
        if sn is None:
            continue
        sn = int(sn)
        oc = s.get("outcome")
        if oc == "triaged" or s.get("flagged_for_review"):
            out.add(sn)
        elif oc in _TOUCHED and assign_doc is not None and sn not in committed:
            out.add(sn)
    return out


def resolved_segments(review_doc: Optional[Dict[str, Any]]) -> Set[int]:
    """Segment numbers a human review has actively resolved.

    A record counts as resolved when ``answers.reviewed is not False`` -- the
    exact rule ``_bundle_reviewed`` uses (an explicit ``reviewed: False`` is a
    deferred/unanswered segment)."""
    if not review_doc:
        return set()
    out: Set[int] = set()
    for rec in review_doc.get("segments", []):
        sn = rec.get("segment_num")
        if sn is None:
            continue
        if (rec.get("answers") or {}).get("reviewed") is not False:
            out.add(int(sn))
    return out


def unresolved_triaged(
    outcome_doc: Optional[Dict[str, Any]],
    assign_doc: Optional[Dict[str, Any]] = None,
    review_doc: Optional[Dict[str, Any]] = None,
) -> Set[int]:
    """Triaged segments NOT yet resolved by a human review."""
    return triaged_segments(outcome_doc, assign_doc) - resolved_segments(review_doc)


# ---------------------------------------------------------------------------
# Dir-based wrapper
# ---------------------------------------------------------------------------

@dataclass
class TriageStatus:
    """The full triage/resolution picture for one video."""
    video_id: str
    seg_failed: bool
    # The segmenter's own account of why its boundaries should be checked by a
    # person. Distinct from seg_failed: the segmenter did not fail, it produced
    # an answer it had to force. Empty means the boundaries were found.
    seg_needs_human: List[str] = field(default_factory=list)
    triaged: Set[int] = field(default_factory=set)
    resolved: Set[int] = field(default_factory=set)
    unresolved: Set[int] = field(default_factory=set)
    # Segments where a reviewer set true_segment_num: the human says the
    # segmentation is wrong about WHICH pellet this is. Non-empty means the
    # video needs re-segmentation before it can be trusted again.
    seg_corrected: Set[int] = field(default_factory=set)

    @property
    def has_triage(self) -> bool:
        return bool(self.triaged)

    @property
    def fully_resolved(self) -> bool:
        """Every triaged element has been addressed (or there were none)."""
        return not self.unresolved

    @property
    def clean(self) -> bool:
        """Ready for kinematics: segmentation sound AND nothing left unresolved.

        "Sound" includes the reviewer's verdict: a human-declared segment
        mislabel (seg_corrected) means the boundaries cannot be trusted even
        though the segmenter reported success."""
        # seg_needs_human is deliberately NOT considered here -- see the note in
        # watcher/review_gate.py. It is recorded, not acted on.
        return (not self.seg_failed) and (not self.seg_corrected) and self.fully_resolved


def _load(dir_: Path, name: str) -> Optional[Dict[str, Any]]:
    p = Path(dir_) / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def triage_status(directory: Path, video_id: str) -> TriageStatus:
    """Read a video's algo outputs + any review from ``directory`` and return the
    full :class:`TriageStatus`. ``directory`` is the Processing dir (fresh run,
    no review yet) or a review bundle (review present)."""
    directory = Path(directory)
    seg = _load(directory, f"{video_id}_segments.json")
    outcome = _load(directory, f"{video_id}_pellet_outcomes.json")
    assign = _load(directory, f"{video_id}_reach_assignments.json")
    review = _load(directory, f"{video_id}_causal_review.json")

    tri = triaged_segments(outcome, assign)
    res = resolved_segments(review)
    return TriageStatus(
        video_id=video_id,
        seg_failed=segmentation_failed(seg),
        seg_needs_human=list((seg or {}).get("needs_human") or []),
        triaged=tri,
        resolved=res,
        unresolved=tri - res,
        seg_corrected=segmentation_corrected(review),
    )
