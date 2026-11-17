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
    archive + the central database.

All core functions take already-loaded JSON docs (dicts) so they do zero I/O and
are trivially testable. ``triage_status`` is the dir-based convenience wrapper.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

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


def triaged_in_range(
    outcome_doc: Optional[Dict[str, Any]],
    assign_doc: Optional[Dict[str, Any]] = None,
    seg_doc: Optional[Dict[str, Any]] = None,
) -> Tuple[Set[int], Set[int]]:
    """``(kept, ghost)``: the triaged segments that exist, and the flags that
    name a segment the segmentation does not have.

    THE one definition of "this video has something to triage", used by the
    gate, the return scan and the review window alike. A flag can name a
    segment that does not exist (a producer numbering by boundary index rather
    than by segment: a flag on segment 21 of a 20-segment video). Nobody can
    answer it, so it never counts as triage -- on any side. Counting it on one
    side only offered the bundle to a reviewer with nothing to ask, and left a
    bundle whose only flags were such ghosts in the queue for ever (the return
    scan found "no triage" and the window found "something to ask"). With no
    segments document nothing is known to be out of range, so nothing is
    dropped."""
    tri = triaged_segments(outcome_doc, assign_doc)
    bounds = (seg_doc or {}).get("boundaries") or []
    if not bounds:
        return tri, set()
    n_segs = max(0, len(bounds) - 1)
    ghost = {s for s in tri if isinstance(s, int) and s > n_segs}
    return tri - ghost, ghost


def resolved_segments(review_doc: Optional[Dict[str, Any]],
                      seg_doc: Optional[Dict[str, Any]] = None) -> Set[int]:
    """Segment numbers a human review has actively resolved.

    A record counts as resolved when it is a person's answer
    (``causal_review_io.is_answered``): not an explicit ``reviewed: False``
    placeholder, and not a pellet saved with its outcome left undecided.

    Pass the video's segments document (``seg_doc``) and each answer counts for
    the CURRENT segment its frames match, exactly as the results step applies
    it (causal_review_io.answered_segments says why). Without it, answers count
    by the segment number they were written against."""
    if not review_doc:
        return set()
    if seg_doc is not None:
        from .causal_review_io import (answered_segments, segmentation_is_human,
                                       segments_from_segmentation)
        current = segments_from_segmentation(seg_doc)
        if current:
            return answered_segments(review_doc, current,
                                     human_segmentation=segmentation_is_human(seg_doc))
    from .causal_review_io import is_answered
    out: Set[int] = set()
    for rec in review_doc.get("segments", []):
        sn = rec.get("segment_num")
        if sn is None:
            continue
        if is_answered(rec):
            out.add(int(sn))
    return out


def gt_resolved_segments(video_id: str, extra_dirs=(), current_segments=None,
                         human_segmentation: Optional[bool] = None) -> "tuple[bool, Set[int]]":
    """What ground truth has already determined for this video.

    GROUND TRUTH IS AN ANSWER. ``evaluate_gate`` has always known that -- it
    subtracts GT-determined segments before deciding, so a GT'd video passes as
    clean. The RELEASE side did not: ``triage_status`` built its resolved set
    from the causal-review document alone, so a bundle whose questions GT had
    answered could never satisfy ``fully_resolved`` and never left the queue.
    The review tool skipped it too (``has_gt`` -> nothing to ask), so nobody
    asked and nobody released, and bundles stuck exactly there,
    indefinitely.

    Returns ``(fully_certified, determined_segment_nums)``, the same shape the
    gate's own ``_gt_certification`` returns, so the two sides cannot drift.
    ``fully_certified`` means reaches AND outcomes are both marked exhaustive
    AND every wanted pellet's outcome is determined (see the comment at the
    computation) -- such a video has no unanswered element by definition. A
    stored ``completion_status`` is ignored.

    ``extra_dirs``: folders to look in BEFORE the corpus-wide index -- the
    folder the video's files are in (the processing folder of a run, a review
    bundle). WHY: ground truth travels with a video's files, but
    the index lists only what was there when it was built, once per process.
    A re-run copies the ground truth into the processing folder AFTER that, so
    the gate could not see it and sent videos whose open segments ground truth
    had answered back to a person. Ground truth is a person's answer, and a
    person's answer is never asked again: a segment whose boundaries, causal
    reach and outcome have all already been reviewed by a human never requires
    a human to look at it again.

    The segment numbers returned are CURRENT ones: each determined GT outcome
    counts for the current segment its frames cover in the GT's own
    segmentation (causal_review_io.index_gt_outcomes_by_segment), the rule
    review answers follow -- never its bare number, which a re-cut reassigns.
    The current segments come from ``current_segments``, else from the first
    ``{video_id}_segments.json`` found in ``extra_dirs``; with neither, or for a
    GT file without its own segmentation, the number is used.

    Never raises: no GT, unreadable GT, or a missing index all yield
    ``(False, set())``, which leaves the previous behaviour exactly as it was."""
    try:
        from .causal_review_io import (_seg_span, _too_short, find_gt,
                                       index_gt_outcomes_by_segment,
                                       segmentation_is_human, segments_from_segmentation)
        p = find_gt(video_id, extra_dirs=tuple(d for d in extra_dirs if d))
        if not p:
            return False, set()
        gt = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return False, set()
    if current_segments is None:
        for d in extra_dirs:
            seg_doc = _load(Path(d), f"{video_id}_segments.json") if d else None
            if seg_doc:
                current_segments = segments_from_segmentation(seg_doc) or None
                if human_segmentation is None:
                    human_segmentation = segmentation_is_human(seg_doc)
                break
    rr = (gt.get("reaches") or {}).get("exhaustive")
    oo = (gt.get("outcomes") or {}).get("exhaustive")
    by_seg, _ = index_gt_outcomes_by_segment(gt, current_segments,
                                             human_segmentation=bool(human_segmentation))
    out: Set[int] = {sn for sn, seg in by_seg.items() if seg.get("determined") is not False}
    # "Fully certified" is COMPUTED here, never taken from a stored flag. WHY:
    # a fully certified video passes the gate as clean even with a failed
    # segmentation, and its ground truth stands in for every element -- so the
    # claim must be true of the file as it is now. The two "exhaustive" flags
    # are switches a person sets in the ground-truth tool, which can be set
    # while some outcomes are still undetermined, and the stored
    # completion_status (written by unified_gt.update_completion_status) can be
    # stale against the file or set while outcomes are undetermined. So: both
    # flags set AND every pellet wanted has an outcome a person determined. The
    # pellets wanted are the video's CURRENT segments when they are known
    # (matched by frames above), else every pellet the ground truth lists. A
    # current segment too short for any pellet presentation (a segmenter
    # artifact, causal_review_io._too_short) is not wanted: no outcome is ever
    # mapped onto it, so wanting it would refuse full certification to every
    # video the segmenter left such a sliver in, with nothing a person could
    # fix in the ground-truth tool.
    if current_segments:
        human = bool(human_segmentation)
        want = {int(s["segment_num"]) for s in current_segments
                if s.get("segment_num") is not None
                and not _too_short(_seg_span(s), human_segmentation=human)}
    else:
        want = {int(s["segment_num"])
                for s in ((gt.get("outcomes") or {}).get("segments") or [])
                if isinstance(s, dict) and s.get("segment_num") is not None}
    determined = {sn for sn, seg in by_seg.items() if seg.get("determined") is True}
    full = bool(rr and oo and want and want <= determined)
    return full, out


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
    # video needs re-segmentation before it can be trusted again -- UNLESS the
    # boundaries have since been fixed by hand (seg_human_fixed below).
    seg_corrected: Set[int] = field(default_factory=set)
    # True when the segments file carries boundary_source == "human": the
    # fix-segmentation tool's stamp that a person set the cuts. A review's old
    # segmentation_wrong record describes a problem that no longer exists once
    # this is True, so it must stop blocking the video.
    seg_human_fixed: bool = False
    # Segments too short to be a pellet presentation ({segment_num: (first,
    # last frame)}), on cuts no person made. A triaged one cannot be answered in
    # triage (causal_review_io.answer_problem); it is a question about the cuts,
    # which deep review fixes.
    too_short: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    # Flags naming a segment that does not exist (triaged_in_range). Never
    # part of ``triaged``; kept so a bundle whose ONLY flags are these can be
    # recognised and returned instead of sitting in triage for ever.
    ghost: Set[int] = field(default_factory=set)

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
        return ((not self.seg_failed)
                and self.fully_resolved
                and (not self.seg_pending_reseg))

    @property
    def seg_pending_reseg(self) -> bool:
        """A reviewer declared a segment mislabel and nobody has hand-fixed the
        boundaries yet -- the video must go through manual re-segmentation."""
        return bool(self.seg_corrected) and not self.seg_human_fixed


def _too_short_segments(seg_doc) -> Dict[int, Tuple[int, int]]:
    """Segments too short to be a pellet presentation, on cuts no person made
    (the same rule the review matching uses)."""
    from .causal_review_io import _too_short, segmentation_is_human, segments_from_segmentation
    human = segmentation_is_human(seg_doc)
    return {int(s["segment_num"]): (int(s["start_frame"]), int(s["end_frame"]))
            for s in segments_from_segmentation(seg_doc)
            if _too_short((int(s["start_frame"]), int(s["end_frame"])), human)}


def short_segment_reason(st: "TriageStatus", segs) -> str:
    """The routing reason for triaged segments too short to be answered. Names
    segmentation, so the deep-review window points the reviewer at fixing the
    cuts."""
    parts = []
    for s in segs:
        a, b = st.too_short[s]
        parts.append("%d (frames %d-%d, %d frame(s))" % (s, a, b, b - a + 1))
    return ("segmentation: segment(s) %s too short to be a pellet presentation -- two "
            "cuts almost on top of each other. No pellet can be answered there until "
            "the cuts are fixed." % "; ".join(parts))


def _load(dir_: Path, name: str) -> Optional[Dict[str, Any]]:
    p = Path(dir_) / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as err:
        logger.warning('Could not read %s, so it counts as absent: %s', p, err)
        return None


def _effective_review_of(directory: Path, video_id: str, seg_doc) -> Optional[Dict[str, Any]]:
    """Every copy of this video's review, merged (causal_review_io.
    effective_review over review_dirs with ``directory`` first). Falls back to
    the copy in ``directory`` alone if the merge itself cannot run."""
    try:
        from .causal_review_io import effective_review, review_dirs
        return effective_review(video_id, review_dirs(video_id, directory), seg_doc)
    except Exception as err:
        logger.warning("%s: could not merge the copies of its review (%s); reading "
                       "the copy in %s only", video_id, err, directory)
        return _load(directory, f"{video_id}_causal_review.json")


def _mislabels_of(directory: Path, video_id: str, merged) -> Optional[Dict[str, Any]]:
    """The mislabel records of every copy (causal_review_io.mislabel_records):
    the merged review drops them when the pellet was left open."""
    try:
        from .causal_review_io import mislabel_records, review_dirs
        doc = mislabel_records(video_id, review_dirs(video_id, directory))
        return doc if doc["segments"] else merged
    except Exception:
        return merged


def triage_status(directory: Path, video_id: str) -> TriageStatus:
    """Read a video's algo outputs + any review from ``directory`` and return the
    full :class:`TriageStatus`. ``directory`` is the Processing dir (fresh run,
    no review yet) or a review bundle (review present)."""
    directory = Path(directory)
    seg = _load(directory, f"{video_id}_segments.json")
    outcome = _load(directory, f"{video_id}_pellet_outcomes.json")
    assign = _load(directory, f"{video_id}_reach_assignments.json")
    # Every copy of the review, merged statement by statement (effective_review
    # over review_dirs: this folder, the queue bundles, the filed folder and
    # the durable copy) -- the reading the window, the release check and the
    # results step use. WHY: this read only the copy in ``directory`` while
    # the window merged every copy, so a pellet answered (or reopened) in
    # another copy was open here and answered there, or the reverse.
    review = _effective_review_of(directory, video_id, seg)

    tri, ghost = triaged_in_range(outcome, assign, seg)
    if ghost:
        logger.warning(
            "%s: triage flags name nonexistent segment(s) %s (only %d segments "
            "exist); ignoring them", video_id, sorted(ghost),
            max(0, len((seg or {}).get("boundaries") or []) - 1))

    res = resolved_segments(review, seg)

    # Ground truth counts as an answer here, exactly as it does in evaluate_gate.
    # Without this the release side is narrower than the admission side: a bundle
    # whose questions GT had determined could never satisfy fully_resolved, while
    # the review tool skipped it because has_gt() said there was nothing to ask.
    # Nobody asked, nobody released, and it sat in the queue forever.
    # The release side reads the corpus index only, NOT ground truth inside
    # ``directory`` (the gate reads both, review_gate._gt_certification).
    # Deliberately, for now: a triage bundle can hold ground truth that
    # answers every open segment and that the index cannot see (it covers
    # this machine's processing folder, not the shared queues). Reading it here
    # would release every such bundle on the next scan -- a full re-run and
    # filing of each -- so it waits on a decision about moving queue items. Pass
    # extra_dirs=(directory,) once that is decided.
    from .causal_review_io import segments_from_segmentation
    gt_full, gt_segs = gt_resolved_segments(
        video_id, current_segments=segments_from_segmentation(seg) or None,
        human_segmentation=(seg or {}).get("boundary_source") == "human")
    res = tri if gt_full else (res | gt_segs)

    return TriageStatus(
        video_id=video_id,
        seg_failed=segmentation_failed(seg),
        seg_needs_human=list((seg or {}).get("needs_human") or []),
        triaged=tri,
        resolved=res,
        unresolved=tri - res,
        seg_corrected=segmentation_corrected(_mislabels_of(directory, video_id, review)),
        seg_human_fixed=(seg or {}).get("boundary_source") == "human",
        too_short=_too_short_segments(seg),
        ghost=ghost,
    )


def deep_review_cleared(directory: Path, video_id: str) -> bool:
    """True when a person has cleared this video's deep review: the clear
    marker ``{video_id}_deep_review_cleared.json`` is in ``directory``. Never
    raises.

    The marker ONLY. A ground-truth file used to count as a clear by its mere
    presence -- even a progress save with nothing determined, and without
    anyone pressing Clear -- and the release then let the segmenter cut the
    video again exactly as before. The review gate applies the same test
    (watcher.review_gate._deep_review_cleared). Ground truth that certifies
    the WHOLE video is a separate, stronger fact; :func:`held_for_segmentation`
    honours it as the gate does."""
    try:
        return (Path(directory) / f"{video_id}_deep_review_cleared.json").is_file()
    except Exception:
        return False


def held_for_segmentation(directory: Path, video_id: str,
                          st: Optional["TriageStatus"] = None) -> bool:
    """True when this video's segmentation must be fixed before anything else
    can be asked: the segmentation failed (or its file is missing or
    unreadable -- there is nothing to trust) AND nobody has released it from
    that: no clear marker (:func:`deep_review_cleared`) and no ground truth
    certifying the whole video (:func:`gt_resolved_segments`; the gate passes
    such a video clean even with a failed segmentation, so holding it would
    hold a video the gate releases).

    THE one definition, used by the return scan (which sends such a triage
    bundle to deep review) and the review window (which then does not offer
    it in triage). WHY one: the window had its own copy that read an
    unreadable segments file as "not failed" and offered the bundle, while the
    scan read it as failed and moved it to deep review under the reviewer --
    and the window's copy ignored a person's clear, so a cleared bundle with
    open pellets was offered by neither. ``st`` saves a second read when the
    caller already has the status."""
    failed = (st.seg_failed if st is not None else
              segmentation_failed(_load(Path(directory), f"{video_id}_segments.json")))
    if not failed:
        return False
    if deep_review_cleared(directory, video_id):
        return False
    try:
        return not gt_resolved_segments(video_id, extra_dirs=(Path(directory),))[0]
    except Exception as err:
        logger.warning("%s: could not read its ground truth, so it counts as not "
                       "certifying the video: %s", video_id, err)
        return True


def bundle_triage(directory: Path, video_id: str) -> Tuple[Set[int], Set[int]]:
    """``(kept, ghost)`` for the video's files in ``directory`` -- the
    :func:`triaged_in_range` rule over its outcomes, assignments and segments.
    A missing or unreadable file counts as absent (``_load``): a missing
    assignments file skips only the touched-without-causal test, never the
    algorithm's own flags, exactly as the gate decides."""
    d = Path(directory)
    return triaged_in_range(_load(d, f"{video_id}_pellet_outcomes.json"),
                            _load(d, f"{video_id}_reach_assignments.json"),
                            _load(d, f"{video_id}_segments.json"))
