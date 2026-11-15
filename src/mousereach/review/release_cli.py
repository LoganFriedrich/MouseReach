"""Bulk release check for deep-review bundles -- a human-invoked CLI.

``mousereach-review-release`` lists every bundle in the deep-review queue with
its review completeness, and (with ``--clear``) writes the release marker for
the complete ones. It exists because completing a review and releasing the
bundle are two separate acts in the review tool, and the second one was being
skipped: bundles sat in the queue with every segment answered and no marker,
so the watcher's return scan never picked them up.

Completeness is judged by ``causal_review_io.is_answered`` -- the one
definition of a person's answer every other check uses -- on the answers whose
frames match each CURRENT segment (``answered_segments``). WHY not
``human.outcome is not None``, the test this used before: the review window
saves a pellet left with no outcome picked with the algorithm's "triaged" in
its outcome and ``reviewed: False``, and a pellet sent back with "are you
sure?" keeps its old outcome under ``reviewed: False``. Both are open
questions; the old test counted both as answered, so a bundle could be
released with a pellet nobody had decided.

A bundle whose every segment is answered is complete whoever cut it. A
hand-corrected segmentation (``boundary_source == "human"`` in the bundle's
segments file) also finishes a bundle that still has open pellets -- WHEN the
bundle was routed for a segmentation problem. The routing reason gates that: the clear marker is a
blanket human-clear token the watcher gate honors, so releasing a bundle
routed for e.g. a QC hold on the strength of a cut-fix would clear a concern
nobody addressed. Such bundles are listed as held back, with their reason;
their open pellets are what the review window shows next.

This deliberately stays a human-invoked bulk form of the review tool's own
CLEAR button, writing the same marker the button writes. Whether clearing
should happen automatically on full answers is an open design decision; until
that is made, a person runs this.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)


SEGMENTATION_REASON_KEYWORDS = ("segmentation", "mislabel", "re-seg", "reseg")


def _names_segmentation(reason: str) -> bool:
    """Does a routing reason say the bundle is here for a segmentation
    problem? Deliberately keyword-loose: reasons are free text."""
    r = (reason or "").lower()
    return any(k in r for k in SEGMENTATION_REASON_KEYWORDS)


def _routing_reason(bundle: Path, stem: str) -> str:
    """The routed_reason recorded when the bundle entered the queue ('' if
    the routing manifest is absent or unreadable)."""
    p = Path(bundle) / f"{stem}_routing.json"
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("routed_reason", ""))
    except Exception:
        return ""


def _boundary_source_human(bundle: Path, stem: str) -> bool:
    """True when a person set this bundle's segment cuts by hand."""
    p = Path(bundle) / f"{stem}_segments.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("boundary_source") == "human"
    except Exception as err:
        logger.warning('Could not read %s, so its boundaries count as not human-made: %s', p, err)
        return False


def _bundle_dirs(queue_root: Optional[Path]) -> List[Path]:
    if not queue_root or not Path(queue_root).exists():
        return []
    return sorted(d for d in Path(queue_root).iterdir()
                  if d.is_dir() and not d.name.startswith("."))


def classify_queue(queue_root: Optional[Path]) -> dict:
    """Classify every bundle in a deep-review queue by how (and whether) it
    can be released. SHARED ENGINE: the napari Review Queues tab's release
    button is the primary interface -- the target user has no computing
    background and nothing may exist CLI-only; the CLI below is the button's
    terminal mirror for power users and scripting. Keys:
      complete        [(stem, answered, total)]   every segment answered, whoever cut it
      fixed_release   [(stem, reason)]            cuts hand-fixed AND routed for segmentation
      fixed_held      [(stem, reason)]            cuts hand-fixed, routed for something
                                                  else, and pellets still open: the
                                                  review window shows it
      partial_walk    [(stem, a, t, n_asked)]     every TRIAGED segment answered (human judges)
      partial         [(stem, answered, total)]
      unreviewed      [stem]
      already         [stem]                      marker present, awaiting the return scan
      not_resegmented [(stem, why)]               nobody re-segmented it since it arrived:
                                                  it cannot leave (review/resegmented.py)
    """
    from mousereach.config import is_outcome_free_tray
    from mousereach.review.resegmented import resegmented_since_entry
    out = {"complete": [], "fixed_release": [], "fixed_held": [],
           "partial_walk": [], "partial": [], "unreviewed": [], "already": [],
           "not_resegmented": []}
    for bundle in _bundle_dirs(queue_root):
        stem = bundle.name
        if not is_outcome_free_tray(stem):
            ok, why = resegmented_since_entry(bundle, stem)
            if not ok:
                out["not_resegmented"].append((stem, why))
                continue
        if (bundle / f"{stem}_deep_review_cleared.json").is_file():
            out["already"].append(stem)
            continue
        # Every pellet answered is complete WHOEVER cut the video. WHY first:
        # this used to come after the hand-cut test, so a bundle whose cuts a
        # person had re-set AND whose every pellet a person had answered, but
        # which was sent here for a reason other than segmentation (a quality
        # hold, a flagged session), was "held back" here, while the review
        # window would not open it (nothing left to answer) and the fixer
        # would not list it (already re-cut): no way out at all.
        answered, total = _completeness(bundle, stem)
        if total and answered == total:
            out["complete"].append((stem, answered, total))
            continue
        if _boundary_source_human(bundle, stem):
            reason = _routing_reason(bundle, stem)
            if _names_segmentation(reason):
                out["fixed_release"].append((stem, reason))
            else:
                out["fixed_held"].append(
                    (stem, reason or "(no routing reason recorded)"))
            continue
        if answered:
            # Partial by full coverage; maybe complete for every segment that
            # was actually ASKED (a triage-style walk). Listed for a human --
            # never auto-released, because the review file does not record
            # which unanswered segments were skipped vs never shown.
            try:
                from mousereach.review.triage_status import triage_status
                st = triage_status(bundle, stem)
                asked = set(st.triaged)
                from mousereach.review.triage_status import resolved_segments
                seg_doc = _segments_doc(bundle, stem)
                got = resolved_segments(_effective_review(bundle, stem, seg_doc), seg_doc)
                if asked and asked <= got:
                    out["partial_walk"].append((stem, answered, total, len(asked)))
                    continue
            except Exception:
                pass
            out["partial"].append((stem, answered, total))
        else:
            out["unreviewed"].append(stem)
    return out


def release_finished(queue_root: Path, cls: Optional[dict] = None):
    """Write the release marker for every releasable bundle (complete +
    fixed_release). Returns (n_released, failures) where failures is a list of
    'stem: error' strings. Held-back and partial bundles are never touched."""
    from mousereach.review.causal_review_io import (
        _get_username, _get_timestamp, _write_json,
    )
    if cls is None:
        cls = classify_queue(queue_root)
    n, failures = 0, []
    for stem, a, t in cls["complete"]:
        try:
            _write_json(Path(queue_root) / stem / f"{stem}_deep_review_cleared.json", {
                "type": "deep_review_cleared",
                "video_stem": stem,
                "cleared_by": _get_username(),
                "cleared_at": _get_timestamp(),
                "reason": "bulk release: every segment answered",
                "gated_on": "is_answered",
            })
            n += 1
        except Exception as e:
            failures.append(f"{stem}: {e}")
    for stem, r in cls["fixed_release"]:
        try:
            _write_json(Path(queue_root) / stem / f"{stem}_deep_review_cleared.json", {
                "type": "deep_review_cleared",
                "video_stem": stem,
                "cleared_by": _get_username(),
                "cleared_at": _get_timestamp(),
                "reason": "segmentation corrected by hand",
                "gated_on": "boundary_source",
                "routing_reason": r,
            })
            n += 1
        except Exception as e:
            failures.append(f"{stem}: {e}")
    return n, failures


def _segments_doc(bundle: Path, stem: str) -> Optional[dict]:
    """The bundle's segments document, or None when absent or unreadable."""
    p = Path(bundle) / f"{stem}_segments.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except Exception as err:
        logger.warning('Could not read %s, so answers count by segment number: %s', p, err)
        return None


def _effective_review(bundle: Path, stem: str, seg_doc: Optional[dict]) -> Optional[dict]:
    """The bundle's review as the review window sees it: every copy (the
    bundle's and the durable one), and for each current segment the newest
    answer or recheck a person gave about its frames
    (causal_review_io.merge_review_copies). WHY: the bundle copy alone can
    lack an answer only the durable copy still holds, or still show an answer
    a later recheck reopened. None when no copy exists."""
    from mousereach.review.causal_review_io import (merge_review_copies, review_copies,
                                                    segmentation_is_human,
                                                    segments_from_segmentation)
    current = segments_from_segmentation(seg_doc) if seg_doc else None
    doc, _ = merge_review_copies(review_copies(stem, [bundle]), current or None,
                                 human_segmentation=segmentation_is_human(seg_doc))
    return doc


def _completeness(bundle: Path, stem: str) -> Tuple[int, int]:
    """(answered, total) segments for the bundle's review (_effective_review):
    answered by ``is_answered``, matched to the CURRENT segments by frames
    when the bundle has a segments file (total = its segments), else by
    number (total = the bundle copy's records). (0, 0) when the bundle holds
    no readable review file."""
    from mousereach.review.causal_review_io import segments_from_segmentation
    from mousereach.review.triage_status import resolved_segments
    p = bundle / f"{stem}_causal_review.json"
    if not p.is_file():
        return 0, 0
    try:
        own = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0
    segs = [r for r in (own.get("segments") or []) if isinstance(r, dict)]
    seg_doc = _segments_doc(bundle, stem)
    doc = _effective_review(bundle, stem, seg_doc) or own
    current = segments_from_segmentation(seg_doc) if seg_doc else []
    if current:
        return len(resolved_segments(doc, seg_doc)), len(current)
    return len(resolved_segments(doc)), len(segs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="mousereach-review-release",
        description="List re-segmentation-queue bundles by review completeness "
                    "(every segment answered); --clear writes the release "
                    "marker for fully-answered bundles so the watcher's "
                    "return scan re-injects them.")
    ap.add_argument("--clear", action="store_true",
                    help="write {stem}_deep_review_cleared.json for every "
                         "COMPLETE bundle (every segment answered). Without "
                         "this flag, list only.")
    args = ap.parse_args(argv)

    from mousereach.config import Paths
    from mousereach.review.causal_review_io import (
        _get_username, _get_timestamp, _write_json,
    )

    queue = getattr(Paths, "DEEP_REVIEW", None)
    bundles = _bundle_dirs(queue)
    if not bundles:
        print(f"Re-segmentation queue is empty or missing: {queue}")
        return 0

    cls = classify_queue(queue)
    complete, partial = cls["complete"], cls["partial"]
    unreviewed, already = cls["unreviewed"], cls["already"]
    fixed_release, fixed_held = cls["fixed_release"], cls["fixed_held"]
    partial_walk = cls["partial_walk"]
    not_reseg = cls["not_resegmented"]

    print(f"Re-segmentation queue: {len(bundles)} bundle(s)")
    print(f"  complete (every segment answered): {len(complete)}")
    for stem, a, t in complete:
        print(f"    {stem}  {a}/{t}")
    print(f"  segmentation fixed by hand, releasable (routed for "
          f"segmentation): {len(fixed_release)}")
    for stem, r in fixed_release:
        print(f"    {stem}  [{r}]")
    print(f"  segmentation fixed by hand, HELD BACK (routed for something "
          f"a cut-fix does not address): {len(fixed_held)}")
    for stem, r in fixed_held:
        print(f"    {stem}  [{r}]")
    print(f"  answered for every triaged segment (partial walk -- human "
          f"judgement, use the Clear button): {len(partial_walk)}")
    for stem, a, t, k in partial_walk:
        print(f"    {stem}  {a}/{t} answered, all {k} triaged covered")
    print(f"  partial: {len(partial)}")
    for stem, a, t in partial:
        print(f"    {stem}  {a}/{t}")
    print(f"  no review answers yet: {len(unreviewed)}")
    print(f"  NOT re-segmented since it arrived -- cannot leave until someone "
          f"sets or confirms every cut in the segmentation fixer: {len(not_reseg)}")
    for stem, why in not_reseg:
        print(f"    {stem}")
    if already:
        print(f"  already cleared (marker present, awaiting return scan): "
              f"{len(already)}")

    if not args.clear:
        if complete or fixed_release:
            print("\nRun again with --clear to release the complete and "
                  "fixed-releasable bundle(s). The watcher's return scan "
                  "re-injects them over the following cycles.")
        return 0

    n, failures = release_finished(Path(queue), cls)
    for f in failures:
        print(f"[FAIL] {f}")
    print(f"\nReleased {n} of {len(complete) + len(fixed_release)} "
          f"releasable bundle(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
