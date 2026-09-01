"""Bulk release check for deep-review bundles -- a human-invoked CLI.

``mousereach-review-release`` lists every bundle in the deep-review queue with
its review completeness, and (with ``--clear``) writes the release marker for
the complete ones. It exists because completing a review and releasing the
bundle are two separate acts in the review tool, and the second one was being
skipped: bundles sat in the queue with every segment answered and no marker,
so the watcher's return scan never picked them up.

Completeness is judged on ``human.outcome`` per segment. (An earlier version
of this docstring claimed ``answers.reviewed`` was unusable; measured over
every segment record the truth is narrower: unanswered placeholder records
carry ``reviewed: False`` while answered records omit the key entirely, with
zero disagreements against ``human.outcome``. A check written as
``is not False`` therefore works too. ``human.outcome`` stays the criterion
here because it is the substance of the review, not a bookkeeping flag.)

A hand-corrected segmentation (``boundary_source == "human"`` in the bundle's
segments file) also finishes a bundle -- WHEN the bundle was routed for a
segmentation problem. The routing reason gates that: the clear marker is a
blanket human-clear token the watcher gate honors, so releasing a bundle
routed for e.g. a QC hold on the strength of a cut-fix would clear a concern
nobody addressed. Such bundles are listed as held back, with their reason.

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
    except Exception:
        return False


def _bundle_dirs(queue_root: Optional[Path]) -> List[Path]:
    if not queue_root or not Path(queue_root).exists():
        return []
    return sorted(d for d in Path(queue_root).iterdir()
                  if d.is_dir() and not d.name.startswith("."))


def _completeness(bundle: Path, stem: str) -> Tuple[int, int]:
    """(answered, total) segments for the bundle's causal review, by
    human.outcome. (0, 0) when no review file exists."""
    p = bundle / f"{stem}_causal_review.json"
    if not p.is_file():
        return 0, 0
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0
    segs = [r for r in (doc.get("segments") or []) if isinstance(r, dict)]
    answered = sum(1 for r in segs
                   if (r.get("human") or {}).get("outcome") is not None)
    return answered, len(segs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="mousereach-review-release",
        description="List deep-review bundles by review completeness "
                    "(human.outcome per segment); --clear writes the release "
                    "marker for fully-answered bundles so the watcher's "
                    "return scan re-injects them.")
    ap.add_argument("--clear", action="store_true",
                    help="write {stem}_deep_review_cleared.json for every "
                         "COMPLETE bundle (all segments carry a human "
                         "outcome). Without this flag, list only.")
    args = ap.parse_args(argv)

    from mousereach.config import Paths
    from mousereach.review.causal_review_io import (
        _get_username, _get_timestamp, _write_json,
    )

    queue = getattr(Paths, "DEEP_REVIEW", None)
    bundles = _bundle_dirs(queue)
    if not bundles:
        print(f"Deep-review queue is empty or missing: {queue}")
        return 0

    complete, partial, unreviewed, already = [], [], [], []
    fixed_release, fixed_held, partial_walk = [], [], []
    for bundle in bundles:
        stem = bundle.name
        if (bundle / f"{stem}_deep_review_cleared.json").is_file():
            already.append(stem)
            continue
        if _boundary_source_human(bundle, stem):
            reason = _routing_reason(bundle, stem)
            if _names_segmentation(reason):
                fixed_release.append((stem, reason))
            else:
                fixed_held.append((stem, reason or "(no routing reason recorded)"))
            continue
        answered, total = _completeness(bundle, stem)
        if total and answered == total:
            complete.append((stem, answered, total))
        elif answered:
            # Partial by full-coverage; maybe complete for every segment that
            # was actually ASKED (a triage-style walk). Listed for a human --
            # never auto-released, because the review file does not record
            # which unanswered segments were skipped vs never shown.
            try:
                from mousereach.review.triage_status import triage_status
                st = triage_status(bundle, stem)
                asked = set(st.triaged)
                doc = json.loads((bundle / f"{stem}_causal_review.json")
                                 .read_text(encoding="utf-8"))
                got = {r.get("segment_num") for r in doc.get("segments", [])
                       if isinstance(r, dict)
                       and (r.get("human") or {}).get("outcome") is not None}
                if asked and asked <= got:
                    partial_walk.append((stem, answered, total, len(asked)))
                    continue
            except Exception:
                pass
            partial.append((stem, answered, total))
        else:
            unreviewed.append(stem)

    print(f"Deep-review queue: {len(bundles)} bundle(s)")
    print(f"  complete (every segment has a human outcome): {len(complete)}")
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
    if already:
        print(f"  already cleared (marker present, awaiting return scan): "
              f"{len(already)}")

    if not args.clear:
        if complete or fixed_release:
            print("\nRun again with --clear to release the complete and "
                  "fixed-releasable bundle(s). The watcher's return scan "
                  "re-injects them over the following cycles.")
        return 0

    n = 0
    for stem, a, t in complete:
        bundle = Path(queue) / stem
        try:
            _write_json(bundle / f"{stem}_deep_review_cleared.json", {
                "type": "deep_review_cleared",
                "video_stem": stem,
                "cleared_by": _get_username(),
                "cleared_at": _get_timestamp(),
                "reason": "bulk release: every segment carries a human "
                          "outcome",
                "gated_on": "human.outcome",
            })
            n += 1
            print(f"[OK] released {stem} ({a}/{t})")
        except Exception as e:
            print(f"[FAIL] {stem}: {e}")
    for stem, r in fixed_release:
        bundle = Path(queue) / stem
        try:
            _write_json(bundle / f"{stem}_deep_review_cleared.json", {
                "type": "deep_review_cleared",
                "video_stem": stem,
                "cleared_by": _get_username(),
                "cleared_at": _get_timestamp(),
                "reason": "segmentation corrected by hand",
                "gated_on": "boundary_source",
                "routing_reason": r,
            })
            n += 1
            print(f"[OK] released {stem} (hand-fixed segmentation)")
        except Exception as e:
            print(f"[FAIL] {stem}: {e}")
    print(f"\nReleased {n} of {len(complete) + len(fixed_release)} "
          f"releasable bundle(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
