"""Bulk release check for deep-review bundles -- a human-invoked CLI.

``mousereach-review-release`` lists every bundle in the deep-review queue with
its review completeness, and (with ``--clear``) writes the release marker for
the complete ones. It exists because completing a review and releasing the
bundle are two separate acts in the review tool, and the second one was being
skipped: bundles sat in the queue with every segment answered and no marker,
so the watcher's return scan never picked them up.

Completeness is judged on ``human.outcome`` -- the field reviewers actually
fill -- NEVER on ``answers.reviewed``, which nothing in the codebase has ever
written (it is False on all 1,982 review files as of 2026-09-01; a release
condition keyed on it would fire never).

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
    for bundle in bundles:
        stem = bundle.name
        if (bundle / f"{stem}_deep_review_cleared.json").is_file():
            already.append(stem)
            continue
        answered, total = _completeness(bundle, stem)
        if total and answered == total:
            complete.append((stem, answered, total))
        elif answered:
            partial.append((stem, answered, total))
        else:
            unreviewed.append(stem)

    print(f"Deep-review queue: {len(bundles)} bundle(s)")
    print(f"  complete (every segment has a human outcome): {len(complete)}")
    for stem, a, t in complete:
        print(f"    {stem}  {a}/{t}")
    print(f"  partial: {len(partial)}")
    for stem, a, t in partial:
        print(f"    {stem}  {a}/{t}")
    print(f"  no review answers yet: {len(unreviewed)}")
    if already:
        print(f"  already cleared (marker present, awaiting return scan): "
              f"{len(already)}")

    if not args.clear:
        if complete:
            print("\nRun again with --clear to release the complete "
                  "bundle(s). The watcher's return scan re-injects them "
                  "within one cycle.")
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
    print(f"\nReleased {n} of {len(complete)} complete bundle(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
