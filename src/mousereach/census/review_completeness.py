"""Measure how much of a human review is actually done.

WHY THIS EXISTS -- READ BEFORE CHANGING ANYTHING HERE
-----------------------------------------------------
The obvious field to check is ``answers.reviewed`` on each segment. It is WRONG.

Measured across 1,982 review documents on a live corpus: ``answers.reviewed`` is
False on EVERY segment of EVERY file. Nothing in the codebase has ever written it.
Any release condition, dashboard count or test keyed on that field reports "no
reviews exist" forever, no matter how much work a reviewer has done.

The reviewer's actual work lands in ``human.outcome`` and ``human.causal_reach``.
On the same corpus 1,982 files carry ``human.outcome`` on at least one segment and
1,900 carry ``human.causal_reach``.

So: completeness = segments with a non-null ``human.outcome``, over total segments.

A review document can exist in two places -- inside the queue bundle, and in the
durable review store. They can disagree, because a bundle can be re-staged after a
review was saved. Take the MORE COMPLETE of the two; a reviewer's work is not undone
by the bundle being rebuilt.

No lab paths appear in this module. All roots are parameters.
"""
from __future__ import annotations

import json
from pathlib import Path


REVIEW_SUFFIX = "_causal_review.json"


def _read(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def completeness_of_document(doc):
    """(answered, total, reviewer, reviewed_at) for one review document.

    ``answered`` counts segments where a human recorded an outcome. Returns None if
    the document has no segment list at all.
    """
    if not isinstance(doc, dict):
        return None
    segments = doc.get("segments") or []
    if not segments:
        return None
    answered = sum(
        1 for s in segments
        if isinstance(s, dict) and (s.get("human") or {}).get("outcome") is not None
    )
    return (answered, len(segments), doc.get("reviewer"), doc.get("reviewed_at"))


def completeness_for_stem(stem, bundle_dir=None, durable_dir=None):
    """Best-known completeness for one video, across both storage locations.

    ``bundle_dir`` is the queue bundle directory for this video (may not exist).
    ``durable_dir`` is the shared review store directory (may not exist).
    Returns a dict, or None when no review document exists anywhere.
    """
    candidates = []
    if bundle_dir:
        candidates.append(Path(bundle_dir) / (stem + REVIEW_SUFFIX))
    if durable_dir:
        candidates.append(Path(durable_dir) / (stem + REVIEW_SUFFIX))

    best = None
    for path in candidates:
        if not path.exists():
            continue
        result = completeness_of_document(_read(path))
        if result is None:
            continue
        answered, total, reviewer, when = result
        if best is None or answered > best["answered"]:
            best = {
                "stem": stem,
                "answered": answered,
                "total": total,
                "reviewer": reviewer,
                "reviewed_at": when,
                "source": str(path),
            }
    if best is not None:
        best["complete"] = best["total"] > 0 and best["answered"] == best["total"]
    return best


def scan_queue(queue_dir, durable_dir=None):
    """Completeness for every bundle in a review queue directory.

    Returns (rows, skipped). ``rows`` is one dict per bundle that has a review
    document; ``skipped`` names bundles with none.

    WHY dot-directories are skipped: tool and agent scratch directories land inside
    queue folders and are otherwise counted as phantom videos awaiting review.
    """
    queue_dir = Path(queue_dir)
    rows, skipped = [], []
    if not queue_dir.exists():
        return rows, skipped
    for d in sorted(queue_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        row = completeness_for_stem(d.name, bundle_dir=d, durable_dir=durable_dir)
        if row is None:
            skipped.append(d.name)
        else:
            rows.append(row)
    return rows, skipped


def summarise(rows):
    """Counts a dashboard can display directly."""
    complete = [r for r in rows if r["complete"]]
    partial = [r for r in rows if not r["complete"] and r["answered"] > 0]
    untouched = [r for r in rows if r["answered"] == 0]
    return {
        "with_review_document": len(rows),
        "complete": len(complete),
        "partial": len(partial),
        "untouched": len(untouched),
        "complete_stems": [r["stem"] for r in complete],
    }
