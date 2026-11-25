"""Measure how much of a human review is actually done.

WHY THIS EXISTS -- READ BEFORE CHANGING ANYTHING HERE
-----------------------------------------------------
A segment is answered when ``causal_review_io.is_answered`` says so -- the one
definition every check of a person's answer uses: not a ``reviewed: False``
record (a segment the reviewer never reached, a pellet left with no outcome
picked, or a pellet sent back with "are you sure?"), and for a pellet an
outcome other than "triaged", "uncertain" or none.

WHY not ``human.outcome is not None``, the test this module used before: a
pellet left with no outcome picked is saved with the algorithm's "triaged" in
its outcome, and a pellet sent back for a recheck keeps its old outcome. Both
are open questions, and the old test counted both as answered, so this census
could call a video complete that the release check holds.

So: completeness = answered segments, over total segments.

A review document can exist in two places -- inside the queue bundle, and in the
durable review store. They can disagree, because a bundle can be re-staged after a
review was saved, and a recheck ("are you sure?") reopens a pellet in one copy only.
Each segment is judged on the NEWEST statement a person made about it in either
copy (``causal_review_io.merge_review_copies``, the rule the review window uses):
an answer in one copy is not undone by a placeholder in the other, and a later
recheck reopens it. (Taking the copy with MORE answers instead would count a
reopened pellet as answered here while the release check held it.)

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

    ``answered`` counts segments a person answered (``is_answered``). Returns
    None if the document has no segment list at all.
    """
    from mousereach.review.causal_review_io import is_answered
    if not isinstance(doc, dict):
        return None
    segments = doc.get("segments") or []
    if not segments:
        return None
    answered = sum(
        1 for s in segments
        if isinstance(s, dict) and is_answered(s)
    )
    return (answered, len(segments), doc.get("reviewer"), doc.get("reviewed_at"))


def completeness_for_stem(stem, bundle_dir=None, durable_dir=None):
    """Best-known completeness for one video, across both storage locations.

    ``bundle_dir`` is the queue bundle directory for this video (may not exist).
    ``durable_dir`` is the shared review store directory (may not exist).
    Returns a dict, or None when no review document exists anywhere.
    """
    from mousereach.review.causal_review_io import is_answered, merge_review_copies
    candidates = []
    if bundle_dir:
        candidates.append(Path(bundle_dir) / (stem + REVIEW_SUFFIX))
    if durable_dir:
        candidates.append(Path(durable_dir) / (stem + REVIEW_SUFFIX))

    found = []
    for path in candidates:
        if not path.exists():
            continue
        doc = _read(path)
        if completeness_of_document(doc) is None:
            continue
        found.append((path, doc))
    if not found:
        return None
    merged, _ = merge_review_copies([doc for _, doc in found])
    newest_path, newest = max(found, key=lambda f: str(f[1].get("reviewed_at") or ""))
    total = max(len(doc.get("segments") or []) for _, doc in found)
    answered = sum(1 for r in (merged or {}).get("segments") or [] if is_answered(r))
    return {
        "stem": stem,
        "answered": answered,
        "total": total,
        "reviewer": newest.get("reviewer"),
        "reviewed_at": newest.get("reviewed_at"),
        "source": str(newest_path),
        "complete": total > 0 and answered == total,
    }


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
