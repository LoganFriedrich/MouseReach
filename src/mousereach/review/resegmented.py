"""Has a person re-segmented this video since it was sent to deep review?

Deep review is where a video's cuts are fixed. A video may leave it only once
a person has called every boundary -- confirmed or moved, in the segmentation
fixer or a ground-truth file -- AFTER the video last entered the queue (Logan,
2026-10-27: "if it wasn't resegmented (if there isn't a human called boundary
for every boundary somewhere) it can't leave deep review"). Calls made before
it entered do not count: a video sent back to deep review is there because
those very cuts were found wrong. Without this rule a clear marker alone
released a video, and an old clear from an earlier deep review sent an
escalated video straight back to triage on the same cuts
(20250627_CNT0107_P3, 2026-10-27).

Where a person's boundary calls are recorded:
  - ``{stem}_segments.json`` saved by the segmentation fixer:
    ``boundary_source == "human"`` and ``corrected_at``. The fixer saves the
    whole list of cuts the person accepted, so every boundary is theirs.
  - a ground-truth file (``{stem}*ground_truth*.json``) whose
    ``segmentation.boundaries`` are ALL ``verified``, each with ``verified_at``.

When the video entered the queue is ``routed_at`` in ``{stem}_routing.json``,
written by every route into a queue. A bundle without one (older routes) is
judged on the boundary calls alone.

Easy and Flat trays are never segmented (Logan, 2026-10-23) and are not
subject to this rule; callers check ``is_outcome_free_tray`` first.
"""
import json
from pathlib import Path
from typing import Optional, Tuple

import logging
logger = logging.getLogger(__name__)


def _read(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as err:
        logger.warning('Could not read %s, so it counts as absent: %s', path, err)
        return None


def entered_deep_review_at(bundle: Path, stem: str) -> Optional[str]:
    """ISO time the bundle last entered its queue, or None if unrecorded."""
    doc = _read(Path(bundle) / f"{stem}_routing.json")
    return str(doc["routed_at"]) if doc and doc.get("routed_at") else None


def resegmented_since_entry(bundle: Path, stem: str) -> Tuple[bool, str]:
    """(True, how) when a person called every boundary after the video entered
    deep review; (False, why not) otherwise -- the reason is worded for the
    person who has to act on it."""
    bundle = Path(bundle)
    entered = entered_deep_review_at(bundle, stem)

    def after_entry(when) -> bool:
        return entered is None or (when is not None and str(when) >= entered)

    seg = _read(bundle / f"{stem}_segments.json")
    if seg and seg.get("boundary_source") == "human":
        if after_entry(seg.get("corrected_at")):
            return True, "cuts saved by %s at %s" % (seg.get("corrected_by"),
                                                      seg.get("corrected_at"))
        stale = ("its cuts were last set by hand at %s, before it was sent "
                 "here at %s" % (seg.get("corrected_at"), entered))
    else:
        stale = "nobody has set its cuts by hand"

    for gt_path in sorted(bundle.glob(f"{stem}*ground_truth*.json")):
        gt = _read(gt_path) or {}
        bounds = (gt.get("segmentation") or {}).get("boundaries") or []
        if bounds and all(isinstance(b, dict) and b.get("verified") for b in bounds):
            latest = max(str(b.get("verified_at") or "") for b in bounds) or None
            if after_entry(latest):
                return True, "every boundary verified in %s" % gt_path.name

    return False, ("not re-segmented: %s. Open it in the segmentation fixer, "
                   "set or confirm every cut and save; it leaves the queue "
                   "after that." % stale)
