"""Has a person re-segmented this video since it was sent to deep review?

Deep review is where a video's cuts are fixed. A video may leave it only once
a person has set or confirmed every boundary in the segmentation fixer AFTER
the video last entered the queue: a video that was not re-segmented cannot
leave deep review. Calls made before it entered do not count: a video sent
back to deep review is there because those very cuts were found wrong.
Without this rule a clear marker alone released a video, and an old clear
from an earlier deep review sent an escalated video straight back to triage
on the same cuts.

Where a person's boundary calls are recorded: ``{stem}_segments.json`` saved
by the segmentation fixer, with ``boundary_source == "human"`` and
``corrected_at``. The fixer saves the whole list of cuts the person accepted,
and that file is the one the pipeline cuts the video by on the re-run.

A ground-truth file is NOT proof of re-segmentation (it used to be, when every
one of its boundaries was determined). WHY: nothing writes ground-truth
boundaries into the segments file, so on the re-run the segmenter cut the
video again exactly as before and everything after it was computed on the
cuts the person had replaced. Whether ground-truth boundaries should be
written through as hand cuts is an open decision; until it is made, cuts
are set in the fixer.

When the video entered the queue is ``routed_at`` in ``{stem}_routing.json``,
written by every route into a queue. A bundle without one (older routes) is
judged on the boundary calls alone.

Easy and Flat trays are never segmented and are not
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

    return False, ("not re-segmented: %s. Open it in the segmentation fixer, "
                   "set or confirm every cut and save; it leaves the queue "
                   "after that." % stale)


def needs_recut(bundle: Path, stem: str) -> Tuple[bool, str]:
    """(True, why) when this re-segmentation bundle can leave only after a
    person sets or confirms its cuts, and nobody has done that since it
    arrived; (False, why not) otherwise.

    WHY one question for the segmentation fixer's list: every way out of the
    queue for a pillar video requires the proof above (the return scan, the
    Clear button, the release button). The fixer used to list only bundles
    that came for a segmentation reason and whose cuts were never set by
    hand. Two kinds of bundle then had no way out at all: one whose cuts a
    person had set on an EARLIER visit and that was sent back (its old hand
    cuts hid it, yet they no longer count), and one sent for another reason,
    such as a flagged session or a quality hold (no segmentation reason, so
    never listed, yet the proof is still required). The fixer now lists
    exactly the bundles this says need it, so each has a visible next step.

    Easy and Flat trays are never segmented and never need this."""
    from mousereach.config import is_outcome_free_tray
    if is_outcome_free_tray(stem):
        return False, "Easy/Flat tray: not segmented"
    ok, why = resegmented_since_entry(bundle, stem)
    if ok:
        return False, why
    return True, why
