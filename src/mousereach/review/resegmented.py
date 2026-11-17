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
written by every route into a queue. A bundle whose ``routed_at`` cannot be
read is dated by entered_deep_review_at's fallbacks (never "any time ever"),
and one that cannot be dated at all cannot leave until that is put right.

Easy and Flat trays are never segmented and are not
subject to this rule; callers check ``is_outcome_free_tray`` first.
"""
import json
import os
from datetime import datetime
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


def _epoch(text) -> Optional[float]:
    """Seconds since the epoch for an ISO time written by this package
    (``datetime.now().isoformat()``), or None when it cannot be read."""
    try:
        return datetime.fromisoformat(str(text)).timestamp()
    except (TypeError, ValueError):
        return None


def _created(path: Path) -> Optional[float]:
    """When ``path`` was created, where the file system records it:
    ``st_birthtime`` where it exists, else ``st_ctime`` on Windows (creation
    time there). Elsewhere ``st_ctime`` is the last metadata change, not the
    creation, so it is not used."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    t = getattr(st, "st_birthtime", None)
    if t is None and os.name == "nt":
        t = st.st_ctime
    return t


def entered_deep_review_at(bundle: Path, stem: str) -> Optional[str]:
    """ISO time the bundle last entered its queue, or None if it cannot be
    told at all.

    ``routed_at`` in ``{stem}_routing.json`` when it can be read. Otherwise
    the LATEST of: the routing record's modified time (it is written when the
    video is routed), the review manifest's ``provenance.staged_at`` (written
    when the bundle is staged), and the bundle folder's creation time.

    WHY a fallback, and why the latest: with no readable ``routed_at`` the
    rule used to accept ANY hand cut ever saved -- including the very cuts a
    video was sent back here to have fixed. Each fallback is a time at or
    after the video arrived, so the latest of them is the safe side: a cut
    saved after it is certainly a new one, and at worst a person confirms
    the cuts once more."""
    bundle = Path(bundle)
    routing = bundle / f"{stem}_routing.json"
    doc = _read(routing)
    if doc and _epoch(doc.get("routed_at")) is not None:
        return str(doc["routed_at"])
    times = []
    try:
        times.append(routing.stat().st_mtime)
    except OSError:
        pass
    for name in (f"{stem}_manifest.json", "manifest.json"):
        man = _read(bundle / name)
        if man:
            t = _epoch((man.get("provenance") or {}).get("staged_at"))
            if t is not None:
                times.append(t)
            break
    t = _created(bundle)
    if t is not None:
        times.append(t)
    if not times:
        return None
    return datetime.fromtimestamp(max(times)).isoformat()


def resegmented_since_entry(bundle: Path, stem: str) -> Tuple[bool, str]:
    """(True, how) when a person called every boundary after the video entered
    deep review; (False, why not) otherwise -- the reason is worded for the
    person who has to act on it."""
    bundle = Path(bundle)
    entered = entered_deep_review_at(bundle, stem)
    if entered is None:
        return False, ("not re-segmented: when this video arrived in the queue "
                       "cannot be told (no routing record, review manifest or "
                       "folder date could be read), so no saved cuts can be "
                       "shown to be new. Check that its folder can be read, "
                       "then set or confirm every cut in the segmentation "
                       "fixer and save.")
    entered_t = _epoch(entered)

    def after_entry(when) -> bool:
        t = _epoch(when)
        return t is not None and t >= entered_t

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
