"""
before_first_cut.py - a reach that starts before the first cut belongs to NO pellet.

WHY (owner decision D2-3, 2026-11-19): "a reach that starts before the first
cut belongs to NO pellet." The first cut is where the first pellet's window
begins; before it no pellet is presented, so a reach then cannot have touched
pellet 1. The reach step used to attach such a reach to the NEAREST segment
(always segment 1) and flag that segment ("reach_start_outside_all_segments")
-- a flag nothing read, so the reach was counted, measured and exported as one
of pellet 1's reaches.

Now every place that maps reaches to pellets keeps it out:

  * the reach step writes it into the reaches file's own top-level list,
    ``reaches_before_first_cut`` (never inside a segment), so outcome
    detection, reach assignment and kinematics never see it as a pellet's;
  * a reaches file written before this (the reach sits inside segment 1) is
    read through ``split_before_first_cut``, which moves it to that list, by
    every reader that maps reaches to pellets (outcome detection, reach
    assignment, the truth layering, kinematics);
  * the features file lists it under ``reaches_before_first_cut`` (its frames,
    no pellet, no measurements), so the file still says the reach happened.

Which reach: one whose START frame is before the first cut -- the rule the reach
step uses to place every reach (a reach belongs to the segment holding its
start frame). A reach starting before the cut and ending after it is still
before the first cut.

Plain dicts only (no pose data), so every reader can use it.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

BEFORE_FIRST_CUT_KEY = "reaches_before_first_cut"

# The flag the reach step used to put on segment 1 for such a reach.
OLD_FLAG_REASON = "reach_start_outside_all_segments"


def first_cut(doc: Optional[Dict[str, Any]]) -> Optional[int]:
    """The first cut of a reaches (or features) file: the first frame of its
    lowest-numbered segment. None when no segment carries a start frame."""
    best = None
    for seg in (doc or {}).get("segments") or []:
        try:
            num, start = int(seg.get("segment_num")), int(seg.get("start_frame"))
        except (TypeError, ValueError):
            continue
        if best is None or num < best[0]:
            best = (num, start)
    return None if best is None else best[1]


def starts_before(reach: Dict[str, Any], cut: Optional[int]) -> bool:
    """True when ``reach`` starts before frame ``cut`` (no cut: False)."""
    if cut is None:
        return False
    try:
        return int(reach.get("start_frame")) < int(cut)
    except (TypeError, ValueError, AttributeError):
        return False


_PERSON = ("human_review", "ground_truth")


def person_credited(reach: Dict[str, Any]) -> bool:
    """True for a features-file reach a PERSON credited as its pellet's causal
    reach (causal_reach set, and the credit is a person's: causal_reach_source
    human_review / ground_truth, or -- older files, which have no
    causal_reach_source -- the pellet's outcome_source is a person's).

    WHY it stays with its pellet: what a person said is correct (owner
    decision 2026-11-19). A person who named a reach starting a few frames
    before the first cut as pellet 1's causal reach has said that reach is
    pellet 1's. The kinematics step keeps that answer too: the person's pick
    is matched by its frames, and with the reach out of segment 1 it is added
    back as the person's reach (truth_resolver, a reach the detector missed)."""
    if not reach.get("causal_reach"):
        return False
    src = reach.get("causal_reach_source")
    if src is not None:
        return src in _PERSON
    return reach.get("outcome_source") in _PERSON


def split_before_first_cut(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A copy of a reaches file in which every reach that starts before the
    first cut sits in the top-level ``reaches_before_first_cut`` list and in
    no segment (module docstring says why). A file that already has no such
    reach inside a segment is returned as it is (not copied).

    A reach a person credited as its pellet's causal reach stays where it is
    (person_credited says why). The moved reaches keep their ids and frames
    and lose their reach number and segment; the segment they leave is renumbered 1..n and its count
    updated, and the old flag for them is taken off it when nothing else
    flagged it."""
    if not doc:
        return doc
    cut = first_cut(doc)
    if cut is None:
        return doc
    if not any(starts_before(r, cut) and not person_credited(r)
               for seg in doc.get("segments") or [] for r in (seg.get("reaches") or [])):
        return doc
    out = copy.deepcopy(doc)
    moved: List[Dict[str, Any]] = list(out.get(BEFORE_FIRST_CUT_KEY) or [])
    for seg in out.get("segments") or []:
        keep = []
        for r in seg.get("reaches") or []:
            if starts_before(r, cut) and not person_credited(r):
                r = dict(r)
                r.pop("reach_num", None)
                r.pop("segment_num", None)
                if r.get("review_note") == OLD_FLAG_REASON:
                    r.pop("review_note", None)
                moved.append(r)
            else:
                keep.append(r)
        if len(keep) != len(seg.get("reaches") or []):
            for i, r in enumerate(keep, start=1):
                if "reach_num" in r:
                    r["reach_num"] = i
                # A features file's per-reach context, kept true for the
                # pellet's own reaches.
                if "is_first_reach" in r:
                    r["is_first_reach"] = (i == 1)
                if "is_last_reach" in r:
                    r["is_last_reach"] = (i == len(keep))
                if "n_reaches_in_segment" in r:
                    r["n_reaches_in_segment"] = len(keep)
            seg["reaches"] = keep
            seg["n_reaches"] = len(keep)
            if seg.get("flag_reason") == OLD_FLAG_REASON:
                seg["flagged_for_review"] = False
                seg["flag_reason"] = None
    moved.sort(key=lambda r: (int(r.get("start_frame") or 0)))
    out[BEFORE_FIRST_CUT_KEY] = moved
    if "segments_flagged" in out:
        out["segments_flagged"] = sum(1 for s in out.get("segments") or []
                                      if s.get("flagged_for_review"))
    return out


def reaches_before_first_cut(doc: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every reach of a reaches file that starts before its first cut: the
    top-level list plus any still inside a segment (older files)."""
    return list((split_before_first_cut(doc) or {}).get(BEFORE_FIRST_CUT_KEY) or [])
