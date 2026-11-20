"""
mousereach.review.truth_resolver -- per-element human-truth layering for the
production kinematics output.

Every element of a video (each reach; each per-segment outcome) is resolved
INDEPENDENTLY by taking the HIGHEST layer that has a call:

    GT  >  causal review  >  triage review  >  algo

Any element no human layer touched keeps its algo value ("unconflicted algo
persists"). This is the single place production kinematics obtains its truth --
the feature extractor calls resolve_truth_layers() instead of applying causal
review alone, so that GROUND TRUTH (and GT reach boundaries, which review never
touches) also flow into the shipped numbers.

Only what a person DETERMINED in the ground-truth file is ground truth: a reach
whose start and end were both determined (truth_resolver.gt_reach_is_determined),
an outcome marked determined. The tool also saves the algorithm's own items it
was seeded with; those are never applied as ground truth.

GT is whole-COMPONENT authoritative only where that component's `exhaustive`
flag is set (GT stores `exhaustive` separately for segmentation / reaches /
outcomes):
  - reaches.exhaustive == True  -> the GT reach set is COMPLETE. Each segment's
       reaches ARE the GT reaches; algo reaches GT never labeled are false
       positives and are dropped. GT reaches flagged exclude_from_analysis drop.
       A segment still holding an undetermined GT reach is not finished: there
       only its determined reaches are merged, as in the non-exhaustive case.
  - reaches.exhaustive == False -> GT is positive-only: it overrides the algo
       reach it overlaps and adds any GT reach with no algo match, but does NOT
       drop unlabeled algo reaches (absence is not a reliable negative for
       supplementary GT).
  - outcomes.exhaustive is analogous for per-segment outcomes (GT overrides the
       segments it determined; non-exhaustive leaves undetermined segments algo).

Provenance is stamped on every element:
    outcome_source in {"algo","human_review","ground_truth"}                (per segment)
    reach_source   in {"algo","ground_truth","human_review"}                (per reach)

A reviewer's typed causal reach (the review window's "Undetected reach (enter
frames manually)") that no detected reach covers (causal_review_io.
typed_reach_match) is a reach the detector missed. Since 2026-10-26 it joins the
segment's reaches exactly as a supplementary GT reach does -- it replaces the
detected reach it overlaps, if any, and its kinematics are computed over the
frames the person typed -- and it is the segment's causal reach. Before, it was
resolved only as a pointer into the detected reaches, so its answer was lost or
landed on a neighbouring reach.

Non-destructive (deep copies) and never raises on missing layers: a video with
no human input resolves to pure algo. ASCII-only logging (Windows cp1252).

Where the human answers come from: ONE read of every copy of the video's
review -- the file or folder the caller names (primary_dir), the triage and
re-segmentation queue bundles, the video's filed folder and the durable store
-- merged per segment to the newest answer (or recheck that reopened one) a
person gave about its frames (causal_review_io.effective_review, the same read
the review window, the release check and the census use). WHY one read: the
copies used to be layered by WHERE they sat (a queue bundle outranking the
durable copy), and the filed folder's copy was not read at all, so an answer
that lived only in a copy this step did not prefer was not applied, and an
older answer in a queue bundle beat a newer one elsewhere. Triage and
full-video reviews write the same schema; both are "human_review".
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # keep the touched-outcome set consistent with the review bridge
    from mousereach.review.causal_review_io import _TOUCHED_OUTCOMES
except Exception:  # pragma: no cover - defensive
    _TOUCHED_OUTCOMES = {"retrieved", "displaced_sa", "displaced_outside"}

# The pipeline's one frame convention: a window is its first and last frame,
# both included (causal_review_io.reach_file_segment_span states it).
from mousereach.review.causal_review_io import (reach_duration_frames,
                                                reach_file_segment_span)


def _reach_file_pellet(seg: dict) -> dict:
    """A reaches-file segment as a current pellet, for a run with no segments
    file: its frames in the one convention (last frame included, as
    pellet_span gives them), never the stored end, which is the next
    segment's first frame. Without cuts there is no re-cut to honour, so
    ``cuts_set_at`` is not carried."""
    out = {k: v for k, v in seg.items() if k != "cuts_set_at"}
    try:
        out["start_frame"], out["end_frame"] = reach_file_segment_span(seg)
    except (TypeError, ValueError, KeyError):
        pass
    return out


# ---------------------------------------------------------------------------
# small IO helpers
# ---------------------------------------------------------------------------
def _read_json(p) -> Optional[dict]:
    try:
        with open(p) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as err:
        logger.error('Could not read %s, so any human answers in it are ignored: %s', p, err)
        return None


# ---------------------------------------------------------------------------
# What a triage answer says about the reaches themselves
# ---------------------------------------------------------------------------
# Owner rules 2026-11-19: (1) a causal answer includes the person checking that
# reach's start and end, so the frames saved with it are a person's frames for
# that reach; (2) naming the causal reach of a segment also answers every other
# reach of that segment -- they all missed ("untouched": every reach missed).
# The triage window saves both on the segment's record
# (causal_review_io.REACH_CALLS_KEY / CAUSAL_BOUNDS_KEY):
#
#   human.causal_reach_bounds = {"start", "end"}    the causal reach's frames
#                                                   as the person checked them
#   human.reach_calls = [{"reach_id", "start", "end", "call": "causal"|"miss",
#                         "detected", "bounds": "human"|"detected"|"typed"}]
#
# They are read here so the results use them: the causal reach is measured
# over the person's frames, and every reach the record calls a miss carries a
# person's "missed" (feature_extractor: reach_outcome / reach_outcome_source).
# Older records carry neither field; for them the rule "named causal reach =>
# every other reach missed" still gives the misses (reach_outcome_of), and the
# causal reach keeps the detector's frames.
CAUSAL_BOUNDS_FIELD = "causal_reach_bounds"
REACH_CALLS_FIELD = "reach_calls"


def person_reach_bounds(rec) -> Optional[Tuple[int, int]]:
    """(start, end) a person checked for the causal reach of ``rec`` (a review
    segment record): ``human.causal_reach_bounds``, else the ``reach_calls``
    entry called causal whose bounds are the person's. None when the record
    holds neither (older records)."""
    human = (rec or {}).get("human") or {}
    src = human.get(CAUSAL_BOUNDS_FIELD)
    if not isinstance(src, dict):
        src = next((c for c in human.get(REACH_CALLS_FIELD) or []
                    if isinstance(c, dict) and c.get("call") == "causal"
                    and c.get("bounds") == "human"), None)
    if not src:
        return None
    try:
        s, e = int(src["start"]), int(src.get("end", src["start"]))
    except (KeyError, TypeError, ValueError):
        return None
    return (min(s, e), max(s, e))


def person_missed_reaches(rec) -> List[Tuple[int, int]]:
    """The frames of every reach ``rec``'s ``human.reach_calls`` calls a
    miss ([] when it calls none, or the record is older than the field)."""
    out = []
    for c in ((rec or {}).get("human") or {}).get(REACH_CALLS_FIELD) or []:
        if not isinstance(c, dict) or c.get("call") != "miss":
            continue
        try:
            out.append((int(c["start"]), int(c.get("end", c["start"]))))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(set(out))


# ---------------------------------------------------------------------------
# per-segment OUTCOME layers
# ---------------------------------------------------------------------------
def _seg_overrides_from_review(doc: Optional[dict], source: str,
                               current_segments=None,
                               human_segmentation: bool = False) -> Dict[int, dict]:
    """{segment_num: override} from a causal-review doc. An override carries the
    human outcome, the causal reach id (only for touched outcomes), any abnormal
    ignore-ranges, and the provenance source label.

    Keys are numbers in the CURRENT segmentation. Pass ``current_segments`` and
    each review is matched to the frames the reviewer actually saw; without it,
    matching falls back to the segment number the review was written against,
    which a re-cut can have reassigned to different footage.
    """
    from mousereach.review.causal_review_io import (MAX_TYPED_REACH_FRAMES,
                                                    index_review_by_segment, is_answered,
                                                    typed_reach_match)

    out: Dict[int, dict] = {}
    matched, notes = index_review_by_segment(doc, current_segments,
                                             human_segmentation=human_segmentation)
    for n in notes:
        logger.warning("review re-anchoring: %s", n)
    for sn, rec in matched.items():
        ans = rec.get("answers") or {}
        # Only a person's ANSWER counts (is_answered): not a reviewed:False
        # placeholder, and not a pellet saved with its outcome left on the
        # algorithm's "triaged" -- which was applied here as a human answer
        # until 2026-10-26, the same gap apply_review_overrides had.
        if not is_answered(rec):
            continue
        human = rec.get("human") or {}
        ho = human.get("outcome")
        if ho is None:
            continue
        hc = human.get("causal_reach") or {}
        stored = hc.get("reach_id")
        cid = stored
        review_reach = None
        # The FRAMES are the durable fact; the reach_id is not. Reach ids are
        # renumbered on every run, so a review written against an earlier
        # reach list names an id this run does not have -- the id-only match
        # then found nothing and the human's answer silently vanished from the
        # reach rows (e.g. a
        # pick stored as id 42 at frames 3989-4004 while this run's reach at
        # those frames is id 36). Older reviews store frames only. So: resolve
        # by frames whenever they exist, and fall back to the stored id only
        # when they do not or nothing overlaps.
        if hc.get("start") is not None and current_segments:
            _hs, _he = int(hc["start"]), int(hc.get("end", hc["start"]))
            here = [r for _seg in current_segments
                    if int(_seg.get("segment_num", -1)) == int(sn)
                    for r in (_seg.get("reaches") or [])]
            match = typed_reach_match(_hs, _he, here)
            if match is not None:
                best = match.get("reach_id")
                if stored is not None and best != stored:
                    logger.info("review causal pick re-anchored by frames: segment %s id %s -> %s",
                                sn, stored, best)
                cid = best
            elif ho in _TOUCHED_OUTCOMES:
                # No detected reach is this one: the detector missed it. The
                # frames the person gave are the fact; the stored id, if any,
                # names some other reach. Added to the reach list below --
                # unless it is longer than any reach can be (a mistyped frame
                # number), which is never measured: no causal reach, and the
                # review tool refuses such an answer (answer_problem).
                cid = None
                if _he - _hs + 1 <= MAX_TYPED_REACH_FRAMES:
                    review_reach = (_hs, _he)
                else:
                    logger.warning("segment %s: typed reach %d-%d (%d frames) is longer than "
                                   "any reach; not added", sn, _hs, _he, _he - _hs + 1)
        bounds = person_reach_bounds(rec) if ho in _TOUCHED_OUTCOMES else None
        if review_reach is not None and bounds is not None:
            # A reach the detector missed: the frames the person checked are
            # its frames (they may differ from the frames first typed).
            review_reach = bounds
        out[int(sn)] = {
            "outcome": ho,
            "causal_reach_id": cid if ho in _TOUCHED_OUTCOMES else None,
            "review_causal_reach_id": stored,
            "review_reach": review_reach,
            # The frames the person checked for the causal reach they named
            # (person_reach_bounds), put on that reach by
            # _apply_reviewed_bounds; None when the record does not say so.
            "review_reach_bounds": bounds if (cid is not None and review_reach is None) else None,
            "missed_reaches": person_missed_reaches(rec),
            "abnormal_ranges": ans.get("abnormal_ranges"),
            # Who answered this segment, not who saved the file last
            # (causal_review_io.segment_reviewer).
            "reviewer": rec.get("reviewer") or (doc or {}).get("reviewer"),
            "source": source,
        }
    return out


def _seg_overrides_from_gt(gt: Optional[dict], current_segments=None,
                           human_segmentation: bool = False) -> Tuple[Dict[int, dict], bool]:
    """(overrides, exhaustive) for the per-segment OUTCOME layer from GT.

    Keys are numbers in the CURRENT segmentation: each GT outcome is matched
    to the current segment its frames cover, in the GT's own segmentation
    (causal_review_io.index_gt_outcomes_by_segment) -- never by bare number,
    which a re-cut reassigns. Without ``current_segments``, or for a GT file
    without its own segmentation, the number is used."""
    from mousereach.review.causal_review_io import index_gt_outcomes_by_segment
    oc = (gt or {}).get("outcomes") or {}
    out: Dict[int, dict] = {}
    by_seg, notes = index_gt_outcomes_by_segment(gt, current_segments,
                                                 human_segmentation=human_segmentation)
    for n in notes:
        logger.warning("ground-truth re-anchoring: %s", n)
    for sn, seg in sorted(by_seg.items()):
        if seg.get("determined") is False:
            continue
        ho = seg.get("outcome")
        if ho is None:
            continue
        # No causal_reach_id: the id a ground-truth file stores was copied
        # from the algorithm's reach list of the run it was made on, and no
        # tool lets a person set it. The causal reach is found by frames
        # (_apply_outcome_layers).
        out[int(sn)] = {
            "outcome": ho,
            "interaction_frame": seg.get("interaction_frame"),
            "reviewer": (gt or {}).get("last_modified_by") or (gt or {}).get("created_by"),
            "source": "ground_truth",
        }
    return out, bool(oc.get("exhaustive"))


def _reach_holding_frame(reaches: List[dict], frame) -> Optional[dict]:
    """The reach whose frames (inclusive) contain ``frame``; a ground-truth
    reach is preferred over a detected one when both do. None when no reach
    does or there is no frame. The rule triage/gt_resolve._find_causal_by_ifr
    uses: the reach in progress at the moment of contact is the one that
    touched the pellet."""
    if frame is None:
        return None
    try:
        f = int(frame)
    except (TypeError, ValueError):
        return None
    hits = []
    for r in reaches or []:
        s, e = r.get("start_frame"), r.get("end_frame")
        if s is None or e is None:
            continue
        if int(s) <= f <= int(e):
            hits.append(r)
    if not hits:
        return None
    hits.sort(key=lambda r: 0 if r.get("reach_source") == "ground_truth" else 1)
    return hits[0]


def _apply_outcome_layers(outcomes_data: dict, layers: List[Dict[int, dict]],
                          reaches_data: Optional[dict] = None) -> dict:
    """Resolve per-segment outcomes. ``layers`` is ordered LOW -> HIGH.

    Merged per FIELD, not per segment (a higher layer used to replace the
    lower layer's whole answer, so a ground-truth outcome threw away the
    review's causal reach, typed reach and ignore windows):

    - outcome, its source and who answered: the highest layer with a call.
    - interaction frame: the highest layer that gives one.
    - ignore windows (abnormal_ranges) and a typed reach: from the review,
      whichever layer wins the outcome -- they say which frames are not
      behaviour and which reach the detector missed, and ground truth
      records neither.
    - causal reach: the review's pick when the review wins. A review that
      marks the pellet touched and picks no reach leaves the causal reach to
      the reach assignment (it did not say which reach). When ground truth
      wins and the pellet was touched, ground truth stores no causal reach a
      person chose (the tool has no control for it; any id in the file was
      copied from the algorithm's reach list of an older run). So, in order:
      (a) the reach -- in this run's resolved reach list, ``reaches_data`` --
      whose frames contain the ground truth's interaction frame; (b) the
      review's pick, when the review gave the same outcome; (c) nothing is
      set, so the reach assignment's answer is merged by the extractor as on
      any segment no person decided. The stored id is never used. Which one
      applied is stamped as ``causal_reach_source`` ("ground_truth",
      "human_review"; the extractor stamps "reach_assignment" for (c)).
    - an untouched ground-truth outcome: no causal reach (key present, None:
      "no reach did this").

    A review and ground truth that disagree on the outcome are logged with the
    segment, both outcomes and who gave the review. Non-destructive; stamps
    provenance and preserves the algo originals as algo_outcome /
    algo_causal_reach_id."""
    out = copy.deepcopy(outcomes_data)
    reach_lists: Dict[int, List[dict]] = {}
    for rseg in (reaches_data or {}).get("segments", []):
        try:
            reach_lists[int(rseg.get("segment_num"))] = rseg.get("reaches") or []
        except (TypeError, ValueError):
            continue
    for seg in out.get("segments", []):
        if "outcome_source" not in seg:
            seg["outcome_source"] = "algo"
        sn = seg.get("segment_num")
        if sn is None:
            continue
        ovs = [lay[int(sn)] for lay in layers if lay.get(int(sn))]
        if not ovs:
            continue
        top = ovs[-1]
        review = next((o for o in reversed(ovs) if o.get("source") == "human_review"), None)
        seg["algo_outcome"] = seg.get("outcome")
        seg["algo_causal_reach_id"] = seg.get("causal_reach_id")
        seg["outcome"] = top["outcome"]
        seg["outcome_source"] = top["source"]
        if top.get("reviewer"):
            seg["reviewed_by"] = top["reviewer"]
        ifr = next((o["interaction_frame"] for o in reversed(ovs)
                    if o.get("interaction_frame") is not None), None)
        if ifr is not None:
            seg["interaction_frame"] = ifr
        ranges = next((o["abnormal_ranges"] for o in reversed(ovs) if o.get("abnormal_ranges")), None)
        if ranges:
            seg["abnormal_ranges"] = ranges
        if review is not None and review.get("review_causal_reach_id") is not None:
            seg["review_causal_reach_id"] = review["review_causal_reach_id"]  # the id as reviewed; provenance
        seg.pop("review_reach_frames", None)
        seg.pop("causal_reach_source", None)
        seg.pop("review_reach_bounds", None)
        if review is not None and review.get("missed_reaches"):
            seg["person_missed_reaches"] = [list(w) for w in review["missed_reaches"]]

        if top.get("source") != "ground_truth":
            if (top["outcome"] in _TOUCHED_OUTCOMES and top.get("causal_reach_id") is None
                    and not top.get("review_reach")):
                # A review that marks the pellet touched but names no reach
                # says nothing about WHICH reach: it is not a person's "no
                # reach". The causal reach is left to the reach assignment, as
                # on any pellet whose reach no person decided (the extractor
                # merges it; causal_reach_owned_by_person). WHY: written as
                # None it blocked the assignment's answer and the pellet
                # shipped touched with no causal reach.
                if seg.get("algo_causal_reach_id") is None:
                    seg.pop("causal_reach_id", None)
                continue
            seg["causal_reach_id"] = top.get("causal_reach_id")
            if seg["causal_reach_id"] is not None:
                seg["causal_reach_source"] = top.get("source")
                if top.get("review_reach_bounds"):
                    seg["review_reach_bounds"] = list(top["review_reach_bounds"])
            if top.get("review_reach"):
                seg["review_reach_frames"] = list(top["review_reach"])
            continue

        # Ground truth decided the outcome.
        if review is not None and review.get("outcome") != top["outcome"]:
            logger.warning("segment %s: ground truth says %s, the review by %s says %s; "
                           "ground truth is applied", sn, top["outcome"],
                           review.get("reviewer") or "unknown", review.get("outcome"))
        if top["outcome"] not in _TOUCHED_OUTCOMES:
            seg["causal_reach_id"] = None
        else:
            held = _reach_holding_frame(reach_lists.get(int(sn), []), ifr)
            if held is not None and held.get("reach_id") is not None:
                seg["causal_reach_id"] = held["reach_id"]
                seg["causal_reach_source"] = "ground_truth"
            elif review is not None and review.get("outcome") == top["outcome"] and \
                    (review.get("review_reach") or review.get("causal_reach_id") in
                     {r.get("reach_id") for r in reach_lists.get(int(sn), [])}):
                # The review's pick counts only while that reach is still in
                # this run's resolved list: a ground-truth reach may have
                # replaced it, and a dangling id would block the assignment.
                if review.get("review_reach"):
                    # A reach the detector missed: _add_reviewed_reaches adds
                    # it and makes it the causal reach.
                    seg["review_reach_frames"] = list(review["review_reach"])
                else:
                    seg["causal_reach_id"] = review["causal_reach_id"]
                    if review.get("review_reach_bounds"):
                        seg["review_reach_bounds"] = list(review["review_reach_bounds"])
                seg["causal_reach_source"] = "human_review"
            elif seg.get("algo_causal_reach_id") is None:
                # (c): the outcome file named no causal reach (the current
                # detector writes none), so the key is left out and the
                # extractor merges the reach assignment's answer.
                seg.pop("causal_reach_id", None)
        if review is not None and review.get("review_reach") and "review_reach_frames" not in seg:
            # The reviewer's typed reach is still a reach that happened: it
            # joins the reach list, not as the causal reach, unless ground
            # truth's reach list is complete (_add_reviewed_reaches).
            seg["review_reach_frames"] = list(review["review_reach"])
            seg["review_reach_not_causal"] = True
    return out


# ---------------------------------------------------------------------------
# per-reach layer: GT, and (since 2026-10-26) a reviewer-typed reach the
# detector missed; review never edits the boundaries of a detected reach
# ---------------------------------------------------------------------------
def _gt_reach_dict(r: dict, reach_num: int) -> dict:
    """Map a GT reach onto the algo reach-dict schema the extractor consumes.

    max_extent_* are left None on purpose: the algo's precomputed extent is for
    the algo's window, which is not the GT window. The extractor recomputes the
    DLC-derived kinematics fresh over the exact GT [start,end] frames -- that is
    the whole point of honoring the GT boundary."""
    s = r.get("start_frame")
    e = r.get("end_frame")
    return {
        "reach_id": r.get("reach_id"),
        "reach_num": reach_num,
        "start_frame": s,
        "end_frame": e,
        "apex_frame": r.get("apex_frame"),
        # Frames from start to end, both included -- what the detector writes
        # and the window the extractor measures (it was e - s, one frame short).
        "duration_frames": reach_duration_frames(r),
        "max_extent_pixels": None,
        "max_extent_ruler": None,
        "reach_source": "ground_truth",
    }


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _merge_reaches(algo_reaches: List[dict], gt_reaches: List[dict]) -> List[dict]:
    """Non-exhaustive per-element merge: each GT reach overrides the algo reach it
    overlaps (GT wins); algo reaches with no GT overlap persist; GT reaches with
    no algo match are added. Sorted by start_frame; reach_num reassigned 1..n."""
    used = set()
    tagged: List[Tuple[str, dict]] = []
    for g in gt_reaches:
        gs, ge = g.get("start_frame"), g.get("end_frame")
        for idx, a in enumerate(algo_reaches):
            if idx in used:
                continue
            asf, aef = a.get("start_frame"), a.get("end_frame")
            if None in (asf, aef, gs, ge):
                continue
            if _overlap((gs, ge), (asf, aef)):
                used.add(idx)
                break
        tagged.append(("gt", g))
    for idx, a in enumerate(algo_reaches):
        if idx not in used:
            tagged.append(("algo", a))
    tagged.sort(key=lambda t: (t[1].get("start_frame") or 0))
    final: List[dict] = []
    for i, (src, r) in enumerate(tagged):
        if src == "gt":
            final.append(_gt_reach_dict(r, i + 1))
        else:
            d = dict(r)
            d["reach_num"] = i + 1
            d.setdefault("reach_source", "algo")
            final.append(d)
    return final


def _add_reviewed_reaches(reaches_data: dict, outcomes_data: dict,
                          gt_reaches_complete: bool = False) -> dict:
    """Add each reviewer-typed reach the detector missed (the module docstring
    says why) to its segment's reaches, and make it that segment's causal reach.
    It is merged exactly as a supplementary GT reach is (_merge_reaches: it
    replaces the detected reach it overlaps), stamped reach_source
    "human_review", and given a reach id no other reach in the video has.
    ``outcomes_data`` is updated in place. Non-destructive for reaches_data."""
    wanted = {}
    for seg in outcomes_data.get("segments", []):
        frames = seg.get("review_reach_frames")
        if not frames or seg.get("segment_num") is None:
            continue
        if seg.get("outcome_source") == "ground_truth" and gt_reaches_complete:
            # Ground truth marks its reach list complete: a reach it does not
            # list is not added. Said so, never silently.
            logger.warning("segment %s: the review's typed reach %s-%s is not added -- "
                           "ground truth marks its reach list complete",
                           seg.get("segment_num"), frames[0], frames[1])
            seg.pop("review_reach_frames", None)
            seg.pop("review_reach_not_causal", None)
            if seg.get("causal_reach_source") == "human_review" and seg.get("causal_reach_id") is None:
                seg.pop("causal_reach_source", None)
                if seg.get("algo_causal_reach_id") is None:
                    seg.pop("causal_reach_id", None)
            continue
        wanted[int(seg["segment_num"])] = (seg, frames)
    if not wanted:
        return reaches_data
    out = copy.deepcopy(reaches_data)
    next_id = 1 + max([int(r.get("reach_id") or 0) for sg in out.get("segments", [])
                       for r in sg.get("reaches", [])] + [0])
    for seg in out.get("segments", []):
        try:
            sn = int(seg.get("segment_num"))
        except (TypeError, ValueError):
            continue
        if sn not in wanted:
            continue
        oseg, (s, e) = wanted[sn]
        for a in seg.get("reaches", []):
            a.setdefault("reach_source", "algo")
        seg["reaches"] = _merge_reaches(seg.get("reaches", []), [
            {"reach_id": next_id, "start_frame": int(s), "end_frame": int(e)}])
        for r in seg["reaches"]:
            if r.get("reach_id") == next_id:
                r["reach_source"] = "human_review"
        seg["n_reaches"] = len(seg["reaches"])
        if not oseg.pop("review_reach_not_causal", False):
            oseg["causal_reach_id"] = next_id
            oseg["causal_reach_source"] = "human_review"
        next_id += 1
    return out


def _apply_reviewed_bounds(reaches_data: dict, outcomes_data: dict) -> dict:
    """Put the frames a person checked for a causal reach on that reach
    (person_reach_bounds says where they come from). Non-destructive.

    The reach keeps its id (the causal pick names it); its start and end
    become the person's, its detector frames are kept as algo_start_frame /
    algo_end_frame, it is stamped reach_source "human_review", its duration
    is counted from the new frames, an apex outside them is dropped and the
    detector's extent (measured over the detector's window) is cleared -- the
    extractor measures the reach over the person's frames, as it does for a
    ground-truth reach. A reach whose frames ground truth determined keeps
    them: ground truth outranks a review (module docstring)."""
    wanted = {}
    for seg in outcomes_data.get("segments", []):
        b = seg.get("review_reach_bounds")
        if b and seg.get("causal_reach_id") is not None and seg.get("segment_num") is not None:
            wanted[int(seg["segment_num"])] = (seg["causal_reach_id"], int(b[0]), int(b[1]))
    if not wanted:
        return reaches_data
    out = copy.deepcopy(reaches_data)
    for seg in out.get("segments", []):
        try:
            sn = int(seg.get("segment_num"))
        except (TypeError, ValueError):
            continue
        if sn not in wanted:
            continue
        cid, s, e = wanted[sn]
        for r in seg.get("reaches", []):
            if r.get("reach_id") != cid or r.get("reach_source") == "ground_truth":
                continue
            if (r.get("start_frame"), r.get("end_frame")) != (s, e):
                r["algo_start_frame"] = r.get("start_frame")
                r["algo_end_frame"] = r.get("end_frame")
                r["start_frame"], r["end_frame"] = s, e
                r["duration_frames"] = reach_duration_frames(r)
                apex = r.get("apex_frame")
                if apex is None or not (s <= int(apex) <= e):
                    r["apex_frame"] = None
                r["max_extent_pixels"] = None
                r["max_extent_ruler"] = None
            r["reach_source"] = "human_review"
            r["bounds_checked_by_person"] = True
    return out


def _segment_of_reach(r: dict, spans: Dict[int, Tuple[int, int]]) -> Optional[int]:
    """The CURRENT segment a ground-truth reach lies in, by its frames: the
    segment holding most of the reach's frames (on a tie, the one holding its
    start). None when the reach has no frames or lies in no current segment;
    the caller then falls back to the number the reach was labelled with."""
    s, e = r.get("start_frame"), r.get("end_frame")
    if s is None or not spans:
        return None
    s = int(s)
    e = int(e) if e is not None else s
    best, best_ov = None, 0
    for sn, (a, b) in sorted(spans.items()):
        ov = max(0, min(e, b) - max(s, a) + 1)
        if ov > best_ov or (ov == best_ov and ov and a <= s <= b):
            best, best_ov = sn, ov
    return best


def gt_reach_is_determined(r: dict) -> bool:
    """True when a person determined BOTH frames of a ground-truth reach.

    The ground-truth tool seeds its reach list from the algorithm's reaches
    and marks a start or end "determined" only when a person sets or confirms
    it; a saved file can therefore hold algorithm reaches nobody looked at
    (the progress save writes every reach). Such a reach is the algorithm's
    guess carried in a ground-truth file, not ground truth, so it is never
    applied as ground truth. Older files (schema 1) say ``start_verified`` /
    ``end_verified`` for the same thing; both spellings are read, because
    reading only the new one would throw away every older file's reaches."""
    def _flag(new, old):
        v = r.get(new)
        return bool(r.get(old)) if v is None else bool(v)
    return _flag("start_determined", "start_verified") and         _flag("end_determined", "end_verified")


def _apply_gt_reaches(reaches_data: dict, gt: Optional[dict]) -> dict:
    """Resolve per-segment reaches against GT. Exhaustive -> GT reach set is the
    complete truth for every segment; non-exhaustive -> overlap-merge only where
    GT labeled reaches. Non-destructive.

    Only reaches a person determined (gt_reach_is_determined: start AND end)
    are applied. A segment is replaced by the ground truth's reach set only
    when the set is exhaustive AND every ground-truth reach in that segment is
    determined; when the set is marked exhaustive but a segment still holds an
    undetermined reach, the person had not finished that segment, so only its
    determined reaches are overlap-merged and the algorithm's other reaches
    stay (logged with the segment and count). WHY: undetermined reaches are
    the algorithm's seeds from the run the file was made on; applying them as
    "ground truth" stamped an old algorithm guess reach_source ground_truth,
    and an exhaustive set built from them dropped real detected reaches.
    A reach the person flagged exclude_from_analysis is a person's statement
    whatever its frame flags: every detected reach it overlaps is taken out
    of the segment, exhaustive set or not, and recorded under
    ``excluded_reaches`` with the person's reason (_drop_excluded_reaches).

    Each GT reach goes to the CURRENT segment its frames lie in, not to the
    segment NUMBER it was labelled with: a re-cut renumbers segments, and a
    reach placed by number would join (or, exhaustive, replace) the reaches of
    other footage. The number is used only when the current segments carry no
    frames, or no current segment holds the reach."""
    rr = (gt or {}).get("reaches") or {}
    exhaustive = bool(rr.get("exhaustive"))
    listed = [r for r in (rr.get("reaches") or []) if isinstance(r, dict)]
    gt_reaches = [r for r in listed
                  if not r.get("exclude_from_analysis") and gt_reach_is_determined(r)]
    # Undetermined, not excluded: they keep a segment from being replaced.
    open_reaches = [r for r in listed
                    if not r.get("exclude_from_analysis") and not gt_reach_is_determined(r)]
    excluded = [r for r in listed if r.get("exclude_from_analysis")]
    if not exhaustive and not gt_reaches and not excluded:
        return reaches_data  # nothing GT can say about reaches

    out = copy.deepcopy(reaches_data)
    spans: Dict[int, Tuple[int, int]] = {}
    for seg in out.get("segments", []):
        try:
            sn = int(seg.get("segment_num"))
            # First and last frame, both included: _segment_of_reach counts
            # overlap that way, and the reaches file's stored end is the next
            # segment's first frame (reach_file_segment_span).
            spans[sn] = reach_file_segment_span(seg)
        except (TypeError, ValueError, KeyError):
            continue
    # The first cut: a ground-truth reach that starts before it belongs to no
    # pellet (owner decision D2-3, reach/core/before_first_cut.py). It goes to
    # the file's reaches_before_first_cut list, never into segment 1 by the
    # segment number the tool seeded it with.
    _first_cut = min((a for a, _b in spans.values()), default=None)
    _BEFORE = -2

    def _place(r, log=True):
        try:
            if _first_cut is not None and int(r.get("start_frame")) < _first_cut:
                return _BEFORE
        except (TypeError, ValueError):
            pass
        sn = _segment_of_reach(r, spans)
        if sn is None:
            sn = r.get("segment_num")
        elif log and r.get("segment_num") is not None and int(r["segment_num"]) != sn:
            logger.info("ground-truth reach %s-%s labelled segment %s lies in current "
                        "segment %s; placed there", r.get("start_frame"),
                        r.get("end_frame"), r.get("segment_num"), sn)
        return int(sn) if sn is not None else -1

    by_seg: Dict[int, List[dict]] = {}
    for r in gt_reaches:
        by_seg.setdefault(_place(r), []).append(r)
    for lst in by_seg.values():
        lst.sort(key=lambda r: (r.get("start_frame") or 0))
    if by_seg.get(_BEFORE):
        from mousereach.reach.core.before_first_cut import BEFORE_FIRST_CUT_KEY
        gt_before = by_seg.pop(_BEFORE)
        kept = [a for a in (out.get(BEFORE_FIRST_CUT_KEY) or [])
                if not any(_overlap((a.get("start_frame"), a.get("end_frame")),
                                    (g.get("start_frame"), g.get("end_frame")))
                           for g in gt_before if None not in (a.get("start_frame"),
                                                              a.get("end_frame")))]
        out[BEFORE_FIRST_CUT_KEY] = sorted(
            kept + [_gt_reach_dict(g, 0) for g in gt_before],
            key=lambda r: (r.get("start_frame") or 0))
    open_by_seg: Dict[int, int] = {}
    for r in open_reaches:
        sn = _place(r, log=False)
        open_by_seg[sn] = open_by_seg.get(sn, 0) + 1

    excl_windows = []
    for r in excluded:
        try:
            a, b = int(r["start_frame"]), int(r["end_frame"])
        except (TypeError, ValueError, KeyError):
            continue
        excl_windows.append((min(a, b), max(a, b), r.get("exclude_reason")))

    for seg in out.get("segments", []):
        for a in seg.get("reaches", []):
            a.setdefault("reach_source", "algo")
        if excl_windows:
            _drop_excluded_reaches(seg, excl_windows)
        try:
            sn = int(seg.get("segment_num"))
        except (TypeError, ValueError):
            continue
        gtr = by_seg.get(sn, [])
        if exhaustive and open_by_seg.get(sn):
            logger.warning("ground truth marks its reaches complete, but segment %s still "
                           "holds %d reach(es) nobody determined: only its %d determined "
                           "reach(es) are applied, the detected reaches stay",
                           sn, open_by_seg[sn], len(gtr))
            if gtr:
                seg["reaches"] = _merge_reaches(seg.get("reaches", []), gtr)
                seg["n_reaches"] = len(seg["reaches"])
        elif exhaustive:
            # GT is the complete reach set for this segment (possibly empty ->
            # every algo reach here was a false positive and is dropped).
            seg["reaches"] = [_gt_reach_dict(r, i + 1) for i, r in enumerate(gtr)]
            seg["n_reaches"] = len(seg["reaches"])
            # Said on the segment because this is the ONE place ground truth
            # removes reaches outright (elsewhere a ground-truth reach only
            # replaces the reach it overlaps, or an exclusion lists what it
            # took out). A kinematics-only re-run of a filed video accepts a
            # filed reach going only when something decided it
            # (kinematics/core/filed_decisions.changes_not_owed); this is how
            # it knows that ground truth, complete for this pellet, did.
            # Never set on a segment ground truth left open (branch above).
            seg["reach_list_complete_by_ground_truth"] = True
        elif gtr:
            seg["reaches"] = _merge_reaches(seg.get("reaches", []), gtr)
            seg["n_reaches"] = len(seg["reaches"])
    _unique_reach_ids(out)
    return out


def _drop_excluded_reaches(seg: dict, windows) -> None:
    """Take every detected reach of ``seg`` whose frames overlap a reach a
    person marked "exclude from analysis" in the ground-truth tool out of the
    segment's reaches, and record it under ``excluded_reaches`` (in place).

    WHY: the exclusion is a person's statement that those frames are not a
    valid reach. It used to remove only the ground-truth copy of the reach;
    the detected reach over the same frames stayed and was measured like any
    other, unless the reach set was marked complete. It now works like a
    reviewer's ignore window (feature_extractor.split_reaches_by_ignore_windows):
    every detected reach that overlaps it (inclusive, by one frame is enough)
    is left out, and the features file lists it in the segment's
    ``reaches_left_out`` with the person's reason, so the file still says the
    reach happened. A reach a person determined in the ground truth is never
    removed by this -- it is applied after, by the merge below."""
    kept, gone = [], []
    for r in seg.get("reaches", []) or []:
        try:
            rs, re_ = int(r.get("start_frame")), int(r.get("end_frame"))
        except (TypeError, ValueError):
            kept.append(r)
            continue
        hit = next((w for w in windows if rs <= w[1] and re_ >= w[0]), None)
        if hit is None:
            kept.append(r)
            continue
        gone.append({"reach_id": r.get("reach_id"), "start_frame": rs, "end_frame": re_,
                     "excluded_by_person": {"start_frame": hit[0], "end_frame": hit[1],
                                            "reason": hit[2]}})
    if gone:
        seg["reaches"] = kept
        seg["n_reaches"] = len(kept)
        seg["excluded_reaches"] = (seg.get("excluded_reaches") or []) + gone


def _unique_reach_ids(reaches_data: dict) -> None:
    """Give a ground-truth reach a new id when a detected reach of this run
    already has its id (in place). WHY: a ground-truth reach keeps the id it
    had in the run the file was made on, and ids are renumbered on every run,
    so a merged list could hold two reaches with one id -- and the causal
    reach is named by id, so both would be marked causal. Detected reaches
    keep theirs (reviews and the reach assignment name them)."""
    segs = reaches_data.get("segments", [])
    taken = {r.get("reach_id") for sg in segs for r in sg.get("reaches", [])
             if r.get("reach_source", "algo") != "ground_truth" and r.get("reach_id") is not None}
    next_id = 1 + max([int(i) for i in taken if isinstance(i, int)] +
                      [int(r.get("reach_id") or 0) for sg in segs for r in sg.get("reaches", [])
                       if isinstance(r.get("reach_id"), int)] + [0])
    for sg in segs:
        for r in sg.get("reaches", []):
            if r.get("reach_source") != "ground_truth":
                continue
            if r.get("reach_id") is None or r.get("reach_id") in taken:
                r["reach_id"] = next_id
                next_id += 1
            taken.add(r["reach_id"])


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def resolve_truth_layers(
    reaches_data: dict,
    outcomes_data: dict,
    video_stem: Optional[str] = None,
    primary_dir=None,
    extra_gt_dirs=(),
    human_segmentation: bool = False,
    applied: Optional[dict] = None,
    extra_review_dirs=(),
    segments_doc: Optional[dict] = None,
) -> Tuple[dict, dict]:
    """Return (reaches_data, outcomes_data) with every element resolved by
    GT > causal review > triage review > algo (see module docstring).

    ``primary_dir`` is an extra location to look for a causal-review file (e.g.
    the processing dir the extractor is running in). ``extra_gt_dirs`` are extra
    dirs to check for the video's unified GT before the corpus index. Never
    raises: any missing layer is simply absent.

    ``applied``, when a dict is given, receives ``'ground_truth'``: the path of
    the ground-truth file this call found and layered (None when it found
    none). WHY: the kinematics step records in the processing manifest which
    ground truth its numbers were made with, and the version scan re-runs a
    filed video whose filed ground truth differs from it. Recording the file
    that happened to sit beside the run instead could name a file this call
    never read (the corpus index is built once per process), and the video
    would then count as brought up to date when it was not.

    ``applied`` also receives ``'problems'``: a list (empty when none) of the
    copies of a person's answers this call found but could not read -- a
    review copy or the ground-truth file. Such a file is left out of the
    layering, so the result is NOT the person's result; the kinematics
    extractor refuses to write features while the list is not empty."""
    from mousereach.config import Paths
    from mousereach.review.causal_review_io import find_gt

    stem = (video_stem or reaches_data.get("video_name")
            or outcomes_data.get("video_name") or "")

    # The pellets these results were produced from. Reviews are matched
    # against their frames, so a human judgement follows the footage it was
    # made about even when a re-cut renumbers the segments around it.
    # Their frames come from the segments file through the one definition
    # (causal_review_io.pellet_span, via current_pellets) -- the frames the
    # review window saved on each answer and the review gate matches against.
    # NOT from the reaches file's own segment list, which ends each segment
    # one frame later (at the next cut) and adds a segment after the last cut:
    # matched against that, the results step and the gate disagreed about
    # which answers count (pellet_span says what that cost). The reaches file
    # supplies each pellet's detected reaches, by number. Only a run with no
    # segments file falls back to the reaches file's list; it has no person's
    # cuts, so no re-cut to honour.
    from mousereach.review.causal_review_io import (current_pellets,
                                                    recut_voided_segments,
                                                    segmentation_is_human)
    _cur_segs = current_pellets(segments_doc, reaches_data) if segments_doc else []
    _recut_cuts_unreadable = False
    if _cur_segs:
        human_segmentation = human_segmentation or segmentation_is_human(segments_doc)
    else:
        _recut_cuts_unreadable = bool(segments_doc) and segmentation_is_human(segments_doc)
        _cur_segs = [_reach_file_pellet(s)
                     for s in (reaches_data.get("segments") or []) if isinstance(s, dict)]

    # ---- per-segment OUTCOME layers, low -> high --------------------------
    # The human layer: every copy of the review, newest answer per segment
    # (module docstring, "Where the human answers come from").
    from mousereach.review.causal_review_io import effective_review, review_dirs
    primary = primary_dir
    if primary is not None and Path(primary).is_file() and \
            Path(primary).name != f"{stem}_causal_review.json":
        primary = None   # some other video's file is never read for this one
    problems: List[str] = []
    if applied is not None:
        applied['problems'] = problems
        applied['recut_voided'] = []
    # The reaches file must have been made on these cuts: each pellet's
    # segment there starts at the same cut. WHY: answers are matched to the
    # segments file's pellets and the reaches/outcomes are keyed by number; a
    # reaches file made on other cuts (a step command or backfill run beside
    # newer cuts, where no review gate checks for stale outputs) would put a
    # person's answer on another pellet's reaches. Refused, never guessed.
    _cut_mismatch = []
    if _cur_segs and segments_doc:
        _starts = {}
        for _s in reaches_data.get("segments") or []:
            try:
                _starts[int(_s["segment_num"])] = int(_s["start_frame"])
            except (KeyError, TypeError, ValueError):
                continue
        _cut_mismatch = [p["segment_num"] for p in _cur_segs
                         if _starts.get(p["segment_num"]) != p["start_frame"]]
    if _recut_cuts_unreadable:
        # A person cut this video, but their cuts cannot be read: which of
        # their answers a re-cut voided cannot be told, so nothing is applied
        # as if it could (the extractor refuses while problems is not empty).
        problems.append("the segments file a person saved holds no usable cuts, so "
                        "which of their answers still stand cannot be told")
    if _cut_mismatch:
        problems.append("the reaches file was not made on the cuts in the segments file "
                        "(segment(s) %s start elsewhere), so a person's answers cannot be "
                        "matched to its reaches; re-run reach detection on the current cuts"
                        % ", ".join(str(n) for n in _cut_mismatch[:10]))
    # Plus the folders the caller names (the run's own folder): the copy the
    # extractor was handed may be the durable one, and the copy beside the run
    # must still be merged in -- every copy counts, never one picked by time.
    _dirs = review_dirs(stem, primary) + [Path(d) for d in (extra_review_dirs or ()) if d]
    # When a re-cut (a person's or the segmenter's) set the current cuts and
    # which cuts it replaced
    # (carried on each pellet, causal_review_io.recut_keys): an answer given
    # before then counts only on a pellet whose frames did not move
    # (causal_review_io.answer_survives_recut, the one rule).
    _plain = [{k: s.get(k) for k in ("segment_num", "start_frame", "end_frame", "cuts_set_at",
                                     "cuts_replaced")
               if s.get(k) is not None} for s in _cur_segs]
    review_doc = effective_review(
        stem, _dirs, current_segments=_plain or None,
        human_segmentation=human_segmentation, unreadable=problems) if stem else None

    layers: List[Dict[int, dict]] = [
        _seg_overrides_from_review(review_doc, "human_review", _cur_segs,
                                   human_segmentation),
    ]

    gt_path = find_gt(stem, extra_dirs=extra_gt_dirs) if stem else None
    if applied is not None:
        applied['ground_truth'] = gt_path
    gt = None
    if gt_path:
        try:
            # utf-8-sig: a read error is now a permanent hold, so a byte-order mark
            # written by another tool must not turn a good file into one.
            with open(gt_path, encoding='utf-8-sig') as f:
                gt = json.load(f)
        except FileNotFoundError:
            gt = None
        except Exception as err:
            logger.error('Could not read %s, so the ground truth in it is not applied: %s',
                         gt_path, err)
            problems.append(f"the ground-truth file {gt_path} could not be read ({err})")
    if gt:
        # Ground truth is matched by frames on the same pellets, but the re-cut
        # rule for review answers is not applied to it (unchanged here): a
        # ground-truth file carries its own cuts and is put back on a pellet
        # only where they match (index_gt_outcomes_by_segment, unified_gt).
        gt_out, _exhaustive_out = _seg_overrides_from_gt(
            gt, [{k: v for k, v in s.items() if k != "cuts_set_at"} for s in _cur_segs],
            human_segmentation)
        layers.append(gt_out)                                     # GT tier (top)
    # ---- answers a re-cut voided, still unasked ---------------------------
    # The pellets whose person's answer a re-cut voided and that nobody has
    # answered on the new cuts (recut_voided_segments -- the review gate's own
    # check), less those ground truth decides. Not applied, and NOT left to
    # the algorithm's call either: the caller is told (applied['recut_voided'])
    # and the extractor refuses to write features until they are asked again
    # in triage. WHY: voiding used to be a log line here, and a pellet the
    # gate had not asked about took the algorithm's call over the person's.
    if _cur_segs and review_doc and applied is not None:
        open_ = recut_voided_segments(review_doc, _plain, human_segmentation=human_segmentation)
        open_ -= set((gt_out if gt else {}).keys())
        applied['recut_voided'] = sorted(open_)

    # ---- per-reach GT layer FIRST: a ground-truth pellet's causal reach is
    # the reach of THIS run's resolved list holding its interaction frame, so
    # the reach list is resolved before the outcomes.
    reaches_out = _apply_gt_reaches(reaches_data, gt) if gt else reaches_data

    outcomes_out = _apply_outcome_layers(outcomes_data, layers, reaches_out)

    # ---- then reviewer-typed reaches the detector missed
    reaches_out = _add_reviewed_reaches(
        reaches_out, outcomes_out,
        gt_reaches_complete=bool(((gt or {}).get("reaches") or {}).get("exhaustive")))
    # ---- and the frames a person checked for the causal reach they named
    reaches_out = _apply_reviewed_bounds(reaches_out, outcomes_out)

    if gt:
        logger.info(
            "truth layering %s: GT applied (reaches.exhaustive=%s, "
            "outcomes.exhaustive=%s); human review read=%s",
            stem,
            bool(((gt.get("reaches") or {}).get("exhaustive"))),
            bool(((gt.get("outcomes") or {}).get("exhaustive"))),
            bool(review_doc),
        )
    return reaches_out, outcomes_out
