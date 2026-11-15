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

GT is whole-COMPONENT authoritative only where that component's `exhaustive`
flag is set (GT stores `exhaustive` separately for segmentation / reaches /
outcomes):
  - reaches.exhaustive == True  -> the GT reach set is COMPLETE. Each segment's
       reaches ARE the GT reaches; algo reaches GT never labeled are false
       positives and are dropped. GT reaches flagged exclude_from_analysis drop.
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
        out[int(sn)] = {
            "outcome": ho,
            "causal_reach_id": cid if ho in _TOUCHED_OUTCOMES else None,
            "review_causal_reach_id": stored,
            "review_reach": review_reach,
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
        out[int(sn)] = {
            "outcome": ho,
            "causal_reach_id": seg.get("causal_reach_id") if ho in _TOUCHED_OUTCOMES else None,
            "interaction_frame": seg.get("interaction_frame"),
            "reviewer": (gt or {}).get("last_modified_by") or (gt or {}).get("created_by"),
            "source": "ground_truth",
        }
    return out, bool(oc.get("exhaustive"))


def _apply_outcome_layers(outcomes_data: dict, layers: List[Dict[int, dict]]) -> dict:
    """Resolve per-segment outcomes. ``layers`` is ordered LOW -> HIGH; the
    highest layer with a call for a segment wins. Non-destructive; stamps
    provenance and preserves the algo originals as algo_outcome /
    algo_causal_reach_id."""
    out = copy.deepcopy(outcomes_data)
    merged: Dict[int, dict] = {}
    for lay in layers:
        for sn, ov in lay.items():
            merged[sn] = ov  # later (higher-priority) layer wins
    for seg in out.get("segments", []):
        if "outcome_source" not in seg:
            seg["outcome_source"] = "algo"
        sn = seg.get("segment_num")
        if sn is None:
            continue
        ov = merged.get(int(sn))
        if not ov:
            continue
        seg["algo_outcome"] = seg.get("outcome")
        seg["algo_causal_reach_id"] = seg.get("causal_reach_id")
        seg["outcome"] = ov["outcome"]
        seg["causal_reach_id"] = ov.get("causal_reach_id")
        if ov.get("review_causal_reach_id") is not None:
            seg["review_causal_reach_id"] = ov["review_causal_reach_id"]  # the id as reviewed; provenance
        seg["outcome_source"] = ov["source"]
        if ov.get("reviewer"):
            seg["reviewed_by"] = ov["reviewer"]
        if ov.get("interaction_frame") is not None:
            seg["interaction_frame"] = ov["interaction_frame"]
        if ov.get("abnormal_ranges"):
            seg["abnormal_ranges"] = ov["abnormal_ranges"]
        seg.pop("review_reach_frames", None)
        if ov.get("review_reach"):
            seg["review_reach_frames"] = list(ov["review_reach"])
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
        "duration_frames": (e - s) if (s is not None and e is not None) else None,
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


def _add_reviewed_reaches(reaches_data: dict, outcomes_data: dict) -> dict:
    """Add each reviewer-typed reach the detector missed (the module docstring
    says why) to its segment's reaches, and make it that segment's causal reach.
    It is merged exactly as a supplementary GT reach is (_merge_reaches: it
    replaces the detected reach it overlaps), stamped reach_source
    "human_review", and given a reach id no other reach in the video has.
    ``outcomes_data`` is updated in place. Non-destructive for reaches_data."""
    wanted = {}
    for seg in outcomes_data.get("segments", []):
        frames = seg.get("review_reach_frames")
        if frames and seg.get("outcome_source") == "human_review" and seg.get("segment_num") is not None:
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
        oseg["causal_reach_id"] = next_id
        next_id += 1
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


def _apply_gt_reaches(reaches_data: dict, gt: Optional[dict]) -> dict:
    """Resolve per-segment reaches against GT. Exhaustive -> GT reach set is the
    complete truth for every segment; non-exhaustive -> overlap-merge only where
    GT labeled reaches. Non-destructive.

    Each GT reach goes to the CURRENT segment its frames lie in, not to the
    segment NUMBER it was labelled with: a re-cut renumbers segments, and a
    reach placed by number would join (or, exhaustive, replace) the reaches of
    other footage. The number is used only when the current segments carry no
    frames, or no current segment holds the reach."""
    rr = (gt or {}).get("reaches") or {}
    exhaustive = bool(rr.get("exhaustive"))
    gt_reaches = [r for r in (rr.get("reaches") or []) if not r.get("exclude_from_analysis")]
    if not exhaustive and not gt_reaches:
        return reaches_data  # nothing GT can say about reaches

    out = copy.deepcopy(reaches_data)
    spans: Dict[int, Tuple[int, int]] = {}
    for seg in out.get("segments", []):
        try:
            sn = int(seg.get("segment_num"))
            spans[sn] = (int(seg["start_frame"]), int(seg["end_frame"]))
        except (TypeError, ValueError, KeyError):
            continue
    by_seg: Dict[int, List[dict]] = {}
    for r in gt_reaches:
        sn = _segment_of_reach(r, spans)
        if sn is None:
            sn = r.get("segment_num")
        elif r.get("segment_num") is not None and int(r["segment_num"]) != sn:
            logger.info("ground-truth reach %s-%s labelled segment %s lies in current "
                        "segment %s; placed there", r.get("start_frame"),
                        r.get("end_frame"), r.get("segment_num"), sn)
        by_seg.setdefault(int(sn) if sn is not None else -1, []).append(r)
    for lst in by_seg.values():
        lst.sort(key=lambda r: (r.get("start_frame") or 0))

    for seg in out.get("segments", []):
        for a in seg.get("reaches", []):
            a.setdefault("reach_source", "algo")
        try:
            sn = int(seg.get("segment_num"))
        except (TypeError, ValueError):
            continue
        gtr = by_seg.get(sn, [])
        if exhaustive:
            # GT is the complete reach set for this segment (possibly empty ->
            # every algo reach here was a false positive and is dropped).
            seg["reaches"] = [_gt_reach_dict(r, i + 1) for i, r in enumerate(gtr)]
            seg["n_reaches"] = len(seg["reaches"])
        elif gtr:
            seg["reaches"] = _merge_reaches(seg.get("reaches", []), gtr)
            seg["n_reaches"] = len(seg["reaches"])
    return out


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
) -> Tuple[dict, dict]:
    """Return (reaches_data, outcomes_data) with every element resolved by
    GT > causal review > triage review > algo (see module docstring).

    ``primary_dir`` is an extra location to look for a causal-review file (e.g.
    the processing dir the extractor is running in). ``extra_gt_dirs`` are extra
    dirs to check for the video's unified GT before the corpus index. Never
    raises: any missing layer is simply absent."""
    from mousereach.config import Paths
    from mousereach.review.causal_review_io import find_gt

    stem = (video_stem or reaches_data.get("video_name")
            or outcomes_data.get("video_name") or "")

    # The segmentation these results were produced from. Reviews are matched
    # against these frame ranges, so a human judgement follows the footage it
    # was made about even when a re-cut renumbers the segments around it.
    _cur_segs = reaches_data.get("segments") or []

    # ---- per-segment OUTCOME layers, low -> high --------------------------
    # The human layer: every copy of the review, newest answer per segment
    # (module docstring, "Where the human answers come from").
    from mousereach.review.causal_review_io import effective_review, review_dirs
    primary = primary_dir
    if primary is not None and Path(primary).is_file() and \
            Path(primary).name != f"{stem}_causal_review.json":
        primary = None   # some other video's file is never read for this one
    review_doc = effective_review(
        stem, review_dirs(stem, primary),
        current_segments=[{"segment_num": s.get("segment_num"),
                           "start_frame": s.get("start_frame"),
                           "end_frame": s.get("end_frame")} for s in _cur_segs] or None,
        human_segmentation=human_segmentation) if stem else None

    layers: List[Dict[int, dict]] = [
        _seg_overrides_from_review(review_doc, "human_review", _cur_segs, human_segmentation),
    ]

    gt_path = find_gt(stem, extra_dirs=extra_gt_dirs) if stem else None
    gt = _read_json(gt_path) if gt_path else None
    if gt:
        gt_out, _exhaustive_out = _seg_overrides_from_gt(gt, _cur_segs, human_segmentation)
        layers.append(gt_out)                                     # GT tier (top)

    outcomes_out = _apply_outcome_layers(outcomes_data, layers)

    # ---- per-reach layers: GT, then reviewer-typed reaches the detector missed
    reaches_out = _apply_gt_reaches(reaches_data, gt) if gt else reaches_data
    reaches_out = _add_reviewed_reaches(reaches_out, outcomes_out)

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
