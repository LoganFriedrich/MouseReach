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
    outcome_source in {"algo","human_review","ground_truth"}   (per segment)
    reach_source   in {"algo","ground_truth"}                  (per reach)

Non-destructive (deep copies) and never raises on missing layers: a video with
no human input resolves to pure algo. ASCII-only logging (Windows cp1252).

Layer separation, in the data as it exists today:
  - triage review  = the quick, triaged-elements-only pass  -> Pending bundle
                     (Paths.TRIAGE_REVIEW / <stem> / <stem>_causal_review.json)
  - causal review  = the thorough full-video pass           -> Deep_Review bundle
                     (Paths.DEEP_REVIEW / <stem> / <stem>_causal_review.json),
                     plus any review saved in the processing dir (primary_dir).
Both write the same _causal_review.json schema; they are separated here by which
bundle they came from so causal outranks triage per the precedence above.
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
    except Exception:
        return None


def _review_doc(stem: str, base) -> Optional[dict]:
    """Read ``<stem>_causal_review.json`` from a review bundle base, accepting
    either ``base/<stem>/<file>`` (bundle layout) or ``base/<file>`` (a bare
    processing dir). Returns the parsed doc or None."""
    if not base or not stem:
        return None
    base = Path(base)
    for cand in (base / stem / f"{stem}_causal_review.json",
                 base / f"{stem}_causal_review.json"):
        try:
            if cand.is_file():
                d = _read_json(cand)
                if d:
                    return d
        except OSError:
            continue
    return None


def _durable_review_doc(stem: str) -> Optional[dict]:
    """Read this video's review from the durable store, or None.

    The three lookups above search places a review can DISAPPEAR from: the two
    NAS review queues, whose bundles are regenerated and torn down, and the
    caller's processing dir, which is one node's local disk. A review that
    outlived its bundle but whose video has not been archived yet exists in
    none of them -- it is exactly the case the durable store was added for, and
    without this the extractor would quietly fall back to the algorithm's answer
    while a human answer sat on disk unread.

    Lowest priority of the review layers: any live copy wins, because a reviewer
    may have edited it since.
    """
    if not stem:
        return None
    try:
        from mousereach.review.causal_review_io import durable_review_path
        p = durable_review_path(stem)
        if p is not None and p.is_file():
            return _read_json(p) or None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# per-segment OUTCOME layers
# ---------------------------------------------------------------------------
def _seg_overrides_from_review(doc: Optional[dict], source: str,
                               current_segments=None) -> Dict[int, dict]:
    """{segment_num: override} from a causal-review doc. An override carries the
    human outcome, the causal reach id (only for touched outcomes), any abnormal
    ignore-ranges, and the provenance source label.

    Keys are numbers in the CURRENT segmentation. Pass ``current_segments`` and
    each review is matched to the frames the reviewer actually saw; without it,
    matching falls back to the segment number the review was written against,
    which a re-cut can have reassigned to different footage.
    """
    from mousereach.review.causal_review_io import index_review_by_segment

    out: Dict[int, dict] = {}
    matched, notes = index_review_by_segment(doc, current_segments)
    for n in notes:
        logger.warning("review re-anchoring: %s", n)
    for sn, rec in matched.items():
        ans = rec.get("answers") or {}
        if ans.get("reviewed") is False:
            continue
        human = rec.get("human") or {}
        ho = human.get("outcome")
        if ho is None:
            continue
        hc = human.get("causal_reach") or {}
        stored = hc.get("reach_id")
        cid = stored
        # The FRAMES are the durable fact; the reach_id is not. Reach ids are
        # renumbered on every run, so a review written against an earlier
        # reach list names an id this run does not have -- the id-only match
        # then found nothing and the human's answer silently vanished from the
        # reach rows (612 reviewed segments in the 2026-08-31 snapshot; e.g. a
        # pick stored as id 42 at frames 3989-4004 while this run's reach at
        # those frames is id 36). Older reviews store frames only. So: resolve
        # by frames whenever they exist, and fall back to the stored id only
        # when they do not or nothing overlaps.
        if hc.get("start") is not None and current_segments:
            _hs, _he = int(hc["start"]), int(hc.get("end", hc["start"]))
            best, best_ov = None, 0
            for _seg in current_segments:
                if int(_seg.get("segment_num", -1)) != int(sn):
                    continue
                for _r in (_seg.get("reaches") or []):
                    _rs, _re = _r.get("start_frame"), _r.get("end_frame")
                    if _rs is None or _re is None:
                        continue
                    ov = min(_he, int(_re)) - max(_hs, int(_rs)) + 1
                    if ov > best_ov:
                        best, best_ov = _r.get("reach_id"), ov
            span = max(1, _he - _hs + 1)
            if best is not None and best_ov >= 0.5 * span:
                if stored is not None and best != stored:
                    logger.info("review causal pick re-anchored by frames: segment %s id %s -> %s",
                                sn, stored, best)
                cid = best
        out[int(sn)] = {
            "outcome": ho,
            "causal_reach_id": cid if ho in _TOUCHED_OUTCOMES else None,
            "review_causal_reach_id": stored,
            "abnormal_ranges": ans.get("abnormal_ranges"),
            "reviewer": (doc or {}).get("reviewer"),
            "source": source,
        }
    return out


def _seg_overrides_from_gt(gt: Optional[dict]) -> Tuple[Dict[int, dict], bool]:
    """(overrides, exhaustive) for the per-segment OUTCOME layer from GT."""
    oc = (gt or {}).get("outcomes") or {}
    out: Dict[int, dict] = {}
    for seg in oc.get("segments", []):
        sn = seg.get("segment_num")
        if sn is None:
            continue
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
    return out


# ---------------------------------------------------------------------------
# per-reach layer (GT only -- review never edits reach boundaries/existence)
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


def _apply_gt_reaches(reaches_data: dict, gt: Optional[dict]) -> dict:
    """Resolve per-segment reaches against GT. Exhaustive -> GT reach set is the
    complete truth for every segment; non-exhaustive -> overlap-merge only where
    GT labeled reaches. Non-destructive."""
    rr = (gt or {}).get("reaches") or {}
    exhaustive = bool(rr.get("exhaustive"))
    gt_reaches = [r for r in (rr.get("reaches") or []) if not r.get("exclude_from_analysis")]
    if not exhaustive and not gt_reaches:
        return reaches_data  # nothing GT can say about reaches

    out = copy.deepcopy(reaches_data)
    by_seg: Dict[int, List[dict]] = {}
    for r in gt_reaches:
        sn = r.get("segment_num")
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
    durable_doc = _durable_review_doc(stem)
    triage_doc = _review_doc(stem, getattr(Paths, "TRIAGE_REVIEW", None))
    causal_doc = _review_doc(stem, getattr(Paths, "DEEP_REVIEW", None))
    primary_doc = _review_doc(stem, primary_dir) if primary_dir else None

    layers: List[Dict[int, dict]] = [
        _seg_overrides_from_review(durable_doc, "human_review", _cur_segs),  # copy of record (fallback)
        _seg_overrides_from_review(triage_doc, "human_review", _cur_segs),   # triage tier
        _seg_overrides_from_review(causal_doc, "human_review", _cur_segs),   # causal tier
        _seg_overrides_from_review(primary_doc, "human_review", _cur_segs),  # causal tier (processing dir)
    ]

    gt_path = find_gt(stem, extra_dirs=extra_gt_dirs) if stem else None
    gt = _read_json(gt_path) if gt_path else None
    if gt:
        gt_out, _exhaustive_out = _seg_overrides_from_gt(gt)
        layers.append(gt_out)                                     # GT tier (top)

    outcomes_out = _apply_outcome_layers(outcomes_data, layers)

    # ---- per-reach layer (GT only) ---------------------------------------
    reaches_out = _apply_gt_reaches(reaches_data, gt) if gt else reaches_data

    if gt:
        logger.info(
            "truth layering %s: GT applied (reaches.exhaustive=%s, "
            "outcomes.exhaustive=%s); reviews triage=%s causal=%s primary=%s",
            stem,
            bool(((gt.get("reaches") or {}).get("exhaustive"))),
            bool(((gt.get("outcomes") or {}).get("exhaustive"))),
            bool(triage_doc), bool(causal_doc), bool(primary_doc),
        )
    return reaches_out, outcomes_out
