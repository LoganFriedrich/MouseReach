"""
Per-reach assignment v1: cascade-trusted JOIN of reaches to outcomes.

Takes the v8 reach detector's per-video reaches and the v6 cascade
outcome detector's per-segment outcomes and produces a permanent
per-reach output table in which every reach has a final outcome label
already stamped. Downstream kinematic analysis reads this table
directly without re-deriving outcomes.

This is the simple JOIN form: trust the cascade's `interaction_frame`
attribution to identify the causal reach in each touched segment, and
label all other reaches as `miss`. Reaches in triaged segments inherit
the `triaged` label so kinematic analysis can exclude them by default.

Per-reach label set:
  - `causal_retrieved` -- causal reach for a retrieved-class segment
  - `causal_displaced_sa` -- causal reach for a displaced_sa-class segment
  - `causal_abnormal_exception` -- causal reach for an abnormal_exception segment
  - `miss` -- non-causal reach (touched segment) OR any reach in an
    untouched segment
  - `triaged` -- any reach in a segment the cascade flagged for review
  - `unassigned` -- reach not contained by any segment (data quality issue)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


def _collapse(o: Optional[str]) -> Optional[str]:
    if o == "displaced_outside":
        return "displaced_sa"
    return o


def _label_for_reach(
    reach_start: int,
    reach_end: int,
    seg_outcome: Optional[str],
    seg_ifr: Optional[int],
    seg_flagged_for_review: bool,
) -> str:
    """Compute the per-reach label for a single reach in a single
    segment, given the segment's cascade outcome + IFR.
    """
    outcome = _collapse(seg_outcome)
    if outcome == "triaged" or seg_flagged_for_review:
        return "triaged"
    if outcome in (None, "untouched", "uncertain", "unknown"):
        return "miss"
    contains_ifr = (
        seg_ifr is not None
        and reach_start <= int(seg_ifr) <= reach_end
    )
    if outcome == "abnormal_exception":
        return "causal_abnormal_exception" if contains_ifr else "miss"
    if outcome in ("retrieved", "displaced_sa"):
        return f"causal_{outcome}" if contains_ifr else "miss"
    return "miss"


def _segment_for_frame(
    frame: int,
    segment_bounds: Sequence[Tuple[int, int]],
) -> Optional[int]:
    """Returns the index of the segment containing `frame`, or None if
    no segment contains it (between-segments transition zone)."""
    for i, (lo, hi) in enumerate(segment_bounds):
        if lo <= frame <= hi:
            return i
    return None


def assign_reaches_v1(
    *,
    reaches: List[Dict],
    segments_with_outcomes: List[Dict],
    video_id: Optional[str] = None,
) -> Dict:
    """Join per-video reaches with per-segment cascade outcomes.

    Parameters
    ----------
    reaches : list of dict
        Each reach must have at minimum ``start_frame`` and ``end_frame``.
        Optional: ``reach_id`` (auto-assigned 0..n-1 if missing).
    segments_with_outcomes : list of dict
        v6 cascade output -- each segment has ``segment_num``,
        ``outcome``, ``interaction_frame``, ``outcome_known_frame``,
        and ``flagged_for_review``. Must also have ``start_frame`` and
        ``end_frame`` so reaches can be matched to the segment that
        contains them.
    video_id : str, optional
        Stamped onto the output for traceability.

    Returns
    -------
    dict in the standard reach-assignments JSON shape:
        {
          "video_id": str,
          "detector": "assignment_v1",
          "version": "1.0.0",
          "n_reaches": int,
          "reaches": [
            {
              "reach_id": int,
              "segment_num": int | None,
              "start_frame": int,
              "end_frame": int,
              "label": str,
              "is_causal": bool,
              "segment_outcome": str | None,
              "segment_ifr": int | None,
            },
            ...
          ]
        }
    """
    seg_bounds: List[Tuple[int, int]] = []
    seg_index_by_num: Dict[int, int] = {}
    for i, s in enumerate(segments_with_outcomes):
        sf = s.get("start_frame")
        ef = s.get("end_frame")
        if sf is None or ef is None:
            seg_bounds.append((-1, -1))
            continue
        seg_bounds.append((int(sf), int(ef)))
        sn = s.get("segment_num")
        if sn is not None:
            seg_index_by_num[int(sn)] = i

    out_reaches: List[Dict] = []
    for ri, r in enumerate(reaches):
        rs = int(r.get("start_frame"))
        re = int(r.get("end_frame"))
        rid = r.get("reach_id", ri)

        # Find which segment this reach belongs to. Prefer the explicit
        # segment_num if the reach already has one; otherwise resolve
        # by frame containment.
        seg_idx: Optional[int] = None
        seg_num_in: Optional[int] = r.get("segment_num")
        if seg_num_in is not None:
            seg_idx = seg_index_by_num.get(int(seg_num_in))
        if seg_idx is None:
            mid = (rs + re) // 2
            seg_idx = _segment_for_frame(mid, seg_bounds)

        if seg_idx is None:
            # Reach falls outside any segment (transition zone or
            # data-quality artifact).
            out_reaches.append({
                "reach_id": int(rid),
                "segment_num": None,
                "start_frame": rs,
                "end_frame": re,
                "label": "unassigned",
                "is_causal": False,
                "segment_outcome": None,
                "segment_ifr": None,
            })
            continue

        seg = segments_with_outcomes[seg_idx]
        seg_num = seg.get("segment_num")
        seg_outcome = seg.get("outcome")
        seg_ifr = seg.get("interaction_frame")
        seg_flagged = bool(seg.get("flagged_for_review", False))

        label = _label_for_reach(rs, re, seg_outcome, seg_ifr, seg_flagged)
        is_causal = label.startswith("causal_")

        out_reaches.append({
            "reach_id": int(rid),
            "segment_num": int(seg_num) if seg_num is not None else None,
            "start_frame": rs,
            "end_frame": re,
            "label": label,
            "is_causal": is_causal,
            "segment_outcome": _collapse(seg_outcome),
            "segment_ifr": int(seg_ifr) if seg_ifr is not None else None,
        })

    return {
        "video_id": video_id,
        "detector": "assignment_v1",
        "version": "1.0.0",
        "n_reaches": len(out_reaches),
        "reaches": out_reaches,
    }
