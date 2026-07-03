"""Canonical reach aligner.

Match a GT reach list to an algo reach list **by frame window** and stamp a shared
``matched_id`` on each correspondence, so every analysis joins on frame overlap --
never on a raw ``reach_id`` index.

Why this exists (hard rule): GT reach ids and algo reach ids are independent indices
into their own lists. The moment the algo hallucinates or drops even one reach, every
downstream index shifts and the two id-spaces stop corresponding. A ``reach_id`` is
only ever valid *within* the one output that generated it. Any cross-source reach
comparison MUST be ``GT (start, end)`` vs ``algo (start, end)`` by frame overlap.
This module is the single place that rule is implemented; use it instead of
hand-rolling a match anywhere.

Example
-------
>>> from mousereach.improvement.reach_align import align_reaches
>>> tbl = align_reaches(gt_reaches=[(100, 110), (200, 212)],
...                     algo_reaches=[(60, 70), (101, 109), (200, 214)])
>>> tbl[["matched_id", "status", "gt_start", "algo_start"]]
   matched_id   status  gt_start  algo_start
0           0  matched     100.0       101.0
1           1  matched     200.0       200.0
2           2       fp       NaN        60.0

The GT reach that the algo *dropped* becomes ``fn``; the algo reach with no GT
(a phantom / hallucination) becomes ``fp``. Matched rows carry both sides' frames
plus ``start_delta``/``end_delta`` (algo - gt). Join any per-reach attribute onto the
table via ``gt_index`` / ``algo_index`` (positions in the ORIGINAL input lists).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

import pandas as pd

from .reach_detection.metrics import Reach, match_reaches

ReachLike = Union[Sequence[int], Dict[str, Any], Any]

_COLS = ["matched_id", "status", "gt_index", "gt_start", "gt_end",
         "algo_index", "algo_start", "algo_end", "start_delta", "end_delta"]


def _start_end(item: ReachLike):
    """Pull (start_frame, end_frame) from a tuple/list, a dict (start_frame/end_frame
    or start/end), or an object with those attributes."""
    if isinstance(item, Reach):
        return item.start_frame, item.end_frame
    if isinstance(item, dict):
        s = item.get("start_frame", item.get("start"))
        e = item.get("end_frame", item.get("end"))
    elif isinstance(item, (tuple, list)):
        s, e = item[0], item[1]
    else:
        s = getattr(item, "start_frame", None)
        e = getattr(item, "end_frame", None)
    if s is None or e is None:
        raise ValueError(f"reach has no resolvable (start, end): {item!r}")
    return int(s), int(e)


def _to_reaches(items: Sequence[ReachLike]) -> List[Reach]:
    """Normalize any reach-like list into ``Reach(start, end, index)`` where index is
    the position in the ORIGINAL input list (so the caller can map back)."""
    out: List[Reach] = []
    for i, it in enumerate(items or []):
        s, e = _start_end(it)
        out.append(Reach(start_frame=s, end_frame=e, index=i))
    return out


def align_reaches(gt_reaches: Sequence[ReachLike],
                  algo_reaches: Sequence[ReachLike],
                  *, window: int = 10, strict: bool = False) -> pd.DataFrame:
    """Align GT and algo reaches by frame proximity and stamp a shared ``matched_id``.

    Parameters
    ----------
    gt_reaches, algo_reaches : sequence of reach-like
        Each element may be a ``(start, end)`` tuple/list, a dict with
        ``start_frame``/``end_frame`` (or ``start``/``end``), or an object with those
        attributes. Order does not matter; matching is by frame window.
    window : int, default 10
        Maximum absolute start-frame distance for a match (non-strict). +/-10f is
        already past typical reach duration.
    strict : bool, default False
        Use the canonical strict reach-detection criterion (asymmetric start
        tolerance + span agreement) instead of the +/-window rule.

    Returns
    -------
    pandas.DataFrame
        One row per correspondence with columns ``matched_id`` (unique per row),
        ``status`` ('matched' | 'fn' = GT-only/missed | 'fp' = algo-only/phantom),
        ``gt_index``/``gt_start``/``gt_end``, ``algo_index``/``algo_start``/``algo_end``
        (indices are positions in the original input lists; None on the absent side),
        and ``start_delta``/``end_delta`` (algo - gt; None unless matched).
    """
    gt = _to_reaches(gt_reaches)
    algo = _to_reaches(algo_reaches)
    results = match_reaches(algo, gt, window=window, strict=strict)
    rows: List[Dict[str, Any]] = []
    for mid, r in enumerate(results):
        matched = r.status == "matched"
        rows.append({
            "matched_id": mid,
            "status": r.status,
            "gt_index": r.gt_reach_index,
            "gt_start": r.gt_start,
            "gt_end": r.gt_end,
            "algo_index": r.algo_reach_index,
            "algo_start": r.algo_start,
            "algo_end": r.algo_end,
            "start_delta": (r.algo_start - r.gt_start) if matched else None,
            "end_delta": (r.algo_end - r.gt_end) if matched else None,
        })
    return pd.DataFrame(rows, columns=_COLS)


def causal_reach_row(align_table: pd.DataFrame, interaction_frame: int):
    """Return the aligned row whose GT reach window SPANS ``interaction_frame`` -- the
    causal reach for a touched segment (GT ``interaction_frame`` is the "when was the
    pellet interacted with" annotation; the causal reach is the GT reach spanning it).
    Returns None if no GT reach spans it. Restrict ``align_table`` to one segment first.
    """
    if interaction_frame is None:
        return None
    for _, row in align_table.iterrows():
        gs, ge = row["gt_start"], row["gt_end"]
        if pd.notna(gs) and pd.notna(ge) and gs <= interaction_frame <= ge:
            return row
    return None
