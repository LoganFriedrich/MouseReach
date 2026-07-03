"""
Per-reach assignment v2: two-signal AGREEMENT GATE for causal attribution.

Builds on v1's IFR-containment rule by adding an independent
displacement-based signal. For each touched segment, v2 picks the
causal reach two ways:

  (A) IFR-pick -- the reach whose [start, end] contains the segment's
      cascade ``interaction_frame`` (same as v1).
  (B) displacement-pick -- the reach with the strongest on-pillar to
      off-pillar pellet transition, measured via median pellet radius
      (distance to pillar center / pillar_r) in before/after windows.

If both signals agree (same reach by frame overlap), the reach is
COMMITTED as causal. If they disagree, the whole segment is TRIAGED
for manual review.

Per-reach label set (same as v1 plus triaged-on-disagreement):
  - ``causal_retrieved``
  - ``causal_displaced_sa``
  - ``causal_abnormal_exception``
  - ``miss``
  - ``triaged``
  - ``unassigned``
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers carried over from v1
# ---------------------------------------------------------------------------

def _collapse(o: Optional[str]) -> Optional[str]:
    if o == "displaced_outside":
        return "displaced_sa"
    return o


def _segment_for_frame(
    frame: int,
    segment_bounds: Sequence[Tuple[int, int]],
) -> Optional[int]:
    """Returns the index of the segment containing ``frame``."""
    for i, (lo, hi) in enumerate(segment_bounds):
        if lo <= frame <= hi:
            return i
    return None


# ---------------------------------------------------------------------------
# Signal A: IFR containment (identical to v1)
# ---------------------------------------------------------------------------

def _ifr_pick(
    reaches_in_seg: List[Tuple[int, int, int]],
    interaction_frame: int,
) -> Optional[int]:
    """Return the index (into reaches_in_seg) of the reach whose
    [start, end] contains ``interaction_frame``, or None."""
    for idx, (rs, re, _orig_idx) in enumerate(reaches_in_seg):
        if rs <= interaction_frame <= re:
            return idx
    return None


# ---------------------------------------------------------------------------
# Signal B: displacement pick (on-pillar -> off-pillar transition)
# ---------------------------------------------------------------------------

def _compute_pellet_radius_series(
    dlc_df: pd.DataFrame,
    pillar_geom: pd.DataFrame,
    lk_threshold: float = 0.5,
) -> np.ndarray:
    """Per-frame pellet radius = dist(pellet, pillar_center) / pillar_r.
    NaN where Pellet_likelihood < lk_threshold."""
    px = dlc_df["Pellet_x"].to_numpy(dtype=float)
    py = dlc_df["Pellet_y"].to_numpy(dtype=float)
    plk = dlc_df["Pellet_likelihood"].to_numpy(dtype=float)
    cx = pillar_geom["pillar_cx"].to_numpy()
    cy = pillar_geom["pillar_cy"].to_numpy()
    pr = pillar_geom["pillar_r"].to_numpy()

    dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    radius = dist / pr
    radius[plk < lk_threshold] = np.nan
    return radius


def _displacement_pick(
    reaches_in_seg: List[Tuple[int, int, int]],
    radius: np.ndarray,
    seg_start: int,
    seg_end: int,
) -> Optional[int]:
    """Pick the reach with the strongest on-pillar -> off-pillar
    displacement score.

    For each reach (rs, re) in the segment, compute:
      before = median pellet radius over [max(rs-15, prev_reach_end+1), rs)
      after  = median pellet radius over (re, min(re+31, next_reach_start)]

    If >60% of the after-window is NaN (pellet absent), treat after as
    6.0 (vanished = displaced).

    score = (after - before) if before < 2.5 else (after - before) * 0.3

    Returns index into reaches_in_seg of the max-score reach, or None
    if no reach has a computable score.
    """
    if not reaches_in_seg:
        return None

    n_reaches = len(reaches_in_seg)
    scores = []

    for i, (rs, re, _orig_idx) in enumerate(reaches_in_seg):
        # Before window: [max(rs-15, prev_reach_end+1), rs)
        if i == 0:
            before_lo = max(rs - 15, seg_start)
        else:
            prev_re = reaches_in_seg[i - 1][1]
            before_lo = max(rs - 15, prev_re + 1)
        before_hi = rs  # exclusive upper bound (up to but not including rs)

        if before_lo < before_hi:
            before_vals = radius[before_lo:before_hi]
        else:
            before_vals = np.array([])

        # After window: (re, min(re+31, next_reach_start)]
        # Capped at next reach's start to avoid bleeding
        after_lo = re + 1  # exclusive lower bound
        if i < n_reaches - 1:
            next_rs = reaches_in_seg[i + 1][0]
            after_hi = min(re + 31, next_rs)
        else:
            after_hi = min(re + 31, seg_end + 1)

        if after_lo < after_hi:
            after_vals = radius[after_lo:after_hi]
        else:
            after_vals = np.array([])

        # Compute medians (suppress all-NaN warnings -- expected when
        # pellet likelihood is below threshold for the entire window)
        if len(before_vals) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                before_med = float(np.nanmedian(before_vals))
        else:
            before_med = np.nan

        if len(after_vals) > 0:
            nan_frac = float(np.sum(np.isnan(after_vals))) / len(after_vals)
            if nan_frac > 0.6:
                after_med = 6.0  # pellet vanished = displaced
            else:
                after_med = float(np.nanmedian(after_vals))
        else:
            after_med = np.nan

        if np.isnan(before_med) or np.isnan(after_med):
            scores.append(float("-inf"))
            continue

        diff = after_med - before_med
        if before_med < 2.5:
            score = diff
        else:
            score = diff * 0.3

        scores.append(score)

    if all(s == float("-inf") for s in scores):
        return None

    return int(np.argmax(scores))


# ---------------------------------------------------------------------------
# Frame-overlap comparison for agreement check
# ---------------------------------------------------------------------------

def _reaches_overlap(
    reach_a: Tuple[int, int],
    reach_b: Tuple[int, int],
) -> bool:
    """True if two reach windows refer to the same physical reach
    (have any frame overlap)."""
    return reach_a[0] <= reach_b[1] and reach_b[0] <= reach_a[1]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def assign_reaches_v2(
    *,
    reaches: List[Dict],
    segments_with_outcomes: List[Dict],
    dlc_df: pd.DataFrame,
    video_id: Optional[str] = None,
    window: int = 10,
) -> Dict:
    """Join per-video reaches with per-segment cascade outcomes using
    the two-signal agreement gate.

    Parameters
    ----------
    reaches : list of dict
        Each reach must have ``start_frame`` and ``end_frame``.
        Optional: ``reach_id`` (auto-assigned 0..n-1 if missing).
    segments_with_outcomes : list of dict
        v6 cascade output -- each segment has ``segment_num``,
        ``outcome``, ``interaction_frame``, ``outcome_known_frame``,
        and ``flagged_for_review``. Must also have ``start_frame`` and
        ``end_frame``.
    dlc_df : pd.DataFrame
        Raw DLC dataframe for the video (used by displacement-pick
        signal B). Must contain Pellet_x/y/likelihood and SA columns.
    video_id : str, optional
        Stamped onto the output for traceability.
    window : int, default 10
        Not used by the core logic but kept for API symmetry with the
        reach_align module.

    Returns
    -------
    dict in the standard reach-assignments JSON shape (same as v1).
    """
    from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
    from mousereach.lib.pillar_geometry import compute_pillar_geometry_series

    # --- Precompute pillar geometry on cleaned DLC ---
    cleaned_dlc = clean_dlc_bodyparts(dlc_df)
    pillar_geom = compute_pillar_geometry_series(cleaned_dlc)
    radius = _compute_pellet_radius_series(cleaned_dlc, pillar_geom)

    # --- Build segment bounds index ---
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

    # --- Group reaches by segment ---
    # Each entry: (start_frame, end_frame, reach_list_index)
    reach_records: List[Tuple[int, int, int]] = []
    reach_to_seg_idx: List[Optional[int]] = []

    for ri, r in enumerate(reaches):
        rs = int(r.get("start_frame"))
        re_ = int(r.get("end_frame"))
        reach_records.append((rs, re_, ri))

        seg_num_in: Optional[int] = r.get("segment_num")
        seg_idx: Optional[int] = None
        if seg_num_in is not None:
            seg_idx = seg_index_by_num.get(int(seg_num_in))
        if seg_idx is None:
            mid = (rs + re_) // 2
            seg_idx = _segment_for_frame(mid, seg_bounds)
        reach_to_seg_idx.append(seg_idx)

    # Build per-segment reach lists (sorted by start frame)
    reaches_by_seg: Dict[int, List[Tuple[int, int, int]]] = {}
    for ri, seg_idx in enumerate(reach_to_seg_idx):
        if seg_idx is not None:
            reaches_by_seg.setdefault(seg_idx, []).append(reach_records[ri])
    for seg_idx in reaches_by_seg:
        reaches_by_seg[seg_idx].sort(key=lambda x: x[0])

    # --- Determine agreement per touched segment ---
    # For each touched segment, run both signals and decide commit/triage.
    # seg_decision[seg_idx] = ("commit", causal_reach_list_idx) or ("triage", None)
    TOUCHED_OUTCOMES = {"displaced_sa", "displaced_outside", "retrieved"}
    seg_decision: Dict[int, Tuple[str, Optional[int]]] = {}

    for seg_idx, seg in enumerate(segments_with_outcomes):
        outcome = seg.get("outcome")
        flagged = bool(seg.get("flagged_for_review", False))

        if outcome == "triaged" or flagged:
            # Already triaged by cascade -- propagate
            seg_decision[seg_idx] = ("triage", None)
            continue

        collapsed = _collapse(outcome)
        if collapsed not in TOUCHED_OUTCOMES:
            # Untouched / unknown -- no causal reach to assign
            continue

        ifr = seg.get("interaction_frame")
        if ifr is None:
            # No IFR -> cannot run signal A -> triage
            seg_decision[seg_idx] = ("triage", None)
            continue

        ifr = int(ifr)
        seg_reaches = reaches_by_seg.get(seg_idx, [])
        if not seg_reaches:
            # No reaches in this segment -> nothing to assign
            continue

        sf = int(seg.get("start_frame", 0))
        ef = int(seg.get("end_frame", len(radius) - 1))

        # Signal A: IFR containment
        ifr_idx = _ifr_pick(seg_reaches, ifr)

        # Signal B: displacement
        disp_idx = _displacement_pick(seg_reaches, radius, sf, ef)

        if ifr_idx is not None and disp_idx is not None:
            # Check agreement by frame overlap
            ifr_reach = (seg_reaches[ifr_idx][0], seg_reaches[ifr_idx][1])
            disp_reach = (seg_reaches[disp_idx][0], seg_reaches[disp_idx][1])
            if _reaches_overlap(ifr_reach, disp_reach):
                # AGREE -> commit the IFR-picked reach
                causal_orig_idx = seg_reaches[ifr_idx][2]
                seg_decision[seg_idx] = ("commit", causal_orig_idx)
            else:
                # DISAGREE -> triage
                seg_decision[seg_idx] = ("triage", None)
        else:
            # One or both signals absent -> cannot confirm agreement.
            # Triage for safety (absence is not agreement).
            seg_decision[seg_idx] = ("triage", None)

    # --- Build output reach list ---
    out_reaches: List[Dict] = []

    for ri, r in enumerate(reaches):
        rs = int(r.get("start_frame"))
        re_ = int(r.get("end_frame"))
        rid = r.get("reach_id", ri)
        seg_idx = reach_to_seg_idx[ri]

        if seg_idx is None:
            out_reaches.append({
                "reach_id": int(rid),
                "segment_num": None,
                "start_frame": rs,
                "end_frame": re_,
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

        collapsed = _collapse(seg_outcome)

        decision = seg_decision.get(seg_idx)

        if decision is not None:
            action, causal_ri = decision
            if action == "triage":
                label = "triaged"
            elif action == "commit" and causal_ri == ri:
                label = f"causal_{collapsed}"
            else:
                label = "miss"
        else:
            # Non-touched segment (untouched, etc.) or no decision made
            # -- all reaches in these segments are misses
            if collapsed == "triaged" or bool(seg.get("flagged_for_review", False)):
                label = "triaged"
            else:
                label = "miss"

        is_causal = label.startswith("causal_")

        out_reaches.append({
            "reach_id": int(rid),
            "segment_num": int(seg_num) if seg_num is not None else None,
            "start_frame": rs,
            "end_frame": re_,
            "label": label,
            "is_causal": is_causal,
            "segment_outcome": collapsed,
            "segment_ifr": int(seg_ifr) if seg_ifr is not None else None,
        })

    return {
        "video_id": video_id,
        "detector": "assignment_v2",
        "version": "2.0.0",
        "n_reaches": len(out_reaches),
        "reaches": out_reaches,
    }
