"""Per-reach triage reduction for algo 4 (assignment v2).

When algo 4 triages a segment (the two causal-reach signals disagree, or algo 3
already triaged it), the whole segment is currently held for human review. But
we frequently already KNOW that most reaches in that segment are misses: a reach
that does not change the pellet's on-pillar state -- on-pillar before AND after
(``on->on``), or off-pillar-in-the-SA before AND after (``off->off``) -- cannot
have displaced or retrieved the pellet, because both outcomes require the pellet
to leave the pillar.

This module **recategorizes** those known-miss reaches from ``triaged`` to
``miss`` (positive scoring -- nothing is deleted; the reach count never changes)
and leaves only the genuinely ambiguous transition reach(es) ``triaged`` for a
human. It NEVER determines an outcome: whether the surviving candidate was a
displacement or a retrieval is algo 3's job, and if the segment is triaged that
job is already exhausted. The only new label this pass writes is ``miss``.

Precision is the whole point (a dropped causal reach corrupts kinematics AND
poisons the training pool), so the rule is deliberately conservative and purely
LOCAL:

  * ``on``  = pellet CONFIDENTLY DETECTED and inside the pillar circle;
  * ``off`` = pellet CONFIDENTLY DETECTED and clearly in the SA (well off the
              pillar). Crucially, *not-detected* is NOT "off" -- a tracking
              dropout must not read as a departure -- it is ``uncertain``;
  * a reach is a ``miss`` iff BOTH its immediate paw-clear pre and post windows
    are the SAME confident state (on->on or off->off);
  * anything else stays ``triaged``: an on->off transition (the departure, a
    genuine candidate), or ANY uncertainty (a None/mixed/undetected read).

The causal reach's pre (on) and post (off) differ by construction, so it can
never be scored a miss. Physics + thresholds are reused verbatim from algo 3's
stage 21 (the immediate on<->off transition detector).

Residual triage kind (why a reach still needs a human):
    reach_uncertain   -- outcome known (algo-4 disagreement), which reach unsure
    outcome_uncertain -- reach known (single candidate), what happened unsure
    both_uncertain    -- neither pinned
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series

# Reuse algo 3's exact per-reach on-pillar test + thresholds (stage 21). We
# import rather than fork so the two stay identical by construction.
from mousereach.outcomes.v6_cascade.stage_21_causal_reach_via_immediate_on_off_transition import (  # noqa: E501
    _check_on_pillar_in_window,
    PAW_BODYPARTS,
    PELLET_LK_HIGH,
    PAW_LK_THR,
    ON_PILLAR_RADII,
    PELLET_OFF_PILLAR_RADII_FOR_SA,
    IMMEDIATE_WINDOW_FRAMES,
    MIN_PAW_CLEAR_FRAMES_REQUIRED,
)
from mousereach.lib.cv_pellet_presence import CV_ON

# Residual triage-kind labels (why a reach still needs review).
TRIAGE_REACH_UNCERTAIN = "reach_uncertain"      # outcome known, which reach unsure
TRIAGE_OUTCOME_UNCERTAIN = "outcome_uncertain"  # reach known, what happened unsure
TRIAGE_BOTH_UNCERTAIN = "both_uncertain"        # neither pinned


def compute_pellet_states(
    raw_dlc_df: pd.DataFrame, seg_start: int, seg_end: int,
    cv_state: Optional[np.ndarray] = None, cv_valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame ``on_pillar``, ``off_pillar`` and ``paw_past_y`` boolean arrays
    over [seg_start, seg_end], computed exactly as algo 3 stage 21 does
    (per-segment cleaning incl. the Pellet; raw likelihoods for the detection
    gates, cleaned coordinates for geometry).

    on_pillar  : detected (lk >= high) AND within the pillar circle (<= 1.0 r).
    off_pillar : detected (lk >= high) AND clearly in the SA (> 1.5 r).
    A frame that is neither (undetected, or in the 1.0-1.5 r edge zone) counts
    as neither -- i.e. 'can't tell', so it never proves a miss.

    CV confluence (optional): the small pellet keypoint drops out (-> ``uncertain``)
    and can misfire off-pillar. When per-video CV states are supplied
    (``cv_state``/``cv_valid``, whole-video arrays aligned to ``raw_dlc_df``; see
    ``mousereach.lib.cv_pellet_presence.compute_cv_states``), CV is used ONLY to
    ADD on-pillar reads: a paw-clear ``CV_ON`` frame (a pellet-sized sphere found
    ON the pillar, pillar-first) becomes on-pillar. This recovers on-pillar frames
    the keypoint dropped AND overrides a fragile keypoint ``off`` when the sphere
    is really on the pillar. CV NEVER asserts ``off`` -- off stays keypoint-derived,
    merely suppressed where on -- so it can only move a frame toward on and can
    never invent an ``off`` that would score a real departure a miss.
    """
    sub_raw = raw_dlc_df.iloc[seg_start:seg_end + 1]
    n = len(sub_raw)
    z = np.zeros(0, dtype=bool)
    if n == 0:
        return z, z, z

    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    geom = compute_pillar_geometry_series(sub)
    pillar_cx = geom["pillar_cx"].to_numpy(dtype=float)
    pillar_cy = geom["pillar_cy"].to_numpy(dtype=float)
    pillar_r = geom["pillar_r"].to_numpy(dtype=float)
    slit_y_line = pillar_cy + pillar_r

    pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
    pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
    pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
    dist_radii = (np.sqrt((pellet_x - pillar_cx) ** 2 + (pellet_y - pillar_cy) ** 2)
                  / np.maximum(pillar_r, 1e-6))

    paw_past_y = np.zeros(n, dtype=bool)
    for bp in PAW_BODYPARTS:
        py = sub[f"{bp}_y"].to_numpy(dtype=float)
        pl = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
        paw_past_y |= (py <= slit_y_line) & (pl >= PAW_LK_THR)

    detected = pellet_lk >= PELLET_LK_HIGH
    on_pillar = detected & (dist_radii <= ON_PILLAR_RADII) & (~paw_past_y)
    off_pillar = detected & (dist_radii > PELLET_OFF_PILLAR_RADII_FOR_SA) & (~paw_past_y)

    # CV confluence: CV is used ONLY to ADD on-pillar reads -- it recovers an
    # on-pillar frame the keypoint dropped, and overrides a fragile keypoint
    # 'off' when the sphere is actually on the pillar (pillar-first localization).
    # It NEVER asserts 'off': off stays keypoint-derived and is merely suppressed
    # where CV/keypoint sees on. So CV can only turn a frame toward on and can
    # never invent an 'off' that would score a real causal reach a miss.
    if cv_state is not None and cv_valid is not None:
        st = np.asarray(cv_state[seg_start:seg_end + 1])
        vv = np.asarray(cv_valid[seg_start:seg_end + 1], dtype=bool)
        if st.shape[0] == n and vv.shape[0] == n:
            cv_on = vv & (~paw_past_y) & (st == CV_ON)
            on_pillar = on_pillar | cv_on
            off_pillar = off_pillar & (~on_pillar)
    return on_pillar, off_pillar, paw_past_y


def _state_in_window(
    on_pillar: np.ndarray, off_pillar: np.ndarray, paw_past_y: np.ndarray,
    window_start: int, window_end: int, direction: str,
) -> str:
    """Return ``"on"``, ``"off"`` or ``"uncertain"`` for the paw-clear frames
    immediately adjacent to a reach boundary.

    Uses algo 3's ``_check_on_pillar_in_window`` over the same paw-clear frames
    for both masks: ALL clear frames on-pillar -> on; ALL clear frames
    detected-off-pillar -> off; anything mixed, undetected, or too few
    paw-clear frames -> uncertain.
    """
    is_on, _, _ = _check_on_pillar_in_window(
        on_pillar, paw_past_y, window_start=window_start, window_end=window_end,
        direction=direction, immediate_window_frames=IMMEDIATE_WINDOW_FRAMES,
        min_paw_clear_required=MIN_PAW_CLEAR_FRAMES_REQUIRED,
    )
    if is_on is True:
        return "on"
    is_off, _, _ = _check_on_pillar_in_window(
        off_pillar, paw_past_y, window_start=window_start, window_end=window_end,
        direction=direction, immediate_window_frames=IMMEDIATE_WINDOW_FRAMES,
        min_paw_clear_required=MIN_PAW_CLEAR_FRAMES_REQUIRED,
    )
    if is_off is True:
        return "off"
    return "uncertain"


def classify_triaged_reaches(
    reach_windows_local: List[Tuple[int, int]],
    on_pillar: np.ndarray,
    off_pillar: np.ndarray,
    paw_past_y: np.ndarray,
    cv_verified: bool = False,
) -> List[str]:
    """Return ``"miss"`` or ``"triaged"`` for each reach (in the given order).

    Two miss rules; everything else stays ``triaged``:

      * ``on->on`` -- pellet confidently on the pillar before AND after the
        reach: it never left, so the reach did nothing.
      * ``off->off`` -- pellet confidently in the SA (off the pillar) before AND
        after the reach: it was already gone, so this reach cannot have taken a
        pillar-pellet.

    Both are miss states by the same physics: displacement AND retrieval both
    require the pellet to leave the pillar, so a reach across which the pellet's
    on/off state did not change did nothing to it. A confident ``off`` requires
    the pellet to be DETECTED clearly in the SA -- a mere dropout reads
    ``uncertain``, not ``off`` -- so an ``off`` genuinely means the pellet was
    seen off the pillar, not just untracked.

    The causal reach is an ``on->off`` transition -- its pre (on) and post (off)
    differ by construction, so it can never be scored a miss -- and any uncertain
    (not-detected / mixed) read stays ``triaged`` too. CV resolution (see
    ``compute_pellet_states``) recovers the causal reach's true ``on`` pre-read
    when the keypoint dropped it, so a genuine departure stays ``on->off`` rather
    than collapsing to a spurious ``off->off``.
    """
    n = len(on_pillar)
    states = [
        (_state_in_window(on_pillar, off_pillar, paw_past_y, 0, ls, "before"),
         _state_in_window(on_pillar, off_pillar, paw_past_y, le + 1, n, "after"))
        for (ls, le) in reach_windows_local
    ]
    # First confident on->off transition establishes that the pellet has left --
    # used only for the no-CV fallback gate below.
    departure_idx: Optional[int] = None
    for i, (pre, post) in enumerate(states):
        if pre == "on" and post == "off":
            departure_idx = i
            break
    labels: List[str] = []
    for i, (pre, post) in enumerate(states):
        if pre == "on" and post == "on":
            labels.append("miss")     # never left the pillar
        elif pre == "off" and post == "off":
            # off->off is physically a miss (pellet already off the pillar). When
            # the off states are CV-verified we trust it unconditionally. Without
            # CV the keypoint 'off' is fragile, so we keep the conservative gate:
            # only score off->off a miss AFTER a confident on->off departure has
            # been established, so a false off on the causal reach's pre-window
            # cannot score it a miss.
            if cv_verified or (departure_idx is not None and i > departure_idx):
                labels.append("miss")
            else:
                labels.append("triaged")
        else:
            labels.append("triaged")  # on->off departure, or any uncertain read

    # Recursion / departure-window (ON side, pellet-identity-safe). This pellet
    # leaves the pillar once. The FIRST confident on->off in the segment IS that
    # departure -- a LATER on-run is the NEXT pellet arriving at the boundary, not
    # this one, so we must bound to the first departure (naively taking the last
    # on-frame conflates the two pellets and blows up the window). Any reach
    # ending before the last on-frame of that first on-run was entirely before the
    # departure (pellet still on) -> miss, even if its own windows read uncertain.
    # The causal reach brackets the departure (on at its start), so a strict
    # inequality never rules it out. The off side is intentionally omitted: a
    # spurious keypoint 'off' is fragile, so we only ever rule out on the
    # CV-robust on side. Under tracking noise this degrades to conservative (fewer
    # misses), never unsafe.
    on_idx = np.flatnonzero(on_pillar)
    off_idx = np.flatnonzero(off_pillar)
    last_on = -1
    if on_idx.size and off_idx.size:
        first_on = int(on_idx[0])
        off_after_first_on = off_idx[off_idx > first_on]
        if off_after_first_on.size:
            first_off = int(off_after_first_on[0])          # this pellet's departure
            on_before_dep = on_idx[on_idx < first_off]
            if on_before_dep.size:
                last_on = int(on_before_dep[-1])
    if last_on >= 0:
        for i, (ls, le) in enumerate(reach_windows_local):
            if labels[i] == "triaged" and le < last_on:
                labels[i] = "miss"      # pellet still on-pillar after this reach
    return labels


def triage_reason(outcome_known: bool, n_candidates: int) -> str:
    """Why the residual candidate(s) still need a human.

    outcome_known == True  -> algo-4 disagreement: the cascade already committed
    the outcome, so the only open question is which reach -> reach_uncertain.
    outcome_known == False -> algo 3 could not determine the outcome; if we
    pinned a single candidate reach the open question is what happened
    (outcome_uncertain), otherwise both are open (both_uncertain).
    """
    if outcome_known:
        return TRIAGE_REACH_UNCERTAIN
    if n_candidates == 1:
        return TRIAGE_OUTCOME_UNCERTAIN
    return TRIAGE_BOTH_UNCERTAIN


def reduce_triaged_segment(
    raw_dlc_df: pd.DataFrame,
    seg_start: int,
    seg_end: int,
    seg_reaches: List[Dict],
    outcome_known: bool,
    cv_state: Optional[np.ndarray] = None,
    cv_valid: Optional[np.ndarray] = None,
) -> Tuple[Dict[int, str], Optional[str]]:
    """Recategorize the known-miss reaches of one triaged segment.

    Parameters
    ----------
    raw_dlc_df : DataFrame
        The raw (un-cleaned) DLC for the whole video.
    seg_start, seg_end : int
        Absolute segment frame bounds.
    seg_reaches : list of dict
        Reaches in this segment; each needs ``reach_id``, ``start_frame``,
        ``end_frame`` (absolute frames).
    outcome_known : bool
        True if the cascade committed an outcome for this segment (algo-4
        disagreement triage); False if algo 3 itself triaged (outcome unknown).

    Returns
    -------
    (labels, reason)
        labels : {reach_id: "miss" | "triaged"}
        reason : the residual triage-kind string.
    """
    if not seg_reaches:
        return {}, None

    on_pillar, off_pillar, paw_past_y = compute_pellet_states(
        raw_dlc_df, seg_start, seg_end, cv_state=cv_state, cv_valid=cv_valid)
    n = len(on_pillar)
    if n == 0:
        return ({r["reach_id"]: "triaged" for r in seg_reaches},
                triage_reason(outcome_known, len(seg_reaches)))

    order = sorted(range(len(seg_reaches)), key=lambda k: int(seg_reaches[k]["start_frame"]))
    local: List[Tuple[int, int]] = []
    for k in order:
        r = seg_reaches[k]
        # Clamp BOTH ends into [0, n-1]. A reach whose frames fall outside the
        # segment bounds (a mis-assigned reach from the caller) must not drive an
        # out-of-range window index into the state arrays -- that raises
        # IndexError deep in the stage-21 window scan. Clamping keeps the pass
        # robust; a genuinely out-of-range reach collapses to an edge window and
        # reads 'uncertain' (stays a candidate), never crashes.
        ls = min(n - 1, max(0, int(r["start_frame"]) - seg_start))
        le = min(n - 1, max(0, int(r["end_frame"]) - seg_start))
        local.append((ls, le))

    reach_labels = classify_triaged_reaches(
        local, on_pillar, off_pillar, paw_past_y, cv_verified=(cv_state is not None))

    labels: Dict[int, str] = {}
    for pos, k in enumerate(order):
        labels[seg_reaches[k]["reach_id"]] = reach_labels[pos]

    n_candidates = sum(1 for v in labels.values() if v == "triaged")

    # Anomaly guard (precision-first): a triaged segment definitionally had a
    # non-miss outcome, so it must retain at least one candidate. If the pass
    # somehow scored EVERY reach a miss, we could not attribute the outcome to
    # any reach -- do NOT silently drop it; revert the whole segment to triaged.
    if n_candidates == 0:
        labels = {rid: "triaged" for rid in labels}
        n_candidates = len(labels)

    return labels, triage_reason(outcome_known, n_candidates)
