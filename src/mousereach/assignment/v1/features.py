"""
Per-reach feature extraction for the v1 assignment classifier.

The classifier asks: "did this reach cause the pellet's state change?"
The most discriminating signal is what happened TO THE PELLET as a
result of this specific reach -- if the pellet was on the pillar
before and gone after, the reach caused it. If the pellet was already
gone (or never moved), this reach is a miss.

Feature design
--------------
We compute three windows per reach:
  PRE:    [reach.start - PRE_W,  reach.start - 1]    (state before)
  DURING: [reach.start,          reach.end]          (the reach itself)
  POST:   [reach.end + 1,        reach.end + POST_W] (state after)

For each window, we summarize all 18 bodyparts x (x, y, lk) using
mean/min/max. Plus reach-level metadata (span, position in segment,
reach order, n_reaches_in_segment).

Crucially, we also compute DELTA features: post - pre. These directly
encode "what changed during this reach", which is the causal signal.

Per ALL-DLC rule, we include every bodypart's x/y/lk -- not just
pellet/pillar/hand. The model decides what's informative.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from mousereach.reach.v8.features import BODYPARTS
from mousereach.lib.causal_attribution import (
    compute_reach_apex, classify_end_state, find_displaced_signature_runs,
    find_off_pillar_transition_frame, reach_contains_or_precedes_transition,
)
from mousereach.lib.pillar_geometry import (
    compute_pillar_geometry_series, pellet_inside_pillar_circle,
    pellet_dist_from_pillar_center,
)


PRE_WINDOW = 30
POST_WINDOW = 200

# Per-reach causal-attribution feature windows (around APEX, not start/end)
APEX_PRE_WINDOW = 30
APEX_POST_WINDOW = 30
PELLET_HIGH_LK = 0.95
PELLET_VISIBLE_LK = 0.7
PELLET_MISSING_LK = 0.3
PILLAR_VISIBLE_LK = 0.7


def _window_stats(
    dlc_df: pd.DataFrame, start: int, end: int, prefix: str,
) -> Dict[str, float]:
    """Return mean/min/max of x, y, lk for each bodypart within
    [start, end] (inclusive). Empty windows return zeros.
    """
    out: Dict[str, float] = {}
    n = len(dlc_df)
    s = max(0, start)
    e = min(n - 1, end)

    for bp in BODYPARTS:
        cols = (f"{bp}_x", f"{bp}_y", f"{bp}_likelihood")
        for c, suf in zip(cols, ("x", "y", "lk")):
            if c not in dlc_df.columns or e < s:
                out[f"{prefix}_{bp}_{suf}_mean"] = 0.0
                out[f"{prefix}_{bp}_{suf}_min"] = 0.0
                out[f"{prefix}_{bp}_{suf}_max"] = 0.0
                continue
            arr = dlc_df[c].iloc[s:e + 1].to_numpy(dtype=np.float32)
            out[f"{prefix}_{bp}_{suf}_mean"] = float(arr.mean())
            out[f"{prefix}_{bp}_{suf}_min"] = float(arr.min())
            out[f"{prefix}_{bp}_{suf}_max"] = float(arr.max())
    return out


def extract_reach_features(
    dlc_df: pd.DataFrame,
    reach_start: int,
    reach_end: int,
    seg_start: int,
    seg_end: int,
    reach_order: int,            # 0-indexed within segment
    n_reaches_in_segment: int,
    pre_window: int = PRE_WINDOW,
    post_window: int = POST_WINDOW,
) -> Dict[str, float]:
    """Extract the per-reach feature vector for assignment classification."""
    out: Dict[str, float] = {}

    pre = _window_stats(
        dlc_df, reach_start - pre_window, reach_start - 1, prefix="pre")
    during = _window_stats(
        dlc_df, reach_start, reach_end, prefix="during")
    post = _window_stats(
        dlc_df, reach_end + 1, reach_end + post_window, prefix="post")

    out.update(pre)
    out.update(during)
    out.update(post)

    # DELTA features: post - pre (state change across reach)
    for bp in BODYPARTS:
        for suf in ("x", "y", "lk"):
            for stat in ("mean", "min", "max"):
                pre_k = f"pre_{bp}_{suf}_{stat}"
                post_k = f"post_{bp}_{suf}_{stat}"
                out[f"delta_{bp}_{suf}_{stat}"] = post[post_k] - pre[pre_k]

    # Reach-level metadata
    span = reach_end - reach_start + 1
    seg_len = seg_end - seg_start + 1
    out["reach_span"] = float(span)
    out["reach_relative_start"] = float(
        (reach_start - seg_start) / max(1, seg_len))
    out["reach_relative_end"] = float(
        (reach_end - seg_start) / max(1, seg_len))
    out["reach_order"] = float(reach_order)
    out["n_reaches_in_segment"] = float(n_reaches_in_segment)
    out["is_first_reach"] = 1.0 if reach_order == 0 else 0.0
    out["is_last_reach"] = (
        1.0 if reach_order == n_reaches_in_segment - 1 else 0.0)
    out["segment_length"] = float(seg_len)

    return out


def feature_columns() -> List[str]:
    """Canonical column order for the per-reach feature matrix."""
    cols: List[str] = []
    for prefix in ("pre", "during", "post"):
        for bp in BODYPARTS:
            for suf in ("x", "y", "lk"):
                for stat in ("mean", "min", "max"):
                    cols.append(f"{prefix}_{bp}_{suf}_{stat}")
    for bp in BODYPARTS:
        for suf in ("x", "y", "lk"):
            for stat in ("mean", "min", "max"):
                cols.append(f"delta_{bp}_{suf}_{stat}")
    cols.extend([
        "reach_span", "reach_relative_start", "reach_relative_end",
        "reach_order", "n_reaches_in_segment",
        "is_first_reach", "is_last_reach", "segment_length",
    ])
    return cols


# ============================================================================
# CAUSAL-ATTRIBUTION FEATURES (additive, 2026-04-30)
#
# These layer on top of the baseline features above. They encode the
# decision rules from `feature_philosophy_event_anchored_walking.md`:
# - Reach apex = max nose-to-paw distance (NOT start/end)
# - Pre-/post-apex 30f windows for state inspection
# - Per-frame pillar-circle membership using smoothed-SA geometry
# - Displaced signature is segment-scoped, no off-paw filter
# - Causal reach found by back-walking from validated end state
# ============================================================================

def extract_reach_causal_features(
    dlc_df: pd.DataFrame,
    reach_start: int,
    reach_end: int,
    seg_start: int,
    seg_end: int,
    end_state_cache: dict = None,
    transition_cache: dict = None,
    displaced_cache: dict = None,
) -> Dict[str, float]:
    """Per-reach causal-attribution features.

    The three caches let the caller compute end-state, transition frame,
    and displaced-signature once per segment instead of per reach
    (these are segment-level computations).
    """
    out: Dict[str, float] = {}
    apex = compute_reach_apex(reach_start, reach_end, dlc_df)
    out["reach_apex_frame_offset"] = float(apex - reach_start)

    # Pre-apex / post-apex windows (apex-anchored, ≤30 frames)
    pre_s = max(0, apex - APEX_PRE_WINDOW)
    pre_e = max(0, apex - 1)
    post_s = min(len(dlc_df) - 1, apex + 1)
    post_e = min(len(dlc_df) - 1, apex + APEX_POST_WINDOW)

    # Per-frame primitives over the two windows
    def _window_pellet_state(ws: int, we: int) -> Dict[str, float]:
        if we < ws:
            return {"inside_count": 0.0, "off_pillar_visible_count": 0.0,
                    "missing_run": 0.0, "pillar_revealed_count": 0.0,
                    "n_frames": 0.0}
        sub = dlc_df.iloc[ws:we + 1]
        geom = compute_pillar_geometry_series(sub)
        inside = pellet_inside_pillar_circle(sub, pillar_geom=geom).to_numpy()
        plk = sub["Pellet_likelihood"].to_numpy(dtype=float)
        pillar_lk = sub["Pillar_likelihood"].to_numpy(dtype=float)
        dist = pellet_dist_from_pillar_center(sub, pillar_geom=geom).to_numpy()
        radius = geom["pillar_r"].to_numpy()
        off_pillar_visible = (plk >= PELLET_HIGH_LK) & (dist > radius) & np.isfinite(dist)
        # Longest missing run (lk < missing threshold)
        missing = (plk < PELLET_MISSING_LK).astype(int)
        run_len = 0
        max_run = 0
        for v in missing:
            if v:
                run_len += 1
                if run_len > max_run:
                    max_run = run_len
            else:
                run_len = 0
        return {
            "inside_count": float(inside.sum()),
            "off_pillar_visible_count": float(off_pillar_visible.sum()),
            "missing_run": float(max_run),
            "pillar_revealed_count": float((pillar_lk >= PILLAR_VISIBLE_LK).sum()),
            "n_frames": float(len(sub)),
        }

    pre = _window_pellet_state(pre_s, pre_e)
    post = _window_pellet_state(post_s, post_e)
    for k, v in pre.items():
        out[f"pre_apex_{k}"] = v
    for k, v in post.items():
        out[f"post_apex_{k}"] = v

    # Did this reach cause an on->off transition?
    out["pre_apex_inside_pillar_frac"] = (
        pre["inside_count"] / pre["n_frames"]) if pre["n_frames"] > 0 else 0.0
    out["post_apex_inside_pillar_frac"] = (
        post["inside_count"] / post["n_frames"]) if post["n_frames"] > 0 else 0.0
    out["transition_evidence"] = (
        out["pre_apex_inside_pillar_frac"] - out["post_apex_inside_pillar_frac"])

    # Segment-level features (cached)
    if displaced_cache is None:
        displaced_cache = find_displaced_signature_runs(
            dlc_df, seg_start, seg_end)
    out["seg_displaced_signature_run_length"] = float(
        displaced_cache.get("longest_run_length", 0))
    out["seg_displaced_signature_n_qualifying"] = float(
        displaced_cache.get("n_qualifying_frames", 0))

    if end_state_cache is None:
        end_state_cache = classify_end_state(dlc_df, seg_end)
    es_cls = end_state_cache.classification
    out["seg_end_state__on_pillar"] = 1.0 if es_cls == "on_pillar" else 0.0
    out["seg_end_state__off_pillar_visible"] = (
        1.0 if es_cls == "off_pillar_visible" else 0.0)
    out["seg_end_state__missing"] = 1.0 if es_cls == "missing" else 0.0
    out["seg_end_state__ambiguous"] = 1.0 if es_cls == "ambiguous" else 0.0

    if transition_cache is None:
        transition_cache = {
            "frame": find_off_pillar_transition_frame(
                dlc_df, seg_start, seg_end, end_state_cache)
        }
    transition_frame = transition_cache.get("frame")
    out["seg_transition_frame_known"] = (
        1.0 if transition_frame is not None else 0.0)
    out["this_reach_contains_transition"] = 1.0 if (
        transition_frame is not None and
        reach_contains_or_precedes_transition(apex, reach_start, reach_end,
                                              transition_frame)
    ) else 0.0

    # Composite hypothesis-evidence features
    out["evidence_caused_displaced"] = float(
        (out["pre_apex_inside_pillar_frac"] >= 0.5) and
        (displaced_cache.get("longest_run_length", 0) >= 3)
    )
    out["evidence_caused_retrieved"] = float(
        (out["pre_apex_inside_pillar_frac"] >= 0.5) and
        (displaced_cache.get("longest_run_length", 0) == 0) and
        (post["missing_run"] >= 5) and
        (es_cls in ("missing", "off_pillar_visible"))
    )

    return out


def causal_feature_columns() -> List[str]:
    """Canonical column order for the causal-attribution feature block."""
    base = [
        "reach_apex_frame_offset",
        "pre_apex_inside_count", "pre_apex_off_pillar_visible_count",
        "pre_apex_missing_run", "pre_apex_pillar_revealed_count",
        "pre_apex_n_frames",
        "post_apex_inside_count", "post_apex_off_pillar_visible_count",
        "post_apex_missing_run", "post_apex_pillar_revealed_count",
        "post_apex_n_frames",
        "pre_apex_inside_pillar_frac", "post_apex_inside_pillar_frac",
        "transition_evidence",
        "seg_displaced_signature_run_length",
        "seg_displaced_signature_n_qualifying",
        "seg_end_state__on_pillar", "seg_end_state__off_pillar_visible",
        "seg_end_state__missing", "seg_end_state__ambiguous",
        "seg_transition_frame_known", "this_reach_contains_transition",
        "evidence_caused_displaced", "evidence_caused_retrieved",
    ]
    return base
