"""
Believable pellet trace -- the foundation layer for outcome scoring.

This SURFACES primitives the pipeline already computes (the calculated pillar
geometry from ``pillar_geometry``, the on-pillar test, the SA bounding box used
by the v6 cascade stages, pellet confidence) into ONE explicit, per-frame trace
of the pellet's presence and location -- and makes two things trustworthy that
the older ``get_pellet_trajectory`` did not:

  1. **Full-frame.** ``get_pellet_trajectory`` keeps only frames where the
     pellet AND the SA points are all confident, then drops the rest. That
     hides occlusions, gaps, and artifacts. This trace keeps EVERY frame and
     marks its state, so absence / occlusion / artifact are visible -- which is
     the whole point (retrieved is read from sustained real-absence).

  2. **Confidence + physics gated location.** A location is only believed when
     (a) DLC confidence is high enough AND (b) it is physically reachable from
     the last believed location (not a teleport). Low confidence => probably not
     the pellet (artifact). A jump the pellet physically cannot make in the
     elapsed frames => the tracker snapping to the pillar tip / glare, not the
     pellet -- rejected even at high momentary confidence.

The trace is REACH-INDEPENDENT (pellet + tray geometry only), so it is immune to
reach-detector changes. The outcome stages (something-vs-nothing, what, when)
read off this trace.

Geometry note: the SA is a box whose top edge (toward the mouse / the exit) is
SATL-SATR and bottom edge is SABL-SABR; the in-SA test reuses the v6 cascade
bounding-box formula. The reachability budget is a first-pass physics bound to
be GT-calibrated later (see the redesign plan, Phase 5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pillar_geometry import compute_pillar_geometry_series

DEFAULT_LK_PRESENT = 0.7          # DLC confidence below this => not the pellet (Tier-1 impossibility)
DEFAULT_MAX_STEP_RULER = 0.5      # max plausible pellet displacement per frame, in ruler units
                                  # (the SA is ~1 ruler wide; >0.5 ruler in one frame is a teleport).
                                  # FIRST-PASS physics bound -- GT-calibrate in Phase 5.


def _col(dlc_df: pd.DataFrame, name: str, n: int) -> np.ndarray:
    return dlc_df[name].to_numpy(dtype=float) if name in dlc_df.columns else np.full(n, np.nan)


def compute_pellet_trace(
    dlc_df: pd.DataFrame,
    lk_present: float = DEFAULT_LK_PRESENT,
    max_step_ruler: float = DEFAULT_MAX_STEP_RULER,
    pillar_geom: pd.DataFrame = None,
) -> pd.DataFrame:
    """Per-frame believable pellet trace for ``dlc_df`` (whole video or a slice).

    Returns a DataFrame indexed like ``dlc_df`` with columns:
      pellet_x, pellet_y, pellet_lk : raw DLC pellet
      pillar_cx, pillar_cy, pillar_r, ruler : calculated pillar (tray-relative)
      dist_px, dist_ruler           : pellet -> pillar-center distance (px / ruler units)
      conf_ok                       : pellet_lk >= lk_present
      reachable                     : (x,y) reachable from the last believed point (not a teleport)
      present                       : conf_ok AND reachable -- the pellet is really detected here
      on_pillar                     : present AND within the calculated pillar circle
      in_sa                         : present AND within the SA bounding box
      sa_top_y, sa_bot_y, sa_left_x, sa_right_x : SA box edges (for viewers / downstream)
      bel_x, bel_y                  : believed location (raw where present, else NaN)
    """
    n = len(dlc_df)
    px = _col(dlc_df, "Pellet_x", n)
    py = _col(dlc_df, "Pellet_y", n)
    plk = _col(dlc_df, "Pellet_likelihood", n)

    if pillar_geom is None:
        pillar_geom = compute_pillar_geometry_series(dlc_df)
    cx = pillar_geom["pillar_cx"].to_numpy()
    cy = pillar_geom["pillar_cy"].to_numpy()
    r = pillar_geom["pillar_r"].to_numpy()
    ruler = pillar_geom["ruler"].to_numpy()

    dist_px = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        dist_ruler = dist_px / np.where(ruler > 0, ruler, np.nan)

    # SA bounding box (reuse the v6 cascade formula: top=SATL/SATR, bottom=SABL/SABR).
    sabl_x, sabl_y = _col(dlc_df, "SABL_x", n), _col(dlc_df, "SABL_y", n)
    sabr_x, sabr_y = _col(dlc_df, "SABR_x", n), _col(dlc_df, "SABR_y", n)
    satl_x, satl_y = _col(dlc_df, "SATL_x", n), _col(dlc_df, "SATL_y", n)
    satr_x, satr_y = _col(dlc_df, "SATR_x", n), _col(dlc_df, "SATR_y", n)
    sa_top_y = (satl_y + satr_y) / 2.0
    sa_bot_y = (sabl_y + sabr_y) / 2.0
    sa_left_x = np.minimum(sabl_x, satl_x)
    sa_right_x = np.maximum(sabr_x, satr_x)
    in_box = (py >= sa_top_y) & (py <= sa_bot_y) & (px >= sa_left_x) & (px <= sa_right_x)

    conf_ok = plk >= lk_present

    # Reachability (physics): forward-walk. A confident point is believed only if it is
    # within max_step_ruler * ruler of the last believed point per elapsed frame. This
    # rejects teleports -- DLC snapping to the pillar tip / glare while the real pellet is
    # elsewhere or gone -- even when the spurious point has high momentary confidence.
    med_ruler = float(np.nanmedian(ruler)) if np.isfinite(ruler).any() else 0.0
    reachable = np.zeros(n, dtype=bool)
    last_i = -1
    for i in range(n):
        if not conf_ok[i] or not (np.isfinite(px[i]) and np.isfinite(py[i])):
            continue
        if last_i < 0:
            reachable[i] = True
            last_i = i
            continue
        gap = i - last_i
        step = float(np.hypot(px[i] - px[last_i], py[i] - py[last_i]))
        rl = ruler[i] if (np.isfinite(ruler[i]) and ruler[i] > 0) else med_ruler
        budget = max_step_ruler * rl * gap
        if budget <= 0 or step <= budget:
            reachable[i] = True
            last_i = i
        # else: teleport -> not believed; keep last_i and wait for a reachable point.

    present = conf_ok & reachable
    on_pillar = present & (dist_px <= r)
    in_sa = present & in_box
    bel_x = np.where(present, px, np.nan)
    bel_y = np.where(present, py, np.nan)

    return pd.DataFrame({
        "pellet_x": px, "pellet_y": py, "pellet_lk": plk,
        "pillar_cx": cx, "pillar_cy": cy, "pillar_r": r, "ruler": ruler,
        "dist_px": dist_px, "dist_ruler": dist_ruler,
        "conf_ok": conf_ok, "reachable": reachable, "present": present,
        "on_pillar": on_pillar, "in_sa": in_sa,
        "sa_top_y": sa_top_y, "sa_bot_y": sa_bot_y,
        "sa_left_x": sa_left_x, "sa_right_x": sa_right_x,
        "bel_x": bel_x, "bel_y": bel_y,
    }, index=dlc_df.index)
