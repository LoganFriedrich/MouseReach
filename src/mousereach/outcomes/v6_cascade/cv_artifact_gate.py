"""CV artifact-triage gate for the v6 cascade (Stage-98 CV family).

A ``displaced_sa`` commit is only trustworthy if a real pellet physically
arrived at the landing spot. This gate samples the landing patch directly in
the video: if it did NOT transition empty->bright when the pellet supposedly
landed (normalized change < ``CV_CHANGE_GATE_T``), the tracker latched onto a
pre-existing bright speck (dust, an SA corner marker, a reflection) -- the real
pellet was retrieved, not displaced. Such a commit is triaged for human review
rather than committed as a silent displaced/retrieved cross-misassignment.

Discriminator (per segment):
  R    = the real pellet's brightness, sampled on-pillar pre-interaction (the
         pellet's appearance in THIS video/lighting; used only as a scale).
  Bmed = the landing spot's brightness sampled across the whole pre-interaction
         window (what is there BEFORE the pellet could arrive).
  A    = the landing spot's brightness over the settled SA frames (AFTER).
  change = (A - Bmed) / R.  A real displacement lifts the spot from empty to
  bright (large positive change); an artifact does not (change ~ 0).

Validation (corpus-wide, train+gen, 2026-07-02): true displaced pellets have
change p10=+0.26 (train) / +0.36 (gen); all retrieved-as-displaced artifacts
have change <= +0.05. Threshold 0.10 sits in the gap. It catches every artifact
at ~1.0% (train) / ~1.4% (gen) over-triage, and the over-triaged true displaced
are almost all pellets landing on a genuine pre-existing speck (correctly
suspect). Over-triage generalizes (train ~= gen); the gate never converts a
commit into a WRONG commit -- only into triage.

Requires video access. When no video is available (``video_dir`` is None or the
file is not found), the gate is a no-op and leaves the displaced commit intact,
exactly like Stage 98's CV-skip behavior.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import StageDecision
from .guards import lrun

# Normalized empty->bright change below which a displaced landing is treated as
# a DLC artifact (pre-existing speck) and triaged. Validated 2026-07-02.
CV_CHANGE_GATE_T = 0.10

_PAW = ("RightHand", "RHLeft", "RHOut", "RHRight")

# One-entry VideoCapture cache: detect() processes a single video at a time, so
# all its segments share one path -> one open per video instead of per commit.
_cap_cache = {"path": None, "cap": None}


def _get_cap(video_path):
    import cv2
    if _cap_cache["path"] == str(video_path) and _cap_cache["cap"] is not None:
        return _cap_cache["cap"]
    if _cap_cache["cap"] is not None:
        try:
            _cap_cache["cap"].release()
        except Exception:
            pass
    cap = cv2.VideoCapture(str(video_path))
    _cap_cache["path"] = str(video_path)
    _cap_cache["cap"] = cap
    return cap


def _patch_bright(frame, x, y, hw=3):
    """90th-percentile intensity in a small (pellet-sized) patch -- a bright-
    object detector robust to sub-pixel DLC placement."""
    h, w = frame.shape[:2]
    x = int(round(x)); y = int(round(y))
    x0 = max(0, x - hw); x1 = min(w, x + hw + 1)
    y0 = max(0, y - hw); y1 = min(h, y + hw + 1)
    if x1 <= x0 or y1 <= y0:
        return np.nan
    return float(np.percentile(frame[y0:y1, x0:x1], 90))


def landing_change(dlc_df, seg_start, seg_end, reach_windows, video_path):
    """Return the normalized empty->bright change ``(A - Bmed)/R`` at the
    pellet's SA landing spot, or ``None`` if it cannot be measured.

    A large positive value means a real pellet arrived (dark spot became
    bright); a value near zero means the spot was already bright before the
    pellet could have arrived (a pre-existing speck / static feature).
    """
    try:
        import cv2
    except Exception:
        return None
    ce = seg_end - 5
    sub_raw = dlc_df.iloc[seg_start:ce + 1]
    n = len(sub_raw)
    if n < 20:
        return None
    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    g = compute_pillar_geometry_series(sub)
    cx = g["pillar_cx"].to_numpy(float)
    cy = g["pillar_cy"].to_numpy(float)
    r = g["pillar_r"].to_numpy(float)
    px = sub["Pellet_x"].to_numpy(float)
    py = sub["Pellet_y"].to_numpy(float)
    plk = sub_raw["Pellet_likelihood"].to_numpy(float)
    rad = np.sqrt((px - cx) ** 2 + (py - cy) ** 2) / np.maximum(r, 1e-6)
    conf = plk >= 0.7
    pawx = np.nanmean([sub[f"{k}_x"].to_numpy(float) for k in _PAW], axis=0)
    pawy = np.nanmean([sub[f"{k}_y"].to_numpy(float) for k in _PAW], axis=0)
    rs = [a for (a, b) in (reach_windows or []) if seg_start <= a <= seg_end]
    first = (min(rs) - seg_start) if rs else n // 4
    # On-pillar reference frames: pellet on pillar, high lk, paw clear.
    ref = [i for i in range(0, max(1, first))
           if conf[i] and rad[i] < 1.5 and plk[i] > 0.9
           and np.hypot(pawx[i] - px[i], pawy[i] - py[i]) > 2 * r[i]]
    if not ref:
        ref = [i for i in range(0, max(1, first)) if conf[i] and rad[i] < 2.0][:5]
    pre = list(range(0, max(1, first)))
    idx = np.where(conf & (rad > 3.0))[0]
    if not ref or idx.size < 3 or not pre:
        return None
    win = idx[-60:]
    lx = float(np.median(px[win])); ly = float(np.median(py[win]))
    settled = idx[-30:]
    cap = _get_cap(video_path)
    if cap is None or not cap.isOpened():
        return None

    def grab(fr):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ok, f = cap.read()
        return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if ok else None

    R = []
    for i in ref[:8]:
        f = grab(seg_start + i)
        if f is not None:
            R.append(_patch_bright(f, px[i], py[i]))
    Bs = []
    step = max(1, len(pre) // 20)
    for i in pre[::step][:25]:
        f = grab(seg_start + i)
        if f is not None:
            Bs.append(_patch_bright(f, lx, ly))
    A = []
    for i in settled:
        f = grab(seg_start + i)
        if f is not None:
            A.append(_patch_bright(f, lx, ly))
    if not R or not A or not Bs:
        return None
    R = float(np.nanmedian(R))
    Bmed = float(np.nanmedian(Bs))
    A = float(np.nanmedian(A))
    if not np.isfinite(R) or R <= 1e-6 or not np.isfinite(Bmed) or not np.isfinite(A):
        return None
    return (A - Bmed) / R


def _resolve_video(video_dir, video_id):
    if video_dir is None or not video_id:
        return None
    d = Path(video_dir)
    for ext in (".mp4", ".avi"):
        p = d / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


def wrap_cv_artifact_guard(stage, video_dir):
    """Wrap a stage's ``decide()`` so a ``displaced_sa`` commit whose landing
    spot did not transition empty->bright (change < ``CV_CHANGE_GATE_T``) is
    TRIAGED as a DLC artifact instead of committed. No-op when no video is
    available for the segment."""
    orig = stage.decide

    def decide(seg):
        dec = orig(seg)
        if dec.decision == "commit" and dec.committed_class == "displaced_sa":
            vp = _resolve_video(video_dir, seg.video_id)
            if vp is not None:
                chg = landing_change(seg.dlc_df, seg.seg_start, seg.seg_end,
                                     seg.reach_windows, vp)
                if chg is not None and chg < CV_CHANGE_GATE_T:
                    return StageDecision(
                        decision="triage",
                        reason=(f"cv_artifact_landing_no_pellet_arrival "
                                f"(change={chg:.2f} < {CV_CHANGE_GATE_T}; "
                                f"pellet-colored speck already at landing spot)"),
                        features={"cv_landing_change": float(chg)})
        return dec

    stage.decide = decide
    return stage
