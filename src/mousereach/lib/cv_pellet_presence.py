"""CV pellet localization at/around the CALCULATED pillar.

The pellet is a white sphere. CV LOCATES it each frame (it does not merely ask
"is the pillar bright"), which lets us verify the pellet's on/off state from the
pixels instead of trusting the small, drop-prone pellet keypoint:

  * ON   -- a pellet-sized bright blob sits ON the calculated pillar.
  * OFF  -- a pellet-sized bright blob sits clearly OFF the pillar (in the SA):
            the pellet really is over there, so it has left the pillar.
  * GONE -- no pellet-sized bright blob anywhere near the pillar/SA: the pellet
            was removed from the pillar at/before this point.
  * UNCERTAIN -- can't tell (paw occluding, uncalibratable video, unreadable
            frame): CV abstains and the keypoint is used instead.

Why this matters: the keypoint's *off* read is unreliable -- a poorly tracked
on-pillar pellet can be misread as off, and an ``off->off`` miss rule riding on
that drops real causal reaches. Localizing the sphere in the pixels fixes it:
where the keypoint falsely says off, CV sees the sphere still ON the pillar.

Calibration is PER VIDEO (absolute white level shifts with lighting) and
SELF-CONTAINED -- the on-brightness is learned from the frames where the pellet
keypoint is confident, so there is nothing for a downstream lab to tune.

Safety: for the reduction, ON maps to on-pillar and OFF/GONE map to off-pillar;
UNCERTAIN falls back to the keypoint. CV only *decides* a frame when it can
localize the sphere with a paw-clear view, so it does not invent an ``off`` that
would kill a genuine departure.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 always present in the mousereach env
    cv2 = None

# Per-frame CV verdicts.
CV_UNCERTAIN = 0
CV_ON = 1
CV_OFF = 2
CV_GONE = 3

# Classification radii (in units of the calc-pillar radius r).
ON_RADII = 1.3          # blob centroid within 1.3 r of pillar centre -> ON
SA_MAX_RADII = 12.0     # a pellet-sized blob out to 12 r counts as OFF (in SA);
                        # farther than that it is not plausibly the pellet
# Pellet blob area window, in units of the pillar-disk area (pi r^2). The white
# sphere is a similar scale to the pillar top; the paw is much larger, so an
# upper bound rejects the paw.
AREA_MIN_FACTOR = 0.15
AREA_MAX_FACTOR = 8.0


def _disk_p90(gray: np.ndarray, cx: float, cy: float, rad: float) -> float:
    if np.isnan(cx) or np.isnan(cy):
        return np.nan
    h, w = gray.shape
    x0, x1 = max(0, int(cx - rad)), min(w, int(cx + rad) + 1)
    y0, y1 = max(0, int(cy - rad)), min(h, int(cy + rad) + 1)
    if x1 <= x0 or y1 <= y0:
        return np.nan
    ys, xs = np.mgrid[y0:y1, x0:x1]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= rad ** 2
    vals = gray[y0:y1, x0:x1][mask]
    return float(np.percentile(vals, 90)) if vals.size else np.nan


def _calibrate_threshold(
    mp4_path: str, cx: np.ndarray, cy: np.ndarray, pillar_r: np.ndarray,
    on_kp: np.ndarray, off_kp: np.ndarray, n_sample: int = 120,
    min_confident: int = 10, margin: float = 10.0,
) -> Optional[float]:
    """Per-video pellet-white threshold, learned from confident keypoint frames.
    Sits above the confident-OFF pillar-disk brightness (so a bright blob really
    is the sphere) yet at/below the confident-ON low tail. None if not
    calibratable."""
    if cv2 is None:
        return None
    on_idx = np.where(on_kp)[0]
    off_idx = np.where(off_kp)[0]
    if on_idx.size < min_confident:
        return None
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    rng = np.random.RandomState(0)

    def _sample(idx):
        if idx.size == 0:
            return np.array([])
        pick = np.sort(rng.choice(idx, min(n_sample, idx.size), replace=False))
        vals = []
        for f in pick:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, im = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            rad = max(4.0, 1.5 * (pillar_r[f] if not np.isnan(pillar_r[f]) else 0.0))
            vals.append(_disk_p90(gray, cx[f], cy[f], rad))
        return np.array([v for v in vals if not np.isnan(v)])

    on_b = _sample(on_idx)
    off_b = _sample(off_idx)
    cap.release()
    if on_b.size < min_confident:
        return None
    on_lo = float(np.percentile(on_b, 20))
    off_hi = float(np.percentile(off_b, 95)) + margin if off_b.size >= min_confident else 0.0
    thr = max(on_lo, off_hi)
    if off_b.size >= min_confident and thr <= float(np.median(off_b)) + margin:
        return None
    return thr


def compute_cv_states(
    mp4_path: str, cx: np.ndarray, cy: np.ndarray, pillar_r: np.ndarray,
    on_kp: np.ndarray, off_kp: np.ndarray, paw_past_y: np.ndarray,
    pellet_x: np.ndarray, pellet_y: np.ndarray, pellet_lk: np.ndarray,
    frames_needed: Optional[Iterable[int]] = None,
    pellet_lk_high: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Per-frame CV pellet localization.

    frames_needed: if given, only those frame indices are localized (via seeks) --
    everything else stays CV_UNCERTAIN and falls back to the keypoint. This lets
    a caller localize ONLY the triaged segments' windows instead of decoding the
    whole video (the reduction only consults those frames). If None, every frame
    is localized in one sequential pass. Calibration always uses a video-wide
    sample of confident keypoint frames regardless.

    Returns (cv_state, cv_valid, threshold):
      cv_state : int8 array of CV_ON / CV_OFF / CV_GONE / CV_UNCERTAIN.
      cv_valid : bool, True where CV actually decided (calibrated, frame read,
                 paw clear). Where False, the caller keeps the keypoint state.
      threshold: the per-video pellet-white threshold (None if uncalibratable).
    """
    n = len(cx)
    cv_state = np.full(n, CV_UNCERTAIN, dtype=np.int8)
    cv_valid = np.zeros(n, dtype=bool)
    if cv2 is None:
        return cv_state, cv_valid, None
    thr = _calibrate_threshold(mp4_path, cx, cy, pillar_r, on_kp, off_kp)
    if thr is None:
        return cv_state, cv_valid, None

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return cv_state, cv_valid, None

    def _localize(im, f):
        # CV abstains when the paw is in the SA (it can occlude / be mistaken for
        # the pellet) -- keep those frames for the keypoint.
        if paw_past_y[f] or np.isnan(cx[f]) or np.isnan(cy[f]) or np.isnan(pillar_r[f]):
            return
        r = max(1.0, pillar_r[f])
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        bw = (gray >= thr).astype(np.uint8)
        _, _, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
        disk_area = np.pi * r * r
        amin, amax = AREA_MIN_FACTOR * disk_area, AREA_MAX_FACTOR * disk_area
        on_found = off_found = False
        for i in range(1, stats.shape[0]):
            area = stats[i, cv2.CC_STAT_AREA]
            if not (amin <= area <= amax):
                continue
            d = np.hypot(cent[i][0] - cx[f], cent[i][1] - cy[f]) / r
            if d <= ON_RADII:
                on_found = True          # a pellet-sized sphere ON the pillar
            elif d <= SA_MAX_RADII:
                off_found = True         # ... or localized off the pillar (in SA)
        cv_valid[f] = True
        # PILLAR-FIRST: if the sphere is on the pillar, it is ON regardless of any
        # other bright blob in the SA. A distractor / previous pellet in the SA
        # must never make CV assert 'off' while the pellet sits on the pillar --
        # that was what dropped real causal reaches.
        if on_found:
            cv_state[f] = CV_ON
        elif off_found:
            cv_state[f] = CV_OFF
        else:
            cv_state[f] = CV_GONE

    if frames_needed is None:
        f = 0
        while f < n:
            ok, im = cap.read()
            if not ok:
                break
            _localize(im, f)
            f += 1
    else:
        needed = sorted({int(x) for x in frames_needed if 0 <= int(x) < n})
        i = 0
        while i < len(needed):
            j = i
            while j + 1 < len(needed) and needed[j + 1] == needed[j] + 1:
                j += 1
            # one seek per contiguous run, then sequential reads through it
            cap.set(cv2.CAP_PROP_POS_FRAMES, needed[i])
            for f in range(needed[i], needed[j] + 1):
                ok, im = cap.read()
                if not ok:
                    break
                _localize(im, f)
            i = j + 1
    cap.release()
    return cv_state, cv_valid, thr
