"""
Post-processing: per-frame in-reach probabilities -> reach windows.

Given a per-frame probability series p(t), produce a list of
(start_frame, end_frame) reach windows by:
  1. Threshold p(t) > THRESHOLD into a binary in-reach mask.
  2. Connect runs of in-reach frames that are within MERGE_GAP frames
     of each other (small gaps from noisy probability are bridged).
  3. Drop runs shorter than MIN_SPAN.

v8.0.2 adds an optional fourth step:
  4. Trim leading frames of each reach where the paw is poorly tracked
     by DLC (sustained low-likelihood run). See `trim_leading_sustained_lk`.

v8.0.3 adds an optional fifth step:
  5. Split each reach at the trough between two prominent peaks in the
     hand-to-BoxL normalized distance trajectory. Catches the
     paw-visibility merger and apparatus-quirk merger failure modes
     where the GBM emits one algo span covering two real reaches.
     See `apex_split_at_trough`.

The threshold + merge gap + min span are calibration knobs. Defaults
are starting points to be tuned per the eval (over-call OK; iterate to
maximize TP recall while keeping start delta tight).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Hand-keypoint likelihood columns used to compute paw_mean_lk for the
# leading-trim postprocess. Must match the DLC bodypart names in
# features.BODYPARTS for hand keypoints.
HAND_LK_LIKELIHOOD_COLS = (
    "RightHand_likelihood",
    "RHLeft_likelihood",
    "RHOut_likelihood",
    "RHRight_likelihood",
)


@dataclass
class ReachSpan:
    start_frame: int
    end_frame: int  # inclusive

    @property
    def span(self) -> int:
        return self.end_frame - self.start_frame + 1


def probabilities_to_reaches(
    proba: np.ndarray,
    threshold: float = 0.5,
    merge_gap: int = 2,
    min_span: int = 3,
) -> List[ReachSpan]:
    """Convert a per-frame in-reach probability array to a list of reaches.

    Parameters
    ----------
    proba : np.ndarray
        Per-frame probability of being in a reach. Length = num_frames.
    threshold : float
        Frames with proba > threshold are flagged in-reach.
    merge_gap : int
        Adjacent in-reach runs separated by <= merge_gap not-in-reach
        frames are merged into one reach. Default 2.
    min_span : int
        Reaches shorter than this many frames (after merging) are
        dropped. Default 3.

    Returns
    -------
    list of ReachSpan, sorted by start_frame.
    """
    mask = (proba > threshold).astype(np.int8)
    n = len(mask)
    if n == 0:
        return []

    # Find runs of 1s
    runs = _find_runs(mask)

    # Merge runs separated by small gaps
    merged = _merge_close_runs(runs, merge_gap)

    # Drop short runs
    out = [r for r in merged if (r.end_frame - r.start_frame + 1) >= min_span]
    return out


def _find_runs(mask: np.ndarray) -> List[ReachSpan]:
    """Find contiguous runs of 1s; return list of (start, end) inclusive."""
    runs = []
    in_run = False
    s = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            s = i
        elif not v and in_run:
            in_run = False
            runs.append(ReachSpan(start_frame=s, end_frame=i - 1))
    if in_run:
        runs.append(ReachSpan(start_frame=s, end_frame=len(mask) - 1))
    return runs


def _merge_close_runs(runs: List[ReachSpan], gap: int) -> List[ReachSpan]:
    if not runs:
        return []
    merged = [runs[0]]
    for r in runs[1:]:
        prev = merged[-1]
        if r.start_frame - prev.end_frame - 1 <= gap:
            merged[-1] = ReachSpan(
                start_frame=prev.start_frame, end_frame=r.end_frame)
        else:
            merged.append(r)
    return merged


def compute_paw_mean_lk(dlc_df) -> np.ndarray:
    """Per-frame mean DLC likelihood across the 4 hand keypoints.

    Returns an array of length n_frames with the per-frame mean of
    RightHand, RHLeft, RHOut, RHRight likelihoods.
    """
    cols = list(HAND_LK_LIKELIHOOD_COLS)
    arr = dlc_df[cols].to_numpy(dtype=np.float32)
    return arr.mean(axis=1)


def trim_leading_sustained_lk(
    reaches: List[ReachSpan],
    paw_mean_lk: np.ndarray,
    threshold: float = 0.60,
    sustain_n: int = 3,
    min_span: int = 3,
) -> List[ReachSpan]:
    """Trim leading frames of each reach where the paw is poorly tracked.

    For each reach, walk inward from start; at each candidate frame F,
    only trim if frames [F, F+1, ..., F+sustain_n-1] are ALL below the
    likelihood threshold. This sustain check protects TPs against
    isolated low-likelihood frames (DLC jitter at reach onset) while
    still catching the early-start false-positive events whose
    leading frames have consistently low paw confidence.

    Reaches reduced below min_span after trimming are dropped from the
    output entirely.

    Calibrated 2026-05-21 on the model 3.1 DLC corpus:
      - LOOCV: TP +44 / FP -92 / FN -44 vs v8.0.1 baseline.
        TOLERANCE_ERROR(start_early) reduced 83 -> 7 (92% reduction).
      - Holdout (19 videos): TP +51 / FP -100 / FN -51.
        TOLERANCE_ERROR(start_early) reduced 71 -> 1 (99% reduction).
      - start_delta abs_median held at 0 on both corpora (Cardinal Rule).

    Snapshots:
      Improvement_Snapshots/reach_detection/v8.0.1_dev_leading_trim_sustained_sweep/
      Improvement_Snapshots/reach_detection/v8.0.1_dev_sustained_trim_holdout_gate/

    Parameters
    ----------
    reaches : list of ReachSpan
        Output of `probabilities_to_reaches`.
    paw_mean_lk : np.ndarray
        Per-frame mean DLC likelihood across the 4 hand keypoints
        (see `compute_paw_mean_lk`). Length must equal video frames.
    threshold : float
        Likelihood cutoff: a frame is "low-lk" if paw_mean_lk < threshold.
        Default 0.60 (calibrated).
    sustain_n : int
        Number of consecutive frames that must all be low-lk for the
        leading frame to be trimmed. Default 3 (calibrated).
    min_span : int
        Reaches with span (end - start + 1) below this after trimming
        are dropped entirely. Default 3 (matches probabilities_to_reaches).

    Returns
    -------
    list of ReachSpan (sorted by start_frame).
    """
    if not reaches:
        return []
    n_frames = len(paw_mean_lk)
    out = []
    for r in reaches:
        s, e = r.start_frame, r.end_frame
        new_s = s
        while new_s <= e:
            window_end = new_s + sustain_n
            if window_end > n_frames or window_end > e + 1:
                # not enough frames left to satisfy the sustain check
                break
            window = paw_mean_lk[new_s:window_end]
            if np.any(np.isnan(window)):
                break
            if np.any(window >= threshold):
                break  # at least one confident frame in window; stop trimming
            new_s += 1
        if e - new_s + 1 >= min_span:
            out.append(ReachSpan(start_frame=new_s, end_frame=e))
    return out


# ---- v8.0.3 apex-split postprocess ----

# Hand and apparatus keypoints used to compute the per-frame
# hand-centroid-to-BoxL normalized distance signal.
HAND_KEYPOINTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
BOXL_KEYPOINT = "BOXL"
BOXR_KEYPOINT = "BOXR"

NORM_POS_SMOOTH_WINDOW = 5
APPARATUS_WIDTH_FLOOR = 1e-3


def compute_hand_to_boxl_norm_pos(dlc_df) -> np.ndarray:
    """Per-frame distance from hand centroid to BoxL, normalized by
    apparatus width (BoxL-to-BoxR distance).

    Returns an array of length n_frames. Each value is in [0, ~1.x]
    where 0 = hand at BoxL, 1 = hand at BoxR. Smoothed with a 5-frame
    centered moving average on every input coordinate before the
    distance computation.

    Used by `apex_split_at_trough` to detect double-hump reach
    trajectories indicating two real reaches merged into one algo span.
    """
    def smooth(x):
        return pd.Series(x).rolling(NORM_POS_SMOOTH_WINDOW,
                                     center=True,
                                     min_periods=1).mean().to_numpy(dtype=np.float32)

    hand_x = smooth(np.mean(
        [dlc_df[f"{kp}_x"].to_numpy() for kp in HAND_KEYPOINTS], axis=0))
    hand_y = smooth(np.mean(
        [dlc_df[f"{kp}_y"].to_numpy() for kp in HAND_KEYPOINTS], axis=0))
    boxl_x = smooth(dlc_df[f"{BOXL_KEYPOINT}_x"].to_numpy())
    boxl_y = smooth(dlc_df[f"{BOXL_KEYPOINT}_y"].to_numpy())
    boxr_x = smooth(dlc_df[f"{BOXR_KEYPOINT}_x"].to_numpy())
    boxr_y = smooth(dlc_df[f"{BOXR_KEYPOINT}_y"].to_numpy())
    apparatus = np.sqrt((boxr_x - boxl_x) ** 2 + (boxr_y - boxl_y) ** 2)
    dist_boxl = np.sqrt((hand_x - boxl_x) ** 2 + (hand_y - boxl_y) ** 2)
    return dist_boxl / np.maximum(apparatus, APPARATUS_WIDTH_FLOOR)


def apex_split_at_trough(
    reaches: List[ReachSpan],
    norm_pos: np.ndarray,
    prominence: float = 0.12,
    depth_min: float = 0.5,
    peak2_rel_max: float = 0.85,
    min_distance: int = 4,
    min_span: int = 3,
) -> List[ReachSpan]:
    """Split each reach at the deep trough between two prominent peaks.

    For each emitted reach span, examine the hand-centroid-to-BoxL
    normalized distance signal over that span. If `scipy.signal.find_peaks`
    detects 2+ peaks meeting the prominence threshold AND the trough
    between two consecutive peaks is at least `depth_min` deep AND the
    last peak is at < `peak2_rel_max` of the span length, split the
    reach at the deepest trough frame.

    Rationale: each real reach has one extension apex (peak of
    hand-to-BoxL distance). When the GBM emits one algo span covering
    two real reaches (paw-visibility merger, apparatus-quirk merger),
    the normalized-position trajectory has a clean double-hump with a
    deep trough between. The peak2_rel guard rejects end-of-reach
    artifacts (grab, hold, jitter) where the second peak is near the
    end of the algo span and represents a single-reach pattern.

    Calibrated 2026-05-22 on the model 3.1 DLC corpus (16-video
    calibration LOOCV + 19-video holdout):
      - Calibration: TP +84 / FP -35 / FN -84 vs v8.0.2 baseline.
        MERGED topology reduced 57 -> 10 (82% reduction).
      - Holdout: TP +22 / FP -7 / FN -22.
        MERGED topology reduced 17 -> 3 (82% reduction).
      - start_delta abs_median held at 0 on both corpora.
      - span_delta abs_median held at 0 on both corpora.
      - Over-splits (apex FRAGMENTED whose GT was baseline TP):
        2 cal, 1 hol.
      - FRAGMENTED ratio vs baseline: 1.4x cal, 1.1x hol (under 2x).

    Sub-sweeps that selected these defaults:
      - prominence sweep (0.05..0.15 at depth=0.5, peak2=0.85): 0.12
        chosen for the over-split / MERGED-catch tradeoff sweet spot.
      - depth sweep (0.4..0.7 at prom=0.12, peak2=0.85): 0.5 matches
        0.4 at prom=0.12 (prominence is the binding constraint); above
        0.5 loses MERGED catches.
      - peak2_rel sweep (0.70..0.90 at prom=0.12, depth=0.5): 0.85 is
        the cal saturation point and is Pareto-optimal vs 0.90 on
        holdout (fewer over-splits at same FN gain).

    Snapshot: Improvement_Snapshots/reach_detection/
    v8.0.2_dev_apex_split_holdout_gate/RESULTS.md

    Parameters
    ----------
    reaches : list of ReachSpan
        Output of `probabilities_to_reaches` (optionally already trimmed
        by `trim_leading_sustained_lk`).
    norm_pos : np.ndarray
        Per-frame normalized hand-to-BoxL distance, length = n_frames.
        See `compute_hand_to_boxl_norm_pos`.
    prominence : float
        scipy.signal.find_peaks prominence threshold. Default 0.12
        (calibrated). Higher = fewer false splits inside single reaches.
    depth_min : float
        Minimum trough depth (max(peak1, peak2) - trough) required for
        a split. Default 0.5 (calibrated). At the default prominence,
        prominence is the binding constraint so depth_min=0.5 acts as
        a sanity floor rather than the primary filter.
    peak2_rel_max : float
        Suppress the split if the last detected peak's position is at
        >= this fraction of the algo span. Default 0.85 (calibrated).
        Filters end-of-reach grab/hold/jitter patterns.
    min_distance : int
        scipy.signal.find_peaks min_distance between peaks. Default 4.
    min_span : int
        Minimum span (in frames) of each half after the split.
        Default 3. If either half would be shorter, the split is not made.

    Returns
    -------
    list of ReachSpan (sorted by start_frame).
    """
    if not reaches:
        return []
    n_frames = len(norm_pos)
    out = []
    for r in reaches:
        s, e = r.start_frame, r.end_frame
        if e >= n_frames:
            out.append(r); continue
        sig = norm_pos[s:e + 1]
        if len(sig) < 3 or np.any(np.isnan(sig)):
            out.append(r); continue
        peaks, _ = find_peaks(sig, prominence=prominence, distance=min_distance)
        if len(peaks) < 2:
            out.append(r); continue
        peak2_rel = peaks[-1] / (len(sig) - 1)
        if peak2_rel >= peak2_rel_max:
            out.append(r); continue
        best_depth = 0.0
        best_trough_frame: Optional[int] = None
        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i + 1]
            if p2 - p1 < 2:
                continue
            between = sig[p1:p2 + 1]
            t_local = int(np.argmin(between))
            depth = max(float(sig[p1]), float(sig[p2])) - float(between[t_local])
            if depth > best_depth:
                best_depth = depth
                best_trough_frame = s + p1 + t_local
        if best_depth < depth_min or best_trough_frame is None:
            out.append(r); continue
        half1_span = best_trough_frame - s + 1
        half2_span = e - (best_trough_frame + 1) + 1
        if half1_span < min_span or half2_span < min_span:
            out.append(r); continue
        out.append(ReachSpan(start_frame=s, end_frame=best_trough_frame))
        out.append(ReachSpan(start_frame=best_trough_frame + 1, end_frame=e))
    return out
