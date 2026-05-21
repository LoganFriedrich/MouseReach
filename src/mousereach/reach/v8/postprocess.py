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

The threshold + merge gap + min span are calibration knobs. Defaults
are starting points to be tuned per the eval (over-call OK; iterate to
maximize TP recall while keeping start delta tight).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

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
