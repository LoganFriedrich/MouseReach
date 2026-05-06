"""
Post-processing: per-frame in-reach probabilities -> reach windows.

Given a per-frame probability series p(t), produce a list of
(start_frame, end_frame) reach windows by:
  1. Threshold p(t) > THRESHOLD into a binary in-reach mask.
  2. Connect runs of in-reach frames that are within MERGE_GAP frames
     of each other (small gaps from noisy probability are bridged).
  3. Drop runs shorter than MIN_SPAN.

The threshold + merge gap + min span are calibration knobs. Defaults
are starting points to be tuned per the eval (over-call OK; iterate to
maximize TP recall while keeping start delta tight).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


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
