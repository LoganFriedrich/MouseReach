"""
Per-frame in-reach label extraction from GT reach annotations.

Labels are binary: 1 if the frame falls within any GT reach span,
0 otherwise. Frame is considered in-reach iff
    reach.start_frame <= f <= reach.end_frame.

For exhaustive=True videos, label-0 frames are real true negatives.
For exhaustive=False videos, label-0 frames are NOT reliable negatives
(see `feedback_exhaustive_is_gold_standard.md`). Trainers should
respect this distinction and either weight or filter accordingly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_gt_reaches(gt_path: Path) -> Tuple[List[Dict], bool]:
    """Load GT reaches plus the video's exhaustive flag.

    Returns
    -------
    reaches : list of dict
        Each with at least `start_frame`, `end_frame`, `segment_num`.
    exhaustive : bool
        True iff the GT marks this video as having every reach labeled.
    """
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    reaches = gt.get("reaches", {}).get("reaches", []) or []
    exhaustive = bool(gt.get("outcomes", {}).get("exhaustive", False))
    return reaches, exhaustive


def per_frame_labels(n_frames: int, reaches: List[Dict]) -> np.ndarray:
    """Build a length-`n_frames` int8 array; 1 inside any reach span, 0
    elsewhere. Inclusive of both reach endpoints.
    """
    labels = np.zeros(n_frames, dtype=np.int8)
    for r in reaches:
        s = r.get("start_frame")
        e = r.get("end_frame")
        if s is None or e is None:
            continue
        s = max(0, int(s))
        e = min(n_frames - 1, int(e))
        if e >= s:
            labels[s:e + 1] = 1
    return labels


def reach_id_per_frame(n_frames: int, reaches: List[Dict]) -> np.ndarray:
    """For each frame, the index (within `reaches`) of the GT reach it
    belongs to, or -1 if not in any reach.

    Useful for grouping at training/eval time so a single reach is
    never split across train/val splits.
    """
    out = np.full(n_frames, -1, dtype=np.int32)
    for i, r in enumerate(reaches):
        s = r.get("start_frame")
        e = r.get("end_frame")
        if s is None or e is None:
            continue
        s = max(0, int(s))
        e = min(n_frames - 1, int(e))
        if e >= s:
            out[s:e + 1] = i
    return out
