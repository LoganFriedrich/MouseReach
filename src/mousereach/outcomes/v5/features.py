"""
Per-segment feature extraction for the v5 outcome detector.

Approach
--------
For each (segment_start, segment_end) in a video, we extract features
over a wider window: [segment_start - PRE_PAD, segment_end + POST_PAD].
The wide post-pad captures settling and the next ASPA cycle's
revealing of whether the pellet is still on the pillar -- the outcome
is often determinable only well after the segment's nominal end.

Per-frame features are reused from `reach.v8.features.extract_features`
(405 features per frame). We aggregate them over the segment window
into length-invariant statistics: mean, std, min, max, p10, p90.

Plus segment-level metadata:
  - segment_length
  - post_pad_used (post-pad clipped at video end)
  - effective_window_length

Output: one row per segment, with deterministic column ordering.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from mousereach.reach.v8.features import (extract_features as extract_per_frame_features,
                                           feature_columns as per_frame_feature_columns)


PRE_PAD = 30
POST_PAD = 500

# Stats applied to each per-frame feature column over the segment window.
# Each must be a function array -> scalar.
# Stats applied to each per-frame feature column over the segment window.
# Trimmed to mean/min/max for v1 to keep the per-segment feature count
# manageable (405 per-frame features * 3 stats = 1215 per-segment).
# Add std / percentiles in a later iteration if the model needs them.
STAT_FUNCS: Dict[str, callable] = {
    "mean": lambda a: float(np.mean(a)) if len(a) else 0.0,
    "min":  lambda a: float(np.min(a)) if len(a) else 0.0,
    "max":  lambda a: float(np.max(a)) if len(a) else 0.0,
}


def extract_segment_features(
    dlc_df: pd.DataFrame,
    seg_start: int,
    seg_end: int,
    pre_pad: int = PRE_PAD,
    post_pad: int = POST_PAD,
    per_frame_feats: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Build a per-segment feature dict.

    Parameters
    ----------
    dlc_df : per-frame DLC dataframe (full video)
    seg_start, seg_end : inclusive segment boundaries (frame indices)
    per_frame_feats : optional pre-computed per-frame feature dataframe
        for the full video. If None, will compute from `dlc_df`. Pass
        a precomputed dataframe across many segments of the same video
        to avoid recomputation.
    """
    n_frames = len(dlc_df)

    if per_frame_feats is None:
        per_frame_feats = extract_per_frame_features(dlc_df)

    # Window
    win_start = max(0, seg_start - pre_pad)
    win_end = min(n_frames - 1, seg_end + post_pad)
    win = per_frame_feats.iloc[win_start:win_end + 1]

    out: Dict[str, float] = {}
    for col in per_frame_feature_columns():
        arr = win[col].to_numpy(dtype=np.float32)
        for sname, sfn in STAT_FUNCS.items():
            out[f"{col}__{sname}"] = sfn(arr)

    out["segment_length"] = float(seg_end - seg_start + 1)
    out["effective_window_length"] = float(win_end - win_start + 1)
    out["post_pad_used"] = float(min(post_pad, n_frames - 1 - seg_end))
    out["pre_pad_used"] = float(min(pre_pad, seg_start))

    return out


def feature_columns() -> List[str]:
    """Canonical column order for the per-segment feature matrix."""
    cols: List[str] = []
    for c in per_frame_feature_columns():
        for s in STAT_FUNCS:
            cols.append(f"{c}__{s}")
    cols.extend(["segment_length", "effective_window_length",
                 "post_pad_used", "pre_pad_used"])
    return cols
