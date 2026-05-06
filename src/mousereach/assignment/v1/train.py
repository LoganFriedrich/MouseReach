"""
Reach assignment v1 trainer.

Binary classifier: causal (1) vs miss (0) per reach within a touched
segment. Trained per LOOCV fold on GT reaches; applied at inference
to v8-detected reaches.

At inference, for each touched segment we score every reach and pick
the highest-scoring as the segment's causal reach. Other reaches are
tagged as misses.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import feature_columns

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    val_video_ids: List[str]
    rows: List[Dict]   # one per per-reach prediction


def train_one_fold(
    train_df: pd.DataFrame,
    train_video_ids: Sequence[str],
    val_video_ids: Sequence[str],
    max_iter: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    random_state: int = 42,
) -> "tuple[FoldResult, HistGradientBoostingClassifier]":
    """Train one assignment fold; return val per-reach predictions."""
    feat_cols = feature_columns()

    train = train_df.loc[train_df["video_id"].isin(train_video_ids)]
    val = train_df.loc[train_df["video_id"].isin(val_video_ids)]

    X = train[feat_cols].to_numpy(dtype=np.float32)
    y = train["causal"].to_numpy(dtype=np.int8)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    sw = np.where(
        y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg)).astype(np.float32) \
        if (n_pos and n_neg) else None

    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
        random_state=random_state, early_stopping=False)
    clf.fit(X, y, sample_weight=sw)

    rows: List[Dict] = []
    if len(val):
        Xv = val[feat_cols].to_numpy(dtype=np.float32)
        proba = clf.predict_proba(Xv)[:, 1]
        # Pick max-prob reach within each (video, segment) as causal
        val_meta = val[["video_id", "segment_num", "reach_id",
                        "reach_start_frame", "reach_end_frame",
                        "segment_outcome", "interaction_frame", "causal"]].copy()
        val_meta["proba_causal"] = proba

        # Group by (video, segment), pick argmax
        for (vid, sn), grp in val_meta.groupby(["video_id", "segment_num"]):
            best_idx = grp["proba_causal"].idxmax()
            for idx, r in grp.iterrows():
                pred_causal = 1 if idx == best_idx else 0
                rows.append({
                    "video_id": r["video_id"],
                    "segment_num": int(r["segment_num"]),
                    "reach_id": int(r["reach_id"]),
                    "reach_start_frame": int(r["reach_start_frame"]),
                    "reach_end_frame": int(r["reach_end_frame"]),
                    "segment_outcome": r["segment_outcome"],
                    "interaction_frame": int(r["interaction_frame"]),
                    "gt_causal": int(r["causal"]),
                    "pred_causal": pred_causal,
                    "proba_causal": float(r["proba_causal"]),
                })

    return FoldResult(val_video_ids=list(val_video_ids), rows=rows), clf


def loocv(
    train_df: pd.DataFrame,
    train_video_ids: Sequence[str],
    **kwargs,
) -> List[FoldResult]:
    folds = []
    for i, val_vid in enumerate(train_video_ids):
        train_ids = [v for v in train_video_ids if v != val_vid]
        print(f"  fold {i+1}/{len(train_video_ids)}: val={val_vid}", flush=True)
        fold, _ = train_one_fold(train_df, train_ids, [val_vid], **kwargs)
        folds.append(fold)
    return folds


def causal_recall_summary(rows: Sequence[Dict]) -> Dict:
    """Per-segment-outcome breakdown of causal-attribution accuracy."""
    by_outcome: Dict[str, Dict[str, int]] = {}
    for r in rows:
        out = r["segment_outcome"]
        b = by_outcome.setdefault(out, {"n_segments": 0, "causal_correct": 0,
                                        "n_reaches": 0})
        b["n_reaches"] += 1
        # The "causal_correct" check is per-segment, not per-reach. We
        # count it once per segment by detecting "the predicted causal
        # reach was the GT causal reach". Iterate by (video, segment)
        # later for accurate count.
    # Recompute per-segment correctly:
    by_outcome = {}
    seen_segments = set()
    for r in rows:
        key = (r["video_id"], r["segment_num"])
        if key in seen_segments:
            continue
        seen_segments.add(key)
        out = r["segment_outcome"]
        b = by_outcome.setdefault(out, {"n_segments": 0, "causal_correct": 0})
        b["n_segments"] += 1
    # Find segments where pred_causal==1 row also has gt_causal==1
    for r in rows:
        if r["pred_causal"] == 1 and r["gt_causal"] == 1:
            out = r["segment_outcome"]
            by_outcome[out]["causal_correct"] += 1
    return by_outcome
