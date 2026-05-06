"""
Outcome v5 trainer.

Multinomial classification of segment outcomes (4 classes: retrieved /
displaced_sa / untouched / abnormal_exception). Single-fold trainer +
LOOCV wrapper. Headline reporting via Sankey + directional confusion.

abnormal_exception is included as a real predict-able class -- the
model is asked to predict it, and we measure how often it does. Per
the user's clarification: we still TRY to call those cases correctly,
just acknowledge they're hard.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import feature_columns
from .labels import OUTCOME_CLASSES

logger = logging.getLogger(__name__)


@dataclass
class FoldRow:
    """One per-segment prediction record."""
    video_id: str
    segment_num: int
    gt_label: str
    algo_label: str
    gt_exhaustive: bool


@dataclass
class FoldResult:
    val_video_ids: List[str]
    rows: List[FoldRow]


def train_one_fold(
    train_df: pd.DataFrame,
    train_video_ids: Sequence[str],
    val_video_ids: Sequence[str],
    max_iter: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    random_state: int = 42,
    only_known_outcomes: bool = True,
) -> Tuple[FoldResult, HistGradientBoostingClassifier]:
    """Train one fold and return val predictions per row.

    `only_known_outcomes`: drop training rows where outcome_label is
    None (rare; reflects unlabeled segments). Default True.
    """
    feat_cols = feature_columns()

    train_mask = train_df["video_id"].isin(train_video_ids)
    if only_known_outcomes:
        train_mask &= train_df["outcome_label"].notna()
    train = train_df.loc[train_mask]

    val_mask = train_df["video_id"].isin(val_video_ids)
    val = train_df.loc[val_mask]

    label_to_int = {c: i for i, c in enumerate(OUTCOME_CLASSES)}
    int_to_label = {i: c for c, i in label_to_int.items()}

    X_train = train[feat_cols].to_numpy(dtype=np.float32)
    y_train = np.array(
        [label_to_int.get(l, -1) for l in train["outcome_label"]],
        dtype=np.int8)

    # Drop any rows whose outcome wasn't in our class list (shouldn't happen
    # after the displaced_outside collapse but defensive)
    keep = y_train >= 0
    X_train = X_train[keep]
    y_train = y_train[keep]

    # Class-balanced sample weights so untouched and displaced_sa don't
    # drown out retrieved + abnormal_exception.
    n = len(y_train)
    weights = np.zeros(n, dtype=np.float32)
    for cls_int in range(len(OUTCOME_CLASSES)):
        mask = y_train == cls_int
        n_cls = int(mask.sum())
        if n_cls > 0:
            weights[mask] = n / (len(OUTCOME_CLASSES) * n_cls)

    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
        random_state=random_state, early_stopping=False,
    )
    clf.fit(X_train, y_train, sample_weight=weights)

    rows: List[FoldRow] = []
    for vid in val_video_ids:
        sub = val.loc[val["video_id"] == vid]
        if len(sub) == 0:
            continue
        Xv = sub[feat_cols].to_numpy(dtype=np.float32)
        preds_int = clf.predict(Xv)
        for (_, r), pi in zip(sub.iterrows(), preds_int):
            rows.append(FoldRow(
                video_id=vid,
                segment_num=int(r["segment_num"]),
                gt_label=str(r["outcome_label"]) if r["outcome_label"] is not None else None,
                algo_label=int_to_label[int(pi)],
                gt_exhaustive=bool(r["exhaustive"]),
            ))

    return FoldResult(val_video_ids=list(val_video_ids), rows=rows), clf


def loocv(
    train_df: pd.DataFrame,
    train_video_ids: Sequence[str],
    **kwargs,
) -> List[FoldResult]:
    """Leave-one-video-out CV across all videos in train_video_ids.

    Outcome training uses ALL videos (not just exhaustive) per the
    Phase A finding that outcome labels are per-segment-complete in
    both kinds.
    """
    folds = []
    for i, val_vid in enumerate(train_video_ids):
        train_ids = [v for v in train_video_ids if v != val_vid]
        print(f"  fold {i+1}/{len(train_video_ids)}: val={val_vid}", flush=True)
        fold, _ = train_one_fold(
            train_df, train_ids, [val_vid], **kwargs)
        folds.append(fold)
    return folds


def confusion_matrix(rows: Sequence[FoldRow]) -> Dict[str, int]:
    """Build a `gt__algo` confusion matrix dict.

    Compatible with the existing Sankey runner
    (`outcome/_run_notebooks.py:run_sankey`), which reads
    `scalars["outcome_label"]["confusion_matrix"]`.
    """
    cm: Dict[str, int] = defaultdict(int)
    for r in rows:
        if r.gt_label is None:
            continue
        cm[f"{r.gt_label}__{r.algo_label}"] += 1
    return dict(cm)


def directional_summary(rows: Sequence[FoldRow]) -> Dict[str, Dict]:
    """Per-class shift summary -- for each GT class, where did its
    rows go on the algo side? And vice versa."""
    by_gt: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_algo: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r.gt_label is None:
            continue
        by_gt[r.gt_label][r.algo_label] += 1
        by_algo[r.algo_label][r.gt_label] += 1
    return {
        "by_gt": {k: dict(v) for k, v in by_gt.items()},
        "by_algo": {k: dict(v) for k, v in by_algo.items()},
    }
