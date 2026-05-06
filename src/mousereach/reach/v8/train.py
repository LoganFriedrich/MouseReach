"""
Reach detection v8 trainer.

Single-fold trainer + cross-validation wrapper. Training data is
filtered to exhaustive=True videos by default (gold-standard
negatives). Predictions are converted to reaches via post-processing
and evaluated per the user-mandated reach-detection metric standard
(TP iff start within +/- 2f of GT AND span match; report start delta
+ span delta distributions for TPs).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .eval import (AlgoReach, GTReach, MatchResult, evaluate_reaches,
                   summarize_results)
from .features import feature_columns
from .postprocess import probabilities_to_reaches

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    val_video_ids: List[str]
    threshold: float
    summary: dict
    raw_results: List[MatchResult]


def train_one_fold(
    train_pool_df: pd.DataFrame,
    train_video_ids: Sequence[str],
    val_video_ids: Sequence[str],
    threshold: float = 0.5,
    merge_gap: int = 2,
    min_span: int = 3,
    only_exhaustive_for_train: bool = True,
    max_iter: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    random_state: int = 42,
    class_weight_imbalance: bool = True,
) -> Tuple[FoldResult, HistGradientBoostingClassifier]:
    """Train a single model on `train_video_ids`, evaluate on
    `val_video_ids`, return per-reach FoldResult.

    `train_pool_df` is the full train_pool.parquet contents. The
    function slices it; caller does not need to pre-filter.
    """
    feat_cols = feature_columns()

    # Build train mask
    train_mask = train_pool_df["video_id"].isin(train_video_ids)
    if only_exhaustive_for_train:
        train_mask &= train_pool_df["exhaustive"]
    train = train_pool_df.loc[train_mask]

    val_mask = train_pool_df["video_id"].isin(val_video_ids)
    val = train_pool_df.loc[val_mask]

    X_train = train[feat_cols].to_numpy(dtype=np.float32)
    y_train = train["label"].to_numpy(dtype=np.int8)

    sample_weight = None
    if class_weight_imbalance:
        n = len(y_train)
        n_pos = int(y_train.sum())
        n_neg = n - n_pos
        if n_pos > 0 and n_neg > 0:
            w_pos = n / (2.0 * n_pos)
            w_neg = n / (2.0 * n_neg)
            sample_weight = np.where(y_train == 1, w_pos, w_neg).astype(np.float32)

    clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        early_stopping=False,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    # Validate per-video so reach assembly stays per-video
    all_results: List[MatchResult] = []
    for vid in val_video_ids:
        vmask = val["video_id"] == vid
        Xv = val.loc[vmask, feat_cols].to_numpy(dtype=np.float32)
        proba = clf.predict_proba(Xv)[:, 1]

        algo_reaches_raw = probabilities_to_reaches(
            proba, threshold=threshold, merge_gap=merge_gap, min_span=min_span)
        algo_reaches = [
            AlgoReach(start_frame=r.start_frame, end_frame=r.end_frame,
                      video_id=vid, index=i)
            for i, r in enumerate(algo_reaches_raw)
        ]

        # Build GT reaches from the per-frame label + reach_id metadata
        sub = val.loc[vmask].sort_values("frame")
        rid = sub["reach_id"].to_numpy()
        frames = sub["frame"].to_numpy()
        gt_reaches: List[GTReach] = []
        unique_rids = sorted(set(rid[rid >= 0].tolist()))
        for ri in unique_rids:
            rmask = rid == ri
            f = frames[rmask]
            gt_reaches.append(GTReach(
                start_frame=int(f.min()),
                end_frame=int(f.max()),
                video_id=vid,
                index=ri,
            ))

        results = evaluate_reaches(algo_reaches, gt_reaches, video_id=vid)
        all_results.extend(results)

    summary = summarize_results(all_results)
    fold = FoldResult(val_video_ids=list(val_video_ids),
                      threshold=threshold,
                      summary=summary,
                      raw_results=all_results)
    return fold, clf


def loocv_evaluate(
    train_pool_df: pd.DataFrame,
    train_video_ids: Sequence[str],
    threshold: float = 0.5,
    merge_gap: int = 2,
    min_span: int = 3,
    only_exhaustive_for_train: bool = True,
    only_evaluate_exhaustive: bool = True,
    **trainer_kwargs,
) -> List[FoldResult]:
    """Run leave-one-video-out CV. Returns one FoldResult per fold."""
    folds = []
    eligible_val = list(train_video_ids)
    if only_evaluate_exhaustive:
        # Restrict val videos to exhaustive ones (gold-standard reach eval).
        exh_set = set(
            train_pool_df.loc[train_pool_df["exhaustive"], "video_id"]
            .unique().tolist())
        eligible_val = [v for v in train_video_ids if v in exh_set]

    for i, val_vid in enumerate(eligible_val):
        train_ids = [v for v in train_video_ids if v != val_vid]
        logger.info(f"LOOCV fold {i+1}/{len(eligible_val)}: val={val_vid}")
        print(f"  fold {i+1}/{len(eligible_val)}: val={val_vid}", flush=True)
        fold, _clf = train_one_fold(
            train_pool_df, train_ids, [val_vid],
            threshold=threshold,
            merge_gap=merge_gap, min_span=min_span,
            only_exhaustive_for_train=only_exhaustive_for_train,
            **trainer_kwargs,
        )
        s = fold.summary
        print(f"    TP={s['n_tp']:>4} FP={s['n_fp']:>4} FN={s['n_fn']:>4}  "
              f"start_delta median={s['tp_start_delta']['median']} "
              f"abs_median={s['tp_start_delta']['abs_median']}  "
              f"span_delta median={s['tp_span_delta']['median']} "
              f"abs_median={s['tp_span_delta']['abs_median']}",
              flush=True)
        folds.append(fold)
    return folds


def aggregate_folds(folds: Sequence[FoldResult]) -> dict:
    """Sum TP/FP/FN across folds and recompute delta distributions."""
    all_results: List[MatchResult] = []
    for f in folds:
        all_results.extend(f.raw_results)
    return summarize_results(all_results)
