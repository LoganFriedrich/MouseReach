"""
Reach detection evaluation, per the user-mandated reporting standard.

A detected reach is TP iff:
    abs(algo.start_frame - gt.start_frame) <= START_TOL  (default 2)
    AND
    abs(algo.span - gt.span) <= max(SPAN_TOL_REL * gt.span, SPAN_TOL_ABS)
    (default 0.5 * gt.span and 5 frames)

Greedy nearest-Δstart matching: each GT and each algo reach matches at
most once. Unmatched GT -> FN. Unmatched algo -> FP.

For matched (TP) pairs, we additionally collect:
    start_delta = algo.start - gt.start  (signed; sign tells bias)
    span_delta  = algo.span - gt.span    (signed)

Reports go into snapshot/figures/ as the canonical reach-eval artifact.
NOT precision/recall/F1 leading; numerical TP/FP/FN counts + delta
distributions only.

See `feedback_no_f1.md` and `reach_outcome_evaluation_format.md` in
cross-session memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Sequence

import numpy as np


@dataclass
class GTReach:
    start_frame: int
    end_frame: int
    video_id: str = ""
    index: int = -1

    @property
    def span(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class AlgoReach:
    start_frame: int
    end_frame: int
    video_id: str = ""
    index: int = -1

    @property
    def span(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class MatchResult:
    """One row of the per-reach evaluation."""
    status: str           # "tp", "fp", "fn"
    video_id: str
    gt_index: int = -1     # -1 if not matched (FP)
    algo_index: int = -1   # -1 if not matched (FN)
    start_delta: int = 0   # algo - gt; only meaningful for tp
    span_delta: int = 0    # algo.span - gt.span; only meaningful for tp


def evaluate_reaches(
    algo_reaches: Sequence[AlgoReach],
    gt_reaches: Sequence[GTReach],
    video_id: str = "",
    start_tol: int = 2,
    span_tol_rel: float = 0.5,
    span_tol_abs: int = 5,
) -> List[MatchResult]:
    """Match algo to GT and produce the per-reach result list.

    Both inputs are assumed to be from the SAME video. Cross-video
    eval should iterate per-video and concatenate the result lists.
    """
    # Candidate pairs satisfying both start and span tolerances
    candidates = []
    for gi, g in enumerate(gt_reaches):
        for ai, a in enumerate(algo_reaches):
            d_start = abs(a.start_frame - g.start_frame)
            if d_start > start_tol:
                continue
            tol = max(span_tol_rel * g.span, span_tol_abs)
            d_span = abs(a.span - g.span)
            if d_span > tol:
                continue
            candidates.append((d_start, d_span, gi, ai))

    # Sort by start delta first, then span delta -- closest first
    candidates.sort(key=lambda x: (x[0], x[1]))

    matched_gt: set = set()
    matched_algo: set = set()
    results: List[MatchResult] = []

    for _ds, _dsp, gi, ai in candidates:
        if gi in matched_gt or ai in matched_algo:
            continue
        matched_gt.add(gi)
        matched_algo.add(ai)
        g = gt_reaches[gi]
        a = algo_reaches[ai]
        results.append(MatchResult(
            status="tp",
            video_id=video_id,
            gt_index=gi,
            algo_index=ai,
            start_delta=a.start_frame - g.start_frame,
            span_delta=a.span - g.span,
        ))

    for gi, g in enumerate(gt_reaches):
        if gi not in matched_gt:
            results.append(MatchResult(
                status="fn", video_id=video_id, gt_index=gi))

    for ai, a in enumerate(algo_reaches):
        if ai not in matched_algo:
            results.append(MatchResult(
                status="fp", video_id=video_id, algo_index=ai))

    return results


def summarize_results(results: Sequence[MatchResult]) -> dict:
    """Numerical summary of a result list. Returns counts + delta
    distribution stats. Not precision/recall/F1.
    """
    tps = [r for r in results if r.status == "tp"]
    fps = [r for r in results if r.status == "fp"]
    fns = [r for r in results if r.status == "fn"]
    start_deltas = np.array([r.start_delta for r in tps], dtype=np.int32)
    span_deltas = np.array([r.span_delta for r in tps], dtype=np.int32)

    def pct(arr, q):
        return int(np.percentile(arr, q)) if len(arr) > 0 else None

    return {
        "n_tp": len(tps),
        "n_fp": len(fps),
        "n_fn": len(fns),
        "tp_start_delta": {
            "n": len(tps),
            "mean": float(start_deltas.mean()) if len(tps) else None,
            "median": int(np.median(start_deltas)) if len(tps) else None,
            "abs_median": int(np.median(np.abs(start_deltas))) if len(tps) else None,
            "p10": pct(start_deltas, 10),
            "p90": pct(start_deltas, 90),
            "min": int(start_deltas.min()) if len(tps) else None,
            "max": int(start_deltas.max()) if len(tps) else None,
        },
        "tp_span_delta": {
            "n": len(tps),
            "mean": float(span_deltas.mean()) if len(tps) else None,
            "median": int(np.median(span_deltas)) if len(tps) else None,
            "abs_median": int(np.median(np.abs(span_deltas))) if len(tps) else None,
            "p10": pct(span_deltas, 10),
            "p90": pct(span_deltas, 90),
            "min": int(span_deltas.min()) if len(tps) else None,
            "max": int(span_deltas.max()) if len(tps) else None,
        },
    }
