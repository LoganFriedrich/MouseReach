"""Schema for the reach detection evaluator's metrics output."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class ReachMatch:
    video_id: str
    status: str               # "tp" | "fp" | "fn"
    gt_start: Optional[int] = None
    gt_end: Optional[int] = None
    algo_start: Optional[int] = None
    algo_end: Optional[int] = None
    start_delta: Optional[int] = None    # algo - gt; only when tp
    span_delta: Optional[int] = None     # algo.span - gt.span; only when tp

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReachDetectionScalars:
    n_videos: int = 0
    n_tp: int = 0
    n_fp: int = 0
    n_fn: int = 0
    triage_count: int = 0
    start_delta_median: Optional[int] = None
    start_delta_abs_median: Optional[int] = None
    start_delta_p10: Optional[int] = None
    start_delta_p90: Optional[int] = None
    start_delta_min: Optional[int] = None
    start_delta_max: Optional[int] = None
    span_delta_median: Optional[int] = None
    span_delta_abs_median: Optional[int] = None
    span_delta_p10: Optional[int] = None
    span_delta_p90: Optional[int] = None
    span_delta_min: Optional[int] = None
    span_delta_max: Optional[int] = None
    matches: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
