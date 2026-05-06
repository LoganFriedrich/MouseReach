"""Schema for the segmentation evaluator's metrics output."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class BoundaryMatch:
    """One paired (algo, gt) boundary, or unmatched on either side."""
    video_id: str
    status: str           # "matched" | "fp" | "fn"
    algo_frame: Optional[int] = None
    gt_frame: Optional[int] = None
    delta: Optional[int] = None      # algo - gt; only when matched

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SegmentationScalars:
    n_videos: int = 0
    n_videos_with_gt_boundaries: int = 0
    n_matched: int = 0
    n_fp: int = 0
    n_fn: int = 0
    delta_median: Optional[int] = None
    delta_abs_median: Optional[int] = None
    delta_p10: Optional[int] = None
    delta_p90: Optional[int] = None
    delta_min: Optional[int] = None
    delta_max: Optional[int] = None
    triage_count: int = 0    # boundaries the algo flagged for review
    matches: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
