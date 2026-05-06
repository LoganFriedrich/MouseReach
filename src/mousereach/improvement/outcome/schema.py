"""Schema for the outcome (algo 3) evaluator -- per-PELLET (per-segment)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class SegmentOutcomeRow:
    video_id: str
    segment_num: int
    gt_outcome: Optional[str] = None
    algo_outcome: Optional[str] = None
    gt_interaction_frame: Optional[int] = None
    algo_interaction_frame: Optional[int] = None
    interaction_delta: Optional[int] = None
    gt_exhaustive: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OutcomeScalars:
    n_videos: int = 0
    n_segments: int = 0
    n_correct: int = 0
    triage_count: int = 0    # algo flagged for review
    confusion_matrix: Dict[str, int] = field(default_factory=dict)
    per_class: Dict[str, Dict[str, int]] = field(default_factory=dict)
    rows: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
