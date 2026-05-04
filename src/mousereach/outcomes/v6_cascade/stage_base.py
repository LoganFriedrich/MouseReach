"""
Lightweight Stage interface for the cascade.

Each Stage:
- Takes a `SegmentInput` (DLC, segment range, reach windows from GT or
  v8, etc.)
- Computes a small feature set focused on its specific question
- Either COMMITS a class with a name + reason, or CONTINUES to the
  next stage with a deferral reason

The cascade runner threads SegmentInput through each Stage in order
until one COMMITS, or all stages defer (final stage produces TRIAGE).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class SegmentInput:
    """Inputs available to every cascade stage for one segment."""
    video_id: str
    segment_num: int
    seg_start: int             # inclusive
    seg_end: int               # inclusive
    dlc_df: pd.DataFrame       # full-video DLC (we slice as needed)
    reach_windows: List[Tuple[int, int]] = field(default_factory=list)
    # ^ list of (reach_start, reach_end) tuples for reaches in this
    # segment. From GT or v8 reach detector at evaluation time.


@dataclass
class StageDecision:
    """Output of one stage for one segment.

    A commit MUST emit when-frames per the trust framework
    (`per_algo_evaluation_toolset.md` + 2026-05-01 cascade design):
    - committed_class: outcome class
    - whens["outcome_known_frame"]: always required for any commit
    - whens["interaction_frame"]: required when committed_class is
      retrieved or displaced_sa; null/absent for untouched
    """
    decision: str               # "commit" | "continue" | "triage"
    committed_class: Optional[str] = None    # set if decision == "commit"
    whens: Dict[str, Optional[int]] = field(default_factory=dict)
    reason: str = ""            # human-readable why
    features: Dict[str, Any] = field(default_factory=dict)
    # ^ the small set of features the stage computed -- saved for
    # diagnostics and per-stage analysis


class Stage:
    """Base class. Subclass and implement `decide`."""
    name: str = "unnamed"
    target_class: Optional[str] = None  # which outcome class this stage commits to (if any)

    def decide(self, seg: SegmentInput) -> StageDecision:
        raise NotImplementedError
