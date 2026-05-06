"""
Per-segment label extraction for the v5 outcome detector.

Loads (segment_num, outcome_label, interaction_frame, outcome_known_frame)
tuples from a video's unified GT JSON. Collapses displaced_outside ->
displaced_sa per the existing metrics standard.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


# 4 classes; displaced_outside is collapsed.
OUTCOME_CLASSES = ["retrieved", "displaced_sa", "untouched", "abnormal_exception"]


def _collapse(o: Optional[str]) -> Optional[str]:
    if o == "displaced_outside":
        return "displaced_sa"
    return o


def load_segment_labels(gt_path: Path) -> List[Dict]:
    """Load per-segment label rows from a unified GT JSON.

    Returns list of dicts with keys: segment_num, outcome_label,
    interaction_frame (or None), outcome_known_frame (or None),
    exhaustive (the video-level flag, copied per-row for convenience).
    """
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    outcomes_block = gt.get("outcomes", {})
    exhaustive = bool(outcomes_block.get("exhaustive", False))
    segs = outcomes_block.get("segments", []) or []

    rows = []
    for seg in segs:
        rows.append({
            "segment_num": int(seg["segment_num"]),
            "outcome_label": _collapse(seg.get("outcome")),
            "interaction_frame": seg.get("interaction_frame"),
            "outcome_known_frame": seg.get("outcome_known_frame"),
            "exhaustive": exhaustive,
        })
    return rows
