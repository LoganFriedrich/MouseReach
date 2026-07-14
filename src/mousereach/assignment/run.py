"""Run algo-4 (reach assignment) for a single video as a pipeline step.

Algo-4 is the two-signal agreement-gate assignment (``assign_reaches_v2``): it
joins the per-video reaches with the per-segment cascade outcomes and commits a
causal reach only when the IFR-join pick and the pellet-displacement pick agree
(otherwise the segment is left for triage). This module is the importable
entrypoint the watcher's pipeline paths call so causal-reach attribution is
produced automatically -- previously it ran only in the manual review staging
path, so auto-processed videos never got a ``_reach_assignments.json``.

Reads the already-computed ``{stem}_segments.json`` / ``{stem}_reaches.json`` /
``{stem}_pellet_outcomes.json`` from ``processing_dir`` plus the DLC pose h5, and
writes ``{stem}_reach_assignments.json`` beside them.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load(directory: Path, name: str) -> Optional[Dict[str, Any]]:
    p = Path(directory) / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def assign_reaches_for_video(
    processing_dir: Path,
    video_id: str,
    dlc_path: Path,
    *,
    write: bool = True,
) -> Optional[Dict[str, Any]]:
    """Run algo-4 for one video. Returns the assignment result dict, or None if a
    required input is missing (no segments / reaches / outcomes).

    ``dlc_path`` is the DLC pose h5 (used by the displacement-pick signal).
    """
    from .cli import _segments_with_outcomes, _reaches_list
    from .v2.assign import assign_reaches_v2
    from ..reach.v8.features import load_dlc_h5

    processing_dir = Path(processing_dir)
    seg_doc = _load(processing_dir, f"{video_id}_segments.json")
    reach_doc = _load(processing_dir, f"{video_id}_reaches.json")
    outcome_doc = _load(processing_dir, f"{video_id}_pellet_outcomes.json")
    if seg_doc is None or reach_doc is None or outcome_doc is None:
        logger.warning(
            f"Assignment (algo-4) skipped for {video_id}: missing "
            f"{'segments' if seg_doc is None else ''} "
            f"{'reaches' if reach_doc is None else ''} "
            f"{'outcomes' if outcome_doc is None else ''}".strip()
        )
        return None

    dlc_df = load_dlc_h5(Path(dlc_path))
    merged_segs = _segments_with_outcomes(seg_doc, outcome_doc)
    reaches_list = _reaches_list(reach_doc)
    assign_result = assign_reaches_v2(
        reaches=reaches_list,
        segments_with_outcomes=merged_segs,
        dlc_df=dlc_df,
        video_id=video_id,
    )

    if write:
        out = processing_dir / f"{video_id}_reach_assignments.json"
        out.write_text(json.dumps(assign_result, indent=2) + "\n", encoding="utf-8")
        n_causal = sum(1 for r in assign_result.get("reaches", []) if r.get("is_causal"))
        logger.info(
            f"Assignment (algo-4) complete: {video_id} "
            f"({n_causal} causal / {len(assign_result.get('reaches', []))} reaches)"
        )
    return assign_result
