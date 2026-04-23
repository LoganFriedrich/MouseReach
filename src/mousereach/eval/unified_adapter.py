"""
Adapter to load unified ground truth files as split-format equivalents.

Unified GT files (*_unified_ground_truth.json) contain all three GT sections
(segmentation, reaches, outcomes) in a single file. This module converts each
section into the dict shape expected by the split-format evaluators, respecting
completion_status flags to skip incomplete sections.
"""

import json
from pathlib import Path
from typing import Dict, Optional


def load_unified_as_seg_gt(path: Path) -> Optional[Dict]:
    """Load a unified GT file and return data shaped like split _seg_ground_truth.json.

    Split seg format:
        {"boundaries": [int, int, ...], "n_boundaries": int, "video_name": str, ...}

    Unified format:
        segmentation.boundaries = list of dicts with 'frame' key

    Returns:
        Dict shaped like split seg GT, or None if segmentation is incomplete.
    """
    data = _load_unified(path)
    if data is None:
        return None

    # Check completion status
    completion = data.get("completion_status", {})
    if not completion.get("segments_complete", False):
        return None

    seg_section = data.get("segmentation", {})
    if not seg_section:
        return None

    # Convert list-of-dicts boundaries to flat list of ints
    raw_boundaries = seg_section.get("boundaries", [])
    flat_boundaries = []
    for b in raw_boundaries:
        if isinstance(b, dict):
            flat_boundaries.append(b["frame"])
        elif isinstance(b, (int, float)):
            flat_boundaries.append(int(b))

    result = {
        "video_name": data.get("video_name", ""),
        "type": "seg_ground_truth",
        "boundaries": sorted(flat_boundaries),
        "n_boundaries": len(flat_boundaries),
    }

    # Carry over optional metadata
    for key in ("created_by", "created_at", "total_frames", "fps"):
        if key in data:
            result[key] = data[key]

    return result


def load_unified_as_reach_gt(path: Path) -> Optional[Dict]:
    """Load a unified GT file and return data shaped like split _reach_ground_truth.json.

    Split reach format:
        {"segments": [{"segment_num": int, "reaches": [...], ...}], "gt_complete": bool, ...}

    Unified format:
        reaches.reaches = flat list of reach dicts with segment_num field

    Returns:
        Dict shaped like split reach GT, or None if reaches section is incomplete.
    """
    data = _load_unified(path)
    if data is None:
        return None

    completion = data.get("completion_status", {})
    if not completion.get("reaches_complete", False):
        return None

    reach_section = data.get("reaches", {})
    if not reach_section:
        return None

    flat_reaches = reach_section.get("reaches", [])

    # Group reaches by segment_num to match split format
    segments_map: Dict[int, list] = {}
    for r in flat_reaches:
        seg_num = r.get("segment_num", 0)
        if seg_num not in segments_map:
            segments_map[seg_num] = []
        start = r.get("start_frame", 0)
        end = r.get("end_frame", 0)
        apex = r.get("apex_frame")
        # Compute fallback apex if missing/None
        if apex is None and start is not None and end is not None:
            apex = (start + end) // 2

        segments_map[seg_num].append({
            "reach_id": r.get("reach_id"),
            "start_frame": start,
            "apex_frame": apex,
            "end_frame": end,
            # Carry over fields the evaluator may use
            "max_extent_ruler": r.get("max_extent_ruler", 0),
            "exclude_from_analysis": r.get("exclude_from_analysis", False),
            "exclude_reason": r.get("exclude_reason", ""),
            "human_corrected": r.get("human_corrected", False),
            "human_verified": r.get("human_verified", False),
            "source": r.get("source", "unified_gt"),
        })

    # Build segment list sorted by segment_num
    segments = []
    for seg_num in sorted(segments_map.keys()):
        reaches = segments_map[seg_num]
        segments.append({
            "segment_num": seg_num,
            "reaches": reaches,
            "n_reaches": len(reaches),
            "human_verified": True,  # unified GT is human-authored
        })

    result = {
        "video_name": data.get("video_name", ""),
        "type": "reach_ground_truth",
        "gt_complete": True,
        "n_segments": len(segments),
        "segments": segments,
        "total_reaches": len(flat_reaches),
        "verified_reaches": len(flat_reaches),
    }

    for key in ("created_by", "created_at"):
        if key in data:
            result[key] = data[key]

    return result


def load_unified_as_outcome_gt(path: Path) -> Optional[Dict]:
    """Load a unified GT file and return data shaped like split _outcome_ground_truth.json.

    Split outcome format:
        {"segments": [{"segment_num": int, "outcome": str, ...}], ...}

    Unified format:
        outcomes.segments = list of dicts with same fields

    Returns:
        Dict shaped like split outcome GT, or None if outcomes section is incomplete.
    """
    data = _load_unified(path)
    if data is None:
        return None

    completion = data.get("completion_status", {})
    if not completion.get("outcomes_complete", False):
        return None

    outcome_section = data.get("outcomes", {})
    if not outcome_section:
        return None

    raw_segments = outcome_section.get("segments", [])

    # Map to split format -- fields are already compatible
    segments = []
    for seg in raw_segments:
        segments.append({
            "segment_num": seg.get("segment_num"),
            "outcome": seg.get("outcome", "uncertain"),
            "interaction_frame": seg.get("interaction_frame"),
            "outcome_known_frame": seg.get("outcome_known_frame"),
            "causal_reach_id": seg.get("causal_reach_id"),
            "confidence": seg.get("confidence", 1.0),
            "human_verified": True,
            "pellet_visible_start": seg.get("pellet_visible_start"),
            "pellet_visible_end": seg.get("pellet_visible_end"),
        })

    result = {
        "video_name": data.get("video_name", ""),
        "type": "outcome_ground_truth",
        "n_segments": len(segments),
        "segments": segments,
    }

    for key in ("created_by", "created_at"):
        if key in data:
            result[key] = data[key]

    return result


def find_unified_gt_files(gt_dir: Path) -> Dict[str, Path]:
    """Find all unified GT files in a directory, keyed by video_id.

    Returns:
        Dict mapping video_id -> unified GT file path
    """
    result = {}
    if not gt_dir or not gt_dir.exists():
        return result
    for f in gt_dir.glob("*_unified_ground_truth.json"):
        video_id = f.stem.replace("_unified_ground_truth", "")
        result[video_id] = f
    return result


def _load_unified(path: Path) -> Optional[Dict]:
    """Load and validate a unified GT file."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        # Basic validation
        if data.get("type") != "unified_ground_truth":
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None
