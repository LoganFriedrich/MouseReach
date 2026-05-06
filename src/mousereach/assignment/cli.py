"""
CLI for reach assignment (v1).

Production usage:
    mousereach-assign-reaches -i Processing/

For each video directory under the input root that has both a
``*_reaches.json`` (from the v8 reach detector) and a
``*_pellet_outcomes.json`` (from the v6 cascade outcome detector) AND
a segments file (``*_segments.json`` or ``*_segmentation.json``), this
command writes ``*_reach_assignments.json`` next to them: a permanent
per-reach output table where each reach has its outcome label already
stamped. Downstream kinematic analysis reads this directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mousereach.assignment.v1 import VERSION as V1_VERSION
from mousereach.assignment.v1 import assign_reaches_v1


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _segment_bounds_from_segmentation(seg_data: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Extract (start_frame, end_frame) per segment from the segments JSON.

    Tolerates the two known shapes:
      - {"segments": [{"segment_num", "start_frame", "end_frame"}, ...]}
      - {"boundaries": [{"frame": int}, ...]} (segments are pairs of
        consecutive boundaries; segment_num = i+1; end is boundary[i+1]-1)
    """
    if "segments" in seg_data:
        out = []
        for s in seg_data["segments"]:
            sf = s.get("start_frame")
            ef = s.get("end_frame")
            if sf is None or ef is None:
                continue
            out.append((int(sf), int(ef)))
        return out
    boundaries = seg_data.get("boundaries", [])
    frames = []
    for b in boundaries:
        if isinstance(b, dict):
            frames.append(int(b["frame"]))
        else:
            frames.append(int(b))
    return [(frames[i], frames[i + 1] - 1) for i in range(len(frames) - 1)]


def _segments_with_outcomes(
    segments_doc: Dict[str, Any],
    outcomes_doc: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Merge segment bounds with cascade outcomes into the shape
    `assign_reaches_v1` expects."""
    seg_bounds = _segment_bounds_from_segmentation(segments_doc)
    out_segs_by_num = {
        s["segment_num"]: s
        for s in outcomes_doc.get("segments", [])
        if s.get("segment_num") is not None
    }

    merged = []
    for i, (sf, ef) in enumerate(seg_bounds):
        seg_num = i + 1
        outcome = out_segs_by_num.get(seg_num, {})
        merged.append({
            "segment_num": seg_num,
            "start_frame": sf,
            "end_frame": ef,
            "outcome": outcome.get("outcome"),
            "interaction_frame": outcome.get("interaction_frame"),
            "outcome_known_frame": outcome.get("outcome_known_frame"),
            "flagged_for_review": bool(outcome.get("flagged_for_review", False)),
        })
    return merged


def _reaches_list(reaches_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "reaches" in reaches_doc:
        return list(reaches_doc["reaches"])
    return []


def _find_video_inputs(video_dir: Path) -> Optional[Tuple[Path, Path, Path]]:
    """Resolve the (segments_json, reaches_json, outcomes_json) trio
    in `video_dir`. Returns None if any are missing."""
    seg_candidates = list(video_dir.glob("*_segmentation.json")) + list(video_dir.glob("*_segments.json"))
    reach_candidates = list(video_dir.glob("*_reaches.json"))
    outcome_candidates = list(video_dir.glob("*_pellet_outcomes.json"))
    if not seg_candidates or not reach_candidates or not outcome_candidates:
        return None
    return seg_candidates[0], reach_candidates[0], outcome_candidates[0]


def _process_video_dir(video_dir: Path) -> Optional[Path]:
    inputs = _find_video_inputs(video_dir)
    if inputs is None:
        return None
    seg_path, reach_path, outcome_path = inputs

    segments_doc = _load_json(seg_path)
    reaches_doc = _load_json(reach_path)
    outcomes_doc = _load_json(outcome_path)

    video_id = (
        outcomes_doc.get("video_id")
        or reaches_doc.get("video_id")
        or video_dir.name
    )

    merged_segs = _segments_with_outcomes(segments_doc, outcomes_doc)
    reaches = _reaches_list(reaches_doc)

    result = assign_reaches_v1(
        reaches=reaches,
        segments_with_outcomes=merged_segs,
        video_id=video_id,
    )

    out_path = video_dir / f"{video_id}_reach_assignments.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return out_path


def _walk_input_root(root: Path) -> Iterable[Path]:
    """Yield directories that look like per-video processing dirs."""
    if not root.exists():
        return
    if _find_video_inputs(root) is not None:
        yield root
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and _find_video_inputs(child) is not None:
            yield child


def main_batch():
    parser = argparse.ArgumentParser(
        description=("Stamp per-reach outcome labels by joining v6 cascade "
                     "outcomes onto v8 reach detector outputs (assignment v1)."),
    )
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Processing root or single video dir.")
    args = parser.parse_args()

    print(f"mousereach-assign-reaches v{V1_VERSION}")
    print(f"  input: {args.input}")

    written = []
    for vdir in _walk_input_root(args.input):
        out = _process_video_dir(vdir)
        if out is not None:
            print(f"  wrote {out}")
            written.append(out)

    if not written:
        print("  no per-video inputs found (need *_segmentation.json + "
              "*_reaches.json + *_pellet_outcomes.json)")
        sys.exit(1)
    print(f"Done. {len(written)} reach-assignment files written.")


if __name__ == "__main__":
    main_batch()
