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
    """Extract a flat list of reach dicts from a reach detector output.

    Handles two schemas:
      - v8+ (flat): top-level ``reaches: [...]``
      - v7.x (nested): top-level ``segments: [{reaches: [...]}, ...]``
    """
    if isinstance(reaches_doc.get("reaches"), list):
        return list(reaches_doc["reaches"])
    flat: List[Dict[str, Any]] = []
    for seg in reaches_doc.get("segments", []) or []:
        seg_reaches = seg.get("reaches") or []
        seg_num = seg.get("segment_num")
        for r in seg_reaches:
            # Stamp the segment_num onto the reach dict if not already present
            # so downstream code can group reaches by segment.
            if "segment_num" not in r and seg_num is not None:
                r = {**r, "segment_num": seg_num}
            flat.append(r)
    return flat


def _find_video_inputs(video_dir: Path) -> Optional[Tuple[Path, Path, Path]]:
    """Resolve the (segments_json, reaches_json, outcomes_json) trio
    in `video_dir`. Returns None if any are missing.

    Picks the first matching candidate of each type. Use ``_find_inputs_for_stem``
    when the dir contains multiple videos and the caller knows which stem
    to resolve.
    """
    seg_candidates = list(video_dir.glob("*_segmentation.json")) + list(video_dir.glob("*_segments.json"))
    reach_candidates = list(video_dir.glob("*_reaches.json"))
    outcome_candidates = list(video_dir.glob("*_pellet_outcomes.json"))
    if not seg_candidates or not reach_candidates or not outcome_candidates:
        return None
    return seg_candidates[0], reach_candidates[0], outcome_candidates[0]


def _find_inputs_for_stem(root: Path, stem: str) -> Optional[Tuple[Path, Path, Path]]:
    """Resolve inputs for a specific video stem in a flat-layout dir."""
    seg = root / f"{stem}_segments.json"
    if not seg.exists():
        seg = root / f"{stem}_segmentation.json"
    reach = root / f"{stem}_reaches.json"
    out = root / f"{stem}_pellet_outcomes.json"
    if not seg.exists() or not reach.exists() or not out.exists():
        return None
    return seg, reach, out


def _process_inputs(seg_path: Path, reach_path: Path, outcome_path: Path,
                     output_dir: Path, video_id_hint: str) -> Optional[Path]:
    """Run assignment v1 on a single resolved input triple. Returns output path."""
    segments_doc = _load_json(seg_path)
    reaches_doc = _load_json(reach_path)
    outcomes_doc = _load_json(outcome_path)

    video_id = (
        outcomes_doc.get("video_id")
        or reaches_doc.get("video_id")
        or video_id_hint
    )

    merged_segs = _segments_with_outcomes(segments_doc, outcomes_doc)
    reaches = _reaches_list(reaches_doc)

    result = assign_reaches_v1(
        reaches=reaches,
        segments_with_outcomes=merged_segs,
        video_id=video_id,
    )

    out_path = output_dir / f"{video_id}_reach_assignments.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return out_path


def _process_video_dir(video_dir: Path) -> Optional[Path]:
    """Single-video-dir processing. Kept for the legacy per-video-subdir layout."""
    inputs = _find_video_inputs(video_dir)
    if inputs is None:
        return None
    return _process_inputs(*inputs, output_dir=video_dir, video_id_hint=video_dir.name)


def _walk_input_root(root: Path) -> Iterable[Path]:
    """Yield directories that look like per-video processing dirs.

    Three layouts are supported:
      1. ``root`` itself has exactly one video's worth of files.
      2. ``root`` contains per-video subdirectories.
      3. ``root`` contains MANY videos' files flat (e.g., an improvement
         quarantine's ``algo_outputs_*``/ folder). In this case we yield
         ``root`` once per video stem so ``_process_video_dir_for_stem``
         can pick the right files.
    """
    if not root.exists():
        return
    # Count videos in flat layout: number of distinct stems with all 3 files
    flat_stems = _flat_layout_stems(root)
    if len(flat_stems) > 1:
        # Flat layout with multiple videos. Yield root once per stem.
        for stem in flat_stems:
            yield (root, stem)
        return
    # Single-video at root
    if _find_video_inputs(root) is not None:
        yield root
        return
    # Per-video subdirectories
    for child in sorted(root.iterdir()):
        if child.is_dir() and _find_video_inputs(child) is not None:
            yield child


def _flat_layout_stems(root: Path) -> list:
    """Return sorted list of video stems where root has segs+reaches+outcomes."""
    stems = []
    for seg in sorted(root.glob("*_segments.json")):
        stem = seg.stem[: -len("_segments")]
        if (root / f"{stem}_reaches.json").exists() and (root / f"{stem}_pellet_outcomes.json").exists():
            stems.append(stem)
    return stems


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
    for item in _walk_input_root(args.input):
        # _walk_input_root yields either a Path (per-video dir layout) or
        # a (root, stem) tuple (flat layout with multiple videos).
        if isinstance(item, tuple):
            root, stem = item
            inputs = _find_inputs_for_stem(root, stem)
            if inputs is None:
                continue
            out = _process_inputs(*inputs, output_dir=root, video_id_hint=stem)
        else:
            out = _process_video_dir(item)
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
