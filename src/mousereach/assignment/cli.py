"""
CLI for reach assignment (algo-4).

Production usage:
    mousereach-assign-reaches -i Processing/

For each video under the input root that has ``*_segments.json`` (from the
segmenter), ``*_reaches.json`` (from the v8 reach detector),
``*_pellet_outcomes.json`` (from the v6 cascade outcome detector) AND its
DLC pose ``*DLC*.h5`` side by side, this command writes
``*_reach_assignments.json`` next to them: a permanent per-reach output
table where each reach has its outcome label and causal-reach decision
already stamped. Downstream kinematic analysis reads this directly.

This command runs the SAME code path as the automatic pipeline:
``mousereach.assignment.run.assign_reaches_for_video`` (assignment v2, the
two-signal agreement gate), which the watcher, ``pipeline/run_all.py`` and
``pipeline/reprocess_to_current.py`` all call. It has no assignment logic
of its own.

WHY: until 2026-08 this command called assignment v1 (a single-signal IFR
join, ``1.0.0``) while everything automatic ran v2 (``2.1.0``). Running it
by hand over a processing folder silently overwrote the v2 file the
pipeline had written with a weaker v1 file at the same path, and the
processing manifest kept saying 2.1.0. v1 is retained under
``assignment/v1`` for provenance but is no longer reachable from any
command.

The input readers ``_segment_bounds_from_segmentation``,
``_segments_with_outcomes`` and ``_reaches_list`` live here because
``assignment/run.py`` and ``review/staging.py`` import them; they are the
one shared reading of the three input files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mousereach.assignment.run import assign_reaches_for_video
from mousereach.assignment.v2 import VERSION


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
    """Merge segment bounds with cascade outcomes into the shape the
    assignment algorithms expect."""
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
      - nested (what the v8 reach detector writes, and what every file in
        the Analyzed tree carries): top-level ``segments: [{reaches: [...]}]``
      - flat (older / alternate form): top-level ``reaches: [...]``
    The flat form is honoured first only because a file carrying both would
    be declaring the flat list authoritative; no current file carries both.
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


# ---------------------------------------------------------------------------
# Finding videos to assign
# ---------------------------------------------------------------------------

def _stems_in(directory: Path) -> List[str]:
    """Video stems in ``directory`` that carry all three assignment inputs.

    The file names are exactly the ones ``assign_reaches_for_video`` reads
    (``{stem}_segments.json`` etc.), so anything listed here is something the
    production step can actually run on.
    """
    stems = []
    for seg in sorted(directory.glob("*_segments.json")):
        stem = seg.name[: -len("_segments.json")]
        if ((directory / f"{stem}_reaches.json").exists()
                and (directory / f"{stem}_pellet_outcomes.json").exists()):
            stems.append(stem)
    return stems


def _pose_for(directory: Path, stem: str) -> Optional[Path]:
    """The DLC pose h5 for ``stem`` in ``directory``, or None if there is none.

    Assignment v2 needs the pose for its pellet-displacement signal, so a
    video without one cannot be assigned by the production path. When more
    than one pose file matches (two DLC models analysed the same video) the
    first in sorted order is used -- the rule ``pipeline/run_all.py`` applies
    -- and the choice is printed so a wrong pick can never be silent.
    """
    h5s = sorted(directory.glob(f"{stem}DLC*.h5"))
    if not h5s:
        return None
    if len(h5s) > 1:
        print(f"  [!] {stem}: {len(h5s)} pose files match, using {h5s[0].name}")
    return h5s[0]


def _candidate_dirs(root: Path) -> Iterable[Path]:
    """Directories to look in: ``root`` itself when it holds the files of
    one or many videos side by side (the flat Processing/ layout), otherwise
    its immediate subdirectories (the legacy one-folder-per-video layout)."""
    if not root.exists():
        return
    if _stems_in(root):
        yield root
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and _stems_in(child):
            yield child


def main_batch():
    parser = argparse.ArgumentParser(
        description=("Stamp per-reach outcome labels and the causal reach by "
                     "joining v6 cascade outcomes onto v8 reach detector "
                     f"outputs (assignment v{VERSION}, the same code path the "
                     "automatic pipeline runs)."),
    )
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Processing root or single video dir.")
    args = parser.parse_args()

    print(f"mousereach-assign-reaches (assignment v{VERSION})")
    print(f"  input: {args.input}")

    written: List[Path] = []
    skipped = 0
    for directory in _candidate_dirs(args.input):
        for stem in _stems_in(directory):
            pose = _pose_for(directory, stem)
            if pose is None:
                print(f"  [!] {stem}: no DLC pose (*DLC*.h5) beside the inputs; "
                      "skipped (assignment v2 needs the pose)")
                skipped += 1
                continue
            result = assign_reaches_for_video(directory, stem, pose)
            if result is None:
                # The runner logs which input it could not read.
                print(f"  [!] {stem}: assignment did not run (missing input)")
                skipped += 1
                continue
            out = directory / f"{stem}_reach_assignments.json"
            reaches = result.get("reaches", [])
            n_causal = sum(1 for r in reaches if r.get("is_causal"))
            print(f"  wrote {out} ({n_causal} causal / {len(reaches)} reaches)")
            written.append(out)

    if not written:
        print("  no per-video inputs found (need *_segments.json + *_reaches.json + "
              "*_pellet_outcomes.json + *DLC*.h5 side by side)")
        sys.exit(1)
    print(f"Done. {len(written)} reach-assignment files written"
          + (f", {skipped} skipped" if skipped else "") + ".")


if __name__ == "__main__":
    main_batch()
