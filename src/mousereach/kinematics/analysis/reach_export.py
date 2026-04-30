#!/usr/bin/env python3
"""Export reach-level kinematics to CSV for downstream analysis.

Reads ``*_features.json`` (Step 5 output) when present and emits the full
extended kinematic feature set per reach. Falls back to ``*_reaches.json`` +
``*_pellet_outcomes.json`` for videos without a features file (in which case
only the basic identifier / outcome / extent columns are populated).

Column dictionary lives at ``docs/REACH_KINEMATIC_DATA_DICTIONARY.md``.
All temporal values are in frames; a separate framerate is required to
convert to seconds and is intentionally not multiplied in here.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from mousereach.config import Paths


# Static (non-extended) columns that come from the dataclass / JSON top-level.
# Order matters for CSV column ordering.
STATIC_COLUMNS: List[str] = [
    "video",
    "segment_num",
    "reach_num",
    "reach_id",
    "outcome",
    "causal_reach",
    "interaction_frame",
    "distance_to_interaction",
    "is_first_reach",
    "is_last_reach",
    "n_reaches_in_segment",
    "start_frame",
    "apex_frame",
    "end_frame",
    "duration_frames",
    "max_extent_pixels",
    "max_extent_ruler",
    "max_extent_mm",
    "hand_angle_at_apex_deg",
    "hand_rotation_total_deg",
    "head_width_at_apex_mm",
    "nose_to_slit_at_apex_mm",
    "head_angle_at_apex_deg",
    "head_angle_change_deg",
    "mean_likelihood",
    "frames_low_confidence",
    "flagged_for_review",
    "flag_reason",
    "source",
    "human_corrected",
]


def _row_from_features_reach(
    video: str,
    segment_num: int,
    seg_outcome: Optional[str],
    reach: Dict[str, Any],
) -> Dict[str, Any]:
    """Flatten one reach record from features.json into a CSV row."""
    row: Dict[str, Any] = {
        "video": video,
        "segment_num": segment_num,
        "reach_num": reach.get("reach_num"),
        "reach_id": reach.get("reach_id"),
        "outcome": reach.get("outcome") or seg_outcome,
        "causal_reach": reach.get("causal_reach"),
        "interaction_frame": reach.get("interaction_frame"),
        "distance_to_interaction": reach.get("distance_to_interaction"),
        "is_first_reach": reach.get("is_first_reach"),
        "is_last_reach": reach.get("is_last_reach"),
        "n_reaches_in_segment": reach.get("n_reaches_in_segment"),
        "start_frame": reach.get("start_frame"),
        "apex_frame": reach.get("apex_frame"),
        "end_frame": reach.get("end_frame"),
        "duration_frames": reach.get("duration_frames"),
        "max_extent_pixels": reach.get("max_extent_pixels"),
        "max_extent_ruler": reach.get("max_extent_ruler"),
        "max_extent_mm": reach.get("max_extent_mm"),
        "hand_angle_at_apex_deg": reach.get("hand_angle_at_apex_deg"),
        "hand_rotation_total_deg": reach.get("hand_rotation_total_deg"),
        "head_width_at_apex_mm": reach.get("head_width_at_apex_mm"),
        "nose_to_slit_at_apex_mm": reach.get("nose_to_slit_at_apex_mm"),
        "head_angle_at_apex_deg": reach.get("head_angle_at_apex_deg"),
        "head_angle_change_deg": reach.get("head_angle_change_deg"),
        "mean_likelihood": reach.get("mean_likelihood"),
        "frames_low_confidence": reach.get("frames_low_confidence"),
        "flagged_for_review": reach.get("flagged_for_review"),
        "flag_reason": reach.get("flag_reason"),
        # `source` and `human_corrected` aren't part of ReachFeatures and
        # would only come from the legacy reach-detector record. Leave blank.
        "source": None,
        "human_corrected": None,
    }
    extended = reach.get("extended") or {}
    if isinstance(extended, dict):
        for k, v in extended.items():
            row[k] = v
    return row


def _row_from_reaches_json(
    video: str,
    segment_num: int,
    ruler_pixels: float,
    seg_outcome: Optional[str],
    reach: Dict[str, Any],
) -> Dict[str, Any]:
    """Flatten one reach record from the legacy reaches.json schema (no extended)."""
    extent_ruler = reach.get("max_extent_ruler")
    extent_mm = extent_ruler * 9.0 if extent_ruler is not None else None
    return {
        "video": video,
        "segment_num": segment_num,
        "reach_num": reach.get("reach_num", reach.get("reach_id")),
        "reach_id": reach.get("reach_id"),
        "outcome": seg_outcome,
        "causal_reach": None,
        "interaction_frame": None,
        "distance_to_interaction": None,
        "is_first_reach": None,
        "is_last_reach": None,
        "n_reaches_in_segment": None,
        "start_frame": reach.get("start_frame"),
        "apex_frame": reach.get("apex_frame"),
        "end_frame": reach.get("end_frame"),
        "duration_frames": reach.get("duration_frames"),
        "max_extent_pixels": reach.get("max_extent_pixels"),
        "max_extent_ruler": extent_ruler,
        "max_extent_mm": extent_mm,
        "hand_angle_at_apex_deg": None,
        "hand_rotation_total_deg": None,
        "head_width_at_apex_mm": None,
        "nose_to_slit_at_apex_mm": None,
        "head_angle_at_apex_deg": None,
        "head_angle_change_deg": None,
        "mean_likelihood": None,
        "frames_low_confidence": None,
        "flagged_for_review": None,
        "flag_reason": None,
        "source": reach.get("source", "algorithm"),
        "human_corrected": reach.get("human_corrected", False),
    }


def _emit_features_video(features_path: Path) -> List[Dict[str, Any]]:
    """Read a features.json and return one row per reach."""
    with open(features_path) as f:
        data = json.load(f)
    video = data.get("video_name") or features_path.stem.replace("_features", "")
    rows: List[Dict[str, Any]] = []
    for seg in data.get("segments", []):
        seg_num = seg.get("segment_num")
        seg_outcome = seg.get("outcome")
        for reach in seg.get("reaches", []):
            rows.append(_row_from_features_reach(video, seg_num, seg_outcome, reach))
    return rows


def _emit_reaches_video(reaches_path: Path) -> List[Dict[str, Any]]:
    """Fallback: emit basic rows when features.json is unavailable."""
    video = reaches_path.stem.replace("_reaches", "")
    with open(reaches_path) as f:
        reach_data = json.load(f)

    outcome_path = reaches_path.parent / f"{video}_pellet_outcomes.json"
    outcomes_by_seg: Dict[int, str] = {}
    if outcome_path.exists():
        with open(outcome_path) as f:
            for seg in json.load(f).get("segments", []):
                outcomes_by_seg[seg["segment_num"]] = seg.get("outcome", "unknown")

    rows: List[Dict[str, Any]] = []
    for seg in reach_data.get("segments", []):
        seg_num = seg["segment_num"]
        seg_outcome = outcomes_by_seg.get(seg_num)
        ruler_px = seg.get("ruler_pixels", 34.3)
        for reach in seg.get("reaches", []):
            rows.append(_row_from_reaches_json(video, seg_num, ruler_px, seg_outcome, reach))
    return rows


def collect_rows(processing_dir: Path) -> List[Dict[str, Any]]:
    """Walk a Processing/ directory and emit one row per reach across all videos."""
    rows: List[Dict[str, Any]] = []
    feature_files = sorted(processing_dir.glob("*_features.json"))
    feature_video_names = {f.stem.replace("_features", "") for f in feature_files}

    for fp in feature_files:
        rows.extend(_emit_features_video(fp))

    # Fallback for videos with reaches.json but no features.json yet.
    for rp in sorted(processing_dir.glob("*_reaches.json")):
        video = rp.stem.replace("_reaches", "")
        if video in feature_video_names:
            continue
        rows.extend(_emit_reaches_video(rp))
    return rows


def write_csv(rows: List[Dict[str, Any]], output_csv: Path) -> List[str]:
    """Write rows to CSV with stable column ordering and return the column list."""
    if not rows:
        return []
    extended_keys: List[str] = []
    seen = set(STATIC_COLUMNS)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                extended_keys.append(k)
                seen.add(k)
    fieldnames = STATIC_COLUMNS + extended_keys

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return fieldnames


def main() -> None:
    processing = Paths.PROCESSING
    output_csv = processing.parent / "reach_kinematics.csv"
    rows = collect_rows(processing)
    if not rows:
        print("No reaches found.")
        return
    fieldnames = write_csv(rows, output_csv)
    print(f"Exported {len(rows)} reaches to: {output_csv}")
    print(f"Columns: {len(fieldnames)} (static: {len(STATIC_COLUMNS)}, extended: {len(fieldnames) - len(STATIC_COLUMNS)})")

    # Quick stats per outcome.
    from collections import defaultdict
    by_outcome = defaultdict(list)
    for r in rows:
        by_outcome[r.get("outcome") or "unknown"].append(r)
    print("\nBy outcome:")
    for outcome, reaches in sorted(by_outcome.items()):
        durs = [r["duration_frames"] for r in reaches if r.get("duration_frames")]
        ext = [r["max_extent_mm"] for r in reaches if r.get("max_extent_mm")]
        line = f"  {outcome:20s}: {len(reaches):4d} reaches"
        if durs:
            line += f" | duration_frames mean={sum(durs)/len(durs):.0f}"
        if ext:
            line += f" | max_extent_mm mean={sum(ext)/len(ext):.2f}"
        print(line)


if __name__ == "__main__":
    main()
