"""
Reach detection accuracy metrics for the MouseReach Improvement Process.

Matching rule
-------------
For each GT reach, find the algo reach with minimum
|algo_start - gt_start| within a +/-*window* frame window (default 10).

- None found --> that GT reach is a **fn** (miss / false negative).
- Found --> **matched** with:
    - start_delta = algo_start - gt_start (signed)
    - end_delta   = algo_end   - gt_end   (signed)
  (+ = algo late, - = algo early)
- Multiple algo reaches within the window of the same GT: the closest
  (by |start_delta|) wins. Others become **fp** (false positive / phantom).
- Algo reaches with no matching GT --> **fp** (phantom).

Rationale: mice complete most reaches in ~6 frames, contacted ones ~12f.
A +/-10f window is already past typical reach duration; wider would bleed
into adjacent reaches.

Deliverables written to output_dir
-----------------------------------
- ``reach_matches.csv`` -- one row per event (matched, fn, or fp).
- ``per_video.csv`` -- one row per video with aggregate stats.
- ``scalars.json`` -- global summary with count and delta histograms.

Usage::

    from mousereach.improvement.reach_detection.metrics import (
        compute_reach_detection_metrics,
        match_reaches,
    )

    scalars = compute_reach_detection_metrics(
        gt_dir=Path("Y:/.../gt"),
        algo_dir=Path("Y:/.../outputs_reach_v7.1.0"),
        output_dir=Path("Y:/.../metrics"),
    )
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Reach:
    """Minimal reach representation for matching.

    Attributes
    ----------
    start_frame : int
        First frame of the reach.
    end_frame : int
        Last frame of the reach.
    index : int
        0-based position in the source list (GT or algo).
    """
    start_frame: int
    end_frame: int
    index: int


@dataclass
class ReachMatchResult:
    """Result of matching a single reach (GT or algo).

    Attributes
    ----------
    status : str
        One of "matched", "fn", "fp".
    gt_reach_index : int or None
        0-based index into the GT reach list (None for fp).
    algo_reach_index : int or None
        0-based index into the algo reach list (None for fn).
    gt_start : int or None
        GT start_frame (None for fp).
    gt_end : int or None
        GT end_frame (None for fp).
    algo_start : int or None
        Algo start_frame (None for fn).
    algo_end : int or None
        Algo end_frame (None for fn).
    start_delta : int or None
        algo_start - gt_start (None for fn/fp).
    end_delta : int or None
        algo_end - gt_end (None for fn/fp).
    """
    status: str
    gt_reach_index: Optional[int]
    algo_reach_index: Optional[int]
    gt_start: Optional[int]
    gt_end: Optional[int]
    algo_start: Optional[int]
    algo_end: Optional[int]
    start_delta: Optional[int]
    end_delta: Optional[int]


# ---------------------------------------------------------------------------
# Pure matching logic
# ---------------------------------------------------------------------------

def match_reaches(
    algo_reaches: List[Reach],
    gt_reaches: List[Reach],
    window: int = 10,
) -> List[ReachMatchResult]:
    """Match algo reaches to GT reaches within a +/- *window* frame tolerance.

    Algorithm:
      1. For each GT reach, find all algo reaches whose start_frame is
         within +/- window frames.
      2. Build candidate pairs (gt_idx, algo_idx, abs_start_delta).
      3. Sort by abs_start_delta ascending.
      4. Greedily assign: each GT and each algo reach can appear in at
         most one match. If already assigned, skip.
      5. Unmatched GT reaches -> fn. Unmatched algo reaches -> fp.

    Parameters
    ----------
    algo_reaches : list of Reach
        Algorithm-detected reaches.
    gt_reaches : list of Reach
        Ground-truth reaches.
    window : int
        Maximum absolute start-frame distance for a match (inclusive).

    Returns
    -------
    list of ReachMatchResult
        One entry per matched pair, plus one per fn and one per fp.
    """
    # Build candidate pairs: (abs_delta, gt_idx, algo_idx)
    candidates: List[Tuple[int, int, int]] = []
    for gr in gt_reaches:
        for ar in algo_reaches:
            d = abs(ar.start_frame - gr.start_frame)
            if d <= window:
                candidates.append((d, gr.index, ar.index))

    # Sort by abs_delta so closest pairs are assigned first
    candidates.sort(key=lambda x: x[0])

    matched_gt: set = set()
    matched_algo: set = set()
    results: List[ReachMatchResult] = []

    for _abs_d, gi, ai in candidates:
        if gi in matched_gt or ai in matched_algo:
            continue
        matched_gt.add(gi)
        matched_algo.add(ai)
        gr = gt_reaches[gi]
        ar = algo_reaches[ai]
        results.append(ReachMatchResult(
            status="matched",
            gt_reach_index=gi,
            algo_reach_index=ai,
            gt_start=gr.start_frame,
            gt_end=gr.end_frame,
            algo_start=ar.start_frame,
            algo_end=ar.end_frame,
            start_delta=ar.start_frame - gr.start_frame,
            end_delta=ar.end_frame - gr.end_frame,
        ))

    # FN: GT reaches with no match
    for gr in gt_reaches:
        if gr.index not in matched_gt:
            results.append(ReachMatchResult(
                status="fn",
                gt_reach_index=gr.index,
                algo_reach_index=None,
                gt_start=gr.start_frame,
                gt_end=gr.end_frame,
                algo_start=None,
                algo_end=None,
                start_delta=None,
                end_delta=None,
            ))

    # FP: algo reaches with no match
    for ar in algo_reaches:
        if ar.index not in matched_algo:
            results.append(ReachMatchResult(
                status="fp",
                gt_reach_index=None,
                algo_reach_index=ar.index,
                gt_start=None,
                gt_end=None,
                algo_start=ar.start_frame,
                algo_end=ar.end_frame,
                start_delta=None,
                end_delta=None,
            ))

    return results


# ---------------------------------------------------------------------------
# GT and algo loading
# ---------------------------------------------------------------------------

def _load_gt_reaches(gt_path: Path) -> List[Reach]:
    """Load GT reaches from either unified or split GT format.

    Unified format (``*_unified_ground_truth.json``):
        ``{"reaches": {"exhaustive": true, "reaches": [{"start_frame": ..., "end_frame": ...}, ...]}}``

    Split format (``*_reach_ground_truth.json``):
        ``{"segments": [{"reaches": [{"start_frame": ..., "end_frame": ...}, ...]}, ...]}``

    Returns
    -------
    list of Reach
        Sorted by start_frame, with sequential 0-based indices.
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_reaches: List[dict] = []

    if "reaches" in data and isinstance(data["reaches"], dict):
        # Unified format
        reach_data = data["reaches"]
        raw_reaches = reach_data.get("reaches", [])
    elif "segments" in data:
        # Split format -- gather reaches from all segments
        for seg in data["segments"]:
            for r in seg.get("reaches", []):
                raw_reaches.append(r)

    # Filter out excluded reaches
    filtered = [
        r for r in raw_reaches
        if not r.get("exclude_from_analysis", False)
    ]

    # Sort by start_frame and assign indices
    filtered.sort(key=lambda r: r["start_frame"])
    return [
        Reach(
            start_frame=int(r["start_frame"]),
            end_frame=int(r["end_frame"]),
            index=i,
        )
        for i, r in enumerate(filtered)
    ]


def _load_algo_reaches(algo_path: Path) -> List[Reach]:
    """Load algo reaches from a ``*_reaches.json`` file.

    Format: ``{"segments": [{"reaches": [{"start_frame": ..., "end_frame": ...}, ...]}, ...]}``

    Returns
    -------
    list of Reach
        Sorted by start_frame, with sequential 0-based indices.
    """
    with open(algo_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_reaches: List[dict] = []
    for seg in data.get("segments", []):
        for r in seg.get("reaches", []):
            raw_reaches.append(r)

    # Filter out excluded reaches
    filtered = [
        r for r in raw_reaches
        if not r.get("exclude_from_analysis", False)
    ]

    filtered.sort(key=lambda r: r["start_frame"])
    return [
        Reach(
            start_frame=int(r["start_frame"]),
            end_frame=int(r["end_frame"]),
            index=i,
        )
        for i, r in enumerate(filtered)
    ]


def _discover_video_ids(gt_dir: Path) -> List[str]:
    """Discover video IDs with exhaustive reach GT from unified GT files.

    Only includes videos where ``reaches.exhaustive`` is True in the
    unified GT file. Falls back to split reach GT if no unified files.

    Returns sorted list of video ID strings.
    """
    # Check unified GT files first
    unified = list(gt_dir.glob("*_unified_ground_truth.json"))
    ids_from_unified = []
    if unified:
        for f in unified:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                reach_data = data.get("reaches", {})
                if (reach_data.get("exhaustive")
                        and reach_data.get("reaches")
                        and len(reach_data["reaches"]) > 0):
                    vid = f.stem.replace("_unified_ground_truth", "")
                    ids_from_unified.append(vid)
            except (json.JSONDecodeError, KeyError):
                continue

    # Also check split reach GT files
    split = list(gt_dir.glob("*_reach_ground_truth.json"))
    ids_from_split = []
    if split:
        for f in split:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # Only include if gt_complete is True
                if data.get("gt_complete"):
                    vid = f.stem.replace("_reach_ground_truth", "")
                    ids_from_split.append(vid)
            except (json.JSONDecodeError, KeyError):
                continue

    combined = sorted(set(ids_from_unified) | set(ids_from_split))
    return combined


def _find_gt_file(gt_dir: Path, video_id: str) -> Optional[Path]:
    """Find the GT file for a video ID, preferring unified format."""
    unified = gt_dir / f"{video_id}_unified_ground_truth.json"
    if unified.exists():
        # Verify it has exhaustive reach GT
        try:
            with open(unified, "r", encoding="utf-8") as f:
                data = json.load(f)
            reach_data = data.get("reaches", {})
            if (reach_data.get("exhaustive")
                    and reach_data.get("reaches")
                    and len(reach_data["reaches"]) > 0):
                return unified
        except (json.JSONDecodeError, KeyError):
            pass
    split = gt_dir / f"{video_id}_reach_ground_truth.json"
    if split.exists():
        return split
    return None


def _find_algo_file(algo_dir: Path, video_id: str) -> Optional[Path]:
    """Find the algo _reaches.json for a video ID."""
    exact = algo_dir / f"{video_id}_reaches.json"
    if exact.exists():
        return exact
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_reach_detection_metrics(
    gt_dir: Path,
    algo_dir: Path,
    output_dir: Path,
    video_ids: Optional[List[str]] = None,
    window: int = 10,
) -> Dict[str, Any]:
    """Compute reach detection metrics for evaluation.

    Loads GT and algo reaches for each video, matches them within
    a +/- *window* frame tolerance on start_frame, and writes three
    deliverables:

    - ``reach_matches.csv``: one row per reach event (matched, fn,
      or fp) across all videos.
    - ``per_video.csv``: one row per video with aggregate stats.
    - ``scalars.json``: global summary with count and delta histograms.

    Parameters
    ----------
    gt_dir : Path
        Directory containing GT files (unified or split format).
    algo_dir : Path
        Directory containing ``*_reaches.json`` algo output files.
    output_dir : Path
        Where to write the three deliverables.
    video_ids : list of str, optional
        Explicit list of video IDs to evaluate. If None, auto-discovers
        from GT files with exhaustive reach GT in *gt_dir*.
    window : int
        Maximum start-frame distance for reach matching (inclusive,
        default 10).

    Returns
    -------
    dict
        The scalars dict (same structure written to scalars.json).
    """
    gt_dir = Path(gt_dir)
    algo_dir = Path(algo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if video_ids is None:
        video_ids = _discover_video_ids(gt_dir)

    if not video_ids:
        logger.warning("No video IDs with reach GT found in %s", gt_dir)
        return {}

    # Collect all rows for reach_matches.csv
    match_rows: List[Dict[str, Any]] = []
    # Collect per-video summary rows
    video_rows: List[Dict[str, Any]] = []

    # Global accumulators
    total_gt = 0
    total_algo = 0
    total_matched = 0
    total_fp = 0
    total_fn = 0
    all_start_deltas: List[int] = []
    all_end_deltas: List[int] = []
    count_deltas: List[int] = []  # n_algo - n_gt per video
    n_perfect = 0  # videos where n_fp == 0 and n_fn == 0

    skipped = []

    for vid in video_ids:
        gt_file = _find_gt_file(gt_dir, vid)
        algo_file = _find_algo_file(algo_dir, vid)

        if gt_file is None:
            logger.warning("No GT file for %s -- skipping", vid)
            skipped.append(vid)
            continue
        if algo_file is None:
            logger.warning("No algo file for %s -- skipping", vid)
            skipped.append(vid)
            continue

        gt_reaches = _load_gt_reaches(gt_file)
        algo_reaches = _load_algo_reaches(algo_file)
        n_gt = len(gt_reaches)
        n_algo = len(algo_reaches)

        results = match_reaches(algo_reaches, gt_reaches, window=window)

        # Per-video accumulators
        vid_matched = 0
        vid_fp = 0
        vid_fn = 0
        vid_start_deltas: List[int] = []
        vid_end_deltas: List[int] = []

        for mr in results:
            row = {
                "video_id": vid,
                "gt_reach_index": mr.gt_reach_index,
                "gt_start": mr.gt_start,
                "gt_end": mr.gt_end,
                "algo_reach_index": mr.algo_reach_index,
                "algo_start": mr.algo_start,
                "algo_end": mr.algo_end,
                "start_delta": mr.start_delta,
                "end_delta": mr.end_delta,
                "status": mr.status,
            }
            match_rows.append(row)

            if mr.status == "matched":
                vid_matched += 1
                vid_start_deltas.append(mr.start_delta)
                vid_end_deltas.append(mr.end_delta)
            elif mr.status == "fn":
                vid_fn += 1
            elif mr.status == "fp":
                vid_fp += 1

        abs_start = [abs(d) for d in vid_start_deltas]
        abs_end = [abs(d) for d in vid_end_deltas]

        video_rows.append({
            "video_id": vid,
            "n_gt": n_gt,
            "n_algo": n_algo,
            "n_matched": vid_matched,
            "n_fp": vid_fp,
            "n_fn": vid_fn,
            "count_delta": n_algo - n_gt,
            "mean_abs_start_delta": round(float(np.mean(abs_start)), 2) if abs_start else None,
            "mean_abs_end_delta": round(float(np.mean(abs_end)), 2) if abs_end else None,
        })

        total_gt += n_gt
        total_algo += n_algo
        total_matched += vid_matched
        total_fp += vid_fp
        total_fn += vid_fn
        all_start_deltas.extend(vid_start_deltas)
        all_end_deltas.extend(vid_end_deltas)
        count_deltas.append(n_algo - n_gt)
        if vid_fp == 0 and vid_fn == 0:
            n_perfect += 1

    # Build histograms (non-cumulative, per exact frame)
    def _build_histogram(values: List[int]) -> Dict[str, int]:
        hist: Dict[str, int] = defaultdict(int)
        for v in values:
            hist[str(v)] += 1
        return dict(sorted(hist.items(), key=lambda x: int(x[0])))

    def _build_count_histogram(values: List[int]) -> Dict[str, int]:
        hist: Dict[str, int] = defaultdict(int)
        for v in values:
            hist[str(v)] += 1
        return dict(sorted(hist.items(), key=lambda x: int(x[0])))

    n_videos = len(video_ids) - len(skipped)

    scalars: Dict[str, Any] = {
        "n_videos": n_videos,
        "total": {
            "n_gt": total_gt,
            "n_algo": total_algo,
            "n_matched": total_matched,
            "n_fp": total_fp,
            "n_fn": total_fn,
        },
        "count_delta_per_video_histogram": _build_count_histogram(count_deltas),
        "start_delta_histogram": _build_histogram(all_start_deltas),
        "end_delta_histogram": _build_histogram(all_end_deltas),
        "n_perfect_videos": n_perfect,
    }

    # Write deliverables
    df_matches = pd.DataFrame(match_rows)
    df_matches.to_csv(output_dir / "reach_matches.csv", index=False)

    df_videos = pd.DataFrame(video_rows)
    df_videos.to_csv(output_dir / "per_video.csv", index=False)

    with open(output_dir / "scalars.json", "w", encoding="utf-8") as f:
        json.dump(scalars, f, indent=2, ensure_ascii=False)

    n_processed = len(video_ids) - len(skipped)
    logger.info(
        "Reach detection metrics: %d videos processed, %d skipped",
        n_processed, len(skipped),
    )

    return scalars
