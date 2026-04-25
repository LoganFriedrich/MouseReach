"""
Segmentation boundary-accuracy metrics for the MouseReach Improvement Process.

Matching rule
-------------
For each GT boundary, find the algo boundary with minimum
|algo_frame - gt_frame| within a +/-*window* frame window (default 20).

- None found --> that GT boundary is a **miss**.
- Found --> GT boundary is **matched**; signed_delta = algo_frame - gt_frame
  (+ = algo late, - = algo early).
- Algo boundaries not matched to any GT --> **phantom** (false positive).
- If two GT boundaries claim the same algo boundary, the closer one wins;
  the other GT boundary becomes a miss.

Boundary subsets
----------------
- ``all``: all 21 boundaries per video (indices 0-20, 1-indexed B1-B21).
- ``inter_pellet_B2_B20``: boundaries 2-20 (1-indexed), the 19 inter-pellet
  transitions. These matter most for per-pellet scoring.
- ``endpoint_B1_B21``: boundaries 1 and 21 only (first and last).

Deliverables written to output_dir
-----------------------------------
- ``boundary_deltas.csv`` -- one row per emitted algo boundary OR per
  unmatched GT boundary.
- ``per_video.csv`` -- one row per video with aggregate stats.
- ``scalars.json`` -- nested structure with histogram and summary stats
  per subset.

Usage::

    from mousereach.improvement.segmentation.metrics import (
        compute_segmentation_metrics,
        match_boundaries,
    )

    scalars = compute_segmentation_metrics(
        gt_dir=Path("Y:/.../gt"),
        algo_dir=Path("Y:/.../outputs_v2.2.0"),
        output_dir=Path("Y:/.../metrics"),
        video_ids=None,  # auto-discover from GT dir
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
class MatchResult:
    """Result of matching a single boundary (GT or algo).

    Attributes
    ----------
    status : str
        One of "matched", "miss", "phantom".
    algo_boundary_index : int or None
        0-based index into the algo boundary list (None for misses).
    gt_boundary_index : int or None
        0-based index into the GT boundary list (None for phantoms).
    algo_frame : int or None
        Frame number from algo (None for misses).
    gt_frame : int or None
        Frame number from GT (None for phantoms).
    signed_delta : int or None
        algo_frame - gt_frame (None for miss/phantom).
    """
    status: str
    algo_boundary_index: Optional[int]
    gt_boundary_index: Optional[int]
    algo_frame: Optional[int]
    gt_frame: Optional[int]
    signed_delta: Optional[int]


# ---------------------------------------------------------------------------
# Pure matching logic
# ---------------------------------------------------------------------------

def match_boundaries(
    algo_frames: List[int],
    gt_frames: List[int],
    window: int = 20,
) -> List[MatchResult]:
    """Match algo boundaries to GT boundaries within a +/- *window* tolerance.

    Algorithm:
      1. For each GT boundary, find the closest algo boundary within the
         window. Record (gt_idx, algo_idx, abs_delta).
      2. Sort all candidate pairs by abs_delta ascending.
      3. Greedily assign: each GT and each algo boundary can appear in at
         most one match. If a GT or algo boundary is already assigned, skip.
      4. Unmatched GT boundaries -> miss. Unmatched algo boundaries -> phantom.

    Parameters
    ----------
    algo_frames : list of int
        Algo-emitted boundary frames (typically 21 ints).
    gt_frames : list of int
        Ground-truth boundary frames (typically 21 ints).
    window : int
        Maximum absolute distance for a match (inclusive).

    Returns
    -------
    list of MatchResult
        One entry per matched pair, plus one per miss and one per phantom.
    """
    # Build candidate pairs: (abs_delta, gt_idx, algo_idx)
    candidates: List[Tuple[int, int, int]] = []
    for gi, gf in enumerate(gt_frames):
        for ai, af in enumerate(algo_frames):
            d = abs(af - gf)
            if d <= window:
                candidates.append((d, gi, ai))

    # Sort by abs_delta so closest pairs are assigned first
    candidates.sort(key=lambda x: x[0])

    matched_gt: set = set()
    matched_algo: set = set()
    results: List[MatchResult] = []

    for _abs_d, gi, ai in candidates:
        if gi in matched_gt or ai in matched_algo:
            continue
        matched_gt.add(gi)
        matched_algo.add(ai)
        sd = algo_frames[ai] - gt_frames[gi]
        results.append(MatchResult(
            status="matched",
            algo_boundary_index=ai,
            gt_boundary_index=gi,
            algo_frame=algo_frames[ai],
            gt_frame=gt_frames[gi],
            signed_delta=sd,
        ))

    # Misses: GT boundaries with no match
    for gi, gf in enumerate(gt_frames):
        if gi not in matched_gt:
            results.append(MatchResult(
                status="miss",
                algo_boundary_index=None,
                gt_boundary_index=gi,
                algo_frame=None,
                gt_frame=gf,
                signed_delta=None,
            ))

    # Phantoms: algo boundaries with no match
    for ai, af in enumerate(algo_frames):
        if ai not in matched_algo:
            results.append(MatchResult(
                status="phantom",
                algo_boundary_index=ai,
                gt_boundary_index=None,
                algo_frame=af,
                gt_frame=None,
                signed_delta=None,
            ))

    return results


# ---------------------------------------------------------------------------
# GT and algo loading
# ---------------------------------------------------------------------------

def _load_gt_boundaries(gt_path: Path) -> List[int]:
    """Load GT boundary frames from either split or unified GT format.

    Split format (``*_seg_ground_truth.json``):
        ``{"boundaries": [int, ...], ...}``

    Unified format (``*_unified_ground_truth.json``):
        ``{"segmentation": {"boundaries": [{"frame": int, ...}, ...]}}``

    Returns
    -------
    list of int
        Sorted boundary frame numbers.
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "segmentation" in data:
        # Unified format
        raw = data["segmentation"]["boundaries"]
        frames = [b["frame"] for b in raw]
    else:
        # Split format
        frames = data["boundaries"]

    return sorted(int(f) for f in frames)


def _load_algo_boundaries(algo_path: Path) -> List[int]:
    """Load algo boundary frames from a ``_segments.json`` file.

    Returns
    -------
    list of int
        Boundary frame numbers as stored (should already be sorted).
    """
    with open(algo_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [int(b) for b in data["boundaries"]]


def _discover_video_ids(gt_dir: Path) -> List[str]:
    """Discover video IDs from GT files in a directory.

    Looks for unified GT first (``*_unified_ground_truth.json``), then
    falls back to split seg GT (``*_seg_ground_truth.json``).

    Returns sorted list of video ID strings.
    """
    unified = list(gt_dir.glob("*_unified_ground_truth.json"))
    if unified:
        ids = sorted({f.stem.replace("_unified_ground_truth", "") for f in unified})
        return ids

    split = list(gt_dir.glob("*_seg_ground_truth.json"))
    if split:
        ids = sorted({f.stem.replace("_seg_ground_truth", "") for f in split})
        return ids

    return []


def _find_gt_file(gt_dir: Path, video_id: str) -> Optional[Path]:
    """Find the GT file for a video ID, preferring unified format."""
    unified = gt_dir / f"{video_id}_unified_ground_truth.json"
    if unified.exists():
        return unified
    split = gt_dir / f"{video_id}_seg_ground_truth.json"
    if split.exists():
        return split
    return None


def _find_algo_file(algo_dir: Path, video_id: str) -> Optional[Path]:
    """Find the algo _segments.json for a video ID."""
    exact = algo_dir / f"{video_id}_segments.json"
    if exact.exists():
        return exact
    return None


# ---------------------------------------------------------------------------
# Subset tagging
# ---------------------------------------------------------------------------

def _tag_subset(gt_index: int, n_gt: int) -> str:
    """Return the subset tag for a GT boundary by its 0-based index.

    Boundaries are 1-indexed in user parlance (B1..B21), so:
    - B1 = index 0, B21 = index 20  -> endpoint_B1_B21
    - B2..B20 = index 1..19          -> inter_pellet_B2_B20
    """
    if gt_index == 0 or gt_index == n_gt - 1:
        return "endpoint_B1_B21"
    return "inter_pellet_B2_B20"


# ---------------------------------------------------------------------------
# Per-subset scalar computation
# ---------------------------------------------------------------------------

def _compute_subset_scalars(
    deltas: List[int],
    n_gt: int,
    n_algo: int,
    n_phantom: int,
    n_miss: int,
) -> Dict[str, Any]:
    """Compute scalars for one subset.

    Parameters
    ----------
    deltas : list of int
        Signed deltas for matched boundaries in this subset.
    n_gt : int
        Total GT boundaries in this subset.
    n_algo : int
        Total algo boundaries in this subset (matched + phantom).
    n_phantom : int
        Phantom count for this subset.
    n_miss : int
        Miss count for this subset.

    Returns
    -------
    dict
        Scalars including delta_histogram (non-cumulative, per exact delta).
    """
    histogram: Dict[str, int] = defaultdict(int)
    for d in deltas:
        histogram[str(d)] += 1

    abs_deltas = [abs(d) for d in deltas]

    return {
        "n_gt_boundaries": n_gt,
        "n_algo_boundaries": n_algo,
        "delta_histogram": dict(sorted(histogram.items(), key=lambda x: int(x[0]))),
        "n_phantom": n_phantom,
        "n_miss": n_miss,
        "mean_signed_delta": round(float(np.mean(deltas)), 2) if deltas else None,
        "median_abs_delta": int(np.median(abs_deltas)) if abs_deltas else None,
        "mean_abs_delta": round(float(np.mean(abs_deltas)), 2) if abs_deltas else None,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_segmentation_metrics(
    gt_dir: Path,
    algo_dir: Path,
    output_dir: Path,
    video_ids: Optional[List[str]] = None,
    window: int = 20,
) -> Dict[str, Any]:
    """Compute boundary-accuracy metrics for segmentation evaluation.

    Loads GT and algo boundaries for each video, matches them within
    a +/- *window* frame tolerance, and writes three deliverables:

    - ``boundary_deltas.csv``: one row per boundary event (matched, miss,
      or phantom) across all videos.
    - ``per_video.csv``: one row per video with aggregate stats.
    - ``scalars.json``: nested per-subset summary with delta histogram.

    Parameters
    ----------
    gt_dir : Path
        Directory containing GT files (unified or split format).
    algo_dir : Path
        Directory containing ``*_segments.json`` algo output files.
    output_dir : Path
        Where to write the three deliverables.
    video_ids : list of str, optional
        Explicit list of video IDs to evaluate. If None, auto-discovers
        from GT files in *gt_dir*.
    window : int
        Maximum frame distance for boundary matching (inclusive, default 20).

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
        logger.warning("No video IDs found in %s", gt_dir)
        return {}

    # Collect all rows for boundary_deltas.csv
    delta_rows: List[Dict[str, Any]] = []
    # Collect per-video summary rows
    video_rows: List[Dict[str, Any]] = []

    # Per-subset accumulators
    subset_deltas: Dict[str, List[int]] = defaultdict(list)
    subset_n_gt: Dict[str, int] = defaultdict(int)
    subset_n_algo: Dict[str, int] = defaultdict(int)
    subset_n_phantom: Dict[str, int] = defaultdict(int)
    subset_n_miss: Dict[str, int] = defaultdict(int)

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

        gt_frames = _load_gt_boundaries(gt_file)
        algo_frames = _load_algo_boundaries(algo_file)
        n_gt = len(gt_frames)

        results = match_boundaries(algo_frames, gt_frames, window=window)

        # Per-video accumulators
        vid_matched = 0
        vid_phantom = 0
        vid_miss = 0
        vid_deltas: List[int] = []

        for mr in results:
            # Determine subset tag
            if mr.gt_boundary_index is not None:
                subset = _tag_subset(mr.gt_boundary_index, n_gt)
            else:
                # Phantom -- no GT index. Tag as "all" only.
                subset = None

            row = {
                "video_id": vid,
                "boundary_source": "gt" if mr.status != "phantom" else "algo",
                "algo_boundary_index": mr.algo_boundary_index,
                "gt_boundary_index": mr.gt_boundary_index,
                "algo_frame": mr.algo_frame,
                "gt_frame": mr.gt_frame,
                "signed_delta": mr.signed_delta,
                "status": mr.status,
                "subset_tag": subset if subset else "",
            }
            delta_rows.append(row)

            if mr.status == "matched":
                vid_matched += 1
                vid_deltas.append(mr.signed_delta)

                # Accumulate into "all" and the specific subset
                subset_deltas["all"].append(mr.signed_delta)
                subset_n_gt["all"] += 1
                subset_n_algo["all"] += 1
                if subset:
                    subset_deltas[subset].append(mr.signed_delta)
                    subset_n_gt[subset] += 1
                    subset_n_algo[subset] += 1

            elif mr.status == "miss":
                vid_miss += 1
                subset_n_gt["all"] += 1
                subset_n_miss["all"] += 1
                if subset:
                    subset_n_gt[subset] += 1
                    subset_n_miss[subset] += 1

            elif mr.status == "phantom":
                vid_phantom += 1
                subset_n_algo["all"] += 1
                subset_n_phantom["all"] += 1
                # Phantoms don't belong to a GT-indexed subset

        abs_deltas = [abs(d) for d in vid_deltas]
        video_rows.append({
            "video_id": vid,
            "n_gt": n_gt,
            "n_algo": len(algo_frames),
            "n_matched": vid_matched,
            "n_phantom": vid_phantom,
            "n_miss": vid_miss,
            "median_abs_delta": int(np.median(abs_deltas)) if abs_deltas else None,
            "mean_abs_delta": round(float(np.mean(abs_deltas)), 2) if abs_deltas else None,
            "mean_signed_delta": round(float(np.mean(vid_deltas)), 2) if vid_deltas else None,
        })

    # Build scalars
    subsets = ["all", "inter_pellet_B2_B20", "endpoint_B1_B21"]
    scalars: Dict[str, Any] = {}
    for s in subsets:
        scalars[s] = _compute_subset_scalars(
            deltas=subset_deltas.get(s, []),
            n_gt=subset_n_gt.get(s, 0),
            n_algo=subset_n_algo.get(s, 0),
            n_phantom=subset_n_phantom.get(s, 0),
            n_miss=subset_n_miss.get(s, 0),
        )

    # Write deliverables
    df_deltas = pd.DataFrame(delta_rows)
    df_deltas.to_csv(output_dir / "boundary_deltas.csv", index=False)

    df_videos = pd.DataFrame(video_rows)
    df_videos.to_csv(output_dir / "per_video.csv", index=False)

    with open(output_dir / "scalars.json", "w", encoding="utf-8") as f:
        json.dump(scalars, f, indent=2, ensure_ascii=False)

    n_processed = len(video_ids) - len(skipped)
    logger.info(
        "Segmentation metrics: %d videos processed, %d skipped",
        n_processed, len(skipped),
    )

    return scalars
