"""
Shared input loaders for the per-algo improvement evaluators.

A snapshot directory is the unit of evaluation. Its layout:

  <snapshot_dir>/
    manifest.json     -- references the source corpus (gt_dir + algo_outputs_dir)
    algo_outputs/     -- {video_id}_segments.json,
                          {video_id}_reaches.json,
                          {video_id}_pellet_outcomes.json
    metrics/          -- written by analyze.py
    figures/          -- written by graph.py

Each analyze.py reads from algo_outputs/ + the gt_dir referenced in the
manifest. Each graph.py reads from metrics/ ONLY -- never raw data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SnapshotPaths:
    """Resolved paths for a snapshot."""
    snapshot_dir: Path
    gt_dir: Path
    algo_outputs_dir: Path
    metrics_dir: Path
    figures_dir: Path
    video_ids: List[str]


def load_snapshot_paths(snapshot_dir: Path) -> SnapshotPaths:
    """Read the snapshot's manifest.json and resolve paths.

    Manifest is expected to contain at least:
      gt_dir: path to GT JSONs
      algo_outputs_dir: path to algo output JSONs (defaults to
        snapshot_dir/algo_outputs if not given)
      video_ids: list of video IDs (defaults to discovery from gt_dir)

    Falls back to quarantine-layout defaults when the manifest is missing
    or under-specified: sibling ``gt/`` for GT JSONs, and the first of
    ``algo_outputs_current/`` / ``algo_outputs/`` that exists for algo
    output JSONs. This lets the eval framework run directly against
    iteration quarantines without an extra adapter manifest.
    """
    snapshot_dir = Path(snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {})

    gt_dir = Path(manifest["gt_dir"]) if "gt_dir" in manifest else None
    if gt_dir is None:
        # Quarantine convention: GT JSONs live in <snapshot>/gt/
        cand = snapshot_dir / "gt"
        if cand.is_dir():
            gt_dir = cand
        else:
            raise ValueError(
                f"manifest.json must specify gt_dir for snapshot {snapshot_dir} "
                f"(and no sibling gt/ directory was found)")

    if "algo_outputs_dir" in manifest:
        algo_outputs_dir = Path(manifest["algo_outputs_dir"])
    else:
        # Prefer algo_outputs_current/ (post-resolution, what production sees)
        # then algo_outputs/ (raw, pre-resolution baseline).
        for name in ("algo_outputs_current", "algo_outputs"):
            cand = snapshot_dir / name
            if cand.is_dir():
                algo_outputs_dir = cand
                break
        else:
            algo_outputs_dir = snapshot_dir / "algo_outputs"

    video_ids = manifest.get("video_ids")
    if video_ids is None:
        video_ids = sorted(
            p.stem.replace("_unified_ground_truth", "")
            for p in gt_dir.glob("*_unified_ground_truth.json"))

    metrics_dir = snapshot_dir / "metrics"
    figures_dir = snapshot_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return SnapshotPaths(
        snapshot_dir=snapshot_dir,
        gt_dir=gt_dir,
        algo_outputs_dir=algo_outputs_dir,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        video_ids=list(video_ids),
    )


# ---------------------------------------------------------------------------
# GT loaders
# ---------------------------------------------------------------------------

def gt_path(gt_dir: Path, video_id: str) -> Path:
    return Path(gt_dir) / f"{video_id}_unified_ground_truth.json"


def load_gt_video(gt_dir: Path, video_id: str) -> dict:
    """Load the full unified-GT JSON for a video."""
    p = gt_path(gt_dir, video_id)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_gt_segments(gt_dir: Path, video_id: str) -> Tuple[List[dict], bool]:
    """Return (per-segment GT rows, exhaustive_flag)."""
    gt = load_gt_video(gt_dir, video_id)
    block = gt.get("outcomes", {}) or {}
    segs = block.get("segments", []) or []
    exhaustive = bool(block.get("exhaustive", False))
    return segs, exhaustive


def load_gt_reaches(gt_dir: Path, video_id: str) -> List[dict]:
    gt = load_gt_video(gt_dir, video_id)
    return (gt.get("reaches", {}) or {}).get("reaches", []) or []


def load_gt_boundaries(gt_dir: Path, video_id: str) -> List[int]:
    """GT segment boundaries if present (segmentation eval).

    Handles two schemas:
      - Legacy: ``boundaries: [int, int, ...]`` (frame numbers)
      - Unified GT v2.0: ``boundaries: [{"index": i, "frame": F, "determined": bool,
        ...}, ...]``
    Returns a flat list of integer frame numbers either way.
    """
    gt = load_gt_video(gt_dir, video_id)
    block = gt.get("segmentation", {}) or {}
    raw = list(block.get("boundaries", []) or [])
    out: List[int] = []
    for b in raw:
        if isinstance(b, dict):
            f = b.get("frame")
            if f is not None:
                out.append(int(f))
        else:
            out.append(int(b))
    return out


# ---------------------------------------------------------------------------
# Algo loaders
# ---------------------------------------------------------------------------

def load_algo_segments(algo_dir: Path, video_id: str) -> dict:
    """Load the segments JSON. Returns the full dict."""
    p = Path(algo_dir) / f"{video_id}_segments.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_algo_boundaries(algo_dir: Path, video_id: str) -> List[int]:
    return list(load_algo_segments(algo_dir, video_id).get("boundaries", []) or [])


def load_algo_reaches(algo_dir: Path, video_id: str) -> dict:
    p = Path(algo_dir) / f"{video_id}_reaches.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_algo_outcomes(algo_dir: Path, video_id: str) -> dict:
    p = Path(algo_dir) / f"{video_id}_pellet_outcomes.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def write_scalars(metrics_dir: Path, scalars: dict, name: str = "scalars.json") -> Path:
    out = Path(metrics_dir) / name
    out.write_text(json.dumps(scalars, indent=2), encoding="utf-8")
    return out


def read_scalars(metrics_dir: Path, name: str = "scalars.json") -> dict:
    p = Path(metrics_dir) / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
