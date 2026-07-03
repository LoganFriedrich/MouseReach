"""
Causal Review I/O -- save format and corpus index for the causal review tool.

Two outputs per reviewed video:
  1. Per-video ``{video}_causal_review.json`` -- standalone review record
  2. Corpus index (append/update) at
     ``{NAS_ROOT}/review_records/causal_review_index.json``

The per-video file is the primary artifact. The corpus index is a
lookup table so the active-learning loop can bulk-read all reviews
without scanning every video directory.

The save is idempotent: re-reviewing a (video, segment) pair
overwrites the prior entry in both per-video file and index.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_username() -> str:
    return os.environ.get("USERNAME", os.environ.get("USER", "unknown"))


def _get_timestamp() -> str:
    return datetime.now().isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def collect_provenance(video_dir: Path, video_stem: str) -> Dict[str, Any]:
    """Read version stamps from the algo JSON files next to the video.

    Pulls ``version`` / ``detector`` from _reaches.json,
    _pellet_outcomes.json, _segments.json, and _reach_assignments.json
    when they exist. Also reads pipeline_versions.json if present.
    """
    prov: Dict[str, Any] = {}

    for suffix, key in [
        ("_segments.json", "segmenter"),
        ("_reaches.json", "reach_detector"),
        ("_pellet_outcomes.json", "outcome_detector"),
        ("_reach_assignments.json", "assignment"),
    ]:
        path = video_dir / f"{video_stem}{suffix}"
        if not path.exists():
            # Try _segmentation.json as alternate
            if suffix == "_segments.json":
                path = video_dir / f"{video_stem}_segmentation.json"
            if not path.exists():
                continue
        try:
            data = _read_json(path)
            entry: Dict[str, Any] = {}
            if "version" in data:
                entry["version"] = data["version"]
            if "detector" in data:
                entry["detector"] = data["detector"]
            if "segmenter_version" in data:
                entry["version"] = data["segmenter_version"]
            prov[key] = entry
        except Exception:
            pass

    # DLC model/scorer from HDF5 column header (best-effort)
    try:
        import pandas as pd
        h5_files = list(video_dir.glob(f"{video_stem}*.h5"))
        if h5_files:
            df = pd.read_hdf(h5_files[0], stop=0)
            if hasattr(df.columns, "levels"):
                scorer = df.columns.get_level_values(0)[0]
                prov["dlc_scorer"] = scorer
    except Exception:
        pass

    return prov


# ---------------------------------------------------------------------------
# Per-segment review record
# ---------------------------------------------------------------------------

def build_segment_record(
    segment_num: int,
    pellet_num: Optional[int],
    algo_outcome: Optional[str],
    algo_causal_reach: Optional[Dict[str, int]],
    algo_interaction_frame: Optional[int],
    human_outcome: Optional[str],
    human_causal_reach: Optional[Dict[str, int]],
    is_phantom: bool,
    agreed: bool,
    answers: Dict[str, Any],
    notes: str = "",
) -> Dict[str, Any]:
    """Build a single segment's review record."""
    rec: Dict[str, Any] = {
        "segment_num": segment_num,
        "pellet_num": pellet_num,
        "algo": {
            "outcome": algo_outcome,
            "causal_reach": algo_causal_reach,
            "interaction_frame": algo_interaction_frame,
        },
        "human": {
            "outcome": human_outcome,
            "causal_reach": human_causal_reach,
            "is_phantom": is_phantom,
            "agreed": agreed,
        },
        "answers": answers,
        "notes": notes,
    }
    return rec


# ---------------------------------------------------------------------------
# Save per-video review file
# ---------------------------------------------------------------------------

def save_causal_review(
    video_stem: str,
    output_dir: Path,
    segments: List[Dict[str, Any]],
    provenance: Dict[str, Any],
    reviewer: Optional[str] = None,
) -> Path:
    """Write (or overwrite) the per-video causal review JSON.

    Parameters
    ----------
    video_stem : str
        Video identifier (without extension).
    output_dir : Path
        Directory to write the review file into.
    segments : list of dict
        Per-segment review records (from ``build_segment_record``).
    provenance : dict
        Algo version stamps (from ``collect_provenance``).
    reviewer : str, optional
        Reviewer name; defaults to OS username.

    Returns
    -------
    Path to the written file.
    """
    reviewer = reviewer or _get_username()
    timestamp = _get_timestamp()

    doc: Dict[str, Any] = {
        "type": "causal_review",
        "schema_version": "1.0",
        "video_stem": video_stem,
        "reviewer": reviewer,
        "reviewed_at": timestamp,
        "provenance": provenance,
        "segments": segments,
    }

    out_path = output_dir / f"{video_stem}_causal_review.json"
    _write_json(out_path, doc)
    return out_path


# ---------------------------------------------------------------------------
# Corpus index
# ---------------------------------------------------------------------------

def _default_index_dir() -> Optional[Path]:
    """Resolve the corpus index directory.

    Prefers NAS_ROOT/review_records/ if NAS is configured,
    otherwise falls back to PROCESSING_ROOT/review_records/.
    """
    try:
        from mousereach.config import Paths
        if Paths.NAS_ROOT is not None:
            return Paths.NAS_ROOT / "review_records"
        if Paths.PROCESSING_ROOT is not None:
            return Paths.PROCESSING_ROOT / "review_records"
    except Exception:
        pass
    return None


def update_corpus_index(
    video_stem: str,
    review_file_path: Path,
    segments: List[Dict[str, Any]],
    reviewer: str,
    reviewed_at: str,
    index_dir: Optional[Path] = None,
) -> Path:
    """Append or update corpus-level index entries for a reviewed video.

    The index is a JSON file mapping ``(video_stem, segment_num)`` pairs
    to their review file path plus key summary fields. Re-reviewing a
    segment overwrites the prior entry (idempotent).

    Parameters
    ----------
    video_stem : str
        Video identifier.
    review_file_path : Path
        Absolute path to the per-video review JSON.
    segments : list of dict
        Per-segment review records.
    reviewer : str
        Who reviewed.
    reviewed_at : str
        ISO timestamp.
    index_dir : Path, optional
        Override index directory. Defaults to NAS/review_records/.

    Returns
    -------
    Path to the index file.
    """
    if index_dir is None:
        index_dir = _default_index_dir()
    if index_dir is None:
        # No NAS or PROCESSING configured -- skip silently
        return Path("(no index directory configured)")

    index_path = index_dir / "causal_review_index.json"
    index = _read_json(index_path)

    # Ensure top-level structure
    if "type" not in index:
        index["type"] = "causal_review_index"
        index["schema_version"] = "1.0"
    if "entries" not in index:
        index["entries"] = {}

    entries = index["entries"]

    for seg in segments:
        seg_num = seg.get("segment_num")
        key = f"{video_stem}__seg{seg_num}"
        entries[key] = {
            "video_stem": video_stem,
            "segment_num": seg_num,
            "pellet_num": seg.get("pellet_num"),
            "review_file": str(review_file_path),
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "human_outcome": seg.get("human", {}).get("outcome"),
            "agreed": seg.get("human", {}).get("agreed", False),
        }

    index["last_updated"] = _get_timestamp()
    _write_json(index_path, index)
    return index_path
