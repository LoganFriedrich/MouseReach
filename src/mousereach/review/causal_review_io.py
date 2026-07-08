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
    manual_segmentation: Optional[Dict[str, Any]] = None,
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
    if manual_segmentation:
        doc["manual_segmentation"] = manual_segmentation

    out_path = output_dir / f"{video_stem}_causal_review.json"
    _write_json(out_path, doc)
    return out_path


def load_causal_review(video_stem: str, output_dir: Path):
    """Load a saved per-video causal review, if present.

    Returns ``(doc, {segment_num: record})``; ``(None, {})`` if no file. The
    per-segment records are exactly what ``_collect_answers`` produced, so they
    can be dropped straight back into the widget's ``_review_records`` to restore
    a prior review session (resume + re-populate the panel).
    """
    path = Path(output_dir) / f"{video_stem}_causal_review.json"
    if not path.exists():
        return None, {}
    doc = _read_json(path)
    by_seg: Dict[int, Any] = {}
    for rec in doc.get("segments", []):
        sn = rec.get("segment_num")
        if sn is not None:
            by_seg[int(sn)] = rec
    return doc, by_seg


# ---------------------------------------------------------------------------
# Review -> outcomes bridge: apply a reviewer's corrections onto the algo's
# per-segment outcomes so a correction made in the Causal Review tool actually
# flows FORWARD into kinematics (which reads outcome + causal_reach_id from the
# outcomes). This is the missing link between review and the data product.
# ---------------------------------------------------------------------------

_TOUCHED_OUTCOMES = ("retrieved", "displaced_sa", "displaced_outside")


def apply_review_overrides(
    outcomes_data: Dict[str, Any],
    review_by_seg: Dict[int, Dict[str, Any]],
    reviewer: Optional[str] = None,
) -> Dict[str, Any]:
    """Override the algo's per-segment ``outcome`` + ``causal_reach_id`` with the
    human review, for segments that were actually reviewed.

    NON-DESTRUCTIVE: returns a deep copy. The algo's originals are preserved on
    each segment as ``algo_outcome`` / ``algo_causal_reach_id``; provenance is
    recorded as ``outcome_source`` ("human_review" or "algo").

    - A touched human outcome (retrieved / displaced_sa / displaced_outside)
      sets ``causal_reach_id`` to the human's causal-reach id.
    - untouched / abnormal_exception carry NO causal reach (``causal_reach_id``
      = None), so kinematics excludes them from causal-reach kinematics.
    - abnormal ignore-windows are carried onto the segment as ``abnormal_ranges``.
    - Unreviewed segments (or ``reviewed: False`` placeholders) are untouched.
    """
    import copy
    out = copy.deepcopy(outcomes_data)
    by_seg = {int(k): v for k, v in (review_by_seg or {}).items()}
    for seg in out.get("segments", []):
        seg["outcome_source"] = "algo"
        sn = seg.get("segment_num")
        if sn is None:
            continue
        rec = by_seg.get(int(sn))
        if not rec:
            continue
        ans = rec.get("answers") or {}
        if ans.get("reviewed") is False:
            continue
        human = rec.get("human") or {}
        ho = human.get("outcome")
        if ho is None:
            continue
        hc = human.get("causal_reach") or {}
        seg["algo_outcome"] = seg.get("outcome")
        seg["algo_causal_reach_id"] = seg.get("causal_reach_id")
        seg["outcome"] = ho
        seg["causal_reach_id"] = hc.get("reach_id") if ho in _TOUCHED_OUTCOMES else None
        seg["outcome_source"] = "human_review"
        if reviewer:
            seg["reviewed_by"] = reviewer
        if ans.get("abnormal_ranges"):
            seg["abnormal_ranges"] = ans["abnormal_ranges"]
    return out


def load_and_apply_review(
    outcomes_data: Dict[str, Any],
    review_path: Path,
    video_stem: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a causal-review file and apply its corrections onto ``outcomes_data``.

    ``review_path`` may be the review JSON itself OR a directory containing
    ``{video_stem}_causal_review.json``. Returns ``outcomes_data`` unchanged if
    no review is found (safe no-op)."""
    review_path = Path(review_path)
    doc = None
    if review_path.is_file():
        doc = _read_json(review_path)
    elif review_path.is_dir():
        stem = video_stem or outcomes_data.get("video_name") or ""
        cand = review_path / f"{stem}_causal_review.json"
        if cand.exists():
            doc = _read_json(cand)
    if not doc:
        return outcomes_data
    by_seg = {}
    for rec in doc.get("segments", []):
        sn = rec.get("segment_num")
        if sn is not None:
            by_seg[int(sn)] = rec
    return apply_review_overrides(outcomes_data, by_seg, reviewer=doc.get("reviewer"))


# ---------------------------------------------------------------------------
# Ground-truth detection -- a video that's already GT'd shouldn't be reviewed;
# we already know the answer. GT files are chained to a video by stem and may
# live anywhere (improvement snapshots, Processing, next to the video, or in the
# bundle), so we index them by stem across the known roots.
# ---------------------------------------------------------------------------

GT_INDEX_ROOTS = [
    Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement"),
    Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing"),
]

_GT_INDEX_CACHE: Optional[Dict[str, Path]] = None


def build_gt_index(roots=None) -> Dict[str, Path]:
    """Scan the GT roots for ``*_unified_ground_truth.json`` -> {stem: path}
    (most-recently-modified wins on duplicates)."""
    roots = roots or GT_INDEX_ROOTS
    idx: Dict[str, Path] = {}
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        for p in r.rglob("*_unified_ground_truth.json"):
            stem = p.name.split("_unified_ground_truth.json")[0]
            try:
                if stem not in idx or p.stat().st_mtime > idx[stem].stat().st_mtime:
                    idx[stem] = p
            except OSError:
                idx.setdefault(stem, p)
    return idx


def gt_index(refresh: bool = False) -> Dict[str, Path]:
    """Cached stem -> unified-GT-path index (built once per process)."""
    global _GT_INDEX_CACHE
    if _GT_INDEX_CACHE is None or refresh:
        _GT_INDEX_CACHE = build_gt_index()
    return _GT_INDEX_CACHE


def find_gt(video_stem: str, extra_dirs=()) -> Optional[Path]:
    """Path to the video's unified GT if it exists ANYWHERE, else None.

    Checks co-located dirs first (bundle / next-to-video -- "chained at the hip"),
    then the corpus-wide GT index.
    """
    for d in extra_dirs:
        p = Path(d) / f"{video_stem}_unified_ground_truth.json"
        if p.exists():
            return p
    return gt_index().get(video_stem)


def has_gt(video_stem: str, extra_dirs=()) -> bool:
    return find_gt(video_stem, extra_dirs) is not None


# ---------------------------------------------------------------------------
# Session flags -- "this mouse+day has a physical issue; review ALL its videos"
# ---------------------------------------------------------------------------

def session_key(video_stem: str) -> str:
    """Group key for videos of the same mouse on the same day: {date}_{mouse}.

    e.g. ``20250624_CNT0101_P1`` -> ``20250624_CNT0101``. A session-level issue
    (a cage artifact that day, a misplaced pellet, etc.) affects every ``P#`` of
    that mouse+day, so they should all get human review.
    """
    parts = str(video_stem).split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else str(video_stem)


def _flagged_sessions_path(review_root: Path) -> Path:
    return Path(review_root) / "flagged_sessions.json"


def load_flagged_sessions(review_root: Path) -> Dict[str, Any]:
    """Return {session_key: flag_record} for the review corpus."""
    p = _flagged_sessions_path(review_root)
    if p.exists():
        return _read_json(p).get("sessions", {})
    return {}


def is_session_flagged(video_stem: str, review_root: Path) -> bool:
    return session_key(video_stem) in load_flagged_sessions(review_root)


def flag_session(video_stem: str, review_root: Path, reason: str = "",
                 flagged_by: Optional[str] = None) -> str:
    """Flag the video's whole session (mouse+day) as must-be-human-reviewed.

    Idempotent: re-flagging the same session updates the record. Returns the
    session key. Videos already GT'd/reviewed are still honored -- the flag only
    forces review of the ones that aren't yet resolved.
    """
    key = session_key(video_stem)
    p = _flagged_sessions_path(review_root)
    doc = _read_json(p) if p.exists() else {}
    doc.setdefault("type", "flagged_sessions")
    doc.setdefault("schema_version", "1.0")
    sessions = doc.setdefault("sessions", {})
    sessions[key] = {
        "session_key": key,
        "reason": reason,
        "flagged_by": flagged_by or _get_username(),
        "flagged_at": _get_timestamp(),
        "requires_human_review": True,
    }
    doc["last_updated"] = _get_timestamp()
    _write_json(p, doc)
    return key


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
