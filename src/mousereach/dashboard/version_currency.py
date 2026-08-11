"""Is each video CURRENT with the algorithm versions we ship right now?

The dashboard's most important signal: for every video, do the algo versions
recorded in its processing manifest match the versions declared in the shipped
``pipeline_versions.json``? If not, the video is out of date and needs
reprocessing to become current -- which the GUI can then trigger.

This module builds a stem -> manifest index across the pipeline roots once (the
manifest scan is the slow part, so it is cached by the caller) and answers the
per-video currency question via ``compare_manifest_to_current``.

ASCII-only console output (Windows cp1252).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def build_manifest_index(roots) -> Dict[str, Path]:
    """Map ``{video_stem: manifest_path}`` by scanning ``roots`` for
    ``*_processing_manifest.json`` (most-recently-modified wins on duplicates).
    This walks the archive tree, so it is the expensive step -- cache it."""
    idx: Dict[str, Path] = {}
    for r in roots:
        if not r:
            continue
        r = Path(r)
        if not r.exists():
            continue
        try:
            for p in r.rglob("*_processing_manifest.json"):
                stem = p.name.replace("_processing_manifest.json", "")
                try:
                    if stem not in idx or p.stat().st_mtime > idx[stem].stat().st_mtime:
                        idx[stem] = p
                except OSError:
                    idx.setdefault(stem, p)
        except Exception as e:
            logger.debug(f"manifest scan of {r} failed: {e}")
    return idx


def _load(mp: Optional[Path]) -> Optional[dict]:
    if not mp:
        return None
    try:
        return json.loads(Path(mp).read_text(encoding="utf-8"))
    except Exception:
        return None


def version_status(video_id: str, manifest_index: Dict[str, Path],
                   current_versions: Optional[dict]) -> str:
    """Return 'current', 'outdated', or 'unknown' for one video.

    'unknown' when there is no shipped-versions file or no manifest for the video
    (so we cannot judge)."""
    if not current_versions or not current_versions.get("versions"):
        return "unknown"
    manifest = _load(manifest_index.get(video_id))
    if manifest is None:
        return "unknown"
    from mousereach.pipeline.versions import compare_manifest_to_current
    cmp = compare_manifest_to_current(manifest, current_versions)
    return "current" if cmp.get("is_current") else "outdated"


def build_version_maps(video_ids, manifest_index: Dict[str, Path],
                       current_versions: Optional[dict], max_workers: int = 16):
    """Pre-compute ``({video_id: status}, {video_id: dlc_scorer})`` in ONE pass.

    Why this exists: the per-video helpers above each re-open and re-parse the
    video's manifest off the NAS (~22 ms a call). The dashboard asks BOTH
    questions for every table row, so a 3,776-row repaint cost ~168 s of blocking
    GUI work -- and every later repaint (refresh, filter, sort) paid it again,
    which read as a permanent freeze. Loading each manifest exactly once here,
    off the GUI thread, turns those per-row questions into dict lookups.

    The loads are I/O-bound (NAS round-trips, not CPU), so a small thread pool
    collapses the wall-clock cost roughly by ``max_workers``.

    Returns maps keyed by video id; ids with no manifest are simply absent, and
    callers fall back to 'unknown' / None.
    """
    from concurrent.futures import ThreadPoolExecutor

    from mousereach.pipeline.versions import compare_manifest_to_current

    have_versions = bool(current_versions and current_versions.get("versions"))
    wanted = [v for v in video_ids if v in manifest_index]

    def one(vid):
        manifest = _load(manifest_index.get(vid))
        if manifest is None:
            return vid, None, None
        status = None
        if have_versions:
            cmp = compare_manifest_to_current(manifest, current_versions)
            status = "current" if cmp.get("is_current") else "outdated"
        scorer = (manifest.get("dlc_model") or {}).get("dlc_scorer") or None
        return vid, status, scorer

    status_map: Dict[str, str] = {}
    dlc_map: Dict[str, str] = {}
    if not wanted:
        return status_map, dlc_map
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for vid, status, scorer in pool.map(one, wanted):
                if status:
                    status_map[vid] = status
                if scorer:
                    dlc_map[vid] = scorer
    except Exception as e:
        logger.debug(f"version map build failed: {e}")
    return status_map, dlc_map


def outdated_components(video_id: str, manifest_index: Dict[str, Path],
                        current_versions: Optional[dict]) -> List[str]:
    """The stale component names for a video (empty if current/unknown)."""
    if not current_versions:
        return []
    manifest = _load(manifest_index.get(video_id))
    if manifest is None:
        return []
    from mousereach.pipeline.versions import compare_manifest_to_current
    return list(compare_manifest_to_current(manifest, current_versions).get("stale_components", []))
