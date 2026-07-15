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
