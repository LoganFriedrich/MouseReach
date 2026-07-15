"""Backfill the watcher database from the archive.

The watcher database only knows about videos it has processed itself. Videos
archived before the database existed (or on other machines) are on the NAS but
invisible to the dashboard. This one-time, idempotent backfill registers every
archived video into the database as ``archived`` so:

  * the dashboard shows the WHOLE corpus (not just what this node ran), and
  * "Reprocess outdated" reaches every video (the reprocessing scanner acts on
    ``archived`` videos).

It records nothing about versions -- the dashboard's Version column reads each
video's manifest directly -- so backfilling is safe and repeatable: already-known
videos are skipped.

ASCII-only console output (Windows cp1252).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

_SUFFIX = "_reaches.json"  # a reliable per-video marker in the archive


def _parse_stem(stem: str) -> Dict[str, str]:
    """Pull the light metadata the dashboard groups by from a video stem like
    ``20250624_CNT0102_P1`` -> date / animal_id / tray_type."""
    meta: Dict[str, str] = {}
    parts = stem.split("_")
    if len(parts) >= 1 and parts[0]:
        meta["date"] = parts[0]
    if len(parts) >= 2 and parts[1]:
        meta["animal_id"] = parts[1]
    if len(parts) >= 3 and parts[2]:
        meta["tray_type"] = parts[2][0]  # 'P' from 'P1'
    return meta


def backfill_archive(db, archive_root, progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, int]:
    """Register every archived video into ``db`` as 'archived'. Idempotent.

    Returns ``{'new': n, 'existing': n, 'errors': n}``. ``progress(done, total)``
    is called periodically (total is an estimate == number of videos found)."""
    archive_root = Path(archive_root)
    result = {"new": 0, "existing": 0, "errors": 0}
    if not archive_root.exists():
        return result

    reaches = list(archive_root.rglob(f"*{_SUFFIX}"))
    total = len(reaches)
    seen = set()
    for i, rf in enumerate(reaches):
        stem = rf.name[: -len(_SUFFIX)]
        if stem in seen:
            continue
        seen.add(stem)
        try:
            if db.get_video(stem):
                result["existing"] += 1
            else:
                meta = _parse_stem(stem)
                mp4 = rf.parent / f"{stem}.mp4"
                source = str(mp4 if mp4.exists() else rf.parent / f"{stem}.mp4")
                db.register_video(
                    stem, source,
                    state="archived", archived_at=db._now(), **meta,
                )
                result["new"] += 1
        except Exception as e:
            result["errors"] += 1
            logger.debug(f"backfill {stem}: {e}")
        if progress and (i % 100 == 0):
            progress(i + 1, total)
    if progress:
        progress(total, total)
    logger.info(
        f"Archive backfill: {result['new']} new, {result['existing']} already "
        f"known, {result['errors']} errors"
    )
    return result
