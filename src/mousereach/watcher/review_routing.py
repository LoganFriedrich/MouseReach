"""Whole-video bundle routing for the human-review holds.

A video the algo cannot commit -- segmentation FAILED, or ANY element was left
triaged / flagged -- is held OUT of the archive and connectome.db until a human
clears it. This module MOVES the whole video bundle (the mp4, the DLC pose h5,
and every algo output JSON) out of the local Processing dir into a
self-contained review bundle under the Triage or Deep_Review queue on the NAS,
so that:

  * the video travels WITH its outputs -- the review tool opens it in place,
  * the Processing dir is left clean (no video stalls there forever), and
  * kinematics + DB sync never run until the bundle comes back through the gate.

Every move is recorded in a ``{stem}_routing.json`` manifest (source,
destination, reason, moved files, timestamp) for a full audit trail. The move
is idempotent: re-routing a video that already has a bundle replaces it.

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Files that belong to a video bundle. The DLC pose h5 is named
# ``{stem}DLC_resnet...h5`` (next char after the stem is 'D'); the mp4 is
# ``{stem}.mp4`` (next char '.'); every algo output is ``{stem}_*.json`` (next
# char '_'). Requiring the next char to be one of these prevents a stem that is
# a strict prefix of another (CNT0312_P2 vs CNT0312_P21) from stealing files.
_BOUNDARY_CHARS = (".", "_", "D")


def _bundle_files(source_dir: Path, video_id: str) -> List[Path]:
    """Every file in ``source_dir`` that belongs to ``video_id``.

    Uses a prefix-boundary guard so ``CNT0312_P2`` does not also match
    ``CNT0312_P21``'s files.
    """
    out: List[Path] = []
    if not source_dir.exists():
        return out
    n = len(video_id)
    for f in source_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if not name.startswith(video_id):
            continue
        # exact stem match (name == stem, unlikely) or a valid boundary char
        if len(name) == n or name[n] in _BOUNDARY_CHARS:
            out.append(f)
    return out


def _safe_move(src: Path, dst: Path) -> None:
    """shutil.move that overwrites an existing destination file.

    Cross-filesystem moves (C: Processing -> Y: NAS) are copy+delete under the
    hood; that is intended -- the bundle must live on the NAS so any node / the
    GUI can review it.
    """
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def move_video_bundle(
    video_id: str,
    source_dir: Path,
    dest_root: Path,
    reason: str,
    *,
    extra_sources: Optional[Iterable[Path]] = None,
    extra_manifest: Optional[Dict] = None,
) -> Tuple[Path, List[str]]:
    """Move a whole video bundle from ``source_dir`` into ``dest_root/{stem}/``.

    Parameters
    ----------
    video_id : str
        The video stem (e.g. ``20250716_CNT0213_P3``).
    source_dir : Path
        The Processing dir the bundle currently lives in.
    dest_root : Path
        The queue root (``Paths.TRIAGE_REVIEW`` or ``Paths.DEEP_REVIEW``). The
        per-video bundle folder ``dest_root/{stem}/`` is created.
    reason : str
        Human-readable routing reason, recorded in the manifest and shown in the
        review tool (e.g. ``"segmentation_failed"``, ``"outcome triaged: 2 seg"``).
    extra_sources : iterable of Path, optional
        Additional files to pull into the bundle that do NOT live in
        ``source_dir`` (e.g. the canonical mp4 if it was not co-located). Missing
        files are skipped silently.
    extra_manifest : dict, optional
        Extra key/values merged into the routing manifest (e.g. triaged element
        detail, DB state, node hostname).

    Returns
    -------
    (bundle_dir, moved_file_names)
    """
    source_dir = Path(source_dir)
    dest_root = Path(dest_root)
    bundle = dest_root / video_id
    bundle.mkdir(parents=True, exist_ok=True)

    moved: List[str] = []
    for f in _bundle_files(source_dir, video_id):
        try:
            _safe_move(f, bundle / f.name)
            moved.append(f.name)
        except Exception as e:
            logger.warning(f"Routing {video_id}: could not move {f.name}: {e}")

    for f in extra_sources or []:
        f = Path(f)
        if not f.exists():
            continue
        target = bundle / f.name
        if target.exists():
            continue  # already brought in by the main sweep
        try:
            _safe_move(f, target)
            moved.append(f.name)
        except Exception as e:
            logger.warning(f"Routing {video_id}: could not move extra {f.name}: {e}")

    manifest = {
        "video_id": video_id,
        "routed_reason": reason,
        "source_dir": str(source_dir),
        "bundle_dir": str(bundle),
        "moved_files": sorted(moved),
        "routed_at": datetime.now().isoformat(),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (bundle / f"{video_id}_routing.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.info(f"Routed {video_id} -> {bundle} ({len(moved)} files) reason={reason}")
    return bundle, moved


def read_routing_manifest(bundle_dir: Path, video_id: str) -> Optional[Dict]:
    """Read a bundle's ``{stem}_routing.json`` manifest, or None if absent."""
    p = Path(bundle_dir) / f"{video_id}_routing.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
