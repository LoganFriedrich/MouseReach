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
destination, reason, moved files, files that failed to move, timestamp) for a
full audit trail. The move is idempotent: re-routing a video that already has
a bundle replaces it.

A NEW bundle is never visible half-built. It is assembled in the hidden
``dest_root/.incoming/{stem}/`` and published with ONE directory rename, so a
queue reader sees either no bundle or the whole bundle. WHY: the move used to
create the empty ``{stem}`` folder first and then fill it file by file
(cross-volume copies take seconds); the running watcher's return scan saw the
empty folder, retired it to _Problematic mid-route, and every remaining move
failed with "No such file or directory" (2026-09-14).

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

# Staging folder, directly under a queue root, where a NEW bundle is assembled
# before it is published into place by one rename. WHY a dot-folder: the queue
# readers that judge, retire or release bundles skip names starting with "."
# (review_return._bundles, census/review_completeness, dashboard/folder_scan,
# review/release_cli, review/queue_index, watcher/reconcile), so a bundle that
# is still being filled can be neither retired as an empty dir nor judged as a
# bundle. Other readers of the queue roots (collage_provenance,
# reprocess_to_current's skip list, two review widgets) do NOT filter dot-
# folders; they key on stem-named files or folder names, so ".incoming" matches
# no video there by accident, not by rule -- a new reader must add the filter.
# WHY under the same queue root: the final rename must stay on one volume to be
# a single atomic directory rename. Other modules import this name -- do not
# rename it without updating them.
INCOMING_DIR_NAME = ".incoming"


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

    Retries transient Windows sharing violations (WinError 32/5): the pipeline
    routes a bundle seconds after its own readers finish with the mp4, and the
    handle is sometimes still closing. Without the retry the mp4 silently
    stayed behind in Processing while the rest of the bundle moved -- 81 such
    part-moved bundles on 2026-09-09 alone. A genuinely stuck file still
    raises to the caller after the retries.
    """
    from mousereach.pipeline.fsutil import retry_transient

    def _mv():
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))

    retry_transient(_mv, what=src.name)


def _make_incoming(build: Path) -> None:
    """Create ``dest_root/.incoming/{stem}/``; reuse it if it already exists.

    An existing folder is residue of an earlier route that was interrupted
    mid-build: its files are part of THIS video's bundle, so they are kept and
    published with it rather than failing the route. Retries a vanished parent
    because a concurrent route finishing in the same queue removes an empty
    ``.incoming`` folder, and that can land between Path.mkdir creating the
    parent and creating the leaf.
    """
    for attempt in range(3):
        try:
            build.mkdir(parents=True, exist_ok=True)
            return
        except FileNotFoundError:
            if attempt == 2:
                raise


def _write_routing_manifest(folder: Path, video_id: str, manifest: Dict) -> None:
    # WHY the retrying writer: this lands on the share, where a transient hold
    # on the file is routine; a bare write_text raised straight out of the
    # route after every file had already moved.
    from mousereach.pipeline.fsutil import dump_json_with_retry
    dump_json_with_retry(folder / f"{video_id}_routing.json", manifest, indent=2)


def _publish_incoming(
    build: Path, bundle: Path, video_id: str, manifest: Dict,
    manifest_written: bool = True,
) -> None:
    """Move the assembled ``build`` folder into place as ``bundle``.

    One directory rename (same volume, so atomic): readers see no bundle or
    the whole bundle. If the rename fails -- normally because ``bundle``
    appeared meanwhile (a concurrent route of the same stem), otherwise a lock
    that outlived the retries -- fall back to moving each built file into
    ``bundle`` with the usual overwrite semantics. WHY not raise: the video has
    already left Processing; files stranded in the hidden staging folder would
    be invisible to every queue reader, a silent loss. The fallback moves are
    same-volume renames, so the visible half-built window is milliseconds, not
    the seconds of a cross-volume copy. The routing manifest moves LAST, updated
    with any file that failed here, so the published record names the husk.

    ``manifest_written`` False means the manifest could not be written into
    ``build``; it is then written into the published ``bundle`` (a failure
    there raises, with the bundle already visible -- the pre-staging
    behaviour).
    """
    from mousereach.pipeline.fsutil import retry_transient

    try:
        retry_transient(lambda: build.rename(bundle), what=build.name)
    except OSError as e:
        if not bundle.is_dir():
            logger.warning(
                f"Routing {video_id}: could not publish {build} as {bundle} "
                f"({e}); moving files in one by one")
            bundle.mkdir(parents=True, exist_ok=True)
        manifest_name = f"{video_id}_routing.json"
        try:
            built = sorted(build.iterdir())
        except OSError:
            # A concurrent route of the same stem published the shared
            # staging folder between the failed rename and this listing.
            built = []
        for f in built:
            if f.name == manifest_name or not f.is_file():
                continue
            try:
                _safe_move(f, bundle / f.name)
            except Exception as e2:
                logger.warning(
                    f"Routing {video_id}: could not merge {f.name} into "
                    f"{bundle}: {e2}")
                manifest["failed_files"].append(
                    {"name": f.name, "error": str(e2)})
        _write_routing_manifest(bundle, video_id, manifest)
        try:
            (build / manifest_name).unlink()
        except OSError:
            pass
        # Only an EMPTY staging folder is removed; a file that failed to merge
        # stays where it is (never deleted) and is reused by the next route.
        try:
            build.rmdir()
        except OSError:
            pass
    else:
        # Outside the try on purpose: a failed write here must raise, not be
        # mistaken for a failed rename and sent down the merge fallback.
        if not manifest_written:
            _write_routing_manifest(bundle, video_id, manifest)
    # Remove ``.incoming`` itself only when empty -- another route may be
    # building its own bundle there right now.
    try:
        build.parent.rmdir()
    except OSError:
        pass


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
        ``bundle_dir`` is always the FINAL ``dest_root/{stem}/`` folder, never
        the staging folder, so callers can write into it straight away.

    A new bundle is built in ``dest_root/.incoming/{stem}/`` and published by
    one rename (see the module docstring for why). A bundle that ALREADY exists
    (idempotent re-route, or a queue-to-queue divert into an existing bundle)
    is already visible, so files are moved straight into it as before.
    """
    source_dir = Path(source_dir)
    dest_root = Path(dest_root)
    bundle = dest_root / video_id
    staged = not bundle.is_dir()
    if staged:
        build = dest_root / INCOMING_DIR_NAME / video_id
        _make_incoming(build)
    else:
        build = bundle

    def _published_by_other_route() -> bool:
        # Two routes of the same stem share ``.incoming/{stem}``. When the
        # other route renames it into place while this one is still filling
        # it, this route's remaining files -- and its record -- belong in that
        # now-visible bundle. Without this every later move failed and the
        # manifest write raised out of the route, losing its record.
        return staged and not build.is_dir() and bundle.is_dir()

    def _move_in(src: Path) -> None:
        try:
            _safe_move(src, build / src.name)
        except OSError:
            if not _published_by_other_route():
                raise
            _safe_move(src, bundle / src.name)

    moved: List[str] = []
    # Recorded in the manifest, not only logged: a part-moved bundle (a husk)
    # must be visible in the record the review tool and audits read.
    failed: List[Dict[str, str]] = []
    for f in _bundle_files(source_dir, video_id):
        try:
            _move_in(f)
            moved.append(f.name)
        except Exception as e:
            logger.warning(f"Routing {video_id}: could not move {f.name}: {e}")
            failed.append({"name": f.name, "error": str(e)})

    for f in extra_sources or []:
        f = Path(f)
        if not f.exists():
            continue
        if (build / f.name).exists():
            continue  # already brought in by the main sweep
        try:
            _move_in(f)
            moved.append(f.name)
        except Exception as e:
            logger.warning(f"Routing {video_id}: could not move extra {f.name}: {e}")
            failed.append({"name": f.name, "error": str(e)})

    manifest = {
        "video_id": video_id,
        "routed_reason": reason,
        "source_dir": str(source_dir),
        "bundle_dir": str(bundle),
        "moved_files": sorted(moved),
        "failed_files": failed,
        "routed_at": datetime.now().isoformat(),
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    if not staged:
        _write_routing_manifest(bundle, video_id, manifest)
    elif _published_by_other_route():
        _write_routing_manifest(bundle, video_id, manifest)
        try:
            build.parent.rmdir()
        except OSError:
            pass
    else:
        # Once files have moved, the publish is ALWAYS attempted. WHY: an
        # exception here used to skip it, stranding every file -- the video's
        # only copy, already gone from Processing -- in the hidden staging
        # folder that no queue reader lists, while the caller marked the video
        # failed with nothing left to retry.
        manifest_written = True
        try:
            _write_routing_manifest(build, video_id, manifest)
        except Exception as e:
            logger.warning(
                f"Routing {video_id}: could not write the routing manifest in "
                f"{build} ({e}); publishing the bundle first, then writing it")
            manifest_written = False
        _publish_incoming(build, bundle, video_id, manifest,
                          manifest_written=manifest_written)

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
