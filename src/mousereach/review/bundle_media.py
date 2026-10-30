"""Find a review bundle's video and pose file, even when its manifest is stale.

A review bundle's manifest records ABSOLUTE ``canonical_video_path`` and
``canonical_dlc_h5_path``. Those paths go stale whenever a folder above the
bundle is renamed (for example the deep-review queue folder being renamed):
the files are still there, inside the bundle, but the manifest points at a
folder that no longer exists. Opening the manifest path blindly made every such
bundle unopenable.

The rule here: the manifest's path wins when it is a real file (it may point at
the finished-work tree, which is where a copy-free bundle's media live);
otherwise fall back to the copy inside the bundle folder; otherwise to the
folder the video is filed in NOW (``filed_media_dirs``), because a copy-free
bundle's manifest goes stale the same way when the finished-work folders are
reorganised; otherwise ``None``, so the caller can say plainly that nothing
was found.

Pure: no Qt, no napari, no writes. Never raises for a missing or odd path --
an old folder name may now be a plain FILE, and walking into it must read as
"not found", not as a crash.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional


def _is_file(p: Any) -> bool:
    """True when ``p`` names an existing regular file. Never raises.

    WHY the try: a path that runs THROUGH a plain file (an old folder name
    replaced by a file) can raise NotADirectoryError on some platforms instead
    of just reporting False.
    """
    if not p:
        return False
    try:
        return Path(p).is_file()
    except (OSError, ValueError, TypeError):
        return False


def _stem_for(manifest: Optional[Mapping[str, Any]],
              bundle_dir: Optional[Path], stem: Optional[str]) -> str:
    """The video stem: explicit argument, else the manifest's, else the bundle
    folder name (bundles are laid out as ``<queue>/<stem>/``)."""
    if stem:
        return str(stem)
    if manifest and manifest.get("video_stem"):
        return str(manifest["video_stem"])
    if bundle_dir is not None:
        return Path(bundle_dir).name
    return ""


def filed_media_dirs(stem: str) -> list:
    """Where a filed video's media are now: its filed folder under Analyzed
    (the filing rule's answer), then the same-named folder of each per-model
    pose tree beside it (``<project folder>/DLC Model <N>/<cohort folder>``).
    [] when Analyzed is not configured or the stem cannot be placed.

    WHY: a copy-free review bundle records ABSOLUTE paths into Analyzed when it
    is made. Moving the finished-work folders (2026-10-30: project folders
    regrouped by grant) left those paths pointing at folders that no longer
    exist, and every such bundle unopenable. Asking the filing rule finds the
    files wherever they are filed today."""
    if not stem:
        return []
    try:
        from mousereach.config import Paths
        from mousereach.archive.core import filed_folder
        root = getattr(Paths, "ANALYZED_OUTPUT", None)
        if not root:
            return []
        home = filed_folder(stem, root)
        out = [home]
        try:
            out += sorted(d / home.name for d in home.parent.glob("DLC Model *") if d.is_dir())
        except OSError:
            pass
        return out
    except Exception:
        return []


def resolve_bundle_video(manifest: Optional[Mapping[str, Any]],
                         bundle_dir: Optional[Path],
                         stem: Optional[str] = None) -> Optional[Path]:
    """The video to open for a review bundle, or None if there is none.

    Order: the manifest's ``canonical_video_path`` if it is a file, else
    ``bundle_dir/<stem>.mp4`` if that is a file, else ``<stem>.mp4`` in the
    folder the video is filed in now (``filed_media_dirs``), else None.
    """
    declared = (manifest or {}).get("canonical_video_path")
    if _is_file(declared):
        return Path(declared)
    s = _stem_for(manifest, bundle_dir, stem)
    # WHY require a stem: without one the fallback name would be ".mp4", which
    # can never be the right video.
    if not s:
        return None
    if bundle_dir is not None:
        local = Path(bundle_dir) / f"{s}.mp4"
        if _is_file(local):
            return local
    for d in filed_media_dirs(s)[:1]:
        filed = d / f"{s}.mp4"
        if _is_file(filed):
            return filed
    return None


def resolve_bundle_pose(manifest: Optional[Mapping[str, Any]],
                        bundle_dir: Optional[Path],
                        stem: Optional[str] = None,
                        expected_scorer: Optional[str] = None) -> Optional[Path]:
    """The pose (.h5) file for a review bundle, or None if there is none.

    Order: the manifest's ``canonical_dlc_h5_path`` if it is a file, else the
    pose inside the bundle folder chosen by ``select_pose_file``, else the pose
    where the video is filed now (``filed_media_dirs``), else None.

    WHY select_pose_file and not the first glob hit: a re-posed video can carry
    pose files from two models side by side, and the filesystem's listing order
    is arbitrary. select_pose_file prefers the declared model (``expected_scorer``
    or the pipeline's declared scorer), then the newest file.
    """
    declared = (manifest or {}).get("canonical_dlc_h5_path")
    if _is_file(declared):
        return Path(declared)
    s = _stem_for(manifest, bundle_dir, stem)
    # WHY require a stem: "DLC*.h5" alone would match any video's pose.
    if not s:
        return None
    hits = []
    for d in ([Path(bundle_dir)] if bundle_dir is not None else []) + filed_media_dirs(s):
        try:
            hits = [p for p in d.glob(f"{s}DLC*.h5") if _is_file(p)]
        except OSError:
            # The folder may be a plain file or unreachable: not found.
            hits = []
        if hits:
            break
    if not hits:
        return None
    from mousereach.pipeline.manifest import select_pose_file
    return select_pose_file(hits, expected_scorer=expected_scorer)
