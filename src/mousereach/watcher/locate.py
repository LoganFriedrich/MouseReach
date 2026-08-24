"""Find a video's files on THIS node, or say honestly that they are not here.

The watcher's database row for a video carries ``source_path``, ``current_path``
and ``dlc_output_path``. Every handler downstream reads one of those and assumes
a file is there. Three separate ways of getting that wrong have each taken the
pipeline down:

  ``Path(None)``      raised ``TypeError: expected str, bytes or os.PathLike``
                      and failed 950 of 954 videos on the DLC PC.
  ``Path(x or '')``   is ``Path('.')``, the current directory, which *exists* --
                      so the obvious guard never fires, the algorithms are handed
                      a directory as their HDF5 file, and 723 videos went to the
                      human review queue as "[Errno 13] Permission denied: '.'".
  a placeholder       ``source_path='recovered'`` let cross-node recovery mint
                      rows for videos this node has no files for. They sit in a
                      working state forever, are picked up every cycle, and crash
                      or churn on each pass.

The rule these all break is the same one: a recorded path is a claim, not a
fact. This module checks the claim. Everything here returns a real file or
``None`` -- never a placeholder, never a directory standing in for a file.
"""

import logging
from pathlib import Path
from typing import Iterable, List, Optional

from mousereach.pipeline.manifest import select_pose_file

logger = logging.getLogger(__name__)


def node_search_dirs() -> List[Path]:
    """Directories on this node where a video's files can legitimately live.

    Ordered by how current they are: the working folders first, the NAS staging
    hand-off next, the archive last. Only existing directories are returned, so
    a node with no NAS mapped simply searches fewer places.
    """
    from mousereach.config import Paths

    candidates = [
        Paths.PROCESSING,
        Paths.DLC_QUEUE,
        Paths.DLC_STAGING,
        Paths.SINGLE_ANIMAL_OUTPUT,
    ]
    out = []
    for d in candidates:
        if not d:
            continue
        try:
            if Path(d).is_dir():
                out.append(Path(d))
        except OSError:
            continue
    return out


def archive_dir_for(video_id: str) -> Optional[Path]:
    """This video's archive folder, computed from its id -- no tree walk.

    Returns None if it cannot be computed or does not exist.
    """
    try:
        from mousereach.archive.core import get_archive_destination
        d = get_archive_destination(video_id)
    except Exception:
        return None
    try:
        return Path(d) if d and Path(d).is_dir() else None
    except OSError:
        return None


def resolve_pose_input(raw, video_id: str, *search_dirs):
    """Resolve a video's pose file, or None if it genuinely has none.

    An absent path must stay absent -- see the module docstring for why the
    obvious ``Path(raw or '')`` guard cannot work. Tests ``is_file()`` rather
    than ``exists()`` so a directory can never satisfy "is this my pose file".

    Args:
        raw: the recorded dlc_output_path (may be None/'')
        video_id: stem used to glob the fallback directories
        *search_dirs: directories to search, in order, if raw does not resolve

    Returns:
        Path to the pose file, or None.
    """
    if raw:
        p = Path(raw)
        if p.is_file():
            return p
    for d in search_dirs:
        if not d:
            continue
        try:
            hits = list(Path(d).glob(f"{video_id}DLC*.h5"))
        except OSError:
            continue
        if hits:
            chosen = select_pose_file(hits)
            if chosen is not None and Path(chosen).is_file():
                return Path(chosen)
    return None


def locate_pose_file(video_id: str, raw=None, extra_dirs: Iterable = (),
                     search_archive: bool = True) -> Optional[Path]:
    """This video's pose file anywhere on this node, or None.

    Searches the recorded path first, then ``extra_dirs``, then the working
    folders, then (unless ``search_archive`` is False) this video's archive
    folder. ``select_pose_file`` picks the declared model when a video has pose
    from more than one.
    """
    dirs = list(extra_dirs) + node_search_dirs()
    if search_archive:
        archive = archive_dir_for(video_id)
        if archive:
            dirs.append(archive)
    return resolve_pose_input(raw, video_id, *dirs)


def locate_video_file(video_id: str, raw=None, extra_dirs: Iterable = (),
                      search_archive: bool = True) -> Optional[Path]:
    """This video's .mp4 anywhere on this node, or None.

    Same search order as ``locate_pose_file``. Used before a row is created or
    acted on, so a video with no file here is never given a state that says
    there is one.

    Pass ``search_archive=False`` from any handler that MOVES or WRITES BESIDE
    what it finds -- staging and DLC inference both do. A hit in the archive is
    the finished copy of the video; moving it out, or dropping a new pose file
    next to it, would damage the archive to satisfy a queue.
    """
    if raw:
        p = Path(raw)
        if p.is_file():
            return p
    dirs = list(extra_dirs) + node_search_dirs()
    if search_archive:
        archive = archive_dir_for(video_id)
        if archive:
            dirs.append(archive)
    for d in dirs:
        if not d:
            continue
        try:
            candidate = Path(d) / f"{video_id}.mp4"
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None
