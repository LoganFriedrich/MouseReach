"""Find a video's files on THIS node, or say honestly that they are not here.

The watcher's database row for a video carries ``source_path``, ``current_path``
and ``dlc_output_path``. Every handler downstream reads one of those and assumes
a file is there. Three separate ways of getting that wrong have each taken the
pipeline down:

  ``Path(None)``      raised ``TypeError: expected str, bytes or os.PathLike``
                      and failed nearly every video on a GPU node.
  ``Path(x or '')``   is ``Path('.')``, the current directory, which *exists* --
                      so the obvious guard never fires, the algorithms are handed
                      a directory as their HDF5 file, and hundreds of videos went to the
                      human review queue as "[Errno 13] Permission denied: '.'".
  a placeholder       ``source_path='recovered'`` let cross-node recovery mint
                      rows for videos this node has no files for. They sit in a
                      working state forever, are picked up every cycle, and crash
                      or churn on each pass.

The rule these all break is the same one: a recorded path is a claim, not a
fact. This module checks the claim. Everything here returns a real file or
``None`` -- never a placeholder, never a directory standing in for a file.

GPU NODES AND THE POSED FOLDER
------------------------------
``Paths.DLC_STAGING`` (Processing/Posed on the shared drive) is the processing
server's intake: a GPU node stages a finished pose there and the processing
server alone takes it in. It is searched by default because the processing
server legitimately finds its work there. A GPU node must not: a Posed hit
lets it copy another node's hand-off into its own DLC queue (posing the same
video twice), run DLC with its input in Posed (DLC writes the .h5 beside its
input, i.e. into the shared intake), decide a video is "already staged" and
mark it archived, or count the server's copy as its own in-flight file. Each
of those duplicates or corrupts a hand-off another node owns. So GPU-side
callers pass ``include_staging=False`` / ``search_staging=False``, which drops
the folder from the search AND refuses a recorded path that points into it.
"""

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from mousereach.pipeline.manifest import select_pose_file

logger = logging.getLogger(__name__)


def node_search_dirs(include_staging: bool = True) -> List[Path]:
    """Directories on this node where a video's files can legitimately live.

    Ordered by how current they are: the working folders first, the NAS staging
    hand-off next, the archive last. Only existing directories are returned, so
    a node with no NAS mapped simply searches fewer places.

    ``include_staging=False`` drops ``Paths.DLC_STAGING``: a GPU node never
    reads from the processing server's intake (see the module docstring).
    The order of the rest is unchanged.

    Only the TOP of ``Paths.SINGLE_ANIMAL_OUTPUT`` is searched, never its
    ``.inflight/<host>/`` claim folders. WHY: a single found there has been
    claimed by a node for pose; finding it by name would let this node copy
    and pose another node's claim. A node reaches its own claim through the
    path recorded on its row (``raw``), not through this search.
    """
    from mousereach.config import Paths

    candidates = [
        Paths.PROCESSING,
        Paths.DLC_QUEUE,
        Paths.DLC_STAGING if include_staging else None,
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


def _path_keys(p) -> set:
    """Comparable spellings of a path: as written (made absolute) and resolved.

    Both, because a mapped drive and its network name resolve differently on
    some Python versions, and a path that cannot be resolved still has to be
    comparable. Deliberately NOT os.path.samefile: over a network share
    Windows can report the same file id for every folder, and samefile would
    then call any two folders on that share "the same".
    """
    keys = set()
    try:
        keys.add(os.path.normcase(os.path.abspath(os.fspath(p))))
    except (TypeError, ValueError, OSError):
        pass
    try:
        keys.add(os.path.normcase(str(Path(p).resolve())))
    except (TypeError, ValueError, OSError, RuntimeError):
        pass
    return keys


def _staging_keys() -> set:
    """``_path_keys`` of the configured staging folder; empty when none."""
    try:
        from mousereach.config import Paths
        staging = Paths.DLC_STAGING
    except Exception:
        return set()
    return _path_keys(staging) if staging else set()


def _folder_is_or_under(folder, keys: set) -> bool:
    """True when ``folder`` or any folder above it has one of ``keys``.

    Walks the parents of the absolute and the resolved spelling as strings,
    so a deep path costs one resolve, not one per level (these run against a
    network share).
    """
    if not keys or not folder:
        return False
    chains = []
    try:
        a = Path(os.path.abspath(os.fspath(folder)))
        chains.append([a] + list(a.parents))
    except (TypeError, ValueError, OSError):
        pass
    try:
        r = Path(folder).resolve()
        chains.append([r] + list(r.parents))
    except (TypeError, ValueError, OSError, RuntimeError):
        pass
    return any(os.path.normcase(str(f)) in keys for chain in chains for f in chain)


def is_in_staging(path) -> bool:
    """True when ``path`` lies in ``Paths.DLC_STAGING`` (directly, or in a
    folder beneath it such as its claim folder).

    WHY beneath too: every file under the processing server's intake belongs
    to that server's hand-off, not just the top level. False when no staging
    folder is configured or the path is empty.
    """
    if not path:
        return False
    try:
        parent = Path(path).parent
    except (TypeError, ValueError):
        return False
    return _folder_is_or_under(parent, _staging_keys())


def is_in_archive(path) -> bool:
    """True when ``path`` lies in the filed folder (``Paths.ANALYZED_OUTPUT``,
    directly or anywhere beneath it). False when none is configured or the
    path is empty.

    WHY: ``search_archive=False`` used to drop the archive only from the
    SEARCH; a recorded path pointing into it was still returned. A handler
    that moves what it finds (staging) would then move a filed video's
    results out of the archive."""
    if not path:
        return False
    try:
        from mousereach.config import Paths
        root = Paths.ANALYZED_OUTPUT
    except Exception:
        return False
    if not root:
        return False
    try:
        parent = Path(path).parent
    except (TypeError, ValueError):
        return False
    return _folder_is_or_under(parent, _path_keys(root))


def _recorded_path_allowed(raw, search_archive: bool, search_staging: bool) -> bool:
    """Whether a recorded path may be returned as-is: not inside the staging
    folder when ``search_staging`` is False, not inside the filed folder when
    ``search_archive`` is False."""
    if not search_staging and is_in_staging(raw):
        return False
    if not search_archive and is_in_archive(raw):
        return False
    return True


# Files a file manager may leave in any folder; they are not part of a bundle.
_NOT_BUNDLE_FILES = frozenset({"desktop.ini", "thumbs.db", ".ds_store"})


def _holds_a_bundle(folder: Path) -> bool:
    """True when ``folder`` exists and holds at least one file of a bundle
    (any regular file, apart from file-manager leftovers and dot-files).
    Raises OSError when the folder cannot be read.

    WHY not just "the folder exists": an empty or leftover residue folder
    (a route that never got its files, a bundle emptied by hand) is not a
    hold. Counted as one, the processing server recorded a video it had no
    row for as 'triage' / 'deep_review' and took nothing in; the review
    return scan then retired the empty folder, and the row stayed held for
    good with no bundle and no review tool listing it."""
    if not folder.is_dir():
        return False
    with os.scandir(folder) as it:
        for e in it:
            name = e.name
            if name.startswith('.') or name.lower() in _NOT_BUNDLE_FILES:
                continue
            if e.is_file():
                return True
    return False


def held_or_filed_on_share(video_id: str) -> Optional[Tuple[str, str]]:
    """Is this video finished, or held for a person? ``(state, reason)`` if
    so, None if it is not.

    ``state`` is what a row for it should say: 'archived', or the review
    queue's state ('triage' / 'deep_review').

    WHY ask the shared drive instead of a node's database: the database is
    exactly what cannot be trusted here. A new node, or one whose database was
    rebuilt, knows nothing about the videos already analysed or held. The
    shared drive is the authority:

      * a bundle folder in Processing/Review/triage/<stem> or
        deep_review/<stem> means a person holds the video -- taking it in
        again would build a second result beside the one under review;
      * <queue>/.incoming/<stem> is a route into that queue still being
        written (review_routing publishes it with one rename) -- the same
        hold, a moment earlier;
      * <stem>_processing_manifest.json in the video's archive folder means
        it was analysed and filed.
    A queue folder counts only when it holds bundle files (_holds_a_bundle):
    an empty one is residue, not a hold. A leftover kept for a person (a
    folder holding the note review_routing.LEFT_NOTE) is not a hold either
    (review_routing.is_queue_bundle): it is not the video's live copy, and
    counting it kept a video that needs a pose from ever getting one.

    The queues are checked first because a held video can ALSO have an older
    manifest in the archive (a reprocess under review), and the hold is the
    more current truth -- the same order the filing step uses. Deep review
    first: it supersedes triage. Never raises: an unreadable location counts
    as "not there".

    Used by a GPU node before it poses a video (DLCOrchestrator.
    _already_done_or_held), by the CPU pose pool, and by the processing
    server before it registers a posed video it has no row for
    (state.WatcherStateManager.discover_dlc_staged).
    """
    from mousereach.config import Paths
    try:
        from mousereach.watcher.review_routing import (INCOMING_DIR_NAME,
                                                       is_queue_bundle)
    except ImportError:
        INCOMING_DIR_NAME = ".incoming"

        def is_queue_bundle(folder):
            return True
    for qroot, qstate in ((getattr(Paths, 'DEEP_REVIEW', None), 'deep_review'),
                          (getattr(Paths, 'TRIAGE_REVIEW', None), 'triage')):
        if not qroot:
            continue
        try:
            if (is_queue_bundle(Path(qroot) / video_id)
                    and _holds_a_bundle(Path(qroot) / video_id)):
                return qstate, f"its bundle is held for a person in the {qstate} queue"
            if (is_queue_bundle(Path(qroot) / INCOMING_DIR_NAME / video_id)
                    and _holds_a_bundle(Path(qroot) / INCOMING_DIR_NAME / video_id)):
                return qstate, (f"its bundle is being routed into the {qstate} "
                                f"queue ({INCOMING_DIR_NAME})")
        except OSError:
            continue
    try:
        from mousereach.archive.core import get_archive_destination
        archived_here = get_archive_destination(video_id)
        if (archived_here is not None
                and (Path(archived_here) / f"{video_id}_processing_manifest.json").is_file()):
            return 'archived', "already analysed and filed in the archive"
    except Exception as err:
        logger.warning('%s: could not check the archive, so it counts as not filed: %s',
                       video_id, err)
    return None


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


def _search_dirs(video_id: str, extra_dirs: Iterable, search_archive: bool,
                 search_staging: bool) -> list:
    """extra_dirs, then this node's working folders, then the archive -- with
    the staging folder left out everywhere when ``search_staging`` is False
    (an extra_dirs entry that IS staging would otherwise put it back)."""
    dirs = list(extra_dirs)
    if not search_staging:
        keys = _staging_keys()
        if keys:
            dirs = [d for d in dirs if d and not _folder_is_or_under(d, keys)]
    dirs += node_search_dirs(include_staging=search_staging)
    if search_archive:
        archive = archive_dir_for(video_id)
        if archive:
            dirs.append(archive)
    return dirs


def locate_pose_file(video_id: str, raw=None, extra_dirs: Iterable = (),
                     search_archive: bool = True,
                     search_staging: bool = True) -> Optional[Path]:
    """This video's pose file anywhere on this node, or None.

    Searches the recorded path first, then ``extra_dirs``, then the working
    folders, then (unless ``search_archive`` is False) this video's archive
    folder. ``select_pose_file`` picks the declared model when a video has pose
    from more than one.

    ``search_staging=False`` (every GPU-node caller): the processing server's
    intake folder is not searched, and a recorded ``raw`` path inside it is
    ignored as if it were absent -- see the module docstring. Likewise
    ``search_archive=False`` ignores a recorded path inside the filed folder.
    """
    if raw and not _recorded_path_allowed(raw, search_archive, search_staging):
        raw = None
    dirs = _search_dirs(video_id, extra_dirs, search_archive, search_staging)
    return resolve_pose_input(raw, video_id, *dirs)


def locate_video_file(video_id: str, raw=None, extra_dirs: Iterable = (),
                      search_archive: bool = True,
                      search_staging: bool = True) -> Optional[Path]:
    """This video's .mp4 anywhere on this node, or None.

    Same search order as ``locate_pose_file``. Used before a row is created or
    acted on, so a video with no file here is never given a state that says
    there is one.

    Pass ``search_archive=False`` from any handler that MOVES or WRITES BESIDE
    what it finds -- staging and DLC inference both do. A hit in the archive is
    the finished copy of the video; moving it out, or dropping a new pose file
    next to it, would damage the archive to satisfy a queue. A recorded
    ``raw`` path inside the filed folder is then ignored too (it used to be
    returned whenever the file existed).

    Pass ``search_staging=False`` from every GPU-node handler: the processing
    server's intake folder is not searched, and a recorded ``raw`` path inside
    it is ignored (not returned) -- see the module docstring.
    """
    if raw:
        p = Path(raw)
        if _recorded_path_allowed(p, search_archive, search_staging):
            if p.is_file():
                return p
    dirs = _search_dirs(video_id, extra_dirs, search_archive, search_staging)
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
