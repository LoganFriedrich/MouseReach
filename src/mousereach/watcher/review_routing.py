"""Whole-video bundle routing for the human-review holds.

A video the algo cannot commit -- segmentation FAILED, or ANY element was left
triaged / flagged -- is held OUT of the archive and the central database until a human
clears it. This module MOVES the whole video bundle (the mp4, the DLC pose h5,
and every algo output JSON) out of the local Processing dir into a
self-contained review bundle under the Triage or Deep_Review queue on the NAS,
so that:

  * the video travels WITH its outputs -- the review tool opens it in place,
  * the Processing dir is left clean (no video stalls there forever), and
  * kinematics + DB sync never run until the bundle comes back through the gate.

Every move is recorded in a ``{stem}_routing.json`` manifest (source,
destination, reason, moved files, files that failed to move, timestamp) for a
full audit trail. A video is never merged into a bundle it already has in
the destination queue: review_gate.route_to_queue refuses, or first clears an
old folder as a leftover (clear_leftover), before anything moves.

A NEW bundle is never visible half-built. It is assembled in the hidden
``dest_root/.incoming/{stem}/`` and published with ONE directory rename, so a
queue reader sees either no bundle or the whole bundle. WHY: the move used to
create the empty ``{stem}`` folder first and then fill it file by file
(cross-volume copies take seconds); the running watcher's return scan saw the
empty folder, retired it mid-route, and every remaining move failed with
"No such file or directory".

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import os
import json
import logging
import shutil
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Files that belong to a video bundle. The DLC pose h5 is named
# ``{stem}DLC_resnet...h5`` (next char after the stem is 'D'); the mp4 is
# ``{stem}.mp4`` (next char '.'); every algo output is ``{stem}_*.json`` (next
# char '_'). Requiring the next char to be one of these prevents a stem that is
# a strict prefix of another (CNT0312_P2 vs CNT0312_P21) from stealing files.
from mousereach.pipeline.fsutil import VIDEO_NAME_BOUNDARY as _BOUNDARY_CHARS  # noqa: E402

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


# ---------------------------------------------------------------------------
# DEEP REVIEW SUPERSEDES TRIAGE
#
# A video in the deep-review queue is not allowed to be in the triage queue as
# well. Deep review is the stronger hold (the whole video, cuts included, is in
# question), so a triage copy of such a video is never created and, when one
# exists, the move into deep review empties it (review_gate._drop_triage_copy):
# a video or pose file the deep-review bundle has none of is moved in; then
# the folder is a leftover (clear_leftover, block comment below) -- what has a
# current copy elsewhere (in the deep-review bundle, or the review in the
# durable review store) is deleted, and what a person wrote and has no such
# copy stays where it is, listed for a person. No answer file is carried into
# the deep-review bundle (re-segmentation voids every answer after it). Every
# reader ignores a triage folder of a video in deep review (census, dashboard,
# reconcile, the return scan, the review tools' triage list). WHY: a router
# that did not check sent videos already in deep review into triage as well; a
# reviewer then answered a triage copy holding older results while the
# deep-review copy was the one that counted.
# ---------------------------------------------------------------------------

def is_queue_bundle(folder) -> bool:
    """True when ``folder`` is a video's bundle in a review queue (published,
    or still being built in ``.incoming``): a folder that is NOT a leftover
    kept for a person (one holding LEFT_NOTE, clear_leftover). Never raises;
    an unreadable folder counts as no bundle.

    WHY one predicate for every reader: a leftover kept for a person stays in
    its queue under the video's name, so "is there a folder named after the
    video" answers yes for it. A reader that took that as "the video is held
    here" treated a folder that is not the video's live copy as its hold: a
    later route into triage saw "deep review supersedes triage", relabelled
    the row and moved nothing, and the video was left with no live bundle at
    all (the return scan skips such folders, so nothing moved it on). Every
    place that asks "is the video held in this queue" asks this."""
    try:
        p = Path(folder)
        return p.is_dir() and not (p / LEFT_NOTE).is_file()
    except (OSError, TypeError, ValueError):
        return False


def deep_review_stems() -> set:
    """Every video with a bundle in the deep-review queue, published or still
    being built there (``.incoming``). A leftover kept for a person is not a
    bundle (is_queue_bundle). Empty when the queue is not configured or cannot
    be read."""
    from ..config import Paths
    root = getattr(Paths, "DEEP_REVIEW", None)
    out = set()
    if not root:
        return out
    for folder in (Path(root), Path(root) / INCOMING_DIR_NAME):
        try:
            with os.scandir(folder) as entries:
                for e in entries:
                    if (not e.name.startswith(".") and e.is_dir()
                            and is_queue_bundle(e.path)):
                        out.add(e.name)
        except OSError:
            continue
    return out


def in_deep_review(video_id: str) -> bool:
    """Whether ``video_id`` has a bundle in the deep-review queue (published or
    being built; a leftover kept for a person does not count, is_queue_bundle).
    Never raises."""
    from ..config import Paths
    root = getattr(Paths, "DEEP_REVIEW", None)
    if not root:
        return False
    return (is_queue_bundle(Path(root) / video_id)
            or is_queue_bundle(Path(root) / INCOMING_DIR_NAME / video_id))


def superseded_by_deep_review(bundle_dir) -> bool:
    """True when ``bundle_dir`` is a bundle in the TRIAGE queue whose video is
    also in deep review -- a copy every reader ignores (deep review supersedes
    triage). False for a deep-review bundle, when the video's deep-review
    folder is only a leftover kept for a person, and on any error."""
    from ..config import Paths
    triage = getattr(Paths, "TRIAGE_REVIEW", None)
    if not triage or not bundle_dir:
        return False
    try:
        if (os.path.normcase(os.path.abspath(str(Path(bundle_dir).parent)))
                != os.path.normcase(os.path.abspath(str(triage)))):
            return False
    except (TypeError, ValueError):
        return False
    return in_deep_review(Path(bundle_dir).name)


def _bundle_files(source_dir: Path, video_id: str) -> List[Path]:
    """Every file in ``source_dir`` that belongs to ``video_id``.

    Uses a prefix-boundary guard so ``CNT0312_P2`` does not also match
    ``CNT0312_P21``'s files.
    """
    out: List[Path] = []
    if not source_dir.exists():
        return out
    n = len(video_id)
    # The name is tested BEFORE asking whether the entry is a file, and the
    # listing's own entry answers that (no extra round trip on Windows). WHY:
    # routing a finished video takes its files from the cohort's Analyzed
    # folder -- tens of thousands of entries -- and asking the share "is this a
    # file?" for every one cost minutes per video; a nightly check ran out of
    # its time limit with most of its videos left.
    with os.scandir(source_dir) as entries:
        for e in entries:
            name = e.name
            if not name.startswith(video_id):
                continue
            # exact stem match (name == stem, unlikely) or a valid boundary char
            if not (len(name) == n or name[n] in _BOUNDARY_CHARS):
                continue
            if e.is_file():
                out.append(Path(e.path))
    return out


# ---------------------------------------------------------------------------
# WHICH OF A VIDEO'S FILES A PERSON WROTE
#
# A bundle holds the video and its pose (INPUTS), what the pipeline made
# (MACHINE OUTPUTS: one fixed list, the one the archive's supersede step uses,
# plus the queue's own records) and what a person did: a review, ground truth,
# a clear marker, validation files, cuts set by hand. Every step that deletes,
# retires or overwrites a video's files asks these functions, and only machine
# outputs and inputs are ever deleted or replaced without a person; anything
# else counts as a person's file. WHY an allowlist of machine outputs and not a
# list of person files: a person-written name nobody listed (an older tool's
# validation file, a renamed clear marker) must fall on the safe side.
# ---------------------------------------------------------------------------

_QUEUE_RECORD_SUFFIXES = ("_manifest.json", "_routing.json")


def is_queue_record(video_id: str, name: str) -> bool:
    """A queue's own record of a bundle (its review manifest, its routing
    record, or the legacy bare manifest.json) -- not a file of the video."""
    return name == "manifest.json" or name in tuple(
        f"{video_id}{s}" for s in _QUEUE_RECORD_SUFFIXES)


def is_input_file(video_id: str, name: str) -> bool:
    """The video itself or a pose file (with the pose step's side files)."""
    return name == f"{video_id}.mp4" or name.startswith(f"{video_id}DLC")


def input_kind(video_id: str, name: str) -> Optional[str]:
    """'video' for the video, 'pose<extension>' for a pose-step file, None for
    anything else. Two pose files of different models are the same kind."""
    if name == f"{video_id}.mp4":
        return "video"
    if name.startswith(f"{video_id}DLC"):
        return "pose" + Path(name).suffix.lower()
    return None


def is_machine_output(video_id: str, name: str) -> bool:
    """A file the pipeline makes again whenever the video is analysed again
    (archive.supersede's list), or one of the queue's own records. Name only:
    a hand-cut segments file passes here -- is_person_file reads it."""
    from mousereach.archive.supersede import _ALGO_SUFFIXES
    if not name.startswith(video_id):
        return False
    rest = name[len(video_id):]
    return rest in _ALGO_SUFFIXES or rest in _QUEUE_RECORD_SUFFIXES


# Marks a review tool leaves in a file the pipeline also writes. A person who
# checks or corrects the algorithm's segments, reaches or outcomes saves them
# INTO the machine-named file: the segmentation review saves
# ``validation_status`` 'validated'; the re-segmentation fixer
# ``boundary_source`` 'human'; the reach and outcome reviews and the ground-
# truth tool ``human_verified`` / ``human_corrected`` (per segment or per
# reach) and 'validated'; the triage-clearing tools ``triage_cleared``. The
# pipeline itself writes 'needs_review' or 'auto_approved' and never sets
# these flags. Only the three files the review tools save into are read
# (_REVIEWED_IN_PLACE); the other machine outputs (features, the processing
# manifest, the triage record, reach assignments) are derived from them and
# made again by the pipeline, so a copy of a person's mark in one of those is
# not the person's work itself.
_PERSON_FLAGS = ("human_verified", "human_corrected", "triage_cleared")
_REVIEWED_IN_PLACE = ("_segments.json", "_reaches.json", "_pellet_outcomes.json")


def _holds_a_persons_mark(doc) -> bool:
    """True when the parsed JSON ``doc`` carries any mark above, at the top
    level or anywhere inside (a segment, a reach), or a segment the review
    guard counts as human-locked (review/clear_guard)."""
    stack = [doc]
    while stack:
        d = stack.pop()
        if isinstance(d, dict):
            if d.get("validation_status") == "validated":
                return True
            if d.get("boundary_source") == "human":
                return True
            if any(d.get(k) is True for k in _PERSON_FLAGS):
                return True
            stack.extend(v for v in d.values() if isinstance(v, (dict, list)))
        elif isinstance(d, list):
            stack.extend(v for v in d if isinstance(v, (dict, list)))
    try:
        from mousereach.review.clear_guard import is_outcome_locked, is_reach_locked
        for seg in (doc or {}).get("segments") or []:
            if isinstance(seg, dict) and (is_outcome_locked(seg) or is_reach_locked(seg)):
                return True
    except Exception:
        pass
    return False


def is_person_file(path, video_id: str) -> bool:
    """True when ``path`` holds something a person did and must never be
    deleted or replaced without a person:

      * any file of the video that is neither an input nor a machine output
        (a review, ground truth, a clear marker, an unknown name);
      * a segments, reaches or outcomes file that a review tool saved a
        person's check or correction into (_holds_a_persons_mark:
        'validated', hand-set cuts, human_verified / human_corrected /
        triage_cleared anywhere in it);
      * such a file that cannot be read or parsed (the safe side: what it
        holds cannot be told).

    The queue's own records (review manifest, routing record) are never a
    person's file. Callers that find a byte-identical copy of the file in the
    video's live copy may still delete it -- nothing is lost.

    WHY content is read and not only the name: the review tools save their
    corrections into the very file the pipeline writes, so a name-only rule
    called a person's corrected reaches or outcomes "machine output", and a
    clean-up deleted or replaced them."""
    path = Path(path)
    name = path.name
    if is_input_file(video_id, name):
        return False
    if not is_machine_output(video_id, name):
        return True
    if name[len(video_id):] not in _REVIEWED_IN_PLACE:
        return False
    try:
        return _holds_a_persons_mark(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return True


_COMPARE_BYTES_UP_TO = 64 * 1024 * 1024


def _same_content(a, b) -> bool:
    """True when ``a`` and ``b`` hold the same bytes. Files up to 64 MB (every
    person's file) are compared byte for byte; WHY not the archive's quick
    size-and-time check: two saves of a review a second apart can share both.
    Larger files (a video) use that quick check, then SHA-256. Never raises."""
    try:
        a, b = Path(a), Path(b)
        sa, sb = a.stat().st_size, b.stat().st_size
        if sa != sb:
            return False
        if sa <= _COMPARE_BYTES_UP_TO:
            return a.read_bytes() == b.read_bytes()
        from mousereach.archive.core import _same_file
        return bool(_same_file(a, b))
    except Exception:
        return False


class PersonFileInTheWay(OSError):
    """A move would have replaced a file a person wrote with a different one."""


def _move_keeping_person_files(src: Path, dst: Path, video_id: str) -> None:
    """_safe_move, except that a person's file at ``dst`` is never replaced by
    a DIFFERENT file (PersonFileInTheWay is raised and nothing moves). An
    identical copy may replace it -- nothing is lost."""
    if (dst.exists() and not same_path(src, dst) and is_person_file(dst, video_id)
            and not _same_content(src, dst)):
        raise PersonFileInTheWay(
            f"{dst} is a person's file and differs from {src}; not replaced")
    _safe_move(src, dst)


def _move_leaving_no_copy(src: Path, dst: Path, video_id: str) -> None:
    """_move_keeping_person_files, and when the move fails with ``src`` still
    in place, any copy it left at ``dst`` is deleted before the error is
    raised again. WHY: a file held open (Windows sharing violation) cannot be
    renamed, so shutil.move copies it and then fails to delete the source --
    the whole copy is left at the destination while the source stays. The
    file was then in two places: the bundle (or the hidden build folder)
    held a copy, Processing or the filed folder still held the original, and
    an all-or-nothing route that "put everything back" left that copy behind.
    The copy is deleted only when it is a copy of ``src`` (new at ``dst``, or
    byte-identical to ``src``) -- nothing that was there before is lost."""
    existed = dst.exists()
    try:
        _move_keeping_person_files(src, dst, video_id)
    except PersonFileInTheWay:
        raise
    except Exception:
        try:
            if (src.is_file() and dst.is_file() and not same_path(src, dst)
                    and (not existed or _same_content(src, dst))):
                dst.unlink()
                logger.warning(f"Routing {video_id}: {src.name} could not be moved; "
                               f"the copy the move left at {dst} was deleted, the "
                               f"file stays at {src}")
        except OSError as e:
            logger.error(f"Routing {video_id}: {src.name} could not be moved, and the "
                         f"copy the move left at {dst} could not be deleted ({e}); "
                         f"the file is now in both places -- delete {dst} by hand")
        raise


def same_path(a, b) -> bool:
    """True when ``a`` and ``b`` name the same file or folder (case and
    separators normalised, as Windows compares them). Never raises."""
    try:
        if os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b))):
            return True
        return os.path.samefile(str(a), str(b))
    except (OSError, TypeError, ValueError):
        return False


def _safe_move(src: Path, dst: Path) -> None:
    """shutil.move that overwrites an existing destination file.

    A move of a file onto ITSELF does nothing. WHY: the overwrite below deletes
    the destination first, and when source and destination are the same file
    that deleted the file itself -- a route of a bundle onto its own folder
    (Flag Session pressed in the re-segmentation window) deleted every file of
    the bundle except the one the window still had open.

    Cross-filesystem moves (local Processing -> NAS) are copy+delete under the
    hood; that is intended -- the bundle must live on the NAS so any node / the
    GUI can review it.

    Retries transient Windows sharing violations (WinError 32/5): the pipeline
    routes a bundle seconds after its own readers finish with the mp4, and the
    handle is sometimes still closing. Without the retry the mp4 silently
    stayed behind in Processing while the rest of the bundle moved, leaving
    part-moved bundles. A genuinely stuck file still
    raises to the caller after the retries.
    """
    from mousereach.pipeline.fsutil import retry_transient

    if same_path(src, dst):
        return

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
    # WHY replaced in one step (fsutil.write_json_atomic, which also retries a
    # transient hold on the share): the return scan and the dashboard REWRITE
    # an existing record (crossing files off failed_files). Opening it for
    # writing emptied it first, so a lost share connection or a killed process
    # part-way left a truncated record: read_routing_manifest then returns
    # None and the bundle loses its audit trail and its list of files that
    # never moved. Now a failed write leaves the old record whole.
    from mousereach.pipeline.fsutil import write_json_atomic
    write_json_atomic(folder / f"{video_id}_routing.json", manifest, indent=2)


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
                _move_keeping_person_files(f, bundle / f.name, video_id)
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


class RouteIncompleteError(OSError):
    """An all-or-nothing route could not move every file; everything it had
    moved was put back and nothing was routed."""


def _undo_route(video_id: str, done, build: Optional[Path]) -> None:
    """Move every file of an abandoned route back where it came from (newest
    first), then remove the staging folder ``build`` if it is left empty. A
    file that will not go back is logged as an ERROR naming both places."""
    for src, dst in reversed(done):
        try:
            _safe_move(dst, src)
        except Exception as e:
            logger.error(f"Routing {video_id}: could not move {dst} back to {src} "
                         f"while undoing an unfinished route -- move it back by "
                         f"hand: {e}")
    if build is not None:
        for d in (build, build.parent):
            try:
                d.rmdir()
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
    all_or_nothing: bool = False,
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
    all_or_nothing : bool
        True: if any file cannot be moved, every file already moved goes back
        to where it came from, the staging folder is removed, and
        RouteIncompleteError is raised -- nothing is published. For a route
        out of the FILED folder (a filed generation must never be half
        moved). False (the pipeline's own routes out of Processing): the
        route goes on, and the files that failed are listed in the routing
        record's ``failed_files``; the return scan moves them in later
        (review_return.repair_unfinished_routes).

    Returns
    -------
    (bundle_dir, moved_file_names)
        ``bundle_dir`` is always the FINAL ``dest_root/{stem}/`` folder, never
        the staging folder, so callers can write into it straight away.

    A new bundle is built in ``dest_root/.incoming/{stem}/`` and published by
    one rename (see the module docstring for why). A route never merges into a
    bundle the video already has in that queue: review_gate.route_to_queue
    (through _clear_the_way) removes, sets aside or refuses on any existing
    <queue>/<video> folder BEFORE calling this. What is left of moving into an
    existing folder here is narrow: an empty folder that could not be removed
    (nothing to replace), and a concurrent route of the same video that
    published first (the rest of this route's files go into it, and a
    person's file there is never replaced -- _move_keeping_person_files).
    """
    source_dir = Path(source_dir)
    dest_root = Path(dest_root)
    bundle = dest_root / video_id
    if same_path(source_dir, bundle):
        # The video is already where it is being sent: nothing to move, and
        # the routing record is NOT rewritten (it would list every file as
        # "failed" and replace the record of the route that brought it here).
        # WHY a guard here as well as in review_gate.route_to_queue: moving
        # each file onto itself deleted it (see _safe_move).
        logger.warning(f"Routing {video_id}: {source_dir} already is the bundle "
                       f"in {dest_root}; nothing moved")
        return bundle, []
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

    def _move_in(src: Path) -> Path:
        # Never replaces a person's file already in the bundle with a
        # different one (it stays at the source and is listed as failed).
        # Returns where the file went.
        # A file that cannot be moved leaves no copy behind
        # (_move_leaving_no_copy).
        try:
            _move_leaving_no_copy(src, build / src.name, video_id)
            return build / src.name
        except PersonFileInTheWay:
            raise
        except OSError:
            if not _published_by_other_route():
                raise
            _move_leaving_no_copy(src, bundle / src.name, video_id)
            return bundle / src.name

    moved: List[str] = []
    done: List[Tuple[Path, Path]] = []
    # Recorded in the manifest, not only logged: a part-moved bundle (a husk)
    # must be visible in the record the review tool and audits read. Each
    # entry names where the file still is ("source"), so the return scan can
    # finish the route later.
    failed: List[Dict[str, str]] = []
    for f in _bundle_files(source_dir, video_id):
        try:
            done.append((f, _move_in(f)))
            moved.append(f.name)
        except Exception as e:
            logger.warning(f"Routing {video_id}: could not move {f.name}: {e}")
            failed.append({"name": f.name, "error": str(e), "source": str(f)})

    for f in extra_sources or []:
        f = Path(f)
        if not f.exists():
            continue
        if (build / f.name).exists():
            continue  # already brought in by the main sweep
        try:
            done.append((f, _move_in(f)))
            moved.append(f.name)
        except Exception as e:
            logger.warning(f"Routing {video_id}: could not move extra {f.name}: {e}")
            failed.append({"name": f.name, "error": str(e), "source": str(f)})

    if failed and all_or_nothing:
        _undo_route(video_id, done, build if staged else None)
        raise RouteIncompleteError(
            f"{video_id}: {len(failed)} file(s) could not be moved "
            f"({', '.join(x['name'] for x in failed)}); every file moved was put "
            f"back in {source_dir} and nothing was routed")

    manifest = {
        "video_id": video_id,
        "routed_reason": reason,
        "source_dir": str(source_dir),
        "bundle_dir": str(bundle),
        "moved_files": sorted(moved),
        "failed_files": failed,
        "routed_at": datetime.now().isoformat(),
        # The machine that made the route. WHY: a failed file's "source" can be
        # a path on that machine's own disk (its Processing folder); only that
        # machine may finish the move later (review_return._finish_failed_moves).
        "routed_by": socket.gethostname(),
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
    except Exception as err:
        logger.warning('Could not read %s, so it counts as absent: %s', p, err)
        return None


# ---------------------------------------------------------------------------
# A QUEUE FOLDER THAT IS NOT THE VIDEO'S LIVE BUNDLE: A LEFTOVER
#
# A video is in ONE place at a time. A folder in a review queue that is not the
# video's live bundle -- an empty or records-only residue of a return, an old
# bundle of a video that has since been analysed again, the remains of a triage
# copy after the video moved into deep review -- is a leftover. What happens to
# each file in it (clear_leftover):
#
#   * deleted, once it is verified that a current copy exists elsewhere:
#       - the queue's own records (review manifest, routing record);
#       - any file the video's live copy holds byte for byte (the caller names
#         the live copy: the live bundle, or the files a route is bringing);
#       - a review the durable review store already holds byte for byte
#         (nothing is ever written into the store from here: a leftover's
#         review was made on cuts that are being, or have been, redone, and a
#         copy put in the store would be matched to the new cuts by frame
#         span -- a re-cut voids every answer after it);
#       - a machine output (archive.supersede's list), when the caller names
#         a live copy of the video at all -- the pipeline makes those again
#         from that copy;
#   * kept, WHERE IT IS: anything else -- a person's file (a review the store
#     does not hold, ground truth, a clear marker, hand-set cuts, a file with
#     an unknown name), a video or pose file the live copy does not hold byte
#     for byte, a subfolder. Nothing is moved anywhere.
#
# A folder that ends up empty is removed. A folder with something kept stays
# in its queue with a note in it (LEFT_NOTE) saying why and what to do. It is
# listed for a person in the review tool -- the Review Queues tab's 'Folders
# left for a person' (leftover_folders, read from the queue folders) -- with
# one button, "Nothing in it is needed: delete this folder"
# (delete_leftover_folder; the only deletion of a person's work, and only on
# a person's word). It is also on this machine's list for the dashboard's
# "Needs your attention" panel (<NAS root>/attention/
# <host>_queue_folders_for_a_person.json) until the folder or its note is
# gone. Such a folder is not a bundle to ANY reader (is_queue_bundle): the
# return scan skips it, no reader counts it as the video's hold, and a route
# meeting it is refused instead of merging into it.
#
# WHY this rule: a folder a person wrote in used to be either deleted with
# everything in it (a triage copy of a video moving into deep review) or moved
# out of the queue to a folder nobody looks at; and machine leftovers piled up
# in an archive folder that is meant only for cases the pipeline cannot
# process at all. A machine leftover with a current copy elsewhere is worth
# nothing; a person's work is never thrown away or hidden by the machine.
# ---------------------------------------------------------------------------

LEFT_NOTE = "LEFT_FOR_A_PERSON.json"
ATTENTION_SUFFIX = "_queue_folders_for_a_person.json"


def person_entries(folder, video_id: str) -> List[str]:
    """Names in ``folder`` that a person wrote (is_person_file), plus every
    subfolder (unknown content counts as a person's -- the safe side)."""
    out = []
    for f in Path(folder).iterdir():
        if f.is_dir() or (f.is_file() and is_person_file(f, video_id)):
            out.append(f.name)
    return sorted(out)


def left_for_a_person(folder) -> bool:
    """True when ``folder`` is a leftover kept for a person (it holds the
    note clear_leftover writes). Never raises."""
    try:
        return (Path(folder) / LEFT_NOTE).is_file()
    except OSError:
        return False


def review_held_by_store(review, video_id: str) -> bool:
    """True when the durable review store already holds ``review`` byte for
    byte. Nothing is written into the store. WHY: a leftover's review was
    made on cuts that are being, or have been, redone (a re-cut voids every
    answer after it), and the review readers match the store's copy to the
    current cuts by frame span -- a copy put there now would carry old
    answers onto new cuts; and replacing a different stored review could put
    an older answer where every reader looks. Never raises."""
    try:
        from mousereach.review.causal_review_io import durable_review_path
        dest = durable_review_path(video_id)
        if dest is None:
            return False
        dest = Path(dest)
        return dest.is_file() and _same_content(review, dest)
    except Exception as e:
        logger.warning(f"{video_id}: could not check the review in "
                       f"{Path(review).parent} against the durable store: {e}")
        return False


def clear_leftover(folder, video_id: str, live: Dict[str, Path], why: str,
                   what_to_do: str) -> List[str]:
    """Deal with the leftover queue folder ``folder`` by the rule in the block
    comment above. ``live`` maps each file name of the video's live copy to
    its path (empty when no live copy can be named). Returns the names kept
    (empty when the folder is gone). When anything is kept, the note is
    written and the folder is listed for a person. Never raises; a file that
    cannot be deleted is kept."""
    folder = Path(folder)
    kept: List[str] = []
    deleted: List[str] = []
    try:
        entries = sorted(folder.iterdir())
    except OSError as e:
        logger.warning(f"{video_id}: the leftover folder {folder} could not be "
                       f"listed ({e}); left where it is")
        return ["(the folder could not be listed)"]
    for f in entries:
        name = f.name
        if name == LEFT_NOTE:
            continue
        try:
            if not f.is_file():
                kept.append(name)
                continue
            twin = live.get(name)
            if is_queue_record(video_id, name):
                drop = True
            elif twin is not None and not same_path(f, twin) and _same_content(f, twin):
                drop = True
            elif name == f"{video_id}_causal_review.json":
                drop = review_held_by_store(f, video_id)
            elif is_person_file(f, video_id) or is_input_file(video_id, name):
                drop = False
            else:
                drop = bool(live)               # a machine output
            if drop:
                f.unlink()
                deleted.append(name)
            else:
                kept.append(name)
        except OSError as e:
            kept.append(name)
            logger.warning(f"{video_id}: {f} could not be deleted ({e}); kept")
    note = folder / LEFT_NOTE
    if not kept:
        try:
            if note.exists():
                note.unlink()
            folder.rmdir()
            logger.info(f"{video_id}: leftover folder {folder} removed ({why}; "
                        f"{len(deleted)} file(s) with a current copy elsewhere "
                        f"deleted)")
            drop_for_a_person(folder=folder)
            return []
        except OSError as e:
            kept.append(f"(the empty folder could not be removed: {e})")
    try:
        from mousereach.pipeline.fsutil import write_json_atomic
        write_json_atomic(note, {
            "video_id": video_id,
            "left_at": datetime.now().isoformat(timespec="seconds"),
            "why": why, "what_to_do": what_to_do, "kept": kept,
            "by": socket.gethostname()}, indent=2)
    except Exception as e:
        logger.error(f"{video_id}: could not write the note into {folder}: {e}")
    logger.warning(f"{video_id}: leftover folder {folder} kept where it is for a "
                   f"person ({why}); it holds {', '.join(kept)}"
                   + (f"; {len(deleted)} file(s) with a current copy elsewhere "
                      f"deleted" if deleted else ""))
    note_for_a_person(
        "left_for_a_person", video_id, folder,
        says=(f"{video_id}: the folder {folder} is not the video's live copy, but it "
              f"holds {', '.join(kept)}, which has no current copy anywhere else, so "
              f"it was left where it is (nothing a person wrote is ever deleted by "
              f"the machine). Why: {why}. What to do: {what_to_do} When nothing in "
              f"it is needed any more, press 'Nothing in it is needed: delete this "
              f"folder' (Review Queues tab, 'Folders left for a person', or "
              f"Resolve... here); this line then goes away by itself."))
    return kept


# --------------------------------------------------- lists for a person

def attention_dir() -> Optional[Path]:
    """<NAS root>/attention, where the watchers' lists for a person go
    (watcher/attention.py) -- or None when no NAS root is configured, or when
    the review root is not inside it. WHY the second: these lists describe the
    review queues under the review root (config derives it from the NAS
    root). A review root elsewhere is not the queues that attention folder
    speaks for -- a test pointing the queues at a temporary folder, say -- and
    writing there would replace the real lists with ones about other
    folders."""
    from ..config import Paths
    nas = getattr(Paths, "NAS_ROOT", None)
    if not nas:
        return None
    review = getattr(Paths, "REVIEW_ROOT", None)
    if review:
        try:
            n = os.path.normcase(os.path.abspath(str(nas))).rstrip("\\/")
            r = os.path.normcase(os.path.abspath(str(review)))
            if not (r == n or r.startswith(n + os.sep)):
                return None
        except (TypeError, ValueError):
            return None
    return Path(nas) / "attention"


def _list_name() -> str:
    return f"{socket.gethostname()}{ATTENTION_SUFFIX}"


def _read_items() -> List[Dict]:
    d = attention_dir()
    if d is None:
        return []
    try:
        data = json.loads((d / _list_name()).read_text(encoding="utf-8"))
        return list(data.get("items") or [])
    except (OSError, ValueError, AttributeError):
        return []


def _change_list(change) -> None:
    """Apply ``change`` (items -> items) to this machine's list of queue
    folders for a person. The read, change and write happen while holding the
    list's lock file (causal_review_io._locked_update), so the watcher, a
    review window and the dashboard on one machine never overwrite each
    other's update. Nothing is written when the change changes nothing.
    Never raises."""
    d = attention_dir()
    if d is None:
        return
    before = _read_items()
    if change(list(before)) == before:
        return
    try:
        from mousereach.review.causal_review_io import _locked_update

        def _apply(doc):
            doc["items"] = change(list(doc.get("items") or []))
            doc["written_by"] = socket.gethostname()
        _locked_update(d / _list_name(), _apply, wait_s=10.0)
    except Exception as e:
        logger.warning(f"Could not update the attention list {_list_name()}: {e}")


def note_for_a_person(kind: str, video_id: str, folder, says: str) -> None:
    """List ``folder`` for a person (one item per folder; a newer note of the
    same folder replaces the older). Kinds: 'left_for_a_person' (a leftover
    kept where it is, clear_leftover), 'queue_folder_in_the_way' (a route was
    refused because of the folder), 'return_blocked' (a return could not put
    the bundle's files in place). Never raises."""
    folder = str(folder)
    item = {"kind": kind, "needs_person": True, "video_id": video_id,
            "folder": folder, "says": says, "host": socket.gethostname()}
    _change_list(lambda items: [i for i in items if i.get("folder") != folder] + [item])


def note_folder_in_the_way(video_id: str, folder, says: str) -> None:
    """A route of ``video_id`` was refused because of ``folder`` (see
    note_for_a_person)."""
    note_for_a_person("queue_folder_in_the_way", video_id, folder, says)


def drop_for_a_person(video_id: Optional[str] = None, folder=None,
                      kind: Optional[str] = None) -> None:
    """Take items off this machine's list: those of ``video_id`` and/or of
    ``folder``, and only of ``kind`` when given. Never raises."""
    f = str(folder) if folder is not None else None

    def _keep(i):
        if kind is not None and i.get("kind") != kind:
            return True
        if video_id is not None and i.get("video_id") != video_id:
            return True
        if f is not None and i.get("folder") != f:
            return True
        return False
    _change_list(lambda items: [i for i in items if _keep(i)])


def clear_folder_in_the_way(video_id: str) -> None:
    """A route of ``video_id`` has gone through: its refusals are resolved."""
    drop_for_a_person(video_id=video_id, kind="queue_folder_in_the_way")


def refresh_for_a_person() -> None:
    """Drop items whose folder is gone, and leftovers whose note is gone (a
    person dealt with them). Only the listed folders are looked at, never the
    whole queue. Never raises."""
    def _still(i):
        try:
            p = Path(str(i.get("folder") or ""))
            if not p.exists():
                return False
            if i.get("kind") == "left_for_a_person":
                return left_for_a_person(p)
            return True
        except OSError:
            return True
    _change_list(lambda items: [i for i in items if _still(i)])


# ------------------------------------------- the leftovers, for the review tool

def leftover_folders() -> List[Dict]:
    """Every leftover kept for a person (a folder holding LEFT_NOTE) directly
    in either review queue, as attention-style items: ``kind``
    'left_for_a_person', ``video_id``, ``folder``, ``queue``, ``kept`` (the
    names the note lists) and ``says`` (why, and what to do, from the note).
    Read from the queue folders themselves, not from any machine's list, so a
    leftover is shown even when the list that named it was lost. Never
    raises; a queue that cannot be read gives nothing.

    WHY: a leftover holding a person's work stays where it is and must be
    listed in the review tool (the Review Queues tab's 'Folders left for a
    person'), with a way to deal with it there, not only in a log or on one
    machine's dashboard."""
    from ..config import Paths
    out: List[Dict] = []
    for queue, root in (("triage", getattr(Paths, "TRIAGE_REVIEW", None)),
                        ("deep review", getattr(Paths, "DEEP_REVIEW", None))):
        if not root:
            continue
        try:
            with os.scandir(root) as entries:
                folders = sorted(Path(e.path) for e in entries
                                 if not e.name.startswith(".") and e.is_dir())
        except OSError:
            continue
        for folder in folders:
            if not left_for_a_person(folder):
                continue
            try:
                note = json.loads((folder / LEFT_NOTE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                note = {}
            if not isinstance(note, dict):
                note = {}
            kept = [str(k) for k in (note.get("kept") or [])]
            out.append({
                "kind": "left_for_a_person", "needs_person": True,
                "video_id": folder.name, "folder": str(folder), "queue": queue,
                "kept": kept,
                "says": (f"{folder.name} ({queue} queue): the folder {folder} is not "
                         f"the video's live copy, but it holds "
                         f"{', '.join(kept) or 'files'} with no current copy anywhere "
                         f"else, so it was left where it is. Why: "
                         f"{note.get('why') or '(the note could not be read)'}. "
                         f"What to do: {note.get('what_to_do') or 'look at what it holds.'}"),
            })
    return out


def delete_leftover_folder(folder, video_id: str) -> List[str]:
    """Delete the leftover ``folder`` and everything in it, on a person's
    word (the review tool's 'Nothing in it is needed: delete this folder').
    Returns the names deleted. Raises ValueError, with nothing deleted, when
    ``folder`` is not a leftover kept for a person directly in a review queue
    under the video's name, and RuntimeError when another program is moving
    the video's folder right now.

    WHY so narrow: this is the one place that deletes a person's file, and
    only because a person looked and said so. It never deletes a live bundle
    (no note), a folder outside the queues, or a folder of another video."""
    from ..config import Paths
    folder = Path(folder)
    queues = [q for q in (getattr(Paths, "TRIAGE_REVIEW", None),
                          getattr(Paths, "DEEP_REVIEW", None)) if q]
    if folder.name != video_id or not any(same_path(folder.parent, q) for q in queues):
        raise ValueError(f"{folder} is not a folder of {video_id} directly in a review "
                         f"queue; nothing was deleted")
    if not left_for_a_person(folder):
        raise ValueError(f"{folder} holds no {LEFT_NOTE}: it is not a leftover kept "
                         f"for a person (it may be the video's live bundle); nothing "
                         f"was deleted")
    from .review_return import _claim_return_nested, _release_return_claim
    claimed, release = _claim_return_nested(video_id)
    if not claimed:
        raise RuntimeError(f"{video_id} is being moved by another program right now "
                           f"(the watcher, a review window or the route command); "
                           f"try again in a minute. Nothing was deleted.")
    try:
        if not left_for_a_person(folder):
            raise ValueError(f"{folder} is no longer a leftover kept for a person; "
                             f"nothing was deleted")
        names = sorted(p.name for p in folder.iterdir() if p.name != LEFT_NOTE)
        # The note goes LAST. WHY: a delete that stops part-way (a file held
        # open) must leave a folder that is still marked as a leftover; a
        # folder without its note would be taken for the video's live bundle
        # by every reader (is_queue_bundle).
        for name in names:
            p = folder / name
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()
        (folder / LEFT_NOTE).unlink()
        folder.rmdir()
    finally:
        if release:
            _release_return_claim(video_id)
    logger.warning(f"{video_id}: a person deleted the leftover folder {folder} "
                   f"(it held {', '.join(names) or 'nothing but its note'})")
    drop_for_a_person(folder=folder)
    return names
