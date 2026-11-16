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
full audit trail. The move is idempotent: re-routing a video that already has
a bundle replaces it.

A NEW bundle is never visible half-built. It is assembled in the hidden
``dest_root/.incoming/{stem}/`` and published with ONE directory rename, so a
queue reader sees either no bundle or the whole bundle. WHY: the move used to
create the empty ``{stem}`` folder first and then fill it file by file
(cross-volume copies take seconds); the running watcher's return scan saw the
empty folder, retired it to _Problematic mid-route, and every remaining move
failed with "No such file or directory".

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


# ---------------------------------------------------------------------------
# DEEP REVIEW SUPERSEDES TRIAGE
#
# A video in the deep-review queue is not allowed to be in the triage queue as
# well. Deep review is the stronger hold (the whole video, cuts included, is in
# question), so a triage copy of such a video is never created and, when one
# exists, it is deleted by the move into deep review (review_gate.route_to_queue)
# and ignored by every reader (census, dashboard, reconcile, the return scan,
# the review tools' triage list). WHY: a router that did not check sent videos
# already in deep review into triage as well; a reviewer then answered a triage
# copy holding older results while the deep-review copy was the one that counted.
# ---------------------------------------------------------------------------

def deep_review_stems() -> set:
    """Every video with a bundle in the deep-review queue, published or still
    being built there (``.incoming``). Empty when the queue is not configured
    or cannot be read."""
    from ..config import Paths
    root = getattr(Paths, "DEEP_REVIEW", None)
    out = set()
    if not root:
        return out
    for folder in (Path(root), Path(root) / INCOMING_DIR_NAME):
        try:
            with os.scandir(folder) as entries:
                for e in entries:
                    if not e.name.startswith(".") and e.is_dir():
                        out.add(e.name)
        except OSError:
            continue
    return out


def in_deep_review(video_id: str) -> bool:
    """Whether ``video_id`` has a bundle in the deep-review queue (published or
    being built). Never raises."""
    from ..config import Paths
    root = getattr(Paths, "DEEP_REVIEW", None)
    if not root:
        return False
    try:
        return ((Path(root) / video_id).is_dir()
                or (Path(root) / INCOMING_DIR_NAME / video_id).is_dir())
    except OSError:
        return False


def superseded_by_deep_review(bundle_dir) -> bool:
    """True when ``bundle_dir`` is a bundle in the TRIAGE queue whose video is
    also in deep review -- a copy every reader ignores (deep review supersedes
    triage). False for a deep-review bundle and on any error."""
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


def is_machine_output(video_id: str, name: str) -> bool:
    """A file the pipeline makes again whenever the video is analysed again
    (archive.supersede's list), or one of the queue's own records. Name only:
    a hand-cut segments file passes here -- is_person_file reads it."""
    from mousereach.archive.supersede import _ALGO_SUFFIXES
    if not name.startswith(video_id):
        return False
    rest = name[len(video_id):]
    return rest in _ALGO_SUFFIXES or rest in _QUEUE_RECORD_SUFFIXES


def is_person_file(path, video_id: str) -> bool:
    """True when ``path`` holds something a person did and must never be
    deleted or replaced without a person: any file of the video that is
    neither an input nor a machine output, and a segments file whose cuts a
    person set (``boundary_source`` 'human'; an unreadable segments file
    counts as a person's -- the safe side)."""
    path = Path(path)
    name = path.name
    if is_input_file(video_id, name):
        return False
    if not is_machine_output(video_id, name):
        return True
    if name == f"{video_id}_segments.json":
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("boundary_source") == "human"
        except (OSError, ValueError, AttributeError):
            return True
    return False


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

    def _move_in(src: Path) -> None:
        # Never replaces a person's file already in the bundle with a
        # different one (it stays at the source and is listed as failed).
        try:
            _move_keeping_person_files(src, build / src.name, video_id)
        except PersonFileInTheWay:
            raise
        except OSError:
            if not _published_by_other_route():
                raise
            _move_keeping_person_files(src, bundle / src.name, video_id)

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
    except Exception as err:
        logger.warning('Could not read %s, so it counts as absent: %s', p, err)
        return None


# ---------------------------------------------------------------------------
# A QUEUE FOLDER THAT IS NOT THE VIDEO'S LIVE BUNDLE LEAVES THE QUEUE
#
# A video is in ONE place at a time. A folder in a review queue that is not the
# video's live bundle -- an empty or records-only residue of a return, an old
# bundle of a video that has since been analysed again, the remains of a triage
# copy after the video moved into deep review -- must not stay in the queue:
# every later route of the video would meet it, and a reader would take it for
# a bundle. Nor may it be deleted when a person wrote something in it. So it is
# taken out of the queue WHOLE, in one folder rename (all or nothing: a folder
# holding a file that is open somewhere does not move at all), to
#
#   <review root>/_Problematic/_kept_for_a_person/<video>[__<time>]/
#       when it holds anything a person wrote (is_person_file, or a subfolder):
#       a note SET_ASIDE.json says where it came from, why, and what to do, and
#       it is listed in <NAS root>/attention/kept_for_a_person.json -- the
#       dashboard's "needs your attention" panel -- until a person deletes the
#       folder;
#   <review root>/_Problematic/_superseded_queue_folders_<date>/<video>[__<time>]/
#       when it holds only what the pipeline makes again (machine outputs,
#       inputs, queue records): kept for the record with the same note, not
#       listed (nothing in it needs a person).
#
# A folder that cannot be moved stays where it is, and the caller lists it in
# this machine's <NAS root>/attention/<host>_queue_folders_in_the_way.json
# (note_folder_in_the_way) until it is gone.
# ---------------------------------------------------------------------------

PROBLEMATIC_DIR_NAME = "_Problematic"
KEPT_FOR_A_PERSON_DIR = "_kept_for_a_person"
SUPERSEDED_DIR_PREFIX = "_superseded_queue_folders_"
SET_ASIDE_NOTE = "SET_ASIDE.json"
KEPT_ATTENTION_FILE = "kept_for_a_person.json"
IN_THE_WAY_ATTENTION_SUFFIX = "_queue_folders_in_the_way.json"


def person_entries(folder, video_id: str) -> List[str]:
    """Names in ``folder`` that a person wrote (is_person_file), plus every
    subfolder (unknown content counts as a person's -- the safe side)."""
    out = []
    for f in Path(folder).iterdir():
        if f.is_dir() or (f.is_file() and is_person_file(f, video_id)):
            out.append(f.name)
    return sorted(out)


def kept_for_a_person_root() -> Optional[Path]:
    from ..config import Paths
    root = getattr(Paths, "REVIEW_ROOT", None)
    return Path(root) / PROBLEMATIC_DIR_NAME / KEPT_FOR_A_PERSON_DIR if root else None


def set_aside_queue_folder(folder, video_id: str, why: str, what_to_do: str) -> Path:
    """Take ``folder`` out of its queue whole (see the block comment above) and
    return where it went. Raises OSError, with nothing moved, when it cannot
    be moved (no review root configured, a file in it held open ...)."""
    from ..config import Paths
    from mousereach.pipeline.fsutil import retry_transient, write_json_atomic
    folder = Path(folder)
    root = getattr(Paths, "REVIEW_ROOT", None)
    if not root:
        raise OSError("no review root is configured; nowhere to set the folder aside")
    # Only a folder of the queues under this review root, checked before
    # anything is created. WHY: the move is one rename, which only works
    # within one volume, and a folder from elsewhere (a queue configured
    # apart, a test's temporary folder) must not create folders in the real
    # _Problematic on its way to failing.
    try:
        r = os.path.normcase(os.path.abspath(str(root))).rstrip("\\/")
        f = os.path.normcase(os.path.abspath(str(folder)))
        inside = f.startswith(r + os.sep)
    except (TypeError, ValueError):
        inside = False
    if not inside:
        raise OSError(f"{folder} is not inside the review root {root}; not set aside")
    person = person_entries(folder, video_id)
    now = datetime.now()
    parent = (Path(root) / PROBLEMATIC_DIR_NAME
              / (KEPT_FOR_A_PERSON_DIR if person
                 else f"{SUPERSEDED_DIR_PREFIX}{now:%Y-%m-%d}"))
    dest = parent / video_id
    if dest.exists():
        dest = parent / f"{video_id}__{now:%Y%m%d_%H%M%S_%f}"
    parent.mkdir(parents=True, exist_ok=True)
    retry_transient(lambda: os.rename(str(folder), str(dest)), what=folder.name)
    note = {"video_id": video_id, "set_aside_at": now.isoformat(timespec="seconds"),
            "from": str(folder), "why": why, "what_to_do": what_to_do,
            "person_files": person, "by": socket.gethostname()}
    try:
        write_json_atomic(dest / SET_ASIDE_NOTE, note, indent=2)
    except Exception as e:
        logger.error(f"{video_id}: {folder} was set aside to {dest}, but its note "
                     f"could not be written ({e})")
    logger.warning(f"{video_id}: {folder} taken out of its review queue to {dest} "
                   f"({why})" + (f"; it holds a person's files ({', '.join(person)}) "
                                 f"and is listed for a person" if person else ""))
    if person:
        refresh_kept_attention()
    return dest


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


def _write_attention_list(name: str, items: List[Dict]) -> None:
    """Write <NAS root>/attention/<name> (watcher/attention.py) when its items
    changed; an empty list is written only over an earlier one. Never raises."""
    d = attention_dir()
    if d is None:
        return
    try:
        from mousereach.pipeline.fsutil import write_json_atomic
        target = d / name
        try:
            old = json.loads(target.read_text(encoding="utf-8")).get("items")
        except (OSError, ValueError, AttributeError):
            old = None
        if old == items or (not items and not target.exists()):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target, {"written_by": socket.gethostname(), "items": items},
                          indent=1)
    except Exception as e:
        logger.warning(f"Could not write the attention list {name}: {e}")


def refresh_kept_attention() -> None:
    """List every folder under _kept_for_a_person in
    <NAS root>/attention/kept_for_a_person.json. ONE shared list, not one per
    machine: it is recomputed from the shared folder, so every machine would
    write the same items. A folder a person deleted drops off on the next
    refresh (the return scan refreshes it every ROUTE_REPAIR_INTERVAL_S).
    Never raises."""
    root = kept_for_a_person_root()
    items: List[Dict] = []
    try:
        folders = (sorted(p for p in root.iterdir() if p.is_dir())
                   if root and root.is_dir() else [])
    except OSError:
        return
    for d in folders:
        try:
            note = json.loads((d / SET_ASIDE_NOTE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            note = {}
        vid = note.get("video_id") or d.name.split("__")[0]
        theirs = ", ".join(note.get("person_files") or []) or "see the folder"
        items.append({
            "kind": "kept_for_a_person", "needs_person": True, "video_id": vid,
            "folder": str(d),
            "says": (f"{vid}: a folder holding files a person wrote ({theirs}) was "
                     f"taken out of the review queue and kept in {d}. Why: "
                     f"{note.get('why') or 'see SET_ASIDE.json in the folder'}. "
                     f"What to do: "
                     f"{note.get('what_to_do') or 'decide what is still needed.'} "
                     f"Then delete the folder; this item goes away by itself."),
        })
    _write_attention_list(KEPT_ATTENTION_FILE, items)


def _in_the_way_name() -> str:
    return f"{socket.gethostname()}{IN_THE_WAY_ATTENTION_SUFFIX}"


def _in_the_way_items() -> List[Dict]:
    d = attention_dir()
    if d is None:
        return []
    try:
        data = json.loads((d / _in_the_way_name())
                          .read_text(encoding="utf-8"))
        return list(data.get("items") or [])
    except (OSError, ValueError, AttributeError):
        return []


def note_folder_in_the_way(video_id: str, folder, says: str) -> None:
    """List, for a person, a queue folder that stopped a route or could not be
    taken out of its queue. One item per folder, kept until the folder is gone
    (refresh_folders_in_the_way) or the video is routed
    (clear_folder_in_the_way). Never raises."""
    folder = str(folder)
    items = [i for i in _in_the_way_items() if i.get("folder") != folder]
    items.append({"kind": "queue_folder_in_the_way", "needs_person": True,
                  "video_id": video_id, "folder": folder, "says": says})
    _write_attention_list(_in_the_way_name(), items)


def clear_folder_in_the_way(video_id: str) -> None:
    """Drop ``video_id``'s items from this machine's in-the-way list (a route
    of the video has gone through). Never raises."""
    items = _in_the_way_items()
    kept = [i for i in items if i.get("video_id") != video_id]
    if kept != items:
        _write_attention_list(_in_the_way_name(), kept)


def refresh_folders_in_the_way() -> None:
    """Drop items whose folder no longer exists (a person dealt with it).
    Never raises."""
    items = _in_the_way_items()
    kept = []
    for i in items:
        try:
            if Path(str(i.get("folder") or "")).exists():
                kept.append(i)
        except OSError:
            kept.append(i)
    if kept != items:
        _write_attention_list(_in_the_way_name(), kept)
