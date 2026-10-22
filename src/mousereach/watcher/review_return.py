"""Return path: re-inject a human-cleared held video into the pipeline.

The forward gate (review_gate) MOVES un-committable videos into the TRIAGE or
DEEP_REVIEW queue and holds them out of kinematics + connectome.db. This module
is the other half: it scans those queues for bundles a human has cleared and
moves them back into the local Processing dir, set to ``processing`` state, so
the pipeline re-runs them (Priority-2 'pipeline' work). On the re-run the review
file travels with the bundle, so the gate re-checks and -- now that every
triaged element is resolved -- lets the video through to kinematics.

Clear signals:
  TRIAGE      -- every triaged element resolved (``triage_status.fully_resolved``
                 and segmentation not failed). Works with the existing review
                 tool today: resolve all elements -> next scan re-injects.
                 EXCEPTION: if the reviewer set ``true_segment_num`` anywhere
                 (declared a segment mislabeled), the video is DIVERTED to the
                 DEEP_REVIEW queue for manual re-segmentation instead of being
                 re-injected -- re-running with the wrong boundaries would
                 compute kinematics over the wrong frames.
  DEEP_REVIEW -- an explicit ``{stem}_deep_review_cleared.json`` marker (written
                 by the deep tools when a reviewer clears the flag) OR a
                 co-located ``{stem}_unified_ground_truth.json`` (the deep GT
                 tool produced the answer). Re-injection restarts the pipeline
                 from segmentation.

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import functools
import os
import shutil
import logging
import socket
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..config import Paths
from . import claim_marker, node_status
from ..review.triage_status import triage_status
from .review_routing import _safe_move

try:
    from .review_routing import INCOMING_DIR_NAME
except ImportError:  # a review_routing that predates the staging folder
    INCOMING_DIR_NAME = ".incoming"

logger = logging.getLogger(__name__)

# Queue-only metadata that must NOT follow the bundle back into Processing,
# matched by EXACT name ({stem} + suffix, or the bare name). WHY exact: a plain
# endswith("_manifest.json") also matched {stem}_processing_manifest.json -- the
# provenance record -- and deleted it from every returned bundle. Every writer
# of queue metadata uses exactly these names (review_gate and review staging
# write {stem}_manifest.json, review_routing writes {stem}_routing.json).
# Keeping the old processing manifest is the lesser risk, not a free one. Stage
# reuse is decided from the stage outputs, never from this file, and a normal
# re-run regenerates it before the review gate. But that regeneration is
# best-effort (a failure is only a warning in the pipeline runners), and a
# segmentation failure is routed to deep review before it runs: in those cases
# the OLD manifest stays beside the new outputs, and kinematics stamps its
# version onto it. Deleting it lost the provenance of EVERY return, which is
# worse. Setting a stale manifest aside before regenerating it is not yet done.
_QUEUE_METADATA_STEM_SUFFIXES = ("_manifest.json", "_routing.json")
_QUEUE_METADATA_NAMES = ("manifest.json",)


def _declared_scorer() -> str:
    """The pose model the pipeline currently declares, or "" when unknown."""
    try:
        from mousereach.pipeline.versions import get_current_versions
        return ((get_current_versions(Paths.NAS_ROOT) or {})
                .get("versions") or {}).get("dlc_scorer") or ""
    except Exception:
        return ""


def _resolve_inputs(bundle: Path, stem: str, processing_dir=None,
                    recorded_pose=None):
    """Find the mp4 and pose file for a bundle that is being returned.

    Gate-routed bundles (review_gate -> review_routing) ARE self-contained: the
    mp4 and pose moved into the bundle with the algo JSONs, beside a
    bundle-local ``{stem}_manifest.json``. Only LEGACY staged bundles carry the
    algo JSONs alone, with a ``{stem}_manifest.json`` pointing at the mp4 and
    pose in Analyzed. Returning a bundle without resolving the inputs means the
    pipeline re-runs the video with no pose at all -- which is how 723 videos
    ended up in the deep-review queue as "segmentation_failed" on 2026-08-19.

    Note this must run BEFORE the bundle is emptied: ``{stem}_manifest.json`` is
    queue metadata (_is_queue_metadata), so the move loop deletes the very file
    that says where a legacy bundle's inputs live.

    Looks in: the bundle itself -> the manifest's canonical paths -> Analyzed,
    by stem -> pose staging -> ``processing_dir`` (where the return delivers the
    bundle) and ``recorded_pose`` (the pose the video's row names). Nothing is
    copied; the returned paths are used as-is, so a pose sitting in Analyzed is
    read from there rather than duplicated locally.

    Returns:
        (mp4_path_or_None, pose_path_or_None)
    """
    import json
    from mousereach.pipeline.manifest import select_pose_file

    def _first_file(paths):
        for p in paths:
            if p and Path(p).is_file():
                return Path(p)
        return None

    mp4 = _first_file(bundle.glob(f"{stem}.mp4"))
    pose_hits = [p for p in bundle.glob(f"{stem}DLC*.h5") if p.is_file()]
    pose = select_pose_file(pose_hits) if pose_hits else None

    if mp4 is None or pose is None:
        manifest = bundle / f"{stem}_manifest.json"
        if manifest.is_file():
            try:
                d = json.loads(manifest.read_text())
                mp4 = mp4 or _first_file([d.get('canonical_video_path')])
                pose = pose or _first_file([d.get('canonical_dlc_h5_path')])
            except Exception as e:
                logger.debug(f"{stem}: could not read bundle manifest: {e}")

    if (mp4 is None or pose is None) and Paths.ANALYZED_OUTPUT:
        root = Path(Paths.ANALYZED_OUTPUT)
        # Walk Analyzed without entering Analyzed/Archive/: superseded files
        # keep their original names there, so an rglob returned a reviewed
        # video on an archived mp4 / old-generation pose as if it were live.
        # DLC Model <N>/ folders stay visible -- they are live pose storage.
        from mousereach.pipeline.analyzed_tree import first_file, iter_files
        try:
            if mp4 is None:
                mp4 = first_file(root, f"{stem}.mp4")
            if pose is None:
                hits = list(iter_files(root, f"{stem}DLC*.h5"))
                pose = select_pose_file(hits) if hits else None
        except OSError as e:
            logger.debug(f"{stem}: could not search Analyzed: {e}")

    # Fourth and last: the Processing/Posed staging area. A GPU node's freshly
    # staged pose sits there before any intake pulls it; a video routed to a
    # queue in that window (or whose bundle omitted its h5) otherwise refuses
    # to return in a loop -- one did, every ~2-3 minutes, 2026-09-08. Only a
    # DECLARED-scorer pose is accepted from staging: an old-model pose here
    # is stale by definition, and silently posing with it is the wrong-pose
    # failure this resolver exists to prevent.
    if pose is None and getattr(Paths, "DLC_STAGING", None):
        try:
            staging = Path(Paths.DLC_STAGING)
            hits = [p for p in staging.glob(f"{stem}DLC*.h5") if p.is_file()]
            if hits:
                from mousereach.pipeline.versions import get_current_versions
                declared = ((get_current_versions(Paths.NAS_ROOT) or {})
                            .get("versions") or {}).get("dlc_scorer") or ""
                cur = [p for p in hits if declared and declared in p.name]
                if cur:
                    pose = select_pose_file(cur)
            if mp4 is None:
                cand = staging / f"{stem}.mp4"
                if cand.is_file():
                    mp4 = cand
        except OSError as e:
            logger.debug(f"{stem}: could not search staging: {e}")

    # Fifth: this node's own processing folder -- where the return is about to
    # put the bundle -- and the pose the video's row records. WHY: an older
    # divert between queues moved only the mp4 and left the pose and every
    # result in the processing folder of the node that ran the video. Two
    # human-cleared videos could then never return: each scan searched the
    # bundle, its manifest, Analyzed and staging, found nothing, and tried
    # again two minutes later -- thousands of times from 2026-09-28 -- while
    # the pose sat in the folder the return delivers to. Declared-scorer only,
    # the staging rule: an old-model pose left behind here is stale.
    if pose is None and (processing_dir or recorded_pose):
        declared = _declared_scorer()
        cands = []
        if recorded_pose:
            cands.append(Path(recorded_pose))
        if processing_dir:
            try:
                cands += list(Path(processing_dir).glob(f"{stem}DLC*.h5"))
            except OSError as e:
                logger.debug(f"{stem}: could not search {processing_dir}: {e}")
        cur = [c for c in cands if declared and declared in c.name and c.is_file()]
        if cur:
            pose = select_pose_file(cur)
    if mp4 is None and processing_dir:
        cand = Path(processing_dir) / f"{stem}.mp4"
        if cand.is_file():
            mp4 = cand

    return mp4, pose


# A return that cannot go ahead -- no pose anywhere, or files of the video
# still in the queue's build folder -- is not tried again until this long has
# passed. WHY: nothing a scan does changes either condition, and retrying every
# two minutes cost seconds of every scan (5 s per such bundle, searching
# Analyzed) and buried the log under thousands of identical ERRORs. In memory:
# a restart tries each once more, which is what a person fixing it would want.
RETURN_RETRY_S = 3600
_return_retry_after: Dict[str, float] = {}


def _return_backing_off(stem: str) -> bool:
    until = _return_retry_after.get(stem)
    return until is not None and time.monotonic() < until


def _return_failed(stem: str) -> None:
    _return_retry_after[stem] = time.monotonic() + RETURN_RETRY_S


def _bundles(queue_root: Optional[Path]) -> List[Path]:
    # Dot-directories are never bundles (a stray .omc/ in both queues was
    # iterated as a phantom video every scan on 2026-09-01, and -- having no
    # segments file -- read as seg_failed and got divert-retried forever).
    if not queue_root or not Path(queue_root).exists():
        return []
    found = [d for d in Path(queue_root).iterdir()
             if d.is_dir() and not d.name.startswith(".")]
    # Only a bounded number of these re-enter the pipeline per scan
    # (MAX_RETURNS_PER_SCAN), so WHICH ones is a real decision, not a detail:
    # raw directory order returned whatever the filesystem happened to list
    # first, while every other part of the watcher worked to the lab's order.
    # Cleared videos now come back in that same order.
    try:
        from mousereach.watcher.work_priority import order_items
        return order_items(found)
    except Exception:
        return sorted(found)


# Each scan read every triage bundle's segments, outcomes, assignments and
# review files to decide whether it was cleared: 7-8 s of every scan, every two
# minutes, for a queue that mostly does not change between scans (2026-10-21).
# A bundle's verdict depends only on its own files and on the video's ground
# truth (triage_status), so it is reused until either changes: the bundle's
# files are listed with their sizes and modified times (one listing per bundle,
# about 2 s for the whole queue) and the ground-truth file's path and time.
_status_cache: Dict[str, tuple] = {}


def _bundle_signature(bundle: Path) -> tuple:
    """(name, size, modified time) of every entry in ``bundle`` -- folders too,
    so an empty signature means an empty folder exactly as ``any(iterdir())``
    did. Raises OSError."""
    out = []
    with os.scandir(bundle) as it:
        for e in it:
            st = e.stat()
            out.append((e.name, st.st_size if e.is_file() else -1, st.st_mtime_ns))
    return tuple(sorted(out))


# (bundle signature, twin signature) -> verdict of _is_leftover_copy, so a
# duplicate that is NOT a leftover is hashed once, not on every scan.
_leftover_cache: Dict[str, tuple] = {}


def _file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _is_leftover_copy(bundle: Path, twin: Path, signature: tuple) -> bool:
    """True when every entry of ``bundle`` is a plain file with a byte-identical
    file of the same name in ``twin``.

    WHY: historical escalations split a bundle across the queues -- the jsons
    and pose moved to deep review, the mp4 stayed in triage. What remains in
    triage then holds nothing the deep-review copy lacks, so archiving it loses
    nothing, while leaving it made every return scan log the same "exists in
    BOTH queues" line for weeks (20250808_CNT0113_P1, 231 times from
    2026-10-19) with nobody ever doing the "deliberate cleanup" it asked for.
    Sizes are compared before any content is read; a copy that differs in
    anything stays for a person. Raises OSError.
    """
    twin_sig = _bundle_signature(twin)
    key = (signature, twin_sig)
    hit = _leftover_cache.get(str(bundle))
    if hit is not None and hit[0] == key:
        return hit[1]
    twin_sizes = {name: size for name, size, _ in twin_sig}
    verdict = bool(signature) and all(
        size >= 0 and twin_sizes.get(name) == size
        for name, size, _ in signature)
    if verdict:
        verdict = all(_file_sha256(bundle / name) == _file_sha256(twin / name)
                      for name, _, _ in signature)
    _leftover_cache[str(bundle)] = (key, verdict)
    return verdict


def _gt_signature(stem: str) -> tuple:
    try:
        from mousereach.review.causal_review_io import find_gt
        g = find_gt(stem)
        return (str(g), Path(g).stat().st_mtime_ns) if g else ()
    except Exception:
        return ()


def _cached_triage_status(bundle: Path, stem: str, signature: tuple):
    key = (signature, _gt_signature(stem))
    hit = _status_cache.get(str(bundle))
    if hit is not None and hit[0] == key:
        return hit[1]
    st = triage_status(bundle, stem)
    _status_cache[str(bundle)] = (key, st)
    return st


def _is_queue_metadata(name: str, stem: str) -> bool:
    return (name in _QUEUE_METADATA_NAMES
            or any(name == f"{stem}{s}" for s in _QUEUE_METADATA_STEM_SUFFIXES))


def _ensure_registered(db, stem: str, source_path: Path) -> bool:
    """Give this node a row for ``stem`` if it has none. False (and the bundle
    must stay in its queue) when that fails.

    WHY before touching the queue: a node that has never seen the video (fresh
    DB, or the review was cleared on a different machine) has no row to update
    -- and the return used to move the bundle's files out, delete the bundle,
    and only then discover it could not record the state. The clearance was
    consumed, the files landed in Processing with nothing referencing them, and
    the function returned True. 274 bundles went that way in one run before it
    was caught."""
    try:
        if db.get_video(stem) is None:
            # Parse the stem's own metadata into the row. Rows registered here
            # used to carry only the id and path -- no tray_type/date/animal --
            # and tray-aware work selection silently degrades to a random pick
            # on a NULL tray_type. This stops NEW metadata-less rows; rows that
            # already exist with NULLs are handled by the orchestrator's
            # read-side filename fallback -- both halves are required.
            meta = {}
            try:
                from .validator import validate_single_filename
                result = validate_single_filename(f"{stem}.mp4")
                if result.valid and result.parsed:
                    meta = {
                        'date': result.parsed['date'],
                        'animal_id': result.parsed['animal_id'],
                        'experiment': result.parsed['experiment'],
                        'cohort': result.parsed['cohort'],
                        'subject': result.parsed['subject'],
                        'tray_type': result.parsed['tray_type'],
                    }
            except Exception:
                meta = {}
            db.register_video(video_id=stem, source_path=str(source_path), **meta)
            logger.info(f"Return {stem}: registered (not previously known to this node)")
        return True
    except Exception as e:
        logger.error(
            f"Return {stem}: cannot register video, leaving the bundle in the "
            f"queue for a later attempt: {e}"
        )
        return False


def _deep_review_cleared(bundle: Path, stem: str) -> bool:
    """True if a reviewer has cleared this deep-review bundle."""
    if (bundle / f"{stem}_deep_review_cleared.json").exists():
        return True
    # The deep GT tool produced the answer in place.
    if (bundle / f"{stem}_unified_ground_truth.json").exists():
        return True
    return False


def _ensure_durable_review(bundle: Path, stem: str) -> None:
    """Copy this bundle's human review to the durable store before the bundle is
    emptied. Never raises; a failure is logged and the return still proceeds,
    because refusing to return the video would strand the reviewer's clearance.
    """
    src = Path(bundle) / f"{stem}_causal_review.json"
    if not src.is_file():
        return
    try:
        from mousereach.review.causal_review_io import durable_review_path
        dest = durable_review_path(stem)
        if dest is None:
            logger.warning(
                "Return %s: no durable review store configured; the review is "
                "about to move to this node's local disk only.", stem)
            return
        if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        logger.info(f"Return {stem}: human review copied to the durable store")
    except Exception as e:
        logger.error(
            "Return %s: could NOT make a durable copy of the human review (%s). "
            "After this return its only copy is on this node's local disk.",
            stem, e)


# --- cross-node bundle claims -------------------------------------------
# With several watchers scanning the SAME shared queues, two nodes that spot
# the same cleared bundle in one pass would both start moving its files and
# tear the bundle in half across machines -- the split-bundle failure class,
# cross-node edition. Claimed only at MUTATION time (returns, diverts,
# retires), never for mere scanning, with the same exclusive-create marker the
# posed-video intake uses (watcher/claim_marker.py): exactly one node's create
# succeeds. A claim is broken only when its holder has stopped reporting
# (claim_marker.GONE_AFTER_S, by the file server's clock), never for its age.
# The rule this replaced was a check, write, sleep and read-back (not
# atomic), went ahead unclaimed when the claim folder could not be made, and
# broke another node's marker after two hours by age alone.


def _return_claims_dir():
    root = getattr(Paths, "REVIEW_ROOT", None)
    return (Path(root) / ".return_claims") if root else None


def _claim_return(stem: str) -> bool:
    d = _return_claims_dir()
    if d is None:
        return True                            # no shared queues configured
    host = socket.gethostname()
    if claim_marker.take(d, stem, host, suffix=".claim"):
        return True
    # Held by another node. If that node has stopped running, its claim is a
    # crashed move's leftover: break it and try once more.
    if stem in claim_marker.sweep_gone(d, host, node_status.status_dir(Paths.NAS_ROOT),
                                       suffix=".claim"):
        return claim_marker.take(d, stem, host, suffix=".claim")
    return False


def _release_return_claim(stem: str) -> None:
    d = _return_claims_dir()
    if d is None:
        return
    claim_marker.release(d, stem, socket.gethostname(), suffix=".claim")


def _with_return_claim(fn):
    """Guard a bundle-mutating helper with the cross-node claim."""
    @functools.wraps(fn)
    def wrapper(bundle, stem, *a, **k):
        if not _claim_return(stem):
            logger.debug(f"{stem}: bundle claimed by another node; skipping")
            return False
        try:
            return fn(bundle, stem, *a, **k)
        finally:
            _release_return_claim(stem)
    return wrapper


@_with_return_claim
def _return_to_processing(bundle: Path, stem: str, processing_dir: Path, db,
                          reason: str) -> bool:
    """Move a cleared bundle's data files into ``processing_dir`` and set the
    video to ``processing`` so the pipeline re-runs it. Queue-only metadata
    (manifest / routing) is dropped so the bundle disappears from the queue.
    Returns True on success."""
    # Refuse while this video still has files in the queue's build folder
    # (<queue>/.incoming/<stem>). WHY: a route whose publish fell back to moving
    # files in one by one leaves a file it could not move there -- most likely
    # the mp4, held open by a scanner or indexer -- beside the published
    # bundle, and later routes write straight into the visible bundle and never
    # read staging. Returning now would re-run the video without that file (the
    # Analyzed search no longer finds it either). A person folds it in first.
    leftover = Path(bundle).parent / INCOMING_DIR_NAME / stem
    try:
        stuck = sorted(p.name for p in leftover.iterdir() if p.is_file())
    except OSError:
        stuck = []
    if stuck:
        logger.error(
            f"Return {stem}: {len(stuck)} file(s) of this video are still in "
            f"{leftover} ({', '.join(stuck)}), left there by a route that did "
            f"not finish. Leaving the bundle in the queue -- move them into "
            f"{bundle} first. (Tried again in {RETURN_RETRY_S // 60} min.)")
        _return_failed(stem)
        return False

    processing_dir = Path(processing_dir)
    processing_dir.mkdir(parents=True, exist_ok=True)

    # Make sure the DB can hold this video BEFORE touching the queue (see
    # _ensure_registered for why).
    if not _ensure_registered(db, stem, processing_dir / f"{stem}.mp4"):
        return False

    # Resolve the inputs BEFORE emptying the bundle -- the move loop deletes
    # _manifest.json, which is what names them. Returning a video whose pose
    # cannot be found guarantees the pipeline fails on it, so refuse and leave
    # the clearance in the queue rather than spend it on a doomed run.
    try:
        recorded_pose = (db.get_video(stem) or {}).get('dlc_output_path')
    except Exception:
        recorded_pose = None
    mp4_src, pose_src = _resolve_inputs(bundle, stem, processing_dir=processing_dir,
                                        recorded_pose=recorded_pose)
    if pose_src is None:
        logger.error(
            f"Return {stem}: no pose file of the current model found (bundle, "
            f"manifest, Analyzed, staging, or this node's processing folder). "
            f"Leaving the bundle in the queue -- returning it would re-run the "
            f"video with no pose. (Tried again in {RETURN_RETRY_S // 60} min.)"
        )
        _return_failed(stem)
        return False

    # This function is where a review leaves shared storage: every bundle file is
    # MOVED onto this node's local processing dir and the bundle directory is then
    # removed. That is the moment a reviewer's answers stop being visible to any
    # other machine, and if this node's processing dir is later cleared -- or the
    # video never reaches the canonical results dir -- the review is gone.
    #
    # Kinematics reads the review AFTER this point, so losing it here means the
    # human's outcome and causal reach never reach the data product. Make the
    # durable copy before touching anything.
    _ensure_durable_review(bundle, stem)

    # dlc_output_path must name the pose _resolve_inputs SELECTED. A bundle can
    # hold two poses (a re-pose leaves the old model's file beside the new one),
    # and recording "whichever .h5 was moved last" re-ran such videos on the
    # wrong -- often the old -- pose. A selected pose inside the bundle is
    # recorded at its destination; one outside the bundle (legacy manifest,
    # Analyzed, staging) is used in place, so pose_src stands.
    h5_dest: Optional[Path] = pose_src
    moved = 0
    for f in list(bundle.iterdir()):
        if not f.is_file():
            continue
        if _is_queue_metadata(f.name, stem):
            try:
                f.unlink()
            except OSError:
                pass
            continue
        dest = processing_dir / f.name
        try:
            _safe_move(f, dest)
            moved += 1
            if Path(f) == Path(pose_src):
                h5_dest = dest
        except Exception as e:
            logger.warning(f"Return {stem}: could not move {f.name}: {e}")
            return False

    if moved == 0:
        return False

    kw = {'dlc_output_path': str(h5_dest)} if h5_dest is not None else {}
    try:
        # NOTE FOR ANYONE REASONING ABOUT THE 'processing' POOL: this is a
        # SECOND ENTRY POINT into 'processing', beside intake. Any guard or
        # invariant of the form "videos in processing got there through
        # intake, so a check at intake is sufficient" is wrong because of
        # this path -- a human-cleared review re-enters here, bypassing
        # intake entirely. (An orchestrator work-selection guard was designed
        # on that assumption and had to be redone; hence this sign.)
        #
        # A cleared review routinely returns a video whose row has no legal
        # path to 'processing' (typically 'archived': the bundle sat in a
        # review queue while the row was closed out). That is exactly what
        # force_state exists for -- go straight there instead of letting
        # update_state log an ERROR first and then forcing anyway. The WARNING
        # force_state writes is the honest audit line; the ERROR was noise
        # that fired on every legitimate return.
        state = None
        try:
            row = db.get_video(stem)
            state = row['state'] if row else None
        except Exception:
            state = None
        from .db import VIDEO_TRANSITIONS
        if state is not None and "processing" not in VIDEO_TRANSITIONS.get(state, []):
            db.force_state(stem, "processing",
                           reason="review cleared; returned to Processing", **kw)
        else:
            db.update_state(stem, "processing", **kw)
    except Exception:
        # The row exists (registered above), so a legal-transition failure is
        # what lands here. force_state is the documented escape hatch and logs
        # the bypass; without it the video would sit in Processing unreferenced.
        try:
            db.force_state(stem, "processing",
                           reason="review cleared; returned to Processing", **kw)
        except Exception as e2:
            # Do NOT delete the bundle -- the files are already in Processing but
            # the queue entry is the only remaining pointer to this work.
            logger.error(
                f"Return {stem}: could not set 'processing' state ({e2}). "
                f"Bundle kept in the queue so the clearance is not lost."
            )
            return False

    # Remove the now-empty bundle dir so it leaves the queue.
    try:
        bundle.rmdir()
    except OSError:
        pass

    _return_retry_after.pop(stem, None)
    logger.info(f"Returned {stem} to Processing ({moved} files) reason={reason}")
    return True


@_with_return_claim
def _retire_stale_bundle(bundle: Path, stem: str, video_state: str,
                         why: str = None) -> bool:
    """Move a stale queue bundle into the _Problematic archive (never delete).

    A bundle is STALE when the pipeline has re-handled its video since it was
    staged -- the video's live state says processing/processed/archiving/
    archived while the bundle still carries the old era's flags. Leaving it
    would re-divert (and re-flip a healthy video) on every scan; deleting it
    would destroy provenance. Archive it beside the other reconciled bundles,
    with a note saying why. Never raises; False means the bundle stayed put
    (it will be re-attempted next scan).
    """
    import json
    from datetime import datetime

    try:
        root = Paths.REVIEW_ROOT
        if not root:
            return False
        dest = (Path(root) / "_Problematic"
                / f"_stale_seg_failed_bundles_{datetime.now():%Y-%m-%d}" / stem)
        dest.mkdir(parents=True, exist_ok=True)
        note = {
            "retired_at": datetime.now().isoformat(),
            "video_state_at_retirement": video_state,
            "why": why or (
                    "triage bundle carried seg-failed flags from an earlier "
                    "era, but the pipeline has since re-handled this video; "
                    "diverting it would have flipped a healthy video into "
                    "deep_review and blocked its archive"),
        }
        (dest / f"{stem}_stale_retire.json").write_text(
            json.dumps(note, indent=2), encoding="utf-8")
        for f in list(bundle.iterdir()):
            if f.is_file():
                _safe_move(f, dest / f.name)
        bundle.rmdir()
        return True
    except Exception as e:
        logger.warning("Return scan: could not retire stale bundle %s: %s",
                       stem, e)
        return False


# ---------------------------------------------------------------------------
# Machine-only quality holds the current check would not make
# ---------------------------------------------------------------------------
# A deep-review bundle routed "qc_needs_review" is held by the machine's own
# quality check, not by a person. When that check is corrected, a hold it
# would no longer make is a machine error sitting in a person's queue. WHY
# (2026-10-22): the check demanded an outcomes file Easy and Flat trays never
# get, and 469 Easy/Flat bundles were held in deep review for that alone.
#
# So a bundle with that reason and no file a person wrote has the check re-run
# on it. The pose-tracking layer is NOT re-run (it would read every pose file
# over the share): its flags are carried over from the bundle's stored
# verdict, so a hold that layer made stays a hold. The file-based layers
# (structural, cross-step, statistical) are recomputed. If nothing critical
# remains, the video goes back to the posed folder and re-runs in its priority
# turn -- NOT straight into 'processing', because anything there blocks intake
# (max_local_pending), and released low-priority videos would hold up new
# work. The bundle's machine outputs are archived under _Problematic, never
# deleted.
QC_RECHECK_BUDGET_S = 30            # seconds of re-checks per scan
_qc_recheck_cache: Dict[str, tuple] = {}
_PERSON_FILE_SUFFIXES = ("_causal_review.json", "_deep_review_cleared.json",
                         "_unified_ground_truth.json")


def _routed_reason(bundle: Path, stem: str) -> str:
    import json
    try:
        return (json.loads((bundle / f"{stem}_routing.json").read_text(encoding="utf-8"))
                .get("routed_reason") or "")
    except Exception:
        return ""


def _person_touched(signature: tuple) -> bool:
    return any(name.endswith(_PERSON_FILE_SUFFIXES) or "ground_truth" in name
               for name, _, _ in signature)


def _qc_hold_lifted(bundle: Path, stem: str, signature: tuple) -> Optional[bool]:
    """True when the bundle was held by the quality check alone and the current
    check would not hold it, False when it still would (or the hold was for
    another reason), None when it cannot tell (no stored verdict). Remembered
    per bundle until the bundle changes, so a scan reads each bundle once."""
    import json
    hit = _qc_recheck_cache.get(str(bundle))
    if hit is not None and hit[0] == signature:
        return hit[1]
    verdict: Optional[bool]
    try:
        if _routed_reason(bundle, stem) != "qc_needs_review":
            verdict = False
            _qc_recheck_cache[str(bundle)] = (signature, verdict)
            return verdict
        stored = json.loads((bundle / f"{stem}_triage.json").read_text(encoding="utf-8"))
        if any(f.get("category") == "dlc_coherence" and f.get("severity") == "critical"
               for f in stored.get("flags", [])):
            verdict = False
        else:
            from mousereach.pipeline.triage import triage_video
            verdict = triage_video(stem, bundle, skip_dlc_check=True).n_critical == 0
    except Exception:
        verdict = None
    _qc_recheck_cache[str(bundle)] = (signature, verdict)
    return verdict


@_with_return_claim
def _release_to_posed(bundle: Path, stem: str, db) -> bool:
    """Send a lifted hold back to the posed folder for a fresh run (see the
    block comment above). The mp4 and pose files go to the posed folder; every
    other file goes to _Problematic/_released_qc_holds_<date>/<stem>/ with a
    note. Nothing in the posed folder is ever replaced. Never raises."""
    import json
    from datetime import datetime
    posed = getattr(Paths, "DLC_STAGING", None)
    root = getattr(Paths, "REVIEW_ROOT", None)
    if not posed or not root or db is None:
        return False
    posed = Path(posed)
    try:
        files = [f for f in bundle.iterdir() if f.is_file()]
        inputs = [f for f in files
                  if f.name == f"{stem}.mp4" or f.name.startswith(f"{stem}DLC")]
        poses = [f for f in inputs if f.suffix == ".h5"]
        if not any(f.suffix == ".mp4" for f in inputs) or not poses:
            logger.warning("Release %s: no mp4 or no pose file in the bundle; left "
                           "for a person", stem)
            _return_failed(stem)
            return False
        # The posed folder usually still holds this video's files: intake
        # copies a video in and leaves its posed files there until the video
        # is archived, and a held video is not archived. A byte-identical copy
        # there is the one live copy -- the bundle's own is a duplicate, set
        # aside with the outputs and removed once the release has succeeded.
        # A copy that differs in any way is for a person. WHY: without this,
        # 67 of the first 74 releases were refused on 2026-10-22.
        twins, clash = [], []
        for f in inputs:
            there = posed / f.name
            if there.exists():
                if (there.stat().st_size == f.stat().st_size
                        and _file_sha256(there) == _file_sha256(f)):
                    twins.append(f)
                else:
                    clash.append(f.name)
        if clash:
            logger.warning("Release %s: %s in the posed folder differs from the "
                           "bundle's copy; left in deep review", stem, ", ".join(clash))
            _return_failed(stem)
            return False
        if not _ensure_registered(db, stem, posed / f"{stem}.mp4"):
            return False
        from mousereach.pipeline.manifest import select_pose_file
        pose = select_pose_file(poses, expected_scorer=_declared_scorer() or None)

        dest = (Path(root) / "_Problematic"
                / f"_released_qc_holds_{datetime.now():%Y-%m-%d}" / stem)
        if dest.exists():       # released once already today: keep that copy too
            dest = dest.with_name(f"{stem}_{datetime.now():%H%M%S}")
        dest.mkdir(parents=True, exist_ok=True)
        plan = ([(f, dest / f.name) for f in files if f not in inputs or f in twins]
                + [(f, posed / f.name) for f in inputs if f not in twins])
        done = []
        try:
            for src, dst in plan:
                _safe_move(src, dst)
                done.append((src, dst))
            db.force_state(stem, "dlc_complete",
                           reason="quality hold lifted by the current check; "
                                  "returned to the posed folder",
                           current_path=str(posed / f"{stem}.mp4"),
                           dlc_output_path=str(posed / pose.name))
        except Exception as e:
            # Undo every move: the bundle stays exactly as it was, in the queue.
            for src, dst in reversed(done):
                try:
                    _safe_move(dst, src)
                except Exception:
                    pass
            try:
                dest.rmdir()            # only if the undo left it empty
            except OSError:
                pass
            logger.error("Release %s: could not complete (%s); bundle kept", stem, e)
            _return_failed(stem)
            return False
        for f in twins:                 # verified identical to the posed copy above
            try:
                (dest / f.name).unlink()
            except OSError:
                pass
        try:
            (dest / f"{stem}_release_note.json").write_text(json.dumps({
                "released_at": datetime.now().isoformat(),
                "why": "held in deep review by the quality check alone, with no "
                       "file a person wrote; the current check no longer holds "
                       "it, so the video was sent back to the posed folder for a "
                       "fresh run and these machine outputs were set aside here",
                "returned_to": str(posed),
            }, indent=2), encoding="utf-8")
        except OSError:
            pass
        try:
            bundle.rmdir()
        except OSError:
            pass
        _return_retry_after.pop(stem, None)
        logger.info("Released %s: quality hold lifted by the current check; back "
                    "in the posed folder for a fresh run", stem)
        return True
    except Exception as e:
        logger.warning("Release %s: %s", stem, e)
        _return_failed(stem)
        return False


# How many bundles one scan may re-inject before yielding back to the work loop.
# Unbounded, this starves everything else: each return moves 6 files across the
# NAS (~10 s), so a 748-deep queue occupied the loop for ~2 hours while
# 'processing' climbed and not one pipeline ran. Returning is not more important
# than running -- interleave them.
MAX_RETURNS_PER_SCAN = 10

# An EMPTY queue folder is only treated as residue once it is this old. WHY: a
# router may have just created it and still be copying files in (cross-volume
# copies take seconds) -- the return scan retired exactly such a folder
# mid-route on 2026-09-14 and every remaining move failed. Age is judged
# conservatively because the share's clock can run ahead of this machine's by
# many minutes: an mtime in the future counts as young, and so does a folder
# whose age cannot be read. Real residue just waits for a later scan.
EMPTY_DIR_GRACE_SECONDS = 30 * 60

# <queue>/.incoming/<stem> folders already named in a warning, so a stuck one is
# reported once per process instead of on every scan.
_WARNED_INCOMING: set = set()


def _younger_than_grace(path: Path) -> bool:
    try:
        age = time.time() - Path(path).stat().st_mtime
    except OSError:
        return True
    return age < EMPTY_DIR_GRACE_SECONDS


def _warn_stale_incoming() -> None:
    """Name any bundle still being assembled in a queue's staging folder long
    after a route should have finished. Never moves or deletes it: the files
    are a video's only copy (it has already left Processing), and whether the
    route is still running or died half-way is for a person to judge."""
    for queue_root in (getattr(Paths, "TRIAGE_REVIEW", None),
                       getattr(Paths, "DEEP_REVIEW", None)):
        if not queue_root:
            continue
        try:
            building = [d for d in (Path(queue_root) / INCOMING_DIR_NAME).iterdir()
                        if d.is_dir()]
        except OSError:
            continue
        for d in building:
            if str(d) in _WARNED_INCOMING or _younger_than_grace(d):
                continue
            _WARNED_INCOMING.add(str(d))
            logger.warning(
                "Return scan: %s is older than %d minutes -- being routed or an "
                "interrupted route; a person should check",
                d, EMPTY_DIR_GRACE_SECONDS // 60)


def scan_review_queues(db, processing_dir: Path,
                       limit: int = MAX_RETURNS_PER_SCAN) -> Dict[str, int]:
    """Re-inject human-cleared held bundles, at most ``limit`` per scan.

    The remainder are picked up on subsequent cycles, so a deep queue drains
    steadily instead of blocking the pipeline until it is empty.
    """
    summary = {"triage_returned": 0, "deep_returned": 0, "deferred": 0,
               "diverted_to_deep": 0}

    def _budget_left():
        return (summary["triage_returned"] + summary["deep_returned"]
                + summary["diverted_to_deep"]) < limit

    _warn_stale_incoming()

    # TRIAGE: every triaged element resolved (and segmentation sound).
    seen_bundles = set()
    for bundle in _bundles(Paths.TRIAGE_REVIEW):
        if not _budget_left():
            summary["deferred"] += 1
            continue
        stem = bundle.name
        seen_bundles.add(str(bundle))
        try:
            signature = _bundle_signature(bundle)
            if not signature:
                # An EMPTY dir is not a bundle -- it is residue of an earlier
                # divert/return that moved the files and left the folder.
                # Sep-04 left several; one hid behind the both-queues skip all
                # weekend, then diverted "(0 files)" and re-flipped a parked
                # video the moment its twin side cleared (2026-09-08).
                # Routing zero files while flipping db state is pure phantom
                # action; retire the folder instead, whatever the video's
                # state -- but only once it is past EMPTY_DIR_GRACE_SECONDS:
                # a young empty folder may be a route still filling it.
                if _younger_than_grace(bundle):
                    logger.debug("Return scan: %s is an empty dir younger than "
                                 "%d s; left alone (may be mid-route)",
                                 stem, EMPTY_DIR_GRACE_SECONDS)
                    continue
                if _retire_stale_bundle(
                        bundle, stem, "(empty dir)",
                        why="empty directory left behind by an earlier "
                            "divert/return; not a bundle"):
                    summary["empty_retired"] = summary.get("empty_retired", 0) + 1
                    logger.info("Return scan: %s is an EMPTY dir, not a "
                                "bundle; retired to _Problematic", stem)
                continue
        except OSError:
            continue
        try:
            st = _cached_triage_status(bundle, stem, signature)
        except Exception:
            continue
        # A human's deep-review clearance outranks the machine's seg self-check
        # -- the SAME honor rule the gate applies (review_gate seg_failed
        # branch). Without it here, a cleared video with unresolved triage
        # segments ping-pongs forever: gate honors the clear and routes to
        # triage, this scan sees raw seg_failed and diverts to deep review,
        # the clear marker instantly releases it, the pipeline re-runs, repeat
        # (20250708_CNT0210_P4 made 43 laps on 2026-09-09).
        cleared = _deep_review_cleared(bundle, stem)
        if st.seg_failed or st.seg_pending_reseg:
            _dr = getattr(Paths, "DEEP_REVIEW", None)
            if _dr and (Path(_dr) / stem).exists():
                # A same-stem bundle ALREADY sits in deep review (historical
                # escalations split bundles across queues: jsons moved, mp4
                # left behind). Diverting would merge blindly into the
                # reviewer's copy, and retrying the impossible move every
                # scan consumed the whole return budget -- the queue
                # live-locked on 2026-09-01 with human-cleared releases
                # starved behind it. A triage copy that is only a byte-identical
                # leftover of the deep-review bundle is archived (it holds
                # nothing the reviewer's copy lacks); any other duplicate is
                # left for a person -- one info line, no budget spent, no
                # route attempted.
                try:
                    leftover = _is_leftover_copy(bundle, Path(_dr) / stem,
                                                 signature)
                except OSError:
                    leftover = False
                if leftover and _retire_stale_bundle(
                        bundle, stem, "(copy in deep review)",
                        why="leftover of a split escalation: every file is a "
                            "byte-identical copy of one in the deep review "
                            "bundle, which is the copy being reviewed"):
                    summary["leftover_retired"] = summary.get("leftover_retired", 0) + 1
                    logger.info("Return scan: %s triage copy is a byte-identical "
                                "leftover of its deep review bundle; retired to "
                                "_Problematic", stem)
                    continue
                logger.info("Return scan: %s exists in BOTH queues; divert "
                            "skipped (duplicate needs deliberate cleanup)", stem)
                continue
        if st.seg_failed:
            # STALE-BUNDLE GUARD: if the video's CURRENT state says the
            # pipeline has re-handled it since this bundle was staged
            # (re-segmented clean, review applied, finished or filed), the
            # bundle is a leftover from the earlier failed era -- diverting it
            # would flip a healthy video into deep_review and block its
            # archive. That happened twice on 2026-09-04: videos re-completed
            # CLEAN seconds earlier were dragged back by their own stale
            # triage husks. Retire the husk to the _Problematic archive
            # (never delete) and leave the video alone.
            try:
                row = db.get_video(stem) if db else None
                cur_state = (row or {}).get('state')
            except Exception:
                cur_state = None
            # 'unresolvable' guards too: a person parked that video with a
            # reason, and a husk must never un-park it (one did, 2026-09-08).
            if cur_state in ('processing', 'processed', 'archiving',
                             'archived', 'unresolvable'):
                if _retire_stale_bundle(bundle, stem, cur_state):
                    summary["stale_retired"] = summary.get("stale_retired", 0) + 1
                    logger.info(
                        "Return scan: %s triage bundle is STALE (video is now "
                        "'%s'); retired to _Problematic instead of diverting",
                        stem, cur_state)
                continue
            # A human already cleared the failed segmentation: the deep-review
            # question is ANSWERED, so diverting would only lap the cycle.
            # Fall through instead -- the bundle is an ordinary triage bundle
            # now, waiting for its triaged segments to be answered.
            if not cleared:
                # A seg-failed bundle can NEVER satisfy the triage release
                # condition below, and nothing else moves it -- it sat in the
                # triage queue with no route out (40 bundles at the time this
                # was added). A failed segmentation makes the whole video
                # untrustworthy, which is deep review's definition, so send it
                # there. The deep-review release path honors the human clear
                # marker, so a cleared bundle would come straight back --
                # which is why cleared bundles never take this divert.
                if not _claim_return(stem):
                    continue
                try:
                    from .review_gate import route_to_queue
                    route_to_queue(
                        stem, bundle, Paths.DEEP_REVIEW,
                        reason="segmentation failed -- triage cannot release "
                               "this bundle; needs deep review",
                        db=db, db_state="deep_review",
                    )
                    summary["diverted_to_deep"] += 1
                    logger.info(
                        "Return scan: %s diverted triage -> deep review "
                        "(segmentation failed; no triage route out)", stem)
                except Exception as e:
                    logger.error("Return scan: could not divert %s to deep "
                                 "review: %s", stem, e)
                finally:
                    _release_return_claim(stem)
                continue
        if not (st.has_triage and st.fully_resolved
                and (not st.seg_failed or cleared)):
            continue
        if st.seg_pending_reseg:
            # The reviewer answered everything BUT also declared at least one
            # segment mislabeled (true_segment_num set). The boundaries cannot
            # be trusted, so re-injecting would re-run kinematics over the wrong
            # frame windows and archive the video as finished with the
            # segmentation error intact. Divert to DEEP_REVIEW for manual
            # re-segmentation instead; the review file travels with the bundle
            # and re-attaches by frame span after the boundaries are fixed.
            if not _claim_return(stem):
                continue
            try:
                from .review_gate import route_to_queue
                route_to_queue(
                    stem, bundle, Paths.DEEP_REVIEW,
                    reason="reviewer declared segment mislabel "
                           f"(true_segment_num on segments {sorted(st.seg_corrected)}) "
                           "-- needs re-segmentation",
                    db=db, db_state="deep_review",
                )
                summary["diverted_to_deep"] += 1
                logger.info(
                    "Return scan: %s diverted triage -> deep review; reviewer "
                    "corrected segment number(s) %s", stem, sorted(st.seg_corrected))
            except Exception as e:
                logger.error("Return scan: could not divert %s to deep review: %s",
                             stem, e)
            finally:
                _release_return_claim(stem)
            continue
        if _return_backing_off(stem):
            continue
        if _return_to_processing(bundle, stem, processing_dir, db, "triage_cleared"):
            summary["triage_returned"] += 1
    for gone in [k for k in _status_cache if k not in seen_bundles]:
        _status_cache.pop(gone, None)

    # DEEP_REVIEW: an explicit clear marker or in-bundle GT.
    # Own budget, NOT shared with the triage loop's: on 2026-09-01 the triage
    # loop's diverts consumed the whole shared budget every scan and 46
    # human-released deep bundles were starved indefinitely. Releases of
    # finished human work must never queue behind machine-initiated moves.
    recheck_deadline = time.monotonic() + QC_RECHECK_BUDGET_S
    summary.setdefault("qc_released", 0)
    for bundle in _bundles(Paths.DEEP_REVIEW):
        if summary["deep_returned"] >= limit:
            summary["deferred"] += 1
            continue
        stem = bundle.name
        # No empty-dir retire here: an empty folder has no clear marker, so it
        # is never acted on, and EMPTY_DIR_GRACE_SECONDS has nothing to guard.
        if _deep_review_cleared(bundle, stem):
            if not _return_backing_off(stem) and _return_to_processing(
                    bundle, stem, processing_dir, db, "deep_review_cleared"):
                summary["deep_returned"] += 1
            continue
        # A quality hold the current check would not make (block comment at
        # QC_RECHECK_BUDGET_S). Its own budget, after human-cleared returns
        # in the same pass, and bounded in time so re-checks never hold up
        # the work loop.
        if (summary["qc_released"] >= limit or _return_backing_off(stem)
                or time.monotonic() > recheck_deadline):
            continue
        try:
            signature = _bundle_signature(bundle)
        except OSError:
            continue
        if _person_touched(signature):
            continue
        if _qc_hold_lifted(bundle, stem, signature) and _release_to_posed(bundle, stem, db):
            summary["qc_released"] += 1

    if (summary["triage_returned"] or summary["deep_returned"]
            or summary["diverted_to_deep"] or summary["qc_released"]):
        logger.info(
            f"Review-return scan: {summary['triage_returned']} triage + "
            f"{summary['deep_returned']} deep-review videos re-injected, "
            f"{summary['diverted_to_deep']} diverted to deep review, "
            f"{summary['qc_released']} lifted quality holds sent back to the posed folder"
            + (f" ({summary['deferred']} left for later cycles)"
               if summary["deferred"] else "")
        )
    return summary
