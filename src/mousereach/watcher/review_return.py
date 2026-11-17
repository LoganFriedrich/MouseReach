"""Return path: re-inject a human-cleared held video into the pipeline.

The forward gate (review_gate) MOVES un-committable videos into the TRIAGE or
DEEP_REVIEW queue and holds them out of kinematics + the central database. This module
is the other half: it scans those queues for bundles a human has cleared and
moves them back into the local Processing dir, set to ``processing`` state, so
the pipeline re-runs them (Priority-2 'pipeline' work). On the re-run the review
file travels with the bundle, so the gate re-checks and -- now that every
triaged element is resolved -- lets the video through to kinematics.

Where the re-run starts: just after the step that sent the video to the queue,
with every later step run -- the quality triage, the hand-score comparison and
the gate included (watcher/resume_point.py has the rule and its reasons). The
return writes that point to the row (``resume_stage``); the pipeline reuses the
earlier steps' outputs only when they are current.

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
                 by the deep tools when a reviewer clears the flag), and, for
                 a Pillar tray, cuts a person set or confirmed in the
                 segmentation fixer since the video arrived
                 (review/resegmented.py). A ground-truth file in the bundle is
                 NOT a clear: nothing writes its boundaries into the segments
                 file, so releasing on it re-ran the video on the very cuts a
                 person had replaced. Re-injection resumes just after
                 segmentation: the person's cuts are kept and everything after
                 them is remade.

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import filecmp
import functools
import os
import logging
import socket
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import Paths, is_outcome_free_tray
from . import claim_marker, node_status
from ..review.triage_status import triage_status
from .review_routing import LEFT_NOTE, _safe_move, same_path as _same_path

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
    pipeline re-runs the video with no pose at all -- which is how hundreds of
    videos ended up in the deep-review queue as "segmentation_failed".

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
    # to return in a loop, every few minutes. Only a
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
    # again two minutes later -- thousands of times -- while
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
    # Dot-directories are never bundles (a stray dot-folder in both queues was
    # iterated as a phantom video every scan, and -- having no
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
# review files to decide whether it was cleared: seconds of every scan, every two
# minutes, for a queue that mostly does not change between scans.
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
    BOTH queues" line for weeks (hundreds of times for one video) with
    nobody ever doing the "deliberate cleanup" it asked for.
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
    the function returned True. Hundreds of bundles went that way in one run
    before it was caught."""
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
    """True if a reviewer has cleared this deep-review bundle: its clear
    marker exists. The SAME test the review gate applies
    (review_gate._deep_review_cleared), so the scan and the gate cannot
    disagree about whether a person cleared a video.

    WHY the marker only: a ground-truth file in the bundle used to count as a
    clear by its mere presence -- even a progress save with nothing in it
    determined, and without anyone pressing Clear. The return then released
    the video, the segmenter cut it again exactly as before (nothing copies
    ground-truth boundaries into the segments file), and the gate, which
    reads only the marker, could send it straight back."""
    from .review_gate import _deep_review_cleared as _gate_cleared
    return _gate_cleared(stem, Path(bundle))


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
        # One-step replacement (temporary file, then one rename), keeping the
        # source's modified time as copy2 did -- the newer-than test above
        # relies on it. WHY not copy2 straight onto dest: that empties the
        # durable copy first, so a failure part-way left the copy of record
        # truncated, and a truncated review is read as "no answers".
        from mousereach.pipeline.fsutil import copy_file_atomic
        copy_file_atomic(src, dest)
        logger.info(f"Return {stem}: human review copied to the durable store")
    except Exception as e:
        logger.error(
            "Return %s: could NOT make a durable copy of the human review (%s). "
            "After this return its only copy is on this node's local disk.",
            stem, e)


# --- bundle claims: one PROCESS at a time, across machines --------------
# With several watchers scanning the SAME shared queues, two nodes that spot
# the same cleared bundle in one pass would both start moving its files and
# tear the bundle in half across machines -- the split-bundle failure class,
# cross-node edition. Claimed only at MUTATION time (returns, diverts,
# retires, clean-ups, a question written into a held bundle), never for mere
# scanning, with the same exclusive-create marker the posed-video intake uses
# (watcher/claim_marker.py): exactly one node's create succeeds. A claim is
# broken only when its holder has stopped reporting (claim_marker.GONE_AFTER_S,
# by the file server's clock), never for its age.
#
# The claim belongs to ONE PROCESS (claim_marker.this_process), and within it
# to one thread. WHY: on the machine where review happens, the watcher's
# return scan, a review window and mousereach-route-to-queue are separate
# processes of the same host. A claim that belonged to the host let any of
# them act while another was half-way through moving the same bundle (a flag
# written into an outcomes file the scan was copying out, then lost). A
# claim whose process has ended is taken over (crash leftovers). The same
# thread may take it again while holding it (a divert started by the scan
# routes the bundle, and the route takes the claim too): it is counted, and
# the marker goes only when the outermost holder releases it.
#
# The rule this replaced was a check, write, sleep and read-back (not
# atomic), went ahead unclaimed when the claim folder could not be made, and
# broke another node's marker after two hours by age alone.

_claims_lock = threading.Lock()
# stem -> [holding thread id, depth]: this process's claims.
_claims_held: Dict[str, list] = {}


def _return_claims_dir():
    root = getattr(Paths, "REVIEW_ROOT", None)
    return (Path(root) / ".return_claims") if root else None


def _take_marker(d: Path, stem: str) -> bool:
    host, me = socket.gethostname(), claim_marker.this_process()
    if claim_marker.take(d, stem, host, suffix=".claim", process=me):
        return True
    # Held by another node. If that node has stopped running, its claim is a
    # crashed move's leftover: break it and try once more.
    if stem in claim_marker.sweep_gone(d, host, node_status.status_dir(Paths.NAS_ROOT),
                                       suffix=".claim"):
        return claim_marker.take(d, stem, host, suffix=".claim", process=me)
    return False


def _claim_return(stem: str) -> bool:
    """Take the claim on ``stem``'s bundle for this thread (see the block
    comment above). True when held afterwards; every True must be matched by
    one _release_return_claim. False while another machine, another process
    or another thread of this process holds it."""
    me = threading.get_ident()
    with _claims_lock:
        held = _claims_held.get(stem)
        if held is not None:
            if held[0] != me:
                return False
            held[1] += 1
            return True
        d = _return_claims_dir()
        if d is not None and not _take_marker(d, stem):
            return False
        _claims_held[stem] = [me, 1]
        return True


def _claim_return_nested(stem: str):
    """``(claimed, release)`` for callers that may run inside a holder of the
    claim (a route the return scan started). The claim is counted per thread,
    so ``release`` is simply ``claimed``: releasing an inner claim never drops
    the outer one."""
    ok = _claim_return(stem)
    return ok, ok


def _release_return_claim(stem: str) -> None:
    """Give back one _claim_return of this thread; the marker goes with the
    outermost one. A claim this thread does not hold is never touched."""
    me = threading.get_ident()
    with _claims_lock:
        held = _claims_held.get(stem)
        if held is None or held[0] != me:
            return
        held[1] -= 1
        if held[1] > 0:
            return
        _claims_held.pop(stem, None)
        d = _return_claims_dir()
        if d is None:
            return
        claim_marker.release(d, stem, socket.gethostname(), suffix=".claim",
                             process=claim_marker.this_process())


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


def _set_processing_state(db, stem: str, h5_dest: Optional[Path],
                          resume: Optional[str] = None) -> None:
    """Set ``stem`` to 'processing' after a return; raise when that cannot be
    done (the caller then puts the files back).

    NOTE FOR ANYONE REASONING ABOUT THE 'processing' POOL: this is a SECOND
    ENTRY POINT into 'processing', beside intake. Any guard or invariant of the
    form "videos in processing got there through intake, so a check at intake
    is sufficient" is wrong because of this path -- a human-cleared review
    re-enters here, bypassing intake entirely. (An orchestrator work-selection
    guard was designed on that assumption and had to be redone; hence this
    sign.)

    A cleared review routinely returns a video whose row has no legal path to
    'processing' (typically 'archived': the bundle sat in a review queue while
    the row was closed out). That is exactly what force_state exists for -- go
    straight there instead of letting update_state log an ERROR first and then
    forcing anyway. The WARNING force_state writes is the honest audit line."""
    kw = {'dlc_output_path': str(h5_dest)} if h5_dest is not None else {}
    # Where the run resumes: just after the step that sent the video here
    # (resume_point.resume_stage_for; the caller reads it before the moves,
    # since the bundle's manifest is queue metadata and is dropped by them).
    kw['resume_stage'] = resume
    # A re-run scope left on the row (a filed video's kinematics-only re-run
    # that the gate sent to a queue) is cleared: a person has just answered
    # questions about this video, so its next run must go through every check
    # (a resumed run always does), and the pipeline must not read a kept
    # 'kinematics' scope back from the row (orchestrator.kept_rerun_start_stage)
    # in place of the resume point.
    kw['reprocess_scope'] = None
    reason = "review cleared; returned to Processing"
    state = None
    try:
        row = db.get_video(stem)
        state = row['state'] if row else None
    except Exception:
        state = None
    from .db import VIDEO_TRANSITIONS
    try:
        if state is not None and "processing" not in VIDEO_TRANSITIONS.get(state, []):
            db.force_state(stem, "processing", reason=reason, **kw)
        else:
            db.update_state(stem, "processing", **kw)
    except Exception:
        # The row exists (registered before the moves), so a legal-transition
        # failure is what lands here; force_state is the documented escape
        # hatch. If it fails too, this raises and the files go back.
        db.force_state(stem, "processing", reason=reason, **kw)


def _undo_return_moves(stem: str, done, current) -> None:
    """Put back every file a failed return moved (newest first). ``current``
    is the (source, destination, destination existed before) of the move that
    raised: when its source is still in the bundle, a copy the move left at
    the destination is deleted (a move across drives copies, then deletes the
    source -- a held source leaves a full copy behind) -- but only when the
    move made it: the destination was not there before, or it is byte for
    byte the bundle's file. WHY: a same-named file already in Processing,
    which the move's overwrite could not delete in time, is not this move's
    copy and may be the only one of its kind. A file that will not go back is
    logged as an ERROR naming both places: the bundle is then split and a
    person must finish it."""
    if current is not None:
        src, dst, existed = current
        try:
            if (Path(src).is_file() and Path(dst).is_file() and not _same_path(src, dst)
                    and (not existed or filecmp.cmp(str(src), str(dst), shallow=False))):
                Path(dst).unlink()
        except OSError as err:
            logger.error("Return %s: could not delete the partial copy %s of %s: %s",
                         stem, dst, src, err)
    for src, dst in reversed(done):
        try:
            _safe_move(dst, src)
        except Exception as err:
            logger.error("Return %s: could not move %s back to %s while undoing a "
                         "failed return, so this bundle is now split -- move it back "
                         "by hand: %s", stem, dst, src, err)


def _files_in_the_way(stem: str, plan) -> List[str]:
    """The planned destinations a return may not replace: a different file
    of the same name already in the processing folder that is a person's
    file (review_routing.is_person_file -- a review, ground truth, a clear
    marker, cuts set by hand ...) or an input (the video, a pose file). An
    identical file is replaced (nothing is lost), and so is a machine output
    of an earlier run (the bundle's own file is the current copy of it: the
    video's live copy is the bundle). Never raises; a file that cannot be
    compared counts as in the way."""
    from .review_routing import is_input_file, is_person_file
    out = []
    for src, dst in plan:
        dst = Path(dst)
        try:
            if not dst.is_file() or _same_path(src, dst):
                continue
            if filecmp.cmp(str(src), str(dst), shallow=False):
                continue
            if is_person_file(dst, stem) or is_input_file(stem, dst.name):
                out.append(dst.name)
        except OSError:
            out.append(dst.name)
    return out


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

    # A crystallized (publication-locked) video is never moved back to
    # Processing: its filed results must stay as published. This used to be
    # forced through (force_state below), lock and all.
    from .db import crystallized_refusal
    locked = crystallized_refusal(db, stem, "returned to Processing")
    if locked:
        logger.error(f"Return {stem}: {locked} The bundle stays in the queue. "
                     f"(Tried again in {RETURN_RETRY_S // 60} min.)")
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

    # Delete crash leftovers BEFORE moving anything. A one-step write's
    # temporary file (".<name>.<random>.tmp", see pipeline/fsutil.py) left by a
    # crash between writing it and the rename is not one of the video's files:
    # carried into Processing it would sit there for good (nothing sweeps
    # dot-names there). The bundle's real files are complete, and a review's
    # content is in the durable copy, written before the bundle copy.
    # WHY before the moves, and refuse on failure: a leftover that cannot be
    # deleted is held open by something -- most likely a review window writing
    # into this bundle at this moment. Finding that out half-way through the
    # moves left the video set to 'processing' with a bundle folder still in
    # the queue (bundle.rmdir() fails on the leftover) and nothing said. Now
    # nothing is moved, the refusal is logged, and the return is tried again.
    from mousereach.pipeline.fsutil import is_leftover_temp, remove_temp_file
    try:
        leftovers = [f for f in bundle.iterdir()
                     if f.is_file() and is_leftover_temp(f.name)]
    except OSError:
        leftovers = []
    stuck_temps = []
    for f in leftovers:
        if remove_temp_file(f):
            logger.warning(f"Return {stem}: deleted a leftover temporary "
                           f"file {f.name} instead of moving it")
        else:
            stuck_temps.append(f.name)
    if stuck_temps:
        logger.error(
            f"Return {stem}: could not delete the temporary file(s) "
            f"{', '.join(stuck_temps)} in {bundle} -- something has them open "
            f"(a review window saving into this bundle?). Nothing was moved; "
            f"the bundle stays in the queue. (Tried again in "
            f"{RETURN_RETRY_S // 60} min.)")
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
    #
    # ALL OR NOTHING. Every move is planned first, each one done is recorded,
    # and the state is set inside the same try: if any move -- or the state
    # write -- fails, every file already moved goes back into the bundle and
    # the return is retried after RETURN_RETRY_S. The queue records (manifest,
    # routing) are deleted only after the state is set. WHY: the moves used to
    # stop at the first failure (a file held open by a review window, an
    # indexer, a backup pass) with earlier files -- the video, its pose, the
    # reviewer's answers, possibly the clear marker -- already in Processing
    # and the queue records already deleted. The half left in the queue then
    # read as unanswered and no review tool listed it; only a person could
    # put it back together. Inputs move first and the clear marker LAST, so
    # even an undo that cannot finish never leaves a clear in the queue
    # beside a bundle missing its files.
    h5_dest: Optional[Path] = pose_src
    resume = _resume_stage(bundle, stem, db, reason)
    metadata: List[Path] = []
    plan: List[tuple] = []
    for f in list(bundle.iterdir()):
        if not f.is_file():
            continue
        if _is_queue_metadata(f.name, stem):
            metadata.append(f)
            continue
        if is_leftover_temp(f.name):
            # One that appeared after the check above (a save that started in
            # between): never carried along. Deleted if it can be; if not, it
            # keeps the bundle folder in the queue below, which is logged.
            if not remove_temp_file(f):
                logger.warning(f"Return {stem}: a temporary file {f.name} "
                               f"appeared during the return and could not be "
                               f"deleted; it is left in {bundle}")
            continue
        plan.append((f, processing_dir / f.name))

    def _move_order(item):
        name = item[0].name
        if name == f"{stem}_deep_review_cleared.json":
            return 2
        if name == f"{stem}.mp4" or name.startswith(f"{stem}DLC"):
            return 0
        return 1
    plan.sort(key=_move_order)

    if not plan:
        return False

    # A different same-named file already in the processing folder that is
    # a person's, or an input, is never replaced: the whole return waits and
    # the folder is listed for a person. WHY: the move replaced it, and an
    # undo could restore only what this return had brought -- a person's
    # same-named file (hand-set cuts, a review) was gone for good. Setting it
    # aside in a hidden folder hid it from everyone instead.
    blocking = _files_in_the_way(stem, plan)
    if blocking:
        from .review_routing import note_for_a_person
        says = (f"{stem} is ready to leave its review queue ({bundle}), but the "
                f"processing folder {processing_dir} already holds a different "
                f"{', '.join(blocking)} -- a person's file or the video's video or "
                f"pose -- which a return never replaces. Nothing was moved; the "
                f"bundle stays in the queue. Look at the file(s) in "
                f"{processing_dir}: delete them when they are not needed (or move "
                f"them somewhere you keep such files). The return is tried again "
                f"within {RETURN_RETRY_S // 60} minutes and this line then goes "
                f"away by itself.")
        logger.error(f"Return {stem}: {says}")
        note_for_a_person("return_blocked", stem, bundle, says)
        _return_failed(stem)
        return False

    done: List[tuple] = []
    current = None
    try:
        for src, dst in plan:
            current = (src, dst, Path(dst).exists())
            _safe_move(src, dst)
            done.append((src, dst))
            current = None
            if Path(src) == Path(pose_src):
                h5_dest = dst
        _set_processing_state(db, stem, h5_dest, resume)
    except Exception as e:
        _undo_return_moves(stem, done, current)
        logger.error(
            f"Return {stem}: could not complete ({e}); every file moved so far "
            f"was put back and the bundle stays in the queue. (Tried again in "
            f"{RETURN_RETRY_S // 60} min.)")
        _return_failed(stem)
        return False
    moved = len(done)
    from .review_routing import drop_for_a_person
    drop_for_a_person(folder=bundle, kind="return_blocked")

    # The queue records go last, once the video is back in Processing and its
    # state says so (see ALL OR NOTHING above).
    for f in metadata:
        try:
            f.unlink()
        except OSError:
            pass

    # Remove the now-empty bundle dir so it leaves the queue.
    try:
        bundle.rmdir()
    except OSError as e:
        # Said, not swallowed: an emptied-but-present bundle folder stays in
        # the queue listing although the video is now 'processing'.
        logger.warning(f"Return {stem}: the video is back in Processing but "
                       f"its bundle folder {bundle} could not be removed "
                       f"({e}); it is left in the queue")

    _return_retry_after.pop(stem, None)
    logger.info(f"Returned {stem} to Processing ({moved} files) reason={reason}")
    return True


def _resume_stage(bundle: Path, stem: str, db, reason: str) -> str:
    """Where the returned video's run resumes (resume_point.resume_stage_for):
    from the step recorded when it was routed -- the bundle's manifest first
    (it travels with the bundle), then the row -- and the queue it is
    returning from. Never raises."""
    from .resume_point import resume_stage_for
    queue = "deep_review" if reason == "deep_review_cleared" else "triage"
    step = None
    try:
        import json
        doc = json.loads((Path(bundle) / f"{stem}_manifest.json").read_text(encoding="utf-8"))
        step = (doc.get("provenance") or {}).get("held_at_step")
    except Exception:
        step = None
    if not step:
        try:
            step = (db.get_video(stem) or {}).get("held_at_step")
        except Exception:
            step = None
    resume = resume_stage_for(step, queue)
    logger.info(f"Return {stem}: held at step '{step or 'not recorded'}' "
                f"({queue}); its run resumes at '{resume}' with every later step")
    return resume


def _live_copy_of(stem: str, processing_dir=None) -> Dict[str, Path]:
    """The files of ``stem``'s current copy outside the review queues: this
    machine's processing folder first, then its filed results in Analyzed.
    Empty when neither holds a file of it (or neither can be read). Used as
    the "current copy elsewhere" a leftover's files are checked against
    (review_routing.clear_leftover). Never raises."""
    from .review_routing import _bundle_files
    out: Dict[str, Path] = {}
    folders = []
    if processing_dir is not None:
        folders.append(Path(processing_dir))
    try:
        from mousereach.archive.core import get_archive_destination
        dest = get_archive_destination(stem)
        if dest:
            folders.append(Path(dest))
    except Exception:
        pass
    for d in folders:
        try:
            for f in _bundle_files(d, stem):
                out.setdefault(f.name, f)
        except OSError:
            continue
    return out


@_with_return_claim
def _retire_stale_bundle(bundle: Path, stem: str, video_state: str,
                         why: str = None, live: Optional[Dict[str, Path]] = None) -> bool:
    """Clear a stale queue bundle as a leftover (review_routing.clear_leftover).

    A bundle is STALE when the pipeline has re-handled its video since it was
    staged -- the video's live state says processing/processed/archiving/
    archived while the bundle still carries the old era's flags -- or it is an
    empty folder, or a byte-identical copy of the video's deep-review bundle.
    Leaving it as a bundle would re-divert (and re-flip a healthy video) on
    every scan. ``live`` names the video's current copy (the caller's choice:
    the deep-review bundle, or _live_copy_of); what in the bundle has a
    current copy there is deleted, and anything else -- what a person wrote, a
    video or pose with no identical copy, every file when no current copy can
    be named -- stays where it is, with a note, listed for a person. A
    bundle with the note is skipped by every later scan.

    WHY not the archive folder it used to go to: that folder is only for
    cases the pipeline cannot process at all, and a stale bundle is a machine
    leftover -- worth nothing once a current copy exists, and a person's work
    in it is never moved where nobody looks.

    Returns True when the bundle no longer acts as a bundle (removed, or kept
    with its note); False when it could not be dealt with (tried again next
    scan). Never raises."""
    try:
        from .review_routing import clear_leftover
        clear_leftover(
            bundle, stem, live or {},
            why=(why or "stale bundle") + f" (the video's state is '{video_state}')",
            what_to_do=("the video no longer waits in this queue. Look at what is "
                        "kept here; if the video should be reviewed again, route it "
                        "again (mousereach-route-to-queue)."))
        return True
    except Exception as e:
        logger.warning("Return scan: could not clear the stale bundle %s: %s",
                       stem, e)
        return False


# ---------------------------------------------------------------------------
# Machine-only quality holds the current check would not make
# ---------------------------------------------------------------------------
# A deep-review bundle routed "qc_needs_review" is held by the machine's own
# quality check, not by a person. When that check is corrected, a hold it
# would no longer make is a machine error sitting in a person's queue. WHY:
# the check demanded an outcomes file Easy and Flat trays never get, and
# Easy/Flat bundles were held in deep review for that alone.
#
# So a bundle with that reason and no file a person wrote has the check re-run
# on it. The pose-tracking layer is NOT re-run (it would read every pose file
# over the share): its flags are carried over from the bundle's stored
# verdict, so a hold that layer made stays a hold. The file-based layers
# (structural, cross-step, statistical) are recomputed. If nothing critical
# remains, the video goes back to the posed folder and re-runs in its priority
# turn -- NOT straight into 'processing', because anything there blocks intake
# (max_local_pending), and released low-priority videos would hold up new
# work. The bundle's machine outputs are deleted once the video is back in the
# posed folder: that is the video's current copy, and its fresh run makes them
# again. (They used to be moved to an archive folder that is meant only for
# cases the pipeline cannot process at all.)
QC_RECHECK_BUDGET_S = 30            # seconds of re-checks per scan
_qc_recheck_cache: Dict[str, tuple] = {}
# Cleared videos held because nobody re-segmented them: logged once per watcher
# run, not once per scan (the scan runs every few seconds).
_held_unsegmented_logged: set = set()
# Triage copies of videos in deep review already reported (once per process).
_both_queues_logged: set = set()
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


# bundle path -> ((signature, row updated_at), verdict), so Analyzed is read once
# per bundle and row change, not on every scan.
_superseded_cache: Dict[str, tuple] = {}


def _superseded_by_archive(bundle: Path, stem: str, row: Optional[tuple],
                           signature: tuple) -> Optional[str]:
    """Why this queue bundle is a leftover of a video that has since been
    archived, or None when it is not (or that cannot be told).

    ``row`` is (state, updated_at) from this node's database. Only an
    'archived' video qualifies, and only a bundle holding nothing a person
    wrote (no review, clear marker or ground truth). Then the bundle is a
    leftover when it holds no results at all -- a file a return could not
    take along, typically the .mp4 the review tool still had open -- or when
    every file in it is older than the video's archived results.

    WHY: such a bundle sat in a person's queue and asked for work already
    done: videos a person had cleared were sent back to deep review as a bare
    .mp4 (see orchestrator.queue_folder_holds_this_run), and were cleared a
    second time by hand. Bundles the bench-disagreement
    router puts in a queue are never caught: their results leave Analyzed, and
    their routing record is written after the archive.
    """
    if not row or row[0] != "archived" or _person_touched(signature):
        return None
    key = (signature, row[1])
    hit = _superseded_cache.get(str(bundle))
    if hit is not None and hit[0] == key:
        return hit[1]
    names = {name for name, _, _ in signature}
    why = None
    if f"{stem}_segments.json" not in names:
        why = ("holds no results (a file a return could not take along); the "
               "video has since been archived")
    else:
        try:
            from mousereach.archive.core import get_archive_destination
            archived = (Path(get_archive_destination(stem))
                        / f"{stem}_processing_manifest.json").stat().st_mtime_ns
            if max(m for _, _, m in signature) < archived:
                why = "every file predates the video's archived results"
        except Exception:
            why = None
    _superseded_cache[str(bundle)] = (key, why)
    return why


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
        # A stored tracking hold still stands -- except a ruler-variation
        # hold, which the current check no longer makes (triage.
        # RULER_UNSTABLE_DESC): those videos go back for a run.
        from mousereach.pipeline.triage import RULER_UNSTABLE_DESC
        if any(f.get("category") in ("dlc_coherence", "pose_unusable")
               and f.get("severity") == "critical"
               and not str(f.get("description", "")).startswith(RULER_UNSTABLE_DESC)
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
    block comment above). The mp4 and pose files go to the posed folder; once
    the row says so, what is left in the bundle is a leftover
    (review_routing.clear_leftover): machine outputs, queue records and
    copies of the posed files are deleted, and a person's file (an Easy or
    Flat tray is released even with one, see the scan) stays where it is,
    with the note, listed under 'Folders left for a person' in the Review
    Queues tab. Nothing in the posed folder is ever replaced. Never raises."""
    posed = getattr(Paths, "DLC_STAGING", None)
    root = getattr(Paths, "REVIEW_ROOT", None)
    if not posed or not root or db is None:
        return False
    # A crystallized (publication-locked) video is never moved
    # (db.crystallized_refusal).
    from .db import crystallized_refusal
    locked = crystallized_refusal(db, stem, "released to the posed folder")
    if locked:
        logger.error(f"Release {stem}: {locked} The bundle stays in its queue.")
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
        # is archived, and a held video is not archived. The same file there
        # is the one live copy -- the bundle's own is a duplicate, set aside
        # with the outputs and removed once the release has succeeded. A copy
        # that differs is for a person. WHY: without this, most releases were
        # refused.
        #
        # "The same file": same size and same modified time -- the quick check
        # rsync and robocopy use; every copy in this chain (posed folder ->
        # processing -> review bundle) keeps the modified time, and nearly
        # every real pair matches that way. Same size with a different time
        # is settled by comparing SHA-256. WHY not always hash: reading both
        # copies (a large video over the share) made each release take seconds
        # and a backlog a half-day trickle.
        twins, clash = [], []
        for f in inputs:
            there = posed / f.name
            if there.exists():
                a_st, b_st = there.stat(), f.stat()
                if a_st.st_size != b_st.st_size:
                    clash.append(f.name)
                elif (abs(a_st.st_mtime - b_st.st_mtime) < 2
                        or _file_sha256(there) == _file_sha256(f)):
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

        plan = [(f, posed / f.name) for f in inputs if f not in twins]
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
                except Exception as err:
                    logger.error('%s: could not move %s back to %s while undoing a release, so this bundle is now split: %s', stem, dst, src, err)
                    pass
            logger.error("Release %s: could not complete (%s); bundle kept", stem, e)
            _return_failed(stem)
            return False
        # The video's current copy is now in the posed folder, so what is left
        # in the bundle is a leftover (review_routing.clear_leftover): its
        # machine outputs, queue records and copies of the posed files are
        # deleted; a person's file (an Easy or Flat tray is released even with
        # one, see the scan) stays where it is, listed for a person.
        from .review_routing import _bundle_files, clear_leftover
        clear_leftover(
            bundle, stem, {f.name: f for f in _bundle_files(posed, stem)},
            why="the quality hold was lifted by the current check and the video "
                "was sent back to the posed folder for a fresh run",
            what_to_do="nothing here is used by the fresh run. Look at what is kept.")
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
# NAS (seconds each), so a deep queue occupied the loop for hours while
# 'processing' climbed and not one pipeline ran. Returning is not more important
# than running -- interleave them.
MAX_RETURNS_PER_SCAN = 10

# An EMPTY queue folder is only treated as residue once it is this old. WHY: a
# router may have just created it and still be copying files in (cross-volume
# copies take seconds) -- the return scan retired exactly such a folder
# mid-route and every remaining move failed. Age is judged
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


# ---------------------------------------------------------------------------
# Finishing routes that did not finish
# ---------------------------------------------------------------------------
# Two things a route can leave undone, both finished here:
#   * The review manifest. A route publishes the bundle, records the hold in
#     the database, and only then writes the review manifest -- the file the
#     review tools list a bundle by. If that last write fails, the bundle is in
#     the queue, the row says so, and no review tool offers it. The scan
#     writes such a missing manifest again from the routing record.
#   * Files that could not be moved. A route out of Processing goes on when a
#     file cannot be moved (most often the mp4, still held open) and lists it
#     under failed_files in the routing record. Nothing moved such files in
#     afterwards: the reviewer's bundle lacked them, and Processing kept a file
#     of a video whose row said it was in a queue. The scan now moves each one
#     in from where the record says it is, and lists every bundle still
#     missing a file in <NAS root>/attention/<host>_unfinished_routes.json (the
#     dashboard's "needs your attention" panel) until it is whole.
# Checked at most every ROUTE_REPAIR_INTERVAL_S, not on every scan. Each pass
# lists every bundle once (the triage scan's listing is reused for the triage
# queue), and a bundle's routing record is READ only when its size or modified
# time changed since the last pass (_record_flags). WHY: opening a record on
# the share costs tens of milliseconds, several times a listing, and reading
# every record of a queue of thousands of bundles every ten minutes kept the
# watcher's loop on the share for minutes at a time. Both cases are rare, and
# whether a manifest is missing is read from the listing alone.
ROUTE_REPAIR_INTERVAL_S = 600
_route_repair_due = [0.0]
# bundle path -> ((size, mtime) of its routing record, has failed_files,
# has finished_later)
_record_flags_cache: Dict[str, tuple] = {}


def _record_flags(bundle: Path, stem: str, signature: tuple) -> tuple:
    """``(has failed_files, has finished_later)`` of ``bundle``'s routing
    record, read only when the record's (size, modified time) in
    ``signature`` changed since it was last read. (False, False) when there
    is no record."""
    from .review_routing import read_routing_manifest
    key = next(((size, mtime) for name, size, mtime in signature
                if name == f"{stem}_routing.json"), None)
    if key is None:
        _record_flags_cache.pop(str(bundle), None)
        return False, False
    hit = _record_flags_cache.get(str(bundle))
    if hit is not None and hit[0] == key:
        return hit[1], hit[2]
    record = read_routing_manifest(bundle, stem) or {}
    flags = (bool(record.get("failed_files")), bool(record.get("finished_later")))
    _record_flags_cache[str(bundle)] = (key, flags[0], flags[1])
    return flags


def _repair_missing_manifest(bundle: Path, stem: str,
                             signature: Optional[tuple] = None) -> bool:
    """Write ``{stem}_manifest.json`` into a published bundle that has a
    routing record but no manifest. True when one was written. Never raises.

    Only a bundle that still holds a file of the video besides its routing
    record is repaired -- a return empties a bundle by removing the queue
    records last, and a folder down to its records is on its way out, not a
    bundle missing its manifest. Done under the return claim, so a return on
    another node never has a manifest written back into a folder it is
    emptying. ``signature`` (_bundle_signature) saves listing the bundle
    again. The step that sent the video is taken from the routing record
    (route_to_queue keeps it there), so the return still resumes just after
    it."""
    try:
        if signature is not None:
            names = {name for name, size, _ in signature if size >= 0}
        else:
            names = {p.name for p in bundle.iterdir() if p.is_file()}
    except OSError:
        return False
    if f"{stem}_routing.json" not in names or f"{stem}_manifest.json" in names:
        return False
    if not any(n for n in names if not _is_queue_metadata(n, stem)):
        return False
    if not _claim_return(stem):
        return False
    try:
        from .review_gate import _write_review_manifest
        from .review_routing import read_routing_manifest
        record = read_routing_manifest(bundle, stem) or {}
        _write_review_manifest(bundle, stem, record.get("routed_reason") or "",
                               held_at_step=record.get("held_at_step"))
        logger.warning("Return scan: %s had no review manifest (its route could not "
                       "write it); written again so the review tools list it", stem)
        return True
    except Exception as e:
        logger.error("Return scan: %s has no review manifest and it could not be "
                     "written (%s); the review tools do not list it until it is", stem, e)
        return False
    finally:
        _release_return_claim(stem)


def _iso_to_epoch(text) -> Optional[float]:
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(text)).timestamp()
    except (TypeError, ValueError):
        return None


def _on_shared_storage(path) -> bool:
    """True when ``path`` is on the shared storage every machine sees (under
    the NAS root or the review root); False for a machine's own disk."""
    for root in (getattr(Paths, "NAS_ROOT", None), getattr(Paths, "REVIEW_ROOT", None)):
        if not root:
            continue
        try:
            r = os.path.normcase(os.path.abspath(str(root))).rstrip("\\/")
            q = os.path.normcase(os.path.abspath(str(path)))
            if q == r or q.startswith(r + os.sep):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _held_in_this_queue(bundle: Path, stem: str, db) -> Tuple[bool, str]:
    """``(held, why_not)``: whether the watcher row says the video is held in
    the queue ``bundle`` is in ('triage' for the triage queue, 'deep_review'
    for the deep-review queue). No database: True (nothing to ask). A row
    this machine does not have, or that cannot be read: False."""
    if db is None:
        return True, ""
    from .review_gate import _same_folder
    if _same_folder(bundle.parent, getattr(Paths, "DEEP_REVIEW", None)):
        wanted = "deep_review"
    elif _same_folder(bundle.parent, getattr(Paths, "TRIAGE_REVIEW", None)):
        wanted = "triage"
    else:
        return False, "the bundle is not in a configured review queue"
    try:
        row = db.get_video(stem)
    except Exception as e:
        return False, "the watcher record could not be read (%s)" % e
    if not row:
        return False, "this machine's watcher has no record of the video"
    if row.get("state") != wanted:
        return False, ("the watcher record says the video is '%s', not held in this "
                       "queue" % row.get("state"))
    return True, ""


def _finish_failed_moves(bundle: Path, stem: str, db=None):
    """Finish what the routing record of ``bundle`` lists as failed_files.
    Returns ``(still_missing, moved_count)``: the entries this machine lists
    for a person (empty when the bundle is whole, or what is left belongs to
    another machine) and how many files this call finished. Never raises.

    ONLY WHILE THE VIDEO IS HELD IN THAT QUEUE: with a watcher database
    (``db``), a file at the source is moved in, or deleted as a copy, only
    when the video's row says it is held in the bundle's queue
    (_held_in_this_queue). Otherwise every entry with a file at its source is
    listed for a person and nothing is moved or deleted. WHY: the source is
    the processing folder the video was routed from, and the video may be
    running there again (a row and folder that disagree -- a bundle left in a
    queue while the video went back to Processing). A re-run's video and
    pose keep their modified times when they are copied in or returned, so
    the "newer than the route" rule below cannot tell them from the routed
    files: without this check the scan could move the running video's mp4
    into the old bundle, or delete it as an "identical copy". Without a
    database (the dashboard's Resolve button on a machine with no watcher) a
    person asked for it, and it proceeds.

    For each listed file, by where its copies are:
      * not in the bundle, still at the source: moved in -- unless the source
        file is newer than the route (modified more than a minute after
        ``routed_at``): that is a later run's file, and moving it in would mix
        generations, so it stays listed for a person;
      * in the bundle AND still at the source: the move copied it but could
        not delete the original (a file held open cannot be renamed, so a
        move copies and then deletes). Byte-identical: the copy at the source
        is deleted and the file counts as finished. Different: the bundle's
        copy may be cut off -- listed for a person, nothing deleted;
      * in the bundle only: whole (a person moved it in by hand, as the
        attention panel asks, or an earlier pass could not update the
        record) -- counted as finished, so it leaves the record;
      * nowhere: listed (missing).

    WHICH MACHINE: a source on shared storage is finished by any machine. A
    source on a machine's own disk (its Processing folder) is finished only
    by the machine that made the route (``routed_by`` in the record); another
    machine leaves that entry alone and does not list it -- the same path on
    its own disk is a different file (it may be its own run of the same
    video). An entry of a record written before records named the machine is
    never acted on when its source is on a machine's own disk: if a file is at
    that path here, the entry is listed for a person.

    Done under the return claim. The routing record is rewritten (in one step)
    only when this call finished a file, and keeps its modified time. WHY:
    readers judge a bundle by its files' times (_superseded_by_archive), and
    crossing an entry off the list is not new content."""
    from .review_routing import (INCOMING_DIR_NAME as _INC, _move_leaving_no_copy,
                                 _same_content, read_routing_manifest,
                                 _write_routing_manifest, same_path)
    from mousereach.pipeline.fsutil import retry_transient
    record = read_routing_manifest(bundle, stem) or {}
    failed = list(record.get("failed_files") or [])
    if not failed:
        return [], 0
    here = socket.gethostname()
    routed_by = record.get("routed_by")
    if not _claim_return(stem):
        return [], 0                      # another machine is on it; next pass
    try:
        held, why_not = _held_in_this_queue(bundle, stem, db)
        routed = _iso_to_epoch(record.get("routed_at"))
        keep, listed, finished = [], [], []
        finished_entries = []
        for entry in failed:
            name = str(entry.get("name") or "")
            target = bundle / name
            candidates = [entry.get("source"),
                          str(Path(record.get("source_dir") or "") / name)
                          if record.get("source_dir") else None,
                          str(bundle.parent / _INC / stem / name)]
            src, foreign = None, None
            for c in candidates:
                if not c or same_path(c, target):
                    continue
                if not _on_shared_storage(c) and routed_by != here:
                    # Another machine's disk. A record without the machine:
                    # only a file that is actually here is in question.
                    if routed_by is not None or Path(c).is_file():
                        foreign = foreign or c
                    continue
                if Path(c).is_file():
                    src = Path(c)
                    break
            in_bundle = target.is_file()
            try:
                if src is None and foreign is not None:
                    keep.append(entry)
                    if routed_by is None:
                        listed.append(dict(entry, error=(
                            "routed before routing records named the machine; "
                            "check %s on the machine that routed it%s" % (
                                foreign, " -- the bundle has a copy, so delete that "
                                "one if identical" if in_bundle else ""))))
                    continue                   # the routing machine finishes it
                if src is None:
                    if not in_bundle:
                        keep.append(entry)
                        listed.append(entry)
                    else:
                        # In the bundle only: whole -- moved in by a person
                        # (the attention panel asks for exactly that) or by
                        # a pass whose record update failed. The entry is
                        # dropped from the record like a file moved here, so
                        # the bundle stops counting as unfinished.
                        finished.append(name)
                        finished_entries.append(dict(entry, finished="found in the bundle"))
                    continue
                if not held:
                    err = ("%s; the file at %s was neither moved nor deleted -- a "
                           "person decides" % (why_not, src))
                    keep.append(dict(entry, error=err))
                    listed.append(dict(entry, error=err))
                    continue
                if in_bundle:
                    if _same_content(src, target):
                        retry_transient(src.unlink, what=src.name)
                        finished.append(name)
                        finished_entries.append(dict(
                            entry, finished="identical copy at the source deleted"))
                    else:
                        err = ("a different copy is still at %s; the bundle's copy "
                               "may be incomplete -- a person decides which is "
                               "right" % src)
                        keep.append(dict(entry, error=err))
                        listed.append(dict(entry, error=err))
                    continue
                if routed is not None and src.stat().st_mtime > routed + 60:
                    err = ("the file at the source is newer than the route; not moved "
                           "-- a person decides which copy belongs in the bundle")
                    keep.append(dict(entry, error=err))
                    listed.append(dict(entry, error=err))
                    continue
                _move_leaving_no_copy(src, target, stem)
                finished.append(name)
                finished_entries.append(dict(entry, finished="moved in"))
            except Exception as e:
                keep.append(dict(entry, error=str(e)))
                listed.append(dict(entry, error=str(e)))
        if finished:
            record["failed_files"] = keep
            # The whole entry is kept (its error and where the file was), with
            # how and when it was finished. WHY: the entry is the evidence of
            # what the route could not do; a bare name lost it.
            from datetime import datetime
            stamp = datetime.now().isoformat(timespec="seconds")
            record["finished_later"] = (list(record.get("finished_later") or [])
                                        + [dict(e, finished_at=stamp, finished_on=here)
                                           for e in finished_entries])
            record_path = bundle / f"{stem}_routing.json"
            try:
                before = record_path.stat()
            except OSError:
                before = None
            try:
                _write_routing_manifest(bundle, stem, record)
                if before is not None:
                    # The record keeps its modified time. WHY: readers judge
                    # a bundle by its files' times (_superseded_by_archive:
                    # "every file predates the archived results"), and
                    # crossing a file off this list is not new content.
                    os.utime(record_path, ns=(before.st_atime_ns, before.st_mtime_ns))
            except Exception as e:
                logger.error("Return scan: %s: finished %s but could not update its "
                             "routing record: %s", stem, ", ".join(finished), e)
            logger.warning("Return scan: %s: finished %s -- left behind by a route "
                           "that could not move %s at the time",
                           stem, ", ".join(finished),
                           "it" if len(finished) == 1 else "them")
        return listed, len(finished)
    finally:
        _release_return_claim(stem)


_unfinished_listed: List = [None]


def _write_unfinished_attention(unfinished: Dict[str, List[Dict[str, str]]],
                                folders: Optional[Dict[str, str]] = None) -> None:
    """List the bundles still missing a file in this machine's attention file
    (watcher/attention.py), rewritten only when the list changes. Each item
    names the bundle (``folder``) and this machine (``host``), which the
    dashboard's Resolve dialog acts on (watcher/attention_actions.py). Never
    raises."""
    from .review_routing import attention_dir
    d = attention_dir()            # None: no NAS root, or queues outside it
    if d is None:
        return
    items = []
    for stem, missing in sorted(unfinished.items()):
        names = ", ".join(f"{m.get('name')} ({m.get('error') or 'not moved'})"
                          for m in missing)
        items.append({
            "kind": "route_unfinished", "needs_person": True, "video_id": stem,
            "folder": (folders or {}).get(stem), "host": socket.gethostname(),
            "says": (f"{stem} is in a review queue without all its files: {names}. "
                     f"The watcher moves a file in by itself once it can; if it stays "
                     f"listed, press 'Resolve...' in the Pipeline Dashboard (on the "
                     f"machine named in the item when the file is on that machine's "
                     f"own disk)."),
        })
    if _unfinished_listed[0] == items:
        return
    try:
        from mousereach.pipeline.fsutil import write_json_atomic
        host = socket.gethostname()
        target = d / f"{host}_unfinished_routes.json"
        if not items and not target.exists():
            # Nothing to list and no earlier list to empty: write nothing (a
            # machine that never had an unfinished route gets no file).
            _unfinished_listed[0] = items
            return
        d.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target,
                          {"written_by": host, "items": items}, indent=1)
        _unfinished_listed[0] = items
    except Exception as e:
        logger.warning("Return scan: could not write the unfinished-routes attention "
                       "list: %s", e)


def repair_unfinished_routes(force: bool = False, db=None,
                             signatures: Optional[Dict[str, tuple]] = None) -> Dict[str, int]:
    """Finish what an interrupted route left undone, in both queues (see the
    block comment above). Runs at most every ROUTE_REPAIR_INTERVAL_S unless
    ``force``. ``db``: the watcher database; a file at a route's source is
    moved or deleted only while it says the video is held in that queue
    (_finish_failed_moves). ``signatures``: bundle path -> _bundle_signature
    already taken this scan (saves listing those bundles again). Never
    raises."""
    out = {"manifests_written": 0, "route_files_finished": 0}
    now = time.monotonic()
    if not force and now < _route_repair_due[0]:
        return out
    _route_repair_due[0] = now + ROUTE_REPAIR_INTERVAL_S
    unfinished: Dict[str, List[Dict[str, str]]] = {}
    folders: Dict[str, str] = {}
    seen = set()
    for queue_root in (getattr(Paths, "TRIAGE_REVIEW", None),
                       getattr(Paths, "DEEP_REVIEW", None)):
        try:
            bundles = _bundles(queue_root)
        except OSError:
            continue
        for bundle in bundles:
            stem = bundle.name
            seen.add(str(bundle))
            try:
                signature = (signatures or {}).get(str(bundle))
                if signature is None:
                    signature = _bundle_signature(bundle)
            except OSError:
                continue
            if any(name == LEFT_NOTE for name, _, _ in signature):
                continue            # a leftover kept for a person, not a bundle
            try:
                failed, finished_later = _record_flags(bundle, stem, signature)
                if failed:
                    still, n_moved = _finish_failed_moves(bundle, stem, db=db)
                    out["route_files_finished"] += n_moved
                    if still:
                        unfinished[stem] = still
                        folders[stem] = str(bundle)
                    if n_moved:
                        _record_flags_cache.pop(str(bundle), None)
                        finished_later = True
                        failed = bool(still)
                if queue_root == getattr(Paths, "DEEP_REVIEW", None):
                    triage = getattr(Paths, "TRIAGE_REVIEW", None)
                    if (finished_later and not failed
                            and triage and (Path(triage) / stem).is_dir()
                            and not (Path(triage) / stem / LEFT_NOTE).exists()):
                        # A deep-review route that was unfinished is whole
                        # now (finished by the watcher or by a person): the
                        # triage copy kept while its files were missing is
                        # emptied (review_gate._drop_triage_copy). Checked on
                        # every pass, not only the pass that finished it, so
                        # a triage copy that could not be emptied then (a
                        # file held open) is emptied once it can be.
                        from .review_gate import _drop_triage_copy
                        _drop_triage_copy(stem, bundle)
            except Exception as e:
                logger.warning("Return scan: could not finish the route of %s: %s",
                               stem, e)
            if _repair_missing_manifest(bundle, stem, signature):
                out["manifests_written"] += 1
    for gone in [k for k in _record_flags_cache if k not in seen]:
        _record_flags_cache.pop(gone, None)
    _write_unfinished_attention(unfinished, folders)
    # The list of queue folders that need a person is checked against disk
    # here (only the listed folders), so a folder a person dealt with drops
    # off by itself.
    from .review_routing import refresh_for_a_person
    refresh_for_a_person()
    return out


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

    def _states(bundles):
        # One query for the whole queue (_superseded_by_archive needs each
        # video's state); without a database it is an empty map and that
        # check simply never fires.
        try:
            return db.states_of(b.name for b in bundles) if db is not None else {}
        except Exception:
            return {}

    def _retire_if_superseded(bundle, stem, row, signature, queue):
        """True when the bundle is a leftover of an archived video (it is
        cleared against the filed results, or left for the next scan if that
        could not be done)."""
        why = _superseded_by_archive(bundle, stem, row, signature)
        if not why:
            return False
        if _retire_stale_bundle(bundle, stem, "archived", why=why,
                                live=_live_copy_of(stem)):
            summary["superseded_retired"] = summary.get("superseded_retired", 0) + 1
            logger.info("Return scan: %s %s bundle cleared as a leftover: %s",
                        stem, queue, why)
        return True

    # TRIAGE: every triaged element resolved (and segmentation sound).
    seen_bundles = set()
    triage_signatures: Dict[str, tuple] = {}
    triage_bundles = _bundles(Paths.TRIAGE_REVIEW)
    triage_states = _states(triage_bundles)
    from .review_routing import deep_review_stems
    deep_stems = deep_review_stems()
    for bundle in triage_bundles:
        if not _budget_left():
            summary["deferred"] += 1
            continue
        stem = bundle.name
        seen_bundles.add(str(bundle))
        try:
            signature = _bundle_signature(bundle)
            triage_signatures[str(bundle)] = signature
            if any(name == LEFT_NOTE for name, _, _ in signature):
                # A leftover kept for a person (review_routing.clear_leftover):
                # not a bundle, never returned, diverted or cleared again here.
                continue
            if not signature:
                # An EMPTY dir is not a bundle -- it is residue of an earlier
                # divert/return that moved the files and left the folder.
                # One such folder hid behind the both-queues skip for days,
                # then diverted "(0 files)" and re-flipped a parked video the
                # moment its twin side cleared.
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
                                "bundle; removed", stem)
                continue
        except OSError:
            continue
        if _retire_if_superseded(bundle, stem, triage_states.get(stem), signature,
                                 "triage"):
            continue
        try:
            st = _cached_triage_status(bundle, stem, signature)
        except Exception as err:
            logger.warning('%s: could not read its review status, so it was skipped this scan: %s', stem, err)
            continue
        # A human's deep-review clearance outranks the machine's seg self-check
        # -- the SAME honor rule the gate applies (review_gate seg_failed
        # branch). Without it here, a cleared video with unresolved triage
        # segments ping-pongs forever: gate honors the clear and routes to
        # triage, this scan sees raw seg_failed and diverts to deep review,
        # the clear marker instantly releases it, the pipeline re-runs, repeat
        # (one video made dozens of laps).
        cleared = _deep_review_cleared(bundle, stem)
        # DEEP REVIEW SUPERSEDES TRIAGE (review_routing): whatever this triage
        # copy's status, a video that is in deep review is never returned,
        # released or diverted from the triage queue -- the deep-review copy is
        # the one that counts. Since the rule was made, any move into deep
        # review empties the triage copy without losing a person's file --
        # what remains leaves the queue, kept for a person
        # (review_gate._drop_triage_copy); a copy left from before is handled
        # here as below.
        if stem in deep_stems:
            _dr = getattr(Paths, "DEEP_REVIEW", None)
            if _dr:
                # A same-stem bundle ALREADY sits in deep review (historical
                # escalations split bundles across queues: jsons moved, mp4
                # left behind). Diverting would merge blindly into the
                # reviewer's copy, and retrying the impossible move every
                # scan consumed the whole return budget -- the queue
                # live-locked with human-cleared releases
                # starved behind it. A triage copy that is only a byte-identical
                # leftover of the deep-review bundle is deleted (it holds
                # nothing the reviewer's copy lacks); any other copy is emptied
                # the way a route into deep review empties it
                # (review_gate._drop_triage_copy): what has a current copy
                # elsewhere is deleted, the rest stays where it is, listed for
                # a person. No budget is spent and no route is attempted.
                try:
                    leftover = _is_leftover_copy(bundle, Path(_dr) / stem,
                                                 signature)
                except OSError:
                    leftover = False
                twin = Path(_dr) / stem
                if leftover:
                    try:
                        live = {g.name: g for g in twin.iterdir() if g.is_file()}
                    except OSError:
                        live = {}
                    if live and _retire_stale_bundle(
                            bundle, stem, "(copy in deep review)",
                            why="leftover of a split escalation: every file is a "
                                "byte-identical copy of one in the deep review "
                                "bundle, which is the copy being reviewed",
                            live=live):
                        summary["leftover_retired"] = summary.get("leftover_retired", 0) + 1
                        logger.info("Return scan: %s triage copy is a byte-identical "
                                    "leftover of its deep review bundle; deleted", stem)
                        continue
                if stem not in _both_queues_logged:
                    _both_queues_logged.add(stem)
                    logger.warning(
                        "Return scan: %s is in deep review and has a triage copy "
                        "that differs from it; the triage copy is ignored (deep "
                        "review supersedes triage) and emptied as a leftover: what "
                        "has no current copy elsewhere stays, listed under 'Folders "
                        "left for a person' in the Review Queues tab", stem)
                    from .review_gate import _drop_triage_copy
                    if twin.is_dir():
                        _drop_triage_copy(stem, twin)
                continue
        if st.seg_failed:
            # STALE-BUNDLE GUARD: if the video's CURRENT state says the
            # pipeline has re-handled it since this bundle was staged
            # (re-segmented clean, review applied, finished or filed), the
            # bundle is a leftover from the earlier failed era -- diverting it
            # would flip a healthy video into deep_review and block its
            # archive. That has happened: videos re-completed
            # CLEAN seconds earlier were dragged back by their own stale
            # triage husks. Clear the husk as a leftover against the video's
            # current copy (_retire_stale_bundle) and leave the video alone.
            try:
                row = db.get_video(stem) if db else None
                cur_state = (row or {}).get('state')
            except Exception:
                cur_state = None
            # 'unresolvable' guards too: a person parked that video with a
            # reason, and a husk must never un-park it.
            if cur_state in ('processing', 'processed', 'archiving',
                             'archived', 'unresolvable'):
                if _retire_stale_bundle(bundle, stem, cur_state,
                                        live=_live_copy_of(stem, processing_dir)):
                    summary["stale_retired"] = summary.get("stale_retired", 0) + 1
                    logger.info(
                        "Return scan: %s triage bundle is STALE (video is now "
                        "'%s'); cleared as a leftover instead of diverting",
                        stem, cur_state)
                continue
            # A human already cleared the failed segmentation: the deep-review
            # question is ANSWERED, so diverting would only lap the cycle.
            # Fall through instead -- the bundle is an ordinary triage bundle
            # now, waiting for its triaged segments to be answered.
            if not cleared:
                # A seg-failed bundle can NEVER satisfy the triage release
                # condition below, and nothing else moves it -- it sat in the
                # triage queue with no route out. A failed segmentation makes the whole video
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
        # A triaged segment too short to be a pellet presentation can never be
        # answered here (the review tool refuses it): the cuts are the
        # question, so the bundle goes to deep review -- unless a person has
        # cleared it there and kept those cuts, when there is no pellet in such
        # a segment to ask about (the gate applies the same rule).
        short = sorted(s for s in st.unresolved if s in st.too_short)
        if short and not cleared:
            if not _claim_return(stem):
                continue
            try:
                from .review_gate import route_to_queue
                from mousereach.review.triage_status import short_segment_reason
                route_to_queue(stem, bundle, Paths.DEEP_REVIEW,
                               reason=short_segment_reason(st, short),
                               db=db, db_state="deep_review")
                summary["diverted_to_deep"] += 1
                logger.info("Return scan: %s diverted triage -> deep review (segment(s) "
                            "%s too short to be a pellet; the cuts need fixing)",
                            stem, short)
            except Exception as e:
                logger.error("Return scan: could not divert %s to deep review: %s",
                             stem, e)
            finally:
                _release_return_claim(stem)
            continue
        still_open = set(st.unresolved) - set(short)
        if not (st.has_triage and not still_open
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
    # Routes that did not finish (block comment at ROUTE_REPAIR_INTERVAL_S),
    # reusing the triage listings this pass already took.
    try:
        summary.update(repair_unfinished_routes(db=db, signatures=triage_signatures))
    except Exception as e:      # never stops the returns
        logger.warning("Return scan: could not check for unfinished routes: %s", e)

    # DEEP_REVIEW: an explicit clear marker or in-bundle GT.
    # Own budget, NOT shared with the triage loop's: the triage loop's
    # diverts once consumed the whole shared budget every scan and
    # human-released deep bundles were starved indefinitely. Releases of
    # finished human work must never queue behind machine-initiated moves.
    recheck_deadline = time.monotonic() + QC_RECHECK_BUDGET_S
    summary.setdefault("qc_released", 0)
    deep_bundles = _bundles(Paths.DEEP_REVIEW)
    deep_states = _states(deep_bundles)
    for bundle in deep_bundles:
        if summary["deep_returned"] >= limit:
            summary["deferred"] += 1
            continue
        stem = bundle.name
        # A leftover kept for a person (review_routing.clear_leftover) is not a
        # bundle: never returned or released.
        if (bundle / LEFT_NOTE).exists():
            continue
        # No empty-dir retire here: an empty folder has no clear marker, so it
        # is never acted on, and EMPTY_DIR_GRACE_SECONDS has nothing to guard.
        if _deep_review_cleared(bundle, stem):
            # A clear releases the video only once a person has re-segmented
            # it since it arrived here (review/resegmented.py says why). Easy
            # and Flat trays are never segmented, so the rule does not apply.
            if not is_outcome_free_tray(stem):
                from mousereach.review.resegmented import resegmented_since_entry
                ok, why = resegmented_since_entry(bundle, stem)
                if not ok:
                    if stem not in _held_unsegmented_logged:
                        _held_unsegmented_logged.add(stem)
                        logger.warning("Re-segmentation queue: %s is cleared but %s",
                                       stem, why)
                    continue
            if not _return_backing_off(stem) and _return_to_processing(
                    bundle, stem, processing_dir, db, "deep_review_cleared"):
                summary["deep_returned"] += 1
            continue
        row = deep_states.get(stem)
        if row and row[0] == "archived":
            try:
                signature = _bundle_signature(bundle)
            except OSError:
                signature = ()
            if signature and _retire_if_superseded(bundle, stem, row, signature,
                                                   "deep review"):
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
        # An Easy or Flat video is never held (review_gate), so a
        # machine-made hold on one -- a quality flag or a failed segmentation
        # -- is released even when a person's file sits in the bundle: what a
        # person can answer there is a pellet outcome, and outcomes do not
        # matter for these trays. The person's file is not lost: it stays in
        # the bundle folder, listed for a person (_release_to_posed). A session a person
        # flagged by hand ("reviewer flagged session ...") is their call and
        # stays.
        outcome_free = is_outcome_free_tray(stem)
        if _person_touched(signature) and not outcome_free:
            continue
        if outcome_free:
            lifted = _routed_reason(bundle, stem) in ("qc_needs_review", "segmentation_failed")
        else:
            lifted = _qc_hold_lifted(bundle, stem, signature)
        if lifted and _release_to_posed(bundle, stem, db):
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
