"""mousereach-repose-videos -- send named videos back to a GPU for a new pose.

WHY THIS EXISTS
---------------
The re-pose round trip (watcher/repose.py) asks a GPU node for a new pose when
the DECLARED MODEL changes: every decision in it is keyed on the scorer name in
the pose file. It cannot help when the model is the same but the POSE ITSELF is
not trusted -- e.g. poses made by a machine whose output the lab has decided
not to keep (a CPU pose pool whose poses differ slightly from the GPU nodes').
Such a pose carries the declared scorer, so:

  * the version scan calls the video current and never marks it;
  * the publish and adopt steps treat a declared-scorer pose in the posed
    folder or in the results folder as "already satisfied" and would narrow
    the row straight to a re-run ON THAT SAME POSE, with no GPU involved;
  * a GPU node closes a request when it finds a declared-scorer pose in the
    archive folder.

This command takes a list of video ids and does, for each one, the few steps
that let the existing round trip carry it, IN THIS ORDER:

  1. MOVE every live copy of the distrusted pose out of every place the round
     trip looks: the video's results folder in Analyzed (and any
     ``DLC Model N`` copy of that folder), its review bundle (triage or deep
     review), and its copy in the posed folder. The pose and every output
     computed from it go to ``Analyzed/Archive/<label>/`` through the normal
     checksum-verified supersede move. A separate archive root, not the usual
     ``DLC Model N`` folder, because the distrusted pose has the SAME name as
     the pose that will replace it -- mixed together nobody could tell later
     which file came from which machine.
  2. CHECK that no pose of the video is left in any of those places. If one
     is, the video is left exactly as it now is and reported -- its row is
     NOT marked (marking it would let the server re-run on the old pose).
  3. MARK the row 'outdated' with reprocess_scope 'full' and a mark_reason.
     The reason is what keeps the version scan from un-marking it: the scan
     never un-marks a row that carries one.
  4. WRITE the re-pose request into the shared Repose_Queue, naming the video
     file a GPU node should copy.

The video file itself is never archived and never deleted:

  * a FILED video's mp4 stays where it is, beside its human review (if any);
    the re-run copies that folder down and applies the review again. A
    person's cuts there (``_segments.json`` marked ``boundary_source:
    "human"``) stay too -- cuts are frame numbers, valid on any pose, and the
    segmenter keeps them on the re-run (archive/supersede.py keeps them);
  * a HELD video's bundle (triage / deep review) leaves its queue FIRST, as
    one folder rename to ``Processing/Repose_Queue/_parked/<video_id>/``,
    under the review queues' own claim (step 0 in the code says why); the
    pose and outputs are then moved out of the parked folder, so what stays
    there is the mp4 and the queue's notes, and the request names that mp4.
    A run stopped after the rename finishes from the parked folder when run
    again. WHY there: the request must name a file a GPU node may read
    (never the posed folder -- that is the server's intake), and the review
    queues are not searched for videos. The posed folder's duplicate mp4 goes
    to ``_parked/<video_id>/posed_copy/``.

REFUSED, left untouched and reported:
  * a held video with a saved human review (``_causal_review.json``,
    ``_deep_review_cleared.json`` or ``_unified_ground_truth.json`` in its
    bundle, or a durable review), or with a person's cuts in its bundle (a
    ``_segments.json`` marked ``boundary_source: "human"``): a person is
    working on it; ask them first;
  * a row in any state other than archived / triage / deep_review (in flight,
    crystallized, failed ...), or with no row at all;
  * a video whose request is already out (queued or claimed by a GPU node).

Then the round trip takes over unchanged: a GPU node claims the request,
copies the video, poses it and stages the pose into the posed folder; the
server adopts it and re-runs every step after posing.

AFTER THE ROUND TRIP: THE PARKED FOLDER IS CLEARED
--------------------------------------------------
The re-run copies the parked video down, and when it ends the video has a
new home: its filed folder in Analyzed, or a new bundle in a review queue.
The parked folder is then a leftover copy, which would stay for good --
and while it existed this command would refuse any later re-pose of the video.
``clear_parked`` deletes it, but only when every one of these holds
(the lab's rule for machine leftovers: deleted once a current copy elsewhere
is verified, never set aside, and anything that may be a person's work is
never deleted):

  * no re-pose request for the video is queued or held by a GPU node;
  * the video's row is filed ('archived' / 'crystallized') or held in a
    review queue ('triage' / 'deep_review') -- never mid-run;
  * its current copy is there: ``<id>.mp4`` in the filed folder, or in the
    bundle of the queue its row names; and the parked ``<id>.mp4`` (and the
    ``posed_copy/<id>.mp4``, when there is one) has the same size and the
    same SHA-256 as that copy;
  * the parked folder holds nothing else than the video, the queue's notes
    (``<id>_manifest.json``, ``<id>_routing.json``) and ``posed_copy/``. Any
    other file is left where it is, with the folder, and named in the report.

It runs (1) here, when a re-pose meets an old parked folder (instead of
refusing); (2) by itself on the processing server, a few folders per hour
(the watcher's scan, ``ProcessingOrchestrator._maybe_clear_finished_parked``);
and (3) on demand: ``mousereach-repose-videos --clear-finished [--dry-run]``.

Run ``--dry-run`` first: it prints exactly what would move where and changes
nothing. ASCII-only console output (Windows consoles cannot print Unicode).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from mousereach.config import Paths
from mousereach.watcher import repose

logger = logging.getLogger(__name__)

PARKED = "_parked"                       # subfolder of the Repose_Queue
_ALGO = ("_reaches.json", "_features.json", "_pellet_outcomes.json",
         "_segments.json", "_processing_manifest.json", "_triage.json",
         "_reach_assignments.json", "_results.csv")   # supersede's algo outputs
OK_STATES = ("archived", "triage", "deep_review")
HUMAN_SUFFIXES = ("_causal_review.json", "_deep_review_cleared.json",
                  "_unified_ground_truth.json")
POSE_EXTS = (".h5", ".csv", ".pickle")


def _is_pose(p: Path, video_id: str) -> bool:
    return (p.name.startswith(video_id + "DLC") and p.suffix.lower() in POSE_EXTS)


def _files(folder: Optional[Path], video_id: str) -> List[Path]:
    """This video's files directly in ``folder`` (none when it is missing)."""
    if folder is None:
        return []
    n = len(video_id)

    def mine(name: str) -> bool:
        # "<id>" followed by a non-digit: P1 must not pick up a P10.
        return (name.startswith(video_id)
                and (len(name) == n or not name[n].isdigit()))
    try:
        with os.scandir(folder) as it:
            return sorted(Path(e.path) for e in it if mine(e.name) and e.is_file())
    except OSError:
        return []


def _results_dir(video_id: str) -> Optional[Path]:
    try:
        from mousereach.archive.core import get_archive_destination
        return Path(get_archive_destination(video_id))
    except Exception:
        return None


def _model_dirs(results: Optional[Path]) -> List[Path]:
    """``Analyzed/<project>/DLC Model N/<cohort>`` folders beside the results
    folder: the per-model pose trees, which the re-run's pose search reads."""
    if results is None:
        return []
    try:
        return sorted(d / results.name for d in results.parent.glob("DLC Model *")
                      if (d / results.name).is_dir())
    except OSError:
        return []


def _bundle(video_id: str) -> Optional[Path]:
    # Deep review first: it supersedes triage (review_routing.py).
    for q in (getattr(Paths, "DEEP_REVIEW", None), getattr(Paths, "TRIAGE_REVIEW", None)):
        if q and (Path(q) / video_id).is_dir():
            return Path(q) / video_id
    return None


def _incoming(video_id: str) -> bool:
    try:
        from mousereach.watcher.review_routing import INCOMING_DIR_NAME
    except ImportError:
        INCOMING_DIR_NAME = ".incoming"
    for q in (getattr(Paths, "TRIAGE_REVIEW", None), getattr(Paths, "DEEP_REVIEW", None)):
        if q and (Path(q) / INCOMING_DIR_NAME / video_id).exists():
            return True
    return False


def _durable_review(video_id: str) -> Optional[Path]:
    try:
        from mousereach.review.causal_review_io import durable_review_dir
        d = durable_review_dir()
        if d is not None:
            p = Path(d) / f"{video_id}_causal_review.json"
            if p.is_file():
                return p
    except Exception:
        pass
    return None


def _scorer(poses: Iterable[Path]) -> str:
    for p in poses:
        if p.suffix.lower() == ".h5":
            s = repose.scorer_of(p)
            if s:
                return s
    return ""


def _manifest_for(folder: Path, video_id: str, scorer: str) -> dict:
    """The processing manifest of ``folder`` with its pose scorer set to the
    pose actually there, so supersede picks the pose files up. The posed copy
    has no manifest of its own."""
    m = {}
    try:
        p = folder / f"{video_id}_processing_manifest.json"
        if p.is_file():
            m = json.loads(p.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        m = {}
    m = dict(m)
    m["dlc_model"] = dict(m.get("dlc_model") or {}, dlc_scorer=scorer)
    return m


def _move(src: Path, dst_dir: Path, dry_run: bool) -> Optional[Path]:
    """Move one file into ``dst_dir``, never over an existing file."""
    dst = dst_dir / src.name
    if dry_run:
        return dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"{dst} already exists; nothing was overwritten")
    shutil.move(str(src), str(dst))
    return dst


PARKED_NOTES = ("_manifest.json", "_routing.json")   # the old queue's own notes
POSED_COPY = "posed_copy"
DONE_STATES = ("archived", "crystallized")
QUEUE_STATES = ("triage", "deep_review")


def _sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _same_file(a: Path, b: Path) -> bool:
    """Same size and same SHA-256: the verification the lab's leftover rule
    asks for before a copy is deleted (size alone is not proof)."""
    try:
        return a.stat().st_size == b.stat().st_size and _sha256(a) == _sha256(b)
    except OSError:
        return False


def _current_copy(row: dict, video_id: str,
                  results_dir_of: Optional[Callable[[str], Optional[Path]]] = None
                  ) -> Tuple[Optional[Path], str]:
    """(the video's current mp4, where) -- the copy that makes the parked one
    a leftover -- or (None, why there is none)."""
    state = row.get("state")
    if state in DONE_STATES:
        results = (results_dir_of or _results_dir)(video_id)
        mp4 = results / f"{video_id}.mp4" if results is not None else None
        if mp4 is not None and mp4.is_file():
            return mp4, "filed"
        return None, "it is filed but its filed folder has no video file"
    if state in QUEUE_STATES:
        q = getattr(Paths, "DEEP_REVIEW" if state == "deep_review" else "TRIAGE_REVIEW", None)
        mp4 = Path(q) / video_id / f"{video_id}.mp4" if q else None
        if mp4 is not None and mp4.is_file():
            return mp4, state
        return None, (f"its row says '{state}' but that queue holds no bundle with its "
                      f"video (a re-pose run that stopped part way leaves it so: run "
                      f"mousereach-repose-videos again for it)")
    return None, f"its row is '{state}' (only a filed or held video's copy is verified)"


def clear_parked(db, video_id: str, *, dry_run: bool = True,
                 results_dir_of: Optional[Callable[[str], Optional[Path]]] = None) -> dict:
    """Delete ``<Repose_Queue>/_parked/<video_id>/`` once its round trip has
    ended and the video's current copy is verified (module docstring, "AFTER
    THE ROUND TRIP"). Returns ``{"video_id", "status", "why", "removed"}``;
    status is 'cleared', 'would_clear' (dry run), 'kept' or 'none' (no
    parked folder). Never raises; anything uncertain keeps the folder."""
    rep = {"video_id": video_id, "status": "kept", "why": "", "removed": []}
    try:
        qdir = repose.repose_queue_dir()
        parked = Path(qdir) / PARKED / video_id if qdir else None
        if parked is None or not parked.is_dir():
            rep["status"], rep["why"] = "none", "no parked folder"
            return rep
        if (repose.queue_path(video_id, qdir).exists()
                or repose.inflight_path(video_id, qdir).exists()):
            rep["why"] = "its re-pose request is still out (round trip under way)"
            return rep
        row = db.get_video(video_id) if db is not None else None
        if not row:
            rep["why"] = "no row for it in this machine's database"
            return rep
        current, where = _current_copy(row, video_id, results_dir_of)
        if current is None:
            rep["why"] = where
            return rep
        # Only files this cleanup knows; anything else may be a person's.
        files, unknown = [], []
        for e in os.scandir(parked):
            p = Path(e.path)
            if e.is_dir() and e.name == POSED_COPY:
                for f in os.scandir(p):
                    if f.is_file() and f.name == f"{video_id}.mp4":
                        files.append(Path(f.path))
                    else:
                        unknown.append(f"{POSED_COPY}/{f.name}")
            elif e.is_file() and (e.name == f"{video_id}.mp4"
                                  or e.name in tuple(video_id + s for s in PARKED_NOTES)):
                files.append(p)
            else:
                unknown.append(e.name)
        if unknown:
            rep["why"] = ("it holds files this clean-up does not know, left for a "
                          "person: " + ", ".join(sorted(unknown)))
            return rep
        for f in files:
            if f.suffix.lower() == ".mp4" and not _same_file(f, current):
                rep["why"] = (f"{f.name} in it differs from the current copy ({current}); "
                              f"left for a person")
                return rep
        rep["why"] = f"its video is verified identical to the {where} copy {current}"
        if dry_run:
            rep["status"] = "would_clear"
            rep["removed"] = [str(f) for f in files]
            return rep
        # Asked once more right before deleting: hashing two videos takes a
        # while, and a re-pose request or a move of the current copy may have
        # happened meanwhile.
        if (repose.queue_path(video_id, qdir).exists()
                or repose.inflight_path(video_id, qdir).exists()
                or not current.is_file()):
            rep["why"] = "its round trip or its current copy changed while it was checked"
            return rep
        for f in files:
            f.unlink()
            rep["removed"].append(str(f))
        if (parked / POSED_COPY).is_dir():
            (parked / POSED_COPY).rmdir()
        parked.rmdir()
        rep["status"] = "cleared"
        try:
            db.log_step(video_id, repose.STEP, "parked_cleared",
                        message=f"parked copy {parked} deleted: {rep['why']}")
        except Exception:
            pass
    except Exception as e:
        rep["status"], rep["why"] = "kept", f"could not check or clear it: {e}"
    return rep


def parked_videos() -> List[str]:
    """Every video id with a folder in ``<Repose_Queue>/_parked/``."""
    qdir = repose.repose_queue_dir()
    if not qdir:
        return []
    try:
        with os.scandir(Path(qdir) / PARKED) as it:
            return sorted(e.name for e in it if e.is_dir())
    except OSError:
        return []


def clear_finished(db, *, dry_run: bool = True, limit: int = 0,
                   skip: Iterable[str] = ()) -> List[dict]:
    """``clear_parked`` for every parked folder (at most ``limit`` checked, 0 =
    all; ids in ``skip`` are passed over). Never raises."""
    out = []
    skip = set(skip)
    for vid in parked_videos():
        if vid in skip:
            continue
        if limit and len(out) >= limit:
            break
        out.append(clear_parked(db, vid, dry_run=dry_run))
    return out


def plan_and_move(db, video_id: str, *, label: str, reason: str, hostname: str,
                  declared: str, dry_run: bool = True,
                  results_dir_of: Optional[Callable[[str], Optional[Path]]] = None) -> dict:
    """Do steps 1-4 (module docstring) for one video. Returns a report dict:
    ``status`` is 'requested', 'would_request', 'refused' or 'incomplete'."""
    rep = {"video_id": video_id, "status": "", "why": "", "where": "",
           "moved": [], "parked": [], "mp4": "", "request": ""}

    def refuse(why):
        rep["status"], rep["why"] = "refused", why
        return rep

    row = db.get_video(video_id)
    if not row:
        return refuse("no row in this node's database")
    state = row.get("state")
    if state not in OK_STATES:
        return refuse(f"row is '{state}'; only {', '.join(OK_STATES)} are re-posed")
    from mousereach.watcher.db import crystallized_refusal
    locked = crystallized_refusal(db, video_id, "re-posed")
    if locked:
        return refuse(locked)
    qdir = repose.repose_queue_dir()
    if not qdir:
        return refuse("no shared Repose_Queue configured on this node")
    if (repose.queue_path(video_id, qdir).exists()
            or repose.inflight_path(video_id, qdir).exists()):
        return refuse("a re-pose request for it is already out")
    if _incoming(video_id):
        return refuse("its bundle is being routed into a review queue right now")

    results = (results_dir_of or _results_dir)(video_id)
    bundle = _bundle(video_id)
    posed = Path(Paths.DLC_STAGING) if getattr(Paths, "DLC_STAGING", None) else None
    archive_root = Path(Paths.ANALYZED_OUTPUT) / "Archive" / label
    parked = Path(qdir) / PARKED / video_id

    if bundle is not None:
        rep["where"] = bundle.parent.name
        from mousereach.archive.supersede import is_human_cuts
        human = [f.name for f in _files(bundle, video_id)
                 if f.name.endswith(HUMAN_SUFFIXES)
                 or (f.name.endswith("_segments.json") and is_human_cuts(f))]
        durable = _durable_review(video_id)
        if human or durable is not None:
            return refuse("held for a person who has saved work on it ("
                          + ", ".join(human + ([durable.name + " (durable)"]
                                               if durable else [])) + ")")
        if parked.exists():
            # Left from an earlier round trip that has ended: cleared when its
            # current copy is verified (clear_parked), else refused as before.
            cleared = clear_parked(db, video_id, dry_run=dry_run)
            if cleared["status"] not in ("cleared", "would_clear"):
                return refuse(f"{parked} already exists and was not cleared: "
                              f"{cleared['why']}; nothing is overwritten")
            rep["parked"].append(f"old parked copy {parked}: {cleared['status']} "
                                 f"({cleared['why']})")
    elif state in ("triage", "deep_review") and parked.is_dir():
        # A run stopped after the bundle left its queue: finish from the
        # parked folder (the moves below are no-ops for what already went).
        rep["where"] = "parked (resumed)"
    elif results is not None and _files(results, video_id):
        rep["where"] = "filed"
    else:
        return refuse("no results folder and no review bundle found")
    held = bundle is not None or rep["where"] == "parked (resumed)"

    # ---- 0. a held bundle leaves its queue in ONE rename, under the queue's
    # own claim. WHY: moving files out of the bundle
    # while it still sat in the queue would let the watcher's review-return scan
    # read a bundle with no segments file as "segmentation failed" and
    # diverted what was left of it into deep review in the middle of the
    # move. Renaming the whole folder first means the queue only ever holds
    # the complete bundle or nothing, and taking the claim the scan takes
    # (``.return_claims/<video>.claim``) means the scan is never halfway
    # through moving it when the rename happens. The claim is held under its
    # own name, not this machine's, because the watcher on the same machine
    # would otherwise count it as its own.
    if bundle is not None and not dry_run:
        claim_dir = Path(Paths.REVIEW_ROOT) / ".return_claims" if getattr(
            Paths, "REVIEW_ROOT", None) else None
        holder = f"{hostname}-repose"
        from mousereach.watcher import claim_marker
        if claim_dir is not None and not claim_marker.take(
                claim_dir, video_id, holder, suffix=".claim"):
            return refuse("the watcher is moving this bundle right now; run it again later")
        try:
            if not bundle.is_dir():
                return refuse("its bundle left the queue while this run looked at it")
            parked.parent.mkdir(parents=True, exist_ok=True)
            os.rename(bundle, parked)
        except OSError as e:
            rep["status"], rep["why"] = "incomplete", f"could not park the bundle: {e}"
            return rep
        finally:
            if claim_dir is not None:
                claim_marker.release(claim_dir, video_id, holder, suffix=".claim")
        rep["parked"].append(f"{bundle} -> {parked}")

    # ---- 1. move the distrusted pose (and its outputs) out of every place ----
    from mousereach.archive.supersede import supersede_video_outputs
    # WHY the results folder even for a held video: a video re-run while filed
    # can be held AND have a pose in Analyzed; any declared-scorer pose left
    # there would be adopted as "the new pose" without a GPU.
    here = (bundle if (dry_run and bundle is not None) else parked) if held else None
    places = [p for p in (here, results) if p is not None]
    places += _model_dirs(results)
    if posed is not None:
        places.append(posed)
    for place in places:
        files = _files(place, video_id)
        poses = [f for f in files if _is_pose(f, video_id)]
        if not poses:
            continue
        scorer = _scorer(poses) or declared
        sup = supersede_video_outputs(
            video_id, place, archive_root,
            manifest=_manifest_for(place, video_id, scorer),
            dry_run=dry_run,
            # Only this video's own files (supersede's own glob "<id>*" would
            # also take a "<id>0" neighbour's); in the posed folder only the pose.
            only=({f.name for f in poses} if place == posed
                  else {f.name for f in files}))
        if sup.get("failed") or sup.get("error"):
            rep["status"] = "incomplete"
            rep["why"] = f"could not move from {place}: {sup.get('failed') or sup.get('error')}"
            return rep
        for n in sup.get("pose", []) + sup.get("algo", []):
            rep["moved"].append(f"{place}/{n} -> {archive_root}")

    # ---- the video file: kept in place (filed) or parked (held) ----
    try:
        if not held:
            mp4 = results / f"{video_id}.mp4"
            if not mp4.is_file() and not dry_run:
                rep["status"], rep["why"] = "incomplete", f"no video file in {results}"
                return rep
        else:
            # The bundle already left its queue as one folder (step 0); what
            # stays in it is the video and the queue's notes.
            mp4 = parked / f"{video_id}.mp4"
            if dry_run:
                for f in _files(bundle, video_id):
                    if _is_pose(f, video_id) or any(f.name.endswith(s) for s in _ALGO):
                        continue                    # would have gone to the archive
                    rep["parked"].append(f"{f} -> {parked / f.name}")
        if posed is not None:
            for f in _files(posed, video_id):
                if f.suffix.lower() == ".mp4":
                    rep["parked"].append(
                        f"{f} -> {_move(f, parked / 'posed_copy', dry_run)}")
        if held and not dry_run and not mp4.is_file():
            alt = parked / "posed_copy" / f"{video_id}.mp4"
            if not alt.is_file():
                rep["status"] = "incomplete"
                rep["why"] = f"no video file in its bundle or the posed folder; row NOT marked"
                return rep
            mp4 = alt
        if posed is not None:
            if not dry_run:
                try:
                    from mousereach.watcher import claim_marker
                    claim_marker.release(posed / ".claims", video_id, hostname)
                except Exception as e:
                    logger.debug(f"{video_id}: posed claim not released ({e})")
    except OSError as e:
        rep["status"], rep["why"] = "incomplete", f"move failed: {e}"
        return rep
    rep["mp4"] = str(mp4)

    # ---- 2. nothing of the old pose may be left where the round trip looks ----
    if not dry_run:
        left = [str(f) for place in places for f in _files(place, video_id)
                if _is_pose(f, video_id)]
        if left:
            rep["status"] = "incomplete"
            rep["why"] = "pose still present, row NOT marked: " + ", ".join(left)
            return rep

    body = {
        "video_id": video_id,
        "declared_scorer": declared,
        "video_rel": repose._relative_to_root(mp4),
        "video_path": str(mp4),
        "requested_by": hostname,
        "requested_at": datetime.now().isoformat(),
        "reason": f"pose not trusted ({reason})",
    }
    rep["request"] = str(repose.queue_path(video_id, qdir))
    if dry_run:
        rep["status"] = "would_request"
        return rep

    # ---- 3. mark the row (only now: the old pose is gone) ----
    db.update_state(video_id, "outdated", reprocess_scope="full",
                    mark_reason=reason, dlc_output_path=None,
                    current_path=str(mp4), error_message=None)
    # ---- 4. the request ----
    repose.write_json_atomic(repose.queue_path(video_id, qdir), body)
    try:
        db.log_step(video_id, repose.STEP, "published",
                    message=f"repose-videos: {reason}; moved {len(rep['moved'])} "
                            f"file(s) to {archive_root}; video {mp4}")
    except Exception:
        pass
    rep["status"] = "requested"
    return rep


def _main_clear_finished(args) -> int:
    """mousereach-repose-videos --clear-finished: clear_parked for every
    parked folder, printing each result. ASCII only."""
    from mousereach.watcher.db_location import open_watcher_db_if_present
    db, why = open_watcher_db_if_present()
    if db is None:
        print(f"ERROR: {why}; nothing done. Run 'mousereach-watch' once on this "
              f"machine first.", file=sys.stderr)
        return 2
    print(f"[watcher db] {db.db_path}")
    reports = clear_finished(db, dry_run=args.dry_run, limit=args.limit or 0)
    counts: Dict[str, int] = {}
    for r in reports:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        print(f"{r['video_id']}: {r['status']} -- {r['why']}")
    print("")
    print("Summary: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                         or "no parked folders"))
    if args.dry_run:
        print("DRY RUN -- nothing was deleted.")
    if args.report:
        Path(args.report).write_text(json.dumps(reports, indent=1), encoding="utf-8")
        print(f"Report written to {args.report}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="mousereach-repose-videos",
        description="Send named videos back to a GPU node for a new pose, because "
                    "their current pose is not trusted. Moves the old pose and every "
                    "output made from it to Analyzed/Archive/<label>/, then marks the "
                    "video for a new pose and writes the re-pose request. Run on the "
                    "processing server. Use --dry-run first.")
    ap.add_argument("--ids",
                    help="text file with one video id per line")
    ap.add_argument("--label",
                    help="name of the archive folder the old outputs go to "
                         "(Analyzed/Archive/<label>/), e.g. 'CPU pose <dates>'")
    ap.add_argument("--reason",
                    help="why, recorded on each video (required)")
    ap.add_argument("--clear-finished", action="store_true",
                    help="instead of re-posing: delete every parked folder "
                         "(Processing/Repose_Queue/_parked/<video>/) whose round trip "
                         "has ended and whose video is verified identical to its "
                         "current copy (filed, or in a review queue). Anything else is "
                         "kept and listed with the reason. Use --dry-run first.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would happen; change nothing")
    ap.add_argument("--limit", type=int, default=0,
                    help="act on at most this many videos (0 = all)")
    ap.add_argument("--report", help="also write the per-video report here (JSON)")
    args = ap.parse_args(argv)

    if args.clear_finished:
        return _main_clear_finished(args)
    if not (args.ids and args.label and args.reason):
        print("ERROR: --ids, --label and --reason are required (or use --clear-finished)",
              file=sys.stderr)
        return 2
    ids = [ln.strip() for ln in Path(args.ids).read_text(encoding="utf-8").splitlines()
           if ln.strip() and not ln.startswith("#")]
    if args.limit:
        ids = ids[:args.limit]
    if not args.reason.strip() or not args.label.strip():
        print("ERROR: --reason and --label must not be empty", file=sys.stderr)
        return 2
    declared = repose.declared_scorer()
    if not declared:
        print("ERROR: the declared DLC model could not be read from "
              "pipeline_versions.json; nothing done", file=sys.stderr)
        return 2
    from mousereach.watcher.db_location import open_watcher_db_if_present
    # Never create the database: a re-pose recorded in a new empty file is a
    # request the watcher never sees, while the bundles have already moved.
    db, why = open_watcher_db_if_present()
    if db is None:
        print(f"ERROR: {why}; nothing done. Run 'mousereach-watch' once on this "
              f"machine first.", file=sys.stderr)
        return 2
    print(f"[watcher db] {db.db_path}")
    host = socket.gethostname()
    reports, counts = [], {}
    for vid in ids:
        try:
            r = plan_and_move(db, vid, label=args.label, reason=args.reason.strip(),
                              hostname=host, declared=declared, dry_run=args.dry_run)
        except Exception as e:
            r = {"video_id": vid, "status": "incomplete", "why": f"error: {e}"}
        reports.append(r)
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        line = f"{vid}: {r['status']}"
        if r.get("where"):
            line += f" [{r['where']}]"
        if r.get("why"):
            line += f" -- {r['why']}"
        print(line)
        for m in r.get("moved", []) + r.get("parked", []):
            print(f"    {m}")
        if r.get("request"):
            print(f"    request {r['request']} names {r.get('mp4')}")
    print("")
    print("Summary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if args.dry_run:
        print("DRY RUN -- nothing was moved, marked or requested.")
    if args.report:
        Path(args.report).write_text(json.dumps(reports, indent=1), encoding="utf-8")
        print(f"Report written to {args.report}")
    return 0 if not counts.get("incomplete") else 1


if __name__ == "__main__":
    sys.exit(main())
