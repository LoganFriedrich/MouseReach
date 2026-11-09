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
    the re-run copies that folder down and applies the review again;
  * a HELD video's bundle (triage / deep review) is taken out of its queue:
    its mp4 and the queue's own notes go to
    ``Processing/Repose_Queue/_parked/<video_id>/`` and the request names that
    file. WHY there: the request must name a file a GPU node may read
    (never the posed folder -- that is the server's intake), and the review
    queues are not searched for videos. The posed folder's duplicate mp4 goes
    to ``_parked/<video_id>/posed_copy/``.

REFUSED, left untouched and reported:
  * a held video with a saved human review (``_causal_review.json``,
    ``_deep_review_cleared.json`` or ``_unified_ground_truth.json`` in its
    bundle, or a durable review): a person is working on it; ask them first;
  * a row in any state other than archived / triage / deep_review (in flight,
    crystallized, failed ...), or with no row at all;
  * a video whose request is already out (queued or claimed by a GPU node).

Then the round trip takes over unchanged: a GPU node claims the request,
copies the video, poses it and stages the pose into the posed folder; the
server adopts it and re-runs every step after posing.

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
from typing import Callable, Dict, Iterable, List, Optional

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
    for q in (getattr(Paths, "TRIAGE_REVIEW", None), getattr(Paths, "DEEP_REVIEW", None)):
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
        human = [f.name for f in _files(bundle, video_id)
                 if f.name.endswith(HUMAN_SUFFIXES)]
        durable = _durable_review(video_id)
        if human or durable is not None:
            return refuse("held for a person who has saved work on it ("
                          + ", ".join(human + ([durable.name + " (durable)"]
                                               if durable else [])) + ")")
        if parked.exists():
            return refuse(f"{parked} already exists (an earlier re-pose?); "
                          f"nothing is overwritten")
    elif results is not None and _files(results, video_id):
        rep["where"] = "filed"
    else:
        return refuse("no results folder and no review bundle found")

    # ---- 1. move the distrusted pose (and its outputs) out of every place ----
    from mousereach.archive.supersede import supersede_video_outputs
    # WHY the results folder even for a held video: a video re-run while filed
    # can be held AND have a pose in Analyzed; any declared-scorer pose left
    # there would be adopted as "the new pose" without a GPU.
    places = [p for p in (bundle, results) if p is not None]
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
        if bundle is None:
            mp4 = results / f"{video_id}.mp4"
            if not mp4.is_file() and not dry_run:
                rep["status"], rep["why"] = "incomplete", f"no video file in {results}"
                return rep
        else:
            # What is left of the bundle (the video, the queue's notes, any
            # subfolder) moves as ONE folder rename -- nothing is picked apart
            # and nothing can be left behind in the queue.
            mp4 = parked / f"{video_id}.mp4"
            for f in _files(bundle, video_id):
                if dry_run and (_is_pose(f, video_id) or any(
                        f.name.endswith(s) for s in _ALGO)):
                    continue                        # would have gone to the archive
                rep["parked"].append(f"{f} -> {parked / f.name}")
            if not dry_run:
                parked.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(bundle), str(parked))
        if posed is not None:
            for f in _files(posed, video_id):
                if f.suffix.lower() == ".mp4":
                    rep["parked"].append(
                        f"{f} -> {_move(f, parked / 'posed_copy', dry_run)}")
        if bundle is not None and not dry_run and not mp4.is_file():
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="mousereach-repose-videos",
        description="Send named videos back to a GPU node for a new pose, because "
                    "their current pose is not trusted. Moves the old pose and every "
                    "output made from it to Analyzed/Archive/<label>/, then marks the "
                    "video for a new pose and writes the re-pose request. Run on the "
                    "processing server. Use --dry-run first.")
    ap.add_argument("--ids", required=True,
                    help="text file with one video id per line")
    ap.add_argument("--label", required=True,
                    help="name of the archive folder the old outputs go to "
                         "(Analyzed/Archive/<label>/), e.g. 'CPU pose <dates>'")
    ap.add_argument("--reason", required=True,
                    help="why, recorded on each video (required)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would happen; change nothing")
    ap.add_argument("--limit", type=int, default=0,
                    help="act on at most this many videos (0 = all)")
    ap.add_argument("--report", help="also write the per-video report here (JSON)")
    args = ap.parse_args(argv)

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
    from mousereach.watcher.db import WatcherDB
    from mousereach.watcher.db_location import resolve_watcher_db_path
    db_path = resolve_watcher_db_path()
    print(f"[watcher db] {db_path}")
    db = WatcherDB(db_path)
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
