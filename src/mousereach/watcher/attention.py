"""What needs a person, as the watchers list it.

Each watcher writes its own list into <NAS root>/attention/ (one JSON file per
machine and kind, e.g. <host>_hand_scores.json, see hand_scores.py), with
items of this shape:

    {"kind": "...", "needs_person": true/false, "video_id": "...",
     "cohort_id": "...", "says": "<one sentence a person can act on>"}

"needs_person" true is something a person must fix (e.g. a hand-score check
that cannot run); false is something that is only waiting, shown, never as
an alarm (e.g. an analysed video held before triage and kinematics until its
cohort's sheet is next imported). A GPU node's lists about COLLAGES
(<host>_collages.json, <host>_collage_claims.json) name the recording in
"collage" and carry cohort_id None, because a collage holds several animals.
The lists are rewritten whenever
they change and empty when nothing is listed, so reading them all is the
current picture. Read by the Pipeline Dashboard's red panel, and by any other
tool that shows the lab what needs doing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List


def read_items(nas_root) -> List[dict]:
    """Every item of every list under <nas_root>/attention/. A list that
    cannot be read becomes one item saying so (needs_person), never silence."""
    d = Path(nas_root) / "attention"
    out: List[dict] = []
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for item in data.get("items") or []:
                out.append(dict(item, listed_in=f.name))
        except Exception as e:
            out.append({"kind": "unreadable_list", "needs_person": True, "listed_in": f.name,
                        "says": "The list %s could not be read (%s)." % (f.name, e)})
    return out


def write_failed_reruns(nas_root, hostname: str, rows) -> None:
    """Write this machine's list of filed videos whose re-run failed for good
    to <nas_root>/attention/<host>_failed_reruns.json (replaced each time;
    an empty list when there are none). ``rows`` are watcher rows (video_id,
    error_message). needs_person: the filed results stand, but what the
    re-run was for (a newer algorithm, a person's answers) has not reached
    them, and whatever stopped it may need a person. Never raises."""
    import logging
    try:
        from mousereach.pipeline.fsutil import write_json_atomic
        # Created when missing, as hand_scores.write_attention does. WHY not
        # skip: a missing folder would leave a person no list of these videos
        # and no word that it is missing.
        d = Path(nas_root) / "attention"
        d.mkdir(parents=True, exist_ok=True)
        items = [{
            "kind": "filed_rerun_failed",
            "needs_person": True,
            "video_id": r.get("video_id"),
            "says": ("%s: %s. Its filed results are unchanged and it stays filed; "
                     "the version scan offers the re-run again once the declared "
                     "versions change or a review or ground truth is saved for it, "
                     "or press Re-run for it. The watcher "
                     "log and processing log say what failed."
                     % (r.get("video_id"), r.get("error_message") or "re-run failed")),
        } for r in sorted(rows, key=lambda r: r.get("video_id") or "")]
        write_json_atomic(d / f"{hostname}_failed_reruns.json",
                          {"written_by": hostname, "items": items}, indent=1)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Could not write the failed re-runs attention list: %s", e)


def write_list(nas_root, hostname: str, kind: str, items) -> bool:
    """Replace this machine's list ``<nas_root>/attention/<host>_<kind>.json``
    with ``items`` (an empty list when nothing is listed). Written to a
    temporary file and swapped in, so a reader never sees half a list.
    True when written. Never raises: it is called from the watcher's scan,
    and a list that cannot be written must not stop the work it describes
    (the failure is logged at WARNING instead).

    Every item a watcher lists about a COLLAGE (kinds collage_crop_parked and
    collage_claim_stuck) carries "collage" and a cohort_id of None. WHY None:
    a collage holds several animals, and a tool that copies items into a
    cohort's own to-do list must not file a recording-level problem under
    one animal's cohort."""
    import logging
    try:
        from mousereach.pipeline.fsutil import write_json_atomic
        d = Path(nas_root) / "attention"
        d.mkdir(parents=True, exist_ok=True)
        write_json_atomic(d / f"{hostname}_{kind}.json",
                          {"written_by": hostname, "items": list(items)}, indent=1)
        return True
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Could not write the attention list %s_%s.json: %s", hostname, kind, e)
        return False


def note_video(nas_root, hostname: str, kind: str, video_id: str, says: str) -> bool:
    """List ``video_id`` in this machine's ``<host>_<kind>.json`` (one item per
    video; a newer note replaces the older) with ``needs_person`` true. The
    read, change and write happen under the list's lock file, so two programs
    on one machine never overwrite each other's change. True when written.
    Never raises (logged at WARNING instead), for the reason write_list gives.

    WHY one item per VIDEO, not per folder: the lists keyed by folder
    (review_routing.note_for_a_person) suit queue folders, which belong to one
    video; a filed cohort folder holds many videos."""
    item = {"kind": kind, "needs_person": True, "video_id": video_id, "says": says}
    return _change_video_list(
        nas_root, hostname, kind,
        lambda items: [i for i in items if i.get("video_id") != video_id] + [item])


def drop_video(nas_root, hostname: str, kind: str, video_id: str) -> bool:
    """Take ``video_id`` off this machine's ``<host>_<kind>.json``. Nothing is
    written (and no folder made) when the list does not name it. Never raises."""
    return _change_video_list(
        nas_root, hostname, kind,
        lambda items: [i for i in items if i.get("video_id") != video_id])


def _change_video_list(nas_root, hostname, kind, change) -> bool:
    import logging
    if not nas_root:
        return False
    path = Path(nas_root) / "attention" / f"{hostname}_{kind}.json"
    try:
        try:
            before = list((json.loads(path.read_text(encoding="utf-8")) or {}).get("items") or [])
        except FileNotFoundError:
            before = []
        if change(list(before)) == before:
            return False
        from mousereach.review.causal_review_io import _locked_update

        def _apply(doc):
            doc["items"] = change(list(doc.get("items") or []))
            doc["written_by"] = hostname
        _locked_update(path, _apply, wait_s=10.0)
        return True
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Could not update the attention list %s: %s", path.name, e)
        return False


def without_repeats(items) -> List[dict]:
    """``items`` with repeats of the same collage problem dropped, keeping the
    first. WHY: every GPU node that asks about a stuck crop claim lists the
    same collage, and a person should see it once. Items that name no
    collage are all kept."""
    seen = set()
    out: List[dict] = []
    for i in items:
        key = (i.get("kind"), i.get("collage")) if i.get("collage") else None
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        out.append(i)
    return out
