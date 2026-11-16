"""What needs a person, as the watchers list it.

Each watcher writes its own list into <NAS root>/attention/ (one JSON file per
machine and kind, e.g. <host>_hand_scores.json, see hand_scores.py), with
items of this shape:

    {"kind": "...", "needs_person": true/false, "video_id": "...",
     "cohort_id": "...", "says": "<one sentence a person can act on>"}

"needs_person" true is something a person must fix (e.g. a hand-score check
that cannot run); false is something that is only waiting, shown, never as
an alarm (e.g. an analysed video held before triage and kinematics until its
cohort's sheet is next imported). The lists are rewritten whenever
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
