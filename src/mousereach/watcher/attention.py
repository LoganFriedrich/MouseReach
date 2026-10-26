"""What needs a person, as the watchers list it.

Each watcher writes its own list into <NAS root>/attention/ (one JSON file per
machine and kind, e.g. <host>_hand_scores.json, see hand_scores.py), with
items of this shape:

    {"kind": "...", "needs_person": true/false, "video_id": "...",
     "cohort_id": "...", "says": "<one sentence a person can act on>"}

"needs_person" true is something a person must fix; false is something that
is only waiting (shown, never as an alarm). The lists are rewritten whenever
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
