"""mousereach-label-mismatch: the record of the RETIRED label-mismatch hold.

RETIRED 2026-11-20 (owner rule: "Just trust what is written down")
-------------------------------------------------------------------
A video is always processed and filed under the name a person gave it. The
watcher no longer holds a video whose results match another tray's hand
scores better than its own: it is compared with its OWN row only, and a
disagreement with that row goes to triage like any other. So no video is
held as a label mismatch any more, and no new decision is taken:

  --list        still works: says nothing is held (and lists any hold an
                attention list still carries from before the retirement)
  --decisions   still works: prints every decision on record (the history)
  --sheet-error / --name-wrong / --labels-right
                refused with a message: there is nothing left to decide.

A "the sheet is wrong" decision already on record still stands: that video is
filed with the disagreement recorded, never sent to triage over it
(hand_scores.check). The record file is kept for good.

What follows describes the hold as it was, for reading the record.

WHAT A LABEL MISMATCH IS
------------------------
A video whose results match the hand scores of ANOTHER tray of the same animal
and day clearly better than its own tray's (watcher/hand_scores.label_mismatch
has the rule and its numbers). Either the recording was named with the wrong
tray, or the sheet has the trays' rows the other way round. No review answer
can settle that, so the watcher holds the video outside every review queue
until a person decides which it is.

THE TWO DECISIONS
-----------------
  --sheet-error VIDEO --by NAME
      "The sheet is a human error; keep it as it is." The sheet is never
      edited: the lab keeps sheet errors as they are, because how often the
      bench gets a tray wrong is itself something to measure. The video goes
      on through every other check and is filed with the disagreement and this
      decision recorded (in the video's outcomes file, under "hand_scores").

  --name-wrong VIDEO --as RIGHT_NAME --by NAME
      "The file name is wrong." The decision is recorded and the video stays
      held until the recording is renamed the documented way (an evident-typo
      correction, the original and a record kept in
      Archive/manual_corrections). This command does not rename anything: the
      rename touches the recording and every video cut from it, and is done
      by whoever maintains the pipeline: this command only RECORDS the
      decision. The maintainer puts the renamed results in place of the held
      ones in its processing folder (the video then checks itself again), or,
      when the recording is re-cut under the right name, retires the held
      video's row; files moved away without either leave the row listed in
      red as "the check could not run".

  --labels-right VIDEO --by NAME
      "Neither is wrong": another tray's row happened to fit. The video is
      compared with its own row as usual, and a real disagreement goes to
      triage.

Every decision is appended to <storage drive>/review_records/
label_decisions.jsonl -- who, when, which video, which two sheet rows and how
well each agreed -- and never removed, so the human error rate can be counted
from it (--decisions prints it). The watcher reads it on its next pass; there
is nothing to restart.

  --list        the videos held as a label mismatch now (every watcher's list)
  --decisions   every decision on record

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import argparse
import json
import sys


def _held_items():
    from mousereach.config import Paths
    from mousereach.watcher.attention import read_items
    nas = getattr(Paths, "NAS_ROOT", None)
    return [i for i in (read_items(nas) if nas else []) if i.get("kind") == "label_mismatch"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="mousereach-label-mismatch",
        description="The label-mismatch hold is RETIRED (2026-11-20): videos are filed "
                    "as named. --list and --decisions read what was recorded.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="videos held as a label mismatch now")
    g.add_argument("--decisions", action="store_true", help="every decision on record")
    g.add_argument("--sheet-error", metavar="VIDEO",
                   help="RETIRED 2026-11-20: refused, nothing to decide")
    g.add_argument("--name-wrong", metavar="VIDEO",
                   help="RETIRED 2026-11-20: refused, nothing to decide")
    g.add_argument("--labels-right", metavar="VIDEO",
                   help="RETIRED 2026-11-20: refused, nothing to decide")
    ap.add_argument("--as", dest="new_name", metavar="RIGHT_NAME")
    ap.add_argument("--by", metavar="NAME", help="who decides (kept with the decision)")
    ap.add_argument("--note", default="", help="anything else to keep with the decision")
    a = ap.parse_args(argv)
    from mousereach.watcher import hand_scores as hs

    if a.list:
        items = _held_items()
        if not items:
            print("No video is held as a label mismatch (the hold was retired on "
                  "2026-11-20: videos are filed as named).")
        for i in items:
            print("%s (cohort %s): %s" % (i.get("video_id"), i.get("cohort_id"), i.get("says")))
        return 0
    if a.decisions:
        p = hs.decisions_path()
        if p is None or not p.is_file():
            print("No decision on record.")
            return 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                print("%s  %-12s %s by %s%s" % (e.get("at"), e.get("decision"), e.get("video_id"),
                                                e.get("by"), (" -> %s" % e["new_name"])
                                                if e.get("new_name") else ""))
        return 0

    video = a.sheet_error or a.name_wrong or a.labels_right
    # RETIRED 2026-11-20: no hold, so nothing to decide. Refused before
    # anything is written, so the record keeps only real decisions.
    print("[FAIL] The label-mismatch hold was retired on 2026-11-20 (owner rule: trust "
          "the name a person gave). %s is processed and filed under its own name; no "
          "decision is needed. Nothing recorded." % video, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
