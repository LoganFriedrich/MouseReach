"""mousereach-label-mismatch: list the videos held as a LABEL MISMATCH, and
record a person's decision about one.

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


def _mismatch_from_item(item):
    """The LabelMismatch an attention item describes (for the record), or None."""
    from mousereach.watcher.hand_scores import LabelMismatch, RowMatch
    own = item.get("own_row") or {}
    rows = item.get("matched_rows") or []
    if not own or not rows:
        return None
    return LabelMismatch(item.get("video_id"),
                         RowMatch(own.get("key"), own.get("agree"), own.get("compared")),
                         [RowMatch(r.get("key"), r.get("agree"), r.get("compared"))
                          for r in rows])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="mousereach-label-mismatch",
        description="List videos held as a label mismatch, or record which is wrong: "
                    "the recording's name or the sheet.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="videos held as a label mismatch now")
    g.add_argument("--decisions", action="store_true", help="every decision on record")
    g.add_argument("--sheet-error", metavar="VIDEO",
                   help="the sheet is a human error: keep it, file the video, record it")
    g.add_argument("--name-wrong", metavar="VIDEO",
                   help="the recording's name is wrong (give the right one with --as)")
    g.add_argument("--labels-right", metavar="VIDEO",
                   help="neither is wrong: compare the video with its own row as usual")
    ap.add_argument("--as", dest="new_name", metavar="RIGHT_NAME")
    ap.add_argument("--by", metavar="NAME", help="who decides (kept with the decision)")
    ap.add_argument("--note", default="", help="anything else to keep with the decision")
    a = ap.parse_args(argv)
    from mousereach.watcher import hand_scores as hs

    if a.list:
        items = _held_items()
        if not items:
            print("No video is held as a label mismatch.")
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
    held = {i.get("video_id"): i for i in _held_items()}
    if video not in held:
        print("[FAIL] %s is not held as a label mismatch (mousereach-label-mismatch --list "
              "shows which are). Nothing recorded." % video, file=sys.stderr)
        return 1
    try:
        entry = hs.record_decision(
            video, (hs.DECISION_SHEET_ERROR if a.sheet_error else
                    hs.DECISION_NAME_WRONG if a.name_wrong else hs.DECISION_LABELS_RIGHT),
            a.by, mismatch=_mismatch_from_item(held[video]), new_name=a.new_name,
            note=a.note)
    except Exception as e:
        print("[FAIL] %s. Nothing recorded." % e, file=sys.stderr)
        return 1
    if a.sheet_error:
        print("[OK] Recorded: the sheet is wrong for %s (by %s, %s). The sheet stays as it "
              "is; the watcher files the video with the disagreement recorded on its next "
              "pass." % (video, entry["by"], entry["at"]))
    elif a.labels_right:
        print("[OK] Recorded: the labels of %s are right (by %s, %s). The watcher compares "
              "it with its own row on its next pass; a real disagreement goes to triage."
              % (video, entry["by"], entry["at"]))
    else:
        print("[OK] Recorded: %s should be named %s (by %s, %s). It stays held until the "
              "recording is renamed the documented way (originals and a record in "
              "Archive/manual_corrections); then it checks itself again."
              % (video, a.new_name, entry["by"], entry["at"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
