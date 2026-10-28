"""Before a video is filed: its pellets against the session's hand scores.

WHY
---
Filing a video means it is done: what is filed is what the analyses use.
Where a lab scores every tray by hand at the bench, the hand score is a
second, independent reading of each pellet. A pellet the pipeline and the
bench disagree about is not done -- a person has to look at it first. So the
watcher checks every video against its hand scores at the last step before
filing:

    every pellet agrees, or a person answered it   -> filed
    a pellet the person has not answered disagrees -> TRIAGE, the pellet
        flagged "bench disagreement: the hand score says retrieved; the
        algorithm says displaced_sa" -- the wording the review tool already
        knows from the nightly bench check, so it shows "hand score mismatch
        (hand: retrieved, algo: displaced_sa)" and opens on the algorithm's
        answer (is_bench_mismatch, plain_flag_reason)
    no hand scores for the session:
        project not in required_projects           -> filed (nothing to compare)
        project in required_projects               -> WAITS, unfiled, until the
            cohort's sheet has been imported after the video was ready (the
            scores are entered the same day; the import may run only
            nightly). If that import did not bring them, the session was
            never scored by hand: there is nothing to compare, so nothing can
            disagree, and the video is filed.

What a person answered is never compared: a person's answer is the truth
for that pellet, whatever the bench says. Easy and Flat trays have no pellet
outcomes and are never checked.

WHERE THE SCORES COME FROM
--------------------------
A CSV file another tool writes (in this lab's setup, the database tool),
one row per hand-scored pellet with columns video_key (the session as this
pipeline names its video: 20250709_ABC0207_P1), pellet (1-20) and score
(0 = not touched, 1 = displaced, 2 = retrieved), and beside it a JSON file
of the same name holding:
    "last_successful_import": {cohort id: when its last sheet import ended}
    "sheets_seen":            [the cohort ids that have a sheet]
    "full_import_at":         when the last import of every sheet ended
Times are ISO 8601 (2026-10-26T20:05:00), in the same clock as the watcher's
own (local time, no zone); one that cannot be read, or a zone mixed with
none, counts as not known. A cohort with an import is judged by its own
import time. A cohort with
none either has a sheet (its first import is still to come: wait) or has
no sheet (settled by the next import of every sheet). Keys the file does
not have count as not known, and the video waits. The pipeline never imports
that tool; it only reads the file.

CONFIGURATION (read on every check, so an edit applies to a running watcher):

    "hand_scores": {
        "path": "<the CSV file>",
        "required_projects": ["PROJECT_A"]
    }

in this user's ~/.mousereach/config.json, or -- the usual place, so every
person on every PC gets it -- in the lab's shared settings file on the
storage drive, <nas_root>/mousereach_lab_config.json (config.lab_setting).
A "hand_scores" section in the user's own file replaces the lab file's.
Without "path" nothing is checked and videos are filed as before.

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# What happened to the PELLET: retrieved, displaced or untouched. ("Missed" is a
# word for a reach that did not interact with the pellet, never for a pellet.)
HAND_WORDS = {0: "untouched", 1: "displaced", 2: "retrieved"}
ALGO_CLASS = {"untouched": 0, "displaced_sa": 1, "displaced_outside": 1, "retrieved": 2}
FLAG_REASON = "bench disagreement"

AGREES = "agrees"
DISAGREES = "disagrees"
WAITING = "waiting"
NOT_CHECKED = "not_checked"


@dataclass
class Verdict:
    status: str
    reason: str
    disagreements: Dict[int, str] = field(default_factory=dict)   # segment -> words
    cohort_id: Optional[str] = None
    last_import: Optional[str] = None     # the cohort's last successful sheet import
    has_sheet: Optional[bool] = None      # the cohort has a sheet (None: not known)
    full_import_at: Optional[str] = None  # the last import of every sheet ended


def settings() -> dict:
    """The "hand_scores" settings ({} if none): this user's own config.json if
    it has the section, else the lab's shared settings file on the storage
    drive (config.lab_setting). Every reader -- the check before filing, the
    scores-file stamp, the review window's check-your-answer box -- comes
    through here, so they all read the same file for the same person."""
    from mousereach.config import lab_setting
    try:
        value = lab_setting("hand_scores")
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


_CACHE = {"key": None, "scores": None, "manifest": None}


def load(path: Path):
    """({video_key: {pellet: score}}, manifest dict) for the scores file, read
    again only when the file changes. Raises when it cannot be read."""
    path = Path(path)
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime)
    if _CACHE["key"] != key:
        scores: Dict[str, Dict[int, int]] = {}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scores.setdefault(row["video_key"], {})[int(row["pellet"])] = int(row["score"])
        manifest = {}
        mp = path.with_suffix(".json")
        if mp.is_file():
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        _CACHE.update(key=key, scores=scores, manifest=manifest)
    return _CACHE["scores"], _CACHE["manifest"]


def scores_file_stamp() -> Optional[float]:
    """Modification time of the configured scores file, or None."""
    p = settings().get("path")
    try:
        return Path(p).stat().st_mtime if p else None
    except OSError:
        return None


def pellet_hand_score(video_id: str, pellet: int):
    """(score, "") for one pellet's hand score (0/1/2, see HAND_WORDS), or
    (None, why not) when there is none to compare with. Never raises.

    Used by the review window's speed bump (review/double_disagreement.py),
    which asks a person to look again before saving an answer that disagrees
    with BOTH the algorithm and this score. The pellet is keyed exactly as
    ``check`` keys it (the segment number), so the review window and the
    before-filing check always read the same score for the same pellet. A
    missing file or row is not an error here: the speed bump simply does not
    run for that answer."""
    try:
        from mousereach.config import is_outcome_free_tray
        if is_outcome_free_tray(video_id):
            return None, "Easy/Flat tray: no pellet outcomes"
        cfg = settings()
        if not cfg.get("path"):
            return None, "no hand-scores file configured"
        scores, _manifest = load(Path(cfg["path"]))
    except Exception as e:
        return None, "hand-scores file could not be read (%s)" % e
    hand = scores.get(video_id)
    if hand is None:
        return None, "no hand scores for this session"
    s = hand.get(int(pellet))
    if s not in HAND_WORDS:
        return None, "no hand score for pellet %s" % pellet
    return s, ""


def check(video_id: str, folder: Path) -> Verdict:
    """Compare a processed video in ``folder`` with its hand scores. Reads
    only; flags and moves nothing (see the module docstring for what the
    caller does with each verdict). Raises when a file it needs exists but
    cannot be read -- a guess could file a video unchecked."""
    from mousereach.config import is_outcome_free_tray
    from mousereach.sync.database import parse_subject_id

    if is_outcome_free_tray(video_id):
        return Verdict(NOT_CHECKED, "Easy/Flat tray: no pellet outcomes")
    cfg = settings()
    if not cfg.get("path"):
        return Verdict(NOT_CHECKED, "no hand-scores file configured")
    subject = parse_subject_id(video_id) or ""
    project = subject.split("_")[0]
    cohort_id = "_".join(subject.split("_")[:2]) or None
    scores, manifest = load(Path(cfg["path"]))
    last = (manifest.get("last_successful_import") or {}).get(cohort_id)
    hand = scores.get(video_id)
    if hand is None:
        if project and project in (cfg.get("required_projects") or []):
            return Verdict(WAITING, "no hand scores for this session", cohort_id=cohort_id,
                           last_import=last, full_import_at=manifest.get("full_import_at"),
                           has_sheet=(cohort_id in manifest["sheets_seen"]
                                      if manifest.get("sheets_seen") is not None else None))
        return Verdict(NOT_CHECKED, "no hand scores for this session (not required "
                       "for %s)" % (project or "this project"), cohort_id=cohort_id)

    from mousereach.watcher.route_cli import answered_segments_of
    answered = answered_segments_of(video_id, folder)
    if "all" in answered:
        return Verdict(AGREES, "every pellet answered by a person", cohort_id=cohort_id)
    out = json.loads((Path(folder) / f"{video_id}_pellet_outcomes.json").read_text(encoding="utf-8"))
    disagree = {}
    for seg in out.get("segments", []):
        n = seg.get("segment_num")
        if n is None or n in answered:
            continue
        a, h = ALGO_CLASS.get(seg.get("outcome")), hand.get(int(n))
        if a is None or h is None or a == h:
            continue
        disagree[int(n)] = "the hand score says %s; the algorithm says %s" % (
            HAND_WORDS[h], seg.get("outcome"))
    if disagree:
        return Verdict(DISAGREES, "%d pellet(s) disagree with the hand score: %s" % (
            len(disagree), ", ".join(str(s) for s in sorted(disagree))), disagree,
            cohort_id=cohort_id)
    return Verdict(AGREES, "every pellet agrees with the hand score", cohort_id=cohort_id)


def _after(a: Optional[str], b: Optional[str]) -> Optional[bool]:
    """Whether time ``a`` is after time ``b``, read as times, not compared as
    text: two tools write these, and text order is time order only while both
    write the same shape. None when either cannot be read or one has a zone
    and the other not -- the caller then treats it as not known (wait)."""
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(a)) > datetime.fromisoformat(str(b))
    except (TypeError, ValueError):
        return None


def scores_never_came(verdict: Verdict, ready_since: Optional[str]) -> bool:
    """True once the cohort's sheet has been imported successfully after the
    video was ready and the session still has no hand scores: it was never
    scored by hand, so there is nothing to compare and the video is filed.
    Before that import it only waits. WHY file rather than hold: a record
    that does not exist cannot disagree with the video, and the video is the
    only record of that session (Logan, 2026-10-26)."""
    if verdict.status != WAITING or not ready_since:
        return False
    if verdict.last_import:
        return _after(verdict.last_import, ready_since) is True
    # No import of this cohort at all. With a sheet, its first import is still
    # to come: wait. With no sheet (or a cohort name no sheet has), nothing will
    # come -- an import of EVERY sheet after the video was ready settles it.
    # Not the file's own write time: a hand import of one other cohort moves
    # that too, and would file this video before its sheet's first import.
    if verdict.has_sheet is not False:
        return False
    return bool(verdict.full_import_at) and _after(verdict.full_import_at, ready_since) is True


def write_attention(nas_root: Path, hostname: str, waits: Dict[str, dict]) -> Optional[Path]:
    """Write this machine's list of videos waiting for hand scores to
    <nas root>/attention/<host>_hand_scores.json (replaced each time; empty
    list when none wait), for the panels that show what needs a person.
    Never raises."""
    try:
        d = Path(nas_root) / "attention"
        d.mkdir(parents=True, exist_ok=True)
        items = []
        for vid, w in sorted(waits.items()):
            if w.get("error"):
                # The check could not run, so the video is held (never filed
                # unchecked) -- which is silent unless a person sees it: a
                # wrong "path" in a machine's config would stop its filing.
                items.append({
                    "kind": "hand_score_check_failed", "needs_person": True,
                    "video_id": vid, "cohort_id": w.get("cohort_id"),
                    "says": ("%s is processed but not filed: the check against its hand "
                             "scores could not run (%s). Fix the \"hand_scores\" setting "
                             "(in the lab's mousereach_lab_config.json on the storage "
                             "drive, or in this PC's config.json if it has its own), or "
                             "the file it names; the video is filed by itself once the "
                             "check runs." % (vid, w["error"])),
                })
                continue
            items.append({
                "kind": "hand_scores_waiting",
                "needs_person": False,
                "video_id": vid,
                "cohort_id": w.get("cohort_id"),
                "ready_since": w.get("ready_since"),
                "last_import": w.get("last_import"),
                "says": ("%s is processed and waits for the %s sheet's next import, to be "
                         "compared with its hand scores (filed without the comparison if "
                         "that import has none)." % (vid, w.get("cohort_id"))),
            })
        tmp = d / f".{hostname}_hand_scores.json.tmp"
        tmp.write_text(json.dumps({"written_by": hostname, "items": items}, indent=1),
                       encoding="utf-8")
        dest = d / f"{hostname}_hand_scores.json"
        tmp.replace(dest)
        return dest
    except Exception as e:
        logger.warning("Could not write the hand-scores attention list: %s", e)
        return None
