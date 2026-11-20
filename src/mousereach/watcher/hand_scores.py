"""After analysis, before triage: a video's pellets against the session's hand scores.

WHY
---
Filing a video means it is done: what is filed is what the analyses use.
Where a lab scores every tray by hand at the bench, the hand score is a
second, independent reading of each pellet. A pellet the pipeline and the
bench disagree about is not done -- a person has to look at it first. So the
watcher compares every analysed video with its hand scores right after the
analysis and the quality check, BEFORE the gate decides whether the video
goes to triage, deep review or on to kinematics and filing:

    every pellet agrees, or a person answered it   -> on to the gate as usual
    a pellet the person has not answered disagrees -> the pellet is flagged
        "bench disagreement: the hand score says retrieved; the algorithm
        says displaced_sa", and the gate sends the video to TRIAGE once, with
        that pellet and any other open question together. The wording is the
        one the review tool already knows from the nightly bench check, so it
        shows "hand score mismatch (hand: retrieved, algo: displaced_sa)" and
        opens on the algorithm's answer (is_bench_mismatch,
        plain_flag_reason). A reason the pipeline had already flagged that
        pellet for is kept after it: "... (also: <the old reason>)".
    no hand scores for the session:
        project not in required_projects           -> on to the gate
        project in required_projects               -> HELD, in Processing,
            before triage and kinematics, until its scores are there. Until
            the cohort's sheet has been imported after the video entered
            Processing (the scores are entered the same day; the import may
            run only nightly) it only WAITS. If that import did not bring
            them, the scores are MISSING -- in a required project every
            session is scored by hand before its videos enter the pipeline,
            so the sheet row is absent or misnamed, or the import did not
            match it to this video. That is a defect for a person to fix: the
            video stays held, listed in red, and goes on by itself once its
            scores arrive. It is never filed without the comparison.
            (Before, such a video went on as "never scored by hand" -- which
            could not be told from a row the import failed to match -- and
            was filed uncompared. Those filed videos are compared once their
            scores arrive, and each disagreement is listed for a person:
            watcher/filed_hand_scores.py.)

WHY before triage and not before filing (moved 2026-10-28; from 2026-10-26
it was the last step before filing): checked at filing, a video with a
triage question went to the queue, was answered, came back, and only then
met the hand score -- and a disagreement sent it to the queue a second time.
Checked before the gate, a person sees every question about a video in one
visit, and a video that disagrees never gets kinematics first.

What a person answered is never compared: a person's answer is the truth
for that pellet, whatever the bench says.

Segment N is compared with hand pellet N only while nothing says the cuts
number the pellets wrongly. When a review record says a segment is really
another pellet (true_segment_num) and the current cuts, matched by frames,
still number it otherwise, the numbering of every segment is in doubt: the
check raises PelletNumbersUnknown instead of comparing a pellet with another
pellet's score, and the caller sends the video to manual segmentation (the
re-segmentation queue): the problem is in the cuts, so that is the step a
person has to fix. A person's re-cut saved after the mislabel record settles
it -- the newest human statement about the cuts wins. Easy and Flat trays have no pellet
outcomes and are never checked.

EVERY PELLET'S SCORE IS KEPT WITH THE VIDEO
-------------------------------------------
Whenever the sheet was read for a video, ``record`` writes what it said into
the video's <video>_pellet_outcomes.json under the top-level key
"hand_scores":

    "hand_scores": {"pellets": {"1": 2, "2": 0, ...},   # segment -> 0/1/2
                    "verdict": "agrees" | "disagrees" | "not_required"
                               (| "never_scored": only in videos filed
                               before a missing score held the video),
                    "source_mtime": <the scores file's modification time>,
                    "cohort_last_import": <the cohort's last sheet import>,
                    "read_at": <when this check read it>}

"pellets" is empty when the session has no row in the sheet. Nothing is
written for Easy/Flat trays or when no scores file is configured. WHY: the
review window's check-your-answer box (review/double_disagreement.py) reads
the score from here (score_from_outcomes), so the reviewer sees exactly the
score the pipeline compared with, travelling inside the bundle, and the
review window needs no access to the scores file at all.

WHERE THE SCORES COME FROM
--------------------------
A CSV file another tool writes (for example a database tool),
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
Without "path" nothing is checked and videos go on to the gate as before.

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
# RETIRED 2026-11-20 (owner rule: "Just trust what is written down"):
# check() never returns this any more. It was the verdict for a video whose
# calls matched ANOTHER tray's row better than its own, which was then held
# outside the review queues. Kept so outcomes files and attention lists
# written before the retirement still read.
LABEL_MISMATCH = "label_mismatch"
# The lab suspended the comparison for this session (suspension): it is left
# out of the decision to triage -- no bench flags, no label-mismatch hold.
SUSPENDED = "suspended"
# A video a person decided has a SHEET error (label_decision, recorded while
# the label-mismatch hold existed): the sheet is kept as it is, the video goes
# on and is filed with the disagreement and the decision recorded, never
# flagged for triage over it. Still honoured after the hold's retirement.
SHEET_ERROR = "sheet_error"

# The two decisions a person can make about a label mismatch.
DECISION_SHEET_ERROR = "sheet_error"
DECISION_NAME_WRONG = "name_wrong"
# "Neither: the labels are right" -- the rule called a mismatch where there is
# none (another tray's row happened to fit). The video is then compared with
# its own row as usual, and a real disagreement goes to triage.
DECISION_LABELS_RIGHT = "labels_right"

# When another row "matches clearly better" (label_mismatch). All three must
# hold for a row other than the video's own:
#   * at least LABEL_MIN_PELLETS pellets compared with it (a pellet counts
#     when the algorithm called it and the row scored it);
#   * the algorithm agrees with it on at least LABEL_MIN_AGREEMENT of them;
#   * it beats the video's own row by at least LABEL_MIN_MARGIN pellets, counted
#     only where the two rows differ: (pellets the algorithm agrees with the
#     other row on and not the own row) minus (the reverse).
# WHY these numbers (the data behind them is in the lab's work record,
# 2026-11-18): when the label is right, the algorithm disagrees with its own
# row on no pellet in most videos and on at most two in nearly all, and one
# tray's row of an animal often looks like its other trays' (a mouse that
# retrieves nearly everything gives near-identical rows) -- so agreement with
# another row alone proves nothing, and the margin is counted only where the
# two rows DIFFER. A margin of 1 or 2 is within the algorithm's ordinary
# error on a correctly labeled video. At 3 the other tray's own video
# rarely matched this video's row back -- mostly not swaps. From 4 up most did, the pattern a swap
# leaves. Fewer than 10 compared pellets is too few to tell a swap from
# chance.
LABEL_MIN_PELLETS = 10
LABEL_MIN_AGREEMENT = 0.90
LABEL_MIN_MARGIN = 4


@dataclass
class RowMatch:
    """How a video's calls compare with one hand-score row (label_mismatch)."""
    key: str                 # the row's video_key, e.g. 20250709_ABC0207_P4
    agree: int               # pellets the algorithm agrees with this row on
    compared: int            # pellets compared (called AND scored in the row)
    margin: int = 0          # vs the own row, where the two rows differ (see above)


@dataclass
class LabelMismatch:
    """A video whose calls match another tray's hand scores, not its own."""
    video_id: str
    own: RowMatch                           # the video's own row
    matches: list                           # [RowMatch], best first; ties all listed
    # The reverse check, when the partner's own video is analyzed: how its calls
    # compare with THIS video's row. None when it was not found or not compared.
    partner_back: Optional[RowMatch] = None
    partner_video: Optional[str] = None


def algo_classes(outcomes_doc) -> Dict[int, int]:
    """{segment: 0/1/2} for every segment the algorithm called with an outcome
    a hand score can be compared with (ALGO_CLASS); a 'triaged' segment, or one
    with no outcome, is left out."""
    out = {}
    for seg in (outcomes_doc or {}).get("segments") or []:
        n, a = seg.get("segment_num"), ALGO_CLASS.get(seg.get("outcome"))
        if n is not None and a is not None:
            out[int(n)] = a
    return out


def suspension(video_id: str, cfg: Optional[dict] = None) -> Optional[dict]:
    """The lab's entry suspending the hand-score comparison for this session,
    or None. Read from the "suspended_sessions" list of the "hand_scores"
    settings (settings() says where; read on every check, so adding or
    removing an entry applies to a running watcher):

        "suspended_sessions": [
            {"video_key": "<date>_<animal>_<tray>",
             "reason": "<why, in plain words>",
             "set_by": "<who decided>", "set_at": "<when, ISO 8601>"}
        ]

    WHY: when the lab cannot trust a session's hand scores to belong to its
    video -- a recording whose tray labels disagree with the sheet, while
    nobody knows yet whether the file names or the sheet rows are the swapped
    ones -- no conclusion about a mismatch can be drawn, and asking a person
    to overrule the bench pellet by pellet is wrong. The comparison is then
    left out for exactly those sessions; every other reason to triage (an
    undecided pellet, a quality check) still applies. A list in the lab's
    settings, never in code: which sessions is a lab fact, and removing the
    entry restores the comparison (the triage recheck then compares again).
    An entry with no "video_key" is ignored; one with no reason still
    suspends, and says no reason was given."""
    if cfg is None:
        cfg = settings()
    for e in (cfg or {}).get("suspended_sessions") or []:
        if isinstance(e, dict) and str(e.get("video_key") or "") == str(video_id):
            return e
    return None


def suspension_words(entry: dict) -> str:
    """'the bench comparison is suspended for this session: <reason> (set by
    <who>, <when>)' for a suspension entry."""
    who = ", ".join(str(x) for x in (entry.get("set_by"), entry.get("set_at")) if x)
    return "the bench comparison is suspended for this session: %s%s" % (
        entry.get("reason") or "no reason given", " (set by %s)" % who if who else "")


def _same_animal_day(video_id: str, keys) -> list:
    """The hand-score rows of the same animal on the same day, other trays
    (every row whose key has the video's date and animal parts and another
    tray part), sorted."""
    parts = str(video_id).split("_")
    if len(parts) < 3:
        return []
    prefix = "%s_%s_" % (parts[0], parts[1])
    return sorted(k for k in keys if k != video_id and str(k).startswith(prefix)
                  and "_" not in str(k)[len(prefix):])


def label_mismatch(video_id: str, calls: Dict[int, int],
                   scores: Dict[str, Dict[int, int]]) -> Optional[LabelMismatch]:
    """The other tray(s) whose hand scores the calls match clearly better than
    the video's own row, or None. Pure: reads nothing.

    ``calls`` is algo_classes() of the video's outcomes; ``scores`` the whole
    scores file as load() reads it. Every pellet the algorithm called is
    compared, answered or not: the question is which tray the recording is,
    not whether one pellet's call is right.

    WHY this exists (owner ruling 2026-11-18): when the calls disagree with
    the video's own row on many pellets but match another tray of the same
    animal and day on nearly all, the file name or the sheet has the trays
    swapped. No review answer can fix that -- a person in triage would be
    asked to overrule the bench pellet by pellet, or to confirm a call that
    is right for the wrong tray -- so it must never go to triage. The video is
    held until the name or the sheet is corrected (orchestrator
    _hand_score_precheck).

    The thresholds and why: LABEL_MIN_PELLETS, LABEL_MIN_AGREEMENT,
    LABEL_MIN_MARGIN above. When several rows qualify, every row as good as
    the best (no pellet between them) is named: identical rows cannot be
    told apart, and it is still not this video's own row."""
    own_row = scores.get(video_id)
    if not own_row or not calls:
        return None
    own_n = [n for n in calls if n in own_row]
    own = RowMatch(video_id, sum(1 for n in own_n if calls[n] == own_row[n]), len(own_n))
    good = []
    for key in _same_animal_day(video_id, scores.keys()):
        row = scores.get(key) or {}
        both = [n for n in calls if n in row]
        agree = sum(1 for n in both if calls[n] == row[n])
        if len(both) < LABEL_MIN_PELLETS or agree < LABEL_MIN_AGREEMENT * len(both):
            continue
        differ = [n for n in both if n in own_row and row[n] != own_row[n]]
        margin = (sum(1 for n in differ if calls[n] == row[n])
                  - sum(1 for n in differ if calls[n] == own_row[n]))
        if margin >= LABEL_MIN_MARGIN:
            good.append(RowMatch(key, agree, len(both), margin))
    if not good:
        return None
    good.sort(key=lambda m: (-(m.agree / m.compared), -m.margin, m.key))
    best = good[0]
    ties = [m for m in good if m.agree * best.compared == best.agree * m.compared]
    return LabelMismatch(video_id, own, ties)


def check_partner(mismatch: LabelMismatch, partner_calls: Optional[Dict[int, int]],
                  scores: Dict[str, Dict[int, int]]) -> LabelMismatch:
    """Fill in the reverse check: the best matching row's OWN video, when it
    is analyzed (``partner_calls``, algo_classes of its outcomes), compared
    with THIS video's row. A swap shows both ways. Evidence for the reason
    text only: a hold never waits for it, because the partner may not be
    analyzed yet. Returns ``mismatch``."""
    if not mismatch.matches or not partner_calls:
        return mismatch
    row = scores.get(mismatch.video_id) or {}
    both = [n for n in partner_calls if n in row]
    mismatch.partner_video = mismatch.matches[0].key
    mismatch.partner_back = RowMatch(mismatch.video_id,
                                     sum(1 for n in both if partner_calls[n] == row[n]),
                                     len(both))
    return mismatch


def partner_calls(partner_id: str, folder=None) -> Optional[Dict[int, int]]:
    """algo_classes of the partner video's outcomes, wherever its live copy is:
    the same working folder, a review queue bundle (deep review first, it
    supersedes triage), or its archive folder (computed from its id, no tree
    walk -- locate.archive_dir_for). None when it is not analyzed or cannot be
    read. Never raises: the reverse check is evidence, never a condition."""
    try:
        from mousereach.config import Paths
        from mousereach.watcher.locate import archive_dir_for
        places = [Path(folder)] if folder else []
        for q in (getattr(Paths, "DEEP_REVIEW", None), getattr(Paths, "TRIAGE_REVIEW", None)):
            if q:
                places.append(Path(q) / partner_id)
        places.append(archive_dir_for(partner_id))
        for d in places:
            if not d:
                continue
            p = Path(d) / f"{partner_id}_pellet_outcomes.json"
            if p.is_file():
                return algo_classes(json.loads(p.read_text(encoding="utf-8")))
    except Exception as e:
        logger.debug("reverse label check for %s skipped: %s", partner_id, e)
    return None


def decisions_path() -> Optional[Path]:
    """The append-only record of every decision a person made about a label
    mismatch: <storage drive>/review_records/label_decisions.jsonl (beside
    the review answers, which are also a person's statements of record), or
    None when no storage drive is configured. One JSON object per line."""
    try:
        from mousereach.review.causal_review_io import _default_index_dir
        d = _default_index_dir()
    except Exception:
        d = None
    return Path(d) / "label_decisions.jsonl" if d else None


def record_decision(video_id: str, decision: str, by: str, mismatch: Optional["LabelMismatch"] = None,
                    new_name: Optional[str] = None, note: str = "") -> dict:
    """Append a person's decision about a label mismatch to the record
    (decisions_path) and return it. Raises when it cannot be written: a
    decision that is not on record does not count.

    ``decision`` is DECISION_SHEET_ERROR ("the sheet is a human error, keep it
    as it is": the video is filed with the disagreement recorded) or
    DECISION_NAME_WRONG ("the file name is wrong": the recording is renamed
    the documented way, ``new_name`` saying to what) or DECISION_LABELS_RIGHT
    ("neither: the labels are right", the rule was wrong here -- the video is
    compared with its own row as usual).

    WHY a record of its own, kept for good: the lab's rule is that the sheets
    are what they are -- a sheet error is never corrected, because how often
    the bench gets a tray wrong is itself something to measure. Each line
    says who decided, when, about which video and which two sheet rows (the
    video's own and the one its results match, with how well each agreed), so
    the human error rate can be counted from this file alone."""
    from datetime import datetime
    if decision not in (DECISION_SHEET_ERROR, DECISION_NAME_WRONG, DECISION_LABELS_RIGHT):
        raise ValueError("unknown decision %r" % decision)
    if not str(by or "").strip():
        raise ValueError("say who decided (--by NAME)")
    if decision == DECISION_NAME_WRONG and not new_name:
        raise ValueError("a wrong file name needs the right one (--as NEW_NAME)")
    entry = {"video_id": video_id, "decision": decision, "by": str(by).strip(),
             "at": datetime.now().isoformat(timespec="seconds")}
    if new_name:
        entry["new_name"] = new_name
    if note:
        entry["note"] = note
    if mismatch is not None:
        entry["own_row"] = {"key": mismatch.own.key, "agree": mismatch.own.agree,
                            "compared": mismatch.own.compared}
        entry["matched_rows"] = [{"key": m.key, "agree": m.agree, "compared": m.compared}
                                 for m in mismatch.matches]
    p = decisions_path()
    if p is None:
        raise RuntimeError("no storage drive configured: the decision cannot be recorded")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        import os
        os.fsync(f.fileno())
    return entry


def decision_for(video_id: str) -> Optional[dict]:
    """The latest decision recorded for this video (record_decision), or None.
    Raises when the record exists but cannot be read: a decision that cannot
    be read must not be taken as none (the video would be held for ever
    silently) -- the caller's check then fails in the open."""
    p = decisions_path()
    if p is None or not p.is_file():
        return None
    last = None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        if e.get("video_id") == video_id:
            last = e
    return last


def _tray(key: str) -> str:
    return str(key).rsplit("_", 1)[-1]


def describe_mismatch(m: LabelMismatch) -> str:
    """The plain-words reason for a held label mismatch: which row the results
    match and how well, which row they were checked against, the reverse check
    when there is one, and the two fixes."""
    parts = str(m.video_id).split("_")
    who = "%s %s" % (parts[1], parts[0]) if len(parts) >= 3 else m.video_id
    named = " and ".join("%s (%d/%d)" % (_tray(x.key), x.agree, x.compared) for x in m.matches)
    text = ("results match %s %s's hand scores, not %s's (%d/%d): the file name or the "
            "sheet has the trays swapped" % (who, named, _tray(m.video_id), m.own.agree,
                                             m.own.compared))
    if m.partner_back is not None:
        text += ("; the %s video matches %s's scores (%d/%d)" % (
            _tray(m.partner_video), _tray(m.video_id), m.partner_back.agree,
            m.partner_back.compared))
    text += (" -- a person decides which is wrong: the recording's NAME (it is renamed the "
             "documented way, original and a record kept in Archive/manual_corrections: "
             "mousereach-label-mismatch --name-wrong %s --as <right name> --by <name>) or "
             "the SHEET (a sheet error is kept as it is and recorded, and the video is "
             "filed with the disagreement: mousereach-label-mismatch --sheet-error %s "
             "--by <name>). No review answer can settle this, so it is not sent to triage"
             % (m.video_id, m.video_id))
    return text


@dataclass
class Verdict:
    status: str
    reason: str
    disagreements: Dict[int, str] = field(default_factory=dict)   # segment -> words
    cohort_id: Optional[str] = None
    last_import: Optional[str] = None     # the cohort's last successful sheet import
    has_sheet: Optional[bool] = None      # the cohort has a sheet (None: not known)
    full_import_at: Optional[str] = None  # the last import of every sheet ended
    # What the sheet says for this session, {segment: 0/1/2}: {} when the sheet
    # was read and has no row for it, None when it was not read at all (Easy/
    # Flat tray, no scores file configured). record() keeps it with the video.
    hand: Optional[Dict[int, int]] = None
    # LABEL_MISMATCH / SHEET_ERROR: which other row(s) the calls match.
    mismatch: Optional["LabelMismatch"] = None
    # A person's decision about the label mismatch (label_decision), or None.
    decision: Optional[dict] = None


def settings() -> dict:
    """The "hand_scores" settings ({} if none): this user's own config.json if
    it has the section, else the lab's shared settings file on the storage
    drive (config.lab_setting). Every reader -- the check, the scores-file
    stamp, the attention list -- comes through here, so they all read the
    same file for the same person. (The review window reads no setting: it
    reads the score the check kept in the outcomes file, score_from_outcomes.)

    {} means "no hand_scores section anywhere": nothing is checked. A
    settings file that exists but cannot be read, or a shared pipeline folder
    that cannot be reached, RAISES (config.lab_setting with strict=True), so
    check() raises and the watcher holds the video as a check that could not
    run -- listed in red for a person. WHY: this used to answer {} for those
    too, and every video then went on to filing as "no hand-scores file
    configured", never compared; a section that is merely unreadable is not
    a section that is absent."""
    from mousereach.config import lab_setting
    value = lab_setting("hand_scores", strict=True)
    return value if isinstance(value, dict) else {}


_CACHE = {"key": None, "scores": None, "manifest": None}

# How often, and how long apart, load() reads the pair again when the
# manifest's hash does not match the scores file (see load()).
PAIR_RETRIES = 3
PAIR_RETRY_WAIT_S = 1.0


class ScoresPairMismatch(Exception):
    """The scores file and its manifest do not belong together: the manifest
    names the SHA-256 of the scores file it was written with, and the scores
    file read has another. Raised by load() after a few short retries; the
    caller holds the video as a check that could not run."""


def _pair_key(path: Path):
    """What identifies the pair as read: both files' size and modification
    time in nanoseconds (None for a manifest that is not there). WHY
    nanoseconds: two writes of the same length within one float-rounded
    time would otherwise read as one."""
    st = path.stat()
    mp = path.with_suffix(".json")
    try:
        mst = mp.stat()
        m = (mst.st_size, mst.st_mtime_ns)
    except FileNotFoundError:
        m = None
    return (str(path), st.st_size, st.st_mtime_ns, m)


def load(path: Path):
    """({video_key: {pellet: score}}, manifest dict) for the scores file and
    the JSON manifest beside it, read again only when EITHER file changes.
    Raises when they cannot be read, or when they do not belong together.

    WHY both files, and the hash: the tool that writes them replaces the
    scores file first and the manifest second -- two renames, not one. Keyed
    on the scores file alone, a read that fell between the two kept the OLD
    manifest beside the NEW scores until the next import, a day later: the
    cohort's import time was then the old one, and a video whose scores had
    just arrived could be judged by the wrong import. The manifest carries
    the SHA-256 of the scores file it was written with ("scores_sha256");
    when that does not match the bytes read, the pair is read again a few
    times, a second apart (the writer is between its two renames), and if it
    still does not match this raises WITHOUT caching, so the next check reads
    again. A manifest with no hash (written before the key existed) is
    accepted as it is."""
    import hashlib
    import io
    import time
    path = Path(path)
    for attempt in range(PAIR_RETRIES + 1):
        key = _pair_key(path)
        if _CACHE["key"] == key:
            return _CACHE["scores"], _CACHE["manifest"]
        raw = path.read_bytes()
        manifest = {}
        mp = path.with_suffix(".json")
        if mp.is_file():
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        want = manifest.get("scores_sha256") if isinstance(manifest, dict) else None
        have = hashlib.sha256(raw).hexdigest()
        if want and str(want).lower() != have:
            if attempt < PAIR_RETRIES:
                time.sleep(PAIR_RETRY_WAIT_S)
                continue
            raise ScoresPairMismatch(
                "the hand-scores file %s does not match the manifest beside it (the "
                "manifest was written for another version of it); if this lasts, run "
                "the tool that writes them again" % path)
        scores: Dict[str, Dict[int, int]] = {}
        for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")):
            scores.setdefault(row["video_key"], {})[int(row["pellet"])] = int(row["score"])
        # The key taken BEFORE the read: if either file changed during it,
        # the next call sees a new key and reads again.
        _CACHE.update(key=key, scores=scores, manifest=manifest)
        return scores, manifest


class PelletNumbersUnknown(Exception):
    """A person said this video's cuts number a pellet wrongly, and the
    current cuts still disagree with them: which hand score belongs to which
    segment is not known. Raised by check() instead of comparing a pellet with
    another pellet's hand score; the watcher then sends the video to manual
    segmentation (the re-segmentation queue) with this message as the reason,
    because the problem is in the cuts."""


def unresolved_mislabels(review_doc, seg_doc=None) -> Dict[int, Optional[int]]:
    """{segment number as the record wrote it: current segment its frames
    match (None when none)} for every review record in which a person said
    the segment is really another pellet (``true_segment_num`` with
    ``segmentation_wrong``) and the CURRENT cuts do not number that stretch
    as they said. Empty when there is none.

    Matched by frames (causal_review_io.index_review_by_segment): after a
    person re-cut the video, the record's frames fall in the segment that now
    carries the number they gave, and the correction is resolved. A record
    whose frames match no current segment cannot be resolved either way --
    unless a person cut the video AFTER the record was written: the newest
    human statement about the cuts wins, so their cuts settle it (otherwise a
    video re-cut in manual segmentation could be sent back there forever)."""
    from mousereach.review.causal_review_io import (index_review_by_segment,
                                                    segmentation_is_human,
                                                    segments_from_segmentation)
    records = [r for r in (review_doc or {}).get("segments") or []
               if isinstance(r, dict) and r.get("segmentation_wrong")
               and r.get("true_segment_num") is not None]
    if not records:
        return {}
    current = segments_from_segmentation(seg_doc) if seg_doc else None
    from mousereach.review.causal_review_io import (CUTS_SET_AT_KEY, REPLACED_CUTS_KEY, _when,
                                                    given_before_recut, person_cuts_set_at)
    # A person's statement that the cuts number a pellet wrongly, made before
    # a person set the current cuts, is settled by those cuts -- on every
    # pellet, also one whose frames the re-cut did not move. WHY not the
    # answer rule (causal_review_io.answer_survives_recut, which keeps an
    # answer on an unmoved pellet): this is a statement about the cutting, and
    # a re-cut can renumber a pellet without moving its frames; the person who
    # re-cut has had the last word on the cuts, so holding to the older
    # statement would send the video back to re-segmentation for ever.
    # A PERSON's re-cut only (person_cuts_set_at): the segmenter re-cutting
    # the video is not anyone's word on the statement, so it settles nothing.
    cut_at = _when(person_cuts_set_at(seg_doc)) if seg_doc else None
    # The statements left are matched to the current cuts by their frames
    # only: the answer rule's re-cut keys are taken off the current pellets,
    # so a segmenter re-cut does not void a statement (it is still checked
    # against the new cuts, and holds the video while they disagree).
    if current:
        current = [{k: v for k, v in s.items() if k not in (CUTS_SET_AT_KEY, REPLACED_CUTS_KEY)}
                   for s in current]
    out: Dict[int, Optional[int]] = {}
    for rec in records:
        if given_before_recut(rec.get("reviewed_at") or (review_doc or {}).get("reviewed_at"),
                              cut_at):
            continue
        mapping, _ = index_review_by_segment(
            {"segments": [rec]}, current or None,
            human_segmentation=segmentation_is_human(seg_doc))
        now = next(iter(mapping), None)
        if now is None or int(rec["true_segment_num"]) != int(now):
            out[int(rec.get("segment_num") or 0)] = now
    return out


def _review_and_segments(video_id: str, folder: Path):
    """(review document or None, segments document or None) for the video, the
    review being every mislabel record in EVERY copy of the review
    (causal_review_io.mislabel_records) -- never one copy picked by file time,
    which could miss a correction another copy holds. Only the mislabel
    records are needed here (unresolved_mislabels)."""
    from mousereach.review.causal_review_io import mislabel_records, review_dirs
    sp = Path(folder) / f"{video_id}_segments.json"
    seg_doc = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else None
    review = mislabel_records(video_id, review_dirs(video_id, folder))
    return (review if review["segments"] else None), seg_doc


def scores_file_stamp() -> Optional[float]:
    """Modification time of the configured scores file, or None (also when
    the settings cannot be read: the check itself then says so). Never
    raises."""
    try:
        p = settings().get("path")
        return Path(p).stat().st_mtime if p else None
    except Exception:
        return None


# What scores_pair_stamp answers when it has no modification times to give.
# Distinct strings, never None, for the two failures. WHY: a held video is
# checked again only when the stamp CHANGES. A video held because the
# settings could not be read was kept with stamp None; once the file was
# fixed, a lab with no scores file configured also answered None, nothing had
# "changed", and the video stayed held until the watcher restarted.
STAMP_SETTINGS_UNREADABLE = "settings-unreadable"
STAMP_SCORES_FILE_UNREADABLE = "scores-file-unreadable"


def scores_pair_stamp():
    """(scores file's modification time, manifest's modification time) when a
    scores file is configured and can be seen; None when none is configured;
    STAMP_SETTINGS_UNREADABLE when the settings cannot be read;
    STAMP_SCORES_FILE_UNREADABLE when the configured file cannot be seen.
    Never raises.

    The watcher keeps this with each held video and checks the video again
    only when it changes. WHY both files: the manifest is replaced after the
    scores file, so a video checked between the two renames would otherwise
    wait for the NEXT change of the scores file -- a day later -- although
    the manifest it lacked arrived a second after."""
    try:
        p = settings().get("path")
    except Exception:
        return STAMP_SETTINGS_UNREADABLE
    if not p:
        return None
    try:
        csv_mtime = Path(p).stat().st_mtime
    except Exception:
        return STAMP_SCORES_FILE_UNREADABLE
    try:
        man_mtime = Path(p).with_suffix(".json").stat().st_mtime
    except OSError:
        man_mtime = None
    return (csv_mtime, man_mtime)


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
    # Suspended by the lab (suspension): left out of the decision entirely --
    # not compared, not waited for, never a label mismatch. ``hand`` is {} so
    # record() keeps the word and the reason with the video for the review
    # window, and no score to check answers against.
    held_out = suspension(video_id, cfg)
    if held_out is not None:
        return Verdict(SUSPENDED, suspension_words(held_out),
                       cohort_id="_".join(subject.split("_")[:2]) or None, hand={})
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
                                      if manifest.get("sheets_seen") is not None else None),
                           hand={})
        return Verdict(NOT_CHECKED, "no hand scores for this session (not required "
                       "for %s)" % (project or "this project"), cohort_id=cohort_id,
                       last_import=last, hand={})
    # From here on every verdict carries the session's scores (``hand``), so
    # record() can keep each pellet's score with the video -- also when no
    # pellet is compared because a person answered them all.
    hand = dict(hand)

    # A video is checked against ITS OWN row only: the name a person gave it is
    # trusted (owner rule 2026-11-20: "Just trust what is written down"). The
    # label-mismatch hold that used to stop a video whose results matched
    # ANOTHER tray's row better than its own is retired; such a video goes on
    # like any other, and a disagreement with its own row goes to triage as
    # usual. label_mismatch() is kept only to describe the rows in the
    # record of a sheet-error decision made while the hold existed.
    #
    # A recorded "the sheet is wrong" decision (label_decision; the record is
    # kept for good) still stands: the video is filed with the disagreement
    # recorded, never sent to triage over it. Not applied while a reviewer's
    # statement that the cuts number a pellet wrongly stands (the numbering
    # is then in doubt; the check below routes it to manual segmentation).
    out_path = Path(folder) / f"{video_id}_pellet_outcomes.json"
    out = (json.loads(out_path.read_text(encoding="utf-8"))
           if out_path.is_file() else None)
    dec = decision_for(video_id)
    if out is not None and dec and dec.get("decision") == DECISION_SHEET_ERROR:
        review, seg_doc = _review_and_segments(video_id, folder)
        if not unresolved_mislabels(review, seg_doc):
            mm = label_mismatch(video_id, algo_classes(out), scores)
            rows = (describe_mismatch(mm).split(" -- ")[0] if mm is not None
                    else "compared with its own row")
            return Verdict(SHEET_ERROR, (
                "a person (%s, %s) decided the sheet is wrong for this session: "
                "kept as it is and recorded; %s" % (dec.get("by"), dec.get("at"), rows)),
                cohort_id=cohort_id, last_import=last, hand=hand, mismatch=mm,
                decision=dec)

    from mousereach.watcher.route_cli import answered_segments_of
    answered = answered_segments_of(video_id, folder)
    if "all" in answered:
        return Verdict(AGREES, "every pellet answered by a person", cohort_id=cohort_id,
                       last_import=last, hand=hand)
    if out is None:
        out = json.loads(out_path.read_text(encoding="utf-8"))   # raises: as before
    # Segment N is compared with hand pellet N only while nothing says the
    # cuts number the pellets wrongly. A person's mislabel that the current
    # cuts still contradict means the numbering of every other segment is in
    # doubt: hold the video rather than compare a pellet with another
    # pellet's score (unresolved_mislabels says how it is decided).
    open_segs = [s.get("segment_num") for s in out.get("segments", [])
                 if s.get("segment_num") is not None and s.get("segment_num") not in answered]
    if open_segs:
        review, seg_doc = _review_and_segments(video_id, folder)
        wrong = unresolved_mislabels(review, seg_doc)
        if wrong:
            raise PelletNumbersUnknown(
                "a reviewer said segment(s) %s show a different pellet, and the "
                "current cuts still number them otherwise, so which hand score "
                "belongs to which segment is not known; the cuts must be set "
                "or confirmed in manual segmentation before the hand scores are "
                "compared"
                % ", ".join(str(k) for k in sorted(wrong)))
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
            cohort_id=cohort_id, last_import=last, hand=hand)
    return Verdict(AGREES, "every pellet agrees with the hand score", cohort_id=cohort_id,
                   last_import=last, hand=hand)


def flag_disagreements(folder: Path, video_id: str, verdict: Verdict) -> list:
    """Flag every pellet of a DISAGREES verdict for triage in the video's
    <video>_pellet_outcomes.json, with the reason "bench disagreement: the
    hand score says X; the algorithm says Y". Returns the segments flagged.

    A pellet the pipeline had already flagged keeps that reason after the
    bench one, "(also: ...)": the reviewer must see both, and the text must
    still START with "bench disagreement" -- that is how the review tool
    recognises it (is_bench_mismatch). On a second check of the same file
    the old reason is our own bench text: only the "(also: ...)" part it
    carries is kept, so flagging twice gives the same text as once.

    Used by the pipeline before the review gate (orchestrator
    _hand_score_precheck) and by anything that re-checks videos already
    waiting in a review queue, so both write exactly the same flag."""
    from mousereach.watcher.route_cli import flag_segments
    out_path = Path(folder) / f"{video_id}_pellet_outcomes.json"
    details = dict(verdict.disagreements)
    for seg in json.loads(out_path.read_text(encoding="utf-8")).get("segments", []):
        n, old = seg.get("segment_num"), str(seg.get("flag_reason") or "")
        if n not in details or not seg.get("flagged_for_review"):
            continue
        if old.lower().startswith(FLAG_REASON):
            i = old.find(" (also: ")
            old = old[i + len(" (also: "):-1] if i >= 0 and old.endswith(")") else ""
        if old:
            details[n] = "%s (also: %s)" % (details[n], old)
    return flag_segments(out_path, list(details), FLAG_REASON, details=details)


def clear_bench_flags(folder: Path, video_id: str, keep=()) -> list:
    """Take back every "bench disagreement" flag in the video's outcomes file
    except on the segments in ``keep``; returns the segments cleared. A flag
    that carried an earlier reason, "(also: <reason>)", goes back to that
    reason and stays flagged; one that did not is unflagged.

    WHY: a bench flag is the result of ONE comparison with ONE sheet row. When
    that row turns out to be the wrong tray (a label mismatch), or the sheet
    is corrected and the video now agrees, a flag left from the old comparison
    would send the video to triage over a disagreement that no longer exists.
    Called before every comparison result is applied, so the flags in the
    file are always those of the latest comparison. Callers keep the flags of
    pellets a person answered (``keep``): the flag is part of the record of
    why that pellet was asked, and a bundle whose every flag is answered is
    what the ordinary return releases -- taking an answered pellet's flag away
    could leave a bundle with nothing flagged and nothing to release it. A
    person's answers are in the review file and are not touched."""
    out_path = Path(folder) / f"{video_id}_pellet_outcomes.json"
    if not out_path.is_file():
        return []
    data = json.loads(out_path.read_text(encoding="utf-8"))
    keep = {k if k == "all" else int(k) for k in (keep or ())}
    cleared = []
    for seg in data.get("segments", []):
        n, reason = seg.get("segment_num"), str(seg.get("flag_reason") or "")
        if "all" in keep:
            break
        if n is None or int(n) in keep or not reason.lower().startswith(FLAG_REASON):
            continue
        i = reason.find(" (also: ")
        old = reason[i + len(" (also: "):-1] if i >= 0 and reason.endswith(")") else ""
        seg["flag_reason"] = old or None
        seg["flagged_for_review"] = bool(old)
        cleared.append(int(n))
    if cleared:
        from mousereach.pipeline.fsutil import write_json_atomic
        write_json_atomic(out_path, data, indent=2)
    return cleared


# The word kept in the outcomes file for each verdict (see the module
# docstring, "EVERY PELLET'S SCORE IS KEPT WITH THE VIDEO"). A WAITING verdict
# is never recorded by the watcher any more: such a video is held until its
# scores arrive. "never_scored" stays here because videos filed before that
# rule carry it (they were filed uncompared). The processing server compares
# those once their scores arrive and lists every disagreement for a person,
# never touching the filed video (watcher/filed_hand_scores.py).
RECORD_WORDS = {AGREES: "agrees", DISAGREES: "disagrees", WAITING: "never_scored",
                NOT_CHECKED: "not_required", LABEL_MISMATCH: "label_mismatch",
                SUSPENDED: "suspended", SHEET_ERROR: "sheet_error"}
RECORD_KEY = "hand_scores"


def record(folder: Path, video_id: str, verdict: Verdict, stamp: Optional[float]) -> bool:
    """Keep what the sheet said for this session in the video's
    <video>_pellet_outcomes.json, under the top-level key "hand_scores"
    (module docstring). Returns True when written, False when there is
    nothing to keep (the sheet was not read: Easy/Flat tray, no scores file
    configured, or no outcomes file). Raises when the outcomes file exists
    but cannot be read or written -- the caller then holds the video, as for
    any check that could not run.

    Read-modify-write of the WHOLE file, written to a temporary file beside it
    and then put in place with one rename: every other key the pipeline wrote
    is kept, and a reader never sees a half-written file."""
    from datetime import datetime
    if verdict.hand is None:
        return False
    path = Path(folder) / f"{video_id}_pellet_outcomes.json"
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    data[RECORD_KEY] = {
        "pellets": {str(int(k)): int(v) for k, v in sorted(verdict.hand.items())},
        "verdict": RECORD_WORDS.get(verdict.status, verdict.status),
        "source_mtime": stamp,
        "cohort_last_import": verdict.last_import,
        "read_at": datetime.now().isoformat(timespec="seconds"),
    }
    if verdict.status == SUSPENDED:
        data[RECORD_KEY]["reason"] = verdict.reason
    # A label mismatch and a person's decision about it travel with the video
    # into its filed folder, so a filed sheet error can be counted from the
    # filed results too (the decisions record is the other place).
    if verdict.mismatch is not None:
        m = verdict.mismatch
        data[RECORD_KEY]["label_mismatch"] = {
            "own_row": {"key": m.own.key, "agree": m.own.agree, "compared": m.own.compared},
            "matched_rows": [{"key": x.key, "agree": x.agree, "compared": x.compared}
                             for x in m.matches]}
    if verdict.decision:
        data[RECORD_KEY]["decision"] = verdict.decision
    # fsutil.write_json_atomic, not a "<name>.tmp" of our own. WHY: that
    # temporary name starts with the video's name, so one left by a crash was
    # taken by the router and the archiver as one of the video's files; it
    # was also never flushed to disk before the rename, and a rename refused
    # while a reader held the file failed at once.
    from mousereach.pipeline.fsutil import write_json_atomic
    write_json_atomic(path, data, indent=2)
    return True


def pellet_of_record(record, records=None):
    """(pellet number, "") for the hand score a review record's answer is
    checked against, or (None, why not).

    The record's own ``true_segment_num`` when a person said this segment is
    really another pellet; else its segment number -- unless another record
    of the same video says the cuts number a pellet wrongly
    (``segmentation_wrong`` with a different ``true_segment_num``): then the
    numbering of every segment is in doubt and no score is compared."""
    rec = record or {}
    if rec.get("segmentation_wrong") and rec.get("true_segment_num") is not None:
        return int(rec["true_segment_num"]), ""
    for other in (records or {}).values() if isinstance(records, dict) else (records or []):
        if (isinstance(other, dict) and other is not record and other.get("segmentation_wrong")
                and other.get("true_segment_num") is not None
                and other.get("segment_num") is not None
                and int(other["true_segment_num"]) != int(other["segment_num"])):
            return None, ("a reviewer said segment %s shows pellet %s, so which "
                          "hand score belongs to which segment is not known"
                          % (other["segment_num"], other["true_segment_num"]))
    n = rec.get("segment_num")
    return (int(n), "") if n is not None else (None, "no segment number")


def score_from_outcomes(outcomes_doc, pellet):
    """(score, "") for one pellet's hand score (0/1/2, see HAND_WORDS) as
    record() kept it in the video's outcomes file, or (None, why not) when
    there is none to compare with. Never raises.

    Used by the review window's check-your-answer box
    (review/double_disagreement.py). WHY from the outcomes file and not the
    scores file: it is the score the pipeline compared this pellet with, it
    travels inside the review bundle, and the review window then needs no
    setting or access to the scores file. Keyed by pellet number, as check()
    keys it; the caller passes the pellet the record is about
    (pellet_of_record), not a bare segment number."""
    try:
        kept = (outcomes_doc or {}).get(RECORD_KEY)
        if not isinstance(kept, dict):
            return None, "no hand scores kept with this video (analysed before they were kept)"
        s = (kept.get("pellets") or {}).get(str(int(pellet)))
    except Exception as e:
        return None, "hand scores kept with this video could not be read (%s)" % e
    if kept.get("verdict") == RECORD_WORDS[SUSPENDED]:
        return None, kept.get("reason") or "the bench comparison is suspended for this session"
    if s not in HAND_WORDS:
        if not kept.get("pellets"):
            return None, "this session has no hand scores (%s)" % kept.get("verdict")
        return None, "no hand score for pellet %s" % pellet
    return s, ""


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
    video was ready and the session still has no hand scores: the scores are
    MISSING, not merely late. The watcher then keeps holding the video, now
    as a defect a person must fix (listed in red); before that import it
    only waits. WHY hold and not file: in a required project every session is
    scored by hand before its videos enter the pipeline, so no scores after
    an import means a sheet row that is absent, misnamed, or not matched to
    this video -- which looks exactly like "never scored" from here. Filed
    uncompared, the video would never be compared again."""
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


def no_sheet_for_cohort(verdict: Verdict) -> bool:
    """True when the cohort has never been imported and the latest import of
    every sheet found no sheet by its name -- the case in which
    scores_never_came calls the scores missing without any import of the
    cohort. The remedy then is a sheet (or a corrected cohort name), not a
    row: the messages must say so, or a person goes looking for a sheet that
    does not exist."""
    return not verdict.last_import and verdict.has_sheet is False


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
                    "says": ("%s is analysed but held before triage and kinematics: the "
                             "check against its hand scores could not run (%s). Fix the "
                             "\"hand_scores\" setting (in the lab's "
                             "mousereach_lab_config.json on the storage drive, or in this "
                             "PC's config.json if it has its own), or the file it names; "
                             "the video goes on by itself once the check runs."
                             % (vid, w["error"])),
                })
                continue
            if w.get("label_mismatch"):
                # RETIRED 2026-11-20: nothing sets this any more (check()
                # never returns LABEL_MISMATCH, and a hold restored from
                # before the retirement comes back as a plain hold).
                # The results match another tray's hand scores (label_mismatch):
                # held outside the review queues until the name or the sheet is
                # corrected. The structured fields let a reader (a database
                # tool's fix-these list) name both trays without parsing "says".
                items.append({
                    "kind": "label_mismatch", "needs_person": True,
                    "video_id": vid, "cohort_id": w.get("cohort_id"),
                    "ready_since": w.get("ready_since"),
                    "own_row": w.get("own_row"),
                    "matched_rows": w.get("matched_rows") or [],
                    "partner_back": w.get("partner_back"),
                    "answers_kept": w.get("answers_kept") or [],
                    "says": "%s is analyzed but held outside the review queues: %s." % (
                        vid, w.get("says") or "its results match another tray's hand scores"),
                })
                continue
            if w.get("missing") and w.get("no_sheet"):
                # No sheet at all for the cohort (no_sheet_for_cohort): an
                # import of every sheet ran after the video was ready and
                # none carries this cohort's name. Telling the person to fix
                # a row in "the <cohort> sheet" would send them to a sheet
                # that does not exist. A misspelled cohort in the VIDEO's name
                # cannot be fixed by renaming the held files (see below), so
                # that case goes to whoever maintains the pipeline.
                items.append({
                    "kind": "hand_scores_missing", "needs_person": True,
                    "no_sheet": True,
                    "video_id": vid, "cohort_id": w.get("cohort_id"),
                    "ready_since": w.get("ready_since"),
                    "last_import": w.get("last_import"),
                    "says": ("%s is analysed but held before triage and kinematics: no "
                             "tracking sheet was found for cohort %s -- an import of every "
                             "sheet ran after the video was ready and none is named for "
                             "this cohort -- so its hand scores cannot be found. Add the "
                             "cohort's tracking sheet to the sheets folder, or correct the "
                             "cohort in the sheet's file name if it is misspelled. If it is "
                             "the video's name that carries the wrong cohort, do not rename "
                             "the held files: tell whoever maintains the pipeline. The "
                             "video goes on by itself after the next import brings its "
                             "scores." % (vid, w.get("cohort_id"))),
                })
                continue
            if w.get("missing"):
                # The cohort's sheet was imported after the video was ready
                # and still has no row for it: a defect, not a wait
                # (scores_never_came). The remedy named is the SHEET only.
                # WHY never "rename the video": a held video is a
                # 'processing' row whose files sit in this watcher's local
                # Processing folder, and nothing renames such a video -- the
                # renamed files would belong to no row and the hold would
                # never clear. The sheet decides when and what was recorded.
                items.append({
                    "kind": "hand_scores_missing", "needs_person": True,
                    "video_id": vid, "cohort_id": w.get("cohort_id"),
                    "ready_since": w.get("ready_since"),
                    "last_import": w.get("last_import"),
                    "says": ("%s is analysed but held before triage and kinematics: the "
                             "%s sheet was imported after the video was ready and has no "
                             "hand scores for this session. The scores are missing from "
                             "the sheet, or the sheet's row does not match the video. "
                             "Fix the sheet: enter the session's scores, or correct its "
                             "row so the date, animal and tray match the video's. The "
                             "video goes on by itself after the next import brings its "
                             "scores." % (vid, w.get("cohort_id"))),
                })
                continue
            items.append({
                "kind": "hand_scores_waiting",
                "needs_person": False,
                "video_id": vid,
                "cohort_id": w.get("cohort_id"),
                "ready_since": w.get("ready_since"),
                "last_import": w.get("last_import"),
                "says": ("%s is analysed and held before triage and kinematics until the "
                         "%s sheet's next import, to be compared with its hand scores "
                         "(if that import has none, it stays held and is listed as "
                         "missing its scores)."
                         % (vid, w.get("cohort_id"))),
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
