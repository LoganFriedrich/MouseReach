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
            was filed uncompared, never to be compared again.)

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
    from mousereach.review.causal_review_io import answer_survives_recut, recut_time
    # The one re-cut rule (causal_review_io.answer_survives_recut): a person's
    # statement made before a person set the current cuts is settled by them.
    cut_at = recut_time(current)
    out: Dict[int, Optional[int]] = {}
    for rec in records:
        if not answer_survives_recut(rec, review_doc, cut_at):
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

    from mousereach.watcher.route_cli import answered_segments_of
    answered = answered_segments_of(video_id, folder)
    if "all" in answered:
        return Verdict(AGREES, "every pellet answered by a person", cohort_id=cohort_id,
                       last_import=last, hand=hand)
    out = json.loads((Path(folder) / f"{video_id}_pellet_outcomes.json").read_text(encoding="utf-8"))
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


# The word kept in the outcomes file for each verdict (see the module
# docstring, "EVERY PELLET'S SCORE IS KEPT WITH THE VIDEO"). A WAITING verdict
# is never recorded by the watcher any more: such a video is held until its
# scores arrive. "never_scored" stays here because videos filed before that
# rule carry it (they were filed uncompared; a person can list them by it).
RECORD_WORDS = {AGREES: "agrees", DISAGREES: "disagrees", WAITING: "never_scored",
                NOT_CHECKED: "not_required"}
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
