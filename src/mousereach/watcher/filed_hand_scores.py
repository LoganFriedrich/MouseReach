"""Filed videos that were never compared with their hand scores, compared now.

WHY THIS EXISTS
---------------
For a while a video with no hand scores in a project that requires them was
filed anyway, "never scored by hand", with the comparison skipped
(watcher/hand_scores.py says why that is no longer so: such a video is now
held until its scores arrive). Each of those videos keeps, in its filed
``<video>_pellet_outcomes.json``, the record ``"hand_scores": {"verdict":
"never_scored", ...}``. Missing scores are the pipeline's defect (the sheet
row was absent, misnamed, or not matched to the video), and once a later
sheet import brings them nothing compared them: a pellet the pipeline and the
bench disagree about stayed in the analysis data with no flag.

WHAT IT DOES
------------
After every change of the hand-scores file (the same stamp the hold uses,
hand_scores.scores_pair_stamp), the processing server's watcher looks at
every filed outcomes file in Analyzed (never a superseded generation under
Archive/) whose kept verdict is ``never_scored`` and runs the ordinary check
(hand_scores.check) on it, READ-ONLY. It writes the result to the attention
list ``<NAS root>/attention/<host>_filed_hand_scores.json`` (replaced whole
each time), which the Pipeline Dashboard's red panel and
mousereach-watch-status show:

  * scores arrived and a pellet disagrees -> one item per video, naming each
    pellet ("the hand score says retrieved; the algorithm says displaced_sa");
  * scores arrived and every pellet agrees -> nothing listed;
  * still no scores, and the cohort's sheet has been imported since the video
    was filed -> one item per cohort listing those videos: the sheet rows that
    should match them are absent or misnamed (the same defect a held video is
    listed for, hand_scores.write_attention);
  * still no scores and no import of the cohort since -> nothing yet;
  * the numbering of the cuts is in doubt (hand_scores.PelletNumbersUnknown),
    or the check could not run, or it ran but did not compare the pellets
    (its verdict is label_mismatch, sheet_error or not_checked) -> one item
    per video saying it was not compared, and why (the verdict is named).

WHAT IT NEVER DOES
------------------
It never moves, re-files, re-opens or writes into a filed video: filed means
done, and a filed file whose time changed could look newer than its
features file to the checks that compare them. Whether a filed video found
to disagree is re-opened is not decided; it is listed and left exactly as it
is. Nothing is routed to a review queue.

The walk reads each outcomes file once per run of the watcher: what a file
says is remembered by (path, size, modification time), so later runs read
only files that changed. It runs on its own thread, so the watcher's work
does not wait for it.

ASCII-only text (Windows consoles cannot print Unicode).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LIST_KIND = "filed_hand_scores"           # <host>_filed_hand_scores.json
NEVER_SCORED = "never_scored"
OUTCOMES_SUFFIX = "_pellet_outcomes.json"

# (path, size, mtime_ns) -> (video_id, kept verdict). Process-wide on purpose:
# the walk is the expensive part and a file's kept verdict changes only when
# the file does.
_VERDICT_CACHE: Dict[Tuple[str, int, int], Tuple[str, Optional[str]]] = {}


def _kept(path: Path, st) -> Tuple[str, Optional[str], Optional[str]]:
    """(video_id, kept verdict, read_at) of one outcomes file."""
    vid = path.name[: -len(OUTCOMES_SUFFIX)]
    key = (str(path), st.st_size, st.st_mtime_ns)
    hit = _VERDICT_CACHE.get(key)
    if hit is not None:
        return hit[0], hit[1], hit[2] if len(hit) > 2 else None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        hs = doc.get("hand_scores") if isinstance(doc, dict) else None
        verdict = hs.get("verdict") if isinstance(hs, dict) else None
        read_at = hs.get("read_at") if isinstance(hs, dict) else None
    except (OSError, ValueError):
        verdict, read_at = None, None
    _VERDICT_CACHE[key] = (vid, verdict, read_at)
    return vid, verdict, read_at


class ListNotWritten(RuntimeError):
    """The list was left as it was: the walk could not be trusted, or the
    list could not be written. The Rechecker then tries again on its next
    scan (it does not mark the scores-file stamp done)."""


def _walk(analyzed_root) -> Tuple[Dict[str, dict], int]:
    """(never-scored candidates, number of outcomes files seen at all)."""
    from mousereach.pipeline.analyzed_tree import iter_file_entries
    out: Dict[str, dict] = {}
    seen = 0
    for e in iter_file_entries(analyzed_root, "*" + OUTCOMES_SUFFIX):
        seen += 1
        p = Path(e.path)
        try:
            st = e.stat()
        except OSError:
            continue
        vid, verdict, read_at = _kept(p, st)
        if verdict == NEVER_SCORED:
            out[vid] = {"folder": p.parent, "filed_check_at": read_at}
    return out, seen


def never_scored_filed(analyzed_root) -> Dict[str, dict]:
    """``{video_id: {"folder": Path, "filed_check_at": str|None}}`` for every
    filed outcomes file whose kept hand-score verdict is never_scored."""
    return _walk(analyzed_root)[0]


def _suspended() -> set:
    """Sessions the lab's settings exempt from the comparison
    ("suspended_sessions": [{"video_key": ...}, ...]); empty when none."""
    try:
        from mousereach.watcher import hand_scores
        rows = hand_scores.settings().get("suspended_sessions") or []
        return {r.get("video_key") for r in rows if isinstance(r, dict)}
    except Exception:
        return set()


def compare(candidates: Dict[str, dict]) -> List[dict]:
    """Run the ordinary check on each filed never-scored video and return the
    attention items (module docstring says which cases are listed). Reads
    only."""
    from mousereach.watcher import hand_scores
    items: List[dict] = []
    missing_by_cohort: Dict[str, List[str]] = {}
    suspended = _suspended()
    for vid in sorted(candidates):
        if vid in suspended:
            continue
        c = candidates[vid]
        folder = Path(c["folder"])
        try:
            v = hand_scores.check(vid, folder)
        except hand_scores.PelletNumbersUnknown as e:
            items.append(_item(vid, None, (
                "%s was filed before it could be compared with its hand scores, and it "
                "still cannot be: %s. It stays filed and unchanged." % (vid, e))))
            continue
        except Exception as e:
            items.append(_item(vid, None, (
                "%s was filed before it could be compared with its hand scores; the "
                "comparison could not run now (%s). It stays filed and unchanged; it is "
                "tried again after the next change of the hand-scores file." % (vid, e))))
            continue
        if v.status == hand_scores.DISAGREES:
            pellets = "; ".join("pellet %d: %s" % (n, words)
                                for n, words in sorted(v.disagreements.items()))
            items.append(dict(_item(vid, None, (
                "%s was filed before its hand scores arrived; they have arrived and "
                "disagree with the pipeline: %s. It stays filed and unchanged (whether a "
                "filed video is re-opened for this is not decided); the disagreement is "
                "in the analysis data until then." % (vid, pellets))),
                kind="filed_hand_score_disagreement",
                disagreements={str(k): w for k, w in sorted(v.disagreements.items())}))
        elif v.status == hand_scores.WAITING:
            # Missing, not merely late, once the cohort's sheet has been
            # imported after the video was filed (the same judgement as a
            # held video's, hand_scores.scores_never_came).
            if hand_scores.scores_never_came(v, c.get("filed_check_at")):
                missing_by_cohort.setdefault(v.cohort_id or "?", []).append(vid)
        elif v.status not in (hand_scores.AGREES, hand_scores.SUSPENDED):
            # LABEL_MISMATCH, SHEET_ERROR, NOT_CHECKED: the check ran but did
            # not compare the pellets. Say so by name; a silent gap here would
            # read as "compared, agrees".
            items.append(_item(vid, None, (
                "%s was filed before it could be compared with its hand scores, and it "
                "was not compared now (check result: %s; %s). It stays filed and "
                "unchanged." % (vid, v.status, v.reason))))
    for cohort, vids in sorted(missing_by_cohort.items()):
        shown = ", ".join(vids[:20]) + (" ... and %d more" % (len(vids) - 20)
                                        if len(vids) > 20 else "")
        items.append({
            "kind": "filed_hand_scores_missing", "needs_person": True,
            "cohort_id": None if cohort == "?" else cohort, "video_ids": vids,
            "says": ("%d filed video(s) of cohort %s were filed without hand scores, and the "
                     "%s sheet has been imported since without a row for them: %s. Their "
                     "rows are absent from the sheet or do not match the videos' date, "
                     "animal and tray. Fix the sheet; the next import compares them. The "
                     "videos stay filed and unchanged."
                     % (len(vids), cohort, cohort, shown)),
        })
    return items


def _item(vid: str, cohort: Optional[str], says: str) -> dict:
    """One per-video item. ``cohort`` is None for every per-video item here:
    a tool that copies a cohort's red items into a file beside its tracking
    sheet (for the people who fix the sheet) must not get a disagreement
    whose remedy is not decided, or a comparison that could not run. Only
    the missing-scores item, whose remedy IS the sheet, names its cohort."""
    return {"kind": "filed_hand_scores_not_compared", "needs_person": True,
            "video_id": vid, "cohort_id": cohort, "says": says}


def run(nas_root, hostname: str, analyzed_root) -> List[dict]:
    """Find, compare and write this machine's list. Returns the items.

    Raises (ListNotWritten, or the walk's own error) whenever the list is NOT
    replaced, so the caller does not count the run as done and tries again
    on its next scan; the list is then left as it was. WHY each refusal:

      * the Analyzed folder is missing or unreadable, or the walk found no
        outcomes file at all: an empty walk of an unreachable share must not
        be read as "nothing to list" -- it would wipe every red item until
        the next change of the scores file;
      * the list could not be written (attention.write_list returned False):
        marking the run done would leave the old list in place until the
        next change of the scores file or restart of the watcher.
    """
    from mousereach.watcher import attention
    root = Path(analyzed_root) if analyzed_root else None
    if root is None or not root.is_dir():
        raise ListNotWritten(f"the Analyzed folder {analyzed_root} cannot be read now; "
                             f"the list was left as it was")
    candidates, seen = _walk(root)
    if seen == 0:
        raise ListNotWritten(f"no outcomes file was found under {root} (unreadable "
                             f"share?); the list was left as it was")
    items = compare(candidates)
    if not attention.write_list(nas_root, hostname, LIST_KIND, items):
        raise ListNotWritten(f"the attention list {hostname}_{LIST_KIND}.json could not "
                             f"be written; it is tried again on the next scan")
    return items


class Rechecker:
    """Runs ``run`` on its own thread whenever the hand-scores stamp has
    changed since the last finished run (``maybe_start``, called from the
    watcher's scan). One run at a time. A run that fails (including one that
    could not trust its walk or write its list, see ``run``) does not mark
    the stamp done, so the next scan tries again."""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._done_stamp = object()          # matches no stamp: first call runs

    def maybe_start(self, nas_root, hostname: str, analyzed_root) -> bool:
        """True when a run was started. Never raises."""
        try:
            from mousereach.watcher import hand_scores
            if not nas_root or not analyzed_root:
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            if not hand_scores.settings().get("path"):
                return False
            stamp = hand_scores.scores_pair_stamp()
            if not isinstance(stamp, tuple) or stamp == self._done_stamp:
                return False

            def job():
                try:
                    items = run(nas_root, hostname, analyzed_root)
                    self._done_stamp = stamp
                    if items:
                        logger.warning(f"{len(items)} item(s) about filed videos never "
                                       f"compared with their hand scores (attention list "
                                       f"{hostname}_{LIST_KIND}.json)")
                except Exception as e:
                    logger.warning(f"Could not compare filed never-scored videos with "
                                   f"their hand scores: {e}")
            self._thread = threading.Thread(target=job, daemon=True,
                                            name="filed-hand-scores")
            self._thread.start()
            return True
        except Exception as e:
            logger.debug(f"filed hand-score recheck not started: {e}")
            return False
