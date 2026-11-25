"""
mousereach.kinematics.core.answers_applied -- is every person's answer IN a
features file?

Rule: a person's answer -- ground truth, a deep-review
(re-segmentation) answer, a triage answer -- is ALWAYS applied, with no hold,
queue or question. Two places act on that with this one check:

  * the archive step, before it files a video (orchestrator.
    _apply_answers_before_filing): a features file that lacks an answer is
    made again with it first; filing never completes without it;
  * the version scan over filed videos (watcher/reprocessor.py): a filed
    features file that lacks an answer gets a kinematics-only re-run, which
    applies it (filed_decisions.settle_against_filed).

WHY a check of the CONTENT and not of a stamp: the manifest's review sha256
stamp, or, without one, the times of the files, say only that a run SAW the
file, not that the answer is in the result. Filed features can lack answers
in two ways:
  1. ground truth filed beside the video that its run never found (the
     extractor looked for it only in the processing tree), with a manifest
     lacking the ground-truth stamp, so the time rule called it applied;
  2. review answers applied by an older rule to a different pellet than the
     one whose frames they are about (re-cut or renumbered pellets), with a
     stamp that said the review was applied.
The stamp stays (it is what makes the version scan cheap); this check is
what decides.

What "in the features file" means, per pellet, for what a person decides
now (FeatureExtractor.resolve_decisions, the extractor's own resolution, so
the two can never disagree):
  * the outcome, when a person's answer decides it (outcome_source
    human_review or ground_truth);
  * the causal reach, by FRAMES, when a person decided it
    (causal_reach_owned_by_person);
  * every reach a person placed (reach_source human_review / ground_truth),
    by frames.
Answers given on cuts that have since moved do not apply (the pellet's span
changed) and are not required here.

Never writes. Never raises: a failure to read is returned as a line, so the
caller holds rather than files on a guess.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

HUMAN = ("human_review", "ground_truth")


def _window(r):
    try:
        return int(r.get("start_frame")), int(r.get("end_frame"))
    except (TypeError, ValueError, AttributeError):
        return None


def _fmt(w) -> str:
    if w is None:
        return "none"
    if isinstance(w, tuple) and len(w) == 2:
        return "frames %s-%s" % w
    return str(w)


def _any_answer(video_id: str, folder: Path, review_path=None) -> bool:
    """True when a copy of the video's review or a ground-truth file exists in
    any place the extractor reads them from (the same lookups it uses)."""
    from mousereach.review.causal_review_io import (find_gt, resolve_review_path,
                                                    review_copies, review_dirs)
    if review_path is not None or resolve_review_path(video_id, folder) is not None:
        return True
    bad: list = []
    if review_copies(video_id, list(review_dirs(video_id, folder)) + [folder],
                     unreadable=bad) or bad:
        return True     # an unreadable copy: the full check reports it
    gt_dirs = [folder]
    try:
        from mousereach.archive.core import get_archive_destination
        gt_dirs.append(Path(get_archive_destination(video_id)))
    except Exception:
        pass
    return find_gt(video_id, extra_dirs=gt_dirs) is not None


def unapplied_answers(video_id: str, folder, review_path=None) -> List[str]:
    """One line per pellet whose person's answer the features file in
    ``folder`` does not carry; empty when every answer is in it (or the video
    has no pellets to answer -- an Easy/Flat tray has no outcomes file).

    ``review_path``: the review copy to start from (resolve_review_path(
    video_id, folder) when None); every other copy is merged in as the
    extractor does."""
    folder = Path(folder)
    features_p = folder / f"{video_id}_features.json"
    reaches_p = folder / f"{video_id}_reaches.json"
    outcomes_p = folder / f"{video_id}_pellet_outcomes.json"
    if not outcomes_p.is_file():
        return []
    if not features_p.is_file() or not reaches_p.is_file():
        return []        # nothing to compare: the archive step's own checks say so
    try:
        doc = json.loads(features_p.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"the features file {features_p.name} could not be read ({e}), so "
                f"whether it carries every person's answer cannot be told"]
    # No person's answer anywhere -- no review copy, no ground truth: nothing
    # to carry. Asked first because it is cheap and true of most videos.
    try:
        if not _any_answer(video_id, folder, review_path):
            return []
    except Exception as e:
        return [f"whether a person answered this video could not be told ({e})"]
    try:
        from mousereach.kinematics.core.feature_extractor import (
            FeatureExtractor, KinematicsProblem, causal_reach_owned_by_person,
            split_reaches_by_ignore_windows)
        if review_path is None:
            from mousereach.review.causal_review_io import resolve_review_path
            review_path = resolve_review_path(video_id, folder)
        ex = FeatureExtractor()
        ex.problems = []
        try:
            _r, _o, pairs = ex.resolve_decisions(reaches_p, outcomes_p,
                                                 review_path=review_path,
                                                 ask_voided=False)
        except KinematicsProblem as e:
            return [f"a person's answers could not be worked out ({e})"]
    except Exception as e:
        return [f"a person's answers could not be worked out ({e})"]

    filed = {}
    for g in doc.get("segments") or []:
        try:
            filed[int(g.get("segment_num"))] = g
        except (TypeError, ValueError):
            continue
    out: List[str] = []
    for sd, od in pairs:
        k = int(sd["segment_num"])
        human_outcome = od.get("outcome_source") in HUMAN
        measured, left_out = split_reaches_by_ignore_windows(sd.get("reaches") or [],
                                                             od.get("abnormal_ranges"))
        everyone = list(measured) + list(sd.get("excluded_reaches") or []) + list(left_out)
        person_reaches = sorted({w for r in everyone
                                 if isinstance(r, dict) and r.get("reach_source") in HUMAN
                                 for w in [_window(r)] if w})
        person_causal = human_outcome and causal_reach_owned_by_person(od)
        if not (human_outcome or person_reaches or person_causal):
            continue
        g = filed.get(k)
        if g is None:
            out.append(f"pellet {k}: a person answered it and the features file has no "
                       f"such pellet")
            continue
        words = []
        if human_outcome and g.get("outcome") != od.get("outcome"):
            words.append("outcome %s, the person's answer is %s (%s)" % (
                g.get("outcome"), od.get("outcome"), od.get("outcome_source")))
        filed_all = [r for r in (g.get("reaches") or []) + (g.get("reaches_left_out") or [])
                     if isinstance(r, dict)]
        if person_causal:
            want = None
            cid = od.get("causal_reach_id")
            if cid is not None:
                want = next((_window(r) for r in everyone
                             if isinstance(r, dict) and r.get("reach_id") == cid), None)
            have = None
            if g.get("causal_reach_id") is not None:
                have = next((_window(r) for r in filed_all
                             if r.get("reach_id") == g.get("causal_reach_id")),
                            ("reach id", g.get("causal_reach_id")))
            if want != have:
                words.append("causal reach %s, the person's is %s" % (_fmt(have), _fmt(want)))
        have_windows = {w for w in (_window(r) for r in filed_all) if w}
        missing = [w for w in person_reaches if w not in have_windows]
        if missing:
            words.append("%d reach(es) a person placed are not in it (first: %s)" % (
                len(missing), _fmt(missing[0])))
        if words:
            out.append("pellet %d: %s" % (k, "; ".join(words)))
    return out
