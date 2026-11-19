"""
mousereach.kinematics.core.filed_decisions -- what a kinematics-only re-run of a
FILED video may change.

A filed video is re-run "for kinematics only" (the watcher's re-run scope
'kinematics', orchestrator.filed_rerun_reuses_analysis) for two reasons:

  1. the kinematic extractor changed (a new version is declared in
     pipeline_versions.json). Such a run exists to recompute MEASUREMENTS --
     speeds, paths, seconds. It runs no algorithm, no quality triage and no
     hand-score comparison, because nothing those checks judge is supposed to
     change.
  2. a person gave an answer the filed kinematics have not applied yet: a
     review saved after filing, or ground truth saved or changed after filing
     (the version scan's rules: video_prep.core.collage_provenance.
     review_applied and pipeline.manifest.ground_truth_applied).

The feature extractor does not copy the filed decisions; it works them out
again from the filed reaches, outcomes, reach assignment, every copy of the
review and the ground truth (review.truth_resolver). If the rules that turn
those inputs into decisions changed since filing, or an input the filing run
did not see is found now, the same files give DIFFERENT reaches, outcomes or
causal reaches -- and a kinematics-only run would file them silently, without
any of the checks a change of decision must pass. Found on 2026-11-17: a
ground-truth file that sat beside four filed videos but was not applied when
they were filed (the extractor then looked for ground truth only in the
processing tree), and a review answer that newer review matching places on a
different pellet than the filing run did.

THE RULE: in a kinematics-only re-run of a filed video, the reaches (their
frames), the reaches left out of the measurements, each pellet's outcome and
its causal reach must come out exactly as filed -- except where an answer the
filed kinematics have NOT applied decides them (reason 2 above). Anything else
is not written: the extractor raises KinematicsProblem with the changes in
words, the watcher holds the run, and the filed results stand untouched
(orchestrator._restore_filed_after_failed_rerun puts the row back to
'archived' with the reason and lists it for a person). Nothing is re-decided
silently, and nothing a person decided is dropped silently.

Pure functions over plain dicts; never touches a file except
owed_human_layers, which only reads.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The two kinds of a person's answer a kinematics-only re-run can be owed.
REVIEW = "human_review"
GROUND_TRUTH = "ground_truth"

# How many pellets a hold message names before it says "and N more".
_NAMED = 6


def _window(r) -> Optional[Tuple[int, int]]:
    try:
        return int(r.get("start_frame")), int(r.get("end_frame"))
    except (TypeError, ValueError, AttributeError):
        return None


def _left_out_kind(item) -> str:
    """Whose statement left a reach out: a reviewer's ignore window, or a
    reach a person excluded in the ground-truth tool."""
    if isinstance(item, dict) and item.get("excluded_by_person") is not None:
        return GROUND_TRUTH
    return REVIEW


def _causal_frames(causal_id, reaches, left_out):
    """The frames of the causal reach, found by its id among the segment's
    reaches and its left-out reaches. Ids are renumbered when a reach list
    changes, so decisions are compared by FRAMES, never by id."""
    if causal_id is None:
        return None
    for r in list(reaches or []) + list(left_out or []):
        if isinstance(r, dict) and r.get("reach_id") == causal_id:
            w = _window(r)
            if w is not None:
                return w
    return ("reach id", causal_id)


def _decision(outcome, causal_id, reaches, left_out) -> dict:
    return {
        "outcome": outcome,
        "causal": _causal_frames(causal_id, reaches, left_out),
        "reaches": sorted(w for w in (_window(r) for r in reaches or []) if w),
        "left_out": sorted((w[0], w[1], _left_out_kind(x)) for x, w in
                           ((x, _window(x)) for x in left_out or []) if w),
    }


def decisions_of_features(doc) -> Dict[int, dict]:
    """{pellet: decision} of a features file (its ``segments``)."""
    # A filed reach that starts before the first cut was filed inside pellet 1
    # by the reach step's old rule; it belongs to no pellet (owner decision
    # D2-3), and the extractor now takes it out of pellet 1. That is not a
    # change of any pellet's decision, so it is not compared (none of them was
    # ever a causal reach).
    from mousereach.reach.core.before_first_cut import first_cut, starts_before
    cut = first_cut(doc)
    out = {}
    for g in (doc or {}).get("segments") or []:
        try:
            k = int(g.get("segment_num"))
        except (TypeError, ValueError):
            continue
        out[k] = _decision(g.get("outcome"), g.get("causal_reach_id"),
                           [r for r in g.get("reaches") or [] if not starts_before(r, cut)],
                           [r for r in g.get("reaches_left_out") or []
                            if not starts_before(r, cut)])
    return out


def decision_of_resolved(seg_data, outcome_data, measured, left_out) -> dict:
    """One pellet's decision as the extractor is about to write it, with who
    decided each part (the provenance the truth layering stamps)."""
    d = _decision(outcome_data.get("outcome"), outcome_data.get("causal_reach_id"),
                  measured, left_out)
    d["outcome_source"] = outcome_data.get("outcome_source") or "algo"
    d["causal_reach_source"] = outcome_data.get("causal_reach_source")
    d["reach_sources"] = {w: (r.get("reach_source") or "algo")
                          for r in list(seg_data.get("reaches") or [])
                          for w in [_window(r)] if w}
    # Whether ground truth replaced this pellet's whole reach list with one it
    # declares complete (truth_resolver._apply_gt_reaches stamps it). Only
    # then may ground truth explain a filed reach that is simply gone.
    d["reaches_complete_by"] = (GROUND_TRUTH if seg_data.get(
        "reach_list_complete_by_ground_truth") else None)
    return d


def _overlaps(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _fmt(w) -> str:
    if w is None:
        return "none"
    if isinstance(w, tuple) and len(w) == 2 and isinstance(w[0], int):
        return "frames %d-%d" % w
    return str(w)


def changes_not_owed(filed: Dict[int, dict], fresh: Dict[int, dict],
                     owed: Iterable[str] = ()) -> List[str]:
    """Each change from ``filed`` to ``fresh`` that no owed answer explains,
    one line per pellet, in words for a person. Empty: the run keeps every
    filed decision (or changes only what an owed answer decides).

    ``owed``: the kinds of a person's answer the filed kinematics have not
    applied (REVIEW, GROUND_TRUTH; owed_human_layers). A change counts as
    owed only when the fresh decision was made by such an answer:
      - an outcome, when its outcome_source is owed;
      - a causal reach, when its causal_reach_source or the pellet's
        outcome_source is owed, or when the pellet's reaches changed in an
        owed way (a reach replaced by a person's frames gets a new id);
      - a reach, when every reach that appears is a person's (reach_source
        owed), every reach that goes is replaced by one of those or left out
        by an owed statement, or -- ground truth owed -- the pellet's whole
        reach list now IS the ground truth's, which declares it complete
        (``reaches_complete_by``), and every change to the left-out reaches
        comes from an owed kind of statement.

    WHY a reach that just goes is not explained by owed ground truth alone:
    ground truth that covers only outcomes, or lists some reaches without
    declaring the list complete, removes no reach. A filed reach that is gone
    then went for another reason -- a changed reach rule in the extractor, a
    review copy the filing run did not see -- and filing that silently is
    exactly what this check exists to stop. Only ground truth that replaced
    the pellet's reaches with its complete list can have removed it.
    """
    owed = set(owed or ())
    problems: List[str] = []
    if set(filed) != set(fresh):
        problems.append("the pellets differ (filed %s, now %s)"
                        % (sorted(filed), sorted(fresh)))
    for k in sorted(set(filed) & set(fresh)):
        a, b = filed[k], fresh[k]
        words = []
        added = set(b["reaches"]) - set(a["reaches"])
        removed = set(a["reaches"]) - set(b["reaches"])
        lo_a = set(a["left_out"])
        lo_b = set(b["left_out"])
        lo_changed = lo_a ^ lo_b
        reaches_changed = bool(added or removed or lo_changed)
        reaches_owed = False
        if reaches_changed:
            srcs = b.get("reach_sources") or {}
            gt_complete = (GROUND_TRUTH in owed
                           and b.get("reaches_complete_by") == GROUND_TRUTH)
            newly_left_out = {(s, e) for s, e, kind in lo_b - lo_a if kind in owed}
            reaches_owed = (
                all(srcs.get(w) in owed for w in added)
                and all(kind in owed for _, _, kind in lo_changed)
                and all(any(_overlaps(w, x) for x in added) or w in newly_left_out
                        or gt_complete for w in removed))
            if not reaches_owed:
                first = (" (first: %s -> %s)" % (
                    _fmt(min(removed)) if removed else "-",
                    _fmt(min(added)) if added else "-")) if (removed or added) else ""
                words.append("reaches: %d filed gone, %d new by %s%s%s%s" % (
                    len(removed), len(added),
                    ", ".join(sorted({srcs.get(w, "?") for w in added})) or "-", first,
                    "; the reaches left out changed" if lo_changed else "",
                    "; the owed ground truth does not list this pellet's reaches as "
                    "complete, so it explains no filed reach that is gone without a "
                    "person's reach over its frames"
                    if removed and GROUND_TRUTH in owed and not gt_complete else ""))
        if a["outcome"] != b["outcome"] and b.get("outcome_source") not in owed:
            words.append("outcome %s -> %s (by %s)" % (
                a["outcome"], b["outcome"], b.get("outcome_source") or "algo"))
        if a["causal"] != b["causal"]:
            explained = (b.get("causal_reach_source") in owed
                         or b.get("outcome_source") in owed
                         or (reaches_changed and reaches_owed))
            if not explained:
                words.append("causal reach %s -> %s (by %s)" % (
                    _fmt(a["causal"]), _fmt(b["causal"]),
                    b.get("causal_reach_source") or "algo"))
        if words:
            problems.append("pellet %d: %s" % (k, "; ".join(words)))
    return problems


def hold_message(problems: List[str], owed: Iterable[str] = ()) -> str:
    """The one line a person reads for a held kinematics-only re-run."""
    owed = sorted(set(owed or ()))
    shown = problems[:_NAMED]
    more = len(problems) - len(shown)
    return ("this kinematics-only re-run of a filed video would change what was "
            "filed, and no answer the filed results have not applied yet asks for it "
            "(owed: %s). Not written; the filed results stand until a person decides. "
            "%s%s" % (", ".join(owed) or "none", "; ".join(shown),
                      (" ... and %d more pellet(s)" % more) if more > 0 else ""))


def owed_human_layers(video_id: str, filed_dir) -> frozenset:
    """The kinds of a person's answer this filed video's kinematics have NOT
    applied yet, by the version scan's own rules (the scan re-runs a filed
    video for exactly these): REVIEW when the review the scan finds
    (causal_review_io.review_path_index) is not the one the filed manifest
    records as applied (collage_provenance.review_applied); GROUND_TRUTH when
    the ground truth filed beside the video is not the one it records
    (manifest.ground_truth_applied).

    A filed manifest that cannot be read owes nothing: then any change holds
    the run, the cautious way. Never raises."""
    owed = set()
    filed_dir = Path(filed_dir)
    try:
        manifest = json.loads((filed_dir / f"{video_id}_processing_manifest.json")
                              .read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("%s: filed manifest unreadable (%s); no answer counts as owed",
                       video_id, e)
        return frozenset()
    try:
        features_mtime = (filed_dir / f"{video_id}_features.json").stat().st_mtime
    except OSError:
        features_mtime = None
    try:
        from mousereach.review.causal_review_io import review_path_index
        from mousereach.video_prep.core.collage_provenance import review_applied
        hit = review_path_index([video_id]).get(video_id)
        if hit is not None:
            path, mtime, size = hit
            if not review_applied(manifest, path, review_mtime=mtime, review_size=size,
                                  features_mtime=features_mtime):
                owed.add(REVIEW)
    except Exception as e:
        logger.warning("%s: could not tell whether its review is owed (%s)", video_id, e)
    try:
        from mousereach.pipeline.manifest import ground_truth_applied
        gt = filed_dir / f"{video_id}_unified_ground_truth.json"
        if gt.is_file() and not ground_truth_applied(
                manifest, gt, gt_mtime=gt.stat().st_mtime, features_mtime=features_mtime):
            owed.add(GROUND_TRUTH)
    except Exception as e:
        logger.warning("%s: could not tell whether its ground truth is owed (%s)",
                       video_id, e)
    return frozenset(owed)
