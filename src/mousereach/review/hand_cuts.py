"""Saving a person's segment cuts: the one way every cut editor writes them.

WHY THIS EXISTS
---------------
A video's cuts decide which frames belong to which pellet, so everything after
segmentation (reaches, outcomes, reach assignment, kinematics) is computed on
them. When a PERSON sets the cuts, three things must hold, or the correction is
lost or poisons the data:

1. The cuts are marked as a person's: ``boundary_source: "human"`` with
   ``corrected_at`` (when) and ``corrected_by`` (who). The segmenter keeps a
   file with that mark instead of re-cutting it (segmentation/core/batch.py),
   supersede and re-runs keep it (archive/supersede.is_human_cuts), and
   ``corrected_at`` is when the cuts were set
   (causal_review_io.cuts_set_at), so every answer a person gave on the OLD
   cuts is void and asked again (owner ruling: a re-cut voids everything after
   segmentation; nothing is carried across it).
2. Everything computed from the old cuts is archived and removed
   (``invalidate_stale_downstream``), so the pipeline recomputes it from the
   new cuts instead of describing frames that no longer belong together.
3. The file is replaced in one step (a temporary file, then ``os.replace``),
   so a crash or a full disk never leaves half a segments file.

The segmentation fixer (review/fix_segmentation_widget.py) has always saved
this way. The per-step boundary review screen ("Step 2b - Review Boundaries",
``mousereach-segment-review``, the Boundaries tab of "Review Tool (Tabbed)")
used to rewrite the segments file in place WITHOUT the mark: the next segmenter
run replaced the person's cuts, answers given on the old cuts were not voided,
and nothing after segmentation was recomputed. Both now save through here.

WHERE A CUT EDITOR MAY NOT SAVE (``save_refusal``)
--------------------------------------------------
The boundary screen opens any folder, so it asks first:

* a folder inside the filed tree, or a video locked for a publication
  (crystallized): a filed video has passed every check. The segmentation
  fixer works only on the re-segmentation queue and has no rule for changing a
  filed video's cuts, so nothing is written there (how a filed video's cuts
  should be corrected is an open question for the lab, not something a screen
  may decide);
* a review-queue folder other than the re-segmentation (deep review) queue,
  i.e. triage: a re-cut belongs in deep review (deep review beats triage). The
  review window's "Escalate: bad segmentation -> re-segmentation" sends the
  video there, where its cuts are set;
* a folder that is no longer there (the bundle left its queue while open):
  writing would recreate it holding only these cuts.

Qt-free so every rule here is testable without a window.
ASCII-only console output (Windows consoles cannot print Unicode).
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Outputs computed from a video's cuts. Every one describes the OLD cuts once
# the cuts change.
DOWNSTREAM_SUFFIXES = ("_reaches.json", "_pellet_outcomes.json",
                       "_reach_assignments.json", "_features.json")


# ---------------------------------------------------------------------------
# Every boundary is a person's explicit call (owner decision MRR-08d,
# 2026-11-19: "Force a human to at least look at every single boundary ...
# can't save and can't send the video out of resegmentation until they have
# explicitly accepted that a boundary is already correctly placed or have
# corrected it.")
# ---------------------------------------------------------------------------

# The record, in the saved segments file, of what a person did with each cut.
# One entry per cut, keyed by the cut's FRAME (never by its position in the
# list, which shifts when a cut is moved past another, removed or added --
# the old "guided_walk" record was keyed by position, so a later answer
# overwrote an earlier one and a walk could not be checked):
#   {"frame": <the cut's frame now>, "action": "accepted" | "moved" | "added"
#    | "removed", "shown_frame": <frame the person was shown, for moved and
#    accepted>, "by": <user>, "at": <ISO time>}
BOUNDARY_CALLS_KEY = "boundary_calls"
# The actions that make a cut a person's call. "removed" is a call too, but
# about a cut that is no longer there.
CALLED_ACTIONS = ("accepted", "moved", "added")

# The routing reason that sends a video back to the re-segmentation queue
# because cuts were saved without every boundary being explicitly accepted or
# moved. It names segmentation, so saving the cuts releases the video
# (release_after_hand_cuts); and the calls a person already made on the cuts
# it holds are kept for it (carried_calls), so only the remaining boundaries
# are walked.
RECUT_REMAINING_REASON = ("segmentation: boundaries not all explicitly "
                          "accepted/corrected")


def _cut_list(seg: dict) -> List[int]:
    out = []
    for b in (seg or {}).get("boundaries") or []:
        try:
            out.append(int(b.get("frame", b.get("index", 0))) if isinstance(b, dict) else int(b))
        except (TypeError, ValueError, AttributeError):
            continue
    return sorted(out)


def called_frames(seg: dict) -> set:
    """The cut frames a person explicitly accepted, moved to, or added, by
    the record in ``seg``. ``boundary_calls`` when present; else the older
    position-keyed ``guided_walk`` record read for what survived of it (an
    answer's ``corrected_frame``, else its ``algo_frame`` when confirmed --
    the frame the person was shown, so the shifting position does not matter;
    an answer a later one overwrote is lost, and its cut counts as not
    called), plus every cut that is not one of the algorithm's
    (``algo_boundaries``): the old fixer recorded no walk answer for a cut
    added with the manual controls, but a cut the algorithm never proposed
    can only have been placed by the person; else nothing."""
    seg = seg or {}
    calls = seg.get(BOUNDARY_CALLS_KEY)
    out = set()
    if isinstance(calls, list):
        for c in calls:
            if isinstance(c, dict) and c.get("action") in CALLED_ACTIONS:
                try:
                    out.add(int(c["frame"]))
                except (KeyError, TypeError, ValueError):
                    pass
        return out
    algo = seg.get("algo_boundaries")
    if isinstance(algo, list) and algo:
        algo_set = set(_cut_list({"boundaries": algo}))
        out |= {b for b in _cut_list(seg) if b not in algo_set}
    for r in seg.get("guided_walk") or []:
        if not isinstance(r, dict) or r.get("removed"):
            continue
        try:
            if r.get("corrected_frame") is not None:
                out.add(int(r["corrected_frame"]))
            elif r.get("confirmed"):
                out.add(int(r["algo_frame"]))
        except (KeyError, TypeError, ValueError):
            pass
    return out


def uncalled_boundaries(seg: dict) -> List[int]:
    """The cut frames of ``seg`` no person explicitly accepted, moved or
    added (called_frames). Empty means every boundary is a person's call."""
    called = called_frames(seg)
    return [b for b in _cut_list(seg) if b not in called]


def uncalled_reason(seg: dict) -> str:
    """Words for a person when ``seg`` has cuts nobody called, '' otherwise."""
    left = uncalled_boundaries(seg)
    if not left:
        return ""
    n = len(_cut_list(seg))
    shown = ", ".join(str(f) for f in left[:10]) + (" ..." if len(left) > 10 else "")
    return ("%d of %d boundaries were saved without being explicitly accepted or "
            "moved (frames %s)" % (len(left), n, shown))


class HandCutsNotSaved(RuntimeError):
    """The cuts were not saved; ``stage`` says which step refused
    ('archive' or 'invalidate') and the message says why."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def invalidate_stale_downstream(seg_path, video_stem, archive_dir):
    """Archive-then-remove every output computed from boundaries that are
    about to be replaced.

    A boundary change voids reaches, outcomes, assignments and features --
    they all describe the OLD cuts. Leaving them in place re-circulated
    corrected videos to triage with data for different spans, and risked a
    causal reach attributed to the wrong pellet. Removing them
    also forces every stage to recompute even under a reuse-scoped rerun.

    Raises on failure -- the caller must refuse the save rather than leave a
    half-invalidated video. Returns the list of archived+removed file names.
    """
    removed = []
    for suffix in DOWNSTREAM_SUFFIXES:
        stale = Path(seg_path).parent / (video_stem + suffix)
        if stale.exists():
            # The folder is made here, only when something goes into it, so a
            # save that has nothing to archive leaves no empty archive folder.
            Path(archive_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy2(stale, Path(archive_dir) / stale.name)
            stale.unlink()
            removed.append(stale.name)
    return removed


def archive_dir_for(seg_path) -> Path:
    """The dated folder a save copies the replaced files into.

    The segmentation fixer's place: ``<Processing>/_archived/
    segmentation_before_human_fix_<time>``, where ``<Processing>`` is the
    shared Processing folder that holds the review queues. WHY the same
    place for every editor: one place to look for what a person's cuts
    replaced, and no new folder kind in the lab's tree. A machine with no
    review queues configured (steps run by hand) keeps it beside the segments
    file instead. "_archived" folders are never read as live outputs
    (pipeline/analyzed_tree.SUPERSEDED_DIR_NAMES).
    """
    stamp = "segmentation_before_human_fix_%s" % datetime.now().strftime("%Y%m%d_%H%M%S")
    root = None
    try:
        from mousereach.config import Paths
        review_root = getattr(Paths, "REVIEW_ROOT", None)
        if review_root:
            root = Path(review_root).parent / "_archived"
    except Exception:
        root = None
    if root is None:
        root = Path(seg_path).parent / "_archived"
    return root / stamp


def _inside(folder: Path, root) -> bool:
    if not root:
        return False
    try:
        f, r = Path(folder).resolve(), Path(root).resolve()
    except Exception:
        return False
    return f == r or r in f.parents


def save_refusal(folder, video_stem: str) -> Optional[str]:
    """Why a person's cuts may NOT be saved into ``folder`` from a cut editor
    that can open any folder, or None when they may (module docstring says
    why each rule exists). The text is shown to the person as it is."""
    folder = Path(folder)
    if not folder.is_dir():
        return ("%s is no longer there (the watcher or another window moved this "
                "video while it was open here). Nothing was saved. Load the video "
                "again from where it is now." % folder)
    from mousereach.pipeline.analyzed_tree import filed_tree_refusal
    if filed_tree_refusal(folder):
        return ("%s is filed: it passed every check and its files are the record "
                "of that run. Its cuts are not changed from this screen. Nothing "
                "was saved." % video_stem)
    from mousereach.pipeline.publication_lock import lock_refusal
    try:
        locked = lock_refusal(video_stem, "saving cuts")
    except Exception as e:
        locked = ("whether %s is locked for a publication could not be checked "
                  "(%s); not saving cuts." % (video_stem, e))
    if locked:
        return locked + " Nothing was saved."
    try:
        from mousereach.config import Paths
        review_root = getattr(Paths, "REVIEW_ROOT", None)
        deep = getattr(Paths, "DEEP_REVIEW", None)
    except Exception:
        review_root = deep = None
    if _inside(folder, review_root) and not _inside(folder, deep):
        return ("%s is in a review queue other than re-segmentation. Cuts are "
                "set in re-segmentation: in the review window press 'Escalate: "
                "bad segmentation -> re-segmentation', then set the cuts there. "
                "Nothing was saved." % video_stem)
    return None


def mark_as_hand_cuts(seg: dict, boundaries, algo_boundaries, corrected_by: str) -> dict:
    """``seg`` with the fields that make its cuts a person's (module docstring,
    point 1). The reason the video was sent for its cuts is answered now
    (``needs_human`` moves to ``needs_human_resolved``): leaving it set would
    send the video straight back on the next pass."""
    seg = dict(seg or {})
    seg["needs_human_resolved"] = list(seg.get("needs_human") or [])
    seg["boundaries"] = [int(b) for b in boundaries]
    seg["algo_boundaries"] = [int(b) for b in algo_boundaries]
    seg["boundary_source"] = "human"
    seg["corrected_by"] = corrected_by
    seg["corrected_at"] = datetime.now().isoformat()
    seg["needs_human"] = []
    return seg


def commit_hand_cuts(seg_path, video_stem: str, seg: dict, archive_dir) -> List[str]:
    """Write ``seg`` (already marked by ``mark_as_hand_cuts``) as the video's
    segments file, in the fixer's order: copy the file it replaces into
    ``archive_dir``, archive and remove everything computed from the old cuts,
    then replace the file in one step.

    Raises HandCutsNotSaved before the segments file is touched when the
    original cannot be archived or the old outputs cannot be removed.
    Returns the names of the outputs archived and removed."""
    seg_path = Path(seg_path)
    try:
        if seg_path.is_file():
            Path(archive_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy2(seg_path, Path(archive_dir) / seg_path.name)
    except Exception as e:
        raise HandCutsNotSaved("archive", "Refusing to overwrite the original: %s" % e)
    try:
        removed = invalidate_stale_downstream(seg_path, video_stem, archive_dir)
    except Exception as e:
        raise HandCutsNotSaved("invalidate", "The corrected cuts were NOT saved: %s" % e)
    tmp = seg_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(seg, indent=2))
    os.replace(tmp, seg_path)
    return removed


def asked_again_note(bundle: Path, stem: str) -> str:
    """Words for the status line when saving these cuts voided answers a
    person gave on the old cuts: those pellets are asked again in triage after
    the re-run (review_gate.recut_voided_open). '' when there are none or the
    check cannot run -- the gate makes the same check either way, so the note
    is information only."""
    try:
        from mousereach.watcher.review_gate import recut_voided_open
        n = len(recut_voided_open(stem, bundle))
    except Exception:
        return ""
    if not n:
        return ""
    return ("  %d pellet(s) answered on the old cuts will be asked again in triage "
            "after the re-run." % n)


def release_after_hand_cuts(bundle, stem: str) -> str:
    """Release a re-segmentation bundle whose cuts a person just saved, when
    that answers why it was queued. Returns words for the status line ('' when
    the folder is not a queued bundle).

    Saving hand cuts finishes the work the video was routed here FOR when the
    routing reason names segmentation -- but the clear marker is a blanket
    human-clear token the watcher gate honors, so writing it for a bundle
    routed on other grounds (e.g. a QC hold) would release concerns nobody
    addressed. Gate on the routing reason."""
    try:
        bundle = Path(bundle)
        marker = bundle / f"{stem}_deep_review_cleared.json"
        routing = bundle / f"{stem}_routing.json"
        if not routing.is_file() or marker.exists():
            return ""
        # Every boundary must be a person's explicit call before the video
        # leaves re-segmentation (owner decision MRR-08d, comment at
        # BOUNDARY_CALLS_KEY). The segmentation fixer refuses such a save; this
        # catches any other cut editor that saved into the queue.
        try:
            seg = json.loads((bundle / f"{stem}_segments.json").read_text(encoding="utf-8"))
        except Exception:
            seg = {}
        left = uncalled_reason(seg)
        if left:
            return ("  NOT released: %s. Open it in the segmentation fixer, accept "
                    "or move each remaining boundary and save." % left)
        from .release_cli import _names_segmentation, _routing_reason
        reason_txt = _routing_reason(bundle, stem)
        if not _names_segmentation(reason_txt):
            return ("  NOT auto-released: routed for '%s', which fixing "
                    "cuts does not address -- use Re-segmentation Review's "
                    "tool's Clear when that is resolved."
                    % (reason_txt or "an unrecorded reason"))
        # Create-only, never into a recreated folder
        # (causal_review_io.write_clear_marker_once says why).
        from .causal_review_io import (write_clear_marker_once,
                                       _get_username, _get_timestamp)
        if write_clear_marker_once(marker, {
            "type": "deep_review_cleared",
            "video_stem": stem,
            "cleared_by": _get_username(),
            "cleared_at": _get_timestamp(),
            "reason": "segmentation corrected by hand",
            "gated_on": "boundary_source",
            "routing_reason": reason_txt,
        }):
            return "  Released back to the pipeline." + asked_again_note(bundle, stem)
        return ("  Not released from here: it has "
                "already left the queue or been released.")
    except Exception as e:
        return "  (release check failed: %s)" % e
