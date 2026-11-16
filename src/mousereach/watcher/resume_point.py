"""Where a video returning from a review queue resumes.

THE RULE. A video a review queue returns resumes just after the step that sent
it there, and every later step runs -- the quality triage, the hand-score
comparison and the review gate included, because they come after every
algorithm step. Each held video records which step sent it (the watcher row's
``held_at_step``, and ``held_at_step`` in the queue bundle's manifest), written
when it is routed (review_gate.route_to_queue).

    step that sent it        what the person answers         resumes at
    -----------------        ------------------------         ----------
    segmentation             the cuts (re-segmentation /      reach
      (every route into        deep review): every output       (reach, outcome and
      deep review)             after the cuts is remade         assignment are remade)
    outcome                  a pellet's outcome or its        assignment
      (the gate's triage)      causal reach (triage)            (the outcomes are kept)
    assignment               the causal reach, when reach     checks
      (reach assignment        assignment could not run         (all four algorithm
      could not run)                                            outputs are kept)

"checks" means: quality triage, hand-score comparison, review gate, then
kinematics. The person's answers are applied from the review file by the gate
and the feature extractor; no algorithm reads them, so remaking the steps
before the one that sent the video would only redo work.

WHAT HAPPENS TO THE PERSON'S ANSWERS WHEN AN OUTPUT IS REMADE. On the SAME
cuts they persist, as designed: the review file stays in the processing folder;
the gate counts a segment it answers as answered, and the feature extractor
applies each answer to the segment whose frames it covers, on the remade reach,
outcome or assignment outputs. That is the case when the start moves earlier
to reach, outcome or assignment because an output is out of date (below).

KNOWN DEFECT, NOT THE DESIGN: answers are still carried across a re-cut. The
design rule is that re-segmentation voids everything after it, a person's
earlier outcome and causal-reach answers included: they are kept as a record
only and asked again, never carried by frames or by segment number. Two paths
here still re-cut and then apply the old answers by frames
(causal_review_io.index_review_by_segment): the deep-review return (resumes at
reach on the new cuts) and a triage return whose algorithm cuts are out of date
or unreadable (current_start moves its start back to segmentation, so the
segmenter cuts again). The voiding is owed by the review-answer-records work
(saving cuts voids later answers); this module only decides where a run starts
and does not touch the review file.

WHY deep review always resumes at reach: deep review is where cuts are fixed,
and everything computed from the cuts is remade on the new ones. Segmentation
itself is not run again for its own sake: the segmentation step keeps a
person's cuts as they are.

AN OUTPUT IS REUSED ONLY WHEN IT IS CURRENT. Before reusing the outputs of the
steps before the resume point, each one's recorded algorithm version is checked
against the versions declared now (pipeline_versions.json, with its
``compatible_versions``), and the pose the outputs were made from (the
processing manifest that travels with the bundle) against the pose the run
will use and the declared pose model. The run starts at the EARLIEST step
whose output is missing, out of date or unreadable; a pose that differs means
everything after the pose is remade (start at segmentation). When the
declaration or the processing manifest cannot be read, nothing is reused
(start at segmentation): an output that cannot be shown current is not
trusted. A person's cuts (boundary_source 'human') are exempt from the
segmenter's version, as everywhere else (the version scan exempts them too).
The placeholder reach-assignment file a hold writes when reach assignment
could not run (it carries ``assignment_failed`` and no version) is kept when
the run resumes at the checks: the person's causal-reach answers replace it.

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# The steps a run can start at, in order. 'checks' starts after every algorithm
# step: quality triage, hand-score comparison, review gate, kinematics.
RUN_STAGES = ("segmentation", "reach", "outcome", "assignment", "checks")

# Step that sent the video -> where its run resumes.
RESUME_AFTER = {
    "segmentation": "reach",
    "reach": "outcome",
    "outcome": "assignment",
    "assignment": "checks",
}

# The step each review queue stands for, when the routing named none: deep
# review is where cuts are fixed; triage asks about pellet outcomes and causal
# reaches.
QUEUE_STEP = {"deep_review": "segmentation", "triage": "outcome"}

# Each algorithm step's output file, the version key inside it, and its name in
# pipeline_versions.json (pipeline.manifest.TRACKED_STAGES).
_STEP_OUTPUT = (
    ("segmentation", "_segments.json", "segmenter_version", "segmenter"),
    ("reach", "_reaches.json", "detector_version", "reach_detector"),
    ("outcome", "_pellet_outcomes.json", "detector_version", "outcome_detector"),
    ("assignment", "_reach_assignments.json", "version", "assignment"),
)


def resume_stage_for(held_at_step: Optional[str], queue: Optional[str]) -> str:
    """Where a run resumes after a review queue returns the video.

    ``held_at_step`` is the step recorded when the video was routed (None when
    the routing predates the record); ``queue`` is 'deep_review' or 'triage'.
    A deep-review return always resumes at reach (module docstring). An
    unknown step resumes after the step its queue stands for; with nothing
    known at all, at segmentation (every step runs)."""
    if queue == "deep_review":
        return RESUME_AFTER["segmentation"]
    step = held_at_step if held_at_step in RESUME_AFTER else QUEUE_STEP.get(queue or "")
    return RESUME_AFTER.get(step, "segmentation")


def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def current_start(processing_dir, video_id: str, wanted: str, declared: dict,
                  pose_scorer: str) -> Tuple[str, str]:
    """``(start, why)``: where the run really starts, given that it wants to
    start at ``wanted`` (one of RUN_STAGES) after a review return.

    ``declared`` is pipeline_versions.json as read (get_current_versions);
    ``pose_scorer`` is the scorer of the pose file the run will use. ``why``
    says, for the log, why the start moved earlier ('' when it did not)."""
    if wanted not in RUN_STAGES:
        return "segmentation", f"unknown resume step {wanted!r}"
    if wanted == "segmentation":
        return "segmentation", ""
    folder = Path(processing_dir)
    versions = (declared or {}).get("versions") or {}
    if not versions:
        return "segmentation", "the declared versions could not be read"
    compat = (declared or {}).get("compatible_versions") or {}

    # The pose. Outputs made from another pose than the one this run uses, or
    # than the model declared now, are all out of date.
    manifest = _read_json(folder / f"{video_id}_processing_manifest.json")
    made_from = ((manifest or {}).get("dlc_model") or {}).get("dlc_scorer") or ""
    if not made_from:
        return "segmentation", "no processing manifest says which pose the outputs were made from"
    if pose_scorer and made_from != pose_scorer:
        return "segmentation", f"outputs made from pose {made_from}, this run uses {pose_scorer}"
    declared_pose = versions.get("dlc_scorer") or ""
    if declared_pose and made_from != declared_pose:
        return "segmentation", f"outputs made from pose {made_from}, declared is {declared_pose}"

    # Imported here: review_gate imports this module inside its functions.
    from .review_gate import ASSIGNMENT_FAILED_KEY

    seg_doc = _read_json(folder / f"{video_id}_segments.json") or {}
    human_cuts = seg_doc.get("boundary_source") == "human"
    for step, suffix, key, component in _STEP_OUTPUT:
        if RUN_STAGES.index(step) >= RUN_STAGES.index(wanted):
            break
        doc = _read_json(folder / f"{video_id}{suffix}")
        if doc is None:
            return step, f"{video_id}{suffix} is missing or unreadable"
        if step == "segmentation" and human_cuts:
            continue                         # a person's cuts: version-exempt
        if step == "assignment" and wanted == "checks" and ASSIGNMENT_FAILED_KEY in doc:
            # The placeholder the hold writes when reach assignment could not
            # run (orchestrator._hold_for_failed_assignment). It carries no
            # version, yet it IS the output the person answered: their
            # causal-reach answers in the review file replace it. Remaking it
            # would re-run the very step that sent the video (the rule above).
            continue
        want_v = versions.get(component) or ""
        if not want_v:
            continue                         # nothing declared: nothing to enforce
        have_v = str(doc.get(key) or (doc.get("diagnostics") or {}).get(key) or "")
        if have_v != str(want_v) and have_v not in {str(v) for v in (compat.get(component) or [])}:
            return step, f"{component} output is version {have_v or 'unrecorded'}, declared {want_v}"
    return wanted, ""


def restored_hold_start(processing_dir, video_id: str, declared: dict, pose_scorer: str,
                        code_versions: dict) -> Tuple[str, str]:
    """``(start, why)`` for a video held for its hand scores whose hold was
    restored after a watcher restart: 'checks' when every algorithm output it
    has may be reused, else the first step whose output may not.

    An output is reused only when it passes current_start (it exists, was
    made from the pose this run uses and the pose model declared now, and
    carries the declared version or a compatible one) AND carries the version
    the RUNNING code stamps (``code_versions``, pipeline.versions
    .code_stamped_versions) or a version declared compatible with it. WHY both: the held video was analysed by an
    earlier process, and a restart is how new code is deployed. Reused
    unguarded, outputs of the old algorithms would go on to kinematics and
    filing; the declaration alone can lag behind a deploy. A person's cuts
    are exempt from the segmenter's version, as everywhere else."""
    start, why = current_start(processing_dir, video_id, "checks", declared, pose_scorer)
    if start == "segmentation":
        return start, why
    folder = Path(processing_dir)
    compat = (declared or {}).get("compatible_versions") or {}
    from .review_gate import ASSIGNMENT_FAILED_KEY
    seg_doc = _read_json(folder / f"{video_id}_segments.json") or {}
    human_cuts = seg_doc.get("boundary_source") == "human"
    for step, suffix, key, component in _STEP_OUTPUT:
        if RUN_STAGES.index(step) >= RUN_STAGES.index(start):
            break
        doc = _read_json(folder / f"{video_id}{suffix}")
        if doc is None:
            return step, f"{video_id}{suffix} is missing or unreadable"
        if step == "segmentation" and human_cuts:
            continue
        if step == "assignment" and ASSIGNMENT_FAILED_KEY in doc:
            continue
        want_v = str((code_versions or {}).get(component) or "")
        if not want_v:
            continue                         # this code stamps no version: nothing to enforce
        have_v = str(doc.get(key) or (doc.get("diagnostics") or {}).get(key) or "")
        # A version declared compatible counts as current here too, as in
        # current_start. WHY: declaring the old version compatible is how a
        # deploy says its outputs stay valid; re-running them anyway would
        # throw away work the lab chose to keep.
        if have_v != want_v and have_v not in {str(v) for v in (compat.get(component) or [])}:
            return step, (f"{component} output is version {have_v or 'unrecorded'}, "
                          f"the running code makes {want_v}")
    return start, why
