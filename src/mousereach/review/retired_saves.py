"""The whole-video Review Tool's save into the algorithm's output files.

The old whole-video Review Tool (``GroundTruthWidget`` in review mode: the
"3 - Review Tool" launcher tab, ``mousereach --reviews`` / ``--step 2b/3b/4b``
and ``mousereach-review-legacy``) saved a person's corrections straight into
``{video}_segments.json``, ``{video}_reaches.json`` and
``{video}_pellet_outcomes.json``.

WHY that save is retired rather than repaired: the pipeline does not keep a
correction made that way. The next run of a step writes its file again from
the algorithm and the correction is gone (corrected cuts were not marked
``boundary_source: "human"``, which is the only mark the segmenter keeps);
nothing after the corrected step was redone, so the later files still
described the old cuts or reaches; and the watcher's release check never
reads these marks, so a video "fixed" this way never left its review queue.
Each kind of correction has one tool whose save the pipeline honours:

* segment cuts -> the Segmentation Fixer (``mousereach-fix-segmentation``,
  or Review Queues -> Open Re-segmentation), which marks the cuts as
  a person's and clears what was computed on the old cuts;
* a pellet's outcome or causal reach -> the review window
  (``mousereach-review-tool``, or Review Queues -> Open Triage Review), which
  saves a review record that survives re-runs;
* reach start/end frames -> the Ground Truth tool (``mousereach-gt``), whose
  file the results step layers over the algorithm's.

NOT retired here: the separate per-step screens (boundary review
``mousereach-segment-review``, reach review ``mousereach-review-reaches``,
outcome review ``mousereach-review-pellet-outcomes`` and their napari menu
entries). They still save into the step's own file and mark it
``validation_status: "validated"``, which is the only sign-off a person
running the steps by hand, without the watcher, has for a step that triage
marked ``needs_review``. Whether to retire them, and what would replace that
sign-off, is an open design question.
"""
from __future__ import annotations

CUTS = "segment cuts"
REACHES = "reach frames"
OUTCOMES = "pellet outcomes and causal reaches"

_WHERE = {
    CUTS: ("the Segmentation Fixer: mousereach-fix-segmentation, "
           "or Review Queues -> Open Re-segmentation"),
    REACHES: "the Ground Truth tool: mousereach-gt",
    OUTCOMES: ("the review window: mousereach-review-tool, "
               "or Review Queues -> Open Triage Review"),
}


def refusal_message(kinds) -> str:
    """The text shown when a retired save is pressed. ASCII only: it is also
    printed to the console."""
    if isinstance(kinds, str):
        kinds = [kinds]
    lines = ["Not saved. This screen used to write corrections into the "
             "algorithm's own files, where the next run erased them.",
             "Make the correction with:"]
    for k in kinds:
        lines.append("  - %s: %s" % (k, _WHERE[k]))
    return "\n".join(lines)


def refuse_algo_file_save(kinds) -> bool:
    """Tell the person the save is retired and where to go. Writes nothing.

    Returns False so a caller that checks the result knows nothing was saved.
    """
    msg = refusal_message(kinds)
    print(msg)
    try:
        from napari.utils.notifications import show_warning
        show_warning(msg)
    except Exception:
        pass
    return False
