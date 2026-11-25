"""Retired: the old per-segment triage-clearing tool.

``mousereach-triage-clearing`` must not save the reviewer's call straight INTO
the algorithm's own output files (``_pellet_outcomes.json``, ``_reaches.json``)
and a ground-truth file. That would break the pipeline's rules in four ways:

* The answer was not a review record, so the next run of outcome detection
  or reach detection rewrote the files and the person's call was lost.
* It never wrote the reach-assignment file, so a touched pellet it "cleared"
  stayed open for the watcher's release check and never left the queue.
* It wrote the reaches file and then the outcomes file with no way back, so
  a failure between the two left them disagreeing.
* It took the most recently marked reach as the causal one instead of asking
  the reviewer which reach caused the outcome, and saved every reach the
  algorithm had seeded into the ground truth, checked or not.

The review window (``mousereach-review-tool``) does this job correctly: it
saves a review record the watcher reads, that survives re-runs, and releases
the video once every question is answered. The old tool's spot-check mode
(``--qc-count`` / ``--qc-report``) goes too: it walked the same retired
worklist and its "algorithm is wrong" verdict wrote a flag back into the
algorithm's outcomes file, which the next run overwrote.

WHY a stub instead of deleting the command: every machine that installed an
earlier version still has a ``mousereach-triage-clearing`` program pointing
here. Without this module it would crash with an import error; with it, it
says what to use instead and stops without touching anything.
"""
from __future__ import annotations

RETIRED_MESSAGE = (
    "mousereach-triage-clearing has been retired and does nothing.\n"
    "It saved answers into the algorithm's own files, where the next run\n"
    "erased them and the watcher never saw them.\n"
    "To review the triage queue, open the review window instead:\n"
    "    mousereach-review-tool\n"
    "(or the Review Queues tab in the main 'mousereach' window)."
)


def main(argv=None) -> int:
    """Print where the work moved and stop. Writes nothing, opens nothing.

    Any arguments the old tool took are accepted and ignored, so an old
    shortcut or script gets this message rather than a usage error.
    ``--help`` still prints a usage line, so the generated command
    reference shows the retirement instead of an error."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="mousereach-triage-clearing",
        description=RETIRED_MESSAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_known_args(argv)
    print(RETIRED_MESSAGE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
