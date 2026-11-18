"""Triage handling: GT auto-resolve and human-review tooling.

When the outcome detector flags a segment for triage (low confidence,
ambiguous physics, etc.), one of two things resolves it:

  1. **GT auto-resolve** (this module's :mod:`gt_resolve`): if a
     unified GT JSON exists for the video, lift the triage flag and
     stamp the GT outcome / interaction frame / causal reach. This is
     the production-pipeline fast path — for any video that has been
     ground-truthed, the human work is already done and we just import
     it. Runs as a step in the normal processing pipeline.

  2. **Human review** (the napari ``mousereach-review-tool``, the
     review window in :mod:`mousereach.review.causal_review_widget`):
     for segments where no GT is available, a reviewer answers each
     triaged segment and the answer is saved as a review record that
     the watcher reads. (The older triage-clearing tool, which wrote
     ``triage_cleared=True`` into the algorithm files, is retired.)

After either path, the segment's data flows into normal kinematic
analysis just like any other committed segment.

The two-level evaluation framework reports metrics both pre-resolution
(algo alone) and post-resolution (algo + GT auto-resolve), so we can
see what production-realistic accuracy looks like vs the algo's
in-isolation performance.
"""
