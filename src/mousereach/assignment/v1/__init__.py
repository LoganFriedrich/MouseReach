"""
Reach assignment v1.

Per-reach binary classifier predicting causal-vs-miss within touched
segments. Trained per LOOCV fold against GT causal labels (the reach
whose [start_frame, end_frame] contains GT's interaction_frame for
the segment).

At inference time:
  - For each touched segment, score every reach.
  - Pick the reach with the highest causal probability as the segment's
    causal reach. Other reaches are tagged as misses.

For untouched segments, all reaches are tagged not-on-pellet (no
causal classification needed).
"""
VERSION = "1.0.0_dev"
