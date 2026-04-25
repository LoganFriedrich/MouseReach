"""
Segmentation phase of the MouseReach Improvement Process.

Tracks improvements to the pellet-boundary segmenter across versions.
Snapshots capture the algorithm's logic diagram, evaluation metrics on the
GT corpus, and per-boundary accuracy data.

Code being diagrammed:
  src/mousereach/segmentation/core/segmenter_robust.py    (v2.1.x, library)
  src/mousereach/segmentation/core/segmenter_multi.py     (v2.2.0+)
  src/mousereach/segmentation/core/proposers.py           (v2.2.0+)
  src/mousereach/segmentation/core/consensus.py           (v2.2.0+)
"""
