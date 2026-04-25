"""
MouseReach Improvement Process framework.

Provides structured snapshot tracking for algorithm improvement across
four phases: segmentation, reach_detection, outcome, and features.

Each phase produces versioned snapshots containing:
  - manifest.json: metadata, code hash, metrics summary
  - vault/: Obsidian vault with logic diagrams and notes
  - figures/: rendered diagrams, comparison plots
  - metrics/: quantitative evaluation data

Snapshots accumulate in MouseReach_Pipeline/Improvement_Snapshots/<phase>/
and are created/managed by code in this package.

Usage:
    from mousereach.improvement.lib.manifest import Manifest
    from mousereach.improvement.lib.snapshot_io import write_snapshot, read_snapshot
"""
