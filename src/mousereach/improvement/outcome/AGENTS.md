# improvement/outcome/ -- AGENTS.md

**CODE.** Outcome-classification phase of the MouseReach Improvement Process.

## Figures: USE `_run_notebooks.py` -- do NOT hand-roll

Canonical figure runner: [`_run_notebooks.py`](_run_notebooks.py)

Available functions:
- `run_sankey(snapshot_dir)` -- confusion-matrix Sankey from `scalars.json`
- `run_interaction_frame_violin(snapshot_dir)` -- per-class interaction-frame delta violin
- `run_summary_table(snapshot_dir)` -- formatted accuracy/precision/recall table

These runners use:
- `mousereach.improvement.lib.palette` -- shared colors and class order
- `<snapshot>/metrics/scalars.json` -- input metric data (must contain
  `outcome_label.confusion_matrix` for sankey, `interaction_frame.delta_histogram`
  for violin, etc.)

Output goes to `<snapshot>/figures/`.

## Per-reach Sankey (v4.0.0+)

When per-reach (instead of per-segment) confusion data is needed:
1. Build the per-reach `confusion_matrix` (e.g. via reach matching +
   per-reach outcome attribution -- see compute_reach_confusion in this dir
   if it exists, else compute inline).
2. Write `<snapshot>/metrics/scalars.json` with the same structure
   (`{"outcome_label": {"confusion_matrix": {...}}}`) but using per-reach
   counts and the extended class set (`miss`, `absent` in addition to
   the segment-level outcome classes).
3. Call `run_sankey(snapshot_dir)`.

Class set for per-reach Sankey: `retrieved`, `displaced_sa`,
`displaced_outside`, `untouched`, `uncertain`, `miss`, `absent`.

## Metrics computation

The metrics module is [`metrics.py`](metrics.py). It computes per-segment
accuracy and writes `scalars.json` + `outcome_per_segment.csv` to a
snapshot's metrics dir.

For per-reach metrics, see `metrics.py` for matching functions and
extend rather than reinvent.
