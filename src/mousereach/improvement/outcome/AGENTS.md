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
1. Build the per-reach `confusion_matrix` via `compute_per_reach_confusion`
   in `metrics.py` -- DO NOT replicate the labeling logic anywhere else.
2. Write `<snapshot>/metrics/scalars.json` with the same structure
   (`{"outcome_label": {"confusion_matrix": {...}}}`) but using per-reach
   counts and the extended class set.
3. Call `run_sankey(snapshot_dir)`.

Full class set for per-reach Sankey: `retrieved`, `displaced_sa`,
`displaced_outside`, `untouched`, `uncertain`, `miss`, `absent`,
`abnormal_exception`, `triaged`.

### Critical labeling rules (set in `_label_reach_by_side` 2026-04-30)

**`abnormal_exception` is causal-only.** The GT-causal reach in an
abnormal_exception segment (the reach whose [start, end] contains
GT's `interaction_frame`) gets the abnormal_exception label. Other
reaches in those segments label as `miss`. On the algo side, only the
reach matching `causal_reach_id` gets the label. This is the standard
-- DO NOT propagate to all reaches in those segments.

**`triaged` is would-be-causal-only.** When the algo flags a segment
for manual review (e.g., max v1 assignment probability < threshold),
only the algo's best-guess reach (`would_be_causal_reach_id`) gets
the triaged label on the algo side. Other reaches default to `miss`.
GT side never sees triaged.

### This IS the algo 4 evaluation format

For evaluating the reach assignment algorithm (algo 4 of the
4-algo decomposition -- see `four_algo_decomposition.md`), the
per-reach Sankey above IS the canonical format. Reference runner:
`scripts/restart_phase_d_per_reach_sankey.py`. The evaluation MUST
include both `absent` categories AND `triaged` -- skipping either
hides real algo failure modes.

## Metrics computation

The metrics module is [`metrics.py`](metrics.py). It computes per-segment
accuracy and writes `scalars.json` + `outcome_per_segment.csv` to a
snapshot's metrics dir.

For per-reach metrics, see `metrics.py` for matching functions and
extend rather than reinvent.

## Reach matching (HOW per-reach pairs are formed)

Per-reach metrics match algo reaches to GT reaches by **start-frame
proximity within a +/- 10-frame window**, greedy nearest-first
assignment. See `match_reaches` in
`mousereach/improvement/reach_detection/metrics.py`. Reach index is NOT
used. End-frame is NOT used. A real reach the algo detected ~12 frames
late produces TWO entries in the per-reach confusion matrix: a GT-side
`fn` (algo label = `absent`) AND an algo-side `fp` (GT label =
`absent`). That double-count is a known cost of start-frame matching --
mind it when building training labels for triage classifiers.

## Triage design principles (user direction, 2026-04-30)

Triage = flag a reach as "data too unreliable to trust the algo's
outcome." Purpose, in order of importance:
1. Protect kinematic analysis from contaminated retrieved/displaced
   reaches.
2. Surface a manual-review queue.

Two acceptable strategies:
- **Path A: ML triage.** Decision tree / random forest. Positive class
  = GT=abnormal_exception reaches. Negative class = matched-GT
  retrieved/displaced reaches. **Acceptable over-triage budget:
  1-3% extra.** Better to manually review a few real reaches than to
  admit one abnormal into kinematics. Bias loss toward
  precision-on-negatives.
- **Path B: skip triage, accept the limit.** No triage system; ~5%
  error rate on contacted-reach outcomes; all results become
  qualitative trend assessment.

Decision criterion: Path A only if held-out FP rate on the negative
class is <3%. Otherwise fall back to Path B.

Hard "do not" rules:
- **No global post-hoc filters** that re-classify segments after the
  algo has committed (step 9 cross-segment artifact attempt over-fired
  catastrophically -- 186 false reclassifications).
- **No single-signal triage rules.** Need multiple independent
  data-quality features that all agree.
- **Per-REACH triage**, not per-segment, because that's the unit
  kinematics consume.

See also `~/.claude/projects/y--2-Connectome/memory/triage_design_principles.md`.
