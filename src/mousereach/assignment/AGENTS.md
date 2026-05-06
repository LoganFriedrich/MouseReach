# assignment/ -- AGENTS.md

**This is CODE.** Reach-assignment algorithm (algo 4 of the 4-algo
decomposition). Joins reaches to segments and pellets, identifies the
causal reach for each touched segment's outcome.

See `four_algo_decomposition.md` in cross-session memory for the
broader pipeline context.

## Subdirectories

| Dir | Purpose |
|-----|---------|
| `v1/` | First (and current) version. Per-reach binary classifier (causal vs miss). Trained against GT causal labels via interaction_frame containment. |

DO NOT name new versions `ml/` or similar. Use versioned dirs (`v1`,
`v2`, ...) -- see `four_algo_decomposition.md` for the naming rule.

## Canonical evaluation: per-reach Sankey

The reach assignment algo is evaluated via the per-reach Sankey
documented in `algo4_assignment_evaluation_format.md` (cross-session
memory). This Sankey serves two roles:

1. **Algo 4-specific eval** -- did the assignment classifier correctly
   pick the causal reach in each touched segment?
2. **Pipeline-wide holistic view** -- because algo 4 is the final
   output, its per-reach Sankey integrates errors from algos 1-3
   (segmentation, reach detection, outcome). When the user asks "how
   is the pipeline doing overall," this is the figure to show.

DO NOT use this Sankey to answer narrow algo 1/2/3 questions. Each
algo has its own narrow evaluation -- see
`per_algo_evaluation_toolset.md`.

The Sankey is rendered by:

1. Build inputs:
   - Algo `_pellet_outcomes.json` per video with `outcome` (incl.
     `triaged`), `causal_reach_id`, `would_be_causal_reach_id` (for
     triaged segments).
   - Algo `_reaches.json` per video with reaches keyed by `reach_id`.
2. Call `mousereach.improvement.outcome.metrics.compute_per_reach_confusion`.
3. Write the result to `<snapshot>/metrics/scalars.json` under
   `outcome_label.confusion_matrix`.
4. Call `mousereach.improvement.outcome._run_notebooks.run_sankey`.

Reference runner: `scripts/restart_phase_d_per_reach_sankey.py`.

## Critical: abnormal_exception and triaged are causal-only

The labeling logic in `_label_reach_by_side` (in
`mousereach/improvement/outcome/metrics.py`) was fixed 2026-04-30 so
that:
- `abnormal_exception` labels ONLY the GT-causal reach (the one whose
  span contains GT's `interaction_frame`). Other reaches in those
  segments label as `miss`.
- `triaged` labels ONLY the algo's best-guess reach
  (`would_be_causal_reach_id`). Other reaches in triaged segments
  label as `miss`.

DO NOT regress this. The propagated-to-all-reaches version inflated
those bars by ~10x and hid the real algo behavior.

## Triage decision granularity

Currently triage is a per-segment decision: in a touched segment, if
`max(P(causal))` across reaches < threshold (default 0.40), the
segment is triaged. The algo's best-guess reach is recorded as
`would_be_causal_reach_id` so the per-reach Sankey can label it. Per-
reach triage is also defensible -- granularity is a design choice
under user direction.
