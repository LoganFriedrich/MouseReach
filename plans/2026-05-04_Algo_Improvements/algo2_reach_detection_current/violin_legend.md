# Reach Boundary Delta -- Violin Plot

## What question this answers

For every algorithm-detected reach that was matched to a ground-truth reach,
how far off were the start and end frames -- and in which direction?

- **start_delta** = `algo_start - gt_start`
- **end_delta** = `algo_end - gt_end`

## What improvement looks like

- Mean and median converge toward 0.
- Distributions get tighter.
- FP and FN counts drop.

## Rendering params

- SNAPSHOT_DIR: `Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\reach_detection\reach_v7.2.0_head_consistency`
- FIGSIZE: (12, 5)
- DPI: 300

## Data summary

- Matched reaches: 2683
- FP (phantom): 420
- FN (miss): 20
