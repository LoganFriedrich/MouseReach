# Interaction Frame Delta -- Violin Plot

## What question this answers

For segments where both GT and algorithm identified a touched outcome
(retrieved, displaced_sa, displaced_outside), how close is the algorithm's
interaction_frame to the GT's interaction_frame? The signed delta is
`algo_interaction_frame - gt_interaction_frame`, so negative = algorithm
detects interaction earlier, positive = later.

## What improvement looks like

- Mean and median converge toward 0 (no systematic bias).
- The distribution gets tighter (fewer frames of error).
- Fewer outliers in the tails.

## Red-flag patterns

- **Mean shift away from 0**: systematic timing bias.
- **Long tails**: outlier segments with very wrong interaction timing.
- **Bimodal distribution**: two populations of errors (different failure modes).

## Exhaustive flag

NOTE: Videos not marked exhaustive=True have potentially inflated label_correct_wrong_reach counts (algo may catch real reaches GT did not label).

## Rendering params

- SNAPSHOT_DIR: `Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\outcome\v6_cascade_2026-05-04`
- FIGSIZE: (10, 4)
- DPI: 300

## Data summary

- Eligible segments: 455
- Median |delta|: 2 frames
- Mean signed delta: -19.9 frames
- Exact match (delta=0): 59 (13%)
- Within 5 frames: 394 (87%)
