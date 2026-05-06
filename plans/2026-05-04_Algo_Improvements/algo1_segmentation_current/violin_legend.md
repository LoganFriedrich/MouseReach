# Boundary Signed Delta -- Violin Plot

## What question this answers

For every algorithm-emitted boundary that was matched to a ground-truth boundary,
how far off was it -- and in which direction? The **signed delta** is
`algo_frame - gt_frame`, so negative = algorithm fires early, positive = late.

## What improvement looks like

- Mean and median converge toward 0 (no systematic bias).
- The distribution gets tighter (fewer frames of error).
- Fewer outliers in the tails.
- Phantom and miss counts drop to 0.

## Red-flag patterns

- **Mean shift away from 0**: systematic bias (algorithm consistently early or late).
- **Long tails**: outlier boundaries that are very far from GT.
- **Many phantom/miss**: fundamental segment-count error, not just boundary placement.

## Rendering params

- SNAPSHOT_DIR: `Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\segmentation\seg_v2.3.0_endpoint_phase_offsets`
- FIGSIZE: (10, 6)
- DPI: 300
- Subsets: all, inter_pellet_B2_B20, endpoint_B1_B21

## Data summary

- **All boundaries**: 986 matched, 1 phantom, 1 miss
- **Inter-pellet (B2–B20)**: 893 matched, 0 phantom, 0 miss
- **Endpoints (B1 + B21)**: 93 matched, 0 phantom, 1 miss
