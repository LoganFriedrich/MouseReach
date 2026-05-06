# Boundary Accuracy Summary Table

## What question this answers

What fraction of algorithm boundaries land exactly on GT, within 1 frame, within 5,
within 10, or beyond 10 -- and how many boundaries are phantom (no GT match) or
missed (GT with no algo match)?

## Column definitions (NON-overlapping buckets)

Percentage columns partition ALL boundaries into exhaustive, mutually exclusive buckets
that sum to 100%.

| Column | Definition |
|--------|------------|
| N_GT | Number of ground-truth boundaries |
| N_algo | Number of algorithm-emitted boundaries |
| delta=0 (%) | Matched with exactly 0 frame error |
| abs(delta)=1 (%) | Matched with exactly 1 frame error |
| 2-5 (%) | Matched with 2-5 frames error |
| 6-10 (%) | Matched with 6-10 frames error |
| >10 (%) | Matched with >10 frames error |
| miss (%) | GT boundaries with no algo match |
| phantom (%) | Algo boundaries with no GT match |
| median abs(delta) | Median absolute error (matched only) |
| mean signed delta | Mean signed error (matched only) |

## Rendering params

- SNAPSHOT_DIR: `Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\segmentation\seg_v2.1.3_phantom_first_post_validation`
- FIGSIZE: (16, 4)
- DPI: 300
