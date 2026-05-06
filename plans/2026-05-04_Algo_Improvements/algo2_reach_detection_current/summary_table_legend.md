# Reach Detection Accuracy Summary Table

## What question this answers

What fraction of reach boundaries land exactly on GT, within 1 frame, within 5,
within 10, or beyond 10 -- and how many reaches are FP or FN?

## Layout

- **Top section**: overall counts (n_videos, n_gt_total, n_algo_total, n_matched, n_perfect_videos)
- **Bottom section**: two rows (Start delta, End delta) with non-overlapping accuracy buckets

## Column definitions (NON-overlapping buckets)

| Column | Definition |
|--------|------------|
| delta=0 (%) | Matched with exactly 0 frame error |
| |delta|=1 (%) | Matched with exactly 1 frame error |
| 2-5 (%) | Matched with 2-5 frames error |
| 6-10 (%) | Matched with 6-10 frames error |
| FP | False positives (algo reaches with no GT match) |
| FN | False negatives (GT reaches with no algo match) |
| med|d| | Median absolute error (matched only) |
| mean d | Mean signed error (matched only) |

## Rendering params

- SNAPSHOT_DIR: `Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\reach_detection\reach_v7.2.0_head_consistency`
- FIGSIZE: (16, 5)
- DPI: 300
