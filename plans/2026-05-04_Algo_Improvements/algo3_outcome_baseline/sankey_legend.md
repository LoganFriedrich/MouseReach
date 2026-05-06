# Outcome Classification Flow -- Sankey Diagram

## What question this answers

How does the algorithm's outcome classification compare to ground truth?
Each flow shows how many segments with a given GT outcome were classified
by the algorithm. Correct flows (GT == algo) are colored by outcome class;
misclassified flows are red with counts labeled.

## What improvement looks like

- Thicker correct flows (same color, GT to algo).
- Thinner or absent red cross-flows.
- Equal bar heights on both sides (no systematic over/under-prediction).

## Red-flag patterns

- **Thick red flows**: systematic misclassification (e.g., retrieved -> displaced_sa).
- **Unbalanced bars**: algorithm over-predicts one class at expense of another.
- **Many flows to uncertain**: algorithm abstaining too often.

## Exhaustive flag

NOTE: Videos not marked exhaustive=True have potentially inflated label_correct_wrong_reach counts (algo may catch real reaches GT did not label).

## Rendering params

- SNAPSHOT_DIR: `Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\outcome\outcome_master_pre_v4.0.0_baseline`
- FIGSIZE: (10, 7)
- DPI: 300

## Data summary

- Total segments: 6682
- Correct: 5098 (76.3%)
