# Outcome Accuracy Summary Table

## What question this answers

Three-section summary of outcome classification quality:
1. **Label accuracy**: How often does the algorithm get the outcome right?
   Overall and per-class precision/recall/F1.
2. **Interaction frame accuracy**: When both sides agree on a touched outcome,
   how close is the algorithm's interaction_frame to GT?
3. **Causal reach matching**: For correctly-labeled touched segments, does the
   algorithm identify the same causal reach as GT?

## Column definitions

### Section 1 -- Label
| Column | Definition |
|--------|------------|
| Class | Outcome class or OVERALL |
| N_GT | Ground truth count |
| N_algo | Algorithm count |
| Precision | TP / (TP + FP) for this class |
| Recall | TP / (TP + FN) for this class |
| F1 | Harmonic mean of P and R |
| Strict Acc % | Correct / total (overall only) |
| Committed Acc % | Correct / committed (overall only) |
| Abstention % | Uncertain / total (overall only) |

### Section 2 -- Interaction frame
Non-overlapping buckets that sum to 100% of all paired segments.

### Section 3 -- Causal reach
Verdict breakdown for touched (non-untouched) segments.

## Exhaustive flag

NOTE: Videos not marked exhaustive=True have potentially inflated label_correct_wrong_reach counts (algo may catch real reaches GT did not label).

## Rendering params

- SNAPSHOT_DIR: `Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\outcome\outcome_master_pre_v4.0.0_baseline`
- FIGSIZE: (16, 12)
- DPI: 300
