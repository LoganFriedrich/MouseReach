# Algorithm Improvement vs Deep Learning - Decision Log

**Date:** 2026-02-06
**Decision:** Improve algorithm rules FIRST, bank DL for later

---

## Why Not Deep Learning Yet

Only **28 outcome misclassifications** across all GT data. Not enough training signal.

| Mistake Pattern | Count | Notes |
|----------------|-------|-------|
| untouched -> retrieved | 9 | Algorithm misses successful grabs |
| displaced_sa -> retrieved | 9 | Algorithm confuses displacement with retrieval |
| uncertain -> retrieved | 2 | Low-confidence calls that were actually retrievals |
| displaced_sa -> untouched | 2 | |
| Other | 6 | Various edge cases |

**28 mistakes is not enough for deep learning.** But the feature patterns ARE clear enough to write better rules.

## What We Learned from DLC Analysis

| Feature | Correct Untouched | Correct Retrieved | Mistake (untouched->retrieved) |
|---------|-------------------|-------------------|-------------------------------|
| min_hand_pellet_dist | **34.4 px** | 3.8 px | **3.6 px** |
| pellet_visible | **100%** | 53% | **67%** |
| total_pellet_movement | **7.1** | 89.9 | **129.6** |
| pellet_moved_toward_mouse | 53% | **88%** | **78%** |

**Key insight:** When the algo calls "untouched" but hand got close (3.6px) and pellet moved a lot (129.6), it's actually a retrieval.

## Actionable Rule Improvements

1. **Retrieved detection**: If `min_hand_pellet_dist < 10` AND (`pellet_visible < 80%` OR `pellet_movement > 50`), likely retrieved
2. **Boundary timing**: Algorithm is 2 frames early on average - shift +2
3. **Reach end detection**: Use hand velocity drop-off after apex

## When to Revisit Deep Learning

- When GT data reaches 100+ verified videos (currently 25)
- When outcome mistakes exceed 200+ examples
- When rule improvements plateau and can't get below ~3% error

## Files Created (Banked)

All training code is in `Y:\2_Connectome\Behavior\MouseReach\training\`:

| File | Purpose |
|------|---------|
| `config.py` | All hyperparams, body part config, training settings |
| `feature_extraction.py` | DLC feature engineering (velocity, acceleration, distances) |
| `data_loader.py` | PyTorch datasets for boundary/reach/outcome tasks |
| `models.py` | TCN, BiLSTM-CRF, MLP architectures |
| `train.py` | Full training loop with focal loss, session-level CV |
| `watchdog_model.py` | Corrective watchdog concept (outcome + reach end) |
| `learn_from_mistakes.py` | Feature analysis of algorithm failure cases |
| `analyze_failures.py` | Confusion matrix and failure counting |
| `DEEP_LEARNING_PROPOSAL.md` | Full proposal document |
