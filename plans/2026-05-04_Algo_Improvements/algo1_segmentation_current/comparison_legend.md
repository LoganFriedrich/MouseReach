# v2.2.0 vs v2.3.0 Comparison -- Endpoint Phase Offsets

**Date**: 2026-04-25
**Experiment**: Replace velocity-only endpoint rescue with phase-offset-based detection for B1/B21.
**Verdict**: NET REGRESSION -- v2.3.0 introduced 1 miss + 1 phantom. Source code reverted.

## Comparison Table

| Metric | v2.2.0 | v2.3.0 | Delta |
|---|---|---|---|
| n_phantom (all) | 0 | 1 | +1 (WORSE) |
| n_miss (all) | 0 | 1 | +1 (WORSE) |
| within_5f (all) | 973/987 (98.6%) | 971/986 (98.5%) | -2 (WORSE) |
| within_10f (all) | 986/987 (99.9%) | 985/986 (99.9%) | -1 (WORSE) |
| mean_abs_delta (all) | 1.28 | 1.28 | 0.00 |
| median_abs_delta (all) | 1 | 1 | 0 |
| mean_signed_delta (all) | -0.44 | -0.44 | 0.00 |
| **Endpoint mean_abs_delta** | **2.43** | **2.41** | **-0.02 (marginal)** |
| Endpoint n_miss | 0 | 1 | +1 (WORSE) |
| Endpoint median_abs_delta | 2 | 2 | 0 |
| Endpoint mean_signed_delta | -1.26 | -1.31 | -0.05 |
| Inter-pellet mean_abs_delta | 1.16 | 1.16 | 0.00 |
| Inter-pellet n_phantom | 0 | 0 | 0 |
| Inter-pellet n_miss | 0 | 0 | 0 |

## Analysis

The phase-offset approach failed to improve endpoint accuracy for several reasons:

1. **B1**: The SABL centered-entry + 8f offset produces a frame that is similar to
   what the velocity rescue already finds. The velocity peak and the centered-entry
   point are correlated signals -- the velocity peak occurs during the same
   transition event. Adding a fixed offset to the centered-entry frame does not
   systematically improve on the velocity peak frame because the per-video variance
   in the offset (IQR = 4f) is comparable to the per-video variance in the velocity
   peak's error.

2. **B21**: SABL's centered-exit has median offset = 0f (already at GT), but IQR =
   7.0f. The wide IQR means the centered-exit frame is NOT consistently at GT --
   it's sometimes early, sometimes late. v2.2.0's velocity rescue, despite its
   imperfections, handles more edge cases (e.g., videos where the SA departure
   doesn't produce a clean centered-exit signal). The v2.3.0 code's search window
   or fallback logic lost 1 boundary that v2.2.0 captured.

3. **The large-error outliers are unchanged.** The 3 videos with |B1 delta| > 5f and
   8 videos with |B21 delta| > 5f in v2.2.0 are caused by apparatus-specific timing
   anomalies (stuck trays, rapid-fire presentations, DLC hallucination patterns) that
   a simple offset cannot fix.

## Conclusion

The empirical phase offset table IS informative (it quantifies the temporal relationship
between body-part transitions and GT), but using it as a replacement for the velocity
rescue is not an improvement. The right path forward for B1/B21 accuracy likely requires:

- A fundamentally different B1/B21 detection strategy (e.g., likelihood-regime-change
  detection as suggested in the hallucination survey)
- Per-video adaptive offsets rather than corpus-wide medians
- Or accepting that the current ~2.4f MAE on endpoints is near the limit of what
  DLC-based detection can achieve for B1/B21 given the apparatus physics

## Phase Offset Table (for reference)

| Boundary | Bodypart | N | Median Offset | IQR | Interpretation |
|---|---|---|---|---|---|
| B1 | SABL | 47 | +8.0 | 4.0 | SA enters ~8f before GT B1 |
| B1 | SATL | 47 | +7.0 | 5.0 | SA enters ~7f before GT B1 |
| B1 | SABR | 47 | +3.0 | 6.0 | Right corners enter ~3f before GT B1 |
| B1 | SATR | 47 | +1.0 | 5.5 | Right corners enter ~1f before GT B1 |
| B21 | SABL | 47 | 0.0 | 7.0 | SA exit is at GT B21 (but high variance) |
| B21 | SATL | 47 | -6.0 | 12.0 | WIDE -- unreliable for B21 |
| B21 | SABR | 47 | -2.0 | 3.5 | Right corners exit ~2f after GT B21 |
| B21 | SATR | 47 | +1.0 | 6.0 | Moderate variance |
