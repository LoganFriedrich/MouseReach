# Archived Old Ground Truth Files

**Archived:** 2026-02-05
**Reason:** These files contain contaminated/unverified data that was incorrectly used for algorithm evaluation.

## Problems with These Files

1. **Reach GT files**: 28.2% of reaches were auto-seeded from algorithm output with `human_verified: false`
2. **Segment GT files**: 100% were algorithm output copies with no human verification
3. **Outcome GT files**: Mostly okay (96.8% verified) but incomplete coverage

## Correct Data Location

Use the **unified ground truth files** instead:
```
Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing\*_unified_ground_truth.json
```

These unified GT files:
- Have proper verification tracking (`verified: true/false` per item)
- Were reviewed by Colin (SCHULTZC) and friedrichl
- Use the unified schema with `completion_status` tracking
- Cover all 25 videos with full verification

## Do Not Use These Files

These archived files should NOT be used for:
- Algorithm evaluation
- Performance metrics
- Accuracy comparisons

They are kept only for historical reference.
