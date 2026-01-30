# Ground Truth Files Needing Review

**Generated:** 2026-01-21
**Analysis Summary:** Algorithm accuracy cannot be properly evaluated because GT files contain unverified data.

---

## Executive Summary

| GT Type | Human-Verified | Auto-Seeded (UNVERIFIED) | Trustworthiness |
|---------|---------------|--------------------------|-----------------|
| **Reaches** | 524 (71.8%) | 206 (28.2%) | PARTIAL |
| **Outcomes** | 387 (96.8%) | 13 (3.2%) | GOOD |
| **Segments** | 0 (0%) | 420 (100%) | NONE |

**Critical Issue:** The reach and segment GT files contain data the human never verified. We cannot know if the algorithm is correct because we're comparing it to its own output in many cases.

---

## Problem 1: Reach GT Files (28.2% UNVERIFIED)

### What's Wrong
- 206 out of 730 reaches in GT have `human_corrected: false` and `source: "algorithm"`
- These were auto-seeded by the algorithm and the human never touched them
- When evaluating, 82% of "false positives" actually match these unverified reaches
- **We can't tell if these are real reaches the human just didn't look at, or garbage the human should have deleted**

### Files Needing Review
```
Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing\20250912_CNT0310_P1_reach_ground_truth.json
  - 290 total reaches
  - 196 human-touched (68%)
  - 94 auto-seeded UNVERIFIED (32%)

Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing\20251021_CNT0405_P4_reach_ground_truth.json
  - 440 total reaches
  - 328 human-touched (75%)
  - 112 auto-seeded UNVERIFIED (25%)
```

### What Human Needs To Do
For each GT file, open in the GT tool and:
1. Go through each reach with `human_corrected: false`
2. Either:
   - Verify it's correct (set `human_verified: true`)
   - Delete it (if it's not a real reach)
   - Correct the boundaries (if timing is wrong)

---

## Problem 2: Segment GT Files (100% UNVERIFIED)

### What's Wrong
- ALL 420 boundaries across 20 GT files have NO human corrections
- The segment GT files are essentially copies of algorithm output
- There's no tracking field for human verification
- **We cannot evaluate segmenter accuracy at all**

### Files Needing Review
ALL segment GT files:
```
Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing\*_seg_ground_truth.json
(20 files, 21 boundaries each = 420 total)
```

### What Human Needs To Do
For each video:
1. Watch the video and identify segment boundaries manually
2. Compare to GT file boundaries
3. Correct any that are wrong
4. The GT tool needs a "mark as verified" feature for segments

---

## Problem 3: Outcome GT Files (3.2% UNVERIFIED)

### What's Wrong (Minor Issue)
- 13 out of 400 outcomes are not human-verified
- This is a small percentage, but they should still be reviewed

### Files With Unverified Outcomes
```
20250709_CNT0216_P3_outcome_ground_truth.json: 7 unverified
20251009_CNT0307_P4_outcome_ground_truth.json: 3 unverified
20251009_CNT0309_P1_outcome_ground_truth.json: 1 unverified
20251021_CNT0401_P4_outcome_ground_truth.json: 1 unverified
20250806_CNT0312_P2_outcome_ground_truth.json: 1 unverified
```

---

## Algorithm Improvement Analysis

### Reach Detector (v3.5.0)
- **Recall: 98%** against human-touched GT (excellent)
- **Precision: 65-70%** BUT 82% of "false positives" match unverified GT
- **Actual precision is probably ~94%** if unverified reaches are real
- **Conclusion: Algorithm is GOOD, GT needs review**
- Attempted tightening extent threshold from -15 to -10, but recall dropped to 85% - reverted

### Outcome Classifier (v2.4.4)
- **Accuracy: 98.4%** against human-verified GT (excellent)
- Only 6 errors out of 387
- All 6 errors are correctly flagged for human review
- **Conclusion: Algorithm is EXCELLENT, no changes needed**

### Segmenter (v2.1.0)
- **Cannot evaluate** - GT IS algorithm output
- Boundaries match within ±5 frames because they're copies
- **Conclusion: Need manual GT creation before improvement is possible**

---

## Recommendations

### Priority 1: Fix Segment GT (HIGH)
The segment GT is useless. Options:
1. Add `human_verified` field to segment GT format
2. Manually review at least 5 videos to create true GT
3. Then we can evaluate and improve segmenter

### Priority 2: Verify Reach GT (MEDIUM)
The 206 unverified reaches need review:
1. Open each reach GT file in GT tool
2. Review reaches with `human_corrected: false`
3. Verify, delete, or correct each one
4. This will give us true precision numbers

### Priority 3: Verify Outcome GT (LOW)
Only 13 unverified outcomes - quick review needed.

---

## Impact on "100% Match" Goal

You asked: "If the rules are right, why isn't algorithm at 100%?"

**Answer: We can't tell because the GT is contaminated.**

- Reach GT: 28% of items were never verified by human
- Segment GT: 100% is algorithm output, no human verification
- Outcome GT: 96.8% is good, but we have no segment GT to contextualize

Once GT is cleaned up, we can:
1. Measure true accuracy
2. Identify actual algorithm errors (not GT errors)
3. Improve rules to get closer to 100%

---

## Files Modified During Analysis

```
Y:\2_Connectome\Behavior\MouseReach\src\mousereach\reach\core\reach_detector.py
  - Added note about attempted v3.6 threshold change (reverted)
  - No functional changes
```
