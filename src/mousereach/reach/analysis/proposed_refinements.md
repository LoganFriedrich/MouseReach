# Proposed Algorithm Refinements

Based on analysis of 28 boundary disagreements between algorithm and human.

## Current Algorithm Issues

### Issue 1: Early Start (16 cases)
**Pattern**: Algorithm starts when hand likelihood first crosses 0.30, but human waits for 0.5+

**Example**:
- Frame 740: likelihood=0.32 → ALGO START
- Frame 746: likelihood=0.54 → HUMAN START

**Root cause**: 0.30 threshold is too low - catches "hand emerging" not "reach started"

### Issue 2: Late End (12 cases)
**Pattern**: After reach ends, confidence drops to 0.35-0.45 but stays above 0.30. A new reach follows, and both get merged.

**Example**:
- Frame 340: likelihood=1.00 → HUMAN END
- Frames 341-350: likelihood=0.35-0.52 (algorithm continues)
- Frame 355: likelihood=0.90 (new reach, gets merged)
- Frame 363: likelihood=0.63 → ALGO END

**Root cause**: Algorithm only ends when hand "disappears" (<0.30), not when it "retracts" (drops from high to medium)

---

## Proposed Rule Changes

### Rule 1: Require Confident Start
**Current**: Start when ANY hand point likelihood >= 0.30
**Proposed**: Start when ANY hand point likelihood >= 0.50 (or when likelihood RISES FROM <0.30 TO >0.50)

**Rationale**: The 0.30-0.50 range represents "hand barely visible" or "hand emerging". True reach start is when tracking becomes confident.

### Rule 2: End on Confidence Drop (not just disappearance)
**Current**: End when hand will be invisible (<0.30) for 2+ consecutive frames
**Proposed**: ALSO end when confidence drops from HIGH (>0.70) to MEDIUM (<0.50) even if still above 0.30

**Rationale**: A drop from 1.00 to 0.40 indicates paw retraction even though paw is still technically visible. Human annotators recognize this as reach end.

### Rule 3: Don't Merge Through Confidence Valleys
**Current**: Merge reaches separated by ≤2 frames
**Proposed**: Don't merge if the gap contains frames where confidence dropped below 0.50 (even if it stayed above 0.30)

**Rationale**: The 0.35-0.50 "valley" between reaches indicates two separate reaches, not a single reach with tracking dropout.

---

## Implementation Approach

```python
# Rule 1: Confident Start
START_THRESHOLD = 0.50  # Was 0.30

# Rule 2: End on Drop
def should_end_reach(prev_likelihood, curr_likelihood):
    # Original rule: hand disappeared
    if curr_likelihood < 0.30:
        return True
    # New rule: significant confidence drop
    if prev_likelihood >= 0.70 and curr_likelihood < 0.50:
        return True
    return False

# Rule 3: Smart Merge
def should_merge(gap_frames, dlc_df):
    # Don't merge if any gap frame has confidence 0.30-0.50
    for frame in gap_frames:
        likelihood = get_best_hand_likelihood(dlc_df.iloc[frame])
        if 0.30 <= likelihood < 0.50:
            return False  # Valley detected, keep separate
    return True  # True dropout, merge
```

---

## Expected Impact

| Metric | Current | After Rules |
|--------|---------|-------------|
| Early starts | 16 cases | ~0 cases |
| Late ends | 12 cases | ~0 cases |
| Start accuracy | 92% within ±2 | ~98% within ±2 |
| End accuracy | 95% within ±2 | ~98% within ±2 |
| Recall | 95.0% | ~95% (may drop slightly) |
| Precision | 90.1% | ~93% (fewer merges) |
