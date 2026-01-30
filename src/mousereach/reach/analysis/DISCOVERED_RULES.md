# Reach Detection Rules - Derived from Ground Truth Analysis

Analysis date: 2026-01-12
Video analyzed: 20251021_CNT0405_P4 (321 reaches, 37109 frames)

---

## Summary of Discovered Rules

### Rule 1: Nose Engagement Gate

**Plain Language:** "A reach can only start when the mouse's nose is close to the slit"

**Derived Threshold:**
- Nose must be within **20 pixels** of slit center (midpoint of BOXL and BOXR)
- Based on 95th percentile of nose-to-slit distance at labeled reach starts (21.3 pixels)

**Implementation:**
```python
nose_engaged = nose_distance_to_slit < 20  # pixels
```

---

### Rule 2: Reach Start Detection

**Plain Language:** "A reach starts when ANY hand point first becomes visible while the nose is engaged"

**Findings:**
- 100% of labeled reaches have at least one hand point visible at start
- RightHand visible at 99.7% of starts
- RHRight visible at 99.7%
- RHLeft visible at 99.1%
- RHOut visible at 91.9%

**Implementation:**
```python
def reach_starts(frame):
    if not nose_engaged(frame):
        return False

    # Hand just appeared (wasn't visible in previous frame)
    any_hand_visible_now = any(hand_likelihood[point] >= 0.3 for point in hand_points)
    any_hand_visible_before = any(prev_hand_likelihood[point] >= 0.3 for point in hand_points)

    return any_hand_visible_now and not any_hand_visible_before
```

---

### Rule 3: Reach End Detection

**Plain Language:** "A reach ends either when the hand is about to disappear, OR when the hand retracts left to start a new reach"

**Two cases identified:**

#### Case A: Hand Disappearance (94.6% of reaches)
- Hand disappears within 5 frames after marked reach end
- Mean time to disappear: 1.6 frames after end
- **Rule: End reach 1-2 frames BEFORE hand likelihood drops below threshold**

#### Case B: Consecutive Reach (hand stays visible)
- 46% of inter-reach gaps have hand visible throughout
- 90.5% show hand moving LEFT between consecutive reaches
- Hand X position at reach end: ~5 pixels right of slit center
- Hand X position at next reach start: ~0 pixels (at slit center)

**Implementation:**
```python
def reach_ends(frame, reach_start_frame):
    # Case A: Hand about to disappear
    hand_visible_now = any(hand_likelihood[point] >= 0.3 for point in hand_points)
    hand_visible_soon = any(hand_likelihood_next[point] >= 0.3 for point in hand_points)

    if hand_visible_now and not hand_visible_soon:
        return True  # End at current frame (before disappearance)

    # Case B: Hand retracting for new reach
    # Detect significant leftward movement (X velocity < -2 pixels/frame)
    if hand_x_velocity < -2:  # Moving left
        return True

    return False
```

---

## Key Statistics from Analysis

| Metric | Value |
|--------|-------|
| Nose distance from slit at start | Median: 14.5px, 95th%: 21.3px |
| Nose distance from slit at end | Median: 15.8px, 90th%: 22.7px |
| Reach duration | Median: 12 frames, Mean: 22 frames |
| Hand X at reach start | ~1 pixel right of slit center |
| Hand X at reach end | ~5 pixels right of slit center |
| Inter-reach gap | Median: 30 frames |
| Minimum gap | 3 frames |

---

## Algorithm Design

### State Machine

```
IDLE -> ENGAGED -> REACHING -> IDLE
         |             ^
         |             |
         +-------------+ (consecutive reach: hand retracts)
```

### States:
1. **IDLE**: Not engaged, no reach in progress
2. **ENGAGED**: Nose near slit, waiting for hand to appear
3. **REACHING**: Hand visible, tracking reach

### Transitions:
- IDLE -> ENGAGED: nose_distance < 20px
- ENGAGED -> REACHING: any_hand_visible becomes True
- REACHING -> IDLE: hand disappears (all likelihoods < 0.3)
- REACHING -> REACHING: hand retracts left (X velocity < -2), start new reach

---

## Next Steps

1. Implement algorithm in `reach_detector.py`
2. Run on test video
3. Compare detected reaches to ground truth
4. Identify failure cases and refine rules
5. Iterate until accuracy is acceptable
