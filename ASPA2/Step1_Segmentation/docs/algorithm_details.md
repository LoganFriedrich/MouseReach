# Segmentation Algorithm - Technical Details

## Algorithm Name
`sabl_centered_crossing_v2`

## Version
2.1.0

## Overview

The algorithm detects tray advance boundaries by tracking when the Scoring Area Left (SABL) point crosses the center of the box. This crossing happens once per tray advance (21 times per video), with a distinctive velocity signature.

---

## Input Data

### Required DLC Columns
- `SABL_x`, `SABL_y` - Scoring area left edge position
- `SABL_likelihood` - Tracking confidence
- `BOXL_x`, `BOXR_x` - Box reference points (for centering)

### Data Preprocessing
1. **Center coordinates:** `sabl_rel = SABL_x - box_center_x`
2. **Smooth position:** Gaussian filter, σ=3 frames
3. **Calculate velocity:** `velocity = |diff(sabl_rel_smooth)|`
4. **Smooth velocity:** Gaussian filter, σ=5 frames

---

## Detection Logic

### Step 1: Find Crossing Events

```python
for each frame:
    if abs(sabl_rel) < 5 pixels:           # SABL near center
        if velocity > velocity_threshold:   # Moving fast
            mark as potential boundary
```

- Default velocity threshold: 1.2 px/frame
- Lower threshold (0.8) used if initial pass finds <21 candidates

### Step 2: Group Consecutive Frames

Adjacent frames are grouped into "events":
- Minimum gap between events: 30 frames
- Each event gets a score based on:
  - Proximity to exact center (closer = better)
  - Velocity at crossing (higher = better)

```python
score = (1 / (1 + abs(sabl_rel))) * velocity
```

### Step 3: Select Best Frame Per Event

For each event, pick the frame with highest score.

### Step 4: Validate and Adjust

**If exactly 21 candidates:**
- Use them directly
- Calculate intervals and CV

**If 19-23 candidates:**
- Remove lowest-confidence if >21
- Interpolate gaps if <21
- Flag as needing review

**If <19 or >23:**
- Fall back to grid fitting based on expected ~1837 frame interval
- Flag as likely failed

---

## Edge Case Handling

### Late-Start Videos
- Some videos start recording after first tray advance
- Detection: First candidate at frame >500 with no earlier motion
- Solution: Trust first detection as B1, don't project backward

### Stuck Tray
- Mouse blocks pellet, tray can't advance for 2-3 minutes
- Creates one long interval followed by several short ones
- Detection: Interval CV >0.5 with one interval >3x median
- Handling: Still detect all 21 boundaries, flag as anomaly

### Slow First Boundary
- B1 sometimes has lower velocity (tray starts from rest)
- Solution: If <21 found at threshold 1.2, retry at 0.8

---

## Output Format

```json
{
    "video_name": "20250806_CNT0311_P2",
    "segmenter_version": "2.1.0",
    "algorithm": "sabl_centered_crossing_v2",
    "total_frames": 48689,
    "fps": 60.0,
    "boundaries": [3278, 5117, 6954, ...],
    "n_boundaries": 21,
    "diagnostics": {
        "method_used": "sabl_centered_crossing",
        "velocity_threshold_used": 1.2,
        "n_primary_candidates": 21,
        "interval_cv": 0.0023,
        "mean_interval": 1836.5,
        "anomalies": []
    }
}
```

---

## Diagnostics Explained

| Field | Meaning | Good Value |
|-------|---------|------------|
| `n_primary_candidates` | How many boundary events detected | 21 |
| `interval_cv` | Coefficient of variation of segment durations | <0.05 |
| `mean_interval` | Average frames between boundaries | ~1837 |
| `velocity_threshold_used` | What threshold found boundaries | 1.2 (or 0.8 if retry) |
| `anomalies` | List of issues detected | Empty |

### Common Anomalies
- `"used_lower_velocity_threshold"` - B1 was slow, used 0.8 threshold
- `"stuck_tray_detected"` - One interval was >3x median
- `"late_start_video"` - First boundary at frame >500
- `"boundaries_interpolated"` - Had to fill in missing boundaries

---

## Performance Characteristics

### Accuracy (on 20 ground truth videos)
- 418/420 boundaries correct (99.5%)
- Mean absolute error: 9.7 frames (0.16 seconds)
- 18/20 videos perfect (all 21 boundaries within 20 frames)

### Known Failure Modes
1. **False positive from operator intervention:** If someone manually advances tray during stuck period, creates crossing signature
2. **Extreme multi-stuck videos:** Multiple back-to-back stuck trays can confuse detection
3. **Poor DLC tracking:** If SABL is lost for extended periods, boundaries in that region will be wrong

### Runtime
- ~2-3 seconds per video (37,000 frames)
- Bottleneck is DLC file loading, not algorithm

---

## Code Location

`Step1_Segmentation/core/segmenter_robust.py`

Main functions:
- `segment_video_robust(dlc_path)` - Entry point
- `find_centered_crossings(sabl_rel, velocity)` - Core detection
- `fit_grid_to_candidates(candidates, ...)` - Validation/adjustment
- `save_segmentation(boundaries, diag, path)` - Output
