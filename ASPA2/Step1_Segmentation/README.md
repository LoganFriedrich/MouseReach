# Step 1: Segmentation

## What This Step Does

**Goal:** Identify the 21 boundary frames that divide the video into 22 segments (garbage_pre + pellets 1-20 + garbage_post).

**Why it matters:** Everything downstream depends on knowing which pellet we're analyzing. If segmentation is wrong, we'll attribute reaches to the wrong pellet, corrupting all results.

---

## The Scripts (Run In Order)

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `1_batch_segment.py` | Run segmentation algorithm on all DLC files in a folder | DLC .h5 files | `*_segments.json` files |
| `2_triage_results.py` | Move files to appropriate review queue based on confidence | Folder with segments | Files moved to Seg_AutoReview/, Seg_NeedsReview/, or Failed/ |
| `3_review_tool.py` | GUI for human to review/correct boundaries | Video + DLC + segments | `*_seg_validation.json` (approval record) |
| `4_advance_validated.py` | Move validated files to next stage | Seg_AutoReview/ or Seg_NeedsReview/ | Files moved to Seg_Validated/ |

---

## How The Algorithm Works (Plain English)

### The Physical Setup
- Mouse is in a box with a conveyor tray
- Tray has 20 pellet positions
- Conveyor advances every ~30.6 seconds (1837 frames at 60fps)
- Each advance creates a "boundary" where pellet N exits and pellet N+1 enters

### What We Track
DeepLabCut tracks 17 points including:
- **SABL, SABR** - Scoring Area left/right edges (on the tray)
- **BOXL, BOXR** - Box reference points (stationary)

### The Key Insight
When the tray advances:
1. SABL moves from LEFT of box center → crosses center → moves to RIGHT
2. This happens quickly (high velocity)
3. The crossing point is the boundary

### Detection Logic
```
For each frame:
  1. Calculate SABL position relative to box center
  2. Calculate SABL velocity (how fast it's moving)
  3. If SABL is NEAR CENTER (±5 pixels) AND MOVING FAST (>1.2 px/frame):
     → This is a potential boundary

Group nearby detections into events
Pick the frame with best score (closest to center + highest velocity)
Result: 21 boundary frames
```

### Edge Cases Handled
1. **Late-start videos:** Recording started after first advance (B1 at frame 3000+)
   - Algorithm checks if there's real motion before first detection
   - If no motion → trust first detection as B1

2. **Stuck tray:** Mouse blocks pellet, tray can't advance for 2-3 minutes
   - Creates one long segment followed by rapid-fire short ones
   - Algorithm detects these and flags as anomaly (for human review)

3. **Slow B1:** First boundary sometimes has low velocity
   - Algorithm tries lower threshold (0.8) if initial pass finds <21 boundaries

---

## Confidence & Triage

The algorithm reports confidence metrics:

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Boundaries detected | 21 | 19-23 | <19 or >23 |
| Interval CV | <0.05 | 0.05-0.5 | >0.5 (unless stuck tray) |
| Anomalies | 0 | 1-2 | >2 |

**Triage rules:**
- 21 boundaries + CV<0.05 + no anomalies → `Seg_AutoReview/` (quick check)
- 21 boundaries + (high CV OR anomalies) → `Seg_NeedsReview/` (careful review)
- <21 boundaries OR algorithm failed → `Failed/` (investigate)

---

## Files Created

### Segments File (`*_segments.json`)
```json
{
  "video_name": "20250806_CNT0311_P2",
  "segmenter_version": "2.1.0",
  "algorithm": "sabl_centered_crossing_v2",
  "boundaries": [3278, 5117, 6954, ...],  // 21 frame numbers
  "n_boundaries": 21,
  "diagnostics": {
    "interval_cv": 0.0023,
    "n_primary_candidates": 21,
    "anomalies": ["used_lower_velocity_threshold"]
  }
}
```

### Validation File (`*_seg_validation.json`)
```json
{
  "video_id": "20250806_CNT0311_P2",
  "original_boundaries": [3278, 5117, ...],
  "validated_boundaries": [3278, 5117, ...],  // may be corrected
  "changes_made": 0,
  "validated_by": "lfriedrich",
  "validated_at": "2025-12-19T14:32:00Z",
  "validation_time_seconds": 45,
  "notes": ""
}
```

---

## Directory Flow

```
A:\!!! DLC Input\DLC_Complete\
    │
    │  [1_batch_segment.py]
    │  Creates *_segments.json next to each .h5 file
    │
    │  [2_triage_results.py]
    │  Moves VIDEO + DLC + SEGMENTS as a bundle
    ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Seg_AutoReview/     Seg_NeedsReview/          Failed/     │
│  (high confidence)   (needs careful look)      (broken)    │
│        │                    │                      │       │
│        │  [3_review_tool.py - human reviews]       │       │
│        │                    │                      │       │
│        ▼                    ▼                      ▼       │
│  Creates *_seg_validation.json                   Fix or    │
│        │                    │                   discard    │
│        └────────┬───────────┘                              │
│                 │                                          │
│                 │  [4_advance_validated.py]                │
│                 ▼                                          │
│        Seg_Validated/                                      │
│        (ready for Step 2)                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Using The Review Tool (3_review_tool.py)

1. **Launch:** `python 3_review_tool.py`
2. **Select video:** Click "Select Video" → pick .mp4
3. **Select DLC:** Click "Select DLC" → pick .h5 (auto-loads segments if found)
4. **Navigate:**
   - Slider: scrub through video
   - Arrow keys: ±1 frame
   - `<< Prev` / `Next >>`: jump to boundaries
   - `< -1seg` / `+1seg >`: jump ~1837 frames
5. **Correct boundaries:**
   - Navigate to correct frame
   - Click "Set Boundary Here"
6. **Save:** Click "Save Validation" when done

**What to look for:**
- Boundary should be when SABL crosses box center
- All 21 boundaries should be present
- No duplicate or missing boundaries

---

## Troubleshooting

### "Only found 19 boundaries"
- Check for stuck tray (one very long segment)
- May need manual annotation of missing boundaries

### "Boundaries look shifted by ~1800 frames"
- Algorithm may have missed B1 or detected a false B1
- Check the first few minutes of video manually

### "High CV but boundaries look correct"
- Stuck tray causes high CV (expected)
- If boundaries are actually correct, approve anyway

### "Algorithm completely failed"
- Check DLC tracking quality (SABL may be lost)
- Check if video is unusual (different camera angle, etc.)

---

## Performance

Tested on 20 ground-truth validated videos:
- **418/420 boundaries correct (99.5%)**
- Mean error: 9.7 frames (0.16 seconds)
- 18/20 videos perfect (21/21)
- 2 videos with 1 miss each (extreme edge cases)

---

## Files In This Folder

```
Step1_Segmentation/
├── README.md                 ← You are here
├── 1_batch_segment.py        ← Run segmentation
├── 2_triage_results.py       ← Sort into queues
├── 3_review_tool.py          ← GUI for review
├── 4_advance_validated.py    ← Move validated forward
├── core/
│   └── segmenter_robust.py   ← The actual algorithm
└── docs/
    └── algorithm_details.md  ← Deep dive on the math
```
