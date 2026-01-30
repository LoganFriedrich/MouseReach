# MouseReach User Guide

A step-by-step guide to processing mouse reaching behavior videos from raw footage to final data export.

---

## Quick Start (NEW in v2.3.0)

The fastest way to use MouseReach:

```bash
conda activate mousereach
mousereach
```

This launches all MouseReach tools in a single napari window with tabs. You can:
- Switch between tools using tabs
- Load a video once, it's shared across all tools
- Work through the pipeline step-by-step

**With a video:**
```bash
mousereach "path/to/your/video.mp4"
```

**Review tools only (for ground truth annotation):**
```bash
mousereach --reviews "path/to/video.mp4"
```

---

## Overview

MouseReach processes multi-animal reaching videos through a 7-step pipeline:

```
Multi-Animal Video → Crop → DLC → Segment → Reaches → Outcomes → Features → Export
     (collage)       Step0  Step1   Step2    Step3     Step4      Step5     Step6
```

Each step produces output files that feed into the next step. Human review checkpoints ensure data quality.

**Key Features (v2.3.0):**
- Unified launcher (`mousereach` command) with all tools in tabs
- Shared video layer (load once, not 3x per tool)
- Cross-tool awareness (save in one step, downstream steps auto-refresh)
- Drag-and-drop video support
- Single-folder architecture (all files in Processing/, status in JSON)

---

## Prerequisites

Before starting, ensure you have:

1. **MouseReach installed** - See `INSTALL.md`
2. **FFmpeg installed** - Required for video cropping
3. **Trained DLC model** - Your 17-point model for pose estimation
4. **Multi-animal collage video(s)** - 2x4 grid layout (.mkv format)

---

## Step 0: Video Preparation

**Goal:** Split 8-camera collage into individual animal videos.

### Input
Multi-animal collage video with filename format:
```
20250704_CNT0101,CNT0205,CNT0305,CNT0306,CNT0102,CNT0605,CNT0309,CNT0906_P1.mkv
├─────── ───────────────────────────────────────────────────────────── ──┤
  Date              8 Animal IDs (positions 1-8)                      Session
```

The collage layout is 2 rows × 4 columns:
```
┌────┬────┬────┬────┐
│ 1  │ 2  │ 3  │ 4  │  ← Top row
├────┼────┼────┼────┤
│ 5  │ 6  │ 7  │ 8  │  ← Bottom row
└────┴────┴────┴────┘
```

### Process

**Option A: Crop a single video**
```bash
mousereach-crop -i path/to/collage.mkv -o output_folder/
```

**Option B: Crop all videos in a directory**
```bash
mousereach-crop -i /path/to/Unanalyzed/Multi-Animal/ -o /path/to/Unanalyzed/Single_Animal/
```

**Option C: Crop and queue for DLC**
```bash
mousereach-crop -i input_folder/ -o output_folder/ --queue
```

### Output
Individual .mp4 files for each animal:
```
20250704_CNT0101_P1.mp4
20250704_CNT0205_P1.mp4
20250704_CNT0305_P1.mp4
... (8 files per collage)
```

### Notes
- Animal ID with cohort "00" (e.g., CNT0001) are skipped (blank positions)
- Requires FFmpeg in PATH

---

## Step 1: DeepLabCut Processing

**Goal:** Generate pose tracking data from individual videos.

### Input
- Single-animal .mp4 videos from Step 0
- Your trained DLC project config

### Process

Run DLC analysis using your preferred method:

**Option A: DeepLabCut GUI**
```bash
python -m deeplabcut
# Use Analyze Videos tab
```

**Option B: Python script**
```python
import deeplabcut
deeplabcut.analyze_videos('path/to/config.yaml', ['video1.mp4', 'video2.mp4'])
```

**Option C: MouseReach batch command**
```bash
mousereach-dlc-batch -i videos/ -c path/to/config.yaml
```

### Quality Check
```bash
mousereach-dlc-quality *.h5 -o quality_reports/
```

This checks:
- Reference point stability (BOXL, BOXR corners)
- Tracking coverage (% high-confidence frames)
- Critical point reliability (hand markers)

### Output
DLC tracking files:
```
20250704_CNT0101_P1DLC_resnet50_YourModelMar15shuffle1_500000.h5
```

### Notes
- GPU recommended for faster processing
- ~2-5 minutes per video depending on length and hardware

---

## Step 2: Segmentation

**Goal:** Identify the 21 boundary frames that divide each video into 20 pellet delivery segments.

### Input
- DLC .h5 files from Step 1
- Corresponding .mp4 videos (in same directory or locatable)

### Process

**1. Run batch segmentation (v2.3+ single-folder)**
```bash
mousereach-segment -i Processing/
```

**2. Triage results (v2.3+ updates JSON validation_status)**
```bash
mousereach-triage -i Processing/
```

Updates `validation_status` in JSON:
- `auto_approved` - High confidence, can auto-advance
- `needs_review` - Requires human verification

**3. Review flagged files (Napari)**
```bash
napari
```
Then: **Plugins → MouseReach Segmentation → Boundary Review Tool**

- Load a `*_segments.json` file
- Video and boundaries display automatically
- Adjust boundaries as needed
- Save when done

**Keyboard shortcuts:**
| Key | Action |
|-----|--------|
| Space | Play/Pause |
| Left/Right | Previous/Next frame |
| Shift+Left/Right | Jump 10 frames |
| N / P | Next/Previous file |
| Ctrl+S | Save |

**4. Advance validated files (v2.3+ updates JSON status)**
```bash
mousereach-advance -i Processing/
```

### Output
Segmentation JSON files:
```
20250704_CNT0101_P1_segments.json
```

Contains:
- 21 boundary frame numbers
- Confidence scores
- Validation status

---

## Step 3: Reach Detection

**Goal:** Detect individual reaching movements within each pellet segment.

### Input
- Validated segmentation files (`*_segments.json`)
- DLC .h5 files
- Video files

### Process

**1. Run batch detection (v2.3+ single-folder)**
```bash
# Process all videos in Processing/
mousereach-detect-reaches -i Processing/

# Skip videos with ground truth files (for reprocessing)
mousereach-detect-reaches -i Processing/ -s "*reaches_ground_truth.json"
```

**2. Triage results (auto-runs, updates JSON validation_status)**
```bash
mousereach-triage-reaches -i Processing/
```

Updates `validation_status` in JSON:
- `auto_approved` - High confidence
- `needs_review` - Requires review

**3. Review flagged files (Napari)**
```bash
napari
```
Then: **Plugins → MouseReach Reach Detection → Reach Annotator**

- Load a `*_reaches.json` file
- Reaches displayed as colored overlays
- Edit start/end frames, add/delete reaches

**Keyboard shortcuts:**
| Key | Action |
|-----|--------|
| Space | Play/Pause forward |
| Shift+R | Play reverse |
| Left/Right | Previous/Next frame |
| Shift+Left/Right | Jump 10 frames |
| N / P | Next/Previous segment |
| S | Set reach start at current frame |
| E | Set reach end at current frame |
| A | Add new reach |
| Delete | Delete selected reach |
| Ctrl+S | Save progress |

**Playback speeds:** 1x, 2x, 4x, 8x, 16x (click buttons)

**4. Advance validated files (v2.3+ updates JSON status)**
```bash
mousereach-advance-reaches -i Processing/
```

### Output
Reach detection JSON files:
```
20250704_CNT0101_P1_reaches.json
```

Contains per-segment:
- Reach count
- Start/end frames for each reach
- Apex frame (maximum extension)
- Confidence scores

---

## Step 4: Pellet Outcomes

**Goal:** Classify the outcome of each pellet segment.

### Outcome Categories

| Outcome | Description |
|---------|-------------|
| `retrieved` | Mouse grabbed and ate the pellet |
| `displaced_sa` | Pellet knocked into scoring area (not eaten) |
| `displaced_outside` | Pellet knocked outside scoring area |
| `untouched` | Pellet still on pillar at segment end |
| `no_pellet` | No pellet visible at segment start |
| `uncertain` | Could not determine outcome |

### Input
- Validated segmentation files (`*_segments.json`)
- DLC .h5 files
- Video files

### Process

**1. Run batch detection (v2.3+ single-folder)**
```bash
# Process all videos in Processing/
mousereach-detect-outcomes -i Processing/

# Skip videos with ground truth files (for reprocessing)
mousereach-detect-outcomes -i Processing/ -s "*outcomes_ground_truth.json"
```

**2. Triage results (auto-runs, updates JSON validation_status)**
```bash
mousereach-triage-outcomes -i Processing/
```

Updates `validation_status` in JSON:
- `auto_approved` - High confidence
- `needs_review` - Requires review

**3. Review flagged files (Napari)**
```bash
napari
```
Then: **Plugins → MouseReach Pellet Outcomes → Pellet Outcome Annotator**

- Load a `*_pellet_outcomes.json` file
- Review each segment's classification
- Correct misclassifications

**Keyboard shortcuts:**
| Key | Action |
|-----|--------|
| Space | Play/Pause forward |
| Shift+R | Play reverse |
| Left/Right | Previous/Next frame |
| N / P | Next/Previous segment |
| R | Mark as Retrieved |
| D | Mark as Displaced (SA) |
| O | Mark as Displaced (Outside) |
| U | Mark as Untouched |
| I | Set interaction frame |
| K | Set outcome-known frame |
| Ctrl+S | Save progress |

**Playback speeds:** 0.25x, 0.5x, 1x, 2x, 4x, 8x, 16x (click buttons or use menu)

**4. Advance validated files (v2.3+ updates JSON status)**
```bash
mousereach-advance-outcomes -i Processing/
```

### Output
Outcome JSON files:
```
20250704_CNT0101_P1_pellet_outcomes.json
```

Contains per-segment:
- Outcome classification
- Confidence score
- Key frames (interaction, outcome known)
- Flag reasons for uncertain cases

---

## Step 5: Grasp Kinematics

**Goal:** Extract kinematic features from each reach, linked to pellet outcomes.

### Input
- Validated reach files (`*_reaches.json`)
- Validated outcome files (`*_pellet_outcomes.json`)
- DLC .h5 files

### Process

**1. Run batch feature extraction**
```bash
mousereach-grasp-analyze -i Processing/
```

**2. Triage results**
```bash
mousereach-grasp-triage -i Processing/
```

**3. Review features (optional)**
```bash
mousereach-grasp-review
```

### Output
Feature files:
```
20250704_CNT0101_P1_features.json
```

Contains per-reach:
- Reach extent (pixels, ruler units, mm)
- Velocity at apex and peak velocity
- Trajectory straightness and smoothness
- Hand angle and rotation
- Contextual features (reach number, causal reach flag)
- Linked outcome classification

---

## Step 6: Export

**Goal:** Generate analysis-ready spreadsheets from processed data.

### Input
- Validated outcome files from Step 4

### Process

**Export to Excel:**
```bash
mousereach-export -i Processing/ -o results.xlsx
```

**Export to CSV:**
```bash
mousereach-export -i Processing/ -o csv_folder/ --format csv
```

### Output

**Excel file** contains multiple sheets:
- **Summary** - Per-video statistics
- **Segments** - Per-segment details
- **Reaches** - Individual reach data

**Key columns:**
| Column | Description |
|--------|-------------|
| video_name | Source video |
| segment_id | Pellet segment (1-20) |
| outcome | R/D/O/U |
| n_reaches | Reaches in segment |
| first_reach_frame | When reaching started |
| success_rate | % Retrieved |

---

## Typical Workflow Summary

### Option A: Unified Launcher (Recommended for Review)

```bash
# Launch all tools in one window
mousereach

# Or with a specific video
mousereach "path/to/video.mp4"

# Or just review tools for ground truth annotation
mousereach --reviews "path/to/video.mp4"
```

Then use the tabs to switch between Step 2b (Boundaries), Step 3b (Reaches), and Step 4b (Outcomes).

### Option B: CLI Batch Processing

```bash
# 1. Crop collages
mousereach-crop -i Multi-Animal/ -o Single_Animal/ --queue

# 2. Run DLC (use your preferred method)
# ... DLC processing ...

# 3. Segment (v2.3+ single-folder architecture)
mousereach-segment -i Processing/
# Auto-triage updates validation_status in JSON
# Review in Napari if needed, then advance
mousereach-advance -i Processing/

# 4. Detect reaches
mousereach-detect-reaches -i Processing/
# Auto-triage updates validation_status in JSON
# Review in Napari if needed, then advance
mousereach-advance-reaches -i Processing/

# 5. Classify outcomes
mousereach-detect-outcomes -i Processing/
# Auto-triage updates validation_status in JSON
# Review in Napari if needed, then advance
mousereach-advance-outcomes -i Processing/

# 6. Extract kinematic features
mousereach-grasp-analyze -i Processing/

# 7. Export
mousereach-export -i Processing/ -o results.xlsx
```

---

## Troubleshooting

### "Video file not found"
- Ensure .mp4 is in same directory as .h5/.json files
- Or set video path explicitly in the tool

### Napari plugin not appearing
- Reinstall: `pip install -e .` (from MouseReach root directory)
- Restart Napari completely

### Keybinding conflicts in Napari
- Only load one MouseReach plugin at a time
- Restart Napari when switching between plugins

### DLC tracking looks wrong
- Check DLC model quality with `mousereach-dlc-quality`
- May need to retrain model or filter bad videos

### Segmentation boundaries incorrect
- Use Napari review tool to manually adjust
- Check that pellet delivery mechanism is visible in video

---

## File Organization (v2.3+ Single-Folder Architecture)

```
Y:\2_Connectome\Behavior\MouseReach_Pipeline\
├── Raw_Videos\
│   └── Multi-Animal\           # Original collages (archive)
│
├── DLC_Queue\                  # Videos waiting for DLC processing
│
├── Processing\                 # ALL active videos live here
│   ├── video.mp4               # Single-animal video
│   ├── videoDLC*.h5            # DLC tracking results
│   ├── video_segments.json     # Boundaries (validation_status in JSON)
│   ├── video_reaches.json      # Reaches (validation_status in JSON)
│   ├── video_pellet_outcomes.json  # Outcomes (validation_status in JSON)
│   └── video_features.json     # Kinematic features per reach (Step 5)
│
├── Results\
│   └── exports\                # Final Excel/CSV files
│
└── Failed\                     # Processing errors
```

**Note:** v2.3+ uses JSON `validation_status` field ("needs_review", "auto_approved",
"validated") instead of separate folders. All files for a video stay together in Processing/.

---

## Getting Help

- Check individual `Step*/README.md` files for detailed options
- Use `--help` with any command: `mousereach-segment --help`
- Report issues: [GitHub Issues](https://github.com/LoganFriedrich/MouseReach/issues)

---

*MouseReach v2.3.0 | Updated 2025-01-21*
