# ASPA2 - Automated Skilled Pellet Assessment v2

## What Is This?

ASPA2 is a pipeline for analyzing mouse skilled reaching behavior from video. It takes DeepLabCut tracking data and extracts meaningful behavioral metrics: which pellet the mouse is interacting with, whether reaches were successful, kinematic features of the reach, etc.

## The Big Picture

```
VIDEO → DLC TRACKING → SEGMENTATION → REACH DETECTION → SCORING → EXPORT
                            ↑
                       YOU ARE HERE
                       (Step 1)
```

**Step 1 (Segmentation):** Identify which of the 20 pellets the mouse is currently interacting with. The conveyor advances every ~30.6 seconds, creating 21 boundaries that divide the video into 22 segments (garbage_pre + pellets 1-20 + garbage_post).

**Step 2 (Reach Detection):** [Future] Detect individual reaches within each segment.

**Step 3 (Scoring):** [Future] Classify reach outcomes (success, fail, drop, etc.)

**Step 4 (Export):** [Future] Generate summary statistics and sync with SharePoint.

---

## Directory Structure

```
A:\
├── !!! DLC Input\                    ← WORKING DIRECTORY (staging area)
│   ├── DLC_Queue\                    ← Videos waiting for DLC processing
│   ├── DLC_Complete\                 ← DLC done, ready for segmentation
│   ├── Seg_AutoReview\               ← High-confidence, quick check needed
│   ├── Seg_NeedsReview\              ← Low-confidence or anomalies, careful review
│   ├── Seg_Validated\                ← Human-approved, ready for Step 2
│   ├── Score_NeedsReview\            ← [Future]
│   ├── Score_Validated\              ← [Future] Ready for export
│   └── Failed\                       ← Problems requiring investigation
│
├── MouseReach\
│   └── ASPA2\                        ← CODE LIVES HERE
│       ├── Step1_Segmentation\       ← All segmentation scripts
│       ├── Step2_ReachDetection\     ← [Future]
│       ├── Step3_Scoring\            ← [Future]
│       └── docs\                     ← Documentation
│
X:\ (NAS)
├── ! DLC Output\
│   ├── Unanalyzed\                   ← Source videos (copy FROM here)
│   │   ├── Multi-Animal\             ← Original collages (NEVER DELETE)
│   │   └── Single_Animal\            ← Cropped, ready for processing
│   └── Analyzed\                     ← Completed files (copy TO here)
│       └── Sort\{GroupName}\
```

---

## How Files Flow

```
NAS Unanalyzed/Single_Animal/
        │
        │ (1) Copy to Analysis PC
        ▼
A:\!!! DLC Input\DLC_Queue\
        │
        │ (2) Run DeepLabCut
        ▼
A:\!!! DLC Input\DLC_Complete\
        │
        │ (3) Run Step1_Segmentation scripts
        ▼
A:\!!! DLC Input\Seg_AutoReview\  ──or──  Seg_NeedsReview\  ──or──  Failed\
        │                                        │
        │ (4) Human reviews                      │
        └──────────────┬─────────────────────────┘
                       ▼
A:\!!! DLC Input\Seg_Validated\
        │
        │ (5) [Future: Run Step2, Step3...]
        ▼
A:\!!! DLC Input\Score_Validated\
        │
        │ (6) Copy back to NAS, delete local
        ▼
NAS Analyzed/Sort/{GroupName}/
```

---

## Quick Start

```powershell
# Activate environment
conda activate DLC-env

# Navigate to code
cd A:\MouseReach\ASPA2\Step1_Segmentation

# Run segmentation on a folder of DLC files
python 1_batch_segment.py

# Triage results into review queues
python 2_triage_results.py

# Review/correct segments (opens GUI)
python 3_review_tool.py

# Move validated files forward
python 4_advance_validated.py
```

---

## File Naming Convention

```
20250806_CNT0311_P2.mp4
│        │       │└─ repetition number
│        │       └── tray type (P=pillar)
│        └────────── animal ID (cohort + subject)
└─────────────────── date (YYYYMMDD)

Derived files share the same prefix:
  20250806_CNT0311_P2.mp4                    ← source video
  20250806_CNT0311_P2DLC_resnet50_...h5      ← DLC tracking
  20250806_CNT0311_P2_segments.json          ← segmentation boundaries
  20250806_CNT0311_P2_seg_validation.json    ← human approval record
```

---

## Requirements

- Python 3.8+
- DeepLabCut environment
- PyQt5
- pandas, numpy, scipy
- opencv-python

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.1.0 | 2025-12-19 | Segmentation algorithm handles stuck trays, late-start videos |
| 2.0.0 | 2025-12-18 | SABL crossing detection method |
| 1.0.0 | 2025-12-17 | Initial motion peak method |

---

## Current Status

- **Step 1 (Segmentation):** ✅ Complete - 99.5% accuracy on 20 ground truth videos
- **Step 2 (Reach Detection):** ❌ Not started
- **Step 3 (Scoring):** ❌ Not started
- **Step 4 (Export):** ❌ Not started

---

*Blackmore Lab - Marquette University*
*Last updated: 2025-12-19*
