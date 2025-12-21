# ASPA2 Pipeline Specification

## Overview

This document defines the data pipeline for ASPA2 (Automated Skilled Pellet Assessment v2), from video capture through final scoring.

**Core Principles:**
1. **Never lose originals** - Multi-animal collage videos are the source of truth
2. **Work with copies** - Always process copies, retain upstream states
3. **No silent advancement** - Files only move forward with explicit approval
4. **Track everything** - Who, what, when for every action
5. **Fail safe** - Uncertain? Hold for review.

---

## Physical Infrastructure

```
BEHAVIORAL ROOM                          ANALYSIS ROOM              NAS (X:)
┌─────────────────────┐                  ┌─────────────┐           ┌─────────────────────┐
│ Collection PC 1     │                  │ Analysis PC │           │ Source of Truth     │
│ Collection PC 2     │                  │ (DLC + ASPA)│           │ Long-term storage   │
│ 8 cameras each      │                  │             │           │                     │
│ OBS → collage       │───────────────▶  │ A:\!!! DLC  │◀────────▶ │ X:\! DLC Output\    │
│ Arduino control     │   copy/move      │    Input\   │  sync     │                     │
└─────────────────────┘                  └─────────────┘           └─────────────────────┘
```

---

## File Naming Convention

### Multi-Animal (Collage)
```
20250704_CNT0101,CNT0205,CNT0305,CNT0306,CNT0102,CNT0605,CNT0309,CNT0906_P1.mkv
│        │                                                        ││└─ rep #
│        │                                                        │└── tray type
│        └─ 8 animal IDs (left→right, top row then bottom row)    │
└─ date (YYYYMMDD)                                                │
                                                                  │
Position mapping:                                                 │
  1=top-left, 2=top-center-left, 3=top-center-right, 4=top-right │
  5=bot-left, 6=bot-center-left, 7=bot-center-right, 8=bot-right │
```

### Single-Animal (Cropped)
```
20250806_CNT0311_P2.mp4
│        │  │ ││ │└─ repetition number (1, 2, 3...)
│        │  │ ││ └── tray type (P=pillar, E=easy, F=flat)
│        │  │ │└──── subject number (01-99)
│        │  │ └───── cohort number (01-99, 00=skip/blank)
│        │  └─────── cohort letters (CNT, etc.)
│        └────────── animal identifier
└─────────────────── date (YYYYMMDD)

Unique ID = "20250806_CNT0311_P2" (everything before extension)
```

### Derived Files (all share the unique ID prefix)
```
20250806_CNT0311_P2.mp4                              # Source video
20250806_CNT0311_P2DLC_resnet50_MPSAOct27shuffle1_100000.h5    # DLC tracking
20250806_CNT0311_P2DLC_resnet50_MPSAOct27shuffle1_100000.csv   # DLC tracking (alt)
20250806_CNT0311_P2_meta_filtered.csv                # DLC filtered output
20250806_CNT0311_P2_dlc_quality.json                 # DLC quality report
20250806_CNT0311_P2_segments.json                    # Segmentation boundaries
20250806_CNT0311_P2_seg_validation.json              # Segmentation approval record
20250806_CNT0311_P2_scores.json                      # Reach outcomes
20250806_CNT0311_P2_score_validation.json            # Score approval record
```

---

## Tray Types

| Code | Name | Status | Notes |
|------|------|--------|-------|
| P | Pillar | **Primary** | Full tracking: pellet, pillar, tray, reaches |
| E | Easy | Future | Closer SA, flat surface. Limited tracking. |
| F | Flat | Future | Hybrid design. Limited tracking. |

For E/F trays: Can extract engagement, reach count, body angle, attentiveness.
Cannot reliably track pellet/pillar/tray positions.

---

## Pipeline Stages

### Stage 0: Capture
**Location:** Collection PC `A:\Video Cropper\Multi-Animal\`
**Action:** OBS records 8-camera collage
**Output:** `{date}_{id1,id2,...,id8}_{tray}{rep}.mkv`
**QC:** Visual check of camera alignment before recording

### Stage 1: Crop & Archive
**Location:** Collection PC or NAS
**Action:** Split collage into 8 single-animal videos
**Script:** `updated_crop_script.py`
**Output:** 8× `{date}_{id}_{tray}{rep}.mp4` (skips cohort "00")
**QC:** 
- All 8 regions processed (or correctly skipped)
- Output videos play correctly
- Original multi-animal archived

**Archive locations:**
- Multi-animal → `X:\! DLC Output\Unanalyzed\Multi-Animal\` (PERMANENT)
- Single-animal → `X:\! DLC Output\Unanalyzed\Single_Animal\`

### Stage 2: DLC Processing
**Location:** Analysis PC `A:\!!! DLC Input\`
**Action:** DeepLabCut batch inference
**Input:** `.mp4` files
**Output:** `.h5` + `.csv` + `.pickle` files
**QC:**
- [ ] Output files exist and non-empty
- [ ] Likelihood scores acceptable (mean > 0.7?)
- [ ] Reference points stable (BOXL, BOXR std < 5px)
- [ ] SA point coverage > 80%

### Stage 3: Segmentation
**Location:** Analysis PC
**Action:** Boundary detection algorithm
**Script:** `segmenter_robust.py` (v2.1.0+)
**Input:** DLC `.h5` file
**Output:** `*_segments.json`
**QC:**
- [ ] Exactly 21 boundaries detected
- [ ] Confidence score > 0.90
- [ ] No anomalies flagged, OR anomalies reviewed
- [ ] Interval CV < 0.05 (unless stuck tray flagged)

**Triage:**
| Condition | Action |
|-----------|--------|
| 21 boundaries, conf > 0.95, no anomalies | → `auto_review` (quick visual check) |
| 21 boundaries, conf 0.85-0.95 OR anomalies | → `needs_review` (careful check) |
| < 21 boundaries OR conf < 0.85 | → `needs_review` (manual annotation) |
| Segmentation failed | → `failed` (investigate) |

### Stage 4: Segmentation Validation
**Location:** Analysis PC
**Action:** Human reviews/corrects boundaries
**Tool:** `boundary_annotator_v2.py`
**Input:** Video + DLC + segments JSON
**Output:** `*_seg_validation.json`

**Validation record contains:**
```json
{
  "video_id": "20250806_CNT0311_P2",
  "segmenter_version": "2.1.0",
  "original_boundaries": [...],
  "validated_boundaries": [...],
  "changes_made": 3,
  "validated_by": "lfriedrich",
  "validated_at": "2025-12-19T14:32:00Z",
  "validation_time_seconds": 45,
  "notes": "Stuck tray B16-B17, corrected manually"
}
```

### Stage 5: Scoring (P-trays only)
**Location:** Analysis PC
**Action:** Analyze reaches within each segment
**Script:** TBD
**Input:** DLC tracking + validated segments
**Output:** `*_scores.json`
**QC:** TBD (depends on scoring algorithm)

### Stage 6: Score Validation
**Location:** Analysis PC
**Action:** Human reviews/corrects reach outcomes
**Tool:** TBD
**Output:** `*_score_validation.json`

### Stage 7: Export & Archive
**Location:** NAS
**Action:** Move completed files to final location
**Destination:** `X:\! DLC Output\Analyzed\Sort\{group}\`
**Integration:** Sync with SharePoint scoring sheets

---

## Proposed NAS Structure

```
X:\! DLC Output\
│
├── Unanalyzed\
│   ├── Multi-Animal\           # Original collages (NEVER DELETE)
│   └── Single_Animal\          # Cropped, awaiting processing
│
├── Processing\                 # NEW - work in progress
│   ├── DLC_Queue\              # Ready for DLC
│   ├── DLC_Complete\           # DLC done, awaiting segmentation
│   ├── Seg_AutoReview\         # High-conf, needs quick visual check
│   ├── Seg_NeedsReview\        # Low-conf or anomalies, needs careful review
│   ├── Seg_Validated\          # Segmentation approved, ready for scoring
│   ├── Score_NeedsReview\      # Scoring done, needs validation
│   └── Score_Validated\        # Fully complete, ready for archive
│
├── Failed\                     # NEW - problems requiring investigation
│   ├── DLC_Failed\
│   ├── Seg_Failed\
│   └── Score_Failed\
│
└── Analyzed\
    └── Sort\
        ├── Multi-Animal\       # Archived collages (copy from Unanalyzed)
        ├── Single_Animal\      # Legacy? Or flat archive?
        └── {GroupFolders}\     # Organized by experimental group
```

**Alternative: Status-file approach**

Instead of moving files between folders, keep files in place and track status in a database/JSON:

```
X:\! DLC Output\
├── Videos\
│   ├── Multi-Animal\           # All collages
│   └── Single_Animal\          # All single-animal + derived files
├── Status\
│   └── pipeline_status.json    # Tracks state of every video
└── Analyzed\
    └── Sort\{groups}\          # Final organized output
```

The status file would track:
```json
{
  "20250806_CNT0311_P2": {
    "stage": "seg_validated",
    "dlc_complete": "2025-12-18T10:00:00Z",
    "dlc_quality": "good",
    "seg_complete": "2025-12-18T10:05:00Z", 
    "seg_confidence": 0.97,
    "seg_validated_by": "lfriedrich",
    "seg_validated_at": "2025-12-19T14:32:00Z"
  }
}
```

**Pros of status-file:** Less file movement, easier to query, harder to lose files
**Cons:** Requires discipline, status can get out of sync with reality

**Recommendation:** Start with folder-based (more visible), consider status-file later if folder management becomes burdensome.

---

## Quality Reports

Each stage generates a quality report. These accumulate:

### DLC Quality Report (`*_dlc_quality.json`)
```json
{
  "video_id": "20250806_CNT0311_P2",
  "dlc_model": "resnet50_MPSAOct27shuffle1_100000",
  "total_frames": 48689,
  "fps": 60.0,
  "reference_quality": "good",
  "boxl_std": 2.3,
  "boxr_std": 1.8,
  "point_coverage": {
    "SABL": 0.94,
    "SABR": 0.91,
    "SATL": 0.88,
    "SATR": 0.85,
    "PELLET": 0.72
  },
  "mean_likelihood": 0.87,
  "issues": []
}
```

### Segmentation Report (`*_segments.json`)
Already defined - includes version, boundaries, confidence, anomalies.

### Validation Reports
Track who approved, when, what changes were made.

---

## User Tracking

Read Windows username automatically:
```python
import os
username = os.getlogin()  # or os.environ.get('USERNAME')
```

All validation records include:
- `validated_by`: Windows username
- `validated_at`: ISO timestamp
- `validation_time_seconds`: How long review took
- `notes`: Optional free-text

---

## Scripts Inventory

| Script | Stage | Purpose |
|--------|-------|---------|
| `Convert_to_mp4.py` | 0→1 | MKV to MP4 conversion |
| `updated_crop_script.py` | 1 | Split collage into single-animal |
| `dlc_batch_analyze.py` | 2 | Run DLC on queue (TBD) |
| `dlc_quality_check.py` | 2 | Generate DLC quality reports (TBD) |
| `util_batch_segment.py` | 3 | Batch segmentation |
| `segmenter_robust.py` | 3 | Core segmentation algorithm |
| `boundary_annotator_v2.py` | 4 | Segmentation review/correction |
| `pipeline_manager.py` | all | Move files, track status (TBD) |

---

## Open Questions

1. **Batch tracking:** Should we track which videos came from the same recording session? (Same collage = same conveyor timing)

2. **Automatic re-processing:** If segmenter is updated, should we flag old segments as "needs re-run"?

3. **Partial failures:** If 7/8 videos from a collage succeed but 1 fails, how do we handle?

4. **SharePoint integration:** Direct API sync, or manual export/import?

5. **Backup strategy:** How often? Where? (NAS RAID? Cloud?)

---

## Next Steps

1. [ ] Review this spec, adjust as needed
2. [ ] Implement `pipeline_manager.py` with folder-based staging
3. [ ] Add DLC quality check stage
4. [ ] Build scoring algorithm (Stage 5)
5. [ ] Build score validation tool (Stage 6)
6. [ ] Test full pipeline on 1 batch of videos
7. [ ] Document and train other users

---

*Version: 0.1 (draft)*
*Last updated: 2025-12-19*
*Author: Claude + Logan Friedrich*
