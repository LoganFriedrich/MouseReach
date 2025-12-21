# ASPA2 Development Summary
**Date:** 2025-12-19
**Project:** Automated Skilled Pellet Assessment v2 (ASPA2)
**Collaborators:** Logan Friedrich, Claude (Anthropic)

---

## Executive Summary

We have developed a robust video segmentation algorithm for the ASPA2 pipeline that achieves **99.2% accuracy** (125/126 boundaries within 50 frames) on ground-truth validated videos. We've also drafted a comprehensive pipeline specification for the entire workflow from video capture through final scoring.

---

## What We Built

### 1. Robust Segmentation Algorithm (`segmenter_robust.py` v2.1.0)

**Core Discovery:** The scoring area (SA) tracking points cross the box center with a distinctive velocity signature at each boundary. This is more reliable than motion peaks alone.

**Algorithm:**
- Primary method: SABL position crosses box center (±5px) with velocity > 0.8-1.2 px/frame
- Handles late-start videos (B1 at frame 3000+)
- Handles stuck trays with rapid-fire boundaries (65-168 frame intervals)
- Handles normal ~1837 frame intervals
- Falls back to motion peak detection if primary fails

**Performance on 6 Ground Truth Videos:**
| Video | Result | Notes |
|-------|--------|-------|
| CNT0104_P2 | 21/21 ✓ | Normal video |
| CNT0413_P2 | 21/21 ✓ | Normal video |
| CNT0415_P1 | 21/21 ✓ | Normal video |
| CNT0408_P1 | 20/21 | Stuck tray B1→B2, false positive |
| CNT0311_P2 | 21/21 ✓ | Late start + stuck tray B16→B17 |
| CNT0312_P2 | 21/21 ✓ | Late start + stuck tray B17→B18 |

**Overall:** 125/126 boundaries correct (99.2%)

### 2. Boundary Annotator Tool (`boundary_annotator_v2.py`)

- PyQt5-based GUI for reviewing/correcting segment boundaries
- DLC overlay showing tracking points
- Frame-by-frame navigation with ±1/±10/±1seg jumps
- Loads pre-computed segments (fast startup)
- Saves ground truth JSON files
- Version checking (warns if segments outdated)

### 3. Batch Processing Script (`util_batch_segment.py`)

- Processes all DLC files in a folder
- Generates `*_segments_v2.json` for each video
- Triage report: good/warnings/failed counts
- Ready for integration with pipeline staging

### 4. Pipeline Specification (`ASPA2_PIPELINE_SPEC.md`)

Comprehensive document covering:
- Physical infrastructure (collection PCs, analysis PC, NAS)
- File naming conventions
- All pipeline stages (capture → crop → DLC → segment → validate → score → archive)
- Quality gates at each transition
- Proposed NAS folder structure
- User tracking for validation records
- Scripts inventory

---

## File Inventory

### Core Algorithm Files
| File | Location | Purpose |
|------|----------|---------|
| `segmenter_robust.py` | `aspa2_core/` | Boundary detection algorithm v2.1.0 |
| `boundary_annotator_v2.py` | `tools/` | GUI for segment validation |
| `util_batch_segment.py` | `scripts/` | Batch process DLC files |

### Documentation
| File | Location | Purpose |
|------|----------|---------|
| `ASPA2_PIPELINE_SPEC.md` | Project root | Full pipeline specification |
| `ASPA2_DEVELOPMENT_PLAN.md` | Project root | Original development roadmap |
| `2025-12-19_ASPA2_Summary.md` | Project root | This document |

### Ground Truth Data (6 videos validated)
| Video ID | Status | Notes |
|----------|--------|-------|
| 20250820_CNT0104_P2 | ✓ Validated | Clean, normal timing |
| 20251029_CNT0408_P1 | ✓ Validated | Stuck tray early (B1→B2) |
| 20251031_CNT0413_P2 | ✓ Validated | Clean, normal timing |
| 20251031_CNT0415_P1 | ✓ Validated | Clean, normal timing |
| 20250806_CNT0311_P2 | ✓ Validated | Late start + stuck tray (B16→B17) |
| 20250806_CNT0312_P2 | ✓ Validated | Late start + stuck tray (B17→B18) |

### Supporting Scripts (from existing workflow)
| File | Purpose |
|------|---------|
| `Convert_to_mp4.py` | MKV to MP4 conversion |
| `updated_crop_script.py` | Split 8-camera collage into single videos |

---

## Key Technical Discoveries

### 1. Boundary Signature
- SABL x-position relative to box center: **-18px during segment → +3px at boundary**
- Velocity peak of **2.2-2.4 px/frame** within ±3 frames of boundary
- Consistent across all videos regardless of timing anomalies

### 2. Conveyor Timing
- Normal interval: **1837 ± 3 frames** (30.6 seconds at 60fps)
- CV < 0.002 for normal videos (conveyor acts as precision clock)
- Videos from same collage share identical B1 timing (within 4 frames)

### 3. Edge Cases Identified
- **Late start:** B1 can be at frame 3000+ (no motion before = trust first detection)
- **Stuck tray:** Long gap (2-3 min) followed by rapid-fire boundaries (65-168 frames apart)
- **False positives:** Operator intervention during stuck tray can create crossing signature

### 4. Batch Structure
- 8 mice per recording (4 per conveyor, 2 conveyors)
- Same-day/same-run videos share timing characteristics
- Can use batch-mates for cross-validation

---

## What's Next

### Immediate (Pipeline Infrastructure)
1. [ ] Create NAS directory structure
2. [ ] Implement `pipeline_manager.py` for file staging
3. [ ] Add DLC quality check stage
4. [ ] Integrate segmentation triage (auto_review / needs_review / failed)

### Medium-term (Scoring)
5. [ ] Design reach detection algorithm
6. [ ] Build scoring tool
7. [ ] Build score validation interface
8. [ ] Create ground truth for reach outcomes

### Long-term (Integration)
9. [ ] SharePoint sync for final data
10. [ ] Multi-user workflow testing
11. [ ] Documentation and training
12. [ ] Handle E/F tray types (if needed)

---

## How to Use Current Tools

### Run Batch Segmentation
```bash
cd A:\MouseReach\ASPA2
conda activate DLC-env
python scripts/util_batch_segment.py
# Select folder containing DLC .h5 files
# Outputs: *_segments_v2.json files
```

### Validate Segments
```bash
python tools/boundary_annotator_v2.py
# Select video file
# Review/correct boundaries
# Save ground truth
```

### Check Algorithm Version
```python
from aspa2_core.segmenter_robust import SEGMENTER_VERSION
print(SEGMENTER_VERSION)  # Should be "2.1.0"
```

---

## Repository Structure (Proposed)

```
MouseReach/
├── ASPA2/
│   ├── aspa2_core/
│   │   ├── __init__.py
│   │   └── segmenter_robust.py      # Core algorithm
│   ├── tools/
│   │   └── boundary_annotator_v2.py # Validation GUI
│   ├── scripts/
│   │   └── util_batch_segment.py    # Batch processing
│   ├── tests/
│   │   └── ground_truth/            # Validated JSONs
│   ├── docs/
│   │   └── ...
│   ├── ASPA2_PIPELINE_SPEC.md
│   ├── ASPA2_DEVELOPMENT_PLAN.md
│   └── 2025-12-19_ASPA2_Summary.md  # This file
└── README.md
```

---

## Contact / Context

This work is part of the Blackmore Lab's mouse behavioral analysis pipeline. The goal is to automate scoring of skilled reaching tasks to replace manual pellet-by-pellet scoring.

**Key constraint:** Segmentation must be ~100% accurate because all downstream analysis depends on correctly identifying which pellet the mouse is interacting with.

---

*Document version: 1.0*
*Last updated: 2025-12-19*
