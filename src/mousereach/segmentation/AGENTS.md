<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# segmentation

## Purpose
Step 2 of the MouseReach pipeline: Detects the 21 trial boundaries (pellet presentation events) in DLC tracking data using scoring area (SA) anchor point motion. Implements a multi-strategy detection algorithm with primary SABL crossing detection, secondary anchor validation, and fallback methods. Includes automatic confidence-based triage and interactive Napari-based review tool.

## Key Files
| File | Description |
|------|-------------|
| `cli.py` | CLI entry points for `mousereach-segment`, `mousereach-triage`, `mousereach-advance`, `mousereach-segment-review` |
| `review_widget.py` | Napari interactive boundary review widget with keyboard shortcuts |
| `core/segmenter_robust.py` | Multi-strategy boundary detection algorithm (SABL crossing + fallback methods) |
| `core/batch.py` | Batch segmentation orchestration with validation status management |
| `core/triage.py` | Auto-triage logic: routes high-confidence results to auto-review, low-confidence to manual review |
| `core/advance.py` | Advances validated segmentations to next pipeline stage |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `core/` | Segmentation algorithms, triage logic, and batch processing |

## For AI Agents

### Working In This Directory
- **Algorithm approach:** Detects SABL (scoring area bottom-left anchor) crossing the box center (BOXL-BOXR midpoint) with positive velocity
- **Expected output:** Exactly 21 trial boundaries per video
- **Detection hierarchy:**
  1. Primary: SABL x-position crossing with velocity filter
  2. Validation: SABR/SATL/SATR motion correlation at candidate frames
  3. Fallback: Aggregate SA motion magnitude peak detection
- **Validation statuses:**
  - `auto_approved` - High confidence (21 boundaries, low CV, good anchor correlation)
  - `needs_review` - Requires human verification
  - `validated` - Human-reviewed and approved
- **Key parameters:**
  - Velocity threshold: >0.03 ruler units/frame
  - Smoothing: 5-frame median filter
  - Minimum interval: 60 frames between boundaries (prevent duplicates)
  - Maximum interval: 600 frames (detect missing boundaries)

### CLI Commands
```bash
# Batch segmentation with auto-triage (v2.3+ single-folder)
mousereach-segment -i Processing/

# Manual triage existing results (updates validation_status in JSON)
mousereach-triage -i Processing/

# Advance validated files to next stage (updates JSON status)
mousereach-advance -i Processing/

# Interactive review tool (Napari)
mousereach-segment-review [optional_video.mp4]
```

### Review Tool Keyboard Shortcuts
```
SPACE         - Set current boundary to this frame
N             - Next boundary
P             - Previous boundary
S             - Save validated (for pipeline)
Left/Right    - Move 1 frame
Shift+L/R     - Move 10 frames
```

### Pipeline Integration
- **Input stage:** `Paths.DLC_COMPLETE` - Videos with DLC .h5 tracking files
- **Triage destinations:**
  - `Paths.SEG_AUTO_REVIEW` - High-confidence auto-approved segmentations
  - `Paths.SEG_NEEDS_REVIEW` - Low-confidence requiring human review
  - `Paths.FAILED` - Detection failures
- **Output stage:** `Paths.SEG_VALIDATED` - Human-validated or auto-approved segmentations
- **Output format:** `{video_id}_segments.json` with metadata (boundaries, confidence, validation status, algorithm version)

### Segmentation Output Schema
```json
{
  "video_id": "20250704_CNT0101_P1",
  "boundaries": [120, 450, 780, ...],  // 21 frame indices
  "num_boundaries": 21,
  "detection_method": "primary",
  "confidence_score": 0.95,
  "validation_status": "auto_approved",
  "segmenter_version": "2.3",
  "timestamp": "2026-01-16T10:30:00"
}
```

## Dependencies

### Internal
- `mousereach.config.Paths` - Pipeline path configuration
- `mousereach.config.is_supported_tray_type` - Tray validation

### External
- `pandas`, `numpy` - DLC data loading and signal processing
- `scipy` - Peak detection, signal filtering
- `napari` - Interactive review widget
- `opencv-python` (cv2) - Video playback in review widget
- `qtpy` - Qt bindings for widget UI

<!-- MANUAL: -->
