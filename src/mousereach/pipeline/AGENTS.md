<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# pipeline

## Purpose
Batch processing orchestration for the MouseReach analysis pipeline. Provides a unified widget and processor that automatically runs videos through all three analysis stages (segmentation → outcome detection → reach detection) in a single operation. Files remain in the Processing/ folder throughout, with status tracked via JSON metadata rather than folder location.

## Key Files
| File | Description |
|------|-------------|
| `core.py` | Core pipeline orchestration logic, includes UnifiedPipelineProcessor for running all stages and utilities for scanning pipeline status |
| `batch_widget.py` | Napari widget (UnifiedPipelineWidget) for running the complete pipeline with progress tracking and visual feedback |
| `__init__.py` | Package exports for UnifiedPipelineWidget, UnifiedPipelineProcessor, and pipeline status utilities |

## Subdirectories
None

## For AI Agents

### Working In This Directory
- The pipeline uses a **unified architecture** where all files stay in `Processing/` folder - never move files between folders
- Status is tracked via `validation_status` fields in JSON files, not by folder location
- Pipeline flow: DLC files → Segmentation (auto-triage) → Outcomes + Reaches (parallel) → Validation
- Files that "need review" are paused but don't block other files from processing
- The widget provides both automatic batch processing and targeted reprocessing of specific files
- All processing is done in background threads to avoid blocking the napari UI
- Progress callbacks use stage names: 'segmentation', 'outcomes', 'reaches', 'advancing'

### Key Patterns
- **Scan before run**: Always call `scan_pipeline_status()` to determine what needs processing
- **Non-blocking**: Processing runs in worker threads with Qt signals for UI updates
- **Stateless**: Each run starts fresh by scanning current pipeline state
- **Fail-safe**: Errors are caught per-file, allowing batch to continue
- **Auto-triage**: Segmentation results automatically advance or pause based on quality metrics

## Dependencies

### Internal
- `mousereach.config` - Paths configuration (PROCESSING_ROOT)
- `mousereach.segmentation.core.batch` - Segmentation processing and validation
- `mousereach.outcomes.core.pellet_outcome` - Outcome detection (PelletOutcomeDetector)
- `mousereach.reach.core.reach_detector` - Reach detection (ReachDetector)
- `mousereach.reach.core.triage` - Anomaly detection for reach validation

### External
- `napari` - GUI viewer and notifications
- `qtpy` - Qt widgets for UI (QWidget, QPushButton, QProgressBar, etc.)
- `pathlib` - Path handling
- `json` - JSON file I/O for metadata
- `shutil` - File movement operations
- `threading` - Background processing
- `dataclasses` - Structured data (PipelineStatus, UnifiedResults)

<!-- MANUAL: -->
