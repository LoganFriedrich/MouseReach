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
| `manifest.py` | `create_processing_manifest()` -- the ONLY place a `_processing_manifest.json` is written. Also pushes a row to the version index (see below). |
| `version_index.py` | Per-video version index (SQLite/WAL): which algo versions each video was processed with. Writers push, readers read. CLI: `mousereach-version-index-build` / `-status`. |
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

### Writers push, readers read (version index)
- `create_processing_manifest()` is the single manifest writer, so it is also the
  single place that pushes to `version_index.py`. Any new path that records what
  versions processed a video must go through it, NOT write manifests directly.
- The index stores **facts** (the video's own `pipeline_versions` + `dlc_scorer`),
  never the derived "current/outdated" verdict -- the verdict depends on the
  shipped `pipeline_versions.json`, which changes, so it is derived in memory at
  read time and a version bump needs no index rewrite.
- Index writes are **best-effort**: a bookkeeping index must never fail a video's
  processing. The manifest JSON on disk stays the source of truth, and the index
  is fully rebuildable (`mousereach-version-index-build`).
- Rationale: the dashboard used to answer "is this video current?" by rglobbing
  the archive and re-parsing every manifest off the NAS (~22 ms each, thousands of
  videos), which froze the GUI. Data known at write time should be recorded at
  write time.

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
