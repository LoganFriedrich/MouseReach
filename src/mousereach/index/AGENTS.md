<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# index/

## Purpose
Infrastructure module providing fast pipeline file tracking via a cached index file. Eliminates slow folder scanning on network drives (~30s) by maintaining a single JSON index (~5ms read). Tracks file locations, validation statuses, metadata from JSON files, and folder modification times for smart invalidation. Critical for GUI responsiveness and pipeline state queries.

## Key Files
| File | Description |
|------|-------------|
| `index.py` | `PipelineIndex` class: load/save index, query videos by stage/status, validation tracking |
| `scanner.py` | Full folder scanning logic: rebuild index, extract metadata from JSON files, CLI tools |

## For AI Agents

### Working In This Directory
- **Performance Critical**: Index read must be <10ms for fast GUI startup
- Index structure (v2.0):
  - Single `Processing/` folder architecture (no Seg_NeedsReview, Reach_NeedsReview folders)
  - Validation status stored IN JSON files (`validation_status` field)
  - Files stay co-located (video + DLC + segments + reaches + outcomes in same folder)
- Index file location: `{PROCESSING_ROOT}/pipeline_index.json`
- Index auto-rebuilds if version mismatch or corruption detected
- Smart invalidation: tracks folder mtimes, only rescans changed folders

### Key Concepts

#### Index Structure
```json
{
  "version": "2.0",
  "generated_at": "2026-01-16T10:30:00",
  "processing_root": "<your-pipeline-folder>",
  "folder_mtimes": {"Processing": 1234567890.0},
  "videos": {
    "20250704_CNT0101_P1": {
      "video_id": "20250704_CNT0101_P1",
      "current_stage": "Processing",
      "files": {"Processing": ["...DLC.h5", "..._segments.json", ...]},
      "metadata": {
        "seg_validation": "validated",
        "reach_validation": "needs_review",
        "outcome_validation": "not_started",
        "tray_type": "P1",
        "tray_supported": true
      }
    }
  }
}
```

#### Validation Status Values
- `not_started`: No JSON file yet
- `needs_review`: Human review required
- `auto_approved`: High confidence, pre-approved for review
- `validated`: Human verified, ready for next stage

#### Tray Type Filtering
- Filters out unsupported tray types (Flat/Easy) by default
- `tray_supported` flag in metadata determines visibility
- All query methods have `include_unsupported` parameter

### CLI Commands
```bash
# Rebuild entire index (full scan)
mousereach-index-rebuild

# Show index status
mousereach-index-status

# Refresh changed folders only
mousereach-index-refresh

# Refresh specific folders
mousereach-index-refresh Processing DLC_Queue
```

### Common Query Patterns
```python
from mousereach.index import PipelineIndex

index = PipelineIndex()
index.load()  # Fast: single file read

# Get videos needing review
seg_review = index.get_needs_seg_review()
reach_review = index.get_needs_reach_review()
outcome_review = index.get_needs_outcome_review()

# Check if ready to archive
ready = index.get_ready_to_archive()

# Get validation status
status = index.get_pipeline_status("20250704_CNT0101_P1")
# Returns: {"seg": "validated", "reach": "needs_review", "outcome": "not_started"}
```

### Index Update Events
Pipeline modules call these methods to keep index synchronized:
- `record_file_created()`: New JSON file created (segments, reaches, outcomes)
- `record_validation_changed()`: Validation status updated
- `record_gt_created()`: Ground truth file created
- `record_gt_complete()`: All GT items human-verified
- `remove_video()`: Video archived (deleted from index)

### Scanner Logic (scanner.py)
- Scans folders, identifies videos by DLC .h5 files (canonical identifiers)
- Parses JSON files to extract metadata (validation_status, confidence, counts)
- Handles legacy files (pre-validation_status) with confidence-based triage
- Updates folder mtimes for smart invalidation

## Dependencies

### Internal
- `mousereach.config`: `PROCESSING_ROOT`, `get_video_id()`, `FilePatterns`, `parse_tray_type()`

### External
- `json`: Index file I/O and JSON parsing
- `pathlib.Path`: File system operations
- `datetime`: Timestamp tracking

<!-- MANUAL: -->
