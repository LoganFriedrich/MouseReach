# Watcher Integration Tests

## Overview

Comprehensive integration tests for the `mousereach.watcher` package.

- **Test Count:** 39 tests across 6 test classes
- **Dependencies:** Uses tempdir + in-memory SQLite (no real video files or DLC models required)
- **Coverage:** Database, validators, router, transfer, state manager, orchestrator

## Running Tests

```bash
# From MouseReach root directory
export PYTHONPATH=/path/to/MouseReach/src
python -m pytest tests/test_watcher_integration.py -v

# Quick run
python -m pytest tests/test_watcher_integration.py -q

# With coverage
python -m pytest tests/test_watcher_integration.py --cov=mousereach.watcher
```

## Test Classes

### 1. TestWatcherDB (11 tests)
- Database initialization and table creation
- Collage and video registration
- State transitions (valid and invalid)
- Error handling (mark_failed, reset_failed)
- Pipeline summary generation
- Audit logging

### 2. TestValidator (8 tests)
- Collage filename validation (valid/invalid formats)
- Single video filename validation
- Date validation (rejects future dates)
- Animal ID validation (requires 8 for collages)
- Tray code validation (P, E, F)
- Quarantine file handling with metadata

### 3. TestRouter (5 tests)
- P tray: Full pipeline
- E/F tray: Skip outcome_detection
- should_run_step logic
- Unknown tray defaults to full pipeline

### 4. TestTransfer (5 tests)
- safe_copy with verification
- safe_move operations
- File stability checking (quick non-blocking)

### 5. TestStateManagerIntegration (4 tests)
- Discover new collages in scan directory
- Skip non-video files
- Quarantine invalid filenames
- Collage stability tracking

### 6. TestOrchestratorIntegration (3 tests)
- Work item priority ordering
- Dry run mode
- Run-once processing

### 7. TestEdgeCases (3 tests)
- Double registration idempotency
- Nonexistent video error handling
- Database thread safety

## Test Data

All tests use fixtures:
- `tmp_path`: pytest temporary directory
- `temp_db`: In-memory SQLite WatcherDB
- `watcher_config`: Test WatcherConfig instance
- `state_manager`: WatcherStateManager with mocked dependencies
- `sample_collage_name`: Valid collage filename example
- `sample_single_name`: Valid single video filename example

## Mocked Dependencies

- DLC processing (deeplabcut module)
- Video cropping (mousereach.video_prep.core.cropper)
- Pipeline processing (mousereach.pipeline.core)
- File system paths (using tmp_path)

## Notes

- Tests do NOT require actual video files
- Tests do NOT require network drives
- Tests do NOT require DLC models
- All external dependencies are mocked
- Thread safety is tested with concurrent registrations
