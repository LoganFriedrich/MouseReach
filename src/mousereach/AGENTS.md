<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# mousereach

## Purpose

Main Python package for the MouseReach pipeline. Contains all modules for video processing, pose estimation integration, behavioral analysis, and data export.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package definition, version (2.3.0), convenience imports |
| `config.py` | **CENTRAL CONFIG** - Paths, DLCPoints, Thresholds, Geometry, FilePatterns |
| `state.py` | Qt signal/slot state management for widget coordination |
| `launcher.py` | Main GUI entry point - tabbed napari interface |
| `lazy_video.py` | RAM-aware video loading with automatic caching strategy |
| `napari.yaml` | Widget registration for napari plugin system |
| `migrate_to_processing.py` | Migration script for v2.3 single-folder architecture |

## Subdirectories

| Directory | Purpose | Pipeline Step |
|-----------|---------|---------------|
| `video_prep/` | Crop 8-camera collages (see `video_prep/AGENTS.md`) | Step 0 |
| `dlc/` | DeepLabCut integration (see `dlc/AGENTS.md`) | Step 1 |
| `segmentation/` | Trial boundary detection (see `segmentation/AGENTS.md`) | Step 2 |
| `reach/` | Reach attempt detection (see `reach/AGENTS.md`) | Step 3 |
| `outcomes/` | Pellet outcome classification (see `outcomes/AGENTS.md`) | Step 4 |
| `kinematics/` | Grasp feature extraction (see `kinematics/AGENTS.md`) | Step 5 |
| `export/` | Excel/CSV export (see `export/AGENTS.md`) | Step 6 |
| `sync/` | Database sync for reach_data table (see `sync/AGENTS.md`) | Post-pipeline |
| `archive/` | NAS archival (see `archive/AGENTS.md`) | Post-pipeline |
| `index/` | Pipeline file index for fast startup (see `index/AGENTS.md`) | Infrastructure |
| `pipeline/` | Batch processing orchestration (see `pipeline/AGENTS.md`) | Infrastructure |
| `review/` | Unified review widget components (see `review/AGENTS.md`) | Infrastructure |
| `eval/` | Algorithm evaluation framework (see `eval/AGENTS.md`) | Development |
| `performance/` | Performance tracking and metrics (see `performance/AGENTS.md`) | Development |
| `analysis/` | Streamlit dashboard and data explorer (see `analysis/AGENTS.md`) | Analysis |
| `dashboard/` | Pipeline status napari widget (see `dashboard/AGENTS.md`) | Infrastructure |
| `docs/` | Algorithm documentation extractor (see `docs/AGENTS.md`) | Development |
| `setup/` | Configuration wizard (see `setup/AGENTS.md`) | Infrastructure |
| `ui/` | Shared UI utilities (see `ui/AGENTS.md`) | Infrastructure |

## For AI Agents

### Working In This Directory

1. **config.py is the source of truth** for all paths, thresholds, and constants
2. **state.py** coordinates between widgets via Qt signals - understand it before modifying widgets
3. **launcher.py** is the main entry point (`mousereach` command)
4. Changes to module structure require `pyproject.toml` updates

### Module Organization Pattern

Each pipeline step follows this structure:
```
step_name/
├── __init__.py     ← Re-exports from core/
├── cli.py          ← CLI entry points (main_batch, main_triage, main_review)
├── core/           ← Core algorithms
│   └── algorithm.py
└── review_widget.py or widget.py  ← napari GUI
```

### Key Imports

```python
from mousereach.config import Paths, Thresholds, DLCPoints, FilePatterns
from mousereach.config import get_video_id, parse_tray_type
from mousereach.state import MouseReachStateManager
```

### Testing Requirements

- Use `mousereach-eval` to test algorithm changes against ground truth
- Review widgets provide visual validation
- No pytest tests exist yet - consider adding for core algorithms

### Common Patterns

1. **Video ID extraction**: `get_video_id(filename)` strips suffixes to get base ID
2. **Validation status**: JSON files have `validation_status` field (needs_review/auto_approved/validated)
3. **Pipeline index**: `mousereach.index` caches file locations for fast startup
4. **State signals**: Widgets emit signals when data changes, other widgets listen

## Dependencies

### Internal

| Module | Depends On |
|--------|------------|
| All modules | `config.py` |
| All widgets | `state.py`, `ui/` |
| `reach/` | `segmentation/` (needs segment boundaries) |
| `outcomes/` | `segmentation/`, `reach/` |
| `kinematics/` | `outcomes/`, `reach/` |
| `export/` | All previous steps |
| `sync/` | `kinematics/` (consumes `_features.json`), `config.py` |
| `eval/` | Ground truth files from `review/` |

### External

| Package | Used By |
|---------|---------|
| `napari` | All widgets |
| `qtpy` | All widgets |
| `pandas` | Data processing |
| `opencv-python` | Video I/O |
| `h5py` | DLC data files |
| `deeplabcut` | `dlc/` module |

## Ripple Effects

| If You Change... | Check/Update... |
|------------------|-----------------|
| `config.py` Paths | All modules that import Paths |
| `config.py` Thresholds | Algorithm behavior in `core/` modules |
| `state.py` signals | All widgets that connect to those signals |
| `napari.yaml` | Widget registration, launcher.py |
| Any `core/` algorithm | Corresponding evaluator in `eval/` |
| JSON output format | `review/` widgets, `export/`, `analysis/` |

<!-- MANUAL: Add package-level development notes below -->
