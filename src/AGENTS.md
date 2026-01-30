<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# src

## Purpose

Source code root directory. Contains the `mousereach` Python package with all pipeline modules.

## Key Files

| File | Description |
|------|-------------|
| (none at this level) | All code is in `mousereach/` subdirectory |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `mousereach/` | Main Python package (see `mousereach/AGENTS.md`) |
| `mousereach.egg-info/` | Build artifact (auto-generated, not indexed) |

## For AI Agents

### Working In This Directory

- This is just a container for `setuptools` packaging
- All actual code lives in `mousereach/`
- Navigate to `mousereach/AGENTS.md` for module documentation

### Package Structure

```
src/
└── mousereach/           ← The actual package
    ├── __init__.py
    ├── config.py         ← Central configuration
    ├── state.py          ← Qt signal/slot state management
    ├── launcher.py       ← Main entry point
    ├── napari.yaml       ← Widget registration
    └── [18 submodules]/  ← Pipeline steps and utilities
```

## Dependencies

### Internal

- `pyproject.toml` (parent) - Defines package name, entry points
- `mousereach/config.py` - All path and threshold configuration

<!-- MANUAL: Add source-level notes below -->
