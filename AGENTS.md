<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# MouseReach

**Automated Single Pellet Assessment v2** - A 6-step pipeline for analyzing mouse reaching behavior in neuroscience research.

## Purpose

MouseReach processes video recordings of mice performing single-pellet reaching tasks. It uses DeepLabCut for pose estimation, then applies custom algorithms to detect trial boundaries, individual reach attempts, and classify pellet outcomes (Retrieved/Displaced/Missed). The pipeline produces quantitative kinematic data for behavioral neuroscience research.

## Key Files

| File | Description |
|------|-------------|
| `pyproject.toml` | Package definition, 50+ CLI entry points, dependencies |
| `CLAUDE.md` | Development context - reinstall rules, architecture notes |
| `README.md` | Quick start guide |
| `INSTALL.md` | Full installation instructions |
| `USER_GUIDE.md` | End-user workflow documentation |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `src/` | All source code (see `src/AGENTS.md`) |
| `.claude/` | Claude Code rules and detailed documentation (see `.claude/AGENTS.md`) |
| `conda_envs/` | Conda environment (not indexed - binary files) |

## For AI Agents

### Working In This Directory

1. **NEVER run `pip install`** - It hangs on network drives. Tell user to run it manually.
2. **Editable install** - Changes to `.py` files are live; no reinstall needed.
3. **Reinstall needed ONLY when** changing `pyproject.toml` (new CLI commands or dependencies).
4. **Activate conda first**: `conda activate mousereach`

### Architecture (v2.3+)

- **Single Processing folder** - All post-DLC files stay together
- **JSON-based status** - `validation_status` in JSON files, not folder location
- **Status values**: `needs_review`, `auto_approved`, `validated`
- Files only leave Processing via archive when ALL statuses are `validated`

### Pipeline Steps

```
Step 0: Video Prep    → Crop 8-camera collages to single-animal clips
Step 1: DLC           → DeepLabCut pose estimation (18 bodyparts)
Step 2: Segmentation  → Detect 21 trial boundaries per session
Step 3: Reach         → Detect individual reach attempts within trials
Step 4: Outcomes      → Classify: Retrieved (R), Displaced (D), Missed (M)
Step 5: Kinematics    → Extract 48 kinematic features per reach
Step 6: Export        → Generate Excel/CSV analysis results
```

### Testing Requirements

- No automated test suite yet (pytest configured but no tests)
- Manual testing via review widgets
- Ground truth files: `*_seg_ground_truth.json`, `*_reach_ground_truth.json`, `*_outcome_ground_truth.json`
- Evaluation: `mousereach-eval --all [path]`

### Common Patterns

- **CLI entry points** defined in `pyproject.toml [project.scripts]`
- **Napari widgets** registered in `src/mousereach/napari.yaml`
- **Config** centralized in `src/mousereach/config.py`
- **State management** via Qt signals in `src/mousereach/state.py`

## Dependencies

### External

| Package | Version | Purpose |
|---------|---------|---------|
| `deeplabcut` | latest | Pose estimation |
| `tensorflow` | >=2.10 | DLC backend |
| `napari` | >=0.4.17 | GUI framework |
| `opencv-python` | >=4.5 | Video processing |
| `pandas` | >=1.3 | Data analysis |
| `numpy` | ==1.26.4 | Pinned for TF/DLC compatibility |

### Environment Variables

| Variable | Purpose | Required? |
|----------|---------|-----------|
| `MouseReach_PROCESSING_ROOT` | Pipeline working folders | **Yes** (run `mousereach-setup`) |
| `MouseReach_NAS_DRIVE` | Archive location | Optional |

## Ripple Effects

| If You Change... | Also Update... |
|------------------|----------------|
| `config.py` paths | `.claude/rules/machines.md` |
| `pyproject.toml` scripts | `CLAUDE.md`, `.claude/rules/pipeline-steps.md` |
| Core algorithm (`core/*.py`) | `.claude/rules/pipeline-steps.md` |
| JSON output format | `.claude/rules/data-formats.md` |
| Widget names | `napari.yaml`, `.claude/rules/widgets.md` |
| Environment variables | `CLAUDE.md`, `.claude/rules/machines.md` |

<!-- MANUAL: Add project-specific notes below this line -->
