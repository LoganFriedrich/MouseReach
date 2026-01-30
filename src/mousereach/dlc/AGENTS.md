<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# dlc

## Purpose
Step 1 of the MouseReach pipeline: DeepLabCut pose estimation wrapper for batch processing single-animal reaching videos. Manages DLC batch inference, quality validation, and file movement through pipeline stages. Integrates with user-trained DLC models for 18-bodypart tracking (hands, scoring area corners, reference points, pellet, etc.).

## Key Files
| File | Description |
|------|-------------|
| `cli.py` | CLI entry points for `mousereach-dlc-batch`, `mousereach-dlc-quality` |
| `widget.py` | Napari widget for DLC model selection, batch processing, and quality checks |
| `core/batch.py` | DLC batch inference wrapper, config YAML preprocessing, file management |
| `core/quality.py` | Quality validation: reference point stability, likelihood scores, bodypart coverage |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `core/` | DLC integration and quality assessment algorithms |

## For AI Agents

### Working In This Directory
- **DLC model location:** User-provided (not included in repo). Train your own model for your camera setup.
- **Expected bodyparts:** 18 points including Reference, SABL/SABR/SATL/SATR (scoring area), BOXL/BOXR (slit), RightHand/RHLeft/RHOut/RHRight, Pellet, Pillar, Nose, ears, foot, tail
- **Config preprocessing:** `fix_config_yaml()` strips broken `video_sets` entries before running DLC
- **Quality thresholds:**
  - Reference stability: BOXL/BOXR std < 3px (good), < 5px (fair)
  - Likelihood threshold: 0.6 (matches DLC pcutoff default)
  - Critical coverage: SABL, SABR, RightHand, BOXL, BOXR must have >80% high-confidence frames
- **Unsupported trays:** Videos with tray types E/F are automatically filtered out

### CLI Commands
```bash
# Batch DLC processing
mousereach-dlc-batch -i DLC_Queue/ -c /path/to/dlc/project/config.yaml --gpu 0

# Quality check outputs
mousereach-dlc-quality *.h5 -o quality_reports/

# Full workflow (find videos, process, move completed)
# (See core/batch.py:run_dlc_workflow)
```

### Pipeline Integration (v2.3+ Single-Folder Architecture)
- **Input stage:** `Paths.DLC_QUEUE` - Cropped videos awaiting pose estimation
- **Output stage:** `Paths.PROCESSING` - All pipeline files stay in Processing/ folder
- **Auto-advancement:** `move_completed_to_output()` moves processed videos to Processing/
- **Preview generation:** Creates compressed preview videos during file movement for review widgets
- **Status tracking:** Validation status tracked in JSON metadata, not folder location

### Quality Assessment
Quality reports include:
- Reference point stability (BOXL/BOXR standard deviation)
- Per-bodypart likelihood coverage (fraction of frames > 0.6 likelihood)
- Critical point coverage (SABL, SABR, RightHand, etc.)
- Overall rating: good/fair/poor/failed

## Dependencies

### Internal
- `mousereach.config.Paths` - Pipeline path configuration
- `mousereach.config.is_supported_tray_type` - Tray type validation
- `mousereach.video_prep.compress` - Preview video creation

### External
- `deeplabcut` (optional, required for actual processing) - Pose estimation
- `pandas`, `numpy` - DLC data loading and quality analysis
- `napari`, `qtpy` - GUI widget
- `h5py` - Reading DLC .h5 outputs

<!-- MANUAL: -->
