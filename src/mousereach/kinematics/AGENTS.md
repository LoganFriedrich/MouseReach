<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# kinematics/

## Purpose
Extracts kinematic and behavioral features from reaches linked to pellet outcomes (Pipeline Step 5). Computes detailed movement features including reach extent in physical units (mm), velocity profiles, trajectory straightness, and temporal characteristics. Links reach kinematics to outcome classifications to identify which reach caused each outcome, enabling analysis of successful vs. unsuccessful reaching strategies.

## Key Files
| File | Description |
|------|-------------|
| `cli.py` | CLI entry points for batch feature extraction, triage, and review |
| `widget.py` | Napari-based data viewer for exploring extracted features |
| `_reach_outcome_validator.py` | Validation logic for reach-outcome linkage quality |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `core/` | Feature extraction algorithms (`feature_extractor.py`) |
| `analysis/` | Analysis modules for group statistics, temporal patterns, and export to ODC-SCI format |
| `widgets/` | Additional UI components for feature visualization |

## For AI Agents

### Working In This Directory
- Feature extraction requires validated reaches (Step 4) + pellet outcomes (Step 3) + DLC tracking data
- Causal reach identification: determines which reach in a segment caused the outcome (based on apex proximity to pellet interaction frame)
- Extent features: max hand extension in pixels, ruler units (normalized to 9mm ruler), and physical mm units
- Velocity features: instantaneous velocity at apex frame, computed in px/frame and mm/sec (30 fps assumed)
- Trajectory features: straightness metrics, path curvature, acceleration profiles
- Contextual features: reach number within segment, first/last reach flags, total reaches per segment
- Output format: `*_features.json` with per-reach feature vectors linked to outcome classifications
- Analysis modules support group-level statistics, temporal pattern detection, and export to standardized formats

### CLI Commands (v2.3+ Single-Folder Architecture)
```bash
# Batch feature extraction - all files in Processing/
mousereach-grasp-analyze -i Processing/

# Triage updates validation_status in JSON
mousereach-grasp-triage -i Processing/

# Review individual feature file
mousereach-grasp-review video_features.json
```

**Note:** v2.3+ keeps all files in Processing/. Output is `*_features.json`.

## Dependencies

### Internal
- `mousereach.reach` - Reach detection results (`_reaches.json`) provide temporal boundaries
- `mousereach.outcomes` - Outcome classifications (`_pellet_outcomes.json`) for causal reach linkage
- `mousereach.dlc` - DeepLabCut tracking data for computing kinematic features
- `mousereach.segmentation` - Segment boundaries for contextual feature extraction
- `mousereach.state` - Pipeline state management

### External
- `pandas` - Reading DLC `.h5` tracking data and computing velocity/acceleration
- `numpy` - Numerical computations for trajectory analysis and feature extraction
- `scipy` - Signal processing for velocity smoothing and peak detection
- `napari` - Interactive feature visualization
- `matplotlib` - Plotting trajectory paths and kinematic profiles

<!-- MANUAL: -->
