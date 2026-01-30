<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# outcomes/

## Purpose
Classifies pellet outcomes for each trial segment (Pipeline Step 4). Determines whether the pellet was retrieved (successfully grasped and consumed), displaced within the scoring area, displaced outside the scoring area, or left untouched. Uses geometric tracking of pellet position relative to the pillar throughout each trial segment, with outcome categories based on pellet visibility changes and displacement magnitude.

## Key Files
| File | Description |
|------|-------------|
| `cli.py` | CLI entry points for batch outcome detection, triage, advance, and interactive review |
| `review_widget.py` | Napari-based interactive widget for reviewing and correcting outcome classifications |
| `pillar_geometry_widget.py` | Widget for configuring pillar geometry parameters used in outcome detection |
| `_review.py` | Backend logic for interactive outcome review sessions |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `core/` | Core outcome classification algorithms and batch processing logic |

## For AI Agents

### Working In This Directory
- Outcome classification uses pellet position tracking in ruler units relative to pillar center
- Pillar position calculated from geometric center of SABL/SABR anchor points
- Classification rules: Retrieved (R) = pellet disappears near pillar, Displaced (D) = moves but stays in scoring area, Displaced Outside (O) = moves outside bounds or disappears away from pillar, Untouched (U) = no significant movement
- Requires reach detection results (`_reaches.json`) for causal reach attribution (linking outcomes to specific reaches)
- Output format: `*_pellet_outcomes.json` with outcome classification per segment
- Scoring area bounds defined by SATL, SATR, SABL, SABR tracking points
- Triage system: auto-review (high confidence) vs. needs-review (borderline cases)

### CLI Commands (v2.3+ Single-Folder Architecture)
```bash
# Batch processing - all files in Processing/
mousereach-detect-outcomes -i Processing/

# Triage updates validation_status in JSON (auto_approved or needs_review)
mousereach-triage-outcomes -i Processing/

# Advance marks files ready for export
mousereach-advance-outcomes -i Processing/

# Interactive review
mousereach-review-pellet-outcomes --outcomes video_pellet_outcomes.json
mousereach-review-pellet-outcomes --dir Processing/
```

**Note:** v2.3+ keeps all files in Processing/. Status tracked via `validation_status` field in JSON.

## Dependencies

### Internal
- `mousereach.segmentation` - Segment boundaries define temporal windows for outcome analysis
- `mousereach.reach` - Reach detection results for linking outcomes to causal reaches
- `mousereach.dlc` - DeepLabCut tracking data for pellet/pillar position tracking
- `mousereach.state` - Pipeline state management
- `mousereach.index` - Fast file indexing for batch operations

### External
- `pandas` - Reading DeepLabCut `.h5` tracking files
- `numpy` - Geometric calculations for pellet displacement and scoring area bounds
- `napari` - Interactive visualization and review UI
- `magicgui` - UI widgets for Napari review interface

<!-- MANUAL: -->
