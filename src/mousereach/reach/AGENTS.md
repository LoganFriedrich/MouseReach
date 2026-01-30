<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# reach/

## Purpose
Detects individual mouse reaching attempts within pellet presentation segments (Pipeline Step 3). Uses DeepLabCut pose estimation to identify when the mouse extends its paw through the slit toward the pellet by tracking hand visibility while the nose is engaged. Each reach is defined by start, apex, and end frames based on hand point tracking (RightHand, RHLeft, RHOut, RHRight) combined with nose engagement at the slit opening.

## Key Files
| File | Description |
|------|-------------|
| `cli.py` | CLI entry points for batch reach detection, triage, advance, and interactive review |
| `review_widget.py` | Napari-based interactive widget for reviewing and correcting reach detection results |
| `_review.py` | Backend logic for interactive reach review sessions |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `core/` | Core reach detection algorithms and batch processing logic |
| `analysis/` | Analysis scripts for evaluating detection accuracy and debugging boundary errors |

## For AI Agents

### Working In This Directory
- Reach detection requires nose engagement check (distance to slit center) + hand visibility threshold
- Detection rules: start = first visible hand point while nose engaged, end = hand disappears or retracts
- Minimum duration filtering removes noise from brief hand visibility artifacts
- Apex detection identifies the frame with maximum hand extension (rightmost x-position)
- Output format: `*_reaches.json` with start/apex/end frames for each detected reach
- Triage system: auto-review (high confidence) vs. needs-review (borderline cases)
- Human validation workflow: review → validate → advance to next stage

### CLI Commands (v2.3+ Single-Folder Architecture)
```bash
# Batch processing - all files in Processing/
mousereach-detect-reaches -i Processing/

# Triage updates validation_status in JSON (auto_approved or needs_review)
mousereach-triage-reaches -i Processing/

# Advance marks files ready for outcome detection
mousereach-advance-reaches -i Processing/

# Interactive review
mousereach-review-reaches --reaches video_reaches.json
mousereach-review-reaches --dir Processing/
```

**Note:** v2.3+ keeps all files in Processing/. Status tracked via `validation_status` field in JSON.

## Dependencies

### Internal
- `mousereach.segmentation` - Segment boundaries (`_segments.json`) define temporal windows for reach detection
- `mousereach.dlc` - DeepLabCut tracking data (`.h5` files) provides hand/nose position tracking
- `mousereach.state` - Pipeline state management for tracking processing status
- `mousereach.index` - Fast file indexing for batch operations

### External
- `pandas` - Reading DeepLabCut `.h5` tracking files
- `numpy` - Numerical computations for distance/threshold checks
- `napari` - Interactive visualization and review UI
- `magicgui` - UI widgets for Napari review interface

<!-- MANUAL: -->
