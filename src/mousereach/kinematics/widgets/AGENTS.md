<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# widgets/

## Purpose
Interactive UI widgets for visualizing and exploring extracted kinematic features. Provides tools for browsing feature extraction results, inspecting individual reach kinematics, comparing segments, and navigating through video-level summaries.

## Key Files
| File | Description |
|------|-------------|
| `feature_viewer.py` | Interactive viewer for browsing `*_grasp_features.json` files, displaying video summaries, segment-by-segment navigation, and per-reach feature inspection |

## For AI Agents

### Working In This Directory
- `FeatureViewer` loads feature JSON files from `core/feature_extractor.py` and provides interactive browsing interface
- Supports navigation through hierarchical feature structure:
  - **Video-level summary**: Total reaches, outcome distribution, extractor version
  - **Segment-level view**: Segment outcome, number of reaches, causal reach ID
  - **Reach-level inspection**: Individual reach features (extent, velocity, trajectory, outcome linkage)
- Useful for visual inspection during development/debugging of feature extraction algorithms
- Can be integrated into Napari viewer or run standalone for quick feature file inspection
- Display methods use formatted text output (suitable for CLI or notebook integration)

### Usage Pattern
```python
# Load and view features
viewer = FeatureViewer(Path('video_features.json'))

# Display video-level summary
viewer.show_summary()

# Navigate segments
viewer.current_segment = 5
viewer.show_segment_details()

# Inspect individual reaches
viewer.current_reach = 2
viewer.show_reach_details()
```

### Integration with Main Widget
- `feature_viewer.py` provides data viewing logic
- Parent `widget.py` (in `../`) handles Napari integration and interactive visualization
- Separation allows feature viewing without heavy Napari dependency (useful for headless analysis)

## Dependencies

### Internal
- `mousereach.kinematics.core` - Loads feature extraction results (`*_features.json`)

### External
- `pandas` - DataFrame formatting for tabular feature display
- `json` - Load feature files

<!-- MANUAL: -->
