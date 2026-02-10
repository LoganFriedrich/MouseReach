<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# core/

## Purpose
Core feature extraction algorithms for computing kinematic and behavioral features from DLC tracking data. Extracts reach-level, segment-level, and video-level features including temporal characteristics (duration, timing), spatial features (extent in pixels/ruler units/mm), velocity profiles (instantaneous and peak velocities), trajectory metrics (straightness, curvature), and contextual information (reach position within segment, outcome linkage).

## Key Files
| File | Description |
|------|-------------|
| `feature_extractor.py` | Main feature extraction engine with `FeatureExtractor` class and data structures (`ReachFeatures`, `SegmentFeatures`, `VideoFeatures`) for organizing kinematic features at multiple hierarchical levels |

## For AI Agents

### CRITICAL: Frame Boundary Accuracy IS Data Quality

**Kinematics are only as good as the reach boundaries they're computed over.** If a reach start or end frame is wrong by even a few frames, every kinematic measure extracted from that reach (velocity, acceleration, trajectory shape, peak extension, duration) is contaminated with non-behavioral frames. This noise blurs distributions and can create differential measurement artifacts between experimental groups. There is no "good enough" boundary accuracy - every mismatch vs human ground truth corrupts the kinematic data. Never describe boundary accuracy as "good" - report the error rate and what needs to be done to fix it.

### Working In This Directory
- `FeatureExtractor` is the main entry point - takes validated reach detection + outcome classification + DLC tracking data as input
- Feature extraction operates on reaches that have been linked to pellet outcomes (causal reach identification)
- Three-level feature hierarchy:
  - **ReachFeatures**: Individual reach kinematics (extent, velocity, trajectory, outcome linkage)
  - **SegmentFeatures**: Segment-level aggregations (mean/std of reach features, outcome summary)
  - **VideoFeatures**: Session-level statistics (total reaches, success rates, performance metrics)
- Physical unit conversion: pixels → ruler units (9mm calibration) → millimeters
- Velocity computation: DLC hand position derivatives smoothed with Savitzky-Golay filter, converted to mm/sec assuming 30 fps
- Causal reach linkage: identifies which reach caused the outcome based on apex frame proximity to pellet interaction frame
- Output format: JSON with nested structure preserving reach → segment → video hierarchy
- All features use dataclasses for type safety and easy serialization via `asdict()`

### Key Classes and Functions
```python
# Main extraction workflow
extractor = FeatureExtractor(dlc_file, reaches_file, outcomes_file)
video_features = extractor.extract_all_features()

# Data structures
ReachFeatures     # Single reach: extent, velocity, trajectory, outcome link
SegmentFeatures   # Segment aggregation: reach statistics, outcome counts
VideoFeatures     # Video summary: session metrics, success rates
```

## Dependencies

### Internal
- `mousereach.reach` - Reach detection results (`_reaches.json`) provide temporal boundaries and extent measurements
- `mousereach.outcomes` - Pellet outcome classifications (`_pellet_outcomes.json`) for causal reach identification
- `mousereach.dlc` - DeepLabCut tracking data (`.h5` files) for position time series
- `mousereach.segmentation` - Segment boundaries for contextual feature assignment

### External
- `pandas` - Load DLC `.h5` tracking data and manipulate time series
- `numpy` - Numerical computations for velocity, acceleration, and trajectory metrics
- `scipy` - Signal processing (Savitzky-Golay filter) for velocity smoothing
- `dataclasses` - Feature data structures with type annotations

<!-- MANUAL: -->
