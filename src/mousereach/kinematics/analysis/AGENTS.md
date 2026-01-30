<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# analysis/

## Purpose
Multi-level analysis framework for aggregating, analyzing, and exporting MouseReach kinematic data across sessions, mice, cohorts, timepoints, and experimental conditions. Provides tools for temporal trend analysis (within-session fatigue, across-session learning), group comparisons (cohorts, phases, day-of-week effects), validation against manual scoring, and export to standardized formats (ODC-SCI).

## Key Files
| File | Description |
|------|-------------|
| `data_loader.py` | Parses video filenames to extract metadata (date, mouse ID, phase), loads feature JSON files, organizes data into hierarchical structure (Mouse → Sessions → Segments → Reaches) |
| `temporal_analyzer.py` | Analyzes temporal trends within sessions (early vs. late performance, fatigue indicators) and across sessions (learning curves, longitudinal tracking) |
| `group_analyzer.py` | Group-level analysis for cohorts, training phases, and calendar effects (day-of-week, time-of-day performance patterns) |
| `validation_analyzer.py` | Compares algorithm outputs with manual/ground truth scoring, computes confusion matrices and per-outcome precision/recall metrics |
| `odc_sci_exporter.py` | Exports analysis results to Open Data Commons for Spinal Cord Injury (ODC-SCI) standard format, generating Excel files with standardized tabs |

## For AI Agents

### Working In This Directory
- **Data Loading Pipeline**: `VideoMetadata.from_filename()` parses filenames like `20250624_CNT0115_P2` to extract date, mouse ID, and training phase
- **Hierarchical Organization**: Data organized as Mouse → Sessions → Segments → Reaches, enabling multi-level aggregation
- **Temporal Analysis**:
  - **Within-session**: Early (first 25%) vs. late (last 25%) segments to detect fatigue
  - **Across-session**: Learning curves, longitudinal tracking of individual mice over weeks/months
  - Linear regression for slope-based trend detection
- **Group Analysis**:
  - Day-of-week effects (Monday fatigue, weekend effects)
  - Training phase progression (P1 → P2 → P3 → P4)
  - Cohort comparisons (control vs. experimental groups)
- **Validation Workflow**: Load ground truth files, compute outcome confusion matrix, calculate reach linkage accuracy
- **ODC-SCI Export**: Standardized Excel output with tabs for Session Summary, Reach Kinematics, Outcome Statistics, Temporal Trends
- Analysis modules expect feature JSON files from `core/feature_extractor.py`

### Key Classes and Workflows
```python
# Load data
loader = DataLoader(base_dir)
loader.load_all_videos()

# Temporal analysis
temporal = TemporalAnalyzer(loader)
within_trends = temporal.analyze_within_session('video_name')
across_trends = temporal.analyze_across_sessions('mouse_id')

# Group analysis
group = GroupAnalyzer(loader)
day_stats = group.analyze_by_day_of_week()
phase_stats = group.analyze_by_phase()

# Validation
validator = ValidationAnalyzer(base_dir)
metrics = validator.compare_to_ground_truth('video_name')

# Export
exporter = ODC_SCI_Exporter(loader)
exporter.export_behavioral_summary(output_path)
```

## Dependencies

### Internal
- `mousereach.kinematics.core` - Feature extraction results (`*_features.json`) provide input data
- `mousereach.reach` - Reach detection for validation comparisons
- `mousereach.outcomes` - Outcome classifications for validation comparisons

### External
- `pandas` - DataFrame operations for aggregation and statistical analysis
- `numpy` - Numerical computations for statistics and trend analysis
- `scipy` - Linear regression for temporal trend detection
- `xlsxwriter` - Excel file generation for ODC-SCI export
- `matplotlib` / `seaborn` - Plotting for group comparisons and temporal trends

<!-- MANUAL: -->
