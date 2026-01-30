<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# analysis

## Purpose
Interactive Streamlit dashboard and data science toolkit for behavioral analysis of mouse reaching data. Provides statistical comparisons (t-tests, effect sizes), dimensionality reduction (PCA), clustering, correlation with external datasets (connectome, histology), and publication-ready figure generation. Designed for exploratory analysis, hypothesis testing, and generating figures for scientific papers. Includes both GUI (dashboard) and programmatic API for Jupyter/scripting workflows.

## Key Files
| File | Description |
|------|-------------|
| `dashboard.py` | Main Streamlit web app: Interactive data explorer with tabs for overview, group comparisons, PCA analysis, external correlations, and data export. Launches via `mousereach-analyze`. |
| `explorer.py` | ReachExplorer: Pre-computed statistics database (SQLite) for instant queries. Hierarchical structure (population → mouse → session → segment → reach) with KinematicProfile, SuccessFailComparison, TemporalPattern dataclasses. Build with `mousereach-build-explorer`. |
| `explorer_cli.py` | CLI for building/querying the explorer database. Faster than loading raw JSON for large datasets. |
| `data.py` | Data loading utilities: `load_all_data()`, `load_data_with_metadata()`, `build_unified_reach_data()`. ReachDataFrame class with filtering, grouping, and feature matrix extraction. Handles merging with experimental metadata (Test_Phase, Weight, etc.) from tracking spreadsheets. |
| `stats.py` | Statistical functions: `compare_groups()` (t-test/Mann-Whitney), `run_pca()`, `cluster_reaches()`, `correlate_with_external()`. Returns dataclass results with p-values, effect sizes, confidence intervals. |
| `plots.py` | Plotting functions: `plot_comparison()`, `plot_pca_scores()`, `plot_learning_curve()`, `plot_correlation_heatmap()`. Uses seaborn/matplotlib with publication-ready styling. Includes `save_publication_figure()` for SVG/PDF export. |
| `cli.py` | Entry point for `mousereach-analyze` command. Launches dashboard or exports data to CSV/Excel. |
| `__init__.py` | Module exports for all data/stats/plotting functions. Optional import handling for seaborn (graceful degradation if not installed). |

## Subdirectories
None

## For AI Agents

### Working In This Directory
- **Data loading**: Expects `*_features.json` files (preferred) or `*_reaches.json` + `*_pellet_outcomes.json` in processing folders
- **Experimental metadata**: Use `load_data_with_metadata()` with `tracking_dir` pointing to `Connectome_XX_Animal_Tracking.xlsx` files to add Test_Phase, Weight, etc. columns
- **Feature matrix**: Call `ReachDataFrame.get_feature_matrix(standardize=True)` for PCA/clustering input (removes NaN rows, standardizes columns)
- **Filtering**: Use `ReachDataFrame.filter(mouse_id=..., timepoint=..., outcome=..., exclude_flagged=True)` for subset selection
- **Grouping**: `group_by_mouse()`, `group_by_session()` for aggregated statistics
- **Timepoint mapping**: TIMEPOINT_MAPPING dict translates Test_Phase codes (2D, 3B, etc.) to experimental phases (Pre-Injury, Post-Injury, Rehab_Easy, etc.)
- **Explorer database**: Pre-compute statistics with `build_explorer_database(reach_df, output_path)` for instant queries on large datasets (avoids loading all JSON files)

### CLI Commands
```bash
# Launch interactive dashboard
mousereach-analyze
mousereach-analyze --port 8502                    # Custom port
mousereach-analyze --data-dir Processing/         # Custom data directory

# Include experimental metadata
mousereach-analyze --tracking-dir /path/to/Animal_Tracking

# Export data (no GUI)
mousereach-analyze --export results.csv           # CSV export
mousereach-analyze --export results.xlsx          # Excel with multiple sheets (Reaches, Sessions, Mice)

# Build explorer database (for fast queries)
mousereach-build-explorer --input Processing/ --output reaches.db
mousereach-query-explorer reaches.db --mouse IH5 --profile
```

## Dependencies

### Internal
- `mousereach.config.Paths` - Access to processing root directory
- `mousereach.kinematics.*` - Kinematic feature extraction (trajectory straightness, smoothness, etc.)

### External
- **Required**: `pandas`, `numpy`, `scipy`, `matplotlib`, `streamlit`, `openpyxl` (for Excel export)
- **Optional**: `seaborn` (publication-style plots), `scikit-learn` (PCA, clustering), `sqlite3` (explorer database)
- Standard library: `json`, `pathlib`, `typing`, `io`, `base64`

## Dashboard Features

### Tab 1: Overview
- Total reaches, mice, sessions, timepoints
- Outcome distribution (stacked bar chart by timepoint/tray type)
- Summary statistics table (duration, extent, velocity, straightness, smoothness)

### Tab 2: Compare Groups
- Select grouping variable (timepoint, tray_type, outcome, mouse_id)
- Choose two groups and metric to compare
- Displays: means ± std, p-value with significance stars, effect size interpretation
- Visualizations: Box plot, violin plot, or raincloud plot
- "Compare all metrics" option for batch comparisons
- Download buttons for SVG/PNG figures

### Tab 3: PCA Analysis
- Automatic feature matrix extraction with standardization
- Scree plot (variance explained per PC)
- Score plot (PC1 vs PC2) with optional coloring by phase/outcome/mouse
- Loadings plot showing feature contributions to each PC
- Export scores and loadings tables

### Tab 4: Correlations
- Upload external data (CSV/Excel) with `mouse_id` column
- Automatic matching with behavioral data (aggregated to mouse level)
- Correlation heatmap (behavioral metrics vs external metrics)
- Significance testing with p-values
- Filter for significant correlations (p < 0.05)

### Tab 5: Export
- Download filtered data as CSV or Excel
- Excel export includes 3 sheets: Reaches, Sessions (aggregated), Mice (aggregated)
- Column list display

## Typical Workflow

**Interactive Exploration:**
1. Launch dashboard: `mousereach-analyze`
2. Enter data directory, click "Load Data"
3. Apply filters (mouse, timepoint, outcome, exclude flagged)
4. Navigate tabs to explore data:
   - Overview: Get sense of data distribution
   - Compare: Test hypotheses (Pre vs Post, Success vs Fail)
   - PCA: Find latent patterns in kinematics
   - Correlations: Link behavior to histology/connectome
   - Export: Save results for external analysis (R, Prism, etc.)

**Programmatic Analysis (Jupyter):**
```python
from mousereach.analysis import load_all_data, compare_groups, run_pca, plot_comparison
from pathlib import Path

# Load data
data = load_all_data(Path("Processing/"), use_features=True, exclude_flagged=True)

# Filter
pre = data.filter(timepoint="Pre-Injury", outcome="retrieved")
post = data.filter(timepoint="Post-Injury_1D", outcome="retrieved")

# Compare
result = compare_groups(
    pre.df['max_extent_mm'],
    post.df['max_extent_mm'],
    metric_name="Max Extent (mm)",
    group1_name="Pre-Injury",
    group2_name="Post-Injury 1D"
)

print(f"p-value: {result.p_value:.4f}")
print(f"Effect size: {result.effect_size_name} = {result.effect_size:.2f}")

# Plot
fig = plot_comparison(pre.df['max_extent_mm'], post.df['max_extent_mm'],
                      group1_name="Pre", group2_name="Post",
                      metric_name="Max Extent (mm)")
fig.savefig("pre_vs_post.svg")
```

## Explorer Database (Advanced)

**Use Case**: Large datasets (>10,000 reaches) where loading all JSON files is slow.

**Build:**
```bash
mousereach-build-explorer --input Processing/ --output reaches.db
```

**Query:**
```python
from mousereach.analysis.explorer import ReachExplorer

with ReachExplorer("reaches.db") as explorer:
    # Population stats
    pop = explorer.get_population_stats()
    print(f"Overall success rate: {pop['profile']['success_rate']:.2%}")

    # Mouse profile
    mouse = explorer.get_mouse_stats("IH5")
    print(f"IH5: {mouse['profile']['n_reaches']} reaches")

    # Session query
    session = explorer.get_session_stats("IH5_20240501_P_1")
    print(f"Fatigue pattern: {session['fatigue']}")

    # Compare mice
    comparison = explorer.compare_mice(["IH5", "IH7", "IH9"], feature="success_rate")
    print(comparison)
```

**Database Schema**:
- `population_stats`: Single row with aggregate stats across all data
- `mouse_stats`: Per-animal profiles with learning curves
- `session_stats`: Per-video profiles with fatigue patterns (early/middle/late)
- `reaches`: Individual reach records for granular queries

**Pre-computed Statistics**:
- KinematicProfile: mean/std/median/IQR for extent, duration, velocity, straightness
- SuccessFailComparison: T-tests comparing successful vs failed reaches per feature
- TemporalPattern: How performance changes over time (fatigue within session, learning across sessions)

<!-- MANUAL: -->
