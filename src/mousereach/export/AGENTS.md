<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# export/

## Purpose
Step 6 of the MouseReach pipeline: Exports fully analyzed reach behavior data to Excel/CSV formats for statistical analysis. Aggregates results from segmentation, reach detection, and outcome classification stages into researcher-friendly spreadsheets with summary statistics.

## Key Files
| File | Description |
|------|-------------|
| `cli.py` | Command-line interfaces for `mousereach-export` and `mousereach-summary` |
| `core/exporter.py` | Core export logic: load JSON results, convert to DataFrames, write Excel/CSV |
| `core/summary.py` | Aggregate statistics: compile all videos, calculate success rates, outcome counts |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `core/` | Core export and summary algorithms |

## For AI Agents

### Working In This Directory
- Export modules consume `*_pellet_outcomes.json` files from the `Processing/` folder
- Excel output creates multi-sheet workbooks: "Summary" (per-video stats), "Details" (per-segment data)
- Summary statistics include: total reaches, success rates (RETRIEVED/DISPLACED/MISSED), confidence scores
- All export functions use pandas for data manipulation
- Output format is optimized for R/Python statistical analysis downstream

### CLI Commands
```bash
# Export to Excel (default)
mousereach-export -i Processing/ -o results.xlsx

# Export to CSV
mousereach-export -i Processing/ -o output_dir/ --format csv

# Generate summary statistics
mousereach-summary -i Processing/
mousereach-summary -i Processing/ -o summary.json
```

### Typical Workflow
1. All pipeline stages (segmentation, reach detection, outcome classification) complete
2. Videos validated and ready in `Processing/`
3. Run `mousereach-export` to generate final Excel/CSV outputs
4. Run `mousereach-summary` for quick aggregate statistics

## Dependencies

### Internal
- `mousereach.config`: Path configuration
- Result JSON formats from:
  - `mousereach.outcomes` (pellet_outcomes.json)
  - `mousereach.reach` (reaches.json)
  - `mousereach.segmentation` (segments.json)

### External
- `pandas`: DataFrame operations and Excel/CSV writing
- `openpyxl`: Excel workbook engine (for multi-sheet support)

<!-- MANUAL: -->
