<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# export/core

## Purpose
Core export algorithms for converting MouseReach analysis results (pellet outcomes, reach detection) into Excel/CSV formats with summary statistics.

## Key Files
| File | Description |
|------|-------------|
| `exporter.py` | Load outcome JSON files and export to Excel workbook with Summary and Details sheets |
| `summary.py` | Compile aggregate statistics across multiple videos - total retrieved/displaced/missed, success rates |

## For AI Agents

### Algorithm Details
**exporter.py:**
- Flattens nested JSON (video → segments) to flat DataFrame
- Creates Excel with 2 sheets: Summary (per-video stats) and Details (per-segment data)
- Summary includes: n_segments, n_retrieved, n_displaced, n_missed, success_rate, total_reaches
- CSV export creates single flat file (mousereach_details.csv)

**summary.py:**
- Compiles all *_pellet_outcomes.json files in a directory
- Calculates aggregate statistics across all videos
- Returns structured dict with videos list, summary stats, and metadata
- Optional save to JSON (excludes full video data in summary file)

### Data Flow
```
*_pellet_outcomes.json → load_all_results() → results_to_dataframe() → Excel/CSV
                      ↘ compile_results() → summary stats → JSON
```

### Output Formats
**Excel Summary Sheet:**
- video_name (index)
- n_segments, n_retrieved, n_displaced, n_missed
- total_reaches, mean_confidence, success_rate

**Excel Details Sheet:**
- video_name, segment_num, outcome, confidence
- n_reaches, causal_reach_id

**Summary JSON:**
- summary: n_videos, total_segments, outcome counts, overall_success_rate
- metadata: compiled_at timestamp, source_dir
- video_names: list of processed videos

### Modifying This Code
**When adding new fields:**
- Update results_to_dataframe() to include new segment-level fields
- Update groupby aggregation in export_to_excel() for summary stats
- Ensure new fields exist in JSON or provide defaults

**When changing summary calculations:**
- compile_results() assumes outcome is one of: RETRIEVED, DISPLACED, MISSED
- Success rate calculation: total_retrieved / total_segments
- Handle zero-segment videos gracefully (avoid division by zero)

**Common pitfalls:**
- JSON files may have nested structures - flatten carefully
- Missing fields should use defaults (e.g., n_reaches defaults to 0)
- Excel engine requires openpyxl installed
- Large datasets (>1000 videos) may need memory optimization

## Dependencies
**External:**
- pandas (DataFrame operations)
- openpyxl (Excel export engine)
- json (JSON file I/O)

**Internal:**
- None (pure export logic)

<!-- MANUAL: -->
