<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# performance

## Purpose
Automatic performance tracking module that logs algorithm vs human-corrected comparison metrics during validation workflow. Stores append-only JSON logs per algorithm (segmentation, reach detection, outcome classification) with aggregate statistics. Provides CLI tools for viewing performance history, generating scientific reports, and monitoring algorithm improvements over time. Integrates seamlessly with validation review widgets to capture real-world performance without manual intervention.

## Key Files
| File | Description |
|------|-------------|
| `logger.py` | PerformanceLogger class: Logs algorithm performance during validation. Auto-called by review widgets when saving validated results. Stores JSON logs in `PROCESSING_ROOT/performance_logs/`. Computes aggregate statistics (mean precision/recall, success rates, timing errors). |
| `metrics.py` | Metric computation functions: `compute_segmentation_metrics()`, `compute_reach_metrics()`, `compute_outcome_metrics()`. Returns dataclass instances (SegmentationMetrics, ReachMetrics, OutcomeMetrics) with precision/recall, F1, confusion matrices, timing errors. |
| `report.py` | Report generation: Converts performance logs into publication-ready reports with plots (learning curves, error trends). Markdown and PDF export support. |
| `widget.py` | Napari widget for visualizing performance metrics in GUI. Shows real-time statistics during validation sessions. |
| `cli.py` | Command-line tools: `mousereach-perf` (view summary), `mousereach-perf-eval` (batch evaluation), `mousereach-perf-report` (generate reports). |
| `__init__.py` | Module exports for logger and metric computation functions. |

## Subdirectories
None

## For AI Agents

### Working In This Directory
- **Logging is automatic**: Review widgets call `PerformanceLogger.log_reach_detection()` etc. when saving validated results
- **Log files** are stored in `{PROCESSING_ROOT}/performance_logs/`:
  - `segmentation_performance.json`
  - `reach_detection_performance.json`
  - `outcome_performance.json`
- **Log structure**: Each file contains `{"log_version": "1.0.0", "algorithm": "...", "entries": [...], "aggregate": {...}}`
- **Entry fields**: `video_id`, `logged_at` (ISO timestamp), `validator` (username), `algorithm_version`, metrics dict, raw comparison details
- **Aggregate stats**: Computed on every append - mean/std accuracy, precision/recall, total counts
- **Validator username**: Auto-detected via `os.getlogin()` or environment variables
- **Version tracking**: Algorithm version stored per entry (defaults to `detector_version` field in JSON outputs)

### CLI Commands
```bash
# View performance summary
mousereach-perf                          # All algorithms
mousereach-perf --seg                    # Segmentation only
mousereach-perf --reach                  # Reach detection only
mousereach-perf --outcome                # Outcome classification only

# Batch evaluation (re-evaluate all validated videos)
mousereach-perf-eval --data-dir Processing/

# Generate scientific report
mousereach-perf-report --output algorithm_performance.pdf
mousereach-perf-report --format markdown --since 2026-01-01  # Filter by date
```

## Dependencies

### Internal
- `mousereach.config.PROCESSING_ROOT` - Default location for performance logs
- Review widgets (`mousereach.review.*`) - Automatic logging integration

### External
- `numpy` - Statistical computations (mean, std, quantiles)
- `scipy.stats` - T-tests and effect size calculations
- `pandas` - DataFrame operations for report generation
- `matplotlib` / `seaborn` - Performance trend plots (optional, for reports)
- Standard library: `json`, `os`, `pathlib`, `datetime`, `dataclasses`, `typing`

## Typical Workflow

1. **Human validates video**: Use `mousereach-review-reach` or outcome review GUI
2. **Automatic logging**: On save, widget calls `logger.log_reach_detection(video_id, algo_data, human_data)`
3. **Metrics computed**: `compute_reach_metrics()` compares algo vs human, returns ReachMetrics dataclass
4. **Entry appended**: New entry added to `reach_detection_performance.json`
5. **Aggregate updated**: Mean precision/recall recalculated across all entries
6. **View progress**: Run `mousereach-perf` to see updated statistics

**Manual Batch Evaluation** (for ground truth files):
```bash
# Evaluate all videos with GT files against algorithm outputs
mousereach-perf-eval --data-dir dev_SampleData/
# This adds entries with log_source="batch_eval" instead of "validation"
```

## Performance Log Example

```json
{
  "log_version": "1.0.0",
  "algorithm": "reach_detection",
  "algorithm_version": "v2.3.1",
  "entries": [
    {
      "video_id": "IH5_20240501_P_1",
      "logged_at": "2026-01-16T14:32:10",
      "log_source": "validation",
      "validator": "friedrichl",
      "algorithm_version": "v2.3.1",
      "algo_count": 45,
      "human_count": 43,
      "metrics": {
        "precision": 0.93,
        "recall": 0.95,
        "f1": 0.94,
        "n_matched": 41,
        "n_missed": 2,
        "n_extra": 3,
        "n_corrected": 5,
        "mean_start_error": 4.2,
        "mean_end_error": 3.8
      },
      "segment_metrics": [...]
    }
  ],
  "aggregate": {
    "n_videos": 1,
    "mean_precision": 0.93,
    "mean_recall": 0.95,
    "mean_f1": 0.94,
    "std_f1": 0.0,
    "total_missed": 2,
    "total_extra": 3,
    "total_corrected": 5,
    "last_updated": "2026-01-16T14:32:10"
  }
}
```

## Integration with Evaluation Module

**Key Difference**:
- **`eval/`**: Compare algorithm vs ground truth (batch evaluation of test datasets)
- **`performance/`**: Track algorithm vs human corrections (real-world validation workflow)

Both modules share similar metric computations (`metrics.py`) but serve different purposes:
- Evaluation: One-time analysis for algorithm development
- Performance: Continuous tracking for production monitoring

<!-- MANUAL: -->
