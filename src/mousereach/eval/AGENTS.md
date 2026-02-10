<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# eval

## Purpose
Algorithm evaluation framework for comparing MouseReach pipeline outputs (segmentation, reach detection, outcome classification) against human-annotated ground truth files. Generates detailed performance reports with error categorization, statistical metrics (precision, recall, F1), and actionable recommendations for algorithm improvement. Supports frame-level tolerance matching and per-video analysis.

## Key Files
| File | Description |
|------|-------------|
| `base.py` | Abstract base class (BaseEvaluator) defining evaluation workflow: load GT/algo files, compare, categorize errors, generate reports. Includes ErrorCategory and EvalResult dataclasses. |
| `seg_evaluator.py` | SegmentationEvaluator: Compares boundary frame detection with tolerance matching. Computes accuracy, missed/extra boundaries, timing errors. |
| `reach_evaluator.py` | ReachEvaluator: Matches reach sequences per segment. Computes precision/recall/F1, timing errors (start/apex/end), extent errors. |
| `outcome_evaluator.py` | OutcomeEvaluator: Compares pellet outcome classifications (retrieved, displaced, etc.). Generates confusion matrix and per-class metrics. |
| `cli.py` | Command-line interface for running evaluations. Supports single-algorithm or batch evaluation with customizable tolerance and report export. |
| `__init__.py` | Module exports for all evaluator classes and result types. |

## Subdirectories
None

## For AI Agents

### CRITICAL: Frame Boundary Accuracy IS Data Quality

**Every frame boundary error corrupts downstream kinematic data.** When reporting evaluation results, never describe accuracy as "good" or "excellent." Always report the remaining error rate and what needs to be done to reduce it. 88% accuracy means 12% of kinematic data is contaminated with non-behavioral frames. The goal is always convergence toward exact human agreement on every reach, every frame. Every mismatch is a bug to understand and fix, not a tolerable margin.

### Working In This Directory
- **Ground truth files** must follow naming convention: `*_ground_truth.json` for the GT pattern to match
- **Tolerance parameter** controls frame-level matching strictness (default: 5 for seg, 10 for reach, 15 for outcome)
- **Error categories** are pre-defined in each evaluator's `_init_error_categories()` method
- Evaluators expect specific JSON structures:
  - Segmentation: `{"boundaries": [frame1, frame2, ...]}` or `{"segments": [{"start": frame, "end": frame}]}`
  - Reaches: `{"segments": [{"reaches": [{"start": f, "apex": f, "end": f}]}]}`
  - Outcomes: `{"segments": [{"outcome": "retrieved|displaced_sa|..."}]}`
- Video ID extraction removes suffixes like `_ground_truth`, `_segments`, etc. (see `extract_video_id()` in base.py)

### CLI Commands
```bash
# Evaluate single algorithm
mousereach-eval --seg [path]              # Segmentation
mousereach-eval --reach [path]            # Reach detection
mousereach-eval --outcome [path]          # Outcome classification

# Batch evaluation
mousereach-eval --all [path]              # All 3 algorithms sequentially

# Custom tolerance and output
mousereach-eval --reach --tolerance 15 --output report.txt

# Separate GT and algorithm directories
mousereach-eval --seg --gt-dir GT/ --algo-dir results/
```

## Dependencies

### Internal
- `mousereach.config.Paths` - Access to processing root directory for default paths

### External
- `numpy` - Numerical operations for timing error statistics
- `pathlib` - File path handling
- Standard library: `json`, `dataclasses`, `typing`, `sys`

## Typical Workflow

1. **Prepare Ground Truth**: Manually annotate videos, save as `{video_id}_ground_truth.json`
2. **Run Algorithm**: Process same videos through MouseReach pipeline
3. **Evaluate**: `mousereach-eval --all path/to/data/`
4. **Analyze Report**: Review error categories, metrics, recommendations
5. **Iterate**: Adjust algorithm parameters based on error patterns
6. **Re-evaluate**: Track improvement over time

## Output Format

**Console Report:**
```
=== REACH DETECTION EVALUATION ===
Videos evaluated: 10
Successful: 10

Overall Metrics:
  Precision: 0.92 (92% of algo reaches were correct)
  Recall: 0.88 (88% of GT reaches were detected)
  F1 Score: 0.90

Error Categories:
  Missed Reaches: 12 (algorithm failed to detect)
  Extra Reaches: 8 (false positives)
  Timing Corrections: 15 (detected but start/end adjusted)

RECOMMENDATIONS:
  - Improve reach start detection (mean error: 5.2 frames)
  - Investigate missed reaches in segments 15-21 (fatigue?)
```

<!-- MANUAL: -->
