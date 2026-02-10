<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# reach/analysis/

## Purpose
Ground truth analysis, algorithm evaluation, and rule derivation utilities. Scripts for comparing algorithm detections to human annotations, discovering patterns in GT data, and tuning detection thresholds.

## Key Files
| File | Description |
|------|-------------|
| `DISCOVERED_RULES.md` | Data-driven reach detection rules derived from 321-reach ground truth analysis (20251021_CNT0405_P4). Documents nose engagement threshold (20px), hand visibility patterns, and end detection rules. |
| `evaluate_algorithm.py` | Algorithm evaluation script. Compares detected reaches to ground truth using IoU matching (≥0.3 threshold). Reports precision, recall, F1 score, and frame-level accuracy. |
| `gt_pattern_analysis.py` | Ground truth pattern discovery. Analyzes nose position, hand visibility, velocity at boundaries, inter-reach gaps. Used to derive detection rules. |
| `analyze_reach_boundaries.py` | Deep dive into reach boundary decisions. Analyzes short gaps between consecutive reaches to understand how humans distinguish multiple reaches. |
| `analyze_boundary_errors.py` | Boundary error analysis comparing algorithm vs human-labeled start/end frames. |
| `determine_split_threshold.py` | Statistical analysis to determine optimal duration threshold for splitting long merged reaches. |
| `debug_fragmentation.py` | Diagnoses false positive reaches caused by fragmentation (algorithm splitting single GT reach into multiple). |
| `discrepancy_report.py` | Generates reports on algorithm/GT mismatches for human review. |
| `annotate_discrepancies.py` | Annotates videos with algorithm vs GT comparison overlays. |
| `evaluate_human_touched.py` | Evaluates only on human-corrected reaches (filters out uncorrected algorithm output from GT). |
| `evaluate_full_gt.py` | Full evaluation including all GT reaches regardless of human correction status. |
| `proposed_refinements.md` | Historical record of proposed algorithm improvements (may be outdated). |

## For AI Agents

### CRITICAL: Frame Boundary Accuracy IS Data Quality

**Every frame boundary error corrupts downstream kinematic data.** Reach boundaries define the windows over which kinematics are computed. Wrong boundaries = computing kinematics over non-behavioral frames = noise that contaminates scientific results. When reporting evaluation results, never describe accuracy as "good" - report the error rate and what needs to be fixed. The goal is always exact convergence with human judgment on every reach, every frame.

### Algorithm Evaluation Workflow

**1. Generate Ground Truth (Manual):**
- Run algorithm: `mousereach-detect-reaches -i Processing/`
- Open review GUI: `mousereach-gui` (loads `_reaches.json`)
- Human adds/removes/corrects reaches, saves as `_reach_ground_truth.json`

**2. Derive Rules from GT:**
```bash
python -m mousereach.reach.analysis.gt_pattern_analysis
```
- Outputs statistics on nose position, hand visibility, velocity patterns
- Key findings documented in `DISCOVERED_RULES.md`

**3. Implement Rules in Algorithm:**
- Update `reach_detector.py` with new thresholds
- Increment VERSION
- Document changes in docstring

**4. Evaluate Performance:**
```bash
python -m mousereach.reach.analysis.evaluate_algorithm
```
- Computes precision, recall, F1 score
- Analyzes start/end frame accuracy (target: ±5 frames)
- Identifies false positives (extra detections) and false negatives (missed reaches)

**5. Iterate:**
- Low precision → too many false positives (tighten thresholds, improve end detection)
- Low recall → missing reaches (lower thresholds, improve start detection)
- Large frame errors → boundary detection needs refinement

### Key Metrics

**Match Criteria:**
- IoU (Intersection over Union) ≥ 0.3 between detected and GT reach
- Greedy matching: best IoU wins

**Target Performance (v3.2.0):**
- Precision: >90% (few false positives)
- Recall: >90% (few missed reaches)
- F1 Score: >90%
- Start frame error: Median <5 frames
- End frame error: Median <5 frames

**Current Known Issues:**
- Long reaches (>50 frames) prone to premature end detection (tracking dropout)
- Consecutive reaches with brief gaps (<5 frames) may merge
- Low DLC confidence causes missed reaches

### Modifying Evaluation Scripts

**When adding new analysis:**
1. Use existing GT loader pattern from `evaluate_algorithm.py`
2. Extract DLC data with `load_dlc_data()` from `gt_pattern_analysis.py`
3. Print results to console (scripts are command-line tools, not library functions)

**Ground Truth Format:**
- Same as `_reaches.json` but with human corrections
- Additional fields: `source`, `human_corrected`, `original_start`, `original_end`
- Segment-level `human_verified` flag indicates human reviewed entire segment

**Common Analysis Patterns:**
- Load GT + DLC data
- Iterate through reaches extracting frame-level features
- Compute statistics (mean, median, percentiles)
- Print formatted report to console

## Dependencies
- **pandas, numpy** - Data analysis
- **DeepLabCut h5 files** - Pose tracking data
- **Ground truth JSON** - Human-annotated reaches (`*_reach_ground_truth.json`)
- **mousereach.reach.core** - Algorithm implementation to evaluate
- **mousereach.config** - Path configuration (Paths.PROCESSING)

<!-- MANUAL: -->
