<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# reach/core/

## Purpose
Core reach detection algorithms and processing logic. Contains the main ReachDetector state machine, geometric calibration, batch processing, quality control, and pipeline advancement utilities.

## Key Files
| File | Description |
|------|-------------|
| `reach_detector.py` | Main reach detection algorithm (v3.2.0). State machine for identifying reach start, apex, and end frames using DeepLabCut hand/nose tracking. Implements data-driven rules from ground truth analysis. |
| `geometry.py` | Per-segment geometric calibration. Converts pixel measurements to ruler units (SABL-SABR = 9mm physical ruler). Computes ideal pillar position from 55° triangle geometry. |
| `batch.py` | Batch processing for reach detection. Finds DLC/segment file pairs, validates segmentation status, processes multiple videos. Enforces validation gates (only processes validated/auto-approved files). |
| `triage.py` | **DEPRECATED** Quality control sorting. Checks for anomalies (wrong segment count, suspiciously high/low reach counts). Now superseded by unified validation_status in JSON. |
| `advance.py` | Pipeline advancement. Marks reaches as validated, moves files between pipeline folders, updates validation metadata. |

## For AI Agents

### Algorithm Details

**ReachDetector State Machine (reach_detector.py):**
- **States:** IDLE → ENGAGED → REACHING → IDLE
- **Reach Start:** Nose within 25px of slit center + ANY hand point likelihood ≥ 0.5
- **Reach End:** Hand disappears (2+ consecutive frames below threshold) OR hand retracts >40% of extension
- **Key Thresholds:**
  - `HAND_LIKELIHOOD_THRESHOLD = 0.5` (matches review widget display)
  - `NOSE_ENGAGEMENT_THRESHOLD = 25` pixels
  - `MIN_REACH_DURATION = 2` frames
  - `SPLIT_THRESHOLD_FRAMES = 25` frames (95th percentile of GT duration)
  - `GAP_TOLERANCE = 2` frames (merges brief tracking dropouts)

**Post-processing:**
1. Filter negative extent reaches (hand never passed slit)
2. Merge reaches separated by <2 frames (tracking dropout)
3. Split long reaches (>25 frames) with confidence drop patterns

**Geometry (geometry.py):**
- Physical ruler = 9mm (SABL-SABR distance)
- Pillar position computed from 55° isoceles triangle
- All reach extent measurements normalized to ruler units (invariant to camera zoom)

**Batch Processing (batch.py):**
- **Validation Gate:** Only processes files with `validation_status` = "validated" or "auto_approved"
- Blocks files with `validation_status` = "needs_review"
- Searches for segment files in priority order: `_segments.json`, `_seg_validation.json`, `_segments_v2.json`, `_seg_ground_truth.json`
- Skip patterns: `skip_if_exists=["*_reach_ground_truth.json"]` prevents reprocessing GT files

### Modifying This Code

**When changing reach_detector.py:**
1. Update `VERSION` string (semantic versioning)
2. Run evaluation: `python -m mousereach.reach.analysis.evaluate_algorithm`
3. Check precision/recall against ground truth (target: >90% F1 score)
4. Document threshold changes in docstring and DISCOVERED_RULES.md

**When changing thresholds:**
- Match `HAND_LIKELIHOOD_THRESHOLD` to review widget display threshold (currently 0.5)
- Nose engagement threshold derived from GT 95th percentile (25px)
- Duration/split thresholds from GT statistics (see analysis/DISCOVERED_RULES.md)

**Testing:**
- Ground truth video: `20251021_CNT0405_P4` (321 reaches, validated)
- Run: `mousereach-detect-reaches -i Processing/`
- Compare: `python -m mousereach.reach.analysis.evaluate_algorithm`

**Common Pitfalls:**
- Changing likelihood thresholds without updating review widget → inconsistent human/algo labels
- Splitting/merging logic changes can cause cascading ID reassignment bugs
- Geometry calculations assume stable SABL/SABR tracking (check `stable_margin` in segment)

## Dependencies
- **pandas, numpy** - Data processing
- **DeepLabCut** - Pose tracking data format (.h5 files)
- **mousereach.reach.analysis** - Evaluation and ground truth comparison
- **mousereach.index** - Pipeline index updates (optional, doesn't fail detection)
- **Segmentation output** - Requires validated segment boundaries from Step 2

<!-- MANUAL: -->
