<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# segmentation/core

## Purpose
Core segmentation algorithms for detecting 21 trial boundaries in mouse reaching videos using DeepLabCut tracking data. Includes batch processing, quality triage, and pipeline advancement logic.

## Key Files
| File | Description |
|------|-------------|
| `segmenter_robust.py` | **PRIMARY ALGORITHM** - Multi-strategy boundary detection using SABL anchor crossing detection with fallback methods |
| `batch.py` | Batch processing wrapper with validation status classification (auto_approved/needs_review) |
| `triage.py` | Result classification and file management (DEPRECATED in unified pipeline v2.3+) |
| `advance.py` | Move validated files to next pipeline stage with logging and index updates |

## For AI Agents

### Core Algorithm (segmenter_robust.py)

**Scientific Basis:**
During single-pellet reaching task, automated dispenser presents 21 pellets in sequence. Each presentation causes scoring area (SA) frame to move as tray advances. Algorithm detects trial boundaries by tracking SABL (bottom-left SA corner) crossing box center position.

**Detection Strategy (3-tier):**
1. **Primary:** SABL centered crossing with velocity threshold
   - Smooth SABL x-position (5-frame median filter)
   - Detect positive crossings of box center (BOXL-BOXR midpoint)
   - Require velocity >0.03 ruler units/frame (1.2 after smoothing)
   - Validate with SABR/SATL/SATR correlated motion

2. **Secondary:** Multi-anchor validation
   - Check agreement across all SA points (SABL, SABR, SATL, SATR)
   - Window: ±3 frames for alignment tolerance
   - Boosts confidence when multiple anchors agree

3. **Fallback:** Motion peak detection
   - Aggregate motion signal across SA points
   - Peak detection with 20-second minimum spacing
   - Lower confidence (0.5) than primary

**Key Parameters:**
| Parameter | Value | Unit | Rationale |
|-----------|-------|------|-----------|
| VELOCITY_THRESHOLD | 0.03 (primary) / 0.8 (fallback) | ruler/frame | Filters noise while catching slow presentations |
| CROSSING_WINDOW | 3 | frames | Alignment tolerance for multi-anchor validation |
| MIN_INTERVAL | 300 | frames | ~5 sec at 60fps, prevents duplicate detection |
| MAX_INTERVAL | 1200 | frames | ~20 sec at 60fps, flags missed boundaries |
| EXPECTED_BOUNDARIES | 21 | count | Standard pellet count per session |
| EXPECTED_INTERVAL | 1839 | frames | ~30.65 sec at 60fps (typical trial spacing) |

**Quality Assessment:**
- Reference stability: BOXL/BOXR std <5px (they should be stationary)
- SA coverage: % frames with likelihood >0.5
- Interval CV: std/mean of boundary intervals (<0.10 = good, <0.30 = acceptable)
- Anomaly classification: CRITICAL/WARNING/INFO severity levels

**Output JSON Structure:**
```json
{
  "segmenter_version": "2.1.0",
  "boundaries": [21 frame numbers],
  "overall_confidence": 0.0-1.0,
  "reference_quality": "good/suspect/bad",
  "sa_coverage": {SABL, SABR, SATL, SATR percentages},
  "detection": {n_primary, n_fallback, methods[], confidences[]},
  "intervals": {mean_frames, std_frames, cv, mean_seconds},
  "anomalies": ["text descriptions"],
  "anomaly_details": [{text, severity, explanation, boundaries_affected}],
  "boundary_flags": {boundary_idx: {issues[], needs_check}}
}
```

### Batch Processing (batch.py)

**Unified Pipeline (v2.3+):**
- All files stay in single Processing/ folder
- validation_status stored in JSON metadata ("auto_approved", "needs_review", "validated")
- No folder-based triage (status determines review queue)

**Auto-approval criteria:**
- Exactly 21 boundaries detected
- Interval CV <0.30 (reasonable consistency)
- No CRITICAL anomalies
- High primary candidate coverage

**Processing flow:**
1. Find DLC .h5 files in input_dir
2. Run segment_video_robust() on each
3. Save *_segments.json with validation_status
4. Move all associated files to Processing/ folder
5. Update pipeline index

### Triage (triage.py)

**⚠️ DEPRECATED:** Folder-based triage is deprecated in v2.3+. Use unified pipeline with JSON-based status tracking instead.

**Classification logic (for reference):**
- `good` → auto-approve (21 boundaries, CV <0.10, high confidence, no warnings)
- `warning` → needs review (some concerns but potentially usable)
- `failed` → critical issues (cannot use without major fixes)

**INFO-level anomalies do NOT prevent auto-approval** (they indicate normal algorithm adaptation like "Used lower velocity threshold").

**Anomaly severity classification:**
- CRITICAL: Segmentation failed, primary method unavailable, <19 boundaries, fallback used
- WARNING: Late start, interpolated boundaries, timing drift, stuck tray
- INFO: Lower velocity threshold, pre-trial movements (normal variations)

**Unsupported tray types:**
- E (Easy) and F (Flat) trays are NOT supported
- Only P (Pillar) trays work with this algorithm
- Use reject_unsupported_tray_type() to move them to NAS

### Advancement (advance.py)

**Purpose:** Mark validated files as ready for reach detection (v2.3+ updates JSON validation_status).

**File bundle includes:**
- video file (.mp4/.avi/.mkv)
- *_segments.json
- *_seg_validation.json
- DLC outputs (.h5, .csv, .pickle)
- Preview video (*_preview.mp4) if exists

**Logging:** Records timestamp, user, video_id, action to advance_log.txt

**Index updates:** Tracks file moves and validation status changes

### Modifying This Code

**When changing segmentation algorithm:**
- Bump SEGMENTER_VERSION in segmenter_robust.py
- Update SEGMENTER_ALGORITHM identifier
- Adjust thresholds in KEY PARAMETERS section
- Test against ground truth files (see __main__ section)
- Update anomaly classification logic if adding new detection methods

**When changing validation criteria:**
- Update classify_segments() thresholds in triage.py
- Adjust Thresholds config in mousereach.config
- Consider impact on auto-approval rate (aim for >70% auto-approved on good videos)

**When adding new bodyparts:**
- Update BODYPARTS list in segmenter_robust.py
- Update CRITICAL_POINTS if new points affect segmentation
- Retrain DLC model and validate on test set

**Common pitfalls:**
- Reference instability (BOXL/BOXR moving) breaks centered crossing detection
- Low SABL coverage (<80%) degrades boundary detection accuracy
- Manual pellet placement disrupts expected interval patterns
- Camera vibration can trigger false positives (check velocity threshold)
- INFO anomalies are OK - don't treat them as failures

**Ground truth validation:**
- Test files in __main__ section reference ground truth boundaries
- Target: <50 frames error for >95% of boundaries
- Mean absolute error: <20 frames (~0.33 seconds at 60fps)

## Dependencies

**External:**
- pandas (DLC data loading)
- numpy (signal processing)
- scipy (signal filtering, peak detection)

**Internal:**
- mousereach.config (Paths, Thresholds, FilePatterns)
- mousereach.index.PipelineIndex (pipeline state tracking)
- mousereach.video_prep.compress (preview video creation)

<!-- MANUAL: -->
