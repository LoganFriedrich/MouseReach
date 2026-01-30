<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# core

## Purpose
Core algorithm implementation for pellet outcome detection. Classifies pellet presentation outcomes (retrieved, displaced, untouched) using geometric tracking and multi-stage progressive validation.

## Key Files
| File | Description |
|------|-------------|
| `pellet_outcome.py` | Main detection algorithm - classifies pellet outcomes using position tracking, visibility changes, paw proximity, and eating signatures |
| `geometry.py` | Per-segment geometric calculations - computes ruler-normalized coordinates, ideal pillar position from 55° triangle, distance measurements |
| `batch.py` | Batch processing logic - finds matching file sets (DLC, segments, reaches), processes multiple videos, handles skip patterns |
| `triage.py` | Sort results into review queues - checks anomalies, loads ground truth files, determines which videos need human review (DEPRECATED - unified pipeline uses JSON status) |
| `advance.py` | Move validated outcomes to next pipeline stage - marks files as validated, moves associated files, updates pipeline index |

## For AI Agents

### Algorithm Details

**Detection Rules (from pellet_outcome.py)**:
- **Retrieved (R)**: Pellet visible at start, disappears near pillar (< 0.20 ruler units), eating signature detected
- **Displaced in SA (D)**: Pellet moves > 0.25 ruler from pillar, stays visible, remains in scoring area bounds
- **Displaced Outside (O)**: Pellet moves outside scoring area or disappears away from pillar
- **Untouched (U)**: Pellet position unchanged (< 0.25 ruler movement) throughout segment
- **Uncertain**: Ambiguous tracking quality or conflicting signals

**Key Thresholds (ruler units = SABL-SABR distance = 9mm)**:
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `ON_PILLAR_THRESHOLD` | 0.20 | ~1.8mm - pellet on pillar |
| `DISPLACED_THRESHOLD` | 0.25 | ~2.25mm - minimum for displaced |
| `PILLAR_PERP_DISTANCE` | 0.944 | Geometric pillar-to-SA distance (55° triangle) |
| `CONFIDENCE_THRESHOLD` | 0.6 | Min DLC likelihood for pellet |

**Multi-Stage Progressive Validation**:
1. **Stage 0**: Early retrieval detection (paw grab detection)
2. **Stage 1**: Initial hypothesis (visibility + position)
3. **Stage 2**: Feature validation (eating signature, paw proximity, sustained displacement)
4. **Stage 3**: Temporal consistency (behavior patterns over time)
5. **Stage 4**: Final confidence scoring and review flagging

**Geometry System**:
- All measurements normalized to "ruler" (SABL-SABR distance = 9mm)
- Camera zoom varies per video - pixel values NOT transferable
- Ideal pillar position computed from 55° isosceles triangle geometry
- Per-segment calibration using median SABL/SABR during stable periods

### Modifying This Code

**Critical Dependencies**:
- Requires 18 DLC bodyparts including: Pellet, Pillar, SABL, SABR, SATR, SATL, RightHand, RHLeft, RHOut, RHRight, Nose, BOXR
- Segment boundaries must be validated before outcome detection
- Reach detection results optional but improve causal attribution

**Threshold Tuning**:
- Changing `ON_PILLAR_THRESHOLD` / `DISPLACED_THRESHOLD` affects classification sensitivity
- v2.4.4 reduced thresholds based on ground truth analysis - test carefully before changing
- Ground truth validation widget: `outcomes/review_widget.py`

**Version Tracking**:
- Algorithm version stored in `pellet_outcome.py` VERSION constant (currently v2.4.4)
- Each output JSON includes `detector_version` for backward compatibility
- Major changes should increment version and document in VALIDATION HISTORY section

**Performance Notes**:
- Batch processing supports `skip_if_exists` patterns to avoid reprocessing
- Pipeline index tracking for fast startup (see advance.py)
- Triage functionality is DEPRECATED - unified pipeline keeps files in Processing/ with JSON status

**Common Pitfalls**:
- Pellet tracking degrades when occluded by paw (algorithm handles this via visibility drops)
- Tray wobble can mimic displacement (algorithm checks relative movement vs SA reference points)
- Multiple reaches in segment may have ambiguous causal attribution (uses closest reach before interaction)

## Dependencies
- **Upstream**: Requires validated segmentation (`_segments.json`) and DLC tracking (`.h5`)
- **Optional**: Reach detection results (`_reaches.json`) for causal reach attribution
- **Downstream**: Outputs feed into review widgets and final export
- **External**: pandas, numpy, pathlib, mousereach.config (for paths), mousereach.index (for pipeline tracking)

<!-- MANUAL: -->
