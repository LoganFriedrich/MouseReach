<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# dlc/core

## Purpose
Core DeepLabCut processing algorithms for batch inference, output validation, and quality assessment of mouse tracking data.

## Key Files
| File | Description |
|------|-------------|
| `batch.py` | DeepLabCut batch inference wrapper with YAML config fixing, video queuing, and automated file movement to Processing/ |
| `quality.py` | Quality validation for DLC tracking output - checks reference point stability, SA anchor coverage, and hand point tracking confidence |

## For AI Agents

### Algorithm Details
**batch.py:**
- Wraps DeepLabCut analyze_videos() for batch processing
- Fixes common YAML issues (strips broken video_sets section)
- Filters unsupported tray types (E/F - only P/Pillar supported)
- Creates preview videos for memory-efficient review widgets
- Moves completed files (video + DLC outputs) to Processing/ folder (v2.3+ architecture)

**quality.py:**
- Expected 18 bodyparts (see BODYPARTS list in source)
- Critical points: SABL, SABR, RightHand, BOXL, BOXR
- Reference stability: BOXL/BOXR std should be <5px (they're fixed points)
- Quality grades: good/fair/poor/failed based on likelihood and coverage
- Default likelihood threshold: 0.6 (matches DLC pcutoff)

### Quality Metrics Thresholds
| Metric | Good | Fair | Poor |
|--------|------|------|------|
| BOXL/BOXR std (px) | <3 | <5 | ≥5 |
| Mean likelihood | >0.8 | >0.7 | <0.7 |
| Critical point coverage | >85% | >70% | <70% |

### Modifying This Code
**When changing batch processing:**
- DLC config path is user-selected (not hardcoded) via file dialog
- fix_config_yaml() strips video_sets to avoid multi-line YAML parsing errors
- Preview video creation can be disabled but saves memory in Napari widgets
- Pipeline paths use environment variables (MouseReach_PROCESSING_ROOT)

**When changing quality checks:**
- Adjust CRITICAL_POINTS list if DLC model changes
- Reference point stability is critical - unstable BOXL/BOXR means camera shake
- SA coverage thresholds directly impact segmentation success rate
- Ground truth validation must use likelihood_threshold=0.6 to match filtering

**Common pitfalls:**
- DLC config.yaml with broken video_sets causes parse failures (fixed by fix_config_yaml)
- Tray types E/F are unsupported and should be rejected before DLC processing
- Preview videos must exist in same folder as source for review widget efficiency

## Dependencies
**External:**
- deeplabcut (optional - only needed for actual inference)
- pandas, numpy

**Internal:**
- mousereach.config (Paths configuration)
- mousereach.video_prep.compress (preview video creation)
- mousereach.index.PipelineIndex (pipeline state tracking)

<!-- MANUAL: -->
