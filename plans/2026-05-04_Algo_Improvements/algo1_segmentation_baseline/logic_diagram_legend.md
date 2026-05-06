# Segmenter v2.1.3 Logic Diagram Legend

## What This Diagram Shows

The full decision tree for the `sabl_centered_crossing_v2` segmentation
algorithm as implemented at commit `8e43976` in `segmenter_robust.py`.

## Phases

| Phase | Description | Code Location |
|-------|-------------|---------------|
| Phase 1 | Reference and SA coverage quality assessment | `assess_reference_quality()`, `assess_sa_quality()` |
| Phase 2 | Primary SABL centered-crossing candidate detection | `find_centered_crossings()`, `_pick_best_frame()` |
| Phase 2b | Fallback: aggregate motion peaks (if primary yields < 10) | `find_motion_peaks()` |
| Phase 3 | Fit grid to candidates (drop/fill/interpolate to 21) | `fit_grid_to_candidates()` |
| Phase 4 | Validate and correct: phantom removal + endpoint projection | `_validate_and_correct_boundaries()` |
| Phase 5 | Anomaly detection and emit `_segments.json` | `detect_anomalies()`, `save_segmentation()` |

## Annotation Convention

- `[T]` = tunable threshold (current value shown; principled to sweep)
- `[F]` = feature-definition-limited (changing it redefines the underlying feature)
- `[S]` = structural constant (physics of the apparatus)

## Source Code

- **File**: `src/mousereach/segmentation/core/segmenter_robust.py`
- **Commit**: `8e43976` (2026-04-23T20:00:56-05:00)
- **Algorithm**: `sabl_centered_crossing_v2`
- **Architecture**: single-SABL candidate producer with SABR/SATL/SATR as validators

## Key Design Decisions

1. SABL is the sole candidate producer. Other SA corners validate only.
2. Phantom removal runs BEFORE endpoint projection (the "phantom-first"
   in the version name). This prevents dropping real B21 when a phantom
   and a missing B1 co-occur.
3. Bracket check in phantom test prevents dropping legitimate rapid-fire
   advance boundaries.
