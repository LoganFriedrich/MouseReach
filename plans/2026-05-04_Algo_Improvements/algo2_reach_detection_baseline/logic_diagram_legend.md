# Reach Detector v6.0.0 Logic Diagram Legend

## What This Diagram Shows

The full decision tree for the `nose_engagement_state_machine_v6` reach
detection algorithm as implemented at commit `c66e4bb`. This is the v6
state machine with nose-engagement gating, START_CONFIRM delay, multiple
end-of-reach conditions, post-detection splitting, and optional ML polishing.

## Phases

| Phase | Description | Code Location |
|-------|-------------|---------------|
| Preprocessing | Slit center + segment geometry | `reach_detector.py`: `_get_slit_center()`, `geometry.py`: `compute_segment_geometry()` |
| State Machine | Per-frame nose engagement + hand visibility + reach end detection | `reach_detector.py`: `_is_nose_engaged()`, `_any_hand_visible()`, `detect()` |
| Post-Processing | Min duration, splitting, polishing, apex | `reach_detector.py`: `detect()`, `boundary_refiner.py`, `boundary_polisher.py` |
| Output Assembly | VideoReaches construction | `reach_detector.py`: `detect()`, `save_results()` |

## Key Files (all under src/mousereach/reach/core/)

| File | Role |
|------|------|
| `reach_detector.py` | Full v6 state machine implementation |
| `boundary_refiner.py` | Split long reaches at DLC confidence dips |
| `boundary_polisher.py` | ML-based boundary refinement (optional) |
| `geometry.py` | Shared utilities: load_dlc, load_segments, ruler computation |

## Annotation Convention

- `[T]` = tunable threshold (current value shown; principled to sweep)
- `[F]` = feature-definition-limited (changing it redefines the underlying feature)
- `[S]` = structural constant (physics of the apparatus)

## Design Constraints Worth Preserving

1. **Nose engagement is the primary gate.** Without it, any hand visibility
   event (grooming, posture shifts) triggers a reach. The 25 px threshold
   ensures the mouse is oriented toward the slit.

2. **Multiple end-of-reach conditions** (disappear, confidence drop, retraction,
   return-to-start) handle the diversity of reach endings in real videos.
   The RETRACTION_CONFIRM and BP_SWITCH_GRACE parameters prevent false
   endings from bodypart swaps.

3. **Extent gates are disabled** (v2.5.1-dev). 55.9% of real human-annotated
   reaches have negative signed extent (hand does not cross BOXR). No
   threshold on this 1-D projection cleanly separates real from phantom.

4. **Boundary splitting** at DLC confidence dips helps existence detection
   (+3 matches) at the cost of imprecise split points (~5f early-end).

## Rendering

**Open `vault/` in Obsidian**, view `logic_diagram.md`, export PNG via
Obsidian's export-to-image command.

## Source

- **Commit**: `c66e4bb` (2026-04-23T15:34:46-05:00)
- **Branch**: `feature/reach-detector-v7.1.0`
