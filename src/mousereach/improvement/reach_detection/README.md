# Reach Detection Phase -- Improvement Process

## What This Is

This directory holds reach-detection-specific logic for the MouseReach
Improvement Process. The reach detector identifies individual reaching
attempts within each pellet-presentation segment, using hand/nose/pellet
DLC tracks to detect reach onset, peak extension, and retraction.

## Code Being Diagrammed

- `src/mousereach/reach/core/reach_detector.py` (production)
- `src/mousereach/reach/core/reach_detector_v8.py` (experimental)

## Key Metrics

- **Primary**: reach count accuracy vs GT (per-segment, per-video)
- **Secondary**: reach boundary accuracy (start/end frame tolerance)
- **Tertiary**: false positive rate, false negative rate

## To Add a New Snapshot

1. Create a directory under
   `MouseReach_Pipeline/Improvement_Snapshots/reach_detection/` named
   `reach_v{VERSION}_{short_description}`.

2. Use the snapshot_io helper (same pattern as segmentation -- see
   `improvement/segmentation/README.md` for a worked example).

3. Add `vault/logic_diagram.md` with the Mermaid diagram of the reach
   detection algorithm.

4. Add `figures/logic_diagram_legend.md` and populate `metrics/`.
