# Outcome Classification Phase -- Improvement Process

## What This Is

This directory holds outcome-classification-specific logic for the
MouseReach Improvement Process. The outcome classifier determines what
happened to each pellet presentation: retrieved, displaced into SA,
displaced outside, or untouched.

## Code Being Diagrammed

- `src/mousereach/outcomes/` (production outcome classification)

## Key Metrics

- **Primary**: outcome accuracy vs GT (per-pellet, per-video)
- **Secondary**: confusion matrix across outcome categories
- **Tertiary**: confidence calibration

## To Add a New Snapshot

1. Create a directory under
   `MouseReach_Pipeline/Improvement_Snapshots/outcome/` named
   `outcome_v{VERSION}_{short_description}`.

2. Use the snapshot_io helper (same pattern as segmentation -- see
   `improvement/segmentation/README.md` for a worked example).

3. Add `vault/logic_diagram.md` with the Mermaid diagram.

4. Add `figures/logic_diagram_legend.md` and populate `metrics/`.
