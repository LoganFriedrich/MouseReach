# Features Phase -- Improvement Process

## What This Is

This directory holds feature-extraction-specific logic for the MouseReach
Improvement Process. The feature extractor computes per-reach kinematic
metrics (velocity, acceleration, trajectory shape, peak extension, area
under curve, etc.) that feed downstream statistical analyses.

## Code Being Diagrammed

- `src/mousereach/kinematics/core/feature_extractor.py` (production)

## Key Metrics

- **Primary**: feature completeness (fraction of reaches with all features computed)
- **Secondary**: feature stability (test-retest reliability across reprocessing)
- **Tertiary**: correlation with manual measurements

## To Add a New Snapshot

1. Create a directory under
   `MouseReach_Pipeline/Improvement_Snapshots/features/` named
   `feat_v{VERSION}_{short_description}`.

2. Use the snapshot_io helper (same pattern as segmentation -- see
   `improvement/segmentation/README.md` for a worked example).

3. Add `vault/logic_diagram.md` with the Mermaid diagram of the feature
   extraction pipeline.

4. Add `figures/logic_diagram_legend.md` and populate `metrics/`.
