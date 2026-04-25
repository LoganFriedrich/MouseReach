# Segmentation Phase -- Improvement Process

## What This Is

This directory holds the segmentation-specific logic for the MouseReach
Improvement Process. The segmenter splits each video into ~20 pellet
presentation segments by detecting tray-advance events from DLC-tracked
apparatus corners (SABL, SABR, SATL, SATR).

## Artifacts Per Snapshot

Each snapshot in `MouseReach_Pipeline/Improvement_Snapshots/segmentation/`
contains:

| Artifact | Description |
|----------|-------------|
| `manifest.json` | Version, commit hash, metrics summary, inputs list |
| `vault/logic_diagram.md` | Mermaid diagram of the algorithm, viewable in Obsidian |
| `vault/.obsidian/` | Minimal Obsidian config so the vault/ dir opens as a vault |
| `figures/logic_diagram.png` | Rendered PNG of the diagram (if available) |
| `figures/logic_diagram_legend.md` | What the diagram shows, which code it reflects |
| `metrics/` | Quantitative evaluation data (boundary accuracy, pellet-in-segment, etc.) |

## Code Being Diagrammed

- **v2.1.x** (single-SABL): `segmentation/core/segmenter_robust.py`
- **v2.2.0+** (multi-proposer): `segmentation/core/segmenter_multi.py`,
  `proposers.py`, `consensus.py`
- Both versions use helpers from `segmenter_robust.py` (load_dlc,
  get_clean_signal, compute_velocity, detect_anomalies, etc.)

## Key Metrics

- **Primary**: pellet-in-segment containment (does each algo-produced segment
  contain a tracked pellet for >=30 confident frames?)
- **Secondary**: boundary-frame accuracy at +/-5f and +/-10f tolerance
- **Tertiary**: interval CV, anomaly count

## metrics.py -- Boundary Accuracy Module

`metrics.py` computes boundary-accuracy metrics by matching algo-emitted
boundaries against ground-truth boundaries within a configurable frame window.

### What It Computes

For each GT boundary, the closest algo boundary within +/-20 frames is found.
If none exists, that GT boundary is a **miss**. Algo boundaries not matched to
any GT are **phantoms** (false positives). When two GT boundaries claim the
same algo boundary, the closer one wins.

Three boundary subsets are evaluated independently:

- `all`: all 21 boundaries per video.
- `inter_pellet_B2_B20`: boundaries 2-20 (1-indexed). The 19 inter-pellet
  transitions -- these matter most for per-pellet scoring.
- `endpoint_B1_B21`: boundaries 1 and 21 only.

### Output Files (written to `metrics/` in each snapshot)

| File | Contents |
|------|----------|
| `boundary_deltas.csv` | One row per boundary event (matched/miss/phantom) across all videos |
| `per_video.csv` | One row per video with n_matched, n_miss, n_phantom, mean/median delta |
| `scalars.json` | Per-subset summary: delta histogram, miss/phantom counts, mean/median stats |

### How to Run

```python
from mousereach.improvement.segmentation.metrics import compute_segmentation_metrics
from pathlib import Path

scalars = compute_segmentation_metrics(
    gt_dir=Path("Y:/.../gt"),
    algo_dir=Path("Y:/.../outputs_v2.2.0"),
    output_dir=Path("Y:/.../metrics"),
)
```

The function auto-discovers video IDs from GT files. Pass `video_ids=` to
restrict to a subset. Pass `window=` to change the matching tolerance
(default 20 frames).

### How to Add a New Metric

1. Add a pure function in `metrics.py` (no module-level I/O).
2. Wire it into `compute_segmentation_metrics` to write its output file(s)
   to `output_dir` and include summary numbers in the returned scalars dict.
3. Add a test in `test_metrics.py` covering the core logic.
4. Update the subset tags if the new metric uses different boundary groupings.

## To Add a New Snapshot

1. Create a new directory under
   `MouseReach_Pipeline/Improvement_Snapshots/segmentation/` named
   `seg_v{VERSION}_{short_description}` (e.g. `seg_v2.3.0_tray_motion_rescue`).

2. Use the snapshot_io helper:
   ```python
   from mousereach.improvement.lib.manifest import Manifest
   from mousereach.improvement.lib.snapshot_io import write_snapshot, snapshot_dir

   sd = snapshot_dir("segmentation", "seg_v2.3.0_tray_motion_rescue")
   m = Manifest(
       version_id="v2.3.0",
       tag="tray_motion_rescue",
       timestamp="2026-05-01T12:00:00-05:00",
       code_hash="abc1234",
       inputs=["47-GT-video corpus"],
       metrics_summary={"boundaries_at_5f": "980/987"},
       description="Added tray-motion signal as fifth proposer.",
   )
   write_snapshot(sd, m)
   ```

3. Add `vault/logic_diagram.md` with the Mermaid diagram.

4. Add `figures/logic_diagram_legend.md` explaining what the diagram shows.

5. Populate `metrics/` with evaluation CSVs or JSON once the eval harness runs.

6. Optionally update `src/mousereach/docs/algorithm_diagrams/segmenter.mmd`
   with the new version's diagram (dual-home pattern: canonical source in
   docs, frozen copy in the snapshot vault).
