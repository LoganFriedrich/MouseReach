# MouseReach Improvement Process

## What This Is

A framework for tracking algorithm improvements across the MouseReach
pipeline. Each improvement is captured as a **snapshot** -- a self-contained
directory with the algorithm's logic diagram, evaluation metrics, and
metadata -- so that progress is visible, reproducible, and presentable.

## Architecture

```
src/mousereach/improvement/          <-- CODE (this directory)
  lib/
    manifest.py                      Manifest dataclass + JSON serialization
    snapshot_io.py                    Read/write snapshots, resolve paths
    palette.py                       Shared colors for figures
    pptx_export.py                   (stub) PowerPoint export helpers
    vault_template/                  Starter Obsidian vault copied into new snapshots
  segmentation/                      Segmentation phase logic
  reach_detection/                   Reach detection phase logic
  outcome/                           Outcome classification phase logic
  features/                          Feature extraction phase logic

MouseReach_Pipeline/Improvement_Snapshots/   <-- DATA (output snapshots)
  segmentation/
    seg_v2.1.3_phantom_first_post_validation/
    seg_v2.2.0_multi_proposer/
    ...
  reach_detection/
  outcome/
  features/
```

Code lives here (in the git-tracked tool directory). Snapshot data
accumulates in `MouseReach_Pipeline/Improvement_Snapshots/` (data-only,
not in git).

## How to Use

### Create a snapshot programmatically

```python
from mousereach.improvement.lib.manifest import Manifest
from mousereach.improvement.lib.snapshot_io import write_snapshot, snapshot_dir

sd = snapshot_dir("segmentation", "seg_v2.3.0_tray_motion_rescue")
m = Manifest(
    version_id="v2.3.0",
    tag="tray_motion_rescue",
    timestamp="2026-05-01T12:00:00-05:00",
    code_hash="abc1234",
    pipeline_versions={"segmenter": "2.3.0", "reach_detector": "7.0.0"},
    inputs=["47-GT-video corpus"],
    metrics_summary={"boundaries_at_5f": "980/987"},
    artifacts=["vault/logic_diagram.md", "figures/logic_diagram.png"],
    description="Added tray-motion signal as fifth proposer.",
)
write_snapshot(sd, m)
```

### Read back a snapshot

```python
from mousereach.improvement.lib.snapshot_io import read_snapshot, snapshot_dir

sd = snapshot_dir("segmentation", "seg_v2.1.3_phantom_first_post_validation")
m = read_snapshot(sd)
print(m.version_id, m.tag, m.metrics_summary)
```

### List all snapshots for a phase

```python
from mousereach.improvement.lib.snapshot_io import list_snapshots

for sd in list_snapshots("segmentation"):
    print(sd.name)
```

### View a snapshot in Obsidian

Open `MouseReach_Pipeline/Improvement_Snapshots/<phase>/<snapshot>/vault/`
as an Obsidian vault. The `.obsidian/` directory is pre-created so Obsidian
recognizes it immediately. Mermaid diagrams render inline.

## How to Generate Figures From a Snapshot

**Figures are NOT hand-rolled per-session.** Each phase has a runner
script that produces the canonical figures (Sankey, violin, summary
table, etc.) from a snapshot's `metrics/scalars.json`. Runners use the
shared palette in `lib/palette.py` and the layout patterns developed
across the project so figures stay consistent.

Per-phase runners:
- Outcome: `outcome/_run_notebooks.py` -- `run_sankey()`,
  `run_interaction_frame_violin()`, `run_summary_table()`.
- (Other phases follow the same pattern -- check
  `<phase>/_run_notebooks.py` or `<phase>/*.ipynb`.)

Usage from Python:
```python
from pathlib import Path
from mousereach.improvement.outcome._run_notebooks import run_sankey
run_sankey(Path("Y:/.../Improvement_Snapshots/outcome/outcome_vX.Y.Z_dev_<tag>"))
```

The runner reads `metrics/scalars.json` and writes
`figures/sankey.png` + `figures/sankey_legend.md` into the snapshot.

**If you find yourself writing matplotlib code for a Sankey or any
other phase figure, STOP and use the runner script.** Curated
renderers exist for consistency, color palette, label-collision
avoidance, and paper-quality output. Hand-rolled figures are not a
substitute.

## How to Add a New Phase

1. Create `src/mousereach/improvement/<phase_name>/` with `__init__.py`
   and `README.md` following the pattern of existing phases.

2. Create `MouseReach_Pipeline/Improvement_Snapshots/<phase_name>/` for
   snapshot accumulation.

3. Document the phase's key metrics, code being diagrammed, and snapshot
   structure in the README.

## How to Add a New Figure Type

1. Add a rendering function in a new module under the relevant phase
   directory (e.g. `improvement/segmentation/render_sankey.py`).

2. Import palette colors from `improvement/lib/palette.py`.

3. Output figures to the snapshot's `figures/` directory.

4. Update the snapshot's `manifest.json` artifacts list.

## Annotation Conventions for Diagrams

Every decision node and threshold in a logic diagram must be annotated:

| Tag | Meaning | Example |
|-----|---------|---------|
| `[T]` | Tunable threshold -- principled to sweep | velocity > 0.8 px/frame |
| `[F]` | Feature-definition-limited -- changing it redefines the feature | center_target = 2.5 px |
| `[S]` | Structural constant -- physics of the apparatus | 20 pellets per tray |
