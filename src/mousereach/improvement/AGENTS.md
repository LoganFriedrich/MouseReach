# improvement/ -- AGENTS.md

**This is CODE.** Algorithm-improvement-process tooling for MouseReach.
Snapshot DATA accumulates in `Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\`.

## Critical: do NOT hand-roll figures from this snapshot data

Each phase has a canonical figure-runner script. Use it. Hand-rolling
matplotlib figures from snapshot scalars defeats the curated palette,
layout, label-collision-avoidance, and paper-quality conventions the
user invested in.

Phase figure runners (use these, not your own matplotlib):

| Phase | Runner | Functions |
|---|---|---|
| outcome | `outcome/_run_notebooks.py` | `run_sankey`, `run_interaction_frame_violin`, `run_summary_table` |
| segmentation | (see `segmentation/*.ipynb` / runner if present) | -- |
| reach_detection | (see `reach_detection/` / runner if present) | -- |
| features | (see `features/` / runner if present) | -- |

Quick usage:
```python
from mousereach.improvement.outcome._run_notebooks import run_sankey
run_sankey(Path("Y:/.../Improvement_Snapshots/outcome/outcome_vX.Y.Z_<tag>"))
```

Reads `<snapshot>/metrics/scalars.json`, writes `<snapshot>/figures/sankey.png`
+ `sankey_legend.md`. Consistent palette via `lib/palette.py`.

## Shared utilities

| Module | What it provides |
|---|---|
| `lib/palette.py` | `OUTCOME_COLORS`, `OUTCOME_CLASS_ORDER`, `OUTCOME_VERDICT_*`, phase colors, segmentation/reach palettes |
| `lib/manifest.py` | `Manifest` dataclass for snapshot manifest.json |
| `lib/snapshot_io.py` | `write_snapshot`, `read_snapshot`, path resolution |

## Adding a new figure

If a runner doesn't already produce the figure you need:
1. Add the rendering function alongside the existing runner (e.g. add
   a function to `outcome/_run_notebooks.py`)
2. Import colors from `lib/palette.py` -- never hardcode
3. Output to `<snapshot>/figures/`
4. Update `manifest.json` `artifacts` list

DO NOT create a new top-level script that bypasses the runner pattern.

## When extending palette categories

If your figure introduces a new outcome class (e.g. `miss`, `absent`
for per-reach analysis), ADD it to `lib/palette.py`'s `OUTCOME_COLORS`
and `OUTCOME_CLASS_ORDER` so all downstream tools render it
consistently. Extend; don't fork.
