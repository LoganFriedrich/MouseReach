# improvement/segmentation/ -- AGENTS.md

**CODE.** Segmentation phase of the MouseReach Improvement Process.

## Figures: USE `_run_notebooks.py` -- do NOT hand-roll

Canonical figure runner: [`_run_notebooks.py`](_run_notebooks.py)

Available functions (mirror the per-phase pattern):
- `run_violin(snapshot_dir)` -- boundary-delta distribution per subset
- `run_summary_table(snapshot_dir)` -- formatted boundary-error / phantom / miss table

Notebooks (`*.ipynb`) are interactive views; production figures come
from the runner.

Reads from `<snapshot>/metrics/scalars.json`. Writes to `<snapshot>/figures/`.
Uses palette `improvement/lib/palette.py` (`SEGMENTATION_COLORS`,
`SEGMENTATION_LABELS`, `SEGMENTATION_SUBSET_ORDER`).

## Metrics

[`metrics.py`](metrics.py) computes the segmentation scalars:
- `inter_pellet_B2_B20` (interior boundaries)
- `endpoint_B1_B21` (first/last)
- `all` (combined)
Each subset has `delta_histogram`, `n_phantom`, `n_miss`, `mean_signed_delta`,
`median_abs_delta`, `mean_abs_delta`.

## When extending

Don't fork the runner; add a function inside `_run_notebooks.py` and
import colors from the palette. Update the snapshot's `manifest.json`
artifacts list.
