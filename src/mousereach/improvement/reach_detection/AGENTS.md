# improvement/reach_detection/ -- AGENTS.md

**CODE.** Reach detection phase of the MouseReach Improvement Process.

## Figures: USE `_run_notebooks.py` -- do NOT hand-roll

Canonical figure runner: [`_run_notebooks.py`](_run_notebooks.py)

Available functions:
- `run_violin(snapshot_dir)` -- start/end delta distributions
- `run_summary_table(snapshot_dir)` -- TP/FP/FN counts + boundary deltas
  (per memory `feedback_no_f1.md`: lead with TP/FP/FN, no F1)

Reads `<snapshot>/metrics/scalars.json`. Writes `<snapshot>/figures/`.
Uses palette `improvement/lib/palette.py` (`REACH_DETECTION_COLORS`,
`REACH_DETECTION_LABELS`, `REACH_DETECTION_DELTA_ORDER`).

## Metrics

[`metrics.py`](metrics.py) computes reach-matching scalars:
- `n_gt`, `n_algo`, `n_matched`, `n_fp`, `n_fn`
- start/end delta histograms
- per-video count_delta_per_video_histogram

[`kinematic_damage.py`](kinematic_damage.py) -- per-reach kinematic damage
classification (true_miss / fragmented / cropped) for "meaningful" (slit-
crossing) reaches per memory `meaningful_reaches_prioritize_not_exclude.md`.

[`triangulate.py`](triangulate.py) -- 3-point triangulation against pre-DLC
baseline + best-post-DLC + experiment.

[`paw_geometry_survey/`](../../../../../../Behavior/MouseReach_Pipeline/Improvement_Snapshots/reach_detection/paw_geometry_survey/)
(in snapshots) holds the corpus survey of valid paw cluster geometry,
producing `percentile_thresholds.json` for use in DLC-validity filters.

## When extending

Match function `match_reaches(algo_reaches, gt_reaches, window=10)` in
metrics.py is the canonical reach-matching primitive. The outcome
phase's per-reach computation also imports it via
`from mousereach.improvement.reach_detection.metrics import match_reaches, Reach`.

Don't fork the runner; add a function inside `_run_notebooks.py`. Don't
hardcode colors -- import from the palette.
