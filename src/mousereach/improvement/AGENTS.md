# improvement/ -- AGENTS.md

**This is CODE.** Algorithm-improvement-process tooling for MouseReach.
Snapshot DATA accumulates in `Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\`.

## REPORTING STANDARDS (READ BEFORE EVERY MODEL EVALUATION)

The user has mandated how reach detection and outcome classification
performance is reported. This is not preference -- it is the standard.
Drifting back to summary scalars is the failure mode being prevented.

### Outcome model evaluation
- **LEAD with the Sankey** (`outcome/_run_notebooks.py:run_sankey`).
  Show GT class -> algo class flow widths and call out the
  directional shifts (e.g. `displaced_sa -> retrieved` is a different
  problem than `retrieved -> displaced_sa`).
- **Within correct categorical calls, report causal-reach correctness
  separately**: did the algo identify the SAME reach as causal (TP) or
  a different one (FP/FN)? Per-reach matching, not segment-level
  agreement.
- **Precision/recall are supporting only**, never the lead.
- **F1 is banned** (see `feedback_no_f1.md` in cross-session memory).

### Reach detection evaluation
Two layers, both required.

**Layer 1: detection TP/TN/FP/FN.** A detected reach counts as TP iff
its start_frame is within +/- **2 frames** of a GT reach start AND
its frame span is roughly the same. (Note: the existing
`match_reaches` uses +/- 10f, which is appropriate for outcome
causal-reach matching, NOT for headline reach detection eval.)

**Layer 2: for TPs only, distribution of error.** Histogram or violin
of `algo_start - gt_start` and `algo_span - gt_span`. Tells us if the
model is biased early/late and how tight the agreement is.

### Don't drift back
Before sending any model-performance message: re-read the cross-session
memory entry `reach_outcome_evaluation_format.md`. If the report leads
with a P/R/F1 table or a single accuracy scalar, it has failed and
must be redone with Sankey + direction as the lead.

### Per-reach eval Sankeys (algo vs GT or review) -- USE THIS, don't hand-roll

**To measure how the algo is doing / generalizing PER REACH against ground
truth or human review alone, use `per_reach_sankey_eval.py`. This is the
canonical generator -- do not hand-roll a per-reach Sankey for this.**

```
python -m mousereach.improvement.per_reach_sankey_eval algo-vs-gt
python -m mousereach.improvement.per_reach_sankey_eval algo-vs-review
```

- `algo_vs_gt` -- algo (exactly as run) vs ground truth, per reach. Reproduces
  the canonical algo-4 confusion (`outcome/metrics.compute_per_reach_confusion`)
  and renders it **algo-left / GT-right**.
- `algo_vs_review` -- algo (exactly as run) vs human causal-review, per reach,
  over the Causal Review tool's reviewed bundles. **Triage is a per-segment
  decision: one `triaged` mark per triaged segment, re-routed to the human's
  resolution -- NOT one per reach (that ~10x inflation is the documented
  anti-pattern; see `assignment/AGENTS.md`). `absent` is dropped except when a
  human causal reach overlaps no algo reach** (a reach the algo missed entirely).

Both figures are the algo EXACTLY AS RUN vs the reference -- there is NO
"post-correction" panel (that only proves the results file was edited, not how
the algo performed). Writes a dated snapshot (`metrics/scalars.json` +
`figures/*.png`) under `MouseReach_Improvement/model40_eval/`.

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
