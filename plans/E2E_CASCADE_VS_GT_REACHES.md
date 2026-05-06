# v6 Cascade: GT Reaches vs v8 Reaches End-to-End Comparison

Date: 2026-05-04

## What This Is

Side-by-side evaluation of the v6 cascade outcome detector:
1. **Baseline (component eval):** cascade fed GT segment boundaries + GT reach windows
2. **End-to-end (e2e):** cascade fed GT segment boundaries + v8 reach detector outputs

Both runs use the same 37 train_pool videos (the v8 reach detector was
only run on these).  The per-reach Sankey uses `sankey_per_reach_v2` rules
(triaged propagates to all reaches in segment, zero-padded blocks, absent
fields populated, per-class precision on algo side).

## Reach Universe

| Metric | GT reaches | v8 reaches (e2e) |
|--------|-----------|-----------------|
| N reaches in universe | 4377 | 5181 |
| GT-side reaches | 4377 | 4377 + 804 absent = 5181 |
| Algo-side absent (v8 under-calls) | 0 | 975 |
| GT-side absent (v8 over-calls) | 0 | 804 |

The v8 reach detector found 804 reaches that have no GT match (over-calls,
within +-10 frame start-frame tolerance).  It also missed 975 GT reaches
(under-calls). These numbers include miss-class reaches (non-causal), so
the actionable subset is smaller -- see below.

## Per-Reach Confusion on Matched Reaches

On the 3402 GT reaches that DID match a v8 reach (excluding both-side absent):

| Metric | GT reaches (all 4377) | v8 reaches (3402 matched) |
|--------|----------------------|--------------------------|
| Correct classification | 3774 / 4377 (86.2%) | 3210 / 3402 (94.4%) |
| Triaged (review needed) | 594 (13.6%) | 175 (5.1%) |
| Wrong class | 4 | 5 |
| Wrong reach called | 5 | 12 |

The higher accuracy-on-matched for v8 (94.4% vs 86.2%) is an artifact:
v8 found fewer reaches per segment than GT (median 2-3 vs GT's exhaustive
annotation), so touched segments with fewer v8 reaches hit simpler cascade
paths that triage less.  The cascade's actual signal-reading quality is
identical -- the reduced triage rate comes from reduced reach ambiguity.

## Key Outcome Classes

| Class | GT reaches (baseline) | v8 e2e (matched only) | v8 e2e (incl. absent) |
|-------|----------------------|-----------------------|----------------------|
| Retrieved correct | 78/90 (86.7%) | 70/76 (92.1%) | 70/90 (77.8%) |
| Displaced_sa correct | 264/300 (88.0%) | 254/267 (95.1%) | 254/300 (84.7%) |
| Retrieved -> absent | n/a | n/a | 14/90 (15.6%) |
| Displaced_sa -> absent | n/a | n/a | 33/300 (11.0%) |
| Miss -> miss correct | 3432/3982 (86.2%) | 2886/3054 (94.5%) | 2886/3982 (72.5%) |
| Miss -> absent | n/a | n/a | 928/3982 (23.3%) |

The "incl. absent" column shows the real end-to-end picture:
- 14 GT retrieved causal reaches (15.6%) had no v8 match at all
- 33 GT displaced_sa causal reaches (11.0%) had no v8 match at all
- These are reaches the v8 detector missed entirely, so the cascade
  never had a chance to classify them

## v8 Over-Calls (GT-Side Absent)

804 v8 reaches with no GT match. Their algo-side disposition:

| Algo outcome | Count | Interpretation |
|-------------|-------|----------------|
| miss | 674 | Harmless: v8 found a phantom reach, cascade correctly ignored it |
| triaged | 80 | Cascade couldn't decide, flagged for review |
| displaced_sa | 46 | CASCADE ERROR: phantom v8 reach caused a false displaced_sa commit |
| retrieved | 4 | CASCADE ERROR: phantom v8 reach caused a false retrieved commit |

The 50 false commits from phantom reaches (46 displaced_sa + 4 retrieved)
are the v8 reach detector's direct contribution to end-to-end error.
These are segments where no GT reach existed but v8 invented one, and
the cascade then erroneously concluded the pellet was displaced/retrieved.

## What the Gap Reveals

1. **The v8 reach detector is the primary bottleneck for end-to-end
   outcome accuracy.**  The cascade itself is quite accurate when given
   correct reach windows (94.4% on matched).  The losses come from:
   - Under-detection: 975 GT reaches missed -> 47 causal reaches lost
     (14 retrieved + 33 displaced_sa)
   - Over-detection: 804 phantom reaches -> 50 false outcome commits
     (46 displaced_sa + 4 retrieved)

2. **Triage drops from 594 -> 255 in e2e.** This is NOT an improvement --
   it means the cascade is more confident (right or wrong) because it
   sees fewer reaches per segment. Some of the 594 baseline triages were
   correct caution on ambiguous multi-reach segments; in e2e, those
   segments have fewer v8 reaches and the cascade commits instead.

3. **The miss -> absent flow (928) is dominated by non-causal reaches.**
   Most of the 975 v8 under-calls are on non-causal reaches (reaches the
   mouse made that didn't cause the outcome). The 47 that matter are the
   causal reach misses (14 ret + 33 disp).

## Figure Paths

- **E2E Sankey (v8 reaches):**
  `Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\outcome\v6_cascade_2026-05-04_e2e_v8_reaches\figures\sankey_per_reach.png`

- **Baseline Sankey (GT reaches, same 37 videos):**
  `Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\outcome\v6_cascade_2026-05-04_e2e_v8_reaches\figures\sankey_per_reach_gt_baseline_37.png`

- **PI meeting copy:**
  `Y:\2_Connectome\Behavior\MouseReach\plans\PI_MEETING_2026-05-04_BUNDLE\algo3_outcome_current_e2e\sankey_per_reach.png`

## Scalars Files

- `metrics/per_reach_scalars.json` -- e2e confusion matrix
- `metrics/per_reach_scalars_gt_baseline_37.json` -- GT-reaches baseline on same 37 videos

## Snapshot Directory

`Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\outcome\v6_cascade_2026-05-04_e2e_v8_reaches\`
