# v6 Cascade E2E: v8.0.0 Production Reaches (Holdout)

Date: 2026-05-03

## What This Is

Three-way comparison of the v6 cascade outcome detector under three
reach-input conditions, answering: does the production v8.0.0 reach
detector recover the gap between component-eval (GT reaches) and e2e
(dev v8 reaches)?

## Three-Way Comparison

| Eval mode | Videos | Universe | Correct | Strict acc | absent_algo | absent_gt | triaged | wrong_class | wrong_reach |
|-----------|--------|----------|---------|------------|-------------|-----------|---------|-------------|-------------|
| Component-eval (GT reaches, 4 exh holdout) | 4 | 328 | 318 | 97.0% | 0 | 0 | 8 | 1 | 0 |
| E2E w/ dev v8 reaches (yesterday, 37 train_pool) | 37 | 5181 | 3210 | 62.0% | 975 | 804 | 255 | n/a | 12 |
| **E2E w/ v8.0.0 prod reaches (4 exh holdout)** | **4** | **339** | **316** | **93.2%** | **7** | **11** | **2** | **2** | **1** |

### Interpretation

The production v8.0.0 reach detector massively closes the gap:

- **93.2% strict accuracy** vs 62.0% with dev v8 reaches -- on a harder
  comparison (holdout videos never seen during cascade calibration).
- Only **3.8 percentage points below** the 97.0% component-eval ceiling.
- The residual 3.8% gap decomposes as:
  - 7 algo-side absent (GT reaches the v8 detector missed)
  - 11 GT-side absent (phantom v8 reaches with no GT match)
  - 2 triaged segments
  - 2 wrong class commits
  - 1 wrong reach called
- On matched reaches only (excluding absent on both sides): **98.4%**
  accuracy -- essentially at parity with component-eval.

### Why Yesterday's E2E Was 62% and Today's Is 93%

Yesterday's run used DEV v8 reaches: the in-sample v8 detector trained
per-video during LOOCV development, which had much worse reach detection
quality (over-calling, under-calling). The 804 GT-absent and 975
algo-absent reaches dominated the error budget.

Today's run uses the PRODUCTION v8.0.0 model: a single global model
trained on all 47 GT corpus videos with BSW w=0.8. It was tested on these
same 4 holdout videos (precision 84.8%, recall 91.8%), and the reach
detection quality is dramatically better. Only 7+11=18 unmatched reaches
total vs 804+975=1779 yesterday.

The two runs also differ in video set (4 holdout vs 37 train_pool), but
the dominant effect is reach detector quality.

## Per-Segment Results (4 Holdout Videos)

| GT class | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| untouched | 39 | 39 | 100.0% |
| displaced_sa | 34 | 35 | 97.1% |
| retrieved | 3 | 5 | 60.0% |
| Triage | 1 | - | - |
| Wrong commits | 2 | - | - |

Wrong commits:
- `20250708_CNT0210_P3 seg 14`: GT=retrieved, algo=retrieved, but wrong
  bout (Stage 9 vanished-after-reach attributed to wrong reach)
- `20250708_CNT0210_P3 seg 10`: GT=retrieved, algo=displaced_sa (Stage 21)
  -- the same genuinely ambiguous case that was wrong in component-eval

Expected triage: 0/1 handled (the 1 abnormal_exception segment was
committed as retrieved by the cascade, same as component-eval).

## v8.0.0 Reach Detection Quality on Holdout

| Video | GT reaches | v8 reaches | Matched | GT unmatched | v8 unmatched |
|-------|-----------|-----------|---------|-------------|-------------|
| 20250626_CNT0102_P4 | 18 | 16 | 15 | 3 | 1 |
| 20250708_CNT0210_P3 | 225 | 229 | 224 | 1 | 5 |
| 20250811_CNT0303_P4 | 19 | 21 | 17 | 2 | 4 |
| 20251024_CNT0402_P4 | 66 | 66 | 65 | 1 | 1 |
| **Total** | **328** | **332** | **321** | **7** | **11** |

97.6% of GT reaches were matched by v8. 96.7% of v8 reaches matched GT.

## Per-Reach Confusion Matrix

```
miss -> miss:                    279  (correct)
displaced_sa -> displaced_sa:     33  (correct)
retrieved -> retrieved:            4  (correct)
absent -> miss:                   10  (phantom v8 reach, cascade ignored)
miss -> absent:                    6  (v8 missed a non-causal GT reach)
abnormal_exception -> retrieved:   1  (cascade committed wrong on abn_exc)
absent -> retrieved:               1  (phantom v8 reach, cascade false retrieved)
displaced_sa -> absent:            1  (v8 missed a causal GT reach)
displaced_sa -> triaged:           1  (cascade triaged instead of committing)
miss -> triaged:                   1  (triage on a miss-class reach)
miss -> wrong_reach_called:        1  (correct outcome, wrong reach attributed)
retrieved -> displaced_sa:         1  (wrong class: GT=ret, algo=disp_sa)
```

## Verdict

**SHIP as v6.0.0 production.**

Decision rule was: if e2e strict acc on the 4 holdout videos >= 80%,
ship. The result is 93.2% -- well above the threshold.

Supporting evidence:
1. **93.2% strict per-reach accuracy** on held-out videos.
2. **98.4% on matched reaches** -- the cascade itself is near-perfect
   when reaches are correct.
3. **The 3.8% gap from component-eval ceiling is almost entirely reach
   detection error** (7 missed + 11 phantom = 18 unmatched reaches),
   not cascade logic error.
4. Only 2 wrong class commits across 80 segments, and one of those
   (seg 10) was already wrong in component-eval (genuinely ambiguous).
5. Untouched detection is 100%. Displaced_sa is 97.1%. Retrieved is
   60% but on only 5 segments (3/5 correct, 1 wrong class, 1 wrong bout).
6. The cascade's wrong-commit rate (2/79 scorable = 2.5%) is within the
   acceptable range for production use.

## Snapshot

`Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\outcome\v6_cascade_e2e_v8_production\`

Contents:
- `algo_outputs/{video}_pellet_outcomes.json` (4 videos)
- `algo_outputs/{video}_reaches.json` (4 videos, v8.0.0 reach outputs)
- `metrics/per_reach_scalars.json`
- `metrics/per_segment_scalars.json`
- `figures/sankey_per_reach.png` (v2 renderer)
- `figures/sankey.png` (per-segment 4-class)
