# MouseReach update — week of 2026-04-27 to 2026-05-04

## What happened

Last week we discovered mousereach was basically broken in a way the old reporting was hiding. The new DLC model had nuked most of the rule-based outcome detector, and on top of that, our reporting was claiming "same reach as GT" was correct when it almost never was — of 94 GT-retrieved reaches, only **19 were attributed to the right reach**.

So we built better reporting tools, discovered the real performance, and rebuilt the outcome detector. Now it only commits when it's certain and sends everything else to human review.

## How it was vs how it is

| | Baseline ("v3.1", last stable) | Now (v6 cascade) |
|---|---|---|
| Wrong commits | 41 | **9** |
| Committed accuracy | 95.6% | **98.9%** |
| Cases routed to manual review | 0 | 97 (10%) |

## Per-algo numbers (in our standard reporting format)

### Algo 3 — outcome detection (per-segment, 4-class)

| | Baseline | Cascade |
|---|---|---|
| retrieved precision / recall | 90.5% / 89.6% | **93.6% / 83.0%** |
| displaced_sa precision / recall | 95.5% / 97.2% | **99.4% / 83.2%** |
| untouched precision / recall | 97.2% / 97.4% | **99.7% / 98.2%** |
| Wrong commits | 41 | **9** |
| IFR delta median (committed) | (in baseline scalars) | (in cascade scalars) |

### Algo 4 — reach assignment (per-reach, holistic)

Strict accuracy: baseline **95.0%** → cascade **98.4%** (component-eval, GT reaches as input).

Cross-flow improvements:
- ret→mis (retrieval missed): 78 → **14**
- sa→mis (displaced missed): 25 → **47** (more triaged here, not "wrong" — see triaged column once landed)
- mis→ret (wrong reach called retrieved): 12 → **4**
- mis→sa (wrong reach called displaced): 69 → **10**

### Algo 1 (segmentation) and Algo 2 (reach detection)

Pre-existing snapshots; no changes from us this week. The reach detector is now the bottleneck — fixing it is the next priority.

## Caveat / next step

The cascade above was evaluated on **GT reaches as input** (component-eval). End-to-end production performance depends on also fixing the reach detector. That gap is what we go after next.

## Figures (in slide order)

1. Baseline per-reach Sankey: `outcome_master_pre_v4.0.0_baseline\figures\sankey.png`
2. Cascade per-reach Sankey: `v6_cascade_2026-05-04\figures\sankey_per_reach.png`
3. Cascade per-segment Sankey: `v6_cascade_2026-05-04\figures\sankey.png`
