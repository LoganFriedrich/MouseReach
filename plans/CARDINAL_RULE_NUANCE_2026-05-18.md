# The Cardinal Rule and boundary precision: what the data says

**Author:** notes from the 2026-05-18 reach-detection investigation
**Audience:** anyone deciding how strictly to apply the Cardinal Rule
**Status:** Revised after running the value-drift diagnostic. An earlier
draft argued the rule's enforcement was over-strict for trajectory
features. The empirical data contradicted that claim and this revision
walks the conclusion back to what the numbers actually support.

---

## The rule, as stated in `CLAUDE.md`

> Every frame boundary error corrupts kinematic data. Reach detection
> exists to extract kinematics (velocity, acceleration, trajectory shape,
> peak extension). When a reach boundary is wrong — even by a few frames —
> you are computing kinematics over frames that are NOT part of the
> behavior being studied. Those frames are noise. The noise contaminates
> means, blurs distributions, and can create differential measurement
> artifacts between experimental groups.

This document tests that claim empirically on the v8.0.0 generalization
corpus. The headline result: **the rule is largely correct, with one
specific feature class that has genuine leeway.**

---

## What the strict matching rule actually measures

The current evaluation enforces the Cardinal Rule through a binary
**strict matching** criterion: an algo reach counts as a TP only if
`|algo_start - gt_start| <= 2` AND `|algo_span - gt_span| <= max(0.5 * gt_span, 5)`.
Anything else is FP or FN.

This is a binary threshold on what's actually a gradient — the
**kinematic-value drift** between an algo-window computation and the
corresponding GT-window computation grows continuously as the boundary
error grows. The question this document asks is: how well does the
strict cutoff correspond to actual kinematic corruption?

---

## Empirical findings (v8.0.0 production, 20-video generalization, 3432 matched pairs)

Six representative kinematic features computed twice for each
permissive-matched (algo, GT) pair — once with the algo window, once
with the GT window — using the same underlying DLC trajectory. The
mean absolute percent difference between the two computations:

| Feature | Class | Strict-accept (n=3235) | Strict-reject (n=197) |
|---|---|---:|---:|
| `extension_past_nose` (at apex) | A | **0.00%** | **0.00%** |
| `paw_width_at_start` | B | 4.4% | 15.3% |
| `paw_width_at_end` | B | 1.6% | 14.8% |
| `duration_frames` | B | 3.7% | **104.5%** |
| `total_path` | C | 9.7% | **286.9%** |
| `peak_speed` | C | 6.1% | **129.3%** |

Source data:
`Improvement_Snapshots/reach_detection/v8.0.0_kinematic_value_drift/metrics/summary.json`.

**By drift magnitude:**

| \|start_delta\| | n | total_path drift | peak_speed drift | duration drift |
|---:|---:|---:|---:|---:|
| 0 frames | 2876 | 19% | 9% | 6% |
| 1-2 frames | 461 | **60%** | 39% | 22% |
| 3-5 frames | 74 | 56% | 26% | 46% |
| 6-10 frames | 21 | 57% | 8% | 92% |

Drift in trajectory features starts at the 1-2 frame error level. The
synthetic-anchor design (anchor positions computed from `BoxL`/`Nose`/
`BoxR` at start/end frames) does not materially cushion these features
— the bulk of `total_path` is mid-window paw-to-paw distance, which is
directly affected by which frames are in the window.

---

## The feature-class breakdown the data supports

**Class A — Apex-anchored (genuinely robust to boundary drift).**
Features computed AT the apex frame, or that use the apex as a
reference point. Drift is 0.00% across all strict-accept and
strict-reject subsets and across every drift bucket. The GT apex is
inside the algo window 99.2% of the time on this corpus, so these
features end up computed at the same frame regardless of boundary error.

  - `extension_past_nose_*` (per paw point)
  - `*_apex_speed_*` (per paw point — speed at the per-point apex)
  - `paw_width_proxy_at_apex_*`, `paw_outline_area_at_apex_*`,
    `paw_spread_max_at_apex_*` (paw shape at apex)
  - `hand_angle_at_apex_deg`
  - `head_width_at_apex_mm`, `nose_to_slit_at_apex_mm`,
    `head_angle_at_apex_deg` (body posture at apex)
  - `righthand_visibility_at_apex` (per paw)
  - `paw_apex_lead_frames`, `paw_leading_point`

**Class B — Boundary-frame-direct (rises with drift magnitude).**
Features computed AT start_frame or end_frame literally. A drift of
`d` frames means the measurement is computed at a frame `d` frames
away from the true boundary. The data shows 1-15% drift on these
features, plateauing somewhat because paw shape and visibility don't
change drastically frame-to-frame, but consistently non-zero in the
strict-reject subset.

  - `duration_frames`
  - `paw_width_proxy_at_start_*`, `_at_end_*`
  - `paw_outline_area_at_start_*`, `_at_end_*`
  - `paw_spread_max_at_start_*`, `_at_end_*`
  - `righthand_visibility_at_start`, `_at_end` (per paw)

**Class C — Window-aggregate (highly sensitive — corrected from prior draft).**
Integrated over all frames in `[start_frame, end_frame]`. **The data
shows much higher drift than the synthetic-anchor design suggested.**
At 1-2 frame boundary drift, `total_path` drifts ~60% on average;
`peak_speed` drifts ~39%. In the strict-reject subset, `total_path`
drifts ~287%. These features are corrupted by even small boundary
errors.

  - `total_path_*`, `path_directness`, `lateral_spread_*`,
    `swept_area_*`, `lateral_deviation_*`, `motion_smoothness`
  - `mean_speed_*`, `peak_speed_*`
  - `paw_width_proxy_max/min/mean/range` (and aggregates for
    outline_area, spread_max)
  - `paw_velocity_correlation`
  - `tray_contact_duration_frames`
  - `hand_rotation_total_deg`
  - `visibility_max/min/mean/range` (per paw), `frames_any_paw_low_confidence`

---

## How well does the strict cutoff correspond to kinematic corruption?

The strict matching rule (start ±2, span ±5 or ±50%) corresponds
roughly to a 10% Class C drift threshold:

- Strict-accept events: ~10% mean `total_path` drift, ~6% `peak_speed` drift.
- Strict-reject events: ~290% mean `total_path` drift, ~130% `peak_speed` drift.

So the strict cutoff IS effectively separating "small kinematic
corruption" from "large kinematic corruption." It is not measuring
this directly — it is measuring boundary frames — but the two are
strongly correlated in practice. The rule's enforcement is doing real
work for Class B and Class C features.

The one place the strict rule is over-strict relative to actual
kinematic damage is Class A features. A strict-reject event with
boundaries off by 5 frames still has `extension_past_nose`, apex speed,
hand angle, body posture at apex, and all other apex-anchored
measurements identical to a strict-accept event with perfect
boundaries. The strict TP/FP/FN headline cannot distinguish these.

---

## The recommendation

**Keep the Cardinal Rule. The data backs the principle.** Boundary
errors materially corrupt the majority of the project's emitted
kinematic features (Class B and Class C). The current strict matching
rule's enforcement is approximately calibrated to a ~10% drift
threshold and is defensible as a headline metric.

**Surface kinematic-damage metrics alongside strict TP/FP/FN, not
replacing them.** Specifically:

1. **Apex-inclusion rate** — fraction of GT reaches where the algo
   window contains the GT apex frame. Captures Class A safety. v8.0.0
   on generalization: 99.2%. This number tells a different — and
   correct — story than the strict matching rule does about
   apex-anchored features.

2. **Coverage rate** — fraction of GT-marked reach frames inside the
   algo window. Captures rough Class C exposure. v8.0.0: 98.9% have
   coverage ≥ 80%.

3. **Class-specific drift summaries** — for the strict-reject subset,
   the per-feature drift numbers in this document directly characterize
   what's being lost when the strict criterion fails. These are
   feature-specific and downstream-informative.

The point is not to weaken the Cardinal Rule. It is to give downstream
consumers a more textured picture: which of their features are robust
to v8.0.0's boundary near-misses (Class A), which are not (Class B and
Class C), and how much corruption is expected on the ~5.8% of reaches
that the strict rule rejects.

---

## What this does NOT argue

- **It does not argue that the strict matching rule is wrong.** The
  data shows it correlates well with kinematic damage for Class B and
  Class C features.
- **It does not argue for relaxing boundary precision.** Tighter
  boundaries reduce drift in every class. Boundary precision is the
  right thing to chase.
- **It does not argue against the Cardinal Rule.** The rule is
  empirically supported for the majority of emitted kinematic features.

What it does argue: **the rule has one specific exception worth
recognizing — apex-anchored measurements are robust to boundary errors
of up to ~10 frames on this corpus.** If a downstream analysis uses
only Class A features (e.g., peak extension and hand angle at apex,
without trajectory aggregates or boundary measurements), boundary
near-misses up to that magnitude do not corrupt the analysis. The
strict matching rule penalizes these as if they did, which underplays
v8.0.0's quality on those specific features.

---

## Open question

The 197 strict-reject events have:

- Class A features intact (0% drift)
- Class B features drifted 15% on average
- Class C features drifted 130-290% on average

These are the events worth investigating case-by-case for fixability.
If the algo systematically over-extends in a recognizable pattern (the
2026-05-04 within_gt FP analysis suggested it does — paw retraction
motion past GT_end), a targeted post-process that cuts the over-extension
without affecting strict-accept events could meaningfully reduce Class
C drift without changing the model. That's where engineering effort
likely pays off.

The 0.8% of GT reaches whose apex falls OUTSIDE the algo window — about
28 events — are the cases where ALL feature classes are corrupted
(Class A loses the apex; Class B and C lose the window). These are the
worst events on the corpus and should be the highest priority for
inspection.

---

## Provenance

Source data files (raw drift numbers + per-event records):

  - `Improvement_Snapshots/reach_detection/v8.0.0_kinematic_completeness_generalization/metrics/summary.json`
  - `Improvement_Snapshots/reach_detection/v8.0.0_kinematic_value_drift/metrics/summary.json`
  - `Improvement_Snapshots/reach_detection/v8.0.0_kinematic_value_drift/metrics/drift_per_event.json`

Runners that produced them (in `scripts/`):

  - `diagnose_v8_kinematic_completeness.py`
  - `diagnose_v8_kinematic_value_drift.py`

Both runners are read-only on existing snapshots and DLC files;
neither modifies any algorithm code, model, or GT.
