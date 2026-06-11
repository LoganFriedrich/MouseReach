# FP/FN Widget + GT Editing — Integration State (2026-05-19)

Shared coordination doc between the widget-owning Claude (running on
Logan's local workstation, integrating GT editing into the FP/FN review
widget) and the server-side Claude (running on Y: with access to the
analysis runners, accessible at the y--2-Connectome Claude project).

Any Claude resuming this work in a new session should read this file
first; it captures the cross-machine integration state that neither
side's private memory fully holds.

---

## Production state (as of 2026-05-20, post-GT-review re-score)

- `mousereach` package: **2.10.0-dev**
- Reach detector: **v8.0.1** (BSW b=1/w=0.8 + merge_gap=0)
- v8 production model bundle: `mousereach/reach/v8/models/v8.0.0_bsw_w0.8.joblib` (unchanged from v8.0.0; v8.0.1 is a postprocess default change only)
- Cumulative-best LOOCV (re-scored 2026-05-20 against updated GT, span+outside_gt_seg filtered): **TP=2155 / FP=182 / FN=233** (was 2069 / 266 / 295)
- Holdout generalization (19 exhaustive videos, re-scored 2026-05-20 against updated GT, span+outside_gt_seg filtered): **TP=3555 / FP=153 / FN=155** (was 3501 / 204 / 191)
- Metric convention update 2026-05-19: post-match filter excludes reaches with span < 4 frames from headline TP/FP/FN counts.
  See `mousereach.improvement.reach_detection.metrics.MIN_REPORTED_SPAN`.
- Metric convention update 2026-05-20: post-match filter additionally excludes algo reaches whose start_frame falls outside the GT segmentation window (before first GT boundary or after last GT boundary) from headline FP counts. Physically-impossible "phantoms" in apparatus dead time. See `mousereach.improvement.reach_detection.metrics.is_outside_gt_segmentation`.
- Calibration LOOCV snapshot is now `v8.0.0_dev_merge_gap_0_candidate_rescored_2026-05-20/`. Source snapshot `v8.0.0_dev_merge_gap_0_candidate/` (original 2026-05-18 acceptance run) preserved intact for the v8.0.1 ship-decision provenance.

Allowed metric reporting per project rules: TP, FP, FN counts; boundary delta distributions; per-category breakdowns; Cardinal-Rule-aligned scalars (apex inclusion, coverage). **Forbidden**: precision/recall scalars, F1, accuracy, AUC.

## The widget context

Logan is reviewing the 33 v8.0.1 FP/FN review manifests via a local
widget (also-Claude-built). Reviewing reveals GT errors that should be
corrected. Today's work added a GT editing capability to the widget so
edits can be saved back to the canonical GT files, then re-scored on
the server side.

### Manifests being reviewed

```
Y:\2_Connectome\Behavior\MouseReach_Improvement\fpfn_review_manifests\v8.0.1\
  calibration_loocv\     -- 14 manifests
  holdout_2026_05_11\    -- 19 manifests
```

Each manifest contains TP/FP/FN events per reach with:
- `kind`: TP / FP / FN (legacy categorization, preserved for backwards-compat)
- `detector`: {start, end} of algo reach (or null)
- `gt`: {start, end} of GT reach (or null)
- `category`: failure-mode label (within_gt, tolerance_miss_start, etc.) -- legacy sub-categorization
- `kinematically_excluded`: bool (true if either side's span < 4)
- `outside_gt_segmentation`: bool (true if FP algo_start falls outside [B_first, B_last] of GT segmentation; always false for TP and FN)
- `start_delta`, `span_delta` for TPs
- **`topology`** (NEW 2026-05-20): one of TP / TOLERANCE_ERROR / MERGED / FRAGMENTED / FALSE_POSITIVE / FALSE_NEGATIVE / COMPLEX. Topology-based label from connected-components analysis of the algo-GT overlap graph. See "Topology event types" section below.
- **`topology_sub`** (NEW 2026-05-20): optional finer label. For TOLERANCE_ERROR: "start_off", "span_off", or "start_and_span_off". For MERGED: "{N}_gt". For FRAGMENTED: "{N}_algo". For COMPLEX: "{N}_algo_{M}_gt". None for TP, FALSE_POSITIVE, FALSE_NEGATIVE.
- **`component_id`** (NEW 2026-05-20): integer, unique per video, groups events in the same connected component of the overlap graph. A MERGED event's three rows (1 algo FP + 2 GT FNs under legacy kind labels) all share the same component_id so the widget can group them as one logical event.

Per-manifest top-level fields:
- `gt_segmentation`: `{n_boundaries, first_frame, last_frame}` for quick reference
- **`topology_summary`** (NEW 2026-05-20): `{TP: N, TOLERANCE_ERROR: N, MERGED: N, FRAGMENTED: N, FALSE_POSITIVE: N, FALSE_NEGATIVE: N, COMPLEX: N}`. Counts one entry per connected component (not per row), so a MERGED that spans 3 rows still counts as 1 in this summary.

Schema documented in widget-owner's `SCHEMA.md`.

## Topology event types (locked 2026-05-20)

Each connected component of the (algo, GT) overlap graph gets a topology label based on its (n_algo, n_gt) shape:

| Topology | Shape | What it means |
|---|---|---|
| **TP** | 1 algo + 1 GT, tolerance passes (start within +/-2f AND span within tolerance) | Algo found the reach correctly. |
| **TOLERANCE_ERROR** | 1 algo + 1 GT, overlap exists, tolerance fails | Algo found the reach but boundaries are off. Sub: "start_off", "span_off", or "start_and_span_off". |
| **MERGED** | 1 algo + 2+ GT | Algo collapsed multiple GT reaches into one. Sub: "{N}_gt" where N = GT count. |
| **FRAGMENTED** | 2+ algo + 1 GT | Algo split one GT reach into multiple. Sub: "{N}_algo" where N = algo count. |
| **FALSE_POSITIVE** | 1 algo + 0 GT | Algo emitted a reach where there is no GT anywhere -- true phantom. |
| **FALSE_NEGATIVE** | 0 algo + 1 GT | GT reach has no algo overlap anywhere -- true miss. |
| **COMPLEX** | 2+ algo + 2+ GT | Both merge and fragment in one component. Rare. |

Why the renaming (from PHANTOM/MISS -> FALSE_POSITIVE/FALSE_NEGATIVE; from BOUNDARY_ERROR -> TOLERANCE_ERROR): with MERGED/FRAGMENTED/TOLERANCE_ERROR carved out, what remains as "FP" or "FN" is a real detection failure -- not an artifact of how the matcher labeled boundary issues. So the literal "false positive" / "false negative" names are honest here.

## Widget changes needed for the new fields

**Minimal (backwards-compatible):**
1. Read `topology` if present and color/icon the row accordingly. Existing TP/FP/FN logic keeps working since `kind` is unchanged.
2. Optional: filter "show only X topology" (e.g., "show only MERGED events").
3. Optional: display `component_id` so user can see grouping.

**Bigger (when ready):**
1. Group rows by `component_id` with expand/collapse. A MERGED component renders as one collapsed row by default ("MERGED: 1 algo spanning 2 GT reaches"), expanding to show the constituent rows.
2. Replace the "FP=N FN=N TP=N" summary panel with the 7-label topology counts from `topology_summary`.

**What does NOT change:**
- Headline metric counter still reports TP/FP/FN under old greedy matching (no change to count_filtered_metrics, no change to filtered headline numbers).
- All existing fields preserved (kind, category, detector, gt, kinematically_excluded, outside_gt_segmentation, start_delta, span_delta).

The topology classifier is currently inline in the manifest generator (Logan's Option A 2026-05-20). If/when we promote topology to be the headline counter source of truth (phase 2), the classifier moves into `mousereach.improvement.reach_detection.metrics` and the runners use it.

## Corpus-wide topology breakdown (post-2026-05-20 GT-edit rescore)

Reference baseline for sanity-checking widget renders:

| Topology | Holdout (19) | Cal LOOCV (14) |
|---|---:|---:|
| TP | 3570 | 2152 |
| TOLERANCE_ERROR | 96 | 85 |
| MERGED | 16 | 58 |
| FRAGMENTED | 9 | 10 |
| FALSE_POSITIVE | 75 | 49 |
| FALSE_NEGATIVE | 44 | 24 |
| COMPLEX | 0 | 3 |

## GT-tool integration (locked in 2026-05-19)

### Auto-resolved GT paths

The widget loads GT deterministically from corpus + video_id:

```python
GT_ROOTS = {
    "calibration_loocv":  r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\gt",
    "holdout_2026_05_11": r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations\generalization_test_2026-05-11\gt",
}
gt_path = Path(GT_ROOTS[manifest["corpus"]]) / f"{manifest['video_id']}_unified_ground_truth.json"
```

Verified 2026-05-19: 14/14 calibration manifests and 19/19 holdout
manifests resolve cleanly under this mapping. No missing GT files.

### File format

`*_unified_ground_truth.json` (unified schema). Each reach record has:

```json
{
  "reach_id": 1,
  "segment_num": 1,
  "start_frame": 194,
  "start_determined": true,
  "start_determined_by": "schultzc",
  "start_determined_at": "2026-04-28T09:36:28.235320",
  "end_frame": 204,
  "end_determined": true,
  "end_determined_by": "...",
  "end_determined_at": "...",
  "apex_frame": 199,
  "exclude_from_analysis": false,
  "exclude_reason": null,
  "comment": null
}
```

Top-level wrapper keys to preserve on write: `video_name`, `type`,
`schema_version`, `created_at`, `last_modified_at`, `last_modified_by`,
`segmentation`, `reaches.exhaustive`, `outcomes`, `completion_status`.

The widget should round-trip the entire JSON, only mutate what the user
edited, and update `last_modified_at` / `last_modified_by` on save.

### Widget behavior (locked)

- On manifest open: auto-resolve GT path from `GT_ROOTS[corpus] / video_id`. If file exists, auto-load. If not, show "no GT auto-found; click Load GT to pick one." Display resolved path prominently in UI.
- "Load GT..." manual override button still present for non-canonical cases.
- Save: atomic write (write to temp, rename to final) to the path currently loaded.
- Unknown corpus field: falls back to manual Load GT (future-proof).

### Multiple GT mirrors exist (housekeeping caveat)

Five directories on Y: contain `*_unified_ground_truth.json` files
totaling 155 files. The canonical paths above are the ones the
server-side analysis reads from. Other paths:

```
Y:\2_Connectome\Behavior\MouseReach_Improvement\MouseReach_Generalization_Testing\gt\
  20 files -- mirror of the 20-video holdout set. Worth verifying with diff;
  may need deprecation or sync rule.

Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\gt_complete_only\
  21 files -- exhaustive subset of the 47-video calibration corpus.
  May need to be kept in sync with the parent gt/ dir.

Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations\
  2026-04-28_outcome_v4.0.0_dev_walkthrough\gt\
  47 files -- iteration-specific snapshot of the calibration GT. Should
  be treated as historical record, not edited.
```

If the widget only writes to the two paths in `GT_ROOTS`, the mirrors
will drift. Worth a one-time decision: which paths are canonical, which
should auto-sync from canonical, which should be archived.

## Server-side re-score loop after GT edits

After Logan edits GT in the widget and saves to one of the `GT_ROOTS`
paths, the server-side Claude (this side) runs:

### For holdout GT edits

```bash
# Re-score holdout against updated GT
"C:\2_Connectome\envs\mousereach\python.exe" "Y:\2_Connectome\Behavior\MouseReach\scripts\holdout_generalization_test_merge_gap_0.py"

# Regenerate manifests
"C:\2_Connectome\envs\mousereach\python.exe" "Y:\2_Connectome\Behavior\MouseReach\scripts\generate_fpfn_review_manifests_v8_0_1.py"
```

Total: ~1 minute. Widget re-loads manifests from
`fpfn_review_manifests\v8.0.1\holdout_2026_05_11\`.

### For calibration GT edits (pragmatic re-score, no retrain)

A small re-score helper would need to be drafted that:
1. Loads the most recent calibration LOOCV snapshot's per-event records
   (algo_start_frame/algo_end_frame from `v8.0.0_dev_merge_gap_0_candidate/metrics/loocv_aggregate.json`)
2. Loads updated calibration GT from `validation_runs\DLC_2026_03_27\gt\`
3. Re-matches the existing algo reaches against the updated GT
4. Writes a new scoring snapshot

This script is NOT YET WRITTEN. Will be drafted when first calibration
GT edit is made. Easier than retraining; calibration model output is
frozen at the last LOOCV result and only the matching changes.

For strict calibration re-train (full LOOCV from scratch with updated
labels), the parquet at
`Improvement_Snapshots\_corpus\2026-04-30_restart_inventory\phase_b_dataset\train_pool.parquet`
needs to be rebuilt first. Requires the corpus-build pipeline. Not
needed for analysis-only re-scoring.

## Known GT anomaly to fix

`20250625_CNT0106_P2_unified_ground_truth.json` reach_id 133 has
`start_frame=24247, end_frame=22476` (end < start by 1770 frames).
Clearly a typo. Apex is at 22464. Worth being the first GT edit Logan
makes via the widget; corrected values likely around start=22463,
end=22476, apex=22464.

## Per-video flags (apparatus / behavioral notes from review)

Per-video diagnostic context discovered while reviewing manifests.
Helps interpret manifest events on these videos -- some "algo errors"
are downstream of apparatus or behavioral quirks rather than detector
bugs.

- **`20251010_CNT0308_P2`** (flagged 2026-05-19): "unusual BOXL that
  overhangs and produces artifacts when the pellet is near, leading to
  many early starts and false positives." Boundary box left edge has
  a physical apparatus quirk that contaminates DLC keypoint signal in
  the approach window. Early-start FPs on this video are downstream of
  this, not the detector seeing something the model can fix. When
  reviewing this manifest, weight apparatus-driven FPs lower in
  priority for algo-fix discussion; they're a separate class.

- **`20250806_CNT0316_P3`** (flagged 2026-05-19): similar BOXL issue
  to CNT0308_P2 -- "DLC keeps interpreting [the BOXL apparatus edge]
  as a paw." Same apparatus class; FPs driven by this should be
  weighted lower in algo-fix priority. Two videos with this pattern
  so far -- worth tracking for a possible apparatus-class flag if
  more turn up.

- **`20251008_CNT0303_P2`** (flagged 2026-05-20): lots of examples of
  "the mouse hanging on the edge of the box after a reach / at the
  end of the reach." Behavioral pattern that likely contaminates
  end_frame detection -- algo may over-extend reach endings because
  the hand stays visible on the box edge past the true behavioral
  end. Expect end_delta-positive boundary errors on this video to be
  driven by this pattern rather than detector tuning. Worth checking
  whether other videos have it too (could be a per-mouse trait).

- **`20250813_CNT0314_P4`** (flagged 2026-05-20): "has a ton of false
  negatives, suspect its because the nose is off for most reaches,
  mouse seems to be standing taller than normal." v8's nose-engagement
  post-filter requires >= 30% of frames in a reach to have nose within
  25 px of slit center; a taller-standing mouse keeps its nose
  consistently above that band, so legitimate reaches get filtered out
  as failing the nose-engagement check, producing FNs. Expect this
  pattern on other tall-standing mice too -- could be a per-mouse
  posture trait that argues for relaxing the nose-engagement
  threshold or making it adaptive.

- **`20251022_CNT0413_P4`** (flagged 2026-05-19): merge-heavy mouse;
  paw stays visible between rapid consecutive reaches rather than
  fully retracting. Causes residual merges that mg=0 cannot split
  (no proba dip to split on). See queued direction #2a below.

(Add new flags here as they're discovered during manifest review.)

## Queued directions (in order of discussion, not priority)

1. **Threshold / per-reach confidence work.** Emit per-reach
   confidence as an additive feature for downstream weighting + widget
   priority sorting.

2. **CNT0301_P3 chronic-FP investigation.** This single video has
   ~80 FPs across mg=2 / mg=1 / mg=0. Likely DLC-quality or
   apparatus-state specific. (TPs unchanged at 161 across all merge_gap
   settings -- so the chronic FPs are persistent over-detection, not
   residual merges.)

2a. **CNT0413_P4 residual-merge mechanism (observed 2026-05-19).**
   This merge-heavy video was the biggest mg=0 winner (+30 TP, -30 FN
   vs mg=2). The residual merges that mg=0 still doesn't split are
   driven by a behavioral pattern: this mouse keeps its paw visible
   between rapid consecutive reaches rather than retracting fully. The
   GBM proba stays saturated through 1-3 frame inter-GT gaps because
   the per-frame features look similar to mid-reach features, so
   merge_gap can't split (no sub-threshold frame to split on). Fix
   needs a different signal (hand-velocity zero-crossing, brief
   direction reversals, trajectory-shape) rather than proba-dip-based
   logic. Likely generalizes to other mice with similar behavior.

3. **Smart postprocess: split-on-deep-dip + un-merge-on-shallow-dip.**
   Targets residual merges + fragmentation that mg=0 doesn't address.
   Diagnostic-first. **Note 2026-05-19**: doesn't help the CNT0413_P4-
   style residual merges where there is NO dip at all (paw-visibility
   mechanism above). Dip-based splitting is for the subset where the
   model produced a shallow dip; the no-dip cases need a different
   approach entirely.

4. **GT review pass via the widget** (this session's work). Iterate on
   the 33 manifests, correct GT errors, re-score, manifest regen.

## Calibration vs holdout corpora are on different DLC weights (discovered 2026-05-19)

The calibration LOOCV corpus and the holdout corpus were inferred with
different DLC weights, both labeled with the same scorer string
(`DLC_resnet50_MPSAOct27shuffle1_100000`). Same project + shuffle +
iteration count, but weights diverged across the 2026-03-27 DLC retrain.

| Corpus | DLC inference date | DLC weights vintage |
|---|---|---|
| Calibration (47-video train_pool, incl. 16-video LOOCV exhaustive subset) | 2026-02-19 (per `iterations/2026-04-28_outcome_v4.0.0_dev_walkthrough/MANIFEST.json`) | pre-Mar-27 (old) |
| Holdout (20-video generalization set) | 2026-04-27 (mtime-spread on `iterations/generalization_test_2026-05-11/dlc/` consistent with fresh inference) | post-Mar-27 (new) |

The 2026-03-27 retrain (`A:\AIs\MPSA-LF-2025-10-27\dlc-models\iteration-0\MPSAOct27-trainset95shuffle1\train\snapshot-100000`) lands between the two
inference dates.

**Implication for widget review:**

Calibration manifests reflect v8 behavior on old DLC h5s; holdout
manifests reflect v8 behavior on new DLC h5s. GT edits apply equally
either way (GT is the human label, independent of DLC). But events on
calibration manifests that look weird (e.g., the chronic-FP video
20250812_CNT0301_P3 with ~80 FPs) may be DLC-shift artifacts rather
than algo bugs. Before opening a per-video investigation off the back
of a calibration manifest, sanity-check whether new-DLC inference
resolves it.

**Implication for server-side analysis:**

Holdout numbers (TP=3501/FP=212/FN=191 filtered at v8.0.1) are the
production-relevant signal; calibration LOOCV (TP=2069/FP=270/FN=295
filtered) describes v8 against its training-DLC distribution. Future v8
iterations that improve calibration LOOCV but degrade holdout should be
rejected -- holdout is the new-DLC test.

Logan's stance: not blocking; the improvements shipped to date are real
(the holdout deltas of TP +156, FP -53, FN -156 from mg=2 -> mg=0 are
apples-to-apples on new DLC). Leave calibration corpus on old DLC for
now. Re-infer on new DLC only if/when it becomes load-bearing.

## Recent decisions and their rationale

| Decision | Date | Rationale |
|---|---|---|
| Ship merge_gap=0 to production (v8.0.1) | 2026-05-18 | LOOCV ACCEPT + holdout PASS; +142 TP, -33 FP, -142 FN vs mg=2. |
| Set MIN_REPORTED_SPAN=4 for headline metrics | 2026-05-19 | Reaches < 4 frames are kinematically marginal; smoothness needs >= 4 frames; mice typically reach 6-15 frames. |
| Use Option 2 (post-match filter) not Option 1 (pre-match) | 2026-05-19 | Avoids asymmetric-exclusion artifacts where one side's match orphans into FP/FN. Cleaner semantics. |
| GT_ROOTS auto-resolve mapping for widget | 2026-05-19 | 14/14 + 19/19 clean resolution; eliminates "user picks wrong GT" risk. |
| Pragmatic re-score (no parquet rebuild) for analysis-only updates | 2026-05-19 | Model outputs are frozen at last LOOCV; only matching changes against updated GT. Cheap and consistent. |
| Add outside_gt_segmentation FP exclusion to headline metrics | 2026-05-20 | Phantoms in apparatus dead time (start_frame < B_first or > B_last) are physically impossible -- no pellet present, no detector fix possible. 4 cal-LOOCV FPs and 9 holdout FPs dropped under this rule. Algo unchanged; metric-side only. |

## Pointers to other documentation

- Widget schema (widget-owner): `SCHEMA.md` (in widget owner's project, not on Y:)
- Recalibration playbook: `Y:\2_Connectome\Behavior\MouseReach\RECALIBRATION_PLAYBOOK.md`
- Cardinal Rule nuance writeup: `Y:\2_Connectome\Behavior\MouseReach\plans\CARDINAL_RULE_NUANCE_2026-05-18.md`
- Prior session handoff: `Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough\REACH_DETECTION_HANDOFF_2026-05-05.md`
- Server-side memory (private to this Claude): `C:\Users\SCHULTZC\.claude\projects\y--2-Connectome\memory\reach_detection_state_2026-05-18.md`
- Production version change log: `Y:\2_Connectome\Behavior\MouseReach_Pipeline\pipeline_versions.json` (note: in MouseReach_Pipeline, not _Improvement)

## For the widget-side Claude resuming this work

Read this file. Then read the widget's `SCHEMA.md`. The integration
loop is locked in; what remains is implementation of the auto-resolve
behavior and atomic-write save semantics. The two GT_ROOTS paths above
are verified clean.

## For the server-side Claude resuming this work

Read this file. Then read the private memory file at
`reach_detection_state_2026-05-18.md` (it points here). After any GT
edit notification from Logan, run the re-score + manifest regen scripts
listed under "Server-side re-score loop." For calibration GT edits,
draft the pragmatic re-score helper script at first need; it doesn't
exist yet.
