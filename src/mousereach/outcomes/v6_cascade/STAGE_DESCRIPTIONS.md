# Cascade Stage Descriptions (v6.0.0_dev)

Canonical description of every cascade stage in plain English, including
empirical validation provenance for each stage. Update this file in
lockstep with any code change to the stage; the validation block must
reflect the *exact* configuration that produced the trust numbers
reported. If anything in the validation provenance changes (DLC model,
segmenter version, stage code, GT contents, commit-rule constants,
corpus snapshot), re-run validation and update the empirical result.

## Current cascade (2026-05-03 expanded)

| Stage | Class | Question |
|-------|-------|----------|
| 0  | (triage) | Is this an abnormally short segment (data-quality issue)? |
| 1  | untouched | Did the pellet's position never change? |
| 2  | untouched | Was the pellet stable on the pillar at end of segment? |
| 3  | untouched | Did no paw ever enter the pellet area? |
| 4  | untouched | Does the pellet return to the pillar after the last reach? |
| 5  | untouched | Was the pellet off-pillar throughout the segment? |
| 6  | untouched | Was the pellet predominantly on pillar despite minor noise? |
| 7  | displaced_sa | Did the pellet settle off-pillar in the SA late in the segment, with on→off transition at a reach? |
| 8  | displaced_sa | Did the pellet visibly displace from pillar to SA via the legacy multi-gate algorithm? |
| 9  | retrieved | Given the segment is touched, did post-reach evidence show the pellet vanished (rather than landing in the SA)? |
| 10 | displaced_sa | Brief-but-confident displacement evidence (5+ sustained off-pillar in-SA frames) with single-bout pre-evidence and no-bout post-evidence. |
| 11 | retrieved | Pellet effectively invisible throughout segment + paw activity (single-GT-reach restriction). |
| 12 | retrieved | Single-reach segment with post-reach pellet observations clustered ABOVE slit-y-line (in-mouth track) and zero below-slit observations. |
| 13 | retrieved | (disabled) Pillar-bodypart lk transition + pellet completely gone for single-reach -- no current residuals match cleanly. |
| 14 | displaced_sa | (disabled) Single-reach moderate displacement evidence -- false-positive rate too high to converge to 100% trust. |
| 15 | retrieved | (disabled) Multi-reach above-slit-vs-in-SA partition -- above-slit signal noisy in multi-reach segments. |
| 16 | displaced_sa | Pick max-pellet-displacement reach as causal (>= 1.5 radii, only one reach above threshold) with pre-causal-at-rest defense. |
| 17 | displaced_sa | Pick dominant-max-displacement reach (max >= 3x next-largest) for multi-disp-reach cases. |
| 18 | displaced_sa | (disabled) First-significant-displacement-reach pick -- 22% trust, GT typically picks LAST paw-over-pellet not FIRST. |
| 19 | retrieved | (disabled) Pillar-lk transition at first GT reach with multi-reach allowance -- bout-pick disagreement with GT. |
| 99 | (triage) | Catchall: anything left → human review |

**Validation regime:** GT segments + GT reaches as inputs (component-eval
mode — evaluates outcome detection in isolation from upstream
segmentation/reach errors).

**Trust definition (user-mandated):**
- untouched: class match + OKF±3
- touched: class match + same GT reach (algo's IFR falls in the same
  reach window as GT's IFR)
- triage: target segment was marked `expected_triage` in GT (e.g.,
  abnormal_exception, short-segment data-quality cases)

**Performance (corpus 2026-05-03 train_pool, 740 GT segments):**

| Class | Yield | Trust |
|-------|-------|-------|
| untouched | 285/285 non-triage GT (100%) | 100% on every commit |
| displaced_sa | 275/351 non-triage GT (78.3%) | 100% on every commit |
| retrieved | 45/91 non-triage GT (49.5%) | 100% on every commit |
| abnormal_exception | 0/0 non-triage GT (N/A) | All 7 expected_triage |
| Expected_triage handled | 13/13 (100%) | 100% routed to triage |
| **Wrongful commits** | **0** across all stages | |

**Per-stage commit counts (2026-05-03 expanded cascade):**

| Stage | Commits | Trust |
|-------|--------:|------:|
| 1 | 187 | 100% |
| 2 | 15 | 100% |
| 3 | 1 | 100% |
| 4 | 73 | 100% |
| 5 | 3 | 100% |
| 6 | 7 | 100% |
| 7 | 195 | 100% |
| 8 | 63 | 100% |
| 9 | 41 | 100% |
| 10 | 2 | 100% |
| 11 | 2 | 100% |
| 12 | 2 | 100% |
| 13 | 0 | (disabled) |
| 14 | 0 | (disabled) |
| 15 | 0 | (disabled) |
| 16 | 14 | 100% |
| 17 | 1 | 100% |
| 18 | 0 | (disabled) |
| 19 | 0 | (disabled) |

**Notes on disabled stages (2026-05-03 iteration log):**
The cascade-extension session produced 5 disabled stages (13, 14, 15,
18, 19). Each represented a different angle that didn't converge to
100% trust at any tested threshold:

- **Stage 13** (Pillar-lk transition for single-reach retrieved):
  Logically clean but no residuals match in current ordering --
  Stage 9's coverage already includes these patterns.
- **Stage 14** (single-reach moderate displacement evidence): brief-
  visibility retrieved cases trip the displacement gate. Without
  retrieved-vs-displaced discriminator beyond Stage 7's existing
  defenses, can't tighten without losing all yield.
- **Stage 15** (multi-reach above-slit-vs-in-SA partition):
  above-slit signal too noisy in multi-reach segments. Mouse-position
  variability + DLC pellet noise generates false positives.
- **Stage 18** (first-significant-displacement reach pick): 22% trust;
  GT typically picks LAST paw-over-pellet for displaced (not FIRST),
  even when displacement happened at first reach. Wrong-direction
  selection.
- **Stage 19** (Pillar-lk + first-GT-reach for multi-reach retrieved):
  17% trust; same issue as Stage 18 (wrong-direction reach selection).

Stages 10, 11, 12, 16, 17 are the WORKING extensions added in this
session. They contribute +21 correct commits over the prior cascade
state (+17 displaced, +4 retrieved) at 100% trust.

The detailed text in the sections below is mixed-vintage — current
section headers reflect the 2026-05-03 numbering, but per-stage
validation provenance blocks may be from earlier configurations and
should be regenerated when re-validating.

## Stage 1: Pellet position never changed (NEW 2026-05-02)

**Description:** The strongest possible untouched signal. Across all
high-confidence, non-during-reach frames in the segment's clean zone,
if the pellet's xy position (in pillar-relative coordinates) stays
within tight tolerance, the pellet did not move. By physical causality
(pellet motion requires reach), no reach-caused change occurred this
segment. Catches both pellet stable on pillar and pellet stationary
off-pillar (territory the renumbered later stages don't always catch
cleanly).

**Defenses against false-commit:**
- Position measured in **pillar-relative coordinates**, not absolute
  pixels. Apparatus tray drifts laterally over the ~30 s segment;
  absolute-coord stds reflect that drift, not pellet motion.
- Position must be stable in BOTH pre-reach AND post-reach windows
  separately (≥ 30 confident frames each, median position within 2 px).
  Otherwise displacement-during-reach with mostly post-displacement
  rest could pass the overall std check.

**Commit-rule constants:**
- `MIN_CONFIDENT_FRAMES = 100` (overall non-during-reach pellet-lk≥0.7)
- `PELLET_LK_THRESHOLD = 0.7`
- `MAX_POSITION_STD_PX = 1.0` (each axis, in pillar-relative pixels)
- `MIN_WINDOW_FRAMES = 30` (each of pre-reach and post-reach)
- `MAX_PRE_POST_DIST_PX = 2.0` (median pre vs post distance)

**Empirical (2026-05-02 component-eval, GT segs + GT reaches, 740 train-pool segments):**
- Commits: 187 / 187 trust pass (100.0% class+OKF±3)
- All commits are GT-untouched

(Module: `mousereach.outcomes.v6_cascade.stage_1_pellet_position_never_changed`)

## Stage 2: Pellet on pillar at end

**Description:** Excludes any possibly confounded cases and just asks if
the pellet ended up untouched at the end of the segment. We know that
the pellet couldn't have been eaten or displaced if the pellet is still
on the pillar at the end of the segment, and testing against ground
truth confirms that — see Validation Provenance below. Confound examples
include:

- Frames during any reach in the segment (the paw is in motion, may be
  over the pellet, may be moving the pellet)
- Frames in the few-frame transition right at the end of the segment
  where the apparatus is starting to move for the next pellet to be
  loaded (the reference points get jittery and observations there are
  unreliable)
- Brief moments at the end of the segment where the paw is close enough
  to the pellet that the paw could be obscuring or pushing the pellet
- Brief moments at the end of the segment where pellet detection
  confidence drops (the pellet might still be on the pillar but the
  camera isn't seeing it cleanly)
- Brief moments at the end of the segment where the pellet has drifted
  outside the calculated pillar position by even a small amount

Confounded cases are moved into Stage 2 for further examination.

### Validation provenance

- **Validation date:** 2026-05-01
- **Cascade stage code version:** mousereach v6.0.0_dev
  (`mousereach.outcomes.v6_cascade.stage_2_pellet_stable_untouched`)
- **Trust calibrator code version:**
  `mousereach.outcomes.v6_cascade.trust_calibrator` (commit on master at
  validation time)
- **DLC model:** `DLC_resnet50_MPSAOct27shuffle1_100000` from project
  `Y:/2_Connectome/Validation_Runs/DLC_2026_03_27/dlc_project`, run
  completed 2026-02-19 21:13:08+00:00
- **Segmenter version used to produce segment boundaries:** v2.1.2
- **Reach detector version used to produce reach windows:** v7.1.0 (GT
  reach windows used for this validation, not algo reaches)
- **Pillar geometry (per-frame, used to derive "inside pillar circle"):**
  smoothed-SA-position (3-frame centered moving average of SABL and
  SABR), then SA midpoint, ruler = SABL→SABR distance, pillar center =
  SA midpoint − 0.944 × ruler perpendicular (image-coords y), pillar
  radius = 0.10 × ruler. Source: `mousereach.lib.pillar_geometry`.
- **Stage 1 commit-rule constants:**
  - Throughout-segment stability: pellet-inside-circle frac ≥ 0.95
    across non-during-reach frames
  - Analysis window: 11 frames at `(seg_end − 15, seg_end − 5)`
  - Pellet-visible threshold: lk ≥ 0.7
  - Paw-near-pellet threshold: RightHand–Pellet distance <
    2.0 × pillar_radius (with RightHand_lk ≥ 0.7)
  - Transition zone half-width: 5 frames either side of segment boundary
  - Cascade OKF emit: `seg_end − 5` (last clean-zone frame)
- **Ground-truth corpus:** v4.0.0_dev_walkthrough quarantine snapshot at
  `Y:/2_Connectome/Validation_Runs/DLC_2026_03_27/iterations/2026-04-28_outcome_v4.0.0_dev_walkthrough/`,
  created 2026-04-28
- **GT corrections incorporated since corpus creation:**
  - 7 segments migrated from displaced/retrieved to abnormal_exception
    (2026-04-29, walkthrough cases 6, 8, 9, 10, 11, 14, 15)
  - 6 untouched-segment GT files patched 2026-05-01 to clear
    interaction_frame fields the GT tool would not let the GT-er nullify
    (audit trail: `gt_patch_2026-05-01_untouched_interaction_frame.md`)
- **GT correction applied at runtime by trust calibrator:** for any
  untouched segment whose GT outcome_known_frame is greater than
  `seg_end − 5`, the GT outcome_known_frame is "walked back" to
  `seg_end − 5` for trust comparison (logical: if outcome was knowable
  at the GT-er's later frame, it was knowable at every earlier frame in
  the current segment too)
- **CV-fold split:** train pool only (37 videos, 740 segments). Test
  holdout (10 videos, 200 segments) was NOT used in this validation; it
  remains frozen until Phase E.
- **Train pool video IDs (37 videos, 290 GT untouched segments among them):**
  `20250624_CNT0107_P3, 20250624_CNT0115_P2, 20250627_CNT0105_P1,
  20250630_CNT0104_P3, 20250701_CNT0110_P2, 20250701_CNT0111_P1,
  20250709_CNT0216_P3, 20250710_CNT0215_P4, 20250716_CNT0213_P3,
  20250806_CNT0311_P2, 20250806_CNT0312_P2, 20250807_CNT0307_P2,
  20250808_CNT0302_P3, 20250811_CNT0305_P2, 20250812_CNT0301_P3,
  20250813_CNT0314_P4, 20250820_CNT0103_P2, 20250820_CNT0103_P3,
  20250821_CNT0102_P2, 20250821_CNT0110_P4, 20250909_CNT0209_P4,
  20250910_CNT0216_P1, 20250912_CNT0209_P3, 20251007_CNT0316_P2,
  20251008_CNT0301_P4, 20251009_CNT0309_P1, 20251009_CNT0310_P2,
  20251010_CNT0305_P2, 20251010_CNT0308_P2, 20251021_CNT0401_P4,
  20251022_CNT0413_P4, 20251023_CNT0401_P4, 20251028_CNT0404_P4,
  20251029_CNT0408_P1, 20251030_CNT0403_P1, 20251031_CNT0407_P1,
  20251031_CNT0415_P1`
- **Test holdout video IDs (NOT used in this validation):**
  `20250626_CNT0102_P4, 20250708_CNT0210_P3, 20250711_CNT0210_P2,
  20250811_CNT0303_P4, 20250820_CNT0104_P2, 20250905_CNT0306_P2,
  20251007_CNT0314_P3, 20251009_CNT0307_P4, 20251024_CNT0402_P4,
  20251031_CNT0413_P2`

### Empirical result

- 740 segments evaluated (the entire train pool)
- 97 committed as untouched at this stage
- 643 deferred to Stage 2
- Trust at ±3-frame outcome-known-frame tolerance (after walk-back):
  **97/97 = 100%**
- Yield on the GT untouched class: **97/290 = 33.4%** of all train-pool
  ground-truth untouched segments captured at this stage
- Zero false commits on touched cases (no GT retrieved, displaced_sa, or
  abnormal_exception was ever wrongly committed as untouched)

## Stage 3: No reach that could have possibly touched the pellet

**Description:** Asks whether the mouse ever performed any reach during
the segment that could have possibly made contact with the pellet. A
"reach that could have possibly touched the pellet" requires, at
minimum, that the paw extended past the slit-closest pillar edge into
pellet-reaching territory at sustained tracking confidence -- without
that, contact was physically impossible regardless of where the pellet
was. If no such reach occurred anywhere across the entire segment
(excluding the few-frame transition zone at the segment boundary), the
segment can be confidently committed as untouched without needing to
know anything about the pellet's behavior. This is complementary to
Stage 1: Stage 1 watches the pellet (it's still on the pillar at the
end), Stage 2 watches the reach itself (none ever happened that could
have made contact). Confound examples include:

- Frames in the few-frame transition right at the end of the segment
  where the apparatus is starting to move for the next pellet to be
  loaded (DLC tracking gets unreliable -- excluded by trimming the
  analysis window to end at `seg_end - 5`)
- Single-frame DLC noise where one paw point is briefly hallucinated
  past the pillar y line at high confidence (filtered by requiring at
  least 3 consecutive past-line frames AND a 3-frame rolling mean of
  per-frame max paw confidence to reach the empirical floor)
- Brief moments where the paw is at the slit ledge but not actually
  extending across into pellet territory (the y-line gate excludes
  these because the paw hasn't crossed past the slit-closest pillar
  edge)
- Paw-near-but-not-past-pillar postures where one or two paw points
  flicker past the line at low confidence (filtered by the per-frame
  max-likelihood-across-paw-points predicate -- the most confident
  past-line paw point in each frame must average above the floor across
  3 consecutive frames)

Confounded cases are moved into Stage 3 for further examination.

### Validation provenance

- **Validation date:** 2026-05-01
- **Cascade stage code version:** mousereach v6.0.0_dev
  (`mousereach.outcomes.v6_cascade.stage_3_paw_never_in_pellet_area`)
- **Trust calibrator code version:**
  `mousereach.outcomes.v6_cascade.trust_calibrator` (commit on master at
  validation time)
- **DLC model:** `DLC_resnet50_MPSAOct27shuffle1_100000` from project
  `Y:/2_Connectome/Validation_Runs/DLC_2026_03_27/dlc_project`, run
  completed 2026-02-19 21:13:08+00:00
- **Segmenter version used to produce segment boundaries:** v2.1.2
- **Reach detector version used to produce reach windows:** v7.1.0 (GT
  reach windows used for this validation, not algo reaches)
- **Pillar geometry (per-frame, used to derive the slit-closest pillar
  y line):** smoothed-SA-position (3-frame centered moving average of
  SABL and SABR), then SA midpoint, ruler = SABL→SABR distance, pillar
  center = SA midpoint − 0.944 × ruler perpendicular (image-coords y),
  pillar radius = 0.10 × ruler. Slit-closest pillar y line =
  `pillar_cy + pillar_r` (in image coords; smaller y is upward toward
  the slit). Source: `mousereach.lib.pillar_geometry`.
- **Stage 2 commit-rule constants:**
  - Paw bodyparts evaluated: `RightHand`, `RHLeft`, `RHOut`, `RHRight`
  - In-zone predicate per paw point per frame: `paw_y <= pillar_cy +
    pillar_r` (paw point is at or past the slit-closest pillar edge --
    that is, on the slit side of that edge in image coords)
  - Per-frame in-zone max-likelihood: `max(paw_lk for paw in paws if
    paw_y <= pillar_cy + pillar_r)`
  - Minimum consecutive in-zone frames to even consider the run: 3
  - Rolling-mean window for sustained-confidence test: 3 frames
  - Empirical likelihood floor for the 3-frame rolling mean of the
    per-frame in-zone max-likelihood: **0.22** (the absolute lowest
    sustained-3-frame max-likelihood observed across all 381
    GT-confirmed causal contact reaches in the train-pool corpus -- if
    the rolling mean never reaches 0.22 at any point in the segment's
    clean zone, no real contact reach in the corpus ever sustained the
    paw confidently in pellet territory at that level either)
  - Commit trigger: in the clean zone, **no** 3-frame rolling-mean
    in-zone max-likelihood ever reaches the floor (i.e., the mouse
    never confidently extended the paw past the slit-closest pillar y
    line for 3 sustained frames)
  - Transition zone half-width: 5 frames either side of segment
    boundary (clean zone for Stage 2 = `[seg_start, seg_end - 5]`)
  - Cascade OKF emit on commit: `seg_end − 5` (last clean-zone frame)
- **Ground-truth corpus:** v4.0.0_dev_walkthrough quarantine snapshot at
  `Y:/2_Connectome/Validation_Runs/DLC_2026_03_27/iterations/2026-04-28_outcome_v4.0.0_dev_walkthrough/`,
  created 2026-04-28
- **Empirical-floor derivation corpus:** all 381 GT-confirmed causal
  contact reaches in the train-pool 37 videos (the same train-pool
  GT-confirmed reaches whose contact frames sit by definition in the
  in-zone region). For each reach the per-frame in-zone max-likelihood
  was computed across the 4 paw bodyparts using the y-line gate; the
  3-frame rolling mean was computed across the entire reach window;
  the minimum of that rolling mean across the reach was recorded; the
  absolute minimum across all 381 reaches was 0.22.
- **GT corrections incorporated since corpus creation:**
  - 7 segments migrated from displaced/retrieved to abnormal_exception
    (2026-04-29, walkthrough cases 6, 8, 9, 10, 11, 14, 15)
  - 6 untouched-segment GT files patched 2026-05-01 to clear
    interaction_frame fields the GT tool would not let the GT-er nullify
    (audit trail: `gt_patch_2026-05-01_untouched_interaction_frame.md`)
- **GT correction applied at runtime by trust calibrator:** for any
  untouched segment whose GT outcome_known_frame is greater than
  `seg_end − 5`, the GT outcome_known_frame is "walked back" to
  `seg_end − 5` for trust comparison (logical: if outcome was knowable
  at the GT-er's later frame, it was knowable at every earlier frame in
  the current segment too)
- **CV-fold split:** train pool only (37 videos, 740 segments). Test
  holdout (10 videos, 200 segments) was NOT used in this validation; it
  remains frozen until Phase E.
- **Train pool video IDs (37 videos, 290 GT untouched segments among them):**
  `20250624_CNT0107_P3, 20250624_CNT0115_P2, 20250627_CNT0105_P1,
  20250630_CNT0104_P3, 20250701_CNT0110_P2, 20250701_CNT0111_P1,
  20250709_CNT0216_P3, 20250710_CNT0215_P4, 20250716_CNT0213_P3,
  20250806_CNT0311_P2, 20250806_CNT0312_P2, 20250807_CNT0307_P2,
  20250808_CNT0302_P3, 20250811_CNT0305_P2, 20250812_CNT0301_P3,
  20250813_CNT0314_P4, 20250820_CNT0103_P2, 20250820_CNT0103_P3,
  20250821_CNT0102_P2, 20250821_CNT0110_P4, 20250909_CNT0209_P4,
  20250910_CNT0216_P1, 20250912_CNT0209_P3, 20251007_CNT0316_P2,
  20251008_CNT0301_P4, 20251009_CNT0309_P1, 20251009_CNT0310_P2,
  20251010_CNT0305_P2, 20251010_CNT0308_P2, 20251021_CNT0401_P4,
  20251022_CNT0413_P4, 20251023_CNT0401_P4, 20251028_CNT0404_P4,
  20251029_CNT0408_P1, 20251030_CNT0403_P1, 20251031_CNT0407_P1,
  20251031_CNT0415_P1`
- **Test holdout video IDs (NOT used in this validation):**
  `20250626_CNT0102_P4, 20250708_CNT0210_P3, 20250711_CNT0210_P2,
  20250811_CNT0303_P4, 20250820_CNT0104_P2, 20250905_CNT0306_P2,
  20251007_CNT0314_P3, 20251009_CNT0307_P4, 20251024_CNT0402_P4,
  20251031_CNT0413_P2`
- **Cascade ordering at validation:** Stage 1 ran first; Stage 2 was
  evaluated only on the 643 segments that Stage 1 deferred. Stage 2's
  empirical numbers are therefore conditional on Stage-1-deferred input
  and not directly comparable to a hypothetical "Stage 2 only" run.

### Empirical result

- 643 segments evaluated (the Stage 1 deferral pool: 740 train-pool
  segments minus Stage 1's 97 commits)
- 10 committed as untouched at this stage
- 633 deferred to Stage 3
- Trust at ±3-frame outcome-known-frame tolerance (after walk-back):
  **10/10 = 100%**
- Yield on the GT untouched class at this stage:
  **10/(290 − 97) = 10/193 = 5.2%** of the still-residual GT untouched
  segments after Stage 1
- Cumulative yield on the GT untouched class through Stages 1+2:
  **(97 + 10)/290 = 107/290 = 36.9%** of all train-pool ground-truth
  untouched segments captured by the cascade so far
- Zero false commits on touched cases (no GT retrieved, displaced_sa,
  or abnormal_exception was ever wrongly committed as untouched at this
  stage)

## Stage 4: Pellet observed back on pillar after the last reach attempt

**Description:** Asks whether the pellet, after the last reach attempt
in the segment, is observed sustained back on the pillar without any
sign that the actual pillar has become exposed (which would indicate
the pellet is no longer there). The reasoning rests on a physical
constraint: a pellet that was actually displaced cannot return to the
pillar -- the mouse cannot put it back, and gravity does not arrange
for an off-pillar pellet to climb back onto the pillar surface. So
if the pellet is observed sustained on the pillar at any point AFTER
the last reach attempt (with a settling buffer to let the pellet come
to rest), no reach in this segment can have displaced it. A second,
complementary check guards against a known DLC failure mode where the
pillar gets relabeled as the pellet after a real displacement: when
the actual pellet is gone, the now-exposed pillar becomes visible and
the raw `Pillar` bodypart rises to confident detection. Stage 3
therefore requires both (a) sustained pellet-on-pillar evidence after
the last reach, AND (b) the raw Pillar bodypart never reaching
confident detection in the same window. Confound examples include:

- Frames during the few-frame transition right at the end of the
  segment where the apparatus is starting to move for the next pellet
  to be loaded (excluded by trimming the analysis window to end at
  `seg_end - 5`)
- Frames during any reach attempt in the segment where any paw point
  is past the slit-closest pillar y line and could be occluding the
  pellet (excluded by the same paw-past-y-line predicate Stage 2
  uses)
- Brief moments immediately after a reach ends where the pellet may
  be wobbling or jittering at the pillar surface before settling
  back to a stable position (handled by a 15-frame post-bout settling
  buffer applied AFTER the last paw-past-y-line bout ends)
- DLC label-switch cases where the now-exposed pillar after a real
  displacement gets re-labeled as "Pellet" by DLC at high lk -- these
  segments would falsely satisfy the pellet-on-pillar check; rejected
  by the simultaneous requirement that the raw Pillar bodypart's
  likelihood stays below the displacement-signal threshold across the
  same eligible window
- Single-frame DLC failures on SA bodyparts (lk near 0, position
  jumping wildly) that contaminate per-frame pillar geometry --
  handled upstream by the `mousereach.lib.dlc_cleaning` impossibility
  filter applied to SA bodyparts before computing pillar geometry,
  AND to the Pellet bodypart before checking on-pillar position
- Whole-frame recording artifacts (camera bumps, dropped frames)
  where the static landmarks BOXL/BOXR/Reference also move from
  their physically-immovable positions -- handled by the same
  cleaning filter via Tier 3 (frame-level artifact detection)

Confounded cases are moved into Stage 4 for further examination.

### Validation provenance

- **Validation date:** 2026-05-01
- **Cascade stage code version:** mousereach v6.0.0_dev
  (`mousereach.outcomes.v6_cascade.stage_4_pellet_returns_to_pillar`)
- **Trust calibrator code version:**
  `mousereach.outcomes.v6_cascade.trust_calibrator` (commit on master at
  validation time)
- **DLC bodypart cleaning filter version:**
  `mousereach.lib.dlc_cleaning` v1.0 (added 2026-05-01). Three-tier
  impossibility predicates: (1) per-bodypart lk < 0.5; (2) per-SA-
  bodypart deviation > 15 px from 5-frame rolling median; (3) per-
  frame BOXL/BOXR/Reference frame-to-frame motion > 5 px. Default
  thresholds derived empirically across the same 37 train-pool videos
  (1.4M frames) on 2026-05-01.
- **Pillar geometry function:**
  `mousereach.lib.pillar_geometry.compute_pillar_geometry_series_cleaned`
  (added 2026-05-01) -- applies the impossibility filter to SA
  bodyparts before computing the per-frame pillar geometry, so
  single-frame DLC failures on SA do not contaminate per-frame pillar
  position/radius.
- **DLC model:** `DLC_resnet50_MPSAOct27shuffle1_100000` from project
  `Y:/2_Connectome/Validation_Runs/DLC_2026_03_27/dlc_project`, run
  completed 2026-02-19 21:13:08+00:00
- **Segmenter version used to produce segment boundaries:** v2.1.2
- **Reach detector version used to produce reach windows:** v7.1.0 (GT
  reach windows used for this validation, not algo reaches; Stage 3
  itself does not consume reach windows directly -- it derives reach
  attempts from the per-frame paw-past-y-line predicate)
- **Stage 3 commit-rule constants:**
  - Paw bodyparts evaluated: `RightHand`, `RHLeft`, `RHOut`, `RHRight`
  - Paw-past-y-line predicate: `paw_y <= pillar_cy + pillar_r AND
    paw_lk >= 0.5`
  - "Pellet on pillar" predicate:
    `Pellet_lk >= 0.7 AND dist(Pellet, pillar_center) <= 1.2 *
    pillar_radius AND not paw-past-y-line at this frame`
  - Pillar-buffer factor for "on pillar": **1.2** (1 radius for the
    physical pillar circle plus 0.2 buffer for pellet's own image-
    space size and detection edge noise)
  - Post-bout settling buffer after the last paw-past-y-line bout
    ends: **15 frames** (lets the pellet settle at the pillar surface
    if it was wobbled by the reach attempt)
  - Sustained-pellet-on-pillar: **3+ consecutive eligible frames** in
    the post-settling window
  - Pillar-bodypart-rises rejection threshold: **`Pillar_lk >= 0.5`**
    at any point in the post-settling, paw-not-past-y-line window
    causes the segment to defer to Stage 4 regardless of any
    pellet-on-pillar evidence (because the now-exposed pillar
    becoming confidently detected is a positive signal that the
    pellet has actually been displaced)
  - Transition zone half-width: 5 frames either side of segment
    boundary (clean zone for Stage 3 = `[seg_start, seg_end - 5]`)
  - Cascade OKF emit on commit: `seg_end − 5` (last clean-zone frame),
    parallel to Stages 1 and 2
- **Pillar-bodypart-rises empirical derivation:** across all 705
  GT-labeled segments in the 37 train-pool videos with usable
  post-event windows (excluding the few segments too short to have a
  clean zone), the maximum raw `Pillar_likelihood` in the post-event,
  paw-not-past-y-line eligible window:
  - For 341 GT `displaced_sa` segments: 100% had max Pillar_lk >= 0.5
    (sustained at 0.998+ across the full post-event window); p50 of
    max = 0.999
  - For 89 GT `retrieved` segments: 100% had max Pillar_lk >= 0.5
    (sustained); p50 of max = 0.999
  - For 275 GT `untouched` segments: 9.8% had max Pillar_lk >= 0.5;
    p50 of max = 0.237; p90 = 0.494
  - Threshold of 0.5 sits cleanly between the displaced/retrieved
    distribution (which is essentially "always above") and the
    untouched distribution (which is essentially "always below" with
    a small mis-tracking tail). The 9.8% of untouched that sit on
    the wrong side of the threshold defer to Stage 4 -- they are not
    falsely committed.
- **Ground-truth corpus:** v4.0.0_dev_walkthrough quarantine snapshot
  at
  `Y:/2_Connectome/Validation_Runs/DLC_2026_03_27/iterations/2026-04-28_outcome_v4.0.0_dev_walkthrough/`,
  created 2026-04-28
- **GT corrections incorporated since corpus creation:**
  - 7 segments migrated from displaced/retrieved to abnormal_exception
    (2026-04-29, walkthrough cases 6, 8, 9, 10, 11, 14, 15)
  - 6 untouched-segment GT files patched 2026-05-01 to clear
    interaction_frame fields the GT tool would not let the GT-er
    nullify (audit trail:
    `gt_patch_2026-05-01_untouched_interaction_frame.md`)
- **GT correction applied at runtime by trust calibrator:** for any
  untouched segment whose GT outcome_known_frame is greater than
  `seg_end − 5`, the GT outcome_known_frame is "walked back" to
  `seg_end − 5` for trust comparison
- **CV-fold split:** train pool only (37 videos, 740 segments). Test
  holdout (10 videos, 200 segments) was NOT used in this validation;
  it remains frozen until Phase E.
- **Train pool video IDs (37 videos, 290 GT untouched segments among them):**
  `20250624_CNT0107_P3, 20250624_CNT0115_P2, 20250627_CNT0105_P1,
  20250630_CNT0104_P3, 20250701_CNT0110_P2, 20250701_CNT0111_P1,
  20250709_CNT0216_P3, 20250710_CNT0215_P4, 20250716_CNT0213_P3,
  20250806_CNT0311_P2, 20250806_CNT0312_P2, 20250807_CNT0307_P2,
  20250808_CNT0302_P3, 20250811_CNT0305_P2, 20250812_CNT0301_P3,
  20250813_CNT0314_P4, 20250820_CNT0103_P2, 20250820_CNT0103_P3,
  20250821_CNT0102_P2, 20250821_CNT0110_P4, 20250909_CNT0209_P4,
  20250910_CNT0216_P1, 20250912_CNT0209_P3, 20251007_CNT0316_P2,
  20251008_CNT0301_P4, 20251009_CNT0309_P1, 20251009_CNT0310_P2,
  20251010_CNT0305_P2, 20251010_CNT0308_P2, 20251021_CNT0401_P4,
  20251022_CNT0413_P4, 20251023_CNT0401_P4, 20251028_CNT0404_P4,
  20251029_CNT0408_P1, 20251030_CNT0403_P1, 20251031_CNT0407_P1,
  20251031_CNT0415_P1`
- **Test holdout video IDs (NOT used in this validation):**
  `20250626_CNT0102_P4, 20250708_CNT0210_P3, 20250711_CNT0210_P2,
  20250811_CNT0303_P4, 20250820_CNT0104_P2, 20250905_CNT0306_P2,
  20251007_CNT0314_P3, 20251009_CNT0307_P4, 20251024_CNT0402_P4,
  20251031_CNT0413_P2`
- **Cascade ordering at validation:** Stage 1 → Stage 2 → Stage 3.
  Stage 3 was evaluated only on the 633 segments that Stage 2
  deferred (after Stage 1 committed 97 and Stage 2 committed 10).
  Stage 3's empirical numbers are therefore conditional on
  Stage-2-deferred input.
- **Known limitation (deferred to later stages):** Stage 3 cannot
  distinguish between (i) a true untouched segment where the pellet
  was loaded off-center by the operator and never sat at the calc'd
  pillar position, and (ii) a segment that needs displacement-class
  treatment. Both paths defer at this stage, so neither is falsely
  committed -- they wait for later stages designed for those classes.

### Empirical result

- 633 segments evaluated (the Stage 2 deferral pool: 740 train-pool
  segments minus Stage 1's 97 and Stage 2's 10 commits)
- 150 committed as untouched at this stage
- 483 deferred to Stage 4
- Trust at ±3-frame outcome-known-frame tolerance (after walk-back):
  **150/150 = 100%**
- Yield on the GT untouched class at this stage:
  **150/(290 − 97 − 10) = 150/183 = 82.0%** of the still-residual GT
  untouched segments after Stages 1 and 2
- Cumulative yield on the GT untouched class through Stages 1+2+3:
  **(97 + 10 + 150)/290 = 257/290 = 88.6%** of all train-pool
  ground-truth untouched segments captured by the cascade so far
- Cumulative whole-corpus state after Stages 1+2+3:
  - 257 of 740 segments committed (all as untouched, all correct)
  - 483 deferred (33 GT untouched still pending, plus 351 displaced_sa
    + 92 retrieved + 7 abnormal_exception that future cascade stages
    will commit)
- Zero false commits on touched cases (no GT retrieved, displaced_sa,
  or abnormal_exception was ever wrongly committed as untouched at
  this stage; the Pillar-bodypart-rises rejection caught all 10
  candidate false commits cleanly)

## Stage 5: Pellet was off-pillar from segment start and stayed off

**Description:** Asks whether the pellet entered this segment already
off the pillar (sitting somewhere in the scoring area from a prior
segment's displacement, or operator-loaded off-center) and never
returned to the pillar throughout the segment. The reasoning: a
segment is GT untouched if no reach IN THAT SEGMENT caused
interaction. A pellet that came in already displaced is fair game
for a Stage-3-deferring untouched commit as long as nothing in the
current segment moves it back onto the pillar (physically impossible
per the existing memory rule) and nothing apparent-retrieves it (also
impossible from the off-pillar starting state). The mouse is
allowed to move the pellet around within the SA -- that's still
untouched in the GT sense. Two artifact patterns are also caught
here as TRIAGE outputs (segments needing human review): the Pellet+
Pillar co-detection artifact (impossible co-localization at high
lk -- often from plexiglass over the SA), and the
pellet-appears-retrieved-from-off-pillar-state pattern (no sustained
pellet detection after the segment-start off-pillar evidence --
would imply impossible retrieval). Confound examples include:

- Frames during the few-frame transition right at the end of the
  segment (excluded by trimming the analysis window to end at
  `seg_end - 5`).
- Tray motion ongoing at segment start (segmenter's boundary lands
  near a tray event and the tray takes time to fully settle): the
  pellet's apparent position drifts during settling, and post-
  settling DLC tracking can have small (~2 px) calibration offsets
  from the calc'd pillar center. Handled by the SA-stability
  precondition: skip past the segment start until the SA centroid
  velocity stays below the stable threshold for sustained frames,
  and only evaluate the pellet's "off-pillar at start" check from
  that settled point onward.
- Pellet sitting at the pillar's edge (not actually displaced) but
  reading 1-2 radii from calc'd pillar center due to 3D projection
  or per-video tracking offsets. Handled by the off-pillar threshold
  of 3.0 radii, well above pillar-edge artifacts (which sit at 1-2
  radii) and below the bulk of true displaced-pellet positions
  (empirically p10=3.6, p50=4.5 radii from pillar center after
  displacement).
- Real displacement happening within the first 30 frames of the
  segment (mouse can displace the pellet quickly, segmenter doesn't
  pad reaches): the segment-start window is CAPPED at the first
  paw-past-y-line bout, so post-reach off-pillar evidence cannot
  count as "started off-pillar."
- Pellet wobbling back onto the pillar after segment start: the
  "never returns to pillar" check rejects any segment where the
  pellet is observed sustained ON pillar (within `1.2 * pillar_r`,
  matching Stage 3's on-pillar buffer) for 5+ consecutive eligible
  frames after the segment-start window. Either our segment-start
  detection was wrong, or the pellet impossibly returned -- defer.
- DLC bodypart impossibility events (lk near 0 with random
  positions, single-frame outliers, whole-frame artifacts): handled
  upstream by `mousereach.lib.dlc_cleaning` applied to SA + Pellet.

Confounded cases either defer (continue) or triage (out-of-cascade).

### Validation provenance

- **Validation date:** 2026-05-02
- **Cascade stage code version:** mousereach v6.0.0_dev
  (`mousereach.outcomes.v6_cascade.stage_5_pellet_off_pillar_throughout`)
- **Trust calibrator code version:**
  `mousereach.outcomes.v6_cascade.trust_calibrator` (commit on master
  at validation time)
- **DLC bodypart cleaning filter version:**
  `mousereach.lib.dlc_cleaning` v1.0 (default thresholds: lk < 0.5
  for Tier 1, 15 px deviation from rolling median for Tier 2,
  5 px frame-to-frame motion of static landmarks for Tier 3).
- **Pillar geometry function:**
  `mousereach.lib.pillar_geometry.compute_pillar_geometry_series_cleaned`
  (cleans SA bodyparts before computing per-frame geometry).
- **DLC model:** `DLC_resnet50_MPSAOct27shuffle1_100000` from
  project `Y:/2_Connectome/Validation_Runs/DLC_2026_03_27/dlc_project`,
  run completed 2026-02-19 21:13:08+00:00.
- **Segmenter version:** v2.1.2.
- **Reach detector version:** v7.1.0 (GT reach windows used; Stage 4
  itself derives reach attempts from the per-frame paw-past-y-line
  predicate).
- **Stage 4 commit-rule constants:**
  - Paw bodyparts evaluated: `RightHand`, `RHLeft`, `RHOut`, `RHRight`.
  - Paw-past-y-line predicate:
    `paw_y <= pillar_cy + pillar_r AND paw_lk >= 0.5`.
  - Co-detection triage predicate (FIRST priority -- emits triage):
    `Pellet_lk >= 0.7 AND Pillar_lk >= 0.7 AND
    dist(Pellet, Pillar) <= 1.0 * pillar_r`, sustained 3+
    consecutive frames anywhere in clean zone.
  - SA stability precondition: SA centroid frame-to-frame velocity
    `< 2.0 px/frame` for 5+ consecutive frames defines the
    "settled" point; segment-start window starts there. Search
    range is the first 200 clean-zone frames (defer if SA never
    settles).
  - Segment-start window: `[settled_idx, settled_idx + 30)`, capped
    further at the first paw-past-y-line bout. Off-pillar evidence
    must be in this window, BEFORE any reach.
  - Off-pillar threshold: `pellet_dist > 3.0 * pillar_r`. Empirically
    grounded -- GT displaced_sa pellets sit at p10=3.6, p50=4.5,
    p90=6.3 radii after displacement; threshold of 3.0 is below
    bulk and clear of pillar-edge artifacts.
  - Sustained off-pillar evidence: 5+ consecutive eligible frames
    (paw-not-past-y-line + pellet at lk >= 0.7 + dist > 3.0r).
  - Never-returns-to-pillar check: across the rest of the clean zone
    (after segment-start window), no run of 5+ consecutive frames
    with `pellet_dist <= 1.2 * pillar_r AND pellet_lk >= 0.7 AND
    paw_not_past_y_line` (matches Stage 3's on-pillar definition).
    If observed, defer.
  - Pellet-remains-observable check (ELSE TRIAGE): after segment-
    start window, must observe a 5+ consecutive run of `pellet_lk
    >= 0.7 AND paw_not_past_y_line`. If not, segment is anomalous
    (impossible retrieval from off-pillar state) and routed to
    triage.
  - Transition zone half-width: 5 frames at the segment end (clean
    zone for Stage 4 = `[seg_start, seg_end - 5]`).
  - Cascade OKF emit on commit: `seg_end - 5` (parallel to
    Stages 1, 2, 3).
- **Off-pillar threshold empirical derivation:** for all 337 GT
  displaced_sa segments in the train pool with non-null
  interaction_frame, computed median pellet-distance-from-pillar
  (radii) across confident-pellet frames in the late-segment window
  (50+ frames after IFR through `seg_end - 5`). Distribution: p1=2.17,
  p5=3.31, p10=3.60, p25=3.96, p50=4.49, p75=5.35, p90=6.31, p95=7.33,
  p99=7.88, max=8.84. Pillar-to-SA distance (apparatus geometry) is
  9.44 radii. Off-pillar threshold of 3.0 catches >=p5 of GT
  displaced (95% sensitivity) while excluding pillar-edge calibration
  artifacts.
- **Co-detection triage empirical derivation:** documented in memory
  rule `pellet_pillar_cooccurrence_is_artifact.md` (2026-05-01).
  Threshold of 1.0 pillar-radius for "Pellet and Pillar at the same
  location" is the user-mandated value reflecting the physical
  occlusion constraint (pellet sits ON pillar; both can't be visible
  at the same image position simultaneously).
- **Ground-truth corpus:** v4.0.0_dev_walkthrough quarantine snapshot
  at
  `Y:/2_Connectome/Validation_Runs/DLC_2026_03_27/iterations/2026-04-28_outcome_v4.0.0_dev_walkthrough/`,
  created 2026-04-28.
- **GT corrections incorporated since corpus creation:** same as
  Stages 1-3 (no additional corrections specific to Stage 4).
- **GT correction applied at runtime by trust calibrator:** for any
  untouched segment whose GT outcome_known_frame is greater than
  `seg_end - 5`, walked back to `seg_end - 5` for trust comparison.
- **CV-fold split:** train pool only (37 videos, 740 segments). Test
  holdout (10 videos, 200 segments) frozen until Phase E.
- **Train pool / test holdout video IDs:** same as Stages 1-3.
- **Cascade ordering at validation:** Stage 1 -> 2 -> 3 -> 4. Stage 4
  was evaluated only on the 483 segments that Stages 1, 2, and 3
  deferred. Stage 4's empirical numbers are conditional on Stage-3-
  deferred input.

### Empirical result

- 483 segments evaluated (Stage 3's deferral pool: 740 minus
  S1's 97, minus S2's 10, minus S3's 150 commits).
- 3 committed as untouched at this stage.
- 22 triaged (1 GT abnormal_exception, 10 GT displaced_sa, 3 GT
  retrieved, 8 GT untouched).
- 458 deferred to Stage 5.
- Trust at +/-3-frame outcome-known-frame tolerance (after walk-back):
  **3/3 = 100%**.
- Yield on the GT untouched class at this stage:
  **3/(290 - 97 - 10 - 150) = 3/33 = 9.1%** of the still-residual
  GT untouched segments after Stages 1-3.
- Cumulative yield on the GT untouched class through Stages 1-4:
  **(97 + 10 + 150 + 3)/290 = 260/290 = 89.7%** of all train-pool
  GT untouched segments captured by the cascade so far.
- Triage breakdown (segments routed to manual review, not
  committed): co-detection artifacts (typically plexiglass-over-SA
  videos) capture some untouched + displaced + retrieved + abnormal
  cases; apparent-retrieval-from-off-pillar-state captures cases
  where the pellet was off-pillar at start but never reappeared.
  Both categories are out-of-scope for cascade auto-commit and
  correctly punt to human review.
- Zero false commits on touched cases (no GT retrieved, displaced_sa,
  or abnormal_exception was ever wrongly committed as untouched at
  this stage).

### Known limitations

- Yield on the GT untouched class is low (3/33 = 9.1% of Stage-3-
  deferred untouched). The remaining 30 GT untouched in residual
  after Stage 4 are likely cases where: (a) the pellet's tracked
  position is in a structural calibration grey zone (1.5-3.0 radii
  from calc'd pillar center) -- close enough to the pillar that it
  doesn't satisfy the 3.0-radii off-pillar threshold but far enough
  that Stage 1/Stage 3 didn't commit; (b) per-video DLC pellet
  tracking offsets put the pellet in the calibration grey zone
  systematically (e.g., `20250710_CNT0215_P4` segment 7 -- pellet
  visually on pillar but tracked at 1.5 radii sustained throughout).
  These cases are deferred rather than committed, preserving 100%
  trust at the cost of yield. Future cascade work could add a
  per-video calibration check or a stage targeted at the calibration
  grey zone specifically.

## Stage 9: Pellet vanished after reach (retrieved) (NEW 2026-05-03)

**Reframe:** by the time a segment reaches Stage 9, the upstream
cascade has already excluded `untouched`. By apparatus physics the
pellet was on the pillar at segment start (ASPA loads it before each
segment). So Stage 9 does NOT need to re-prove "pellet was on pillar
early" -- that's redundant with upstream. The stage answers only the
remaining question: given the segment is touched, does post-reach
evidence show the pellet vanished (retrieved) rather than landing in
the SA (displaced)?

**Causal reach pick (GT semantic = "last paw-over-pellet"):**
- Walk LEFT-to-RIGHT through reaches; the FIRST reach satisfying
  post-vanish conditions is the reach AFTER WHICH the pellet first
  becomes invisible. Subsequent reaches are paw-over-empty-pillar
  (they trivially also pass post-vanish, but they're not causal).
- Chain forward at gaps ≤ 20 frames so back-to-back contested
  reaches resolve to the LAST in chain (matches GT's "last paw-over-
  pellet" semantic for chained retrieval actions).

**Trust-preserving defenses (no wrongful commits):**
- Late-visibility: pellet must be essentially gone (≤10% confident
  observations) in the late half of the clean zone.
- Anti-displaced: from the FIRST reach onward, sustained off-pillar
  pellet observations (`pellet_lk≥0.7 AND dist>1.5r AND not paw-past-
  slit`, sustained ≥3 consecutive frames) must total ≤5 frames. Any
  more = pellet is still in apparatus = displaced, not retrieved.
- Post-causal vanish budget: from causal reach end onward, sustained
  pellet observations must total ≤5 frames. Brief in-mouth track
  consumed but anything more rejects.
- Too-many-skipped-reaches: if more than 3 earlier reaches failed
  post-conditions because the pellet was visible after them, the
  pellet was around for many reaches and the "clean" pick later is a
  contested-displaced case, not a clean retrieval. Defer.
- Uncovered-paw defense: sustained un-annotated paw-past-slit
  activity (using strict paw filter: ≥2 paw bodyparts at lk≥0.7,
  empirical 2026-05-03) before the chain start = unlabeled causal
  reach; defer.

**Cascade emit on commit:**
- `committed_class`: "retrieved"
- `whens["interaction_frame"]`: middle of causal reach window
- `whens["outcome_known_frame"]`: causal reach end + 5

**Commit-rule constants (current):**
- `LATE_FRACTION = 0.5`
- `MAX_LATE_VISIBILITY_FRAC = 0.1`
- `PELLET_LK_THR = 0.95` (for confident pellet-on-pillar)
- `PAW_LK_THR = 0.5`
- `ON_PILLAR_RADII = 1.0`
- `CHAIN_GAP_THRESHOLD = 20`
- `MIN_SUSTAINED_PELLET_RUN = 3`
- `MAX_POST_OFF_PILLAR_FOR_RETRIEVED = 5`
- `MAX_POST_ANY_PELLET_OBS = 5`
- `MAX_INSEG_OFF_PILLAR_IN_SA = 5` (used as anti-displaced budget)
- `MAX_SKIPPED_REACHES = 3`
- `MAX_UNCOVERED_PAW_IN_GAP = 5`
- `MIN_UNCOVERED_PAW_RUN = 10`
- `OKF_VANISH_OFFSET = 5`

**Empirical (2026-05-03, train_pool, 740 GT segments, GT segs + GT
reaches):**
- 41 committed (all retrieved class) at 100% trust (class match +
  same GT reach for the IFR anchor).
- 50 GT-retrieved segments still in residual at end of cascade,
  routed to manual review.
- Yield on the GT retrieved class: 41/91 non-triage GT = 45.1%.
- Zero wrongful commits across the whole cascade.

(Module: `mousereach.outcomes.v6_cascade.stage_9_pellet_vanished_after_reach`)
