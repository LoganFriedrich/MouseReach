# Cascade Stage Descriptions (v6.0.0_dev)

> This file describes the outcome cascade at version 6.0.0_dev. The cascade that runs
> today is version 6.1.0: 33 stages (0, 1, 2, 3, 4, 5, 6, 6b, 7-29, 98, 99), and several
> stages below behave differently (for example Stage 0 is bypassed, Stage 13 is enabled,
> Stage 5 defers instead of triaging on a Pellet+Pillar co-detection). For what each stage
> does now, read `docs/ALGO_3_OUTCOME_CASCADE.md`; the stage code in this folder is
> authoritative.

A plain-English description of each cascade stage and the reason for its rule.

## Current cascade

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
| 18 | displaced_sa | (disabled) First-significant-displacement-reach pick -- low trust, hand-marked causal reach is typically the LAST paw-over-pellet, not the FIRST. |
| 19 | retrieved | (disabled) Pillar-lk transition at first GT reach with multi-reach allowance -- bout-pick disagreement with GT. |
| 99 | (triage) | Catchall: anything left → human review |

**Validation regime:** hand-marked segments and reaches as inputs (component evaluation
mode — evaluates outcome detection in isolation from upstream
segmentation/reach errors).

**Trust definition:**
- untouched: class match + OKF±3
- touched: class match + same GT reach (algo's IFR falls in the same
  reach window as GT's IFR)
- triage: target segment was marked `expected_triage` in GT (e.g.,
  abnormal_exception, short-segment data-quality cases)

**Disabled stages (13, 14, 15, 18, 19).** Each represented an angle that did not
reach 100% trust at any tested threshold, so it commits nothing in this version:

- **Stage 13** (pillar-likelihood transition for single-reach retrieved): logically clean, but
  Stage 9's coverage already includes these patterns.
- **Stage 14** (single-reach moderate displacement evidence): brief-visibility retrieved cases
  trip the displacement gate; without a retrieved-vs-displaced discriminator beyond Stage 7's
  defenses it cannot be tightened without losing all yield.
- **Stage 15** (multi-reach above-slit-vs-in-SA partition): the above-slit signal is too noisy in
  multi-reach segments (mouse-position variability plus pellet-tracking noise gives false positives).
- **Stage 18** (first-significant-displacement reach pick): low trust, because the hand-marked
  causal reach for a displaced pellet is typically the LAST paw-over-pellet reach, not the first,
  even when the displacement happened at the first reach.
- **Stage 19** (pillar likelihood plus first reach for multi-reach retrieved): low trust for the
  same wrong-direction reach selection as Stage 18.

The detailed text in the sections below may describe earlier configurations of a stage.

## Stage 1: Pellet position never changed

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

(Module: `mousereach.outcomes.v6_cascade.stage_1_pellet_position_never_changed`)

## Stage 2: Pellet on pillar at end

**Description:** Excludes any possibly confounded cases and just asks if
the pellet ended up untouched at the end of the segment. We know that
the pellet couldn't have been eaten or displaced if the pellet is still
on the pillar at the end of the segment, and testing against ground
truth confirms that. Confound examples
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

## Stage 5: Pellet was off-pillar from segment start and stayed off

**Description:** Asks whether the pellet entered this segment already
off the pillar (sitting somewhere in the scoring area from a prior
segment's displacement, or operator-loaded off-center) and never
returned to the pillar throughout the segment. The reasoning: a
segment is GT untouched if no reach IN THAT SEGMENT caused
interaction. A pellet that came in already displaced is fair game
for a Stage-3-deferring untouched commit as long as nothing in the
current segment moves it back onto the pillar (physically impossible:
a pellet off the pillar cannot climb back onto it) and nothing apparent-retrieves it (also
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

### Known limitations

- Yield on the hand-marked untouched class is low. The remaining hand-marked
  untouched segments left after Stage 4 are likely cases where: (a) the pellet's tracked
  position is in a structural calibration grey zone (1.5-3.0 radii
  from calc'd pillar center) -- close enough to the pillar that it
  doesn't satisfy the 3.0-radii off-pillar threshold but far enough
  that Stage 1/Stage 3 didn't commit; (b) per-video DLC pellet
  tracking offsets put the pellet in the calibration grey zone
  systematically (e.g., a segment whose pellet is
  visually on pillar but tracked at 1.5 radii sustained throughout).
  These cases are deferred rather than committed, preserving 100%
  trust at the cost of yield. Future cascade work could add a
  per-video calibration check or a stage targeted at the calibration
  grey zone specifically.

## Stage 9: Pellet vanished after reach (retrieved)

**Reframe:** by the time a segment reaches Stage 9, the upstream
cascade has already excluded `untouched`. By apparatus physics the
pellet was on the pillar at segment start (the apparatus loads it before each
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
  chosen empirically) before the chain start = unlabeled causal
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

Validated against hand-marked segments: every commit is required to match the hand-marked class and
causal reach; segments no rule decides go to manual review.

(Module: `mousereach.outcomes.v6_cascade.stage_9_pellet_vanished_after_reach`)
