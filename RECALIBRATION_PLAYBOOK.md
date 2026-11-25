# MouseReach Recalibration Playbook

Standard procedure for adapting MouseReach when an upstream input
characteristic changes and the existing thresholds no longer fit. The
common trigger is a DLC model update, but the same playbook applies for
any change that alters what MouseReach analyzes (camera rig change,
tray geometry tweak, etc.).

> **Read this before you touch a single threshold.** The whole point of
> the procedure is to keep changes reversible, A/B comparable, and
> validated against an independent holdout — so you never convince
> yourself a change is good when you've actually just overfit the
> calibration set.

---

## When to run this playbook

Run it when ANY of the following happen and you believe MouseReach's
behavior may need to change to accommodate:

- A new DLC model is adopted (new shuffle, new iteration, new training set).
- The camera hardware, lighting, or tray geometry changes in a way that
  shifts tracking characteristics.
- An algorithm component (segmenter, reach detector, outcome classifier)
  has a structural bug fix that is expected to change numerical outputs.
- Any non-trivial threshold or heuristic is being proposed.

Do NOT run this playbook for pure refactors that must produce
byte-identical outputs — that's a separate correctness concern tested via
snapshot diffs, not a recalibration.

---

## Overview — 5 steps

1. **Version bump and branch.** Make the next version reachable on a
   feature branch while keeping the current one as the baseline.
2. **Run the current version on the ground-truth corpus** to establish
   a baseline scoring snapshot.
3. **Identify calibration targets** from the eval deltas — which classes,
   which error categories, which videos. Decide which gaps are
   acceptable and which are worth chasing.
4. **Make principled edits, re-run, A/B compare.** Same corpus, same
   downstream steps, diff the eval reports.
5. **Generalization test.** 5-10 P-tray videos without ground truth yet.
   Run the new version first, THEN ground-truth those videos, THEN
   score. If the new version wins on fresh material, ship it. If it only
   wins on the calibration corpus, you overfit and need to stop.

---

## Step 1 — Version bump and branch

Convention for MouseReach:

- Master's own `pyproject.toml` version already
  carries `-dev`, so a `-dev` suffix does not mark a recalibration.
  Mark a recalibration by its branch name (below), and on ship bump the
  changed stage's own version constant (see "On final ship").
- Branch off `master` with a name that describes the trigger, e.g.
  `feature/new-dlc-recalibration` or `feature/tray-geometry-retune`.
- Tag the current baseline so it's easy to check out and rerun:
  `git tag v2.4.0-pre-<triggername>` before branching (optional but
  useful when the master commit is ambiguous).
- DO NOT delete the current version's code. Git history + branch tags
  are enough; no in-place `_archive/` copies needed for code (they rot).

On final ship:

- Squash/merge the feature branch to master.
- Bump version to the target release (`2.4.0` → `2.5.0`, dropping
  `-dev`).
- `pipeline_versions.json` declares one version
  per STAGE, not the package version: `segmenter`, `reach_detector`,
  `outcome_detector`, `assignment`, `kinematic_extractor`. Each is
  compared with the constant the code stamps into its output files:
  `SEGMENTER_VERSION` in `segmentation/core/segmenter_multi.py`,
  `VERSION` in `reach/v8/__init__.py`, `VERSION` in
  `outcomes/v6_cascade/__init__.py`, `VERSION` in
  `assignment/v2/assign.py`, `FeatureExtractor.VERSION` in
  `kinematics/core/feature_extractor.py`. So: bump the changed stage's
  own version constant, then set that stage's entry in
  `pipeline_versions.json`. `mousereach-version-check` reports any
  drift between that file and the installed code; `--init` writes the
  file from the installed code's versions; `--mark` marks outdated
  videos for reprocessing.

---

## Step 2 — Baseline against the ground-truth corpus

The canonical evaluation harness is `mousereach-eval --all` (see
[`src/mousereach/eval/`](src/mousereach/eval/)). It scores reach
detection, segmentation, and outcome classification against per-video
GT JSON files.

For any recalibration, establish **two numbers before making any edits**:

1. Current version on the GT corpus, with the CURRENT upstream inputs
   (e.g., current DLC model's h5 files).
2. Current version on the GT corpus, with the NEW upstream inputs
   (e.g., new DLC's h5 files) if the trigger is an upstream change.

The gap between (1) and (2) is the regression the new upstream
introduced. The gap between (2) and the post-edit score is how much
your recalibration recovered. Don't skip (1) — without it you can't
tell whether a post-edit 94% is a gain or a loss.

Store both eval reports under a timestamped or triggered directory:
a validation folder of your choosing outside the pipeline data folders, e.g.
`<validation root>\<trigger>_<date>\reports\`. Keep them.
They are the provenance for the decision.

### What to report (and what NOT to lead with)

**Lead with raw TP / FP / FN counts**, not F1.

- **TP** (true positive) — algorithm flagged a reach/outcome the GT
  also has.
- **FP** (false positive) — algorithm flagged something the GT doesn't
  (mapped to `phantom_reaches` in the eval report, or `retrieved_phantom`
  / `untouched_as_displaced` for outcome classes).
- **FN** (false negative) — GT has it, algorithm missed (mapped to
  `missed_reaches` for reach detection, or `retrieved_missed` /
  `displaced_as_untouched` for outcome classes).

From those, precision = TP / (TP + FP), recall = TP / (TP + FN).

F1 and accuracy are derived scalars and should be reported as
**secondary, for-summary-only** columns. They hide trade-offs: a
recalibration that trades FN for FP at a 1:1 rate looks like "F1
unchanged" but is a meaningful directional shift that you need to
see to decide whether to tune back. Comparison tables should look
like:

| Category | FP baseline | FP new | FN baseline | FN new | (F1) |
|---|---:|---:|---:|---:|---:|

Always with FP and FN as the first columns for each metric family.
Accuracy and F1 come last, in parentheses.

---

## Step 3 — Identify calibration targets, decide which to chase

Read the eval report's per-error-category counts and per-video error
lists. Group failures into:

- **Threshold drift** — same category of error across many videos, small
  magnitudes, plausibly caused by the upstream input shift. Easiest
  target. Example: phantom reaches spike because cleaner DLC triggers
  the reach detector on small motions that the old DLC smoothed away.
- **Structural break** — extreme magnitudes (>100× the median) on a
  few videos, or a whole class dropping to P=0%/R=0%. Not a threshold
  problem. Investigate the code path; the bug is qualitative.
- **Acceptable gap** — edge cases where the algorithm is unlikely to
  ever catch the event correctly (e.g., pellet flies out of frame as a
  single-frame streak). Write down why this is acceptable. Do not chase.

For each target you decide to chase, write a one-paragraph rationale
**before making the edit**:

- Which error category and how many failures it covers.
- What code path / parameter you think is responsible.
- Why you believe a principled change (not "tune until the eval passes")
  will fix it.
- What you expect to see in the eval report after the fix.

Save these rationales in the feature branch's commit messages. Future-you
reviewing a PR six months later will need them.

---

## Step 4 — Edit, re-run, A/B compare

**Edits are small, commented, and principled.** Each edit gets an
inline comment explaining the WHY (not just the WHAT), in the form:

```python
MIN_REACH_EXTENT = 0.08  # raised from 0.03 ruler/frame (v2.5.0-dev):
                         # new DLC produces less positional noise; the
                         # old threshold was tuned for old-DLC jitter
                         # and now triggers on real sub-reach motion.
```

Comments like this make the recalibration self-documenting.

**After each edit** (or batch of edits in one commit):

1. Re-run the downstream pipeline on the same GT corpus. For new-DLC
   recalibration: segmenter -> reaches -> outcomes -> reach assignment -> features on the same
   h5 files that were used for the baseline.
2. Re-run `mousereach-eval --all` and save the report alongside the
   baseline in the validation dir's `reports/` subfolder.
3. Diff the eval reports. Look at:
   - Overall accuracy / F1 per feature (reach, seg, outcome).
   - Per-category error counts — did the target category drop? Did
     any OTHER category regress?
   - Per-video success/failure — did any video that was passing now
     fail? (A regression disguised as a wash.)

If the target category dropped AND no others regressed significantly,
keep the edit. If another category regressed, you've traded one
problem for another — back the edit out and think harder.

---

## Step 5 — Generalization test

Calibration on 20-50 GT videos is prone to overfitting. Every
recalibration must pass a fresh-holdout test before being declared done.

**Procedure:**

1. Pick 5-10 P-tray videos that have NOT been ground-truthed yet.
   Select randomly across cohorts (don't pick all from one cohort).
2. Run the new (recalibrated) MouseReach on those videos, writing
   outputs to a separate subtree (e.g.,
   `<validation root>\<trigger>_<date>\generalization\`).
3. Ground-truth those 10 videos through the usual Review / GT tool
   process. This takes substantial human time per video.
4. Score the new MouseReach output against the fresh GT. Also run the
   old MouseReach on the same 10 videos and score its output — so the
   comparison on fresh material is apples-to-apples.
5. If the new version's accuracy/F1 on the fresh 10 is ≥ the old
   version's on the same 10, ship it. If it's worse, you overfit the
   calibration corpus and need to either (a) back out the edits that
   matter least to the calibration wins, or (b) add more GT videos to
   the calibration corpus and re-run from Step 3.

**Bias trap:** do NOT look at the fresh 10 videos' algorithm output
before ground-truthing them. The GT annotation has to be independent
of what the algorithm said or you corrupt the test. Scripts/processes
should enforce this if possible (GT tool hides algo output until after
annotation is saved).

---

## Deliverables at the end

A recalibration is complete when all of these exist:

- Feature branch merged to master with version bumped.
- `pipeline_versions.json` on the Pipeline side updated so the
  reprocessing scanner knows which videos need re-running.
- `reports/` directory under the validation run holds:
  - `baseline_current_inputs.md` (step 2.1)
  - `baseline_new_inputs.md` (step 2.2)
  - `post_edit_new_inputs.md` (step 4)
  - `generalization_old.md`, `generalization_new.md` (step 5)
- A top-level `SUMMARY.md` in the validation run dir that reads like a
  paper abstract: what triggered the recalibration, what edits were
  made, what the numbers said, whether generalization passed.

---

## Anti-patterns (don't do these)

- **Tuning on the eval report until it looks good.** That's gradient
  descent by hand on the calibration corpus. Guaranteed to overfit.
- **Skipping the fresh-holdout test** because you "know" the edits are
  right. Nobody knows. The test is cheap. Do it.
- **Editing without a written rationale.** Three months later you will
  not remember why that threshold is 0.08 vs. 0.03.
- **Chasing an acceptable gap.** Document the gap, move on. Every
  recalibration pass attempting to close a known-hard edge case at the
  expense of clearer signals is wasted effort.
- **Making multiple unrelated edits in one commit.** Bisectability is
  the only way to figure out which edit caused the regression you
  notice three passes later.
- **Re-using a calibration corpus as the generalization set.** If it's
  been seen, it's not fresh. Pull fresh videos.

---

## Example: a new-DLC recalibration

A validation run for a DLC model swap would live at
`<validation root>\<trigger>_<date>\` (a folder of your choosing outside the
pipeline data folders), holding the
reports, the generalization queue, and SUMMARY.md once it exists.

Typical triggering circumstance: a new DLC training run produces cleaner
tracking, but MouseReach thresholds tuned for the old DLC's noise
profile now trigger on small motions that were previously smoothed
away, so outcome accuracy on the GT corpus drops with the new DLC and
the old thresholds. Such a recalibration targets phantom reaches,
retrieval-miss rate, interaction timing, and reach-extent truncation.
The `displaced_outside` class is an example of an acceptable gap
(pellets flying out as single-frame streaks; essentially undetectable).

---

## Known follow-up work (tracked so it doesn't get lost)

### Nose-referenced reach metrics

The pre-v8 extent (`max_hand_x - boxr_x`) is no
longer computed: the current reach detector (v8, 8.1.0) leaves
`max_extent_pixels` and `max_extent_ruler` null, and `max_extent_mm` is
null on every current reach. The text below describes that old definition.

The old extent definition
(`extent = max_hand_x - boxr_x`) is a signed 1-D projection relative to
the apparatus (right slit edge). This creates two issues:

1. **Negative extent** is possible and common — it happens whenever the
   hand didn't cross BOXR. A large share of ground-truth-marked
   real reaches have negative extent, which means "negative extent" is
   NOT a valid "not a reach" signal even though the pre-v8 detectors'
   thresholds treated it that way (no current threshold reads extent). The filtering conflates partial-reach (hand behind slit, real
   intent) with non-reach (hand waving, no intent).
2. **Apparatus-relative anchoring** tracks the box, not the animal.
   A reach extends *from the mouse*, not from the slit. The nose is
   the anatomically correct anchor.

Proposed redesign (separate ticket, separate version bump):
- Redefine `extent = max_t distance(nose(t), hand(t))` over the reach
  window. Always non-negative. 2-D Euclidean.
- Done, and settled the other way: reach paths are
  anchored at `start_frame - 1` to the midpoint of the nose and BOXL and
  at `end_frame + 1` to the midpoint of the nose and BOXR
  (`kinematics/core/anchored_path.py`, `ANCHOR_FRACTION_FROM_NOSE = 0.5`),
  and every paw-position measure (path, length, straightness ...) uses
  them. Still open: the nose-referenced extent in the first bullet, to
  replace the null `max_extent_*` fields.

This is a feature-definition change that deserves its own branch
(`feature/nose-referenced-reach-metrics`) and its own recalibration
pass. Do NOT bundle it into a threshold-tuning recalibration — the
confounding makes it impossible to attribute which change caused which
delta in the eval numbers.
