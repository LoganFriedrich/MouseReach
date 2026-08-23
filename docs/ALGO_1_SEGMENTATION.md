# Segmentation: cutting a video into one stretch per pellet

Describes: `src/mousereach/segmentation/` (all of it), plus the places outside it that run segmentation or read its output: `src/mousereach/pipeline/core.py`, `src/mousereach/pipeline/run_all.py`, `src/mousereach/pipeline/reprocess_to_current.py`, `src/mousereach/watcher/orchestrator.py`, `src/mousereach/watcher/review_gate.py`, `src/mousereach/review/staging.py`, `src/mousereach/review/triage_status.py`, `src/mousereach/review/fix_segmentation_widget.py`, `scripts/backfill_segmentation_candidates.py`

Verified against: 4c54e46 (2026-08-23). Working tree clean apart from an unrelated draft document.

---

## 1. What this actually does

A session video shows a mouse reaching for food pellets. A machine (the ASPA tray) advances a new pellet into place roughly every 30 seconds. Segmentation finds the frames at which the tray advanced, so that everything downstream can talk about "segment 7" instead of "frames 11213 to 13052".

It does **not** look at the video. It reads a DeepLabCut pose file (`.h5`), which gives, per frame, an x/y position and a "likelihood" (a 0-to-1 confidence from the tracking network) for each of 18 tracked points (`reach/v8/features.py:44-53`). Segmentation reads **seven** of them:

- `BOXL`, `BOXR` -- two points on the enclosure that should never move. Used as a fixed reference (`segmenter_robust.py:208-228`).
- `SABL`, `SABR`, `SATL`, `SATR` -- four corners of the tray's scoring area. These sweep sideways when the tray advances. That sweep is the signal (`segmenter_multi.py:222`).
- `Pellet` -- only its likelihood, only to find where in the recording pellets exist at all (`segmenter_multi.py:129-131`).

An eighth, `Pillar`, is named in the code but only inside a test that is switched off (`tray_motion.py:133`, section 5.5).

The output is a single file, `{video_id}_segments.json`, whose core content is a list of **exactly 21 frame numbers**, always sorted, always length 21. 21 is hard-coded as the expectation (`n_expected: int = 21`, `segmenter_multi.py:85`) and is forced by a safety net (section 5.6). All 3423 archived output files have exactly 21. **The count is never evidence that anything worked.**

### How boundaries become segments

Segmentation emits boundaries, not segments. The two direct consumers turn them into segments differently, and the difference is real:

- Reach detection (`reach/core/span_to_reaches.py:284-293`): **21 segments**. Segment *i* runs from boundary *i* up to (not including) boundary *i+1*; segment 21 runs from the last boundary to the end of the video.
- Outcome detection (`outcomes/core/batch.py:197-198`): **20 segments**. Segment *i* runs from boundary *i* to `boundary i+1 - 1`. There is nothing after boundary 21.

So the two files disagree by exactly one segment at the end. In 400 sampled archived videos, every `_reaches.json` had 21 segments and every `_pellet_outcomes.json` had 20. Section 8 explains what that costs.

**Frames before boundary 1 belong to no segment in either convention.** Neither consumer creates a "segment 0", and neither treats the lead-in as segment 1.

---

## 2. Which code actually runs

There are two segmenters in the tree.

| File | Status |
|---|---|
| `core/segmenter_multi.py` -- `segment_video_multi()` | Every production path runs this. Version string `2.2.3` (`segmenter_multi.py:53`). |
| `core/segmenter_robust.py` -- `segment_video_robust()` | Mostly a library of helper functions (signal loading, quality checks, anomaly text, and the function that writes the JSON file). Version string `2.1.3` (`segmenter_robust.py:103`). **It is still called in one place**: the napari boundary-review widget runs it live when it can find no precomputed segments file for the video it is opening (`segmentation/review_widget.py:867-870`). Its result is shown in the window; it reaches disk only if the reviewer saves. |

### Everything that runs a segmenter

Seven call sites, in three groups.

**Through `segmentation/core/batch.py:process_single`**, which calls `segment_video_multi(dlc_path)` then `save_segmentation(...)` (`core/batch.py:79-80`):

1. `mousereach-segment -i <dir>` -> `segmentation/cli.py:main_batch` -> `core/batch.py:process_batch` -> `:179`
2. The unified pipeline runner -> `pipeline/core.py:492`
3. The watcher, local-pipeline path -> `watcher/orchestrator.py:1010`
4. The watcher, dependency-aware reprocess path -> `watcher/orchestrator.py:1891`
5. A manual single-video run -> `pipeline/run_all.py:74`
6. Reprocessing to current versions -> `pipeline/reprocess_to_current.py:211`

**Bypassing `process_single` entirely:**

7. The review-bundle stager calls `segment_video_multi` and `save_segmentation` directly (`review/staging.py:299-300`). Because it skips `process_single`, it also skips the grading of section 9 and never calls `add_validation_status`. The same function instead writes a **hand-built** segments JSON when a reviewer supplies their own boundaries (`staging.py:284-296`): that file has `segmenter_version: "manual_resegmentation"`, and no `detection` block, no `intervals`, no `sa_coverage`, no `candidates` and no `needs_human`.

One more script re-runs the segmenter without using its boundaries: `scripts/backfill_segmentation_candidates.py:95` calls `segment_video_multi` purely to regenerate the candidate list for an old file, and deliberately leaves the stored boundaries alone.

### Who sets `validation_status`

Only three of those paths ever write `validation_status` into the file: `process_batch` (`core/batch.py:191-198`), `pipeline/core.py:500-508`, and human review. **The watcher does not.** `orchestrator.py:1846` imports `add_validation_status` and never calls it -- that import is the only mention of the name in the file. `run_all.py`, `reprocess_to_current.py` and `staging.py` do not set it either. In the archived corpus, 2159 of 3423 files have no `validation_status` key at all.

### Two constants with the same name and different values

`SEGMENTER_VERSION` is `"2.2.3"` in `segmenter_multi.py:53` and `"2.1.3"` in `segmenter_robust.py:103`. Which one you get depends on where you import from:

- `from mousereach.segmentation.core.batch import SEGMENTER_VERSION` -> 2.2.3 (`core/batch.py:22`)
- `from mousereach.segmentation.core import SEGMENTER_VERSION` -> **2.1.3** (`core/__init__.py:79-85` re-exports the robust one)

This ambiguity caused a long-lived bug. `save_segmentation` used to stamp `segmenter_robust`'s own constant into every file, so files produced by the 2.2.x segmenter claimed to be 2.1.3, and the pipeline's version check never marked a single video as needing reprocessing for the 2.2 changes. That is now fixed in both places: `save_segmentation` takes the version as an argument and defaults to importing `segmenter_multi`'s constants (`segmenter_robust.py:853-878`), and the pipeline-index update writes the same value (`segmenter_robust.py:991`). An earlier draft of this document said the index half was still broken. It is not.

Two consequences are still live:

- The napari review widget imports the **robust** constant as "the current version" (`review_widget.py:781-782`) and warns when the file disagrees (`:836-838`). Every correctly stamped file now trips that warning and is labelled `OUTDATED v2.2.3` against a "current version" of 2.1.3. The warning also tells the user to run `batch_segment.py`, a file that does not exist.
- The version check reads `segmenter_version` straight out of `_segments.json` (`pipeline/manifest.py:153`), and the pipeline declares `segmenter: "2.2.3"` (`MouseReach_Pipeline/pipeline_versions.json`). The 3423 files under `Analyzed/` were corrected, but 637 segments files under `Y:\...\MouseReach_Pipeline\Processing` are still stamped `2.1.3` and so read as outdated.

---

## 3. Reference quality -- the gate before anything else

`assess_reference_quality` (`segmenter_robust.py:208-228`) takes `BOXL_x` and `BOXR_x` over the whole file, blanks frames whose likelihood is below 0.5, fills them by interpolation, and takes the standard deviation of each.

| Both standard deviations | Verdict |
|---|---|
| under 5 pixels | `good` |
| under 15 pixels | `suspect` |
| otherwise | `bad` |
| either point missing from the file | `missing` |

It also returns `box_center` = the midpoint of the two medians. Every proposer measures corner positions relative to that number.

**`suspect` changes nothing.** It is written into the output file. The only branch on the verdict is `ref_quality == 'bad'` (`segmenter_multi.py:187`, `:203`), plus one line that records anything other than `good` as a reason a person should look (`segmenter_multi.py:404`) -- which nothing acts on (section 7). Across 3423 archived files, 3411 are `good`, 7 `suspect`, 5 `bad`.

A separate measure, `assess_sa_quality` (`segmenter_robust.py:231-241`), records for each of the four corners the fraction of frames whose likelihood exceeds 0.5. **It is recorded and never read.** Grep finds no consumer of `sa_coverage` anywhere in the tree, and nothing in the segmenter branches on it. In the worst current-segmenter archived case all four corners tracked on about 28% of frames, and nothing about that video's handling differed.

### The over-long-recording rescue (added in 2.2.3)

If the whole-file verdict is `bad`, the code does not immediately give up (`segmenter_multi.py:187-200`). A common cause of `bad` is that the operator left the camera running long past the session; the trailing junk frames have no reference tracking and wreck the standard deviation even though the session itself tracked fine. So:

1. Find the pellet-active window (section 4).
2. If that window covers less than 90% of the file, re-run the reference check on just that window.
3. If the window's verdict is not `bad`, adopt its `box_center` and quality, re-measure corner coverage on the window, and continue -- with an anomaly line reading `reference rescued on pellet-active window [lo,hi] of N frames`.

**What the rescue does not do:** it changes only the reference numbers. Candidate detection in step 5.1 still runs over the *whole* dataframe, junk frames included (`segmenter_multi.py:222-233` passes `df`, not the sub-window). Trimming the out-of-session candidates is left to the pellet-window gate, which only acts under specific conditions.

This rescue fired on 11 of the 2159 archived files the current segmenter produced.

### The bailout

If the reference is still `bad` (or `BOXL`/`BOXR` are absent), the segmenter gives up entirely (`segmenter_multi.py:203-217`): it returns 21 **evenly spaced** frames at `total_frames / 22` intervals, with every boundary's method set to `fallback` and every confidence set to `0.0`. These numbers have no relationship to the tray at all.

The bailout returns before the block that fills `candidates` and `needs_human` (`segmenter_multi.py:369-439`), so **a bailed-out file carries an empty `needs_human` list and no candidate timepoints** -- the two things a person correcting it would most want.

No file in the archive was produced by the current segmenter's bailout. All five archived `reference_quality: bad` files came from the older segmenter; section 9 explains why the review gate does not catch them.

---

## 4. The pellet-active window

`_pellet_active_window` (`segmenter_multi.py:121-136`) computes where in the recording pellets are present:

1. Mark each frame 1 if `Pellet_likelihood > 0.5`, else 0 (`pellet_window_lk_threshold = 0.5`).
2. Smooth with a centred 60-frame box average (`pellet_window_smooth = 60`), which at 60 fps is one second.
3. Keep frames where the smoothed value exceeds 0.3 (`pellet_window_active_frac = 0.3`), i.e. a pellet was visible in at least about 30% of the surrounding second.
4. Return `(first such frame, last such frame)`.

If there is no `Pellet_likelihood` column, or no frame passes, it returns `None` and everything that depends on it becomes a no-op.

This is used for two things only: the reference rescue above, and the pellet-window gate below. It is a "where is the session" marker, never a per-segment "this segment must contain a pellet" test.

---

## 5. How boundaries are proposed and chosen

### 5.1 Four independent proposers

`sa_proposer` (`core/proposers.py:80-178`) runs once per corner, on `SABL`, `SABR`, `SATL`, `SATR` (`segmenter_multi.py:222-233`). For one corner:

- Take the x positions. Frames whose likelihood is below 0.5 are blanked and linearly interpolated (`get_clean_signal`, `segmenter_robust.py:179-198`).
- Speed = absolute frame-to-frame change, then a 30-frame box average (`compute_velocity`, `segmenter_robust.py:201-205`; `sa_smooth_window = 30`).
- **Pass 1, centred crossings** (`proposers.py:104-121`). Keep frames where the corner sits between 5 pixels left and 10 pixels right of `box_center` (`sa_center_range = (-5, 10)`) **and** speed exceeds 0.8 (`sa_vel_threshold`). Group runs of such frames that are 25 frames or less apart into one event (`sa_min_gap = 25`). From each event pick the single frame with the best score, where score = `-(distance from 2.5 px right of centre) + 2 * speed` (`proposers.py:62-77`). One candidate per event.
- **Pass 2, endpoint rescue** (`proposers.py:124-176`). Runs only if pass 1 produced 15 or more candidates. It has two halves and **they are not symmetric**:
  - Looking *before* the first candidate happens only if that first candidate sits more than `0.7 x cadence` into the recording (`proposers.py:137`), where cadence is the median pass-1 interval between 500 and 5000 frames, or 1839 if there is none.
  - Looking *after* the last candidate has no such condition (`proposers.py:159`).

  In both halves it adds every speed peak above 1.4 (`sa_endpoint_vel_threshold`) as a candidate, provided it is more than 25 frames from an existing one. Neither half is limited to one candidate, so a noisy stretch can add several.

A fifth proposer exists, `pellet_swap_proposer` (`proposers.py:200-269`), which fires on pellet likelihood drops and position jumps. It is **off** (`pellet_enabled: bool = False`, `segmenter_multi.py:74`); the comment says it was too noisy in production. The code is dead but present. `get_all_sa_candidates` (`proposers.py:181-197`) has no callers anywhere.

### 5.2 The pellet-window gate (added in 2.2.2)

`_gate_candidates_to_window` (`segmenter_multi.py:139-156`), applied per corner at `segmenter_multi.py:238-254`.

**The problem it solves,** per the design comment at `segmenter_multi.py:95-113`: a stronger tracking model (Model 4.0) tracks the corners so stably that it produces plausible slow-motion candidates during pre-session setup and post-session shutdown, when no pellet is on the pillar at all. Those extra candidates push a proposer above 21, which knocks it out of the reliable path in 5.3 and into the fragile one, which can shift the whole segment numbering by one. The case the comment names is `CNT0312_P2`, an 8376-frame miss on the first boundary.

**What it does.** Drop candidates that fall outside `[window_start - 200, window_end + 200]` frames (`pellet_window_margin = 200`).

**What stops it from doing harm.** Two guards, both in `_gate_candidates_to_window`:

- It runs **only** on a corner that has **more than 21** candidates (`:150`). A corner already at or below 21 is untouched, so the gate can never strip a real boundary from a clean count.
- If trimming would leave **fewer than 21**, the trim is discarded entirely and the original list is kept (`:154`). That means it was not clean phantom removal.

It only excludes the dead zones at the two ends. A middle segment with no visible pellet is always kept. Each trim writes an anomaly line: `pellet_window_gate: SATL dropped 1 dead-zone candidate(s)`. Across the archive this fired 350 times in 297 files, dropping 765 candidates in total -- 674 of them from `SATL`.

### 5.3 The two selection paths

All surviving candidates from all four corners are pooled and clustered by frame proximity: sort by frame, start a new cluster whenever the next candidate is more than 30 frames from the previous one (`merge_window = 30`; `consensus.py:29-44`). Each cluster becomes one merged candidate (`consensus.py:47-83`) with:

- a **frame**, taken from the highest-confidence `SABL` or `SATL` candidate in the cluster; if the cluster contains neither, the highest-confidence candidate of any corner (`consensus.py:58-63`).
- a **consensus score** = the average of the cluster's individual confidences, multiplied by 1.5 (four distinct corners), 1.3 (three), 1.1 (two) or 0.9 (one), capped at 1.0.

Then (`segmenter_multi.py:269-298`):

**Path A -- SABL-primary** (`segmenter_multi.py:275-288`). Taken when `SABL` produced *exactly* 21 candidates. For each of the 21 SABL frames, find the nearest merged cluster; if it is less than 30 frames away, use the cluster's frame, otherwise keep the raw SABL frame. Records `SABL-primary mode: 21 SABL candidates`. 1772 of the 2159 current-segmenter archived files took it.

**Path B -- consensus** (`segmenter_multi.py:290-298`). Everything else. `select_boundaries` (`consensus.py:95-220`) runs a chain of repair steps in order:

- No merged candidates at all: lay an even grid and record `No candidates found - using evenly spaced fallback` (`consensus.py:107-110`).
- More than 23: keep only the densest plausible time window (`consensus.py:122-151`).
- Still more than 21: drop the lowest-ranked, by fewest agreeing corners then lowest consensus score (`consensus.py:154-163`).
- Between 17 and 20: project past the last boundary, or before the first, or interpolate into the widest internal gaps (`consensus.py:171-202`).
- Further off than that: abandon the candidates and lay a uniform grid from the first candidate at the median interval, snapping to a candidate only if one sits within 20% of an interval (`consensus.py:204-220`). **This last branch skips the repair step of 5.4 entirely.**

Records `Consensus mode: SABL=N != 21`. 387 archived files took path B. The SABL counts that put them there: 168 files at 22, 91 at 20, 63 at 23, and a tail out to 50; one file had 11.

Both branches assign a local variable `selected` (`segmenter_multi.py:285`, `:290`) which is never read again.

### 5.4 Phantom removal and endpoint projection

`_validate_and_correct` (`consensus.py:223-284`) is reached only from path B, and only from its middle branches (`consensus.py:167-168`, `:200-201`). A near-identical copy, `_validate_and_correct_boundaries` (`segmenter_robust.py:375`), lives on the robust path and nothing the production pipeline runs reaches it.

- **Phantom removal** (`consensus.py:232-254`). Look for a pair of adjacent gaps that are each shorter than 0.95 x the median gap, whose *sum* is within 15% of the median, and where the gaps immediately before and after are each within 15% of the median. That signature means the detector wedged a spurious boundary inside one real tray advance. Drop the boundary between them, and repeat. 63 removals in 60 archived files.
- **Endpoint projection** (`consensus.py:256-282`). While there are fewer than 21 boundaries and there is a gap at the start or end larger than 0.9 x the median, insert a boundary one median interval before the first (or after the last). Capped at 4 insertions. 42 first-boundary projections in 41 archived files; 13 last-boundary projections in 13.

The endpoint loop tests `len(frames) < 21` as a literal (`consensus.py:265`) rather than the configured `n_expected`. Today those are the same number, so it has no effect; it would silently break if `n_expected` were ever changed.

### 5.5 The tray-motion gate (added in 2.2.1)

`core/tray_motion.py`, applied at `segmenter_multi.py:335-345`. It asks, per boundary: does the pose data actually show a tray cycle here?

The module documents two tests. **Only one of them runs.**

- **Test 1, corner excursion (live)** (`tray_motion.py:101-125`). In the 50 frames each side of the boundary (`tray_motion_window = 50`), take each corner's x range (largest minus smallest). If any corner spans at least 30 pixels (`tray_motion_excursion_threshold = 30.0`), pass. The rationale in the module docstring is that a real cycle sweeps the tray a long way, while an operator nudging the apparatus only shifts it 5-15 pixels.
- **Test 2, pillar likelihood drop (dead)** (`tray_motion.py:127-150`). The idea was that a fresh pellet lands on the pillar after each cycle and hides the pillar tip, so pillar likelihood should fall sharply across a real boundary. It is **hard-disabled** by a module-level constant, `PILLAR_LK_TEST_ENABLED = False` (`tray_motion.py:56`). The comment above it (`:44-55`) says why: smoke-testing on the 47-video corpus showed it rejecting real boundaries on many videos, because plenty of segments have stable pillar likelihood across a boundary. The comment is explicit that the memory note claiming pillar likelihood always drops is not borne out by the corpus.

Consequences worth knowing:

- The configuration field `tray_motion_pillar_lk_drop_threshold` (`segmenter_multi.py:93`) is threaded through three function calls and **has no effect on anything**. `PILLAR_LK_TEST_ENABLED` is a module constant with no configuration hook, so the test cannot be turned back on without editing the file.
- The comment at `tray_motion.py:130-131` says the disabled test is "still computed and logged as a diagnostic". It is not. The whole block is inside `if PILLAR_LK_TEST_ENABLED and ...` (`tray_motion.py:133`), so nothing is computed and nothing is logged.
- The excursion test reads `df[f"{bp}_x"]` **raw** (`tray_motion.py:110`) -- it is the only position read in this subsystem that skips the likelihood filter. Positions from badly-tracked frames jump around wildly and count toward the 30-pixel span, so a boundary sitting in a poorly-tracked stretch can pass this gate on tracking noise alone.

**What happens on rejection** (`replace_invalid_boundaries`, `tray_motion.py:156-199`): the boundary is thrown away and replaced by a projection -- the nearest earlier surviving boundary plus *n* median intervals, or, if there is no earlier survivor, the nearest later one minus *n* intervals. The median is taken over the surviving boundaries only. If nothing survives anywhere, the original frame is kept. The result is clamped to the video and re-sorted (`:198-199`). **Re-sorting means a projection that lands past its neighbour silently renumbers every segment after it.** Each rejection writes `tray_motion_gate_rejected b{index}@{original}: {reason}`. 78 rejections across 65 archived files; the worst single file, `20251205_CNT0405_P1`, has 10.

### 5.6 The safety net

Before the tray-motion gate (`segmenter_multi.py:300-327`):

- More than 21 boundaries: truncate to the first 21 by frame order, not by quality, and record `safety_net: discarded N boundary(ies) beyond 21`.
- Fewer than 21: repeatedly append `last + median interval` (or `last + 1839` if there are fewer than two boundaries) until there are 21, clamped to the last frame, and record `safety_net: invented N boundary(ies) at the median cadence ... these mark no observed tray movement`.

The two `safety_net:` anomaly lines are new; no archived file contains one yet.

---

## 6. What "confidence" means, and what it does not

Two numbers in the output are called confidence.

**Per-boundary confidence** (`detection.confidences`). Set at `segmenter_multi.py:352-367`: find the merged cluster nearest the final boundary; if it is within 30 frames, the confidence is that cluster's **consensus score** and the "method" is the sorted list of corners in the cluster joined with `+` (for example `SABL+SABR+SATL+SATR`). Otherwise the method is `interpolated` and the confidence is `0.0`. If there were no merged candidates at all, the method is `fallback` and the confidence is `0.0`.

The consensus score is built from (a) how close to `box_center + 2.5 px` the corner was, (b) how fast it was moving, scaled so any speed at or above 2.5 scores full marks, and (c) how many corners agreed (`proposers.py:74-76`, `consensus.py:65-74`). **It is not a probability that the frame is correct.** Nothing in it compares the boundary to anything external.

It also has almost no range. Across the 45,339 boundaries in the current-segmenter archive, **92.3% have a per-boundary confidence of exactly 1.0**, because the 1.5x four-corner multiplier saturates the cap.

**Overall confidence** (`overall_confidence`) is the plain mean of the per-boundary confidences (`segmenter_robust.py:973`). Its distribution across those 2159 files: minimum 0.511, 1st percentile 0.942, median 0.988, maximum 1.0. It is not a usable ranking signal. It is, however, the only segmentation number the pipeline's review gate reads (section 9).

Note that the tray-motion gate runs *after* the boundary is chosen but *before* confidences are assigned, and a projected replacement frame is looked up against the merged clusters like any other -- so a boundary the gate rejected and replaced can still end up labelled with four-corner agreement and a confidence of 1.0.

---

## 7. "Found or forced" -- the `needs_human` verdict, which nothing acts on

Added 2026-08-22, then switched off 2026-08-23.

Because the safety net guarantees 21 boundaries, a forced answer and a measured one look identical downstream. The segmenter now says which it was. At `segmenter_multi.py:386-439` it builds a list of plain-English reasons, `needs_human`, from four sources plus the interval structure:

1. Boundaries invented by the safety net.
2. Boundaries discarded by the safety net.
3. Boundaries whose method is `interpolated` or `fallback`.
4. `reference_quality` other than `good`.
5. Any segment at least 1.6x the median length ("what a missed tray advance looks like"), any segment at most half the median ("what an extra cut looks like"), or a first-half/second-half median cadence ratio of 1.35 or more.

It also saves every merged candidate it considered, chosen or not, as `candidates` -- frame, which corners voted, the consensus score, and whether it became a boundary (`segmenter_multi.py:369-383`). Both fields go into the JSON at `segmenter_robust.py:931-932`.

**Nothing routes on `needs_human`.** It was wired into the review gate for a day and removed. The comment that replaced it (`watcher/review_gate.py:89-106`) records why: the rule fired on about 10% of ordinary videos with no evidence that was the right 10%, and against the three videos a human had actually judged mis-segmented it caught one -- the other two had textbook segmentation output because they were *offset* rather than malformed, which no measurement taken inside a single video can see. `TriageStatus.seg_needs_human` is still declared (`review/triage_status.py:137`) and still filled from the file (`:184`), and **nothing reads it**; `TriageStatus.clean` explicitly excludes it (`:154-156`).

Recomputing the parts derivable from an archived file -- reasons 3, 4 and 5; the invented and discarded counts are not recorded anywhere -- 240 of the 2159 current-segmenter files (11.1%) would carry a non-empty `needs_human`, which matches the 10% in the code comment. The commonest reason is a short segment (207 files), then a long segment (127).

The one thing that reads it and changes behaviour is the correction tool, `mousereach-fix-segmentation` (`review/fix_segmentation_widget.py`). It walks the deep-review queue, skips any bundle whose `needs_human` is empty (`:201`), sorts what remains so videos with the most unused candidates come first and videos with none come last (`:212-217`), shows the reasons to the reviewer (`:246-248`), and on save writes the new boundary list, copies `needs_human` into `needs_human_resolved`, and empties `needs_human` (`:451-459`). Since nothing puts videos into that queue *because of* `needs_human`, the tool only sees videos that landed there for some other reason: of 120 bundles in the deep-review queue today, 119 have a segments file and 49 of those carry a non-empty `needs_human`.

`scripts/backfill_segmentation_candidates.py` adds `candidates` and `needs_human` to already-segmented files without moving any boundary. It has one defect: it reads the boundary methods from `seg.get("boundary_methods")` (`:188`), a key that exists in no segments file -- the real path is `seg["detection"]["methods"]` -- so reason 3 can never fire in a backfilled file. It has been run on 114 files under `Y:\...\Processing`.

No file under `Analyzed/` has `needs_human` or `candidates`.

---

## 8. What depends on a segment being right

This is why boundary accuracy matters, stated concretely.

**Segment membership.** Reach detection finds reach spans across the whole video and then assigns each to a segment by which `[seg_start, seg_end)` window contains its **start** frame (`span_to_reaches.py:303-304`). Move a boundary and reaches cross from one segment to the next. A reach whose start falls outside every segment -- in practice, before boundary 1 -- is not dropped: it is attached to the nearest segment by frame distance, and that segment is flagged `reach_start_outside_all_segments` (`span_to_reaches.py:126-138`), which becomes a per-reach `review_note` (`:166-168`). So the lead-in before the first tray advance is quietly folded into segment 1.

**Everything numbered within a segment.** `reach_num` is assigned by position within the segment's own sorted span list (`span_to_reaches.py:157-172`). The kinematics extractor derives `is_first_reach`, `is_last_reach` and `n_reaches_in_segment` from that same list (`kinematics/core/feature_extractor.py:307-320`). One reach moved across a boundary changes all four values, on both sides of the boundary.

**Pellet number.** The causal-review tool sets `pellet_num = seg_num` outright, with the comment "In the segmenter's numbering, segment N == pellet N" (`review/causal_review_widget.py:1037-1044`). So a segmentation that misses one tray advance renumbers every pellet after it, and the pellet a person scored as number 7 on the bench is compared against footage of pellet 8. That is the failure the correction tool exists for (`review/fix_segmentation_widget.py:11-16`).

**The pixel-to-millimetre scale.** `compute_segment_geometry` (`reach/core/geometry.py:89-113`) measures the tray corners over the middle of each segment window and derives the pillar position and `ruler_pixels` from them. Boundaries in the wrong place put tray-in-motion frames inside that window, which shifts the corner medians and therefore the scale that every distance in that segment is expressed in.

**Reaches silently dropped at the end of every video.** The kinematics extractor pairs reach segments with outcome segments **positionally**: `zip(reaches_data['segments'], outcomes_data['segments'])` (`kinematics/core/feature_extractor.py:248`). The reaches file has 21 segments and the outcomes file has 20, so `zip` stops at 20 and the last reach segment is never processed. Across the whole archive that is 2554 reaches (0.49% of 520,441) in 444 videos: detected, written to `_reaches.json`, and absent from `_features.json` and from the database. `span_to_reaches.py:30-35` documents the positional zip and warns that dropping or misordering any segment corrupts everything after the first gap.

**Human corrections do not renumber cleanly.** The napari boundary reviewer compares old and new lists with `zip(original_boundaries, self.boundaries)` (`segmentation/review_widget.py:1194`); if the reviewer changed the number of boundaries, the extra entries are silently missing from the change log. The correction tool imposes no count at all -- it accepts any list of two or more (`fix_segmentation_widget.py:434-436`) -- so after a human fix a video can have any number of segments, and the 21/20 split above no longer holds.

---

## 9. What a failure looks like

### The one failure the pipeline recognises

`review/triage_status.py:44-56` defines `segmentation_failed` as: a missing or empty file, **or** `overall_confidence <= 0`, **or** any anomaly line containing the text "reference quality" (case-insensitive). `watcher/review_gate.py:87-88` sends such a video to deep human review.

Those conditions describe the current segmenter's total bailout (section 3), which sets every confidence to 0.0 and writes `Bad reference quality: bad`. Every partial failure -- ten rejected boundaries, consensus mode with only 11 SABL candidates, a zero-length segment, a video whose intervals say a tray advance was missed -- is invisible to this gate.

**The test does not catch the older segmenter's bailouts.** All five archived files with `reference_quality: bad` were produced by segmenter 2.1.0, and their anomaly text is `Primary method unavailable: ref_quality=bad, sabl=True` -- which does not contain the string "reference quality". Their `overall_confidence` is 0.3 or 0.5, above zero. So none of the five trips either half of the test. Three of them (`20250708_CNT0210_P4`, `CNT0211_P4`, `CNT0212_P4`) additionally carry `validation_status: auto_approved`: a segmentation that never looked at the tray, marked as needing no review.

### Uniform slicing scores well on the grading rule

`core/batch.py:85-95` grades a video on two things: are there 21 boundaries, and is the interval coefficient of variation (standard deviation divided by mean) below 0.3? Both reward evenness. Uniform slicing produces perfectly even intervals, and so does every interpolated or projected boundary.

The clearest live example is `20251205_CNT0405_P1`: ten of its 21 boundaries are `interpolated`, its mean confidence is 0.51, and its interval coefficient of variation is **0.0011** -- far below the threshold, and lower than a well-segmented video's, precisely because half its boundaries were laid on an even grid. `process_single` would call it `good`. Across the current-segmenter archive, 2081 of 2159 files come in under 0.3.

The `failed` branch at `core/batch.py:93-95` is unreachable: it fires only when the boundary count is not 21, and the safety net guarantees it is. The only way `process_single` returns `failed` is through its exception handler.

### Zero-length segments ship

**12 of the 2159 files the current segmenter produced contain two identical adjacent boundary frames**, i.e. a segment of length zero. Nothing rejects them. Ten of the twelve have an interval coefficient of variation of 0.229, comfortably under the 0.3 threshold.

Two causes are visible in the anomaly lines:

- Seven come from path A: it maps each of the 21 SABL frames onto its nearest merged cluster within 30 frames, and two SABL candidates 26 to 59 frames apart can map to the **same** cluster and collapse. Six of these seven duplicate at a frame between 1989 and 2045; the seventh duplicates at frame 31445. **None of them is otherwise clean** -- every one carries a `Very short interval ...: 0 frames` line, most also carry `Interval drift detected`, one carries a pellet-window-gate line, and the frame-31445 case also carries `Very long interval` and `Possible stuck tray`.
- Five come from a tray-motion-gate projection landing exactly on an existing boundary.

### Exceptions are swallowed

- `core/batch.py:108-115` catches everything from segmentation and returns `status: 'failed'` with the message. No file is written.
- `pipeline/core.py:518-519` catches everything and increments a counter. The exception object is bound to `e` and never used, never logged.
- `save_segmentation`'s pipeline-index update is wrapped in a bare `except Exception: pass` (`segmenter_robust.py:995-996`), as is `process_batch`'s (`core/batch.py:219-220`). In `process_batch` that second handler also hides a real bug: on a failed video the result dict has no `output_file` key, so `core/batch.py:211` raises `KeyError` and it is silently discarded.
- The two index updates disagree about what `seg_confidence` means. `save_segmentation` writes `overall_confidence` (`segmenter_robust.py:986`); `process_batch` then overwrites the same key with the interval coefficient of variation (`core/batch.py:215`).

### The severity classifier is largely dead, and miscounts what is left

`classify_anomaly_severity` (`segmenter_robust.py:658-724`) turns anomaly text into CRITICAL / WARNING / INFO. It checks text patterns first (`:672-677`), then three context conditions (`:680-685`), then more text patterns, then defaults to INFO (`:724`).

- Of the three CRITICAL text patterns, `"Primary method unavailable"` and `"fallback motion detection"` belong to the dead robust segmenter and the current one never emits them. **`"No candidates found"` is reachable** -- the live consensus path emits `No candidates found - using evenly spaced fallback` (`consensus.py:109`) -- though it has fired on zero of the 2159 current-segmenter files, and on 2 of the older ones.
- The INFO pattern `"lower velocity threshold"` likewise belongs to the dead segmenter.
- The current segmenter's bailout writes `Bad reference quality: bad`, which matches no text pattern. It does **not** fall through to INFO: the bailout sets every confidence to 0.0, so `mean_confidence < 0.50` (`segmenter_robust.py:684`) fires and the line is CRITICAL.

The three context checks (`:680-685`) are: fewer than 19 boundaries (impossible by construction), 5 or more interpolated boundaries, or mean confidence below 0.50. Because they run for *every* anomaly line, a file that trips one has *all* of its anomalies stamped CRITICAL. `20251205_CNT0405_P1` reports `critical: 11` -- eleven anomaly lines, all stamped by the one condition `Too many boundaries interpolated`, of which ten are tray-motion rejections and one is the consensus-mode line. It is the only file in the current-segmenter archive with any critical count at all. **`anomaly_summary.critical` counts anomaly lines, not distinct problems.**

Finally, `save_segmentation` decides the pipeline index's `seg_validation` value as `"needs_review" if warning > 0 else "auto_review"` (`segmenter_robust.py:992`) -- **the critical count is not consulted.** `20251205_CNT0405_P1` has 11 criticals and 0 warnings, so the index records it as `auto_review`.

### A stricter grader exists and never runs automatically

`segmentation/core/triage.py` grades far harder than `core/batch.py` does: it fails a video on any critical anomaly (`:438-443`) or on fewer than 10 primary candidates (`:445-450`). It is reachable only from `mousereach-triage` (`segmentation/cli.py:135`). No automated path calls it. Its `triage_reason` field appears on all 1264 older archived files and on none of the 2159 newer ones.

### Tests

`segmentation/core/test_tray_motion.py` is a real pytest suite over the tray-motion gate (`validate_tray_motion`, `replace_invalid_boundaries`, `apply_tray_motion_gate`). Nothing tests `segment_video_multi`, the proposers, the consensus merge, the safety net, or `save_segmentation`.

---

## 10. The output file, field by field

Written by `save_segmentation` (`segmenter_robust.py:853-996`). Everything below is at the top level of `{video_id}_segments.json` unless noted. All numbers are converted from numpy to plain JSON types first.

| Field | Written at | What it is |
|---|---|---|
| `segmenter_version` | `:927` | Version of the segmenter that ran; defaults to `segmenter_multi`'s `"2.2.3"`. The 2159 corrected files under `Analyzed/` say 2.2.3; the 1264 older ones say 2.1.0; the 637 under `Processing/` still say 2.1.3. |
| `needs_human` | `:931` | Plain-English reasons a person should check the cuts. Section 7. **No archived file has this key.** Nothing routes on it. |
| `candidates` | `:932` | Every merged candidate the segmenter considered, chosen or not. Section 7. **No archived file has this key.** |
| `segmenter_algorithm` | `:933` | Same source: `"multi_proposer_sabl_primary_v1+tray_motion_gate+pellet_window_gate"`. **All 3423 archived files still say `"sabl_centered_crossing_v2"`**, the old value, including the 2159 whose version was corrected. No file on disk records the real algorithm string. |
| `segmented_at` | `:936` | ISO timestamp of the run. **No file on disk has it yet** -- not in `Analyzed/`, not in either `Processing/` tree. Before it, "which segmenter made this" had to be guessed from the file's modification date. |
| `video_name` | `:938` | The DLC file's stem, so it includes the tracking-model suffix, e.g. `20251225_CNT0415_P4DLC_resnet101_MPSAOct27shuffle3_100000`. Not the video id. Human review overwrites it with the plain id, so the field's meaning depends on whether a human touched the file. |
| `total_frames` | `:939` | Number of rows in the pose file. |
| `fps` | `:940` | **Always 60.0. It is never measured.** It is the default argument of `segment_video_multi` (`segmenter_multi.py:160`) and no caller passes one -- not `core/batch.py:79`, not `staging.py:299`, not the backfill script. All 3423 archived files say 60.0. |
| `boundaries` | `:941` | The 21 frame numbers, sorted. This is the only field the rest of the pipeline depends on. |
| `reference_quality` | `:944` | `good` / `suspect` / `bad` / `missing`, from section 3. After a rescue this is the *rescued window's* verdict, not the whole file's. Read by nothing except the backfill script. |
| `sa_coverage` | `:945-950` | Per-corner fraction (0-1) of frames whose tracking likelihood exceeded 0.5. **No code reads it.** After a rescue, measured on the rescued window. |
| `detection.n_primary` | `:954` | Number of `SABL` candidates after gating. Anything other than 21 means path B was taken. **0 means the bailout.** Read only by `core/triage.py:404`, which no automated path runs. |
| `detection.n_fallback` | `:955` | **Misleading name.** It is the count of candidates from the other three corners (`SABR + SATL + SATR`), computed as total minus SABL (`segmenter_multi.py:445`). Nothing "fell back". Across the current-segmenter archive: minimum 24, 10th percentile 46, median 59, 90th percentile 63, maximum 126. |
| `detection.methods` | `:956` | Per boundary, either the `+`-joined names of the corners that agreed (e.g. `SABL+SABR+SATL+SATR`) or the literal `interpolated` or `fallback`. Corpus totals over 45,339 boundaries: 31439 four-corner, 6066 `SABL+SABR+SATL`, 3824 `SABL+SATL+SATR`, 3449 `SABL+SATL`, 65 `interpolated`, 0 `fallback`, and **24 where neither `SABL` nor `SATL` was in the cluster** -- meaning the frame came from `SABR` or `SATR`, contrary to `consensus.py:51-53`. |
| `detection.confidences` | `:957` | Per boundary, the consensus score. See section 6: 92.3% are exactly 1.0. |
| `intervals.mean_frames`, `.std_frames`, `.cv` | `:962-964` | Mean, standard deviation and coefficient of variation of the 20 gaps between boundaries. `cv` is the number `core/batch.py` grades on. |
| `intervals.mean_seconds` | `:965` | `mean_frames / fps`, and `fps` is always 60. |
| `anomalies` | `:969` | Free-text lines accumulated through the run. This is the honest record of what happened -- the mode taken, gate trims and rejections, phantom removals, projections, safety-net actions and interval complaints. Three of these strings contain a Unicode right-arrow (`segmenter_robust.py:638,642,646`), which violates the project's ASCII-only rule and raises `UnicodeEncodeError` if the line is printed to a Windows console. That is not hypothetical; it happened while scanning the corpus for this document. |
| `anomaly_details` | `:970` | Each anomaly with a `severity`, an `explanation`, and `boundaries_affected`. See section 9. |
| `anomaly_summary` | `:971` | Counts of `critical` / `warning` / `info`. Counts anomaly *lines*, not distinct problems. |
| `boundary_flags` | `:972` | Meant to mark which boundaries need a human look. **Empty (`{}`) in every one of the 2159 current-segmenter files.** It can only be populated from `boundaries_affected`, which is non-empty for exactly three classifications: `"lower velocity threshold"` (dead segmenter), `"Late-start"`, and `"Estimated B1"`. The current segmenter can emit `Estimated B1` (`consensus.py:209`) but has not done so once in the archive. In the 818 older files where it is non-empty, the only key ever used is `"1"`. `improvement/segmentation/analyze.py:102` uses this field as its segmentation triage count, so that metric is structurally zero for current files. |
| `overall_confidence` | `:973` | Mean of `detection.confidences`. Read by the review gate as the failure test (`<= 0`). |

### Fields added afterwards, not by the segmenter

- `validation_status` (`auto_approved` / `needs_review` / `validated`) and `validation_timestamp` are added by `add_validation_status` (`core/batch.py:30-39`), called only from `process_batch` and from `pipeline/core.py`. Section 2 lists the paths that never set them.
- Human review in the napari boundary tool (`segmentation/review_widget.py:1229-1252`) overwrites `boundaries` in place and adds `n_boundaries`, `boundary_corrections` (per boundary: was it moved, from what frame, by whom, when), `validation_status: "validated"`, and `validation_record` (an audit trail with the original boundary list and the deltas). It also rewrites `video_name` to the plain video id. **It does not recompute `intervals`, `detection`, `anomalies` or `overall_confidence`, and it does not clear `needs_human`** -- after a human moves a boundary, those fields still describe the algorithm's original answer. No file in the archive has been through this path: `n_boundaries` appears in zero of 3423.
- The correction tool (`review/fix_segmentation_widget.py:450-459`) overwrites `boundaries` and adds `algo_boundaries`, `boundary_source: "human"`, `corrected_by`, `corrected_at`, `needs_human_resolved`, and an emptied `needs_human`. It archives the original first and refuses to save if it cannot (`:439-448`). It does not recompute `intervals`, `detection`, `anomalies`, `overall_confidence` or `n_boundaries` either, and it does not re-run reach or outcome detection -- those files still describe the old cuts until the video goes back through the pipeline.
- `candidates_backfilled_at` and `candidates_backfill_note` are added by `scripts/backfill_segmentation_candidates.py:135-142`.
- `triage_reason` appears in the 1264 older files, from `core/triage.py`.
- `segmenter_version_provenance` appears in the 2159 files whose version stamps were corrected.
- **`_seg_validation.json` has no writer.** A dozen modules glob for `{video_id}_seg_validation.json` as a higher-priority alternative to `_segments.json`, and `advance_videos` (`core/advance.py:120`) does nothing unless it finds one. Nothing in the tree creates that file.

---

## 11. Configuration

**There is no configuration.** Every tunable lives in the `MultiProposerConfig` dataclass (`segmenter_multi.py:57-118`), and the class is never instantiated with non-default arguments anywhere in the tree. `segment_video_multi` accepts a `config` argument; no caller passes one. The same is true of `fps`. Changing any of these requires editing the source.

| Setting | Value | Effect |
|---|---|---|
| `sa_vel_threshold` | 0.8 | Minimum smoothed corner speed for a pass-1 candidate. |
| `sa_center_range` | (-5, 10) | How far from the reference centre a corner may sit and still count. Pixels. |
| `sa_center_target` | 2.5 | The position the scoring function prefers, in pixels right of centre. |
| `sa_min_gap` | 25 | Frames. Candidates this close or closer merge into one event. |
| `sa_smooth_window` | 30 | Frames in the speed-smoothing average. |
| `sa_endpoint_vel_threshold` | 1.4 | Speed needed for a pass-2 endpoint-rescue candidate. |
| `pellet_enabled` | `False` | Keeps the pellet proposer off. |
| `merge_window` | 30 | Frames. Cluster width, and the tolerance used when snapping SABL frames to clusters and when labelling method and confidence. |
| `expected_interval` | 1839.0 | Frames between tray advances, about 30.6 s at 60 fps. Used as a fallback whenever a measured median is unavailable or implausible. |
| `n_expected` | 21 | Number of boundaries forced. Not honoured by the endpoint-projection loop, which hard-codes 21 (`consensus.py:265`). |
| `tray_motion_gate_enabled` | `True` | Section 5.5. |
| `tray_motion_window` | 50 | Frames each side of a boundary in which corner excursion is measured. |
| `tray_motion_excursion_threshold` | 30.0 | Pixels. The only live test in the gate. |
| `tray_motion_pillar_lk_drop_threshold` | 0.3 | **No effect.** The test it belongs to is disabled by a module constant with no configuration hook. |
| `pellet_window_gate_enabled` | `True` | Section 5.2, and the reference rescue in section 3. |
| `pellet_window_lk_threshold` | 0.5 | Pellet likelihood above which a frame counts as "pellet visible". |
| `pellet_window_smooth` | 60 | Frames in the presence-smoothing average. |
| `pellet_window_active_frac` | 0.3 | Smoothed presence above which a frame is inside the active window. |
| `pellet_window_margin` | 200 | Frames of slack each side of the active window before a candidate is called a dead-zone phantom. |

Thresholds outside the dataclass: `detect_anomalies` (`segmenter_robust.py:628-655`) hard-codes `expected = 1839` for its stuck-tray and drift checks, independent of `fps` and of the configured interval; the likelihood cut of 0.5 is a default argument of `get_clean_signal` (`segmenter_robust.py:180`) and a literal in `assess_sa_quality` (`:238`); `SAME_BOUNDARY_FRAMES = 30` sits in the correction tool (`fix_segmentation_widget.py:57`).

---

## 12. Declared and does nothing

The most useful list in this document.

- `sa_coverage` in the output file -- computed, written, read by nobody.
- `needs_human` -- computed, written, and deliberately not acted on (`review_gate.py:89-106`). `TriageStatus.seg_needs_human` is filled (`triage_status.py:184`) and never read.
- `boundary_flags` -- empty in every current-segmenter file.
- `tray_motion_pillar_lk_drop_threshold` -- passed through three calls, gated behind a constant that is `False`.
- `reference_quality: "suspect"` -- written, never branched on inside the segmenter.
- `detection.n_primary` and `detection.n_fallback` -- read only by `core/triage.py`, which nothing runs automatically.
- The `failed` status branch in `process_single` (`core/batch.py:93-95`) -- unreachable, because the safety net guarantees 21 boundaries.
- Local variables `selected` (`segmenter_multi.py:285`, `:290`) and `n_unused` (`:384`) -- computed, never read.
- `get_all_sa_candidates` (`proposers.py:181-197`) -- no callers.
- `pellet_swap_proposer` (`proposers.py:200-269`) -- reachable only if `pellet_enabled` is flipped in source.
- `_validate_and_correct_boundaries` (`segmenter_robust.py:375`) -- superseded by the copy in `consensus.py`; the production pipeline never reaches it.
- `mousereach-advance` **cannot run**: `segmentation/cli.py:179` calls `advance_videos(args.input, require_validation=not args.force)`, and `advance_videos` (`core/advance.py:104-108`) takes `(source_dir, dest_dir, verbose)` with no such keyword and no `**kwargs`. The command raises `TypeError` immediately. Even if it ran it would find nothing, because it globs for `_seg_validation.json`, which nothing writes.
- The backfill script's "interpolated or fell back" reason -- it reads a key, `boundary_methods`, that no segments file has (`backfill_segmentation_candidates.py:188`).

---

## 13. Documentation in this subsystem that is wrong

Do not trust these. All were checked against the code at 4c54e46.

- `segmenter_robust.py`'s module docstring (lines 1-89) describes an algorithm with a 5-frame median filter, a velocity threshold of 0.03, a minimum interval of 300 frames and a maximum of 1200. **None of those numbers appear in any code**, and the function it describes is not what the pipeline runs. Its "OUTPUT FORMAT" section (`:61-69`) lists a `validation_status` field that `save_segmentation` does not write, and gives the reference-quality values as `"good" / "acceptable" / "poor"`; the code emits `good` / `suspect` / `bad` / `missing`.
- `core/__init__.py`'s docstring (lines 50-67) gives an output schema containing `boundary_confidence` and `diagnostics`, neither of which anything writes. (`n_boundaries` in the same block is real, but appears only after a human saves in the napari boundary tool -- and no archived file has it.) It states an auto-approval rule of `overall_confidence >= 0.85`; the real rule is 21 boundaries and interval coefficient of variation below 0.3 (`core/batch.py:85-95`). It says segments are "~1800 frames at 30fps"; the code assumes 60.
- `segmenter_multi.py`'s own module docstring (lines 1-23) still calls itself v2.2.0 and says segmenter_robust "is unchanged". Both the version constant and that file have moved since.
- `segmentation/AGENTS.md` and `segmentation/core/AGENTS.md` (both dated 2026-01-16) present `segmenter_robust.py` as the **primary algorithm**, do not mention `segmenter_multi.py` at all, list parameter values that exist nowhere in the code, and describe folder-based triage destinations the current pipeline does not use.
- `tray_motion.py:130-131` claims the disabled pillar test is still computed and logged. It is not.
- `consensus.py:51-53` says `SABR` and `SATR` "never set the frame". They do, in 24 boundaries out of 45,339, when a cluster contains neither `SABL` nor `SATL`.
- `core/batch.py`'s module docstring lists `"auto_approved" / "needs_review" / "validated"` as the statuses, without noting that most calling paths never set one.
- `segmentation/cli.py:24-25` and `:115-117` document an auto-approval rule based on a 0.85 confidence threshold and a `Failed/` folder. Neither exists in the code the CLI calls.
- `segmentation/review_widget.py:838` tells the user to re-run `batch_segment.py`. There is no such file.

---

## Note on the corpus figures

Every count above that refers to "the corpus" or "the archive" comes from reading all 3423 `*_segments.json` files under `Y:\LAB_ROOT\Behavior\MouseReach_Pipeline\Analyzed\` on 2026-08-23. 2159 carry `segmenter_version: 2.2.3` (the current segmenter); 1264 carry `2.1.0`. Figures attributed to "the current segmenter" are restricted to the 2159. Counts about queues and in-flight work come from `Y:\...\MouseReach_Pipeline\Processing\` (637 segments files, 120 deep-review bundles) on the same date.

---

## Contested claims

This document was written from the source, then checked by a second reader
whose job was to disprove it. The statements below are ones they disputed and
that were not resolved. Neither side is authoritative: spot-checking found the
checker wrong at least once. **Do not rely on anything listed here without
opening the code yourself.** Everything not listed survived two passes.

- **"A stricter grader exists and never runs automatically. segmentation/core/triage.py grades far harder than core/batch.py does: it fails a video on any critical anomaly (:438-443) or on fewer than 10 primary candidates (:445-450). It is reachable only from mousereach-triage (segmentation/cli.py:135).**
  - disputed because: Lines 438-450 sit inside classify_segments_graduated (def at triage.py:381), which has NO callers anywhere in the tree - grep for classify_segments_graduated returns only its own definition. mousereach-triage calls triage_results (cli.py:135), which at triage.py:675 calls classify_segments (def at :189), a different function whose equivalent checks are at :271-275 ('if severity_counts[critical] > 
- **"detection.n_primary ... Read only by core/triage.py:404, which no automated path runs."**
  - disputed because: Line 404 is inside the uncalled classify_segments_graduated. The read that mousereach-triage actually reaches is triage.py:243 ('n_primary = detection.get(\'n_primary\', n_boundaries)'), used at :274 and :282. 'Read only by :404' points at dead code and misses the live read.
- **"The 3423 files under Analyzed/ were corrected, but 637 segments files under Y:\...\MouseReach_Pipeline\Processing are still stamped 2.1.3."**
  - disputed because: Only 2159 of the 3423 were corrected. A full scan of Analyzed/ gives exactly ('2.2.3', has segmenter_version_provenance): 2159 and ('2.1.0', no provenance key): 1264 - the older 1264 files were never touched by the correction pass. The document's own field table says 'the 2159 corrected files under Analyzed/', so the sentence also contradicts the rest of the document. (The 637-at-2.1.3 half is cor
- **"triage_reason appears in the 1264 older files, from core/triage.py."**
  - disputed because: segmentation/core/triage.py contains no write of a triage_reason key, and never has: grep finds the string nowhere in that module, and 'git log -S"triage_reason" -- src/mousereach/segmentation' returns no commits. triage_results (:584-700) only moves files and prints; it never opens a segments JSON for writing. The only triage_reason writer in the tree is outcomes/core/triage.py:203, which patches
- citation could not be resolved: `scripts/backfill_segmentation_candidates.py:95 - cited as the segment_video_multi call. Line 95 is a why.append(...) string inside verdict_for(). The import is `
- citation could not be resolved: `scripts/backfill_segmentation_candidates.py:135-142 - cited as where candidates_backfilled_at and candidates_backfill_note are added. Those keys are set at :214`
- citation could not be resolved: `segmentation/core/triage.py:438-443, :445-450 and :404 - these lines exist and their text matches the claims, but they are inside classify_segments_graduated, w`
- citation could not be resolved: `reach/v8/features.py:44-53 (minor) - cited for the 18 tracked points. Line 44 is blank; the comment is at 45 and the BODYPARTS list runs 46-54.`

