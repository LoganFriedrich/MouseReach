# Segmentation: cutting a video into one stretch per pellet

Describes: `src/mousereach/segmentation/` (all of it), plus the three places outside it that call segmentation: `src/mousereach/pipeline/core.py`, `src/mousereach/pipeline/run_all.py`, `src/mousereach/pipeline/reprocess_to_current.py`

Verified against: 61d98b9 (2026-08-21)

---

## 1. What this actually does

A session video shows a mouse reaching for food pellets. A machine (the ASPA tray) advances a new pellet into place roughly every 30 seconds. Segmentation finds the frames at which the tray advanced, so that everything downstream can talk about "segment 7" instead of "frames 11213 to 13052".

It does **not** look at the video. It reads a DeepLabCut pose file (`.h5`), which gives, per frame, an x/y position and a "likelihood" (a 0-to-1 confidence from the tracking network) for each of ~18 tracked points. Segmentation uses six of them:

- `BOXL`, `BOXR` -- two points on the enclosure that should never move. Used as a fixed reference.
- `SABL`, `SABR`, `SATL`, `SATR` -- four corners of the tray's scoring area. These sweep sideways when the tray advances. That sweep is the signal.
- `Pellet` -- only its likelihood, only to find where in the recording pellets exist at all.

The output is a single file, `{video_id}_segments.json`, whose core content is a list of **exactly 21 frame numbers**, always sorted, always length 21. 21 is hard-coded as the expectation (`n_expected: int = 21`, `segmentation/core/segmenter_multi.py:85`) and is forced by a safety net (see section 5.6) -- the count is never evidence that anything worked.

### How boundaries become segments

Segmentation emits boundaries, not segments. The two direct consumers disagree about how to turn one into the other, and the difference is real:

- Reach detection (`reach/core/span_to_reaches.py:284-292`, matching `reach/core/reach_detector.py:1039-1054` and `reach/core/reach_detector_v8.py:291-292`): **21 segments**. Boundary *i* starts segment *i+1*; the last segment runs from the last boundary to the end of the video.
- Outcome detection (`outcomes/core/batch.py:197-198`): **20 segments**, formed from consecutive boundary pairs, ending at `next boundary - 1`. The tail of the video after boundary 21 is dropped.

So segment 1 is the stretch *before* the first detected tray advance -- normally before any pellet is presented -- and reach detection has a 21st segment that outcome detection does not.

---

## 2. Which code actually runs

There are two segmenters in the tree. Only one runs.

| File | Status |
|---|---|
| `core/segmenter_multi.py` -- `segment_video_multi()` | **This is what runs.** Version string `2.2.3`. |
| `core/segmenter_robust.py` -- `segment_video_robust()` | **Never called by anything.** It is imported only as a library of helper functions (signal loading, quality checks, anomaly text, and the function that writes the JSON file). Version string `2.1.3`. |

Confirmed by grep: every caller in the tree goes through `segmentation/core/batch.py:process_single`, which calls `segment_video_multi` (`core/batch.py:79`). `segment_video_robust` has no callers outside its own `__main__` block.

The four entry points, all of which funnel into `process_single`:

1. `mousereach-segment -i <dir>` -> `segmentation/cli.py:main_batch` -> `core/batch.py:process_batch`
2. The watcher / orchestrator -> `pipeline/core.py:492`
3. A manual single-video run -> `pipeline/run_all.py:66`
4. Reprocessing to current versions -> `pipeline/reprocess_to_current.py:141`

`process_single` calls `segment_video_multi(dlc_path)` with no other arguments and then `save_segmentation(...)` (`core/batch.py:79-80`).

### Two constants with the same name and different values

`SEGMENTER_VERSION` is `"2.2.3"` in `segmenter_multi.py:53` and `"2.1.3"` in `segmenter_robust.py:103`. Which one you get depends on where you import from:

- `from mousereach.segmentation.core.batch import SEGMENTER_VERSION` -> 2.2.3
- `from mousereach.segmentation.core import SEGMENTER_VERSION` -> **2.1.3** (`core/__init__.py:80-85` re-exports the robust one)
- `segmentation/review_widget.py:781` imports the robust one, so the napari review tool compares files stamped 2.2.3 against a "current version" of 2.1.3.

This ambiguity is what caused the bug fixed at this commit: `save_segmentation` used to stamp `segmenter_robust`'s own constant into every file, so all 2159 files produced by the 2.2.x segmenter claimed to be 2.1.3, and the pipeline's version check never marked a single video as needing reprocessing for the 2.2 changes. `save_segmentation` now takes the version as an argument and defaults to importing `segmenter_multi`'s constants (`segmenter_robust.py:855-866`). **One half of the bug is still live:** the pipeline-index update six lines further down still writes the module constant, so the index records `segmenter_version: "2.1.3"` for files the JSON stamps 2.2.3 (`segmenter_robust.py:969`).

---

## 3. Reference quality -- the gate before anything else

`assess_reference_quality` (`segmenter_robust.py:190-215`) takes `BOXL_x` and `BOXR_x` over the whole file, fills low-likelihood frames by interpolation, and takes the standard deviation of each.

| Both standard deviations | Verdict |
|---|---|
| under 5 pixels | `good` |
| under 15 pixels | `suspect` |
| otherwise | `bad` |
| either point missing from the file | `missing` |

It also returns `box_center` = the midpoint of the two medians. Every proposer measures corner positions relative to that number.

**`suspect` does nothing.** It is written into the output file and never tested anywhere in the segmenter. Only `bad` (or a missing point) changes behaviour. Across 3423 archived output files, 3411 are `good`, 7 `suspect`, 5 `bad`.

A separate measure, `assess_sa_quality` (`segmenter_robust.py:218-229`), records for each of the four corners the fraction of frames whose likelihood exceeds 0.5. **It is recorded and never used** -- nothing in the segmenter branches on it. A corner with 3% coverage still proposes boundaries.

### The over-long-recording rescue (added in 2.2.3)

If the whole-file verdict is `bad`, the code does not immediately give up (`segmenter_multi.py:187-200`). A common cause of `bad` is that the operator left the camera running long past the session; the trailing junk frames have no reference tracking and wreck the standard deviation even though the session itself tracked fine. So:

1. Find the pellet-active window (section 4).
2. If that window covers less than 90% of the file, re-run the reference check on just that window.
3. If the window's verdict is not `bad`, adopt its `box_center` and quality, re-measure corner coverage on the window, and continue -- with an anomaly line reading `reference rescued on pellet-active window [lo,hi] of N frames`.

**What the rescue does not do:** it changes only the reference numbers. Candidate detection in step 5.1 still runs over the *whole* dataframe, junk frames included (`segmenter_multi.py:222-233` passes `df`, not the sub-window). Trimming the out-of-session candidates is left to the pellet-window gate, which only acts under specific conditions.

This rescue fired on 11 of the 2159 files the current segmenter produced.

### The bailout

If the reference is still `bad` (or `BOXL`/`BOXR` are absent), the segmenter gives up entirely (`segmenter_multi.py:203-217`): it returns 21 **evenly spaced** frames at `total_frames / 22` intervals, with every boundary's method set to `fallback` and every confidence set to `0.0`. These numbers have no relationship to the tray at all. See section 7 for how badly this is handled downstream.

---

## 4. The pellet-active window

`_pellet_active_window` (`segmenter_multi.py:121-136`) computes where in the recording pellets are present:

1. Mark each frame 1 if `Pellet_likelihood > 0.5`, else 0 (`pellet_window_lk_threshold = 0.5`).
2. Smooth with a 60-frame box average (`pellet_window_smooth = 60`).
3. Keep frames where the smoothed value exceeds 0.3 (`pellet_window_active_frac = 0.3`), i.e. a pellet was visible in at least ~30% of the surrounding second.
4. Return `(first such frame, last such frame)`.

If there is no `Pellet_likelihood` column, or no frame passes, it returns `None` and everything that depends on it becomes a no-op.

This is used for two things only: the reference rescue above, and the pellet-window gate below. It is deliberately used as a "where is the session" marker, never as a per-segment "this segment must contain a pellet" test.

---

## 5. How boundaries are proposed and chosen

### 5.1 Four independent proposers

`sa_proposer` (`core/proposers.py:80-178`) runs once per corner, on `SABL`, `SABR`, `SATL`, `SATR` (`segmenter_multi.py:222-233`). For one corner:

- Take the x positions. Frames whose likelihood is below 0.5 are blanked and linearly interpolated (`get_clean_signal`, `segmenter_robust.py:157-176`).
- Speed = absolute frame-to-frame change, then a 30-frame box average (`compute_velocity`, `segmenter_robust.py:179-183`; `sa_smooth_window = 30`).
- **Pass 1 (centred crossings).** Keep frames where the corner sits between 5 pixels left and 10 pixels right of `box_center` (`sa_center_range = (-5, 10)`) **and** speed exceeds 0.8 (`sa_vel_threshold`). Group runs of such frames that are within 25 frames of one another into one event (`sa_min_gap = 25`). From each event pick the single frame with the best score, where score = `-(distance from 2.5 px right of centre) + 2 * speed` (`proposers.py:62-77`). One candidate per event.
- **Pass 2 (endpoint rescue).** Only if pass 1 produced 15 or more candidates. Looks *before* the first candidate and *after* the last for any speed peak above 1.4 (`sa_endpoint_vel_threshold`), and adds every such event as a candidate provided it is more than 25 frames from an existing one. The docstring says this catches a first or last boundary whose position drifted outside the centred window; note it is not limited to one candidate per side, so a noisy lead-in can add several.

A fifth proposer exists, `pellet_swap_proposer` (`proposers.py:200-269`), which fires on pellet likelihood drops and position jumps. It is **off** (`pellet_enabled: bool = False`, `segmenter_multi.py:74`) because it added 30-40 spurious candidates per video. The code is dead but present.

### 5.2 The pellet-window gate (added in 2.2.2)

`_gate_candidates_to_window` (`segmenter_multi.py:139-156`), applied per corner at `segmenter_multi.py:238-254`.

**The problem it solves.** A stronger tracking model (Model 4.0) tracks the corners so stably that it produces plausible slow-motion candidates during pre-session setup and post-session shutdown, when no pellet is on the pillar at all. Those extra candidates push a proposer above 21, which knocks it out of the reliable path in 5.3 and into the fragile one, which can shift the whole segment numbering by one -- the documented case is `CNT0312_P2`, an 8376-frame miss on the first boundary.

**What it does.** Drop candidates that fall outside `[window_start - 200, window_end + 200]` frames (`pellet_window_margin = 200`).

**What stops it from doing harm.** Two guards, both in `_gate_candidates_to_window`:

- It runs **only** on a corner that has **more than 21** candidates. A corner already at or below 21 is untouched, so the gate can never strip a real boundary from a clean count.
- If trimming would leave **fewer than 21**, the trim is discarded entirely and the original list is kept. That means it was not clean phantom removal.

It only excludes the dead zones at the two ends. A middle segment with no visible pellet is always kept. Each trim writes an anomaly line: `pellet_window_gate: SATL dropped 1 dead-zone candidate(s)`. This fired on 350 boundary-level trims across the corpus.

### 5.3 The two selection paths

All surviving candidates from all four corners are pooled and clustered by frame proximity: sort by frame, start a new cluster whenever the next candidate is more than 30 frames from the previous one (`merge_window = 30`; `consensus.py:29-44`). Each cluster becomes one merged candidate (`consensus.py:47-83`) with:

- a **frame**, taken from the highest-confidence `SABL` or `SATL` candidate in the cluster. `SABR` and `SATR` are described as vote-only because they run about 7 frames early against human-marked truth.
- a **consensus score** = the average of the cluster's individual confidences, multiplied by 1.5 (four distinct corners), 1.3 (three), 1.1 (two) or 0.9 (one), capped at 1.0.

Then (`segmenter_multi.py:269-298`):

**Path A -- SABL-primary.** Taken when `SABL` produced *exactly* 21 candidates. For each of the 21 SABL frames, find the nearest merged cluster; if it is within 30 frames, use the cluster's frame, otherwise keep the raw SABL frame. Records the anomaly `SABL-primary mode: 21 SABL candidates`. This is the good path -- 1772 of the corpus's current-segmenter files took it.

**Path B -- consensus.** Everything else. `select_boundaries` (`consensus.py:95-220`) runs a chain of repair heuristics: a pre-filter that keeps only the densest plausible time window when there are more than 23 merged candidates; dropping the lowest-ranked candidates when there are more than 21 (ranked by number of agreeing corners, then consensus score); projecting or interpolating when there are between 17 and 20; and, when the count is further off than that, abandoning the candidates and laying a uniform grid from the first candidate at the median interval, snapping to a candidate only if one sits within 20% of an interval. Records `Consensus mode: SABL=N != 21`. 387 files took this path.

### 5.4 Phantom removal and endpoint projection

Both `_validate_and_correct_boundaries` (`segmenter_robust.py:355-480`, unused -- it lives on the dead robust path) and its copy `_validate_and_correct` (`consensus.py:223-284`, the live one) do the same two things, and only path B reaches the live one:

- **Phantom removal.** Look for a pair of adjacent gaps that are both shorter than the median gap, whose *sum* is within 15% of the median, and where the gaps immediately before and after are each within 15% of the median. That signature means the detector wedged a spurious boundary inside one real tray advance. Drop the boundary between them, and repeat. 63 removals across the corpus.
- **Endpoint projection.** While there are fewer than 21 boundaries and there is a gap at the start or end larger than 0.9 x the median, insert a boundary one median interval before the first (or after the last). Capped at 4 insertions. 42 first-boundary projections across the corpus.

The endpoint loop tests `len(frames) < 21` as a literal (`consensus.py:265`) rather than the configured `n_expected`. Today those are the same number, so it has no effect; it would silently break if `n_expected` were ever changed.

### 5.5 The tray-motion gate (added in 2.2.1)

`core/tray_motion.py`, applied at `segmenter_multi.py:319-329`. It asks, per boundary: does the pose data actually show a tray cycle here?

The module documents two tests. **Only one of them runs.**

- **Test 1, corner excursion (live).** In the 50 frames each side of the boundary (`tray_motion_window = 50`), take each corner's x range (largest minus smallest). If any corner spans at least 30 pixels (`tray_motion_excursion_threshold = 30.0`), pass. The rationale is that a real cycle sweeps the tray a long way, while an operator nudging the apparatus only shifts it 5-15 pixels.
- **Test 2, pillar likelihood drop (dead).** The idea was that a fresh pellet lands on the pillar after each cycle and hides the pillar tip, so pillar likelihood should fall sharply across a real boundary. It is **hard-disabled** by a module-level constant, `PILLAR_LK_TEST_ENABLED = False` (`tray_motion.py:56`). The comment above it says why: smoke-testing on the 47-video corpus showed it rejecting real boundaries on many videos, because plenty of segments have stable pillar likelihood across a boundary. The comment is explicit that the memory note claiming pillar likelihood always drops is not borne out by the corpus.

Consequences worth knowing:

- The configuration field `tray_motion_pillar_lk_drop_threshold` (`segmenter_multi.py:93`) is threaded through three function calls and **has no effect on anything**. `PILLAR_LK_TEST_ENABLED` is a module constant with no configuration hook, so the test cannot be turned back on without editing the file.
- The comment at `tray_motion.py:129-131` says the disabled test is "still computed and logged as a diagnostic". It is not. The whole block is inside `if PILLAR_LK_TEST_ENABLED and ...` (`tray_motion.py:133`), so nothing is computed and nothing is logged.
- The excursion test reads `df[f"{bp}_x"]` **raw** (`tray_motion.py:110`) -- it is the only position read in this subsystem that skips the likelihood filter. Positions from badly-tracked frames jump around wildly and count toward the 30-pixel span, so a boundary sitting in a poorly-tracked stretch can pass this gate on tracking noise alone.

**What happens on rejection** (`replace_invalid_boundaries`, `tray_motion.py:156-199`): the boundary is thrown away and replaced by a projection -- the nearest earlier surviving boundary plus *n* median intervals, or, if there is no earlier survivor, the nearest later one minus *n* intervals. The median is taken over the surviving boundaries only. If nothing survives anywhere, the original frame is kept. The result is clamped to the video and re-sorted. **Re-sorting means a projection that lands past its neighbour silently renumbers every segment after it.** Each rejection writes `tray_motion_gate_rejected b{index}@{original}: {reason}`. 78 rejections across the corpus, spread over a handful of videos -- one file, `20251205_CNT0405_P1`, has 11 of them.

### 5.6 The safety net

Before the tray-motion gate (`segmenter_multi.py:301-311`): if there are more than 21 boundaries, truncate to the first 21 -- by frame order, not by quality. If fewer, repeatedly append `last + median interval` (or 1839 if there are fewer than two boundaries) until there are 21, clamped to the last frame. This is why the count is always 21 and why the count is never evidence of anything.

---

## 6. What "confidence" means, and what it does not

Two numbers in the output are called confidence.

**Per-boundary confidence** (`detection.confidences`). Set at `segmenter_multi.py:336-351`: find the merged cluster nearest the final boundary; if it is within 30 frames, the confidence is that cluster's **consensus score** and the "method" is the sorted list of corners in the cluster joined with `+` (for example `SABL+SABR+SATL+SATR`). Otherwise the method is `interpolated` and the confidence is `0.0`.

The consensus score is built from (a) how close to `box_center + 2.5 px` the corner was, (b) how fast it was moving, scaled so any speed at or above 2.5 scores full marks, and (c) how many corners agreed. **It is not a probability that the frame is correct.** Nothing in it compares the boundary to anything external.

It also has almost no range. Across the 45,339 boundaries produced by the current segmenter, **92.3% have a per-boundary confidence of exactly 1.0**, because the 1.5x four-corner multiplier saturates the cap.

**Overall confidence** (`overall_confidence`) is the plain mean of the per-boundary confidences (`segmenter_robust.py:955`). Its distribution across the 2159 current-segmenter files: minimum 0.51, 1st percentile 0.94, median 0.988, maximum 1.0. It is not a usable ranking signal. It is, however, the only segmentation number the pipeline gate reads (section 7).

Note that the tray-motion gate runs *after* the boundary is chosen but *before* confidences are assigned, and a projected replacement frame is looked up against the merged clusters like any other -- so a boundary the gate rejected and replaced can still be labelled with four-corner agreement and a confidence of 1.0.

---

## 7. What a failure looks like

### The one failure the pipeline recognises

`review/triage_status.py:44-56` defines `segmentation_failed` as: `overall_confidence <= 0`, **or** any anomaly line containing the text "reference quality". Both conditions describe exactly one thing -- the total bailout of section 3, which produces uniform slices and all-zero confidences. `watcher/review_gate.py:87-88` sends such a video to deep human review.

That is the whole safety net. Every partial failure -- 11 rejected boundaries, consensus mode with only 11 SABL candidates, a zero-length segment -- is invisible to the gate.

### The bailout is auto-approved

`core/batch.py:85-95` decides a video's status from two things: are there 21 boundaries (always yes), and is the interval coefficient of variation below 0.3? Uniform slicing produces perfectly even intervals, so the coefficient of variation is essentially zero, so the status is `good`, so `pipeline/core.py:498-500` writes `validation_status: "auto_approved"`. A segmentation that never looked at the tray is marked as needing no review. Three archived files show exactly this outcome (`20250708_CNT0210_P4`, `CNT0212_P4`, `CNT0211_P4`: `reference_quality: bad`, `n_primary: 0`, `validation_status: auto_approved`). The gate's `overall_confidence <= 0` check catches the current bailout because it sets confidences to 0.0, but `validation_status` in the file says the opposite.

### Zero-length segments ship

**12 of the 2159 files the current segmenter produced contain two identical boundary frames**, i.e. a segment of length zero. Nothing rejects them; the interval coefficient of variation barely moves, so they are graded `good`.

Two causes are visible in the anomaly lines:

- Path A (SABL-primary) maps each of the 21 SABL frames onto its nearest merged cluster within 30 frames. Two SABL candidates 26 to 59 frames apart can map to the **same** cluster and collapse to one frame. Seven of the twelve are this: a duplicate near frame 2000, with `SABL-primary mode: 21 SABL candidates` and no other complaint.
- A tray-motion-gate projection landing exactly on an existing boundary (`20250808_CNT0301_P4`, `20251007_CNT0313_P1`).

### Exceptions are swallowed

- `core/batch.py:108-115` catches everything from segmentation and returns `status: 'failed'` with the message. No file is written.
- `pipeline/core.py:516-517` catches everything and increments a counter. The exception object is bound and never used, never logged.
- `save_segmentation`'s index update is wrapped in a bare `except Exception: pass` (`segmenter_robust.py:973-974`), as is `process_batch`'s (`core/batch.py:219-220`).

### The severity classifier is mostly dead code

`classify_anomaly_severity` (`segmenter_robust.py:645-711`) is what turns anomaly text into CRITICAL / WARNING / INFO. Its three CRITICAL text patterns are `"No candidates found"`, `"Primary method unavailable"` and `"fallback motion detection"`. **The current segmenter emits none of these strings** -- they belong to the dead robust segmenter. Likewise the INFO pattern `"lower velocity threshold"`.

The current segmenter's own bailout writes `Bad reference quality: bad`, which matches no pattern and falls through to the default, INFO (`segmenter_robust.py:710`).

What still fires are the three context checks that ignore the text entirely (`segmenter_robust.py:659-665`): fewer than 19 boundaries (impossible by construction), 5 or more interpolated boundaries, or mean confidence below 0.50. Because those checks run for *every* anomaly line, a file that trips one of them has *all* of its anomalies classified CRITICAL -- `anomaly_summary.critical` counts anomaly lines, not distinct problems. One corpus file reports `critical: 11`, which is really one problem repeated.

Finally, `save_segmentation` decides the pipeline index's `seg_validation` value as `"needs_review" if warning > 0 else "auto_review"` (`segmenter_robust.py:970`) -- **the critical count is not consulted.** A file with 11 criticals and 0 warnings is recorded in the index as `auto_review`.

---

## 8. The output file, field by field

Written by `save_segmentation` (`segmenter_robust.py:840-975`). Everything below is at the top level of `{video_id}_segments.json` unless noted. All numbers are converted from numpy to plain JSON types first.

| Field | Written at | What it is |
|---|---|---|
| `segmenter_version` | `:914` | Version of the segmenter that ran. Since this commit, defaults to `segmenter_multi`'s `"2.2.3"`. Files written before this commit say `"2.1.3"` or `"2.1.0"` regardless of what actually ran. |
| `segmenter_algorithm` | `:915` | Same source: `"multi_proposer_sabl_primary_v1+tray_motion_gate+pellet_window_gate"`. **Every one of the 3423 archived files still says `"sabl_centered_crossing_v2"`** -- the old, wrong value, including the 2159 whose version was corrected. No file on disk yet records the real algorithm string. |
| `segmented_at` | `:918` | ISO timestamp of the run. **Added at this commit; no file on disk has it yet.** Before it, "which segmenter made this" had to be guessed from the file's modification date. |
| `video_name` | `:920` | The DLC file's stem, so it includes the tracking-model suffix, e.g. `20251225_CNT0415_P4DLC_resnet101_MPSAOct27shuffle3_100000`. Not the video id. (Human review overwrites it with the plain id -- see below.) |
| `total_frames` | `:921` | Number of rows in the pose file. |
| `fps` | `:922` | **Always 60.0. It is never measured.** It is the default argument of `segment_video_multi`, and the only caller (`core/batch.py:79`) does not pass one. |
| `boundaries` | `:923` | The 21 frame numbers. Sorted. This is the only field the pipeline actually depends on. |
| `reference_quality` | `:926` | `good` / `suspect` / `bad` / `missing`, from section 3. After a rescue this is the *rescued window's* verdict, not the whole file's. |
| `sa_coverage` | `:927-932` | Per-corner fraction (0-1, despite older docs calling it a percentage) of frames whose tracking likelihood exceeded 0.5. Recorded for humans; **no code reads it.** After a rescue, measured on the rescued window. |
| `detection.n_primary` | `:936` | Number of `SABL` candidates after gating. Anything other than 21 means path B was taken. **0 means the bailout.** |
| `detection.n_fallback` | `:937` | **Misleading name.** It is the count of candidates from the other three corners (`SABR + SATL + SATR`), computed as total minus SABL (`segmenter_multi.py:357`). Nothing "fell back". Typical values are 60-80. |
| `detection.methods` | `:938` | Per boundary, either the `+`-joined names of the corners that agreed (e.g. `SABL+SABR+SATL+SATR`) or the literal `interpolated`. Corpus totals: 31439 four-corner, 6066 `SABL+SABR+SATL`, 3824 `SABL+SATL+SATR`, 3449 `SABL+SATL`, 65 `interpolated`, and about 20 where no `SABL` or `SATL` was present at all -- meaning the frame came from `SABR` or `SATR` after all, contrary to `consensus.py:50-53`. |
| `detection.confidences` | `:939` | Per boundary, the consensus score. See section 6: 92.3% are exactly 1.0. |
| `intervals.mean_frames`, `.std_frames`, `.cv` | `:944-946` | Mean, standard deviation and coefficient of variation (standard deviation divided by mean) of the 20 gaps between boundaries. `cv` is the number `batch.py` grades on. |
| `intervals.mean_seconds` | `:947` | `mean_frames / 60`, since `fps` is always 60. |
| `anomalies` | `:951` | Free-text lines accumulated through the run. This is the honest record of what happened -- the mode taken, gate trims and rejections, phantom removals, projections, and interval complaints. Three of these strings contain a Unicode right-arrow (`segmenter_robust.py:625,629,633`), which violates the project's ASCII-only rule for printed output and will raise `UnicodeEncodeError` if the file is printed to a Windows console. |
| `anomaly_details` | `:952` | Each anomaly with a `severity`, an `explanation`, and `boundaries_affected`. See section 7 -- the classifier's text patterns no longer match what the segmenter emits. |
| `anomaly_summary` | `:953` | Counts of `critical` / `warning` / `info`. Counts anomaly *lines*, not distinct problems. |
| `boundary_flags` | `:954` | Meant to mark which boundaries need a human look. **Empty (`{}`) in every single one of the 2159 files the current segmenter produced.** It can only be populated from `boundaries_affected`, which is non-empty only for three anomaly texts the current segmenter almost never emits. In the 818 older files where it is non-empty, the only key ever used is `"1"`. `improvement/segmentation/analyze.py:102` uses this field as its segmentation triage count, so that metric is structurally zero. |
| `overall_confidence` | `:955` | Mean of `detection.confidences`. Read by the pipeline gate as the failure test (`<= 0`). |

### Fields added afterwards, not by the segmenter

- `validation_status` (`auto_approved` / `needs_review` / `validated`) and `validation_timestamp` are added by `add_validation_status` (`core/batch.py:30-39`), called from `process_batch` and from `pipeline/core.py`. **They are not added by `run_all.py` or `reprocess_to_current.py`, which call `process_single` directly and ignore its return value.** In the archived corpus, 2159 of 3423 files have no `validation_status` at all.
- Human review in napari (`review_widget.py:1163-1252`) overwrites `boundaries` in place and adds `n_boundaries`, `boundary_corrections` (per boundary: was it moved, from what frame, by whom, when), `validation_status: "validated"`, and `validation_record` (an audit trail with the original boundary list and the deltas). **It does not recompute `intervals`, `detection.confidences`, `anomalies` or `overall_confidence`** -- after a human moves a boundary, those fields still describe the algorithm's original answer. It also rewrites `video_name` from the DLC-suffixed stem to the plain video id, so the field's meaning depends on whether a human touched the file.
- `triage_reason` appears in older files, from `core/triage.py`.
- `segmenter_version_provenance` appears in the 2159 files whose stamps were corrected on 2026-08-21.

---

## 9. Configuration

**There is no configuration.** Every tunable lives in the `MultiProposerConfig` dataclass (`segmenter_multi.py:57-118`), and grep shows the class is never instantiated with non-default arguments anywhere in the tree. `segment_video_multi` accepts a `config` argument; its only caller does not pass one (`core/batch.py:79`). The same is true of `fps`. Changing any of these requires editing the source.

The values that matter, and what each does:

| Setting | Value | Effect |
|---|---|---|
| `sa_vel_threshold` | 0.8 | Minimum smoothed corner speed for a pass-1 candidate. |
| `sa_center_range` | (-5, 10) | How far from the reference centre a corner may sit and still count. Pixels. |
| `sa_center_target` | 2.5 | The position the scoring function prefers, in pixels right of centre. |
| `sa_min_gap` | 25 | Frames. Candidates closer than this merge into one event. |
| `sa_smooth_window` | 30 | Frames in the speed-smoothing average. |
| `sa_endpoint_vel_threshold` | 1.4 | Speed needed for a pass-2 endpoint-rescue candidate. |
| `pellet_enabled` | `False` | Keeps the pellet proposer off. |
| `merge_window` | 30 | Frames. Cluster width, and the tolerance used when snapping SABL frames to clusters and when labelling method/confidence. |
| `expected_interval` | 1839.0 | Frames between tray advances, about 30.6 s at 60 fps. Used as a fallback whenever a measured median is unavailable or implausible. |
| `n_expected` | 21 | Number of boundaries forced. |
| `tray_motion_gate_enabled` | `True` | Section 5.5. |
| `tray_motion_window` | 50 | Frames each side of a boundary in which corner excursion is measured. |
| `tray_motion_excursion_threshold` | 30.0 | Pixels. The only live test in the gate. |
| `tray_motion_pillar_lk_drop_threshold` | 0.3 | **No effect.** The test it belongs to is disabled by a module constant. |
| `pellet_window_gate_enabled` | `True` | Section 5.2, and the reference rescue in section 3. |
| `pellet_window_lk_threshold` | 0.5 | Pellet likelihood above which a frame counts as "pellet visible". |
| `pellet_window_smooth` | 60 | Frames in the presence-smoothing average. |
| `pellet_window_active_frac` | 0.3 | Smoothed presence above which a frame is inside the active window. |
| `pellet_window_margin` | 200 | Frames of slack each side of the active window before a candidate is called a dead-zone phantom. |

One threshold is not in the dataclass at all: `detect_anomalies` (`segmenter_robust.py:615-643`) hard-codes `expected = 1839` for its "stuck tray" check, independent of `fps` and of the configured interval.

---

## 10. Documentation in this subsystem that is wrong

Do not trust these. All were checked against the code at 61d98b9.

- `segmenter_robust.py`'s module docstring (lines 1-89) describes an algorithm with a 5-frame median filter, a velocity threshold of 0.03, a minimum interval of 300 frames and a maximum of 1200. **None of those numbers appear anywhere in the code**, and the function it describes is never called. Its "OUTPUT FORMAT" section lists a `validation_status` field that `save_segmentation` does not write.
- `core/__init__.py`'s docstring (lines 50-70) gives an output schema with `n_boundaries`, `boundary_confidence` and `diagnostics` -- none of which exist -- and states an auto-approval rule of `overall_confidence >= 0.85`. The real rule is 21 boundaries and interval coefficient of variation below 0.3 (`core/batch.py:85-95`). It also says segments are "~1800 frames at 30fps"; the code assumes 60.
- `segmentation/AGENTS.md` and `segmentation/core/AGENTS.md` (both dated 2026-01-16) present `segmenter_robust.py` as the primary algorithm, list parameter values that do not exist, and describe folder-based triage destinations that the current pipeline does not use.
- `tray_motion.py:129-131` claims the disabled pillar test is still computed and logged. It is not.
- `consensus.py:50-53` says `SABR` and `SATR` "never set the frame". They do, in about 20 boundaries out of 45,339, when a cluster contains neither `SABL` nor `SATL`.
- `core/batch.py`'s module docstring lists `"auto_approved" / "needs_review" / "validated"` as the statuses, without noting that two of the four calling paths never set one.

---

## Note on the corpus figures

Every count in this document that refers to "the corpus" comes from reading all 3423 `*_segments.json` files under `Y:\LAB_ROOT\Behavior\MouseReach_Pipeline\Analyzed\` on 2026-08-21. 2159 of them carry `segmenter_version: 2.2.3` (the current segmenter); 1264 carry `2.1.0` and were produced by the old robust segmenter. Figures attributed to "the current segmenter" are restricted to the 2159.
