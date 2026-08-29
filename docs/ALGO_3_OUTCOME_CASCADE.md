# Outcome detection: what happened to each pellet

Describes: `src/mousereach/outcomes/v6_cascade/` (all files), `src/mousereach/outcomes/core/` (`batch.py`, `pellet_outcome.py`, `triage.py`, `advance.py`, `geometry.py`, `__init__.py`), `src/mousereach/outcomes/cli.py`, `src/mousereach/outcomes/_review.py`, `src/mousereach/lib/pillar_geometry.py`

Verified against: b65fcf0 (2026-08-23)

---

## What this step does

The video has already been cut into **segments** (one per pellet presentation) and **reaches** (windows where the mouse pushed a paw out) have already been found. This step answers one question per segment: *what happened to that pellet?*

It writes one file per video, `{video}_pellet_outcomes.json`, with one record per segment.

It reads the video's pixels in only two narrow places (described below). Everything else is computed from the DeepLabCut pose file — the per-frame x, y and confidence value for each tracked body part.

---

## Which detector runs

There are two detectors in this directory.

**v6 cascade** (`v6_cascade/`, version string `6.1.0` at `v6_cascade/__init__.py:16`). This is what the batch pipeline and the watcher run:

- `outcomes/core/batch.py:154` `process_single()` runs v6 unless the caller passes `legacy=True`. Every caller of it uses the default: `watcher/orchestrator.py:1068` and `:1981`, `pipeline/run_all.py:91`, `pipeline/reprocess_to_current.py:213`.
- `mousereach-detect-outcomes` runs v6 through the same `process_batch` -> `process_single` path (`outcomes/cli.py` `main_batch`). Since 2026-08 it has no detection or input-reading code of its own.

**Legacy detector** (`core/pellet_outcome.py`; its `VERSION` constant is `"4.0.0_step2"` at line 110). Its output has a completely different shape (see "What is not in the file").

### `--legacy` reaches the legacy detector (since 2026-08; it did not before)

`mousereach-detect-outcomes --legacy` passes `legacy=True` by name through `process_batch` (`core/batch.py`) into `process_single`, whose `legacy` branch is the only place that constructs `PelletOutcomeDetector`. Before 2026-08 the flag printed a legacy banner and ran v6 anyway: `process_batch` called `process_single(dlc, seg, reach, output_dir)` with four positional arguments and `legacy` is the fifth parameter, so it stayed `False`. Any file from that era stamped `6.1.0` was produced by v6 whatever flag was typed.

### The path that does run the legacy detector

napari's **"Step 2 - Run Pipeline"** widget (`napari.yaml:22-25` → `pipeline/batch_widget.py:455` → `pipeline/core.py:412` `UnifiedPipelineProcessor.run`) reaches `pipeline/core.py:566-569`, which constructs `PelletOutcomeDetector` directly and calls `save_results`. This bypasses `process_single` entirely. A video processed through that widget gets a legacy-shaped outcomes file stamped `4.0.0_step2`, not `6.1.0`. It also calls `detect(dlc_path, seg_path)` with no reach file at all (`pipeline/core.py:567`).

Nothing announces which detector produced a given file except the `detector_version` string inside it. Anything reading outcomes files has to cope with at least three shapes: v6 (`6.1.0`), archived legacy (`2.4.4`), and legacy-as-it-stands-today (`4.0.0_step2`).

The rest of this document describes v6 unless it says otherwise.

**Outcome detection is skipped entirely for tray types E and F.** `pipeline/run_all.py:87-95`, `watcher/orchestrator.py:1062` and `:1971` all check the tray letter parsed from the filename and skip the step, because those trays have no reliable pellet. No outcomes file is produced at all for those videos.

---

## Inputs, and what happens when one is missing

`detect_outcomes_v6_cascade(dlc_df, segments, reaches, video_id, video_dir)` (`v6_cascade/detector.py:183`).

| Input | Where it comes from | If absent |
|---|---|---|
| Pose data (`dlc_df`) | the DeepLabCut `.h5`, loaded by `mousereach.reach.v8.features.load_dlc_h5` | hard failure; the whole video fails |
| `segments` | `{video}_segments.json`. Boundary frames become inclusive pairs `(b[j], b[j+1]-1)` at `core/batch.py:197-198`. Segment numbers are 1-based array positions (`detector.py:217`), **not** read from the segments file | hard failure |
| `reaches` | `{video}_reaches.json` | silently becomes an empty list (`core/batch.py:199-202`). The cascade still runs — see below for exactly what changes |
| `video_dir` | searched for `{video}.avi/.mp4/.mkv` next to the output directory (`core/batch.py:145-151`, called at `:203-204`) | the two pixel-reading checks become no-ops. Nothing is printed |

A reach belongs to a segment if its **start** frame falls inside the segment (`detector.py:166-180`). A reach that runs past the segment end is still assigned to the segment it started in.

### What "no reaches" actually does

It is not true that everything then triages. Stages differ:

- Stages 1, 3, 5, 6b, 8 and 98 never read the reach list at all. They behave identically.
- Stage 4's search window normally starts after the last reach ends; with no reaches it starts at frame 0 of the segment (`stage_4_pellet_returns_to_pillar.py:93-96`), so it becomes *more* likely to commit `untouched`.
- Stages 2 and 6 use reaches only to exclude during-reach frames from their averages; with none, every frame counts, making them slightly stricter.
- Stage 7 — the single largest committer in the archive — defers immediately with `no_reaches_in_segment` (`stage_7_pellet_settled_off_pillar_late.py:437-442`). Stages 9-29 similarly lose their evidence.

So the effect is a systematic shift toward `untouched` and toward triage, not uniform triage.

### The reach-loading bug that used to exist in the CLI (fixed 2026-08)

`core/batch.py:107-142` `_extract_reaches()` reads the current reach-file format, where reaches are nested per segment under `segments[].reaches`, and falls back to an older flat top-level `reaches` list. It is now the only reader: `mousereach-detect-outcomes` calls `process_batch`, which calls `process_single`, which calls it.

**Until 2026-08 `outcomes/cli.py` carried a second, private copy of `_extract_reaches` that read only the flat top-level `reaches` key.** Current reach files have no top-level `reaches` key at all — a scan of 200 `_reaches.json` files under `MouseReach_Pipeline/Analyzed` found the key absent from all 200, and per-segment nested reaches present in 172. So that copy returned `[]` on every real reach file, and `mousereach-detect-outcomes` produced materially different (and worse) outcomes than the pipeline did on the same inputs: on one archived video re-run 2026-08-29, 14 of 20 segments differed and 11 of them fell through to `triaged`. It also used `r.get("start_frame") or r.get("start")`, so a reach starting at frame 0 would have been dropped even if the format had matched. The copy is gone; after the fix the same re-run reproduced every per-segment field of the archived file exactly.

---

## What a wrong segment boundary does to this step

Segment identity is not something this step checks — it is something this step inherits and silently trusts.

**Numbering is positional.** `segment_num` is the 1-based index into the list built from the boundary array (`detector.py:216-217`); it is never read from the segments file. If the segmenter emits one boundary too few or too many, every segment number after that point shifts, and the outcome recorded for "segment 7" describes a different pellet presentation. Anything keyed to pellet number downstream inherits that shift with no way to detect it.

**The analysis window is the boundary.** Every stage slices the pose data positionally as `dlc_df.iloc[seg_start : seg_end - 5 + 1]`. Move a boundary and you move the whole window.

**Reach membership follows the boundary.** A reach is assigned by its start frame (`detector.py:166-180`). A boundary that lands mid-reach hands that reach to the wrong segment, which changes which reaches each stage sees, which changes the answer.

**Every late-zone test is measured backwards from `seg_end`.** Stage 1 looks at the last 30 clean frames; stages 9, 11 and 21 at the last 50%; stages 16 and 17 at the last 30%; stage 8 at the last 25%. A boundary placed too late drags the *next* pellet's arrival into the current segment's late zone. A boundary placed too early truncates the evidence that a stage needs.

**Segment start matters too.** Stage 5 decides whether the pellet arrived already lying in the tray by looking at the first 30 frames after the tray settles (`stage_5_pellet_off_pillar_throughout.py:397-412`). An early boundary means it is looking at a still-moving tray.

**`outcome_known_frame` for every `untouched` commit is exactly `seg_end - 5`.** A boundary that is N frames off moves that reported frame by N frames, one for one.

**Very short segments produce triage.** If `seg_end - 5 <= seg_start`, most stages return `continue` with reason `too_short` and the segment falls to stage 99.

**Nothing records which boundaries were used.** The outcomes file carries exactly four top-level keys — `video_id`, `detector`, `detector_version`, `segments` — and no segment start/end frames, no segmenter version, no source-file reference. If `{video}_segments.json` is later regenerated with different boundaries, the outcomes file silently no longer matches it, and no code compares them.

---

## The vocabulary the stages share

Nearly every stage is built from the same handful of measurements.

**Clean zone.** Almost every stage ignores the last 5 frames of the segment: `clean_end = seg_end - 5`. The reasoning (written out at `stage_2_pellet_stable_untouched.py:45-52`) is that the segmenter's boundary marks "the old segment is clearly ending", and the frames around it are a no-man's-land where the tray is moving.

**Pillar circle.** The pillar is the small post the pellet sits on. Its position is *not* tracked directly for this purpose — it is computed from the two front corners of the scoring-area tray, `SABL` and `SABR` (`lib/pillar_geometry.py:31-32` for the constants, `:97-105` for the arithmetic):

- `ruler` = pixel distance between SABL and SABR
- pillar centre = midpoint of SABL/SABR, moved `0.944 * ruler` upward
- pillar radius = `0.10 * ruler`

Three things follow, and all three matter:

1. **It moves with the tray.** That is deliberate — when the mouse shifts the apparatus, the pellet's position relative to the pillar still makes sense.
2. **"Upward" means straight up in the image, not perpendicular to the tray.** The docstring at `pillar_geometry.py:14` says "perpendicular", but the code only subtracts from y (`:104`). If the tray sits at an angle in frame, the computed pillar does not tilt with it.
3. **The confidence of SABL and SABR is never checked.** No function in `pillar_geometry.py` reads a likelihood column. If DeepLabCut misplaces a tray corner, the pillar circle moves with it and no stage notices.

**Radii — and what one radius actually is.** Distances are expressed as multiples of `pillar_r`. That is 0.10 ruler units. The tray ruler is 9 mm (`outcomes/core/geometry.py:17`) and the physical pillar is 4.125 mm across, i.e. 0.229 ruler units in radius (`outcomes/review_widget.py:1154-1160`). So the cascade's "radius" is a fixed fraction of tray width, roughly 0.9 mm, and is **less than half the real pillar's radius**. "Within 1.0 radii" is about 0.9 mm from the computed centre; "more than 3.0 radii" is about 2.7 mm. The v6 cascade never converts to millimetres anywhere.

**Slit line.** `pillar_cy + pillar_r` — a horizontal line just below the pillar. A body part above it (smaller y) is inside the mouse's reaching space; below it is out in the tray. Used two ways: a paw above the line counts as "paw is out reaching", and a *pellet* above the line is taken as evidence the pellet is in the mouse's mouth or paw rather than lying in the tray.

**Paw parts.** Four body parts are pooled as "the paw": `RightHand`, `RHLeft`, `RHOut`, `RHRight`. (Stage 2 is the exception — its paw-nearby test uses `RightHand` alone, `stage_2_pellet_stable_untouched.py:152-160`.)

**"Sustained" / "run".** `guards.lrun()` (`guards.py:24-33`) returns the longest unbroken stretch of consecutive frames where a condition holds. Almost every threshold in the cascade is a run length, not a total count — a real event lasts many frames, tracking noise does not.

**Cleaning.** Most stages pass the pose data through `mousereach.lib.dlc_cleaning.clean_dlc_bodyparts` first, which replaces implausible tray-corner and pellet *positions* with a rolling median of nearby confident frames. It does **not** touch likelihood columns (stated at `dlc_cleaning.py:37-38`, and the code only assigns `_x`/`_y` at `:239-250`). So it makes no difference whether a stage reads confidence from the cleaned or the raw frame — they are the same numbers. Where cleaning does matter is the pillar circle, because that is computed from corner positions with no confidence filter at all.

**Two stages do not clean.** `stage_1` and `stage_2` compute pillar geometry straight from raw positions. `pillar_geometry.py:125-129` says this is on purpose — their validation provenance is locked in. Stage 1 is the second-largest committer in the archive (14,649 of 43,180 segments), so a meaningful share of all outcomes rests on uncleaned tray corners. (Stage 1's *guard* does clean, because `guards.pellet_displaced_or_vanished` cleans internally.)

---

## The labels it can write

Only four values ever appear in the `outcome` field of a v6 file. Across `v6_cascade/`, ten `committed_class="displaced_sa"` sites, ten `committed_class="retrieved"` sites, eight `committed_class="untouched"` sites, and nothing else.

| Label | Meaning |
|---|---|
| `untouched` | No reach in this segment moved the pellet. This includes the case where the pellet arrived already lying in the tray from a previous segment and the mouse pushed it around within the tray — it was never on the pillar to be taken off it (`stage_5_pellet_off_pillar_throughout.py:10-23`) |
| `retrieved` | The pellet left the apparatus: the mouse grasped it and ate it. The evidence is always some form of "the pellet stopped being visible after a reach and never came back" |
| `displaced_sa` | The pellet was knocked off the pillar and came to rest in the scoring area (the tray) |
| `triaged` | The cascade declined to decide. This is not an outcome; it is a request for a human to look |

Three further labels exist in the documentation, the review tool's key bindings and the legacy detector, but **v6 never writes them**: `displaced_outside`, `no_pellet`, `uncertain`. `outcomes/cli.py:20-24` and `outcomes/core/__init__.py:18-23` both advertise all of them as live categories. They are not.

---

## How the cascade works

`detector.py:112-146` builds a fixed list of 33 stage objects, in a fixed order. For each segment, `detector.py:234-243` calls each stage's `decide()` in turn. Each returns one of three things:

- **commit** — "I am confident; the answer is X". The loop stops. Nothing after this stage is consulted.
- **triage** — "I recognise this as a case nobody should score automatically". The loop also stops.
- **continue** — "not my case", with a short reason. Move on.

First one to commit or triage wins. There is no voting, no confidence score, no comparison between stages. Ordering is the entire arbitration mechanism: the early stages are the safe, obvious ones, and the late ones are progressively more speculative rescues of what is left.

The stage list is rebuilt from scratch for every video (`detector.py:212`). All stages are stateless.

Every stage also fills a `features` dictionary with the numbers it computed. **None of it is ever written anywhere.** `StageDecision.features` (`stage_base.py:50`) is read by no production code path; `detector.py:247-268` builds the output record from `committed_class`, `whens` and `reason` only. Every distance, count and intermediate that would explain a decision is discarded at the end of each segment.

### The two frames a commit emits

A commit must supply `whens["outcome_known_frame"]`, and (for `retrieved` and `displaced_sa`) `whens["interaction_frame"]`.

- **`outcome_known_frame`** — the earliest frame from which the outcome is determinable. For every `untouched` stage this is exactly `seg_end - 5`, the last clean frame (verified at all eight untouched commit sites). For touched outcomes it is anchored to the event — the reach end plus a small settling offset, or the frame the pellet first appears at its resting place.
- **`interaction_frame`** — when the paw was over the pellet. Always `null` for `untouched`. For touched outcomes it is a point inside the causal reach window: the middle of the reach in most stages, 40% of the way in for stages 7 and 8 (`IFR_POSITION_IN_BOUT = 0.4`).

`stage_base.py:39-44` states this contract. It is not enforced anywhere — a stage that committed without setting `outcome_known_frame` would emit `null` and nothing would complain (`detector.py:251-252`, `.get()` with no default). In practice no stage has ever done so: across all 43,180 archived v6 segments, all 40,275 committed segments carry a non-null `outcome_known_frame`, and all 24,882 touched ones carry a non-null `interaction_frame`.

---

## Guards wrapped around every stage

After the stage list is built, `detector.py:154-163` wraps three extra checks around **every** stage's `decide()`, plus one attribute override. Each only does anything when the stage underneath commits `displaced_sa`. They run in this order (vanish, then presence, then pixels):

**1. Vanish guard** (`guards.py:123-138`, test at `guards.py:62-73`). If the pellet's confidence drops below 0.5 for 60 or more consecutive frames in the clean zone, a `displaced_sa` commit is converted to **continue**. A displaced pellet stays visible in the tray; a pellet that disappears for two seconds was retrieved.

`guards.py:106-111` defines `DISPLACED_VANISH_GUARD_CLASSES`, a set naming exactly four stage classes — Stage16, Stage17, Stage27, Stage29 — and the module docstring (`guards.py:10`) says the guard applies only to those. `detector.py:88` imports the set and **never uses it**; the wrapping at `detector.py:159` is unconditional. So the guard is on all 33 stages. The stages that get it against the design note's intent, and that actually commit `displaced_sa`, are **7, 8, 10, 21 and the two retry stages 22 and 25** (plus 14, 18, 20 and 24, which are disabled and never commit anything). Stage 7 alone accounts for 16,212 of 43,180 archived decisions, so this is not a marginal difference.

**2. Scoring-area presence guard** (`guards.py:141-158`, test at `guards.py:76-99`). If the pellet was never held at more than 3 pillar-radii from the pillar, at confidence ≥ 0.7, for 30 or more consecutive frames, a `displaced_sa` commit is converted to **continue**. "Displaced" means the pellet ended up somewhere, and somewhere is a place it stays. The guards docstring correctly says this one applies to all stages.

**3. Pixel check on the landing spot** (`cv_artifact_gate.py:171-195`, measurement at `:81-157`). This is the only guard that can *create* a triage. It opens the video, finds where the pellet supposedly landed (median position of the last 60 confident off-pillar sightings), and measures brightness in a small patch there before and after. It computes `(after − before) / pellet_brightness`, where `pellet_brightness` is the same pellet sampled on the pillar earlier in this same video. A real pellet arriving makes a dark spot bright. If the change is below **0.10** (`CV_CHANGE_GATE_T`, `cv_artifact_gate.py:45`), the tracker had latched onto something already bright at that location — dust, a corner marker, a reflection — and the commit becomes **triage** with reason `cv_artifact_landing_no_pellet_arrival`.

This guard is a no-op if `video_dir` is `None`, if the file is not found, if `cv2` will not import, if the segment is shorter than 20 clean frames, if there are fewer than three confident off-pillar sightings, or if any of the sampled brightness measurements come back unusable — every one of those returns `None` and the commit stands. It accepts `.mp4` and `.avi` (`cv_artifact_gate.py:164`).

**4. Paw-confidence override.** `detector.py:157-158` raises `paw_lk_threshold` from 0.5 to 0.9 on exactly two stages, `Stage16DisplacedViaMaxDisplacement` and `Stage17DisplacedViaDominantMaxDisplacement` (`guards.py:114-117`), because DeepLabCut 4.0 detects an approaching paw past the slit far more often than 3.x did. The override is applied only to the objects in the top-level list. Stages 22 and 25 construct **their own private copies** of stages 16 and 17 inside `__init__` (`stage_22_retry_with_stabilized_dlc.py:48-53`, `stage_25_retry_with_strict_pellet_confidence.py:46-51`), and those copies keep the 0.5 threshold. The three wrapping guards still apply to 22 and 25, because they wrap the outer object's `decide` — confirmed in the archive, where stage 22 appears 9 times with a `cv_artifact` triage reason.

---

## The stages, in order

Names in the first column are exactly the strings that appear in the `stage` field of the output file. They come from `detector.py:112-146`, and several differ from the stage class's own `name` attribute — e.g. the file `stage_2_pellet_stable_untouched.py` reports as `stage_2_stable_on_pillar`.

### Untouched stages (0–6b)

| Stage | Asks | Commits when |
|---|---|---|
| `stage_0_short_segment_triage` | — | **Never does anything.** See "Stages that can never fire" |
| `stage_1_position_never_changed` | Is the pellet still sitting on the pillar at the end? | In the last 30 clean frames (needs at least 10), ≥ 50% have the pellet at confidence ≥ 0.9 and within 1.0 radii of the pillar centre — **and** the shared touched-guard does not fire |
| `stage_2_stable_on_pillar` | Was the pellet parked inside the pillar circle all segment, and is the last 11-frame window clean? | Pellet inside the circle for ≥ 95% of non-reach frames, and in the final 11-frame window either every frame is clean or ≥ 60% are. "Clean" = pellet confidence ≥ 0.7, inside the circle, and `RightHand` not within 2 median radii at confidence ≥ 0.7 |
| `stage_3_paw_never_in_pellet_area` | Could any reach even have touched it? | The paw never got within 2.5 radii of the pillar centre while above the slit at confidence ≥ 0.5 (best 3-frame average), **and** the pellet was not sustained beyond 3 radii for 15+ frames. No vanish test |
| `stage_4_pellet_returns_to_pillar` | After the last reach, is the pellet back on the pillar? | 3 consecutive frames with the pellet within 1.2 radii, at least 15 frames after the last reach ends, with no paw out; blocked if the `Pillar` body part becomes visible in that window (confidence ≥ 0.5) or if the pellet vanished (confidence < 0.5) for 60+ frames. No sustained-in-tray test |
| `stage_5_pellet_off_pillar_throughout` | Did the pellet arrive already lying in the tray, and stay there? | Three checks: (a) in a window starting when the tray settles and ending at either +30 frames or the first paw-over-the-slit bout, 5 consecutive frames with the pellet at confidence ≥ 0.7 and more than 3 radii out; (b) it never returns within 1.2 radii for 5 consecutive frames afterwards; (c) it is seen again for 5 consecutive confident frames afterwards |
| `stage_6_predominantly_on_pillar` | Was it visible and on the pillar essentially the whole time? | Pellet visible in ≥ 99% of non-reach frames **and** inside 1.0 radii in ≥ 99% of those, and the shared touched-guard does not fire |
| `stage_6b_never_entered_sa` | Same question, tolerating more tracking noise | Confident (≥ 0.7) in ≥ 70% of frames, present in ≥ 50% of the last 30, median radius when confident < 1.8, never below 0.7 for more than 10 frames in a row, never sustained beyond 3 radii for 15+ frames |

Stages **1, 3 and 4** carry a "(4.0 recalibration)" note in their docstrings explaining what the old DeepLabCut-3.1 signal was and why it died. Stages 5 and 6b do not — `stage_5` mentions "4.0" only as the divisor in a four-corner average, and `stage_6b` never mentions it at all.

Only stages **1 and 6** use the shared safety net `guards.pellet_displaced_or_vanished` (`guards.py:36-59`), which requires that the pellet was neither sustained beyond 3 radii for 15+ frames nor absent (confidence < 0.5) for 60+ frames. Stage 3 implements the sustained-in-tray half inline and has no vanish test; stage 4 implements the vanish half inline and has no sustained-in-tray test. They are three different guards, not one shared one.

Stage 5 is the only untouched stage that can triage: if the pellet started off-pillar and then apparently vanished, that is physically impossible (nothing can be retrieved out of the tray), so the segment goes to a human with reason `pellet_appears_retrieved_from_off_pillar_state` (`stage_5_pellet_off_pillar_throughout.py:506-518`). Across 2,159 archived files this path fired zero times.

### Touched stages (7–29)

| Stage | Commits | Asks |
|---|---|---|
| `stage_7_settled_off_pillar_late` | `displaced_sa` | In the last half of the segment, is the pellet parked at one spot more than 1.0 radii off the pillar, inside the tray quadrilateral, at confidence ≥ 0.95, for ≥ 100 frames? Then it walks back from the first sustained off-pillar sighting to pick the causal reach, and requires (a) that reach's pellet displacement ≥ 1.5 radii, (b) the pellet never seen at that resting spot before it, and (c) that reach to also be the largest-displacement reach — otherwise it defers |
| `stage_8_pellet_displaced_to_sa` | `displaced_sa` | Same idea, different route, and it uses paw-over-slit bouts rather than the reach list: pellet confidently within 1.2 radii for 5 frames in the 30 before the bout, then accumulating 40 frames at rest more than 1.0 radii out inside the tray. Pellet confidence 0.95, paw confidence 0.95 |
| `stage_9_pellet_vanished_after_reach` | `retrieved` | Given the segment is touched, did the pellet disappear rather than land in the tray? Late-half visibility ≤ 10%; from the causal reach onward, at most 5 sustained frames of any pellet sighting; from the first reach onward, at most 5 sustained frames of the pellet off-pillar inside the tray. Reaches less than 20 frames apart are chained into one retrieval action. Has a second attempt (`_recheck`, `stage_9...py:176`) that re-picks the causal reach using a sustained on-pillar test when the first pass deferred with `no_candidate_reach_for_retrieval` |
| `stage_10_pillar_revealed_after_reach` | `displaced_sa` | *(filename is stale — this is not about the pillar; the docstring is honest)* One run of ≥ 5 confident frames (≥ 0.95) with the pellet more than 2.0 radii out and inside the tray, after a reach; requires ≥ 30 such frames in total afterwards and **zero** sustained sightings back within 1.5 radii |
| `stage_11_single_reach_clean_displacement` | `retrieved` | *(filename is stale — this commits retrieved; the docstring is honest)* Did tracking lose the pellet completely? ≤ 100 sustained frames of pellet sighting (confidence ≥ 0.7) in the whole clean zone, ≤ 5 in the late half, and a single reach or single paw bout to hang the interaction frame on |
| `stage_12_retrieved_pellet_above_slit` | `retrieved` | Are the post-reach pellet sightings *above* the slit line (in the mouse's face) rather than below it (in the tray)? Needs ≥ 3 sustained above-slit frames and ≤ 5 below-slit ones. Requires exactly one paw-over-slit bout |
| `stage_13_retrieved_via_pillar_lk_transition` | `retrieved` | In a segment with exactly one reach, does the `Pillar` body part go from hidden (confidence < 0.3 for ≥ 5 frames) to revealed (> 0.5 for ≥ 30 frames) across the reach, with the pellet gone afterwards? |
| `stage_14`, `stage_15` | — | **Disabled** |
| `stage_16_displaced_via_max_displacement` | `displaced_sa` | Across the segment's reaches, is there exactly one where the pellet's median position (30 frames before vs after) shifts by ≥ 1.5 radii? Requires ≥ 50 off-pillar sightings in the last 30% of the segment |
| `stage_17_displaced_via_dominant_max_displacement` | `displaced_sa` | Same, but when several reaches show displacement and one is at least 3× the next largest |
| `stage_18`, `stage_19`, `stage_20` | — | **Disabled** |
| `stage_21_causal_reach_via_on_off_transition` | `displaced_sa` **or** `retrieved` | The physics test: exactly one reach has the pellet confidently (≥ 0.9) on the pillar in a paw-free window immediately before (≥ 2 on-pillar frames among ≥ 3 paw-clear frames within 10) and none immediately after. If more than one reach shows this, the detection is noisy and it defers. Class then follows: pellet later parked in the tray (≥ 30 frames within a 15 px cluster, > 1.5 radii out) → displaced; late half essentially empty (≤ 3 sightings) → retrieved |
| `stage_22_retry_with_stabilized_dlc` | whatever the inner stage says | Re-runs stages 21, 9, 16, 17 in that order against pose data where short gaps in pellet tracking have been filled in. The code passes gaps of ≤ 5 frames with endpoints within 10 px (`:58-59`); its own docstring says 10 frames and 15 px (`:18-20`). Never touches confident detections, never smooths across a real position jump |
| `stage_23`, `stage_24` | — | **Disabled** |
| `stage_25_retry_with_strict_pellet_confidence` | whatever the inner stage says | Re-runs the same four stages with every pellet detection below confidence 0.85 zeroed out, on the theory that most false "pellet seen in the tray" evidence is the tracker firing on the wrong object. Adds its own block: if the *original* pose data shows the pellet sustained off-pillar (> 1.5 radii at confidence ≥ 0.5) for more than 10 frames, a `retrieved` commit from an inner stage is refused |
| `stage_26_retrieved_via_unique_vanish_reach` | `retrieved` | Per reach, in windows capped by the neighbouring reaches (30 before, 60 after): pellet confidently seen before, essentially absent after (< 20% of post frames). Commits if **exactly one** reach shows this, no earlier reach already moved the pellet ≥ 10 px, no run of ≥ 10 frames afterwards with the pellet confidently (≥ 0.85) more than 1.5 radii out, and no unannotated paw activity of ≥ 10 frames before it |
| `stage_27_displaced_sa_via_unique_high_displacement` | `displaced_sa` | Exactly one reach moves the pellet ≥ 10 px, no reach shows the vanish signal, that reach is the first to move it at all (≥ 5 px), and the pellet stays visible off-pillar afterwards. **Also triages**: if the chosen reach starts within 30 frames of the segment start or ends within 60 frames of the segment end, boundary noise makes the pick unreliable and it goes to a human |
| `stage_28_retrieved_via_pillar_visibility_transition` | `retrieved` | Exactly one reach where the `Pillar` body part goes from hidden (< 0.4) to clearly visible (> 0.8, a rise of > 0.5) **and** the same reach shows the pellet-vanish signal. Works in raw pixels — this stage never computes pillar geometry |
| `stage_29_displaced_sa_pillar_disambiguated_multi_disp` | `displaced_sa` | Two or more reaches move the pellet ≥ 10 px, but exactly one of them is the reach that reveals the pillar. That one is causal — the later ones were bouncing an already-displaced pellet. Also raw pixels |
| `stage_98_lost_in_shadow_triage` | triage only | See below |
| `stage_99_residual_triage` | triage only | Always triages. Reached by anything left |

The touched half has a clear shape: 7–17 are the broad rules, 21–25 are careful re-examinations of what those missed, and 26–29 are late per-reach rules built on the physical constraints that the pellet cannot return to the pillar, cannot be retrieved out of the tray, and cannot move without a reach moving it.

### What actually fires

Over the 2,159 archived files carrying `detector_version: 6.1.0` in `MouseReach_Pipeline/Analyzed` (43,180 segments), the stage field breaks down as:

```
16212  stage_7_settled_off_pillar_late          2137  stage_99_residual_triage
14649  stage_1_position_never_changed            427  stage_3_paw_never_in_pellet_area
 2361  stage_8_pellet_displaced_to_sa            247  stage_25_retry_with_strict_pellet_confidence
 1569  stage_9_pellet_vanished_after_reach       215  stage_22_retry_with_stabilized_dlc
 1137  stage_21_causal_reach_via_on_off_...      139  stage_4_pellet_returns_to_pillar
 1104  stage_26_retrieved_via_unique_vanish      100  stage_6b_never_entered_sa
  790  stage_28_retrieved_via_pillar_vis...       72  stage_13, 59 stage_5, 42 stage_12
  777  stage_27_displaced_sa_via_unique_...       37  stage_11, 26 stage_10, 14 stage_17
  684  stage_16_displaced_via_max_disp            13  stage_2, 6 stage_6, 3 stage_98
  360  stage_29_displaced_sa_pillar_disamb...
```

Labels: `displaced_sa` 20,548, `untouched` 15,393, `retrieved` 4,334, `triaged` 2,905. Two stages account for 71% of all decisions. Six stages (2, 6, 10, 11, 17, 98) each account for well under a tenth of a percent, and the eight disabled stages account for none. This is an observation of the archive, not a property of the code, but it is the fastest way to see which parts of this subsystem matter.

---

## Stages that can never fire

Eight of the 33 stages cannot produce any decision. They defer 100% of the time. All are still constructed, still called once per segment, and still cost time.

| Stage | Why |
|---|---|
| `stage_0_short_segment_triage` | `decide()` returns `continue` on its only line, reason `stage0_bypassed` (`stage_0_short_segment_triage.py:64`). Segment length is algorithm 1's problem now. Its 37-line docstring still describes the length-300 triage it no longer performs |
| `stage_14_single_reach_moderate_displacement` | `MIN_SUSTAINED_DISPLACEMENT_FRAMES = 100000` (line 53) |
| `stage_15_multi_reach_retrieved_above_slit` | `MIN_POST_ABOVE_SLIT = 10000` (line 60) |
| `stage_18_displaced_via_first_significant_displacement` | `DISPLACEMENT_RADII_MIN = 100000` (line 59) |
| `stage_19_retrieved_via_pillar_lk_first_reach` | `MIN_PRE_LOW_FRAMES = 100000` (line 62) |
| `stage_20_per_bout_classifier_displaced` | `MIN_PRE_BOUT_ON_PILLAR = 100000` (line 73) |
| `stage_23_retrieved_with_pillar_tip_noise` | `CLUSTER_STD_PX_MAX = 0` (line 61), tested as `if cluster_std > 0: defer` (line 236) — it demands that five or more separate pellet sightings sit at *exactly* the same pixel |
| `stage_24_transition_triangulation` | `MIN_TRANSITION_STRENGTH = 100.0` (line 58) |

Each of the seven threshold-disabled ones carries a dated comment explaining what was tried and why it did not reach acceptable accuracy, and says the file is kept for documentation. That is a defensible choice, but it means the docstrings at the top of those files describe behaviour that does not happen, and the archive above confirms none of them has ever committed anything.

Two helper modules are reachable only from dead code: `pellet_calibration.py` is imported only by stage 20, and `transition_detector.py` only by stage 24. A third, `trust_calibrator.py` (225 lines, computes per-stage agreement with ground truth), is imported by no code under `src/` — but it *is* imported by eight one-off calibration scripts under `scripts/`. Its docstring says "the trust score is what determines triage at runtime"; that is wrong about runtime. No runtime code reads a trust score. Triage is decided entirely by the ordering and the hard-coded thresholds.

---

## When it declines to decide

A segment comes out as `"outcome": "triaged"` with `"flagged_for_review": true` and a `flag_reason` string. Five modules can emit a triage (`grep 'decision="triage"'` in `v6_cascade/` returns `cv_artifact_gate.py:187`, `stage_27...py:249` and `:260`, `stage_5...py:508`, `stage_98...py:400`, `stage_99...py:35`). Archive counts:

1. **`stage_99_residual_triage`** — nothing committed. Reason `fell_through_all_committing_stages`. By far the largest source: **2,137** of 2,905.
2. **The pixel guard on displaced commits** — `cv_artifact_landing_no_pellet_arrival`. A stage was willing to say "displaced", but the video shows nothing arrived at the landing spot. **408**, attributed to whichever stage tried to commit (134 stage 7, 85 stage 16, 75 stage 27, 51 stage 21, 26 stage 29, 24 stage 8, 9 stage 22, 3 stage 10, 1 stage 17).
3. **`stage_27`, causal reach too near a segment edge** — `causal_reach_too_near_segment_start` (**256**) / `..._end` (**101**).
4. **`stage_98_lost_in_shadow_triage`** — the deliberate "this is unscorable and here is why" case. **3**.
5. **`stage_5`**, when a pellet that started in the tray appears to vanish. **0**.

2137 + 408 + 256 + 101 + 3 = 2,905, which matches the label total exactly.

`detector.py:256-268` also has a fallback for a segment where every stage returned `continue`, producing `stage: "residual (auto-triage)"` and reason `fell_through_all_stages`. Since stage 99 always triages, this cannot happen, and it does not appear in any archived file.

### Stage 98 in detail

Stage 98 is the only stage whose purpose is to give a triage a *specific* reason instead of the generic one. It fires when tracking lost the pellet **and** the video shows the tray is genuinely too dark to track in. Its gates, in order:

- pellet detected (≥ 0.7) in less than 10% of clean-zone frames — otherwise tracking worked, defer (`:147`)
- no in-mouth signal: fewer than 3 sustained frames of the pellet above the slit line — otherwise this is a retrieval, defer (`:248`)
- at least 3 sustained frames of the pellet off-pillar inside the tray, over the whole clean zone (`:264`)
- at least 3 sustained such frames in the **late half** of the clean zone as well, or it defers with `no_late_zone_off_pillar_in_sa_evidence` (`:281-294`)
- then it opens the video, samples 10 evenly spaced frames, takes the bounding box of the four tray corners, and measures the fraction of pixels darker than intensity 60. If no frame reaches 30% dark, defer (`:387`)

An earlier block at `:231-233` computes the same late-zone count using a 0.75 cut-point and stores it under the same feature key; `:281-283` overwrites it with the 0.5 version. The 0.75 computation has no effect.

**It only looks for `{video_id}.mp4`** (`stage_98_lost_in_shadow_triage.py:121-127`). The directory search that supplies `video_dir` accepts `.avi`, `.mp4` and `.mkv` (`core/batch.py:148`). For a video stored as `.avi`, `video_dir` is found, this stage reports `video_unavailable`, and the segment falls to stage 99 with the generic reason. The pixel guard on displaced commits does not have this problem — it checks `.mp4` and `.avi` (`cv_artifact_gate.py:164`).

Every failure inside stage 98 becomes a `continue`: `cv2` not installed (`:336-343`), video will not open (`:345-351`), tray box degenerate (`:324-329`). Nothing is logged. The result is indistinguishable from "not a shadow case".

---

## Exactly what lands in `{video}_pellet_outcomes.json`

Written by `core/batch.py:210-211` (pipeline) or `cli.py:189-191` (command line). Built at `detector.py:246-275`. Here is a real file, unedited:

```json
{
  "video_id": "20250624_CNT0103_P1",
  "detector": "v6_cascade",
  "detector_version": "6.1.0",
  "segments": [
    {
      "segment_num": 1,
      "outcome": "displaced_sa",
      "outcome_known_frame": 1684,
      "interaction_frame": 1664,
      "stage": "stage_7_settled_off_pillar_late",
      "flagged_for_review": false
    }
  ]
}
```

Top level — exactly four keys, no more. All 2,159 archived v6 files have exactly this key set.

| Key | Value |
|---|---|
| `video_id` | whatever the caller passed; the pipeline derives it from the pose filename by splitting on `"DLC"` (`core/batch.py:193`) |
| `detector` | the constant string `"v6_cascade"` |
| `detector_version` | `"6.1.0"`. Read by `pipeline/manifest.py:152-158` to decide whether a video needs reprocessing |
| `segments` | the list |

Per segment, committed case — exactly six keys (40,275 archived segments, all with this key set):

| Key | Value |
|---|---|
| `segment_num` | 1-based position in the segment list |
| `outcome` | `retrieved`, `displaced_sa` or `untouched` |
| `outcome_known_frame` | integer frame, absolute in the video |
| `interaction_frame` | integer frame, or `null` for `untouched` |
| `stage` | the build label of the deciding stage |
| `flagged_for_review` | always `false` here |

Per segment, triaged case — the same six keys plus `flag_reason`, with `outcome` set to `"triaged"` and both frame fields set to `null` (2,905 archived segments). The human-readable reason string is the **only** surviving trace of why the cascade behaved as it did.

### What is not in the file

**Computed and thrown away:**
- `StageDecision.features` — every measurement each stage made, discarded per segment.
- `StageDecision.reason` for committed segments. The reason string is built (often with the actual numbers interpolated into it) and then dropped; only triage reasons survive.

**Documented but never produced by v6:**
- `confidence` — advertised per segment in `outcomes/core/__init__.py:77`, implied by `cli.py:31-33`, and used as the triage gate in `core/triage.py:128-135`. v6 has no notion of confidence at all.
- `causal_reach_id` — advertised at `outcomes/core/__init__.py:76`. v6 identifies a causal reach internally in most touched stages and does not record which one it was. Causal reach attribution is a **separate step** (algorithm 4) that writes `{video}_reach_assignments.json` (`assignment/run.py:71-73`). It never writes back into the outcomes file. (`outcomes/AGENTS.md` does not name this field — it only says the step "requires reach detection results (`_reaches.json`) for causal reach attribution".)
- `causal_reach_frame` — exists only as a field of the legacy dataclass (`core/pellet_outcome.py:135`) and in evaluation code that reads legacy files.
- `summary` — the per-video counts block, in the documented schema at `core/__init__.py:84-92`. Not written. `core/triage.py:81` and `_review.py:84` both index it directly.
- `n_segments`, `video_name`, `total_frames`, `detected_at`, `validated`, `validated_by`, `corrections_made`, `segments_flagged` — all fields of the legacy `VideoOutcomes` dataclass (`core/pellet_outcome.py:144-161`). v6 writes none of them. Note `video_name` vs `video_id`: the key changed, so `_review.py:83`'s `data.get('video_name')` prints `None`.
- `validation_status` — documented at `cli.py:26-58` as the field that drives the whole review workflow (`auto_approved` / `needs_review` / `validated`). v6 never writes it. The legacy detector does, at detection time (`core/pellet_outcome.py:1771-1772`), so files from the napari batch-pipeline path carry it. `pipeline/manifest.py:229` reads it and, per `manifest.py:167-177`, records `"unknown"` for every v6 video.
- `pellet_visible_start` / `pellet_visible_end` / `distance_from_pillar_start` / `distance_from_pillar_end` — legacy-only.

**Present only in the legacy detector's output.** All 1,233 archived files with `detector_version: 2.4.4` carry `confidence`, `causal_reach_id`, `causal_reach_frame`, `summary`, `n_segments`, `distance_from_pillar_*`, `pellet_visible_*`, `human_verified` and `original_outcome`. They carry **no `stage` key at all** (0 of 24,660 segments), and `"outcome_known_frame": null` on every single segment (24,660 of 24,660) — the legacy detector documents that field as set by an annotator and never fills it. So a consumer reading a mixed corpus sees the *older* files carrying more fields than the newer ones, and one important field carrying less.

### What later steps add

The napari triage-clearing tool writes back into this file when a human resolves a segment (`review/triage_clearing.py:564-595`). It sets, on that segment only: `flagged_for_review: false`, `triage_cleared: true`, `human_verified: true`, `cleared_by`, `cleared_at`, `original_triage_reason`, `original_outcome` (only if the human changed the outcome), and `causal_reach_id`; it overwrites `outcome` always, and `interaction_frame` / `outcome_known_frame` only when the human supplied a non-null value. So `causal_reach_id` would appear in an outcomes file **only** for human-cleared segments.

In practice it appears nowhere: a scan of every `_pellet_outcomes.json` under `Analyzed` (3,392 files) and `Processing` (637 files) found **zero** v6 segments carrying `triage_cleared`, `human_verified`, `cleared_by`, `original_triage_reason` or `causal_reach_id`. Every v6 file in the pipeline is exactly as the detector wrote it.

`flagged_for_review` is what routes a segment into the human worklist (`review/triage_queue.py:196`). That module also independently triggers on any segment whose outcome is touched but for which algorithm 4 committed no causal reach, even though `flagged_for_review` is false (`triage_queue.py:203-213`) — so the file's own flag is not the complete picture of what needs review.

---

## Things that are broken or misleading

**Three of the four installed command-line tools do not work against v6 output.** (`pyproject.toml:100-104` installs four commands plus one deprecated alias, `mousereach-review-outcomes`.)

- `mousereach-advance-outcomes` crashes immediately. `cli.py:340` calls `advance_videos(args.input, require_validation=not args.force)`; the function signature is `advance_videos(input_dir, output_dir, verbose=True)` (`core/advance.py:38`). There is no `require_validation` parameter and `output_dir` has no default. Every invocation raises `TypeError`.
- `mousereach-review-pellet-outcomes` crashes on any v6 file. `_review.py:84` does `data['summary']` on the raw JSON; v6 files have no `summary` key. It also offers `displaced_outside`, `no_pellet` and `uncertain` as corrections, which the detector never produces.
- `mousereach-triage-outcomes` does not crash, and it is not harmless. `core/triage.py:77-84` requires `data['n_segments']` and `data['summary']`; on a v6 file the `KeyError` is caught at `:85-90` and stored as the video's `error`. `triage_results` then counts the video as `failed` and **writes into the file** `validation_status: "needs_review"` and `triage_reason: "'n_segments'"` (`core/triage.py:184-186`, `:199-205`). Every v6 video, always. The function also raises a `DeprecationWarning` when called (`:153-160`).
- `mousereach-detect-outcomes` works (since 2026-08): it runs the pipeline path (`process_batch` -> `process_single`), so it reads reaches the way the watcher does and its `--legacy` flag reaches the legacy detector. Before that it silently loaded zero reaches from current reach files (see "Inputs" above) and `--legacy` did nothing.

**Docstrings and names that describe something other than what the code does.** Each checked individually against the code at this commit:

- `v6_cascade/__init__.py:4` and `detector.py:7` say "30 stages". There are 33 (stage 6b is extra, and 0/98/99 are outside the 0–29 numbering).
- `detector.py:27` says `detector_version` is `"6.0.0"`. It is `6.1.0`.
- Three stage files open with the wrong stage number: `stage_2_...py:2` says "Stage 1", `stage_5_...py:2` says "Stage 4", `stage_8_...py:2` says "Stage 5". The cascade was renumbered and these three headers were not. Files 1, 3, 4, 6, 6b and 7 are correctly headed.
- `stage_10_pillar_revealed_after_reach.py` has nothing to do with the pillar being revealed, and `stage_11_single_reach_clean_displacement.py` commits `retrieved`. In both cases the **filename** is the stale part; the docstrings describe the real behaviour.
- `stage_5_pellet_off_pillar_throughout.py:35-42` says co-detection of pellet and pillar at the same place is triaged. The code defers instead (`:306-322`), with an inline comment explaining the change. The docstring above it was not updated.
- `stage_2`'s commit comments (`:188-195`) say the emitted frame is `seg_end - 10`; the value is computed as `seg_end - 5` at `:138` and emitted unchanged.
- `stage_11`'s docstring says "< 10 sustained frames" and "≤ 2 late"; the constants are 100 and 5 (`:59`, `:61`).
- `stage_22`'s docstring says gaps of ≤ 10 frames and endpoints within 15 px; the call passes 5 and 10.0 (`:58-59`).
- `stage_26:22` and `stage_27:15` say the displacement threshold is 12 px; both constants are 10.0.
- `guards.py:10` says the vanish guard applies to four named stage classes; it applies to all 33.
- `lib/pillar_geometry.py:14` describes the pillar offset as "perpendicular" to the tray front edge; the code offsets straight up in image y (`:104`).
- `v6_cascade/STAGE_DESCRIPTIONS.md` (854 lines) is headed "v6.0.0_dev" and its stage table still says stage 0 triages abnormally short segments.
- `outcomes/AGENTS.md` and `outcomes/core/__init__.py` describe the legacy geometric algorithm, ruler units, `displaced_outside`, confidence scores and the `validation_status` workflow. Neither mentions the v6 cascade at all.

**Parameters and imports that have no effect.**
- `Stage2PelletStableUntouched` is constructed with `commit_distance_radii=1.5` (`detector.py:115`). The attribute is stored (`stage_2:85`) and never read. `start_end_window` is likewise stored (`:86`) and never read, and the feature its docstring names as an input, `pellet_position_start_end_distance_in_radii`, is only ever set to `None` in one early-return branch (`:116`) — it is never computed.
- `stage_2` imports `detect_tray_motion_onset` (`:34`) and never calls it. It also defines `ANCHOR_BACK_OFFSET = 10` (`:53`), used only inside a comment.
- `DISPLACED_VANISH_GUARD_CLASSES` is imported by `detector.py:88` and never used.

**Failures that are swallowed.** `process_batch` (`core/batch.py:291-295`) and the v6 CLI loop (`cli.py:208-211`) both catch every exception per video, print `FAILED: <message>`, and continue. Nothing is raised and no file is written; a video with a corrupt pose file simply has no outcomes file afterwards. The watcher's reprocess path is better: `watcher/orchestrator.py:1994-1998` logs the step as failed, marks the video failed, and re-raises. Inside stage 98 and the pixel guard, every video-access failure becomes a silent `continue`.

---

## Configuration

There is almost none. This subsystem has **no config file, no environment variables and no tunable settings** — nothing under `v6_cascade/` reads `os.environ` or `mousereach.config`. Every threshold in the cascade is a module-level constant in the stage file that uses it. Changing behaviour means editing code.

The only things that vary at runtime:

| What | Where | Effect |
|---|---|---|
| `legacy=True` | `core/batch.py` `process_single`, passed through by `process_batch` | Runs the old geometric detector: different algorithm, different output shape, different version string. Reached by `process_single(..., legacy=True)` from Python or by `mousereach-detect-outcomes --legacy` (since 2026-08; before that the flag did not reach it) |
| The napari "Step 2 - Run Pipeline" widget | `pipeline/core.py:566-569` | A separate code path that does run the legacy detector, on every video it processes, with no reach file |
| `video_dir` | `core/batch.py:145-151`, defaults to searching the output directory | If no video file is found, the pixel-based artefact guard and stage 98 both become no-ops. Displaced commits that would have been triaged are committed instead. Nothing reports that this happened |
| Tray type | `pipeline/run_all.py:87-95`, `watcher/orchestrator.py:1062`, `:1971` | Tray `E` or `F` → the whole step is skipped, no file written |
| Whether reaches exist | `core/batch.py:199-202` | No reach file → empty reach list → stage 7 and most touched stages defer, stage 4 loosens; net shift toward `untouched` and toward triage |

The stage list itself (`detector.py:112-146`) is hard-coded and identical for every video. There is no way to enable, disable or reorder a stage without editing `detector.py` or the stage's constants.

---

## Contested claims

This document was written from the source, then checked by a second reader
whose job was to disprove it. The statements below are ones they disputed and
that were not resolved. Neither side is authoritative: spot-checking found the
checker wrong at least once. **Do not rely on anything listed here without
opening the code yourself.** Everything not listed survived two passes.

- **Section 3 (Units): "15 segments fall outside 25-50 px, with a minimum of 2.3 and a maximum of 230.2."**
  - disputed because: The count is 11, not 15. The minimum (2.3) and maximum (230.2) are both correct, so only the count is wrong. Everything else in that paragraph checks out: median ruler_pixels 34.5, 5th-95th percentile 34.3 to 38.4, over exactly 22,920 segments.
- **Section 5.10 (outcome_source, "Filled today"): "On every reach whose features file was written by the current extractor (92% of Processing; older files predate the field)."**
  - disputed because: The share is 89.7%, not 92%. The qualitative claim is right -- outcome_source is non-null on every reach written by extractor 2.0.0 and null on every reach from the older 1.0.0 files -- but the number is off, and it contradicts the document's own arithmetic elsewhere: section 5.3 states that 10.3% of Processing reaches carry a legacy max_extent value, which puts the current-extractor share at 89.7
- **Section 5.7 (Head and body at the apex): all four rows -- head_width_at_apex_mm, nose_to_slit_at_apex_mm, head_angle_at_apex_deg, head_angle_change_deg -- are listed as "99.9%" filled.**
  - disputed because: Only nose_to_slit_at_apex_mm is 99.9%. The other three are lower: head_width_at_apex_mm 99.7%, head_angle_at_apex_deg 99.7%, head_angle_change_deg 99.5%. The difference is small but it is a stated measurement, and head_angle_change_deg is lower than the other two for a reason the table hides -- it needs both ears tracked above likelihood 0.7 at the reach START frame as well as at the apex (feature
- **Section 3 and the section 11 failure table: "If ruler_pixels is 0 or negative the conversion factor is set to 0.0 and every millimetre value in that segment reads exactly 0, not null."**
  - disputed because: True for a negative ruler_pixels, but not for exactly 0. The five cited guard sites (:606, :689, :884, :1048, :1246) all read `RULER_MM / ruler_pixels if ruler_pixels > 0 else 0`, but there is a sixth conversion at feature_extractor.py:406 that has no guard: `px_to_mm = self.RULER_MM / ruler_pixels`. With ruler_pixels == 0 that raises ZeroDivisionError inside _extract_reach_features, outside the t
- **Section 8 (reach_data.csv): "from position 6 onward every column is labelled with the name of the column before it."**
  - disputed because: The shift runs the other way. The SELECT has one extra column (segment_num) inserted at index 5 relative to ALL_COLUMNS, so from index 5 on, the data sitting under a given header name actually belongs to the column BEFORE that name -- equivalently, each data column is labelled with the name of the column AFTER it. The three concrete examples the document gives immediately afterwards are all correc
- **Section 4, parenthetical: "grep -rn \"reach_detector_v8\\|ReachDetectorV8\" src/ returns only the file itself and one docstring mention."**
  - disputed because: That grep returns five hits, not four: three inside reach_detector_v8.py itself, the docstring mention in span_to_reaches.py:279, and a fifth in src/mousereach/improvement/reach_detection/README.md:13. The substantive claim the parenthetical supports -- that nothing imports the class -- is correct; only the stated grep result is incomplete.

---

## Update 2026-08-23

Watcher-side only, no cascade behaviour changed: failures around the cascade's
neighbours (assignment after it, feature extraction and database sync below it)
now write `failed` rows to the step audit table instead of vanishing into
warnings, so a run that half-happened is countable.
