# Kinematics: every value MouseReach measures from a reach

Describes: `src/mousereach/kinematics/core/feature_extractor.py`, `src/mousereach/kinematics/cli.py`, `src/mousereach/kinematics/analysis/reach_export.py`, `src/mousereach/export/features_csv.py`, `src/mousereach/sync/database.py`, `src/mousereach/review/truth_resolver.py`, `REACH_KINEMATIC_DATA_DICTIONARY.md`

Verified against: b65fcf0 (2026-08-23). A `:123` reference means a line of `feature_extractor.py` at that commit, and the percentages were measured then. Sections 5.4, 5.5, 6 (opening), 6.1, 6.3 and 6.7 describe extractor 2.3.0: they were checked against the code at 966f182 and name functions instead of lines.

---

## 0. How the numbers in this document were measured

Three sources, always named where used.

* **Processing corpus** — every video in `<nas_root>\Processing` that has a `_features.json`. Most of those videos were produced by the current detectors; the rest by older ones.
* **Analyzed archive** — a random sample of the `_features.json` files under `<nas_root>\Analyzed`.
* **`docs/FIELD_AUDIT.md`** — the repository's own generated audit. Its database column comes from a parquet snapshot taken before this document was verified, so it cannot show anything that changed after that.

The live central database (`central_db`) was read read-only where noted; it is frequently locked by a running watcher.

---

## 1. What this subsystem is

One class, `FeatureExtractor` (`feature_extractor.py:169`), turns three files into one file.

**Reads**

| Input | Produced by | What is used from it |
|---|---|---|
| `{video}DLC*.h5` | DeepLabCut | Every body-part x, y and likelihood, one row per video frame (`:1507-1525`; falls back to a sibling `.csv`) |
| `{video}_reaches.json` | reach detection | Segment list with `start_frame`, `end_frame`, `ruler_pixels`, `n_reaches`, and every reach's `reach_id`, `reach_num`, `start_frame`, `apex_frame`, `end_frame`, `duration_frames`, `max_extent_pixels`, `max_extent_ruler` |
| `{video}_pellet_outcomes.json` | outcome detection | Per-segment `outcome`, `confidence`, `flagged_for_review`, `interaction_frame`, `causal_reach_id` |

**Writes** `{video}_features.json`: a video record containing segment records, each containing reach records. The whole thing is produced by `dataclasses.asdict` (`:165`), so every declared field appears in the file whether or not anything filled it in.

**Also reads, if present**: human review and ground-truth files, through `resolve_truth_layers` (`:220-225`). These can replace a segment's outcome and can replace the reach list itself. See §9.

**Run by**: `mousereach-grasp-analyze` (`kinematics/cli.py:17`, entry point at `pyproject.toml:132`), the watcher (`watcher/orchestrator.py:1154` and `:2103`), `pipeline/run_all.py:121`, and `pipeline/reprocess_to_current.py:259`.

**Sizes.** `ReachFeatures` declares 44 fields: 43 named values plus a nested dictionary called `extended` holding a further 165 named values. `SegmentFeatures` declares 26: 25 values plus the reach list. Counted by parsing the dataclasses; the 161 from before extractor 2.3.0 confirmed by reading shipped files, and the four anchor columns added by 2.3.0 read from the code (§6.7).

---

## 2. Read this before the tables: nothing the algorithm produces links a reach to an outcome

The extractor decides which reach caused the pellet's fate with one comparison (`feature_extractor.py:380-384`):

```python
causal_reach_id = outcome_data.get('causal_reach_id')
if causal_reach_id is not None and causal_reach_id == reach_id:
    features.causal_reach = True
    features.outcome = outcome_data['outcome']
    features.interaction_frame = outcome_data.get('interaction_frame')
```

**The current outcome detector never writes `causal_reach_id`.** Every segment it emits carries exactly seven keys — `segment_num`, `outcome`, `outcome_known_frame`, `interaction_frame`, `stage`, `flagged_for_review`, and `flag_reason` when triaged. Measured over every outcome file in the Processing corpus:

| Outcome detector | Segments with `causal_reach_id` | Segments with `confidence` |
|---|---|---|
| `v6_cascade` 6.1.0 | **none** | **none** |
| unnamed, version 2.4.4 (older) | some | all |

So `causal_reach = True` only ever arrives from somewhere other than the current algorithm. In the Analyzed-archive sample, the causal reaches come mostly from old pre-v6 files, then from human causal reviews, a few from ground-truth files, and **none** with `outcome_source = 'algo'`. Human reviews and ground truth put a causal reach id onto the segment through the truth-layering step (`review/truth_resolver.py`), and the extractor then picks it up. Since 2026-09-01 that step resolves a human's causal pick by its **frames** whenever the review records them, and only falls back to the stored reach id when it does not: reach ids are renumbered on every run, so a review written against last month's reach list named an id this run does not have, and the human's answer silently vanished from the reach rows. The stored id is kept beside the resolved one for provenance.

**Two things follow, and they are the largest holes in this data.**

1. **`interaction_frame` is thrown away.** The current outcome detector supplies a non-null `interaction_frame` on most of its segments. `SegmentFeatures` has no field for it, and the reach only receives it inside the `causal_reach_id` branch above. So it is discarded. Everything keyed to the moment of contact goes null with it: the reach's `interaction_frame`, and all ten extended `*_at_contact_*` columns. Across the Analyzed sample those ten are populated on well under one percent of reaches.

2. **`distance_to_interaction` can never be filled, for a second and independent reason.** The extractor reads it from `outcome_data['reach_features']` (`:387-389`). No outcomes file contains a `reach_features` key — on no segment, at either detector version. In the older detector the dictionary is genuinely computed (`outcomes/core/pellet_outcome.py:1645-1689`), returned by `find_causal_reach`, assigned to a local variable at `:1572`, and then not passed into the `PelletOutcome` built at `:1591-1600`; the dataclass has no such field. So `.get('reach_features', {})` returns empty every time. Even if causal reaches came back, this stays null.

**The causal link is computed — by a different stage, into a file the extractor never opens.** `{video}_reach_assignments.json` (detector `assignment_v2` 2.1.0, present for every current video in Processing) carries per reach: `is_causal`, `label`, `segment_outcome`, `segment_ifr`, `segment_num`. In a sample of these files, a small share of reaches are marked `is_causal`. Nothing under `src/mousereach/kinematics/` references that file.

---

## 3. Units: how a pixel becomes a millimetre, and where the seconds come from

**Space.** Every segment carries `ruler_pixels`: the straight-line pixel distance between the median positions of the two lower corners of the sample area, `SABL` and `SABR`, taken over a stable middle stretch of the segment and, where at least 50 frames qualify, restricted to frames where both corners are tracked above likelihood 0.9 (`reach/core/geometry.py:97-113`, formula at `:68`). That distance is declared to be 9.0 mm (`feature_extractor.py:177`, `RULER_MM = 9.0`). So

```
mm  = pixels  * 9.0 / ruler_pixels
mm² = pixels² * (9.0 / ruler_pixels)²
```

Measured over the Processing corpus, `ruler_pixels` sits in a narrow band for normal segments. Nothing sanity-checks the number. A few segments fall far outside that band, in both directions; a segment whose `ruler_pixels` is a small fraction of normal has every millimetre value in it inflated by the same factor, silently. If `ruler_pixels` is 0 or negative the conversion factor is set to `0.0` and every millimetre column in that segment reads exactly `0`, not null (`:606`, `:689`, `:884`, `:1048`, `:1246`).

**Time.** Frame counts are exact integers. Anything in seconds divides by **the video's own frame rate** (since extractor 2.5.0, `kinematics/core/frame_rate.py`): the rate the video file's header states, or, when the video is not beside the pose file, the rate DeepLabCut recorded from that header in `<pose>_meta.pickle`. A lab that wants one fixed rate states it (`FeatureExtractor(frame_rate=...)`); a rate below 1 or above 1000 is refused. The rate used and where it came from are written at the top of every features file (`frame_rate`, `frame_rate_source`). If no rate can be found, nothing is guessed: a line says so and the kinematics are not written (`KinematicsProblem`), so the video is held rather than filed with values missing. Five fields use the rate: `velocity_at_apex_mm_per_sec`, `segment_duration_sec`, `time_to_first_reach_sec`, `time_to_outcome_sec` (a copy of `segment_duration_sec`), `mean_inter_reach_interval_sec`. Everything in the `extended` block is per *frame*, never per second.

**Before 2.5.0** the extractor divided by a constant 30 that was an assumption, never a measurement. Videos recorded at 60 frames per second therefore carry, on rows older than 2.5.0, seconds values twice their true length and a `velocity_at_apex_mm_per_sec` converted on a 30-per-second time base. `extractor_version` on the row says which applies.

**Repeated frames (since 2.5.0).** A video's frame rate is how often it shows a frame, not how often a camera took a picture. A recording program that writes frames faster than its cameras deliver pictures repeats the last picture until the next arrives. On a repeated frame the tracked hand does not move; on the next new frame it has moved for two or more frame times at once. Read step by step, a speed then swings between nothing and double. So every *speed* measure reads the path through `fill_repeated_frames` (`kinematics/core/anchored_path.py`): a detected frame within 0.5 px (`REPEATED_FRAME_MAX_PX`) of the last new position counts as a repeat of it, and the repeats are placed evenly on the straight line to the next new position, so each step is the distance between two pictures over the frames between them. Only runs of one or two such frames (`MAX_REPEATED_FRAMES`) are filled, because a recorder repeats a picture for a frame or two; a point that stays put longer is a hand that is really still, and a pause followed by a move keeps the move's own speed. On a video with no repeats the only change is that a point moving under 0.5 px for one or two frames has that movement evened out with the step that follows. It applies to `peak_velocity_px_per_frame`, `velocity_at_apex_px_per_frame` and `_mm_per_sec`, `trajectory_smoothness`, each point's `_peak_speed_*`, `_apex_speed_*` and `_motion_smoothness`, and `paw_velocity_correlation`. Positions, extension, the path and its length, straightness, directness and the mean speeds read the frames as tracked (repeats do not change a path's length).

**A reach that ends on a repeated frame (2.5.0).** The paw is seen until the first new picture that no longer shows it, so the last detected frame (`end_frame`) can repeat the picture before it. That repeat is not a stop: the hand kept moving toward the end point, which it reached by `end_frame + 1`. So a closing run of one or two repeats is placed on the straight line from the last new picture to the end anchor, the same way a run between two pictures is placed on the line to the next picture. The steps from the last new picture onward are then hidden travel toward a calculated point, like the end anchor step, so peak and apex speed leave them out (`tracked_speeds` in `anchored_path.py`): an apex on the last new picture before a closing repeat has no speed, exactly as an apex on the last detected frame has none. The smoothness values and `paw_velocity_correlation` keep those steps, as they keep the anchor steps. The start needs no such rule: the fill always counts the first detected frame as a new position, whatever the frame before the reach holds, because the hand is never read in that frame. Before this, the closing repeat stayed where it was, so the hand read as standing still on the last frame and then jumping to the end point: an apex just before it read a speed of 0, and both smoothness values and the velocity correlation saw a stop that did not happen.

**Timing at the reach boundaries.** When a recorder writes frames faster than its cameras deliver pictures, each picture's capture time is known only to within a frame, and the frame where the paw is first seen, and the first frame after it is last seen, are each up to one picture interval after the true event. `start_frame`, `end_frame` and `duration_frames` are exact frame indices, but their real-time meaning carries that slack, and the furthest extension is a lower bound because the true furthest point can fall between two pictures.

**Before 2.5.0**, on a video with repeated frames, the peak speeds read about twice the hand's speed, a speed at the apex was nothing or about double depending on which frame the apex fell on, the smoothness values measured the alternation, and `paw_velocity_correlation` was raised by every point stopping on the same repeated frames. Do not compare these columns across that version.

**Per frame is per frame of that video.** A value per frame on a 30 frames-per-second video covers twice the time of one on a 60 frames-per-second video. Compare per-frame values only between videos of the same `frame_rate`, or multiply by it to get per second.

**Angles.** Degrees, from `atan2`, so in the range -180 to +180.

**Likelihood.** DeepLabCut's own confidence, 0 to 1, dimensionless.

**Frame windows are inclusive.** A reach's rows are `df.iloc[start_frame:end_frame+1]` (`:396`), so `duration_frames` frames are used. Algorithm-detected reaches set `duration_frames = end - start + 1` (`reach/core/span_to_reaches.py:162`), which matches. Reaches substituted from ground truth set `duration_frames = end - start` (`review/truth_resolver.py:204`), which is one frame short of the window actually measured.

---

## 4. Which direction is "out"? Parts of the file are labelled backwards

This matters because it decides what "how far did the mouse reach" means.

**Measured from the tracking data** (median positions in two videos from different recording setups, both laid out the same way):

| Landmark | x | y |
|---|---|---|
| BOXL (left edge of slit) | left of BOXR | same as BOXR |
| BOXR (right edge of slit) | right of BOXL | same as BOXL |
| Pillar (where the pellet sits) | between BOXL and BOXR | larger than the slit |
| Pellet | at the pillar | at or just past the pillar |
| SA base (SABL / SABR) | either side of the pillar | largest |
| Nose | near the slit | smallest |

The slit runs along **x**: BOXL and BOXR sit apart in x at the same y. The pellet sits at larger **y** than the slit, and the nose at smaller y. So **image y is the extension axis and image x is the sideways axis.**

Three places in the code agree with that, and two do not.

* **Correct.** The `extended` block treats +y as extension: each paw point's apex is `argmax(paw_y − nose_y)` and its `extension_past_nose` is that maximum (`feature_extractor.py:1057-1069`), with a comment at `:1044-1045` stating the reasoning.
* **Correct.** The reach's primary `apex_frame` is the frame of maximum 2-D straight-line distance from nose to `RightHand` (`lib/causal_attribution.py:50-73`, bodypart constant at `:37`), set by the detector in use at `span_to_reaches.py:163`. It is a distance, not an axis, so it is not affected by this problem. (There is an old class, `ReachDetectorV8._find_apex`, that used maximum hand *x* — `reach_detector_v8.py:367-381`. Nothing imports it; `grep -rn "reach_detector_v8\|ReachDetectorV8" src/` returns only the file itself and one docstring mention.)
* **Wrong.** `max_extent_pixels` is inherited from reach detection. The current detector writes `None` (below). Older detectors computed `max_hand_x − BOXR_x` (`reach_detector.py:741`, `reach_detector_v8.py:307`) — a sideways offset. The extractor multiplies whatever it finds straight into millimetres (`:392-393`). Over the reaches in the Processing corpus that still carry a value, the median is **slightly negative** and more than half are negative. A reach extent that is negative more than half the time is not measuring extension.
* **Wrong.** `_compute_pellet_positioning` labels the y difference "lateral" and the x difference "depth" (`:923-933`) — backwards on both. See §7.

---

## 5. Per-reach fields (43, outside `extended`)

"Filled today" is measured on the Processing corpus unless another source is named. Sections 5.4 and 5.5 changed with extractor 2.3.0 and have not been re-measured, so they say when a value is empty instead of how often.

### 5.1 Identity and position in the segment

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `reach_id` | Number for this reach, unique within the video | Copied from reach detection (`:352`) | count | 100% |
| `reach_num` | Which reach this is inside its segment, from 1 | Copied from reach detection (`:360`); the detector assigns it with `enumerate(..., start=1)` (`span_to_reaches.py:160`) | count | 100% |
| `segment_num` | Which pellet presentation this reach belongs to | Set from the enclosing segment (`:317`). The constructor still initialises it to `0` with the comment "Will be set by caller" (`:361`) | count | **Mixed, and a live problem.** The assignment at `:317` landed on 2026-08-20 in commit `5bac3b0`. A large share of reaches in the Processing corpus (files written before that commit) still read `0`; every reach in the Analyzed archive sample reads `0`. And it no longer reaches the database at all — see §8. |
| `is_first_reach` | Is this the first reach of the segment | `i == 0` (`:318`) | true/false | 100% present |
| `is_last_reach` | Is this the last reach of the segment | `i == n-1` (`:319`) | true/false | 100% present |
| `n_reaches_in_segment` | How many reaches in this segment | `len(seg['reaches'])` (`:307`, `:320`) | count | 100% |

### 5.2 Outcome linkage — see §2

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `causal_reach` | Is this the reach that decided the pellet's fate | `causal_reach_id == reach_id` (`:380-382`) | true/false | A small share of reaches in Processing and in the Analyzed sample — **never from the current algorithm.** Only legacy files, human review, and ground truth ever set it. |
| `outcome` | What happened to the pellet | Copied from the segment, **only onto the causal reach** (`:383`) | text | Same rows as `causal_reach`; null on every other reach |
| `interaction_frame` | Frame the paw met the pellet | Copied from the segment, only onto the causal reach (`:384`) | frame index | Same. The detector knows this frame for most segments and it is discarded (§2) |
| `distance_to_interaction` | Frames between apex and the interaction | Read from `outcome_data['reach_features']` (`:387-389`) | frames | **0%, everywhere, permanently** — that key exists in no outcomes file ever written (§2) |

### 5.3 Extent — inherited, wrong axis, and null on current data

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `max_extent_pixels` | Intended: how far the paw reached | Copied from reach detection (`:366`) | pixels | **Null on every reach the current detector produces.** `span_to_reaches.py:179-180` writes `None` on purpose: "Extent is not cheaply available from the flat detector". Only Processing reaches from the older 5.3.0 detector carry a value; those measure the sideways axis (§4). |
| `max_extent_ruler` | Same, divided by the 9 mm ruler | Copied from reach detection (`:367`) | ruler lengths (1.0 = 9 mm) | Same rows |
| `max_extent_mm` | Same, in millimetres | `max_extent_ruler * 9.0` (`:392-393`) | mm | Same rows |

Ground-truth-substituted reaches also get `None` for all three, deliberately: the algorithm's extent was measured over the algorithm's window, not the human's (`review/truth_resolver.py:192-206`).

### 5.4 Speed and path of the wrist point

All four use only the `RightHand` point, along its anchored path (`anchored_hand_path`, §6.1; computed in `_compute_velocity_features`). That path is the detected positions with the start anchor before them and the end anchor after them, so a reach of n detected frames has n + 1 steps. Mean velocity uses all of them. Peak velocity and velocity at apex use only the n - 1 steps between two detected frames (`tracked_steps`), because the first and last steps join a calculated point to a detected one and are charged to one frame, so they measure geometry and not how fast the hand moved. Unlike the `extended` block, the nose is not subtracted, so these are measured in image coordinates. The detected positions are not gated on likelihood; the nose and slit corner that place an anchor are (§6.1).

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `peak_velocity_px_per_frame` | Fastest single-frame move of the wrist along its anchored path | Largest of the n - 1 straight-line distances between two detected frames (a closing repeated frame's steps excepted, §3 'A reach that ends on a repeated frame'). The first and last steps, between an anchor and a detected position, do not count, so this can never be an anchor jump | pixels per frame | Empty for a reach of one detected frame. A reach with no anchor still has it |
| `mean_velocity_px_per_frame` | Average per-frame move of the wrist along its anchored path | Mean of all n + 1 distances, anchor steps included: the whole trip, nose to nose | pixels per frame | Empty when the reach has no anchor |
| `velocity_at_apex_px_per_frame` | Wrist speed leaving the apex | The step *starting* at the apex, to the next detected frame. Never an anchor step | pixels per frame | Empty when the apex is the last detected frame (no detected frame follows it), or is the last new picture before a closing repeated frame (§3), or no apex frame (third note below). A reach with no anchor still has it |
| `velocity_at_apex_mm_per_sec` | The same converted | `× 9.0/ruler_pixels × frame_rate` (the video's own rate; before 2.5.0 an assumed 30) | mm per second | Same rows as `velocity_at_apex_px_per_frame` |

Rows from other extractor versions, by the version on the row. **Before 2.3.0:** every step ran between detected positions, with no anchors, so peak and apex are measured as in 2.4.0, while mean, straightness and smoothness are not. **2.3.0 exactly:** peak could be an anchor jump, and `velocity_at_apex` on a last-frame apex was the step to the end anchor; do not compare these two values across that version.

Three notes on these values. First, peak and mean skip a missing step, but the apex step is read from the unfiltered series, so a gap cannot shift which frame "at apex" refers to; a missing step at the apex just leaves the value empty. Second, there is no quality gate on the detected positions anywhere in this section: DeepLabCut always gives a coordinate, so a frame where it was 5% confident contributes at full weight to every path length, speed and area below.

Third, an inconsistency worth knowing: `_extract_reach_features` computes the apex index for velocity and hand angle as `apex_frame - start_frame if apex_frame else None`, which treats frame 0 as missing, while the body features test `if apex_frame is not None`. A reach whose apex is video frame 0 would get body features but no velocity at apex and no hand angle at apex.

### 5.5 Path shape of the wrist point

Computed in `_compute_trajectory_features` on the same anchored path as §5.4, in image coordinates.

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `trajectory_straightness` | 1.0 = the wrist went in a straight line from anchor to anchor; near 0 = it wandered, or went out and came back near where it started | Straight-line distance from the start anchor to the end anchor ÷ total length of the anchored path | dimensionless 0-1 | Empty when the reach has no anchor |
| `trajectory_smoothness` | 1.0 = perfectly smooth; smaller = jerkier | `1 / (1 + mean|third difference of position|)` along the anchored path, so the anchors are in the differences | dimensionless 0-1 | Empty when the reach has no anchor, and for a reach of one detected frame, whose three path points are too few for a third difference |

Rows from an extractor before 2.3.0 ran these between the first and last detected positions, and smoothness was empty below four detected frames.

### 5.6 Paw orientation

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `hand_angle_at_apex_deg` | Which way the paw was pointing at apex | `atan2(RHRight_y − RHLeft_y, RHRight_x − RHLeft_x)` per frame (`:540-542`), sampled at the apex (`:551-552`) | degrees, -180 to 180 | 100% |
| `hand_rotation_total_deg` | Total turning of the paw across the reach | Sum of absolute frame-to-frame angle changes, with ±180 wraparound handled (`:554-559`) | degrees, always ≥ 0 | 100% |

### 5.7 Head and body at the apex

Each needs both landmarks above likelihood 0.7 at that frame, otherwise null.

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `head_width_at_apex_mm` | Ear-to-ear distance at apex; a body-size proxy | distance(LeftEar, RightEar) × mm-per-pixel (`:613-619`) | mm | Nearly all |
| `nose_to_slit_at_apex_mm` | Nose to the right edge of the slit at apex | distance(Nose, BOXR) × mm-per-pixel (`:626-632`) | mm | Nearly all. Named "slit" but measured to one corner of it. |
| `head_angle_at_apex_deg` | Head orientation at apex | ear-to-ear angle (`:639-644`) | degrees, -180 to 180 | Nearly all |
| `head_angle_change_deg` | How much the head turned between reach start and apex | apex angle − start angle, wrapped into -180..180 (`:646-657`) | degrees | Nearly all |

### 5.8 Tracking quality of the wrist point

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `mean_likelihood` | Average DeepLabCut confidence in the wrist across the reach | mean of `RightHand_likelihood` (`:577`) | 0-1 | 100% |
| `frames_low_confidence` | Frames where wrist confidence was under 0.5 | count (`:578`) | frames | Present on every reach; non-zero on a small share |
| `tracking_quality_score` | — | **Nothing computes it.** Declared at `:83`, read by three exporters and given a database column, assigned nowhere in the repository. | — | **0%, always** |

### 5.9 Flags

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `flagged_for_review` | Does this reach need a human | **Nothing sets it.** Declared `False` at `:86`, never assigned. | true/false | **Always false.** Outcome detection does flag *segments* — a minority of them — and the extractor keeps that at segment level as `outcome_flagged`. The per-reach copy is dead. |
| `flag_reason` | Why | Never assigned (`:87`) | text | **Always null.** `SegmentFeatures` has no `flag_reason` field either, so the detector's reason string is dropped at both levels. |

### 5.10 Where the answer came from

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `outcome_source` | `algo`, `human_review`, or `ground_truth` for this segment's outcome | Copied from the segment after truth layering (`:373`); the segment default `algo` is stamped at `truth_resolver.py:164-165` | text | On every reach whose features file was written by the current extractor (most of Processing; older files predate the field) |
| `reach_source` | `algo`, `ground_truth`, or `human_review` (a reach a reviewer typed that the detector missed, since 2026-10-26) for this reach's frame boundaries | `reach.get('reach_source', 'algo')` (`:374`) | text | Same — but no database column holds it (§8) |
| `causal_reach_source` | Who named the segment's causal reach: `ground_truth` (the reach whose frames contain the ground truth's interaction frame), `human_review` (the reviewer's pick or typed reach), `reach_assignment` (the reach assignment step's pick, merged by the extractor), or null (no causal reach, or one carried by an old outcome file) | Stamped on the segment by `truth_resolver._apply_outcome_layers` / `_add_reviewed_reaches`, or by the extractor's assignment merge; copied onto every reach | text | Features files written since this field was added; no database column holds it |
| `reviewed_by` | Who gave the review answer for this pellet: since 2026-10-28 the person named in that segment's own review record; for a review file saved before that date, whoever saved the file last (all a file then recorded) | Copied from the segment (`:375`) | text | Empty in Processing (no reviewed videos there); filled only on reviewed videos in the Analyzed archive (see `FIELD_AUDIT.md` for the current rate) |
| `algo_outcome` | What the algorithm said before a human overrode it | Copied from the segment (`:376`), stamped at `truth_resolver.py:172` | text | Same rows as `reviewed_by` |
| `algo_causal_reach_id` | Which reach the algorithm blamed before a human overrode it | Copied from the segment (`:377`), stamped at `truth_resolver.py:173` from whatever `causal_reach_id` the algorithm had | count | **0%.** It preserves the algorithm's causal reach id, and the current detector never sets one (§2). `FIELD_AUDIT.md` also reports it empty. |

### 5.11 Fields declared and never computed by anything

Four more with the same story as `tracking_quality_score`: a name in the dataclass, a database column, exporter lines that read them, and no code anywhere that assigns them.

| Field | Declared | Intended meaning | Replacement that does exist |
|---|---|---|---|
| `grasp_aperture_max_mm` | `:66` | Widest paw opening during the reach | `paw_width_proxy_max_mm` in `extended` |
| `grasp_aperture_at_contact_mm` | `:67` | Paw opening when it met the pellet | `paw_width_proxy_at_contact_mm` (itself null, §2) |
| `apex_distance_to_pellet_mm` | `:76` | How close the paw got to the pellet | none |
| `lateral_deviation_mm` | `:77` | How far sideways the paw strayed | `righthand_lateral_deviation_mm` and the same for the three other paw points |

---

## 6. The `extended` block: 165 values per reach

Produced by `_extended_features` and stored as one nested dictionary. If any part of it raises, the whole block is replaced by a single key `_extended_features_error` holding the message. Before the kinematics-problems change the reach then shipped with all 165 values missing and no other warning; now `extract()` raises `KinematicsProblem` naming the reach and writes no features file, and the archive step refuses a features file that still carries the key (§11).

Rows from an extractor before 2.3.0 hold the same block without the four anchor columns of §6.7, so 161 values. The ten `_at_contact` columns are null almost always (§2). A reach with no anchor (§6.1) also has null path measures: 12 path columns of each paw point (§6.2) and `paw_velocity_correlation`. Its peak and apex speed are still given, because they read detected frames only (on 2.3.0 rows they were null too).

### 6.1 The coordinate frame these use

Two things are done before every measurement of where the paw is (`anchored_hand_path` in `kinematics/core/anchored_path.py`, whose text explains the rule):

1. **Nose subtraction.** Every paw position becomes `paw − nose` at the same frame, so head movement does not count as paw movement.
2. **Two anchor frames** are added, because a reach starts and ends at the nose and the hand is hidden in these frames:
   * frame before detection (`start_frame - 1`): the midpoint of Nose and BOXL in that frame
   * frame after the paw disappears (`end_frame + 1`): the midpoint of Nose and BOXR in that frame

   These are geometry, not tracking, so the reach always starts and ends at a defined place instead of wherever DeepLabCut first noticed the paw. Since extractor 2.3.0 the anchors also cover the wrist velocity and trajectory measures (§5.4, §5.5), which do not subtract the nose, and `paw_velocity_correlation` (§6.3), which does.

**Which measures include the anchors.** Mean wrist velocity, trajectory straightness and smoothness, total path, lateral spread, swept area, path directness, motion smoothness, lateral deviation, mean speed, and velocity correlation. Since extractor 2.4.0, peak and apex speed (wrist and per-point) leave out the first and last steps, so they come only from steps between two detected frames: `tracked_steps` in `anchored_path.py` says why. The apex frame and `extension_past_nose` use detected frames, because they ask for the paw's furthest point and an anchor can never be that. Paw shape, visibility and the head measures use detected frames, because the anchors say nothing about them. `duration_frames` still counts detected frames.

**Where a landmark is untrusted.** `anchor_start_source` and `anchor_end_source` on the row say how each anchor was placed (`anchor_frame`, `nearest_frame`, `video_edge`, or `none`). With `none`, the path measures are empty.

**Every place in the code that reads paw position, and whether it uses the anchors:**

| Place | What it does with paw position | Anchors? | Why |
|---|---|---|---|
| `kinematics/core/feature_extractor.py`: mean wrist velocity, trajectory, per-point path and mean speed, velocity correlation | Measures movement | Yes | This is the rule |
| `kinematics/core/feature_extractor.py`: peak and apex speed, wrist and per-point | How fast the hand moved | No, the steps to and from them are left out | Those steps measure geometry, not speed |
| `kinematics/core/feature_extractor.py`: hand angle, paw shape, visibility, pellet-motion count | Paw shape, or how well it was seen | No | At an anchor all points are one place and nobody saw them |
| `kinematics/analysis/real_kinematics.py` | Measures movement | Yes, except `peak_speed_mm_s`, which leaves out the anchor steps | As the two rows above |
| `biomech/dlc_to_trc.py` | Limb motion for one reach | Yes | The limb's trip from and back to the nose is part of the movement |
| `biomech/skeleton_inference.py` | Solves one frame as tracked | Through its input | Pass `anchored_reach_frames` to solve one reach |
| `lib/causal_attribution.py` | Paw's furthest point from the nose | No | Asks for a maximum, which an anchor can never be |
| `reach/` detectors, refiners, polisher, and `reach/v8/` | Decide when a reach happens | No | They decide from what was seen, and the anchors depend on their answer |
| `reach/analysis/` | Evaluate reach detection | No | Same |
| `outcomes/` | Decide what happened to the pellet | No | Decides from what was seen |
| `lib/pillar_geometry.py` | Lists tracked point names | No | Reads no paw position |
| `pipeline/triage.py`, `dlc/core/quality.py` | Tracking quality | No | Judges the tracker, so needs what it reported |
| `review/ground_truth_widget.py`, `reach/review_widget.py` | Show tracked values, find visible frames | No | A reviewer must see what was tracked |
| `config.py` | Lists point names | No | Reads no position |

### 6.2 Per paw point — four points × 19 values = 76

The four points are `RightHand`, `RHLeft`, `RHRight`, `RHOut` (`:990`), written in column names as `righthand`, `rhleft`, `rhright`, `rhout`. The data dictionary is explicit that the anatomical meaning of these labels is not validated; treat them as four stable points on the paw outline.

| Column suffix | Plain English | How computed | Units |
|---|---|---|---|
| `_apex_frame` | The frame this point reached furthest out | `argmax(paw_y − nose_y)` over real frames (`:1057-1060`) | frame index |
| `_extension_past_nose_px` / `_mm` | How far past the nose this point got, along the out-through-the-slit direction | that maximum value (`:1061`, `:1068-1069`) | pixels / mm |
| `_total_path_px` / `_mm` | Total distance travelled, following every wiggle | sum of frame-to-frame distances on the anchored path (`:1072-1077`) | pixels / mm |
| `_lateral_spread_px` / `_mm` | Side-to-side range covered | `max(x) − min(x)` on the anchored path (`:1080-1085`) | pixels / mm |
| `_swept_area_px2` / `_mm2` | Area enclosed by the loop the paw traced | shoelace polygon over the anchored path (`:1088-1096`) | pixels² / mm² |
| `_path_directness` | 1.0 = straight from entry anchor to exit anchor; smaller = more winding | anchor-to-anchor distance ÷ total path (`:1102-1106`) | dimensionless |
| `_motion_smoothness` | 1.0 = smoothest; smaller = jerkier | `1/(1 + mean|third difference|)` on the anchored path (`:1109-1115`) | dimensionless |
| `_lateral_deviation_px` / `_mm` | Furthest the path strayed from the straight anchor-to-anchor line | max perpendicular distance (`:1123-1131`) | pixels / mm |
| `_mean_speed_px_per_frame` / `_mm_per_frame` | Average speed | total path ÷ number of steps (`:1134-1141`) | pixels or mm per frame |
| `_peak_speed_px_per_frame` / `_mm_per_frame` | Fastest single step | largest step between two detected frames; never an anchor step. Null for a reach of one detected frame | pixels or mm per frame |
| `_apex_speed_px_per_frame` / `_mm_per_frame` | Speed at this point's own apex | the step from the apex frame to the next detected frame. Null when this point's apex is the last detected frame, or the last new picture before a closing repeated frame (§3) | pixels or mm per frame |

Rows from extractors before 2.4.0 differ on those two: their peak could be an anchor step, and a last-frame apex speed was the step to the end anchor. Those are geometry, not measured speed, which is why 2.4.0 removed them.

### 6.3 Coordination between the four points — 3 values

Computed in `_cross_paw_timing`.

| Column | Plain English | How computed | Units |
|---|---|---|---|
| `paw_apex_lead_frames` | How spread out in time the four points' apexes were | latest apex − earliest apex; null if fewer than two points have an apex | frames |
| `paw_leading_point` | Which point apexed first | the point with the smallest apex frame, as a name. Apexes come from detected frames (§6.2). A tie goes to the first in the order `righthand`, `rhleft`, `rhright`, `rhout`; null if fewer than two points have an apex | text: one of the four names |
| `paw_velocity_correlation` | How much the four points sped up and slowed down together | mean of the pairwise Pearson correlations of the four frame-to-frame speed series (up to six pairs; a pair with a constant series is skipped), on the anchored nose-relative paths, so the first and last steps, from the start anchor and to the end anchor, are in every series (unchanged in 2.4.0). Null when the reach has no anchor or no pair could be correlated | dimensionless, -1 to 1 |

On rows from an extractor before 2.3.0, `paw_velocity_correlation` used detected frames only.

### 6.4 Paw shape — 3 measures × 8 time points × 2 units = 48

Computed per frame on real frames only, in raw pixel coordinates (no nose subtraction), then summarised (`_paw_shape_features`, `:1213-1332`).

| Measure | Column root | Per-frame calculation | Units |
|---|---|---|---|
| Paw width | `paw_width_proxy` | distance from `RHLeft` to `RHRight` (`:1249-1252`) | px / mm |
| Paw spread | `paw_spread_max` | largest of the six pairwise distances among the four points (`:1254-1260`) | px / mm |
| Paw outline area | `paw_outline_area` | shoelace area of all four points, ordered each frame by angle around their centre so rotation does not break the polygon (`:1262-1280`) | px² / mm² |

Eight summaries each (`:1290-1330`): `_at_start` (first reach frame), `_at_apex` (the reach's primary apex — see §4), `_at_end` (last reach frame), `_at_contact` (the interaction frame), `_max`, `_min`, `_mean`, `_range` (max − min).

All are populated except the six `_at_contact_*` columns, which are almost always null for want of an interaction frame (§2).

### 6.5 Visibility of each paw point — 4 points × 8 + 1 = 33

`_paw_visibility_features` (`:1334-1395`). Reports DeepLabCut likelihood rather than filtering on it, on the stated reasoning that a point disappearing often means the paw rotated, closed, or was occluded.

| Column | Meaning | Units |
|---|---|---|
| `{point}_visibility_at_start` / `_at_apex` / `_at_end` | Likelihood at those frames | 0-1 |
| `{point}_visibility_at_contact` | Likelihood at the interaction frame | 0-1 — almost always null (4 columns) |
| `{point}_visibility_max` / `_min` / `_mean` / `_range` | Across the reach | 0-1 |
| `frames_any_paw_low_confidence` | Frames where the *worst* of the four points was under 0.5 (`:1387-1391`) | frames |

Measured over the Processing corpus, only a small share of reaches contain any such frame. Within detected reaches, paw tracking confidence is essentially always high.

### 6.6 Tray contact — 1 value

| Column | Meaning | How computed | Units |
|---|---|---|---|
| `tray_contact_duration_frames` | Intended: frames where the pellet was jiggling while the wrist was hidden — a proxy for the paw being on the tray | count of frames where the 4-frame rolling standard deviation of `Pellet_x` exceeds 2 px **and** `Pillar_likelihood < 0.5` **and** `RightHand_likelihood < 0.5` (`:1414-1421`) | frames |

**This value is zero on almost every reach.** Only a handful of reaches in the whole Processing corpus have a non-zero count. That is structural, not incidental: the window is a *detected reach*, which means the wrist was visible enough to be detected, while the third condition requires the wrist to be invisible. Returns 0 for reaches shorter than 4 frames (`:1411-1412`) and `None` if any of those three columns is missing from the tracking file (`:1422-1423`). The 2-pixel threshold is a raw pixel constant, not scaled by `ruler_pixels`, so it means a different physical distance on differently-zoomed videos.

### 6.7 How the anchors were placed — 4 values

New in extractor 2.3.0, from `anchored_hand_path` on the `RightHand` point. All four paw points share the same anchors, because an anchor depends only on the nose and a slit corner.

| Column | Plain English | How computed | Units |
|---|---|---|---|
| `anchor_start_source` / `anchor_end_source` | How this end's anchor was placed | `anchor_frame`: nose and slit corner both trusted (likelihood at least 0.7) in the anchor frame. `nearest_frame`: a landmark was not trusted there, so its nearest trusted position within 30 frames was used. `video_edge`: the anchor frame is outside the video, so the nearest frame inside it stands in. `none`: no trusted landmark in range, so there is no anchor | text |
| `anchor_start_in_neighbour_reach` / `anchor_end_in_neighbour_reach` | The anchor frame falls inside another detected reach | Only the nose and slit are read from that frame, never the hand, and the anchor stands | True / False |

---

## 7. Per-segment fields (28)

One record per pellet presentation. All are written to `{video}_features.json`; only five of them ever reach the database (§8).

| Field | Plain English | How computed | Units | Notes |
|---|---|---|---|---|
| `segment_num` | Which pellet presentation | copied (`:289`) | count | The only place the pellet number is recorded correctly |
| `start_frame`, `end_frame` | Segment bounds | copied (`:290-291`) | frame index | |
| `ruler_pixels` | The 9 mm reference in pixels for this segment | copied (`:292`) | pixels | drives every mm conversion in the segment |
| `outcome` | What happened to the pellet | copied from outcome detection (`:293`) | text | 100% |
| `outcome_confidence` | Detector's confidence | copied (`:294`) | 0-1 | **Null on every segment the current detector produces.** Present only on the segments of legacy-detector videos. |
| `outcome_flagged` | Detector asked for human review | copied (`:295`) | true/false | A minority of current segments |
| `n_reaches` | Reach count | since extractor 2.2.0: the number of reaches MEASURED, after a reviewer's ignore windows (next three rows) are applied; before 2.2.0 copied from the reach file | count | Equals `n_reaches_in_segment` and the length of `reaches` |
| `n_reaches_left_out` | Reaches a person took out of the measurements: by a reviewer's ignore window, or by a reach marked "exclude from analysis" in the ground-truth tool | `split_reaches_by_ignore_windows` (extractor 2.2.0); `truth_resolver._drop_excluded_reaches` | count | 0 when the segment has neither |
| `reaches_left_out` | Which reaches were left out and why | list of `{reach_id, start_frame, end_frame, ignore_window}` (ignore window) or `{reach_id, start_frame, end_frame, excluded_by_person: {start_frame, end_frame, reason}}` (ground-truth exclusion: the excluded reach's frames and the reason the person typed) | list | null when nothing was left out. The reach is recorded here so the file still says it happened; it has no row in `reaches` and no kinematics. Every detected reach that overlaps an excluded reach by one frame or more is left out |
| `ignore_windows` | The reviewer's ignore windows for this segment | copied from the review (`answers.abnormal_ranges`, carried onto the segment as `abnormal_ranges`) | list of `{start_frame, end_frame, reason}` | null when none. A window names frames where something other than a reach moved the pellet. **A reach touching a window by even one frame is left out of every measurement**, because its velocity, path and extent would otherwise be computed over frames a person has said are not the reach. The temporal-context and pellet-positioning fields above are computed from the measured reaches only |
| `causal_reach_id` | Which reach caused the outcome | copied (`:297`) | count | Null on every current segment (§2) |
| `attention_score` | Share of frames the mouse was at the tray | frames where `Nose_likelihood > 0.9` and `Nose_y > median(BOXR_y) − 80`, ÷ total × 100 (`:790-808`) | percent 0-100 | The 80 is raw pixels, not scaled by `ruler_pixels`. |
| `mean_head_width_mm` | Average ear-to-ear distance | mean over frames with both ears above likelihood 0.7 (`:694-702`) | mm | |
| `mean_nose_to_slit_mm` | Average nose-to-BOXR distance | same gate on Nose and BOXR (`:709-716`) | mm | |
| `mean_nose_height` | Average nose y position | mean of `Nose_y` over frames with `Nose_likelihood > 0.7` (`:723-726`) | **raw image pixels, not converted, not referenced to anything** | comparable only within one camera setup |
| `mean_head_angle_deg` | Average head orientation | plain arithmetic mean of ear-to-ear angles, over frames with both ears above 0.7 (`:734-739`) | degrees | **Not valid as written.** Measured angles span nearly the full circle; averaging a wrap-around quantity arithmetically is meaningless when it straddles ±180. |
| `head_angle_variance` | Head-orientation steadiness | plain variance of the same angles (`:740`); null when fewer than 2 frames pass the gate | degrees² | Same problem, worse: the value is dominated by the ±180 jump, not by head steadiness. |
| `nose_position_variance` | Postural steadiness | `var(Nose_x) + var(Nose_y)` over frames with `Nose_likelihood > 0.7`, null if fewer than 2 (`:749-756`) | **pixels², not converted to mm²** | |
| `segment_duration_sec` | Length of the presentation | `(end − start + 1) / frame_rate` | seconds (before 2.5.0: an assumed 30 frames per second) | |
| `time_to_first_reach_sec` | Delay before the mouse first reached | `(first reach start − segment start) / frame_rate` | seconds | null if the segment has no reaches |
| `time_to_outcome_sec` | Intended: when the outcome happened | **`= segment_duration_sec`, unconditionally** (`:844`) | seconds | Carries no information beyond segment length. The comment says "assume outcome happens at segment end". The outcome detector *does* compute a real answer, `outcome_known_frame`, present on most current segments; the extractor never reads it. |
| `mean_inter_reach_interval_sec` | Average gap between reaches | mean of (next start − previous end) / frame_rate | seconds | null when fewer than 2 reaches; can be negative if reaches overlap |
| `pellet_lateral_offset_mm` | Labelled sideways offset of the pellet | `abs(pellet_y − BOXR_y) × mm-per-pixel` (`:924-925`) | mm | **Measures the out-through-the-slit direction, not sideways** (§4) |
| `pellet_depth_offset_mm` | Labelled depth offset of the pellet | `abs((pellet_x − BOXR_x) − 30) × mm-per-pixel` (`:930-933`) | mm | **Measures the sideways direction** (§4), and subtracts a hard-coded ideal of 30 raw pixels that belongs on the other axis |
| `pellet_position_idealness` | 0-1 score for how well placed the pellet was | `1 − (0.6 × min(lateral/50, 1) + 0.4 × min(depth_deviation/40, 1))`, floored at 0 (`:938-942`) | dimensionless 0-1 | Built from the two mislabelled axes with three more raw-pixel constants (50, 40, 30). Over the Processing corpus it stays low and never comes near 1 — consistent with the depth term sitting at full penalty nearly always. |
| `mean_tracking_quality` | Average confidence over every tracked point | mean of all `*_likelihood` columns in the segment (`:967-976`) | 0-1 | includes apparatus landmarks, not just the mouse |
| `tracking_dropout_frames` | Frames where any tracked point was under 0.5 | count (`:979-981`) | frames | |
| `reaches` | The reach records | | list | |

The three pellet-position values are measured over a window that runs from the segment start to the start of the first reach, or the first 30 frames if the segment has no reaches (`:886-892`), using the median pellet position over frames where `Pellet_likelihood > 0.7` (`:902-911`). All three are null if the pellet is never tracked well in that window.

---

## 8. What survives the trip to the database and the CSVs

The extractor's own file is complete. Everything downstream loses something, and the database path is currently broken.

### `{video}_features.json`

Everything above.

### The central database (`central_db`), table `reach_data` — **inserts fail after commit `5bac3b0`**

One row per reach is built at `sync/database.py:576-608` and inserted at `:619-632`.

* The insert names `ALL_COLUMNS` (`:84-92`), which is `REACH_JSON_COLUMNS` (41 names, `:61-78`) plus five session-identity, five segment-context, four metadata and six provenance columns — 61 in all.
* `segment_num` is **not** in that list. It was removed by commit `5bac3b0` on 2026-08-20, on the reasoning (comment at `:56-60`) that the reach's copy was always 0 and should come from the segment instead. The row dictionary does carry `segment_num` from the segment (`:568`), but the INSERT statement never names it.
* The live table declares `segment_num INTEGER NOT NULL` with no default. Reproducing the exact statement against a fresh table built from `CREATE_REACH_DATA_SQL` raises `IntegrityError: NOT NULL constraint failed: reach_data.segment_num`. **Every insert fails.**
* This is invisible. The DELETE and the INSERTs share one connection block that is only committed at the end (`:623-634`), so the failure rolls the DELETE back — existing rows survive, nothing is corrupted, and nothing new arrives. `sync_file_to_database` then catches every exception and returns `False` (`:857-858`), so the callers' own `except` blocks never fire. The watcher logs it at DEBUG as "Database sync skipped (subject not in DB or DB unavailable)" (`orchestrator.py:2129-2130`) — a message that names two causes, neither of which is the real one.
* Confirmed on a live database: the newest `imported_at` in `reach_data` dated from the moment commit `5bac3b0` landed, and nothing was written after it.

When it did work, the table:

* Kept **41** of the 43 per-reach fields, plus the entire `extended` dictionary as a JSON blob in one column, `extended_features` (`:595`).
* Kept exactly five segment-level fields: `segment_outcome`, `segment_outcome_confidence`, `segment_outcome_flagged`, `attention_score`, `pellet_position_idealness` (`:565-574`).
* Dropped **`reach_source`** — computed on every reach, no column.
* Dropped **the other fourteen segment measurements entirely**: `mean_head_width_mm`, `mean_nose_to_slit_mm`, `mean_nose_height`, `mean_head_angle_deg`, `head_angle_variance`, `nose_position_variance`, `segment_duration_sec`, `time_to_first_reach_sec`, `time_to_outcome_sec`, `mean_inter_reach_interval_sec`, `pellet_lateral_offset_mm`, `pellet_depth_offset_mm`, `mean_tracking_quality`, `tracking_dropout_frames`. No column exists for any of them. They are computed on every video and read by nobody. They also never appear in `docs/FIELD_AUDIT.md`, because that tool compares reach-level record keys against database columns, and these are neither.

### `database_dump/reach_data.csv` beside the central database — **an older file on disk has misaligned columns**

`export_csv` (`sync/database.py:749-805`) selects 57 columns (`:759-784`). Relative to the table it includes `segment_num` and omits `extended_features`, `outcome_source`, `reviewed_by`, `algo_outcome`, `algo_causal_reach_id`. So all 161 extended values and all review provenance are absent from the file that downstream tools read.

Older code wrote the header from `ALL_COLUMNS` while the rows came from that separate SELECT. Commit `eec74e6` (2026-08-21) fixed it by taking the header from the query itself. **A file written before the fix** carries a 61-name header over 57-value rows. Verified by reading one: the header has 61 names, the first data row has 57 values, and from position 6 onward every column is labelled with the name of the column before it. The column headed `reach_id` holds pellet numbers; the one headed `causal_reach` holds outcome strings such as `displaced_sa`; the one headed `max_extent_pixels` holds durations. Such a file is not regenerated until a sync succeeds, because `export_csv` runs only after a successful sync (`:854`), and syncs fail while the insert bug above stands.

### `mousereach-reach-export`

`kinematics/analysis/reach_export.py`. The command (`main`, `:292-301`) writes **one aggregate file**, `reach_kinematics.csv`, beside the Processing directory. Its rows come from each video's `_features.json`; a video in Processing with no features file yet gets basic rows from its `_reaches.json` and `_pellet_outcomes.json` -- the algorithm's own calls before any person's answer, labelled `outcome_source` `algo` -- and the command prints the names of those videos so nobody takes their rows for results. It does not write `{video}_results.csv` and does not write cohort CSVs. Those are separate functions: `write_video_results_csv` (`:239-253`) is called only from `pipeline/reprocess_to_current.py:267`, and `export_cohort` (`:267-289`) has no caller anywhere in `src/`.

The column list is not fixed. `write_csv` (`:218-236`) walks every row and appends every key not already in `STATIC_COLUMNS` (`:24-62`), so `fieldnames` is a superset of everything present and the `extrasaction="ignore"` at `:232` can never drop anything. All 161 extended values are carried.

Six values are nevertheless missing, for a different and worse reason: **the row builder never reads them.** `_row_from_features_reach` (`:65-113`) copies 33 named keys and then `reach["extended"]`, and `velocity_at_apex_px_per_frame`, `velocity_at_apex_mm_per_sec`, `peak_velocity_px_per_frame`, `mean_velocity_px_per_frame`, `trajectory_straightness` and `trajectory_smoothness` are not among them. They exist in the features file and simply never enter the row.

### `mousereach-features-csv`

`export/features_csv.py` globs `*_grasp_features.json` (`:16`). Nothing in the pipeline writes a file with that name — kinematics writes `{video}_features.json`. The command always prints "No *_grasp_features.json files found!" and exits (`:18-21`). Its row dictionary (`:38-77`) has 30 columns, of which 5 are the never-computed fields of §5.11; the other 25 are fields the extractor does populate.

### `kinematics/analysis/odc_sci_exporter.py`

Reads the features file (`:109-110`). Six of its columns read keys that the features schema does not have, so they are constants: `Exclude From Analysis`, `Exclude Reason`, `Human Corrected`, `Source`, `Review Note` (`:157-161`, all reach-level keys that live in `_reaches.json`, not in `ReachFeatures`), and `Segment Flagged` / `Segment Flag Reason` (`:162-163`, which look for `flagged_for_review` and `flag_reason` on a segment record whose actual field is `outcome_flagged` and which has no reason field at all).

---

## 9. Human corrections, and one path that cannot fire

`extract` calls `resolve_truth_layers` (`:220-225`), which resolves each segment's outcome and each reach independently by ground truth > human review > algorithm -- the human review read once from every copy of the video's review, newest answer per segment (`causal_review_io.effective_review`, the read the review tools use) -- and can replace the reach list itself when ground truth marks its reach set exhaustive (`:275-282`). Provenance lands in `outcome_source` and `reach_source`.

That resolver takes a **directory** as `primary_dir` and looks for `<dir>/<video>/<video>_causal_review.json` or `<dir>/<video>_causal_review.json` (`truth_resolver.py:72-88`). But every production caller passes a **file** path: `resolve_review_path` returns the review file itself (`review/causal_review_io.py:83-90`) and the watcher, `run_all.py` and `reprocess_to_current.py` hand that straight in as `review_path`; the CLI builds `Path(_rd) / f"{video_name}_causal_review.json"` (`kinematics/cli.py:123`) and does the same. The lookup then builds paths like `.../X_causal_review.json/X_causal_review.json`, `is_file()` returns false, and that layer is silently skipped.

The practical damage is limited, because the resolver also checks `Paths.TRIAGE_REVIEW` and `Paths.DEEP_REVIEW` on its own (`:315-316`), and the review tool's default save location is `Paths.TRIAGE_REVIEW` (`review/staging.py:57`). A review saved only in the processing directory, or only next to the video, does not reach kinematics.

If `resolve_truth_layers` raises, the code prints a line and falls back to the older review-only path, which does accept a file (`:226-239`); if that raises too it prints a second line and continues with the raw algorithm output.

**A re-run for kinematics only keeps what was filed (since 2026-11-17).** Because the decisions above are worked out again on every run, a filed video re-run only for its kinematics (a new extractor version, or an answer saved after filing) could come out with different reaches, outcomes or causal reaches than were filed -- when the resolving rules changed since filing, or a file the filing run did not see is found now. In such a re-run the watcher hands `extract` the filed features file (`keep_filed=`) and the kinds of answer the filed results have not applied yet (`owed=`). If the reaches (by frames), the reaches left out, an outcome or a causal reach (by frames) would differ, and the new decision is not made by an owed answer, nothing is written: `KinematicsProblem` names each pellet and what would change, the video is held, and the filed results stand. Rule and reasons: `kinematics/core/filed_decisions.py`; operator view: `docs/PIPELINE_AS_BUILT.md`, "a kinematics-only re-run keeps the filed decisions".

---

## 10. What a correct segment is load-bearing for

If a segment boundary is in the wrong place, or a segment is renumbered, these change.

| What depends on it | Where | What goes wrong |
|---|---|---|
| **Which outcome is attached to which segment** | `:248` — `zip(reaches_data['segments'], outcomes_data['segments'])` | The two files are paired **by position in the list**, not by `segment_num`. Nothing compares the two numbers, and nothing compares their frame ranges. If the outcome file was written against a different segmentation from the reach file, every outcome is silently attached to the wrong pellet. |
| **Every millimetre value in the segment** | `:292`, then `:406`, `:606`, `:689`, `:884`, `:1048`, `:1246` | `ruler_pixels` is measured over a stable stretch *of that segment*. Move the boundary and you move the frames the SABL/SABR medians come from. |
| **Every segment-level measurement** | `:784`, `:688`, `:964` | `attention_score`, all six body/posture values, and both quality values are computed over `df.iloc[start_frame:end_frame+1]`. |
| **The pellet-position values** | `:886-892` | The assessment window runs from the segment start to the first reach's start. A segment that begins too early measures the *previous* pellet's position; one with no reaches measures a fixed 30 frames from wherever it begins. |
| **`segment_duration_sec`, `time_to_first_reach_sec`, `time_to_outcome_sec`** | `:832-844` | All three are pure functions of the boundaries. `time_to_outcome_sec` is the duration, so it is nothing but a boundary readout. |
| **`segment_num`, `is_first_reach`, `is_last_reach`, `n_reaches_in_segment`** | `:307`, `:317-320` | These come from the reach detector's assignment of reaches to segments, not from anything kinematics checks. A reach that lands in the wrong segment gets the wrong pellet number and shifts the first/last flags and the count for two segments at once. |
| **Which ground-truth reaches replace which algorithm reaches** | `truth_resolver._apply_gt_reaches` | Only a GT reach whose start AND end a person determined (`gt_reach_is_determined`; older files say `start_verified` / `end_verified`) is applied; the tool's undetermined seeds are the algorithm's guesses and were applied as `reach_source: ground_truth` before. With the reach set marked exhaustive, a segment's reaches are replaced only when every GT reach in it is determined; otherwise its determined reaches are overlap-merged and the detected ones stay (logged with the segment and count). GT reaches are bucketed by the CURRENT segment holding most of their frames (`_segment_of_reach`); the `segment_num` they were labelled with is used only when the current segments carry no frames or none holds the reach. Before this, a re-cut after the ground truth was written put GT reaches on the wrong footage. (Review *outcomes*, and ground-truth *outcomes*, are safer: they are re-anchored by frame range when `current_segments` is passed -- a ground-truth outcome by the frames of its segment in the ground truth's own segmentation, `causal_review_io.index_gt_outcomes_by_segment`. That segmentation places each saved cut by its own `index`, not by its position in the list (`gt_segments`): a file saved with only the cuts a person determined has segment N only where cuts N-1 and N are both present; any other outcome keeps its number.) |
| **The pellet number recorded downstream** | `sync/database.py:568` | The database's `segment_num` column is taken from the segment record; it is the only place the pellet number is preserved. |

Two things segments do **not** control. Reach frame windows are never clipped to the segment — a reach is measured over its own `start_frame`..`end_frame` regardless of where the boundary sits (`:396`). And the extractor never re-derives or validates a boundary; it takes `start_frame` and `end_frame` verbatim.

---

## 11. Failures that produce a file anyway

| Failure | What happens | Where |
|---|---|---|
| Anything inside the extended feature block raises | No longer produces a file: `extract()` raises `KinematicsProblem` naming the reach, no features file is written and the watcher holds the video as failed. The archive step refuses a features file still carrying `_extended_features_error` (from an older extractor) | `_extract_reach_features`, `KinematicsProblem` in `feature_extractor.py`; `refuse_filing_with_failed_measures` in `orchestrator.py` |
| Truth layering raises, or a review copy, the ground-truth file or the reach assignment file cannot be read | No longer produces a file: `extract()` raises `KinematicsProblem` listing each problem (it used to print a line and fall back to the review alone, then to the algorithm's answers) | `extract` in `feature_extractor.py`; `resolve_truth_layers(applied=...)['problems']` in `truth_resolver.py` |
| Database sync fails, for any reason | `sync_file_to_database` returns `False` and never raises | `sync/database.py:857-858` |
| Database sync fails, from the CLI | `except Exception: pass` — and the call cannot raise anyway, so nothing is printed either way | `kinematics/cli.py:143-144` |
| Database sync fails, from the watcher | `logger.debug("Database sync skipped ... subject not in DB or DB unavailable")` — the wrong cause | `orchestrator.py:2126-2130`; the other watcher path (`:1166-1170`) logs nothing at all on a `False` return |
| Feature extraction fails, from either watcher role | Since 2026-11-15: the row is marked `failed` ("not filed: no kinematics ...") and the video is NOT filed; before that it was logged as a warning and filed anyway | `BaseOrchestrator._make_features` / `_hold_without_kinematics` in `orchestrator.py` |
| The features file of a Pillar video still holds a pellet with no decided outcome (`triaged`, `uncertain` or none) when it is about to be filed | The row is marked `failed` ("not filed: a pellet is still undecided in its kinematics ...", pellets named) and the video is NOT filed; the failed-video retry runs it again through the review gate. Before, it was filed with `outcome: triaged` | `refuse_filing_with_undecided_pellet` in `orchestrator.py` |
| The reach file and the outcome file describe different segments | Nothing checks | `feature_extractor.py:248` |
| `ruler_pixels` is 0 or negative | Every millimetre value in the segment becomes exactly `0.0`, not null | `:606`, `:689`, `:884`, `:1048`, `:1246` |

---

## 12. Reconciling with `docs/FIELD_AUDIT.md`

That document reports fill rates over the videos it scanned, using a parquet snapshot for its database column. Every kinematics-owned field it lists as empty, with the reason:

| Field | Audit bucket | Why it is empty |
|---|---|---|
| `tracking_quality_score` | never computed | Declared at `:83`, assigned nowhere in the repository |
| `apex_distance_to_pellet_mm` | never computed | Declared at `:76`, assigned nowhere |
| `lateral_deviation_mm` | never computed | Declared at `:77`, assigned nowhere; the per-paw versions in `extended` exist instead |
| `grasp_aperture_max_mm` | never computed | Declared at `:66`, assigned nowhere; `paw_width_proxy_max_*` exists instead |
| `grasp_aperture_at_contact_mm` | never computed | Declared at `:67`, assigned nowhere; the replacement is also null for want of an interaction frame |
| `distance_to_interaction` | never computed | **Mis-classified — this is lost in transit.** The older outcome detector computes it (`outcomes/core/pellet_outcome.py:1645-1689`) into a `reach_features` dictionary that `find_causal_reach` returns and the caller never writes onto the record it builds at `:1591`. The extractor's `outcome_data.get('reach_features', {})` therefore always returns empty (`:387`). |
| `max_extent_pixels`, `max_extent_ruler` | never computed | **Half right.** The current reach detector deliberately writes `None` (`reach/core/span_to_reaches.py:179-180`), and the audit's own corpus is current-stage-only, so 0% is correct there. In the wider table these columns are not empty — parts of the Processing corpus and of the Analyzed sample carry legacy values. Do not read "never computed" as "the column is empty everywhere". |
| `max_extent_mm` | never computed | Derived from `max_extent_ruler`, so it follows it exactly (`:392-393`) |
| `algo_causal_reach_id` | never computed | Copied from the segment (`:377`); the segment value preserves the algorithm's causal reach id, which the current outcome detector never emits |
| `flagged_for_review` (reach level) | lost in transit | Outcome detection flags *segments*; the extractor keeps that as segment `outcome_flagged` and never copies anything onto the reach field at `:86` |
| `flag_reason` (reach level) | lost in transit | Same; `SegmentFeatures` has no `flag_reason` field at all, so the reason string is dropped at the segment level too |
| `segment_num` (features 0%) | lost in transit | The assignment at `:317` landed on 2026-08-20 in commit `5bac3b0`; every audited file predates it. It now reaches the features file on Processing reaches written after it, and reaches the database on none (§8) |
| `outcome_known_frame` (filled at the outcome stage, absent from features) | lost in transit | The extractor reads `interaction_frame` and `causal_reach_id` from the outcome record and nothing else |
| `is_causal`, `label`, `segment_ifr`, `triage_reason` (from reach assignment) | lost in transit | `{video}_reach_assignments.json` is never opened by the kinematics package (§2) |
| `extended`, `reach_source` | not in the database | `extended` is in fact stored, as the `extended_features` JSON blob (`sync/database.py:595`, which the audit separately reports as filled); the audit is looking for a column named `extended`, which does not exist. `reach_source` genuinely has no column. |

The audit's small `causal_reach` and `interaction_frame` fill rates come from legacy files, human reviews and ground truth only. Nothing the current algorithm produces contributes to either (§2).

---

## 13. Documentation in this area that is wrong

* `REACH_KINEMATIC_DATA_DICTIONARY.md:89` defines `distance_to_interaction` as "`interaction_frame - apex_frame`, in frames". Nothing computes that; the value is read from a dictionary that is never written (§2).
* `REACH_KINEMATIC_DATA_DICTIONARY.md:39-43` says tracking-quality gating is unnecessary because "numerically robust accumulation (`np.nansum`, `np.nanmean`, etc.) tolerates occasional NaN frames within a tracked reach". There are no NaN frames — DeepLabCut always emits a coordinate (whole videos checked, zero nulls). The conclusion happens to be defensible; the stated reason is not, and it obscures the real consequence, which is that low-confidence frames enter every trajectory measurement at full weight.
* `feature_extractor.py:361` still initialises `segment_num=0` with the comment "Will be set by caller". As of 2026-08-20 that is true — `:317` sets it. The comment is now merely stale rather than wrong.
* `sync/database.py:56-60` says the extractor "leaves it 0 on every reach". That stopped being true on 2026-08-20, and the change made on the strength of that comment is what broke the database insert (§8).

---

## Contested claims

This document was written from the source, then checked by a second reader
whose job was to disprove it. The statements below are ones they disputed and
that were not resolved. Neither side is authoritative: spot-checking found the
checker wrong at least once. **Do not rely on anything listed here without
opening the code yourself.** Everything not listed survived two passes.

- **"`ReachDetector.detect` — the v6 algorithm — has **no caller anywhere in `src/`**. Its only caller was a one-off diagnostic script."**
  - disputed because: Both halves are false. There ARE two call sites in src/ (both unreachable, but present), and the named script is not one of the callers, while a different file is.
- **"`spatial_refiner.py` (1008 lines) has **no importer at all**"**
  - disputed because: It has an importer.
- **"Also dead: `boundary_refiner.py` and `boundary_polisher.py`, whose only importer is the dead `reach_detector.py` (`:101-102`)."**
  - disputed because: True for boundary_polisher.py, false for boundary_refiner.py, which has two other importers.
- **"[v8/eval.py] is called by `v8/train.py:123`."**
  - disputed because: v8_figures.py does not call or import v8/eval.py. It consumes eval output that has already been serialised to plain dicts.
- **"In a typical file the first boundary comes some way into the recording, so the lead-in frames before it are outside every segment."**
  - disputed because: The size of that region was stated wrongly (by more than an order of magnitude). The structural point (frames before the first boundary belong to no segment) is right.
- **"Measured: a small number of shipped files contain a bare `NaN`, always in `ruler_pixels`. Most are the trailing segment 21, length 0 or 1, with no reaches. A few are segment 1 with length 0, flagged, holding orphan reaches."**
  - disputed because: The cases split differently: not all of the no-reach cases are the trailing segment 21. Some are mid-video segments, which the document's framing ("trailing segment 21" vs "segment 1") implicitly rules out. This matters because mid-video NaN segments fall inside segments 1-20, the range kinematics actually reads, so their segment-level millimetre features are NaN too.
- **"`review_widget.py:2125-2126` reads `original_start_frame` / `original_end_frame`, but nothing anywhere writes those keys"**
  - disputed because: Another tool writes exactly those keys, and writes them into the same `{video}_reaches.json` file. The conclusion (the two GT-export fields come out null) still holds for reaches produced and edited by this widget, but the stated reason is wrong and the omission hides a second writer of the reaches file.
- **"`boxr_x` ... **Nothing reads it.** No code anywhere loads `boxr_x` back out of this file"**
  - disputed because: Stated as an absolute, and one place does read it. The substantive point survives (that reader is itself never called), but as written the sentence is contradicted by a one-line grep — exactly what a reference document cannot afford.
- citation could not be resolved: ``reach_detector_v8.py:20-52`, cited as "its performance table" — does not resolve. Lines 20-52 are the Phase-2 algorithm description and the KEY PARAMETERS tabl`

---

## Update 2026-08-23: the algorithm's causal reach now reaches the data

Extractor 2.1.0. The extractor loads `{video}_reach_assignments.json`
(auto-discovered beside the outcomes file; `assignments_path` overrides) and
marks `causal_reach`, `outcome` and `interaction_frame` on the reach algo-4
credited -- for segments no person has decided. (Corrected later, same
extractor version: the rule written here first -- "a segment carrying a
`causal_reach_id` KEY is owned by human review, ground truth or the pre-v6
detector" -- let the legacy detector's guess, which carries that key on every
segment, block the assignment's answer. See "Who owns a causal reach" below.)
A reviewer's `causal_reach_id: None` means "no reach did this" and stands. On the
algo-only test video the features now name exactly the reaches the assignment
file names; on a fully human-reviewed video all causal marks are
`human_review`-sourced and none land on human-untouched segments.

Older reviews that stored the human's causal pick as frames only (no reach id)
are also recovered now: the resolver matches the frames to the reach they name
(>=50% overlap, within the reviewed segment). On the test video every pick
resolves except hand-drawn reaches that exist in no reaches file and so have no
row to mark.

`distance_to_interaction` remains never-computed -- its definition is unclear
and inventing one was worse than leaving it empty. Corpus backfill and the
2.1.0 declaration in pipeline_versions.json are deliberately deferred; until
declared, newly processed videos will read as version-stale.

---

## Who owns a causal reach (the assignment merge)

The extractor puts the reach assignment's causal reach (algo-4,
`{video}_reach_assignments.json`) on every touched segment that no PERSON has
decided. Whether a person decided a segment is read from what the truth
layering stamped on it (`causal_reach_owned_by_person` in
`kinematics/core/feature_extractor.py`), never from whether the outcome file
merely holds a `causal_reach_id` key:

| The segment | Its causal reach in the features file |
|---|---|
| A review answered it (`outcome_source` = `human_review`) | The reviewer's pick. An untouched answer, or a touched answer with no pick, stays without a causal reach. |
| Ground truth decided it and the pellet was untouched | No causal reach. |
| Ground truth decided it, pellet touched, and a reach was found from a person's statement (`causal_reach_source` = `ground_truth`: the reach holding the ground truth's interaction frame; or `human_review`: the review's pick for the same outcome) | That reach. |
| Ground truth decided it, pellet touched, but no reach holds its interaction frame and no review named one | The reach assignment's pick. |
| An older file marks the segment resolved by a person in place (`triage_cleared` or `human_verified`) | The reach that file names. |
| Anything else -- including a legacy outcome file that names a causal reach on every segment | The reach assignment's pick, when it has one; otherwise whatever the outcome file says. |

If the reach a person excluded in the ground-truth tool is the segment's causal reach, the segment
keeps `causal_reach_id` naming a reach that is now only in `reaches_left_out`, and no measured
reach is marked causal. A reviewer's ignore window does the same. Nothing is substituted.

On every person-owned segment the assignment's pick is still recorded as
`algo_causal_reach_id`, so an override stays visible as one. No extractor version
change: the result differs only for a video whose legacy outcome file sits
beside a reach assignment file, and such a video is re-run by hand if one is
found.
