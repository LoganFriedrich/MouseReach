# Kinematics: every value MouseReach measures from a reach

Describes: `src/mousereach/kinematics/core/feature_extractor.py`, `src/mousereach/kinematics/cli.py`, `src/mousereach/kinematics/analysis/reach_export.py`, `src/mousereach/export/features_csv.py`, `src/mousereach/sync/database.py`, `REACH_KINEMATIC_DATA_DICTIONARY.md`

Verified against: 61d98b9 (2026-08-21)

---

## 1. What this subsystem is

One class, `FeatureExtractor` (`feature_extractor.py:169`), turns three files into one file.

**Reads**

| Input | Produced by | What is used from it |
|---|---|---|
| `{video}DLC*.h5` | DeepLabCut | Every body-part x, y and likelihood, one row per video frame |
| `{video}_reaches.json` | reach detection | Segment list, per-segment `ruler_pixels`, and every reach's `reach_id`, `reach_num`, `start_frame`, `apex_frame`, `end_frame`, `duration_frames` |
| `{video}_pellet_outcomes.json` | outcome detection | Per-segment `outcome`, `confidence`, `flagged_for_review`, `interaction_frame`, `causal_reach_id` |

**Writes** `{video}_features.json`: a video record containing segment records, each containing reach records. The whole thing is produced by `dataclasses.asdict` (`feature_extractor.py:165`), so every declared field appears in the file whether or not anything filled it in.

**Also reads, if present:** a human review or ground-truth file, through `resolve_truth_layers` (`feature_extractor.py:218-247`). This can replace a segment's outcome and can replace the reach list itself. See §9.

**Run by**: `mousereach-grasp-analyze` (`kinematics/cli.py:16`), the watcher (`watcher/orchestrator.py:1148` and `:2095`), `pipeline/run_all.py:118`, and `pipeline/reprocess_to_current.py:256`.

**Sizes.** Per reach the extractor emits 43 named fields plus a nested dictionary called `extended` holding a further 161 named values — 204 numbers per reach. Per segment it emits 25 fields plus the reach list. The 161 was confirmed by running `_extended_features` and by reading a shipped file; both give 161.

---

## 2. Read this before reading the tables: outcome linkage is empty in all current output

Every field that connects a reach to what happened to the pellet is null in everything the pipeline produces today. This is not a schema question — the code is fine; the input it needs stopped arriving.

The extractor decides which reach caused the outcome with one comparison (`feature_extractor.py:381-383`):

```python
causal_reach_id = outcome_data.get('causal_reach_id')
if causal_reach_id is not None and causal_reach_id == reach_id:
    features.causal_reach = True
```

The outcome detector that runs today is `v6_cascade` version 6.1.0, and it does not write `causal_reach_id`. Measured over every video in the local Processing directory that has both files:

| Outcome detector that produced the file | Videos | Any segment carrying `causal_reach_id`? |
|---|---|---|
| `v6_cascade` 6.1.0 | 891 | 0 of 891 |
| unnamed detector, version 2.4.4 (older) | 199 | 135 of 199 |

Across 300 current videos: 4,340 segments, **zero** with a causal reach id, though 694 of those segments are scored `retrieved` and 2,369 `displaced_sa`. 32,887 reaches, **zero** marked causal.

The causal link still exists — it moved to a file the extractor never opens. `{video}_reach_assignments.json` (detector `assignment_v2` 2.1.0, 892 files present) carries per reach: `is_causal`, `label`, `segment_outcome`, `segment_ifr`, `segment_num`. Nothing in `src/mousereach/kinematics/` references it (grep for "assign" in that package returns one unrelated comment).

**Fields that are consequently null on every reach in current output:** `causal_reach` (false), `outcome`, `interaction_frame`, `distance_to_interaction`, and the ten extended `*_at_contact_*` columns. `SegmentFeatures.causal_reach_id` is null on every segment. The video summary's `causal_reaches` count is 0.

---

## 3. Units: how a pixel becomes a millimetre, and where the seconds come from

**Space.** Every segment has `ruler_pixels`: the pixel distance between the two lower corners of the sample area, `SABL` and `SABR`, taken as a median over a stable stretch of the segment (`reach/core/geometry.py:104-113`). That distance is declared to be 9.0 mm (`feature_extractor.py:177`, `RULER_MM = 9.0`). So

```
mm = pixels * 9.0 / ruler_pixels
mm² = pixels² * (9.0 / ruler_pixels)²
```

Measured over 4,000 segments: median `ruler_pixels` = 36.0, so about 0.25 mm per pixel; 5th-95th percentile 33.6-37.9. Nothing sanity-checks this number. Over 8,000 segments, 6 fell outside 25-50 px, including values of 3.7 and 230.2 — a segment with `ruler_pixels = 3.7` has every millimetre value in it inflated roughly tenfold, silently. If `ruler_pixels` is 0 or negative the conversion factor becomes 0.0 and every millimetre column in that segment reads exactly 0 rather than null (`feature_extractor.py:1043`, `:1239`).

**Time.** Frame counts are exact integers. Anything in seconds divides by `FRAMERATE = 30.0` (`feature_extractor.py:176`), which is a constant, not a measurement — the comment on that line says so, and the data dictionary says framerate "is not a measured quantity in this pipeline". Only five values use it: `velocity_at_apex_mm_per_sec`, `segment_duration_sec`, `time_to_first_reach_sec`, `time_to_outcome_sec`, `mean_inter_reach_interval_sec`. Everything in the `extended` block is per *frame*, never per second.

**Angles.** Degrees, from `atan2`, so in the range -180 to +180.

**Likelihood.** DeepLabCut's own confidence, 0 to 1, dimensionless.

**Frame windows are inclusive.** A reach's rows are `df.iloc[start_frame:end_frame+1]`, so `duration_frames` frames are used.

---

## 4. Which direction is "out"? The file disagrees with itself

This matters because it decides what "how far did the mouse reach" means.

**Measured from the tracking data** (two videos, one ASPA and one CNT, medians in image pixels):

| Landmark | x | y |
|---|---|---|
| BOXL (left edge of slit) | 160 | 411 |
| BOXR (right edge of slit) | 179 | 409 |
| Pillar (where the pellet sits) | 167 | 433 |
| SA base (SABL/SABR) | 153 / 190 | 470 |

The slit runs along **x**. The pellet sits at larger **y** than the slit. During reaches the paw goes from y ≈ 400 to y ≈ 443 while x moves less than 10 px. So **image y is the extension axis and image x is the sideways axis.**

The two halves of this file take opposite views:

* The `extended` block treats +y as extension: `extension_past_nose` is `paw_y - nose_y` (`feature_extractor.py:1063-1067`). This is correct.
* `max_extent_pixels` is inherited from reach detection, where it is `max_hand_x - BOXR_x` (`reach/core/reach_detector_v8.py:307`, `reach_detector.py:749`) — a sideways offset. The extractor multiplies it straight into millimetres (`feature_extractor.py:392-393`).
* `_compute_pellet_positioning` labels the y difference "lateral" and the x difference "depth" (`feature_extractor.py:922-931`) — backwards on both.

Consequence, measured over the 201,446 database rows that have a value for `max_extent_pixels`: median **-0.7 px**, mean -0.56 px, 54% of values negative, minimum -24.3 px. In millimetres the column runs from -5.98 mm upward. A reach extent that is negative more than half the time is not measuring extension.

Same consequence for the reach's primary `apex_frame`, which the reach detector defines as the frame of maximum hand *x* (`reach_detector_v8.py:367-378`). Every field below whose name ends `_at_apex` is sampled at that frame. Inside the `extended` block each paw point additionally computes its own apex as maximum nose-relative *y* — so one reach record carries two apexes defined on two different axes.

---

## 5. Per-reach fields (43, outside `extended`)

"Filled today" is judged from the code plus measurement on shipped `2.0.0` files and the 361,587-row database snapshot.

### 5.1 Identity and position in the segment

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `reach_id` | Unique number for this reach in this video | Copied from reach detection (`:352`) | count | yes |
| `reach_num` | Which reach this is inside its segment, starting at 1 | Copied from reach detection | count | yes |
| `segment_num` | Which pellet presentation this reach belongs to | Set from the enclosing segment (`:317`) | count | **fixed 2026-08-20, not yet in any data.** Added by commit 5bac3b0 on 2026-08-20. Every `_features.json` written before that has `0` on every reach, which is what a shipped file still shows. It also does not reach the database — see §8. |
| `is_first_reach` | Is this the first reach of the segment | `i == 0` (`:318`) | true/false | yes |
| `is_last_reach` | Is this the last reach of the segment | `i == n-1` (`:319`) | true/false | yes |
| `n_reaches_in_segment` | How many reaches in this segment | `len(seg['reaches'])` (`:320`) | count | yes |

### 5.2 Outcome linkage — all null today (§2)

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `causal_reach` | Is this the reach that decided the pellet's fate | `causal_reach_id == reach_id` (`:381-383`) | true/false | **false on every reach** — the current outcome detector emits no causal reach id |
| `outcome` | What happened to the pellet | Copied from the segment, **only onto the causal reach** (`:384`) | text | **null on every reach** |
| `interaction_frame` | Frame the paw met the pellet | Copied from the segment, only onto the causal reach (`:385`) | frame index | **null on every reach** |
| `distance_to_interaction` | Frames between apex and the interaction | Read from `outcome_data['reach_features']` (`:387-389`) | frames | **never, for a second independent reason.** The outcome detector computes this dictionary (`outcomes/core/pellet_outcome.py:1645-1689`) and then drops it: `find_causal_reach` returns it, the caller assigns it to a local variable at `:1572`, and it is never placed on the `PelletOutcome` that gets written. So the key `reach_features` does not exist in any outcomes file, and `.get(..., {})` returns empty every time. Even if causal reaches came back, this would stay null. |

### 5.3 Extent — inherited, wrong axis, and null in current output

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `max_extent_pixels` | Intended: how far the paw reached | Copied from reach detection (`:366`) | pixels | **null.** The detector in use (`reach/core/span_to_reaches.py:179-180`) writes `None` on purpose: "Extent is not cheaply available from the flat detector". Older detectors did fill it; those values measure the sideways axis (§4) and are negative 54% of the time. |
| `max_extent_ruler` | Same, divided by the 9 mm ruler | Copied from reach detection (`:367`) | ruler lengths (1.0 = 9 mm) | null, same reason |
| `max_extent_mm` | Same, in millimetres | `max_extent_ruler * 9.0` (`:392-393`) | mm | null whenever the source is null |

### 5.4 Speed and path of the wrist point

All four use only the `RightHand` point, and none of them gate on likelihood.

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `peak_velocity_px_per_frame` | Fastest single-frame move of the wrist during the reach | max of frame-to-frame straight-line distance (`:474`) | pixels per frame | yes, 100% |
| `mean_velocity_px_per_frame` | Average per-frame move of the wrist | mean of the same (`:475`) | pixels per frame | yes, 100%. Database median 3.86, max 136. |
| `velocity_at_apex_px_per_frame` | Wrist speed at the apex | The step *starting* at the apex: `velocity[apex_frame - start_frame]` (`:469-471`) | pixels per frame | yes, ~94%. Null when the reach has no apex, or when `apex_frame` is exactly 0 — `apex_frame - start_frame if apex_frame else None` (`:397`) treats frame 0 as missing. |
| `velocity_at_apex_mm_per_sec` | The same converted | `× 9.0/ruler_pixels × 30.0` (`:402-405`) | mm per second, assuming 30 fps | yes, ~94%. Database median 2.7 mm/s, 75th percentile 69 mm/s, max 3,774 mm/s — a distribution worth checking before use. |

Note on the apex index: `_compute_velocity_features` removes missing values from the speed series *before* indexing into it (`:460` then `:471`), so a gap earlier in the reach would shift which frame "at apex" refers to. In practice this never fires — the DeepLabCut `.h5` files contain no missing coordinates at all (checked on two full videos: zero nulls in `RightHand_x`, `Nose_x`, `Pellet_x`, `RHLeft_x`). That same fact means there is no gating anywhere: a frame where DeepLabCut was 5% confident contributes its coordinate at full weight to every path length, speed and area below.

### 5.5 Path shape of the wrist point

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `trajectory_straightness` | 1.0 = the wrist went in a straight line; near 0 = it wandered | first-to-last straight-line distance ÷ total path length (`:497-503`) | dimensionless 0-1 | yes, 100%. Database median 0.22. |
| `trajectory_smoothness` | 1.0 = perfectly smooth; smaller = jerkier | `1 / (1 + mean|third difference of position|)` (`:519`) | dimensionless 0-1 | yes, ~98.5%. Null for reaches under 3 frames. |

### 5.6 Paw orientation

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `hand_angle_at_apex_deg` | Which way the paw was pointing at apex | `atan2(RHRight_y - RHLeft_y, RHRight_x - RHLeft_x)` at the apex (`:534-539`) | degrees, -180 to 180 | yes, ~99% |
| `hand_rotation_total_deg` | Total turning of the paw across the reach | Sum of absolute frame-to-frame angle changes, with wraparound handled (`:554-559`) | degrees, always ≥ 0 | yes, 100% |

### 5.7 Head and body at the apex

Each needs both landmarks above likelihood 0.7 at that frame, otherwise null.

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `head_width_at_apex_mm` | Ear-to-ear distance at apex; a body-size proxy | distance(LeftEar, RightEar) × mm-per-pixel (`:614-620`) | mm | yes, ~99% |
| `nose_to_slit_at_apex_mm` | Nose to the right edge of the slit at apex | distance(Nose, BOXR) × mm-per-pixel (`:627-633`) | mm | yes, ~99.7%. Named "slit" but measured to one corner of it. |
| `head_angle_at_apex_deg` | Head orientation at apex | ear-to-ear angle (`:640-644`) | degrees, -180 to 180 | yes, ~99% |
| `head_angle_change_deg` | How much the head turned between reach start and apex | apex angle − start angle, wrapped into -180..180 (`:647-659`) | degrees | yes, ~97% |

### 5.8 Tracking quality of the wrist point

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `mean_likelihood` | Average DeepLabCut confidence in the wrist across the reach | mean of `RightHand_likelihood` (`:577`) | 0-1 | yes, 100% |
| `frames_low_confidence` | Frames where wrist confidence was under 0.5 | count (`:578`) | frames | present on every reach but zero on ~97.5% of them |
| `tracking_quality_score` | — | **nothing computes it.** Declared at `:83`, read by three exporters and a database column, assigned nowhere in the repository. | — | **never** |

### 5.9 Flags

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `flagged_for_review` | Does this reach need a human | **nothing sets it.** Declared `False` at `:86`, never assigned. | true/false | **always false.** The outcome detector does flag *segments* (11.9% carry a reason), and the extractor keeps that at segment level as `outcome_flagged`, but the per-reach copy is dead. |
| `flag_reason` | Why | Never assigned (`:87`) | text | **always null** |

### 5.10 Where the answer came from

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `outcome_source` | `algo`, `human_review`, or `ground_truth` for this segment's outcome | Copied from the segment after truth layering (`:373`) | text | yes, on every reach |
| `reach_source` | `algo` or `ground_truth` for this reach's frame boundaries | `reach.get('reach_source', 'algo')` (`:374`) | text | yes — but no database column holds it (§8) |
| `reviewed_by` | Who reviewed | Copied from the segment (`:375`) | text | ~20% |
| `algo_outcome` | What the algorithm said before a human overrode it | Copied from the segment (`:376`) | text | ~20% |
| `algo_causal_reach_id` | Which reach the algorithm blamed before a human overrode it | Copied from the segment (`:377`); the segment value is stamped by `review/truth_resolver.py:173` from whatever `causal_reach_id` was there | count | **never varies across 1,377 videos.** The value it copies is the algorithm's causal reach id, and the current detector never sets one (§2). |

### 5.11 Fields declared and never computed by anything

Four more that the audit lists, all with the same story: a name in the dataclass, database columns, exporter lines that read them, and no code anywhere that assigns them.

| Field | Declared | Intended meaning | Replacement that does exist |
|---|---|---|---|
| `grasp_aperture_max_mm` | `:66` | Widest paw opening during the reach | `paw_width_proxy_max_mm` in `extended` |
| `grasp_aperture_at_contact_mm` | `:67` | Paw opening when it met the pellet | `paw_width_proxy_at_contact_mm` (itself null, §2) |
| `apex_distance_to_pellet_mm` | `:76` | How close the paw got to the pellet | none |
| `lateral_deviation_mm` | `:77` | How far sideways the paw strayed | `righthand_lateral_deviation_mm` and the three other paw points |

---

## 6. The `extended` block: 161 values per reach

Produced by `_extended_features` (`feature_extractor.py:1425`) and stored as one nested dictionary. If any part of it raises, the whole block is replaced by a single key `_extended_features_error` holding the message (`:443-444`) — the reach still ships, with 161 values missing and no warning anywhere else. Measured across 3,748 reaches in 60 videos: zero such errors.

### 6.1 The coordinate frame these use

Two things are done before any trajectory measurement (`_build_augmented_nose_relative_trajectory`, `:998`):

1. **Nose subtraction.** Every paw position becomes `paw − nose` at the same frame, so head movement does not count as paw movement.
2. **Two synthetic anchor frames** are added, one before the reach and one after:
   * before: the midpoint of Nose and BOXL at `start_frame`, i.e. `(BOXL − Nose)/2` in nose-relative terms
   * after: the midpoint of Nose and BOXR at `end_frame`
   These are geometry, not tracking, so the reach always starts and ends at a defined place instead of wherever DeepLabCut first noticed the paw. Path length, swept area, speeds, directness and lateral deviation all include them. Extension, the per-point apex, and everything in §6.3 and §6.4 use real frames only.

### 6.2 Per paw point — four points × 19 values = 76

The four points are `RightHand`, `RHLeft`, `RHRight`, `RHOut` (`:990`), written in column names as `righthand`, `rhleft`, `rhright`, `rhout`. The data dictionary is explicit that the anatomical meaning of these labels is not validated; treat them as four stable points on the paw outline.

| Column suffix | Plain English | How computed | Units |
|---|---|---|---|
| `_apex_frame` | The frame this point reached furthest out | `argmax(paw_y − nose_y)` over real frames (`:1058-1061`) | frame index |
| `_extension_past_nose_px` / `_mm` | How far past the nose this point got, along the out-through-the-slit direction | that maximum value (`:1066-1067`) | pixels / mm |
| `_total_path_px` / `_mm` | Total distance travelled, following every wiggle | sum of frame-to-frame distances on the anchored path (`:1073-1075`) | pixels / mm |
| `_lateral_spread_px` / `_mm` | Side-to-side range covered | `max(x) − min(x)` (`:1079-1084`) | pixels / mm |
| `_swept_area_px2` / `_mm2` | Area enclosed by the loop the paw traced | shoelace polygon over the path (`:1088-1096`) | pixels² / mm² |
| `_path_directness` | 1.0 = straight from entry anchor to exit anchor; smaller = more winding | anchor-to-anchor distance ÷ total path (`:1099-1103`) | dimensionless |
| `_motion_smoothness` | 1.0 = smoothest; smaller = jerkier | `1/(1 + mean|third difference|)` (`:1106-1115`) | dimensionless |
| `_lateral_deviation_px` / `_mm` | Furthest the path strayed from the straight anchor-to-anchor line | max perpendicular distance (`:1119-1131`) | pixels / mm |
| `_mean_speed_px_per_frame` / `_mm_per_frame` | Average speed | total path ÷ number of steps (`:1134-1141`) | pixels or mm per frame |
| `_peak_speed_px_per_frame` / `_mm_per_frame` | Fastest single step | max step length | pixels or mm per frame |
| `_apex_speed_px_per_frame` / `_mm_per_frame` | Speed at this point's own apex | the step just after the apex frame, or the one just before if the apex is the last frame (`:1145-1157`) | pixels or mm per frame |

All 76 are populated on essentially every reach: measured over 32,887 reaches in 300 videos, every one of these is non-null on more than 99.9%.

### 6.3 Coordination between the four points — 3 values

| Column | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `paw_apex_lead_frames` | How spread out in time the four points' apexes were | latest apex − earliest apex (`:1176`) | frames | yes |
| `paw_leading_point` | Which point apexed first | earliest apex, as a name (`:1178`) | text: one of the four | yes |
| `paw_velocity_correlation` | How much the four points sped up and slowed down together | mean of the six pairwise Pearson correlations of the frame-to-frame speed series (`:1195-1205`) | dimensionless, -1 to 1 | yes |

### 6.4 Paw shape — 3 measures × 8 time points × 2 units = 48

Computed per frame on real frames only, in raw pixel coordinates (no nose subtraction), then summarised (`_paw_shape_features`, `:1213`).

| Measure | Column root | Per-frame calculation | Units |
|---|---|---|---|
| Paw width | `paw_width_proxy` | distance from `RHLeft` to `RHRight` (`:1250-1253`) | px / mm |
| Paw outline area | `paw_outline_area` | shoelace area of all four points, ordered each frame by angle around their centre so rotation does not break the polygon (`:1263-1281`) | px² / mm² |
| Paw spread | `paw_spread_max` | largest of the six pairwise distances (`:1256-1261`) | px / mm |

Eight summaries each (`:1288-1323`): `_at_start` (first reach frame), `_at_apex` (the reach's primary apex — the wrist's, see §4), `_at_contact` (the interaction frame), `_at_end` (last reach frame), `_max`, `_min`, `_mean`, `_range` (max − min).

All are populated except the six `_at_contact_*` columns, which are **null on every reach today** because there is no interaction frame (§2).

### 6.5 Visibility of each paw point — 4 points × 8 + 1 = 33

`_paw_visibility_features` (`:1334`). Reports DeepLabCut likelihood rather than filtering on it, on the reasoning that a point disappearing often means the paw rotated or closed.

| Column | Meaning | Units |
|---|---|---|
| `{point}_visibility_at_start` / `_at_apex` / `_at_end` | Likelihood at those frames | 0-1 |
| `{point}_visibility_at_contact` | Likelihood at the interaction frame | 0-1 — **null on every reach today** (4 columns) |
| `{point}_visibility_max` / `_min` / `_mean` / `_range` | Across the reach | 0-1 |
| `frames_any_paw_low_confidence` | Frames where the *worst* of the four points was under 0.5 (`:1391`) | frames |

Measured over 3,748 reaches: 42 of them contain any such frame, 42 frames out of 61,430 reach frames — 0.1%. Within detected reaches, paw tracking confidence is essentially always high.

### 6.6 Tray contact — 1 value

| Column | Meaning | How computed | Units |
|---|---|---|---|
| `tray_contact_duration_frames` | Frames where the pellet was jiggling while the wrist was hidden — a proxy for the paw being on the tray | count of frames where the 4-frame rolling standard deviation of `Pellet_x` exceeds 2 px **and** `Pillar_likelihood < 0.5` **and** `RightHand_likelihood < 0.5` (`:1414-1420`) | frames |

Returns 0 for reaches shorter than 4 frames, and null if any of those three columns is missing from the tracking file. The 2-pixel threshold is a raw pixel constant, not scaled by `ruler_pixels`, so it means a different physical distance on differently-zoomed videos.

---

## 7. Per-segment fields (25)

One record per pellet presentation. All are written to `{video}_features.json`; only five of them reach the database (§8).

| Field | Plain English | How computed | Units | Notes |
|---|---|---|---|---|
| `segment_num` | Which pellet presentation | copied | count | |
| `start_frame`, `end_frame` | Segment bounds | copied | frame index | |
| `ruler_pixels` | The 9 mm reference in pixels for this segment | copied from reach detection | pixels | drives every mm conversion in the segment |
| `outcome` | What happened to the pellet | copied from outcome detection (`:293`) | text | |
| `outcome_confidence` | Detector's confidence | copied (`:294`) | 0-1 | in the database this varies on only 0.5% of rows |
| `outcome_flagged` | Detector asked for human review | copied (`:295`) | true/false | ~9.6% |
| `n_reaches`, `causal_reach_id` | Reach count; which reach caused the outcome | copied (`:297`, `:299`) | count | `causal_reach_id` null on every current segment (§2) |
| `attention_score` | Share of frames the mouse was at the tray | frames where `Nose_likelihood > 0.9` and `Nose_y > BOXR_y − 80`, ÷ total × 100 (`:798-806`) | percent 0-100 | The 80 is raw pixels, not scaled by `ruler_pixels`. Database median 60.7. |
| `mean_head_width_mm` | Average ear-to-ear distance | mean over frames with both ears above 0.7 (`:696-704`) | mm | |
| `mean_nose_to_slit_mm` | Average nose-to-BOXR distance | same gate (`:710-718`) | mm | |
| `mean_nose_height` | Average nose y position | plain mean of `Nose_y` (`:726`) | **raw image pixels, not converted, not referenced to anything** | comparable only within one camera setup |
| `mean_head_angle_deg` | Average head orientation | plain arithmetic mean of ear-to-ear angles (`:737`) | degrees | **Not valid as written.** Measured angles span -160 to +177 degrees; averaging a wrap-around quantity arithmetically is meaningless when it straddles ±180. |
| `head_angle_variance` | Head-orientation steadiness | plain variance of the same angles (`:738`) | degrees² | Same problem, worse: measured median 7,949 (a spread of about 89 degrees), 94% of segments above 1,000. Dominated by the ±180 jump, not by head steadiness. |
| `nose_position_variance` | Postural steadiness | `var(Nose_x) + var(Nose_y)` (`:754`) | **pixels², not converted to mm²** | |
| `segment_duration_sec` | Length of the presentation | `(end − start + 1) / 30` (`:829`) | seconds at an assumed 30 fps | |
| `time_to_first_reach_sec` | Delay before the mouse first reached | `(first reach start − segment start) / 30` (`:834-836`) | seconds | null if the segment has no reaches |
| `time_to_outcome_sec` | Intended: when the outcome happened | **`= segment_duration_sec`, unconditionally** (`:844`) | seconds | Carries no information beyond segment length. The comment says "assume outcome happens at segment end". Note the outcome detector *does* compute a real answer, `outcome_known_frame`, present on 94.5% of segments — the extractor never reads it. |
| `mean_inter_reach_interval_sec` | Average gap between reaches | mean of (next start − previous end)/30 (`:848-857`) | seconds | null when fewer than 2 reaches; can be negative if reaches overlap |
| `pellet_lateral_offset_mm` | Labelled sideways offset of the pellet | `abs(pellet_y − BOXR_y) × mm-per-pixel` (`:924-925`) | mm | **Measures the out-through-the-slit direction, not sideways** (§4) |
| `pellet_depth_offset_mm` | Labelled depth offset of the pellet | `abs((pellet_x − BOXR_x) − 30) × mm-per-pixel` (`:929-932`) | mm | **Measures the sideways direction** (§4), and subtracts a hard-coded ideal of 30 raw pixels that belongs on the other axis |
| `pellet_position_idealness` | 0-1 score for how well placed the pellet was | `1 − (0.6 × min(lateral/50, 1) + 0.4 × min(depth_deviation/40, 1))`, floored at 0 (`:938-942`) | dimensionless 0-1 | Built from the two mislabelled axes above with three more raw-pixel constants (50, 40, 30). Measured over 359,811 rows: mean 0.371, standard deviation 0.074, never above 0.722 — consistent with the depth term sitting at full penalty nearly always. |
| `mean_tracking_quality` | Average confidence over every tracked point | mean of all `*_likelihood` columns in the segment (`:975`) | 0-1 | includes apparatus landmarks, not just the mouse |
| `tracking_dropout_frames` | Frames where any tracked point was under 0.5 | count (`:979-981`) | frames | |
| `reaches` | The reach records | | list | |

---

## 8. What survives the trip to the database and the CSVs

The extractor's file is complete. Everything downstream loses something.

**`{video}_features.json`** — everything above.

**`connectome.db`, table `reach_data`** (`sync/database.py`), one row per reach:

* Keeps the 43 per-reach fields, plus the entire `extended` dictionary as a JSON blob in one column, `extended_features` (`:595`).
* Keeps exactly five segment-level fields: `segment_outcome`, `segment_outcome_confidence`, `segment_outcome_flagged`, `attention_score`, `pellet_position_idealness` (`:564-572`).
* **Drops the other fourteen segment measurements entirely** — `mean_head_width_mm`, `mean_nose_to_slit_mm`, `mean_nose_height`, `mean_head_angle_deg`, `head_angle_variance`, `nose_position_variance`, `segment_duration_sec`, `time_to_first_reach_sec`, `time_to_outcome_sec`, `mean_inter_reach_interval_sec`, `pellet_lateral_offset_mm`, `pellet_depth_offset_mm`, `mean_tracking_quality`, `tracking_dropout_frames`. No column exists for any of them. They are computed on every video and read by nobody. They also never appear in `docs/FIELD_AUDIT.md`, because that tool inspects reach-level records in the features file and database columns, and these are neither.
* **Drops `reach_source`** — computed on every reach, no column.
* **`segment_num` is not written.** `ALL_COLUMNS` (`sync/database.py:83-92`) is built from `REACH_JSON_COLUMNS` plus four fixed lists, and `segment_num` is in none of them; the insert names exactly those columns (`:619-621`). The row dictionary does carry `segment_num` from the segment (`:568`), and nothing uses it. The table declares `segment_num INTEGER NOT NULL` with no default (`:107`). Commit 5bac3b0 removed the name from `REACH_JSON_COLUMNS` and did not add it back to `ALL_COLUMNS`. I could not read the live database to confirm the runtime behaviour — it was locked by a running watcher, exactly as `field_audit.py` warns — and the local parquet snapshot predates that commit by two hours, so it cannot show the effect either. What is certain from the code: the value the extractor now computes is not in the insert statement.

**`Databases/database_dump/reach_data.csv`** (`sync/database.py:757-783`) selects a fixed list that includes `segment_num` but **omits `extended_features`, `reach_source`, `outcome_source`, `reviewed_by`, `algo_outcome` and `algo_causal_reach_id`.** All 161 extended values and all review provenance are absent from the file that downstream `mousedb` recipes read.

**`mousereach-reach-export`** (`kinematics/analysis/reach_export.py`) writes `{video}_results.csv` and cohort CSVs. It carries all 161 extended values, but its fixed column list (`:24-58`) **omits every velocity and trajectory field**: `velocity_at_apex_px_per_frame`, `velocity_at_apex_mm_per_sec`, `peak_velocity_px_per_frame`, `mean_velocity_px_per_frame`, `trajectory_straightness`, `trajectory_smoothness`. The writer uses `extrasaction="ignore"` (`:216`), so they are dropped without a message.

**`mousereach-features-csv`** (`export/features_csv.py`) globs `*_grasp_features.json` (`:16`). Nothing in the pipeline writes a file with that name — kinematics writes `{video}_features.json`. The command always prints "No *_grasp_features.json files found!" and exits. Its column list is also built almost entirely from the never-computed fields of §5.11.

---

## 9. Human corrections, and one path that cannot fire

`extract` calls `resolve_truth_layers` (`feature_extractor.py:220-224`), which resolves each segment's outcome and each reach independently by ground truth > deep review > triage review > algorithm, and can replace the reach list itself when ground truth marks its reach set complete. Provenance lands in `outcome_source` and `reach_source`.

That resolver takes a **directory** as `primary_dir` and looks for `<dir>/<video>/<video>_causal_review.json` or `<dir>/<video>_causal_review.json` (`review/truth_resolver.py:72-88`). But every production caller passes a **file** path: the watcher passes the result of `resolve_review_path` (`orchestrator.py:1151`, `:2098`), and the CLI builds `Path(_rd) / f"{video_name}_causal_review.json"` (`kinematics/cli.py:120`). Both then arrive as `primary_dir`. The lookup builds nonsense paths, `is_file()` returns false, and that layer is silently skipped. Reviews stored in the two standard bundle directories still apply, because those are found independently. A review saved only in the processing directory does not.

If `resolve_truth_layers` raises, the code falls back to the older review-only path (`:238-246`), and if that raises too it prints a line and continues with the raw algorithm output.

---

## 10. Failures that produce a file anyway

| Failure | What happens | Where |
|---|---|---|
| Anything inside the extended feature block raises | The reach ships with `extended = {'_extended_features_error': message}` — 161 values missing, one key present | `feature_extractor.py:443-444` |
| Truth layering raises | Printed, then fall back to review-only, then to raw algorithm | `:227`, `:245` |
| Database sync fails from the CLI | `except Exception: pass` — no message at all | `kinematics/cli.py:139-141` |
| Database sync fails from the watcher | Logged as a warning; the video is still marked processed | `orchestrator.py:1168`, `:2131` |
| Feature extraction fails from the watcher's local pipeline | Logged as a warning, then `update_state(video_id, 'processed')` runs anyway | `orchestrator.py:1170-1173` |
| The reach file and the outcome file describe different segments | Nothing checks. They are paired by position: `zip(reaches_data['segments'], outcomes_data['segments'])` | `feature_extractor.py:253` |
| `ruler_pixels` is 0 or negative | Every millimetre value in the segment becomes exactly `0.0`, not null | `:1043`, `:1239` |

---

## 11. Reconciling with `docs/FIELD_AUDIT.md`

That document reports fill rates over 1,377 finished videos. Every kinematics-owned field it lists as empty, with the reason:

| Field | Audit bucket | Why it is empty |
|---|---|---|
| `tracking_quality_score` | never computed | Declared at `:83`, assigned nowhere in the repository |
| `apex_distance_to_pellet_mm` | never computed | Declared at `:76`, assigned nowhere |
| `lateral_deviation_mm` | never computed | Declared at `:77`, assigned nowhere; the per-paw versions in `extended` exist instead |
| `grasp_aperture_max_mm` | never computed | Declared at `:66`, assigned nowhere; `paw_width_proxy_max_*` exists instead |
| `grasp_aperture_at_contact_mm` | never computed | Declared at `:67`, assigned nowhere; the replacement is also null for want of an interaction frame |
| `distance_to_interaction` | never computed | **Mis-classified — this is lost in transit.** The outcome detector computes it (`outcomes/core/pellet_outcome.py:1652`, `:1676`) into a `reach_features` dictionary that `find_causal_reach` returns and the caller never writes to the outcome record. The extractor's `outcome_data.get('reach_features', {})` therefore always returns empty (`:387`) |
| `max_extent_pixels`, `max_extent_ruler` | never computed | The current reach detector deliberately writes `None` (`reach/core/span_to_reaches.py:179-180`) |
| `max_extent_mm` | never computed | Derived from `max_extent_ruler`, so null with it (`:392-393`) |
| `algo_causal_reach_id` | never computed | Copied from the segment (`:377`); the segment value is the algorithm's causal reach id, which the current outcome detector never emits |
| `flagged_for_review` (reach level) | lost in transit | Outcome detection flags *segments*; the extractor keeps that as segment `outcome_flagged` but never copies anything onto the reach field at `:86` |
| `flag_reason` (reach level) | lost in transit | Same; `SegmentFeatures` has no `flag_reason` field at all, so the reason string is dropped at the segment level too |
| `segment_num` (features 0%) | lost in transit | The assignment at `:317` landed on 2026-08-20 in commit 5bac3b0; every audited file predates it. It still does not reach the database (§8) |
| `outcome_known_frame` (94.5% at the outcome stage, absent from features) | lost in transit | The extractor reads `interaction_frame` and `causal_reach_id` from the outcome record and nothing else |
| `is_causal`, `label`, `segment_ifr`, `triage_reason` (from reach assignment) | lost in transit | `{video}_reach_assignments.json` is never opened by the kinematics package (§2) |
| `extended`, `reach_source` | not in the database | `extended` is in fact stored, as the `extended_features` JSON blob (`sync/database.py:595`); the audit reads the reach-level key name `extended`, which has no column of that name. `reach_source` genuinely has no column. |

The audit's `causal_reach` at 1.6% and `interaction_frame` at 0.7% are legacy rows only. In current output both are zero (§2).

---

## 12. Documentation in this area that is wrong

* `src/mousereach/kinematics/core/AGENTS.md` says velocity is "smoothed with Savitzky-Golay filter" and lists scipy as a dependency. The extractor imports only numpy and pandas, and applies no smoothing of any kind. The same file gives the wrong constructor and method names: it shows `FeatureExtractor(dlc_file, reaches_file, outcomes_file)` and `extract_all_features()`; the real interface is `FeatureExtractor()` then `.extract(dlc_path, reaches_path, outcomes_path, review_path=None)`.
* `REACH_KINEMATIC_DATA_DICTIONARY.md` defines `distance_to_interaction` as "`interaction_frame - apex_frame`, in frames". Nothing computes that; the value is read from a dictionary that is never written (§11).
* The same document says tracking quality gating is unnecessary because "numerically robust accumulation tolerates occasional NaN frames within a tracked reach". There are no NaN frames — DeepLabCut always emits a coordinate. Low-confidence frames enter every trajectory measurement at full weight.
* The docstring on `_extract_reach_features` and the field comment at `feature_extractor.py:23` say `segment_num` is "Will be set by caller" / set by the caller. It is now, as of 2026-08-20; before that it was not, and `sync/database.py:57` still carries a comment stating that the extractor leaves it 0 on every reach.
