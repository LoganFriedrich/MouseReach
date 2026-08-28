# Kinematics: every value MouseReach measures from a reach

Describes: `src/mousereach/kinematics/core/feature_extractor.py`, `src/mousereach/kinematics/cli.py`, `src/mousereach/kinematics/analysis/reach_export.py`, `src/mousereach/export/features_csv.py`, `src/mousereach/sync/database.py`, `src/mousereach/review/truth_resolver.py`, `REACH_KINEMATIC_DATA_DICTIONARY.md`

Verified against: b65fcf0 (2026-08-23)

---

## 0. How the numbers in this document were measured

Three sources, always named where used.

* **Processing corpus** — the 1,146 videos in `C:\LAB_ROOT\Behavior\MouseReach_Pipeline\Processing` that have a `_features.json`: 22,920 segments, 199,690 reaches. 947 of those videos were produced by the current detectors; 199 by older ones.
* **Analyzed archive** — a 150-file random sample of the 2,610 `_features.json` files under `Y:\...\MouseReach_Pipeline\Analyzed`: 3,000 segments, 21,953 reaches.
* **`docs/FIELD_AUDIT.md`** — the repository's own generated audit over 1,377 videos. Its database column comes from a parquet snapshot taken 2026-08-20 12:20, so it cannot show anything that changed after that.

Live `connectome.db` was read read-only where noted; it is frequently locked by a running watcher.

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

**Sizes.** `ReachFeatures` declares 44 fields: 43 named values plus a nested dictionary called `extended` holding a further 161 named values. `SegmentFeatures` declares 26: 25 values plus the reach list. Counted by parsing the dataclasses; the 161 confirmed by reading shipped files.

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

**The current outcome detector never writes `causal_reach_id`.** Every segment it emits carries exactly seven keys — `segment_num`, `outcome`, `outcome_known_frame`, `interaction_frame`, `stage`, `flagged_for_review`, and `flag_reason` when triaged. Measured over all 1,146 outcome files in the Processing corpus:

| Outcome detector | Videos | Segments | Segments with `causal_reach_id` | Segments with `confidence` |
|---|---|---|---|---|
| `v6_cascade` 6.1.0 | 947 | 18,940 | **0** | **0** |
| unnamed, version 2.4.4 (older) | 199 | 3,980 | 1,495 | 3,980 |

So `causal_reach = True` only ever arrives from somewhere other than the current algorithm. In the Analyzed-archive sample, the 906 causal reaches break down as: 739 in old pre-v6 files, 149 from a human causal review, 18 from a ground-truth file, and **zero** with `outcome_source = 'algo'`. Human reviews and ground truth put a causal reach id onto the segment through the truth-layering step (`review/truth_resolver.py:122` and `:145`), and the extractor then picks it up.

**Two things follow, and they are the largest holes in this data.**

1. **`interaction_frame` is thrown away 13,285 times.** The current outcome detector supplies a non-null `interaction_frame` on 13,285 of its 18,940 segments (70%). `SegmentFeatures` has no field for it, and the reach only receives it inside the `causal_reach_id` branch above. So it is discarded. Everything keyed to the moment of contact goes null with it: the reach's `interaction_frame`, and all ten extended `*_at_contact_*` columns. Across the Analyzed sample those ten are populated on 48 of 10,398 reaches (0.5%).

2. **`distance_to_interaction` can never be filled, for a second and independent reason.** The extractor reads it from `outcome_data['reach_features']` (`:387-389`). No outcomes file contains a `reach_features` key — 0 of 22,920 segments, at either detector version. In the older detector the dictionary is genuinely computed (`outcomes/core/pellet_outcome.py:1645-1689`), returned by `find_causal_reach`, assigned to a local variable at `:1572`, and then not passed into the `PelletOutcome` built at `:1591-1600`; the dataclass has no such field. So `.get('reach_features', {})` returns empty every time. Even if causal reaches came back, this stays null.

**The causal link is computed — by a different stage, into a file the extractor never opens.** `{video}_reach_assignments.json` (detector `assignment_v2` 2.1.0, 947 files present in Processing) carries per reach: `is_causal`, `label`, `segment_outcome`, `segment_ifr`, `segment_num`. In a 200-file sample, 2,592 of 30,050 reaches are marked `is_causal`. Nothing under `src/mousereach/kinematics/` references that file.

---

## 3. Units: how a pixel becomes a millimetre, and where the seconds come from

**Space.** Every segment carries `ruler_pixels`: the straight-line pixel distance between the median positions of the two lower corners of the sample area, `SABL` and `SABR`, taken over a stable middle stretch of the segment and, where at least 50 frames qualify, restricted to frames where both corners are tracked above likelihood 0.9 (`reach/core/geometry.py:97-113`, formula at `:68`). That distance is declared to be 9.0 mm (`feature_extractor.py:177`, `RULER_MM = 9.0`). So

```
mm  = pixels  * 9.0 / ruler_pixels
mm² = pixels² * (9.0 / ruler_pixels)²
```

Measured over all 22,920 segments of the Processing corpus: median `ruler_pixels` 34.5 (about 0.26 mm per pixel), 5th-95th percentile 34.3 to 38.4. Nothing sanity-checks the number. 15 segments fall outside 25-50 px, with a minimum of 2.3 and a maximum of 230.2. A segment with `ruler_pixels = 2.3` has every millimetre value in it inflated roughly fifteenfold, silently. If `ruler_pixels` is 0 or negative the conversion factor is set to `0.0` and every millimetre column in that segment reads exactly `0`, not null (`:606`, `:689`, `:884`, `:1048`, `:1246`).

**Time.** Frame counts are exact integers. Anything in seconds divides by `FRAMERATE = 30.0` (`:176`), which is a constant, not a measurement — the comment on that line says "assumed - not authoritative", and the data dictionary says framerate "is not a measured quantity in this pipeline". It is used at four places (`:407`, `:833`, `:838`, `:853`) producing five fields: `velocity_at_apex_mm_per_sec`, `segment_duration_sec`, `time_to_first_reach_sec`, `time_to_outcome_sec` (a copy of `segment_duration_sec`), `mean_inter_reach_interval_sec`. Everything in the `extended` block is per *frame*, never per second.

**Angles.** Degrees, from `atan2`, so in the range -180 to +180.

**Likelihood.** DeepLabCut's own confidence, 0 to 1, dimensionless.

**Frame windows are inclusive.** A reach's rows are `df.iloc[start_frame:end_frame+1]` (`:396`), so `duration_frames` frames are used. Algorithm-detected reaches set `duration_frames = end - start + 1` (`reach/core/span_to_reaches.py:162`), which matches. Reaches substituted from ground truth set `duration_frames = end - start` (`review/truth_resolver.py:204`), which is one frame short of the window actually measured.

---

## 4. Which direction is "out"? Parts of the file are labelled backwards

This matters because it decides what "how far did the mouse reach" means.

**Measured from the tracking data** (two videos, one ASPA and one CNT; medians in image pixels):

| Landmark | x | y |
|---|---|---|
| BOXL (left edge of slit) | 164 / 149 | 405 / 399 |
| BOXR (right edge of slit) | 181 / 165 | 403 / 396 |
| Pillar (where the pellet sits) | 171 / 157 | 424 / 420 |
| Pellet | 170 / 158 | 430 / 422 |
| SA base (SABL / SABR) | 153, 190 / 142, 176 | 465 / 455 |
| Nose | 185 / 191 | 315 / 281 |

The slit runs along **x**: BOXL and BOXR sit ~17 px apart in x at the same y. The pellet sits at larger **y** than the slit, and the nose at smaller y. So **image y is the extension axis and image x is the sideways axis.**

Three places in the code agree with that, and two do not.

* **Correct.** The `extended` block treats +y as extension: each paw point's apex is `argmax(paw_y − nose_y)` and its `extension_past_nose` is that maximum (`feature_extractor.py:1057-1069`), with a comment at `:1044-1045` stating the reasoning.
* **Correct.** The reach's primary `apex_frame` is the frame of maximum 2-D straight-line distance from nose to `RightHand` (`lib/causal_attribution.py:50-73`, bodypart constant at `:37`), set by the detector in use at `span_to_reaches.py:163`. It is a distance, not an axis, so it is not affected by this problem. (There is an old class, `ReachDetectorV8._find_apex`, that used maximum hand *x* — `reach_detector_v8.py:367-381`. Nothing imports it; `grep -rn "reach_detector_v8\|ReachDetectorV8" src/` returns only the file itself and one docstring mention.)
* **Wrong.** `max_extent_pixels` is inherited from reach detection. The current detector writes `None` (below). Older detectors computed `max_hand_x − BOXR_x` (`reach_detector.py:741`, `reach_detector_v8.py:307`) — a sideways offset. The extractor multiplies whatever it finds straight into millimetres (`:392-393`). Over the 20,666 reaches in the Processing corpus that still carry a value, the median is **-1.2 px** and 57% are negative, minimum -15.0. A reach extent that is negative more than half the time is not measuring extension.
* **Wrong.** `_compute_pellet_positioning` labels the y difference "lateral" and the x difference "depth" (`:923-933`) — backwards on both. See §7.

---

## 5. Per-reach fields (43, outside `extended`)

"Filled today" is measured on the Processing corpus (199,690 reaches) unless another source is named.

### 5.1 Identity and position in the segment

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `reach_id` | Number for this reach, unique within the video | Copied from reach detection (`:352`) | count | 100% |
| `reach_num` | Which reach this is inside its segment, from 1 | Copied from reach detection (`:360`); the detector assigns it with `enumerate(..., start=1)` (`span_to_reaches.py:160`) | count | 100% |
| `segment_num` | Which pellet presentation this reach belongs to | Set from the enclosing segment (`:317`). The constructor still initialises it to `0` with the comment "Will be set by caller" (`:361`) | count | **Mixed, and a live problem.** The assignment at `:317` landed on 2026-08-20 in commit `5bac3b0`. 32% of reaches in the Processing corpus (63,908 of 199,690) still read `0`; every reach in the Analyzed archive sample reads `0`. And it no longer reaches the database at all — see §8. |
| `is_first_reach` | Is this the first reach of the segment | `i == 0` (`:318`) | true/false | 100% present |
| `is_last_reach` | Is this the last reach of the segment | `i == n-1` (`:319`) | true/false | 100% present |
| `n_reaches_in_segment` | How many reaches in this segment | `len(seg['reaches'])` (`:307`, `:320`) | count | 100% |

### 5.2 Outcome linkage — see §2

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `causal_reach` | Is this the reach that decided the pellet's fate | `causal_reach_id == reach_id` (`:380-382`) | true/false | 0.75% of Processing reaches, 4.1% of the Analyzed sample — **never from the current algorithm.** Only legacy files, human review, and ground truth ever set it. |
| `outcome` | What happened to the pellet | Copied from the segment, **only onto the causal reach** (`:383`) | text | Same rows as `causal_reach`; null on every other reach |
| `interaction_frame` | Frame the paw met the pellet | Copied from the segment, only onto the causal reach (`:384`) | frame index | Same. The detector knows this frame for 70% of segments and it is discarded (§2) |
| `distance_to_interaction` | Frames between apex and the interaction | Read from `outcome_data['reach_features']` (`:387-389`) | frames | **0%, everywhere, permanently** — that key exists in no outcomes file ever written (§2) |

### 5.3 Extent — inherited, wrong axis, and null on current data

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `max_extent_pixels` | Intended: how far the paw reached | Copied from reach detection (`:366`) | pixels | **Null on every reach the current detector produces.** `span_to_reaches.py:179-180` writes `None` on purpose: "Extent is not cheaply available from the flat detector". 10.3% of Processing reaches (all from the older 5.3.0 detector) carry a value; those measure the sideways axis (§4). |
| `max_extent_ruler` | Same, divided by the 9 mm ruler | Copied from reach detection (`:367`) | ruler lengths (1.0 = 9 mm) | Same rows |
| `max_extent_mm` | Same, in millimetres | `max_extent_ruler * 9.0` (`:392-393`) | mm | Same rows |

Ground-truth-substituted reaches also get `None` for all three, deliberately: the algorithm's extent was measured over the algorithm's window, not the human's (`review/truth_resolver.py:192-206`).

### 5.4 Speed and path of the wrist point

All four use only the `RightHand` point (`:451-452`), and none of them gate on likelihood.

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `peak_velocity_px_per_frame` | Fastest single-frame move of the wrist during the reach | max of frame-to-frame straight-line distance (`:455-457`, `:475`) | pixels per frame | 100% |
| `mean_velocity_px_per_frame` | Average per-frame move of the wrist | mean of the same (`:476`) | pixels per frame | 100% |
| `velocity_at_apex_px_per_frame` | Wrist speed at the apex | The step *starting* at the apex: `velocity[apex_frame - start_frame]` (`:469-471`) | pixels per frame | 94.7%. **Null exactly when the apex is the reach's last frame** — there is no step starting there, so the index falls off the end of the array and the guard at `:470` returns `None`. Checked on 755 null cases in an 80-video sample: 755 of 755 had `apex_frame == end_frame`. |
| `velocity_at_apex_mm_per_sec` | The same converted | `× 9.0/ruler_pixels × 30.0` (`:405-407`) | mm per second, at an assumed 30 fps | 94.7%, same rows |

Two notes on the apex index. First, `_compute_velocity_features` strips missing values out of the speed series *before* indexing into it (`:460` then `:471`), so a gap earlier in the reach would shift which frame "at apex" refers to. In practice this cannot fire: the DeepLabCut `.h5` files contain no missing coordinates at all (12 whole videos checked, 24.1 million cells, zero nulls). Second, that same fact means there is no quality gate anywhere in this section — a frame where DeepLabCut was 5% confident contributes its coordinate at full weight to every path length, speed and area below.

Third, an inconsistency worth knowing: `:399` and `:415` compute the apex index as `apex_frame - start_frame if apex_frame else None`, which treats frame 0 as missing, while `:425` uses `if apex_frame is not None`. A reach whose apex is video frame 0 would get body features but no velocity or hand angle. No such reach was found in the corpus.

### 5.5 Path shape of the wrist point

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `trajectory_straightness` | 1.0 = the wrist went in a straight line; near 0 = it wandered | first-to-last straight-line distance ÷ total path length (`:494-500`) | dimensionless 0-1 | 100% |
| `trajectory_smoothness` | 1.0 = perfectly smooth; smaller = jerkier | `1 / (1 + mean|third difference of position|)` (`:507-518`) | dimensionless 0-1 | 98.7%. Null for reaches under 3 frames. |

### 5.6 Paw orientation

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `hand_angle_at_apex_deg` | Which way the paw was pointing at apex | `atan2(RHRight_y − RHLeft_y, RHRight_x − RHLeft_x)` per frame (`:540-542`), sampled at the apex (`:551-552`) | degrees, -180 to 180 | 100% |
| `hand_rotation_total_deg` | Total turning of the paw across the reach | Sum of absolute frame-to-frame angle changes, with ±180 wraparound handled (`:554-559`) | degrees, always ≥ 0 | 100% |

### 5.7 Head and body at the apex

Each needs both landmarks above likelihood 0.7 at that frame, otherwise null.

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `head_width_at_apex_mm` | Ear-to-ear distance at apex; a body-size proxy | distance(LeftEar, RightEar) × mm-per-pixel (`:613-619`) | mm | 99.9% |
| `nose_to_slit_at_apex_mm` | Nose to the right edge of the slit at apex | distance(Nose, BOXR) × mm-per-pixel (`:626-632`) | mm | 99.9%. Named "slit" but measured to one corner of it. |
| `head_angle_at_apex_deg` | Head orientation at apex | ear-to-ear angle (`:639-644`) | degrees, -180 to 180 | 99.9% |
| `head_angle_change_deg` | How much the head turned between reach start and apex | apex angle − start angle, wrapped into -180..180 (`:646-657`) | degrees | 99.9% |

### 5.8 Tracking quality of the wrist point

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `mean_likelihood` | Average DeepLabCut confidence in the wrist across the reach | mean of `RightHand_likelihood` (`:577`) | 0-1 | 100% |
| `frames_low_confidence` | Frames where wrist confidence was under 0.5 | count (`:578`) | frames | Present on every reach; non-zero on 5.1% |
| `tracking_quality_score` | — | **Nothing computes it.** Declared at `:83`, read by three exporters and given a database column, assigned nowhere in the repository. | — | **0%, always** |

### 5.9 Flags

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `flagged_for_review` | Does this reach need a human | **Nothing sets it.** Declared `False` at `:86`, never assigned. | true/false | **Always false.** Outcome detection does flag *segments* — 2,172 of 18,940 in the Processing corpus — and the extractor keeps that at segment level as `outcome_flagged`. The per-reach copy is dead. |
| `flag_reason` | Why | Never assigned (`:87`) | text | **Always null.** `SegmentFeatures` has no `flag_reason` field either, so the detector's reason string is dropped at both levels. |

### 5.10 Where the answer came from

| Field | Plain English | How computed | Units | Filled today |
|---|---|---|---|---|
| `outcome_source` | `algo`, `human_review`, or `ground_truth` for this segment's outcome | Copied from the segment after truth layering (`:373`); the segment default `algo` is stamped at `truth_resolver.py:164-165` | text | On every reach whose features file was written by the current extractor (92% of Processing; older files predate the field) |
| `reach_source` | `algo` or `ground_truth` for this reach's frame boundaries | `reach.get('reach_source', 'algo')` (`:374`) | text | Same — but no database column holds it (§8) |
| `reviewed_by` | Who reviewed | Copied from the segment (`:375`) | text | 0% in Processing (no reviewed videos there); 8.5% in the Analyzed sample; `FIELD_AUDIT.md` reports 19.6% over its 1,377 videos |
| `algo_outcome` | What the algorithm said before a human overrode it | Copied from the segment (`:376`), stamped at `truth_resolver.py:172` | text | Same rows as `reviewed_by` |
| `algo_causal_reach_id` | Which reach the algorithm blamed before a human overrode it | Copied from the segment (`:377`), stamped at `truth_resolver.py:173` from whatever `causal_reach_id` the algorithm had | count | **0%.** It preserves the algorithm's causal reach id, and the current detector never sets one (§2). `FIELD_AUDIT.md` also reports 0.0% over 1,377 videos. |

### 5.11 Fields declared and never computed by anything

Four more with the same story as `tracking_quality_score`: a name in the dataclass, a database column, exporter lines that read them, and no code anywhere that assigns them.

| Field | Declared | Intended meaning | Replacement that does exist |
|---|---|---|---|
| `grasp_aperture_max_mm` | `:66` | Widest paw opening during the reach | `paw_width_proxy_max_mm` in `extended` |
| `grasp_aperture_at_contact_mm` | `:67` | Paw opening when it met the pellet | `paw_width_proxy_at_contact_mm` (itself null, §2) |
| `apex_distance_to_pellet_mm` | `:76` | How close the paw got to the pellet | none |
| `lateral_deviation_mm` | `:77` | How far sideways the paw strayed | `righthand_lateral_deviation_mm` and the same for the three other paw points |

---

## 6. The `extended` block: 161 values per reach

Produced by `_extended_features` (`:1425-1464`) and stored as one nested dictionary. If any part of it raises, the whole block is replaced by a single key `_extended_features_error` holding the message (`:443-444`) — the reach still ships, with 161 values missing and no other warning. Measured across all 199,690 reaches in the Processing corpus: **zero** such errors.

Of the 161 values, 151 are non-null on every one of 21,508 reaches checked. Only the ten `_at_contact` columns are ever null, and they are null almost always (§2).

### 6.1 The coordinate frame these use

Two things are done before the per-paw trajectory measurements (`_build_augmented_nose_relative_trajectory`, `:998-1032`):

1. **Nose subtraction.** Every paw position becomes `paw − nose` at the same frame, so head movement does not count as paw movement.
2. **Two synthetic anchor frames** are added, one before the reach and one after:
   * before: the midpoint of Nose and BOXL at `start_frame`, i.e. `(BOXL − Nose)/2` in nose-relative terms (`:1023-1024`)
   * after: the midpoint of Nose and BOXR at `end_frame` (`:1027-1028`)

   These are geometry, not tracking, so the reach always starts and ends at a defined place instead of wherever DeepLabCut first noticed the paw.

**Which measures include the two synthetic points.** Everything computed from the array `x`/`y` returned at `:1050` does: total path, lateral spread, swept area, path directness, motion smoothness, lateral deviation, mean speed, peak speed, and apex speed. Only the apex frame and `extension_past_nose` strip them, via `real_y = y[1:-1]` (`:1057`). Everything in §6.3, §6.4 and §6.5 uses real frames only.

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
| `_peak_speed_px_per_frame` / `_mm_per_frame` | Fastest single step | max step length (`:1136`, `:1142-1143`) | pixels or mm per frame |
| `_apex_speed_px_per_frame` / `_mm_per_frame` | Speed at this point's own apex | the step immediately after the apex frame in the anchored array (`:1147-1159`) | pixels or mm per frame |

One dead branch: `:1154-1157` is an `elif` meant to handle the apex being the last frame. It cannot run. The anchored arrays have `n_real + 2` points, so the step array has `n_real + 1` entries, while the apex index runs 0 to `n_real − 1`; `apex_idx + 1` is at most `n_real`, always inside the array. When the apex is the last real frame, apex speed is therefore the step from that frame to the synthetic post-anchor — a geometric quantity, not a measured one.

### 6.3 Coordination between the four points — 3 values

| Column | Plain English | How computed | Units |
|---|---|---|---|
| `paw_apex_lead_frames` | How spread out in time the four points' apexes were | latest apex − earliest apex (`:1176`); null if fewer than two points have an apex | frames |
| `paw_leading_point` | Which point apexed first | the point with the smallest apex frame, as a name (`:1177-1178`) | text: one of the four |
| `paw_velocity_correlation` | How much the four points sped up and slowed down together | mean of the six pairwise Pearson correlations of the frame-to-frame speed series, real frames only (`:1186-1207`) | dimensionless, -1 to 1 |

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

Measured over the Processing corpus: 5,337 of 199,690 reaches (2.7%) contain any such frame, 5,667 frames in total. Within detected reaches, paw tracking confidence is essentially always high.

### 6.6 Tray contact — 1 value

| Column | Meaning | How computed | Units |
|---|---|---|---|
| `tray_contact_duration_frames` | Intended: frames where the pellet was jiggling while the wrist was hidden — a proxy for the paw being on the tray | count of frames where the 4-frame rolling standard deviation of `Pellet_x` exceeds 2 px **and** `Pillar_likelihood < 0.5` **and** `RightHand_likelihood < 0.5` (`:1414-1421`) | frames |

**This value is zero on 199,682 of 199,690 reaches.** Only 8 reaches in the whole Processing corpus have a non-zero count. That is structural, not incidental: the window is a *detected reach*, which means the wrist was visible enough to be detected, while the third condition requires the wrist to be invisible. Returns 0 for reaches shorter than 4 frames (`:1411-1412`) and `None` if any of those three columns is missing from the tracking file (`:1422-1423`). The 2-pixel threshold is a raw pixel constant, not scaled by `ruler_pixels`, so it means a different physical distance on differently-zoomed videos.

---

## 7. Per-segment fields (25)

One record per pellet presentation. All are written to `{video}_features.json`; only five of them ever reach the database (§8).

| Field | Plain English | How computed | Units | Notes |
|---|---|---|---|---|
| `segment_num` | Which pellet presentation | copied (`:289`) | count | The only place the pellet number is recorded correctly |
| `start_frame`, `end_frame` | Segment bounds | copied (`:290-291`) | frame index | |
| `ruler_pixels` | The 9 mm reference in pixels for this segment | copied (`:292`) | pixels | drives every mm conversion in the segment |
| `outcome` | What happened to the pellet | copied from outcome detection (`:293`) | text | 100% |
| `outcome_confidence` | Detector's confidence | copied (`:294`) | 0-1 | **Null on every segment the current detector produces.** Present on 3,980 of 22,920 Processing segments — exactly the 199 legacy videos. |
| `outcome_flagged` | Detector asked for human review | copied (`:295`) | true/false | 11.5% of current segments |
| `n_reaches` | Reach count | copied from the reach file (`:296`) | count | A separate source from `n_reaches_in_segment`, which counts the list |
| `causal_reach_id` | Which reach caused the outcome | copied (`:297`) | count | Null on every current segment (§2) |
| `attention_score` | Share of frames the mouse was at the tray | frames where `Nose_likelihood > 0.9` and `Nose_y > median(BOXR_y) − 80`, ÷ total × 100 (`:790-808`) | percent 0-100 | The 80 is raw pixels, not scaled by `ruler_pixels`. Median 45.6 over the Processing corpus. |
| `mean_head_width_mm` | Average ear-to-ear distance | mean over frames with both ears above likelihood 0.7 (`:694-702`) | mm | |
| `mean_nose_to_slit_mm` | Average nose-to-BOXR distance | same gate on Nose and BOXR (`:709-716`) | mm | |
| `mean_nose_height` | Average nose y position | mean of `Nose_y` over frames with `Nose_likelihood > 0.7` (`:723-726`) | **raw image pixels, not converted, not referenced to anything** | comparable only within one camera setup |
| `mean_head_angle_deg` | Average head orientation | plain arithmetic mean of ear-to-ear angles, over frames with both ears above 0.7 (`:734-739`) | degrees | **Not valid as written.** Measured angles span -166 to +176 degrees; averaging a wrap-around quantity arithmetically is meaningless when it straddles ±180. |
| `head_angle_variance` | Head-orientation steadiness | plain variance of the same angles (`:740`); null when fewer than 2 frames pass the gate | degrees² | Same problem, worse: median 6,426 (a spread of about 80 degrees), 90% of segments above 1,000. Dominated by the ±180 jump, not by head steadiness. |
| `nose_position_variance` | Postural steadiness | `var(Nose_x) + var(Nose_y)` over frames with `Nose_likelihood > 0.7`, null if fewer than 2 (`:749-756`) | **pixels², not converted to mm²** | |
| `segment_duration_sec` | Length of the presentation | `(end − start + 1) / 30` (`:832-833`) | seconds at an assumed 30 fps | |
| `time_to_first_reach_sec` | Delay before the mouse first reached | `(first reach start − segment start) / 30` (`:836-839`) | seconds | null if the segment has no reaches |
| `time_to_outcome_sec` | Intended: when the outcome happened | **`= segment_duration_sec`, unconditionally** (`:844`) | seconds | Carries no information beyond segment length. The comment says "assume outcome happens at segment end". The outcome detector *does* compute a real answer, `outcome_known_frame`, present on 16,768 of 18,940 current segments (88.5%); the extractor never reads it. |
| `mean_inter_reach_interval_sec` | Average gap between reaches | mean of (next start − previous end)/30 (`:847-855`) | seconds | null when fewer than 2 reaches; can be negative if reaches overlap |
| `pellet_lateral_offset_mm` | Labelled sideways offset of the pellet | `abs(pellet_y − BOXR_y) × mm-per-pixel` (`:924-925`) | mm | **Measures the out-through-the-slit direction, not sideways** (§4) |
| `pellet_depth_offset_mm` | Labelled depth offset of the pellet | `abs((pellet_x − BOXR_x) − 30) × mm-per-pixel` (`:930-933`) | mm | **Measures the sideways direction** (§4), and subtracts a hard-coded ideal of 30 raw pixels that belongs on the other axis |
| `pellet_position_idealness` | 0-1 score for how well placed the pellet was | `1 − (0.6 × min(lateral/50, 1) + 0.4 × min(depth_deviation/40, 1))`, floored at 0 (`:938-942`) | dimensionless 0-1 | Built from the two mislabelled axes with three more raw-pixel constants (50, 40, 30). Over the Processing corpus: mean 0.358, standard deviation 0.062, never above 0.679 — consistent with the depth term sitting at full penalty nearly always. |
| `mean_tracking_quality` | Average confidence over every tracked point | mean of all `*_likelihood` columns in the segment (`:967-976`) | 0-1 | includes apparatus landmarks, not just the mouse |
| `tracking_dropout_frames` | Frames where any tracked point was under 0.5 | count (`:979-981`) | frames | |
| `reaches` | The reach records | | list | |

The three pellet-position values are measured over a window that runs from the segment start to the start of the first reach, or the first 30 frames if the segment has no reaches (`:886-892`), using the median pellet position over frames where `Pellet_likelihood > 0.7` (`:902-911`). All three are null if the pellet is never tracked well in that window.

---

## 8. What survives the trip to the database and the CSVs

The extractor's own file is complete. Everything downstream loses something, and the database path is currently broken.

### `{video}_features.json`

Everything above.

### `connectome.db`, table `reach_data` — **inserts have failed since 2026-08-20**

One row per reach is built at `sync/database.py:576-608` and inserted at `:619-632`.

* The insert names `ALL_COLUMNS` (`:84-92`), which is `REACH_JSON_COLUMNS` (41 names, `:61-78`) plus five session-identity, five segment-context, four metadata and six provenance columns — 61 in all.
* `segment_num` is **not** in that list. It was removed by commit `5bac3b0` on 2026-08-20, on the reasoning (comment at `:56-60`) that the reach's copy was always 0 and should come from the segment instead. The row dictionary does carry `segment_num` from the segment (`:568`), but the INSERT statement never names it.
* The live table declares `segment_num INTEGER NOT NULL` with no default. Reproducing the exact statement against a fresh table built from `CREATE_REACH_DATA_SQL` raises `IntegrityError: NOT NULL constraint failed: reach_data.segment_num`. **Every insert fails.**
* This is invisible. The DELETE and the INSERTs share one connection block that is only committed at the end (`:623-634`), so the failure rolls the DELETE back — existing rows survive, nothing is corrupted, and nothing new arrives. `sync_file_to_database` then catches every exception and returns `False` (`:857-858`), so the callers' own `except` blocks never fire. The watcher logs it at DEBUG as "Database sync skipped (subject not in DB or DB unavailable)" (`orchestrator.py:2129-2130`) — a message that names two causes, neither of which is the real one.
* Confirmed on the live database: the newest `imported_at` in `reach_data` is `2026-08-20T14:25:47`, twenty seconds after commit `5bac3b0` landed. Nothing has been written in the three days since.

When it did work, the table:

* Kept **41** of the 43 per-reach fields, plus the entire `extended` dictionary as a JSON blob in one column, `extended_features` (`:595`).
* Kept exactly five segment-level fields: `segment_outcome`, `segment_outcome_confidence`, `segment_outcome_flagged`, `attention_score`, `pellet_position_idealness` (`:565-574`).
* Dropped **`reach_source`** — computed on every reach, no column.
* Dropped **the other fourteen segment measurements entirely**: `mean_head_width_mm`, `mean_nose_to_slit_mm`, `mean_nose_height`, `mean_head_angle_deg`, `head_angle_variance`, `nose_position_variance`, `segment_duration_sec`, `time_to_first_reach_sec`, `time_to_outcome_sec`, `mean_inter_reach_interval_sec`, `pellet_lateral_offset_mm`, `pellet_depth_offset_mm`, `mean_tracking_quality`, `tracking_dropout_frames`. No column exists for any of them. They are computed on every video and read by nobody. They also never appear in `docs/FIELD_AUDIT.md`, because that tool compares reach-level record keys against database columns, and these are neither.

### `Databases/database_dump/reach_data.csv` — **the file on disk has misaligned columns**

`export_csv` (`sync/database.py:749-805`) selects 57 columns (`:759-784`). Relative to the table it includes `segment_num` and omits `extended_features`, `outcome_source`, `reviewed_by`, `algo_outcome`, `algo_causal_reach_id`. So all 161 extended values and all review provenance are absent from the file that downstream `mousedb` recipes read.

Older code wrote the header from `ALL_COLUMNS` while the rows came from that separate SELECT. Commit `eec74e6` (2026-08-21) fixed it by taking the header from the query itself. **The file on Y: predates the fix** — it is dated 2026-08-20 20:18 and carries a 61-name header over 57-value rows. Verified by reading it: the header has 61 names, the first data row has 57 values, and from position 6 onward every column is labelled with the name of the column before it. The column headed `reach_id` holds pellet numbers; the one headed `causal_reach` holds outcome strings such as `displaced_sa`; the one headed `max_extent_pixels` holds durations. It has not been regenerated since, because `export_csv` runs only after a successful sync (`:854`) and syncs have been failing.

### `mousereach-reach-export`

`kinematics/analysis/reach_export.py`. The command (`main`, `:292-301`) writes **one aggregate file**, `reach_kinematics.csv`, beside the Processing directory. It does not write `{video}_results.csv` and does not write cohort CSVs. Those are separate functions: `write_video_results_csv` (`:239-253`) is called only from `pipeline/reprocess_to_current.py:267`, and `export_cohort` (`:267-289`) has no caller anywhere in `src/`.

The column list is not fixed. `write_csv` (`:218-236`) walks every row and appends every key not already in `STATIC_COLUMNS` (`:24-62`), so `fieldnames` is a superset of everything present and the `extrasaction="ignore"` at `:232` can never drop anything. All 161 extended values are carried.

Six values are nevertheless missing, for a different and worse reason: **the row builder never reads them.** `_row_from_features_reach` (`:65-113`) copies 33 named keys and then `reach["extended"]`, and `velocity_at_apex_px_per_frame`, `velocity_at_apex_mm_per_sec`, `peak_velocity_px_per_frame`, `mean_velocity_px_per_frame`, `trajectory_straightness` and `trajectory_smoothness` are not among them. They exist in the features file and simply never enter the row.

### `mousereach-features-csv`

`export/features_csv.py` globs `*_grasp_features.json` (`:16`). Nothing in the pipeline writes a file with that name — kinematics writes `{video}_features.json`. The command always prints "No *_grasp_features.json files found!" and exits (`:18-21`). Its row dictionary (`:38-77`) has 30 columns, of which 5 are the never-computed fields of §5.11; the other 25 are fields the extractor does populate.

### `kinematics/analysis/odc_sci_exporter.py`

Reads the features file (`:109-110`). Six of its columns read keys that the features schema does not have, so they are constants: `Exclude From Analysis`, `Exclude Reason`, `Human Corrected`, `Source`, `Review Note` (`:157-161`, all reach-level keys that live in `_reaches.json`, not in `ReachFeatures`), and `Segment Flagged` / `Segment Flag Reason` (`:162-163`, which look for `flagged_for_review` and `flag_reason` on a segment record whose actual field is `outcome_flagged` and which has no reason field at all).

---

## 9. Human corrections, and one path that cannot fire

`extract` calls `resolve_truth_layers` (`:220-225`), which resolves each segment's outcome and each reach independently by ground truth > deep review > triage review > algorithm (`review/truth_resolver.py:315-334`), and can replace the reach list itself when ground truth marks its reach set exhaustive (`:275-282`). Provenance lands in `outcome_source` and `reach_source`.

That resolver takes a **directory** as `primary_dir` and looks for `<dir>/<video>/<video>_causal_review.json` or `<dir>/<video>_causal_review.json` (`truth_resolver.py:72-88`). But every production caller passes a **file** path: `resolve_review_path` returns the review file itself (`review/causal_review_io.py:83-90`) and the watcher, `run_all.py` and `reprocess_to_current.py` hand that straight in as `review_path`; the CLI builds `Path(_rd) / f"{video_name}_causal_review.json"` (`kinematics/cli.py:123`) and does the same. The lookup then builds paths like `.../X_causal_review.json/X_causal_review.json`, `is_file()` returns false, and that layer is silently skipped.

The practical damage is limited, because the resolver also checks `Paths.TRIAGE_REVIEW` and `Paths.DEEP_REVIEW` on its own (`:315-316`), and the review tool's default save location is `Paths.TRIAGE_REVIEW` (`review/staging.py:57`). A review saved only in the processing directory, or only next to the video, does not reach kinematics.

If `resolve_truth_layers` raises, the code prints a line and falls back to the older review-only path, which does accept a file (`:226-239`); if that raises too it prints a second line and continues with the raw algorithm output.

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
| **Which ground-truth reaches replace which algorithm reaches** | `truth_resolver.py:262`, `:271`, `:274` | GT reaches are bucketed by `segment_num` and applied to the segment with that number. If the segmentation was re-cut after the ground truth was written, GT reaches land on the wrong footage. (Review *outcomes* are safer: they are re-anchored by frame range when `current_segments` is passed, `truth_resolver.py:100-103`.) |
| **The pellet number recorded downstream** | `sync/database.py:568` | The database's `segment_num` column is taken from the segment record; it is the only place the pellet number is preserved. |

Two things segments do **not** control. Reach frame windows are never clipped to the segment — a reach is measured over its own `start_frame`..`end_frame` regardless of where the boundary sits (`:396`). And the extractor never re-derives or validates a boundary; it takes `start_frame` and `end_frame` verbatim.

---

## 11. Failures that produce a file anyway

| Failure | What happens | Where |
|---|---|---|
| Anything inside the extended feature block raises | The reach ships with `extended = {'_extended_features_error': message}` — 161 values missing, one key present | `feature_extractor.py:443-444` |
| Truth layering raises | Printed, then fall back to review-only, then to raw algorithm | `:226-227`, `:238-239` |
| Database sync fails, for any reason | `sync_file_to_database` returns `False` and never raises | `sync/database.py:857-858` |
| Database sync fails, from the CLI | `except Exception: pass` — and the call cannot raise anyway, so nothing is printed either way | `kinematics/cli.py:143-144` |
| Database sync fails, from the watcher | `logger.debug("Database sync skipped ... subject not in DB or DB unavailable")` — the wrong cause | `orchestrator.py:2126-2130`; the other watcher path (`:1166-1170`) logs nothing at all on a `False` return |
| Feature extraction fails, from the watcher's local pipeline | Logged as a warning, then `update_state(video_id, 'processed')` runs anyway | `orchestrator.py:1171-1174` |
| The reach file and the outcome file describe different segments | Nothing checks | `feature_extractor.py:248` |
| `ruler_pixels` is 0 or negative | Every millimetre value in the segment becomes exactly `0.0`, not null | `:606`, `:689`, `:884`, `:1048`, `:1246` |

---

## 12. Reconciling with `docs/FIELD_AUDIT.md`

That document reports fill rates over 1,377 videos, using a parquet snapshot dated 2026-08-20 12:20 for its database column. Every kinematics-owned field it lists as empty, with the reason:

| Field | Audit bucket | Why it is empty |
|---|---|---|
| `tracking_quality_score` | never computed | Declared at `:83`, assigned nowhere in the repository |
| `apex_distance_to_pellet_mm` | never computed | Declared at `:76`, assigned nowhere |
| `lateral_deviation_mm` | never computed | Declared at `:77`, assigned nowhere; the per-paw versions in `extended` exist instead |
| `grasp_aperture_max_mm` | never computed | Declared at `:66`, assigned nowhere; `paw_width_proxy_max_*` exists instead |
| `grasp_aperture_at_contact_mm` | never computed | Declared at `:67`, assigned nowhere; the replacement is also null for want of an interaction frame |
| `distance_to_interaction` | never computed | **Mis-classified — this is lost in transit.** The older outcome detector computes it (`outcomes/core/pellet_outcome.py:1645-1689`) into a `reach_features` dictionary that `find_causal_reach` returns and the caller never writes onto the record it builds at `:1591`. The extractor's `outcome_data.get('reach_features', {})` therefore always returns empty (`:387`). |
| `max_extent_pixels`, `max_extent_ruler` | never computed | **Half right.** The current reach detector deliberately writes `None` (`reach/core/span_to_reaches.py:179-180`), and the audit's own corpus is current-stage-only, so 0% is correct there. In the wider table these columns are not empty — 10.3% of the Processing corpus and 53% of the Analyzed sample carry legacy values. Do not read "never computed" as "the column is empty everywhere". |
| `max_extent_mm` | never computed | Derived from `max_extent_ruler`, so it follows it exactly (`:392-393`) |
| `algo_causal_reach_id` | never computed | Copied from the segment (`:377`); the segment value preserves the algorithm's causal reach id, which the current outcome detector never emits |
| `flagged_for_review` (reach level) | lost in transit | Outcome detection flags *segments*; the extractor keeps that as segment `outcome_flagged` and never copies anything onto the reach field at `:86` |
| `flag_reason` (reach level) | lost in transit | Same; `SegmentFeatures` has no `flag_reason` field at all, so the reason string is dropped at the segment level too |
| `segment_num` (features 0%) | lost in transit | The assignment at `:317` landed on 2026-08-20 in commit `5bac3b0`; every audited file predates it. It now reaches the features file on 68% of Processing reaches and reaches the database on none (§8) |
| `outcome_known_frame` (94.5% at the outcome stage, absent from features) | lost in transit | The extractor reads `interaction_frame` and `causal_reach_id` from the outcome record and nothing else |
| `is_causal`, `label`, `segment_ifr`, `triage_reason` (from reach assignment) | lost in transit | `{video}_reach_assignments.json` is never opened by the kinematics package (§2) |
| `extended`, `reach_source` | not in the database | `extended` is in fact stored, as the `extended_features` JSON blob (`sync/database.py:595`, which the audit separately reports at 99.5%); the audit is looking for a column named `extended`, which does not exist. `reach_source` genuinely has no column. |

The audit's `causal_reach` at 1.6% and `interaction_frame` at 0.7% are legacy files, human reviews and ground truth only. Nothing the current algorithm produces contributes to either (§2).

---

## 13. Documentation in this area that is wrong

* `src/mousereach/kinematics/core/AGENTS.md:56` lists scipy as a dependency and says velocity is "smoothed with Savitzky-Golay filter". The extractor imports only `dataclasses`, `pathlib`, `typing`, `json`, `numpy` and `pandas` (`:8-14`), and applies no smoothing of any kind. The same file gives the wrong interface — it shows `FeatureExtractor(dlc_file, reaches_file, outcomes_file)` and `extract_all_features()`; the real interface is `FeatureExtractor()` then `.extract(dlc_path, reaches_path, outcomes_path, review_path=None)`. It also describes `SegmentFeatures` as "mean/std of reach features" and `VideoFeatures` as carrying "success rates"; `SegmentFeatures` aggregates no reach kinematics at all, and the video summary (`:1489-1503`) has no success rate.
* `REACH_KINEMATIC_DATA_DICTIONARY.md:89` defines `distance_to_interaction` as "`interaction_frame - apex_frame`, in frames". Nothing computes that; the value is read from a dictionary that is never written (§2).
* `REACH_KINEMATIC_DATA_DICTIONARY.md:39-43` says tracking-quality gating is unnecessary because "numerically robust accumulation (`np.nansum`, `np.nanmean`, etc.) tolerates occasional NaN frames within a tracked reach". There are no NaN frames — DeepLabCut always emits a coordinate (12 whole videos checked, 24.1 million cells, zero nulls). The conclusion happens to be defensible; the stated reason is not, and it obscures the real consequence, which is that low-confidence frames enter every trajectory measurement at full weight.
* `feature_extractor.py:361` still initialises `segment_num=0` with the comment "Will be set by caller". As of 2026-08-20 that is true — `:317` sets it. The comment is now merely stale rather than wrong.
* `sync/database.py:56-60` says the extractor "leaves it 0 on every reach". That stopped being true on 2026-08-20, and the change made on the strength of that comment is what broke the database insert (§8).

---

## Contested claims

This document was written from the source, then checked by a second reader
whose job was to disprove it. The statements below are ones they disputed and
that were not resolved. Neither side is authoritative: spot-checking found the
checker wrong at least once. **Do not rely on anything listed here without
opening the code yourself.** Everything not listed survived two passes.

- **"`ReachDetector.detect` — the v6 algorithm — has **no caller anywhere in `src/`**. Its only caller in the repository is the one-off `scripts/diagnose_passes.py`."**
  - disputed because: Both halves are false. There ARE two call sites in src/ (both unreachable, but present), and the named script is not one of the callers, while a different file is.
- **"`spatial_refiner.py` (1008 lines) has **no importer at all**"**
  - disputed because: It has an importer.
- **"Also dead: `boundary_refiner.py` and `boundary_polisher.py`, whose only importer is the dead `reach_detector.py` (`:101-102`)."**
  - disputed because: True for boundary_polisher.py, false for boundary_refiner.py, which has two other importers.
- **"[v8/eval.py] is called by `v8/train.py:123`."**
  - disputed because: v8_figures.py does not call or import v8/eval.py. It consumes eval output that has already been serialised to plain dicts.
- **"In a typical file the first boundary is near frame 2000, so the first ~2000 frames are outside every segment."**
  - disputed because: Off by more than an order of magnitude. The structural point (frames before the first boundary belong to no segment) is right, but the size of that region is wrong.
- **"Measured: 13 of the 947 files contain a bare `NaN`, always in `ruler_pixels`. Eleven are the trailing segment 21, length 0 or 1, with no reaches. Two are segment 1 with length 0, flagged, holding 12 and 2 orphan reaches."**
  - disputed because: The counts split differently: only nine of the eleven no-reach cases are the trailing segment 21. Two are mid-video segments, which the document's framing ("trailing segment 21" vs "segment 1") implicitly rules out. This matters because mid-video NaN segments fall inside segments 1-20, the range kinematics actually reads, so their segment-level millimetre features are NaN too.
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
credited -- for segments no higher authority has resolved. The ordering is
strict and tested: a segment carrying a `causal_reach_id` KEY is owned by human
review, ground truth or the pre-v6 detector and is never overridden; a
reviewer's `causal_reach_id: None` means "no reach did this" and stands. On the
algo-only test video the features now name exactly the 4 reaches the assignment
file names; on a fully human-reviewed video all causal marks are
`human_review`-sourced and none land on human-untouched segments.

Older reviews that stored the human's causal pick as frames only (no reach id)
are also recovered now: the resolver matches the frames to the reach they name
(>=50% overlap, within the reviewed segment). 16 of 18 picks on the test video
resolve; the other 2 are hand-drawn reaches that exist in no reaches file and so
have no row to mark.

`distance_to_interaction` remains never-computed -- its definition is unclear
and inventing one was worse than leaving it empty. Corpus backfill and the
2.1.0 declaration in pipeline_versions.json are deliberately deferred; until
declared, newly processed videos will read as version-stale. See UNFINISHED.md.
