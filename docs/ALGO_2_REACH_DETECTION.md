# Reach detection: finding each attempt

Describes: `src/mousereach/reach/v8/`, `src/mousereach/reach/core/`, `src/mousereach/reach/cli.py`, `src/mousereach/reach/review_widget.py`, `compute_reach_apex` in `src/mousereach/lib/causal_attribution.py`, and the two places that call reach detection (`src/mousereach/reach/core/batch.py`, `src/mousereach/review/staging.py`)

The file:line references below were taken from an older version of the code and many no longer match; find the code by the function, class or constant named beside each reference.

---

## What this subsystem does, in one paragraph

It takes one video's pose-tracking file (a DeepLabCut `.h5`, giving an x, a y and a confidence number for each of 18 tracked points on every frame) plus a list of segment boundary frames, and writes `{video}_reaches.json`: a list of reach attempts, each one a start frame and an end frame, filed under the segment it started in. It never opens the video pixels. It does not decide whether a reach succeeded — that is the outcome detector's job.

The thing that decides where reaches are is a trained machine-learning model that scores **every frame independently** with "how likely is it that this frame is inside a reach". Everything else is bookkeeping around that score. **The model never sees the segment boundaries.** Segments are applied afterwards, to file the reaches; they cannot move a reach boundary.

---

## Which code actually runs (and which does not)

There are three complete reach detectors in this tree. **Only one runs.**

| File | Version string | Status |
|---|---|---|
| `reach/v8/` (the package) | `8.1.0` (`v8/__init__.py:35`) | **This is production.** |
| `reach/core/reach_detector_v8.py` | `7.2.0` (`:93`) | Dead. Nothing imports it; the only mention outside the file is a docstring line at `span_to_reaches.py:279`. |
| `reach/core/reach_detector.py` | `6.0.0` (`:105`) | Dead as a detector. Its `Reach` / `SegmentReaches` / `VideoReaches` dataclasses and its `save_results` writer are still used (see below). |

`ReachDetector.detect` — the v6 algorithm — has **no caller anywhere in `src/`**. Its only caller was a one-off diagnostic script that is not part of this repository. What the three live callers use from that file is `ReachDetector.save_results` (a `@staticmethod` at `reach_detector.py:1104`) plus the three dataclasses, which `span_to_reaches.py:53` imports (`from .reach_detector import Reach, SegmentReaches, VideoReaches` — that line imports the dataclasses only, not the writer).

Also dead: `boundary_refiner.py` and `boundary_polisher.py`, whose only importer is the dead `reach_detector.py` (`:101-102`). `spatial_refiner.py` (1008 lines) has **no importer at all** — `reach_detector.py:281` sets `self._spatial_refiner = None` and only mentions the module in a comment at `:280`.

**Confusing naming to be aware of.** `reach_detector_v8.py` is *not* the v8 detector. It is an older rule-based detector (paw-visibility runs, a velocity "shoot" test, a nose-near-the-slit filter), and it declares its own version as `7.2.0` while its docstring header says `v7.1.0`. It even defines a module-level function called `detect_reaches_v8` (`:396`), which is a different function from the real one in `v8/__init__.py:94`. Nothing calls the one in `reach_detector_v8.py`. Its whole module docstring — visibility runs, `SHOOT_MAG`, `NOSE_ENG_MIN`, the performance table — describes code that never executes.

### The one live path

```
DLC .h5 + {video}_segments.json
  -> span_to_reaches.detect_video_reaches()        (core/span_to_reaches.py:230)
       -> v8.detect_reaches_v8(dlc_df)             (v8/__init__.py:94)  -> [(start,end), ...]
       -> span_to_reaches.build_video_reaches()    (core/span_to_reaches.py:78)
  -> ReachDetector.save_results()                  (core/reach_detector.py:1104)
  -> {video}_reaches.json
```

Two places call `detect_video_reaches`:

- **`reach/core/batch.py:163`**, inside `process_single`. This is the busy one. It is reached from `process_batch` (`batch.py:244`, which is what `mousereach-detect-reaches` runs), from both watcher pipelines (`watcher/orchestrator.py`, the two places that import `reach.core.batch.process_single`; the Run Pipeline tab's "Run chosen videos...", `pipeline/run_all.py`, reaches it only through the watcher's own handlers, `watcher/run_one.py`), and from `pipeline/reprocess_to_current.py`.
- **`review/staging.py:304`**, building a review bundle.

Both then call `ReachDetector.save_results` (`batch.py:168`, `staging.py:305`). The napari "Run Pipeline" tab is not a third caller: it runs one pass of the watcher, which goes through `process_single` like every other watcher run. `span_to_reaches` itself writes nothing to disk.

---

## How a candidate reach is proposed

### Step 1 — build a feature table

`v8/features.py:extract_features` (`:76-148`) turns the pose file into **405 numbers per frame**. For each of the 18 tracked points (`features.py:46-54`: `Reference`; the four scoring-area corners `SATL`/`SABL`/`SABR`/`SATR`; the two box edges `BOXL`/`BOXR`; `Pellet`; `Pillar`; the four paw points `RightHand`/`RHLeft`/`RHOut`/`RHRight`; `Nose`; `RightEar`; `LeftEar`; `LeftFoot`; `TailBase`) it computes 14 numbers (assigned at `features.py:124-137`):

- raw x, raw y, raw confidence;
- x and y smoothed with a 5-frame centred moving average;
- velocity in x and y — the smoothed value 2 frames ahead minus the value 2 frames behind, divided by 4 (`_centered_diff`, `features.py:151-160`);
- acceleration in x and y, the same difference applied again to velocity;
- speed = sqrt(vx² + vy²);
- change in confidence, same centred difference;
- the largest speed anywhere in a ±10-frame window, and in a ±20-frame window;
- the lowest confidence anywhere in a ±10-frame window.

Then it adds the straight-line distance between every pair of points, on smoothed coordinates — 153 pairs (`features.py:140-145`). 18 × 14 + 153 = 405.

Two things worth knowing:

- **Nothing is filtered by anatomy or by confidence.** Distances between two fixed apparatus corners are included even though they barely change. That is deliberate; the module docstring at `features.py:28-32` says so, and the code matches.
- **A missing tracked point is silently replaced by zeros** (`features.py:90-97`). If a pose file used different point names — a different DeepLabCut project, or a multi-animal file whose column index carries an extra level — every feature for the missing points, including all pairwise distances involving them, becomes 0.0, and the model still returns predictions. There is no error and no warning.

### Step 2 — score every frame

`v8/__init__.py:173-181` loads a joblib bundle and calls `predict_proba`. Loaded bundles are cached per path for the life of the process (`:82-91`).

The bundle is a plain dict with exactly two keys, `model` and `feature_columns` — no version, no training record, no pose-model identifier. (Verified by loading `v8.1.0_bsw_w0.7_model4.0.joblib`: keys `['model', 'feature_columns']`, 405 columns.)

The model is a **scikit-learn `HistGradientBoostingClassifier`** — a gradient-boosted decision-tree ensemble. It is a plain per-frame binary classifier with no memory of neighbouring frames; all temporal context comes from the smoothing, velocity and rolling-window features above.

Default bundle: `v8/models/v8.1.0_bsw_w0.7_model4.0.joblib` (`v8/__init__.py:45`). Its sidecar `.meta.json` records: trained on DLC model 4.0 (resnet101 shuffle3 @100000), 37 training videos, 1,413,715 training frames, 405 features, `max_iter` 200, `learning_rate` 0.05, `max_depth` 6, `random_state` 42, exhaustively-annotated videos only. It records **no accuracy numbers, no training date and no video list** — unlike the older sidecar `v8.0.0_bsw_w0.8.meta.json`, which records all three.

The older bundle `v8.0.0_bsw_w0.8.joblib` is still shipped for pose from DLC model 3.1, but nothing selects it automatically. You would have to pass `model_path` by hand, and no caller does.

**There is no check that the pose file was produced by the pose model the classifier was trained on.** `pipeline_versions.json` declares `dlc_scorer` as `DLC_resnet101_<project><date>shuffle3_100000`, but the scorer name is the outermost level of the `.h5` column index and `geometry.load_dlc` (`geometry.py:136-137`) throws that level away before features are built. Running the 4.0-trained model over 3.1 pose data produces confident output with no warning.

**Nothing in this repository can produce the shipped model.** No file under `src/` or `scripts/` calls `joblib.dump` — there is no model writer at all. The bundle's own sidecar credits work outside the repository (a separate recalibration workspace). Separately, `v8/train.py` **cannot reproduce it even in principle**: the bundle metadata records boundary sample weighting (`bsw_buffer: 1, bsw_weight: 0.7`, i.e. down-weighting training frames near a reach boundary), and `train.py:73-90` applies class-balance weights only, with no boundary term. A working boundary-weighting implementation existed in a one-off script outside this repository, but that script trained on DLC 3.1 pose data and never saved a model. `train.py` also evaluates with `probabilities_to_reaches` alone (`:99-100`) — none of the post-processing below — so the numbers it prints are not comparable to production.

### Step 3 — turn scores into spans

`postprocess.probabilities_to_reaches` (`postprocess.py:56-94`):

1. A frame is "in a reach" if its probability is **strictly greater than** `threshold` (`:81`). Production threshold is 0.5 (`v8/__init__.py:49`).
2. Contiguous in-reach frames form a run (`:97-111`).
3. Runs separated by a gap of `merge_gap` or fewer frames are joined (`:114-125`). **Production `merge_gap` is 0** (`v8/__init__.py:50`). Runs produced by step 2 are always separated by at least one below-threshold frame, so the test at `:120` can never pass. **No merging ever happens in production.** The parameter exists and has no effect at its shipped value.
4. Runs shorter than `min_span` frames are dropped (`:93`). Production `min_span` is 3.

Segments are not consulted at any point here. The model runs over the whole video.

---

## How start and end frames are decided

The raw start is the first frame of a run scored above 0.5; the raw end is the last frame of that run (end is **inclusive**). Three adjustments then run, in this fixed order (`v8/__init__.py:192-217`):

### 1. Leading trim — moves the start later

`postprocess.trim_leading_sustained_lk` (`postprocess.py:139-210`). It computes `paw_mean_lk`: the plain mean of the four paw points' confidences on each frame (`:128-136`). Walking forward from the start, it deletes a frame only if that frame **and** the next `sustain_n - 1` frames are *all* below the confidence threshold (`:205`). It stops at the first window containing one confident frame, at the first window containing a NaN (`:203`), or when fewer than `sustain_n` frames remain before the end (`:199`). If what survives is shorter than `min_span`, **the whole reach is deleted** (`:208`).

Production settings: threshold **0.90**, sustain **2** frames (`v8/__init__.py:61-62`). Enabled.

### 2. Trailing trim — moves the end earlier

`postprocess.trim_trailing_sustained_lk` (`postprocess.py:213-295`), the mirror image, walking backwards from the end. Same delete-the-whole-reach behaviour if it shrinks below `min_span` (`:293`). Production settings: threshold **0.90**, sustain **2** (`v8/__init__.py:78-80`). Enabled.

The parameter docs for both trims still say "Default 0.60" and "Default 3" and cite calibration on DLC model 3.1 (`postprocess.py:178`, `:181`, `:260`, `:263`). Those are the old values. The values that actually run are the 0.90 / 2 constants in `v8/__init__.py`, which the calling code passes in.

### 3. Apex split — cuts one span into two

`postprocess.apex_split_at_trough` (`postprocess.py:340-458`). It builds a per-frame signal (`:310-337`): the distance from the paw centroid (mean of the four paw points) to the `BOXL` corner, divided by the `BOXL`-to-`BOXR` distance, with every input coordinate smoothed over 5 frames first. Roughly, "how far out is the paw, as a fraction of the apparatus width".

For each span it runs `scipy.signal.find_peaks`. It splits only when all of these hold (`:432-457`):

- two or more peaks, each with prominence ≥ 0.12, at least 4 frames apart;
- the last peak sits before 85% of the way through the span;
- some pair of consecutive peaks at least 2 frames apart has a valley between them at least 0.5 deep, measured as the higher of the two peaks minus the valley (the deepest such valley becomes the cut point);
- both halves would be at least `min_span` (3) frames long.

A span is passed through untouched if its end frame is past the end of the signal, or if the signal over the span is shorter than 3 frames or contains a NaN (`:427-431`).

The cut yields `[start … trough]` and `[trough+1 … end]`. The two halves **touch** — there is no gap frame between them, by construction.

Limits worth writing down:

- **It splits at most once per span.** The two halves are appended and never re-examined, so a span covering three real reaches comes out as two.
- **The halves are never re-trimmed.** The split runs after both trims, so the interior boundaries created here have had no confidence-based cleanup applied.
- The split point is a valley in a paw-position curve — not a paw disappearance, not a velocity reversal.

### Nothing tunes any of this at run time

`span_to_reaches.py:258` calls `detect_reaches_v8(dlc_df)` with no arguments beyond the pose table, so every threshold, trim setting and split setting is left at its module constant. No other code in `src/` references any of those constants. There is no configuration file, no environment variable and no command-line flag that changes reach detection behaviour. Changing any of it means editing `v8/__init__.py`.

---

## Filing reaches into segments

`build_video_reaches` (`span_to_reaches.py:78-227`) does the bookkeeping. The detector produced a flat list for the whole video; the boundaries come from `{video}_segments.json`.

**Where the boundaries come from.** `geometry.load_segments` (`:141-146`) reads the key `boundaries`, falling back to `validated_boundaries`, and otherwise returns an empty list. The outcome stage reads the same file with a *different* reader (`outcomes/core/batch.py:90-104`) that also understands a `segmentation.boundaries[].frame` shape. Every segments file in the working folder uses the plain `boundaries` key, so the two readers currently agree — but a file in that third shape would give the outcome stage 21 boundaries and give reach detection zero, silently.

**Segment convention** (`segments_from_boundaries`, `:269-293`): each boundary frame **starts** a segment, so the number of segments equals the number of boundaries, and the last segment runs to the end of the video. `seg_start` is inclusive, `seg_end` is exclusive — segment *n*'s `end_frame` is exactly segment *n+1*'s `start_frame`, and the last segment's `end_frame` equals `total_frames`. This is the only exclusive end in the pipeline's files. Every other window (reaches, pellets, outcome segments, the features file) is first-and-last-frame inclusive, and a reader turns this one into that convention through `causal_review_io.reach_file_segment_span`. A pillar-tray video's segments file has 21 boundaries, so its reach file has 21 segments. Easy and Flat tray videos are not segmented: their segments file holds the single boundary `[0]`, so the reach file has ONE segment, from frame 0 (inclusive) to `total_frames` (exclusive), covering the whole video.

A reach belongs to the segment containing its **start** frame (`_segment_index_for_span_start`, `:296-312`).

**Frames before the first boundary belong to no segment.** In a typical file the first boundary comes some way into the recording, so the lead-in frames before it are outside every segment. A reach that STARTS before the first cut belongs to no pellet: it is written into the reaches file's own top-level list `reaches_before_first_cut` (its id, frames and apex; no reach number, no segment) and into no segment, and no segment is flagged for it. Every reader that maps reaches to pellets -- outcome detection's input, reach assignment, the ground-truth layering, kinematics and the exports -- also takes such a reach out of segment 1 when it reads an older file (`reach/core/before_first_cut.py`). Never dropped. (Older files attached such a reach to the nearest segment by frame distance -- always segment 1 -- and that segment got `flagged_for_review: true` with `flag_reason: "reach_start_outside_all_segments"`, a flag nothing read; the readers named above handle them.)

**Every reach in a flagged segment gets the review note, not just the orphan.** `:165-168` tests the segment's flag, not the individual span. So most reaches that carry the note can be ordinary reaches that happened to share a segment with an orphan.

One segment entry is emitted per boundary, in order, including empty ones (loop at `:152-195`). The module docstring at `:28-41` explains why: the kinematics step pairs segments by number and holds a video whose reach segments skip a pellet the outcomes file has.

The apex frame is computed here, not by the detector: `compute_reach_apex` (`lib/causal_attribution.py:50-73`) returns the frame inside `[start, end]` where the straight-line distance from `Nose` to `RightHand` is greatest. It uses only `RightHand` (`causal_attribution.py:37`) and **ignores confidence entirely** — a badly-tracked `RightHand` frame can win.

### What breaks if a segment boundary is in the wrong place

Reach *boundaries* are immune: the model never sees segments, so moving a boundary cannot move a start or end frame. Everything about reach *identity* is affected. Concretely:

- **Which segment a reach belongs to.** A boundary that moves across a reach's start frame moves that reach into the neighbouring segment.
- **`reach_num`** — 1-based within the segment (`:160`). Moving one reach renumbers the tail of both segments.
- **`reach_id`** — a single counter incremented as the code walks segments in order (`:146`, `:161`). It is not derived from the frame number, so a membership change renumbers every later reach in the video. This matters because `reach_id` is the handle downstream code uses to name a specific reach.
- **`n_reaches`** on the segment, and the `reaches_per_segment_mean` / `_std` in the summary.
- **`is_first_reach`, `is_last_reach`, `n_reaches_in_segment`** in the kinematics output — the extractor derives all three from the position of the reach within the reach file's per-segment list (`kinematics/core/feature_extractor.py:307-320`). It does not recompute them.
- **`ruler_pixels`, and therefore every millimetre value for that segment.** `compute_segment_geometry` (`geometry.py:89-125`) takes the median x and y of `SABL` and of `SABR` over the middle of *that segment's* frame range (preferring frames where both corners are tracked above 0.9 confidence, but only if more than 50 such frames exist), then measures the distance between those two median points. That distance is the 9 mm reference. The kinematics extractor uses the reach file's per-segment `ruler_pixels` for its conversions (`feature_extractor.py:262`, `:277`, `:292`, `:312`). A wrong boundary changes the frames the scale is measured over.
- **Which pellet the reach is scored against.** The kinematics extractor pairs each reach segment with the outcome segment of the same `segment_num` (`feature_extractor.pair_segments_by_number`); before that it zipped the two lists by position. A pellet one file has and the other does not (other than the one trailing stretch below), or a number listed twice, now holds the video (`KinematicsProblem`) instead of pairing a pellet's reaches with another pellet's outcome. The outcomes file carries no frames, so two files with the same numbers made on different cuts are not caught here; the review gate's stale-output checks catch that.

Two consequences of how the two files are built:

- **The reach file has 21 segments; the outcome file has 20.** The outcome stage builds its own segment list as `[(boundaries[j], boundaries[j+1] - 1) for j in range(len(boundaries) - 1)]` (`outcomes/core/batch.py:197-198`) — one fewer. The pairing leaves out a reach segment numbered after every outcome segment, exactly as the positional zip did, so **reach segment 21, everything after the last boundary, never reaches kinematics.** The outcome and features files have 20 segments, and the reaches that sit in segment 21 are dropped. Segments 1–20 do line up frame-for-frame, so the pairing is correct for them.
- **The outcome stage ignores the reach file's own numbering.** `_extract_reaches` (`outcomes/core/batch.py:107-142`) flattens the nested structure back to a bare list of `(start_frame, end_frame)` pairs, discarding `reach_id`, `reach_num` and `segment_num`.

**Zero-length and one-frame segments.** If the segmenter emits a duplicate or near-duplicate boundary, the resulting segment has no frames to measure geometry over, and `ruler_pixels` comes out as `NaN`. `json.dump` then writes the bare token `NaN`, which the JSON specification does not allow; Python's own `json.load` accepts it, so the pipeline's readers cope, but a strict parser will not. A bare `NaN` can appear in a file, always in `ruler_pixels`: typically in the trailing segment 21 (length 0 or 1, no reaches), or in a segment 1 of length 0, flagged, holding orphan reaches — those reaches are inside the range kinematics reads, and their millimetre conversions are all NaN.

---

## Exactly what lands in `{video}_reaches.json`

Written by `ReachDetector.save_results` (`reach_detector.py:1104-1140`), which serialises the dataclasses with `asdict` and adds two fields.

### Top level

| Field | Filled with |
|---|---|
| `detector_version` | `"8.1.0"` — `v8.VERSION`, re-exported at `span_to_reaches.py:65`, stamped at `:212`. |
| `video_name` | Pose filename stem with everything from `DLC_` onward removed (`:253-255`). If the stem contains no `DLC_`, the whole stem is used. |
| `total_frames` | Row count of the pose table. |
| `boxr_x` | Median `BOXR_x` over the whole video (`geometry.py:128-130`), 1 decimal place. **Nothing reads it.** No code anywhere loads `boxr_x` back out of this file; every consumer that needs the slit edge recomputes it from the pose data. It is a leftover from the old detector, kept because the file shape is fixed. |
| `n_segments` | Number of boundaries. |
| `segments` | Array, one entry per boundary (below). |
| `summary` | Object (below). |
| `detected_at` | Local timestamp, ISO format. |
| `validated` | `false` at write time. Set `true` only by the napari review window (`review_widget.py:2028`). |
| `validated_by`, `validated_at` | `null` at write time; filled by the review window (`:2029-2030`). |
| `corrections_made`, `reaches_added`, `reaches_removed` | **0 from the detector, always.** They are *not* dead fields: the napari review window increments them as a person works — `reaches_added` at `review_widget.py:1854`, `reaches_removed` at `:1896`, `corrections_made` via `_increment_corrections` (`:1904-1906`) called from `_set_reach_start` (`:1697`) and `_set_reach_end` (`:1722`) — and writes them back into the file when the reviewer validates (`:2019-2021`, `:2037-2038`). Nothing in the automated pipeline touches them. |
| `segments_flagged` | Count of segments with `flagged_for_review: true`. |
| `validation_status` | `"needs_review"` as written (`reach_detector.py:1115`, the default of `save_results`). In the watcher, a later unified triage step overwrites it (the watcher's unified triage step in `watcher/orchestrator.py`). |
| `validation_timestamp` | Local timestamp of the write. |

Shipped files also carry a top-level `triage_reason`, added by that later triage step, not by this subsystem.

### `summary`

| Field | Filled with |
|---|---|
| `total_reaches` | Real count. |
| `n_segments` | Real count. |
| `reaches_per_segment_mean` | Real, 1 decimal place. |
| `reaches_per_segment_std` | Real. |
| `mean_duration_frames` | Real. |
| `mean_extent_ruler` | **Always 0.0.** The list it averages is created at `span_to_reaches.py:150` and never appended to — see the comment at `:199`. |

### Per segment

| Field | Filled with |
|---|---|
| `segment_num` | 1-based, in boundary order. |
| `start_frame` | Boundary frame, inclusive. |
| `end_frame` | **Exclusive** — equals the next segment's `start_frame`, or `total_frames` for the last one. |
| `ruler_pixels` | Distance in pixels between the median `SABL` point and the median `SABR` point over the stable middle of the segment (`geometry.py:89-125`), rounded to 1 decimal. This is the 9 mm physical reference. `NaN` when the segment is 0 or 1 frames long. |
| `n_reaches` | Real count. |
| `reaches` | Array (below), sorted by start frame. |
| `flagged_for_review` | `true` only when an orphan reach was attached here. |
| `flag_reason` | `null`, or `"reach_start_outside_all_segments"` in older files (no longer produced). |

### Per reach

Filled with real values:

| Field | Filled with |
|---|---|
| `reach_id` | 1-based, unique across the video, assigned in segment order. |
| `reach_num` | 1-based within the segment. |
| `start_frame` | After trims and split. |
| `end_frame` | Inclusive, after trims and split. |
| `duration_frames` | `end - start + 1` (`span_to_reaches.py:162`). Always ≥ 3; the minimum observed across the shipped reaches is exactly 3. |
| `apex_frame` | Frame of maximum nose-to-`RightHand` distance within the reach. |
| `source` | `"algorithm"` from the detector; `"human_added"` for a reach a person adds in the napari reach review screen (`review_widget.py:1849`). |
| `human_corrected` | `false` from the detector; set `true` by the napari reach review screen when a person moves a boundary (`review_widget.py:1700`, `:1728`); written into this file when that screen saves. |
| `review_note` | `null`, or `"reach_start_outside_all_segments"` for **every** reach in a flagged segment (older files). |

**Written and empty on every reach the current detector produces** — verified across every reach in the shipped `8.1.0` files, where each of these is `null` on every reach:

| Field | Why it is empty |
|---|---|
| `max_extent_pixels` | Set to `None` at `span_to_reaches.py:179`. The comment at `:177-178` says extent "is not cheaply available from the flat detector". |
| `max_extent_ruler` | Same. |
| `confidence` | The dataclass declares it (`reach_detector.py:134`); `build_video_reaches` never passes it. Only the dead v6 detector computes one (`reach_detector.py:756-758`, `:815-817`, `:889-891`). |
| `start_confidence`, `end_confidence` | Same (`reach_detector.py:135-136`). |
| `pose_alignment` | Declared at `reach_detector.py:147` and documented in the comment at `:138-146` as how squarely the mouse faces the slit. The only code that computes it is in the dead `reach_detector_v8.py:206-252`. |
| `original_start`, `original_end` | `null` unless a person moves that end of the reach in the review window (`move_reach_start` / `move_reach_end` in `review_widget.py`); then the algorithm's frame, kept the first time that end moves. |
| `exclude_reason` | `null` from the detector. |
| `exclude_from_analysis` | `false` from the detector; only the review window sets it. |

**Consequence downstream, stated plainly.** The kinematics extractor copies `max_extent_pixels` and `max_extent_ruler` straight through and derives `max_extent_mm` from the ruler value (`kinematics/core/feature_extractor.py:366-367`, `:392-393`). Because all three inputs are `null`, **reach extent is null on every reach produced by the current detector.** A reach from an `8.1.0` reach file has no extent, and a reach from the older `5.3.0` detector has one. So extent is not universally null in the database or in older files — it is null on everything current, and non-null only on legacy rows. Anything that reads reach extent, including the ODC-SCI exporter and `reach_export.py`, is reading a field that stopped being computed when v8 landed.

---

## Failures that are swallowed

- **`reach_detector.py:1121-1133`.** After the file is written, `save_results` tries to record it in the pipeline index. Line `:1127` reads `results.summary.total_reaches`, but `summary` is a plain dict (`span_to_reaches.py:202-209`), so this raises `AttributeError` on every run. The `except Exception: pass` at `:1132-1133` hides it. Verified by calling `save_results` with the index class stubbed out: the JSON is written, and the index records nothing. **The pipeline index is never updated by reach detection.**
- **`reach_detector.py:1136-1140`.** The database sync immediately after is wrapped the same way (`except Exception: pass`), so a sync failure is equally invisible.
- **An empty or unrecognised boundaries list produces an empty result, not an error.** If `load_segments` returns `[]`, `build_video_reaches` has nothing to attach spans to, so the loop at `span_to_reaches.py:126-138` hits the `continue` at `:135` for every span and discards them all. The output is a structurally valid file with zero segments and zero reaches.
- **The watcher paths are better.** Both watcher reach-detection steps in `watcher/orchestrator.py` log the step as failed, mark the video failed, and re-raise.
- **`review_widget.py:1077-1080`** shows the error and prints a traceback.

## Commands and code that do not work

- **`mousereach-triage-reaches` crashes on every v8 file.** `triage.check_anomalies` reads each reach's confidence with `r.get('confidence', 0)` and compares it to 0.30 (`triage.py:144-145`). v8 writes `confidence: null`, so the `0` default never applies and `None < 0.30` raises `TypeError`. Reproduced against a shipped file. The call at `triage.py:216` is not inside a `try`, so the command dies.
- **`mousereach-advance-reaches` crashes immediately.** `cli.py:173` calls `advance_videos(args.input, require_validation=not args.force)`, but `advance.advance_videos` is defined as `(input_dir, output_dir, verbose)` (`advance.py:45-49`). Verified: `TypeError: advance_videos() got an unexpected keyword argument 'require_validation'`. Note also that `advance.py:86` **moves files** with `shutil.move`, which contradicts the single-folder architecture described in the comment at `core/__init__.py:163-165` and in the command's own help text.
- **`_run_detection` in the napari reach widget would run the wrong algorithm — but nothing calls it.** `review_widget.py:1059-1080` instantiates the dead v6 `ReachDetector` and would overwrite `{video}_reaches.json` with v6 output. It is not connected to any button, signal or menu; a repository-wide search for `_run_detection` finds only its own definition (plus an unrelated one in the outcomes widget). It is dead code and should be deleted rather than wired up.
- **`mousereach-review-reaches` is not the napari tool.** `cli.py:180-194` documents it as launching a GUI, but it calls `_review.interactive_review` (`_review.py:157`), a text prompt loop in the terminal. The napari widget is `reach.review_widget:ReachAnnotatorWidget`, reached through the napari plugin manifest (`napari.yaml:40-41`), the dashboard, or the unified review widget.
- **The review window's edits keep `duration_frames` exact.** The detector writes `end - start + 1` (`span_to_reaches.py:162`). Recomputing it as `end - start` when a person moves a start, moves an end or adds a reach would leave every human-touched reach one frame short, so the review window counts both ends (`move_reach_start`, `move_reach_end` in `review_widget.py`). The kinematics no longer trust a stored value either: since extractor 2.6.0 they count each reach's frames themselves (`causal_review_io.reach_duration_frames`).
- **A human-added reach gets a guessed apex.** `review_widget.py:1833` sets `apex_frame` to the midpoint of the new reach, not the nose-to-hand maximum the detector uses. Its `reach_id` is `max_id + 1` (`:1830`), so it does not sit in temporal order with the rest.
- **Ground-truth export keeps the pre-correction frames.** The window records the algorithm's frames as `original_start` / `original_end`, each the first time a person moves that end, whichever end moves first. The export (`gt_reach_record`) writes them as `original_start_frame` / `original_end_frame`, the names the ground-truth readers use. Reading keys of those output names, which nothing writes, would leave both fields `null`. A field stays `null` for an end nobody moved.

## Stale documentation inside the code

Treat these as wrong, not as history:

- `reach/core/__init__.py:13-14` describes the v6 nose-engagement / hand-visibility algorithm as *the* algorithm, and `:54` gives the output's `detector_version` as `"3.3.0"`. Neither has been true since v8 landed.
- `reach/core/reach_detector.py`'s class docstring (`:188-197`) describes the same dead v6 rules.
- `reach_detector_v8.py`'s entire module docstring, including its performance table (`:20-52`), describes code with no callers, and its header line calls it v7.1.0 while `:93` says `7.2.0`.
- `v8/__init__.py:7-10` says the production model was trained on 20 exhaustive videos with holdout precision 84.8% / recall 91.8%. Those are the *previous* model's numbers (v8.0.0, DLC 3.1) and they match that bundle's sidecar exactly. The shipped model is v8.1.0, 37 training videos on DLC 4.0, and no accuracy number for it is recorded anywhere in the tree.
- `v8/__init__.py:118-120` says `model_path` defaults to `v8.0.0_bsw_w0.8.joblib`. It defaults to `v8.1.0_bsw_w0.7_model4.0.joblib` (`:45`).
- The trim parameter docs in `postprocess.py` state 0.60 / 3; production runs 0.90 / 2.
- `postprocess.py:22-24` calls the threshold, merge gap and min span "calibration knobs" with "defaults to be tuned". Nothing tunes them; see above.

## Testing

**No test exercises the v8 detector, its feature builder, or its post-processing.** There is no test for `extract_features`, `probabilities_to_reaches`, either trim, `apex_split_at_trough`, `build_video_reaches`, or the segment-filing rules.

The repository is not untested in general: there are suites for outcomes, segmentation, tray motion and the version simulator. (the `tests/` directory now holds the main suite, about 300 test files.) None of them test the detector's own decisions; the few that import reach code replace the detector with a stand-in. (The reach-matching evaluation suite lived under the improvement subsystem, which has moved out of this repo as an accessory.)

## How performance is measured, when it is measured

`v8/eval.py` is the scoring code for this detector. A detected reach counts as correct only if its start is within 2 frames of a human-marked start **and** its length is within `max(0.5 × human length, 5 frames)` (`eval.py:67-91`). Matching is greedy, closest start first then closest length; each human reach and each detected reach matches at most once (`:93-126`). The summary reports counts of correct / spurious / missed plus the distribution of start and length errors, and deliberately does not report precision, recall or F1 (`:129-166`).

It is called by `v8/train.py:123`. It is not part of the production run — nothing in the pipeline scores a reach file against ground truth.

Note that the improvement framework (now a separate accessory tool) carries a *different* matcher with its own tolerances. The two are not interchangeable; check which one a reported number came from.

---

## filed is final

A new version of this algorithm is used only for videos that are not filed yet. A filed video's
result from this step is final: the version scan never marks a filed video to run this step again,
and a filed video's kinematics-only re-run never recomputes it (if the filed output is missing, that
re-run is given up and the filed results stand). To see how a new version would do on filed videos,
use the evaluation tools (or bring-current without finalize); they write outside Analyzed and never
replace a filed file. See docs/PIPELINE_AS_BUILT.md, "filed is final".
