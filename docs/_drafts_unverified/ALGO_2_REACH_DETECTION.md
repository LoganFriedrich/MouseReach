# Reach detection: finding each attempt

Describes: `src/mousereach/reach/v8/`, `src/mousereach/reach/core/`, `src/mousereach/reach/cli.py`, `compute_reach_apex` in `src/mousereach/lib/causal_attribution.py`, and the three places that call reach detection (`src/mousereach/pipeline/core.py`, `src/mousereach/watcher/orchestrator.py`, `src/mousereach/review/staging.py`)

Verified against: 61d98b9 (2026-08-21)

---

## What this subsystem does, in one paragraph

It takes one video's pose-tracking file (a DeepLabCut `.h5`, which gives an x, y and a confidence number for each of 18 tracked points on every frame) plus a list of segment boundary frames, and it writes `{video}_reaches.json`: a list of reach attempts, each one a start frame and an end frame, filed under the segment it started in. It does not look at the video pixels. It does not decide whether a reach succeeded — that is the outcome detector's job.

The thing that actually decides where reaches are is a trained machine-learning model that scores **every frame independently** with "how likely is it that this frame is inside a reach". Everything else is bookkeeping around that score.

---

## Which code actually runs (and which does not)

There are three complete reach detectors in this tree. **Only one runs.**

| File | Version string | Status |
|---|---|---|
| `reach/v8/` (the package) | `8.1.0` | **This is production.** |
| `reach/core/reach_detector_v8.py` | says `7.2.0` | Dead. Nothing imports it — `span_to_reaches.py:279` mentions it only in a comment. |
| `reach/core/reach_detector.py` | says `6.0.0` | Dead as a detector, but its `Reach` / `SegmentReaches` / `VideoReaches` dataclasses and its `save_results` writer are still used (imported at `span_to_reaches.py:53`). Its one live caller as a detector is the "Run Detection" button in the napari review window (`review_widget.py:1064`) — see the warning below. |

Also dead, because their only caller is the dead `reach_detector.py`: `boundary_refiner.py`, `boundary_polisher.py`, `spatial_refiner.py`. `spatial_refiner.py` is 1008 lines and is explicitly switched off at `reach_detector.py:281` (`self._spatial_refiner = None`).

**Confusing naming to be aware of.** `reach_detector_v8.py` is *not* the v8 detector. It is the old rule-based detector (paw-visibility runs plus a "down-left shoot" velocity test plus a nose-near-the-slit filter). Its whole module docstring — visibility runs, `SHOOT_MAG`, `NOSE_ENG_MIN`, the performance table — describes code that never executes. The real v8 is the `v8/` package and it uses none of those ideas.

### The one live path

```
DLC .h5 + {video}_segments.json
  -> span_to_reaches.detect_video_reaches()        (core/span_to_reaches.py:230)
       -> v8.detect_reaches_v8(dlc_df)             (v8/__init__.py:94)   -> [(start,end), ...]
       -> span_to_reaches.build_video_reaches()    (core/span_to_reaches.py:78)
  -> ReachDetector.save_results()                  (core/reach_detector.py:1104)
  -> {video}_reaches.json
```

Three callers, all reaching the same function:

- `reach/core/batch.py:163` (`process_single`) — used by the production watcher (`watcher/orchestrator.py:976` and `:1847` import it as `reach_single`) and by the `mousereach-detect-reaches` command.
- `pipeline/core.py:588` — the napari/desktop pipeline.
- `review/staging.py:304` — building a review bundle.

---

## How a candidate reach is proposed

### Step 1 — build a feature table

`v8/features.py:extract_features` turns the pose file into **405 numbers per frame**. For each of the 18 tracked points (`features.py:265-273`: `Reference`, the four scoring-area corners `SATL`/`SABL`/`SABR`/`SATR`, the two box edges `BOXL`/`BOXR`, `Pellet`, `Pillar`, the four paw points `RightHand`/`RHLeft`/`RHOut`/`RHRight`, `Nose`, `RightEar`, `LeftEar`, `LeftFoot`, `TailBase`) it computes 14 numbers (`features.py:322-356`):

- raw x, raw y, raw confidence;
- x and y smoothed with a 5-frame centred moving average;
- velocity in x and y, as a centred difference over ±2 frames of the smoothed position;
- acceleration in x and y, the same difference applied again to velocity;
- speed = sqrt(vx² + vy²);
- change in confidence, centred difference over ±2 frames;
- the largest speed anywhere in a ±10-frame window, and in a ±20-frame window;
- the lowest confidence anywhere in a ±10-frame window.

Then it adds the straight-line distance between every pair of points, on smoothed coordinates — 153 pairs (`features.py:359-364`). 18 × 14 + 153 = 405.

Two things worth knowing:

- **Nothing is filtered by anatomy or by confidence.** Distances between two fixed apparatus corners are included even though they are near-constant. That is deliberate (`features.py:247-253`).
- **A missing tracked point is silently replaced by zeros** (`features.py:309-316`). If a pose file were produced by a model with different point names, every feature for the missing points — including all the pairwise distances involving them — becomes 0.0 and the model still returns predictions. There is no error and no warning.

### Step 2 — score every frame

`v8/__init__.py:175-181` loads a joblib bundle and calls `predict_proba`. The bundle is a plain dict with exactly two keys, `model` and `feature_columns`; there is no version, no training record and no pose-model identifier inside it (verified by loading `v8.1.0_bsw_w0.7_model4.0.joblib`).

The model is a **scikit-learn `HistGradientBoostingClassifier`** — a gradient-boosted decision-tree ensemble. It is a plain per-frame binary classifier. It has no memory of neighbouring frames; all temporal context comes from the smoothing, velocity and rolling-window features described above.

Default bundle: `v8/models/v8.1.0_bsw_w0.7_model4.0.joblib` (`v8/__init__.py:45`). Its sidecar `.meta.json` records: trained on DLC model 4.0 (resnet101 shuffle3 @100000), 37 training videos, 1,413,715 training frames, 405 features, `max_iter` 200, `learning_rate` 0.05, `max_depth` 6, `random_state` 42, exhaustively-annotated videos only.

The older bundle `v8.0.0_bsw_w0.8.joblib` is still shipped for use with pose from DLC model 3.1, but nothing selects it automatically — you would have to pass `model_path` by hand, and no production caller does.

**There is no check that the pose file was produced by the pose model the classifier was trained on.** `pipeline_versions.json` declares `dlc_scorer` as `DLC_resnet101_MPSAOct27shuffle3_100000`, but `detect_reaches_v8` never looks at the scorer name embedded in the `.h5`. Running the 4.0-trained model over 3.1 pose data produces confident, wrong-ish answers silently.

**`train.py` cannot reproduce the shipped model.** The bundle metadata records "boundary sample weighting" (`bsw_buffer: 1, bsw_weight: 0.7`) — down-weighting training frames near a reach boundary. `v8/train.py:73-90` applies class-balance weights only; it has no boundary weighting at all. The code that actually produced the model lives in `scripts/restart_phase_b_bsw_retune_w_sweep.py:159-203`. `train.py` also evaluates without any of the post-processing steps below, so numbers it prints are not comparable to production.

### Step 3 — turn scores into spans

`postprocess.probabilities_to_reaches` (`postprocess.py:56`):

1. A frame is "in a reach" if its probability is **strictly greater than** `threshold` (`postprocess.py:81`). Production threshold is 0.5 (`v8/__init__.py:49`).
2. Contiguous in-reach frames form a run (`postprocess.py:97-111`).
3. Runs separated by a gap of `merge_gap` or fewer frames are joined (`postprocess.py:114-125`). **Production `merge_gap` is 0** (`v8/__init__.py:50`), and any two runs are separated by at least one frame, so **no merging ever happens in production.** The parameter exists and has no effect at its shipped value.
4. Runs shorter than `min_span` frames are dropped (`postprocess.py:93`). Production `min_span` is 3, so no reach shorter than 3 frames can exist.

Segments are not consulted at any point here. The model runs over the whole video.

---

## How start and end frames are decided

The raw start is the first frame of a run scored above 0.5; the raw end is the last frame of that run (end is **inclusive**). Three adjustments then run, in this fixed order (`v8/__init__.py:192-217`):

### 1. Leading trim — moves the start later

`postprocess.trim_leading_sustained_lk` (`postprocess.py:139`). It computes `paw_mean_lk`: the plain mean of the four paw points' confidences on each frame (`postprocess.py:128-136`). Walking forward from the start, it deletes a frame only if that frame **and** the next `sustain_n - 1` frames are *all* below the confidence threshold. It stops at the first window containing one confident frame. If fewer than `sustain_n` frames remain before the end, it stops (`postprocess.py:199`). If the survivor is shorter than `min_span`, **the whole reach is deleted** (`postprocess.py:208`).

Production settings: threshold **0.90**, sustain **2** frames (`v8/__init__.py:61-62`). Enabled.

### 2. Trailing trim — moves the end earlier

`postprocess.trim_trailing_sustained_lk` (`postprocess.py:213`), the mirror image, walking backwards from the end. Same delete-the-whole-reach behaviour if it shrinks below `min_span`. Production settings: threshold **0.90**, sustain **2** (`v8/__init__.py:79-80`). Enabled.

The in-code docstrings for both trims still say "Default 0.60 … Default 3" and cite calibration on DLC model 3.1 (`postprocess.py:179-181`, `:261-264`). Those are the old values. The values that actually run are the 0.90 / 2 constants in `v8/__init__.py`, which the calling code passes in.

### 3. Apex split — cuts one span into two

`postprocess.apex_split_at_trough` (`postprocess.py:340`). It builds a per-frame signal: the distance from the paw centroid (mean of the four paw points) to the `BOXL` corner, divided by the `BOXL`-to-`BOXR` distance, everything smoothed over 5 frames (`postprocess.py:310-337`). Roughly, "how far out is the paw, as a fraction of the apparatus width".

For each span it runs `scipy.signal.find_peaks`. It splits when all of these hold (`postprocess.py:432-454`):

- two or more peaks, each with prominence ≥ 0.12, at least 4 frames apart;
- the last peak sits before 85% of the way through the span;
- the deepest valley between two consecutive peaks is at least 0.5 deep;
- both halves would be at least `min_span` (3) frames long.

The cut yields `[start … trough]` and `[trough+1 … end]`. The two halves are **touching** — there is no gap frame between them, by construction.

Limits worth writing down:

- **It splits at most once per span.** A span covering three real reaches comes out as two.
- **The halves are never re-trimmed.** The split runs after both trims, so the interior boundaries created here have had no confidence-based cleanup applied.
- The split point is a valley in a paw-position curve, not a paw-disappearance or a velocity reversal.

### Nothing tunes these at run time

`span_to_reaches.py:258` calls `detect_reaches_v8(dlc_df)` with no arguments beyond the pose table. Every threshold, trim setting and split setting listed above is fixed at its module constant. There is no configuration file, no environment variable and no command-line flag that changes reach detection behaviour. Changing any of it means editing `v8/__init__.py`.

---

## Filing reaches into segments

`build_video_reaches` (`span_to_reaches.py:78`) does the bookkeeping. The detector produced a flat list for the whole video; segments come from `{video}_segments.json`.

Segment convention (`span_to_reaches.py:269-293`): each boundary frame **starts** a segment, so the number of segments equals the number of boundaries, and the last segment runs to the end of the video. `seg_start` is inclusive, `seg_end` is exclusive — segment *n*'s `end_frame` is exactly segment *n+1*'s `start_frame`. Confirmed in shipped output (segment 1: 1992–3832; segment 2: 3832–5669).

A reach belongs to the segment containing its **start** frame (`span_to_reaches.py:296-312`).

**Frames before the first boundary belong to no segment.** In a typical file the first boundary is around frame 2000, so the first ~2000 frames are outside every segment. A reach detected there is an "orphan": it is attached to the nearest segment by frame distance and never dropped, and that segment gets `flagged_for_review: true` with `flag_reason: "reach_start_outside_all_segments"` (`span_to_reaches.py:126-138`).

**Every reach in a flagged segment gets the review note, not just the orphan.** `span_to_reaches.py:165-168` tests the segment's flag, not the individual span. In a 199-file sample of shipped v8.1.0 output, 14 files had a flagged segment and 214 reaches carried the note — most of them ordinary reaches that happened to share a segment with one orphan.

One segment entry is always emitted per boundary, in order, including empty ones. This is load-bearing: the kinematics extractor pairs reach-segments with outcome-segments by position, so a dropped segment would misalign everything after it (`span_to_reaches.py:28-38`).

The apex frame is computed here, not by the detector: `compute_reach_apex` (`lib/causal_attribution.py:50-73`) returns the frame inside `[start, end]` where the distance from `Nose` to `RightHand` is greatest. It uses only `RightHand` (`causal_attribution.py:37`) and **ignores confidence entirely** — a badly-tracked `RightHand` frame can win.

---

## Exactly what lands in `{video}_reaches.json`

Written by `ReachDetector.save_results` (`reach_detector.py:1104`), which serialises the dataclasses and adds two fields.

### Top level

| Field | Filled with |
|---|---|
| `detector_version` | `"8.1.0"` — `v8.VERSION`, stamped at `span_to_reaches.py:65`. |
| `video_name` | Pose filename stem with everything from `DLC_` onward removed. |
| `total_frames` | Row count of the pose table. |
| `boxr_x` | Median `BOXR_x` over the whole video, 1 decimal place. **Nothing in v8 detection uses this.** It is a leftover reference from the old detector, kept because the file shape is fixed. |
| `n_segments` | Number of boundaries. |
| `segments` | Array, one entry per boundary (below). |
| `summary` | Object (below). |
| `detected_at` | Local timestamp, ISO format. |
| `validated` | Always `false` at write time. Set `true` only by the napari review window (`review_widget.py:2028`). |
| `validated_by`, `validated_at` | Always `null` at write time; filled by the review window. |
| `corrections_made`, `reaches_added`, `reaches_removed` | **Always 0. Never incremented anywhere in the codebase.** See "Written and never filled" below. |
| `segments_flagged` | Count of segments with `flagged_for_review: true`. Usually 0. |
| `validation_status` | `"needs_review"` — the default of `save_results`. In the watcher, a later unified triage step overwrites this with its own verdict (`watcher/orchestrator.py:1109-1121`). |
| `validation_timestamp` | Local timestamp of the write. |

### `summary`

| Field | Filled with |
|---|---|
| `total_reaches` | Real count. |
| `n_segments` | Real count. |
| `reaches_per_segment_mean` | Real. |
| `reaches_per_segment_std` | Real. |
| `mean_duration_frames` | Real. |
| `mean_extent_ruler` | **Always 0.0.** The list it averages is created at `span_to_reaches.py:150` and never appended to — see the comment at `:199`. |

### Per segment

| Field | Filled with |
|---|---|
| `segment_num` | 1-based, in boundary order. |
| `start_frame` | Boundary frame, inclusive. |
| `end_frame` | **Exclusive** — equals the next segment's `start_frame`, or `total_frames` for the last one. |
| `ruler_pixels` | Distance in pixels between the `SABL` and `SABR` corners, taken as the median over the stable middle of the segment, preferring frames where both corners are tracked above 0.9 confidence (`geometry.py:89-125`). This is the 9 mm physical reference. |
| `n_reaches` | Real count. |
| `reaches` | Array (below), sorted by start frame. |
| `flagged_for_review` | `true` only when an orphan reach was attached here. |
| `flag_reason` | `null`, or `"reach_start_outside_all_segments"`. |

### Per reach

Filled with real values:

| Field | Filled with |
|---|---|
| `reach_id` | 1-based, unique across the video. |
| `reach_num` | 1-based within the segment. |
| `start_frame` | After trims and split. |
| `end_frame` | Inclusive, after trims and split. |
| `duration_frames` | `end - start + 1`. Always ≥ 3. |
| `apex_frame` | Frame of maximum nose-to-`RightHand` distance within the reach. |
| `source` | Always `"algorithm"` from the detector; `"human_added"` if a person adds one in the review window. |
| `human_corrected` | `false` from the detector; set `true` by the review window when a person moves a boundary. |
| `review_note` | `null`, or `"reach_start_outside_all_segments"` for every reach in a flagged segment. |

**Written and always empty in production output** (confirmed against shipped files):

| Field | Why it is empty |
|---|---|
| `max_extent_pixels` | Set to `None` at `span_to_reaches.py:178`. The comment says extent "is not cheaply available from the flat detector". |
| `max_extent_ruler` | Same. |
| `confidence` | The dataclass declares it (`reach_detector.py:137-140`); `build_video_reaches` never passes it. Only the dead v6 detector ever computed one. |
| `start_confidence` | Same. |
| `end_confidence` | Same. |
| `pose_alignment` | Declared and documented at `reach_detector.py:142-155` as how squarely the mouse faces the slit. The only code that computes it is in the **dead** `reach_detector_v8.py:206-251`. Production always writes `null`. |
| `original_start`, `original_end` | `null` unless a person edits the reach in the review window. |
| `exclude_from_analysis` | Always `false` from the detector; only the review window sets it. |
| `exclude_reason` | Always `null` from the detector. |

**Consequence downstream, worth stating plainly:** the kinematics extractor copies `max_extent_pixels` and `max_extent_ruler` straight through and derives `max_extent_mm` from the ruler value (`kinematics/core/feature_extractor.py:366-393`). Because all three inputs are `null`, **`max_extent_pixels`, `max_extent_ruler` and `max_extent_mm` are null on every reach in every current `_features.json`.** Verified on shipped output. The paw-reach extent measurements in the ODC-SCI exporter and `reach_export.py` are reading these same empty fields.

---

## Failures that are swallowed

- **`pipeline/core.py:610-611`.** The whole outcomes-and-reaches block is wrapped in `except Exception: results.outcome_failed += 1`. Nothing is logged, no traceback is printed, and a reach-detection failure is counted as an **outcome** failure. In this path a crash inside reach detection is invisible.
- **The watcher path is better.** `watcher/orchestrator.py:1963-1967` logs the step as failed, marks the video failed and re-raises.
- **`review_widget.py:1078-1082`** shows the error and prints a traceback.

## Commands that do not work

- **`mousereach-triage-reaches` crashes on every v8 file.** `triage.check_anomalies` reads each reach's confidence with `r.get('confidence', 0)` and compares it to 0.30 (`triage.py:144-145`). v8 writes `confidence: null`, the default never applies, and `None < 0.30` raises `TypeError`. Reproduced against a shipped file. The call at `triage.py:216` is not inside a try, so the command dies.
  Note the same function is called from `pipeline/core.py:597` with a stripped-down dict whose segments carry only `n_reaches` — so there the reach loop never executes, the crash never happens, and the low-confidence check is dead code. In that path the only anomaly checks that can fire are "segment count is not 20 or 21" and "a segment has more than 100 reaches".
- **`mousereach-advance-reaches` crashes immediately.** `cli.py:173` calls `advance_videos(args.input, require_validation=not args.force)`, but `advance.advance_videos` is defined as `(input_dir, output_dir, verbose)` (`advance.py:44-48`) — no such keyword, and `output_dir` is missing. Note also that `advance.py` **moves files** with `shutil.move`, which contradicts the single-folder architecture described in `core/__init__.py:162-166` and in the command's own help text.
- **The review window's "Run Detection" button runs the wrong algorithm.** `review_widget.py:1062-1071` instantiates the dead v6 `ReachDetector` and overwrites `{video}_reaches.json` with v6 output stamped `detector_version: "6.0.0"`. Pressing it silently replaces production results with results from a different detector.
- **Ground-truth export loses the pre-correction frames.** `review_widget.py:2125-2126` reads `original_start_frame` / `original_end_frame`, but the widget writes `original_start` / `original_end` (`:1688`, `:1716`). Those two GT fields are therefore always `null`.
- **`corrections_made` / `reaches_added` / `reaches_removed` are never incremented.** They are initialised to 0 in all three detectors and read in the review window's validation record (`review_widget.py:2016-2021`), so the "N changes" message it shows after validating always says "no changes".

## Stale documentation inside the code

Treat these as wrong, not as history:

- `reach/core/__init__.py` and `reach/core/reach_detector.py` describe the v6 nose-engagement/hand-visibility algorithm as *the* algorithm. It has not run in production since v8 landed.
- `reach_detector_v8.py`'s entire module docstring, including its performance table, describes code with no callers.
- `v8/__init__.py:7-12` says the production model was trained on 20 exhaustive videos with holdout precision 84.8% / recall 91.8%. That is the *previous* model (v8.0.0, DLC 3.1). The shipped model is v8.1.0, 37 training videos on DLC 4.0.
- `v8/__init__.py:119-120` says `model_path` defaults to `v8.0.0_bsw_w0.8.joblib`. It defaults to `v8.1.0_bsw_w0.7_model4.0.joblib` (`:45`).
- The trim docstrings in `postprocess.py` state 0.60 / 3; production runs 0.90 / 2.

## Testing

There are no automated tests for reach detection. `tests/` contains one file, `test_watcher_integration.py`.

## How performance is measured, when it is measured

`v8/eval.py` is the scoring code. A detected reach counts as correct only if its start is within 2 frames of a human-marked start **and** its length is within `max(0.5 × human length, 5 frames)` (`eval.py:67-91`). Matching is greedy, closest-start first; each human reach and each detected reach matches at most once. The summary reports counts of correct / spurious / missed plus the distribution of start and length errors — deliberately not precision, recall or F1 (`eval.py:129-166`). This code is called by the trainer and by the one-off scripts in `scripts/`; it is not part of the production run.
