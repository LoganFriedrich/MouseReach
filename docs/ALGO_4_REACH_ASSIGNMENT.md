# Reach Assignment — Which Reach Caused the Outcome

Describes: `src/mousereach/assignment/` (`run.py`, `cli.py`, `v1/assign.py`, `v2/assign.py`, `v2/triage_reduction.py`), plus the callers that write and read its output file (`src/mousereach/pipeline/run_all.py`, `src/mousereach/pipeline/reprocess_to_current.py`, `src/mousereach/watcher/orchestrator.py`, `src/mousereach/review/staging.py`, `src/mousereach/review/triage_status.py`, `src/mousereach/review/triage_queue.py`, `src/mousereach/review/qc_pool.py`, `src/mousereach/review/queue_index.py`, `src/mousereach/review/causal_review_widget.py`, `src/mousereach/pipeline/manifest.py`, `src/mousereach/archive/supersede.py`)

Verified against: b65fcf0 (2026-08-23)

---

## What this stage does

Earlier stages have already produced three things for a video:

- **Segments** — the video cut into stretches, one per pellet the tray presents.
- **Reaches** — every time the mouse pushed a paw out through the slit, with a start frame and an end frame.
- **Outcomes** — for each segment, what happened to that pellet: `retrieved`, `displaced_sa` (knocked off the pillar into the staging area), `untouched`, or `triaged` (the outcome detector could not tell).

A segment usually contains many reaches but only one thing happened to the pellet. This stage decides **which single reach did it**, and writes one row per reach saying so.

That is all it does. It never decides *what* happened to a pellet — that is already fixed by the outcome stage, and this code copies it across unchanged.

## Which version actually runs

There are two implementations in the tree and they are not both live.

| Path | Version | Where it runs |
|---|---|---|
| `assignment/v2/assign.py` → `assign_reaches_v2` | `2.1.0` (`v2/assign.py:40`) | **Everything in production.** |
| `assignment/v1/assign.py` → `assign_reaches_v1` | `1.0.0` (`v1/__init__.py:25`) | Nothing. Retained for provenance; the command line tool stopped calling it in 2026-08. |

The command line tool `mousereach-assign-reaches` (declared in `pyproject.toml`, implemented at `assignment/cli.py` `main_batch`) calls the same `assign_reaches_for_video` the pipeline calls, so it also runs **v2** — verified 2026-08-29 by re-running it on an archived video and getting the identical per-reach table. Until 2026-08 it called **v1** instead: running it over a processing folder by hand overwrote the v2 file at the same path with a v1 file carrying `"detector": "assignment_v1"` and a weaker set of decisions — v1 has no agreement test, commits whichever reach contains the interaction frame, and labels every reach in a triaged segment `triaged`. Any `_reach_assignments.json` stamped `1.0.0` dates from that.

Nothing warned about that overwrite at the time, and version tracking would not have caught it either: the processing manifest records each stage's version when the manifest is written (`pipeline/manifest.py:233-236`), and the repair pass that reads versions back off disk only fills in blanks — it never corrects a version already recorded (`manifest.py:376-391`). A manifest saying `2.1.0` can therefore sit next to a file saying `1.0.0`.

Production entry points, all of which run v2:

- `assignment/run.py:32` `assign_reaches_for_video(...)` — the importable one-video step. Called from the watcher (`watcher/orchestrator.py:1080` and `:2010`), the manual full run (`pipeline/run_all.py:93`), and the version-driven reprocess (`pipeline/reprocess_to_current.py:215-216`).
- `review/staging.py:327` calls `assign_reaches_v2` directly when building a review bundle, and writes the same filename at `staging.py:333`.

`v1/features.py`, `v1/train.py` and `v1/overrides.py` are a machine-learning classifier from an old development effort. Apart from `v1/train.py` importing `v1/features.py`, nothing in `src/` imports them; five one-off scripts under `scripts/` do (`restart_phase_d_build_dataset.py`, `restart_phase_d_loocv_gt_reaches.py`, `restart_phase_d_per_reach_sankey.py`, `restart_phase_d_v2_causal_features.py`, `restart_phase_d_v3_overrides.py`). They do not run in any pipeline.

## What it reads

`assign_reaches_for_video` (`run.py:32`) reads three files from the processing directory (`run.py:49-51`) plus a pose file the caller hands it:

- `{video}_segments.json`
- `{video}_reaches.json`
- `{video}_pellet_outcomes.json`
- the DeepLabCut pose file (`.h5`), loaded at `run.py:61`

If any of the first three is missing it logs a warning and returns `None` (`run.py:52-59`). No output file is written. See "How failure behaves" for why that is worse than it sounds.

Three helper functions in the command line module (`assignment/cli.py`, which since 2026-08 itself calls `assign_reaches_for_video`) do the joining, and `run.py:62-63` uses them:

- `cli.py:50` `_segment_bounds_from_segmentation` turns the segments file into `(start_frame, end_frame)` pairs. It accepts two shapes; production files carry a `boundaries` list of frame numbers and no `segments` key, so the second branch runs (`cli.py:67-74`): segment *i* runs from `boundaries[i]` to `boundaries[i+1] - 1`, and segment numbers start at 1 (`cli.py:92`).
- `cli.py:106` `_reaches_list` flattens the reaches file. Production reaches files are nested (`segments: [{reaches: [...]}]`), so the second branch runs (`cli.py:119-128`) and stamps each reach with the `segment_num` of the block it came from.
- `cli.py:77` `_segments_with_outcomes` merges the two by segment number, keeping `outcome`, `interaction_frame`, `outcome_known_frame` and `flagged_for_review`. **`outcome_known_frame` is merged and then never read by v2** — grep it in `v2/assign.py` and there are no hits.

## Assigning each reach to a segment

For every reach (`v2/assign.py:273-285`):

1. If the reach already carries a `segment_num`, look up the segment with that number (`:280-281`).
2. Only if that fails, take the reach's **midpoint frame** and find the segment containing it (`:282-284`).
3. If neither works, the reach is `unassigned`.

Because production reaches files are nested by segment, step 1 almost always succeeds and the midpoint rule almost never runs. The reach-to-segment mapping is therefore **inherited from whichever segmentation the reach detector ran against**, not recomputed here.

## What a wrong segment boundary does to this stage

This stage never recomputes boundaries. It re-derives segment spans from the boundaries list with the same formula the outcome stage uses (`outcomes/core/batch.py:197-198`; `review/staging.py:150` on the review path), so the two agree by construction *as long as both read the same segments file*. Nothing checks that they did. If the segmentation is re-run and the outcomes are not, this stage pairs new frame spans with old outcomes purely by segment number, silently.

Specific consequences of a boundary in the wrong place:

- **Pellet number is passed through, right or wrong.** The `segment_num` written into every output row is the segmenter's ordinal position — its claim about which pellet presentation this is. If the boundaries are offset by one presentation, everything downstream keyed to pellet number inherits the error, and this stage has no means of noticing.
- **Reaches past the last boundary are dropped from the analysis entirely.** The merge builds pairs of consecutive boundaries, so the stretch before the first boundary and the stretch after the last are not segments. The reach detector *does* emit a trailing block: for a 21-boundary file it produces 21 reach blocks while the merge builds 20 segments, and the block-21 reaches match no segment. Checked on 400 paired videos: the set of `unassigned` reaches is exactly the set of reaches in the reach detector's trailing block, every time (101 reaches across those 400). Corpus-wide, 364 of 222,679 reaches (0.16%) are `unassigned`.
- **Signal A can be moved out of reach.** It needs the outcome stage's interaction frame to fall inside a reach that is in this segment. A misplaced boundary can put the real causal reach in the neighbouring segment; Signal A then finds nothing and the segment is triaged.
- **Signal B's windows change.** The before/after windows are bounded by the neighbouring reaches in the same segment and clamped to the segment's own start and end (`v2/assign.py:133-137`, `:148-152`). Adding a reach to a segment or removing one changes its neighbours' windows and can change which reach wins.
- **The triage-reduction pass assumes one pellet per segment.** It reads pellet state only over `[seg_start, seg_end]` (`triage_reduction.py:95`) and anchors on the first confident on-pillar-to-off-pillar departure inside those bounds. The code comment at `triage_reduction.py:231-242` states the assumption outright: a later on-pillar run is the *next* pellet arriving, not this one. A late boundary pulls the next pellet's arrival into the window; an early one cuts the departure out of it.
- **A reach whose frames fall outside its segment is clamped, not rejected.** `triage_reduction.py:327-328` squeezes both ends into the segment; the result reads "uncertain", so the reach stays triaged rather than crashing the pass.

What this stage does **not** produce: `reach_num`, `is_first_reach` and `n_reaches_in_segment` are kinematics fields computed elsewhere from the reaches file. The only segment-identity field this stage emits is `segment_num`. Ordering still matters internally, though — reaches within a segment are sorted by start frame (`v2/assign.py:292-293`) and Signal B's windows are defined against that order.

## How a reach is credited as the cause

Only segments whose outcome is `displaced_sa`, `displaced_outside` or `retrieved` are candidates (`v2/assign.py:298`). For each such segment, two independent picks are computed and they must agree.

### Signal A — the interaction-frame pick

The outcome stage records an `interaction_frame`: the frame at which it believes the paw and the pellet interacted. Signal A picks the first reach whose `[start_frame, end_frame]` window contains that frame (`v2/assign.py:68-77`). If no reach's window contains it, Signal A produces nothing.

### Signal B — the displacement pick

Signal B ignores the outcome stage entirely and asks the pose data which reach moved the pellet off the pillar.

First a per-frame **pellet radius** is built (`v2/assign.py:84-101`): the distance from the pellet keypoint to the estimated pillar centre, divided by the pillar radius. A value near 0 means the pellet is sitting on the pillar; large values mean it is far away. Frames where the pellet keypoint's confidence is below `0.5` are set to "unknown" (`:100`).

The pillar centre and radius are estimated per frame from the two bottom staging-area landmarks, SABL and SABR, on a 3-frame smoothed copy of their positions (`lib/pillar_geometry.py:69-112`; the offset `0.944` and radius fraction `0.10` are at `:31-32`). It is a calculated circle, not the tracked `Pillar` keypoint, so it moves with the tray.

The pose is lightly cleaned first (`v2/assign.py:250-251`), but the cleaning call uses its defaults, and the default for `other_bodyparts_to_clean` is `None` (`lib/dlc_cleaning.py:144`) — so only the four staging-area corners are cleaned and **the pellet coordinates used here are raw**. The triage-reduction pass later in the same stage does the opposite and cleans the pellet explicitly (`triage_reduction.py:101`). The two halves of this stage disagree about that.

Then, for each reach in the segment (`v2/assign.py:131-187`):

- **Before window**: up to 15 frames immediately before the reach starts, never reaching back past the previous reach's end; for the first reach it is clamped to the segment start (`:133-137`).
- **After window**: up to 30 frames immediately after the reach ends, never reaching forward into the next reach's start; for the last reach it is clamped to the segment end (`:147-152`).
- Take the median radius in each window.
- If more than 60% of the after-window frames have no confident pellet detection, the after value is set to `6.0` — the pellet vanished, which counts as displaced (`:169-171`).
- Score = `after − before`. If the before value is `2.5` or more (the pellet was already far from the pillar before this reach), the score is multiplied by `0.3` (`:181-185`). Note that this shrinks the score toward zero in both directions: it demotes a positive score, but it *raises* a negative one.
- A reach scores minus infinity if either median is unusable (`:177-179`) — an empty window, or a before-window with no confident pellet reading at all.

The highest-scoring reach wins. If every reach scores minus infinity, Signal B produces nothing (`:189-192`).

The constants `15`, `30`, `0.6`, `6.0`, `2.5`, `0.3` and the `0.5` confidence floor are hard-coded literals. There is no configuration file, environment variable or function argument that changes any of them.

### The agreement test

Both picks must exist, and their frame windows must overlap by at least one frame (`v2/assign.py:199-205`, `:336-340`). If they do, the **Signal A reach** is committed as causal (`:341-343`). If they do not overlap, or if either signal produced nothing, the whole segment is triaged (`:344-350`). The comment at `:348-349` gives the reason plainly: absence is not agreement.

## When it declines to credit anything

A segment is left with no causal reach in every one of these cases:

| Situation | Code |
|---|---|
| Outcome stage already said `triaged`, or set `flagged_for_review` | `v2/assign.py:305-308` |
| Outcome is `untouched`, missing, or anything outside the touched set | `v2/assign.py:310-313` (no decision recorded at all) |
| Outcome is touched but `interaction_frame` is missing | `v2/assign.py:315-319` |
| No reach was assigned to the segment | `v2/assign.py:322-325` |
| Signal A found no reach containing the interaction frame | `v2/assign.py:347-350` |
| Signal B could not score any reach | `v2/assign.py:347-350` |
| Both signals fired but picked non-overlapping reaches | `v2/assign.py:344-346` |

There is no confidence score and no threshold. `assignment/AGENTS.md:65-70` describes triage as firing when `max(P(causal))` falls below a default of `0.40`, and describes a `would_be_causal_reach_id` field. Neither exists in this code: v2 computes no probabilities and never writes that field.

## Cleaning up inside a triaged segment

A triaged segment would otherwise mark every one of its reaches as needing a human. `v2/triage_reduction.py` re-labels the reaches it can already prove are innocent, so a reviewer sees only the real candidates.

The physics: both `retrieved` and `displaced_sa` require the pellet to leave the pillar. So a reach across which the pellet's on-pillar state did not change cannot be the cause.

Per frame in the segment (`triage_reduction.py:70-137`), the pellet is classified as:

- **on** — detected with confidence at least `0.9` and within `1.0` pillar radii of the pillar centre (`:120-121`);
- **off** — detected with confidence at least `0.9` and more than `1.5` pillar radii away (`:122`);
- **neither** — anything else, including a tracking dropout. A dropout is deliberately *not* read as "off".

A frame where any of the four right-hand keypoints is detected (confidence at least `0.5`) up in the pellet's area — at or above a line one pillar-radius below the pillar centre — counts as neither, so the paw cannot occlude the answer (`:114-118`). All these thresholds are imported directly from the outcome cascade's stage 21 (`triage_reduction.py:52-61`; the values are at `stage_21_causal_reach_via_immediate_on_off_transition.py:62-70` and `:84`) so the two cannot drift.

Immediately before and after each reach, up to 10 paw-clear frames are collected, searching at most 30 frames out, and at least 3 are required. All of them on → `on`; all off → `off`; mixed, or too few, → `uncertain` (`triage_reduction.py:140-166`, implemented at `stage_21_...py:111-158`).

Then (`triage_reduction.py:201-258`):

- `on → on` — the pellet never left. Re-labelled **miss** (`:215-216`).
- `off → off` — the pellet was already gone. Re-labelled **miss**, but only if a confident `on → off` departure was already seen earlier in the segment (`:224`). Without that guard, a single false "off" reading on the real causal reach's before-window would wrongly clear it.
- `on → off` — the departure. Stays **triaged**; this is the candidate.
- Anything uncertain — stays **triaged** (`:228-229`).
- Finally, the code finds the last on-pillar frame that precedes the segment's first departure, and re-labels **miss** any still-triaged reach that *ends before* that frame (`:243-257`): the pellet was demonstrably still on the pillar after that reach finished.

If this pass somehow labels every reach a miss, the whole segment is reverted to fully triaged (`triage_reduction.py:344-346`) — the outcome has to be attributable to something.

**The optional computer-vision confirmation is never used in production.** `compute_pellet_states` and `reduce_triaged_segment` accept `cv_state` / `cv_valid` arguments that would recover on-pillar frames the keypoint dropped, and would let `off → off` be trusted unconditionally. `v2/assign.py:376-377` calls `reduce_triaged_segment` without them, so `cv_verified` is always `False` (`triage_reduction.py:332`) and the CV branch at `triage_reduction.py:130-136` is dead in every production run.

**The whole reduction pass is wrapped in a bare `except Exception: pass`** (`v2/assign.py:381-382`). If it raises for any reason, the segment stays fully triaged and nothing is logged.

You can nevertheless tell from the output file whether it ran. On a non-empty segment, `reduce_triaged_segment` always returns a reason string (`triage_reduction.py:313-315` and `:348`; the only `None` return is for an empty reach list at `:307-308`, which the caller has already skipped at `v2/assign.py:373-374`), and that string is stamped onto every reach that stays `triaged` (`v2/assign.py:448-449`). So **a `triaged` reach with no `triage_reason`, in a file stamped version 2.1.0, means the pass raised for that segment.** Across the corpus this never happens: of 20,507 `triaged` reaches in 2.1.0 files, all 20,507 carry a reason. The 2,012 reasonless triaged reaches on disk are all in older 2.0.0 files, written before the pass existed.

## What each per-reach label means

| Label | Meaning |
|---|---|
| `causal_retrieved` | Both signals agreed on this reach, and the segment outcome was `retrieved`. This reach got the pellet to the mouth. |
| `causal_displaced_sa` | Both signals agreed on this reach, and the outcome was `displaced_sa`. This reach knocked the pellet off the pillar. |
| `miss` | Everything else that is decided: a non-causal reach in a committed segment, any reach in an untouched segment, any reach in a segment with no outcome, or a reach the triage-reduction pass proved innocent. |
| `triaged` | A human still has to look at this reach. Either the outcome stage triaged the segment, or the two signals here disagreed — and the reduction pass could not rule this reach out. |
| `unassigned` | The reach could not be placed in any segment. |
| `causal_abnormal_exception` | **Never produced.** See below. |

`causal_abnormal_exception` is listed in the v2 module docstring (`v2/assign.py:21`) and is genuinely produced by v1 (`v1/assign.py:54-55`). In v2 it cannot occur: `abnormal_exception` is not in `TOUCHED_OUTCOMES` (`v2/assign.py:298`), so such a segment gets no decision and all its reaches fall through to `miss` (`v2/assign.py:429-432`). This is currently harmless because the live outcome detector never emits that outcome either — grepping `committed_class=` across `outcomes/v6_cascade/` yields only `untouched`, `retrieved` and `displaced_sa` (the three variable-class stages resolve to `displaced_sa` or `retrieved` at `stage_21_...py:414,478` and `stage_24_transition_triangulation.py:253,255`). But the docstring is wrong about what the code can output.

`displaced_outside` is in the same position. `_collapse` (`v2/assign.py:47-50`) folds it into `displaced_sa`, and it is listed in `TOUCHED_OUTCOMES`, but the v6 cascade cannot emit it. It exists on disk only in files written by the older, legacy outcome detector (16 segments across 2,703 legacy-format `_pellet_outcomes.json` files). No assignment file in the corpus was built from one: the only `segment_outcome` values that appear across all 2,015 assignment files are `retrieved`, `displaced_sa`, `untouched`, `triaged` and null.

`v2/assign.py:429-430`'s check for a triaged segment is unreachable, because a segment with `outcome == "triaged"` or `flagged_for_review` always already has a decision recorded at `:305-307`, so control never reaches the `else` branch for it.

## What the corpus actually contains

Measured 2026-08-23 over 2,015 `*_reach_assignments.json` files (222,679 reaches) under `MouseReach_Pipeline`. 1,651 files are at the current version 2.1.0; 364 are older 2.0.0 files.

Labels, all files: `miss` 182,379 · `triaged` 22,519 · `causal_displaced_sa` 14,534 · `causal_retrieved` 2,883 · `unassigned` 364. That is 17,417 causal reaches, 7.8% of the total, consistent with the 7.9% in `docs/FIELD_AUDIT.md`.

Triage reasons, 2.1.0 files only: `both_uncertain` 14,758 · `reach_uncertain` 5,362 · `outcome_uncertain` 387.

Per segment, restricted to the 1,651 current-version files and to segments that have at least one reach in them: 13,329 `displaced_sa`, 2,793 `retrieved`, 3,621 `untouched`, 1,991 already triaged by the outcome stage. Of the 16,122 touched segments, **15,448 (95.8%) got a committed causal reach and 674 (4.2%) did not** — the two signals disagreed, or one of them produced nothing.

## Exactly what lands in `{video}_reach_assignments.json`

Written by `run.py:72-73` (and `staging.py:333`), pretty-printed with a trailing newline.

```json
{
  "video_id": "20250624_CNT0110_P3",
  "detector": "assignment_v2",
  "version": "2.1.0",
  "n_reaches": 385,
  "reaches": [ ... ]
}
```

Top level (`v2/assign.py:452-458`) — four keys plus the list. There is no timestamp, no record of which input files were read, and no counts by label.

Each entry in `reaches` (`v2/assign.py:436-445`), in file order, one per reach in the input, always:

| Key | Type | Value |
|---|---|---|
| `reach_id` | int | Copied from the reaches file. Unique within a video, 1-based, and it skips numbers. Falls back to the list position if absent (`:390`). |
| `segment_num` | int or null | The segment this reach was assigned to. `null` only for `unassigned`. |
| `start_frame` | int | Copied unchanged from the reaches file. |
| `end_frame` | int | Copied unchanged. |
| `label` | string | One of the labels above. |
| `is_causal` | bool | `true` exactly when `label` starts with `causal_` (`:434`). Carries no information the label does not. |
| `segment_outcome` | string or null | The segment's outcome, after `displaced_outside` is folded into `displaced_sa` (`:443`). `null` for `unassigned`. |
| `segment_ifr` | int or null | The segment's `interaction_frame`, copied from the outcome stage. Not this reach's own anything, and it appears on every reach in the segment, misses included. |

One optional key:

| Key | When present | Values |
|---|---|---|
| `triage_reason` | Only on `label == "triaged"` reaches, and only when the reduction pass returned a reason (`v2/assign.py:448-449`) | `reach_uncertain` — the segment's outcome is committed, only which reach is open. `outcome_uncertain` — one candidate reach remains but what happened is unknown. `both_uncertain` — neither is pinned. (`triage_reduction.py:261-274`) |

`triage_reason` is a property of the whole **segment**, stamped onto every surviving triaged reach in it, not a per-reach judgement. `reach_uncertain` does not distinguish a two-signal disagreement from a cascade segment that carried a committed outcome and a review flag; both take that branch.

**No kinematic values appear here.** No velocity, no extent, no apex. This file is a labelling table only.

**Human review never edits this file.** Reviewer answers go into `{video}_causal_review.json` (`review/causal_review_io.py:278`). The only writers of `_reach_assignments.json` anywhere in `src/` are `run.py:73` and `staging.py:333` (the command line tool writes through `run.py`). Rebuilding a review bundle re-runs the whole chain and overwrites it (`staging.py:298-333`).

## Who reads it — and who does not

### Nothing in the scientific data path reads `is_causal`

The per-reach feature table and the database never see this file. `pipeline/run_all.py:121-125` shows the handoff: the kinematic extractor is given the pose file, the reaches file, the outcomes file and the human review file. The assignment file is not passed and is not opened. Grepping `kinematics/`, `export/`, `analysis/` and `sync/` for `reach_assignments` or `ASSIGN_SUFFIX` returns nothing.

`docs/FIELD_AUDIT.md` records the consequence in its "Lost In Transit" table (heading at line 41; at this commit the `is_causal` row is line 49): `is_causal` carries information on 7.9% of reaches at the stage that computes it, and the features and database columns both show `-`. In that table `-` means the field does not exist in that layer at all — as distinct from the explicit `0.0%` used for fields that do exist and are always empty (e.g. `flag_reason`, line 47). The same table lists `label`, `segment_ifr` and `triage_reason` in the same position.

`segment_outcome` is the one exception, and it is a name collision rather than a hand-off. The audit shows its database column at 100.0%, but that column is filled by `sync/database.py:569` from the segment outcome carried in the *features* file, not from this file.

Where analysis code does use the word "causal", it gets its answer from somewhere else:

- `analysis/data.py:419` reads a field called `causal_reach` and calls the resulting column `is_causal`. That line is inside `load_features_from_json` (`data.py:354`), so it comes from the **features** file. The reaches loader (`data.py:280-353`) never touches it.
- That `causal_reach` field is set by the kinematics stage, not here: `kinematics/core/feature_extractor.py:379-382` sets it on the reach whose id equals the outcome document's `causal_reach_id`. The v6 cascade never writes `causal_reach_id` (`outcomes/v6_cascade/detector.py:247-268` has no such key). It arrives from a human causal review applied in memory at extraction time (`review/causal_review_io.py:496-499`, reached via `feature_extractor.py:222-237`, with the review path supplied by `run_all.py:120`), or from a ground-truth overlay (`review/truth_resolver.py:173-175`). A separate offline napari widget also writes it (`kinematics/_reach_outcome_validator.py:579-584`). `docs/FIELD_AUDIT.md` shows `causal_reach` populated on 1.6% of rows, which is roughly the share of segments a human has reviewed.
- `analysis/data.py:1249` `derive_reach_outcomes` re-implements causal attribution from scratch: it takes the reach containing `interaction_frame`, and if none contains it, falls back to the last reach ending before it (`:1286-1296`). That is Signal A alone, with an extra fallback this stage deliberately does not have, and with no agreement test. It is called from `analysis/cli.py:188`. Its answers can differ from this file's, and nothing reconciles the two.

So the two-signal agreement gate — the thing this stage exists to provide — currently influences nothing downstream of the review queue.

### What does read it

- **Triage routing.** `review/triage_status.py:59-67` collects the segment numbers that have a committed causal reach, using `is_causal` and `segment_num` only — it never looks at `label`. `triaged_segments` (`:70-95`) then treats a touched segment with no committed causal reach as triaged, which sends the video to human review through the gate (`watcher/review_gate.py:83`).
- **The worklist builder.** `review/triage_queue.py:96-111` `_causal_segments` accepts either `is_causal` or a `label` starting with `causal`, so it is a slightly wider test than `triage_status`'s. It flags any touched segment not in that set (`:203-205`, `:213`). `review/qc_pool.py:106-112` uses the same wider test to build the quality-control pool.
- **The review tool.** `review/causal_review_widget.py:878` loads the file, `:996-999` indexes it by segment, `:1059-1068` picks out the committed reach for display; `:2580-2581` and `:2639-2641` use it to decide whether a bundle still has anything to review. `review/queue_index.py:156-158` does the same for the queue listing.
- **Version tracking.** `pipeline/manifest.py:156` reads the top-level `version` key into the processing manifest, so a video can be marked stale when this stage's version changes. The declared current value lives in `MouseReach_Pipeline/pipeline_versions.json` (`"assignment": "2.1.0"`).
- **Archiving.** `archive/supersede.py:49-52` moves the file aside on reprocess rather than overwriting it.
- **Auditing and evaluation.** `pipeline/field_audit.py:61`. (Per-reach Sankey evaluation lives in the improvement accessory tool, outside this repo.)

## How failure behaves

- **A missing input** → warning logged, `None` returned, no file written (`run.py:52-59`).
- **An exception during assignment, under the watcher** → caught and logged as a warning; the video continues through the pipeline (`orchestrator.py:1081-1082`, `:2015-2016`). On the second path the "assignment completed" step log at `:2013-2014` is skipped and no "failed" step is recorded either, so the watcher database shows the step started and never finished.
- **Under `reprocess_to_current`** → the exception is caught along with the other algorithm steps, recorded in `summary["error"]`, and the function returns without writing a manifest (`reprocess_to_current.py:217-219`). It is not re-raised.
- **Under `run_all`** → there is no local `try`, so it reaches the outer handler, which logs it and returns it in `result["error"]` (`run_all.py:139-143`).
- **In all of the no-file cases, the safety net does not fire.** `review/triage_status.py:93` applies the "touched outcome with no committed causal reach" rule only when the assignment document is not `None`. A video with *no* assignment file therefore contributes no triage from this stage and can pass the review gate with zero causal attribution recorded. **A video that fails this stage looks cleaner than one that succeeded with disagreements.** The worklist builder behaves the opposite way — `_causal_segments(None)` returns an empty set and `triage_queue.py:203-205` does not check for a missing document, so there every touched segment is flagged. The two tools disagree about a missing file.
- **The triage-reduction pass failing** is silent in the logs by construction (`v2/assign.py:381-382`), though it is visible in the output file as a `triaged` reach with no `triage_reason`.

## Configuration

There is essentially none. Every threshold in both signals and in the triage-reduction pass is a hard-coded literal or an import from the outcome cascade's stage 21 constants. The only behavioural switches are:

| Setting | Where | Effect |
|---|---|---|
| Tray type in the video filename | `pipeline/run_all.py:86-89`, `watcher/orchestrator.py:1062,1971` | Tray types `E` and `F` skip outcome detection **and** this stage entirely. No assignment file is produced for them. `reprocess_to_current.py:215-216` has no such check and runs the stage regardless. |
| `write=True` (default) | `run.py:37` | Set `False` to compute the result and return it without writing the file. |
| `window=10` | `v2/assign.py:218` | **Has no effect.** It is accepted, documented as "kept for API symmetry" (`:238-240`), and never referenced in the function body. |
| `cv_state` / `cv_valid` | `triage_reduction.py:283-284` | Would enable the computer-vision confirmation. Never supplied by any caller. |
| `lk_threshold=0.5` | `v2/assign.py:87` | The pellet-confidence floor for Signal B. Never overridden by the one caller. |

---

## Contested claims

This document was written from the source, then checked by a second reader
whose job was to disprove it. The statements below are ones they disputed and
that were not resolved. Neither side is authoritative: spot-checking found the
checker wrong at least once. **Do not rely on anything listed here without
opening the code yourself.** Everything not listed survived two passes.

- **Configuration table: "`cv_state` / `cv_valid` | `triage_reduction.py:283-284` | Would enable the computer-vision confirmation. **Never supplied by any caller.**" — and the body claim "**The optional computer-vision confirmation is never used in production.** ... so `cv_verified` is always `False` (`**
  - disputed because: A caller does supply them. The napari causal review tool passes both arguments into `reduce_triaged_segment`, and it is user-facing production software with a documented flag to switch it on. When that flag is set, `cv_verified` is True, so the `off -> off` branch at triage_reduction.py:224 trusts the state unconditionally, and the CV branch at triage_reduction.py:130-136 executes. What IS true is
- **Output field table: "`reach_id` | int | Copied from the reaches file. Unique within a video, 1-based, **and it skips numbers**."**
  - disputed because: It does not skip numbers. Reach ids are a contiguous run starting at 1 in every assignment file in the corpus, without exception. The rest of the row is correct (copied from the reaches file, unique, 1-based, falls back to list position at v2/assign.py:390 — and that fallback never fires in the corpus, since no file has a 0-based minimum). The "skips numbers" clause should be cut, or replaced with
- **Header: "Verified against: `eda9d78` (branch `master`, 2026-08-21; working tree clean and identical to it)."**
  - disputed because: `eda9d78` is no longer the tip of master, and the working tree is not clean. This is the one line a reader uses to decide whether the rest of the document is current, so it has to be right. Nothing in the body breaks — I checked both files the newer commit touched and every cited line still resolves — but the provenance line needs updating to `4c54e46` (and the dirty-tree claim dropped or re-check
- **"grepping `committed_class=` across `outcomes/v6_cascade/` yields only `untouched`, `retrieved` and `displaced_sa` (**the three variable-class stages** resolve to `displaced_sa` or `retrieved` at `stage_21_...py:414,478` and `stage_24_transition_triangulation.py:253,255`)"**
  - disputed because: There are two variable-class stages, not three. The four cited line numbers are all correct and the overall conclusion (the v6 cascade cannot emit `abnormal_exception` or `displaced_outside`) is correct — only the count is wrong. Two further nits in the same sentence: the third `committed_class=committed_class` site is a calibration harness, not a cascade stage, and the grep also yields a `None` (
- **"**`outcome_known_frame` is merged and then never read by v2** — grep it in `v2/assign.py` and there are no hits."**
  - disputed because: The bolded claim is right — nothing in the executable code reads it — but the verification instruction handed to the reader is wrong. A grep returns one hit. A reader who follows the instruction, gets a hit, and cannot immediately tell it is a docstring will lose confidence in the whole document. It should say the only mention is in the function's docstring at line 230, and that no code line reads

---

## Update 2026-08-23: is_causal is consumed now

The headline defect this document described -- nothing downstream reads
`is_causal` -- is fixed at extractor 2.1.0: kinematics loads the assignment
file and carries the credited reach into the features (and from there the
database), beneath the human-review and ground-truth layers. An assignment
failure in the watcher now also writes a `failed` row to the step audit table
on both machine roles, so it is countable; it previously left only a
started/completed count mismatch. The gate still does not hold a video whose
assignment never ran -- deliberately, because 1,233 older videos have no
assignment file and would flood triage on their next re-gate. That decision is
recorded in UNFINISHED.md.
