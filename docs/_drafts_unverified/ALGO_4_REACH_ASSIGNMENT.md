# Reach Assignment — Which Reach Caused the Outcome

Describes: `src/mousereach/assignment/` (`run.py`, `cli.py`, `v1/`, `v2/`), plus the callers that write and read its output file (`src/mousereach/pipeline/run_all.py`, `src/mousereach/pipeline/reprocess_to_current.py`, `src/mousereach/watcher/orchestrator.py`, `src/mousereach/review/staging.py`, `src/mousereach/review/triage_status.py`, `src/mousereach/review/triage_queue.py`, `src/mousereach/pipeline/manifest.py`)

Verified against: 61d98b9 (2026-08-21)

---

## What this stage does

Earlier stages have already produced three things for a video:

- **Segments** — the video cut into stretches, one per pellet the tray presents.
- **Reaches** — every time the mouse pushed a paw out through the slit, with a start frame and an end frame.
- **Outcomes** — for each segment, what happened to that pellet: `retrieved`, `displaced_sa` (knocked off the pillar into the staging area), `untouched`, or `triaged` (the outcome detector could not tell).

A segment usually contains many reaches but only one thing happened to the pellet. This stage decides **which single reach did it**, and writes one row per reach saying so.

That is all it does. It never decides *what* happened to a pellet — that is already fixed by the outcome stage and this code copies it verbatim.

## Which version actually runs

There are two implementations in the tree and they are not both live.

| Path | Version | Where it runs |
|---|---|---|
| `assignment/v2/assign.py` → `assign_reaches_v2` | `2.1.0` (`v2/assign.py:40`) | **Everything in production.** |
| `assignment/v1/assign.py` → `assign_reaches_v1` | `1.0.0` | Only from the `mousereach-assign-reaches` command line tool. |

The command line tool `mousereach-assign-reaches` (declared at `pyproject.toml:122`, implemented at `assignment/cli.py:212`) still calls **v1** (`cli.py:153`). Nothing in the automatic pipeline uses it. If someone runs that command by hand over a processing folder, it will overwrite good v2 files with v1 files that carry `"detector": "assignment_v1"` and a different, weaker set of decisions. Nothing warns them.

Production entry points, all of which run v2:

- `assignment/run.py:127` `assign_reaches_for_video(...)` — the importable one-video step. Called from the watcher (`watcher/orchestrator.py:1080` and `:2010`), the manual full run (`pipeline/run_all.py:93`), and the version-driven reprocess (`pipeline/reprocess_to_current.py:216`).
- `review/staging.py:327` calls `assign_reaches_v2` directly when building a review bundle.

`v1/features.py`, `v1/train.py` and `v1/overrides.py` are a machine-learning classifier from an old development effort. Nothing in `src/` imports them; only four one-off scripts under `scripts/` do. They do not run in any pipeline.

## What it reads

`assign_reaches_for_video` (`run.py:144-158`) reads four files from the processing directory:

- `{video}_segments.json`
- `{video}_reaches.json`
- `{video}_pellet_outcomes.json`
- the DeepLabCut pose file (`.h5`), passed in by the caller

If any of the first three is missing it logs a warning and returns `None` (`run.py:147-154`). No output file is written. See "How failure behaves" below for why that is worse than it sounds.

Two helper functions from the v1 command line module do the joining, and v2 reuses them:

- `cli.py:31` `_segment_bounds_from_segmentation` turns the segments file into `(start_frame, end_frame)` pairs. The production segments file stores a `boundaries` list, so segment *i* runs from `boundaries[i]` to `boundaries[i+1] - 1`, and segment numbers start at 1 (`cli.py:48-55`, `cli.py:73`). **The stretch after the last boundary is not a segment** — reaches there get no segment.
- `cli.py:87` `_reaches_list` flattens the reaches file. Production reaches files are nested (`segments: [{reaches: [...]}]`), so this branch runs and stamps each reach with its `segment_num` (`cli.py:96-106`).
- `cli.py:58` `_segments_with_outcomes` merges the two, keeping `outcome`, `interaction_frame`, `outcome_known_frame` and `flagged_for_review`. **`outcome_known_frame` is merged and then never read by v2.**

## Assigning each reach to a segment

For every reach (`v2/assign.py:273-285`):

1. If the reach already carries a `segment_num`, use it.
2. Otherwise take the reach's **midpoint frame** and find the segment containing it (`v2/assign.py:283-284`).
3. If neither works, the reach is `unassigned`.

The midpoint rule means a reach that straddles a segment boundary lands wholly in whichever segment holds its middle frame.

## How a reach is credited as the cause

Only segments whose outcome is `displaced_sa`, `displaced_outside` or `retrieved` are candidates (`v2/assign.py:298`). For each such segment, two independent picks are computed and they must agree.

### Signal A — the interaction-frame pick

The outcome stage records an `interaction_frame`: the frame at which it believes the paw and the pellet interacted. Signal A picks the reach whose `[start_frame, end_frame]` window contains that frame (`v2/assign.py:68-77`). If no reach's window contains it, Signal A produces nothing.

### Signal B — the displacement pick

Signal B ignores the outcome stage entirely and asks the pose data which reach moved the pellet off the pillar.

First a per-frame **pellet radius** is built (`v2/assign.py:84-101`): the distance from the pellet keypoint to the estimated pillar centre, divided by the pillar radius. A value near 0 means the pellet is sitting on the pillar; large values mean it is far away. Frames where the pellet keypoint's confidence is below `0.5` are set to "unknown".

The pillar centre and radius are estimated per frame from the two bottom staging-area landmarks (`lib/pillar_geometry.py:69`), on a lightly cleaned copy of the pose (`v2/assign.py:250-251`). Note: the cleaning call uses its defaults, which do **not** clean the Pellet keypoint — so the pellet coordinates used here are raw.

Then, for each reach in the segment (`v2/assign.py:131-187`):

- **Before window**: up to 15 frames immediately before the reach starts, never reaching back past the previous reach's end.
- **After window**: up to 30 frames immediately after the reach ends, never reaching forward into the next reach's start.
- Take the median radius in each window.
- If more than 60% of the after-window frames have no confident pellet detection, treat the after value as `6.0` — the pellet vanished, which counts as displaced (`v2/assign.py:169-171`).
- Score = `after − before`. If the before value is `2.5` or more (the pellet was already far from the pillar before this reach), the score is multiplied by `0.3` to push it down (`v2/assign.py:181-186`).
- A reach with an unusable before or after window scores minus infinity.

The highest-scoring reach wins. If every reach scores minus infinity, Signal B produces nothing (`v2/assign.py:189-192`).

The constants `15`, `30`, `0.6`, `6.0`, `2.5`, `0.3` and the `0.5` confidence floor are hard-coded literals. There is no configuration file, environment variable or function argument that changes any of them.

### The agreement test

Both picks must exist, and their frame windows must overlap by at least one frame (`v2/assign.py:199-205`, `:336-343`). If they do, the **Signal A reach** is committed as causal. If they do not overlap, or if either signal produced nothing, the whole segment is triaged (`v2/assign.py:344-350`).

The comment in the code states the reason plainly: absence is not agreement.

## When it declines to credit anything

A segment is left with no causal reach in every one of these cases:

| Situation | Code |
|---|---|
| Outcome stage already said `triaged`, or set `flagged_for_review` | `v2/assign.py:305-307` |
| Outcome is `untouched`, missing, or anything outside the touched set | `v2/assign.py:310-313` (no decision recorded at all) |
| Outcome is touched but `interaction_frame` is missing | `v2/assign.py:315-319` |
| No reach was assigned to the segment | `v2/assign.py:322-324` |
| Signal A found no reach containing the interaction frame | `v2/assign.py:347-350` |
| Signal B could not score any reach | `v2/assign.py:347-350` |
| Both signals fired but picked non-overlapping reaches | `v2/assign.py:344-346` |

There is no confidence score and no threshold. The `0.40` probability threshold described in `assignment/AGENTS.md` belongs to a design that is not in this code — v2 has no probabilities at all.

## Cleaning up inside a triaged segment

A triaged segment would otherwise mark every one of its reaches as needing a human. `v2/triage_reduction.py` re-labels the reaches it can already prove are innocent, so a reviewer sees only the real candidates.

The physics: both `retrieved` and `displaced_sa` require the pellet to leave the pillar. So a reach across which the pellet's on-pillar state did not change cannot be the cause.

Per frame in the segment (`triage_reduction.py:70-137`), the pellet is classified as:

- **on** — detected with confidence at least `0.9` and within 1.0 pillar radii of the pillar centre;
- **off** — detected with confidence at least `0.9` and more than 1.5 pillar radii away;
- **neither** — anything else, including a tracking dropout. A dropout is deliberately *not* read as "off".

Frames where a paw is above the slit line are excluded from all three, so the paw cannot occlude the answer. The thresholds are imported directly from the outcome cascade's stage 21 (`triage_reduction.py:52-61`) so the two cannot drift.

Immediately before and after each reach, up to 10 paw-clear frames are collected (searching at most 30 frames out); at least 3 are required. All of them on → `on`; all off → `off`; mixed or too few → `uncertain` (`triage_reduction.py:140-166`, `outcomes/v6_cascade/stage_21_...py`).

Then (`triage_reduction.py:213-258`):

- `on → on` — the pellet never left. Re-labelled **miss**.
- `off → off` — the pellet was already gone. Re-labelled **miss**, but only if a confident `on → off` departure was already seen earlier in the segment. Without that guard, a single false "off" reading on the real causal reach's before-window would wrongly clear it.
- `on → off` — the departure. Stays **triaged**; this is the candidate.
- Anything uncertain — stays **triaged**.
- Finally, any still-triaged reach that *ends before* the last on-pillar frame preceding the segment's first departure is re-labelled **miss**: the pellet was demonstrably still on the pillar after that reach finished.

If this pass somehow labels every reach a miss, the whole segment is reverted to fully triaged (`triage_reduction.py:344-346`) — the outcome has to be attributable to something.

**The optional computer-vision confirmation is never used in production.** `compute_pellet_states` and `reduce_triaged_segment` accept `cv_state` / `cv_valid` arguments that would recover on-pillar frames the keypoint dropped, and would let `off → off` be trusted unconditionally. `v2/assign.py:376-377` calls `reduce_triaged_segment` without them, so `cv_verified` is always `False` (`triage_reduction.py:332`) and the entire CV branch at `triage_reduction.py:130-136` is dead in every production run.

**The whole reduction pass is wrapped in a bare `except Exception: pass`** (`v2/assign.py:381-382`). If it raises for any reason, the segment stays fully triaged and nothing is logged. There is no way to tell from the output, the logs, or the file whether the pass ran.

## What each per-reach label means

| Label | Meaning |
|---|---|
| `causal_retrieved` | Both signals agreed on this reach, and the segment outcome was `retrieved`. This reach got the pellet to the mouth. |
| `causal_displaced_sa` | Both signals agreed on this reach, and the outcome was `displaced_sa`. This reach knocked the pellet off the pillar. |
| `miss` | Everything else that is decided: a non-causal reach in a committed segment, any reach in an untouched segment, any reach in a segment with no outcome, or a reach the triage-reduction pass proved innocent. |
| `triaged` | A human still has to look at this reach. Either the outcome stage triaged the segment, or the two signals here disagreed — and the reduction pass could not rule this reach out. |
| `unassigned` | The reach could not be placed in any segment. |
| `causal_abnormal_exception` | **Never produced.** See below. |

`causal_abnormal_exception` is listed in the v2 module docstring (`v2/assign.py:21`) and is genuinely produced by v1 (`v1/assign.py:81-82`). In v2 it cannot occur: `abnormal_exception` is not in `TOUCHED_OUTCOMES` (`v2/assign.py:298`), so such a segment gets no decision and all its reaches fall through to `miss` (`v2/assign.py:429-432`). This is currently harmless because the live outcome detector never emits that outcome either — the v6 cascade commits only `untouched`, `retrieved`, `displaced_sa` and `triaged` (`outcomes/v6_cascade/detector.py:250,262`). But the docstring is wrong about what the code can output.

For the same reason, `displaced_outside` never occurs in practice, which makes `_collapse` (`v2/assign.py:47-50`) and the `displaced_outside` entry in `TOUCHED_OUTCOMES` inert. `v2/assign.py:429-430`'s check for a triaged segment is unreachable, because such a segment always already has a decision recorded.

Measured over 400 production files (49,000 reaches) under `MouseReach_Pipeline/`: `miss` 40,423, `triaged` 4,693, `causal_displaced_sa` 3,223, `causal_retrieved` 594, `unassigned` 57. About 7.8% of reaches are credited as causal, which matches the 7.9% in `docs/FIELD_AUDIT.md:49`.

## Exactly what lands in `{video}_reach_assignments.json`

Written by `run.py:167-168` (and `staging.py:333`), pretty-printed with a trailing newline.

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

Each entry in `reaches` (`v2/assign.py:436-450`), in file order, one per reach in the input, always:

| Key | Type | Value |
|---|---|---|
| `reach_id` | int | Copied from the reaches file. Unique within a video, 1-based, may skip numbers. Falls back to the list position if absent. |
| `segment_num` | int or null | The segment this reach was assigned to. `null` only for `unassigned`. |
| `start_frame` | int | Copied unchanged from the reaches file. |
| `end_frame` | int | Copied unchanged. |
| `label` | string | One of the labels above. |
| `is_causal` | bool | `true` exactly when `label` starts with `causal_` (`v2/assign.py:434`). Carries no information the label does not. |
| `segment_outcome` | string or null | The segment's outcome, copied from the outcome stage. `null` for `unassigned`. |
| `segment_ifr` | int or null | The segment's `interaction_frame`, copied from the outcome stage. Not this reach's own anything. |

One optional key:

| Key | When present | Values |
|---|---|---|
| `triage_reason` | Only on `label == "triaged"` reaches, and only when the reduction pass ran and returned a reason (`v2/assign.py:448-449`) | `reach_uncertain` — the outcome is known, only which reach is open. `outcome_uncertain` — one candidate reach remains but what happened is unknown. `both_uncertain` — neither is pinned. (`triage_reduction.py:261-274`) |

Note that `triage_reason` is a property of the whole **segment**, stamped onto every surviving triaged reach in it, not a per-reach judgement.

**No kinematic values appear here.** No velocity, no extent, no apex. This file is a labelling table only.

**Human review never edits this file.** Reviewer answers go into `{video}_causal_review.json`. The only writers of `_reach_assignments.json` are `run.py:168`, `staging.py:333` and the v1 command line tool at `cli.py:160`.

## Who reads it — and who does not

### Nothing in the scientific data path reads `is_causal`

The per-reach feature table and the database never see this file. `pipeline/run_all.py:121-125` shows the handoff: the kinematic extractor is given the pose file, the reaches file, the outcomes file and the human review file. The assignment file is not passed and is not opened. Grepping `kinematics/`, `export/`, `analysis/` and `db/` for `reach_assignments` returns nothing.

`docs/FIELD_AUDIT.md:49` records the consequence: `is_causal` is present on 7.9% of reaches at the stage that computes it, and on **0%** of the features file and **0%** of the database. The same audit lists `label`, `segment_ifr`, `segment_outcome` and `triage_reason` as also produced here and also never carried forward.

Where analysis code does use the word "causal", it re-derives the answer itself and gets a different one:

- `analysis/data.py:419` reads a field called `causal_reach` from the *reaches* file, not from here. That field is only ever set by hand, in a napari validation widget (`kinematics/_reach_outcome_validator.py:580-583`), which is why `docs/FIELD_AUDIT.md` shows it populated on 1.6% of rows.
- `analysis/data.py:1249-1307` `derive_reach_outcomes` re-implements causal attribution from scratch: it takes the reach containing `interaction_frame`, and if none contains it, falls back to the last reach ending before it. That is Signal A alone, with an extra fallback this stage deliberately does not have, and with no agreement test. Its answers can differ from this file's, and nothing reconciles the two.

So the two-signal agreement gate — the thing this stage exists to provide — currently influences nothing downstream of the review queue.

### What does read it

- **Triage routing.** `review/triage_status.py:60-93` and `review/triage_queue.py:96-111` read `is_causal` (and `label`) to answer: does this segment have a committed causal reach? A touched segment with no committed causal reach counts as triaged and sends the video to human review. `review/qc_pool.py:111` uses the same test to build the quality-control pool.
- **The review tool.** `review/causal_review_widget.py:1062`, `:2592` and `review/queue_index.py:157` display the committed reach.
- **Version tracking.** `pipeline/manifest.py:156` reads the top-level `version` key into the processing manifest, so a video can be marked stale when this stage's version changes. The declared current value lives in `MouseReach_Pipeline/pipeline_versions.json` (`"assignment": "2.1.0"`).
- **Archiving.** `archive/supersede.py:52` moves the file aside on reprocess rather than overwriting it.
- **Evaluation.** `improvement/per_reach_sankey_eval.py:448,513`.

## How failure behaves

- **A missing input** → warning logged, `None` returned, no file written (`run.py:147-154`).
- **An exception during assignment** → the watcher catches it and logs a warning (`orchestrator.py:1081-1082`, `:2015-2016`); the video continues through the pipeline. `reprocess_to_current.py:216` re-raises and aborts that video's reprocess. `run_all.py` lets it abort the run and reports it in `result["error"]`.
- **In all of the no-file cases, the safety net does not fire.** `review/triage_status.py:92` only applies the "touched outcome with no committed causal reach" rule when the assignment document is not `None`. So a video with *no* assignment file is treated as having *no* triage from this stage, and can pass the review gate with zero causal attribution recorded. A video that fails this stage looks cleaner than one that succeeded with disagreements.
- **The triage-reduction pass failing** is silent by construction (bare `except: pass`, `v2/assign.py:381-382`).

## Configuration

There is essentially none. Every threshold in both signals and in the triage-reduction pass is a hard-coded literal or an import from the outcome cascade's stage 21 constants. The only behavioural switches are:

| Setting | Where | Effect |
|---|---|---|
| Tray type in the video filename | `pipeline/run_all.py:86-88`, `watcher/orchestrator.py:1062,2004` | Tray types `E` and `F` skip outcome detection **and** this stage entirely. No assignment file is produced for them. |
| `write=True` (default) | `run.py:132` | Set `False` to compute the result and return it without writing the file. |
| `window=10` | `v2/assign.py:218` | **Has no effect.** It is accepted, documented as "kept for API symmetry", and never referenced in the function body. |
| `cv_state` / `cv_valid` | `triage_reduction.py:283-284` | Would enable the computer-vision confirmation. Never supplied by any caller. |
| `lk_threshold=0.5` | `v2/assign.py:87` | The pellet-confidence floor for Signal B. Never overridden by the one caller. |
