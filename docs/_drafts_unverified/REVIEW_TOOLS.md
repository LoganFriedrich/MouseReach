# MouseReach Review Tools

Describes: `src/mousereach/review/`, plus the parts outside it that decide when a human is asked and what happens to the answer — `src/mousereach/watcher/review_gate.py`, `src/mousereach/watcher/review_return.py`, `src/mousereach/kinematics/core/feature_extractor.py` (where reviews enter the numbers), `src/mousereach/pipeline/reprocess_to_current.py`, `src/mousereach/triage/gt_resolve.py`, `src/mousereach/config.py` (queue locations), `src/mousereach/napari.yaml` and `pyproject.toml` (how the tools are started), `scripts/backfill_causal_review_spans.py`.

Verified against: 61d98b9 (2026-08-21)

---

## What a "review" is here

The pipeline measures reaching from video. When the algorithms cannot commit to an answer, the video is held out of the final numbers and a person is asked. A review tool is a napari window that shows a stretch of video, asks a question, and writes the answer to a JSON file next to the video's other results.

Two things then have to happen, and they are handled by different code:

1. The answer has to **release** the video — get it out of the holding folder and back through the pipeline.
2. The answer has to **be used** — replace the algorithm's call in the data that kinematics computes from.

These use different matching rules (see [Re-attaching a review](#re-attaching-a-review-after-reprocessing)), and that difference matters.

---

## The tools, and the question each one answers

| Tool | How it starts | The question it exists to answer | What it writes |
|---|---|---|---|
| **Triage Review** | `mousereach-review-tool`; GUI "Review Queues" tab -> Open Triage Review | "The algorithm could not decide one specific thing about this segment. What is the answer?" | `{stem}_causal_review.json` in the video's bundle |
| **Deep Review** | GUI "Review Queues" tab -> Open Deep Review (no console command) | "Segmentation or quality control failed on this whole video. Is it usable, and can it go back through the pipeline?" | `{stem}_causal_review.json` plus `{stem}_deep_review_cleared.json` |
| **Ground Truth Tool** | `mousereach-gt` or `mousereach-unified-review` | "What actually happens in this video, frame by frame, regardless of what the algorithm said?" | `{stem}_unified_ground_truth.json` |
| **Review Tool** | `mousereach-review-legacy`; the "3 - Review Tool" tab of `mousereach` | "The algorithm's output files are wrong — correct them in place." | Overwrites `{stem}_segments.json`, `{stem}_reaches.json`, `{stem}_pellet_outcomes.json` |
| **Triage Clearing + spot-check** | `mousereach-triage-clearing` | Older per-segment version of Triage Review, plus "does the algorithm still agree with a human on cases it was confident about?" | Edits the algo JSONs; writes a ground-truth file; writes `_QC/qc_state.json` and `_QC/qc_drift_log.jsonl` |

Two more pieces in the same folder are not review tools:

- `staging.py` — builds review bundles by running the four algorithms on a video's canonical files (`python -m mousereach.review.staging`, no console command). Used for bulk staging and by the in-tool re-segmentation button.
- `queue_launcher_widget.py` — the "Review Queues" tab in the `mousereach` launcher: counts what is waiting in each queue and opens the right tool in its own napari window (`queue_launcher_widget.py:118-144`).

**Dead or near-dead code in this folder** (see [Declared but does nothing](#declared-but-does-nothing)): `base.py`, the `SavePanel` class in `save_panel.py`, and `unified_widget.py`.

---

## Triage Review (the one used routinely)

`CausalReviewWidget` in `causal_review_widget.py`, launched with `triage_only=True` by `main()` at `causal_review_widget.py:3057`.

### How a video is chosen

`load_pending_queue` (`causal_review_widget.py:2707`) picks a video **at random** from the pool of bundles that still need review, and caches that pool. Each "next video" draws from the cache and only re-scans when the cache runs out (`_load_next_video`, `causal_review_widget.py:2732`). Random is deliberate — it keeps the reviewed set spread across cohorts and dates.

A bundle is in the pool when all of these hold (`_needs_review_pool`, `causal_review_widget.py:2609`; `_bundle_needs_review`, `:2549`):

- it is a directory containing a manifest file,
- segmentation did not fail (confidence <= 0, or an anomaly mentioning "reference quality" — `:2561`),
- it has at least one triaged element (`_bundle_has_triage`, `:2577`),
- its mouse+day session is **not** flagged (see the warning under [Session flag](#the-session-flag-button-does-the-opposite-of-what-it-says)),
- no ground-truth file exists for the video anywhere (ground truth counts as the answer),
- its triaged elements are not already all resolved (`_bundle_reviewed`, `:2490`).

### What counts as a triaged element

Set once, in `triage_status.triaged_segments` (`triage_status.py:70`), and re-implemented identically in the widget (`:2577`) and the worklist builder (`triage_queue.py:152`). A segment is triaged when either:

- the outcome detector wrote `outcome == "triaged"` or `flagged_for_review: true`, or
- the outcome is a *touched* one — `retrieved`, `displaced_sa`, `displaced_outside` — and the reach-assignment file has no reach marked `is_causal` for that segment. The pellet moved, but nothing says which reach moved it.

In triage mode the walk visits **only** these segments (`_visible_indices`, `:1122`). Everything else is skipped.

### What the human is shown and asked

The whole video is loaded once as a decode-on-demand layer; navigation only moves the playhead (`_load_video`, `:487`). The view opens on the relevant reach — the algorithm's causal reach if it has one, otherwise the reach nearest the interaction frame — 25 frames early (`OPEN_LEAD`, `:484`), with a 45-frame pad around the reach (`WINDOW_PAD`). Deep-learning tracking points and the pillar circle are drawn as overlays.

The question panel has one of three shapes, decided by `_segment_triage` (`:1763`):

- **`reach_uncertain`** — the outcome is known; the algorithm just cannot say which reach caused it. Asked: *"Which reach caused it?"* with an outcome-correction dropdown underneath.
- **`outcome_uncertain`** — exactly one candidate reach survives narrowing, so the reach is effectively pinned. Asked: *"What did it do to the pellet?"* plus an optional reach override.
- **`both_uncertain`** — neither is known. Both questions.

The reach list is narrowed by `assignment.v2.triage_reduction.reduce_triaged_segment` to reaches that could not be ruled out as misses; the ruled-out ones are built but hidden behind a "Show all reaches" checkbox (`_make_reach_picker`, `:1814`). There is an explicit "No reach — nothing acted on this pellet" radio button, and choosing `untouched` clears and disables the reach picker so the two answers cannot contradict each other (`_sync_reach_picker_to_outcome`, `:1984`).

Every non-boundary segment also gets an **"Ignore windows"** box — frame ranges where something that is not a reach moved the pellet — and a free-text notes box (`_make_abnormal_ranges_widget`, `:2116`).

Two more buttons: **Flag Session** (`:2465`) and, in deep-review mode only, **Clear -> re-enter pipeline** (`:3020`). There is also a segmentation editor reachable from the panel that lets the reviewer re-cut the video and re-run the algorithms on their cuts (`_reseg_apply`, `:1625`) — that calls `stage_video(..., boundaries_override=...)`, which reads the video and pose from their canonical archive locations, so it fails on a bundle whose video is not archived there (the failure is caught and shown as "Re-segmentation pipeline failed").

Keyboard shortcuts are playback only — space, arrows, speed keys 1-6 (`_setup_keybindings`, `:2914`). There is no key for next/previous segment; those are buttons.

### Two ways an unanswered question becomes an answer

Both are worth knowing before trusting `outcome_source: "human_review"` in the data.

1. **Any navigation records the current segment.** "Save Segment + Next", "Next Segment", and "Prev Segment" all call `_collect_answers()` and store the result (`:2429`, `:2879`, `:2889`). There is no "I did not answer this" state.
2. **`_collect_answers` starts from the algorithm's values.** It initialises `human_outcome = seg["outcome"]` and `human_causal_reach = seg["causal_reach"]` (`:2254-2256`) and only replaces them where the reviewer actively picked something. The reach picker also pre-selects the candidate nearest the interaction frame (`:1900-1909`).

The consequence: clicking past a `reach_uncertain` segment without touching anything writes a record whose `human.outcome` is the algorithm's outcome and whose `human.causal_reach` is either the pre-selected reach or nothing. On an `outcome_uncertain` segment where the reviewer leaves the "-- select outcome --" sentinel alone, `human.outcome` becomes the string `"triaged"`. That record then counts as resolved everywhere (`resolved_segments`, `triage_status.py:98`, only checks that `answers.reviewed` is not `False`), releases the video from the queue, and is stamped into the outcomes as `outcome_source: "human_review"` by `apply_review_overrides` (`causal_review_io.py:446`). A segment can therefore reach kinematics labelled `"triaged"` with a human's name on it.

### What is written

`_save_review` (`:2795`) writes `{stem}_causal_review.json` into the bundle directory (`_review_dir`, `:880`). It contains **one record per segment in the video**, not per reviewed segment: segments the reviewer never visited get a placeholder with `answers: {"reviewed": false}` (`:2818-2836`). It also calls `update_corpus_index` (`causal_review_io.py:678`), which appends to `{NAS}/review_records/causal_review_index.json`; failures there are printed and swallowed.

On reopening, `_load_saved_review` (`:892`) drops the placeholders, restores the answers, and jumps to the first segment that is either unscored or whose **algorithm outcome has changed since the review** — a changed algorithm call is treated as needing another look.

### The record format

Written by `build_segment_record` (`causal_review_io.py:167`), document `schema_version: "1.1"`:

```
segment_num       the segment's number at the time of review
segment_span      {"start": ..., "end": ...} -- the frames the reviewer saw
pellet_num
algo              {outcome, causal_reach, interaction_frame}   as it stood
human             {outcome, causal_reach, is_phantom, agreed}
answers           {triage_kind, outcome_pick/outcome_override, causal_pick,
                   abnormal_ranges, reviewed: false for placeholders}
notes             free text
```

`is_phantom` is always `False` — nothing in the widget ever sets it (`causal_review_widget.py:2257`, `:2350`). `provenance` on the document is filled by `collect_provenance` (`causal_review_io.py:112`), which reads version stamps out of the four algo JSONs and the tracking-model name out of the pose file; every read is wrapped in a bare `except` and silently skipped on failure, so a missing key means "could not read it", not "not applicable".

---

## Deep Review

Same widget, `deep_review=True` (`causal_review_widget.py:198-202`). Differences:

- It reads the `DEEP_REVIEW` queue, not the triage queue.
- `triage_only` is forced off, so **every** segment is walked.
- The **Clear -> re-enter pipeline** button appears. It saves the review, then writes `{stem}_deep_review_cleared.json` into the bundle (`:3020`). That marker is what the watcher looks for.

It has no console command. It is reachable from the "Review Queues" launcher tab (`queue_launcher_widget.py:123`) or from the pipeline dashboard for one named video (`dashboard/widget.py:1488`).

---

## Ground Truth Tool

`GroundTruthWidget(review_mode=False)` — `mousereach-gt`, `mousereach-unified-review`, both to `ground_truth_widget.py:3018`.

Question: what really happens in this video. There is no accept/verify step — **setting a value is the answer** (`ground_truth_widget.py:92-104`). It seeds itself from the algorithm's output, and the human corrects boundary frames, reach start/end frames, outcome classes, interaction frames and outcome-known frames, with per-item comments and per-component "exhaustive" flags meaning "this component is completely labelled for this video".

Two save buttons (`:585-625`):

- **Save Progress** (`_save_progress`, `:1971`) writes everything, including untouched algorithm-seeded rows.
- **Save as Ground Truth** (`_save_ground_truth`, `:1980`) writes a filtered copy containing only items a human actually determined.

Both write `{stem}_unified_ground_truth.json` next to the video (`unified_gt.save_unified_gt`, `unified_gt.py:252`). Ground truth outranks every review downstream, and having one anywhere removes the video from the review queue entirely.

The video dropdown is filled from the pipeline index (`_populate_video_dropdown`, `:652`), capped at 30 entries; if the index is unavailable the dropdown falls back to "Browse for video..." and the error is shown in the status line.

`--algo-dir` on the command line changes where the tool reads a segment's decision window and which information panel it shows. It does **not** change where a review-mode save writes: `_algo_files_dir()` returns the video's own parent directory unconditionally (`:2030`).

---

## Review Tool (edits the algorithm's files)

`GroundTruthWidget(review_mode=True)` — `mousereach-review-legacy`, and the tab the `mousereach` launcher loads as "3 - Review Tool" (`launcher.py:274-286`; the comment above it about a tabbed widget is stale).

One button, "Save & Continue" -> `_save_to_algo_files` (`:2041`). It rewrites the three algorithm JSONs in place with `validation_status: "validated"`, per-boundary correction records, and `human_corrected` / `original_*` fields, then kicks off a background database-refresh check (`_maybe_update_database`, `:2209`).

**Its reach corrections do not reach the data.** It reads the existing reaches out of the nested `segments[].reaches` structure, then writes the corrected reaches to a new **top-level** `reaches` key (`:2185`) and leaves the nested `segments` untouched. The reach detector writes the nested form (`reach/core/reach_detector.py:1084-1099`) and kinematics reads the nested form (`kinematics/core/feature_extractor.py:248`, `:308`). Corrected reach boundaries saved here are therefore invisible to every downstream consumer except the causal review widget, which happens to prefer the flat key when it exists (`causal_review_widget.py:1000-1013`). Outcome and boundary edits in this tool *do* land, because those are written back into the structures that are read.

---

## Triage Clearing and the spot-check pool

`mousereach-triage-clearing` -> `triage_clearing.py:664`. This is the older per-segment clearing walk; the AGENTS.md in this folder still calls it the routine tool and still maps it to `mousereach-review-tool`, which is wrong — that name has pointed at the causal review widget since it was added to `pyproject.toml:107`.

It subclasses the Ground Truth widget and drives a worklist built by `triage_queue.py`. Two sources, chosen by flag:

- `--corpus-root <dir>` (default: the triage queue, or `MOUSEREACH_ROUTINE_ROOT`) — a folder of per-video bundles. Videos whose segmentation failed are held out and printed as a separate "manual re-seg lane" (`triage_queue.py:247`).
- `--algo-dir <dir>` — one flat folder of algorithm outputs (the quarantine layout used by the improvement harness).

For each triaged segment it loads only that segment's frames plus padding (`--pre-pad`/`--post-pad`, default 30 each) and asks the reviewer to mark the causal reach with S and E, set the outcome, and write a note about why the algorithm missed it. Save (`_save_current_triage_segment`, `triage_clearing.py:477`) refuses unless both a causal reach and an outcome are set, then writes, **per segment only**:

- in `_reaches.json`: `flagged_for_review: false`, `triage_cleared: true`, `cleared_by`/`cleared_at`, the causal reach updated with `human_corrected: true` and the note, and **every other reach in the segment marked `exclude_from_analysis: true`**;
- in `_pellet_outcomes.json`: the same clearing fields plus `human_verified: true`, the human's outcome, interaction frame, outcome-known frame and `causal_reach_id`;
- a `{stem}_unified_ground_truth.json` next to the bundle's algo JSONs (`_unified_gt_path_for_video`, `:642`).

Two consequences of that third write are easy to miss. The file it saves is `self.gt`, the *unfiltered* object seeded from the algorithm — rows the human never touched are in it with `determined: false`. Downstream truth-layering skips undetermined rows, so no false values leak into the numbers, but **the file's existence alone** makes `has_gt()` true (`causal_review_io.py:571`), which (a) removes the video from the causal review tool's queue as "already ground-truthed" and (b) in a deep-review bundle counts as a clear signal all by itself (`watcher/review_return.py:112-114`).

`--qc-count N` blends N spot-checks into the same walk (`qc_pool.py`). These are segments the algorithm was **confident** about — a committed outcome, and for touched outcomes a committed causal reach (`iter_passing_segments`, `qc_pool.py:78`) — sampled round-robin across cohort+date strata, never-checked first (`sample`, `:184`). The reviewer gets two buttons instead of the marking UI: *Algo is RIGHT* logs agreement; *Algo is WRONG* logs disagreement **and** sets `flagged_for_review` back on that segment so it re-enters triage (`_flag_segment_for_triage`, `triage_clearing.py:429`). State lives in `<review root>/_QC/qc_state.json` and an append-only `qc_drift_log.jsonl`. `--qc-report` prints the agreement rate and exits without opening napari.

---

## Re-attaching a review after reprocessing

### The problem

A review is a fact about a stretch of video. `segment_num` is not — the segmenter hands out those numbers fresh on every re-cut. Before 2026-08-21, a review of "segment 7" was applied to whatever the new segment 7 turned out to be, and the result was recorded as `outcome_source: "human_review"`, indistinguishable from a real one.

### What was added

`build_segment_record` now stores `segment_span` — the exact `{start, end}` frames the reviewer was looking at — on every record, and the document schema version went to `1.1` (`causal_review_io.py:167`, `:256`; commit 597d449). Reviews written before that were backfilled in place by `scripts/backfill_causal_review_spans.py`, which refuses to write a span unless the neighbouring segmentation still matches the review (same segment count, and every reviewed record's causal reach starting inside the span that number would get); files that fail either check are skipped and listed by name.

`index_review_by_segment` (`causal_review_io.py:337`, commit e2cb9a8) then maps each **current** segment number to the review record that describes it:

- A record with no span is matched by number, and only if that number exists in the current segmentation. That is all that was ever available for those records.
- A record with a span is scored against every current segment by overlap. The score is `overlapping frames / the shorter of the two ranges`.
- The best match must beat the runner-up by at least **0.15** (`MIN_MARGIN`). If it does not, the reviewed stretch has been split across two new segments and the record is **dropped** — a note says so.
- The best match must cover at least **0.5** of the shorter range (`MIN_SPAN_OVERLAP`). Below that the record is **dropped** as describing different footage.
- If two records both want the same current segment, the weaker overlap is dropped.
- Where a record survives but lands on a different number than it was written with, a note records the move.

All notes are returned to the caller and logged at warning level (`truth_resolver.py:109`, `causal_review_io.py:524`). Nothing is written back to the review file; the matching is done fresh every time.

### Where the matching is actually used — and where it is not

Frame matching runs **only on the path that puts a review into the numbers**:

- `truth_resolver.resolve_truth_layers` (`truth_resolver.py:289`), called by the feature extractor (`kinematics/core/feature_extractor.py:220-226`), passes the current segmentation from `reaches_data["segments"]`.
- The extractor's fallback `load_and_apply_review` does the same (`feature_extractor.py:230-237`).

Every decision about whether a video is **finished with review** still matches by segment number alone:

- `triage_status.resolved_segments` (`triage_status.py:98`) — used by the pipeline gate (`watcher/review_gate.py:104`), by the return scan that re-injects videos (`watcher/review_return.py:244`), and by the review tool's own "is this bundle done" check (`causal_review_widget.py:2490`).

So after a re-segmentation, a review whose frames no longer match anything is correctly **dropped** from the outcomes — and still counts as a resolution that releases the video from the queue. The video finishes with the algorithm's original `triaged` call in the data and nothing flagging it. That gap is real at this commit.

### Where a saved review is looked for

`resolve_review_path` (`causal_review_io.py:55`) checks, and returns the most recently modified hit: an explicit directory the caller passes (usually the processing directory), then the **triage** queue's bundle for that video, then the directory holding the canonical video. It never raises; nothing found means the caller no-ops. It does **not** look in the deep-review queue — a deep review only reaches this lookup because the return scan moves the file into the processing directory with the rest of the bundle.

`truth_resolver` looks in its own set of places (`truth_resolver.py:315-317`): the triage queue, the deep-review queue, and a directory passed in.

### Which answer wins

`resolve_truth_layers` resolves **each element independently**, taking the highest layer that has a call (`truth_resolver.py:289`):

```
ground truth  >  deep review  >  triage review  >  algorithm
```

An element no human touched keeps the algorithm's value. Provenance is stamped per segment as `outcome_source` (`algo` / `human_review` / `ground_truth`) and per reach as `reach_source` (`algo` / `ground_truth`), with the algorithm's originals preserved as `algo_outcome` and `algo_causal_reach_id`. Reviews never change reach boundaries or reach existence; only ground truth does that, and only where `reaches.exhaustive` is set does it also delete algorithm reaches it never labelled (`_apply_gt_reaches`, `:249`).

---

## The queues, and how a video leaves one

### Where they are

Set in `config.py:134-136`, derived from `nas_root` in `~/.mousereach/config.json`:

| Setting | Path | Holds |
|---|---|---|
| `Paths.REVIEW_ROOT` | `<nas>/Processing/Review` | parent of both queues; `flagged_sessions.json` and `_QC/` live here |
| `Paths.TRIAGE_REVIEW` | `<nas>/Processing/Review/triage` | per-element questions |
| `Paths.DEEP_REVIEW` | `<nas>/Processing/Review/flagged_for_review` | failed segmentation, failed quality control, escalations |

If `nas_root` is unset, all of these are `None` and the launcher tab reports "queue not configured".

### How a video enters

`run_gate` (`watcher/review_gate.py:191`) runs after reach assignment and before kinematics, and **moves the whole bundle** out of the processing directory into a queue:

1. Ground truth marks the video exhaustively complete -> **clean**, proceed (even if segmentation failed).
2. Segmentation failed, or quality control returned `needs_review` -> **DEEP_REVIEW**.
3. Any triaged segment not resolved by ground truth or a saved review -> **TRIAGE**.
4. Otherwise -> **clean**, kinematics runs.

A queue manifest is written so the review tool can open the bundle in place, and the database state is set to `triage` or `deep_review`; if the normal state transition is rejected, the code forces the state rather than letting disk and database disagree (`:158-176`).

`reprocess_to_current.py:66` stages bundles the same way for the bring-current path, and never clobbers a bundle that already exists.

### How a video leaves

Only `scan_review_queues` (`watcher/review_return.py:225`) moves a bundle out. It runs inside the watcher every 10th poll cycle (`watcher/orchestrator.py:1345`, `:1391`) and handles at most **10 bundles per scan** (`MAX_RETURNS_PER_SCAN`) so returning does not starve the pipeline.

- **Triage**: the bundle leaves when it has triaged elements, all of them are resolved by a saved review, and segmentation has not failed (`:247`).
- **Deep review**: the bundle leaves when `{stem}_deep_review_cleared.json` exists, or a ground-truth file sits in the bundle (`:108`).

Returning moves the bundle's data files (including the review file) into the local processing directory and sets the video back to `processing`, so the pipeline re-runs it and the gate re-checks. It refuses and leaves the bundle alone if the pose file cannot be found, or if the database row cannot be created or set — a clearance is never spent on a run that would fail (`:118-214`).

**The review tool itself never moves anything.** There is no `shutil` call anywhere in `review/`, except the ground-truth migration helper. A bundle sitting in the folder with a complete review stays there until the watcher's return scan picks it up.

**A triage bundle with zero triaged elements never leaves.** The return condition requires `st.has_triage` (`:247`), so a bundle whose triage flags have since been cleared by other means sits in the folder indefinitely.

### The scan-free queue index

`queue_index.py` keeps a SQLite list (`{NAS}/review_records/triage_queue.db`) of videos needing review, so the tool does not have to read every bundle. It is push/pop: `pop` happens in the tool when a video's triaged elements are all resolved (`causal_review_widget.py:2778`, best-effort, failures swallowed).

**`push` is called from exactly one place** — `reprocess_to_current._push_review_index` (`reprocess_to_current.py:108`), the bring-current path. The live watcher gate does not push. That matters because of how the tool reads the index (`_review_pool_paths_unfiltered`, `causal_review_widget.py:2629`): it takes the index rows whose parent is the requested queue folder and, **if there are any, returns only those** — the folder is not scanned at all. So on a machine where bring-current has pushed even one row, bundles routed into the same folder by the watcher are invisible to the review queue until that row is popped. `seed_from_folder` (`queue_index.py:301`) exists to rebuild the index from the folder but is not called anywhere in the codebase.

### The session flag button does the opposite of what it says

The button is labelled "Flag Session (needs review)", its tooltip says it marks every video of that mouse+day as must-be-human-reviewed, and after clicking it the tool tells the user "all its videos need human review" (`causal_review_widget.py:417-424`, `:2465`). `flag_session` writes the session key into `Processing/Review/flagged_sessions.json` (`causal_review_io.py:630`).

The only code that reads that file is `_bundle_needs_review` (`causal_review_widget.py:2553`) and the unused `seed_from_folder`, and what it does is:

```python
if is_session_flagged(stem, root):
    return False   # this bundle does NOT need review
```

Flagging a session **removes** every one of that mouse+day's unreviewed videos from the review queue. Nothing in the gate, the return scan, kinematics or the database ever reads the flag. There is no path by which a flagged session gets more human attention.

### Automatic release from ground truth

`mousereach-resolve-triage-from-gt` (`triage/gt_resolve.py`) lifts triage flags on segments a ground-truth file already answers, writing the same clearing fields the clearing tool writes. The comment in `pyproject.toml:114-117` says it "runs as a step in the normal processing pipeline". It does not — the only caller in the tree is the improvement evaluation harness (`improvement/eval_all.py:167`).

---

## Declared but does nothing

Each of these is code that runs, produces a value, and has no reader.

- **Ignore windows / `abnormal_ranges`.** The reviewer marks frame ranges where something other than a reach moved the pellet; the answer is saved, carried onto the segment by both `apply_review_overrides` (`causal_review_io.py:479`) and `truth_resolver` (`truth_resolver.py:181`) — and read by nothing. There is not one reference to `abnormal_ranges` outside `src/mousereach/review/`. Kinematics never excludes those frames.
- **`exclude_from_analysis` on reaches.** The triage clearing tool marks every non-causal reach in a cleared segment with it (`triage_clearing.py:552-556`), and the ground truth tool sets it too. The feature extractor never checks it; the only consumer in the tree is a column in the ODC-SCI export (`kinematics/analysis/odc_sci_exporter.py:157`). Those reaches still contribute to per-segment kinematics.
- **`causal_review_index.json`.** Written on every save (`causal_review_io.py:678`) so that, per its docstring, "the active-learning loop can bulk-read all reviews". Nothing reads it — no reader exists in `src/` or `scripts/`.
- **`outcomes.exhaustive` in ground truth.** `_seg_overrides_from_gt` computes it and returns it; the caller assigns it to `_exhaustive_out` and never uses it (`truth_resolver.py:328`). Ground-truth outcome overrides apply per determined segment either way, so the flag changes nothing for outcomes. For *reaches*, `exhaustive` is load-bearing and does drop unlabelled algorithm reaches.
- **`is_phantom`** in every review record: always `False`, never set.
- **`base.py`** — `AlgoGTReviewMixin`, `DiffItem`, `DiffSummary`. Imported by `review/__init__.py` and documented in this folder's AGENTS.md as the way to build a review widget. No class in the codebase uses the mixin, and no code constructs a `DiffSummary`.
- **`SavePanel`** in `save_panel.py` — never instantiated. Its sibling `SimpleSavePanel` is the one the three per-step review widgets use.
- **`unified_widget.UnifiedReviewWidget`** — reachable only from the napari plugin menu as "Review Tool (Tabbed)" (`napari.yaml:28`). Its own docstring claims the command `mousereach-review-tool` starts it; that command starts the causal review tool. The launcher's "Review Tool" tab loads `GroundTruthWidget` instead (`launcher.py:280`).

## Documentation that is wrong at this commit

Stated here so nobody rediscovers it the hard way:

- `review/AGENTS.md` calls `TriageClearingWidget` the routine tool and maps it to `mousereach-review-tool`. That command is the causal review widget; the clearing tool is `mousereach-triage-clearing`.
- `queue_index.py:186-190` says "The tool MOVES a fully-reviewed video OUT of the folder". The tool removes the database row only; the watcher moves the folder.
- `triage_clearing.py:14` and `unified_widget.py:17` both claim `mousereach-review-tool`.

---

## Configuration summary

| Setting | Where | Effect |
|---|---|---|
| `nas_root` | `~/.mousereach/config.json` | Root of both queues. Unset -> queues are `None`, tools report "not configured". |
| `MOUSEREACH_ROUTINE_ROOT` | environment | Overrides the corpus root the clearing tool's worklist scans (`triage_queue.py:490`). |
| `MOUSEREACH_TRIAGE_ALGO_DIR` | environment | Overrides the flat algo directory fallback (`triage_queue.py:465`). |
| `CONNECTOME_ROOT` | environment | Base for the two lookups above; defaults to `Y:\LAB_ROOT`. |
| `--pending-dir` | `mousereach-review-tool` | Which queue folder to review. Defaults to the triage queue. |
| `--all-segments` | `mousereach-review-tool` | Walk every segment instead of only triaged ones. |
| `--worklist FILE` | `mousereach-review-tool` | CSV or JSON of `vid` + `segment_num`. Only those videos are offered and only those segments walked — and unlike the normal pool, already-reviewed and ground-truthed videos are included (`causal_review_widget.py:2652`). |
| `--cv` | `mousereach-review-tool` | Use pixel-based pellet localisation to narrow candidate reaches. Off by default because it decodes the video over the network. Without it the narrowing uses the tracking data only (`_get_cv_states`, `:1706`). |
| `--corpus-root` / `--algo-dir` | `mousereach-triage-clearing` | Bundle-per-video layout vs one flat folder. `--algo-dir` wins. |
| `--include-cleared` | `mousereach-triage-clearing` | Put already-cleared segments back in the worklist. |
| `--qc-count N` / `--qc-report` | `mousereach-triage-clearing` | Blend N spot-checks into the session / print the agreement rate and exit. `--qc-count` is ignored outside corpus-root mode. |
| `--pre-pad` / `--post-pad` | `mousereach-triage-clearing`, `mousereach-gt` | Frames of context loaded around the segment. Default 30 for clearing, 0 for the ground truth tool. |
| `preserve_clears` | `staging.stage_video` argument, default `True` | Captures human-cleared segments before re-running the algorithms and re-applies them afterwards, matched by segment number (`clear_guard.py:47`). **Skipped entirely when boundaries change**, because a re-cut renumbers the segments; a cleared segment whose number is gone is reported as skipped, not silently dropped. |
