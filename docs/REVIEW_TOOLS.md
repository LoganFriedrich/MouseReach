# MouseReach Review Tools

Describes: `src/mousereach/review/`, plus the code outside it that decides when a person is asked and what happens to the answer — `src/mousereach/watcher/review_gate.py`, `src/mousereach/watcher/review_return.py`, `src/mousereach/watcher/orchestrator.py`, `src/mousereach/kinematics/core/feature_extractor.py`, `src/mousereach/pipeline/run_all.py`, `src/mousereach/pipeline/reprocess_to_current.py`, `src/mousereach/triage/gt_resolve.py`, `src/mousereach/config.py`, `src/mousereach/launcher.py`, `src/mousereach/napari.yaml`, `pyproject.toml`, `scripts/backfill_causal_review_spans.py`.

Verified against: b65fcf0 (2026-08-23)

---

## What a "review" is here

The pipeline measures reaching from video. When the algorithms cannot commit to an answer, the video is held out of the final numbers and a person is asked. A review tool is a napari window that shows a stretch of video, asks a question, and writes the answer to a JSON file that sits with the video's other results.

Two separate things then have to happen, and different code does each:

1. The answer has to **release** the video — get it out of the holding folder and back through the pipeline.
2. The answer has to **be used** — replace the algorithm's call in the data the kinematics step computes from.

These two use different matching rules, they look in different places, and right now they disagree. **Read [Reviews mostly do not reach the numbers](#reviews-mostly-do-not-reach-the-numbers) before trusting any statement that human review is applied.**

---

## The tools, and the question each one answers

| Tool | How it starts | The question it exists to answer | What it writes |
|---|---|---|---|
| **Triage Review** | `mousereach-review-tool`; GUI "Review Queues" tab -> Open Triage Review | "The algorithm could not decide one specific thing about this segment. What is the answer?" | `{stem}_causal_review.json` in the video's bundle |
| **Deep Review** | GUI "Review Queues" tab -> Open Deep Review (no console command) | "Segmentation or quality control failed on this whole video. Is it usable, and can it go back through the pipeline?" | `{stem}_causal_review.json` plus `{stem}_deep_review_cleared.json` |
| **Segmentation Fixer** | `mousereach-fix-segmentation` | "Are this video's segment cuts in the right places?" — nothing else | Overwrites `{stem}_segments.json` with the corrected cuts |
| **Ground Truth Tool** | `mousereach-gt` or `mousereach-unified-review` | "What actually happens in this video, frame by frame, regardless of what the algorithm said?" | `{stem}_unified_ground_truth.json` |
| **Review Tool** | `mousereach-review-legacy`; the "3 - Review Tool" tab of `mousereach` | "The algorithm's output files are wrong — correct them in place." | Overwrites `{stem}_segments.json`, `{stem}_reaches.json`, `{stem}_pellet_outcomes.json` |
| **Triage Clearing + spot-check** | `mousereach-triage-clearing` | Older per-segment version of Triage Review, plus "does the algorithm still agree with a person on cases it was confident about?" | Edits the algorithm JSONs; writes a ground-truth file; writes `_QC/qc_state.json` and `_QC/qc_drift_log.jsonl` |

Command names come from `pyproject.toml:106`, `:113`, `:115-118`.

Three more files in `review/` are worth naming, and none of them is a review tool:

- `staging.py` — builds a review bundle by running the four algorithms on a video's canonical files (`stage_video`, `staging.py:213`). No console command; used for bulk staging and by the in-tool re-segmentation button.
- `queue_launcher_widget.py` — the "Review Queues" tab in the `mousereach` launcher (`launcher.py:367-372`). It counts what is waiting in each queue (`:88`, `:105-115`) and opens the right tool in its own napari window (`:118`, `:123`, `:128`).
- `comparison_panel.py` — a shared side-by-side panel used by the outcome step's own review widget (`outcomes/review_widget.py:481-487`), not by anything in this folder.

**Dead or near-dead code in this folder** (see [Declared but does nothing](#declared-but-does-nothing)): `base.py`, the `SavePanel` class in `save_panel.py`, and `unified_widget.py`.

---

## Triage Review (the one used routinely)

`CausalReviewWidget` in `causal_review_widget.py`, launched with `triage_only=True` by `main()` at `causal_review_widget.py:3106`.

### How a video is chosen

`load_pending_queue` (`:2756`) picks a video **at random** from the pool of bundles that still need review, and caches that pool (`:2771-2775`). Each "next video" draws from the cache, and the cache is only rebuilt when it runs out (`_load_next_video`, `:2781`). Random is deliberate — it keeps the reviewed set spread across cohorts and dates.

A bundle is in the pool when all of these hold (`_needs_review_pool`, `:2658`; `_bundle_needs_review`, `:2598`):

- it is a directory containing a manifest file,
- segmentation did not fail — confidence at or below 0, or an anomaly mentioning "reference quality" (`_segmentation_failed`, `:2610`),
- it has at least one triaged element (`_bundle_has_triage`, `:2626`),
- its mouse+day session is **not** flagged (`:2602-2603`; see [the session flag](#the-session-flag-button-does-the-opposite-of-what-it-says)),
- no ground-truth file exists for the video anywhere — ground truth counts as the answer (`causal_review_widget.py:2604-2605`; `has_gt`, `causal_review_io.py:596`),
- its triaged elements are not already all resolved (`_bundle_reviewed`, `:2539`).

The failed-segmentation skip at `:2665-2666` runs in **both** modes, including deep review. That is a problem for the Deep Review tool: see [Deep Review](#deep-review).

### What counts as a triaged element

A segment is triaged when either:

- the outcome detector wrote `outcome == "triaged"` or `flagged_for_review: true`, or
- the outcome is a *touched* one — `retrieved`, `displaced_sa`, `displaced_outside` — and the reach-assignment file has no reach marked `is_causal` for that segment. The pellet moved, but nothing says which reach moved it.

That rule exists in `triage_status.triaged_segments` (`triage_status.py:70`) and is written out a second time, identically, inside the widget (`_bundle_has_triage`, `causal_review_widget.py:2626`; the walk's own copy at `:1087-1088`).

A **third** version exists in the worklist builder used by the older clearing tool, and it is **not** the same rule (`triage_queue.py:196-222`). It never tests `outcome == "triaged"`; it also triggers on a `flagged_for_review` flag in the *reaches* file, which `triage_status` never reads; its test for "has a committed causal reach" also accepts a reach whose `label` starts with `causal` (`triage_queue.py:107`), which `triage_status` does not; and it drops segments already marked `triage_cleared` unless `--include-cleared` is passed. In practice the missing `outcome == "triaged"` test does not change the result for v6 output, because the v6 cascade always writes `flagged_for_review: true` alongside a `triaged` outcome (`outcomes/v6_cascade/detector.py:260-267`). The other three differences are real.

In triage mode the walk visits **only** triaged segments (`_visible_indices`, `:1122`). Everything else is skipped.

### What the person is shown and asked

The whole video is loaded once as a decode-on-demand layer; navigation only moves the playhead (`_load_video`, `:487`). The view opens on the relevant reach — the reviewer's own earlier answer if there is one, otherwise the algorithm's causal reach, otherwise the reach nearest the interaction frame (`_relevant_reach`, `:590`) — landing 25 frames early (`OPEN_LEAD`, `:485`), with 45 frames of padding around the reach (`WINDOW_PAD`, `:482`; `_compute_segment_window`, `:635`). Tracking points and the pillar circle are drawn as overlays.

The question panel takes one of three shapes, decided by `_segment_triage` (`:1770`):

- **`reach_uncertain`** — the outcome is already committed (`retrieved`, `displaced_sa`, `displaced_outside`, `abnormal` or `abnormal_exception`, `:1790-1793`); only the reach is open. Asked: *"Which reach caused it?"*, with an outcome-correction dropdown underneath.
- **`outcome_uncertain`** — exactly one candidate reach survives narrowing, so the reach is effectively pinned. Asked: *"What did it do to the pellet?"*, plus an optional reach override.
- **`both_uncertain`** — neither is known. Both questions.

The reach list is narrowed by `assignment.v2.triage_reduction.reduce_triaged_segment` to reaches that could not be ruled out as misses (`_candidate_reach_ids`, `:1675`); the ruled-out ones are built too but hidden behind a "Show all reaches" checkbox (`_make_reach_picker`, `:1859`). There is an explicit "No reach — nothing acted on this pellet" option (`:1906`), and choosing `untouched` clears and disables the reach picker so the two answers cannot contradict each other (`_sync_reach_picker_to_outcome`, `:2029`).

Every segment also gets an **"Ignore windows"** box — frame ranges where something that is not a reach moved the pellet (`:1369`, `_make_abnormal_ranges_widget`, `:2161`) — a free-text notes box, and a **"Segmentation says this is segment N"** spin box for saying the cuts are offset (`:1321`, `_make_segment_number_fix`, `:1821`).

Two more buttons: **Flag Session** (`:2514`) and, in deep-review mode only, **Clear -> re-enter pipeline** (`:3069`). A segmentation editor reachable from the panel lets the reviewer re-cut the video and re-run the algorithms on their cuts (`_reseg_apply`, `:1632`); it calls `stage_video(..., boundaries_override=...)`, which reads the video and pose from their canonical archive locations, so it fails on a bundle whose video is not archived there (the failure is caught and shown as "Re-segmentation pipeline failed"). After a successful re-cut the widget empties its in-memory answers (`:1666`), and because saving always rewrites the whole review document (`save_causal_review`, `causal_review_io.py:236-279`), any answer given before the re-cut and not re-entered is replaced by an unanswered placeholder.

Keyboard shortcuts are playback only — space, `b` for reverse, arrows, and speed keys 1-6 (`_setup_keybindings`, `:2963`). There is no key for next/previous segment; those are buttons.

### Escalating out of triage (since 2026-08-25)

Two ways a triage reviewer sends a video to deep review instead of finishing
it here:

- **The corrected segment number.** The per-segment question "Segmentation
  says this is segment N. If that's wrong, which segment is this actually?"
  writes `true_segment_num` + `segmentation_wrong` into the record. The
  reviewer keeps answering normally; at return time the watcher sees the
  correction and diverts the whole video to the deep-review queue for manual
  re-segmentation rather than re-injecting it (`review_return.py`,
  `TriageStatus.seg_pending_reseg`). Once the cuts are hand-fixed
  (`boundary_source: "human"`), the old correction stops blocking the video.
- **The escalate button.** *Escalate: bad segmentation -> deep review* saves
  the answers given so far (they re-attach by frame span after the boundary
  fix), moves the bundle to the deep-review queue with the notes text as the
  routing reason, and loads the next video (`_escalate_to_deep_review`).

### Two ways an unanswered question becomes an answer

Both matter before trusting `outcome_source: "human_review"` in the data.

1. **Any navigation records the current segment.** "Save Segment + Next" (`:2480`), "Next Segment" (`:2939`) and "Prev Segment" (`:2930`) all call `_collect_answers()` and store the result. There is no "I did not answer this" state.
2. **`_collect_answers` starts from the algorithm's values.** It initialises `human_outcome = seg["outcome"]` and `human_causal_reach = seg["causal_reach"]` (`:2299-2300`) and only replaces them where the reviewer actively picked something. The reach picker also pre-selects the candidate nearest the interaction frame (`:1946-1955`).

The consequence: clicking past a `reach_uncertain` segment without touching anything writes a record whose `human.outcome` is the algorithm's outcome (`:2331`) and whose `human.causal_reach` is the pre-selected reach. On an `outcome_uncertain` or `both_uncertain` segment where the reviewer leaves the `-- select outcome --` sentinel alone (`:2344`, `:2356`), `human.outcome` stays the string `"triaged"`. That record counts as resolved everywhere — `resolved_segments` (`triage_status.py:98`) only checks that `answers.reviewed` is not `False` — so it releases the video from the queue, and where the review does get applied it is stamped `outcome_source: "human_review"` (`causal_review_io.py:500`). A segment can therefore reach the data labelled `"triaged"` with a person's name on it.

### What is written

`_save_review` (`:2844`) writes `{stem}_causal_review.json` into the bundle directory (`_review_dir`, `:880`) **and, since 2026-08-24, a durable copy at `{NAS}/review_records/reviews/{stem}_causal_review.json` first**. The bundle copy is the one the reviewer edits; the durable copy is the one nothing regenerates. Before this, a bundle-resident review was destroyed by any reprocess that recreated the bundle, and 41 reviewed videos had no other copy anywhere. It contains **one record per segment in the video**, not per reviewed segment: segments the reviewer never visited get a placeholder with `answers: {"reviewed": false}` (`:2867-2880`). It also calls `update_corpus_index` (`causal_review_io.py:690`), which appends to `{NAS}/review_records/causal_review_index.json`; failures there are printed and swallowed (`:2910-2911`).

On reopening, `_load_saved_review` (`:892`) drops the placeholders (`:920-923`) and jumps to the first segment that is either unscored or whose **algorithm outcome has changed since the review** (`:925-937`) — a changed algorithm call is treated as needing another look. Note that this restore matches records to segments **by segment number** (`load_causal_review`, `causal_review_io.py:283-300`), not by frames.

### The record format

Written by `build_segment_record` (`causal_review_io.py:167`):

```
segment_num       the segment's number at the time of review
segment_span      {"start": ..., "end": ...} -- the frames the reviewer saw
pellet_num
algo              {outcome, causal_reach, interaction_frame}   as it stood
human             {outcome, causal_reach, is_phantom, agreed}
answers           {triage_kind, outcome_pick/outcome_override, causal_pick,
                   abnormal_ranges, reviewed: false for placeholders}
notes             free text
true_segment_num  present only if the reviewer changed the segment number
segmentation_wrong  true, alongside true_segment_num
```

New saves stamp the document `schema_version: "1.1"` (`causal_review_io.py:268`). Older files were given spans by a backfill that left the version alone, so **no existing review file carries 1.1**: all 662 review files under `MouseReach_Pipeline/Analyzed` say `1.0` and all 662 have spans.

`is_phantom` is always `False` — nothing sets it (`causal_review_widget.py:2301`, `:2875`, `:2291`).

`provenance` is filled by `collect_provenance` (`causal_review_io.py:112`), which reads version stamps out of the four algorithm JSONs and the tracking-model name out of the pose file. Every read is wrapped in a bare `except`, so a missing key means "could not read it", not "not applicable". It also captures less than it looks like it does: it only copies keys named `version`, `detector` and `segmenter_version` (`:137-142`), while the reach detector and the outcome cascade both write their version under `detector_version`. In real files this shows as `"reach_detector": {}` and `"outcome_detector": {"detector": "v6_cascade"}` with no version at all — verified in `20250624_CNT0110_P2_causal_review.json` and others.

---

## Working the deep-review queue: the operator's walkthrough

This section is for the person doing the work, no codebase knowledge assumed.
Everything below it in this document is the engineering detail behind these
steps.

**Opening the queue.** The launcher is NOT on the PATH of a plain PowerShell,
cmd, or VS Code terminal -- typing `mousereach` there says "not recognized".
Three ways that actually work, pick one:

- **Desktop shortcut** (processing server): double-click **MouseReach** on the
  desktop.
- **Any terminal, no setup** -- run the executable by its full path. On the
  processing server:
  `C:\LAB_ROOT\envs\mousereach\Scripts\mousereach.exe`
  (On another lab machine, the same path under wherever that machine keeps its
  mousereach conda environment -- e.g. `A:\...` on the DLC PCs. If unsure, ask
  or search for `mousereach.exe` under the machine's conda `envs` folder.)
- **Anaconda Prompt** (Start menu -> "Anaconda Prompt"): activate the
  environment first, then the short command works:
  `conda activate C:\LAB_ROOT\envs\mousereach` then `mousereach`.

When napari opens, find the **Review Queues** tab in the right-hand dock. It
shows how many videos are waiting in each queue and has three buttons: *Open
Triage Review*, *Open Deep Review*, and *Open Re-segmentation*. Each opens in
its own window.

**Which button to press.** Deep-review videos are there for one of two broad
reasons, and each has its tool:

- **The segmentation is the problem** (the boundaries between pellet
  presentations are wrong -- this is most of the queue, and every video the
  triage escalate button or a corrected segment number sent here). Press
  **Open Re-segmentation**. The tool loads a video, parks the playback on the
  first segment, and asks three questions per segment: is this really segment
  N; does it start where the algorithm says (within 10 frames); does it end
  where the algorithm says. Answer yes with a click, or answer no by clicking
  one of the offered candidate frames (each click shows you that exact frame
  in the video), by scrubbing to the right frame yourself and pressing "Use
  current frame", or by typing the frame number. The on-screen notes say
  which frame to pick: a segment STARTS on the frame after the scoring area
  jumps, and ENDS on the frame before the next jump. Press *Confirm answers ->
  next segment* each time; press *Save these cuts* when the video is done.
  Saving marks the cuts as human-made -- the pipeline will keep them.
- **Something else is wrong with the whole video** (escalated for a reason
  that is not boundaries, or you need to re-judge every segment's outcome).
  Press **Open Deep Review**. It walks every segment of the video asking the
  same outcome/causal-reach questions as triage review.

**Finishing a video.** Fixing the cuts does NOT release the video by itself.
After the cuts are right (or after a full deep review), open **Deep Review**
on that video and press **Clear -> re-enter pipeline**. That is the release:
the watcher then re-runs the video from segmentation onward -- keeping any
human-made cuts -- and the results flow to the database automatically. Nothing
else needs to be done.

**If you get a video that is not actually broken**, press *Skip this video* in
the re-segmentation tool (it stays in the queue), or clear it through Deep
Review if you have judged every segment and all is well.

---

## Deep Review

The same widget with `deep_review=True` (`causal_review_widget.py:198-203`). Differences:

- It reads the `DEEP_REVIEW` queue, not the triage queue (`queue_launcher_widget.py:123-126`).
- `triage_only` is forced off (`:210`), so **every** segment is walked.
- The **Clear -> re-enter pipeline** button appears. It saves the review, then writes `{stem}_deep_review_cleared.json` into the bundle (`_clear_deep_review`, `:3069`, marker at `:3085`). That marker is what the watcher looks for.

It has no console command. It opens from the "Review Queues" launcher tab (`queue_launcher_widget.py:123`) or from the pipeline dashboard for one named video (`dashboard/widget.py:1488`).

**It cannot open the videos it exists for.** The bundle pool skips any bundle whose segmentation failed (`:2665-2666`), and "segmentation failed" is the main reason the gate routes a video to deep review in the first place (`review_gate.py:87-88`). Of the 120 bundles in the deep-review queue right now, 17 meet that test and are invisible to the tool; one more has no segments file at all, which the gate counts as failed (`triage_status.segmentation_failed`, `triage_status.py:50-51`) but the widget counts as fine (`:2610`, which returns `False` on any read error). The Segmentation Fixer below is the tool that can open them.

---

## Segmentation Fixer (the re-segmentation tool)

`mousereach-fix-segmentation`, or the **Open Re-segmentation** button on the
Review Queues tab (`queue_launcher_widget._open_reseg`). Segment cuts only: no
outcomes, no reaches, no causal attribution.

It walks the deep-review queue (or `--queue-dir`) and offers bundles that need a
human's cuts, three ways in (`_needs_reseg`): the segmenter's own non-empty
`needs_human` list (its record of having had to force its answer, written by
`save_segmentation`, `segmentation/core/segmenter_robust.py:931`); a
reviewer-declared segment mislabel (`segmentation_wrong` in the bundle's
travelling causal review); or a routing reason naming segmentation (the triage
escalate button, or the watcher's mislabel diverts). One way out:
`boundary_source: "human"` in the segments file means the cuts were already
hand-fixed, and the bundle is not offered again. Videos with the most unused
candidate cuts are offered first; videos with no candidates at all are last,
because those have to be marked from scratch.

**The guided walk** (added 2026-08-25) is how the tool opens: one segment at a
time, playhead parked at the segment's algo start, three questions --

1. *Is this segment number N (= pellet number N)?* Yes, or "No, it is actually
   segment __". A no is recorded in the saved `guided_walk` answers; the
   numbering itself only changes when a cut is added or removed, and Save
   refuses (with an explicit override) while a denied identity stands with the
   cuts unchanged.
2. *Does this segment start within 10 frames of {algo start}?* Yes, or the real
   frame -- typed, taken from the playhead ("Use current frame"), or picked
   from the **candidate chips**: the segmenter's own nearby candidate tray
   advances, each labelled with the frame it implies plus its evidence
   (`n_proposers`/4, consensus score). Clicking a chip jumps the playhead there
   and fills it in as the answer. Operator note shown in the tool: the start is
   the frame AFTER the scoring-area jump.
3. *Does this segment end within 10 frames of {algo end}?* Same controls; the
   end is the frame BEFORE the next jump (the tool moves the following cut to
   frame+1 itself).

Corrections apply to the cuts immediately; the per-segment answers are saved
into the segments file as `guided_walk`. Below the walk, the full candidate
table and manual add/drop-cut controls remain for work the walk cannot express,
with a legend stating what a boundary means (boundary 1 = start of segment 1
and end of the pre-pellet setup frames; each cut sits on the first frame after
a scoring-area jump).

Saving copies the original aside, then rewrites `{stem}_segments.json` with
`boundaries`, `algo_boundaries` (what the algorithm had), `boundary_source:
"human"`, `corrected_by`/`corrected_at`, `needs_human` emptied, and
`guided_walk` (`_save`).

Two things it does not do:

- **It does not release the video.** It writes no `{stem}_deep_review_cleared.json` and no ground-truth file, which are the only two things the return scan accepts as a deep-review clearance (`review_return.py`). A video whose cuts have been fixed sits in the deep-review queue until someone opens the Deep Review tool on it and presses Clear.
- **It does not re-run anything.** The bundle's reaches, outcomes and assignments still describe the old cuts until the video goes back through the pipeline. The re-run keeps the human cuts: segmentation preserves a `boundary_source: "human"` segments file instead of overwriting it (`segmentation/core/batch.py::process_single`).

Nothing routes videos here on the strength of `needs_human` alone. That routing was added and then deliberately switched off — the gate records the verdict and ignores it, with the reasoning written out in full at `review_gate.py:89-105`. What DOES route videos here (since 2026-08-25): a reviewer setting a corrected segment number in triage (the return scan diverts the video instead of re-injecting it, `review_return.py`), the triage tool's "Escalate: bad segmentation" button, and the ReprocessingScanner's divert when a pending review declares a mislabel (`reprocessor.py`).

---

## Ground Truth Tool

`GroundTruthWidget(review_mode=False)` — `mousereach-gt` and `mousereach-unified-review`, both to `ground_truth_widget.py:3018`.

Question: what really happens in this video. There is no accept/verify step — **setting a value is the answer** (`ground_truth_widget.py:92-104`). It seeds itself from the algorithm's output, and the person corrects boundary frames, reach start/end frames, outcome classes, interaction frames and outcome-known frames, with per-item comments and per-component "exhaustive" flags meaning "this component is completely labelled for this video".

Two save buttons (`:585-625`):

- **Save Progress** (`_save_progress`, `:1971`) writes everything, including untouched algorithm-seeded rows.
- **Save as Ground Truth** (`_save_ground_truth`, `:1980`) writes a filtered copy containing only items a person actually determined.

Both write `{stem}_unified_ground_truth.json` next to the video (`unified_gt.save_unified_gt`, `unified_gt.py:252`). Ground truth outranks every review downstream, and having one anywhere removes the video from the review queue entirely (`_bundle_needs_review`, `causal_review_widget.py:2604-2605`).

The video dropdown is filled from the pipeline index (`_populate_video_dropdown`, `:652`) and capped at 30 entries (`:675`); if the index is unavailable the dropdown falls back to "Browse for video..." and the error is shown in the status line (`:699-705`).

`--algo-dir` on the command line changes where the tool reads a segment's decision window and which information panel it shows (`:3028-3033`). It does **not** change where a save writes: `_algo_files_dir()` returns the video's own parent directory unconditionally (`:2030-2039`).

---

## Review Tool (edits the algorithm's files)

`GroundTruthWidget(review_mode=True)` — `mousereach-review-legacy` (`ground_truth_widget.py:3069`), and the tab the `mousereach` launcher loads as "3 - Review Tool" (`launcher.py:274-283`; the comment above it at `:268` about a tabbed widget is stale).

One button, "Save & Continue" -> `_save_to_algo_files` (`:2041`). It rewrites the three algorithm JSONs in place with `validation_status: "validated"`, per-boundary correction records, and `human_corrected` / `original_*` fields, then starts a background database-refresh check in a daemon thread (`_maybe_update_database`, `:2209`).

**Its reach corrections do not reach the data.** It reads the existing reaches out of the nested `segments[].reaches` structure (`:2126-2129`), then writes the corrected reaches to a new **top-level** `reaches` key (`:2152`) and leaves the nested `segments` untouched. The reach detector writes the nested form (`reach/core/reach_detector.py:151-158` defines it; `save_results` at `:1104-1119` serialises it), and the kinematics step reads the nested form (`kinematics/core/feature_extractor.py:248`, `:308`). Corrected reach boundaries saved here are therefore invisible to every downstream consumer except the causal review widget, which happens to prefer the flat key when it exists (`causal_review_widget.py:1005-1011`). Outcome and boundary edits in this tool *do* land, because those are written back into the structures that are read (`:2184-2188`).

---

## Triage Clearing and the spot-check pool

`mousereach-triage-clearing` -> `triage_clearing.py:664`. This is the older per-segment clearing walk. Its own docstring (`triage_clearing.py:14`) and the folder's `AGENTS.md` (`review/AGENTS.md:86-89`) both still say its command is `mousereach-review-tool`; that name has pointed at the causal review widget since `pyproject.toml:106`.

It subclasses the Ground Truth widget and drives a worklist built by `triage_queue.py`. Two sources, chosen by flag:

- `--corpus-root <dir>` (default: the triage queue, or `MOUSEREACH_ROUTINE_ROOT`) — a folder of per-video bundles. Videos whose segmentation failed are held out and printed as a separate "manual re-seg lane" (`scan_corpus_root_for_triage`, `triage_queue.py:247`; printed at `triage_clearing.py:783-788`).
- `--algo-dir <dir>` — one flat folder of algorithm outputs, the layout the improvement harness uses. It wins over `--corpus-root` (`triage_clearing.py:749`).

For each triaged segment it loads only that segment's frames plus padding (`--pre-pad`/`--post-pad`, default 30 each) and asks the reviewer to mark the causal reach with S and E, set the outcome, and write a note about why the algorithm missed it. Save (`_save_current_triage_segment`, `:477`) refuses unless both a causal reach and an outcome are set (`:500-512`), then writes, **for that segment only**:

- in `_reaches.json`: `flagged_for_review: false`, `triage_cleared: true`, `cleared_by`/`cleared_at`, the causal reach updated with `human_corrected: true` and the note, and **every other reach in the segment marked `exclude_from_analysis: true`** (`:545-556`);
- in `_pellet_outcomes.json`: the same clearing fields plus `human_verified: true`, the person's outcome, interaction frame, outcome-known frame and `causal_reach_id` (`:570-590`);
- a `{stem}_unified_ground_truth.json` next to the bundle's algorithm JSONs (`:599`; path from `_unified_gt_path_for_video`, `:642`).

Two consequences of that third write are easy to miss. The object it saves is `self.gt`, the *unfiltered* one seeded from the algorithm — rows nobody touched are in it with `determined: false`. The truth layering skips undetermined rows (`truth_resolver.py:138-139`), so no false values leak into the numbers, but **the file's existence alone** makes `has_gt()` true (`causal_review_io.py:596`), which (a) removes the video from the causal review tool's queue as "already ground-truthed" and (b) in a deep-review bundle counts as a clearance all by itself (`review_return.py:112-114`).

`--qc-count N` blends N spot-checks into the same walk (`qc_pool.py`). These are segments the algorithm was **confident** about — segmentation did not fail, the outcome is a committed class, the segment is not flagged, and for touched outcomes a causal reach was committed (`iter_passing_segments`, `qc_pool.py:78-142`) — sampled round-robin across cohort+date strata, never-checked first (`sample`, `:184`). The reviewer gets two buttons instead of the marking UI: *Algo is RIGHT* logs agreement; *Algo is WRONG* logs disagreement **and** sets `flagged_for_review` back on that segment so it re-enters triage (`_flag_segment_for_triage`, `:429-444`). State lives in `<parent of the corpus root>/_QC/qc_state.json` plus an append-only `qc_drift_log.jsonl` (`qc_pool.py:160`, `:167-173`). `--qc-report` prints the agreement rate and exits without opening napari (`triage_clearing.py:729-742`). `--qc-count` is ignored outside corpus-root mode (`:803-804`).

---

## Re-attaching a review after reprocessing

### The problem

A review is a fact about a stretch of video. `segment_num` is not — the segmenter hands out those numbers fresh on every re-cut. A review of "segment 7" used to be applied to whatever the new segment 7 turned out to be, and the result was recorded as `outcome_source: "human_review"`, indistinguishable from a real one.

### What was added

`build_segment_record` stores `segment_span` — the exact `{start, end}` frames the reviewer was shown — on every record (`causal_review_io.py:167`, span built by `_segment_span`, `causal_review_widget.py:81-91`). Reviews written before that were backfilled in place by `scripts/backfill_causal_review_spans.py`, which refuses to write a span unless the neighbouring segmentation still matches the review: same segment count, and every record's algorithm causal reach starting inside the span that number would get (`verify`, `backfill_causal_review_spans.py:124-156`). Files failing either check are skipped and listed by name. The script does not touch the document's `schema_version`.

All 662 archived review files under `MouseReach_Pipeline/Analyzed` now carry spans. Eight of the 66 review files sitting in the live queues do not, and those fall back to number matching.

`index_review_by_segment` (`causal_review_io.py:349`) maps each **current** segment number to the review record that describes it:

- A record with no span is matched by number, and only if that number exists in the current segmentation (`:404-408`). That is all that was ever available for those records.
- A record with a span is scored against every current segment by overlap. The score is `overlapping frames / the shorter of the two ranges` (`:415`).
- The best match must beat the runner-up by at least **0.15** (`MIN_MARGIN`, `:346`). If it does not, the reviewed stretch has been split across two new segments and the record is **dropped**, with a note (`:423-430`).
- The best match must cover at least **0.5** of the shorter range (`MIN_SPAN_OVERLAP`, `:342`). Below that the record is **dropped** as describing different footage (`:432-437`).
- If two records both want the same current segment, the weaker overlap is dropped (`:439-445`).
- Where a record survives but lands on a different number than it was written with, a note records the move (`:449-453`).

Notes are returned to the caller and logged at warning level (`truth_resolver.py:109-110`, `causal_review_io.py:536-537`). Nothing is written back to the review file; the matching is redone every time.

`true_segment_num` / `segmentation_wrong` — the answer the reviewer gives with the "Segmentation says this is segment N" control — plays no part in any of this. See [Declared but does nothing](#declared-but-does-nothing).

### Frame matching runs on one path only

Frame matching happens where a review is turned into numbers:

- `truth_resolver.resolve_truth_layers` (`truth_resolver.py:289`), called by the feature extractor (`kinematics/core/feature_extractor.py:220-225`), passes the current segmentation from `reaches_data["segments"]` (`truth_resolver.py:312`).
- The extractor's fallback `load_and_apply_review` does the same (`feature_extractor.py:228-239`).

Every decision about whether a video is **finished with review** still matches by segment number alone — `triage_status.resolved_segments` (`triage_status.py:98`), used by the pipeline gate (`review_gate.py:114`, `_review_resolved_segments` at `:122-135`), by the return scan (`review_return.py:244`), and by the review tool's own "is this bundle done" check (`causal_review_widget.py:2539`).

So after a re-segmentation, a review whose frames no longer match anything is correctly **dropped** from the outcomes — and still counts as a resolution that releases the video from the queue. The video finishes with the algorithm's original `triaged` call in the data and nothing flagging it.

---

## Reviews mostly do not reach the numbers

This is the most important thing in this document.

`resolve_truth_layers` looks for a review in exactly three places (`truth_resolver.py:315-317`):

0. `review_records/reviews/<stem>_causal_review.json` (the durable copy -- always written, belongs to no bundle)
1. `Processing/Review/triage/<stem>/<stem>_causal_review.json`
2. `Processing/Review/flagged_for_review/<stem>/<stem>_causal_review.json`
3. whatever the caller passes as `primary_dir`

`_review_doc` (`truth_resolver.py:72-88`) reads a review by appending a filename to what it is given. It accepts a **directory**. Every production caller passes the result of `resolve_review_path` (`run_all.py:120-124`, `orchestrator.py:1151-1155`, `orchestrator.py:2100-2104`, `reprocess_to_current.py:258-260`), and `resolve_review_path` returns the path to the JSON **file** (`causal_review_io.py:83-90`). Appending a filename to a file path yields nothing, so **lookup 3 never contributes**.

The `except` branch that would rescue this (`feature_extractor.py:226-239`) calls `load_and_apply_review`, which does accept a file path (`causal_review_io.py:526-527`). It only runs if `resolve_truth_layers` raises, and `resolve_truth_layers` is written not to raise.

So a review is applied only while its bundle is still sitting in one of the two queue folders. Once the return scan has moved the bundle back into Processing and deleted the bundle directory (`review_return.py:163-180`, `:208-211`), the review that released the video is invisible to the kinematics step and the numbers keep the algorithm's call.

Checked directly on `20251028_CNT0413_P1`, whose review sits in `Processing` and has four segments a person answered (including "algorithm said triaged, person said displaced_sa"):

- `resolve_truth_layers(..., primary_dir=<the file>)` — what the pipeline does — leaves every segment `outcome_source: "algo"`, and segments 9, 16 and 19 still read `triaged`.
- `resolve_truth_layers(..., primary_dir=<the directory>)` returns `displaced_sa`, `displaced_outside`, `displaced_sa` with `outcome_source: "human_review"`.
- Its shipped `_features.json`, written six days after the review, carries `outcome_source: "algo"` on all 220 reaches.

Across 400 features files in the local Processing folder, `outcome_source` is `"algo"` on every reach and `"human_review"` on none, while 300 of 300 sampled review files in the same folder contain a real human outcome.

Reviews **have** landed historically — 262 of 600 sampled archived features files carry `human_review` — and since lookup 3 cannot work and ground truth is stamped `ground_truth` rather than `human_review`, the queue-folder lookup is the only way those values could have got there. The deciding factor is whether the bundle was still in a queue folder when kinematics ran.

The gate, by contrast, opens the review file correctly (`review_gate.py:130-133`) and counts it as resolving the video. That asymmetry is the whole failure: the answer is good enough to release the video and not good enough to change the number.

---

## What depends on a segment being in the right place

The kinematics step pairs each segment's reaches with that segment's outcome and copies segment-level facts onto every reach in it (`feature_extractor.py:248`, `:308`, `:317-320`, `:373-384`). Everything below is decided by which segment a reach falls in:

| Field on a reach | How it is decided |
|---|---|
| `segment_num` | the segment it was grouped into (`:317`) |
| `is_first_reach`, `is_last_reach`, `n_reaches_in_segment` | position and count within that segment's reach list (`:318-320`) |
| `causal_reach` | true when the reach's id equals that segment's `causal_reach_id` (`:380-382`) |
| `outcome`, `interaction_frame` | copied from the segment's outcome, only onto the causal reach (`:383-384`) |
| `outcome_source`, `reviewed_by`, `algo_outcome`, `algo_causal_reach_id` | copied from the segment's outcome record onto **every** reach in it (`:373-377`) |
| pellet number | the review tool treats segment N as pellet N with no separate source (`causal_review_widget.py:1044`, banner at `:1239-1245`) |

Three specific hazards:

1. **The pairing is positional, not by number.** `zip(reaches_data['segments'], outcomes_data['segments'])` at `feature_extractor.py:248` pairs the first reach-segment with the first outcome-segment and so on, and stops at the shorter list. If the two files disagree on how many segments there are — which is what the Segmentation Fixer produces until the video is re-run, since it rewrites only `_segments.json` — outcomes attach to the wrong reaches with nothing raised and no record left.

2. **An offset is invisible.** Every segment still looks well formed; only the numbering is shifted, so pellet 7 as scored on the bench is compared against footage of pellet 8. The reviewer can now say so with the "Segmentation says this is segment N" control, and the answer is stored — but nothing reads it (see below), so an offset a person has identified still does not change any number.

3. **A re-cut can silently discard the review.** `index_review_by_segment` drops any review whose frames no longer line up with a current segment, while the release check still counts it as resolved by number. The video finishes carrying the algorithm's call.

Reaches themselves are grouped by segment in two different ways depending on file shape: nested output is trusted as grouped by the detector (`causal_review_widget.py:1024-1030`), while flat output and any manual re-cut are re-grouped by which segment window contains each reach's midpoint (`:1011-1023`).

---

## The queues, and how a video leaves one

### Where they are

Set in `config.py:134-136`, derived from `NAS_ROOT`:

| Setting | Path | Holds |
|---|---|---|
| `Paths.REVIEW_ROOT` | `<nas>/Processing/Review` | parent of both queues; `flagged_sessions.json` and `_QC/` live here |
| `Paths.TRIAGE_REVIEW` | `<nas>/Processing/Review/triage` | per-element questions |
| `Paths.DEEP_REVIEW` | `<nas>/Processing/Review/flagged_for_review` | failed segmentation, failed quality control, escalations |

An unset `nas_root` does **not** make these `None`. `NAS_ROOT` falls back to `<NAS drive>/! DLC Output` whenever a NAS drive is configured (`config.py:100-101`), so the three queue paths resolve to real folders in the old layout and the node looks like it is working. `Paths.NAS_ROOT_ORIGIN` (`:106`) is what distinguishes `config` from `fallback` from `unset`. Only when both `nas_root` and the NAS drive are unset are the queues `None`, and only then does the launcher tab report "queue not configured" (`queue_launcher_widget.py:112-113`). The code carries the same warning in a comment at `config.py:102-105`.

### How a video enters

`run_gate` (`review_gate.py:209`) runs after reach assignment and before kinematics. `evaluate_gate` (`:68`) decides, in order:

1. Ground truth marks the video exhaustively complete -> **clean**, proceed, even if segmentation failed (`:85-86`).
2. Segmentation failed — no segments file, confidence at or below 0, or a "reference quality" anomaly -> **DEEP_REVIEW** (`:87-88`).
3. Quality control returned `needs_review` -> **DEEP_REVIEW** (`:107-108`).
4. Any triaged segment not resolved by ground truth or a saved review -> **TRIAGE** (`:114-118`).
5. Otherwise -> **clean**, kinematics runs (`:119`).

The segmenter's own `needs_human` verdict is read into the status object (`triage_status.py:184`) and deliberately not acted on (`review_gate.py:89-105`).

On a hold, `route_to_queue` (`:159`) moves the whole bundle out of the processing directory, writes a queue manifest so the review tool can open the bundle in place (`:138-156`), and sets the database state to `triage` or `deep_review`. If the normal state transition is rejected the code forces the state rather than letting disk and database disagree (`:178-194`).

`reprocess_to_current._stage_review_bundle` (`reprocess_to_current.py:66`) stages bundles the same way for the bring-current path, and never clobbers a bundle that already exists (`:80-82`).

### How a video leaves

Only `scan_review_queues` (`review_return.py:225`) moves a bundle out. It runs inside the watcher every 10th poll cycle (`orchestrator.py:1345`, `:1391-1395`) and handles at most **10 bundles per scan** (`MAX_RETURNS_PER_SCAN`, `review_return.py:222`) so returning does not starve the pipeline.

- **Triage**: the bundle leaves when it has triaged elements, all of them are resolved by a saved review, and segmentation has not failed (`:247`).
- **Deep review**: the bundle leaves when `{stem}_deep_review_cleared.json` exists, or a ground-truth file sits in the bundle (`:108-115`).

Returning moves the bundle's data files, including the review file, into the local processing directory, drops the queue-only manifest, deletes the now-empty bundle directory, and sets the video back to `processing` so the pipeline re-runs it and the gate re-checks (`_return_to_processing`, `:118-214`). It refuses and leaves the bundle alone if the database row cannot be created (`:141-146`), if the pose file cannot be found (`:152-159`), or if the state cannot be set (`:198-205`) — a clearance is never spent on a run that would fail.

**No review tool moves a bundle between folders.** The only `shutil` calls in `review/` are the Segmentation Fixer copying a file aside before overwriting it (`fix_segmentation_widget.py:444`) and the ground-truth migration helper (`unified_gt.py:877`). A bundle with a complete review sits in the folder until the watcher's return scan picks it up.

**A triage bundle with zero triaged elements never leaves.** The return condition requires `st.has_triage` (`:247`), so a bundle whose triage flags were cleared by some other route sits in the folder indefinitely.

### The scan-free queue index

`queue_index.py` keeps a SQLite list at `{NAS}/review_records/triage_queue.db` of videos needing review, so the tool does not have to read every bundle. It is push/pop. `pop` happens in the tool when a video's triaged elements are all resolved (`causal_review_widget.py:2827-2842`, best-effort, failures swallowed).

**`push` is called from exactly one place** — `reprocess_to_current._push_review_index` (`reprocess_to_current.py:108`), the bring-current path, and it skips deep-review bundles entirely (`:114-115`). The live watcher gate does not push. That matters because of how the tool reads the index (`_review_pool_paths_unfiltered`, `causal_review_widget.py:2678`): it takes the index rows whose parent is the requested queue folder and, **if there are any, returns only those** (`:2693-2696`) — the folder is not scanned at all. So on a machine where bring-current has pushed even one row, bundles routed into the same folder by the watcher are invisible to the triage queue until that row is popped. Deep review is unaffected, because nothing is ever pushed with that parent.

`seed_from_folder` (`queue_index.py:120`) exists to rebuild the index from the folder and **is not called anywhere in the codebase**.

### The session flag button does the opposite of what it says

The button is labelled "Flag Session (needs review)", its tooltip says it marks every video of that mouse+day as must-be-human-reviewed, and after clicking it the tool says "all its videos need human review" (`causal_review_widget.py:417-426`, `:2514-2528`). `flag_session` writes the session key into `Processing/Review/flagged_sessions.json` (`causal_review_io.py:642`).

Two pieces of code read that file: `_bundle_needs_review` (`causal_review_widget.py:2602-2603`) and the never-called `seed_from_folder` (`queue_index.py:160`). What the first one does is:

```python
if is_session_flagged(stem, root):
    return False   # this bundle does NOT need review
```

Flagging a session **removes** every one of that mouse+day's unreviewed videos from the review queue. Nothing in the gate, the return scan, the kinematics step or the database reads the flag. There is no path by which a flagged session gets more human attention.

### Automatic release from ground truth

`mousereach-resolve-triage-from-gt` (`triage/gt_resolve.py`) lifts triage flags on segments a ground-truth file already answers, writing the same clearing fields the clearing tool writes. The comment above it in `pyproject.toml:120-123` says it "runs as a step in the normal processing pipeline". It does not — the only caller in the tree is the improvement evaluation harness (`improvement/eval_all.py:163-167`).

---

## Declared but does nothing

Each of these is code that runs, produces a value, and has no reader.

- **Ignore windows / `abnormal_ranges`.** The reviewer marks frame ranges where something other than a reach moved the pellet; the answer is saved, carried onto the segment by both `apply_review_overrides` (`causal_review_io.py:503-504`) and `truth_resolver` (`truth_resolver.py:181-182`) — and read by nothing. There is no reference to `abnormal_ranges` anywhere outside `src/mousereach/review/`. The kinematics step never excludes those frames.
- **`true_segment_num` and `segmentation_wrong`.** The reviewer's answer to "this is really segment N" is written into the record (`causal_review_io.py:226-228`) and read by nothing. Grepping `src/` and `scripts/` for either name returns only the writer and the widget that supplies it.
- **`outcomes.exhaustive` in ground truth.** `_seg_overrides_from_gt` computes it and returns it; the caller assigns it to `_exhaustive_out` and never uses it (`truth_resolver.py:328`). Ground-truth outcome overrides apply per determined segment either way, so the flag changes nothing for outcomes. For *reaches*, `exhaustive` is load-bearing and does drop unlabelled algorithm reaches (`_apply_gt_reaches`, `:275-279`).
- **`TriageStatus.clean`** (`triage_status.py:152-156`) — a property with no callers. The gate and the return scan read `seg_failed`, `unresolved` and `fully_resolved` directly.
- **`TriageStatus.seg_needs_human`** (`:137`) — populated from the segmentation file (`:184`) and deliberately not acted on (`review_gate.py:89-105`).
- **`is_phantom`** in every review record: always `False`, never set.
- **Boundary segments.** `is_boundary` is hard-coded `False` (`causal_review_widget.py:1043`) and set nowhere else, so the three branches that handle a boundary segment (`:1236`, `:1284`, `:2281`) never run and the guard at `:1749` is always true.
- **`causal_review_index.json`.** Written on every save (`causal_review_io.py:690`, path at `:729`) so that, per the module docstring, the active-learning loop can bulk-read all reviews. Nothing reads it — no reader exists in `src/` or `scripts/`.
- **`base.py`** — `AlgoGTReviewMixin`, `DiffItem`, `DiffSummary`. Re-exported by `review/__init__.py:116-119` and documented in this folder's `AGENTS.md` as the way to build a review widget. No class in the codebase uses the mixin and no code constructs a `DiffSummary`.
- **`SavePanel`** in `save_panel.py:24` — never instantiated. Its sibling `SimpleSavePanel` (`:217`) is what the three per-step review widgets use.
- **`unified_widget.UnifiedReviewWidget`** — reachable only from the napari plugin menu as "Review Tool (Tabbed)" (`napari.yaml:28-30`) and from its own `main` (`unified_widget.py:1123`). Its docstring claims the command `mousereach-review-tool` starts it (`:17`); that command starts the causal review tool. The launcher's "Review Tool" tab loads `GroundTruthWidget` instead (`launcher.py:282`).

### Not on this list, though a previous draft said so

**`exclude_from_analysis` on reaches has real readers.** The triage clearing tool marks every non-causal reach in a cleared segment with it (`triage_clearing.py:554-555`) and the ground truth tool sets it too (`ground_truth_widget.py:1784`). What is true is narrower: the **kinematics feature extractor** never checks it, so those reaches still contribute to per-segment kinematics, the flag is not carried into `{video}_features.json`, and it is not a column in `connectome.db`. It **is** read by the analysis dataframe, which drops flagged reaches by default (`analysis/data.py:120-127`, `:167-168`, used by `load_all_data`, `:432-497`) — though on that module's default path the rows come from the features file, where the field is absent, so the filter has nothing to act on; it bites only when loading from `_reaches.json` (`:343`). It is also read by the truth resolver when merging ground-truth reaches (`truth_resolver.py:255`) and throughout the improvement metrics (`improvement/reach_detection/metrics.py:482`, `:518`, `:926`; `kinematic_damage.py:105`, `:173`; `_compute_baseline.py:58`, `:109`). The ODC-SCI export carries it as a column (`kinematics/analysis/odc_sci_exporter.py:157`).

---

## Which answer wins, where a review is applied at all

`resolve_truth_layers` (`truth_resolver.py:289`) builds layers low to high and merges them (`_apply_outcome_layers`, `:153-183`):

```
ground truth  >  processing-dir review  >  deep-review-queue review  >  triage-queue review  >  algorithm
```

Two things to be precise about:

- **The unit is a segment, not an element.** Within the per-segment outcome layers, a higher layer replaces the whole override for that segment — outcome, causal reach id and interaction frame together (`:160-162`, `:172-180`). Only a segment that no layer covers keeps the algorithm's values. The separate per-reach layer (ground truth only) is merged reach by reach (`_apply_gt_reaches`, `:249`).
- **Both review tiers are stamped the same.** Triage-queue and deep-review-queue reviews are both labelled `"human_review"` (`:320-322`); the data does not record which tool produced the answer.

Ground truth only overrides segments it marks determined (`:138-139`). Provenance is stamped per segment as `outcome_source` (`algo` / `human_review` / `ground_truth`) and per reach as `reach_source` (`algo` / `ground_truth`), with the algorithm's originals preserved as `algo_outcome` and `algo_causal_reach_id` (`:172-176`). Reviews never change reach boundaries or reach existence; only ground truth does, and only where `reaches.exhaustive` is set does it also delete algorithm reaches it never labelled (`:275-279`).

The processing-dir layer is the one that does not work — see [Reviews mostly do not reach the numbers](#reviews-mostly-do-not-reach-the-numbers).

### Where a saved review is looked for

Two different lookups, with different rules:

- `resolve_review_path` (`causal_review_io.py:55`) returns the **file**, most recently modified wins (`:90`), checking an explicit directory the caller passes (usually the processing directory), then the triage queue's bundle for that video, then the directory holding the canonical video. It never raises; nothing found means the caller does nothing. It does not look in the deep-review queue. This is what the gate and every kinematics caller use.
- `truth_resolver` uses its own set: the triage queue, the deep-review queue, and the `primary_dir` it is handed (`truth_resolver.py:315-317`). It needs directories.

---

## Documentation that is wrong at this commit

Stated here so nobody rediscovers it the hard way:

- `review/AGENTS.md:86-89` calls `TriageClearingWidget` the routine tool and maps it to `mousereach-review-tool`. That command is the causal review widget; the clearing tool is `mousereach-triage-clearing`.
- `queue_index.py:7` says "The tool MOVES a fully-reviewed video OUT of the folder". The tool removes the database row only; the watcher moves the folder.
- `triage_clearing.py:14` and `unified_widget.py:17` both claim `mousereach-review-tool`.
- `causal_review_widget.py:2509-2510` describes `_review_root()` as `Processing/Review/triage`. It returns the parent of the queue folder, `Processing/Review`, which is where `flagged_sessions.json` actually lives.
- `causal_review_widget.py:2307-2308` says "a left-unanswered element stays 'triaged' — we never fabricate a decision the reviewer didn't make". The outcome string does stay `triaged`, but the record still counts as resolved, releases the video, and is stamped `outcome_source: "human_review"`.
- `pyproject.toml:120-123` says the ground-truth triage resolver runs as a pipeline step. It does not.

---

## Configuration summary

| Setting | Where | Effect |
|---|---|---|
| `nas_root` | `~/.mousereach/config.json` | Root of both queues. Unset falls back to the legacy `<NAS drive>/! DLC Output` layout rather than to `None` (`config.py:100-106`); queues are `None` only when the NAS drive is unset too. |
| `MOUSEREACH_ROUTINE_ROOT` | environment | Overrides the corpus root the clearing tool's worklist scans (`triage_queue.py:490`). |
| `MOUSEREACH_TRIAGE_ALGO_DIR` | environment | Overrides the flat algorithm directory fallback (`triage_queue.py:465`). |
| `CONNECTOME_ROOT` | environment | Base for the two lookups above; defaults to `Y:\LAB_ROOT`. |
| `--pending-dir` | `mousereach-review-tool` | Which queue folder to review. Defaults to the triage queue (`causal_review_widget.py:3123`). |
| `--all-segments` | `mousereach-review-tool` | Walk every segment instead of only triaged ones (`:3126`). |
| `--worklist FILE` | `mousereach-review-tool` | CSV or JSON of `vid` + `segment_num`. Only those videos are offered and only those segments walked — and unlike the normal pool, already-reviewed and ground-truthed videos are included (`:2709-2718`). |
| `--cv` | `mousereach-review-tool` | Use pixel-based pellet localisation to narrow candidate reaches. Off by default because it decodes the video over the network; without it the narrowing uses tracking data only (`_get_cv_states`, `:1713`, gated at `:1727`). |
| `--queue-dir` | `mousereach-fix-segmentation` | Queue of bundles to work through; defaults to the deep-review queue (`fix_segmentation_widget.py:482-490`). |
| `--corpus-root` / `--algo-dir` | `mousereach-triage-clearing` | Bundle-per-video layout vs one flat folder. `--algo-dir` wins (`triage_clearing.py:749`). |
| `--include-cleared` | `mousereach-triage-clearing` | Put already-cleared segments back in the worklist. |
| `--qc-count N` / `--qc-report` | `mousereach-triage-clearing` | Blend N spot-checks into the session / print the agreement rate and exit. `--qc-count` is ignored outside corpus-root mode (`:803-804`). |
| `--pre-pad` / `--post-pad` | `mousereach-triage-clearing`, `mousereach-gt` | Frames of context loaded around the segment. Default 30 for clearing (`:708-713`), 0 for the ground truth tool (`ground_truth_widget.py:3037-3040`). |
| `--algo-dir` | `mousereach-gt`, `mousereach-review-legacy` | Where to read a segment's decision window and which info panel to show. Does not change where a save writes. |
| `preserve_clears` | `staging.stage_video` argument, default `True` | Captures human-cleared segments before re-running the algorithms and re-applies them afterwards, matched by segment number (`clear_guard.py:47`). **Skipped entirely when boundaries change** (`staging.py:339`), because a re-cut renumbers the segments; a cleared segment whose number is gone is reported as skipped, not silently dropped (`:348-350`). |

---

## Contested claims

This document was written from the source, then checked by a second reader
whose job was to disprove it. The statements below are ones they disputed and
that were not resolved. Neither side is authoritative: spot-checking found the
checker wrong at least once. **Do not rely on anything listed here without
opening the code yourself.** Everything not listed survived two passes.

- **extended_features holds json.dumps(reach['extended'] or {}) (database.py:595)**
  - disputed because: The quoted expression is not the code. Line 595 uses reach.get('extended'), not reach['extended']. The difference is behavioural, not cosmetic: with the bracket form a reach dictionary lacking the key would raise KeyError and abort the whole video's sync; with .get it silently yields the two-character string {}. The document's own following sentence ('features files produced by extractor 1.0.0 hav
- **Every caller either ignores that return value or logs at warning or debug level.**
  - disputed because: One of the seven callers does neither. reprocess_to_current.py records the returned boolean in the summary it returns, and reprocess_batch collects every per-video summary into the results list it returns. So on the reprocessing path the False is preserved as data rather than discarded or logged, which narrowly qualifies the neighbouring headline 'the failure is not reported anywhere'. The other s
- **Verified against: 4c54e46 (master, 2026-08-23), working tree clean.**
  - disputed because: HEAD is 4c54e46 on branch master, but the working tree is not clean. Two files are modified. Neither is under any path this document describes, so no code claim is affected and every citation still resolves at this commit; the header's own assertion is simply false as written.

---

## Update 2026-08-23: frame-only causal picks are honoured

Reviews written before the causal pick carried a reach id stored only the
frames. The truth resolver now matches those frames to the reach they name
(best overlap >=50%, within the reviewed segment), so those human answers reach
the data instead of being silently dropped. Picks that match no detected reach
-- hand-drawn reaches -- remain unresolvable by design: there is no features
row for a reach the detector never emitted.
