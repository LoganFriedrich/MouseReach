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
| **Re-segmentation Review** | GUI "Review Queues" tab -> Open Re-segmentation Review (no console command) | "Segmentation or quality control failed on this whole video. Is it usable, and can it go back through the pipeline?" | `{stem}_causal_review.json` plus `{stem}_deep_review_cleared.json` |
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

`load_pending_queue` picks the next video from the pool of bundles that still need review **in the lab's work order** (the order file and panel described in WORK_PRIORITY.md): it takes the highest-priority group that has a video waiting -- for example the first project, or a study's session list within a project -- and picks **at random inside that group** (`_pick_by_priority`, which sorts by `WorkPriority.rank`). Work the order marks "not taken on until nothing else is waiting" (Easy and Flat trays by default) comes after everything else, whatever its group. At launch the pool is just the list of bundle folders in the queue folder, by name -- under a second, where reading every bundle took about 6 minutes (changed 2026-10-23). Each pick then checks only the video it picked against the rules below (`_pick_eligible`); one that fails is dropped from the pool and the pick is made again. Each "next video" draws from the pool minus what was already reviewed, and the folder is listed again when it runs out (`_load_next_video`). The random draw inside a group is deliberate: it keeps the reviewed set spread across cohorts and dates within that group. The status line says how the video was chosen ("top-priority pick", "priority pick (group N)", or "random pick" when no order is set). Until 2026-10 the pick was random over the whole pool.

A picked video is shown when all of these hold (`_bundle_eligible`; `_bundle_needs_review`):

- it is a directory containing a manifest file,
- segmentation did not fail — confidence at or below 0, or an anomaly mentioning "reference quality" (`_segmentation_failed`, `:2610`),
- it has at least one triaged element (`_bundle_has_triage`, `:2626`),
- no ground-truth file exists for the video anywhere — ground truth counts as the answer (`causal_review_widget.py:2604-2605`; `has_gt`, `causal_review_io.py:596`),
- its triaged elements are not already all resolved (`_bundle_reviewed`, `:2539`).

The failed-segmentation skip (`_bundle_eligible`) runs in **both** modes, including re-segmentation. That is a problem for the Re-segmentation Review tool: see [Re-segmentation Review](#re-segmentation).

### What counts as a triaged element

A segment is triaged when either:

- the outcome detector wrote `outcome == "triaged"` or `flagged_for_review: true`, or
- the outcome is a *touched* one — `retrieved`, `displaced_sa`, `displaced_outside` — and the reach-assignment file has no reach marked `is_causal` for that segment. The pellet moved, but nothing says which reach moved it.

That rule exists in `triage_status.triaged_segments` (`triage_status.py:70`) and is written out a second time, identically, inside the widget (`_bundle_has_triage`, `causal_review_widget.py:2626`; the walk's own copy at `:1087-1088`).

A **third** version exists in the worklist builder used by the older clearing tool, and it is **not** the same rule (`triage_queue.py:196-222`). It never tests `outcome == "triaged"`; it also triggers on a `flagged_for_review` flag in the *reaches* file, which `triage_status` never reads; its test for "has a committed causal reach" also accepts a reach whose `label` starts with `causal` (`triage_queue.py:107`), which `triage_status` does not; and it drops segments already marked `triage_cleared` unless `--include-cleared` is passed. In practice the missing `outcome == "triaged"` test does not change the result for v6 output, because the v6 cascade always writes `flagged_for_review: true` alongside a `triaged` outcome (`outcomes/v6_cascade/detector.py:260-267`). The other three differences are real.

In triage mode the walk visits **only** triaged segments (`_visible_indices`, `:1122`). Everything else is skipped.

### What the person is shown and asked

The whole video is loaded once as a decode-on-demand layer; navigation only moves the playhead (`_load_video`, `:487`). The view opens on the relevant reach — the reviewer's own earlier answer if there is one, otherwise the algorithm's causal reach, otherwise the reach nearest the interaction frame (`_relevant_reach`, `:590`) — landing 25 frames early (`OPEN_LEAD`, `:485`), with 45 frames of padding around the reach (`WINDOW_PAD`, `:482`; `_compute_segment_window`, `:635`). **An `untouched` call opens 30 frames before the segment's end instead, whether or not the mouse reached** (`_opens_at_end`, since 2026-10-24): untouched is judged by where the pellet is when the segment ends. Before that, an untouched pellet with no reaches opened at the end but one with reaches -- every hand-score mismatch the algorithm called untouched -- opened on a reach, in triage a median of 172 frames and up to 1,825 frames before the end, so the reviewer had to skip forward to see it. If the reviewer changes the call and picks a reach, the pellet opens on that reach. Tracking points are drawn as coloured dots; in re-segmentation each also carries its name, because the names jumping is what tells a reviewer it is time to call a new segment, and triage leaves them off for speed. There is no pillar circle (dropped for speed, 2026-10-23).

The question panel takes one of three shapes, decided by `_segment_triage` (`:1770`):

- **`reach_uncertain`** — the outcome is already committed (`retrieved`, `displaced_sa`, `displaced_outside`, `abnormal` or `abnormal_exception`, `:1790-1793`); only the reach is open. Asked: *"Which reach caused it?"*, with an outcome-correction dropdown underneath.
- **`outcome_uncertain`** — exactly one candidate reach survives narrowing, so the reach is effectively pinned. Asked: *"What did it do to the pellet?"*, plus an optional reach override.
- **`both_uncertain`** — neither is known. Both questions.

The reach list is narrowed by `assignment.v2.triage_reduction.reduce_triaged_segment` to reaches that could not be ruled out as misses (`_candidate_reach_ids`, `:1675`); the ruled-out ones are built too but hidden behind a "Show all reaches" checkbox (`_make_reach_picker`, `:1859`). There is an explicit "No reach — nothing acted on this pellet" option (`:1906`), and choosing `untouched` clears and disables the reach picker so the two answers cannot contradict each other (`_sync_reach_picker_to_outcome`, `:2029`).

**A reach the list does not have: "Undetected reach (enter frames manually)".** Type its first and last frame. What is saved (since 2026-10-26):

- If a detected reach covers at least half of the frames you typed, that detected reach is the answer -- it is the same reach, and the detector's frames are kept (`causal_review_io.typed_reach_match`).
- Otherwise it is a reach the detector missed. Your frames are saved as they are, and when the results are made it is added to the pellet's reach list -- exactly as a ground-truth reach is: it replaces any detected reach it overlaps, its measurements are computed over your frames, and it is the reach credited with the pellet (`reach_source: human_review`).
- Either way the record keeps the frames you typed, as `typed`.
- A typed reach longer than 300 frames is refused on the spot ("The reach you typed runs N frames ... no reach lasts that long"): no real reach comes close (detected reaches: median 13 frames, 99.9% at most 94), so it is a mistyped frame number, and measuring it would put nonsense into the data. The results step never adds one either (`MAX_TYPED_REACH_FRAMES`).

Until that date a typed reach was swapped for any detected reach touching it by a single frame, and a reach the detector missed was never added, so its pellet's credit was lost or landed on a neighbouring reach (of 199 typed reaches on 2026-10-25: 14 lost the causal reach, 6 pointed to another reach, 2 ended with an outcome other than the person's).

Every segment also gets an **"Ignore windows"** box — frame ranges where something that is not a reach moved the pellet (`:1369`, `_make_abnormal_ranges_widget`, `:2161`) — and a free-text notes box. There is no control for saying the cuts are wrong: a wrong pellet or wrong cuts is never answered here, it is escalated (see "Escalating out of triage").

Two more buttons: **Flag Session** (`:2514`) and, in re-segmentation mode only, **Clear -> re-enter pipeline** (`:3069`). The tool cannot re-cut a video (its re-cut editor was removed 2026-10-23); cuts are fixed in re-segmentation.

Keyboard shortcuts are playback only — space, `b` for reverse, arrows, and speed keys 1-6 (`_setup_keybindings`, `:2963`). There is no key for next/previous segment; those are buttons.

**The keys work while the video has the keyboard; what you type in the panel stays in the panel** (since 2026-10-24, `review/keyboard.py`; the Re-segmentation tool works the same way). Click the video to give it the keyboard. While the keyboard is in the review panel -- a number box, the notes, a button or choice you just clicked, the outcome list -- letters, numbers, Backspace, Delete and Enter never run a key; Space and the arrow keys still drive the video when the control you are on does not use them, so stepping after picking an outcome works as before. Choosing "Undetected reach (enter frames manually)" (or "No, it is at frame:" in the Re-segmentation tool) puts the keyboard in its number box, so the frame you type next lands there.

Why: Logan, 2026-10-24 -- "when I type numbers into text fields ... sometimes keybinds go off". A number box keeps the digits typed into it, but the keyboard was often still on the control clicked just before (that choice, a "Use current frame" button, the outcome list), and napari hands any key a panel control leaves unused to the video's keys. Those were napari's own keys for the selected layer -- the tracking dots, added last -- not the review tool's: 1 deleted selected dots, 2-5 switched the dots to add, select, pan or transform mode (in add mode a click on the video adds a dot), Delete and Backspace deleted dots, and on the video layer 2 let a drag move the picture (and from a panel button, napari's Ctrl+Backspace deleted the dots layer itself). So the speed keys 1-5 never set the speed. Now the review tool's keys are bound on its video and dots layers too and napari's editing keys are switched off there: a review window edits nothing.

### Why this video is here: the banner (since 2026-10-23)

Everything that holds a segment in triage is on screen, because the window is
all a reviewer sees (Logan, 2026-10-23). A purple banner above the video says it
for the segment on screen only, in as few words as possible:

```
Open: outcome + causal reach (possible tracking artifact)
```

- **Open** -- what is open on this segment, in the same words as its question:
  *causal reach*, *outcome*, or *outcome + causal reach*. In brackets, the reason
  only when there is one beyond the open question itself: *possible tracking
  artifact*, *reach at segment edge*, *pellet out of view*, or *hand score
  mismatch (hand: X, algo: Y)* when the database's nightly check against the
  hand-scored sheet sent a finished video back. A note a person wrote when
  sending it is shown as written.
- **Nothing else** (since 2026-10-24). The tool never says that a segment was
  re-cut, that a person answered something before, or how: a reviewer answers
  what is open, and those words only made it unclear what to do (Logan: "If
  something is in triage it is because we have no idea what the correct answer
  is for that segment. Don't tell me it was resegmented or previously answered
  by a human."). Pellets already answered are not walked at all (see "Triage
  walks only the pellets still open"); an answer given earlier in the same
  session is filled in again if you go back to it, without comment. How
  earlier answers were matched is logged at debug level only.

Part of every triage answer is checking that the event happened to the pellet
shown. If it did not, answer nothing and press **Escalate: bad segmentation ->
re-segmentation** (see "Escalating out of triage").

### Fitting a small screen, and plain words (since 2026-10-23 evening)

- The reach list offers only reaches inside the pellet's own frames. The reach
  file groups reaches made before the first pellet arrived into segment 1, and
  the list used to offer them as answers for pellet 1.
- Why a pellet was flagged is a short label (see the banner above), never the
  outcome step's internal code.
- The panel fits a 1366-pixel screen: speed buttons and the action buttons are on
  their own rows and long questions wrap.
- `mousereach-review-tool` opens with napari's layer panels hidden and the
  no tracking-point names in triage, so the video gets the room (Window menu
  brings the panels back; re-segmentation shows the names).

### An answer that cannot count is not recorded (since 2026-10-23)

Two answers can never be used for the segment they are given for, and the tool
refuses them at the moment of answering, with a message saying why and what to
do -- nothing is recorded for that segment, and *Save & next* stays on it:

- **a reach outside the segment's frames.** As the video is cut, that reach
  cannot have touched this segment's pellet;
- **any answer on a segment too short to be a pellet presentation** (fewer than
  5 frames: two cuts almost on top of each other), unless a person made the cuts.

The same two tests decide what the pipeline drops when it applies answers
(`causal_review_io.answer_problem`, shared with `index_review_by_segment`), so
the tool never accepts an answer the pipeline will throw away. Part of triage is
checking that what happened, happened to the pellet shown: when the algorithm
is on the wrong pellet or segment, **do not answer -- write what you saw in the
notes and press *Escalate: bad segmentation -> re-segmentation***. The cuts are
fixed there, by a person, in the re-segmentation tool.

A triaged segment that is too short to be a pellet is never left in triage at
all: the gate sends the video to re-segmentation with the reason "segmentation:
segment(s) N ... too short to be a pellet presentation", and the return scan
moves any such bundle already sitting in triage. If a person has cleared the
video in re-segmentation and kept those cuts, there is no pellet in such a segment
to ask about, and it stops holding the video.

### Escalating out of triage (since 2026-08-25)

Part of triage is checking that the pellet on screen is the pellet the
segment claims to be. If it is not -- the cuts are offset, missing or
misplaced -- the reviewer answers nothing and presses **Escalate: bad
segmentation -> re-segmentation**. It saves the answers given so far (they
re-attach by frame span after the boundary fix), moves the bundle to the
re-segmentation queue with the notes text as the routing reason, and loads the
next video (`_escalate_to_deep_review`).

**Escalating cancels any earlier re-segmentation clear (since 2026-10-27).**
Pressing the button means a new problem with the cuts has been found, so a
video that was cleared in re-segmentation before is no longer cleared. Its old
`{stem}_deep_review_cleared.json` is renamed to
`{stem}_deep_review_cleared_superseded_<time>.json` (kept, never deleted) as
the video moves, and it waits in re-segmentation until someone clears it again.
Only the button does this (`escalate_to_deep_review` in `review_gate.py`);
the watcher's own routes into re-segmentation happen only when there is no clear.
Before this, the old clear went along with the video, the watcher read it as
"done" and released the video within seconds, and it came back to triage on
the same cuts -- the button seemed to do nothing.

**Nothing leaves the re-segmentation queue until it has been re-segmented
(since 2026-10-27).** A clear is not enough on its own: the video stays
until a person has set or confirmed every cut AFTER it arrived in the queue
-- by saving in the segmentation fixer (which marks the whole list of cuts
as theirs), or in a ground-truth file with every boundary verified. Cuts set
by hand before it arrived do not count: a video sent back is there because
those cuts were found wrong. Pressing Clear on a video nobody has
re-segmented says so ("cannot be cleared yet -- not re-segmented: ...") and
leaves it in place; a complete answer-save there keeps the answers and says
the same. The Review Queues tab and `mousereach-review-release` list these
videos as "not re-segmented since they arrived". Easy and Flat trays are
never segmented, so the rule does not apply to them. The check is
`review/resegmented.py`; the watcher applies it in the return scan
(`review_return.py`), so a clear written any other way is held too.

Until 2026-10-23 each segment also had a box, "Segmentation says this is
segment N. If that's wrong, which segment is this actually?", which wrote
`true_segment_num` + `segmentation_wrong` into the record. It was removed
because the button does the same job in one step. Records written with it
are still honoured: at return time the watcher diverts such a video to the
re-segmentation queue (`review_return.py`, `TriageStatus.seg_pending_reseg`), and
once the cuts are hand-fixed (`boundary_source: "human"`) the old correction
stops blocking it.

### A hand-score mismatch opens on the algorithm's answer (since 2026-10-24)

When the bench check sends a finished pellet back because the hand-scored sheet
disagrees with the algorithm (banner: *hand score mismatch (hand: X, algo: Y)*),
the outcome list opens with the **algorithm's outcome selected**, not
`-- select outcome --`. The algorithm is usually the one that is right (Logan,
2026-10-24), so if it is, "Save Segment + Next" is all it takes, and that records
"the algorithm was right". Every other outcome stays in the list for when it is
not. A pellet the algorithm itself could not decide still opens on
`-- select outcome --`: there is no outcome to offer (`_make_outcome_picker`,
`is_bench_mismatch`).

### A pellet asked again: "are you sure?" (since 2026-10-26)

A pellet a person already answered can be sent back to triage on purpose, when
there is reason to think the review slipped -- for example an answer of
"retrieved" where the frames show the pellet lying in the scoring area for the
rest of the segment (Logan: "toss these cases back at the review queue and be
like - are you sure"). It arrives like any other triaged pellet, and its banner
says why it is asked again and what the answer on file is. Answer it as usual:
keep the earlier answer if it was right, or change it. Saving that pellet ends
the question; until then the video stays in triage.

What is kept: the earlier answer is not lost. The review file holds it under
`answers.recheck.previous` (with the question and when it was asked), and a copy
of the file as it was is in `Analyzed/Archive/manual_corrections/` named
`<video>_causal_review.json.before_recheck_<time>`. How a pellet is sent back:
`mousereach-route-to-queue --queue triage --recheck --worklist <file>` (without
`--recheck`, a pellet a person answered is never sent back).

### "Check your answer": when both the algorithm and the tray sheet disagree (since 2026-10-28)

**When the box appears.** You pick an outcome for a pellet and save it ("Save
Segment + Next", "Save Review", "Next Segment" or "Prev Segment"). If your
answer disagrees with what the algorithm said for that pellet **and** with the
tray sheet's hand score for that pellet, a box appears before anything is
saved. It says plainly what each one said, for example:

> Algorithm: displaced (in SA). Tray sheet: displaced. You: retrieved.
>
> Did you watch well past the causal reach? A pellet can vanish for a moment
> and still be in the SA.

The question is there because stopping too soon after the reach is the usual
way this goes wrong: a pellet knocked off the pillar can bounce out of view for
a few frames and still be lying in the SA afterwards.

**The two buttons.**

- **Take another look** (the highlighted button; pressing Enter, Esc or closing
  the box does the same). Nothing is saved. You stay on the same pellet, the
  video stays where it was, and you can keep watching. Save again when you are
  ready -- if you still give the same answer, the box asks once more.
- **Keep my answer.** Your answer is saved and the tool moves on exactly as it
  would have without the box. The saved answer also notes that the box was shown
  and that you kept your answer (`answers.double_disagreement` in the review
  file), so it can be found later.

**Why it exists.** Answers that looked like tiredness mistakes were found among
finished pellets: "retrieved" where the algorithm and the tray sheet both said
displaced. You are watching the video, so your answer outranks both of them --
the box never changes or refuses your answer. It is only a speed bump in front of
the one kind of answer most likely to be clicked on autopilot: one that two
separate readings of the same pellet both contradict.

**When the box does NOT appear.**

- Your answer agrees with the algorithm, or agrees with the tray sheet.
- The algorithm had no outcome for the pellet (it could not decide).
- There is no hand score to compare with: no scores file is set up on this
  computer, the file cannot be read, or it has no row for this session or this
  pellet. The answer is saved as usual and one line in the console says
  `[!] Algorithm-and-sheet check not run for <video> segment <n>: <why>`.
- The answer is not a pellet outcome (for example `abnormal`).
- You already saw the box for this pellet and kept this same answer.

**How "disagrees" is judged.** Each comparison is made only at the detail the
other side can see:

- *The tray sheet* records a pellet as untouched, displaced or retrieved. It has
  one "displaced", so it cannot tell "displaced in the SA" from "displaced
  outside the SA": either of those answers agrees with a sheet that says
  displaced. The answer is put on the sheet's scale with the same table the
  before-filing hand-score check uses (`watcher/hand_scores.py`, `ALGO_CLASS`).
- *The algorithm* cannot see a pellet leave the SA. A pellet that leaves unseen
  looks to it exactly like one that was eaten, so it calls it "retrieved" (it
  never says "displaced outside SA"). Against the algorithm, "retrieved" and
  "displaced outside SA" therefore count as the same answer ("the pellet left
  the SA"). "Retrieved" against the algorithm's "displaced (in SA)" is still a
  disagreement: the algorithm saw the pellet land in the SA.

The hand scores are read from the same scores file the before-filing check
reads (the `"hand_scores"` `"path"` setting in this computer's
`~/.mousereach/config.json`; see `watcher/hand_scores.py`). The decision itself
is `review/double_disagreement.py`; the box is `_passes_double_check` in
`causal_review_widget.py`. If you press Escalate, Flag Session or Clear while the
open pellet's answer would bring up the box, the box comes first; "Take another
look" then stops that button too: the video is not moved (Flag Session has
already recorded the session flag itself; press it again when ready).

### Two ways an unanswered question becomes an answer

Both matter before trusting `outcome_source: "human_review"` in the data.

1. **Any navigation records the current segment.** "Save Segment + Next" (`:2480`), "Next Segment" (`:2939`) and "Prev Segment" (`:2930`) all call `_collect_answers()` and store the result. There is no "I did not answer this" state.
2. **`_collect_answers` starts from the algorithm's values.** It initialises `human_outcome = seg["outcome"]` and `human_causal_reach = seg["causal_reach"]` (`:2299-2300`) and only replaces them where the reviewer actively picked something. The reach picker also pre-selects the candidate nearest the interaction frame (`:1946-1955`).

The consequence: clicking past a `reach_uncertain` segment without touching anything writes a record whose `human.outcome` is the algorithm's outcome and whose `human.causal_reach` is the pre-selected reach -- a confirmation of the highlighted reach, by design.

**Leaving `-- select outcome --` alone leaves the pellet OPEN (since 2026-10-26).** On an `outcome_uncertain` or `both_uncertain` segment where no outcome is picked, the record is saved with `answers.reviewed: false`, and the status line says `[OPEN] Segment N left open -- no outcome was picked`. The video stays in triage until someone picks an outcome (or presses Escalate, if the pellet or the cuts are wrong). Moving on without picking is allowed -- it is how you come back to a pellet later -- but it is never an answer.

Before that date the record was saved with `human.outcome: "triaged"` and nothing marking it open, and every check counted it as answered: the video left the queue, was filed, and the results step stamped `"triaged"` onto the pellet as `outcome_source: "human_review"`. On 2026-10-26 that was 125 records in filed videos and 121 in the queues; their pellets reached the exports as "Undetermined". One rule now says what an answer is -- `causal_review_io.is_answered`: not a `reviewed: false` placeholder, and, for a pellet, an outcome other than `triaged`, `uncertain` or none -- and the release check, the gate, the hand-score check, routing, the restore and the results step all use it. Records saved the old way therefore count as open wherever they are, and the review window shows those pellets again.

### What is written

`_save_review` (`:2844`) writes `{stem}_causal_review.json` into the bundle directory (`_review_dir`, `:880`) **and, since 2026-08-24, a durable copy at `{NAS}/review_records/reviews/{stem}_causal_review.json` first**. The bundle copy is the one the reviewer edits; the durable copy is the one nothing regenerates. Before this, a bundle-resident review was destroyed by any reprocess that recreated the bundle, and 41 reviewed videos had no other copy anywhere. It contains **one record per segment in the video**, not per reviewed segment: segments the reviewer never visited get a placeholder with `answers: {"reviewed": false}` (`:2867-2880`). It also calls `update_corpus_index` (`causal_review_io.py:690`), which appends to `{NAS}/review_records/causal_review_index.json`; failures there are printed and swallowed (`:2910-2911`).

On reopening, `_load_saved_review` drops the placeholders and jumps to the first segment no one has answered. An answer stands whatever the algorithm has said since -- a person's answer is what happened in those frames (until 2026-10-24 a changed algorithm call re-opened it, and reviewers were asked again about pellets they had answered).

**Triage walks only the pellets still open when the video loads** (since 2026-10-24, `open_triaged_indices`). A video can come back -- the nightly hand-score check sends finished videos back for pellets nobody has looked at -- while its pellet file still flags the pellets a person answered before; "Next" used to walk those too and show the earlier answers. Now they are skipped (the console says how many), and a pellet answered in this session stays reachable with Prev.

**Since 2026-10-22 the restore matches each saved answer to the segment whose frames it covers** (`restore_review_for_segments`), the same rule the release check and the results step use. Until then it matched by segment number. After a re-cut renumbered the segments, an answer about other footage came back on today's segment N, so the tool counted N as done and never showed it. The release check, which matches by frames, still counted N as unanswered, so the video could never be finished. The reviewer's new answer for the old frames was then saved beside the old one, and the next save stamped segment N's frames onto the old answer. What the restore does now:

- An answer goes back on the segment whose frames it covers, under that segment's current number.
- An earlier answer whose frames match no current segment is **kept in the file** under `unmatched_answers`. It is never applied to anything, and it is never deleted by a later save.
- A segment with no answer covering its frames is shown for review, even if an answer once carried its number.

### The record format

Written by `build_segment_record` (`causal_review_io.py:167`):

```
segment_num       the segment's number at the time of review
segment_span      {"start": ..., "end": ...} -- the frames the reviewer saw
pellet_num
algo              {outcome, causal_reach, interaction_frame}   as it stood
human             {outcome, causal_reach, is_phantom, agreed}
answers           {triage_kind, outcome_pick/outcome_override, causal_pick,
                   abnormal_ranges, reviewed: false for placeholders,
                   double_disagreement: present only when the "check your
                   answer" box was shown and the reviewer kept the answer:
                   {shown, kept, human_outcome, algo_outcome, hand_score,
                   hand_word, at}}
notes             free text
true_segment_num  older records only (the control was removed 2026-10-23)
segmentation_wrong  true, alongside true_segment_num
```

New saves stamp the document `schema_version: "1.1"` (`causal_review_io.py:268`). Older files were given spans by a backfill that left the version alone, so **no existing review file carries 1.1**: all 662 review files under `MouseReach_Pipeline/Analyzed` say `1.0` and all 662 have spans.

`is_phantom` is always `False` — nothing sets it (`causal_review_widget.py:2301`, `:2875`, `:2291`).

`provenance` is filled by `collect_provenance` (`causal_review_io.py:112`), which reads version stamps out of the four algorithm JSONs and the tracking-model name out of the pose file. Every read is wrapped in a bare `except`, so a missing key means "could not read it", not "not applicable". It also captures less than it looks like it does: it only copies keys named `version`, `detector` and `segmenter_version` (`:137-142`), while the reach detector and the outcome cascade both write their version under `detector_version`. In real files this shows as `"reach_detector": {}` and `"outcome_detector": {"detector": "v6_cascade"}` with no version at all — verified in `20250624_CNT0110_P2_causal_review.json` and others.

---

## Working the re-segmentation queue: the operator's walkthrough

> **Names (2026-10-27).** The queue for videos whose cuts need fixing is the
> **re-segmentation queue**; the review tool opened on it is **Re-segmentation
> Review**. Both used to be called "deep review" -- a name kept for
> ground-truthing with the ground-truth tool, which is a different job. The
> folder (`Processing/Review/deep_review`), the database state `deep_review`
> and the clear file `{video}_deep_review_cleared.json` keep their old names:
> every machine reads them.

This section is for the person doing the work, no codebase knowledge assumed.
Everything below it in this document is the engineering detail behind these
steps.

**Opening the queue.** Open an **Anaconda Prompt** (Start menu -> "Anaconda
Prompt"), activate the mousereach environment, and launch:

```
conda activate <mousereach env>
mousereach
```

On another lab machine, activate that machine's own mousereach environment
(the path after `conda activate` differs -- e.g. under `A:\` on the DLC PCs;
`conda env list` shows what the machine has). Note the bare `mousereach`
command only exists AFTER activation -- a plain PowerShell/cmd/VS Code
terminal without the activation says "not recognized". (Fallbacks if conda is
unavailable: the **MouseReach** desktop shortcut on the processing server, or
the executable's full path,
`<mousereach env>\Scripts\mousereach.exe`, which works from any
terminal.)

When napari opens, find the **Review Queues** tab in the right-hand dock. It
shows how many videos are waiting in each queue and has three buttons: *Open
Triage Review*, *Open Re-segmentation Review*, and *Open Re-segmentation*. Each opens in
its own window.

**Which button to press.** Deep-review videos are there for one of two broad
reasons, and each has its tool:

- **The segmentation is the problem** (the boundaries between pellet
  presentations are wrong -- this is most of the queue, and every video the
  triage escalate button or a corrected segment number sent here). Press
  **Open Re-segmentation**. The tool walks the video one BOUNDARY at a time:
  it parks the playback ON each boundary and asks "Is this the boundary?" --
  is this frame the first one after the scoring-area jump. The header states
  explicitly which pellet is BEFORE the boundary and which is AFTER it
  (boundary 1 has the pre-roll before it and pellet 1 after; the last
  boundary has the post-roll after it) -- check the footage on both sides
  matches. The DLC tracking points are drawn on the video; the scoring-area
  points jumping is the tell. Answer Yes with a click, or answer No by
  clicking one of the offered candidate frames (each click shows you that
  exact frame), by scrubbing there and pressing "Use current frame", or by
  typing the frame number. If there is no tray advance near the claimed
  boundary at all, press *No advance here -- remove this boundary* (the two
  segments merge and everything renumbers); if a boundary was MISSED, add a
  cut with the manual controls below the questions. Press *Confirm -> next
  boundary* each time; press *Save these cuts* when the video is done.
  Saving marks the cuts as human-made -- the pipeline will keep them.
- **Something else is wrong with the whole video** (escalated for a reason
  that is not boundaries, or you need to re-judge every segment's outcome).
  Press **Open Re-segmentation Review**. It walks every segment of the video asking the
  same outcome/causal-reach questions as triage review.

**Finishing a video.** Fixing the cuts does NOT release the video by itself.
After the cuts are right (or after a full re-segmentation), open **Re-segmentation Review**
on that video and press **Clear -> re-enter pipeline**. That is the release:
the watcher then re-runs the video from segmentation onward -- keeping any
human-made cuts -- and the results flow to the database automatically. Nothing
else needs to be done.

**If you get a video that is not actually broken**, press *Skip this video* in
the re-segmentation tool (it stays in the queue), or clear it through Deep
Review if you have judged every segment and all is well.

---

## Re-segmentation Review

The same widget with `deep_review=True` (`causal_review_widget.py:198-203`). Differences:

- It reads the `DEEP_REVIEW` queue, not the triage queue (`queue_launcher_widget.py:123-126`).
- `triage_only` is forced off (`:210`), so **every** segment is walked.
- The **Clear -> re-enter pipeline** button appears. It saves the review, then writes `{stem}_deep_review_cleared.json` into the bundle (`_clear_deep_review`, `:3069`, marker at `:3085`). That marker is what the watcher looks for.
- The same marker can be written in bulk from a terminal: `mousereach-review-release` lists every re-segmentation bundle's completeness (judged on `human.outcome` per segment -- never `answers.reviewed`, which nothing has ever written) and with `--clear` releases the fully-answered ones. It is the button's bulk form for reviews completed without the second step, not an automatic release.

It has no console command. It opens from the "Review Queues" launcher tab (`queue_launcher_widget.py:123`) or from the pipeline dashboard for one named video (`dashboard/widget.py:1488`).

**It cannot open the videos it exists for.** The bundle pool skips any bundle whose segmentation failed (`:2665-2666`), and "segmentation failed" is the main reason the gate routes a video to re-segmentation in the first place (`review_gate.py:87-88`). Of the 120 bundles in the re-segmentation queue right now, 17 meet that test and are invisible to the tool; one more has no segments file at all, which the gate counts as failed (`triage_status.segmentation_failed`, `triage_status.py:50-51`) but the widget counts as fine (`:2610`, which returns `False` on any read error). The Segmentation Fixer below is the tool that can open them.

---

## Segmentation Fixer (the re-segmentation tool)

`mousereach-fix-segmentation`, or the **Open Re-segmentation** button on the
Review Queues tab (`queue_launcher_widget._open_reseg`). Segment cuts only: no
outcomes, no reaches, no causal attribution.

It walks the re-segmentation queue (or `--queue-dir`) and offers bundles that need a
human's cuts, three ways in (`_needs_reseg`): the segmenter's own non-empty
`needs_human` list (its record of having had to force its answer, written by
`save_segmentation`, `segmentation/core/segmenter_robust.py:931`); a
reviewer-declared segment mislabel (`segmentation_wrong` in the bundle's
travelling causal review); or a routing reason naming segmentation (the triage
escalate button, or the watcher's mislabel diverts). One way out:
`boundary_source: "human"` in the segments file means the cuts were already
hand-fixed, and the bundle is not offered again. Videos with the most unused
candidate cuts are offered first; videos with no candidates at all are last,
because those have to be marked from scratch. Which videos come first is now
also subject to the lab's work order (`priority_order.json`, see
docs/WORK_PRIORITY.md); the most-unused-candidates rule breaks ties within a
group. Easy and Flat videos (idle-only in the order) come after every other video,
whatever group they belong to (since 2026-10-22; before, a group was ranked by
the order alone).

Until 2026-09-13 this was the one review tool that could not play a video at
all -- only step frame by frame. It now has the same transport and speed
buttons as the other tools (Rev / Play / Stop and 0.25x to 8x), driven by the
video's real frame rate.

**The guided walk** (added 2026-08-25, redesigned boundary-centric the same
day) is how the tool opens: one BOUNDARY at a time, playhead parked on it. A
segment-based draft asked three questions per segment and thereby asked about
every interior cut twice (segment k's end and segment k+1's start are the
same cut) and never visited the pre-roll; the boundary framing fixes both.
Per boundary:

- The header states the boundary's role explicitly: which pellet (or
  pre-roll/post-roll) is BEFORE it and which is AFTER it.
- *Is this the boundary?* -- is this frame the first one after the
  scoring-area jump, within 10 frames. Yes (pre-selected), or the real frame
  -- typed, taken from the playhead ("Use current frame"), or picked from the
  **candidate chips**: the segmenter's own nearby candidate tray advances,
  each labelled with its frame plus its evidence (`n_proposers`/4, consensus
  score). Clicking a chip jumps the playhead there and fills it in as the
  answer. Chips within the question's 10-frame tolerance are hidden (same
  answer as Yes).
- *No advance here -- remove this boundary* deletes an invented cut in one
  click: the two segments it separated merge, and every later segment
  renumbers automatically.

Corrections apply to the cuts immediately; the per-boundary answers are saved
into the segments file as `guided_walk` records ({boundary_num, algo_frame,
confirmed, corrected_frame?/removed?}). Below the walk, the full candidate
table and manual add/drop-cut controls remain -- adding a MISSED boundary
happens there ("Add a cut at the current frame"), and the walk picks it up
immediately. The DLC tracking points are drawn on the video, and video
navigation (play/reverse/speeds, frame steps, go-to-frame, and the causal
review tool's exact keyboard scheme) is identical to the other review tools.

Saving copies the original aside, then rewrites `{stem}_segments.json` with
`boundaries`, `algo_boundaries` (what the algorithm had), `boundary_source:
"human"`, `corrected_by`/`corrected_at`, `needs_human` emptied, and
`guided_walk` (`_save`).

Two things it does not do:

- **It releases the video only when the routing reason names segmentation.** On save it writes `{stem}_deep_review_cleared.json` when the bundle's recorded routing reason mentions segmentation/mislabel -- fixing cuts is then exactly the work the video was routed for. A bundle routed on other grounds (a QC hold, an unrecorded reason) is deliberately NOT released by a cut-fix, because the clear marker is a blanket human-clear token the watcher gate honors; the status bar says so and the Re-segmentation Review tool's Clear (or `mousereach-review-release`) remains the path out once the real concern is resolved.
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

Two consequences of that third write are easy to miss. The object it saves is `self.gt`, the *unfiltered* one seeded from the algorithm — rows nobody touched are in it with `determined: false`. The truth layering skips undetermined rows (`truth_resolver.py:138-139`), so no false values leak into the numbers, but **the file's existence alone** makes `has_gt()` true (`causal_review_io.py:596`), which (a) removes the video from the causal review tool's queue as "already ground-truthed" and (b) in a re-segmentation bundle counts as a clearance all by itself (`review_return.py:112-114`).

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
- Spans shorter than **5 frames** (`MIN_REAL_SEGMENT_FRAMES`) are excluded on BOTH sides before scoring: a zero-length segment is not footage, and because the score normalises by the shorter range, a 1-frame segment inside any review span would score 100%, tie the true target, and the tie would drop the human's answer (seen live 2026-09-01). A degenerate current segment is skipped as a target; a review recorded against one is dropped, each with a note.
- The best match must beat the runner-up by at least **0.15** (`MIN_MARGIN`, `:346`). If it does not, the reviewed stretch has been split across two new segments and the record is **dropped**, with a note (`:423-430`).
- The best match must cover at least **0.5** of the shorter range (`MIN_SPAN_OVERLAP`, `:342`). Below that the record is **dropped** as describing different footage (`:432-437`).
- If two records both want the same current segment, the weaker overlap is dropped (`:439-445`).
- Where a record survives but lands on a different number than it was written with, a note records the move (`:449-453`).

Notes are returned to the caller and logged at warning level (`truth_resolver.py:109-110`, `causal_review_io.py:536-537`). Nothing is written back to the review file; the matching is redone every time.

`true_segment_num` / `segmentation_wrong` (found only in records written before 2026-10-23, when the control that wrote them was removed) play no part in any of this.

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
2. `Processing/Review/deep_review/<stem>/<stem>_causal_review.json`
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

2. **An offset is invisible.** Every segment still looks well formed; only the numbering is shifted, so pellet 7 as scored on the bench is compared against footage of pellet 8. A triage reviewer who sees it escalates the video to re-segmentation, where the cuts are fixed.

3. **A re-cut can silently discard the review.** `index_review_by_segment` drops any review whose frames no longer line up with a current segment. Until 2026-10-21 the release check still counted such a review as resolved by number, so the video finished carrying the algorithm's call. Since then the release check matches by frames as well, and the segment stays open for a person.

Reaches themselves are grouped by segment in two different ways depending on file shape: nested output is trusted as grouped by the detector (`causal_review_widget.py:1024-1030`), while flat output and any manual re-cut are re-grouped by which segment window contains each reach's midpoint (`:1011-1023`).

---

## The queues, and how a video leaves one

### Where they are

Set in `config.py:164-166`, derived from `NAS_ROOT`:

| Setting | Path | Holds |
|---|---|---|
| `Paths.REVIEW_ROOT` | `<nas>/Processing/Review` | parent of both queues; `flagged_sessions.json` and `_QC/` live here |
| `Paths.TRIAGE_REVIEW` | `<nas>/Processing/Review/triage` | per-element questions |
| `Paths.DEEP_REVIEW` | `<nas>/Processing/Review/deep_review` | failed segmentation, failed quality control, escalations |

The re-segmentation folder was once `Processing/Review/flagged_for_review`. That name is retired for the folder only -- `flagged_for_review` is still a field in the segments, reaches and outcomes files and a database column. A migrated drive holds a plain file at the old folder path, so old code that tries to use it fails loudly instead of rebuilding it.

An unset `nas_root` makes all three `None` (since 2026-10-22), and the launcher tab reports "queue not configured" (`queue_launcher_widget.py:112-113`). Run `mousereach-setup`, or use the Setup panel's "Shared pipeline folder" field, to set it. **Why no fallback:** until 2026-10-22 an unset `nas_root` was guessed as `<NAS drive>/! DLC Output`, a layout retired in 2026, so the queue paths resolved to real-looking folders in the old place and the review tools would have worked there as if nothing were wrong. Saving from setup also used to delete `nas_root`, which made the guess a live risk on every machine. `Paths.NAS_ROOT_ORIGIN` is now `config` or `unset`.

### How a video enters

`run_gate` (`review_gate.py:209`) runs after reach assignment and before kinematics. `evaluate_gate` (`:68`) decides, in order:

1. Ground truth marks the video exhaustively complete -> **clean**, proceed, even if segmentation failed (`:85-86`).
1b. An Easy or Flat tray video -> **clean**, always (since 2026-10-23): these trays are never held for a person. They are not segmented and get no outcomes, so there is nothing a reviewer could answer; a machine problem is only recorded in `_triage.json`. The stale-output checks still apply (they send a video round again, not to a person).
2. Segmentation failed — no segments file, confidence at or below 0, or a "reference quality" anomaly -> **DEEP_REVIEW** (`:87-88`).
3. Quality control returned `needs_review` -> **DEEP_REVIEW** (`:107-108`). Quality control calls a missing segments, reaches or outcomes file critical, but asks for the outcomes file only where outcome detection runs: Easy and Flat trays skip it by design (`watcher/router.SKIP_STEPS`), and until 2026-10-22 every Easy/Flat video was held here for that missing file (`pipeline/triage.py`, `check_structural`).
4. Any triaged segment not resolved by ground truth or a saved review -> **TRIAGE** (`:114-118`).
5. Otherwise -> **clean**, kinematics runs (`:119`).

The segmenter's own `needs_human` verdict is read into the status object (`triage_status.py:184`) and deliberately not acted on (`review_gate.py:89-105`).

On a hold, `route_to_queue` (`:159`) moves the whole bundle out of the processing directory, writes a queue manifest so the review tool can open the bundle in place (`:138-156`), and sets the database state to `triage` or `deep_review`. If the normal state transition is rejected the code forces the state rather than letting disk and database disagree (`:178-194`).

A new bundle is built out of sight and appears in one step. `move_video_bundle` (`review_routing.py`) moves the files into `<queue>/.incoming/<stem>/`, writes the routing manifest there, and then publishes the folder as `<queue>/<stem>/` with a single directory rename. **Why:** a move onto the shared drive is a copy that takes seconds per file, and the watcher's return scan reads the queues meanwhile. When the bundle folder was created first and filled in place, the scan saw it empty and filed it away mid-route (every remaining move then failed with "No such file or directory"), and a half-filled folder could be judged as a bundle before its files arrived. The readers that judge, retire or release bundles skip names starting with `.` (readers that key on stem-named files match nothing there by accident, not by rule), and the build folder sits under the queue itself so the rename never crosses a drive. A bundle that already exists (a re-route, or a divert into an existing bundle) is already visible and is filled in place as before. The routing manifest lists `moved_files` and `failed_files` (each file that could not be moved, with the error), so a part-moved bundle shows in the record rather than only in a log. A route that dies partway leaves its files in `.incoming/<stem>/` and the next route of that video reuses them; until then `mousereach-reconcile` lists the video as held, "being routed into <queue>, or an interrupted route -- check if this persists". If a concurrent route of the same video publishes the shared build folder first, the rest of this route's files and its routing manifest go into the published bundle, and a failed manifest write never stops the publish. A file that could not be merged in the fallback stays in `.incoming/<stem>/`, and the return refuses that bundle, naming the folder, until a person folds the file in -- returning would re-run the video without it (typically a locked mp4).

`reprocess_to_current._stage_review_bundle` (`reprocess_to_current.py:66`) stages bundles for the bring-current path and never clobbers a bundle that already exists (`:80-82`), but it does NOT build out of sight: it still creates `<queue>/<stem>/` first and fills it file by file, so a return scan can read it half-filled.

### How a video leaves

**What counts as answered (since 2026-10-21).** A triaged segment counts as answered only when a saved answer is about ITS frames -- the frames the reviewer was shown, stored with every answer -- not merely when an answer carries its number. After a video is re-cut, segment numbers shift; an old answer then counts for the segment whose frames it covers, and any segment no answer covers stays open for review. Short segments a person cut are real segments. The gate, the return scan, this tool's queue and the queue index all use this one rule (`causal_review_io.answered_segments`), so they agree with what the results contain.

**When a file holds two answers for the same frames (since 2026-10-22).** The old number-based restore left 59 review files with two records for one stretch of video, and the results step used whichever came first in the file. That was sometimes the left-blank one and sometimes the older one. The rule now:

| The two records | What counts |
|---|---|
| A decision and a placeholder, or a decision and a left-blank `triaged` | The decision |
| Two decisions that agree | That decision, once |
| Two decisions that **disagree** (a different outcome, causal reach or ignore windows) | **Neither.** The segment counts as unanswered and goes back to a person, who sees it in this tool like any open segment. |

**An answer whose causal reach lies wholly outside the frames it records is not applied.** It is about other footage: these are mostly September 2026 answers that the old restore put on the wrong segment, typically shifted by one. The segment counts as unanswered, and the answer stays in the file.

Only `scan_review_queues` (`review_return.py`) moves a bundle out. It runs inside the watcher every 10th poll cycle (`orchestrator.py:1345`, `:1391-1395`) and handles at most **10 bundles per scan** (`MAX_RETURNS_PER_SCAN`, `review_return.py`) so returning does not starve the pipeline.

Because the queues are shared storage scanned by every watcher node, every mutation the scan performs — a return, a stale-bundle retire, a divert to re-segmentation — first takes a per-video claim marker in `{REVIEW_ROOT}/.return_claims/` (`_claim_return` / `_with_return_claim`, `review_return.py`). The marker is created exclusively (`watcher/claim_marker.py`): the file server creates it only if it does not already exist, so when two nodes race exactly one succeeds, and the marker records which host holds it. A stem another node holds is skipped this pass and re-examined next scan. A marker is broken only when its holder has stopped writing its status report (`Processing/.node_status/<host>.json`) for 3 hours, both times read from the file server so the machines' clock differences do not matter; an old marker whose holder is still reporting is never broken. If the claim folder cannot be made, nothing is moved that pass. With no shared root configured the claim is a no-op. (Until 2026-10-21 the marker was written, re-read after 0.5 s, and broken after 2 hours by age alone.) Scanning a bundle never takes a claim — only touching it does.

**An empty triage folder is left alone for 30 minutes before it is filed away.** An empty folder is not a bundle; it is residue of an earlier divert or return, and the scan retires it to `_Problematic` -- but only once its modification time is older than `EMPTY_DIR_GRACE_SECONDS` (30 minutes, `review_return.py`). **Why:** a router may have just created the folder and still be moving files in (routing into an existing bundle, or after a failed rename, fills the folder in place), and the time is stamped by the share's clock, which can differ from the scanning node's by many minutes. A time in the future, or one that cannot be read, counts as young. Real residue simply waits for a later scan. The same scan warns once per watcher run about any `<queue>/.incoming/<stem>/` older than 30 minutes and never touches it: those files are the video's only copy, and whether the route is still running or died is for a person to judge.

**A bundle of a video that has since been archived is filed away (since 2026-10-22).** If this node's record says the video is `archived` and nobody wrote anything in the bundle (no review, no clear marker, no ground truth), the bundle is a leftover when either (a) it holds no results at all -- typically just the .mp4 a return could not delete while the review tool still had it open -- or (b) every file in it is older than the video's archived results in Analyzed. The scan moves it to `_Problematic` with a note saying which (`_superseded_by_archive`, `review_return.py`), in either queue. **Why:** such bundles asked a person for work already done; between 2026-09-10 and 09-29 twenty videos a person had cleared came back to re-segmentation as a bare .mp4 and fifteen were cleared a second time. A bundle the nightly bench-disagreement scan puts in a queue is never caught: its results leave Analyzed when it is routed, and its routing record is written after the archive. Anything a person wrote keeps the bundle in the queue.

**A triage copy that only repeats the re-segmentation bundle is filed away (since 2026-10-22).** Older escalations sometimes split a bundle: the results moved to re-segmentation and the mp4 stayed in triage. When the same video has a folder in both queues and segmentation failed, the scan compares the two. If every file in the triage folder has a file of the same name in the re-segmentation folder with exactly the same bytes, the triage folder holds nothing the reviewer's copy lacks, and the scan moves it to `_Problematic` with a note saying why (`_is_leftover_copy`, `review_return.py`). The re-segmentation bundle is not touched. If anything differs (a different byte, or a file the re-segmentation copy does not have), both copies stay where they are, and the log says "exists in BOTH queues" each scan until a person sorts them out. **Why:** until this rule, every such leftover sat there for good, and the scan logged the same line every few minutes (one video, 231 times in three days). Sizes are compared first and the files are read only when every size matches. A pair that is not a leftover is read once, not on every scan.

- **Triage**: the bundle leaves when it has triaged elements, all of them are resolved, and segmentation has not failed (`:247`). "Resolved" means a saved review answered the segment **or ground truth determined it** — `triage_status` folds in `gt_resolved_segments` since 2026-08-28, and treats a fully certified video (`completion_status.all_complete`, or reaches and outcomes both exhaustive) as resolved outright.

  Before that fix the release side was narrower than the admission side. `evaluate_gate` had always subtracted GT-determined segments, but `triage_status.resolved_segments` read the causal-review document alone, so a bundle whose questions GT had answered could never satisfy `fully_resolved`. The review tool skipped the same bundle because `has_gt()` said there was nothing to ask. Nobody asked and nobody released: measured 2026-08-28, 7 bundles were stuck exactly there, one of them since July.
- **Deep review**: the bundle leaves when `{stem}_deep_review_cleared.json` exists, or a ground-truth file sits in the bundle (`:108-115`).
- **Deep review, held by the quality check alone (since 2026-10-22)**: a bundle whose routing reason is `qc_needs_review` and that holds no file a person wrote (a review, a clear marker, ground truth) has the quality check re-run on it. The pose-tracking layer is not re-run -- its stored flags are carried over, so a hold it made stays (except a ruler-variation hold: since 2026-10-23 the check only records a ruler that varies within a video, so those holds are released) -- and the file-based layers are. If nothing critical remains, the bundle leaves for a fresh run: its mp4 and pose files go back to the posed folder (`Processing/Posed`), its other files go to `_Problematic/_released_qc_holds_<date>/<stem>/` with a note, and the video's state becomes `dlc_complete`, so intake takes it in its priority turn (Easy and Flat only when nothing else waits). It is not sent straight to `processing`: anything there blocks intake of new work. Nothing in the posed folder is replaced. The posed folder usually still holds the video's files (intake leaves them there until the video is archived); the same file there -- same size and modified time (the quick check rsync and robocopy use; every copy in the chain keeps the time), or the same SHA-256 when the times differ -- is kept as the one copy and the bundle's duplicate is removed, while a copy that differs leaves the bundle for a person. If a move fails, every move is undone and the bundle stays. Re-checks take at most 30 seconds of each scan, and a bundle is re-checked once until its files change (`_qc_hold_lifted`, `_release_to_posed`, `review_return.py`). **Why:** a corrected check otherwise leaves every hold it made by mistake in a person's queue -- the check once demanded an outcomes file Easy and Flat trays never get, and 469 such bundles sat in re-segmentation for that alone.
- **Triage with a failed segmentation**: the return scan diverts the bundle to re-segmentation -- it can never satisfy the triage release condition (which requires a sound segmentation), and before this divert existed such bundles had no route out at all. The processing-side gate honors an existing human clear marker even when the segmentation self-check still reports failure, and since 2026-09-09 the return scan honors the same marker: a bundle carrying `{stem}_deep_review_cleared.json` is never diverted (the re-segmentation question is answered) and may release from triage once its triaged segments are answered, seg self-check notwithstanding. Before that, the gate's honor and the scan's raw `seg_failed` divert fought each other and a cleared video with unresolved triage segments lapped processing -> triage -> re-segmentation -> processing indefinitely (one video made 43 laps in a day).

Returning moves the bundle's data files, including the review file, into the local processing directory, drops the queue-only metadata, deletes the now-empty bundle directory, and sets the video back to `processing` so the pipeline re-runs it and the gate re-checks (`_return_to_processing`). Queue-only metadata is matched by exact name -- `<stem>_manifest.json`, `<stem>_routing.json`, `manifest.json` -- so `<stem>_processing_manifest.json`, the video's provenance record, travels back with the bundle. **Why:** the old suffix match on `_manifest.json` caught the processing manifest too and deleted it from every returned bundle. A normal re-run regenerates it before the review gate; that regeneration is best-effort, though, so if it fails, or a segmentation failure routes the video before it runs, the old manifest stays beside the new outputs. The pose recorded for the re-run is the one `_resolve_inputs` selected (`select_pose_file`), not whichever `.h5` happened to move last. **Why:** a bundle can hold two poses (a re-pose leaves the old model's file beside the new one), and the last-moved rule re-ran such videos on the wrong, often older, pose. Gate-routed bundles carry their own mp4 and pose; only older staged bundles point at an mp4 and pose left in Analyzed. It refuses and leaves the bundle alone if files of the video are still in the queue's `.incoming/<stem>/` build folder, if the database row cannot be created, if the pose file cannot be found, or if the state cannot be set — a clearance is never spent on a run that would fail. The pose is looked for, in order, in the bundle, the bundle's manifest, Analyzed, pose staging (`Processing/Posed`), and -- since 2026-10-21 -- the processing folder the return delivers to and the pose the video's own row records; outside the bundle only a pose of the currently declared model is accepted. **Why the last two:** an older divert between the queues moved only the mp4 into re-segmentation and left the pose and results on the node that ran the video, and two cleared videos then failed to return on every scan for weeks. A return refused for a missing pose or for files left in `.incoming` is tried again after an hour (`RETURN_RETRY_S`), not on every scan, and its ERROR line says so. Each scan re-reads a triage bundle's files only when the bundle has changed since the last scan (its files' names, sizes and modified times) or the video's ground truth has; saving an answer changes the bundle, so it is seen on the next scan.

**No review tool moves a bundle between folders.** The only `shutil` calls in `review/` are the Segmentation Fixer copying a file aside before overwriting it (`fix_segmentation_widget.py:444`) and the ground-truth migration helper (`unified_gt.py:877`). A bundle with a complete review sits in the folder until the watcher's return scan picks it up.

**A triage bundle with zero triaged elements never leaves.** The return condition requires `st.has_triage` (`:247`), so a bundle whose triage flags were cleared by some other route sits in the folder indefinitely.

### The scan-free queue index (no longer read by the tool)

`queue_index.py` keeps a SQLite list at `{NAS}/review_records/triage_queue.db` of videos needing review. It is push/pop: `push` is called only from the bring-current path (`reprocess_to_current._push_review_index`), never by the live watcher, and `pop` happens in the tool when a video's triaged elements are all resolved (best-effort).

Until 2026-10-23 the tool READ this list at launch and, if it held any row for the queue folder, offered only those rows -- so one row pushed by bring-current hid every video the watcher had routed there. On 2026-10-23 it held a single row, for a bundle long gone, so every launch fell back to reading every bundle (about 6 minutes). The tool now lists the queue folder instead (see "How a video is chosen"); the list is still popped, but nothing reads it.

### The session flag button (FIXED 2026-08-28 -- it used to do the opposite of what it says)

**Now:** pressing "Flag Session (needs review)" saves the current review, records the
session in `Processing/Review/flagged_sessions.json` as provenance, and **routes the
video to the re-segmentation queue** via `route_to_queue`, then advances to the next video.
The flag is a note about the session; the divert is the action. Nothing filters on the
flag any more.

**What it used to do.** The button is labelled "Flag Session (needs review)", its tooltip
said it marks every video of that mouse+day as must-be-human-reviewed, and the toast said
"all its videos need human review". `flag_session` wrote the session key to
`flagged_sessions.json` and that was the entire effect. Two pieces of code read that file:
`_bundle_needs_review` and the never-called `seed_from_folder` (`queue_index.py:160`).
The first one did this:

```python
if is_session_flagged(stem, root):
    return False   # this bundle does NOT need review
```

So flagging a session **removed** every one of that mouse+day's unreviewed videos from the
queue, and nothing in the gate, the return scan, kinematics or the database read the flag
to compensate. There was no path by which a flagged session got more attention.

Measured on 2026-08-28, before the fix: of 40 reviewable bundles in the triage queue, 33
were invisible for this reason, across 18 sessions -- 14 of them flagged in a single
afternoon by an operator working through the queue. The queue read as empty and the flag
was the cause. Removing the filter put 24 videos back in front of the reviewer (the other
9 were legitimately resolved by ground truth).

### Automatic release from ground truth

`mousereach-resolve-triage-from-gt` (`triage/gt_resolve.py`) lifts triage flags on segments a ground-truth file already answers, writing the same clearing fields the clearing tool writes. The comment above it in `pyproject.toml:120-123` says it "runs as a step in the normal processing pipeline". It does not — the only caller in the tree is the improvement evaluation harness (`improvement/eval_all.py:163-167`).

---

## What importing the review package loads (since 2026-10-24)

`import mousereach.review` loads only the light parts: the review-file I/O (`causal_review_io`, standard library only). The GUI names it offers -- the three widgets, `ComparisonPanel` and friends, `AlgoGTReviewMixin` -- are imported the first time something uses them (`_LAZY` and `__getattr__` in `review/__init__.py`), so `from mousereach.review import CausalReviewWidget` works exactly as before. Why: the watcher imports this package for `causal_review_io` and `triage_status`, and while the widgets were imported eagerly, a headless watcher loaded napari, Qt, vispy, pandas and scipy (about 2,600 modules, 4.6 s) the first time a cycle touched review code -- and a widget that failed to import took the watcher's review paths down with it. A new GUI module re-exported from the package goes in `_LAZY`, not in a plain import (`tests/test_review_package_loads_gui_lazily.py` fails otherwise).

## Declared but does nothing

Each of these is code that runs, produces a value, and has no reader.

- **Ignore windows / `abnormal_ranges`.** The reviewer marks frame ranges where something other than a reach moved the pellet; the answer is saved, carried onto the segment by both `apply_review_overrides` (`causal_review_io.py:503-504`) and `truth_resolver` (`truth_resolver.py:181-182`) — and read by nothing. There is no reference to `abnormal_ranges` anywhere outside `src/mousereach/review/`. The kinematics step never excludes those frames.
- **`outcomes.exhaustive` in ground truth.** `_seg_overrides_from_gt` computes it and returns it; the caller assigns it to `_exhaustive_out` and never uses it (`truth_resolver.py:328`). Ground-truth outcome overrides apply per determined segment either way, so the flag changes nothing for outcomes. For *reaches*, `exhaustive` is load-bearing and does drop unlabelled algorithm reaches (`_apply_gt_reaches`, `:275-279`).
- **`TriageStatus.clean`** (`triage_status.py:152-156`) — a property with no callers. The gate and the return scan read `seg_failed`, `unresolved` and `fully_resolved` directly.
- **`TriageStatus.seg_needs_human`** (`:137`) — populated from the segmentation file (`:184`) and deliberately not acted on (`review_gate.py:89-105`).
- **`is_phantom`** in every review record: always `False`, never set.
- **Boundary segments.** `is_boundary` is hard-coded `False` (`causal_review_widget.py:1043`) and set nowhere else, so the three branches that handle a boundary segment (`:1236`, `:1284`, `:2281`) never run and the guard at `:1749` is always true.
- **`causal_review_index.json`.** Written on every save (`causal_review_io.py:690`, path at `:729`) so that, per the module docstring, the active-learning loop can bulk-read all reviews. Nothing reads it — no reader exists in `src/` or `scripts/`.
- **`base.py`** — `AlgoGTReviewMixin`, `DiffItem`, `DiffSummary`. Re-exported by `review/__init__.py` (loaded on first use, like every GUI name there) and documented in this folder's `AGENTS.md` as the way to build a review widget. No class in the codebase uses the mixin and no code constructs a `DiffSummary`.
- **`SavePanel`** in `save_panel.py:24` — never instantiated. Its sibling `SimpleSavePanel` (`:217`) is what the three per-step review widgets use.
- **`unified_widget.UnifiedReviewWidget`** — reachable only from the napari plugin menu as "Review Tool (Tabbed)" (`napari.yaml:28-30`) and from its own `main` (`unified_widget.py:1123`). Its docstring claims the command `mousereach-review-tool` starts it (`:17`); that command starts the causal review tool. The launcher's "Review Tool" tab loads `GroundTruthWidget` instead (`launcher.py:282`).

### Not on this list, though a previous draft said so

**`exclude_from_analysis` on reaches has real readers.** The triage clearing tool marks every non-causal reach in a cleared segment with it (`triage_clearing.py:554-555`) and the ground truth tool sets it too (`ground_truth_widget.py:1784`). What is true is narrower: the **kinematics feature extractor** never checks it, so those reaches still contribute to per-segment kinematics, the flag is not carried into `{video}_features.json`, and it is not a column in `connectome.db`. It **is** read by the analysis dataframe, which drops flagged reaches by default (`analysis/data.py:120-127`, `:167-168`, used by `load_all_data`, `:432-497`) — though on that module's default path the rows come from the features file, where the field is absent, so the filter has nothing to act on; it bites only when loading from `_reaches.json` (`:343`). It is also read by the truth resolver when merging ground-truth reaches (`truth_resolver.py:255`) and throughout the improvement metrics (`improvement/reach_detection/metrics.py:482`, `:518`, `:926`; `kinematic_damage.py:105`, `:173`; `_compute_baseline.py:58`, `:109`). The ODC-SCI export carries it as a column (`kinematics/analysis/odc_sci_exporter.py:157`).

---

## Which answer wins, where a review is applied at all

`resolve_truth_layers` (`truth_resolver.py:289`) builds layers low to high and merges them (`_apply_outcome_layers`, `:153-183`):

```
ground truth  >  processing-dir review  >  re-segmentation-queue review  >  triage-queue review  >  algorithm
```

Two things to be precise about:

- **The unit is a segment, not an element.** Within the per-segment outcome layers, a higher layer replaces the whole override for that segment — outcome, causal reach id and interaction frame together (`:160-162`, `:172-180`). Only a segment that no layer covers keeps the algorithm's values. The separate per-reach layer (ground truth only) is merged reach by reach (`_apply_gt_reaches`, `:249`).
- **Both review tiers are stamped the same.** Triage-queue and re-segmentation-queue reviews are both labelled `"human_review"` (`:320-322`); the data does not record which tool produced the answer.

Ground truth only overrides segments it marks determined (`:138-139`). Provenance is stamped per segment as `outcome_source` (`algo` / `human_review` / `ground_truth`) and per reach as `reach_source` (`algo` / `ground_truth` / `human_review`), with the algorithm's originals preserved as `algo_outcome` and `algo_causal_reach_id` (`:172-176`). Reviews never change the boundaries of a detected reach. A review adds a reach only when the reviewer typed one the detector missed (`_add_reviewed_reaches`, since 2026-10-26; see "A reach the list does not have" above). Ground truth changes boundaries and existence, and only where `reaches.exhaustive` is set does it also delete algorithm reaches it never labelled (`:275-279`). Only real answers are applied (`is_answered`): until 2026-10-26 this step applied a pellet left on "-- select outcome --" as a human answer of "triaged".

The processing-dir layer is the one that does not work — see [Reviews mostly do not reach the numbers](#reviews-mostly-do-not-reach-the-numbers).

### Where a saved review is looked for

Two different lookups, with different rules:

- `resolve_review_path` (`causal_review_io.py:55`) returns the **file**, most recently modified wins (`:90`), checking an explicit directory the caller passes (usually the processing directory), then the triage queue's bundle for that video, then the directory holding the canonical video. It never raises; nothing found means the caller does nothing. It does not look in the re-segmentation queue. This is what the gate and every kinematics caller use.
- `truth_resolver` uses its own set: the triage queue, the re-segmentation queue, and the `primary_dir` it is handed (`truth_resolver.py:315-317`). It needs directories.

---

## Documentation that is wrong at this commit

Stated here so nobody rediscovers it the hard way:

- `review/AGENTS.md:86-89` calls `TriageClearingWidget` the routine tool and maps it to `mousereach-review-tool`. That command is the causal review widget; the clearing tool is `mousereach-triage-clearing`.
- `queue_index.py:7` says "The tool MOVES a fully-reviewed video OUT of the folder". The tool removes the database row only; the watcher moves the folder.
- `triage_clearing.py:14` and `unified_widget.py:17` both claim `mousereach-review-tool`.
- `causal_review_widget.py:2509-2510` describes `_review_root()` as `Processing/Review/triage`. It returns the parent of the queue folder, `Processing/Review`, which is where `flagged_sessions.json` actually lives.
- `pyproject.toml:120-123` says the ground-truth triage resolver runs as a pipeline step. It does not.

---

## Configuration summary

| Setting | Where | Effect |
|---|---|---|
| `nas_root` | `~/.mousereach/config.json` | Root of both queues. Unset means the queues are `None` and the tools say "not configured" (no guessed folder since 2026-10-22). |
| `MOUSEREACH_ROUTINE_ROOT` | environment | Overrides the corpus root the clearing tool's worklist scans (`triage_queue.py:490`). |
| `MOUSEREACH_TRIAGE_ALGO_DIR` | environment | Overrides the flat algorithm directory fallback (`triage_queue.py:465`). |
| `CONNECTOME_ROOT` | environment | Base for the two lookups above; no default -- unset means both lookups are skipped (since 2026-08-28). |
| `--pending-dir` | `mousereach-review-tool` | Which queue folder to review. Defaults to the triage queue (`causal_review_widget.py:3123`). |
| `--all-segments` | `mousereach-review-tool` | Walk every segment instead of only triaged ones (`:3126`). |
| `--worklist FILE` | `mousereach-review-tool` | CSV or JSON of `vid` + `segment_num`. Only those videos are offered and only those segments walked — and unlike the normal pool, already-reviewed and ground-truthed videos are included (`:2709-2718`). |
| `--cv` | `mousereach-review-tool` | Use pixel-based pellet localisation to narrow candidate reaches. Off by default because it decodes the video over the network; without it the narrowing uses tracking data only (`_get_cv_states`, `:1713`, gated at `:1727`). |
| `--queue-dir` | `mousereach-fix-segmentation` | Queue of bundles to work through; defaults to the re-segmentation queue (`fix_segmentation_widget.py:482-490`). |
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
