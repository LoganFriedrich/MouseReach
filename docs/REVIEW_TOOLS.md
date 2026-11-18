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
| **Review Tool** (retired) | `mousereach-review-legacy` | Nothing. It prints which tool to use for each kind of correction and stops. The "3 - Review Tool" tab is gone from `mousereach`. | Nothing |
| **Triage Clearing** (retired) | `mousereach-triage-clearing` | Nothing. It prints "retired -- use mousereach-review-tool" and stops. | Nothing |

Command names come from `pyproject.toml:106`, `:113`, `:115-118`.

Three more files in `review/` are worth naming, and none of them is a review tool:

- `staging.py` — builds a review bundle by running the four algorithms on a video's canonical files (`stage_video`, `staging.py:213`). Run as `python -m mousereach.review.staging --stems <video> ...` (or `--cnt` / `--all`) for bulk staging; no window calls it (the review windows only use its path lookup, `resolve_canonical_paths`). A person's cuts (a segments file marked `boundary_source: "human"`, in the bundle or, for a new bundle, beside the filed video) are kept as they are and the segmenter does not run (before, only the old "manual" markers were recognised and a restage re-cut over the fixer's cuts). It refuses, creating nothing, a video held in deep review when staging into any other queue (deep review supersedes triage) and a video whose bundle the watcher is building in that queue's `.incoming` folder. Into a review queue (the configured triage or deep-review folder, its default) it no longer writes the bundle itself: it builds the four files in a temporary folder and hands them to the watcher's router (`review_gate.route_to_queue`), which records the hold in the watcher database ('triage' / 'deep_review'), refuses a crystallized video and never merges into a live bundle. Before anything is written it refuses: a live bundle of the video in that queue (with or without overwrite -- a reviewer may be working in it), a video the watcher database on this machine does not know, is working on or holds, a crystallized one, and a machine with no readable watcher database. Run it on the processing machine. Any other `--pending-dir` (a scratch folder) is written directly, as before, and the watcher record is left alone.
- `queue_launcher_widget.py` — the "Review Queues" tab in the `mousereach` launcher (`launcher.py:367-372`). It counts what is waiting in each queue (`:88`, `:105-115`) and opens the right tool in its own napari window (`:118`, `:123`, `:128`). Its "Folders left for a person" panel lists every leftover queue folder kept for a person and every route refused because of a queue folder, and opens the Resolve window over them (`_left_items`, `_open_left_for_a_person`; see "Leftover queue folders").
- `comparison_panel.py` — a shared side-by-side panel used by the outcome step's own review widget (`outcomes/review_widget.py:481-487`), not by anything in this folder.

**Dead or near-dead code in this folder** (see [Declared but does nothing](#declared-but-does-nothing)): `base.py`, the `SavePanel` class in `save_panel.py`, and `unified_widget.py`.

---

## Triage Review (the one used routinely)

`CausalReviewWidget` in `causal_review_widget.py`, launched with `triage_only=True` by `main()` at `causal_review_widget.py:3106`.

### How a video is chosen

`load_pending_queue` picks the next video from the pool of bundles that still need review **in the lab's work order** (the order file and panel described in WORK_PRIORITY.md): it takes the highest-priority group that has a video waiting -- for example the first project, or a study's session list within a project -- and picks **at random inside that group** (`_pick_by_priority`, which sorts by `WorkPriority.rank`). Work the order marks "not taken on until nothing else is waiting" (Easy and Flat trays by default) comes after everything else, whatever its group. At launch the pool is just the list of bundle folders in the queue folder, by name -- under a second, instead of reading every bundle first (changed 2026-10-23). Each pick then checks only the video it picked against the rules below (`_pick_eligible`); one that fails is dropped from the pool and the pick is made again. Each "next video" draws from the pool minus what was already reviewed, and the folder is listed again when it runs out (`_load_next_video`). The random draw inside a group is deliberate: it keeps the reviewed set spread across cohorts and dates within that group. The status line says how the video was chosen ("top-priority pick", "priority pick (group N)", or "random pick" when no order is set). Until 2026-10 the pick was random over the whole pool.

A picked video is shown when all of these hold (`_bundle_eligible`; `_bundle_needs_review`):

- it is a directory containing a manifest file,
- it is not waiting for its segmentation to be fixed: the segmentation did not fail (confidence at or below 0, or an anomaly mentioning "reference quality"), its segments file is not missing or unreadable — **or** a person has already cleared it: the `_deep_review_cleared.json` marker, or ground truth that certifies the whole video (both exhaustive flags set and every pellet's outcome determined). This is `triage_status.held_for_segmentation`, the **same** rule the watcher's return scan uses to send a triage bundle to deep review, so the window never offers a video the scan is about to move, and a cleared video with open pellets is always offered (`_held_for_segmentation`). A skipped video is named in the window's log.
- it has at least one triaged element on a segment that exists (`_bundle_has_triage`, which calls `triage_status.bundle_triage` — the same rule the gate and the return scan use),
- no ground-truth file exists for the video anywhere — ground truth counts as the answer (`_bundle_needs_review`; `has_gt` in `causal_review_io.py`). This applies to Triage Review only: in Re-segmentation Review a video is shown whether or not it has ground truth, because ground truth never takes a video out of that queue -- only Clear (or the fixer's save) does, so hiding it there would leave it with no button that could release it (since 2026-11-17),
- its triaged elements are not already all resolved (`_bundle_reviewed`, `:2539`).

The failed-segmentation skip (`_bundle_eligible`) runs in **both** modes, including re-segmentation. That is a problem for the Re-segmentation Review tool: see [Re-segmentation Review](#re-segmentation).

### What counts as a triaged element

A segment is triaged when either:

- the outcome detector wrote `outcome == "triaged"` or `flagged_for_review: true`, or
- the outcome is a *touched* one — `retrieved`, `displaced_sa`, `displaced_outside` — and the reach-assignment file has no reach marked `is_causal` for that segment. The pellet moved, but nothing says which reach moved it.

That rule exists once, in `triage_status.triaged_segments`, and the review window, the gate and the return scan all call it (through `triage_status.triaged_in_range` / `bundle_triage`). Two refinements apply everywhere alike:

- **A missing reach-assignment file** skips only the "touched but no causal reach" test. The outcome detector's own flags still count, so a flagged video with no assignment file is still offered for triage. (The window used to have its own copy that treated a missing file as "nothing to triage", hiding such a video from every reviewer while the watcher kept it in the queue.)
- **A flag on a segment that does not exist** (for example segment 21 of a 20-segment video) is not a question anyone can answer. It is ignored on every side, with a warning in the log. A triage bundle whose **only** flags are such ghosts is returned to the pipeline by the next return scan, exactly like a fully answered one, instead of sitting in the queue for ever.
- **mousereach-route-to-queue refuses to send a video to triage when there would be nothing to ask**: none of the requested segments exists, and the video has no unanswered flagged segment. Nothing is moved and the refusal is printed.

(A third, different version lived in the worklist builder of the old triage-clearing tool, `triage_queue.py`; that tool is retired and the builder removed, see "Triage Clearing (retired)".)

In triage mode the walk visits **only** triaged segments (`_visible_indices`, `:1122`). Everything else is skipped.

### What the person is shown and asked

The whole video is loaded once as a decode-on-demand layer; navigation only moves the playhead (`_load_video`, `:487`). The view opens on the relevant reach — the reviewer's own earlier answer if there is one, otherwise the algorithm's causal reach, otherwise the reach nearest the interaction frame (`_relevant_reach`, `:590`) — landing 25 frames early (`OPEN_LEAD`, `:485`), with 45 frames of padding around the reach (`WINDOW_PAD`, `:482`; `_compute_segment_window`, `:635`). **An `untouched` call opens 30 frames before the segment's end instead, whether or not the mouse reached** (`_opens_at_end`, since 2026-10-24): untouched is judged by where the pellet is when the segment ends. Before that, an untouched pellet with no reaches opened at the end but one with reaches opened on a reach, often far before the end, so the reviewer had to skip forward to see it. If the reviewer changes the call and picks a reach, the pellet opens on that reach. Tracking points are drawn as coloured dots; in re-segmentation each also carries its name, because the names jumping is what tells a reviewer it is time to call a new segment, and triage leaves them off for speed. There is no pillar circle (dropped for speed, 2026-10-23).

The question panel takes one of three shapes, decided by `_segment_triage` (`:1770`):

- **`reach_uncertain`** — the outcome is already committed (`retrieved`, `displaced_sa`, `displaced_outside`, `abnormal` or `abnormal_exception`, `:1790-1793`); only the reach is open. Asked: *"Which reach caused it?"*, with an outcome-correction dropdown underneath.
- **`outcome_uncertain`** — exactly one candidate reach survives narrowing, so the reach is effectively pinned. Asked: *"What did it do to the pellet?"*, plus an optional reach override.
- **`both_uncertain`** — neither is known. Both questions.

The reach list is narrowed by `assignment.v2.triage_reduction.reduce_triaged_segment` to reaches that could not be ruled out as misses (`_candidate_reach_ids`, `:1675`); the ruled-out ones are built too but hidden behind a "Show all reaches" checkbox (`_make_reach_picker`, `:1859`). There is an explicit "No reach — nothing acted on this pellet" option (`:1906`), and choosing `untouched` clears and disables the reach picker so the two answers cannot contradict each other (`_sync_reach_picker_to_outcome`, `:2029`).

**A reach the list does not have: "Undetected reach (enter frames manually)".** Type its first and last frame. What is saved (since 2026-10-26):

- If a detected reach covers at least half of the frames you typed, that detected reach is the answer -- it is the same reach, and the detector's frames are kept (`causal_review_io.typed_reach_match`).
- Otherwise it is a reach the detector missed. Your frames are saved as they are, and when the results are made it is added to the pellet's reach list -- exactly as a ground-truth reach is: it replaces any detected reach it overlaps, its measurements are computed over your frames, and it is the reach credited with the pellet (`reach_source: human_review`).
- Either way the record keeps the frames you typed, as `typed`.
- A typed reach longer than 300 frames is refused on the spot ("The reach you typed runs N frames ... no reach lasts that long"): no real reach comes close, so it is a mistyped frame number, and measuring it would put nonsense into the data. The results step never adds one either (`MAX_TYPED_REACH_FRAMES`).

Until that date a typed reach was swapped for any detected reach touching it by a single frame, and a reach the detector missed was never added, so its pellet's credit was lost or landed on a neighbouring reach.

Every segment also gets an **"Ignore windows"** box — frame ranges where something that is not a reach moved the pellet (`:1369`, `_make_abnormal_ranges_widget`, `:2161`) — and a free-text notes box. There is no control for saying the cuts are wrong: a wrong pellet or wrong cuts is never answered here, it is escalated (see "Escalating out of triage").

Two more buttons: **Flag Session** (`:2514`) and, in re-segmentation mode only, **Clear -> re-enter pipeline** (`:3069`). The tool cannot re-cut a video (its re-cut editor was removed 2026-10-23); cuts are fixed in re-segmentation.

**The pellets shown are always the segments file's cuts.** That editor saved its cuts inside the review file, as a `manual_segmentation` block. Some old review files still hold one. The window ignores it: it does not build pellets from it and does not treat the video as hand-cut because of it (`_build_segment_list`). Saving keeps the block unchanged as a record of what a person once did (`save_causal_review` copies it from an earlier copy of the review). Nothing else reads it. If a video's old block shows cuts different from its segments file and those cuts were right, send the video to re-segmentation (Escalate) and set them there.

Keyboard shortcuts are playback only — space, `b` for reverse, arrows, and speed keys 1-6 (`_setup_keybindings`, `:2963`). There is no key for next/previous segment; those are buttons.

**The keys work while the video has the keyboard; what you type in the panel stays in the panel** (since 2026-10-24, `review/keyboard.py`; the Re-segmentation tool works the same way). Click the video to give it the keyboard. While the keyboard is in the review panel -- a number box, the notes, a button or choice you just clicked, the outcome list -- letters, numbers, Backspace, Delete and Enter never run a key; Space and the arrow keys still drive the video when the control you are on does not use them, so stepping after picking an outcome works as before. Choosing "Undetected reach (enter frames manually)" (or "No, it is at frame:" in the Re-segmentation tool) puts the keyboard in its number box, so the frame you type next lands there.

Why: typing numbers into a text field sometimes set off key bindings. A number box keeps the digits typed into it, but the keyboard was often still on the control clicked just before (that choice, a "Use current frame" button, the outcome list), and napari hands any key a panel control leaves unused to the video's keys. Those were napari's own keys for the selected layer -- the tracking dots, added last -- not the review tool's: 1 deleted selected dots, 2-5 switched the dots to add, select, pan or transform mode (in add mode a click on the video adds a dot), Delete and Backspace deleted dots, and on the video layer 2 let a drag move the picture (and from a panel button, napari's Ctrl+Backspace deleted the dots layer itself). So the speed keys 1-5 never set the speed. Now the review tool's keys are bound on its video and dots layers too and napari's editing keys are switched off there: a review window edits nothing.

### Why this video is here: the banner (since 2026-10-23)

Everything that holds a segment in triage is on screen, because the window is
all a reviewer sees. A purple banner above the video says it
for the segment on screen only, in as few words as possible:

```
Open: outcome + causal reach (possible tracking artifact)
```

- **Open** -- what is open on this segment, in the same words as its question:
  *causal reach*, *outcome*, or *outcome + causal reach*. In brackets, the reason
  only when there is one beyond the open question itself: *possible tracking
  artifact*, *reach at segment edge*, *pellet out of view*, or *hand score
  mismatch (hand: X, algo: Y)* when a check against the hand-scored sheet
  sent the video to triage. A note a person wrote when
  sending it is shown as written.
- **Nothing else** (since 2026-10-24). The tool never says that a segment was
  re-cut, that a person answered something before, or how: a reviewer answers
  what is open, and those words only made it unclear what to do: a segment is
  in triage because its correct answer is not known, and that is all the
  reviewer needs to be told. Pellets already answered are not walked at all (see "Triage
  walks only the pellets still open"); an answer given earlier in the same
  session is filled in again if you go back to it, without comment. How
  earlier answers were matched is logged at debug level only.

In **Re-segmentation Review** the banner says why the whole video was sent there
and what to do (since 2026-11-17 every recorded reason is shown):

- a reason about the **cuts** (it names segmentation or a mislabel): fix or
  confirm the cuts in the Re-segmentation tool and save; saving releases it;
- **the automatic quality check could not run** (reason starts
  `triage_failed`): nothing has checked the video, so watch enough to judge the
  tracking and the cuts; if they are fine confirm the cuts in the
  Re-segmentation tool, save, then press Clear; if the tracking is broken press
  "Flag Session (needs review)" instead;
- a **quality hold** (`qc_needs_review`): the same, and the flag stays with the
  video's data. The banner lists what the quality check found -- each critical
  finding, as "whole video: ..." or "segment N: ...", read from the
  `<video>_triage.json` file that travels with the bundle -- and the
  re-segmentation tool shows the same list after "Sent here:". Look at exactly
  those things. If the bundle has no such file (an older hold), the banner says
  so. The stored routing reason stays the bare word `qc_needs_review`: the
  watcher's re-check of quality holds and the re-segmentation tool's release
  both match that word, and a finding written into it could match by accident.
  (Which kinds of finding should hold a video at all is a question for the lab;
  until it is settled, every critical finding holds it, as before.);
- a **flagged session**: the banner shows the session and the note the
  reviewer wrote when flagging it; check the video for that, confirm or fix the
  cuts, answer anything flagged, then press Clear;
- **any other recorded reason** (for example one typed on
  `mousereach-route-to-queue`) is shown as written, with the same steps.

Only a video with no recorded reason at all says "no routing reason was
recorded". Before this, every reason except the cuts and quality holds got
that wording, so a reviewer was never told about a failed quality check or
shown a flagging reviewer's note.

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

Two more answers are refused the same way, because they contradict themselves:

- **a touched outcome (retrieved, displaced_sa, displaced_outside) with no reach
  picked.** Saying which reach touched the pellet is part of the answer. Pick
  the reach, or type its frames under *Undetected reach*. If no reach touched
  the pellet, choose *untouched*, or mark the frames where something else moved
  it under *Ignore windows* (with no reach picked, that records the pellet as
  moved by something other than a reach). WHY: such an answer used to be saved
  as complete, and the pellet reached the results with no causal reach at all;
- **an ignore window that covers the whole reach picked as causal.** An ignore
  window takes its frames out of the measurements, so the pellet's own reach
  would not be measured. Make the window cover only the frames that are not the
  reach, or leave the reach unpicked if no reach moved the pellet.

A review saved before this rule that marks a pellet touched with no reach is
still used for its outcome; its causal reach in the results is the reach
assignment's, because the person did not say which reach it was.

The first two tests decide what the pipeline drops when it applies answers
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
segmentation -> re-segmentation**. It saves the answers given so far (kept as
a record: once a person saves new cuts they are void and their pellets are
asked again -- see "A re-cut voids the answers given before it"), moves the bundle to the
re-segmentation queue with the notes text as the routing reason, and loads the
next video (`_escalate_to_deep_review`).

**The watcher's record follows the move (since 2026-11-15).** Escalate and Flag Session now
record the video as `deep_review` in the watcher's own database as they move it (the same
database the watcher writes; it is never created by the tool). Before, the row kept saying
`triage` while the video sat in re-segmentation, so status views and the queue folders
disagreed until it came back. If the tool cannot open that database, the video still moves and
the message says the record was not updated; the record catches up when the video returns.
The same note appears when the database opened but could not RECORD the move -- the reviewer's
login cannot write it, or the file it found is not the watcher's (it has no record of the video):
"Note: the watcher's record was NOT updated (<why>) ... Tell whoever runs the watcher."
Before, that case was only written to the log and the message just said "moved".

**Checking a reviewer's login (whoever installs or updates the review tools).** The window finds
the watcher database the same way the watcher does: `watcher.db_path` in that login's
`~/.mousereach/config.json` if set, otherwise (on a GPU PC only) `watcher.db` in that login's
processing folder. A login with no `~/.mousereach/config.json` at all is refused the same way
(run `mousereach-setup` in it). On a processing server the login's settings MUST name `watcher.db_path`: without
it every move says "Note: the watcher's record was NOT updated (... name no watcher database
(watcher.db_path) ...)" -- copy the value from the settings of the login that runs the watcher.
The refusal depends on the login's settings saying `"mode": "processing_server"` in the `watcher`
section: a reviewer's login on the processing server whose settings carry no `mode` is taken for a
GPU PC and silently uses `watcher.db` in its processing folder, which may be an old file the
watcher does not write. Give every reviewer's login both `mode` and `db_path`. So
in every login people review from, check once that this resolves to the database the watcher
actually writes, and that the login may write it: escalate one test video and read the message.
No "record was NOT updated" note means it is right. The login must also be able to create and
delete files in `<review root>/.return_claims/` (the bundle claims, below). Without that, Escalate
and Flag Session refuse every time with the "another program ... is moving this video's bundle
right now" message, although nothing else is moving it.

### When the window will not move a video

Escalate, Flag Session and Clear act on the **queue bundle** the window is showing. In these
cases they do nothing, and the message says why:

- **You opened a video yourself with Browse / Load Video.** Opening a video by hand lets go of
  the queue bundle that was on screen: Escalate says "No bundle loaded", Flag Session says "No
  bundle loaded: nothing was flagged or moved" (it writes no session note anywhere -- not in the
  queue and not beside the video you opened), and Clear says there is nothing to clear. The
  window still knows which queue it was walking: press **Save Review + Next Video** to save the
  review of the video you opened and go back to the next video in the queue. That review is
  saved beside the video you opened (as the Save Review button saves it for any video opened by
  hand), and only when you answered at least one segment; with nothing answered the window says
  "No segments reviewed yet", writes nothing, and goes on to the queue. The queue video
  you left counts as visited for the rest of this window's session, as it would if you had
  moved on with Save Review + Next Video. **Why:** the window used to keep the previous queue
  video's bundle, so these buttons moved THAT video's folder under the name of the video you
  had opened, and answers could be saved into the wrong bundle. And a session note written
  with no queue loaded used to land in the folder of the video you had opened, which may be a
  filed folder where nothing reads it and nothing the review tools do should write.
- **The video did not load.** When a video cannot be opened (a missing or broken file), the
  window shows the error and forgets everything it had loaded -- the previous video's answers,
  its segments and its bundle. Every button then says no video is loaded until a video loads
  cleanly, except **Save Review + Next Video**, which stays usable when the window was walking
  a queue: with nothing loaded it saves nothing and goes straight on to the next video in the
  queue. The bundle that failed is skipped for the rest of this window's session (also when its
  video file could not be found at all). This holds for the very first bundle of a launch too:
  if it cannot be opened, Save Review + Next Video is usable at once and moves on to the next
  bundle, with no need to close and relaunch the window. **Why:** the window used to set the new
  video's name before the load failed, and a save then wrote the previous video's answers under
  the new name; and a launch whose first bundle had no video never learned which queue it was
  in, so the next-video button stayed disabled.
- **Your answers did not save.** Each of the three buttons saves the review first. If that save
  fails (the share is unreachable, a file is locked), the button shows "Your answers did not
  save, so <video> was NOT moved" (Clear: "was NOT cleared"), the window stays on the video
  with your answers still on screen, and nothing is moved. Press the button again once saving
  works. Flag Session has already recorded its session note by then; only the move waits.
  **Why:** Escalate and Flag Session used to move the video and load the next one anyway, so
  the answers, which existed only on screen, were lost without a word.
- **Another program is moving the video right now.** Escalate and Flag Session take the same
  per-video claim the watcher's return scan takes before it touches a bundle (the marker in
  `<review root>/.return_claims/`, described under "bundle claims" below). If the watcher, the
  route command or a re-pose holds it at that moment, the button says "another program ... is
  moving this video's bundle right now; nothing was moved -- try again in a minute" and the
  window stays on the video. Press the button again a minute later. The same holds for every
  route of a bundle OUT of a queue (`route_to_queue`, `mousereach-route-to-queue`): it is
  refused with nothing moved, and the watcher treats that like any other refused route (it
  waits and tries again; it never counts against the video). **Why:** these moves used to take
  no claim, so the return scan could be moving the same folder at the same moment and the
  bundle ended up split between two places. **Limit:** the claim protects a move only when every
  machine and program that moves queue bundles runs this version. In an older version only the
  return scan takes the claim; its other routes out of a queue (a watcher's re-route, the route
  command, a review window's Escalate or Flag Session) move the bundle without looking for it.
  A mixed set
  of machines is no worse than before, but the protection is complete only once every machine
  has been updated and every review window restarted.

**Escalating cancels any earlier re-segmentation clear (since 2026-10-27).**
Pressing the button means a new problem with the cuts has been found, so a
video that was cleared in re-segmentation before is no longer cleared. Its old
`{stem}_deep_review_cleared.json` is renamed to
`{stem}_deep_review_cleared_superseded_<time>.json` (kept, never deleted) as
the video moves, and it waits in re-segmentation until someone clears it again.
The Escalate and Flag Session buttons do this (`supersede_deep_review_clear` in
`review_gate.py`); the watcher's own routes into re-segmentation happen only when there is
no clear.
Before this, the old clear went along with the video, the watcher read it as
"done" and released the video within seconds, and it came back to triage on
the same cuts -- the button seemed to do nothing.

**Nothing leaves the re-segmentation queue until it has been re-segmented
(since 2026-10-27).** A clear is not enough on its own: the video stays
until a person has set or confirmed every cut AFTER it arrived in the queue
-- by saving in the segmentation fixer, which marks the whole list of cuts
as theirs and is the file the pipeline cuts the video by. A ground-truth file
does NOT count, however many of its boundaries are determined (since
2026-11-17; before that it did): nothing copies ground-truth boundaries into
the segments file, so a video released that way was cut again by the
segmenter exactly as before. Set or confirm the cuts in the fixer even when
you have ground truth for the video. Cuts set
by hand before it arrived do not count: a video sent back is there because
those cuts were found wrong. When it arrived is read from its routing record
(`routed_at`); for a video whose routing record is missing or unreadable it is
the latest of the routing record's file date, its review manifest's
`staged_at` and the date its folder was created (since 2026-11-17 -- before,
such a video accepted any cuts ever saved by hand, even the ones it was sent
back to have fixed). A video that cannot be dated at all says so and stays. Pressing Clear on a video nobody has
re-segmented says so ("cannot be cleared yet -- not re-segmented: ...") and
leaves it in place; a complete answer-save there keeps the answers and says
the same. The Review Queues tab and `mousereach-review-release` list these
videos as "not re-segmented since they arrived". Easy and Flat trays are
never segmented, so the rule does not apply to them. The check is
`review/resegmented.py`; the watcher applies it in the return scan
(`review_return.py`), so a clear written any other way is held too.

**Every video that needs this is listed in the re-segmentation tool.** The
tool (Open Re-segmentation) lists every video in the queue that cannot leave
until its cuts are set or confirmed, whatever sent it there: the escalate
button, a flagged session, a quality hold, or the segmenter itself. That
includes a video whose cuts a person fixed on an earlier visit and that was
sent back -- those earlier cuts no longer count, so the video is listed again.
Once you save its cuts there, the tool no longer lists it.

**A video the segmenter could not cut is listed too (since 2026-11-17).** When
the segmenter fails with an error on a Pillar video, the watcher sends it here
with no segments file at all (its reason starts "segmentation_failed:"). The
tool lists it after the others, with the note "The segmenter could not run on
this video (...): there are no cuts yet". There is nothing to confirm: step
through the video, add every cut by hand ("Add a cut at the current frame" at
each tray advance)
and save. Saving writes the video's first segments file, every cut marked as
yours, and releases the video, which then re-runs from the step after
segmentation on your cuts. The video's length is taken from its pose file, as
the segmenter takes it (from the video's own frame count only when no pose file
is found); if neither can be read, the length is unknown and the save is
refused with a message saying so (nothing is written) -- fix the files first. Before this the tool skipped such a video, and since
every way out of the queue needs cuts saved here, it could never leave.

Until 2026-10-23 each segment also had a box, "Segmentation says this is
segment N. If that's wrong, which segment is this actually?", which wrote
`true_segment_num` + `segmentation_wrong` into the record. It was removed
because the button does the same job in one step. Records written with it
are still honoured: at return time the watcher diverts such a video to the
re-segmentation queue (`review_return.py`, `TriageStatus.seg_pending_reseg`), and
once the cuts are hand-fixed (`boundary_source: "human"`) the old correction
stops blocking it. The pipeline's review gate holds such a video the same way
on every run, re-runs included: while the current cuts still number that
stretch of frames otherwise, it goes to the re-segmentation queue with a
reason starting "segmentation: a reviewer said segment(s) N show a different
pellet". Set or confirm the cuts in the fixer and save; that releases it.

### A hand-score mismatch opens on the algorithm's answer (since 2026-10-24)

When a pellet is in triage because the hand-scored sheet disagrees with the
algorithm (banner: *hand score mismatch (hand: X, algo: Y)*) -- found by the
watcher's own check right after analysis (since 2026-10-28, before the video
reaches triage, so it arrives together with the video's other questions) or,
for an older filed video, by an external check routed in with
`mousereach-route-to-queue` --
the outcome list opens with the **algorithm's outcome selected**, not
`-- select outcome --`. The algorithm is usually the one that is right, so if it is, "Save Segment + Next" is all it takes, and that records
"the algorithm was right". Every other outcome stays in the list for when it is
not. A pellet the algorithm itself could not decide still opens on
`-- select outcome --`: there is no outcome to offer (`_make_outcome_picker`,
`is_bench_mismatch`).

### A pellet asked again: "are you sure?" (since 2026-10-26)

A pellet a person already answered can be sent back to triage on purpose, when
there is reason to think the review slipped -- for example an answer of
"retrieved" where the frames show the pellet lying in the scoring area for the
rest of the segment. It arrives like any other triaged pellet, and its banner
says why it is asked again and what the answer on file is. Answer it as usual:
keep the earlier answer if it was right, or change it. Saving that pellet ends
the question; until then the video stays in triage.

What is kept: the earlier answer is not lost. The review file holds it under
`answers.recheck.previous` (with the question and when it was asked), and a copy
of the file as it was is in `Analyzed/Archive/manual_corrections/` named
`<video>_causal_review.json.before_recheck_<time>.travelling` (the copy in the
video's folder) and `....before_recheck_<time>.durable` (the copy of record) -- one
file per copy, so neither backup replaces the other. How a pellet is sent back:
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
- There is no hand score to compare with: the video was analysed before the
  pipeline started keeping hand scores with each video (2026-10-28), the
  session has no hand scores, the lab has no scores file set up, or the sheet
  has no score for this pellet. The answer is saved as usual and one line in
  the console says
  `[!] Algorithm-and-sheet check not run for <video> segment <n>: <why>`.
- The answer is not a pellet outcome (for example `abnormal`).
- You already saw the box for this pellet and kept this same answer.

**How "disagrees" is judged.** Each comparison is made only at the detail the
other side can see:

- *The tray sheet* records a pellet as untouched, displaced or retrieved. It has
  one "displaced", so it cannot tell "displaced in the SA" from "displaced
  outside the SA": either of those answers agrees with a sheet that says
  displaced. The answer is put on the sheet's scale with the same table the
  watcher's hand-score check uses (`watcher/hand_scores.py`, `ALGO_CLASS`).
- *The algorithm* cannot see a pellet leave the SA. A pellet that leaves unseen
  looks to it exactly like one that was eaten, so it calls it "retrieved" (it
  never says "displaced outside SA"). Against the algorithm, "retrieved" and
  "displaced outside SA" therefore count as the same answer ("the pellet left
  the SA"). "Retrieved" against the algorithm's "displaced (in SA)" is still a
  disagreement: the algorithm saw the pellet land in the SA.

**Which pellet's hand score.** The score looked up is the one for the pellet the
answer is about: the segment's number, or -- in an older record where a person said
the segment is really another pellet (`true_segment_num`) -- that pellet. If any
other record of the same video says the cuts number a pellet wrongly, which score
belongs to which segment is not known, and the box does not appear (the console
says why) rather than compare the answer with another pellet's score
(`hand_scores.pellet_of_record`).

The hand score shown is the one the pipeline itself compared the pellet with:
when the watcher checks a video against the tray sheet it keeps every pellet's
hand score inside the video's own outcomes file (`<video>_pellet_outcomes.json`,
key `"hand_scores"`), and the review window reads it from there
(`hand_scores.score_from_outcomes`). So the review window needs no setting of
its own and no access to the scores file, and the score always travels with the
video into the queue. A video analysed before 2026-10-28 has no kept scores
until it is analysed again, and the box does not appear for it. The decision itself
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

**Stepping past a pellet you already answered keeps your answer as saved.** When you come back to an answered pellet (Prev, or a video that returns to triage), the window puts your saved answer back on screen: the outcome (an outcome the dropdown does not normally list, such as `abnormal_exception`, is added to the list so it shows), the reach you picked, found by its FRAMES (a re-run gives the reaches new numbers, so the number you picked may now belong to another reach), or the frames you typed, and your ignore windows. If you then move on without changing anything, the stored answer is kept exactly as it was, with your name and the time you answered. Only a change you make is saved as a new answer, under your name and the current time. WHY: the window used to read the panel back and save it as a new answer, so merely stepping past a pellet could replace your reach with another one that now had the old number, or replace an `abnormal_exception` answer with the default.

**Leaving `-- select outcome --` alone leaves the pellet OPEN (since 2026-10-26).** On an `outcome_uncertain` or `both_uncertain` segment where no outcome is picked, the record is saved with `answers.reviewed: false`, and the status line says `[OPEN] Segment N left open -- no outcome was picked`. The video stays in triage until someone picks an outcome (or presses Escalate, if the pellet or the cuts are wrong). Moving on without picking is allowed -- it is how you come back to a pellet later -- but it is never an answer.

Before that date the record was saved with `human.outcome: "triaged"` and nothing marking it open, and every check counted it as answered: the video left the queue, was filed, and the results step stamped `"triaged"` onto the pellet as `outcome_source: "human_review"`. Such pellets reached the exports as "Undetermined". One rule now says what an answer is -- `causal_review_io.is_answered`: not a `reviewed: false` placeholder, and, for a pellet, an outcome other than `triaged`, `uncertain` or none -- and the release check, the gate, the hand-score check, routing, the restore and the results step all use it. Records saved the old way therefore count as open wherever they are, and the review window shows those pellets again.

The re-segmentation window's release-on-save, the bulk release button (`mousereach-review-release`) and the census use the same rule: a record counts as answered only when `is_answered` says so, never merely because an outcome is written in it.

### What is written

`_save_review` (`:2844`) writes `{stem}_causal_review.json` into the bundle directory (`_review_dir`, `:880`) **and, since 2026-08-24, a durable copy at `{NAS}/review_records/reviews/{stem}_causal_review.json` first**. The bundle copy is the one the reviewer edits; the durable copy is the one nothing regenerates. Before this, a bundle-resident review was destroyed by any reprocess that recreated the bundle, and such a review had no other copy anywhere. It contains **one record per segment in the video**, not per reviewed segment: segments the reviewer never visited get a placeholder with `answers: {"reviewed": false}` (`:2867-2880`). It also calls `update_corpus_index` (`causal_review_io.py:690`), which appends to `{NAS}/review_records/causal_review_index.json`. If the index cannot be updated, the window shows a warning saying the answers are saved and the index was not updated; the review copies are already written by then. The video's index entries are written again only when the video is saved again -- nothing retries on its own -- so the warning asks for one more Save; a video that leaves the queue without another save keeps its answers in both review copies (which is what the pipeline reads) but stays missing from the shared index (it used to be a console print nobody saw). Both review copies are replaced in one step (temporary file, then one rename -- `_write_json`, see "How review files are written" below): a failed write never leaves a cut-off copy. If the **working copy** (the one in the bundle) cannot be written, the save stops with an error (napari shows it as an error message) and the earlier copies are complete; nothing is moved or cleared on a failed save. If only the **durable copy** cannot be written, the save still goes ahead -- refusing would lose the answers outright -- and the window shows a warning that the lasting copy could not be written and the review is saved only in the bundle folder, which reprocessing can replace. Save again; if the warning keeps coming back, the `review_records/reviews/` folder is unreachable and whoever looks after the pipeline must be told (it is also written to the log).

**Every copy is read, and a save never drops an answer.** A review exists in more than one place: the copy in the bundle, sometimes one beside the video, and the durable copy. On opening a video the window reads ALL of them, and for each segment takes the newest thing a person said about its frames: an answer, or a recheck ("are you sure?") that reopened one. A newer answer replaces an older one; a newer recheck reopens an older answer; a "not reviewed" placeholder or a pellet left with no outcome picked never replaces an answer, whatever its date. When saving, the tool reads every existing copy again before writing: for each segment this save leaves unanswered, the newest earlier answer (or recheck) for its frames is kept in its place, under the name and time of the person who gave it, and an earlier answer about frames no current segment covers is kept in `unmatched_answers` (shown nowhere, applied nowhere, never deleted). The console says when a save kept an earlier answer this way. Why: the window used to read only the first copy that had any records and never the durable one, and the save wrote what the window held over every copy. A bundle opened without a working copy started empty, the reviewer answered what was open, and the save replaced the durable copy's earlier answers with placeholders -- those answers were then gone from every copy, and the pipeline used the algorithm's call for those pellets. The bulk release check and the census read the copies the same way (`merge_review_copies`).

On reopening, `_load_saved_review` drops the placeholders and jumps to the first segment no one has answered. An answer stands whatever the algorithm has said since -- a person's answer is what happened in those frames (until 2026-10-24 a changed algorithm call re-opened it, and reviewers were asked again about pellets they had answered).

**Triage walks only the pellets still open when the video loads** (since 2026-10-24, `open_triaged_indices`). A video can come back for pellets nobody has looked at -- a re-run raised a new question, or, for an older filed video, an external hand-score check sent it back (since 2026-10-28 the watcher compares new videos with their hand scores before they first reach triage, so those questions arrive in the first visit) -- while its pellet file still flags the pellets a person answered before; "Next" used to walk those too and show the earlier answers. Now they are skipped (the console says how many), and a pellet answered in this session stays reachable with Prev.

**Since 2026-10-22 the restore matches each saved answer to the segment whose frames it covers** (`restore_review_for_segments`), the same rule the release check and the results step use. Until then it matched by segment number. After a re-cut renumbered the segments, an answer about other footage came back on today's segment N, so the tool counted N as done and never showed it. The release check, which matches by frames, still counted N as unanswered, so the video could never be finished. The reviewer's new answer for the old frames was then saved beside the old one, and the next save stamped segment N's frames onto the old answer. What the restore does now:

- An answer goes back on the segment whose frames it covers, under that segment's current number.
- An earlier answer whose frames match no current segment is **kept in the file** under `unmatched_answers`. It is never applied to anything, and it is never deleted by a later save.
- A segment with no answer covering its frames is shown for review, even if an answer once carried its number.

### A re-cut voids the answers given before it -- unless the pellet did not move

When a person saves new cuts for a video (the segmentation fixer in deep review), an answer given
on the old cuts is kept only on a pellet whose frames are EXACTLY the same on the old cuts and the
new ones (the owner's ruling of 2026-11-18: "If resegmenting doesn't alter, then it's fine to
keep"). Every other earlier answer is void:

- its pellet's frames moved, even by one frame (a cut at either end of it moved);
- or the files do not show which cuts it was given on: the answer recorded no frames (very old
  review files), the segments file does not record the cuts the re-cut replaced, or the answer's
  frames are not a pellet of those replaced cuts. Then it is treated as moved.

Void means: the answer is not used anywhere (not by the review window, the release check, the
review gate or the results), the pellet counts as unanswered again and is ASKED AGAIN in triage,
and the old answer stays in the review file as a record (under `unmatched_answers` after the next
save). A kept answer is used by every one of them, exactly as if it had been given on the new cuts.

How the tool tells (`causal_review_io.answer_survives_recut`, the one place this rule is decided;
`recut_verdict` gives the reason, which is also written in the log):

- when: the fixer writes the time of the save into the segments file (`corrected_at`); each
  answer carries the time it was given (`reviewed_at`). An answer given at or after the save
  always counts. An answer whose time cannot be read counts as given before.
- which cuts it was given on: the fixer also writes the cuts it had loaded -- the cuts the re-cut
  replaced -- into the segments file (`algo_boundaries`); each answer carries the frames the
  window showed for its pellet (`segment_span`). An earlier answer is kept only when those frames
  are a pellet of the replaced cuts AND the same frames are a pellet of the current cuts. Both are
  checked because before 2026-11-18 a save could stamp the current pellet's frames on an old answer
  while keeping its old time; frames that were a pellet of the replaced cuts were stamped on them.
- a pellet's frames are always counted the same way: from its cut to the frame before the next cut
  (`pellet_span`).

Saving the cuts in the fixer counts as a re-cut even if no cut was moved: the save throws away
every result built on the old cuts, but answers on pellets whose frames did not change stand. A
statement that the cuts number a pellet wrongly ("this segment is really pellet N", the hand-score
check) is different: any later re-cut BY A PERSON settles it, on every pellet, because a re-cut can
renumber a pellet without moving its frames.

**The segmenter's re-cut is a re-cut too.** When the segmenter cuts a video again and its new cuts
differ from the ones it replaces (a triage return whose automatic cuts were out of date, a re-run
of a filed video, `mousereach-bring-current`), it writes into the new segments file when
(`segmenter_recut_at`) and which cuts it replaced (`segmenter_recut_replaced`) --
`causal_review_io.segmenter_recut_record`, called from `segmenter_robust.save_segmentation`. Every
reader then applies the rule above with those in place of `corrected_at` / `algo_boundaries`
(`causal_review_io.cuts_set_at`, `replaced_cuts`). Where the earlier cuts come from: the segments
file the new one overwrites (the watcher's re-run copies the filed one down first); bring-current,
which segments into an empty work folder, passes the filed copy. What is recorded:

- the same cuts as before: no re-cut; the earlier file's own record, if any, is carried over, so
  running the segmenter again never voids anything;
- different cuts: the time and the replaced cuts -- earlier answers on pellets that moved are
  void and asked again, those on unchanged pellets stand;
- an earlier segments file that cannot be read: a re-cut with no record of the old cuts -- every
  earlier answer is asked again;
- no earlier segments file anywhere: nothing is recorded (a video segmented for the first time has
  no earlier answers).

The segmenter never writes `boundary_source: "human"`, `corrected_at` or `algo_boundaries`: those
mean "a person set these cuts" to the gate, the re-segmentation proof and the segmenter itself. A
segmenter re-cut does NOT settle a "this segment is really pellet N" statement; only a person's
saved cuts do.

**One rule, one pellet frame (2026-11-18).** Before 0c36d0d, an earlier answer was kept when its
pellet's frames were exactly unchanged. That test broke silently: the review gate and the
review window frame a pellet from the segments file (it ends the frame before the next cut), while
the results step framed it from the reaches file (it ends AT the next cut). So on a pellet whose
cuts did not move the gate counted the answer and passed the video, and the results step voided
the same answer and wrote the algorithm's call in its place -- with only a log line. Now:

- a pellet's frames are defined in one place, `causal_review_io.pellet_span`, and every reader --
  the window (which stamps them on each answer it saves), the gate, the release check, the
  hand-score check and the results step (`current_pellets`, from the segments file) -- uses it;
- whether an answer survives a re-cut is decided in one place, `answer_survives_recut`. From
  0c36d0d until the owner's ruling later on 2026-11-18 it voided every earlier answer; since that
  ruling it keeps an earlier answer on a pellet whose frames did not change (above). Because every
  reader frames pellets the same way, they now keep and void the same answers;
- the results step checks again on its own: if it finds a pellet whose answer a re-cut voided and
  that nobody has answered since, it refuses to write the kinematics
  (`feature_extractor.RecutVoidedAnswers`). In the watcher that refusal sends the video to
  **triage** with those pellets flagged, exactly as the gate does
  (`review_gate.route_recut_voided_to_triage`); every other way of running kinematics (the step
  command, bring-current, the backfill) writes no features file for the video and says why.
  A voided answer is never dropped silently.
- answering a voided pellet again counts at once, even with the same answer as before: the save
  stamps it with the time you gave it, never with the voided answer's older time (which would
  leave it void and ask it again on every pass). An earlier answer the re-cut KEEPS (its pellet
  did not move) keeps the name and time of the person who gave it when a later save carries it
  along (`save_causal_review(cuts_replaced=...)`); it is not credited to whoever saved next.

WHY: an earlier answer on a pellet whose frames moved was given about other footage; fitting it
onto the new cuts, by overlapping frames or by pellet number, is a guess about footage nobody
judged on the new cuts. A pellet whose frames did not change is the same footage, so the person's
answer about it still holds. Before these rules, earlier answers were matched to the new cuts by
frame overlap and applied.

Where a voided pellet is answered again depends on why the video was in deep review:

- **Sent for another reason** (a flagged session, a quality hold): saving the cuts does not
  release it. Open it in Re-segmentation Review: the voided pellets are open there like any
  unanswered pellet, and the video is released on the save that answers the last open pellet
  (or with "Clear -> re-enter pipeline").
- **Sent for its cuts**: saving the cuts releases it at once (the fixer's status line says how
  many pellets answered on the old cuts will be asked again). The video re-runs from reaches on
  the new cuts, and the review gate then sends it to **triage** with each voided pellet flagged,
  so the triage window walks those pellets like any other open pellet. Each one is answered on
  the new cuts before the video can go on to kinematics or be filed; it never takes the
  algorithm's call unasked.

The same gate check runs on every run of a video, so it also covers a **filed video run again**
(for example after a new algorithm version) whose review holds answers given before a person
re-cut it: those pellets are asked again in triage instead of being filed with the algorithm's
call. Once a pellet has an answer given after the re-cut, it is never asked again for this
reason.

How the gate tells (`review_gate.recut_voided_open`, `causal_review_io.recut_voided_segments`;
the results step calls the same function): a voided answer is about the current pellet its frames
overlap most (an answer that recorded no
frames: the pellet with its number). Only a real answer counts -- a pellet saved with nothing
picked asks nothing. Pellets ground truth determines are not asked. The pellet is flagged in the
outcomes file with the code `answered_before_recut` (`review_gate.flag_recut_voided`); the window
shows no words for it, only what is open on the pellet, and the gate's routing reason in the
watcher log names it.

The screen says nothing about this: as everywhere in triage and deep review, the window never
says that a video was re-cut or that a pellet was answered before (see "Nothing else" above). A
voided pellet is simply open again. Which answers were voided, and why, is logged at debug level
and kept in the review file.

What is NOT covered yet: when the SEGMENTER (not a person) cuts a video again -- for example a
triage return whose algorithm cuts are out of date -- earlier answers are still matched to the
new cuts by frame overlap, because the segmenter's cuts record no time.

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
reviewer          who gave THIS segment's answer (since 2026-10-28; absent
                  on a placeholder nobody answered)
reviewed_at       when that answer was given (same rule)
```

**Who answered which segment (since 2026-10-28).** The file as a whole also
has a `reviewer` and a `reviewed_at`. Those mean only "who saved this file
last, and when". Until 2026-10-28 they were the only record of who answered,
so when a second person opened a video someone else had partly answered and
saved one answer, the file then named the second person for every answer in
it, including all of the first person's. Now each answered segment carries its
own `reviewer` and `reviewed_at`:

- When you save, a segment whose answer you gave or changed in that save gets
  your name and the time.
- A segment whose answer is the same as it was in the file before your save
  keeps the name and time it already had. So opening someone else's video,
  stepping through their answers and saving does not make them yours.
- An unanswered segment (a placeholder) has no name.
- A file saved before 2026-10-28 has no per-segment names. Anything that needs
  to know who answered a segment then uses the file's own `reviewer` and
  `reviewed_at` instead, exactly as before -- old files still read, and nothing
  rewrites them. The next save of such a file keeps that file-level name on the
  answers it does not change.

"The same answer" means the same outcome, causal reach, notes and answer
details; what the algorithm said, the segment's number and its frames do not
count, because they change when the video is re-run or re-cut without anyone
answering anything. The corpus index (`causal_review_index.json`), the results
step (the `reviewed_by` field it writes onto each pellet) and the "are you sure?"
re-check (`mousereach-route-to-queue --recheck`) all credit each segment to the
person in its own record (`causal_review_io.segment_reviewer`).

New saves stamp the document `schema_version: "1.1"` (`causal_review_io.py:268`). Older files were given spans by a backfill that left the version alone, so a backfilled file still says `1.0` although it has spans; only files saved since the field was added say `1.1`.

`is_phantom` is always `False` — nothing sets it (`causal_review_widget.py:2301`, `:2875`, `:2291`).

`provenance` is filled by `collect_provenance` (`causal_review_io.py:112`), which reads version stamps out of the four algorithm JSONs and the tracking-model name out of the pose file. Every read is wrapped in a bare `except`, so a missing key means "could not read it", not "not applicable". It also captures less than it looks like it does: it only copies keys named `version`, `detector` and `segmenter_version` (`:137-142`), while the reach detector and the outcome cascade both write their version under `detector_version`. In real files this shows as `"reach_detector": {}` and `"outcome_detector": {"detector": "v6_cascade"}` with no version at all.

---

### How review files are written

Every file a review tool or the watcher writes about a person's answers -- the review copies (`{stem}_causal_review.json`, beside the bundle and in `review_records/reviews/`), the clear markers (`{stem}_deep_review_cleared.json`), ground truth (`{stem}_unified_ground_truth.json`), the corpus index (`review_records/causal_review_index.json`), `flagged_sessions.json`, and the answers a routing command reopens or flags -- is written the same way:

1. the whole new content is written to a hidden temporary file beside the real one, named `.<file name>.<random>.tmp`;
2. it is flushed to disk;
3. it is swapped in with one rename.

The same holds when the watcher returns a cleared bundle: before the bundle is emptied it **merges** the bundle's review into the durable copy in `review_records/reviews/` -- for each pellet the newest answer or "are you sure?" recheck either copy holds is kept (`causal_review_io.merge_review_into`); it never replaces one file with the other by modified time, which used to lose a recheck or answer only the durable copy held. Other parts of a record are kept from whichever copy has them: a record saying "this segment is really pellet N", saved with the pellet left open, stays even when only the durable copy held it. When the durable copy is the newer save, everything else it holds stays too -- its "reviewed by" and "reviewed at", and anything a person wrote at the top of the review such as notes -- so an older bundle copy never replaces newer text. Before the durable copy is replaced, the copy as it was is saved to `Analyzed/Archive/manual_corrections/<video>_causal_review.json.before_return_merge_<time>.durable` (beside the durable copy under that name when no Analyzed folder is configured); if that backup cannot be made, the durable copy is left exactly as it was and the log says so. The merged copy goes to a temporary file first and is swapped in with one rename. When there is no durable copy yet, the bundle's is copied. A merge that fails is logged and the return goes ahead; the earlier durable copy stays complete. Re-staging a video into a review bundle merges the other copies into the bundle's copy the same way; before the bundle's copy is replaced it is saved to `Analyzed/Archive/manual_corrections/<video>_causal_review.json.before_restore_merge_<time>.bundle`, and without that backup it is left as it was.

What this means for a person:

- **A crash cannot cut a file in half.** Anyone reading the file sees either the old complete version or the new complete version. Before, a crash mid-save left a truncated file, and a truncated review was read as "no answers" -- the next save then had nothing to carry forward from it.
- **A save can be refused for a few seconds.** Windows will not replace a file another program has open. The rename is retried for about ten seconds; if it still fails, the save reports an error and the previous file is unchanged. Nothing is half-written. Close whatever has the file open (an editor, another review window) and save again.
- **Two windows saving at once no longer lose each other's entries.** The corpus index and `flagged_sessions.json` are single files that every window reads, changes and writes back. Before, two reviewers saving at the same moment both read the old file and the second write dropped the first one's entries. Now each change is made while holding a lock file beside it (`causal_review_index.json.lock`, `flagged_sessions.json.lock`): a window waits for the other to finish (normally well under a second; at most 3 seconds in a review window, see below), reads the file again, adds its own change and writes. When a window finishes it deletes its lock file; Windows refuses to delete a file that anything has open at that instant (a virus scanner, an indexer, another window checking whether an old lock is stale), so the delete is retried for about two seconds instead of being given up at once. A waiting window only looks at the lock's age (the file's details, without opening it) until the lock is old enough to be stale, so waiting windows do not get in the way of the release. If the delete still fails after two seconds, an error naming the lock file is written to the log, and the lock then behaves like one left by a closed window (below). Taking the lock is equally forgiving: if the file server refuses for a moment to create the lock file (Windows does this while the previous lock is being deleted), the window waits as it would for a held lock instead of showing an error. If the refusal repeats while no lock file exists at all (typically: your Windows account may not create files in that folder), the window stops retrying after under half a second and shows the refusal ("Could not flag session: ... Access is denied", or the index warning on Save) -- ask whoever manages the share to give your account write access to that folder. Whatever happens, a window never waits past its 3 seconds; and if the lock file is created but its contents cannot be written (a full or disconnected share), the empty lock is deleted again before the error is shown, so it never blocks other windows.
- **What you see when a lock is in the way.** A review window waits at most **3 seconds** for the lock (`GUI_LOCK_WAIT_S`), so the window never hangs for long; the wait happens while the window is busy saving, so it may not respond for those few seconds. Then it gives up and says so, naming the lock file:
  - on **Save**: "Your answers are saved. The shared review index was not updated ..." Your answers are safe -- both review copies are written before the index, and the pipeline reads those, not the index. Press **Save** again a few seconds later: that writes this video's index entries. If you do not, they stay missing from the shared index (nothing adds them later on its own), including after the video leaves the queue.
  - on **Flag Session**: "Could not flag session: ... delete `<path>.lock` by hand while no review window is open". The session is NOT flagged and the video is NOT moved; press Flag Session again once the lock is gone.
  Background jobs that change these files wait up to a minute instead.
- **A lock left behind by a closed window.** If a review window is killed (or the computer loses the share) in the middle of a save, its lock stays. It is removed automatically once it is **ten minutes** old by the file server's clock. Until then every save shows the index warning above (answers still saved) and every Flag Session press fails. To clear it sooner: make sure no review window is open on any computer, then delete the `.lock` file the message names (it sits beside `causal_review_index.json` or `flagged_sessions.json`). Do not force-close a window that is saving -- that is how such a lock is left.
- **Two windows clearing the same stale lock** cannot remove each other's fresh lock: the one that clears it checks that what it removed is the same old lock it judged stale and puts it back otherwise. One extremely unlikely case remains (a third window arriving within the same few milliseconds); it is written to the log as an error.
- **A leftover `.tmp` file is harmless.** It is only left if the computer died between steps 1 and 3. Its name starts with a dot, so the router and the archiver -- which take every file whose name starts with the video's name -- never carry it along. When the watcher returns a cleared bundle to Processing it moves every file in the bundle, so there it first deletes every such leftover, before moving anything, and writes a warning naming each one in the watcher log (the bundle's real files are complete, and the review's content is already in the durable copy, which is written before the bundle copy). If a leftover cannot be deleted -- something has it open, most likely a review window saving into that bundle right then -- the return moves nothing, leaves the video and its bundle in the queue exactly as they were, writes an error naming the file, and tries again later (the same retry as any other refused return). If this keeps happening for one video, close any review window open on it. Anywhere else it can be deleted by hand.

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

On another machine, activate that machine's own mousereach environment
(the path after `conda activate` differs from machine to machine;
`conda env list` shows what the machine has). Note the bare `mousereach`
command only exists AFTER activation -- a plain PowerShell/cmd/VS Code
terminal without the activation says "not recognized". (Fallbacks if conda is
unavailable: a **MouseReach** desktop shortcut, if one was made, or
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

Every video in this queue that is a pillar tray must have its cuts set or
confirmed by a person before it can leave -- even one sent for a flagged
session or a quality hold, and even one whose cuts were fixed on an earlier
visit. Each such video is listed in **Open Re-segmentation**, and the line at
the top says why it was sent and why it cannot leave yet. If the cuts look
right, answer Yes at every boundary and press *Save these cuts*: confirming
them is the step the queue waits for.

**Finishing a video.** Saving the cuts releases a video that was sent for its
cuts. A video sent for another reason (a flagged session, a quality hold) is
released in one of two ways once its cuts are saved: if every pellet is
already answered, the **Release finished videos** button on the Review Queues
tab (or `mousereach-review-release --clear`) lists it as complete and releases
it; otherwise open **Re-segmentation Review**, which shows its open pellets,
answer them and press **Clear -> re-enter pipeline**. That is the release:
the watcher then re-runs the video from the step after segmentation -- your
cuts are kept as they are; reaches, pellet outcomes and reach assignment are
made again on them, then the quality check, the hand-score comparison and the
review gate -- and the results flow to the database automatically. Nothing
else needs to be done. This holds even for a video that had been filed and was
in the queue because a kinematics-only re-run sent it there: the release wipes
that re-run's "kinematics only" mark, so the video goes through every check
again after your answers (docs/PIPELINE_AS_BUILT.md, "a returned video resumes
just after the step that sent it").

**Where a video released from triage starts again.** Answering a triage
video's questions releases it into the step after the one that asked them:
after pellet outcomes for an outcome or causal-reach question (reach
assignment is made again, the outcomes are kept), after reach assignment when
the question was asked because reach assignment could not run (reach
assignment is not tried again; your causal-reach answers stand). The quality
check, the hand-score comparison and the review gate always run again after
your answers. An earlier step's results are kept only if they were made with
the algorithm versions and the pose model that are current now; if anything
changed while the video waited, the watcher starts at that step instead and
makes that step's results and every later one again.

**What happens to your answers when results are made again.** On the SAME
cuts, they are kept and used, as designed: the watcher does not ask you again;
each answer is applied to the pellet whose frames it covers, on the remade
reaches, outcomes or reach assignment. The video comes back to you only if the
review gate finds a question no answer covers.

**A re-cut by a person voids your earlier answers on pellets it moved.** Once
a person saves new cuts (the segments file's `corrected_at`), an answer given
before then counts only on a pellet whose frames are exactly the same on the
old cuts and the new ones; on any other pellet -- moved by even one frame, or
no record of the cuts the answer was given on -- it does not count. Each such
answer is kept in the review file as a record, is not applied, and its pellet
is asked again: the review gate sends the video to **triage** with each such
pellet flagged, so you answer it on the new cuts instead of the algorithm's
call being filed. This holds after a deep-review release and when a filed
video is run again. See "A re-cut voids the answers given before it" above.

**The segmenter's own re-cut follows the same rule.** When a triage
video's automatic cuts turned out to be out of date while it waited and the
watcher started it again at segmentation, the segmenter cuts it again. If its
new cuts differ from the ones it replaces, it writes into the segments file
when (`segmenter_recut_at`) and which cuts it replaced
(`segmenter_recut_replaced`), and your earlier answers are judged exactly as
after a person's re-cut: kept on a pellet whose frames did not change, asked
again on any other. If its new cuts are the same as before, nothing is voided.
These are separate names from the fixer's `corrected_at` / `algo_boundaries`,
so the segmenter's cuts are never taken for a person's. (Before this, the
segmenter's new cuts recorded no time and earlier answers were fitted onto
them by overlapping frames.)

**If you get a video that is not actually broken**, press *Skip this video* in
the re-segmentation tool (it stays in the queue), or clear it through Deep
Review if you have judged every segment and all is well.

---

## Re-segmentation Review

The same widget with `deep_review=True` (`causal_review_widget.py:198-203`). Differences:

- It reads the `DEEP_REVIEW` queue, not the triage queue (`queue_launcher_widget.py:123-126`).
- `triage_only` is forced off (`:210`), so **every** segment is walked.
- The **Clear -> re-enter pipeline** button appears. It saves the review, then writes `{stem}_deep_review_cleared.json` into the bundle (`_clear_deep_review`, `:3069`, marker at `:3085`). That marker is what the watcher looks for.
- **The release button checks the queue again when you confirm (since 2026-11-17).** The list on the Review Queues tab is made when the tab refreshes, and you may confirm minutes later. On OK the queue is read again and only videos that are STILL finished are released; a video that has left the queue since, was sent back, or was already released by someone else is skipped, and the message after the release says how many were skipped and why. The button never creates a folder: before this, a video that had left the queue came back as a folder holding only a clear marker. A clear marker that is already there is never overwritten. `mousereach-review-release --clear` works the same way. The marker is written whole or not at all (the record is written to a temporary file in the bundle folder and then linked to the marker name), so a crash part-way never leaves an empty marker that has lost who cleared the video and when.
- **No other button recreates a video that has left the queue either.** While you have a video open, the watcher (or another window's release) can move it out of the queue. Then:
  - **Clear** says the video has already left the re-segmentation queue, writes nothing, and stays where it is -- press Next to go on. If someone else already released it, it says so and moves to the next video; their marker is kept as it is.
  - **Save** (and every button that saves first) keeps your answers in the lasting copy of the review (`review_records/reviews/`) and says the video has left the queue; nothing is written into the queue. A save that leaves every pellet answered releases the video only if it is still in the queue.
  - **The segmentation fixer** refuses to save cuts for a video that is no longer in the queue ("This video has left the queue"; nothing is saved), and releases after a save only if the video is still there.
  WHY: each of these used to write through a helper that creates missing folders, so the video came back as a folder holding only a clear marker -- a "cleared" video with no files, held for ever -- or only a review copy. Every clear marker is now written by one create-only writer (`causal_review_io.write_clear_marker_once`), and the review save is told not to create a queue folder (`save_causal_review(create_folder=False)`, `ReviewFolderGone`).
- The same marker can be written in bulk from a terminal: `mousereach-review-release` lists every re-segmentation bundle's completeness and with `--clear` releases the fully-answered ones. "Answered" is the one definition, `is_answered`, applied to the answers whose frames match each segment of the video's CURRENT cuts: a pellet left with no outcome picked, or sent back with "are you sure?", is open even though an outcome is written in its record. Until this was changed the release check, the window's release-on-save and the census counted any record with an outcome in it as answered, so a video could be released, or reported complete, with a pellet nobody had decided. It is the button's bulk form for reviews completed without the second step, not an automatic release.
- A bundle whose every segment is answered is **complete whoever cut it**. A bundle whose cuts a person re-set after it arrived and whose pellets were all answered used to be "held back" when it had been sent for a reason other than segmentation; the window would not open it (nothing to answer) and the fixer would not list it (already re-cut), so nothing could release it. Held back now means: cuts re-set, sent for another reason, and pellets still open -- which the window shows.

It has no console command. It opens from the "Review Queues" launcher tab (`queue_launcher_widget.py:123`) or from the pipeline dashboard for one named video (`dashboard/widget.py`, `_open_review_for_video`; the dashboard's database buttons open only the watcher's existing database and never create one, see PIPELINE_AS_BUILT; its Publication Lock buttons work only on the processing server, or on a GPU PC with also_process on, as `mousereach-crystallize` does).

**It cannot open the videos it exists for.** The window does not offer a bundle that is waiting for its segmentation to be fixed, in this queue or in triage. "Waiting for its segmentation" is one shared rule, `triage_status.held_for_segmentation`, which the window and the watcher's return scan both call: the segmentation failed, OR its segments file is missing or cannot be read (there are no cuts to trust), AND no person has cleared the video for deep review. "Segmentation failed" is the main reason the gate routes a video to re-segmentation in the first place, so most bundles in this queue are skipped here. A bundle with no segments file is treated the same way everywhere: held, not offered (the window used to have its own copy of the rule that counted such a bundle as fine). Open these with the Segmentation Fixer below; once a person has set or confirmed the cuts and cleared the video, this window offers it again.

---

## Segmentation Fixer (the re-segmentation tool)

`mousereach-fix-segmentation`, or the **Open Re-segmentation** button on the
Review Queues tab (`queue_launcher_widget._open_reseg`). Segment cuts only: no
outcomes, no reaches, no causal attribution.

It walks the re-segmentation queue (or `--queue-dir`) and offers every pillar
bundle that cannot leave until a person sets or confirms its cuts, and whose
cuts nobody has set since it arrived (`_needs_reseg` ->
`resegmented.needs_recut`). That is the same proof every way out of the queue
demands, so the list and the exits cannot disagree. Whatever sent the video
-- the segmenter's own `needs_human` list, a reviewer's mislabel, the triage
escalate button, a flagged session, a quality hold -- it is listed. Cuts set
by hand on an earlier visit do not hide it: they are older than its arrival,
so they no longer count. Once a person saves cuts here, the bundle is not
offered again. Easy and Flat trays are never listed (they are not segmented).
The line under the video name says why the video was sent and why it cannot
leave yet.

Why: the list used to be "needs_human, a mislabel, or a routing reason naming
segmentation -- unless the cuts were ever set by hand". A video sent back
after an earlier hand fix, and a video sent for a flagged session, were then
in the queue with no tool that would show them, while every exit still
required new cuts. Videos with the most unused
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

- **The save archives only what it replaces.** The `_archived/segmentation_before_human_fix_<time>` folder is made only when a segments file or a computed output is copied into it; a first save for a video with no segments file and nothing computed leaves no empty folder.
- **It releases the video only when the routing reason names segmentation.** On save it writes `{stem}_deep_review_cleared.json` when the bundle's recorded routing reason mentions segmentation/mislabel -- fixing cuts is then exactly the work the video was routed for. A bundle routed on other grounds (a QC hold, an unrecorded reason) is deliberately NOT released by a cut-fix, because the clear marker is a blanket human-clear token the watcher gate honors; the status bar says so and the Re-segmentation Review tool's Clear (or `mousereach-review-release`) remains the path out once the real concern is resolved.
- **It does not re-run anything.** The bundle's reaches, outcomes and assignments still describe the old cuts until the video goes back through the pipeline. The re-run keeps the human cuts: segmentation preserves a `boundary_source: "human"` segments file instead of overwriting it (`segmentation/core/batch.py::process_single`).

Nothing routes videos here on the strength of `needs_human` alone. That routing was added and then deliberately switched off — the gate records the verdict and ignores it, with the reasoning written out in full at `review_gate.py:89-105`. What DOES route videos here (since 2026-08-25): a reviewer setting a corrected segment number in triage (the return scan diverts the video instead of re-injecting it, `review_return.py`), the triage tool's "Escalate: bad segmentation" button, and the ReprocessingScanner's divert when a pending review declares a mislabel (`reprocessor.py`).

---

## Ground Truth Tool

`GroundTruthWidget(review_mode=False)` — `mousereach-gt` and `mousereach-unified-review`, both to `ground_truth_widget.py:3018`.

Question: what really happens in this video. There is no accept/verify step — **setting a value is the answer** (`ground_truth_widget.py:92-104`). It seeds itself from the algorithm's output, and the person corrects boundary frames, reach start/end frames, outcome classes, interaction frames and outcome-known frames, with per-item comments and per-component "exhaustive" flags meaning "this component is completely labelled for this video".

Two save buttons (`:585-625`):

- **Save Progress** (`_save_progress`, `:1971`) writes everything, including untouched algorithm-seeded rows.
- **Save as Ground Truth** (`_save_ground_truth`, `:1980`) writes a filtered copy containing only items a person actually determined: boundaries and outcomes marked determined, reaches with a determined start or end. It keeps your "complete" (exhaustive) marks for boundaries, reaches and outcomes, with who set them and when. (Before this was fixed, this button wrote the file without the marks, so pressing it erased the "complete" marks a Save Progress had written.)
  - **When it refuses.** If you marked a part complete and that part still holds items you have not determined, the save is refused: nothing is written, and a red message names the part and how many of its items are not determined (for example "reaches is marked complete, but 3 of its 40 reaches are not determined and would be left out"). WHY: the finished file keeps only determined items, so "complete" would then claim the shorter list is every item, and the undetermined ones -- detected items you have not got to -- would be read as ruled out. What to do: determine those items (or remove the "complete" mark from that part), then press Save as Ground Truth again. Save Progress always saves, everything included.

Both write `{stem}_unified_ground_truth.json` next to the video (`unified_gt.save_unified_gt`, `unified_gt.py:252`). Ground truth outranks every review downstream, and having one anywhere removes the video from the Triage Review queue (`_bundle_needs_review`). It does not take a video out of the re-segmentation queue: that needs the cuts set in the segmentation fixer and a clear (see "Nothing leaves the re-segmentation queue until it has been re-segmented").

**What of your ground truth reaches the results.** Only what you determined. The tool starts from the algorithm's reaches, boundaries and outcomes, and a progress save writes those seeds into the file too; a seed you never set or confirmed is the algorithm's guess, not yours, and the results never use it as ground truth:

- **Reaches.** A reach counts only when both its start and its end are determined (files from the tool's older format say "verified"; both spellings count). A determined reach replaces the detected reach it overlaps, and its measurements are computed over your frames. A reach whose start or end is still undetermined changes nothing: the detected reaches stay as they are.
- **"Reaches complete" (exhaustive).** When you mark the reach list complete, a pellet's detected reaches are replaced by your list for that pellet -- so a detected reach you did not list is dropped as a false detection. That happens only for a pellet whose listed reaches are ALL determined. If a pellet still holds a reach you have not determined, you have not finished that pellet: only its determined reaches are applied, the detected reaches stay, and the watcher log says which pellet and how many reaches were left open. To finish it, determine (or exclude) every reach in that pellet.
- **A reach you mark "exclude from analysis"** is your statement whatever its frame flags: every detected reach whose frames overlap it (one shared frame is enough) is left out of the kinematics, whether or not the reach list is complete -- the same rule as an ignore window in the review window. The video's features file lists each such reach under its pellet's `reaches_left_out`, with your frames and your reason (`excluded_by_person`), so the file still says the reach happened; it has no kinematics row. Before this, an exclusion removed the detected reach only when the reach list was marked complete; otherwise the reach was measured like any other.
- **Pellet outcomes, and which reach gets the credit.** A determined outcome replaces the algorithm's outcome and any review's outcome for that pellet. For a retrieved or displaced pellet, **the interaction frame you set decides the causal reach**: the reach (yours if you determined one there, else the detected one) whose frames contain your interaction frame is the reach credited with the pellet. The tool has no separate "causal reach" control, and a reach id stored in the file is not used -- it was copied from the algorithm's reach list of the run the file was made on, and reach ids are renumbered on every run. If you set no interaction frame (or no reach contains it), the reach a person picked in the review window is used when that review gave the same outcome; otherwise the reach assignment's pick is used, as for any pellet no person decided. The results say which applied (`causal_reach_source`: `ground_truth`, `human_review` or `reach_assignment`). For an untouched pellet no reach is credited.
- **What a review keeps when your ground truth decides the pellet.** The review's ignore windows still take their frames out of the measurements, and a reach the reviewer typed because the detector missed it still joins the reach list -- unless your reach list is marked complete, in which case only your reaches count (the watcher log says the typed reach was not added). When the review and your ground truth disagree on the outcome, yours is applied and the watcher log names the pellet, both outcomes and the reviewer. (Before this, a ground-truth outcome threw the whole review answer away -- causal reach, typed reach and ignore windows -- and the pellet was filed with no causal reach.)

**Opening a ground truth again after the video was re-run.** The tool starts from the current algorithm outputs and lays your saved ground truth over them:

- **Reaches are matched by their frames**, not by their number: reach numbers change on every run, so your saved reach goes onto the current reach that shares the most frames with it, and that reach keeps its current number. A saved reach that shares no frame with any current reach is shown as its own row (with a new number if its old one is taken) when you had touched it (set a frame, excluded it or commented); an untouched seed from an older run is not brought back. Before this, your reach took over whichever current reach happened to have its number, and that reach disappeared from the list.
- **Your saved cuts come back on the cut with their frame, not on the cut with their number.** When the file is opened again, each saved cut goes on the current cut within 30 frames of it (the segmenter's own merge window: closer than that, two cuts are one tray advance). A cut you moved further than that goes back on its row only when nothing was added or removed around it: between the nearest cuts that did match by frame, the number of cuts must be the same as when you saved, and putting it there must keep the cuts in time order. Otherwise it is put on no row: a yellow message when the video opens says which saved cut(s) match no current cut, and they stay in the file under `segmentation.unmatched` (kept by both save buttons, never read as the file's segmentation). Set those cuts again if needed. Before this, saved cut N was put on today's cut N by position, so when today's segmentation had one cut more or fewer, your determined cut replaced another cut on its row while the real cut stayed beside it -- and the next save wrote that as your determined segmentation (`unified_gt.match_saved_boundaries`; the old separate `_seg_ground_truth.json` files are matched the same way).
- **Outcomes come back on their pellet only when the cuts are the same.** A saved outcome is shown on pellet N only when pellet N has exactly the frames it had when you gave the answer (your determined cuts are restored first, so a ground truth whose cuts you all determined always matches). If the video has been re-cut since, the answers given on the old cuts no longer apply -- they are about other footage -- so they are not put on any pellet. They are not lost either: a yellow message when the video opens says how many saved outcomes match no pellet, and they stay in the file under `outcomes.unmatched` (kept by both save buttons, never used for the results). If you had marked outcomes complete, Save as Ground Truth will refuse until those pellets are determined again (or the mark is removed), because the pellets they were about now have no determined outcome. Determine those pellets again on the new cuts. Before this, an outcome was put back by pellet number and, after a re-cut, landed on another pellet's footage -- and the next save wrote it there as yours.

A save never rewrites the ground-truth file in place. The new content is written to a hidden temporary file in the same folder (its name starts with a dot, `.{stem}_unified_ground_truth.json.<random>.tmp`), flushed to disk, and then swapped in with one rename. So a crash, a closed window or a lost network connection in the middle of a save leaves the previous ground truth complete -- never a cut-off file that every reader would take for "no ground truth". If another program has the file open at that moment, the rename is tried again for about ten seconds; if it still cannot be done, the save shows an error and the file on disk is the previous version. Save again once the other program has let go of the file. (`fsutil.write_json_atomic`; the same writer is used for every review file -- see "How review files are written" above.)

**Ground truth saved for a video that is already filed.** Once the ground-truth file sits in the video's filed folder (beside its features file), the watcher's version scan notices that it was saved or changed after the video's kinematics were made and re-runs that video's kinematics -- no algorithm and no quality or hand-score check runs again. The review gate does run, as on every kinematics-only re-run of a filed video: if the video still has a segment that neither a person's review nor your ground truth answers, the gate takes the video out of the filed state and sends it to triage for that segment (whether a filed video should ever be taken back like this is an open decision; docs/PIPELINE_AS_BUILT.md, "Still open, waiting on a decision", item 1). The re-run applies your ground truth: its reach frames replace the detector's, its pellet outcomes replace the algorithm's, and the kinematics are computed over your frames. Nothing to press: it happens on the next scan. How the scan tells: the processing manifest records which ground-truth file the kinematics were actually made with (`applied_ground_truth`, its content fingerprint, or "none"); a filed file whose content differs is owed a re-run, until a re-run has really applied it. Ground-truth segment boundaries are not applied this way (cuts are changed in the re-segmentation tools). Ground truth kept anywhere else (not beside the filed video) is not watched this way. docs/PIPELINE_AS_BUILT.md, "ground truth saved after filing", has the details.

The video dropdown is filled from the pipeline index (`_populate_video_dropdown`, `:652`) and capped at 30 entries (`:675`); if the index is unavailable the dropdown falls back to "Browse for video..." and the error is shown in the status line (`:699-705`).

`--algo-dir` on the command line changes where the tool reads a segment's decision window and which information panel it shows (`:3028-3033`). It does **not** change where a save writes: `_algo_files_dir()` returns the video's own parent directory unconditionally (`:2030-2039`).

---

## Review Tool (retired) and the per-step review screens

**The old whole-video Review Tool no longer saves.** `GroundTruthWidget` in review mode -- `mousereach-review-legacy`, and the "3 - Review Tool" tab that `mousereach --reviews` and `mousereach --step 2b/3b/4b` used to open -- saved a person's corrections straight into `_segments.json`, `_reaches.json` and `_pellet_outcomes.json`. Now:

- `mousereach-review-legacy` prints which tool to use for each kind of correction and stops (exit code 1) without opening a window;
- the launcher has no "3 - Review Tool" tab; `mousereach --reviews` and `--step 2b/3b/4b` still start, print a note that the old review tabs are retired, and open the tabs that are always there (Review Queues among them); a video named on that command line is loaded into those tabs;
- the tool's "Save & Continue" writes nothing and shows the same message. Its background database-refresh check after a save goes with it.

**Why it was retired and not repaired.** A correction written into the algorithm's own file is not kept by the pipeline: the next run of that step writes the file again from the algorithm and the correction is gone (corrected cuts were stamped `validation_status: "validated"`, but the segmenter keeps only cuts marked `boundary_source: "human"`); nothing after the corrected step was redone, so the reaches and outcomes still described the old cuts; the watcher's release check does not read these marks, so a video corrected this way did not leave its review queue; and its reach corrections went to a new top-level `reaches` key that the kinematics step does not read.

Which tool keeps which kind of correction instead:

- **Segment cuts** -> the Segmentation Fixer (`mousereach-fix-segmentation`, or Review Queues -> Open Re-segmentation). It marks the cuts as a person's, so a later segmentation run keeps them, and clears what was computed on the old cuts.
- **A pellet's outcome or causal reach** -> the review window (`mousereach-review-tool`, or Review Queues -> Open Triage Review). The answer is a review record that survives re-runs and releases the video.
- **Reach start and end frames** -> the Ground Truth tool (`mousereach-gt`). The results step layers its file over the algorithm's reaches.

**The per-step review screens are NOT retired and still save.** Boundary review ("Step 2b - Review Boundaries", `mousereach-segment-review`), outcome review ("Step 3b - Review Outcomes", `mousereach-review-pellet-outcomes`) and reach review ("Step 4b - Review Reaches", `mousereach-review-reaches`, including its "mark as causal reach" button), each also in the napari plugin menu and as the tabs of "Review Tool (Tabbed)", save into that step's own file and mark it `validation_status: "validated"`. They also still send an entry to the algorithm-versus-human comparison log (`PerformanceLogger`) and to the pipeline index (`PipelineIndex.record_validation_changed`); the retired Review Tool never did either. **Why they stay:** for a person running the steps by hand, without the watcher, a per-step screen's save is the only way to sign off a step that triage marked `needs_review` -- reach detection skips a video whose segmentation still reads `needs_review` (unless run with `--skip-validation-check`), and `mousereach-archive` files only a video whose three steps read `validated` or `auto_approved`. **Their limit is the same as the Review Tool's:** a later run of that step writes the file again from the algorithm, so a correction saved there lasts only until that step is run again, and the watcher's review queues never read it. With the watcher, review from the Review Queues tab.

Files that the retired Review Tool already edited keep their marks (`validation_status: "validated"`, `human_corrected`, `human_verified`); the clean-up rules still treat such a file as a person's file. The code that refuses is `review/retired_saves.py`.

---

## Triage Clearing (retired)

`mousereach-triage-clearing` was an older per-segment clearing walk. It is retired: running it now prints a message pointing to `mousereach-review-tool` (or the **Review Queues** tab of `mousereach`) and exits with an error code. It opens no window and writes no file. The command name is still installed so that an old shortcut or script gets that message instead of crashing.

**Why it was retired.** It saved the reviewer's call straight into the algorithm's own files, which broke the pipeline in four ways:

- The call went into `_pellet_outcomes.json` and `_reaches.json`, not into a review record. The next run of outcome or reach detection rewrites those files, so the person's answer was lost.
- It never wrote `_reach_assignments.json`. A touched pellet it "cleared" still had no causal reach as far as the watcher's release check was concerned, so the video never left the triage queue.
- It wrote the reaches file and then the outcomes file, with no way back if the second write failed, leaving the two disagreeing.
- It took the most recently marked reach as the causal one instead of asking which reach moved the pellet, and saved every reach the algorithm had seeded into the ground-truth file, checked or not.

The review window (Triage Review, above) does the same job correctly: the answer is a review record (`{stem}_causal_review.json`) that the watcher reads, that survives re-runs, and that releases the video once every question is answered.

Its spot-check mode (`--qc-count`, `--qc-report`) went with it: it walked the same retired worklist, and its "algorithm is wrong" verdict wrote a flag back into the algorithm's outcomes file, which the next run overwrote. Files it would have written (`_QC/qc_state.json`, `_QC/qc_drift_log.jsonl`) are not read by anything. Its worklist builder (`triage_queue.py`), with the `MOUSEREACH_ROUTINE_ROOT` and `MOUSEREACH_TRIAGE_ALGO_DIR` settings, is gone as well.

Outcome and reaches files that an old copy of the tool already edited keep their marks (`triage_cleared`, `human_verified`); the clean-up rules still treat such a file as a person's file.

---

## Re-attaching a review after reprocessing

**On the same cuts only.** Everything below is how an answer finds its pellet when the cuts have
NOT been redone since it was given (a re-run of the later steps). After a re-cut -- a person's in
the segment fixer, or the segmenter's when its new cuts replaced different ones -- an answer
given before it counts only on a pellet whose frames are exactly unchanged; every other one is
void, kept as a record and asked again in triage, never re-attached to the new cuts by frame span
(owner ruling R2; see "A re-cut voids the answers given before it").

### The problem

A review is a fact about a stretch of video. `segment_num` is not — the segmenter hands out those numbers fresh on every re-cut. A review of "segment 7" used to be applied to whatever the new segment 7 turned out to be, and the result was recorded as `outcome_source: "human_review"`, indistinguishable from a real one.

### What was added

`build_segment_record` stores `segment_span` — the exact `{start, end}` frames the reviewer was shown — on every record (`causal_review_io.py:167`, span built by `_segment_span`, `causal_review_widget.py:81-91`). Reviews written before that were backfilled in place by `scripts/backfill_causal_review_spans.py`, which refuses to write a span unless the neighbouring segmentation still matches the review: same segment count, and every record's algorithm causal reach starting inside the span that number would get (`verify`, `backfill_causal_review_spans.py:124-156`). Files failing either check are skipped and listed by name. The script does not touch the document's `schema_version`.

A review file without spans (one the backfill skipped) falls back to number matching.

`index_review_by_segment` (`causal_review_io.py:349`) maps each **current** segment number to the review record that describes it:

- A record with no span is matched by number, and only if that number exists in the current segmentation (`:404-408`). That is all that was ever available for those records.
- A record with a span is scored against every current segment by overlap. The score is `overlapping frames / the shorter of the two ranges` (`:415`).
- Spans shorter than **5 frames** (`MIN_REAL_SEGMENT_FRAMES`) are excluded on BOTH sides before scoring: a zero-length segment is not footage, and because the score normalises by the shorter range, a 1-frame segment inside any review span would score 100%, tie the true target, and the tie would drop the human's answer. A degenerate current segment is skipped as a target; a review recorded against one is dropped, each with a note.
- The best match must beat the runner-up by at least **0.15** (`MIN_MARGIN`, `:346`). If it does not, the reviewed stretch has been split across two new segments and the record is **dropped**, with a note (`:423-430`).
- The best match must cover at least **0.5** of the shorter range (`MIN_SPAN_OVERLAP`, `:342`). Below that the record is **dropped** as describing different footage (`:432-437`).
- If two records both want the same current segment, the weaker overlap is dropped (`:439-445`).
- Where a record survives but lands on a different number than it was written with, a note records the move (`:449-453`).

Notes are returned to the caller and logged at warning level (`truth_resolver.py:109-110`, `causal_review_io.py:536-537`). Nothing is written back to the review file; the matching is redone every time.

`true_segment_num` / `segmentation_wrong` (found only in records written before 2026-10-23, when the control that wrote them was removed) play no part in any of this.

### Frame matching: where a review is read

Frame matching happens where a review is turned into numbers:

- `truth_resolver.resolve_truth_layers` (`truth_resolver.py:289`), called by the feature extractor (`kinematics/core/feature_extractor.py:220-225`), passes the current segmentation from `reaches_data["segments"]` (`truth_resolver.py:312`).
- The extractor's fallback `load_and_apply_review` does the same (`feature_extractor.py:228-239`).

Every decision about whether a video is **finished with review** reads the review the same way: `triage_status.triage_status` takes EVERY copy of the video's review (the copy in the bundle or folder, the copy beside the video, the durable copy of record) and merges them with `causal_review_io.effective_review`, which matches each record to a current segment by its frames and keeps, per pellet, the newest thing a person said (an answer or an "are you sure?" recheck). `triage_status.resolved_segments` then counts a segment as answered only on that merged, frame-matched review. The pipeline gate (`review_gate.evaluate_gate`) and the return scan (`review_return`) go through `triage_status`; the review tool's own "is this bundle done" check (`causal_review_widget.bundle_review_complete`) reads the merged review with the same `effective_review` and counts with the same `resolved_segments`.

So after a re-segmentation, an answer whose frames no longer match any current segment does not count as answering any of them: it is kept in the review file as an unmatched answer (never deleted), and the pellets it no longer covers stay open until a person answers them on the new cuts.

---

## Reviews mostly do not reach the numbers

This is the most important thing in this document.

`resolve_truth_layers` looks for a review in exactly three places (`truth_resolver.py:315-317`):

0. `review_records/reviews/<stem>_causal_review.json` (the durable copy -- always written, belongs to no bundle)
1. `Processing/Review/triage/<stem>/<stem>_causal_review.json`
2. `Processing/Review/deep_review/<stem>/<stem>_causal_review.json`
3. whatever the caller passes as `primary_dir`

`_review_doc` (`truth_resolver.py:72-88`) reads a review by appending a filename to what it is given. It accepts a **directory**. Every production caller passes the result of `resolve_review_path` (`orchestrator.py:1151-1155`, which the Run All Steps tab runs through `watcher/run_one.py`; `orchestrator.py:2100-2104`, `reprocess_to_current.py:258-260`), and `resolve_review_path` returns the path to the JSON **file** (`causal_review_io.py:83-90`). Appending a filename to a file path yields nothing, so **lookup 3 never contributes**.

The `except` branch that would rescue this (`feature_extractor.py:226-239`) calls `load_and_apply_review`, which does accept a file path (`causal_review_io.py:526-527`). It only runs if `resolve_truth_layers` raises, and `resolve_truth_layers` is written not to raise.

So a review is applied only while its bundle is still sitting in one of the two queue folders. Once the return scan has moved the bundle back into Processing and deleted the bundle directory (`review_return.py:163-180`, `:208-211`), the review that released the video is invisible to the kinematics step and the numbers keep the algorithm's call.

Checked directly on a video (here called `20250101_CNT0101_P1`) whose review sits in `Processing` and has four segments a person answered (including "algorithm said triaged, person said displaced_sa"):

- `resolve_truth_layers(..., primary_dir=<the file>)` — what the pipeline does — leaves every segment `outcome_source: "algo"`, and segments 9, 16 and 19 still read `triaged`.
- `resolve_truth_layers(..., primary_dir=<the directory>)` returns `displaced_sa`, `displaced_outside`, `displaced_sa` with `outcome_source: "human_review"`.
- Its shipped `_features.json`, written days after the review, carries `outcome_source: "algo"` on every reach.

Across the features files checked in a local Processing folder, `outcome_source` was `"algo"` on every reach and `"human_review"` on none, while every review file checked in the same folder contained a real human outcome.

Reviews **have** landed historically — some archived features files carry `human_review` — and since lookup 3 cannot work and ground truth is stamped `ground_truth` rather than `human_review`, the queue-folder lookup is the only way those values could have got there. The deciding factor is whether the bundle was still in a queue folder when kinematics ran.

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

1. **The pairing is by number, and a mismatch holds the video (since 2026-11-18).** The kinematics step pairs each reach segment with the outcome segment of the SAME `segment_num` (`feature_extractor.pair_segments_by_number`); it used to zip the two lists by position, so a disagreement attached outcomes to the wrong reaches with nothing raised. The reaches file normally has one more segment than the outcomes file (the stretch after the last cut, which is not a pellet); that stretch is left out, as before. Anything else -- a pellet the outcomes file has and the reaches file does not, a pellet in the middle with no outcome, more than one reach segment after the last outcome (pellets at the end with no outcome), a number listed twice, a file with no segment list -- writes no kinematics: the video is held with the reason (`KinematicsProblem`), and the fix is to re-run it from reach detection. What this cannot see: two files with the same pellet numbers made on different cuts (the outcomes file carries no frames to compare). The review gate's stale-output checks are what catch that case.

2. **An offset is invisible.** Every segment still looks well formed; only the numbering is shifted, so pellet 7 as scored on the bench is compared against footage of pellet 8. A triage reviewer who sees it escalates the video to re-segmentation, where the cuts are fixed.

3. **A re-cut can silently discard the review.** `index_review_by_segment` drops any review whose frames no longer line up with a current segment. Until 2026-10-21 the release check still counted such a review as resolved by number, so the video finished carrying the algorithm's call. Since then the release check matches by frames as well, and the segment stays open for a person.

Reaches themselves are grouped by segment in two different ways depending on file shape: nested output is trusted as grouped by the detector (`causal_review_widget.py:1024-1030`), while flat output and any manual re-cut are re-grouped by which segment window contains each reach's midpoint (`:1011-1023`).

---

## The queues, and how a video leaves one

### Deep review supersedes triage (since 2026-11-15)

**A video that is in the re-segmentation (deep review) queue is never in the triage queue as
well.** Deep review is the stronger hold: the whole video, cuts included, is in question.
- **Nothing routes a video in deep review into triage.** The watcher's gate, the return scan's
  diverts, `mousereach-route-to-queue` and every other route go through `route_to_queue`, which
  creates no triage bundle for such a video, leaves it in deep review, records `deep_review` in the
  watcher database and logs the refusal. `mousereach-route-to-queue` reports it as `deep` (not a
  failure) and flags no segment.
- **Moving a video into deep review from anywhere but its triage bundle empties its triage
  bundle, keeping what a person wrote** (Flag Session from a window not on the triage bundle,
  the watcher's own routes, `mousereach-route-to-queue`) once the deep-review bundle is complete:
  - the video, or a pose file, is moved into the deep-review bundle when that bundle has no video
    (or no pose file of that type) at all -- it may be the only copy;
  - then the triage folder is a **leftover** (see "Leftover queue folders" below), with the
    deep-review bundle as the video's current copy: its machine results and queue records, any
    file the deep-review bundle holds byte for byte, and the triage review when the durable
    review store (the review records folder every reader looks in) already holds it byte for
    byte are deleted. Nothing is written into the store from here: the triage answers were made on
    cuts that are about to be redone, and the store is read after the re-cut (where answers given
    before a person's new cuts count only on unchanged pellets; see below).
  - everything else -- ground truth, a review the store does not hold byte for byte, a clear marker,
    hand-set cuts, a video or pose file that differs from the deep-review bundle's -- **stays in the
    triage folder**, with a note, and is listed under "Folders left for a person" in the Review
    Queues tab (and in the dashboard's "Needs your attention" panel).
  **No answer is moved into the deep-review bundle this way** -- not the review, not ground truth,
  not hand-set cuts, not a clear marker: the video's cuts are about to be redone, and
  re-segmentation voids every answer made on the old cuts.
  **Escalate and the mislabel divert are different:** they move the triage bundle ITSELF into deep
  review, and every file in it goes along -- the review and any ground truth included. Once a person
  saves new cuts, every answer given before then -- in the bundle's review or the durable review
  store's copy -- no longer counts unless its pellet's frames did not change; each voided answer
  is kept as a record and its pellet is asked again in triage (see "A re-cut voids the answers given
  before it").
  If a file failed to move into deep review, the triage folder is not touched at all, because the
  file may still be there: the attention panel lists it. Once the deep-review bundle has every
  file -- the watcher moved it in, or a person moved it in by hand or with the Resolve button --
  the next check (within ten minutes) empties the triage folder as above. The emptying is done
  under the video's claim (the same claim every queue move takes): if another program or machine
  is moving that folder at that moment, the route leaves it alone and the watcher's next check
  deals with it -- two programs never work on one folder at once.
  **Why:** the whole triage folder used to be deleted, and a person's ground truth or clear marker
  in it had no copy anywhere.
- **A triage copy of a video in deep review is ignored everywhere:** the triage review tool never
  shows it and the Review Queues tab does not count it; the census, the dashboard's folder view and
  `mousereach-reconcile` count the video in deep review only; the return scan never returns,
  releases or diverts it. The return scan empties such a copy the same way the first time it sees
  it (one log line), so copies left from before this rule are dealt with too: a copy that only
  repeats the deep-review bundle byte for byte is deleted; any other keeps what has no copy
  elsewhere, listed.

**Why:** a router that did not check sent videos already in deep review into triage from their
filed results. The triage copy held older results than the deep-review copy, and a reviewer could
answer the wrong one.

### Moving a video into or out of a queue never deletes or overwrites a person's file

A queue bundle holds what a person did (a review, ground truth, a clear marker, cuts set by
hand) as well as what the machine made. Every move into, out of or between the queues follows
these rules:

- **A video sent to the queue it is already in is left exactly as it is.** If a route is asked
  to move a bundle into the queue it is already in, nothing is moved and the bundle's routing
  record and manifest are not rewritten; only the watcher database is set to that queue's state.
  (Flag Session in the re-segmentation window does not route at all -- see "Flag Session on a
  video that is already in the re-segmentation queue does not move it" below.) **Why:** the move
  used to "move" each file onto itself, and the overwrite step deleted the file first -- every
  file of the bundle except the video the window had open was lost.

- **A video is never poured into a folder it already has in that queue.** When a route finds
  `<queue>/<video>` already there and starts anywhere else (the processing folder, the filed
  results in Analyzed, the other queue), it looks at that folder before anything moves:
  - empty, or holding only the queue's records: not a bundle -- deleted, and the route goes on;
  - an old copy of the video: a **leftover** (see "Leftover queue folders" below), with the files
    this route brings as the video's current copy. Everything with a current copy elsewhere
    is deleted (a review only when the durable review store already holds it byte for byte), and
    the route goes on. If anything is left -- a person's file, or a video or pose
    file this route does not bring -- the folder stays where it is, listed, and the route is
    refused (it never merges into that folder): see the refusal below. A folder is an old copy
    when it holds only what the pipeline makes (results, video, pose -- no queue record and
    nothing a person wrote), or when this route is a pipeline run of the video that started after
    the folder was last routed or written (the run has superseded it). So an Escalate from triage
    is never blocked by a deep-review folder of machine results alone;
  - the video's live bundle: nothing is moved and the route stops with "<video> could not be
    routed to <queue> ... the queue already holds its live bundle ...". A folder is the live
    bundle when the watcher state says the video is held in that queue, when there is no watcher
    database to ask, or when it is a published bundle (it holds the queue's routing record or
    review manifest) or holds a file a person wrote -- unless the route is a pipeline run that
    started after the folder was last routed or written. **Why the folder and not only the
    watcher state:** the state is one machine's record and can be out of date. A bundle once
    escalated from the triage window without the record being updated kept the state `triage`
    while the video and the reviewer's saved review sat in the deep-review queue; judged by the
    state alone, the next route of that video would have moved the reviewer's bundle out of the
    queue. The Review Queues tab ("Folders left for a person") and the dashboard's attention
    panel list the refusal (file
    `attention/<machine>_queue_folders_for_a_person.json` under the pipeline's NAS folder) until
    the folder is gone or the video is routed. **What to do:** a video is in one place at a time
    and nothing is merged into a live bundle. If the bundle in the queue is the copy to keep,
    finish its review as usual: when it returns to the pipeline, it replaces the newer run's
    results in the processing folder. If the newer run should replace it, a person must take that
    bundle out of the queue first. (There is no button for either: which copy is right is a
    person's call, and the machine never deletes or hides a copy that may be the only good one.)
    From the review window's Escalate button it shows as an error box and the video stays where
    it was. When the watcher's own pipeline meets such a refusal (or an old folder that keeps a
    person's file), the video is NOT marked failed: it stays in the
    processing folder as it was, its processing log gets the line "review_gate refused" with the
    reason, and the pipeline tries it again after half an hour. **Why:** a refusal is not a result
    about the video, and marking it failed used up its retries while a window was open.
    **This repeats every half hour until the cause goes away** -- for a queue that holds the
    video's live bundle, or an old folder kept for a person, until a person has dealt with that
    folder; nothing ends it by itself. Each repeat is cheap: when the refusal came after the analysis (at the review step
    or the hand-score check), the next try reuses the results already in the processing folder and
    only checks and routes again; segmentation, reach detection and outcomes are not redone. (After
    a watcher restart the video runs in full once; a refusal after a failed segmentation always
    segments again, because that is what sends it to manual segmentation.)
  **Why:** the files used to be poured into the existing bundle, replacing same-named files -- an
  older filed generation replaced the held run's results, and a review a person had saved there
  could be replaced by an older one. Refusing every existing folder instead made every later run
  of the video fail when the folder was only a leftover.
- **`mousereach-route-to-queue` never moves a held video, and puts its question in the held
  bundle.** A video whose watcher state is `triage` or `deep_review`, or that has a bundle in
  either queue (published, or still being built in the queue's `.incoming` folder), is not moved
  and its filed copy is not touched. The command prints it as `held` (`"held": true` with
  `--json`). If segments were asked about (`--flag-segments`, or a worklist's `segment_nums`),
  they are flagged inside the held triage bundle's `<video>_pellet_outcomes.json` -- the file the
  triage review tool walks -- with the reason given, exactly as for a routed video (a segment a
  person already answered is skipped; `--recheck` reopens it). That is a success (exit 0). When
  the question cannot be recorded -- the video is held in deep review (its results are made again
  after re-segmentation), only the watcher state says it is held, the held bundle has no outcomes
  file or not those segments, its files cannot be read, deep review was asked for a video held in
  triage, or the return scan is moving the bundle right now -- the line says `FAIL ... question
  not recorded: <why>` (`"unrecorded": [segments]` with `--json`) and the exit code is 1: ask
  again later (after the video is filed again, a normal route takes the question).
- **A route-to-queue run that does not route changes nothing in the filed results.** The
  command decides what to ask by reading only; it moves the video first and writes the flags into
  the bundle's own `<video>_pellet_outcomes.json` after the move has succeeded. A route that fails
  (every file goes back) or is refused (the video is held) leaves every filed file exactly as it
  was. If the move succeeds but the flags cannot be written, the line says `FAIL ... routed;
  question not recorded: <why>` and the exit code is 1; run the command again for that video -- it
  is now held, so the question goes into its bundle. While the watcher's return scan (or a review
  window) is moving the same video's bundle, the command prints `wait` for it and moves nothing;
  offer it again later.

- **A route records the hold as soon as the video is in the queue.** The order is: move the
  bundle in, set the watcher state (`triage` / `deep_review`), then write the review manifest (the
  `<video>_manifest.json` the review tools list a bundle by). If that last write fails, the error
  is logged and the video is still correctly held. The watcher's return scan looks for such a
  bundle every ten minutes (a bundle with a routing record, some of the video's files and no
  manifest) and writes the manifest again from the routing record -- including the step that sent
  the video, so its return still resumes just after that step; the log says "had no review
  manifest ... written again so the review tools list it". That check opens a bundle's routing
  record only when the record changed since the last check, so it stays quick on a queue of
  thousands of bundles. Until then the bundle is in the queue
  folder but not offered in the review tools. **Why:** the manifest used to be written before the
  state, with a plain write; a passing hold on the share made the route fail with the files
  already moved, the row never said "held", and no tool offered the bundle.

- **Returning a cleared video to the pipeline is all or nothing.** The return plans every file
  it will move, moves the video and pose first and a re-segmentation clear marker last, and sets
  the video to `processing` as part of the same step. If any file cannot be moved (held open by a
  review window, an indexer or a backup) or the state cannot be set, every file already moved is
  put back into the bundle, the bundle stays in its queue exactly as it was, the log says "could
  not complete ... every file moved so far was put back", and the return is tried again after an
  hour.
  **A file of the same name already in the processing folder:** identical to the bundle's -- it
  is simply replaced (nothing is lost); a machine result of an earlier run (segments, reaches,
  outcomes ...) -- replaced by the bundle's, which is the video's current copy; **a person's file
  (a review, ground truth, hand-set cuts ...) or a different video or pose -- the return does not
  start at all.** Nothing is moved, the bundle stays whole in its queue, the log says "the
  processing folder ... already holds a different <file> ... which a return never replaces", and
  the dashboard's "Needs your attention" panel lists it. **What to do:** look at that file in the
  processing folder; delete it when it is not needed (or move it to where you keep such files).
  The return is tried again within an hour and the line then goes away by itself. **Why:** the
  return used to replace such a file -- a person's hand-set cuts were lost.
  The bundle's queue records (`<video>_manifest.json`, `<video>_routing.json`) are deleted
  only after the state is set. If a file cannot even be put back, an ERROR names it and both
  places: that bundle is split and a person moves the file back. **Why:** the return used to stop
  at the first file it could not move, with the video, pose and answers already in Processing and
  the queue records already deleted; the half left behind read as unanswered and no review tool
  listed it.

- **What counts as "a person's file".** The video (`<video>.mp4`) and its pose files
  (`<video>DLC...`) are the inputs. The machine outputs are one fixed list -- `_segments.json`,
  `_reaches.json`, `_pellet_outcomes.json`, `_reach_assignments.json`, `_features.json`,
  `_processing_manifest.json`, `_triage.json`, `_results.csv` -- plus the queue's own records
  (`_manifest.json`, `_routing.json`). Every other file of the video is treated as a person's: a
  review, any ground-truth file, a clear marker (current or set aside), an older tool's
  validation file. A `_segments.json` whose cuts a person set (`"boundary_source": "human"`, or a
  file that cannot be read) is a person's too. Anything not on the machine list falls on the safe
  side on purpose: a person-written name nobody listed is never deleted by accident.
- **A move never replaces a person's file with a different one.** If a route finds a person's
  file of the same name already in the bundle and the incoming file differs, the incoming file
  stays where it was and the routing record lists it under `failed_files` with "is a person's file
  and differs ... not replaced". An identical copy may replace it (nothing is lost).
- **A stale bundle is a leftover, never filed away unseen.** The return scan clears bundles it
  judges leftovers -- for example a triage bundle of a video the pipeline has since re-handled,
  or a bundle of a video that has since been filed -- by the rule in "Leftover queue folders"
  below, with the video's current copy (this machine's processing folder, or its filed results)
  as the copy to check against. **Why:** such bundles used to be moved to an archive folder that
  nothing reads, a person's ground truth or hand-set cuts included.

- **A route that could not move every file is finished later.** When the watcher sends a video
  from its processing folder to a queue and a file cannot be moved (most often the video, still
  held open for a moment), the route goes on and the routing record lists that file under
  `failed_files`, with where it still is. Every ten minutes the watcher's return scan moves such a
  file into the bundle once it can (the log says "finished <file> -- left behind by a route that
  could not move it at the time"), and the routing record then lists it under `finished_later`. A file at the source that
  is newer than the route is NOT moved in (it belongs to a later run). A listed file that is in
  the bundle and no longer at its old place (a person moved it in with the Resolve button, or
  it arrived some other way) counts as finished too: it is crossed off the routing record (the
  record keeps its old modified time) and listed under `finished_later`. Until a bundle is whole, it
  is listed in the dashboard's red "Needs your attention" panel (file
  `attention/<machine>_unfinished_routes.json` under the pipeline's NAS folder), with what is
  missing and why. If it stays listed, press **Resolve...** in the Pipeline Dashboard (see
  "The Resolve... button" below); the listing clears, and for a video in deep review its triage
  folder is emptied on the next check. The routing record keeps each finished entry whole (what
  went wrong, where the file was) with how and when it was finished.
  - **A file held open is never left in two places.** A file that is open somewhere cannot be
    renamed, so moving it copies it and then fails to delete the original; the route now deletes
    that copy again, so the file stays only where it was. If an older route left such a copy (the
    file in the bundle AND at its old place), the check compares the two: identical -- the old
    place's copy is deleted and the file counts as finished; different -- both are kept and the
    panel lists it ("a different copy is still at ...; a person decides which is right").
  - **Only the machine that made the route touches its own disk.** A file left in a machine's own
    processing folder is finished only by that machine (the routing record names it,
    `routed_by`); other machines leave it alone and do not list it. A file left on the NAS is
    finished by any machine. A record written before records named the machine is never acted on
    for a file on a machine's own disk: if such a file is there, the panel lists it for a person.
  - **Only while the video is still held in that queue.** The watcher moves a left-behind file in,
    or deletes an identical copy at its old place, only when its own record says the video is
    still held in that queue (`triage` for the triage queue, `deep_review` for the deep-review
    queue). If the record says anything else -- for example the video is running again in
    Processing -- or this machine has no record of the video, nothing is moved or deleted and the
    panel lists the file with that reason ("the watcher record says the video is 'processing', not
    held in this queue ... a person decides"). **Why:** the old place is the processing folder the
    video was routed from, and a video running again there brings its video and pose files back
    with their old modified times, so they look like the routed files; without this check the scan
    could move the running video's file into the old bundle, or delete it. The Resolve button's
    "Try to move the missing files in now" uses the same check when the computer has a watcher
    record; a file still at its old place gets no button -- the window shows a line saying why
    instead ("<file>: no button ... the video may be running there again"), and the watcher
    finishes the move by itself once the video is back in that queue. On a computer without a
    watcher record, the person pressing the button decides.
- **A route out of the filed results is all or nothing.** `mousereach-route-to-queue` and the
  watcher's "segment mislabel" divert take a video's files out of its filed folder in Analyzed.
  If any file cannot be moved, every file already moved goes back to the filed folder, nothing
  appears in the queue, the watcher state is unchanged, and the command reports the video as
  `FAIL` with the reason; run it again later. **Why:** a filed generation left half moved still
  counted as a filed result for the files left behind, while the queue held the rest.

### Leftover queue folders

A video is in one place at a time, so a folder in a review queue that is not the video's live
bundle is a **leftover**. This covers: an empty folder or one holding only queue records (left by a
return that could not finish its last step); an old copy of a video met by a newer run's route
when the video is not held in that queue; a stale bundle the return scan judges a leftover; what is
left of a triage copy when the video moves into deep review; and a released quality hold (see
"Holds the machine made alone" further down). For each file in it, by the rule in
`review_routing.clear_leftover`:

- **Deleted, once it is verified that a current copy exists elsewhere:** the queue's own records
  (`<video>_manifest.json`, `<video>_routing.json`); any file the video's current copy holds byte
  for byte (the live bundle, the files a route is bringing, the processing folder or the filed
  results -- whichever the step names); the review, when the durable review store already holds it
  byte for byte (nothing is ever written into the store from a leftover: its answers were made on
  cuts that are being or have been redone, and the review readers match the store's copy to the
  current cuts); and the machine results (segments,
  reaches, outcomes, ...), when a current copy of the video exists at all and no person checked or
  corrected them -- the pipeline makes those again from it.
- **Kept, where it is:** everything else -- a person's file (a review the store does not hold byte for byte,
  ground truth, a clear marker, hand-set cuts, any file not on the machine list above), a video or
  pose file with no identical copy elsewhere, a subfolder, and every file when no current copy of
  the video can be found at all. Nothing is moved anywhere.

**Which results files count as a person's.** The review tools save a person's work INTO the
files the pipeline also writes, so the file's name does not settle it; what is inside does. A
segments, reaches or outcomes file counts as a person's file when it holds any of:
`validation_status` "validated" (the per-step segmentation, reach and outcome review screens when
a person saves; the retired whole-video Review Tool in files it saved before it was retired),
`boundary_source` "human" (cuts set in Re-segmentation), `human_verified` or `human_corrected` true
on any segment or reach (the per-step reach and outcome review screens; the retired Review Tool in
older files), or `triage_cleared` true on any segment (written by `mousereach-resolve-triage-from-gt`, and by
the retired triage-clearing tool in files it saved before it was retired). A file
of those three that cannot be read counts as a person's too (what it holds cannot be told). Such a
file is never deleted as a leftover, never replaced by a route moving a newer run into a bundle,
and never replaced by a return moving a bundle back into a processing folder -- unless the other
copy is the same file byte for byte. The pipeline itself only ever writes "needs_review" or
"auto_approved" and never sets those marks (`review_routing.is_person_file`). **Why:** judged by
name alone, a person's corrected reaches or outcomes were "machine results", and clean-up deleted
or replaced them.

A folder that ends up empty is removed. A folder with something kept **stays in its queue** with a
note `LEFT_FOR_A_PERSON.json` inside (why, what to do, what was kept). It is listed in the review
tool: the **Review Queues** tab, panel **"Folders left for a person"**, which reads the queue
folders themselves, so every such folder is shown whichever machine left it. It is also listed in
the Pipeline Dashboard's red **"Needs your attention"** panel (file
`attention/<machine>_queue_folders_for_a_person.json` under the pipeline's NAS folder) with the
same words.

**Such a folder is not the video's hold, to anything.** The watcher's return scan never returns,
releases, diverts or clears again a folder holding that note; and no part of the pipeline counts it
as "the video is held in this queue": not the deep-review-beats-triage rule, not the route command,
not a GPU machine deciding whether a video still needs a pose, not a re-run looking for a pose,
not the census, the consistency check or the dashboard's counts (`review_routing.is_queue_bundle`).
So a deep-review folder that is only a leftover never stops the video being routed into triage;
the route goes ahead as normal. **Why:** a reader that counted the leftover as the hold relabelled
the video "deep review", moved nothing, and the video was left with no live bundle anywhere. A route
INTO the same queue that meets the folder is refused rather than merged into it (the video waits in
its processing folder and is tried again every half hour).

**What to do:** in the Review Queues tab press **"Show and resolve..."** under "Folders left for a
person". Each row says which video, why the folder was left and what was kept. Open the folder in
File Explorer (the row names it), look at what is kept, and keep anything you need wherever such
things are kept. When nothing in the folder is needed any more, press **"Nothing in it is needed:
delete this folder"**: it asks first, naming every file it will delete, then deletes the folder and
everything in it. It deletes only a folder that holds the note, directly in a review queue, under
the video's name -- never a live bundle, never anything else. The machine itself never deletes such
a folder; this button is a person deciding to. The row then goes away at once (and a waiting route
goes through on its next try).

A route refused because the queue holds the video's **live** bundle is listed in the same panel
with no button: open that queue's review window and finish the bundle's review; once it has
returned, the route is tried again by itself.
**Why:** a folder a person wrote in used to be deleted whole (a triage copy of a video moving
into deep review) or moved to an archive folder that nothing reads; and machine leftovers piled up
in that archive folder, which is meant only for cases the pipeline cannot process at all. A
machine leftover with a current copy elsewhere is worth nothing; a person's work is never thrown
away or hidden by the machine.

### Recordings held on a recording PC

The same red panel also lists every recording a recording PC holds back instead of sending it to
the intake (a name that is not a valid session name, an empty or broken file, or a different file
of the same name already on the shared drive), one line per file: which PC, which file, and why.
These have no button: the file is on the recording PC. `docs/SENDING_RECORDINGS.md` says what to do
for each reason.

### The Resolve... button

When the dashboard's red "Needs your attention" panel lists a review bundle that a route left
without all its files (`route_unfinished`), a **Resolve...** button appears under it. It opens a
window with one row per such bundle, its explanation, and a button for each thing that can be
done. Every button first says what it will do and asks "Go ahead?"; afterwards a message says what
was done. The same window opens from the Review Queues tab's "Folders left for a person" panel.

- **Try to move the missing files in now** -- runs the watcher's ten-minute step at once. It moves
  each missing file into the bundle; where the bundle already holds the same file byte for byte, it
  deletes the duplicate still at the route's old place (nothing is lost). A copy that differs is
  never deleted.
- **Accept the bundle without <file>** -- offered only when the file is in neither place: crosses
  it off the routing record (the record keeps the entry, who accepted it, when, and on which
  machine). Use it only when the file is gone for good.
- When a file is in the bundle AND still at its old place and the two differ, there is no button:
  the line says where both copies are. Compare them, keep the right one in the bundle and delete
  the other by hand; the watcher crosses the file off by itself once only the bundle's copy is
  left.
- While the watcher record says the video is not held in the bundle's queue (for example it is
  running again in Processing), a file still at its old place gets no button either: the line says
  so, so the running video's files are never taken.

If the watcher, a review window or the route command is moving the same video at that moment, the
button refuses ("being moved by another program right now -- try again in a minute") and changes
nothing.

A leftover folder kept for a person (and a route refused because of one) has one button,
**Nothing in it is needed: delete this folder** -- see "Leftover queue folders" above; it is the
only button that deletes a person's file, and it names every file before it asks. A route refused
because the queue holds the video's live bundle gets a line saying to finish that bundle's review,
and no button. The other review-queue lines in the panel -- a return waiting on a file in the
processing folder, a person's file left in a processing folder -- have no buttons: each says what
to do, and goes away by itself once the file in question is gone.

### Where they are

Set in `config.py:164-166`, derived from `NAS_ROOT`:

| Setting | Path | Holds |
|---|---|---|
| `Paths.REVIEW_ROOT` | `<nas>/Processing/Review` | parent of both queues; `flagged_sessions.json` and `_QC/` live here |
| `Paths.TRIAGE_REVIEW` | `<nas>/Processing/Review/triage` | per-element questions |
| `Paths.DEEP_REVIEW` | `<nas>/Processing/Review/deep_review` | failed segmentation, failed quality control, escalations |

The re-segmentation folder was once `Processing/Review/flagged_for_review`. That name is retired for the folder only -- `flagged_for_review` is still a field in the segments, reaches and outcomes files and a database column. A migrated drive holds a plain file at the old folder path, so old code that tries to use it fails loudly instead of rebuilding it.

An unset `nas_root` makes all three `None` (since 2026-10-22), and the launcher tab reports "queue not configured" (`queue_launcher_widget.py:112-113`). Run `mousereach-setup`, or use the Setup panel's "Shared pipeline folder" field, to set it. **Why no fallback:** until 2026-10-22 an unset `nas_root` was guessed as a fixed folder on the NAS drive from a retired layout, so the queue paths resolved to real-looking folders in the old place and the review tools would have worked there as if nothing were wrong. Saving from setup also used to delete `nas_root`, which made the guess a live risk on every machine. `Paths.NAS_ROOT_ORIGIN` is now `config` or `unset`.

### How a video enters

**A filed video whose pose from an older model cannot be used is sent for a new pose, not to a
queue.** Before segmentation, on every tray and every run, the watcher asks one question: is this
pose usable (`pipeline/pose_check.py`)? Usable means the pose file loads as a pose table with at
least one frame, has the two ruler points `SABL` and `SABR`, and -- for a tray that is segmented,
i.e. not Easy or Flat -- the reference points `BOXL` and `BOXR`. A pose that is not usable is a
problem of the pose step, and nothing a person does in a review window fixes a pose. What happens
then is settled for ONE case only:
- a video that is already filed (its video is in Analyzed) whose unusable pose comes from an
  OLDER pose model is marked for a new pose (state `outdated`, re-run scope `full`); the watcher
  then asks a GPU PC for a pose from the current model over the shared re-pose queue
  (`Processing/Repose_Queue`), and the new pose is taken in when it arrives. Its processing log
  gets a `pose_check failed` line, and it is shown in grey under the red panel ("N video(s) wait
  for a new pose ...", file `attention/<machine>_waiting_for_pose.json`): it goes on by itself, and
  the line disappears once a usable pose arrives.
- **every other case goes on exactly as before this check existed** -- the check only writes the
  `pose_check failed` line into the processing log: a filed video whose unusable pose comes from
  the CURRENT model (a new pose from the same model may come out the same, and the re-pose queue
  would take this same file again), and a video that is not filed yet (there is no way built for
  a GPU PC to reach its video for a new pose). Such a video usually fails segmentation and goes
  to the re-segmentation queue, as it always did. **What the watcher should do in these cases is
  waiting on a lab decision**, and so is how many unusable poses a video may have before it is
  shown in red.
The quality check's two pose-level findings -- "Could not read DLC file" and "Missing SABL or
SABR" -- take the same path for the one settled case; in every other case they hold the video
for review as before. A pose file that cannot even be opened (a permission problem, a drive that
blinked) is a machine fault and is handled as before: the video is marked failed, never sent for
review.

`run_gate` (`review_gate.py:209`) runs after reach assignment and before kinematics. `evaluate_gate` (`:68`) decides, in order:

1. Ground truth certifies the whole video -> **clean**, proceed, even if segmentation failed. "Certifies the whole video" is computed from the ground-truth file as it is now (`triage_status.gt_resolved_segments`): BOTH the reaches and the outcomes are marked exhaustive AND every pellet wanted (the video's current segments, leaving out slivers too short to be a pellet presentation; else every pellet the ground truth lists) has an outcome a person marked determined. A stored `completion_status` is not trusted, because it can be stale or set while outcomes are still undetermined.
1b. An Easy or Flat tray video -> **clean**, always (since 2026-10-23): these trays are never held for a person. They are not segmented and get no outcomes, so there is nothing a reviewer could answer; a machine problem is only recorded in `_triage.json`. The stale-output checks still apply (they send a video round again, not to a person).
2. Segmentation failed — no segments file, confidence at or below 0, or a "reference quality" anomaly -> **DEEP_REVIEW** (`:87-88`).
3. Quality control returned `needs_review` -> **DEEP_REVIEW** (`:107-108`). Quality control calls a missing segments, reaches or outcomes file critical, but asks for the outcomes file only where outcome detection runs: Easy and Flat trays skip it by design (`watcher/router.SKIP_STEPS`), and until 2026-10-22 every Easy/Flat video was held here for that missing file (`pipeline/triage.py`, `check_structural`).
3b. The quality check could not run at all (it raised) -> **DEEP_REVIEW** (since 2026-11-15), routed reason `triage_failed: the automatic quality check could not run on this video -- a person checks it`. The reviewer checks the video in the re-segmentation tools and clears it as usual; a clear is honoured on later runs. Before, a crashed check let the video pass as clean.
4. Any triaged segment not resolved by ground truth or a saved review -> **TRIAGE** (`:114-118`).
4b. Reach assignment (which reach caused each touched pellet) could not run -> **TRIAGE** (since 2026-11-15), routed reason `reach assignment could not run (<error>) -- choose the causal reach for segment(s) ...`. Its file is replaced by one naming no causal reach, so exactly the touched pellets are asked; the reviewer picks the causal reach for each and the video is released as for any triage hold. A video with no touched pellet has nothing to ask and is not held. Before, a failed assignment let the video pass with no causal reach.
5. Otherwise -> **clean**, kinematics runs (`:119`).

The segmenter's own `needs_human` verdict is read into the status object (`triage_status.py:184`) and deliberately not acted on (`review_gate.py:89-105`).

**Where the gate looks for ground truth (since 2026-11-03).** First in the folder the video is in
-- the processing folder during a run, the bundle in a queue -- and only then in the corpus-wide
index (`triage_status.gt_resolved_segments`, the one reading of ground truth shared by the gate
and `mousereach-route-to-queue`). The queue's release check does not yet look inside the bundle:
a triage bundle holding ground truth that answers every open segment would be released on the
next scan, which waits on a decision. Until it is decided, one such case is LISTED rather than
left unseen: a triage bundle whose automatic segmentation failed and whose ground truth inside the
bundle certifies the whole video is neither sent to re-segmentation (the gate would pass it) nor
released (its triaged pellets count as open) nor offered by the review window (it has ground
truth). The return scan lists it for a person -- the Pipeline Dashboard's red panel and the
Review Queues tab's "Folders left for a person" -- with that reason and no button
(`review_return._note_gt_not_counted`, kind `triage_ground_truth_not_counted`), and takes it off
the list once it no longer applies. Nothing is moved; the video stays in the triage queue as it
is. **Why:** the index is built once per watcher
process, and a re-run copies the ground truth into the processing folder after that; reading the
index alone, the gate sent videos whose open segments ground truth had answered back to a person.
The rule: a segment whose boundaries, causal reach and outcome have all already been answered by a
person never needs a person again.

**A filed video re-run for kinematics only (since 2026-11-03)** re-runs none of the four algorithms,
no quality check (so item 3 does not apply) and no hand-score
comparison; the gate still runs and can ask only about a segment no person answered
(`filed_rerun_reuses_analysis`, `orchestrator.py`; docs/PIPELINE_AS_BUILT.md, update of 2026-11-03).

On a hold, `route_to_queue` (`:159`) moves the whole bundle out of the processing directory, writes a queue manifest so the review tool can open the bundle in place (`:138-156`), and sets the database state to `triage` or `deep_review`. If the normal state transition is rejected the code forces the state rather than letting disk and database disagree (`:178-194`).

A new bundle is built out of sight and appears in one step. `move_video_bundle` (`review_routing.py`) moves the files into `<queue>/.incoming/<stem>/`, writes the routing manifest there, and then publishes the folder as `<queue>/<stem>/` with a single directory rename. **Why:** a move onto the shared drive is a copy that takes seconds per file, and the watcher's return scan reads the queues meanwhile. When the bundle folder was created first and filled in place, the scan saw it empty and filed it away mid-route (every remaining move then failed with "No such file or directory"), and a half-filled folder could be judged as a bundle before its files arrived. The readers that judge, retire or release bundles skip names starting with `.` (readers that key on stem-named files match nothing there by accident, not by rule), and the build folder sits under the queue itself so the rename never crosses a drive. A route into a queue that already holds a bundle of the video is refused before anything moves (see "Moving a video into or out of a queue never deletes or overwrites a person's file"); `move_video_bundle` itself still fills an existing bundle in place for the two cases that need it: a concurrent route of the same video that published first, and the fallback when the publishing rename fails. The routing manifest lists `moved_files` and `failed_files` (each file that could not be moved, with the error), so a part-moved bundle shows in the record rather than only in a log. A route that dies partway leaves its files in `.incoming/<stem>/` and the next route of that video reuses them; until then `mousereach-reconcile` lists the video as held, "being routed into <queue>, or an interrupted route -- check if this persists". If a concurrent route of the same video publishes the shared build folder first, the rest of this route's files and its routing manifest go into the published bundle, and a failed manifest write never stops the publish. A file that could not be merged in the fallback stays in `.incoming/<stem>/`, and the return refuses that bundle, naming the folder, until a person folds the file in -- returning would re-run the video without it (typically a locked mp4).

`reprocess_to_current._stage_review_bundle` stages bundles for the bring-current path. It goes through `review_gate.route_to_queue` with this machine's watcher database: it builds out of sight, never puts a triage copy beside a deep-review bundle, never overwrites a bundle already in a queue, and records the move. (Before, it created `<queue>/<stem>/` first and filled it file by file, so a return scan could read it half-filled, and it recorded nothing.)

### How a video leaves

**What counts as answered (since 2026-10-21).** A triaged segment counts as answered only when a saved answer is about ITS frames -- the frames the reviewer was shown, stored with every answer -- not merely when an answer carries its number. After a video is re-cut, segment numbers shift; an old answer then counts for the segment whose frames it covers, and any segment no answer covers stays open for review. Short segments a person cut are real segments. The gate, the return scan, this tool's queue and the queue index all use this one rule (`causal_review_io.answered_segments`), so they agree with what the results contain.

**When a file holds two answers for the same frames (since 2026-10-22).** The old number-based restore left some review files with two records for one stretch of video, and the results step used whichever came first in the file. That was sometimes the left-blank one and sometimes the older one. The rule now:

| The two records | What counts |
|---|---|
| A decision and a placeholder, or a decision and a left-blank `triaged` | The decision |
| Two decisions that agree | That decision, once |
| Two decisions that **disagree** (a different outcome, causal reach or ignore windows) | **Neither.** The segment counts as unanswered and goes back to a person, who sees it in this tool like any open segment. |

**An answer whose causal reach lies wholly outside the frames it records is not applied.** It is about other footage: these are mostly answers that the old number-based restore put on the wrong segment, typically shifted by one. The segment counts as unanswered, and the answer stays in the file.

Only `scan_review_queues` (`review_return.py`) moves a bundle out. It runs inside the watcher every 10th poll cycle (`orchestrator.py:1345`, `:1391-1395`) and handles at most **10 bundles per scan** (`MAX_RETURNS_PER_SCAN`, `review_return.py`) so returning does not starve the pipeline.

Because the queues are shared storage scanned by every watcher node, every mutation the scan performs — a return, a stale-bundle retire, a divert to re-segmentation — first takes a per-video claim marker in `{REVIEW_ROOT}/.return_claims/` (`_claim_return` / `_with_return_claim`, `review_return.py`). The marker is created exclusively (`watcher/claim_marker.py`): the file server creates it only if it does not already exist, so when two nodes race exactly one succeeds, and the marker records which host holds it (its text is written first and the finished file is put in place in one step, so a marker is never seen empty even when its writer stops half way) -- and which process: the claim belongs to one process (its id and start time), not to the machine, because on the machine where review happens the watcher's return scan, a review window and `mousereach-route-to-queue` are separate processes that act on the same bundles (a flag written into a bundle the scan was moving out used to be lost). Another live process of the same machine is refused like another machine; a claim whose process has ended (a crash) is taken over at once. A marker of this machine that names no process (written by an older version, for example by a review window started before an update) is treated as held by another process until it is an hour old (`claim_marker.LEGACY_MARKER_TAKEOVER_S`), then taken over -- no move under a claim lasts that long; restarting every review window after an update ends this at once. A stem another node or process holds is skipped this pass and re-examined next scan. A marker is broken only when its holder has stopped writing its status report (`Processing/.node_status/<host>.json`) for 3 hours, both times read from the file server so the machines' clock differences do not matter; an old marker whose holder is still reporting is never broken. If the claim folder cannot be made, nothing is moved that pass. With no shared root configured the claim is a no-op. (Until 2026-10-21 the marker was written, re-read after 0.5 s, and broken after 2 hours by age alone.) Scanning a bundle never takes a claim — only touching it does.

**An empty triage folder is left alone for 30 minutes before it is removed.** An empty folder is not a bundle; it is residue of an earlier divert or return, and the scan removes it -- but only once its modification time is older than `EMPTY_DIR_GRACE_SECONDS` (30 minutes, `review_return.py`). **Why:** a router may have just created the folder and still be moving files in (routing into an existing bundle, or after a failed rename, fills the folder in place), and the time is stamped by the share's clock, which can differ from the scanning node's by many minutes. A time in the future, or one that cannot be read, counts as young. Real residue simply waits for a later scan. The same scan warns once per watcher run about any `<queue>/.incoming/<stem>/` older than 30 minutes and never touches it: those files are the video's only copy, and whether the route is still running or died is for a person to judge.

**A bundle of a video that has since been archived is cleared as a leftover (since 2026-10-22; earlier versions moved it to an archive folder).** If this node's record says the video is `archived` and nobody wrote anything in the bundle (no review, no clear marker, no ground truth), the bundle is a leftover when either (a) it holds no results at all -- typically just the .mp4 a return could not delete while the review tool still had it open -- or (b) every file in it is older than the video's archived results in Analyzed. The scan clears it by the rule in "Leftover queue folders" above, against the video's filed results (`_superseded_by_archive`, `_retire_stale_bundle`, `review_return.py`), in either queue: what the filed results hold too is deleted; a video with no identical filed copy stays, with the note, listed. **Why:** such bundles asked a person for work already done: videos a person had cleared came back to re-segmentation as a bare .mp4 and were cleared a second time. A bundle an external integrator routes into a queue (`mousereach-route-to-queue`) is never caught: its results leave Analyzed when it is routed, and its routing record is written after the archive. Anything a person wrote keeps the bundle in the queue.

**A triage copy that only repeats the re-segmentation bundle is deleted (since 2026-10-22; earlier versions moved it to an archive folder).** Older escalations sometimes split a bundle: the results moved to re-segmentation and the mp4 stayed in triage. When the same video has a folder in both queues and segmentation failed, the scan compares the two. If every file in the triage folder has a file of the same name in the re-segmentation folder with exactly the same bytes, the triage folder holds nothing the reviewer's copy lacks, and the scan deletes it (`_is_leftover_copy`, `review_return.py`). The re-segmentation bundle is not touched. If anything differs (a different byte, or a file the re-segmentation copy does not have), the scan empties the triage copy as a route into deep review does (see "Deep review supersedes triage" above): what has a current copy elsewhere is deleted, the rest stays with a note, listed for a person, and the scan leaves it alone from then on. **Why:** until this rule, every such leftover sat there for good, and the scan logged the same line every few minutes. Sizes are compared first and the files are read only when every size matches. A pair that is not a leftover is read once, not on every scan.

- **Triage**: the bundle leaves when it has triaged elements, all of them are resolved, and segmentation has not failed (`:247`). "Resolved" means a saved review answered the segment **or ground truth determined it** — `triage_status` folds in `gt_resolved_segments` since 2026-08-28, and treats a fully certified video as resolved outright. "Fully certified" is worked out each time it is read: reaches and outcomes both marked exhaustive in the ground-truth tool **and** every pellet's outcome determined by a person. Marking the lists exhaustive while some outcome is still undetermined does not certify the video (the ground-truth tool says so when you mark it), and a stored `completion_status` is ignored. A segment only a few frames long that the segmenter left between two pellets (too short for any pellet to be presented in, see `MIN_REAL_SEGMENT_FRAMES`) is not counted: no outcome can be recorded for it, so it never stands in the way of a video whose real pellets are all determined.

  Before that fix the release side was narrower than the admission side. `evaluate_gate` had always subtracted GT-determined segments, but `triage_status.resolved_segments` read the causal-review document alone, so a bundle whose questions GT had answered could never satisfy `fully_resolved`. The review tool skipped the same bundle because `has_gt()` said there was nothing to ask. Nobody asked and nobody released, so such bundles stayed stuck there.
- **Deep review**: the bundle leaves only when a person has cleared it -- its clear marker `{stem}_deep_review_cleared.json` exists (`_deep_review_cleared`, `review_return.py`, the same test the review gate makes) -- and, for a Pillar tray, a person has also saved its cuts in the Segmentation Fixer after the video arrived in the queue (`resegmented_since_entry`, `review/resegmented.py`). Until both are true the video stays, and the scan logs once why ("is cleared but not re-segmented: ..."). Easy and Flat trays are never segmented, so for them the marker alone releases the bundle. **Ground truth does not release a deep-review bundle**, and saved ground-truth boundaries are not proof of new cuts: nothing copies ground-truth boundaries into the segments file, so a video released on ground truth alone would be cut again exactly as before and sent straight back. To release a video, set or confirm its cuts in the Segmentation Fixer, save, then press Clear. (Older versions also released a bundle when any ground-truth file sat in it, even an empty progress save.)
- **Deep review, held by the quality check alone (since 2026-10-22)**: a bundle whose routing reason is `qc_needs_review` and that holds no file a person wrote (a review, a clear marker, ground truth) has the quality check re-run on it. The pose-tracking layer is not re-run -- its stored flags are carried over, so a hold it made stays (except a ruler-variation hold: since 2026-10-23 the check only records a ruler that varies within a video, so those holds are released) -- and the file-based layers are. If nothing critical remains, the bundle leaves for a fresh run: its mp4 and pose files go back to the posed folder (`Processing/Posed`), the video's state becomes `dlc_complete`, and then its other files are cleared as a leftover with the posed folder as the current copy (machine results, records and copies of the posed files deleted; for an Easy or Flat tray, which is released even with a person's file in it, that file stays in the bundle folder with a note, listed -- nothing goes to an archive folder), so intake takes it in its priority turn (Easy and Flat only when nothing else waits). It is not sent straight to `processing`: anything there blocks intake of new work. Nothing in the posed folder is replaced. The posed folder usually still holds the video's files (intake leaves them there until the video is archived); the same file there -- same size and modified time (the quick check rsync and robocopy use; every copy in the chain keeps the time), or the same SHA-256 when the times differ -- is kept as the one copy and the bundle's duplicate is removed, while a copy that differs leaves the bundle for a person. If a move fails, every move is undone and the bundle stays. Re-checks take at most 30 seconds of each scan, and a bundle is re-checked once until its files change (`_qc_hold_lifted`, `_release_to_posed`, `review_return.py`). **Why:** a corrected check otherwise leaves every hold it made by mistake in a person's queue -- the check once demanded an outcomes file Easy and Flat trays never get, and such bundles sat in re-segmentation for that alone.
- **Triage with a failed segmentation**: the return scan diverts the bundle to re-segmentation -- it can never satisfy the triage release condition (which requires a sound segmentation), and before this divert existed such bundles had no route out at all. The processing-side gate honors an existing human clear marker even when the segmentation self-check still reports failure, and since 2026-09-09 the return scan honors the same marker: a bundle carrying `{stem}_deep_review_cleared.json` is never diverted (the re-segmentation question is answered) and may release from triage once its triaged segments are answered, seg self-check notwithstanding. The scan also leaves the bundle in triage, undiverted, when the video's ground truth certifies every part of it (`_gt_certifies_whole_video`, `review_return.py`: the ground truth marks the whole video complete, or its reaches and its outcomes both exhaustive -- the test the review gate makes): the gate passes such a video even with a failed segmentation, so diverting it would hold a video the gate releases. Ground truth that covers only part of the video does not stop the divert -- the gate would send that video to re-segmentation anyway. (Older versions let any ground-truth file stop the divert.) Before that, the gate's honor and the scan's raw `seg_failed` divert fought each other and a cleared video with unresolved triage segments lapped processing -> triage -> re-segmentation -> processing indefinitely.

Returning moves the bundle's data files, including the review file, into the local processing directory, sets the video back to `processing` so the pipeline re-runs it and the gate re-checks, and only then drops the queue-only metadata and deletes the now-empty bundle directory (`_return_to_processing`). It is all or nothing: if a move or the state write fails, every moved file goes back into the bundle (see "Returning a cleared video to the pipeline is all or nothing" above). Queue-only metadata is matched by exact name -- `<stem>_manifest.json`, `<stem>_routing.json`, `manifest.json` -- so `<stem>_processing_manifest.json`, the video's provenance record, travels back with the bundle. **Why:** the old suffix match on `_manifest.json` caught the processing manifest too and deleted it from every returned bundle. A normal re-run regenerates it before the review gate; that regeneration is best-effort, though, so if it fails, or a segmentation failure routes the video before it runs, the old manifest stays beside the new outputs. The pose recorded for the re-run is the one `_resolve_inputs` selected (`select_pose_file`), not whichever `.h5` happened to move last. **Why:** a bundle can hold two poses (a re-pose leaves the old model's file beside the new one), and the last-moved rule re-ran such videos on the wrong, often older, pose. Gate-routed bundles carry their own mp4 and pose; only older staged bundles point at an mp4 and pose left in Analyzed. It refuses and leaves the bundle alone if files of the video are still in the queue's `.incoming/<stem>/` build folder, if the database row cannot be created, if the pose file cannot be found, or if the state cannot be set — a clearance is never spent on a run that would fail. The pose is looked for, in order, in the bundle, the bundle's manifest, Analyzed, pose staging (`Processing/Posed`), and -- since 2026-10-21 -- the processing folder the return delivers to and the pose the video's own row records; outside the bundle only a pose of the currently declared model is accepted. **Why the last two:** an older divert between the queues moved only the mp4 into re-segmentation and left the pose and results on the node that ran the video, and such cleared videos then failed to return on every scan. A return refused for a missing pose or for files left in `.incoming` is tried again after an hour (`RETURN_RETRY_S`), not on every scan, and its ERROR line says so. Each scan re-reads a triage bundle's files only when the bundle has changed since the last scan (its files' names, sizes and modified times) or the video's ground truth has; saving an answer changes the bundle, so it is seen on the next scan.

**No review tool moves a bundle between folders.** The only `shutil` calls in `review/` are the Segmentation Fixer copying a file aside before overwriting it (`fix_segmentation_widget.py:444`) and the ground-truth migration helper (`unified_gt.py:877`). A bundle with a complete review sits in the folder until the watcher's return scan picks it up.

**A triage bundle with zero triaged elements never leaves.** The return condition requires `st.has_triage` (`:247`), so a bundle whose triage flags were cleared by some other route sits in the folder indefinitely.

### The queue index (removed)

There used to be a SQLite list of the videos waiting in the triage queue, at `{NAS}/review_records/triage_queue.db` (`review/queue_index.py`). The bring-current routine added a row when it staged a bundle and the review window removed one on every *Save & Next*, but nothing had read the list since the window started listing the queue folder itself (see "How a video is chosen"). It has been removed: no tool writes or reads it.

WHY it was removed rather than left alone: a SQLite database in WAL mode on a shared network drive, written by more than one machine, can lock or corrupt, and it did nothing. What this means for an operator: nothing changes in how videos are offered. A `triage_queue.db` file already on the shared drive is now inert. Keep it (move it to an archive folder if it is in the way); it is never deleted, and nothing will recreate it.

### The session flag button (FIXED 2026-08-28 -- it used to do the opposite of what it says)

**Now:** pressing "Flag Session (needs review)" saves the current review, records the
session in `Processing/Review/flagged_sessions.json` as provenance, and **routes the
video to the re-segmentation queue** via `route_to_queue`, then advances to the next video.
The flag is a note about the session; the divert is the action. Nothing filters on the
flag any more: the last place that did, the queue-index seed (`seed_from_folder`), used to
skip every video of a flagged session and no longer does.

**What Flag Session does NOT do.** Despite its name, it moves only the video on screen. The
same mouse's other videos from that day are not moved and not held: those still in triage stay
there and are offered as usual, and those already filed stay filed. If they need the same look,
open each one (or route it with `mousereach-route-to-queue`). Whether the button should also
send or hold the session's other videos, or be renamed to what it does, is an open question
for the lab.

**Flagging cancels any earlier re-segmentation clear, exactly as Escalate does.** Before the
video moves, an old `{stem}_deep_review_cleared.json` from an earlier visit is renamed to
`{stem}_deep_review_cleared_superseded_<time>.json` (kept, never deleted). Otherwise the
watcher would read the old clear as "done" and release the video straight back on the same
cuts. If the old clear cannot be renamed (a file in use), the video is NOT moved and an
error says so -- press the button again once the file is free. In the re-segmentation queue
a flagged video waits until its cuts are set or confirmed in the re-segmentation tool, which
lists it (see "Every video that needs this is listed in the re-segmentation tool").

**Flag Session on a video that is already in the re-segmentation queue does not move it.** If the
window is showing a bundle that sits in the re-segmentation queue (whether or not the window was
opened in re-segmentation mode), the button records the session flag and sets aside any earlier
clear as above, and then says "<video> is already in the re-segmentation queue, so it stays
there". Nothing is moved and nothing in the bundle changes apart from the renamed clear; the video
waits until its cuts are set again. **Why:** the button used to route the bundle into the queue it
was already in, and that deleted its files (see "Moving a video into or out of a queue never
deletes or overwrites a person's file").

**What it used to do.** The button is labelled "Flag Session (needs review)", its tooltip
said it marks every video of that mouse+day as must-be-human-reviewed, and the toast said
"all its videos need human review". `flag_session` wrote the session key to
`flagged_sessions.json` and that was the entire effect. Two pieces of code read that file:
`_bundle_needs_review` and the never-called `seed_from_folder` (`queue_index.py`, since removed with the queue index).
The first one did this:

```python
if is_session_flagged(stem, root):
    return False   # this bundle does NOT need review
```

So flagging a session **removed** every one of that mouse+day's unreviewed videos from the
queue, and nothing in the gate, the return scan, kinematics or the database read the flag
to compensate. There was no path by which a flagged session got more attention.

Before the fix, an operator flagging sessions while working through the queue could make
most reviewable bundles invisible: the queue read as empty and the flag was the cause.
Removing the filter put those videos back in front of the reviewer.

### Automatic release from ground truth

`mousereach-resolve-triage-from-gt` (`triage/gt_resolve.py`) lifts triage flags on segments a ground-truth file already answers, writing the same clearing fields the clearing tool writes. The comment above it in `pyproject.toml:120-123` says it "runs as a step in the normal processing pipeline". It does not — the only caller in the tree is the improvement evaluation harness (`improvement/eval_all.py:163-167`).

---

### A video taken out of a queue to be posed again (`mousereach-repose-videos`, since 2026-11-09)

A video can leave the triage or deep-review queue WITHOUT anyone answering it: when its pose is
not trusted and it has to be posed again on a graphics-card machine (see PIPELINE_AS_BUILT.md,
"Update 2026-11-09"). Its questions were asked about a pose that is being thrown away, so they
are withdrawn with it; the new pose gets its own run, its own gate and, if needed, new questions.

**What a reviewer sees.** The bundle disappears from the queue in one step (the whole folder is renamed out; it is never seen half-emptied). Its video and the queue's notes
move to `Processing/Repose_Queue/_parked/<video>/`; its pose and the algorithm's outputs move to
`Analyzed/Archive/<label>/`. Nothing is deleted.

**A person's work is never taken out this way.** A held video with a saved review
(`_causal_review.json` in the bundle or in `review_records/reviews/`), a deep-review clearance or
ground truth is refused and left in its queue; the command lists it, and a person decides. A
FILED video keeps its review beside it, and the re-run applies it again (segments matched by
frames). The causal reach is re-applied by its reach number from the old reaches file, so after
a re-pose that renumbers reaches it can point at a different reach: compare the reach's frames
after the re-run.

When the re-posed video comes back with questions, it re-enters the queue as an ordinary bundle, video included (`_reprocess_copy_set` takes the video from the posed folder; before that fix the bundle had none).

## What importing the review package loads (since 2026-10-24)

`import mousereach.review` loads only the light parts: the review-file I/O (`causal_review_io`, standard library only). The GUI names it offers -- the three widgets, `ComparisonPanel` and friends, `AlgoGTReviewMixin` -- are imported the first time something uses them (`_LAZY` and `__getattr__` in `review/__init__.py`), so `from mousereach.review import CausalReviewWidget` works exactly as before. Why: the watcher imports this package for `causal_review_io` and `triage_status`, and while the widgets were imported eagerly, a headless watcher loaded napari, Qt, vispy, pandas and scipy (about 2,600 modules, 4.6 s) the first time a cycle touched review code -- and a widget that failed to import took the watcher's review paths down with it. A new GUI module re-exported from the package goes in `_LAZY`, not in a plain import (`tests/test_review_package_loads_gui_lazily.py` fails otherwise).

## Declared but does nothing

Each of these is code that runs, produces a value, and has no reader.

- ~~**Ignore windows / `abnormal_ranges`.**~~ Read by the kinematics since extractor 2.2.0 (2026-10-30): a reach that overlaps a window by any frame is left out of every measurement, and the features file records it under the segment's `reaches_left_out` (see `docs/KINEMATICS_FIELDS.md` §7). Until then the windows were saved, carried onto the segment, and read by nothing.
- **`outcomes.exhaustive` in ground truth.** `_seg_overrides_from_gt` computes it and returns it; the caller assigns it to `_exhaustive_out` and never uses it (`truth_resolver.py:328`). Ground-truth outcome overrides apply per determined segment either way, so the flag changes nothing for outcomes. For *reaches*, `exhaustive` is load-bearing and does drop unlabelled algorithm reaches (`_apply_gt_reaches`, `:275-279`).
- **`TriageStatus.clean`** (`triage_status.py:152-156`) — a property with no callers. The gate and the return scan read `seg_failed`, `unresolved` and `fully_resolved` directly.
- **`TriageStatus.seg_needs_human`** (`:137`) — populated from the segmentation file (`:184`) and deliberately not acted on (`review_gate.py:89-105`).
- **`is_phantom`** in every review record: always `False`, never set.
- **Boundary segments.** `is_boundary` is hard-coded `False` (`causal_review_widget.py:1043`) and set nowhere else, so the three branches that handle a boundary segment (`:1236`, `:1284`, `:2281`) never run and the guard at `:1749` is always true.
- **`causal_review_index.json`.** Written on every save (`causal_review_io.py:690`, path at `:729`) so that, per the module docstring, the active-learning loop can bulk-read all reviews. Nothing reads it — no reader exists in `src/` or `scripts/`.
- **`base.py`** — `AlgoGTReviewMixin`, `DiffItem`, `DiffSummary`. Re-exported by `review/__init__.py` (loaded on first use, like every GUI name there) and documented in this folder's `AGENTS.md` as the way to build a review widget. No class in the codebase uses the mixin and no code constructs a `DiffSummary`.
- **`SavePanel`** in `save_panel.py:24` — never instantiated. Its sibling `SimpleSavePanel` (`:217`) is what the three per-step review widgets use.
- **`unified_widget.UnifiedReviewWidget`** — reachable only from the napari plugin menu as "Review Tool (Tabbed)" (`napari.yaml:28-30`) and from its own `main` (`unified_widget.py:1123`). Its docstring claims the command `mousereach-review-tool` starts it (`:17`); that command starts the causal review tool. Its three tabs are the per-step review screens, which still save (see "Review Tool (retired) and the per-step review screens").

### Not on this list, though a previous draft said so

**`exclude_from_analysis` on reaches has real readers.** The retired triage-clearing tool marked every non-causal reach in a cleared segment with it (files it already edited keep the mark) and the ground truth tool sets it (`ground_truth_widget.py:1784`). What is true is narrower: the **kinematics feature extractor** does not read the flag in `_reaches.json` (a flag the legacy in-place tools write there is lost on the next run anyway), the flag is not carried into `{video}_features.json`, and it is not a column in `connectome.db`. The ground-truth tool's exclusion does reach the kinematics: the truth layering takes every detected reach it overlaps out of the measurements and the features file lists it under `reaches_left_out` (`truth_resolver._drop_excluded_reaches`). It **is** read by the analysis dataframe, which drops flagged reaches by default (`analysis/data.py:120-127`, `:167-168`, used by `load_all_data`, `:432-497`) — though on that module's default path the rows come from the features file, where the field is absent, so the filter has nothing to act on; it bites only when loading from `_reaches.json` (`:343`). It is also read by the truth resolver when merging ground-truth reaches (`truth_resolver.py:255`) and throughout the improvement metrics (`improvement/reach_detection/metrics.py:482`, `:518`, `:926`; `kinematic_damage.py:105`, `:173`; `_compute_baseline.py:58`, `:109`). The ODC-SCI export carries it as a column (`kinematics/analysis/odc_sci_exporter.py:157`).

---

## Which answer wins, where a review is applied at all

`resolve_truth_layers` (`truth_resolver.py:289`) builds layers low to high and merges them (`_apply_outcome_layers`, `:153-183`):

```
ground truth  >  human review  >  algorithm
```

**The human-review layer is ONE read of every copy of the video's review** -- the file or folder the caller hands it (the processing folder's review), the triage and re-segmentation queue bundles, the video's filed folder and the durable store -- merged per segment to the newest answer, or recheck that reopened one, a person gave about that segment's frames (`causal_review_io.effective_review` with `review_dirs`). It is the same read the review window, its "already reviewed" check, the release check and the census use, so the results step applies exactly the answers the review tools show as given. Why: the copies used to be layered by where they sat (processing folder over re-segmentation bundle over triage bundle over durable copy), and the filed folder's copy was not read at all. An answer that lived only in a copy this step did not prefer was not applied, an older answer in a queue bundle beat a newer one elsewhere, and a pellet reopened with "are you sure?" in one copy was still applied from another.

Two things to be precise about:

- **The unit is a segment, not an element.** Within the per-segment outcome layers, a higher layer replaces the whole override for that segment — outcome, causal reach id and interaction frame together (`:160-162`, `:172-180`). Only a segment that no layer covers keeps the algorithm's values. The separate per-reach layer (ground truth only) is merged reach by reach (`_apply_gt_reaches`, `:249`).
- **Every review answer is stamped the same.** Triage-queue and re-segmentation-queue answers are both labelled `"human_review"`; the data does not record which tool produced the answer.

Ground truth only overrides segments it marks determined (`:138-139`). Provenance is stamped per segment as `outcome_source` (`algo` / `human_review` / `ground_truth`) and per reach as `reach_source` (`algo` / `ground_truth` / `human_review`), with the algorithm's originals preserved as `algo_outcome` and `algo_causal_reach_id` (`:172-176`). Reviews never change the boundaries of a detected reach. A review adds a reach only when the reviewer typed one the detector missed (`_add_reviewed_reaches`, since 2026-10-26; see "A reach the list does not have" above). Ground truth changes boundaries and existence, and only where `reaches.exhaustive` is set does it also delete algorithm reaches it never labelled (`:275-279`). Only real answers are applied (`is_answered`): until 2026-10-26 this step applied a pellet left on "-- select outcome --" as a human answer of "triaged".

**Ground truth is matched by frames, like a review answer.** Each ground-truth outcome is about the frames of its segment in the ground truth's OWN segmentation (`segmentation.boundaries` in the file); it applies to, and counts as answering, the current segment those frames fall in (`causal_review_io.index_gt_outcomes_by_segment`, used by the results step and by `triage_status.gt_resolved_segments`, which the gate, the return scan and `mousereach-route-to-queue` use). An outcome whose frames match no current segment, or split across two, is not applied and is logged. A ground-truth file with no segmentation of its own is matched by number, the only handle it has. Why: ground truth was applied by segment number, so after a re-cut renumbered the segments its outcome would land on a different pellet's footage. Ground-truth REACHES are placed the same way: each goes to the current segment holding most of its frames (`_apply_gt_reaches`), and the segment number it was labelled with is used only when the current segments carry no frames or none holds it. Where the reach set is marked exhaustive, a current segment's reaches are replaced by the ground-truth reaches lying in it.

The processing-dir layer is the one that does not work — see [Reviews mostly do not reach the numbers](#reviews-mostly-do-not-reach-the-numbers).

### Where a bundle's video and pose are found (since 2026-10-30)

A bundle that carries no copy of its video records where the video was filed,
as an absolute path, when the bundle is made. That path goes stale when the
finished-work folders are reorganised (for example when each project's folder
is moved inside a grant folder). The review window therefore looks, in
order: the path the bundle recorded; a copy inside the bundle folder; the
folder the video is filed in NOW, by the lab's filing rules (and, for the
pose, the same-named folder of each `DLC Model <N>` tree beside it)
(`review/bundle_media.py`, `filed_media_dirs`). The queue index and the
review window's "already answered" check look in that filed folder too, for a
review saved beside the video. Nothing in the bundle is rewritten.

### Where a saved review is looked for

Two different lookups, with different rules:

- `resolve_review_path` (`causal_review_io.py:55`) returns the **file**, most recently modified wins, checking an explicit directory the caller passes (usually the processing directory), then the triage queue's bundle for that video, then the directory holding the canonical video, then the durable copy. It never raises. It now only names WHICH file the results step records in the run's manifest (so a later change to that file re-runs the kinematics); **no check decides what is answered from it any more.**
- `truth_resolver` (the results step) reads every copy through `effective_review`: the folder or file it is handed, the run's own folder, the two queue bundles, the filed folder and the durable copy.
- The review window, the bulk release check and the census read EVERY copy they know of -- the bundle's, the one beside the video (window only), and the durable copy -- and take, per segment, the newest answer or recheck a person gave about its frames (`causal_review_io.review_copies` + `merge_review_copies`). The window's restore, its "is this video already reviewed?" check (which decides whether the window offers a video at all; `bundle_review_complete`) and the bulk release check all go through one function, `causal_review_io.effective_review`, so they cannot disagree about which pellets are answered. In particular a pellet reopened with "are you sure?" in the durable copy keeps the video on offer, and an answer only the durable copy holds counts. The save reads the same copies before writing and keeps any earlier answer the save itself does not replace (`carry_earlier_answers`).
- **The pipeline side reads the same way.** The review gate and the return scan (`triage_status`), the hand-score check, and `mousereach-route-to-queue` (which never re-asks an answered pellet) all read every copy merged through `effective_review`. If a copy exists but cannot be read, `mousereach-route-to-queue` refuses the video and moves nothing, saying "its review or cuts could not be read": the unreadable copy may hold the only answer, and routing around it would ask the person again. Fix or restore that copy, then route again. An "are you sure?" reopen decides on every copy merged and writes the reopened pellet into **both** the copy in the video's folder (created if missing) and the durable copy, each backed up first to `Analyzed/Archive/manual_corrections/` under its own name (`.travelling` / `.durable` at the end). Records in which a person said a segment is really another pellet are read from every copy too (`mislabel_records`): the merged review leaves them out when the pellet was left open.

---

## Documentation that is wrong at this commit

Stated here so nobody rediscovers it the hard way:

- `unified_widget.py:17` claims `mousereach-review-tool`.
- `causal_review_widget.py:2509-2510` describes `_review_root()` as `Processing/Review/triage`. It returns the parent of the queue folder, `Processing/Review`, which is where `flagged_sessions.json` actually lives.
- `pyproject.toml:120-123` says the ground-truth triage resolver runs as a pipeline step. It does not.

---

## Configuration summary

| Setting | Where | Effect |
|---|---|---|
| `nas_root` | `~/.mousereach/config.json` | Root of both queues. Unset means the queues are `None` and the tools say "not configured" (no guessed folder since 2026-10-22). |
| `CONNECTOME_ROOT` | environment | Base for the two lookups above; no default -- unset means both lookups are skipped (since 2026-08-28). |
| `--pending-dir` | `mousereach-review-tool` | Which queue folder to review. Defaults to the triage queue (`causal_review_widget.py:3123`). |
| `--all-segments` | `mousereach-review-tool` | Walk every segment instead of only triaged ones (`:3126`). |
| `--worklist FILE` | `mousereach-review-tool` | CSV or JSON of `vid` + `segment_num`. Only those videos are offered and only those segments walked — and unlike the normal pool, already-reviewed and ground-truthed videos are included (`:2709-2718`). |
| `--cv` | `mousereach-review-tool` | Use pixel-based pellet localisation to narrow candidate reaches. Off by default because it decodes the video over the network; without it the narrowing uses tracking data only (`_get_cv_states`, `:1713`, gated at `:1727`). |
| `--queue-dir` | `mousereach-fix-segmentation` | Queue of bundles to work through; defaults to the re-segmentation queue (`fix_segmentation_widget.py:482-490`). |
| `--pre-pad` / `--post-pad` | `mousereach-gt` | Frames of context loaded around the segment. Default 0 for the ground truth tool (`ground_truth_widget.py:3037-3040`). |
| `--algo-dir` | `mousereach-gt` | Where to read a segment's decision window and which info panel to show. Does not change where a save writes. |
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
