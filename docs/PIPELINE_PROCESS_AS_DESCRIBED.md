# The pipeline process, as described by Logan — 2026-08-20

This file is the **reference description**: how the person who built the system says it is
supposed to work. It is recorded verbatim and must not be edited, summarised, or "corrected"
to match observed behaviour. When the code and this file disagree, that is a finding to be
reported, not a discrepancy to be tidied away.

`PIPELINE_AS_BUILT.md` documents how the code actually behaves today.
`PIPELINE_DESCRIBED_VS_BUILT.md` records where the two differ.

---

## Verbatim

> Reviewing of the kind that we are doing should be performed with the causal revew tool. The
> causal review tool is one of two deep review tool options. There is already a deep review
> folder. Additionally, there should already be a process that all videos are to move along and
> that process should be reocrded in project documentation. Human reviews conduted with the GT,
> causal review, or triage review tools are facts about the particular claims they make that
> will always be true no matter which algo versions or DLC versions etc. Therefore, they should
> be stated as facts about the video itself and I believe they already are.
>
> Here's an example of how this works in reference to the triage review tool: Humans record
> video collages with OBS. They then save, rename, and relocate these collage videos to
> "mousereach_pipeline/unprocessed" on Y. A PC that has a compadible GPU (a "DLC" PC, like the
> lab PC for example) runs a very specific watcher: one of the two watchers that exists, namely
> the watcher that runs a script that crops collages into single mouse videos and renames hem
> according to which mouse is in the video which is determined by video coordinates and string
> locations for the title of the collage video. The cropping/renaming process relocates the
> single animal videos to the processing folder, leaving the parent collage in the unprocessed
> folder because it truely has not been completely processed yet. The watcher then runs
> deeplabcut (DLC) with model 4 on the single mouse videos in processing. This determines where
> objects that we are interested in are located in space (with confidences about their certainty
> because it uses a dlCNN (deeplabcut) to do this) throughout each video. That is all that the
> watcher on the DLC PC does. The server (this PC that you are running on) runs another
> different watcher which takes single mouse videos that are in processing (and processed) and
> looks to see: 1. If the video has a finished DLC output file (that it is not processing with
> "the crop/DLC watcher (watcher 1)" right now. 2. That it does have a DLC file (since those are
> needed for MouseReach algos to run). 3. That the DLC file it does have is the most recent
> version (else it kicks the old version (wrong model) DLC files, algo files, and kinematics
> files to an archive and moves the single animal video back into processing to be reprocessed
> (note that any human reviewing like GT or causal review or triage review files are NOT disposed
> of - they instead should be moving with the single mouse video back to processing, since they
> are always true no matter what algo or DLC versions are happening since they are facts about
> reference frame ranges in that video which are correct independent of dlc and algo versions).
> Once that check is complete on that video, it either does the aformentioned archiving and
> rearranging of the associated files or it runs the mousereach algos in order on the files. If
> any algo fails or has an issue, the video is sent to triage where it remains until the watcher
> notices that that triaged element has been reviewed by a human (that the triage file has been
> updated). Triage is complicated as should be discussed later if needed but you should be able
> to just read the scipts to understand how that process does work now and how that process does
> work now should also be recorded as a plain english .md file somewhere, as should be the case
> about how everything in the system works and those documentation files should be getting
> updated any time how something works changes so that the next session can come along and just
> read how it works - which there should be clear instructions to do by the way. Anyhow. If a
> video makes it all the way through all of the algorithems of MouseReach successfully with no
> triaging (or after it has had all of its problematic elements been reviewed, which can
> reasonably take multiple round per video sometimes) then the video and all associated files are
> placed into processed and are organized according to project and cohort and then and only then
> are the kinematic results moved into mousedb where the new elements replace the old elements
> only after the old elements have been archived.

---

## The claims this makes, enumerated for checking

Each is checked in `PIPELINE_DESCRIBED_VS_BUILT.md`.

| # | Claim |
|---|---|
| D1 | Collages are recorded with OBS and placed in `MouseReach_Pipeline/Unanalyzed` on Y: |
| D2 | Exactly two watchers exist |
| D3 | Watcher 1 runs on a CUDA "DLC PC" |
| D4 | Watcher 1 crops collages into single-mouse videos |
| D5 | Naming is by mouse, from video coordinates + string positions in the collage title |
| D6 | Cropped singles are relocated to Processing |
| D7 | The parent collage STAYS in Unanalyzed (not fully processed yet) |
| D8 | Watcher 1 then runs DLC Model 4 on the singles in Processing |
| D9 | Cropping and DLC are ALL watcher 1 does |
| D10 | Watcher 2 runs on the server, over singles in Processing and Processed |
| D11 | Watcher 2 checks the video is not currently being worked by watcher 1 |
| D12 | Watcher 2 checks a DLC file exists |
| D13 | Watcher 2 checks the DLC file is the current version |
| D14 | If outdated: old DLC + algo + kinematics files are ARCHIVED |
| D15 | If outdated: the video moves back to Processing to be reprocessed |
| D16 | Human review files (GT / causal / triage) are NEVER discarded on reprocessing |
| D17 | Human review files MOVE WITH the video back to Processing |
| D18 | Otherwise watcher 2 runs the MouseReach algos in order |
| D19 | Any algo failure or issue sends the video to triage |
| D20 | The video stays in triage until the watcher sees a human updated the triage file |
| D21 | On full success with no outstanding triage, the video AND all associated files move to Processed |
| D22 | Processed is organised by project and cohort |
| D23 | Only THEN do kinematic results move into mousedb |
| D24 | In mousedb, new elements replace old ones only AFTER the old ones are archived |
