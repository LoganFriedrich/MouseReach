# The pipeline process, as intended

This file is the **reference description**: how the system is supposed to work, as stated by
its designer. It must not be "corrected" to match observed behaviour. When the code and this
file disagree, that is a finding to be reported, not a discrepancy to be tidied away.

`PIPELINE_AS_BUILT.md` documents how the code actually behaves today.

---

## Summary

Human reviews made with the GT, causal review or triage review tools are facts about the
frame ranges they describe in a video. They stay true whatever algorithm or DLC version is
current, so they are stored as facts about the video itself and are never discarded.

Every video moves along one recorded process. People record collage videos, name them, and
place them in the pipeline's intake folder. A GPU node runs the crop/DLC watcher (watcher 1):
it crops each collage into single-mouse videos, named from the collage's grid positions and
the animal IDs in its file name, moves the singles to Processing (the parent collage stays in
the intake folder, because it is not fully processed yet), and runs the current DLC model on
them. That is all watcher 1 does.

The processing node runs a second watcher. For each single in Processing it checks that the
video is not being worked by watcher 1, that a DLC file exists, and that the DLC file is the
current version. An outdated video has its old DLC, algorithm and kinematics files archived
and goes back to Processing to be reprocessed; its human review files move with it. Otherwise
the watcher runs the MouseReach algorithms in order. Any algorithm failure or issue sends the
video to triage, where it stays until the watcher sees that a human has updated the triage
file (this can take several rounds per video). A video that makes it through every algorithm
with no outstanding triage is placed, with all its files, into the finished folder organised
by project and cohort, and only then are its kinematic results moved into the database, where
new elements replace old ones only after the old ones have been archived.

How each part works is to be written down in plain-English documentation and kept current
whenever behaviour changes.

---

## The claims this makes, enumerated for checking

Each can be checked against `PIPELINE_AS_BUILT.md`.

| # | Claim |
|---|---|
| D1 | Collages are recorded (e.g. with OBS) and placed in `<nas_root>/Unanalyzed` |
| D2 | Exactly two watchers exist |
| D3 | Watcher 1 runs on a CUDA GPU node |
| D4 | Watcher 1 crops collages into single-mouse videos |
| D5 | Naming is by mouse, from video coordinates + string positions in the collage title |
| D6 | Cropped singles are relocated to Processing |
| D7 | The parent collage STAYS in Unanalyzed (not fully processed yet) |
| D8 | Watcher 1 then runs the current DLC model on the singles in Processing |
| D9 | Cropping and DLC are ALL watcher 1 does |
| D10 | Watcher 2 runs on the processing node, over singles in Processing and Processed |
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
