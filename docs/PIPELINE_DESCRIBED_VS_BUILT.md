# Where the pipeline does not do what it is supposed to

Describes: src/mousereach/watcher, src/mousereach/archive, src/mousereach/sync
Verified against: 61d98b9 (2026-08-21)

The intended process is recorded verbatim in `PIPELINE_PROCESS_AS_DESCRIBED.md`
and broken into 24 numbered requirements. This document checks each one against
the code as it stands on 2026-08-21.

**Read these as defects in the code, not as errors in the description.** The
description is the specification. Where the two disagree, the code is what needs
to change -- or the specification needs to be revised deliberately, in writing.

**How this was checked.** Each group of requirements was traced against the
source by a separate reviewer who had to cite file and line for every statement.
Every reported gap was then given to a second reviewer whose only instruction was
to disprove it, defaulting to "not a real gap" when uncertain. Six reported gaps
did not survive that and are listed at the end. The 11 below did.

---

## Summary

| | count |
|---|---|
| Requirements met as written | 12 |
| Met only partly | 7 |
| Not met | 5 |

---

## 1. Results are destroyed, not kept

These two break the rule that nothing is thrown away. Both lose work permanently, and neither announces it.

### D14 -- NOT MET

**Supposed to:** If outdated: old DLC + algo + kinematics files are ARCHIVED

**What the code does:**

In the running watcher, nothing is archived. When a video is marked outdated and re-run, the sequence is: copy every file whose name starts with the video identifier from the Analyzed output folder down to the local Processing folder (a copy - the originals stay put), re-run the algorithms so new outputs overwrite the copies in Processing, then move the whole set back to Analyzed. That final move is done by archive_video, which calls Python's shutil.move for each file onto the same filename in the destination. When a file of that name already exists there, shutil.move silently replaces it. So the old algorithm outputs and the old kinematics file ('<video id>_features.json') are destroyed by overwrite, not moved aside. The old pose file is a different case: it is never moved or deleted at all. If the model changed, the new pose has a different filename (the model name is embedded in it), so both generations end up sitting next to each other in the same folder, and select_pose_file decides which one gets used.

If the reason for the reprocess is a changed pose model, the second watcher does not even do the copy-down: it forces the video into the state 'dlc_queued' and returns, leaving the graphics-card machine to pose it again. That path writes the new pose beside the video and archives nothing.

Be careful with the word 'archive' when reading this code. In the watcher, 'archive' means the final step that moves a finished video's outputs out of the working Processing folder into Analyzed/<project>/<cohort> - the live, current results location, not a historical store.

Code that does exactly what the claim describes does exist, but nothing automatic calls it. archive/supersede.py moves a video's about-to-be-replaced pose and algorithm outputs into a separate Archive tree under the pipeline root, laid out as 'Archive/DLC Model <generation>/<algorithm version stack>/', using checksum-verified moves (copy, verify a sha256 hash, only then delete the source), never overwriting anything already there (identical file - drop the incoming copy; same name but different content - save it as .1, .2 and so on), and explicitly leaving the video file, the ground-truth file and the human review file in place. Its only caller in the entire package is pipeline/reprocess_to_current.py, a manually driven 'bring one video up to the current stack' tool with no command-line entry point in pyproject.toml and no caller in the watcher, in scripts, or anywhere else in the source tree.

**Why it matters:** This is a real data-loss gap, not a naming quibble. The description's guarantee is that a reprocess is non-destructive. In the automatic pipeline it is destructive: the previous generation's segments, reaches, outcomes, assignments, triage record, processing manifest and kinematics file are overwritten in place with no copy kept, so there is no way afterwards to see what the earlier model or earlier algorithm version produced for that video, or to reproduce a figure made from it. The one thing that does survive is the old pose file, and only by accident of its filename. The correct, checksum-verified archiving code is written and tested-looking but is only reachable by hand.

Evidence: `src/mousereach/watcher/orchestrator.py:1574`, `src/mousereach/watcher/orchestrator.py:1577`, `src/mousereach/watcher/orchestrator.py:1495`, `src/mousereach/watcher/orchestrator.py:1497`, `src/mousereach/watcher/orchestrator.py:2122`, `src/mousereach/watcher/orchestrator.py:2132`, `src/mousereach/archive/core.py:174`, `src/mousereach/archive/core.py:183`

### D24 -- NOT MET

**Supposed to:** In mousedb, new elements replace old ones only AFTER the old ones are archived

**What the code does:**

The database write is a plain delete-then-insert with no archiving step of any kind. For each video, the syncer opens connectome.db (Y:/2_Connectome/Databases/connectome.db, sync/database.py:41), runs DELETE FROM reach_data WHERE video_name = :video_name, then inserts the new rows and commits (sync/database.py:613-631). The old rows are not copied anywhere first, there is no history or archive table, and the table-creation code adds no triggers that would preserve them (sync/database.py:372-410). Nothing else in the repository copies reach_data rows aside before a sync. The flat CSV dump written after every sync is likewise overwritten in place, not versioned (sync/database.py:746-795). The only 'archive the old before writing the new' machinery that exists is at the FILE level, not the database level: archive/supersede.py moves a video's previous algorithm outputs into a version-stamped Archive tree with checksum verification before the new ones are put in place, and it deliberately never archives the human ground-truth or human-review files or the video itself (archive/supersede.py:1-35, :161). That file-level supersede is called from exactly one place - the bring-current reprocessing tool (pipeline/reprocess_to_current.py:293-294) - and never from the watcher. Separately, a mirror of the Y: drive is copied to the X: drive by a periodic robocopy job in add-only mode (watcher/backup.py:28-33, :133-150); that copies the whole connectome.db file, so it preserves a possibly-stale snapshot of the database, but it is a disk backup and not a record of what a given sync replaced.

**Why it matters:** When a video is reprocessed with a newer algorithm version, its previous per-reach rows are destroyed. There is no way to query what the numbers used to be, no way to compare an old analysis against a new one inside the database, and nothing to roll back to if a bad algorithm version syncs a whole cohort. The files behind the old rows may survive - but only for videos that went through the bring-current tool, which is the one path that calls the file-level archiver.

Evidence: `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\sync\database.py:41`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\sync\database.py:613`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\sync\database.py:623`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\sync\database.py:372`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\sync\database.py:746`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\archive\supersede.py:161`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\pipeline\reprocess_to_current.py:293`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\backup.py:133`

---

## 2. A failure does not reach a human

The pipeline is supposed to stop and ask when something goes wrong. In two cases it does not stop, and in several it stops somewhere nobody is looking.

### D19 -- NOT MET

**Supposed to:** Any algo failure or issue sends the video to triage

**What the code does:**

An algorithm failing is not what sends a video to triage. Triage is reserved for videos where the algorithms SUCCEEDED but their output contains a question a human has to answer. Failures go to three other places, and some are ignored altogether.

What actually happens per failure:

- Segmentation throws an error, or the pose file cannot be read at all: the video's state in the watcher database is set to 'failed', with an error message. Nothing moves anywhere. No watcher ever picks a 'failed' video up again; a person has to run the command mousereach-watch-reprocess to reset it.
- Segmentation runs but reports it could not produce trustworthy boundaries (for example an over-long recording): the whole bundle of files is moved to the DEEP REVIEW queue (the folder Processing/Review/flagged_for_review on the NAS), not the triage queue.
- Reach detection throws: state 'failed', and the error is re-raised. Same dead end.
- Outcome detection throws: state 'failed', re-raised. Same dead end.
- Reach assignment throws: a warning is written to the log and the pipeline CONTINUES. Worse, if no assignment file was produced, the later check that would have caught "this pellet was touched but no reach was credited for it" is skipped, so such a video can pass the gate as clean and have its kinematics written to the database with no causal-reach attribution at all.
- The automatic quality check itself throwing: warning only; the verdict defaults to 'auto approved' and the video proceeds.
- The automatic quality check returning a critical verdict: DEEP REVIEW, not triage.
- Kinematic feature extraction throwing: warning only. The video is still marked 'processed' and goes on to be archived - it just has no kinematics and nothing was written to connectome.db.
- The database write failing: warning only.

The only route into the triage queue is the gate finding at least one segment that the outcome cascade marked 'triaged' or flagged for review, or a segment whose pellet was touched but which has no committed causal reach - and that no human answer already covers.

**Why it matters:** Three practical consequences. First, videos whose algorithms crashed are NOT waiting in a review folder for a person - they sit in the watcher database in state 'failed' and are invisible to anyone who only looks at the review queues; nothing retries them automatically. Second, a genuine segmentation failure lands in the DEEP REVIEW folder, which needs the deep tools and a different clearing action than triage. Third, and most serious scientifically, two failures are swallowed silently: reach assignment failing lets a video reach kinematics with no causal-reach attribution, and kinematic extraction failing still marks the video 'processed' and archives it, so it looks finished while carrying no kinematic results.

Evidence: `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:1903`, `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:1881`, `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:1892`, `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:1933`, `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:1965`, `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:1986`, `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:2039`, `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:2105`

---

## 3. Results reach the database before they are filed

The intended order exists to guarantee that every number in the database has a permanently filed set of files behind it. That guarantee does not hold.

### D23 -- PARTLY MET

**Supposed to:** Only THEN do kinematic results move into mousedb

**What the code does:**

In the watcher - the automated pipeline the description is about - the order is the reverse of what is claimed. Once the review gate returns 'clean', the code immediately extracts the kinematic features, writes {video}_features.json into the working folder, and pushes it into connectome.db from that working folder, before anything has moved anywhere (server path: gate at watcher/orchestrator.py:2047, database push at watcher/orchestrator.py:2095-2096; GPU-machine path with watcher.also_process = true: gate at watcher/orchestrator.py:1132, database push at watcher/orchestrator.py:1167-1168). Only after that does the video get marked 'processed' (watcher/orchestrator.py:2110 / :1174), and only on a later pass of the work loop is the 'move to Analyzed' job picked up and run (watcher/orchestrator.py:1476-1487 / :512-525). So the numbers are in the database first and the files move afterwards - and, per D21, on the processing server the move may never happen at all. There is one code path that does behave exactly as claimed: the 'bring current' reprocessing tool moves the finished files into the video's folder under Analyzed first and pushes to the database only after confirming the features file actually landed there (pipeline/reprocess_to_current.py:293-320, with the database call at :312-317 guarded by 'features.json in moved').

**Why it matters:** The claimed ordering is a safety property: it means every row in the database has a permanently filed set of source files behind it. The watcher does not provide that property. A row can exist in connectome.db whose source files are still sitting in a local working folder, or were never filed because the archive step kept failing its readiness check. Provenance columns in the row point at the working-folder path, not the final one. The one place the claimed order does hold is the bring-current tool, which is also the only place that archives the previous version of the files before replacing them.

Evidence: `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:2047`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:2095`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:2110`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:2133`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:1132`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:1167`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:1174`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:1204`

### D21 -- PARTLY MET

**Supposed to:** On full success with no outstanding triage, the video AND all associated files move to Processed

**What the code does:**

A video that clears the human-review gate does eventually get its whole file set moved to the final folder, and the move does take everything - the code collects every file in the working directory whose name starts with the video id (the .mp4, the DeepLabCut pose files, every algorithm .json, the manifest, the triage file, and any ground-truth or human-review .json sitting alongside) and moves them together (archive/core.py:146, archive/core.py:148, archive/core.py:183). Three things do not match the claim as worded. First, the folder is not called 'Processed'. There is no folder of that name anywhere in the path settings; the destination is called 'Analyzed' (config.py:123). 'processed' in this system is a state label in the watcher's own tracking database, not a folder (watcher/orchestrator.py:1174, watcher/orchestrator.py:2110). Second, the move is not part of the success path - it is a separate job the watcher picks up on some later pass round its loop, after the video has already been marked 'processed' and after its numbers have already gone into the database (watcher/orchestrator.py:512-525 on the GPU machine, watcher/orchestrator.py:1476-1487 on the server). Third, whether the move actually happens differs by machine role. On a GPU machine configured with watcher.also_process = true, the move is forced through with the readiness check switched off, so it reliably happens (watcher/orchestrator.py:1204-1208). On a machine configured with watcher.mode = 'processing_server', the move is attempted with the readiness check switched ON (watcher/orchestrator.py:2133), and that check requires a cached file index to say that segmentation, reach detection and outcome detection are each 'validated' or 'auto_approved' (archive/core.py:56-79, index/index.py:426-441). Nothing in the watcher daemon writes those entries: segmentation's single-video entry point never touches the index (segmentation/core/batch.py:65-115), the current outcome detector writes its .json directly without touching the index (outcomes/core/batch.py:209-211), and reach detection writes 'needs_review' into the index by default (reach/core/reach_detector.py:1104, reach/core/batch.py:168). The index is only refreshed when a person launches the napari front end or the dashboard (launcher.py:216-228). So on the server, the move to Analyzed can fail with 'Not ready: ... not validated' and simply be retried forever (watcher/orchestrator.py:2160-2165 logs it as a non-fatal retry), while the video's numbers are already in the database.

**Why it matters:** Two practical consequences. The name difference matters because someone looking for a 'Processed' folder will not find one, and because 'processed' is also a database state that means something different (algorithms finished, files not yet moved). The role difference matters more: on the processing server the final move is gated on bookkeeping the server itself never writes, so a video can sit in the local working folder indefinitely with its results already in the database and no error raised - only a repeated 'will retry' log line.

Evidence: `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\config.py:123`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\archive\core.py:146`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\archive\core.py:148`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\archive\core.py:183`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\archive\core.py:56`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\archive\core.py:134`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:1174`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:1204`

---

## 4. Work happens on a different machine than described

One setting, watcher.also_process, moves most of the pipeline onto the graphics-card machines. All three lab machines have it switched on.

### D2 -- PARTLY MET

**Supposed to:** Exactly two watchers exist

**What the code does:**

There are two pipeline watcher ROLES, but they are not two separate programs, and they are not the only watcher daemons in the package. One command, `mousereach-watch`, reads the `watcher.mode` setting from ~/.mousereach/config.json and builds one of two orchestrator objects: mode "processing_server" builds ProcessingOrchestrator, anything else builds DLCOrchestrator (cli.py:292, cli.py:299, cli.py:302). If `watcher.mode` is absent, the code defaults to "dlc_pc" (config.py:676). The two classes live in one file (orchestrator.py:356 for DLCOrchestrator, orchestrator.py:1315 for ProcessingOrchestrator) and share a common base class. A second setting, `watcher.also_process`, blurs the division of labour: when it is true, the DLC-role watcher does not hand work off at all - it runs the analysis algorithms locally and archives the results itself (orchestrator.py:513, orchestrator.py:528, orchestrator.py:548), so on such a machine a single watcher does both jobs. All three graphics-card machines are configured with also_process true in the shipped machine profiles. Beyond these, the package ships two further long-running watcher daemons: a backup watcher that copies Y: to X: with robocopy on a timer (backup.py:25, registered as `mousereach-backup` at pyproject.toml:197), and a database sync watcher that watches the Processing folder and pushes results into the central database (sync/watcher.py:121, registered as `mousereach-sync-watch` at pyproject.toml:174). There is also a class literally named FileWatcher (watcher.py:32), but it is a helper used inside the orchestrators, not a daemon anyone starts.

**Why it matters:** "Two watchers" is right about the number of pipeline roles but wrong about the structure. Anyone looking for two programs will not find them - there is one program whose behaviour is chosen by a config value, so debugging "which watcher am I running" means reading a config file, not looking at which command was typed. And the also_process setting means a graphics-card machine can run the entire pipeline by itself, without the server ever touching those videos - so the handoff the description assumes may not be happening at all for a given video.

Evidence: `src/mousereach/watcher/cli.py:292`, `src/mousereach/watcher/cli.py:299`, `src/mousereach/watcher/cli.py:302`, `src/mousereach/config.py:676`, `src/mousereach/config.py:678`, `src/mousereach/watcher/orchestrator.py:356`, `src/mousereach/watcher/orchestrator.py:1315`, `src/mousereach/watcher/orchestrator.py:513`

### D9 -- NOT MET

**Supposed to:** Cropping and DLC are ALL watcher 1 does

**What the code does:**

The first watcher (the class named DLCOrchestrator, selected when the machine's config file says watcher.mode = "dlc_pc") has five kinds of work in its queue, not two. In order of priority they are: archive a locally finished video to the network drive; run the whole analysis pipeline locally; hand a posed video off to the shared handoff folder on the network drive; run DeepLabCut on one single-mouse video; crop one collage. Cropping and DeepLabCut are only two of those five. Two of the five are gated on the setting watcher.also_process. When watcher.also_process is false, the first watcher crops, runs DeepLabCut, and then MOVES the video plus its pose file to the shared handoff folder on the network drive so the other machine can pick them up - that move is a third job it does. When watcher.also_process is true, the first watcher never hands anything off: after DeepLabCut it runs segmentation, reach detection, outcome detection, and reach assignment; writes a provenance manifest; runs the quality-control triage step; runs the review gate that can divert a video into the human triage queue or the deep-review queue; extracts kinematic features; writes those features into the shared connectome database; and finally archives the whole bundle into the final Analyzed folder on the network drive. That is the entire rest of the pipeline, done on the first watcher's machine. This is not a hypothetical setting: the shipped machine profiles set also_process to true for all three graphics-card machines in the lab (the lab DLC PC and both behaviour-room PCs). On top of all that, the first watcher inherits a periodic housekeeping scan from the shared base class that both watchers run: roughly every 30 minutes it walks the final Analyzed folder's videos, compares each one's recorded tool versions against the currently declared versions, also checks whether a human review file is newer than the archived kinematics, and re-labels stale videos as "outdated". It also quarantines collage and single-video files whose names it cannot parse, and it registers any pre-cropped single-mouse videos it finds in the network Processing/Single_Animal folder.

**Why it matters:** The description treats the first watcher as a pure pose-estimation front end whose output always crosses to the server. On the lab's actual machines that is not what happens: with watcher.also_process = true the graphics-card machines produce final kinematics and write into connectome.db themselves, and the video never passes through the server at all. Anyone debugging "why did this video not appear in the handoff folder", or looking for the machine that made a particular kinematic result, will look in the wrong place. It also means the review gate, the triage queue routing, and the database write exist in two separate copies of the same logic (one in each watcher), so a fix applied to one is not automatically applied to the other.

Evidence: `src/mousereach/config.py:676`, `src/mousereach/config.py:678`, `src/mousereach/watcher/cli.py:298`, `src/mousereach/watcher/orchestrator.py:501`, `src/mousereach/watcher/orchestrator.py:513`, `src/mousereach/watcher/orchestrator.py:522`, `src/mousereach/watcher/orchestrator.py:528`, `src/mousereach/watcher/orchestrator.py:533`

### D18 -- PARTLY MET

**Supposed to:** Otherwise watcher 2 runs the MouseReach algos in order

**What the code does:**

There is a real, fixed running order, and it is close to what the claim says - but which machine runs it is a configuration choice, and two of the steps are conditional.

The ordered run is: segmentation, then reach detection, then outcome detection, then reach assignment (the step that decides which reach caused the pellet's fate), then a provenance manifest is written, then an automatic quality check, then a decision gate, and only if the gate says the video is clean does kinematic feature extraction run and the result get written into connectome.db. That order is hard-coded in one function on the processing server.

Three things the claim leaves out:

1. The same ordered run also exists on the GPU machine. Each machine's ~/.mousereach/config.json has a watcher.also_process setting. When it is false (the setting on this processing server - the key is simply absent, and the default is false), the GPU machine hands the posed video to the server and the server's watcher runs the algorithms. When it is true, the GPU machine runs the identical sequence itself immediately after DeepLabCut and never hands the video over. All three GPU machines in the lab profile file are set to also_process: true, so in the lab as configured today the GPU machines normally run the algorithms, not the second watcher.

2. Outcome detection and reach assignment are skipped entirely for videos whose filename says tray type E or F; those videos go from reach detection straight to the manifest/quality-check/gate steps.

3. When a video is being re-run because a tool version changed, the code can deliberately reuse a stage's existing output instead of re-running it, starting only from the first stage that went stale. So on a re-run, "runs the algorithms in order" can mean "reuses the first one or two and runs the rest".

**Why it matters:** Anyone reading the description will look for held or failed videos on the server. If a GPU machine has watcher.also_process set to true, that machine ran the algorithms, wrote the results, and put any held video into the review queues itself - and, as noted under D20, it never brings them back.

Evidence: `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:1811`, `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:1856`, `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:1909`, `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:1939`, `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:1973`, `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:2009`, `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:2047`, `Y:/2_Connectome/Behavior/MouseReach/src/mousereach/watcher/orchestrator.py:2057`

---

## 5. Files are not where the description says

Not dangerous on their own, but they send anyone debugging to the wrong folder, and one of them hides a single point of failure.

### D6 -- NOT MET

**Supposed to:** Cropped singles are relocated to Processing

**What the code does:**

The cropped single-mouse videos are not put in any folder called "Processing". The crop runs inside a scratch folder on the machine's own fast drive (processing_root + "/watcher_working"), and each finished single is then COPIED - not moved - into a separate local folder called DLC_Queue (processing_root + "/DLC_Queue"). The scratch copies are deleted immediately afterwards, so the only surviving cropped file is the one in DLC_Queue. On the two lab GPU machines processing_root is A:\MouseReach_Pipeline, so the singles land in A:\MouseReach_Pipeline\DLC_Queue, a purely local folder that the other machines cannot see. There genuinely are folders named "Processing" in this system - a local one (processing_root + "/Processing") and a shared network zone (nas_root + "/Processing", holding sub-folders Single_Animal, DLC_Complete, Review, Failed, Quarantine) - and the settings file even defines Paths.SINGLE_ANIMAL_OUTPUT as nas_root/Processing/Single_Animal. The automatic cropping path never writes to either of them. The only thing that writes to Processing/Single_Animal is the hand-run command-line tool `mousereach-crop`, whose default output directory is that folder. Two consequences worth knowing: (1) singles produced by the automatic path exist on exactly one machine's local disk until DeepLabCut has finished with them, so if that machine dies the crops are gone and the collage must be re-cropped; (2) if a person does drop singles into Processing/Single_Animal by hand, the watcher notices them and records them in its database in the "validated" state, but the GPU machine's work queue has no branch for "validated" videos at all - it only ever picks up videos recorded as dlc_queued, dlc_complete, processing or processed, plus collages recorded as stable - so those files sit there and are never posed.

**Why it matters:** Anyone following the description would look in a Processing folder for freshly cropped videos and find nothing, because they are in DLC_Queue on the GPU machine's local drive. It also hides a real fragility: between cropping and the end of DeepLabCut there is exactly one copy of each single, on one machine, invisible to the rest of the lab. And it makes the "drop files into Processing/Single_Animal" route look supported when in practice such files are registered and then never worked on.

Evidence: `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:145`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:749`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:751`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:816`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:818`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:819`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:820`, `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\watcher\orchestrator.py:859`

### D15 -- PARTLY MET

**Supposed to:** If outdated: the video moves back to Processing to be reprocessed

**What the code does:**

There are two different "outdated" branches, and only one of them puts the video back into a Processing folder.

First, nothing moves when a video is found to be outdated. The scanner that compares each archived video's provenance file against the declared current versions only writes a database row: it sets the video's state to 'outdated' and stores a 'reprocess_scope' saying which stage to restart from (watcher/reprocessor.py:100-157). No file is touched at that point.

Later, the processing-server watcher looks for videos in state 'outdated' as its fourth priority (watcher/orchestrator.py:1489-1505) and splits on the stored scope:

- Scope is anything other than 'full' (an algorithm version changed, or a newly saved human review needs applying): the watcher picks the video's pose file out of the finished-work tree, then COPIES every file sitting in that pose file's own folder whose name starts with the video id into the node's local Processing folder, verifying each copy (watcher/orchestrator.py:1563-1577, watcher/transfer.py:67). It then forces the database state to 'processing' and immediately runs the algorithm pipeline on the local copies (watcher/orchestrator.py:1581-1602). So a copy of the video does land in Processing and is reprocessed - but it is a copy, not a move: the archived originals stay exactly where they were.

- Scope is 'full' (the pose model itself changed): no file is copied or moved at all. The watcher just forces the state to 'dlc_queued' and returns (watcher/orchestrator.py:1493-1498). Only the DLC-machine watcher consumes 'dlc_queued' work (watcher/orchestrator.py:561-567), and the watcher database is per-machine, living at the node's own processing root (watcher/db.py:116-119). On a machine configured with watcher.mode = 'processing_server' there is no DLC handler in the same process, so this branch parks the video rather than sending it anywhere. Even on a DLC machine, the handler re-poses whatever path the database recorded as 'current_path' and marks the video failed if that file is gone (watcher/orchestrator.py:877, 912-914) - nothing copies the video back out of the finished-work tree for it.

Two more things the claim does not capture. The word "Processing" here means the node's own local scratch folder, PROCESSING_ROOT/Processing on a local drive (config.py:152), not the shared network folder that is also called Processing (config.py:118, 134). And the copy loop only picks up files that live in the same folder as the chosen pose file. In this pipeline the current pose files sit in a separate pose-only tree (Analyzed/Connectome/DLC Model 4/<cohort>/), while the video, the older pose, the algorithm outputs and any human review file sit in the cohort folder (Analyzed/Connectome/<cohort>/). Because the pose chooser prefers the declared current model (pipeline/manifest.py:90-124, and pipeline_versions.json declares the resnet101/shuffle3 model), the source folder for most videos is the pose-only tree - so the video file itself, and anything else beside it, is not copied into Processing at all.

Separately, there is a second, manually run bring-current tool that does not use Processing in any sense: it works in a temporary folder and writes results back beside the video (pipeline/reprocess_to_current.py:156-160, 290-306).

**Why it matters:** The claim describes one behaviour; the code has two, and only the algorithm-version branch reprocesses. If the pose model changes, the video is parked in a queue that the processing server itself cannot service, so it silently stops making progress. And because the reprocess copies from the pose file's folder rather than the video's folder, a video whose current pose lives in the separate pose-only tree arrives in Processing without its own mp4 - the pipeline then runs on pose data only, and the point in the run that hands the video file to the review gate has nothing to hand over.

Evidence: `src/mousereach/watcher/reprocessor.py:100-157`, `src/mousereach/watcher/orchestrator.py:1489-1505`, `src/mousereach/watcher/orchestrator.py:1563-1577`, `src/mousereach/watcher/orchestrator.py:1581-1602`, `src/mousereach/watcher/orchestrator.py:561-567`, `src/mousereach/watcher/orchestrator.py:877`, `src/mousereach/watcher/orchestrator.py:912-914`, `src/mousereach/watcher/transfer.py:67`

### D17 -- PARTLY MET

**Supposed to:** Human review files MOVE WITH the video back to Processing

**What the code does:**

Whether the review files travel with the video depends on which route the video takes back, and in the main route they do not need to travel at all because the code goes and finds them.

Route one - a human clears a video that was held in a review queue. Everything in the bundle folder is moved into the local Processing folder, with the sole exception of the two queue bookkeeping files; that includes "<video>_causal_review.json", "<video>_unified_ground_truth.json" and the deep-review clearance marker (watcher/review_return.py:163-180). Here the claim is accurate: the human files move with the video.

Route two - the version-staleness reprocess. Files are copied, not moved, and only from the folder that holds the chosen pose file (watcher/orchestrator.py:1563-1577). When the video, its pose and its review file all sit together in the cohort folder, the review file is copied along. When the current pose lives in the separate pose-only tree - which is where every current-model pose in this pipeline sits - the source folder contains no review file and none is copied.

That second case does not break the review, because travelling is not what makes a review count. Both the gate that decides whether a video may proceed and the kinematics step that consumes the review look the file up by name across several known places: the working folder, the triage queue bundle for that video, and the folder holding the canonical video (review/causal_review_io.py:52-87, review_gate.py:104-117, watcher/orchestrator.py:2067-2073). So an archived review sitting beside the video is still found and applied.

Ground truth is looked up differently and less completely. The gate's "this video is fully human-certified" check finds ground truth through an index built by scanning two roots only: the improvement working area and the network Processing folder (review/causal_review_io.py:384-386, 419-429). The finished-work tree is not one of them, so a ground-truth file that was archived beside the video is invisible to that check unless a copy also exists under one of the two scanned roots.

The stated reason - that these are facts about frame ranges and therefore survive any algorithm or model change - is not how the code stores or re-applies them. A review record is applied by matching its "segment_num" to the segment numbers in the freshly regenerated outcome file, and its chosen causal reach by "reach_id" (review/causal_review_io.py:298-345). Both are indices the algorithms reassign on every run. The frame range IS recorded, as "segment_span" on each record, and the code comment says in as many words that this is what makes the record durable (review/causal_review_io.py:176-199) - but nothing anywhere reads that field back; the only uses are the widget writing it (review/causal_review_widget.py:2250, 2359, 2830). There is a separate module written to protect human decisions across a re-run by matching on segment number and reporting any that no longer exist (review/clear_guard.py:1-18), but it is used only by the bundle re-staging tool (review/staging.py:340), not by the watcher pipeline.

**Why it matters:** Two gaps follow. First, the durability the claim asserts is aspirational: reviews are re-attached by segment number and reach id, so if a re-segmentation cuts the video differently, segment 7's review silently starts describing a different stretch of footage. The frame range that would prevent this is written into every record and never read. Second, the ground-truth lookup does not search the finished-work tree, so a video whose ground truth was archived alongside it can be re-triaged for questions a human already answered.

Evidence: `src/mousereach/watcher/review_return.py:163-180`, `src/mousereach/watcher/review_return.py:36-37`, `src/mousereach/watcher/orchestrator.py:1563-1577`, `src/mousereach/watcher/orchestrator.py:2067-2073`, `src/mousereach/review/causal_review_io.py:52-87`, `src/mousereach/review/causal_review_io.py:176-199`, `src/mousereach/review/causal_review_io.py:298-345`, `src/mousereach/review/causal_review_io.py:384-386`

---

## Requirements the code does meet

Stated here so the list of gaps is not mistaken for the whole picture.

| | requirement |
|---|---|
| D1 | Collages are recorded with OBS and placed in `MouseReach_Pipeline/Unanalyzed` on Y: |
| D3 | Watcher 1 runs on a CUDA "DLC PC" |
| D4 | Watcher 1 crops collages into single-mouse videos |
| D5 | Naming is by mouse, from video coordinates + string positions in the collage title |
| D7 | The parent collage STAYS in Unanalyzed (not fully processed yet) |
| D8 | Watcher 1 then runs DLC Model 4 on the singles in Processing |
| D10 | Watcher 2 runs on the server, over singles in Processing and Processed |
| D12 | Watcher 2 checks a DLC file exists |
| D13 | Watcher 2 checks the DLC file is the current version |
| D16 | Human review files (GT / causal / triage) are NEVER discarded on reprocessing |
| D20 | The video stays in triage until the watcher sees a human updated the triage file |
| D22 | Processed is organised by project and cohort |

---

## Reported gaps that did not survive checking

Each of these was reported as a divergence by the first reviewer and then
withdrawn when a second reviewer went back to the code. They are recorded so
the same false alarms are not raised again.

| | first read | after checking | why it was withdrawn |
|---|---|---|---|
| D1 | PARTIAL | TRUE | The claimed divergence does not hold up. (1) Intake folder: config.py:109 builds the intake path as NAS_ROOT / "Unanalyzed" / "Multi-Animal", and NAS_ROOT on this machine is Y:\2_Connectome\Behavior\MouseReach_Pipeline (nas_root key, config.py:99-100). That resolves to Y:\2_Connectome\Behavior\Mouse |
| D8 | PARTIAL | TRUE | D8 claims the GPU machine's watcher then runs DeepLabCut with Model 4 on the single-mouse videos it just cropped. The code does exactly that. The other reviewer's two grounds for PARTIAL do not support a divergence. (1) "The model is not fixed in the code" is a statement about mechanism, not behavio |
| D10 | PARTIAL | TRUE | D10 says: "Watcher 2 runs on the server, over singles in Processing and Processed." All three parts hold in the code, and the other reviewer's three objections are either misfiled under a different claim or rest on a folder the live pipeline does not use.  RUNS ON THE SERVER. src/mousereach/watcher/ |
| D11 | FALSE | PARTIAL | The other reviewer read the mechanisms correctly but graded the claim too harshly, and mis-framed what the description actually claimed.  What the description says, in its own words, is a single numbered test: watcher 2 "looks to see: 1. If the video has a finished DLC output file (that it is not pr |
| D13 | PARTIAL | TRUE | The code does what D13 says: the watcher checks that a video's pose was produced by the currently declared DeepLabCut model, and requeues it for a fresh pose when it wasn't. The other reviewer's two headline "gaps" do not hold. First, they claim the check is confined to the second watcher and runs e |
| D20 | PARTIAL | TRUE | D20 says the video stays in triage until the watcher sees that a human updated the triage file. The code does this. (1) The hold: when the gate decides triage, the whole bundle is moved out of the local Processing folder into the shared triage queue, the state is set to 'triage', and the pipeline re |

---

## Update 2026-08-21: one gap partly closed

**D19 (any algorithm failure or issue sends the video to triage)** was recorded
as NOT MET, because triage is only for videos where the algorithms succeeded and
left a question, while genuine failures set a database state nobody watches.

Part of that is now addressed for segmentation specifically. The segmenter always
emitted exactly 21 boundaries, so a forced segmentation was indistinguishable
downstream from a measured one and nothing was ever routed anywhere. It now
reports `needs_human` when it had to invent boundaries to reach that count,
discard detected ones to fit it, interpolate or fall back rather than detect, or
work from reference tracking that was not `good`. The review gate sends those
videos to DEEP_REVIEW, and `TriageStatus.clean` is false while the flag is set,
so they do not reach kinematics or the database first.

`mousereach-fix-segmentation` works that queue.

**Still not met:** the other failures under D19 are unchanged. A reach detector
or outcome detector that throws still sets state `failed` with nothing watching
it; a reach-assignment failure is still swallowed with a warning and lets the
video pass as clean; a quality-check failure still defaults to approved.
