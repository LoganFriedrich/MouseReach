# How the MouseReach pipeline actually works

Describes: src/mousereach/watcher, src/mousereach/archive, src/mousereach/config.py, src/mousereach/pipeline/
Verified against: b65fcf0 (2026-08-23), with sections 4, 5 and 6 re-verified
against the pathless-row and DLC-staleness fixes of 2026-08-24 (see the
update section at the end), and the collage-claim, re-crop and Posed
statements in sections 2, 3 and 4 corrected against 05de530 plus the
GPU-node guards of 2026-09-15. The pausing statements in sections 1, 3 and 4
and the update "the watcher pauses itself while a recording program runs"
were verified against the working tree on top of 73f9f48 (2026-09-15).

Written 2026-08-21 by reading the code, not the documentation. Each section
was traced against the source by a separate reviewer, and every statement
that contradicted the intended process was then handed to a second reviewer
whose job was to disprove it. Only the ones that survived are stated here as
differences.

The companion document is `PIPELINE_PROCESS_AS_DESCRIBED.md`, which records
how the pipeline is *supposed* to work.

**This describes behaviour as of 2026-08-21.** If you change the pipeline,
change this file in the same commit.

---

## 1. Where videos come in, and what is watching

HOW VIDEOS ENTER THE SYSTEM, AND HOW MANY WATCHERS THERE ARE - as the code actually works today.

1. THE FRONT DOOR

There is exactly one folder the system watches for new eight-mouse collage videos. The code builds it as NAS_ROOT / "Unanalyzed" / "Multi-Animal" (src/mousereach/config.py:109). NAS_ROOT is not hard-coded; it comes from the `nas_root` key in each machine's ~/.mousereach/config.json (src/mousereach/config.py:96). On every machine in the shipped machine profiles that key is <nas_root>, so the real front door today is:

    <nas_root>\Unanalyzed\Multi-Animal

Three things about this folder are easy to get wrong:

  - The scan is one level deep. The code lists that folder's direct contents with iterdir() (src/mousereach/watcher/state.py:70). A video placed in Unanalyzed itself, or in any other subfolder of Unanalyzed, is never seen.
  - Only five file extensions count: .mkv, .avi, .mp4, .mov, .wmv (src/mousereach/watcher/state.py:27 and :75).
  - The filename is a hard gate. It must read YYYYMMDD_{eight comma-separated animal IDs}_{tray letter}{run number}.ext - for example 20250101_CNT0101,CNT0102,...,CNT0108_P1.mkv. The date must be a real date and not in the future, and the tray letter must be P, E or F (src/mousereach/watcher/validator.py:106 and :111-118). A file that fails any of these is moved to a quarantine folder and never processed (src/mousereach/watcher/state.py:112). The quarantine folder defaults to NAS_ROOT\Processing\Quarantine (src/mousereach/config.py:720-726).

What software records the videos is outside the code. Recording, naming, and copying the file into Unanalyzed\Multi-Animal are human steps the code simply assumes have happened. The only thing the code can know about recording is optional configuration: a GPU node can be given the name of the program that records (watcher.pause_while_running), and then pauses itself while that program is open (see the update of 2026-09-15, "the watcher pauses itself while a recording program runs", at the end).

There is one other, deliberate way in: a one-off command copies archived ASPA collages into that same folder, re-encoding their names into the required format first (src/mousereach/aspa/import_collages.py:84, :229). It is run by hand, not on a timer.

Once a collage has been cropped it is left in place; much later, a separate slow-cadence job on the processing node moves a collage out to Analyzed\Multi-Animal, but only once every single-mouse video cut from it has finished the whole pipeline cleanly (src/mousereach/watcher/orchestrator.py:1404-1420).

2. HOW MANY WATCHERS - THE HONEST COUNT

There is ONE pipeline watcher PROGRAM with TWO ROLES, plus TWO other unrelated watcher daemons.

The pipeline watcher is started by the command `mousereach-watch` (pyproject.toml:178). On startup it reads the `watcher.mode` setting from ~/.mousereach/config.json and picks one of two behaviours (src/mousereach/watcher/cli.py:292-302):

  - mode = "processing_server"  -> ProcessingOrchestrator (src/mousereach/watcher/orchestrator.py:1315)
  - anything else, including a missing value, which defaults to "dlc_pc" (src/mousereach/config.py:676) -> DLCOrchestrator (src/mousereach/watcher/orchestrator.py:356)

The two roles watch DIFFERENT folders, which is the cleanest way to tell them apart:

  - The graphics-card role watches the collage front door, Unanalyzed\Multi-Animal, plus Unanalyzed\Single_Animal (src/mousereach/watcher/watcher.py:62-70). It crops collages, runs DeepLabCut, and hands the results off.
  - The processing-server role never looks at the front door at all. It watches only NAS_ROOT\Processing\Posed, the folder where posed videos are staged (src/mousereach/config.py:128, src/mousereach/watcher/orchestrator.py:1364-1371). It never crops a collage and never runs DeepLabCut - its work queue only contains archive, intake, pipeline and reprocess items (src/mousereach/watcher/orchestrator.py:1432-1531).

The clean two-role split has an important exception. A second setting, `watcher.also_process` (src/mousereach/config.py:678), when true, makes the graphics-card role skip the handoff entirely: it runs the analysis algorithms locally and archives the finished results itself (src/mousereach/watcher/orchestrator.py:513, :528, :548). On a graphics-card machine with also_process set to true, ONE watcher does the entire job end to end and the processing node never sees those videos.

Two further watcher daemons ship in the same package and are genuinely separate programs:

  - `mousereach-backup` (pyproject.toml:197) runs a timed copy of the configured folders to a backup location using robocopy (src/mousereach/watcher/backup.py, `BackupWatcher`). It only starts if a `backup` section is enabled in the config file (src/mousereach/watcher/backup.py, `main`).
  - `mousereach-sync-watch` (pyproject.toml:174) watches the Processing folder and pushes new results into the central database (src/mousereach/sync/watcher.py:121). Note that the pipeline watcher already syncs results to that database itself (src/mousereach/watcher/orchestrator.py:1167, :2095), so this daemon is an alternative tool rather than a required part of the flow.

A class called FileWatcher also exists (src/mousereach/watcher/watcher.py:32) but it is a helper used inside both orchestrators, not something anyone launches.

3. WHO RUNS THE GRAPHICS-CARD ROLE, AND HOW IT IS ENFORCED

The role is declared by a human in the config file, not detected from the hardware. But once declared, the command-line entry point checks the hardware before it will start. For any mode other than "processing_server" it requires: the network drive to exist, the DeepLabCut configuration file to exist, ffmpeg on the command path, and a usable graphics card (src/mousereach/watcher/cli.py:238-266). "Usable" means TensorFlow or PyTorch can see a card (src/mousereach/gpu.py:72); the PyTorch check is explicitly CUDA and returns false for a CPU-only build (src/mousereach/gpu.py:152), and the TensorFlow check on Windows is CUDA-backed too, with the code warning that TensorFlow past 2.10 has no native Windows card support (src/mousereach/gpu.py:122-130). With no card, the watcher prints the problem and exits (src/mousereach/watcher/cli.py:263, :274-276). The card number to use comes from `watcher.dlc_gpu_device`, default 0 (src/mousereach/config.py:658), and is passed straight to DeepLabCut as its `gputouse` argument (src/mousereach/watcher/orchestrator.py:930, src/mousereach/dlc/core/batch.py:272).

Two gaps in that enforcement are worth knowing:

  - The napari control panel starts the same DLCOrchestrator directly, with no hardware check at all (src/mousereach/watcher/control_widget.py, `WatcherControlWidget._start`), and lets the operator override the mode from a dropdown. It also takes no one-watcher-per-machine lock; since 2026-09-15 it refuses to start one while a watcher started elsewhere is running on the same PC (`health.watcher_running`).
  - Even on the command line, if the card check itself throws an error, the failure is downgraded to a log warning and startup continues (src/mousereach/watcher/cli.py:267-269).

The processing-server role has no graphics-card requirement, which is consistent with its own code: when it finds a video that needs DeepLabCut re-run, it explicitly hands it back rather than doing it, with the comment "can't do it here (no CUDA)" (src/mousereach/watcher/orchestrator.py:1494-1497).

4. THE SETTINGS THAT DECIDE ALL OF THIS

All of these live in ~/.mousereach/config.json on each machine, and setup defaults for a lab's machines come from the lab's own profiles file (found via MOUSEREACH_LAB_PROFILES, then ~/.mousereach/lab_profiles.json, then src/mousereach/setup/lab_profiles.json; the shipped .example is only a template), matched by hostname and by which drive letters are present (the hostname rule is checked since 2026-10-22).

  - nas_root - the shared pipeline root. Everything above (the front door, the staging folder, the review queues, the final Analyzed folder) hangs off it. If it is unset, every one of those paths is unset too and each tool says "nas_root is not configured" and names mousereach-setup (since 2026-10-22; before that the code guessed an older fixed layout -- see the update of that date at the end).
  - processing_root - the local fast drive where work actually happens.
  - watcher.mode - "dlc_pc" (the default when absent) or "processing_server". Chooses which of the two roles the one watcher program plays.
  - watcher.also_process - true means a graphics-card machine runs the whole pipeline itself instead of handing off; false means it stops after DeepLabCut and stages the result for the server.
  - watcher.dlc_gpu_device - which graphics card number DeepLabCut uses, default 0.
  - watcher.pause_while_running - programs that pause this watcher while they run (process names such as recorder.exe). Empty by default, which means the watcher never checks for any program. A running watcher follows edits to it without a restart.
  - watcher.pause_resume_grace_seconds - how long every listed program must have been closed before work starts again, default 120.
  - watcher.recording_days / watcher.recording_hours - optional, e.g. "Mon-Fri" and "08:00-18:00" on the machine's clock: when recording can happen at all. Outside them an open listed program does not pause the watcher. Empty = any day / any hour (since 2026-10-22).
  - watcher.pause_during_recording_hours - optional, true/false (default false). True = the recording days/hours ARE the rule: no work at all inside them, whether or not a recording program is open, and full work outside them whatever is open (a pose running when the window starts is stopped). Set it on a machine that films in fixed hours, where a recording program left open after filming would otherwise keep the node idle. Edit ~/.mousereach/config.json; a running watcher follows the change without a restart.
  - backup.enabled - whether the separate backup daemon will start.

---

## 2. Splitting a collage into one video per mouse

HOW COLLAGES ARE SPLIT AND NAMED - AS THE CODE ACTUALLY DOES IT TODAY

Where the work happens

Two different watcher programs exist. Each machine runs exactly one of them, chosen by a single setting in that machine's own configuration file at ~/.mousereach/config.json, under the "watcher" section, key "mode" (config.py:676). If mode is set to "processing_server", the machine runs ProcessingOrchestrator. Any other value - and the built-in default is "dlc_pc" - makes it run DLCOrchestrator (watcher/cli.py:298-303). All collage splitting lives in DLCOrchestrator; ProcessingOrchestrator contains no cropping code at all. A second setting, watcher.also_process (config.py:678), decides whether that same machine ALSO runs the behaviour algorithms locally after pose estimation; it has no effect on cropping.

Where collages are found

The watcher polls a folder on the shared network storage. The folder is not called "unprocessed" - the code builds it as NAS_ROOT/Unanalyzed/Multi-Animal (config.py, `MULTI_ANIMAL_SOURCE`), where NAS_ROOT comes from the machine's config key "nas_root"; there is no fallback, so with that key missing the folder is unset (config.py, `NAS_ROOT`). Any file there with extension .mkv, .avi, .mp4, .mov or .wmv is treated as a candidate collage (watcher/state.py:27, 75).

Name checking happens before anything else

Every newly seen collage has its file name validated (watcher/validator.py:103-229). The name must be DATE_ID1,ID2,ID3,ID4,ID5,ID6,ID7,ID8_TRAYRUN.ext, for example 20250101_CNT0101,CNT0102,CNT0103,CNT0104,CNT0105,CNT0106,CNT0107,CNT0108_P1.mkv. The date must be eight digits, be a real date, and not be in the future. There must be exactly eight comma-separated mouse identifiers, each one letters followed by at least four digits. The trailing label must be one of the letters P, E or F followed by digits (P for the pillar tray, E for easy, F for flat). At least one of the eight positions must be a real mouse. A collage that fails any of these is physically moved into a quarantine folder alongside a small text file recording why, and is marked quarantined in the watcher's own bookkeeping database (watcher/state.py, watcher/validator.py quarantine_file). Released from quarantine (see the update "a file released from quarantine is taken back in"), its name is checked again on the next scan. A collage that passes is recorded, then re-checked on later polls until its file size has stopped changing, at which point it is marked ready to crop (watcher/watcher.py:76-82).

When cropping actually runs

Cropping is the LOWEST-priority job the graphics-card machine does. On each cycle it first stages already-finished work back to the network, then runs pose estimation on any single-mouse video waiting for it, and only if both of those queues are empty does it pick up a new collage (watcher/orchestrator.py:530-580). Before starting, it claims the collage in a shared database (the pipeline_collages table of watcher_central.db on the shared root) so two machines cannot crop the same one. Since 2026-09-15 this fails closed: a node that cannot obtain a claim does not crop, and a collage whose children are all already finished or held for review is not cropped at all. The update section of 2026-09-15 at the end of this document has the details.

The cropping step itself

The collage is copied - copied, not moved - from the network folder to a scratch folder on the machine's own fast disk, at PROCESSING_ROOT/watcher_working (watcher/orchestrator.py:146, 741-745). The original stays on the network untouched; nothing in the watcher ever moves or deletes it. (A separate archive_collages function exists that would move collages away, but it is only reachable from the hand-run command-line tool, never from the watcher - video_prep/core/cropper.py:264, video_prep/cli.py:121.)

crop_collage then does the split (video_prep/core/cropper.py:105-184). It reads the eight mouse identifiers out of the file name and pairs them, strictly in order, with eight fixed rectangles: four across the top row and four across the bottom, each 480 pixels wide and 540 tall, which assumes the collage is exactly 1920 by 1080 (video_prep/core/cropper.py:39-48, 129). Nothing checks the real resolution of the incoming file. For each occupied position it shells out to ffmpeg with a crop filter and writes an .mp4 (video_prep/core/cropper.py:149-158). Video is re-encoded (only audio is copied through). Failures are recorded per position and the other positions continue.

How the single-mouse files get their names

The output name is DATE_MOUSEID_TRAYLABEL.mp4 - for instance 20250101_CNT0101_P1.mp4 (video_prep/core/cropper.py:145). The date and the tray label are lifted straight from the collage's file name; the mouse identifier is the one whose turn it is in the comma-separated list, and its turn is set by which grid cell is being cut. So the identity of every single-mouse video comes from two things and two things only: the order the human typed the eight names into the collage's file name, and the fixed reading-order of the grid. No text is read off the image, and nothing cross-checks that the name in slot three really is the mouse filmed by camera three.

Empty positions

A position whose cohort digits are "00" means an empty box and is meant to be skipped with no video produced (video_prep/core/cropper.py:133-142). The test used is a fixed character offset - characters 4 and 5 of the identifier (video_prep/core/cropper.py:55-61). That lands on the cohort digits only when the project prefix is exactly three letters (CNT, OPT, LAT). For a four-letter prefix such as ENCR, an identifier like ENCR0001 is not recognised as blank, so an empty-box video is cropped, named and sent onward. Everywhere else in the codebase the same question is answered correctly by finding where the letters end and the digits begin (config.py:293-317, watcher/validator.py:52-96).

Provenance written next to the collage

Immediately after cropping, a small record file named <collagename>_crop_manifest.json is written beside the collage (video_prep/core/cropper.py:178-182, video_prep/core/collage_provenance.py:189-243). It lists the collage, the date, the tray label, all eight identifiers, how many children were expected and how many were written, and for each of the eight positions its number, its mouse identifier, whether it succeeded, was skipped or failed, and its output file name. Writing it is best-effort - a failure here prints a warning and does not stop the crop.

Where the single-mouse videos go

This is where the real behaviour differs from the folder names people use in conversation. Each successful crop is registered in the watcher's bookkeeping database with its mouse identifier, project, cohort, subject, tray type and grid position (watcher/orchestrator.py:769-787). It is then COPIED to PROCESSING_ROOT/DLC_Queue on the machine's own local disk - not to the network (config.py:148, watcher/orchestrator.py:816-824). The path config.py:116 defines, NAS_ROOT/Unanalyzed/Single_Animal, is scanned for singles that someone put there by hand, but the watcher's own crop output never lands there. Finally, the local scratch copies of both the collage and all eight crops are deleted (watcher/orchestrator.py:856-859), so from that moment the only copies that exist are the untouched original on the network and the per-mouse files sitting in the local pose-estimation queue.

Re-running a collage

If a collage is picked up again later - its file is still sitting in the intake folder, after all - each child is checked first. A child that has already moved past the earliest stages is left alone rather than being reset and re-queued. Only children in the states "discovered", "validated" or "failed" are re-driven (watcher/orchestrator.py:791-812). Since 2026-09-15 the shared drive is asked too, because this node's database may be new or rebuilt. If a child's archive folder holds its processing manifest, or its bundle is in the triage or deep-review queue (or that queue's .incoming build folder), the child is recorded in that state and not queued for pose. If every child is in that position, the collage is not cropped.

Other ways cropping can be started

The same crop_collage function is also reachable by hand: a command-line tool (video_prep/cli.py:38, 52, 104) and a graphical panel (video_prep/widget.py:264, 284). Those paths write the crops wherever the user points them and can optionally copy them into the pose-estimation queue and move the collages to an archive folder. They are not part of the automatic watcher flow.

---

## 3. Where the cropped videos go, and pose estimation

HOW CROPPING AND POSE ESTIMATION ACTUALLY WORK TODAY

The cast. Two watcher programs exist, and which one a machine runs is decided by a single setting in that machine's own settings file at ~/.mousereach/config.json: watcher.mode. Set to "dlc_pc" the machine runs the DLCOrchestrator (the crop-and-pose watcher); set to "processing_server" it runs the ProcessingOrchestrator (the analysis watcher). This section covers the crop-and-pose watcher only. Machines with graphics cards run it; the analysis node does not.

Two roots. Every path in the system hangs off two settings. nas_root is the shared network folder everyone can see. processing_root is the machine's own fast local disk. Folders under nas_root are shared; folders under processing_root are private to one machine.

Step 1 - noticing a collage. Every thirty seconds the watcher lists the shared folder Unanalyzed\Multi-Animal. Any new video file whose name parses correctly is recorded in the watcher's database. It is then watched until its size stops changing for a configured number of seconds (watcher.stability_wait_seconds), which is how the system avoids grabbing a recording that is still being written. Only then does it become eligible for cropping.

Step 2 - choosing what to do next. The watcher does exactly one job per cycle, chosen by a fixed priority order: finish videos whose pose is already done, then pose a video that is waiting, and only when nothing at all is in flight does it start cropping the next collage. So cropping is deliberately the lowest priority - the machine drains what it has started before opening a new collage.

Step 3 - cropping. Before touching the file the machine claims the collage in the shared coordination database (the pipeline_collages table of watcher_central.db on the shared root), so two GPU machines watching the same folder cannot crop the same collage twice. No claim means no crop (since 2026-09-15). This covers a database that cannot be reached, a claim check that errors, and a claim another machine holds. A claim left unfinished for a day can be taken over, and a failed crop gives its claim back; the 2026-09-15 update section at the end has the rules and the reasons. It then COPIES the collage from the network into a scratch folder on its own disk (processing_root\watcher_working) and verifies the copy by size. The collage on the network is not moved, renamed or deleted. The scratch copy is cut with ffmpeg into eight fixed rectangles - the recording is a 1920x1080 grid of eight camera views, two rows of four, each cell 480x540. Grid positions whose animal identifier has cohort "00" mean "no mouse here" and are skipped. Each surviving cell is written as its own mp4 named {date}_{animal}_{tray}.mp4, into the same scratch folder.

Step 4 - where the cropped singles go. Each single is registered in the watcher's database and then COPIED (with a size check) into a second local folder: processing_root\DLC_Queue. The scratch copies, and the scratch copy of the collage, are then deleted. Note what this means: the cropped singles do NOT go to any folder called Processing. The settings file does define a shared network folder Unanalyzed\Single_Animal, and the hand-run command `mousereach-crop` writes there by default, but the automatic path never uses it. Between cropping and the end of pose estimation, each single exists in exactly one place - one machine's local disk - and is invisible to the other nodes. If a single had already been cropped and taken past this point in an earlier pass, it is left alone rather than reset; the two exceptions the code deliberately re-drives are children recorded as "failed" and children recorded as merely "validated", the latter because that state means the copy into DLC_Queue never succeeded and no file exists anywhere.

Since 2026-09-15 each child is also checked against the shared drive before it is queued. An archive manifest, or a bundle in a review queue, means the child is recorded as finished or held rather than posed again; a new or rebuilt node database cannot know that on its own.

A gap worth recording: the cropper is meant to leave a small provenance file beside the collage listing which child came from which grid position, but in the automatic path it is written beside the temporary copy in the scratch folder, so it never appears next to the collage on the network. Nothing downstream breaks, because the later retirement sweep falls back to reconstructing the child list from the collage filename.

Step 5 - pose estimation. When a single is sitting in DLC_Queue the watcher runs DeepLabCut on it, one video at a time, and tells DeepLabCut to write its output into that same DLC_Queue folder. The pose files (.h5, plus .csv) therefore land beside the video in DLC_Queue, not in a Processing folder. Which network is used is set in two places, neither of them in the code:
  - watcher.dlc_config_path in the machine's settings file names the DeepLabCut project's config.yaml. If it is unset or missing, nothing is posed and the video simply stays queued - it is not marked failed.
  - Which trained network inside that project - DeepLabCut calls these "shuffles" - is resolved in order from: an explicit argument, then watcher.dlc_shuffle in the machine's settings file, then the shuffle number parsed out of the dlc_scorer entry in the shared file pipeline_versions.json at the top of nas_root. For example, on a machine that does not set dlc_shuffle, a declared scorer ending in shuffle3_100000 means shuffle 3 is what runs. If none of the three sources names a shuffle, the code refuses to pose at all rather than accept DeepLabCut's silent default of shuffle 1, and it stops posing on that machine instead of failing the video, because an unresolvable model is a machine problem and not a video problem. After each video the code reads the model name out of the pose filename DeepLabCut just wrote and rejects the result if it is not the declared one, so a mismatched model cannot be handed downstream. Both safeguards exist so the watcher cannot quietly produce pose from an older model while the rest of the pipeline is calibrated for a newer one.
The watcher also re-checks the queue folder each cycle for pose files that appeared without its noticing - for instance after a crash mid-run - and records those videos as posed.

That only rescues a video whose pose file made it to disk. A watcher stopped DURING the pose left the video in 'dlc_running', which no job list selects, so it stopped moving for good - and silently, because the queue looked busy rather than broken. The same held for a video stopped mid-archive ('archiving') and a collage stopped mid-crop ('cropping'). Since 2026-09-13 every watcher, before it takes on anything new, puts such work back where it came from: 'dlc_running' returns to the pose queue, 'archiving' returns to 'processed' to be filed again, 'cropping' returns to 'stable' to be cut again, and each is named in the log as it happens (`BaseOrchestrator._reclaim_orphaned_work`). This is safe rather than a guess about a live run because a named global mutex allows exactly one watcher per machine, so anything still sitting in one of those states at startup was left by a process that is already gone. A video stopped mid-ANALYSIS needs no rescue and gets none: both watchers already select 'processing', and the pipeline reuses whatever stage outputs already exist, so it resumes rather than restarts. `mousereach-watch-status` names anything caught this way, so it is visible even while the watcher is down.

Step 6 - what happens after pose, and the setting that decides it. watcher.also_process in the machine's settings file controls this:
  - also_process = false: the machine is finished. Every file belonging to that video - the mp4, the .h5, the .csv - is MOVED out of the local DLC_Queue onto the shared network folder Processing\Posed, and the video is recorded as done-on-this-machine with both recorded paths (video and pose) pointing at the staged copies. (Until 2026-09-12 only the video path was updated; the recorded pose path kept naming the local DLC_Queue file the move had just removed, so archived rows pointed at a pose file that no longer existed.) Some other machine picks it up from there and runs the analysis steps. This is the arrangement the plain description assumes.
  - also_process = true: the machine keeps the work. It stages the pose and the video into its local Processing folder, runs segmentation, reach detection, outcome detection and reach assignment there, and then archives the results from that folder straight to the network under Analyzed\{project}\{cohort}, deleting its local copies in both Processing and DLC_Queue. (Until 2026-09-12 the pipeline had already moved to Processing but the archive step still read DLC_Queue, so it filed the mp4 and the raw pose with no manifest, segments, reaches or outcomes.) Nothing is ever staged to Processing\Posed.
On a machine configured with also_process = true, the crop-and-pose watcher in fact runs the entire pipeline, and the "hand off to the server" route is unused.

Step 7 - what eventually happens to the collage. It stays in Unanalyzed\Multi-Animal throughout everything above. It is removed only by the OTHER watcher - the analysis-server one - which runs a slow sweep roughly every thirty minutes over that folder. For each collage it works out the set of single-mouse children implied by the filename and checks whether every one of them has reached the final Analyzed output, was processed with the currently declared tool versions, and has no review outstanding. Only if all of that holds for every child does it stamp a completion record and MOVE the collage (and its provenance file, if one exists) to Analyzed\Multi-Animal, which the backup watcher can mirror to a backup location. It never deletes. So a collage still sitting in Unanalyzed means at least one of its children is unfinished, out of date, or waiting on a human.

Dropping a single-mouse video by hand into the shared folder Unanalyzed\Single_Animal works, and is the supported way to do it. The crop-and-pose watcher notices the file, records it, copies it onto its own disk and queues it for pose, and it goes through the rest of the pipeline like anything else. Until 2026-09-13 it did not: the file was noticed and written into the database in a state ("validated") that no job list selected, so it was recorded and then never worked on, silently and for good. `mousereach-process-animal` still exists and still works, but nobody needs it for this any more. A file dropped straight into a node's OWN DLC_Queue is picked up too, on the next cycle rather than only at the next restart; DeepLabCut's own by-products (a `..._labeled.mp4` beside a pose) are skipped rather than mistaken for unprocessed videos.

---

## 4. What each watcher actually does

HOW THIS PART OF THE PIPELINE ACTUALLY WORKS TODAY

There are two watcher programs in the code, and one machine runs exactly one of them. The choice is made at startup from that machine's own settings file (~/.mousereach/config.json), from the key watcher.mode. The value "dlc_pc" starts the first watcher; the value "processing_server" starts the second. There is no way to run both on one machine (src/mousereach/watcher/cli.py:298).

WHAT THE FIRST WATCHER DOES

The first watcher runs on a machine with a graphics card. Each cycle it scans two folders on the network drive: the collage intake folder (Unanalyzed/Multi-Animal) and the network folder Unanalyzed/Single_Animal (src/mousereach/watcher/watcher.py:60-72). Files whose names it cannot parse are moved to a quarantine folder (src/mousereach/watcher/state.py:110-124); since 2026-09-15 a misnamed single is quarantined only once it has stopped changing, because moving a file mid-copy breaks the copy. It then picks one job per cycle from a five-item priority list (src/mousereach/watcher/orchestrator.py:501-583):

1. Crop one collage. It copies the collage from the network drive to a local scratch folder, cuts it into single-mouse videos, names each one by which mouse it contains, registers each in its local tracking database, and copies each single into a folder called DLC_Queue on its OWN LOCAL DISK - not to the network Unanalyzed/Single_Animal folder. The local scratch copies are then deleted; the original collage is left where it was (src/mousereach/watcher/orchestrator.py:711-866, and src/mousereach/config.py:149).

2. Run DeepLabCut on one queued single. Output lands beside the video in the same local DLC_Queue folder (src/mousereach/watcher/orchestrator.py:871-966).

3. What happens next depends entirely on one setting: watcher.also_process (src/mousereach/config.py:678).

   - watcher.also_process = FALSE: the first watcher MOVES the video, its pose file, and the accompanying csv from its local disk into the shared handoff folder on the network drive, Processing/Posed (the constant DLC_STAGING, src/mousereach/config.py:128; the move is src/mousereach/watcher/orchestrator.py:1246-1313). It marks the video "archived" as far as its own bookkeeping is concerned, and is done with it. This is the case the description assumes.

   - watcher.also_process = TRUE: nothing is ever handed off. The first watcher runs the entire rest of the pipeline itself, on its own machine, using the same code the server would use: segmentation, reach detection, outcome detection, reach assignment, a provenance manifest, the quality-control triage step, the review gate that can divert a video into the human triage queue or the deep-review queue, kinematic feature extraction, a write into the shared connectome database, and finally an archive of the whole bundle into the final Analyzed folder organised by project and cohort (src/mousereach/watcher/orchestrator.py:968-1187 and 1188-1245).

So the statement "cropping and DeepLabCut are all the first watcher does" is not accurate in either configuration. With also_process off it also performs the handoff move; with also_process on it produces the final scientific product.

Two further things the first watcher does that are easy to miss. It USED TO inherit a periodic housekeeping scan from the shared base class, running at startup and then roughly every 30 minutes, which walks the archived videos in its own database, reads each one's manifest out of the final Analyzed folder, and re-labels as "outdated" any video whose recorded tool versions no longer match the declared current versions or whose human review file is newer than its archived kinematics (`ReprocessingScanner.scan`). On a first-watcher machine that had no follow-through: the first watcher's job list has no entry for "outdated" videos, so a video re-labelled this way left the "archived" state and nothing on that machine ever picked it up again. Worse, the state was not inert - it synced to connectome.db, other nodes adopted it during startup recovery, and it came back as a pathless row on a machine that had no file for the video. As of 2026-08-24 the scan is gated on `handles_reprocessing`, which is true only for the second watcher, so the first watcher no longer marks anything outdated. Only the second watcher acts on "outdated" (`ProcessingOrchestrator._get_next_work_item`). Separately, single-mouse videos found sitting in the network Unanalyzed/Single_Animal folder are registered in the database at state "validated" (`WatcherStateManager.discover_new_singles`). Until 2026-09-13 "validated" was not one of the states the first watcher's job list selected from, so those videos were recorded and then never processed; the job list now has a branch for them. Since 2026-09-15 a single is registered only once it has stopped changing. The branch first claims it by renaming it into Single_Animal/.inflight/<machine>/, then copies it onto the node and queues it for pose. See "Update 2026-09-15: videos can be dropped into either Unanalyzed folder at any time" at the end of this document.

WHAT THE SECOND WATCHER REQUIRES BEFORE IT WILL TOUCH A VIDEO

The second watcher runs on the server. It has exactly one source of new work: the network handoff folder Processing/Posed. Each cycle it globs that folder for DeepLabCut output files, and for each one it requires a matching video file with the same name to also be present in that same folder before it will register the video at all. If the video's filename does not parse, the video is quarantined instead (src/mousereach/watcher/orchestrator.py:1363-1372; src/mousereach/watcher/state.py:212-300). It does not scan the network Unanalyzed/Single_Animal folder, and it does not walk the final Analyzed folder looking for work.

Before copying a registered video in, it takes a claim: it creates a marker file named after the video, containing its own hostname, in a hidden .claims subfolder of the handoff folder. The create is exclusive (watcher/claim_marker.py), so when several machines race exactly one succeeds; a machine that loses leaves the video out of its choices for 10 minutes and takes the next one in the work order. The holder keeps the claim while the video's files are in the handoff folder -- through analysis, review and failure -- and lets go after archiving has removed them. A sweep every 10 minutes makes the claims match what the machine holds: it claims any video in the handoff folder that it has in hand (taken in and not given back -- in analysis, review, failed or filed) and that has no claim, and lets go of a video that has left the folder or is waiting again (a lifted review hold goes back as 'dlc_complete' for a fresh run, which any machine may take). Another machine's claim is broken only when that machine has stopped writing its status report for 3 hours by the file server's clock, never for its age (`ProcessingOrchestrator._claim_video`, `_sweep_posed_claims`; changed 2026-10-22 from write, wait half a second, read back, and delete after 24 hours). This claim exists only to stop two PROCESSING machines watching the same handoff folder from grabbing the same video. It is not a check against the first watcher: the first watcher never reads or writes those markers, and the second watcher never reads the first watcher's own collage-level claims, which live in the pipeline_collages table of watcher_central.db on the shared root and are only created by the first watcher (src/mousereach/watcher/orchestrator.py:721-727 versus the second watcher's constructor at :1326-1361, which never builds a coordinator).

There is therefore no explicit "is the other watcher still working on this?" check anywhere. What keeps the two apart is the shape of the handoff: the first watcher holds the files on its own local disk until DeepLabCut has finished, and only then moves them into the shared folder. Two caveats. First, the move is implemented as copy-to-the-final-name then delete the source (src/mousereach/watcher/transfer.py:108-130), and discovery only tests that the names exist - there is no wait for the files to stop growing. A partially copied video is visible under its finished name. The subsequent intake copy compares source and destination sizes and rejects a mismatch (src/mousereach/watcher/transfer.py:85-95), which usually turns such a race into a failed intake rather than a truncated file being analysed, but that is an after-the-fact guard, not a wait. Second, on a machine with also_process = true nothing is ever staged, so no contention arises there at all.

Once a video is claimed, its files are copied - copied, not moved, so the handoff folder keeps a copy - into a folder named Processing on the SERVER'S OWN LOCAL DISK (<processing_root>\Processing; the constant is built from the processing_root setting, src/mousereach/config.py:152). All algorithm work happens there. This is a different place from the network Processing zone under nas_root, despite the identical folder name.

What the second watcher does NOT check before running the algorithms: it does not verify that the pose file was produced by the currently declared DeepLabCut model. There is no version comparison anywhere in the run-the-algorithms path (src/mousereach/watcher/orchestrator.py:1811-2120). The only thing resembling a version check at this point is tie-breaking: if a video has more than one pose file, the code prefers the one from the declared model, and otherwise takes the newest and logs a warning that the video should be re-posed - but it still proceeds (src/mousereach/pipeline/manifest.py:89-134). Version currency is checked later and after the fact, by the staleness scanner walking already-archived videos in the Analyzed folder.

The second watcher's full job list, in priority order, is: archive a finished video to the network drive (first since 2026-08-30, so results reach the share continuously); take in a newly discovered video from the handoff folder; run the algorithms on a video sitting in its local Processing folder; and re-run an "outdated" video (`ProcessingOrchestrator._select_work_item`). Within each kind of job, which video goes next follows the configured work order (docs/WORK_PRIORITY.md; since 2026-10-22 including session lists and "not until nothing else is waiting" work, which comes last everywhere). For that last case, if the staleness scanner decided the video genuinely needs a NEW pose, the server cannot help - it has no suitable graphics card. Until 2026-08-24 it pushed the video into the 'dlc_queued' state and left it, which went nowhere: the server's own job list never selects that state, the GPU node's queue folder is local and invisible to the server, and the only thing that crossed machines was the state itself, through connectome.db, where it became a pathless row that crashed the GPU node's stager. It now holds those videos in 'outdated', names them once in the log, and lets the videos that CAN be re-run here through - so one un-poseable video no longer blocks the rest of the queue. Otherwise it pulls that video's pose file and video back out of the Analyzed folder into its local Processing folder and re-runs from the earliest stale stage (:1540-1605). Its local Processing folder is also refilled from a third direction: roughly every ten poll cycles it scans the two human review queues and moves any bundle a human has finished clearing back into Processing so the pipeline re-runs it (:1388-1397; src/mousereach/watcher/review_return.py:118-260). The review queues are shared storage, so when more than one node runs this scan, every queue mutation (return, retire, divert, a route-to-queue question or move) first takes a per-video claim marker in Review/.return_claims/ -- an exclusive create, so exactly one node's claim succeeds (watcher/claim_marker.py); the marker names the holding process (id and start time), so another live process of the same machine is refused too and a crashed process's claim is taken over (`claim_marker.take(..., process=this_process())`) -- by renaming the old marker to a name of its own first and deleting it only if it still names the ended process, so two processes that both saw the crashed holder can never both end up holding the claim; a marker of the same machine that names no process (an older version's) counts as another process's until it is `claim_marker.LEGACY_MARKER_TAKEOVER_S` (one hour) old, so an old review window and an updated watcher never both hold it; and within a process the claim is counted per thread so a route started by the scan does not drop the scan's claim; broken only when its holder has stopped writing its status report for 3 hours by the file server's clock, never for its age; no claim is needed when no shared root is configured, and none is taken (the move is skipped) when the claim folder cannot be made (review_return.py, _claim_return; changed 2026-10-21 from write, wait, read back and a 2-hour age rule). Intake is throttled by watcher.max_local_pending (default 2 since 2026-10-21; it was 200): once that many videos are sitting in the local Processing folder, no new videos are taken in until one is analysed, archived or cleared (:1444-1465). WHY 2: intake is visited before the pipeline, so with 200 the server took videos in for hours before analysing any, and every video it held was one no other node could take.

---

## 5. Checking the pose file, and what happens to old results

HOW THIS PART OF THE PIPELINE ACTUALLY WORKS TODAY

Vocabulary, so nothing below is guesswork:
- "pose file" = the DeepLabCut output, a .h5 file recording where each tracked body part sits in every frame. Its filename embeds the model that produced it, e.g. 20250101_CNT0101_P1DLC_resnet101_ExampleJan1shuffle3_100000.h5. That embedded string is called the "scorer".
- "manifest" = a small text file written next to a video after it is processed, named <video id>_processing_manifest.json. It records which scorer and which algorithm versions were used.
- "the version declaration" = one file, pipeline_versions.json, at the root of the pipeline tree on the network drive. It names the scorer and the four algorithm versions that count as current right now.

WHICH PROGRAM DOES THIS
One program, mousereach.watcher, runs on every machine. What it does is decided by watcher.mode in that machine's ~/.mousereach/config.json (config.py:676): "processing_server" builds the ProcessingOrchestrator, anything else builds the DLCOrchestrator (watcher/cli.py:298-302). A second setting, watcher.also_process (config.py:678), lets a graphics-card machine run the analysis algorithms itself instead of handing them off; when it is true the DLCOrchestrator runs a near-copy of the same pipeline code locally (orchestrator.py:968). So "the second watcher's job" is really "the analysis half of the program", and it runs on a processing_server-mode node always and on a graphics-card machine when watcher.also_process is true.

1. DOES A POSE FILE EXIST? — yes, checked, every time
Before the algorithms run, the code resolves the pose file through `resolve_pose_input` (moved 2026-08-24 from orchestrator.py to `watcher/locate.py`, which now holds every "is this claimed path a real file?" helper). It tries the path recorded in the watcher's database; if that is blank or is not a real file it globs the working folder for <video id>DLC*.h5. It deliberately asks "is this a file?" rather than "does this path exist?", because an empty string in Python turns into the current directory, which exists — that exact bug once fed a directory to the algorithms as a pose file and dumped videos into the human review queue (the comment at orchestrator.py:48-89 records it). No real file means the video is marked failed with "DLC h5 not found" and nothing else runs (orchestrator.py:1837-1839; the graphics-card copy at 992-996). Two other doors have the same guard: the path that pulls a finished video back for re-running does not run without a pose either -- but since 2026-10-22 it no longer fails the video for it (`_no_pose_for_reprocess`): a pose found only in the video's review bundle puts the video in that queue's state, a video still in Analyzed is marked as needing a new pose (the GPU nodes are asked for one), and only a video with neither pose nor video file is marked failed, saying so, and the path that returns a human-cleared video from the review queue refuses to return it rather than re-run it blind (review_return.py:152-159). Note the check is only "a readable file is there" — it does not open the file or judge whether it is complete.

2. IS THE POSE CURRENT? — checked, but only for already-finished videos, and from the manifest, not the file
About every thirty minutes (every sixtieth poll cycle at the thirty-second default; the review-return scan runs on its own much shorter cadence, since it is a cheap queue listdir while this sweep walks the whole archive) the processing side runs a sweep called the ReprocessingScanner (orchestrator.py:1341, 1373-1382). It walks the videos its database lists as 'archived' — videos that already completed and were moved into the final Analyzed output tree (reprocessor.py:97) — and then re-checks the rows already marked 'outdated', un-marking any whose versions compare current again (the two-way door; without it a declaration accident could mark rows forever). Rows carrying an explicit `mark_reason` (hand-marked targeted re-runs, e.g. of videos a compatible-version bugfix applies to) are never un-marked; scanner marks write no reason. For each it loads that video's manifest (reprocessor.py:106) and compares it to the version declaration (versions.py:138-195):
- scorer differs -> the video's ANALYSIS is stale, because segments, reaches and outcomes were computed from the older pose. Whether the POSE has to be made again is a separate question, and since 2026-08-24 the scanner asks it: it indexes the pose files in the Analyzed tree once and checks whether a pose from the declared scorer already exists for that video. If it does, the scope is 'segmentation' - every post-DLC stage re-runs against the pose that is already there, and no GPU is used. If it does not, the scope is 'full' and the video genuinely needs re-posing;
- one of segmenter / reach detector / outcome detector / kinematic extractor differs -> stale from that stage onward, and the re-run reuses the still-current earlier outputs (reprocessor.py:35-42, orchestrator.py:1846-1853). Exception: a manifest version listed under `compatible_versions` for that stage in `pipeline_versions.json` does NOT stale the video (versions.py) -- this is how a bugfix bump whose output only changes for pathological videos (which get re-marked by hand) avoids marking the whole corpus outdated;
- a human review the archived kinematics have not APPLIED also triggers a re-run of kinematics alone, so the reviewer's corrections reach the results and the central database. "Applied" is decided by content identity since 2026-09-09: the manifest records which review the extractor applied (reviewed_at + sha256, stamped by record_kinematic_version), and the scanner compares that stamp against the resolved review. It used to be a file-mtime comparison, which mixed two clocks -- SMB-written reviews carry the NAS server's clock, archived features keep the local clock -- and a NAS clock skew could make every fresh re-run look staler than the review it had just applied, reprocessing reviewed videos in a loop. Legacy manifests without the stamp fall back to the review's in-file reviewed_at vs the manifest's created_at (same process clock), and only lastly to mtimes;
- a video whose archived segmentation is HUMAN-authored (segments boundary_source == "human") is exempt from segmenter-version staleness, on both the mark side and the un-mark door: a human's cuts are facts about the video, no segmenter release stales them, and the re-run preserves rather than re-cuts them -- so before this exemption the manifest kept recording the old segmenter version and the scanner re-marked such videos after every single re-run, sweeping the same video into the versioned Archive over and over.
Videos found stale get the database state 'outdated' plus a 'reprocess_scope' (reprocessor.py:150). The same sweep is what "mousereach-version-check" prints, and "--mark" is the manual way to trigger it (watcher/cli.py:1231-1290).
Three things this does NOT do, which matter:
- It inspects the pose file only to answer "does a pose from the declared model already exist?", and only when the manifest names an older model (added 2026-08-24). It still never opens a pose file or judges its contents. If the manifest is absent the video is counted "no_manifest" and skipped. Where the manifest and the file on disk disagree, the manifest still wins on the question of what produced the current results - which is correct, because that is what it records. It is not corrected in place; the reprocess run rewrites it with the pose that actually ran, which is the only honest way for it to change.
- Videos on their way through for the first time are never checked. The pipeline function itself (orchestrator.py:1811) has no version comparison anywhere in it. A fresh video is analysed with whatever pose the graphics-card machine produced.
- If either side of a comparison is an empty string, that component is treated as current (versions.py:159, 167, 183). An unset "dlc_scorer" in pipeline_versions.json makes the pose-currency check a silent no-op with no warning.
Videos deliberately locked for a publication are put in the state 'crystallized', not 'archived', so the sweep never sees them (versions.py:242, reprocessor.py:164).
One related safeguard falls short of a check: when a video has pose files from two models side by side, select_pose_file prefers the one matching the declared current scorer; if none matches it takes the newest, logs a warning that the video should be re-posed, and proceeds anyway (manifest.py:89-135).

Bench-sheet disagreement ROUTING of filed videos is NOT part of the watcher (it was, briefly, 2026-08-25 to 2026-08-28). What the watcher does since 2026-10-26 is compare each video it analyses with a hand-scores FILE, if one is configured -- since 2026-10-28 right before the review gate (`_hand_score_precheck`, `watcher/hand_scores.py`; see the 2026-10-28 update) -- and MouseReach runs exactly as before with no such file. Comparing already-filed videos with the sheets stays an integrator's job -- a database tool that holds both -- and MouseReach must run with no such tool present. What MouseReach provides is the generic command `mousereach-route-to-queue` (watcher/route_cli.py): flag specific segments of an archived video and move it into the triage or deep-review queue with a recorded reason, through the same route_to_queue the pipeline uses itself. Whatever decides a video needs a person asks through that command; the watcher never reaches into another tool's environment or database. Likewise the central-database sync (sync/database.py) is configuration (`central_db`), absent by default, and cross-node coordination lives in MouseReach's own watcher_central.db on the NAS root.

3. WHAT HAPPENS TO THE OLD FILES - superseded, not overwritten (since 2026-08-21)
For a video marked outdated with a narrow scope, the second watcher copies every file whose name starts with the video identifier from Analyzed down to the local Processing folder — a copy, so the originals stay where they are (orchestrator.py:1574-1577). The algorithms re-run and overwrite the copies. Then the finished set is moved back to Analyzed/<project>/<cohort> by archive_video. That move used to be a bare shutil.move onto the same filename, which silently replaced whatever was there, destroying the previous generation's segments, reaches, outcomes, assignments, triage record, manifest and kinematics with nothing kept. Since commit 627877d (2026-08-21) archive_video calls supersede_video_outputs FIRST whenever anything for that video is already at the destination: the earlier generation is swept, checksum-verified, into the versioned Archive tree, and if that sweep reports any failure the archive REFUSES to proceed rather than move new files on top of old ones. The video itself and any ground-truth or human-review file are deliberately left in place, so a review still travels with its video. Since 2026-10-22 a file that is identical to the archived one (same size and time, or the same SHA-256 when the times differ) is not swept either: a re-run copies the video and pose down and hands them back unchanged, and sweeping them stored a duplicate pose per re-run and cost needless copying time.
The old pose file is the one exception, and only by accident: it is never moved or deleted. Because the model name is part of the filename, a new-model pose lands beside the old one and both stay.
For a video marked outdated because the pose model changed AND which has no pose from the declared model on disk (scope 'full'), the processing node does not do the copy-down. Before 2026-08-24 it forced the state to 'dlc_queued'; that reached no graphics-card machine and did reach cross-node recovery, where it produced pathless rows, so from then until 2026-09-12 it simply held the video in 'outdated' and logged the list once, and a person had to carry the video to a GPU and back. Since 2026-09-12 the ask travels over shared storage instead (src/mousereach/watcher/repose.py): right after each version scan the node writes one small request file per such video into Processing/Repose_Queue/; any graphics-card watcher pulls a couple at a time (claiming each by an atomic rename into Repose_Queue/.inflight/, copying the archived mp4 into its own local DLC_Queue, and queueing it for pose by the ordinary transitions, archived -> outdated -> dlc_queued); when the new pose lands in Processing/Posed the processing node ADOPTS it -- the row stays 'outdated' but its scope narrows to 'segmentation' and its recorded pose path points at the staged file -- and the copy-down route below re-runs every post-DLC stage against it, with the archived results folder (human segmentation and reviews included) copied down beside it. The request file is the idempotency key: it exists, queued or inflight, from publish until adoption (or until a GPU node that also processes archives the result itself), the GPU node heartbeats it while it holds the video, and a request nobody has touched for a day returns to the queue. Nothing is pushed to any machine; a lab with one GPU PC and a shared drive publishes and consumes on its own drive. Videos whose manifest names an old model but which DO have the declared pose on disk are not in this category at all: they get scope 'segmentation' and take the ordinary copy-down route below.
Be careful with the word "archive" in this code: the watcher's archive step means "move the finished outputs from the working folder into Analyzed/<project>/<cohort>", which is the live current-results location, not a historical store.

4. THE ARCHIVING CODE THE DESCRIPTION DESCRIBES DOES EXIST - and the watcher now calls it (since 2026-08-21)
mousereach/archive/supersede.py does exactly what the description says should happen. supersede_video_outputs (supersede.py:161) moves a video's about-to-be-replaced pose and algorithm outputs into a separate Archive tree at <pipeline root>/Analyzed/Archive, organised as "DLC Model <generation>/<seg…_reach…_out…_asn… version stack>/" so one pose is stored once per model generation while many algorithm variants live under it. Every move is checksum-verified: copy, compare sha256, and only then delete the source (supersede.py:105-128). It never overwrites — an identical file already there (under the base name OR any existing .N version, checked since 2026-09-09) means the incoming copy is simply dropped, and a genuinely different same-named file is saved as .1, .2 and so on. Before the .N identity check, a reprocess loop that re-swept unchanged content added another byte-identical multi-MB copy per lap. The video itself, the ground-truth file and the human review file are explicitly never archived (supersede.py:55-57, 204-206), matching the rule that human judgements are facts about the video and stay with it.
It now has two callers: archive_video (archive/core.py), which is the watcher's own filing step, and pipeline/reprocess_to_current.py:293-294, a hand-run bring-one-video-up-to-the-current-stack tool with no command-line entry point. The sentence that stood here -- that nothing automatic called it -- was true until 2026-08-21 and is not now. It has to be invoked by hand, and only does the archiving when called with finalize=True and only for videos the review gate judged clean; held videos are staged into a review queue instead and their old outputs are left untouched (reprocess_to_current.py:66-108, 280-300).

NET EFFECT
Existence of the pose file is properly guarded on every route into the algorithms. Currency of the pose is checked only as a background sweep over already-finished videos, from a text record rather than the file, and can be inert if the version declaration leaves the scorer blank. When a re-run happens inside the watcher, the previous generation's results are now swept into the versioned Archive tree first, checksum-verified, and the archive refuses to proceed if that sweep fails -- so what an earlier model produced for a video remains recoverable. That was not true before 2026-08-21, when the previous generation was overwritten in place.

---

## 6. Reprocessing, and what happens to human review files

HOW SENDING AN OUTDATED VIDEO BACK FOR REPROCESSING ACTUALLY WORKS TODAY\n\nVocabulary first, because the folder names in the code are not the words people use.\n- \"Processing\" in the reprocessing code means the machine's OWN local scratch folder: <processing_root>/Processing, where processing_root comes from that machine's ~/.mousereach/config.json (config.py:56-57, 152). There is also a shared network folder called Processing that holds the post-pose staging area (Processing/Posed), the failures folder and the two human-review queues; cropped single-mouse videos wait in Unanalyzed/Single_Animal (config.py:116, 128, 164-166). These are different places with the same name.\n- \"Analyzed\" is the finished-work tree on the network, organised by project and cohort (config.py:123). In practice it holds two kinds of folder side by side: cohort folders such as Analyzed/GrantA/PROJA/PROJA_03/ containing the video, its algorithm outputs and any human files, and, where a bulk pose job has been run, a pose-only tree such as Analyzed/GrantA/DLC Model 4/<cohort>/ containing the current-model pose files.\n- Each machine keeps its own small SQLite bookkeeping database at <processing_root>/watcher.db (db.py:116-119), with a summary exported to a shared one on the network after each archive (orchestrator.py:2146).\n\nSTEP 1 - DECIDING A VIDEO IS OUT OF DATE\nA scanner walks every video the local database calls 'archived' and reads that video's provenance file, <video>_processing_manifest.json, out of the Analyzed tree (reprocessor.py:100-113, 193-217). It compares that file against the declared current versions in MouseReach_Pipeline/pipeline_versions.json (versions.py:138-200). A video is out of date if the pose model differs, if any tracked stage's recorded version differs, or if a stage recorded no version at all. A video that is fully current is ALSO pulled back if a saved human review is newer than its kinematics output, so a reviewer's answer reaches the final data (reprocessor.py:114-136, 175-191).\n\nThe scanner writes no pipeline data (its only file is a cache of what it has read, version_scan_cache.json in the machine's own processing root; see the update of 2026-10-22 on keeping it across restarts). It sets the database state to 'outdated' and stores a \"scope\": 'full' when the pose model changed, otherwise the earliest stale stage - segmentation, reach, outcome or kinematics - so that stage and everything after it re-run while current upstream results are reused (reprocessor.py:26-42, 149-157). This scan runs at watcher start-up, then on a timer (orchestrator.py:216-241), and again every tenth poll cycle on the processing-server watcher (orchestrator.py:1378-1387). It can also be run by hand with `mousereach-version-check --mark` (cli.py:1275-1285).\n\nSTEP 2 - WHAT HAPPENS TO AN OUTDATED VIDEO\nThe processing-server watcher picks up outdated videos as its fourth priority, behind archiving, intake, and running the pipeline (order changed 2026-08-30: archiving moved from last to first so finished results reach the NAS continuously instead of only when the staged supply runs dry).\n\nIf the scope is 'full', nothing is copied or moved and the video stays 'outdated'. Since 2026-08-24 'full' means only "this video has no pose from the declared model anywhere in the archive" - a manifest naming an old model is not enough on its own, because the bulk re-pose had already produced current pose files for most of them. The row is no longer forced to 'dlc_queued': that reached no graphics-card machine (the DLC queue folder is local to each node) and did reach cross-node recovery, where the state became a pathless row that crashed the stager. Videos in this state are listed in the log once, and since 2026-09-12 a re-pose request for each is published to the shared Processing/Repose_Queue folder after every version scan, for any graphics-card watcher to pull (section 5 describes the round trip; watcher/repose.py implements it). When the new pose is staged back, the row's scope narrows to 'segmentation' and the ordinary copy-down re-run below takes over. On a DLC machine the pose handler now resolves the video file rather than trusting the recorded path, and a video with no file on that node is recorded 'unresolvable' instead of failed-and-retried.\n\nIf the scope is anything else, the watcher picks the pose to re-run against -- since 2026-09-12 a pose the row already points at (a freshly staged one from the declared model, adopted by the re-pose round trip) first, then a declared-model pose in Processing\Posed, and only then the video's results folder and the same cohort folder of each DLC Model tree, with a search of the whole Analyzed tree only when no declared-model pose is there (since 2026-10-22; the walk took minutes per re-run) (preferring the declared current model, else the newest - manifest.py:90-124) -- and copies two folders' worth of files into the local Processing folder, verifying every copy: from the pose file's own folder only the DLC files, and from the video's results folder (Analyzed/<project>/<cohort>) everything whose name begins with the video id, the video, the previous outputs and any human files included (orchestrator.py _reprocess_copy_set). Nothing is deleted from Analyzed - this is a copy, so the archived set stays intact until a re-archive overwrites it. The state is forced to 'processing' and the standard algorithm run starts immediately, skipping the stages upstream of the stale one (orchestrator.py:1581-1602, 1843-1853).\n\nWhen the declared current pose files live in a separate pose-only tree, the pose comes from one folder and everything else from the video's results folder; that two-folder copy set (added after a pose-folder-only copy re-segmented away a reviewer's hand-set cuts) is what puts the video, the older pose and any human files beside the new pose in Processing.\n\nA second, entirely separate route exists for bringing videos current by hand: pipeline/reprocess_to_current.py. It never uses Processing at all - it copies the pose into a temporary folder, runs the algorithms there, and on request moves the results back beside the video, first moving the previous generation's outputs into a version-stamped Archive tree (reprocess_to_current.py:156-160, 285-321).\n\nSTEP 3 - HUMAN FILES\nThere are two human file types, not three. Ground truth is <video>_unified_ground_truth.json. Both the causal review tool and the fast triage review tool write the same file, <video>_causal_review.json - a segment counts as resolved when that file's record for it says it was reviewed (triage_status.py:16-18). The separate <video>_triage.json is the algorithm's own quality verdict, not a human file.\n\nNothing in any reprocessing path deletes either human file -- but until 2026-08-24 that sentence was true and still misleading. Nothing had to delete a review to lose it. The review was written into the triage bundle, the bundle is transient, and returning a cleared bundle MOVES its files onto one node's LOCAL disk and removes the directory. Shared storage then held no copy until the video was archived, so a review could exist only on one machine's local disk, or only inside a NAS bundle, one reprocess from gone. Reviews are now written to {NAS}/review_records/reviews/ at save time, before the working copy, and that copy belongs to no bundle's lifecycle. The version-driven reprocess only copies. The archiving step moves every file starting with the video id from Processing into the cohort folder, so human files travel with everything else (archive/core.py:144-190). The one routine that deliberately clears out old outputs - the versioned \"supersede\" move used by the manual bring-current tool - carries an explicit never-touch list containing the ground-truth file, the causal-review file and the video itself, and records them as kept (supersede.py:49-57, 131-136, 197-215). The automatic watcher never calls that routine, which is a separate reason human files are safe there. The only deletions in this area are queue bookkeeping files when a cleared bundle leaves a review queue (review_return.py:36-37, 166-170), staging copies after archiving (orchestrator.py:1225-1231, 2151-2158), and a manual reset helper that removes only algorithm outputs (pipeline/core.py:311-333).\n\nDo the human files travel with the video? Only on one of the two return routes. When a person clears a video that was held in the triage or deep-review queue, every file in that bundle - review file, ground truth, clearance marker - is MOVED into the local Processing folder, minus the two bookkeeping files, and the video is set back to 'processing' (review_return.py:118-214). On the version-staleness route they travel too, because the copy set takes everything beginning with the video id from the results folder (only the DLC files from the pose folder).\n\nThat mostly does not matter, because the code goes looking for the review rather than relying on it having travelled. Both the gate that decides whether a video may proceed and the kinematics step search for <video>_causal_review.json in the working folder, in that video's triage-queue bundle, and in the folder holding the canonical video, taking the newest hit (causal_review_io.py:52-87; review_gate.py:104-117; orchestrator.py:2067-2073). Ground truth is looked up differently: through an index built by scanning only the improvement working area and the network Processing folder (causal_review_io.py:384-386, 419-429). The Analyzed tree is not scanned, so ground truth archived beside a video is invisible to the gate's \"fully human-certified\" shortcut unless a copy also exists under one of those two roots.\n\nSTEP 4 - THE RE-RUN ITSELF\nThe re-run regenerates the algorithm outputs from scratch; it does not preserve the human flags that were written into the old algorithm files. The human decision lives in the review file and is re-applied at the very end: the gate treats a segment as resolved if the review says so, and the kinematics step passes the review file to the feature extractor so the human's outcome and chosen causal reach replace the algorithm's, with the algorithm's originals kept alongside for provenance (review_gate.py:96-101; orchestrator.py:2067-2078; causal_review_io.py:298-345). If the video comes out clean it goes to 'processed', gets archived into Analyzed/<project>/<cohort>, and its kinematics are pushed into connectome.db by a per-video replace. If anything is still unresolved, the whole bundle is moved back out to the triage or deep-review queue and nothing reaches kinematics or the database (review_gate.py:191-222).\n\nTHE ONE THING THAT IS NOT TRUE YET -- CORRECTED 2026-08-24\nThis section used to say that a review is re-attached by segment number and that nothing reads \"segment_span\" back. That was true when it was written and is no longer. `index_review_by_segment` (causal_review_io.py) matches a review to the CURRENT segmentation on frame overlap, ignoring the numbers: it needs 50% overlap, requires the best candidate to beat the runner-up by 15%, and DROPS a review whose frames straddle two segments or match none, saying so in its notes rather than guessing. `resolve_truth_layers` threads the current segments into it, and the kinematics extractor calls that. So a re-cut no longer stamps a human judgement onto footage nobody looked at.\n\nWhat WAS still broken, and was fixed the same day: the resolver looked for a review only in the two NAS review queues and the caller's processing dir -- all three places a review can vanish from. A review that outlived its bundle but whose video was not archived yet was in none of them, so the extractor fell back to the algorithm's answer while the human's sat in the durable store unread. The durable store is now the lowest-priority layer in that stack (any live copy still wins, since a reviewer may have edited it). Pinned by a test that fails against the previous code with the human's outcome discarded.\n\nThe corpus index also now stores segment_span, the human's chosen causal reach and their answers, so a review whose file is lost is reconstructable; it previously held a file path plus two summary fields. A module that guards human decisions across a re-run does exist, and reports rather than silently drops decisions whose segment vanished (clear_guard.py:1-18), but it is wired only into the bundle re-staging tool (staging.py:340), not into the watcher pipeline. So today, a re-segmentation that renumbers segments can silently re-point an old review at different footage.\n\nCONFIGURATION THAT CHANGES ANY OF THIS\n- watcher.mode ('dlc_pc' or 'processing_server', config.py:676): selects which watcher class runs. Only the processing-server watcher acts on 'outdated' videos and returns cleared review bundles; only the DLC watcher consumes 'dlc_queued'.\n- watcher.also_process (config.py:678): on a pose machine, run the algorithms locally straight after posing and archive directly, instead of staging for another machine.\n- processing_root and nas_root in ~/.mousereach/config.json (config.py:56-57, 99-105): set the local scratch Processing folder and the network tree respectively. If nas_root is left unset every derived path - staging, Analyzed, the review queues - is unset too, and the tools say so (until 2026-10-22 the code silently fell back to an old layout instead).\n- watcher.dlc_shuffle (config.py:665-667): if unset, the pose model used is taken from the declared dlc_scorer in pipeline_versions.json, so producer and version checker cannot disagree.\n- MouseReach_Pipeline/pipeline_versions.json: the declared current versions. Editing it is what makes existing videos outdated on the next scan.

---

## 7. Running the algorithms, and what happens when one fails

HOW THE ALGORITHMS ACTUALLY GET RUN, AND WHAT HAPPENS WHEN ONE DOES NOT SUCCEED\n(code root: <repo>, branch master)\n\nHISTORICAL SNAPSHOT: this block describes the code as it was when it was written; its line numbers are stale, and two of its statements are no longer true -- (1) a 'failed' video IS picked up again: the processing watcher retries it after a wait, up to watcher.max_retries times, and a filed video whose re-run fails for good goes back to 'archived' (see \"Update: a filed video whose re-run fails stays filed\"); (2) a returned video does NOT re-run from segmentation: it resumes just after the step that sent it (see \"Update: a returned video resumes just after the step that sent it\"). Where this block and those updates disagree, the updates are right.\n\nWHICH MACHINE RUNS THE ALGORITHMS\nEach machine has a file ~/.mousereach/config.json with a \"watcher\" section. Two settings decide the behaviour here:\n- watcher.mode: \"dlc_pc\" (the GPU machines that crop collages and run DeepLabCut) or \"processing_server\".\n- watcher.also_process: only meaningful on a GPU machine. False means \"after posing the video, put it on the NAS in Processing/Posed and let the server take it\". True means \"after posing the video, run the whole analysis here as well\".\nA missing also_process key defaults to false (src/mousereach/config.py:678). Both machine roles contain a full copy of the analysis sequence: ProcessingOrchestrator._run_pipeline (orchestrator.py:1811) on the server, and DLCOrchestrator._run_local_pipeline (orchestrator.py:968) on a GPU machine. The two are near-identical, step for step.\n\nTHE SEQUENCE, IN ORDER\nBefore anything runs, the code resolves the DeepLabCut pose file for real: it uses the recorded path only if it is an actual file, otherwise it searches the working directory for {video}DLC*.h5, and returns nothing rather than a placeholder. If there is no pose file the video is marked 'failed' and the run stops (orchestrator.py:1830-1841). This guard exists because an empty path once resolved to the current directory, which \"exists\", and videos were pushed into the human review folder in bulk as a result.\n\n1. Segmentation - splits the video into per-pellet segments (orchestrator.py:1856).\n2. Reach detection (orchestrator.py:1909).\n3. Outcome detection - what happened to each pellet (orchestrator.py:1939). Skipped entirely if the filename says tray type E or F.\n4. Reach assignment - which reach caused that outcome (orchestrator.py:1973). Also skipped for tray types E and F. It runs before the gate on purpose, because the gate treats \"pellet was touched but no reach was credited\" as a question for a human.\n5. A provenance manifest recording which tool and model versions produced the outputs (orchestrator.py:1989).\n6. An automatic quality check across all the outputs, which returns either 'auto approved' or 'needs review' and stamps that verdict into each output file plus a {video}_triage.json record (orchestrator.py:2009).\n7. The gate (orchestrator.py:2047, implemented in src/mousereach/watcher/review_gate.py). Nothing reaches kinematics or the database until the gate says clean.\n8. Only if clean: kinematic feature extraction, applying any saved human review corrections, then a write into connectome.db (orchestrator.py:2057-2103). The video is then marked 'processed' and later archived.\n\nOn a re-run triggered by a tool version change, the code can start from the first stale stage and reuse the earlier outputs instead of recomputing them (orchestrator.py:1846-1854).\n\nWHAT THE GATE DECIDES (review_gate.py:68-101)\n- If ground truth marks the whole video as exhaustively scored by a human, it is clean regardless of anything the algorithms flagged.\n- Else, if segmentation is unusable, DEEP REVIEW.\n- Else, if the automatic quality check said 'needs review', DEEP REVIEW.\n- Else, if any segment is still flagged and unanswered, TRIAGE.\n- Else, clean.\nA segment counts as flagged when the outcome cascade marked it \"triaged\" or set flagged_for_review, or when the outcome says the pellet was touched (retrieved, displaced in the scoring area, displaced outside) but no reach in the assignment file is marked causal (src/mousereach/review/triage_status.py:70-95).\n\nWHERE HELD VIDEOS GO\nBoth queues live on the NAS so any machine and the review tools can see them (src/mousereach/config.py:164-166):\n- Triage queue: Processing/Review/triage\n- Deep review queue: Processing/Review/deep_review\nThe whole bundle is MOVED out of the local Processing directory into a folder named after the video: the mp4, the pose file, and every algorithm output, plus a {video}_routing.json audit record and a {video}_manifest.json that lets the review tool open the video in place (review_routing.py:73-149, review_gate.py:120-138). The database state becomes 'triage' or 'deep_review'; if that state change is rejected it is forced, because the files have already moved and the disk is the truth (review_gate.py:158-176).\n\nWHAT HAPPENS WHEN A STEP DOES NOT SUCCEED - THIS IS NOT THE SAME AS TRIAGE\n- Segmentation raises an exception, or the pose file cannot be opened at all: state 'failed'. Nothing moves. This is treated as an infrastructure problem, deliberately kept away from human reviewers (orchestrator.py:1881-1887, and the reasoning at orchestrator.py:93).\n- Segmentation runs but reports unusable boundaries: DEEP REVIEW (orchestrator.py:1892-1901).\n- Reach detection or outcome detection raises: state 'failed' and the error is re-raised; the outer dispatcher logs it and marks 'failed' again (orchestrator.py:1933-1937, :1965-1969, :1514-1538).\n- Reach assignment raises: a log warning, and the pipeline continues (orchestrator.py:1986). If no assignment file was written, the \"touched pellet with no causal reach\" check is skipped, so such a video can pass the gate as clean and be written to the database with no causal-reach attribution.\n- The automatic quality check raises: a warning; the verdict stays 'auto approved' and the video proceeds (orchestrator.py:2039).\n- Kinematic feature extraction raises: a warning only. The video is still marked 'processed' and will be archived - with no kinematics and nothing written to connectome.db (orchestrator.py:2105-2110).\n- The database write fails: a warning only (orchestrator.py:2102).\n[SUPERSEDED: a failed video is now retried after a wait, up to watcher.max_retries times, and a filed video whose re-run fails for good goes back to 'archived' -- see the update named above.] No watcher ever picks up a video in state 'failed' - none of the work queues query that state (orchestrator.py:1432-1512, :501-580). A person has to run mousereach-watch-reprocess, optionally with --all-failed, to reset them (src/mousereach/watcher/cli.py:540-640).\n\nGETTING A TRIAGED VIDEO BACK INTO THE PIPELINE\nThe review tool, opened on a bundle, saves the human's answers as {video}_causal_review.json inside that bundle folder (src/mousereach/review/causal_review_widget.py:880-885, :2843-2850). The {video}_triage.json file in the same folder is the automatic quality check's record and plays no part in release.\n\nA watcher in processing-server mode calls the return scan from its scan step every tenth polling cycle - roughly every five minutes at the default 30-second poll - and returns at most 10 bundles per scan so a deep queue does not starve actual processing (orchestrator.py:1388-1397, :1348, src/mousereach/watcher/review_return.py:222). A watcher in dlc_pc mode never calls it (its scan step, orchestrator.py:483-496, does only file discovery and DeepLabCut completion checks), so a GPU machine running with also_process true can put videos into triage but cannot take them out.\n\nFor each bundle in the triage queue the scan recomputes the picture from the bundle's own files and releases it only when there is at least one flagged segment, every flagged segment has an answer whose \"reviewed\" field is not false, and segmentation reads as sound OR a human's deep-review clear marker is in the bundle (since 2026-09-09 the scan honors {video}_deep_review_cleared.json exactly as the gate does: such a bundle is never diverted to deep review and releases once its segments are answered -- previously the gate's honor and the scan's raw seg_failed divert fought each other and a cleared video lapped the three states indefinitely). Before touching anything it makes sure the video has a row in this machine's watcher database, registering it if the review was cleared on another machine (review_return.py:134-146). It then locates the mp4 and pose file - in the bundle, then via the bundle manifest, then by searching the Analyzed archive - and refuses to return the bundle if no pose file can be found, so a cleared review is not spent on a run guaranteed to fail (review_return.py:40-95, :152-159). The data files are moved into the local Processing directory, the two bookkeeping files are deleted, the state is set to 'processing' (forced if the normal transition is rejected), and the now-empty bundle folder is removed (review_return.py:161-214).\n\n[SUPERSEDED: a returned video now resumes just after the step that sent it -- triage at reach assignment or at the checks, deep review at reach detection -- and every later step and check runs; see \"Update: a returned video resumes just after the step that sent it\".] The returned video then re-runs the whole sequence from segmentation. The human's review file came back with it, so the gate sees the answers and lets it through, and feature extraction loads that same file and applies the reviewer's corrections before computing kinematics (orchestrator.py:2066-2078). Deep-review bundles come back the same way (now: resuming at reach detection, the person's cuts kept), but the release signal is different: either a {video}_deep_review_cleared.json marker written by the deep tools, or a {video}_unified_ground_truth.json produced by the ground-truth tool (review_return.py:108-115, :252-259).\n\nTWO EDGES WORTH KNOWING\n- The gate treats a segment as resolved if ground truth determined it, or if a review file found in any of several locations answers it. The return scan is stricter: it only reads {video}_causal_review.json inside the bundle and does not consider ground truth. A video whose flagged segments include ground-truth-determined ones can therefore sit in the queue after the reviewer has answered everything actually asked of them (review_gate.py:91-100 vs review_return.py:244-249).\n- The server's scan step returns immediately if no DeepLabCut staging directory is configured (orchestrator.py:1364-1366), and that staging path only exists when nas_root is set. With nas_root missing, the review-return scan never runs at all.

---

## 8. Finishing a video, and reaching the database

HOW THE END OF THE PIPELINE ACTUALLY WORKS TODAY

Two names first, because the code does not use the words in the description. There is no folder called "Processed". The final destination folder is called "Analyzed" (config.py:123). Separately, "processed" is a status word the watcher stores in its own small tracking database to mean "the algorithms finished on this video"; it says nothing about where the files are (watcher/orchestrator.py:1174 and :2110).

WHICH MACHINE DOES WHAT

Each machine has a file at ~/.mousereach/config.json with a "watcher" section. Two settings decide the behaviour here (config.py:676, :678):
  - watcher.mode = "processing_server" runs the server orchestrator; anything else (default "dlc_pc") runs the GPU-machine orchestrator (watcher/cli.py:298-303).
  - watcher.also_process = true tells a GPU machine to run the analysis algorithms itself rather than handing the video to the server. Default is false.
The two roles reach the finish line by slightly different code, and the difference matters.

THE GATE - WHAT COUNTS AS SUCCESS

After segmentation, reach detection, outcome detection and reach assignment have run, the code takes a snapshot of provenance, runs a quality check, and then calls a single decision function (watcher/review_gate.py, called at watcher/orchestrator.py:2047 and :1132). It returns one of three answers:
  - deep review, if segmentation failed or the quality check raised something critical;
  - triage, if any flagged item has not been answered by a person;
  - clean, otherwise. A video with exhaustive human ground truth is treated as clean regardless of what the algorithms flagged.
On deep review or triage the whole file bundle is physically moved out of the working folder into a review queue and the code stops there - no kinematics, no database. Only "clean" continues. That part of the description is accurate.

WHAT HAPPENS ON "CLEAN", IN THE ORDER IT ACTUALLY HAPPENS

1. Kinematic features are computed and written as {video}_features.json into the working folder the video is currently in - NOT into Analyzed (watcher/orchestrator.py:2088-2094 and :1160-1166). If a human review file exists it is found and applied at this point.
2. The features file is pushed straight into the shared database from that working folder (watcher/orchestrator.py:2095-2096 and :1167-1168). The database is <central_db>, table reach_data, one row per reach (sync/database.py:41).
3. The video is marked "processed" in the watcher's tracking database.
4. Later - on a subsequent pass of the watcher's work loop, as a separate job - the files are moved to Analyzed (watcher/orchestrator.py:1476-1487 on the server, :512-525 on a GPU machine).

So the database is written BEFORE the files are filed, not after. This is the reverse of the described order, and it is the same on both machine roles.

THE MOVE TO ANALYZED

The move collects every file in the source folder whose name begins with the video id - the .mp4, the DeepLabCut pose files, all algorithm .json files, the provenance manifest, the triage file, and any ground-truth or human-review files sitting alongside - and moves them as a group (archive/core.py:146-148, :183). The destination is Analyzed/{project folder}/{cohort folder}/, worked out from the animal id in the filename by the lab's filing rules (the "filing" section of the lab settings file; see "Filing folders come from the lab settings" below): with the rule {"folder": "GrantA/PROJA", "cohort_folder": "{CODE}_{NN}"}, PROJA0304 goes to Analyzed/GrantA/PROJA/PROJA_03/. A project with no rule goes to Analyzed/<CODE>/<CODE>/ (archive/core.py filed_folder, config.py AnimalID.get_project_and_cohort). An unparseable id lands in Analyzed/UNKNOWN/UNKNOWN/.

The two roles call this differently, and it is the main practical difference between them:
  - GPU machine with watcher.also_process = true: the move is called with the readiness check switched off, so it just runs (watcher/orchestrator.py:1204-1208). Files come from the machine's local DLC_Queue folder, and leftovers there are deleted afterwards.
  - Processing server: the move is called with the readiness check switched ON (watcher/orchestrator.py:2133). That check consults a cached index file and demands that segmentation, reach detection and outcome detection each be recorded there as "validated" or "auto_approved" (archive/core.py:56-79, index/index.py:426-441). The watcher daemon never writes those records: segmentation's single-video entry point does not touch the index (segmentation/core/batch.py:65-115), the current outcome detector writes its results file directly without touching it (outcomes/core/batch.py:209-211), and reach detection records "needs_review" by default (reach/core/reach_detector.py:1104, reach/core/batch.py:168). The index is only refreshed when someone opens the napari front end or the dashboard (launcher.py:216-228). When the check fails, the move is logged as failed-will-retry and is not treated as an error (watcher/orchestrator.py:2160-2165) - so the video can sit in the local working folder indefinitely with its results already in the database.

If the move does succeed, the watcher marks the video "archived", copies its processing log into a shared audit database on the network drive called watcher_central.db (a provenance log, a different file from connectome.db - watcher/db.py:1200-1220), releases its multi-machine claim, and deletes the staged copies from the network staging folder.

HOW THE DATABASE WRITE WORKS

For each video the syncer flattens the features file into one row per reach and then, in a single transaction, deletes every existing row for that video name and inserts the new ones (sync/database.py:613-631). Nothing is archived first. There is no history table, no archive table, and no database trigger preserving the old rows (sync/database.py:372-410). A flat CSV dump of the whole table is rewritten in place afterwards (sync/database.py:746-795). So a reprocessed video's earlier numbers are simply gone.

The database push is deliberately silent about failure. It returns false and does nothing at all if the file is not a features file, if the animal id cannot be parsed, if the database is unreachable, or if the animal is not already listed as a known subject in the database; it never raises (sync/database.py:796-841). The server logs that as a debug-level "skipped" line. A video can therefore complete successfully with zero rows written and no visible complaint.

Two more things worth knowing. Videos on the Easy or Flat trays skip outcome detection entirely, which means they also skip kinematics and the database, but they still reach "processed" and still get moved to Analyzed (watcher/orchestrator.py:1938-1944, :1060-1063). And the watcher separately writes the video's pipeline state (not its kinematics) into connectome.db as it goes, including before the move to Analyzed (watcher/orchestrator.py:404-411, watcher/coordination.py:200).

THE ONE PATH THAT DOES MATCH THE DESCRIPTION

A separate tool, the "bring current" reprocessor, exists to re-run old videos on the current algorithm versions. It behaves exactly as the description says the pipeline should: it checks the same gate; if the video is held it stages a review bundle and stops; if the video is clean it first moves the previous outputs into a version-stamped Archive tree with checksum verification, then moves the new outputs into the video's folder under Analyzed, and only then - and only if the features file actually landed - pushes to the database (pipeline/reprocess_to_current.py:280-320). That file-level archiver never archives the video itself or the human ground-truth and human-review files, which stay with the video permanently (archive/supersede.py:20-33). Note that this archiving is of FILES only; the database rows it replaces are still hard-deleted. The watcher never calls this archiver.

BACKUPS

A periodic job copies the configured folders (backup.sync_dirs, or the built-in default list) from the source root to a backup location using robocopy in add-only mode - newer files are copied over, deleted files are left in place on the backup; claim folders, liveness reports, half-written files and SQLite rollback journals are never copied (see the update below on what the backup leaves out) (watcher/backup.py, `DEFAULT_SYNC_DIRS` and `_sync_dir`). That preserves a copy of the connectome.db file, but it is a whole-file disk backup on a timer, not a record of which rows a given sync replaced.

---

---

## Segmentation now says when it forced the answer (added 2026-08-21)

The segmenter emits exactly 21 boundaries every time. That count is hard-coded
and guaranteed by a safety net, so a forced segmentation and a measured one
looked identical downstream, and the count was never evidence that anything
worked. (Since 2.2.4 the boundaries are also guaranteed strictly increasing:
duplicates -- which became zero-length segments and self-generated triage work
-- are deduped and re-projected, recorded in `anomalies` and `needs_human`.) Reviewers hit the consequence directly: the algorithm's outcomes were
right and the segment NUMBERING was wrong, drifting by one and then two through a
single video, which made bench pellet 7 get compared against footage of pellet 8.

Three changes:

- `segment_video_multi` now keeps every candidate timepoint it considered, used
  or not, in a `candidates` list in `{video}_segments.json` -- frame, which of
  the four tray corners proposed it, how strongly they agreed, and whether it
  became a boundary. These were previously discarded, so when the chosen
  boundaries were wrong nobody could see what the alternatives had been.
- It also writes `needs_human`: its own account of why the boundaries want
  checking. It is non-empty when boundaries were invented at the median cadence
  to reach 21, when real detections were discarded to fit that count, when
  boundaries were interpolated or fell back rather than being detected, when
  reference tracking was not `good`, or when three or more detected candidates
  went unused. Empty means the boundaries were found rather than forced.
- `needs_human` is RECORDED ONLY. Nothing routes on it. It was briefly wired
  into the review gate and was switched off the same day: the rule fired on
  a sizeable share of ordinary videos, and checked against videos a human had
  actually judged mis-segmented it caught few of them. Most of those had textbook
  segmentation output, because they were offset rather than malformed, and
  nothing measurable inside a single video can see that. It never routed a
  video before being disabled. The established route remains a person noticing
  during review and pressing "Flag Session", which works.

`mousereach-fix-segmentation` is the tool for that queue. It does segmentation
and nothing else: it lists the candidate timepoints, lets you take or drop cuts
and add one at the current frame, shows the segment lengths so a missing cut is
visible as an over-long segment, and on save archives the original, records the
algorithm's cuts alongside the corrected ones, stamps who corrected it, and
clears `needs_human` so the video moves on.

---

## Update 2026-08-23: every watcher command reads the same database

The daemon honoured the node's `db_path` config override; the seven other
commands -- status, reprocess, quarantine, process-animal, version-check,
crystallize, uncrystallize -- hardcoded the fallback and could silently read a
stale database. Crystallize, the brake that protects published videos from
reprocessing, would then have found no videos and protected nothing. All eight
sites now resolve through one helper that loads the node config and prints the
path it chose, so a wrong database is loud.


---

## Update 2026-08-24: a row never claims a file the machine does not have

Three failures on a GPU node, one cause.

The staging step crashed with `TypeError: expected str, bytes or os.PathLike
object, not NoneType`. Videos sat in the node's database with no
`current_path` and no `dlc_output_path`, churning. Collages could not enter the
pipeline at all.

All three came from cross-node recovery. On startup a node reads
`pipeline_videos` and `pipeline_collages` out of connectome.db and copies the
states it finds into its own database. Those tables carry state and no usable
file path -- every row in `pipeline_videos` had
`source_path` NULL, because `sync_video_state` was an `INSERT OR REPLACE`, which
is a DELETE plus an INSERT: every column the caller did not name was reset to
NULL, and callers name a state and a timestamp, never a path. Recovery
substituted the string `'recovered'`, adopted the remote state anyway, and the
node ended up with videos in `dlc_complete` that it had never held. The work
loop picked one up every cycle and `Path(None)` raised. For collages the
substitute was the owning HOSTNAME, which `_process_collage` passed to
`Path().exists()`; that always failed, and because the row existed, the intake
scan skipped the file forever -- the collage was locked out with the .mkv
sitting in the intake folder.

What changed:

- `sync_video_state` is an UPSERT. It updates the columns it is given and
  leaves the rest alone, so a path written once survives later state syncs.
- `watcher/locate.py` is the one place that answers "is this claimed path a real
  file, and if not, is the file anywhere on this node?". It holds
  `resolve_pose_input` (moved from orchestrator.py) plus `locate_video_file` and
  `locate_pose_file`. Everything there returns a real file or None -- never a
  placeholder, never a directory. Handlers that MOVE what they find, or write
  beside it, pass `search_archive=False`, so a hit in the archive can never make
  the stager empty the archive or DLC drop a pose file into it.
- Recovery adopts an in-flight state only if this node actually has the file.
  Otherwise the row is recorded `unresolvable` with the owning host and its state
  in `error_message`. A collage this node cannot see is not registered at all,
  so the intake scan can register it properly the moment it can.
- `unresolvable` is a new terminal video state. It is deliberately not `failed`:
  `failed` is a retry state, and it reads to people as "something went wrong with
  this animal's data". Nothing went wrong -- the file is on another machine. The
  work loop never selects it, `error_count` is not incremented, and
  `_recover_local_dlc_queue` puts the video straight back into the pipeline if
  its file ever appears on this node.
- `register_video` no longer requires a source path. Given none, it looks for the
  video on this node; found, the real path is recorded, and not found, the row is
  created `unresolvable`. The `NOT NULL constraint failed: videos.source_path`
  error cannot happen from any caller.
- `discover_new_collages` repairs a collage row whose recorded path is not a
  file, which unblocks the collages already poisoned by the old behaviour.
- `mousereach-watch-unresolvable` lists, sweeps and retries. `--sweep` moves
  existing pathless rows out of the work loop; nothing is deleted.

Separately, the DLC-staleness rule was too blunt. A manifest naming an older
scorer set scope 'full', which meant "put it back on a GPU". Most such videos
can already have the declared pose sitting in the archive (for example in a
`DLC Model <generation>/` tree from a bulk pose job); re-posing them would spend
GPU-hours regenerating files that exist. The scanner now
indexes the archive's pose files once per scan and downgrades those videos to
scope 'segmentation' -- every post-DLC stage re-runs against the pose already
there. Videos with no current pose keep scope 'full'. Their manifests are not
edited: the manifest correctly records what produced the current results, and the
reprocess run rewrites it with the pose that actually ran.


---

## Update 2026-09-14: which watcher database is authoritative, and one that is not

A check of whether a large set of videos still needed a new pose found that
they did not. Their manifests recorded the declared scorer and the current version
of every tracked stage, and the processing node's own live database held no
'outdated' rows at all. The check took far longer than it should have, because
two separate places make a stale or empty database read like live state. Neither
is described anywhere else, so both are recorded here.

THE PER-HOST FILE UNDER watcher_state IS A BACKUP, NOT STATE

`nas_root/watcher_state/<hostname>/watcher.db` is written by `backup_db`
(watcher/coordination.py:175) as a plain `shutil.copy2` of that node's local
database, through a `.tmp` file and a rename. Exactly one caller reads it --
`restore_db` (:208) -- and only when the local database is missing or smaller
than 4 KB, which means a fresh or wiped node. Nothing else in `src/` or `tests/`
refers to `watcher_state` at all.

So it is a restore safety net with no defined freshness. `_backup_local_db`
(watcher/orchestrator.py:801) runs opportunistically inside the work loop
(:1262), not on a timer, so the file's timestamp records when that node last
completed a cycle -- not when its contents were true.

Reading it as pipeline state is a mistake, and an expensive one. A GPU node's
backup can hold many videos in 'outdated' (many at scope 'full', which reads as
"needs a new pose each") while the processing node's live database holds zero
outdated rows, and the manifests for those same videos record the declared
scorer and current versions throughout.

The authoritative database for a node is the one that node's own watcher writes:
the `db_path` override in its `~/.mousereach/config.json`, else
`processing_root/watcher.db`. For reprocessing questions the processing node's
copy is the one that decides, because it is the only role with
`reprocesses_partial` true (orchestrator.py:433; the GPU role sets it false at
:728 and therefore passes `full_only=True`, marking only the scope it can
drain). Note that `handles_reprocessing` does NOT distinguish the roles -- the
GPU role sets it from `also_process` (:756), which is true on any GPU node that
also processes.

A gap worth recording: such a backup has also been seen holding rows at scope
'kinematics', which `full_only=True` should have prevented that node from
marking. Whether they predate the flag or arrive by another path is not
established here.

THE DASHBOARD STILL HARDCODES THE PATH THE CLI STOPPED HARDCODING

The 2026-08-23 update above moved all eight watcher commands onto
`_resolve_db_path`, which honours the node's `db_path` override and prints the
path it chose so a wrong database is loud. The dashboard was never included. It
resolves correctly in one place (dashboard/widget.py:892) and hardcodes the
fallback in four others: :365, :593, :1284 and :1415.

On a node whose override points at, for example,
`processing_root/watcher_local.db`, the hardcoded `processing_root/watcher.db`
sitting beside it can carry the full schema and zero rows. The GUI then opens an
empty database while the watcher writes a full one -- the same failure the CLI had, silent, and indistinguishable on screen
from "there is nothing to do".


---

## Update 2026-09-14: `mousereach-reconcile` checks videos against "done", from files only

A new read-only command (watcher/reconcile.py) is the first step of
DESIGN_FILESYSTEM_AS_STATE.md. It judges every single-animal video on the share
against the definition of done -- analysis current or declared compatible, with
every saved human review reflected, AND sitting beside the video in
`Analyzed/<project>/<cohort>/` as `archive/core.get_archive_destination` works it
out from the name -- and lists each mismatch per video. Exit 0 means none, 1
means some, 2 means it could not judge (no share configured, or no
`pipeline_versions.json`).

WHAT IT READS, AND WHAT IT DOES NOT

Folders only: the review queues (`bundles_in`, date-named directories only),
each queue's build folder `<queue>/.incoming/` (see the routing update below),
Failed, Quarantine, the two folders a single waits in, read from `Paths` when the
check runs (`Unanalyzed/Single_Animal`, `Processing/Posed`), every leftovers folder
at `Processing/_leftovers_pending_cleanup_*/*`, and one
pruned walk of `Analyzed` that skips `DLC Model*`, `Multi-Animal`, `Archive`,
`Folder Template`, `UNKNOWN` and anything starting with `.` or `_`. It never opens
a watcher database, so a stale or empty database cannot mislead it, and it writes
nothing.

Currency is not re-implemented. It calls the version scanner's own rules
(`compare_manifest_to_current`, `ReprocessingScanner._drop_human_seg_staleness`,
`ReprocessingScanner._pending_review_path`), so the check and the watcher cannot
disagree about what "current" means.

WHAT COUNTS AS A MISMATCH

  * not_current -- analysis in the right place but outdated, missing a recorded
    version, or missing a saved human review.
  * wrong_place -- analysis in a different folder than the name says, a second
    copy elsewhere, or a current analysis with no video beside it.
  * stray_review_bundle -- already done, but a bundle still sits in a queue.

Not mismatches: held in a review queue, in Failed or Quarantine, waiting in a
pre-analysis folder, or an unsupported tray. A copy in a leftovers folder (set
aside when the folder layout changed) of a video that exists elsewhere is a
leftover pending cleanup; a video whose ONLY copy is there is listed per video,
because it may be real work.

Collages are not judged yet.


---

## Update 2026-09-14: no walk of Analyzed reads superseded outputs

Superseded outputs keep their original names (`{stem}_features.json`,
`{stem}_processing_manifest.json`, `{stem}DLC_...h5`), so a plain `rglob` over
Analyzed cannot tell an older generation from the live one. Every consequence
was silent: the version scan could read a superseded manifest and mark a
current video outdated, a returning review bundle could be re-run on an
old-model pose, `mousereach-route-to-queue` could write flags into an archived
file and move an archive folder into a review queue, and the cohort reach
export could gain rows from old generations.

Every walk of the Analyzed tree now goes through `pipeline/analyzed_tree.py`
(`iter_files`, `first_file`, `is_superseded_dir`) and never enters a folder
named `Archive` (the superseded-output root) or `_archived` (MouseReach's own
pre-modification backups, e.g. from the manifest backfill CLI).

`DLC Model <N>/` folders are deliberately still walked. They are live pose
storage: a finished video's current pose can sit only there, and the version
scan and the pose lookups depend on finding it.

WHAT WAS SWITCHED

The version scanner (pose index, manifest and features index, the per-video
fallbacks, the mislabel divert to deep review), the reprocess pose lookup in
the orchestrator, the review-return fallback lookups, `mousereach-route-to-queue`,
the dashboard's folder scan and manifest index, the census walk, collage
provenance, the manifest backfill CLIs, the field audit, the cohort reach
export, reprocess-to-current, the fix-segmentation video lookup, and the ASPA
feed, importer and sync cohort listings. `mousereach-reconcile` shares the same
constant. Depth-limited globs such as `*/*/name` are filtered as well, because
`Analyzed/Archive/<folder>/<file>` is exactly two levels deep.

An unlistable folder behaves as it did under `rglob`: a PermissionError is
skipped, anything else (a network error, for instance) raises. `os.walk`'s
default would have skipped it silently and returned a partial answer that
looked complete.

The manifest backfill CLIs now put their backup copies under
`<NAS_ROOT>/_archived` whatever `--root` is; with `--root Analyzed/<project>`
the old default put them inside Analyzed.

The archive root itself moved under Analyzed in the stage-folder change below;
contract tests in `tests/test_analyzed_tree.py` tie the skipped name to
`archive.supersede.default_archive_root`.


---

## Update 2026-09-14: stage folders -- the folder a video is in says where it is

The shared folder now follows DESIGN_FILESYSTEM_AS_STATE.md. Three stage folders
were renamed, and the superseded-output archive moved under the finished tree:

    Unanalyzed/Single_Animal/       was Processing/Single_Animal             (Paths.SINGLE_ANIMAL_OUTPUT)
    Processing/Posed/               was Processing/DLC_Complete              (Paths.DLC_STAGING)
    Processing/Review/deep_review/  was Processing/Review/flagged_for_review (Paths.DEEP_REVIEW)
    Analyzed/Archive/               was the top-level Archive/ for superseded outputs
                                    (archive.supersede.default_archive_root)

WHY: a folder names a place work rests and what it is waiting for. A cut single
is waiting for a pose, not being worked; a posed video waits for the algorithms;
the deep-review queue sits beside triage under the name of what it is. A
top-level Archive/ beside Unanalyzed/ and Processing/ read like a stage.

`pipeline/pipe_structure.TARGET_DIRS` builds exactly this skeleton and never
creates a retired name; `tests/test_stage_layout.py` holds it and `config.Paths`
together. `pipeline/pipe_migrate.py` is gone: once the paths were repointed its
map would have moved live waiting work back into the retired folders.

GUARD FILES

On a migrated share a plain FILE sits at each retired folder name. Old code that
tries to create or list one raises (FileExistsError, NotADirectoryError) instead
of silently rebuilding the folder and working where nothing looks. Listings that
may meet one read it as empty (`census.runner.ids_in_dir` and `bundles_in` use
`is_dir()`). The copies left in the old folders were set aside under
`Processing/_leftovers_pending_cleanup_<date>/` until cleanup;
`mousereach-reconcile` reads them as leftovers, never as a stage.

AN UNMIGRATED SHARE IS NOT SILENT

Where a retired name is still a real FOLDER (a share that was never migrated),
`mousereach-watch` refuses to start, `ensure_pipe_structure` reports it under
`failed`, and `mousereach-reconcile` prints a warning before anything else and
exits 1. All three check by stat only and never list the folder.

REVIEW BUNDLES

Bundle manifests store absolute paths, which go stale when a folder above the
bundle is renamed. The review tool resolves the video and pose through
`review/bundle_media.py`: the manifest path when it is a file, else the bundle's
own copy. A new manifest never points at pose staging: when the only pose found
is the staged one, `review_gate._write_review_manifest` copies it into the bundle.

A top-level `Archive/` may still hold `historical/` (read-only source material,
the ASPA collage importer's default source).


---

## Update 2026-09-14: routing into a review queue -- built out of sight, returned with its provenance

The first real run of `mousereach-route-to-queue --worklist` (an integrator's
bench-vs-pipeline disagreement scan, which moves videos from Analyzed into a
queue) exposed three defects. What changed, and why:

A QUEUE BUNDLE APPEARS IN ONE RENAME

`review_routing.move_video_bundle` builds a NEW bundle in
`<queue>/.incoming/<stem>/` and publishes it as `<queue>/<stem>/` with one
directory rename. WHY: it used to create the empty `<queue>/<stem>/` first and
move the files in one at a time, and a move onto the share is a copy that takes
seconds per file. The running watcher's return scan saw the empty folder,
retired it to `_Problematic` mid-route, and every remaining move failed with
"No such file or directory". A partly filled folder could also be judged as a
bundle (`triage_status`) before all its files had arrived.

WHY a dot-folder under the queue root: the queue readers that judge, retire or
release bundles (`review_return._bundles`, census `bundles_in` and
`review_completeness`, dashboard `folder_scan`, `release_cli`, `queue_index`,
`reconcile`) skip names starting with ".", so a bundle being filled is neither
an empty folder to retire nor a bundle to judge; and staying under the queue
root keeps the rename on one volume, where a directory rename is a single step.
Other readers of the queue roots (`collage_provenance`, `reprocess_to_current`'s
skip list, two review widgets) do not filter dot-folders; they key on
stem-named files or folder names, so `.incoming` matches no video there by
accident, not by rule. A new reader must add the filter.

Only `move_video_bundle` builds out of sight.
`reprocess_to_current._stage_review_bundle` (the bring-current path) still
creates `<queue>/<stem>/` first and fills it file by file; the 30-minute
empty-folder grace below covers it only while it is completely empty.

  * A bundle that already exists (a re-route, or a divert into an existing
    bundle) is already visible, so files move straight into it as before.
  * If the rename fails (normally because the same stem's bundle appeared
    meanwhile), the built files are moved in one by one -- same-volume, so
    milliseconds -- rather than left in the hidden folder where no queue reader
    would ever see them.
  * Two routes of the same video share `<queue>/.incoming/<stem>/`. If one
    publishes it while the other is still filling it, the other moves its
    remaining files, and writes its routing manifest, into the published
    bundle. WHY: its moves used to fail and its manifest write raised out of
    the route, losing its record and leaving files in their source.
  * Once files have moved, the publish is always attempted, even when writing
    the routing manifest into the build folder fails; the manifest is then
    written into the published bundle (with the retrying JSON writer,
    `fsutil.dump_json_with_retry`). WHY: that exception used to skip the
    publish and strand every file -- the video's only copy -- in the hidden
    folder, while the caller marked the video failed with nothing left in
    Processing to retry.
  * A file that could not be moved in the fallback stays in
    `<queue>/.incoming/<stem>/` beside the published bundle (never deleted),
    and `review_return._return_to_processing` refuses to return that bundle,
    naming the folder, while the build folder holds files. WHY: the file most
    likely to be stuck is a locked mp4; later routes write straight into the
    visible bundle and never read staging, so returning would re-run the video
    without it. `mousereach-reconcile` adds a note naming the leftover to that
    video's queue verdict.
  * The routing manifest `<stem>_routing.json` records `failed_files` beside
    `moved_files`: each file that could not be moved, with its error. WHY: a
    part-moved bundle used to be a log warning only, invisible in the record the
    review tool and audits read.
  * A route that dies partway leaves its files in `<queue>/.incoming/<stem>/`,
    and the next route of that video reuses them. `mousereach-reconcile` lists
    such a video as `held_for_person`, found in `incoming:<queue>`, with "being
    routed into <queue>, or an interrupted route -- check if this persists". A
    route takes seconds, so the same video there on a later run means a route
    died.

AN EMPTY QUEUE FOLDER IS LEFT ALONE FOR 30 MINUTES

The return scan still removes an empty triage folder (residue of an earlier
divert or return; earlier versions filed it away to `_Problematic`), but
only once the folder's
modification time is 30 minutes old (`review_return.EMPTY_DIR_GRACE_SECONDS`).
WHY: "empty" is not "abandoned". A router may have just created the folder and
still be moving files in (a bundle that already exists, and a failed rename, are
filled in place), and the share stamps that time with its own clock, which can
differ from the scanning node's by many minutes. So age is judged
conservatively: a time in the future counts as young, and so does a folder whose
time cannot be read. Real residue just waits for a later scan; an empty folder
holds no work. The deep-review loop never retires empty folders (an empty folder
has no clear marker, so it is never acted on).

The same scan names, in a warning once per watcher run, any
`<queue>/.incoming/<stem>/` older than 30 minutes: "being routed or an
interrupted route; a person should check". It never moves or deletes one. WHY:
those files are the video's only copy (it has already left where it came from),
and whether a route is still running or died half-way is for a person to judge.

RETURNING A CLEARED BUNDLE KEEPS ITS PROVENANCE AND ITS CHOSEN POSE

`review_return._return_to_processing`:

  * Queue-only metadata is matched by exact name (`<stem>_manifest.json`,
    `<stem>_routing.json`, `manifest.json`). WHY: the old suffix match on
    `_manifest.json` also caught `<stem>_processing_manifest.json`, the video's
    provenance record, and deleted it from every returned bundle. Stage reuse
    is decided from the stage outputs, never from this file, and a normal re-run
    regenerates it before the review gate. When that regeneration FAILS, the
    pipeline now moves the old manifest into `<processing dir>/_stale_manifests/`
    (`_set_aside_stale_manifest`, logged as an ERROR) instead of leaving it to
    be stamped by kinematics and archived as this run's provenance. WHY: a
    stale manifest makes reconcile read the video as current; with none, the
    video reads as not current and a person sees it. Still open: a real
    segmentation failure is routed to deep review before the manifest is
    regenerated, so that re-routed bundle carries the previous run's manifest.
  * The pose recorded for the re-run is the one `_resolve_inputs` selected
    (`select_pose_file`). WHY: it used to be whichever `.h5` the move loop moved
    last, so a bundle holding two pose generations re-ran on the wrong, often
    older, one.
  * Gate-routed bundles are self-contained (the mp4 and pose move in with the
    algorithm outputs); only legacy staged bundles point at an mp4 and pose left
    in Analyzed. The docstring had said the opposite.

MOUSEREACH-ROUTE-TO-QUEUE WRITES THE WATCHER'S OWN DATABASE, OR NOTHING

`route_cli.main` opened a bare `WatcherDB()`, which defaults to
`<processing_root>/watcher.db` and ignores the node's `db_path` override. Every
state write went to an unused decoy database while the bundles moved on disk
("Disk and DB now disagree"). It now resolves the database the way the daemon
does (`db_location.resolve_watcher_db_path`: the configured `db_path`, else
`<processing_root>/watcher.db`), prints `[watcher db] <path>` (to stderr with
`--json`), and REFUSES to route -- exit 1, nothing flagged, nothing moved --
when that file does not exist or cannot be opened. WHY refuse: routing on disk
only leaves disk and database disagreeing for every video in a worklist, and
opening a missing file would create an empty database, which is just a new
decoy.

A video the daemon is working on is DEFERRED, not routed: watcher state
`dlc_queued`, `dlc_running`, `dlc_complete`, `processing`, `processed` or
`archiving` (`route_cli.IN_FLIGHT_STATES`), checked before any flag is written.
It prints `wait`, carries `deferred: true` with `--json`, and does not make the
exit code 1. WHY: the command now writes the daemon's live database. Setting
'triage' underneath a running pipeline makes the pipeline's own 'processing' ->
'processed' write an illegal transition, and the daemon marks the video failed
with fresh outputs in Processing and old outputs in the queue; a 'processed' or
'archiving' video's Analyzed copy is older than what the daemon is about to
archive. WHY not exit 1: an integrator treats a failing exit as an
error, and a video that is merely busy should just be offered again on its next
run.

THE ARCHIVE STEP WAITS FOR A BUNDLE STILL BEING BUILT

`_archive_to_nas` already skips a video whose bundle sits in a queue on disk and
adopts the queue state. It now also skips -- without touching state -- a video
with a `<queue>/.incoming/<stem>/` folder. WHY: the queue check looks for
`<queue>/<stem>/`, which only appears when the route's final rename runs; until
then, archiving would file the leftovers while the rest of the bundle is on its
way into the queue. The next cycle sees the published bundle, or the route
failed and the files are where they were.

THE ARCHIVE STEP ADOPTS A QUEUE FOLDER ONLY IF IT HOLDS THIS RUN

A folder named after the video in a review queue is not always that video's
bundle. When a return cannot delete a file the review tool still has open (the
.mp4, typically), a husk holding just that file stays behind in the queue.
`_archive_to_nas` used to adopt the queue's state for any such folder; the clean
re-run that followed the person's answers was then forced back to 'triage', the
return scan read the husk as a failed segmentation, and the video went to deep
review as a bare .mp4 for a person to clear a second time. Now (`queue_folder_holds_this_run`) the folder counts as this
run's bundle only when this run's results are no longer in Processing (every
route moves them out together, the processing manifest among them -- this is
what the bench-disagreement router does), or when the folder's routing record,
written after this run, lists the processing manifest as a file it failed to
move. Otherwise the run is archived, a warning names the leftover, and the
return scan files the leftover away. A routing record that cannot be read keeps
the old, cautious behaviour.

THE REVIEW MANIFEST NAMES THE CHOSEN POSE

`review_gate._write_review_manifest` records `select_pose_file` over the bundle's
`<stem>DLC*.h5` files as `canonical_dlc_h5_path`, not the first name sorted.
WHY: the review tools trust that pointer before choosing for themselves, and a
bundle holding two pose generations sorted the OLD model's file first.


---

## Update 2026-09-15: extra GPU nodes cannot re-pose finished videos, spin on collage claims, or touch Processing/Posed

Several GPU nodes (watcher.mode "dlc_pc", `DLCOrchestrator`) share the collage
intake folder Unanalyzed/Multi-Animal. One processing server
(`ProcessingOrchestrator`) is the only node that takes posed videos in from
Processing/Posed (`Paths.DLC_STAGING`). Three defects had to be fixed before
more GPU nodes could join. Each is described below: what the code does now, why,
and what an operator sees in the watcher log.

FINISHED OR HELD WORK IS NEVER POSED AGAIN

Before, only adopting a video from Unanalyzed/Single_Animal asked whether it
was already finished. Cropping a collage and running DLC did not. A node with a
new, rebuilt or partial database therefore re-cropped and re-posed finished
children, wasting GPU time, and fed duplicate processing and archiving
downstream.

`DLCOrchestrator._already_done_or_held(stem)` now asks the shared drive, never
the node's own database, which is exactly what cannot be trusted here:

  * a bundle folder `Processing/Review/triage/<stem>` or
    `Processing/Review/deep_review/<stem>`, or `<queue>/.incoming/<stem>` (a
    route into that queue still being built), means a person holds the video.
    Posing it again would build a second result beside the one under review.
  * `<stem>_processing_manifest.json` in the video's archive folder
    (`archive.core.get_archive_destination`) means it was analysed and filed.

The queues are checked first, because a video held for review can also have an
older manifest in the archive; `_archive_to_nas` uses the same order.

It is asked at four points:

  1. Adopting a single from Unanalyzed/Single_Animal (as before, now including
     review holds). Asked before the single is claimed, so a video that needs
     no pose is never moved into .inflight (see "videos can be dropped into
     either Unanalyzed folder at any time", below).
  2. Before cropping. A child's name follows from the collage filename
     (`collage_provenance.expected_offspring`). If EVERY child is finished or
     held, the collage is not copied or cropped. Each child row is recorded in
     its real state, the collage becomes 'cropped' locally with
     videos_created = 0, and the shared claim is marked 'cropped'.
  3. After cropping, for each child. A finished or held child is recorded, not
     queued for DLC. A child whose row on this node has already moved past
     discovery (for example 'dlc_running') is left exactly as it is.
  4. Just before DLC inference, as the last line of defence for rows queued
     another way. The run is refused and the row is set to the state the drive
     shows. Exception: a re-pose request (mark_reason starting "re-pose
     request") for an archived or deep-review video is posed, because posing
     a finished video again is what the request asks for. A triage hold is
     refused, as the re-pose consumer refuses it.

A child recorded this way has source_path "(no file on this node)"; see
WATCHER_DB_DICTIONARY.md.

In the log (INFO): `<stem>: already analysed and filed in the archive;
recorded as 'archived', not queued for DLC`, and `Collage <name>: not cropped:
all N expected child video(s) are already analysed or held for review`. When a
refused video's copy is still in DLC_Queue, a WARNING names the file and says
it is not removed automatically. A hold is a queue folder (or the
`.incoming` folder of a route still being written) that holds at least one
bundle file; an EMPTY queue folder is residue, not a hold. WHY not delete the
copy all the same: the check only sees that the queue folder holds files, not
that they are this same video, complete -- and a crop is deleted only once a
copy elsewhere is verified. Delete such a file by hand to free the space (the
same leftover existed before; it is not closed here).

Re-pose requests: a GPU node copies 'triage' into its own rows, but only the
processing server's review-return scan ever moves a video out of triage. So a
GPU node's 'triage' row could outlive the hold and refuse every re-pose request
for that video until a restart. `repose.consume_requests` now re-drives a
'triage' row whose bundle is no longer in the triage queue
(`repose._held_in_triage_on_disk`). If that cannot be told (no queue
configured, an unreadable share), it keeps refusing.

COLLAGE CLAIMS FAIL CLOSED, EXPIRE, AND ARE GIVEN BACK

The claim is a row in the `pipeline_collages` table of watcher_central.db on
the shared root: the collage filename, the holding node, the state ('cropping'
or 'cropped') and claimed_at. It is not in connectome.db. Before, claims failed
OPEN: a claim error, or a coordinator that never started, cropped anyway. A
lost claim counted as progress, so the node re-picked the same collage at full
speed and copied its whole watcher.db to the share after each attempt. A claim
was never released or expired, so a collage whose crop failed on the claiming
node was blocked for every node, for good.

The rules now, each with its reason:

  1. No coordinator, no crop. If the coordination database cannot be opened at
     startup, collages are not even offered to the work loop (offering one only
     to refuse it would copy watcher.db to the share every poll). The database
     is tried again every 5 minutes, with no restart needed
     (`_COORDINATOR_RETRY_S`). Posing and staging carry on meanwhile.
     Log: WARNING at startup, then `Collage claims are unavailable on this node
     since <time> (<reason>)` every hour for as long as it lasts; INFO
     `Collage coordination database is available again` when it recovers.
     Cross-node recovery runs only at the next restart, because it rewrites
     rows and must not run under live work.
     WHY: a node that started while the share was still coming up used to crop
     every collage without a claim, and a restart was the only cure.
  2. A claim check that raises (share or database trouble) means no crop this
     poll. The collage stays 'stable' and is tried again next poll. Log: one
     WARNING per collage per outage, `could not check the shared claim`; a
     check that works in between re-arms it. WHY: an error is evidence of
     neither "mine" nor "another node's". Treating it as "mine" is how every
     node cropped the same collage whenever the share blinked.
  3. Claim held by another node:
       * that node FINISHED the crop (claim 'cropped'): the collage is recorded
         'cropped' here for good, and the processing log (step 'claim', status
         'skipped') names the node. INFO `already cropped by <node>`.
       * the claim is live: the collage stays 'stable' and is not offered again
         for 30 minutes (`_CLAIM_RECHECK_S`), then the claim is asked about
         again. INFO `claimed by <node> ... asked about again in 30 minutes`.
         The wait is kept in memory, so a restart asks once more straight away.
     In both cases the handler reports no progress, so the loop sleeps instead
     of spinning. WHY not park a live claim for good: that claim can still be
     given back after a failed crop, or go stale. With two or three GPU nodes,
     every node that had looked would otherwise never crop the collage.
  4. Stale claims. A 'cropping' claim held by another node with claimed_at
     older than 24 hours (`COLLAGE_CLAIM_STALE_S`, the same value and reasoning
     as `repose.STALE_S`) is taken over by a conditional update. Of two nodes
     racing for it, exactly one wins. Log: WARNING `took over the crop claim
     <node> left in 'cropping' ...`. Two safeguards:
       * A holder that re-uses its own 'cropping' claim restarts claimed_at.
         WHY: age is the only evidence. A holder that came back after a day and
         cropped under its old timestamp would otherwise be taken over mid-crop.
       * A stale claim is NOT taken over while any child of the collage has a
         row in `pipeline_videos`, matched by collage_id or by the child names
         the collage filename implies. Log: WARNING `... NOT taken over -- a
         second crop would pose them again`, repeated each time a node asks.
         WHY: a stale 'cropping' row does not mean nothing was cropped. The
         holder may have queued children and then failed, or finished and
         failed only to record 'cropped'. Its children then sit in its DLC
         queue, in Posed or on the processing server, where the finished-work
         check cannot see them. This is also why a cropped child is now synced
         to `pipeline_videos` as 'dlc_queued' the moment it is queued, not
         first at 'dlc_running'.
  5. After a failed crop the holder gives the claim back (deletes the row),
     unless this node carries any child of the collage: queued by this attempt,
     or left in flight here by an earlier one. Then the claim is kept, with a
     WARNING `crop failed while this node carries N child video(s) of it;
     keeping the shared claim`. The node retries its own failed collage on
     later scans (`discover_new_collages` re-validates a failed collage whose
     file is present, up to watcher.max_retries). A finished claim is never
     given back. WHY: once a child is on this node it will be posed here, and a
     second crop elsewhere would pose it twice.
  6. 'cropped' is written to the shared claim only while this node still holds
     it. A failed write is a WARNING (`the shared claim could not be marked
     'cropped'`) and is retried every poll until it lands. WHY: before, it was
     a debug line. Since claims can be taken over, a missed 'cropped' could
     otherwise let a finished collage be cropped again a day later. If the
     claim turns out to belong to another node by then, a WARNING says so (`no
     longer held by this node ... a person should check that its child videos
     are not posed twice`).

When a collage stays in the intake folder and no node crops it:

  * Look in each GPU node's watcher log for the collage name.
  * `NOT taken over` repeating means its holder left it half-done. If that node
    will run again, starting its watcher lets it finish. If it will not, a
    person first decides which children are still missing (every child already
    finished or held is skipped automatically anyway). Then delete the claim
    row with any SQLite tool, while no watcher is cropping that collage:
    `DELETE FROM pipeline_collages WHERE filename = '<collage file name>';`
    The next node that polls claims and crops it.
  * `claims are unavailable` means that node cannot reach watcher_central.db
    on the shared root.

A GPU NODE NEVER READS FROM, WRITES INTO, OR REASONS FROM PROCESSING/POSED

Posed is the processing server's intake. Before, GPU handlers could resolve
files INTO it, because the search list for "where can this video's files be"
included it:

  * adopting a single could copy a Posed mp4 into the node's DLC_Queue and
    pose it a second time;
  * DLC could run with its input in Posed, and DLC writes the .h5 beside its
    input, into the shared intake;
  * staging could find the video "already there" and force-mark it 'archived'
    on the strength of another node's file;
  * the local pipeline could take its pose or video from Posed, and a re-pose
    consumer could count a Posed copy as "already going here".

`watcher/locate.py` now has an opt-out. `node_search_dirs(include_staging=False)`
and `locate_video_file` / `locate_pose_file(search_staging=False)` leave the
folder out of the search, ignore a recorded path inside it (or beneath it), and
drop an `extra_dirs` entry that points there. `locate.is_in_staging(path)`
answers the question directly.

Every GPU-side caller uses the opt-out: adopting a single, DLC inference,
staging, both lookups of the local pipeline, the re-pose consumer's
`_local_file_exists` and `resolve_request_video` (which also refuses a request
whose recorded path points into Posed; the request waits and is served from
the archive), `repose.archived_video`, and cross-node recovery at startup. The
re-pose publish pass on a GPU node that also processes calls
`publish_pending(staging_dir=False)` (`reads_pose_staging` is False on
`DLCOrchestrator`). The processing server's own intake from Posed is unchanged.

Staging decides "already staged" by what THIS call staged, never by what sits
in Posed. The pose recorded on the row is the one this call moved there
(`_own_staged_pose`). A stage that was cut short leaves the mp4 in Posed and
this node's pose in its DLC_Queue; resuming it stages the leftovers and marks
the video 'archived' only on a pose it staged itself. With nothing of its own
left, the row becomes 'unresolvable': INFO `already in NAS staging
(Processing/Posed) and no file for it on this node; left for the processing
server`. The files in Posed are left untouched. WHY 'unresolvable' and not
'failed': nothing went wrong with the video (this node may even have staged it
fully and been stopped before recording that). 'failed' is a retry state that
reads as a verdict on the data, and its ERROR trace used to fill the failed
count with videos already safely with the server.

Every no-work path in the GPU handlers now returns False, so the main loop
sleeps for a poll interval instead of re-picking the same row at full speed.

Known gaps, not yet closed: `WatcherDB.register_video` called without a
source_path, and `mousereach-watch-unresolvable --retry`, still search with the
default folder list, which includes Posed. Neither runs in the GPU work loop's
normal path. [`--retry` is closed: see "Update: failed and unresolvable videos"
near the end of this document.]

---

## Update 2026-09-15: the watcher pauses itself while a recording program runs

WHY. A GPU node that is also a recording PC poses videos whenever nobody is
recording. A pose takes the GPU, CPU and disk for many minutes, and a recording that
drops frames because of it is behaviour that can never be filmed again, while a
pose can always be run later. A node cannot know a recording schedule; what it
can see is whether the recording program is open, and operators close it when
they are not recording. So the watcher pauses itself while a configured
program runs, and recording always wins. Code: `watcher/recording_guard.py`,
`dlc/core/interruptible.py` and `interruptible_worker.py`, and
`BaseOrchestrator` / `DLCOrchestrator` in `watcher/orchestrator.py`.

WHAT PAUSES THE WATCHER

Two independent causes, answered in one place (`recording_guard.pause_reason`)
so the watcher, the Watcher Control panel, the CLI and the dashboard health line
use the same words:

  1. The hand pause: `watcher_paused.flag` in the processing root, set by
     `mousereach-watch-toggle --pause` (or the plain toggle) or the panel's
     Pause button. Checked first. Reason text: `paused by hand
     (watcher_paused.flag)`.
  2. A recording program: any name in `watcher.pause_while_running`. EMPTY BY
     DEFAULT, and empty means exactly the behaviour from before: no process is
     ever listed and the pose runs in-process as it always did. Names match
     case-insensitively on the program name only; a pasted full path or a
     name without `.exe` still matches (`recording_guard._match_key`). WHY so
     forgiving: a name typed slightly differently from Task Manager must not
     silently fail to protect a recording. On Windows the process list comes
     from one system snapshot (`windows_process_table`); psutil elsewhere or as
     a fallback. WHY not psutil on Windows: it opens each process in turn, and
     one listing took over a minute and a half on a busy machine, while the
     check runs every few seconds. The list is re-read at most every 5 s.
     Reason text: `recorder.exe is running`.

If programs are listed but the check cannot run (no process list, psutil
missing, the listing raises), the reason is `cannot check for recording
programs: ...` and the watcher stays PAUSED. WHY: the one outcome this exists
to prevent is posing on top of a recording; an idle node is visible and
recoverable, a spoiled recording is not.

WHEN IT RESUMES

The hand pause ends when the flag is removed (`--resume`, Resume). A recording
pause ends only after every listed program has been closed for
`watcher.pause_resume_grace_seconds` (default 120). The timer starts at the
first check that finds none running; opening one again restarts it; after a
check that could not run, the grace period applies too, because that check may
have hidden a recording. Reason text meanwhile: `recorder.exe closed 30 s ago;
resuming after 120 s`. WHY a grace period: operators close and reopen the
program between animals, and a pose started in that gap would compete with the
next recording or be thrown away seconds later. Resume cannot override an open
recording program.

WHERE THE PAUSE IS CHECKED

  * At the top of every main-loop pass (as before, now with both causes).
  * Again after the scan, right before the chosen item is started. WHY: a scan
    can take many minutes on a shared drive, and a recording that started
    during it must stop new work before it begins. The item is not touched and
    stays in its state.
  * Inside the GPU scan, before re-pose requests are taken
    (`DLCOrchestrator._scan_phase`, phase C). WHY: taking a request copies a
    whole archived video over the network onto the node's disk. The heartbeat
    of claims already held still runs.
  * `run_once` (the panel's Run Once and `mousereach-watch --once`): before the
    scan and before every item, for the stop flag and for BOTH pause causes.
    This is a change: run_once used to ignore the hand pause and the stop flag.
    WHY the hand pause too: before recording programs could be listed, the
    hand pause was the only filming protection, and a Run Once that ignored it
    posed on top of filming. The panel checks first and says why it will not
    start ("paused by hand -- press Resume first", or the recording reason).

A POSE ALREADY RUNNING

  * No programs listed: the pose runs in-process through `run_dlc_batch`,
    exactly as before, and cannot be stopped part-way.
  * Programs listed: the pose runs in a child process, `python -m
    mousereach.dlc.core.interruptible_worker <args.json>`, through
    `run_dlc_single_interruptible`. The parent asks
    `DLCOrchestrator._recording_abort_reason` once before starting (a reason
    means the child never starts) and then every 5 s. On a reason it kills the
    child and everything the child started, waits for them to exit, and
    removes ONLY this run's pose files in DLC_Queue: `<stem>DLC*` files ending
    .h5, .csv or .pickle that were not in the listing taken just before the
    child started AND whose modified time is at or after that start (2 s
    allowance for coarse file-system clocks). An earlier finished pose of the
    same video, other videos' poses and any other file survive. WHY both
    conditions: cleanup must never be able to delete a finished pose.
  * The video then goes back to `dlc_queued` with its `current_path` unchanged
    (forced if the row had meanwhile moved to a state with no legal move
    there), a processing_log row step `dlc`, status `aborted`, message = the
    reason, is written, `dlc_queued` is synced to the shared record, and
    error_count is NOT increased and `mark_failed` is not called. WHY: the video
    did nothing wrong, and a busy recording week would otherwise use up its
    retries and park it as failed. The handler reports no progress, so the loop
    sleeps and then finds the watcher paused. Log (INFO): `<video>: pose
    stopped after N s because <reason>; partial output removed, back in the DLC
    queue (not counted as a failure)`.
  * A child that finishes by itself is kept even if a stop was asked for in the
    same instant. A child that exits without writing a result is a failed pose
    (as a DLC error always was), with the end of its output quoted; its log is
    kept in the system temporary folder only when the pose failed.
  * The hand pause does NOT stop a pose that is already running. Pressing Pause
    means "start nothing new"; only a recording, which cannot be redone, stops
    a pose part-way.
  * The child cannot outlive the watcher. On Windows it is put in a Job Object
    with "kill on job close", so the operating system kills it and its
    children whenever the watcher process exits, including a closed console
    window, an ended scheduled task or a crash. WHY: before, the pose ran
    in-process and died with the watcher; a windowless child would otherwise
    keep the GPU with nothing left to stop it when recording starts. Belt and
    braces, and the only mechanism elsewhere: when programs are listed, a GPU
    watcher's startup reclaim first kills any pose worker whose watcher
    process is gone (`interruptible.kill_orphaned_workers`), and only then
    returns `dlc_running` rows to the queue, so a second pose of the same video
    cannot start beside a leftover one.

WHAT IS NOT STOPPED PART-WAY

(Corrected 2026-09-18: a crop IS now stopped part-way -- see the update
"a crop stops too, and the node says when it is safe to record" at the end.)

An archive or staging copy already running finishes, and so does a local
pipeline run on an also_process node (segmentation, reaches, outcomes and
assignment together take a few seconds per video). The pause takes effect
after them. Only new work is blocked.

RE-POSE CLAIMS WHILE PAUSED

The scan, which heartbeats this node's claimed re-pose requests, is skipped
while paused. So the paused loop heartbeats them itself, at most once per poll
interval (`DLCOrchestrator._repose_heartbeat`, `_while_paused`); a failing
heartbeat only logs a WARNING. WHY: a claim with no heartbeat for
`repose.STALE_S` (a day) is handed back, and a long recording day would give
away work this node had already copied and queued.

SETTINGS APPLY WITHOUT A RESTART

`WatcherConfig.load` records the settings file and its modified time. On every
pause check and every 5 s poll of a running pose, the watcher stats that file.
When it has changed, `BaseOrchestrator._refresh_recording_settings` re-reads
ONLY `pause_while_running` and `pause_resume_grace_seconds`. A change of grace
alone keeps the guard and its timer; a changed list builds a new guard that
inherits the old one's grace timer (`RecordingGuard.inherit_history`), which
can only make the watcher wait longer. A file that cannot be parsed (for example
caught mid-save) changes nothing and gives one WARNING per change. Log (INFO):
`Recording-program settings changed: ...`. Every other setting still applies
at the next start. WHY: the panel, the CLI and the status commands all read the
file; a watcher that kept its start-up list would pose on top of a recording
while every screen said PAUSED. A pose or crop that was started in-process
before a program was first listed runs to its end, since it cannot be stopped.

WHAT A PERSON SEES

  * Watcher log, INFO: at start `Pauses while any of these programs run: ...`;
    once on entering a pause `Watcher PAUSED: <reason>. No new work starts;
    <what to do>.`; once when the cause changes kind (hand / recording program
    / cannot check) `Watcher still PAUSED, now because: <reason>.`; once on
    leaving `Watcher RESUMED: no longer paused (was: <reason>).`; and
    `Run-once stopped <when>: ...`. Never a line per poll.
  * Watcher Control panel: the fields "Pause while these programs are running"
    and "Resume after (seconds)" (saved by Save config, which keeps every other
    key and says right away whether a listed program is running, so a
    mistyped name shows); a label beside Pause/Resume, `Paused: <reason>` or
    `Watching for: <names> ...`; RUNNING also for a watcher started outside the
    panel, in which case Start and Run Once refuse.
  * Dashboard health line: when the watcher runs and is paused, a second line
    saying PAUSED, why, and what to do (`health.pause_line`).
  * Commands: `mousereach-watch-recorders` (--list, --add, --remove, --grace;
    edits only those two keys, atomically); `mousereach-watch-toggle --pause`,
    `--resume`, `--status` (the plain toggle still flips); a `Pause:` line and
    a `Stage folders:` block in `mousereach-watch-status`.
  * `health.watcher_running` (and the Restart processor button's targets) now
    match only the watcher itself (`health.is_watcher_command`). WHY: matching
    every command beginning `mousereach-watch` made a PC with no watcher look
    RUNNING whenever someone ran `mousereach-watch-status`.

Known gaps: a one-off status command or a freshly opened panel builds its own
guard, which cannot see the running watcher's grace countdown (it says "not
paused" while the watcher is still waiting; `mousereach-watch-status` prints a
reminder line). Two different programs with the same program name cannot be
told apart.

---

## Update 2026-09-15: videos can be dropped into either Unanalyzed folder at any time

The goal: people copy new videos onto the share straight off the recording PCs,
at any time, even while several GPU nodes run. Collages
(Unanalyzed/Multi-Animal) already allowed this: they wait until they stop
changing (`check_collage_stability`) and are claimed in the shared collage table
before one node crops them. Singles (Unanalyzed/Single_Animal,
`Paths.SINGLE_ANIMAL_OUTPUT`) did not, for two reasons:

  * `discover_new_singles` registered an mp4 as 'validated' the moment it
    appeared. A GPU node could copy, and pose, a file still being copied in.
  * `DLCOrchestrator._adopt_single_for_dlc` copied the file into the node's own
    DLC_Queue and LEFT THE ORIGINAL in the folder. Nothing ever removed it.
    Every GPU node without a row for that video (another node, or a rebuilt
    database) posed it again, unless an archive manifest or a review hold
    already existed.

WHAT THE CODE DOES NOW

(a) Stability wait (`WatcherStateManager._single_is_stable`). A single is
registered only once its size AND modified time have not changed for
`watcher.stability_wait_seconds`. The size rule is the collages' own
(`transfer.check_file_stable_quick`). The modified time is compared too because
some copy programs set the final size first and fill the file in afterwards.
The first sighting never counts. The sightings are kept in memory, because the
videos table has no size columns, so a watcher restart simply waits again.
Names starting with "." (a Mac's `._<name>.mp4` helper files) are never taken
in, and neither are temporary endings (.part/.tmp/.partial). A misnamed single
is quarantined only once stable. WHY: a file being copied in carries its final
name from the first byte, and moving it to quarantine mid-copy breaks the copy.

(b) The claim is a rename (watcher/single_claim.py). After the finished-or-held
check and before any copy, the node renames
`<Single_Animal>/<stem>.mp4` to `<Single_Animal>/.inflight/<machine>/<stem>.mp4`.
WHY a rename and not a marker file: a rename within one share folder is one
operation on the file server. Exactly one node's rename finds the file; every
other node's fails with "file not found". Check-then-write markers (the
processing server's `Processing/Posed/.claims` until 2026-10-22, now an
exclusive create) can let two nodes both believe they won. WHY a dot-folder under the same folder: a same-folder rename never
crosses a volume, and readers that list only the top of the folder never see a
claimed video as waiting a second time. The claim also sets the file's modified
time to now, because a rename keeps the old time and a video copied in last
week would otherwise look like a stale claim at once.

(c) What each outcome records, and what an operator sees:

  * claimed: row 'dlc_queued'; log INFO `<stem>: taken from the shared singles
    folder (held in .../.inflight/<machine> until handed on); adopted onto this
    node and queued for DLC`.
  * another node's rename won: nothing recorded against the row on that pass,
    one INFO `not taken from the shared singles folder`. On the next pass the
    row becomes 'unresolvable' with reason `gone from the shared singles folder
    before this node took it: <machine> took it for pose ...`
    (`single_claim.LEFT_FOLDER_REASON`). If the file comes back (see (e)), the
    intake scan validates the row again by itself.
  * the rename was refused while the file is still there (normally still being
    copied in; also a video player holding it open, or an account that may
    read the folder but not rename in it): the row stays 'validated' and is
    parked for 2 min (`_SINGLE_REFUSED_BACKOFF_S`), so the collages behind it
    are still cropped. INFO the first time. One WARNING naming the file and the
    likely causes once the refusal has lasted 10 min.
  * a same-named file is in the folder while ANOTHER node holds a claim on that
    name (someone copied the batch in again): not claimed, not posed. The row
    becomes 'unresolvable' with reason `a second copy of a video another node
    already holds for pose ...` (`single_claim.DUPLICATE_OF_CLAIM_REASON`),
    which is never re-driven, and a WARNING asks a person to delete the second
    copy. WHY: posing it would put two poses of one name into Processing/Posed.
  * the copy onto the node failed: the claimed file is put back in the folder,
    and the row is 'failed' with a reason starting "could not copy it from the
    shared singles folder into the DLC queue". The singles intake offers it
    again by itself an hour after the failure, while its failure count is below
    `watcher.max_retries` and the file is back at the top of the folder, stable
    and claimed by no node (see "A GPU node retries a single whose copy into
    its DLC queue failed" in "Update: failed and unresolvable videos"). When
    the retries are used up it stays 'failed': fix the cause (a full disk, a
    dropped share), then run `mousereach-watch-reprocess <stem>` -- with the
    file back at the top of the shared singles folder the reset puts the row
    to 'validated' and any GPU node takes it. [Before that update this node
    never retried such a row by itself.]

(d) The claimed file is removed only once the video's next copy is confirmed
(`_retire_claimed_single`): the staged mp4 in Processing/Posed (or, if the
server already took it in, the stage's own verified copy), the archived mp4
after a local archive, or the mp4 inside a review bundle. The next copy must
have the same size. Otherwise it is kept, with a WARNING naming it. A removal
is logged as processing_log step 'single_claim'. WHY never earlier: until the
video moves on, the claimed file may be the only copy on the share, since the
node's queue is on its own disk.

(e) Claims that end without the video moving on:

  * Every scan and every paused poll refreshes (touches) this node's claims
    whose rows are still being worked or were handed on
    (`_CLAIM_KEEPALIVE_STATES`). A claim whose row is failed, unresolvable or
    missing is not refreshed, and (since 2026-10-21) is given back to the
    folder at once (`_give_back_abandoned`, WARNING). WHY: refreshing a claim
    this node has given up on would hold the only shared copy from every other
    node for as long as this node runs.
  * A failed pose, or a queued or posed video with no local copy left, gives
    the claim back to the folder at once (`_release_claim_given_up`, WARNING).
  * After 3 pose failures in a row the node takes no new singles for 30 min
    (`_POSE_FAILURE_BRAKE`, WARNING). WHY: a node whose DeepLabCut is broken
    would otherwise claim and fail every single dropped on the share.
  * Every running GPU node sweeps for claims untouched for 24 h
    (`single_claim.STALE_S`, the re-pose request rule) -- or, since 2026-10-21,
    untouched for 3 h by a holder whose status report has also been quiet for
    3 h (see "Update 2026-10-21: a node keeps only what it starts") -- at most
    every 300 s, never while paused, and moves each back to the folder
    (WARNING). It never
    deletes and never overwrites. If a same-named file is already back in the
    folder, nothing moves, and a WARNING asks a person to keep one copy. That
    warning repeats at most once a day. Trade-off: a node that comes back after
    more than a day may pose a video another node has since taken.

(f) The processing server's `_rescue_misfiled_singles` still moves a pose-less
stray mp4 from Processing/Posed to the TOP of Unanalyzed/Single_Animal. It
skips any name a node holds in .inflight, and never moves anything into the
claim folder.

(g) Readers count a claimed single as waiting in Unanalyzed/Single_Animal,
never as lost or as a mismatch: `mousereach-reconcile` (detail `claimed
by <machine> for pose`, plus `not refreshed for N h -- that machine's watcher
may be stopped` once a claim is more than an hour old), the census (crop_dlc),
the dashboard folder scan ("cropped", with `claimed_by`), the collage
downstream index, and `mousereach-watch-process-animal` (lists them as already
being posed and does not queue them again).

Known gaps:

  * GPU nodes still running older code copy singles without claiming, so a
    double pose stays possible until every node runs this version.
  * `mousereach-watch-process-animal` and `mousereach-crop --queue` copy
    singles into a DLC queue without claiming them.
  * Originals that the old adoption left behind in Single_Animal are not cleaned
    up. A node that has a row for one skips it. A node without a row checks the
    archive and review holds before it would claim one.
  * Only a running GPU watcher sweeps stale claims. In a lab with a single GPU
    node, a claim that node gave up on comes back only after that node runs
    for 24 h without refreshing it; reconcile shows its age meanwhile.
  * A losing node's row is recorded 'unresolvable' (with a WARNING), and the
    dashboard health line counts it with the other unresolvable rows.
  * Rows parked before 2026-09-15 under the old text ("registered from the
    shared singles folder, but the file is no longer there") are not re-driven
    automatically.

---

## Update 2026-09-18: a crop stops too, and the node says when it is safe to record

WHY. Pausing for a recording program was not immediate. A pose stopped within
seconds, but a crop already running finished first -- up to eight ffmpeg
re-encodes, minutes of CPU and disk -- so an operator who opened the recording
program had to wait, with nothing on screen telling them when the machine was
theirs. Re-cropping a collage costs minutes of a machine nobody is using; a
recording that drops frames is behaviour that can never be filmed again. Code:
`video_prep/core/crop_interruptible.py` and `crop_worker.py`,
`watcher/record_notice.py`, and `BaseOrchestrator` in `watcher/orchestrator.py`.

A CROP ALREADY RUNNING

  * No programs listed: the crop runs in-process through `crop_collage`,
    exactly as before, and cannot be stopped part-way.
  * Programs listed: the crop runs in a child process, `python -m
    mousereach.video_prep.core.crop_worker <args.json>`, through
    `run_crop_collage_interruptible`. The parent asks
    `BaseOrchestrator._recording_abort_reason` once before starting (a reason
    means the child never starts) and then every 2 s. On a reason it kills the
    child and everything it started, and removes ONLY this run's output in the
    working folder: files ending .mp4 (a cropped single) or .json (the crop
    manifest) that were not in the listing taken just before the child started
    AND whose modified time is at or after that start (2 s allowance). The
    collage copy itself (.mkv) and every older file survive. The child is tied
    to the watcher by the same Job Object as a pose, so it cannot outlive it.
  * The collage is then put back to `stable` with `force_collage_state` (WHY
    forced: `cropping` may only move to `cropped` or `failed`, and this is
    neither -- the collage is simply waiting again), a processing_log row step
    `crop`, status `aborted`, message = the reason is written, and the local
    copy of the collage is deleted. The collage is NOT marked failed, so it
    keeps its retries, and `_release_claim_after_failed_crop` is NOT called:
    the shared claim stays with this node, which comes back to the collage.
    The handler reports no progress, so the loop sleeps and then finds itself
    paused. Log (INFO): `Collage <name>: crop stopped after N s because
    <reason>; partial singles removed, waiting to be cropped again (not counted
    as a failure)`.
  * A crop that finishes by itself is kept even if a stop was asked for in the
    same instant. A child that exits without writing a result is a failed crop,
    with the end of its output quoted and its log kept.

WHAT THE PERSON AT THE MACHINE SEES

`watcher/record_notice.py` shows two Windows message boxes on that machine,
from a background thread (the watcher never waits for a click), at most one of
each per recording episode:

  * "MouseReach is stopping" -- the first time the recording guard gives a
    reason while work is running (asked from `_recording_abort_reason`, which
    the running pose or crop polls). It says to wait for the next message.
  * "Safe to record" -- from the pause check at the top of the main loop, which
    is only reached between work items. So by then the pose or crop has already
    been killed and nothing of ours is running: the GPU and disk are free.

When the watcher resumes, `RecordNotices.back_to_work` CLOSES both boxes (found
by window title, only our own titles) and then arms the next episode. WHY closing
them: nothing dismisses a message box by itself, so a "Safe to record" box could
stay on screen after the recording program was closed and the node had gone back
to work -- the last thing the operator saw would say the machine was free while it
was posing again. A box that cannot be shown OR closed (no desktop, not Windows, a failing
call) is logged at DEBUG and changes nothing -- the INFO log lines are written
either way.

On a node with nothing in flight, opening the recording program pauses the
watcher and shows only the "Safe to record" box; closing it resumes work after
the grace period plus up to one poll. The "MouseReach is stopping" box does not
appear then, correctly: it is raised from `_recording_abort_reason`, which only
runs from inside a pose or crop. Turned off with `watcher.notify_safe_to_record:
false`; it is on by default but says nothing at all on a node with no recording
programs listed, which is every node that does not record.

STILL NOT STOPPED PART-WAY

An archive or staging copy already running finishes (seconds to minutes; a
half-copied file on the share is worse than the wait), and so does a local
pipeline run on an also_process node -- segmentation, reaches, outcomes and
assignment together take a few seconds per video.

---

## Update 2026-09-18: startup recovery no longer re-adopts pathless rows

WHY. `PipelineCoordinator.recover_local_db` has two branches. The one that
registers a video the local database has never seen already asked "are the files
actually on this node?" before adopting a working state (NODE_LOCAL_STATES), and
recorded the row as unresolvable when they were not. The branch that ADVANCES a
row the database already has did not ask. So a pathless local row was forced to
the remote working state on every restart, selected by the work loop, failed to
stage ("no video file for it on this node"), and dropped back to unresolvable --
which sorts below the remote state, so the next restart repeated it.

Each such row cost one work slot on every startup, indefinitely. Nothing was
lost and no video was harmed; the node simply could not reach real work for the
first minutes after each start.

The advance branch now applies the same test: a remote state in
NODE_LOCAL_STATES whose file cannot be located here (search_archive=False,
search_staging=False, as in the other branch) marks the row unresolvable once --
and leaves it alone if it already says unresolvable -- instead of advancing it.
States that live on the share (archived, crystallized, triage, deep_review) are
adopted exactly as before, with no file search at all, because adopting them is
how a node learns not to redo finished work.

Tests: `tests/test_coordination_recovery_pathless.py`.

ONE LAYER DOWN: THE WORK LOOP STILL SELECTED THEM (fixed the same day)

Recovery no longer creates such rows, but the GPU node's staging bucket still
OFFERED every `dlc_complete` row, placeholder included. The handler then spent a
whole poll interval rediscovering what the row already said -- one row per poll,
delaying the real work queued behind them.

`DLCOrchestrator._select_work_item` now filters that bucket on the NO_FILE_HERE
placeholder before `_pick_from_pool` sees it. A row with NO recorded path is still
offered, because `locate_video_file` may find its file -- that is how a node picks
up work which genuinely is here. The retirement path itself is unchanged. The
constant is read from `watcher.db` with a getattr fallback rather than through
`self.db`, so a caller holding a database double is not coupled to it.

Tests: `tests/test_work_selection_skips_known_pathless.py`.

## Update 2026-10-21: every cycle says where its time went

**What was wrong.** The watcher's loop scans for work, picks ONE item, does it,
and starts again. None of the scans recorded how long it took. On a
processing node, a video's pipeline step could take seconds while the next item
started many minutes later, and the log showed a version scan followed by
minutes with no line at all, so the time could not be attributed to any step.

**What changed.** `src/mousereach/watcher/cycle_timer.py` times every phase of
a cycle (`version and review scan`, `scan`, `pick`, `work`) and, on a
processing_server-mode node, each part of the scan separately (`stale claims`,
`re-pose adoption`, `misfiled singles`, `posed discovery`, `review return`,
`collage retirement`). When a cycle takes 5 seconds or more, one INFO line
names every step:

    Cycle 512.4 s: scan 491.0 s [stale claims 0.1, re-pose adoption 2.0,
    misfiled singles 1.2, posed discovery 9.8, review return 477.6],
    pick 0.4 s, work 21.0 s

A single step of 5 seconds or more is also logged as it finishes
(`Cycle step scan/review return took 477.6 s`), so a step that never finishes
can be found from the last one that did.

**How to use it.** Search the watcher log for `Cycle ` to see the time budget
of each cycle. Nothing about what the watcher does has changed; it only says
how long each part took.

Tests: `tests/test_cycle_timer.py`.

## Update 2026-10-21: the posed-video folder is listed once per pass

**What the cycle timing showed.** On a processing node with a large posed-video
backlog, most of a cycle went to the `misfiled singles` step, not to the video.

**What was wrong.** `misfiled singles` (`ProcessingOrchestrator._rescue_misfiled_singles`)
runs on every cycle and looks for a video left in `Processing/Posed` without a
pose. For each video in that folder it listed the whole folder again, twice: once
to look for the video's pose file, once for a half-written file. The folder is on
the shared drive and holds every posed video waiting to be analysed together with
its pose files, so with many videos waiting that was two listings of a large
folder per video, every cycle.

**What changed.** The folder is listed once per pass; the pose and half-written
checks look names up in that one listing. What the check decides is unchanged (all
its earlier tests pass unchanged); only the number of listings is.

Still to address, from the same timing: the 30-minute version and review scan
and the 3-hourly collage retirement run inside the work loop,
so the cycle that hits them stops all work until they finish.

Tests: `tests/test_misfiled_video.py::test_the_folder_is_listed_once_per_pass_not_once_per_video`.

---

## Update 2026-10-21: a node keeps only what it starts; a stopped node's videos come back in hours

**What was wrong.** A GPU node that cut a recording kept every one of its videos
(up to eight) claimed in its own `Unanalyzed/Single_Animal/.inflight/<machine>/`
folder and in its local DLC queue, and posed them one at a time. It kept them
claimed through a pause -- a recording node may pause for a whole recording day
-- and another node could take them over only after the claims had gone
untouched for a day. So videos waited on one machine while the other GPU nodes
had nothing to do.

**What changed** (`watcher/single_claim.py`, `DLCOrchestrator` in
`watcher/orchestrator.py`):

1. **A node keeps only what it is about to start.** After cutting, a node keeps
   `single_claim.KEEP_AHEAD` (2) videos queued for its own GPU -- the next one
   and one ready behind it. Every further video of the recording is copied into
   the top of `Unanalyzed/Single_Animal` under a temporary name and renamed into
   place (`publish_to_front_door`), unclaimed, so any node -- this one included
   -- takes it the ordinary way. The cutting node's row for it is parked
   `unresolvable` with the reason "in the shared singles folder for any node to
   take" (`SHARED_REASON`), which the intake re-drives once the file is in the
   folder and stable: if no other node takes it, the cutting node does. A video
   that cannot be put in the shared folder is kept, as before.
2. **Pausing gives back what has not started.** Whenever a node is paused, every
   claimed video still waiting in `dlc_queued` goes back to the shared folder
   (`_give_back_unstarted`). Its local queue copy is removed once the shared
   copy is confirmed the same size (a node re-adopts whatever it finds in its
   local queue when it starts, and would pose the video while another node does
   too). A video being posed when the pause starts finishes and is not given
   back.
3. **A video a node has given up on goes back at once** (row failed,
   unresolvable, quarantined or missing), instead of waiting a day to go stale
   (`_give_back_abandoned`).
4. **A stopped node's videos come back in hours, not a day.** A claim goes back
   to the folder when its holder's status report
   (`Processing/.node_status/<machine>.json`) has been quiet for 3 hours AND the
   claimed file itself has not been touched for 3 hours -- all times read from
   the file server, never from the machines' own clocks, which can disagree by
   many minutes (`watcher/claim_marker.py`, `GONE_AFTER_S`). Both are required so a
   live node whose status writes alone fail keeps the videos it is working. The
   one-day rule for an untouched claim stays, for nodes still running older code.

**What a person sees.** Videos cut by one machine appear in
`Unanalyzed/Single_Animal` and are posed by whichever GPU node is free. A
paused machine's queue empties into that folder. The log says each time:
"put in the shared singles folder for any node to take", "Paused: gave back N
video(s) not yet started", "... has stopped running (no status report for over
3 h); returned to the shared singles folder".

**Mixed versions.** A node on older code keeps its old behaviour (it keeps
everything it cuts), which is safe; it takes shared videos the ordinary way.
Every node should run this code for the sharing to be complete.

Tests: `tests/test_singles_intake_claim.py` (the tests after "a stopped node's
videos come back").

## Update 2026-10-21: the review-return scan finds a pose left on this node, retries failures hourly, and re-reads only what changed

**What was wrong.** Three things:

1. Human-cleared deep-review videos could never return. An
   older divert between the queues had moved only the mp4 into deep review and left
   the pose and every result in the processing folder of the node that ran the video.
   The return searched the bundle, its manifest, Analyzed and pose staging -- not the
   folder it delivers to, nor the pose the video's own row names -- found nothing,
   logged an ERROR and tried again on the next scan, two minutes later -- thousands of
   ERRORs per video, each try spending seconds searching Analyzed.
2. Every scan re-read every triage bundle's segments, outcomes, assignments and review
   files to decide whether it was cleared: seconds of each two-minute scan, although
   almost no bundle changes between scans.

**What changed** (`watcher/review_return.py`):

1. The resolver also looks in the processing folder the return delivers to, and at the
   pose the video's row records, accepting only a pose of the declared model (the rule
   pose staging already had). Such a pose is used in place.
2. A return that cannot go ahead (no pose, or files still in the queue's build folder)
   is tried again after an hour (`RETURN_RETRY_S`), not every scan. The ERROR says so.
   The wait is kept in memory, so a watcher restart tries each once more.
3. A triage bundle's verdict is reused until the bundle changes: each scan lists the
   bundle's files with their sizes and modified times (one cheap listing per bundle)
   and the video's ground-truth file, and re-reads the bundle only when either changed
   -- a person saving an answer changes the bundle, so it is read on the next scan.

Tests: `tests/test_review_return_local_pose.py`.

## Update 2026-10-21: ASPA recordings can be brought in by a list of sessions

`mousereach-aspa-import-collages --sessions-file <list>` (`aspa/import_collages.py`) copies
in only the archive recordings that hold at least one listed session, in the list's order,
instead of whole cohorts. A session is one animal's tray run, `{date}_{animal}_{tray}{position}`,
written the lab's way or the pipeline's; one per line, `#` comments allowed. WHY: a recording
holds eight animals, and the sessions one question needs sit in far fewer recordings than
their cohorts. The dry run prints how many recordings would come in, every listed session that
has no recording in the archive, and every recording the pipeline already has. A recording
already in the intake folder, or cut and moved to `Analyzed/Multi-Animal`, is never copied
again (this also applies without a list).

Lists can be kept in the pipeline's `priority_lists/` folder (data only). Tests: `tests/test_aspa_import_by_sessions.py`.

## Update 2026-10-21: a segment counts as answered only when the answer is about its frames

**What was wrong.** Four places decided whether a person had answered a segment -- the
review gate (may a freshly run video be archived?), the review-return scan (may a bundle
leave the queue?), the review tool's queue view, and the queue index -- and all four
counted an answer by the segment NUMBER it was written against. The results step matches
an answer by the FRAMES the reviewer saw (schema 1.1 `segment_span`). After a person
re-cut a video, an old answer for "segment 7" -- about other footage -- counted as
answering today's segment 7: the video left review, the results step correctly ignored
the answer, and today's segment 7 reached the data unwatched. An integrator's bench check
could then find the disagreement and send the same videos back on every run, and the return
scan would release them again within minutes, with nobody looking.
Separately, a segment shorter than 5 frames was always treated as a segmenter artifact,
even when a person had cut it.

**What changed.** One definition, `causal_review_io.answered_segments`: a CURRENT segment is
answered when a review record's frames match it (`index_review_by_segment`, the matching
the results step already used) and the record is an answer (`is_answered`, since
2026-10-26: not a `reviewed: False` placeholder, and for a pellet an outcome other than
`triaged`, `uncertain` or none -- a pellet whose outcome was left unpicked is open).
`triage_status.resolved_segments(review, segments_doc)` uses it; the gate, the return scan,
the review tool and the queue index all pass the video's segments document. Without a
segments document, answers still count by number (the only handle there is). When the
segments document says `boundary_source: "human"`, short segments are real: answers about
them are matched and applied, never dropped as artifacts: a person's segmentation is what
gets logged, even when a range is short.

**What a person sees.** A video whose earlier answers no longer line up with its current
segments stays in (or returns to) the triage queue until the segments nobody has watched
are answered; the review tool lists it as not done. An answer is never applied to footage
nobody looked at.

Tests: `tests/test_answered_means_matched_by_frames.py`.

**Update the same evening: the import copies on the file server.** `--sessions-file` used
to copy each recording by reading it into the machine running the import and writing it
back to the same share. That can fill the machine's network link and slow every other job
on it (intake, analysis) for as long as the import runs. The import now copies with the
system copy (`CopyFileW`), which, between folders on the same share, the file server
performs itself, with no data through the importing machine
(`aspa/import_collages.copy_file`). A stopped run's half-written `.partial` file is replaced
by the next run.

## Update 2026-10-21 (night): intake finds a video's files by name first; retirement waits an interval after start

**Intake.** Taking a posed video in could take minutes, almost none of it copying: finding the
video's files (`ProcessingOrchestrator._get_associated_files`) listed the posed-video folder
and asked the share "is this a file?" for every entry -- thousands in a busy folder -- before
looking at the name. It now tests the name first and lets the listing's own entry answer the
rest. Same files found; intake takes seconds.

**Collage retirement** (finished recordings to `Analyzed/Multi-Animal`) first runs one
interval (3 h) after the watcher starts instead of at start. It reads every processing
manifest in Analyzed -- many minutes on a busy share -- the work loop waits for
it, and a restart is exactly when a backlog is waiting. Nothing waits on retirement.

Tests: `tests/test_intake_lists_names_first.py`.

## Update 2026-10-21 (night): the version scan reads folder listings, not one file at a time

**What was slow.** The half-hourly version and review scan (`ReprocessingScanner.scan`) checked
every archived video separately: its manifest's modified time (one network round trip each),
each place its review might be (`resolve_review_path`: the triage bundle, beside the canonical
video, the durable store -- several missed lookups each), and the review's sha256 (re-read every
pass). With a slow share and a large archive, a warm pass could take many minutes, every 30
minutes, and the processing node's work loop waited for all of it.

**What changed.** One listing of each `Analyzed/<project>/<cohort>` folder gives every manifest,
features file and modified time (`ReprocessingScanner._index_archive`; never inside a superseded
folder). Every video's review comes from `causal_review_io.review_path_index`: the same places in
the same order with the same newest-wins rule as `resolve_review_path`, from a few listings. Review
fingerprints are kept between passes until the file changes. Verified against the per-video
lookup before deploying: identical manifests and features, and the same review for every
sampled video.

Found on the way: the "beside the canonical video" place in `resolve_review_path` looks for
`Analyzed/<cohort>/<video>.mp4`, so it never resolves when the filing rules put a video one level
deeper (`Analyzed/<grant>/<cohort>/`), and encoded ASPA names fail it outright -- so reviews are
found in the triage bundle or the durable store. Kept as is (the index follows the same rule);
worth a look.

Tests: `tests/test_version_scan_from_listings.py`.

## Update 2026-10-21 (night): a shared video another node took costs no work slot

Every GPU node registers every video it sees in the shared singles folder ('validated').
When another node takes one first, the file is gone; the adopt handler found that out and
set the row aside -- one row per work cycle. Once nodes shared what they cut (keep 2, share
the rest), that became routine: a node could spend many cycles in a row on it while
recordings waited to be cut. Now, before choosing, the node lists the places the
handler takes a video from -- its own claim folder, its local Processing and DLC queue, the
top of the shared folder -- and sets aside, in one pass, every 'validated' row whose file is
in none of them (unresolvable, "gone from the shared singles folder", re-driven if the file
comes back), exactly as the handler would. A row whose file IS in the shared folder is offered
whatever its recorded path says (`DLCOrchestrator._split_gone_singles`). With no shared folder
configured, every row is offered as before.

## Update 2026-10-22: the collage-retirement check finds its facts from listings

`collage_provenance.build_complete_stems` (which offspring are truly done, so their collage can be
retired) read every processing manifest in Analyzed every run and looked each offspring's
review up separately: many minutes on a slow share, while the processing node's
work loop waited. One walk now collects every live manifest and features file with its modified
time from the listing (`analyzed_tree.iter_file_entries`, same files and rules as `iter_files`);
manifests are re-read only when they changed (kept for the life of the process); reviews come
from `causal_review_io.review_path_index`. Same answers. Tests: `tests/test_retirement_from_listings.py`.

**Update 2026-10-22: the version scan remembers older reviews' dates.** For a video whose
manifest predates the `applied_review` fingerprint, the scan compares the review's in-file
`reviewed_at` with the manifest's `created_at`, and it opened the review on every pass to read
that one field, which took about half of a warm pass. The value is now kept until the review
file changes (same key as the fingerprints: path, size, modified time); a failed read is never
kept. A warm pass now reads no review files; what remains is the folder listings.

## Update 2026-10-22: where is each session of a list (coverage report)

`mousereach-session-coverage --sessions-file <list>` (until the command is installed on a
machine: `python -m mousereach.watcher.coverage --sessions-file <list>`) places every listed
session in exactly one stage, from shared storage only: analysed, in review, review parked, the
shared record's state (waiting to be posed / being posed / posed, waiting for analysis / being
analysed / needs attention), the files in the shared singles and posed folders, or the
recording's own state (waiting to be cut / being cut / cut). A session with no recording in the
intake is "not imported". It writes `<list>.coverage.json` beside the list (counts by stage and
each session's stage) and prints the counts. Read-only. WHY: once a list's recordings are
imported its sessions spread across the pipeline, and nothing said how far the list had got
without asking every machine. Tests: `tests/test_session_coverage.py`.

It waits up to 10 minutes for the shared record to be readable (it waited 30 s until 2026-10-26,
which a scheduled run could exceed and fail with "database is locked": on a slow share an ordinary
write can hold the file for minutes). A once-a-day report has no reason to give up sooner.

**Update 2026-10-29: it reads a local copy, never the record in place.** The 10-minute wait
could run out too, when something held the record for longer, and the report failed.
Waiting longer would only move the limit. The report now copies the record to a temporary
local file and reads that; a file copy is not blocked by the database's locks, and a long read
can no longer hold up the watchers' writes either. The copy is trusted only when it was taken
between two writes: the record commits through a rollback journal, so a write in progress
always has a `watcher_central.db-journal` file beside it, and every finished write raises a
counter in the file's header. A copy taken with no journal present and the same counter before
and after is accepted; otherwise the report waits 30 s and copies again, up to 20 times (10
minutes). If every try met a write, it stops with "could not take a settled copy ... it was being
written to every time", and that line appears wherever the report's output is logged.
The temporary copy is deleted when the report finishes. Tests: `tests/test_session_coverage.py` (a copy between writes, none
while a write is under way, a copy a write overtook is taken again).

**Fix: a pose that was handed on is no longer reported as analysed.** The shared record has two
tables, and the word `archived` means different things in them. In `videos` (a copy of a node's
row made when that node files a video into Analyzed) it means filed. In `pipeline_videos` (the
live record every node writes) a posing node -- a GPU PC or the CPU pose pool -- writes `archived`
when it hands the finished pose on to the posed folder; nothing has been analysed yet. The report
used to read both as "analysed", so a posed video waiting for (or in) analysis counted as done.
Now an `archived` from `pipeline_videos` is placed by where the pose actually is: a processing
node's claim in `Posed/.claims/<video>.claimed` -> "taken in for analysis"; the video in `Posed`
-> "posed, waiting for analysis"; neither, and not in review or Analyzed -> "handed on; not in
Posed, review or Analyzed" (a stage worth a person's look). The stored word is unchanged:
recovery relies on it to stop a second pose of the same video. Test:
`test_a_pose_hand_on_is_never_reported_as_analysed`.

## Update 2026-10-22: a GPU node retries the shared-record writes that failed

A GPU node writes each video's state to the shared record (`watcher_central.db`,
`pipeline_videos`) as it cuts, poses and hands it off. Those writes were best-effort: a failure
(the shared database locked) was logged at DEBUG and dropped, and the record fell behind for good --
all of a collage's singles could exist while only some of them were on record. Other nodes read that record before cutting a collage, so a missing child is a collage
another node may cut and pose again. Now `DLCOrchestrator._sync_to_connectome` keeps the latest
failed state of each video and retries it at the start of every scan; a newer successful write
for the same video drops it, and a retry never replaces a row updated since the original attempt
(`PipelineCoordinator.sync_video_state_unless_newer`, stamped with the attempt time). At startup
the node puts on record every video it has that the record lacks entirely, never touching an
existing row. The fill writes in batches (`insert_video_states_if_absent`, one transaction per
200 rows: one connection per row opened the shared file hundreds of times in seconds and hit
"unable to open database file"), and a fill the record interrupts is finished by a later scan, at most every
10 minutes (`_finish_shared_record_fill`), instead of waiting for the next restart. Tests:
`tests/test_shared_record_writes_retried.py`.

## Update 2026-10-22: work-order selectors take ASPA lab ids

An ASPA video carries an encoded id (ASPA0401); the lab names the animal D01. Session lists
already accepted either spelling, but an `animal_id` or `session` selector in the work order
written with the lab id matched nothing. `work_priority.field_labels` now adds
the lab spelling of every encoded ASPA id (`mousereach.aspa.identity`); other ids are
unchanged. Tests: `tests/test_work_priority.py` (TestAspaLabIds).

## Update 2026-10-22: a collage with a slot ffmpeg could not cut is not "cropped"

A GPU node cutting a collage skipped any slot ffmpeg failed on and then recorded the collage
`cropped` anyway -- "0 singles" when every slot failed -- so nothing ever retried it, and the
sessions of the failed slots were lost (a collage could have every slot fail in the same
instant); found when a census of every collage was checked against every single-animal row. Now
(`DLCOrchestrator._process_collage`) a failed slot fails the collage through the ordinary
failure path: its row says `failed` with each failed slot and ffmpeg's message, the working
copies are cleaned up, the singles that were cut stay queued, and the shared claim is kept
while this node carries any of them. A failed collage whose file is still in the intake is
retried (`state.py`, up to `max_retries`), and a retry skips children that already exist. The
cropper now runs ffmpeg through `subprocess` and keeps the tail of its error output in the
slot's error (it used `os.system`, so the message went to a console nobody reads), and
removes any partial file a failed run left. Tests: `tests/test_gpu_node_guards_orchestrator.py`,
`tests/test_cropper_ffmpeg_errors.py`.

## Update 2026-10-22: a re-run that finds no pose no longer fails the video

When the server re-runs an outdated video and finds no pose -- not on the row, not in
staging, not in Analyzed -- it used to mark the video failed, for example when the
bench-disagreement router had moved its bundle, pose included, into triage. A missing pose is not a scientific outcome. `ProcessingOrchestrator._no_pose_for_reprocess`
now (1) puts the video in a review queue's state when its bundle there holds the pose; (2) when
its video is in Analyzed, sets `reprocess_scope` to `full` and logs a 'skipped' reprocess step --
the row stays `outdated`, and the next version scan publishes a re-pose request that a GPU node
serves; (3) only when no video can be found either, marks it failed with "no pose and no video
found". Tests: `tests/test_no_pose_reprocess.py`.

## Update 2026-10-22 (historical; superseded): a queue bundle of an archived video was moved to a side folder

(No longer so: nothing is moved to `_Problematic` any more, which is only for cases the pipeline cannot process at all. Such a folder is now a leftover: what has a current copy elsewhere is deleted, and what does not stays where it is, with a note, listed under 'Folders left for a person' in the Review Queues tab -- see "Leftover queue folders" in docs/REVIEW_TOOLS.md.)

The return scan reads every queued video's state in one query per queue
(`WatcherDB.states_of`). For a video this node had `archived`, a bundle holding
nothing a person wrote WAS moved to `_Problematic` with a note when it held no
results at all (no `<stem>_segments.json` -- a file a return could not take
along) or when every file in it is older than the video's processing manifest in
Analyzed (`_superseded_by_archive`, `review_return.py`; both queues, under the
same per-video claim as every other move). Analyzed is read once per bundle and
row change, not on every scan. Bundles the bench-disagreement router places are
never caught: their results leave Analyzed and their routing record postdates the
archive. Companion to the archive-step fix above (`queue_folder_holds_this_run`):
that one stops new husks from derailing a run; this one clears the ones already
sitting in a person's queue. Tests: `tests/test_review_return_superseded.py`.

## Update 2026-10-22 (historical; superseded): a leftover triage copy of a deep-review bundle was moved to a side folder

(No longer so: nothing is moved to `_Problematic` any more, which is only for cases the pipeline cannot process at all. Such a folder is now a leftover: what has a current copy elsewhere is deleted, and what does not stays where it is, with a note, listed under 'Folders left for a person' in the Review Queues tab -- see "Leftover queue folders" in docs/REVIEW_TOOLS.md.) A triage folder that differs from the deep-review bundle does NOT stay whole any more either: it is emptied as a leftover the way a route into deep review empties it -- what the deep-review bundle or the review store holds byte for byte, queue records and machine results are deleted, and only what a person wrote with no copy elsewhere is kept, listed.

When a video has a folder in both review queues and its segmentation failed, the return scan
used to skip it with "exists in BOTH queues; divert skipped (duplicate needs deliberate
cleanup)" -- every scan, for good, because nothing ever did that cleanup. The scan then compared the two folders (`_is_leftover_copy`,
`review_return.py`): if every file in the triage folder had a byte-identical file of the same
name in the deep-review folder, the triage folder was moved to `_Problematic` with a note, under
a return claim like every other move, and the deep-review bundle was untouched. A pair that
differed in anything stayed for a person. (Both outcomes are replaced by the leftover rule in
the banner above: a person's unsuperseded files are kept and listed, everything else is deleted.) Sizes are compared before any bytes are
read, and a pair's verdict is kept until either folder changes. Tests:
`tests/test_review_return_stale.py`.

## Update 2026-10-22: the version scan keeps what it has read across restarts

The version scan's three caches -- each manifest it parsed, each review's fingerprint, and each
older review's reviewed_at -- lived only in memory, so the first scan after every watcher start
read all of them again over the share, many minutes on a slow share, while the work loop waited. They are
now written after each scan to `version_scan_cache.json` in the machine's own processing root
(local disk, never the share; each machine's scanner keeps its own) and read back when the
watcher starts (`ReprocessingScanner(cache_path=...)`, `reprocessor.local_cache_path`). Nothing
in it is trusted blindly: a manifest is used only while its modified time is unchanged, a review
entry only while its path, size and modified time are unchanged, exactly as within one run. Only
entries for files the scan just saw are written back, so videos that left the archive leave the
file. A missing, damaged or older-format file only means a cold first scan; deleting it is always
safe. The by-hand `mousereach-version-check` keeps its caches in memory only. Tests:
`tests/test_version_scan_cache_on_disk.py`.

## Update 2026-10-22: setup keeps every setting, and an unset pipeline root is not guessed

Two defects that compounded. Saving from `mousereach-setup` or the in-app Setup panel
rewrote `~/.mousereach/config.json` with three settings only (processing_root, nas_drive,
watcher), so every other one was deleted on save -- the shared pipeline folder (`nas_root`),
the backup's folders, anything else the file held. And with `nas_root` gone, the
pipeline root was guessed from an old fixed layout under `<nas_drive>`: every
shared folder then pointed at the old place. The watcher refused to start there (no
`pipeline_versions.json`), but the review tools and intake commands make no such check. The
Setup panel added a third fault: its "NAS data root" field showed a pipeline folder as its
example but saved it as the drive, so the guessed root came out one level too deep.

Now: `save_config` (`setup/wizard.py`) merges into the file -- the settings it is given are
set, the watcher section is merged, and everything else is kept. The wizard asks for the shared
pipeline folder as its step 2 (pre-filled from the existing setting, then the machine's lab
profile) and builds the pipeline folders there, never under a guessed folder. The Setup panel's
field is the shared pipeline folder and saves `nas_root`. `config.py` no longer guesses: with
`nas_root` unset every shared path is `None` and `NAS_ROOT_ORIGIN` is `unset`, so each tool
says it is not configured and names `mousereach-setup`. A machine that sets `nas_root` sees
no change. Tests: `tests/test_setup_keeps_settings.py`.


## Update 2026-10-22: Easy and Flat videos are no longer held for the outcomes file they never get

Easy (E) and Flat (F) trays skip outcome detection, reach assignment and kinematics by design,
then reach "processed" and are filed in Analyzed with their segments and reaches (no features
file, so nothing reaches the database). That last part had stopped happening: the quality
check's structural layer still required the outcomes file and called its absence critical, so
the gate sent every Easy/Flat video to deep review. The structural layer now asks for the outcomes file
only when the per-tray step table (`watcher/router.SKIP_STEPS`) runs outcome detection for that
tray; a Pillar video without one is still critical, and a name with no recognisable tray is
held to the full set. Checked on copies of held Flat bundles: each now passes the check
and the gate. Tests: `tests/test_qc_easy_flat_no_outcomes.py`.

## Update 2026-10-22: a quality hold the current check would not make leaves the person's queue

When the quality check is corrected, the holds it made by mistake used to stay in deep review
until a person cleared each one. The return scan now re-checks every deep-review bundle routed
`qc_needs_review` that holds no file a person wrote: the pose-tracking layer's stored flags are
carried over (that layer reads the whole pose file; a hold it made stays, except a ruler-variation
hold -- see the 2026-10-23 update), the file-based layers are recomputed. A bundle with nothing critical left goes back to the posed folder for a fresh run
(mp4 and pose files; its other files used to be set aside under `_Problematic/_released_qc_holds_<date>/`
-- now they are cleared as a leftover, see "Queue leftovers" under "Update: routing into and out of the review queues" below)
and its row becomes `dlc_complete`, so intake takes it in its priority turn -- not `processing`,
which would block intake of new work. The first use: Easy/Flat bundles held only for the
outcomes file they never get. Bounded to 30 s of each scan; one re-check
per bundle until it changes; nothing in the posed folder is ever replaced. The posed folder usually
still holds the held video's files (intake leaves them there until archiving); the same file
there (same size and modified time, or the same SHA-256 when the times differ) is kept as the
one copy and the bundle's duplicate removed, a differing one leaves the bundle for a person (without
this most releases were refused as a clash; hashing only when the times differ keeps each release
quick). Tests:
`tests/test_release_lifted_qc_holds.py`.

## Update 2026-10-22: recordings left on old storage -- inventory and verified bring-in

`mousereach-legacy-inventory --from <old storage root>` (until installed on a machine:
`python -m mousereach.pipeline.legacy_storage --from <old storage root>`) lists every video
under a retired pipeline folder and says, for each, whether the pipeline root already has it:
the same file; the session's cut video, results or shared-record row; or, for a recording,
every animal session held one of those ways or by a recording under another name (blank
slots ignored). With `--bring-in` it copies each video that exists only on the old storage
into the stage folder that picks it up -- a recording into the intake, a cut video into the
shared singles folder, whatever the tray (Easy and Flat are taken last there, by the
machines' deferral, as everywhere). A recording whose cut videos all come in too goes to
`Analyzed/Multi-Animal` (it is already cut) instead of the intake. Each copy goes to
`<name>.part`, is checked against the original by size and SHA-256, then renamed; nothing
is ever replaced, and each copy is logged in `<root>/legacy_inventory/<label>_brought_in.jsonl`.
Left for a person and named in the report: names that are not pipeline names (extra text,
animal ranges), malformed animal ids, the same name at a different size, a recording
only some of whose cut videos are on the old storage, and copies of one name in several old
folders that differ in size (copies at one size are one video, brought in once). The old storage is only read. Report:
`<root>/legacy_inventory/<label>.csv` and `<label>.summary.json`; "only here" reaching zero
means the old storage holds no pipeline video the pipeline lacks. WHY: videos only in the
retired folder were invisible to every machine that could process them, and the old storage
could not be retired while they were there. Tests: `tests/test_legacy_storage.py`.
**Changed 2026-10-23:** recordings only, always into the intake. Cut videos are never brought in
(the input to MouseReach is the recording; the pipeline's own cut records which recording each single
came from): a cut whose recording comes in is reported and left, and one with no recording anywhere is
named for the rebuild route (`mousereach-rebuild-recording`). The "already cut -> Analyzed/Multi-Animal"
placement and the "partly cut" status are gone with it. Also since 2026-10-23: a recording of the same day and tray, the
same size to the byte, under another name on the pipeline root is "renamed since it was copied" and not
brought in -- a person corrected an animal id on the pipeline's copy (for example a mistyped id), and
the old copy under the old name would have recreated the animal the correction removed.

## Update 2026-10-22: the backup says what failed, skips test residue, and fails loudly

The backup could report "Robocopy error ... exit code 11" for a tree on every run, and
nothing else: robocopy runs with its file listing off and its error lines
were never logged, and `mousereach-backup --once` exited 0 regardless, so the scheduled task
showed success to every check that reads task results. Run by hand, every failure was a
`pytest-cache-files-*` folder -- residue pytest leaves, unreadable, in a code checkout on a
network share; never data. Now (`watcher/backup.py`): those folders and `.pytest_cache` are
excluded; a failed run logs robocopy's own error lines (the file and the reason, first ten);
and `--once` exits 1 when any folder did not sync completely, so the task result shows it. Tests:
`tests/test_backup_reports_failures.py`.

## Update 2026-10-22: running watchers follow edits to the work order

The work order (`priority_order.json` at the pipeline root, or a machine's own
`watcher.work_priority`) was resolved once per watcher process, so an edit -- including a Save
in the priority panel, which says machines pick it up within a minute -- reached a running
watcher only when it was restarted. `BaseOrchestrator.work_priority` now re-resolves at most once
a minute (`PRIORITY_REREAD_S`; the file read itself was already cached for a minute) and logs the
order at the first resolve and again, marked "(changed)", whenever it changes. Applies to every
watcher once it runs this code. Tests: `tests/test_work_priority.py` (TestRunningWatcherFollowsEdits).

## Update 2026-10-22: every bypass of the video state rules says why

`force_state` (the escape hatch from the state machine) logs each use at WARNING with the reason
it is given -- the only record of why a video skipped its normal transitions. Seven call sites
gave none, so the log read "reason: not given" for each of them: posed-folder
discovery, orphan recovery, cross-node recovery, database-confirmed archiving, re-runs of
archived videos, crystallize and uncrystallize. Each now states its reason, and
`tests/test_state_bypasses_give_reasons.py` reads the source so a new call without one fails.

## Update 2026-10-22: a shared-drive outage idles re-running instead of failing videos

When the shared drive dropped, the re-run step marked each outdated video FAILED
("Archive directory not found for reprocessing"), about one a second -- infrastructure failures
recorded as the videos' fault, which a failed state must never mean. Now the processing node
offers no re-run work while the Analyzed archive cannot be seen (`_archive_reachable`): the
videos keep their state, the log says once that the archive is out of reach and once that it is
back, and the re-run step itself declines rather than failing a video if it ever gets there.
Tests: `tests/test_outage_idles_reprocessing.py`.

## Update 2026-10-22: the watcher's dry run only reads

`mousereach-watch --dry-run`, the step the install guide has every new machine run first,
built the whole watcher (whose start-up runs recovery) and then called the real scans, which
register files and move videos between states -- so it changed the local database and touched
the shared one. It is now a read-only
preview (`watcher/preview.py`), run before the single-instance lock, the database and the watcher
are built: the database is opened read-only, the folders the machine serves are listed, the
work order is shown, and the next pick is worked out by the watcher's own selection code on a
stand-in database that has no write methods. It can run beside a live watcher. The two old
`dry_run` methods print the same preview. Tests: `tests/test_dry_run_is_read_only.py` (the
database bytes and the folder tree are unchanged after a preview).

## Update 2026-10-22: a node's status report says what it is doing

Each node's report on the shared drive (`Processing/.node_status/<host>.json`) said "working"
every loop turn and nothing else, so a node churning on nothing looked exactly like a
productive one, and a node deep in a long pose looked like one whose report had gone stale.
It now also carries `doing` (the item in hand: type, video, since when -- written
as the item starts), `last_done` (the last item finished) and `done_last_hour`. The Nodes
section of `mousereach-watch-status` shows them on each node's line. A node shows them once it
runs this code. The last item and the hourly count are kept by the running watcher, so they
start again from nothing when it restarts. Tests: `tests/test_node_status_says_what_it_is_doing.py`.

## Update 2026-10-22: a list of sessions can lead within its project

The work order could say "this project, these cohorts" but not "these sessions a study needs".
The order file now takes `session_lists` beside `cohorts`: per project, files of sessions in
the shared pipeline folder (one per line, lab or pipeline ids). Within that project the listed
sessions come first, then its cohorts, then the rest. A recording counts as listed when any of
its animals' sessions is (it must be cut to reach that animal). New selector fields `session`
and `session_list`; list files are re-read at most once a minute. The priority panel has a box
for it. A panel Save now keeps keys it does not manage (notes, hand-written selectors), and keys
or list files the order cannot use are named in its status line. Tests:
`tests/test_session_list_tier.py`.

## Update 2026-10-22: the work order at the remaining selection points

Re-pose requests were taken from the shared queue alphabetically, so one project's older
videos could go to the graphics cards ahead of a preferred project's whatever the order said;
a GPU machine now takes them in the work order (`repose._in_lab_order`: idle-only work last, then tier, then
oldest first). And every place that sorts work by the order -- the review-return scan's order,
the review tools' queue listings and next-video choice, the re-segmentation queue -- now puts
idle-only work (Easy/Flat by default) after everything else, whatever its tier, through one
sort key, `WorkPriority.rank`. Before, they sorted by tier alone, so an Easy video in a preferred
project came ahead of Pillar videos of the next. Tests: `tests/test_repose.py`
(test_consume_takes_requests_in_the_lab_order), `tests/test_session_list_tier.py`.

## Update 2026-10-22: a GPU node sets aside local copies of videos already finished

Copies of videos finished long ago sat in a GPU node's own pose queue and processing folder for
good: never picked (their rows are finished) and never removed. They came from the old behaviour
where a node kept every video it cut (it keeps 2 since 55e5ad8). At startup a GPU node now moves
such copies out of `DLC_Queue` and `Processing` (`DLCOrchestrator._set_aside_finished_copies`):
only when its row is finished (archived, outdated or crystallized) AND the video's processing
manifest is in Analyzed where the video belongs. Every file of that video moves together to
`<processing_root>/_finished_elsewhere/<date>/<folder>/` -- set aside, never deleted; one log line
says how many. Tests: `tests/test_set_aside_finished_copies.py`.

Later the same day the step moved to `BaseOrchestrator` and the processing node runs it at
startup too, for such copies left in its own `Processing` after the videos were filed into
Analyzed. On the server only `archived` and `crystallized` count as
finished, not `outdated`: the server re-runs an outdated video by copying its archived files down
into `Processing` while the row still says `outdated` (`_reprocess_video`), and those copies are a
re-run's working set.


## Update 2026-10-22: startup recovery no longer clears a review hold with an older record

At start, a graphics-card machine compares each row of its own record with the shared one
(`watcher_central.db`, `pipeline_videos`) and moves rows forward to match. A review hold
(triage, deep review) is not among the ordered states, so it ranked below every shared state
and was overwritten. Shared `archived` from a pose-only machine means "this machine handed it
off", not "analysed", so a hold could become `archived` with no results anywhere, even from a
shared record written weeks before the hold.
Now a hold is replaced only by a shared record written after it (`REVIEW_HOLD_STATES`,
`_remote_is_newer` in `watcher/coordination.py`), and never by the optional reach-table
cross-check, which cannot say when its rows were written. When either time is missing the hold
is kept. Tests: `tests/test_coordination_recovery_pathless.py`.

## Update 2026-10-22: the review tool restores answers by frames; doubled and misplaced answers

The 2026-10-21 change made the gate, the return scan, the queue views and the queue index count
an answer by the frames it covers, but the review tool's own resume (`_load_saved_review`) still
restored saved answers by segment NUMBER. After a re-cut, an answer about other footage sat on
today's segment N: the tool counted N as done and skipped it, while the release check counted N
open, so the video could never be finished from the tool. The reviewer's fresh answer for the old
frames was saved beside the old one, and `_collect_answers` stamped the current segment's frames
onto the old answer at the next save.

Found in `review_records/reviews`:
- Files carrying two records for the same frames. Most such stretches resolve the same
  under any rule. In some, a placeholder or a left-blank `triaged` record sat first and was
  applied over a person's decision; in others, two decisions disagree.
- Touched answers that name a causal reach wholly outside the frames they record. Applied,
  they made a neighbouring segment's reach the causal reach.

What changed:
- `restore_review_for_segments` restores by frames and renumbers each answer to its current segment.
- `index_review_by_segment`, and so the results step and every "answered?" check:
  - prefers a decision over a placeholder or a blank;
  - counts two agreeing decisions once;
  - applies neither of two disagreeing decisions;
  - drops an answer whose reach is outside its own frames.
  Each of these is noted in the log.
- Answers that match no current segment are carried in the file under `unmatched_answers` and
  never applied.

Already archived videos keep their stored results until they are next run. Tests:
`tests/test_review_restore_by_frames.py`.


## Update 2026-10-22: the route command never flags a segment a person has answered

`mousereach-route-to-queue` used to flag whatever segments its caller listed. A caller -- for
example an integrator's scheduled bench-sheet scan -- cannot always know what a person answered: a segment with
no detected reach has no reach row in the database, so nothing there says who decided it. A
flagged segment the release check counts as answered goes straight back out with nobody looking,
and the next scan sends it again. Now the command reads the video's review and drops every
requested segment a person (or ground truth) has answered, matched by frames
(`answered_segments_of` in `watcher/route_cli.py`, the release check's own rule). It reports them
as "answered", and does not route the video at all when nothing is left (exit 0, tag "done").
Tests: `tests/test_route_cli_db.py`.

## Update 2026-10-22: a re-run finds the archived pose where it lives

Every re-run of an archived video (`_reprocess_video`) found its pose file by walking the whole
Analyzed tree: many seconds on a quiet share and minutes on a busy one, for a re-run whose own
work takes seconds. With a few re-runs queued, the node's throughput collapsed. The pose is always in the video's results folder (`Analyzed/<project>/<cohort>`)
or the same cohort folder of a per-model tree (`Analyzed/<project>/DLC Model <N>/<cohort>`), both
computable from the video id. `archived_pose_candidates` looks there first and walks the archive
only when no pose from the declared model is found there, which is the old behaviour. Tests:
`tests/test_reprocess_pose_lookup.py`.

Correction, measured on a live re-run: the archive walk was only part of it. Most of the time
went to `_reprocess_copy_set`, which decides which files to copy down. It asked
`Path.is_file()` of every entry in the cohort folder, one network stat each, before
looking at the name. It now checks the name first and uses the directory listing's own
file flag (no extra round trip on Windows). Test: `tests/test_reprocess_copy_set.py`.

## Update 2026-10-22: re-archiving leaves unchanged files where they are

`archive_video` swept everything already at the destination into the versioned Archive before
moving the new set in. On a re-run the video and the pose come back byte-identical: they were
copied down from the archive and never changed. Sweeping them stored another copy of the pose for
every re-run and moved the video and pose back over the share each time. Files identical to
the archived copy (`_same_file`: same size and modified time within 2 s, else the same SHA-256) are
now left in place and the local copy dropped. Only files that actually changed are a new
generation. Tests: `tests/test_archive_identical_files.py`.

## Update 2026-10-22: watcher commands refuse an option they do not know

The watcher's commands (`mousereach-watch`, `-watch-status`, `-watch-reprocess`, `-watch-quarantine`,
`-watch-process-animal`, `-version-check`, `-crystallize`, `-uncrystallize`, `-watch-unresolvable`)
read their arguments by hand. An option they did not know was silently ignored, so a typo such as
`mousereach-watch --dryrun` started the real watcher instead of the read-only preview. The usage text
each prints for `--help` is now also the list of options it accepts: anything else stops the command
before it does anything, with the usage, "ERROR: unknown option(s): ..." and "Nothing was changed."
(exit 2). `mousereach-watch-toggle` and `mousereach-watch-recorders` already refused unknown
arguments. Tests: `tests/test_watcher_cli_unknown_options.py`.

`mousereach-watch` itself also refuses any WORD (it takes options only): a script that ran
`mousereach-watch stop` meaning to stop the watcher instead started one (or, with one
already running, reported it and did nothing). It now exits 2 and says how stopping works
(`watcher_stop.flag` or the Stop button; `mousereach-watch-toggle --pause` to pause).

## Update 2026-10-22: the server sends the shared record the exports it missed

After archiving, a video's record and log go to `watcher_central.db` (the audit table `videos`).
A failed export was a warning and never retried, so archived videos from a share outage could be
missing there for good. At every start the processing node now compares its archived rows with the shared table and sends what is missing
(`ProcessingOrchestrator._export_missed_to_central`; only the server, where "archived" means filed
in Analyzed). Tests: `tests/test_missed_central_exports.py`.

**Changed 2026-10-26.** That check looked for a video with no row at all, from any machine. A video
posed on a GPU PC already has that PC's row, so an export from the server that failed was never sent
again. It now compares the
server's OWN rows (`source_machine`) with its archived videos and re-sends any that is missing or
older than the local row. It runs at start and then every 30 minutes (cycle step "scan/shared
record"), at most 50 videos per pass so a backlog never holds up the work loop. The export itself
waits up to 30 s for another machine's write to finish, instead of Python's default 5 s.


## Update 2026-10-26: asking a person again about a pellet they answered (--recheck)

`mousereach-route-to-queue --recheck` reopens pellets a person already answered instead of skipping
them (`route_cli.reopen_answered`): each matched (by frames) review record keeps its answer under
`answers.recheck.previous`, is marked `answers.reviewed = False` (open), and the pellet is flagged with
the question; the review file is first copied to `Analyzed/Archive/manual_corrections/`. The video
stays in triage until that pellet is answered again. WHY: an answer can be doubtful on its face --
for example "retrieved" while the frames show the pellet lying in the scoring area afterwards -- and
the person is asked again rather than the answer overridden. Ground truth is never
reopened. Tests: `tests/test_route_recheck_answered.py`.
Only a record that is an answer (`is_answered`) is reopened: a placeholder, or a pellet saved with
no outcome decided (also an older record holding the algorithm's "triaged" with no `reviewed` flag),
is already open, so there is nothing to ask again.

## Update 2026-10-28: each review answer records who gave it

A causal review file (`<video>_causal_review.json`, its durable copy under `review_records/reviews/`)
carried one `reviewer` / `reviewed_at` for the whole file, rewritten by every save -- so one answer
saved by a second person re-credited every earlier answer in the file to them. `save_causal_review`
now gives every answered segment record its own `reviewer` and `reviewed_at`
(`stamp_segment_reviewers`): this save's for an answer that is new or changed since the file was last
saved (compared on outcome, causal reach, phantom flag, answers, notes, true segment number --
matched by segment number, else by frame span), the earlier record's otherwise (or, for a file
written before this, that file's own file-level values). Placeholders get none. The file-level fields
are kept and mean "last save". Readers that need who answered a segment -- `update_corpus_index`,
`apply_review_overrides` (the `reviewed_by` stamped onto results), `truth_resolver`, and
`route_cli.reopen_answered` -- use the record's own values and fall back to the file-level ones
(`segment_reviewer`). Existing review files are not rewritten. Operator description:
docs/REVIEW_TOOLS.md "The record format". Tests: `tests/test_review_per_segment_reviewer.py`.

## Update 2026-10-28: the hand-score check runs before triage, not before filing

The comparison of a video's pellets with the session's hand scores (see the 2026-10-26 update below
for what it compares and where the scores come from) moved from the last step before filing to the
point right after the analysis and the quality check, BEFORE the review gate
(`_hand_score_precheck` in `watcher/orchestrator.py`; on the processing server in `_run_pipeline`, on a
node that also analyses in `_run_local_pipeline`). WHY: at filing, a video with a triage question
went to the queue, was answered, came back, and only then met the hand score -- a disagreement sent
it to the queue a second time. Before the gate, every question about a video reaches a person in one
visit, and no video gets kinematics before it has been compared. The check is NOT inside
`review_gate.evaluate_gate`, which stays pure and is also used by `pipeline/run_all.py` and
`pipeline/reprocess_to_current.py` (those batch tools do not compare with hand scores).

What each answer does now:

- **A pellet disagrees** (and no person answered it): the pellet is flagged (`flagged_for_review`,
  `flag_reason` "bench disagreement: the hand score says X; the algorithm says Y") and the run goes
  on to the gate. A flagged pellet is a triage question (`triage_status.triaged_segments`), so the
  gate sends the video to triage once, with that pellet and every other open question. Nothing is
  routed by the check itself. A reason the pipeline had already flagged the pellet for is kept after
  the bench text, "... (also: <old reason>)", so the review tool still recognises it (it looks for
  "bench disagreement" at the start) and the reviewer sees both.
- **Agrees, not required, Easy/Flat, no scores file configured**: on to the gate, as before.
- **No hand scores yet in a required project**: the video is HELD. It stays in 'processing' with all
  its files in the local Processing folder; the run stops before the gate (no triage, no
  kinematics, no filing). It is listed (grey, waiting) in `<NAS root>/attention/<host>_hand_scores.json`,
  the processing log gets one `hand_score_check` / `waiting` row, and the pipeline step does not pick
  it again until the scores file changes. When it does change, the video is run again only to be
  checked again: its segmentation, reaches, outcomes and assignment are reused, the quality check and
  the hand-score check run again. Once the cohort's sheet has been imported after the video entered
  'processing' (`processing_started_at`, kept from the first wait) and still has no scores for it,
  the session was never scored by hand and the video goes on to the gate. (Changed: such a video
  now stays held, in red, as missing its scores -- see "a video whose hand scores cannot be found
  is held as a defect" below.)
- **The check cannot run**: held the same way, listed in red (`needs_person`, kind
  `hand_score_check_failed`), and tried again when the scores file changes -- and also every 10 minutes even when nothing changed
  (`HAND_SCORE_ERROR_RETRY_S`): a check that failed on a passing fault, such as the storage drive
  dropping for a moment while the scores file was read, used to keep the video held in red until
  the next import rewrote the scores file, up to a day later.
- **The pellet numbers are in doubt**: a review record says a segment is really another pellet
  (`true_segment_num` with `segmentation_wrong`, written by the review tool's old mislabel box) and
  the current cuts still number that stretch of frames otherwise. Segment N is then not known to be
  hand pellet N for ANY segment, so nothing is compared: the check raises `PelletNumbersUnknown` and
  the watcher sends the video to MANUAL SEGMENTATION -- the re-segmentation queue -- with the reason
  "segmentation: a reviewer said segment(s) N show a different pellet ..." and stops the run
  (`_route_pellet_numbers_unknown`; the database row becomes `deep_review`). The problem is in the
  cuts, so that is the step's review queue; it used to hold the video in Processing, where no review
  tool could reach it. In the queue the segmentation fixer lists it, and saving the cuts releases it
  (the reason names segmentation). The record is matched to the current cuts by its frames
  (`hand_scores.unresolved_mislabels`): once a person's re-cut puts those frames in the segment
  carrying the number they gave -- or a person saves cuts AFTER the record was written, the newest
  human statement about the cuts -- the correction is settled and the check compares again. Pellets a person answered are never compared,
  so a video whose every pellet is answered is not held. WHY: the check paired segment N with hand
  pellet N and never read `true_segment_num`, so after a mislabel it could compare a pellet with
  another pellet's hand score and flag, or pass, the wrong one.

Held videos do not count against `max_local_pending` (`_held_for_hand_scores`): they are analysed
already, and a night of them would otherwise stop all intake until the next sheet import. Videos
whose route into a review queue was refused are left out of the count for the same reason
(`_held_for_refused_route`; see "A refused route never fails the video"). They
occupy local disk exactly as the waiting 'processed' videos did before. The list of held videos lived
only in the watcher's memory: after a restart it was empty, every held video was picked once more,
ran in full and was checked again. (Changed: the processing server now rebuilds the list from its
attention list at start -- see "a hand-score hold survives a watcher restart" below.) On a node
that also analyses, a held video picked again keeps its row as it is (`processing_started_at` is not
re-stamped).

Removed: the check before filing (`_hand_scores_allow_filing`, and its calls in `_archive_to_nas` and
`_archive_locally_processed`), and the filing picks' skip of waiting videos. There is no second check
at filing: a 'processed' video has passed this one. Videos that were 'processed' but unfiled when
this was deployed were never compared; bringing them through the new check is a deploy step, not
something the watcher does by itself.

Visible to a person: the Pipeline Dashboard's grey line ("analysed video(s) are held before triage
and kinematics until their hand scores arrive"), the attention list, and the census, which prints
"of which waiting for hand scores: N" under the analysis line. Tests:
`tests/test_hand_score_check_before_triage.py` (renamed from `..._before_filing.py`),
`tests/test_archive_gate.py`.

## Update 2026-10-28: every pellet's hand score is kept with the video

Whenever the hand-score check reads the sheet for a video, `hand_scores.record` writes what the sheet
said into the video's `<video>_pellet_outcomes.json`, under a new top-level key:

    "hand_scores": {"pellets": {"1": 2, "2": 0, ...},
                    "verdict": "agrees" | "disagrees" | "never_scored" | "not_required",
                    "source_mtime": <modification time of the scores file that was read>,
                    "cohort_last_import": <the cohort's last sheet import, or null>,
                    "read_at": <when it was read, local time>}

`pellets` is keyed by segment number (the same numbering `check` compares with) and holds 0
(untouched), 1 (displaced) or 2 (retrieved) for every pellet the sheet has for that session -- also
when no pellet was compared because a person had answered them all. It is `{}` when the sheet has no
row for the session (`not_required`: a project that does not need hand scores; `never_scored`
appears only in videos filed before missing scores began to be held -- see "a video whose hand
scores cannot be found is held as a defect" below -- and is no longer written). Nothing is written for an Easy or Flat
tray, or on a machine with no scores file configured. The whole file is read, the key set, and the
file written to a temporary file and renamed into place, so every other key stays and no reader sees
half a file; `record` raises when the file cannot be read or written, so its caller can hold the
video rather than pass it unchecked. Every later writer of the outcomes file (the QC verdict stamp, `flag_segments`)
reads and rewrites the whole document, so the key travels with the bundle into the review queues and
into Analyzed. `hand_scores.score_from_outcomes(doc, pellet)` reads one pellet's kept score back, or
`(None, why not)`; it never raises. WHY: the review window's check-your-answer box can then show the
very score the pipeline compared with, from the bundle itself. Readers that do not know the key
ignore it. Tests: `tests/test_hand_score_check_before_filing.py` (the `record` tests).

## Update 2026-10-26: checked against the hand scores before filing

(Superseded 2026-10-28: the check now runs before the review gate, not before filing -- see the
update of that date above. This section is kept as the record of what ran from 2026-10-26.)

(Earlier versions filed a video with no hand scores without the comparison; that no longer happens:
such a video is now held as missing its scores -- see "a video whose hand scores cannot be found is
held as a defect".)

The last step before a video is filed (`_archive_to_nas` on the processing server,
`_archive_locally_processed` on a node that also analyses) is now a comparison of its pellets with
the session's hand scores (`watcher/hand_scores.py`, configured by the `hand_scores` section of
`~/.mousereach/config.json` -- or, since 2026-10-28, of the lab settings file on the storage drive,
see the update of that date; README "Checked against the hand scores before triage"). A pellet that
disagrees, and that no person has answered, is flagged (`flagged_for_review`, `flag_reason` "bench
disagreement: the hand score says X; the algorithm says Y") and the video goes to triage through `route_to_queue`, as the review tool
already shows such reasons. A session with no hand scores in a required project leaves the video in
'processed', unfiled and not picked again until the scores file changes, listed (grey, waiting) in
`<NAS root>/attention/<host>_hand_scores.json`; once the cohort's sheet has been imported after the
video was ready and still has no scores for it, the session was never scored by hand and the video
is filed without the comparison (`scores_never_came`: with nothing to compare to, nothing can
disagree). A cohort with no successful import at all waits if the manifest's
`sheets_seen` lists it (its first import is coming) and otherwise files after the manifest's
`full_import_at` (the last import of every sheet) passes the time it was ready -- not the file's own
`written_at`, which a hand import of one other cohort also moves. A check that cannot run never files
the video, and its item is marked `needs_person` (kind `hand_score_check_failed`), so a wrong setting
shows in red instead of quietly stopping the filing. WHY: filing means done, so a video must not leave processing until it
has no issues; before this the comparison ran later, on videos already filed, and moved them back.
Tests: `tests/test_hand_score_check_before_filing.py`.

The same scores are also used by the review window (since 2026-10-28) to ask a person to look
again before saving an answer that disagrees with BOTH the algorithm and the hand score
(`review/double_disagreement.py`; docs/REVIEW_TOOLS.md "Check your answer"). It reads the score the
check kept in the video's outcomes file (`hand_scores.score_from_outcomes`, see "every pellet's hand
score is kept with the video" above), not the scores file: a reviewer needs no setting, and a video
analysed before scores were kept simply gets no box. (For a few hours on 2026-10-28 it read the
scores file through `pellet_hand_score`, which is removed.)

The Pipeline Dashboard shows every item of every list in `<NAS root>/attention/` whose
`needs_person` is true in a red "Needs your attention (N)" panel at its top (hidden when there is
none), and counts the waiting ones on a grey line (`watcher/attention.py`, `dashboard/widget.py`
`_update_attention`; every 5 minutes and on Refresh). Tests: `tests/test_attention_panel.py`.

## Update 2026-10-22: the shared version index is no longer in WAL mode

`pipeline_records/version_index.db` (on the shared drive; written when a manifest is written, read by
the dashboard on any machine) opened itself in SQLite's WAL mode. SQLite documents WAL as unsafe on a
network filesystem: its `-shm` index is shared memory, which machines reading the file over the share
do not share. It now uses the default rollback journal (`PRAGMA journal_mode=DELETE`), which also
converts a file left in WAL by older code. Writes stay best-effort and the index stays rebuildable from
the manifests. Tests: `tests/test_version_index_journal.py`.

## Update 2026-10-22: a re-pose request is never claimed twice

A graphics-card machine claims a re-pose request by moving it into `Repose_Queue/.inflight/`. The move
was `os.replace`, which overwrites: a request published again while another machine's claim for the
same video was still in flight was claimed a second time, and two GPUs posed one video. `_claim` in
`watcher/repose.py` now refuses an existing claim (os.rename on Windows; a hard link elsewhere), the
second machine logs "another node already holds this re-pose" once and leaves both files alone, and
the leftover request is closed by the existing "already satisfied" check when the pose lands. Tests: `tests/test_repose.py`.

## Update 2026-10-22: recording days and hours

A recording PC paused whenever its recording program was open, and a program left open after the
day's recording kept the PC idle until someone closed it. A lab usually knows when recording can
happen at all, so that is now configuration: `watcher.recording_days` (e.g. "Mon-Fri", or a list) and
`watcher.recording_hours` (e.g. "08:00-18:00", may cross midnight). Outside them the guard does not
even read the process list and an open program does not pause the watcher; inside them nothing
changed. Both empty by default (any day, any hour = the old behaviour); an unreadable value is treated
as empty, which pauses more, never less. Set from Watcher Control ("Recording happens only on" /
"...between") or `mousereach-watch-recorders --days ... --hours ...`; a running watcher follows edits.
This replaces the "pause only while a recording is being written" mode built earlier the same day
(cc14be4, withdrawn in 410bfb8): the schedule is a surer signal than the files. Tests:
`tests/test_recording_hours.py`, `tests/test_watch_recorders_cli.py`,
`tests/test_control_widget_recording.py`.

## Update 2026-10-22: a finished video carries no error message

A row's `error_message` is its current error, but reaching 'archived' did not clear the previous one:
archived rows could still show failures from runs that later succeeded (for example "Archive
directory not found for reprocessing" from a share outage), and every status view read them as current. `update_state` and `force_state` now clear it on the way to
'archived', and each watcher start clears it on rows already archived (`clear_errors_on_archived`;
logged once with the count). Other states keep theirs ('unresolvable' uses it for its reason). The
history stays in `processing_log`. Tests: `tests/test_archived_clears_error.py`.


## Update 2026-10-22: a posed video is claimed with an exclusive create

Before taking a video in from `Processing/Posed`, a processing machine claims it with a marker in
`Processing/Posed/.claims/`. The marker was written, read back after half a second and trusted --
two machines could both believe they had won -- and every marker was deleted after 24 hours whatever
its holder was doing, so a video one machine had held for a day (in review, say) could be taken in
by another. The claim is now the exclusive create the review queues already use
(`watcher/claim_marker.py`): exactly one racing machine's create succeeds. The holder keeps it while
the video's files are in the posed folder (through analysis, review and failure -- intake copies the
files and leaves them, and only archiving removes them), and lets go after archiving has removed
them; if a file cannot be removed the claim stays. A sweep every 10 minutes (`_sweep_posed_claims`,
shown as "posed claims" in the cycle timing line, formerly "stale claims") breaks another machine's
claims once that machine has stopped writing its status report for 3 hours (file-server clock);
claims every video in the posed folder that this machine has in hand and that has no claim -- under
the 24-hour rule most posed copies of videos in review had lost theirs, so to any machine taking in posed videos they looked like waiting work; and lets go of this
machine's claims on videos that have left the folder or are waiting again. A machine that
finds a video held elsewhere records that it did no work (it used to count the refusal as progress
and pick the same video again at once) and leaves the video out of its choices for 10 minutes, so
the next one in the work order is taken in. A claim folder that cannot be made means no intake, not
intake without a claim. (The review-queue half was d4876f4.) Tests:
`tests/test_posed_intake_claims.py`.

## Update 2026-10-22: setup finds the lab's own profiles, and a profile fills in the whole role

`mousereach-setup` and the Setup panel pre-fill a machine's settings from the lab's machine
profiles. Four defects, all in `setup/wizard.py` and `setup/setup_widget.py`:
1. The lab's file sat beside the code, git-ignored, so no machine that pulls the code had one, and
   setup fell back to the shipped `.example` -- whose drive patterns matched real machines and
   offered them a retired folder layout. Now the file is looked for at `MOUSEREACH_LAB_PROFILES`
   (one shared copy), then `~/.mousereach/lab_profiles.json`, then beside the code; the template
   is never matched (it now holds placeholders only).
2. A profile's `hostname` rule was documented but never checked, so two PCs with the same drive
   layout took the same (first) profile. It is checked now; every rule given must hold.
3. Only five watcher settings were taken from a profile; `mode`, `also_process`,
   `pause_while_running` and the recording days and hours were dropped, so a recording PC set up
   from scratch would not pause for the recording program. Every profile setting is now filled
   in where the machine has none, and other sections (e.g. `backup`) are added when missing.
4. The Setup panel's Detect overwrote both folders with the profile's, and Save let the profile's
   watcher settings replace the machine's. It now fills only what is missing, like the wizard.
A machine's own settings are never replaced by a profile. Install Step 6/7 rewritten. Tests:
`tests/test_setup_lab_profiles.py`. Also: `mousereach-setup --list-roles` names the file it read and
marks the role this machine gets ("<- THIS MACHINE (hostname match)"), and an identity file
(`~/.mousereach/machine_role.json`) naming a role that is not in the profiles is reported instead of
silently skipped.

## Update 2026-10-23: the census lists collages by name

The pipeline census (`mousereach.census`, which an integrator may run on a schedule with a time limit)
found collages with `rglob("*")` and an `is_file()` check on every entry: one round trip to the share
per file. On an archive root of many files, few of them collage-shaped, that could run past the
caller's time limit, leaving the caller with the previous run's counts. `collect_collages` now walks with `os.walk` and
filters on the name, the walk's own listing saying which entries are files -- the rule
`walk_analyzed` already followed -- with `walk_onerror` keeping rglob's rule for a folder that
cannot be listed. Tests: `tests/test_census_collage_listing.py`.

## Update 2026-10-23: a ruler that varies within a video never holds it

The quality check measures the ruler (the SABL-SABR distance) across the video. A variation over 15%
was a 'critical' flag, which sent the whole video to deep review for a person. It is now a warning:
recorded in `_triage.json`, never a reason to hold. Why: the millimetre scale is an
estimate (one median per pellet segment, over the segment's steady middle), every kinematic also
exists in pixels, and no conclusion is drawn from a single reach -- data preparation and statistics
average such variation away. Videos already held for it are released by the review-return scan
(their stored flag is recognised by its text, `triage.RULER_UNSTABLE_DESC`). A pose file with no
ruler points at all is still critical: that is a wrong model, not variation. Tests:
`tests/test_release_lifted_qc_holds.py`.

## Update 2026-10-23: Easy and Flat are session-only -- not segmented, never held, reaches and attention

Design: Easy and Flat trays are never held for a person; their pellet outcomes do not
matter (pellets are presented, but which pellet a reach went to is not used); they are not segmented;
they need no kinematics; their attention is wanted; they come last, behind every other priority. One
test decides it everywhere: `config.is_outcome_free_tray` (tray letter E or F).
* **Not segmented.** The segmentation step writes one whole-video span (a single boundary at frame 0,
  `segmenter_algorithm` "whole_video_outcome_free_tray") instead of running the segmenter, so reach
  detection runs over the whole session and the ruler is one estimate for the video. Checked before
  the keep-human-boundaries rule. Verified on a filed Easy video: the same reaches either way, and a
  whole-video ruler close to the per-segment median.
* **Never held.** The gate passes them (`review_gate.evaluate_gate`); the quality check no longer counts
  their segments. Videos already held by the machine (a quality flag or failed segmentation) are
  released by the review-return scan even when a person's file is in the bundle -- what a person can
  answer there is an outcome -- and that file is kept where it is, in the bundle folder, with a
  note, listed under 'Folders left for a person' in the Review Queues tab (earlier versions kept
  it with the set-aside outputs). A session a person
  flagged by hand stays.
* **Reaches and attention, no kinematics or outcomes.** After the gate, `FeatureExtractor.
  extract_session_only` writes the features file: each reach's id and frames and the span's
  `attention_score`, every kinematic and outcome field empty. The database takes reaches only from a
  features file, and the session's reach count and mean attention are worked out from those rows.
* **Filed ones.** The version scanner still never re-runs Easy/Flat for version changes (they would get
  nothing they use, and a re-pose would spend GPU time), with one exception: a filed one with no
  features file -- every one filed before this day -- is marked outdated from segmentation once, pose
  kept. The work order's idle-only rule (`idle_only_tray_types` E, F) runs it only when nothing else waits.
Tests: `tests/test_easy_flat_session_only.py`, `tests/test_release_lifted_qc_holds.py`.

## Update 2026-10-23: a lost recording is rebuilt from its singles; singles are never fed

The input to MouseReach is the multi-animal recording: the pipeline's own cut is what records which
recording and grid position each single came from. Feeding singles into the singles intake (as the ASPA
feed did, or by hand) skipped that. Now:
* `mousereach-rebuild-recording` (video_prep/core/rebuild.py) rebuilds a recording that exists nowhere
  from its singles: each one back in its 480x540 cell of the 1920x1080 grid (the cut's own layout,
  `cropper.CROP_COORDS`), black for an empty position, refusing singles that do not line up in time or
  are not 480x540, never replacing anything. The file says it is rebuilt, and a record goes to
  `<nas_root>/pipeline_records/rebuilt_recordings/`. It is written into the recording intake.
* `mousereach-aspa-feed` rebuilds ASPA recordings that are lost instead of copying cuts: a cut whose
  recording survives in the archive is left to the recording import; otherwise the cuts of one day and
  tray that are the same length to the frame are rebuilt as one recording (every cut of a recording is
  exactly as long as it). Positions come from a recorded group holding all of them (nearest day), else
  id order; the record says which. A position no cut fills is empty. More than eight cuts of one length,
  or an unreadable cut, is named for a person. (Copying the nearest recorded day's groups instead is
  not safe: groups change from day to day, so it would pair cuts of different recordings.)
  Also for a person, never rebuilt: cuts as long as a recording that survives that day (they came out
  of it under a name it does not list -- an identity question), cuts as long as another cut of that
  day whose recording survives (same reason; this catches a recording trimmed after it was cut), and
  cuts under a minute long.
* `mousereach-aspa-import-collages` reads two kinds of name that are not in standard form: a
  separator typo between ids (read as a comma when that gives exactly eight ids and one reading), and
  one recording holding two cohorts on different runs (`_AP3,BP1`), copied in once per run with the
  other cohort's seats empty. Each is listed by the dry run and kept in the mapping file with its
  original name. Before this, such a recording was left out and its animals' cuts looked lost.
* `mousereach-aspa-name-audit` (aspa/name_audit.py) checks every archive recording's name against
  ASPA's own cuts (each cut traced to its recording by length, or through the recording naming it or a
  cut exactly as long when the recording was trimmed after it was cut; every disagreement confirmed by
  the picture -- the cut matches one seat of the recording far more closely than any other --
  because two recordings of one day can be a few frames apart in length by chance) and writes
  `<nas_root>/pipeline_records/archive_name_conflicts.json`: recordings whose seats the cuts name for
  animals they do not list. The import runs the same check on the days it imports and does not copy a
  contested recording in. A recording held in quarantine counts as in the pipeline for the import and
  the feed. Nothing is renamed: which name is right is for a person.
Tests: `tests/test_rebuild_recording.py`, `tests/test_aspa_feed_to_shared_folder.py`,
`tests/test_aspa_name_audit.py`.

## Update 2026-10-23: triage shows why, and never takes an answer the pipeline would drop

Videos a reviewer had answered could come back: an answer was dropped by the review matching without a
word on screen -- because it named a reach outside the segment's frames, or answered a segment only a
few frames long -- and the video then asked again, sometimes about a segment already answered more than
once. Now:
* the review tool refuses those two kinds of answer at the moment of answering and says why and what
  to do -- if the event belongs to another pellet, save nothing and Escalate
  (`causal_review_io.answer_problem`, the same tests `index_review_by_segment` applies);
* a banner in triage lists every open question and every earlier answer that was not used, with the
  reason in plain words (`plain_review_note`);
* a triaged segment too short to be a pellet presentation sends the video to deep review with a
  segmentation reason (gate) and moves a bundle already in triage there (return scan), because no
  answer in triage can count for it; a person's deep-review clear that kept the cuts means there is no
  pellet there to ask about.
* the banner also gives the routing reason, and beside each pellet the reason it was sent back when
  another system marked it: `mousereach-route-to-queue` takes per-pellet words in a worklist
  (`segment_details`) and stores each as that pellet's `flag_reason` -- so an integrator's bench check
  can say, per pellet, what the hand score and the algorithm each say. Finished videos sent back for
  a new pellet had looked like the reviewer's earlier answers being lost.
Tests: `tests/test_triage_answers_that_cannot_count.py`.

## Update 2026-10-24: posing on a machine's CPUs (`mousereach-cpu-pose`)

Posing is the slow step, and only GPU nodes did it. A machine with many idle cores
can add to that by running `mousereach-cpu-pose` beside its watcher.

**What was measured.** One DeepLabCut process on the CPU is slow and uses only a few
cores, whatever thread count it is given -- so speed comes from many small processes side
by side, until they start slowing each other. A CPU pose is the same as a GPU pose for
practical purposes: compared frame by frame with a GPU pose of the same video, body-part
positions differed by a fraction of a pixel, no confident point differed by more than 1 px,
and no point crossed the 0.5 likelihood line.

**What it does, step by step** (`watcher/cpu_pose_pool.py`), with the GPU nodes' own code
wherever it exists:

1. **Picks** the next unclaimed video at the top of `Unanalyzed/Single_Animal` in the
   work order (`work_priority.best_tier_choice`: best group first, random inside it, idle-only
   trays last).
2. **Never re-poses**: a video finished or held for a person is skipped, by the GPU node's own
   rule (`DLCOrchestrator._already_done_or_held`), before and after claiming.
3. **Claims** it by renaming it into `.inflight/<hostname>-cpu/` (`single_claim.claim_single`),
   so no two machines pose one video. Other machines see the pool alive through its status
   report `Processing/.node_status/<hostname>-cpu.json`, and take its claims back only after
   it has been quiet for hours, as for any node.
4. **Poses** a copy on the local disk (`<processing root>/CPU_Pose`) in a child process at
   below-normal priority, up to `--workers` at once, `--threads` each.
5. **Hands on** to `Processing/Posed` the GPU nodes' way: every file copied under a temporary
   name, then renamed into place with the pose (.h5) last -- the processing watcher keys on
   the .h5. The claimed copy is removed only once the handed-on video is confirmed. The pose
   must come from the declared model, or it is not handed on.
6. **Records** `dlc_running` and then the hand-off on the shared record (watcher_central.db),
   as a GPU node does.

**When something goes wrong.** A failed pose puts the video straight back in the shared folder
for any machine and keeps the pool off it for an hour; its log stays in `CPU_Pose`. At start the
pool hands on any pose a previous run finished and gives back any claim it left. Stopping:
`cpu_pose_stop.flag` in the processing root (no new videos; running poses finish and are handed
on; then it exits), or Ctrl+C / ending its task (running poses are stopped; their videos are
given back at the next start).

Tests: `tests/test_cpu_pose_pool.py`.

## Update 2026-10-27: a video leaves the re-segmentation queue only once it is re-segmented

The deep-review queue (`Processing/Review/deep_review`) is where a video's cuts are fixed.
Until now a clear marker (`{video}_deep_review_cleared.json`) released a video on its own,
whatever was in it and whenever it was written. That let an escalated video bounce: the
triage tool's "Escalate: bad segmentation" button moved it here, a clear left over from an
earlier deep review went with it, and the return scan released it within seconds; it re-ran
on the same cuts and was back in triage before anyone could fix them.

What holds now:

1. **Escalating sets aside an earlier clear.** The button
   (`review_gate.escalate_to_deep_review`) renames any clear in the bundle to
   `{video}_deep_review_cleared_superseded_<time>.json` before moving it. The file is kept
   and travels with the bundle; nothing reads it as a clear. Only the button does this --
   the watcher's own routes into the queue fire only when there is no clear.
2. **A clear releases a video only if a person has re-segmented it since it arrived**
   (`review/resegmented.py`): the segments file was saved by a person in the segmentation
   fixer (`boundary_source: "human"`, `corrected_at`) after the bundle's `routed_at`.
   Hand-set cuts from before it arrived do not count. With no readable `routed_at` the
   arrival is the latest of the routing record's modified time, the review manifest's
   `provenance.staged_at` and the bundle folder's creation time
   (`resegmented.entered_deep_review_at`); it used to be "any time", so any hand cut ever
   saved counted. A bundle that cannot be dated cannot leave. A ground-truth file does not count
   either (since 2026-11-17): nothing writes its boundaries into the segments file, so a
   video released on it was cut again by the segmenter exactly as before. And "cleared"
   means the clear marker and nothing else -- the return scan
   (`review_return._deep_review_cleared`) makes the same test as the review gate; a
   ground-truth file in the bundle used to count as a clear by its presence alone.
   In the triage queue's scan, a bundle whose segmentation reported failure is diverted to
   re-segmentation unless it has a clear marker or ground truth certifying the WHOLE video
   (the gate passes such a video clean, `review_gate.evaluate_gate`); partial ground truth
   no longer stops the divert (the gate would send that video to re-segmentation anyway). The return scan enforces it (one log line per video per
   watcher run); the review tool's Clear button and its complete-answers auto-clear refuse
   and say why; `mousereach-review-release` and the Review Queues tab list such videos as
   "not re-segmented since they arrived". Easy and Flat trays are never segmented and are
   exempt.

After the person's save, the video re-enters the pipeline with the person's cuts as its only
cuts; everything computed from cuts is recomputed from them, and if questions remain it goes
to triage again -- now about the new segments.

Tests: `tests/test_escalation_voids_old_deep_review_clear.py`.

Names: from 2026-10-27 the tools and operator docs call this queue the
**re-segmentation queue** (and its review tool Re-segmentation Review);
"deep review" is kept for ground-truthing with the ground-truth tool. The
sections above keep the old name as they were written. Folder, database state
and file names are unchanged.

## Update 2026-10-28: lab-wide settings come from one file on the storage drive

Before this, every setting was read only from the person's own `~/.mousereach/config.json`. A
setting the whole lab shares therefore had to be typed into every person's file on every PC, and
anyone whose file lacked it silently got nothing: a reviewer with no `hand_scores` section never saw
the review window's "check your answer" box.

Now `config.lab_setting(key)` looks a setting up in this order: the person's own `config.json` if it
has the key (the whole value, sections are not merged), else the lab settings file
`<nas_root>/mousereach_lab_config.json`, else unset. The file is found from `nas_root` alone, which
every machine and user already has: no environment variable, no per-user or per-PC step. Both files
are read on every call, so an edit reaches a running watcher and an open review window. A missing or
unreadable lab file (not JSON, or not a JSON object) is reported once per run as a log warning naming
the file, and the setting stays unset -- the same as before the file existed. A user config with no
`nas_root` has no lab file and says nothing. (Not for the hand-score settings and the filing rules:
those two stop instead of reading "unset" -- see "an unreadable lab settings file holds the
hand-score check" below.)

WHY a new file rather than the lab profiles file (`setup/lab_profiles.json`): that file holds per-PC
DEFAULTS that `mousereach-setup` copies into a PC's `config.json` once, matched by hostname and found
through `MOUSEREACH_LAB_PROFILES`. The lab settings file is the opposite: read live, the same for
everyone, never copied. Keeping them apart keeps it clear whether an edit takes effect by itself
(lab settings) or only after setup is re-run (profiles).

Reader of it today: `hand_scores.settings()`, which every hand-score reader goes through -- the check
before triage (`check`), the scores-file stamp the watcher uses to avoid re-checking a waiting video
(`scores_file_stamp`) and the attention list (`_write_score_attention`). The review window reads no
setting: it reads the score the check kept in the video's outcomes file (`score_from_outcomes`). The "check could not run" attention message now names the lab file as the usual place to fix
the setting. Operator steps: INSTALL.md, Part 6, Step 4. Tests: `tests/test_lab_config.py`.

## Update 2026-10-28: startup recovery no longer adopts another machine's claimed single

A single that a machine has claimed for pose sits in `Single_Animal/.inflight/<that machine>/`
on the shared drive, and the shared record's file path for the video points at that claimed
copy. Because the folder is shared, that file exists for every machine. When a graphics-card
watcher started, cross-node recovery asked only "does the recorded file exist?", got yes, and
took the other machine's in-progress ('dlc_running') video as its own. The startup reclaim then
put it back in this machine's queue, and the video was posed a second time. Every restart of a
graphics-card watcher did this to all the videos the CPU pose pool was posing at that moment.

Recovery now treats a file in ANOTHER machine's claim folder as "not on this node"
(`coordination._file_on_this_node`). The video is recorded as being elsewhere (unresolvable on
this node, with the holder named), exactly like any video whose files are on another machine,
and is never picked up here. A file in this machine's OWN claim folder is still its own.

What an operator sees: nothing, except that a graphics-card watcher no longer spends its first
minutes after a restart on videos another machine already has. A machine that adopted such
videos before this change records them as elsewhere at its next restart.


## Update 2026-10-30: Filing folders come from the lab settings

**What was wrong.** Which folder under `Analyzed` a project's finished videos go
in was written into the code: a fixed map of project codes to folder names,
and per-cohort folders for two codes only. A second project of the same grant
was therefore filed BESIDE the grant's folder, not inside it, all its cohorts
in one folder, and changing that needed a code change. Several lookups also
assumed a filed video sits exactly two folders below `Analyzed`
(`<project>/<cohort>`): the version scan's index, its per-video manifest
lookup, and the route tool's outcomes lookup.

**What it is now.**

- `config.filing_rules()` reads the `filing` section of the shared lab settings
  file (`<nas_root>/mousereach_lab_config.json`) -- the lab file ONLY, never a
  person's own config.json, so every machine files in the same place. It is
  read at most once a minute. A malformed rule raises, and the archive step
  stops with that message rather than filing somewhere unexpected.
- `AnimalID.get_project_and_cohort` applies the rule; a project with no rule
  files under `<CODE>/<CODE>`.
- `archive.core.filed_folder(video_id, root)` is the one answer to "where is
  this video filed"; `get_archive_destination` is it under `Analyzed`.
- The version scan's index walks every folder (the pruned walk of
  `pipeline/analyzed_tree.py`), at any depth. The per-video manifest lookup
  and the route tool look in the filed folder first and walk the tree only if
  the file is not there.
- Review bundles that carry no copy of their video record absolute paths into
  `Analyzed`. When the manifest's path no longer exists and the bundle has no
  copy, `review/bundle_media.py` now looks where the video is filed now, and the
  per-model pose tree beside it (`<project folder>/DLC Model <N>/<cohort folder>`).
  The queue index and the review window look there too for saved reviews.
- The work-priority labels of a video include every level of its project
  folder, so the work order may name a grant or a project.

**Moving videos when a rule changes.** Changing a rule moves nothing. Until the
filed videos are moved, the reconcile report lists them as in the wrong place,
and the watcher re-runs nothing because of it. To move them: stop the
processing watcher with its stop flag; move each old cohort folder to its new
place on the same drive (a rename; nothing is copied); move each per-model pose
tree's cohort folders the same way (`<project folder>/DLC Model <N>/<cohort
folder>`); restart the watcher. mousedb's reach import recognises a results
file that only moved (same name, size and time as a ledger entry whose old path
is gone) and carries its ledger entry over instead of importing it again.

**Not changed.** `review/staging.py resolve_canonical_paths` still looks for a
video in `<root>/CNTnn/`, which matches no layout the filing rules produce; the review lookups
that depend on it ("beside the canonical video") find nothing there today,
exactly as before (see the 2026-10-22 note above). Making that lookup work
would change which copy of a review wins and add a lookup per video to every
version scan, so it is left for its own change.

Tests: `tests/test_filing_rules.py`.

## Update 2026-10-30: the version scan walked the archive once per manifest-less video

**What was wrong.** The scan builds one index of every processing manifest in
`Analyzed` (`ReprocessingScanner._index_archive`). For a video the index did not
hold, it fell back to
`iter_files(archive_dir, "<stem>_processing_manifest.json")` -- a full recursive
walk of the whole tree, to look for ONE file name. A video with no manifest on
disk never satisfies that fallback, so it paid a whole-tree walk on every pass,
for ever, and the cost grew with the tree.

**What it cost.** Measured read-only against a copy of a watcher's own database
and cache, nearly the whole scan was the per-video loop, not the one index walk,
and nearly all of that loop was the fallback walks: one full walk for each video
with no manifest. Pass times grew far faster than the archive did, so this was
not corpus growth. **The orchestrator processes nothing while a scan runs**, so
a video taken in could wait hours to start.

**What it is now.** `_deep_manifest_index()` does that walk at most ONCE per
scan, on first need, and every missed stem is served from it; it is dropped at
the start of each scan because the tree changes between passes. Same tree, same
exclusions (never inside a superseded folder), same first-hit-wins order, so the
manifest found for any given stem is unchanged -- scanning the same archive
gives the same scanned, current, outdated, no-manifest and error counts either
way.

**Still per-video, deliberately.** `_segments_human_fixed` and
`_divert_mislabel_to_deep_review` each walk the tree, but are reached only for a
video whose pending review declares a segment mislabel -- a rare case. Left alone rather than widening the change on an unmeasured cost.

Tests: `tests/test_version_scan_walks_once.py` asserts the WALK COUNT does not
grow with the number of videos the index missed, and that the fallback still
finds a manifest filed outside the indexed folders while still never reading a
superseded one.

## Update 2026-10-30: ignore windows reach the kinematics; a review handed as a file is read; failed videos are retried

Three open items, closed together.

**Ignore windows now change the measurements.** The review tool's "Ignore
windows" box marks frame ranges where something other than a reach moved the
pellet. The answer was saved as `answers.abnormal_ranges`, carried onto the
segment as `abnormal_ranges`, and read by nothing outside `review/`. Now
`FeatureExtractor.extract` (kinematics/core/feature_extractor.py,
`split_reaches_by_ignore_windows`) leaves out every reach that overlaps a
window by any frame before the temporal context,
the pellet positioning and the per-reach features are computed. The segment
record gains `n_reaches_left_out`, `reaches_left_out` (each left-out reach's id,
frames and the window) and `ignore_windows`; `n_reaches` is the number
measured. Extractor version 2.1.0 -> 2.2.0. The output is the same for a video
with no ignore windows, so `pipeline_versions.json` declares 2.1.0 compatible
(the segmenter 2.2.3 precedent) and the archived videos whose review carries
a window are marked for a kinematics-only re-run by hand, once, instead of
staling the whole archive. They are found as every file in
`review_records/reviews/` with a non-empty `abnormal_ranges`.

**A review handed to the truth layering as a FILE is read.** The extractor
passes its `review_path` -- the review file -- to `resolve_truth_layers` as
`primary_dir`, and `truth_resolver._review_doc` tried only `<dir>/<stem>/<file>`
and `<dir>/<file>`. A review saved only in the processing folder was never
read by the layering; the durable copy of record normally covered it. The
lookup now accepts the file itself, and only when it is this video's review by
name.

**A failed video is retried later.** A crash in segmentation, reach or outcome
detection marked the video 'failed' and no work bucket ever looked at that
state again (the embedded report above still says "No watcher ever picks up a
video in state 'failed'"; that was true until this update).
`ProcessingOrchestrator._select_work_item` has a new
drain bucket between archive and intake (first placed after the pipeline
bucket; moved ahead of intake so that a due retry runs when its hour is up
rather than whenever intake and pipeline both happen to come back empty): a 'failed' row is put back to
'processing' and run when `_failed_retry_due` holds -- `error_count` below
`watcher.max_retries` (3), at least `RETRY_FAILED_AFTER_S` (one hour) since
`last_error_at`, and the video and a runnable pose still in this node's
Processing folder. The attempt is written to the processing log as step
'retry'. A video that has used up its retries stays failed and is listed by
`mousereach-watch-status`; `mousereach-watch-reprocess` still resets it. The
case this is for: a transient failure such as "[WinError 5] Access is denied"
on a `_segments.json.tmp`. The GPU role is unchanged: it already re-drives a failed pose at startup.

Tests: `tests/test_ignore_windows_leave_reaches_out.py`,
`tests/test_truth_resolver_reads_review_file.py`,
`tests/test_failed_videos_are_retried_later.py`.

## Update 2026-10-30: a GPU node's shared-record writes survive an unreachable shared database

Found on a GPU node restarted while `watcher_central.db` was locked. Four gaps in
`DLCOrchestrator`'s shared-record code, all closed in one change:

1. **Writes were dropped while the coordinator was unavailable.**
   `_sync_to_connectome` returned before queuing when `self.coordinator` was
   None, so every state change between a failed startup connect and the
   reconnect was lost for good (a hand-off to Posed, a pose start). Now the write is queued in the same failed-writes list that a
   locked database already fed, and the reconnect lands it.
2. **A write that failed just before a stop was lost with the process.** The
   failed-writes list is memory only. It still is -- the durable copy is the
   node's own `watcher.db` -- because the startup fill now brings the record to
   it (3).
3. **The startup fill only ADDED rows that were missing entirely.** A row of
   this host that existed but was behind stayed wrong. `_put_missing_on_shared_record`
   now also corrects every row carrying this host's name whose state differs
   from the local one, with `sync_video_state_unless_newer`, so a row updated
   since (by this host's later write or another host's) is left alone. Rows
   owned by another host are never touched. It also marks itself pending when
   it could not run at all (coordinator None), so the reconnect's next scan runs
   it instead of forgetting it.
4. **The reconnect was tried only when a collage came up for selection**, so a
   node that started mid-pose stayed disconnected until that pose ended. The
   scan phase now tries `_retry_coordinator` first, and a successful reconnect
   schedules the reconciliation for that scan.

The processing server's `ProcessingOrchestrator` does not use this path (it has
no coordinator and writes `videos`/`processing_log` through `export_to_central_db`,
whose own re-send is `_export_missed_to_central`).

Tests: `tests/test_shared_record_survives_a_lost_connection.py`.

## Update 2026-10-30: failures that used to be silent in review and the watcher

**Clearing a deep review now stops if the answers did not save.** It used to carry
on, sending the video onward and losing what the person had entered. They now see
"Your answers did not save" and the video stays where it is.

**A database column that cannot be added is now reported.** Only "already exists",
the normal case, stays quiet.

**These now write a line to the log instead of saying nothing.** What the code does
next has not changed, so look in the log when a video behaves as if something were
missing:

- An unreadable review, status, routing or segments file, which still counts as
  absent. A file that does not exist is normal and stays quiet.
- An unreadable human review file in the truth layers, logged as an ERROR, because
  the video is then measured without those answers.
- A release that could not be undone, logged as an ERROR, because that bundle is
  left split between two folders and needs a person to put it back together.
- A triage verdict that could not be written into a results file, a leftover file
  after archiving, and a failed check of the archive or of this node's own videos.
- A saved reach answer that could not be restored to the screen, and a video left
  out of the re-segmentation queue because it could not be read.
- Old ground-truth corrections that could not be applied.

The log is not somewhere a reviewer looks, so only the first change above reaches
the person at the moment it happens.

**The rest were read and left alone.** Of 149 silent handlers in these two folders,
the others only affect the display, an optional feature, or something retried on the
next pass.

## Update 2026-10-31: both ends of a reach are anchored at the nose

**What changed.** A reach starts and ends at the nose, and the hand is hidden there
just before it is detected and just after it disappears. Every measure of where the
paw is now counts two extra frames: the frame before detection, with the hand between
the left slit corner and the nose, and the frame after the paw disappears, with the
hand between the right slit corner and the nose. Before, the wrist velocity and
trajectory columns ignored these frames, and the per-point columns placed them using
the wrong frames.

**What to expect in the numbers.** Paths get longer. Peak speed could be set by the
jump to or from an anchor until the 2026-11-02 update below. Duration does not change, because it still counts detected
frames. The extractor is version 2.3.0 and no older version counts as current, so
every filed video has its kinematics recalculated, with old results kept in the archive.

**When a landmark cannot be trusted.** Each reach row says how each anchor was placed,
and a reach with no anchor has empty path measures. It is not measured from a guess.

**Where the detail is.** `kinematics/core/anchored_path.py` explains the rule, and
`docs/KINEMATICS_FIELDS.md` section 6.1 lists every place that reads paw position and
why it does or does not use the anchors.

## Update 2026-11-02: peak speed and speed at the apex leave out the anchor steps

**What changed.** Peak speed and speed at the apex, for the wrist and for each paw
point, now come only from steps between two detected frames. The first and last steps
of a path join a calculated point to a detected one and are charged to a single frame,
so they say how far apart those places are, not how fast the hand moved. Measured over
them, the peak was the anchor jump on about half of all reaches. The path, its length,
its straightness, mean speed and velocity correlation still run nose to nose.

**What to expect in the numbers.** The wrist's peak speed and speed at the apex return to
what they were before extractor 2.3.0 on the same reach. A reach whose furthest point is its last
detected frame has no speed at the apex, because no detected frame follows. A reach of
one detected frame has no peak speed. A reach with no anchor now does have both, since
neither reads an anchor. No column was added or removed.

**What happens to filed videos.** The extractor is version 2.4.0 and no older version
counts as current, so the watcher's version scan marks every filed Pillar video for a
kinematics recalculation (old results go to the archive) and works through them on its
own. Until it finishes, a file can hold rows from both versions: `extractor_version` on
the row says which.

**The midpoint is settled.** "Between the slit edge and the nose" is the midpoint, held
in `ANCHOR_FRACTION_FROM_NOSE`. It was an assumption before this update.

**Where the detail is.** `tracked_steps` in `kinematics/core/anchored_path.py` and
sections 5.4, 6.1 and 6.2 of `docs/KINEMATICS_FIELDS.md`.

## Update 2026-11-03: a recalculation re-runs no algorithm and no triage, and never re-asks a person's answer

**The rules.** Only the calculations that changed are recalculated, without re-running the four
algorithm steps or the triage. And a segment whose boundaries, causal reach and outcome have all
already been reviewed by a person never, under any condition, requires a person to look at it again.

**What a version bump now does, by what changed.** The version scan marks a filed video
'outdated' with the earliest stage whose version changed (`reprocess_scope`). The re-run then:

| What changed | Recomputed | Quality triage and hand-score check | Review gate |
|---|---|---|---|
| Only the kinematic extractor (scope `kinematics`) | feature extraction only | **skipped** | runs; asks only about segments no person answered |
| A saved review the filed kinematics have not applied yet (also scope `kinematics`) | feature extraction only (it applies the review) | **skipped** | runs; same |
| Outcome detector (scope `outcome`) | outcomes, reach assignment, kinematics | run | runs |
| Reach detector (scope `reach`) | reaches onward | run | runs |
| Segmenter, or a new pose (scope `segmentation` / `full`) | everything | run | runs |

The quality triage and the hand-score comparison judge the four algorithms' outputs; when none
was recomputed they are not run again. The watcher log says "filed video, re-run for 'kinematics'
only ... no quality triage, no hand-score check", and the watcher database gets a `triage` step
marked `skipped`.

**Ground truth beside the video is an answer the gate now sees.** The gate and
`mousereach-route-to-queue` read ground truth through one function,
`triage_status.gt_resolved_segments`, which now looks in the video's own folder (the processing
folder of a run, or the video's results folder) before the corpus-wide index. Before this, only the index
was read. It is built once per watcher process, so ground truth copied into the processing folder
by a re-run was invisible, and videos whose open segments ground truth had answered were sent
back to a person.

**Still open, waiting on a decision.** (1) A filed video with a segment NO person ever answered
(the algorithm's own "triaged" outcome, filed before the gate existed or before the question was
answered): the gate still sends it to triage on a kinematics-only re-run, as it always has. Two
later changes start such a re-run in more cases, and each of them reaches this same gate: a review
saved for a filed video that has no processing manifest, and ground truth saved or changed after
filing (both described under their own "Update" headings below). So until the decision is made, a
filed video with a never-answered segment can be taken back to triage by either of those too, not
only by a version change. Nothing about what the gate does was changed by them. (2) A
saved answer the matching code sets aside because its causal reach lies outside the frames it
records, or because two saved answers for the same frames disagree (`index_review_by_segment`):
those segments still count as unanswered. The code that sets them aside says why -- a review-tool
fault once filed answers under the wrong segment -- so whether such an answer settles its segment
is a decision for the lab, not a guess for the code. (3) `mousereach-route-to-queue --recheck`
("are you sure?") still reopens a person's answer on purpose when someone asks for it. (4) The
queue's release check still reads ground truth from the index only, so a triage bundle holding
ground truth that answers every open segment is not released by it yet.

**When every check still runs:** a new video; a video coming back from a review queue (it
resumes just after the step that sent it (resume_stage), with every later check); a video held for its hand scores and picked again; a
filed video whose re-run had to remake any of segmentation, reaches, outcomes or reach assignment
(including a reused output that turned out to be missing); a video never filed (no
`archived_at`); and a video whose filing the watcher cannot read.

**A retry or a restart keeps the scope (changed later; see "Update: a filed video's re-run keeps
where it starts").** This paragraph used to say that a watcher killed in the middle of a
kinematics-only re-run, or a re-run that failed and was retried, ran the video again through every
stage and every check. It no longer does: the start stage is read back from the row.

**Where it lives.** `filed_rerun_reuses_analysis` in `watcher/orchestrator.py` (the rule and its
reasons), used by `ProcessingOrchestrator._run_pipeline`; `gt_resolved_segments` in
`review/triage_status.py`; tests in `tests/test_filed_rerun_skips_checks.py`.

## Update 2026-11-09: sending videos back for a new pose when the pose itself is not trusted

**The problem.** The re-pose round trip (`watcher/repose.py`, section 5 and the 2026-09-12 notes)
asks a graphics-card machine for a new pose only when the DECLARED MODEL changes, and every one of
its decisions goes by the scorer name in the pose file. A pose made with the declared model by a
machine whose poses the lab decides not to keep -- for example the CPU pose pool
(`watcher/cpu_pose_pool.py`), whose poses are close to but not identical to a graphics card's --
carries the declared scorer. So the version scan calls such a video current and never marks it;
and if someone marked it by hand, the publish and adopt steps would find the declared-scorer pose
still sitting in the posed folder or the results folder, treat it as "the new pose", and re-run the
analysis on the very pose that was to be replaced, with no graphics card involved.

**The command.** `mousereach-repose-videos --ids <file> --label <name> --reason <text> [--dry-run]
[--limit N] [--report <file.json>]`, run on the processing server. For each video, in this order:

1. Every live copy of the pose is moved out of every place the round trip looks: the results folder
   in Analyzed (and the same cohort folder of any `DLC Model N` tree), the review bundle if the
   video is held in triage or deep review, and the copy in `Processing/Posed`. The pose and every
   output computed from it (segments, reaches, outcomes, assignments, triage verdict, features,
   processing manifest) go to `Analyzed/Archive/<label>/` by the normal checksum-verified supersede
   move. **Why a separate archive folder:** the distrusted pose has exactly the same file name as
   the pose that replaces it; in the usual `DLC Model N` folder the two would be indistinguishable
   later.
2. It checks that no pose of the video is left in any of those places. If one is, the video is left
   as it is and reported "incomplete"; its row is NOT marked.
3. Only then is the row marked 'outdated', scope 'full', with the reason as its `mark_reason`. The
   reason is what keeps the version scan from un-marking a row whose versions compare current.
4. It writes the re-pose request into `Processing/Repose_Queue/`, naming the video file a
   graphics-card machine should copy.

**The video file is never archived or deleted.** A filed video's mp4 stays in its results folder
beside any human review, and the request names it. A held video's bundle is renamed, as one folder,
to `Processing/Repose_Queue/_parked/<video>/` (the video and the queue's notes are what is left in
it), the posed folder's duplicate mp4 goes to `_parked/<video>/posed_copy/`, the server's claim on
the posed copy is released, and the request names the parked mp4. **Why parked there:** a
graphics-card machine may not read from the posed folder (it is the server's intake) and the review
queues are not searched for videos, so the request must name a file somewhere else on the shared
drive; the request queue's own folder keeps everything about a re-pose in one place. Nothing reads
subfolders of the request queue as requests.

**Refused and left untouched:** a held video with a saved human review, deep-review clearance or
ground truth (a person is working on it); a row in any state other than archived, triage or
deep_review, or no row; a video whose request is already out; a bundle still being routed into a
queue. `--dry-run` prints what would move where and changes nothing.

**From there the existing round trip carries it unchanged:** a graphics-card watcher claims the
request, copies the video, poses it, stages the pose in `Processing/Posed`; the server adopts it
(scope narrows to 'segmentation', pose path recorded) and re-runs segmentation, reaches, outcomes
and kinematics on it, then the hand-score check, the review gate, and filing or triage. A filed
video's human review is applied again by the extractor (segments matched by frames; the causal
reach by its reach number in the old reaches file -- after a re-pose that renumbers reaches, check
that reach by its frames).

**A held bundle leaves its queue in one rename (fixed 2026-11-09).** At first the command moved the pose and outputs out of a held bundle while the bundle still sat in the triage queue. The watcher's review-return scan read one such half-emptied bundle as "segmentation failed" and moved the rest of it into deep review in the middle of the run. The bundle is now renamed, as one folder, to `Repose_Queue/_parked/<video>/` before anything is moved out of it. The rename runs under the claim the scan itself takes (`Processing/Review/.return_claims/<video>.claim`, held as `<host>-repose`), so the queue only ever holds the whole bundle or nothing. If the scan holds the claim, the video is refused for now. A run stopped after the rename finishes from the parked folder when it is run again. Tests are in `tests/test_repose_videos.py`.

**The re-run takes the video from the posed folder when there is no filed copy (fixed 2026-11-09).** A held video has no results folder holding its mp4. The re-run copies only DLC files from the posed folder, so a held video re-posed this way ran without a video, and the triage bundle it routed had none for the reviewer to open. `_reprocess_copy_set` (`watcher/orchestrator.py`) now adds `<video>.mp4` from the posed folder, then from `Repose_Queue/_parked/<video>/`, whenever no other folder supplied one. A filed copy still wins. Tests are in `tests/test_reprocess_copy_set.py`.

**What it does not do.** It does not touch the database integration: rows already imported from a
filed video's old results stay until the re-run is filed and imported, which replaces them. The
parked folders are left after the round trip closes; they hold duplicates of videos that have been
re-filed or re-queued.

**Where it lives.** `watcher/repose_videos.py` (`plan_and_move`, `main`); tests in
`tests/test_repose_videos.py`, including one that fails if the row is marked while any copy of the
pose is still in place.

## Update 2026-11-15: a video is filed only with its kinematics

**What a person sees.** A video whose kinematics could not be made is NOT filed into Analyzed. Its
row goes to `failed` with an error message that starts `not filed: no kinematics` and says why
(the extractor's error, or which input file was missing). Its files stay in the local Processing
folder. `mousereach-watch-status` lists it among the failed videos with that message.

**What happens next, with nobody doing anything.** On the processing server the failed-video retry
(see the update "failed videos are retried" above) runs it again once its wait is over, up to
`watcher.max_retries` times. If the kinematics work on a retry, it is filed as normal. If they keep
failing, it stays `failed` for a person: fix the cause, then `mousereach-watch-reprocess <video>`.
On a GPU machine with `also_process` true a failed video waits for a person in the same way.

**What counts as "no kinematics".**
- Pillar trays: the reaches file or the pellet-outcomes file is missing (kinematics never ran), or
  the feature extractor raised.
- Easy and Flat trays: the session file (reaches and attention) could not be written, including
  when there is no reaches file.

**Why.** Filing means done for good, and `<video>_features.json` is what the database takes from
the filed folder. Before this, a failed extraction was only a warning: the video went on to
`processed` and was filed with no features file, i.e. filed as finished with no kinematics.

**A pellet left undecided is not filed either.** The archive step also reads the features file it
is about to file. If a Pillar video's file still holds a pellet with no decided outcome (the
algorithm's `triaged`, an older tool's `uncertain`, or none), the video is not filed: its row goes
to `failed` with an error starting `not filed: a pellet is still undecided in its kinematics`,
naming the pellets. The same retry as above runs it again, through the review gate, which holds it
in triage if the pellet really is open; if it is refused after the last retry it stays `failed`
for a person (`mousereach-watch-status` lists it). A re-run of a video that is already filed and is
refused this way leaves the filed copy as it was. Easy and Flat trays are never checked (no outcome
is computed for them). **Why:** the gate and the kinematics read a person's answers separately; when
they disagreed (the gate saw ground truth beside the video that the kinematics did not), the gate
passed the video and the algorithm's `triaged` was filed and exported as the pellet's outcome. The
check sits where the file is filed, so no later disagreement between two readers can file an open
question. (`refuse_filing_with_undecided_pellet`, `orchestrator.py`; test
`tests/test_filing_refuses_undecided_pellet.py`.)

**The archive step checks too.** Before it moves anything, the archive step (both
`_archive_to_nas` on the server and `_archive_locally_processed` on a GPU machine) refuses a video
whose results folder holds its files but no `<video>_features.json`, and marks it failed the same
way. Why a second check: a row can reach `processed` other than by a fresh run (a person's reset,
an interrupted filing put back at startup). Not refused: a run whose features file was already
filed by an earlier, part-way attempt (see the next update), and an empty folder (the archive step
reports that itself).

**Where it lives.** `watcher/orchestrator.py`: `BaseOrchestrator._make_features` (one kinematics
step for both machine roles), `_hold_without_kinematics`, `refuse_filing_without_features`;
tests in `tests/test_filing_requires_kinematics.py`.

## Update 2026-11-15: filing is recorded before the files move; a filed run is never marked failed

**Before.** The archive step moved the files into Analyzed and only then wrote the row's state
(`processed -> archiving -> archived`, back to back). If the watcher died between the move and
those writes, the row stayed `processed` with the files already filed; the next attempt found
nothing in Processing ("No files found in Processing/") and marked a filed video `failed`.

**Now, on both machine roles:**
- `archiving` is written before anything moves. A row left in `archiving` therefore means "a
  filing attempt was interrupted"; the next watcher start puts it back to `processed` (startup
  reclaim, as before).
- A failed attempt that leaves the files in Processing (a file in use, the share gone) puts the
  row back to `processed` and backs off as before, so it is tried again. (`archiving ->
  processed` is a legal transition for exactly this.)
- When an attempt finds nothing left in Processing, it looks in the video's filed folder. If
  that folder holds the video and a `<video>_features.json` at least as new as the row's
  `processing_started_at`, those are this run's results, filed by the interrupted attempt: the
  row is recorded `archived` (with the usual clean-up of the posed copy and the claim). Only when
  nothing of this run is filed is the row marked `failed` for a person, as before.

**How "this run" is recognised.** Every run writes a fresh features file after the row entered
`processing`, and filing keeps a file's modified time. An earlier generation filed before this run
started is older than the stamp and does not count. Anything unreadable counts as "not filed".

**Where it lives.** `watcher/orchestrator.py`: `filed_this_run`, `_archive_to_nas`,
`_archive_locally_processed`; `watcher/db.py` `VIDEO_TRANSITIONS['archiving']`; tests in
`tests/test_filing_state_recorded.py`.


## Update 2026-11-15: a quality triage that cannot run holds the video

**Before.** The automatic quality check (the unified triage, `pipeline/triage.py`) gives the review
gate its quality verdict. When the check itself raised an error, the verdict stayed "auto
approved": the video passed the gate unchecked, went on to kinematics and was filed.

**Now (both machine roles).** When the check raises, the error is logged and recorded as a failed
`triage` step, and the gate is given the verdict `triage_failed`. The gate holds the video in the
**deep-review queue** with the routed reason `triage_failed: the automatic quality check could not
run on this video -- a person checks it`. What a person does: open it in the re-segmentation (deep
review) tools, check the video, and clear it as for any deep-review hold. Once a person has cleared
it, the gate honours that clear on later runs (the same rule as a failed quality check), so a check
that crashes the same way every time cannot bounce the video back. The return scan does not release
such a hold on its own (it only releases holds the quality check made and would no longer make).
Easy and Flat trays are never held, as before.

**No old verdict beside new outputs.** Any `<video>_triage.json` from an earlier run is removed
before the check runs, so a crash cannot leave an earlier verdict sitting with the new outputs.

**Where it lives.** `watcher/orchestrator.py` `BaseOrchestrator._run_quality_triage` (one step for
both roles); `watcher/review_gate.py` `QC_TRIAGE_FAILED`, `evaluate_gate`; tests in
`tests/test_triage_fails_closed.py`.

## Update 2026-11-15: Escalate and Flag Session record the move in the watcher database

The review tool's **Escalate: bad segmentation** and **Flag Session** buttons move a video from the
triage queue into deep review. They used to move it with no database, so the row still said
`triage` while the bundle was in deep review. They now open the watcher's own database
(`db_location.open_watcher_db_if_present`: the configured `watcher.db_path`, else
`<processing_root>/watcher.db`; never created) and record `deep_review` as part of the move
(`review_gate.escalate_to_deep_review(..., db=...)`, `route_to_queue(..., db_state="deep_review")`).
If the database cannot be opened, the move still happens, the tool says the record was not
updated, and the row catches up when the video returns. Tests:
`tests/test_escalate_records_deep_review.py`.


## Update 2026-11-15: a crystallized video is never moved

`mousereach-crystallize` locks filed videos for a publication (watcher state `crystallized`). Their
filed results must stay exactly as published. Before, several paths forced their way through the
lock: `mousereach-route-to-queue` (its "in flight" check did not include the lock), the gate's
`route_to_queue` (it forced the queue state when the normal transition was refused), the review
return (it forced `processing` on any row) and the version scan's mislabel divert.

**Now every one of them refuses before anything moves:**
- `route_to_queue` raises `CrystallizedError` when the watcher database says the video is
  crystallized; nothing is moved or written.
- `mousereach-route-to-queue` reports such a video as `lock` (`"locked": true` with `--json`),
  with a message naming the lock. It is not counted as a failure in the exit code.
- The review return and the release of a lifted quality hold leave the bundle where it is and log
  the refusal (the return is tried again hourly, so it shows in the log until a person acts).
- The version scan's mislabel divert logs the refusal and leaves the filed results.
- `mousereach-watch-process-animal` skips a crystallized video ("already crystallized").
- `mark_failed` and `mark_unresolvable` refuse a crystallized row too: they raise
  `CrystallizedError` and leave the row (state, error message, error count) unchanged. So an
  error in any handler cannot turn a locked video into a failed one that the retry work would
  pick up, or into an unresolvable one that the pipeline re-drives when a file turns up.
- `force_state` itself will not take a row out of `crystallized`; only the unlock command
  (`mousereach-uncrystallize`) may, and it says so explicitly (`unlock_crystallized=True`). A
  dashboard "Re-run" on a locked video shows "Could not mark it: ... is crystallized".
- `mousereach-repose-videos` already refused anything not archived or held.

**What a person does** to put a locked video through review again: unlock it with
`mousereach-uncrystallize`, then route it. **Limit:** the lock lives in the watcher database of the
machine where `mousereach-crystallize` ran; a command run against another machine's database does
not see it.

**Where it lives.** `watcher/db.py` (`CrystallizedError`, `crystallized_refusal`, `force_state`),
`watcher/review_gate.py` `route_to_queue`, `watcher/route_cli.py`, `watcher/review_return.py`,
`watcher/reprocessor.py`, `watcher/cli.py` (process-animal); tests in
`tests/test_crystallized_never_moved.py`.

## Update 2026-11-15: a GPU node poses only a file it holds; process-animal claims what it copies

**Pose step.** A GPU node finds the video for a `dlc_queued` row by name, and that search also
looks at the top of the shared singles folder (`Unanalyzed/Single_Animal`). DeepLabCut writes its
pose beside its input, so a row with no copy in the node's own DLC queue (after cross-node
recovery, or a hand retry) was posed in place in the shared folder, unclaimed: another node could
claim and pose the same video at the same time. The pose step (`_process_single_dlc`) now poses
only a copy in this node's DLC queue or in its own claim folder
(`Single_Animal/.inflight/<this node>/`). Otherwise nothing is posed and the row becomes
`unresolvable` with a reason saying where the copy was:
- an unclaimed copy at the top of the shared folder: the reason starts "in the shared singles
  folder for any node to take", which the singles intake re-drives -- the next scan claims the
  file properly (moves it into the node's claim folder) and copies it into the DLC queue;
- a copy in another node's claim folder, or anywhere else: refused with the reason; a person
  looks (`mousereach-watch-unresolvable --list`).

**`mousereach-watch-process-animal`.** It used to copy singles from the shared folder into the DLC
queue without claiming them, so two nodes could pose one video. It now claims each single for this
machine first (the same claim the watcher's own intake takes) and copies from the claimed file. A
single it cannot claim -- another machine took it meanwhile, or a program has it open -- is skipped
("could not claim it in the singles folder"). A copy that fails gives the single back to the
shared folder. The watcher on the same machine keeps the claim alive and removes the claimed copy
once the video has moved on, as for its own claims.

**Where it lives.** `watcher/orchestrator.py` `DLCOrchestrator._not_held_for_pose`,
`_process_single_dlc`; `watcher/cli.py` `main_process_animal`; tests in
`tests/test_pose_only_claimed_files.py`.

## Update 2026-11-15: deep review supersedes triage

The rule: **a video in the deep-review queue is not allowed to be in the triage queue as well.**
Deep review always wins; a triage copy of such a video is emptied without losing a person's
file: what has a current copy elsewhere is deleted and the rest stays where it is, listed (see
below).

**Routing.** Every route into a review queue goes through `review_gate.route_to_queue`:
- into triage while the video has a deep-review bundle (published, or being built in
  `deep_review/.incoming/`): nothing is moved or created, the row is set to `deep_review`, the
  refusal is logged, and the deep-review bundle is returned. When the gate itself asked (a re-run
  of a video that is in deep review), that run's files are then taken out of the local Processing
  folder so the video is in one place only (`review_gate.discard_run_outputs`): the run's outputs
  are deleted (they are made again when the video leaves deep review); the video or a pose file is
  deleted only when the deep-review bundle holds the same file, and one that could be the only copy
  is set aside, never deleted, to `<processing root>/_finished_elsewhere/<date>/Processing/` (the
  folder the watcher already used for such copies); a person's file (review, ground truth, clear
  marker, hand-set cuts -- `review_routing.is_person_file`) is deleted only when the deep-review
  bundle holds it byte for byte, and otherwise stays where it is and is listed
  (`note_for_a_person`, kind `person_file_left`). A route from anywhere else (a filed folder, a
  queue) never deletes anything;
- into deep review: after the bundle is complete (its routing record lists no file that failed to
  move), the video's triage folder is emptied (`_drop_triage_copy` -> `_empty_triage_copy`): the
  video or a pose file is moved into the deep-review bundle when that bundle has no file of that
  kind (`review_routing.input_kind`); then the folder is cleared as a leftover against the
  deep-review bundle (`review_routing.clear_leftover`, see "queue leftovers" below). No review,
  ground truth, hand-set cuts or clear marker is carried into the deep-review bundle this way.
  With a failed file the triage folder is kept untouched and listed (`queue_folder_in_the_way`);
  `repair_unfinished_routes` empties it as above on every pass once the deep-review bundle's route
  is whole (its record has `finished_later` and no `failed_files` left), whoever moved the missing
  file in. Escalate and the return scan's mislabel divert are not this path: they route the
  triage bundle itself, and its review and ground truth travel with it into deep review (unchanged;
  the re-segmentation-voids-answers work will change it).

`mousereach-route-to-queue --queue triage` checks the rule before it flags any segment and
reports such a video as `deep` (`"in_deep_review": true` with `--json`); it is not counted as a
failure.

**Readers.** Wherever both queues are read, the video counts as deep review only: the census
(`census/runner.py`), the dashboard folder scan (deep review ranks above triage), reconcile, the
return scan (never returns, releases or diverts such a triage copy), the triage review tool and
the Review Queues tab, the GPU node's "already held" check, the re-run's "pose only in a review
bundle" check, the archive step's queue check, the re-pose consumer (a video in deep review is
not "held in triage") and `mousereach-repose-videos` (parks the deep-review bundle).

**Where it lives.** `watcher/review_routing.py` (`deep_review_stems`, `in_deep_review`,
`superseded_by_deep_review`, and the block comment above them), `watcher/review_gate.py`
`route_to_queue`, `_drop_triage_copy`; tests in `tests/test_deep_review_supersedes_triage.py`.

## Update 2026-11-15: a reach assignment that fails holds the video in triage

**Before.** When reach assignment (which reach caused each touched pellet) raised, the watcher
logged a warning and went on. No assignment file was written, which switched off the gate's
"touched pellet with no credited reach" hold, so the video could reach kinematics and filing with
no causal reach anywhere.

**Now (both machine roles).** A problem holds the video at the step that has it. When assignment
raises, or cannot read its inputs, the step is recorded as failed and
`<video>_reach_assignments.json` is replaced by one that names no causal reach and carries
`"assignment_failed": "<the error>"`. Every reader then sees each touched pellet without a causal
reach:
- the gate holds the video in **triage** with the reason `reach assignment could not run (<error>)
  -- choose the causal reach for segment(s) ...`;
- the triage review tool asks a person for the causal reach of exactly those pellets;
- the return scan releases the video once they are answered, and the re-run applies the answers.
A video whose pellets were all untouched needs no causal reach and is not held. An earlier run's
assignment file is replaced as well, because it describes outcomes this run recomputed. If the
placeholder file cannot be written, the video is marked failed instead (never passed through).

**Where it lives.** `watcher/orchestrator.py` `BaseOrchestrator._hold_for_failed_assignment`;
`watcher/review_gate.py` `ASSIGNMENT_FAILED_KEY`, `evaluate_gate`; tests in
`tests/test_assignment_failure_holds_video.py`.

## Update: files holding answers, routing records and stage outputs are replaced in one step

Several watcher steps rewrite a JSON file that already exists. They used to open the file for writing, which empties it first, so a crash or a lost share connection part-way through left a cut-off file. A cut-off review is read as "no answers"; a cut-off stage output fails the next step. Each of these writes now goes to a hidden temporary file beside the real one (`.<file name>.<random>.tmp`), is flushed to disk, and is swapped in with one rename (`pipeline/fsutil.py`: `write_json_atomic`, `copy_file_atomic`). Anyone reading the file sees the old complete version or the new complete version, never a mix. If Windows refuses the rename because another program has the file open, it is tried again for about ten seconds and then the step fails loudly with the old file unchanged.

The temporary name starts with a dot on purpose: the router and the archiver take every file whose name starts with the video's name, and a dot-name never matches, so a temporary file left by a crash is never carried along as one of the video's outputs. The one mover that takes EVERY file in a folder -- returning a cleared review bundle to Processing (`review_return._return_to_processing`) -- deletes such leftovers before it moves anything and logs a warning naming each; nothing would ever remove one from Processing. If a leftover cannot be deleted (held open, e.g. by a review window saving into the bundle), the return moves nothing, logs an error naming it and is retried later, so the video is never left half in the queue and half in Processing. (A bundle released to the posed folder is then cleared as a leftover, temporary files included: copies and machine results are deleted, and a person's file stays where it is with a note, listed under 'Folders left for a person' in the Review Queues tab -- see "Leftover queue folders" in docs/REVIEW_TOOLS.md. Nothing goes to `_Problematic`.) Elsewhere a leftover can be deleted by hand.

Where this applies on the watcher side:

- **Returning a cleared review bundle.** Before the bundle is emptied, its review is copied over the durable copy in `review_records/reviews/` when the bundle's is newer (`review_return._ensure_durable_review`). A copy that fails is logged and the return goes ahead; the earlier durable copy stays complete.
- **Routing a filed video back to a queue** (`mousereach-route-to-queue`). Flagging segments rewrites the video's `<video>_pellet_outcomes.json`, and `--recheck` rewrites its review file after copying it to `Archive/manual_corrections/<name>.before_recheck_<time>` (`route_cli.flag_segments`, `route_cli.reopen_answered`). Both replace the file in one step. If the write fails, the command stops with an error, the file is exactly as it was, and the command can simply be run again.
- **Stamping the quality verdict.** After the automatic quality check, its verdict (`validation_status`, `triage_reason`) is written into `<video>_segments.json`, `<video>_reaches.json` and `<video>_pellet_outcomes.json` (`orchestrator._run_quality_triage`). Each of the three is replaced in one step; a crash mid-stamp used to leave a cut-off stage output -- the video's actual results. Before stamping, any temporary file an earlier crashed stamp left beside these three is deleted. A stamp that fails is logged and the file keeps its earlier content, as before.
- **Keeping the hand scores with the video.** The sheet's scores for the session are written into `<video>_pellet_outcomes.json` under `hand_scores` (`hand_scores.record`). This already went through a temporary file, but one named `<video>_pellet_outcomes.json.tmp` -- a name that starts with the video's, so a leftover was carried along by the router and the archiver. It now uses the shared writer: dot-named temporary file, flushed, rename retried while the file is held. A failure holds the video, as before.

## Update: a filed video's re-run keeps where it starts through a retry or a restart

**Before.** The version scan marks a filed video 'outdated' with the stage to start from
(`reprocess_scope`, see the table under "a recalculation re-runs no algorithm and no triage"). That
start stage then travelled only inside the one run the watcher started for it. If that run failed
(the kinematics step itself, say), the failed-video retry an hour later picked the bare row; if the
watcher stopped mid-run, the next start picked the 'processing' row the same way. Both ran with no
start stage, so a filed video whose only change was its kinematics was re-run through all four
algorithms, the quality triage and the hand-score comparison -- and could be sent back to a person
for a question already answered.

**Now.** When the watcher runs a video without being told where to start (the retry, a restart,
the ordinary pipeline pick), it reads the row:

| The row says | The run |
|---|---|
| `reprocess_scope` = `kinematics`, the video has been filed before (`archived_at` set), and this watcher is not holding it for its hand scores | kinematics only: no algorithm, no quality triage, no hand-score check; the review gate still runs |
| anything else (no scope, `segmentation`, `reach`, `outcome`, never filed, held for hand scores) | every stage and every check, as before |

Only `kinematics` is carried over: a run from `segmentation`, `reach` or `outcome` recomputes an
algorithm and every check runs on it anyway. The log line says "filed video's 'kinematics' re-run
resumed from its row (retry or restart)".

**When the kept scope is narrowed.** The moment a kinematics-only re-run finds it has to remake an
algorithm output after all (a copied-down output was missing), the row's scope is changed to that
stage -- `segmentation`, `reach` or `outcome` (reach assignment counts as `outcome`) -- before the
stage runs. A later pick then runs every stage and every check. WHY: the fact that this run remade
an output lives only inside the run. Without the change, a watcher restart while the video waits
for its hand scores, or a failure between the remake and the checks, would pick the video up again
as kinematics-only, find the remade output on disk, reuse it, and file it with no quality triage
and no hand-score comparison -- an output no check ever judged.

**When the kept scope is wiped**, so that it can never skip the checks for a run whose inputs have
just changed:
- the video is filed (both machine roles: the row reaches 'archived');
- a review queue returns the video (a person has just answered questions about it; its next run
  resumes just after the step that held it and goes through every check -- see "a returned video
  resumes just after the step that sent it");
- the review gate finds the outputs out of step with the cuts ("stale outputs": outcomes older than
  the segments, or reaches older than a person's boundary correction). The video stays in
  'processing' and its next pick recomputes from segmentation; without the wipe it would reuse the
  same out-of-step outputs and meet the same answer on every pick;
- the row enters any state outside its re-run. The scope means something only while the row is
  'outdated', 'processing', 'failed' (waiting for its retry), 'processed' or 'archiving'; entering
  any other state -- triage, deep review, a quality hold, 'dlc_complete', 'unresolvable', or
  'archived' (including a filed video put back after its re-run failed for good) -- clears it, and
  so does a person's `mousereach-watch-reprocess` reset of a failed video to an earlier state.
  WHY: a scope used to outlive its re-run on such rows, and a later, unrelated run (a quality-hold
  release, an intake) read the leftover 'kinematics' back and skipped every algorithm, the quality
  triage and the hand-score comparison on outputs nobody had re-checked.
- the row enters its re-run's states from OUTSIDE them without being given a scope. A scope is
  carried only between 'outdated', 'processing', 'failed', 'processed' and 'archiving'. A row
  that comes into one of them from a review queue, 'unresolvable' or 'archived' starts with no
  scope, unless the step that moves it gives one (the version scan and Re-run always do). This
  holds for every way a row changes state, a video marked failed while it sits in a review queue
  included. WHY: rows written by older code still carry a scope in those outside states; without
  this, such a row marked 'failed' or 'outdated' would bring its old 'kinematics' into the very
  states where it is read back.

**Rows left with a scope by older code** need no manual repair. They cannot be read back (the last
point above), and every program that opens the watcher database clears them, the first time one
does and never after. The clearing cannot be undone from the database, and it is done by
whichever program happens to open the database first, so it is written down before it happens.
In that program's log, in this order:
1. "Clearing a leftover re-run scope on N row(s) that are not in a re-run -- per state: archived:
   A, triage: B, ..." -- how many, and how many in each state;
2. one or more lines "Leftover re-run scope being cleared: <video> (<state>) = <scope>; ..." naming
   EVERY row and the value it held (forty rows to a line);
3. then the rows named are cleared -- only those -- and
4. "Cleared a leftover re-run scope on N row(s) that are not in a re-run".
To put a value back by hand, the lines of step 2 hold everything needed; nothing reads a scope in
those states, so there is no reason to. If the database is locked at that moment, the clean-up
waits for the next open.

**If the database was locked when a program opened it.** Two columns this change added to the
watcher database (`resume_stage`, `held_at_step`) are created when a program opens the database.
A database locked at that moment refuses, and the log says "column(s) ... could not be added".
Nothing has to be done: the program adds them again at its next state change, and until it can,
its state changes go through without them (as they did before the columns existed).

**For an operator.** Nothing to do. A filed video that is being re-run for its kinematics and shows
'failed' in `mousereach-watch-status` is retried after an hour as a kinematics-only run, as the
status line "retrying after a failure" says.

**Where it lives.** `kept_rerun_start_stage` in `watcher/orchestrator.py` (the rule and its
reasons), read by `ProcessingOrchestrator._run_pipeline`; narrowed there (`scope_for_recomputed`)
before a stage is remade; the scope is wiped in
`ProcessingOrchestrator._gate_held`, at filing in `_archive_to_nas` (both roles) and in
`review_return._return_to_processing`, and on every other state change by `WatcherDB.update_state`,
`force_state` and `reset_failed` (`watcher/db.py` `REPROCESS_SCOPE_STATES`). Tests in
`tests/test_filed_rerun_skips_checks.py` and `tests/test_rerun_scope_ends_with_its_round.py`.

## Update: a stage whose version was never recorded starts the re-run at that stage

**Before.** A filed video's processing manifest can lack a stage's version ("not recorded":
`not_run`, `unknown`, empty). The version scan counted such a video as out of date, as it should,
but chose the start stage from the stages that ran at an OLD version only. A manifest whose only
gap was the kinematics version therefore gave an empty list, and an empty list means "start at
segmentation": all four algorithms, the quality triage and the hand-score comparison were run
again on a filed video, and it could be sent back to a person. A stale reach-assignment version
did the same, because assignment had no start stage in the scan's map.

**Now.** A stage that was never recorded counts like a stale one, at its own stage, and the re-run
starts at the earliest stage either list names:

| Out of date or not recorded | Re-run starts at |
|---|---|
| segmenter | segmentation |
| reach detector | reach |
| outcome detector, or reach assignment (it is recomputed with the outcomes) | outcome |
| kinematic extractor only | kinematics (no algorithm, no quality triage, no hand-score check) |

`mousereach-version-check` lists a not-recorded stage as `<stage> (not recorded)` beside the stale
ones, so the reason a video is marked is visible.

**Where it lives.** `watcher/reprocessor.py` (`_STAGE_FOR_COMPONENT`, `earliest_stale_stage`, the
mark side of `ReprocessingScanner.scan`). Tests in `tests/test_version_scan_start_stage.py`
(including one that fails if a stage tracked in `pipeline/manifest.py` `TRACKED_STAGES` has no
start stage).

## Update: a review saved for a filed video with no processing manifest reaches its kinematics

**Before.** The version scan skips a filed video that has no processing manifest (it has no
versions to compare) and counts it as `no_manifest`. It skipped it before looking for a saved
review, so a person's answers saved for such a video never reached its filed kinematics.

**Now.** For a filed video with no manifest the scan still asks whether a saved review is waiting
to be applied (the same check as for every other video; without a manifest it compares the
review's time with the filed features file, and a review with no features file at all counts as
waiting). If one is waiting, the video is marked 'outdated' for a **kinematics-only** re-run
(listed as `human_review, no manifest`), and the re-run writes a manifest, so from the next scan on
the video's versions are compared like any other. A review that says a segment was mislabelled is
sent to deep review instead, exactly as for a video with a manifest. A video with no manifest and
no waiting review is still only counted (`no_manifest` in `mousereach-version-check`).

**Only when the scan could list the archive.** If this pass's one listing of Analyzed failed (a
network hiccup), every filed video looks as if it had no manifest and no features file, and each
one with a saved review or ground truth would be marked for a re-run it does not owe. So on such a
pass a video with no manifest is only counted (`no_manifest`), never marked, and filed ground truth
is not looked at at all; the next pass that lists the archive judges it properly.

**Where it lives.** `watcher/reprocessor.py` (`ReprocessingScanner.scan`,
`_mark_kinematics_rerun`). Tests in `tests/test_version_scan_start_stage.py`.

## Update: ground truth saved after filing re-runs the video's kinematics

**Before.** The version scan re-ran a filed video's kinematics when a saved causal review was not
the one its kinematics had applied. Nothing did the same for ground truth
(`<video>_unified_ground_truth.json`): ground truth saved or changed after a video was filed
reached its features only if some other reason happened to re-run the video.

**Now.**
- When kinematics are made, the feature extractor looks for the video's ground truth **beside the
  run first** (the processing folder; a re-run copies the filed ground truth down there), and only
  then in the ground-truth index. WHY: that index is built once per watcher process, so in a watcher
  that has been running for days it does not know a file copied down for a re-run -- and the re-run
  then made the kinematics without the very ground truth it was started for.
- The processing manifest then records, by content, the ground truth the extractor **actually
  applied**: `applied_ground_truth` = `{file, last_modified_at, sha256}`; `null` when it looked and
  found none; no entry at all when the extractor could not say (its truth layering failed -- the
  older time-based rule below then decides). WHY the applied file and not "the file beside the
  run": the two differ exactly when the lookup missed, and stamping the file beside the run would
  record ground truth as applied that never reached the numbers -- the video would never be
  re-run for it.
- The version scan reads the ground-truth files in the filed folders from the same single listing
  of the archive it already makes (no extra network calls) and asks, for each, whether the
  kinematics were made with it as it is now:
  1. the manifest has the stamp: the file's content against the stamp; a `null` stamp (no ground
     truth applied) means a file filed now is owed, whatever its dates say;
  2. a manifest from before the stamp: the file's own `last_modified_at` against the manifest's
     `created_at`;
  3. only when either is missing: the file's modified time against the features file's.
- If not, the video is marked 'outdated' for a **kinematics-only** re-run, listed with
  `ground_truth` among its reasons (also for a video with no manifest). After the re-run the stamp
  matches and the video is not marked again; a later change marks it once more. A marked video
  whose ground truth is still owed is not un-marked by the scan's "versions compare current" pass.
- Recording retirement uses the same rule, so a video the scan will re-run for its ground truth does
  not count as complete for retiring its recording.

Only ground truth beside the filed video is watched; ground truth kept anywhere else does not
trigger a re-run.

**Why kinematics only, not from reach detection.** Ground truth can move a reach's frames. That is
applied by the feature extractor itself: its truth layering puts the ground truth's reaches over the
algorithm's (replacing the reach each one overlaps, adding the ones the detector missed, and -- when
the ground truth marks its reach list complete -- dropping the algorithm's others), and computes the
kinematics over the ground truth's frames (`reach_source` = `ground_truth`). Reach detection does
not read ground truth at all, so starting the re-run there would recompute the same algorithm reaches
and add nothing -- while turning a filed video's re-run into a full one with every check. Ground
truth's pellet outcomes are layered the same way. Ground-truth SEGMENT boundaries are not applied by
the kinematics step (or by any step of the pipeline): a re-cut is done in the re-segmentation tools.

**For an operator.** Nothing to do: ground truth saved beside a filed video reaches its features at
the next version scan (about every half hour). If the re-run cannot apply it, the video is offered
again on the next scan.

**Where it lives.** `pipeline/manifest.py` (`ground_truth_applied`, the one rule;
`record_kinematic_version` writes the stamp, `NOT_RESOLVED` = "could not say");
`kinematics/core/feature_extractor.py` (`extract` looks beside the run first and sets
`applied_ground_truth`); `review/truth_resolver.py` (`resolve_truth_layers(applied=...)` reports the
file it found); `watcher/orchestrator.py` `_make_features` passes what the extractor applied; `watcher/reprocessor.py` (`_index_archive`, `_pending_ground_truth`, the mark
side and the un-mark side of `scan`); `video_prep/core/collage_provenance.py`
`build_complete_stems`. Tests in `tests/test_version_scan_start_stage.py`,
`tests/test_filed_rerun_skips_checks.py` and `tests/test_ground_truth_beside_the_run_is_applied.py`
(including a ground-truth file created after the index was built, and a reach whose frames the
ground truth moves).

## Update: a filed video whose re-run fails stays filed

**Before.** A filed video being re-run (marked 'outdated' by the version scan or a person's
Re-run) that failed was marked 'failed' like any video. After its retries it stayed 'failed' for
good, although its filed results in Analyzed were untouched and still what the database pulls.
Every status view showed it as failed, and the version scan, which looks only at 'archived' rows,
never looked at it again. A re-run that could not even start (copying its files down failed, or no
processing folder configured) was marked 'failed' too, and the retry then ran the pipeline on the
partly copied files.

**Now.** A filed video whose filed results are still in Analyzed never ends in 'failed' because of
a re-run (the one exclusion, a video `mousereach-repose-videos` took apart, is below). When its
re-run has used up its retries (`watcher.max_retries`, 3), or its working files are gone so no
retry could run, or the re-run could not start on its last try (see "A re-run that cannot start"
below), the processing watcher:
1. puts the row back to **'archived'**, keeping when it was filed (`archived_at`), and wiping its
   re-run scope and a review return's resume point (the next re-run starts afresh, with the scope
   the version scan works out then);
2. writes the reason into `error_message`, starting "re-run of a filed video failed: ..." and
   ending with the folder the failed run's working files were moved to (step 4) (the
   watcher's start-up clean-up of old errors leaves this message alone; filing the video again
   clears it), and sets `error_count` back to 0 so the next re-run has its retries;
3. logs a `reprocess` step as failed, with the reason;
4. moves this video's working files out of the local processing folder into
   `<processing_root>/_finished_elsewhere/<date>/Processing/` (moved, never deleted -- the same
   place the start-up clean-up puts copies of finished videos). When that dated folder already
   holds a file of the same name -- the video was set aside earlier the same day, by a try whose
   copy failed or an earlier failed run -- this set goes into its own sub-folder,
   `.../Processing/<video>_again_<HHMMSS_microseconds>/`, and step 2's message names that
   sub-folder. WHY: the dated folder is per day; moving a second set onto the first replaced the
   first set's files of the same name. A file that would still land on a namesake is left in the
   processing folder with a warning in the log, never moved over it;
5. lists the video in `<NAS root>/attention/<host>_failed_reruns.json`, which the Pipeline
   Dashboard's red panel shows ("needs a person"). The list is rewritten from the rows at start-up,
   after every such failure and after every version scan, so a video filed again drops off at the
   next scan.

**"Its working files are gone" means all three of these:** the video is not in the processing
folder, the file the row records as the video (`current_path`) does not exist, and the file it
records as the pose (`dlc_output_path`) does not exist. WHY all three: a video a review queue
returned can run with its video or pose left where it was, outside the processing folder. Judged
by the processing folder alone, one passing failure of such a run put the video straight back to
'archived' and parked the answers the person had just given. While any of the three is still
there, the row stays 'failed' and keeps its retries.

**The one failed re-run of a filed video that stays 'failed': it carries something a person made
that has no other copy.** Before the row is put back to 'archived' the watcher looks in the
processing folder for four kinds of file:
- `<video>_segments.json` whose cuts a person set (it says `"boundary_source": "human"`);
- ground truth: any of the video's files with `ground_truth` in its name
  (`<video>_unified_ground_truth.json`);
- the deep-review "cleared" marker, `<video>_deep_review_cleared.json`;
- a review's answers, `<video>_causal_review.json`.

Each one found is compared with the same-named file in the video's filed folder; a review's
answers are also compared with the copy in the durable review store. "The same" means the same
content, not merely the same name. If ANY of them has no identical copy, the row is NOT put back
to 'archived'. It stays 'failed', listed by `mousereach-watch-status`; nothing in the processing
folder is moved; no fingerprint is written; and the row's `error_message` gains one sentence,
"left as failed, files in place: <file names> in the processing folder was made by a person and
has no other copy". A file the watcher cannot read or compare counts as having no other copy.

WHY: each of these comes to the processing folder from a review queue, and until the video is
filed again that can be the only place it exists. Ground truth and the clear marker have no second
store at all. A review's answers normally have a durable copy, but not when no durable store is
configured or when the copy failed at the return (the return only logs that and goes on). Cuts
have none. Putting the row back to 'archived' moves every file of the video into
`_finished_elsewhere` -- a folder named for finished videos, which operators clear -- and the next
re-run copies down from the filed folder only, so the person's work would silently never be
applied and could then be deleted. (The first version of this step exempted hand-set cuts only;
ground truth, the clear marker and a review with no durable copy were moved aside with everything
else.)

A file that IS identical to its copy -- ground truth copied down from the filed folder for a
re-run, a review whose durable copy the return made -- does not keep the row 'failed': moving
that duplicate aside loses nothing.

What to do with such a video: find out why the run failed (the `error_message` and the processing
log say), fix that, and reset the video with
`mousereach-watch-reprocess --from-step processing <video id>`; the run starts again on the files
in the processing folder, the person's file among them.

**A re-run that cannot start.** Before a re-run, the watcher copies the video's files down from
Analyzed into its own processing folder. If any copy fails (usually a file locked for a moment),
or no processing folder is configured, the re-run has not started, and that is not yet a re-run
that failed for good. For EVERY such video, filed or not, the watcher:
1. moves the files THIS try copied down out of the processing folder into
   `<processing_root>/_finished_elsewhere/<date>/Processing/` (moved, never deleted; a second
   set the same day goes into its own `<video>_again_<time>/` sub-folder, as in step 4 above,
   never over the first). WHY: a
   partly copied set left there used to be picked up by a retry and run on files that had not all
   arrived. A file that was ALREADY in the processing folder before the try is never moved. WHY:
   for a video that was never filed (a first pass that failed, then Re-run pressed) the processing
   folder holds the only copies of its video, its pose, a person's cuts and its review file, and
   the row points at them; moved aside, the next try found "no pose and no video" for files that
   were sitting in a folder named for finished videos. A copied-down file still has its original
   where it came from, so moving the copy loses nothing;
2. leaves the row **'outdated'** (it still owes its re-run) with `error_message` "re-run could not
   start (attempt N): <reason>", ending with the folder the copies were moved to, and
   `last_error_at` now. `error_count` is not touched: that is the pipeline's own retry count;
3. logs a `reprocess` step as failed, with the reason;
4. tries again after an hour (the same wait a failed video has). A try that copies everything
   starts the re-run and clears the message.
After `watcher.max_retries` (3) tries in a row that could not start, the re-run counts as failed
for good: a filed video goes back to 'archived' as described above (fingerprinted and in the red
panel) -- unless its processing folder holds something a person made that has no other copy (the
rule above), in which case it is marked 'failed' and that file stays; any other video is marked
'failed' too, with the files it had before the re-run still in the processing folder. (Still open, and the same as before this change: when such a never-filed video
already had its own older copy of the file that could not be copied, the failed-video retry can
run on that older copy.) WHY not straight back to 'archived'
on the first failed copy: an 'archived' row with the fingerprint is not offered again until the
declared versions change or a review or ground truth is saved, so one passing lock would have
parked an owed re-run indefinitely.

**A file already in the processing folder is not copied, and never deleted.** A video that failed
its first pass and then had Re-run pressed still records its pose in the local processing folder,
so its copy-down "copied" every file of the video onto itself. The copy routine
(`transfer.safe_copy`) treated the resulting same-file error as a failed copy and deleted the
destination -- which was the only copy: the video, the pose, a person's cuts and the review files
were all lost, and the re-run then could not start. Now a source that is already the destination
counts as copied (and, for `safe_move`, as moved): nothing is written and nothing is deleted.

**When the re-run is offered again.** The row also keeps a fingerprint of what the re-run was
attempted against (`pipeline_versions_hash`): the declared versions (`pipeline_versions.json`,
including `compatible_versions`), the saved review the scan would apply and the ground truth filed
beside the video. On every pass the version scan works the fingerprint out again for such a row:
- the same: the video is **not** marked again. It is counted under `failed_rerun_waiting` (named in
  `failed_rerun_waiting_videos`). WHY: the filed manifest is still out of date, so without this the
  scan would mark the video again within half an hour, the re-run would fail the same way through
  all its retries, and so on for ever -- each round a wasted run, and the video dropping off and
  reappearing in the list a person reads;
- different (the declaration changed, or a review or ground truth was saved since the failure): the
  scan judges the video as for any other filed video and marks it if it is still out of date;
- if the change makes the filed results count as current (for example a version declared
  compatible), the failure message is cleared and the video leaves the list.

A person can press **Re-run** for the video at any time; that marks it at once, whatever the
fingerprint says.

**How the watcher decides "filed".** Two things must both be true: the row has been filed
(`archived_at` set), AND the video's `<video>_features.json` and `<video>_processing_manifest.json`
are in its filed folder (`Analyzed/<project>/<cohort>/`) right now. It does not matter which run
failed: the version scan's re-run, a person's Re-run, a video a review queue returned after a
person answered its questions, or a run the gate sent back because its outputs were out of step
with the cuts (which wipes the re-run scope). WHY not the re-run scope or `archived_at` alone: a
review return and a stale-outputs gate both wipe the scope, and `archived_at` is never wiped, so it
means "was filed once", not "its filed results are there now".

**A video that does not pass that test is marked 'failed'**, exactly as before, listed by
`mousereach-watch-status` for a person; its working files stay where they are, except that after
a re-run that could not start the files that try copied down have been moved aside (above):
- a video that was never filed -- a first pass that failed and had Re-run pressed, or an
  'unresolvable' row sent round again;
- **a video `mousereach-repose-videos` took apart**: that command moves the filed pose and every
  output made from it into `Analyzed/Archive/<label>/` (and a held video's bundle into
  `Repose_Queue/_parked/`) before the re-run, so nothing filed stands for it until the re-run
  files it again.
WHY: 'archived' means filed, and filed means passed every check; a video whose checked results are
not in Analyzed must never read as filed.

**A returned video's answers.** When the failed run was a review return, the person's answers are
not lost. Normally the return copied the review to the durable review store before anything
moved; then the file in the processing folder is a duplicate, it is set aside with the rest, the
durable copy stays where it is, and the fingerprint above includes that review, so changing it
offers the re-run again; until then the video is in the red panel. When the durable copy is NOT
there or is not the same -- no durable store configured, or the copy failed -- the review in the
processing folder is the only one, and the video stays 'failed' with it in place (the rule
above). The same holds for ground truth and a deep-review clear marker the return brought, which
have no durable store: they keep the video 'failed' unless the filed folder holds the same file.

**For an operator.** Seeing "re-run of a filed video failed" in the red panel: the video's data in
the database is the filed, older version and is complete (a video whose filed results were moved
away is never in this list -- it shows as failed in `mousereach-watch-status` instead). Read the reason; if it names a file that
could not be copied or a missing setting, fix that and press Re-run for the video (or wait for the
next version change). A video only waiting after a copy that failed is NOT in the red panel: it
stays 'outdated', its reason starting "re-run could not start (attempt N)", and is tried again
each hour by itself. Nothing is lost: the reason on the row (and in the red panel) ends with the
exact folder under `_finished_elsewhere` that the failed run's files, or the copies made for a try
that could not start, were moved to. Only copies of files that are also somewhere else (in
Analyzed, or a review's answers in the durable review store) and outputs of the failed run are
ever moved there by this step. A never-filed video's own files stay in the processing folder, and
so does everything of a video that carries a person's cuts, ground truth, deep-review clear
marker or review answers with no identical copy elsewhere -- that video shows as failed in
`mousereach-watch-status`, its reason naming the file, not in the red panel.

**Where it lives.** `watcher/orchestrator.py` `ProcessingOrchestrator._restore_filed_after_failed_rerun`
(with `_is_filed_rerun`, `_filed_results_present`, `_filed_rerun_gave_up`, `_working_files_findable`,
`_unfiled_person_files` (with `_unfiled_cuts_of_a_person`, `_same_content`,
`_note_person_files_kept`), `_set_aside_working_copies`, `_set_aside_destination`, `_set_aside_folder`,
`_write_failed_rerun_attention`), called from the retry bucket of `_select_work_item` and from
`_rerun_could_not_start` (with `_rerun_start_wait_over`, which the outdated pick in
`_select_work_item` uses), itself called from `_reprocess_video`; `watcher/db.py`
`FAILED_RERUN_PREFIX`, `RERUN_NOT_STARTED_PREFIX`, `clear_errors_on_archived`;
`watcher/attention.py` `write_failed_reruns`. Tests in `tests/test_failed_filed_rerun_stays_filed.py`.

## Update: a returned video resumes just after the step that sent it

**Before.** A video a review queue returned (triage answered, or deep review cleared) went back to
'processing' with no record of why it had been held, and its run started at segmentation: all four
algorithms were run again on a video a person had just cleared, then every check. That was extra
work and extra ways to fail a cleared video.

**Now.**
1. **Every hold records which step sent the video.** When a video is moved into a review queue
   (`review_gate.route_to_queue`, used by the gate, a failed segmentation, the hand-score check,
   the Escalate and Flag Session buttons and `mousereach-route-to-queue`), the watcher row gets
   `held_at_step` and the bundle's `<video>_manifest.json` gets `provenance.held_at_step`:

   | Held by | `held_at_step` |
   |---|---|
   | anything that sends the video to deep review (re-segmentation) | `segmentation` |
   | the gate's triage: a pellet outcome or causal reach to answer | `outcome` |
   | the gate's triage because reach assignment could not run | `assignment` |

   A new hold also wipes any resume point left from an earlier round.
2. **The return writes where the run resumes** (`resume_stage` on the row), just after that step:

   | `held_at_step` | Resumes at | Remade | Kept |
   |---|---|---|---|
   | `segmentation` (every deep-review return) | `reach` | reaches, outcomes, assignment | the cuts (a person's cuts are kept as they are) |
   | `outcome` | `assignment` | reach assignment | cuts, reaches, outcomes |
   | `assignment` | `checks` | nothing | all four algorithm outputs |

   Deep review always resumes after segmentation, because everything computed from the cuts is
   remade on the new ones. A bundle routed before this record existed resumes after the step its
   queue stands for (deep review: segmentation; triage: outcome).

   **The person's answers on the SAME cuts persist, as designed.** The review file stays with the
   video. The gate counts a segment it answers as answered, and kinematics applies each answer to
   the segment whose frames it covers, on the reaches, outcomes or assignment the run remade
   (an out-of-date output, point 4).

   **Known defect: answers are still carried across a re-cut.** The design rule is that
   re-segmentation voids everything after it, a person's earlier outcome and causal-reach answers
   included: they are kept as a record only and asked again, never carried by frames or by segment
   number. Two paths still break that rule: a deep-review return (the cuts were redone, and the
   old answers are applied by frames to the new segments; one that matches no segment is kept in
   the file and applied to nothing), and a triage return whose algorithm cuts are out of date or
   unreadable (point 4 moves its start back to segmentation, the segmenter cuts again, and the old
   answers are applied the same way). This is not new -- before this change every return re-cut
   the video -- and it is scheduled for removal by the review-answer-records work (saving cuts
   voids later answers and re-runs from reaches). Until then, re-check already-answered pellets
   after a re-cut instead of trusting the carried answers.
3. **Every later step runs.** The quality triage, the hand-score comparison and the review gate
   always run on a returned video -- they come after every algorithm step -- and then kinematics.
   A returned video is never treated as a filed video's "downstream calculation only" re-run,
   even if it had been filed before.
4. **An earlier output is reused only when it is current.** Before reusing the outputs of the steps
   before the resume point, the watcher checks each one's recorded algorithm version against
   `pipeline_versions.json` (its `compatible_versions` count as current), and the pose the outputs
   were made from (the processing manifest that travels with the bundle) against the pose file
   this run uses and the declared pose model. The run starts at the earliest step whose output is
   missing, out of date or unreadable; a different pose means starting at segmentation. If the
   declaration or the processing manifest cannot be read, nothing is reused (start at
   segmentation). A person's cuts are exempt from the segmenter's version, as in the version scan.
   The reach-assignment file of a video held because reach assignment could not run is the
   placeholder the hold wrote (`assignment_failed`, no version); when that video resumes at
   `checks` the placeholder is kept, because the person's causal-reach answers replace it --
   re-running reach assignment would redo the step that sent the video.
5. **A retry or a restart resumes at the same point, and nothing later does**: `resume_stage`
   (with `held_at_step`) stays on the row only while the returned video's own round lasts --
   'processing', a 'failed' run waiting for its retry, and 'processed' -> 'archiving' while it is
   filed (`watcher/db.py` `RESUME_POINT_STATES`). It is wiped when the video is filed (both
   roles), by a new hold (which writes its own `held_at_step`), when the gate finds stale outputs
   (then the next pick recomputes from segmentation), when a version re-run of a filed video
   (the version scan, Re-run) starts, and whenever the row enters any other state:
   a person's reset of a failed video (`mousereach-watch-reprocess`, `reset_failed`) to an earlier
   state, a send for a new pose, 'unresolvable', 'outdated', or a forced state. Such a row's next
   run is a fresh one with every step and check. (`held_at_step` is also kept while the video sits
   in a review queue, 'triage' or 'deep_review': it is that hold's record.) WHY: a leftover resume point used to make a reset
   or re-posed video reuse the earlier round's segmentation, reach and outcome outputs, and a pose
   redone with the same model carries the same scorer name, so the pose test of point 4 could not
   tell those outputs came from the pose that was being replaced.

The processing log shows each resumed run as a `pipeline` step `resumed`, for example "returned from
review: resumes at 'assignment'; every later step and check runs", or, when an output was out of
date, "resumes at 'outcome' instead of 'assignment' -- outcome_detector output is version 3.9,
declared 4.0".

**For an operator.** Nothing to do. A returned video takes less time than before, because the steps
before the one that held it are not redone unless something changed while it waited.

**A video waiting for its hand scores is not a returned video.** One held where it is for its
session's hand scores (it stays in 'processing') is picked again only when the scores file changes,
to be compared again. That pick reuses every output it has -- reach assignment included -- and
runs only the checks; an output that is missing is remade, with every step after it.

**Where it lives.** `watcher/resume_point.py` (the rule, `resume_stage_for`, and the currency test,
`current_start`); `review_gate.route_to_queue` and `run_gate` record the step;
`review_return._return_to_processing` (`_resume_stage`) writes the resume point;
`ProcessingOrchestrator._run_pipeline` / `_resume_point` read it; `watcher/db.py` adds the two
columns and ends them with their round (`RESUME_POINT_STATES`). Tests in `tests/test_review_return_resumes_after_the_step.py`.

## Update 2026-11-16: the cropper loads on a machine that is not set up yet

**Before.** The hand-run cropping code (`video_prep/core/cropper.py`, behind `mousereach-crop`, `mousereach-prep` and the Video Prep panel) worked out its default archive folder, `<nas_root>/Analyzed/Multi-Animal`, the moment it was loaded. On a machine with no `nas_root` that sum fails, so just loading the cropper -- and with it the Video Prep panel and every piece of code that imports it -- stopped with `TypeError: unsupported operand type(s) for /: 'NoneType' and 'str'`, a message that says nothing about setup.

**Now.** The default folders (collages to cut, cut singles, the pose queue, the collage archive) are looked up only when a command actually needs one (`cropper.default_dir`). On a set-up machine they are exactly the same folders as before. On a machine that is not set up, everything loads; giving the folders explicitly (`-i`, `-o`) works; and leaving one out stops with "MouseReach is not configured ... Run: mousereach-setup". Tests: `tests/test_cropper_imports_without_config.py`.
## Update: a damaged config.json is refused, never read as "not set up"

**What was wrong.** When `~/.mousereach/config.json` held a JSON error (a missing comma after a
hand edit, say), every MouseReach program read it as an EMPTY file, without a word. Every path
was then unset, the tools said "MouseReach is not configured -- run mousereach-setup", the lab's
shared settings file could not be found (it is found from `nas_root`), so the hand-score check
was silently skipped, and a save from `mousereach-setup` or the Setup panel rewrote the file with
only the few settings setup manages -- every other setting in it lost.

**What happens now** (`config.read_settings_file`, `config._load_config`):

- A missing file, or an empty one, still means "nothing set up yet".
- A file that exists but cannot be read, is not valid JSON, or does not hold a `{ ... }` object
  stops the program with a message naming the file, the line and column of the error, and what
  to do (fix that spot in a text editor, or restore the file from a backup). A byte-order mark
  at the start (Notepad writes one) is accepted. A file saved in another encoding than UTF-8 (an
  accented letter saved as "ANSI") is refused the same way, naming the byte that cannot be read
  and saying to save the file again as UTF-8.
- A program that already read the file cleanly -- a running watcher whose file is being edited
  -- keeps using the settings as it last read them, and logs the problem once per change of the
  file, so a half-saved edit does not stop it.
- `mousereach-setup` prints `[FAIL]` with the same message and exits with code 1. The app
  (`mousereach`) prints `[FAIL]` with the message and opens only the Setup screen, which shows it
  in red; the Setup panel of an app that was already open shows it in red the same way. Neither
  writes anything over the damaged file. (Importing MouseReach no longer reads the file by
  itself -- `from mousereach import Paths` is looked up on first use -- so these two commands get
  the chance to say what is wrong; `mousereach.settings_file` holds the reader and the error.)
- Every other command (the watcher, the review tools, the batch commands) stops with a Python
  traceback whose LAST line is the same message: the file, the line and column, and what to do.

**For an operator.** If a MouseReach command stops with "The settings file ... is damaged: ...
at line N, column M", open that file in a text editor, go to that line, and fix it (most often a
missing comma or quote at the end of the line before). Then run the command again.

Tests: `tests/test_damaged_config_file_is_refused.py`.

## Update: an unreadable lab settings file holds the hand-score check; it never switches it off

**What was wrong.** The hand-score settings (`hand_scores`: which scores file, which projects
need it) usually live only in the lab settings file on the storage drive,
`<nas_root>/mousereach_lab_config.json`. That file was read leniently: a file that did not parse
(one bad hand edit), or a storage drive that could not be reached for a moment, read exactly like
"no such setting". The check before triage then answered "no hand-scores file configured" and
every analysed video went on to the gate, kinematics and filing WITHOUT being compared with its
hand scores -- and filed means done. The filing rules (`filing`) are read from the same file, and
the same lenient read made every project fall back to a folder under its own code, which no lookup
that asks the rules ever searches.

**What happens now** (`config.load_lab_config(strict=True)`, `config.lab_setting(..., strict=True)`):

- The two settings that guard a check read the lab file strictly. "Not set" now means only: the
  shared pipeline folder can be seen and there is no lab file in it, or the file has no such
  section. A lab file that exists but cannot be read or parsed, or a shared pipeline folder that
  cannot be reached, is an error, never "not set". So is a lab file that exists but is EMPTY: the
  lab file is only ever written whole, by a person who put settings in it, so an empty one is a
  file caught in the middle of being saved by an editor, or blanked by accident -- read as "no
  settings", that moment switched the hand-score check off for the videos checked in it. (A
  person's own `config.json` that is empty still just means "not set up yet".)
- **Hand scores** (`hand_scores.settings`): that error makes the check one that could not run.
  The video is HELD where it is (in 'processing', in the local Processing folder, before triage,
  kinematics and filing), and listed in RED in `<nas_root>/attention/<host>_hand_scores.json`
  (`hand_score_check_failed`, `needs_person`), with the message naming the file and what is
  wrong. Nothing is recorded in its outcomes file as if it had been checked. Once the file is
  fixed, the watcher sees the change at its next cycle, checks the video again, and it goes on by
  itself. The attention list is written whenever a video is held, also while the settings cannot
  be read.
- **Filing rules** (`config.filing_rules`): the rules the process last read cleanly are kept, with
  a warning once per run; when it has none yet, the archive step stops with the message (as it
  does for a malformed rule) instead of filing into a fallback folder. The failure is logged once
  and remembered for a minute, like good rules: the file is not read again for every video in
  every cycle (on an unreachable share each read can hang for seconds).
- Every other lab-wide setting still reads leniently (a warning once per run; the setting is
  unset).

**For an operator.** If the Pipeline Dashboard or the attention list shows analysed videos held
in red because "the check against its hand scores could not run" and the reason names
`mousereach_lab_config.json`, open that file in a text editor and fix the spot the message names
(or reconnect the storage drive if it says the folder cannot be reached; if it says the file is
empty, restore it from a backup or write its settings back in). The held videos go on by
themselves; nothing else needs doing. A bad edit of that file therefore holds every analysed Pillar
video, of any project, on every processing machine until it is fixed -- which projects need the
comparison is itself in that file, so none can be let through. That is intended: none of them can
be compared until then. Once it is fixed, the held videos go on by themselves also in a lab that
has no `hand_scores` section at all (the watcher tells "settings unreadable" apart from "no
scores file set", `hand_scores.scores_pair_stamp`).

Tests: `tests/test_hand_scores_held_when_lab_file_unreadable.py`, `tests/test_filing_rules.py`.

## Update: the hand-scores file and its manifest are read as a pair

**What was wrong.** The hand scores come as two files written by another tool: the scores CSV and,
beside it, a JSON manifest (`last_successful_import` per cohort, `sheets_seen`, `full_import_at`).
The writer replaces the CSV first and the manifest second -- two renames. The watcher cached the
pair keyed on the CSV alone, and checked a held video again only when the CSV changed. A read that
fell between the two renames therefore kept the OLD manifest beside the NEW scores until the next
import (normally a day later), and a video waiting for its cohort's import -- which is recorded in
the manifest -- was not looked at again when the manifest arrived.

**What happens now** (`hand_scores.load`, `hand_scores.scores_pair_stamp`):

- The cache is keyed on both files' size and modification time: a change to either is read.
- When the manifest has `scores_sha256` (the SHA-256 of the CSV it was written with) and the CSV
  read does not match it, both are read again up to 3 times, a second apart. If they still do not
  match, `ScoresPairMismatch` is raised and nothing is cached: the check could not run, the video
  is held and listed in red ("does not match the manifest beside it"), and it is checked again when
  either file changes. A manifest without `scores_sha256` (written before the key existed) is
  accepted as it is.
- A held video is picked again when the CSV OR the manifest changes (the stamp kept with each held
  video is both modification times). The `source_mtime` kept in a video's outcomes file is still the
  CSV's modification time.

**For an operator.** Nothing to do. If a red item says the hand-scores file "does not match the
manifest beside it" for more than a few minutes, run the sheet import again (it rewrites both
files); the held videos then go on by themselves.

Tests: `tests/test_hand_scores_file_pair.py`.

## Update: a video whose hand scores cannot be found is held as a defect, never filed uncompared

**What was wrong.** In a project listed in `hand_scores.required_projects`, a video with no hand
scores was held until its cohort's sheet had been imported after the video was ready. If that
import still had no scores for the session, the watcher took it as "never scored by hand" and let
the video go on to the gate, kinematics and filing WITHOUT the comparison (its outcomes file kept
the verdict `never_scored`). But in such a project every session is scored by hand at the bench
before its videos enter the pipeline, so no scores after an import means the sheet's row is
absent or misnamed, or the import did not match it to the video -- which looked exactly the same
from the watcher. Such a video was filed uncompared, and nothing ever compared it later.

**What happens now** (`_hand_score_precheck`, `hand_scores.write_attention`):

- Until the cohort's sheet has been imported after the video entered 'processing', the video
  WAITS: held where it is, listed grey (`hand_scores_waiting`), one `hand_score_check` /
  `waiting` row in the processing log -- as before.
- After that import, still no scores: the scores are MISSING. The video STAYS held (in
  'processing', files in the local Processing folder, before triage, kinematics and filing). It is
  listed in red (`hand_scores_missing`, `needs_person`) with what to check, the processing log gets
  one `hand_score_check` / `missing` row, and the watcher log one warning. Nothing is written into
  its outcomes file. It is checked again whenever the scores file or its manifest changes, and goes
  on by itself -- compared like any other video -- once an import brings its scores.
- A cohort with no import at all and no sheet (or a name no sheet has): after the next import of
  every sheet, its videos are held as missing their scores the same way, but their red line, log
  row and warning say that NO TRACKING SHEET was found for the cohort (the attention item carries
  `"no_sheet": true`; `hand_scores.no_sheet_for_cohort`). WHY a separate message: "the <cohort>
  sheet was imported ... fix its row" would be false here and send the person looking for a sheet
  that does not exist, so the hold would never clear.
- The census counts these videos with the other hand-score holds.
- Projects not in `required_projects`, Easy and Flat trays, and a machine with no scores file
  configured are not checked, as before.

**For an operator.** A red line "... the <cohort> sheet was imported after the video was ready and
has no hand scores for this session" means: open that cohort's tracking sheet and find the
session (its date, the animal, the tray). If the row is missing, enter the scores; if it is there,
correct the row so its date, animal and tray match the video's exactly -- the sheet decides. Do
not rename the held video: it is being worked in the watcher's own Processing folder, nothing
renames a video there, and renamed files would belong to no video, so the hold would never clear.
After the next sheet import the video is compared and carries on; nothing else needs doing.

A red line "... no tracking sheet was found for cohort <cohort> ..." means the sheets folder has
no tracking sheet for that cohort at all. Add the cohort's sheet to the sheets folder, or, if a
sheet is there under a misspelled cohort, correct the cohort in its file name. If instead the
video's name carries the wrong cohort, do not rename the held files (same reason as above): tell
whoever maintains the pipeline. After the next sheet import the video is compared and carries on.

**Videos filed before this change** with the kept verdict `never_scored` were filed without the
comparison. They can be listed from their outcomes files (key `hand_scores`, `verdict`
`never_scored`) without running anything; the watcher does not compare them again by itself.

Tests: `tests/test_missing_hand_scores_hold.py`, `tests/test_hand_score_check_before_triage.py`.

## Update: a hand-score hold survives a watcher restart

**What was wrong.** The list of videos held for their hand scores (waiting for the sheet's import,
or missing their scores) lived only in the processing watcher's memory. At every start it was
emptied, so each held video -- still in 'processing', its files in place -- was picked again and
run through segmentation, reach detection, outcome detection and reach assignment in full, only to
be held again. A night of held videos cost hours of analysis after every restart.

**What happens now** (`ProcessingOrchestrator._restore_score_waits`, `_restored_hold_start`,
`resume_point.restored_hold_start`):

- At start the watcher reads its own attention list, `<NAS root>/attention/<host>_hand_scores.json`
  (it rewrites that file whenever a hold changes, so it is the record of what was held). Each
  item that is a wait (`hand_scores_waiting`) or missing scores (`hand_scores_missing`) is
  restored, but only when the video's row is still 'processing' with the same
  `processing_started_at` the hold was measured from. Anything else -- the video moved on, was
  reset, or its row was re-stamped -- is not restored and runs as a fresh pick. A check that could
  not run (`hand_score_check_failed`) is not restored: it is simply checked again in full.
- A restored video still does not count against `max_local_pending`, keeps its place in the list
  a person sees, and is picked once to be checked again.
- A restored video that had come back from a review queue (its row still carries `resume_stage`)
  resumes exactly as any review return does (`_resume_point`): the cuts and answers the person
  worked on are kept while they pass the declared-version test, and the steps after the one that
  sent it run again. The rule below is for every other restored hold.
- That run reuses an algorithm output only while it is CURRENT: it exists, was made from the pose
  this run uses and the pose model declared now, carries the declared version (or a compatible
  one, `pipeline_versions.json`), AND carries the version the running code makes
  (`code_stamped_versions`) or one declared compatible. WHY the running code too: a restart is how new code is deployed, and
  the declaration can lag behind it; reused unguarded, outputs of the old algorithms would go on
  to kinematics and filing. The run starts at the first output that is not current and runs every
  step after it; a person's cuts are exempt from the segmenter's version, as everywhere else. The
  processing log shows it as a `pipeline` / `resumed` row, for example "held for hand scores before
  a restart: its analysis is reused; the checks run again" or "... runs again from 'reach' --
  reach_detector output is version X, the running code makes Y".
- When the attention list cannot be read, nothing is restored and held videos are analysed again
  in full, as before (a warning in the watcher log says so).

**Not changed.** A graphics-card node that also analyses (`also_process`) still starts with an
empty list; its held videos run in full once after a restart, as before.

**For an operator.** Nothing to do. Restarting the processing watcher no longer re-analyses every
held video.

Tests: `tests/test_hand_score_hold_survives_restart.py`.

## Update: routing into and out of the review queues is all-or-nothing and never deletes a person's file

Every route into a queue goes through `review_gate.route_to_queue` (which calls
`review_routing.move_video_bundle`), and every return out of one through
`review_return._return_to_processing`. The rules below are what those functions do; the
operator-level description is in `docs/REVIEW_TOOLS.md`, "Moving a video into or out of a
queue never deletes or overwrites a person's file".

- **Routing onto itself.** When the source folder IS the destination bundle (a video already in
  the queue it is sent to), `route_to_queue` moves nothing, rewrites neither the routing record
  nor the review manifest, and only records the queue's state in the watcher database.
  `move_video_bundle` returns at once in that case too, and `_safe_move` does nothing when source
  and destination are the same file. Before, each file was "moved" onto itself and the overwrite
  step deleted it. Tests: `tests/test_route_onto_itself_moves_nothing.py`.
- **Flag Session in the re-segmentation queue.** `CausalReviewWidget._flag_session` checks the
  folder the bundle is in (not the window's mode); when that is the deep-review queue it records
  the flag, sets aside any earlier clear and does not route. Test:
  `test_flag_session_in_the_resegmentation_queue_keeps_every_file`.
- **No merge into an existing folder.** When `<queue>/<video>` exists and the source is any
  other folder, `route_to_queue` calls `review_gate._clear_the_way` before anything moves: an
  empty folder is removed; the video's live bundle (`review_gate.queue_folder_is_live`: live when
  the row is held in THAT queue -- 'triage' for the triage queue, 'deep_review' for the
  deep-review queue -- or there is no database, or the folder holds a queue record or a person's
  file and the route is not a pipeline run whose `processing_started_at` is later than the
  folder's `routed_at` and newest file time, `folder_last_touched`) raises `AlreadyInQueueError`
  and is listed (`note_folder_in_the_way`, kind `queue_folder_in_the_way`); any other folder
  (records only, or not the live bundle, or a folder already holding the leftover note) is cleared
  as a leftover under the return claim, with the files the route brings as the live copy
  (`review_routing.clear_leftover`) -- when it is gone the route goes on, when something was kept
  the route raises `AlreadyInQueueError` (listed by the leftover's own item). The route command
  reports a held video as `held` before touching anything (`route_cli._held_in_a_queue`) and
  writes any requested flags into the held triage bundle's outcomes under the return claim
  (`route_cli._ask_in_held_bundle`; never into a folder holding the leftover note); a question it
  cannot record is `unrecorded` and fails the exit code. For a video it does route, it decides
  what to ask by reading only, takes the return claim, routes (all or nothing), and only then
  writes the flags into the BUNDLE's outcomes (`route_cli._record_question`), so a failed or
  refused route leaves the filed folder byte for byte as it was. The attention lists are written
  only when the review root lies inside the NAS root (`review_routing.attention_dir`). WHY the
  folder and not the row alone: rows of bundles escalated before the window recorded the move
  stayed 'triage' while the reviewer's bundle sat in deep review, and a row-only test would have
  cleared that bundle away. Tests: `tests/test_route_never_merges_into_a_held_bundle.py`,
  `tests/test_queue_folder_never_blocks_a_route.py`, `tests/test_live_bundle_judged_by_its_folder.py`.
- **Queue leftovers are deleted once a current copy exists, never set aside.**
  `review_routing.clear_leftover(folder, video, live, why, what_to_do)` is the one rule for every
  leftover queue folder: the old copy a route meets (`_clear_the_way`), a triage copy of a video in
  deep review (`_empty_triage_copy`, also reached from the return scan the first time it sees such
  a copy), a stale or superseded bundle and an empty folder (`review_return._retire_stale_bundle`,
  `live` = the deep-review twin, or `_live_copy_of`: this machine's processing folder, then the
  filed results), and a released quality hold (`_release_to_posed`, `live` = the posed folder).
  Deleted: queue records; a file `live` holds byte for byte; the review when the durable store holds
  it byte for byte (`review_held_by_store` only compares; nothing is written into the store --
  WHY: a leftover's answers were made on cuts that are being or have been redone, and the review
  readers match the store's copy to the current cuts by frame span, so a copy put there would
  carry old answers onto new cuts; and replacing a different stored review could put an older
  answer where every reader looks); machine outputs when `live` is not empty. Kept where it is: anything
  else. A folder left with something kept gets the note `LEFT_FOR_A_PERSON.json` and an item
  (`left_for_a_person`) in this machine's `<NAS root>/attention/<host>_queue_folders_for_a_person.json`;
  the return scan skips a folder holding the note (triage loop: from the listing it already takes;
  deep loop: one existence check per bundle), `repair_unfinished_routes` skips it, and
  `refresh_for_a_person` drops the item once the folder or its note is gone (it looks only at
  listed folders). The list is changed only under its lock file (`causal_review_io._locked_update`;
  WHY: the watcher, a review window and the dashboard on one machine all write it). WHY at all:
  machine leftovers went to `_Problematic`, which is only for cases the pipeline cannot process at
  all, and a person's files went there unseen or were deleted. Tests:
  `tests/test_queue_folder_never_blocks_a_route.py`, `tests/test_person_files_are_kept_apart.py`,
  `tests/test_review_return_stale.py`, `tests/test_review_return_superseded.py`,
  `tests/test_release_lifted_qc_holds.py`, `tests/test_deep_review_keeps_triage_person_files.py`.
- **A leftover is never a hold.** `review_routing.is_queue_bundle(folder)` (a folder, not holding
  `LEFT_NOTE`) is the one test every reader of the queue folders uses for "the video is held here":
  `deep_review_stems`, `in_deep_review` and `superseded_by_deep_review` (so `route_to_queue` takes
  its triage-to-deep-review branch only when a LIVE deep-review bundle exists), `route_cli.
  _held_in_a_queue` and its held-bundle question, `DLCOrchestrator._already_done_or_held` (also the
  CPU pose pool's rule, and what `_record_finished_child` records), `_retire_claim_after_hold`,
  `ProcessingOrchestrator._no_pose_for_reprocess`, `repose._held_in_triage_on_disk` (which still
  answers True when the share cannot be read), `review_gate._drop_triage_copy`, `census.runner.
  bundles_in` (also the consistency check's `reconcile`) and the dashboard's `folder_scan`. The
  return scan already skipped such folders. WHY: a later triage route of a video whose deep-review
  folder was only a leftover took "deep review supersedes triage", set the row to 'deep_review',
  moved nothing and returned the leftover; the run's files stayed in Processing and the video had no
  live bundle. Tests: `tests/test_leftover_is_not_a_hold.py`.
- **A person's results file is read, not judged by name.** `review_routing.is_person_file` reads
  `<video>_segments.json`, `_reaches.json` and `_pellet_outcomes.json` (`_REVIEWED_IN_PLACE`) and
  counts one as a person's when it holds `validation_status` 'validated', `boundary_source` 'human',
  or `human_verified` / `human_corrected` / `triage_cleared` true anywhere inside (top level, per
  segment, per reach), or a segment `review/clear_guard` counts as locked; an unreadable one counts as
  a person's. Other machine outputs (features, processing manifest, triage record, assignments) are
  judged by name: they are derived and made again. Every caller (`clear_leftover`,
  `_move_keeping_person_files`, `review_return._files_in_the_way`, `queue_folder_is_live`,
  `discard_run_outputs`) still deletes or replaces a byte-identical copy. WHY: the review tools save
  into the machine-named files (the segmentation review writes 'validated' with no
  `boundary_source`), so a name rule deleted or replaced a person's corrections. Tests:
  `tests/test_reviewed_results_are_a_persons_files.py`.
- **Leftovers are listed in the review tool.** `review_routing.leftover_folders()` lists every
  folder holding `LEFT_NOTE` directly in either queue, read from the folders (not from any machine's
  attention list); the Review Queues tab's "Folders left for a person" panel shows them, plus
  `queue_folder_in_the_way` items from the attention lists, in `dashboard/attention_dialog.py`'s
  window. `attention_actions.actions_for` gives `left_for_a_person` (and `queue_folder_in_the_way`
  over such a folder) one action, 'delete_leftover' -> `review_routing.delete_leftover_folder`: only
  a folder named for the video, directly in a configured queue, holding the note, under the video's
  return claim; the note is deleted last, so a delete that stops part-way leaves a folder still
  marked as a leftover. A `queue_folder_in_the_way` over a live bundle gets a `note:` line (finish
  its review). Tests: `tests/test_leftovers_are_listed_in_the_review_tool.py`.
- **The dashboard's Resolve window.** `route_unfinished` items have these actions
  (`watcher/attention_actions.py`, `actions_for` / `perform`, shown by `dashboard/attention_dialog.py`):
  'finish_now' (the watcher's `_finish_failed_moves`, with this computer's watcher record) and
  'accept_missing:<file>' (only when the file is in neither place; crosses it off under the bundle
  claim, keeping the whole entry under `resolved_by_a_person`). Two differing copies, or a file at
  the source of a video not held in the bundle's queue (`_source_not_held` ->
  `review_return._held_in_this_queue`), get a `note:` line instead of a button. WHY no other
  button: every earlier action set a copy aside or deleted one, and which copy is right is a
  person's call. Test: `tests/test_attention_items_are_resolved_in_the_gui.py`.
- **An unusable pose of a filed older-model video is sent for a new pose.**
  `BaseOrchestrator._pose_check_before_segmentation` runs `pipeline.pose_check.check_pose` before
  segmentation in both pipelines (the processing server's `_run_pipeline` and a GPU node's local
  pipeline). Unusable (does not load as a pose table, no frames, or lacks SABL/SABR, plus
  BOXL/BOXR on a segmented tray) -> `_pose_is_unusable(video_id, why, pose_path)`, which logs a
  `pose_check failed` step and then acts only in the one settled case,
  `_new_pose_requestable`: a filed video (row `archived_at` set and `repose.archived_video` found)
  whose unusable pose is from another model than the declared one (`repose.scorer_of(pose_path) !=
  repose.declared_scorer()`) is set `outdated` with `reprocess_scope='full'` and `mark_reason='pose
  unusable: ...'` (the existing re-pose round trip then publishes a request:
  `declared_pose_in_archive` finds no declared-model pose), the run stops, and the row is listed
  grey in `attention/<host>_waiting_for_pose.json` (`_write_pose_attention`, 'outdated' rows only);
  a later usable pose clears `mark_reason`. Every other case returns 'as_before' and the run goes on
  exactly as before the check existed. The quality triage's 'Could not read DLC file' and 'Missing
  SABL or SABR' flags carry category `pose_unusable` (`pose_check.POSE_UNUSABLE`);
  `_run_quality_triage` returns `review_gate.QC_POSE_UNUSABLE` for them only when
  `_new_pose_requestable` holds, and the verdict as before otherwise. A file that cannot be opened
  stays with `segmentation_could_not_run`. OPEN (lab decisions, not implemented): what to do with
  a filed video whose unusable pose is the declared model's own (the round trip matches a
  declared-model pose by file name only, so it would find the same file and never write a
  request), how a video that is not filed gets a new pose, and after how many unusable poses a
  video is listed red. Test: `tests/test_unusable_pose_is_never_sent_to_review.py`.
- **A refused route never fails the video.** In the watcher's pipeline (the processing server's
  `_gate_held` and segmentation-failed route, and the same two places of a GPU node that also
  processes), `AlreadyInQueueError` is caught: nothing has moved, the video stays 'processing' with
  its files in Processing, one line `review_gate refused` goes into its processing log, and the
  pipeline does not pick it again for 30 minutes (`ROUTE_REFUSED_WAIT_S`,
  `_note_route_refused` / `_route_refused_waiting`). WHY: a refusal is a passing state of the
  shared folders (a file held open in a review window, another program holding the claim), and
  `mark_failed` counted it towards the video's retry limit, so an open window could leave the video
  failed for good. The refusal is listed in the attention panel.
  The wait repeats every 30 minutes for as long as the cause lasts (a live bundle in the queue, or
  an old folder kept for a person: until a person has dealt with that folder). A refusal after the analysis (`_gate_held`, the local
  pipeline's gate, `_route_pellet_numbers_unknown`) is noted `analysed=True`
  (`_refused_after_analysis`), and the next run reuses the stage outputs in Processing exactly as
  a video held for its hand scores does (`_forget_route_refusal` returns True; both pipelines'
  resume test) -- no re-segmentation every half hour. The segmentation-failed route's refusal is
  not, so its next run segments again. The list is in memory: after a restart the video runs in
  full once (`tests/test_refused_route_reuses_the_analysis.py`).
  The route the hand-score check makes (pellet numbers in doubt -> manual segmentation,
  `_route_pellet_numbers_unknown`) is handled the same way: a refusal there is a refused route,
  not "the hand-score check could not run". A video waiting after a refused route does not count
  against `max_local_pending` (`_held_for_refused_route`), from the refusal until the pipeline
  next starts on it (`_forget_route_refusal`), exactly as videos held for their hand scores do not:
  it is analysed already and only waits for a queue folder, and a refusal can last a weekend (a
  window holding a file open) or until a person deals with a folder. Counted, two of them stopped all
  intake with the default limit of 2. Like the hand-score holds, the list lives only in the
  watcher's memory: after a restart each such video is run once more and, if still refused,
  listed again. Test: `tests/test_refused_route_does_not_fail_the_video.py`.
- **Hold recorded before the manifest.** `route_to_queue` sets the database state right after
  `move_video_bundle`, then writes the review manifest with the one-step writer
  (`fsutil.write_json_atomic`, retried on a passing hold); a failure there is logged, not raised.
  `review_return.repair_unfinished_routes` (called by every return scan, acting at most every
  `ROUTE_REPAIR_INTERVAL_S` = 10 minutes) writes a missing manifest again for any bundle in either
  queue that has a routing record and some file of the video, under the return claim, with the
  step that sent the video (`route_to_queue` keeps `held_at_step` in the routing record too), so
  the return still resumes just after that step. Whether a manifest is missing is read from the
  bundle's listing (the triage scan's own listing is reused); a routing record is opened only when
  its size or modified time changed since the last pass (`_record_flags`, `_record_flags_cache`).
  WHY: opening a record on the share costs several times a listing, and reading every record of a
  queue of thousands of bundles every ten minutes kept the watcher on the share for minutes.
  Tests: `tests/test_route_records_hold_before_manifest.py`,
  `tests/test_unfinished_routes_are_finished.py::test_a_routing_record_is_read_only_when_it_changed`.
- **Returns are all or nothing.** `_return_to_processing` plans every move (inputs first, the
  clear marker last), records each one done, and sets 'processing' (`_set_processing_state`) in
  the same try. Any failure puts every moved file back (`_undo_return_moves`, newest first; a
  partial copy at the destination of the move that failed is deleted when its source is still in
  the bundle and the move created it -- the destination was not there before, or is identical --
  so a file of another run already in Processing is never deleted), backs the bundle off for
  `RETURN_RETRY_S`, and returns False. The queue records are
  deleted only after the state is set. Before any move, `_files_in_the_way` checks every planned
  destination: a different same-named file in the processing folder that is a person's file or an
  input stops the return (nothing moves; backed off; listed as `return_blocked`, dropped when a
  return of that bundle goes through); an identical file or an older run's machine output is
  replaced. WHY: the move replaced it (a person's hand-set cuts were lost), and setting it aside in
  a hidden folder hid it from everyone. Tests: section (f) of `tests/test_review_return_safety.py`.
- **Person's files.** `review_routing.is_person_file` / `is_machine_output` / `is_input_file`
  classify a video's files: machine outputs are `archive.supersede._ALGO_SUFFIXES` plus the queue
  records; anything else not an input, and a `_segments.json` with `boundary_source` 'human' (or
  unreadable), is a person's. `move_video_bundle` (and its publish fallback) never replaces a
  person's file with a different one (`PersonFileInTheWay`, listed in `failed_files`).
  `review_return._retire_stale_bundle` clears a stale bundle as a leftover against the live copy
  the caller names (see "Queue leftovers" above). Tests:
  `tests/test_person_files_are_kept_apart.py`, `tests/test_queue_folder_never_blocks_a_route.py`.
- **Deep review beats triage without deleting a person's file.** `_drop_triage_copy` no longer
  removes the triage folder whole, and `discard_run_outputs` keeps a person's file where it is
  (listed) unless the deep-review bundle holds it byte for byte (see "deep review supersedes
  triage" above). No answer is carried into the deep-review bundle by `_empty_triage_copy`. WHY:
  re-segmentation voids every answer after it; a review or ground truth moved into the deep-review
  bundle travelled back into Processing with it. NOT YET DONE: the review's copy in the durable
  store is still read by the review readers (`causal_review_io.find_review`, `truth_resolver`),
  which match it to the new cuts by frame span (`index_review_by_segment` drops it when its frames
  no longer fall in one segment); and Escalate and the mislabel divert still carry the triage
  bundle's review and ground truth into deep review. Voiding answers made on the old cuts is
  separate work. Tests: `tests/test_deep_review_keeps_triage_person_files.py`.
- **Queue folder check reads the dict entries.** `orchestrator.queue_folder_holds_this_run`
  reads each `failed_files` entry's `name` (entries are dicts with `name`, `error`, `source`; the
  oldest records hold bare names), so a route that could not move the run's processing manifest is
  recognised as holding this run. Before, the str() of the dict was taken for a file name and the
  check never matched. Tests: `tests/test_queue_folder_holds_this_run.py`.
- **Unfinished routes.** `move_video_bundle(..., all_or_nothing=True)` (passed through
  `route_to_queue` / `route_deep_review`) puts every moved file back and raises
  `RouteIncompleteError` when any file fails; `route_cli.route_video` and the reprocessor's
  mislabel divert use it, since their source is the filed folder. The pipeline's own routes out of
  Processing keep going and record each failed file with its `source`;
  `review_return.repair_unfinished_routes` (the same ten-minute pass as the manifest repair) moves
  each one in under the return claim when it is still at its recorded place and no newer than
  `routed_at` (+60 s), records it under `finished_later` -- the whole entry (its `error` and
  `source`) with `finished` (how), `finished_at` and `finished_on` -- and lists every bundle still missing a
  file in `<NAS root>/attention/<host>_unfinished_routes.json` (written only when the list
  changes, and only when there is something to list or an earlier list to empty). A move that
  fails leaves no copy at its destination (`review_routing._move_leaving_no_copy`: a file held
  open is copied and then not deleted by `shutil.move`; the copy is deleted when it is new there
  or identical). `_finish_failed_moves` handles a listed file that is in the bundle AND still at
  its source: identical -- the source copy is deleted and recorded under `finished_later`;
  different -- listed, nothing deleted. A listed file that is in the bundle and at none of its
  candidate places (the entry's `source`, `<source_dir>/<name>`, `.incoming/<video>/<name>`; a
  candidate that is the bundle's own file is skipped) is crossed off as finished too -- a person
  moved it in, as the attention panel asks. The record is rewritten with `os.utime` putting back
  its old modified time, because `_superseded_by_archive` judges a bundle by its files' times. Every
  write of a routing record replaces it in one step (`review_routing._write_routing_manifest` ->
  `fsutil.write_json_atomic`): a rewrite cut off part-way leaves the old record whole instead of a
  truncated one that reads as no record (`tests/test_routing_record_rewrite_is_atomic.py`). Every routing record names the machine that made it
  (`routed_by`); a source not under the NAS or review root is finished only by that machine
  (`_on_shared_storage`), and an entry of a record without `routed_by` whose local source file
  exists is listed, never acted on. With a watcher database (`scan_review_queues` passes its `db`
  to `repair_unfinished_routes`, and on to `_finish_failed_moves`), a file at a source is moved or
  deleted only while the row says the video is held in that bundle's queue
  (`_held_in_this_queue`); otherwise, or when the row is missing, the entry is listed for a person
  and nothing moves. WHY: intake and returns keep a file's modified time, so a re-run's mp4 at the
  source looks older than the route and the newer-than-the-route rule cannot protect it. Tests:
  `tests/test_unfinished_routes_are_finished.py` (the held-file tests hold a real open handle).



## Update: failed and unresolvable videos -- state changes, resets, retries and re-drives

This section collects how a video that went wrong on one machine is changed, reset, retried
and taken up again. Each part says what a person sees and what to do.

**A state change is one step.** Every change of a video's state in the watcher database
(`update_state`, `force_state`, `mark_failed`, `mark_unresolvable`, `reset_failed`) reads the
row's current state and writes the new one while holding the database's write lock. Before, the
read and the write were two separate statements, and another program writing the same database
(a review window, route-to-queue, a command-line tool) could change the row in between; its
change was then silently overwritten by a state that had been checked against the old one. Now
the second program waits for the first (or, after the wait limit, gets "database is locked",
which the watcher retries on a network drive) and is checked against the state the first one
wrote. A `force_state` WARNING line is written only after the change is saved. For an operator:
nothing to do. Where it lives: `watcher/db.py` `WatcherDB._state_write`; tests in
`tests/test_state_change_under_write_lock.py`.

**A person's reset (`mousereach-watch-reprocess`).** Three defects are fixed:
- With no `--from-step` the restart point used to be guessed from the row alone: 'discovered'
  for a row with no recorded pose and no validation time -- not a legal state after 'failed',
  so the reset raised -- or 'validated', which on a GPU node means "waiting in the shared singles
  folder", where the file was not. It is now read from the files this machine holds, exactly as
  the automatic retries read them (`WatcherDB.infer_reset_state`): video and pose in the
  Processing folder -> `processing`; video in the DLC queue -> `dlc_complete` with a pose from
  the declared model beside it, else `dlc_queued`; a single whose copy into the DLC queue failed
  (error starting "could not copy it from the shared singles folder") and that is back at the
  top of the shared singles folder -> `validated` (that failure gave it back there, and its
  message tells a person to run this command). No files here: nothing changes and the
  `--from-step` choices are printed.
- A reset kept the failure count, so the very next failure was already past
  `watcher.max_retries` and the automatic retries never applied again. A reset now sets the
  count to 0; the old count and error are written to the processing log (step `reset`).
- For a video that was not 'failed' the command asked "Reset anyway?" and printed "Reset" --
  but nothing was changed. It now changes only a 'failed' row (the UPDATE itself requires
  'failed'), says "<video> not changed: it is '<state>', not failed." with what to do instead,
  and exits with code 1. `--all-failed` prints how many were really reset ("Reset K of N").
  The word after `--from-step` is no longer mistaken for the video id.
Where it lives: `watcher/db.py` `reset_failed`, `infer_reset_state`; `watcher/cli.py`
`main_reprocess`; tests in `tests/test_reset_failed_video.py`.

**`mousereach-watch-process-animal` never leaves a claim behind.** It queues a single only when
the video has no row on this machine or its row is `validated`, `failed` or `unresolvable` --
the states that can go to `dlc_queued`. Every other state is skipped ("SKIP <video> (already
<state>)") BEFORE anything is claimed. It used to skip a fixed list that missed `triage`,
`deep_review`, `outdated`, `archiving` and `quarantined`: for those it claimed the shared file,
then the move to `dlc_queued` raised, and the command stopped mid-list (or left its copy in the
DLC queue) with the video in this machine's claim folder. A video a node already holds in its
claim folder is skipped too ("a copy is being posed by <node>; this one is a second copy").
Whatever fails after the claim -- the copy or the state change -- gives the claim back to the
shared folder and removes the DLC-queue copy this run made. For an operator: a claim left by the
old behaviour on a held video sits in `Single_Animal/.inflight/<this machine>/`; move that file
back to the top of `Single_Animal` by hand. Where it lives: `watcher/cli.py`
`main_process_animal`; tests in `tests/test_process_animal_states.py`.

**`mousereach-watch-unresolvable --retry` takes back only this node's own copy.** It used to
search every folder, the posed folder and the filed folder included, and force `dlc_queued` on
whatever it found, on any machine. On the processing server no work step selects `dlc_queued`,
so such a row was stranded; on a GPU node a posed or filed copy is refused by the pose step and
parked again. Now:
- on a processing server it changes nothing and says that its watcher takes a video back by
  itself (the posed-folder scan) once the video and its pose are in Processing/Posed;
- on a GPU node only `DLC_Queue/<video>.mp4` is queued (`dlc_queued`); a copy at the top of the
  shared singles folder keeps the row `unresolvable` with the shared-folder reason, so the
  singles intake claims it properly (a row already waiting for the intake, or recorded as a
  second copy of a video another node holds, keeps its reason); any other copy is listed and
  left alone.
For an operator: on the processing server, rows put in `dlc_queued` by the old `--retry` are not
picked up by anything. `mousereach-watch-status` lists them under dlc_queued; a person checks
each one (where are its video and pose?) before anything is changed. Where it lives: `watcher/cli.py` `_retry_unresolvable`; tests in
`tests/test_unresolvable_retry.py`.

**Two kinds of 'unresolvable' row come back by themselves.**
- *Processing server, intake found the posed files incomplete.* A GPU node hands a video and its
  pose over one file at a time, so the server's intake can look while the set is still incomplete.
  The row is parked `unresolvable` with the reason "posed files not all in the posed folder when
  intake looked (...)" (`db.INTAKE_MISS_REASON`) and the server lets go of its posed claim at once.
  The posed-folder scan (`state.discover_dlc_staged`) used to skip every video it already knew, so
  such a row came back only through the re-pose adoption -- as a re-run of a filed video, in the
  lowest-priority bucket. Now that scan sets exactly such a row back to `dlc_complete` (ready for
  intake) as soon as the video and its pose are both in the posed folder; the re-pose adoption
  leaves it alone. Any other 'unresolvable' reason is not taken in by this scan.
- *GPU node, startup recovery: another node had the video before its pose finished.* The shared
  record says another node has the video at `discovered`, `validated`, `dlc_queued` or
  `dlc_running`, and this node has no file. The reason now starts "no file on this node; another
  node had it before its pose" (`single_claim.ELSEWHERE_BEFORE_POSE_REASON`), one of the reasons
  the singles intake re-drives: if that node gives the video back to the top of the shared singles
  folder, and no node holds a claim on it, the intake sets the row back to `validated` and any node
  may take it -- without waiting for this node's next restart. When the other node is past the
  pose (or the video is filed) the reason stays "no file on this node; <node> has it at state
  '<state>'", which is never re-driven.
For an operator: nothing to do. Rows parked before this change with the old wording come back
only at a restart, as before. Where it lives: `watcher/db.py` `INTAKE_MISS_REASON`;
`watcher/state.py` `discover_dlc_staged`; `watcher/repose.py` `adopt_staged_reposes`;
`watcher/coordination.py` `_elsewhere_reason`; `watcher/single_claim.py` `REDRIVE_REASONS`;
`ProcessingOrchestrator._intake_from_staging`. Tests in `tests/test_unresolvable_rows_come_back.py`.

**A GPU node retries a single whose copy into its DLC queue failed.** When the copy of a claimed
single into the node's DLC queue fails (a full disk, a dropped share), the video goes back to the
shared singles folder and the row becomes `failed` with a reason starting "could not copy it from
the shared singles folder into the DLC queue" (`single_claim.COPY_FAILED_REASON`). A GPU node's
work loop has no job list for failed rows, and the singles intake skipped every video it knew, so
this node never took it again -- with a single GPU node, nothing did. Now the singles intake sets
such a row back to `validated` (ready to be taken) once an hour has passed since the failure, while
its failure count is below `watcher.max_retries`, the file is back at the top of the folder and
stable, and no node holds a claim on it. The failure still counts; after the last retry the row
stays `failed` for a person (`mousereach-watch-reprocess <video>` once the cause is fixed). WHY the
hour and the limit: without them a node with a full disk would copy a whole video every couple of
minutes. Only this reason is retried this way: a failed pose or hand-off is not. Where it lives:
`watcher/state.py` `_copy_failed_retry_due`, `discover_new_singles`;
`DLCOrchestrator._adopt_single_for_dlc`. Tests in `tests/test_copy_failed_single_retried.py`.

**The hand-off to the posed folder moves only files this GPU node holds.** The stage step
(`_stage_to_nas`) finds a `dlc_complete` video by name and MOVES every file of it that it finds
into Processing/Posed. Its search also covered the top of the shared singles folder, and a path
recorded on the row inside the filed folder was used even with the archive search switched off.
So a row with no local copy (after cross-node recovery, or after a node gave the video back) moved
a shared, unposed video into Posed and deleted it from the singles folder; a row recording a filed
path would have moved a filed video's results. Now:
- the stage step hands on only a copy in this node's DLC queue or its own claim folder (the pose
  step's rule). Otherwise nothing is moved and the row becomes `unresolvable`, with the reason
  naming where the copy was; an unclaimed copy at the top of the shared folder gets the
  shared-folder reason, so the singles intake claims it properly;
- `locate_video_file` / `locate_pose_file` with the archive search off ignore a recorded path
  inside the filed folder, as they already ignored one inside Posed with the staging search off;
- startup recovery does not count a file at the top of the shared singles folder as "on this
  node" (it is waiting for any node to claim it);
- `mousereach-watch-unresolvable --sweep` on a GPU node looks only in this node's own folders: a
  copy in the filed folder, in Posed, at the top of the shared folder or in another node's claim
  folder is not recorded against a working state; such a row is swept to `unresolvable`. On a
  processing server (the role is read from the machine's watcher settings) the sweep still counts
  a copy in Posed -- where the server works a posed video from -- and, for a row in `outdated` or
  `archiving` only, the video's filed copy, so those rows get their path filled in as before.
For an operator: nothing to do. Where it lives: `watcher/locate.py` (`is_in_archive`);
`DLCOrchestrator._stage_to_nas`; `watcher/coordination.py` `_file_on_this_node`; `watcher/cli.py`
`main_unresolvable`. Tests in `tests/test_hand_off_moves_only_own_files.py`.

**A GPU node works its DLC-queue copy only while it holds the video.** The rule is "a node either
holds the claim or holds no copy it works". A node gives a claim back when its pose fails, when a
hand-off fails (the claim is no longer kept alive) or when it pauses; when the give-back could not
confirm the shared copy, the node kept its local DLC-queue copy -- and at its next start (or when a
person queued the row by hand) it posed that copy again while another node posed the shared one.
Now, at the start of every pose and when the watcher starts (`_recover_local_dlc_queue`, for rows
that are `unresolvable`, or `failed` with retries left), a DLC-queue copy is NOT worked when this node holds no claim
on the video (`Single_Animal/.inflight/<this node>/`), no re-pose was requested for it, and the
video is also:
- at the top of the shared singles folder -- the row stays `unresolvable` with the shared-folder
  reason, and the singles intake claims it properly (on this node or another);
- in another node's claim folder -- `unresolvable`, naming that node;
- already in the posed folder, the video AND a pose of it -- `unresolvable`, naming the posed copy.
  A video there without a pose is a hand-off that stopped half way (the video is copied first);
  the node finishes the hand-off from its own copies instead (`failed` with retries left ->
  `dlc_complete` when its pose is beside the queue copy).
A `failed` row that has used all its retries (`error_count` has reached `watcher.max_retries`) is
not looked at by the restart at all: it stays `failed` for a person (`mousereach-watch-reprocess`),
and the log says "failed with no retries left ... the row stays 'failed'". WHY: re-labelling it
`unresolvable` took it off the failed list, the singles intake then set it back to `validated`, and
the video was posed again once per restart with no limit.
A video in none of those places (one this node cut and could not publish, or one dropped straight
into its DLC queue) is worked as before, and so is a video this node holds the claim on. The
processing log shows step `dlc`, status `skipped` with the reason. A DLC-queue copy that is not
worked for one of these reasons is a machine leftover: the node DELETES `DLC_Queue/<video>.mp4` once
the copy it found elsewhere is confirmed to be the same size (the log says "deleted its DLC_Queue
copy ...; the current copy is ..."). If the sizes differ, or the other copy cannot be read, the
queue copy stays where it is -- it may be the only good copy -- and is not worked; nothing is ever
moved to a side folder. For an operator: nothing to do.
Known leftover (not closed here): a failed pose or hand-off is still retried only at a restart on
the node that failed it, and the CPU-pool fix below covers its own hand-off only.
Where it lives: `DLCOrchestrator._queue_copy_elsewhere`, `_drop_superseded_queue_copy`, `_process_single_dlc`,
`_recover_local_dlc_queue`. Tests in `tests/test_queue_copy_needs_the_claim.py`.

**The CPU pose pool retries a failed hand-on while it runs.** When the pool (`mousereach-cpu-pose`)
finished a pose but could not hand it on to Processing/Posed (a dropped share, say), the pose waited
in its work folder "until the next start" and its claim was no longer refreshed -- only running
poses were. A day later the shared copy went back to the singles folder for any node, which posed
the video again, and the pool then handed on its own pose too at its next start. Now the pool keeps
the claim of such a pose alive with the running ones and tries the hand-on again: at once, then
after 1, 2, 4 ... minutes, never more than 30 minutes apart (each try copies a whole video over the
share). While the shared singles folder cannot be read at all (the share is down), nothing is
decided: the pose and its claim wait. Only when the folder can be read and this machine's claimed
copy is not in it (another machine broke the claim, or a person moved the file) is the pose not
handed on. Its local copies in `CPU_Pose` are machine leftovers, and what happens to them depends
on whether the video's current copy can be confirmed somewhere else -- the video file, the same size
as the local one, at the top of the shared singles folder, in another machine's claim folder, or in
Processing/Posed WITH its pose file (`<video>DLC...h5`) beside it; or the video already filed or held
in a review queue. A video alone in Processing/Posed, with no pose beside it, does NOT count: it is a
hand-on that stopped half way, which the server never takes in (the graphics-card nodes apply the
same rule):
- confirmed: the local copies are DELETED, and the log says "its claim is gone and its current copy
  is at ...; its pose is not handed on and its local copies were deleted";
- not confirmed anywhere: they stay exactly where they are (the local video may be the only copy),
  the log warns once "its claim is gone and no current copy of the video was found elsewhere ...;
  find where the video went", and the pool looks again every cycle -- as soon as a current copy
  turns up, the leftovers are deleted.
Nothing is ever moved to a side folder. When the pool starts, finished poses a previous run left
with no claim of this machine are checked the same way -- but only when this machine's claim folder
could actually be read: during a share outage the pool does not take "I could not read my claims"
for "I have no claims", and leaves such copies for the next start. A video in `CPU_Pose` whose pose
never finished (no `.h5` beside it) is never handed on, by a retry or at a start: a bare video in
Processing/Posed would sit there untaken while its claimed shared copy was deleted. Its claim is
simply no longer kept alive, so after the usual quiet period it goes back to the singles folder for
any node to pose. Its log for a failed try says "could not hand
on (...); ... its claim is kept, and the hand-on is tried again, waiting longer after each failure".
For an operator: nothing to do, unless the "no current copy ... found elsewhere" warning appears --
then find where the video went (a person moved it, or another machine lost it); the copy in
`CPU_Pose` is kept until a current copy exists elsewhere. Where it lives: `watcher/cpu_pose_pool.py`
`CpuPosePool._retry_hand_ons`, `_current_copy_elsewhere`, `recover`, `run`. Tests in
`tests/test_cpu_pool_hand_on_retry.py`.

**The processing server asks the shared drive before taking in a posed video it has no row for.**
A video's posed copy stays in Processing/Posed while the video is analysed, held in a review queue
or failed; it is removed only once the video is filed. A server whose database has no row for such a
video (a rebuilt or replaced database, a row never written) used to register the posed copy as new
work, analyse the video again and route it into the queue folder that already held it. The
posed-folder scan (`state.discover_dlc_staged`) now asks the same question a GPU node asks before
every pose (`locate.held_or_filed_on_share`):
- the video's bundle is in the triage or deep-review queue (or being routed into one, `.incoming`):
  the row is recorded in that queue's state and nothing is taken in. The queue folder must hold
  bundle files (any file apart from dot-files and file-manager leftovers such as `desktop.ini`); an
  EMPTY folder is residue, not a hold, and the video is taken in as usual. WHY: counted as a hold,
  an empty folder made the row 'triage' or 'deep_review'; the review return scan then removed the
  empty folder, and the row stayed held for good with nothing in any review tool. This NARROWS the
  hold rule everywhere it is used -- before, a queue folder counted as a hold as soon as it existed.
  A folder kept for a person (one holding the left-over note) is not a hold either, as in every
  other reader of the review queues (`review_routing.is_queue_bundle`).
  The same rule now applies to a GPU node's check before a pose
  (`DLCOrchestrator._already_done_or_held`) and to the CPU pose pool: an empty queue folder no
  longer stops a pose. Only files directly in the queue folder count: real bundles are flat files
  (the review routing writes them that way), so a folder holding only sub-folders or dot-files is
  not a bundle;
- the video is filed and the posed pose is the one it was filed with (same DLC scorer as the filed
  manifest): the row is recorded `archived` and nothing is taken in;
- the video is filed but the posed pose is a different one: new pose information, taken in as
  before.
The processing log shows step `intake`, status `skipped`, with the reason. For an operator: nothing
to do.
Known leftovers (not closed here): the intake of a video that already HAS a row does not ask this
question; and a held, failed or being-analyzed video's posed copy still stays in Processing/Posed,
so that folder does not yet hold only waiting work -- whether that copy is removed once the review
bundle is published with the same video and pose, or kept somewhere that is not a waiting-work
location, is a decision still to be made. Where it
lives: `watcher/locate.py` `held_or_filed_on_share` (moved there from
`DLCOrchestrator._already_done_or_held`, which now calls it); `watcher/state.py`
`_held_or_filed_instead`. Tests in `tests/test_posed_scan_asks_held_or_filed.py`.

## Update 2026-11-17: a collage that cannot be cropped, and a crop claim that is stuck, are listed for a person

What this means for an operator: a GPU node that cannot crop a collage now says so where a person
looks (the Pipeline Dashboard's red "Needs your attention" panel), instead of only in its log.
Each part below says what you see and what to do.

**An error before the crop starts.** If something fails while a node is still checking a collage
(its shared claim, or whether its children already exist) the collage has not started cropping, so
it cannot be recorded `failed` -- the database only lets a collage that was cropping fail. That
refused write used to escape into the watcher's main loop, and the same collage was picked again on
the very next pass. Now (`DLCOrchestrator._fail_collage_after_dispatch_error`) the collage is left
as it was (`stable`), the error is logged at ERROR with the collage's name, and the collage is not
picked again for 30 minutes (`_CLAIM_RECHECK_S`), so a passing error is retried and a lasting one
does not spin the loop. A collage that was already cropping is recorded `failed` with the error,
exactly as before. Nothing to do; if the same ERROR comes back every 30 minutes, its message names
the cause. The wait is in memory: a watcher restart tries the collage again straight away.
Tests: `tests/test_collage_crop_failures_are_visible.py`.

**A collage that has used up its retries.** A collage whose crop failed is tried again while its
file is still in the intake folder (Unanalyzed\Multi-Animal), up to `watcher.max_retries` (3) times.
After that it used to be "left parked" with a DEBUG line that nobody reads, and the sessions on it
were never cut. Now:
- the watcher log says it once per watcher run at WARNING, with the last error (`state.py`
  `discover_new_collages`, which records every such collage in `parked_collages` on each scan);
- every scan, the GPU node lists it in `<NAS root>/attention/<host>_collages.json`
  (`DLCOrchestrator._list_parked_collages`), so it appears in the Pipeline Dashboard's red "Needs
  your attention" panel with its name and last error. The item's kind is `collage_crop_parked`; it
  names the recording in `collage` and has no cohort (`cohort_id` empty), because a collage holds
  several animals and must not be filed under one animal's cohort;
- before listing, the node asks the shared claim: if another node has already cropped the collage,
  the row here is set `cropped` (the processing log names the node) and nothing is listed; if
  another node is cropping it right now (a claim younger than 24 hours), it is left to that node;
- the dashboard's health line now also names every collage this machine could not crop, with how
  often it was retried and the last error (`watcher/health.py`).

What to do when one is listed: read its error. Fix the cause, then on the machine named in the list
(the file name starts with it) open Watcher Control, raise `max_retries` by one in "Configuration
(applies on next start)", press Save config, and restart the watcher (Stop, then Start). The collage
is tried once more (failed videos on that machine get one more try too). When the collage crops, it drops off the
list at the next scan; it also drops off if its file is taken out of the intake folder. The list
file is rewritten only when it changes, and a list that cannot be written is a WARNING in the log,
never a stop of the scan.

**A crop claim that waits for its holder for ever.** A GPU node takes over another node's crop
claim once it has stayed unfinished ("cropping") for more than 24 hours -- except when videos cut
from that collage are already on record, because a second crop would pose them twice (see "Stale
claims" in the 2026-09-15 update above). Such a collage then waits for the node that holds the
claim, and if that node never runs again it is never finished; the only trace used to be a log
WARNING on each node that asked. Now every GPU node that meets such a claim lists it in
`<NAS root>/attention/<host>_collage_claims.json` (`DLCOrchestrator._list_stuck_collage_claims`;
kind `collage_claim_stuck`, with `holder` and `claimed_at`, no cohort), and the dashboard's red
panel shows it -- once, even when several nodes list it (`attention.without_repeats`). It leaves the
list when the claim is finished, given back or taken over, when the claim is younger than 24 hours
again, or when the collage is no longer waiting on this node (cropped or failed here); the list is
emptied at each watcher's first scan and rebuilt as claims are asked about (each collage is asked
about at most every 30 minutes).

What to do when one is listed: start the watcher on the machine named as the holder. At its start
it crops the collage again, records it as cropped (see the next paragraph), or lists it as a
collage it could not crop. If that machine will never run again, a person has to decide what is
left to crop: no button releases a claim yet.

**A "cropped" mark lost at a restart.** When a node finishes cropping a collage it marks its shared
claim "cropped"; if that write fails it is resent until it lands (rule 6 of the 2026-09-15 update).
The list of marks still to send was kept only in memory, so a watcher stopped or crashed before the
mark landed forgot it, and the shared claim stayed "cropping" under that node's name -- after a day
another node could take it over and crop the collage again, or, with its videos on record, it waited
for ever. Now (`DLCOrchestrator._seed_unsynced_collage_claims`) the node rebuilds that list from the
records themselves: every shared claim that is this node's and still "cropping" while the collage is
"cropped" (or later) here. It does this at startup, after every reconnect to the shared database,
and before the first resend; the resend (`_retry_unsynced_collage_claims`) now runs on every scan,
not only when a collage is about to be picked. The log says so at WARNING (`were cropped here but
their shared claim still says 'cropping' ... sending 'cropped' again`). Nothing to do.

**A cut video that reached neither the share nor the node's queue.** After cutting a collage, a GPU
node puts each single-animal video in two places: the shared drive (already claimed by this node, so
another node can take it over) and its own DLC queue. If BOTH failed for one video -- the share could
not be written and the local copy failed -- the working file was the
only copy left, yet it was deleted with the working folder while the collage was recorded "cropped"
here and on the shared claim, so no node ever cut that video again. Now
(`DLCOrchestrator._process_collage`) such a video counts as a slot that could not be cut: the
collage is recorded "failed" with an error naming the video (`... was cut, but copied neither to the
shared drive nor into this node's DLC queue`), it is retried like any failed collage (above), the
retry takes the kept file into the queue (a child left 'validated' with its working file kept is first
offered to the single-adopt path) or cuts it again, and its working file is kept until a later attempt has placed a copy.
The videos that were placed stay queued. A video that reached the shared drive but not the queue is
not a failure: the shared copy is claimed by this node and is taken from there.
On a node with NO DLC queue configured the video is instead marked unresolvable with the reason 'no local DLC
queue is configured on this node' and its file is kept; the collage can still finish 'cropped' in that case. If the retries run
out, the collage is listed in the red panel as described above.

## Update: a claim marker is never left empty

Claim markers (`Processing/Posed/.claims/`, `Review/.return_claims/`, made by
`watcher/claim_marker.py`) used to be created empty and then filled in with the name of the
machine holding the claim. A machine that stopped between those two steps -- a crash, a kill, a
network drive that dropped -- left an EMPTY marker. An empty marker reads as "claimed by an
unknown machine", so even the machine that made it was refused the claim and could not release
it, and nobody could touch that video until the marker was three hours old and the sweep broke
it. Now the marker's text is written first, to a temporary file beside it
(`<marker>.<machine>.<process>.<random>.tmp`), and the finished file is put in place under the
marker's name in one step that refuses to replace a marker already there (a rename on Windows, a
hard link elsewhere). So a marker is either there and names its holder, or is not there at all,
and the racing rule is unchanged: exactly one machine's marker appears. A machine that cannot
write the marker (the drive is unreachable) simply does not get the claim and tries again later;
the temporary file is removed. On a file system with no hard links (not Windows) the old
create-then-write is used, so a claim is still possible there.
For an operator: nothing to do. A leftover `*.tmp` file in a claims folder is harmless (nothing
reads it as a claim) and may be deleted. Where it lives: `claim_marker._create`. Tests in
`tests/test_claim_marker.py`.

## Update: one watcher per machine, its configured role only, and a left-behind stop flag refuses a start

Three ways a machine used to end up running the wrong watcher, or silently none:

- **A second watcher from the Watcher Control panel.** The command-line watcher took a
  machine-wide lock (a Windows named mutex) so that a second one exits; the panel's Start and Run
  Once buttons did not. They looked for another watcher only by reading running programs' command
  lines, which may not show a watcher of another Windows login. Now every start -- command line,
  scheduled task, Start, Run Once -- takes the SAME lock (`watcher/singleton.py`), and a start that
  finds it taken refuses ("A watcher is already running on this PC"). The panel keeps the lock while
  its watcher runs and gives it back when that watcher ends; a process that ends in any other way
  (a crash, a forced kill) loses it automatically. A lock created by another Windows login may be
  refused to this one with "access denied"; that also counts as taken. Where there is no such lock
  (not Windows) a start goes on, as before, and the log says the check is not available.
- **A watcher of the other role from the panel.** Start and Run Once used whatever the "Run as"
  drop-down showed instead of the machine's configured role, so a GPU-node watcher could run on a
  processing machine's database (or the reverse) and treat that machine's videos as work of its
  own. Now a "Run as" that differs from `mode` in the machine's settings is refused with a message
  saying how to change the role on purpose (change mode in the settings, Save, start again).
  Save also moves "Run as" to the saved mode, so that start is not refused a second time.
- **A stop flag left behind.** `watcher_stop.flag` (`cpu_pose_stop.flag` for the CPU pose pool)
  makes a running watcher finish its current item and exit. A flag forgotten after a stop or an
  update made every later start exit on its first turn with one INFO line and exit code 0 -- a
  scheduled task reported success while the machine did nothing. Now a start that finds the flag
  refuses: a WARNING (and, on the command line, a line on the error output) naming the file and how
  long ago it was left, exit code 3 for `mousereach-watch` and `mousereach-cpu-pose`, a message box
  in the panel, and, while no watcher runs, a line in the Dashboard's health report. The flag is
  NEVER deleted by a start: the update procedure relies on it keeping a stopped machine stopped
  until the person updating removes it. The CPU pool first gives back or hands on what a previous
  run left (as at every start), then refuses. A flag created while a watcher runs still stops it
  normally, with exit code 0. The Dashboard's Restart button, which a person presses on purpose,
  still clears the flag before it starts a watcher, as before.
For an operator: if a watcher will not start, read the message; USER_GUIDE.md "The watcher (or the
CPU pose pool) will not start" says what to do for each. Where it lives: `watcher/singleton.py`,
`cli.main_watch`, `WatcherControlWidget._take_start_guards`, `CpuPosePool.run`,
`health.health_report`. Tests in `tests/test_start_guards.py`.

## Update: the CPU pose pool shows the poses it could not hand on, tries them once more at a stop, and gives a video back after a day of failures

Three gaps were left after the CPU pose pool learned to retry a failed hand-on while it runs:

- **Nothing showed such a pose.** Its video stays claimed by the pool, but the pool's status report
  (`Processing/.node_status/<machine>-cpu.json`, read by `mousereach-watch-status`) listed only
  running poses. It now also lists them as `waiting_hand_on`. A list that does not empty means
  the hand-on keeps failing; the pool log says why.
- **A stop abandoned them.** When the pool stopped (stop flag, running poses finished), finished
  poses still waiting for their next try were left as they were, their claims no longer kept alive,
  until the next start. Now the pool tries each of them once more before it exits, whatever its
  wait. Any still not handed on are named in one warning ("stopping with N finished pose(s) not
  handed on ..."); the next start hands them on, as before.
- **A failure of the machine's own held the video for ever.** Some failures never clear by
  themselves -- a damaged local file, a posed folder this machine may not write -- and the video
  stayed claimed, unposed for everyone, for as long as the pool ran. Now, once the hand-on has
  failed for 24 hours AND at least 10 times (`HAND_ON_GIVE_UP_S`, `HAND_ON_GIVE_UP_ATTEMPTS`), the
  pool gives up: the claimed video goes back to the top of the shared singles folder for any node
  to pose, this pool leaves it alone for an hour, and the local copies are deleted once the video
  is confirmed back in the shared folder (otherwise they stay where they are). The pool log says
  "its hand-on failed N times over H h; the video is given back ...". WHY both conditions: no
  tries are made while the shared drive cannot be read, so a day-long outage is not a day of
  failures, and one failed try after the drive comes back does not give up. If the claimed video
  cannot be given back (a file of that name is already in the singles folder), nothing changes and
  the hand-on keeps being tried.
For an operator: nothing to do unless the give-up warning appears or `waiting_hand_on` stays
filled -- then read the pool log for the hand-on errors and fix what this machine cannot write.
Where it lives: `watcher/cpu_pose_pool.py` `CpuPosePool._status`, `_retry_hand_ons`,
`_give_up_hand_on`, `_last_hand_on_try`, `run`. Tests in `tests/test_cpu_pool_hand_on_retry.py`.

## Update: the status command and the dashboard health line name the stuck videos

`mousereach-watch-status` printed only counts by state. Three kinds of stuck video were not on
it: an analysed video whose filing failed (it stays `processed` while the watcher waits out a
back-off kept only in its memory, so it looked like one about to be filed), a failed video's
reason (in its row, never printed), and the holds the watchers list for a person in
`<NAS root>/attention/`. Now, under the counts, the command lists: the videos held after a failed
filing attempt (found from the log: a `processed` row whose newest log row, by id, is a failed
`archive` step -- `WatcherDB.get_videos_held_by_failed_archive`), the failed videos with their
error, and every attention item with its list file and that list's age (a list of a watcher that
has stopped is not rewritten, so its age says how current it is). Each part is read on its own;
one that cannot be read prints a `[!]` line and the rest still shows. `--json` carries them under
`held_and_failed`. The Pipeline Dashboard's health line also names the videos held after a failed
filing. Tests: `tests/test_status_shows_held_and_failed.py`.

## Update: only the watcher creates its database; other tools refuse a missing one

Opening the watcher database on a path where it does not exist CREATES an empty one there. A
tool pointed at the wrong file -- the default `<processing_root>/watcher.db` on a node whose
watcher writes a configured `watcher.db_path`, or a machine whose watcher has never run -- then
worked silently on that empty decoy. Now only `mousereach-watch` and the Watcher Control panel's
Start / Run Once create the database. Everything else opens the database the watcher really
writes (`db_location.resolve_watcher_db_path`: the configured `db_path` first) and refuses when
it is missing:

- `mousereach-watch-process-animal` checks before its y/N question; nothing is copied or queued.
- `mousereach-uncrystallize` refuses (exit 1) instead of unlocking nothing in a new file.
- `mousereach-repose-videos` refuses (exit 2) before any bundle moves.
- The archive command's filing record (`archive.core.record_archived`) logs a WARNING that the
  filing was not recorded and returns False.
- The Pipeline Dashboard's database buttons (Re-run video, the health line, Publication Lock,
  Reprocess outdated, Import archive) and its video list use the live database and show
  "no watcher database at <path> -- start the watcher on this machine once to create it" when it
  is missing. **Reprocess outdated** in particular used the default path, so on a node with a
  `db_path` override it marked videos out of date in the decoy, where the watcher never looked.

What a person does when a tool says the database is missing: start the watcher on that machine
once (`mousereach-watch`, or Start in the Watcher Control panel); if the watcher is already
running there, check `watcher.db_path` in `~/.mousereach/config.json` (`mousereach-watch-info`
shows it). A decoy `watcher.db` left beside a configured `db_path` is not read by anything in a
login whose config sets `db_path`; before renaming it aside, check every login's config on that
machine (`mousereach-watch-info` in each), because a login whose own config has no `db_path`
resolves `<processing_root>/watcher.db` as its real database. Tests: `tests/test_tools_never_create_a_decoy_watcher_db.py`,
`tests/test_archive_gate.py`.

## Update: publication locks are set and lifted on the processing server only

A publication lock (`mousereach-crystallize`, state `crystallized`) is written only to the
watcher database of the machine that runs the command, and only that machine's version scan
reads it. Videos are filed and re-run by the machine that analyses them -- the processing
server. A lock made on a pose-only GPU PC landed in a database whose rows mean "pose handed on",
protected nothing, and looked done. Now `mousereach-crystallize` (except `--list`),
`mousereach-uncrystallize` and the Pipeline Dashboard's Lock / Unlock buttons refuse unless the
machine's `watcher.mode` is `processing_server`; a GPU PC with `watcher.also_process` on (a lab
with no processing server) may lock, with a printed note that the lock is known to that machine
only. An unreadable configuration refuses. The rule and its reason are in
`pipeline/versions.py` `publication_lock_gate`. Tests: `tests/test_publication_lock_gate.py`.
A lock written to the file filed results sit beside (so every machine sees it) is a later step,
not built.

## Update: mousereach-reconcile sees a bundle parked for a new pose

`mousereach-repose-videos` takes a held video's review bundle out of its queue as one rename to
`Processing/Repose_Queue/_parked/<video>/`, and during the re-pose round trip that folder holds
the video's only copy. No reader named the folder, so `mousereach-reconcile` reported such a
video nowhere. It now reads the parked folder as a place: with a re-pose request in the queue or
in flight (or the new pose back in the posed folder) the video is "waiting for machine"; with no
request -- a re-pose run that stopped after parking -- it is "needs person", with the instruction
to run `mousereach-repose-videos` again for it (that run finishes from the parked folder); when a
GPU node gave up on the request it is also "needs person", naming the `<video>.failed.json` in
the re-pose queue instead (running the command again would be refused, the row being marked); and a
parked copy still there after the video was filed or held again is named in the video's detail.

Not changed here (follow-ups): the census and the Pipeline Dashboard's scan do not yet name the
parked folder; a stopped re-pose run still leaves the video's row saying it is in its review
queue; and nothing yet removes a parked folder once the re-posed video has been filed. Test:
`tests/test_reconcile.py` (the `parked` tests).

## Update: the backup leaves out claim folders and half-written files

The backup (`mousereach-backup`, and the Watcher Control panel's "Back up now") is add-only:
whatever it copies once stays on the backup for good. It used to copy the claim markers of
returns from review (`.return_claims`), each machine's liveness report (`.node_status`), files
still being written (`*.part`, `*.partial`) and SQLite rollback journals (`*.db-journal`).
Restored, a claim marker blocks every other machine from that video, a status report makes a
dead machine look alive, and a journal makes SQLite undo writes that finished long ago. Now
(`watcher/backup.py`, `_sync_dir`) robocopy leaves out the folders `.claims`, `.return_claims`
and `.node_status` and the files `*.tmp`, `*.part`, `*.partial`, `*.db-journal` and `*.db-shm`.
The hand-over folders `.inflight` and `.incoming` and SQLite `*-wal` files are still copied:
they hold real data. Nothing already on the backup is removed (the backup is add-only by design), so
a backup made before this change may still hold such files; the restore steps below say to
delete a restored `*.db-journal` before opening its database and to run `mousereach-reconcile`
after any restore.

The panel's "Back up now" built its own backup from the two folder boxes and ignored
`backup.sync_dirs` and the time limit, so it copied the full default scope; it also said
"Backup complete." when a folder failed. It now builds the backup through the same function as
the command (`backup.watcher_from_config`) and says "Backup INCOMPLETE" when a folder did not
copy completely. "Save backup settings" used to replace the whole `backup` section of the
config, dropping `sync_dirs` and the timings; it now changes only the switch and the two folders.
Test: `tests/test_backup_leaves_out_state.py`.

**What it copies.** The folders listed in `backup.sync_dirs` in
`~/.mousereach/config.json` (paths relative to `backup.source_root`), or the
built-in default list when that is not set. New and changed files are copied;
a file that is later moved or deleted on the source stays on the backup, at
every place it passed through. Nothing on the backup is ever removed.

**What it never copies, and why.** These are machine state or files still
being written, and a copy of them would be wrong the moment it is restored:

| Left out | What it is | Why a restored copy would do harm |
|---|---|---|
| `.claims`, `.return_claims` folders | "this machine is working on that video" markers | a restored marker blocks every other machine from the video |
| `.node_status` folder | each machine's "I was alive at this time" report | a restored report makes a dead machine look alive |
| `*.tmp`, `*.part`, `*.partial` | a file another job is still writing | a truncated file under a name nothing reads; the finished file is copied next run |
| `*.db-journal` | SQLite's record of a write in progress | restored beside its database, SQLite "undoes" writes that in fact finished long ago |
| `*.db-shm` | an index SQLite rebuilds itself | none needed |
| `__pycache__`, `.git`, `.pytest_cache`, `pytest-cache-files-*`, `*.pyc` | code and test leftovers | not data |

Still copied on purpose: the `.inflight` and `.incoming` folders (they hold real
videos in the middle of a hand-over -- possibly the only copy at that moment) and
SQLite `*-wal` files (they hold finished writes not yet folded into the `.db`
file; a database restored without its `-wal` loses them).

**The panel button.** Watcher Control -> Backups -> "Back up now" runs the same
backup as `mousereach-backup --once`, with the same scope (`backup.sync_dirs`)
and time limit, using the two folders typed in the panel. When a folder did not
copy completely the panel says "Backup INCOMPLETE" (the watcher log names the
files); run it again. "Save backup settings" changes only the on/off switch and
the two folders; the scope and the other backup settings in the config file are
kept.

**Restoring from the backup.** A restore gives you FILES, not the pipeline's
state. Do it in this order:

1. Stop the watcher on every machine that uses the restored folders.
2. Copy the files back.
3. Before opening any restored database (`*.db`), delete any `*.db-journal`
   file beside it. Backups made before this rule may still hold one, and
   SQLite would use it to undo finished writes.
4. Run `mousereach-reconcile` and read what it reports. Be aware: a restore
   can put a video in two places (its filed copy, and an old copy restored
   into a waiting or review-queue folder it passed through). Reconcile calls
   such a video done because its filed copy is complete; it does not point
   at the old copy. So after a restore, a copy of an already-filed video in a
   waiting or queue folder is an old one from the backup, not new work.
5. Start the watchers again.

## Update: a file released from quarantine is taken back in

A file whose name the watcher cannot read is moved to the quarantine folder (a note
`<file>.quarantine.json` beside it says why) and keeps its row in the watcher database in state
'quarantined'. Both intake scans skipped every file that already had a row, so a file released
under its OLD name -- the Quarantine tab pre-fills the old name, and a name can become valid when
a validator rule is corrected -- stayed in the intake folder for ever, although the tab had said
"the watcher will pick it up". Only a new name made a new row.

Now (`watcher/state.py`, `_recheck_released_collage` / `_recheck_released_single`): when an
intake scan finds a file whose row is 'quarantined' (a single only once its copy has finished,
the same stability wait as any new single), it checks the name again. A valid name takes the
state machine's own 'quarantined' -> 'validated' step, with the date, animals and tray read from
the name and the old error cleared, and the video goes on like any new one. A name that is still
invalid is quarantined again, with a fresh note, exactly as a new file would be. If the check itself fails (for example the database cannot be written), the watcher log
gets a WARNING naming the file, the rest of the scan goes on, and the file is tried again on the
next poll.

The Quarantine tab's "Release selected" now checks the corrected name with the rules of the
intake chosen under "Release to" BEFORE moving anything (`validator.release_from_quarantine`).
A name that intake cannot read is refused: the file and its note stay in quarantine, and the
message says what is wrong and, where it can, suggests a corrected name. The old name is not
refused -- it may be valid now. A file of the same name already in the intake folder also stops
the release; nothing is overwritten. Tests: `tests/test_quarantine_release_is_taken_back.py`.

How to release a quarantined file (the "Quarantine" panel of the MouseReach window):
1. Click Refresh. Each held file is listed with the reason it was held.
2. In "Corrected name", type the name the file should have (or leave it, if the name is fine
   now). A single-animal video is named `YYYYMMDD_<animal>_<tray><run>.mp4`; a whole-tray
   recording `YYYYMMDD_<eight animals, comma-separated>_<tray><run>.mkv`.
3. Under "Release to", pick the single-animal intake for a one-mouse video or the collage intake
   for a whole-tray recording.
4. Click "Release selected". If the name is refused, the message says why; fix the name and try
   again. Once released, the watcher takes the file in on its next scan.

Not changed here (follow-ups): `mousereach-watch-quarantine --list` reads the quarantined rows
of the watcher database of the machine it runs on, not the quarantine folder. So it does not list
a file another machine quarantined (each machine keeps its own rows; the folder is shared), and
it still lists a file already released under a NEW name, because the row of the old name stays
'quarantined' (the new name is a new video to the watcher). The Quarantine panel lists the
quarantine folder itself and is the accurate view.

## Update: mousereach-watch-quarantine --release works

`mousereach-watch-quarantine --release FILE` imported a class that does not exist in
`watcher/validator.py`, so it printed an import error and exited 1 every time; it also had no way
to say which intake the file should go back to. It is now `--release FILE --to single|collage
[--as NEW_NAME]` and calls the same routine as the Quarantine tab
(`validator.release_from_quarantine`): the name is checked with the chosen intake's rules before
anything moves, nothing in the intake folder is overwritten, and a refusal exits 1 with the
reason. The usage is in docs/CLI_REFERENCE.md. Test: `tests/test_quarantine_release_cli.py`.

The Quarantine panel (its own panel in the MouseReach window, named "Quarantine") does the
same release with buttons and is the easier way. `--release` uses the same routine:

- `FILE` is the name of the file in the quarantine folder (as the Quarantine
  panel shows it, or as it appears in that folder). `--to` is required: it says which intake folder the file goes
  back to, and which naming rules the name must pass.
- The name (`--as NEW_NAME`, or the file's own name) is checked BEFORE anything
  moves. If the chosen intake cannot read it, nothing moves; the error says
  what is wrong and, where it can, suggests a corrected name. The old name is
  allowed -- it may be valid now.
- A file of that name already in the intake folder stops the release; nothing
  is overwritten.
- On success the file is in the intake folder, its `.quarantine.json` note is
  removed, and the watcher takes it in on its next scan. The command exits 0;
  any refusal exits 1.

Example: `mousereach-watch-quarantine --release 20250101_badname.mp4 --to single --as 20250101_ABC0101_P1.mp4`
