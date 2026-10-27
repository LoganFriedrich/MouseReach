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

The companion documents are `PIPELINE_PROCESS_AS_DESCRIBED.md`, which records
how the pipeline is *supposed* to work, and `PIPELINE_DESCRIBED_VS_BUILT.md`,
which lists requirement by requirement where the code does not meet it.

**This describes behaviour as of 2026-08-21.** If you change the pipeline,
change this file in the same commit.

---

## 1. Where videos come in, and what is watching

HOW VIDEOS ENTER THE SYSTEM, AND HOW MANY WATCHERS THERE ARE - as the code actually works today.

1. THE FRONT DOOR

There is exactly one folder the system watches for new eight-mouse collage videos. The code builds it as NAS_ROOT / "Unanalyzed" / "Multi-Animal" (src/mousereach/config.py:109). NAS_ROOT is not hard-coded; it comes from the `nas_root` key in each machine's ~/.mousereach/config.json (src/mousereach/config.py:96). On every machine in the shipped machine profiles that key is <nas_root>, so the real front door today is:

    <nas_root>\Unanalyzed\Multi-Animal

It currently holds 844 video files. Three things about this folder are easy to get wrong:

  - The scan is one level deep. The code lists that folder's direct contents with iterdir() (src/mousereach/watcher/state.py:70). A video placed in Unanalyzed itself, or in the sibling Unanalyzed\DREDDs or Unanalyzed\LAT folders, is never seen.
  - Only five file extensions count: .mkv, .avi, .mp4, .mov, .wmv (src/mousereach/watcher/state.py:27 and :75).
  - The filename is a hard gate. It must read YYYYMMDD_{eight comma-separated animal IDs}_{tray letter}{run number}.ext - for example 20250704_CNT0101,CNT0205,...,CNT0906_P1.mkv. The date must be a real date and not in the future, and the tray letter must be P, E or F (src/mousereach/watcher/validator.py:106 and :111-118). A file that fails any of these is moved to a quarantine folder and never processed (src/mousereach/watcher/state.py:112). The quarantine folder defaults to NAS_ROOT\Processing\Quarantine (src/mousereach/config.py:720-726).

What software records the videos is outside the code. Recording, naming, and copying the file into Unanalyzed\Multi-Animal are human steps the code simply assumes have happened. The only thing the code can know about recording is optional configuration: a GPU node can be given the name of the program that records (watcher.pause_while_running), and then pauses itself while that program is open (see the update of 2026-09-15, "the watcher pauses itself while a recording program runs", at the end).

There is one other, deliberate way in: a one-off command copies archived ASPA collages into that same folder, re-encoding their names into the required format first (src/mousereach/aspa/import_collages.py:84, :229). It is run by hand, not on a timer.

Once a collage has been cropped it is left in place; much later, a separate slow-cadence job on the processing server moves a collage out to Analyzed\Multi-Animal, but only once every single-mouse video cut from it has finished the whole pipeline cleanly (src/mousereach/watcher/orchestrator.py:1404-1420).

2. HOW MANY WATCHERS - THE HONEST COUNT

There is ONE pipeline watcher PROGRAM with TWO ROLES, plus TWO other unrelated watcher daemons.

The pipeline watcher is started by the command `mousereach-watch` (pyproject.toml:178). On startup it reads the `watcher.mode` setting from ~/.mousereach/config.json and picks one of two behaviours (src/mousereach/watcher/cli.py:292-302):

  - mode = "processing_server"  -> ProcessingOrchestrator (src/mousereach/watcher/orchestrator.py:1315)
  - anything else, including a missing value, which defaults to "dlc_pc" (src/mousereach/config.py:676) -> DLCOrchestrator (src/mousereach/watcher/orchestrator.py:356)

The two roles watch DIFFERENT folders, which is the cleanest way to tell them apart:

  - The graphics-card role watches the collage front door, Unanalyzed\Multi-Animal, plus Unanalyzed\Single_Animal (src/mousereach/watcher/watcher.py:62-70). It crops collages, runs DeepLabCut, and hands the results off.
  - The processing-server role never looks at the front door at all. It watches only NAS_ROOT\Processing\Posed, the folder where posed videos are staged (src/mousereach/config.py:128, src/mousereach/watcher/orchestrator.py:1364-1371). It never crops a collage and never runs DeepLabCut - its work queue only contains archive, intake, pipeline and reprocess items (src/mousereach/watcher/orchestrator.py:1432-1531).

The clean two-role split has an important exception. A second setting, `watcher.also_process` (src/mousereach/config.py:678), when true, makes the graphics-card role skip the handoff entirely: it runs the analysis algorithms locally and archives the finished results itself (src/mousereach/watcher/orchestrator.py:513, :528, :548). All three graphics-card machines in the shipped machine profiles have also_process set to true (src/mousereach/setup/lab_profiles.json, profiles "NAS / DLC PC", "Vid&DLC1PC", "Vid&DLC2PC"). On those machines, ONE watcher does the entire job end to end and the processing server never sees those videos.

Two further watcher daemons ship in the same package and are genuinely separate programs:

  - `mousereach-backup` (pyproject.toml:197) runs a timed copy of Y: to the X: backup drive using robocopy (src/mousereach/watcher/backup.py:25). It only starts if a `backup` section is enabled in the config file (src/mousereach/watcher/backup.py:196). The processing-server machine profile enables it.
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

  - nas_root - the shared pipeline root. Everything above (the front door, the staging folder, the review queues, the final Analyzed folder) hangs off it. If it is unset, every one of those paths is unset too and each tool says "nas_root is not configured" and names mousereach-setup (since 2026-10-22; before that the code guessed an old layout, NAS_DRIVE\"! DLC Output" -- see the update of that date at the end).
  - processing_root - the local fast drive where work actually happens (A: on the graphics-card machines, C: on the server).
  - watcher.mode - "dlc_pc" (the default when absent) or "processing_server". Chooses which of the two roles the one watcher program plays.
  - watcher.also_process - true means a graphics-card machine runs the whole pipeline itself instead of handing off; false means it stops after DeepLabCut and stages the result for the server.
  - watcher.dlc_gpu_device - which graphics card number DeepLabCut uses, default 0.
  - watcher.pause_while_running - programs that pause this watcher while they run (process names such as recorder.exe). Empty by default, which means the watcher never checks for any program. A running watcher follows edits to it without a restart.
  - watcher.pause_resume_grace_seconds - how long every listed program must have been closed before work starts again, default 120.
  - watcher.recording_days / watcher.recording_hours - optional, e.g. "Mon-Fri" and "08:00-18:00" on the machine's clock: when recording can happen at all. Outside them an open listed program does not pause the watcher. Empty = any day / any hour (since 2026-10-22).
  - backup.enabled - whether the separate Y:-to-X: backup daemon will start.

---

## 2. Splitting a collage into one video per mouse

HOW COLLAGES ARE SPLIT AND NAMED - AS THE CODE ACTUALLY DOES IT TODAY

Where the work happens

Two different watcher programs exist. Each machine runs exactly one of them, chosen by a single setting in that machine's own configuration file at ~/.mousereach/config.json, under the "watcher" section, key "mode" (config.py:676). If mode is set to "processing_server", the machine runs ProcessingOrchestrator. Any other value - and the built-in default is "dlc_pc" - makes it run DLCOrchestrator (watcher/cli.py:298-303). All collage splitting lives in DLCOrchestrator; ProcessingOrchestrator contains no cropping code at all. A second setting, watcher.also_process (config.py:678), decides whether that same machine ALSO runs the behaviour algorithms locally after pose estimation; it has no effect on cropping.

Where collages are found

The watcher polls a folder on the shared network storage. The folder is not called "unprocessed" - the code builds it as NAS_ROOT/Unanalyzed/Multi-Animal (config.py:109), where NAS_ROOT comes from the machine's config key "nas_root", falling back to an older drive layout if that key is missing (config.py:99-105). Any file there with extension .mkv, .avi, .mp4, .mov or .wmv is treated as a candidate collage (watcher/state.py:27, 75).

Name checking happens before anything else

Every newly seen collage has its file name validated (watcher/validator.py:103-229). The name must be DATE_ID1,ID2,ID3,ID4,ID5,ID6,ID7,ID8_TRAYRUN.ext, for example 20250704_CNT0101,CNT0205,CNT0305,CNT0306,CNT0102,CNT0605,CNT0309,CNT0906_P1.mkv. The date must be eight digits, be a real date, and not be in the future. There must be exactly eight comma-separated mouse identifiers, each one letters followed by at least four digits. The trailing label must be one of the letters P, E or F followed by digits (P for the pillar tray, E for easy, F for flat). At least one of the eight positions must be a real mouse. A collage that fails any of these is physically moved into a quarantine folder alongside a small text file recording why, and is marked quarantined in the watcher's own bookkeeping database (watcher/state.py:110-124, watcher/validator.py:408-450). A collage that passes is recorded, then re-checked on later polls until its file size has stopped changing, at which point it is marked ready to crop (watcher/watcher.py:76-82).

When cropping actually runs

Cropping is the LOWEST-priority job the graphics-card machine does. On each cycle it first stages already-finished work back to the network, then runs pose estimation on any single-mouse video waiting for it, and only if both of those queues are empty does it pick up a new collage (watcher/orchestrator.py:530-580). Before starting, it claims the collage in a shared database (the pipeline_collages table of watcher_central.db on the shared root) so two machines cannot crop the same one. Since 2026-09-15 this fails closed: a node that cannot obtain a claim does not crop, and a collage whose children are all already finished or held for review is not cropped at all. The update section of 2026-09-15 at the end of this document has the details.

The cropping step itself

The collage is copied - copied, not moved - from the network folder to a scratch folder on the machine's own fast disk, at PROCESSING_ROOT/watcher_working (watcher/orchestrator.py:146, 741-745). The original stays on the network untouched; nothing in the watcher ever moves or deletes it. (A separate archive_collages function exists that would move collages away, but it is only reachable from the hand-run command-line tool, never from the watcher - video_prep/core/cropper.py:264, video_prep/cli.py:121.)

crop_collage then does the split (video_prep/core/cropper.py:105-184). It reads the eight mouse identifiers out of the file name and pairs them, strictly in order, with eight fixed rectangles: four across the top row and four across the bottom, each 480 pixels wide and 540 tall, which assumes the collage is exactly 1920 by 1080 (video_prep/core/cropper.py:39-48, 129). Nothing checks the real resolution of the incoming file. For each occupied position it shells out to ffmpeg with a crop filter and writes an .mp4 (video_prep/core/cropper.py:149-158). Video is re-encoded (only audio is copied through). Failures are recorded per position and the other positions continue.

How the single-mouse files get their names

The output name is DATE_MOUSEID_TRAYLABEL.mp4 - for instance 20250704_CNT0101_P1.mp4 (video_prep/core/cropper.py:145). The date and the tray label are lifted straight from the collage's file name; the mouse identifier is the one whose turn it is in the comma-separated list, and its turn is set by which grid cell is being cut. So the identity of every single-mouse video comes from two things and two things only: the order the human typed the eight names into the collage's file name, and the fixed reading-order of the grid. No text is read off the image, and nothing cross-checks that the name in slot three really is the mouse filmed by camera three.

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

The cast. Two watcher programs exist, and which one a machine runs is decided by a single setting in that machine's own settings file at ~/.mousereach/config.json: watcher.mode. Set to "dlc_pc" the machine runs the DLCOrchestrator (the crop-and-pose watcher); set to "processing_server" it runs the ProcessingOrchestrator (the analysis watcher). This section covers the crop-and-pose watcher only. Three lab machines with graphics cards run it (DLCLabPC, Vid&DLC1PC, Vid&DLC2PC); the analysis server does not.

Two roots. Every path in the system hangs off two settings. nas_root is the shared network folder everyone can see (<nas_root> in this lab). processing_root is the machine's own fast local disk (A:\MouseReach_Pipeline on the GPU machines, <nas_root> on the analysis server). Folders under nas_root are shared; folders under processing_root are private to one machine.

Step 1 - noticing a collage. Every thirty seconds the watcher lists the shared folder Unanalyzed\Multi-Animal. Any new video file whose name parses correctly is recorded in the watcher's database. It is then watched until its size stops changing for a configured number of seconds (watcher.stability_wait_seconds), which is how the system avoids grabbing a recording that is still being written. Only then does it become eligible for cropping.

Step 2 - choosing what to do next. The watcher does exactly one job per cycle, chosen by a fixed priority order: finish videos whose pose is already done, then pose a video that is waiting, and only when nothing at all is in flight does it start cropping the next collage. So cropping is deliberately the lowest priority - the machine drains what it has started before opening a new collage.

Step 3 - cropping. Before touching the file the machine claims the collage in the shared coordination database (the pipeline_collages table of watcher_central.db on the shared root), so two GPU machines watching the same folder cannot crop the same collage twice. No claim means no crop (since 2026-09-15). This covers a database that cannot be reached, a claim check that errors, and a claim another machine holds. A claim left unfinished for a day can be taken over, and a failed crop gives its claim back; the 2026-09-15 update section at the end has the rules and the reasons. It then COPIES the collage from the network into a scratch folder on its own disk (processing_root\watcher_working) and verifies the copy by size. The collage on the network is not moved, renamed or deleted. The scratch copy is cut with ffmpeg into eight fixed rectangles - the recording is a 1920x1080 grid of eight camera views, two rows of four, each cell 480x540. Grid positions whose animal identifier has cohort "00" mean "no mouse here" and are skipped. Each surviving cell is written as its own mp4 named {date}_{animal}_{tray}.mp4, into the same scratch folder.

Step 4 - where the cropped singles go. Each single is registered in the watcher's database and then COPIED (with a size check) into a second local folder: processing_root\DLC_Queue - on the lab GPU machines, A:\MouseReach_Pipeline\DLC_Queue. The scratch copies, and the scratch copy of the collage, are then deleted. Note what this means: the cropped singles do NOT go to any folder called Processing. The settings file does define a shared network folder Unanalyzed\Single_Animal, and the hand-run command `mousereach-crop` writes there by default, but the automatic path never uses it. Between cropping and the end of pose estimation, each single exists in exactly one place - one machine's local disk - and is invisible to the rest of the lab. If a single had already been cropped and taken past this point in an earlier pass, it is left alone rather than reset; the two exceptions the code deliberately re-drives are children recorded as "failed" and children recorded as merely "validated", the latter because that state means the copy into DLC_Queue never succeeded and no file exists anywhere.

Since 2026-09-15 each child is also checked against the shared drive before it is queued. An archive manifest, or a bundle in a review queue, means the child is recorded as finished or held rather than posed again; a new or rebuilt node database cannot know that on its own.

A gap worth recording: the cropper is meant to leave a small provenance file beside the collage listing which child came from which grid position, but in the automatic path it is written beside the temporary copy in the scratch folder, so it never appears next to the collage on the network. Nothing downstream breaks, because the later retirement sweep falls back to reconstructing the child list from the collage filename.

Step 5 - pose estimation. When a single is sitting in DLC_Queue the watcher runs DeepLabCut on it, one video at a time, and tells DeepLabCut to write its output into that same DLC_Queue folder. The pose files (.h5, plus .csv) therefore land beside the video in DLC_Queue, not in a Processing folder. Which network is used is set in two places, neither of them in the code:
  - watcher.dlc_config_path in the machine's settings file names the DeepLabCut project (A:\AIs\MPSA-LF-2025-10-27\config.yaml on all three lab machines). If it is unset or missing, nothing is posed and the video simply stays queued - it is not marked failed.
  - Which trained network inside that project - DeepLabCut calls these "shuffles" - is resolved in order from: an explicit argument, then watcher.dlc_shuffle in the machine's settings file, then the shuffle number parsed out of the dlc_scorer entry in the shared file pipeline_versions.json at the top of nas_root. No lab machine sets dlc_shuffle, and that shared file currently declares DLC_resnet101_MPSAOct27shuffle3_100000, so shuffle 3 is what runs. In this project shuffle 1 is the older resnet50 network (Model 3.1) and shuffle 3 is the resnet101 network (Model 4.0). If none of the three sources names a shuffle, the code refuses to pose at all rather than accept DeepLabCut's silent default of shuffle 1, and it stops posing on that machine instead of failing the video, because an unresolvable model is a machine problem and not a video problem. After each video the code reads the model name out of the pose filename DeepLabCut just wrote and rejects the result if it is not the declared one, so a mismatched model cannot be handed downstream. Both safeguards exist because the watcher once quietly produced Model 3.1 pose while the rest of the pipeline was calibrated for 4.0.
The watcher also re-checks the queue folder each cycle for pose files that appeared without its noticing - for instance after a crash mid-run - and records those videos as posed.

That only rescues a video whose pose file made it to disk. A watcher stopped DURING the pose left the video in 'dlc_running', which no job list selects, so it stopped moving for good - and silently, because the queue looked busy rather than broken. The same held for a video stopped mid-archive ('archiving') and a collage stopped mid-crop ('cropping'). Since 2026-09-13 every watcher, before it takes on anything new, puts such work back where it came from: 'dlc_running' returns to the pose queue, 'archiving' returns to 'processed' to be filed again, 'cropping' returns to 'stable' to be cut again, and each is named in the log as it happens (`BaseOrchestrator._reclaim_orphaned_work`). This is safe rather than a guess about a live run because a named global mutex allows exactly one watcher per machine, so anything still sitting in one of those states at startup was left by a process that is already gone. A video stopped mid-ANALYSIS needs no rescue and gets none: both watchers already select 'processing', and the pipeline reuses whatever stage outputs already exist, so it resumes rather than restarts. `mousereach-watch-status` names anything caught this way, so it is visible even while the watcher is down.

Step 6 - what happens after pose, and the setting that decides it. watcher.also_process in the machine's settings file controls this:
  - also_process = false: the machine is finished. Every file belonging to that video - the mp4, the .h5, the .csv - is MOVED out of the local DLC_Queue onto the shared network folder Processing\Posed, and the video is recorded as done-on-this-machine with both recorded paths (video and pose) pointing at the staged copies. (Until 2026-09-12 only the video path was updated; the recorded pose path kept naming the local DLC_Queue file the move had just removed, so 339 of 458 archived rows on the lab GPU node pointed at a pose file that no longer existed.) Some other machine picks it up from there and runs the analysis steps. This is the arrangement the plain description assumes.
  - also_process = true: the machine keeps the work. It stages the pose and the video into its local Processing folder, runs segmentation, reach detection, outcome detection and reach assignment there, and then archives the results from that folder straight to the network under Analyzed\{project}\{cohort}, deleting its local copies in both Processing and DLC_Queue. (Until 2026-09-12 the pipeline had already moved to Processing but the archive step still read DLC_Queue, so it filed the mp4 and the raw pose with no manifest, segments, reaches or outcomes.) Nothing is ever staged to Processing\Posed.
All three lab GPU machines are configured with also_process = true, so today the crop-and-pose machines in fact run the entire pipeline, and the "hand off to the server" route is configured but idle. That is the largest single gap between the described design and the running system.

Step 7 - what eventually happens to the collage. It stays in Unanalyzed\Multi-Animal throughout everything above. It is removed only by the OTHER watcher - the analysis-server one - which runs a slow sweep roughly every thirty minutes over that folder. For each collage it works out the set of single-mouse children implied by the filename and checks whether every one of them has reached the final Analyzed output, was processed with the currently declared tool versions, and has no review outstanding. Only if all of that holds for every child does it stamp a completion record and MOVE the collage (and its provenance file, if one exists) to Analyzed\Multi-Animal, which the backup watcher already mirrors to the second storage array. It never deletes. So a collage still sitting in Unanalyzed means at least one of its children is unfinished, out of date, or waiting on a human.

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

   - watcher.also_process = TRUE: nothing is ever handed off. The first watcher runs the entire rest of the pipeline itself, on its own machine, using the same code the server would use: segmentation, reach detection, outcome detection, reach assignment, a provenance manifest, the quality-control triage step, the review gate that can divert a video into the human triage queue or the deep-review queue, kinematic feature extraction, a write into the shared connectome database, and finally an archive of the whole bundle into the final Analyzed folder organised by project and cohort (src/mousereach/watcher/orchestrator.py:968-1187 and 1188-1245). This is the configuration shipped for every graphics-card machine in the lab - the lab DLC PC and both behaviour-room PCs all have also_process set to true in src/mousereach/setup/lab_profiles.json:47, :68, :87.

So the statement "cropping and DeepLabCut are all the first watcher does" is not accurate in either configuration. With also_process off it also performs the handoff move; with also_process on it produces the final scientific product.

Two further things the first watcher does that are easy to miss. It USED TO inherit a periodic housekeeping scan from the shared base class, running at startup and then roughly every 30 minutes, which walks the archived videos in its own database, reads each one's manifest out of the final Analyzed folder, and re-labels as "outdated" any video whose recorded tool versions no longer match the declared current versions or whose human review file is newer than its archived kinematics (`ReprocessingScanner.scan`). On a first-watcher machine that had no follow-through: the first watcher's job list has no entry for "outdated" videos, so a video re-labelled this way left the "archived" state and nothing on that machine ever picked it up again. Worse, the state was not inert - it synced to connectome.db, other nodes adopted it during startup recovery, and it came back as a pathless row on a machine that had no file for the video. As of 2026-08-24 the scan is gated on `handles_reprocessing`, which is true only for the second watcher, so the first watcher no longer marks anything outdated. Only the second watcher acts on "outdated" (`ProcessingOrchestrator._get_next_work_item`). Separately, single-mouse videos found sitting in the network Unanalyzed/Single_Animal folder are registered in the database at state "validated" (`WatcherStateManager.discover_new_singles`). Until 2026-09-13 "validated" was not one of the states the first watcher's job list selected from, so those videos were recorded and then never processed; the job list now has a branch for them. Since 2026-09-15 a single is registered only once it has stopped changing. The branch first claims it by renaming it into Single_Animal/.inflight/<machine>/, then copies it onto the node and queues it for pose. See "Update 2026-09-15: videos can be dropped into either Unanalyzed folder at any time" at the end of this document.

WHAT THE SECOND WATCHER REQUIRES BEFORE IT WILL TOUCH A VIDEO

The second watcher runs on the server. It has exactly one source of new work: the network handoff folder Processing/Posed. Each cycle it globs that folder for DeepLabCut output files, and for each one it requires a matching video file with the same name to also be present in that same folder before it will register the video at all. If the video's filename does not parse, the video is quarantined instead (src/mousereach/watcher/orchestrator.py:1363-1372; src/mousereach/watcher/state.py:212-300). It does not scan the network Unanalyzed/Single_Animal folder, and it does not walk the final Analyzed folder looking for work.

Before copying a registered video in, it takes a claim: it creates a marker file named after the video, containing its own hostname, in a hidden .claims subfolder of the handoff folder. The create is exclusive (watcher/claim_marker.py), so when several machines race exactly one succeeds; a machine that loses leaves the video out of its choices for 10 minutes and takes the next one in the work order. The holder keeps the claim while the video's files are in the handoff folder -- through analysis, review and failure -- and lets go after archiving has removed them. A sweep every 10 minutes makes the claims match what the machine holds: it claims any video in the handoff folder that it has in hand (taken in and not given back -- in analysis, review, failed or filed) and that has no claim, and lets go of a video that has left the folder or is waiting again (a lifted review hold goes back as 'dlc_complete' for a fresh run, which any machine may take). Another machine's claim is broken only when that machine has stopped writing its status report for 3 hours by the file server's clock, never for its age (`ProcessingOrchestrator._claim_video`, `_sweep_posed_claims`; changed 2026-10-22 from write, wait half a second, read back, and delete after 24 hours). This claim exists only to stop two PROCESSING machines watching the same handoff folder from grabbing the same video. It is not a check against the first watcher: the first watcher never reads or writes those markers, and the second watcher never reads the first watcher's own collage-level claims, which live in the pipeline_collages table of watcher_central.db on the shared root and are only created by the first watcher (src/mousereach/watcher/orchestrator.py:721-727 versus the second watcher's constructor at :1326-1361, which never builds a coordinator).

There is therefore no explicit "is the other watcher still working on this?" check anywhere. What keeps the two apart is the shape of the handoff: the first watcher holds the files on its own local disk until DeepLabCut has finished, and only then moves them into the shared folder. Two caveats. First, the move is implemented as copy-to-the-final-name then delete the source (src/mousereach/watcher/transfer.py:108-130), and discovery only tests that the names exist - there is no wait for the files to stop growing. A partially copied video is visible under its finished name. The subsequent intake copy compares source and destination sizes and rejects a mismatch (src/mousereach/watcher/transfer.py:85-95), which usually turns such a race into a failed intake rather than a truncated file being analysed, but that is an after-the-fact guard, not a wait. Second, on a machine with also_process = true nothing is ever staged, so no contention arises there at all.

Once a video is claimed, its files are copied - copied, not moved, so the handoff folder keeps a copy - into a folder named Processing on the SERVER'S OWN LOCAL DISK (on this machine, <nas_root>\Processing; the constant is built from the processing_root setting, src/mousereach/config.py:152). All algorithm work happens there. This is a different place from the network Processing zone on Y:, despite the identical folder name.

What the second watcher does NOT check before running the algorithms: it does not verify that the pose file was produced by the currently declared DeepLabCut model. There is no version comparison anywhere in the run-the-algorithms path (src/mousereach/watcher/orchestrator.py:1811-2120). The only thing resembling a version check at this point is tie-breaking: if a video has more than one pose file, the code prefers the one from the declared model, and otherwise takes the newest and logs a warning that the video should be re-posed - but it still proceeds (src/mousereach/pipeline/manifest.py:89-134). Version currency is checked later and after the fact, by the staleness scanner walking already-archived videos in the Analyzed folder.

The second watcher's full job list, in priority order, is: archive a finished video to the network drive (first since 2026-08-30, so results reach the share continuously); take in a newly discovered video from the handoff folder; run the algorithms on a video sitting in its local Processing folder; and re-run an "outdated" video (`ProcessingOrchestrator._select_work_item`). Within each kind of job, which video goes next follows the lab's work order (docs/WORK_PRIORITY.md; since 2026-10-22 including session lists and "not until nothing else is waiting" work, which comes last everywhere). For that last case, if the staleness scanner decided the video genuinely needs a NEW pose, the server cannot help - it has no suitable graphics card. Until 2026-08-24 it pushed the video into the 'dlc_queued' state and left it, which went nowhere: the server's own job list never selects that state, the DLC machine's queue folder is local and invisible to the server, and the only thing that crossed machines was the state itself, through connectome.db, where it became a pathless row that crashed the DLC machine's stager. It now holds those videos in 'outdated', names them once in the log, and lets the videos that CAN be re-run here through - so one un-poseable video no longer blocks the rest of the queue. Otherwise it pulls that video's pose file and video back out of the Analyzed folder into its local Processing folder and re-runs from the earliest stale stage (:1540-1605). Its local Processing folder is also refilled from a third direction: roughly every ten poll cycles it scans the two human review queues and moves any bundle a human has finished clearing back into Processing so the pipeline re-runs it (:1388-1397; src/mousereach/watcher/review_return.py:118-260). The review queues are shared storage, so when more than one node runs this scan, every queue mutation (return, retire, divert) first takes a per-video claim marker in Review/.return_claims/ -- an exclusive create, so exactly one node's claim succeeds (watcher/claim_marker.py); broken only when its holder has stopped writing its status report for 3 hours by the file server's clock, never for its age; no claim is needed when no shared root is configured, and none is taken (the move is skipped) when the claim folder cannot be made (review_return.py, _claim_return; changed 2026-10-21 from write, wait, read back and a 2-hour age rule). Intake is throttled by watcher.max_local_pending (default 2 since 2026-10-21; it was 200): once that many videos are sitting in the local Processing folder, no new videos are taken in until one is analysed, archived or cleared (:1444-1465). WHY 2: intake is visited before the pipeline, so with 200 the server took videos in for hours before analysing any, and every video it held was one no other node could take.

---

## 5. Checking the pose file, and what happens to old results

HOW THIS PART OF THE PIPELINE ACTUALLY WORKS TODAY

Vocabulary, so nothing below is guesswork:
- "pose file" = the DeepLabCut output, a .h5 file recording where each tracked body part sits in every frame. Its filename embeds the model that produced it, e.g. 20250624_CNT0115_P2DLC_resnet101_MPSAOct27shuffle3_100000.h5. That embedded string is called the "scorer".
- "manifest" = a small text file written next to a video after it is processed, named <video id>_processing_manifest.json. It records which scorer and which algorithm versions were used.
- "the version declaration" = one file, pipeline_versions.json, at the root of the pipeline tree on the network drive. It names the scorer and the four algorithm versions that count as current right now.

WHICH PROGRAM DOES THIS
One program, mousereach.watcher, runs on every machine. What it does is decided by watcher.mode in that machine's ~/.mousereach/config.json (config.py:676): "processing_server" builds the ProcessingOrchestrator, anything else builds the DLCOrchestrator (watcher/cli.py:298-302). A second setting, watcher.also_process (config.py:678), lets a graphics-card machine run the analysis algorithms itself instead of handing them off; when it is true the DLCOrchestrator runs a near-copy of the same pipeline code locally (orchestrator.py:968). So "the second watcher's job" is really "the analysis half of the program", and it runs on the processing server always and on a graphics-card machine when watcher.also_process is true.

1. DOES A POSE FILE EXIST? — yes, checked, every time
Before the algorithms run, the code resolves the pose file through `resolve_pose_input` (moved 2026-08-24 from orchestrator.py to `watcher/locate.py`, which now holds every "is this claimed path a real file?" helper). It tries the path recorded in the watcher's database; if that is blank or is not a real file it globs the working folder for <video id>DLC*.h5. It deliberately asks "is this a file?" rather than "does this path exist?", because an empty string in Python turns into the current directory, which exists — that exact bug fed a directory to the algorithms as a pose file and dumped 723 videos into the human review queue on 2026-08-19 (the comment at orchestrator.py:48-89 records it). No real file means the video is marked failed with "DLC h5 not found" and nothing else runs (orchestrator.py:1837-1839; the graphics-card copy at 992-996). Two other doors have the same guard: the path that pulls a finished video back for re-running does not run without a pose either -- but since 2026-10-22 it no longer fails the video for it (`_no_pose_for_reprocess`): a pose found only in the video's review bundle puts the video in that queue's state, a video still in Analyzed is marked as needing a new pose (the GPU nodes are asked for one), and only a video with neither pose nor video file is marked failed, saying so, and the path that returns a human-cleared video from the review queue refuses to return it rather than re-run it blind (review_return.py:152-159). Note the check is only "a readable file is there" — it does not open the file or judge whether it is complete.

2. IS THE POSE CURRENT? — checked, but only for already-finished videos, and from the manifest, not the file
About every thirty minutes (every sixtieth poll cycle at the thirty-second default; the review-return scan runs on its own much shorter cadence, since it is a cheap queue listdir while this sweep walks the whole archive) the processing side runs a sweep called the ReprocessingScanner (orchestrator.py:1341, 1373-1382). It walks the videos its database lists as 'archived' — videos that already completed and were moved into the final Analyzed output tree (reprocessor.py:97) — and then re-checks the rows already marked 'outdated', un-marking any whose versions compare current again (the two-way door; without it a declaration accident could mark rows forever, which happened twice in August 2026). Rows carrying an explicit `mark_reason` (hand-marked targeted re-runs, e.g. of videos a compatible-version bugfix applies to) are never un-marked; scanner marks write no reason. For each it loads that video's manifest (reprocessor.py:106) and compares it to the version declaration (versions.py:138-195):
- scorer differs -> the video's ANALYSIS is stale, because segments, reaches and outcomes were computed from the older pose. Whether the POSE has to be made again is a separate question, and since 2026-08-24 the scanner asks it: it indexes the pose files in the Analyzed tree once and checks whether a pose from the declared scorer already exists for that video. If it does, the scope is 'segmentation' - every post-DLC stage re-runs against the pose that is already there, and no GPU is used. If it does not, the scope is 'full' and the video genuinely needs re-posing. On the Y: archive as of 2026-08-24 that split was 1,233 videos needing no GPU against 31 that do, so the check is worth roughly 288 GPU-hours;
- one of segmenter / reach detector / outcome detector / kinematic extractor differs -> stale from that stage onward, and the re-run reuses the still-current earlier outputs (reprocessor.py:35-42, orchestrator.py:1846-1853). Exception: a manifest version listed under `compatible_versions` for that stage in `pipeline_versions.json` does NOT stale the video (versions.py) -- this is how a bugfix bump whose output only changes for pathological videos (which get re-marked by hand) avoids marking the whole corpus outdated, the accident that happened twice in August 2026;
- a human review the archived kinematics have not APPLIED also triggers a re-run of kinematics alone, so the reviewer's corrections reach the results and the central database. "Applied" is decided by content identity since 2026-09-09: the manifest records which review the extractor applied (reviewed_at + sha256, stamped by record_kinematic_version), and the scanner compares that stamp against the resolved review. It used to be a file-mtime comparison, which mixed two clocks -- SMB-written reviews carry the NAS server's clock, archived features keep the local clock -- and a ~20-minute NAS clock skew made every fresh re-run look staler than the review it had just applied, reprocessing reviewed videos in a loop (2026-09-09). Legacy manifests without the stamp fall back to the review's in-file reviewed_at vs the manifest's created_at (same process clock), and only lastly to mtimes;
- a video whose archived segmentation is HUMAN-authored (segments boundary_source == "human") is exempt from segmenter-version staleness, on both the mark side and the un-mark door: a human's cuts are facts about the video, no segmenter release stales them, and the re-run preserves rather than re-cuts them -- so before this exemption the manifest kept recording the old segmenter version and the scanner re-marked such videos after every single re-run (one video was swept into the versioned Archive 53 times).
Videos found stale get the database state 'outdated' plus a 'reprocess_scope' (reprocessor.py:150). The same sweep is what "mousereach-version-check" prints, and "--mark" is the manual way to trigger it (watcher/cli.py:1231-1290).
Three things this does NOT do, which matter:
- It inspects the pose file only to answer "does a pose from the declared model already exist?", and only when the manifest names an older model (added 2026-08-24). It still never opens a pose file or judges its contents. If the manifest is absent the video is counted "no_manifest" and skipped. Where the manifest and the file on disk disagree, the manifest still wins on the question of what produced the current results - which is correct, because that is what it records. It is not corrected in place; the reprocess run rewrites it with the pose that actually ran, which is the only honest way for it to change.
- Videos on their way through for the first time are never checked. The pipeline function itself (orchestrator.py:1811) has no version comparison anywhere in it. A fresh video is analysed with whatever pose the graphics-card machine produced.
- If either side of a comparison is an empty string, that component is treated as current (versions.py:159, 167, 183). An unset "dlc_scorer" in pipeline_versions.json makes the pose-currency check a silent no-op with no warning.
Videos deliberately locked for a publication are put in the state 'crystallized', not 'archived', so the sweep never sees them (versions.py:242, reprocessor.py:164).
One related safeguard falls short of a check: when a video has pose files from two models side by side, select_pose_file prefers the one matching the declared current scorer; if none matches it takes the newest, logs a warning that the video should be re-posed, and proceeds anyway (manifest.py:89-135).

Bench-sheet disagreement routing is NOT part of the watcher (it was, briefly, 2026-08-25 to 2026-08-28). Comparing pipeline outcomes with hand-scored sheets is an integrator's job -- a database tool that holds both -- and MouseReach must run with no such tool present. What MouseReach provides is the generic command `mousereach-route-to-queue` (watcher/route_cli.py): flag specific segments of an archived video and move it into the triage or deep-review queue with a recorded reason, through the same route_to_queue the pipeline uses itself. Whatever decides a video needs a person asks through that command; the watcher never reaches into another tool's environment or database. Likewise the central-database sync (sync/database.py) is configuration (`central_db`), absent by default, and cross-node coordination lives in MouseReach's own watcher_central.db on the NAS root.

3. WHAT HAPPENS TO THE OLD FILES - superseded, not overwritten (since 2026-08-21)
For a video marked outdated with a narrow scope, the second watcher copies every file whose name starts with the video identifier from Analyzed down to the local Processing folder — a copy, so the originals stay where they are (orchestrator.py:1574-1577). The algorithms re-run and overwrite the copies. Then the finished set is moved back to Analyzed/<project>/<cohort> by archive_video. That move used to be a bare shutil.move onto the same filename, which silently replaced whatever was there, destroying the previous generation's segments, reaches, outcomes, assignments, triage record, manifest and kinematics with nothing kept. Since commit 627877d (2026-08-21) archive_video calls supersede_video_outputs FIRST whenever anything for that video is already at the destination: the earlier generation is swept, checksum-verified, into the versioned Archive tree, and if that sweep reports any failure the archive REFUSES to proceed rather than move new files on top of old ones. The video itself and any ground-truth or human-review file are deliberately left in place, so a review still travels with its video. Since 2026-10-22 a file that is identical to the archived one (same size and time, or the same SHA-256 when the times differ) is not swept either: a re-run copies the video and pose down and hands them back unchanged, and sweeping them stored a duplicate pose per re-run and cost 60-100 s of copying.
The old pose file is the one exception, and only by accident: it is never moved or deleted. Because the model name is part of the filename, a new-model pose lands beside the old one and both stay.
For a video marked outdated because the pose model changed AND which has no pose from the declared model on disk (scope 'full'), the processing server does not do the copy-down. Before 2026-08-24 it forced the state to 'dlc_queued'; that reached no graphics-card machine and did reach cross-node recovery, where it produced pathless rows, so from then until 2026-09-12 it simply held the video in 'outdated' and logged the list once, and a person had to carry the video to a GPU and back. Since 2026-09-12 the ask travels over shared storage instead (src/mousereach/watcher/repose.py): right after each version scan the node writes one small request file per such video into Processing/Repose_Queue/; any graphics-card watcher pulls a couple at a time (claiming each by an atomic rename into Repose_Queue/.inflight/, copying the archived mp4 into its own local DLC_Queue, and queueing it for pose by the ordinary transitions, archived -> outdated -> dlc_queued); when the new pose lands in Processing/Posed the processing node ADOPTS it -- the row stays 'outdated' but its scope narrows to 'segmentation' and its recorded pose path points at the staged file -- and the copy-down route below re-runs every post-DLC stage against it, with the archived results folder (human segmentation and reviews included) copied down beside it. The request file is the idempotency key: it exists, queued or inflight, from publish until adoption (or until a GPU node that also processes archives the result itself), the GPU node heartbeats it while it holds the video, and a request nobody has touched for a day returns to the queue. Nothing is pushed to any machine; a lab with one GPU PC and a shared drive publishes and consumes on its own drive. Videos whose manifest names an old model but which DO have the declared pose on disk are not in this category at all: they get scope 'segmentation' and take the ordinary copy-down route below.
Be careful with the word "archive" in this code: the watcher's archive step means "move the finished outputs from the working folder into Analyzed/<project>/<cohort>", which is the live current-results location, not a historical store.

4. THE ARCHIVING CODE THE DESCRIPTION DESCRIBES DOES EXIST - and the watcher now calls it (since 2026-08-21)
mousereach/archive/supersede.py does exactly what the description says should happen. supersede_video_outputs (supersede.py:161) moves a video's about-to-be-replaced pose and algorithm outputs into a separate Archive tree at <pipeline root>/Analyzed/Archive, organised as "DLC Model <generation>/<seg…_reach…_out…_asn… version stack>/" so one pose is stored once per model generation while many algorithm variants live under it. Every move is checksum-verified: copy, compare sha256, and only then delete the source (supersede.py:105-128). It never overwrites — an identical file already there (under the base name OR any existing .N version, checked since 2026-09-09) means the incoming copy is simply dropped, and a genuinely different same-named file is saved as .1, .2 and so on. Before the .N identity check, a reprocess loop that re-swept unchanged content added another byte-identical multi-MB copy per lap. The video itself, the ground-truth file and the human review file are explicitly never archived (supersede.py:55-57, 204-206), matching the rule that human judgements are facts about the video and stay with it.
It now has two callers: archive_video (archive/core.py), which is the watcher's own filing step, and pipeline/reprocess_to_current.py:293-294, a hand-run bring-one-video-up-to-the-current-stack tool with no command-line entry point. The sentence that stood here -- that nothing automatic called it -- was true until 2026-08-21 and is not now. It has to be invoked by hand, and only does the archiving when called with finalize=True and only for videos the review gate judged clean; held videos are staged into a review queue instead and their old outputs are left untouched (reprocess_to_current.py:66-108, 280-300).

NET EFFECT
Existence of the pose file is properly guarded on every route into the algorithms. Currency of the pose is checked only as a background sweep over already-finished videos, from a text record rather than the file, and can be inert if the version declaration leaves the scorer blank. When a re-run happens inside the watcher, the previous generation's results are now swept into the versioned Archive tree first, checksum-verified, and the archive refuses to proceed if that sweep fails -- so what an earlier model produced for a video remains recoverable. That was not true before 2026-08-21, when the previous generation was overwritten in place.

---

## 6. Reprocessing, and what happens to human review files

HOW SENDING AN OUTDATED VIDEO BACK FOR REPROCESSING ACTUALLY WORKS TODAY\n\nVocabulary first, because the folder names in the code are not the words people use.\n- \"Processing\" in the reprocessing code means the machine's OWN local scratch folder: <processing_root>/Processing, where processing_root comes from that machine's ~/.mousereach/config.json (config.py:56-57, 152). There is also a shared network folder called Processing that holds the post-pose staging area (Processing/Posed), the failures folder and the two human-review queues; cropped single-mouse videos wait in Unanalyzed/Single_Animal (config.py:116, 128, 164-166). These are different places with the same name.\n- \"Analyzed\" is the finished-work tree on the network, organised by project and cohort (config.py:123). In practice it holds two kinds of folder side by side: cohort folders such as Analyzed/Connectome/CNT03/ containing the video, its algorithm outputs and any human files, and a pose-only tree Analyzed/Connectome/DLC Model 4/<cohort>/ containing the current-model pose files from a bulk pose job.\n- Each machine keeps its own small SQLite bookkeeping database at <processing_root>/watcher.db (db.py:116-119), with a summary exported to a shared one on the network after each archive (orchestrator.py:2146).\n\nSTEP 1 - DECIDING A VIDEO IS OUT OF DATE\nA scanner walks every video the local database calls 'archived' and reads that video's provenance file, <video>_processing_manifest.json, out of the Analyzed tree (reprocessor.py:100-113, 193-217). It compares that file against the declared current versions in MouseReach_Pipeline/pipeline_versions.json (versions.py:138-200). A video is out of date if the pose model differs, if any tracked stage's recorded version differs, or if a stage recorded no version at all. A video that is fully current is ALSO pulled back if a saved human review is newer than its kinematics output, so a reviewer's answer reaches the final data (reprocessor.py:114-136, 175-191).\n\nThe scanner writes no pipeline data (its only file is a cache of what it has read, version_scan_cache.json in the machine's own processing root; see the update of 2026-10-22 on keeping it across restarts). It sets the database state to 'outdated' and stores a \"scope\": 'full' when the pose model changed, otherwise the earliest stale stage - segmentation, reach, outcome or kinematics - so that stage and everything after it re-run while current upstream results are reused (reprocessor.py:26-42, 149-157). This scan runs at watcher start-up, then on a timer (orchestrator.py:216-241), and again every tenth poll cycle on the processing-server watcher (orchestrator.py:1378-1387). It can also be run by hand with `mousereach-version-check --mark` (cli.py:1275-1285).\n\nSTEP 2 - WHAT HAPPENS TO AN OUTDATED VIDEO\nThe processing-server watcher picks up outdated videos as its fourth priority, behind archiving, intake, and running the pipeline (order changed 2026-08-30: archiving moved from last to first so finished results reach the NAS continuously instead of only when the staged supply runs dry).\n\nIf the scope is 'full', nothing is copied or moved and the video stays 'outdated'. Since 2026-08-24 'full' means only "this video has no pose from the declared model anywhere in the archive" - a manifest naming an old model is not enough on its own, because the bulk re-pose had already produced current pose files for most of them. The row is no longer forced to 'dlc_queued': that reached no graphics-card machine (the DLC queue folder is local to each node) and did reach cross-node recovery, where the state became a pathless row that crashed the stager. Videos in this state are listed in the log once, and since 2026-09-12 a re-pose request for each is published to the shared Processing/Repose_Queue folder after every version scan, for any graphics-card watcher to pull (section 5 describes the round trip; watcher/repose.py implements it). When the new pose is staged back, the row's scope narrows to 'segmentation' and the ordinary copy-down re-run below takes over. On a DLC machine the pose handler now resolves the video file rather than trusting the recorded path, and a video with no file on that node is recorded 'unresolvable' instead of failed-and-retried.\n\nIf the scope is anything else, the watcher picks the pose to re-run against -- since 2026-09-12 a pose the row already points at (a freshly staged one from the declared model, adopted by the re-pose round trip) first, then a declared-model pose in Processing\Posed, and only then the video's results folder and the same cohort folder of each DLC Model tree, with a search of the whole Analyzed tree only when no declared-model pose is there (since 2026-10-22; the walk took minutes per re-run) (preferring the declared current model, else the newest - manifest.py:90-124) -- and copies two folders' worth of files into the local Processing folder, verifying every copy: from the pose file's own folder only the DLC files, and from the video's results folder (Analyzed/<project>/<cohort>) everything whose name begins with the video id, the video, the previous outputs and any human files included (orchestrator.py _reprocess_copy_set). Nothing is deleted from Analyzed - this is a copy, so the archived set stays intact until a re-archive overwrites it. The state is forced to 'processing' and the standard algorithm run starts immediately, skipping the stages upstream of the stale one (orchestrator.py:1581-1602, 1843-1853).\n\nBecause the declared current pose model in this pipeline is the resnet101/shuffle3 model and those pose files live in the separate pose-only tree, the pose usually comes from one folder and everything else from the video's results folder; that two-folder copy set (added after a pose-folder-only copy re-segmented away a reviewer's hand-set cuts on 2026-09-08) is what puts the video, the older pose and any human files beside the new pose in Processing.\n\nA second, entirely separate route exists for bringing videos current by hand: pipeline/reprocess_to_current.py. It never uses Processing at all - it copies the pose into a temporary folder, runs the algorithms there, and on request moves the results back beside the video, first moving the previous generation's outputs into a version-stamped Archive tree (reprocess_to_current.py:156-160, 285-321).\n\nSTEP 3 - HUMAN FILES\nThere are two human file types, not three. Ground truth is <video>_unified_ground_truth.json. Both the causal review tool and the fast triage review tool write the same file, <video>_causal_review.json - a segment counts as resolved when that file's record for it says it was reviewed (triage_status.py:16-18). The separate <video>_triage.json is the algorithm's own quality verdict, not a human file.\n\nNothing in any reprocessing path deletes either human file -- but until 2026-08-24 that sentence was true and still misleading. Nothing had to delete a review to lose it. The review was written into the triage bundle, the bundle is transient, and returning a cleared bundle MOVES its files onto one node's LOCAL disk and removes the directory. Shared storage then held no copy until the video was archived. Measured that day across 1,686 reviewed videos: 662 had a durable copy, 983 existed only on one machine's local disk, and 41 existed only inside a Y: bundle, one reprocess from gone. Reviews are now written to {NAS}/review_records/reviews/ at save time, before the working copy, and that copy belongs to no bundle's lifecycle. The version-driven reprocess only copies. The archiving step moves every file starting with the video id from Processing into the cohort folder, so human files travel with everything else (archive/core.py:144-190). The one routine that deliberately clears out old outputs - the versioned \"supersede\" move used by the manual bring-current tool - carries an explicit never-touch list containing the ground-truth file, the causal-review file and the video itself, and records them as kept (supersede.py:49-57, 131-136, 197-215). The automatic watcher never calls that routine, which is a separate reason human files are safe there. The only deletions in this area are queue bookkeeping files when a cleared bundle leaves a review queue (review_return.py:36-37, 166-170), staging copies after archiving (orchestrator.py:1225-1231, 2151-2158), and a manual reset helper that removes only algorithm outputs (pipeline/core.py:311-333).\n\nDo the human files travel with the video? Only on one of the two return routes. When a person clears a video that was held in the triage or deep-review queue, every file in that bundle - review file, ground truth, clearance marker - is MOVED into the local Processing folder, minus the two bookkeeping files, and the video is set back to 'processing' (review_return.py:118-214). On the version-staleness route they travel too, because the copy set takes everything beginning with the video id from the results folder (only the DLC files from the pose folder).\n\nThat mostly does not matter, because the code goes looking for the review rather than relying on it having travelled. Both the gate that decides whether a video may proceed and the kinematics step search for <video>_causal_review.json in the working folder, in that video's triage-queue bundle, and in the folder holding the canonical video, taking the newest hit (causal_review_io.py:52-87; review_gate.py:104-117; orchestrator.py:2067-2073). Ground truth is looked up differently: through an index built by scanning only the improvement working area and the network Processing folder (causal_review_io.py:384-386, 419-429). The Analyzed tree is not scanned, so ground truth archived beside a video is invisible to the gate's \"fully human-certified\" shortcut unless a copy also exists under one of those two roots.\n\nSTEP 4 - THE RE-RUN ITSELF\nThe re-run regenerates the algorithm outputs from scratch; it does not preserve the human flags that were written into the old algorithm files. The human decision lives in the review file and is re-applied at the very end: the gate treats a segment as resolved if the review says so, and the kinematics step passes the review file to the feature extractor so the human's outcome and chosen causal reach replace the algorithm's, with the algorithm's originals kept alongside for provenance (review_gate.py:96-101; orchestrator.py:2067-2078; causal_review_io.py:298-345). If the video comes out clean it goes to 'processed', gets archived into Analyzed/<project>/<cohort>, and its kinematics are pushed into connectome.db by a per-video replace. If anything is still unresolved, the whole bundle is moved back out to the triage or deep-review queue and nothing reaches kinematics or the database (review_gate.py:191-222).\n\nTHE ONE THING THAT IS NOT TRUE YET -- CORRECTED 2026-08-24\nThis section used to say that a review is re-attached by segment number and that nothing reads \"segment_span\" back. That was true when it was written and is no longer. `index_review_by_segment` (causal_review_io.py) matches a review to the CURRENT segmentation on frame overlap, ignoring the numbers: it needs 50% overlap, requires the best candidate to beat the runner-up by 15%, and DROPS a review whose frames straddle two segments or match none, saying so in its notes rather than guessing. `resolve_truth_layers` threads the current segments into it, and the kinematics extractor calls that. So a re-cut no longer stamps a human judgement onto footage nobody looked at.\n\nWhat WAS still broken, and was fixed the same day: the resolver looked for a review only in the two NAS review queues and the caller's processing dir -- all three places a review can vanish from. A review that outlived its bundle but whose video was not archived yet was in none of them, so the extractor fell back to the algorithm's answer while the human's sat in the durable store unread. The durable store is now the lowest-priority layer in that stack (any live copy still wins, since a reviewer may have edited it). Pinned by a test that fails against the previous code with the human's outcome discarded.\n\nThe corpus index also now stores segment_span, the human's chosen causal reach and their answers, so a review whose file is lost is reconstructable; it previously held a file path plus two summary fields. A module that guards human decisions across a re-run does exist, and reports rather than silently drops decisions whose segment vanished (clear_guard.py:1-18), but it is wired only into the bundle re-staging tool (staging.py:340), not into the watcher pipeline. So today, a re-segmentation that renumbers segments can silently re-point an old review at different footage.\n\nCONFIGURATION THAT CHANGES ANY OF THIS\n- watcher.mode ('dlc_pc' or 'processing_server', config.py:676): selects which watcher class runs. Only the processing-server watcher acts on 'outdated' videos and returns cleared review bundles; only the DLC watcher consumes 'dlc_queued'.\n- watcher.also_process (config.py:678): on a pose machine, run the algorithms locally straight after posing and archive directly, instead of staging for another machine.\n- processing_root and nas_root in ~/.mousereach/config.json (config.py:56-57, 99-105): set the local scratch Processing folder and the network tree respectively. If nas_root is left unset every derived path - staging, Analyzed, the review queues - is unset too, and the tools say so (until 2026-10-22 the code silently fell back to an old layout instead).\n- watcher.dlc_shuffle (config.py:665-667): if unset, the pose model used is taken from the declared dlc_scorer in pipeline_versions.json, so producer and version checker cannot disagree.\n- MouseReach_Pipeline/pipeline_versions.json: the declared current versions. Editing it is what makes existing videos outdated on the next scan.

---

## 7. Running the algorithms, and what happens when one fails

HOW THE ALGORITHMS ACTUALLY GET RUN, AND WHAT HAPPENS WHEN ONE DOES NOT SUCCEED\n(code root: <repo>, branch master)\n\nWHICH MACHINE RUNS THE ALGORITHMS\nEach machine has a file ~/.mousereach/config.json with a \"watcher\" section. Two settings decide the behaviour here:\n- watcher.mode: \"dlc_pc\" (the GPU machines that crop collages and run DeepLabCut) or \"processing_server\".\n- watcher.also_process: only meaningful on a GPU machine. False means \"after posing the video, put it on the NAS in Processing/Posed and let the server take it\". True means \"after posing the video, run the whole analysis here as well\".\nThe lab's three GPU machine profiles all ship with also_process: true (src/mousereach/setup/lab_profiles.json:41, :65, :89); this processing server's config has mode \"processing_server\" and no also_process key, which defaults to false (src/mousereach/config.py:678). So both machine roles contain a full copy of the analysis sequence: ProcessingOrchestrator._run_pipeline (orchestrator.py:1811) on the server, and DLCOrchestrator._run_local_pipeline (orchestrator.py:968) on a GPU machine. The two are near-identical, step for step.\n\nTHE SEQUENCE, IN ORDER\nBefore anything runs, the code resolves the DeepLabCut pose file for real: it uses the recorded path only if it is an actual file, otherwise it searches the working directory for {video}DLC*.h5, and returns nothing rather than a placeholder. If there is no pose file the video is marked 'failed' and the run stops (orchestrator.py:1830-1841). This guard exists because an empty path once resolved to the current directory, which \"exists\", and 723 videos were pushed into the human review folder in two hours as a result.\n\n1. Segmentation - splits the video into per-pellet segments (orchestrator.py:1856).\n2. Reach detection (orchestrator.py:1909).\n3. Outcome detection - what happened to each pellet (orchestrator.py:1939). Skipped entirely if the filename says tray type E or F.\n4. Reach assignment - which reach caused that outcome (orchestrator.py:1973). Also skipped for tray types E and F. It runs before the gate on purpose, because the gate treats \"pellet was touched but no reach was credited\" as a question for a human.\n5. A provenance manifest recording which tool and model versions produced the outputs (orchestrator.py:1989).\n6. An automatic quality check across all the outputs, which returns either 'auto approved' or 'needs review' and stamps that verdict into each output file plus a {video}_triage.json record (orchestrator.py:2009).\n7. The gate (orchestrator.py:2047, implemented in src/mousereach/watcher/review_gate.py). Nothing reaches kinematics or the database until the gate says clean.\n8. Only if clean: kinematic feature extraction, applying any saved human review corrections, then a write into connectome.db (orchestrator.py:2057-2103). The video is then marked 'processed' and later archived.\n\nOn a re-run triggered by a tool version change, the code can start from the first stale stage and reuse the earlier outputs instead of recomputing them (orchestrator.py:1846-1854).\n\nWHAT THE GATE DECIDES (review_gate.py:68-101)\n- If ground truth marks the whole video as exhaustively scored by a human, it is clean regardless of anything the algorithms flagged.\n- Else, if segmentation is unusable, DEEP REVIEW.\n- Else, if the automatic quality check said 'needs review', DEEP REVIEW.\n- Else, if any segment is still flagged and unanswered, TRIAGE.\n- Else, clean.\nA segment counts as flagged when the outcome cascade marked it \"triaged\" or set flagged_for_review, or when the outcome says the pellet was touched (retrieved, displaced in the scoring area, displaced outside) but no reach in the assignment file is marked causal (src/mousereach/review/triage_status.py:70-95).\n\nWHERE HELD VIDEOS GO\nBoth queues live on the NAS so any machine and the review tools can see them (src/mousereach/config.py:164-166):\n- Triage queue: Processing/Review/triage\n- Deep review queue: Processing/Review/deep_review\nThe whole bundle is MOVED out of the local Processing directory into a folder named after the video: the mp4, the pose file, and every algorithm output, plus a {video}_routing.json audit record and a {video}_manifest.json that lets the review tool open the video in place (review_routing.py:73-149, review_gate.py:120-138). The database state becomes 'triage' or 'deep_review'; if that state change is rejected it is forced, because the files have already moved and the disk is the truth (review_gate.py:158-176).\n\nWHAT HAPPENS WHEN A STEP DOES NOT SUCCEED - THIS IS NOT THE SAME AS TRIAGE\n- Segmentation raises an exception, or the pose file cannot be opened at all: state 'failed'. Nothing moves. This is treated as an infrastructure problem, deliberately kept away from human reviewers (orchestrator.py:1881-1887, and the reasoning at orchestrator.py:93).\n- Segmentation runs but reports unusable boundaries: DEEP REVIEW (orchestrator.py:1892-1901).\n- Reach detection or outcome detection raises: state 'failed' and the error is re-raised; the outer dispatcher logs it and marks 'failed' again (orchestrator.py:1933-1937, :1965-1969, :1514-1538).\n- Reach assignment raises: a log warning, and the pipeline continues (orchestrator.py:1986). If no assignment file was written, the \"touched pellet with no causal reach\" check is skipped, so such a video can pass the gate as clean and be written to the database with no causal-reach attribution.\n- The automatic quality check raises: a warning; the verdict stays 'auto approved' and the video proceeds (orchestrator.py:2039).\n- Kinematic feature extraction raises: a warning only. The video is still marked 'processed' and will be archived - with no kinematics and nothing written to connectome.db (orchestrator.py:2105-2110).\n- The database write fails: a warning only (orchestrator.py:2102).\nNo watcher ever picks up a video in state 'failed' - none of the work queues query that state (orchestrator.py:1432-1512, :501-580). A person has to run mousereach-watch-reprocess, optionally with --all-failed, to reset them (src/mousereach/watcher/cli.py:540-640).\n\nGETTING A TRIAGED VIDEO BACK INTO THE PIPELINE\nThe review tool, opened on a bundle, saves the human's answers as {video}_causal_review.json inside that bundle folder (src/mousereach/review/causal_review_widget.py:880-885, :2843-2850). The {video}_triage.json file in the same folder is the automatic quality check's record and plays no part in release.\n\nA watcher in processing-server mode calls the return scan from its scan step every tenth polling cycle - roughly every five minutes at the default 30-second poll - and returns at most 10 bundles per scan so a deep queue does not starve actual processing (orchestrator.py:1388-1397, :1348, src/mousereach/watcher/review_return.py:222). A watcher in dlc_pc mode never calls it (its scan step, orchestrator.py:483-496, does only file discovery and DeepLabCut completion checks), so a GPU machine running with also_process true can put videos into triage but cannot take them out.\n\nFor each bundle in the triage queue the scan recomputes the picture from the bundle's own files and releases it only when there is at least one flagged segment, every flagged segment has an answer whose \"reviewed\" field is not false, and segmentation reads as sound OR a human's deep-review clear marker is in the bundle (since 2026-09-09 the scan honors {video}_deep_review_cleared.json exactly as the gate does: such a bundle is never diverted to deep review and releases once its segments are answered -- previously the gate's honor and the scan's raw seg_failed divert fought each other and a cleared video lapped the three states indefinitely). Before touching anything it makes sure the video has a row in this machine's watcher database, registering it if the review was cleared on another machine (review_return.py:134-146). It then locates the mp4 and pose file - in the bundle, then via the bundle manifest, then by searching the Analyzed archive - and refuses to return the bundle if no pose file can be found, so a cleared review is not spent on a run guaranteed to fail (review_return.py:40-95, :152-159). The data files are moved into the local Processing directory, the two bookkeeping files are deleted, the state is set to 'processing' (forced if the normal transition is rejected), and the now-empty bundle folder is removed (review_return.py:161-214).\n\nThe returned video then re-runs the whole sequence from segmentation. The human's review file came back with it, so the gate sees the answers and lets it through, and feature extraction loads that same file and applies the reviewer's corrections before computing kinematics (orchestrator.py:2066-2078). Deep-review bundles come back the same way, but the release signal is different: either a {video}_deep_review_cleared.json marker written by the deep tools, or a {video}_unified_ground_truth.json produced by the ground-truth tool (review_return.py:108-115, :252-259).\n\nTWO EDGES WORTH KNOWING\n- The gate treats a segment as resolved if ground truth determined it, or if a review file found in any of several locations answers it. The return scan is stricter: it only reads {video}_causal_review.json inside the bundle and does not consider ground truth. A video whose flagged segments include ground-truth-determined ones can therefore sit in the queue after the reviewer has answered everything actually asked of them (review_gate.py:91-100 vs review_return.py:244-249).\n- The server's scan step returns immediately if no DeepLabCut staging directory is configured (orchestrator.py:1364-1366), and that staging path only exists when nas_root is set. With nas_root missing, the review-return scan never runs at all.

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

The move collects every file in the source folder whose name begins with the video id - the .mp4, the DeepLabCut pose files, all algorithm .json files, the provenance manifest, the triage file, and any ground-truth or human-review files sitting alongside - and moves them as a group (archive/core.py:146-148, :183). The destination is Analyzed/{project}/{cohort}/, worked out from the animal id in the filename: CNT0304 goes to Analyzed/Connectome/CNT03/, ENCR0102 to Analyzed/Enhancer/ENCR01/, H01 to Analyzed/ASPA/H/ (archive/core.py:82-92, config.py:344-376). An unparseable id lands in Analyzed/UNKNOWN/UNKNOWN/.

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

A periodic job mirrors Behavior/MouseReach_Pipeline, Tissue/MouseBrain_Pipeline and Databases from the Y: drive to the X: drive using robocopy in add-only mode - newer files are copied over, deleted files are left in place on X: (watcher/backup.py:28-33, :133-150). That preserves a copy of the connectome.db file, but it is a whole-file disk backup on a timer, not a record of which rows a given sync replaced.

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
  about 10% of ordinary videos, and checked against the three videos a human had
  actually judged mis-segmented it caught one. Two of those three had textbook
  segmentation output, because they were offset rather than malformed, and
  nothing measurable inside a single video can see that. It never routed a
  video before being disabled. The established route remains a person noticing
  during review and pressing "Flag Session", which works and is in use.

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
crystallize, uncrystallize -- hardcoded the fallback and, on this machine,
silently read a database last written in February. Crystallize, the brake that
protects published videos from reprocessing, would have found no videos and
protected nothing. All eight sites now resolve through one helper that loads
the node config and prints the path it chose, so a wrong database is loud.
Verified: the commands now see 1,330 videos instead of February's 282.


---

## Update 2026-08-24: a row never claims a file the machine does not have

Three failures reported from the DLC node, one cause.

The staging step crashed with `TypeError: expected str, bytes or os.PathLike
object, not NoneType`. Forty-two videos sat in the node's database with no
`current_path` and no `dlc_output_path`, churning. Collages could not enter the
pipeline at all.

All three came from cross-node recovery. On startup a node reads
`pipeline_videos` and `pipeline_collages` out of connectome.db and copies the
states it finds into its own database. Those tables carry state and no usable
file path -- every one of the 2,899 rows in `pipeline_videos` had
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
scorer set scope 'full', which meant "put it back on a GPU". On the Y: archive
1,233 of the 1,264 such videos already had the declared shuffle3 pose sitting in
`Analyzed/Connectome/DLC Model 4/`; re-posing them at about 14 minutes each would
have spent roughly 288 GPU-hours regenerating files that exist. The scanner now
indexes the archive's pose files once per scan and downgrades those videos to
scope 'segmentation' -- every post-DLC stage re-runs against the pose already
there. The 31 with no current pose keep scope 'full'. Their manifests are not
edited: the manifest correctly records what produced the current results, and the
reprocess run rewrites it with the pose that actually ran.


---

## Update 2026-09-14: which watcher database is authoritative, and one that is not

A check of whether roughly 1,100 CNT videos still needed a new pose found that
they did not. Their manifests record the declared scorer and the current version
of every tracked stage, and the processing server's own live database holds no
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

Reading it as pipeline state is a mistake, and an expensive one. On the morning
of 2026-09-14 a GPU node's backup held 1,882 videos in 'outdated' (1,221 at
scope 'full', which reads as "needs about 14 minutes of GPU each") while the
processing server's live database held zero outdated rows, and the manifests for
those same videos recorded the declared scorer and current versions throughout.

The authoritative database for a node is the one that node's own watcher writes:
the `db_path` override in its `~/.mousereach/config.json`, else
`processing_root/watcher.db`. For reprocessing questions the processing server's
copy is the one that decides, because it is the only role with
`reprocesses_partial` true (orchestrator.py:433; the GPU role sets it false at
:728 and therefore passes `full_only=True`, marking only the scope it can
drain). Note that `handles_reprocessing` does NOT distinguish the roles -- the
GPU role sets it from `also_process` (:756), which is true on all three lab GPU
machines.

A gap worth recording: that same backup also held 661 rows at scope
'kinematics', which `full_only=True` should have prevented that node from
marking. Whether they predate the flag or arrive by another path is not
established here.

THE DASHBOARD STILL HARDCODES THE PATH THE CLI STOPPED HARDCODING

The 2026-08-23 update above moved all eight watcher commands onto
`_resolve_db_path`, which honours the node's `db_path` override and prints the
path it chose so a wrong database is loud. The dashboard was never included. It
resolves correctly in one place (dashboard/widget.py:892) and hardcodes the
fallback in four others: :365, :593, :1284 and :1415.

On this processing server the override points at
`processing_root/watcher_local.db`, which holds 4,113 rows (3,896 archived, 169
triage, 47 deep review, 1 unresolvable). The hardcoded
`processing_root/watcher.db` sitting beside it carries the full schema and zero
rows. The GUI therefore opens an empty database while the watcher writes a full
one -- the same failure the CLI had, silent, and indistinguishable on screen
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

Top-level `Archive/` keeps only `historical/` (read-only source material, still
the ASPA collage importer's default source) and development material awaiting
cleanup.


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

The return scan still files an empty triage folder away to `_Problematic`
(residue of an earlier divert or return), but only once the folder's
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
review as a bare .mp4 for a person to clear a second time (20 videos,
2026-09-10..29). Now (`queue_folder_holds_this_run`) the folder counts as this
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
children, about 14 GPU-minutes each, and fed duplicate processing and archiving
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
it is not removed automatically. WHY not delete it: a hold is judged from a
folder existing, and a leftover empty queue folder must not be enough to
destroy a crop. Delete such a file by hand to free the space.

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
normal path.

---

## Update 2026-09-15: the watcher pauses itself while a recording program runs

WHY. A GPU node in a behaviour room poses videos whenever nobody is recording.
A pose takes the GPU, CPU and disk for about 14 minutes, and a recording that
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
    and the row is 'failed' with a message saying so. This node does not retry
    it by itself (the GPU role has no job list for failed rows, and a full disk
    would otherwise copy a whole video every minute). Run
    `mousereach-watch-reprocess <stem>` once the cause is fixed.

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
them: nothing dismisses a message box by itself, and on the first live test
(2026-09-18, added in a later commit) the "Safe to record" box was still on screen
minutes after the recording program was closed and the node had gone back to work
-- the last thing the operator saw said the machine was free while it was posing
again. A box that cannot be shown OR closed (no desktop, not Windows, a failing
call) is logged at DEBUG and changes nothing -- the INFO log lines are written
either way.

FIRST LIVE TEST (2026-09-18, a behaviour-room node). Opening the recording program
paused the watcher within 3 s and showed the "Safe to record" box; closing it
resumed work 2 min 04 s later (the 120 s grace plus one 30 s poll). The
"MouseReach is stopping" box did NOT appear, correctly: it is raised from
`_recording_abort_reason`, which only runs from inside a pose or crop, and that
node had nothing in flight (every video in its queue resolved to a skip). So the
two statements above about stopping a pose or crop PART-WAY for a recording --
and the video going back to its queue rather than being failed -- are proven by
tests but NOT yet by a live recording on a node doing real work. Turned off with `watcher.notify_safe_to_record:
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

Measured on a behaviour-room node on 2026-09-18: 12 rows, one work slot each
(about 32 s), roughly six minutes of every startup, indefinitely. Nothing was
lost and no video was harmed; the node simply could not reach real work for the
first few minutes after each start.

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
whole poll interval rediscovering what the row already said -- one row per poll.
On the same node: six rows, 2 min 41 s of work slots after an unpause (18:14:56 ->
18:17:38, one every ~32 s against `poll_interval_seconds` 30), delaying the real
work queued behind them.

`DLCOrchestrator._select_work_item` now filters that bucket on the NO_FILE_HERE
placeholder before `_pick_from_pool` sees it. A row with NO recorded path is still
offered, because `locate_video_file` may find its file -- that is how a node picks
up work which genuinely is here. The retirement path itself is unchanged. The
constant is read from `watcher.db` with a getattr fallback rather than through
`self.db`, so a caller holding a database double is not coupled to it.

Tests: `tests/test_work_selection_skips_known_pathless.py`.

## Update 2026-10-21: every cycle says where its time went

**What was wrong.** The watcher's loop scans for work, picks ONE item, does it,
and starts again. None of the scans recorded how long it took. On the
processing server, a video's pipeline step took about 20 seconds and the next
item started a median 8 minutes later (90th percentile 11 minutes); from
2026-10-18 to 10-21, 88.7 of 89.4 logged hours fell in gaps longer than two
minutes. The log showed a version scan of about 80 seconds, then roughly eight
minutes with no line at all, so the time could not be attributed to any step.

**What changed.** `src/mousereach/watcher/cycle_timer.py` times every phase of
a cycle (`version and review scan`, `scan`, `pick`, `work`) and, on the
processing server, each part of the scan separately (`stale claims`,
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

**What the cycle timing showed.** The first timed cycle on the processing server
took 1,094 seconds, of which 17 were spent on a video. The rest:

    Cycle 1094.1 s: version and review scan 270.2 s, scan 807.2 s [stale claims 0.0,
    re-pose adoption 1.5, misfiled singles 566.2, posed discovery 1.9,
    review return 34.8, collage retirement 202.7], pick 0.0 s, work 16.6 s

**What was wrong.** `misfiled singles` (`ProcessingOrchestrator._rescue_misfiled_singles`)
runs on every cycle and looks for a video left in `Processing/Posed` without a
pose. For each video in that folder it listed the whole folder again, twice: once
to look for the video's pose file, once for a half-written file. The folder is on
the shared drive and holds every posed video waiting to be analysed together with
its pose files, so with 1,263 videos waiting that was over a thousand listings of
a folder of thousands of files, every cycle.

**What changed.** The folder is listed once per pass; the pose and half-written
checks look names up in that one listing. What the check decides is unchanged (all
its earlier tests pass unchanged); only the number of listings is.

Still to address, from the same line: the 30-minute version and review scan
(270 s) and the 3-hourly collage retirement (203 s) run inside the work loop,
so the cycle that hits them stops all work until they finish.

Tests: `tests/test_misfiled_video.py::test_the_folder_is_listed_once_per_pass_not_once_per_video`.

---

## Update 2026-10-21: a node keeps only what it starts; a stopped node's videos come back in hours

**What was wrong.** A GPU node that cut a recording kept every one of its videos
(up to eight) claimed in its own `Unanalyzed/Single_Animal/.inflight/<machine>/`
folder and in its local DLC queue, and posed them one at a time (about 14
minutes each). It kept them claimed through a pause -- a node pauses for a whole
recording day -- and another node could take them over only after the claims had
gone untouched for a day. So videos waited on one machine while the other GPU
nodes had nothing to do (found 2026-10-21: a video stuck on the node that had
done part of its work).

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
   the file server, never from the machines' own clocks, which disagree by about
   21 minutes (`watcher/claim_marker.py`, `GONE_AFTER_S`). Both are required so a
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

**What was wrong.** Three things, all measured on the processing server on 2026-10-21:

1. Two human-cleared deep-review videos (one CNT, one ASPA) could never return. An
   older divert between the queues had moved only the mp4 into deep review and left
   the pose and every result in the processing folder of the node that ran the video.
   The return searched the bundle, its manifest, Analyzed and pose staging -- not the
   folder it delivers to, nor the pose the video's own row names -- found nothing,
   logged an ERROR and tried again on the next scan, two minutes later. 4,660 and 3,423
   such ERRORs are in the kept logs; each try spent about 5 s searching Analyzed.
2. Every scan re-read every triage bundle's segments, outcomes, assignments and review
   files to decide whether it was cleared: 7-8 s of each two-minute scan, although
   almost no bundle changes between scans.

**What changed** (`watcher/review_return.py`):

1. The resolver also looks in the processing folder the return delivers to, and at the
   pose the video's row records, accepting only a pose of the declared model (the rule
   pose staging already had). Such a pose is used in place.
2. A return that cannot go ahead (no pose, or files still in the queue's build folder)
   is tried again after an hour (`RETURN_RETRY_S`), not every scan. The ERROR says so.
   The wait is kept in memory, so a watcher restart tries each once more.
3. A triage bundle's verdict is reused until the bundle changes: each scan lists the
   bundle's files with their sizes and modified times (about 2 s for the whole queue)
   and the video's ground-truth file, and re-reads the bundle only when either changed
   -- a person saving an answer changes the bundle, so it is read on the next scan.

Tests: `tests/test_review_return_local_pose.py`.

## Update 2026-10-21: ASPA recordings can be brought in by a list of sessions

`mousereach-aspa-import-collages --sessions-file <list>` (`aspa/import_collages.py`) copies
in only the archive recordings that hold at least one listed session, in the list's order,
instead of whole cohorts. A session is one animal's tray run, `{date}_{animal}_{tray}{position}`,
written the lab's way or the pipeline's; one per line, `#` comments allowed. WHY: a recording
holds eight animals, and the sessions one question needs sit in far fewer recordings than
their cohorts (for one study: 338 recordings, 316 GB, instead of whole cohorts at about three
times that). The dry run prints how many recordings would come in, every listed session that
has no recording in the archive, and every recording the pipeline already has. A recording
already in the intake folder, or cut and moved to `Analyzed/Multi-Animal`, is never copied
again (this also applies without a list).

Lists live in the pipeline's `priority_lists/` folder (data only; each list is written by a
lab script, never edited by hand). Tests: `tests/test_aspa_import_by_sessions.py`.

## Update 2026-10-21: a segment counts as answered only when the answer is about its frames

**What was wrong.** Four places decided whether a person had answered a segment -- the
review gate (may a freshly run video be archived?), the review-return scan (may a bundle
leave the queue?), the review tool's queue view, and the queue index -- and all four
counted an answer by the segment NUMBER it was written against. The results step matches
an answer by the FRAMES the reviewer saw (schema 1.1 `segment_span`). After a person
re-cut a video, an old answer for "segment 7" -- about other footage -- counted as
answering today's segment 7: the video left review, the results step correctly ignored
the answer, and today's segment 7 reached the data unwatched. The nightly bench check
then found the disagreement and sent the same videos back every night, and the return
scan released them again within minutes, with nobody looking (seven videos, 2026-10).
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
them are matched and applied, never dropped as artifacts (Logan, 2026-10-21: "if a human
segmented the video then that's what the algo needs to log even if the range was short").

**What a person sees.** A video whose earlier answers no longer line up with its current
segments stays in (or returns to) the triage queue until the segments nobody has watched
are answered; the review tool lists it as not done. An answer is never applied to footage
nobody looked at.

Tests: `tests/test_answered_means_matched_by_frames.py`.

**Update the same evening: the import copies on the file server.** The first full run of
`--sessions-file` (338 recordings) copied each recording by reading it into the
processing server and writing it back to the same share. That filled the server's network
link: taking in one posed video went from seconds to 461 s and a video's work from ~20 s
to ~170 s, for as long as the import ran. The import now copies with the system copy
(`CopyFileW`), which, between folders on the same share, the file server performs itself:
3 s for a 1 GB recording instead of 41 s, with no data through the processing server
(`aspa/import_collages.copy_file`). A stopped run's half-written `.partial` file is replaced
by the next run.

## Update 2026-10-21 (night): intake finds a video's files by name first; retirement waits an interval after start

**Intake.** Taking a posed video in took about 120 s, of which copying took 4 s: finding the
video's files (`ProcessingOrchestrator._get_associated_files`) listed the posed-video folder
and asked the share "is this a file?" for every entry -- about 5,000 -- before looking at the
name. It now tests the name first and lets the listing's own entry answer the rest. Same
files found; intake takes seconds. (It had been this slow all along -- 134 s on 10-19 -- but
only showed once the server took hundreds of videos in a row.)

**Collage retirement** (finished recordings to `Analyzed/Multi-Animal`) first runs one
interval (3 h) after the watcher starts instead of at start. It reads every processing
manifest in Analyzed -- 14 minutes on 2026-10-21 with a busy share -- the work loop waits for
it, and a restart is exactly when a backlog is waiting. Nothing waits on retirement.

Tests: `tests/test_intake_lists_names_first.py`.

## Update 2026-10-21 (night): the version scan reads folder listings, not one file at a time

**What was slow.** The half-hourly version and review scan (`ReprocessingScanner.scan`) checked
every archived video separately: its manifest's modified time (one network round trip each),
each place its review might be (`resolve_review_path`: the triage bundle, beside the canonical
video, the durable store -- several missed lookups each), and the review's sha256 (re-read every
pass). On 2026-10-21, with a slow share, a warm pass took 14 minutes for 4,597 videos, every 30
minutes, and the processing server's work loop waited for all of it.

**What changed.** One listing of each `Analyzed/<project>/<cohort>` folder gives every manifest,
features file and modified time (`ReprocessingScanner._index_archive`; never inside a superseded
folder). Every video's review comes from `causal_review_io.review_path_index`: the same places in
the same order with the same newest-wins rule as `resolve_review_path`, from a few listings. Review
fingerprints are kept between passes until the file changes. Verified on the live archive before
deploying: identical manifests (4,609) and features (4,553); the review index agreed with the
per-video lookup on all 400 sampled videos (246 with a review) and took 2 s for all 4,597.

Found on the way: the "beside the canonical video" place in `resolve_review_path` never resolves on
this lab's layout -- it looks for `Analyzed/CNTnn/<video>.mp4`, while videos live in
`Analyzed/Connectome/CNTnn/`, and ASPA names fail it outright -- so reviews are found in the triage
bundle or the durable store. Kept as is (the index follows the same rule); worth a look.

Tests: `tests/test_version_scan_from_listings.py`.

## Update 2026-10-21 (night): a shared video another node took costs no work slot

Every GPU node registers every video it sees in the shared singles folder ('validated').
When another node takes one first, the file is gone; the adopt handler found that out and
set the row aside -- one row per work cycle. Once nodes shared what they cut (keep 2, share
the rest), that became routine: one node spent twelve cycles in a row on it on 2026-10-21
while 337 recordings waited to be cut. Now, before choosing, the node lists the places the
handler takes a video from -- its own claim folder, its local Processing and DLC queue, the
top of the shared folder -- and sets aside, in one pass, every 'validated' row whose file is
in none of them (unresolvable, "gone from the shared singles folder", re-driven if the file
comes back), exactly as the handler would. A row whose file IS in the shared folder is offered
whatever its recorded path says (`DLCOrchestrator._split_gone_singles`). With no shared folder
configured, every row is offered as before.

## Update 2026-10-22: the collage-retirement check finds its facts from listings

`collage_provenance.build_complete_stems` (which offspring are truly done, so their collage can be
retired) read all ~4,600 processing manifests in Analyzed every run and looked each offspring's
review up separately: 14 minutes on 2026-10-21 with a slow share, while the processing server's
work loop waited. One walk now collects every live manifest and features file with its modified
time from the listing (`analyzed_tree.iter_file_entries`, same files and rules as `iter_files`);
manifests are re-read only when they changed (kept for the life of the process); reviews come
from `causal_review_io.review_path_index`. Same answers. Tests: `tests/test_retirement_from_listings.py`.

**Update 2026-10-22: the version scan remembers older reviews' dates.** For a video whose
manifest predates the `applied_review` fingerprint, the scan compares the review's in-file
`reviewed_at` with the manifest's `created_at`, and it opened the review on every pass to read
that one field: 1,614 files, ~107 s of a ~220 s warm pass (measured 2026-10-22 on a copy of the
server's database). The value is now kept until the review file changes (same key as the
fingerprints: path, size, modified time); a failed read is never kept. Warm pass measured
121 s afterwards, with no review files read; what remains is the folder listings.

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
when the scheduled 05:30 run failed with "database is locked": on a slow share an ordinary write can
hold the file for minutes). A once-a-day report has no reason to give up sooner.

## Update 2026-10-22: a GPU node retries the shared-record writes that failed

A GPU node writes each video's state to the shared record (`watcher_central.db`,
`pipeline_videos`) as it cuts, poses and hands it off. Those writes were best-effort: a failure
(the shared database locked) was logged at DEBUG and dropped, and the record fell behind for good --
one collage's 8 singles all existed, but only 3 were on record a day after the 10-21 lock
failures. Other nodes read that record before cutting a collage, so a missing child is a collage
another node may cut and pose again. Now `DLCOrchestrator._sync_to_connectome` keeps the latest
failed state of each video and retries it at the start of every scan; a newer successful write
for the same video drops it, and a retry never replaces a row updated since the original attempt
(`PipelineCoordinator.sync_video_state_unless_newer`, stamped with the attempt time). At startup
the node puts on record every video it has that the record lacks entirely, never touching an
existing row. The fill writes in batches (`insert_video_states_if_absent`, one transaction per
200 rows: one row per connection opened the shared file ~470 times in 12 s and hit "unable to
open database file"), and a fill the record interrupts is finished by a later scan, at most every
10 minutes (`_finish_shared_record_fill`), instead of waiting for the next restart. Tests:
`tests/test_shared_record_writes_retried.py`.

## Update 2026-10-22: work-order selectors take ASPA lab ids

An ASPA video carries an encoded id (ASPA0401); the lab names the animal D01. Session lists
already accepted either spelling, but an `animal_id` or `session` selector in the work order
written with the lab id matched nothing (plan gap G5). `work_priority.field_labels` now adds
the lab spelling of every encoded ASPA id (`mousereach.aspa.identity`); other ids are
unchanged. Tests: `tests/test_work_priority.py` (TestAspaLabIds).

## Update 2026-10-22: a collage with a slot ffmpeg could not cut is not "cropped"

A GPU node cutting a collage skipped any slot ffmpeg failed on and then recorded the collage
`cropped` anyway -- "0 singles" when every slot failed -- so nothing ever retried it. Two
collages lost 9 sessions that way (one had all 8 slots fail in the same instant on 2026-09-20),
found when a census of every collage was checked against every single-animal row. Now
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
staging, not in Analyzed -- it used to mark the video failed. On 2026-09-01 three videos failed
that way because the bench-disagreement router had moved their bundles, pose included, into
triage. A missing pose is not a scientific outcome. `ProcessingOrchestrator._no_pose_for_reprocess`
now (1) puts the video in a review queue's state when its bundle there holds the pose; (2) when
its video is in Analyzed, sets `reprocess_scope` to `full` and logs a 'skipped' reprocess step --
the row stays `outdated`, and the next version scan publishes a re-pose request that a GPU node
serves; (3) only when no video can be found either, marks it failed with "no pose and no video
found". Tests: `tests/test_no_pose_reprocess.py`.

## Update 2026-10-22: a queue bundle of an archived video is filed away

The return scan reads every queued video's state in one query per queue
(`WatcherDB.states_of`). For a video this node has `archived`, a bundle holding
nothing a person wrote is moved to `_Problematic` with a note when it holds no
results at all (no `<stem>_segments.json` -- a file a return could not take
along) or when every file in it is older than the video's processing manifest in
Analyzed (`_superseded_by_archive`, `review_return.py`; both queues, under the
same per-video claim as every other move). Analyzed is read once per bundle and
row change, not on every scan. Bundles the bench-disagreement router places are
never caught: their results leave Analyzed and their routing record postdates the
archive. Companion to the archive-step fix above (`queue_folder_holds_this_run`):
that one stops new husks from derailing a run; this one clears the ones already
sitting in a person's queue. Tests: `tests/test_review_return_superseded.py`.

## Update 2026-10-22: a leftover triage copy of a deep-review bundle is filed away

When a video has a folder in both review queues and its segmentation failed, the return scan
used to skip it with "exists in BOTH queues; divert skipped (duplicate needs deliberate
cleanup)" -- every scan, for good, because nothing ever did that cleanup (one video logged it
231 times from 2026-10-19). The scan now compares the two folders (`_is_leftover_copy`,
`review_return.py`): if every file in the triage folder has a byte-identical file of the same
name in the deep-review folder, the triage folder is moved to `_Problematic` with a note, under
a return claim like every other move, and the deep-review bundle is untouched. A pair that
differs in anything stays for a person, as before. Sizes are compared before any bytes are
read, and a pair's verdict is kept until either folder changes. Tests:
`tests/test_review_return_stale.py`.

## Update 2026-10-22: the version scan keeps what it has read across restarts

The version scan's three caches -- each manifest it parsed, each review's fingerprint, and each
older review's reviewed_at -- lived only in memory, so the first scan after every watcher start
read all of them again over the share: 623 s on 2026-10-22, while the work loop waited. They are
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
the nightly backup's folders, anything else the file held. And with `nas_root` gone, the
pipeline root was guessed as `<nas_drive>\! DLC Output`, a layout retired in 2026: every
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
the gate sent every Easy/Flat video to deep review. On 2026-10-22 the processing server, idle
apart from them, routed 100 Flat-tray videos there in three hours, and 469 Easy/Flat bundles in
deep review were held for that reason alone. The structural layer now asks for the outcomes file
only when the per-tray step table (`watcher/router.SKIP_STEPS`) runs outcome detection for that
tray; a Pillar video without one is still critical, and a name with no recognisable tray is
held to the full set. Checked on copies of eight held Flat bundles: each now passes the check
and the gate. Tests: `tests/test_qc_easy_flat_no_outcomes.py`.

## Update 2026-10-22: a quality hold the current check would not make leaves the person's queue

When the quality check is corrected, the holds it made by mistake used to stay in deep review
until a person cleared each one. The return scan now re-checks every deep-review bundle routed
`qc_needs_review` that holds no file a person wrote: the pose-tracking layer's stored flags are
carried over (that layer reads the whole pose file; a hold it made stays, except a ruler-variation
hold -- see the 2026-10-23 update), the file-based layers are recomputed. A bundle with nothing critical left goes back to the posed folder for a fresh run
(mp4 and pose files; its other files are set aside under `_Problematic/_released_qc_holds_<date>/`)
and its row becomes `dlc_complete`, so intake takes it in its priority turn -- not `processing`,
which would block intake of new work. The first use: the 469 Easy/Flat bundles held only for the
outcomes file they never get (fixed the same night). Bounded to 30 s of each scan; one re-check
per bundle until it changes; nothing in the posed folder is ever replaced. The posed folder usually
still holds the held video's files (intake leaves them there until archiving); the same file
there (same size and modified time, or the same SHA-256 when the times differ) is kept as the
one copy and the bundle's duplicate removed, a differing one leaves the bundle for a person (67 of
the first 74 releases had been refused as a clash; hashing every pair made each release ~10 s). Tests:
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
brought in -- a person corrected an animal id on the pipeline's copy (CNT1012 -> CNT0112), and the old
copy under the old name would have recreated the animal the correction removed.

## Update 2026-10-22: the backup says what failed, skips test residue, and fails loudly

From 2026-10-13 the nightly backup reported "Robocopy error ... exit code 11" for the Databases
tree every night, and nothing else: robocopy runs with its file listing off and its error lines
were never logged, and `mousereach-backup --once` exited 0 regardless, so the scheduled task
showed success to every check that reads task results. Run by hand, every failure was a
`pytest-cache-files-*` folder -- residue pytest leaves, unreadable, in a code checkout on a
network share; never data. Now (`watcher/backup.py`): those folders and `.pytest_cache` are
excluded; a failed run logs robocopy's own error lines (the file and the reason, first ten);
and `--once` exits 1 when any folder did not sync completely, so the task result shows it.
Checked live: the same run now exits 0 with nothing failed. Tests:
`tests/test_backup_reports_failures.py`.

## Update 2026-10-22: running watchers follow edits to the lab's work order

The lab's work order (`priority_order.json` at the pipeline root, or a machine's own
`watcher.work_priority`) was resolved once per watcher process, so an edit -- including a Save
in the priority panel, which says machines pick it up within a minute -- reached a running
watcher only when it was restarted. `BaseOrchestrator.work_priority` now re-resolves at most once
a minute (`PRIORITY_REREAD_S`; the file read itself was already cached for a minute) and logs the
order at the first resolve and again, marked "(changed)", whenever it changes. Applies to every
watcher once it runs this code. Tests: `tests/test_work_priority.py` (TestRunningWatcherFollowsEdits).

## Update 2026-10-22: every bypass of the video state rules says why

`force_state` (the escape hatch from the state machine) logs each use at WARNING with the reason
it is given -- the only record of why a video skipped its normal transitions. Seven call sites
gave none, so the server log read "reason: not given" 150 times on 2026-10-22: posed-folder
discovery, orphan recovery, cross-node recovery, database-confirmed archiving, re-runs of
archived videos, crystallize and uncrystallize. Each now states its reason, and
`tests/test_state_bypasses_give_reasons.py` reads the source so a new call without one fails.

## Update 2026-10-22: a shared-drive outage idles re-running instead of failing videos

When the shared drive dropped on 2026-09-09, the re-run step marked each outdated video FAILED
("Archive directory not found for reprocessing"), about one a second -- infrastructure failures
recorded as the videos' fault, which a failed state must never mean. Now the processing server
offers no re-run work while the Analyzed archive cannot be seen (`_archive_reachable`): the
videos keep their state, the log says once that the archive is out of reach and once that it is
back, and the re-run step itself declines rather than failing a video if it ever gets there.
Tests: `tests/test_outage_idles_reprocessing.py`.

## Update 2026-10-22: the watcher's dry run only reads

`mousereach-watch --dry-run`, the step the install guide has every new machine run first,
built the whole watcher (whose start-up runs recovery) and then called the real scans, which
register files and move videos between states -- so it changed the local database and touched
the shared one (104 recordings advanced on one machine, 2026-08-30). It is now a read-only
preview (`watcher/preview.py`), run before the single-instance lock, the database and the watcher
are built: the database is opened read-only, the folders the machine serves are listed, the lab's
work order is shown, and the next pick is worked out by the watcher's own selection code on a
stand-in database that has no write methods. It can run beside a live watcher. The two old
`dry_run` methods print the same preview. Tests: `tests/test_dry_run_is_read_only.py` (the
database bytes and the folder tree are unchanged after a preview).

## Update 2026-10-22: a node's status report says what it is doing

Each node's report on the shared drive (`Processing/.node_status/<host>.json`) said "working"
every loop turn and nothing else, so a node churning on nothing looked exactly like a
productive one, and a node deep in a 14-minute pose looked like one whose report had gone stale
(plan gap G14). It now also carries `doing` (the item in hand: type, video, since when -- written
as the item starts), `last_done` (the last item finished) and `done_last_hour`. The Nodes
section of `mousereach-watch-status` shows them on each node's line. A node shows them once it
runs this code. The last item and the hourly count are kept by the running watcher, so they
start again from nothing when it restarts. Tests: `tests/test_node_status_says_what_it_is_doing.py`.

## Update 2026-10-22: a list of sessions can lead within its project (plan Step 1)

The lab's order could say "this project, these cohorts" but not "these sessions a study needs".
The order file now takes `session_lists` beside `cohorts`: per project, files of sessions in
the shared pipeline folder (one per line, lab or pipeline ids). Within that project the listed
sessions come first, then its cohorts, then the rest. A recording counts as listed when any of
its animals' sessions is (it must be cut to reach that animal). New selector fields `session`
and `session_list`; list files are re-read at most once a minute. The priority panel has a box
for it. A panel Save now keeps keys it does not manage (notes, hand-written selectors; plan gap
G4), and keys or list files the order cannot use are named in its status line (G8). Tests:
`tests/test_session_list_tier.py`.

## Update 2026-10-22: the lab's order at the remaining selection points (plan Step 3)

Re-pose requests were taken from the shared queue alphabetically (plan gap G2), so 2022 ASPA
videos went to the graphics cards ahead of 2025 CNT ones whatever the order said; a GPU machine
now takes them in the lab's order (`repose._in_lab_order`: idle-only work last, then tier, then
oldest first). And every place that sorts work by the order -- the review-return scan's order,
the review tools' queue listings and next-video choice, the re-segmentation queue -- now puts
idle-only work (Easy/Flat by default) after everything else, whatever its tier (G7), through one
sort key, `WorkPriority.rank`. Before, they sorted by tier alone, so an Easy video in a preferred
project came ahead of Pillar videos of the next. Tests: `tests/test_repose.py`
(test_consume_takes_requests_in_the_lab_order), `tests/test_session_list_tier.py`.

## Update 2026-10-22: a GPU node sets aside local copies of videos already finished

Copies of videos finished long ago sat in a GPU node's own pose queue and processing folder for
good: never picked (their rows are finished) and never removed. The Lab DLC PC held 34, all CNT
01-04 Pillar with results in Analyzed, the oldest from 2025-07; they came from the old behaviour
where a node kept every video it cut (it keeps 2 since 55e5ad8). At startup a GPU node now moves
such copies out of `DLC_Queue` and `Processing` (`DLCOrchestrator._set_aside_finished_copies`):
only when its row is finished (archived, outdated or crystallized) AND the video's processing
manifest is in Analyzed where the video belongs. Every file of that video moves together to
`<processing_root>/_finished_elsewhere/<date>/<folder>/` -- set aside, never deleted; one log line
says how many. Tests: `tests/test_set_aside_finished_copies.py`.

Later the same day the step moved to `BaseOrchestrator` and the processing server runs it at
startup too: it held three such copies in its own `Processing`, left from 2026-08-26 after the
videos were filed into Analyzed. On the server only `archived` and `crystallized` count as
finished, not `outdated`: the server re-runs an outdated video by copying its archived files down
into `Processing` while the row still says `outdated` (`_reprocess_video`), and those copies are a
re-run's working set.


## Update 2026-10-22: startup recovery no longer clears a review hold with an older record

At start, a graphics-card machine compares each row of its own record with the shared one
(`watcher_central.db`, `pipeline_videos`) and moves rows forward to match. A review hold
(triage, deep review) is not among the ordered states, so it ranked below every shared state
and was overwritten. Shared `archived` from a pose-only machine means "this machine handed it
off", not "analysed", so a hold could become `archived` with no results anywhere. Seen on a
behaviour-room machine on 2026-10-21, from a shared record written three weeks before the hold.
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

Measured in `review_records/reviews`:
- 59 files carry two records for the same frames, 118 such stretches in all. 84 resolve the same
  under any rule. In 13 stretches (12 videos), a placeholder or a left-blank `triaged` record
  sat first and was applied over a person's decision. In 21 stretches (14 videos), two
  decisions disagree.
- 239 touched answers (91 videos, 215 of them saved in 2026-09) name a causal reach
  wholly outside the frames they record. Applied, they made a neighbouring segment's reach the
  causal reach.

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

`mousereach-route-to-queue` used to flag whatever segments its caller listed. A caller -- the
database's nightly bench-sheet scan -- cannot always know what a person answered: a segment with
no detected reach has no reach row in the database, so nothing there says who decided it. A
flagged segment the release check counts as answered goes straight back out with nobody looking,
and the next night's scan sends it again. Now the command reads the video's review and drops every
requested segment a person (or ground truth) has answered, matched by frames
(`answered_segments_of` in `watcher/route_cli.py`, the release check's own rule). It reports them
as "answered", and does not route the video at all when nothing is left (exit 0, tag "done").
Tests: `tests/test_route_cli_db.py`.

## Update 2026-10-22: a re-run finds the archived pose where it lives

Every re-run of an archived video (`_reprocess_video`) found its pose file by walking the whole
Analyzed tree: 36 s on a quiet server and about 4 minutes on a busy one, for a re-run whose own
work takes about 3 s. With a few re-runs queued, the server's throughput fell from about 35
videos an hour to 8. The pose is always in the video's results folder (`Analyzed/<project>/<cohort>`)
or the same cohort folder of a per-model tree (`Analyzed/<project>/DLC Model <N>/<cohort>`), both
computable from the video id. `archived_pose_candidates` looks there first and walks the archive
only when no pose from the declared model is found there, which is the old behaviour. Tests:
`tests/test_reprocess_pose_lookup.py`.

Correction, measured the same morning on a live re-run: the archive walk was about 30 s of it.
Most of the time went to `_reprocess_copy_set`, which decides which files to copy down. It asked
`Path.is_file()` of every entry in the cohort folder, tens of thousands of network stats, before
looking at the name: 175 s. It now checks the name first and uses the directory listing's own
file flag (no extra round trip on Windows). Test: `tests/test_reprocess_copy_set.py`.

## Update 2026-10-22: re-archiving leaves unchanged files where they are

`archive_video` swept everything already at the destination into the versioned Archive before
moving the new set in. On a re-run the video and the pose come back byte-identical: they were
copied down from the archive and never changed. Sweeping them stored another copy of the pose for
every re-run and moved about 100 MB back over the share, 60-100 s per video. Files identical to
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

`mousereach-watch` itself also refuses any WORD (it takes options only): an install script on the lab PC
ran `mousereach-watch stop` meaning to stop the watcher, which instead started one (or, with one
already running, reported it and did nothing). It now exits 2 and says how stopping works
(`watcher_stop.flag` or the Stop button; `mousereach-watch-toggle --pause` to pause).

## Update 2026-10-22: the server sends the shared record the exports it missed

After archiving, a video's record and log go to `watcher_central.db` (the audit table `videos`).
A failed export was a warning and never retried: five archived videos from the August/September
share outages were missing there (sent by hand 2026-10-22). At every start the processing server
now compares its archived rows with the shared table and sends what is missing
(`ProcessingOrchestrator._export_missed_to_central`; only the server, where "archived" means filed
in Analyzed). Tests: `tests/test_missed_central_exports.py`.

**Changed 2026-10-26.** That check looked for a video with no row at all, from any machine. A video
posed on a GPU PC already has that PC's row, so an export from the server that failed was never sent
again (four were lost the morning of 2026-10-26, while the share was slow). It now compares the
server's OWN rows (`source_machine`) with its archived videos and re-sends any that is missing or
older than the local row. It runs at start and then every 30 minutes (cycle step "scan/shared
record"), at most 50 videos per pass so a backlog never holds up the work loop. The export itself
waits up to 30 s for another machine's write to finish, instead of Python's default 5 s.


## Update 2026-10-26: asking a person again about a pellet they answered (--recheck)

`mousereach-route-to-queue --recheck` reopens pellets a person already answered instead of skipping
them (`route_cli.reopen_answered`): each matched (by frames) review record keeps its answer under
`answers.recheck.previous`, is marked `answers.reviewed = False` (open), and the pellet is flagged with
the question; the review file is first copied to `Analyzed/Archive/manual_corrections/`. The video
stays in triage until that pellet is answered again. WHY: answers of "retrieved" whose frames show the
pellet lying in the scoring area afterwards (Logan, 2026-10-26: "are you sure"). Ground truth is never
reopened. Tests: `tests/test_route_recheck_answered.py`.

## Update 2026-10-26: checked against the hand scores before filing

The last step before a video is filed (`_archive_to_nas` on the processing server,
`_archive_locally_processed` on a node that also analyses) is now a comparison of its pellets with
the session's hand scores (`watcher/hand_scores.py`, configured by the `hand_scores` section of
`~/.mousereach/config.json`; README "Checked against the hand scores before filing"). A pellet that
disagrees, and that no person has answered, is flagged (`flagged_for_review`, `flag_reason` "disagrees
with the hand score: ...") and the video goes to triage through `route_to_queue`, as the review tool
already shows such reasons. A session with no hand scores in a required project leaves the video in
'processed', unfiled and not picked again until the scores file changes, listed (grey, waiting) in
`<NAS root>/attention/<host>_hand_scores.json`; once the cohort's sheet has been imported after the
video was ready and still has no scores for it, the session was never scored by hand and the video
is filed without the comparison (`scores_never_came`; Logan, 2026-10-26: "If can't compare to manual,
then can't mismatch manual"). A cohort with no successful import at all waits if the manifest's
`sheets_seen` lists it (its first import is coming) and otherwise files after the manifest's
`full_import_at` (the last import of every sheet) passes the time it was ready -- not the file's own
`written_at`, which a hand import of one other cohort also moves. A check that cannot run never files
the video, and its item is marked `needs_person` (kind `hand_score_check_failed`), so a wrong setting
shows in red instead of quietly stopping the filing. WHY (Logan, 2026-10-26): filing means done, so a video must not leave processing until it
has no issues; before this the comparison ran nightly on videos already filed and moved them back.
Tests: `tests/test_hand_score_check_before_filing.py`.

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
the manifests. Plan Section 0 item 7. Tests: `tests/test_version_index_journal.py`.

## Update 2026-10-22: a re-pose request is never claimed twice

A graphics-card machine claims a re-pose request by moving it into `Repose_Queue/.inflight/`. The move
was `os.replace`, which overwrites: a request published again while another machine's claim for the
same video was still in flight was claimed a second time, and two GPUs posed one video. `_claim` in
`watcher/repose.py` now refuses an existing claim (os.rename on Windows; a hard link elsewhere), the
second machine logs "another node already holds this re-pose" once and leaves both files alone, and
the leftover request is closed by the existing "already satisfied" check when the pose lands. Plan
Section 0 item 8. Tests: `tests/test_repose.py`.

## Update 2026-10-22: recording days and hours (G12)

A behaviour PC paused whenever its recording program was open, and the program is routinely left
open after the day's recording, so the PC idled until someone closed it (two days once; over an hour
on 10-22 with the program open and nothing recorded). The lab knows when recording can happen at all,
so that is now configuration: `watcher.recording_days` (e.g. "Mon-Fri", or a list) and
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
1,249 archived rows on the processing server still showed failures from runs that later succeeded
(1,164 of them "Archive directory not found for reprocessing", from the 2026-09-09 share outage), and
every status view read them as current. `update_state` and `force_state` now clear it on the way to
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
the 24-hour rule most posed copies of videos in review had lost theirs (433 of the 566 there on
10-22), so to any machine taking in posed videos they looked like waiting work; and lets go of this
machine's claims on videos that have left the folder or are waiting again. A machine that
finds a video held elsewhere records that it did no work (it used to count the refusal as progress
and pick the same video again at once) and leaves the video out of its choices for 10 minutes, so
the next one in the work order is taken in. A claim folder that cannot be made means no intake, not
intake without a claim. Plan Section 0 item 1 (the review-queue half was d4876f4). Tests:
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
silently skipped -- two GPU PCs still named retired roles there.

## Update 2026-10-23: the census lists collages by name

The pipeline census (`mousereach.census`, run nightly by mousedb's snapshot job with a one-hour limit)
found collages with `rglob("*")` and an `is_file()` check on every entry: one round trip to the share
per file. One archive root holds about 164,000 files, of which about 5,000 are collage-shaped; on
2026-10-23 the census ran past its hour twice at night (it had taken 656 s the previous afternoon), so
Where Is My Data kept the previous day's counts. `collect_collages` now walks with `os.walk` and
filters on the name, the walk's own listing saying which entries are files -- the rule
`walk_analyzed` already followed -- with `walk_onerror` keeping rglob's rule for a folder that
cannot be listed. Same night, same load: 1,128 s. Tests: `tests/test_census_collage_listing.py`.

## Update 2026-10-23: a ruler that varies within a video never holds it

The quality check measures the ruler (the SABL-SABR distance) across the video. A variation over 15%
was a 'critical' flag, which sent the whole video to deep review for a person. It is now a warning:
recorded in `_triage.json`, never a reason to hold. Why (lab decision): the millimetre scale is an
estimate (one median per pellet segment, over the segment's steady middle), every kinematic also
exists in pixels, and no conclusion is drawn from a single reach -- data preparation and statistics
average such variation away. Videos already held for it are released by the review-return scan
(their stored flag is recognised by its text, `triage.RULER_UNSTABLE_DESC`). A pose file with no
ruler points at all is still critical: that is a wrong model, not variation. Tests:
`tests/test_release_lifted_qc_holds.py`.

## Update 2026-10-23: Easy and Flat are session-only -- not segmented, never held, reaches and attention

Lab decisions of the day: Easy and Flat trays are never held for a person; their pellet outcomes do not
matter (pellets are presented, but which pellet a reach went to is not used); they are not segmented;
they need no kinematics; their attention is wanted; they come last, behind every other priority. One
test decides it everywhere: `config.is_outcome_free_tray` (tray letter E or F).
* **Not segmented.** The segmentation step writes one whole-video span (a single boundary at frame 0,
  `segmenter_algorithm` "whole_video_outcome_free_tray") instead of running the segmenter, so reach
  detection runs over the whole session and the ruler is one estimate for the video. Checked before
  the keep-human-boundaries rule. Verified on a filed Easy video: 79 reaches either way (21 segments
  before, 1 now), ruler 31.9 px against a 32.4 px per-segment median.
* **Never held.** The gate passes them (`review_gate.evaluate_gate`); the quality check no longer counts
  their segments. Videos already held by the machine (a quality flag or failed segmentation) are
  released by the review-return scan even when a person's file is in the bundle -- what a person can
  answer there is an outcome -- and that file is kept with the set-aside outputs. A session a person
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
feed did, and as was done by hand for parked Easy/Flat videos) skipped that. Now:
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
  or an unreadable cut, is named for a person. (The first version copied the nearest recorded day's
  groups; groups change from day to day, so it paired cuts of different recordings and once put an
  animal alone where it had shared a recording. Caught in the dry run; nothing was built that way.)
  Also for a person, never rebuilt: cuts as long as a recording that survives that day (they came out
  of it under a name it does not list -- an identity question), cuts as long as another cut of that
  day whose recording survives (same reason; this catches a recording trimmed after it was cut), and
  cuts under a minute long.
* `mousereach-aspa-import-collages` reads two kinds of name that are not in standard form: a
  separator typo between ids (read as a comma when that gives exactly eight ids and one reading), and
  one recording holding two cohorts on different runs (`_JP3,IP1`), copied in once per run with the
  other cohort's seats empty. Each is listed by the dry run and kept in the mapping file with its
  original name. Before this, such a recording was left out and its animals' cuts looked lost.
* `mousereach-aspa-name-audit` (aspa/name_audit.py) checks every archive recording's name against
  ASPA's own cuts (each cut traced to its recording by length, or through the recording naming it or a
  cut exactly as long when the recording was trimmed after it was cut; every disagreement confirmed by
  the picture -- the cut is one seat of the recording, mean difference ~1 against 20+ for other seats --
  because two recordings of one day were found 3 frames apart by chance) and writes
  `<nas_root>/pipeline_records/archive_name_conflicts.json`: recordings whose seats the cuts name for
  animals they do not list. The import runs the same check on the days it imports and does not copy a
  contested recording in. A recording held in quarantine counts as in the pipeline for the import and
  the feed. Nothing is renamed: which name is right is for a person.
Tests: `tests/test_rebuild_recording.py`, `tests/test_aspa_feed_to_shared_folder.py`,
`tests/test_aspa_name_audit.py`.

## Update 2026-10-23: triage shows why, and never takes an answer the pipeline would drop

Reported by the triage reviewer: videos answered the day before came back. Three were answered, and
each answer was dropped by the review matching without a word on screen: two named a reach outside the
segment's frames, one answered a segment four frames long. The video then asked again (one segment had
been answered three times). Now:
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
  (`segment_details`) and stores each as that pellet's `flag_reason` -- mousedb's nightly bench check
  now says, per pellet, what the hand score and the algorithm each say. Finished videos sent back for
  a new pellet had looked like the reviewer's earlier answers being lost.
Tests: `tests/test_triage_answers_that_cannot_count.py`.

## Update 2026-10-24: posing on a machine's CPUs (`mousereach-cpu-pose`)

Posing is the slow step, and only GPU nodes did it. A machine with many idle cores
can add to that by running `mousereach-cpu-pose` beside its watcher.

**What was measured** (a 104-thread server, 2026-10-24). One DeepLabCut process on the
CPU poses about 2 frames a second and uses about three cores, whatever thread count it
is given -- so speed comes from many small processes side by side: 12 processes of 3
threads made ~22 frames/s (~2 videos an hour) at 28% of the machine; 24 made ~28 frames/s
at 80%, each slowing the others. A CPU pose is the same as a GPU pose: on 602 frames of a
GPU-posed video, the largest difference in any body part's position was 0.14 px, no
confident point differed by more than 1 px, and no point crossed the 0.5 likelihood line.

**What it does, step by step** (`watcher/cpu_pose_pool.py`), with the GPU nodes' own code
wherever it exists:

1. **Picks** the next unclaimed video at the top of `Unanalyzed/Single_Animal` in the lab's
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
   fixer (`boundary_source: "human"`, `corrected_at`) or a ground-truth file has every
   boundary verified -- in either case after the bundle's `routed_at`. Hand-set cuts from
   before it arrived do not count. The return scan enforces it (one log line per video per
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
