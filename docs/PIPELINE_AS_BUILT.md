# How the MouseReach pipeline actually works

Describes: src/mousereach/watcher, src/mousereach/archive, src/mousereach/config.py, src/mousereach/pipeline/
Verified against: b65fcf0 (2026-08-23), with sections 4, 5 and 6 re-verified
against the pathless-row and DLC-staleness fixes of 2026-08-24 (see the
update section at the end).

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

What software records the videos is outside the code entirely. The word "OBS" appears nowhere in the source. Recording, naming, and copying the file into Unanalyzed\Multi-Animal are human steps the code simply assumes have happened.

There is one other, deliberate way in: a one-off command copies archived ASPA collages into that same folder, re-encoding their names into the required format first (src/mousereach/aspa/import_collages.py:84, :229). It is run by hand, not on a timer.

Once a collage has been cropped it is left in place; much later, a separate slow-cadence job on the processing server moves a collage out to Analyzed\Multi-Animal, but only once every single-mouse video cut from it has finished the whole pipeline cleanly (src/mousereach/watcher/orchestrator.py:1404-1420).

2. HOW MANY WATCHERS - THE HONEST COUNT

There is ONE pipeline watcher PROGRAM with TWO ROLES, plus TWO other unrelated watcher daemons.

The pipeline watcher is started by the command `mousereach-watch` (pyproject.toml:178). On startup it reads the `watcher.mode` setting from ~/.mousereach/config.json and picks one of two behaviours (src/mousereach/watcher/cli.py:292-302):

  - mode = "processing_server"  -> ProcessingOrchestrator (src/mousereach/watcher/orchestrator.py:1315)
  - anything else, including a missing value, which defaults to "dlc_pc" (src/mousereach/config.py:676) -> DLCOrchestrator (src/mousereach/watcher/orchestrator.py:356)

The two roles watch DIFFERENT folders, which is the cleanest way to tell them apart:

  - The graphics-card role watches the collage front door, Unanalyzed\Multi-Animal, plus Processing\Single_Animal (src/mousereach/watcher/watcher.py:62-70). It crops collages, runs DeepLabCut, and hands the results off.
  - The processing-server role never looks at the front door at all. It watches only NAS_ROOT\Processing\DLC_Complete, the folder where posed videos are staged (src/mousereach/config.py:120, src/mousereach/watcher/orchestrator.py:1364-1371). It never crops a collage and never runs DeepLabCut - its work queue only contains archive, intake, pipeline and reprocess items (src/mousereach/watcher/orchestrator.py:1432-1531).

The clean two-role split has an important exception. A second setting, `watcher.also_process` (src/mousereach/config.py:678), when true, makes the graphics-card role skip the handoff entirely: it runs the analysis algorithms locally and archives the finished results itself (src/mousereach/watcher/orchestrator.py:513, :528, :548). All three graphics-card machines in the shipped machine profiles have also_process set to true (src/mousereach/setup/lab_profiles.json, profiles "NAS / DLC PC", "Vid&DLC1PC", "Vid&DLC2PC"). On those machines, ONE watcher does the entire job end to end and the processing server never sees those videos.

Two further watcher daemons ship in the same package and are genuinely separate programs:

  - `mousereach-backup` (pyproject.toml:197) runs a timed copy of Y: to the X: backup drive using robocopy (src/mousereach/watcher/backup.py:25). It only starts if a `backup` section is enabled in the config file (src/mousereach/watcher/backup.py:196). The processing-server machine profile enables it.
  - `mousereach-sync-watch` (pyproject.toml:174) watches the Processing folder and pushes new results into the central database (src/mousereach/sync/watcher.py:121). Note that the pipeline watcher already syncs results to that database itself (src/mousereach/watcher/orchestrator.py:1167, :2095), so this daemon is an alternative tool rather than a required part of the flow.

A class called FileWatcher also exists (src/mousereach/watcher/watcher.py:32) but it is a helper used inside both orchestrators, not something anyone launches.

3. WHO RUNS THE GRAPHICS-CARD ROLE, AND HOW IT IS ENFORCED

The role is declared by a human in the config file, not detected from the hardware. But once declared, the command-line entry point checks the hardware before it will start. For any mode other than "processing_server" it requires: the network drive to exist, the DeepLabCut configuration file to exist, ffmpeg on the command path, and a usable graphics card (src/mousereach/watcher/cli.py:238-266). "Usable" means TensorFlow or PyTorch can see a card (src/mousereach/gpu.py:72); the PyTorch check is explicitly CUDA and returns false for a CPU-only build (src/mousereach/gpu.py:152), and the TensorFlow check on Windows is CUDA-backed too, with the code warning that TensorFlow past 2.10 has no native Windows card support (src/mousereach/gpu.py:122-130). With no card, the watcher prints the problem and exits (src/mousereach/watcher/cli.py:263, :274-276). The card number to use comes from `watcher.dlc_gpu_device`, default 0 (src/mousereach/config.py:658), and is passed straight to DeepLabCut as its `gputouse` argument (src/mousereach/watcher/orchestrator.py:930, src/mousereach/dlc/core/batch.py:272).

Two gaps in that enforcement are worth knowing:

  - The napari control panel starts the same DLCOrchestrator directly, with no hardware check at all (src/mousereach/watcher/control_widget.py:281-291), and lets the operator override the mode from a dropdown.
  - Even on the command line, if the card check itself throws an error, the failure is downgraded to a log warning and startup continues (src/mousereach/watcher/cli.py:267-269).

The processing-server role has no graphics-card requirement, which is consistent with its own code: when it finds a video that needs DeepLabCut re-run, it explicitly hands it back rather than doing it, with the comment "can't do it here (no CUDA)" (src/mousereach/watcher/orchestrator.py:1494-1497).

4. THE SETTINGS THAT DECIDE ALL OF THIS

All of these live in ~/.mousereach/config.json on each machine, and defaults for the four known lab machines are in src/mousereach/setup/lab_profiles.json, matched by hostname and by which drive letters are present.

  - nas_root - the shared pipeline root. Everything above (the front door, the staging folder, the review queues, the final Analyzed folder) hangs off it. If it is unset, the code silently falls back to an old layout, NAS_DRIVE\"! DLC Output" (src/mousereach/config.py:98-102), and records that it did so as NAS_ROOT_ORIGIN.
  - processing_root - the local fast drive where work actually happens (A: on the graphics-card machines, C: on the server).
  - watcher.mode - "dlc_pc" (the default when absent) or "processing_server". Chooses which of the two roles the one watcher program plays.
  - watcher.also_process - true means a graphics-card machine runs the whole pipeline itself instead of handing off; false means it stops after DeepLabCut and stages the result for the server.
  - watcher.dlc_gpu_device - which graphics card number DeepLabCut uses, default 0.
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

Cropping is the LOWEST-priority job the graphics-card machine does. On each cycle it first stages already-finished work back to the network, then runs pose estimation on any single-mouse video waiting for it, and only if both of those queues are empty does it pick up a new collage (watcher/orchestrator.py:530-580). Before starting, it tries to claim the collage in a shared database so two machines cannot crop the same one (watcher/orchestrator.py:720-727).

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

This is where the real behaviour differs from the folder names people use in conversation. Each successful crop is registered in the watcher's bookkeeping database with its mouse identifier, project, cohort, subject, tray type and grid position (watcher/orchestrator.py:769-787). It is then COPIED to PROCESSING_ROOT/DLC_Queue on the machine's own local disk - not to the network (config.py:148, watcher/orchestrator.py:816-824). The path config.py:113 defines, NAS_ROOT/Processing/Single_Animal, is scanned for singles that someone put there by hand, but the watcher's own crop output never lands there. Finally, the local scratch copies of both the collage and all eight crops are deleted (watcher/orchestrator.py:856-859), so from that moment the only copies that exist are the untouched original on the network and the per-mouse files sitting in the local pose-estimation queue.

Re-running a collage

If a collage is picked up again later - its file is still sitting in the intake folder, after all - each child is checked first. A child that has already moved past the earliest stages is left alone rather than being reset and re-queued. Only children in the states "discovered", "validated" or "failed" are re-driven (watcher/orchestrator.py:791-812).

Other ways cropping can be started

The same crop_collage function is also reachable by hand: a command-line tool (video_prep/cli.py:38, 52, 104) and a graphical panel (video_prep/widget.py:264, 284). Those paths write the crops wherever the user points them and can optionally copy them into the pose-estimation queue and move the collages to an archive folder. They are not part of the automatic watcher flow.

---

## 3. Where the cropped videos go, and pose estimation

HOW CROPPING AND POSE ESTIMATION ACTUALLY WORK TODAY

The cast. Two watcher programs exist, and which one a machine runs is decided by a single setting in that machine's own settings file at ~/.mousereach/config.json: watcher.mode. Set to "dlc_pc" the machine runs the DLCOrchestrator (the crop-and-pose watcher); set to "processing_server" it runs the ProcessingOrchestrator (the analysis watcher). This section covers the crop-and-pose watcher only. Three lab machines with graphics cards run it (DLCLabPC, Vid&DLC1PC, Vid&DLC2PC); the analysis server does not.

Two roots. Every path in the system hangs off two settings. nas_root is the shared network folder everyone can see (<nas_root> in this lab). processing_root is the machine's own fast local disk (A:\MouseReach_Pipeline on the GPU machines, <nas_root> on the analysis server). Folders under nas_root are shared; folders under processing_root are private to one machine.

Step 1 - noticing a collage. Every thirty seconds the watcher lists the shared folder Unanalyzed\Multi-Animal. Any new video file whose name parses correctly is recorded in the watcher's database. It is then watched until its size stops changing for a configured number of seconds (watcher.stability_wait_seconds), which is how the system avoids grabbing a recording that is still being written. Only then does it become eligible for cropping.

Step 2 - choosing what to do next. The watcher does exactly one job per cycle, chosen by a fixed priority order: finish videos whose pose is already done, then pose a video that is waiting, and only when nothing at all is in flight does it start cropping the next collage. So cropping is deliberately the lowest priority - the machine drains what it has started before opening a new collage.

Step 3 - cropping. Before touching the file the machine tries to claim the collage in the shared connectome.db database, so two GPU machines watching the same folder cannot crop the same collage twice. It then COPIES the collage from the network into a scratch folder on its own disk (processing_root\watcher_working) and verifies the copy by size. The collage on the network is not moved, renamed or deleted. The scratch copy is cut with ffmpeg into eight fixed rectangles - the recording is a 1920x1080 grid of eight camera views, two rows of four, each cell 480x540. Grid positions whose animal identifier has cohort "00" mean "no mouse here" and are skipped. Each surviving cell is written as its own mp4 named {date}_{animal}_{tray}.mp4, into the same scratch folder.

Step 4 - where the cropped singles go. Each single is registered in the watcher's database and then COPIED (with a size check) into a second local folder: processing_root\DLC_Queue - on the lab GPU machines, A:\MouseReach_Pipeline\DLC_Queue. The scratch copies, and the scratch copy of the collage, are then deleted. Note what this means: the cropped singles do NOT go to any folder called Processing. The settings file does define a shared network folder Processing\Single_Animal, and the hand-run command `mousereach-crop` writes there by default, but the automatic path never uses it. Between cropping and the end of pose estimation, each single exists in exactly one place - one machine's local disk - and is invisible to the rest of the lab. If a single had already been cropped and taken past this point in an earlier pass, it is left alone rather than reset; the two exceptions the code deliberately re-drives are children recorded as "failed" and children recorded as merely "validated", the latter because that state means the copy into DLC_Queue never succeeded and no file exists anywhere.

A gap worth recording: the cropper is meant to leave a small provenance file beside the collage listing which child came from which grid position, but in the automatic path it is written beside the temporary copy in the scratch folder, so it never appears next to the collage on the network. Nothing downstream breaks, because the later retirement sweep falls back to reconstructing the child list from the collage filename.

Step 5 - pose estimation. When a single is sitting in DLC_Queue the watcher runs DeepLabCut on it, one video at a time, and tells DeepLabCut to write its output into that same DLC_Queue folder. The pose files (.h5, plus .csv) therefore land beside the video in DLC_Queue, not in a Processing folder. Which network is used is set in two places, neither of them in the code:
  - watcher.dlc_config_path in the machine's settings file names the DeepLabCut project (A:\AIs\MPSA-LF-2025-10-27\config.yaml on all three lab machines). If it is unset or missing, nothing is posed and the video simply stays queued - it is not marked failed.
  - Which trained network inside that project - DeepLabCut calls these "shuffles" - is resolved in order from: an explicit argument, then watcher.dlc_shuffle in the machine's settings file, then the shuffle number parsed out of the dlc_scorer entry in the shared file pipeline_versions.json at the top of nas_root. No lab machine sets dlc_shuffle, and that shared file currently declares DLC_resnet101_MPSAOct27shuffle3_100000, so shuffle 3 is what runs. In this project shuffle 1 is the older resnet50 network (Model 3.1) and shuffle 3 is the resnet101 network (Model 4.0). If none of the three sources names a shuffle, the code refuses to pose at all rather than accept DeepLabCut's silent default of shuffle 1, and it stops posing on that machine instead of failing the video, because an unresolvable model is a machine problem and not a video problem. After each video the code reads the model name out of the pose filename DeepLabCut just wrote and rejects the result if it is not the declared one, so a mismatched model cannot be handed downstream. Both safeguards exist because the watcher once quietly produced Model 3.1 pose while the rest of the pipeline was calibrated for 4.0.
The watcher also re-checks the queue folder each cycle for pose files that appeared without its noticing - for instance after a crash mid-run - and records those videos as posed.

Step 6 - what happens after pose, and the setting that decides it. watcher.also_process in the machine's settings file controls this:
  - also_process = false: the machine is finished. Every file belonging to that video - the mp4, the .h5, the .csv - is MOVED out of the local DLC_Queue onto the shared network folder Processing\DLC_Complete, and the video is recorded as done-on-this-machine. Some other machine picks it up from there and runs the analysis steps. This is the arrangement the plain description assumes.
  - also_process = true: the machine keeps the work. It runs segmentation, reach detection, outcome detection and reach assignment itself, in place, in the same local DLC_Queue folder, and then archives the results straight to the network under Analyzed\{project}\{cohort}, deleting its local copies. Nothing is ever staged to Processing\DLC_Complete.
All three lab GPU machines are configured with also_process = true, so today the crop-and-pose machines in fact run the entire pipeline, and the "hand off to the server" route is configured but idle. That is the largest single gap between the described design and the running system.

Step 7 - what eventually happens to the collage. It stays in Unanalyzed\Multi-Animal throughout everything above. It is removed only by the OTHER watcher - the analysis-server one - which runs a slow sweep roughly every thirty minutes over that folder. For each collage it works out the set of single-mouse children implied by the filename and checks whether every one of them has reached the final Analyzed output, was processed with the currently declared tool versions, and has no review outstanding. Only if all of that holds for every child does it stamp a completion record and MOVE the collage (and its provenance file, if one exists) to Analyzed\Multi-Animal, which the backup watcher already mirrors to the second storage array. It never deletes. So a collage still sitting in Unanalyzed means at least one of its children is unfinished, out of date, or waiting on a human.

One dead end to be aware of. If a person drops single-mouse videos by hand into the shared folder Processing\Single_Animal, the crop-and-pose watcher does notice them and record them in its database, but its work queue has no branch for videos in that state - it only ever picks up videos recorded as queued-for-pose, posed, in-progress or processed, plus collages recorded as stable. Files placed there are therefore registered and then never worked on. The supported way to inject singles by hand is the `mousereach-process-animal` command, which registers them and copies them into DLC_Queue in the state the queue actually looks for.

---

## 4. What each watcher actually does

HOW THIS PART OF THE PIPELINE ACTUALLY WORKS TODAY

There are two watcher programs in the code, and one machine runs exactly one of them. The choice is made at startup from that machine's own settings file (~/.mousereach/config.json), from the key watcher.mode. The value "dlc_pc" starts the first watcher; the value "processing_server" starts the second. There is no way to run both on one machine (src/mousereach/watcher/cli.py:298).

WHAT THE FIRST WATCHER DOES

The first watcher runs on a machine with a graphics card. Each cycle it scans two folders on the network drive: the collage intake folder (Unanalyzed/Multi-Animal) and the network folder Processing/Single_Animal (src/mousereach/watcher/watcher.py:60-72). Files whose names it cannot parse are moved to a quarantine folder (src/mousereach/watcher/state.py:110-124). It then picks one job per cycle from a five-item priority list (src/mousereach/watcher/orchestrator.py:501-583):

1. Crop one collage. It copies the collage from the network drive to a local scratch folder, cuts it into single-mouse videos, names each one by which mouse it contains, registers each in its local tracking database, and copies each single into a folder called DLC_Queue on its OWN LOCAL DISK - not to the network Processing/Single_Animal folder. The local scratch copies are then deleted; the original collage is left where it was (src/mousereach/watcher/orchestrator.py:711-866, and src/mousereach/config.py:149).

2. Run DeepLabCut on one queued single. Output lands beside the video in the same local DLC_Queue folder (src/mousereach/watcher/orchestrator.py:871-966).

3. What happens next depends entirely on one setting: watcher.also_process (src/mousereach/config.py:678).

   - watcher.also_process = FALSE: the first watcher MOVES the video, its pose file, and the accompanying csv from its local disk into the shared handoff folder on the network drive, Processing/DLC_Complete (the constant DLC_STAGING, src/mousereach/config.py:120; the move is src/mousereach/watcher/orchestrator.py:1246-1313). It marks the video "archived" as far as its own bookkeeping is concerned, and is done with it. This is the case the description assumes.

   - watcher.also_process = TRUE: nothing is ever handed off. The first watcher runs the entire rest of the pipeline itself, on its own machine, using the same code the server would use: segmentation, reach detection, outcome detection, reach assignment, a provenance manifest, the quality-control triage step, the review gate that can divert a video into the human triage queue or the deep-review queue, kinematic feature extraction, a write into the shared connectome database, and finally an archive of the whole bundle into the final Analyzed folder organised by project and cohort (src/mousereach/watcher/orchestrator.py:968-1187 and 1188-1245). This is the configuration shipped for every graphics-card machine in the lab - the lab DLC PC and both behaviour-room PCs all have also_process set to true in src/mousereach/setup/lab_profiles.json:47, :68, :87.

So the statement "cropping and DeepLabCut are all the first watcher does" is not accurate in either configuration. With also_process off it also performs the handoff move; with also_process on it produces the final scientific product.

Two further things the first watcher does that are easy to miss. It USED TO inherit a periodic housekeeping scan from the shared base class, running at startup and then roughly every 30 minutes, which walks the archived videos in its own database, reads each one's manifest out of the final Analyzed folder, and re-labels as "outdated" any video whose recorded tool versions no longer match the declared current versions or whose human review file is newer than its archived kinematics (`ReprocessingScanner.scan`). On a first-watcher machine that had no follow-through: the first watcher's job list has no entry for "outdated" videos, so a video re-labelled this way left the "archived" state and nothing on that machine ever picked it up again. Worse, the state was not inert - it synced to connectome.db, other nodes adopted it during startup recovery, and it came back as a pathless row on a machine that had no file for the video. As of 2026-08-24 the scan is gated on `handles_reprocessing`, which is true only for the second watcher, so the first watcher no longer marks anything outdated. Only the second watcher acts on "outdated" (`ProcessingOrchestrator._get_next_work_item`). Separately, single-mouse videos found sitting in the network Processing/Single_Animal folder are registered in the database at state "validated" (src/mousereach/watcher/state.py:180-188), but "validated" is not one of the states the first watcher's job list selects from, so those videos are recorded and then never processed.

WHAT THE SECOND WATCHER REQUIRES BEFORE IT WILL TOUCH A VIDEO

The second watcher runs on the server. It has exactly one source of new work: the network handoff folder Processing/DLC_Complete. Each cycle it globs that folder for DeepLabCut output files, and for each one it requires a matching video file with the same name to also be present in that same folder before it will register the video at all. If the video's filename does not parse, the video is quarantined instead (src/mousereach/watcher/orchestrator.py:1363-1372; src/mousereach/watcher/state.py:212-300). It does not scan the network Processing/Single_Animal folder, and it does not walk the final Analyzed folder looking for work.

Before copying a registered video in, it takes a claim: it writes a marker file named after the video, containing its own hostname, into a hidden .claims subfolder of the handoff folder, and skips the video if a marker already exists naming a different machine. Markers more than 24 hours old are deleted as leftovers from a crashed machine (src/mousereach/watcher/orchestrator.py:1658-1734, 1746). This claim exists only to stop two PROCESSING machines watching the same handoff folder from grabbing the same video. It is not a check against the first watcher: the first watcher never reads or writes those markers, and the second watcher never reads the first watcher's own collage-level claims, which live in the shared connectome database and are only created by the first watcher (src/mousereach/watcher/orchestrator.py:721-727 versus the second watcher's constructor at :1326-1361, which never builds a coordinator).

There is therefore no explicit "is the other watcher still working on this?" check anywhere. What keeps the two apart is the shape of the handoff: the first watcher holds the files on its own local disk until DeepLabCut has finished, and only then moves them into the shared folder. Two caveats. First, the move is implemented as copy-to-the-final-name then delete the source (src/mousereach/watcher/transfer.py:108-130), and discovery only tests that the names exist - there is no wait for the files to stop growing. A partially copied video is visible under its finished name. The subsequent intake copy compares source and destination sizes and rejects a mismatch (src/mousereach/watcher/transfer.py:85-95), which usually turns such a race into a failed intake rather than a truncated file being analysed, but that is an after-the-fact guard, not a wait. Second, on a machine with also_process = true nothing is ever staged, so no contention arises there at all.

Once a video is claimed, its files are copied - copied, not moved, so the handoff folder keeps a copy - into a folder named Processing on the SERVER'S OWN LOCAL DISK (on this machine, <nas_root>\Processing; the constant is built from the processing_root setting, src/mousereach/config.py:152). All algorithm work happens there. This is a different place from the network Processing zone on Y:, despite the identical folder name.

What the second watcher does NOT check before running the algorithms: it does not verify that the pose file was produced by the currently declared DeepLabCut model. There is no version comparison anywhere in the run-the-algorithms path (src/mousereach/watcher/orchestrator.py:1811-2120). The only thing resembling a version check at this point is tie-breaking: if a video has more than one pose file, the code prefers the one from the declared model, and otherwise takes the newest and logs a warning that the video should be re-posed - but it still proceeds (src/mousereach/pipeline/manifest.py:89-134). Version currency is checked later and after the fact, by the staleness scanner walking already-archived videos in the Analyzed folder.

The second watcher's full job list, in priority order, is: take in a newly discovered video from the handoff folder; run the algorithms on a video sitting in its local Processing folder; archive a finished video to the network drive; and re-run an "outdated" video (src/mousereach/watcher/orchestrator.py:1432-1512). For that last case, if the staleness scanner decided the video genuinely needs a NEW pose, the server cannot help - it has no suitable graphics card. Until 2026-08-24 it pushed the video into the 'dlc_queued' state and left it, which went nowhere: the server's own job list never selects that state, the DLC machine's queue folder is local and invisible to the server, and the only thing that crossed machines was the state itself, through connectome.db, where it became a pathless row that crashed the DLC machine's stager. It now holds those videos in 'outdated', names them once in the log, and lets the videos that CAN be re-run here through - so one un-poseable video no longer blocks the rest of the queue. Otherwise it pulls that video's pose file and video back out of the Analyzed folder into its local Processing folder and re-runs from the earliest stale stage (:1540-1605). Its local Processing folder is also refilled from a third direction: roughly every ten poll cycles it scans the two human review queues and moves any bundle a human has finished clearing back into Processing so the pipeline re-runs it (:1388-1397; src/mousereach/watcher/review_return.py:118-260). Intake is throttled by watcher.max_local_pending (default 200): once that many videos are sitting in the local Processing folder, no new videos are taken in until some are archived or cleared (:1444-1465).

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
Before the algorithms run, the code resolves the pose file through `resolve_pose_input` (moved 2026-08-24 from orchestrator.py to `watcher/locate.py`, which now holds every "is this claimed path a real file?" helper). It tries the path recorded in the watcher's database; if that is blank or is not a real file it globs the working folder for <video id>DLC*.h5. It deliberately asks "is this a file?" rather than "does this path exist?", because an empty string in Python turns into the current directory, which exists — that exact bug fed a directory to the algorithms as a pose file and dumped 723 videos into the human review queue on 2026-08-19 (the comment at orchestrator.py:48-89 records it). No real file means the video is marked failed with "DLC h5 not found" and nothing else runs (orchestrator.py:1837-1839; the graphics-card copy at 992-996). Two other doors have the same guard: the path that pulls a finished video back for re-running gives up if no pose exists under the Analyzed tree (orchestrator.py:1558-1561), and the path that returns a human-cleared video from the review queue refuses to return it rather than re-run it blind (review_return.py:152-159). Note the check is only "a readable file is there" — it does not open the file or judge whether it is complete.

2. IS THE POSE CURRENT? — checked, but only for already-finished videos, and from the manifest, not the file
About every thirty minutes (every sixtieth poll cycle at the thirty-second default; the review-return scan runs on its own much shorter cadence, since it is a cheap queue listdir while this sweep walks the whole archive) the processing side runs a sweep called the ReprocessingScanner (orchestrator.py:1341, 1373-1382). It walks the videos its database lists as 'archived' — videos that already completed and were moved into the final Analyzed output tree (reprocessor.py:97) — and then re-checks the rows already marked 'outdated', un-marking any whose versions compare current again (the two-way door; without it a declaration accident could mark rows forever, which happened twice in August 2026). Rows carrying an explicit `mark_reason` (hand-marked targeted re-runs, e.g. of videos a compatible-version bugfix applies to) are never un-marked; scanner marks write no reason. For each it loads that video's manifest (reprocessor.py:106) and compares it to the version declaration (versions.py:138-195):
- scorer differs -> the video's ANALYSIS is stale, because segments, reaches and outcomes were computed from the older pose. Whether the POSE has to be made again is a separate question, and since 2026-08-24 the scanner asks it: it indexes the pose files in the Analyzed tree once and checks whether a pose from the declared scorer already exists for that video. If it does, the scope is 'segmentation' - every post-DLC stage re-runs against the pose that is already there, and no GPU is used. If it does not, the scope is 'full' and the video genuinely needs re-posing. On the Y: archive as of 2026-08-24 that split was 1,233 videos needing no GPU against 31 that do, so the check is worth roughly 288 GPU-hours;
- one of segmenter / reach detector / outcome detector / kinematic extractor differs -> stale from that stage onward, and the re-run reuses the still-current earlier outputs (reprocessor.py:35-42, orchestrator.py:1846-1853). Exception: a manifest version listed under `compatible_versions` for that stage in `pipeline_versions.json` does NOT stale the video (versions.py) -- this is how a bugfix bump whose output only changes for pathological videos (which get re-marked by hand) avoids marking the whole corpus outdated, the accident that happened twice in August 2026;
- a human review file newer than the kinematics file also triggers a re-run of kinematics alone, so the reviewer's corrections reach the results and the central database (reprocessor.py:117, 175-191).
Videos found stale get the database state 'outdated' plus a 'reprocess_scope' (reprocessor.py:150). The same sweep is what "mousereach-version-check" prints, and "--mark" is the manual way to trigger it (watcher/cli.py:1231-1290).
Three things this does NOT do, which matter:
- It inspects the pose file only to answer "does a pose from the declared model already exist?", and only when the manifest names an older model (added 2026-08-24). It still never opens a pose file or judges its contents. If the manifest is absent the video is counted "no_manifest" and skipped. Where the manifest and the file on disk disagree, the manifest still wins on the question of what produced the current results - which is correct, because that is what it records. It is not corrected in place; the reprocess run rewrites it with the pose that actually ran, which is the only honest way for it to change.
- Videos on their way through for the first time are never checked. The pipeline function itself (orchestrator.py:1811) has no version comparison anywhere in it. A fresh video is analysed with whatever pose the graphics-card machine produced.
- If either side of a comparison is an empty string, that component is treated as current (versions.py:159, 167, 183). An unset "dlc_scorer" in pipeline_versions.json makes the pose-currency check a silent no-op with no warning.
Videos deliberately locked for a publication are put in the state 'crystallized', not 'archived', so the sweep never sees them (versions.py:242, reprocessor.py:164).
One related safeguard falls short of a check: when a video has pose files from two models side by side, select_pose_file prefers the one matching the declared current scorer; if none matches it takes the newest, logs a warning that the video should be re-posed, and proceeds anyway (manifest.py:89-135).

Bench-sheet disagreement routing is NOT part of the watcher (it was, briefly, 2026-08-25 to 2026-08-28). Comparing pipeline outcomes with hand-scored sheets is an integrator's job -- a database tool that holds both -- and MouseReach must run with no such tool present. What MouseReach provides is the generic command `mousereach-route-to-queue` (watcher/route_cli.py): flag specific segments of an archived video and move it into the triage or deep-review queue with a recorded reason, through the same route_to_queue the pipeline uses itself. Whatever decides a video needs a person asks through that command; the watcher never reaches into another tool's environment or database. Likewise the central-database sync (sync/database.py) is configuration (`central_db`), absent by default, and cross-node coordination lives in MouseReach's own watcher_central.db on the NAS root.

3. WHAT HAPPENS TO THE OLD FILES - superseded, not overwritten (since 2026-08-21)
For a video marked outdated with a narrow scope, the second watcher copies every file whose name starts with the video identifier from Analyzed down to the local Processing folder — a copy, so the originals stay where they are (orchestrator.py:1574-1577). The algorithms re-run and overwrite the copies. Then the finished set is moved back to Analyzed/<project>/<cohort> by archive_video. That move used to be a bare shutil.move onto the same filename, which silently replaced whatever was there, destroying the previous generation's segments, reaches, outcomes, assignments, triage record, manifest and kinematics with nothing kept. Since commit 627877d (2026-08-21) archive_video calls supersede_video_outputs FIRST whenever anything for that video is already at the destination: the earlier generation is swept, checksum-verified, into the versioned Archive tree, and if that sweep reports any failure the archive REFUSES to proceed rather than move new files on top of old ones. The video itself and any ground-truth or human-review file are deliberately left in place, so a review still travels with its video.
The old pose file is the one exception, and only by accident: it is never moved or deleted. Because the model name is part of the filename, a new-model pose lands beside the old one and both stay.
For a video marked outdated because the pose model changed AND which has no pose from the declared model on disk (scope 'full'), the processing server does not do the copy-down. Before 2026-08-24 it forced the state to 'dlc_queued'; it now leaves the video in 'outdated' and logs the list once, because the forced state reached no graphics-card machine and did reach cross-node recovery, where it produced pathless rows. Videos whose manifest names an old model but which DO have the declared pose on disk are no longer in this category at all: they get scope 'segmentation' and take the ordinary copy-down route below.
Be careful with the word "archive" in this code: the watcher's archive step means "move the finished outputs from the working folder into Analyzed/<project>/<cohort>", which is the live current-results location, not a historical store.

4. THE ARCHIVING CODE THE DESCRIPTION DESCRIBES DOES EXIST - and the watcher now calls it (since 2026-08-21)
mousereach/archive/supersede.py does exactly what the description says should happen. supersede_video_outputs (supersede.py:161) moves a video's about-to-be-replaced pose and algorithm outputs into a separate Archive tree at <pipeline root>/Archive, organised as "DLC Model <generation>/<seg…_reach…_out…_asn… version stack>/" so one pose is stored once per model generation while many algorithm variants live under it. Every move is checksum-verified: copy, compare sha256, and only then delete the source (supersede.py:105-128). It never overwrites — an identical file already there means the incoming copy is simply dropped, a same-named but different file is saved as .1, .2 and so on. The video itself, the ground-truth file and the human review file are explicitly never archived (supersede.py:55-57, 204-206), matching the rule that human judgements are facts about the video and stay with it.
It now has two callers: archive_video (archive/core.py), which is the watcher's own filing step, and pipeline/reprocess_to_current.py:293-294, a hand-run bring-one-video-up-to-the-current-stack tool with no command-line entry point. The sentence that stood here -- that nothing automatic called it -- was true until 2026-08-21 and is not now. It has to be invoked by hand, and only does the archiving when called with finalize=True and only for videos the review gate judged clean; held videos are staged into a review queue instead and their old outputs are left untouched (reprocess_to_current.py:66-108, 280-300).

NET EFFECT
Existence of the pose file is properly guarded on every route into the algorithms. Currency of the pose is checked only as a background sweep over already-finished videos, from a text record rather than the file, and can be inert if the version declaration leaves the scorer blank. When a re-run happens inside the watcher, the previous generation's results are now swept into the versioned Archive tree first, checksum-verified, and the archive refuses to proceed if that sweep fails -- so what an earlier model produced for a video remains recoverable. That was not true before 2026-08-21, when the previous generation was overwritten in place.

---

## 6. Reprocessing, and what happens to human review files

HOW SENDING AN OUTDATED VIDEO BACK FOR REPROCESSING ACTUALLY WORKS TODAY\n\nVocabulary first, because the folder names in the code are not the words people use.\n- \"Processing\" in the reprocessing code means the machine's OWN local scratch folder: <processing_root>/Processing, where processing_root comes from that machine's ~/.mousereach/config.json (config.py:56-57, 152). There is also a shared network folder called Processing that holds the cropped single-mouse videos, the post-pose staging area, the failures folder and the two human-review queues (config.py:118, 124, 134-136). These are different places with the same name.\n- \"Analyzed\" is the finished-work tree on the network, organised by project and cohort (config.py:123). In practice it holds two kinds of folder side by side: cohort folders such as Analyzed/Connectome/CNT03/ containing the video, its algorithm outputs and any human files, and a pose-only tree Analyzed/Connectome/DLC Model 4/<cohort>/ containing the current-model pose files from a bulk pose job.\n- Each machine keeps its own small SQLite bookkeeping database at <processing_root>/watcher.db (db.py:116-119), with a summary exported to a shared one on the network after each archive (orchestrator.py:2146).\n\nSTEP 1 - DECIDING A VIDEO IS OUT OF DATE\nA scanner walks every video the local database calls 'archived' and reads that video's provenance file, <video>_processing_manifest.json, out of the Analyzed tree (reprocessor.py:100-113, 193-217). It compares that file against the declared current versions in MouseReach_Pipeline/pipeline_versions.json (versions.py:138-200). A video is out of date if the pose model differs, if any tracked stage's recorded version differs, or if a stage recorded no version at all. A video that is fully current is ALSO pulled back if a saved human review is newer than its kinematics output, so a reviewer's answer reaches the final data (reprocessor.py:114-136, 175-191).\n\nThe scanner writes nothing to disk. It sets the database state to 'outdated' and stores a \"scope\": 'full' when the pose model changed, otherwise the earliest stale stage - segmentation, reach, outcome or kinematics - so that stage and everything after it re-run while current upstream results are reused (reprocessor.py:26-42, 149-157). This scan runs at watcher start-up, then on a timer (orchestrator.py:216-241), and again every tenth poll cycle on the processing-server watcher (orchestrator.py:1378-1387). It can also be run by hand with `mousereach-version-check --mark` (cli.py:1275-1285).\n\nSTEP 2 - WHAT HAPPENS TO AN OUTDATED VIDEO\nThe processing-server watcher picks up outdated videos as its fourth priority, behind archiving, intake, and running the pipeline (order changed 2026-08-30: archiving moved from last to first so finished results reach the NAS continuously instead of only when the staged supply runs dry).\n\nIf the scope is 'full', nothing is copied or moved and the video stays 'outdated'. Since 2026-08-24 'full' means only "this video has no pose from the declared model anywhere in the archive" - a manifest naming an old model is not enough on its own, because the bulk re-pose had already produced current pose files for most of them. The row is no longer forced to 'dlc_queued': that reached no graphics-card machine (the DLC queue folder is local to each node) and did reach cross-node recovery, where the state became a pathless row that crashed the stager. Videos in this state are listed in the log and wait for a person to put them in front of a GPU. On a DLC machine the pose handler now resolves the video file rather than trusting the recorded path, and a video with no file on that node is recorded 'unresolvable' instead of failed-and-retried.\n\nIf the scope is anything else, the watcher searches the whole Analyzed tree for that video's pose files, picks one (preferring the declared current model, else the newest - manifest.py:90-124), and copies every file in THAT POSE FILE'S OWN FOLDER whose name begins with the video id into the local Processing folder, verifying every copy (orchestrator.py:1557-1577). Nothing is deleted from Analyzed - this is a copy, so the archived set stays intact until a re-archive overwrites it. The state is forced to 'processing' and the standard algorithm run starts immediately, skipping the stages upstream of the stale one (orchestrator.py:1581-1602, 1843-1853).\n\nA practical consequence of copying from the pose file's folder rather than the video's folder: because the declared current pose model in this pipeline is the resnet101/shuffle3 model, and those pose files live in the separate pose-only tree, for most videos the source folder contains only pose files. The video file, the older pose, and any human files beside the video are not copied into Processing.\n\nA second, entirely separate route exists for bringing videos current by hand: pipeline/reprocess_to_current.py. It never uses Processing at all - it copies the pose into a temporary folder, runs the algorithms there, and on request moves the results back beside the video, first moving the previous generation's outputs into a version-stamped Archive tree (reprocess_to_current.py:156-160, 285-321).\n\nSTEP 3 - HUMAN FILES\nThere are two human file types, not three. Ground truth is <video>_unified_ground_truth.json. Both the causal review tool and the fast triage review tool write the same file, <video>_causal_review.json - a segment counts as resolved when that file's record for it says it was reviewed (triage_status.py:16-18). The separate <video>_triage.json is the algorithm's own quality verdict, not a human file.\n\nNothing in any reprocessing path deletes either human file -- but until 2026-08-24 that sentence was true and still misleading. Nothing had to delete a review to lose it. The review was written into the triage bundle, the bundle is transient, and returning a cleared bundle MOVES its files onto one node's LOCAL disk and removes the directory. Shared storage then held no copy until the video was archived. Measured that day across 1,686 reviewed videos: 662 had a durable copy, 983 existed only on one machine's local disk, and 41 existed only inside a Y: bundle, one reprocess from gone. Reviews are now written to {NAS}/review_records/reviews/ at save time, before the working copy, and that copy belongs to no bundle's lifecycle. The version-driven reprocess only copies. The archiving step moves every file starting with the video id from Processing into the cohort folder, so human files travel with everything else (archive/core.py:144-190). The one routine that deliberately clears out old outputs - the versioned \"supersede\" move used by the manual bring-current tool - carries an explicit never-touch list containing the ground-truth file, the causal-review file and the video itself, and records them as kept (supersede.py:49-57, 131-136, 197-215). The automatic watcher never calls that routine, which is a separate reason human files are safe there. The only deletions in this area are queue bookkeeping files when a cleared bundle leaves a review queue (review_return.py:36-37, 166-170), staging copies after archiving (orchestrator.py:1225-1231, 2151-2158), and a manual reset helper that removes only algorithm outputs (pipeline/core.py:311-333).\n\nDo the human files travel with the video? Only on one of the two return routes. When a person clears a video that was held in the triage or deep-review queue, every file in that bundle - review file, ground truth, clearance marker - is MOVED into the local Processing folder, minus the two bookkeeping files, and the video is set back to 'processing' (review_return.py:118-214). On the version-staleness route they are copied only if they happen to sit in the pose file's folder, which usually they do not.\n\nThat mostly does not matter, because the code goes looking for the review rather than relying on it having travelled. Both the gate that decides whether a video may proceed and the kinematics step search for <video>_causal_review.json in the working folder, in that video's triage-queue bundle, and in the folder holding the canonical video, taking the newest hit (causal_review_io.py:52-87; review_gate.py:104-117; orchestrator.py:2067-2073). Ground truth is looked up differently: through an index built by scanning only the improvement working area and the network Processing folder (causal_review_io.py:384-386, 419-429). The Analyzed tree is not scanned, so ground truth archived beside a video is invisible to the gate's \"fully human-certified\" shortcut unless a copy also exists under one of those two roots.\n\nSTEP 4 - THE RE-RUN ITSELF\nThe re-run regenerates the algorithm outputs from scratch; it does not preserve the human flags that were written into the old algorithm files. The human decision lives in the review file and is re-applied at the very end: the gate treats a segment as resolved if the review says so, and the kinematics step passes the review file to the feature extractor so the human's outcome and chosen causal reach replace the algorithm's, with the algorithm's originals kept alongside for provenance (review_gate.py:96-101; orchestrator.py:2067-2078; causal_review_io.py:298-345). If the video comes out clean it goes to 'processed', gets archived into Analyzed/<project>/<cohort>, and its kinematics are pushed into connectome.db by a per-video replace. If anything is still unresolved, the whole bundle is moved back out to the triage or deep-review queue and nothing reaches kinematics or the database (review_gate.py:191-222).\n\nTHE ONE THING THAT IS NOT TRUE YET -- CORRECTED 2026-08-24\nThis section used to say that a review is re-attached by segment number and that nothing reads \"segment_span\" back. That was true when it was written and is no longer. `index_review_by_segment` (causal_review_io.py) matches a review to the CURRENT segmentation on frame overlap, ignoring the numbers: it needs 50% overlap, requires the best candidate to beat the runner-up by 15%, and DROPS a review whose frames straddle two segments or match none, saying so in its notes rather than guessing. `resolve_truth_layers` threads the current segments into it, and the kinematics extractor calls that. So a re-cut no longer stamps a human judgement onto footage nobody looked at.\n\nWhat WAS still broken, and was fixed the same day: the resolver looked for a review only in the two NAS review queues and the caller's processing dir -- all three places a review can vanish from. A review that outlived its bundle but whose video was not archived yet was in none of them, so the extractor fell back to the algorithm's answer while the human's sat in the durable store unread. The durable store is now the lowest-priority layer in that stack (any live copy still wins, since a reviewer may have edited it). Pinned by a test that fails against the previous code with the human's outcome discarded.\n\nThe corpus index also now stores segment_span, the human's chosen causal reach and their answers, so a review whose file is lost is reconstructable; it previously held a file path plus two summary fields. A module that guards human decisions across a re-run does exist, and reports rather than silently drops decisions whose segment vanished (clear_guard.py:1-18), but it is wired only into the bundle re-staging tool (staging.py:340), not into the watcher pipeline. So today, a re-segmentation that renumbers segments can silently re-point an old review at different footage.\n\nCONFIGURATION THAT CHANGES ANY OF THIS\n- watcher.mode ('dlc_pc' or 'processing_server', config.py:676): selects which watcher class runs. Only the processing-server watcher acts on 'outdated' videos and returns cleared review bundles; only the DLC watcher consumes 'dlc_queued'.\n- watcher.also_process (config.py:678): on a pose machine, run the algorithms locally straight after posing and archive directly, instead of staging for another machine.\n- processing_root and nas_root in ~/.mousereach/config.json (config.py:56-57, 99-105): set the local scratch Processing folder and the network tree respectively. If nas_root is left unset the code silently falls back to an old layout and every derived path - staging, Analyzed, the review queues - points somewhere else.\n- watcher.dlc_shuffle (config.py:665-667): if unset, the pose model used is taken from the declared dlc_scorer in pipeline_versions.json, so producer and version checker cannot disagree.\n- MouseReach_Pipeline/pipeline_versions.json: the declared current versions. Editing it is what makes existing videos outdated on the next scan.

---

## 7. Running the algorithms, and what happens when one fails

HOW THE ALGORITHMS ACTUALLY GET RUN, AND WHAT HAPPENS WHEN ONE DOES NOT SUCCEED\n(code root: <repo>, branch master)\n\nWHICH MACHINE RUNS THE ALGORITHMS\nEach machine has a file ~/.mousereach/config.json with a \"watcher\" section. Two settings decide the behaviour here:\n- watcher.mode: \"dlc_pc\" (the GPU machines that crop collages and run DeepLabCut) or \"processing_server\".\n- watcher.also_process: only meaningful on a GPU machine. False means \"after posing the video, put it on the NAS in Processing/DLC_Complete and let the server take it\". True means \"after posing the video, run the whole analysis here as well\".\nThe lab's three GPU machine profiles all ship with also_process: true (src/mousereach/setup/lab_profiles.json:41, :65, :89); this processing server's config has mode \"processing_server\" and no also_process key, which defaults to false (src/mousereach/config.py:678). So both machine roles contain a full copy of the analysis sequence: ProcessingOrchestrator._run_pipeline (orchestrator.py:1811) on the server, and DLCOrchestrator._run_local_pipeline (orchestrator.py:968) on a GPU machine. The two are near-identical, step for step.\n\nTHE SEQUENCE, IN ORDER\nBefore anything runs, the code resolves the DeepLabCut pose file for real: it uses the recorded path only if it is an actual file, otherwise it searches the working directory for {video}DLC*.h5, and returns nothing rather than a placeholder. If there is no pose file the video is marked 'failed' and the run stops (orchestrator.py:1830-1841). This guard exists because an empty path once resolved to the current directory, which \"exists\", and 723 videos were pushed into the human review folder in two hours as a result.\n\n1. Segmentation - splits the video into per-pellet segments (orchestrator.py:1856).\n2. Reach detection (orchestrator.py:1909).\n3. Outcome detection - what happened to each pellet (orchestrator.py:1939). Skipped entirely if the filename says tray type E or F.\n4. Reach assignment - which reach caused that outcome (orchestrator.py:1973). Also skipped for tray types E and F. It runs before the gate on purpose, because the gate treats \"pellet was touched but no reach was credited\" as a question for a human.\n5. A provenance manifest recording which tool and model versions produced the outputs (orchestrator.py:1989).\n6. An automatic quality check across all the outputs, which returns either 'auto approved' or 'needs review' and stamps that verdict into each output file plus a {video}_triage.json record (orchestrator.py:2009).\n7. The gate (orchestrator.py:2047, implemented in src/mousereach/watcher/review_gate.py). Nothing reaches kinematics or the database until the gate says clean.\n8. Only if clean: kinematic feature extraction, applying any saved human review corrections, then a write into connectome.db (orchestrator.py:2057-2103). The video is then marked 'processed' and later archived.\n\nOn a re-run triggered by a tool version change, the code can start from the first stale stage and reuse the earlier outputs instead of recomputing them (orchestrator.py:1846-1854).\n\nWHAT THE GATE DECIDES (review_gate.py:68-101)\n- If ground truth marks the whole video as exhaustively scored by a human, it is clean regardless of anything the algorithms flagged.\n- Else, if segmentation is unusable, DEEP REVIEW.\n- Else, if the automatic quality check said 'needs review', DEEP REVIEW.\n- Else, if any segment is still flagged and unanswered, TRIAGE.\n- Else, clean.\nA segment counts as flagged when the outcome cascade marked it \"triaged\" or set flagged_for_review, or when the outcome says the pellet was touched (retrieved, displaced in the scoring area, displaced outside) but no reach in the assignment file is marked causal (src/mousereach/review/triage_status.py:70-95).\n\nWHERE HELD VIDEOS GO\nBoth queues live on the NAS so any machine and the review tools can see them (src/mousereach/config.py:134-136):\n- Triage queue: Processing/Review/triage\n- Deep review queue: Processing/Review/flagged_for_review\nThe whole bundle is MOVED out of the local Processing directory into a folder named after the video: the mp4, the pose file, and every algorithm output, plus a {video}_routing.json audit record and a {video}_manifest.json that lets the review tool open the video in place (review_routing.py:73-149, review_gate.py:120-138). The database state becomes 'triage' or 'deep_review'; if that state change is rejected it is forced, because the files have already moved and the disk is the truth (review_gate.py:158-176).\n\nWHAT HAPPENS WHEN A STEP DOES NOT SUCCEED - THIS IS NOT THE SAME AS TRIAGE\n- Segmentation raises an exception, or the pose file cannot be opened at all: state 'failed'. Nothing moves. This is treated as an infrastructure problem, deliberately kept away from human reviewers (orchestrator.py:1881-1887, and the reasoning at orchestrator.py:93).\n- Segmentation runs but reports unusable boundaries: DEEP REVIEW (orchestrator.py:1892-1901).\n- Reach detection or outcome detection raises: state 'failed' and the error is re-raised; the outer dispatcher logs it and marks 'failed' again (orchestrator.py:1933-1937, :1965-1969, :1514-1538).\n- Reach assignment raises: a log warning, and the pipeline continues (orchestrator.py:1986). If no assignment file was written, the \"touched pellet with no causal reach\" check is skipped, so such a video can pass the gate as clean and be written to the database with no causal-reach attribution.\n- The automatic quality check raises: a warning; the verdict stays 'auto approved' and the video proceeds (orchestrator.py:2039).\n- Kinematic feature extraction raises: a warning only. The video is still marked 'processed' and will be archived - with no kinematics and nothing written to connectome.db (orchestrator.py:2105-2110).\n- The database write fails: a warning only (orchestrator.py:2102).\nNo watcher ever picks up a video in state 'failed' - none of the work queues query that state (orchestrator.py:1432-1512, :501-580). A person has to run mousereach-watch-reprocess, optionally with --all-failed, to reset them (src/mousereach/watcher/cli.py:540-640).\n\nGETTING A TRIAGED VIDEO BACK INTO THE PIPELINE\nThe review tool, opened on a bundle, saves the human's answers as {video}_causal_review.json inside that bundle folder (src/mousereach/review/causal_review_widget.py:880-885, :2843-2850). The {video}_triage.json file in the same folder is the automatic quality check's record and plays no part in release.\n\nA watcher in processing-server mode calls the return scan from its scan step every tenth polling cycle - roughly every five minutes at the default 30-second poll - and returns at most 10 bundles per scan so a deep queue does not starve actual processing (orchestrator.py:1388-1397, :1348, src/mousereach/watcher/review_return.py:222). A watcher in dlc_pc mode never calls it (its scan step, orchestrator.py:483-496, does only file discovery and DeepLabCut completion checks), so a GPU machine running with also_process true can put videos into triage but cannot take them out.\n\nFor each bundle in the triage queue the scan recomputes the picture from the bundle's own files and releases it only when there is at least one flagged segment, every flagged segment has an answer whose \"reviewed\" field is not false, and segmentation still reads as sound (review_return.py:238-249). Before touching anything it makes sure the video has a row in this machine's watcher database, registering it if the review was cleared on another machine (review_return.py:134-146). It then locates the mp4 and pose file - in the bundle, then via the bundle manifest, then by searching the Analyzed archive - and refuses to return the bundle if no pose file can be found, so a cleared review is not spent on a run guaranteed to fail (review_return.py:40-95, :152-159). The data files are moved into the local Processing directory, the two bookkeeping files are deleted, the state is set to 'processing' (forced if the normal transition is rejected), and the now-empty bundle folder is removed (review_return.py:161-214).\n\nThe returned video then re-runs the whole sequence from segmentation. The human's review file came back with it, so the gate sees the answers and lets it through, and feature extraction loads that same file and applies the reviewer's corrections before computing kinematics (orchestrator.py:2066-2078). Deep-review bundles come back the same way, but the release signal is different: either a {video}_deep_review_cleared.json marker written by the deep tools, or a {video}_unified_ground_truth.json produced by the ground-truth tool (review_return.py:108-115, :252-259).\n\nTWO EDGES WORTH KNOWING\n- The gate treats a segment as resolved if ground truth determined it, or if a review file found in any of several locations answers it. The return scan is stricter: it only reads {video}_causal_review.json inside the bundle and does not consider ground truth. A video whose flagged segments include ground-truth-determined ones can therefore sit in the queue after the reviewer has answered everything actually asked of them (review_gate.py:91-100 vs review_return.py:244-249).\n- The server's scan step returns immediately if no DeepLabCut staging directory is configured (orchestrator.py:1364-1366), and that staging path only exists when nas_root is set. With nas_root missing, the review-return scan never runs at all.

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
