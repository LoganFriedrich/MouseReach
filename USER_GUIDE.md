# MouseReach User Guide

A step-by-step guide to processing mouse reaching behavior videos from raw footage to final data export.

---

## Quick Start

The fastest way to use MouseReach:

```bash
conda activate mousereach
mousereach
```

This launches all MouseReach tools in a single napari window with tabs. You can:
- Switch between tools using tabs
- Load a video once, it's shared across all tools
- Work through the pipeline step-by-step

**With a video:**
```bash
mousereach "path/to/your/video.mp4"
```

**Reviewing:** open the **Review Queues** tab, or run `mousereach-review-tool`.
(`mousereach --reviews` used to open review tabs that saved into the
algorithm's files; it now prints a note and opens the usual tabs.)

---

## Overview

MouseReach processes multi-animal reaching videos through a 7-step pipeline:

```
Multi-Animal Video → Crop → DLC → Segment → Reaches → Outcomes → Features → Export
     (collage)       Step0  Step1   Step2    Step3     Step4      Step5     Step6
```

Each step produces output files that feed into the next step. Human review checkpoints ensure data quality.

**Key Features:**
- Unified launcher (`mousereach` command) with all tools in tabs
- Shared video layer (load once, not 3x per tool)
- Cross-tool awareness (save in one step, downstream steps auto-refresh)
- Drag-and-drop video support
- Single-folder architecture (all files in Processing/, status in JSON)

---

## Prerequisites

Before starting, ensure you have:

1. **MouseReach installed** - See `INSTALL.md`
2. **FFmpeg installed** - Required for video cropping
3. **Trained DLC model** - Your 18-point model for pose estimation (Reference, SATL, SABL, SABR, SATR, BOXL, BOXR, Pellet, Pillar, RightHand, RHLeft, RHOut, RHRight, Nose, RightEar, LeftEar, LeftFoot, TailBase)
4. **Multi-animal collage video(s)** - 2x4 grid layout (.mkv format)

---

## Step 0: Video Preparation

**Goal:** Split 8-camera collage into individual animal videos.

### Input
Multi-animal collage video with filename format:
```
20250101_CNT0101,CNT0102,CNT0103,CNT0104,CNT0105,CNT0106,CNT0107,CNT0108_P1.mkv
├─────── ───────────────────────────────────────────────────────────── ──┤
  Date              8 Animal IDs (positions 1-8)                      Session
```

The collage layout is 2 rows × 4 columns:
```
┌────┬────┬────┬────┐
│ 1  │ 2  │ 3  │ 4  │  ← Top row
├────┼────┼────┼────┤
│ 5  │ 6  │ 7  │ 8  │  ← Bottom row
└────┴────┴────┴────┘
```

### Process

**Option A: Crop a single video**
```bash
mousereach-crop -i path/to/collage.mkv -o output_folder/
```

**Option B: Crop all videos in a directory**
```bash
mousereach-crop -i /path/to/Unanalyzed/Multi-Animal/ -o /path/to/Unanalyzed/Single_Animal/
```

**Option C: Crop and hand the videos on for pose**
```bash
mousereach-crop -i input_folder/ -o output_folder/ --queue
# --queue copies each cut video into the shared singles folder, where a
# graphics-card PC's watcher claims and poses it. Nothing goes into a DLC queue.
```

### Output
Individual .mp4 files for each animal:
```
20250101_CNT0101_P1.mp4
20250101_CNT0102_P1.mp4
20250101_CNT0103_P1.mp4
... (8 files per collage)
```

### Notes
- Animal ID with cohort "00" (e.g., CNT0001) are skipped (blank positions)
- Requires FFmpeg in PATH

---

## Step 1: DeepLabCut Processing

**Goal:** Generate pose tracking data from individual videos.

### Input
- Single-animal .mp4 videos from Step 0
- Your trained DLC project config

### Process

Run DLC analysis using your preferred method:

**Option A: DeepLabCut GUI**
```bash
python -m deeplabcut
# Use Analyze Videos tab
```

**Option B: Python script**
```python
import deeplabcut
deeplabcut.analyze_videos('path/to/config.yaml', ['video1.mp4', 'video2.mp4'])
```

**Option C: MouseReach batch command**
```bash
mousereach-dlc-batch -i videos/ -c path/to/config.yaml
```

### Quality Check
```bash
mousereach-dlc-quality *.h5 -o quality_reports/
```

This checks:
- Reference point stability (BOXL, BOXR corners)
- Tracking coverage (% high-confidence frames)
- Critical point reliability (hand markers)

### Output
DLC tracking files:
```
20250101_CNT0101_P1DLC_resnet50_YourModelMar15shuffle1_500000.h5
```

### Notes
- GPU recommended for faster processing
- ~2-5 minutes per video depending on length and hardware

---

## Step 2: Segmentation

**Goal:** Identify the 21 boundary frames that divide each video into 20 pellet delivery segments.

### Input
- DLC .h5 files from Step 1
- Corresponding .mp4 videos (in same directory or locatable)

### Process

**1. Run batch segmentation (v2.3+ single-folder)**
```bash
mousereach-segment -i Processing/
```

**2. Triage results (v2.3+ updates JSON validation_status)**
```bash
mousereach-triage -i Processing/
```

Updates `validation_status` in JSON:
- `auto_approved` - High confidence, can auto-advance
- `needs_review` - Requires human verification

**3. Review flagged files (Napari)**
```bash
napari
```
Then: **Plugins -> MouseReach - Mouse Reaching Behavior Analysis -> Step 2b - Review Boundaries** (or run `mousereach-segment-review`)

- Load a `*_segments.json` file
- Video and boundaries display automatically
- Adjust boundaries as needed
- Save when done

**Keyboard shortcuts:**
| Key | Action |
|-----|--------|
| Space | Set the current boundary to this frame |
| Enter | Accept the current boundary |
| Left/Right | Previous/Next frame |
| Shift+Left/Right | Jump 10 frames |
| N / P | Next/Previous boundary |
| S | Save |

**4. Go on to the next step**

There is no separate "advance" step. Saving in the review screen marks the
segmentation `validated`, and reach detection (Step 3) reads that mark itself:
it runs on a video whose segmentation reads `validated` or `auto_approved` and
skips one that still reads `needs_review`.

Do not run `mousereach-advance`: on the current version it stops with an error
(`TypeError ... require_validation`) and moves nothing.

What the review screen's save does: it saves the cuts as a person's cuts,
exactly as the Segmentation Fixer does, so running segmentation again keeps
them. Every save, even with no boundary moved, also copies the old
`*_segments.json` and the reaches, outcomes, reach-assignment and kinematics
files of that video into an `_archived` folder and removes them: run reach
detection and the later steps again afterwards. It saves nothing, and says
why, for a video that is filed, locked for a publication, or in the triage
queue. With the watcher running, review from the **Review Queues** tab instead
(cuts are corrected there in the Segmentation Fixer).

### Output
Segmentation JSON files:
```
20250101_CNT0101_P1_segments.json
```

Contains:
- 21 boundary frame numbers
- Confidence scores
- Validation status

---

## Step 3: Reach Detection

**Goal:** Detect individual reaching movements within each pellet segment.

### Input
- Validated segmentation files (`*_segments.json`)
- DLC .h5 files
- Video files

### Process

**1. Run batch detection (v2.3+ single-folder)**
```bash
# Process all videos in Processing/
mousereach-detect-reaches -i Processing/

# Skip videos with ground truth files (for reprocessing)
mousereach-detect-reaches -i Processing/ -s "*_reach_ground_truth.json"
```

**2. Triage results (auto-runs, updates JSON validation_status)**
```bash
mousereach-triage-reaches -i Processing/
```

Updates `validation_status` in JSON:
- `auto_approved` - High confidence
- `needs_review` - Requires review

**3. Review flagged files (Napari)**
```bash
napari
```
Then: **Plugins -> MouseReach - Mouse Reaching Behavior Analysis -> Step 4b - Review Reaches** (or run `mousereach-review-reaches`)

- Load a `*_reaches.json` file
- Reaches displayed as colored overlays
- Edit start/end frames, add/delete reaches

**Keyboard shortcuts:**
| Key | Action |
|-----|--------|
| Space | Play/Pause forward |
| R | Play reverse |
| Left/Right | Previous/Next frame |
| Shift+Left/Right | Jump 10 frames |
| N / P | Next/Previous reach |
| S | Set reach start at current frame |
| E | Set reach end at current frame |
| A | Add new reach |
| Delete | Delete selected reach |
| X | Exclude/include the selected reach |
| Enter | Accept the current reach |
| Esc | Cancel adding a reach |
| Ctrl+S | Save progress |

**Playback speeds:** 0.25x, 0.5x, 1x, 2x, 4x, 8x, 16x (click buttons)

**4. Go on to the next step**

There is no separate "advance" step. Saving in the review screen marks the
reaches file `validated`. Outcome detection (Step 4) does not check that mark;
`mousereach-archive` does: it files a video only when its segments, reaches and
outcomes files all read `validated` or `auto_approved`, it has a
`<video>_features.json` newer than those files with every pellet decided, and --
on a machine whose watcher database knows the video -- the watcher has passed it
(state `processed`).

Do not run `mousereach-advance-reaches`: on the current version it stops with
an error (`TypeError ... require_validation`) and moves nothing.

As with segmentation, re-running reach detection on the video writes
`*_reaches.json` again and the saved correction is gone.

### Output
Reach detection JSON files:
```
20250101_CNT0101_P1_reaches.json
```

Contains per-segment:
- Reach count
- Start/end frames for each reach
- Apex frame (maximum extension)
- Confidence scores

---

## Step 4: Pellet Outcomes

**Goal:** Classify the outcome of each pellet segment.

### Outcome Categories

| Outcome | Description |
|---------|-------------|
| `retrieved` | Mouse grabbed and ate the pellet |
| `displaced_sa` | Pellet knocked into scoring area (not eaten) |
| `displaced_outside` | Pellet knocked outside scoring area |
| `untouched` | Pellet still on pillar at segment end |
| `no_pellet` | No pellet visible at segment start |
| `uncertain` | Could not determine outcome |

### Input
- Validated segmentation files (`*_segments.json`)
- DLC .h5 files
- Video files

### Process

**1. Run batch detection (v2.3+ single-folder)**
```bash
# Process all videos in Processing/
mousereach-detect-outcomes -i Processing/

# Skip videos with ground truth files (for reprocessing)
mousereach-detect-outcomes -i Processing/ -s "*_outcome_ground_truth.json" "*_outcomes_ground_truth.json"
```

**2. Triage results (auto-runs, updates JSON validation_status)**
```bash
mousereach-triage-outcomes -i Processing/
```

Updates `validation_status` in JSON:
- `auto_approved` - High confidence
- `needs_review` - Requires review

**3. Review flagged files (Napari)**
```bash
napari
```
Then: **Plugins -> MouseReach - Mouse Reaching Behavior Analysis -> Step 3b - Review Outcomes** (or run `mousereach-review-pellet-outcomes`)

- Load a `*_pellet_outcomes.json` file
- Review each segment's classification
- Correct misclassifications

**Keyboard shortcuts:**
| Key | Action |
|-----|--------|
| Space | Play/Pause forward |
| Shift+R | Play reverse |
| Left/Right | Previous/Next frame |
| N / P | Next/Previous segment |
| R | Mark as Retrieved |
| D | Mark as Displaced (SA) |
| O | Mark as Displaced (Outside) |
| U | Mark as Untouched |
| I | Set interaction frame |
| K | Set outcome-known frame |
| Ctrl+S | Save progress |

**Playback speeds:** 0.25x, 0.5x, 1x, 2x, 4x, 8x, 16x (click buttons or use menu)

**4. Go on to the next step**

There is no separate "advance" step. Saving in the review screen marks the
outcomes file `validated`, which `mousereach-archive` checks before filing.

Do not run `mousereach-advance-outcomes`: on the current version it stops with
an error (`TypeError ... require_validation`) and moves nothing.

Re-running outcome detection on the video writes `*_pellet_outcomes.json` again
and the saved correction is gone.

### Output
Outcome JSON files:
```
20250101_CNT0101_P1_pellet_outcomes.json
```

Contains per-segment:
- Outcome classification
- Confidence score
- Key frames (interaction, outcome known)
- Flag reasons for uncertain cases

---

## Step 5: Grasp Kinematics

**Goal:** Extract kinematic features from each reach, linked to pellet outcomes.

### Input
- Validated reach files (`*_reaches.json`)
- Validated outcome files (`*_pellet_outcomes.json`)
- DLC .h5 files

### Process

**1. Run batch feature extraction**
```bash
mousereach-grasp-analyze -i Processing/
```

**2. Triage results**
```bash
mousereach-grasp-triage -i Processing/
```

**3. Review features (optional)**
```bash
mousereach-grasp-review Processing/<video>_features.json
```

### Output
Feature files:
```
20250101_CNT0101_P1_features.json
```

Contains per-reach:
- Reach extent (pixels, ruler units, mm)
- Velocity at apex and peak velocity
- Trajectory straightness and smoothness
- Hand angle and rotation
- Contextual features (reach number, causal reach flag)
- Linked outcome classification

---

## Step 6: Export

**Goal:** Generate analysis-ready spreadsheets from processed data.

### Input
- Each video's `<video>_features.json` (written by the kinematics step): the
  result with every person's answer applied. A video that has no features file
  yet is left out of the export, and the command prints its name and why.
  Outcomes are counted with the pipeline's own words (retrieved, displaced_sa,
  displaced_outside, untouched).

### Process

**The filed results (the usual case):** filing moves each finished video into
`Analyzed/<project>/<cohort>/`, so Processing is empty for it afterwards. Export
the filed results with the **Export results** tab, or:
```bash
mousereach-results --level session                # one row per tray
mousereach-results --level reach --format xlsx    # one row per reach
```
Levels: reach, pellet, session, day, animal. Columns and counting rules:
`docs/RESULTS_EXPORT.md`.

**One folder you name** (e.g. a copy of some videos' files):
```bash
mousereach-export -i <folder> -o results.xlsx
mousereach-export -i <folder> -o csv_folder/ --format csv
```

### Output

**Excel file** has two sheets:
- **Summary** - one row per video: n_segments, n_retrieved, mean_confidence,
  total_reaches, n_displaced_sa, n_displaced_outside, n_untouched, and
  success_rate (n_retrieved / n_segments, a fraction, not a percent)
- **Details** - one row per pellet

With `--format csv` only the details are written, as `mousereach_details.csv` in
the output folder.

**Details columns:**
| Column | Description |
|--------|-------------|
| video_name | Source video |
| animal_id | Animal |
| session | Session |
| segment_num | Pellet segment (1-20) |
| outcome | retrieved / displaced_sa / displaced_outside / untouched |
| confidence | Outcome confidence |
| n_reaches | Reaches in segment |
| causal_reach_id | The reach that caused the outcome |

---

## Typical Workflow Summary

### Option A: Unified Launcher (Recommended for Review)

```bash
# Launch all tools in one window
mousereach

# Or with a specific video
mousereach "path/to/video.mp4"

# Ground truth annotation for one video
mousereach-gt "path/to/video.mp4"
```

Review happens from the **Review Queues** tab. The old whole-video Review Tool
that `--reviews` opened is retired: its saves went into the algorithm's own
files, where the next run erased them. The per-step review screens are still
opened on their own (`mousereach-segment-review`, `mousereach-review-reaches`,
`mousereach-review-pellet-outcomes`); see the steps above.

### Option B: CLI Batch Processing

```bash
# 1. Crop collages
mousereach-crop -i Multi-Animal/ -o Single_Animal/ --queue   # hands the cut videos to the pose step (shared singles folder)

# 2. Run DLC (use your preferred method)
# ... DLC processing ...

# 3. Segment (v2.3+ single-folder architecture)
mousereach-segment -i Processing/
# Auto-triage updates validation_status in JSON
# Review flagged videos: mousereach-segment-review (saving marks them validated and
# as a person's cuts, and removes the old reaches/outcomes/kinematics;
# reach detection skips a video whose segmentation still reads needs_review)

# 4. Detect reaches
mousereach-detect-reaches -i Processing/
# Auto-triage updates validation_status in JSON
# Review flagged videos: mousereach-review-reaches (saving marks them validated)

# 5. Classify outcomes
mousereach-detect-outcomes -i Processing/
# Auto-triage updates validation_status in JSON
# Review flagged videos: mousereach-review-pellet-outcomes (saving marks them validated)

# 6. Extract kinematic features
mousereach-grasp-analyze -i Processing/

# 7. Export (the filed results; see docs/RESULTS_EXPORT.md)
mousereach-results --level session
```

---

## Troubleshooting

### "Video file not found"
- Ensure .mp4 is in same directory as .h5/.json files
- Or set video path explicitly in the tool

### Napari plugin not appearing
- Reinstall: `pip install -e .` (from MouseReach root directory)
- Restart Napari completely

### Keybinding conflicts in Napari
- Only load one MouseReach plugin at a time
- Restart Napari when switching between plugins

### DLC tracking looks wrong
- Check DLC model quality with `mousereach-dlc-quality`
- May need to retrain model or filter bad videos

### The watcher will not start

The message, the watcher log and the Dashboard's health line say why. There are
three reasons a start is refused, and each is on purpose:

- **"Not started: a stop request is waiting in ...\watcher_stop.flag (left ...
  ago)"**. That file asks a
  watcher to stop; it is left by a stop or by an update in progress, and while
  it is there every new watcher would stop the moment it starts. Nothing
  deletes it for you. If the stop or update it belongs to is finished, delete
  the file (it is in this machine's processing folder) and start again. From
  the command line the refusal ends with exit code 3, so a scheduled task shows
  it as failed rather than as a success.
- **"A watcher is already running on this PC"**. Only one watcher may run on a
  machine, however it was started (command line, scheduled task, or the
  Watcher Control panel, in any Windows login). Use the one that is running;
  to restart it, press **Restart processor** on the Dashboard.
- **"Not started: 'Run as' is set to ..., but this machine is set up as ..."**.
  The Watcher Control panel runs only the kind of watcher this machine is set
  up for (`mode` in its settings). Put **Run as** back to that. If the
  machine's job has really changed, change **mode** in the settings part of the
  panel, press **Save config**, and start again: Save config also moves **Run as** to the
  mode it saved, so nothing else needs changing.

### Segmentation boundaries incorrect
- Use Napari review tool to manually adjust
- Check that pellet delivery mechanism is visible in video

---

## File Organization (v2.3+ Single-Folder Architecture)

The shared pipeline folder (`nas_root`) has one folder per stage. Each folder is
named for the state the work is in, not for the tool that produced it:

```
<nas_root>\
├── Unanalyzed\
│   ├── Multi-Animal\           # Collages waiting to be cut
│   └── Single_Animal\          # Single-animal videos waiting for a pose
│       └── .inflight\<machine>\  # Videos a machine has taken for pose -- leave them alone
│
├── Processing\
│   ├── Posed\                  # Posed videos waiting for the algorithms
│   ├── Repose_Queue\           # Re-pose request files
│   ├── Review\
│   │   ├── triage\             # Held for a person: per-element questions
│   │   └── deep_review\        # Held for a person: failed segmentation or QC
│   ├── Quarantine\             # Files whose names the watcher cannot read (see below)
│   └── Failed\
│
├── Analyzed\
│   ├── <project>\<cohort>\     # Finished videos
│   └── Archive\                # Superseded outputs, kept for provenance
│
└── Archive\
    └── historical\             # Read-only source material
```

**Dropping new videos:** both `Unanalyzed` folders accept new videos at any time,
even while the watchers are running on several machines. Copy collages into
`Multi-Animal` and cut single-animal videos into `Single_Animal` as soon as they
are recorded. Nothing needs to be stopped first:

- A video is taken only after it has stopped changing for the watcher's
  stability wait (`watcher.stability_wait_seconds`, 60 s by default). A copy
  still in progress is never picked up half-finished.
- Only one machine takes each video. A single that a machine has taken is moved
  into `Single_Animal\.inflight\<machine>\` while that machine poses it.
  **Leave these files alone.** The file there may be the only copy on the
  share. It is deleted automatically once the video has moved on to its next
  folder. If that machine gives the video up, stops running (its status report quiet
  for 3 hours), or nobody touches the file for a day, the video is put back in
  `Single_Animal` for another machine.
- If you copy the same video in again while a machine still holds it, the
  second copy is left where it is and the watcher log says so. Delete the
  second copy.

**On a machine that also records behaviour:** tell it which program you record
with (`mousereach-watch-recorders --add <program>.exe`, or the Watcher Control
panel). From then on that machine does no processing while the recording
program is open -- you do not have to close it, and you do not have to wait:

- Open the recording program and a box says **MouseReach is stopping**. The
  pose or crop running at that moment is stopped within seconds and thrown
  away (the video simply goes back in the queue; nothing is lost but a few
  minutes of machine time).
- A second box says **Safe to record**. That one means nothing of ours is
  running any more: the graphics card and disk are yours.
- Processing starts again on its own a couple of minutes after the recording
  program is closed (`watcher.pause_resume_grace_seconds`).
- To silence the boxes on a machine, set `watcher.notify_safe_to_record` to
  `false` in `~/.mousereach/config.json`. The same information is always in the
  watcher log and in the Watcher Control panel.

Each machine also has a local working folder (`processing_root`):

```
<processing_root>\
├── DLC_Queue\                  # Videos waiting for DLC processing
│
└── Processing\                 # ALL active videos on this machine live here
    ├── video.mp4               # Single-animal video
    ├── videoDLC*.h5            # DLC tracking results
    ├── video_segments.json     # Boundaries (validation_status in JSON)
    ├── video_reaches.json      # Reaches (validation_status in JSON)
    ├── video_pellet_outcomes.json  # Outcomes (validation_status in JSON)
    └── video_features.json     # Kinematic features per reach (Step 5)
```

**Note:** v2.3+ uses JSON `validation_status` field ("needs_review", "auto_approved",
"validated") instead of separate folders. All files for a video stay together in Processing/.


**The Quarantine folder (`Processing\Quarantine`):** the watcher moves a file here,
with a small `.quarantine.json` note saying why, when it cannot read the file's name
(a collage that does not name eight animals, a single without a tray letter, a
single dropped into the multi-animal intake). It never processes a quarantined file.
To get one back, fix its name with the Quarantine tab or
`mousereach-watch-quarantine --release FILE --to single|collage --as NEW_NAME`.

To clear old files out of quarantine, run `mousereach-watch-quarantine --purge` first.
It changes nothing: it lists what it would delete and what it would keep, and why.
Then `mousereach-watch-quarantine --purge --delete` deletes, after you type `y`. The
rules, in plain words:

- **A multi-animal recording (collage) is never deleted.** It is the camera's own
  recording of a whole tray -- the irreplaceable source. Lose it and that piece of
  the experiment is gone for ever. Anything the purge cannot prove to be a
  single-animal video is kept as if it were a collage.
- **A single-animal video is deleted only when the collage it was cut from still
  exists** in the pipeline (`Unanalyzed\Multi-Animal`, or a `Multi-Animal` folder in
  `Analyzed` or `Archive`). A single can always be cut again from its collage; without
  the collage it is the only copy, and it is kept ("only copy -- kept"). A backup
  location is never looked at and does not count.
- A crystallized (publication-locked) video is never touched.
- Each deletion is written to `<nas_root>\watcher_logs\quarantine_purge.jsonl` with
  the collage that was relied on, and the video's "quarantined" rows in the watcher
  databases are taken away so nothing points at a file that is gone. Only the row
  on the machine you run it on goes (plus the shared database's rows); run the
  purge on another node to clear that node's row.

The full details are in `mousereach-watch-quarantine --help` and docs/CLI_REFERENCE.md under `mousereach-watch-quarantine`.

**Old folder names:** `Processing\Single_Animal`, `Processing\DLC_Complete` and
`Processing\Review\flagged_for_review` are retired (now `Unanalyzed\Single_Animal`,
`Processing\Posed` and `Processing\Review\deep_review`). A migrated drive keeps a plain
file at each old path, so old code that tries to use one stops with an error instead of
quietly rebuilding the old folder and working there.

---

## Getting Help

- docs/CLI_REFERENCE.md lists every command and its options
- Use `--help` with any command: `mousereach-segment --help`
- Report issues: on the repository's GitHub Issues page

---

*MouseReach 2.16*
