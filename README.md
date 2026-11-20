# MouseReach - Mouse Reaching Behavior Analysis

Complete pipeline for analyzing mouse single-pellet reaching behavior videos.

---

## Quick Start

```bash
conda activate mousereach
mousereach                           # Launch all tools
mousereach "path/to/video.mp4"       # Launch with video pre-loaded
mousereach-review-tool               # Review the triage queue
```

---

## Installation

```bash
# Clone or download the repository, then:
cd /path/to/MouseReach

# Create conda environment
conda create -n mousereach python=3.10
conda activate mousereach

# Install
pip install -e .

# Configure paths (REQUIRED - tells MouseReach where your data lives)
mousereach-setup
```

After installation, just type `mousereach` to launch.

---

## Pipeline Overview

```
Video Capture -> Crop -> DLC -> Segment -> Reaches -> Outcomes -> Features -> Export
    Step 0      Step 1   Step 2   Step 3     Step 4     Step 5      Step 6
```

| Step | Name | Input | Output |
|------|------|-------|--------|
| 0 | **Video Prep** | 8-cam collage | Single-animal .mp4 |
| 1 | **DLC Processing** | .mp4 video | DLC .h5 tracking |
| 2 | **Segmentation** | DLC .h5 | `_segments.json` (21 boundaries) |
| 3 | **Reach Detection** | Segments + DLC | `_reaches.json` (reach events) |
| 4 | **Pellet Outcomes** | Segments + DLC | `_pellet_outcomes.json` (R/D/U) |
| 5 | **Grasp Kinematics** | Reaches + Outcomes + DLC | `_features.json` (per-reach kinematic features; see REACH_KINEMATIC_DATA_DICTIONARY.md) |
| 6 | **Export** | All results | Excel/CSV |

---

## GUI Tools

Launch with `mousereach` command. Available tabs:

| Tab | Purpose |
|-----|---------|
| **Pipeline Dashboard** | Overview of all videos and pipeline status |
| **0 - Crop Videos** | Crop multi-animal collage to single animals |
| **1 - DLC Analysis** | Run DeepLabCut pose estimation |
| **2 - Run Pipeline** | Run the pipeline by hand: one pass of this machine's watcher over everything waiting, in this machine's role. On the processing machine: the same steps, checks and review queues as the automatic watcher, and finished videos are filed. On a pose-only graphics-card PC: cutting and posing only (the analysis runs on the processing machine). The tab says which. No watcher needs to be running. "Run chosen videos..." runs the video files you pick now, through the same steps (`mousereach-run-videos`); on a single machine it poses a video first and cuts a multi-animal recording first. A filed, held or crystallized video is not run, with the reason shown. A box puts one animal's videos first (the watcher's own priority, kept until cleared) |
| **Review Queues** | Buttons: Open Triage Review, Open Re-segmentation Review, Open Re-segmentation, Release finished videos -> back into pipeline, Show and resolve..., Refresh counts |
| **4 - View Features** | Visualize kinematic features |
| **GT Tool** | Create ground truth files (for evaluation; cuts, reaches and outcomes you determine there are also used as the video's results) |
| **Watcher Control** | Run, monitor and configure this machine's watcher |
| **Quarantine** | See files held because their names could not be read, correct a name and release the file back to the intake |
| **Setup** | Set this machine's processing folder and the shared pipeline folder |
| **DLC Compare** | Compare two DLC models on a video |
| **FP/FN Review** | Browse false/true positive and missed reach events (read-only) |

(The napari Plugins menu lists the same tools under slightly different titles, e.g. "Step 0 - Crop Videos".)

### Where a correction is saved

| Kind of correction | Tool | Saved to |
|------|---------|----------|
| A pellet's outcome or causal reach | Review window (`mousereach-review-tool`, or Review Queues tab) | `{video}_causal_review.json` (a review record the watcher reads) |
| Segment cuts | Segmentation Fixer (`mousereach-fix-segmentation`), or the boundary review screen (`mousereach-segment-review`) | `{video}_segments.json`, marked as a person's cuts (both save the same way) |
| Reach frames, or a full frame-by-frame record | GT Tool (`mousereach-gt`) | `{video}_unified_ground_truth.json` |
| Sign off reaches or outcomes by hand (no watcher) | Per-step review screen (`mousereach-review-reaches`, `mousereach-review-pellet-outcomes`) | That step's own file, marked `validation_status: "validated"` |

The old "Step 3 - Review Tool" tab is gone (`mousereach --reviews` and `mousereach-review-legacy` print a note instead): it saved corrections into all three algorithm output files, where the next run erased them and the watcher never saw them.

The boundary review screen (`mousereach-segment-review`) saves a person's cuts exactly as the Segmentation Fixer does: they are kept by every later segmentation run, the reaches, outcomes and kinematics computed from the old cuts are archived and removed (run those steps again), and answers given on the old cuts are asked again. It writes nothing for a filed or publication-locked video or one in the triage queue, and says why (docs/REVIEW_TOOLS.md, "Review Tool (retired) and the per-step review screens").

The other per-step review commands (`mousereach-review-reaches`, `mousereach-review-pellet-outcomes`) still save into that step's own file and mark it validated. That is how a person running the steps by hand, without the watcher, signs off a step triage flagged. Know its limit: a later run of the same step writes the file again from the algorithm, so a correction saved there lasts only until that step is run again. With the watcher, review from the Review Queues tab instead.

---

## CLI Commands Reference

### Main Launcher

| Command | Description |
|---------|-------------|
| `mousereach` | Launch the napari GUI |

### Configuration & Setup

| Command | Description |
|---------|-------------|
| `mousereach-setup` | Interactive setup wizard for paths |
| `mousereach-setup --set-role NAME` | Declare this PC's role |
| `mousereach-setup --list-roles` | Show available machine roles |
| `mousereach-setup --show` | Show current configuration |
| `mousereach-fix-powershell` | Fix PowerShell execution policy |

### Pipeline Index (Fast Startup)

| Command | Description |
|---------|-------------|
| `mousereach-index-rebuild` | Rebuild video index from scratch |
| `mousereach-index-status` | Check index status |
| `mousereach-index-refresh` | Refresh changed folders only |

---

## Step 0 - Video Preparation

| Command | Description |
|---------|-------------|
| `mousereach-crop` | Crop multi-animal collage to single animals |
| `mousereach-convert` | Convert video formats (MKV to MP4) |
| `mousereach-prep` | Full video preparation pipeline |
| `mousereach-compress` | Compress videos for storage |

```bash
mousereach-crop -i collage.mkv -o output/
mousereach-convert video.mkv
mousereach-compress folder/      # writes <video>_preview.mp4 beside each video
```

---

## Step 1 - DLC Processing

| Command | Description |
|---------|-------------|
| `mousereach-dlc-batch` | Run DLC on multiple videos |
| `mousereach-dlc-quality` | Check DLC output quality |

```bash
mousereach-dlc-batch -i DLC_Queue/ -o Processing/
mousereach-dlc-quality video.h5
```

---

## Step 2 - Segmentation (Trial Boundaries)

| Command | Description |
|---------|-------------|
| `mousereach-segment` | Run segmentation |
| `mousereach-triage` | Auto-classify confidence |
| `mousereach-advance` | Not needed, and currently stops with an error (see USER_GUIDE); the next step reads `validation_status` itself |
| `mousereach-segment-review` | Launch review GUI |
| `mousereach-reject-tray` | Retired: prints that Easy and Flat videos are processed by the watcher and moves nothing |

```bash
mousereach-segment -i Processing/
mousereach-triage -i Processing/
mousereach-segment-review
```

---

## Step 3 - Reach Detection

| Command | Description |
|---------|-------------|
| `mousereach-detect-reaches` | Run reach detection |
| `mousereach-triage-reaches` | Auto-classify confidence |
| `mousereach-advance-reaches` | Not needed, and currently stops with an error (see USER_GUIDE); the next step reads `validation_status` itself |
| `mousereach-review-reaches` | Launch review GUI |

```bash
mousereach-detect-reaches -i Processing/
mousereach-review-reaches --reaches video_reaches.json
```

---

## Step 4 - Pellet Outcomes

| Command | Description |
|---------|-------------|
| `mousereach-detect-outcomes` | Run outcome detection |
| `mousereach-triage-outcomes` | Auto-classify confidence |
| `mousereach-advance-outcomes` | Not needed, and currently stops with an error (see USER_GUIDE); `mousereach-archive` reads `validation_status` itself |
| `mousereach-review-pellet-outcomes` | Launch review GUI |

```bash
mousereach-detect-outcomes -i Processing/
mousereach-review-pellet-outcomes --outcomes video_pellet_outcomes.json
```

---

## Review & Ground Truth Tools

| Command | Description |
|---------|-------------|
| `mousereach-review-tool` | Review window: answer triaged segments (saves a review record) |
| `mousereach-fix-segmentation` | Segmentation Fixer: correct a video's cuts |
| `mousereach-unified-review` | Launch GT Tool (creates GT files) |
| `mousereach-review-legacy` | Retired old Review Tool: prints which tool to use and stops |
| `mousereach-triage-clearing` | Retired: prints "use mousereach-review-tool" and stops |
| `mousereach-migrate-gt` | Migrate old GT to unified format |

```bash
# Answer the triage queue
mousereach-review-tool

# Create ground truth for evaluation
mousereach-unified-review
```

---

## Step 5 - Grasp Kinematics

| Command | Description |
|---------|-------------|
| `mousereach-grasp-analyze` | Extract kinematic features |
| `mousereach-grasp-triage` | Auto-classify quality |
| `mousereach-grasp-review FEATURES_JSON` | Review one video's feature extraction (the *_features.json path is required) |

---

## Step 6 - Export

| Command | Description |
|---------|-------------|
| `mousereach-results` | The FILED results at reach, pellet, session, day or animal level (CSV or Excel) -- also the **Export results** tab |
| `mousereach-export` | Export one folder you name to Excel/CSV |
| `mousereach-summary` | Generate summary statistics for one folder you name |

```bash
mousereach-results --level session                 # every filed video, one row per tray
mousereach-results --level reach --format xlsx     # one row per reach, with a sheet of column meanings
```

Filing moves a video's results out of Processing into `Analyzed/<project>/<cohort>/`,
so the filed results are read from there: `mousereach-results` walks that folder.
Levels, counting rules (success rate = retrieved / pellets presented; a pellet with
no reach counts) and every column: `docs/RESULTS_EXPORT.md`.
`mousereach-export -i <folder>` and `mousereach-summary -i <folder>` still read one
flat folder you name.

Both read each video's `<video>_features.json` -- the result with every
person's answer (review, ground truth) applied -- never the algorithm's own
outcome file. A video without a features file yet (its kinematics have not run,
or were held) is left out, and the command names it. The same holds for
`mousereach-quick-summary`, `mousereach-real-kinematics` and
`mousereach-aspa-sync`; `mousereach-reach-export` still gives basic rows for
such a video and names them as the algorithm's own calls.

---

## Automated Watcher

The watcher runs on every pipeline machine and does that machine's part: on a
graphics-card PC (`watcher.mode` dlc_pc) it cuts new collages and poses the
singles, handing them to the shared Processing/Posed folder (or, with
`watcher.also_process` true, analyses and files them itself); on the processing
machine (processing_server) it runs segmentation, reaches, outcomes, the
hand-score check, kinematics and filing.

| Command | Description |
|---------|-------------|
| `mousereach-watch` | Start the watcher daemon |
| `mousereach-watch --once` | Process all pending work, then exit |
| `mousereach-watch --dry-run` | Show what would be processed |
| `mousereach-watch-status` | Show pipeline status (video/collage counts, recent activity) |
| `mousereach-watch-status --by-animal` | Per-animal breakdown with QC |
| `mousereach-watch-status --log N` | Show last N processing log entries |
| `mousereach-watch-toggle` | Pause/resume the watcher by hand (flips the current state) |
| `mousereach-watch-toggle --pause` / `--resume` | Pause by hand / remove the hand pause (safe to run twice) |
| `mousereach-watch-toggle --status` | Show whether the watcher is paused, and why |
| `mousereach-watch-recorders` | Show the recording programs that pause this PC's watcher |
| `mousereach-watch-recorders --add recorder.exe` | Pause the watcher whenever that program is open |
| `mousereach-watch-info` | Diagnose drives and path accessibility |

**If the shared drive drops, the watchers wait and resume on their own.** Nothing
needs stopping, restarting or remapping: while the shared pipeline folder does not
answer, a watcher does no shared-drive work, marks nothing failed, keeps what it
holds, and still obeys its stop flag; `mousereach-watch-status` shows it as
"waiting". A watcher started while the drive is away waits for it instead of
exiting. Details: docs/PIPELINE_AS_BUILT.md, "a shared-drive outage is a pause".

**Pausing while recording.** A PC that both records videos and runs the
watcher can pause itself: list the recording program's name (as Task Manager's
Details tab shows it) with `mousereach-watch-recorders --add <name>` or in the
Watcher Control panel. While that program is open the watcher starts no new
work and stops a pose part-way (the video is posed again later, not marked
failed); a collage crop already running is stopped part-way too (its half-made singles are deleted and the collage is cropped again later). Work resumes by itself
a set time (default 120 s) after the program closes. Resume cannot override an
open recording program. The list is empty by default, which means the watcher
never checks for any program.

Set when recording can happen at all (`--days Mon-Fri --hours 08:00-18:00`, or
the panel's "Recording happens only on" boxes) and an open recording program
is ignored outside that time -- for a program left open after recording.
Empty by default (any day, any hour). See INSTALL.md (A2).

The watcher runs on every pipeline machine (GPU machines that cut and pose, and the
processing machine that analyses). Set up with `mousereach-setup --set-role "<profile name>"`
then `mousereach-setup`.

### Auto-Launch on Login

The `deploy/` folder contains batch scripts that auto-detect the MouseReach
environment and work on any PC:

1. **Copy startup script** → `Win+R` → `shell:startup` → paste `deploy/MouseReach-Watcher-Startup.bat`
2. **Copy toggle shortcut** → paste `deploy/MouseReach-Toggle.bat` to Desktop

On login, the startup script will:
1. Wait for the shared pipeline folder (`MOUSEREACH_NAS_ROOT`, if set) to be available
2. Launch the watcher daemon in its own terminal window
3. Open a live view of the watcher's log. This needs `MOUSEREACH_PROCESSING_ROOT` (or
   `MOUSEREACH_NAS_ROOT`) set as a user environment variable; without either it prints
   "WARNING: Could not find watcher.log - skipping log window" and opens no log window.

Both windows close by themselves when the watcher stops cleanly, so a restart never
leaves old windows behind (each used to stay open and keep the log file in use). If the
watcher stops with an error, its window stays open on "Press any key" so the error can be
read. After updating MouseReach, copy the script into the Startup folder again: the copy
there is what runs at login.

The **MouseReach Toggle** shortcut lets you quickly pause/resume processing
(e.g., during filming sessions).

### Checked against the hand scores before triage

Filing a video means it is done. Where a lab scores every tray by hand at the
bench, the watcher compares each video with its session's hand scores right
after analysing it -- before it decides whether the video needs a person
(triage or re-segmentation) or goes on to kinematics and filing. That way a
reviewer sees every question about a video in one visit, and no video is
filed before it has been compared:

- **Every pellet agrees** (or a person already answered it in review): the
  video carries on exactly as it would without the check.
- **A pellet disagrees**: that pellet becomes a triage question, and the video
  goes to the **triage** queue together with any other question it has. In
  the review tool its banner reads, for example, *hand score mismatch (hand:
  retrieved, algo: displaced_sa)*, and the outcome list opens on the
  algorithm's answer (the same as for a video sent back by `mousereach-route-to-queue`). Answer it as
  usual; when every flagged pellet is answered the video comes back, finishes
  and is filed. What a person answers is taken as the truth, whatever the
  bench says.
- **The session has no hand scores**, in a project whose every session is
  scored by hand (`required_projects` below): the video is analysed and then
  **held** -- it stays in the Processing folder, before triage and kinematics
  -- until the project's sheet is next imported (a lab that enters scores the
  same day may still import them only on a schedule). If that import brings them, the
  video is compared as above and carries on by itself. If it does not, the
  scores are **missing**: every session of such a project is scored by hand
  before its videos enter the pipeline, so the sheet's row for this session is
  absent or misnamed, or the import could not match it to the video. The video
  stays held -- it is never filed without the comparison -- and is shown in
  red as needing a person: fix the sheet -- enter the session's scores, or
  correct its row so the date, animal and tray match the video's (do not
  rename the held video: nothing would then find its files); after the next
  import the video
  is compared and carries on by itself. A cohort with no sheet import at all
  waits for its sheet's first import if it has a sheet; if it has none (or the
  name matches no sheet), its videos are held as missing their scores after
  the next import of every sheet, and the red line then says **no tracking
  sheet was found for the cohort**: add the cohort's tracking sheet to the
  sheets folder, or correct the cohort in the sheet's file name if it is
  misspelled; if it is the video's name that carries the wrong cohort, do not
  rename the held files -- tell whoever maintains the pipeline. After the next
  import the video is compared and carries on by itself. Held videos do not
  stop new videos being taken in.
- **Easy and Flat trays**, and projects not listed as required, carry on
  without the comparison.
- **The check cannot run** (the `hand_scores` path is wrong, the file cannot
  be read, the lab settings file that holds the setting is damaged, or the
  storage drive it is on cannot be reached): the video is held, never passed
  unchecked, and shown in red as needing a person until the setting or the
  file is fixed; it then carries on by itself. A damaged settings file never
  reads as "hand scores not set up". While the lab settings file cannot be
  read, this holds every analysed Pillar video, of any project -- which
  projects need the comparison is itself in that file.

Every time the sheet is read for a video, every pellet's hand score is kept
inside that video's outcomes file (`<video>_pellet_outcomes.json`, key
`hand_scores`), so it travels with the video into the review queue and into
Analyzed. If the watcher is restarted while videos are held, it rebuilds the
list of held videos from its attention list and checks each once more,
reusing the analysis it already has -- unless the restart brought new
algorithm versions, in which case the steps whose outputs are out of date run
again first.

**The review window uses the same scores.** When a reviewer's answer for a
pellet disagrees with both the algorithm and the hand score, a "check your
answer" box asks them to look again before saving; the reviewer's answer
always wins (docs/REVIEW_TOOLS.md, "Check your answer"). The watcher keeps every
pellet's hand score inside the video's own outcomes file when it compares the
video (key `hand_scores` in `<video>_pellet_outcomes.json`), and the review
window reads it from there -- so a reviewer needs no setting at all, and sees the
very score the pipeline compared with. A video analysed before 2026-10-28 has no
kept scores until it is analysed again; for it the box never appears.

**Where you see it.** The **Pipeline Dashboard** (first tab of `mousereach`)
shows, in red at the top, **Needs your attention (N)** with one line per thing
a person must fix (for example a hand-score check that cannot run, or a session
whose hand scores are missing after its sheet was imported) and what to do; it
is hidden when nothing needs a person, and refreshes every 5 minutes and on
Refresh. Videos held for the sheet's next import are counted on a grey line
below it, and the census (`mousereach-census`) prints how many of the videos in
the analysis step are only waiting for hand scores. The
lists behind it are `<NAS root>/attention/<PC name>_hand_scores.json` (rewritten
whenever they change; `"needs_person": true` marks the ones to act on), and the
watcher log says the same thing for each video ("Holding ... before triage and
kinematics"); the video's processing log has one `hand_score_check` row for
the wait.

To turn it on for the whole lab, put this in the lab settings file on the
storage drive, `<storage root>/mousereach_lab_config.json` (see "Lab settings
file" below; a running watcher and an open review window pick up the change):

```json
{
    "hand_scores": {
        "path": "<path to hand_pellet_scores.csv>",
        "required_projects": ["PROJECT_A"]
    }
}
```

A `"hand_scores"` section in a person's own `~/.mousereach/config.json` still
works and wins over the lab file for that person.

The CSV has one row per hand-scored pellet with the columns `video_key` (the
session as MouseReach names its video, e.g. `20250709_ABC0207_P1`), `pellet`
(1-20) and `score` (0 not touched, 1 displaced, 2 retrieved); a JSON file of the
same name beside it gives `last_successful_import` per cohort (`ABC_02`),
`sheets_seen` (the cohorts that have a sheet) and `full_import_at` (the last
import of every sheet), which is how "still waiting for the sheet's import" is
told from "missing" (held, red). A key it lacks counts as not known, and the video waits. Without
`path`, nothing is checked. The JSON file may also carry `scores_sha256`, the
SHA-256 of the CSV it was written with; when it does and the CSV read does not
match it (the writing tool is between replacing the two files), MouseReach reads
both again a few times, a second apart, and if they still do not match the video
is held as a check that could not run (red) until they do. A held video is
checked again whenever EITHER file changes; a video held because its check
could not run is also checked again every 10 minutes, so a passing fault (the
storage drive dropping for a moment) clears by itself.

---

## Archive

| Command | Description |
|---------|-------------|
| `mousereach-archive` | File by hand a video the watcher has passed (refuses while the watcher runs; see docs/CLI_REFERENCE.md) |

---

## Algorithm Evaluation

| Command | Description |
|---------|-------------|
| `mousereach-eval` | Evaluate algorithm vs ground truth |

---

## Performance Tracking

| Command | Description |
|---------|-------------|
| `mousereach-perf` | View performance metrics |
| `mousereach-perf-eval` | Run performance evaluation |
| `mousereach-perf-report` | Generate performance report |

---

## Analysis Dashboard

| Command | Description |
|---------|-------------|
| `mousereach-build-database` | Build the unified reach database (one row per reach, all metadata attached) |
| `mousereach-build-explorer` | Build statistics database |
| `mousereach-explore` | Launch data explorer |

---

## Algorithm Documentation

| Command | Description |
|---------|-------------|
| `mousereach-docs` | Generate algorithm documentation |

---

## Database Sync

| Command | Description |
|---------|-------------|
| `mousereach-sync` | Sync _features.json files to the central SQLite database named by the `central_db` setting (or the MOUSEREACH_CENTRAL_DB environment variable); with neither set it stops with an error and exits 1. `mousereach-grasp-analyze` also syncs each features file it writes. See docs/SYNC_TO_DATABASE.md |
| `mousereach-sync --force` | Re-sync all files (delete and re-import) |
| `mousereach-sync-status` | Show sync status and reach counts |
| (no separate command) | Every sync that writes at least one file also rewrites `reach_data.csv` in a `database_dump` folder beside the central database |
| `mousereach-sync-watch` | Watch for new features files and auto-sync (currently stops with an error on start; see docs/SYNC_TO_DATABASE.md) |

---

## Output Files

Each processed video produces these files in Processing/:

| File | Contents |
|------|----------|
| `*DLC*.h5` | DeepLabCut tracking data |
| `*_segments.json` | Trial boundaries (pellet presentations) |
| `*_reaches.json` | Detected reaching movements |
| `*_pellet_outcomes.json` | Pellet outcome classifications |
| `*_features.json` | Kinematic features per reach (Step 5) |
| `*_unified_ground_truth.json` | Human-verified ground truth (optional) |

---

## Configuration

MouseReach requires configuration before first use:

```bash
# 1. Declare this PC's role (pre-fills defaults for your lab's setup)
mousereach-setup --set-role "GPU PC"

# 2. Run the setup wizard (paths pre-filled, just hit Enter)
mousereach-setup
```

### Machine Roles

Each PC in the lab serves a specific role. Setting the role tells the wizard
what defaults to use for this machine. The roles are the profiles in your lab's
profiles file (below); the template's are:

| Role | Description | Runs Watcher? |
|------|-------------|---------------|
| **Single machine** | The lab's only MouseReach PC: cuts, poses, analyses, takes reviewed videos back, files | Yes (`standalone`) |
| **Recording + GPU PC** | Records videos; outside recording cuts and poses them on its GPU | Yes (`dlc_pc`) |
| **GPU PC** | Cuts and poses videos on its GPU | Yes (`dlc_pc`) |
| **Processing Server** | Analyses posed videos (reach detection, outcomes, kinematics) and files them | Yes (`processing_server`) |

```bash
mousereach-setup --list-roles      # See available roles
mousereach-setup --set-role NAME   # Set this PC's role
mousereach-setup --show            # Show current configuration
```

The role is saved to `~/.mousereach/machine_role.json`. If no role is set,
the wizard matches the lab's profiles by hostname and drive pattern, or
prompts manually.

A lab describes its machines in its own profiles file: copy
`src/mousereach/setup/lab_profiles.json.example` (a template, never used
as-is), edit it, and either put it at `~/.mousereach/lab_profiles.json` on
each machine or keep one shared copy and set `MOUSEREACH_LAB_PROFILES` to it.
See INSTALL.md, Step 6.

### Configuration File

| Setting | Purpose | Required? |
|---------|---------|-----------|
| `processing_root` | This machine's local working folders (DLC_Queue, Processing) | **Yes** |
| `nas_root` | The shared pipeline folder (Unanalyzed, Processing\Failed, Analyzed, review_records, the lab settings file); unset means each tool stops and names mousereach-setup | **Yes** |
| `nas_drive` | Older setting, still read by mousereach-aspa-import / mousereach-aspa-sync and by mousereach-watch-process-animal's file search | Optional |
| `watcher.*` | Automated pipeline settings (DLC config, GPU, polling) | For watcher PC |
| `watcher.db_path` | The watcher database file. On the processing server every login's settings must carry BOTH `watcher.mode` = `processing_server` AND this setting: a login whose settings say `processing_server` but name no file is refused by the watcher and every tool, rather than guess. The refusal keys on `watcher.mode` -- a login whose settings carry no `watcher.mode` is taken for a GPU PC and uses `<processing_root>/watcher.db` WITHOUT a refusal, so on the processing server check that every login has the mode set. On a GPU PC it defaults to `<processing_root>/watcher.db` | Processing server: **Yes** (with `watcher.mode`) |
| `hand_scores.*` | The hand-scores file, and the projects whose sessions must all have hand scores, checked right after a video is analysed, before triage (see "Checked against the hand scores before triage"). Usually kept in the lab settings file instead (below) | Optional |

Configuration is saved to `~/.mousereach/config.json` and persists across sessions.

### Lab settings file

Settings that are the same for everyone in a lab (today: `hand_scores` and
`filing`) can be written once in `<storage root>/mousereach_lab_config.json`,
where the storage root is the folder `nas_root` names. Every PC and person
finds it from `nas_root`; nothing is set per PC or per user. For each setting
MouseReach uses the person's own `config.json` if it has that key (the whole
section), else the lab file, else leaves it unset -- except `filing`, which is
read from the lab file only. A missing lab file leaves the setting unset. A lab
file that exists but cannot be read (a typing mistake in it), or a storage drive
that cannot be reached, is never read as "unset" for `hand_scores` and `filing`:
videos that need the hand-score check are held, in red, and filing keeps the
rules it last read (or stops, with the reason in the watcher log, if it has
none) until the file is fixed. INSTALL.md, Part 6, Step 4 shows how to create it.

**Filed is final.** Once a video is filed, its cuts, reaches, pellet outcomes
and causal reaches are its final answer: a newer algorithm or pose model is used
only for videos not filed yet, and nothing automatic ever sends a filed video
back to a review queue. Only its kinematics may be recalculated, and that never
holds the video or asks you anything. You can still reopen one filed video on
purpose (`mousereach-route-to-queue`, the dashboard's Re-run); its filed results
stay in Analyzed until the new run is filed and replaces them. To try a new
algorithm version on filed videos, use the evaluation tools -- they never write
into Analyzed. (docs/PIPELINE_AS_BUILT.md, "Update 2026-11-19 (filed is final)".)

`filing` says which folder under `Analyzed` each project's finished videos go
in, and how its cohort folders are named -- for example every project of one
grant inside the grant's folder, one folder per cohort
(`Analyzed/<Grant>/PROJECT_A/PROJECT_A_03/`). A project with no line files in
`Analyzed/<CODE>/<CODE>/`. Every lookup of a filed video (the version scan,
reconcile, the review tools) asks the same rule, so they find the video at any
depth.

---

## Data Flow

```
DLC_Queue/           -> Videos waiting for DLC processing
Processing/          -> ALL active videos + analysis files (v2.3+ unified)
<nas_root>/Processing/Failed/ -> Processing errors (on the shared drive, not in this machine's folder)
_finished_elsewhere/ -> Local copies set aside by the watcher (see below)
```

**`_finished_elsewhere/<date>/<folder>/`** -- when any machine's watcher
starts (a graphics-card PC or the processing machine), it looks in its own `DLC_Queue/` and `Processing/` for copies of
videos that are already finished: the machine's record says so AND the video's
results (its `_processing_manifest.json`) are in `Analyzed/`. Those copies are
moved here, never deleted, and the watcher log says how many and where. A copy
whose results cannot be found in `Analyzed/` is left where it is, whatever the
record says.

The processing watcher also moves files here while it runs, in three cases:
the working files of a filed video whose re-run failed for good (the video
goes back to "archived" and its row names this folder); the copies made for a
re-run that could not start because one file would not copy; and a pose or
video left over when a run was refused because the video is in deep review.
In the first two cases everything moved is either a copy of a file that is
also somewhere else (in `Analyzed/`, or, for a review's answers, in the durable
review store) or an output of the run that failed. A video's only copy of
anything a person made is never moved here by those two. A video can be
set aside more than once on the same day (a try whose copy failed, then the
run that failed for good); the second set then goes into its own sub-folder,
`<video>_again_<time>/`, inside the dated folder, so it never replaces the
first set's files, and the video's reason names the sub-folder. Four kinds of file
count as made by a person: cuts set by hand (`_segments.json` saying
`"boundary_source": "human"`), ground truth (any file with `ground_truth` in
its name), the deep-review "cleared" marker (`_deep_review_cleared.json`) and
a review's answers (`_causal_review.json`). Before moving anything the watcher
compares each one in `Processing/` with the same-named file in the video's
`Analyzed/` folder (a review's answers also with the copy in the durable
review store). If any of them has no identical copy, nothing of that video is
moved: it is left as "failed" with all its files in `Processing/`, and its
reason in `mousereach-watch-status` names the file and says it "was made by a
person and has no other copy".

**Before clearing a dated folder,** check each video in it: its
`_processing_manifest.json` is in `Analyzed/`, and the watcher does not list
it as failed (`mousereach-watch-status`). Then look at the file names: if the
folder holds a `_segments.json`, a file with `ground_truth` in its name, a
`_deep_review_cleared.json` or a `_causal_review.json` for a video, open the
video's `Analyzed/` folder and check the same file is there (a
`_causal_review.json` may instead be in the durable review store,
`<NAS root>/review_records/reviews/`). The watcher made that comparison
before it moved the file; this second look costs a minute and a person's work
cannot be made again. If all of that holds for every video in the folder, it
can be deleted to free space. If you are not sure about one, leave that dated
folder.

**Note:** v2.3+ uses a single Processing/ folder. Status tracked via `validation_status`
field in JSON files ("needs_review", "auto_approved", "validated").

---

## Getting Help

Most commands support `--help`:

```bash
mousereach-segment --help
mousereach-detect-reaches --help
mousereach-export --help
```

---

## Requirements

- Python 3.9-3.10
- NumPy, Pandas, h5py
- Napari (GUI)
- OpenCV (video processing)
- DeepLabCut (pose estimation)
- TensorFlow 2.10 (pinned >=2.10,<2.11; NumPy pinned 1.26.4)

---

## Acknowledgments

- **[DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)** - Markerless pose estimation ([Mathis Lab](http://www.mackenziemathislab.org/))
  > Mathis, A., Mamidanna, P., et al. (2018). DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. *Nature Neuroscience*, 21, 1281-1289.
- **[napari](https://napari.org/)** - Multi-dimensional image viewer
- **[Claude](https://anthropic.com/)** - AI-assisted development (Anthropic)

---

*MouseReach v2.16.0-dev*
*Developed by Logan Friedrich*
