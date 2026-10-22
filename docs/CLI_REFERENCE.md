# MouseReach CLI Reference

Generated 2026-10-22 by `python -m mousereach.docs.generate_cli_reference` --
every entry below is the command's own `--help` output, harvested from
the installed executables, so this file cannot say something the code
does not. **Do not edit by hand; rerun the generator.**

All commands exist only inside the mousereach conda environment:

```
conda activate mousereach    # or the full path of the env if it was created with --prefix
mousereach-<command> --help
```

`mousereach` alone launches the full napari GUI with every tab; the
commands here run one specific piece without the GUI (or launch one
specific tool window). See docs/REVIEW_TOOLS.md for the operator
walkthroughs of the review tools themselves.

## Contents

- [Main launcher](#main-launcher) -- `mousereach`, `MouseReach`
- [Configuration and setup](#configuration-and-setup) -- `mousereach-setup`, `mousereach-fix-powershell`
- [Pipeline Index (fast startup)](#pipeline-index-fast-startup) -- `mousereach-index-rebuild`, `mousereach-index-status`, `mousereach-index-refresh`
- [Step 0 - Video Prep](#step-0-video-prep) -- `mousereach-crop`, `mousereach-convert`, `mousereach-prep`, `mousereach-compress`
- [Step 1 - DLC](#step-1-dlc) -- `mousereach-dlc-batch`, `mousereach-dlc-quality`, `mousereach-dlc-compare`, `mousereach-fpfn-review`
- [Step 2 - Segmentation](#step-2-segmentation) -- `mousereach-segment`, `mousereach-triage`, `mousereach-advance`, `mousereach-segment-review`, `mousereach-review`, `mousereach-reject-tray`
- [Step 3 - Reach Detection](#step-3-reach-detection) -- `mousereach-detect-reaches`, `mousereach-triage-reaches`, `mousereach-advance-reaches`, `mousereach-review-reaches`
- [Step 4 - Pellet Outcomes](#step-4-pellet-outcomes) -- `mousereach-detect-outcomes`, `mousereach-triage-outcomes`, `mousereach-advance-outcomes`, `mousereach-review-pellet-outcomes`, `mousereach-review-outcomes`
- [Routine triage review: the causal review tool, TRIAGED-ONLY, over the Pending queue](#routine-triage-review-the-causal-review-tool-triaged-only-over-the-pending-queue) -- `mousereach-review-tool`
- [Bulk release check for deep-review bundles (human-invoked; gates on human.outcome)](#bulk-release-check-for-deep-review-bundles-human-invoked-gates-on-human-outcome) -- `mousereach-review-release`
- [Pipeline census: total workload, per-stage breakdown, pace and ETA (headless JSON)](#pipeline-census-total-workload-per-stage-breakdown-pace-and-eta-headless-json) -- `mousereach-census`
- [Put a video into a review queue with a reason (generic; the way an external integrator such as a database tool asks MouseReach to hold a video for a person)](#put-a-video-into-a-review-queue-with-a-reason-generic-the-way-an-external-integrator-such-as-a-database-tool-asks-mousereach-to-hold-a-video-for-a-person) -- `mousereach-route-to-queue`
- [Provenance checks -- what the pipeline actually produces, and whether the documents still describe the code.](#provenance-checks-what-the-pipeline-actually-produces-and-whether-the-documents-still-describe-the-code) -- `mousereach-field-audit`, `mousereach-doc-check`, `mousereach-fix-segmentation`, `mousereach-backfill-manifest-versions`, `mousereach-triage-clearing`, `mousereach-unified-review`, `mousereach-gt`, `mousereach-review-legacy`, `mousereach-migrate-gt`
- [Triage auto-resolve (pre-check before human review): if a triaged segment has a matching unified GT entry, lift the flag and copy the GT outcome. Runs as a step in the normal processing pipeline; only segments without GT remain triaged for the napari review tool.](#triage-auto-resolve-pre-check-before-human-review-if-a-triaged-segment-has-a-matching-unified-gt-entry-lift-the-flag-and-copy-the-gt-outcome-runs-as-a-step-in-the-normal-processing-pipeline-only-segments-without-gt-remain-triaged-for-the-napari-review-tool) -- `mousereach-resolve-triage-from-gt`
- [Step 4b - Reach Assignment (joins v8 reaches + v6 cascade outcomes into per-reach permanent output for kinematic analysis)](#step-4b-reach-assignment-joins-v8-reaches-v6-cascade-outcomes-into-per-reach-permanent-output-for-kinematic-analysis) -- `mousereach-assign-reaches`
- [Step 5 - Grasp Kinematics](#step-5-grasp-kinematics) -- `mousereach-grasp-analyze`, `mousereach-grasp-triage`, `mousereach-grasp-review`
- [Step 6 - Export](#step-6-export) -- `mousereach-export`, `mousereach-summary`
- [Archive - Move validated videos to NAS](#archive-move-validated-videos-to-nas) -- `mousereach-archive`
- [Migration - One-time migration from old multi-folder to v2.3+ single-folder architecture](#migration-one-time-migration-from-old-multi-folder-to-v2-3-single-folder-architecture) -- `mousereach-migrate`
- [Algorithm Evaluation](#algorithm-evaluation) -- `mousereach-eval`, `mousereach-eval-direct`, `mousereach-update-algo-ref`, `mousereach-algo-vs-human`
- [Export Utilities](#export-utilities) -- `mousereach-quick-summary`, `mousereach-features-csv`
- [Kinematic Analysis Utilities](#kinematic-analysis-utilities) -- `mousereach-reach-export`, `mousereach-real-kinematics`
- [Performance Tracking](#performance-tracking) -- `mousereach-perf`, `mousereach-perf-eval`, `mousereach-perf-report`
- [Algorithm Documentation](#algorithm-documentation) -- `mousereach-docs`
- [Analysis Dashboard](#analysis-dashboard) -- `mousereach-build-database`
- [Data Explorer (pre-computed statistics database)](#data-explorer-pre-computed-statistics-database) -- `mousereach-build-explorer`, `mousereach-explore`
- [Database Sync - Automatic sync to central connectome database](#database-sync-automatic-sync-to-central-connectome-database) -- `mousereach-sync`, `mousereach-sync-watch`, `mousereach-sync-status`
- [Watcher - Automated pipeline orchestration](#watcher-automated-pipeline-orchestration) -- `mousereach-watch`, `mousereach-watch-status`, `mousereach-watch-reprocess`, `mousereach-watch-quarantine`, `mousereach-watch-unresolvable`, `mousereach-watch-prioritize`, `mousereach-watch-process-animal`, `mousereach-watch-info`, `mousereach-watch-toggle`, `mousereach-watch-recorders`
- [Version tracking and reprocessing](#version-tracking-and-reprocessing) -- `mousereach-version-check`, `mousereach-reconcile`, `mousereach-aspa-import-collages`, `mousereach-backfill-kinematic-versions`, `mousereach-version-index-build`, `mousereach-version-index-status`, `mousereach-crystallize`, `mousereach-uncrystallize`
- [Backup watcher](#backup-watcher) -- `mousereach-backup`
- [Archive migration - One-time Sort/ -> project/cohort restructure](#archive-migration-one-time-sort-project-cohort-restructure) -- `mousereach-migrate-archive`
- [ASPA reprocessing tools](#aspa-reprocessing-tools) -- `mousereach-aspa-import`, `mousereach-aspa-feed`, `mousereach-aspa-sync`, `mousereach-aspa-compare`, `mousereach-session-coverage`, `mousereach-legacy-inventory`

## Main launcher

### `mousereach`

```
MouseReach Tools Launcher v2.3.0
Loading... (first launch may take 30-60s on network drives)

usage: mousereach [-h] [--step {dashboard,0,1,2,4,2b,3b,4b}] [--reviews]
                  [video]

Launch MouseReach tools

positional arguments:
  video                 Optional video file to auto-load

options:
  -h, --help            show this help message and exit
  --step {dashboard,0,1,2,4,2b,3b,4b}, -s {dashboard,0,1,2,4,2b,3b,4b}
                        Launch specific step(s). Can be used multiple times.
  --reviews, -r         Launch only review tools (2b, 3b, 4b)

MouseReach All Tools Launcher
=============================

Launch all MouseReach tools in a single napari window with tabbed widgets.

ARCHITECTURE OVERVIEW
---------------------
This launcher creates a napari viewer and loads multiple widgets as dock tabs:

    +----------------------------------------------------------------+
    |  napari Viewer Window                                          |
    |  +----------------------+-------------------------------------+|
    |  |                      |  Dock Area (right side)             ||
    |  |   Video Display      |  +---------------------------------+||
    |  |   (shared by all     |  | [Dashboard][0-Crop][1-DLC]...   |||
    |  |    widgets)          |  |                                 |||
    |  |                      |  |   Currently Active Widget       |||
    |  |                      |  |                                 |||
    |  |                      |  +---------------------------------+||
    |  +----------------------+-------------------------------------+|
    +----------------------------------------------------------------+

KEY DESIGN DECISIONS
--------------------
1. SHARED VIDEO: All widgets share ONE video layer in napari. When you load
   a video in any widget, all widgets can access it. This avoids loading the
   same video 3 times for different review steps.

2. STATE MANAGER: MouseReachStateManager coordinates between widgets:
   - Tracks which video is currently active
   - Broadcasts video changes to all widgets
   - Handles cross-widget communication (e.g., "segments validated" -> refresh reaches)

3. TAB ORDER: Widgets load in pipeline order:
   Dashboard -> Step 0 -> Step 1 -> Step 2 -> Step 3 (Review) -> Step 4 -> GT Tool

4. TWO REVIEW TOOLS:
   - "3 - Review Tool" (review_mode=True): Edits algorithm JSON files directly
   - "GT Tool" (review_mode=False): Creates separate ground truth files

ENTRY POINTS
------------
This file provides these CLI commands (defined in pyproject.toml):
    mousereach          - Launch all tools
    MouseReach          - Alias for mousereach

USAGE
-----
    mousereach                              # Launch all tools
    mousereach path/to/video.mp4            # Launch with video pre-loaded
    mousereach --step 2b path/to/video.mp4  # Launch only Step 2b review
    mousereach --reviews                    # Launch only review tools (2b, 3b, 4b)

PIPELINE STEPS
--------------
    0   - Video Prep (crop 8-camera collages to single animals)
    1   - DLC Analysis (run DeepLabCut pose estimation)
    2   - Run Pipeline (batch: Segmentation -> Reaches -> Outcomes)
    3   - Review Tool (fix algorithm mistakes in JSON files)
    4   - View Features (visualize extracted kinematics)
    GT  - Ground Truth Tool (create evaluation datasets)
```

### `MouseReach`

Alias of `mousereach` (same entry point: `mousereach.launcher:main`).

## Configuration and setup

### `mousereach-setup`

```
MouseReach Configuration Setup Wizard

Usage:
    mousereach-setup                   Run interactive configuration wizard
    mousereach-setup --show            Show current configuration
    mousereach-setup --set-role NAME   Declare this PC's role (saves locally)
    mousereach-setup --list-roles      Show available machine roles
    mousereach-setup --help            Show this help message

Machine identification priority:
  1. Local identity file (~/.mousereach/machine_role.json)
     Set with: mousereach-setup --set-role "NAS / DLC PC"
  2. Drive-pattern auto-detection (lab_profiles.json)
  3. Fully manual (user enters all paths)

When a role is identified, the wizard pre-fills defaults from the
matching lab profile. You can override any default during setup.

The wizard asks, in order: the data repository (nas_drive); the shared
pipeline folder (nas_root -- the folder holding Unanalyzed, Processing,
Analyzed and pipeline_versions.json, the same one on every machine); the
local processing root; and the watcher settings. It then offers to create
the pipeline folders inside the shared pipeline folder.

Saving writes only those settings and keeps every other one already in the
config file (backup folders, watcher settings it does not ask about). There
is no guessed pipeline folder: with nas_root unset, every tool that needs
the shared folders stops and says so.

Configuration is saved to: ~\.mousereach\config.json
Identity file: ~\.mousereach\machine_role.json
Lab profiles file: <repo>\src\mousereach\setup\lab_profiles.json
```

### `mousereach-fix-powershell`

```
usage: mousereach-fix-powershell [-h]

Fix PowerShell execution policy for network drives.

This sets the execution policy to Bypass for the current user,
which allows conda activate scripts to run from network drives (Y:, etc.)

options:
  -h, --help  show this help message and exit
```

## Pipeline Index (fast startup)

### `mousereach-index-rebuild`

```
usage: mousereach-index-rebuild [-h] [--quiet]

Rebuild the MouseReach pipeline index

options:
  -h, --help   show this help message and exit
  --quiet, -q  Suppress progress output
```

### `mousereach-index-status`

```
usage: mousereach-index-status [-h]

Show the pipeline index: how many videos are in each folder and how far each
one has got. Read-only.

options:
  -h, --help  show this help message and exit
```

### `mousereach-index-refresh`

```
usage: mousereach-index-refresh [-h] [--all] [folders ...]

Refresh specific folders in the pipeline index

positional arguments:
  folders     Folder names to refresh (default: all stale)

options:
  -h, --help  show this help message and exit
  --all, -a   Refresh all folders
```

## Step 0 - Video Prep

### `mousereach-crop`

```
usage: mousereach-crop [-h] [-i INPUT] [-o OUTPUT] [--queue] [-q]

Crop 8-camera collages into single-animal videos

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input file or directory (default: <path>
                        vior\MouseReach_Pipeline\Unanalyzed\Multi-Animal)
  -o OUTPUT, --output OUTPUT
                        Output directory (default: <path>
                        useReach_Pipeline\Unanalyzed\Single_Animal)
  --queue               Also copy outputs to DLC_Queue
  -q, --quiet
```

### `mousereach-convert`

```
usage: mousereach-convert [-h] [-o OUTPUT_DIR] input [input ...]

Convert MKV to MP4

positional arguments:
  input                 Input MKV file(s)

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
```

### `mousereach-prep`

```
usage: mousereach-prep [-h] [-i INPUT] [--no-queue] [--archive] [-q]

Full video prep workflow

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input Multi-Animal directory
  --no-queue            Don't copy to DLC queue
  --archive             Archive original collages after cropping
  -q, --quiet
```

### `mousereach-compress`

```
usage: mousereach-compress [-h] [--pattern PATTERN] [--scale SCALE]
                           [--crf CRF] [--overwrite]
                           folder

Create compressed preview versions of videos for MouseReach review widgets

positional arguments:
  folder                Folder containing videos to compress

options:
  -h, --help            show this help message and exit
  --pattern PATTERN, -p PATTERN
                        Glob pattern for video files (default: *.mp4)
  --scale SCALE, -s SCALE
                        Resolution scale factor (default: 0.75)
  --crf CRF, -q CRF     FFmpeg CRF quality (18-28, higher=smaller, default:
                        28)
  --overwrite, -f       Overwrite existing preview files
```

## Step 1 - DLC

### `mousereach-dlc-batch`

```
usage: mousereach-dlc-batch [-h] -i INPUT -c CONFIG [-o OUTPUT] [--gpu GPU]
                            [--cpu]

Run DLC on videos

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input directory
  -c CONFIG, --config CONFIG
                        DLC config.yaml or project folder
  -o OUTPUT, --output OUTPUT
                        Output directory
  --gpu GPU             GPU device (default: 0)
  --cpu                 Use CPU instead of GPU
```

### `mousereach-dlc-quality`

```
usage: mousereach-dlc-quality [-h] [-o OUTPUT] h5_files [h5_files ...]

Check DLC output quality

positional arguments:
  h5_files              DLC .h5 files

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output directory for reports
```

### `mousereach-dlc-compare`

```
usage: mousereach-dlc-compare [-h] [--model-a MODEL_A] [--model-b MODEL_B]
                              [--split]
                              [video]

DLC Model Comparison Viewer

positional arguments:
  video                 Video file to load

options:
  -h, --help            show this help message and exit
  --model-a MODEL_A, -a MODEL_A
                        DLC H5 file for Model A
  --model-b MODEL_B, -b MODEL_B
                        DLC H5 file for Model B
  --split               Start in split view mode
```

### `mousereach-fpfn-review`

**No `--help` available** (timeout): --help did not return within 45s (likely a GUI entry point with no argument parsing)

Entry point: `mousereach.fpfn_review.widget:main`. Read its module docstring for usage.

## Step 2 - Segmentation

### `mousereach-segment`

```
usage: mousereach-segment [-h] -i INPUT [-o OUTPUT] [-q] [--no-triage]

Detect trial boundaries in DLC tracking files

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Directory containing DLC .h5 files
  -o OUTPUT, --output OUTPUT
                        Output directory for *_segments.json (default:
                        Processing/ under PROCESSING_ROOT). For quarantined
                        improvement runs, pass -o <same as -i> to write
                        outputs alongside the inputs without moving anything
                        into the production pipeline.
  -q, --quiet           Minimal output (no per-file progress)
  --no-triage           Don't auto-update validation_status in JSON files

Examples: mousereach-segment -i Processing/ # Process all DLC files
mousereach-segment -i Processing/ --no-triage # Don't auto-triage results
```

### `mousereach-triage`

```
usage: mousereach-triage [-h] -i INPUT

Sort segmentation results by confidence (updates validation_status in JSON)

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Directory with *_segments.json files to triage

This is usually run automatically by mousereach-segment. Only use manually if
you need to re-triage existing results.
```

### `mousereach-advance`

```
usage: mousereach-advance [-h] -i INPUT [--force]

Mark validated segmentation results as ready for reach detection

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Source directory (typically Processing/)
  --force               Process all videos even without validation flag

Only processes videos with validation_status = "validated" or "auto_approved".
Use --force to process regardless of status (not recommended for production).
NOTE: In v2.3+ single-folder architecture, files stay in Processing/.
```

### `mousereach-segment-review`

```
usage: mousereach-segment-review [-h] [video]

Launch napari boundary review tool for manual verification

positional arguments:
  video       Video file to auto-load (optional)

options:
  -h, --help  show this help message and exit

Keyboard shortcuts: SPACE Set current boundary to this frame N / P Next /
Previous boundary S Save validated boundaries Left / Right Step 1 frame
Shift+Arrows Step 10 frames
```

### `mousereach-review`

*Deprecated: use mousereach-segment-review*

Alias of `mousereach-segment-review` (same entry point: `mousereach.segmentation.cli:main_review`).

### `mousereach-reject-tray`

```
usage: mousereach-reject-tray [-h] [--dry-run] [--folder FOLDER] [--quiet]

Scan for and reject unsupported tray types (E/F)

options:
  -h, --help            show this help message and exit
  --dry-run, -n         Show what would be done without moving files
  --folder FOLDER, -f FOLDER
                        Folder to scan (default: Processing)
  --quiet, -q           Suppress progress output

Examples:
    mousereach-reject-tray --dry-run  # Preview what would be rejected
    mousereach-reject-tray            # Execute rejection

Supported tray types: P (Pillar)
Unsupported tray types: E (Easy), F (Flat)

Unsupported videos are moved to:
    {NAS_DRIVE}/Unanalyzed/Unsupported_Tray_Type/
```

## Step 3 - Reach Detection

### `mousereach-detect-reaches`

```
usage: mousereach-detect-reaches [-h] -i INPUT [-o OUTPUT] [-q]
                                 [--skip-validation-check]
                                 [-s SKIP_IF_EXISTS [SKIP_IF_EXISTS ...]]

Detect reaching movements in DLC tracking data

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Directory with validated segment files
  -o OUTPUT, --output OUTPUT
                        Output directory (default: same as input)
  -q, --quiet           Minimal output (no per-file progress)
  --skip-validation-check
                        Process even without validated segments (not
                        recommended)
  -s SKIP_IF_EXISTS [SKIP_IF_EXISTS ...], --skip-if-exists SKIP_IF_EXISTS [SKIP_IF_EXISTS ...]
                        Skip videos with files matching patterns (e.g.,
                        '*_reaches.json')

Examples: mousereach-detect-reaches -i Processing/ mousereach-detect-reaches
-i Processing/ -s '*_reach_ground_truth.json'
```

### `mousereach-triage-reaches`

```
usage: mousereach-triage-reaches [-h] -i INPUT

Flag anomalous reach detection results for review

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Directory with *_reaches.json files

Checks for unusual patterns that may indicate algorithm errors. Videos with
anomalies are flagged for human review.
```

### `mousereach-advance-reaches`

```
usage: mousereach-advance-reaches [-h] -i INPUT [--force]

Mark validated reach detection results as ready for outcome detection

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Source directory (typically Processing/)
  --force               Process all videos even without validation flag

Only processes videos with validation_status = "validated". Use --force to
process regardless of status (not recommended). NOTE: In v2.3+ single-folder
architecture, files stay in Processing/.
```

### `mousereach-review-reaches`

```
usage: mousereach-review-reaches [-h] [--reaches REACHES] [--dir DIR]

Review and correct reach detection results

options:
  -h, --help         show this help message and exit
  --reaches REACHES  Single *_reaches.json file to review
  --dir DIR          Directory with multiple reach files to review

Examples: mousereach-review-reaches --reaches video_reaches.json # Review
single file mousereach-review-reaches --dir Processing/ # Review all in dir
Keyboard shortcuts (in review tool): N / P Next / Previous reach S / E Set
reach Start / End to current frame A Add new reach DEL Delete current reach
Space Play/pause video
```

## Step 4 - Pellet Outcomes

### `mousereach-detect-outcomes`

```
usage: mousereach-detect-outcomes [-h] -i INPUT [-o OUTPUT] [-q]
                                  [-s SKIP_IF_EXISTS [SKIP_IF_EXISTS ...]]
                                  [--legacy]

Classify pellet outcomes (retrieved/displaced/untouched)

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Directory with reach detection results
  -o OUTPUT, --output OUTPUT
                        Output directory (default: same as input)
  -q, --quiet           Minimal output (no per-file progress)
  -s SKIP_IF_EXISTS [SKIP_IF_EXISTS ...], --skip-if-exists SKIP_IF_EXISTS [SKIP_IF_EXISTS ...]
                        Skip videos with files matching patterns
  --legacy              Run the legacy geometric detector
                        (core/pellet_outcome.py) instead of the v6 cascade

Examples: mousereach-detect-outcomes -i Processing/ mousereach-detect-outcomes
-i Processing/ --legacy mousereach-detect-outcomes -i Processing/ -s
'*_pellet_outcomes.json'
```

### `mousereach-triage-outcomes`

```
usage: mousereach-triage-outcomes [-h] -i INPUT

Flag low-confidence outcome classifications for review

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Directory with *_pellet_outcomes.json files

Checks for ambiguous or low-confidence outcome classifications. Videos with
issues are flagged for human review.
```

### `mousereach-advance-outcomes`

```
usage: mousereach-advance-outcomes [-h] -i INPUT [--force]

Mark validated outcome results as ready for export

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Source directory (typically Processing/)
  --force               Process all videos even without validation flag

Only processes videos with validation_status = "validated". Use --force to
process regardless of status (not recommended). NOTE: In v2.3+ single-folder
architecture, files stay in Processing/.
```

### `mousereach-review-pellet-outcomes`

```
usage: mousereach-review-pellet-outcomes [-h] [--outcomes OUTCOMES]
                                         [--dir DIR]

Review and correct pellet outcome classifications

options:
  -h, --help           show this help message and exit
  --outcomes OUTCOMES  Single *_pellet_outcomes.json file to review
  --dir DIR            Directory with multiple outcome files to review

Examples: mousereach-review-pellet-outcomes --outcomes
video_pellet_outcomes.json mousereach-review-pellet-outcomes --dir Processing/
Keyboard shortcuts (in review tool): N / P Next / Previous segment R Set
outcome to Retrieved D Set outcome to Displaced (scoring area) O Set outcome
to Displaced (outside) U Set outcome to Untouched I Set interaction frame to
current frame Space Play/pause video
```

### `mousereach-review-outcomes`

*Deprecated: use mousereach-review-pellet-outcomes*

Alias of `mousereach-review-pellet-outcomes` (same entry point: `mousereach.outcomes.cli:main_review`).

## Routine triage review: the causal review tool, TRIAGED-ONLY, over the Pending queue

### `mousereach-review-tool`

```
usage: mousereach-review-tool [-h] [--pending-dir PENDING_DIR]
                              [--all-segments] [--worklist WORKLIST] [--cv]

Triage review: resolve just the triaged segments the algo could not, over the
routine Pending queue.

options:
  -h, --help            show this help message and exit
  --pending-dir PENDING_DIR
                        Review queue root of per-video bundles. Default:
                        MouseReach_Pipeline/Processing/Review/triage.
  --all-segments        Walk ALL segments, not just triaged ones (full
                        review).
  --worklist WORKLIST   CSV/JSON naming specific videos and segment numbers.
                        Only those videos are offered and only those segments
                        are walked -- for asking a targeted question of the
                        corpus without editing any video's data to force it.
  --cv                  Use CV pellet localization for tighter narrowing
                        (slower: decodes each loaded video over the NAS).
```

## Bulk release check for deep-review bundles (human-invoked; gates on human.outcome)

### `mousereach-review-release`

```
usage: mousereach-review-release [-h] [--clear]

List deep-review bundles by review completeness (human.outcome per segment);
--clear writes the release marker for fully-answered bundles so the watcher's
return scan re-injects them.

options:
  -h, --help  show this help message and exit
  --clear     write {stem}_deep_review_cleared.json for every COMPLETE bundle
              (all segments carry a human outcome). Without this flag, list
              only.
```

## Pipeline census: total workload, per-stage breakdown, pace and ETA (headless JSON)

### `mousereach-census`

```
usage: mousereach-census [-h] [--json] [--out OUT] [--days DAYS]

Pipeline census: total workload, per-stage breakdown, pace and completion
estimates.

options:
  -h, --help   show this help message and exit
  --json       emit the full machine-readable census on stdout
  --out OUT    also write the JSON to this file
  --days DAYS  trailing window for pace measurement (default 14)
```

## Put a video into a review queue with a reason (generic; the way an external integrator such as a database tool asks MouseReach to hold a video for a person)

### `mousereach-route-to-queue`

```
usage: mousereach-route-to-queue [-h] [--worklist WORKLIST] --queue
                                 {triage,deep_review} --reason REASON
                                 [--flag-segments FLAG_SEGMENTS] [--json]
                                 [video_id]

mousereach-route-to-queue -- put an archived video into a review queue.

WHY THIS IS A PUBLIC COMMAND
----------------------------
MouseReach decides on its own when a video needs a person (segmentation
failed, an element it could not commit to). But OTHER systems can have
reasons too -- for example a database tool that compares the pipeline's
pellet outcomes with hand-scored bench sheets and finds a disagreement. Such
a tool must not reach into MouseReach's internals or its files; it asks
through this command. MouseReach stays independent (it knows nothing about
who asked or why beyond the reason text it records), and the integrator
gets exactly the same routing the pipeline uses itself.

What it does, in order:
  1. finds the video's results in the configured Analyzed tree,
  2. optionally flags specific segments (flagged_for_review=True with the
     given reason, triage_cleared cleared) in {video}_pellet_outcomes.json --
     that is what the triage review tool walks,
  3. moves the video's bundle into the queue with a routing manifest
     (review_gate.route_to_queue), updating the watcher database.

A SEGMENT A PERSON HAS ANSWERED IS NEVER FLAGGED
------------------------------------------------
Before flagging, the video's saved review is read and every requested segment
a person (or ground truth) has already answered -- matched by the FRAMES the
answer is about, the rule the release check uses -- is dropped from the list
and reported as "answered". If nothing is left, the video is not routed.
WHY: the release check counts such a segment answered, so a routed video went
straight back out with nobody looking, and the next night's scan sent it again
(2026-10). A caller cannot always know what a person answered; this command
reads the review itself, so the rule holds whoever asks.

THE WATCHER DATABASE -- the same file the daemon writes, or nothing is routed
-----------------------------------------------------------------------------
The database is resolved exactly as the daemon resolves it
(db_location.resolve_watcher_db_path: the configured watcher.db_path, else
<processing_root>/watcher.db) and printed as "[watcher db] <path>", like every
other watcher command. This command used to open a bare WatcherDB(), which
ignores the db_path override: every state write went to an unused decoy
database while the bundles moved on disk, and the live database never learned
of any of it.

Routing is REFUSED (exit 1, nothing flagged, nothing moved) when that database
file does not exist, or exists but cannot be opened. WHY refuse rather than
"route on disk only": a video moved into a queue while the database still says
'archived' leaves disk and database disagreeing for that video, and a worklist
does it for every video at once. A missing file is not created either --
WatcherDB's constructor would build an empty database, which is just a new
decoy. Fix the configuration (mousereach-setup) or start the watcher once so
its database exists, then route again.

A VIDEO THE WATCHER IS WORKING ON IS DEFERRED
---------------------------------------------
A video whose watcher state is dlc_queued, dlc_running, dlc_complete,
processing, processed or archiving is not flagged and not moved; it is
reported as "wait" (deferred: true with --json). WHY: this command writes the
daemon's live database. Setting 'triage' underneath a running pipeline makes
the pipeline's own 'processing' -> 'processed' write an illegal transition, and
the daemon marks the video failed with fresh outputs in Processing and old ones
in the queue; a processed or archiving video's Analyzed copy is older than the
outputs the daemon is about to archive, so a reviewer would judge stale data.
A deferral is not a failure: offer the video again on a later run.

Usage:
    mousereach-route-to-queue VIDEO_ID --queue triage --reason "bench disagreement" --flag-segments 3,7
    mousereach-route-to-queue VIDEO_ID --queue deep_review --reason "segmentation wrong"
    mousereach-route-to-queue --worklist worklist.json --queue triage --reason "..."
        worklist.json: [{"video_id": "...", "segment_nums": [3, 7]}, ...]

Exit codes:
    0  every requested video was routed, was already not in Analyzed, or was
       deferred because the watcher is working on it, or had
       every requested segment already answered by a person ("done")
    1  some video could not be routed, OR routing was refused because the
       watcher database could not be resolved, does not exist, or cannot be
       opened (then no video was touched)
ASCII-only output. With --json, stdout carries only the JSON results; the
"[watcher db]" line and any refusal message go to stderr.

positional arguments:
  video_id              e.g. 20250624_CNT0115_P2

options:
  -h, --help            show this help message and exit
  --worklist WORKLIST   JSON list of {"video_id", "segment_nums"} to route in
                        one go
  --queue {triage,deep_review}
  --reason REASON       Recorded in the routing manifest
  --flag-segments FLAG_SEGMENTS
                        Comma-separated segment numbers to flag (single-video
                        mode)
  --json                Machine-readable results
```

## Provenance checks -- what the pipeline actually produces, and whether the documents still describe the code.

### `mousereach-field-audit`

```
usage: mousereach-field-audit [-h] [--root ROOT] [--snapshot SNAPSHOT]
                              [--limit LIMIT] [--only-videos ONLY_VIDEOS]
                              [--json JSON] [--markdown MARKDOWN]

Follow every pipeline field from the stage that produces it into the database,
and report the ones that vanish.

options:
  -h, --help            show this help message and exit
  --root ROOT           Tree of finished videos to read (default: Analyzed)
  --snapshot SNAPSHOT   Directory holding reach_data.parquet (optional;
                        without one the database side is reported as not
                        compared)
  --limit LIMIT         Only read this many files per stage (quick pass)
  --only-videos ONLY_VIDEOS
                        Restrict to the video ids listed in this text file
                        (one per line); an integrator can produce such a list,
                        MouseReach does not depend on one
  --json JSON           Also write the full result as JSON
  --markdown MARKDOWN   Also write the report as a markdown document
```

### `mousereach-doc-check`

```
usage: mousereach-doc-check [-h] [--staged] [--message MESSAGE]
                            [--install-hook]

Keep the documents and the code from drifting apart silently.

options:
  -h, --help         show this help message and exit
  --staged           Check what is staged for commit (used by the commit-msg
                     hook)
  --message MESSAGE  Path to the commit message file, as git passes it to a
                     commit-msg hook
  --install-hook     Install the pre-commit hook into this repository
```

### `mousereach-fix-segmentation`

```
usage: mousereach-fix-segmentation [-h] [--queue-dir QUEUE_DIR]

Correct a video's segment cuts, using the candidate tray advances the
segmenter found but did not use.

options:
  -h, --help            show this help message and exit
  --queue-dir QUEUE_DIR
                        Queue of bundles to work through (default: the deep-
                        review queue)
```

### `mousereach-backfill-manifest-versions`

*Causal review, triaged-only (walk only unresolved elements)*

```
usage: mousereach-backfill-manifest-versions [-h] [--root ROOT]
                                             [--stage STAGES]
                                             [--archive-dir ARCHIVE_DIR]
                                             [--apply]

Fill in pipeline stage versions a manifest never recorded, reading each
version from that stage's own output file.

options:
  -h, --help            show this help message and exit
  --root ROOT           Directory to walk (default: the Analyzed tree)
  --stage STAGES        Repeatable. Restrict to these stages (default: all
                        tracked).
  --archive-dir ARCHIVE_DIR
                        Where to copy manifests before modifying them
                        (default: a dated directory under the pipeline's
                        _archived)
  --apply               Actually write. Without this, nothing is modified.
```

### `mousereach-triage-clearing`

*(legacy) GroundTruthWidget-based per-segment triage clearing*

```
usage: mousereach-triage-clearing [-h] [--corpus-root CORPUS_ROOT]
                                  [--algo-dir ALGO_DIR]
                                  [--video-name VIDEO_NAME]
                                  [--include-cleared] [--pre-pad PRE_PAD]
                                  [--post-pad POST_PAD] [--qc-count QC_COUNT]
                                  [--qc-report]

MouseReach Triage Clearing Tool — review and resolve algo-flagged segments one
at a time.

options:
  -h, --help            show this help message and exit
  --corpus-root CORPUS_ROOT
                        ROUTINE mode: root of per-video bundle subdirs (each
                        holds the four algo JSONs). Scans every bundle for
                        unresolved problems and routes failed-segmentation
                        videos to a separate re-seg lane. Defaults to the
                        routine review queue (CONNECTOME_ROOT/Behavior/MouseRe
                        ach_Pipeline/Processing/Review/triage) or
                        MOUSEREACH_ROUTINE_ROOT. Ignored if --algo-dir is
                        given.
  --algo-dir ALGO_DIR   Single flat directory containing *_reaches.json /
                        *_pellet_outcomes.json (quarantine layout). Overrides
                        --corpus-root. Defaults (only if no corpus root is
                        found) to the latest quarantine under CONNECTOME_ROOT/
                        Behavior/MouseReach_Improvement/validation_runs/DLC_*/
                        iterations/*/algo_outputs/.
  --video-name VIDEO_NAME
                        Restrict worklist to this video stem (e.g.
                        20250624_CNT0107_P3).
  --include-cleared     Include already-cleared segments in the worklist (for
                        re-review).
  --pre-pad PRE_PAD     Frames to load before each segment's start (context).
                        Default 30.
  --post-pad POST_PAD   Frames to load after each segment's end (context).
                        Default 30.
  --qc-count QC_COUNT   Routine spot-check: blend N already-passing segments
                        into the session (stratified rotating sample) for
                        confirm-the-algo QC. Only in corpus-root mode. Default
                        0 (triage only).
  --qc-report           Print the routine spot-check agreement/drift summary
                        and exit (does not launch napari).
```

### `mousereach-unified-review`

```
usage: mousereach-unified-review [-h] [--segment SEGMENT] [--frame FRAME]
                                 [--algo-dir ALGO_DIR]
                                 [--mode {outcome,reach,segmentation,general}]
                                 [--screenshot-dir SCREENSHOT_DIR]
                                 [--pre-pad PRE_PAD] [--post-pad POST_PAD]
                                 [video]

MouseReach Ground Truth Tool

positional arguments:
  video                 Video file to load

options:
  -h, --help            show this help message and exit
  --segment SEGMENT     Jump to this segment's decision_window after load
  --frame FRAME         Jump to this absolute frame after load
  --algo-dir ALGO_DIR   Directory containing the *_pellet_outcomes.json to
                        read decision_window from. Defaults to the video's
                        directory.
  --mode {outcome,reach,segmentation,general}
                        Info panel mode (auto-detected from --algo-dir if
                        omitted)
  --screenshot-dir SCREENSHOT_DIR
                        Default save dir for the Screenshot button. Auto-
                        derived from --algo-dir + --segment if omitted.
  --pre-pad PRE_PAD     Extra frames to load BEFORE the decision window
                        (default 0)
  --post-pad POST_PAD   Extra frames to load AFTER the decision window
                        (default 0)
```

### `mousereach-gt`

*GT scoring tool*

Alias of `mousereach-unified-review` (same entry point: `mousereach.review.ground_truth_widget:main`).

### `mousereach-review-legacy`

*Old whole-video review (kept for compat)*

```
usage: mousereach-review-legacy [-h] [--segment SEGMENT] [--frame FRAME]
                                [--algo-dir ALGO_DIR]
                                [--mode {outcome,reach,segmentation,general}]
                                [--screenshot-dir SCREENSHOT_DIR]
                                [--pre-pad PRE_PAD] [--post-pad POST_PAD]
                                [video]

MouseReach Review Tool

positional arguments:
  video                 Video file to load

options:
  -h, --help            show this help message and exit
  --segment SEGMENT     Jump to this segment's decision_window after load
  --frame FRAME         Jump to this absolute frame after load
  --algo-dir ALGO_DIR   Directory containing the *_pellet_outcomes.json
  --mode {outcome,reach,segmentation,general}
                        Info panel mode (auto-detected from --algo-dir if
                        omitted)
  --screenshot-dir SCREENSHOT_DIR
                        Default save dir for the Screenshot button. Auto-
                        derived from --algo-dir + --segment if omitted.
  --pre-pad PRE_PAD     Extra frames to load BEFORE the decision window
                        (default 0)
  --post-pad POST_PAD   Extra frames to load AFTER the decision window
                        (default 0)
```

### `mousereach-migrate-gt`

```
usage: mousereach-migrate-gt [-h] [--archive ARCHIVE] [--dry-run] [folder]

Migrate old separate GT files to unified format

positional arguments:
  folder                Folder to scan (default: Processing folder)

options:
  -h, --help            show this help message and exit
  --archive ARCHIVE, -a ARCHIVE
                        Archive folder for old GT files (default:
                        <folder>/archived_gt/)
  --dry-run, -n         Show what would be done without making changes
```

## Triage auto-resolve (pre-check before human review): if a triaged segment has a matching unified GT entry, lift the flag and copy the GT outcome. Runs as a step in the normal processing pipeline; only segments without GT remain triaged for the napari review tool.

### `mousereach-resolve-triage-from-gt`

```
usage: mousereach-resolve-triage-from-gt [-h] -i INPUT [--gt-dir GT_DIR] [-q]

Auto-resolve triaged segments from unified GT files. For every segment the
outcome detector flagged for review, check if GT already has the answer; if
yes, copy it into the algo output and lift the flag. Production-pipeline fast
path for videos that have been ground-truthed.

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Directory with *_pellet_outcomes.json + *_reaches.json
                        per video.
  --gt-dir GT_DIR       Directory with *_unified_ground_truth.json. Defaults
                        to --input dir, then sibling ../gt/.
  -q, --quiet           Suppress per-video output.
```

## Step 4b - Reach Assignment (joins v8 reaches + v6 cascade outcomes into per-reach permanent output for kinematic analysis)

### `mousereach-assign-reaches`

```
usage: mousereach-assign-reaches [-h] -i INPUT

Stamp per-reach outcome labels and the causal reach by joining v6 cascade
outcomes onto v8 reach detector outputs (assignment v2.1.0, the same code path
the automatic pipeline runs).

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Processing root or single video dir.
```

## Step 5 - Grasp Kinematics

### `mousereach-grasp-analyze`

```
usage: mousereach-grasp-analyze [-h] -i INPUT [-o OUTPUT] [-s SUFFIX]
                                [--overwrite] [--reviews-dir REVIEWS_DIR]

Extract features from reaches linked to outcomes (Step 5)

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input directory with validated reaches (Step 3) and
                        outcomes (Step 4)
  -o OUTPUT, --output OUTPUT
                        Output directory (default: Step5_Features/)
  -s SUFFIX, --suffix SUFFIX
                        File pattern for outcome files
  --overwrite           Overwrite existing feature files
  --reviews-dir REVIEWS_DIR
                        Directory holding {video}_causal_review.json files.
                        When a review exists for a video, the human outcome +
                        causal reach OVERRIDE the algo for the reviewed
                        segments (provenance: outcome_source). The input dir
                        is also checked.
```

### `mousereach-grasp-triage`

```
usage: mousereach-grasp-triage [-h] -i INPUT

Triage feature extraction results

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Directory with feature JSON files
```

### `mousereach-grasp-review`

```
usage: mousereach-grasp-review [-h] feature_file

Review feature extraction for a video

positional arguments:
  feature_file  Path to *_features.json file

options:
  -h, --help    show this help message and exit
```

## Step 6 - Export

### `mousereach-export`

```
usage: mousereach-export [-h] -i INPUT -o OUTPUT [--format {excel,csv}]

Export MouseReach results

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input directory
  -o OUTPUT, --output OUTPUT
                        Output file/directory
  --format {excel,csv}  Output format
```

### `mousereach-summary`

```
usage: mousereach-summary [-h] -i INPUT [-o OUTPUT]

Generate summary statistics

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input directory
  -o OUTPUT, --output OUTPUT
                        Output JSON file
```

## Archive - Move validated videos to NAS

### `mousereach-archive`

```
usage: mousereach-archive [-h] [--dry-run] [--list] [--status VIDEO_ID]
                          [--quiet]
                          [video_id]

Archive fully validated videos to NAS

positional arguments:
  video_id              Video ID to archive (default: all ready videos)

options:
  -h, --help            show this help message and exit
  --dry-run, -n         Show what would be done without moving files
  --list, -l            List videos ready for archive
  --status VIDEO_ID, -s VIDEO_ID
                        Check archive status for a specific video
  --quiet, -q           Suppress progress output

Examples:
    mousereach-archive                    # Archive all ready videos
    mousereach-archive 20250704_CNT0101_P1   # Archive specific video
    mousereach-archive --dry-run          # Preview what would be archived
    mousereach-archive --list             # List ready videos

Requirements:
    A video can only be archived when ALL stages are validated:
    - Segmentation: validated
    - Reach detection: validated
    - Outcome detection: validated
```

## Migration - One-time migration from old multi-folder to v2.3+ single-folder architecture

### `mousereach-migrate`

```
usage: mousereach-migrate [-h] [--dry-run] [--status] [--quiet] [--root ROOT]

Migrate MouseReach pipeline to v2.3+ single-folder architecture

options:
  -h, --help     show this help message and exit
  --dry-run, -n  Show what would be done without making changes
  --status, -s   Show current folder status
  --quiet, -q    Suppress progress output
  --root ROOT    Override processing root (default: from config)

Examples:
    python -m mousereach.migrate_to_processing --dry-run  # Preview changes
    python -m mousereach.migrate_to_processing            # Execute migration
    python -m mousereach.migrate_to_processing --status   # Show folder status

After migration:
    mousereach-index-rebuild   # Rebuild the pipeline index
```

## Algorithm Evaluation

### `mousereach-eval`

```
usage: mousereach-eval [-h] (--seg | --reach | --outcome | --all)
                       [--tolerance TOLERANCE] [--output OUTPUT]
                       [--gt-dir GT_DIR] [--algo-dir ALGO_DIR] [--verbose]
                       [path]

MouseReach Algorithm Evaluation Toolkit

positional arguments:
  path                  Directory containing GT and algorithm files (default:
                        processing root)

options:
  -h, --help            show this help message and exit
  --seg, -s             Evaluate segmentation algorithm
  --reach, -r           Evaluate reach detection algorithm
  --outcome, -o         Evaluate outcome classification algorithm
  --all, -a             Evaluate all algorithms
  --tolerance TOLERANCE, -t TOLERANCE
                        Frame tolerance for matching (default: varies by
                        algorithm)
  --output OUTPUT, --save OUTPUT
                        Save report to file
  --gt-dir GT_DIR       Separate GT directory (if different from algo dir)
  --algo-dir ALGO_DIR   Separate algorithm output directory
  --verbose, -v         Show detailed output

Examples:
    mousereach-eval --seg                     # Evaluate segmentation in processing root
    mousereach-eval --seg dev_SampleData/     # Evaluate segmentation in specific folder
    mousereach-eval --all --tolerance 10      # Evaluate all with custom tolerance
    mousereach-eval --reach -o report.txt     # Save report to file

The evaluator looks for GT files (*_ground_truth.json) and algorithm output
files (*_segments.json, *_reaches.json, *_pellet_outcomes.json) in the
specified directory or the MouseReach processing root.
```

### `mousereach-eval-direct`

```
usage: mousereach-eval-direct [-h]

Quick direct evaluation - evaluate GT files against algorithm output.

options:
  -h, --help  show this help message and exit
```

### `mousereach-update-algo-ref`

```
usage: mousereach-update-algo-ref [-h]

Auto-update the Current Performance section of ALGORITHM_REFERENCE.md.

options:
  -h, --help  show this help message and exit
```

### `mousereach-algo-vs-human`

```
usage: mousereach-algo-vs-human [-h]

Comprehensive comparison: Algorithm vs Human Ground Truth

Compares:
- Segment boundaries (frame accuracy)
- Outcome classification (accuracy, confusion matrix)
- Interaction frame detection (frame error)
- Reach count per segment
- Reach timing (start/apex/end frame errors) when GT available

options:
  -h, --help  show this help message and exit
```

## Export Utilities

### `mousereach-quick-summary`

```
usage: mousereach-quick-summary [-h]

Quick summary for PI - run this to generate a CSV and stats

options:
  -h, --help  show this help message and exit
```

### `mousereach-features-csv`

```
usage: mousereach-features-csv [-h]

Convert grasp_features.json files to CSV for easy viewing

options:
  -h, --help  show this help message and exit
```

## Kinematic Analysis Utilities

### `mousereach-reach-export`

```
usage: mousereach-reach-export

Export every reach in the local Processing folder to reach_kinematics.csv (one row per reach, next to Processing). Takes no options; runs immediately.
```

### `mousereach-real-kinematics`

```
usage: mousereach-real-kinematics

Compute kinematics from the DLC pose for every _reaches.json in the local Processing folder, writing real_kinematics.csv next to Processing. Takes no options; runs immediately.
```

## Performance Tracking

### `mousereach-perf`

```
usage: mousereach-perf [-h] [--algo {segmentation,reach,outcome,all}]
                       [--since SINCE] [--json] [--detailed]

View algorithm performance summary

options:
  -h, --help            show this help message and exit
  --algo {segmentation,reach,outcome,all}, -a {segmentation,reach,outcome,all}
                        Algorithm to show (default: all)
  --since SINCE, -s SINCE
                        Show entries since date (YYYY-MM-DD)
  --json, -j            Output as JSON
  --detailed, -d        Show per-video details

Examples:
  mousereach-perf                    Show all algorithms
  mousereach-perf --algo reach       Show reach detection only
  mousereach-perf --since 2026-01-01 Show entries since date
  mousereach-perf --json             Output as JSON
```

### `mousereach-perf-eval`

```
usage: mousereach-perf-eval [-h] [--algo {segmentation,reach,outcome,all}]
                            [--gt-dir GT_DIR] [--tolerance TOLERANCE]

Run batch evaluation against ground truth files

options:
  -h, --help            show this help message and exit
  --algo {segmentation,reach,outcome,all}, -a {segmentation,reach,outcome,all}
                        Algorithm to evaluate (default: all)
  --gt-dir GT_DIR, -g GT_DIR
                        Directory containing ground truth files
  --tolerance TOLERANCE, -t TOLERANCE
                        Frame tolerance for matching (default: 5)

Examples:
  mousereach-perf-eval --algo reach
  mousereach-perf-eval --algo all --gt-dir Processing/
```

### `mousereach-perf-report`

```
usage: mousereach-perf-report [-h] [--output OUTPUT]
                              [--format {markdown,methods,table}]

Generate scientific report from performance data

options:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output file (default: print to stdout)
  --format {markdown,methods,table}, -f {markdown,methods,table}
                        Output format (default: markdown)

Examples:
  mousereach-perf-report
  mousereach-perf-report --output report.md
  mousereach-perf-report --format methods
```

## Algorithm Documentation

### `mousereach-docs`

```
usage: mousereach-docs [-h]
                       [--algo {segmentation,reach_detection,outcome_classification,feature_extraction,all}]
                       [--output OUTPUT] [--format {markdown,json}]

Extract algorithm documentation from source code

options:
  -h, --help            show this help message and exit
  --algo {segmentation,reach_detection,outcome_classification,feature_extraction,all}, -a {segmentation,reach_detection,outcome_classification,feature_extraction,all}
                        Algorithm to show (default: all)
  --output OUTPUT, -o OUTPUT
                        Output file (default: print to stdout)
  --format {markdown,json}, -f {markdown,json}
                        Output format (default: markdown)

Examples:
  mousereach-docs                    Show all algorithms
  mousereach-docs --algo reach       Show reach detection only
  mousereach-docs --output docs.md   Save to file
  mousereach-docs --format json      Output as JSON
```

## Analysis Dashboard

### `mousereach-build-database`

```
usage: mousereach-build-database [-h] [-i DATA_DIR] [-o OUTPUT]
                                 [-b BRAINGLOBE_PATH] [--no-brainglobe]
                                 [-t TRACKING_DIR] [--no-surgery]
                                 [--derive-outcomes] [--include-flagged]
                                 [--force] [--check]

Build unified reach database (one row per reach, all metadata attached)

options:
  -h, --help            show this help message and exit
  -i DATA_DIR, --input DATA_DIR, -d DATA_DIR, --data-dir DATA_DIR
                        Directory with pipeline output files
                        (*_features.json). Default: Processing folder
  -o OUTPUT, --output OUTPUT
                        Output file path. Supports .csv, .parquet, .xlsx.
                        Default: ./unified_reaches.parquet
  -b BRAINGLOBE_PATH, --brainglobe-path BRAINGLOBE_PATH
                        Path to BrainGlobe region_counts.csv (default: auto-
                        detect)
  --no-brainglobe       Skip loading BrainGlobe connectomics data
  -t TRACKING_DIR, --tracking-dir TRACKING_DIR
                        Directory with Connectome_XX_Animal_Tracking.xlsx
                        files (adds Test_Phase, Weight, surgery)
  --no-surgery          Skip loading surgery/mouse-level metadata
  --derive-outcomes     Derive per-reach outcomes (miss_on_pillar,
                        miss_off_pillar, causal reach)
  --include-flagged     Include flagged/excluded reaches (normally excluded)
  --force, -f           Force rebuild even if database is current
  --check               Only check if database needs rebuilding (don't
                        actually build)

Output Formats:
    .csv     - CSV file (readable in Excel, larger size)
    .parquet - Parquet file (faster loading, smaller size, preserves dtypes)
    .xlsx    - Excel file with multiple sheets (Reaches, Sessions, Mice)

Examples:
    mousereach-build-database
        Build database using default locations, output to unified_reaches.parquet

    mousereach-build-database -o all_reaches.csv
        Export as CSV instead of Parquet

    mousereach-build-database --tracking-dir /path/to/Animal_Tracking
        Include experimental metadata (Test_Phase, Weight, surgery data)

    mousereach-build-database --derive-outcomes
        Add reach-level outcome derivation (miss_on_pillar vs causal reach)
```

## Data Explorer (pre-computed statistics database)

### `mousereach-build-explorer`

```
usage: mousereach-build-explorer [-h] [-i INPUT] [-o OUTPUT]
                                 [--features-dir FEATURES_DIR]

Build reach explorer database with pre-computed statistics

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input Excel or CSV file with reach data, or directory
                        with reach JSON files
  -o OUTPUT, --output OUTPUT
                        Output database path (default: reach_explorer.db in
                        input directory)
  --features-dir FEATURES_DIR
                        Directory with *_features.json files for kinematic
                        features
```

### `mousereach-explore`

```
usage: mousereach-explore [-h] [-d DATABASE] [--mouse MOUSE]
                          [--session SESSION]
                          [--compare COMPARE [COMPARE ...]] [--population]
                          [--list-mice] [--list-sessions LIST_SESSIONS]
                          [--json]

Query the reach explorer database

options:
  -h, --help            show this help message and exit
  -d DATABASE, --database DATABASE
                        Path to explorer database (default: auto-detect)
  --mouse MOUSE         Show stats for specific mouse
  --session SESSION     Show stats for specific session
  --compare COMPARE [COMPARE ...]
                        Compare multiple mice (e.g., --compare CNT0110 CNT0111
                        CNT0112)
  --population          Show population-level statistics
  --list-mice           List all mice in database
  --list-sessions LIST_SESSIONS
                        List sessions for a mouse
  --json                Output as JSON
```

## Database Sync - Automatic sync to central connectome database

### `mousereach-sync`

```
usage: mousereach-sync [-h] [--force] [--dry-run] [--verbose]

Sync MouseReach features to reach_data table in connectome database

options:
  -h, --help     show this help message and exit
  --force, -f    Force sync all files, even if unchanged
  --dry-run, -n  Show what would be synced without actually syncing
  --verbose, -v  Show detailed output
```

### `mousereach-sync-watch`

```
usage: mousereach-sync-watch [-h] [--debounce DEBOUNCE]

Watch Processing folder and sync new features files to database

options:
  -h, --help            show this help message and exit
  --debounce DEBOUNCE, -d DEBOUNCE
                        Seconds to wait after file change before syncing
                        (default: 2.0)
```

### `mousereach-sync-status`

```
usage: mousereach-sync-status [-h] [--json]

Show MouseReach database sync status

options:
  -h, --help  show this help message and exit
  --json      Output as JSON
```

## Watcher - Automated pipeline orchestration

### `mousereach-watch`

```
usage: mousereach-watch [--once] [--dry-run] [--verbose] [--quiet]

Start the automated pipeline watcher daemon for this node (role and paths
from ~/.mousereach/config.json; run mousereach-setup to configure).

  --once      Run one full scan+process cycle, then exit.
  --dry-run   Show what would be processed without doing it.
  --verbose   Debug-level logging.
  --quiet     Warnings and errors only.
```

### `mousereach-watch-status`

```
usage: mousereach-watch-status [--by-animal] [--json] [--log N]

Show this node's pipeline state: counts by state, recent activity, and the
status every node last reported.

  --by-animal   Per-animal breakdown with QC.
  --json        JSON output.
  --log N       Show the last N log entries (default 10).
```

### `mousereach-watch-reprocess`

```
usage: mousereach-watch-reprocess [video_id ...] [--all-failed] [--from-step STEP]

Reset failed videos so the watcher picks them up again.

  video_id ...       Specific video id(s) to reset.
  --all-failed       Reset every video currently in the failed state.
  --from-step STEP   Restart from this pipeline step instead of the beginning.
```

### `mousereach-watch-quarantine`

```
usage: mousereach-watch-quarantine [--list] [--release FILE] [--purge]

Manage files the watcher quarantined instead of processing.

  --list          Show what is quarantined and why.
  --release FILE  Move one file back into the normal flow.
  --purge         Permanently delete everything quarantined (confirm first).
```

### `mousereach-watch-unresolvable`

```
usage: mousereach-watch-unresolvable [--list | --sweep | --retry] [--dry-run]

Handle DB rows whose video file this node cannot find anywhere.

  --list      Show the pathless rows.
  --sweep     Mark them unresolvable so they leave the work loop.
  --retry     Put previously-swept rows back into the work loop.
  --dry-run   With --sweep/--retry: show what would change, change nothing.
```

### `mousereach-watch-prioritize`

```
usage: mousereach-watch-prioritize [-h] [--clear] [animal]

Set, show, or clear the priority animal for the watcher.

    When a priority animal is set, the watcher processes that animal's
    videos first within each work tier (crop, DLC, pipeline, archive).
    Other animals' videos still get processed — just after the priority
    animal's items are handled.

    Usage:
        mousereach-watch-prioritize                Show current priority
        mousereach-watch-prioritize CNT0107        Set priority to CNT0107
        mousereach-watch-prioritize --clear        Remove priority
    

positional arguments:
  animal      animal id to put first, e.g. CNT0107

options:
  -h, --help  show this help message and exit
  --clear     remove the priority animal
```

### `mousereach-watch-process-animal`

```
usage: mousereach-watch-process-animal ANIMAL_ID [--dry-run] [--tray T]

Queue every video for one animal through the pipeline (searches both
Single_Animal pre-cropped videos and Multi-Animal collages).

  ANIMAL_ID   e.g. CNT0107
  --dry-run   Show what would be queued without queueing it.
  --tray T    Only this tray type (e.g. P for pillar).
```

### `mousereach-watch-info`

```
usage: mousereach-watch-info

Show this machine's drives, configured mousereach paths, detected lab role,
and whether the watcher could run here. Takes no options.
```

### `mousereach-watch-toggle`

```
Pause or resume the watcher by hand, or show why it is paused.

    Usage:
        mousereach-watch-toggle            Flip: paused -> active, active -> paused
        mousereach-watch-toggle --pause    Pause by hand (stays paused until --resume)
        mousereach-watch-toggle --resume   Remove the hand pause
        mousereach-watch-toggle --status   Show both pause reasons; change nothing

    Two separate things pause the watcher:
      * the hand pause set here (the file watcher_paused.flag in this
        machine's processing folder, also set by the Pause button), and
      * a recording program listed with mousereach-watch-recorders being open.
    --resume removes only the hand pause. It cannot override a recording
    program: recording always wins, so close the recording program to let
    work start again.

    --pause and --resume do exactly what they say even when run twice, so a
    person or a script never flips the watcher back by accident (the plain
    toggle cannot promise that).
```

### `mousereach-watch-recorders`

```
usage: mousereach-watch-recorders [--list] [--add NAME] [--remove NAME] [--grace SECONDS]
                                  [--folder PATH] [--remove-folder PATH] [--quiet SECONDS]

Show or change the recording programs that pause THIS machine's watcher.
While any listed program is open the watcher starts no new work, and a pose
in progress is stopped and posed again later, because recording always wins:
a pose can be run again, a spoiled recording cannot. A collage crop that has
already started finishes first. A running watcher picks up changes made here
by itself within about a minute.

  --list            Show the list, the grace period, and what is running now.
                    This is what happens with no options at all.
  --add NAME        Add a program. NAME is the program's name as Task Manager
                    shows it on the Details tab, for example recorder.exe.
                    Capital letters, a full path or a missing .exe still match.
  --remove NAME     Take a program off the list.
  --grace SECONDS   How long every listed program must have been closed before
                    work starts again (default 120). WHY: people often close
                    the recording program and open it again a moment later.

Optional -- pause only while a recording is being WRITTEN (off by default):
  --folder PATH         A folder the recording program saves its videos into.
                        Once any folder is listed, an open recording program
                        pauses the watcher only while a file there was written
                        in the last --quiet seconds. WHY: the program is often
                        left open after recording, and the PC then sat idle
                        until someone closed it. A folder that cannot be read
                        keeps the watcher paused. Prove this on the machine,
                        with someone watching a test recording, before relying
                        on it: a pose already running needs ~10-15 s to stop.
  --remove-folder PATH  Take a folder off; with none left, an open program is
                        enough to pause again (the default).
  --quiet SECONDS       How long after the last write the watcher may work
                        again (default 120).

--add, --remove, --folder and --remove-folder may each be given more than once. The settings are saved in
the 'watcher' section of ~/.mousereach/config.json; nothing else in that file
changes. An empty list (the default) means the watcher never pauses for a
program.

Examples:
  mousereach-watch-recorders
  mousereach-watch-recorders --add recorder.exe
  mousereach-watch-recorders --remove recorder.exe
  mousereach-watch-recorders --grace 300
  mousereach-watch-recorders --folder "<path>"
```

## Version tracking and reprocessing

### `mousereach-version-check`

```
usage: mousereach-version-check [--mark] [--init]

Compare each archived video's processing manifest with the declared current
versions (pipeline_versions.json) and report which videos are out of date.

  --mark   Also mark outdated videos for reprocessing.
  --init   Create pipeline_versions.json from the installed code's versions.
```

### `mousereach-reconcile`

```
usage: mousereach-reconcile [-h] [--json] [--all]

Check every single-animal video against the definition of done, from the files
alone. Read-only.

options:
  -h, --help  show this help message and exit
  --json      print everything as JSON
  --all       also list every video that is fine, and every leftover copy
```

### `mousereach-aspa-import-collages`

```
usage: mousereach-aspa-import-collages [-h] [--apply] [--source SOURCE]
                                       [--dest DEST] [--cohorts COHORTS]
                                       [--limit LIMIT]
                                       [--sessions-file SESSIONS_FILE]

Copy ASPA historical collages (cohorts D and later) into the watcher's intake, renamed into pipeline form. The archive is only ever READ -- originals and old ASPA analyses are never modified. Dry run by default.

options:
  -h, --help            show this help message and exit
  --apply               Actually copy. Without this, prints the plan and
                        changes nothing.
  --source SOURCE       ASPA archive root (default: NAS
                        Archive/historical/ASPA).
  --dest DEST           Intake dir (default: Unanalyzed/Multi-Animal).
  --cohorts COHORTS     Comma-separated cohort dirs to import (default:
                        OptD,OptE,OptF,OptG,H,I,J,K,L,M).
  --limit LIMIT         Only act on the first N sessions (testing).
  --sessions-file SESSIONS_FILE
                        Import only the recordings holding at least one
                        session listed in this file (one
                        {date}_{animal}_{tray}{position} per line), in list
                        order.

Examples:
  python -m mousereach.aspa.import_collages
  python -m mousereach.aspa.import_collages --apply
  mousereach-aspa-import-collages --sessions-file sessions.txt
  mousereach-aspa-import-collages --sessions-file sessions.txt --apply
```

### `mousereach-backfill-kinematic-versions`

```
usage: mousereach-backfill-kinematic-versions [-h] [--apply] [--root ROOT]

Stamp the real kinematic extractor version onto processing manifests. Manifests are composed before feature extraction runs, so every one of them records kinematic_extractor='not_run' even when kinematics completed. This reads the version from each video's _features.json and writes it to the sibling manifest, making version currency work for kinematics without re-extracting anything.

options:
  -h, --help   show this help message and exit
  --apply      Write the corrections (default: report only)
  --root ROOT  Directory to walk (default: the Analyzed output tree)

Reports without changing anything unless --apply is given.
Example:
  mousereach-backfill-kinematic-versions            # report only
  mousereach-backfill-kinematic-versions --apply    # write the fix
```

### `mousereach-version-index-build`

```
usage: mousereach-version-index-build [-h] [--db DB]

Build (or rebuild) the per-video version index that the dashboard reads.
Normally you never need this: every processing run pushes its own row
automatically. Run it ONCE to backfill videos processed before the index
existed, or any time you want to rebuild it from scratch -- it is always safe
to re-run, since rows are upserted by video and rebuilt from each video's
manifest on disk.

options:
  -h, --help  show this help message and exit
  --db DB     Index location (default: pipeline_records/version_index.db on
              the NAS root).

Example: mousereach-version-index-build
```

### `mousereach-version-index-status`

```
usage: mousereach-version-index-status [-h] [--db DB]

Show the per-video version index the dashboard reads: where it lives, how many
videos it covers, and how many of those are current vs outdated against the
shipped algorithm versions.

options:
  -h, --help  show this help message and exit
  --db DB     Index location (default: the NAS root).

Example: mousereach-version-index-status
```

### `mousereach-crystallize`

```
usage: mousereach-crystallize (--cohort C | --videos "v1,v2") --label NAME
       mousereach-crystallize --list

Lock archived videos against automatic reprocessing (for publications).

  --cohort C    Every archived video of this cohort (e.g. CNT01).
  --videos LIST Comma-separated video ids.
  --label NAME  Required label naming the lock (e.g. "PNAS_2026").
  --list        Show what is crystallized, by label.
```

### `mousereach-uncrystallize`

```
usage: mousereach-uncrystallize (--label NAME | --videos "v1,v2")

Unlock crystallized videos so reprocessing can touch them again.

  --label NAME   Unlock every video crystallized under this label.
  --videos LIST  Comma-separated video ids.
```

## Backup watcher

### `mousereach-backup`

```
usage: mousereach-backup [--once] [--dry-run] [--verbose]

Backup watcher: mirrors pipeline data to the backup NAS via robocopy (add-only).

  --once      One sync cycle, then exit (default: run as a daemon).
  --dry-run   Show what would be synced without copying.
  --verbose   Debug-level logging.
```

## Archive migration - One-time Sort/ -> project/cohort restructure

### `mousereach-migrate-archive`

```
usage: mousereach-migrate-archive [-h] [--dry-run | --execute]
                                  [--nas-root PATH]

One-time migration from Sort/ to project/cohort archive structure.

Old structure:
  Analyzed/Sort/CNT/             (all CNT videos flat)
  Analyzed/Sort/Multi-Animal/    (collage MKVs)

New structure:
  Analyzed/Connectome/CNT01/
  Analyzed/Connectome/CNT01/Multi-Animal/
  Analyzed/Connectome/CNT02/
  ...

options:
  -h, --help       show this help message and exit
  --dry-run        Show what would happen without moving files (default)
  --execute        Actually move files and update watcher.db
  --nas-root PATH  Override NAS root path (default: from config)

Examples:
  mousereach-migrate-archive              # Dry run - show plan
  mousereach-migrate-archive --execute    # Actually move files
  mousereach-migrate-archive --nas-root <path>
```

## ASPA reprocessing tools

### `mousereach-aspa-import`

```
usage: mousereach-aspa-import [-h] (--cohort COHORT | --all) [--db-path PATH]
                              [--dry-run]

Import old ASPA Post-Processing xlsx files into ASPA.db.

options:
  -h, --help       show this help message and exit
  --cohort COHORT  Import single cohort (e.g. H)
  --all            Import all cohorts found under Analyzed/
  --db-path PATH   Override ASPA.db path (default: ASPA_DB_PATH env or
                   <configured NAS root>/ASPA.db)
  --dry-run        Parse xlsx files but do not write to database
```

### `mousereach-aspa-feed`

```
usage: mousereach-aspa-feed [-h] [--cohort COHORT | --all] [--subjects LIST]
                            [--subjects-file PATH] [--dry-run]
                            [--batch-size N] [--sessions-file PATH]
                            [--source PATH] [--dest PATH]

Copy ASPA's own single-animal cuts from the historical archive into the shared
singles folder, renamed into pipeline form, for any GPU node to pose. Copies
files (does NOT move originals).

options:
  -h, --help            show this help message and exit
  --cohort COHORT       Feed single cohort (e.g. H)
  --all                 Feed every cohort found in the source
  --subjects LIST       Feed only these animals, e.g. 'A01,A02,A07'. Either
                        the lab's own ids or the encoded ids; both work.
                        Without this, every animal in the chosen cohorts is
                        fed.
  --subjects-file PATH  Same, read from a file: commas, spaces or one per
                        line, '#' starts a comment. Use this for a list worth
                        keeping -- and keep it with your own records, not in
                        this repository.
  --dry-run             Show what would be copied without actually copying
  --batch-size N        Maximum number of files to copy per run (default: 50)
  --sessions-file PATH  Feed only these sessions, one
                        {date}_{animal}_{tray}{position} per line (lab or
                        pipeline naming); every listed session not found is
                        named. Searches every cohort unless --cohort is given.
  --source PATH         Folder holding {cohort}/Single_Animal (default: the
                        NAS root's Archive/historical/ASPA).
  --dest PATH, --queue-dir PATH
                        Where the copies go (default: the shared singles
                        folder, Unanalyzed/Single_Animal, where any GPU node
                        takes them).
```

### `mousereach-aspa-sync`

```
usage: mousereach-aspa-sync [-h] (--cohort COHORT | --all) [--db-path PATH]
                            [--dry-run]

Sync mousereach reprocessing results into ASPA.db.

options:
  -h, --help       show this help message and exit
  --cohort COHORT  Sync single cohort (e.g. H)
  --all            Sync all cohorts found under Analyzed/ASPA/
  --db-path PATH   Override ASPA.db path
  --dry-run        Parse results but do not write to database
```

### `mousereach-aspa-compare`

```
usage: mousereach-aspa-compare [-h] [--cohort COHORT] [--output FILE]
                               [--db-path PATH] [--min-overlap RATIO]

Compare old ASPA vs new mousereach results from ASPA.db.

options:
  -h, --help           show this help message and exit
  --cohort COHORT      Restrict comparison to one cohort (default: all
                       cohorts)
  --output FILE        Output CSV path (default: comparison.csv)
  --db-path PATH       Override ASPA.db path
  --min-overlap RATIO  Minimum frame overlap ratio to count as a match
                       (default: 0.3)
```

### `mousereach-session-coverage`

```
usage: mousereach-session-coverage [-h] --sessions-file SESSIONS_FILE
                                   [--out OUT]

Where each session of a list is in the pipeline, from shared storage only.
Read-only.

options:
  -h, --help            show this help message and exit
  --sessions-file SESSIONS_FILE
                        One {date}_{animal}_{tray}{position} per line (lab or
                        pipeline naming), '#' comments allowed.
  --out OUT             Write the result as JSON here (default: beside the
                        list, <list>.coverage.json).
```

### `mousereach-legacy-inventory`

```
usage: mousereach-legacy-inventory [-h] --from OLD_ROOT [--bring-in]
                                   [--limit LIMIT] [--label LABEL]

List every video on an old storage root, say whether the pipeline has it, and
(with --bring-in) copy what is missing into the stage folder that picks it up.
The old storage is only read.

options:
  -h, --help       show this help message and exit
  --from OLD_ROOT  The old storage root to inventory (only read).
  --bring-in       Copy every 'only here' pipeline video to its stage folder,
                   verified. Without it, only the report is written.
  --limit LIMIT    With --bring-in: copy at most this many files this run.
  --label LABEL    Name for the report files (default: from the old root's
                   name).
```

