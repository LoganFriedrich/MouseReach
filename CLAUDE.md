<!-- Parent: ../../CLAUDE.md -->
# MouseReach Pipeline - Claude Context

**Automated Single Pellet Assessment v2** - Mouse reaching behavior analysis pipeline.

> **Part of:** [Connectomics Grant Research Infrastructure](../../CLAUDE.md)

---

## CRITICAL: The Cardinal Rule of Behavioral Algorithms

**Every frame boundary error corrupts kinematic data.** Reach detection, segmentation, and all temporal boundary algorithms exist to define the windows over which kinematics are computed (velocity, acceleration, trajectory shape, peak extension). When a boundary is wrong - even by a few frames - kinematics are computed over frames that are NOT part of the behavior being studied. Those frames are noise that contaminates means, blurs distributions, and can create differential measurement artifacts between experimental groups.

**There is no such thing as "good enough" frame boundary accuracy.** The goal is always convergence toward what a human would call, on every reach, on every frame. Every mismatch between algorithm and human is a bug to understand and fix. Never describe accuracy as "good" or "excellent" - always report the remaining error rate and what needs to be done to reduce it.

This applies to: reach start frames, reach end frames, reach splitting, segmentation boundaries, and any temporal boundary that defines a kinematic analysis window.

---

## IMPORTANT: When to Reinstall

**STOP before suggesting `pip install`** - Think critically about whether it's actually needed.

| Situation | Reinstall Needed? |
|-----------|-------------------|
| Edited existing `.py` files | **NO** - Editable install reloads from source |
| Added new files to existing packages | **NO** - Python finds them automatically |
| Changed `pyproject.toml` (new CLI commands) | **YES** |
| Added new subpackage to `src/mousereach/` | **YES** |
| Changed package dependencies | **YES** |

**If you suggest reinstall, you MUST justify WHY** (e.g., "pyproject.toml was modified to add new CLI command X").

### Terminal Output: ASCII Only
**NEVER use non-ASCII/Unicode characters in `print()` or `logging` calls.** Windows consoles crash on Unicode symbols (`UnicodeEncodeError`). Use ASCII replacements: `->` not arrows, `[OK]`/`[FAIL]`/`[!]` not check/x/warning symbols, `*` not stars, `+/-` not plus-minus. Qt widget text and matplotlib are fine. See root CLAUDE.md for the full replacement table.

### DO NOT ATTEMPT PIP INSTALL DURING SESSIONS

**CRITICAL:** `pip install` commands HANG indefinitely on this network drive setup.
- Do NOT run pip install commands - they will timeout and waste tokens
- Just TELL the user to run it themselves after the session
- Example: "Run `pip install -e Y:\Behavior\MouseReach\MouseReach` to activate new CLI commands"

---

## Git Workflow

**GitHub is the canonical source.** Y: (NAS) keeps a backup clone that stays on `master` and is auto-pulled on startup.

| Location | Branch | Role |
|----------|--------|------|
| GitHub (`origin`) | `master` | Canonical source of truth |
| `Y:\2_Connectome\Behavior\MouseReach` (NAS) | `master` only | Network backup — always mirrors GitHub master |
| `A:\Behavior\MouseReach` (local drives) | Feature branches | Working copies — branch from master, merge back via PR or explicit merge |

**Rules:**
- **Never commit directly to master on local (A:) drives.** Always create a branch first.
- **Y: should never have local edits.** It exists only to mirror master as a NAS backup.
- **Before deploying**, merge your branch to master and push, then deploy pulls master on each machine.
- **Resolve conflicts properly** — don't force-push or skip merge conflicts.

---

## Environment Activation

**Always activate the conda environment before running MouseReach commands:**

```bash
conda activate mousereach
# Or if using a path-based environment:
# conda activate /path/to/conda_envs/mousereach/
```

After activating, you can run any `mousereach-*` command.

**After modifying `pyproject.toml`** (e.g., adding new CLI commands):
```bash
conda activate mousereach
pip install -e /path/to/MouseReach
```

---

## Project Identity

| Field | Value |
|-------|-------|
| Name | MouseReach |
| Version | 2.3.0 |
| Author | Logan Friedrich |
| Package | `src/mousereach/` |
| Python | 3.9-3.10 |

**Project Structure:**
- `MouseReach/` - Main package (source code, CLI entry points)
- `MouseReach_Pipeline/` - Data folders only (DLC_Queue, Processing, Failed, Archive)
- `Archive/` - Old pipeline scripts (pre-package consolidation)

---

## Architecture (v2.3+): Single Processing Folder

**IMPORTANT:** MouseReach v2.3+ uses a simplified single-folder architecture:

| Folder | Purpose |
|--------|---------|
| `DLC_Queue/` | Videos waiting for DLC (GPU machine) |
| `Processing/` | ALL post-DLC files (video + DLC + segments + reaches + outcomes) |
| `Failed/` | Processing errors |

**Review status is determined by `validation_status` in JSON files, NOT folder location.**

### Validation Status Values
- `needs_review` - Needs human verification
- `auto_approved` - High confidence, auto-approved
- `validated` - Human reviewed and approved

### File Co-location
All files for a video stay together in `Processing/`:
```
Processing/
├── 20250704_CNT0101_P1.mp4
├── 20250704_CNT0101_P1DLC_resnet50_....h5
├── 20250704_CNT0101_P1_segments.json        ← has validation_status
├── 20250704_CNT0101_P1_reaches.json         ← has validation_status
├── 20250704_CNT0101_P1_pellet_outcomes.json ← has validation_status
├── 20250704_CNT0101_P1_features.json        ← Step 5 kinematic features
└── 20250704_CNT0101_P1_seg_ground_truth.json (if created)
```

### Archive (Moving to NAS)
Files can only leave `Processing/` via `mousereach-archive` when ALL validation statuses are "validated".

---

## Quick Reference

| Resource | Path |
|----------|------|
| Config | `src/mousereach/config.py` - Paths, DLCPoints, Thresholds, Geometry |
| State | `src/mousereach/state.py` - MouseReachStateManager, widget coordination |
| Launcher | `src/mousereach/launcher.py` - Unified napari GUI |
| Entry Points | `pyproject.toml` - CLI commands (see below) |
| Widgets | `src/mousereach/napari.yaml` - 8 napari widgets |
| Dashboard | `src/mousereach/dashboard/widget.py` - Pipeline overview (uses index) |
| **Index** | `src/mousereach/index/` - Fast file index for startup performance |

---

## Video Naming Convention

**Format:** `YYYYMMDD_CNTXXXX_T#` where:
- `YYYYMMDD` = Date (e.g., 20250701)
- `CNTXXXX` = Animal ID (e.g., CNT0110)
- `T#` = Tray type + run number

**Tray codes:**
| Code | tray_type | run_num | Meaning |
|------|-----------|---------|---------|
| P1 | Pillar | 1 | First pillar tray run of the day |
| P2 | Pillar | 2 | Second pillar tray run of the day |
| P3 | Pillar | 3 | Third pillar tray run of the day |
| F | Flat | 1 | Flat tray run (typically just one) |
| F1 | Flat | 1 | First flat tray run |
| F2 | Flat | 2 | Second flat tray run |

**Example:** `20250701_CNT0110_P2` = Animal CNT0110 on July 1, 2025, second pillar tray run of that day.

---

## Pipeline Overview

```
Video Input (8-cam collage MKV)
    ↓ Step 0: Crop
Single Animal MP4 (480x540)
    ↓ Step 1: DLC (GPU machine)
HDF5 Tracking (*DLC*.h5) - 18 bodyparts
    ↓ Step 2: Segmentation
*_segments.json (21 trial boundaries)
    ↓ Step 3: Reach Detection
*_reaches.json (reach events per segment)
    ↓ Step 4: Pellet Outcomes
*_pellet_outcomes.json (Retrieved/Displaced/Untouched)
    ↓ Step 5: Grasp Kinematics
*_features.json (48 kinematic features per reach)
    ↓ Step 6: Export
results.xlsx (Excel workbook)
```

---

## Core Modules

| Step | Module | Core Algorithm | Primary CLI |
|------|--------|----------------|-------------|
| 0 | `video_prep/` | `core/cropper.py` | `mousereach-crop` |
| 1 | `dlc/` | (external DeepLabCut) | `mousereach-dlc-batch` |
| 2 | `segmentation/` | `core/segmenter_robust.py` | `mousereach-segment` |
| 3 | `reach/` | `core/reach_detector.py` | `mousereach-detect-reaches` |
| 4 | `outcomes/` | `core/pellet_outcome.py` | `mousereach-detect-outcomes` |
| 5 | `kinematics/` | `core/feature_extractor.py` | `mousereach-grasp-analyze` |
| 6 | `export/` | `core/exporter.py` | `mousereach-export` |
| Archive | `archive/` | `core.py` | `mousereach-archive` |
| Dev | `eval/` | Evaluators for all algorithms | `mousereach-eval` |
| Dev | `review/` | Comparison panel for review widgets | (widget component) |

---

## Portable Configuration (Environment Variables)

**MouseReach requires configuration before first use** - run `mousereach-setup` to set paths.

### Configuration Variables

| Variable | Purpose | Required? |
|----------|---------|-----------|
| `MouseReach_PROCESSING_ROOT` | Pipeline working folders | **Yes** |
| `MouseReach_NAS_DRIVE` | Archive/final output location | Optional |

### Quick Setup

```bash
mousereach-setup              # Interactive setup wizard (REQUIRED before first use)
mousereach-setup --show       # View current configuration
```

Configuration is saved to `~/.mousereach/config.json` and persists across sessions.
Works with any drive configuration - D:, E:, network paths, UNC paths, etc.

**Example Setup** (typical single/multi-machine deployment):
```
/path/to/MouseReach/                    ← MouseReach source code
/path/to/conda_envs/mousereach/         ← Conda environment
/path/to/MouseReach_Pipeline/           ← Pipeline working folders (shared with GPU machine)
/path/to/NAS/                           ← NAS archive (raw videos, final outputs)
```

Benefits of network drive setup:
- Supports network drives (can be a remote server location)
- Enables multi-machine setup (GPU machine accesses same folders)
- Works with mapped drives and UNC paths
- Scales to institutional storage systems

### Multi-Machine Setup

**2 machines:** Analysis PC (main dev) + GPU machine (DLC processing)

| Component | Location | Notes |
|-----------|----------|-------|
| Code | Analysis PC only | Clone repo, install with `pip install -e .` |
| Conda Env | Per-machine | Each machine needs its own environment |
| Pipeline Folders | Shared location (network drive) | Both machines must access same path |
| NAS Archive | Shared location | Both machines |

**Key Point:** Both machines must set `MouseReach_PROCESSING_ROOT` to the SAME shared folder path.

---

## Documentation Update Checklist

**Before every commit, check:**

- [ ] Changed `config.py`? → Update root `AGENTS.md` ripple effects table
- [ ] Changed `pyproject.toml` scripts? → Update this file
- [ ] Changed `core/*.py` algorithm? → **Update `ALGORITHM_REFERENCE.md`** (see below) + module's `AGENTS.md`
- [ ] Changed JSON output format? → Update relevant module `AGENTS.md`
- [ ] Added/renamed widget? → Update `napari.yaml` + module `AGENTS.md`
- [ ] Changed environment variables? → Update this file + root `AGENTS.md`

**See:** Root `AGENTS.md` for full ripple effects mapping

---

## ALGORITHM_REFERENCE.md Maintenance

**`ALGORITHM_REFERENCE.md`** is the PI-facing document that explains exactly how each algorithm works, with detailed parameters, thresholds, and evidence of why the current approach is best.

### CRITICAL: Update When Algorithms Change

**Any time you modify algorithm logic in these files, you MUST update the corresponding section in `ALGORITHM_REFERENCE.md`:**

| Algorithm File | Doc Section to Update |
|----------------|----------------------|
| `src/mousereach/segmentation/core/segmenter_robust.py` | Section 1: Segmentation |
| `src/mousereach/reach/core/reach_detector.py` | Section 2: Reach Detection |
| `src/mousereach/outcomes/core/pellet_outcome.py` | Section 3: Outcome Classification |
| `src/mousereach/config.py` (thresholds/geometry) | All sections that reference changed values |

**What to update:**
- Algorithm description (if logic changed)
- Parameters/thresholds (if values changed)
- Version number (bump it in the doc header)
- Algorithm Evolution section (add a new row explaining what changed and why)

### Performance Section Auto-Updates

The "Current Performance" section between `<!-- AUTO-GENERATED PERFORMANCE SECTION START -->` and `<!-- AUTO-GENERATED PERFORMANCE SECTION END -->` markers is auto-generated. To refresh it after re-running evaluation:

```bash
python -m mousereach.eval.update_algorithm_reference
```

This regenerates per-video tables, timing accuracy, confusion matrix, and weakness summary from live eval data.

### Full Eval Report with Plots

```bash
python -m mousereach.eval.report_cli
```

Generates 7 matplotlib figures + text summary. Auto-opens the dashboard.

---

## Pipeline Index (Fast Startup)

**Problem:** Dashboard and batch widgets freeze on startup because they scan network folders (30+ seconds on Y: drive).

**Solution:** Pipeline index file (`PROCESSING_ROOT/pipeline_index.json`) caches all file locations and metadata.

### Automatic Updates (No Action Needed)
- **Processing steps:** Segmentation, reach detection, outcome detection all update index when saving results
- **Triage commands:** Update index when sorting files into review queues
- **Advance commands:** Update index when moving validated files forward

### Manual Updates Required
- **After DLC processing:** GPU machine doesn't update index -> Run `mousereach-index-refresh DLC_Queue`
- **After manual file moves:** Index won't know about files you moved manually -> Run `mousereach-index-refresh`
- **First install:** No index exists yet -> Run `mousereach-index-rebuild`

### Commands
```bash
mousereach-index-rebuild   # Full rebuild from scratch (~30s)
mousereach-index-status    # Show current index status
mousereach-index-refresh   # Refresh only changed folders (smart detection)
```

### Troubleshooting
- **Dashboard shows stale data?** Click "Rebuild Index" button or run `mousereach-index-rebuild`
- **GPU machine processed files?** Run `mousereach-index-refresh DLC_Queue`
- **Index corrupt?** Delete `pipeline_index.json` and run `mousereach-index-rebuild`

---

## CLI Commands

### Main
```bash
mousereach                    # Launch GUI with all widgets
```

### Pipeline Index (Fast Startup)
```bash
mousereach-index-rebuild      # Full rebuild of pipeline index (run after manual file moves)
mousereach-index-status       # Show index status and file counts
mousereach-index-refresh      # Refresh specific folders (or all stale folders)
```

### Step 0 - Video Prep
```bash
mousereach-crop              # Crop 8-cam collage to single videos
mousereach-convert           # Convert video formats
mousereach-prep              # Full prep workflow
mousereach-compress          # Compress videos
```

### Step 1 - DLC
```bash
mousereach-dlc-batch         # Batch DLC processing
mousereach-dlc-quality       # Check DLC output quality
```

### Step 2 - Segmentation
```bash
mousereach-segment           # Batch segmentation
mousereach-triage            # Auto-triage by confidence
mousereach-advance           # Move validated files forward
mousereach-segment-review    # Launch review GUI
```

### Step 3 - Reach Detection
```bash
mousereach-detect-reaches    # Batch reach detection
mousereach-triage-reaches    # Auto-triage
mousereach-advance-reaches   # Move validated forward
mousereach-review-reaches    # Launch review GUI
```

### Step 4 - Pellet Outcomes
```bash
mousereach-detect-outcomes   # Batch outcome detection
mousereach-triage-outcomes   # Auto-triage
mousereach-advance-outcomes  # Move validated forward
mousereach-review-pellet-outcomes  # Launch review GUI
```

### Step 5 - Grasp Kinematics
```bash
mousereach-grasp-analyze     # Extract kinematic features
mousereach-grasp-triage      # Auto-triage
mousereach-grasp-review      # Launch feature viewer
```

### Step 6 - Export
```bash
mousereach-export            # Export to Excel
mousereach-summary           # Generate summary stats
```

### Archive - Move Validated to NAS
```bash
mousereach-archive                    # Archive all ready videos
mousereach-archive video_id           # Archive specific video
mousereach-archive --dry-run          # Preview what would be archived
mousereach-archive --list             # List videos ready for archive
mousereach-archive --status video_id  # Check archive status
```

**Note:** Videos can only be archived when ALL validation statuses are "validated".

### Evaluation & Export (Integrated Scripts)
```bash
mousereach-eval-direct               # Run all evaluators (direct execution, no wrapper)
mousereach-algo-vs-human             # Compare algorithm vs human performance
mousereach-quick-summary             # Quick summary stats for a video
mousereach-features-csv              # Export features to CSV
mousereach-reach-export              # Export reach data to Excel
mousereach-real-kinematics           # Real-world kinematic analysis
```

**Note:** These commands were moved into the mousereach package from standalone scripts. The `Archive/` folder under MouseReach contains old versions of pipeline scripts.

### Algorithm Evaluation
```bash
mousereach-eval --seg [path]     # Evaluate segmentation vs ground truth
mousereach-eval --reach [path]   # Evaluate reach detection vs ground truth
mousereach-eval --outcome [path] # Evaluate outcome classification vs ground truth
mousereach-eval --all [path]     # Evaluate all algorithms
```

### Performance Tracking
```bash
mousereach-perf                  # View performance summary
mousereach-perf --detailed       # Include per-video breakdown
mousereach-perf-eval             # Run batch evaluation against ground truth
mousereach-perf-report           # Generate markdown report for publication
mousereach-perf-report -f methods # Generate methods section text
```

### Algorithm Documentation
```bash
mousereach-docs                  # Extract algorithm documentation from source
mousereach-docs --algo reach     # Show specific algorithm docs
mousereach-docs -o ALGORITHMS.md # Save to file
```

### Database Sync
```bash
mousereach-sync                  # Sync all _features.json to reach_data table
mousereach-sync --force          # Re-sync all (delete and re-import)
mousereach-sync --status         # Show sync status
mousereach-sync --export         # Export reach_data.csv
mousereach-sync --watch          # Watch Processing/ for new features files
```

---

## Fast Launch Options

### If Conda Environment is on LOCAL Drive (C:) - Recommended

| Workflow | Command | Load Time |
|----------|---------|-----------|
| **Unified review** | `mousereach-review-tool video.mp4` | ~15-30s |
| Review boundaries | `mousereach-segment-review video.mp4` | ~15-30s |
| Review outcomes | `mousereach-review-pellet-outcomes` | ~15-30s |
| Review reaches | `mousereach-review-reaches` | ~15-30s |
| View features | `mousereach-grasp-review` | ~15-30s |
| All review tools | `mousereach --reviews video.mp4` | ~30-45s |
| Full GUI | `mousereach` | ~30-60s |

### If Conda Environment is on NETWORK Drive (Y:) - Slow!

All napari commands will be **2-3 minutes** because napari scans all packages over the network.

**FIX:** Move conda environment to local C: drive (see INSTALL.md troubleshooting).

### Alternative: Load widgets on-demand from napari

```bash
napari  # Then: Plugins → MouseReach → [Select widget]
```

This still has the same startup delay, but lets you load only what you need.

---

## Complete Step-by-Step Guide (For New Users)

**Assume you know nothing.** These instructions cover every single step.

### Running the Full Pipeline on New Videos

#### Step 1: Open a Terminal

1. Press `Win + R` on your keyboard
2. Type `cmd` and press Enter
3. A black command prompt window will open

#### Step 2: Activate the Conda Environment

Type this command:

```bash
conda activate mousereach
```

Press Enter. You should see `(mousereach)` appear at the start of your command line.

> Note: If your environment is in a custom location, use the full path:
> `conda activate /path/to/conda_envs/mousereach/`

**If you get an error:** Make sure conda is installed and available in your PATH.

#### Step 3: Launch the MouseReach GUI

Type this command and press Enter:

```bash
mousereach
```

Wait 30-60 seconds. A window will open with the MouseReach interface.

#### Step 4: Navigate to the Pipeline Tab

1. Look at the LEFT side of the window
2. Find the vertical tabs
3. Click on **"Pipeline"**

#### Step 5: Run Processing on Videos

1. In the Pipeline tab, you'll see a button that says **"Reprocess All"**
2. Click that button
3. A progress bar will appear showing the processing status
4. Wait for it to complete (this may take several minutes per video)

**What this does:** Runs segmentation, reach detection, and outcome detection on all videos in the Processing folder.

#### Step 6: Rebuild the Index

After processing completes:

1. Look for the **"Rebuild Index"** button
2. Click it
3. Wait for the index to rebuild (takes ~30 seconds)

**Why:** The index is what makes the dashboard load fast. After processing new videos, you need to rebuild it so the dashboard knows about the new results.

#### Step 7: View Results in Pipeline Overview

1. Click on the **"Pipeline Overview"** tab (if not already there)
2. You'll see a table showing all your videos
3. Each row shows:
   - Video name
   - Segmentation status (✓ = validated, ⚡ = auto-approved, ⏳ = needs review)
   - Reach status (same icons)
   - Outcome status (same icons)

#### Step 8: Review Items That Need Attention

Look for any rows with ⏳ (orange clock) icons - these need human review:

1. Click on the row to select the video
2. Click "Open Review" to open the review interface
3. Make corrections as needed
4. Click "Save" when done

#### Step 9: Verify Everything is Complete

After reviewing, check that all videos have ✓ (green checkmark) in all columns. Only fully validated videos can be archived.

---

### Quick Reference: Status Icons

| Icon | Meaning | Action Needed |
|------|---------|---------------|
| ✓ (green) | Validated | None - ready for archive |
| ⚡ (blue) | Auto-approved | None - high confidence, auto-passed |
| ⏳ (orange) | Needs review | Open review widget and verify |
| ✗ (red) | Failed/Error | Check logs, may need manual fix |

---

### Troubleshooting

**"No validated file pairs found"** when running CLI commands:
- Use the GUI instead (steps above)
- OR add `--skip-validation-check` flag to CLI commands

**GUI won't start / hangs:**
- Make sure conda environment is activated
- Try: `mousereach --debug` for more info

**Processing is slow:**
- This is normal for network drives
- Processing runs locally but reads/writes over network

**Index out of date / Dashboard shows wrong info:**
- Click "Rebuild Index" button in Pipeline tab
- OR run: `mousereach-index-rebuild`

---

## Napari Widgets (10 total)

| Widget | Step | Purpose |
|--------|------|---------|
| `PipelineDashboard` | Dashboard | View all files in pipeline with status, versions, timestamps, and ground truth info |
| `VideoPrepWidget` | 0 | Crop videos |
| `DLCWidget` | 1 | DLC analysis |
| `UnifiedPipelineWidget` | 2 | Run batch pipeline (auto-scans all pipeline folders) |
| `BoundaryReviewWidget` | 2b | Review segmentation |
| `PelletOutcomeAnnotatorWidget` | 3b | Review outcomes |
| `ReachAnnotatorWidget` | 4b | Review reaches |
| `DataViewerWidget` | 5 | View kinematic features |
| `UnifiedReviewWidget` | All | Combined review tool for all steps |
| `PerformanceViewerWidget` | Dev | Algorithm performance tracking |

---

## Algorithm Evaluation (AlgoEval Toolkit)

For algorithm development and optimization, MouseReach includes a standardized evaluation framework.

### Usage

```bash
mousereach-eval --seg dev_SampleData/        # Evaluate segmentation
mousereach-eval --all --tolerance 10         # Custom tolerance (frames)
mousereach-eval --reach -o report.txt        # Save report to file
```

### What It Measures

| Algorithm | Key Metrics | Error Categories |
|-----------|-------------|------------------|
| Segmentation | Accuracy (±5 frames), mean error, timing bias | missed_boundaries, extra_boundaries, early/late detections |
| Reach Detection | Precision, recall, F1, timing errors | missed_reaches, phantom_reaches, timing_errors |
| Outcome Classification | Accuracy, per-class P/R/F1, confusion matrix | retrieved_missed/phantom, displaced_as_untouched |

### Output Format

Reports include:
1. Per-video results with success/failure status
2. Aggregate metrics across all videos
3. Error categorization with examples
4. Actionable recommendations based on error patterns

### Ground Truth Files

Evaluators look for GT files matching these patterns:
- `*_seg_ground_truth.json` - Segmentation boundaries
- `*_reach_ground_truth.json` - Reach events
- `*_outcomes_ground_truth.json` - Pellet outcomes

---

## Review Widget Features

### Algo vs GT Comparison Panel

All review widgets (segmentation, reach, outcome) include a collapsible comparison panel showing:

- **Left column**: Algorithm output (read-only)
- **Right column**: Ground truth or editable copy
- **Diff summary**: Count of matching/differing items
- **Color coding**: Differences highlighted in orange

Click any item in the comparison panel to jump to that boundary/reach/segment.

### Smart Video Loading

Videos now load with RAM-aware caching:

| Available RAM | Strategy | Cache Size |
|---------------|----------|------------|
| >3x video size | Large cache | 1000 frames |
| 1-3x video size | Medium cache | 300 frames |
| 0.5-1x video size | Small cache | 100 frames |
| <0.5x video size | Compressed preview | Auto-created |

No manual intervention needed - the system auto-detects and picks the optimal strategy.

---

## Detailed Documentation

**AGENTS.md Hierarchy** - AI-navigable codebase index:
- `AGENTS.md` (root) - Project overview, ripple effects, dependencies
- `src/mousereach/AGENTS.md` - Package overview, module map
- `src/mousereach/*/AGENTS.md` - Per-module documentation with:
  - Purpose and key files
  - CLI commands
  - Algorithm details (in `core/AGENTS.md`)
  - Testing requirements
  - Internal/external dependencies

Run `/deepinit` to regenerate after major structural changes.
