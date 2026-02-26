# MouseReach - Mouse Reaching Behavior Analysis

Complete pipeline for analyzing mouse single-pellet reaching behavior videos.

---

## Quick Start

```bash
conda activate mousereach
mousereach                           # Launch all tools
mousereach "path/to/video.mp4"       # Launch with video pre-loaded
mousereach --reviews                 # Launch only review tools
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
| 5 | **Grasp Kinematics** | Reaches + Outcomes + DLC | `_features.json` (48 features/reach) |
| 6 | **Export** | All results | Excel/CSV |

---

## GUI Tools

Launch with `mousereach` command. Available tabs:

| Tab | Purpose |
|-----|---------|
| **Pipeline Dashboard** | Overview of all videos and pipeline status |
| **Step 0 - Crop Videos** | Crop multi-animal collage to single animals |
| **Step 1 - DLC Analysis** | Run DeepLabCut pose estimation |
| **Step 2 - Run Pipeline** | Batch segmentation, reach, outcome detection |
| **Step 3 - Review Tool** | Fix algorithm mistakes (edits algo files) |
| **Step 4 - View Features** | Visualize kinematic features |
| **GT Tool** | Create ground truth files for evaluation |

### Review Tool vs GT Tool

| Tool | Purpose | Saves To |
|------|---------|----------|
| **Review Tool** | Fix algorithm mistakes | `_segments.json`, `_reaches.json`, `_pellet_outcomes.json` |
| **GT Tool** | Create evaluation ground truth | `_unified_ground_truth.json` |

**Review Tool**: Use when the algorithm made errors. Corrections are saved with `human_corrected: true` flags directly in the algorithm output files.

**GT Tool**: Use when creating comprehensive ground truth for algorithm evaluation. Creates separate unified GT files with verification status for every element.

---

## CLI Commands Reference

### Main Launcher

| Command | Description |
|---------|-------------|
| `mousereach` | Launch the napari GUI |
| `MouseReach` | Alias for `mousereach` |

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
mousereach-crop collage.mkv -o output/
mousereach-convert video.mkv
mousereach-compress -i folder/ -o compressed/
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
| `mousereach-advance` | Move validated to next stage |
| `mousereach-segment-review` | Launch review GUI |
| `mousereach-reject-tray` | Reject unsupported tray types |

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
| `mousereach-advance-reaches` | Move to next stage |
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
| `mousereach-advance-outcomes` | Move to next stage |
| `mousereach-review-pellet-outcomes` | Launch review GUI |

```bash
mousereach-detect-outcomes -i Processing/
mousereach-review-pellet-outcomes --outcomes video_pellet_outcomes.json
```

---

## Review & Ground Truth Tools

| Command | Description |
|---------|-------------|
| `mousereach-review-tool` | Launch Review Tool (edits algo files) |
| `mousereach-unified-review` | Launch GT Tool (creates GT files) |
| `mousereach-migrate-gt` | Migrate old GT to unified format |

```bash
# Fix algorithm mistakes
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
| `mousereach-grasp-review` | Review analysis |

---

## Step 6 - Export

| Command | Description |
|---------|-------------|
| `mousereach-export` | Export to Excel/CSV |
| `mousereach-summary` | Generate summary statistics |

```bash
mousereach-export -i Processing/ -o results.xlsx
mousereach-summary -i Processing/
```

---

## Automated Watcher

The watcher monitors the data repository for new collage videos and runs the
full pipeline automatically (crop, DLC, segment, reaches, outcomes, archive).

| Command | Description |
|---------|-------------|
| `mousereach-watch` | Start the watcher daemon |
| `mousereach-watch --once` | Process all pending work, then exit |
| `mousereach-watch --dry-run` | Show what would be processed |
| `mousereach-watch-status` | Show pipeline status (video/collage counts, recent activity) |
| `mousereach-watch-status --by-animal` | Per-animal breakdown with QC |
| `mousereach-watch-status --log N` | Show last N processing log entries |
| `mousereach-watch-toggle` | Pause/resume the watcher (for filming sessions) |
| `mousereach-watch-toggle --status` | Show whether watcher is paused or active |
| `mousereach-watch-info` | Diagnose drives and path accessibility |

The watcher runs on the NAS / DLC PC (the machine with direct NAS access and GPU).
Set up with `mousereach-setup --set-role "NAS / DLC PC"` then `mousereach-setup`.

### Auto-Launch on Login

The `deploy/` folder contains batch scripts that auto-detect the MouseReach
environment and work on any PC:

1. **Copy startup script** → `Win+R` → `shell:startup` → paste `deploy/MouseReach-Watcher-Startup.bat`
2. **Copy toggle shortcut** → paste `deploy/MouseReach-Toggle.bat` to Desktop

On login, the startup script will:
1. Wait for the NAS drive (Y:\) to be available
2. Launch the watcher daemon in its own terminal window
3. Launch a status monitor that refreshes every 60 seconds

The **MouseReach Toggle** shortcut lets you quickly pause/resume processing
(e.g., during filming sessions).

---

## Archive

| Command | Description |
|---------|-------------|
| `mousereach-archive` | Move validated videos to NAS |

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
| `mousereach-analyze` | Launch Streamlit dashboard |
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
| `mousereach-sync` | Sync _features.json files to SQLite database |
| `mousereach-sync --force` | Re-sync all files (delete and re-import) |
| `mousereach-sync --status` | Show sync status and reach counts |
| `mousereach-sync --export` | Export reach_data.csv for analysis |
| `mousereach-sync --watch` | Watch for new features files and auto-sync |

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
mousereach-setup --set-role "NAS / DLC PC"

# 2. Run the setup wizard (paths pre-filled, just hit Enter)
mousereach-setup
```

### Machine Roles

Each PC in the lab serves a specific role. Setting the role tells the wizard
what defaults to use for this machine:

| Role | Description | Runs Watcher? |
|------|-------------|---------------|
| **NAS / DLC PC** | Direct-attached NAS, GPU, DLC model | Yes |
| **GPU Filming PC** | Filming/cropping workstation with GPU | No |
| **Processing Server** | Reach detection + kinematics | No |

```bash
mousereach-setup --list-roles      # See available roles
mousereach-setup --set-role NAME   # Set this PC's role
mousereach-setup --show            # Show current configuration
```

The role is saved to `~/.mousereach/machine_role.json`. If no role is set,
the wizard auto-detects based on drive patterns, or prompts manually.

Other labs can edit `lab_profiles.json` (ships with the package) to define
their own machine roles and defaults.

### Configuration File

| Setting | Purpose | Required? |
|---------|---------|-----------|
| `processing_root` | Pipeline working folders (DLC_Queue, Processing, Failed) | **Yes** |
| `nas_drive` | Data repository / archive location | Optional |
| `watcher.*` | Automated pipeline settings (DLC config, GPU, polling) | For watcher PC |

Configuration is saved to `~/.mousereach/config.json` and persists across sessions.

---

## Data Flow

```
DLC_Queue/           -> Videos waiting for DLC processing
Processing/          -> ALL active videos + analysis files (v2.3+ unified)
Failed/              -> Processing errors
```

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
- TensorFlow 2.10+

---

## Acknowledgments

- **[DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)** - Markerless pose estimation ([Mathis Lab](http://www.mackenziemathislab.org/))
  > Mathis, A., Mamidanna, P., et al. (2018). DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. *Nature Neuroscience*, 21, 1281-1289.
- **[napari](https://napari.org/)** - Multi-dimensional image viewer
- **[Claude](https://anthropic.com/)** - AI-assisted development (Anthropic)

---

*MouseReach v2.3.0*
*Developed by Logan Friedrich*
