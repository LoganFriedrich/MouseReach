# MouseReach Installation & Setup Guide

**Last Updated:** 2026-02-13
**Tested On:** Windows 10/11 with NVIDIA GPU

---

## Quick Install

```bash
# Create environment
conda create -n mousereach python=3.10
conda activate mousereach

# Install
cd /path/to/MouseReach
pip install -e .

# Configure
mousereach-setup
```

This choice is important because:
1. **Network/Remote Drive Support:** Y: drive can be a remote network location (even accessed through a server)
2. **Multi-Machine Setup:** Both analysis PC and GPU machine access the same Y: drive for pipeline folders
3. **Data Organization:** Keeping code and working folders on the same drive simplifies path management
4. **Future Scalability:** Works with mapped drives, UNC paths, and institutional storage systems

**You can install elsewhere if needed** (e.g., local C: drive for testing), but Y: is the production default. If you use a different drive, update the environment variable configuration accordingly (see "Part 6: Configure Pipeline Paths").

---

## Quick Install (TL;DR) - Default Y: Drive Setup

If you just want to get running fast on the Y: drive:

```bash
# 1. Copy MouseReach folder to Y: drive
Copy from:  Y:\Behavior\MouseReach\MouseReach\  (or wherever you're reading this from)
To:         Y:\Behavior\MouseReach\MouseReach\  (on target machine)

# 2. Open Anaconda Prompt and run:
cd Y:\Behavior\MouseReach\MouseReach
install.bat

# 3. Run configuration wizard
mousereach-setup
```

That's it! Then type `mousereach` to launch.

---

## Copying MouseReach to a New Machine

### Default: Install on Y: Drive

The recommended installation location is:

```
Y:\Behavior\MouseReach\MouseReach\
```

This keeps code, environment, and pipeline folders on the same Y: drive.

### What to Copy

Copy the entire `MouseReach` folder to the target machine:

```
From: Y:\Behavior\MouseReach\MouseReach\  (source machine)
To:   Y:\Behavior\MouseReach\MouseReach\  (target machine)
```

**Required contents:**
```
MouseReach/
├── src/
│   └── mousereach/           ← Main package (all steps in one place)
│       ├── config.py
│       ├── state.py
│       ├── launcher.py
│       ├── video_prep/  (Step 0)
│       ├── dlc/         (Step 1)
│       ├── segmentation/(Step 2)
│       ├── reach/       (Step 3)
│       ├── outcomes/    (Step 4)
│       ├── kinematics/  (Step 5)
│       └── export/      (Step 6)
├── pyproject.toml
├── install.bat          ← Run this to install
├── INSTALL.md           ← You're reading this
└── USER_GUIDE.md
```

### How to Copy

**Option A: USB Drive (Default)**
1. Copy `Y:\Behavior\MouseReach\MouseReach\` to USB
2. Plug into new machine
3. Copy to `Y:\Behavior\MouseReach\MouseReach\`

**Option B: Network Share**
1. Map the Y: network drive on new machine
2. Copy folder directly

**Option C: Zip and Transfer**
1. Right-click MouseReach folder → Send to → Compressed folder
2. Transfer zip file
3. Extract to `Y:\Behavior\MouseReach\MouseReach\`

### Alternative: Install on Different Drive

If you need to install on a different drive (C:, D:, or local path):

```
To:   C:\MouseReach\  (or D:\MouseReach\, or local path)
```

This works fine for testing, but remember:
- Update environment variable configuration after install (Part 6)
- GPU machine must use the SAME drive path for pipeline folders
- NAS/archive paths (X: drive) can be different

---

## Multi-Machine Considerations

MouseReach is designed for a **2-machine workflow**:

1. **Analysis PC** (main development machine) - Where you'll run this installation
2. **GPU Machine** (optional, for fast DLC processing) - Separate machine with NVIDIA GPU

The Y: drive location you choose here affects the GPU machine:

| Component | Machine | Location |
|-----------|---------|----------|
| MouseReach Code | Analysis PC | `Y:\Behavior\MouseReach\MouseReach\` |
| Conda Environment | Analysis PC | `Y:\conda_envs\mousereach\` |
| Pipeline Folders (DLC_Queue, Processing, Failed) | **Both** (shared) | `Y:\Behavior\MouseReach_Pipeline\` |
| NAS Archive (raw videos) | Both (shared) | X: drive (configurable) |

**Key Point:** The GPU machine must be able to reach the SAME `Y:\Behavior\MouseReach_Pipeline\` folder. This means:
- Both machines need network access to Y: drive
- Or Y: drive is actually a mapped network location (UNC path)
- Set environment variables identically on both machines (Part 6)

If you're setting this up for a single machine only, you can use any drive. But if you plan to use a GPU machine later, use the Y: drive recommendation to keep things consistent.

---

## Prerequisites

Before installing, the new machine needs:

1. **Anaconda or Miniconda** - [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   - Download the Windows 64-bit installer
   - Run installer, accept defaults
   - Check "Add to PATH" if asked

2. **NVIDIA GPU + Drivers** (for fast DLC processing)
   - Update drivers: [nvidia.com/drivers](https://www.nvidia.com/drivers)
   - Not required but strongly recommended

3. **FFmpeg** (for video cropping in Step 0)
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Or install via conda: `conda install -c conda-forge ffmpeg`

---

## Part 1: Create Conda Environment (on Y: Drive)

**Important:** Create the conda environment ON the same Y: drive where MouseReach code lives. This keeps everything together and works with the default configuration.

### Step 1a: Choose Your Environment Location

The default location is:
```
Y:\conda_envs\mousereach\
```

Or if you prefer a different structure:
```
Y:\Behavior\conda_envs\mousereach\
```

### Step 1b: Open a Terminal and Set Environment Path

Open a terminal with conda access:
- **VS Code:** Open Y:\Behavior\MouseReach\MouseReach\ → Terminal → New Terminal
- **Or:** Open Anaconda Prompt from Start Menu

If using VS Code and conda isn't recognized, run this once in Anaconda Prompt first:
```bash
conda init powershell
```
Then restart VS Code.

### Step 1c: Create Environment on Y: Drive

Create the environment on the Y: drive with an explicit prefix:

```bash
# Use -p (prefix) to specify the exact location on Y: drive
conda create -p Y:\conda_envs\mousereach python=3.10 -y

# Activate the environment
conda activate Y:\conda_envs\mousereach
```

Alternatively, if you prefer a different Y: location:
```bash
conda create -p Y:\Behavior\conda_envs\mousereach python=3.10 -y
conda activate Y:\Behavior\conda_envs\mousereach
```

> ⚠️ **Important:**
> - You must run `conda activate Y:\conda_envs\mousereach` (or your chosen path) every time you open a new terminal
> - Using `-p` (prefix) instead of `-n` (named environment) lets you control the exact location
> - This keeps the environment on the same Y: drive as your code

---

## Part 2: Install DeepLabCut

DeepLabCut requires conda for proper installation (especially GPU support):

```bash
# Install DeepLabCut with GPU support via conda-forge
conda install -c conda-forge deeplabcut -y

# Pin numpy to compatible version (DLC install may upgrade it)
pip install numpy==1.26.4
```

### Verify DLC Installation

```bash
python -c "import deeplabcut; print(f'DLC version: {deeplabcut.__version__}')"
```

### Verify GPU Detection (Optional but Recommended)

```bash
python -c "import tensorflow as tf; print('GPUs available:', len(tf.config.list_physical_devices('GPU')))"
```

If GPU is not detected, try:
```bash
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.6 -y
```

---

## Part 3: Install Additional Dependencies

```bash
# Napari for review tools (with all extras)
pip install "napari[all]"

# OpenCV for video processing (may already be installed)
pip install opencv-python

# Excel export support
pip install openpyxl

# Additional utilities
pip install tqdm
```

### Fix NumPy/TensorFlow Compatibility

The above may upgrade NumPy to a version incompatible with TensorFlow. Fix this:

```bash
# Downgrade numpy to compatible version
pip install numpy==1.26.4

# Install missing TensorFlow dependencies
pip install keras-preprocessing libclang tensorflow-io-gcs-filesystem
pip install "protobuf>=3.9.2,<3.20"
```

### Verify DLC Still Works

```bash
python -c "import deeplabcut; print('DLC OK')"
```

If this prints "DLC OK", continue. If you get errors, reinstall DeepLabCut:
```bash
conda install -c conda-forge deeplabcut -y --force-reinstall
```

---

## Part 4: Install MouseReach Pipeline

Navigate to the MouseReach folder:

```bash
cd Y:\Behavior\MouseReach\MouseReach
```

### Install MouseReach (Single Unified Package)

```bash
# Install MouseReach (includes all steps in one package)
pip install -e .
```

This gives you:
- The `mousereach` command which launches all tools in a single napari window with tabs
- All individual CLI commands (mousereach-crop, mousereach-segment, etc.)
- Napari plugins for all review tools

### Fix NumPy (Important!)

The install may upgrade NumPy. Pin it back:

```bash
pip install numpy==1.26.4
```

---

## Part 5: Verify Installation

```bash
# Check Python and key packages
python --version                    # Should be 3.10.x
python -c "import deeplabcut; print('DeepLabCut: OK')"
python -c "import napari; print('Napari: OK')"
python -c "import cv2; print('OpenCV: OK')"

# Check MouseReach CLI commands
mousereach-crop --help              # Step 0: Video cropping
mousereach-dlc-quality --help       # Step 1: DLC quality check
mousereach-segment --help           # Step 2: Segmentation
mousereach-detect-reaches --help    # Step 3: Reach detection
mousereach-detect-outcomes --help   # Step 4: Pellet outcomes
mousereach-grasp-analyze --help     # Step 5: Kinematic features
mousereach-export --help            # Step 6: Export to Excel

# Check Napari plugin
napari  # Once Napari loads, click on plugins in toolbar. Should see: Plugins → MouseReach Segmentation → Boundary Review Tool
```

---

## Part 6: Configure Pipeline Paths

### Step 1: Set Your Machine Role

Each PC in the lab plays a specific role. Setting the role tells the setup
wizard what defaults to use:

```bash
# See available roles
mousereach-setup --list-roles

# Set this PC's role (pick one)
mousereach-setup --set-role "NAS / DLC PC"        # Direct-attached NAS + GPU
mousereach-setup --set-role "GPU Filming PC"       # Filming/cropping workstation
mousereach-setup --set-role "Processing Server"    # Reach detection + kinematics
```

This saves a small file (`~/.mousereach/machine_role.json`) that the wizard
reads on this machine. If you skip this step, the wizard will try to auto-detect
based on drive patterns, or prompt manually.

### Step 2: Run the Setup Wizard

```bash
mousereach-setup
```

The wizard prompts for:
1. **Data Repository** - Root of your long-term storage (NAS mount)
2. **Processing Root** (REQUIRED) - Parent folder for DLC_Queue, Processing, Failed
3. **Watcher** (optional) - Automated pipeline for the NAS/DLC PC

If you set a machine role, defaults are pre-filled — just hit Enter through.

Configuration is saved to `~/.mousereach/config.json`.

### Step 3: Build Pipeline Index (Important for Fast Startup!)

```bash
mousereach-index-rebuild
```

This creates `pipeline_index.json` in your PROCESSING_ROOT folder. Without this
index, the dashboard and batch widgets will scan folders on every startup (slow
on network drives).

### Verify Configuration

```bash
mousereach-setup --show
```

---

## Daily Usage

Every time you start working:

```bash
# 1. Open Anaconda Prompt (or terminal)

# 2. Activate environment (using your chosen path from Part 1)
conda activate Y:\conda_envs\mousereach

# 3. Navigate to MouseReach (optional but recommended)
cd Y:\Behavior\MouseReach\MouseReach

# 4. Launch MouseReach (unified interface with all tools)
mousereach

# Or launch with a video pre-loaded:
mousereach "path/to/video.mp4"

# Or launch only review tools (2b, 3b, 4b):
mousereach --reviews "path/to/video.mp4"

# Or launch only batch processing tools:
mousereach --batch
```

### Alternative: Individual CLI Commands

```bash
mousereach-segment --help      # Step 2 batch segmentation
mousereach-detect-reaches      # Step 3 batch reach detection
mousereach-detect-outcomes     # Step 4 batch outcome detection
```

---

## CLI Commands Reference

| Step | Module | Main Command | Review Command |
|------|--------|--------------|----------------|
| 0 | mousereach.video_prep | `mousereach-crop` | - |
| 1 | mousereach.dlc | `mousereach-dlc-batch` | `mousereach-dlc-quality` |
| 2 | mousereach.segmentation | `mousereach-segment` | `mousereach-segment-review` (Napari) |
| 3 | mousereach.reach | `mousereach-detect-reaches` | `mousereach-review-reaches` |
| 4 | mousereach.outcomes | `mousereach-detect-outcomes` | `mousereach-review-pellet-outcomes` |
| 5 | mousereach.kinematics | `mousereach-grasp-analyze` | `mousereach-grasp-review` |
| Sync | mousereach.sync | `mousereach-sync` | `mousereach-sync --status` |
| 6 | mousereach.export | `mousereach-export` | `mousereach-summary` |

---

## Troubleshooting

### Slow Startup on Network Drives (napari takes 2-3 minutes)

If your conda environment is on a network drive (Y:, mapped drive, UNC path), **napari will be extremely slow** (2-3 minutes) because it scans all installed packages over the network during plugin discovery.

**THE FIX: Move conda environment to local C: drive:**
```bash
# Create NEW environment on local C: drive
conda create -p C:\conda_envs\mousereach python=3.10 -y
conda activate C:\conda_envs\mousereach

# Install MouseReach (code stays on Y:, packages on C:)
cd Y:\Behavior\MouseReach\MouseReach
pip install -e .
```

This keeps:
- **Code on Y:** (for multi-machine access, version control)
- **Packages on C:** (fast local storage for napari startup)

After this change, `mousereach` should start in 30-60 seconds instead of 3 minutes.

**Why this works:** Napari scans all installed packages at startup to discover plugins. On a network drive, each package check is a network round-trip. With 200+ packages, this takes minutes. On a local drive, it takes seconds.

**PowerShell execution policy issues:**
```powershell
mousereach-fix-powershell  # Run once to fix script execution on network drives
```

### "conda: command not found"
- Use **Anaconda Prompt**, not regular Command Prompt or PowerShell

### "No module named 'deeplabcut'"
- Make sure environment is activated: `conda activate mousereach`
- Reinstall: `conda install -c conda-forge deeplabcut -y`

### DLC is slow (using CPU instead of GPU)
- Check GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- Update NVIDIA drivers
- Install CUDA toolkit: `conda install -c conda-forge cudatoolkit=11.8 -y`

### "No module named 'napari'"
```bash
pip install "napari[all]"
```

### Napari plugin not showing
```bash
# First activate your conda environment
conda activate Y:\conda_envs\mousereach

# Navigate to MouseReach
cd Y:\Behavior\MouseReach\MouseReach

# Reinstall with force
pip install -e . --force-reinstall

# Restart napari
```

### PyQt5 / Qt errors
```bash
pip uninstall PyQt5 PyQt5-sip PyQt5-Qt5 -y
pip install PyQt5
```

### Permission denied errors
- Run Anaconda Prompt as Administrator, OR
- Use `pip install --user -e .`

---

## Environment Backup

Once everything works, save your environment:

```bash
conda env export > environment.yml
```

To recreate later:
```bash
conda env create -f environment.yml
```

---

## Starting Fresh

If you need to completely reinstall:

```bash
conda deactivate
conda env remove -n mousereach
# Then start over from Part 1
```

---

## Version Info

This guide was tested with:
- Python 3.10
- DeepLabCut 2.3.x
- TensorFlow 2.x
- CUDA 11.8
- Windows 10/11

---

## Appendix: Deploy the Watcher on the NAS PC (Step by Step)

**What this does:** Sets up the NAS-attached PC to automatically process new
reaching videos. Once running, you drop collage videos on the NAS and the
watcher handles everything — cropping, DLC, segmentation, reach detection,
outcomes, and archiving.

**Before you start, make sure:**
- The PC can see the NAS (its drive letter shows up in File Explorer)
- You know where your DLC model lives (a file called `config.yaml` — ask if unsure)

---

### Step 1: Open a Terminal

1. Click the **Start** button (bottom-left corner of the screen)
2. Type **Anaconda Prompt**
3. Click on **Anaconda Prompt** when it appears in the search results
4. A black window with white text will open — this is your terminal

---

### Step 2: Install Git (if you don't already have it)

Type this and press **Enter**:

```
git --version
```

**If you see a version number** (like `git version 2.43.0`), Git is already
installed. Skip to Step 3.

**If you see an error** like `'git' is not recognized`, you need to install Git:

1. Open a web browser
2. Go to: **https://git-scm.com/download/win**
3. The download should start automatically — if not, click the download link
4. Run the installer that downloads
5. Click **Next** on every screen — accept all the defaults
6. When it finishes, **close your Anaconda Prompt and open a new one**
   (Git won't work in the old window)

Now try `git --version` again. You should see a version number.

---

### Step 3: Download MouseReach from GitHub (First Time Only)

**If MouseReach is already on this PC**, skip to Step 4.

**If this is a brand new PC** and MouseReach has never been downloaded here:

**3a.** First, go to the drive where you want to put MouseReach.

> **Which drive?** Pick any drive with enough space. Each lab is different.
>
> *BlackmoreLab setup:*
>
> | PC | MouseReach lives on |
> |----|---------------------|
> | Processing Server | **Y:** drive |
> | NAS / Watcher PC | **A:** drive |
> | GPU Filming PCs | **A:** drive |
>
> *Other labs:* Use whatever drive makes sense for your setup.
> Ask your lab manager if you're not sure.

Type your drive letter and press **Enter**. For example:

```
A:
```

Then go to the root of that drive:

```
cd \
```

**3b.** Now download MouseReach from GitHub. Type this exactly and press **Enter**:

```
git clone https://github.com/LoganFriedrich/MouseReach.git
```

You'll see progress messages as it downloads. When it's done, you'll have a
new folder called `MouseReach` on your drive.

**3c.** Go into that folder:

```
cd MouseReach
```

You can verify where you are by typing `cd` by itself — it will print the
full path (e.g. `A:\MouseReach` or `Y:\MouseReach`).

Now skip to Step 5 (you already have the latest code since you just downloaded it).

---

### Step 4: Update MouseReach (If Already Downloaded)

If MouseReach was already on this PC from a previous install, update it:

**4a.** Go to the MouseReach folder (use your PC's drive letter — see the
table in Step 3):

```
cd A:\MouseReach
```

**4b.** Download the latest changes:

```
git pull
```

You should see a list of files being updated. If you see "Already up to date",
that's fine — it means you already have the latest version.

---

### Step 5: Install MouseReach into Python

**5a.** First, create the MouseReach environment (skip this if you already did
it before — it won't hurt to try, it will just say it already exists):

```
conda create -n mousereach python=3.10 -y
```

**5b.** Activate the environment:

```
conda activate mousereach
```

You should see `(mousereach)` appear at the beginning of the line.

**5c.** Install MouseReach:

```
pip install -e .
```

(That's `pip install -e` followed by a space and a period/dot.)

Wait for it to finish. You'll see several lines scroll by.
When you see your cursor back with `(mousereach)` at the start, it's done.

> **Note:** If this PC needs DeepLabCut for the watcher, also do Parts 2–3
> from the main install guide above (install DLC, napari, opencv, etc.)
> before continuing.

---

### Step 6: Set Up the Lab Profiles File

This file tells MouseReach about the PCs in your lab.

**6a.** Type this to copy the template:

```
copy src\mousereach\setup\lab_profiles.json.example src\mousereach\setup\lab_profiles.json
```

**6b.** Now open the new file in Notepad to edit it:

```
notepad src\mousereach\setup\lab_profiles.json
```

**6c.** Notepad will open with the file. Find this line:

```
"dlc_config_path": "CHANGE_ME: path to your DLC config.yaml",
```

**6d.** Replace `CHANGE_ME: path to your DLC config.yaml` with the actual
path to your DLC model's config.yaml file.

*BlackmoreLab example:*

```
"dlc_config_path": "A:\\AIs\\MPSA-LF-2025-10-27\\config.yaml",
```

*Other labs:* Use your own DLC model path.

**Important:** Use double backslashes `\\` between folder names, not single ones.

**6e.** Save the file: press **Ctrl+S**, then close Notepad.

---

### Step 7: Tell This PC It's the Watcher

Type this exactly, then press **Enter**:

```
mousereach-setup --set-role "Watcher PC"
```

You should see:

```
Machine role set to: Watcher PC
  Saved to: C:\Users\YourName\.mousereach\machine_role.json
  Description: PC with direct-attached storage and GPU for DLC inference. Runs the watcher.

Run 'mousereach-setup' to configure paths for this role.
```

---

### Step 8: Run the Setup Wizard

Type this exactly, then press **Enter**:

```
mousereach-setup
```

The wizard will ask you a series of questions. Because you set the role in
Step 7, it already knows the answers — **just press Enter for each question**
to accept the defaults.

It will ask about:
1. **Data Repository** (where your NAS is) — press **Enter**
2. **Processing Root** (where pipeline folders are) — press **Enter**
3. **Watcher settings** (DLC path, GPU, polling) — press **Enter** for each

When it says "Configuration saved!", you're done with setup.

---

### Step 9: Test the Watcher (Dry Run)

Before starting for real, do a test run that doesn't actually process anything:

```
mousereach-watch --dry-run
```

This shows what the watcher *would* do without actually doing it.
Check that the paths look right and there are no errors.

---

### Step 10: Start the Watcher

When you're ready to process videos for real:

```
mousereach-watch
```

The watcher is now running. It will:
1. Check the NAS every 30 seconds for new collage videos
2. Crop them into single-animal videos
3. Run DLC pose tracking (using the GPU)
4. Run segmentation, reach detection, and outcome detection
5. Archive the results

**To stop the watcher:** Press **Ctrl+C** in the terminal window.

**To check status without stopping:** Open a second Anaconda Prompt, activate
mousereach, and run:

```
mousereach-watch --status
```

---

### Quick Reference Card

Once setup is done, daily usage is just:

```
conda activate mousereach
mousereach-watch
```

That's it. Leave the terminal window open and the watcher runs in the background.

---

*MouseReach v2.3.0 - Last updated: 2025-01-21*

**Changes in this version:**
- Clarified default installation on Y: drive as the recommended setup
- Updated Part 1 to explicitly create conda environment on Y: drive
- Added "Multi-Machine Considerations" section explaining Y: drive importance
- Updated all path examples to use Y: drive as primary with C: as alternative
- Fixed all references from A: drive to Y: drive
- Added Appendix: Deploy the Watcher on the NAS PC
