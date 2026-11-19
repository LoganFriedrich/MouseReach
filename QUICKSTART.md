# MouseReach Quick Start Guide

**For users who just want to get started quickly.**

For detailed technical information, see [INSTALL.md](INSTALL.md) and [USER_GUIDE.md](USER_GUIDE.md).

---

## First Time Setup (One Time Only)

### Step 1: Install Miniconda

1. Go to: https://docs.conda.io/en/latest/miniconda.html
2. Download **Miniconda3 Windows 64-bit**
3. Run the installer
4. Accept all defaults
5. Restart your computer

### Step 2: Copy MouseReach to Your Computer

Copy the MouseReach folder to your computer:
- **From:** Network drive or USB
- **To:** `C:\MouseReach` (or wherever you prefer)

### Step 3: Run the Installer

1. Press the **Windows key**
2. Type **Anaconda Prompt** and click to open it
3. Type these commands:
   ```
   cd C:\MouseReach
   install.bat
   ```
4. Wait for installation to complete (~10-15 minutes)
5. When you see "INSTALLATION COMPLETE!", you're done!

---

## Daily Usage

### Launching MouseReach

Every time you want to use MouseReach:

1. Open **Anaconda Prompt** (search for it in Start Menu)
2. Type:
   ```
   conda activate mousereach
   mousereach
   ```

That's it! MouseReach will open with all tools in tabs.

### Loading a Video

**Option A: From the command line**
```
mousereach "C:\path\to\your\video.mp4"
```

**Option B: From a per-step review screen**
1. Open the screen with its command (see Reviewing below)
2. Click the **Select Video...** button
3. Navigate to your video file

### Reviewing

Open the **Review Queues** tab in `mousereach` (or run `mousereach-review-tool`).
The old review-only mode, `mousereach --reviews`, is retired: it opened the old
whole-video Review Tool, which saved corrections into the algorithm's own files,
where the next run erased them. It now prints a note and opens the usual tabs.

The per-step review screens below are still there and still save, each opened
on its own: `mousereach-segment-review`, `mousereach-review-pellet-outcomes`
and `mousereach-review-reaches` (the "Step 2b/3b/4b" names below are what these
screens were called as tabs of the retired Review Tool; the launcher no longer
has those tabs). Saving one marks
that step "validated"; when you run the steps by hand, without the watcher,
that is how you sign off a step that triage flagged for review. Saving in
`mousereach-segment-review` also marks the cuts as a person's (a re-run of
segmentation keeps them) and removes the old reaches, outcomes and kinematics
files of that video into an `_archived` folder, so run those steps again.

---

## The Pipeline at a Glance

```
Your Video → Crop → DLC → Segment → Reaches → Outcomes → Features → Export
              (0)   (1)     (2)       (3)        (4)        (5)       (6)
```

| Step | What it does | When to use |
|------|--------------|-------------|
| 0 | Crops 8-camera collages into individual videos | After recording |
| 1 | Runs DeepLabCut pose tracking | After cropping |
| 2 | Finds the 21 boundaries around the 20 pellet presentations | After DLC |
| 3 | Detects reaching movements | After segmentation |
| 4 | Classifies outcomes (Retrieved/Displaced/etc) | After reaches |
| 5 | Extracts kinematic features per reach | After outcomes |
| 6 | Exports results to Excel | When done |

---

## Keyboard Shortcuts

### Reach and outcome review screens (the boundary screen differs: Space sets the boundary, S saves, Enter accepts)
| Key | Action |
|-----|--------|
| **Space** | Play/Pause video |
| **←/→** | Previous/Next frame |
| **Shift+←/→** | Jump 10 frames |
| **Ctrl+S** | Save progress |

### Step 2b - Review Boundaries
| Key | Action |
|-----|--------|
| **N/P** | Next/Previous boundary |
| **Space** | Set boundary to current frame |
| **S** | Save validated boundaries |
| **Enter** | Accept the current boundary |

### Step 4b - Review Reaches
| Key | Action |
|-----|--------|
| **N/P** | Next/Previous reach |
| **S/E** | Set reach Start/End |
| **A** | Add new reach |
| **Delete** | Delete selected reach |

### Step 3b - Review Outcomes
| Key | Action |
|-----|--------|
| **N/P** | Next/Previous segment |
| **R** | Mark as Retrieved |
| **D** | Mark as Displaced (scoring area) |
| **O** | Mark as Displaced (outside) |
| **U** | Mark as Untouched |

---

## Common Issues

### "conda not found"
- Make sure you're using **Anaconda Prompt**, not regular Command Prompt
- Search for "Anaconda Prompt" in the Start Menu

### "mousereach not found"
- Make sure you activated the environment: `conda activate mousereach`
- If still not working, reinstall: `pip install -e .`

### Video won't load
- Make sure the file path has no special characters
- Try copying the video to a simple path like `C:\Videos\`

### Changes not saving
- Press **Ctrl+S** to save (in Step 2b - Review Boundaries, press **S**)
- Check that you have write permission to the folder

---

## Getting Help

1. Check [USER_GUIDE.md](USER_GUIDE.md) for detailed instructions
2. Check [INSTALL.md](INSTALL.md) for installation troubleshooting
3. Open an issue on GitHub

---

*MouseReach v2.16 | Quick Start Guide*
