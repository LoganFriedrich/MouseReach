<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# video_prep

## Purpose
Step 0 of the MouseReach pipeline: Crops 8-camera multi-animal collage videos (2x4 grid) into individual single-animal videos for downstream DLC processing. Handles MKV-to-MP4 conversion, video compression for UI previews, and preparation of videos for the DLC Queue.

## Key Files
| File | Description |
|------|-------------|
| `cli.py` | CLI entry points for `mousereach-crop`, `mousereach-convert`, `mousereach-prep` |
| `widget.py` | Napari GUI widget for interactive video cropping workflow |
| `compress.py` | Creates 75% resolution preview videos using FFmpeg for review widgets |
| `core/cropper.py` | Core cropping logic - parses filenames, runs FFmpeg crop operations, manages file movement |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `core/` | Core cropping and file management algorithms |

## For AI Agents

### Working In This Directory
- **Input format:** Multi-animal collages use comma-separated animal IDs in filenames: `YYYYMMDD_{ID1},{ID2},...,{ID8}_{tray}.mkv`
- **Output format:** Single-animal videos: `YYYYMMDD_{animal_id}_{tray}.mp4`
- **Blank detection:** Animal IDs with cohort "00" (e.g., `CNT0001`) are skipped (empty camera positions)
- **FFmpeg dependency:** All cropping operations require FFmpeg in PATH
- **Crop coordinates:** Hardcoded 2x4 grid at 480x540 per cell (1920x1080 source)

### CLI Commands
```bash
# Crop single collage
mousereach-crop -i collage.mkv

# Batch crop directory
mousereach-crop -i /path/to/collages/

# Full workflow: crop + copy to DLC queue + archive
mousereach-prep -i /path/to/collages/ --archive

# Create compressed previews for review widgets
python -m mousereach.video_prep.compress /path/to/videos/
```

### Pipeline Integration
- **Input source:** `Paths.MULTI_ANIMAL_SOURCE` (configurable via `MouseReach_NAS_DRIVE`)
- **Output destination:** `Paths.SINGLE_ANIMAL_OUTPUT` → copied to `Paths.DLC_QUEUE`
- **Preview creation:** Automatically generates `{video}_preview.mp4` files at 75% resolution

## Dependencies

### Internal
- `mousereach.config.Paths` - Environment-based path configuration

### External
- `ffmpeg` (required, system binary)
- `opencv-python` (cv2) - Video metadata extraction for compression
- `napari` - GUI widget framework
- `qtpy` - Qt bindings for widget UI

<!-- MANUAL: -->
