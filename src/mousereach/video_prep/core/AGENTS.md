<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# core

## Purpose
Core video preparation algorithms for splitting 8-camera collages into single-animal videos. Handles cropping, file organization, and format conversion.

## Key Files
| File | Description |
|------|-------------|
| `cropper.py` | Video cropping and file management - splits 8-camera collage (2x4 grid) into individual MP4s, moves files between pipeline stages, converts MKV to MP4 |

## For AI Agents

### Algorithm Details

**Collage Layout** (8-camera 2x4 grid):
```
Position:  1      2      3      4     (top row, Y=0)
Position:  5      6      7      8     (bottom row, Y=540)
           X=0    X=480  X=960  X=1440
```
- Total collage size: 1920x1080
- Each cell: 480x540 pixels
- Input format: `.mkv` from multi-camera recording system
- Output format: `.mp4` (H.264, cropped video + copied audio)

**Filename Convention**:
```
Input:  YYYYMMDD_ID1,ID2,ID3,ID4,ID5,ID6,ID7,ID8_P1.mkv
Output: YYYYMMDD_ID1_P1.mp4, YYYYMMDD_ID2_P1.mp4, etc.
```
- Animal ID format: `{experiment}{cohort:2d}{subject:2d}` (e.g., CNT0101)
- Cohort "00" → blank position (skipped)
- Experiment codes: CNT, ENCR, OPT, etc.

**Processing Pipeline**:
1. `crop_collage()` - Split one collage into 8 videos (skip blanks)
2. `crop_all()` - Process entire directory of collages
3. `copy_to_dlc_queue()` - Move cropped videos to DLC processing queue
4. `archive_collages()` - Archive processed collages
5. `sort_to_experiment_folders()` - Organize final outputs by experiment

**FFmpeg Commands**:
- Cropping: `ffmpeg -y -i input.mkv -filter:v "crop=W:H:X:Y" -c:a copy output.mp4`
- Format conversion: `ffmpeg -y -i input.mkv -c copy output.mp4`

### Modifying This Code

**Path Configuration**:
- Paths derived from environment variables via `mousereach.config.Paths`
- `MouseReach_PROCESSING_ROOT` → Working folders (REQUIRED - run `mousereach-setup`)
- `MouseReach_NAS_DRIVE` → Archive location (optional)
- Reconfigure using `mousereach-setup` CLI command

**Changing Crop Coordinates**:
- Edit `CROP_COORDS` array in `cropper.py`
- Format: `(width, height, x_offset, y_offset)` in pixels
- Must match actual collage layout from recording system
- Test with single file before batch processing

**Adding New Experiment Codes**:
- Experiment codes extracted via regex: `r'_([A-Z]+)\d{4}'`
- No hardcoded list - automatically detected from filenames
- Files sorted into folders matching experiment code

**File Movement vs Copy**:
- Most functions accept `move: bool` parameter
- Use `move=True` for pipeline progression (save disk space)
- Use `move=False` for archival copies (preserve originals)

**Error Handling**:
- Functions return result dicts with `status` field
- Status values: `'success'`, `'skipped'`, `'failed'`, `'error'`
- Batch operations continue on individual failures

**Blank Position Detection**:
- `is_blank_animal()` checks cohort digits (positions 3:5 in ID)
- Cohort "00" → skip (no animal present)
- Prevents wasted DLC processing time

## Dependencies
- **External**: `ffmpeg` (must be in system PATH)
- **Internal**: `mousereach.config.Paths` for path configuration
- **Upstream**: Raw collage videos from recording system
- **Downstream**: Cropped videos → DLC processing queue → rest of pipeline

<!-- MANUAL: -->
