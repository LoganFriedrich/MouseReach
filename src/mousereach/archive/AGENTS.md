<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# archive/

## Purpose
Post-pipeline archival system: Moves fully validated videos from the `Processing/` working directory to the long-term NAS archive. A video can only be archived when ALL validation stages (segmentation, reach detection, outcome classification) are marked "validated" in the pipeline index. Files are organized by experiment code on the NAS (e.g., `{NAS_DRIVE}/Analyzed/Sort/CNT/`).

## Key Files
| File | Description |
|------|-------------|
| `cli.py` | Command-line interface for `mousereach-archive` with dry-run and status modes |
| `core.py` | Archive logic: validation checks, file moving, NAS destination routing, index updates |

## For AI Agents

### Working In This Directory
- Archive is the FINAL step - videos exit the pipeline only through this module
- Safety check: ALL three validation statuses must be "validated" before archiving
  - `seg_validation == "validated"`
  - `reach_validation == "validated"`
  - `outcome_validation == "validated"`
- Files moved, not copied (removes from Processing/ after successful move)
- Destination determined by experiment code extracted from animal ID (e.g., `CNT0101` → `CNT/` folder)
- After archiving, video is removed from pipeline index
- Uses `PipelineIndex` for validation status checks and post-archive cleanup

### CLI Commands
```bash
# Archive all ready videos
mousereach-archive

# Archive specific video
mousereach-archive 20250704_CNT0101_P1

# Dry-run (preview without moving)
mousereach-archive --dry-run

# List videos ready for archive
mousereach-archive --list

# Check archive status for a video
mousereach-archive --status 20250704_CNT0101_P1
```

### Typical Workflow
1. Videos complete all pipeline stages and validation
2. Run `mousereach-archive --list` to see ready videos
3. Run `mousereach-archive` to move all ready videos to NAS
4. Videos are removed from Processing/ and index

### Error Handling
- If any validation status is not "validated", archive is blocked with clear error message
- If files don't exist in Processing/, archive fails
- Partial failures logged (some files moved, some failed)
- Index only updated on successful archive of all files

## Dependencies

### Internal
- `mousereach.config`: `Paths` (Processing folder, NAS paths), `get_video_id()`, `AnimalID.get_experiment()`
- `mousereach.index.PipelineIndex`: Validation status queries, video removal after archive

### External
- `shutil`: File moving operations
- `pathlib.Path`: Path manipulation

<!-- MANUAL: -->
