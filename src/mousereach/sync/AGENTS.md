<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-26 -->

# sync/

## Purpose
Database synchronization module that syncs `_features.json` files (Step 5 output) into a flat `reach_data` SQLite table for direct SQL/Excel analysis. Each row represents one reach with its linked outcome, kinematic features, and session context.

## Key Files
| File | Description |
|------|-------------|
| `database.py` | Core sync logic: table creation, features JSON parsing, row insertion, CSV export |
| `cli.py` | CLI entry points for `mousereach-sync` command |
| `watcher.py` | File system watcher for auto-syncing new `_features.json` files |
| `__init__.py` | Package exports |

## For AI Agents

### Working In This Directory
- The `reach_data` table has ~60 columns: session identity, reach identity, outcome linkage, kinematic features, segment context, metadata
- Video name parsing extracts session_date, tray_type, run_number from `YYYYMMDD_CNTxxxx_TypeRun` format
- Subject ID conversion: `CNT0115` → `CNT_01_15` (matches mousedb schema)
- Sync is atomic per video: DELETE all rows for video_name, then INSERT new rows in a transaction
- The database lives at `PROCESSING_ROOT/../MouseDB/connectome.db` (shared with mousedb package)
- Only `_features.json` is synced (not `_reaches.json` or `_pellet_outcomes.json` separately)
- 5 feature columns are always NULL (not yet implemented): `tracking_quality_score`, `apex_distance_to_pellet_mm`, `lateral_deviation_mm`, `grasp_aperture_max_mm`, `grasp_aperture_at_contact_mm`

### CLI Commands
```bash
mousereach-sync                  # Sync all _features.json to reach_data table
mousereach-sync --force          # Re-sync all (delete and re-import)
mousereach-sync --status         # Show sync status
mousereach-sync --export         # Export reach_data.csv
mousereach-sync --watch          # Watch Processing/ for new features files
```

## Dependencies

### Internal
- `mousereach.config` - Paths for Processing root
- `mousereach.kinematics` - Produces `_features.json` files that this module consumes

### External
- `sqlalchemy` - Database ORM (optional dependency via `pip install -e ".[sync]"`)
- `watchdog` - File system monitoring for `--watch` mode (optional)

<!-- MANUAL: -->
