# Reaching the database

Describes: `src/mousereach/sync/` (`database.py`, `watcher.py`, `cli.py`, `__init__.py`, `AGENTS.md`), the seven places in the pipeline that call it, and `src/mousereach/pipeline/manifest.py` for the values it copies into the provenance columns.

Verified against: 61d98b9 (2026-08-21)

---

## What this subsystem is

One thing is written: a table called `reach_data`, in the SQLite database file at `Y:\LAB_ROOT\Databases\connectome.db`. One row per reach. Alongside it, a flat comma-separated file — a full dump of that table — at `Y:\LAB_ROOT\Databases\database_dump\reach_data.csv`.

The input is always one file: `{video}_features.json`, the last thing the pipeline writes for a video. Nothing else is ever read into the table. The reach detector's own output, the outcome detector's own output, the reach-assignment file — none of them are read here. Whatever did not make it into the features file cannot reach the database through this path.

Both destination paths are hard-coded module constants (`database.py:44`, `database.py:53`). There is no configuration setting for either.

---

## Read this first: at this commit, no write can succeed

**Every attempt to write a row fails, and the failure is not reported anywhere.**

The table has a column `segment_num` — which pellet, 1 through 20, the reach was aimed at. The column is declared `INTEGER NOT NULL` with no default (`database.py:107`), and the live table on disk has the same declaration.

The code builds `segment_num` into each row correctly, taking it from the enclosing segment (`database.py:568`). But the list of column names used to build the `INSERT` statement, `ALL_COLUMNS` (`database.py:84-92`), does not contain `segment_num`. It was removed from `REACH_JSON_COLUMNS` in commit 5bac3b0 and added back only to the row dictionary, never to the insert list. So the statement omits a column that must not be null, and SQLite rejects it:

```
sqlite3.IntegrityError: NOT NULL constraint failed: reach_data.segment_num
```

Reproduced end to end at 61d98b9 against a fresh database built from `CREATE_REACH_DATA_SQL` and a real features file (`20260731_CNT0508_P4_features.json`): the first `INSERT` raises, and no row is written.

The error then disappears through three layers:

1. `sync_features_file` catches it and re-raises as `RuntimeError` (`database.py:649-650`).
2. `sync_file_to_database` — the function every pipeline stage calls — catches every exception and returns `False` (`database.py:857-858`).
3. Every caller either ignores the return value or logs at warning/debug level. `orchestrator.py:2129-2130` logs `"Database sync skipped (subject not in DB or DB unavailable)"` at debug level — a message that names two causes, neither of which is the actual one.

Consequences to be aware of when reading the current data:

- The `reach_data` table is frozen. Nothing written since this defect landed.
- The flat CSV is frozen too, because the export only runs after a successful sync.
- The record of which files have been synced (`.mousereach_sync_state.json`) is never updated, so `mousereach-sync-status` keeps reporting the same files as pending forever.
- **No data was lost.** The `DELETE` and the `INSERT`s share one transaction; the first `INSERT` raises, the connection closes without committing, and the delete is rolled back. Verified: a pre-existing row for the video being re-synced was still present after the failed sync.

Adding `'segment_num'` to `ALL_COLUMNS` is the whole fix.

---

## What triggers a write

There are three doors into this code. Two work; one is broken.

### Door 1 — the per-video call, `sync_file_to_database` (`database.py:812`)

This is how the pipeline writes. It takes one file path, and it does everything: read, delete, insert, save state, dump the CSV. It never raises.

It is called from seven places:

| Called from | Passes | Effect |
|---|---|---|
| `kinematics/cli.py:141-142` | the features file | writes |
| `pipeline/run_all.py:130-131` | the features file | writes |
| `watcher/orchestrator.py:1167-1168` | the features file | writes |
| `watcher/orchestrator.py:2124-2125` | the features file | writes |
| `pipeline/reprocess_to_current.py:315-317` | the features file, only for videos judged clean | writes |
| `reach/core/reach_detector.py:1137-1138` | `_reaches.json` | **does nothing** |
| `outcomes/core/pellet_outcome.py:1799-1800` | `_pellet_outcomes.json` | **does nothing** |

The last two return immediately at the file-name check (`database.py:832-833`). They look like sync points and are not. They have never written anything.

One further caller disables the function outright: `review/staging.py:203-206` replaces `sync_file_to_database` with a no-op so that staging a video for human review does not touch the database. That is deliberate.

### Door 2 — the batch command, `mousereach-sync` (`cli.py:16`)

Scans the `Processing` folder for `*_features.json` and syncs the ones that are new or whose contents changed. Flags: `--force` (ignore the change check and re-sync everything), `--dry-run` (build the rows, count them, write nothing), `--verbose` (print error messages at the end).

The scan is a non-recursive glob of one folder (`database.py:511`). Videos that have already been archived out of `Processing` are not found by this command; they can only be written through Door 1.

`mousereach-sync-status` (`cli.py:143`) reports the same counts without writing.

### Door 3 — the folder watcher, `mousereach-sync-watch` — **does not run**

`cli.py:137` calls `start_watcher(debounce_seconds=args.debounce, blocking=True)`. `start_watcher` (`watcher.py:219`) has no `debounce_seconds` parameter. The call raises `TypeError: start_watcher() got an unexpected keyword argument 'debounce_seconds'`, which the surrounding `except ValueError` does not catch. The command has never done anything but print a traceback. Verified by calling it exactly as the command-line code does.

The underlying `PipelineWatcher` class is functional if driven from Python, but note two things about it if you ever do: it watches one folder non-recursively (`watcher.py:173-177`), and its sync path (`watcher.py:97-114`) calls `sync_features_file` directly, so it **neither saves the sync-state file nor writes the CSV dump**. A watcher-driven write updates the table only.

---

## Which videos are eligible

A file is skipped, silently, unless all of these hold:

1. The name ends in `_features.json` (`database.py:832`, `database.py:511`).
2. The video name contains `CNT` followed by four digits, or the form `CNT_##_##` (`parse_subject_id`, `database.py:229`). This is how the subject identifier is derived: `CNT0115` becomes `CNT_01_15`. **Anything not named `CNT` is dropped.** Today that is 24 of the 1,088 features files in `Processing` — all of them ASPA videos.
3. The subject already exists in the `subjects` table (`database.py:515` for the batch path, `database.py:848-850` for the per-video path). A new mouse that has not been registered in the database produces no rows and no message.
4. For the batch path only: the file's contents changed since last time. `needs_sync` (`database.py:520`) compares the first 16 hex characters of the file's SHA-256 against `.mousereach_sync_state.json`, which lives inside the `Processing` folder. Delete that file and everything re-syncs.

`check_database` (`database.py:466`) additionally refuses to do anything if the database file has no `subjects` table.

---

## What one write does, in order

`sync_features_file` (`database.py:530`):

1. Load the features JSON.
2. Take `video_name` from inside the file, falling back to the file name (`database.py:551`).
3. Load the sibling manifest `{video}_processing_manifest.json` from the same folder for the provenance columns (`database.py:556`).
4. Parse the date, tray type and run number out of the video name.
5. Walk `segments`, then `reaches` inside each segment, building one row per reach.
6. **If there are zero reaches, return without touching the database** (`database.py:610-611`). See the warning below.
7. If `dry_run`, return the count without touching the database (`database.py:613-614`).
8. Create the table if absent, then attempt eleven `ALTER TABLE ... ADD COLUMN` statements for columns added over time (`database.py:388-414`). Each one fails harmlessly if the column already exists. Because Door 1 constructs a fresh syncer per video, this runs once per video.
9. `DELETE FROM reach_data WHERE video_name = :video_name`.
10. Insert every row, one statement at a time.
11. Commit. Steps 9 to 11 are one transaction.
12. Record the file's hash in the in-memory sync state (written to disk later by the caller).

---

## A write replaces the previous rows, and no copy is kept

Step 9 above is the important one:

```sql
DELETE FROM reach_data WHERE video_name = :video_name
```
(`database.py:625-628`)

Every existing row for that video is destroyed before the new rows go in. There is no archive table, no version column, no soft delete, no `_archived` copy. Once a video is re-synced, its previous numbers are gone from the database and can only be recovered by re-running the pipeline on the old code.

Rows for other videos are untouched — the replacement is scoped to `video_name` and nothing else.

Two consequences worth stating explicitly:

- **Fewer reaches means orphan rows vanish.** If a re-run detects 40 reaches where the old run found 55, the extra 15 rows are deleted and not replaced. This is intended (it is how reprocessing avoids blending two algorithm versions — see the comment at `reprocess_to_current.py:309-312`), but it means row counts drop silently.
- **Zero reaches means the old rows survive.** The early return at `database.py:610-611` happens before the delete. A video that reprocesses down to no reaches at all keeps its previous rows in the table forever, now attributed to a run that no longer produces them. This is the one case where the table and the files on disk can disagree without anything being wrong upstream.

`sync_file_to_database` never consults the sync-state file, so it always deletes and re-inserts, even for a file it just wrote a moment ago.

`imported_at` is a single timestamp taken once per write (`database.py:553`) and stamped identically on every row of that video. It marks when the rows were copied in, not when the video was processed.

---

## Where each column's value comes from

The insert list is `ALL_COLUMNS` (`database.py:84-92`), 61 names — 62 if `segment_num` is restored.

**Session identity (5 columns)**

| Column | Source |
|---|---|
| `subject_id` | derived from the video name, `CNT0115` → `CNT_01_15` (`database.py:229`) |
| `video_name` | the `video_name` key inside the features file, else the file name |
| `session_date` | first eight digits of the name, reformatted `YYYY-MM-DD`; **empty string, not null, if absent** (`database.py:582`) |
| `tray_type` | letters matched by `CNT\d{4}_([A-Za-z]+)(\d+)$`, upper-cased; null if the name does not end that way |
| `run_number` | digits from the same pattern; null if absent |

All 1,064 `CNT` features files currently in `Processing` match both patterns.

**Which pellet (1 column)**

`segment_num` is read from the enclosing segment, never from the reach (`database.py:568`). This is deliberate: reaches used to carry a copy that was always `0`, which is how 160,141 rows were written with no pellet number at all. As noted above, at this commit the column is built and then dropped before the insert.

**The reach measurements (38 columns)**

Copied verbatim, by name, from each reach dictionary (`database.py:587-591`). The list is `REACH_JSON_COLUMNS` (`database.py:61-79`) and covers: identity (`reach_id`, `reach_num`), the outcome link (`outcome`, `causal_reach`, `interaction_frame`, `distance_to_interaction`), position in the segment (`is_first_reach`, `is_last_reach`, `n_reaches_in_segment`), frames (`start_frame`, `apex_frame`, `end_frame`, `duration_frames`), extent, velocity, trajectory shape, hand and head angles, grasp aperture, tracking quality, the review flags, and the human-review provenance (`outcome_source`, `reviewed_by`, `algo_outcome`, `algo_causal_reach_id`).

Rules of the copy:

- A key missing from the reach becomes `NULL`. No warning.
- Four fields are forced to `1`/`0` for SQLite: `causal_reach`, `is_first_reach`, `is_last_reach`, `flagged_for_review` (`database.py:81`).
- **No arithmetic happens here.** This layer does not compute, convert units, or fill gaps. If a number is wrong or missing in the database, it was wrong or missing in the features file.
- A key present in the features file but absent from this list is dropped. `reach_source` is currently in that position — it is written on every reach by the feature extractor and has no column to go to.

**The extended, per-paw measurements (1 column)**

`extended_features` holds `json.dumps(reach['extended'] or {})` (`database.py:595`) — the whole per-paw feature set (one set of path, speed, spread and visibility numbers per paw landmark) as one text blob. It is never `NULL`; when the reach has no extended block it is the two-character string `{}`. Read it with SQLite's `json_extract`, or expand it in pandas.

**Segment-level context, copied onto every reach in the segment (5 columns)**

`segment_outcome` (from the segment's `outcome`), `segment_outcome_confidence`, `segment_outcome_flagged` (forced to 1/0, never null), `attention_score`, `pellet_position_idealness` (`database.py:565-574`).

**Bookkeeping (3 columns)**

`source_file` is the **file name only**, not the path (`database.py:601`) — so you cannot tell from the row whether the file came from `Processing` or from an archived cohort folder. `extractor_version` is the features file's own top-level version string, defaulting to the literal `'unknown'`. `imported_at` as described above.

---

## The provenance columns

Six columns, all read from `{video}_processing_manifest.json` sitting next to the features file (`_load_provenance`, `database.py:417`):

| Column | Manifest key | What it actually means |
|---|---|---|
| `processed_by` | `processed_by` | hostname of the machine that ran the pipeline (`manifest.py:242`) — **not** the machine that ran the sync |
| `mousereach_version` | `pipeline_versions.mousereach` | version of the MouseReach package at the time the manifest was written |
| `dlc_scorer` | `dlc_model.dlc_scorer` | the pose-estimation model suffix, e.g. `DLC_resnet101_MPSAOct27shuffle3_100000`, parsed out of the pose file's name |
| `segmenter_version` | `pipeline_versions.segmenter` | read out of `{video}_segments.json` when the manifest was written |
| `reach_detector_version` | `pipeline_versions.reach_detector` | read out of `{video}_reaches.json` |
| `outcome_detector_version` | `pipeline_versions.outcome_detector` | read out of `{video}_pellet_outcomes.json` |

Each version is whatever that stage stamped into its own output file (`manifest.py:152-158`, `manifest.py:233-236`), so these columns record what each stage *claimed*, which is only as good as the stage's stamp. That distinction is not academic: until commit 61d98b9 the segmenter stamped the wrong module's constant, so `segmenter_version` reads `2.1.3` on the entire existing corpus for work actually done by segmenter 2.2.3. There is a separate backfill script for that (`scripts/backfill_segmenter_version.py`); it corrects the files, not the database rows.

**If the manifest is missing or unreadable, all six columns are `NULL` and nothing is logged** (`database.py:437-438`, `database.py:463-464`). A null here is indistinguishable from a manifest that genuinely recorded nothing.

Things the manifest holds that are **not** copied into the database: the reach-assignment stage's version, the kinematics extractor version under `pipeline_versions` (the `extractor_version` column carries the same number, but from the features file instead), the per-stage validation statuses, and the per-step timestamps.

---

## Columns that are declared and never filled

Beyond `segment_num`, which cannot currently be inserted at all:

**Always null because nothing computes them.** `tracking_quality_score`, `apex_distance_to_pellet_mm`, `lateral_deviation_mm`, `grasp_aperture_max_mm`, `grasp_aperture_at_contact_mm`, `distance_to_interaction`. The keys exist in the features file and carry no value on any reach. `max_extent_pixels`, `max_extent_ruler` and `max_extent_mm` are in the same state — reach extent, one of the more obviously useful measurements the schema advertises, is null on essentially every row. See `docs/FIELD_AUDIT.md`, which is generated by `python -m mousereach.pipeline.field_audit --markdown` and is the authority on this; do not trust the folder's `AGENTS.md`, which lists five such columns.

**Present in the live table, never written by this code.** `test_phase` and `phase_group` were added to the database by something outside MouseReach. Sync neither writes nor clears them — so a re-synced video loses them, because the delete removes the row that held them and the insert does not put them back.

**Computed by the database itself.** `contact_group` and `segment_contact_group` are generated columns in the live table (derived from `interaction_frame` and `segment_outcome`). They do not appear in `CREATE_REACH_DATA_SQL` at all. This is a general caution: the `CREATE TABLE` in the source runs only `IF NOT EXISTS`, so the live table has drifted ahead of it and the source is not a reliable description of the real schema. Read the schema from the database.

**Uniqueness.** `UNIQUE(video_name, reach_id)` (`database.py:203`). Reach identifiers are unique per video across all segments — checked against 40 recent features files, no duplicates — so this constraint does not currently bite.

---

## The flat CSV dump

Path: `Y:\LAB_ROOT\Databases\database_dump\reach_data.csv`.

`CSV_DUMP_PATH` is derived from the module-level `DB_PATH` (`database.py:53`), *not* from the syncer's `db_path`. Point a `DatabaseSyncer` at a different database and it will still overwrite this one shared CSV.

**When it is written:**

- At the end of `sync_all`, only if at least one file was actually synced (`database.py:687-688`).
- After every single-video call through `sync_file_to_database` (`database.py:854`) — so once per video, every video.
- Never by the folder watcher.

It is a full dump of the whole table each time, ordered by subject, date, video, pellet and reach (`database.py:783`). The file is 198 MB at 361,284 rows, so the per-video path rewrites 198 MB of network storage for every video processed.

**What it contains:** 57 columns, listed explicitly in the query at `database.py:760-781`. Five columns of the table are **not** exported: `outcome_source`, `reviewed_by`, `algo_outcome`, `algo_causal_reach_id`, and `extended_features`. Anyone working from the CSV cannot see whether a reach's outcome came from the algorithm or from a human reviewer, and cannot see any per-paw measurement. The generated columns and the row `id` are also absent.

**Header alignment.** The header now comes from the query's own column names (`database.py:792`). It used to be a separately maintained list, and on 2026-08-20 the two drifted: 61 header names were written over 57-value rows, mislabelling every column from the sixth onward — the column headed `causal_reach` contained outcome strings. **The file currently on disk is that broken one** (written 2026-08-20 20:18, before the fix): 61 names, 57 values per row. It cannot regenerate until the `segment_num` defect is fixed, because the export only runs after a successful sync. Treat the current `reach_data.csv` as unusable.

**Failures.** The export catches everything and logs a warning (`database.py:804-809`). It used to catch everything and say nothing, which made an export that failed every time look exactly like one that succeeded.

---

## Configuration

| Setting | Where | Effect |
|---|---|---|
| `processing_root` | `~/.mousereach/config.json`, or the environment variable `MouseReach_PROCESSING_ROOT` | the folder scanned and watched is `<processing_root>/Processing`. On this machine `C:\LAB_ROOT\Behavior\MouseReach_Pipeline`. If unset, the syncer's `processing_path` is `None`, the batch scan finds nothing, and the sync-state file is neither read nor written — but the per-video path still works, because it is handed an explicit file. |
| database location | constructor argument `db_path` only | no configuration key, no environment variable. Defaults to the hard-coded `Y:\LAB_ROOT\Databases\connectome.db`. |
| CSV location | none | hard-coded, and not affected by `db_path`. |
| `dry_run` | constructor argument, `mousereach-sync --dry-run` | builds and counts rows, writes nothing, and does not save sync state (`database.py:355-356`). |
| `force` | `mousereach-sync --force` | ignores the file-hash check and re-syncs every eligible file. |
| `debounce_seconds` | `mousereach-sync-watch --debounce` | **no effect** — the command crashes before reaching the watcher. `PipelineWatcher` accepts the value if driven from Python, but `start_watcher` never passes it on. |

There is no setting that turns database writing off. The only code that suppresses it does so by replacing the function at runtime (`review/staging.py:203-206`).

---

## Failures that are swallowed

Collected in one place, because these are the reasons a write can produce nothing while everything looks normal:

| `database.py` | What is swallowed |
|---|---|
| `349-350` | unreadable sync-state file — treated as "nothing has ever been synced" |
| `361-362` | sync-state file that cannot be written — the next run re-syncs everything |
| `410-411`, `413-414` | every schema migration failure |
| `463-464` | any failure reading the processing manifest — all six provenance columns become null |
| `495-496` | failure to read the subject list — becomes an empty list, so **every file looks ineligible and the batch sync reports nothing to do** |
| `741-745` | failure counting rows for the status report — becomes zeros |
| `804-809` | CSV export failure (logged, not raised) |
| `857-858` | **every failure in the per-video write path** |

`watcher.py:89-90` and `113-114` log sync failures at error level, which is the one path that says anything useful.

`sqlalchemy` and `watchdog` are optional dependencies, declared under the `sync` extra in `pyproject.toml`. Without `sqlalchemy` the engine raises `ImportError` (`database.py:368-369`) and the per-video path swallows it. Both are installed in the `mousereach` environment on this machine, so they are not the current cause of anything.

---

## What this folder's own `AGENTS.md` gets wrong

It is dated 2026-01-26 and has not kept up. Do not rely on it:

- It puts the database at `PROCESSING_ROOT/../MouseDB/connectome.db`. It is at `Y:\LAB_ROOT\Databases\connectome.db`.
- It documents flags `--status`, `--export` and `--watch` on `mousereach-sync`. None exist. The real flags are `--force`, `--dry-run`, `--verbose`, and status and watching are separate commands.
- It says five columns are always null. The real figure is larger — see `docs/FIELD_AUDIT.md`.
- It says "Sync is atomic per video: DELETE all rows for video_name, then INSERT new rows in a transaction." That description of the mechanism is accurate, but at this commit the transaction always rolls back.

`SYNC_PATTERNS` at `database.py:208` is dead code — defined, referenced nowhere.
