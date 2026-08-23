# Reaching the database

Describes: `src/mousereach/sync/` (`database.py`, `watcher.py`, `cli.py`, `__init__.py`, `AGENTS.md`), the seven places in the pipeline that call it, and `src/mousereach/pipeline/manifest.py` for the values it copies into the provenance columns.

Verified against: 4c54e46 (2026-08-23)

---

## What this subsystem is

One thing is written: a table called `reach_data`, in the SQLite database file at `Y:\LAB_ROOT\Databases\connectome.db`. One row per reach. Alongside it, a flat comma-separated file — a full dump of that table — at `Y:\LAB_ROOT\Databases\database_dump\reach_data.csv`.

The input is always one file: `{video}_features.json`, the last thing the pipeline writes for a video. Nothing else is ever read into the table. The reach detector's own output, the outcome detector's own output, the reach-assignment file — none of them are read here. Whatever did not make it into the features file cannot reach the database through this path.

Both destination paths are hard-coded module constants (`database.py:44`, `database.py:53`). There is no configuration setting for either.

---

## Read this first: at this commit, no write can succeed

**Every attempt to write a row fails, and the failure is not reported anywhere.**

The table has a column `segment_num` — which pellet, 1 through 20, the reach was aimed at. It is declared `INTEGER NOT NULL` with no default, both in the `CREATE TABLE` in the source (`database.py:107`) and in the live table on disk.

The code builds `segment_num` into each row correctly, taking it from the enclosing segment (`database.py:568`). But the list of column names used to build the `INSERT` statement, `ALL_COLUMNS` (`database.py:84-92`), does not contain it. Commit `5bac3b0` removed `'segment_num'` from `REACH_JSON_COLUMNS` (the list `ALL_COLUMNS` is built from) and added it back only to the row dictionary, never to the insert list. So the statement omits a column that must not be null, and SQLite rejects it:

```
sqlite3.IntegrityError: NOT NULL constraint failed: reach_data.segment_num
```

Reproduced twice at this commit, both times with a real features file (`20250624_CNT0102_P1_features.json`) and a scratch database: once against a table built from the source's own `CREATE_REACH_DATA_SQL`, and once against a replica of the live table's schema. In both cases the first `INSERT` raises and no row is written.

The error then disappears through three layers:

1. `sync_features_file` catches it and re-raises it as `RuntimeError` (`database.py:649-650`).
2. `sync_file_to_database` — the function every pipeline stage calls — catches every exception and returns `False` (`database.py:857-858`).
3. Every caller either ignores that return value or logs at warning or debug level. `orchestrator.py:2130` logs `"Database sync skipped ... (subject not in DB or DB unavailable)"` at debug level — a message that names two causes, neither of which is the actual one.

Consequences when reading the current data:

- The `reach_data` table is frozen. Its newest row was imported `2026-08-20T14:25:47`, about twenty seconds after commit `5bac3b0` landed. Nothing since.
- The flat CSV is frozen too, because the export only runs after a sync that did not raise. (One narrow exception: see "The flat CSV dump" below.)
- The record of which files have been synced (`.mousereach_sync_state.json`) is not updated, because the hash is recorded only after a successful commit (`database.py:641-645`). So `mousereach-sync-status` keeps reporting the same files as pending forever.
- **No data was lost.** The `DELETE` and the `INSERT`s share one transaction; the first `INSERT` raises, the connection closes without committing, and the delete is rolled back. Verified: a pre-existing row for the video being re-synced was still present after the failed sync.

Restoring `'segment_num'` to the insert list is the whole fix.

---

## What triggers a write

There are three doors into this code. Two work; one is broken.

### Door 1 — the per-video call, `sync_file_to_database` (`database.py:812`)

This is how the pipeline writes. It takes one file path and does everything: read, delete, insert, save state, dump the CSV. It never raises.

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

One further caller disables the function outright: `review/staging.py:202-206` replaces `sync_file_to_database` with a no-op so that staging a video for human review does not touch the database. That is deliberate.

### Door 2 — the batch command, `mousereach-sync` (`cli.py:16`)

Scans the `Processing` folder for `*_features.json` and syncs the ones that are new or whose contents changed. Flags: `--force` (ignore the change check and re-sync everything), `--dry-run` (build the rows, count them, write no rows), `--verbose` (print error messages at the end).

The scan is a non-recursive glob of one folder (`database.py:511`). Videos already archived out of `Processing` are not found by this command; they can only be written through Door 1.

`--dry-run` is not fully read-only. Before it does anything else the command calls `get_status`, which calls `check_database`, which calls `ensure_reach_data_table` (`database.py:483`) — so a dry run can still create the table and add missing columns. It writes no rows and does not save sync state (`database.py:355-356`).

`mousereach-sync-status` (`cli.py:143`) reports the same counts without writing rows (same caveat about the schema).

### Door 3 — the folder watcher, `mousereach-sync-watch` — **does not run**

`cli.py:137` calls `start_watcher(debounce_seconds=args.debounce, blocking=True)`. `start_watcher` (`watcher.py:219`) has no `debounce_seconds` parameter. The call raises `TypeError: start_watcher() got an unexpected keyword argument 'debounce_seconds'`, which the surrounding `except ValueError` (`cli.py:138`) does not catch. The command has never done anything but print a traceback. Verified by calling it exactly as the command-line code does.

The underlying `PipelineWatcher` class works if driven from Python, but note two things if you ever do: it watches one folder non-recursively (`watcher.py:173-177`), and its sync path (`watcher.py:97-114`) calls `sync_features_file` directly, so it **neither saves the sync-state file nor writes the CSV dump**. A watcher-driven write updates the table only.

---

## Which videos are eligible

A file is skipped, silently, unless all of these hold:

1. The name ends in `_features.json` (`database.py:832`, `database.py:511`).
2. The video name contains `CNT` followed by four digits, or the form `CNT_##_##` (`parse_subject_id`, `database.py:229`). This is how the subject identifier is derived: `CNT0115` becomes `CNT_01_15`. **Anything not named `CNT` is dropped.** On this machine today that is 38 of the 1,146 features files in `Processing`, all of them ASPA videos. (These counts are local to one machine and drift; re-measure before quoting them.)
3. The subject already exists in the `subjects` table (`database.py:515` for the batch path, `database.py:848-850` for the per-video path). A new mouse that has not been registered in the database produces no rows and no message.
4. For the batch path only: the file's contents changed since last time. `needs_sync` (`database.py:520`) compares the first 16 hex characters of the file's SHA-256 against `.mousereach_sync_state.json`, which lives inside the `Processing` folder. Delete that file and everything re-syncs.

`check_database` (`database.py:466`) additionally refuses to do anything if the database file has no `subjects` table (`database.py:479-480`), and the engine refuses if the database file does not exist (`database.py:370-371`).

All 1,108 `CNT`-named features files in `Processing` today match both the date pattern and the tray/run pattern described below.

---

## What one write does, in order

`sync_features_file` (`database.py:530`):

1. Load the features JSON.
2. Take `video_name` from inside the file, falling back to the file name (`database.py:551`).
3. Load the sibling manifest `{video}_processing_manifest.json` from the same folder for the provenance columns (`database.py:556`).
4. Parse the date, tray type and run number out of the video name.
5. Walk `segments`, then `reaches` inside each segment, building one row per reach.
6. **If there are zero reaches, return without touching the database** (`database.py:610-611`). See the warning below.
7. If `dry_run`, return the count without inserting (`database.py:613-614`).
8. Create the table if absent, then attempt eleven `ALTER TABLE ... ADD COLUMN` statements for columns added over time (`database.py:388-414`). Each one fails harmlessly if the column already exists. Because Door 1 constructs a fresh syncer per video, this runs once per video.
9. `DELETE FROM reach_data WHERE video_name = :video_name`.
10. Insert every row, one statement at a time.
11. Commit. Steps 9 to 11 are one transaction.
12. Record the file's hash in the in-memory sync state (written to disk later by the caller). If the file lives outside `processing_path` the key falls back to the bare file name (`database.py:641-645`).

---

## A write replaces the previous rows, and no copy is kept

Step 9 above is the important one:

```sql
DELETE FROM reach_data WHERE video_name = :video_name
```
(`database.py:625-628`)

Every existing row for that video is destroyed before the new rows go in. There is no archive table, no version column, no soft delete, no `_archived` copy. Once a video is re-synced, its previous numbers are gone from the database and can only be recovered by re-running the pipeline on the old code.

Rows for other videos are untouched — the replacement is scoped to `video_name` and nothing else.

Three consequences worth stating explicitly:

- **Fewer reaches means orphan rows vanish.** If a re-run detects 40 reaches where the old run found 55, the extra 15 rows are deleted and not replaced. This is intended — it is how reprocessing avoids blending two algorithm versions (see the comment at `reprocess_to_current.py:309-312`) — but it means row counts drop with no message.
- **Zero reaches means the old rows survive.** The early return at `database.py:610-611` happens before the delete. A video that reprocesses down to no reaches at all keeps its previous rows in the table forever, now attributed to a run that no longer produces them. This is the one case where the table and the files on disk can disagree without anything being wrong upstream.
- **Columns filled by anything other than this code are wiped.** `test_phase` and `phase_group` are columns in the live table that MouseReach never writes. The delete removes the row that held them and the insert does not put them back. This has already happened: across a 400-video sample, every row still stamped with the old kinematics extractor (`extractor_version` 1.0.0, imported April 2026) carries a `test_phase`, and every row re-synced in August 2026 under extractor 2.0.0 has `test_phase` null — 15,048 rows with a value, 40,812 without.

`sync_file_to_database` never consults the sync-state file, so it always deletes and re-inserts, even for a file it just wrote a moment ago.

`imported_at` is a single timestamp taken once per write (`database.py:553`) and stamped identically on every row of that video. It marks when the rows were copied in, not when the video was processed.

---

## What depends on the segments being right

This layer does not compute or check segmentation. It copies the grouping it is handed. That makes it a straight amplifier for any boundary error: one misplaced segment boundary moves a reach from one pellet's group to another's, and every field below moves with it.

**Taken from the enclosing segment and stamped identically onto every reach in it** (`database.py:565-574`):

`segment_num`, `segment_outcome`, `segment_outcome_confidence`, `segment_outcome_flagged`, `attention_score`, `pellet_position_idealness`.

A reach on the wrong side of a boundary is therefore labelled with the wrong pellet number, and it inherits the wrong pellet's outcome, the wrong attention score and the wrong pellet-positioning number. Anything joined on pellet number — manual pellet scoring above all — attaches to the wrong reach.

**Copied from the reach, but computed upstream from its position inside its segment:**

`reach_num`, `is_first_reach`, `is_last_reach`, `n_reaches_in_segment`. Checked against 25 features files (3,600 reaches): `reach_num` is always the reach's 1-based index inside its segment, `is_first_reach` is true exactly for index 0, and `n_reaches_in_segment` always equals the length of that segment's reach list. Move one boundary and all four change for the reaches on both sides of it.

**Derived downstream inside SQLite:** `segment_contact_group` is a generated column computed from `segment_outcome`, so it carries the same error onward.

**Row ordering:** the flat CSV is ordered by subject, date, video, `segment_num`, `reach_num` (`database.py:783`), so the dump's row order shifts too.

Three further things this layer does not do:

- **No validation of the grouping.** Nothing checks that `segment_num` falls in 1..20, that segment numbers are contiguous, or that there are twenty of them. The table currently holds 4,302 rows with `segment_num = 0`, a value no pellet has.
- **Pellets with no reaches are absent, not zero.** A segment with an empty reach list contributes no row at all. You cannot count "pellets the mouse never reached for" from `reach_data`; those pellets are simply missing.
- **Re-segmentation leaves no trace.** Because the replace is keyed on `video_name` alone, re-syncing a re-segmented video silently rewrites every row's pellet number with no record of what it was before.

---

## Where each column's value comes from

The insert list is `ALL_COLUMNS` (`database.py:84-92`), 61 names — 62 if `segment_num` is restored.

**Session identity (5 columns)**

| Column | Source |
|---|---|
| `subject_id` | derived from the video name, `CNT0115` → `CNT_01_15` (`database.py:229`) |
| `video_name` | the `video_name` key inside the features file, else the file name |
| `session_date` | first eight digits of the name, reformatted `YYYY-MM-DD`; **empty string, not null, if absent** (`database.py:582`) |
| `tray_type` | letters matched by `CNT\d{4}_([A-Za-z]+)(\d+)$`, upper-cased (`database.py:285`); null if the name does not end that way |
| `run_number` | digits from the same pattern; null if absent |

**Which pellet (1 column)**

`segment_num` is read from the enclosing segment, never from the reach (`database.py:568`). This is deliberate: reaches used to carry a copy that was always `0`. As noted above, at this commit the column is built and then dropped before the insert.

The comment at `database.py:56-60` states that all 160,141 rows imported from 2026-08-03 onward were written with `segment_num = 0`. **The table does not show that.** Of the 165,642 rows imported on or after 2026-08-03, 746 have `segment_num = 0`; across all 361,284 rows only 4,302 do, and 3,556 of those were imported in February 2026. Whatever happened, do not carry the 160,141 figure forward — it does not describe the table anyone will query.

**The reach measurements (41 columns)**

Copied verbatim, by name, from each reach dictionary (`database.py:587-591`). The list is `REACH_JSON_COLUMNS` (`database.py:61-78`) and covers: identity (`reach_id`, `reach_num`), the outcome link (`outcome`, `causal_reach`, `interaction_frame`, `distance_to_interaction`), position in the segment (`is_first_reach`, `is_last_reach`, `n_reaches_in_segment`), frames (`start_frame`, `apex_frame`, `end_frame`, `duration_frames`), extent, velocity, trajectory shape, hand and head angles, grasp aperture, tracking quality, the review flags, and the human-review provenance (`outcome_source`, `reviewed_by`, `algo_outcome`, `algo_causal_reach_id`).

Rules of the copy:

- A key missing from the reach becomes `NULL`. No warning.
- Four fields are forced to `1`/`0` for SQLite: `causal_reach`, `is_first_reach`, `is_last_reach`, `flagged_for_review` (`database.py:81`). A `None` is left as `None`, so it is not converted — and all four are `NOT NULL` in the live table.
- **No arithmetic happens here.** This layer does not compute, convert units, or fill gaps. If a number is wrong or missing in the database, it was wrong or missing in the features file.
- A key present in the features file but absent from this list is dropped. `reach_source` is currently in that position — the feature extractor writes it on every reach and there is no column for it.

**The extended, per-paw measurements (1 column)**

`extended_features` holds `json.dumps(reach['extended'] or {})` (`database.py:595`) — the whole per-paw feature set as one text blob, roughly 7 kB per reach in current output. Read it with SQLite's `json_extract`, or expand it in pandas. Two caveats: rows written before the column existed are `NULL` (all the extractor-1.0.0 rows are), and features files produced by extractor 1.0.0 have no `extended` block, so those rows would get the two-character string `{}` if re-synced.

**Segment-level context, copied onto every reach in the segment (5 columns)**

`segment_outcome` (from the segment's `outcome`), `segment_outcome_confidence`, `segment_outcome_flagged` (forced to 1/0, never null), `attention_score`, `pellet_position_idealness` (`database.py:565-574`).

**Bookkeeping (3 columns)**

`source_file` is the **file name only**, not the path (`database.py:601`) — so you cannot tell from the row whether the file came from `Processing` or from an archived cohort folder. `extractor_version` is the features file's own top-level version string, defaulting to the literal `'unknown'` (`database.py:552`). `imported_at` as described above.

---

## The provenance columns

Six columns, all read from `{video}_processing_manifest.json` sitting next to the features file (`_load_provenance`, `database.py:417`):

| Column | Manifest key | What it actually means |
|---|---|---|
| `processed_by` | `processed_by` | hostname of the machine that ran the pipeline (`manifest.py:242`) — **not** the machine that ran the sync |
| `mousereach_version` | `pipeline_versions.mousereach` | version of the MouseReach package when the manifest was written |
| `dlc_scorer` | `dlc_model.dlc_scorer` | the pose-estimation model suffix, e.g. `DLC_resnet101_MPSAOct27shuffle3_100000`, taken from the pose file's own name (`manifest.py:56-60`) |
| `segmenter_version` | `pipeline_versions.segmenter` | read out of `{video}_segments.json` when the manifest was written |
| `reach_detector_version` | `pipeline_versions.reach_detector` | read out of `{video}_reaches.json` |
| `outcome_detector_version` | `pipeline_versions.outcome_detector` | read out of `{video}_pellet_outcomes.json` |

Each version is whatever that stage stamped into its own output file (`manifest.py:152-158`, `manifest.py:233-236`), so these columns record what each stage *claimed*, which is only as good as the stage's stamp.

That distinction is not academic. Until commit `61d98b9` the segmenter stamped a constant belonging to a different module than the one that actually cut the boundaries, so `segmenter_version` records the writing module's number rather than the running segmenter's. What the column actually contains today: `2.1.0` on 191,828 rows (53%), `2.1.3` on 165,642 rows (46%), and empty on 3,814 rows. The same split appears in the manifests the column is read from — of 1,146 manifests in `Processing`, 947 say `2.1.3` and 199 say `2.1.0`. So the shorthand "the corpus reads 2.1.3" is wrong; it is a two-way split, and the backfill script (`scripts/backfill_segmenter_version.py`) only rewrites files that currently claim `2.1.3` and postdate 2026-07-08. It corrects files, not database rows.

**If the manifest is missing or unreadable, all six columns are `NULL` and nothing is logged** (`database.py:437-438`, `database.py:463-464`). A null here is indistinguishable from a manifest that genuinely recorded nothing. 3,814 rows are in that state.

Things the manifest holds that are **not** copied into the database: the reach-assignment stage's version, the kinematics extractor version under `pipeline_versions` (the `extractor_version` column carries the same number, but read from the features file instead), the per-stage validation statuses, and the per-step timestamps.

---

## Columns that are declared and never filled

Measured over all 361,284 rows of the table, split by which kinematics extractor produced the features file (`1.0.0`: 195,642 rows, imported through April 2026; `2.0.0`, the current one: 165,642 rows, imported August 2026).

**Empty on every row, both extractors.** `distance_to_interaction`, `grasp_aperture_max_mm`, `grasp_aperture_at_contact_mm`, `apex_distance_to_pellet_mm`, `lateral_deviation_mm`, `tracking_quality_score`, `flag_reason`. The keys exist in the features file and carry no value on any reach. `flagged_for_review` is present on every row but is `0` on every row, which tells you nothing.

**Filled by the old extractor, empty under the current one.** `max_extent_pixels`, `max_extent_ruler` and `max_extent_mm` carry a value on all 195,642 extractor-1.0.0 rows and on none of the 165,642 extractor-2.0.0 rows — a clean split, no partial cases. The same pattern holds in the files: across 120 sampled features files, extractor-1.0.0 reaches always carry extent and extractor-2.0.0 reaches never do. So reach extent is not "always null", but the current pipeline stopped producing it, and every new row will lack it. `segment_outcome_confidence` behaves identically: 100% present on extractor-1.0.0 rows, absent on every extractor-2.0.0 row (the key is present in the features file and its value is `null`).

**Mostly empty by nature.** The per-reach `outcome` and `interaction_frame` carry a value on about 4% of rows — they are populated only for the reach the outcome stage tied to the pellet. `segment_outcome` is on 100% of rows.

`docs/FIELD_AUDIT.md` (generated by `python -m mousereach.pipeline.field_audit --markdown`) is the right tool for the file-by-file view, but do not use it as the authority on the table: it reads a 1,377-video subset of videos judged "finished and current at every stage" and takes the database side from a parquet snapshot. It reports `max_extent_*` at 0.5% of the database; the table is at 54%. Query the table when you want to know what is in the table.

**Present in the live table, never written by this code.** `test_phase` and `phase_group` were added to the database by something outside MouseReach. Sync neither writes nor clears them, so a re-synced video loses them (see above).

**Computed by the database itself.** `contact_group` and `segment_contact_group` are generated columns in the live table, derived from `interaction_frame` and `segment_outcome`. The string `contact_group` does not appear anywhere in `database.py`. This is a general caution: the `CREATE TABLE` in the source runs only `IF NOT EXISTS`, so the live table has drifted ahead of it — different column types, extra columns, a differently-named unique constraint. Read the schema from the database, not from the source.

**Uniqueness.** The live table carries `UNIQUE (video_name, reach_id)`; the source writes it as `UNIQUE(video_name, reach_id)` (`database.py:203`). Reach identifiers are unique per video across all segments — checked against 80 features files, no duplicates — so this constraint does not currently bite.

---

## The flat CSV dump

Path: `Y:\LAB_ROOT\Databases\database_dump\reach_data.csv`.

`CSV_DUMP_PATH` is derived from the module-level `DB_PATH` (`database.py:53`), *not* from the syncer's `db_path`. Point a `DatabaseSyncer` at a different database and it will still overwrite this one shared CSV.

**When it is written:**

- At the end of `sync_all`, only if at least one file counted as synced (`database.py:687-688`).
- After every single-video call through `sync_file_to_database` (`database.py:854`) — so once per video, every video.
- Never by the folder watcher.

It is a full dump of the whole table each time, ordered by subject, date, video, pellet and reach (`database.py:783`). The file is 198 MB at 361,284 rows, so the per-video path rewrites 198 MB of network storage for every video processed.

**What it contains:** 57 columns, listed explicitly in the query at `database.py:760-781`. Seven ordinary columns of the table are **not** exported: `outcome_source`, `reviewed_by`, `algo_outcome`, `algo_causal_reach_id`, `extended_features`, `test_phase` and `phase_group`. Anyone working from the CSV cannot see whether a reach's outcome came from the algorithm or from a human reviewer, and cannot see any per-paw measurement. The row `id` and the two generated columns are also absent.

**Header alignment.** The header now comes from the query's own column names (`database.py:792`). It used to be a separately maintained list, and on 2026-08-20 the two drifted: the header was written from `ALL_COLUMNS` while the rows came from the `SELECT`, so 61 names sat over 57-value rows and every column from the sixth onward was mislabelled. **The file currently on disk is that broken one** (written 2026-08-20 20:18, before the fix): the header has 61 names, every one of the 361,284 rows has 57 values, and the column headed `causal_reach` holds outcome strings such as `displaced_sa`. Treat the current `reach_data.csv` as unusable for anything keyed on column name.

It will not regenerate through the normal path while the `segment_num` defect stands, because the export only runs after a sync that did not raise. There is one exception: a features file containing no reaches at all returns early without inserting anything, which counts as success, so syncing such a file would rewrite the dump with the corrected header.

**Failures.** The export catches everything and logs a warning (`database.py:804-809`). It used to catch everything and say nothing, which made an export that failed every time look exactly like one that succeeded.

---

## Configuration

| Setting | Where | Effect |
|---|---|---|
| `processing_root` | `~/.mousereach/config.json`, or the environment variable `MouseReach_PROCESSING_ROOT` (`config.py:56`) | the folder scanned and watched is `<processing_root>/Processing` (`config.py:152`). On this machine `C:\LAB_ROOT\Behavior\MouseReach_Pipeline`. If unset, the syncer's `processing_path` is `None`, the batch scan finds nothing, and the sync-state file is neither read nor written — but the per-video path still works, because it is handed an explicit file. |
| database location | constructor argument `db_path` only | no configuration key, no environment variable. Defaults to the hard-coded `Y:\LAB_ROOT\Databases\connectome.db`. |
| CSV location | none | hard-coded, and not affected by `db_path`. |
| `dry_run` | constructor argument, `mousereach-sync --dry-run` | builds and counts rows, inserts nothing, does not save sync state (`database.py:355-356`). Can still create the table and add columns. |
| `force` | `mousereach-sync --force` | ignores the file-hash check and re-syncs every eligible file. |
| `debounce_seconds` | `mousereach-sync-watch --debounce` | **no effect** — the command crashes before reaching the watcher. `PipelineWatcher` accepts the value if driven from Python, but `start_watcher` never passes it on. |

There is no setting that turns database writing off. The only code that suppresses it does so by replacing the function at runtime (`review/staging.py:202-206`).

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

`watcher.py:89-90` and `watcher.py:113-114` log sync failures at error level, which is the one path that says anything useful.

`sqlalchemy` and `watchdog` are optional dependencies, declared under the `sync` extra in `pyproject.toml:56-58`. Without `sqlalchemy` the engine raises `ImportError` (`database.py:368-369`) and the per-video path swallows it. Both are installed in the `mousereach` environment on this machine, so they are not the current cause of anything.

---

## What this folder's own `AGENTS.md` gets wrong

It is dated 2026-01-26 and has not kept up. Do not rely on it:

- It puts the database at `PROCESSING_ROOT/../MouseDB/connectome.db` (line 24). It is at `Y:\LAB_ROOT\Databases\connectome.db`.
- It documents flags `--status`, `--export` and `--watch` on `mousereach-sync` (lines 32-34). None exist. The real flags are `--force`, `--dry-run` and `--verbose`; status and watching are separate commands.
- It says five columns are always null (line 26). Those five really are always null, but the list is short — see the section above.
- It says "Sync is atomic per video: DELETE all rows for video_name, then INSERT new rows in a transaction" (line 23). That description of the mechanism is accurate, but at this commit the transaction always rolls back.

`SYNC_PATTERNS` at `database.py:208` is dead code — defined, referenced nowhere in `src/` or `scripts/`.

---

## Contested claims

This document was written from the source, then checked by a second reader
whose job was to disprove it. The statements below are ones they disputed and
that were not resolved. Neither side is authoritative: spot-checking found the
checker wrong at least once. **Do not rely on anything listed here without
opening the code yourself.** Everything not listed survived two passes.

- **"**Two stages do not clean.** `stage_1` and `stage_2` compute pillar geometry straight from raw positions."**
  - disputed because: Three untouched stages, not two, compute pillar geometry from uncleaned data. Stage 6 also never cleans: it does not import `clean_dlc_bodyparts` at all and passes the raw slice straight into the geometry function. This matters for the paragraph's own argument, which uses the count to bound how much of the output rests on uncleaned tray corners.
- **"**The analysis window is the boundary.** Every stage slices the pose data positionally as `dlc_df.iloc[seg_start : seg_end - 5 + 1]`."**
  - disputed because: Four committing stages -- 26, 27, 28 and 29 -- never take that slice. They slice per-reach pre/post windows bounded by the neighbouring reaches and capped at `seg_end + 1`, so they can read the last 5 frames the clean zone excludes. Stages 0, 22 and 99 slice nothing at all. Together stages 26-29 account for 3,031 of the 43,180 archived decisions, so this is not a corner case.
- **"The flag's only real effects are the misleading message and a different progress format." (about `mousereach-detect-outcomes --legacy`)**
  - disputed because: `--legacy` has a large substantive effect the document denies, and it is the opposite of what the flag's name suggests: it is the only CLI path that loads reaches correctly. The `--legacy` branch goes through `process_batch` -> `process_single`, which uses `core/batch.py`'s nested-format-aware `_extract_reaches`; the default v6 branch uses `cli.py`'s flat-only copy, which the document itself shows
- **"The confidence of SABL and SABR is never checked. **No function in `pillar_geometry.py` reads a likelihood column.**"**
  - disputed because: The second sentence is false. `pellet_inside_pillar_circle` reads `Pellet_likelihood` and gates on it, and the module defines a likelihood threshold constant for exactly that purpose. The first sentence (SABL/SABR confidence is never checked) is correct and is the load-bearing point; the supporting generalisation is not.
- **"Each of the seven threshold-disabled ones carries a dated comment explaining what was tried and why it did not reach acceptable accuracy, and says the file is kept for documentation."**
  - disputed because: Wrong on both halves. Stage 14 carries no dated comment and no explanation -- just the bare word `disabled` on the constant line. And only three of the seven (15, 18, 19) say the file is kept for documentation; stages 14, 20, 23 and 24 do not.
- **"`outcomes/cli.py:20-24` and `outcomes/core/__init__.py:18-23` both advertise all of them as live categories." (referring to `displaced_outside`, `no_pellet` and `uncertain`)**
  - disputed because: cli.py:20-24 lists only four categories -- retrieved, displaced_sa, displaced_outside, untouched. `no_pellet` and `uncertain` do not appear there. They appear only in `core/__init__.py:22-23` and in the review tool's OUTCOMES list. The claim that both files advertise all three is false for cli.py.
- **"pillar centre = midpoint of SABL/SABR, moved `0.944 * ruler` upward" / "`stage_1` and `stage_2` compute pillar geometry straight from raw positions"**
  - disputed because: The formula is not applied to the raw per-frame SABL/SABR positions. `compute_pillar_geometry_series` first smooths both corners with a 3-frame centered moving average by default, and every cascade stage calls it with that default. So no stage -- including 1 and 2 -- works from "raw positions", and the described geometry recipe is missing a step that changes the numbers.
- citation could not be resolved: ``core/pellet_outcome.py:135` for the `causal_reach_frame` field -- line 135 is blank. The field is declared at line 134 (`causal_reach_frame: Optional[int] = No`

