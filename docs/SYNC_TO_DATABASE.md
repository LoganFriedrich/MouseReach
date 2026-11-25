# Reaching the database

Describes: `src/mousereach/sync/` (`database.py`, `watcher.py`, `cli.py`, `__init__.py`), the three places in the pipeline that call it and the one that switches it off, and `src/mousereach/pipeline/manifest.py` for the values it copies into the provenance columns.

The file:line references below were taken from an older version of the code and many no longer match; find the code by the function, class or constant named beside each reference.

---

## What this subsystem is

> **Status:** the watcher does not call this path at all. There is no in-pipeline `db_sync` step: the integration is an external database tool PULLING features files from the Analyzed tree, and a push step would always fail on configuration and log a failed step on every video. `sync_file_to_database` and the tools below still exist for a lab
> that configures `central_db` deliberately; nothing in the watcher invokes
> them.

One thing is written: a table called `reach_data`, in an OPTIONAL external SQLite database whose location is configuration -- `central_db` in `~/.mousereach/config.json` or the environment variable `MOUSEREACH_CENTRAL_DB` (there is no built-in default, and with no value set the sync does nothing and says so). Why optional: MouseReach must produce complete, usable output standing alone; pushing rows into a central database is an integration a lab may or may not run. One row per reach. Alongside it, a flat comma-separated file — a full dump of that table — as `database_dump/reach_data.csv` in the folder of the configured database.

The input is always one file: `{video}_features.json`, the last thing the pipeline writes for a video. Nothing else is ever read into the table. The reach detector's own output, the outcome detector's own output, the reach-assignment file — none of them are read here. Whatever did not make it into the features file cannot reach the database through this path.

Both destination paths come from one setting: `DB_PATH` (`database.py`) is `central_db` / `MOUSEREACH_CENTRAL_DB`, and the CSV path (`CSV_DUMP_PATH`) is derived from it.

---

## What triggers a write

There are three doors into this code. Two work; one is broken.

### Door 1 — the per-video call, `sync_file_to_database` (`database.py:812`)

This is how the pipeline writes. It takes one file path and does everything: read, delete, insert, save state, dump the CSV. It never raises.

It is called from three places:

| Called from | Passes | Effect |
|---|---|---|
| `kinematics/cli.py:154-155` | the features file | writes |
| `reach/core/reach_detector.py:1140-1141` | `_reaches.json` | **does nothing** |
| `outcomes/core/pellet_outcome.py:1799-1800` | `_pellet_outcomes.json` | **does nothing** |

The watcher (`watcher/orchestrator.py`), the Run Pipeline tab's "Run chosen videos..." (`pipeline/run_all.py`, which now runs the watcher's own steps, `watcher/run_one.py`) and the bring-current routine (`pipeline/reprocess_to_current.py`) are not on this list: none of them calls `sync_file_to_database` any more (bring-current's `sync_db` argument is accepted and ignored), and the database tool pulls filed results from Analyzed itself.

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
2. The video name carries an animal id that `AnimalID.parse` accepts (`{PROJECT}{CC}{SS}`), or the form `{PROJECT}_CC_SS` (`parse_subject_id`): `CNT0115` becomes `CNT_01_15`, `ASPA1011` becomes `ASPA_10_11`. A name with no parseable animal id is skipped.
3. ~~The subject already exists in the `subjects` table.~~ A subject the database has never heard of is CREATED from the video name -- with its cohort and project if those are missing too -- by `DatabaseSyncer.ensure_subject_exists`, called on both the per-video and the batch path. The reason is the usual order of events in a lab: videos are recorded and shipped every day, while the tracking sheet that names the animals may be filled in much later. Gating the machine data on the hand data meant one failing sheet import could silently drop an entire cohort's results. The created cohort gets the video's date as a placeholder `start_date` (the column is NOT NULL) and `notes` beginning `auto-created from video`; mousedb's sheet import recognises that marker and replaces the placeholder with the real start date when the sheet is finally imported. A subject or cohort that already exists is never touched. (Non-`CNT` names such as ASPA are parsed via `AnimalID.parse`, so item 2's "anything not named CNT is dropped" does not hold.)
4. For the batch path only: the file's contents changed since last time. `needs_sync` (`database.py:520`) compares the first 16 hex characters of the file's SHA-256 against `.mousereach_sync_state.json`, which lives inside the `Processing` folder. Delete that file and everything re-syncs.

`check_database` (`database.py:466`) refuses to do anything when no central database is configured, or if the database file has no `subjects` table (`database.py:479-480`), and the engine refuses if the database file does not exist (`database.py:370-371`).

The `CNT`-named features files checked all match both the date pattern and the tray/run pattern described below.

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

- **Fewer reaches means orphan rows vanish.** If a re-run detects 40 reaches where the old run found 55, the extra 15 rows are deleted and not replaced. This is intended — it is how reprocessing avoids blending two algorithm versions — but it means row counts drop with no message.
- **Zero reaches means the old rows survive.** The early return at `database.py:610-611` happens before the delete. A video that reprocesses down to no reaches at all keeps its previous rows in the table forever, now attributed to a run that no longer produces them. This is the one case where the table and the files on disk can disagree without anything being wrong upstream.
- **Columns filled by anything other than this code are wiped.** `test_phase` and `phase_group` are columns in the live table that MouseReach never writes. The delete removes the row that held them and the insert does not put them back.

`sync_file_to_database` never consults the sync-state file, so it always deletes and re-inserts, even for a file it just wrote a moment ago.

`imported_at` is a single timestamp taken once per write (`database.py:553`) and stamped identically on every row of that video. It marks when the rows were copied in, not when the video was processed.

---

## What depends on the segments being right

This layer does not compute or check segmentation. It copies the grouping it is handed. That makes it a straight amplifier for any boundary error: one misplaced segment boundary moves a reach from one pellet's group to another's, and every field below moves with it.

**Taken from the enclosing segment and stamped identically onto every reach in it** (`database.py:565-574`):

`segment_num`, `segment_outcome`, `segment_outcome_confidence`, `segment_outcome_flagged`, `attention_score`, `pellet_position_idealness`.

A reach on the wrong side of a boundary is therefore labelled with the wrong pellet number, and it inherits the wrong pellet's outcome, the wrong attention score and the wrong pellet-positioning number. Anything joined on pellet number — manual pellet scoring above all — attaches to the wrong reach.

**Copied from the reach, but computed upstream from its position inside its segment:**

`reach_num`, `is_first_reach`, `is_last_reach`, `n_reaches_in_segment`. Checked against a sample of features files: `reach_num` is always the reach's 1-based index inside its segment, `is_first_reach` is true exactly for index 0, and `n_reaches_in_segment` always equals the length of that segment's reach list. Move one boundary and all four change for the reaches on both sides of it.

**Derived downstream inside SQLite:** `segment_contact_group` is a generated column computed from `segment_outcome`, so it carries the same error onward.

**Row ordering:** the flat CSV is ordered by subject, date, video, `segment_num`, `reach_num` (`database.py:783`), so the dump's row order shifts too.

Three further things this layer does not do:

- **No validation of the grouping.** Nothing checks that `segment_num` falls in 1..20, that segment numbers are contiguous, or that there are twenty of them.
- **Pellets with no reaches are absent, not zero.** A segment with an empty reach list contributes no row at all. You cannot count "pellets the mouse never reached for" from `reach_data`; those pellets are simply missing.
- **Re-segmentation leaves no trace.** Because the replace is keyed on `video_name` alone, re-syncing a re-segmented video silently rewrites every row's pellet number with no record of what it was before.

---

## Where each column's value comes from

The insert list is `ALL_COLUMNS` (`database.py:84-92`), 62 names (`segment_num` included).

**Session identity (5 columns)**

| Column | Source |
|---|---|
| `subject_id` | derived from the video name, `CNT0115` → `CNT_01_15` (`database.py:229`) |
| `video_name` | the `video_name` key inside the features file, else the file name |
| `session_date` | first eight digits of the name, reformatted `YYYY-MM-DD`; **empty string, not null, if absent** (`database.py:582`) |
| `tray_type` | letters matched by `CNT\d{4}_([A-Za-z]+)(\d+)$`, upper-cased (`database.py:285`); null if the name does not end that way |
| `run_number` | digits from the same pattern; null if absent |

**Which pellet (1 column)**

`segment_num` is read from the enclosing segment, never from the reach (`database.py:568`). This is deliberate: a reach's own copy was always `0`. It is in the insert list (`ALL_COLUMNS`).

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
| `dlc_scorer` | `dlc_model.dlc_scorer` | the pose-estimation model suffix, e.g. `DLC_resnet101_<project><date>shuffle1_100000`, taken from the pose file's own name (`manifest.py:56-60`) |
| `segmenter_version` | `pipeline_versions.segmenter` | read out of `{video}_segments.json` when the manifest was written |
| `reach_detector_version` | `pipeline_versions.reach_detector` | read out of `{video}_reaches.json` |
| `outcome_detector_version` | `pipeline_versions.outcome_detector` | read out of `{video}_pellet_outcomes.json` |

Each version is whatever that stage stamped into its own output file (`manifest.py:152-158`, `manifest.py:233-236`), so these columns record what each stage *claimed*, which is only as good as the stage's stamp.

That distinction is not academic. The segmenter can stamp a constant belonging to a different module than the one that actually cut the boundaries, so `segmenter_version` records the writing module's number rather than the running segmenter's.

**If the manifest is missing or unreadable, all six columns are `NULL` and nothing is logged** (`database.py:437-438`, `database.py:463-464`). A null here is indistinguishable from a manifest that genuinely recorded nothing.

Things the manifest holds that are **not** copied into the database: the reach-assignment stage's version, the kinematics extractor version under `pipeline_versions` (the `extractor_version` column carries the same number, but read from the features file instead), the per-stage validation statuses, and the per-step timestamps.

---

## Columns that are declared and never filled

Sync copies each key verbatim, so a key that is null or missing in the features file is `NULL` in the table; whether a column is filled depends only on what the kinematics extractor that wrote the features file put there. The per-reach `outcome` and `interaction_frame` are mostly empty by nature: they are populated only for the reach the outcome stage tied to the pellet. `interaction_frame` is filled on the causal reach only when the frame lies inside that reach (the contact frame is an artifact), so a reach a person picked over the algorithm's contact frame has it null -- a column derived from `interaction_frame` (such as `contact_group`) must not be read as "this reach missed"; read the reach's outcome instead.

`docs/FIELD_AUDIT.md` (generated by `python -m mousereach.pipeline.field_audit --markdown`) is the right tool for the file-by-file view, but do not use it as the authority on the table: it reads only the videos judged "finished and current at every stage" and takes the database side from a parquet snapshot, so its database figures can differ widely from the table's. Query the table when you want to know what is in the table.

**Present in the live table, never written by this code.** `test_phase` and `phase_group` were added to the database by something outside MouseReach. Sync neither writes nor clears them, so a re-synced video loses them (see above).

**Computed by the database itself.** `contact_group` and `segment_contact_group` are generated columns in the live table, derived from `interaction_frame` and `segment_outcome`. The string `contact_group` does not appear anywhere in `database.py`. This is a general caution: the `CREATE TABLE` in the source runs only `IF NOT EXISTS`, so the live table has drifted ahead of it — different column types, extra columns, a differently-named unique constraint. Read the schema from the database, not from the source.

**Uniqueness.** The live table carries `UNIQUE (video_name, reach_id)`; the source writes it as `UNIQUE(video_name, reach_id)` (`database.py:203`). Reach identifiers are unique per video across all segments — checked against sampled features files, no duplicates — so this constraint does not currently bite.

---

## The flat CSV dump

Path: `<central_db's folder>/database_dump/reach_data.csv` -- beside the configured central database; not written when no central database is configured.

`CSV_DUMP_PATH` is derived from the module-level `DB_PATH` (`database.py:53`), *not* from the syncer's `db_path`. Point a `DatabaseSyncer` at a different database and it will still overwrite this one shared CSV.

**When it is written:**

- At the end of `sync_all`, only if at least one file counted as synced (`database.py:687-688`).
- After every single-video call through `sync_file_to_database` (`database.py:854`) — so once per video, every video.
- Never by the folder watcher.

It is a full dump of the whole table each time, ordered by subject, date, video, pellet and reach (`database.py:783`). The file grows with the table (hundreds of MB for a large one), so the per-video path rewrites the whole file on network storage for every video processed.

**What it contains:** 57 columns, listed explicitly in the query at `database.py:760-781`. Seven ordinary columns of the table are **not** exported: `outcome_source`, `reviewed_by`, `algo_outcome`, `algo_causal_reach_id`, `extended_features`, `test_phase` and `phase_group`. Anyone working from the CSV cannot see whether a reach's outcome came from the algorithm or from a human reviewer, and cannot see any per-paw measurement. The row `id` and the two generated columns are also absent.

**Header alignment.** The header comes from the query's own column names (`database.py:792`). A separately maintained list can drift from the `SELECT`: the header was written from `ALL_COLUMNS` while the rows came from the `SELECT`, so 61 names sat over 57-value rows and every column from the sixth onward was mislabelled. **A dump written before the fix is that broken one**: the header has 61 names, every row has 57 values, and the column headed `causal_reach` holds outcome strings such as `displaced_sa`. Treat such a `reach_data.csv` as unusable for anything keyed on column name.

The next successful sync rewrites it with the correct header.

**Failures.** The export catches everything and logs a warning (`database.py:804-809`). Catching everything and saying nothing would make an export that failed every time look exactly like one that succeeded.

---

## Configuration

| Setting | Where | Effect |
|---|---|---|
| `processing_root` | `~/.mousereach/config.json`, or the environment variable `MouseReach_PROCESSING_ROOT` (`config.py:56`) | the folder scanned and watched is `<processing_root>/Processing` (`config.py:152`). If unset, the syncer's `processing_path` is `None`, the batch scan finds nothing, and the sync-state file is neither read nor written — but the per-video path still works, because it is handed an explicit file. |
| database location | `central_db` in `~/.mousereach/config.json`, or `MOUSEREACH_CENTRAL_DB`, or the constructor argument `db_path` | no default. Unset = `check_database()` returns (False, "no central database configured ...") and nothing is written: `mousereach-sync` and `mousereach-sync-watch` print "Error: no central database configured ..." and exit with code 1; `mousereach-sync-status` prints "Database Status: ERROR" with the same reason and exits with code 0. |
| CSV location | none | derived from the configured central database's folder; not affected by a constructor-passed `db_path`. |
| `dry_run` | constructor argument, `mousereach-sync --dry-run` | builds and counts rows, inserts nothing, does not save sync state (`database.py:355-356`). Can still create the table and add columns. |
| `force` | `mousereach-sync --force` | ignores the file-hash check and re-syncs every eligible file. |
| `debounce_seconds` | `mousereach-sync-watch --debounce` | **no effect** — the command crashes before reaching the watcher. `PipelineWatcher` accepts the value if driven from Python, but `start_watcher` never passes it on. |

Writing is off whenever `central_db` / `MOUSEREACH_CENTRAL_DB` is unset, which is the default. Apart from that, `review/staging.py` replaces `sync_file_to_database` with a no-op while a video is staged for review.

---

## Failures that are swallowed

Collected in one place, because these are the reasons a write can produce nothing while everything looks normal:

| `database.py` | What is swallowed |
|---|---|
| `349-350` | unreadable sync-state file — treated as "nothing has ever been synced" |
| `361-362` | sync-state file that cannot be written — the next run re-syncs everything |
| `410-411`, `413-414` | every schema migration failure |
| `463-464` | any failure reading the processing manifest — all six provenance columns become null |
| `741-745` | failure counting rows for the status report — becomes zeros |
| `804-809` | CSV export failure (logged, not raised) |
| `857-858` | **every failure in the per-video write path** (logged as a warning naming the file, then False is returned) |

`watcher.py:89-90` and `watcher.py:113-114` log sync failures at error level, which is the one path that says anything useful.

`watchdog` is an optional dependency, declared under the `sync` extra in `pyproject.toml`; `sqlalchemy` is a core dependency (in the `sync` extra only, a plain install could not reach the watcher's coordination database). Without `sqlalchemy` the engine raises `ImportError` (`database.py:368-369`) and the per-video path swallows it.

---

## Dead code

`SYNC_PATTERNS` at `database.py:208` is dead code — defined, referenced nowhere in `src/`.

---

## a sync that does not happen is loud

`sync_file_to_database` returning False -- the video's results did NOT reach
the results database (central_db) -- was logged at debug level and treated as a normal skip; the
audit table showed far fewer successes than completed extractions before
anyone noticed. Both watcher paths now write a `db_sync` `failed` row to the
step audit table and log a warning naming the video. The GPU node's local
path, which never even checked the return value, checks it now. Feature
extraction failures on that path also write a `failed` audit row.

## filed is final

A filed video's results in Analyzed are not re-made by newer algorithm versions any more, so what a
database integrator pulled for a filed video stays the answer, except for kinematics (which may be
recalculated, without changing the filed decisions). When a person reopens a filed video, its filed
files stay in Analyzed until the new run is filed and replaces them; an integrator that leaves out a
filed video whose newer run waits in a review queue keeps doing so. See docs/PIPELINE_AS_BUILT.md,
"filed is final".

## session flags

Every features file now carries `session_flag`: the note a person made with the review window's
Flag Session button on the video's mouse+day session (`flagged_by`, `flagged_at`, `reason`,
`session_key`), or null. Export it as a marker column; do not leave a flagged session out or treat
it differently because of it (a flag is only recorded so it can be found).
Filed videos receive it through a kinematics-only recalculation when their session is flagged later.
See docs/PIPELINE_AS_BUILT.md, "session flags are only recorded".

## a reach before the first cut is synced under no pellet

A reach that starts before the first cut belongs to no pellet. An older features file lists such a reach under segment 1; `sync_features_file` reads the file
through `reach/core/before_first_cut.split_before_first_cut`, so the reach is not written as one of
pellet 1's rows (and the rest of pellet 1's reaches are renumbered 1..n). Newer features files list it only at the top level (`reaches_before_first_cut`), which this sync does not write as a
reach row. The features file on disk is never changed.
