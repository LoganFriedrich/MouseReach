# Exporting results: reach, pellet, session, day or animal

Describes: `src/mousereach/export/results.py`, `src/mousereach/export/results_widget.py`

This is how you get the analysis out of MouseReach as a spreadsheet. It needs
nothing but MouseReach: no automatic watcher, no hand-score sheet, no database.

## What gets exported

Every video that has been **filed** -- passed every check and moved into
`<pipeline folder>/Analyzed/<project>/<cohort>/`. For each one the export reads
its `<video>_features.json`: the video's final result, with every person's
review answer and ground truth applied. Superseded results in
`Analyzed/Archive` are skipped. A video still in a review queue or still being
processed is not filed yet and is not in the export.

## How to run it

**In the window** (`mousereach`): the **Export results** tab. Pick the level,
check the folder (it starts at your pipeline folder's `Analyzed`), pick Excel or
CSV, press **Export...** and choose where to save. The tab says how many rows it
wrote and lists any video it could not read.

**In a terminal** (inside the mousereach environment):

```
mousereach-results --level session
mousereach-results --level reach --format xlsx --out C:\MyResults\reaches.xlsx
mousereach-results --level animal --root D:\MouseReach_Data\Analyzed\MyProject
```

| Option | Meaning |
|---|---|
| `--level` | `reach`, `pellet` (same as `segment`), `session` (same as `tray`/`video`), `day`, `animal`. Default `session`. |
| `--root` | The folder of filed results to read. Default: `Analyzed` in your pipeline folder. Point it at one project or cohort folder to export only that. |
| `--out` | The file to write. Default: `<pipeline folder>\Exports\MouseReach_results_<level>_<date>_<time>.<format>`. It may not be inside the folder it reads. |
| `--format` | `csv` (default) or `xlsx`. The xlsx file has a second sheet, `columns`, saying what every column means. |

It prints `[OK] <n> <level> row(s) from <m> filed video(s) -> <file>`, or
`Nothing exported` when there are no filed results there.

## The levels -- what one row is

| Level | One row is | Carries |
|---|---|---|
| reach | one reach | its own results (`reach_...`), its pellet's (`segment_...`), its tray's (`session_...`), its animal-day's (`day_...`) and its animal's (`animal_...`) |
| pellet | one pellet = one segment of a tray | `segment_...`, plus `session_...`, `day_...`, `animal_...` |
| session | one tray = one video | `session_...`, plus `day_...`, `animal_...` |
| day | one animal on one date, all its trays | `day_...`, plus `animal_...` |
| animal | one animal, everything filed for it | `animal_...` |

Every column name starts with the level it describes, so a value repeated from
a higher level (say the tray's success rate on every reach row) is never
mistaken for the row's own.

## How the numbers are counted

Each pellet's outcome is its segment's outcome: `retrieved`, `displaced_sa`
(knocked off, stayed in the scoring area), `displaced_outside`, `untouched`
(`missed` in older files), `abnormal` / `abnormal_exception` (a person saw
something other than a reach move the pellet), or `no_pellet` (nothing was
presented in that segment).

- **pellets presented** = every pellet except `no_pellet`. **A pellet nobody
  reached for still counts.**
- **success rate** = retrieved / pellets presented (a number from 0 to 1).
- **contact rate** = (retrieved + displaced_sa + displaced_outside) / pellets presented.
- A pellet moved by something other than a reach counts as presented and in its
  own column (`moved_not_by_reach`), never as retrieved or displaced. To leave
  such pellets out of a rate, subtract that column from `pellets_presented`.
- Session, day and animal numbers are counted from **pellets**, never from
  reach rows.
- **Easy and Flat trays** (`E`, `F` in the name) have no pellet outcomes (their
  videos are not cut into pellets). They add their reaches to the reach counts
  of the session, day and animal, and nothing to the pellet counts; their
  session rows leave the pellet columns empty.

**A reach's own outcome (`reach_outcome`) is not its pellet's.** Only the one
reach credited with the pellet (`reach_is_causal` = True: flagged as the causal reach, or named by its segment's `segment_causal_reach_id`) carries the pellet's
outcome. Every other reach at that pellet is `missed`, and so is every reach at
an untouched pellet or at a pellet moved by something other than a reach. If a pellet moved and no reach was credited, its reaches
say `undetermined` -- never a guess. `segment_outcome` is the pellet's result
and is the same on every reach row of that pellet.

## Columns

Identity columns (plain names):

| Column | Meaning |
|---|---|
| `animal_id` | The animal, as in the video name `YYYYMMDD_<animal>_<tray><run>` |
| `lab_animal_id`, `lab_session` | Only for a project that re-encodes animal names on the way in: the lab's own name for the animal and the video |
| `date` | Recording date (YYYY-MM-DD), from the video name |
| `session_id` | The session = tray = video (the video name) |
| `tray_type` | `P` pillar (pellets scored); `E` easy, `F` flat (no pellet outcomes) |
| `tray_run` | The tray's number that day |
| `segment_number` | The pellet's number within the tray |
| `reach_number`, `reach_id` | The reach's number in its segment, and its id in the video |
| `source_file` | (session level) the features file the row came from |

Per reach (`reach_...`): `reach_outcome` and `reach_is_causal` (above), then every
per-reach value of the features file -- frames (`reach_start_frame`,
`reach_apex_frame`, `reach_end_frame`, `reach_duration_frames`), extent
(`reach_max_extent_mm`), velocities, angles, tracking quality, and the extended
kinematics. Their meanings: `REACH_KINEMATIC_DATA_DICTIONARY.md`. Frames are
video frames; divide by the video's frame rate for seconds.

Per pellet (`segment_...`): `segment_outcome`, `segment_n_reaches`,
`segment_n_reaches_left_out` (reaches a person took out of the measurements),
`segment_causal_reach_id`, `segment_start_frame`, `segment_end_frame`, and the
per-pellet values of the features file (attention, time to first reach, pellet
position, tracking quality).

Per tray, day and animal (`session_...`, `day_...`, `animal_...`), the same set:

| Name after the prefix | Meaning |
|---|---|
| `pellets_presented` | pellets presented (see above) |
| `retrieved`, `displaced_sa`, `displaced_outside`, `displaced`, `untouched` | pellets with that outcome (`displaced` = both kinds) |
| `moved_not_by_reach` | pellets a person saw moved by something other than a reach (`abnormal`) |
| `undetermined` | pellets with no decided outcome (a filed video has none) |
| `no_pellet` | segments with no pellet presented |
| `success_rate`, `contact_rate` | see above (empty when no pellet was presented) |
| `n_reaches` | reaches measured (all trays) |
| `n_reaches_left_out` | reaches a person took out of the measurements |
| `n_causal_reaches` | reaches credited with a pellet's outcome |
| `reaches_per_pellet` | n_reaches / pellets_presented |
| `mean_reach_max_extent_mm`, `mean_reach_duration_frames`, `mean_reach_peak_velocity_px_per_frame`, `mean_reach_velocity_at_apex_mm_per_sec`, `mean_reach_trajectory_straightness` | the mean over the measured reaches |
| `n_sessions` | (day, animal) trays |
| `n_days` | (animal) recording days |

## Older export commands

- `mousereach-features-csv` runs `mousereach-results --level reach`; the
  `*_grasp_features.json` files it was named for are not written by the pipeline.
- `mousereach-quick-summary` writes the session level of the filed results and
  prints the overall totals.
- `mousereach-build-database`'s Sessions and Mice sheets are this export's
  session and animal levels, so their rates count pellets, not reach rows.
- `mousereach-export` and `mousereach-summary` still read one flat folder you
  name with `-i`; for filed results use `mousereach-results`.
