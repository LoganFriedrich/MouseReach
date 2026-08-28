# Where is my data?

This page is for the person doing the work. No codebase knowledge assumed.

## The short answer

`Y:\LAB_ROOT\Databases\exports\current\` — a folder that is rewritten
every hour with everything the database currently holds, as CSV files you
can open in Excel, R or Python, each with a data dictionary beside it:

| file | one row per | definitions |
|---|---|---|
| `reach_data.csv` | reach the pipeline detected (kinematics, the pellet outcome of its segment, and where that outcome came from) | `reach_data_DATA_DICTIONARY.csv` |
| `manual_scores.csv` | pellet scored by hand from the tray (0 missed / 1 displaced / 2 retrieved) with the session's phase | `manual_scores_DATA_DICTIONARY.csv` |
| `ODC_sessions_<cohort>.csv` | animal per session, in the ODC-SCI `2_ODC_Animal_Tracking` shape (per-tray and daily counts and percentages, weight, injury) | `ODC_sessions_DATA_DICTIONARY.csv` |
| `MANIFEST.json` | — | when the files were written, from which snapshot, row counts, and any problems |
| `README.txt` | — | the same explanation as this table |

An ODC-SCI submission is a dataset file **plus** its data dictionary; both
are here. `MANIFEST.json` says `"complete": true` when every column in every
file has a dictionary entry — if it says false, the problems list names the
undocumented columns, and an upload would be rejected until they are added.

## The longer answer: the "Where Is My Data" tab

Open MouseReach (Anaconda Prompt: `conda activate C:\LAB_ROOT\envs\mousereach`
then `mousereach`; or `mousereach-data-status` for the tab alone) and find
**Where Is My Data** in the right-hand dock. One row per cohort:

| column | meaning |
|---|---|
| Animals | animals the database knows for the cohort. `[N from video only]` means N of them were created by the pipeline from a video before the tracking sheet named them — import the sheet to fill in their details |
| Sheet | the tracking sheet's import status (worked in the **Tracking Sheets** tab): Up to date / Sheet edited since last import / Never imported / LAST IMPORT FAILED |
| Sessions scored | hand-scored animal-days in the database |
| Videos in DB | videos whose reaches are in the database (i.e. are in `reach_data.csv`) |
| In review (triage / deep) | videos waiting for a person. **Their data is not final until they are reviewed and released.** Press *Open Review Queues* to work them |
| Outcomes algo / human | pellet outcomes resting on the algorithm alone vs confirmed or corrected by a person |
| Reaches | rows the cohort contributes to `reach_data.csv` |

Below the table: the export folder, when it was last written, whether it is
complete for an ODC upload, the row count of each file, and any problems.

Buttons: **Refresh** re-reads everything. **Open exports folder** opens the
folder above in Explorer. **Refresh exports now** rewrites `reach_data.csv`
and `manual_scores.csv` immediately from the latest snapshot (the per-cohort
ODC session files refresh on the hourly run, which is the only time the
database may be read safely). **Open Review Queues** opens the queues in
their own window.

## Why the numbers can lag by up to an hour

The tab and the exports read an hourly *snapshot* of the database, not the
live database. The live file sits on a network share, and reading it while
the pipeline writes to it can corrupt the read — so everything human-facing
works from the last safe copy. The tab shows the snapshot's time at the top.

## When something looks wrong

- **A cohort's videos are in the pipeline but "Videos in DB" is low**: check
  *In review* first — held videos are not in the database yet. Then check
  the Sheet column: a failed or never-run import used to block video data
  entirely (it no longer does, as of 2026-08-28, but an old failure is worth
  clearing).
- **"complete for ODC upload: False"**: a new column reached the exports
  without a dictionary entry. The problems list names it; the definition is
  added in `mousedb/exporters/data_dictionary.py`.
- **Nothing in the exports folder**: the hourly refresh has not run yet on
  this database; press *Refresh exports now*.

From a terminal (`conda activate C:\LAB_ROOT\envs\MouseDB`):

```
mousedb-data-status              # the table above, as text
mousedb-current-exports          # rewrite reach_data + manual_scores (+ dictionaries)
mousedb-current-exports --db-ok  # also the ODC session files (only when no watcher is writing)
```
