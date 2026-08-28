# Tracking Sheets: getting the lab's spreadsheets into the database

This page is for the person doing the work. No codebase knowledge assumed.

## What this is about

Each cohort has one tracking workbook, `Connectome_NN_Animal_Tracking.xlsx`
(NN = cohort number), kept in the lab's SharePoint folder. It is the
hand-kept record of that cohort: the animals, weights, ramp days, manual
tray scores (which pellet was missed / displaced / retrieved), the injury,
the injection surgery, the virus batch. The video pipeline never edits it.

The database (`connectome.db`) only knows what has been **imported** from
those workbooks. Importing reads a workbook and copies its contents into
the database; it never changes the workbook. You can import as often as you
like -- rows already in the database are skipped, new or changed rows land.

Animals do NOT have to be in the sheet before their videos are processed.
The video pipeline creates an animal the first time it sees a video for it;
the sheet import fills in the details later. (This was not true before
2026-08-28; it is now, and it is why a late sheet can no longer make video
data disappear.)

## Where to do it

Open MouseReach (Anaconda Prompt: `conda activate C:\LAB_ROOT\envs\mousereach`
then `mousereach`) and find the **Tracking Sheets** tab in the right-hand
dock. Everything below is a button on that tab. (`mousereach-sheets` opens
the tab on its own.)

## What the table tells you

One row per cohort:

| Column | Meaning |
|---|---|
| Cohort | e.g. CNT_05 |
| Status | **Up to date** -- the database matches the sheet's last edit. **Sheet edited since last import** -- someone changed the workbook after it was last imported; import again. **Never imported** -- import it. **LAST IMPORT FAILED** -- the last attempt hit an error; select the row to read it. **N files match -- choose** -- see below. |
| Sheet file | The workbook being read as this cohort's sheet. `[pinned]` means a person chose it. |
| Sheet edited | When that workbook was last saved. |
| Last import | When it was last imported into the database (and the message box shows whether that worked). |

Select a row and the line under the table explains the status in a sentence.

## Importing

1. Press **Import all sheets**. (Or select rows and press **Import selected**.)
2. Watch the box at the bottom. Each cohort reports `OK` with what was
   imported, or `FAIL` with the reason.
3. The table refreshes itself. Everything should read **Up to date**.

An import takes about a minute per cohort (the workbooks are large and live
on a network share).

**The background job.** Every hour the processing server imports all
sheets on its own and takes a fresh copy of the database for analysis. This
tab shows you whether that worked -- a cohort stuck on *Never imported* or
*LAST IMPORT FAILED* means the background job is failing for it too, and
the reason is the one shown.

## When several files match a cohort

The folder sometimes holds more than one workbook for the same cohort -- a
`(2)` copy, a dated backup, a draft with a `1` on the end. The system does
not guess silently: the status says **N files match -- choose**, and until
you choose it uses the most recently edited one and says so.

To choose: select the row, press **This is the sheet**, pick the file from
the list. The choice is remembered on this machine and shown as `[pinned]`.
(To change it, pick again. The pin lives in `~/.mousedb/config.json`.)

## Setting the folder (once per machine, and if it ever moves)

Press **Set sheets folder...** and pick the folder that contains the
`Connectome_NN_Animal_Tracking.xlsx` files. The tab refuses a folder with no
such file in it. **Open folder** opens the current one in Explorer.

If the SharePoint folder is ever moved or renamed, every machine that imports
needs this done again -- the tab will show *no tracking-sheet folder is
configured* until it is.

The folder path is deliberately never written into the code (it contains a
username and the organisation's folder names, and the code is public); it
lives only in that local config file.

## Starting a new cohort

Press **New cohort sheet...**, give the cohort name (e.g. `CNT_06`) and the
food-deprivation start date. A correctly formatted, empty workbook is
written into the sheets folder with every tab the import expects. Fill it
in as the cohort runs; import whenever you like.

## If something goes wrong

- **LAST IMPORT FAILED** with a database message (e.g. *NOT NULL constraint
  failed*, *no such column*): a mismatch between the workbook and the
  database rules. Nothing was written (an import is all-or-nothing per
  cohort). Copy the message and report it.
- **A sheet is open in Excel on another machine**: import still works (it
  reads a shared copy); if it ever refuses with *Permission denied*, retry
  in a minute.
- **The tab says the mousedb environment was not found**: this machine
  cannot import; the processing server can. Importing from any one machine
  updates the shared database for everyone.

## Where the import history lives

Every import attempt -- from this tab or the hourly job, success or failure
with the error text -- is appended to
`Y:\LAB_ROOT\Databases\logs\sheet_imports.jsonl`. The Status column is
computed from that file plus the workbook's last-edit time.

From a terminal (`conda activate C:\LAB_ROOT\envs\MouseDB`):

```
mousedb-sheets status
mousedb-sheets import [--cohort CNT_05] [--dry-run]
mousedb-sheets pin CNT_00 "Connectome_00_Animal_Tracking.xlsx"
mousedb-sheets set-dir "<folder>"
```
