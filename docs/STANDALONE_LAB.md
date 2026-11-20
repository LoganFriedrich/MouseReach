# MouseReach on one PC: from videos to a spreadsheet

This is the whole walkthrough for a lab that runs MouseReach on **one computer**
and does everything by hand: put videos in a folder, press Run, review what the
program could not decide, press Run again, export the results. You do not need
the automatic watcher, hand-score spreadsheets, or any database. Nothing here
requires reading code.

What the computer needs:

- Windows, macOS or Linux with Python (installed below with conda).
- **An NVIDIA graphics card** with its driver, CUDA and cuDNN. MouseReach finds
  the hand, pellet and box in every frame with DeepLabCut, which needs the
  graphics card. Without one, a machine cannot pose videos (posing on the
  processor was removed). `mousereach-watch --dry-run` and the Run button say so
  if no graphics card is found.
- `ffmpeg` on the PATH (cuts multi-animal recordings).
- **A trained DeepLabCut model** whose body parts are MouseReach's: `Reference`,
  `SATL`, `SABL`, `SABR`, `SATR`, `BOXL`, `BOXR`, `Pellet`, `Pillar`, `RightHand`,
  `RHLeft`, `RHOut`, `RHRight`, `Nose`, `RightEar`, `LeftEar`, `LeftFoot`,
  `TailBase`. You need the path of its `config.yaml`. MouseReach does not ship a
  model; use your lab's or ask whoever trained it.
- Disk room for your videos and results.

---

## 1. Install

Open a terminal (on Windows: "Anaconda Prompt").

```
conda create -n mousereach python=3.10
conda activate mousereach
cd <the folder you downloaded MouseReach into>
pip install -e .
```

Every command below is run with the `mousereach` environment active
(`conda activate mousereach` first, each time you open a new terminal).

## 2. Set up this machine as the only MouseReach machine

```
mousereach-setup
```

1. **"Is this the ONLY machine that will run MouseReach in your lab?"** -- press
   Enter (yes). This makes the PC a *single machine*: it cuts, poses, analyses,
   takes reviewed videos back and files the results, all by itself.
2. **Data repository** -- optional; press Enter to leave it empty.
3. **Pipeline folder** -- a folder with room for your videos and results, for
   example `D:\MouseReach_Data`. It is created for you, with its subfolders.
4. **Processing root** -- a *different* folder on a fast local disk, for the
   working files, for example `D:\MouseReach_Work`. (Setup refuses the same
   folder twice: both hold a folder named `Processing`.)
5. **DLC config.yaml path** -- the path of your DeepLabCut model's `config.yaml`.
6. The remaining questions (graphics card number, poll interval, stability
   wait): press Enter for the defaults.

Setup then builds the pipeline folder:

```
D:\MouseReach_Data\
  Unanalyzed\Multi-Animal\      <- put multi-animal recordings here
  Unanalyzed\Single_Animal\     <- or single-animal videos here
  Processing\Posed\             (working: posed videos)
  Processing\Review\triage\     (videos waiting for you: a question per pellet)
  Processing\Review\deep_review\ (videos waiting for you: the pellet cuts)
  Processing\Quarantine\        (files whose names could not be read)
  Processing\Failed\
  Analyzed\                     <- the finished results, by project and cohort
  pipeline_versions.json        (which versions of the tools are current)
```

The same settings can be made in the window: `mousereach`, **Setup** tab, "What
this machine does" = **Single machine**, the two folders and the model, **Save
configuration**, then restart `mousereach`.

## 3. Name your videos (this is required)

MouseReach reads who and what is in a video from its name. A file whose name
does not follow the rule is moved to `Processing\Quarantine` and not analysed
(the **Quarantine** tab shows why and lets you fix the name and release it).

**Animal names** are letters, then a 2-digit cohort, then a 2-digit subject:
`ABC0101` = project ABC, cohort 01, subject 01. Cohort `00` means an empty
position.

**A multi-animal recording** (one video of 8 positions, cut by MouseReach):

```
YYYYMMDD_<8 animal names, comma-separated>_<tray><run>.mp4
20250704_ABC0101,ABC0102,ABC0103,ABC0104,ABC0105,ABC0106,ABC0107,ABC0108_P1.mp4
```

**A single-animal video** (already one animal per video):

```
YYYYMMDD_<animal>_<tray><run>.mp4
20250704_ABC0101_P1.mp4
```

- `YYYYMMDD` is the recording date (a date in the future is refused).
- `<tray>` is `P` (pillar: pellets are scored), `E` (easy) or `F` (flat); easy
  and flat trays give reaches but no pellet outcomes.
- `<run>` is the tray's number that day (`P1`, `P2`, ...).

Wrong: `ABC0101_20250704_P1.mp4` (order), `2025-07-04_ABC0101_P1.mp4` (date),
`20250704_ABC101_P1.mp4` (subject needs 2 digits), `20250704_ABC0101_tray1.mp4`
(tray must be P, E or F plus a number), a recording with 7 names.

## 4. Put the videos in

Copy recordings into `<pipeline folder>\Unanalyzed\Multi-Animal`, or
single-animal videos into `<pipeline folder>\Unanalyzed\Single_Animal`. A file
is taken only once it has stopped changing for the stability wait (60 seconds
by default), so a copy still in progress is never picked up half-finished.

## 5. Run

**In the window**: start `mousereach`, open the **2 - Run Pipeline** tab and
press **Run the pipeline now**. In a terminal: `mousereach-watch --once`.

One press does everything waiting, in order: cut each recording into
single-animal videos, pose each one on the graphics card, then for each video
find the pellet boundaries, the reaches, each pellet's outcome and the reach
that caused it, check the quality, and either **file** it (its results go into
`Analyzed\<project>\<cohort>\`) or **hold it for you** in a review queue when
something could not be decided. It stops when nothing is left. It can take a
long time for many videos; the tab shows what is happening.

To run only some videos: **Run chosen videos...** on the same tab (pick the
files), or `mousereach-run-videos <file> <file> ...`. A video that has no pose
yet is posed first; a recording is cut first. A video that is already filed is
not run again (filed results are final), and one held for review waits for its
review; the tab says so for each.

The automatic watcher (`mousereach-watch`, or **Start** on the Watcher Control
tab) does the same continuously, picking up new videos by itself. You do not
need it.

## 6. Review what MouseReach could not decide

Open the **Review Queues** tab. It shows how many videos wait in each queue.

- **Triage** (Open Triage Review): one question per pellet the program was
  unsure of -- which reach got the pellet, what happened to it. Answer each and
  save.
- **Re-segmentation** (Open Re-segmentation / Re-segmentation Review): the pellet
  boundaries could not be found; you place or accept each boundary, then
  **Clear -> re-enter pipeline**.

Your answers are final: the program uses them and does not ask again.

**Then press Run the pipeline now again.** A reviewed video goes back into the
pipeline at the next pass: on a machine without the automatic watcher, that
pass is your press (the release message says so). It is then finished and
filed, or held again if something new needs you.

## 7. Export the results

**In the window**: the **Export results** tab. Pick the level, leave the folder
at `Analyzed`, pick Excel or CSV, press **Export...**. In a terminal:

```
mousereach-results --level session
mousereach-results --level reach --format xlsx --out D:\Results\reaches.xlsx
```

| Level | One row is |
|---|---|
| `reach` | one reach, with its kinematics |
| `pellet` | one pellet (one segment of a tray) |
| `session` | one tray = one video |
| `day` | one animal on one date, all its trays |
| `animal` | one animal, everything filed for it |

Only **filed** videos are exported. Every row repeats the totals of the levels
above it; each column name starts with its level (`reach_`, `segment_`,
`session_`, `day_`, `animal_`).

## 8. What the main columns mean

- `animal_id`, `date`, `session_id` (the video), `tray_type`, `tray_run`,
  `segment_number` (the pellet's number in the tray), `reach_number` -- from the
  video name and the analysis.
- `segment_outcome` -- what happened to the pellet: `retrieved`, `displaced_sa`
  (knocked off, stayed in the scoring area), `displaced_outside`, `untouched`,
  `abnormal` (a person saw something other than a reach move it), `no_pellet`.
- `reach_outcome` -- what THIS reach did: the pellet's outcome for the one reach
  credited with it (`reach_is_causal` = True), `missed` for every other reach,
  `undetermined` when a pellet moved and no reach was credited.
- `<level>_pellets_presented` -- pellets presented (every one except
  `no_pellet`; **a pellet nobody reached for still counts**).
- `<level>_success_rate` = retrieved / pellets presented (0 to 1);
  `<level>_contact_rate` = (retrieved + displaced) / pellets presented.
- `<level>_n_reaches`, `<level>_mean_reach_max_extent_mm` and the other
  `mean_reach_...` columns -- reach counts and averages.

Every column, and exactly how each number is counted: `docs/RESULTS_EXPORT.md`.
An Excel export also has a second sheet, `columns`, with each column's meaning.
Per-reach kinematics: `REACH_KINEMATIC_DATA_DICTIONARY.md`.

---

## When something goes wrong

| You see | It means / do this |
|---|---|
| A video went to `Processing\Quarantine` | Its name does not follow the rule (step 3). Fix it in the **Quarantine** tab. |
| "No GPU available for DLC inference" | The graphics card, driver, CUDA or cuDNN is missing. Posing needs it. |
| "DLC config not found" | The model path in setup is wrong: run `mousereach-setup` again. |
| "nas_root ... has no pipeline_versions.json" | The pipeline folder was not built: run `mousereach-setup` again and answer yes to "only machine". |
| A reviewed video did not come back | Press **Run the pipeline now** (or `mousereach-watch --once`): without the automatic watcher, nothing takes it back until you do. |
| The export says "Nothing exported" | No video is filed yet: check the Review Queues tab for videos waiting for you. |
| `mousereach-watch-status` | Counts of videos by state, failures with their reasons. |
