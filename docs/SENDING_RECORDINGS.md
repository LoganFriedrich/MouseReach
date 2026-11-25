# Sending recordings from a recording PC to the pipeline

This page is for the people who record and the person who looks after the
recording PCs. You do not need to read any code to use it.

## In one paragraph

After a recording session, a person renames each recording to its session name
(date, the eight animals, the tray), as before. **That rename is the only thing a
person has to do.** Outside recording hours, the MouseReach watcher on the
recording PC copies every renamed recording to the pipeline's intake folder on
the shared drive. It checks that the copy is complete and identical, and only
then deletes the recording from the recording PC. A recording that has not been
renamed is never touched. Anything the watcher cannot send is shown in red, with
the reason.

## What moves, and what never does

| The file in the recording folder | What happens |
|---|---|
| Still has the recording program's own name (`2024-01-02 09-30-00.mkv`) | **Never touched.** Listed as "not named yet". It waits for a person to rename it. |
| Renamed to a valid session name (`YYYYMMDD_<8 animal ids>_<tray>.mkv`) | Sent outside recording hours, then deleted here once the copy is verified. |
| Renamed, but the name is not a valid session name (seven animals, a missing tray, a date that does not exist ...) | **Held, in red**, with the reason. Correct the name in the folder; it goes on the next evening by itself. |
| Still being recorded, or changed in the last 5 minutes, or open in any program | Waits. Looked at again on the next pass. |
| A **failed recording**: empty, a fraction of a second long, unreadable, or broken at its start or end (the recording check, see [RECORDING_ISSUES.md](RECORDING_ISSUES.md)) | **Sent anyway**, and listed as a recording issue (not red). The pipeline sets it aside on the shared drive, never cuts it, and the database records that the session's recording failed. A failed recording still documents what happened that day, so it is not left on this PC. |
| The recording check cannot run on this PC (no ffmpeg) | **Held, in red**, until ffmpeg is installed. |
| The same name is already on the shared drive with the **same** content | It was sent before. Only the copy on this PC is deleted. |
| The same name is already on the shared drive with **different** content | **Held, in red.** Nothing is overwritten and nothing is deleted. A person decides which is right. |

Only files directly in the recording folder count. Subfolders are never looked into.

## The session name, exactly

A session name has three parts joined by underscores: the date, the eight
animals, and the tray. Example (placeholder animal ids):

    20250704_ABC0101,ABC0102,ABC0103,ABC0104,ABC0105,ABC0106,ABC0107,ABC0108_P1.mkv

* **Date**: eight digits, `YYYYMMDD`. It must be a real date and not in the future.
* **Animals**: exactly eight animal ids, separated by **commas** -- nothing else
  separates them. A space is not a separator: it stays part of the id next to it,
  so leave spaces out.
* **Each animal id**: letters, then at least four digits -- two for the cohort and
  two for the subject (`ABC0312` is cohort 03, subject 12). A position with no
  animal is written with cohort `00` (for example `ABC0000`). At least one of the
  eight must be a real animal.
* **Tray**: a capital `P`, `E` or `F` followed by the tray number (`P1`, `E2`, `F1`).
* **File type**: `.mkv`, `.avi`, `.mp4`, `.mov` or `.wmv`.

The recording PC and the pipeline's intake apply the same check
(`validate_collage_filename` in `watcher/validator.py`), and the step that cuts the
recording splits the name the same way. A name that fails is held in red on the
recording PC, with the reason (see "What red means" below).

## When

The watcher looks at the recording folder at most once a minute, between its other
jobs: while it is posing a video, the next look comes after that video is done.

**Sending never holds up posing for long.** Each look may spend at most 2 minutes
(`recording_outbox_cycle_seconds`, 120) starting to send recordings; then the
watcher does its next job (a pose, a cut ...), then sends for up to 2 more
minutes, and so on. A PC with hundreds of recordings waiting therefore sends them
a few at a time between poses, over several hours, instead of stopping its work
for half an hour at a time. Each look carries on where the last one stopped, so
a recording that fails every time does not block the ones after it.

The 2 minutes decide when a recording may be *started*. A recording whose copy
has begun is finished (copied, read back, checked, then deleted here), even if
that takes longer: stopping it half way would throw the copy away. Once the time
is up, nothing more is done to any other recording in that look -- no recording
check, no fingerprint, no look at the shared drive -- they just wait for the
next look.

When you send by hand (`mousereach-send-recordings --once`), the one pass may take
up to 30 minutes (`recording_outbox_pass_minutes`), because there is no other work
to make room for. Run it again to carry on.


Only **outside recording hours**: the days and hours already set for this PC
(`watcher.recording_days`, `watcher.recording_hours`; in the Watcher Control panel
they are "Recording happens only on (days)" and "...between (hours)"). If neither is set, the PC
can never tell recording time from any other time, so **nothing is sent** and
the status says so. If recording hours begin while a copy is running, the copy
stops at once, the unfinished copy on the shared drive is removed, and the
recording stays here until the next evening.

The watcher also sends nothing while it is paused by hand (Pause button or
`mousereach-watch-toggle --pause`).

## How a recording is sent safely

1. The name is looked up on the shared drive first -- without searching it (see
   "How the recording PC finds a recording on the shared drive" below): the intake
   and the recording-issue folder (`Processing/Review/_Problematic`) by name, and
   wherever the processing server's list of recordings says it is. A recording is
   not sent twice just because the pipeline has already moved it on. A copy that
   is found is opened and compared with the recording on this PC (size and
   SHA-256 fingerprint) before anything is decided.
2. The recording is copied to `<name>.partial` in the intake. Nothing in the
   pipeline picks up a `.partial` file. The copy runs at low disk priority.
3. The copy is read back **from the shared drive** and its SHA-256 fingerprint
   is compared with the original's. The size must match too.
4. The `.partial` is renamed to the real name. The rename never replaces a file
   that is already there.
5. Only now is the recording deleted from this PC.

If any step fails, the recording stays on this PC, the unfinished copy is
removed, and it is tried again on the next pass. After 3 failures in a row it is
also shown in red. If the PC or the watcher stops in the middle, nothing is
lost: a leftover `.partial` is removed on a later pass and the copy starts
again; a copy that was finished but whose original was not yet deleted is
recognized on the next pass (same name, same content), and only the original is
deleted then.

## How the recording PC finds a recording on the shared drive

A recording that was sent may since have been moved on by the pipeline: cut and
filed under `Analyzed`, put in `Quarantine`, set aside as a recording issue. To
know whether a recording is already there, something has to know every place a
recording can rest. **The recording PC never searches the shared drive's folders
for that.** Searching hundreds of folders over the network on every pass makes a
recording PC crawl whenever its network connection is slow, and the shared
pipeline database, which is on the same drive, then waits behind it for every
machine.

Instead:

* **The processing server writes the list.** Its watcher looks through every
  place a recording can rest (the intake, `Unanalyzed/Unsupported_Tray_Type`,
  `Processing/Quarantine`, `Processing/Review/_Problematic`, and every
  `Multi-Animal` folder under `Analyzed` and `Archive`) at most every 10
  minutes and writes what it found to one file on the shared drive,
  `Processing/.collage_index.json` (name, place, size). To rewrite it at once,
  on the processing server only, in the mousereach environment:
  `python -m mousereach.watcher.collage_index`.
* **The recording PC reads that one file**, and opens single files by their
  name: in the intake, in the recording-issue folder, and at the place the list
  gives.
* **The list is only a pointer.** The recording PC deletes its own copy only
  after opening the copy on the shared drive and finding the same size and the
  same SHA-256 fingerprint. A list that is out of date can make a recording
  wait; it can never make a recording be deleted wrongly.

What happens in each case:

| On the shared drive | What the recording PC does |
|---|---|
| Found by name, same content | Deletes its own copy (it was sent before). |
| Found by name, different content | **Held, in red.** Nothing is overwritten or deleted. |
| The list says it is somewhere, but it is not there now (moved since the list was written) | **Waits**: neither sent again nor deleted, until the next list (at most 10 minutes) shows where it went. |
| Not found anywhere, but the pipeline's database already has a record of this recording | **Waits** the same way. It is never sent twice: a second copy would be cut twice. |
| Still not found after a newer list | **Held, in red**: a person must find where it went. Nothing was sent or deleted. |
| Not found, and unknown to the pipeline | Sent, verified, then deleted here (a new recording). |
| The list is missing or cannot be read (for example the processing server's watcher is not running) | New recordings are still sent and verified as usual, as long as the pipeline's database answers that it has no record of them. The status line says the list is missing. |
| The list is missing or cannot be read, AND the pipeline's database cannot be asked | **Not sent**: nothing can tell a new recording from one already cut and filed. Tried again every pass; shown in red after 3 passes. Nothing is deleted. It goes by itself once either can be read. |

## What red means, and what to do

Held recordings appear in red in two places:

* `mousereach-watch-status` (under "Nodes", each PC lists its
  `[!] HELD <name>: <reason>` lines), and
* the Pipeline Dashboard's red "Needs your attention" panel.

| The reason says | What to do |
|---|---|
| `name is not a valid session name: ...` | Rename the file in the recording folder. The reason says what is wrong. |
| `the recording check could not run here (...)` | Ask whoever looks after MouseReach to install ffmpeg in the mousereach environment. |
| `a different file of the same name is already on the shared drive` | Two different recordings carry one name. Look at both. The usual cause is a typing mistake in one of the names: correct the wrong one. Never delete either until you know which is which. |
| `... (failed N times in a row)` | The copy keeps failing (shared drive unreachable, disk full ...). The rest of the reason says why. When the cause is fixed it goes by itself. |
| `... still does not show it on the shared drive, so a person must find where it went` | This recording was sent before (the pipeline has a record of it), but it cannot be found on the shared drive now. Look for it there (`Analyzed`, `Processing/Quarantine`, `Processing/Review/_Problematic`). Once a copy with the same content is back where the list sees it, the recording PC deletes its own copy by itself. Never delete the copy on the recording PC by hand until the copy on the shared drive has been found. |
| `the recording folder does not exist or cannot be read` | `watcher.recording_outbox` points at the wrong folder. Correct it. |
| `neither ffprobe nor ffmpeg is installed` | Ask whoever looks after MouseReach to install ffmpeg in the mousereach environment. |

## Looking without changing anything

On the recording PC, in the mousereach environment:

```
conda activate mousereach
mousereach-send-recordings --preview
```

This lists every file and what would happen to it. It copies and deletes
nothing, and it works at any hour.

## Sending now by hand

```
mousereach-send-recordings --once
```

One pass, with exactly the same rules (it still never runs during recording
hours). Use it on a PC whose watcher is not running. A file must have been seen
unchanged for 5 minutes, so the first run after a long break may only note the
files; run it again after 5 minutes.

## Sending a recording again

A recording that was sent is no longer on the recording PC: its copy on the
shared drive is the only one, in the intake or wherever the pipeline has moved it.
To send it again, find it there. Putting a copy back into the recording folder does
not send it again: if the shared drive still has the same file, the copy here is
simply deleted.

## Correcting a name after the recording has left the PC

Once a recording has been sent, the copy on the shared drive is the only one, and
the pipeline may already have cut it into one video per animal under the name it
had. MouseReach has **no command that renames a sent recording**: a rename touches
the recording and every video cut from it, so it is done by whoever looks after the
pipeline, who keeps the original and a record of the correction in
`Analyzed/Archive/manual_corrections`. Do not rename files on the shared drive
yourself.

| The situation | What to do |
|---|---|
| The name was never a valid session name | It never left the PC: it is held in red there. Rename it in the recording folder; it is sent on the next pass outside recording hours. |
| A file whose name the intake could not read is in the Quarantine folder | In `mousereach`, open the **Quarantine** tab, type the right name in the "Corrected name (editable)" column of its row, choose **Collage intake (8-camera videos)** under "Release to:", and press **Release selected**. A name the intake would not accept is refused with the reason, and nothing moves. |
| The name is a valid session name but wrong (wrong animal, wrong tray, wrong date) | Tell whoever looks after the pipeline: the name the recording has and the name it should have. |

**No label-mismatch hold.** The pipeline does not hold a video
whose results fit another tray's hand scores better than its own: a video is
always processed and filed under the name a person gave it, and compared with
its OWN tray's hand scores only. `mousereach-label-mismatch --decisions` still
prints the decisions recorded while the hold existed; the commands that made
decisions are refused. A name that is wrong is still fixed the way the table
above says.

## Settings (in `~/.mousereach/config.json`, section `watcher`)

| Key | Default | Meaning |
|---|---|---|
| `recording_outbox` | not set (sending off) | The folder the recording program saves into, where people rename the files. |
| `recording_outbox_stable_minutes` | 5 | How long a file must be seen unchanged before it counts as finished. |
| `recording_min_minutes`, `recording_min_frames`, `recording_min_kb_per_second` | 5, 3000, 5 | The recording check, shared with the intake: a recording below any of these is a recording issue (see RECORDING_ISSUES.md). It is still sent. |
| `recording_outbox_warn_after_failures` | 3 | Failed copies in a row before a file is shown in red. |
| `recording_outbox_cycle_seconds` | 120 (at least 10) | How long the running watcher may spend starting to send, each time it looks, before it goes back to its other work (posing, cutting). A recording already being copied is finished first. |
| `recording_outbox_pass_minutes` | 30 | How long one pass started by hand (`mousereach-send-recordings --once`) may spend starting to send. The running watcher does not use it. |
| `recording_days`, `recording_hours` | (already exist) | Recording time. Nothing is sent inside it, and nothing at all is sent if both are unset. |
| `pause_while_running` | (already exists) | While a listed recording program runs, the newest file in the folder is taken as its current recording and left alone. If the recording program is left open after the last recording of the day has been renamed, that last recording waits until the program is closed or a new recording starts. |

Example (placeholders, not real paths):

```json
"watcher": {
  "recording_days": "<days recording can happen, e.g. Mon-Fri>",
  "recording_hours": "<hours recording can happen, e.g. 08:00-18:00>",
  "recording_outbox": "<the recording program's output folder>"
}
```

The running watcher picks up `recording_outbox` and the other `recording_outbox_*`
keys when it starts. Restart it (stop flag, then start) after changing them.
Changes to the recording days and hours apply at once.

## Where it is written down

* Every send, skip, stopped copy and delete:
  `recording_sends.jsonl` in the watcher's log folder (`watcher_logs` in this PC's
  processing folder), one line each, with the SHA-256 of every copy. A recording
  deleted here because an identical copy was already on the shared drive
  (`deleted_local_already_on_share`) carries the SHA-256 both copies were found
  to have, so every delete shows what it was based on.
* What the sender remembers between passes (sizes seen, failures, conflicts, and
  `resume_at`, the first recording the last look did not reach):
  `recording_outbox_state.json` in this PC's processing folder. Deleting it is
  safe: the sender then waits 5 minutes before sending anything.
* The latest pass: the `recording_outbox` part of this PC's status file,
  `Processing/.node_status/<this PC>.json` on the shared drive. Its
  `collage_index` part says how old the processing server's list was when the
  pass read it (`status` is `ok`, `missing` or `unreadable`; `age_minutes`).
* The processing server's list of where every recording rests:
  `Processing/.collage_index.json` on the shared drive (written by the
  processing server only).
