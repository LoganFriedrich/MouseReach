# Sending recordings from a recording PC to the pipeline

Verified against: 0c36d0d (2026-11-18), with the change that added this page.

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

## When

The watcher looks at the recording folder at most once a minute, between its other
jobs: while it is posing a video, the next look comes after that video is done.


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

1. The shared drive is searched for the name first: the intake, the
   unsupported-tray, quarantine and recording-issue (`Processing/Review/_Problematic`)
   folders, and every `Multi-Animal` folder under
   `Analyzed` and `Archive`. A recording is not sent twice just because the
   pipeline has already moved it on.
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

## Settings (in `~/.mousereach/config.json`, section `watcher`)

| Key | Default | Meaning |
|---|---|---|
| `recording_outbox` | not set (sending off) | The folder the recording program saves into, where people rename the files. |
| `recording_outbox_stable_minutes` | 5 | How long a file must be seen unchanged before it counts as finished. |
| `recording_min_minutes`, `recording_min_frames`, `recording_min_kb_per_second` | 5, 3000, 5 | The recording check, shared with the intake: a recording below any of these is a recording issue (see RECORDING_ISSUES.md). It is still sent. |
| `recording_outbox_warn_after_failures` | 3 | Failed copies in a row before a file is shown in red. |
| `recording_outbox_pass_minutes` | 30 | How long one pass may spend copying before the watcher does its other work. |
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
  processing folder), one line each, with the SHA-256 of every copy.
* What the sender remembers between passes (sizes seen, failures, conflicts):
  `recording_outbox_state.json` in this PC's processing folder. Deleting it is
  safe: the sender then waits 5 minutes before sending anything.
* The latest pass: the `recording_outbox` part of this PC's status file,
  `Processing/.node_status/<this PC>.json` on the shared drive.
