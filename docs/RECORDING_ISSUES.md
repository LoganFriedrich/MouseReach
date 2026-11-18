# Recording issues: recordings that failed

This page is for anyone who records, reviews, or looks after the pipeline. You
do not need to read any code.

## What a recording issue is

Sometimes a recording fails: the recording program stops after a fraction of a
second, the file is empty, or it cannot be opened at all. Such a file is not a
recording of a session, and cutting it into one video per mouse only gives eight
empty videos for a reviewer. The pipeline therefore **checks every multi-animal
recording before it cuts it**. A recording that fails the check is a
**recording issue**.

A recording is a recording issue when any of these is true:

| Check | Setting (in `~/.mousereach/config.json`, section `watcher`) | Default |
|---|---|---|
| It cannot be read as a video at all (an empty file is this case) | -- | -- |
| It is shorter than a few minutes | `recording_min_minutes` | 5 |
| It has too few frames | `recording_min_frames` | 3000 |
| It is far too small for its length | `recording_min_kb_per_second` | 5 |
| Its first frame, or a frame near its end, cannot be decoded | -- | -- |

The defaults are far below any real session and far above any recording that
stopped by accident. Change them only if your sessions are much shorter.

If the check itself cannot run (ffmpeg is not installed, or the shared drive did
not answer), **nothing is set aside**: an outage is never taken for a failed
recording.

## Where it goes

The recording is moved from `Unanalyzed/Multi-Animal` to

    Processing/Review/_Problematic/

with a reason file beside it, `<recording name>.recording_issue.json`, which
says in plain words what was wrong (for example
`recording issue: 4 frames, 0.08 s -- the recording failed`), the
measurements, who or what found it (the intake check on a named machine, or a
person) and when, and which single-mouse sessions the recording was meant to
hold.

It is **never cut, never reviewed, and never deleted**. No machine tries it
again. Its sessions will never have reach data or kinematics; their hand pellet
scores, if the trays were scored, are kept as usual.

## Where you see it

* `mousereach-watch-status`: a section "Set aside as RECORDING ISSUES", one line
  per recording.
* The Pipeline Dashboard: listed under the red panel, one line per recording.
  It is not shown as an alarm: there is nothing left to do in the pipeline.
* `mousereach-recording-issue --list`: every recording set aside, with its
  reason.
* The pipeline census (`mousereach-census`): the sessions of such a recording
  are counted as `recording_issue`, never as waiting.
* The database tool (mousedb): its Where Is My Data tab and the cohort's
  FIX THESE file say `recording failed: <recording>, <reason>`, the list of
  hand-scored sessions without a video says `recording_failed` instead of
  `missing`, and the session rows of the exports carry the reason. That is
  where a person **acknowledges** it once (see the mousedb guide, "A recording
  that failed").

## Setting a recording aside by hand

If you know a recording failed (or the automatic check let one through), on
any machine with MouseReach:

```
conda activate mousereach
mousereach-recording-issue "<recording file name>" --reason "<what happened, in plain words>" --by "<your name>"
```

The recording must still be in `Unanalyzed/Multi-Animal`. The command refuses
if the recording has already been cut, or is being cut right now by another
machine. Every watcher stops trying to cut it at its next scan, whichever
machine you ran the command on.

To see what the automatic check says about a recording, without moving
anything:

```
mousereach-recording-issue "<recording file name>" --check
```

## Recordings sent from a recording PC

A recording PC that sends recordings to the shared drive (see
[SENDING_RECORDINGS.md](SENDING_RECORDINGS.md)) applies the **same check with
the same settings**. A failed recording is still sent -- it documents what
happened that day -- and is listed there as a recording issue. The pipeline
then sets it aside as above.

## If a recording was set aside by mistake

1. Move the recording from `Processing/Review/_Problematic` back into
   `Unanalyzed/Multi-Animal`.
2. Move its `.recording_issue.json` file out of `_Problematic` into an archive
   folder (keep it; never delete records).
3. If the automatic check was wrong, lower the setting that caught it first,
   because the recording is checked again when it is picked up.
