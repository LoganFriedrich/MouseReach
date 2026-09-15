# Design: the filesystem is the state, the database is derived

STATUS: PARTLY BUILT (2026-09-14). Built: the read-only check (step 0), the
rule that every walk of Analyzed skips superseded outputs, and the stage-folder
layout (steps 1 and 2: config, the folder builder, the archive root, guard-file
handling). Not yet built: claim by rename (step 3), rebuilding the database
from folders (step 4), folder-first readers (step 5).
For what the pipeline does today, read PIPELINE_AS_BUILT.md.

Decided 2026-09-14: a video's position in the pipeline is determined by which
folder contains it. The database is a cache of that fact, rebuildable by
scanning, and never the authority. A watcher becomes an automation
convenience rather than a load-bearing component.


## WHY

Two halves of the system already disagree. Triage and deep review work exactly
this way: a folder holds the video until a person clears it, then it moves
back. The census resolves every session's pipeline position from folders alone,
with no database consulted at all. Everything else drifted into database-only
state, where the folder a file sits in means nothing and a row means
everything.

The cost of that split is not theoretical. A backup copy of one node's database
was once read as current state, and a large batch of finished videos was
reported as needing a fresh pose each. They were done; their results had been
sitting in `Analyzed/` for days. Under this design the question is answered by
looking at which folder holds the video, and the mistake is not available to
make.

A second property matters as much: a person who has read no documentation
should be able to open the share and see where everything is.


## WHAT "DONE" MEANS

A video is done only when BOTH are true:

  1. Its analysis is current, or declared compatible with current. "Current"
     means produced by the pose model and tool versions declared in
     `pipeline_versions.json`, as recorded in the video's own
     `_processing_manifest.json`. Compatibility is declared once, in the same
     file (`compatible_versions`), and is already honoured by
     `versions.compare_manifest_to_current`. Every human analysis of the video
     must be reflected in that current analysis.
  2. That analysis is where it SHOULD be: beside the video in
     `Analyzed/<project>/<cohort>/`, with project and cohort worked out from the
     video's name (`archive/core.get_archive_destination`). Not where it
     happens to be, and not wherever a database row says it is.

Fail either and the video is not done. Agreement between files and records
proves nothing, because both can agree on the wrong folder or an old version.
The right place and the current versions are fixed by rule; the check compares
reality against those rules. There is no exceptions list.

"Outdated" is therefore not a state anyone records. It is computed from the
files every time: the current analysis is simply not there yet.

Human analyses (ground truth, causal reviews) are never outdated. They are
statements about the video itself -- at this frame a reach happened that did
this to the pellet -- so they stay true whichever tool version they were made
alongside. They always stay with the video, are never archived, and must
reach the analysis. `archive/supersede._NEVER_SUFFIXES` and
`reprocessor._drop_human_seg_staleness` already lean this way.

A name known to be wrong is repaired, not tolerated, because the right place
is computed from the name. Each repair is confirmed case by case, starting
with the collage (singles are re-cut from it), and leaves a small note beside
the renamed file recording the old name. Nothing renames files from a pattern.


## STAGE FOLDERS

Every folder is a place work RESTS: waiting for a machine, waiting for a
person, or finished. There are no per-algorithm folders. The four algorithms
run back to back on one local copy in seconds, and nothing ever runs them
independently, so a folder between each would add network moves and no
information.

    Unanalyzed/Multi-Animal/          collage, waiting to be cut
    Unanalyzed/Single_Animal/         single, waiting for a pose
    Processing/Posed/                 posed; the algorithms run from here
    Processing/Review/triage/         held for a person
    Processing/Review/deep_review/    held for a person
    Processing/Failed/                needs a person
    Processing/Quarantine/            filename failed the gate
    Analyzed/<project>/<cohort>/      finished single, with its outputs
    Analyzed/Archive/                 superseded outputs only

Folders that are NOT stages, and that a scan must classify rather than read as
a video's position: `Processing/Repose_Queue/` (request files, not videos),
`Unanalyzed/Unsupported_Tray_Type/` (tray types this pipeline does not
analyse), any folder whose name starts with `.` or `_` (claims, in-flight
work, retired bundles), and any leftover folder from the previous layout while
it waits for cleanup. One such folder holds videos:
`Unanalyzed/Single_Animal/.inflight/<machine>/` holds singles a GPU node has
claimed for pose (watcher/single_claim.py). A scan counts each one as still
waiting in `Unanalyzed/Single_Animal`, never as a stage of its own and never
as lost (2026-09-15).

Both `Unanalyzed/` folders accept new videos at any time, even while nodes run.
A video is taken only once it has stopped changing, and singles are claimed by
a rename so only one node poses each one.

The video and every output it has so far travel together as one bundle.


## ARCHIVE LIVES UNDER THE FINISHED TREE

Superseded outputs move from a top-level `Archive/` to `Analyzed/Archive/`. A
top-level archive beside `Unanalyzed/` and `Processing/` reads like a stage,
and it is not one. Nothing in it is waiting for anything.

What belongs there: superseded pose generations and the algorithm outputs
computed from them, in the existing versioned layout
(`DLC Model <gen>/<algo stack>/`). What does not: any only copy of a video,
anything a current result depends on, and human analyses.

Constraints on the move:

  * The read-only historical source tree (`Archive/historical/`, read by
    `aspa/import_collages.py` and counted by `census/runner.py`) does NOT move.
    It is source data produced by older tools, copied out of and never written
    into. The rule: no code path opens a file for writing beneath it. Nothing
    enforces that automatically today, so every change touching it has to be
    checked by hand against the rule.
  * The move must happen with every watcher stopped. The normal archiving step
    (`archive/core.py` -> `supersede_video_outputs`) writes into the archive
    whenever it replaces an earlier generation, so a watcher running old code
    would recreate the top-level folder right after it moved.
  * The move adds `Analyzed\` to every path beneath. Measure the longest path
    first: anything that would reach 260 characters becomes unopenable on
    Windows. Subtrees that would cross the limit and hold no superseded video
    outputs (for example old development material) stay where they are until
    cleanup.
  * Rename within the share (`os.rename` / `Move-Item`), never `shutil.move`,
    which silently falls back to copy-and-delete across volumes.
  * Then change ONE constant: `archive/supersede.default_archive_root`, which
    becomes `<Analyzed>/Archive`. The constants in `aspa/import_collages.py`
    and `census/runner.py` point at `Archive/historical/`, which does not move,
    and must stay as they are; repointing them would send the collage importer
    and the census looking for historical source material in the wrong place.
    `migrate_to_processing.py` builds an `Archive` path from the LOCAL
    processing root and is out of scope.
  * Every walk of Analyzed already skips folders named `Archive` and
    `_archived` (`pipeline/analyzed_tree.py`), because archived files keep
    their original names and would otherwise be read as live results. Keep the
    archive folder named `Archive` for that reason. `DLC Model <N>/` folders
    are deliberately NOT skipped: inside a project they are live pose storage,
    and hiding them would make finished videos look like they need a pose.


## CLAIM BY RENAME, RECLAIM BY TIMEOUT

One stage, in order:

  1. Look first, then claim: atomically rename the bundle into
     `<stage>/.inflight/<hostname>/<stem>/`. A rename either succeeds or does
     not; `FileNotFoundError` means another machine won. This is the pattern
     `watcher/repose.py` already uses.
  2. Copy to local scratch and work there, touching the in-flight folder as a
     heartbeat.
  3. Write results back beside the video, still inside `.inflight/`.
  4. Atomically move the bundle to the next folder.
  5. Delete the local scratch copy.

This replaces the `.claims` marker files, which are not atomic (write, sleep,
read back).

A crash leaves the bundle in `.inflight/<hostname>/`. Recovery must run on
EVERY poll, not only at startup: the startup reclaim is safe only for the
machine's own folder, and a bundle stranded by a different machine would
otherwise wait forever. A bundle whose heartbeat is older than
`repose.STALE_S` goes back to its stage folder and is simply claimed again.

Nothing is ever deleted from the share to advance a stage. A move that fails
leaves the bundle where it was.


## THE DATABASE AFTER THIS

Derived. A scan of the stage folders reconstructs every row.

  * Calculated from the files, never stored: whether the current analysis
    exists, and therefore what needs re-running.
  * Written as a small note in the video's folder when it becomes true: failure
    reason and retry count, why a person asked for a re-run, and which collage
    the single came from.
  * Kept only in the database, as history: per-step durations, the processing
    log, first-seen time. Losing this never makes a video look unfinished.

A rebuild writes a FRESH database beside the live one and diffs them. A rebuild
that overwrites its own comparison target cannot be checked. Promotion is a
manual rename.

Disagreement between a row and a folder is resolved in the folder's favour,
always.


## WHAT THIS BREAKS, AND MUST BE CHANGED WITH IT

  * DONE: `config.Paths` names the stage folders (the single-animal, post-pose
    and deep-review paths are repointed); the roughly one hundred `Paths.*`
    call sites follow.
  * `watcher/locate.py` search order: the stage folders in order of currency,
    the archive last.
  * `dashboard/widget.py` reads folders only. The database adapter, and its
    reads of a database at a hardcoded path, go.
  * The review gate's ground-truth lookup (`causal_review_io.find_gt`) indexes
    only the local processing tree. Ground truth kept beside a finished video
    must be visible to it, or human truth sits beside the video without ever
    reaching the analysis.
  * `review_return` retires a stale review bundle whenever the video already
    reads as finished. It must refuse when the bundle carries a review the
    finished copy lacks.
  * `review_return._resolve_inputs` assumes bundles are not self-contained,
    although bundles carry their own media. Settle that before any cleanup
    deletes staged copies.
  * `index/index.py` carries a three-entry `STAGES` list from an older
    architecture; update it or remove it.


## MIGRATION ORDER

Each step is reversible until the last.

  0. A read-only check, `mousereach-reconcile`, compares every video against
     the definition of done and reports mismatches PER VIDEO, never as counts
     (counts can match while being wrong in both directions). Databases are
     read only to report where they disagree; they never decide anything. Run
     it before and after every later step: the mismatch set must shrink and
     never grow.
  1. Create the new stage folders (`pipeline/pipe_structure.TARGET_DIRS`,
     idempotent).
  2. The freeze. It cannot be incremental: code resolves a stage to a path and
     then lists it, so there is no moment when both names are right, and a
     machine listing mid-rename fails silently.
       a. Stop every watcher and every importer; close review tools.
       b. Prove the freeze: record folder modification times, wait, compare.
       c. Snapshot every database, with checksums, off the share.
       d. File superseded poses into the archive. Name the OLD scorer
          explicitly: manifests may already name the current model.
       e. Move the archive (constraints above).
       f. Move folders carrying real waiting work only, never leftovers;
          repoint `config.Paths`. The old stage folders
          (`Processing/Single_Animal/`, `Processing/DLC_Complete/`) hold only
          leftover copies of finished work, so they are renamed whole into
          `Processing/_leftovers_pending_cleanup_<date>/`. The leading `_`
          marks them as not a stage, and they wait there for cleanup.
          `Processing/Review/flagged_for_review/` is renamed to
          `Processing/Review/deep_review/`, and the paths recorded inside its
          bundle manifests are rewritten to match.
       f2. Leave a guard FILE (a plain file, not a folder) at each old folder
          name: `Processing/Single_Animal`, `Processing/DLC_Complete`,
          `Processing/Review/flagged_for_review`. WHY: code that still uses an
          old name would otherwise quietly recreate the empty folder and carry
          on working in the old layout, splitting the pipeline in two with no
          error anywhere. With a file in the way, creating or listing that
          folder raises (`FileExistsError`, `NotADirectoryError`) and the
          stale code stops loudly. It follows that new code must never create,
          list or glob those old names, and that generic folder listings must
          not crash when they meet a file where a folder was expected.
       g. Every machine pulls current code before it restarts.
  3. Claim by rename, reclaim by timeout.
  4. Rebuild the database from folders into a fresh file and diff.
  5. Switch readers to folders. This is the point of no return: gate it on
     step 0 reporting no mismatches.

Do not rewrite the work-selection order (`orchestrator._select_work_item`) as
part of this. It encodes two production failures and its tests pin it.


## SETTLED

  * No per-algorithm folders.
  * The historical source tree does not move and is never written.
  * Human analyses are never outdated and never archived.
  * Status lives on disk; history lives in the database.
  * Stranded work is reclaimed automatically on timeout.
  * Migration freezes in place, migrates, and resumes; it does not drain first.
  * Leftover working copies are not moved into the new layout. Cleanup waits
    until every other step is done.
