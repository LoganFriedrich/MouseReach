# Design proposal: the filesystem is the state, the database is derived

STATUS: PROPOSAL. Nothing here is built. This describes a target, not the
running system. For what the pipeline does today, read PIPELINE_AS_BUILT.md.

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

The cost of that split is not theoretical. On 2026-09-14 a backup copy of one
node's database was read as current state and roughly 1,100 finished videos
were reported as needing about 14 minutes of GPU each. They were done. Their
results had been sitting in `Analyzed/` for eleven days. Under this design the
question is answered by looking at which folder holds the video, and the
mistake is not available to make.

A second property matters as much: a person who has read no documentation
should be able to open the share and see where everything is. Today they
cannot, because between cropping and the end of pose estimation a single-mouse
video exists only on one machine's local disk.


## THE INVARIANT

At any moment, exactly one folder on the share holds the canonical copy of a
video, and that folder names what has been done to it. Everything else is a
working copy that may be deleted without loss.

Work still happens on local disk, for the reasons it always did: DeepLabCut
writes its output beside its input, and SQLite over a share loses writes. The
change is that the local copy is explicitly a scratch copy with no authority,
and the canonical file on the share moves when a stage completes.


## STAGE FOLDERS

    Unanalyzed/Multi-Animal/        collage, waiting to be cut
    Unanalyzed/Single_Animal/       single, waiting for a pose
    Processing/Posed/               has a current-model pose
    Processing/Segmented/           segmentation done
    Processing/Reaches/             reach detection done
    Processing/Outcomes/            outcome detection done
    Processing/Assigned/            assignment done
    Processing/Review/triage/       held for a person
    Processing/Review/deep_review/  held for a person
    Processing/Failed/              needs a person, retryable
    Processing/Quarantine/          filename failed the hard gate
    Analyzed/<project>/<cohort>/    finished single, with its outputs
    Analyzed/<project>/<cohort>/Multi-Animal/   finished collage
    Analyzed/Archive/               old work nobody cares about any more

The video and every output it has so far travel together as one bundle. The
existing rule that human truth never leaves the video (ground truth, causal
review) is unchanged.


## ARCHIVE IS A KIND OF FINISHED, AND LIVES UNDER THE FINISHED TREE

`Archive/` moves from the top of the pipeline folder to `Analyzed/Archive/`,
and its meaning is stated in one line: old work nobody cares about any more.
Kept because deleting it would be irreversible, not because anything reads it.

This matters more than it sounds. A top-level `Archive/` sitting beside
`Unanalyzed/` and `Processing/` reads like a stage, and it is not one. Nothing
in it is waiting for anything. Putting it under the finished tree says what it
is by where it is, which is the whole point of this design.

What belongs there: superseded pose generations and the algorithm outputs that
were computed from them, per the existing versioned layout
(`DLC Model <gen>/<algo stack>/`). What does not: any video that is the only
copy, and anything a current result depends on.

Consequences:

  * `archive/supersede.py:default_archive_root` returns `NAS_ROOT / "Archive"`
    and becomes `NAS_ROOT / "Analyzed" / "Archive"`. One line, but every path
    already recorded in an existing archive README points at the old location.
  * The move itself is a rename within one volume, so it is metadata only and
    does not copy the 3.7 TB currently under `Archive/`.
  * `aspa/import_collages.py` and `census/runner.py` both name
    `Archive/historical/ASPA` explicitly and would need the new path. That
    corpus is live source data for an active import, not old work, so whether
    it belongs under an archive named "nobody cares about this" is a separate
    question worth asking.
  * `migrate_to_processing.py:190` also builds an `Archive` path, but from the
    LOCAL processing root rather than the share, and belongs to the one-time
    migration off the previous architecture. Different folder, out of scope
    here; listed so the next person greps once and finds all four.


## THE MOVE IS THE COMMIT

One stage, in order:

  1. Claim: atomically rename the bundle into `<stage>/.inflight/<hostname>/`.
     A rename either succeeds or does not, so two machines cannot claim the
     same video. This is the pattern `watcher/repose.py` already uses.
  2. Copy to local scratch. Work there.
  3. Write results back beside the video, still inside `.inflight/<hostname>/`.
  4. Atomically move the bundle to the next stage folder.
  5. Delete the local scratch copy.

A crash leaves the bundle in `.inflight/<hostname>/`. Recovery moves anything
there back to its own stage folder and it is simply claimed again. That is the
same reclaim already implemented in `BaseOrchestrator._reclaim_orphaned_work`,
applied to folders instead of database states.

Nothing is ever deleted from the share to advance a stage. A move that fails
leaves the bundle where it was.


## THE DATABASE AFTER THIS

Derived. A scan of the stage folders reconstructs every row. The database
remains useful for history, timings, error messages and the processing log,
which the filesystem cannot express. It stops being consulted to answer "where
is this video" or "is this done".

Consequences worth stating plainly:

  * A rebuild command must exist and must be safe to run at any time.
  * Anything that today reads state from a database row needs to read the
    folder instead, or read a cache that a scan refreshes.
  * `watcher_state/<host>/watcher.db` becomes uninteresting rather than
    dangerous: a backup of a cache is harmless.
  * Disagreement between a row and a folder is resolved in the folder's favour,
    always, with no judgement call.


## WHAT THIS BREAKS, AND MUST BE CHANGED WITH IT

  * `config.Paths` gains the stage folders; `DLC_QUEUE` and the local
    `PROCESSING` folder become explicitly scratch.
  * `watcher/locate.py` search order: the stage folders become the places a
    video legitimately lives, and the archive stays last.
  * `census/runner.py` already works this way and gets simpler, not harder.
  * `dashboard/widget.py` reads folders, which also removes the current bug
    where it opens a database at a hardcoded path that on at least one machine
    holds zero rows.
  * `index/index.py` carries a three-entry `STAGES` list from the previous
    architecture; it is either updated to the list above or removed.
  * The archive readiness check currently consults a cached index that the
    watcher never writes, so a stage's completion is not reliably recorded
    anywhere the move could depend on. This has to be settled before a move
    can be gated on it.


## MIGRATION ORDER

Each step leaves the pipeline working.

  1. File superseded poses into the versioned Archive, so each video has one
     current pose beside it. Already validated by dry run; independently
     useful; the same principle at small scale.
  1b. Move `Archive/` to `Analyzed/Archive/` and point
     `default_archive_root` at it. A rename within one volume, so no data is
     copied. Do this AFTER step 1 rather than before, so the pose filing runs
     against a path that is not moving under it.
  2. Create the stage folders and have the watcher WRITE position to them
     while still reading state from the database. Nothing depends on the
     folders yet, so a mistake costs nothing.
  3. Add the rebuild-from-folders command and verify it reproduces the live
     database exactly. This is the gate: if it does not reproduce, the folders
     are not yet the truth and step 4 must not start.
  4. Switch readers over to the folders, one at a time, starting with the
     dashboard and census.
  5. Remove the database as an authority. It stays as history.

Step 3 is the decision point. Until a scan can reproduce the database, this
design is not real and should not be relied upon.


## OPEN QUESTIONS FOR THE LAB

  * Should a single that fails one algorithm sit in the previous stage folder
    or in `Failed/`? Failed is retryable today and carries a retry count.
  * Do collages need per-stage folders too, or is discovered/cropped enough?
  * Should `Analyzed/<project>/<cohort>/` keep videos and outputs together as
    it does now, or separate `Single_Animal/` and `Multi-Animal/` beneath it?
  * How long may a bundle sit in `.inflight/<hostname>/` before recovery
    reclaims it? The pose is the longest step at roughly fourteen minutes.
