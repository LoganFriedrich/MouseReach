"""
core.py - Archive logic for moving completed videos to NAS.

Videos can only be archived when ALL validation statuses are "validated".
This is the only way files leave the Processing/ folder.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import logging
import shutil
import time
from datetime import datetime

from mousereach.config import Paths, get_video_id, AnimalID


def get_archivable_videos() -> List[str]:
    """Get list of videos that are ready to be archived.

    A video is archivable when ALL of:
    - seg_validation == "validated"
    - reach_validation == "validated"
    - outcome_validation == "validated"

    Returns:
        List of video_ids ready for archiving
    """
    from mousereach.index import PipelineIndex

    index = PipelineIndex()
    index.load()
    return index.get_ready_to_archive()


def get_video_files(video_id: str) -> List[Path]:
    """Get all files associated with a video in Processing folder.

    Args:
        video_id: Video identifier (e.g., "20250704_CNT0101_P1")

    Returns:
        List of file paths
    """
    # files_of_video: the one rule for "this video's files" -- a stem that is
    # a prefix of another video's (..._P1 / ..._P10) does not take its files,
    # and an unfinished copy (<name>.part) is never filed.
    from mousereach.pipeline.fsutil import files_of_video
    return files_of_video(Paths.PROCESSING, video_id)


# The three stage outputs, and where each records the verdict triage gave it.
_STAGE_FILES = {
    "seg": "_segments.json",
    "reach": "_reaches.json",
    "outcome": "_pellet_outcomes.json",
}

_ACCEPTED = {"validated", "auto_approved"}


def _status_from_files(video_id: str, source_dir: Path) -> Dict[str, str]:
    """Read each stage's validation_status straight out of its output file."""
    status: Dict[str, str] = {}
    for stage, suffix in _STAGE_FILES.items():
        f = Path(source_dir) / f"{video_id}{suffix}"
        try:
            if not f.is_file():
                status[stage] = "not_started"
                continue
            with open(f) as fh:
                status[stage] = json.load(fh).get("validation_status") or "not_started"
        except Exception:
            # Unreadable is not approved. Saying "not_started" here is honest:
            # we cannot show that a human or the triage step cleared this stage.
            status[stage] = "unreadable"
    return status


def check_archive_ready(video_id: str, source_dir: Path = None) -> Tuple[bool, Dict[str, str]]:
    """Check if a video is ready for archiving.

    Reads the verdict out of the stage output files. It used to ask the pipeline
    index -- a cache whose whole purpose is to save a folder scan for the
    dashboard -- and that cache is allowed to be stale. It was, comprehensively:
    a node held videos in 'processed' that were marked approved by every stage
    ON DISK, while the index said otherwise for the large majority (segmentation
    reading 'auto_review', a spelling only the index ever uses, and reach and
    outcome missing entirely for most). Nothing archived for months and the
    retry loop failed on every pass.

    A cache is the wrong authority for an irreversible decision. The files carry
    validation_status; that is the fact. The index stays what it is good at --
    making the dashboard fast.

    Args:
        video_id: Video identifier
        source_dir: Where the outputs live. Defaults to the local Processing dir.

    Returns:
        (is_ready, status_dict) - status_dict has keys seg, reach, outcome
    """
    if source_dir is None:
        source_dir = Paths.PROCESSING
    if source_dir is None:
        return False, {k: "no processing dir configured" for k in _STAGE_FILES}

    status = _status_from_files(video_id, Path(source_dir))
    is_ready = all(status.get(k) in _ACCEPTED for k in _STAGE_FILES)
    return is_ready, status


# The stage outputs a features file is made from. A features file older than
# any of them was not made from them.
_FEATURES_INPUTS = ("_segments.json", "_reaches.json", "_pellet_outcomes.json")

# The coarsest modified-time step of the file systems involved.
_MTIME_SLACK_S = 2.0


def stale_features_reason(video_id: str, folder, started_at=None) -> Optional[str]:
    """Why ``<stem>_features.json`` in ``folder`` cannot be the kinematics of
    the outputs beside it, or None when it can.

    Reasons: it is missing; it is older than one of the stage outputs present
    (segments, reaches, outcomes); or, when ``started_at`` (the watcher row's
    ``processing_started_at``, ISO text) is given, it is older than the start
    of the run. WHY: a re-run copies the filed generation down, the old
    features file included, and a run whose kinematics then failed -- or a
    row put back to 'processed' by hand -- would otherwise be filed with new
    outputs beside old kinematics. A features file this run made is written
    after every stage output and after the run started, so a fresh one always
    passes. An unreadable modified time counts as stale (the cautious side).
    """
    folder = Path(folder)
    feats = folder / f"{video_id}_features.json"
    try:
        f_mtime = feats.stat().st_mtime
    except FileNotFoundError:
        return f"no {feats.name}"
    except OSError as e:
        return f"{feats.name} could not be read ({e})"
    for suffix in _FEATURES_INPUTS:
        p = folder / f"{video_id}{suffix}"
        try:
            m = p.stat().st_mtime
        except FileNotFoundError:
            continue
        except OSError as e:
            return f"{p.name} could not be read ({e})"
        if m > f_mtime + _MTIME_SLACK_S:
            return (f"{feats.name} is older than {p.name}: it was not made from "
                    f"these outputs (an earlier run's kinematics)")
    if started_at:
        try:
            started = datetime.fromisoformat(str(started_at)).timestamp()
        except (TypeError, ValueError):
            started = None
        if started is not None and f_mtime < started - _MTIME_SLACK_S:
            return (f"{feats.name} is older than the start of this run "
                    f"({started_at}): an earlier run's kinematics")
    return None


def _watcher_db_file_exists() -> bool:
    """Whether this machine's configured watcher database file exists. A
    machine with no watcher configured at all has none (False)."""
    try:
        from mousereach.watcher.db_location import resolve_watcher_db_path
        return Path(resolve_watcher_db_path()).is_file()
    except Exception:
        return False


def command_filing_refusal(video_id: str, source_dir: Path = None) -> Optional[str]:
    """Why the ``mousereach-archive`` COMMAND may not file this video, or None.

    The watcher files a video only after every check has passed: the quality
    triage, the hand-score comparison, the review gate and kinematics. The
    command used to file on the stage files' validation_status alone -- a
    verdict the quality triage writes BEFORE the hand-score check and the
    gate run, so a video held for its hand scores, or held as failed for want
    of kinematics, read 'auto_approved' and was filed. Filing means done for
    good, so the command now asks what the watcher decided:

      * this node's watcher database knows the video and its row is not
        'processed' (passed every check, waiting to be filed) -> refused,
        naming the state;
      * no ``<stem>_features.json``, or one older than the outputs beside it
        (stale_features_reason) -> refused.

    No watcher database at the named place means a machine that does not run
    the watcher (the tool used on its own); then the files are all there is to
    go on. But when this login's settings cannot NAME the watcher database (they
    cannot be read, or a processing server's settings have no
    watcher.db_path -- db_location.watcher_db_unnamed_reason), the command
    refuses: "cannot tell which database" is not "no database", and the row
    that would refuse the filing may be in the one it cannot name.
    Never called by the watcher itself (skip_ready_check), whose own filing
    step makes these checks.
    """
    folder = Path(source_dir) if source_dir else Paths.PROCESSING
    started_at = None
    try:
        from mousereach.watcher.db_location import watcher_db_unnamed_reason
        unnamed = watcher_db_unnamed_reason()
    except Exception as e:
        unnamed = f"the watcher database could not be looked up ({e})"
    if unnamed:
        return f"{unnamed}; not filing without reading the watcher's row"
    try:
        from mousereach.watcher.db_location import open_watcher_db_if_present
        db, why_none = open_watcher_db_if_present()
    except Exception as e:
        db, why_none = None, f"the watcher database could not be opened ({e})"
    if db is None and _watcher_db_file_exists():
        # A database that is there but cannot be opened is not "no watcher on
        # this machine": the row that would refuse the filing cannot be read,
        # so the cautious answer is to refuse.
        return f"{why_none}; not filing without reading the watcher's row"
    if db is not None:
        try:
            row = db.get_video(video_id)
        except Exception as e:
            return f"the watcher database could not be read ({e})"
        if row is not None:
            state = row.get("state")
            if state != "processed":
                return (f"the watcher has this video as '{state}', not 'processed' "
                        f"(passed every check); the watcher files it itself when "
                        f"it has passed")
            started_at = row.get("processing_started_at")
    if folder is None:
        return "no processing folder configured"
    return stale_features_reason(video_id, folder, started_at)


def get_archive_destination(video_id: str) -> Path:
    """Determine archive destination based on project/cohort.

    Args:
        video_id: Video identifier (e.g., "20250704_CNT0101_P1")

    Returns:
        Destination folder path: Analyzed/<project folder>/<cohort folder>/,
        from the lab's filing rules (config.filing_rules)
    """
    return filed_folder(video_id, Paths.ANALYZED_OUTPUT)


def filed_folder(video_id: str, analyzed_root) -> Path:
    """The folder under ``analyzed_root`` where ``video_id`` is filed.

    The one rule for "where does this video live": the archive step files
    there, and every lookup of a filed video should look there first instead
    of guessing a depth. WHY: lookups that assumed Analyzed/<project>/<cohort>
    was exactly two levels stopped finding anything once a project folder could
    be two levels itself (grant/project).
    """
    # Extract animal ID from video_id (format: DATE_ANIMALID_TRAY)
    parts = video_id.split("_")
    if len(parts) >= 2:
        animal_id = parts[1]
        project, cohort = AnimalID.get_project_and_cohort(animal_id)
    else:
        project, cohort = "UNKNOWN", "UNKNOWN"

    return Path(analyzed_root) / project / cohort


def _same_file(a: Path, b: Path) -> bool:
    """Same content: same size and modified times within 2 s (the quick check
    rsync and robocopy use -- every copy in this pipeline keeps the modified
    time), else, at the same size, the same SHA-256. WHY not always hash:
    hashing reads both copies in full over the share."""
    from mousereach.watcher.transfer import verify_file_hash
    try:
        sa, sb = a.stat(), b.stat()
    except OSError:
        return False
    if sa.st_size != sb.st_size:
        return False
    if abs(sa.st_mtime - sb.st_mtime) < 2:
        return True
    return verify_file_hash(a, b, "sha256")


# The attention list a filing refused over a person's cuts goes on
# (<NAS root>/attention/<host>_filing_held_persons_cuts.json).
HELD_PERSONS_CUTS_KIND = "filing_held_persons_cuts"


def _refuse_persons_cuts(result: Dict, video_id: str, dest: Path,
                         kept: List[str], verbose: bool) -> Dict:
    """Refuse to file ``video_id`` because ``dest`` holds a person's cuts this
    run's algorithm cuts would replace. Nothing has moved when this is called.
    Sets ``result["error"]`` and ``result["held_for_person"]``, and lists the
    video in the attention panel (watcher/attention.py) so a person sees it
    without reading a log; filing it later takes it off the list.

    WHY a person: which cuts stand is a judgment about the video. The filed
    cuts are a person's answer, and an answer a person gave is never replaced
    by the algorithm's; but this run's later steps were computed on the
    algorithm's cuts, so its results cannot be filed beside the person's
    either. Retrying cannot change that, so every retry refuses the same way."""
    import socket
    names = ", ".join(kept)
    says = (f"{video_id}: filing stopped before anything moved. Its filed folder "
            f"{dest} holds a person's cuts ({names}), and this run's segments "
            f"file is the algorithm's, so filing would replace the person's cuts. "
            f"The filed results are unchanged and this run's files stay in "
            f"Processing; filing is tried again and refused again until a person "
            f"decides which cuts are right. No button resolves this yet: tell "
            f"whoever maintains the pipeline.")
    result["error"] = (f"refusing to archive: the filed folder of {video_id} holds a "
                       f"person's cuts ({names}) and this run's segments file is not "
                       f"a person's; it would replace them (nothing moved)")
    result["held_for_person"] = {"kind": HELD_PERSONS_CUTS_KIND, "says": says,
                                 "persons_cuts": list(kept)}
    try:
        from mousereach.watcher.attention import note_video
        note_video(getattr(Paths, "NAS_ROOT", None), socket.gethostname(),
                   HELD_PERSONS_CUTS_KIND, video_id, says)
    except Exception:
        pass   # note_video never raises; the error string still says it
    if verbose:
        print(f"  {result['error']}")
    return result


def archive_video(
    video_id: str,
    dry_run: bool = False,
    verbose: bool = True,
    skip_ready_check: bool = False,
    source_dir: Path = None,
) -> Dict:
    """Archive a video to NAS.

    Moves all files from Processing/ to NAS archive, organized by experiment.

    Args:
        video_id: Video identifier
        dry_run: If True, only show what would be done
        verbose: Print progress
        skip_ready_check: If True, skip validation status check
        source_dir: If set, discover files from this directory instead of Processing/

    Returns:
        Dict with archive results
    """
    from mousereach.index import PipelineIndex

    result = {
        "video_id": video_id,
        "success": False,
        "files_moved": [],
        "destination": None,
        "error": None,
    }

    # Check if ready. Only callers OUTSIDE the watcher come here (the
    # mousereach-archive command, single and bulk); the watcher passes
    # skip_ready_check because its own filing step has made every check, and a
    # retry after a part-way move must not be refused for files already filed.
    if not skip_ready_check:
        is_ready, status = check_archive_ready(video_id, source_dir=source_dir)

        if not is_ready:
            not_validated = [k for k, v in status.items() if v not in ("validated", "auto_approved")]
            result["error"] = f"Not ready: {', '.join(not_validated)} not validated"
            if verbose:
                print(f"Cannot archive {video_id}: {result['error']}")
            return result
        why = command_filing_refusal(video_id, source_dir=source_dir)
        if why:
            result["error"] = f"Not filed: {why}"
            if verbose:
                print(f"Cannot archive {video_id}: {why}")
            return result

    # Get files
    if source_dir:
        from mousereach.pipeline.fsutil import files_of_video
        files = files_of_video(source_dir, video_id)
    else:
        files = get_video_files(video_id)
    if not files:
        result["error"] = "No files found in Processing/"
        if verbose:
            print(f"Cannot archive {video_id}: No files found")
        return result

    # Determine destination
    dest = get_archive_destination(video_id)
    result["destination"] = str(dest)

    # A publication lock record in the destination (pipeline/
    # publication_lock.py) means the filed results there must stay exactly as
    # they are: nothing is moved, by the command or by the watcher. A record
    # that cannot be checked counts as a lock.
    from mousereach.pipeline.publication_lock import lock_refusal
    locked = lock_refusal(video_id, "filed", folder=dest)
    if locked:
        result["error"] = f"Not filed: {locked}"
        if verbose:
            print(f"Cannot archive {video_id}: {locked}")
        return result

    if verbose:
        print(f"Archive {video_id}:")
        print(f"  Source: {Paths.PROCESSING}")
        print(f"  Destination: {dest}")
        print(f"  Files: {len(files)}")

    if dry_run:
        if verbose:
            print("  [DRY RUN - no files moved]")
            for f in files:
                print(f"    Would move: {f.name}")
        result["success"] = True
        result["files_moved"] = [f.name for f in files]
        result["dry_run"] = True
        return result

    # Create destination if needed
    dest.mkdir(parents=True, exist_ok=True)

    # FILING IS ONE STEP. Re-filing a filed video replaces a whole generation
    # of results (segments, reaches, outcomes, assignments, manifest,
    # kinematics), and anything that reads the filed folder -- the database
    # integrator's nightly import above all, which takes the features file --
    # must see either the previous generation, complete, or the new one,
    # complete; never a folder of some new and some old files. It used to set
    # the old generation aside first and then move the new files in one by
    # one, so one file that would not move (a handle still open on the share)
    # left the folder split until the retry, up to an hour later. Now:
    #
    #   1. STAGE: every incoming file that differs from the filed one is
    #      copied into the filed folder under a temporary name
    #      (".<name>.<random>.tmp", fsutil.stage_copy), which no reader and no
    #      mover takes for a file of the video. Nothing visible has changed;
    #      a failure here removes the copies and stops.
    #   2. KEEP: the previous generation's files about to be replaced are
    #      copied, checksum-verified, into the versioned Analyzed/Archive
    #      (supersede_video_outputs, keep_source). They stay in the folder.
    #      A failure stops here, nothing visible changed.
    #   3. SWITCH: renames inside the one folder. The old features file and
    #      manifest are taken out of sight first, then each new file takes its
    #      place, the manifest second to last and the features file last.
    #      A rename that fails is undone by renaming back, so the complete
    #      previous generation stands again. Renames in one folder take
    #      milliseconds, and during them the folder holds no features file,
    #      so the database integrator skips the video rather than read a mix.
    #   4. CLEAN UP: only now are the replaced files dropped (their copy is
    #      in the Archive) and the files in Processing removed. A file that
    #      cannot be removed from Processing makes the attempt report failure;
    #      the retry finds everything already filed (identical) and finishes.
    #
    # The one case this cannot cover is the process being killed outright
    # during the renames of step 3. Then the folder may hold part of the new
    # generation and no features file (so it is not imported), the previous
    # generation is complete in the Archive from step 2, every file is still
    # in Processing, and the retry after restart files the rest.
    #
    # A file identical to the one already filed is not a new generation -- a
    # re-run copies the video and pose down from here and hands them back
    # unchanged. Such files are left in place and the local copy dropped,
    # except the features file and the manifest: when anything else changes
    # they are always swapped too, so step 3 can keep the old features file
    # out of sight while the others change.
    from mousereach.pipeline.fsutil import (
        files_of_video as _fov, stage_copy, aside_path, rename_with_retry,
        remove_leftover_temps, remove_temp_file, retry_transient)
    last_names = (f"{video_id}_processing_manifest.json",
                  f"{video_id}_features.json")
    identical = {f.name for f in files
                 if (dest / f.name).exists() and _same_file(f, dest / f.name)}
    if any(f.name not in identical for f in files):
        identical -= set(last_names)
    changing = [f for f in files if f.name not in identical]
    incoming = {f.name: f for f in changing}
    refiling = bool(changing) and bool(_fov(dest, video_id))

    if refiling:
        # A person's cuts in the filed folder are never overwritten by an
        # algorithm's (see archive/supersede.py). Asked with a DRY RUN before
        # anything is copied or renamed. ``only``: just the files this
        # filing replaces -- an unscoped sweep would treat files a part-way
        # earlier attempt already filed as a previous generation.
        try:
            from mousereach.archive.supersede import supersede_video_outputs
            pre = supersede_video_outputs(video_id, dest, only=set(incoming),
                                          incoming=incoming, dry_run=True)
        except Exception as e:
            result["error"] = f"refusing to archive: superseding failed ({e})"
            if verbose:
                print(f"  {result['error']}")
            return result
        if pre.get("human_cuts_kept"):
            return _refuse_persons_cuts(result, video_id, dest,
                                        pre["human_cuts_kept"], verbose)

    def _discard(paths):
        for p in paths:
            remove_temp_file(p)

    # 1. STAGE. Temporary files an interrupted earlier attempt left are
    # removed first (one filing of a video runs at a time: the watcher's
    # archive step, or the command, which refuses while the watcher runs).
    for f in files:
        remove_leftover_temps(dest / f.name)
    staged = {}
    for f in changing:
        try:
            staged[f.name] = stage_copy(f, dest / f.name)
        except Exception as e:
            _discard(staged.values())
            result["error"] = (f"failed to move: could not copy {f.name} into the "
                               f"filed folder ({e}); nothing in {dest} changed and "
                               f"every file stays in Processing")
            if verbose:
                print(f"  {result['error']}")
            return result

    # 2. KEEP the previous generation, as a verified copy in the Archive.
    superseded = None
    if refiling:
        try:
            superseded = supersede_video_outputs(
                video_id, dest, only=set(incoming), incoming=incoming,
                keep_source=True)
        except Exception as e:
            superseded = {"error": str(e)}
        if superseded.get("human_cuts_kept"):
            # Backstop: the dry run said no, so this cannot fire unless the
            # folder changed in between; refused rather than overwritten.
            _discard(staged.values())
            return _refuse_persons_cuts(result, video_id, dest,
                                        superseded["human_cuts_kept"], verbose)
        why = superseded.get("error") or ", ".join(superseded.get("failed") or [])
        if why:
            _discard(staged.values())
            result["error"] = ("refusing to archive: could not preserve the previous "
                               "generation of %s (%s); nothing in the filed folder "
                               "changed" % (video_id, why))
            if verbose:
                print(f"  {result['error']}")
            return result
        n = len(superseded.get("algo", [])) + len(superseded.get("pose", []))
        if n and verbose:
            print(f"  Kept a copy of {n} earlier file(s) -> "
                  f"{superseded.get('generation')}/{superseded.get('stack')}")
    result["superseded"] = superseded

    # 3. SWITCH. done: (path, aside) pairs; aside None = a new file now at
    # path, else the file that was at path now sits at aside.
    done = []
    order = ([n for n in incoming if n not in last_names]
             + [n for n in last_names if n in incoming])
    try:
        for n in last_names:
            if n in staged and (dest / n).exists():
                aside = aside_path(dest / n)
                rename_with_retry(dest / n, aside)
                done.append((dest / n, aside))
        for n in order:
            final = dest / n
            if final.exists():
                aside = aside_path(final)
                rename_with_retry(final, aside)
                done.append((final, aside))
            rename_with_retry(staged[n], final)
            done.append((final, None))
    except Exception as e:
        undone = True
        for path, aside in reversed(done):
            try:
                if aside is None:
                    # The new file's source is still in Processing.
                    retry_transient(path.unlink)
                else:
                    rename_with_retry(aside, path)
            except Exception as e2:
                undone = False
                logging.getLogger(__name__).error(
                    "%s: could not put %s back after a failed filing (%s); a copy "
                    "of the previous generation is in the Archive",
                    video_id, path.name, e2)
        _discard(p for p in staged.values() if p.exists())
        result["error"] = (f"failed to move: could not put {video_id}'s files in "
                           f"place in {dest} ({e}); "
                           + ("the previous generation was put back as it was"
                              if undone else
                              "the previous generation could not all be put back, "
                              "but a copy of it is in the Archive")
                           + " and every file stays in Processing")
        if verbose:
            print(f"  {result['error']}")
        return result

    # 4. CLEAN UP. The replaced files have their verified copy in the Archive
    # (step 2); a file supersede leaves alone by rule (the video, a person's
    # review, ground truth) or does not know was replaced by the move-based
    # filing too.
    for path, aside in done:
        if aside is not None and not remove_temp_file(aside):
            logging.getLogger(__name__).warning(
                "%s: could not remove the replaced %s (left as %s)",
                video_id, path.name, aside.name)
    if verbose:
        for f in files:
            print(f"    {'Already archived (identical)' if f.name in identical else 'Filed'}: "
                  f"{f.name}")

    # The local copies go: features first, then the manifest, then the rest,
    # so a retry that still finds the features file in Processing also finds
    # the manifest beside it (the filing step's provenance check).
    moved = []
    failed = []
    by_name = {f.name: f for f in files}
    drop_order = ([n for n in reversed(last_names) if n in by_name]
                  + [n for n in by_name if n not in last_names])
    for n in drop_order:
        try:
            retry_transient(by_name[n].unlink)
            moved.append(n)
        except FileNotFoundError:
            moved.append(n)
        except Exception as e:
            failed.append(f"{n} (filed, but its copy in Processing could not be "
                          f"removed: {e})")

    result["files_moved"] = moved
    result["success"] = len(moved) == len(files)
    if failed and not result["success"]:
        result["error"] = "failed to move: " + "; ".join(failed)

    # Update index - remove video
    if result["success"]:
        # Filed: an earlier refusal over a person's cuts is resolved.
        try:
            import socket
            from mousereach.watcher.attention import drop_video
            drop_video(getattr(Paths, "NAS_ROOT", None), socket.gethostname(),
                       HELD_PERSONS_CUTS_KIND, video_id)
        except Exception:
            pass
        try:
            index = PipelineIndex()
            index.load()
            index.remove_video(video_id)
            index.save()
            if verbose:
                print("  Index updated")
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to update index: {e}")

    if verbose:
        if result["success"]:
            print(f"  SUCCESS: {len(moved)} files archived")
        else:
            print(f"  PARTIAL: {len(moved)}/{len(files)} files archived")

    return result


def record_archived(video_id: str) -> bool:
    """Tell this node's watcher database that the video has been filed.

    archive_video MOVES the files. Without this the row still says 'processed',
    so the watcher picks the video up again, finds nothing left in Processing,
    and reads 'not ready' forever -- a video correctly filed and permanently
    recorded as unfiled, invisible to the version checker that only looks at
    'archived'. Ten videos went that way in 45 seconds when filing first started
    working; running the bulk command would have done it to the whole backlog.

    'processed' cannot go straight to 'archived'; the state machine routes
    through 'archiving'. Best-effort and never raises: the files are already
    where they belong, and a bookkeeping failure must not look like a data
    failure. Returns True if the state was recorded.
    """
    try:
        from mousereach.watcher.db_location import open_watcher_db_if_present
        # The node's db_path override, not the default. Getting this wrong is
        # how commands can spend months reading a stale database while the
        # daemon writes to a different file; writing to the wrong one would be
        # worse than reading it. And never CREATE it: a missing database means
        # a wrong path or a machine whose watcher never ran, and recording the
        # filing in a new empty file would hide that.
        db, why = open_watcher_db_if_present()
        if db is None:
            __import__("logging").getLogger(__name__).warning(
                "%s: archived on disk, but its state was not recorded: %s. "
                "Run mousereach-watch-status on the machine that analysed it.",
                video_id, why)
            return False
        row = db.get_video(video_id)
        if row is None:
            return False                      # not this node's video
        if row["state"] == "archived":
            return True
        if row["state"] == "processed":
            db.update_state(video_id, "archiving")
        db.update_state(video_id, "archived")
        return True
    except Exception as e:
        logger = __import__("logging").getLogger(__name__)
        logger.warning("%s: archived on disk but the state could not be "
                       "recorded (%s). Run mousereach-watch-status to check.",
                       video_id, e)
        return False


def _row_known_here(video_id: str) -> bool:
    """Whether this machine's watcher database has a row for the video."""
    try:
        from mousereach.watcher.db_location import open_watcher_db_if_present
        db, _ = open_watcher_db_if_present()
        return db is not None and db.get_video(video_id) is not None
    except Exception:
        return False


def archive_all(
    dry_run: bool = False,
    verbose: bool = True
) -> Dict:
    """Archive all videos that are ready.

    Args:
        dry_run: If True, only show what would be done
        verbose: Print progress

    Returns:
        Summary dict with counts and per-video results
    """
    archivable = get_archivable_videos()

    if verbose:
        print(f"Found {len(archivable)} video(s) ready for archive")
        if dry_run:
            print("[DRY RUN MODE]")
        print("-" * 60)

    results = {
        "total": len(archivable),
        "success": 0,
        "failed": 0,
        "videos": [],
        "archived_at": datetime.now().isoformat(),
    }

    for video_id in archivable:
        result = archive_video(video_id, dry_run=dry_run, verbose=verbose)
        if result.get("success") and not dry_run:
            known = _row_known_here(video_id)
            if not record_archived(video_id) and known:
                # Filed on disk, and a row that exists could not be moved to
                # 'archived': counted as a failure so the command exits 1.
                result["success"] = False
                result["error"] = ("filed, but the watcher database could not "
                                   "record it; run mousereach-watch-status")
        results["videos"].append(result)
        if result["success"]:
            results["success"] += 1
        else:
            results["failed"] += 1

    if verbose:
        print("-" * 60)
        print(f"Archived: {results['success']}/{results['total']}")

    return results
