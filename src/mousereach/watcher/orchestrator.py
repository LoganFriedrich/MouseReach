"""
Role-aware orchestrators for the MouseReach watcher.

Two machines, two roles:

  DLCOrchestrator (mode="dlc_pc")
      NAS/DLC PC with GPU.  Scans NAS for collages -> crop -> DLC -> stage
      video+h5 to NAS Processing/Posed/ folder for the processing server.

  ProcessingOrchestrator (mode="processing_server")
      Processing server.  Watches Processing/Posed/ -> intake to local Processing/
      -> segmentation -> reach detection -> outcome detection -> archive to NAS.

Both share a BaseOrchestrator that provides the polling loop, priority animal
system, DB tracking, and graceful shutdown.
"""

import json
import os
import re
import time
import random
import socket
import logging
import threading

# Ensure CUDA paths, cuDNN, and TF_USE_LEGACY_KERAS are set for GPU use
from mousereach.gpu import setup_gpu_env
setup_gpu_env()
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

from mousereach.watcher.db import WatcherDB
from mousereach.watcher.state import WatcherStateManager
from mousereach.watcher.watcher import FileWatcher
from mousereach.watcher.router import TrayRouter
from mousereach.watcher.transfer import safe_copy, safe_move
from mousereach.watcher import single_claim
from mousereach.watcher.locate import (
    resolve_pose_input, locate_pose_file, locate_video_file, node_search_dirs,
)
from mousereach.pipeline.manifest import select_pose_file
from mousereach.watcher.work_priority import load_lab_policy, LAB_PRIORITY_HINT
from mousereach.config import (
    Paths, WatcherConfig, require_processing_root, parse_tray_type,
    get_video_id, AnimalID
)

logger = logging.getLogger(__name__)


def segmentation_could_not_run(error, dlc_path) -> bool:
    """True when segmentation never executed, as opposed to executing and failing.

    These are different things and must not share a destination. The deep-review
    queue means "an algorithm's judgment about this animal's behaviour needs a
    human eye" -- a scientific queue. An unreadable input is not a judgment.
    Routing it there spends the lab's scarcest resource on a file-handle error
    and, worse, buries the real work: on 2026-08-19, 723 such bundles hid the 21
    genuine review items sitting in the same folder.

    A real seg failure (over-long recording, uniform-fallback boundaries) still
    goes to deep review, because a human genuinely has to re-segment it.

    Decided on a fact first -- can the input actually be read? -- and only then
    on the error text, so this does not rest on matching library messages.
    """
    try:
        if dlc_path is None or not Path(dlc_path).is_file():
            return True
        with open(dlc_path, 'rb') as fh:
            fh.read(1)
    except OSError:
        return True
    return bool(re.search(r"Errno \d+|Permission denied|No such file|cannot access|is a directory",
                          str(error), re.IGNORECASE))


class _PoseAborted(Exception):
    """A pose was stopped part-way on purpose (a recording program started).

    Its own type so the DLC handler can tell it apart from a failed pose: an
    aborted pose goes back to the queue and costs the video nothing, a failed
    one is marked failed and spends a retry."""


class _CropAborted(Exception):
    """A crop was stopped part-way on purpose (a recording program started).

    Same idea as _PoseAborted, for the other long step: the half-made single
    videos are deleted and the collage goes back to waiting, without being
    marked failed. Cropping again costs minutes of an idle machine; a recording
    that drops frames cannot be filmed again."""


# =============================================================================
# BASE ORCHESTRATOR
# =============================================================================

def _reprocess_copy_set(source_dirs, pose_dir, video_id):
    """The files a reprocess stages locally, with the pose tree contributing
    ONLY pose artifacts.

    WHY: the pose tree can hold scattered analysis jsons from runs whose pose
    path pointed into it (pre-b742b58), and set-ordered copies let such an
    ALGO segments.json clobber the results dir's HUMAN one -- which
    re-segmented away a reviewer's hand-set cuts (2026-09-08). Results jsons
    come from the results dir alone; the filter is skipped when the pose dir
    is the ONLY source (then its jsons are all we have).
    """
    out = []
    dirs = [Path(d) for d in source_dirs]
    filter_pose = len(dirs) > 1
    for d in dirs:
        for f in d.iterdir():
            if not (f.is_file() and f.stem.startswith(video_id)):
                continue
            if filter_pose and d == Path(pose_dir) and "DLC" not in f.name:
                continue
            out.append(f)
    return out


class BaseOrchestrator:
    """
    Shared infrastructure for all watcher orchestrator roles.

    Provides:
    - Polling loop (run, run_once)
    - Priority animal system
    - DB tracking and logging
    - Graceful shutdown
    """

    # Whether this role can actually re-run an archived video. Only the
    # processing role has a handler for the 'outdated' state; the DLC role's work
    # loop never selects it. Marking videos outdated on a node that cannot act on
    # them is not harmless bookkeeping -- the state syncs to connectome.db, other
    # nodes adopt it in recovery, and the video ends up queued on a machine that
    # has no file for it. So the scan is gated on being able to finish the job.
    handles_reprocessing = False

    # How long to wait before trying to archive a video again, per consecutive
    # failure. Caps at an hour: long enough that a stuck video costs nothing,
    # short enough that fixing whatever blocked it takes effect the same session.
    _ARCHIVE_BACKOFF_SECONDS = (60, 300, 900, 3600)

    def _note_archive_failure(self, video_id: str) -> int:
        """Record a failed archive attempt and return the delay before the next."""
        if not hasattr(self, "_archive_backoff"):
            self._archive_backoff = {}
        n, _ = self._archive_backoff.get(video_id, (0, 0.0))
        n += 1
        delay = self._ARCHIVE_BACKOFF_SECONDS[
            min(n - 1, len(self._ARCHIVE_BACKOFF_SECONDS) - 1)]
        self._archive_backoff[video_id] = (n, time.time() + delay)
        return delay

    def _clear_archive_backoff(self, video_id: str) -> None:
        """Forget a video's failure history once it archives."""
        getattr(self, "_archive_backoff", {}).pop(video_id, None)

    def _already_failed(self, video_id: str) -> bool:
        """True when this row is already recorded failed.

        A handler that marks a video failed and then re-raises would have it
        marked a SECOND time by the dispatch guard, and mark_failed increments
        error_count -- so two real attempts spent a budget of three, and the
        re-pose consumer held a request as 'failed too often' one attempt
        early. The retry budget has to count attempts, not calls.
        """
        try:
            row = self.db.get_video(video_id)
        except Exception:
            return False
        return bool(row) and row.get("state") == "failed"

    def _archive_backoff_active(self, video_id: str) -> bool:
        """True while this video is waiting out a previous archive failure."""
        if not hasattr(self, "_archive_backoff"):
            self._archive_backoff = {}
            return False
        entry = self._archive_backoff.get(video_id)
        return bool(entry) and time.time() < entry[1]


    def __init__(self, config: WatcherConfig, db: WatcherDB):
        self.config = config
        self.db = db
        self.router = TrayRouter()
        self.hostname = socket.gethostname()

        # Create state manager and file watcher
        self.state = WatcherStateManager(db, config)
        self.file_watcher = FileWatcher(config, self.state)

        # Working directory for temporary operations
        self.working_dir = require_processing_root() / "watcher_working"
        self.working_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"{self.__class__.__name__} initialized on {self.hostname}")
        logger.info(f"Working directory: {self.working_dir}")

        # One guard for the life of the watcher, so its grace timer carries
        # across polls (watcher/recording_guard.py). Said once at startup, so
        # a person reading the log knows why this node may sit idle.
        guard = self._get_recording_guard()
        if guard.enabled:
            # _ascii: the names are typed by a person, and one non-ASCII
            # character would crash this line on a Windows console.
            from mousereach.watcher.recording_guard import _ascii
            logger.info(
                "Pauses while any of these programs run: %s (work resumes %d s "
                "after the last one closes).",
                _ascii(", ".join(guard.names)), guard.grace_seconds)

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def _get_recording_guard(self):
        """This watcher's RecordingGuard, built from its config on first use.

        Built lazily as well as in __init__ because tests (and a few tools)
        make an orchestrator without its constructor; with no configured
        programs the guard never pauses, which is today's behaviour exactly.
        """
        guard = getattr(self, '_recording_guard', None)
        if guard is None:
            from mousereach.watcher.recording_guard import RecordingGuard
            config = getattr(self, 'config', None)
            guard = self._recording_guard = RecordingGuard(
                getattr(config, 'pause_while_running', None) or [],
                grace_seconds=getattr(config, 'pause_resume_grace_seconds', 120))
        return guard

    def _is_paused(self) -> bool:
        """True while this watcher must not start new work.

        Two causes, one answer (recording_guard.pause_reason): a person
        paused it by hand (watcher_paused.flag), or a configured recording
        program is running or closed too recently. WHY recording pauses the
        watcher: dropped frames in a recording cannot be filmed again, and a
        pose can always be run later.

        The reason is kept on self._pause_reason for status and logging.
        Exactly ONE log line on entering a pause and ONE on leaving it, never
        one per poll -- a line every 30 seconds for an eight-hour recording
        day buries everything else in the log.
        """
        from mousereach.watcher.recording_guard import pause_reason, _ascii
        self._refresh_recording_settings()
        guard = self._get_recording_guard()
        try:
            root = require_processing_root()
        except Exception:
            root = None
        try:
            reason = pause_reason(root, guard=guard)
        except Exception as e:
            # pause_reason is written not to raise. If it does anyway, fail
            # SAFE only when recording programs are configured (recording
            # must win); otherwise keep the old behaviour, which was to work.
            reason = (_ascii(f"cannot check whether to pause: {type(e).__name__}: {e}")
                      if guard.enabled else None)

        previous = getattr(self, '_pause_reason', None)
        self._pause_reason = reason
        kind, previous_kind = self._pause_kind(reason), self._pause_kind(previous)
        # On-screen messages for the person at a recording machine. This runs
        # between work items, so reaching here paused for a recording means
        # nothing of ours is running any more -- which is exactly what "safe to
        # record" claims. Each is said once per recording episode.
        try:
            if kind == 'recording':
                self._record_notices().safe_to_record()
            elif not reason:
                self._record_notices().back_to_work()
        except Exception as e:
            logger.debug(f"could not update the recording messages: {e}")
        if reason and not previous:
            logger.info("Watcher PAUSED: %s. No new work starts; %s.",
                        reason, self._pause_hint(kind))
        elif reason and kind != previous_kind:
            # Still paused, for a different reason. WHY log it: the last line
            # would otherwise go on telling a person to press Resume after
            # they did, while an open recording program is what holds it now.
            logger.info("Watcher still PAUSED, now because: %s. %s.",
                        reason, self._pause_hint(kind).capitalize())
        elif previous and not reason:
            logger.info("Watcher RESUMED: no longer paused (was: %s).", previous)
        return bool(reason)

    @staticmethod
    def _pause_kind(reason: Optional[str]) -> Optional[str]:
        """'hand', 'unchecked' or 'recording' (a program open, or closed too
        recently). Only a change of KIND is logged; the countdown text in a
        grace-period reason changes every poll and must not be."""
        from mousereach.watcher.recording_guard import HAND_PAUSE_REASON
        if not reason:
            return None
        if reason == HAND_PAUSE_REASON:
            return 'hand'
        if reason.startswith('cannot check'):
            return 'unchecked'
        return 'recording'

    @staticmethod
    def _pause_hint(kind: Optional[str]) -> str:
        if kind == 'hand':
            return "run 'mousereach-watch-toggle --resume' or press Resume to continue"
        if kind == 'unchecked':
            return ("it stays paused until the check works, because recording "
                    "must win")
        return "work resumes on its own once the recording program has been closed"

    def _refresh_recording_settings(self) -> None:
        """Follow edits to the two recording settings without a restart.

        WHY: the Watcher Control panel, mousereach-watch-recorders and the
        status commands all read the settings FILE. A watcher that kept the
        list it started with would pose on top of a recording while every
        screen said PAUSED (a program added after start), or sit paused while
        every screen said it was working (a program removed). So when the file
        this watcher's config came from has changed, the two keys are read
        again and the guard follows them. Only these two: every other setting
        still applies at the next start, as the panel says.

        Cheap: one stat of a local file. A file that cannot be read or parsed
        right now (for example caught mid-save) changes nothing and is tried
        again on the next check. Never raises.
        """
        config = getattr(self, 'config', None)
        source = getattr(config, 'source_file', None)
        if not source:
            return
        try:
            mtime = Path(source).stat().st_mtime
        except OSError:
            return
        seen = getattr(self, '_settings_mtime', None)
        if seen is None:
            seen = getattr(config, 'source_mtime', None)
        if mtime == seen:
            return
        from mousereach.watcher.recording_guard import _ascii, _match_key, clean_names
        try:
            data = json.loads(Path(source).read_text(encoding='utf-8-sig') or '{}')
            section = data.get('watcher') or {}
            if not isinstance(section, dict):
                raise ValueError("the 'watcher' section is not an object")
            fresh = WatcherConfig(section)
        except Exception as e:
            if getattr(self, '_settings_bad_mtime', None) != mtime:
                self._settings_bad_mtime = mtime
                logger.warning(_ascii(
                    f"Could not re-read the recording-program settings from {source} "
                    f"({e}); keeping the ones in use."))
            return
        self._settings_mtime = mtime

        names = list(fresh.pause_while_running)
        grace = fresh.pause_resume_grace_seconds
        old_names = list(getattr(config, 'pause_while_running', None) or [])
        old_grace = getattr(config, 'pause_resume_grace_seconds', 120)
        if names == old_names and grace == old_grace:
            return
        config.pause_while_running = names
        config.pause_resume_grace_seconds = grace

        old_guard = getattr(self, '_recording_guard', None)
        same_programs = (old_guard is not None and
                         [_match_key(n) for n in old_guard.names] ==
                         [_match_key(n) for n in clean_names(names)])
        if same_programs:
            # Only the grace changed: keep the guard, and with it the timer.
            old_guard.grace_seconds = max(0, int(grace))
        else:
            self._recording_guard = None
            self._get_recording_guard().inherit_history(old_guard)
        if names:
            logger.info(_ascii(
                f"Recording-program settings changed: now pauses while any of "
                f"these run: {', '.join(names)} (work resumes {grace} s after the "
                f"last one closes)."))
        else:
            logger.info("Recording-program settings changed: no programs listed; "
                        "this watcher no longer pauses for one.")

    def _recording_abort_reason(self) -> Optional[str]:
        """Why a pose running now must stop, or None: the recording guard as
        it stands this moment, following any edit to the program list.

        WHY not the hand pause too: pressing Pause means "start nothing new",
        and a pose already half done is kept. Only an open recording program
        stops one part-way, because only a recording cannot be redone."""
        self._refresh_recording_settings()
        reason = self._get_recording_guard().reason()
        if reason:
            # Said on this machine's screen, once, while the work is being
            # stopped: the operator is standing there waiting for the GPU.
            self._record_notices().stopping()
        return reason

    def _record_notices(self):
        """This node's on-screen messages about recording (watcher/record_notice.py).

        Built on first use from the config, and only ever says anything when
        recording programs are configured -- a node that never records has
        nobody standing at it waiting for the GPU."""
        notices = getattr(self, '_notices', None)
        if notices is None:
            from mousereach.watcher.record_notice import RecordNotices
            config = getattr(self, 'config', None)
            enabled = (bool(getattr(config, 'pause_while_running', None))
                       and bool(getattr(config, 'notify_safe_to_record', True)))
            notices = self._notices = RecordNotices(enabled=enabled)
        return notices

    def _while_paused(self) -> None:
        """Upkeep that must continue while the watcher is paused. Nothing on
        the base role; the GPU role keeps its re-pose claims alive here."""
        return None

    def _stop_requested(self) -> bool:
        """True when something has asked this watcher to finish and exit.

        A file rather than a signal, because the only universally deliverable
        stop on Windows is a hard kill. psutil's terminate() is an alias for
        kill() there, and a watcher started detached (as the Restart button
        starts it) has no console, so a Ctrl+C event cannot reach it either.
        Killing a watcher mid-pose throws away a 14-minute GPU run and used to
        strand the video outright.

        The loop checks this between work items, so the item in flight always
        finishes first.
        """
        try:
            return (require_processing_root() / "watcher_stop.flag").exists()
        except Exception:
            return False

    # --- States a node sets WHILE it is working on something -----------------
    # Nothing selects these as work. If the process dies here the row stops
    # moving and no later run ever picks it up: the video is stranded, and
    # stranded silently, because the queue looks busy rather than broken.
    #
    # Reclaiming them is unambiguous rather than a guess about a live run: a
    # named global mutex (watcher/cli.py) means exactly one watcher runs per
    # machine and a second one exits, so anything still sitting in one of
    # these AT STARTUP was left behind by a process that is already gone.
    #
    # 'processing' is deliberately absent. Both roles already select it, so an
    # interrupted pipeline run is picked up again on its own -- and picked up
    # where it left off, because the pipeline reuses the stage outputs that
    # already exist rather than redoing them.
    _ORPHANED_VIDEO_STATES = {
        # left behind in -> put back to
        'archiving': 'processed',        # re-files the finished video
    }
    _ORPHANED_COLLAGE_STATES: dict = {}

    def _reclaim_orphaned_work(self) -> dict:
        """Put anything a dead process left mid-flight back in the queue.

        Runs once at startup, before this node takes on anything new. Never
        raises: a node must still start even if reclaiming fails.
        """
        from mousereach.watcher.db import VIDEO_TRANSITIONS, COLLAGE_TRANSITIONS
        reclaimed = {}
        why = ("left mid-run by a watcher that stopped before it finished; "
               "returned to the queue at startup")

        for state, back_to in self._ORPHANED_VIDEO_STATES.items():
            try:
                rows = self.db.get_videos_in_state(state)
            except Exception as e:
                logger.debug(f"could not list '{state}' videos to reclaim: {e}")
                continue
            for row in rows:
                video_id = row['video_id']
                try:
                    if back_to in VIDEO_TRANSITIONS.get(state, []):
                        self.db.update_state(video_id, back_to)
                    else:
                        self.db.force_state(video_id, back_to, reason=why)
                    self.db.log_step(video_id, 'reclaim', 'completed',
                                     message=f"{state} -> {back_to}: {why}")
                    reclaimed[state] = reclaimed.get(state, 0) + 1
                    logger.warning("Reclaimed %s: was left in '%s', back to '%s'",
                                   video_id, state, back_to)
                except Exception as e:
                    logger.error(f"Could not reclaim {video_id} from '{state}': {e}")

        for state, back_to in self._ORPHANED_COLLAGE_STATES.items():
            try:
                rows = self.db.get_collages_in_state(state)
            except Exception as e:
                logger.debug(f"could not list '{state}' collages to reclaim: {e}")
                continue
            for row in rows:
                name = row.get('filename') or row.get('collage_id')
                try:
                    if back_to in COLLAGE_TRANSITIONS.get(state, []):
                        self.db.update_collage_state(name, back_to)
                    else:
                        # No 'reason' parameter on this one, unlike the video
                        # equivalent: anything extra is written as a column.
                        self.db.force_collage_state(name, back_to)
                    reclaimed[state] = reclaimed.get(state, 0) + 1
                    logger.warning("Reclaimed collage %s: was left in '%s', "
                                   "back to '%s'", name, state, back_to)
                except Exception as e:
                    logger.error(f"Could not reclaim collage {name}: {e}")

        if reclaimed:
            logger.warning(
                "Startup reclaim: %s. These were interrupted by a stop or a "
                "crash and would otherwise have sat untouched.",
                ", ".join(f"{n} from '{s}'" for s, n in sorted(reclaimed.items())))
        return reclaimed

    def run(self, shutdown_event: threading.Event):
        """Main orchestrator loop: scan + process in a single thread."""
        logger.info(f"{self.__class__.__name__} starting main loop")

        # Before taking anything new on, put back whatever a previous run was
        # holding when it stopped. Without this an interrupted pose or archive
        # sits in a state nothing selects, for good.
        try:
            self._reclaim_orphaned_work()
        except Exception as e:
            logger.error(f"Startup reclaim failed (continuing): {e}")

        while not shutdown_event.is_set():
            try:
                # Asked to stop. Checked here, between work items, so whatever
                # is in flight has already finished -- a pose is ~14 minutes of
                # GPU and must never be thrown away just to shut down.
                if self._stop_requested():
                    logger.info("Stop requested (watcher_stop.flag); finishing "
                                "here and exiting.")
                    break

                # Paused by hand, or a recording program is running. _is_paused
                # logs once on entering and once on leaving, not every poll.
                if self._is_paused():
                    self._while_paused()
                    shutdown_event.wait(timeout=self.config.poll_interval_seconds)
                    continue

                # Periodically pick up freshly-saved reviews / version staleness.
                self._maybe_review_reprocess_scan()

                # Phase A: Scan for new work
                self._scan_phase()

                # Phase B: Process the highest-priority work item
                work = self._get_next_work_item()

                if work is None:
                    shutdown_event.wait(timeout=self.config.poll_interval_seconds)
                    continue

                # Ask again right before starting the item. WHY: the scan above
                # can take many minutes on a shared drive, and a recording that
                # started during it must stop new work BEFORE it begins -- the
                # check at the top of the loop is already stale. The item is
                # not touched, so it stays in its state and is picked up again
                # once the pause clears.
                if self._is_paused():
                    self._while_paused()
                    shutdown_event.wait(timeout=self.config.poll_interval_seconds)
                    continue

                # Dispatch work to appropriate handler. A dispatch that
                # accomplished nothing must NOT count as progress: without
                # this sleep, one row that fails in milliseconds spins the
                # loop at full speed and every cycle-cheap assumption
                # elsewhere collapses (2026-09-04: a stuck priority-1 row +
                # a cycle-counted scan gate = 15 back-to-back 37-minute
                # scans overnight, ~6 videos processed in 12 hours).
                if self._dispatch_work(work) is False:
                    shutdown_event.wait(timeout=self.config.poll_interval_seconds)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)

        self.shutdown()
        logger.info(f"{self.__class__.__name__} stopped")

    def run_once(self):
        """Run one full cycle: scan + process all pending items, then exit."""
        logger.info(f"{self.__class__.__name__} running once")

        # The same two gates as the main loop. WHY: the GUI "Run Once" button
        # and 'mousereach-watch --once' used to ignore both, so a person could
        # start a pose on top of a recording, or on a watcher they had just
        # asked to stop, with one click.
        if self._run_once_blocked("before starting"):
            return

        self._maybe_review_reprocess_scan(force=True)
        self._scan_phase()

        processed = 0
        while True:
            work = self._get_next_work_item()
            if work is None:
                break
            # Asked before EVERY item: run-once drains everything pending,
            # which can take hours, and a recording may start part-way.
            if self._run_once_blocked(f"after {processed} item(s)"):
                break
            if self._dispatch_work(work) is not False:
                processed += 1

        logger.info(f"Run-once complete: processed {processed} items")

    def _run_once_blocked(self, when: str) -> bool:
        """True (and one log line saying why) when run_once must not start
        work: a stop was requested, or the watcher is paused."""
        if self._stop_requested():
            logger.info("Run-once stopped %s: stop requested (watcher_stop.flag).", when)
            return True
        if self._is_paused():
            logger.info("Run-once stopped %s: watcher is paused (%s). Nothing "
                        "was started.", when, getattr(self, '_pause_reason', None))
            return True
        return False

    def _maybe_review_reprocess_scan(self, force: bool = False):
        """Periodically scan this node's archived videos for freshly-saved reviews
        (and tool-version staleness) and mark them 'outdated', so a RUNNING watcher
        applies reviews automatically -- no manual 'mousereach-version-check' step.
        Runs at startup then on an interval; fully guarded so it can never break
        the main loop."""
        if not self.handles_reprocessing:
            if not getattr(self, '_logged_no_reprocess_role', False):
                self._logged_no_reprocess_role = True
                logger.info(
                    "%s does not reprocess archived videos -- skipping the version "
                    "and review scan. The processing node does this work.",
                    self.__class__.__name__)
            return

        interval = getattr(self.config, 'reprocess_scan_interval_seconds', 1800)
        if not force and (time.monotonic()
                          - getattr(self, '_last_reprocess_scan', 0.0)) < interval:
            return
        try:
            from mousereach.config import Paths
            nas_root = Paths.NAS_ROOT
            if not nas_root:
                return
            from mousereach.watcher.reprocessor import ReprocessingScanner
            # One PERSISTENT scanner instance (shared with any subclass that
            # made its own): its manifest mtime-cache is what turns the
            # steady-state scan from ~37 minutes of NAS reads into stats.
            if getattr(self, '_reprocess_scanner', None) is None:
                self._reprocess_scanner = ReprocessingScanner(self.db, nas_root)
            summary = self._reprocess_scanner.scan(mark_outdated=True,
                                                   adopt_orphans=self.adopts_orphans,
                                                   full_only=not self.reprocesses_partial)
            n_out = summary.get('outdated', 0)
            if n_out:
                logger.info(
                    f"Reprocess scan: {n_out} archived videos marked outdated "
                    f"({summary.get('review_triggered', 0)} from new reviews) "
                    f"-- watcher will reprocess them")
            # The videos that need a NEW pose cannot be re-run here without a
            # GPU; ask for one over shared storage (once per scan, so the
            # cost is one pass per half hour, not per poll).
            self._publish_repose_requests()
        except Exception as e:
            logger.warning(f"Reprocess scan skipped: {e}")
        finally:
            # Stamp AFTER the pass: a scan that runs longer than its interval
            # must still leave a full idle gap before the next one, or the
            # node scans continuously and processes nothing (2026-09-04).
            self._last_reprocess_scan = time.monotonic()

    # Whether the version scan on this node may register disk-archived videos
    # that have no row here. True for the processing role, whose ledger is
    # meant to cover the whole archive; False for a GPU node that also
    # processes, which scans only what it archived itself.
    adopts_orphans = True
    # Whether this node can drain an 'outdated' row whose scope is narrower
    # than 'full' (re-run from segmentation/reach/outcome/kinematics against
    # an existing pose). The processing role can; a GPU node that also
    # processes has no such handler yet, so its scan marks only the rows
    # that need a NEW pose -- the ones its own GPU can do something about.
    reprocesses_partial = True
    # Whether this role may read Processing/Posed (Paths.DLC_STAGING) when it
    # reasons about re-pose requests. True only for the processing role, whose
    # intake that folder is. A GPU node that also processes runs the same
    # publish pass, and must not narrow its rows or close its requests on the
    # strength of a pose that is on its way into the processing server
    # (watcher/locate.py docstring).
    reads_pose_staging = True

    def _publish_repose_requests(self) -> None:
        """Ask a GPU node, over shared storage, for a new pose for every row
        parked 'outdated' with scope 'full' (watcher/repose.py). Runs right
        after each version scan; a row whose new pose is already staged or
        archived, or whose request is already out, costs one lookup."""
        from mousereach.watcher import repose
        try:
            rows = [v for v in self.db.get_videos_in_state('outdated')
                    if (v.get('reprocess_scope') or '') == 'full']
        except Exception as e:
            logger.debug(f"could not list outdated rows for re-pose requests: {e}")
            return
        # Runs with an empty list too: the pass also withdraws requests whose
        # row has since healed and returns claims nobody heartbeats.
        latched = getattr(self, '_repose_latched', None)
        if latched is None:
            latched = self._repose_latched = set()
        # False = do not read staging at all (see reads_pose_staging).
        staging = Paths.DLC_STAGING if self.reads_pose_staging else False
        try:
            repose.publish_pending(self.db, rows, hostname=self.hostname,
                                   staging_dir=staging, latched=latched)
        except Exception as e:
            logger.warning(f"Re-pose request pass failed (non-fatal): {e}")

    # =========================================================================
    # ABSTRACT METHODS (subclasses must implement)
    # =========================================================================

    def _scan_phase(self):
        """Discover new work. Called once per cycle."""
        raise NotImplementedError

    def _get_next_work_item(self):
        """Return the next work dict, or None when there is nothing to do.

        TWO PASSES, and why:

        Some work is worth doing but must never take a machine that could be
        doing something else -- on this pipeline, the Easy and Flat tray
        sessions, where the divot is level with the scoring area, a displaced
        pellet can be retried, and a per-pellet outcome is not a meaningful
        unit. Those sessions are wanted for session-level measures, and wanted
        LAST.

        A preference INSIDE one bucket cannot express that: the buckets are a
        fixed order (finish, then take in, then run, then re-run), so a
        deferred video sitting in an earlier bucket would still beat a
        preferred one in a later bucket. So the bucket sequence runs twice.
        The first pass may not ADMIT deferred work onto this node; only if
        every bucket comes back empty -- the definition of an idle node --
        does the second pass allow it.

        ADMIT, NOT EVERYTHING. Only the buckets that bring NEW work onto this
        node are held back (see the ADMIT/DRAIN split in each role's
        _select_work_item). The buckets that finish work already sitting on
        this node's disk always run, on both passes. Two reasons, both
        observed:

          * A gated intake bucket is visited BEFORE the pipeline bucket that
            drains it, so an idle node would copy a whole cap's worth of
            deferred videos onto local disk before analysing one of them.
            With intake gated and the pipeline ungated, one deferred video is
            taken in, and the next cycle finishes it before another is taken.

          * Gating the drain buckets strands data. A deferred video already on
            local disk would never be analysed while any other work exists
            anywhere, it would hold a slot against max_local_pending forever,
            and that throttles intake of the work we actually wanted first.
            A node with a backlog is never idle, so "later" means "never".

        "Do not take on new low-priority work" is the policy. "Refuse to
        finish low-priority work already on your disk" never was.

        Cost: one extra sweep of the work queries on an otherwise idle cycle.
        Nothing is picked twice -- the first pass returned None.
        """
        work = self._select_work_item(admit_deferred=False)
        if work is None:
            work = self._select_work_item(admit_deferred=True)
        return work

    def _select_work_item(self, admit_deferred: bool = True):
        """One pass over this role's work buckets. Return a work dict or None.

        admit_deferred=False means "do not take deferred work ONTO this node
        on this pass". It applies only to the buckets that admit new work;
        the buckets that drain work already here ignore it. Subclasses
        implement this; _get_next_work_item above calls it, so no role can
        forget the second pass.
        """
        raise NotImplementedError

    def _dispatch_work(self, work: dict):
        """Route work to handler."""
        raise NotImplementedError

    def dry_run(self):
        """Scan without processing. Show what would be done."""
        raise NotImplementedError

    # =========================================================================
    # PRIORITY ANIMAL
    # =========================================================================

    def _get_priority_animal(self) -> Optional[str]:
        """
        Read priority animal from file, if set.

        The priority animal file is written by 'mousereach-watch-prioritize'
        and causes the watcher to prefer that animal's videos in all work queues.

        Returns:
            Animal ID string (e.g. "CNT0107") or None
        """
        priority_file = require_processing_root() / "priority_animal.json"
        if not priority_file.exists():
            return None
        try:
            with open(priority_file) as f:
                data = json.load(f)
            return data.get('animal_id')
        except Exception:
            return None

    def _matches_priority(self, item: dict, priority_animal: str,
                          animal_field: str) -> bool:
        """Check if a work item belongs to the priority animal."""
        animal_value = item.get(animal_field) or ''
        if ',' in animal_value:
            return priority_animal in animal_value.split(',')
        return animal_value == priority_animal

    @property
    def work_priority(self):
        """The lab's ordering policy, resolved once per process.

        The shared priority_order.json on the pipeline drive first, then this
        machine's own watcher.work_priority, then the shipped default -- one
        order for the whole lab, so the processing side and the human review
        tools cannot disagree about what matters.

        Read lazily rather than in __init__ so that an unreadable or malformed
        setting cannot stop a node from starting: the policy falls back to the
        shipped default and complains in the log. See watcher/work_priority.py
        for why this is a preference with a default and not a location with a
        hard stop.
        """
        policy = getattr(self, "_work_priority", None)
        if policy is None:
            policy = load_lab_policy(getattr(self.config, "work_priority", None))
            for complaint in policy.complaints:
                logger.warning("%s. To fix: %s", complaint, LAB_PRIORITY_HINT)
            for line in policy.describe():
                logger.info("Work priority -- %s", line)
            self._work_priority = policy
        return policy

    def _pick_from_pool(self, items: list, priority_animal: Optional[str],
                        animal_field: str, is_collage: bool = False,
                        allow_deferred: bool = True,
                        randomize: bool = True):
        """
        Pick one work item out of a bucket, or None if this pass can take none.

        THIS CAN RETURN None ON A NON-EMPTY BUCKET, which the old contract
        ("None only when items is empty") did not: a bucket holding nothing but
        deferred work yields nothing when allow_deferred is False. Every call
        site must test the result -- picking blind would crash the work loop on
        pick['video_id'] every cycle.

        Order of operations, fixed here so it is stated once instead of
        emerging from five call sites:

          1. PRIORITY ANIMAL narrowing. mousereach-watch-prioritize is a person
             saying "this animal, now". It is explicit, temporary and manual,
             and it outranks both standing policies below, the deferral
             included: if the only work for the requested animal is an Easy
             video, the operator gets that video, and a log line says why. The
             alternative is a button that silently does nothing for a rehab
             animal, which is the kind of silent failure this pipeline is not
             allowed to have.
          2. DEFERRAL, when the caller passes allow_deferred=False. Only the
             buckets that ADMIT new work onto this node do that; see the
             ADMIT/DRAIN split in each role's _select_work_item.
          3. CONFIGURED ORDER. The first tier with any candidate wins; the
             choice is made among that tier only. Items in no tier are last,
             so no configuration can make work unreachable.
          4. Tie-break: random within the tier -- several nodes share these
             pools, so random spreads them -- or the database's own order,
             created_at DESC (newest first), where the caller asks for it.

        Args:
            items: All items in this work bucket
            priority_animal: Animal ID to prefer, or None
            animal_field: DB field containing animal ID(s)
            is_collage: True if items are collages (filename-based tray check)
            allow_deferred: False to hide the policy's deferred work this pass
            randomize: False to keep the database's own order within a tier

        Returns:
            Selected item dict, or None
        """
        if not items:
            return None

        policy = self.work_priority

        # 1. Priority animal -- an explicit human override.
        forced_by_operator = False
        if priority_animal:
            preferred = [i for i in items if self._matches_priority(i, priority_animal, animal_field)]
            if preferred:
                items = preferred  # Only pick from preferred
                forced_by_operator = True

        # 2. Deferral (admitting buckets, first pass only).
        if not allow_deferred:
            if forced_by_operator:
                deferred = [i for i in items
                            if policy.is_deferred(i, is_collage=is_collage)]
                if deferred and not getattr(self, "_logged_priority_over_idle", False):
                    # Latched for the life of the process, so it cannot fire
                    # twice in one cycle even though this runs on both passes.
                    self._logged_priority_over_idle = True
                    logger.info(
                        "Priority animal %s: %d item(s) that would normally wait for "
                        "an idle node are being taken on now because the priority "
                        "animal was set by hand (%s). Clear it to restore the normal "
                        "order.",
                        priority_animal, len(deferred),
                        ", ".join(str(i.get("video_id") or i.get("filename"))
                                  for i in deferred[:5]))
            else:
                items = [i for i in items
                         if not policy.is_deferred(i, is_collage=is_collage)]
                if not items:
                    return None

        # 3. Configured order: first tier with a candidate wins.
        tiered = [(policy.tier(i, is_collage=is_collage), i) for i in items]
        best = min(tier for tier, _ in tiered)
        candidates = [i for tier, i in tiered if tier == best]

        # 4. Tie-break.
        return random.choice(candidates) if randomize else candidates[0]

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _get_associated_files(self, directory: Path, video_id: str) -> list:
        """Get all files associated with a video (mp4, h5, csv, etc.).

        A file still being written under a temporary name (staging writes
        <name>.part and renames when complete) is not one of them."""
        files = []
        if directory.exists():
            for file_path in directory.iterdir():
                if (file_path.is_file() and file_path.name.startswith(video_id)
                        and not file_path.name.endswith(('.part', '.tmp'))):
                    files.append(file_path)
        return files

    def _set_aside_stale_manifest(self, directory: Path, video_id: str, error) -> None:
        """A run could not write its processing manifest: move the one already
        on disk out of the video's file set, so it cannot pass for this run.

        WHY: a manifest left over from an earlier run (a returned review bundle
        keeps its <stem>_processing_manifest.json) would otherwise be stamped
        by record_kinematic_version and archived as the provenance of outputs
        it does not describe -- and reconcile would read the video as current.
        With no manifest, the video reads as not current and a person sees it.
        Moved into a _stale_manifests/ subfolder (never deleted): archiving
        takes only the stem's files at the top of the folder.
        """
        logger.error(f"Manifest creation failed for {video_id}: {error}")
        old = Path(directory) / f"{video_id}_processing_manifest.json"
        if not old.is_file():
            return
        try:
            aside = Path(directory) / "_stale_manifests"
            aside.mkdir(parents=True, exist_ok=True)
            dest = aside / f"{video_id}_processing_manifest.{datetime.now():%Y%m%d_%H%M%S}.json"
            old.replace(dest)
            logger.error(f"{video_id}: set the previous run's processing manifest "
                         f"aside ({dest}); this run has none, so it will read as "
                         f"not current until it is re-run")
        except OSError as e:
            logger.error(f"{video_id}: could not set the stale processing manifest "
                         f"aside ({e}); it may be archived as this run's provenance")

    def shutdown(self):
        """Graceful shutdown."""
        logger.info(f"{self.__class__.__name__} shutting down gracefully")
        try:
            if self.working_dir.exists():
                for file_path in self.working_dir.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                        logger.debug(f"Cleaned up: {file_path.name}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


# =============================================================================
# DLC ORCHESTRATOR (NAS / DLC PC)
# =============================================================================

class DLCOrchestrator(BaseOrchestrator):
    """
    NAS/DLC PC orchestrator.

    Processes one collage at a time:
    - Scans NAS for new collages
    - Crops to singles
    - Runs DLC inference on each single
    - Stages video+h5 back to NAS for the processing PC
    """

    # A GPU node scans only what it archived itself, and marks only the rows
    # that need a new pose (see BaseOrchestrator).
    adopts_orphans = False
    reprocesses_partial = False
    reads_pose_staging = False

    # How often a node whose coordination database failed tries it again.
    # WHY retry at all: a node that started while the share was still coming up
    # used to crop nothing until someone restarted it, silently, for days.
    # Minutes, not seconds: each try opens a database over the share.
    _COORDINATOR_RETRY_S = 300
    # While claims stay unavailable the WARNING repeats this often, so the
    # condition is never one line long gone from view.
    _CLAIMS_UNAVAILABLE_REWARN_S = 3600
    # A collage another host is cropping right now is asked about again after
    # this long. WHY not park it for good: the holder may fail and release, or
    # go stale, and only a node that asks again can then crop it. WHY not every
    # poll: each lost claim is a round trip to the share and a no-progress
    # dispatch. A crop takes minutes to an hour or two.
    _CLAIM_RECHECK_S = 1800

    # This role is the only one that poses and the only one that crops, so it
    # is the only one that can leave a video in 'dlc_running' or a collage in
    # 'cropping'. Reclaiming a pose is what stops a stopped watcher from
    # stranding one: 'dlc_running' -> 'dlc_queued' is a legal transition, and
    # the transition table's own note calls it "re-queue on interrupt".
    #
    # These stay OFF the base class on purpose. The processing role has no
    # GPU and never selects 'dlc_queued', so reclaiming a pose there would
    # swap one dead end for another.
    _ORPHANED_VIDEO_STATES = {
        'archiving': 'processed',
        'dlc_running': 'dlc_queued',
    }
    _ORPHANED_COLLAGE_STATES = {
        'cropping': 'stable',
    }

    def _reclaim_orphaned_work(self) -> dict:
        """Stop any pose a dead watcher left running, THEN requeue its video.

        WHY first: with recording programs configured a pose runs in a child
        process (dlc/core/interruptible.py). On Windows that child dies with
        the watcher, but where it could not be tied to it, it can outlive a
        watcher that was closed or crashed. Requeuing its video while it still
        runs would start a second pose of the same video, into the same files,
        on the same GPU. Only done when programs are configured, because only
        then can such a child exist -- and listing processes is slow on a
        busy machine.
        """
        if getattr(getattr(self, 'config', None), 'pause_while_running', None):
            try:
                from mousereach.dlc.core.interruptible import kill_orphaned_workers
                kill_orphaned_workers()
            except Exception as e:
                logger.warning(f"Could not check for poses left running by an "
                               f"earlier watcher: {e}")
        return super()._reclaim_orphaned_work()

    def __init__(self, config: WatcherConfig, db: WatcherDB):
        super().__init__(config, db)

        # A GPU node that also processes IS the processing role for what it
        # archives: it runs the version scan over its own rows and publishes
        # re-pose requests it then consumes itself next poll. That is how a
        # lab with one GPU PC and a shared drive keeps its archive current.
        # (Rows it marks with a narrower scope are held and named; this role
        # has no post-DLC reprocess handler yet.)
        self.handles_reprocessing = bool(config.also_process)

        # Staging directory on NAS for processing PC to pick up
        self.staging_dir = Paths.DLC_STAGING
        if self.staging_dir:
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Staging directory: {self.staging_dir}")

        # --- Cross-PC coordination via the shared watcher_central.db ---
        # WHY the failure is RECORDED and not just left as coordinator=None:
        # None used to mean "crop without a claim", so a node whose share
        # blinked at startup cropped every collage other nodes were also
        # cropping. The collage handler now refuses to crop without a working
        # coordinator (see _hold_collage_claim) and says why, once, using the
        # reason kept here. Restore and recovery are separate steps on purpose:
        # neither is needed for a claim, so neither may switch claims off.
        self.coordinator = None
        self._coordinator_init_error = None
        try:
            from mousereach.watcher.coordination import (
                PipelineCoordinator, restore_db, backup_db
            )
            self._backup_db = backup_db
            self._restore_db = restore_db
        except Exception as e:
            PipelineCoordinator = None
            self._coordinator_init_error = f"{type(e).__name__}: {e}"

        if PipelineCoordinator is not None:
            # Quick-restore watcher.db from NAS backup if local is empty
            try:
                nas_root = Paths.NAS_ROOT
                if nas_root:
                    restore_db(db.db_path, nas_root, self.hostname)
            except Exception as e:
                logger.warning(f"watcher.db restore from the shared drive skipped: {e}")

            try:
                coordinator = PipelineCoordinator()
                coordinator.ensure_tables()
                self.coordinator = coordinator
            except Exception as e:
                self._coordinator_init_error = f"{type(e).__name__}: {e}"

        if self.coordinator is not None:
            try:
                self.coordinator.recover_local_db(db, self.hostname)
                logger.info("Startup recovery from the coordination database complete")
            except Exception as e:
                logger.warning(f"Cross-node recovery skipped: {e}")
        else:
            self._coordinator_last_attempt = time.monotonic()
            self._coordinator_unavailable_since = datetime.now().isoformat(timespec='seconds')
            logger.warning(
                "Collage coordination database unavailable (%s). Posing and "
                "staging continue, but this node crops no collage until the "
                "database answers; it is tried again every %d minutes.",
                self._coordinator_init_error, self._COORDINATOR_RETRY_S // 60)

        # Scan local DLC_Queue for orphaned files not in DB
        self._recover_local_dlc_queue()

    # =========================================================================
    # COORDINATION HELPERS
    # =========================================================================

    def _sync_to_connectome(self, video_id, state, **kwargs):
        """Best-effort sync video state to connectome.db after local DB update."""
        if not self.coordinator:
            return
        try:
            self.coordinator.sync_video_state(video_id, self.hostname, state, **kwargs)
        except Exception as e:
            logger.debug(f"Connectome sync failed (non-fatal): {e}")

    def _backup_local_db(self):
        """Best-effort backup of local watcher.db to NAS."""
        try:
            nas_root = Paths.NAS_ROOT
            if nas_root and hasattr(self, '_backup_db'):
                self._backup_db(self.db.db_path, nas_root, self.hostname)
        except Exception as e:
            logger.debug(f"DB backup failed (non-fatal): {e}")

    # =========================================================================
    # FINISHED OR HELD: never pose (or crop for) a video that needs no pose
    # =========================================================================

    def _already_done_or_held(self, video_id: str) -> Optional[Tuple[str, str]]:
        """Is this video finished, or held for a person? ``(state, reason)`` if
        so, None if it genuinely still needs a pose.

        ``state`` is what this node's row should say instead: 'archived', or
        the review queue's state ('triage' / 'deep_review').

        WHY ask the shared drive instead of this node's database: the database
        is exactly what cannot be trusted here. A new GPU node, or one whose
        database was rebuilt, knows nothing about the thousands of videos
        already analysed, and a pose costs ~14 GPU-minutes that then feeds
        duplicate processing downstream. The shared drive is the authority:

          * a bundle folder in Processing/Review/triage/<stem> or
            deep_review/<stem> means a person holds the video -- posing it again
            would build a second result beside the one under review;
          * <queue>/.incoming/<stem> is a route into that queue still being
            written (review_routing publishes it with one rename) -- the same
            hold, a moment earlier;
          * <stem>_processing_manifest.json in the video's archive folder means
            it was analysed and filed.

        The queues are checked first because a held video can ALSO have an
        older manifest in the archive (a reprocess under review), and the hold
        is the more current truth -- the same order _archive_to_nas uses.
        Never raises: an unreadable location counts as "not there".
        """
        try:
            from mousereach.watcher.review_routing import INCOMING_DIR_NAME
        except ImportError:
            INCOMING_DIR_NAME = ".incoming"
        for qroot, qstate in ((getattr(Paths, 'TRIAGE_REVIEW', None), 'triage'),
                              (getattr(Paths, 'DEEP_REVIEW', None), 'deep_review')):
            if not qroot:
                continue
            try:
                if (Path(qroot) / video_id).is_dir():
                    return qstate, f"its bundle is held for a person in the {qstate} queue"
                if (Path(qroot) / INCOMING_DIR_NAME / video_id).is_dir():
                    return qstate, (f"its bundle is being routed into the {qstate} "
                                    f"queue ({INCOMING_DIR_NAME})")
            except OSError:
                continue
        try:
            from mousereach.archive.core import get_archive_destination
            archived_here = get_archive_destination(video_id)
            if (archived_here is not None
                    and (Path(archived_here) / f"{video_id}_processing_manifest.json").is_file()):
                return 'archived', "already analysed and filed in the archive"
        except Exception:
            pass
        return None

    def _record_finished_child(self, video_id: str, verdict: Tuple[str, str],
                               collage_filename: str, metadata: dict) -> bool:
        """Record a collage child that needs no pose in the state the shared
        drive says it is in. True if this node's row was set.

        A child whose row here has already moved past discovery is left
        alone: that row is this node's own history of the video (in flight,
        archived, held) and is at least as informative. Only a row that is
        new, 'validated', 'failed' or 'unresolvable' is corrected.

        No file path is recorded (NO_FILE_HERE, as cross-node recovery does for
        finished videos): the crop in the working folder is deleted with the
        rest, and nothing reads a finished row's path.
        """
        state, reason = verdict
        row = self.db.get_video(video_id)
        if row is None:
            self.db.register_video(
                video_id=video_id,
                source_path=getattr(self.db, 'NO_FILE_HERE', '(no file on this node)'),
                collage_id=collage_filename,
                **{k: v for k, v in metadata.items() if v is not None})
            prior = 'discovered'
        else:
            prior = row.get('state')
        if prior not in ('discovered', 'validated', 'failed', 'unresolvable'):
            logger.info(f"{video_id}: {reason}; its row here is already '{prior}', "
                        f"left as it is")
            return False
        self.db.force_state(
            video_id, state,
            reason=f"{reason}; recorded instead of queuing it for DLC from "
                   f"collage {collage_filename}")
        self.db.log_step(video_id, 'crop', 'skipped', message=reason)
        logger.info(f"{video_id}: {reason}; recorded as '{state}', not queued for DLC")
        return True

    @staticmethod
    def _child_metadata(video_id: str, animal_id: str, position, collage_data: dict) -> dict:
        """The columns a cropped child is registered with, from its names."""
        parsed_animal = AnimalID.parse(animal_id) if animal_id else {}
        tray_info = parse_tray_type(f"{video_id}.mp4")
        return {
            'date': collage_data.get('date'),
            'animal_id': animal_id,
            'experiment': parsed_animal.get('experiment'),
            'cohort': parsed_animal.get('cohort'),
            'subject': parsed_animal.get('subject'),
            'tray_type': tray_info.get('tray_type'),
            'tray_position': position,
        }

    # =========================================================================
    # COLLAGE CLAIMS: fail closed
    # =========================================================================

    def _warn_collage_claims_unavailable(self) -> None:
        """Say that this node crops nothing, and why -- once, then again every
        _CLAIMS_UNAVAILABLE_REWARN_S for as long as it lasts."""
        now = time.monotonic()
        last = getattr(self, '_claims_unavailable_warned_at', None)
        if last is not None and now - last < self._CLAIMS_UNAVAILABLE_REWARN_S:
            return
        self._claims_unavailable_warned_at = now
        since = getattr(self, '_coordinator_unavailable_since', None)
        logger.warning(
            "Collage claims are unavailable on this node%s (%s). No collage is "
            "cropped here until the central database (watcher_central.db on the "
            "shared drive) answers; it is tried again every %d minutes. WHY: "
            "without a claim, every GPU node sharing the intake folder can crop "
            "the same collage and pose its children twice.",
            f" since {since}" if since else "",
            getattr(self, '_coordinator_init_error', None)
            or "the coordination database did not start",
            self._COORDINATOR_RETRY_S // 60)

    def _retry_coordinator(self) -> bool:
        """True when a coordinator is available, trying to build one again if
        the constructor's attempt failed and _COORDINATOR_RETRY_S has passed.

        Only a RECORDED startup failure is retried: a node built without the
        constructor, or one whose coordinator was switched off on purpose, is
        left as it is. Cross-node recovery is NOT run here -- it rewrites rows
        at startup, before work begins, and must not run under live work; it
        runs at the next restart.
        """
        if getattr(self, 'coordinator', None) is not None:
            return True
        if getattr(self, '_coordinator_init_error', None) is None:
            return False
        now = time.monotonic()
        last = getattr(self, '_coordinator_last_attempt', None)
        if last is not None and now - last < self._COORDINATOR_RETRY_S:
            return False
        self._coordinator_last_attempt = now
        try:
            from mousereach.watcher.coordination import PipelineCoordinator
            coordinator = PipelineCoordinator()
            coordinator.ensure_tables()
        except Exception as e:
            self._coordinator_init_error = f"{type(e).__name__}: {e}"
            logger.debug(f"coordination database still unavailable: {e}")
            return False
        since = getattr(self, '_coordinator_unavailable_since', None)
        self.coordinator = coordinator
        self._coordinator_init_error = None
        self._coordinator_unavailable_since = None
        self._claims_unavailable_warned_at = None
        logger.info(
            "Collage coordination database is available again%s; this node crops "
            "collages again. Cross-node recovery runs at the next watcher restart.",
            f" (unavailable since {since})" if since else "")
        return True

    def _claim_backoff_active(self, collage_filename: str) -> bool:
        """True while this node waits before asking about a collage another
        host was cropping (see _park_collage_claimed_elsewhere)."""
        until = getattr(self, '_claim_backoff', {}).get(collage_filename)
        return until is not None and time.monotonic() < until

    def _hold_collage_claim(self, collage_filename: str) -> bool:
        """True only when THIS node holds the shared claim on the collage.

        Fails closed on every path. The claim is the only thing stopping
        several GPU nodes from cropping one collage and each posing its
        children (~14 GPU-minutes a child, then duplicate processing and
        archiving downstream); skipping a poll costs nothing, because the
        collage is still in the intake folder next poll.

          * no coordinator (it failed to start): try it again when the retry
            interval has passed (_retry_coordinator); otherwise do not crop,
            with a WARNING repeated while it lasts.
          * the claim call raised (share or database trouble): do not crop;
            one WARNING per collage per outage (a successful check on that
            collage re-arms it); the row stays 'stable', so the next poll
            tries again.
          * another host holds the claim: see _park_collage_claimed_elsewhere.
        """
        if not self._retry_coordinator():
            self._warn_collage_claims_unavailable()
            return False
        coordinator = self.coordinator
        warned = getattr(self, '_claim_error_warned', None)
        if warned is None:
            warned = self._claim_error_warned = set()
        try:
            held = coordinator.try_claim_collage(collage_filename, self.hostname)
        except Exception as e:
            if collage_filename not in warned:
                warned.add(collage_filename)
                logger.warning(
                    f"Collage {collage_filename}: could not check the shared claim "
                    f"({type(e).__name__}: {e}); not cropping it this poll. It "
                    f"stays queued and is tried again next poll.")
            return False
        # The check worked: a later outage must warn again for this collage.
        warned.discard(collage_filename)
        if not held:
            self._park_collage_claimed_elsewhere(collage_filename)
            return False
        getattr(self, '_claim_backoff', {}).pop(collage_filename, None)
        return True

    def _park_collage_claimed_elsewhere(self, collage_filename: str) -> None:
        """Another node holds the claim: stop this node re-picking the collage.

        WHY park rather than just return: the row stayed 'stable', so the loop
        picked the same collage again on the very next pass, lost the claim
        again, and copied the whole local watcher.db to the share after each
        attempt.

        Two cases, told apart by the shared claim row:

          * the crop there FINISHED ('cropped'/'archived'): this node's row is
            set to 'cropped' for good, with the holder named in the collage's
            processing log (step 'claim', status 'skipped'). That is the truth
            -- the collage has been cropped, just not here.
          * the claim is live, stale-but-refused, or could not be read: the row
            stays 'stable' and this node does not ask about it again for
            _CLAIM_RECHECK_S. WHY not 'cropped': a claim can still be released
            after a failed crop, or taken over once stale, and a node that had
            parked the collage for good would never crop it then -- with two or
            three GPU nodes, that was every node that had looked (G2's "blocked
            forever", just moved).
        """
        claim = None
        try:
            claim = self.coordinator.get_collage_claim(collage_filename)
        except Exception as e:
            logger.debug(f"could not read who holds {collage_filename}: {e}")
        claim = claim or {}
        holder = claim.get('hostname') or "another node"
        state = claim.get('state')
        if state in ('cropped', 'archived'):
            reason = (f"already cropped by {holder} (shared claim is '{state}'); "
                      f"not cropped again here")
            try:
                self.db.force_collage_state(collage_filename, 'cropped')
                self.db.log_step(collage_filename, 'claim', 'skipped', message=reason)
            except Exception as e:
                logger.warning(f"Collage {collage_filename}: {reason}, but its row here "
                               f"could not be recorded ({e})")
                return
            logger.info(f"Collage {collage_filename}: {reason}; recorded here as 'cropped'")
            return

        backoff = getattr(self, '_claim_backoff', None)
        if backoff is None:
            backoff = self._claim_backoff = {}
        backoff[collage_filename] = time.monotonic() + self._CLAIM_RECHECK_S
        reason = (f"claimed by {holder} (shared claim is '{state or 'unknown'}', "
                  f"claimed at {claim.get('claimed_at') or '?'}); left to that node, "
                  f"asked about again in {self._CLAIM_RECHECK_S // 60} minutes")
        logged = getattr(self, '_claim_parked_logged', None)
        if logged is None:
            logged = self._claim_parked_logged = set()
        if collage_filename not in logged:
            logged.add(collage_filename)
            try:
                self.db.log_step(collage_filename, 'claim', 'skipped', message=reason)
            except Exception as e:
                logger.debug(f"could not log the claim skip for {collage_filename}: {e}")
        logger.info(f"Collage {collage_filename}: {reason}")

    def _note_unsynced_collage_claim(self, collage_filename: str, singles_created: int,
                                     error) -> None:
        """Remember a 'cropped' that did not reach the shared claim table, so
        it is sent again (_retry_unsynced_collage_claims). WHY it matters now:
        a claim left in 'cropping' used to block other nodes only; since claims
        can be taken over after COLLAGE_CLAIM_STALE_S, a lost 'cropped' is how
        a finished collage gets cropped again a day later."""
        pending = getattr(self, '_unsynced_collage_claims', None)
        if pending is None:
            pending = self._unsynced_collage_claims = {}
        pending[collage_filename] = singles_created
        logger.warning(
            f"Collage {collage_filename}: cropped here, but the shared claim could not "
            f"be marked 'cropped' ({type(error).__name__}: {error}). It is sent again "
            f"each poll until it lands; until then other nodes see an unfinished claim.")

    def _retry_unsynced_collage_claims(self) -> None:
        """Send again every 'cropped' that did not reach the shared table."""
        pending = getattr(self, '_unsynced_collage_claims', None)
        if not pending or getattr(self, 'coordinator', None) is None:
            return
        for name, created in list(pending.items()):
            try:
                changed = self.coordinator.update_collage_state(
                    name, 'cropped', only_if_held_by=self.hostname,
                    singles_created=created)
            except Exception as e:
                logger.debug(f"collage {name}: 'cropped' still not synced ({e})")
                continue
            pending.pop(name, None)
            if changed is False:
                self._warn_claim_lost_while_cropping(name)
            else:
                logger.info(f"Collage {name}: shared claim marked 'cropped' (retried)")

    def _warn_claim_lost_while_cropping(self, collage_filename: str) -> None:
        logger.warning(
            f"Collage {collage_filename}: cropped here, but the shared claim is no "
            f"longer held by this node, so it was not marked 'cropped'. Another node "
            f"took the claim over and may crop it too; a person should check that its "
            f"child videos are not posed twice.")

    def _collage_children_in_flight_here(self, collage_filename: str) -> list:
        """Video ids of this collage's children that THIS node is carrying
        (queued, posing, posed, staged, processing ...). Rows recorded with no
        file here (finished or held children, _record_finished_child) and rows
        that never got a file (discovered, validated, failed) do not count.
        Raises on database errors."""
        conn = self.db._get_connection()
        try:
            rows = conn.execute(
                "SELECT video_id FROM videos WHERE collage_id = ? "
                "AND state NOT IN ('discovered', 'validated', 'quarantined', "
                "'failed', 'unresolvable') AND COALESCE(source_path, '') != ?",
                (collage_filename, getattr(self.db, 'NO_FILE_HERE',
                                           '(no file on this node)'))).fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]

    def _adopt_single_for_dlc(self, work: dict):
        """Bring a video someone left in the shared singles folder onto this
        node and queue it for pose.

        Copying rather than posing it where it lies is the whole point:
        DeepLabCut writes its output beside its input, and the input is on the
        shared drive. Posing in place would scatter pose files across a folder
        every machine reads.

        CLAIM FIRST (watcher/single_claim.py). Several GPU nodes poll the same
        folder, and each used to copy and pose every video it saw. The video is
        now renamed into ``.inflight/<this host>/`` before it is copied; a node
        whose rename finds nothing lost the race and leaves the video to the
        winner, recording nothing against it. The claimed file stays there
        until this node has handed the video on, and is then removed as a
        duplicate (_retire_claimed_single). Order of checks: finished or held
        first (no claim for a video that needs no pose), then the claim, then
        the copy.
        """
        video_id = work['id']
        data = work['data']

        # NEVER pose a video that is already finished. Before this state was
        # worked at all, the shared singles folder was full of videos that had
        # long since been analysed and filed, and being ignored was the only
        # thing keeping them quiet. Now that the state is live, a node whose
        # database does not know them -- a new GPU machine, or a rebuilt
        # database -- would queue every one of them for pose. Measured on the
        # lab's own folder: 2,271 such videos, at about 14 minutes of GPU
        # each, or roughly 530 GPU-hours of redoing finished work.
        #
        # The archive is the authority on what is finished, so ask it, and
        # record the truth instead of burning a card on it. A video held in a
        # review queue is not posed again either (_already_done_or_held).
        verdict = self._already_done_or_held(video_id)
        if verdict:
            state, why = verdict
            try:
                self.db.force_state(
                    video_id, state,
                    reason=f"{why}; recorded as such rather than posed again")
                self.db.log_step(video_id, 'adopt', 'skipped', message=why)
            except Exception as e:
                logger.warning(f"{video_id}: {why}, but the row could not be "
                               f"corrected ({e})")
            logger.info(f"{video_id}: {why}; not posing it again")
            return True

        dlc_queue = Paths.DLC_QUEUE
        if not dlc_queue:
            self.db.mark_unresolvable(
                video_id, "no local DLC queue is configured on this node")
            return False

        # A copy already in THIS node's claim folder is this node's own claim:
        # the watcher stopped between claiming and queuing. Carry on from it
        # instead of claiming again (a second claim would find nothing).
        mine = single_claim.claimed_path(video_id, self.hostname)
        try:
            held_here = mine is not None and mine.is_file()
        except OSError:
            held_here = False
        if held_here:
            src = mine
        else:
            # search_staging=False: Processing/Posed is the processing server's
            # intake. Adopting from it would copy another node's hand-off into
            # this node's queue and pose the same video twice (watcher/locate.py).
            src = locate_video_file(video_id, raw=data.get('current_path'),
                                    extra_dirs=[Paths.SINGLE_ANIMAL_OUTPUT],
                                    search_archive=False, search_staging=False)
            holder = single_claim.claim_holder(src) if src is not None else None
            if holder is not None and holder != self.hostname:
                src = None              # another node's claimed copy is never ours
        if src is None:
            # Gone from the folder. Say who has it when a node's claim shows it;
            # the reason's fixed start lets the intake scan validate the row
            # again if the video comes back (state.discover_new_singles).
            try:
                holder = single_claim.inflight_ids().get(video_id)
            except Exception:
                holder = None
            if holder and holder != self.hostname:
                where = (f"{holder} took it for pose (it is in "
                         f"{single_claim.INFLIGHT_DIR}/{holder})")
            else:
                where = "the file is no longer there"
            self.db.mark_unresolvable(
                video_id, f"{single_claim.LEFT_FOLDER_REASON}: {where}")
            return False

        claimed_here = held_here
        if single_claim.in_front_door(src):
            # A same-named video another node ALREADY holds in .inflight: this
            # file is a second copy (normally the batch was copied in again
            # while the first copy was out). Claiming it would pose the video
            # twice, and both poses would then land in Processing/Posed under
            # one name. Left where it is for a person; recorded with a reason
            # the intake scan never re-drives.
            # Only when the file really is still here: a node whose view is a
            # moment old sees the path of a video another node has just
            # claimed, and that is a lost race, not a second copy.
            try:
                other = single_claim.inflight_ids().get(video_id)
                if other and not Path(src).is_file():
                    other = None
            except Exception:
                other = None
            if other and other != self.hostname:
                self.db.mark_unresolvable(
                    video_id, f"{single_claim.DUPLICATE_OF_CLAIM_REASON}: {other} "
                              f"is posing it (it is in {single_claim.INFLIGHT_DIR}/"
                              f"{other}); remove the second copy at {src}")
                logger.warning(f"{video_id}: {src} is a second copy of a video {other} "
                               f"already holds for pose; not taken. A person should "
                               f"delete this second copy (the first is being posed).")
                return False
            claimed = single_claim.claim_single(src, self.hostname)
            if claimed is None:
                try:
                    still_there = Path(src).is_file()
                except OSError:
                    still_there = False
                if still_there:
                    # Refused while the file is still here: a program has it
                    # open, or this account may not rename in the folder. Park
                    # the row for a while so the collages behind it are not
                    # starved (_single_claim_backoff_active).
                    self._note_single_claim_refused(video_id, src)
                else:
                    # Lost the race. Nothing destructive is recorded: the row
                    # stays 'validated', and the next pass records who has it.
                    self._note_single_claim_lost(video_id)
                return False
            self._clear_single_claim_refused(video_id)
            src = claimed
            claimed_here = True

        dest = Path(dlc_queue) / f"{video_id}.mp4"
        try:
            Path(dlc_queue).mkdir(parents=True, exist_ok=True)
            same = dest.is_file() and dest.stat().st_size == Path(src).stat().st_size
        except OSError:
            same = False
        if not same and not safe_copy(Path(src), dest, verify=True):
            # Give the video back so another node can take it; a claim this
            # node cannot use must not keep it from everyone. THIS node does
            # not retry it by itself: its row is 'failed', and the GPU role
            # has no job list for failed rows (a full local disk would
            # otherwise copy a whole video every minute). A person resets it.
            released = claimed_here and single_claim.release_single(src)
            back = (Path(src).parent.parent.parent / Path(src).name) if released else src
            self.db.mark_failed(
                video_id, f"could not copy {back} into the DLC queue"
                          + ("; put back in the shared singles folder, where another "
                             "GPU node can take it. This node does not retry it on "
                             "its own: fix the cause, then run "
                             f"mousereach-watch-reprocess {video_id}"
                             if released else ""))
            return False
        self.db.update_state(video_id, 'dlc_queued',
                             current_path=str(dest), source_path=str(dest))
        held_note = (f"; the shared copy is held in {Path(src).parent} until this "
                     f"node hands the video on" if claimed_here else "")
        self.db.log_step(video_id, 'adopt', 'completed',
                         message=f"taken from the shared singles folder into "
                                 f"{dlc_queue}{held_note}")
        if claimed_here:
            logger.info(f"{video_id}: taken from the shared singles folder (held in "
                        f"{Path(src).parent} until handed on); adopted onto this "
                        f"node and queued for DLC")
        else:
            logger.info(f"{video_id}: left in the shared singles folder; adopted "
                        f"onto this node and queued for DLC")
        return True

    def _note_single_claim_lost(self, video_id: str) -> None:
        """Say once per video that its claim was not taken. WHY once: the row
        stays 'validated' and is offered again next poll, and a still-copying
        file can refuse the claim for many polls in a row."""
        logged = getattr(self, '_single_claim_lost_logged', None)
        if logged is None:
            logged = self._single_claim_lost_logged = set()
        if video_id in logged:
            return
        logged.add(video_id)
        logger.info(f"{video_id}: not taken from the shared singles folder -- another "
                    f"node claimed it first, or it is still in use (being copied "
                    f"in). Nothing is recorded against it here; it is left to "
                    f"whoever has it.")

    # A refused claim (the file is still in the folder but cannot be renamed)
    # parks the row this long before it is offered again, and after a refusal
    # has lasted _SINGLE_REFUSED_WARN_S it is reported once as a WARNING.
    # WHY a backoff: 'validated' rows are picked before collages, so a single a
    # media player keeps open would otherwise be picked every poll and no
    # collage would ever be cropped on this node. WHY a WARNING only later: a
    # file still being copied in refuses for a minute or two, which is normal.
    _SINGLE_REFUSED_BACKOFF_S = 120
    _SINGLE_REFUSED_WARN_S = 600

    def _single_claim_backoff_active(self, video_id: str) -> bool:
        entry = getattr(self, '_single_claim_refused', {}).get(video_id)
        return entry is not None and time.monotonic() < entry[0]

    def _clear_single_claim_refused(self, video_id: str) -> None:
        getattr(self, '_single_claim_refused', {}).pop(video_id, None)

    def _note_single_claim_refused(self, video_id: str, src) -> None:
        """Park a single whose claim was refused, and say so: INFO the first
        time, one WARNING once the refusal has lasted _SINGLE_REFUSED_WARN_S
        (naming the file and the usual causes)."""
        refused = getattr(self, '_single_claim_refused', None)
        if refused is None:
            refused = self._single_claim_refused = {}
        now = time.monotonic()
        _, first, warned = refused.get(video_id, (0.0, now, False))
        if video_id not in refused:
            logger.info(f"{video_id}: could not be taken from the shared singles folder "
                        f"yet (normally it is still being copied in); tried again in "
                        f"{self._SINGLE_REFUSED_BACKOFF_S // 60} min")
        if not warned and now - first >= self._SINGLE_REFUSED_WARN_S:
            warned = True
            logger.warning(
                f"{video_id}: {src} has refused to be taken for "
                f"{(now - first) / 60:.0f} min. Usually a program has it open (a "
                f"video player, or a copy that stalled), or this computer's account "
                f"may read the folder but not rename files in it. Close the program "
                f"or fix the folder permissions; it is tried again every "
                f"{self._SINGLE_REFUSED_BACKOFF_S // 60} min meanwhile.")
        refused[video_id] = (now + self._SINGLE_REFUSED_BACKOFF_S, first, warned)

    # After this many pose failures in a row, stop taking NEW singles from the
    # shared folder for _POSE_FAILURE_PAUSE_S. WHY: a node whose DeepLabCut is
    # broken (a driver or DLL error) would otherwise claim every single dropped
    # on the share and fail each one in turn, taking them from healthy nodes.
    _POSE_FAILURE_BRAKE = 3
    _POSE_FAILURE_PAUSE_S = 1800

    def _note_pose_result(self, ok: bool) -> None:
        if ok:
            self._pose_failure_streak = 0
            return
        streak = getattr(self, '_pose_failure_streak', 0) + 1
        self._pose_failure_streak = streak
        if streak >= self._POSE_FAILURE_BRAKE:
            self._singles_paused_until = time.monotonic() + self._POSE_FAILURE_PAUSE_S
            logger.warning(
                f"{streak} pose runs in a row have failed on this node. It takes no new "
                f"videos from the shared singles folder for "
                f"{self._POSE_FAILURE_PAUSE_S // 60} min, so other GPU nodes can pose "
                f"them. Check this node's DeepLabCut setup (see the failures above).")

    def _singles_braked(self) -> bool:
        until = getattr(self, '_singles_paused_until', None)
        return until is not None and time.monotonic() < until

    # How often the singles claim area is swept for claims nobody has touched
    # for single_claim.STALE_S. WHY minutes, not every poll: the sweep lists
    # every node's claim folder on the share, and a claim only counts as stale
    # after a day, so a few minutes' delay costs nothing.
    _SINGLE_RECLAIM_S = 300

    def _single_claim_upkeep(self, force: bool = False, sweep: bool = True) -> None:
        """Touch this node's claimed singles, and (``sweep``) every
        _SINGLE_RECLAIM_S hand back any node's claim left untouched for a day.
        Never raises.

        Called from every scan (forced, with the sweep) and from the paused
        loop (heartbeat only). WHY the heartbeat while paused: a long
        recording day skips the scan, and a claim without a heartbeat for a
        day is handed back -- another node would then pose a video this node
        has already copied. WHY no sweep while paused: a paused node takes no
        new work, so handing videos back only helps nodes that are running,
        and those sweep for themselves. The heartbeat is rate-limited to one
        per poll interval unless forced (it lists a folder on the share). It
        runs before the sweep, so a node back from a long stop refreshes its
        own claims before it could count them stale.
        """
        now = time.monotonic()
        interval = float(getattr(getattr(self, 'config', None),
                                 'poll_interval_seconds', 30) or 30)
        last = getattr(self, '_last_single_heartbeat', None)
        if force or last is None or now - last >= interval:
            self._last_single_heartbeat = now
            try:
                single_claim.heartbeat_claims(self.hostname,
                                              only=self._claims_to_keep_alive())
            except Exception as e:
                logger.warning(f"Heartbeat of claimed singles failed (non-fatal): {e}")
        if not sweep:
            return
        last_sweep = getattr(self, '_last_single_reclaim', None)
        if last_sweep is None or now - last_sweep >= self._SINGLE_RECLAIM_S:
            self._last_single_reclaim = now
            try:
                single_claim.reclaim_stale()
            except Exception as e:
                logger.warning(f"Sweep for stale single claims failed (non-fatal): {e}")

    # Row states whose claimed single this node keeps alive. Working states:
    # the node is still posing or processing the video from its local copy.
    # Handed-on states: the video has moved on but its claimed copy could not
    # be confirmed against the next copy and was kept; returning it to the
    # folder could have it posed again before the server has filed it.
    _CLAIM_KEEPALIVE_STATES = frozenset({
        'validated', 'dlc_queued', 'dlc_running', 'dlc_complete', 'processing',
        'processed', 'archiving', 'archived', 'triage', 'deep_review',
    })

    def _claims_to_keep_alive(self):
        """Video ids of this host's claims that the heartbeat refreshes, or
        None (refresh all) when this node's database cannot be read.

        A claim whose row is failed, unresolvable, quarantined or missing is
        NOT refreshed: this node has given up on that video (or never knew
        it), and refreshing the claim would hold the only shared copy from
        every other node for as long as this node runs. Left alone, it goes
        stale after single_claim.STALE_S and a running GPU node returns it.
        Each one is named once at WARNING. WHY None on a database error: a
        database hiccup must never hand back videos this node is posing.
        """
        claims = single_claim.host_claims(self.hostname)
        if not claims:
            return set()
        keep = set()
        warned = getattr(self, '_claim_abandon_warned', None)
        if warned is None:
            warned = self._claim_abandon_warned = set()
        for vid, path in claims.items():
            try:
                row = self.db.get_video(vid)
            except Exception:
                return None
            state = row.get('state') if row else None
            if state in self._CLAIM_KEEPALIVE_STATES:
                keep.add(vid)
                continue
            if (vid, state) not in warned:
                warned.add((vid, state))
                logger.warning(
                    f"{vid}: this node's claimed copy {path} is no longer refreshed "
                    f"(this node's row is {state or 'missing'}); it goes back to the "
                    f"shared singles folder for another node after "
                    f"{single_claim.STALE_S // 3600} h without a refresh")
        return keep

    def _release_claim_given_up(self, video_id: str, why: str) -> bool:
        """Give a claimed single back to the folder at once when this node
        stops working it (pose failed, or no local file any more). Never
        raises. WHY at once rather than waiting for it to go stale: another
        GPU node can pose it now, instead of a day later."""
        try:
            claimed = single_claim.claimed_path(video_id, self.hostname)
            if claimed is None or not claimed.is_file():
                return False
            if single_claim.release_single(claimed):
                logger.warning(f"{video_id}: {why}; its claimed copy went back to the "
                               f"shared singles folder for another node to take")
                return True
        except Exception as e:
            logger.warning(f"{video_id}: could not release its claimed copy ({e})")
        return False

    def _retire_claimed_single(self, video_id: str, handed_on,
                               verified: bool = False) -> bool:
        """Remove this node's claimed copy of a single once the video has
        moved on. True when removed.

        The claimed copy (``<Single_Animal>/.inflight/<host>/<stem>.mp4``) is
        the file this node took from the shared singles folder. It is kept
        until the video reaches its next stage because until then it may be
        the only copy on the share -- this node's queue is on its own disk.
        It is removed only when that next copy is confirmed: ``handed_on``
        exists with the same size, or, when it has already been taken onward
        (the processing server can take a staged video in within seconds),
        ``verified`` says the step that wrote it checked its copy. Otherwise
        it is kept with a WARNING naming it, and the heartbeat keeps it
        claimed so no other node poses it again.

        Never raises: it runs after the work itself has succeeded.
        """
        claimed = None
        try:
            claimed = single_claim.claimed_path(video_id, self.hostname)
            if claimed is None or not claimed.is_file():
                return False
            basis = None
            if handed_on is not None and Path(handed_on).is_file():
                if Path(handed_on).stat().st_size == claimed.stat().st_size:
                    basis = f"its copy at {handed_on} is complete"
            elif verified:
                basis = "the step that handed it on verified its copy"
            if basis is None:
                logger.warning(
                    f"{video_id}: this node's claimed copy {claimed} is KEPT: the "
                    f"video's next copy could not be confirmed ({handed_on}). "
                    f"Nothing is removed without that; delete it by hand once the "
                    f"video is safely on.")
                return False
            claimed.unlink()
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.warning(f"{video_id}: could not remove this node's claimed copy "
                           f"{claimed} ({type(e).__name__}: {e}); it stays claimed "
                           f"and can be deleted by hand once the video is safely on")
            return False
        logger.info(f"{video_id}: removed this node's claimed copy from the shared "
                    f"singles folder ({claimed}); {basis}")
        try:
            self.db.log_step(video_id, 'single_claim', 'completed',
                             message=f"removed claimed shared copy {claimed}; {basis}")
        except Exception:
            pass
        return True

    def _retire_claim_after_hold(self, video_id: str) -> bool:
        """A video routed to a review queue took its mp4 into the bundle
        (``<queue>/<stem>/<stem>.mp4``, watcher/review_routing.py): retire the
        claimed copy against that. Without a bundle mp4 the copy is kept."""
        for qroot in (getattr(Paths, 'TRIAGE_REVIEW', None),
                      getattr(Paths, 'DEEP_REVIEW', None)):
            if not qroot:
                continue
            candidate = Path(qroot) / video_id / f"{video_id}.mp4"
            try:
                if candidate.is_file():
                    return self._retire_claimed_single(video_id, candidate)
            except OSError:
                continue
        return self._retire_claimed_single(video_id, None)

    def _adopt_untracked_queue_files(self) -> int:
        """Pick up anything dropped straight into this node's DLC queue.

        The startup sweep below does the same thing plus retry handling, but
        only at startup: a file dropped while the watcher was running sat
        there until somebody restarted it. This runs every cycle and only
        touches files the database has never heard of, so it cannot disturb
        anything already in flight.

        DeepLabCut's own by-products are skipped rather than quarantined --
        a '..._labeled.mp4' beside a pose is output, not an unprocessed video,
        and treating it as a misnamed one would file it as a problem.
        """
        dlc_queue = Paths.DLC_QUEUE
        if not dlc_queue or not Path(dlc_queue).exists():
            return 0
        adopted = 0
        for mp4 in Path(dlc_queue).glob("*.mp4"):
            if "DLC" in mp4.stem:
                continue                       # a labeled/overlay by-product
            video_id = get_video_id(mp4.name)
            if not video_id:
                continue
            try:
                if self.db.get_video(video_id) is not None:
                    continue                   # already tracked
            except Exception:
                continue
            h5s = list(Path(dlc_queue).glob(f"{video_id}DLC*.h5"))
            state = 'dlc_complete' if h5s else 'dlc_queued'
            try:
                self.db.register_video(video_id=video_id, source_path=str(mp4),
                                       current_path=str(mp4))
                self.db.force_state(
                    video_id, state,
                    reason="found in this node's DLC queue with no database row",
                    current_path=str(mp4),
                    dlc_output_path=str(h5s[0]) if h5s else None)
                adopted += 1
                logger.info(f"{video_id}: dropped into the DLC queue; picked up as "
                            f"'{state}'")
            except Exception as e:
                logger.warning(f"could not pick up {mp4.name}: {e}")
        return adopted

    def _recover_local_dlc_queue(self):
        """Scan DLC_Queue for orphaned MP4s not in local DB.

        Infers state from sibling files:
          - MP4 only -> dlc_queued
          - MP4 + *DLC*.h5 -> dlc_complete
          - MP4 + h5 + pipeline JSONs -> processed
        """
        dlc_queue = Paths.DLC_QUEUE
        if not dlc_queue or not dlc_queue.exists():
            return

        recovered = 0
        for mp4 in dlc_queue.glob("*.mp4"):
            video_id = get_video_id(mp4.name)
            if not video_id:
                continue

            try:
                existing = self.db.get_video(video_id)
            except Exception:
                existing = None

            if existing is not None:
                # 'unresolvable' means a previous pass could find no file for
                # this video here. The file is here now, so the reason is gone:
                # put it back in the pipeline. Without this the terminal state
                # would be a one-way door and a video that arrived late would sit
                # in the database forever with its file on disk beside it.
                if existing.get('state') == 'unresolvable':
                    self.db.force_state(
                        video_id, 'dlc_queued',
                        reason="file has since appeared in DLC_Queue",
                        current_path=str(mp4), source_path=str(mp4),
                        error_message=None)
                    logger.info(f"{video_id}: file found in DLC_Queue; requeued")
                elif (existing.get('state') == 'outdated'
                      and (existing.get('reprocess_scope') or '') == 'full'):
                    # A re-pose that was queued here and then regressed to
                    # 'outdated' (a restart mid-run, before the dlc_queued
                    # sync landed). Its video is here and no pose from the
                    # declared model is beside it: queue it again.
                    from mousereach.watcher import repose
                    declared = repose.declared_scorer()
                    if declared and not [h for h in dlc_queue.glob(f"{video_id}DLC*.h5")
                                         if repose.scorer_of(h) == declared]:
                        repose.clear_stale_poses(dlc_queue, video_id, declared)
                        reason = (existing.get('mark_reason')
                                  or f"{repose.REASON_PREFIX} resumed after restart")
                        self.db.force_state(
                            video_id, 'dlc_queued',
                            reason="video is in DLC_Queue with no declared-model pose; "
                                   "re-pose resumed after restart",
                            current_path=str(mp4), source_path=str(mp4),
                            dlc_output_path=None, error_message=None,
                            mark_reason=reason)
                        logger.info(f"{video_id}: re-pose resumed from DLC_Queue")
                elif (existing.get('state') == 'failed'
                      and int(existing.get('error_count') or 0)
                          < getattr(self.config, 'max_retries', 3)):
                    # A pose or staging failure leaves this node's files where
                    # they were (staging copies everything before it deletes
                    # anything), so the video can simply be run again. This
                    # role's work loop has no 'failed' bucket, so without this
                    # nothing here would ever retry it: the video would sit
                    # failed with its files beside it, and a re-pose request
                    # for it would stay claimed until it went stale a day
                    # later. A restart is the retry.
                    from mousereach.watcher import repose
                    declared = repose.declared_scorer()
                    has_pose = bool(declared) and any(
                        repose.scorer_of(h) == declared
                        for h in dlc_queue.glob(f"{video_id}DLC*.h5"))
                    self.db.force_state(
                        video_id, 'dlc_complete' if has_pose else 'dlc_queued',
                        reason=f"retrying after a failure on this node "
                               f"(attempt {int(existing.get('error_count') or 0) + 1} "
                               f"of {getattr(self.config, 'max_retries', 3)}); "
                               f"its files are still in DLC_Queue",
                        current_path=str(mp4), source_path=str(mp4),
                        error_message=None)
                    logger.info(f"{video_id}: retrying after failure "
                                f"({'pose present' if has_pose else 'needs pose'})")
                continue  # Already tracked

            # Determine state from sibling files
            h5_files = list(dlc_queue.glob(f"{video_id}DLC*.h5"))
            json_files = list(dlc_queue.glob(f"{video_id}_*.json"))
            has_pipeline = any(
                f.name.endswith(('_segments.json', '_reaches.json', '_pellet_outcomes.json'))
                for f in json_files
            )

            if h5_files and has_pipeline:
                target_state = 'processed'
            elif h5_files:
                target_state = 'dlc_complete'
            else:
                target_state = 'dlc_queued'

            try:
                self.db.register_video(
                    video_id=video_id,
                    source_path=str(mp4),
                    current_path=str(mp4),
                )
                if target_state != 'discovered':
                    self.db.force_state(video_id, target_state)
                recovered += 1
                logger.debug(f"Recovered orphan {video_id} as {target_state}")
            except Exception as e:
                logger.debug(f"Could not recover orphan {video_id}: {e}")

        if recovered > 0:
            logger.info(f"Recovered {recovered} orphaned videos from DLC_Queue")

    # =========================================================================
    # SCAN PHASE
    # =========================================================================

    def _scan_phase(self):
        """Scan NAS for new collages/singles and check for DLC completions."""
        # Phase 0: keep this node's claims on shared singles alive and hand back
        # claims a dead node left (watcher/single_claim.py). First, because the
        # scan below can take many minutes on a shared drive.
        self._single_claim_upkeep(force=True)

        # Phase A: Scan for new files and check stability
        scan_result = self.file_watcher.scan()
        if scan_result.new_collages or scan_result.new_singles or scan_result.stable_ready:
            logger.info(
                f"Scan: {scan_result.new_collages} new collages, "
                f"{scan_result.new_singles} new singles, "
                f"{scan_result.stable_ready} now stable"
            )

        # Phase B: Check for DLC completions (h5 files appearing)
        self._scan_for_dlc_completions()

        # Phase B2: Pick up anything dropped straight into this node's queue.
        # The startup sweep catches these too, but only at startup -- a file
        # dropped while the watcher was running sat there until a restart.
        try:
            self._adopt_untracked_queue_files()
        except Exception as e:
            logger.warning(f"Could not check the DLC queue for dropped files: {e}")

        # Phase C: Pull re-pose requests from shared storage. A node that runs
        # the version scan asks for new poses by writing one small file per
        # video into Processing/Repose_Queue; this GPU node copies the archived
        # video into its own DLC_Queue and queues it. Pull, not push: no node
        # needs to know another exists (watcher/repose.py).
        try:
            from mousereach.watcher import repose
            latched = getattr(self, '_repose_latched', None)
            if latched is None:
                latched = self._repose_latched = set()
            self._repose_heartbeat(force=True)
            # Asked again here, not only at the top of the loop. WHY: taking a
            # request copies a whole archived video over the network onto
            # this PC's disk, and the scan before this point can take many
            # minutes on a shared drive. A recording that started meanwhile
            # must not share the disk with that copy. The heartbeat above
            # still ran, so claims this node already holds stay alive.
            if self._is_paused():
                return
            retry_after = getattr(self, '_repose_retry_after', None)
            if retry_after is None:
                retry_after = self._repose_retry_after = {}
            dlc_cfg = self.config.dlc_config_path
            summary = repose.consume_requests(
                self.db, dlc_queue=Paths.DLC_QUEUE, hostname=self.hostname,
                batch=getattr(self.config, 'repose_batch', 2),
                max_retries=getattr(self.config, 'max_retries', 3),
                can_pose=bool(dlc_cfg and Path(dlc_cfg).exists()),
                on_queued=lambda vid, mp4: self._sync_to_connectome(
                    vid, 'dlc_queued', source_path=str(mp4)),
                latched=latched, retry_after=retry_after)
            if summary.get('queued') or summary.get('completed'):
                logger.info(f"Re-pose requests: queued {summary['queued']} video(s) "
                            f"for DLC on this node, {summary['completed']} already posed")
        except Exception as e:
            logger.warning(f"Re-pose request scan failed (non-fatal): {e}")

    def _repose_heartbeat(self, force: bool = False) -> None:
        """Touch this node's claimed re-pose requests so the publisher knows
        the node is alive (repose.heartbeat). Never raises.

        Called from every scan, and from the paused branch of the main loop.
        WHY while paused: the scan is skipped while paused, and a claim with
        no heartbeat for repose.STALE_S (a day) is handed back to the queue --
        so a long recording day would give away work this node already copied
        and queued, and another GPU node would pose it a second time.

        Rate-limited to one call per poll interval unless ``force`` (the scan
        forces it, as it always has): the paused loop wakes every poll, and
        the heartbeat lists a folder on the shared drive.
        """
        interval = float(getattr(self.config, 'poll_interval_seconds', 30) or 30)
        now = time.monotonic()
        last = getattr(self, '_last_repose_heartbeat', None)
        if not force and last is not None and now - last < interval:
            return
        self._last_repose_heartbeat = now
        try:
            from mousereach.watcher import repose
            repose.heartbeat(self.db, hostname=self.hostname)
        except Exception as e:
            logger.warning(f"Re-pose heartbeat failed (non-fatal): {e}")

    def _while_paused(self) -> None:
        """Keep claimed re-pose requests and claimed singles alive while
        paused (see _repose_heartbeat and _single_claim_upkeep)."""
        self._repose_heartbeat()
        self._single_claim_upkeep(sweep=False)

    # =========================================================================
    # WORK QUEUE
    # =========================================================================

    def _select_work_item(self, admit_deferred: bool = True):
        """
        One pass over this node's work buckets -- priority animal first,
        then the configured order, random within the winning tier.

        Called twice per cycle by BaseOrchestrator._get_next_work_item: once
        with admit_deferred=False, and only if that finds nothing at all, once
        with it True. Every pick below is tested for None, because a bucket
        can be non-empty and still yield nothing on the first pass.

        ADMIT / DRAIN. Only the buckets that bring NEW work onto this node
        honour admit_deferred:

          ADMIT (gated):    single_dlc (2), collage (3)
          DRAIN (ungated):  archive_local (1a), local_pipeline (1b, 1c),
                            stage_to_nas (1c)

        The drain buckets finish work whose files are already on this node's
        disk. Holding them back does not save the GPU for anything -- the
        collage and DLC-queue buckets below are where a deferred session would
        actually consume it -- and it does strand results: a node with a
        backlog is never idle, so a deferred archive or stage never happens
        at all. They are still ORDERED by the policy: inside each bucket the
        preferred tier is picked first.

        Priority:
        1. Stage DLC-complete videos to NAS (finish what's done)
        2. Run DLC on a queued single
        3. Crop next collage (only when nothing in-flight)
        """
        priority_animal = self._get_priority_animal()

        # Priority 1a: Archive locally processed videos (also_process mode)
        # DRAIN: the result exists; filing it is a file move, not analysis.
        if self.config.also_process:
            processed = [v for v in self.db.get_videos_in_state('processed')
                         if not self._archive_backoff_active(v['video_id'])]
            pick = self._pick_from_pool(processed, priority_animal, 'animal_id',
                                        randomize=False)
            if pick is not None:
                return {
                    'type': 'archive_local',
                    'id': pick['video_id'],
                    'data': pick
                }

        # Priority 1b: Run local pipeline on DLC-complete videos (also_process mode)
        # DRAIN: the video and its pose are already here, already occupying
        # this node's disk. Finishing it is what frees the node.
        if self.config.also_process:
            processing = self.db.get_videos_in_state('processing')
            pick = self._pick_from_pool(processing, priority_animal, 'animal_id')
            if pick is not None:
                return {
                    'type': 'local_pipeline',
                    'id': pick['video_id'],
                    'data': pick
                }

        # Priority 1c: Videos with DLC complete - stage to NAS or run local pipeline
        # DRAIN either way. Handing the pose to the processing server is a file
        # move, and the server applies its own admit rule when it decides what
        # to RUN -- holding a finished pose on this node's disk buys nothing and
        # hides the work from every other node. Running the pipeline on it here
        # is finishing a video this node already has.
        # For staging, prefer priority animal but no randomization (stage ASAP).
        videos = self.db.get_videos_in_state('dlc_complete')
        # Rows already KNOWN to have no file here never get a work slot. WHY:
        # cross-node recovery learns about videos another machine holds, and
        # registers them with the NO_FILE_HERE placeholder. Each such row was
        # still selected, and the handler then spent the whole poll interval
        # discovering there was no file -- one row per poll, so six of them cost
        # six polls (2 min 41 s measured on a behaviour-room node, 2026-09-18).
        # The retirement itself is right and is left alone; only the cost of
        # rediscovering a fact already written in the row is removed. A row with
        # NO recorded path is NOT skipped: locate_video_file may still find its
        # file, which is how a node picks up work that really is here.
        from mousereach.watcher.db import WatcherDB as _WatcherDB
        _no_file_here = getattr(self.db, 'NO_FILE_HERE', _WatcherDB.NO_FILE_HERE)
        videos = [v for v in videos
                  if (v.get('current_path') or v.get('source_path')) != _no_file_here]
        stage = 'local_pipeline' if self.config.also_process else 'stage_to_nas'
        pick = self._pick_from_pool(videos, priority_animal, 'animal_id',
                                    randomize=False)
        if pick is not None:
            return {
                'type': stage,
                'id': pick['video_id'],
                'data': pick
            }

        # Priority 2: Videos queued for DLC (preferred tier first, random within tier)
        # ADMIT: running DLC is the expensive GPU analysis this node exists for.
        # Starting one on a deferred session is exactly the thing that must wait
        # until there is nothing else. Nothing strands: a queued video holds no
        # slot against a cap, and the next pass takes it the moment the node is
        # idle.
        videos = self.db.get_videos_in_state('dlc_queued')
        pick = self._pick_from_pool(videos, priority_animal, 'animal_id',
                                    allow_deferred=admit_deferred)
        if pick is not None:
            return {
                'type': 'single_dlc',
                'id': pick['video_id'],
                'data': pick
            }

        # Priority 2b: A video somebody left in the shared singles folder.
        # ADMIT: adopting one brings a new file onto this node and creates new
        # GPU work, so it waits behind the videos already queued.
        #
        # discover_new_singles records these as 'validated', and until
        # 2026-09-13 no job list selected that state: the file was noticed,
        # written into the database, and then ignored for good. Somebody
        # dropping a video where the folder layout says videos go is the most
        # ordinary thing a person can do, and it has to work.
        #
        # Two filters (both kept in memory): a single whose claim was just
        # refused waits out its backoff so the collages below still get cropped
        # (_note_single_claim_refused), and none are taken at all while this
        # node's recent poses keep failing (_note_pose_result).
        if self._singles_braked():
            videos = []
        else:
            videos = [v for v in self.db.get_videos_in_state('validated')
                      if not self._single_claim_backoff_active(v['video_id'])]
        pick = self._pick_from_pool(videos, priority_animal, 'animal_id',
                                    allow_deferred=admit_deferred)
        if pick is not None:
            return {
                'type': 'adopt_single',
                'id': pick['video_id'],
                'data': pick
            }

        # Priority 3: Crop next collage (Pillar first, random within tier)
        # Not offered at all when the coordination database failed to start:
        # the handler would refuse every one (no claim, no crop), and picking
        # a collage only to refuse it costs a copy of watcher.db to the share
        # each poll. Keyed on the RECORDED failure, so a node built without
        # the constructor keeps its ordinary selection.
        # The retry (_retry_coordinator) is rate-limited, so a node whose
        # database is still down pays one attempt every few minutes, not one
        # per poll.
        if (getattr(self, '_coordinator_init_error', None) is not None
                and not self._retry_coordinator()):
            self._warn_collage_claims_unavailable()
            return None
        self._retry_unsynced_collage_claims()
        # A collage another host is cropping is skipped until its recheck time
        # (_park_collage_claimed_elsewhere), so the loop moves on to others.
        collages = [c for c in self.db.get_collages_in_state('stable')
                    if not self._claim_backoff_active(c.get('filename'))]
        # ADMIT: cropping a collage creates new local files and new DLC work.
        pick = self._pick_from_pool(collages, priority_animal, 'animal_ids',
                                    is_collage=True,
                                    allow_deferred=admit_deferred)
        if pick is not None:
            return {
                'type': 'collage',
                'id': pick['filename'],
                'data': pick
            }

        return None

    # =========================================================================
    # DISPATCH
    # =========================================================================

    def _dispatch_work(self, work: dict):
        """Route work to appropriate handler."""
        work_type = work['type']
        work_id = work['id']

        try:
            ok = None
            if work_type == 'collage':
                ok = self._process_collage(work)
            elif work_type == 'adopt_single':
                ok = self._adopt_single_for_dlc(work)
            elif work_type == 'single_dlc':
                ok = self._process_single_dlc(work)
            elif work_type == 'stage_to_nas':
                ok = self._stage_to_nas(work)
            elif work_type == 'local_pipeline':
                ok = self._run_local_pipeline(work)
            elif work_type == 'archive_local':
                ok = self._archive_locally_processed(work)
            else:
                # A row producing unrecognised work would be re-selected
                # forever; fail it so the loop cannot spin on it.
                logger.warning(f"Unknown work type: {work_type} -- marking {work_id} failed")
                self.db.mark_failed(work_id, f"unknown work type: {work_type}")
                return False

            # Backup local DB to NAS after each successful work item
            self._backup_local_db()
            # A handler that explicitly declined (returned False) did no
            # work; the main loop must sleep, not spin (legacy handlers
            # returning None count as progress).
            return ok is not False

        except Exception as e:
            error_msg = f"{work_type} failed: {str(e)}"
            logger.error(f"Work item {work_id} failed: {e}", exc_info=True)

            if work_type == 'collage':
                self.db.update_collage_state(work_id, 'failed', validation_error=error_msg)
            elif not self._already_failed(work_id):
                self.db.mark_failed(work_id, error_msg)
            return False

    # =========================================================================
    # DRY RUN
    # =========================================================================

    def dry_run(self):
        """Scan without processing. Shows what would be done."""
        logger.info("DLCOrchestrator dry run")

        # Scan for new files
        scan_result = self.file_watcher.scan()
        print(f"\nScan results:")
        print(f"  New collages found:   {scan_result.new_collages}")
        print(f"  New singles found:    {scan_result.new_singles}")
        print(f"  Collages now stable:  {scan_result.stable_ready}")
        print(f"  Scan time:            {scan_result.scan_time_ms:.1f}ms")

        # Check for DLC completions
        dlc_completions = self._scan_for_dlc_completions()
        print(f"  DLC completions:      {dlc_completions}")

        # Show priority animal if set
        priority_animal = self._get_priority_animal()
        if priority_animal:
            print(f"\n  PRIORITY ANIMAL:      {priority_animal}")

        # Show the ordering policy, so nobody has to read the config file to
        # find out why one video is being taken before another.
        print("\nWork priority:")
        for line in self.work_priority.describe():
            print(f"  {line}")
        for complaint in self.work_priority.complaints:
            print(f"  [!] {complaint}")

        # Show what work would be done
        print(f"\nPending work items:")
        count = 0
        for state_label, state in [
            ("Collages to crop", "stable"),
            ("Videos for DLC", "dlc_queued"),
            ("Videos to stage to NAS", "dlc_complete"),
        ]:
            if state in ("stable",):
                items = self.db.get_collages_in_state(state)
            else:
                items = self.db.get_videos_in_state(state)
            if items:
                print(f"  {state_label}: {len(items)}")
                for item in items[:5]:
                    name = item.get('filename') or item.get('video_id')
                    print(f"    - {name}")
                if len(items) > 5:
                    print(f"    ... and {len(items) - 5} more")
                count += len(items)

        if count == 0:
            print("  (no pending work)")

        print(f"\nStaging directory: {self.staging_dir or '(not configured)'}")
        print()

    # =========================================================================
    # DLC COMPLETION SCANNING
    # =========================================================================

    def _scan_for_dlc_completions(self) -> int:
        """
        Scan for h5 files matching videos in dlc_queued or dlc_running state.

        DLC outputs h5 files to DLC_Queue (same dir as input video).
        This detects them and advances state to dlc_complete.
        """
        dlc_queue_dir = Paths.DLC_QUEUE
        if not dlc_queue_dir or not dlc_queue_dir.exists():
            return 0

        count = 0
        from mousereach.watcher import repose
        declared = repose.declared_scorer()

        for state in ('dlc_queued', 'dlc_running'):
            videos = self.db.get_videos_in_state(state)
            for video in videos:
                video_id = video['video_id']
                h5_files = list(dlc_queue_dir.glob(f"{video_id}DLC*.h5"))
                if not h5_files:
                    continue
                # A video queued to be RE-posed is only complete when a pose
                # from the declared model exists. An older pose beside it (a
                # leftover from an interrupted stage) used to count as done,
                # and the old pose was staged as the "new" one.
                declared_hits = ([h for h in h5_files if repose.scorer_of(h) == declared]
                                 if declared else h5_files)
                if declared_hits:
                    dlc_path = select_pose_file(declared_hits, expected_scorer=declared or None)
                elif (video.get('mark_reason') or '').startswith(repose.REASON_PREFIX):
                    continue                    # still waiting for the new pose
                else:
                    dlc_path = select_pose_file(h5_files)
                if dlc_path:
                    logger.info(f"DLC completion detected: {video_id} -> {dlc_path.name}")

                    if state == 'dlc_queued':
                        self.db.update_state(video_id, 'dlc_running')
                    self.db.update_state(
                        video_id, 'dlc_complete',
                        dlc_output_path=str(dlc_path),
                        current_path=str(dlc_queue_dir / f"{video_id}.mp4")
                    )
                    self.db.log_step(video_id, 'dlc', 'completed', message=dlc_path.name)
                    count += 1

        return count

    # =========================================================================
    # COLLAGE PROCESSING
    # =========================================================================

    def _expected_children(self, collage_filename: str) -> list:
        """The collage's non-blank children, from its filename alone:
        ``[{position, animal_id, offspring_stem}]``. [] when the name cannot be
        parsed (the crop then decides, exactly as before).

        The cropper names each child ``{date}_{animal_id}_{last}.mp4`` from the
        collage name, so a child's id is known BEFORE cropping -- which is what
        lets a collage whose children are all finished be skipped without
        spending a crop on it.
        """
        try:
            from mousereach.video_prep.core.collage_provenance import expected_offspring
            return [c for c in expected_offspring(collage_filename)
                    if not c.get('blank') and c.get('offspring_stem')]
        except Exception as e:
            logger.debug(f"could not derive the children of {collage_filename}: {e}")
            return []

    def _record_collage_needs_no_crop(self, collage_filename: str, collage_data: dict,
                                      children: list, verdicts: dict) -> bool:
        """Every child is finished or held: record that instead of cropping.

        Each child row is set to its real state, the collage is marked
        'cropped' here with the reason in its processing log, and the shared
        claim this node holds is marked 'cropped' too, so no other node crops
        it either. Returns True: the collage has been dealt with and its row
        has left 'stable', so the loop cannot pick it again.
        """
        recorded = 0
        for child in children:
            stem = child['offspring_stem']
            try:
                if self._record_finished_child(
                        stem, verdicts[stem], collage_filename,
                        self._child_metadata(stem, child.get('animal_id', ''),
                                             child.get('position'), collage_data)):
                    recorded += 1
            except Exception as e:
                logger.warning(f"{stem}: {verdicts[stem][1]}, but its row could not "
                               f"be recorded ({e})")
        reason = (f"not cropped: all {len(children)} expected child video(s) are "
                  f"already analysed or held for review "
                  f"({recorded} row(s) corrected on this node)")
        self.db.force_collage_state(collage_filename, 'cropped',
                                    videos_created=0, videos_skipped=len(children))
        self.db.log_step(collage_filename, 'crop', 'skipped', message=reason)
        logger.info(f"Collage {collage_filename}: {reason}")
        try:
            if self.coordinator.update_collage_state(
                    collage_filename, 'cropped', only_if_held_by=self.hostname,
                    singles_created=0) is False:
                self._warn_claim_lost_while_cropping(collage_filename)
        except Exception as e:
            self._note_unsynced_collage_claim(collage_filename, 0, e)
        return True

    def _process_collage(self, work: dict):
        """Copy collage from NAS to local, crop into singles, queue for DLC.

        Returns False whenever this node did nothing with the collage (no
        claim, claim lost, claim check failed) so the main loop sleeps instead
        of re-picking it at full speed; True when the collage was cropped or
        recorded as needing no crop. A failed crop raises, after the shared
        claim is released so another attempt is possible.
        """
        from mousereach.video_prep.core.cropper import crop_collage

        collage_filename = work['id']
        collage_data = work['data']

        logger.info(f"Processing collage: {collage_filename}")

        # Cross-PC dedup: hold the shared claim before anything else. Fails
        # closed -- see _hold_collage_claim.
        if not self._hold_collage_claim(collage_filename):
            return False

        # A collage whose children are ALL finished or held needs no crop.
        # Asked before cropping because the children's ids follow from the
        # collage name; a node with a fresh or partial database would otherwise
        # re-crop and re-pose finished work (see _already_done_or_held).
        children = self._expected_children(collage_filename)
        if children:
            verdicts = {c['offspring_stem']: self._already_done_or_held(c['offspring_stem'])
                        for c in children}
            if all(verdicts.values()):
                return self._record_collage_needs_no_crop(
                    collage_filename, collage_data, children, verdicts)

        self.db.update_collage_state(collage_filename, 'cropping')
        self.db.log_step(collage_filename, 'crop', 'started')

        start_time = time.time()
        # Before the try: a failed crop's cleanup reads it to decide whether
        # the shared claim may be released (_release_claim_after_failed_crop).
        videos_created = 0

        try:
            # Source path on NAS (D:)
            source_path = Path(collage_data['source_path'])
            if not source_path.exists():
                raise FileNotFoundError(f"Collage not found: {source_path}")

            # Copy collage to local working dir (A:)
            local_collage = self.working_dir / source_path.name
            logger.info(f"Copying collage to working dir: {source_path} -> {local_collage}")

            if not safe_copy(source_path, local_collage, verify=True):
                raise IOError(f"Failed to copy collage to working directory")

            # Crop collage to singles. Two ways to run it, exactly as the pose
            # has: with no recording programs configured, in this process as
            # before; with programs configured, in a child process this node can
            # stop part-way (video_prep/core/crop_interruptible.py), asking the
            # recording guard every couple of seconds. WHY: ffmpeg runs once per
            # animal for minutes, and an operator who opened the recording
            # program should not have to wait for that.
            logger.info(f"Cropping collage: {collage_filename}")
            if getattr(self.config, 'pause_while_running', None):
                from mousereach.video_prep.core.crop_interruptible import (
                    run_crop_collage_interruptible)
                crop_outcome = run_crop_collage_interruptible(
                    input_path=local_collage,
                    output_dir=self.working_dir,
                    should_abort=self._recording_abort_reason,
                )
                if crop_outcome.get('status') == 'aborted':
                    raise _CropAborted(crop_outcome.get('abort_reason') or 'stopped part-way')
                if crop_outcome.get('status') != 'success':
                    raise RuntimeError(f"crop failed: {crop_outcome.get('error')}")
                crop_results = crop_outcome.get('results') or []
            else:
                crop_results = crop_collage(
                    input_path=local_collage,
                    output_dir=self.working_dir,
                    verbose=False
                )

            # Register each cropped single and move to DLC_Queue
            videos_created = 0
            videos_skipped = 0

            for result in crop_results:
                if result['status'] == 'skipped':
                    videos_skipped += 1
                    logger.debug(f"Skipped position {result['position']}: {result.get('animal_id', '?')} (blank)")
                    continue

                if result['status'] != 'success':
                    logger.warning(f"Crop failed for position {result['position']}: {result.get('error')}")
                    continue

                output_path = Path(result['output_path'])
                video_id = get_video_id(output_path.name)

                animal_id = result.get('animal_id', '')
                metadata = self._child_metadata(video_id, animal_id,
                                                result.get('position'), collage_data)

                # A child that is finished, or held for a person, is never
                # queued for DLC: its row is recorded in its real state instead
                # (see _already_done_or_held). Checked again here, per child,
                # because the collage as a whole still needed cropping.
                verdict = self._already_done_or_held(video_id)
                if verdict:
                    self._record_finished_child(video_id, verdict, collage_filename,
                                                metadata)
                    videos_skipped += 1
                    continue

                # Register video in DB
                self.db.register_video(
                    video_id=video_id,
                    source_path=str(output_path),
                    collage_id=collage_filename,
                    current_path=str(output_path),
                    **metadata
                )

                # A collage can be re-claimed after its children have already been
                # cropped and run -- e.g. the collage file is still sitting in the
                # intake folder from a previous pass. Those children are finished
                # work. Re-validating one raises on the transition (archived ->
                # validated is not permitted), and PERMITTING that transition would
                # be worse than the crash: it would silently reset finished videos
                # and re-queue them for DLC. So leave any child that has already
                # moved past discovery alone. 'failed' is the one prior state worth
                # re-driving, and failed -> validated is already a legal retry.
                # 'validated' MUST be re-driven, not skipped. It means the child was
                # cropped and registered but the DLC_Queue copy never succeeded --
                # and the cleanup below deletes the working-dir crop regardless, so
                # a 'validated' child has no file anywhere. Re-cropping is the only
                # way to recover it. (Colin's DLC PC had 342 stuck exactly here.)
                prior_state = (self.db.get_video(video_id) or {}).get('state')
                if prior_state and prior_state not in ('discovered', 'validated', 'failed'):
                    videos_skipped += 1
                    logger.info(
                        f"Skipping {video_id}: already at '{prior_state}' "
                        f"(collage re-claimed; child already processed)")
                    continue

                self.db.update_state(video_id, 'validated', current_path=str(output_path))

                # Move single to DLC_Queue on local drive (A:)
                if Paths.DLC_QUEUE:
                    Paths.DLC_QUEUE.mkdir(parents=True, exist_ok=True)
                    dlc_queue_path = Paths.DLC_QUEUE / output_path.name
                    if safe_copy(output_path, dlc_queue_path, verify=True):
                        self.db.update_state(video_id, 'dlc_queued', current_path=str(dlc_queue_path))
                        # Put the child on record in the shared table now, not
                        # at its first pose. WHY: a claim left in 'cropping'
                        # is refused for takeover while its collage has
                        # children on record (coordination.try_claim_collage);
                        # a child first synced at 'dlc_running' was invisible
                        # for as long as it waited in this node's queue.
                        self._sync_to_connectome(video_id, 'dlc_queued',
                                                 collage_id=collage_filename,
                                                 source_path=str(dlc_queue_path))
                        logger.info(f"Created and queued: {video_id}")
                        videos_created += 1
                    else:
                        logger.error(f"Failed to copy {video_id} to DLC_Queue")
                else:
                    logger.error("DLC_QUEUE path not configured")

            # Update collage state
            duration = time.time() - start_time
            self.db.update_collage_state(
                collage_filename,
                'cropped',
                videos_created=videos_created,
                videos_skipped=videos_skipped
            )
            self.db.log_step(
                collage_filename,
                'crop',
                'completed',
                message=f"Created {videos_created} singles, skipped {videos_skipped}",
                duration=duration
            )

            logger.info(f"Collage cropped: {collage_filename} ({videos_created} singles, {videos_skipped} skipped)")

            # Sync collage completion to the shared coordination database. Not
            # best-effort any more: a claim left in 'cropping' can be taken over
            # after a day, so a lost 'cropped' is retried until it lands, and a
            # claim that turns out to be held by another host is said out loud.
            if self.coordinator:
                try:
                    if self.coordinator.update_collage_state(
                            collage_filename, 'cropped', only_if_held_by=self.hostname,
                            singles_created=videos_created) is False:
                        self._warn_claim_lost_while_cropping(collage_filename)
                except Exception as e:
                    self._note_unsynced_collage_claim(collage_filename, videos_created, e)

            # Cleanup working directory
            local_collage.unlink(missing_ok=True)
            for result in crop_results:
                if result['status'] == 'success':
                    Path(result['output_path']).unlink(missing_ok=True)

        except _CropAborted as stopped:
            # Caught BEFORE the generic handler: stopping for a recording is not
            # a failure of this collage. Its half-made singles are already
            # deleted; the local copy of the collage goes too, and the collage
            # waits to be cropped again. The shared claim is KEPT (this node
            # comes back to it, and nothing of it exists anywhere else).
            duration = time.time() - start_time
            local = locals().get('local_collage')
            if local is not None:
                Path(local).unlink(missing_ok=True)
            try:
                # force_: 'cropping' may only go to 'cropped' or 'failed', and
                # this is neither -- the collage is simply waiting again.
                self.db.force_collage_state(collage_filename, 'stable')
                self.db.log_step(collage_filename, 'crop', 'aborted',
                                 message=str(stopped), duration=duration)
            except Exception as e:
                logger.error(f"Collage {collage_filename}: crop was stopped ({stopped}) "
                             f"but the row could not be put back to waiting "
                             f"({type(e).__name__}: {e}); the next watcher start "
                             f"recovers it")
            logger.info(f"Collage {collage_filename}: crop stopped after {duration:.0f} s "
                        f"because {stopped}; partial singles removed, waiting to be "
                        f"cropped again (not counted as a failure)")
            return False

        except Exception as e:
            duration = time.time() - start_time
            try:
                self.db.update_collage_state(collage_filename, 'failed', validation_error=str(e))
                self.db.log_step(collage_filename, 'crop', 'failed', message=str(e), duration=duration)
            finally:
                self._release_claim_after_failed_crop(collage_filename, videos_created)
            raise
        return True

    def _release_claim_after_failed_crop(self, collage_filename: str,
                                         videos_created: int) -> None:
        """Give the shared claim back after this node's crop failed.

        WHY: a claim used to outlive a failed crop forever, so the one node
        that could not crop a collage stopped every other node from trying.
        Released only when this node carries no child of the collage: none
        queued by this attempt, and none left in flight here by an earlier one
        (_collage_children_in_flight_here). Once a child is on this node it
        will be posed here, and letting another node re-crop the collage would
        pose that child a second time. Such a claim is KEPT: this node retries
        its own failed collage (discover_new_collages re-validates it), and no
        other node takes the claim over while those children are on record in
        the shared table (coordination.try_claim_collage). Never raises: the
        crop's own error is the one to report.
        """
        coordinator = getattr(self, 'coordinator', None)
        if coordinator is None:
            return
        try:
            carried = self._collage_children_in_flight_here(collage_filename)
        except Exception as e:
            logger.warning(f"Collage {collage_filename}: crop failed and this node's "
                           f"children of it could not be counted ({e}); keeping the "
                           f"shared claim")
            return
        if videos_created or carried:
            logger.warning(
                f"Collage {collage_filename}: crop failed while this node carries "
                f"{max(videos_created, len(carried))} child video(s) of it; keeping "
                f"the shared claim so no other node crops and poses them again. This "
                f"node retries the collage; if it cannot, a person decides.")
            return
        try:
            coordinator.release_collage_claim(collage_filename, self.hostname)
        except Exception as e:
            logger.warning(f"Collage {collage_filename}: crop failed and the shared "
                           f"claim could not be released ({type(e).__name__}: {e}); "
                           f"other nodes can take it over once it goes stale")

    # =========================================================================
    # DLC INFERENCE
    # =========================================================================

    def _process_single_dlc(self, work: dict):
        """Run DLC inference on a single video.

        Returns True when a pose was produced, False when nothing was run
        (refused, no file here, node not configured) so the main loop sleeps
        instead of re-picking the row at full speed. A failed run raises.
        """
        from mousereach.dlc.core import run_dlc_batch, resolve_dlc_shuffle
        from mousereach.watcher import repose

        video_id = work['id']
        video_data = work['data']

        # Last line of defence against posing finished or held work: rows can
        # reach 'dlc_queued' without passing the crop or adopt checks (queued
        # before those checks existed, or found in DLC_Queue at startup).
        #
        # A RE-POSE REQUEST is the exception for exactly the states the re-pose
        # consumer re-drives (repose.REDRIVE_STATES: an archived video, or one
        # in deep review whose fix needs a new pose) -- posing an already
        # finished video again is what the request asks for. A triage hold
        # stays refused, as the consumer refuses it.
        verdict = self._already_done_or_held(video_id)
        if verdict:
            state, why = verdict
            requested = (video_data.get('mark_reason') or '').startswith(repose.REASON_PREFIX)
            if requested and state in repose.REDRIVE_STATES:
                logger.info(f"{video_id}: {why}, but a re-pose was requested; posing it")
            else:
                self.db.force_state(video_id, state,
                                    reason=f"{why}; not posed again")
                self.db.log_step(video_id, 'dlc', 'skipped', message=why)
                local = (Path(Paths.DLC_QUEUE) / f"{video_id}.mp4") if Paths.DLC_QUEUE else None
                if local is not None and local.is_file():
                    # Not deleted automatically: the finished or held copy lives
                    # elsewhere, but a hold is judged from a folder's existence,
                    # and a leftover empty queue folder must not be enough to
                    # destroy a crop. Named, so a person can free the space.
                    logger.warning(f"{video_id}: {why}; not posing it. Its copy "
                                   f"{local} is not needed for that and is NOT "
                                   f"removed automatically; delete it by hand to "
                                   f"free the space.")
                else:
                    logger.info(f"{video_id}: {why}; not posing it")
                return False

        # Same pathless-row hazard as _stage_to_nas, and the same reason for
        # search_archive=False: DLC writes its .h5 beside its input, so resolving
        # to the archived copy would drop new pose files into the archive.
        # search_staging=False for the same reason one folder over: a video in
        # Processing/Posed is the processing server's intake, and posing it
        # there would write a pose into the shared hand-off folder.
        current_path = locate_video_file(
            video_id, raw=video_data.get('current_path'),
            extra_dirs=[Paths.DLC_QUEUE], search_archive=False,
            search_staging=False)

        if current_path is None:
            self.db.mark_unresolvable(
                video_id,
                "queued for DLC but no video file for it on this node "
                "(recorded path: %r)" % (video_data.get('current_path'),))
            self._release_claim_given_up(video_id, "no local copy left to pose")
            return False

        logger.info(f"Running DLC on {video_id}")

        # Resolve the model up front. An unresolvable shuffle is a problem with
        # this NODE's configuration, not with this video, so it must not consume
        # the video's retry budget -- leave it queued, exactly as an unset
        # dlc_config_path does below.
        try:
            shuffle, _ = resolve_dlc_shuffle()
        except ValueError as e:
            logger.error(
                f"{e}\n"
                f"{video_id} stays in dlc_queued -- no video is marked failed for "
                f"a node configuration problem. DLC is stopped on this node until "
                f"this is fixed."
            )
            return False

        if not self.config.dlc_config_path:
            logger.warning(
                f"DLC config not configured - {video_id} stays in dlc_queued. "
                "Run 'mousereach-setup' to set DLC model path."
            )
            return False

        dlc_config = Path(self.config.dlc_config_path)
        if not dlc_config.exists():
            if dlc_config.is_dir():
                dlc_config = dlc_config / "config.yaml"
            if not dlc_config.exists():
                logger.error(f"DLC config not found: {self.config.dlc_config_path}")
                return False

        if not current_path.exists():
            logger.error(f"Video file not found: {current_path}")
            self.db.mark_failed(video_id, f"Video file not found: {current_path}")
            return False

        self.db.update_state(video_id, 'dlc_running')
        self.db.log_step(video_id, 'dlc', 'started', message=f"GPU {self.config.dlc_gpu_device}")
        # Tell the shared record this node is working on it, so a restart's
        # cross-node recovery does not "advance" the row back to an older
        # 'archived' verdict from before a re-pose.
        self._sync_to_connectome(video_id, 'dlc_running',
                                 source_path=str(current_path))

        start_time = time.time()

        # Two ways to run the pose. With no recording programs configured,
        # exactly as before: in this process. With programs configured, in a
        # child process this node can stop part-way (dlc/core/interruptible.py),
        # asking the recording guard every few seconds. WHY not always the
        # child: in-process is the long-proven path, and a node that never
        # records has nothing to stop for.
        pause_names = getattr(self.config, 'pause_while_running', None) or []

        try:
            # DLC outputs to same directory as input (DLC_Queue)
            dlc_output_dir = current_path.parent
            dlc_output_dir.mkdir(parents=True, exist_ok=True)

            if pause_names:
                from mousereach.dlc.core.interruptible import run_dlc_single_interruptible
                result = run_dlc_single_interruptible(
                    video_path=current_path,
                    config_path=dlc_config,
                    output_dir=dlc_output_dir,
                    gpu=self.config.dlc_gpu_device,
                    shuffle=shuffle,
                    should_abort=self._recording_abort_reason,
                )
                if result and result.get('status') == 'aborted':
                    raise _PoseAborted(result.get('abort_reason') or 'stopped part-way')
                results = [result] if result else []
            else:
                results = run_dlc_batch(
                    video_paths=[current_path],
                    config_path=dlc_config,
                    output_dir=dlc_output_dir,
                    gpu=self.config.dlc_gpu_device,
                    save_as_csv=True,
                    shuffle=shuffle
                )

            duration = time.time() - start_time

            if results and results[0].get('status') == 'success':
                from mousereach.watcher import repose
                declared = repose.declared_scorer()
                chosen_h5 = select_pose_file(
                    dlc_output_dir.glob(f"{video_id}DLC*.h5"),
                    expected_scorer=declared or None)
                dlc_output = str(chosen_h5) if chosen_h5 else None
                # A video re-posed on REQUEST must come back from the declared
                # model; a node whose DLC project produces another scorer
                # would stage a pose nobody adopts, and be asked again a day
                # later, forever. Say so once instead.
                if ((video_data.get('mark_reason') or '').startswith(repose.REASON_PREFIX)
                        and declared and chosen_h5 is not None
                        and repose.scorer_of(chosen_h5) != declared):
                    raise RuntimeError(
                        f"this node's DLC model produced {repose.scorer_of(chosen_h5)} "
                        f"but the declared model is {declared}; check dlc_config_path "
                        f"/ dlc_shuffle on this node")

                self.db.update_state(
                    video_id, 'dlc_complete',
                    dlc_output_path=dlc_output,
                    current_path=str(current_path)
                )
                self.db.log_step(
                    video_id, 'dlc', 'completed',
                    message=f"GPU {self.config.dlc_gpu_device}",
                    duration=duration
                )
                logger.info(f"DLC completed for {video_id} ({duration:.1f}s)")
                self._note_pose_result(True)
                self._sync_to_connectome(video_id, 'dlc_complete',
                                         dlc_completed_at=datetime.now().isoformat())
            else:
                error_msg = results[0].get('error', 'Unknown DLC error') if results else 'No results'
                raise RuntimeError(f"DLC failed: {error_msg}")

        except _PoseAborted as stopped:
            # Caught BEFORE the generic handler: stopping for a recording is
            # not a failure of this video and must not spend its retries.
            return self._requeue_aborted_pose(video_id, current_path, str(stopped),
                                              time.time() - start_time)

        except Exception as e:
            duration = time.time() - start_time
            self.db.mark_failed(video_id, str(e))
            self.db.log_step(video_id, 'dlc', 'failed', message=str(e), duration=duration)
            # A failed pose gives a claimed single back to the folder at once,
            # and counts towards the brake on taking new singles. The GPU role
            # never retries a 'failed' row by itself, so holding the claim
            # would keep the video from every healthy node.
            self._release_claim_given_up(video_id, f"pose failed here ({e})")
            self._note_pose_result(False)
            # If a re-pose was requested over shared storage, leave the
            # requester a note there: its own row just says "waiting".
            try:
                from mousereach.watcher import repose
                repose.note_failure(video_id, self.hostname, str(e))
            except Exception:
                pass
            raise
        return True

    def _requeue_aborted_pose(self, video_id: str, current_path: Path,
                              reason: str, duration: float) -> bool:
        """Put a pose that was stopped for a recording back in the queue.

        The row goes back to 'dlc_queued' with its file where it was, the
        audit trail says 'aborted' and why, and the shared record is told it is
        queued again. NOT marked failed and error_count NOT touched: the video
        did nothing wrong, and a busy recording week would otherwise use up its
        retries and park it as failed. The partial pose files were already
        removed by the interruptible runner.

        Always returns False (no pose was produced), so the main loop sleeps
        instead of re-picking the row at once -- and on its next pass finds the
        watcher paused.
        """
        from mousereach.watcher.db import VIDEO_TRANSITIONS
        why = f"pose stopped part-way ({reason}); back in the DLC queue, not a failure"
        try:
            row = self.db.get_video(video_id) or {}
            if 'dlc_queued' in VIDEO_TRANSITIONS.get(row.get('state'), []):
                self.db.update_state(video_id, 'dlc_queued', current_path=str(current_path))
            else:
                self.db.force_state(video_id, 'dlc_queued', reason=why,
                                    current_path=str(current_path))
            self.db.log_step(video_id, 'dlc', 'aborted', message=reason, duration=duration)
        except Exception as e:
            # Left in 'dlc_running', which the next watcher start reclaims to
            # 'dlc_queued' (_ORPHANED_VIDEO_STATES) -- still never 'failed'.
            logger.error(f"{video_id}: pose was stopped ({reason}) but the row could "
                         f"not be put back in the queue ({type(e).__name__}: {e}); "
                         f"restarting the watcher returns it to the queue")
            return False
        self._sync_to_connectome(video_id, 'dlc_queued', source_path=str(current_path))
        logger.info(f"{video_id}: pose stopped after {duration:.0f} s because {reason}; "
                    f"partial output removed, back in the DLC queue (not counted as "
                    f"a failure)")
        return False

    # =========================================================================
    # LOCAL PIPELINE (also_process mode)
    # =========================================================================

    def _run_local_pipeline(self, work: dict):
        """Run seg/reach/outcomes locally after DLC, then archive directly.

        When also_process=True, DLC PCs run the full pipeline locally instead
        of staging to NAS for the processing server. Reuses the same pipeline
        functions as ProcessingOrchestrator._run_pipeline().

        Returns False when nothing was run (no pose or video here, a file could
        not be staged locally, segmentation could not start), True once the
        pipeline ran -- whether the video then went to review or to 'processed'.
        """
        from mousereach.segmentation.core.batch import process_single as seg_single
        from mousereach.reach.core.batch import process_single as reach_single
        from mousereach.outcomes.core.batch import process_single as outcome_single
        from mousereach.assignment.run import assign_reaches_for_video
        from mousereach.pipeline.manifest import create_processing_manifest
        from mousereach.pipeline.triage import triage_video
        from mousereach.watcher.review_gate import run_gate, route_deep_review, DECISION_CLEAN

        video_id = work['id']
        video_data = work['data']

        # dlc_output_path has bitten this pipeline twice, in opposite directions:
        # Path(None) raised and failed 950 of 954 videos on the DLC PC, and the
        # `or ''` fix for it made Path('') -> Path('.'), which exists, so the
        # missing-pose guard stopped firing and 723 videos were routed to human
        # review with "[Errno 13] Permission denied: '.'". locate_pose_file
        # (through resolve_pose_input) returns a real file or None and never a
        # placeholder path.
        #
        # search_staging=False: a recorded pose path inside Processing/Posed is
        # ignored. That folder is the processing server's intake; running this
        # node's pipeline on a pose found there would process a hand-off the
        # server is taking in at the same time. search_archive=False keeps the
        # old scope (the archive was never searched here).
        dlc_path = locate_pose_file(
            video_id, raw=video_data.get('dlc_output_path'),
            extra_dirs=[Paths.DLC_QUEUE], search_archive=False,
            search_staging=False)
        if dlc_path is None:
            self.db.mark_failed(video_id, f"DLC h5 not found for {video_id}")
            return False

        # Stage outputs land BESIDE the pose file, and this function used to
        # make the pose's own folder the working directory -- so any video
        # whose pose resolved into the archive's model tree scattered its
        # results there while the review bundle was assembled from stale
        # copies in Processing (reviewer diagnosis, 2026-09-08: a corrected
        # video's PERFECT re-detection sat in the pose tree while the queue
        # showed August data with 120 reaches outside their segments). Same
        # defect as _run_pipeline's, same cure: stage the pose locally,
        # always, and work in the local processing dir.
        processing_dir = Path(Paths.PROCESSING)
        processing_dir.mkdir(parents=True, exist_ok=True)
        if Path(dlc_path).parent != processing_dir:
            local = processing_dir / Path(dlc_path).name
            if not local.exists():
                from mousereach.watcher.transfer import safe_copy
                if not safe_copy(Path(dlc_path), local, verify=True):
                    self.db.mark_failed(
                        video_id, f"could not stage pose locally from {dlc_path}")
                    return False
            dlc_path = local
        # The video must sit beside the pose too: the review gate opens it
        # from here, a deep-review bundle is built from here, and the archive
        # step files from here. Without this copy the gate saw no video and
        # the archive filed a bundle with no results (2026-09-12).
        # search_staging=False, as for the pose above: never Processing/Posed.
        local_mp4 = processing_dir / f"{video_id}.mp4"
        if not local_mp4.exists():
            src_mp4 = locate_video_file(
                video_id, raw=video_data.get('current_path'),
                extra_dirs=[Paths.DLC_QUEUE, Path(dlc_path).parent],
                search_archive=False, search_staging=False)
            if src_mp4 is None:
                self.db.mark_failed(
                    video_id, "video file not found on this node for the local "
                              "pipeline (recorded path: %r)" % (video_data.get('current_path'),))
                return False
            if Path(src_mp4) != local_mp4:
                from mousereach.watcher.transfer import safe_copy
                if not safe_copy(Path(src_mp4), local_mp4, verify=True):
                    self.db.mark_failed(
                        video_id, f"could not stage video locally from {src_mp4}")
                    return False
        logger.info(f"Running local pipeline on {video_id} (also_process mode)")

        self.db.update_state(video_id, 'processing')
        self.db.log_step(video_id, 'local_pipeline', 'started')
        pipeline_start = time.time()

        try:
            # Step 1: Segmentation
            self.db.log_step(video_id, 'segmentation', 'started')
            step_start = time.time()
            seg_result = seg_single(dlc_path)
            seg_duration = time.time() - step_start

            if seg_result.get('success', False):
                self.db.log_step(video_id, 'segmentation', 'completed',
                                message=f"boundaries={seg_result.get('n_boundaries', 0)}",
                                duration=seg_duration)
                logger.info(f"Segmentation complete: {video_id} ({seg_duration:.1f}s)")
            else:
                error = seg_result.get('error', 'segmentation failed')
                self.db.log_step(video_id, 'segmentation', 'failed', message=error, duration=seg_duration)
                # Segmentation that COULD NOT RUN is an infrastructure failure, not
                # something to put in front of a reviewer -- see
                # segmentation_could_not_run().
                if segmentation_could_not_run(error, dlc_path):
                    logger.error(
                        f"Segmentation could not run for {video_id} ({error}). "
                        f"Marking failed -- not routing to human review."
                    )
                    self.db.mark_failed(video_id, f"Segmentation could not run: {error}")
                    return False
                # A real seg failure -> DEEP review, not a dead 'failed'. Move the
                # whole bundle out of Processing so a human re-segments it.
                try:
                    route_deep_review(
                        video_id, processing_dir,
                        f"segmentation_failed: {error}", db=self.db,
                        extra_sources=[processing_dir / f"{video_id}.mp4"],
                    )
                except Exception as route_err:
                    logger.warning(f"Deep-review routing failed for {video_id}: {route_err}")
                    self.db.mark_failed(video_id, f"Segmentation failed: {error}")
                    return False
                # The bundle carries the video now; a shared copy this node
                # claimed for it is a duplicate (kept if that is unconfirmed).
                self._retire_claim_after_hold(video_id)
                return True

            # Step 2: Reach Detection
            seg_path = processing_dir / f"{video_id}_segments.json"
            if not seg_path.exists():
                self.db.mark_failed(video_id, "Segments file not created")
                return False

            self.db.log_step(video_id, 'reach_detection', 'started')
            step_start = time.time()
            reach_result = reach_single(dlc_path, seg_path)
            reach_duration = time.time() - step_start
            self.db.log_step(video_id, 'reach_detection', 'completed',
                            message=f"reaches={reach_result.get('total_reaches', 0)}",
                            duration=reach_duration)
            logger.info(f"Reach detection complete: {video_id} ({reach_duration:.1f}s)")

            # Step 3: Outcome Detection (skip for E/F trays)
            tray_info = parse_tray_type(f"{video_id}.mp4")
            tray_type = tray_info.get('tray_type', 'P')
            skip_outcomes = tray_type in ('E', 'F')

            if not skip_outcomes:
                reach_path = processing_dir / f"{video_id}_reaches.json"
                self.db.log_step(video_id, 'outcome_detection', 'started')
                step_start = time.time()
                outcome_result = outcome_single(dlc_path, seg_path, reach_path)
                outcome_duration = time.time() - step_start
                self.db.log_step(video_id, 'outcome_detection', 'completed',
                                message=f"segments={outcome_result.get('n_segments', 0)}",
                                duration=outcome_duration)
                logger.info(f"Outcome detection complete: {video_id} ({outcome_duration:.1f}s)")

            # Step 3.5: Reach Assignment (algo-4) -- causal-reach attribution.
            # Runs before the gate (a touched segment with no committed causal
            # reach counts as triaged).
            if not skip_outcomes:
                try:
                    assign_reaches_for_video(processing_dir, video_id, dlc_path)
                except Exception as e:
                    # Countable, like every other stage. Without this row the only trace of a
                    # failure was a started/completed count mismatch -- and a failed assignment
                    # writes no file, which silently switches off the gate's "touched pellet
                    # with no credited reach" hold. State deliberately NOT changed here; see
                    # docs/UNFINISHED.md before making this fail-closed.
                    self.db.log_step(video_id, 'assignment', 'failed', message=str(e))
                    logger.warning(f"Assignment (algo-4) failed for {video_id}: {e}")

            # Generate provenance manifest (travels with the bundle if held)
            pipeline_duration = time.time() - pipeline_start
            try:
                step_timestamps = {
                    'pipeline_started_at': datetime.fromtimestamp(pipeline_start).isoformat(),
                    'pipeline_completed_at': datetime.now().isoformat(),
                }
                create_processing_manifest(
                    video_id=video_id,
                    processing_dir=processing_dir,
                    dlc_path=dlc_path,
                    step_timestamps=step_timestamps,
                )
            except Exception as e:
                self._set_aside_stale_manifest(processing_dir, video_id, e)

            # Unified QC triage
            qc_verdict = 'auto_approved'
            try:
                triage_result = triage_video(
                    video_id=video_id,
                    processing_dir=processing_dir,
                    h5_path=dlc_path,
                )
                qc_verdict = triage_result.verdict
                for suffix in ['_segments.json', '_reaches.json', '_pellet_outcomes.json']:
                    json_path = processing_dir / f"{video_id}{suffix}"
                    if json_path.exists():
                        try:
                            with open(json_path) as f:
                                data = json.load(f)
                            data['validation_status'] = triage_result.verdict
                            data['triage_reason'] = (
                                '; '.join(f.description for f in triage_result.flags if f.severity == 'critical')
                                if triage_result.verdict == 'needs_review'
                                else 'Unified triage: all checks passed'
                            )
                            with open(json_path, 'w') as f:
                                json.dump(data, f, indent=2)
                        except Exception:
                            pass
                triage_result.save(processing_dir / f"{video_id}_triage.json")
            except Exception as e:
                logger.warning(f"Unified triage failed for {video_id}: {e}")

            # GATE: nothing reaches kinematics / connectome.db until it is CLEAN.
            # seg soft-fail / QC-critical -> DEEP_REVIEW; unresolved triage ->
            # TRIAGE. A held video's whole bundle is MOVED out of Processing; STOP.
            decision = run_gate(
                video_id, processing_dir, self.db,
                qc_verdict=qc_verdict,
                mp4_path=processing_dir / f"{video_id}.mp4",
            )
            if decision != DECISION_CLEAN:
                logger.info(f"Local pipeline held: {video_id} -> {decision} "
                            f"(no kinematics until cleared)")
                self._retire_claim_after_hold(video_id)
                return True

            # Step 4: Feature Extraction + DB sync (CLEAN videos ONLY)
            if not skip_outcomes:
                reach_path = processing_dir / f"{video_id}_reaches.json"
                outcome_path = processing_dir / f"{video_id}_pellet_outcomes.json"
                if reach_path.exists() and outcome_path.exists():
                    try:
                        from mousereach.kinematics.core.feature_extractor import FeatureExtractor
                        from mousereach.review.causal_review_io import resolve_review_path
                        extractor = FeatureExtractor()
                        review_path = resolve_review_path(video_id, processing_dir)
                        if review_path is not None:
                            logger.info(f"Applying human review corrections: {review_path.name}")
                        features = extractor.extract(dlc_path, reach_path, outcome_path,
                                                     review_path=review_path)
                        features_path = processing_dir / f"{video_id}_features.json"
                        with open(features_path, 'w') as f:
                            json.dump(features.to_dict(), f, indent=2)
                        logger.info(f"Feature extraction complete: {video_id}")

                        # The manifest was composed before this ran, so it still
                        # says kinematics never happened. Stamp the truth --
                        # including WHICH review was applied (content identity;
                        # the staleness scanner reads it back).
                        from mousereach.pipeline.manifest import record_kinematic_version
                        record_kinematic_version(video_id, processing_dir,
                                                 extractor.VERSION,
                                                 review_path=review_path)
                        # No database push here: mousedb PULLS features files
                        # from the Analyzed tree (tool independence, 2026-08-28).
                        # The old sync call stayed behind and logged a failed
                        # 'db_sync' step on every video (1,539 times) -- removed.
                    except Exception as e:
                        self.db.log_step(video_id, 'feature_extraction', 'failed', message=str(e))
                        logger.warning(f"Feature extraction failed for {video_id}: {e}")

            self.db.update_state(video_id, 'processed')
            self.db.log_step(video_id, 'local_pipeline', 'completed',
                            message=f"All steps complete ({pipeline_duration:.1f}s total)",
                            duration=pipeline_duration)
            logger.info(f"Local pipeline complete: {video_id} ({pipeline_duration:.1f}s)")
            self._sync_to_connectome(video_id, 'processed',
                                     processed_at=datetime.now().isoformat())

        except Exception as e:
            pipeline_duration = time.time() - pipeline_start
            self.db.log_step(video_id, 'local_pipeline', 'failed', message=str(e), duration=pipeline_duration)
            self.db.mark_failed(video_id, f"Local pipeline error: {e}")
            raise
        return True

    def _archive_locally_processed(self, work: dict):
        """Archive a locally processed video directly to NAS.

        In also_process mode, results go straight to Analyzed/{project}/{cohort}/ on NAS,
        skipping the Processing/Posed staging step entirely.
        """
        from mousereach.archive.core import archive_video

        video_id = work['id']
        logger.info(f"Archiving locally processed {video_id} to NAS")
        self.db.update_state(video_id, 'archiving')
        self.db.log_step(video_id, 'archive', 'started')
        start_time = time.time()

        try:
            dlc_queue = Paths.DLC_QUEUE
            # The local pipeline works in the local Processing folder (it
            # stages the pose and the video there, see _run_local_pipeline),
            # so that is where the results are. Archiving from DLC_Queue --
            # which holds only the mp4 and the raw pose -- filed a video
            # with no manifest, segments, reaches or outcomes (2026-09-12).
            source_dir = Path(Paths.PROCESSING) if Paths.PROCESSING else dlc_queue
            result = archive_video(
                video_id,
                dry_run=False,
                verbose=False,
                skip_ready_check=True,
                source_dir=source_dir,
            )
            duration = time.time() - start_time

            if result.get('success'):
                self._clear_archive_backoff(video_id)
                # 'processed' cannot go straight to 'archived' -- the state
                # machine routes through 'archiving' (db.py VIDEO_TRANSITIONS).
                # Jumping it raised, and because the files had ALREADY moved the
                # video stayed in 'processed' with nothing left in Processing to
                # archive: the next pass then read 'not ready' forever. Filing had
                # never actually run before today, so this had never fired.
                self.db.update_state(video_id, 'archiving')
                self.db.update_state(video_id, 'archived')
                self.db.log_step(video_id, 'archive', 'completed',
                                message=f"Archived {len(result.get('files_moved', []))} files",
                                duration=duration)
                logger.info(f"Archived {video_id} to NAS ({duration:.1f}s)")
                self._sync_to_connectome(video_id, 'archived',
                                         staged_at=datetime.now().isoformat())

                # Export to central DB on NAS for cross-node provenance
                self.db.export_to_central_db(video_id)

                # If this was a re-pose request, the round trip closed here:
                # the new pose and results went straight to the archive, so
                # the request must not stay outstanding (watcher/repose.py).
                try:
                    from mousereach.watcher import repose
                    if repose.close_request(video_id, consumed_by=self.hostname):
                        logger.info(f"{video_id}: re-pose request closed (archived here)")
                except Exception as e:
                    logger.debug(f"{video_id}: could not close re-pose request ({e})")

                # Clean up the local working copies: the results in Processing
                # and the raw inputs in DLC_Queue.
                for d in (source_dir, dlc_queue):
                    if not d or not Path(d).exists():
                        continue
                    for f in self._get_associated_files(Path(d), video_id):
                        try:
                            f.unlink()
                        except Exception:
                            pass
                # The archive holds the video now. A shared copy this node
                # claimed for it is removed only against the archived mp4 (the
                # move is not size-checked, so no 'verified' shortcut here).
                archived_in = result.get('destination')
                self._retire_claimed_single(
                    video_id,
                    Path(archived_in) / f"{video_id}.mp4" if archived_in else None)
                return True
            else:
                error = result.get('error', 'archive failed')
                if error == "No files found in Processing/":
                    # Terminal, not retriable -- see _archive_to_nas for why.
                    self.db.mark_failed(video_id, error)
                    self.db.log_step(video_id, 'archive', 'failed',
                                     message=error, duration=duration)
                    logger.error(
                        f"Archive impossible for {video_id}: {error} -- marked "
                        f"failed (nothing on disk to archive)")
                    return False
                delay = self._note_archive_failure(video_id)
                self.db.log_step(video_id, 'archive', 'failed', message=error, duration=duration)
                logger.warning(f"Archive failed for {video_id}: {error} "
                               f"(retrying in {delay // 60}m)")
                return False

        except Exception as e:
            duration = time.time() - start_time
            # Back off (see _archive_to_nas): a cheap repeating exception must
            # not re-select every cycle with no delay.
            self._note_archive_failure(video_id)
            self.db.log_step(video_id, 'archive', 'failed', message=str(e), duration=duration)
            logger.error(f"Archive error for {video_id}: {e}")
            return False

    def _stage_files(self, video_id: str, files: list) -> list:
        """Move this video's files into NAS staging so the other side can
        never take in a half-staged set. Three rules:

          * COPY everything first, each under a temporary name (<name>.part);
            a copy straight to the final name was visible, and could be
            claimed, from its first byte;
          * only when every copy verified, RENAME them into place, the pose
            (.h5) strictly last -- a staged .h5 is what the processing node's
            discovery keys on, so when it appears everything else is there;
          * only then DELETE the local originals.

        Any failure before the deletes leaves the originals intact and no
        final-named file behind: the temporaries are removed and IOError is
        raised, so the row is marked failed rather than 'archived' with a
        pose path that points at nothing. Returns the staged names.
        """
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        ordered = sorted(files, key=lambda f: (f.suffix.lower() == '.h5', f.name))
        parts = []
        try:
            for file_path in ordered:
                dest = self.staging_dir / file_path.name
                part = dest.with_name(dest.name + '.part')
                if not safe_copy(file_path, part, verify=True):
                    raise IOError(f"could not copy {file_path.name} to staging")
                parts.append((file_path, part, dest))
            for file_path, part, dest in parts:
                try:
                    part.replace(dest)
                except OSError as e:
                    raise IOError(f"could not finalise {dest.name} in staging: {e}")
        except Exception:
            for _, part, _ in parts:
                try:
                    part.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        staged = []
        for file_path, _, dest in parts:
            try:
                file_path.unlink()
            except OSError as e:
                logger.warning(f"{video_id}: staged {dest.name} but could not remove the "
                               f"local copy ({e}); it will be cleaned up later")
            staged.append(dest.name)
            logger.debug(f"Staged: {dest.name}")
        return staged

    # =========================================================================
    # NAS STAGING
    # =========================================================================

    def _stage_to_nas(self, work: dict):
        """
        Move DLC-complete video + h5 from local storage to NAS staging
        (Processing/Posed).

        The processing PC picks these up from the NAS staging folder.
        """
        video_id = work['id']
        video_data = work['data']

        if not self.staging_dir:
            logger.error("DLC_STAGING path not configured (NAS drive not set)")
            self.db.mark_failed(video_id, "DLC_STAGING path not configured")
            return False

        logger.info(f"Staging {video_id} to NAS: {self.staging_dir}")

        # A row can reach here with no path at all. Cross-node recovery adopts
        # another machine's state out of connectome.db, which carries no file
        # paths, so this node can be told "20251225_CNT0405_P4 is dlc_complete"
        # about a video it has never held. Path(None) then raises TypeError, the
        # video is marked failed, and the next cycle picks it up again.
        #
        # search_archive is off deliberately: the loop below MOVES what it finds.
        # A hit in the archive is the finished copy of this video, and moving it
        # into the staging folder would empty the archive to feed a queue.
        #
        # search_staging is off too, and that is a decision about what counts
        # as "already staged". This handler WRITES into staging, so a file found
        # there says nothing about THIS node: the same stem can be another
        # node's hand-off, or the processing server's intake mid-copy. A video
        # used to be force-marked 'archived' here on the strength of such a
        # file. Now only this node's own copies are found, and staging is looked
        # at in exactly one case: finishing this node's own interrupted stage
        # (_resume_interrupted_stage), where the proof is a pose this node
        # stages itself.
        current_path = locate_video_file(
            video_id, raw=video_data.get('current_path'),
            extra_dirs=[Paths.DLC_QUEUE], search_archive=False,
            search_staging=False)

        if current_path is None:
            staged_mp4 = self.staging_dir / f"{video_id}.mp4"
            try:
                staged_already = staged_mp4.is_file()
            except OSError:
                staged_already = False
            if staged_already:
                return self._resume_interrupted_stage(video_id, staged_mp4)
            self.db.mark_unresolvable(
                video_id,
                "nothing to stage: no video file for it on this node "
                "(recorded path: %r)" % (video_data.get('current_path'),))
            self.db.log_step(video_id, 'stage_to_nas', 'skipped',
                             message="no file on this node")
            self._release_claim_given_up(video_id, "no local copy left to stage")
            return False

        self.db.log_step(video_id, 'stage_to_nas', 'started')
        start_time = time.time()

        try:
            # Find all files for this video (mp4, h5, csv)
            source_dir = current_path.parent
            all_files = self._get_associated_files(source_dir, video_id)

            if not all_files:
                raise FileNotFoundError(f"No files found for {video_id} in {source_dir}")

            staged_files = self._stage_files(video_id, all_files)

            duration = time.time() - start_time

            # Mark as archived (done on this machine). Record the STAGED pose
            # path too: until 2026-09-12 only current_path was updated and
            # dlc_output_path kept pointing at the local DLC_Queue file the
            # move above had just removed -- 339 of 458 archived rows on the
            # lab GPU node named a pose file that no longer existed, and any
            # reader that trusts the recorded path (rather than globbing the
            # folder) concluded those videos had no pose. The pose recorded is
            # one THIS call staged, never whatever else sits in staging.
            staged_h5 = self._own_staged_pose(staged_files)
            if staged_h5 is None:
                raise IOError("stage_to_nas incomplete: no pose file staged")
            self.db.update_state(
                video_id, 'archived',
                current_path=str(self.staging_dir / current_path.name),
                dlc_output_path=str(staged_h5),
            )
            self.db.log_step(
                video_id, 'stage_to_nas', 'completed',
                message=f"Staged {len(staged_files)} files to NAS",
                duration=duration
            )

            logger.info(f"Staged {video_id} to NAS ({len(staged_files)} files, {duration:.1f}s)")
            self._sync_to_connectome(video_id, 'archived',
                                     staged_at=datetime.now().isoformat(),
                                     nas_path=str(self.staging_dir))

        except Exception as e:
            duration = time.time() - start_time
            self.db.mark_failed(video_id, str(e))
            self.db.log_step(video_id, 'stage_to_nas', 'failed', message=str(e), duration=duration)
            raise
        # Staged: the processing server has the video. A shared copy this node
        # claimed for it is a duplicate now. _stage_files size-checked every
        # copy it made, which stands in when the server has already taken the
        # staged mp4 in.
        self._retire_claimed_single(video_id, self.staging_dir / f"{video_id}.mp4",
                                    verified=f"{video_id}.mp4" in staged_files)
        return True

    def _own_staged_pose(self, staged_names) -> Optional[Path]:
        """The pose among the files THIS call just staged, or None.

        WHY only those: globbing staging for any pose of this stem could pick
        one another node (or an earlier run) left there, and recording it
        would say this node handed over a pose it did not. When this call
        staged more than one pose, the declared model's wins, as elsewhere.
        """
        from mousereach.watcher import repose
        poses = [self.staging_dir / name for name in staged_names
                 if name.endswith('.h5') and 'DLC' in name]
        return select_pose_file(poses, expected_scorer=repose.declared_scorer() or None)

    def _resume_interrupted_stage(self, video_id: str, staged_mp4: Path):
        """The video is already in staging and has no copy left on this node:
        finish this node's own interrupted stage, or say it cannot.

        _stage_files renames the pose into place LAST and deletes the local
        originals only after every rename, so a stage cut short leaves the
        mp4 in staging with this node's pose still in its DLC_Queue. That is
        the case this finishes: stage whatever is still local, and count the
        video done only on a pose THIS call staged (_own_staged_pose).

        WHY not on anything else in staging: a same-named mp4 or pose there
        can be another node's hand-off or the processing server's intake, and
        is no evidence this node posed or staged the video. With nothing of
        its own left to stage the row is recorded 'unresolvable' -- never
        'archived' -- and the files in staging are left untouched for the
        processing server. WHY not 'failed': nothing went wrong with the video
        (this node may even have staged it fully and been stopped before
        recording it); it simply has no file here. 'failed' is a retry state
        that reads as a verdict on the data, and the ERROR it raised filled the
        failed count with videos already safely with the server.
        """
        leftovers = (self._get_associated_files(Path(Paths.DLC_QUEUE), video_id)
                     if Paths.DLC_QUEUE else [])
        if not leftovers:
            reason = ("already in NAS staging (Processing/Posed) and no file for it on "
                      "this node; left for the processing server")
            self.db.mark_unresolvable(video_id, reason)
            self.db.log_step(video_id, 'stage_to_nas', 'skipped', message=reason)
            logger.info(f"{video_id}: {reason}")
            # The video is with the processing server already; a claimed copy
            # of it is a duplicate if the staged mp4 matches it. Kept (and left
            # to go stale) otherwise.
            self._retire_claimed_single(video_id, staged_mp4)
            return False
        self.db.log_step(video_id, 'stage_to_nas', 'started', message="resuming")
        start_time = time.time()
        try:
            staged = self._stage_files(video_id, leftovers) if leftovers else []
            staged_h5 = self._own_staged_pose(staged)
            if staged_h5 is None:
                raise IOError(
                    "the video is already in NAS staging but this node has no pose "
                    "of its own left to stage; a pose already in staging is not "
                    "evidence that this node staged it")
            logger.info(f"{video_id}: finished an interrupted stage to NAS "
                        f"({len(staged)} file(s)); marking archived")
            self.db.force_state(video_id, 'archived',
                                reason="finished this node's interrupted stage to NAS staging",
                                current_path=str(staged_mp4),
                                dlc_output_path=str(staged_h5))
            self.db.log_step(video_id, 'stage_to_nas', 'completed',
                             message=f"resumed ({len(staged)} file(s) staged)",
                             duration=time.time() - start_time)
        except Exception as e:
            self.db.mark_failed(video_id, str(e))
            self.db.log_step(video_id, 'stage_to_nas', 'failed', message=str(e),
                             duration=time.time() - start_time)
            raise
        self._retire_claimed_single(video_id, staged_mp4)
        return True


# =============================================================================
# PROCESSING ORCHESTRATOR (Processing Server)
# =============================================================================

class ProcessingOrchestrator(BaseOrchestrator):
    """
    Processing server orchestrator.

    Watches the NAS Processing/Posed/ staging folder for new DLC outputs,
    copies them to the local Processing/ folder, and runs the full
    analysis pipeline (segmentation -> reach detection -> outcomes).

    After processing, auto-approved videos are archived to NAS.
    """

    handles_reprocessing = True

    def __init__(self, config: WatcherConfig, db: WatcherDB):
        super().__init__(config, db)

        # Where the DLC PC stages completed files on NAS
        self.staging_dir = Paths.DLC_STAGING
        if self.staging_dir:
            logger.info(f"Watching staging directory: {self.staging_dir}")
        else:
            logger.warning("DLC_STAGING not configured -- no intake possible")

        # Local processing directory
        self.processing_dir = Paths.PROCESSING
        if self.processing_dir:
            self.processing_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Processing directory: {self.processing_dir}")

        # Reprocessing scanner instance (the heavy version-compliance scan
        # itself is scheduled by BaseOrchestrator._maybe_review_reprocess_scan
        # -- ONE scheduler, wall-clock, stamped after completion. _scan_phase
        # used to fire a second, cycle-counted copy of the same scan; two
        # schedulers meant double NAS load, and cycle counts made the cadence
        # depend on an assumption about loop speed that nothing enforces.
        # 2026-09-01: back-to-back scans "fixed" by raising the cycle count.
        # 2026-09-04: recurred anyway -- a stuck priority-1 row failed in
        # 50 ms every cycle, cycles became free, and the counter fired the
        # 37-minute scan 15 times back-to-back overnight (~9.5 of 12 hours
        # scanning, ~6 videos processed). Seconds cannot be raced.
        self._reprocess_scanner = None
        # The review-return scan is a cheap listdir over two queue folders and
        # is how human-cleared work re-enters the pipeline: often, on its own
        # wall clock.
        self._review_return_min_s = 120
        # Collage retirement does two full pipeline-tree scans; rare on
        # purpose. (Its old expression -- every 6th reprocess scan -- had
        # already silently drifted from ~30 min to ~3 h when the scan cadence
        # changed on 2026-09-01; 3 h is the deployed behaviour and is kept,
        # now stated explicitly.)
        self._retire_min_s = 3 * 3600
        self._last_review_return_mono = 0.0
        self._last_retire_mono = 0.0

        if Paths.NAS_ROOT:
            try:
                from mousereach.watcher.reprocessor import ReprocessingScanner
                self._reprocess_scanner = ReprocessingScanner(db, Paths.NAS_ROOT)
                logger.info("Reprocessing scanner enabled")
            except Exception as e:
                logger.warning(f"Reprocessing scanner not available: {e}")

    # =========================================================================
    # SCAN PHASE
    # =========================================================================

    # How long a video must sit in the posed-video folder with no pose beside
    # it before it counts as misfiled rather than mid-handover. Staging copies
    # the video first and renames the pose into place LAST, so a video that
    # has only just arrived legitimately has no pose yet -- without this, the
    # rescue below would pull a video out from under a staging in progress.
    _MISFILED_SETTLE_S = 30 * 60

    def _rescue_misfiled_singles(self) -> int:
        """Move a video left in the posed-video folder back to the front door.

        Processing/Posed is where POSED videos are handed over, and the
        scan that reads it keys on pose files. A bare mp4 dropped there was
        therefore invisible twice over: no pose to find it by, and no database
        row to notice it was missing. It belongs in Unanalyzed/Single_Animal,
        which is where a video that still needs posing goes, and the ordinary
        path takes it from there.

        Deliberately narrow, because this folder is a live handover point.
        Only a video this database has never heard of, with no pose beside it,
        nothing half-written alongside, and settled long enough that it cannot
        be a handover in progress.
        """
        staging = self.staging_dir
        front_door = Paths.SINGLE_ANIMAL_OUTPUT
        if not staging or not front_door or not Path(staging).exists():
            return 0
        try:
            candidates = list(Path(staging).glob("*.mp4"))
        except OSError as e:
            logger.debug(f"could not read the staging folder: {e}")
            return 0
        # Videos a GPU node has already claimed out of the singles folder
        # (watcher/single_claim.py). Moving a same-named video to the front
        # door would have it taken and posed a second time.
        try:
            claimed = single_claim.inflight_ids()
        except Exception:
            claimed = {}
        moved = 0
        now = time.time()
        for mp4 in candidates:
            if "DLC" in mp4.stem:
                continue                      # a labeled/overlay by-product
            video_id = get_video_id(mp4.name)
            if not video_id:
                continue
            if list(Path(staging).glob(f"{video_id}DLC*.h5")):
                continue                      # posed: an ordinary handover
            if list(Path(staging).glob(f"{video_id}*.part")):
                continue                      # still being written
            try:
                if now - mp4.stat().st_mtime < self._MISFILED_SETTLE_S:
                    continue                  # may be a handover in progress
            except OSError:
                continue
            try:
                if self.db.get_video(video_id) is not None:
                    continue                  # known already, not a stray drop
            except Exception:
                continue
            if video_id in claimed:
                continue                      # a GPU node already holds this video
            # Only ever the TOP of the singles folder, never its claim area
            # (.inflight/<host>/): a file put there would read as a claim no
            # node made, so no node would ever pose it.
            dest = Path(front_door) / mp4.name
            if dest.exists():
                continue
            try:
                Path(front_door).mkdir(parents=True, exist_ok=True)
                if safe_move(mp4, dest):
                    moved += 1
                    logger.warning(
                        "%s sat in the posed-video folder with no pose beside "
                        "it. Moved to %s, where a video that still needs "
                        "posing goes.", video_id, front_door)
            except Exception as e:
                logger.warning(f"could not move {mp4.name} to the front door: {e}")
        return moved

    def _scan_phase(self):
        """Discover new DLC outputs in the staging directory."""
        if not self.staging_dir:
            return

        # Clean up stale claims from crashed nodes
        self._cleanup_stale_claims()

        # A video this node already knows about, parked 'outdated' because it
        # needed a NEW pose, is invisible to discover_dlc_staged (which only
        # registers unknown ids). When a GPU node re-poses it and stages the
        # result here, adopt it: the row stays 'outdated' but its scope
        # narrows to 'segmentation' and points at the staged pose, and the
        # outdated handler (_reprocess_video) re-runs it with the archived
        # results copied down beside it (watcher/repose.py -- the return leg
        # of the re-pose round trip).
        try:
            from mousereach.watcher import repose
            adopted = repose.adopt_staged_reposes(self.db, self.staging_dir)
            if adopted:
                logger.info(f"Adopted {len(adopted)} re-posed video(s) from staging: "
                            f"{', '.join(adopted[:5])}"
                            f"{'' if len(adopted) <= 5 else ' ...'}")
        except Exception as e:
            logger.warning(f"Re-pose adoption scan failed (non-fatal): {e}")

        # A video someone put in the posed-video folder without a pose is
        # invisible to the scan below, which finds videos BY their pose file.
        # Send it back to the front door rather than leave it sitting.
        try:
            self._rescue_misfiled_singles()
        except Exception as e:
            logger.warning(f"Could not check staging for misfiled videos: {e}")

        newly_found = self.state.discover_dlc_staged(self.staging_dir)
        if newly_found:
            logger.info(f"Discovered {len(newly_found)} new DLC outputs in staging")

        # The heavy version-compliance scan is NOT fired here -- see __init__:
        # BaseOrchestrator._maybe_review_reprocess_scan is its one scheduler.

        # Periodic review-return scan: re-inject human-cleared held videos
        # (triage fully resolved / deep-review flag cleared) back into Processing
        # so the pipeline re-runs them and the gate re-checks. Wall-clock,
        # stamped after the pass (see __init__ for why never cycle counts).
        if (self.processing_dir and
                time.monotonic() - self._last_review_return_mono
                >= self._review_return_min_s):
            try:
                from mousereach.watcher.review_return import scan_review_queues
                scan_review_queues(self.db, self.processing_dir)
            except Exception as e:
                logger.warning(f"Review-return scan failed: {e}")
            finally:
                self._last_review_return_mono = time.monotonic()

        # Periodic collage retirement: move any collage whose offspring have ALL
        # made it through the pipeline (version-current + review-clean) to ultimate
        # storage (Analyzed/Multi-Animal, backup-synced). Slow cadence.
        if time.monotonic() - self._last_retire_mono >= self._retire_min_s:
            try:
                self._retire_completed_collages()
            finally:
                self._last_retire_mono = time.monotonic()

    def _retire_completed_collages(self):
        """Retire fully-complete collages to ultimate storage. Best-effort; only
        acts on collages every one of whose single-animal offspring is in the final
        Analyzed output, processed with the currently-shipped versions, and with no
        review pending. Never raises out."""
        try:
            from mousereach.config import Paths
            from mousereach.video_prep.core.collage_provenance import (
                retire_completed_collages, build_downstream_index, build_complete_stems)
            downstream = build_downstream_index()
            complete = build_complete_stems()
            summ = retire_completed_collages(
                Paths.MULTI_ANIMAL_SOURCE, downstream,
                complete_stems=complete, dry_run=False)
            if summ.get("retired"):
                logger.info(
                    f"Collage retirement: moved {summ['retired']} fully-complete "
                    f"collage(s) to {summ['dest']} (mirrored by the backup watcher)")
        except Exception as e:
            logger.warning(f"Collage retirement scan failed: {e}")

    # =========================================================================
    # WORK QUEUE
    # =========================================================================

    def _select_work_item(self, admit_deferred: bool = True):
        """
        One pass over the processing server's work buckets.

        Called twice per cycle by BaseOrchestrator._get_next_work_item: once
        with admit_deferred=False, and only if that finds nothing at all, once
        with it True. Every pick below is tested for None, because a bucket can
        be non-empty and still yield nothing on the first pass.

        ADMIT / DRAIN. Only the buckets that bring NEW work onto this node
        honour admit_deferred:

          ADMIT (gated):    intake (2), reprocess (4)
          DRAIN (ungated):  archive (1), pipeline (3)

        The drain buckets finish videos whose files are already on this
        server's disk and already counted against max_local_pending. Gating
        them was tried and is wrong twice over. It strands data -- a node with
        a backlog is never idle, so a deferred video in 'processing' is never
        analysed and never archived, and it holds a slot against the cap
        forever, which then throttles intake of the work we wanted first. And
        because intake is visited BEFORE pipeline, gating only intake without
        also ungating pipeline would let an idle node copy a whole cap's worth
        of deferred videos in before analysing one. With intake gated and
        pipeline ungated, one deferred video is admitted, the next cycle
        finishes it, and the pool never fills with work that cannot leave it.

        The drain buckets are still ORDERED by the policy: inside each one the
        preferred tier is picked first.

        Priority:
        1. Archive: Processed videos -> archive to NAS (finish what's done)
        2. Intake: DLC-complete videos on NAS -> copy to local Processing/
        3. Pipeline: Videos in Processing/ that need seg/reach/outcomes
        4. Reprocess: videos the staleness scanner marked outdated
        """
        priority_animal = self._get_priority_animal()

        # Priority 1: Archive - move processed videos to NAS, before taking on
        # new work. WHY: archive used to be below pipeline, and intake keeps
        # the pipeline pool topped up, so completed results piled up locally and
        # never reached the NAS (or the database that pulls from it) until the
        # staged supply ran dry -- days, during a backlog. Archiving is quick
        # file moves; doing it first keeps results flowing without measurably
        # starving the pipeline. Same finish-what's-done ordering the DLC-node
        # orchestrator has always used.
        #
        # DRAIN, so never gated: filing a finished video is not analysis. The
        # result has already been produced; refusing to file it strands it on
        # local disk, holds a row in a working state, and keeps the result off
        # the NAS and out of every downstream pull. Ordering still applies --
        # the preferred tier is filed first.
        videos = [v for v in self.db.get_videos_in_state('processed')
                  if not self._archive_backoff_active(v['video_id'])]
        pick = self._pick_from_pool(videos, priority_animal, 'animal_id',
                                    randomize=False)
        if pick is not None:
            return {
                'type': 'archive',
                'id': pick['video_id'],
                'data': pick
            }

        # Priority 2: Intake - videos discovered in staging, not yet copied locally
        # ADMIT: this is the door that puts a new video on this node's disk and
        # spends a slot of max_local_pending. It is the one that must wait.
        videos = self.db.get_videos_in_state('dlc_complete')
        if videos:
            # Check disk space cap
            processing_count = len(self.db.get_videos_in_state('processing'))
            if processing_count >= self.config.max_local_pending:
                # First pass only. Both passes see the same cap and the same
                # bucket, and one warning per cycle is one warning per cycle.
                if not admit_deferred:
                    logger.warning(
                        f"Local pending limit reached ({processing_count}/{self.config.max_local_pending}). "
                        "Pausing intake -- review or archive pending videos."
                    )
            else:
                pick = self._pick_from_pool(videos, priority_animal, 'animal_id',
                                            allow_deferred=admit_deferred,
                                            randomize=False)
                if pick is not None:
                    return {
                        'type': 'intake',
                        'id': pick['video_id'],
                        'data': pick
                    }

        # Priority 3: Pipeline - run seg/reach/outcomes on locally staged videos
        # DRAIN, so never gated. The video is already on this disk and already
        # counted against the cap; finishing it is what frees the slot. This is
        # also what makes a deadlock guard unnecessary: a local pool full of
        # deferred videos drains itself on the very first pass, so intake can
        # never be blocked by work that has no way out.
        videos = self.db.get_videos_in_state('processing')
        if videos:
            pick = self._pick_from_pool(videos, priority_animal, 'animal_id')
            if pick is not None:
                return {
                    'type': 'pipeline',
                    'id': pick['video_id'],
                    'data': pick
                }

        # Priority 4: Reprocess outdated videos
        outdated = self.db.get_videos_in_state('outdated')
        if outdated:
            # 'full' means the video needs a genuinely NEW pose, which this node
            # cannot make -- it has no CUDA. It used to be pushed into
            # 'dlc_queued' here, and that went nowhere useful: this node's work
            # loop never selects dlc_queued, the DLC PC's DLC_Queue is a LOCAL
            # folder it cannot see, and the only thing that crossed machines was
            # the state itself, through connectome.db -- where cross-node
            # recovery turned it into a pathless row on the GPU node that
            # crashed on Path(None) and came back every cycle.
            #
            # So the row stays 'outdated' here and the ASK travels over shared
            # storage instead: one request file per video in the Repose_Queue
            # folder, which any GPU node's watcher pulls from (watcher/repose.py).
            # When the new pose comes back through Processing/Posed,
            # _scan_phase adopts it: the row's scope narrows to 'segmentation'
            # and it drains through the 'actionable' branch below.
            # The scanner has already been taught not to call a video DLC-stale
            # when the declared pose is sitting in the archive, so this pool is
            # small, and nothing in it blocks the videos that CAN be re-run here.
            needs_pose = [v for v in outdated
                          if (v.get('reprocess_scope') or '') == 'full']
            actionable = [v for v in outdated
                          if (v.get('reprocess_scope') or '') != 'full']

            if needs_pose and not getattr(self, '_logged_repose_hold', False):
                self._logged_repose_hold = True
                logger.info(
                    "%d outdated video(s) need a new DLC pose and wait here for one: "
                    "%s%s. Re-pose requests for them are published to the shared "
                    "Repose_Queue after each version scan; a GPU node's watcher "
                    "picks them up, and the new pose is adopted from staging.",
                    len(needs_pose),
                    ', '.join(v['video_id'] for v in needs_pose[:5]),
                    '' if len(needs_pose) <= 5 else ' and %d more' % (len(needs_pose) - 5))

            if actionable:
                # post_dlc: re-run seg/reach/outcomes from existing DLC output
                # ADMIT: reprocessing pulls a finished video back onto this
                # node and re-opens work that was already done. It waits.
                pick = self._pick_from_pool(actionable, priority_animal, 'animal_id',
                                            allow_deferred=admit_deferred)
                if pick is not None:
                    return {
                        'type': 'reprocess',
                        'id': pick['video_id'],
                        'data': pick
                    }

        return None

    # =========================================================================
    # DISPATCH
    # =========================================================================

    def _dispatch_work(self, work: dict):
        """Route work to appropriate handler."""
        work_type = work['type']
        work_id = work['id']

        try:
            ok = None
            if work_type == 'intake':
                ok = self._intake_from_staging(work)
            elif work_type == 'pipeline':
                ok = self._run_pipeline(work)
            elif work_type == 'archive':
                ok = self._archive_to_nas(work)
            elif work_type == 'reprocess':
                ok = self._reprocess_video(work)
            else:
                # A row producing unrecognised work would be re-selected
                # forever; fail it so the loop cannot spin on it.
                logger.warning(f"Unknown work type: {work_type} -- marking {work_id} failed")
                self.db.mark_failed(work_id, f"unknown work type: {work_type}")
                return False
            # A handler that explicitly declined (returned False) did no
            # work; the main loop must sleep, not spin (legacy handlers
            # returning None count as progress).
            return ok is not False

        except Exception as e:
            error_msg = f"{work_type} failed: {str(e)}"
            logger.error(f"Work item {work_id} failed: {e}", exc_info=True)
            if not self._already_failed(work_id):
                self.db.mark_failed(work_id, error_msg)
            return False

    # =========================================================================
    # REPROCESS: Re-run pipeline on outdated archived videos
    # =========================================================================

    def _reprocess_video(self, work: dict):
        """Re-run seg/reach/outcomes on an outdated archived video.

        The video's archived files are still on NAS. We copy the DLC h5
        and video to local Processing/, re-run the pipeline, and re-archive.
        """
        video_id = work['id']
        video_data = work['data']

        logger.info(f"Reprocessing outdated video {video_id} (post-DLC)")

        # Find archived files on NAS
        archive_dir = Paths.ANALYZED_OUTPUT if Paths.ANALYZED_OUTPUT else None
        if not archive_dir or not archive_dir.exists():
            self.db.mark_failed(video_id, "Archive directory not found for reprocessing")
            return

        # Which pose to re-run against. A pose the row already points at --
        # a freshly staged one from the declared model, adopted by the
        # re-pose round trip (watcher/repose.py) -- wins over anything in the
        # archive, then a declared-model pose in staging, and only then the
        # archive search that served every reprocess before 2026-09-12.
        from mousereach.watcher import repose
        declared = repose.declared_scorer()
        h5_files = []
        recorded = video_data.get('dlc_output_path')
        if recorded and Path(recorded).is_file() and (
                not declared or repose.scorer_of(Path(recorded)) == declared):
            h5_files = [Path(recorded)]
        if not h5_files and self.staging_dir:
            staged = repose.declared_poses_in(self.staging_dir, declared).get(video_id)
            if staged is not None:
                h5_files = [staged]
        if not h5_files:
            # Never enter Analyzed/Archive/: superseded poses keep their
            # original names there, so an rglob found them beside the live
            # ones and select_pose_file could re-run the video on an old
            # generation's pose (newest mtime wins among same-scorer files).
            # DLC Model <N>/ folders stay visible -- they are live pose storage.
            from mousereach.pipeline.analyzed_tree import iter_files
            h5_files = list(iter_files(archive_dir, f"{video_id}DLC*.h5"))
        if not h5_files:
            self.db.mark_failed(video_id, f"DLC h5 not found in archive for reprocessing")
            return

        source_h5 = select_pose_file(h5_files, expected_scorer=declared or None)

        # The pose file lives in its OWN per-model tree
        # (Analyzed/Connectome/DLC Model 4/CNT01), not beside the video's results
        # (Analyzed/Connectome/CNT01). Copying from the pose file's parent
        # therefore brought over three files and nothing else, with two
        # consequences that both looked like correct behaviour:
        #
        #   - no _segments.json arrived, so the stage-reuse checks further down
        #     ("only re-run from the earliest stale stage") could never find
        #     prior output. Every reprocess silently restarted from
        #     segmentation, recutting boundaries and re-deciding outcomes even
        #     when the only thing out of date was kinematics.
        #   - no mp4 arrived, so a video the review gate held was moved into the
        #     review queue with no video in it for the reviewer to open.
        #
        # get_archive_destination computes the results folder directly from the
        # video id, so this also avoids a second full-tree walk.
        from mousereach.archive.core import get_archive_destination
        source_dirs = {source_h5.parent}
        try:
            results_dir = get_archive_destination(video_id)
            if results_dir and Path(results_dir).is_dir():
                source_dirs.add(Path(results_dir))
        except Exception as e:
            logger.warning("%s: could not resolve the results folder (%s); "
                           "reprocessing from the pose folder alone", video_id, e)
        source_dir = source_h5.parent

        # Copy needed files to local Processing/
        if not self.processing_dir:
            self.db.mark_failed(video_id, "PROCESSING path not configured")
            return

        self.processing_dir.mkdir(parents=True, exist_ok=True)

        # Copy DLC h5 and video
        all_files = _reprocess_copy_set(source_dirs, source_h5.parent, video_id)
        copy_failures = []
        for src_file in all_files:
            dest = self.processing_dir / src_file.name
            if not safe_copy(src_file, dest, verify=True):
                copy_failures.append(src_file.name)
        if copy_failures:
            # A partial working set must not reach the pipeline: a stale local
            # twin of a file that failed to copy would be silently reused by
            # the stage-reuse checks (a locked _reaches.json nearly caused
            # exactly that on 2026-09-04). Fail loudly instead; safe_copy has
            # already absorbed transient locks with retries.
            self.db.mark_failed(
                video_id,
                "reprocess copy failed for: %s" % ", ".join(sorted(copy_failures)))
            return

        # Transition to processing state. Name the model explicitly: the copy
        # set above brings the previous generation's pose down beside the new
        # one, and select_pose_file's default is a cached reading of the
        # declaration -- on a daemon started before the model changed, that
        # cache chose the OLD pose and every re-run re-staled the video.
        local_h5 = select_pose_file(
            self.processing_dir.glob(f"{video_id}DLC*.h5"),
            expected_scorer=declared or None)
        self.db.force_state(
            video_id, 'processing',
            dlc_output_path=str(local_h5) if local_h5 else '',
            current_path=str(self.processing_dir / f"{video_id}.mp4")
        )

        # Run the standard pipeline (seg/reach/outcomes) -- dependency-aware:
        # start from the earliest stale stage and reuse the current upstream outputs.
        start_stage = work.get('data', {}).get('reprocess_scope', 'segmentation')
        reprocess_work = {
            'type': 'pipeline',
            'id': video_id,
            'data': {
                'video_id': video_id,
                'dlc_output_path': str(local_h5) if local_h5 else '',
                'current_path': str(self.processing_dir / f"{video_id}.mp4"),
                'reprocess_start_stage': start_stage,
            }
        }
        self._run_pipeline(reprocess_work)

    # =========================================================================
    # DRY RUN
    # =========================================================================

    def dry_run(self):
        """Scan without processing. Shows what would be done."""
        logger.info("ProcessingOrchestrator dry run")

        # Scan staging for new files
        if self.staging_dir and self.staging_dir.exists():
            newly_found = self.state.discover_dlc_staged(self.staging_dir)
            print(f"\nStaging scan:")
            print(f"  Staging directory:  {self.staging_dir}")
            print(f"  New DLC outputs:    {len(newly_found)}")
        else:
            print(f"\nStaging scan:")
            print(f"  Staging directory:  {self.staging_dir or '(not configured)'}")
            print(f"  Status:             NOT ACCESSIBLE")
            newly_found = []

        # Show priority animal if set
        priority_animal = self._get_priority_animal()
        if priority_animal:
            print(f"\n  PRIORITY ANIMAL:    {priority_animal}")

        # Show the ordering policy, so nobody has to read the config file to
        # find out why one video is being taken before another.
        print("\nWork priority:")
        for line in self.work_priority.describe():
            print(f"  {line}")
        for complaint in self.work_priority.complaints:
            print(f"  [!] {complaint}")

        # Show pending work
        print(f"\nPending work items:")
        count = 0
        for state_label, state in [
            ("Videos to intake from NAS", "dlc_complete"),
            ("Videos to process (seg/reach/outcomes)", "processing"),
            ("Videos to archive to NAS", "processed"),
        ]:
            items = self.db.get_videos_in_state(state)
            if items:
                print(f"  {state_label}: {len(items)}")
                for item in items[:5]:
                    print(f"    - {item.get('video_id')}")
                if len(items) > 5:
                    print(f"    ... and {len(items) - 5} more")
                count += len(items)

        if count == 0:
            print("  (no pending work)")

        # Show local disk usage
        if self.processing_dir and self.processing_dir.exists():
            local_videos = len(list(self.processing_dir.glob("*DLC*.h5")))
            print(f"\nLocal Processing/: {local_videos} videos")
            print(f"  Max local pending: {self.config.max_local_pending}")
        print()

    # =========================================================================
    # MULTI-NODE CLAIMING
    # =========================================================================

    def _claim_video(self, video_id: str) -> bool:
        """Try to claim a video for processing. Returns True if claimed.

        Uses marker files in Processing/Posed/.claims/ to prevent multiple nodes
        from processing the same video simultaneously.
        """
        if not self.staging_dir:
            return True  # No staging = no contention

        claim_dir = self.staging_dir / ".claims"
        try:
            claim_dir.mkdir(exist_ok=True)
        except Exception:
            return True  # Can't create claims dir = single-node mode

        claim_file = claim_dir / f"{video_id}.claimed"

        # Check if already claimed by another host
        if claim_file.exists():
            try:
                content = claim_file.read_text().strip()
                claimer = content.split('\n')[0]
                if claimer != self.hostname:
                    logger.debug(f"{video_id} already claimed by {claimer}")
                    return False
                # We already claimed it (retry after crash?)
                return True
            except Exception:
                return False

        # Try to claim
        try:
            claim_file.write_text(f"{self.hostname}\n{datetime.now().isoformat()}\n")
            # Verify our claim stuck (race window check on network drives)
            time.sleep(0.5)
            content = claim_file.read_text().strip()
            if content.split('\n')[0] == self.hostname:
                logger.info(f"Claimed {video_id} for {self.hostname}")
                return True
            return False
        except Exception:
            return False

    def _release_claim(self, video_id: str):
        """Release a claim after successful archive."""
        if not self.staging_dir:
            return
        claim_file = self.staging_dir / ".claims" / f"{video_id}.claimed"
        try:
            claim_file.unlink(missing_ok=True)
        except Exception:
            pass

    def _cleanup_stale_claims(self):
        """Remove claim files older than 24 hours (crashed nodes)."""
        if not self.staging_dir:
            return
        claim_dir = self.staging_dir / ".claims"
        if not claim_dir.exists():
            return

        stale_threshold = 24 * 3600  # 24 hours
        now = time.time()

        for claim_file in claim_dir.glob("*.claimed"):
            try:
                age = now - claim_file.stat().st_mtime
                if age > stale_threshold:
                    claim_file.unlink(missing_ok=True)
                    logger.info(f"Removed stale claim: {claim_file.stem} (age: {age/3600:.1f}h)")
            except Exception:
                pass

    # =========================================================================
    # INTAKE: Copy from NAS staging to local Processing/
    # =========================================================================

    def _intake_from_staging(self, work: dict):
        """Copy DLC-complete video+h5 from NAS staging to local Processing/."""
        video_id = work['id']
        video_data = work['data']

        if not self.processing_dir:
            logger.error("PROCESSING path not configured")
            self.db.mark_failed(video_id, "PROCESSING path not configured")
            return

        # Multi-node claiming: ensure no other node is processing this video
        if not self._claim_video(video_id):
            logger.debug(f"Skipping {video_id} - claimed by another node")
            return

        logger.info(f"Intake {video_id} from staging to Processing/")

        self.db.log_step(video_id, 'intake', 'started')
        start_time = time.time()

        try:
            # Find all files in staging directory.
            #
            # This used to be Path(current_path or '').parent, which for a
            # pathless row is Path('.').parent -- the current working directory.
            # It exists, so the fallback below never fired, and intake went
            # looking for the video among whatever files happen to sit in the
            # directory the watcher was started from.
            staged = locate_video_file(
                video_id, raw=video_data.get('current_path'),
                extra_dirs=[self.staging_dir], search_archive=False)
            source_dir = staged.parent if staged is not None else self.staging_dir

            all_files = self._get_associated_files(source_dir, video_id)
            if not all_files:
                raise FileNotFoundError(f"No files found for {video_id} in {source_dir}")

            self.processing_dir.mkdir(parents=True, exist_ok=True)

            # Copy files to local Processing/
            copied_files = []
            for file_path in all_files:
                dest_path = self.processing_dir / file_path.name
                if safe_copy(file_path, dest_path, verify=True):
                    copied_files.append(file_path.name)
                    logger.debug(f"Copied: {file_path.name}")
                else:
                    logger.warning(f"Failed to copy: {file_path.name}")

            if not copied_files:
                raise IOError(f"Failed to copy any files for {video_id}")

            duration = time.time() - start_time

            # Find local DLC h5 path. Name the model rather than relying on
            # select_pose_file's cached default: a re-posed video's older
            # pose can still be sitting in this folder from a previous run.
            from mousereach.watcher import repose
            local_h5 = select_pose_file(
                self.processing_dir.glob(f"{video_id}DLC*.h5"),
                expected_scorer=repose.declared_scorer() or None)
            local_mp4 = self.processing_dir / f"{video_id}.mp4"

            # Advance to processing state
            self.db.update_state(
                video_id, 'processing',
                dlc_output_path=str(local_h5) if local_h5 else video_data.get('dlc_output_path'),
                current_path=str(local_mp4)
            )
            self.db.log_step(
                video_id, 'intake', 'completed',
                message=f"Copied {len(copied_files)} files to Processing/",
                duration=duration
            )

            logger.info(f"Intake complete: {video_id} ({len(copied_files)} files, {duration:.1f}s)")

        except Exception as e:
            duration = time.time() - start_time
            self.db.mark_failed(video_id, str(e))
            self.db.log_step(video_id, 'intake', 'failed', message=str(e), duration=duration)
            raise

    # =========================================================================
    # PIPELINE: Run seg -> reach -> outcomes
    # =========================================================================

    def _run_pipeline(self, work: dict):
        """Run segmentation, reach detection, and outcome detection on a video.

        After each step, runs triage to set validation_status in the output JSON.
        After all steps complete, generates a processing manifest with provenance.
        """
        from mousereach.segmentation.core.batch import process_single as seg_single, add_validation_status
        from mousereach.reach.core.batch import process_single as reach_single
        from mousereach.outcomes.core.batch import process_single as outcome_single
        from mousereach.assignment.run import assign_reaches_for_video
        from mousereach.pipeline.manifest import create_processing_manifest
        from mousereach.pipeline.triage import triage_video
        from mousereach.watcher.review_gate import run_gate, route_deep_review, DECISION_CLEAN

        video_id = work['id']
        video_data = work['data']

        # dlc_output_path has bitten this pipeline twice, in opposite directions:
        # Path(None) raised and failed 950 of 954 videos on the DLC PC, and the
        # `or ''` fix for it made Path('') -> Path('.'), which exists, so the
        # missing-pose guard stopped firing and 723 videos were routed to human
        # review with "[Errno 13] Permission denied: '.'". resolve_pose_input
        # returns a real file or None and never a placeholder path.
        dlc_path = resolve_pose_input(
            video_data.get('dlc_output_path'), video_id, self.processing_dir
        )
        if dlc_path is None:
            self.db.mark_failed(video_id, f"DLC h5 not found for {video_id}")
            return

        # Stage outputs land BESIDE the pose file, so a pose outside the local
        # processing dir scatters results into whatever tree holds it -- three
        # restored videos wrote into the archive's model folder exactly this
        # way and archived thin bundles (2026-09-08; the staleness net caught
        # them). Stage the pose locally first, always.
        if Path(dlc_path).parent != Path(self.processing_dir):
            local = Path(self.processing_dir) / Path(dlc_path).name
            if not local.exists():
                from mousereach.watcher.transfer import safe_copy
                if not safe_copy(Path(dlc_path), local, verify=True):
                    self.db.mark_failed(
                        video_id, f"could not stage pose locally from {dlc_path}")
                    return
            dlc_path = local

        logger.info(f"Running pipeline on {video_id}")
        pipeline_start = time.time()

        # Dependency-aware reprocessing: reuse a stage's existing output when it is
        # current AND upstream of the earliest stale stage. Fresh videos have no
        # reprocess_start_stage -> index 0 -> every stage runs.
        _STAGE_IDX = {"segmentation": 0, "reach": 1, "outcome": 2, "kinematics": 3}
        _start_idx = _STAGE_IDX.get(video_data.get("reprocess_start_stage"), 0)
        _seg_out = self.processing_dir / f"{video_id}_segments.json"
        _reach_out = self.processing_dir / f"{video_id}_reaches.json"
        _outcome_out = self.processing_dir / f"{video_id}_pellet_outcomes.json"
        _run_seg = _start_idx <= 0 or not _seg_out.exists()
        _run_reach = _start_idx <= 1 or not _reach_out.exists()
        _run_outcome = _start_idx <= 2 or not _outcome_out.exists()

        # --- Step 1: Segmentation ---
        self.db.log_step(video_id, 'segmentation', 'started')
        step_start = time.time()

        try:
            if _run_seg:
                seg_result = seg_single(dlc_path)
            else:
                logger.info(f"Reusing segmentation for {video_id} (dependency-aware reprocess)")
                seg_result = {'success': True, 'status': 'reused', 'n_boundaries': 0}
            seg_duration = time.time() - step_start

            if seg_result.get('success', False):
                self.db.log_step(
                    video_id, 'segmentation', 'completed',
                    message=f"status={seg_result.get('status')}, boundaries={seg_result.get('n_boundaries', 0)}",
                    duration=seg_duration
                )
                logger.info(f"Segmentation complete: {video_id} ({seg_duration:.1f}s)")
            else:
                error = seg_result.get('error', 'segmentation failed')
                self.db.log_step(video_id, 'segmentation', 'failed', message=error, duration=seg_duration)
                # Segmentation that COULD NOT RUN is an infrastructure failure, not
                # something to put in front of a reviewer -- see
                # segmentation_could_not_run().
                if segmentation_could_not_run(error, dlc_path):
                    logger.error(
                        f"Segmentation could not run for {video_id} ({error}). "
                        f"Marking failed -- not routing to human review."
                    )
                    self.db.mark_failed(video_id, f"Segmentation could not run: {error}")
                    return
                # A real seg failure -> DEEP review, not a dead 'failed'. Over-long
                # recordings / uniform-fallback boundaries need manual re-seg with
                # the deep tools; move the whole bundle out of Processing so a
                # human can clear it and re-inject the video.
                try:
                    route_deep_review(
                        video_id, self.processing_dir,
                        f"segmentation_failed: {error}", db=self.db,
                        extra_sources=[self.processing_dir / f"{video_id}.mp4"],
                    )
                except Exception as route_err:
                    logger.warning(f"Deep-review routing failed for {video_id}: {route_err}")
                    self.db.mark_failed(video_id, f"Segmentation failed: {error}")
                return

        except Exception as e:
            seg_duration = time.time() - step_start
            self.db.log_step(video_id, 'segmentation', 'failed', message=str(e), duration=seg_duration)
            self.db.mark_failed(video_id, f"Segmentation error: {e}")
            raise

        # --- Step 2: Reach Detection ---
        seg_path = self.processing_dir / f"{video_id}_segments.json"
        if not seg_path.exists():
            self.db.mark_failed(video_id, "Segments file not created by segmentation")
            return

        self.db.log_step(video_id, 'reach_detection', 'started')
        step_start = time.time()

        try:
            if _run_reach:
                reach_result = reach_single(dlc_path, seg_path)
            else:
                logger.info(f"Reusing reach detection for {video_id} (dependency-aware reprocess)")
                reach_result = {'total_reaches': 0}
            reach_duration = time.time() - step_start

            self.db.log_step(
                video_id, 'reach_detection', 'completed',
                message=f"reaches={reach_result.get('total_reaches', 0)}",
                duration=reach_duration
            )
            logger.info(f"Reach detection complete: {video_id} ({reach_duration:.1f}s)")

        except Exception as e:
            reach_duration = time.time() - step_start
            self.db.log_step(video_id, 'reach_detection', 'failed', message=str(e), duration=reach_duration)
            self.db.mark_failed(video_id, f"Reach detection error: {e}")
            raise

        # --- Step 3: Outcome Detection (skip for E/F trays) ---
        tray_info = parse_tray_type(f"{video_id}.mp4")
        tray_type = tray_info.get('tray_type', 'P')
        skip_outcomes = tray_type in ('E', 'F')

        if not skip_outcomes:
            reach_path = self.processing_dir / f"{video_id}_reaches.json"

            self.db.log_step(video_id, 'outcome_detection', 'started')
            step_start = time.time()

            try:
                if _run_outcome:
                    outcome_result = outcome_single(dlc_path, seg_path, reach_path)
                else:
                    logger.info(f"Reusing outcomes for {video_id} (dependency-aware reprocess)")
                    outcome_result = {'n_segments': 0}
                outcome_duration = time.time() - step_start

                self.db.log_step(
                    video_id, 'outcome_detection', 'completed',
                    message=f"segments={outcome_result.get('n_segments', 0)}",
                    duration=outcome_duration
                )
                logger.info(f"Outcome detection complete: {video_id} ({outcome_duration:.1f}s)")

            except Exception as e:
                outcome_duration = time.time() - step_start
                self.db.log_step(video_id, 'outcome_detection', 'failed', message=str(e), duration=outcome_duration)
                self.db.mark_failed(video_id, f"Outcome detection error: {e}")
                raise
        else:
            logger.info(f"Skipping outcomes for {video_id} (tray type: {tray_type})")

        # --- Step 3.5: Reach Assignment (algo-4) -- causal-reach attribution ---
        # Runs BEFORE the gate: the gate treats a touched segment with no
        # committed causal reach as triaged, so the assignment must exist first.
        if not skip_outcomes:
            self.db.log_step(video_id, 'assignment', 'started')
            step_start = time.time()
            try:
                if _run_outcome or not (self.processing_dir / f"{video_id}_reach_assignments.json").exists():
                    assign_reaches_for_video(self.processing_dir, video_id, dlc_path)
                else:
                    logger.info(f"Reusing assignment for {video_id} (dependency-aware reprocess)")
                self.db.log_step(video_id, 'assignment', 'completed',
                                 duration=time.time() - step_start)
            except Exception as e:
                # Countable, like every other stage. Without this row the only trace of a
                # failure was a started/completed count mismatch -- and a failed assignment
                # writes no file, which silently switches off the gate's "touched pellet
                # with no credited reach" hold. State deliberately NOT changed here; see
                # docs/UNFINISHED.md before making this fail-closed.
                self.db.log_step(video_id, 'assignment', 'failed', message=str(e))
                logger.warning(f"Assignment (algo-4) failed for {video_id}: {e}")

        # --- Provenance manifest (travels with the bundle if the video is held) ---
        pipeline_duration = time.time() - pipeline_start
        try:
            step_timestamps = {
                'pipeline_started_at': datetime.fromtimestamp(pipeline_start).isoformat(),
                'pipeline_completed_at': datetime.now().isoformat(),
            }
            manifest = create_processing_manifest(
                video_id=video_id,
                processing_dir=self.processing_dir,
                dlc_path=dlc_path,
                step_timestamps=step_timestamps,
            )
            logger.info(
                f"Manifest created: {video_id} "
                f"(DLC={manifest.get('dlc_model', {}).get('dlc_scorer', '?')})"
            )
        except Exception as e:
            self._set_aside_stale_manifest(self.processing_dir, video_id, e)

        # --- Unified QC triage (DLC coherence / structural / cross-step / outliers) ---
        qc_verdict = 'auto_approved'
        try:
            triage_result = triage_video(
                video_id=video_id,
                processing_dir=self.processing_dir,
                h5_path=dlc_path,
            )
            qc_verdict = triage_result.verdict
            for suffix in ['_segments.json', '_reaches.json', '_pellet_outcomes.json']:
                json_path = self.processing_dir / f"{video_id}{suffix}"
                if json_path.exists():
                    try:
                        with open(json_path) as f:
                            data = json.load(f)
                        data['validation_status'] = triage_result.verdict
                        data['triage_reason'] = (
                            '; '.join(f.description for f in triage_result.flags if f.severity == 'critical')
                            if triage_result.verdict == 'needs_review'
                            else 'Unified triage: all checks passed'
                        )
                        with open(json_path, 'w') as f:
                            json.dump(data, f, indent=2)
                    except Exception:
                        pass
            triage_result.save(self.processing_dir / f"{video_id}_triage.json")
            logger.info(
                f"Triage: {video_id} -> {triage_result.verdict} "
                f"({triage_result.n_critical} critical, {triage_result.n_warnings} warnings)"
            )
        except Exception as e:
            logger.warning(f"Unified triage failed for {video_id}: {e}")

        # --- GATE: nothing reaches kinematics / connectome.db until it is CLEAN ---
        # seg soft-fail or QC-critical -> DEEP_REVIEW; any unresolved triaged
        # element -> TRIAGE. A held video's whole bundle is MOVED out of
        # Processing into the queue; STOP here -- no kinematics, no DB sync, no
        # 'processed'. It re-enters via the review -> reprocess path once cleared.
        decision = run_gate(
            video_id, self.processing_dir, self.db,
            qc_verdict=qc_verdict,
            mp4_path=self.processing_dir / f"{video_id}.mp4",
        )
        if decision != DECISION_CLEAN:
            logger.info(f"Pipeline held: {video_id} -> {decision} "
                        f"({pipeline_duration:.1f}s, kinematics deferred until cleared)")
            return

        # --- Step 4/5: Feature Extraction + DB sync (CLEAN videos ONLY) ---
        if not skip_outcomes:
            reach_path = self.processing_dir / f"{video_id}_reaches.json"
            outcome_path = self.processing_dir / f"{video_id}_pellet_outcomes.json"

            if reach_path.exists() and outcome_path.exists():
                self.db.log_step(video_id, 'feature_extraction', 'started')
                step_start = time.time()
                try:
                    from mousereach.kinematics.core.feature_extractor import FeatureExtractor
                    from mousereach.review.causal_review_io import resolve_review_path
                    extractor = FeatureExtractor()
                    # Apply the reviewer's resolution if one exists (reprocess-after-
                    # review path). Safe: None when unreviewed -> raw algo outcome.
                    review_path = resolve_review_path(video_id, self.processing_dir)
                    if review_path is not None:
                        logger.info(f"Applying human review corrections: {review_path.name}")
                    features = extractor.extract(dlc_path, reach_path, outcome_path,
                                                 review_path=review_path)

                    features_path = self.processing_dir / f"{video_id}_features.json"
                    with open(features_path, 'w') as f:
                        json.dump(features.to_dict(), f, indent=2)

                    # The manifest was composed before this ran, so it still says
                    # kinematics never happened. Stamp the truth -- including
                    # WHICH review was applied (content identity; the staleness
                    # scanner reads it back).
                    from mousereach.pipeline.manifest import record_kinematic_version
                    record_kinematic_version(video_id, self.processing_dir,
                                             extractor.VERSION,
                                             review_path=review_path)

                    feat_duration = time.time() - step_start
                    self.db.log_step(
                        video_id, 'feature_extraction', 'completed',
                        message=f"segments={features.n_segments}",
                        duration=feat_duration
                    )
                    logger.info(f"Feature extraction complete: {video_id} ({feat_duration:.1f}s)")
                    # No database push here: mousedb PULLS features files from
                    # the Analyzed tree (tool independence, 2026-08-28). The old
                    # sync call stayed behind, always failed on config, and
                    # logged a failed 'db_sync' step on every video -- removed.

                except Exception as e:
                    feat_duration = time.time() - step_start
                    self.db.log_step(video_id, 'feature_extraction', 'failed', message=str(e), duration=feat_duration)
                    logger.warning(f"Feature extraction failed for {video_id}: {e}")

        self.db.update_state(video_id, 'processed')
        self.db.log_step(
            video_id, 'pipeline', 'completed',
            message=f"All steps complete ({pipeline_duration:.1f}s total)",
            duration=pipeline_duration
        )
        logger.info(f"Pipeline complete: {video_id} ({pipeline_duration:.1f}s total)")

    # =========================================================================
    # ARCHIVE: Move processed videos to NAS
    # =========================================================================

    def _archive_to_nas(self, work: dict):
        """Archive processed video to NAS and clean up local copy."""
        from mousereach.archive.core import archive_video

        video_id = work['id']

        # Disk truth first: the bench-disagreement router (a mousedb job) moves
        # a video's whole bundle into a review queue while this watcher runs,
        # and by design it cannot write this DB then -- so the state stays
        # 'processed' while the outputs are already in the queue. Archiving the
        # leftovers then entombs a lone mp4 in Analyzed and stamps the video
        # 'archived' (2026-08-31: 19 videos reconciled by hand). If the bundle
        # is in a queue on disk, adopt that as the state and skip the archive.
        for qdir, qstate in ((Paths.TRIAGE_REVIEW, 'triage'),
                             (Paths.DEEP_REVIEW, 'deep_review')):
            if qdir and (qdir / video_id).is_dir():
                logger.info(
                    f"Not archiving {video_id}: its bundle is in the {qstate} "
                    f"queue on disk -- reconciling state instead.")
                self.db.force_state(
                    video_id, qstate,
                    reason=f"bundle found in {qstate} queue on disk at archive time")
                self._clear_archive_backoff(video_id)
                return
            # A route still running (or interrupted) builds the bundle in
            # <queue>/.incoming/<stem>/ and publishes it with one rename
            # (review_routing.move_video_bundle), so the check above cannot see
            # it yet. Archiving now would file the leftovers while the rest of
            # the bundle is on its way into the queue. Skip this cycle without
            # touching state: the next cycle sees the published bundle (and
            # adopts its state above), or the route failed and the video's
            # files are where they were.
            try:
                from mousereach.watcher.review_routing import INCOMING_DIR_NAME
            except ImportError:
                INCOMING_DIR_NAME = ".incoming"
            if qdir and (qdir / INCOMING_DIR_NAME / video_id).is_dir():
                logger.info(
                    f"Not archiving {video_id} yet: a route into the {qstate} "
                    f"queue is building its bundle ({INCOMING_DIR_NAME}).")
                return

        # Let go of the outcome detector's cached video handle first: it keeps
        # the LAST processed mp4 open until the next video replaces it, and the
        # move below then fails with "being used by another process" until it
        # does (218 videos cycled through archive backoff this way, 2026-08-31).
        try:
            from mousereach.outcomes.v6_cascade.cv_artifact_gate import release_cap_cache
            release_cap_cache()
        except Exception:
            pass

        logger.info(f"Archiving {video_id} to NAS")
        self.db.log_step(video_id, 'archive', 'started')
        start_time = time.time()

        try:
            # skip_ready_check: state='processed' already encodes what the
            # readiness gate re-derives from files on disk -- and a retry
            # after a partially-failed attempt MUST skip it, because the
            # outputs it checks for are already at the destination (the
            # 2026-08-30 stuck-retry loop). The DLC-node archive path has
            # always passed it for the same reason.
            result = archive_video(video_id, dry_run=False, verbose=False,
                                   skip_ready_check=True)
            duration = time.time() - start_time

            if result.get('success'):
                # 'processed' cannot go straight to 'archived' -- the state
                # machine routes through 'archiving' (db.py VIDEO_TRANSITIONS).
                # Jumping it raised, and because the files had ALREADY moved the
                # video stayed in 'processed' with nothing left in Processing to
                # archive: the next pass then read 'not ready' forever. Filing had
                # never actually run before today, so this had never fired.
                self.db.update_state(video_id, 'archiving')
                self.db.update_state(video_id, 'archived')
                self.db.log_step(
                    video_id, 'archive', 'completed',
                    message=f"Archived {len(result.get('files_moved', []))} files to {result.get('destination')}",
                    duration=duration
                )
                logger.info(f"Archived {video_id} to NAS ({duration:.1f}s)")

                # Export to central DB on NAS for cross-node provenance
                self.db.export_to_central_db(video_id)

                # Release multi-node claim
                self._release_claim(video_id)
                self._clear_archive_backoff(video_id)

                # Remove from staging on NAS (the DLC PC's copy)
                if self.staging_dir:
                    staging_files = self._get_associated_files(self.staging_dir, video_id)
                    for f in staging_files:
                        try:
                            f.unlink()
                            logger.debug(f"Cleaned staging: {f.name}")
                        except Exception:
                            pass
                return True
            else:
                error = result.get('error', 'archive failed')
                if error == "No files found in Processing/":
                    # Nothing to archive will not appear by waiting: the files
                    # are gone (moved by hand, archived with the state write
                    # lost, or a crash between move and write). Backoff kept
                    # one such row invisible for three days -- re-selected at
                    # every expiry, failing in 50 ms, reading as merely slow
                    # (2026-09-04). Terminal and loud; a person resets the row
                    # after restoring files (or force-states it archived if
                    # the outputs are already in Analyzed).
                    self.db.mark_failed(video_id, error)
                    self.db.log_step(video_id, 'archive', 'failed',
                                     message=error, duration=duration)
                    logger.error(
                        f"Archive impossible for {video_id}: {error} -- marked "
                        f"failed (nothing on disk to archive)")
                    return False
                # Other errors are retriable -- but not on EVERY cycle: a video
                # that cannot be archived usually cannot be archived a second
                # later either, and the loop ran at roughly one attempt per
                # second from February to August, logging 326,235 failures
                # against a handful of videos and burying everything else in the
                # log. Back off, and say so once per escalation rather than once
                # per attempt.
                delay = self._note_archive_failure(video_id)
                self.db.log_step(video_id, 'archive', 'failed', message=error, duration=duration)
                logger.warning(
                    f"Archive failed for {video_id}: {error} "
                    f"(retrying in {delay // 60}m)")
                return False

        except Exception as e:
            duration = time.time() - start_time
            # Back off here too: an exception that repeats cheaply would
            # otherwise re-select this row every cycle with no delay at all.
            self._note_archive_failure(video_id)
            self.db.log_step(video_id, 'archive', 'failed', message=str(e), duration=duration)
            logger.error(f"Archive error for {video_id}: {e}")
            # Don't mark_failed — keep in processed state for retry
            return False


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

WatcherOrchestrator = DLCOrchestrator
