"""The single-machine role ("standalone"): one PC that does everything.

WHAT IT IS
----------
``watcher.mode = "standalone"`` in ~/.mousereach/config.json. One machine,
with a graphics card, that

  * cuts multi-animal recordings into single-animal videos,
  * poses them (DeepLabCut, on the graphics card),
  * analyses them (segmentation, reaches, outcomes, causal reach, quality
    check, the hand-score comparison when hand scores are configured, the
    review gate, kinematics),
  * takes reviewed videos back once a person has answered them (the review
    return), and
  * files the finished results into ``Analyzed/<project>/<cohort>/``.

``mousereach-watch --once``, the "Run the pipeline now" button and the
Watcher Control tab's Start / Run Once all run it.

HOW IT IS BUILT, AND WHY
------------------------
Until 2026-11-19 a machine ran exactly one of the watcher's two programs:

  * ``dlc_pc`` (DLCOrchestrator): cut, pose, then hand the pose on through
    ``Processing/Posed``;
  * ``processing_server`` (ProcessingOrchestrator): take posed videos in from
    ``Processing/Posed``, analyse, gate, take reviewed videos back, file.

A lab with one PC could not have both. ``dlc_pc`` with ``also_process`` on
analyses and files what it posed, but through a separate, older copy of the
analysis (``_run_local_pipeline``) that has no review return: every video the
gate sent to a review queue stayed there for good, whatever a person answered.

The standalone role does NOT add a third copy of any step. It runs the two
existing programs, unchanged, on the same machine:

  * the POSING half is a DLCOrchestrator with ``also_process`` off: it cuts and
    poses, and puts each pose in ``Processing/Posed`` exactly as a graphics-card
    PC does;
  * the PROCESSING half is a ProcessingOrchestrator: it takes those poses in
    and does everything after them, exactly as the processing server does,
    review return included.

So every step, check, hold and filing rule is the one the multi-machine lab
uses, and a fix to either program reaches a single-machine lab too.

The two halves keep separate watcher databases, as two machines would:

  * the processing half uses THE watcher database (``watcher.db_path``, or
    ``<processing folder>/watcher.db`` when unset). It is the one every tool
    reads -- the status command, the review tools, Run All Steps, crystallize
    -- because the videos from posing onwards live there;
  * the posing half keeps its own small database beside it,
    ``<processing folder>/watcher_posing.db``, for the recordings it has cut
    and the videos it is posing. WHY separate: the processing half registers a
    posed video when it finds it in ``Processing/Posed`` and only registers
    videos it does not know yet; one shared database would already know every
    video from the posing half and nothing would ever be taken in.

Order of work. ``run_once`` (one pass: the button, ``--once``) runs the posing
half until it has nothing left (every waiting recording cut, every cut video
posed and handed on), then the processing half until it has nothing left. One
press therefore takes a video from the intake folder to filed, or to a review
queue. The automatic watcher (``run``) runs both halves at the same time, each
in its own thread, so the graphics card poses while the processor analyses.
A stop request (``watcher_stop.flag`` or the Stop button) stops both after
their current item; a pause pauses both.

Each half has its own temporary folder, so the processing half finishing
first can never clear a crop the posing half is still writing.

ASCII-only console output (Windows cp1252).
"""
from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

STANDALONE = "standalone"
POSE_ONLY = "dlc_pc"
PROCESSING_SERVER = "processing_server"
KNOWN_ROLES = (POSE_ONLY, PROCESSING_SERVER, STANDALONE)

# The posing half's own database, beside the processing folder (module docstring).
POSING_DB_NAME = "watcher_posing.db"
# The processing half's temporary folder (the posing half keeps watcher_working).
PROCESSING_WORKING_NAME = "watcher_working_processing"


def analyses_and_files(mode) -> bool:
    """True when a machine of role ``mode`` holds the database of the analysed
    and filed videos: the processing server, or a single machine. These are the
    roles on which the review return, Run All Steps, crystallize and the
    processing-side checks work."""
    return (mode or "") in (PROCESSING_SERVER, STANDALONE)


def poses(mode) -> bool:
    """True when a machine of role ``mode`` cuts and poses (needs a graphics
    card, ffmpeg and a DLC model)."""
    return (mode or "") in (POSE_ONLY, STANDALONE)


def role_label(mode) -> str:
    """A short human name for the role, for banners and tabs."""
    if mode == PROCESSING_SERVER:
        return "Processing Server"
    if mode == STANDALONE:
        return "Single machine (cut + pose + analyse + review + file)"
    return "GPU node (DLC)"


def posing_db_path() -> Path:
    """The posing half's database on a single machine (module docstring)."""
    from mousereach.config import require_processing_root
    return require_processing_root() / POSING_DB_NAME


def split_config(config) -> Tuple[object, object]:
    """(posing config, processing config) for a standalone ``config``.

    Copies, never the caller's object: the posing half is a pose-only
    graphics-card node (``also_process`` off, so it hands its poses on through
    Processing/Posed instead of running the older local analysis) whose
    database is the posing database; the processing half is the processing
    server. Every other setting -- recording hours, pause programs, DLC model,
    retries, priorities -- is the same for both halves."""
    pose_cfg = copy.copy(config)
    pose_cfg.mode = POSE_ONLY
    pose_cfg.also_process = False
    try:
        pose_cfg.db_path = posing_db_path()
    except Exception:
        pose_cfg.db_path = None
    proc_cfg = copy.copy(config)
    proc_cfg.mode = PROCESSING_SERVER
    proc_cfg.also_process = False
    return pose_cfg, proc_cfg


class StandaloneOrchestrator:
    """Both watcher programs on one machine (module docstring).

    ``db`` is THE watcher database (the processing half's). The posing half's
    database is opened (created when missing) at ``posing_db_path()``.
    """

    def __init__(self, config, db, posing_db=None):
        from mousereach.watcher.db import WatcherDB
        from mousereach.watcher.orchestrator import DLCOrchestrator, ProcessingOrchestrator
        self.config = config
        self.db = db
        pose_cfg, proc_cfg = split_config(config)
        if posing_db is None:
            posing_db = WatcherDB(Path(pose_cfg.db_path or posing_db_path()))
        self.posing_db = posing_db
        logger.info("Single-machine role: posing half (database %s), then "
                    "processing half (database %s)",
                    getattr(posing_db, "db_path", "?"), getattr(db, "db_path", "?"))
        self.posing = DLCOrchestrator(pose_cfg, posing_db)
        self.processing = ProcessingOrchestrator(proc_cfg, db)
        # Its own temporary folder: each half clears its temporary folder when
        # it stops, and in the automatic watcher the processing half can stop
        # while the posing half is still cropping into watcher_working.
        try:
            from mousereach.config import require_processing_root
            wd = require_processing_root() / PROCESSING_WORKING_NAME
            wd.mkdir(parents=True, exist_ok=True)
            self.processing.working_dir = wd
        except Exception as e:
            logger.warning(f"Could not give the processing half its own temporary "
                           f"folder ({e}); it shares watcher_working")

    # The two halves, in the order one pass runs them.
    @property
    def halves(self):
        return (("posing", self.posing), ("processing", self.processing))

    def run_once(self):
        """One pass: pose everything waiting, then analyse everything posed.

        The posing half first, so a recording dropped in the intake folder is
        cut, posed and handed on before the processing half looks; the
        processing half then takes those poses in, along with any video a
        person released from a review queue, and files or holds each one."""
        logger.info("Single machine: one pass (posing half, then processing half)")
        for name, half in self.halves:
            try:
                half.run_once()
            except Exception as e:
                # The other half still runs: a failed crop must not keep the
                # videos already posed from being analysed.
                logger.error(f"Single machine: the {name} half stopped with an "
                             f"error ({e}); going on with the rest", exc_info=True)
        logger.info("Single machine: pass complete")

    def run(self, shutdown_event: threading.Event):
        """The automatic watcher: both halves at once, each in its own thread,
        until a stop is requested (the stop flag stops both; ``shutdown_event``
        is shared)."""
        threads = []
        for name, half in self.halves:
            t = threading.Thread(target=self._run_half, args=(name, half, shutdown_event),
                                 name=f"watcher-{name}", daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    @staticmethod
    def _run_half(name, half, shutdown_event):
        try:
            half.run(shutdown_event)
        except Exception as e:
            logger.error(f"Single machine: the {name} half crashed ({e})", exc_info=True)

    def shutdown(self):
        for _name, half in self.halves:
            try:
                half.shutdown()
            except Exception:
                pass


def make_orchestrator(config, db):
    """The watcher program for this machine's role: the ONE place the role is
    turned into an orchestrator (mousereach-watch, the Watcher Control tab's
    Start and Run Once, the Run Pipeline tab)."""
    from mousereach.watcher.orchestrator import DLCOrchestrator, ProcessingOrchestrator
    mode = getattr(config, "mode", None)
    if mode == STANDALONE:
        return StandaloneOrchestrator(config, db)
    if mode == PROCESSING_SERVER:
        return ProcessingOrchestrator(config, db)
    return DLCOrchestrator(config, db)


__all__ = ["STANDALONE", "KNOWN_ROLES", "POSING_DB_NAME", "analyses_and_files", "poses",
           "role_label", "posing_db_path", "split_config", "StandaloneOrchestrator",
           "make_orchestrator"]
