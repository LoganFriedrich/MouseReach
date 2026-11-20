"""Run ONE chosen video through the watcher's own steps, now, by hand.

WHAT THIS IS
------------
The code behind the launcher's "Run All Steps" tab. A person picks one video
and presses Run; this drives the processing machine's watcher
(``orchestrator.ProcessingOrchestrator``) for that one video: take it in from
the posed folder, run the analysis (segmentation, reach detection, outcome
detection, reach assignment), the quality check, the hand-score comparison,
the review gate and the kinematics, and file it. Each of those is the
watcher's own handler (``_dispatch_work``), called exactly as the automatic
watcher calls it; nothing here runs a step itself.

WHY (one implementation, two ways to start it)
----------------------------------------------
Running the pipeline by hand from a window must do what the automatic watcher
does: same steps, same order, same gates and holds, same outputs. The tab used
to run its own older copy of the steps in whatever folder the chosen video sat
in: no hand-score comparison, a quality check that let a video through when it
could not run, a push into a database, and no check of where the video was.
Pointed at a filed video it rewrote the filed results in place (no earlier
generation kept, no watcher record); pointed at a video in a review queue it
rewrote the algorithm outputs under the reviewer. A video now goes the one way
the watcher sends it, or it is not run at all and the person is told why.

WHAT IS REFUSED (nothing is run or moved; the reason is shown)
--------------------------------------------------------------
  * a video inside the filed tree, or whose watcher record says it is filed
    ('archived', 'outdated'): filed means done; a filed video is run again
    with the dashboard's "Re-run video" or the version re-run;
  * a video held in a review queue (a live triage or deep-review bundle, or a
    record saying 'triage'/'deep_review'): it goes on by itself once the
    review is finished;
  * a crystallized (publication-locked) video;
  * a video the watcher on this machine does not know and that is not waiting
    in the posed folder (it has no record to run);
  * a video not yet posed, quarantined, or with no file on this machine;
  * when this machine is not the processing machine or a single machine
    (``standalone``, whose processing half is the processing server's
    program; a graphics-card PC set to analyse does so through its own
    watcher), when a watcher already runs on it (it does the video by itself;
    "Put first" on the Run Pipeline tab puts the video's animal first), when a
    stop is requested, or when the watcher is paused.

WHAT RUNS, BY THE VIDEO'S WATCHER STATE
---------------------------------------
  'dlc_complete' (posed, waiting)  -> take it in, then analyse, then file
  'processing'   (taken in)        -> analyse, then file
  'processed'    (passed, unfiled) -> file
  'failed' (before the run)        -> given its retries back once, as
                                      mousereach-watch-reprocess does, then
                                      from the step that reset gives it
After each step the record is read again; the run stops when the video is
filed, held (a review queue, or waiting in Processing for its hand scores or
for a queue folder to be cleared), failed, or did not move on. A step that
fails during the run is not retried here: the run reports [FAIL] and the
watcher's own retry rules apply.

The same guards as Run Once: one watcher per machine (the lock is held for
the whole run), no stop request, not paused. The hand-score holds this
machine already has are read first (``_restore_score_waits``), so this run
neither re-analyses a held video in full nor drops the other held videos from
this machine's list.

ASCII-only text (it is printed and logged on Windows consoles).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Watcher state -> the watcher's work type for it (ProcessingOrchestrator.
# _dispatch_work). Only these states are work a person can start here.
WORK_FOR_STATE = {
    "dlc_complete": "intake",
    "processing": "pipeline",
    "processed": "archive",
}

# The most steps one run takes: intake, pipeline, archive, plus a reset.
_MAX_STEPS = 6

VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv")


def video_id_of(video) -> str:
    """The video id of a chosen file (its name without the extension) or of
    an id given as text."""
    p = Path(str(video))
    return p.stem if p.suffix.lower() in VIDEO_SUFFIXES else p.name


def _inside(path, root) -> bool:
    try:
        if not root:
            return False
        p, r = Path(path).resolve(), Path(root).resolve()
        return p == r or r in p.parents
    except Exception:
        return False


def queue_holding(video_id: str) -> Optional[str]:
    """The review queue holding a live bundle of ``video_id`` ('deep review'
    or 'triage', published or still being built in ``.incoming``), or None.
    A leftover kept for a person is not a bundle (review_routing.
    is_queue_bundle)."""
    from mousereach.config import Paths
    from mousereach.watcher.review_routing import is_queue_bundle, INCOMING_DIR_NAME
    for name, root in (("deep review", getattr(Paths, "DEEP_REVIEW", None)),
                       ("triage", getattr(Paths, "TRIAGE_REVIEW", None))):
        if not root:
            continue
        if (is_queue_bundle(Path(root) / video_id)
                or is_queue_bundle(Path(root) / INCOMING_DIR_NAME / video_id)):
            return name
    return None


def where_refusal(video) -> Optional[str]:
    """Why the chosen file's place alone rules the run out, or None.
    Read before anything is opened: the filed tree and the review queues."""
    from mousereach.config import Paths
    from mousereach.pipeline.analyzed_tree import filed_tree_refusal
    p = Path(str(video))
    video_id = video_id_of(video)
    if p.suffix and filed_tree_refusal(p.parent):
        return (f"{video_id} is filed ({p.parent} is in the filed results). Filed "
                f"means done: its results are not run again here. To run a filed "
                f"video again use \"Re-run video\" on the dashboard, or the "
                f"version re-run (mousereach-version-check --mark).")
    for root in (getattr(Paths, "TRIAGE_REVIEW", None), getattr(Paths, "DEEP_REVIEW", None)):
        if p.suffix and _inside(p.parent, root):
            return (f"{video_id} is in a review queue ({p.parent}). It goes on by "
                    f"itself once its review is finished; open it from the Review "
                    f"Queues tab.")
    held = queue_holding(video_id)
    if held:
        return (f"{video_id} is held in the {held} queue. It goes on by itself once "
                f"its review is finished; open it from the Review Queues tab.")
    return None


def state_refusal(row: dict) -> Optional[str]:
    """Why a video in this watcher state is not run here, or None when the
    state is work this run can start (WORK_FOR_STATE, or 'failed')."""
    video_id = row.get("video_id") or "?"
    state = row.get("state")
    if state in WORK_FOR_STATE or state == "failed":
        return None
    if state in ("archived", "outdated"):
        return (f"{video_id} is filed (watcher state '{state}'). Filed means done: "
                f"it is run again with \"Re-run video\" on the dashboard, or by the "
                f"version re-run.")
    if state == "crystallized":
        return (f"{video_id} is crystallized (locked for a publication); not run. "
                f"Unlock it first with mousereach-uncrystallize if this is intended.")
    if state in ("triage", "deep_review"):
        return (f"{video_id} is held for a person in the {state.replace('_', ' ')} "
                f"queue. It goes on by itself once the review is finished.")
    if state == "archiving":
        return (f"{video_id} is being filed right now (watcher state 'archiving'); "
                f"nothing is started on it.")
    if state in ("discovered", "validated", "dlc_queued", "dlc_running"):
        return (f"{video_id} has no pose yet (watcher state '{state}'). The pose is "
                f"made on a graphics-card PC; once its pose is in the posed folder, "
                f"run it again.")
    if state == "quarantined":
        return (f"{video_id} is quarantined; see the Quarantine tab "
                f"(mousereach-watch-quarantine --list).")
    if state == "unresolvable":
        return (f"This machine has no file for {video_id} (watcher state "
                f"'unresolvable': {row.get('error_message') or '?'}). "
                f"mousereach-watch-unresolvable --list says why.")
    return f"{video_id} is in watcher state '{state}', which is not work this tab starts."


class _VideoLogRelay(logging.Handler):
    """Hands the watcher's log lines about one video to the window's log."""

    def __init__(self, video_id: str, emit: Callable[[str, str], None]):
        super().__init__(level=logging.INFO)
        self.video_id = video_id
        self._emit_to = emit

    def emit(self, record):
        try:
            msg = record.getMessage()
            if self.video_id in msg:
                self._emit_to("watcher", msg)
        except Exception:
            pass


def run_one_video(video, progress: Optional[Callable[[str, str], None]] = None) -> Dict:
    """Run one chosen video through the watcher's own steps (module docstring).

    ``video`` is the chosen file (or its id). ``progress(stage, msg)`` is
    called as the run goes. Returns a summary: ``video``, ``state_before``,
    ``state_after``, ``stages`` (list of {stage, msg}), ``refused`` (why
    nothing was run, or None), ``held`` (None, 'triage', 'deep_review' or
    'processing'), ``hold_reason``, ``error`` and ``done`` (True when the
    video was filed). Never raises."""
    video_id = video_id_of(video)
    result: Dict = {"video": video_id, "state_before": None, "state_after": None,
                    "stages": [], "refused": None, "held": None, "hold_reason": None,
                    "error": None, "done": False}

    def _p(stage: str, msg: str = "") -> None:
        result["stages"].append({"stage": stage, "msg": msg})
        if progress:
            try:
                progress(stage, msg)
            except Exception:
                pass

    def _refuse(why: str) -> Dict:
        result["refused"] = why
        _p("not run", why)
        return result

    try:
        why = where_refusal(video)
        if why:
            return _refuse(why)

        from mousereach.config import WatcherConfig, require_processing_root
        try:
            cfg = WatcherConfig.load()
        except Exception as e:
            return _refuse(f"MouseReach is not set up on this machine ({e}). Use the "
                           f"Setup tab (mousereach-setup), then come back.")
        from mousereach.watcher.standalone import analyses_and_files, STANDALONE, split_config
        if getattr(cfg, "mode", None) == STANDALONE:
            # A single machine: the run is its processing half, the processing
            # server's own program (watcher/standalone.py).
            cfg = split_config(cfg)[1]
        if not analyses_and_files(getattr(cfg, "mode", None)):
            return _refuse(
                f"This machine's role is '{getattr(cfg, 'mode', '?')}', not the processing "
                f"machine, and this tab runs only on the processing machine (a "
                f"graphics-card PC set to analyse does so through its own watcher). "
                f"Run the video on the processing machine (or press \"Run the pipeline now\" on the Run "
                f"Pipeline tab here for this machine's own work).")

        from mousereach.watcher import singleton
        try:
            from mousereach.watcher.health import watcher_running
            import os
            pid = watcher_running()
            if pid and pid != os.getpid():
                return _refuse(
                    "A watcher is already running on this machine; it does this video "
                    "by itself. To have it done first, put its animal first on the Run "
                    "Pipeline tab (\"Put first\").")
        except Exception as e:
            logger.debug(f"running-watcher check unavailable: {e}")
        try:
            left = singleton.stop_flag_left(require_processing_root() / "watcher_stop.flag")
        except Exception:
            left = None
        if left:
            return _refuse(singleton.stop_flag_message(*left))
        lock = None
        try:
            lock = singleton.acquire_watcher_lock()
            if lock is None:
                return _refuse("A watcher is already running on this machine (another "
                               "login or window); it does this video by itself.")
        except singleton.LockUnavailable as e:
            logger.warning(f"Single-instance guard unavailable ({e}); continuing")
            lock = None

        relay = _VideoLogRelay(video_id, _p)
        root_logger = logging.getLogger("mousereach")
        root_logger.addHandler(relay)
        # The launcher configures no logging, so the package's INFO lines
        # (the watcher's own account of each step) would never be made; they
        # are let through for this run only, for the window's log.
        old_level = root_logger.level
        if root_logger.getEffectiveLevel() > logging.INFO:
            root_logger.setLevel(logging.INFO)
        try:
            return _drive(video_id, video, cfg, result, _p, _refuse)
        finally:
            root_logger.setLevel(old_level)
            root_logger.removeHandler(relay)
            singleton.release_watcher_lock(lock)
    except Exception as e:
        logger.exception("run_one_video failed")
        result["error"] = str(e)
        _p("error", str(e))
        return result


def _drive(video_id, video, cfg, result, _p, _refuse) -> Dict:
    """The run itself, under the watcher lock (run_one_video)."""
    from mousereach.watcher.db import crystallized_refusal
    from mousereach.watcher.db_location import open_watcher_db_if_present
    from mousereach.watcher.orchestrator import ProcessingOrchestrator

    # Never create a database: WatcherDB(path) would build an empty decoy and
    # then 'no record' would be the confusing answer.
    db, why_no_db = open_watcher_db_if_present()
    if db is None:
        return _refuse(f"There is no watcher database on this machine to run the video "
                       f"against ({why_no_db}). Check the processing folder in "
                       f"mousereach-setup; nothing was started.")
    orch = ProcessingOrchestrator(cfg, db)
    # This machine's hand-score holds, read first (module docstring).
    orch._score_waits = orch._restore_score_waits()
    if orch._run_once_blocked("before starting"):
        return _refuse("The watcher on this machine is paused or a stop is requested "
                       f"({getattr(orch, '_pause_reason', None) or 'watcher_stop.flag'}); "
                       f"nothing was started.")

    row = db.get_video(video_id)
    if row is None and orch.staging_dir:
        # Waiting in the posed folder but not yet registered: the watcher's
        # own discovery step (the first thing each pass does) registers it.
        staged = Path(orch.staging_dir)
        if (staged / f"{video_id}.mp4").is_file() or any(staged.glob(f"{video_id}DLC*.h5")):
            _p("intake", "registering it from the posed folder (the watcher's scan)")
            orch.state.discover_dlc_staged(orch.staging_dir)
            row = db.get_video(video_id)
    if row is None:
        return _refuse(
            f"The watcher on this machine has no record of {video_id}, and it is not "
            f"waiting in the posed folder ({orch.staging_dir}). Only a video the watcher "
            f"knows can be run here; a posed video goes into the posed folder with its "
            f"pose, where the watcher takes it in.")
    result["state_before"] = row.get("state")
    locked = crystallized_refusal(db, video_id, "run")
    if locked:
        return _refuse(locked)
    why = state_refusal(row)
    if why:
        return _refuse(why)

    # A video that had FAILED before this run is given its retries back once
    # (mousereach-watch-reprocess's reset) -- pressing Run on it is a person
    # asking for it to be run again. A failure DURING this run is never reset
    # here: the run stops and says [FAIL], and the watcher's own retry rules
    # apply from there (resetting it in a loop ran the failing step again and
    # again and wiped its retry count).
    if row.get("state") == "failed":
        if orch._filed_rerun_gave_up(row):
            _p("failed", "a filed video's re-run that has failed for good: put back "
                         "as filed, as the watcher does")
            orch._restore_filed_after_failed_rerun(
                video_id, f"run by hand after it failed for good; last error: "
                          f"{row.get('error_message') or '?'}")
        else:
            new_state = db.reset_failed(video_id, dlc_queue=None,
                                        processing_dir=orch.processing_dir)
            if not new_state:
                result["error"] = (f"it failed before ({row.get('error_message') or '?'}) "
                                   f"and could not be reset")
            else:
                _p("reset", f"failed before ({row.get('error_message') or '?'}); retries "
                            f"given back, starts again at '{new_state}'")

    stalled = None
    for _ in range(_MAX_STEPS if not result["error"] else 0):
        row = db.get_video(video_id) or {}
        state = row.get("state")
        work_type = WORK_FOR_STATE.get(state)
        if work_type is None:
            break                      # filed, held, failed, or not work for this run
        if orch._run_once_blocked(f"before its '{work_type}' step"):
            result["error"] = "stopped: the watcher was paused or a stop was requested"
            break
        _p(work_type, "running")
        orch._dispatch_work({"type": work_type, "id": video_id, "data": row})
        after = db.get_video(video_id) or {}
        if after.get("state") == state:
            stalled = (work_type, after)
            break                      # held where it is, or did not move on

    final = db.get_video(video_id) or {}
    state = final.get("state")
    result["state_after"] = state
    if state == "archived":
        result["done"] = True
        _p("filed", "passed every check and filed")
    elif state in ("triage", "deep_review"):
        result["held"] = state
        result["hold_reason"] = final.get("error_message") or "held for a person"
        _p("held", f"sent to the {state.replace('_', ' ')} queue")
    elif state == "processing":
        result["held"] = "processing"
        wait = (getattr(orch, "_score_waits", {}) or {}).get(video_id)
        if wait:
            result["hold_reason"] = ("waiting for its hand scores"
                                     + (f" ({wait.get('error')})" if wait.get("error") else "")
                                     + "; it goes on once they are imported")
        elif orch._held_for_refused_route(video_id):
            result["hold_reason"] = ("its review-queue folder is in the way; see "
                                     "'Folders left for a person' on the Review Queues tab")
        else:
            result["hold_reason"] = final.get("error_message") or "left in Processing"
        _p("held", result["hold_reason"])
    elif state == "failed":
        result["error"] = final.get("error_message") or "failed"
        _p("failed", result["error"])
    elif not result["error"]:
        # Stopped without moving on (e.g. the intake found the video claimed
        # by another machine, or filing waits out an earlier failure): say
        # which step and what the record says, never only the state.
        if stalled is not None:
            work_type, after = stalled
            result["error"] = (f"its '{work_type}' step did not move it on (watcher state "
                               f"still '{state}'): {after.get('error_message') or 'see the log above'}")
            _p("stopped", result["error"])
        else:
            _p("stopped", f"watcher state now '{state}'")
    return result


__all__ = ["run_one_video", "where_refusal", "state_refusal", "queue_holding",
           "video_id_of", "WORK_FOR_STATE"]
