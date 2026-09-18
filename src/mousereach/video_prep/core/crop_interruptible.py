"""Crop a collage in a child process that can be stopped part-way.

WHY THIS EXISTS
---------------
A GPU node in a behaviour room records videos and, when nobody is recording,
prepares and poses them. Posing already stops the moment a recording program
appears (dlc/core/interruptible.py). Cropping did not: it runs ffmpeg once per
animal inside the watcher, for minutes, with no way to interrupt it -- so an
operator who opened the recording program still had to wait for the crop to
finish before the machine was theirs.

This runs the same crop in a child process and kills it, and everything it
started, as soon as the caller says to stop. The partial crops of THAT run are
deleted and the collage goes back in the queue to be cropped again later:
re-cropping costs minutes of a machine nobody is using, while a recording that
drops frames is behaviour that can never be filmed again.

WHAT IS DELETED ON A STOP
-------------------------
Only files in the output folder that (a) were not there when the child started
AND (b) were written at or after the start, AND (c) end in .mp4 (a cropped
single) or .json (the crop's provenance manifest). The collage copy itself
(.mkv) and anything older are never touched. Two independent checks, because
this cleanup runs in a folder that other work also writes into.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, List, Optional, Set

# Shared with the pose runner rather than copied: killing a process tree, tying
# a child's life to this process, and the "if you cannot answer, stop" rule are
# the same problem, and two copies would drift apart.
from mousereach.dlc.core.interruptible import (
    _ascii, _ask, _close_job, _kill_process_tree, _log_tail, _tie_child_to_this_process,
    _unlink_quietly,
)

import logging

logger = logging.getLogger(__name__)

DEFAULT_CROP_WORKER = "mousereach.video_prep.core.crop_worker"

# What a crop writes beside the collage: the single-animal videos and the crop
# manifest. A closed list, so cleanup can never remove something else.
_CROP_SUFFIXES = (".mp4", ".json")
_MTIME_ALLOWANCE = 2.0


def _crop_outputs(output_dir: Path) -> List[Path]:
    try:
        return [p for p in output_dir.iterdir()
                if p.is_file() and p.name.lower().endswith(_CROP_SUFFIXES)]
    except OSError:
        return []


def _remove_new_partial_outputs(output_dir: Path, before: Set[str], started_at: float) -> List[Path]:
    """Remove the crop files THIS run created, and nothing else."""
    removed = []
    for path in _crop_outputs(output_dir):
        if path.name in before:
            continue
        try:
            if path.stat().st_mtime < started_at - _MTIME_ALLOWANCE:
                continue
            path.unlink()
            removed.append(path)
        except OSError as e:
            logger.debug(f"could not remove partial crop {path}: {e}")
    return removed


def run_crop_collage_interruptible(
    input_path,
    output_dir,
    should_abort: Callable[[], Optional[str]],
    poll_seconds: float = 2.0,
    python_exe: Optional[str] = None,
    worker_module: str = DEFAULT_CROP_WORKER,
    work_dir: Optional[Path] = None,
) -> dict:
    """Crop one collage in a child process, stopping if ``should_abort`` says so.

    Args:
        input_path: The collage video to crop.
        output_dir: Where the single-animal videos are written.
        should_abort: Called before starting and every poll_seconds while the
            crop runs; None to carry on, or a short reason to stop. If it
            raises, the crop stops (a question about a recording that cannot be
            answered is not a reason to keep the machine busy).
        poll_seconds: How often to ask. Two seconds: a recording gets the
            machine back at once, and asking costs nothing next to ffmpeg.
        python_exe / worker_module / work_dir: for tests and unusual installs.

    Returns:
        {"collage", "status": "success", "results": [...]} -- the crop results,
        exactly as crop_collage returned them; or
        {"collage", "status": "aborted", "abort_reason": ...} with this run's
        partial crops removed; or {"collage", "status": "failed", "error": ...}.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    reason = _ask(should_abort)
    if reason:
        logger.info(_ascii(f"Not cropping {input_path.name}: {reason}"))
        return {"collage": str(input_path), "status": "aborted", "abort_reason": reason}

    work_dir = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir())
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    args_path = work_dir / f"mousereach_crop_{input_path.stem}_{token}.args.json"
    result_path = work_dir / f"mousereach_crop_{input_path.stem}_{token}.result.json"
    log_path = work_dir / f"mousereach_crop_{input_path.stem}_{token}.log"

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    keep_log = False
    proc = None
    job = None
    log_file = None
    started_at = time.time()
    before: Set[str] = set()
    try:
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump({"input_path": str(input_path), "output_dir": str(output_dir),
                       "result_path": str(result_path)}, f)
        log_file = open(log_path, "w", encoding="utf-8", errors="replace")

        started_at = time.time()
        before = {p.name for p in _crop_outputs(output_dir)}

        proc = subprocess.Popen(
            [python_exe or sys.executable, "-m", worker_module, str(args_path)],
            stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=env, creationflags=creationflags, close_fds=True)
        job = _tie_child_to_this_process(proc)

        abort_reason = None
        while True:
            try:
                proc.wait(timeout=poll_seconds)
                break
            except subprocess.TimeoutExpired:
                pass
            abort_reason = _ask(should_abort)
            if abort_reason:
                break

        if abort_reason and proc.poll() is None:
            logger.info(_ascii(f"Stopping the crop of {input_path.name}: {abort_reason}"))
            _kill_process_tree(proc)
            log_file.close()
            removed = _remove_new_partial_outputs(output_dir, before, started_at)
            if removed:
                logger.info(_ascii(
                    f"Removed {len(removed)} partial crop file(s) of {input_path.name}: "
                    + ", ".join(p.name for p in removed)))
            return {"collage": str(input_path), "status": "aborted",
                    "abort_reason": abort_reason}

        # The child finished on its own (a finished crop is kept, even if a
        # stop was asked for in the same instant).
        log_file.close()
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            if not isinstance(result, dict) or "status" not in result:
                raise ValueError("result file has no status")
        except (OSError, ValueError) as e:
            keep_log = True
            tail = _log_tail(log_path)
            error = (f"crop child process exited (code {proc.returncode}) without a "
                     f"result ({_ascii(e)}); its output is kept in {log_path}")
            if tail:
                error += "; last lines:\n" + tail
            return {"collage": str(input_path), "status": "failed", "error": error}
        result.setdefault("collage", str(input_path))
        if result.get("status") != "success":
            keep_log = True
            logger.warning(_ascii(
                f"Crop of {input_path.name} failed; its output is kept in {log_path}"))
        return result
    finally:
        if proc is not None and proc.poll() is None:
            _kill_process_tree(proc)
            try:
                _remove_new_partial_outputs(output_dir, before, started_at)
            except Exception:
                pass
        _close_job(job)
        if log_file is not None and not log_file.closed:
            log_file.close()
        _unlink_quietly(args_path)
        _unlink_quietly(result_path)
        _unlink_quietly(result_path.with_name(result_path.name + ".tmp"))
        if not keep_log:
            _unlink_quietly(log_path)
