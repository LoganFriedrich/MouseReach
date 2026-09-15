"""Pose ONE video in a child process that can be stopped part-way.

WHY THIS EXISTS: a GPU node in a room where videos are also recorded poses
videos whenever nobody is recording. Recording must always win. But
deeplabcut.analyze_videos runs in-process and offers no way to interrupt it:
once started, a pose runs to the end, about fourteen minutes per video. A
recording program started in that window would compete with the pose for the
GPU, CPU and disk the whole time -- dropped frames in a recording cannot be
recovered, while a pose can simply be run again later.

So the pose runs in a separate Python process
(mousereach.dlc.core.interruptible_worker), and this parent process keeps
asking a caller-supplied question -- "should this stop?" -- every few seconds.
When the answer is a reason, the child and everything it started are killed,
the half-written pose files from THIS run are removed (so nothing downstream
mistakes a partial pose for a finished one), and the caller gets a result
whose status is 'aborted' instead of 'success' or 'failed'. An aborted pose is
not a failure of the video: the caller should leave it queued, not spend its
retry budget.

WHY THE PARENT STAYS LIGHT: this module never imports deeplabcut (or
tensorflow / torch). Those imports take many seconds and a lot of memory; only
the child pays for them, and killing the child gives all of it back.
"""

import glob
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, List, Optional, Set

logger = logging.getLogger(__name__)

DEFAULT_WORKER_MODULE = "mousereach.dlc.core.interruptible_worker"

# DeepLabCut writes these beside a video named <stem>.mp4:
#   <stem>DLC_<scorer>.h5, <stem>DLC_<scorer>.csv,
#   <stem>DLC_<scorer>_meta.pickle (and on some versions a _full.pickle).
# Only files with these endings are ever removed after an abort. WHY a closed
# list: cleanup must never be able to delete something that is not a pose file,
# however the folder is shared.
_POSE_SUFFIXES = (".h5", ".csv", ".pickle")

# A file counts as "written by this run" only if its modified time is at or
# after the moment the run started. WHY the small allowance: some file systems
# store modified times in coarse steps (two seconds on FAT-formatted drives),
# so a file created in the first instant of the run can carry a timestamp that
# rounds to just before it. It cannot widen the cleanup to an older file: a file
# must ALSO be absent from the listing taken immediately before the child
# started.
_MTIME_SLACK_S = 2.0

# How long to wait for killed processes to actually exit before falling back
# to a harder kill. WHY wait at all: on Windows a file held open by a process
# cannot be deleted, so partial pose files can only be removed once the child
# is really gone.
_KILL_WAIT_S = 10.0

# How many lines of the child's output to quote when it dies without a result.
_LOG_TAIL_LINES = 20


def _ascii(text: str) -> str:
    """Make text safe to log on a Windows console.

    WHY: DeepLabCut, TensorFlow and progress bars print non-ASCII characters,
    and one such character in a log line crashes a cp1252 console.
    """
    return str(text).encode("ascii", "replace").decode("ascii")


def _pose_outputs(output_dir: Path, stem: str) -> List[Path]:
    """This video's DeepLabCut output files in output_dir (<stem>DLC*)."""
    pattern = os.path.join(glob.escape(str(output_dir)), glob.escape(stem) + "DLC*")
    return [Path(p) for p in glob.glob(pattern)
            if p.lower().endswith(_POSE_SUFFIXES) and os.path.isfile(p)]


def _remove_new_partial_outputs(output_dir: Path, stem: str, before: Set[str],
                                started_at: float) -> List[Path]:
    """Remove pose files for `stem` that THIS run created, and nothing else.

    A file is removed only when BOTH hold:
      * its name was not in the listing taken just before the child started
        (WHY: a pose from an earlier, finished run -- possibly from another
        model -- must survive an aborted re-pose), and
      * its modified time is at or after the run's start (WHY: a second,
        independent check, so a listing mistake alone can never delete an old
        file).
    """
    removed = []
    for path in _pose_outputs(output_dir, stem):
        if path.name in before:
            continue
        try:
            if path.stat().st_mtime < started_at - _MTIME_SLACK_S:
                continue
        except OSError:
            continue
        # A just-killed process can hold its file for a moment after exit on
        # Windows, so try a few times before giving up.
        for attempt in range(5):
            try:
                path.unlink()
                removed.append(path)
                break
            except FileNotFoundError:
                break
            except OSError as e:
                if attempt == 4:
                    logger.warning(_ascii(
                        f"Could not remove partial pose file {path} after an "
                        f"aborted pose: {e}. Delete it by hand; it is NOT a "
                        f"finished pose."))
                else:
                    time.sleep(0.5)
    return removed


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Stop the child and every process it started, children first.

    WHY children first: DeepLabCut and its libraries can start helper
    processes. Killing the parent first would leave those running -- still
    holding the GPU and the partial files -- with nobody left to find them.
    """
    try:
        import psutil
    except ImportError:
        psutil = None

    if psutil is not None:
        try:
            parent = psutil.Process(proc.pid)
            children = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            parent, children = None, []
        except Exception:
            parent, children = None, []
        procs = list(reversed(children)) + ([parent] if parent is not None else [])
        for p in procs:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
            except Exception:
                pass
        try:
            _, alive = psutil.wait_procs(procs, timeout=_KILL_WAIT_S)
        except Exception:
            alive = procs
        # On Windows terminate() already is a hard kill; elsewhere it is a
        # polite request, so insist on anything still alive.
        for p in alive:
            try:
                p.kill()
            except Exception:
                pass
        if alive:
            try:
                psutil.wait_procs(alive, timeout=_KILL_WAIT_S)
            except Exception:
                pass

    # Fallback (and belt and braces): make sure the direct child is gone even
    # when psutil is missing or could not see it.
    try:
        proc.wait(timeout=_KILL_WAIT_S)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=_KILL_WAIT_S)
        except subprocess.TimeoutExpired:
            logger.warning(f"DLC child process {proc.pid} did not exit after being killed")


def _tie_child_to_this_process(proc: subprocess.Popen):
    """Make the operating system kill the pose child if THIS process dies.

    WHY: before the pose ran in a child it ran inside the watcher, so it died
    with the watcher. A child started with no console window does not: if the
    watcher's window is closed, its scheduled task is ended, napari is closed
    under the panel's watcher, or the watcher crashes, none of the code that
    stops the child runs (no poll loop, no ``finally``). The pose then keeps the
    GPU for up to fourteen minutes with nothing left to stop it when somebody
    starts recording -- the one thing this module exists to prevent -- and the
    next watcher start would queue the same video again beside it.

    On Windows a Job Object created with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE is
    the operating system's own answer: when the last handle to the job closes
    (which happens automatically when this process exits, for ANY reason), every
    process in the job is killed, including anything the child started after
    joining it. The handle is not inheritable, so the child cannot keep its own
    job alive.

    Returns the job handle (close it with _close_job once the child is gone),
    or None where there is no such mechanism or it could not be set up -- then
    kill_orphaned_workers at the next watcher start is the fallback.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class _BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _BasicLimits),
                        ("IoInfo", _IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info), ctypes.sizeof(info))
        if ok:
            ok = kernel32.AssignProcessToJobObject(job, int(proc._handle))
        if not ok:
            err = ctypes.get_last_error()
            kernel32.CloseHandle(job)
            raise OSError(err, "could not put the pose process in a job")
        return job
    except Exception as e:
        logger.warning(_ascii(
            f"Could not tie the pose process to this watcher ({e}). If the watcher "
            f"dies part-way, the pose keeps running until the next watcher start "
            f"stops it."))
        return None


def _close_job(job) -> None:
    """Close a job handle from _tie_child_to_this_process. Closing it kills
    anything still inside -- wanted: nothing of a finished or stopped pose may
    keep running."""
    if not job:
        return
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle(job)
    except Exception as e:
        logger.debug(f"could not close the pose job handle: {e}")


def kill_orphaned_workers(worker_module: str = DEFAULT_WORKER_MODULE) -> int:
    """Kill pose worker processes whose watcher is gone. Returns how many.

    WHY: the Job Object (see _tie_child_to_this_process) is Windows-only and can
    fail to be set up. A worker whose watcher died keeps the GPU with nothing
    polling the recording check, and when the next watcher start puts its video
    back in the queue, a second pose of the same video would run beside it and
    write the same files. So a GPU watcher calls this at startup, BEFORE it
    requeues interrupted poses.

    Only ORPHANS are killed: a worker whose parent process no longer exists, or
    whose parent id now belongs to a newer process (Windows reuses ids). A
    worker with a live parent belongs to a running watcher and is left alone.
    Matched by the exact module name as a command-line argument, and only among
    Python processes, so nothing else can be hit. Never raises.
    """
    try:
        import psutil
    except ImportError:
        return 0
    killed = 0
    me = os.getpid()
    # Python processes only, and names first: reading command lines (or even
    # names, through psutil) for every process is very slow on a busy machine.
    # On Windows one system snapshot gives names and parents at once.
    pairs = None
    try:
        from mousereach.watcher.recording_guard import windows_process_table
        table = windows_process_table()
        if table:
            pairs = [(pid, ppid) for pid, ppid, exe in table
                     if "python" in (exe or "").lower() and pid != me]
    except Exception:
        pairs = None
    if pairs is None:
        try:
            pairs = [(p.pid, p.info.get("ppid"))
                     for p in psutil.process_iter(["name", "ppid"])
                     if "python" in (p.info.get("name") or "").lower() and p.pid != me]
        except Exception as e:
            logger.debug(f"orphaned pose worker check unavailable: {e}")
            return 0
    for pid, ppid in pairs:
        try:
            p = psutil.Process(pid)
            if worker_module not in (p.cmdline() or []):
                continue
            created = p.create_time()
            try:
                parent = psutil.Process(ppid)
                orphan = parent.create_time() > created + 1.0
            except psutil.NoSuchProcess:
                orphan = True
            if not orphan:
                continue
            procs = list(reversed(p.children(recursive=True))) + [p]
            for q in procs:
                try:
                    q.kill()
                except Exception:
                    pass
            try:
                psutil.wait_procs(procs, timeout=_KILL_WAIT_S)
            except Exception:
                pass
            killed += 1
            logger.warning(_ascii(
                f"Stopped a pose left running by a watcher that is no longer "
                f"running (process {p.pid}); its video is queued again."))
        except Exception:
            continue
    return killed


def _log_tail(log_path: Path) -> str:
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return ""
    return _ascii("\n".join(lines[-_LOG_TAIL_LINES:]))


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.debug(f"could not remove temporary file {path}: {e}")


def _ask(should_abort: Callable[[], Optional[str]]) -> Optional[str]:
    """Ask the caller whether to stop.

    WHY a failing question counts as "stop": the question guards a recording.
    If it cannot be answered, assuming nobody is recording is the unsafe guess.
    """
    try:
        reason = should_abort()
    except Exception as e:
        return _ascii(f"could not check whether to stop: {e}")
    return _ascii(reason) if reason else None


def run_dlc_single_interruptible(
    video_path,
    config_path,
    output_dir,
    gpu,
    shuffle,
    should_abort: Callable[[], Optional[str]],
    poll_seconds: float = 5.0,
    python_exe: Optional[str] = None,
    worker_module: str = DEFAULT_WORKER_MODULE,
    work_dir: Optional[Path] = None,
) -> dict:
    """Pose one video with DeepLabCut in a child process that can be stopped.

    Args:
        video_path: The video to pose.
        config_path: The trained DLC model's config.yaml.
        output_dir: Where DeepLabCut writes <stem>DLC_*.h5 / .csv / _meta.pickle.
        gpu: GPU device number (None for CPU), passed to run_dlc_batch.
        shuffle: Trained shuffle, passed to run_dlc_batch (None = resolved there).
        should_abort: Called before starting and then every poll_seconds while
            the pose runs. Returns None to carry on, or a short reason to stop
            (e.g. "recorder.exe is running"). If it raises, the pose is stopped
            -- see _ask for why.
        poll_seconds: How often to ask. WHY a few seconds: short enough that a
            recording started mid-pose gets the machine back almost at once,
            long enough that asking costs nothing next to the pose.
        python_exe: The Python that runs the child. Default: this one
            (sys.executable), so the child sees the same installed packages.
        worker_module: For tests only -- the module the child runs with -m.
        work_dir: For tests only -- where the temporary argument, result and
            log files go. Default: the system temporary folder.

    Returns:
        The worker's result dict: {'video', 'status': 'success', 'dlc_scorer'}
        or {'video', 'status': 'failed', 'error'}. When should_abort gave a
        reason: {'video', 'status': 'aborted', 'abort_reason'}, with this run's
        partial pose files removed. A child that exits without writing a result
        is reported {'status': 'failed', 'error'} quoting the end of its output.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir) if output_dir is not None else video_path.parent
    stem = video_path.stem

    # Recording already running? Do not even start the child: loading the
    # model alone takes the GPU for a while.
    reason = _ask(should_abort)
    if reason:
        logger.info(_ascii(f"Not posing {video_path.name}: {reason}"))
        return {"video": str(video_path), "status": "aborted", "abort_reason": reason}

    work_dir = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir())
    work_dir.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    args_path = work_dir / f"mousereach_dlc_{stem}_{token}.args.json"
    result_path = work_dir / f"mousereach_dlc_{stem}_{token}.result.json"
    log_path = work_dir / f"mousereach_dlc_{stem}_{token}.log"

    env = dict(os.environ)
    # WHY: the child's output goes to a file, and Python would otherwise encode
    # it with the Windows code page -- the first non-ASCII character printed by
    # a progress bar would crash the pose. Unbuffered so a crash leaves its
    # last lines in the log.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    creationflags = 0
    if sys.platform == "win32":
        # WHY: the watcher may run with no console at all; without this flag
        # every pose would flash a black window in front of the operator.
        creationflags = subprocess.CREATE_NO_WINDOW

    keep_log = False
    proc = None
    job = None
    log_file = None
    started_at = time.time()
    before: Set[str] = set()
    try:
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump({
                "video_path": str(video_path),
                "config_path": str(config_path),
                "output_dir": str(output_dir),
                "gpu": gpu,
                "shuffle": shuffle,
                "result_path": str(result_path),
            }, f)
        log_file = open(log_path, "w", encoding="utf-8", errors="replace")

        # Listing and start time taken immediately before the child starts, so
        # every file in the listing is older than anything the child can write.
        started_at = time.time()
        before = {p.name for p in _pose_outputs(output_dir, stem)} if output_dir.is_dir() else set()

        proc = subprocess.Popen(
            [python_exe or sys.executable, "-m", worker_module, str(args_path)],
            stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=env, creationflags=creationflags, close_fds=True)
        # At once, before the child has had time to start anything of its own
        # (loading Python takes far longer than this call).
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
            logger.info(_ascii(f"Stopping the pose of {video_path.name}: {abort_reason}"))
            _kill_process_tree(proc)
            log_file.close()
            removed = _remove_new_partial_outputs(output_dir, stem, before, started_at)
            if removed:
                logger.info(_ascii(
                    f"Removed {len(removed)} partial pose file(s) of {video_path.name}: "
                    + ", ".join(p.name for p in removed)))
            return {"video": str(video_path), "status": "aborted",
                    "abort_reason": abort_reason}

        # The child finished on its own (possibly in the same instant a stop
        # was asked for -- a finished pose is kept, not thrown away).
        log_file.close()
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            if not isinstance(result, dict) or "status" not in result:
                raise ValueError("result file has no status")
        except (OSError, ValueError) as e:
            keep_log = True
            tail = _log_tail(log_path)
            error = (f"DLC child process exited (code {proc.returncode}) without a "
                     f"result ({_ascii(e)}); its output is kept in {log_path}")
            if tail:
                error += "; last lines:\n" + tail
            return {"video": str(video_path), "status": "failed", "error": error}
        result.setdefault("video", str(video_path))
        if result.get("status") != "success":
            keep_log = True
            logger.warning(_ascii(
                f"Pose of {video_path.name} failed; DLC's output is kept in {log_path}"))
        return result
    finally:
        # Never leave a pose running behind us: if this process is being torn
        # down by an exception (or Ctrl+C), the child must go too, or it keeps
        # the GPU with nobody to collect its result.
        if proc is not None and proc.poll() is None:
            _kill_process_tree(proc)
            try:
                _remove_new_partial_outputs(output_dir, stem, before, started_at)
            except Exception:
                pass
        _close_job(job)
        if log_file is not None and not log_file.closed:
            log_file.close()
        _unlink_quietly(args_path)
        _unlink_quietly(result_path)
        # The worker writes its result under this name first; a kill landing
        # mid-write leaves it behind.
        _unlink_quietly(result_path.with_name(result_path.name + ".tmp"))
        # The child's output is kept only when something went wrong, so a
        # person can read why; successful and stopped runs leave nothing behind.
        if not keep_log:
            _unlink_quietly(log_path)
