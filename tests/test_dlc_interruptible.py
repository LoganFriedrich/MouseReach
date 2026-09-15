"""A pose can be stopped part-way when a recording starts (dlc/core/interruptible.py).

WHY: deeplabcut.analyze_videos runs in-process and cannot be interrupted, so a
pose started just before somebody begins recording would take the GPU, CPU and
disk for about fourteen more minutes. Recording must always win, so the pose
runs in a child process the parent can kill. These tests pin what makes that
safe:
  * a normal run returns exactly what the worker reported,
  * a stop kills the child AND anything it started (a surviving helper would
    keep the GPU with nobody watching it),
  * a stop removes only the half-written pose files of THIS run -- an older,
    finished pose of the same video and other videos' poses survive,
  * a child that dies without a result is a failure, not a success, and its
    output is kept so a person can read why,
  * a stop question that cannot be answered stops the pose (fail safe).

The child here is a small fake worker module (no DeepLabCut) put on PYTHONPATH,
so the tests are fast and need no GPU.
"""
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from mousereach.dlc.core import interruptible
from mousereach.dlc.core.interruptible import run_dlc_single_interruptible

STEM = "20240101_ABC0101_P1"
REASON = "recorder.exe is running"


def _fake_worker(tmp_path, monkeypatch, name, body):
    """Write a fake worker module and put its folder on the child's PYTHONPATH."""
    mods = tmp_path / "fake_modules"
    mods.mkdir(exist_ok=True)
    source = textwrap.dedent('''
        import json, os, sys, time, subprocess
        from pathlib import Path
        args = json.load(open(sys.argv[1], encoding="utf-8"))
        out = Path(args["output_dir"])
        stem = Path(args["video_path"]).stem
        def write_result(d):
            p = Path(args["result_path"])
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(json.dumps(d), encoding="utf-8")
            os.replace(tmp, p)
    ''') + textwrap.dedent(body)
    (mods / f"{name}.py").write_text(source, encoding="utf-8")
    old = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", str(mods) + (os.pathsep + old if old else ""))
    return name


@pytest.fixture
def dirs(tmp_path):
    out = tmp_path / "DLC_Queue"
    out.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    video = out / f"{STEM}.mp4"
    video.write_bytes(b"not really a video")
    return video, out, work


def _run(video, out, work, module, should_abort=lambda: None):
    return run_dlc_single_interruptible(
        video, out / "config.yaml", out, gpu=0, shuffle=1,
        should_abort=should_abort, poll_seconds=0.1,
        python_exe=sys.executable, worker_module=module, work_dir=work)


def test_success_returns_the_worker_result_and_cleans_temp_files(tmp_path, monkeypatch, dirs):
    video, out, work = dirs
    module = _fake_worker(tmp_path, monkeypatch, "fake_worker_ok", '''
        (out / (stem + "DLC_x.h5")).write_bytes(b"pose")
        write_result({"video": args["video_path"], "status": "success",
                      "dlc_scorer": "DLC_x", "gpu": args["gpu"],
                      "shuffle": args["shuffle"]})
    ''')

    result = _run(video, out, work, module)

    assert result == {"video": str(video), "status": "success", "dlc_scorer": "DLC_x",
                      "gpu": 0, "shuffle": 1}
    assert (out / f"{STEM}DLC_x.h5").is_file(), "a finished pose must be kept"
    assert list(work.iterdir()) == [], "args, result and log files must be removed"


def test_stop_kills_the_tree_and_removes_only_this_runs_partial_files(tmp_path, monkeypatch, dirs):
    psutil = pytest.importorskip("psutil")
    video, out, work = dirs
    old_h5 = out / f"{STEM}DLC_old.h5"
    old_meta = out / f"{STEM}DLC_old_meta.pickle"
    for p in (old_h5, old_meta):
        p.write_bytes(b"finished earlier")
        os.utime(p, (time.time() - 3600, time.time() - 3600))
    other = out / "20240102_ABC0102_P1DLC_x.h5"
    other.write_bytes(b"another video's pose")
    not_pose = out / f"{STEM}DLC_notes.txt"
    module = _fake_worker(tmp_path, monkeypatch, "fake_worker_slow", '''
        (out / (stem + "DLC_x.h5")).write_bytes(b"half a pose")
        (out / (stem + "DLC_x_meta.pickle")).write_bytes(b"half")
        (out / (stem + "DLC_notes.txt")).write_bytes(b"not a pose file")
        helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        tmp = out / "pids.tmp"
        tmp.write_text(json.dumps([os.getpid(), helper.pid]), encoding="utf-8")
        os.replace(tmp, out / "pids.json")
        time.sleep(60)
        write_result({"status": "success"})
    ''')
    pids_file = out / "pids.json"

    def should_abort():
        return REASON if pids_file.exists() else None

    t0 = time.monotonic()
    result = _run(video, out, work, module, should_abort)
    elapsed = time.monotonic() - t0

    assert result == {"video": str(video), "status": "aborted", "abort_reason": REASON}
    assert elapsed < 15, "the stop must not wait for the pose to finish"
    for pid in json.loads(pids_file.read_text(encoding="utf-8")):
        if psutil.pid_exists(pid):
            try:
                status = psutil.Process(pid).status()
            except psutil.NoSuchProcess:
                continue
            assert status == psutil.STATUS_ZOMBIE, f"process {pid} survived the stop"
    assert not (out / f"{STEM}DLC_x.h5").exists(), "partial pose must be removed"
    assert not (out / f"{STEM}DLC_x_meta.pickle").exists()
    assert old_h5.read_bytes() == b"finished earlier", "an earlier pose must survive"
    assert old_meta.is_file()
    assert other.is_file(), "another video's pose must survive"
    assert not_pose.is_file(), "only pose-file endings are ever removed"
    assert video.is_file()
    assert list(work.iterdir()) == []


def test_child_without_a_result_is_a_failure_and_keeps_its_output(tmp_path, monkeypatch, dirs):
    video, out, work = dirs
    module = _fake_worker(tmp_path, monkeypatch, "fake_worker_dies", '''
        print("boom: out of GPU memory")
        sys.exit(3)
    ''')

    result = _run(video, out, work, module)

    assert result["status"] == "failed"
    assert result["video"] == str(video)
    assert "code 3" in result["error"]
    assert "boom: out of GPU memory" in result["error"]
    left = sorted(p.name for p in work.iterdir())
    assert len(left) == 1 and left[0].endswith(".log"), left


def test_already_blocked_never_starts_the_child(tmp_path, monkeypatch, dirs):
    # Starting the child is made impossible, not just unobservable: a child
    # killed before Python finished loading would leave no trace of having
    # started, so a "no marker file" check could not catch a lost pre-check.
    video, out, work = dirs
    asked = []

    def should_abort():
        asked.append(1)
        return REASON

    def no_child(*a, **k):
        raise AssertionError("the child must not start while a recording runs")

    monkeypatch.setattr(interruptible.subprocess, "Popen", no_child)
    result = run_dlc_single_interruptible(
        video, out / "config.yaml", out, gpu=0, shuffle=1,
        should_abort=should_abort, poll_seconds=5, work_dir=work)

    assert result == {"video": str(video), "status": "aborted", "abort_reason": REASON}
    assert asked == [1], "asked exactly once, before anything starts"
    assert list(work.iterdir()) == []


def _alive(psutil, pid):
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are Windows-only")
def test_the_pose_dies_with_a_watcher_that_is_killed(tmp_path, monkeypatch, dirs):
    """A hard kill runs no finally and no poll loop. Before the pose ran in a
    child it died with the watcher; it must still, or it keeps the GPU with
    nothing left to stop it when recording starts."""
    psutil = pytest.importorskip("psutil")
    video, out, work = dirs
    module = _fake_worker(tmp_path, monkeypatch, "fake_worker_sleeps", '''
        tmp = out / "pid.tmp"
        tmp.write_text(str(os.getpid()), encoding="utf-8")
        os.replace(tmp, out / "pid.txt")
        time.sleep(120)
        write_result({"status": "success"})
    ''')
    parent_code = textwrap.dedent(f'''
        import sys
        from mousereach.dlc.core.interruptible import run_dlc_single_interruptible
        run_dlc_single_interruptible(
            {str(video)!r}, {str(out / "config.yaml")!r}, {str(out)!r}, gpu=0, shuffle=1,
            should_abort=lambda: None, poll_seconds=0.2, python_exe=sys.executable,
            worker_module={module!r}, work_dir={str(work)!r})
    ''')
    parent = subprocess.Popen([sys.executable, "-c", parent_code])
    child = None
    try:
        pid_file = out / "pid.txt"
        deadline = time.monotonic() + 60
        while not pid_file.exists():
            assert parent.poll() is None, "the stand-in watcher exited early"
            assert time.monotonic() < deadline, "the pose child never started"
            time.sleep(0.1)
        child = int(pid_file.read_text(encoding="utf-8"))
        assert _alive(psutil, child)

        parent.kill()
        parent.wait(timeout=30)

        deadline = time.monotonic() + 20
        while _alive(psutil, child) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not _alive(psutil, child), "the pose outlived its killed watcher"
    finally:
        if parent.poll() is None:
            parent.kill()
        if child is not None:
            try:
                psutil.Process(child).kill()
            except Exception:
                pass


def test_orphaned_workers_are_killed_and_a_live_watchers_worker_is_not(tmp_path, monkeypatch):
    psutil = pytest.importorskip("psutil")
    mods = tmp_path / "fake_modules"
    mods.mkdir()
    (mods / "fake_orphan_worker.py").write_text("import time\ntime.sleep(120)\n",
                                                encoding="utf-8")
    old = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", str(mods) + (os.pathsep + old if old else ""))

    # Its watcher (this test process) is alive: it must be left alone.
    kept = subprocess.Popen([sys.executable, "-m", "fake_orphan_worker"])
    # Its "watcher" exits at once, leaving it an orphan.
    # The orphan gets null standard handles: holding the launcher's captured
    # output pipe open would make this call wait for it to finish sleeping.
    launcher = subprocess.run(
        [sys.executable, "-c",
         "import subprocess, sys; "
         "d = subprocess.DEVNULL; "
         "p = subprocess.Popen([sys.executable, '-m', 'fake_orphan_worker'], "
         "stdin=d, stdout=d, stderr=d); "
         "print(p.pid)"],
        capture_output=True, text=True, timeout=60)
    orphan = int(launcher.stdout.strip())
    try:
        assert _alive(psutil, orphan)
        killed = interruptible.kill_orphaned_workers(worker_module="fake_orphan_worker")
        assert killed == 1
        deadline = time.monotonic() + 20
        while _alive(psutil, orphan) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not _alive(psutil, orphan)
        assert kept.poll() is None, "a worker whose watcher is alive must survive"
    finally:
        for pid in (orphan, kept.pid):
            try:
                psutil.Process(pid).kill()
            except Exception:
                pass


def test_a_stop_question_that_raises_stops_the_pose(tmp_path, monkeypatch, dirs):
    video, out, work = dirs
    module = _fake_worker(tmp_path, monkeypatch, "fake_worker_never", '''
        write_result({"status": "success"})
    ''')

    def broken():
        raise RuntimeError("process list unavailable")

    result = _run(video, out, work, module, should_abort=broken)

    assert result["status"] == "aborted"
    assert "process list unavailable" in result["abort_reason"]


# ------------------------------------------------------- the real worker module

def _args_file(tmp_path):
    result_path = tmp_path / "r.json"
    args = tmp_path / "a.json"
    args.write_text(json.dumps({
        "video_path": str(tmp_path / f"{STEM}.mp4"),
        "config_path": str(tmp_path / "config.yaml"),
        "output_dir": str(tmp_path),
        "gpu": None,
        "shuffle": 2,
        "result_path": str(result_path),
    }), encoding="utf-8")
    return args, result_path


def test_worker_writes_the_single_video_result(tmp_path, monkeypatch):
    import mousereach.dlc.core.batch as batch
    from mousereach.dlc.core import interruptible_worker

    seen = {}

    def fake_batch(video_paths, config_path, output_dir=None, gpu=0,
                   save_as_csv=True, shuffle=None):
        seen.update(videos=video_paths, gpu=gpu, shuffle=shuffle, out=output_dir)
        return [{"video": str(video_paths[0]), "status": "success", "dlc_scorer": "DLC_y"}]

    monkeypatch.setattr(batch, "run_dlc_batch", fake_batch)
    args, result_path = _args_file(tmp_path)

    assert interruptible_worker.main([str(args)]) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "success" and result["dlc_scorer"] == "DLC_y"
    assert seen["videos"] == [tmp_path / f"{STEM}.mp4"]
    assert seen["gpu"] is None and seen["shuffle"] == 2 and seen["out"] == tmp_path
    assert not list(tmp_path.glob("*.tmp")), "no half-written result left behind"


def test_worker_reports_an_exception_as_failed(tmp_path, monkeypatch):
    import mousereach.dlc.core.batch as batch
    from mousereach.dlc.core import interruptible_worker

    def exploding(*a, **k):
        raise ImportError("DeepLabCut not installed")

    monkeypatch.setattr(batch, "run_dlc_batch", exploding)
    args, result_path = _args_file(tmp_path)

    assert interruptible_worker.main([str(args)]) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert "DeepLabCut not installed" in result["error"]
    assert result["video"].endswith(f"{STEM}.mp4")


def test_parent_module_does_not_import_deeplabcut():
    # WHY: the parent is the long-lived watcher; loading DeepLabCut there
    # would hold its memory and GPU context for good and defeat the kill.
    import ast
    tree = ast.parse(Path(interruptible.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert not names & {"deeplabcut", "tensorflow", "torch"}
