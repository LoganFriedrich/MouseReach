"""A crop can be stopped part-way when a recording starts
(video_prep/core/crop_interruptible.py).

WHY: cropping a collage runs ffmpeg once per animal for minutes inside the
watcher, so an operator who opened the recording program had to wait for it.
Re-cropping is cheap; a recording that drops frames cannot be filmed again. So
the crop runs in a child process the parent can kill, and what it half-wrote is
thrown away. These tests pin what makes that safe:
  * a normal run returns exactly what the worker reported,
  * a stop kills the child AND anything it started,
  * a stop removes only THIS run's crops -- older files and the collage itself
    survive,
  * a child that dies without a result is a failure, and its output is kept,
  * a stop question that cannot be answered stops the crop (fail safe).

The child is a small fake worker module (no ffmpeg) put on PYTHONPATH.
"""
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from mousereach.video_prep.core.crop_interruptible import run_crop_collage_interruptible

COLLAGE = "20240101_ABC0101-ABC0102_P1"
REASON = "recorder.exe is running"


def _fake_worker(tmp_path, monkeypatch, name, body):
    mods = tmp_path / "fake_modules"
    mods.mkdir(exist_ok=True)
    source = textwrap.dedent('''
        import json, os, sys, time, subprocess
        from pathlib import Path
        args = json.load(open(sys.argv[1], encoding="utf-8"))
        out = Path(args["output_dir"])
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
    out = tmp_path / "watcher_working"
    out.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    collage = out / f"{COLLAGE}.mkv"
    collage.write_bytes(b"not really a collage")
    return collage, out, work


def _run(collage, out, work, module, should_abort=lambda: None):
    return run_crop_collage_interruptible(
        collage, out, should_abort=should_abort, poll_seconds=0.1,
        python_exe=sys.executable, worker_module=module, work_dir=work)


def test_success_returns_the_worker_result(tmp_path, monkeypatch, dirs):
    collage, out, work = dirs
    module = _fake_worker(tmp_path, monkeypatch, "fake_crop_ok", '''
        (out / "20240101_ABC0101_P1.mp4").write_bytes(b"single")
        write_result({"collage": args["input_path"], "status": "success",
                      "results": [{"position": 1, "status": "success",
                                   "output_path": str(out / "20240101_ABC0101_P1.mp4")}]})
    ''')

    result = _run(collage, out, work, module)

    assert result["status"] == "success"
    assert result["results"][0]["position"] == 1
    assert (out / "20240101_ABC0101_P1.mp4").is_file(), "a finished crop must be kept"


def test_stop_kills_the_child_and_removes_only_this_runs_crops(tmp_path, monkeypatch, dirs):
    collage, out, work = dirs
    old = out / "20231231_ABC0101_P1.mp4"          # an earlier crop, another day
    old.write_bytes(b"older single")
    module = _fake_worker(tmp_path, monkeypatch, "fake_crop_slow", '''
        (out / "20240101_ABC0101_P1.mp4").write_bytes(b"half written")
        (out / "crop_manifest.json").write_text("{}", encoding="utf-8")
        time.sleep(30)
        write_result({"collage": args["input_path"], "status": "success", "results": []})
    ''')
    asked = {"n": 0}

    def should_abort():
        asked["n"] += 1
        return REASON if asked["n"] > 1 else None

    result = _run(collage, out, work, module, should_abort)

    assert result == {"collage": str(collage), "status": "aborted", "abort_reason": REASON}
    assert not (out / "20240101_ABC0101_P1.mp4").exists(), "this run's partial crop goes"
    assert not (out / "crop_manifest.json").exists()
    assert old.read_bytes() == b"older single", "an earlier crop must survive"
    assert collage.is_file(), "the collage itself is never deleted here"


def test_a_recording_already_running_means_no_child_at_all(tmp_path, monkeypatch, dirs):
    collage, out, work = dirs
    module = _fake_worker(tmp_path, monkeypatch, "fake_crop_never", '''
        (out / "should_not_exist.mp4").write_bytes(b"x")
        write_result({"collage": args["input_path"], "status": "success", "results": []})
    ''')

    result = _run(collage, out, work, module, lambda: REASON)

    assert result["status"] == "aborted" and result["abort_reason"] == REASON
    assert not (out / "should_not_exist.mp4").exists()


def test_a_question_that_raises_stops_the_crop(tmp_path, monkeypatch, dirs):
    collage, out, work = dirs
    module = _fake_worker(tmp_path, monkeypatch, "fake_crop_slow2", '''
        time.sleep(30)
    ''')

    def boom():
        raise RuntimeError("process list unreadable")

    result = _run(collage, out, work, module, boom)

    assert result["status"] == "aborted"
    assert "process list unreadable" in result["abort_reason"]


def test_child_without_a_result_is_a_failure_and_its_output_is_kept(tmp_path, monkeypatch, dirs):
    collage, out, work = dirs
    module = _fake_worker(tmp_path, monkeypatch, "fake_crop_dies", '''
        print("ffmpeg exploded")
        sys.exit(9)
    ''')

    result = _run(collage, out, work, module)

    assert result["status"] == "failed"
    assert "ffmpeg exploded" in result["error"]
    assert list(work.glob("*.log")), "the child's output is kept for a person to read"
