"""Publication locks are set and lifted only where videos are filed.

WHY: a lock ('crystallized') is written to the watcher database of the machine
that runs the command, and only that machine's version scan reads it. The
processing server files and re-runs videos, so a lock set on a pose-only GPU
PC protected nothing while looking done. mousereach-crystallize (not --list),
mousereach-uncrystallize and the dashboard's lock buttons now refuse unless the
machine is the processing server; a GPU PC with also_process on (a lab with no
processing server) may lock, with a caveat.

No machine configuration is read: the watcher configuration is faked.
"""
import sys
import types

import pytest

import mousereach.config as config
from mousereach.watcher import cli
from mousereach.watcher.db import WatcherDB


def _role(monkeypatch, mode, also_process=False):
    cfg = types.SimpleNamespace(mode=mode, also_process=also_process, db_path=None)
    monkeypatch.setattr(config.WatcherConfig, "load", staticmethod(lambda: cfg))


@pytest.fixture
def filed(tmp_path, monkeypatch):
    path = tmp_path / "watcher.db"
    db = WatcherDB(path)
    vid = "20240101_ABC9903_P1"
    db.register_video(vid, source_path="x")
    db.force_state(vid, "archived", reason="test")
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: path)
    return db, vid


def _run(monkeypatch, func, *argv):
    monkeypatch.setattr(sys, "argv", list(argv))
    try:
        func()
    except SystemExit as e:
        return e.code
    return 0


def test_a_pose_only_gpu_pc_refuses_to_lock(filed, monkeypatch, capsys):
    db, vid = filed
    _role(monkeypatch, "dlc_pc")
    code = _run(monkeypatch, cli.main_crystallize,
                "mousereach-crystallize", "--videos", vid, "--label", "LABEL_A")
    assert code == 1
    assert "processing server" in capsys.readouterr().err
    assert db.get_video(vid)["state"] == "archived"


def test_a_pose_only_gpu_pc_refuses_to_unlock(filed, monkeypatch, capsys):
    db, vid = filed
    db.force_state(vid, "crystallized", reason="test", crystallized_label="LABEL_A")
    _role(monkeypatch, "dlc_pc")
    code = _run(monkeypatch, cli.main_uncrystallize,
                "mousereach-uncrystallize", "--label", "LABEL_A")
    assert code == 1
    assert db.get_video(vid)["state"] == "crystallized"


def test_the_processing_server_locks(filed, monkeypatch):
    db, vid = filed
    _role(monkeypatch, "processing_server")
    _run(monkeypatch, cli.main_crystallize,
         "mousereach-crystallize", "--videos", vid, "--label", "LABEL_A")
    assert db.get_video(vid)["state"] == "crystallized"


def test_an_analysing_gpu_pc_locks_with_a_caveat(filed, monkeypatch, capsys):
    db, vid = filed
    _role(monkeypatch, "dlc_pc", also_process=True)
    _run(monkeypatch, cli.main_crystallize,
         "mousereach-crystallize", "--videos", vid, "--label", "LABEL_A")
    assert db.get_video(vid)["state"] == "crystallized"
    assert "known to this machine only" in capsys.readouterr().out


def test_list_works_anywhere(filed, monkeypatch, capsys):
    _role(monkeypatch, "dlc_pc")
    assert _run(monkeypatch, cli.main_crystallize, "mousereach-crystallize", "--list") in (0, None)
