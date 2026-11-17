"""mousereach-watch-status names the videos that are stuck, and why.

WHY: the status command printed counts by state only. A video whose filing
failed stays 'processed' while the watcher waits out a back-off kept in its
memory, so it looked like one about to be filed; a failed video's reason
was in its row but not on the screen; and the holds the watchers list for a
person (<NAS>/attention) were not shown at all. A person asking "what is
stuck?" had to open the database by hand.

Everything here is in tmp_path: a real watcher database, a fake home folder
and a fake share. No machine configuration is read.
"""
import json
import os
import pathlib
import sys
import time

import pytest

import mousereach.config as config
import mousereach.watcher.cli as cli
from mousereach.watcher.db import WatcherDB


@pytest.fixture
def machine(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".mousereach").mkdir(parents=True)
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    root = tmp_path / "processing"
    root.mkdir()
    monkeypatch.setattr(config, "require_processing_root", lambda: root)
    db_path = root / "watcher.db"
    db = WatcherDB(db_path)
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: db_path)
    share = tmp_path / "share"
    (share / "attention").mkdir(parents=True)
    monkeypatch.setattr(config.Paths, "NAS_ROOT", share, raising=False)
    return db, share


def _status(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["mousereach-watch-status", *args])
    try:
        cli.main_status()
    except SystemExit as e:
        assert not e.code
    out = capsys.readouterr().out
    out.encode("ascii")            # a Windows console cannot print anything else
    return out


def _video(db, video_id, state, **kw):
    db.register_video(video_id, source_path=str(pathlib.Path("x") / f"{video_id}.mp4"))
    db.force_state(video_id, state, reason="test", **kw)


def test_status_names_a_video_held_after_a_failed_filing(machine, monkeypatch, capsys):
    db, _ = machine
    _video(db, "20990101_TEST0101_P1", "processed")
    db.log_step("20990101_TEST0101_P1", "archive", "failed", message="share not reachable")
    # Filed later after a failure: its newest log row is not a failure.
    _video(db, "20990101_TEST0102_P1", "processed")
    db.log_step("20990101_TEST0102_P1", "archive", "failed", message="old error")
    db.log_step("20990101_TEST0102_P1", "pipeline", "completed")

    out = _status(monkeypatch, capsys, "--log", "0")   # recent activity off
    assert "Held after a failed filing attempt (1)" in out
    assert "20990101_TEST0101_P1: share not reachable" in out
    assert "old error" not in out


def test_status_lists_failed_videos_with_their_error(machine, monkeypatch, capsys):
    db, _ = machine
    _video(db, "20990101_TEST0103_P1", "failed", error_message="pose file unreadable")

    out = _status(monkeypatch, capsys)
    assert "Failed (1)" in out
    assert "20990101_TEST0103_P1: pose file unreadable" in out


def test_status_shows_the_attention_lists_with_their_age(machine, monkeypatch, capsys):
    _, share = machine
    lst = share / "attention" / "NODE-A_hand_scores.json"
    lst.write_text(json.dumps({"items": [
        {"kind": "hand_scores", "needs_person": True, "video_id": "20990101_TEST0104_P1",
         "says": "No hand scores found for 20990101_TEST0104_P1."}]}), encoding="utf-8")
    old = time.time() - 3 * 86400
    os.utime(lst, (old, old))      # a list nobody has rewritten for days

    out = _status(monkeypatch, capsys)
    line = [l for l in out.splitlines() if "TEST0104" in l]
    assert line and "[NEEDS A PERSON]" in line[0]
    assert "list written 3.0 days ago" in line[0]


def test_status_json_carries_the_same_lists(machine, monkeypatch, capsys):
    db, _ = machine
    _video(db, "20990101_TEST0105_P1", "failed", error_message="boom")
    data = json.loads(_status(monkeypatch, capsys, "--json"))
    assert [f["video_id"] for f in data["held_and_failed"]["failed"]] == ["20990101_TEST0105_P1"]


def test_the_dashboard_health_line_names_a_video_held_after_a_failed_filing(machine, monkeypatch):
    from mousereach.watcher import health
    db, _ = machine
    monkeypatch.setattr(health, "watcher_running", lambda: 1234)
    monkeypatch.setattr(health, "_current_pause_reason", lambda: None)
    _video(db, "20990101_TEST0106_P1", "processed")
    db.log_step("20990101_TEST0106_P1", "archive", "failed", message="share not reachable")
    text = "\n".join(health.health_report(db))
    assert "could not be filed" in text and "20990101_TEST0106_P1 (share not reachable)" in text
    assert "Nothing is stuck" not in text
