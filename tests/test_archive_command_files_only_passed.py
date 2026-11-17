"""The mousereach-archive command files only what the watcher has passed.

WHY: the command used to file on the validation_status in the three stage
files alone. The quality triage writes 'auto_approved' there BEFORE the
hand-score check, the review gate and kinematics run, so a video held for its
hand scores (row 'processing'), or one with no kinematics, was filed -- and
filed means done for good. It also ran beside the watcher's own filing, and
when the row was not 'processed' it filed the files and recorded nothing.
"""
import json
import os
import sys
import time

import pytest

import mousereach.archive.core as core

V = "20990101_ZZZ0101_P3"


@pytest.fixture
def env(tmp_path, monkeypatch):
    from mousereach.watcher.db import WatcherDB
    import mousereach.config as cfg

    proc, dest = tmp_path / "Processing", tmp_path / "Analyzed" / "ZZZ01"
    proc.mkdir(parents=True)
    dest.mkdir(parents=True)
    db_path = tmp_path / "watcher.db"
    db = WatcherDB(db_path=db_path)

    class FakeCfg:
        pass
    FakeCfg.db_path = db_path
    monkeypatch.setattr(cfg.WatcherConfig, "load", staticmethod(lambda: FakeCfg()))
    monkeypatch.setattr(core.Paths, "PROCESSING", proc)
    monkeypatch.setattr(core, "get_archive_destination", lambda vid: dest)
    import mousereach.index as index_pkg

    class _NoIndex:
        def load(self): pass
        def remove_video(self, video_id): pass
        def save(self): pass
        def get_ready_to_archive(self): return [V]

    monkeypatch.setattr(index_pkg, "PipelineIndex", _NoIndex)
    import mousereach.watcher.health as health
    monkeypatch.setattr(health, "_watcher_processes", lambda first_only=False: [])
    return proc, dest, db


def _outputs(proc, features=True, features_age=0):
    (proc / f"{V}.mp4").write_bytes(b"v")
    for suffix in ("_segments.json", "_reaches.json", "_pellet_outcomes.json"):
        (proc / f"{V}{suffix}").write_text(json.dumps({"validation_status": "auto_approved"}))
    if features:
        f = proc / f"{V}_features.json"
        f.write_text("{}")
        t = time.time() - features_age
        os.utime(f, (t, t))


def _row(db, state):
    db.register_video(video_id=V, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete",
               "processing", "processed"):
        db.update_state(V, st)
        if st == state:
            return


def test_a_video_held_by_the_watcher_is_not_filed(env):
    proc, dest, db = env
    _outputs(proc)
    _row(db, "processing")             # e.g. held for its hand scores
    r = core.archive_video(V, verbose=False)
    assert not r["success"]
    assert "processing" in r["error"]
    assert not any(dest.iterdir()), "a held video was filed"


def test_a_video_without_kinematics_is_not_filed(env):
    proc, dest, db = env
    _outputs(proc, features=False)
    _row(db, "processed")
    r = core.archive_video(V, verbose=False)
    assert not r["success"]
    assert not any(dest.iterdir())


def test_an_earlier_runs_features_file_is_not_filed(env):
    proc, dest, db = env
    _outputs(proc, features_age=3600)   # copied down from the last filing
    _row(db, "processed")
    r = core.archive_video(V, verbose=False)
    assert not r["success"]
    assert "older" in r["error"]
    assert not any(dest.iterdir())


def test_a_passed_video_is_filed_and_recorded(env):
    proc, dest, db = env
    _row(db, "processed")
    _outputs(proc)
    r = core.archive_video(V, verbose=False)
    assert r["success"], r
    assert core.record_archived(V)
    assert db.get_video(V)["state"] == "archived"


def test_the_command_does_not_file_beside_a_running_watcher(env, monkeypatch):
    proc, dest, db = env
    _row(db, "processed")
    _outputs(proc)
    import mousereach.watcher.health as health
    monkeypatch.setattr(health, "_watcher_processes", lambda first_only=False: [object()])
    from mousereach.archive import cli
    monkeypatch.setattr(sys, "argv", ["mousereach-archive", V])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1
    assert not any(dest.iterdir()), "filed beside the watcher"


def test_the_bulk_command_skips_a_held_video(env, monkeypatch):
    proc, dest, db = env
    _outputs(proc)
    _row(db, "processing")
    from mousereach.archive import cli
    monkeypatch.setattr(sys, "argv", ["mousereach-archive"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1
    assert not any(dest.iterdir())
