"""Every path that files a video asks the watcher's last checks on its features file.

WHY: filing means done for good. The watcher's archive step refuses a features
file that still holds a pellet nobody decided, or a reach whose extended
measures failed; the mousereach-archive command (on a machine with no watcher
row for the video) and bring-current's finalize filed such a file unasked.
Both now refuse it, with nothing moved. (Bring-current's side is tested in
test_bring_current_keeps_the_watchers_rules.py.) Video ids are invented.
"""
import json

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
    WatcherDB(db_path=db_path)          # a watcher database with no row for V

    class FakeCfg:
        pass
    FakeCfg.db_path = db_path
    monkeypatch.setattr(cfg.WatcherConfig, "load", staticmethod(lambda: FakeCfg()))
    monkeypatch.setattr(core.Paths, "PROCESSING", proc)
    monkeypatch.setattr(core.Paths, "NAS_ROOT", tmp_path / "nas", raising=False)
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
    return proc, dest


def _outputs(proc, features_doc):
    (proc / f"{V}.mp4").write_bytes(b"v")
    for suffix in ("_segments.json", "_reaches.json", "_pellet_outcomes.json"):
        (proc / f"{V}{suffix}").write_text(json.dumps({"validation_status": "auto_approved"}))
    (proc / f"{V}_features.json").write_text(json.dumps(features_doc))


def test_the_archive_command_does_not_file_an_undecided_pellet(env):
    proc, dest = env
    _outputs(proc, {"segments": [{"segment_num": 1, "outcome": "retrieved"},
                                 {"segment_num": 2, "outcome": "triaged"}]})
    r = core.archive_video(V, verbose=False)
    assert not r["success"], r
    assert "no decided outcome" in (r.get("error") or ""), r
    assert not any(dest.iterdir()), "a features file with an open pellet was filed"
    assert (proc / f"{V}_features.json").is_file()


def test_the_archive_command_does_not_file_failed_measures(env):
    proc, dest = env
    _outputs(proc, {"segments": [{"segment_num": 1, "outcome": "retrieved", "reaches": [
        {"reach_id": 4, "extended": {"_extended_features_error": "boom"}}]}]})
    r = core.archive_video(V, verbose=False)
    assert not r["success"], r
    assert "_extended_features_error" in (r.get("error") or ""), r
    assert not any(dest.iterdir()), "a features file with failed measures was filed"


def test_the_archive_command_still_files_a_decided_video(env):
    proc, dest = env
    _outputs(proc, {"segments": [{"segment_num": 1, "outcome": "retrieved", "reaches": [
        {"reach_id": 4, "extended": {"peak_mm": 1.0}}]}]})
    r = core.archive_video(V, verbose=False)
    assert r["success"], r
    assert (dest / f"{V}_features.json").is_file()
