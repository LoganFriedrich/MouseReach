"""mousereach-watch-quarantine --release moves a file back, under a valid name.

WHY: the option imported a class that does not exist, so it failed every
time (the Quarantine tab was the only way out of quarantine), and it had no
way to say which intake the file belongs in. It now uses the tab's routine:
the name is checked first, nothing is overwritten, a refusal exits 1.

tmp_path only: the watcher database, config and folders are stand-ins.
"""
import json
from types import SimpleNamespace

import pytest

from mousereach.watcher import cli
from mousereach.watcher.db import WatcherDB

GOOD = "20240102_ABC9901_P1.mp4"


@pytest.fixture
def env(tmp_path, monkeypatch):
    import mousereach.config as mc
    quarantine = tmp_path / "Quarantine"
    single = tmp_path / "Single_Animal"
    multi = tmp_path / "Multi-Animal"
    quarantine.mkdir()
    db_path = tmp_path / "watcher.db"
    WatcherDB(db_path=db_path)
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: db_path)
    monkeypatch.setattr(mc.WatcherConfig, "load",
                        classmethod(lambda cls, *a, **k: SimpleNamespace(
                            get_quarantine_dir=lambda: quarantine)))
    monkeypatch.setattr(mc.Paths, "SINGLE_ANIMAL_OUTPUT", single, raising=False)
    monkeypatch.setattr(mc.Paths, "MULTI_ANIMAL_SOURCE", multi, raising=False)
    (quarantine / "garbled.mp4").write_bytes(b"video")
    (quarantine / "garbled.mp4.quarantine.json").write_text(json.dumps({"error_message": "x"}))
    return SimpleNamespace(quarantine=quarantine, single=single, multi=multi)


def _run(monkeypatch, *args):
    monkeypatch.setattr(cli.sys, "argv", ["mousereach-watch-quarantine", *args])
    try:
        cli.main_quarantine()
    except SystemExit as e:
        return e.code or 0
    return 0


def test_release_under_a_valid_new_name(env, monkeypatch):
    code = _run(monkeypatch, "--release", "garbled.mp4", "--to", "single", "--as", GOOD)
    assert (env.single / GOOD).is_file(), "the file was not released"
    assert (env.single / GOOD).read_bytes() == b"video"
    assert code == 0
    assert not (env.quarantine / "garbled.mp4").exists()
    assert not (env.quarantine / "garbled.mp4.quarantine.json").exists()


def test_release_refuses_an_invalid_name_and_moves_nothing(env, monkeypatch, capsys):
    code = _run(monkeypatch, "--release", "garbled.mp4", "--to", "single")
    assert code == 1
    assert (env.quarantine / "garbled.mp4").exists()
    assert (env.quarantine / "garbled.mp4.quarantine.json").exists()
    assert "Nothing was moved" in capsys.readouterr().err


def test_release_needs_a_destination(env, monkeypatch):
    code = _run(monkeypatch, "--release", "garbled.mp4", "--as", GOOD)
    assert code == 1
    assert (env.quarantine / "garbled.mp4").exists()


def test_release_never_overwrites_a_file_in_the_intake(env, monkeypatch):
    env.single.mkdir()
    (env.single / GOOD).write_bytes(b"other")
    code = _run(monkeypatch, "--release", "garbled.mp4", "--to", "single", "--as", GOOD)
    assert code == 1
    assert (env.single / GOOD).read_bytes() == b"other"
    assert (env.quarantine / "garbled.mp4").exists()
