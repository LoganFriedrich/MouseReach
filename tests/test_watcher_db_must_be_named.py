"""A login that cannot NAME its watcher database refuses; it never falls back
to <processing folder>/watcher.db.

WHY: every Windows login on a machine has its own settings file, but they all
share the processing folder. A login whose settings could not be read, or whose
settings say "processing server" without naming the watcher database
(watcher.db_path), used to be handed <processing folder>/watcher.db -- on a
server whose watcher writes a different file, an old decoy that still exists.
Publication locks, re-run marks and filing records then went into a file the
watcher never reads, without a word.

Everything lives in tmp_path: the settings file is a file there, and every
database is created by the test.
"""
import json
import sys
from pathlib import Path

import pytest

from mousereach import config as mr_config
from mousereach.watcher import cli
from mousereach.watcher import db_location
from mousereach.watcher.db import WatcherDB

VID = "20990101_ZZZ9901_P1"


@pytest.fixture
def login(tmp_path, monkeypatch):
    """A login whose settings file and processing folder are in tmp_path.
    Returns a function that writes the settings file (dict or raw text)."""
    root = tmp_path / "work"
    root.mkdir()
    settings = tmp_path / "home" / ".mousereach" / "config.json"
    settings.parent.mkdir(parents=True)
    monkeypatch.setattr(mr_config, "config_file_path", lambda: settings)
    monkeypatch.setattr(mr_config, "PROCESSING_ROOT", root)
    monkeypatch.setattr(mr_config.Paths, "PROCESSING_ROOT", root, raising=False)

    def write(content):
        if isinstance(content, dict):
            content = json.dumps(content)
        settings.write_text(content, encoding="utf-8")
        return settings
    write.root = root
    return write


def _decoy(root: Path) -> Path:
    """An old database at the default place, with one filed video in it."""
    path = root / "watcher.db"
    db = WatcherDB(path)
    db.register_video(video_id=VID, source_path=str(root / f"{VID}.mp4"))
    db.force_state(VID, "archived", reason="test: filed")
    return path


def _state(path: Path) -> str:
    return WatcherDB(path).get_video(VID)["state"]


def test_processing_server_settings_without_db_path_refuse(login):
    login({"processing_root": str(login.root),
           "watcher": {"mode": "processing_server"}})
    _decoy(login.root)
    with pytest.raises(Exception) as err:
        path = db_location.resolve_watcher_db_path()
        pytest.fail(f"resolved to {path} instead of refusing")
    assert "db_path" in str(err.value)
    assert db_location.watcher_db_unnamed_reason()


def test_unreadable_settings_refuse(login):
    login("{ this is not json")
    _decoy(login.root)
    with pytest.raises(Exception) as err:
        path = db_location.resolve_watcher_db_path()
        pytest.fail(f"resolved to {path} instead of refusing")
    assert "could not" in str(err.value) or "line" in str(err.value)
    db, why = db_location.open_watcher_db_if_present()
    assert db is None and why


def test_named_database_is_used_and_gpu_pc_default_is_kept(login, tmp_path):
    live = tmp_path / "live" / "watcher_live.db"
    login({"processing_root": str(login.root),
           "watcher": {"mode": "processing_server", "db_path": str(live)}})
    assert db_location.resolve_watcher_db_path() == live
    login({"processing_root": str(login.root), "watcher": {"mode": "dlc_pc"}})
    assert db_location.resolve_watcher_db_path() == login.root / "watcher.db"
    assert db_location.watcher_db_unnamed_reason() is None


def test_crystallize_from_an_unnamed_login_writes_nothing(login, monkeypatch, capsys):
    login({"processing_root": str(login.root),
           "watcher": {"mode": "processing_server"}})
    decoy = _decoy(login.root)
    monkeypatch.setattr(sys, "argv", ["mousereach-crystallize", "--videos", VID,
                                      "--label", "LABEL_A"])
    with pytest.raises(SystemExit) as ex:
        cli.main_crystallize()
    assert ex.value.code == 1
    assert _state(decoy) == "archived", "the lock went into a database the watcher never reads"
    assert "db_path" in capsys.readouterr().err


def test_archive_command_refuses_when_the_database_cannot_be_named(login, tmp_path, monkeypatch):
    from mousereach.archive import core
    login({"processing_root": str(login.root),
           "watcher": {"mode": "processing_server"}})
    processing = tmp_path / "work" / "Processing"
    processing.mkdir()
    (processing / f"{VID}_features.json").write_text("{}", encoding="utf-8")
    why = core.command_filing_refusal(VID, source_dir=processing)
    assert why and "db_path" in why, (
        "an unnamed watcher database was read as 'no watcher on this machine' "
        "and the row check was skipped")


def test_a_login_with_no_settings_file_refuses(login, monkeypatch):
    """No settings file: the processing folder comes from the machine-wide
    variable and nothing says which database the watcher uses -- the default
    file there is a guess, so it is refused."""
    # login() never called: this login has no settings file.
    _decoy(login.root)
    with pytest.raises(Exception) as err:
        path = db_location.resolve_watcher_db_path()
        pytest.fail(f"resolved to {path} instead of refusing")
    assert "settings" in str(err.value)
    db, why = db_location.open_watcher_db_if_present()
    assert db is None and why
