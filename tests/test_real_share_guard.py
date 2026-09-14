"""The suite-wide guard in conftest.py notices a test touching real pipeline
folders -- including a folder create that changes nothing on disk.

The guard is pointed at a temp folder standing in for a real root, so these
tests never touch the real share. Each test removes the records it made, or the
guard would (correctly) fail the test itself.
"""
import conftest


def _record(monkeypatch, root, event, args):
    monkeypatch.setattr(conftest, "_REAL_ROOTS", (conftest._norm(root),))
    before = len(conftest._hits)
    conftest._audit(event, args)
    new = conftest._hits[before:]
    del conftest._hits[before:]
    return new


def test_a_folder_create_under_a_real_root_is_recorded(tmp_path, monkeypatch):
    root = tmp_path / "real_share"
    root.mkdir()
    new = _record(monkeypatch, root, "os.mkdir", (str(root / "Processing" / "Posed"), 0o777, None))
    assert [event for event, _ in new] == ["os.mkdir"]


def test_a_write_open_is_recorded_and_a_read_open_is_not(tmp_path, monkeypatch):
    root = tmp_path / "real_share"
    root.mkdir()
    target = str(root / "pipeline_versions.json")
    assert _record(monkeypatch, root, "open", (target, "w", 0)) != []
    assert _record(monkeypatch, root, "open", (target, "r", 0)) == []


def test_a_database_connection_under_a_real_root_is_recorded(tmp_path, monkeypatch):
    root = tmp_path / "real_share"
    root.mkdir()
    assert _record(monkeypatch, root, "sqlite3.connect", (str(root / "watcher.db"),)) != []


def test_paths_outside_the_real_roots_are_ignored(tmp_path, monkeypatch):
    root = tmp_path / "real_share"
    root.mkdir()
    elsewhere = tmp_path / "real_share_sibling" / "x"
    assert _record(monkeypatch, root, "os.mkdir", (str(elsewhere), 0o777, None)) == []
    assert _record(monkeypatch, root, "os.rename", (str(tmp_path / "a"), str(tmp_path / "b"), None, None)) == []
