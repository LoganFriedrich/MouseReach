"""The backup copies data, never machine state or half-written files, and the
panel's "Back up now" button runs the backup the config describes.

WHY: the backup is add-only (nothing on it is ever removed), so whatever it
copies once stays there for good. It copied claim markers (.return_claims),
liveness reports (.node_status), files still being written (*.part, *.partial)
and SQLite rollback journals (*.db-journal). Restored, a claim marker blocks
every other machine from a video, a status report makes a dead machine look
alive, and a journal makes SQLite roll back changes that committed long ago.
Hand-over folders (.inflight, .incoming) and the SQLite *-wal file hold real
data and must still be copied.

The panel button ignored the configured scope (backup.sync_dirs) and said
"Backup complete." even when a folder failed; its Save button wiped the
scope from the config.

No robocopy is run: subprocess.run is replaced. Path.home() points at tmp_path.
"""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.watcher.backup as b


def _robocopy_args(monkeypatch):
    seen = {}

    def fake_run(cmd, **k):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    b.BackupWatcher("S:/src", "B:/dst")._sync_dir("S:/src", "B:/dst")
    cmd = seen["cmd"]
    xd = cmd[cmd.index("/XD") + 1: cmd.index("/XF")]
    xf = cmd[cmd.index("/XF") + 1:]
    return xd, xf


def test_claim_and_status_folders_are_not_copied(monkeypatch):
    xd, _ = _robocopy_args(monkeypatch)
    for name in (".claims", ".return_claims", ".node_status"):
        assert name in xd, name


def test_hand_over_folders_are_still_copied(monkeypatch):
    # They hold real videos mid hand-over -- possibly the only copy.
    xd, _ = _robocopy_args(monkeypatch)
    assert ".inflight" not in xd and ".incoming" not in xd


def test_half_written_files_and_sqlite_journals_are_not_copied(monkeypatch):
    _, xf = _robocopy_args(monkeypatch)
    for pat in ("*.tmp", "*.part", "*.partial", "*.db-journal", "*.db-shm"):
        assert pat in xf, pat


def test_sqlite_wal_is_still_copied(monkeypatch):
    # The write-ahead log holds committed writes; without it they are lost.
    _, xf = _robocopy_args(monkeypatch)
    assert not any("wal" in p.lower() for p in xf)


# --- the panel ---------------------------------------------------------------

cw = pytest.importorskip("mousereach.watcher.control_widget")


def _fake_config(monkeypatch, backup):
    import mousereach.config as mc
    monkeypatch.setattr(mc, "_load_config", lambda: {"backup": backup})


def _press_back_up_now(monkeypatch, src, dst):
    """Press the panel's "Back up now" on a stand-in panel; the backup thread
    runs at once and the line the panel would show is returned."""
    shown = []

    class Signal:
        def connect(self, fn):
            self.fn = fn

        def emit(self, msg):
            self.fn(msg)

    class Worker:
        def __init__(self):
            self.done = Signal()

    class NowThread:
        def __init__(self, target=None, **k):
            self.target = target

        def start(self):
            self.target()

        def is_alive(self):
            return False

    monkeypatch.setattr(cw, "_BackupWorker", Worker)
    monkeypatch.setattr(cw.threading, "Thread", NowThread)
    monkeypatch.setattr(cw, "show_error", lambda m, *a, **k: pytest.fail(m))
    me = SimpleNamespace(
        _bkp_source=SimpleNamespace(text=lambda: src),
        _bkp_dest=SimpleNamespace(text=lambda: dst),
        _bkp_status=SimpleNamespace(setText=lambda m: None),
        _on_backup_done=lambda m: shown.append(m),
    )
    cw.WatcherControlWidget._run_backup(me)
    return shown[-1]


def test_panel_backup_uses_the_configured_scope(monkeypatch):
    _fake_config(monkeypatch, {"enabled": True, "source_root": "/s",
                               "backup_root": "/d", "sync_dirs": ["only/this"],
                               "robocopy_timeout_seconds": 7})
    built = {}

    class Fake(b.BackupWatcher):
        def run_once(self):
            built["dirs"] = self.sync_dirs
            built["timeout"] = self.robocopy_timeout
            built["src"], built["dst"] = str(self.source_root), str(self.backup_root)
            return 0

    monkeypatch.setattr(b, "BackupWatcher", Fake)
    msg = _press_back_up_now(monkeypatch, "/typed/src", "/typed/dst")
    assert msg == "Backup complete."
    assert built["dirs"] == ["only/this"]
    assert built["timeout"] == 7
    # What is typed in the panel wins over the saved folders.
    assert Path(built["src"]) == Path("/typed/src")
    assert Path(built["dst"]) == Path("/typed/dst")


def test_panel_backup_says_when_a_folder_failed(monkeypatch):
    _fake_config(monkeypatch, {"source_root": "/s", "backup_root": "/d"})

    class Fake(b.BackupWatcher):
        def run_once(self):
            return 2

    monkeypatch.setattr(b, "BackupWatcher", Fake)
    msg = _press_back_up_now(monkeypatch, "/s", "/d")
    assert msg != "Backup complete."
    assert "INCOMPLETE" in msg and "2 folder" in msg


def test_panel_save_keeps_the_scope_and_other_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    cfg = tmp_path / ".mousereach" / "config.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"backup": {
        "enabled": False, "source_root": "/old", "backup_root": "/oldb",
        "sync_dirs": ["only/this"], "robocopy_timeout_seconds": 99}}))

    me = SimpleNamespace(
        _bkp_enabled=SimpleNamespace(isChecked=lambda: True),
        _bkp_source=SimpleNamespace(text=lambda: " /new "),
        _bkp_dest=SimpleNamespace(text=lambda: "/newb"),
    )
    monkeypatch.setattr(cw, "show_info", lambda *a, **k: None)
    monkeypatch.setattr(cw, "show_error", lambda m, *a, **k: pytest.fail(m))
    cw.WatcherControlWidget._save_backup(me)

    saved = json.loads(cfg.read_text())["backup"]
    assert saved["enabled"] is True
    assert saved["source_root"] == "/new" and saved["backup_root"] == "/newb"
    assert saved["sync_dirs"] == ["only/this"]
    assert saved["robocopy_timeout_seconds"] == 99
