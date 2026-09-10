"""backup.sync_dirs config override reaches the BackupWatcher.

WHY: the default three-tree scope can exceed the backup volume's free space
(7.0 TB of sources vs 6.3 TB free, measured 2026-09-10). A lab must be able
to protect the trees that fit rather than have robocopy fill the disk and
fail partway with no complete tree. main() used to ignore the constructor's
sync_dirs parameter entirely.
"""
from mousereach.watcher.backup import BackupWatcher


def test_constructor_honors_sync_dirs():
    w = BackupWatcher("/src", "/dst", sync_dirs=["Databases"])
    assert w.sync_dirs == ["Databases"]


def test_constructor_defaults_without_override():
    w = BackupWatcher("/src", "/dst")
    assert w.sync_dirs == BackupWatcher.DEFAULT_SYNC_DIRS
    assert "Behavior/MouseReach_Pipeline" in w.sync_dirs


def test_constructor_honors_robocopy_timeout():
    w = BackupWatcher("/src", "/dst", robocopy_timeout=21600)
    assert w.robocopy_timeout == 21600
    assert BackupWatcher("/src", "/dst").robocopy_timeout == 3600


def test_main_passes_config_sync_dirs(monkeypatch, capsys):
    import mousereach.watcher.backup as b
    import mousereach.config as mc

    monkeypatch.setattr(mc, "_load_config", lambda: {"backup": {
        "enabled": True, "source_root": "/s", "backup_root": "/d",
        "sync_dirs": ["Tissue/MouseBrain_Pipeline", "Databases"],
    }})
    built = {}

    class Spy(BackupWatcher):
        def __init__(self, **kw):
            built.update(kw)
            super().__init__(**kw)

        def dry_run(self):
            pass

    monkeypatch.setattr(b, "BackupWatcher", Spy)
    monkeypatch.setattr(b.sys, "argv", ["mousereach-backup", "--dry-run"])
    b.main()
    assert built["sync_dirs"] == ["Tissue/MouseBrain_Pipeline", "Databases"]


def test_main_rejects_malformed_sync_dirs(monkeypatch):
    import pytest
    import mousereach.watcher.backup as b
    import mousereach.config as mc

    monkeypatch.setattr(mc, "_load_config", lambda: {"backup": {
        "enabled": True, "source_root": "/s", "backup_root": "/d",
        "sync_dirs": "Databases",           # a bare string, not a list
    }})
    monkeypatch.setattr(b.sys, "argv", ["mousereach-backup", "--dry-run"])
    with pytest.raises(SystemExit):
        b.main()
