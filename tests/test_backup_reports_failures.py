"""A backup run says what failed, skips test residue, and fails loudly.

WHY: from 2026-10-13 one folder failed to back up every night. The log said
only "exit code 11" (robocopy ran with its file listing off and its error
lines were never logged), and `--once` exited 0 anyway, so the scheduled task
showed success to every check that reads it. The failures were pytest's
unreadable cache folders in a code checkout -- never data.
"""
import logging
import subprocess
from types import SimpleNamespace

import mousereach.watcher.backup as b

ROBOCOPY_OUT = """
2026/10/22 04:20:57 ERROR 5 (0x00000005) Copying File Y:\\src\\locked.db
Access is denied.
2026/10/22 04:20:58 ERROR 32 (0x00000020) Copying File Y:\\src\\busy.csv
The process cannot access the file because it is being used by another process.
"""


def test_a_failed_copy_is_named_in_the_log(monkeypatch, caplog):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=11, stdout=ROBOCOPY_OUT, stderr=""))
    w = b.BackupWatcher("Y:/src", "X:/dst", sync_dirs=["src"])
    with caplog.at_level(logging.WARNING, logger=b.logger.name):
        assert w._sync_dir("Y:/src", "X:/dst") == (0, 1)
    text = caplog.text
    assert "locked.db" in text and "Access is denied." in text
    assert "busy.csv" in text and "being used by another process" in text


def test_test_residue_is_never_copied(monkeypatch):
    seen = {}
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: seen.setdefault("cmd", cmd)
                        and SimpleNamespace(returncode=1, stdout="", stderr=""))
    b.BackupWatcher("Y:/src", "X:/dst")._sync_dir("Y:/src", "X:/dst")
    xd = seen["cmd"][seen["cmd"].index("/XD") + 1: seen["cmd"].index("/XF")]
    assert "pytest-cache-files-*" in xd and ".pytest_cache" in xd


def _run_main_once(monkeypatch, failures):
    import mousereach.config as mc
    monkeypatch.setattr(mc, "_load_config", lambda: {"backup": {
        "enabled": True, "source_root": "/s", "backup_root": "/d"}})

    class Fake(b.BackupWatcher):
        def run_once(self):
            return failures

    monkeypatch.setattr(b, "BackupWatcher", Fake)
    monkeypatch.setattr(b.sys, "argv", ["mousereach-backup", "--once"])
    return b.main()


def test_once_exits_1_when_a_folder_failed(monkeypatch, capsys):
    assert _run_main_once(monkeypatch, 1) == 1
    assert "did not sync completely" in capsys.readouterr().out


def test_once_exits_0_when_all_synced(monkeypatch, capsys):
    assert _run_main_once(monkeypatch, 0) == 0
