"""mousereach-archive refuses when it cannot tell whether the watcher runs.

WHY: the archive command must not file beside the watcher (two programs moving
one video's files at once can split it between Processing and the filed
folder). It decided "is the watcher running" by reading the command lines of
Python programs. A watcher started by another Windows login cannot have its
command line read (AccessDenied), and that program was simply skipped -- so
from a reviewer's login on the machine that runs the watcher, "cannot tell"
became "not running" and the command filed beside the live watcher.

No real program, lock or configuration is touched: the process table, psutil
and the machine-wide lock are replaced by fakes.
"""
import sys
import types

import pytest

from mousereach.watcher import recording_guard
from mousereach.watcher import singleton

VID = "20990101_ZZZ9901_P1"


class _AccessDenied(Exception):
    pass


class _NoSuchProcess(Exception):
    pass


def _fake_psutil(monkeypatch, cmdlines, denied=(), gone=()):
    mod = types.ModuleType("psutil")
    mod.AccessDenied = _AccessDenied
    mod.NoSuchProcess = _NoSuchProcess

    class _P:
        def __init__(self, pid):
            self.pid = pid

        def cmdline(self):
            if self.pid in denied:
                raise _AccessDenied(self.pid)
            if self.pid in gone:
                raise _NoSuchProcess(self.pid)
            return cmdlines.get(self.pid, [])

    mod.Process = _P
    mod.process_iter = lambda attrs: iter(())
    monkeypatch.setitem(sys.modules, "psutil", mod)


def _table(monkeypatch, *pids):
    monkeypatch.setattr(recording_guard, "windows_process_table",
                        lambda: [(pid, 1, "python.exe") for pid in pids])


def _lock(monkeypatch, answer):
    # On master there is no lock check; raising=False lets the test run there
    # and fail on its assertion.
    monkeypatch.setattr(singleton, "watcher_lock_exists",
                        lambda *a, **k: answer, raising=False)


def test_another_logins_unreadable_program_counts_as_running(monkeypatch):
    from mousereach.archive import cli
    _table(monkeypatch, 501)
    _fake_psutil(monkeypatch, {}, denied={501})
    _lock(monkeypatch, None)            # the lock cannot be checked either
    assert cli.watcher_daemon_running() is True, (
        "a program whose command line cannot be read was taken as 'no watcher'")


def test_the_command_refuses_and_moves_nothing(monkeypatch, capsys):
    from mousereach.archive import cli
    _table(monkeypatch, 501)
    _fake_psutil(monkeypatch, {}, denied={501})
    _lock(monkeypatch, None)
    filed = []
    monkeypatch.setattr(cli, "archive_video", lambda *a, **k: filed.append(a) or {"success": True})
    monkeypatch.setattr(cli, "archive_all", lambda *a, **k: filed.append(a) or {"failed": 0})
    monkeypatch.setattr(sys, "argv", ["mousereach-archive", VID])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1
    assert filed == []
    assert "cannot be told" in capsys.readouterr().out


def test_the_watcher_lock_held_by_another_login_counts_as_running(monkeypatch):
    from mousereach.archive import cli
    _table(monkeypatch)                 # no program visible at all
    _fake_psutil(monkeypatch, {})
    _lock(monkeypatch, True)
    assert cli.watcher_daemon_running() is True


def test_no_lock_and_no_watcher_lets_the_command_run(monkeypatch):
    """The lock check answers "no lock exists": every way of starting a
    watcher takes that lock, so another login's unreadable program is not a
    watcher, and the command may file."""
    from mousereach.watcher import health
    _table(monkeypatch, 501, 502, 503)
    _fake_psutil(monkeypatch, {502: ["python", "-m", "some.other.tool"]},
                 denied={501}, gone={503})
    _lock(monkeypatch, False)
    running, why = health.watcher_presence()
    assert running is False, why


def test_an_ended_program_is_not_cannot_tell(monkeypatch):
    from mousereach.watcher import health
    _table(monkeypatch, 503)
    _fake_psutil(monkeypatch, {}, gone={503})
    _lock(monkeypatch, None)
    assert health.watcher_presence()[0] is False


def test_lock_probe_never_creates_the_lock():
    """Asking must not itself make a later watcher start refuse."""
    if sys.platform != "win32":
        pytest.skip("the machine-wide lock exists on Windows only")
    # A per-login name: creating a machine-wide one needs a privilege a test
    # login may lack, and the probe works the same for both.
    name = "Local\\mousereach-test-probe-%d" % id(object())
    assert singleton.watcher_lock_exists(name) is False
    assert singleton.watcher_lock_exists(name) is False
    handle = singleton.acquire_watcher_lock(name)
    try:
        assert handle, "the probe left the lock behind"
        assert singleton.watcher_lock_exists(name) is True
    finally:
        singleton.release_watcher_lock(handle)
    assert singleton.watcher_lock_exists(name) is False
