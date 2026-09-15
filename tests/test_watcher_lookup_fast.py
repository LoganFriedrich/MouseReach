"""Finding the watcher process must not read every process's command line.

WHY: on a busy server that took about 85 seconds per call, and the Watcher
Control panel asks every few seconds on the GUI thread, which froze the panel.
health._watcher_processes lists program names from one system snapshot and
reads command lines only for Python and MouseReach programs. These tests fake
both the snapshot and psutil, so no real process is touched.
"""
import sys
import types

import pytest

from mousereach.watcher import health, recording_guard


class _FakeProcess:
    def __init__(self, pid, cmdline, calls):
        self.pid = pid
        self._cmdline = cmdline
        self._calls = calls

    def cmdline(self):
        self._calls.append(self.pid)
        return self._cmdline


def _fake_psutil(monkeypatch, cmdlines, calls, iter_rows=None):
    mod = types.ModuleType("psutil")
    mod.Process = lambda pid: _FakeProcess(pid, cmdlines[pid], calls)

    def process_iter(attrs):
        for pid, cmd in (iter_rows or {}).items():
            p = _FakeProcess(pid, cmd, calls)
            p.info = {"cmdline": cmd}
            yield p

    mod.process_iter = process_iter
    monkeypatch.setitem(sys.modules, "psutil", mod)


def test_only_python_and_mousereach_programs_have_their_command_line_read(monkeypatch):
    table = [
        (10, 1, "explorer.exe"),
        (11, 1, "recorder.exe"),
        (12, 1, "python.exe"),
        (13, 1, "mousereach-watch-status.exe"),
        (14, 1, "python.exe"),
    ]
    cmdlines = {
        12: ["python", "-m", "some.other.tool"],
        13: ["mousereach-watch-status.exe"],
        14: ["python", "-c", "from mousereach.watcher.cli import main_watch; main_watch()"],
    }
    calls = []
    monkeypatch.setattr(recording_guard, "windows_process_table", lambda: table)
    _fake_psutil(monkeypatch, cmdlines, calls)

    assert health.watcher_running() == 14
    assert set(calls) <= {12, 13, 14}          # explorer / recorder never inspected
    assert [p.pid for p in health._watcher_processes()] == [14]


def test_falls_back_to_every_process_when_no_snapshot(monkeypatch):
    calls = []
    monkeypatch.setattr(recording_guard, "windows_process_table", lambda: None)
    _fake_psutil(monkeypatch, {}, calls,
                 iter_rows={5: ["mousereach-watch.exe"], 6: ["mousereach-watch-toggle.exe"]})
    assert health.watcher_running() == 5
    assert [p.pid for p in health._watcher_processes()] == [5]


def test_no_watcher_means_none(monkeypatch):
    calls = []
    monkeypatch.setattr(recording_guard, "windows_process_table",
                        lambda: [(20, 1, "python.exe")])
    _fake_psutil(monkeypatch, {20: ["python", "-m", "pytest"]}, calls)
    assert health.watcher_running() is None
