"""Stopping the watcher must never cost the work in flight.

WHY: there is no polite kill available on Windows. psutil's terminate() is
documented as an alias for kill(), and a watcher started detached -- which is
how the Restart button starts one -- has no console, so a Ctrl+C event cannot
reach it either. So the button's only real option was a hard kill, part-way
through a fourteen-minute pose, which before startup reclaim existed stranded
the video outright. An earlier version of this fix claimed to "ask before
insisting" and did not: on Windows both calls are the same call.

The watcher now checks a flag file between work items. Three things have to
hold or the cure is worse than the disease:
  * the loop must actually break on it,
  * it must be cleared on EVERY exit path, because a flag left behind stops
    the next watcher the instant it starts,
  * and a busy watcher must be left alone rather than killed.
"""
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.watcher.orchestrator as orch
import mousereach.watcher.health as health
from mousereach.watcher.orchestrator import DLCOrchestrator


# ---------------------------------------------------------------- the loop

class _Stub:
    """Enough orchestrator to run one pass of the real loop."""

    def __init__(self, root):
        self.db = SimpleNamespace()
        self.config = SimpleNamespace(poll_interval_seconds=0.01)
        self.worked = 0
        self._root = root

    def _scan_phase(self):
        pass

    def _get_next_work_item(self):
        self.worked += 1
        return None

    def shutdown(self):
        pass


def _loop_node(tmp_path, monkeypatch):
    o = object.__new__(DLCOrchestrator)
    stub = _Stub(tmp_path)
    for name in ("db", "config", "_scan_phase", "_get_next_work_item", "shutdown"):
        setattr(o, name, getattr(stub, name))
    o._stub = stub
    monkeypatch.setattr(orch, "require_processing_root", lambda: tmp_path)
    o._reclaim_orphaned_work = lambda: {}
    o._maybe_review_reprocess_scan = lambda *a, **k: None
    return o


def test_the_loop_stops_when_the_flag_is_there(tmp_path, monkeypatch):
    o = _loop_node(tmp_path, monkeypatch)
    (tmp_path / "watcher_stop.flag").write_text("stop")
    o.run(threading.Event())                     # returns rather than hanging
    assert o._stub.worked == 0, "it must not take new work after being asked to stop"


def test_the_loop_keeps_working_when_the_flag_is_absent(tmp_path, monkeypatch):
    o = _loop_node(tmp_path, monkeypatch)
    stop = threading.Event()

    def one_pass():
        o._stub.worked += 1
        stop.set()                               # end the loop after one cycle
        return None

    o._get_next_work_item = one_pass
    o.run(stop)
    assert o._stub.worked == 1


def test_stop_is_checked_before_pause(tmp_path, monkeypatch):
    """A paused watcher must still be stoppable, not wait for someone to
    un-pause it first."""
    o = _loop_node(tmp_path, monkeypatch)
    (tmp_path / "watcher_paused.flag").write_text("paused")
    (tmp_path / "watcher_stop.flag").write_text("stop")
    o.run(threading.Event())                     # must return, not spin paused
    assert o._stub.worked == 0


# ---------------------------------------------------------------- the button

class FakeProc:
    def __init__(self, pid, cmdline, exits=True):
        self.pid = pid
        self.info = {"cmdline": cmdline}
        self._exits = exits
        self.killed = False

    def is_running(self):
        return not self._exits and not self.killed

    def kill(self):
        self.killed = True


@pytest.fixture
def button(tmp_path, monkeypatch):
    """restart_watcher with a fake process table and a fake launcher."""
    monkeypatch.setattr(health, "require_processing_root", lambda: tmp_path,
                        raising=False)
    import mousereach.config as cfg
    monkeypatch.setattr(cfg, "require_processing_root", lambda: tmp_path)
    launched = []
    import subprocess
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: launched.append(a) or SimpleNamespace())
    monkeypatch.setattr(health.time, "sleep", lambda *_: None)
    return SimpleNamespace(root=tmp_path, launched=launched)


def _psutil(monkeypatch, procs, exit_on_wait=True):
    fake = SimpleNamespace(
        process_iter=lambda attrs=None: procs,
        wait_procs=lambda ps, timeout=None: (
            (list(ps), []) if exit_on_wait else ([], list(ps))),
    )
    import sys
    monkeypatch.setitem(sys.modules, "psutil", fake)
    return fake


def test_a_busy_watcher_is_asked_and_then_left_alone(button, monkeypatch):
    """The whole point: it must not be killed mid-pose."""
    p = FakeProc(1, ["mousereach-watch.exe"], exits=False)
    _psutil(monkeypatch, [p], exit_on_wait=False)
    ok, msg = health.restart_watcher(graceful_wait=0.01)
    assert ok is False
    assert not p.killed, "a busy watcher must never be killed by the button"
    assert button.launched == [], "and no replacement may be started"
    assert "busy" in msg.lower()


def test_the_flag_is_always_cleared_even_when_giving_up(button, monkeypatch):
    """A flag left behind would stop the NEXT watcher on sight."""
    p = FakeProc(1, ["mousereach-watch.exe"], exits=False)
    _psutil(monkeypatch, [p], exit_on_wait=False)
    health.restart_watcher(graceful_wait=0.01)
    assert not (button.root / "watcher_stop.flag").exists()


def test_a_watcher_that_stops_is_replaced_and_the_flag_cleared(button, monkeypatch):
    p = FakeProc(1, ["mousereach-watch.exe"], exits=True)
    _psutil(monkeypatch, [p], exit_on_wait=True)
    health.restart_watcher(graceful_wait=0.01)
    assert button.launched, "a stopped watcher must be replaced"
    assert not (button.root / "watcher_stop.flag").exists()
    assert not p.killed


def test_force_kills_only_what_refused_to_go(button, monkeypatch):
    p = FakeProc(1, ["mousereach-watch.exe"], exits=False)
    _psutil(monkeypatch, [p], exit_on_wait=False)
    health.restart_watcher(graceful_wait=0.01, force=True)
    assert p.killed
    assert button.launched
    assert not (button.root / "watcher_stop.flag").exists()


def test_the_console_script_spelling_is_matched(button, monkeypatch):
    """The original bug: only 'main_watch' was matched, so on a machine using
    the console script this stopped nothing and reported success."""
    p = FakeProc(1, ["mousereach-watch.exe"], exits=True)
    _psutil(monkeypatch, [p], exit_on_wait=True)
    health.restart_watcher(graceful_wait=0.01)
    assert (button.root / "watcher_stop.flag").exists() is False
    assert button.launched, "the console-script watcher must have been seen"


def test_nothing_running_just_starts_one(button, monkeypatch):
    _psutil(monkeypatch, [], exit_on_wait=True)
    health.restart_watcher(graceful_wait=0.01)
    assert button.launched
    assert not (button.root / "watcher_stop.flag").exists()
