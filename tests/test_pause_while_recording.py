"""The watcher stands aside while a recording program is running.

WHY: a GPU node in a behaviour room both records videos and poses them. A pose
saturates the GPU, CPU and disk for ~14 minutes, and a recording that drops
frames because of it is behaviour that can never be filmed again, while a pose
can always be re-run. A node cannot know a recording schedule, so it watches
for the recording program itself (watcher.pause_while_running) and pauses
while it runs. Recording must always win:

  * the check fails SAFE (cannot look -> paused),
  * a recording that starts during a long scan stops the next item before it
    begins, and run-once honours the pause and the stop flag,
  * a pose already running is stopped and put back in the queue WITHOUT being
    counted as a failure,
  * claimed re-pose requests stay alive while paused (a day without heartbeat
    hands them to another node),
  * and with no programs configured nothing changes: the in-process DLC path
    runs exactly as before.

No real process is inspected and no real DLC runs: psutil, the process scan,
the clock and the DLC runners are all faked. Everything lives under tmp_path
(tests/conftest.py fails any test that writes into a real pipeline folder).
The placeholder program name is recorder.exe.
"""
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.watcher.orchestrator as orch
import mousereach.watcher.recording_guard as rg
import mousereach.watcher.repose as repose
from mousereach.config import WatcherConfig
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import DLCOrchestrator
from mousereach.watcher.recording_guard import (
    HAND_PAUSE_REASON, RecordingGuard, pause_reason, running_programs,
)

PROGRAM = "recorder.exe"
POSE = "DLC_resnet101_MPSAOct27shuffle3_100000"
VID = "20250101_CNT0101_P1"


# ----------------------------------------------------------------- helpers

class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class Scan:
    """A fake process scan: returns whichever configured names are 'open'."""

    def __init__(self, running=(), error=None):
        self.running = set(running)
        self.error = error
        self.calls = 0

    def __call__(self, names):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return [n for n in names if n in self.running]


def _guard(scan, clock=None, grace=120, cache=0, names=(PROGRAM,)):
    return RecordingGuard(list(names), grace_seconds=grace, clock=clock or Clock(),
                          scan=scan, cache_seconds=cache)


class FakeProc:
    def __init__(self, name):
        self.info = {"name": name}


def _fake_psutil(monkeypatch, names):
    # The Windows snapshot is switched off too, or on Windows the real process
    # list would be read instead of this fake one.
    monkeypatch.setattr(rg, "windows_process_table", lambda: None)
    fake = SimpleNamespace(process_iter=lambda attrs=None: [FakeProc(n) for n in names])
    monkeypatch.setitem(sys.modules, "psutil", fake)
    return fake


# ----------------------------------------------------------------- the guard

def test_a_running_program_blocks_and_a_closed_one_does_not():
    scan = Scan(running=[PROGRAM])
    g = _guard(scan, grace=0)
    assert g.reason() == "recorder.exe is running"
    scan.running.clear()
    assert g.reason() is None


def test_nothing_configured_never_pauses_and_never_looks():
    scan = Scan(running=[PROGRAM])
    g = RecordingGuard([], scan=scan)
    assert not g.enabled
    assert g.reason() is None
    assert scan.calls == 0, "an empty list must mean today's behaviour: no check at all"


def test_blank_entries_are_ignored():
    scan = Scan()
    g = RecordingGuard(["", "   ", None], scan=scan)
    assert g.names == [] and g.reason() is None and scan.calls == 0


def test_work_resumes_only_after_the_grace_period():
    clock, scan = Clock(), Scan(running=[PROGRAM])
    g = _guard(scan, clock=clock, grace=120)
    assert g.reason() == "recorder.exe is running"
    clock.t += 10
    scan.running.clear()
    # The timer starts when the program is first seen gone.
    assert g.reason() == "recorder.exe closed 0 s ago; resuming after 120 s"
    clock.t += 30
    assert g.reason() == "recorder.exe closed 30 s ago; resuming after 120 s"
    clock.t += 89
    assert g.reason() is not None
    clock.t += 2
    assert g.reason() is None


def test_reopening_during_the_grace_period_restarts_it():
    clock, scan = Clock(), Scan(running=[PROGRAM])
    g = _guard(scan, clock=clock, grace=60)
    g.reason()
    scan.running.clear()
    clock.t += 1
    g.reason()
    clock.t += 50
    scan.running.add(PROGRAM)            # the operator opened it again
    assert g.reason() == "recorder.exe is running"
    scan.running.clear()
    clock.t += 1
    g.reason()
    clock.t += 59
    assert g.reason() is not None, "the grace period must start over, not continue"


def test_a_node_that_never_saw_the_program_works_at_once():
    g = _guard(Scan(), grace=120)
    assert g.reason() is None


def test_names_match_case_insensitively(monkeypatch):
    _fake_psutil(monkeypatch, ["System", "RECORDER.EXE", "other.exe"])
    assert running_programs(["recorder.exe"]) == ["recorder.exe"]
    assert running_programs(["Recorder.Exe", "missing.exe"]) == ["Recorder.Exe"]


def test_a_name_typed_without_exe_or_as_a_path_still_matches(monkeypatch):
    _fake_psutil(monkeypatch, ["recorder.exe"])
    assert running_programs(["recorder"]) == ["recorder"]
    assert running_programs(["Programs/Recorder/recorder.exe"]) == \
        ["Programs/Recorder/recorder.exe"]
    assert running_programs(["some folder\\recorder.exe"]) == ["some folder\\recorder.exe"]
    assert running_programs(["record"]) == []


def test_running_programs_with_no_names_does_not_need_psutil(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)   # import psutil -> ImportError
    assert running_programs([]) == []
    assert running_programs(["", "  "]) == []


def test_psutil_missing_keeps_the_watcher_paused(monkeypatch):
    monkeypatch.setattr(rg, "windows_process_table", lambda: None)
    monkeypatch.setitem(sys.modules, "psutil", None)
    g = RecordingGuard([PROGRAM], cache_seconds=0)
    reason = g.reason()
    assert reason.startswith("cannot check for recording programs: psutil is not installed")
    assert "pip install psutil" in reason, "the fix must be named"


@pytest.mark.skipif(sys.platform != "win32", reason="the process snapshot is Windows-only")
def test_the_windows_snapshot_lists_this_python_process_quickly():
    # WHY: psutil took over a minute and a half to list process names on a
    # busy machine; the guard asks every few seconds while a pose runs.
    import os
    import time
    t0 = time.monotonic()
    table = rg.windows_process_table()
    assert time.monotonic() - t0 < 10
    assert table and any(pid == os.getpid() for pid, _ppid, _exe in table)
    assert "python" in dict((pid, exe) for pid, _pp, exe in table)[os.getpid()].lower()


def test_a_new_guard_keeps_the_grace_period_of_the_old_one():
    clock, scan = Clock(), Scan(running=[PROGRAM])
    old = _guard(scan, clock=clock, grace=120)
    old.reason()
    scan.running.clear()
    clock.t += 1
    old.reason()
    clock.t += 30
    new = _guard(scan, clock=clock, grace=120, names=(PROGRAM, "other.exe"))
    new.inherit_history(old)
    assert new.reason() == "recorder.exe closed 30 s ago; resuming after 120 s"


def test_a_scan_that_raises_keeps_the_watcher_paused():
    scan = Scan(error=RuntimeError("access denied"))
    g = _guard(scan)
    assert g.reason() == "cannot check for recording programs: RuntimeError: access denied"


def test_after_a_failed_check_the_grace_period_applies():
    """A check that could not look may have hidden a recording, so work does
    not start the moment the check works again."""
    clock, scan = Clock(), Scan(error=OSError("busy"))
    g = _guard(scan, clock=clock, grace=120)
    assert g.reason().startswith("cannot check")
    scan.error = None
    clock.t += 5
    assert g.reason() == "recorder.exe closed 0 s ago; resuming after 120 s"
    clock.t += 121
    assert g.reason() is None


def test_the_process_list_is_cached():
    clock, scan = Clock(), Scan(running=[PROGRAM])
    g = _guard(scan, clock=clock, cache=5)
    g.reason()
    g.reason()
    clock.t += 4.9
    g.reason()
    assert scan.calls == 1
    clock.t += 0.2
    g.reason()
    assert scan.calls == 2


def test_reasons_are_ascii():
    scan = Scan(error=RuntimeError("caf\u00e9 \u2192 broken"))
    reason = _guard(scan).reason()
    reason.encode("ascii")                           # must not raise


# ----------------------------------------------------------------- pause_reason

def test_the_hand_pause_wins_over_a_recording(tmp_path):
    (tmp_path / "watcher_paused.flag").write_text("paused")
    guard = _guard(Scan(running=[PROGRAM]))
    assert pause_reason(tmp_path, guard=guard) == HAND_PAUSE_REASON
    assert HAND_PAUSE_REASON == "paused by hand (watcher_paused.flag)"


def test_pause_reason_falls_through_to_the_guard(tmp_path):
    assert pause_reason(tmp_path, guard=_guard(Scan(running=[PROGRAM]))) == \
        "recorder.exe is running"
    assert pause_reason(tmp_path, guard=_guard(Scan())) is None


def test_pause_reason_builds_a_guard_from_config(tmp_path, monkeypatch):
    _fake_psutil(monkeypatch, ["recorder.exe"])
    cfg = WatcherConfig({"pause_while_running": [PROGRAM]})
    assert pause_reason(tmp_path, config=cfg) == "recorder.exe is running"
    assert pause_reason(tmp_path, config=WatcherConfig({})) is None


# ----------------------------------------------------------------- config

def test_config_defaults_change_nothing():
    cfg = WatcherConfig({})
    assert cfg.pause_while_running == []
    assert cfg.pause_resume_grace_seconds == 120
    d = cfg.to_dict()
    assert "pause_while_running" not in d, "an unset list is not written"
    assert d["pause_resume_grace_seconds"] == 120


def test_config_round_trips_the_list_and_grace():
    cfg = WatcherConfig({"pause_while_running": [PROGRAM, " ", "Other.exe "],
                         "pause_resume_grace_seconds": 300})
    assert cfg.pause_while_running == [PROGRAM, "Other.exe"]
    again = WatcherConfig(cfg.to_dict())
    assert again.pause_while_running == [PROGRAM, "Other.exe"]
    assert again.pause_resume_grace_seconds == 300


def test_config_forgives_a_bare_string_and_a_bad_grace():
    cfg = WatcherConfig({"pause_while_running": PROGRAM,
                         "pause_resume_grace_seconds": "soon"})
    assert cfg.pause_while_running == [PROGRAM]
    assert cfg.pause_resume_grace_seconds == 120


# ----------------------------------------------------------------- the loop

def _node(tmp_path, monkeypatch, scan=None, names=(PROGRAM,), grace=0):
    """A DLC orchestrator built without its constructor, enough for the loop."""
    o = object.__new__(DLCOrchestrator)
    o.config = SimpleNamespace(poll_interval_seconds=0.01,
                               pause_while_running=list(names),
                               pause_resume_grace_seconds=grace)
    o.db = SimpleNamespace()
    o.hostname = "NODE-A"
    if scan is not None:
        o._recording_guard = RecordingGuard(list(names), grace_seconds=grace,
                                            scan=scan, cache_seconds=0)
    monkeypatch.setattr(orch, "require_processing_root", lambda: tmp_path)
    o._reclaim_orphaned_work = lambda: {}
    o._maybe_review_reprocess_scan = lambda *a, **k: None
    o.shutdown = lambda: None
    return o


def _lines(caplog, text):
    return [r for r in caplog.records if text in r.getMessage()]


def test_entering_and_leaving_a_pause_is_logged_once_each(tmp_path, monkeypatch, caplog):
    scan = Scan(running=[PROGRAM])
    o = _node(tmp_path, monkeypatch, scan=scan)
    caplog.set_level(logging.INFO, logger=orch.logger.name)
    for _ in range(4):
        assert o._is_paused() is True
    assert o._pause_reason == "recorder.exe is running"
    scan.running.clear()
    for _ in range(3):
        assert o._is_paused() is False
    paused = _lines(caplog, "Watcher PAUSED")
    assert len(paused) == 1 and "recorder.exe is running" in paused[0].getMessage()
    assert paused[0].levelno == logging.INFO
    assert len(_lines(caplog, "Watcher RESUMED")) == 1


def test_the_hand_flag_still_pauses_with_no_programs_configured(tmp_path, monkeypatch):
    o = _node(tmp_path, monkeypatch, names=())
    assert o._is_paused() is False
    (tmp_path / "watcher_paused.flag").write_text("paused")
    assert o._is_paused() is True
    assert o._pause_reason == HAND_PAUSE_REASON


def test_a_guard_is_built_from_config_when_the_constructor_was_skipped(tmp_path, monkeypatch):
    o = _node(tmp_path, monkeypatch, names=(PROGRAM,), grace=45)
    g = o._get_recording_guard()
    assert g.names == [PROGRAM] and g.grace_seconds == 45
    assert o._get_recording_guard() is g, "one guard for the life of the watcher"


def test_a_recording_that_starts_during_the_scan_stops_the_next_item(tmp_path, monkeypatch):
    scan = Scan()
    o = _node(tmp_path, monkeypatch, scan=scan)
    stop = threading.Event()
    dispatched, scans = [], []

    def scan_phase():
        scans.append(1)
        scan.running.add(PROGRAM)          # someone opens the recorder mid-scan

    o._scan_phase = scan_phase
    o._get_next_work_item = lambda: {"type": "single_dlc", "id": VID, "data": {}}
    o._dispatch_work = lambda work: dispatched.append(work)
    o._while_paused = lambda: stop.set()
    o.run(stop)
    assert scans == [1]
    assert dispatched == [], "no work may start once the recording program is open"


def test_run_once_does_nothing_while_paused(tmp_path, monkeypatch):
    o = _node(tmp_path, monkeypatch, scan=Scan(running=[PROGRAM]))
    touched = []
    o._maybe_review_reprocess_scan = lambda *a, **k: touched.append("reprocess")
    o._scan_phase = lambda: touched.append("scan")
    o._get_next_work_item = lambda: {"type": "single_dlc", "id": VID, "data": {}}
    o._dispatch_work = lambda work: touched.append("dispatch")
    o.run_once()
    assert touched == []


def test_run_once_does_nothing_when_a_stop_was_requested(tmp_path, monkeypatch):
    o = _node(tmp_path, monkeypatch, names=())
    (tmp_path / "watcher_stop.flag").write_text("stop")
    touched = []
    o._scan_phase = lambda: touched.append("scan")
    o._get_next_work_item = lambda: {"type": "single_dlc", "id": VID, "data": {}}
    o._dispatch_work = lambda work: touched.append("dispatch")
    o.run_once()
    assert touched == []


def test_run_once_stops_between_items_when_a_recording_starts(tmp_path, monkeypatch):
    scan = Scan()
    o = _node(tmp_path, monkeypatch, scan=scan)
    o._scan_phase = lambda: None
    items = [{"type": "single_dlc", "id": f"v{i}", "data": {}} for i in range(3)]
    o._get_next_work_item = lambda: items[0] if items else None
    done = []

    def dispatch(work):
        done.append(work["id"])
        items.pop(0)
        scan.running.add(PROGRAM)          # recording starts during item 1
        return True

    o._dispatch_work = dispatch
    o.run_once()
    assert done == ["v0"]


def test_re_pose_claims_are_heartbeated_while_paused(tmp_path, monkeypatch):
    o = _node(tmp_path, monkeypatch, scan=Scan(running=[PROGRAM]))
    stop = threading.Event()
    beats = []

    def heartbeat(db, *, hostname, repose_dir=None):
        beats.append(hostname)
        stop.set()
        return 1

    monkeypatch.setattr(repose, "heartbeat", heartbeat)
    o._scan_phase = lambda: pytest.fail("the scan must not run while paused")
    # A watchdog, so a loop that never heartbeats fails this test instead of
    # hanging the whole suite (nothing else would ever set the stop event).
    watchdog = threading.Timer(5, stop.set)
    watchdog.start()
    try:
        o.run(stop)
    finally:
        watchdog.cancel()
    assert beats == ["NODE-A"]


def test_re_pose_requests_are_not_taken_while_paused(tmp_path, monkeypatch):
    """Taking a request copies a whole video over the network; a recording that
    started during the (long) scan must not share the disk with it."""
    scan = Scan(running=[PROGRAM])
    o = _node(tmp_path, monkeypatch, scan=scan)
    o.config.dlc_config_path = None
    o.file_watcher = SimpleNamespace(scan=lambda: SimpleNamespace(
        new_collages=0, new_singles=0, stable_ready=0))
    o._scan_for_dlc_completions = lambda: 0
    o._adopt_untracked_queue_files = lambda: None
    beats, consumed = [], []
    monkeypatch.setattr(repose, "heartbeat",
                        lambda db, *, hostname, repose_dir=None: beats.append(hostname) or 1)
    monkeypatch.setattr(repose, "consume_requests",
                        lambda *a, **k: consumed.append(1) or {"queued": 0, "completed": 0})
    o._scan_phase()
    assert beats == ["NODE-A"], "claims already held must still be kept alive"
    assert consumed == []
    scan.running.clear()                       # grace is 0 in _node
    o._scan_phase()
    assert consumed == [1]


def test_a_change_of_pause_cause_is_logged(tmp_path, monkeypatch, caplog):
    o = _node(tmp_path, monkeypatch, scan=Scan(running=[PROGRAM]))
    caplog.set_level(logging.INFO, logger=orch.logger.name)
    flag = tmp_path / "watcher_paused.flag"
    flag.write_text("paused")
    assert o._is_paused()
    flag.unlink()                              # the operator pressed Resume
    assert o._is_paused() and o._is_paused()
    now = _lines(caplog, "still PAUSED, now because")
    assert len(now) == 1 and "recorder.exe is running" in now[0].getMessage()
    assert len(_lines(caplog, "Watcher PAUSED")) == 1


def test_startup_stops_leftover_poses_first_and_only_when_programs_are_listed(monkeypatch):
    import mousereach.dlc.core.interruptible as interruptible
    events = []
    monkeypatch.setattr(interruptible, "kill_orphaned_workers",
                        lambda: events.append("kill") or 0)
    o = object.__new__(DLCOrchestrator)
    o.db = SimpleNamespace(get_videos_in_state=lambda s: events.append("list") or [],
                           get_collages_in_state=lambda s: [])
    o.config = SimpleNamespace(pause_while_running=[])
    assert o._reclaim_orphaned_work() == {}
    assert "kill" not in events, "no programs listed: no pose can have run in a child"
    events.clear()
    o.config.pause_while_running = [PROGRAM]
    o._reclaim_orphaned_work()
    assert events[0] == "kill", "a leftover pose must be stopped before its video is requeued"


# ----------------------------------------------------------------- live settings

def _file_node(tmp_path, monkeypatch, section):
    """A node whose config came from a settings FILE, as a real watcher's does."""
    import json
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"watcher": section}), encoding="utf-8")
    o = _node(tmp_path, monkeypatch, names=())
    o.config = WatcherConfig(section)
    o.config.source_file = cfg_file
    o.config.source_mtime = cfg_file.stat().st_mtime
    return o, cfg_file


def _rewrite(path, section=None, raw=None):
    import json
    import os
    old = path.stat().st_mtime
    path.write_text(raw if raw is not None else json.dumps({"watcher": section}),
                    encoding="utf-8")
    os.utime(path, (old + 10, old + 10))       # a visible change on any clock


def test_a_program_added_while_running_pauses_without_a_restart(tmp_path, monkeypatch, caplog):
    _fake_psutil(monkeypatch, ["recorder.exe"])
    o, cfg_file = _file_node(tmp_path, monkeypatch, {"poll_interval_seconds": 30})
    caplog.set_level(logging.INFO, logger=orch.logger.name)
    assert o._is_paused() is False
    _rewrite(cfg_file, {"poll_interval_seconds": 30, "pause_while_running": [PROGRAM]})
    assert o._is_paused() is True
    assert o._pause_reason == "recorder.exe is running"
    assert o.config.pause_while_running == [PROGRAM], "the pose path now uses the stoppable runner"
    assert o._recording_abort_reason() == "recorder.exe is running"
    assert _lines(caplog, "Recording-program settings changed")


def test_removing_the_program_resumes_but_other_settings_wait_for_a_restart(tmp_path, monkeypatch):
    _fake_psutil(monkeypatch, ["recorder.exe"])
    o, cfg_file = _file_node(tmp_path, monkeypatch, {
        "pause_while_running": [PROGRAM], "pause_resume_grace_seconds": 0,
        "poll_interval_seconds": 30})
    assert o._is_paused() is True
    _rewrite(cfg_file, {"poll_interval_seconds": 5})
    assert o._is_paused() is False
    assert o.config.pause_while_running == []
    assert o.config.poll_interval_seconds == 30


def test_a_grace_only_change_keeps_the_guard_and_its_timer(tmp_path, monkeypatch):
    _fake_psutil(monkeypatch, [])
    o, cfg_file = _file_node(tmp_path, monkeypatch, {
        "pause_while_running": [PROGRAM], "pause_resume_grace_seconds": 120})
    g = o._get_recording_guard()
    _rewrite(cfg_file, {"pause_while_running": ["RECORDER.EXE"],
                        "pause_resume_grace_seconds": 300})
    o._is_paused()
    assert o._get_recording_guard() is g and g.grace_seconds == 300


def test_a_half_written_settings_file_changes_nothing(tmp_path, monkeypatch, caplog):
    _fake_psutil(monkeypatch, ["recorder.exe"])
    o, cfg_file = _file_node(tmp_path, monkeypatch, {"pause_while_running": [PROGRAM]})
    caplog.set_level(logging.INFO, logger=orch.logger.name)
    _rewrite(cfg_file, raw='{"watcher": {"pause_while')
    assert o._is_paused() is True
    assert o._is_paused() is True
    assert o.config.pause_while_running == [PROGRAM]
    assert len(_lines(caplog, "Could not re-read")) == 1, "warned once, not every poll"


def test_a_config_not_loaded_from_a_file_is_never_reread(tmp_path, monkeypatch):
    o = _node(tmp_path, monkeypatch, names=())
    o._refresh_recording_settings()                  # SimpleNamespace: no source
    o.config = WatcherConfig({})
    o._refresh_recording_settings()                  # built from a dict: no source
    assert o.config.pause_while_running == []


def test_watcher_config_load_records_its_file(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".mousereach").mkdir()
    f = tmp_path / ".mousereach" / "config.json"
    f.write_text('{"watcher": {"pause_while_running": ["recorder.exe"]}}', encoding="utf-8")
    cfg = WatcherConfig.load()
    assert cfg.source_file == f and cfg.source_mtime == f.stat().st_mtime
    assert cfg.pause_while_running == [PROGRAM]


# ----------------------------------------------------------------- dashboard health

def test_the_dashboard_health_line_says_why_the_watcher_is_paused(monkeypatch):
    from mousereach.watcher import health
    monkeypatch.setattr(health, "watcher_running", lambda: 4242)
    monkeypatch.setattr(health, "_current_pause_reason", lambda: "recorder.exe is running")
    lines = health.health_report(None)
    assert lines[0] == "The auto-processor is RUNNING."
    assert "PAUSED: recorder.exe is running" in lines[1]
    assert "Close the recording program" in lines[1]
    monkeypatch.setattr(health, "_current_pause_reason", lambda: HAND_PAUSE_REASON)
    assert "Press Resume" in health.health_report(None)[1]
    monkeypatch.setattr(health, "_current_pause_reason",
                        lambda: "cannot check for recording programs: boom")
    assert "stays paused" in health.health_report(None)[1]
    monkeypatch.setattr(health, "_current_pause_reason", lambda: None)
    assert health.health_report(None) == ["The auto-processor is RUNNING."]


@pytest.mark.parametrize("cmd, is_watcher", [
    ("python Scripts\\mousereach-watch.exe", True),
    ("mousereach-watch --once", True),
    ('python -c "from mousereach.watcher.cli import main_watch; main_watch()"', True),
    ("Scripts\\mousereach-watch-status.exe --json", False),
    ("mousereach-watch-toggle --status", False),
    ("mousereach-watch-recorders --list", False),
    ("python -m mousereach.dlc.core.interruptible_worker args.json", False),
])
def test_only_the_watcher_itself_counts_as_a_running_watcher(cmd, is_watcher):
    # WHY: running mousereach-watch-status made a PC with no watcher look
    # RUNNING, so the panel refused to start one.
    from mousereach.watcher.health import is_watcher_command
    assert is_watcher_command(cmd) is is_watcher


def test_the_paused_heartbeat_is_rate_limited(tmp_path, monkeypatch):
    o = _node(tmp_path, monkeypatch, names=())
    o.config.poll_interval_seconds = 30
    beats = []
    monkeypatch.setattr(repose, "heartbeat",
                        lambda db, *, hostname, repose_dir=None: beats.append(1) or 1)
    o._while_paused()
    o._while_paused()
    assert len(beats) == 1
    o._repose_heartbeat(force=True)                  # the scan always beats
    assert len(beats) == 2


def test_a_failing_heartbeat_never_breaks_the_paused_loop(tmp_path, monkeypatch):
    o = _node(tmp_path, monkeypatch, names=())
    monkeypatch.setattr(repose, "heartbeat",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("share gone")))
    o._while_paused()                                # must not raise


# ----------------------------------------------------------------- the pose

LAYOUT = {
    "NAS_ROOT": "share",
    "DLC_STAGING": "share/Processing/Posed",
    "REVIEW_ROOT": "share/Processing/Review",
    "TRIAGE_REVIEW": "share/Processing/Review/triage",
    "DEEP_REVIEW": "share/Processing/Review/deep_review",
    "REPOSE_QUEUE": "share/Processing/Repose_Queue",
    "ANALYZED_OUTPUT": "share/Analyzed",
    "PROCESSING_ROOT": "node",
    "DLC_QUEUE": "node/DLC_Queue",
    "PROCESSING": "node/Processing",
}


@pytest.fixture
def pose_env(tmp_path, monkeypatch):
    import mousereach.dlc.core as dlc_core
    import mousereach.dlc.core.interruptible as interruptible

    dirs = {}
    for name, sub in LAYOUT.items():
        d = tmp_path / sub
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(orch.Paths, name, d, raising=False)
        dirs[name.lower()] = d
    monkeypatch.setattr(orch, "require_processing_root", lambda: dirs["processing_root"])
    monkeypatch.setattr(repose, "declared_scorer", lambda: POSE)
    monkeypatch.setattr(dlc_core, "resolve_dlc_shuffle", lambda: (3, "test"))

    cfg = tmp_path / "config.yaml"
    cfg.write_text("model", encoding="ascii")
    o = object.__new__(DLCOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "watcher.db")
    o.hostname = "NODE-A"
    o.config = SimpleNamespace(also_process=False, dlc_gpu_device=0, dlc_config_path=cfg,
                               max_retries=3, work_priority=None, poll_interval_seconds=30,
                               pause_while_running=[PROGRAM], pause_resume_grace_seconds=120)
    o.coordinator = None
    synced = []
    o._sync_to_connectome = lambda vid, state, **k: synced.append((vid, state))

    mp4 = dirs["dlc_queue"] / f"{VID}.mp4"
    mp4.write_bytes(b"video")
    o.db.register_video(video_id=VID, source_path=str(mp4), current_path=str(mp4))
    o.db.force_state(VID, "dlc_queued", reason="test setup", current_path=str(mp4))

    calls = SimpleNamespace(batch=[], interruptible=[])

    def fake_batch(video_paths, config_path, output_dir, gpu, save_as_csv, shuffle):
        calls.batch.append(Path(video_paths[0]))
        (Path(output_dir) / f"{VID}{POSE}.h5").write_bytes(b"pose")
        return [{"status": "success"}]

    monkeypatch.setattr(dlc_core, "run_dlc_batch", fake_batch)

    def use_interruptible(result, write_pose=False):
        def fake(video_path, config_path, output_dir, gpu, shuffle, should_abort, **kw):
            calls.interruptible.append(SimpleNamespace(
                video=Path(video_path), output_dir=Path(output_dir), gpu=gpu,
                shuffle=shuffle, should_abort=should_abort))
            if write_pose:
                (Path(output_dir) / f"{VID}{POSE}.h5").write_bytes(b"pose")
            return dict(result, video=str(video_path))
        monkeypatch.setattr(interruptible, "run_dlc_single_interruptible", fake)

    return SimpleNamespace(o=o, db=o.db, mp4=mp4, calls=calls, synced=synced,
                           use_interruptible=use_interruptible, **dirs)


def _work(env):
    return {"type": "single_dlc", "id": VID, "data": env.db.get_video(VID)}


def _steps(db, step):
    conn = db._get_connection()
    try:
        return [(r[0], r[1]) for r in conn.execute(
            "SELECT status, message FROM processing_log WHERE video_id=? AND step=? "
            "ORDER BY id", (VID, step))]
    finally:
        conn.close()


def test_a_pose_stopped_for_a_recording_goes_back_to_the_queue(pose_env):
    env = pose_env
    reason = "recorder.exe is running"
    env.use_interruptible({"status": "aborted", "abort_reason": reason})
    env.o._recording_guard = RecordingGuard([PROGRAM], scan=Scan(running=[PROGRAM]),
                                            cache_seconds=0)

    assert env.o._process_single_dlc(_work(env)) is False
    row = env.db.get_video(VID)
    assert row["state"] == "dlc_queued"
    assert int(row.get("error_count") or 0) == 0, "an abort must not spend a retry"
    assert Path(row["current_path"]) == env.mp4
    assert ("aborted", reason) in _steps(env.db, "dlc")
    assert not any(s == "failed" for s, _ in _steps(env.db, "dlc"))
    assert env.synced[-1] == (VID, "dlc_queued")
    assert env.calls.batch == [], "the in-process runner cannot be stopped; not used"

    # The runner was asked to stop on the watcher's own guard.
    call = env.calls.interruptible[0]
    assert call.video == env.mp4 and call.output_dir == env.dlc_queue and call.shuffle == 3
    assert call.should_abort() == reason


def test_an_aborted_pose_from_a_state_with_no_legal_move_is_forced_back(pose_env, monkeypatch):
    env = pose_env
    env.use_interruptible({"status": "aborted", "abort_reason": "recorder.exe is running"})
    real_update = env.db.update_state

    def update(video_id, new_state, **kw):
        if new_state == "dlc_running":
            # Something else moved the row while the pose ran.
            real_update(video_id, new_state, **kw)
            env.db.force_state(video_id, "dlc_complete", reason="test: moved underneath")
            return
        return real_update(video_id, new_state, **kw)

    monkeypatch.setattr(env.db, "update_state", update)
    from mousereach.watcher.db import VIDEO_TRANSITIONS
    monkeypatch.setitem(VIDEO_TRANSITIONS, "dlc_complete",
                        [s for s in VIDEO_TRANSITIONS["dlc_complete"] if s != "dlc_queued"])
    assert env.o._process_single_dlc(_work(env)) is False
    row = env.db.get_video(VID)
    assert row["state"] == "dlc_queued"
    assert int(row.get("error_count") or 0) == 0


def test_an_interruptible_pose_that_finishes_is_handled_as_before(pose_env):
    env = pose_env
    env.use_interruptible({"status": "success", "dlc_scorer": POSE}, write_pose=True)
    assert env.o._process_single_dlc(_work(env)) is True
    row = env.db.get_video(VID)
    assert row["state"] == "dlc_complete"
    assert Path(row["dlc_output_path"]).name == f"{VID}{POSE}.h5"
    assert env.synced[-1] == (VID, "dlc_complete")


def test_an_interruptible_pose_that_fails_is_marked_failed_as_before(pose_env):
    env = pose_env
    env.use_interruptible({"status": "failed", "error": "child died"})
    with pytest.raises(RuntimeError, match="child died"):
        env.o._process_single_dlc(_work(env))
    row = env.db.get_video(VID)
    assert row["state"] == "failed"
    assert int(row.get("error_count") or 0) == 1


def test_no_programs_configured_keeps_the_in_process_dlc_path(pose_env, monkeypatch):
    import mousereach.dlc.core.interruptible as interruptible
    env = pose_env
    env.o.config.pause_while_running = []
    monkeypatch.setattr(interruptible, "run_dlc_single_interruptible",
                        lambda *a, **k: pytest.fail("must not run in a child process"))
    assert env.o._process_single_dlc(_work(env)) is True
    assert env.calls.batch == [env.mp4]
    assert env.db.get_video(VID)["state"] == "dlc_complete"
