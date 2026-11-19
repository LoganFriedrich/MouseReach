"""A shared-drive outage is a pause, never an error, and needs nobody to attend it.

WHY: when the file server holding the shared pipeline folder went away for a
while, every graphics-card PC's watcher hung for hours inside shared-drive calls
with no time limit (the scan, the work-order file, the version declaration,
staging, the shared record) and never read its stop flag; one PC read the
version declaration as "not there" and stopped posing as if it were
misconfigured; and the processing machine's watcher, restarted during the
outage, crashed at startup and stayed down after the drive came back.

These tests stand in for the dead drive with a patched os.stat / os.listdir
that either HANG (until the test lets them go) or raise the network errors
Windows gives (WinError 64 "network name no longer available", 121
"semaphore timeout"), only for paths on the fake share.
"""
import logging
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.watcher.orchestrator as orch
from mousereach import shared_drive
from mousereach.config import Paths
from mousereach.watcher import node_status
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import DLCOrchestrator, ProcessingOrchestrator


def net_error(code=64, path="share"):
    return OSError(0, "The specified network name is no longer available", str(path),
                   code)


class DeadShare:
    """Makes every os.stat / os.listdir / os.scandir under ``root`` hang or raise
    while ``down``; ``mode`` is 'hang' or 'raise'. Hanging calls are released
    when the test ends, so no thread is left stuck."""

    def __init__(self, monkeypatch, root: Path, mode="raise", code=64):
        self.root = os.path.normcase(os.path.abspath(str(root)))
        self.mode, self.code = mode, code
        self.down = True
        self.release = threading.Event()
        self.calls = 0
        for name in ("stat", "listdir", "scandir"):
            real = getattr(os, name)
            monkeypatch.setattr(os, name, self._wrap(real))

    def _on_share(self, path) -> bool:
        try:
            p = os.path.normcase(os.path.abspath(os.fspath(path)))
        except TypeError:
            return False
        return p == self.root or p.startswith(self.root + os.sep)

    def _wrap(self, real):
        def fake(path=".", *a, **k):
            if self.down and self._on_share(path):
                self.calls += 1
                if self.mode == "hang":
                    self.release.wait(30)
                raise net_error(self.code, path)
            return real(path, *a, **k)
        return fake


@pytest.fixture
def share(tmp_path, monkeypatch):
    root = tmp_path / "share"
    root.mkdir()
    (root / "pipeline_versions.json").write_text(
        '{"versions": {"dlc_scorer": "DLC_resnet50_xJan1shuffle3_100000"}}')
    monkeypatch.setattr(Paths, "NAS_ROOT", root, raising=False)
    monkeypatch.setattr(shared_drive, "PROBE_TIMEOUT_S", 0.3)
    monkeypatch.setattr(shared_drive, "READ_TIMEOUT_S", 0.3)
    monkeypatch.setattr(shared_drive, "STATUS_WRITE_TIMEOUT_S", 0.3)
    # Never touch this machine's real drive mappings.
    monkeypatch.setattr(shared_drive, "maybe_remap", lambda *a, **k: None)
    shared_drive._forget_stuck_calls()
    yield root
    shared_drive._forget_stuck_calls()


def loop_node(tmp_path, monkeypatch, cls=DLCOrchestrator):
    """Enough orchestrator to run the real loop: its scan LISTS the shared
    drive (as the real scans do), and it counts what it was asked to do."""
    o = object.__new__(cls)
    o.config = SimpleNamespace(poll_interval_seconds=0.01)
    o.db = SimpleNamespace()
    o.hostname = "node"
    o.scans = 0
    o.statuses = []
    o.dispatched = []

    def scan():
        o.scans += 1
        os.listdir(str(Paths.NAS_ROOT))

    o._scan_phase = scan
    o._get_next_work_item = lambda: None
    o.shutdown = lambda: None
    o._reclaim_orphaned_work = lambda: {}
    o._maybe_review_reprocess_scan = lambda *a, **k: None
    o._maybe_send_recordings = lambda: None
    o._is_paused = lambda: False
    o._report_node_status = lambda state, reason=None: o.statuses.append((state, reason))
    o._start_stop_watchdog = lambda: None
    monkeypatch.setattr(orch, "require_processing_root", lambda: tmp_path)
    return o


def run_in_thread(o, stop):
    t = threading.Thread(target=o.run, args=(stop,), daemon=True)
    t.start()
    return t


# ----------------------------------------------------------------- the cycle

def test_a_cycle_returns_within_the_time_limit_when_the_share_hangs(share, tmp_path,
                                                                    monkeypatch):
    dead = DeadShare(monkeypatch, share, mode="hang")
    o = loop_node(tmp_path, monkeypatch)
    stop = threading.Event()
    try:
        t = run_in_thread(o, stop)
        time.sleep(1.5)                       # several turns at a 0.3 s probe limit
        stop.set()
        t.join(5)
        assert not t.is_alive(), "the loop hung inside a shared-drive call"
        assert o.scans == 0, "no shared-drive work may start while the drive is away"
        assert (node_status.WAITING_FOR_SHARE in [s for s, _ in o.statuses])
    finally:
        dead.release.set()


def test_the_stop_flag_is_honored_during_an_outage(share, tmp_path, monkeypatch):
    dead = DeadShare(monkeypatch, share, mode="hang")
    o = loop_node(tmp_path, monkeypatch)
    try:
        t = run_in_thread(o, threading.Event())       # nothing but the flag stops it
        time.sleep(0.8)
        (tmp_path / "watcher_stop.flag").write_text("stop")
        t.join(5)
        assert not t.is_alive(), "the stop flag was not read during the outage"
    finally:
        dead.release.set()


def test_the_stop_flag_ends_a_watcher_stuck_in_a_call_on_a_dead_drive(share, tmp_path,
                                                                     monkeypatch):
    """The drive dies DURING a scan: the call hangs, so the loop never comes
    round to read the flag. The watchdog ends the process instead."""
    dead = DeadShare(monkeypatch, share, mode="hang")
    dead.down = False                                 # fine at the top of the cycle
    o = loop_node(tmp_path, monkeypatch, cls=DLCOrchestrator)
    del o._start_stop_watchdog                        # the real watchdog
    exits = []

    def hard_exit(code):
        exits.append(code)
        dead.release.set()                            # let the stuck call finish
    monkeypatch.setattr(shared_drive, "_hard_exit", hard_exit)
    monkeypatch.setattr(shared_drive.StopWatchdog, "GRACE_S", 0.5)
    monkeypatch.setattr(shared_drive.StopWatchdog, "POLL_S", 0.05)

    def scan():
        dead.down = True                              # the drive goes, mid-scan
        os.listdir(str(Paths.NAS_ROOT))
    o._scan_phase = scan
    stop = threading.Event()
    try:
        t = run_in_thread(o, stop)
        time.sleep(0.3)
        (tmp_path / "watcher_stop.flag").write_text("stop")
        deadline = time.monotonic() + 8
        while not exits and time.monotonic() < deadline:
            time.sleep(0.05)
        assert exits == [0], "a stop request was not honored while stuck on the dead drive"
    finally:
        dead.release.set()
        stop.set()


def test_the_watchdog_leaves_a_running_pose_alone():
    clock = [1000.0]
    dog = shared_drive.StopWatchdog(
        stop_requested=lambda: True, last_progress=lambda: 0.0,
        posing_since=lambda: 990.0, share_ok=lambda: False,
        exit_fn=lambda code: None, clock=lambda: clock[0])
    assert dog.reason_to_exit() is None               # pose for 10 s: keep it
    clock[0] = 990.0 + dog.pose_protect_s + 1
    assert dog.reason_to_exit()                       # past its protection: exit


def test_the_watchdog_never_exits_while_the_drive_answers():
    dog = shared_drive.StopWatchdog(
        stop_requested=lambda: True, last_progress=lambda: 0.0,
        posing_since=lambda: None, share_ok=lambda: True,
        exit_fn=lambda code: None, clock=lambda: 10_000.0)
    assert dog.reason_to_exit() is None


def test_work_resumes_when_the_share_returns(share, tmp_path, monkeypatch, caplog):
    dead = DeadShare(monkeypatch, share, mode="raise", code=121)
    o = loop_node(tmp_path, monkeypatch)
    stop = threading.Event()
    scans_during_outage = []

    def scan():
        if dead.down:
            scans_during_outage.append(1)
        o.scans += 1
        os.listdir(str(Paths.NAS_ROOT))
        stop.set()                                    # one good cycle is enough
    o._scan_phase = scan
    with caplog.at_level(logging.INFO):
        t = run_in_thread(o, stop)
        time.sleep(0.5)
        dead.down = False
        t.join(5)
    assert not t.is_alive()
    assert scans_during_outage == []
    assert o.scans == 1, "work did not resume when the drive came back"
    text = caplog.text
    assert text.count("Shared drive unavailable since") == 1      # one line, not one per turn
    assert "Shared drive back after" in text


# ---------------------------------------------------------------- no failed marks

@pytest.fixture
def server(tmp_path, monkeypatch, share):
    o = object.__new__(ProcessingOrchestrator)
    o.db = WatcherDB(tmp_path / "w.db")
    o.db.failure_guard = o._refuse_failure_during_outage
    o.config = SimpleNamespace(poll_interval_seconds=0.01)
    o.hostname = "server"
    monkeypatch.setattr(orch, "require_processing_root", lambda: tmp_path)
    return o


def _row(db, vid, state):
    db.register_video(vid, source_path=str(Path(db.db_path).parent / f"{vid}.mp4"))
    db.force_state(vid, state, reason="test setup")


def test_a_share_error_out_of_a_handler_marks_nothing_failed(server, share, monkeypatch):
    vid = "20250101_CNT0101_P1"
    _row(server.db, vid, "archiving")
    DeadShare(monkeypatch, share, mode="raise", code=64)

    def archive(work):
        raise net_error(64, share / "Analyzed")
    server._archive_to_nas = archive
    assert server._dispatch_guarded({"type": "archive", "id": vid, "data": {}}) is False
    row = server.db.get_video(vid)
    assert row["state"] == "processed", "a filing in hand goes back, as at startup"
    assert not row.get("error_count")


def test_a_handler_that_marks_failed_itself_is_refused(server, share, monkeypatch):
    """Handlers mark failed on their own and carry on; the mark itself is refused."""
    vid = "20250101_CNT0101_P2"
    _row(server.db, vid, "processing")
    DeadShare(monkeypatch, share, mode="raise", code=64)

    def pipeline(work):
        try:
            server.db.mark_failed(vid, "segmentation could not read its input")
        except Exception:
            pass                                       # a handler that swallows it
        return False
    server._run_pipeline = pipeline
    server._dispatch_guarded({"type": "pipeline", "id": vid, "data": {}})
    row = server.db.get_video(vid)
    assert row["state"] == "processing"
    assert not row.get("error_count")


def test_unresolvable_is_not_written_while_the_folders_cannot_be_read(server, share,
                                                                      monkeypatch):
    vid = "20250101_CNT0101_P3"
    _row(server.db, vid, "processing")
    DeadShare(monkeypatch, share, mode="raise", code=64)
    with pytest.raises(shared_drive.SharedDriveUnavailable):
        server.db.mark_unresolvable(vid, "no file for it on this node")
    assert server.db.get_video(vid)["state"] == "processing"


def test_a_real_failure_with_the_drive_up_is_still_recorded(server, share):
    vid = "20250101_CNT0101_P4"
    _row(server.db, vid, "processing")
    server.db.mark_failed(vid, "reach detector raised ValueError")
    assert server.db.get_video(vid)["state"] == "failed"


def test_a_winerror_64_message_is_refused_even_if_the_drive_answers_again(server, share):
    vid = "20250101_CNT0101_P5"
    _row(server.db, vid, "processing")
    with pytest.raises(shared_drive.SharedDriveUnavailable):
        server.db.mark_failed(vid, "copy failed: [WinError 64] The specified network "
                                   "name is no longer available")
    assert server.db.get_video(vid)["state"] == "processing"


# ---------------------------------------------------------------- DLC keeps going

def test_dlc_does_not_stop_when_the_declaration_cannot_be_read(share, monkeypatch):
    from mousereach.dlc.core.batch import resolve_dlc_shuffle
    from mousereach.pipeline import versions
    monkeypatch.setattr(versions, "_LAST_GOOD", {})
    assert resolve_dlc_shuffle() == (3, "DLC_resnet50_xJan1shuffle3_100000")
    DeadShare(monkeypatch, share, mode="raise", code=64)
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(
        net_error(64, self)))
    # During the outage the copy last read is used: the node keeps posing.
    assert resolve_dlc_shuffle()[0] == 3


def test_with_no_earlier_copy_an_outage_is_a_wait_not_a_config_error(share, monkeypatch):
    from mousereach.dlc.core.batch import resolve_dlc_shuffle
    from mousereach.pipeline import versions
    monkeypatch.setattr(versions, "_LAST_GOOD", {})
    monkeypatch.setattr(versions, "_last_good_file", lambda: None)
    DeadShare(monkeypatch, share, mode="raise", code=64)
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(
        net_error(64, self)))
    with pytest.raises(shared_drive.SharedDriveUnavailable):
        resolve_dlc_shuffle()


def test_a_hanging_declaration_read_returns_within_its_limit(share, monkeypatch):
    from mousereach.pipeline import versions
    monkeypatch.setattr(versions, "_LAST_GOOD", {})
    versions.get_current_versions()                   # a good read first
    release = threading.Event()
    real = Path.read_text
    monkeypatch.setattr(Path, "read_text",
                        lambda self, *a, **k: (release.wait(30), real(self, *a, **k))[1])
    try:
        t0 = time.monotonic()
        got = versions.get_current_versions()
        assert time.monotonic() - t0 < 3
        assert got["versions"]["dlc_scorer"].endswith("shuffle3_100000")
    finally:
        release.set()


def test_the_work_order_in_use_is_kept_through_an_outage(share, monkeypatch):
    from mousereach.watcher import work_priority as wp
    (share / wp.LAB_PRIORITY_FILENAME).write_text('{"projects": ["PROJECT_A"]}')
    wp.invalidate_lab_policy()
    try:
        assert wp._cached_lab_raw(share, max_age_s=0) == {"projects": ["PROJECT_A"]}
        monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(
            net_error(121, self)))
        assert wp._cached_lab_raw(share, max_age_s=0) == {"projects": ["PROJECT_A"]}
    finally:
        wp.invalidate_lab_policy()


# ---------------------------------------------------------------- startup

def test_startup_waits_for_the_share_instead_of_crashing(share, tmp_path, monkeypatch):
    from mousereach.watcher import cli
    monkeypatch.setattr("mousereach.config.require_processing_root", lambda: tmp_path)
    dead = DeadShare(monkeypatch, share, mode="raise", code=64)
    slept = []

    def sleep(s):
        slept.append(s)
        if sum(slept) >= 40:
            dead.down = False                         # the drive comes back
    assert cli._wait_for_shared_drive_at_start(share, sleep=sleep) == "ok"
    assert slept, "it must have waited"


def test_startup_stops_waiting_when_asked(share, tmp_path, monkeypatch):
    from mousereach.watcher import cli
    monkeypatch.setattr("mousereach.config.require_processing_root", lambda: tmp_path)
    DeadShare(monkeypatch, share, mode="raise", code=64)

    def sleep(s):
        (tmp_path / "watcher_stop.flag").write_text("stop")
    assert cli._wait_for_shared_drive_at_start(share, sleep=sleep) == "stop"


def test_startup_still_refuses_a_folder_that_is_not_there(tmp_path, monkeypatch, share):
    from mousereach.watcher import cli
    monkeypatch.setattr("mousereach.config.require_processing_root", lambda: tmp_path)
    assert cli._wait_for_shared_drive_at_start(tmp_path / "no_such",
                                               sleep=lambda s: None) == "missing"


def test_the_root_check_does_not_crash_on_a_network_error(share, monkeypatch):
    from mousereach.watcher import cli
    # pathlib (3.10) holds its own reference to os.stat, so Path.exists is
    # replaced directly: it raises a network error instead of answering.
    monkeypatch.setattr(Path, "exists",
                        lambda self: (_ for _ in ()).throw(net_error(64, self)))
    problem = cli.check_nas_root(share, "config")
    assert problem and "could not be read" in problem


# ---------------------------------------------------------------- pieces

def test_one_stuck_call_per_purpose_never_a_pile():
    release = threading.Event()
    started = []

    def stuck():
        started.append(1)
        release.wait(30)
    try:
        with pytest.raises(shared_drive.ShareTimeout):
            shared_drive.call_bounded(stuck, timeout=0.1, what="t", key="pile test")
        with pytest.raises(shared_drive.ShareTimeout):
            shared_drive.call_bounded(stuck, timeout=0.1, what="t", key="pile test")
        assert len(started) == 1
    finally:
        release.set()
        shared_drive._forget_stuck_calls()


@pytest.mark.parametrize("code,down", [(64, True), (121, True), (53, True), (67, True),
                                       (5, False), (32, False), (2, False)])
def test_which_errors_mean_the_drive_is_away(code, down):
    assert shared_drive.is_share_down_error(net_error(code)) is down


def test_a_wrapped_network_error_is_still_recognised():
    try:
        try:
            raise net_error(64)
        except OSError as e:
            raise RuntimeError("stage failed") from e
    except RuntimeError as e:
        assert shared_drive.is_share_down_error(e)


def test_status_line_for_a_waiting_node(tmp_path):
    node_status.write(tmp_path, node_status.WAITING_FOR_SHARE,
                      "shared drive unavailable since 2026-11-19 03:00 -- waiting",
                      hostname="PC1", extra={"doing": None})
    line = node_status.describe(tmp_path)
    assert "waiting" in line and "shared drive unavailable since" in line


def test_a_vanished_letter_is_mapped_again_to_the_share_it_had(tmp_path):
    ran = []

    def run(cmd, **k):
        ran.append(cmd)
        return SimpleNamespace(returncode=0, stdout="The command completed successfully.",
                               stderr="")
    said = shared_drive.maybe_remap(
        "Y:\\pipeline", run=run, mapped=lambda d: (2250, None),
        remembered=lambda d: "\\\\server\\share", learned=lambda d: None,
        probe_fn=lambda p, *a, **k: (shared_drive.OK, ""), absent=lambda d: True)
    assert ran == [["net", "use", "Y:", "\\\\server\\share", "/persistent:yes"]]
    assert "mapped it again" in said and "done" in said


def test_a_letter_that_is_mapped_but_silent_is_left_to_windows():
    ran = []
    said = shared_drive.maybe_remap(
        "Y:\\pipeline", run=lambda *a, **k: ran.append(a), mapped=lambda d: (1201, "x"),
        remembered=lambda d: "\\\\server\\share", learned=lambda d: None,
        probe_fn=lambda p, *a, **k: (shared_drive.OK, ""))
    assert ran == [] and said is None


def test_a_local_disk_is_never_mapped_over():
    ran = []
    said = shared_drive.maybe_remap(
        "C:\\pipeline", run=lambda *a, **k: ran.append(a), mapped=lambda d: (2250, None),
        remembered=lambda d: "\\\\server\\share", learned=lambda d: None,
        probe_fn=lambda p, *a, **k: (shared_drive.OK, ""), absent=lambda d: False)
    assert ran == [] and said is None


def test_no_remap_to_a_share_that_does_not_answer_either():
    ran = []
    said = shared_drive.maybe_remap(
        "Y:\\pipeline", run=lambda *a, **k: ran.append(a), mapped=lambda d: (2250, None),
        remembered=lambda d: "\\\\server\\share", learned=lambda d: None,
        probe_fn=lambda p, *a, **k: (shared_drive.DOWN, "no answer"),
        absent=lambda d: True)
    assert ran == [] and "does not answer" in said
