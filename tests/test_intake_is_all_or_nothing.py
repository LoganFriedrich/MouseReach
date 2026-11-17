"""Intake takes in all of a video's files or none.

WHY: intake went on when any one file copied. A run without its video then
analysed it with the outcome detector's video checks silently off, routed
review bundles a person could not open, and after filing the clean-up deleted
the posed copy -- the video was then in no place but its recording.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from mousereach.watcher import orchestrator as o
from mousereach.watcher import repose
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.state import WatcherStateManager

V = "20990101_ZZZ0101_P5"
POSE = "DLC_resnet50_TestModelshuffle1_100"


def test_a_video_that_did_not_copy_stops_the_intake(tmp_path, monkeypatch):
    posed, proc = tmp_path / "Posed", tmp_path / "Processing"
    posed.mkdir()
    (posed / f"{V}.mp4").write_bytes(b"video")
    (posed / f"{V}{POSE}.h5").write_bytes(b"pose")
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=V, source_path=str(posed / f"{V}.mp4"))
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete"):
        db.update_state(V, st)
    db.update_state(V, "dlc_complete", current_path=str(posed / f"{V}.mp4"))

    real = o.safe_copy

    def copy(src, dst, verify=True):
        if str(src).endswith(".mp4"):
            return False                 # the share dropped mid-copy
        return real(src, dst, verify=verify)

    monkeypatch.setattr(o, "safe_copy", copy)
    server = object.__new__(o.ProcessingOrchestrator)
    server.db = db
    server.processing_dir = proc
    server.staging_dir = posed
    server.config = SimpleNamespace(max_retries=3)
    server._claim_video = lambda video_id: True

    server._release_claim = lambda video_id: None
    assert server._intake_from_staging({"id": V, "data": db.get_video(V)}) is False

    row = db.get_video(V)
    # Not 'failed' (a copy problem is not a verdict on the video): parked so
    # the posed-folder scan puts it back in line for intake.
    from mousereach.watcher.db import INTAKE_MISS_REASON
    assert row["state"] == "unresolvable"
    assert row["error_message"].startswith(INTAKE_MISS_REASON)
    assert not proc.exists() or not any(proc.iterdir()), \
        "a partial set was left in Processing"
    assert (posed / f"{V}.mp4").exists() and (posed / f"{V}{POSE}.h5").exists()


def test_a_copy_that_keeps_failing_is_retried_after_a_wait_and_then_held_failed(
        tmp_path, monkeypatch):
    """The files are all in the posed folder but one will not copy (a full
    disk, say). The video must not be copied and thrown away again on every
    pass: the posed-folder scan waits before bringing it back, and after
    max_retries attempts the row is 'failed' with what to do -- and nothing
    picks it up again by itself."""
    posed, proc = tmp_path / "Posed", tmp_path / "Processing"
    posed.mkdir()
    (posed / f"{V}.mp4").write_bytes(b"video")
    (posed / f"{V}{POSE}.h5").write_bytes(b"pose")
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=V, source_path=str(posed / f"{V}.mp4"))
    db.force_state(V, "dlc_complete", reason="test",
                   current_path=str(posed / f"{V}.mp4"))

    copies = []
    real = o.safe_copy

    def copy(src, dst, verify=True):
        copies.append(src)
        if str(src).endswith(".mp4"):
            return False                 # the disk is full, every time
        return real(src, dst, verify=verify)

    monkeypatch.setattr(o, "safe_copy", copy)
    monkeypatch.setattr(repose, "declared_scorer", lambda: "")
    server = object.__new__(o.ProcessingOrchestrator)
    server.db = db
    server.processing_dir = proc
    server.staging_dir = posed
    server.config = SimpleNamespace(max_retries=3)
    server._claim_video = lambda video_id: True
    server._release_claim = lambda video_id: None
    scan = WatcherStateManager(db, SimpleNamespace(stability_wait_seconds=0,
                                                   max_retries=3))

    def wait_over():
        earlier = (datetime.now() - timedelta(hours=2)).isoformat()
        db.set_fields(V, last_error_at=earlier)

    for attempt in (1, 2):
        assert server._intake_from_staging({"id": V, "data": db.get_video(V)}) is False
        row = db.get_video(V)
        assert row["state"] == "unresolvable"
        assert f"attempt {attempt} of 3" in row["error_message"]
        # Straight after the failure the scan does NOT bring it back.
        assert scan.discover_dlc_staged(posed) == []
        assert db.get_video(V)["state"] == "unresolvable"
        wait_over()
        assert scan.discover_dlc_staged(posed) == [V]
        assert db.get_video(V)["state"] == "dlc_complete"

    # The last attempt: held as failed, with the command that runs it again.
    assert server._intake_from_staging({"id": V, "data": db.get_video(V)}) is False
    row = db.get_video(V)
    assert row["state"] == "failed"
    assert "no more automatic tries" in row["error_message"]
    assert "mousereach-watch-reprocess --from-step dlc_complete" in row["error_message"]
    wait_over()
    assert scan.discover_dlc_staged(posed) == []
    assert db.get_video(V)["state"] == "failed"
    # The failed-video retry does not loop it either: its files are not here.
    server.RETRY_FAILED_AFTER_S = 0
    assert server._failed_retry_due(db.get_video(V)) is False
    assert not proc.exists() or not any(proc.iterdir())

    # The cause is fixed and a person runs the command the row names
    # (mousereach-watch-reprocess --from-step dlc_complete <video>, which is
    # db.reset_failed): the next intake takes the video in, with its full
    # retry budget back.
    monkeypatch.setattr(o, "safe_copy", real)
    assert db.reset_failed(V, "dlc_complete") == "dlc_complete"
    assert db.get_video(V)["error_count"] == 0
    assert server._intake_from_staging({"id": V, "data": db.get_video(V)}) is not False
    assert db.get_video(V)["state"] == "processing"
    assert (proc / f"{V}.mp4").is_file() and (proc / f"{V}{POSE}.h5").is_file()


def test_a_video_none_of_whose_files_copied_is_parked_and_counted_not_failed(
        tmp_path, monkeypatch):
    """Every copy fails (a full disk fails them all). This is the same
    infrastructure fact as one failed copy, so it takes the same path: parked
    as unresolvable with the attempt counted, brought back by the posed-folder
    scan only after the wait, and 'failed' only at max_retries -- never
    'failed' on the first attempt."""
    posed, proc = tmp_path / "Posed", tmp_path / "Processing"
    posed.mkdir()
    (posed / f"{V}.mp4").write_bytes(b"video")
    (posed / f"{V}{POSE}.h5").write_bytes(b"pose")
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=V, source_path=str(posed / f"{V}.mp4"))
    db.force_state(V, "dlc_complete", reason="test",
                   current_path=str(posed / f"{V}.mp4"))

    monkeypatch.setattr(o, "safe_copy", lambda src, dst, verify=True: False)
    monkeypatch.setattr(repose, "declared_scorer", lambda: "")
    server = object.__new__(o.ProcessingOrchestrator)
    server.db = db
    server.processing_dir = proc
    server.staging_dir = posed
    server.config = SimpleNamespace(max_retries=2)
    server._claim_video = lambda video_id: True
    released = []
    server._release_claim = lambda video_id: released.append(video_id)
    scan = WatcherStateManager(db, SimpleNamespace(stability_wait_seconds=0,
                                                   max_retries=2))

    assert server._intake_from_staging({"id": V, "data": db.get_video(V)}) is False
    row = db.get_video(V)
    assert row["state"] == "unresolvable"
    from mousereach.watcher.db import INTAKE_COPY_FAILED, INTAKE_MISS_REASON
    assert row["error_message"].startswith(INTAKE_MISS_REASON)
    assert INTAKE_COPY_FAILED in row["error_message"]
    assert "attempt 1 of 2" in row["error_message"]
    assert row["error_count"] == 1 and row["last_error_at"]
    assert released == [V]
    assert scan.discover_dlc_staged(posed) == []
    db.set_fields(V, last_error_at=(datetime.now() - timedelta(hours=2)).isoformat())
    assert scan.discover_dlc_staged(posed) == [V]

    # The last allowed attempt hands it over to 'failed', with what to do.
    assert server._intake_from_staging({"id": V, "data": db.get_video(V)}) is False
    row = db.get_video(V)
    assert row["state"] == "failed"
    assert "no more automatic tries" in row["error_message"]
    assert not proc.exists() or not any(proc.iterdir())
    assert (posed / f"{V}.mp4").exists() and (posed / f"{V}{POSE}.h5").exists()
