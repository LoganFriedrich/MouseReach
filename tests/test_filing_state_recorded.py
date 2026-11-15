"""Filing is recorded before the files move, and a filed run is never 'failed'.

Filing MOVES the files, and used to write the row's state only afterwards
(processed -> archiving -> archived, back to back). A watcher that died between
the move and those writes left the row 'processed' with the files already
filed; the next pass found nothing in Processing and marked the filed video
'failed'. Now:

  * 'archiving' is written before anything moves;
  * a failed attempt that leaves the files here goes back to 'processed', so
    the archive bucket picks it again after its backoff;
  * "nothing left in Processing" with THIS run's results in the filed folder
    is recorded 'archived', not 'failed'; nothing of this run filed is still
    'failed', for a person.

Built under tmp_path; no real folder or database.
"""
import os
import time
from unittest import mock

import pytest

from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB, VIDEO_TRANSITIONS
from mousereach.watcher.orchestrator import (
    DLCOrchestrator, ProcessingOrchestrator, filed_this_run,
)

VID = "20990101_ABC9901_P1"


def _row(db, vid, upto):
    db.register_video(video_id=vid, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete",
               "processing", "processed", "archiving"):
        db.update_state(vid, st)
        if st == upto:
            return


@pytest.fixture
def env(tmp_path, monkeypatch):
    proc = tmp_path / "Processing"
    proc.mkdir()
    filed = tmp_path / "Analyzed" / "PROJ" / "COHORT"
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    monkeypatch.setattr(Paths, "PROCESSING", proc)
    monkeypatch.setattr("mousereach.archive.core.get_archive_destination",
                        lambda vid: filed)

    class _NoIndex:
        def load(self): pass
        def remove_video(self, _): pass
        def save(self): pass
    monkeypatch.setattr("mousereach.index.PipelineIndex", _NoIndex, raising=False)

    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.db.export_to_central_db = lambda vid: None
    o.processing_dir = proc
    o.staging_dir = None
    o.hostname = "NODE-T"
    o._release_claim = lambda vid: None
    return o, proc, filed


def _finished_run(proc, vid=VID):
    for name in (f"{vid}.mp4", f"{vid}_segments.json", f"{vid}_features.json"):
        (proc / name).write_text("x")


def test_archiving_is_recorded_before_anything_moves(env):
    o, proc, filed = env
    _row(o.db, VID, "processed")
    _finished_run(proc)
    seen = {}

    def fake_archive(video_id, **kw):
        seen["state"] = o.db.get_video(video_id)["state"]
        return {"success": True, "files_moved": ["a"], "destination": str(filed)}

    with mock.patch("mousereach.archive.core.archive_video", fake_archive):
        assert ProcessingOrchestrator._archive_to_nas(o, {"id": VID}) is True
    assert seen["state"] == "archiving"
    assert o.db.get_video(VID)["state"] == "archived"


def test_a_failed_attempt_goes_back_to_processed(env):
    o, proc, _ = env
    _row(o.db, VID, "processed")
    _finished_run(proc)
    with mock.patch("mousereach.archive.core.archive_video",
                    return_value={"success": False, "error": "failed to move: x"}):
        assert ProcessingOrchestrator._archive_to_nas(o, {"id": VID}) is False
    assert o.db.get_video(VID)["state"] == "processed"


def test_an_exception_goes_back_to_processed(env):
    o, proc, _ = env
    _row(o.db, VID, "processed")
    _finished_run(proc)
    with mock.patch("mousereach.archive.core.archive_video",
                    side_effect=OSError("share went away")):
        assert ProcessingOrchestrator._archive_to_nas(o, {"id": VID}) is False
    assert o.db.get_video(VID)["state"] == "processed"


def test_a_run_filed_before_its_state_was_written_is_recorded_archived(env):
    """The crash case: the files moved, the watcher died, startup reclaim put
    the row back to 'processed'. The real archive step then finds nothing in
    Processing -- and this run's results in the filed folder."""
    o, proc, filed = env
    _row(o.db, VID, "processed")
    filed.mkdir(parents=True)
    _finished_run(filed)                       # moved there by the dead attempt
    assert ProcessingOrchestrator._archive_to_nas(o, {"id": VID}) is True
    row = o.db.get_video(VID)
    assert row["state"] == "archived"
    assert not row["error_message"]


def test_nothing_here_and_nothing_of_this_run_filed_is_still_failed(env):
    o, proc, filed = env
    _row(o.db, VID, "processed")
    filed.mkdir(parents=True)
    _finished_run(filed)
    old = time.time() - 7 * 86400              # an earlier generation, filed last week
    for f in filed.iterdir():
        os.utime(f, (old, old))
    assert ProcessingOrchestrator._archive_to_nas(o, {"id": VID}) is False
    assert o.db.get_video(VID)["state"] == "failed"


def test_filed_this_run_needs_the_video_beside_the_features(tmp_path):
    from datetime import datetime, timedelta
    started = (datetime.now() - timedelta(minutes=5)).isoformat()
    (tmp_path / f"{VID}_features.json").write_text("{}")
    assert filed_this_run(VID, started, dest=tmp_path) is None
    (tmp_path / f"{VID}.mp4").write_text("v")
    assert filed_this_run(VID, started, dest=tmp_path) == tmp_path
    assert filed_this_run(VID, None, dest=tmp_path) is None


def test_the_local_role_records_a_filed_run_too(tmp_path, monkeypatch):
    proc = tmp_path / "Processing"
    proc.mkdir()
    filed = tmp_path / "Analyzed" / "PROJ" / "COHORT"
    filed.mkdir(parents=True)
    monkeypatch.setattr(Paths, "PROCESSING", proc)
    monkeypatch.setattr(Paths, "DLC_QUEUE", tmp_path / "DLC_Queue")
    monkeypatch.setattr("mousereach.archive.core.get_archive_destination",
                        lambda vid: filed)
    o = DLCOrchestrator.__new__(DLCOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.db.export_to_central_db = lambda vid: None
    o.hostname = "NODE-T"
    o._sync_to_connectome = lambda *a, **k: None
    o._retire_claimed_single = lambda *a, **k: True
    monkeypatch.setattr("mousereach.watcher.repose.close_request",
                        lambda *a, **k: False)
    _row(o.db, VID, "processed")
    _finished_run(filed)
    with mock.patch("mousereach.archive.core.archive_video",
                    return_value={"success": False,
                                  "error": "No files found in Processing/"}):
        assert o._archive_locally_processed({"id": VID}) is True
    assert o.db.get_video(VID)["state"] == "archived"


def test_the_local_role_goes_back_to_processed_on_failure(tmp_path, monkeypatch):
    proc = tmp_path / "Processing"
    proc.mkdir()
    monkeypatch.setattr(Paths, "PROCESSING", proc)
    monkeypatch.setattr(Paths, "DLC_QUEUE", tmp_path / "DLC_Queue")
    o = DLCOrchestrator.__new__(DLCOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.hostname = "NODE-T"
    _row(o.db, VID, "processed")
    _finished_run(proc)
    with mock.patch("mousereach.archive.core.archive_video",
                    return_value={"success": False, "error": "failed to move: x"}):
        assert o._archive_locally_processed({"id": VID}) is False
    assert o.db.get_video(VID)["state"] == "processed"


def test_archiving_may_return_to_processed():
    assert "processed" in VIDEO_TRANSITIONS["archiving"]
