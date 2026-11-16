"""Two kinds of 'unresolvable' row come back by themselves once their file does.

* The processing server's intake parks a posed video whose files were not all
  in the posed folder when it looked (a GPU node still handing them over). The
  posed-folder scan used to skip every known video, so the row came back only
  through the re-pose adoption -- as a re-run of a filed video. Now the scan
  puts it back to 'dlc_complete' when the video and its pose are there.
* Startup recovery records a video another node had before its pose finished.
  Its reason used to be one the singles intake never re-drives, so a video the
  other node gave back to the shared folder came back here only at the next
  restart. Now the intake validates it again like any other waiting single.
tmp_path only; Paths is patched.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from mousereach.watcher import coordination, repose, single_claim
from mousereach.watcher import orchestrator as orch
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.state import WatcherStateManager

VID = "20240101_ABC9901_P1"


@pytest.fixture
def env(tmp_path, monkeypatch):
    from mousereach.config import Paths
    nas = tmp_path / "nas"
    work = tmp_path / "work"
    dirs = {
        "NAS_ROOT": nas, "NAS_DRIVE": nas,
        "ANALYZED_OUTPUT": nas / "Analyzed",
        "SINGLE_ANIMAL_OUTPUT": nas / "Unanalyzed" / "Single_Animal",
        "DLC_STAGING": nas / "Processing" / "Posed",
        "PROCESSING_ROOT": work, "PROCESSING": work / "Processing",
        "DLC_QUEUE": work / "DLC_Queue",
    }
    for name, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Paths, name, d, raising=False)
    monkeypatch.setattr(repose, "declared_scorer", lambda: "")
    db = WatcherDB(db_path=tmp_path / "w.db")
    state = WatcherStateManager(db, SimpleNamespace(stability_wait_seconds=0))
    return Paths, db, state


def _server(Paths, db):
    o = object.__new__(orch.ProcessingOrchestrator)
    o.db = db
    o.hostname = "SERVER"
    o.staging_dir = Paths.DLC_STAGING
    o.processing_dir = Paths.PROCESSING
    o._claim_video = lambda vid: True
    o.released = []
    o._release_claim = lambda vid: o.released.append(vid)
    return o


def test_a_posed_video_missed_by_intake_is_taken_in_again_when_its_files_arrive(env):
    Paths, db, state = env
    db.register_video(VID, str(Paths.DLC_STAGING / f"{VID}.mp4"))
    db.force_state(VID, "dlc_complete", reason="test")
    server = _server(Paths, db)

    # The intake looks before the GPU node has finished handing the files over.
    assert server._intake_from_staging({"id": VID, "data": db.get_video(VID)}) is False
    assert db.get_video(VID)["state"] == "unresolvable"
    assert server.released == [VID]                # the posed claim is let go

    # The files arrive; the next posed-folder scan takes the video in again.
    (Paths.DLC_STAGING / f"{VID}.mp4").write_bytes(b"v")
    (Paths.DLC_STAGING / f"{VID}DLC_resnet50_x.h5").write_bytes(b"p")
    assert repose.adopt_staged_reposes(db, Paths.DLC_STAGING, declared="any-model") == []
    assert state.discover_dlc_staged(Paths.DLC_STAGING) == [VID]
    row = db.get_video(VID)
    assert row["state"] == "dlc_complete"
    assert row["current_path"] == str(Paths.DLC_STAGING / f"{VID}.mp4")
    assert row["error_message"] is None


def test_another_unresolvable_reason_is_not_taken_in_by_the_posed_scan(env):
    Paths, db, state = env
    db.register_video(VID, db.NO_FILE_HERE)
    db.mark_unresolvable(VID, "no file on this node; NODE-OTHER has it at state 'archived'")
    (Paths.DLC_STAGING / f"{VID}.mp4").write_bytes(b"v")
    (Paths.DLC_STAGING / f"{VID}DLC_resnet50_x.h5").write_bytes(b"p")
    assert state.discover_dlc_staged(Paths.DLC_STAGING) == []
    assert db.get_video(VID)["state"] == "unresolvable"


def _recover(monkeypatch, db, remote_state):
    c = coordination.PipelineCoordinator.__new__(coordination.PipelineCoordinator)
    monkeypatch.setattr(c, "get_all_video_states", lambda: {
        VID: {"state": remote_state, "hostname": "NODE-OTHER",
              "source_path": None, "collage_id": None}}, raising=False)
    monkeypatch.setattr(c, "get_all_collage_states", lambda: {}, raising=False)
    c.recover_local_db(db, "NODE-HERE")


def test_a_video_another_node_gave_back_before_its_pose_is_taken_again(env, monkeypatch):
    Paths, db, state = env
    _recover(monkeypatch, db, "dlc_queued")
    assert db.get_video(VID)["state"] == "unresolvable"

    shared = Paths.SINGLE_ANIMAL_OUTPUT / f"{VID}.mp4"
    shared.write_bytes(b"v")                       # the other node gave it back
    state.discover_new_singles(Paths.SINGLE_ANIMAL_OUTPUT)    # first sighting
    assert state.discover_new_singles(Paths.SINGLE_ANIMAL_OUTPUT) == [VID]
    assert db.get_video(VID)["state"] == "validated"


def test_a_video_another_node_has_posed_is_not_redriven(env, monkeypatch):
    """Past the pose the video does not come back through the singles folder;
    a same-named file there is not it coming back."""
    Paths, db, state = env
    _recover(monkeypatch, db, "dlc_complete")
    (Paths.SINGLE_ANIMAL_OUTPUT / f"{VID}.mp4").write_bytes(b"v")
    state.discover_new_singles(Paths.SINGLE_ANIMAL_OUTPUT)
    assert state.discover_new_singles(Paths.SINGLE_ANIMAL_OUTPUT) == []
    assert db.get_video(VID)["state"] == "unresolvable"
