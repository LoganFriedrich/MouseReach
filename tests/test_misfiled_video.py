"""A video put in the wrong pipeline folder is sent to the right one.

WHY: Processing/DLC_Complete is where POSED videos are handed over, and the
scan that reads it finds videos BY their pose file. So a bare mp4 dropped
there was invisible twice over -- no pose to find it by, and no database row
to notice it was missing. It sat forever, in a folder whose name reads like a
perfectly sensible place to put a video.

The rescue has to be narrow, because this folder is a LIVE handover point.
Staging writes the video first and renames the pose into place last, so a
video that has only just arrived legitimately has no pose yet: a naive rule
would pull it out from under the machine that is still staging it. Hence the
settle time, the half-written check, and the never-seen-before check, each of
which is pinned below.
"""
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.watcher.orchestrator as orch
from mousereach.watcher.orchestrator import ProcessingOrchestrator

VID = "20240101_ABC0101_P1"
POSE = "DLC_resnet101_MPSAOct27shuffle3_100000"


class FakeDB:
    def __init__(self, known=()):
        self._known = {v: {"video_id": v, "state": "archived"} for v in known}

    def get_video(self, video_id):
        return self._known.get(video_id)


@pytest.fixture
def server(tmp_path, monkeypatch):
    staging = tmp_path / "DLC_Complete"
    front_door = tmp_path / "Single_Animal"
    staging.mkdir()
    front_door.mkdir()
    monkeypatch.setattr(orch.Paths, "SINGLE_ANIMAL_OUTPUT", front_door)

    o = object.__new__(ProcessingOrchestrator)
    o.db = FakeDB()
    o.staging_dir = staging
    o.staging = staging
    o.front_door = front_door
    return o


def _settled(path: Path):
    """Age a file past the settle time, so it cannot be a live handover."""
    old = time.time() - (ProcessingOrchestrator._MISFILED_SETTLE_S + 600)
    os.utime(path, (old, old))
    return path


def _drop(server, name=f"{VID}.mp4", body=b"a video"):
    p = server.staging / name
    p.write_bytes(body)
    return p


# ---------------------------------------------------------------- the rescue

def test_a_video_with_no_pose_is_sent_to_the_front_door(server):
    _settled(_drop(server))
    assert server._rescue_misfiled_singles() == 1
    assert (server.front_door / f"{VID}.mp4").read_bytes() == b"a video"
    assert not (server.staging / f"{VID}.mp4").exists()


def test_it_lands_where_the_ordinary_path_will_find_it(server):
    """Single_Animal is the pre-pose front door, so nothing else is needed:
    the pose role's own scan takes it from there."""
    _settled(_drop(server))
    server._rescue_misfiled_singles()
    assert sorted(p.name for p in server.front_door.iterdir()) == [f"{VID}.mp4"]


# ---------------------------------------------------------------- the guards

def test_a_posed_video_is_an_ordinary_handover_and_is_left_alone(server):
    _settled(_drop(server))
    (server.staging / f"{VID}{POSE}.h5").write_bytes(b"pose")
    assert server._rescue_misfiled_singles() == 0
    assert (server.staging / f"{VID}.mp4").exists()


def test_a_video_still_being_written_beside_is_left_alone(server):
    _settled(_drop(server))
    (server.staging / f"{VID}{POSE}.h5.part").write_bytes(b"half a pose")
    assert server._rescue_misfiled_singles() == 0
    assert (server.staging / f"{VID}.mp4").exists()


def test_a_video_that_only_just_arrived_is_left_alone(server):
    """The one that protects a live handover: staging writes the video first
    and the pose last, so a fresh arrival has no pose yet and is not misfiled."""
    _drop(server)                      # deliberately NOT aged
    assert server._rescue_misfiled_singles() == 0
    assert (server.staging / f"{VID}.mp4").exists()


def test_a_video_the_database_already_knows_is_left_alone(server):
    """A leftover staging copy of an already-archived video must not be sent
    back to the front door, or it would be posed and processed all over again."""
    server.db = FakeDB(known=[VID])
    _settled(_drop(server))
    assert server._rescue_misfiled_singles() == 0
    assert (server.staging / f"{VID}.mp4").exists()


def test_deeplabcut_by_products_are_skipped(server):
    _settled(_drop(server, name=f"{VID}{POSE}_labeled.mp4", body=b"overlay"))
    assert server._rescue_misfiled_singles() == 0
    assert list(server.front_door.iterdir()) == []


def test_it_does_not_overwrite_something_already_at_the_front_door(server):
    _settled(_drop(server))
    (server.front_door / f"{VID}.mp4").write_bytes(b"the original")
    assert server._rescue_misfiled_singles() == 0
    assert (server.front_door / f"{VID}.mp4").read_bytes() == b"the original"


# ---------------------------------------------------------------- safety

def test_nothing_to_do_is_not_an_error(server):
    assert server._rescue_misfiled_singles() == 0


def test_a_missing_staging_folder_is_not_an_error(server, tmp_path):
    server.staging_dir = tmp_path / "gone"
    assert server._rescue_misfiled_singles() == 0


def test_no_front_door_configured_is_not_an_error(server, monkeypatch):
    monkeypatch.setattr(orch.Paths, "SINGLE_ANIMAL_OUTPUT", None)
    _settled(_drop(server))
    assert server._rescue_misfiled_singles() == 0
    assert (server.staging / f"{VID}.mp4").exists()


def test_several_at_once(server):
    for n in ("20240101_ABC0101_P1", "20240102_ABC0102_P1", "20240103_ABC0103_P1"):
        _settled(_drop(server, name=f"{n}.mp4"))
    # one of them is a genuine handover
    (server.staging / f"20240103_ABC0103_P1{POSE}.h5").write_bytes(b"pose")
    assert server._rescue_misfiled_singles() == 2
    assert sorted(p.name for p in server.front_door.iterdir()) == [
        "20240101_ABC0101_P1.mp4", "20240102_ABC0102_P1.mp4"]
