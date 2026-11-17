"""A GPU node poses only a file it holds; process-animal claims what it copies.

The pose step finds its video by name, and the search also looks at the top of
the shared singles folder. DLC writes its pose beside its input, so a
'dlc_queued' row with no copy in this node's DLC queue was posed IN PLACE in
the shared folder, unclaimed -- another node could take and pose the same
video. Now the pose step accepts only a copy in this node's DLC queue or in its
own claim folder (``<Single_Animal>/.inflight/<host>/``); an unclaimed copy in
the shared folder is handed back to the intake (which claims it properly), and
any other copy is refused. ``mousereach-watch-process-animal`` used to copy
singles into the DLC queue without claiming them; it now claims each first.
tmp_path only; Paths is patched, nothing touches a shared drive.
"""
import sys
from unittest import mock

import pytest

from mousereach.watcher import single_claim
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import DLCOrchestrator

VID = "20240101_ABC9901_P1"
HOST = "NODE-T"


@pytest.fixture
def env(tmp_path, monkeypatch):
    from mousereach.config import Paths
    nas = tmp_path / "nas"
    work = tmp_path / "work"
    dirs = {
        "NAS_ROOT": nas,
        "NAS_DRIVE": nas,
        "ANALYZED_OUTPUT": nas / "Analyzed",
        "MULTI_ANIMAL_SOURCE": nas / "Unanalyzed" / "Multi-Animal",
        "SINGLE_ANIMAL_OUTPUT": nas / "Unanalyzed" / "Single_Animal",
        "DLC_STAGING": nas / "Processing" / "Posed",
        "TRIAGE_REVIEW": nas / "Processing" / "Review" / "triage",
        "DEEP_REVIEW": nas / "Processing" / "Review" / "deep_review",
        "PROCESSING_ROOT": work,
        "PROCESSING": work / "Processing",
        "DLC_QUEUE": work / "DLC_Queue",
    }
    for name, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Paths, name, d, raising=False)
    return Paths, tmp_path


def _node(tmp_path):
    o = DLCOrchestrator.__new__(DLCOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.hostname = HOST
    o._already_done_or_held = lambda vid: None
    return o


def _queued_row(db, path):
    db.register_video(video_id=VID, source_path=str(path), current_path=str(path))
    for st in ("validated", "dlc_queued"):
        db.update_state(VID, st)


def test_the_queue_and_own_claim_are_poseable(env):
    Paths, tmp_path = env
    o = _node(tmp_path)
    assert o._not_held_for_pose(Paths.DLC_QUEUE / f"{VID}.mp4") is None
    own = single_claim.host_dir(HOST) / f"{VID}.mp4"
    assert o._not_held_for_pose(own) is None


def test_another_nodes_claim_and_other_folders_are_not(env):
    Paths, tmp_path = env
    o = _node(tmp_path)
    other = single_claim.host_dir("NODE-OTHER") / f"{VID}.mp4"
    assert "claimed by NODE-OTHER" in o._not_held_for_pose(other)
    assert o._not_held_for_pose(Paths.ANALYZED_OUTPUT / f"{VID}.mp4")


def test_an_unclaimed_shared_copy_is_not_posed_in_place(env):
    Paths, tmp_path = env
    o = _node(tmp_path)
    shared = Paths.SINGLE_ANIMAL_OUTPUT / f"{VID}.mp4"
    shared.write_bytes(b"video")
    _queued_row(o.db, shared)
    with mock.patch("mousereach.dlc.core.run_dlc_batch") as dlc:
        assert o._process_single_dlc({"id": VID, "data": o.db.get_video(VID)}) is False
    dlc.assert_not_called()
    row = o.db.get_video(VID)
    assert row["state"] == "unresolvable"
    # Handed back to the intake, which re-drives exactly these reasons.
    assert row["error_message"].startswith(single_claim.REDRIVE_REASONS)
    assert shared.is_file()
    assert not any(Paths.SINGLE_ANIMAL_OUTPUT.glob(f"{VID}DLC*"))


def test_process_animal_claims_what_it_copies(env, monkeypatch, capsys):
    Paths, tmp_path = env
    from mousereach.watcher import cli
    shared = Paths.SINGLE_ANIMAL_OUTPUT / f"{VID}.mp4"
    shared.write_bytes(b"video")
    WatcherDB(db_path=tmp_path / "w.db")     # the watcher created it (process-animal never does)
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: tmp_path / "w.db")
    monkeypatch.setattr("socket.gethostname", lambda: HOST)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    monkeypatch.setattr(sys, "argv", ["mousereach-watch-process-animal", "ABC9901"])
    cli.main_process_animal()
    out = capsys.readouterr().out
    assert f"QUEUED {VID}" in out
    claimed = single_claim.host_dir(HOST) / f"{VID}.mp4"
    assert claimed.is_file() and not shared.exists()       # claimed, not left open
    assert (Paths.DLC_QUEUE / f"{VID}.mp4").is_file()
    row = WatcherDB(db_path=tmp_path / "w.db").get_video(VID)
    assert row["state"] == "dlc_queued"


def test_process_animal_skips_a_single_it_cannot_claim(env, monkeypatch, capsys):
    Paths, tmp_path = env
    from mousereach.watcher import cli
    (Paths.SINGLE_ANIMAL_OUTPUT / f"{VID}.mp4").write_bytes(b"video")
    WatcherDB(db_path=tmp_path / "w.db")     # the watcher created it (process-animal never does)
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: tmp_path / "w.db")
    monkeypatch.setattr("socket.gethostname", lambda: HOST)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    monkeypatch.setattr(single_claim, "claim_single", lambda src, host: None)
    monkeypatch.setattr(sys, "argv", ["mousereach-watch-process-animal", "ABC9901"])
    cli.main_process_animal()
    assert "could not claim" in capsys.readouterr().out
    assert not (Paths.DLC_QUEUE / f"{VID}.mp4").exists()
