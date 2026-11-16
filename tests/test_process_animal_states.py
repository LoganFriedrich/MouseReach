"""``mousereach-watch-process-animal`` queues a single only from a state that can
go to 'dlc_queued', and never leaves a claim behind.

It used to skip a fixed list of states that missed triage, deep_review,
outdated, archiving and quarantined. For such a row it claimed the shared file
and then the move to 'dlc_queued' raised: the command stopped mid-list (or,
after a copy, left that copy in the DLC queue) with the video held in this
machine's claim folder. Now only no row, 'validated', 'failed' or
'unresolvable' is queued; anything else is skipped before any claim; a video a
node already holds in its claim folder is skipped; and whatever fails after
the claim gives the claim back and removes the queue copy this run made.
tmp_path only; Paths is patched.
"""
import sys

import pytest

from mousereach.watcher import single_claim
from mousereach.watcher.db import WatcherDB

ANIMAL = "ABC9901"
VID = f"20240101_{ANIMAL}_P1"
VID2 = f"20240102_{ANIMAL}_P1"
HOST = "NODE-T"


@pytest.fixture
def env(tmp_path, monkeypatch):
    from mousereach.config import Paths
    from mousereach.watcher import cli
    nas = tmp_path / "nas"
    work = tmp_path / "work"
    dirs = {
        "NAS_ROOT": nas, "NAS_DRIVE": nas,
        "ANALYZED_OUTPUT": nas / "Analyzed",
        "MULTI_ANIMAL_SOURCE": nas / "Unanalyzed" / "Multi-Animal",
        "SINGLE_ANIMAL_OUTPUT": nas / "Unanalyzed" / "Single_Animal",
        "DLC_STAGING": nas / "Processing" / "Posed",
        "PROCESSING_ROOT": work, "PROCESSING": work / "Processing",
        "DLC_QUEUE": work / "DLC_Queue",
    }
    for name, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Paths, name, d, raising=False)
    db = WatcherDB(db_path=tmp_path / "w.db")
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: tmp_path / "w.db")
    monkeypatch.setattr("socket.gethostname", lambda: HOST)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    monkeypatch.setattr(sys, "argv", ["mousereach-watch-process-animal", ANIMAL])
    return Paths, db, cli


def _row_in(db, vid, state):
    db.register_video(vid, "/nowhere")
    db.force_state(vid, state, reason="test")


def _shared(Paths, vid):
    p = Paths.SINGLE_ANIMAL_OUTPUT / f"{vid}.mp4"
    p.write_bytes(b"video")
    return p


def test_a_held_video_is_skipped_before_any_claim_and_the_list_goes_on(env, capsys):
    """A triage row whose copy is already in the DLC queue: the move to
    'dlc_queued' used to raise outside any try and end the command, with the
    video left in this machine's claim folder."""
    Paths, db, cli = env
    shared = _shared(Paths, VID)
    _shared(Paths, VID2)
    (Paths.DLC_QUEUE / f"{VID}.mp4").write_bytes(b"video")
    _row_in(db, VID, "triage")

    cli.main_process_animal()
    out = capsys.readouterr().out

    assert f"SKIP {VID} (already triage)" in out
    assert shared.is_file()                                   # not claimed
    assert not (single_claim.host_dir(HOST) / f"{VID}.mp4").exists()
    assert db.get_video(VID)["state"] == "triage"
    assert f"QUEUED {VID2}" in out                             # the list went on


def test_an_outdated_video_leaves_no_queue_copy_behind(env, capsys):
    Paths, db, cli = env
    shared = _shared(Paths, VID)
    _row_in(db, VID, "outdated")
    cli.main_process_animal()
    assert f"SKIP {VID} (already outdated)" in capsys.readouterr().out
    assert shared.is_file()
    assert not (Paths.DLC_QUEUE / f"{VID}.mp4").exists()


def test_a_video_another_node_holds_is_not_queued_again(env, capsys):
    """A second copy at the top of the folder while another node has the
    first in its claim folder."""
    Paths, db, cli = env
    shared = _shared(Paths, VID)
    other = single_claim.host_dir("NODE-OTHER")
    other.mkdir(parents=True)
    (other / f"{VID}.mp4").write_bytes(b"video")
    cli.main_process_animal()
    out = capsys.readouterr().out
    assert f"SKIP {VID}" in out and "second copy" in out
    assert shared.is_file()
    assert not (Paths.DLC_QUEUE / f"{VID}.mp4").exists()


def test_a_failure_after_the_copy_gives_the_claim_back_and_removes_the_copy(
        env, monkeypatch, capsys):
    Paths, db, cli = env
    shared = _shared(Paths, VID)
    real = WatcherDB.update_state

    def refuse_queue(self, vid, state, **kw):
        if state == "dlc_queued":
            raise ValueError("refused for the test")
        return real(self, vid, state, **kw)

    monkeypatch.setattr(WatcherDB, "update_state", refuse_queue)
    cli.main_process_animal()
    out = capsys.readouterr().out
    assert f"ERROR {VID}" in out
    assert shared.is_file()                                   # given back
    assert not (single_claim.host_dir(HOST) / f"{VID}.mp4").exists()
    assert not (Paths.DLC_QUEUE / f"{VID}.mp4").exists()      # this run's copy removed


@pytest.mark.parametrize("state", ["validated", "failed", "unresolvable"])
def test_waiting_failed_and_unresolvable_rows_are_queued(env, capsys, state):
    Paths, db, cli = env
    _shared(Paths, VID)
    _row_in(db, VID, state)
    cli.main_process_animal()
    assert f"QUEUED {VID}" in capsys.readouterr().out
    assert db.get_video(VID)["state"] == "dlc_queued"
