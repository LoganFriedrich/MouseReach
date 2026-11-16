"""``mousereach-watch-unresolvable --retry`` queues only this GPU node's own copy.

It used to search everywhere (the posed folder and the filed folder included)
and force 'dlc_queued' on any machine: on a processing server no work step
selects that state, so the row was stranded; on a GPU node a posed or filed
copy is refused for pose and parked again. Now a processing server changes
nothing; a GPU node queues a copy in its own DLC queue, leaves a copy at the
top of the shared singles folder to the intake (shared-folder reason), and
lists any other copy without changing anything. tmp_path only.
"""
import sys
from types import SimpleNamespace

import pytest

from mousereach.watcher import single_claim
from mousereach.watcher.db import WatcherDB

VID = "20240101_ABC9901_P1"


@pytest.fixture
def env(tmp_path, monkeypatch):
    from mousereach import config
    from mousereach.config import Paths
    from mousereach.watcher import cli
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
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(VID, db.NO_FILE_HERE)
    db.mark_unresolvable(VID, "no file on this node; NODE-OTHER has it at state 'dlc_running'")
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: tmp_path / "w.db")
    monkeypatch.setattr(sys, "argv", ["mousereach-watch-unresolvable", "--retry"])

    def role(mode):
        monkeypatch.setattr(config.WatcherConfig, "load",
                            classmethod(lambda cls: SimpleNamespace(mode=mode)))
    return Paths, db, cli, role


def test_a_processing_server_changes_nothing(env, capsys):
    Paths, db, cli, role = env
    role("processing_server")
    (Paths.DLC_STAGING / f"{VID}.mp4").write_bytes(b"v")
    cli.main_unresolvable()
    assert "changes nothing here" in capsys.readouterr().out
    assert db.get_video(VID)["state"] == "unresolvable"


def test_a_copy_in_this_nodes_queue_is_queued(env):
    Paths, db, cli, role = env
    role("dlc_pc")
    local = Paths.DLC_QUEUE / f"{VID}.mp4"
    local.write_bytes(b"v")
    cli.main_unresolvable()
    row = db.get_video(VID)
    assert row["state"] == "dlc_queued" and row["current_path"] == str(local)


def test_a_copy_in_the_shared_folder_is_left_for_the_intake(env):
    Paths, db, cli, role = env
    role("dlc_pc")
    (Paths.SINGLE_ANIMAL_OUTPUT / f"{VID}.mp4").write_bytes(b"v")
    cli.main_unresolvable()
    row = db.get_video(VID)
    assert row["state"] == "unresolvable"
    assert row["error_message"].startswith(single_claim.SHARED_REASON)


def test_a_second_copy_keeps_its_reason(env):
    Paths, db, cli, role = env
    role("dlc_pc")
    reason = f"{single_claim.DUPLICATE_OF_CLAIM_REASON}: NODE-OTHER is posing it"
    db.mark_unresolvable(VID, reason)
    (Paths.SINGLE_ANIMAL_OUTPUT / f"{VID}.mp4").write_bytes(b"v")
    cli.main_unresolvable()
    assert db.get_video(VID)["error_message"] == reason


def test_a_posed_copy_is_listed_and_left_alone(env, capsys):
    Paths, db, cli, role = env
    role("dlc_pc")
    (Paths.DLC_STAGING / f"{VID}.mp4").write_bytes(b"v")
    cli.main_unresolvable()
    assert "left as it is" in capsys.readouterr().out
    assert db.get_video(VID)["state"] == "unresolvable"
