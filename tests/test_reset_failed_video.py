"""Resetting a failed video (``mousereach-watch-reprocess``, WatcherDB.reset_failed).

* With no --from-step the restart point is read from the files this node
  holds, as the automatic retries read it -- never 'discovered' (not a legal
  state after 'failed', so the reset raised) and never 'validated'.
* A reset gives the video its automatic retries back (error_count -> 0).
* Only a 'failed' row is changed, and the command never says "Reset" for a
  video it did not change.
tmp_path only; Paths is patched.
"""
import sys

import pytest

from mousereach.watcher import cli
from mousereach.watcher.db import WatcherDB

VID = "20990101_ZZZ0101_P1"
VID2 = "20990101_ZZZ0102_P1"


@pytest.fixture
def env(tmp_path, monkeypatch):
    from mousereach.config import Paths
    work = tmp_path / "work"
    dirs = {"PROCESSING_ROOT": work, "PROCESSING": work / "Processing",
            "DLC_QUEUE": work / "DLC_Queue"}
    for name, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Paths, name, d, raising=False)
    db = WatcherDB(tmp_path / "watcher.db")
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: db.db_path)
    return Paths, db


def _failed_twice(db, vid=VID, validated=True):
    """A pose that failed twice on this node. ``validated=False``: a row that
    reached 'failed' without a validation time (registered from disk)."""
    db.register_video(vid, "/nowhere")
    if validated:
        db.update_state(vid, "validated")
        db.update_state(vid, "dlc_queued")
    else:
        db.force_state(vid, "dlc_queued", reason="test")
    db.update_state(vid, "dlc_running")
    db.mark_failed(vid, "first")
    db.force_state(vid, "dlc_running", reason="test")
    db.mark_failed(vid, "second")
    assert db.get_video(vid)["error_count"] == 2


def test_a_reset_reads_the_restart_point_from_this_nodes_files(env):
    Paths, db = env
    _failed_twice(db, validated=False)
    (Paths.DLC_QUEUE / f"{VID}.mp4").write_bytes(b"v")

    assert db.reset_failed(VID) == "dlc_queued"
    row = db.get_video(VID)
    assert row["state"] == "dlc_queued"
    assert row["current_path"] == str(Paths.DLC_QUEUE / f"{VID}.mp4")


def test_a_validated_row_is_not_sent_back_to_the_shared_folder(env):
    """A row with a validation time was reset to 'validated' -- "waiting in
    the shared singles folder" -- although its file sits in this node's queue."""
    Paths, db = env
    _failed_twice(db, validated=True)
    (Paths.DLC_QUEUE / f"{VID}.mp4").write_bytes(b"v")
    assert db.reset_failed(VID) == "dlc_queued"


def test_video_and_pose_in_processing_restart_at_processing(env):
    Paths, db = env
    _failed_twice(db)
    (Paths.PROCESSING / f"{VID}.mp4").write_bytes(b"v")
    (Paths.PROCESSING / f"{VID}DLC_resnet50_x.h5").write_bytes(b"p")
    assert db.reset_failed(VID) == "processing"


def test_no_files_here_names_the_choices_and_changes_nothing(env):
    Paths, db = env
    _failed_twice(db, validated=False)
    with pytest.raises(ValueError, match="--from-step"):
        db.reset_failed(VID)
    assert db.get_video(VID)["state"] == "failed"


def test_a_reset_gives_the_retries_back_and_logs_the_old_count(env):
    Paths, db = env
    _failed_twice(db)
    db.reset_failed(VID, to_state="dlc_queued")
    assert db.get_video(VID)["error_count"] == 0
    log = [r for r in db.get_recent_log(20) if r.get("step") == "reset"]
    assert log and "error count was 2" in (log[0].get("message") or "")


def test_the_command_does_not_report_a_reset_it_did_not_make(env, monkeypatch, capsys):
    Paths, db = env
    db.register_video(VID, "/nowhere")
    db.update_state(VID, "validated")
    db.update_state(VID, "dlc_queued")
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    monkeypatch.setattr(sys, "argv", ["mousereach-watch-reprocess", VID])
    with pytest.raises(SystemExit) as exit_:
        cli.main_reprocess()
    out = capsys.readouterr().out
    assert exit_.value.code == 1
    assert f"{VID} not changed: it is 'dlc_queued', not failed." in out
    assert "Reset:" not in out
    assert db.get_video(VID)["state"] == "dlc_queued"


def test_all_failed_counts_only_the_rows_really_reset(env, monkeypatch, capsys):
    """A row that leaves 'failed' between the listing and its reset (the
    watcher retries on its own) is not counted as reset."""
    Paths, db = env
    _failed_twice(db, VID)
    _failed_twice(db, VID2)
    listed = db.get_videos_in_state("failed")
    db.force_state(VID2, "processing", reason="the watcher retried it meanwhile")
    monkeypatch.setattr(db.__class__, "get_videos_in_state",
                        lambda self, state: listed)
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    monkeypatch.setattr(sys, "argv", ["mousereach-watch-reprocess", "--all-failed",
                                      "--from-step", "dlc_queued"])
    cli.main_reprocess()
    out = capsys.readouterr().out
    assert "Reset 1 of 2 videos." in out
    assert f"Reset: {VID2}" not in out
    assert db.get_video(VID2)["state"] == "processing"


def test_the_word_after_from_step_is_the_step_not_the_video(env, monkeypatch, capsys):
    Paths, db = env
    _failed_twice(db)
    monkeypatch.setattr(sys, "argv", ["mousereach-watch-reprocess", "--from-step",
                                      "dlc_queued", VID])
    cli.main_reprocess()
    assert db.get_video(VID)["state"] == "dlc_queued"


def test_a_single_whose_dlc_queue_copy_failed_resets_to_the_shared_folder(env, tmp_path, monkeypatch):
    """The copy-into-the-DLC-queue failure gives the single back to the shared
    singles folder and tells a person to run mousereach-watch-reprocess. That
    reset finds the file there and puts the row back to 'validated'."""
    from mousereach.watcher import single_claim
    Paths, db = env
    door = tmp_path / "Single_Animal"
    door.mkdir()
    monkeypatch.setattr(Paths, "SINGLE_ANIMAL_OUTPUT", door, raising=False)
    (door / f"{VID}.mp4").write_bytes(b"v")
    db.register_video(VID, str(door / f"{VID}.mp4"))
    db.update_state(VID, "validated")
    db.mark_failed(VID, f"{single_claim.COPY_FAILED_REASON}: could not copy x")

    assert db.reset_failed(VID) == "validated"
    row = db.get_video(VID)
    assert row["state"] == "validated"
    assert row["source_path"] == str(door / f"{VID}.mp4")
    assert row["current_path"] == str(door / f"{VID}.mp4")
    assert row["error_count"] == 0


def test_an_older_copy_failure_message_resets_to_the_shared_folder_too(env, tmp_path, monkeypatch):
    Paths, db = env
    door = tmp_path / "Single_Animal"
    door.mkdir()
    monkeypatch.setattr(Paths, "SINGLE_ANIMAL_OUTPUT", door, raising=False)
    (door / f"{VID}.mp4").write_bytes(b"v")
    db.register_video(VID, str(door / f"{VID}.mp4"))
    db.update_state(VID, "validated")
    db.mark_failed(VID, f"could not copy {door / (VID + '.mp4')} into the DLC queue; put "
                        f"back in the shared singles folder, where another GPU node can "
                        f"take it. This node does not retry it on its own")
    assert db.reset_failed(VID) == "validated"


def test_any_other_failure_is_not_reset_to_the_shared_folder(env, tmp_path, monkeypatch):
    """Only the copy failure means 'given back to the shared folder'; a pose
    that failed for another reason is not inferred 'validated' from a file of
    the same name there."""
    Paths, db = env
    door = tmp_path / "Single_Animal"
    door.mkdir()
    monkeypatch.setattr(Paths, "SINGLE_ANIMAL_OUTPUT", door, raising=False)
    (door / f"{VID}.mp4").write_bytes(b"v")
    _failed_twice(db)
    with pytest.raises(ValueError):
        db.reset_failed(VID)
    assert db.get_video(VID)["state"] == "failed"
