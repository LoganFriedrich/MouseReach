"""mark_failed never re-labels a crystallized (publication-locked) video.

force_state already refuses to take a row out of 'crystallized'; mark_failed
was a raw UPDATE from any state, so an exception anywhere in a handler could
turn a locked video into 'failed' -- and 'failed' puts a video back into the
retry work. It now refuses the same way. Built under tmp_path.
"""
import pytest

from mousereach.watcher.db import CrystallizedError, WatcherDB

VID = "20990101_ABC9901_P1"


def _db(tmp_path, last):
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete",
               "processing", "processed", "archiving", "archived", "crystallized"):
        db.update_state(VID, st)
        if st == last:
            break
    return db


def test_a_crystallized_row_is_not_marked_failed(tmp_path):
    db = _db(tmp_path, "crystallized")
    with pytest.raises(CrystallizedError):
        db.mark_failed(VID, "something went wrong")
    row = db.get_video(VID)
    assert row["state"] == "crystallized"
    assert not row["error_count"]


def test_any_other_row_is_still_marked_failed(tmp_path):
    db = _db(tmp_path, "processing")
    db.mark_failed(VID, "something went wrong")
    row = db.get_video(VID)
    assert row["state"] == "failed" and row["error_count"] == 1


def test_a_crystallized_row_is_not_marked_unresolvable(tmp_path):
    db = _db(tmp_path, "crystallized")
    with pytest.raises(CrystallizedError):
        db.mark_unresolvable(VID, "no file on this node")
    row = db.get_video(VID)
    assert row["state"] == "crystallized"
    assert row["error_message"] != "no file on this node"


def test_any_other_row_is_still_marked_unresolvable(tmp_path):
    db = _db(tmp_path, "dlc_queued")
    db.mark_unresolvable(VID, "no file on this node")
    assert db.get_video(VID)["state"] == "unresolvable"

