"""Reaching 'archived' clears the row's error message.

WHY (2026-10-22): error_message is the CURRENT error. Reaching 'archived' did
not clear the previous one, so 1,249 finished videos on the processing server
still showed failures from runs that later succeeded, and every status view read
them as current. The history stays in processing_log.
"""
from mousereach.watcher.db import WatcherDB

V = "20250101_CNT0101_P1"


def _db(tmp_path, state="failed", error="Archive directory not found for reprocessing"):
    db = WatcherDB(tmp_path / "w.db")
    db.register_video(video_id=V, source_path="x")
    db.force_state(V, state, reason="test fixture", error_message=error)
    return db


def test_force_state_to_archived_clears_the_error(tmp_path):
    db = _db(tmp_path)
    db.force_state(V, "archived", reason="test")
    assert db.get_video(V)["error_message"] is None


def test_update_state_to_archived_clears_the_error(tmp_path):
    db = _db(tmp_path, state="archiving")
    db.update_state(V, "archived")
    assert db.get_video(V)["error_message"] is None


def test_other_states_keep_their_message(tmp_path):
    db = _db(tmp_path, state="validated")
    db.mark_unresolvable(V, "no file on this node")
    assert db.get_video(V)["error_message"] == "no file on this node"


def test_the_startup_cleanup_clears_only_archived_rows(tmp_path):
    db = _db(tmp_path, state="archived", error="old failure")
    db.register_video(video_id="20250101_CNT0102_P1", source_path="x")
    db.mark_unresolvable("20250101_CNT0102_P1", "no file on this node")
    import sqlite3
    con = sqlite3.connect(tmp_path / "w.db")
    con.execute("UPDATE videos SET error_message='old failure' WHERE video_id=?", (V,))
    con.commit()
    con.close()
    assert db.clear_errors_on_archived() == 1
    assert db.get_video(V)["error_message"] is None
    assert db.get_video("20250101_CNT0102_P1")["error_message"] == "no file on this node"
