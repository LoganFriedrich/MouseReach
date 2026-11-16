"""A single whose copy into a GPU node's DLC queue failed is offered again.

The copy failure marks the row 'failed' and gives the video back to the
shared singles folder. A GPU node's work loop has no job list for failed rows
and the intake skipped every known video, so this node never retried it --
with one GPU node, nothing ever did. Now the intake validates such a row again
an hour after the failure while it has retries left (and only when the file
is back at the top of the folder, stable and claimed by nobody).
tmp_path only (the fixtures of test_singles_intake_claim).
"""
import sqlite3
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import mousereach.watcher.orchestrator as orch
from mousereach.watcher import single_claim
from mousereach.watcher.state import WatcherStateManager

from test_singles_intake_claim import HOST, _adopt, _drop, _node, _row, env  # noqa: F401

VID = "20240101_ABC9901_P1"


def _failed_copy(env, monkeypatch):
    a = _node(env, HOST)
    src = _drop(env, vid=VID)
    _row(a, VID, "validated", src)
    with monkeypatch.context() as m:
        m.setattr(orch, "safe_copy", lambda s, d, verify=True: False)
        assert _adopt(env, a, vid=VID) is False
    assert a.db.get_video(VID)["state"] == "failed"
    return a, src


def _hours_ago(db, hours):
    when = (datetime.now() - timedelta(hours=hours)).isoformat()
    conn = sqlite3.connect(str(db.db_path))
    conn.execute("UPDATE videos SET last_error_at = ? WHERE video_id = ?", (when, VID))
    conn.commit()
    conn.close()


def _intake(a, door):
    state = WatcherStateManager(a.db, SimpleNamespace(stability_wait_seconds=0,
                                                      max_retries=3))
    state.discover_new_singles(door)                    # first sighting
    return state.discover_new_singles(door)


def test_a_failed_copy_is_offered_again_after_an_hour(env, monkeypatch):
    a, src = _failed_copy(env, monkeypatch)
    _hours_ago(a.db, 2)
    assert _intake(a, env.door) == [VID]
    row = a.db.get_video(VID)
    assert row["state"] == "validated" and row["current_path"] == str(src)
    assert row["error_count"] == 1                       # the attempt still counts


def test_not_within_the_hour(env, monkeypatch):
    a, _ = _failed_copy(env, monkeypatch)
    assert _intake(a, env.door) == []
    assert a.db.get_video(VID)["state"] == "failed"


def test_not_once_its_retries_are_used_up(env, monkeypatch):
    a, _ = _failed_copy(env, monkeypatch)
    conn = sqlite3.connect(str(a.db.db_path))
    conn.execute("UPDATE videos SET error_count = 3 WHERE video_id = ?", (VID,))
    conn.commit()
    conn.close()
    _hours_ago(a.db, 2)
    assert _intake(a, env.door) == []
    assert a.db.get_video(VID)["state"] == "failed"


def test_not_while_another_node_holds_it(env, monkeypatch):
    a, src = _failed_copy(env, monkeypatch)
    _hours_ago(a.db, 2)
    other = single_claim.host_dir("NODE-OTHER")
    other.mkdir(parents=True, exist_ok=True)
    (other / f"{VID}.mp4").write_bytes(b"video")
    assert _intake(a, env.door) == []


def test_another_failure_is_not_offered_again(env):
    """A failed pose is not a failed copy: the intake leaves it alone."""
    a = _node(env, HOST)
    _drop(env, vid=VID)
    _row(a, VID, "dlc_running", env.door / f"{VID}.mp4")
    a.db.mark_failed(VID, "DLC failed: out of memory")
    _hours_ago(a.db, 2)
    assert _intake(a, env.door) == []
