"""A state change reads the row's state and writes the new one under ONE write
lock, so a second program cannot change the row in between and then be
silently overwritten (watcher/db.py WatcherDB._state_write).

Each test lets a second connection -- standing for another program that
writes the same database (a review window, a command-line tool) -- try to
change the row at the moment the first writer has read the state and is
checking it. Without the lock that second write lands and the first writer's
UPDATE then overwrites it with a state that was checked against a state the
row no longer had. With the lock the second write is refused while the first
holds the row, and the first writer's check stands.
"""
import sqlite3

import pytest

from mousereach.watcher.db import WatcherDB

VID = "20990101_ZZZ0101_P1"


@pytest.fixture
def db(tmp_path):
    d = WatcherDB(tmp_path / "watcher.db")
    d.register_video(VID, str(tmp_path / f"{VID}.mp4"))
    d.update_state(VID, "validated")
    d.update_state(VID, "dlc_queued")
    return d


def _interloper(db_path, outcome):
    """Another program setting the row to 'unresolvable' with its own
    connection, giving up after 0.2 s if the database is locked."""
    def run(*_a, **_k):
        conn = sqlite3.connect(str(db_path), timeout=0.2, isolation_level=None)
        try:
            conn.execute("UPDATE videos SET state = 'unresolvable' WHERE video_id = ?",
                         (VID,))
            outcome.append("written")
        except sqlite3.OperationalError as err:
            outcome.append(f"refused: {err}")
        finally:
            conn.close()
    return run


def test_a_second_program_cannot_change_the_row_between_the_check_and_the_write(
        db, monkeypatch):
    outcome = []
    real_check = db._validate_transition
    interloper = _interloper(db.db_path, outcome)

    def check_then_interloper(*a, **k):
        real_check(*a, **k)
        interloper()

    monkeypatch.setattr(db, "_validate_transition", check_then_interloper)
    db.update_state(VID, "dlc_running")

    assert outcome and outcome[0].startswith("refused"), (
        f"the second program's write landed between the check and the write: "
        f"{outcome}")
    assert db.get_video(VID)["state"] == "dlc_running"


def test_mark_failed_counts_against_the_state_it_read(db, monkeypatch):
    """mark_failed reads the error count and state, then writes; a second
    program's change in between is refused, not overwritten."""
    db.update_state(VID, "dlc_running")
    outcome = []
    real = db._end_of_round_sql
    interloper = _interloper(db.db_path, outcome)

    def end_sql_then_interloper(*a, **k):
        sql = real(*a, **k)
        interloper()
        return sql

    monkeypatch.setattr(db, "_end_of_round_sql", end_sql_then_interloper)
    db.mark_failed(VID, "pose failed")

    assert outcome and outcome[0].startswith("refused"), outcome
    row = db.get_video(VID)
    assert row["state"] == "failed" and row["error_count"] == 1


def test_a_refused_write_changes_nothing(db, monkeypatch):
    """A check that raises inside the lock rolls everything back."""
    with pytest.raises(ValueError):
        db.update_state(VID, "archived")   # dlc_queued -> archived is not legal
    assert db.get_video(VID)["state"] == "dlc_queued"
    # And the database is not left locked: another writer gets in at once.
    conn = sqlite3.connect(str(db.db_path), timeout=0.2, isolation_level=None)
    try:
        conn.execute("UPDATE videos SET updated_at = updated_at WHERE video_id = ?", (VID,))
    finally:
        conn.close()
