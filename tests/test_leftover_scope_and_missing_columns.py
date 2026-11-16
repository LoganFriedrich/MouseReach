"""Two things the watcher database must survive (watcher/db.py).

1. A re-run scope LEFT on a row by older code. Rows that went on to a review
   queue, 'unresolvable' or 'archived' used to keep their scope. A filed
   video's leftover 'kinematics' is read back by a run handed no start stage
   (orchestrator.kept_rerun_start_stage), which then skips every algorithm and
   check. So a scope is carried only between the states of a re-run: a row
   entering them from outside starts with none, whichever method moves it;
   and opening the database clears the leftovers.

2. A database that was locked when it was opened, so the columns added later
   (resume_stage, held_at_step) could not be added. Every state change writes
   those columns, so one missed ALTER would fail every transition of that
   process. They are added again just before a write.

Built under tmp_path with invented ids; no real folder or database.
"""
import sqlite3

import pytest

from mousereach.watcher.db import WatcherDB

VID = "20990101_ABC0701_P1"


def _raw(path, sql, args=()):
    con = sqlite3.connect(str(path))
    try:
        con.execute(sql, args)
        con.commit()
    finally:
        con.close()


@pytest.fixture
def filed(tmp_path):
    """A filed video's row, then put in ``state`` carrying a leftover scope
    the way older code left it (written straight into the file)."""
    path = tmp_path / "w.db"
    db = WatcherDB(path)
    db.register_video(video_id=VID, source_path="x")
    db.force_state(VID, "processing", reason="test: first pass")
    db.update_state(VID, "processed")
    db.update_state(VID, "archiving")
    db.update_state(VID, "archived")

    def leave(state, scope="kinematics"):
        _raw(path, "UPDATE videos SET state = ?, reprocess_scope = ? WHERE video_id = ?",
             (state, scope, VID))
        return db
    leave.path = path
    return leave


def test_a_held_video_that_fails_does_not_bring_its_old_scope_into_the_retry(filed):
    from mousereach.watcher.orchestrator import kept_rerun_start_stage
    db = filed("triage")
    db.mark_failed(VID, "test: its return could not run")
    row = db.get_video(VID)
    assert row["state"] == "failed"
    assert row["reprocess_scope"] is None
    assert kept_rerun_start_stage(row) is None        # every step and check runs


@pytest.mark.parametrize("state", ["triage", "deep_review", "unresolvable", "archived"])
def test_a_row_entering_a_rerun_from_outside_starts_with_no_scope(filed, state):
    db = filed(state)
    db.force_state(VID, "outdated", reason="test: marked with no scope given")
    assert db.get_video(VID)["reprocess_scope"] is None


def test_a_scope_given_with_the_move_is_kept(filed):
    db = filed("archived", scope="kinematics")
    db.force_state(VID, "outdated", reason="test: version scan", reprocess_scope="reach")
    assert db.get_video(VID)["reprocess_scope"] == "reach"


def test_a_rerun_keeps_its_scope_from_outdated_to_its_retry(filed):
    db = filed("outdated")
    db.force_state(VID, "processing", reason="test: re-run starts")
    db.mark_failed(VID, "test: kinematics failed")
    assert db.get_video(VID)["reprocess_scope"] == "kinematics"
    db.update_state(VID, "processing")                 # the retry
    assert db.get_video(VID)["reprocess_scope"] == "kinematics"


def test_marking_a_row_unresolvable_ends_its_scope(filed):
    db = filed("outdated")
    db.mark_unresolvable(VID, "test: no file on this node")
    assert db.get_video(VID)["reprocess_scope"] is None


def test_opening_the_database_clears_scopes_outside_a_rerun(filed, tmp_path):
    db = filed("triage")
    other = "20990101_ABC0702_P1"
    db.register_video(video_id=other, source_path="y")
    _raw(filed.path, "UPDATE videos SET state = 'outdated', reprocess_scope = 'kinematics' "
                     "WHERE video_id = ?", (other,))
    reopened = WatcherDB(filed.path)
    assert reopened.get_video(VID)["reprocess_scope"] is None          # leftover
    assert reopened.get_video(other)["reprocess_scope"] == "kinematics"  # owed re-run


def test_the_cleanup_writes_down_every_scope_before_it_clears_it(filed, caplog):
    """The clearing cannot be undone from the database and no person is
    watching the program that does it, so the log must hold, BEFORE the
    rows change, the count per state and each row's old value."""
    import logging
    db = filed("triage", scope="kinematics")
    for n, (state, scope) in enumerate(
            [("archived", "reach"), ("archived", "outcome"), ("outdated", "kinematics")], 2):
        vid = f"20990101_ABC070{n}_P1"
        db.register_video(video_id=vid, source_path="y")
        _raw(filed.path, "UPDATE videos SET state = ?, reprocess_scope = ? "
                         "WHERE video_id = ?", (state, scope, vid))
    seen_before_clear = []

    class _Peek(logging.Handler):
        def emit(self, record):
            if "being cleared" in record.getMessage():
                con = sqlite3.connect(str(filed.path))
                try:
                    seen_before_clear.append(con.execute(
                        "SELECT COUNT(*) FROM videos WHERE reprocess_scope IS NOT NULL"
                    ).fetchone()[0])
                finally:
                    con.close()
    peek = _Peek()
    log = logging.getLogger("mousereach.watcher.db")
    log.addHandler(peek)
    try:
        with caplog.at_level(logging.INFO, logger="mousereach.watcher.db"):
            WatcherDB(filed.path)
    finally:
        log.removeHandler(peek)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "on 3 row(s)" in text
    assert "archived: 2" in text and "triage: 1" in text
    assert f"{VID} (triage) = kinematics" in text
    assert "20990101_ABC0702_P1 (archived) = reach" in text
    assert "20990101_ABC0703_P1 (archived) = outcome" in text
    assert "20990101_ABC0704_P1" not in text               # an owed re-run: not touched
    assert seen_before_clear == [4]                        # written down first, cleared after


# ---------------------------------------------------------------- columns

def _columns(path):
    con = sqlite3.connect(str(path))
    try:
        return {r[1] for r in con.execute("PRAGMA table_info(videos)")}
    finally:
        con.close()


@pytest.fixture
def opened_while_locked(tmp_path, monkeypatch):
    """A database from before the resume-point columns, opened by a process
    whose column migration could not run (what a lock does: every ALTER
    fails and is only logged)."""
    path = tmp_path / "old.db"
    first = WatcherDB(path)
    first.register_video(video_id=VID, source_path="x")
    first.force_state(VID, "processing", reason="test: first pass")
    for column in ("resume_stage", "held_at_step"):
        _raw(path, f"ALTER TABLE videos DROP COLUMN {column}")
    assert not {"resume_stage", "held_at_step"} & _columns(path)
    monkeypatch.setattr(WatcherDB, "_migrate_db", lambda self: None)
    db = WatcherDB(path)
    assert not {"resume_stage", "held_at_step"} & _columns(path)
    return db, path


@pytest.mark.skipif(sqlite3.sqlite_version_info < (3, 35),
                    reason="needs ALTER TABLE ... DROP COLUMN to build the old database")
def test_a_state_change_adds_the_columns_the_open_could_not(opened_while_locked):
    db, path = opened_while_locked
    db.force_state(VID, "triage", reason="test: routed to a queue", held_at_step="outcome")
    row = db.get_video(VID)
    assert row["state"] == "triage"
    assert row["held_at_step"] == "outcome"
    assert {"resume_stage", "held_at_step"} <= _columns(path)


@pytest.mark.skipif(sqlite3.sqlite_version_info < (3, 35),
                    reason="needs ALTER TABLE ... DROP COLUMN to build the old database")
@pytest.mark.parametrize("move", ["update", "force", "failed", "unresolvable", "reset"])
def test_every_way_of_changing_state_works_while_a_column_cannot_be_added(
        opened_while_locked, monkeypatch, move):
    """The column still cannot be added (the lock has not cleared): the state
    change itself goes through, as it did before the columns existed."""
    db, path = opened_while_locked
    monkeypatch.setattr(
        WatcherDB, "_ensure_video_columns",
        lambda self, conn: WatcherDB._read_video_columns(conn))
    if move == "update":
        db.update_state(VID, "triage")
        want = "triage"
    elif move == "force":
        db.force_state(VID, "archived", reason="test")
        want = "archived"
    elif move == "failed":
        db.mark_failed(VID, "test")
        want = "failed"
    elif move == "unresolvable":
        db.mark_unresolvable(VID, "test")
        want = "unresolvable"
    else:
        db.mark_failed(VID, "test")
        db.reset_failed(VID, "dlc_complete")
        want = "dlc_complete"
    assert db.get_video(VID)["state"] == want
    assert not {"resume_stage", "held_at_step"} & _columns(path)
