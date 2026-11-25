"""A video's filing time means "filed now".

archived_at is set by filing and holds only while the row is in a filed state
('archived', 'outdated', 'crystallized'). Every path that takes a video out of
the filed state moves the time into previously_filed_at and clears archived_at;
filing sets it again. The watcher's own "has it ever been filed" decisions read
db.last_filed_at and are unchanged. One test per path that leaves the filed
state, the moves between filed states that keep the time, and the one-off
command for rows written before the rule (watcher/filing_time.py).

Built under tmp_path with invented ids; no real folder or database.
"""
import sqlite3

import pytest

from mousereach.config import Paths
from mousereach.watcher.db import FILED_STATES, WatcherDB, last_filed_at

VID = "20990101_ABC9901_P1"


def _filed_db(path, vid=VID):
    db = WatcherDB(db_path=path)
    db.register_video(video_id=vid, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete",
               "processing", "processed", "archiving", "archived"):
        db.update_state(vid, st)
    return db


def _filed_at(db, vid=VID):
    return db.get_video(vid)["archived_at"]


def _left_filed(db, filed_at, vid=VID):
    row = db.get_video(vid)
    assert row["state"] not in FILED_STATES
    assert row["archived_at"] is None
    assert row["previously_filed_at"] == filed_at
    assert last_filed_at(row) == filed_at


@pytest.fixture
def filed(tmp_path, monkeypatch):
    folder = tmp_path / "Analyzed" / "PROJ" / "COHORT"
    folder.mkdir(parents=True)
    for name in (f"{VID}.mp4", f"{VID}_pellet_outcomes.json", f"{VID}_features.json"):
        (folder / name).write_text("{}" if name.endswith(".json") else "v")
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed")
    db = _filed_db(tmp_path / "w.db")
    return folder, db, _filed_at(db)


# --- every path out of the filed state -----------------------------------------

def test_route_to_queue_into_triage(filed):
    from mousereach.watcher.review_gate import route_to_queue
    folder, db, at = filed
    route_to_queue(VID, folder, Paths.TRIAGE_REVIEW, "test", db=db, db_state="triage")
    _left_filed(db, at)


def test_route_of_a_filed_copy_into_deep_review(filed):
    # mousereach-route-to-queue, the version scan's mislabel divert and a
    # re-cut all route a COPY of the filed results (filed_copy_route).
    from mousereach.watcher.filed_copy_route import route_filed_copy
    folder, db, at = filed
    route_filed_copy(VID, folder, Paths.DEEP_REVIEW, "re-cut", db=db,
                     db_state="deep_review")
    _left_filed(db, at)


def test_reprocessing_a_filed_video(filed):
    # The version scan marks it 'outdated' (still filed); the re-run moves it
    # to 'processing' (orchestrator._reprocess_video).
    _, db, at = filed
    db.update_state(VID, "outdated", reprocess_scope="kinematics")
    assert _filed_at(db) == at
    db.force_state(VID, "processing",
                   reason="re-run: archived results copied down to bring it current",
                   resume_stage=None)
    _left_filed(db, at)


def test_a_repose_request(filed):
    from mousereach.watcher import repose
    _, db, at = filed
    repose._requeue_for_dlc(db, VID, "archived", "a new pose was asked for")
    _left_filed(db, at)


def test_retiring_a_row(filed):
    _, db, at = filed
    db.mark_unresolvable(VID, "not a real video: name typo")
    _left_filed(db, at)


def test_a_rerun_that_fails(filed):
    _, db, at = filed
    db.update_state(VID, "outdated")
    db.mark_failed(VID, "re-run could not start")
    _left_filed(db, at)


def test_a_forced_move_to_any_working_state(filed):
    _, db, at = filed
    db.force_state(VID, "dlc_complete", reason="a person's command")
    _left_filed(db, at)


def test_an_escalation_keeps_the_previous_filing_time(filed):
    from mousereach.watcher.review_gate import route_to_queue
    folder, db, at = filed
    route_to_queue(VID, folder, Paths.TRIAGE_REVIEW, "test", db=db, db_state="triage")
    db.update_state(VID, "deep_review")
    _left_filed(db, at)


def test_a_reset_of_a_failed_rerun(filed):
    _, db, at = filed
    db.update_state(VID, "outdated")
    db.mark_failed(VID, "boom")
    db.reset_failed(VID, to_state="processing")
    _left_filed(db, at)


# --- filing sets it again; moves between filed states keep it -------------------

def test_filing_again_sets_a_new_time_and_keeps_the_previous(filed):
    _, db, at = filed
    db.update_state(VID, "outdated")
    db.update_state(VID, "processing")
    with db._state_write() as conn:      # a later filing gets a later time
        conn.execute("UPDATE videos SET previously_filed_at = '2000-01-01' WHERE video_id = ?",
                     (VID,))
    for st in ("processed", "archiving", "archived"):
        db.update_state(VID, st)
    row = db.get_video(VID)
    assert row["archived_at"] and row["archived_at"] != "2000-01-01"
    assert row["previously_filed_at"] == "2000-01-01"


@pytest.mark.parametrize("path", ["unmark", "lock_unlock", "same_state"])
def test_moves_between_filed_states_keep_the_filing_time(filed, path):
    _, db, at = filed
    with db._state_write() as conn:      # make "now" visibly different
        conn.execute("UPDATE videos SET archived_at = '2000-01-01T00:00:00' WHERE video_id = ?",
                     (VID,))
    if path == "unmark":
        db.update_state(VID, "outdated")
        db.update_state(VID, "archived")
    elif path == "lock_unlock":
        db.update_state(VID, "crystallized")
        db.force_state(VID, "archived", reason="unlock", unlock_crystallized=True)
    else:
        db.force_state(VID, "archived", reason="sweep: recording the file")
    row = db.get_video(VID)
    assert row["state"] == "archived"
    assert row["archived_at"] == "2000-01-01T00:00:00"
    assert row["previously_filed_at"] is None


def test_a_caller_restoring_the_filing_time_wins(filed):
    # The failed-re-run restore puts the row back to 'archived' with the time
    # it was filed (orchestrator._restore_filed_after_failed_rerun).
    _, db, at = filed
    db.update_state(VID, "outdated")
    db.update_state(VID, "processing")
    db.force_state(VID, "archived", reason="restore", archived_at=last_filed_at(db.get_video(VID)))
    assert _filed_at(db) == at


# --- the "ever filed" decisions are unchanged ------------------------------------

def test_a_kinematics_rerun_still_knows_it_was_filed(filed):
    from mousereach.watcher.orchestrator import kept_rerun_start_stage
    _, db, at = filed
    db.update_state(VID, "outdated", reprocess_scope="kinematics")
    db.update_state(VID, "processing", reprocess_scope="kinematics")
    row = db.get_video(VID)
    assert row["archived_at"] is None
    assert kept_rerun_start_stage(row) == "kinematics"


def test_bring_current_still_refuses_a_video_that_was_filed(filed, monkeypatch):
    import mousereach.pipeline.reprocess_to_current as rtc
    _, db, at = filed
    db.mark_unresolvable(VID, "retired")
    assert rtc._is_filed(VID, db.get_video(VID))


# --- the one-off command for rows written before the rule ------------------------

def _old_style(db, vid, state, archived_at):
    """A row as the code before the rule left it: not filed, filing time kept."""
    db.register_video(video_id=vid, source_path="x")
    with db._state_write() as conn:
        conn.execute("UPDATE videos SET state = ?, archived_at = ?, updated_at = '2001-01-01' "
                     "WHERE video_id = ?", (state, archived_at, vid))


def test_the_command_lists_then_moves_and_is_done_after(tmp_path, capsys):
    from mousereach.watcher import filing_time
    db = _filed_db(tmp_path / "w.db", vid="20990101_ABC9901_P2")
    _old_style(db, "20990101_ABC9901_P3", "triage", "2026-09-01T00:00:00")
    _old_style(db, "20990101_ABC9901_P4", "unresolvable", "2026-08-01T00:00:00")
    _old_style(db, "20990101_ABC9901_P5", "outdated", "2026-07-01T00:00:00")   # filed: kept
    path = str(tmp_path / "w.db")

    assert filing_time.main(["--db", path]) == 0
    out = capsys.readouterr().out
    assert "Would move the filing time of 2 row(s)" in out
    assert db.get_video("20990101_ABC9901_P3")["archived_at"] == "2026-09-01T00:00:00"

    assert filing_time.main(["--db", path, "--apply"]) == 0
    p3 = db.get_video("20990101_ABC9901_P3")
    assert p3["archived_at"] is None and p3["previously_filed_at"] == "2026-09-01T00:00:00"
    assert p3["state"] == "triage" and p3["updated_at"] == "2001-01-01"
    assert db.get_video("20990101_ABC9901_P4")["previously_filed_at"] == "2026-08-01T00:00:00"
    assert db.get_video("20990101_ABC9901_P5")["archived_at"] == "2026-07-01T00:00:00"
    assert db.get_video("20990101_ABC9901_P2")["archived_at"]
    con = sqlite3.connect(path)
    logged = con.execute("SELECT COUNT(*) FROM processing_log WHERE step = 'filing_time'").fetchone()
    con.close()
    assert logged[0] == 2

    capsys.readouterr()
    assert filing_time.main(["--db", path, "--apply"]) == 0
    assert "of 0 row(s)" in capsys.readouterr().out
