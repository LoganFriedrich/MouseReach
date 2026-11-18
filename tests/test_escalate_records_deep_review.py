"""Escalate and Flag Session record 'deep_review' in the watcher database.

The review tool's Escalate button and Flag Session button move a video's bundle
from the triage queue into deep review. They used to route with no database,
so the bundle sat in deep review while the row still said 'triage': the folder
and the database disagreed until the video came back. Now they pass the
watcher's own database (never creating one) and the row follows the folder.
Built under tmp_path; no real folder or database.
"""
import inspect

import pytest

from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB
from mousereach.watcher import db_location
from mousereach.watcher.review_gate import escalate_to_deep_review

VID = "20990101_ABC9901_P1"


@pytest.fixture
def queues(tmp_path, monkeypatch):
    triage = tmp_path / "triage"
    deep = tmp_path / "deep_review"
    (triage / VID).mkdir(parents=True)
    for name in (f"{VID}.mp4", f"{VID}_segments.json", f"{VID}_pellet_outcomes.json"):
        (triage / VID / name).write_text("x")
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    # The claim folder (<Review>/.return_claims) and the backup folder
    # (<Analyzed>/Archive/manual_corrections) hang off REVIEW_ROOT and
    # ANALYZED_OUTPUT; left unpatched they resolve to the configured real
    # pipeline, which the test guard refuses.
    monkeypatch.setattr(Paths, "REVIEW_ROOT", tmp_path)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed")
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete",
               "processing", "triage"):
        db.update_state(VID, st)
    return triage, deep, db


def test_escalate_moves_the_bundle_and_records_deep_review(queues):
    triage, deep, db = queues
    escalate_to_deep_review(VID, triage / VID, "reviewer escalated", db=db)
    assert (deep / VID / f"{VID}.mp4").is_file()
    assert db.get_video(VID)["state"] == "deep_review"


def test_the_database_is_opened_never_created(tmp_path, monkeypatch):
    missing = tmp_path / "nothing_here.db"
    monkeypatch.setattr(db_location, "resolve_watcher_db_path", lambda: missing)
    db, why = db_location.open_watcher_db_if_present()
    assert db is None and "no watcher database" in why
    assert not missing.exists()


def test_the_existing_database_is_opened(tmp_path, monkeypatch):
    path = tmp_path / "w.db"
    WatcherDB(db_path=path)
    monkeypatch.setattr(db_location, "resolve_watcher_db_path", lambda: path)
    db, why = db_location.open_watcher_db_if_present()
    assert db is not None and why is None


def test_the_review_tool_buttons_pass_the_database():
    from mousereach.review import causal_review_widget as w
    cls = w.CausalReviewWidget
    # Flag Session's move is in _flag_session_divert (run under the claim).
    for fn in (cls._flag_session_divert, cls._escalate_to_deep_review):
        src = inspect.getsource(fn)
        assert "_watcher_db_for_move()" in src, fn.__name__
        assert "db=None" not in src, fn.__name__
    assert 'db_state="deep_review"' in inspect.getsource(cls._flag_session_divert)
