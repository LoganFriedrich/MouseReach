"""Escalate and Flag Session say so when the watcher database did not record the move.

The review window opens the watcher database through the reviewer's own login.
When that database could not record the new state -- here: a database that has
no row for the video, as a database at the wrong path would -- the move was
only logged, and the window said "moved" while the record kept its old state.
Now the window's message says the record was NOT updated and why.
Built under tmp_path; no real queue or database.
"""
from types import SimpleNamespace

import pytest

import mousereach.review.causal_review_widget as crw
from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB

VID = "20990101_ABC9901_P1"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    triage = tmp_path / "triage"
    deep = tmp_path / "deep_review"
    (triage / VID).mkdir(parents=True)
    for name in (f"{VID}.mp4", f"{VID}_segments.json"):
        (triage / VID / name).write_text("x")
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    # The claim folder (<Review>/.return_claims) and the backup folder
    # (<Analyzed>/Archive/manual_corrections) hang off REVIEW_ROOT and
    # ANALYZED_OUTPUT; left unpatched they resolve to the configured real
    # pipeline, which the test guard refuses.
    monkeypatch.setattr(Paths, "REVIEW_ROOT", tmp_path)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed")
    wrong_db = WatcherDB(db_path=tmp_path / "other.db")      # no row for VID
    infos = []
    monkeypatch.setattr(crw, "show_info", lambda msg, *a, **k: infos.append(msg))
    monkeypatch.setattr(crw, "show_error", lambda msg, *a, **k: infos.append("ERR " + msg))
    monkeypatch.setattr(crw, "flag_session", lambda stem, root, reason="", **kw: "key-1")
    w = SimpleNamespace(
        _video_stem=VID, _bundle_dir=triage / VID,
        _review_root=lambda: tmp_path / "root",
        _save_review=lambda: None,
        _notes_edit=SimpleNamespace(toPlainText=lambda: ""),
        _status_label=SimpleNamespace(setText=lambda text: None),
        _load_next_video=lambda: None,
        _watcher_db_for_move=lambda: (wrong_db, ""),
        _release_video_file=lambda: None,
    )
    return w, deep, infos


def test_escalate_says_the_record_was_not_updated(setup):
    w, deep, infos = setup
    crw.CausalReviewWidget._escalate_to_deep_review(w)
    assert (deep / VID / f"{VID}.mp4").is_file()             # the move itself happened
    assert len(infos) == 1 and "record was NOT updated" in infos[0]


def test_flag_session_moves_nothing_so_there_is_no_record_to_update(setup):
    """Flag Session is only a note: no move, no state."""
    w, deep, infos = setup
    crw.CausalReviewWidget._flag_session(w)
    assert not (deep / VID).exists()
    assert len(infos) == 1 and "nothing was moved" in infos[0]


def test_a_recorded_move_adds_no_note(setup, tmp_path):
    w, deep, infos = setup
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete",
               "processing", "triage"):
        db.update_state(VID, st)
    w._watcher_db_for_move = lambda: (db, "")
    crw.CausalReviewWidget._escalate_to_deep_review(w)
    assert db.get_video(VID)["state"] == "deep_review"
    assert len(infos) == 1 and "NOT updated" not in infos[0]
