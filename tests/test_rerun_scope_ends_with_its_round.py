"""A re-run scope ends with its re-run (watcher/db.py REPROCESS_SCOPE_STATES).

WHY: a filed video's 'kinematics' scope is read back from the row by a run
that was handed no start stage (orchestrator.kept_rerun_start_stage), and
such a run skips every algorithm, the quality triage and the hand-score
comparison. A scope left on a row that went on to triage, deep review, a
quality hold or 'unresolvable' was read back by a later, unrelated run --
a quality-hold release or an intake -- which then reused whatever outputs it
found unchecked. Entering any state outside the re-run now clears the scope.

Built under tmp_path with invented ids; no real folder or database.
"""
import pytest

from mousereach.watcher.db import WatcherDB

VID = "20990101_ABC0601_P1"


@pytest.fixture
def db(tmp_path):
    d = WatcherDB(tmp_path / "w.db")
    d.register_video(video_id=VID, source_path="x")
    d.force_state(VID, "processing", reason="test: first pass")
    d.update_state(VID, "processed")
    d.update_state(VID, "archiving")
    d.update_state(VID, "archived")
    d.force_state(VID, "outdated", reason="test: version scan", reprocess_scope="kinematics")
    return d


def test_the_scope_survives_its_own_round(db):
    db.force_state(VID, "processing", reason="test: re-run starts")
    assert db.get_video(VID)["reprocess_scope"] == "kinematics"
    db.mark_failed(VID, "kinematics failed: boom")         # waits for its retry
    assert db.get_video(VID)["reprocess_scope"] == "kinematics"


@pytest.mark.parametrize("state", ["triage", "deep_review", "dlc_complete", "unresolvable"])
def test_a_row_leaving_its_rerun_loses_the_scope(db, state):
    db.force_state(VID, "processing", reason="test: re-run starts")
    db.force_state(VID, state, reason="test: held or parked")
    assert db.get_video(VID)["reprocess_scope"] is None


def test_a_later_run_does_not_read_a_leftover_scope_back(db):
    """A quality hold released to 'dlc_complete' and taken in again: the run
    has no start stage of its own and every step and check runs."""
    from mousereach.watcher.orchestrator import kept_rerun_start_stage
    db.force_state(VID, "processing", reason="test: re-run starts")
    db.force_state(VID, "dlc_complete", reason="test: quality hold released")
    db.update_state(VID, "processing")
    assert kept_rerun_start_stage(db.get_video(VID)) is None


def test_a_caller_may_still_set_a_scope_with_the_state(db):
    db.force_state(VID, "processing", reason="test: re-run starts")
    db.force_state(VID, "unresolvable", reason="test")
    db.force_state(VID, "outdated", reason="test: narrowed", reprocess_scope="segmentation")
    assert db.get_video(VID)["reprocess_scope"] == "segmentation"


def test_a_person_resetting_a_failed_rerun_gets_a_full_run(db):
    db.force_state(VID, "processing", reason="test: re-run starts")
    db.mark_failed(VID, "kinematics failed: boom")
    db.reset_failed(VID, to_state="dlc_complete")
    assert db.get_video(VID)["reprocess_scope"] is None


# ---------------------------------------------------------------------------
# A review return's resume point ends with its round too
# (watcher/db.py RESUME_POINT_STATES)
# ---------------------------------------------------------------------------

@pytest.fixture
def resumed(tmp_path):
    """A row a review return put back in 'processing': it resumes at the
    checks, after a hold at reach assignment."""
    d = WatcherDB(tmp_path / "r.db")
    d.register_video(video_id=VID, source_path="x")
    d.force_state(VID, "processing", reason="test: first pass")
    d.update_state(VID, "triage", held_at_step="assignment")
    d.update_state(VID, "processing", resume_stage="checks")
    return d


def test_the_resume_point_survives_its_own_round(resumed):
    resumed.mark_failed(VID, "kinematics failed: boom")     # waits for its retry
    row = resumed.get_video(VID)
    assert row["resume_stage"] == "checks"
    assert row["held_at_step"] == "assignment"


@pytest.mark.parametrize("to_state", ["dlc_complete", "dlc_queued", "validated"])
def test_a_person_resetting_a_resumed_run_gets_a_fresh_run(resumed, to_state):
    resumed.mark_failed(VID, "boom")
    resumed.reset_failed(VID, to_state=to_state)
    row = resumed.get_video(VID)
    assert row["resume_stage"] is None and row["held_at_step"] is None


@pytest.mark.parametrize("state", ["dlc_complete", "dlc_queued", "unresolvable", "outdated"])
def test_a_row_leaving_its_resumed_run_loses_the_resume_point(resumed, state):
    resumed.force_state(VID, state, reason="test: reset, re-posed or parked")
    row = resumed.get_video(VID)
    assert row["resume_stage"] is None and row["held_at_step"] is None


def test_an_unresolvable_row_found_again_is_a_fresh_run(resumed):
    resumed.update_state(VID, "unresolvable")
    resumed.update_state(VID, "dlc_complete")
    resumed.update_state(VID, "processing")
    assert resumed.get_video(VID)["resume_stage"] is None


def test_a_new_hold_keeps_the_step_it_sets(resumed):
    resumed.update_state(VID, "deep_review", held_at_step="segmentation")
    row = resumed.get_video(VID)
    assert row["held_at_step"] == "segmentation"
    assert row["resume_stage"] is None


def test_a_row_set_to_its_queue_state_keeps_the_step_its_route_recorded(resumed):
    """The watcher sets a row to its queue's state when it finds the bundle
    there (no held_at_step passed): the step the route recorded stays, and
    the resume point of the earlier round goes."""
    resumed.force_state(VID, "triage", reason="test: bundle found in the queue")
    row = resumed.get_video(VID)
    assert row["held_at_step"] == "assignment"
    assert row["resume_stage"] is None
