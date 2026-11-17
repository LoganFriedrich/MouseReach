"""A route into a review queue that is refused before anything moves leaves
the video in Processing to run again later -- it is not marked failed.

A refusal (review_gate.AlreadyInQueueError) means another program held the
video's claim, or a queue folder of the video could not be set aside because
a file in it was open in a review window. Those pass. Marking the video failed
counted towards its retry limit, so an open window could use the retries up
and leave the video failed for good. Built under tmp_path with an invented id.
"""
import json

import pytest

from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import ProcessingOrchestrator
from mousereach.watcher.review_gate import DECISION_CLEAN, AlreadyInQueueError

VID = "20990101_ABC9901_P1"


@pytest.fixture
def env(tmp_path, monkeypatch):
    proc = tmp_path / "Processing"
    proc.mkdir()
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "Review" / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "Review" / "deep_review")
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.processing_dir = proc
    o._hand_score_precheck = lambda vid, d: True
    o._run_quality_triage = lambda *a, **k: "auto_approved"
    o.db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete", "processing"):
        o.db.update_state(VID, st)
    (proc / f"{VID}_segments.json").write_text(json.dumps({"boundaries": [0, 100]}))
    (proc / f"{VID}.mp4").write_bytes(b"video")
    return o, proc


def _refusing_gate(*a, **k):
    raise AlreadyInQueueError("a file in the old queue folder is open")


def test_a_refused_route_holds_the_video_without_failing_it(env):
    o, proc = env
    held = o._run_checks_before_kinematics(VID, proc / "pose.h5", 1.0, None,
                                           _refusing_gate, DECISION_CLEAN)
    assert held is True                       # no kinematics, no filing
    row = o.db.get_video(VID)
    assert row["state"] == "processing"       # not 'failed'
    assert int(row.get("error_count") or 0) == 0
    assert (proc / f"{VID}.mp4").is_file()    # nothing moved
    # It waits before the pipeline takes it again, then is due once more.
    assert o._route_refused_waiting(VID)
    o._route_refusals[VID] = 0
    assert not o._route_refused_waiting(VID)


def test_videos_whose_route_was_refused_do_not_stop_intake(env, monkeypatch):
    """A refusal can last a weekend (a window holding a file open) or until a
    person presses Resolve. The waiting videos are analysed already: counted
    against max_local_pending, two of them stopped all intake."""
    from types import SimpleNamespace
    from mousereach.watcher import work_priority as wp
    monkeypatch.setattr(wp, "read_lab_priority", lambda *a, **k: None)
    wp.invalidate_lab_policy()
    o, proc = env
    other = "20990101_ABC9902_P1"
    o.db.register_video(video_id=other, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete", "processing"):
        o.db.update_state(other, st)
    for vid in (VID, other):
        o._run_checks_before_kinematics(vid, proc / "pose.h5", 1.0, None,
                                        _refusing_gate, DECISION_CLEAN)
        assert o.db.get_video(vid)["state"] == "processing"
    new = "20990101_ABC9903_P1"
    o.db.register_video(video_id=new, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete"):
        o.db.update_state(new, st)
    o.config = SimpleNamespace(max_local_pending=2, also_process=False,
                               work_priority=None)
    o._get_priority_animal = lambda: None
    o._archive_reachable = lambda: True
    o._held_elsewhere_active = lambda vid: False
    work = o._select_work_item(admit_deferred=True)
    assert work["type"] == "intake" and work["id"] == new
    # The wait is over: the refused video is due to run again, and still does
    # not count; once the pipeline starts on it, it is work again and counts.
    o._route_refusals[VID] = 0
    assert not o._route_refused_waiting(VID)
    work = o._select_work_item(admit_deferred=True)
    assert work["type"] == "intake" and work["id"] == new
    assert o._held_for_refused_route(VID)
    o._forget_route_refusal(VID)
    assert not o._held_for_refused_route(VID)
    assert o._held_for_refused_route(other)
    wp.invalidate_lab_policy()


def test_a_refused_route_from_the_hand_score_check_is_a_refused_route(
        tmp_path, monkeypatch):
    """The hand-score check sends a video whose pellet numbers are in doubt
    to manual segmentation. When that route is refused, the video is a
    refused route like any other -- logged 'review_gate refused', run again
    after the wait -- not a hand-score check that "could not run" (listed
    with the wrong reason, and waiting for the next scores file)."""
    import mousereach.watcher.hand_scores as hs
    import mousereach.watcher.review_gate as rg
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path / "nas", raising=False)
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.hostname = "node-a"
    o.db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete", "processing"):
        o.db.update_state(VID, st)
    folder = tmp_path / "Processing"
    folder.mkdir()
    (folder / f"{VID}.mp4").write_bytes(b"video")

    def _doubt(*a, **k):
        raise hs.PelletNumbersUnknown("a person said segment 2 is pellet 3")

    def _refuse(*a, **k):
        raise AlreadyInQueueError("the queue already holds its live bundle")

    monkeypatch.setattr(hs, "check", _doubt)
    monkeypatch.setattr(hs, "scores_file_stamp", lambda: 5.0)
    monkeypatch.setattr(rg, "route_deep_review", _refuse)
    assert o._hand_score_precheck(VID, folder) is False      # stops this run
    assert VID not in getattr(o, "_score_waits", {})         # not a scores wait
    assert o._route_refused_waiting(VID)                     # runs again later
    steps = [(r["step"], r["status"]) for r in o.db.get_recent_log(50)
             if r["video_id"] == VID]
    assert ("review_gate", "refused") in steps
    assert o.db.get_video(VID)["state"] == "processing"
    assert (folder / f"{VID}.mp4").is_file()
