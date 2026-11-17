"""A video whose route into a review queue was refused after its analysis is
run again only to try the route again: its stage outputs are reused.

A refusal because the queue holds the video's live bundle lasts until a
person presses Resolve. The video waits in Processing and is picked again
every half hour; each pick used to re-run segmentation, reach detection and
outcomes from scratch. A refusal of the route after a FAILED segmentation is
different: the next run must segment again to route again. Built under
tmp_path with an invented id; every stage is a stand-in.
"""
import json

import pytest

from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import ProcessingOrchestrator
from mousereach.watcher.review_gate import AlreadyInQueueError

VID = "20990101_ABC9901_P1"


@pytest.fixture
def env(tmp_path, monkeypatch):
    proc = tmp_path / "Processing"
    proc.mkdir()
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "Review" / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "Review" / "deep_review")
    monkeypatch.setattr(Paths, "REVIEW_ROOT", tmp_path / "Review")
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path / "nas", raising=False)
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.processing_dir = proc
    o.hostname = "TEST-HOST"
    o._pose_check_before_segmentation = lambda *a, **k: True
    o._hand_score_precheck = lambda vid, d: True
    o._run_quality_triage = lambda *a, **k: "auto_approved"
    o.db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete", "processing"):
        o.db.update_state(VID, st)
    (proc / f"{VID}.mp4").write_bytes(b"video")
    pose = proc / f"{VID}DLC_resnet50_TestModelJan1shuffle1_100.h5"
    pose.write_bytes(b"pose")

    calls = []

    def seg(dlc, *a, **k):
        calls.append("segmentation")
        (proc / f"{VID}_segments.json").write_text(json.dumps({"boundaries": [0, 100]}))
        return {"success": True, "n_boundaries": 2}

    def reach(dlc, seg_path, *a, **k):
        calls.append("reach")
        (proc / f"{VID}_reaches.json").write_text("{}")
        return {"total_reaches": 0}

    def outcome(dlc, seg_path, reach_path, *a, **k):
        calls.append("outcome")
        (proc / f"{VID}_pellet_outcomes.json").write_text("{}")
        return {"n_segments": 0}

    import mousereach.segmentation.core.batch as segb
    import mousereach.reach.core.batch as reachb
    import mousereach.outcomes.core.batch as outb
    import mousereach.assignment.run as assign
    import mousereach.watcher.review_gate as rg
    monkeypatch.setattr(segb, "process_single", seg)
    monkeypatch.setattr(segb, "add_validation_status", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(reachb, "process_single", reach)
    monkeypatch.setattr(outb, "process_single", outcome)

    def assigned(folder, vid, dlc):
        (proc / f"{VID}_reach_assignments.json").write_text("{}")
        return {}
    monkeypatch.setattr(assign, "assign_reaches_for_video", assigned)

    def refusing_gate(*a, **k):
        raise AlreadyInQueueError("the queue already holds its live bundle")
    monkeypatch.setattr(rg, "run_gate", refusing_gate)

    def run():
        row = o.db.get_video(VID)
        o._run_pipeline({"id": VID, "data": dict(row, dlc_output_path=str(pose))})
    return o, proc, calls, run


def test_a_refused_route_after_the_analysis_reuses_the_stage_outputs(env):
    o, proc, calls, run = env
    run()
    assert calls == ["segmentation", "reach", "outcome"]
    assert o._route_refused_waiting(VID)                   # refused, waits
    assert o.db.get_video(VID)["state"] == "processing"
    o._route_refusals[VID] = 0                             # the wait is over
    del calls[:]
    run()
    assert calls == []                                     # nothing recomputed
    assert o._route_refused_waiting(VID)                   # refused again, waits


def test_after_a_restart_the_video_runs_in_full_once(env):
    o, proc, calls, run = env
    run()
    o._route_refusals = {}                                 # a restart forgets
    o._refused_after_analysis = set()
    del calls[:]
    run()
    assert calls == ["segmentation", "reach", "outcome"]


def test_a_refused_route_after_a_failed_segmentation_segments_again(env):
    o, proc, calls, run = env
    o._note_route_refused(VID, AlreadyInQueueError("refused"))   # not analysed
    (proc / f"{VID}_segments.json").write_text("{}")
    (proc / f"{VID}_reaches.json").write_text("{}")
    (proc / f"{VID}_pellet_outcomes.json").write_text("{}")
    o._route_refusals[VID] = 0
    run()
    assert calls[:1] == ["segmentation"]
