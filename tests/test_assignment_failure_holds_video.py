"""A reach assignment that fails holds the video for a person, at that step.

Reach assignment (which reach caused each touched pellet) used to fail with a
warning only. It wrote no file, which switched off the gate's "touched pellet
with no credited reach" hold, so the video went on to kinematics and filing
with no causal reach anywhere. Now the assignment file is replaced by one that
names no causal reach and says why; the gate holds the video in TRIAGE with a
reason saying the reach assignment could not run, every touched pellet is
asked of a person, and the return scan's release rule (every triaged pellet
answered) applies as for any triage hold. Built under tmp_path.
"""
import json
from pathlib import Path
from unittest import mock

import pytest

from mousereach.config import Paths
from mousereach.review.triage_status import triage_status
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import ProcessingOrchestrator
from mousereach.watcher.review_gate import (
    ASSIGNMENT_FAILED_KEY, DECISION_CLEAN, DECISION_TRIAGE, evaluate_gate,
)

VID = "20990101_ABC9901_P1"
POSE = f"{VID}DLC_resnet50_testshuffle1_100.h5"


def _outputs(d, outcomes):
    (d / f"{VID}_segments.json").write_text(json.dumps(
        {"overall_confidence": 0.9,
         "boundaries": [100 * i for i in range(len(outcomes) + 1)]}))
    (d / f"{VID}_reaches.json").write_text(json.dumps({"segments": []}))
    (d / f"{VID}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": i + 1, "outcome": oc} for i, oc in enumerate(outcomes)]}))


@pytest.fixture
def server(tmp_path, monkeypatch):
    proc = tmp_path / "Processing"
    proc.mkdir()
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(Paths, "PROCESSING", proc)
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.processing_dir = proc
    o._hand_score_precheck = lambda vid, d: True
    o._held_for_hand_scores = lambda vid: False
    o.db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete", "processing"):
        o.db.update_state(VID, st)
    (proc / f"{VID}.mp4").write_bytes(b"video")
    (proc / POSE).write_bytes(b"pose")
    return o, proc, triage


def _run(o, proc, outcomes, assign):
    def seg(dlc):
        _outputs(proc, outcomes)
        return {"success": True, "status": "ok", "n_boundaries": 3}

    class _Triage:
        verdict = "auto_approved"; flags = []; n_critical = 0; n_warnings = 0

        def save(self, p):
            Path(p).write_text("{}")

    with mock.patch("mousereach.segmentation.core.batch.process_single", seg), \
            mock.patch("mousereach.reach.core.batch.process_single",
                       return_value={"total_reaches": 0}), \
            mock.patch("mousereach.outcomes.core.batch.process_single",
                       return_value={"n_segments": len(outcomes)}), \
            mock.patch("mousereach.assignment.run.assign_reaches_for_video", assign), \
            mock.patch("mousereach.pipeline.manifest.create_processing_manifest",
                       return_value={}), \
            mock.patch("mousereach.pipeline.triage.triage_video",
                       return_value=_Triage()):
        o._run_pipeline({"id": VID, "data": {"dlc_output_path": str(proc / POSE)}})


def _crash(*a, **k):
    raise RuntimeError("assignment crashed")


def test_a_failed_assignment_holds_the_video_in_triage(server):
    o, proc, triage = server
    _run(o, proc, ["retrieved", "untouched"], _crash)
    assert o.db.get_video(VID)["state"] == "triage"
    routing = json.loads((triage / VID / f"{VID}_routing.json").read_text())
    assert "reach assignment could not run" in routing["routed_reason"]
    assert "assignment crashed" in routing["routed_reason"]
    assert not (proc / f"{VID}_features.json").exists()       # no kinematics


def test_inputs_it_could_not_read_hold_it_too(server):
    o, proc, triage = server
    _run(o, proc, ["displaced_sa"], lambda *a, **k: None)
    assert o.db.get_video(VID)["state"] == "triage"


def test_the_placeholder_asks_only_about_touched_pellets(tmp_path):
    _outputs(tmp_path, ["retrieved", "untouched", "displaced_outside"])
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = mock.Mock()
    o._hold_for_failed_assignment(VID, tmp_path, "boom")
    doc = json.loads((tmp_path / f"{VID}_reach_assignments.json").read_text())
    assert doc["reaches"] == [] and doc[ASSIGNMENT_FAILED_KEY] == "boom"
    assert triage_status(tmp_path, VID).unresolved == {1, 3}
    decision, reason, _ = evaluate_gate(VID, tmp_path)
    assert decision == DECISION_TRIAGE and "segment(s) 1, 3" in reason


def test_nothing_touched_means_nothing_to_ask(tmp_path):
    _outputs(tmp_path, ["untouched", "untouched"])
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = mock.Mock()
    o._hold_for_failed_assignment(VID, tmp_path, "boom")
    decision, _, _ = evaluate_gate(VID, tmp_path)
    assert decision == DECISION_CLEAN


def test_an_older_assignment_file_is_replaced(tmp_path):
    """An earlier run's file describes outcomes this run recomputed."""
    _outputs(tmp_path, ["retrieved"])
    (tmp_path / f"{VID}_reach_assignments.json").write_text(json.dumps(
        {"reaches": [{"segment_num": 1, "is_causal": True}]}))
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = mock.Mock()
    o._hold_for_failed_assignment(VID, tmp_path, "boom")
    assert triage_status(tmp_path, VID).unresolved == {1}


@pytest.fixture(autouse=True)
def _the_stand_in_pose_counts_as_usable(monkeypatch):
    """These tests use a stand-in pose file that is not a real pose table;
    the pose check before segmentation (pipeline/pose_check.py) is not what
    they test."""
    try:
        from mousereach.pipeline import pose_check
    except ImportError:         # code without the pose check: nothing to stand in for
        return
    monkeypatch.setattr(pose_check, "check_pose", lambda *a, **k: (True, ""))
