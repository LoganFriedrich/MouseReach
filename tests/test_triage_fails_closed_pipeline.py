"""The quality triage failing closed, seen through the pipeline's own checks.

Runs the processing server's checks before kinematics with a triage that
raises and the real review gate: the video must end up held in deep review,
not pass as clean, and an earlier run's ``_triage.json`` must be gone. (Before
the fix the verdict stayed 'auto_approved', the gate said clean, and the old
record stayed beside the new outputs.) Built under tmp_path.
"""
import json

import pytest

from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import ProcessingOrchestrator
from mousereach.watcher.review_gate import DECISION_CLEAN, run_gate

VID = "20990101_ABC9901_P1"


def _crash(**kw):
    raise RuntimeError("triage crashed")


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
    o.db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete", "processing"):
        o.db.update_state(VID, st)
    (proc / f"{VID}_segments.json").write_text(json.dumps(
        {"overall_confidence": 0.9, "boundaries": [0, 100]}))
    (proc / f"{VID}_pellet_outcomes.json").write_text(json.dumps(
        {"segments": [{"segment_num": 1, "outcome": "untouched"}]}))
    (proc / f"{VID}.mp4").write_bytes(b"video")
    (proc / f"{VID}_triage.json").write_text(json.dumps({"verdict": "auto_approved"}))
    return o, proc, tmp_path


def test_a_crashed_triage_holds_the_video(env):
    o, proc, tmp_path = env
    held = o._run_checks_before_kinematics(VID, proc / "pose.h5", 1.0, _crash,
                                           run_gate, DECISION_CLEAN)
    assert held is True
    assert o.db.get_video(VID)["state"] == "deep_review"
    bundle = tmp_path / "Review" / "deep_review" / VID
    assert bundle.is_dir()
    # The earlier run's verdict did not travel with the held video.
    assert not (bundle / f"{VID}_triage.json").exists()
    assert not (proc / f"{VID}_triage.json").exists()
