"""A filed video re-run only because a downstream calculation changed re-runs
none of the four algorithms, no quality triage and no hand-score comparison
(orchestrator.filed_rerun_reuses_analysis). The review gate still runs, and it
can only ask about a segment no person has answered -- ground truth beside the
video counts as an answer (triage_status.gt_resolved_segments).

WHY (Logan, 2026-11-03): "What should happen is that only what calculations
changed should be recalculated without rerunning the 4 step / triage algo
parts." And: "Any segment whose boundaries and causal reach and outcome have
all already been reviewed by a human should never under any condition require
a human to look at it ever again."

The other side is tested too: a new video, a video back from review, and a
filed video whose re-run recomputed an upstream stage all still go through
every check.
"""
import json

import pytest

import mousereach.watcher.orchestrator as o
from mousereach.watcher.db import WatcherDB

VID = "20250709_ABC0207_P1"


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------

def test_the_rule_matches_only_a_filed_rerun_that_recomputed_nothing():
    f = o.filed_rerun_reuses_analysis
    assert f("kinematics", "2026-10-01T10:00:00", []) is True
    # a new video, a review return, a held video picked again: no start stage
    assert f(None, "2026-10-01T10:00:00", []) is False
    assert f(None, None, []) is False
    # never filed
    assert f("kinematics", None, []) is False
    # an upstream stage was recomputed: the checks judge it
    for stage in ("segmentation", "reach", "outcome", "assignment"):
        assert f("kinematics", "2026-10-01T10:00:00", [stage]) is False
    assert f("segmentation", "2026-10-01T10:00:00", ["segmentation", "reach"]) is False


# ---------------------------------------------------------------------------
# The pipeline run
# ---------------------------------------------------------------------------

@pytest.fixture
def rig(tmp_path, monkeypatch):
    """A processing folder holding a filed video's reused outputs, every
    algorithm replaced by a recorder, and every check replaced by one that
    records it ran (and holds the video, so a check that runs is visible)."""
    import mousereach.segmentation.core.batch as segb
    import mousereach.reach.core.batch as reachb
    import mousereach.outcomes.core.batch as outb
    import mousereach.assignment.run as assign
    import mousereach.pipeline.manifest as man
    import mousereach.pipeline.triage as tri
    import mousereach.watcher.review_gate as rg
    import mousereach.kinematics.core.feature_extractor as fe

    calls = []
    folder = tmp_path / "work"
    folder.mkdir()
    for name in ("_segments.json", "_reaches.json", "_pellet_outcomes.json",
                 "_reach_assignments.json"):
        (folder / f"{VID}{name}").write_text("{}", encoding="utf-8")
    pose = folder / f"{VID}DLC_test.h5"
    pose.write_bytes(b"")
    monkeypatch.setattr(o, "resolve_pose_input", lambda *a, **k: pose)

    def _stage(name, result):
        def run(*a, **k):
            calls.append(name)
            return result
        return run

    monkeypatch.setattr(segb, "process_single", _stage("segmentation", {"success": True}))
    monkeypatch.setattr(reachb, "process_single", _stage("reach", {"total_reaches": 0}))
    monkeypatch.setattr(outb, "process_single", _stage("outcome", {"n_segments": 0}))
    monkeypatch.setattr(assign, "assign_reaches_for_video", _stage("assignment", None))
    monkeypatch.setattr(man, "create_processing_manifest", lambda **k: {})
    monkeypatch.setattr(man, "record_kinematic_version", lambda *a, **k: None)

    class _Triage:
        verdict = "needs_review"
        flags = []
        n_critical = 1
        n_warnings = 0

        def save(self, path):
            pass

    def _triage(**k):
        calls.append("triage")
        return _Triage()
    monkeypatch.setattr(tri, "triage_video", _triage)

    def _gate(*a, **k):
        calls.append("gate")
        return rg.DECISION_TRIAGE
    monkeypatch.setattr(rg, "run_gate", _gate)

    class _Features:
        n_segments = 0

        def to_dict(self):
            return {"extractor": "new"}

    class _Extractor:
        VERSION = "9.9.9"

        def extract(self, *a, **k):
            calls.append("kinematics")
            return _Features()
    monkeypatch.setattr(fe, "FeatureExtractor", _Extractor)

    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = WatcherDB(tmp_path / "local.db")
    orch.db.register_video(video_id=VID, source_path="x")
    orch.processing_dir = folder
    orch.hostname = "HOST"
    orch._score_waits = {}

    def _precheck(video_id, folder_):
        calls.append("hand_scores")
        return True
    orch._hand_score_precheck = _precheck

    def file_it():
        """Bring the row to 'archived' the way the archive step does."""
        orch.db.force_state(VID, "processing", reason="test fixture")
        orch.db.update_state(VID, "processed")
        orch.db.update_state(VID, "archiving")
        orch.db.update_state(VID, "archived")

    def run(start_stage=None):
        data = {"video_id": VID, "dlc_output_path": str(pose)}
        if start_stage:
            orch.db.force_state(VID, "outdated", reason="test",
                                reprocess_scope=start_stage)
            orch.db.force_state(VID, "processing", reason="test re-run")
            data["reprocess_start_stage"] = start_stage
        else:
            orch.db.force_state(VID, "processing", reason="test fixture")
        calls.clear()
        orch._run_pipeline({"id": VID, "data": data})
        return list(calls)

    return type("Rig", (), {"orch": orch, "folder": folder, "run": staticmethod(run), "calls": calls,
                            "file_it": staticmethod(file_it)})


def test_a_filed_kinematics_only_rerun_runs_no_algorithm_triage_or_hand_score_check(rig):
    rig.file_it()
    calls = rig.run("kinematics")
    assert calls == ["gate"]                          # only the gate; it held here
    assert "kinematics" not in calls                  # held: nothing filed


def test_a_filed_kinematics_only_rerun_the_gate_passes_is_extracted_and_processed(rig, monkeypatch):
    import mousereach.watcher.review_gate as rg

    def _clean(*a, **k):
        rig.calls.append("gate")
        return rg.DECISION_CLEAN
    monkeypatch.setattr(rg, "run_gate", _clean)
    rig.file_it()
    calls = rig.run("kinematics")
    assert calls == ["gate", "kinematics"]
    assert rig.orch.db.get_video(VID)["state"] == "processed"   # on to filing
    feats = json.loads((rig.folder / f"{VID}_features.json").read_text())
    assert feats == {"extractor": "new"}


def test_a_new_video_goes_through_every_check(rig):
    calls = rig.run(None)
    assert calls == ["segmentation", "reach", "outcome", "assignment",
                     "triage", "hand_scores", "gate"]
    assert "kinematics" not in calls                  # the gate held it
    assert rig.orch.db.get_video(VID)["state"] == "processing"


def test_a_video_back_from_review_goes_through_every_check_even_if_filed_before(rig):
    """A review return re-enters 'processing' with no start stage: a full run,
    and the gate confirms the answers resolved everything."""
    rig.file_it()
    calls = rig.run(None)
    assert "gate" in calls and "triage" in calls and "hand_scores" in calls


def test_a_filed_rerun_that_recomputes_an_upstream_stage_goes_through_every_check(rig):
    rig.file_it()
    calls = rig.run("outcome")
    assert calls[:2] == ["outcome", "assignment"]
    assert "segmentation" not in calls and "reach" not in calls
    assert calls[2:] == ["triage", "hand_scores", "gate"]


def test_a_kinematics_rerun_whose_reused_output_is_missing_is_checked(rig):
    """A reused output that is not there is made again -- and then judged."""
    rig.file_it()
    (rig.folder / f"{VID}_pellet_outcomes.json").unlink()
    calls = rig.run("kinematics")
    assert "outcome" in calls and "gate" in calls


def test_a_kinematics_rerun_of_a_never_filed_video_is_checked(rig):
    calls = rig.run("kinematics")
    assert calls == ["triage", "hand_scores", "gate"]
