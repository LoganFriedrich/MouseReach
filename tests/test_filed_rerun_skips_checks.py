"""A filed video re-run only because a downstream calculation changed re-runs
none of the four algorithms, no quality triage and no hand-score comparison
(orchestrator.filed_rerun_reuses_analysis). The review gate still runs, and it
can only ask about a segment no person has answered -- ground truth beside the
video counts as an answer (triage_status.gt_resolved_segments).

WHY: only the calculations that changed are recalculated, without re-running
the four algorithms or triage. And a segment whose boundaries, causal reach
and outcome have all already been reviewed by a person never requires a
person to look at it again.

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


# ---------------------------------------------------------------------------
# The scope is kept on the row: a retry or a restart resumes it
# ---------------------------------------------------------------------------
# WHY: the start stage used to travel only inside the work item. The failed
# retry and a restart picked the bare row and ran a filed video through all
# four algorithms, the quality triage and the hand-score comparison again
# (orchestrator.kept_rerun_start_stage).

def _run_from_row(rig):
    """Run the video the way the retry bucket and the pipeline bucket do:
    the bare row, no start stage in the work item."""
    rig.calls.clear()
    row = rig.orch.db.get_video(VID)
    rig.orch._run_pipeline({"id": VID, "data": dict(row, state="processing")})
    return list(rig.calls)


def _mark_kinematics_rerun(rig):
    rig.orch.db.force_state(VID, "outdated", reason="test",
                            reprocess_scope="kinematics")
    rig.orch.db.force_state(VID, "processing", reason="test re-run")


def test_a_failed_kinematics_rerun_retried_runs_only_the_gate(rig):
    rig.file_it()
    _mark_kinematics_rerun(rig)
    rig.orch.db.mark_failed(VID, "kinematics failed")
    rig.orch.db.update_state(VID, "processing")      # what the retry bucket does
    calls = _run_from_row(rig)
    assert calls == ["gate"]


def test_a_kinematics_rerun_picked_after_a_restart_runs_only_the_gate(rig):
    rig.file_it()
    _mark_kinematics_rerun(rig)
    assert _run_from_row(rig) == ["gate"]


def test_a_kept_scope_of_an_upstream_stage_is_a_full_run(rig):
    """Only 'kinematics' is read back: an upstream scope recomputes an
    algorithm, and every check runs on it anyway."""
    rig.file_it()
    rig.orch.db.force_state(VID, "outdated", reason="test", reprocess_scope="outcome")
    rig.orch.db.force_state(VID, "processing", reason="test re-run")
    calls = _run_from_row(rig)
    assert calls[0] == "segmentation" and "triage" in calls and "hand_scores" in calls


def test_a_video_held_for_hand_scores_is_checked_again(rig):
    rig.file_it()
    _mark_kinematics_rerun(rig)
    rig.orch._score_waits = {VID: {"stamp": None}}
    calls = _run_from_row(rig)
    assert "triage" in calls and "hand_scores" in calls


def test_stale_outputs_clear_the_kept_scope_so_the_next_pick_recomputes(rig, monkeypatch):
    import mousereach.watcher.review_gate as rg

    def _stale(*a, **k):
        rig.calls.append("gate")
        return rg.DECISION_STALE
    monkeypatch.setattr(rg, "run_gate", _stale)
    rig.file_it()
    _mark_kinematics_rerun(rig)
    assert _run_from_row(rig) == ["gate"]
    assert rig.orch.db.get_video(VID)["reprocess_scope"] is None
    calls = _run_from_row(rig)
    assert calls[:4] == ["segmentation", "reach", "outcome", "assignment"]


def test_a_review_return_clears_the_kept_scope_and_the_next_run_is_checked(rig, tmp_path, monkeypatch):
    import mousereach.watcher.review_return as rr
    triage = tmp_path / "Review" / "triage"
    bundle = triage / VID
    bundle.mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", tmp_path / "Review" / "deep_review")
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", tmp_path / "Review")
    monkeypatch.setattr(rr.Paths, "ANALYZED_OUTPUT", None, raising=False)
    monkeypatch.setattr(rr.Paths, "DLC_STAGING", None, raising=False)
    monkeypatch.setattr(rr, "_ensure_durable_review", lambda *a, **k: None)
    (bundle / f"{VID}.mp4").write_bytes(b"x")
    (bundle / f"{VID}DLC_resnet50_ModelAshuffle1_100.h5").write_bytes(b"x")
    rig.file_it()
    rig.orch.db.force_state(VID, "triage", reason="test", reprocess_scope="kinematics")
    assert rr._return_to_processing(bundle, VID, rig.folder, rig.orch.db, "triage_cleared")
    assert rig.orch.db.get_video(VID)["reprocess_scope"] is None
    calls = _run_from_row(rig)
    assert "triage" in calls and "hand_scores" in calls


def test_filing_clears_the_kept_scope(rig, tmp_path, monkeypatch):
    from unittest import mock
    from mousereach.config import Paths
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    monkeypatch.setattr(Paths, "PROCESSING", rig.folder)
    monkeypatch.setattr("mousereach.archive.core.get_archive_destination",
                        lambda vid: tmp_path / "Analyzed" / "PROJ" / "COHORT")
    orch = rig.orch
    orch.db.export_to_central_db = lambda vid: None
    orch.staging_dir = None
    orch._release_claim = lambda vid: None
    (rig.folder / f"{VID}_features.json").write_text("{}", encoding="utf-8")
    (rig.folder / f"{VID}.mp4").write_bytes(b"x")
    rig.file_it()
    orch.db.force_state(VID, "processed", reason="test", reprocess_scope="kinematics")
    with mock.patch("mousereach.archive.core.archive_video",
                    lambda video_id, **kw: {"success": True, "files_moved": ["a"],
                                            "destination": "x"}):
        o.ProcessingOrchestrator._archive_to_nas(orch, {"id": VID})
    row = orch.db.get_video(VID)
    assert row["state"] == "archived" and row["reprocess_scope"] is None


def test_the_kinematics_step_stamps_the_ground_truth_beside_the_run(rig, monkeypatch):
    """The manifest records which ground-truth file was beside the run, so
    the version scan can tell when one is saved or changed after filing
    (pipeline.manifest.ground_truth_applied)."""
    import mousereach.pipeline.manifest as man
    import mousereach.watcher.review_gate as rg
    seen = {}
    monkeypatch.setattr(man, "record_kinematic_version",
                        lambda *a, **k: seen.update(k))
    monkeypatch.setattr(rg, "run_gate", lambda *a, **k: rg.DECISION_CLEAN)
    gt = rig.folder / f"{VID}_unified_ground_truth.json"
    gt.write_text("{}", encoding="utf-8")
    rig.file_it()
    rig.run("kinematics")
    assert seen.get("ground_truth_path") == gt
