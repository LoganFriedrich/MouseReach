"""A filed video re-run only because a downstream calculation changed re-runs
none of the four algorithms, no quality triage, no hand-score comparison and
-- since the lab decision of 2026-11-19 ('triage has nothing to do with
kinematics'; filed is final) -- no review gate either
(orchestrator.filed_rerun_reuses_analysis). An output it would have to
recompute ends the re-run instead, and the filed results stand.

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
    # The outcome step needs the video beside the pose (it is fetched, or the
    # video is unresolvable): a Pillar run without it never reaches the gate.
    (folder / f"{VID}.mp4").write_bytes(b"x")
    monkeypatch.setattr(o, "resolve_pose_input", lambda *a, **k: pose)
    # The pose stand-in is an empty file; the pose test is not what is tested.
    import mousereach.pipeline.pose_check as pc
    monkeypatch.setattr(pc, "check_pose", lambda *a, **k: (True, ""))
    # The video's filed folder, here empty: a kinematics-only re-run looks
    # there for the filed features it must keep (test_kinematics_only_rerun_
    # keeps_filed_decisions.py), and with none filed there is nothing to keep.
    import mousereach.archive.core as arch
    monkeypatch.setattr(arch, "get_archive_destination",
                        lambda vid: tmp_path / "no_filed_features_here")

    def _stage(name, result):
        def run(*a, **k):
            calls.append(name)
            return result
        return run

    monkeypatch.setattr(segb, "process_single", _stage("segmentation", {"success": True}))
    monkeypatch.setattr(reachb, "process_single", _stage("reach", {"total_reaches": 0}))
    monkeypatch.setattr(outb, "process_single", _stage("outcome", {"n_segments": 0}))
    monkeypatch.setattr(assign, "assign_reaches_for_video", _stage("assignment", None))
    def _manifest(**k):
        # A real manifest file of this run: filing requires one.
        from pathlib import Path as _P
        m = _P(k["processing_dir"]) / f"{k['video_id']}_processing_manifest.json"
        if not m.exists():
            m.write_text("{}")
        return {}
    monkeypatch.setattr(man, "create_processing_manifest", _manifest)
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
    restored = []
    orch._restore_filed_after_failed_rerun = lambda vid, why: (
        calls.append("restored"), restored.append(why))

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
                            "file_it": staticmethod(file_it), "restored": restored})


def test_a_filed_kinematics_only_rerun_runs_no_algorithm_triage_or_hand_score_check(rig):
    rig.file_it()
    calls = rig.run("kinematics")
    assert calls == ["kinematics"]                    # no gate either (2026-11-19)
    assert rig.orch.db.get_video(VID)["state"] == "processed"


def test_a_filed_kinematics_only_rerun_is_extracted_and_processed_without_the_gate(rig):
    rig.file_it()
    calls = rig.run("kinematics")
    assert calls == ["kinematics"]
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


def test_a_kinematics_rerun_whose_reused_output_is_missing_is_given_up(rig):
    """A filed video's algorithms are never re-run (filed is final): a reused
    output that is not there ends the re-run, and the filed results stand."""
    rig.file_it()
    (rig.folder / f"{VID}_pellet_outcomes.json").unlink()
    calls = rig.run("kinematics")
    assert calls == ["restored"]
    assert "pellet_outcomes" in rig.restored[0]


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


def test_a_failed_kinematics_rerun_retried_runs_only_kinematics(rig):
    rig.file_it()
    _mark_kinematics_rerun(rig)
    rig.orch.db.mark_failed(VID, "kinematics failed")
    rig.orch.db.update_state(VID, "processing")      # what the retry bucket does
    calls = _run_from_row(rig)
    assert calls == ["kinematics"]


def test_a_kinematics_rerun_picked_after_a_restart_runs_only_kinematics(rig):
    rig.file_it()
    _mark_kinematics_rerun(rig)
    assert _run_from_row(rig) == ["kinematics"]


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
    """A run with checks whose gate finds stale outputs clears a kept scope.
    (A kinematics-only re-run of a filed video runs no gate at all.)"""
    import mousereach.watcher.review_gate as rg

    def _stale(*a, **k):
        rig.calls.append("gate")
        return rg.DECISION_STALE
    monkeypatch.setattr(rg, "run_gate", _stale)
    rig.orch.db.set_fields(VID, reprocess_scope="kinematics")   # never filed
    rig.orch.db.force_state(VID, "processing", reason="test")
    _run_from_row(rig)
    assert rig.orch.db.get_video(VID)["reprocess_scope"] is None


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
    (rig.folder / f"{VID}_processing_manifest.json").write_text("{}", encoding="utf-8")
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


def test_the_kinematics_step_stamps_the_ground_truth_the_extractor_applied(rig, monkeypatch):
    """The manifest records the ground truth the extractor really applied --
    not merely the file that sat beside the run -- so the version scan can
    tell when one is saved or changed after filing, and never takes a file the
    extractor missed for one it applied (pipeline.manifest.ground_truth_applied)."""
    import mousereach.pipeline.manifest as man
    import mousereach.watcher.review_gate as rg
    import mousereach.kinematics.core.feature_extractor as fe
    seen = {}
    monkeypatch.setattr(man, "record_kinematic_version",
                        lambda *a, **k: seen.update(k))
    monkeypatch.setattr(rg, "run_gate", lambda *a, **k: rg.DECISION_CLEAN)
    beside = rig.folder / f"{VID}_unified_ground_truth.json"
    beside.write_text("{}", encoding="utf-8")
    elsewhere = rig.folder.parent / "elsewhere" / beside.name
    base = fe.FeatureExtractor

    class _Applies(base):
        def extract(self, *a, **k):
            self.applied_ground_truth = elsewhere      # what its lookup found
            return base.extract(self, *a, **k)
    monkeypatch.setattr(fe, "FeatureExtractor", _Applies)
    rig.file_it()
    rig.run("kinematics")
    assert seen.get("ground_truth_path") == elsewhere   # not the file beside the run



# ---------------------------------------------------------------------------
# A kept scope never outlives a recompute
# ---------------------------------------------------------------------------
# WHY: the evidence that a run recomputed an algorithm output lives only inside
# that run. Read back as 'kinematics' on the next pick, the scope would reuse
# the freshly recomputed output and skip the quality triage and the hand-score
# comparison for it -- after a restart while the video waited for its hand
# scores, or after a failure between the recompute and the checks. The scope
# is narrowed to the recomputed stage before that stage runs
# (orchestrator.scope_for_recomputed).

def test_a_recomputed_output_held_for_hand_scores_is_checked_after_a_restart(rig):
    rig.file_it()
    _mark_kinematics_rerun(rig)
    (rig.folder / f"{VID}_reach_assignments.json").unlink()

    def _hold_once(video_id, folder_):
        rig.calls.append("hand_scores")
        # what the real assignment step would have left behind
        (rig.folder / f"{VID}_reach_assignments.json").write_text("{}", encoding="utf-8")
        return False                                   # held: scores not in yet
    rig.orch._hand_score_precheck = _hold_once
    calls = _run_from_row(rig)
    assert calls == ["assignment", "triage", "hand_scores"]
    assert rig.orch.db.get_video(VID)["reprocess_scope"] == "outcome"

    rig.orch._score_waits = {}                          # a watcher restart
    calls = _run_from_row(rig)
    assert "triage" in calls and "hand_scores" in calls


def test_a_kept_kinematics_scope_with_a_missing_output_recomputes_nothing(rig, monkeypatch):
    """Under a kept 'kinematics' scope a missing reaches file used to be
    recomputed (and judged on retry). Filed is final: nothing is recomputed."""
    import mousereach.reach.core.batch as reachb
    rig.file_it()
    _mark_kinematics_rerun(rig)
    (rig.folder / f"{VID}_reaches.json").unlink()

    def _reach_writes_then_crashes(*a, **k):
        rig.calls.append("reach")
        (rig.folder / f"{VID}_reaches.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("reach step crashed after writing")
    monkeypatch.setattr(reachb, "process_single", _reach_writes_then_crashes)
    # Filed is final (2026-11-19): the missing reaches file is never
    # recomputed for a filed video; the re-run is given up instead.
    calls = _run_from_row(rig)
    assert calls == ["restored"]


def test_the_narrowed_scope_is_the_earliest_recomputed_stage():
    f = o.scope_for_recomputed
    assert f(["assignment"]) == "outcome"
    assert f(["reach", "outcome", "assignment"]) == "reach"
    assert f(["segmentation"]) == "segmentation"
    assert f([]) == "segmentation"


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
