"""A video a review queue returns resumes just after the step that sent it,
and every later step runs (watcher/resume_point.py).

WHY: a returned video used to re-enter at segmentation. All four algorithms
were run again on a video a person had just cleared -- extra ways to fail it.
Now each hold records WHICH STEP sent the video;
the return resumes just after it; the quality triage, the hand-score comparison
and the review gate always run again (they come after every algorithm step).
An earlier step's output is reused only when its version and pose match what
is declared now; otherwise the run starts at the earliest out-of-date step.

Built under tmp_path with invented ids; no real folder or database.

The rule's module (watcher/resume_point.py) is imported inside the tests that
use it, so the pipeline-run tests below fail on code without it by what the
run does, not by an import error.
"""
import json

import pytest

import mousereach.watcher.orchestrator as o
from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB

VID = "20990101_ABC0501_P1"
SCORER = "DLC_resnet50_TestModelshuffle1_100"
POSE = f"{VID}{SCORER}.h5"
VERSIONS = {"segmenter": "2.0", "reach_detector": "3.0", "outcome_detector": "4.0",
            "assignment": "1.0", "kinematic_extractor": "5.0", "dlc_scorer": SCORER}
OUTPUTS = {"_segments.json": ("segmenter_version", "2.0"),
           "_reaches.json": ("detector_version", "3.0"),
           "_pellet_outcomes.json": ("detector_version", "4.0"),
           "_reach_assignments.json": ("version", "1.0")}


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def test_each_step_resumes_just_after_itself():
    from mousereach.watcher import resume_point as rp
    assert rp.resume_stage_for("outcome", "triage") == "assignment"
    assert rp.resume_stage_for("assignment", "triage") == "checks"
    assert rp.resume_stage_for("segmentation", "deep_review") == "reach"


def test_deep_review_always_resumes_after_segmentation():
    """Deep review is where cuts are fixed; everything made from the cuts is remade."""
    from mousereach.watcher import resume_point as rp
    assert rp.resume_stage_for("assignment", "deep_review") == "reach"
    assert rp.resume_stage_for(None, "deep_review") == "reach"


def test_an_unrecorded_step_resumes_after_the_step_its_queue_stands_for():
    from mousereach.watcher import resume_point as rp
    assert rp.resume_stage_for(None, "triage") == "assignment"
    assert rp.resume_stage_for(None, None) == "segmentation"


# ---------------------------------------------------------------------------
# The pipeline run
# ---------------------------------------------------------------------------

@pytest.fixture
def rig(tmp_path, monkeypatch):
    """A processing folder holding a returned video's current outputs and its
    processing manifest, every algorithm and check replaced by a recorder.
    The gate holds the video (so a gate that runs is visible); the queues are
    under tmp_path."""
    # The pose stand-in is an empty file; the pose test is not what is tested
    # (an unusable pose of a video not filed starts it over, 2026-11-19).
    import mousereach.pipeline.pose_check as _pc
    monkeypatch.setattr(_pc, "check_pose", lambda *a, **k: (True, ""))
    import mousereach.segmentation.core.batch as segb
    import mousereach.reach.core.batch as reachb
    import mousereach.outcomes.core.batch as outb
    import mousereach.assignment.run as assign
    import mousereach.pipeline.manifest as man
    import mousereach.pipeline.triage as tri
    import mousereach.watcher.review_gate as rg

    nas = tmp_path / "nas"
    nas.mkdir()
    (nas / "pipeline_versions.json").write_text(json.dumps({"versions": VERSIONS}),
                                                encoding="utf-8")
    monkeypatch.setattr(Paths, "NAS_ROOT", nas, raising=False)
    review = tmp_path / "Review"
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", review / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", review / "deep_review")
    monkeypatch.setattr(Paths, "REVIEW_ROOT", review, raising=False)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", None, raising=False)
    monkeypatch.setattr(Paths, "DLC_STAGING", None, raising=False)

    calls = []
    folder = tmp_path / "work"
    folder.mkdir()

    def write_outputs(where, **versions):
        for suffix, (key, v) in OUTPUTS.items():
            (where / f"{VID}{suffix}").write_text(
                json.dumps({key: versions.get(suffix, v)}), encoding="utf-8")
        (where / f"{VID}_processing_manifest.json").write_text(json.dumps({
            "video_id": VID, "dlc_model": {"dlc_scorer": versions.get("pose", SCORER)}}),
            encoding="utf-8")
        (where / POSE).write_bytes(b"x")
        (where / f"{VID}.mp4").write_bytes(b"x")

    def _stage(name, result):
        def run(*a, **k):
            calls.append(name)
            return result
        return run

    monkeypatch.setattr(segb, "process_single", _stage("segmentation", {"success": True}))
    monkeypatch.setattr(reachb, "process_single", _stage("reach", {"total_reaches": 0}))
    monkeypatch.setattr(outb, "process_single", _stage("outcome", {"n_segments": 0}))
    monkeypatch.setattr(assign, "assign_reaches_for_video", _stage("assignment", {}))
    def _manifest(**k):
        # A real manifest file of this run: filing requires one.
        from pathlib import Path as _P
        m = _P(k["processing_dir"]) / f"{k['video_id']}_processing_manifest.json"
        if not m.exists():
            m.write_text("{}")
        return {}
    monkeypatch.setattr(man, "create_processing_manifest", _manifest)

    class _Triage:
        verdict = "auto_approved"
        flags = []
        n_critical = 0
        n_warnings = 0

        def save(self, path):
            pass

    monkeypatch.setattr(tri, "triage_video", lambda **k: calls.append("triage") or _Triage())

    def _gate(*a, **k):
        calls.append("gate")
        return rg.DECISION_TRIAGE
    monkeypatch.setattr(rg, "run_gate", _gate)

    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = WatcherDB(tmp_path / "local.db")
    orch.db.register_video(video_id=VID, source_path="x")
    orch.processing_dir = folder
    orch.hostname = "HOST"
    orch._score_waits = {}
    orch._hand_score_precheck = lambda vid, f: calls.append("hand_scores") or True

    def run_from_row():
        calls.clear()
        row = orch.db.get_video(VID)
        orch._run_pipeline({"id": VID, "data": dict(row, state="processing")})
        return list(calls)

    return type("Rig", (), {"orch": orch, "folder": folder, "calls": calls,
                            "review": review, "write_outputs": staticmethod(write_outputs),
                            "run_from_row": staticmethod(run_from_row)})


def _return_from(rig, queue, held_at_step=None, monkeypatch=None, before_route=None,
                 **versions):
    """Put the video in ``queue`` the way a route does (row + bundle manifest),
    then return it the way the return scan does once a person has answered.
    ``before_route(src)`` may change the routed outputs first."""
    import mousereach.watcher.review_return as rr
    from mousereach.watcher.review_gate import route_to_queue
    src = rig.folder.parent / "routed_from"
    src.mkdir(exist_ok=True)
    rig.write_outputs(src, **versions)
    if before_route is not None:
        before_route(src)
    rig.orch.db.force_state(VID, "processing", reason="test: first pass")
    dest = Paths.DEEP_REVIEW if queue == "deep_review" else Paths.TRIAGE_REVIEW
    # held_at_step only when a test names one: code without the record does
    # not take the argument, and the run's behaviour is what is under test.
    extra = {"held_at_step": held_at_step} if held_at_step is not None else {}
    bundle = route_to_queue(VID, src, dest, "test hold", db=rig.orch.db,
                            db_state=queue, **extra)
    assert rig.orch.db.get_video(VID)["state"] == queue
    if monkeypatch is not None:
        monkeypatch.setattr(rr, "_ensure_durable_review", lambda *a, **k: None)
    reason = "deep_review_cleared" if queue == "deep_review" else "triage_cleared"
    assert rr._return_to_processing(bundle, VID, rig.folder, rig.orch.db, reason)


def test_a_triage_return_resumes_after_the_outcomes_with_every_check(rig, monkeypatch):
    _return_from(rig, "triage", monkeypatch=monkeypatch)
    calls = rig.run_from_row()
    assert calls == ["assignment", "triage", "hand_scores", "gate"]


def test_a_return_after_a_failed_assignment_resumes_at_the_checks(rig, monkeypatch):
    _return_from(rig, "triage", held_at_step="assignment", monkeypatch=monkeypatch)
    calls = rig.run_from_row()
    assert calls == ["triage", "hand_scores", "gate"]


def test_a_return_after_a_failed_assignment_keeps_the_hold_placeholder(rig, monkeypatch):
    """The real case: the assignment file in the bundle is the placeholder the
    hold writes when reach assignment could not run (no version in it). The
    person's causal-reach answers replace it, so it is kept, not remade --
    re-running assignment would redo the step that sent the video."""
    def hold(src):
        rig.orch._hold_for_failed_assignment(VID, src, "its inputs could not be read")
        doc = json.loads((src / f"{VID}_reach_assignments.json").read_text(encoding="utf-8"))
        assert "version" not in doc          # what the hold really writes
    _return_from(rig, "triage", held_at_step="assignment", monkeypatch=monkeypatch,
                 before_route=hold)
    calls = rig.run_from_row()
    assert calls == ["triage", "hand_scores", "gate"]


def test_the_placeholder_is_kept_only_when_resuming_at_the_checks(tmp_path):
    """A placeholder still means reach assignment has to run when the return
    resumes at assignment or earlier (a triage return after the outcomes)."""
    from mousereach.watcher import resume_point as rp
    from mousereach.watcher.review_gate import ASSIGNMENT_FAILED_KEY
    for suffix, (key, v) in OUTPUTS.items():
        (tmp_path / f"{VID}{suffix}").write_text(json.dumps({key: v}), encoding="utf-8")
    (tmp_path / f"{VID}_reach_assignments.json").write_text(json.dumps(
        {"video_id": VID, "reaches": [], ASSIGNMENT_FAILED_KEY: "x"}), encoding="utf-8")
    (tmp_path / f"{VID}_processing_manifest.json").write_text(json.dumps(
        {"dlc_model": {"dlc_scorer": SCORER}}), encoding="utf-8")
    declared = {"versions": VERSIONS}
    assert rp.current_start(tmp_path, VID, "checks", declared, SCORER) == ("checks", "")
    assert rp.current_start(tmp_path, VID, "assignment", declared, SCORER)[0] == "assignment"


def test_a_deep_review_return_remakes_everything_after_segmentation(rig, monkeypatch):
    _return_from(rig, "deep_review", monkeypatch=monkeypatch)
    calls = rig.run_from_row()
    assert calls == ["reach", "outcome", "assignment", "triage", "hand_scores", "gate"]


def test_an_out_of_date_output_moves_the_start_to_its_step(rig, monkeypatch):
    """The outcome detector's declared version changed while the video waited:
    its outputs are remade, not reused under the person's answers."""
    _return_from(rig, "triage", monkeypatch=monkeypatch, **{"_pellet_outcomes.json": "3.9"})
    calls = rig.run_from_row()
    assert calls == ["outcome", "assignment", "triage", "hand_scores", "gate"]


def test_outputs_made_from_another_pose_are_all_remade(rig, monkeypatch):
    _return_from(rig, "triage", monkeypatch=monkeypatch, pose="DLC_resnet50_OldModelshuffle1_50")
    calls = rig.run_from_row()
    assert calls[0] == "segmentation"
    assert calls[-3:] == ["triage", "hand_scores", "gate"]


def test_the_resume_point_survives_a_restart(rig, monkeypatch):
    _return_from(rig, "triage", monkeypatch=monkeypatch)
    rig.orch._hand_score_precheck = lambda vid, f: rig.calls.append("hand_scores") and False
    rig.run_from_row()                                    # held for its hand scores
    rig.orch._score_waits = {}                             # a watcher restart
    rig.orch._hand_score_precheck = lambda vid, f: rig.calls.append("hand_scores") or True
    calls = rig.run_from_row()
    assert calls == ["assignment", "triage", "hand_scores", "gate"]


def test_a_reset_after_a_resumed_run_failed_runs_every_step(rig, monkeypatch):
    """A person resets a failed resumed run to an earlier state (to re-pose it,
    or to run it again from the start): the next run is a fresh one. It must
    not read the earlier round's resume point back and reuse the outputs it
    finds -- a pose redone with the same model carries the same scorer name,
    so those outputs cannot be told apart from current ones
    (watcher/db.py RESUME_POINT_STATES)."""
    _return_from(rig, "triage", held_at_step="assignment", monkeypatch=monkeypatch)
    rig.orch.db.mark_failed(VID, "boom")
    rig.orch.db.reset_failed(VID, to_state="dlc_complete")
    rig.orch.db.update_state(VID, "processing")            # taken in again
    calls = rig.run_from_row()
    assert calls == ["segmentation", "reach", "outcome", "assignment",
                     "triage", "hand_scores", "gate"]


def test_a_video_held_for_its_hand_scores_reuses_its_assignments_when_picked_again(rig):
    """Picked again only because its scores file changed: every output it has,
    reach assignment included, is reused (only the checks run again)."""
    rig.write_outputs(rig.folder)
    rig.orch.db.force_state(VID, "processing", reason="test: analysed, waiting")
    rig.orch._score_waits = {VID: {"stamp": "s"}}
    calls = rig.run_from_row()
    assert "assignment" not in calls
    assert not {"segmentation", "reach", "outcome"} & set(calls)


# ---------------------------------------------------------------------------
# What a hold and a return record
# ---------------------------------------------------------------------------

def test_a_hold_records_its_step_on_the_row_and_in_the_bundle(rig, monkeypatch):
    from mousereach.watcher.review_gate import route_to_queue
    src = rig.folder.parent / "routed_from"
    src.mkdir()
    rig.write_outputs(src)
    rig.orch.db.force_state(VID, "processing", reason="test", resume_stage="assignment")
    bundle = route_to_queue(VID, src, Paths.TRIAGE_REVIEW, "test hold",
                            db=rig.orch.db, db_state="triage", held_at_step="assignment")
    row = rig.orch.db.get_video(VID)
    assert row["held_at_step"] == "assignment"
    assert row["resume_stage"] is None                    # an old resume point never outlives a new hold
    manifest = json.loads((bundle / f"{VID}_manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["held_at_step"] == "assignment"


def test_the_gate_records_the_step_its_questions_are_about(tmp_path, monkeypatch):
    import mousereach.watcher.review_gate as rg
    seen = {}
    monkeypatch.setattr(rg, "evaluate_gate", lambda *a, **k: (rg.DECISION_TRIAGE, "why", None))
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: seen.update(k) or tmp_path)
    folder = tmp_path / "work"
    folder.mkdir()
    rg.run_gate(VID, folder)
    assert seen["held_at_step"] == "outcome"
    (folder / f"{VID}_reach_assignments.json").write_text(
        json.dumps({rg.ASSIGNMENT_FAILED_KEY: "inputs unreadable"}), encoding="utf-8")
    rg.run_gate(VID, folder)
    assert seen["held_at_step"] == "assignment"


def test_filing_ends_the_record(rig, tmp_path, monkeypatch):
    from unittest import mock
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
    orch.db.force_state(VID, "processed", reason="test", resume_stage="assignment",
                        held_at_step="outcome")
    with mock.patch("mousereach.archive.core.archive_video",
                    lambda video_id, **kw: {"success": True, "files_moved": ["a"],
                                            "destination": "x"}):
        o.ProcessingOrchestrator._archive_to_nas(orch, {"id": VID})
    row = orch.db.get_video(VID)
    assert row["state"] == "archived"
    assert row["resume_stage"] is None and row["held_at_step"] is None


# ---------------------------------------------------------------------------
# The version and pose test on its own
# ---------------------------------------------------------------------------

def test_nothing_is_reused_when_the_declaration_cannot_be_read(tmp_path):
    from mousereach.watcher import resume_point as rp
    for suffix, (key, v) in OUTPUTS.items():
        (tmp_path / f"{VID}{suffix}").write_text(json.dumps({key: v}), encoding="utf-8")
    start, why = rp.current_start(tmp_path, VID, "checks", {}, SCORER)
    assert start == "segmentation" and why


def test_a_persons_cuts_are_exempt_from_the_segmenter_version(tmp_path):
    from mousereach.watcher import resume_point as rp
    for suffix, (key, v) in OUTPUTS.items():
        (tmp_path / f"{VID}{suffix}").write_text(json.dumps({key: v}), encoding="utf-8")
    (tmp_path / f"{VID}_segments.json").write_text(
        json.dumps({"segmenter_version": "1.0", "boundary_source": "human"}), encoding="utf-8")
    (tmp_path / f"{VID}_processing_manifest.json").write_text(
        json.dumps({"dlc_model": {"dlc_scorer": SCORER}}), encoding="utf-8")
    start, _ = rp.current_start(tmp_path, VID, "checks", {"versions": VERSIONS}, SCORER)
    assert start == "checks"
