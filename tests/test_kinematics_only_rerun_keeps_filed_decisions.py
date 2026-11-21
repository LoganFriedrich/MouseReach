"""A kinematics-only re-run of a filed video recomputes measurements, applies
EVERY person's answer that applies now, and keeps every outcome and causal
reach no person decides as filed. It is never held for a difference from the
filed results (owner rule 2026-11-21: "a human's answer is always applied,
including to filed videos; filed is final only protects filed results from
algorithm re-runs"; kinematics/core/filed_decisions.settle_against_filed).
Until 2026-11-21 such a run was held whenever a change was not explained by an
answer the version scan counted as not yet applied -- and the answer was then
never applied (66 filed videos).

WHY: the extractor works the decisions out again from the filed files (reaches,
outcomes, assignment, every review copy, the ground truth). When the rules
that do that changed since filing, or a file the filing run did not see is
found now, the same files give different decisions -- and a kinematics-only
run, which skips the quality triage and the hand-score comparison, filed them
silently. Seen on filed videos: ground truth that sat beside a video when it
was filed but was not applied then, and review answers that newer matching
places on a different pellet, or no longer applies at all.

Built under tmp_path with invented ids; no real folder, database or
configuration. Names that only this change adds are imported inside the test
that needs them, so the module still loads on code without the change; there
most tests fail on the new keyword the rule is passed by, and the pipeline-run
test fails by what it asserts (no filed features handed to the extractor).
"""
import json
import pickle

import numpy as np
import pandas as pd
import pytest

import mousereach.review.causal_review_io as crio
from mousereach.config import Paths
from mousereach.kinematics.core.feature_extractor import FeatureExtractor

VID = "20990101_ABC0911_P1"
POINTS = ("Nose", "BOXL", "BOXR", "SABL", "SABR", "Pellet", "Pillar",
          "RightHand", "RHLeft", "RHOut", "RHRight")


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """A processing folder holding a filed video's reused outputs (one pellet,
    two detected reaches, the algorithm says the pellet was displaced by reach
    1), and the video's filed folder holding the features file and manifest
    the filing run wrote -- made here by running the extractor once, as the
    filing run did, before anything below changes."""
    monkeypatch.setattr(crio, "_GT_INDEX_CACHE", {})
    monkeypatch.setattr(crio, "review_dirs", lambda *a, **k: [], raising=False)
    monkeypatch.setattr(crio, "durable_review_dir", lambda: None, raising=False)
    monkeypatch.setattr(crio, "review_path_index", lambda stems: {}, raising=False)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    proc = tmp_path / "Processing"
    filed = tmp_path / "Analyzed" / "PROJ" / "PROJ_01"
    proc.mkdir()
    filed.mkdir(parents=True)
    monkeypatch.setattr(Paths, "PROCESSING", proc)
    n = 200
    cols, data = [], []
    for p in POINTS:
        for c, v in (("x", 100.0), ("y", 150.0), ("likelihood", 1.0)):
            cols.append(("scorer", p, c))
            data.append(np.full(n, v) + (np.arange(n) * 0.5 if c == "y" and p.startswith("R") else 0))
    pose = proc / f"{VID}DLC_x.h5"
    pd.DataFrame(np.array(data).T, columns=pd.MultiIndex.from_tuples(cols)).to_hdf(pose, key="df")
    with open(pose.with_name(pose.stem + "_meta.pickle"), "wb") as f:
        pickle.dump({"data": {"fps": 60.0}}, f)
    (proc / f"{VID}_reaches.json").write_text(json.dumps(
        {"video_name": VID, "total_frames": n, "n_segments": 1,
         "segments": [{"segment_num": 1, "start_frame": 0, "end_frame": 199,
                       "ruler_pixels": 34.0, "n_reaches": 2,
                       "reaches": [
                           {"reach_id": 1, "reach_num": 1, "start_frame": 20, "end_frame": 30,
                            "apex_frame": 25, "duration_frames": 11},
                           {"reach_id": 2, "reach_num": 2, "start_frame": 120, "end_frame": 135,
                            "apex_frame": 128, "duration_frames": 16}]}]}), encoding="utf-8")
    (proc / f"{VID}_pellet_outcomes.json").write_text(json.dumps(
        {"video_name": VID, "segments": [{"segment_num": 1, "outcome": "displaced_sa",
                                          "start_frame": 0, "end_frame": 199}]}),
        encoding="utf-8")
    (proc / f"{VID}_reach_assignments.json").write_text(json.dumps(
        {"video_id": VID, "reaches": [{"reach_id": 1, "segment_num": 1, "is_causal": True}]}),
        encoding="utf-8")
    feats = FeatureExtractor().extract(pose, proc / f"{VID}_reaches.json",
                                       proc / f"{VID}_pellet_outcomes.json")
    (filed / f"{VID}_features.json").write_text(json.dumps(feats.to_dict()), encoding="utf-8")
    (filed / f"{VID}_processing_manifest.json").write_text(json.dumps(
        {"video_id": VID, "created_at": "2099-01-02T10:00:00",
         "applied_ground_truth": None}), encoding="utf-8")
    return proc, filed, pose


def _rerun(proc, filed, pose, owed=()):
    return FeatureExtractor().extract(
        pose, proc / f"{VID}_reaches.json", proc / f"{VID}_pellet_outcomes.json",
        keep_filed=filed / f"{VID}_features.json", owed=owed)


def _ground_truth(folder, outcome="untouched"):
    """Ground truth a person made: reach 2's frames moved, and the pellet's
    outcome. The kind of file found beside filed videos that their filing run
    had not applied."""
    p = folder / f"{VID}_unified_ground_truth.json"
    p.write_text(json.dumps({
        "video_name": VID,
        "reaches": {"exhaustive": False, "reaches": [
            {"reach_id": 9, "segment_num": 1, "start_frame": 118, "end_frame": 140,
             "start_determined": True, "end_determined": True, "apex_frame": 128}]},
        "outcomes": {"exhaustive": False, "segments": [
            {"segment_num": 1, "outcome": outcome, "determined": True,
             "start_frame": 0, "end_frame": 199}]},
    }), encoding="utf-8")
    return p


def _reach_frames(feats):
    return sorted((r["start_frame"], r["end_frame"])
                  for s in feats.to_dict()["segments"] for r in s["reaches"])


def test_with_nothing_changed_the_measurements_are_written(dirs):
    proc, filed, pose = dirs
    feats = _rerun(proc, filed, pose)
    seg = feats.to_dict()["segments"][0]
    assert seg["outcome"] == "displaced_sa" and seg["causal_reach_id"] == 1
    assert _reach_frames(feats) == [(20, 30), (120, 135)]


def test_ground_truth_the_filing_run_did_not_apply_is_applied_owed_or_not(dirs):
    """Ground truth the filing run did not apply is applied, though nothing
    marks it as owed (the manifest's stamps said it was applied)."""
    proc, filed, pose = dirs
    _ground_truth(proc)
    ex = FeatureExtractor()
    feats = ex.extract(pose, proc / f"{VID}_reaches.json", proc / f"{VID}_pellet_outcomes.json",
                       keep_filed=filed / f"{VID}_features.json")
    seg = feats.to_dict()["segments"][0]
    assert seg["outcome"] == "untouched"
    assert (118, 140) in _reach_frames(feats) and (120, 135) not in _reach_frames(feats)
    assert any("displaced_sa -> untouched" in c for c in ex.changes_from_filed)


def test_ground_truth_the_filed_results_owe_is_applied(dirs):
    proc, filed, pose = dirs
    _ground_truth(proc)
    feats = _rerun(proc, filed, pose, owed={"ground_truth"})
    seg = feats.to_dict()["segments"][0]
    assert seg["outcome"] == "untouched"
    assert (118, 140) in _reach_frames(feats) and (120, 135) not in _reach_frames(feats)


def test_a_filed_decision_no_person_decides_now_stays_as_filed(dirs, monkeypatch):
    """A person's answer is in the filed results but no longer found by the
    re-run (e.g. answered on cuts that have since moved). Nothing decides the
    pellet now, so the filed decision stands -- the algorithm does not decide
    a filed video again -- and the run is not held."""
    proc, filed, pose = dirs
    review = proc / f"{VID}_causal_review.json"
    review.write_text(json.dumps({
        "type": "causal_review", "schema_version": "1.1", "video_stem": VID,
        "reviewer": "someone", "reviewed_at": "2099-01-01T09:00:00",
        "segments": [{"segment_num": 1, "segment_span": {"start": 0, "end": 199},
                      "reviewed": True,
                      "human": {"outcome": "retrieved",
                                "causal_reach": {"start": 120, "end": 135, "reach_id": 2}},
                      "answers": {"outcome_pick": "retrieved",
                                  "causal_pick": {"start": 120, "end": 135, "reach_id": 2}}}]}),
        encoding="utf-8")
    monkeypatch.setattr(crio, "review_dirs", lambda *a, **k: [proc])
    answered = FeatureExtractor().extract(pose, proc / f"{VID}_reaches.json",
                                          proc / f"{VID}_pellet_outcomes.json")
    assert answered.to_dict()["segments"][0]["outcome"] == "retrieved"
    (filed / f"{VID}_features.json").write_text(json.dumps(answered.to_dict()), encoding="utf-8")
    review.unlink()                     # the answer is no longer found by the re-run
    ex = FeatureExtractor()
    feats = ex.extract(pose, proc / f"{VID}_reaches.json", proc / f"{VID}_pellet_outcomes.json",
                       keep_filed=filed / f"{VID}_features.json")
    seg = feats.to_dict()["segments"][0]
    assert seg["outcome"] == "retrieved"
    assert any("outcome kept as filed (retrieved" in c for c in ex.changes_from_filed)
    causal = [r for r in seg["reaches"] if r["reach_id"] == seg["causal_reach_id"]]
    assert [(r["start_frame"], r["end_frame"]) for r in causal] == [(120, 135)]


def test_renumbered_reach_ids_are_not_a_change():
    from mousereach.kinematics.core import filed_decisions as fd
    filed = fd.decisions_of_features({"segments": [{
        "segment_num": 1, "outcome": "retrieved", "causal_reach_id": 4,
        "reaches": [{"reach_id": 4, "start_frame": 10, "end_frame": 20}]}]})
    seg = {"segment_num": 1, "reaches": [{"reach_id": 7, "start_frame": 10, "end_frame": 20}]}
    fresh = {1: fd.decision_of_resolved(
        seg, {"outcome": "retrieved", "causal_reach_id": 7}, seg["reaches"], [])}
    assert fd.changes_not_owed(filed, fresh) == []


def test_owed_answers_are_read_by_the_version_scans_rules(dirs, monkeypatch):
    from mousereach.kinematics.core.filed_decisions import owed_human_layers
    proc, filed, pose = dirs
    assert owed_human_layers(VID, filed) == frozenset()
    # The manifest says the kinematics applied no ground truth; one filed now is owed.
    _ground_truth(filed)
    assert owed_human_layers(VID, filed) == frozenset({"ground_truth"})
    # A review the filed results never applied is owed.
    review = filed / f"{VID}_causal_review.json"
    review.write_text(json.dumps({"reviewed_at": "2099-01-03T10:00:00", "segments": []}),
                      encoding="utf-8")
    st = review.stat()
    monkeypatch.setattr(crio, "review_path_index",
                        lambda stems: {VID: (review, st.st_mtime, st.st_size)})
    assert owed_human_layers(VID, filed) == frozenset({"ground_truth", "human_review"})


def test_the_watcher_applies_the_answer_in_a_kinematics_only_rerun(
        dirs, tmp_path, monkeypatch):
    """The watcher's kinematics step of a filed video's kinematics-only
    re-run writes the features with the person's answer and goes on to
    filing; nothing is held, nobody is asked."""
    import mousereach.archive.core as arch
    from mousereach.watcher.db import WatcherDB
    from mousereach.watcher.orchestrator import ProcessingOrchestrator
    proc, filed, pose = dirs
    monkeypatch.setattr(arch, "get_archive_destination", lambda vid: filed)
    (proc / f"{VID}_processing_manifest.json").write_text("{}", encoding="utf-8")
    _ground_truth(proc)
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.db.export_to_central_db = lambda vid: None
    o.processing_dir = proc
    o.hostname = "NODE-T"
    o.db.register_video(video_id=VID, source_path="x")
    o.db.force_state(VID, "processing", reason="test")
    o._extract_features_and_finish(VID, pose, False, 1.0, keep_filed_decisions=True)
    row = o.db.get_video(VID)
    assert row["state"] == "processed"
    written = json.loads((proc / f"{VID}_features.json").read_text())
    assert written["segments"][0]["outcome"] == "untouched"


# ---------------------------------------------------------------------------
# The pipeline run hands the rule to the kinematics step only for a filed
# video's kinematics-only re-run
# ---------------------------------------------------------------------------

@pytest.fixture
def rig(tmp_path, monkeypatch):
    """Every algorithm and check replaced by a recorder; the extractor records
    the arguments it is given (test_filed_rerun_skips_checks.py builds the
    same rig)."""
    import mousereach.segmentation.core.batch as segb
    import mousereach.reach.core.batch as reachb
    import mousereach.outcomes.core.batch as outb
    import mousereach.assignment.run as assign
    import mousereach.pipeline.manifest as man
    import mousereach.pipeline.triage as tri
    import mousereach.watcher.review_gate as rg
    import mousereach.kinematics.core.feature_extractor as fe
    import mousereach.archive.core as arch
    import mousereach.watcher.orchestrator as o
    from mousereach.watcher.db import WatcherDB

    seen = []
    folder = tmp_path / "work"
    filed = tmp_path / "filed"
    folder.mkdir()
    filed.mkdir()
    for name in ("_segments.json", "_reaches.json", "_pellet_outcomes.json",
                 "_reach_assignments.json"):
        (folder / f"{VID}{name}").write_text("{}", encoding="utf-8")
    (filed / f"{VID}_features.json").write_text("{}", encoding="utf-8")
    (filed / f"{VID}_processing_manifest.json").write_text("{}", encoding="utf-8")
    pose = folder / f"{VID}DLC_test.h5"
    pose.write_bytes(b"")
    (folder / f"{VID}.mp4").write_bytes(b"x")
    monkeypatch.setattr(o, "resolve_pose_input", lambda *a, **k: pose)
    monkeypatch.setattr(arch, "get_archive_destination", lambda vid: filed)

    def _ok(result):
        return lambda *a, **k: result
    monkeypatch.setattr(segb, "process_single", _ok({"success": True}))
    monkeypatch.setattr(reachb, "process_single", _ok({"total_reaches": 0}))
    monkeypatch.setattr(outb, "process_single", _ok({"n_segments": 0}))
    monkeypatch.setattr(assign, "assign_reaches_for_video", _ok({"assigned": True}))

    def _manifest(**k):
        from pathlib import Path as _P
        m = _P(k["processing_dir"]) / f"{k['video_id']}_processing_manifest.json"
        if not m.exists():
            m.write_text("{}")
        return {}
    monkeypatch.setattr(man, "create_processing_manifest", _manifest)
    monkeypatch.setattr(man, "record_kinematic_version", lambda *a, **k: None)

    class _Triage:
        verdict = "clean"
        flags = []
        n_critical = 0
        n_warnings = 0

        def save(self, path):
            pass
    monkeypatch.setattr(tri, "triage_video", lambda **k: _Triage())
    monkeypatch.setattr(rg, "run_gate", lambda *a, **k: rg.DECISION_CLEAN)

    class _Features:
        n_segments = 0

        def to_dict(self):
            return {}

    class _Extractor:
        VERSION = "9.9.9"

        def extract(self, *a, **k):
            seen.append(k)
            return _Features()
    monkeypatch.setattr(fe, "FeatureExtractor", _Extractor)

    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = WatcherDB(tmp_path / "local.db")
    orch.db.register_video(video_id=VID, source_path="x")
    orch.processing_dir = folder
    orch.hostname = "HOST"
    orch._score_waits = {}
    orch._hand_score_precheck = lambda video_id, folder_: False
    # The pose stand-in is not a real pose; the pose test is not what is tested.
    import mousereach.pipeline.pose_check as pc
    monkeypatch.setattr(pc, "check_pose", lambda *a, **k: (True, ""))

    def run(start_stage=None, filed_before=True):
        if filed_before:
            orch.db.force_state(VID, "processing", reason="test fixture")
            orch.db.update_state(VID, "processed")
            orch.db.update_state(VID, "archiving")
            orch.db.update_state(VID, "archived")
        data = {"video_id": VID, "dlc_output_path": str(pose)}
        if start_stage:
            orch.db.force_state(VID, "outdated", reason="test", reprocess_scope=start_stage)
            orch.db.force_state(VID, "processing", reason="test re-run")
            data["reprocess_start_stage"] = start_stage
        else:
            orch.db.force_state(VID, "processing", reason="test fixture")
        seen.clear()
        orch._run_pipeline({"id": VID, "data": data})
        return list(seen)

    return type("Rig", (), {"run": staticmethod(run), "filed": filed, "orch": orch,
                            "seen": seen, "pose": pose})


def test_a_filed_kinematics_only_rerun_hands_the_filed_features_to_the_extractor(rig):
    seen = rig.run("kinematics")
    assert len(seen) == 1
    assert str(seen[0].get("keep_filed")) == str(rig.filed / f"{VID}_features.json")


def test_a_kinematics_step_not_marked_as_a_filed_rerun_keeps_nothing(rig):
    """Every other run (a new video, a review return, a re-run that
    recomputed an algorithm) reaches the kinematics step without the rule:
    _run_pipeline passes filed_rerun_reuses_analysis, False for all of them
    (tested in test_filed_rerun_skips_checks.py)."""
    (rig.orch.processing_dir / f"{VID}_processing_manifest.json").write_text("{}")
    rig.orch.db.force_state(VID, "processing", reason="test fixture")
    rig.seen.clear()
    rig.orch._extract_features_and_finish(VID, rig.pose, False, 1.0)
    assert len(rig.seen) == 1
    assert "keep_filed" not in rig.seen[0]


def test_an_owed_review_does_not_let_a_ground_truth_change_through():
    """Only the kind of answer that is owed may change what it decides: a
    review saved after filing does not make an old ground truth's change
    acceptable."""
    from mousereach.kinematics.core import filed_decisions as fd
    filed = {1: {"outcome": "displaced_sa", "causal": (10, 20),
                 "reaches": [(10, 20)], "left_out": []}}
    fresh = {1: {"outcome": "abnormal_exception", "causal": None, "reaches": [(10, 18)],
                 "left_out": [], "outcome_source": "ground_truth",
                 "causal_reach_source": None, "reach_sources": {(10, 18): "ground_truth"}}}
    assert fd.changes_not_owed(filed, fresh, owed={"human_review"})
    assert fd.changes_not_owed(filed, fresh, owed={"ground_truth"}) == []


# ---------------------------------------------------------------------------
# Owed ground truth explains a filed reach that is simply gone only when it
# replaced the pellet's reaches with a list it declares complete.
# ---------------------------------------------------------------------------

def _outcomes_only_ground_truth(folder, outcome="untouched"):
    """Ground truth that decides the pellet's outcome and lists no reach."""
    p = folder / f"{VID}_unified_ground_truth.json"
    p.write_text(json.dumps({
        "video_name": VID,
        "reaches": {"exhaustive": False, "reaches": []},
        "outcomes": {"exhaustive": False, "segments": [
            {"segment_num": 1, "outcome": outcome, "determined": True,
             "start_frame": 0, "end_frame": 199}]},
    }), encoding="utf-8")
    return p


def _drop_detected_reach(proc, reach_id):
    """The reaches file now lacks one filed reach -- as a changed reach rule or
    a review copy the filing run did not see would leave it; not the ground
    truth's doing."""
    p = proc / f"{VID}_reaches.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    seg = doc["segments"][0]
    seg["reaches"] = [r for r in seg["reaches"] if r["reach_id"] != reach_id]
    seg["n_reaches"] = len(seg["reaches"])
    p.write_text(json.dumps(doc), encoding="utf-8")


def test_owed_ground_truth_without_a_complete_reach_list_removes_no_reach():
    from mousereach.kinematics.core import filed_decisions as fd
    filed = {1: {"outcome": "displaced_sa", "causal": (10, 20),
                 "reaches": [(10, 20), (40, 50)], "left_out": []}}
    fresh = {1: {"outcome": "displaced_sa", "causal": (10, 20), "reaches": [(10, 20)],
                 "left_out": [], "outcome_source": "algo", "causal_reach_source": None,
                 "reach_sources": {(10, 20): "algo"}, "reaches_complete_by": None}}
    changes = fd.changes_not_owed(filed, fresh, owed={"ground_truth"})
    assert changes and "1 filed gone" in changes[0]
    assert "does not list this pellet's reaches as complete" in changes[0]
    # The same change made by ground truth that replaced the pellet's reach
    # list with its complete one is owed.
    fresh[1]["reaches_complete_by"] = "ground_truth"
    assert fd.changes_not_owed(filed, fresh, owed={"ground_truth"}) == []
    # ... but only while that ground truth is owed.
    assert fd.changes_not_owed(filed, fresh, owed={"human_review"})


def test_a_reach_gone_for_another_reason_does_not_stop_the_answer(dirs):
    """Outcome-only ground truth; the re-run also finds a filed reach gone for
    a reason that is not the ground truth. The person's outcome is applied
    anyway, and the line for the log says what else differs."""
    proc, filed, pose = dirs
    _outcomes_only_ground_truth(proc)
    _drop_detected_reach(proc, 1)
    ex = FeatureExtractor()
    feats = ex.extract(pose, proc / f"{VID}_reaches.json", proc / f"{VID}_pellet_outcomes.json",
                       keep_filed=filed / f"{VID}_features.json")
    assert feats.to_dict()["segments"][0]["outcome"] == "untouched"
    note = " ".join(ex.changes_from_filed)
    assert "displaced_sa -> untouched (a person's answer: ground_truth)" in note
    assert "1 filed not here now" in note


def test_an_algorithm_answer_that_differs_from_the_filed_one_is_not_written(dirs):
    """The reach assignment now names another reach than the filed results
    did, and no person decided it: the filed causal reach stays (an algorithm
    never re-decides a filed video), and nothing is held."""
    proc, filed, pose = dirs
    (proc / f"{VID}_reach_assignments.json").write_text(json.dumps(
        {"video_id": VID, "reaches": [{"reach_id": 2, "segment_num": 1, "is_causal": True}]}),
        encoding="utf-8")
    ex = FeatureExtractor()
    feats = ex.extract(pose, proc / f"{VID}_reaches.json", proc / f"{VID}_pellet_outcomes.json",
                       keep_filed=filed / f"{VID}_features.json")
    seg = feats.to_dict()["segments"][0]
    assert seg["outcome"] == "displaced_sa"
    causal = [r for r in seg["reaches"] if r["reach_id"] == seg["causal_reach_id"]]
    assert [(r["start_frame"], r["end_frame"]) for r in causal] == [(20, 30)]
    assert any("causal reach kept as filed" in c for c in ex.changes_from_filed)


def test_complete_ground_truth_that_leaves_a_reach_out_is_applied(dirs):
    """Ground truth that lists the pellet's reaches as complete and leaves
    filed reach 1 out removed it: owed, so it is applied."""
    proc, filed, pose = dirs
    (proc / f"{VID}_unified_ground_truth.json").write_text(json.dumps({
        "video_name": VID,
        "reaches": {"exhaustive": True, "reaches": [
            {"reach_id": 2, "segment_num": 1, "start_frame": 120, "end_frame": 135,
             "start_determined": True, "end_determined": True, "apex_frame": 128}]},
        "outcomes": {"exhaustive": False, "segments": [
            {"segment_num": 1, "outcome": "untouched", "determined": True,
             "start_frame": 0, "end_frame": 199}]},
    }), encoding="utf-8")
    feats = _rerun(proc, filed, pose, owed={"ground_truth"})
    assert _reach_frames(feats) == [(120, 135)]
    assert feats.to_dict()["segments"][0]["outcome"] == "untouched"


def test_a_filed_kinematics_only_rerun_passes_no_owed_kinds(rig):
    """Since 2026-11-21 every person's answer is applied on a filed video's
    kinematics-only re-run, so the watcher no longer passes which kinds are
    owed (filed_decisions.settle_against_filed)."""
    seen = rig.run("kinematics")
    assert len(seen) == 1
    assert "owed" not in seen[0]


def test_the_watcher_applies_ground_truth_the_filed_manifest_already_records(
        dirs, tmp_path, monkeypatch):
    """Owner rule 2026-11-21: ground truth a person made is applied to a filed
    video on a kinematics-only re-run, even when the version scan would not
    call it owed (here the filed manifest's record is not consulted at all:
    the watcher counts every person's answer as owed)."""
    import mousereach.archive.core as arch
    from mousereach.watcher.db import WatcherDB
    from mousereach.watcher.orchestrator import ProcessingOrchestrator
    proc, filed, pose = dirs
    monkeypatch.setattr(arch, "get_archive_destination", lambda vid: filed)
    (proc / f"{VID}_processing_manifest.json").write_text("{}", encoding="utf-8")
    _ground_truth(proc)
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.db.export_to_central_db = lambda vid: None
    o.processing_dir = proc
    o.hostname = "NODE-T"
    o.db.register_video(video_id=VID, source_path="x")
    o.db.force_state(VID, "processing", reason="test")
    why = o._make_features(VID, proc, pose, False, keep_filed_decisions=True)
    assert why is None
    seg = json.loads((proc / f"{VID}_features.json").read_text())["segments"][0]
    assert seg["outcome"] == "untouched"
