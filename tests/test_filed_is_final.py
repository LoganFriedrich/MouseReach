"""Filed is final (lab rule, 2026-11-19).

Once a video is filed, the four algorithms (segmentation, reach detection,
outcome detection, reach assignment) are never re-run onto it; a newer version
applies only to videos not filed yet. Only kinematics may be recalculated, and
a kinematics recalculation never holds the video or asks a person anything.
Nothing automatic reopens a filed video; a person's command may, and then the
filed copy stays until the new run replaces it. Evaluation re-runs never write
into the filed folder.

Built under tmp_path with invented ids; no real folder or database.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.review.causal_review_io as crio
from mousereach.watcher import reprocessor
from mousereach.watcher.reprocessor import ReprocessingScanner

SCORER = "DLC_resnet50_TestModelshuffle1_100"
NEW_SCORER = "DLC_resnet101_TestModelshuffle3_100"
STEM = "20990101_ABC0101_P1"
CURRENT = {"segmenter": "2.0", "reach_detector": "3.0", "outcome_detector": "4.0",
           "assignment": "1.0", "kinematic_extractor": "5.0"}


class DB:
    """The few calls the scan makes, recorded."""

    def __init__(self, archived=(), outdated=None):
        self.rows = {s: {"video_id": s, "state": "archived"} for s in archived}
        for s, fields in (outdated or {}).items():
            self.rows[s] = dict({"video_id": s, "state": "outdated"}, **fields)

    def get_videos_in_state(self, state):
        return [dict(r) for r in self.rows.values() if r["state"] == state]

    def _get_connection(self):
        raise RuntimeError("no db in this test")

    def force_state(self, vid, state, reason=None, **kw):
        self.rows[vid].update(kw, state=state)

    def update_state(self, vid, state, **kw):
        self.force_state(vid, state, **kw)

    def set_fields(self, vid, **kw):
        self.rows[vid].update(kw)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    import mousereach.config as config
    monkeypatch.setattr(config, "filing_rules", lambda: {
        "ABC": {"folder": "Project_A", "cohort_folder": "{CODE}{NN}"}})
    nas = tmp_path / "nas"
    cohort = nas / "Analyzed" / "Project_A" / "ABC01"
    cohort.mkdir(parents=True)
    (nas / "pipeline_versions.json").write_text(json.dumps({
        "versions": dict(CURRENT, dlc_scorer=SCORER)}), encoding="utf-8")
    import mousereach.pipeline.version_index as vi
    monkeypatch.setattr(vi, "VersionIndex", lambda *a, **k: SimpleNamespace(
        upsert_from_manifest=lambda *a, **k: None))
    monkeypatch.setattr(crio, "review_path_index", lambda stems: {})
    monkeypatch.setattr(crio, "resolve_review_path", lambda *a, **k: None)
    return nas, cohort


def _filed(cohort, versions, stem=STEM, scorer=SCORER):
    m = {"video_stem": stem, "created_at": "2099-01-01T10:00:00",
         "dlc_model": {"dlc_scorer": scorer}, "pipeline_versions": versions}
    (cohort / f"{stem}_processing_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    (cohort / f"{stem}_features.json").write_text("{}", encoding="utf-8")
    (cohort / f"{stem}.mp4").write_bytes(b"x")


def _scan(nas, db):
    return ReprocessingScanner(db, nas, cache_path=nas / "cache.json").scan(
        mark_outdated=True, adopt_orphans=False)


# --- the version scan -------------------------------------------------------

@pytest.mark.parametrize("component", ["segmenter", "reach_detector",
                                       "outcome_detector", "assignment"])
def test_a_newer_algorithm_never_marks_a_filed_video(archive, component):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT, **{component: "0.1"}))
    db = DB(archived=[STEM])
    summary = _scan(nas, db)
    assert db.rows[STEM]["state"] == "archived"
    assert summary["outdated"] == 0
    assert summary["filed_final_kept"] == 1
    assert summary["filed_final_kept_videos"] == [
        {"video_id": STEM, "components": [component]}]


def test_a_newer_pose_model_never_marks_a_filed_video(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT), scorer=NEW_SCORER)      # manifest names another model
    db = DB(archived=[STEM])
    summary = _scan(nas, db)
    assert db.rows[STEM]["state"] == "archived"
    assert summary["outdated_full"] == 0
    assert "dlc" in summary["filed_final_kept_videos"][0]["components"]


def test_an_algorithm_stage_the_manifest_never_recorded_is_not_rerun(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT, reach_detector="unknown"))
    db = DB(archived=[STEM])
    _scan(nas, db)
    assert db.rows[STEM]["state"] == "archived"


def test_a_newer_kinematic_extractor_is_still_recalculated(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT, segmenter="0.1", kinematic_extractor="4.9"))
    db = DB(archived=[STEM])
    _scan(nas, db)
    assert db.rows[STEM]["state"] == "outdated"
    assert db.rows[STEM]["reprocess_scope"] == "kinematics"


def test_an_easy_tray_without_its_session_file_gets_kinematics_only(archive):
    nas, cohort = archive
    stem = "20990101_ABC0101_E1"
    _filed(cohort, dict(CURRENT), stem=stem)
    (cohort / f"{stem}_features.json").unlink()
    _filed(cohort, dict(CURRENT))               # one listed features file, so the listing counts
    db = DB(archived=[stem])
    _scan(nas, db)
    assert db.rows[stem]["state"] == "outdated"
    assert db.rows[stem]["reprocess_scope"] == "kinematics"


def test_an_algorithm_mark_the_scan_made_earlier_is_lifted(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT, segmenter="0.1"))
    db = DB(outdated={STEM: {"reprocess_scope": "segmentation", "mark_reason": None}})
    _scan(nas, db)
    assert db.rows[STEM]["state"] == "archived"
    assert db.rows[STEM]["reprocess_scope"] is None


def test_an_algorithm_mark_that_still_owes_kinematics_is_narrowed(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT, segmenter="0.1", kinematic_extractor="4.9"))
    db = DB(outdated={STEM: {"reprocess_scope": "segmentation", "mark_reason": None}})
    summary = _scan(nas, db)
    assert db.rows[STEM]["state"] == "outdated"
    assert db.rows[STEM]["reprocess_scope"] == "kinematics"
    assert summary["narrowed_to_kinematics"] == 1


def test_a_persons_rerun_mark_is_never_changed(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT, segmenter="0.1"))
    db = DB(outdated={STEM: {"reprocess_scope": "segmentation",
                             "mark_reason": "a person pressed Re-run"}})
    _scan(nas, db)
    assert db.rows[STEM]["state"] == "outdated"
    assert db.rows[STEM]["reprocess_scope"] == "segmentation"


def test_filed_is_final_keeps_only_kinematics():
    from mousereach.watcher.reprocessor import filed_is_final
    out, kept = filed_is_final({
        "is_current": False, "needs_full_reprocess": True,
        "stale_components": ["dlc", "segmenter", "kinematic_extractor"],
        "unrecorded_components": ["assignment"]})
    assert out["stale_components"] == ["kinematic_extractor"]
    assert out["unrecorded_components"] == []
    assert out["needs_full_reprocess"] is False and out["is_current"] is False
    assert kept == ["dlc", "segmenter", "assignment (not recorded)"]


# --- the re-pose adopt step ---------------------------------------------------

def test_a_pose_turning_up_for_a_filed_video_reruns_nothing(tmp_path, monkeypatch):
    import mousereach.watcher.repose as repose
    from mousereach.watcher.db import WatcherDB
    staging = tmp_path / "Posed"
    staging.mkdir()
    (staging / f"{STEM}{NEW_SCORER}.h5").write_bytes(b"new")
    monkeypatch.setattr(repose, "_archived_scorer", lambda vid: SCORER)
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=STEM, source_path="x")
    db.force_state(STEM, "archived", reason="test")
    assert repose.adopt_staged_reposes(db, staging, repose_dir=tmp_path / "q",
                                       declared=NEW_SCORER) == []
    assert db.get_video(STEM)["state"] == "archived"


# --- a kinematics recalculation never holds -----------------------------------

def _bare_orchestrator(tmp_path):
    import mousereach.watcher.orchestrator as o
    from mousereach.watcher.db import WatcherDB
    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = WatcherDB(tmp_path / "local.db")
    orch.db.register_video(video_id=STEM, source_path="x")
    orch.processing_dir = tmp_path / "work"
    orch.processing_dir.mkdir()
    orch.hostname = "HOST"
    orch._score_waits = {}
    return orch


def test_a_kinematics_recalculation_runs_no_gate(tmp_path):
    import mousereach.watcher.review_gate as rg
    orch = _bare_orchestrator(tmp_path)
    gate = []
    held = orch._run_checks_before_kinematics(
        STEM, None, 0.0, None, lambda *a, **k: gate.append(1) or rg.DECISION_TRIAGE,
        rg.DECISION_CLEAN, rerun_triage_and_hand_scores=False)
    assert held is False and gate == []


def test_a_refused_kinematics_recalculation_leaves_the_filed_results_standing(tmp_path):
    orch = _bare_orchestrator(tmp_path)
    restored, held = [], []
    orch._make_features = lambda *a, **k: "the filed decisions would change"
    orch._restore_filed_after_failed_rerun = lambda vid, why: restored.append(why)
    orch._hold_without_kinematics = lambda vid, why: held.append(why)
    orch._extract_features_and_finish(STEM, None, False, 0.0, keep_filed_decisions=True)
    assert held == [] and len(restored) == 1
    assert "filed decisions would change" in restored[0]


def test_a_kinematics_recalculation_never_recomputes_a_missing_output(tmp_path):
    orch = _bare_orchestrator(tmp_path)
    for n in ("segments", "reaches"):
        (orch.processing_dir / f"{STEM}_{n}.json").write_text("{}")
    why = orch._filed_kinematics_refusal(STEM, orch.processing_dir / "pose.h5")
    assert why and f"{STEM}_pellet_outcomes.json" in why


def test_a_kinematics_recalculation_never_asks_for_a_new_pose(tmp_path, monkeypatch):
    import mousereach.pipeline.pose_check as pc
    orch = _bare_orchestrator(tmp_path)
    for n in ("segments", "reaches", "pellet_outcomes"):
        (orch.processing_dir / f"{STEM}_{n}.json").write_text("{}")
    monkeypatch.setattr(pc, "check_pose", lambda *a, **k: (False, "no SABL"))
    why = orch._filed_kinematics_refusal(STEM, orch.processing_dir / "pose.h5")
    assert why and "never sent for a new pose" in why
    monkeypatch.setattr(pc, "check_pose", lambda *a, **k: (True, ""))
    assert orch._filed_kinematics_refusal(STEM, orch.processing_dir / "pose.h5") is None


# --- a person's route keeps the filed copy ------------------------------------

def test_a_route_of_a_filed_video_copies_and_leaves_the_filed_folder_alone(tmp_path, monkeypatch):
    import mousereach.watcher.review_gate as rg
    from mousereach.watcher.filed_copy_route import route_filed_copy
    filed = tmp_path / "Analyzed" / "ABC01"
    filed.mkdir(parents=True)
    other = "20990101_ABC0101_P11"
    for name in (f"{STEM}.mp4", f"{STEM}_pellet_outcomes.json",
                 f"{STEM}_features.json", f"{other}_features.json"):
        (filed / name).write_bytes(name.encode())
    before = {p.name: p.read_bytes() for p in filed.iterdir()}
    queue = tmp_path / "triage"
    queue.mkdir()
    seen = {}

    def fake_route(video_id, source, dest_root, reason="", **kw):
        seen["names"] = sorted(p.name for p in Path(source).iterdir())
        bundle = Path(dest_root) / video_id
        bundle.mkdir()
        for p in Path(source).iterdir():
            p.rename(bundle / p.name)
        return bundle
    monkeypatch.setattr(rg, "route_to_queue", fake_route)
    bundle = route_filed_copy(STEM, filed, queue, "a person asked")
    assert seen["names"] == sorted([f"{STEM}.mp4", f"{STEM}_features.json",
                                    f"{STEM}_pellet_outcomes.json"])
    assert {p.name: p.read_bytes() for p in filed.iterdir()} == before   # untouched
    assert (bundle / f"{STEM}.mp4").read_bytes() == f"{STEM}.mp4".encode()
    assert not (queue / ".filed_copies").exists()


def test_a_failed_route_of_a_filed_video_leaves_no_copy(tmp_path, monkeypatch):
    import mousereach.watcher.review_gate as rg
    from mousereach.watcher.filed_copy_route import route_filed_copy
    filed = tmp_path / "Analyzed"
    filed.mkdir()
    (filed / f"{STEM}_pellet_outcomes.json").write_text("{}")
    queue = tmp_path / "triage"
    queue.mkdir()

    def refuse(*a, **k):
        raise rg.AlreadyInQueueError("held")
    monkeypatch.setattr(rg, "route_to_queue", refuse)
    with pytest.raises(rg.AlreadyInQueueError):
        route_filed_copy(STEM, filed, queue, "a person asked")
    assert (filed / f"{STEM}_pellet_outcomes.json").exists()
    assert not (queue / ".filed_copies").exists()


# --- bring-current never finalizes a filed video -------------------------------

def test_bring_current_refuses_to_finalize_a_filed_video(monkeypatch):
    import mousereach.pipeline.reprocess_to_current as rtc
    import mousereach.watcher.db_location as dbl
    row = {"video_id": STEM, "state": "archived", "archived_at": "2099-01-01"}
    fake = SimpleNamespace(get_video=lambda vid: row)
    monkeypatch.setattr(dbl, "open_watcher_db_if_present", lambda: (fake, ""))
    import mousereach.watcher.db as wdb
    monkeypatch.setattr(wdb, "crystallized_refusal", lambda *a, **k: None)
    why = {}
    assert rtc._finalize_refusal(STEM, why_out=why, refuse_filed=True) == rtc.FILED_IS_FINAL_SAYS
    assert why["not_a_defect"] == "filed"
    # The review-bundle stager (a person's tool) is not refused for it.
    assert rtc._finalize_refusal(STEM) is None


# --- evaluation never writes beside the reaches it evaluates -------------------

def test_the_version_simulator_writes_nothing_beside_the_reaches(tmp_path):
    from mousereach.eval import version_simulator as vs
    folder = tmp_path / "filed"
    folder.mkdir()
    reaches = folder / f"{STEM}_reaches.json"
    reaches.write_text(json.dumps({"segments": [{"reaches": []}]}))
    gt = folder / f"{STEM}_reach_ground_truth.json"
    gt.write_text("{}")
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    seen = {}

    class Evaluator:
        algo_dir = folder

        def extract_video_id(self, p):
            return STEM

        def compare(self, vid):
            seen["dir"] = self.algo_dir
            seen["names"] = sorted(p.name for p in Path(self.algo_dir).iterdir())
            return None

    version = next(iter(vs.ALGORITHMIC_VERSIONS.values()))
    vs.simulate_version(reaches, gt, version, Evaluator())
    assert Path(seen["dir"]) != folder
    assert seen["names"] == [f"{STEM}_reaches.json"]
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
