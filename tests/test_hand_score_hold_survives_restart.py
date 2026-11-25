"""A video held for its hand scores stays held through a watcher restart, and
its analysis is reused only while it is current.

WHY: the list of held videos lived only in the watcher's memory. After every
restart each held video was run through segmentation, reach and outcome
detection again in full, only to be held again. The holds are now rebuilt from
this machine's attention list. A restart is also how new code is deployed, so
a restored hold reuses an output only when it carries the version the running
code makes (and the declared one): outputs of older algorithms are remade.
"""
import json
import sqlite3

import pytest

import mousereach.watcher.hand_scores as hs
import mousereach.watcher.route_cli as rc
from mousereach.watcher.db import WatcherDB

VID = "20250709_ABC0207_P1"
STARTED = "2026-01-05T10:00:00"
SCORER = "DLC_testmodel"


def _versions():
    from mousereach.pipeline.versions import code_stamped_versions
    v = code_stamped_versions()
    for k in ("segmenter", "reach_detector", "outcome_detector", "assignment"):
        v.setdefault(k, "test-" + k)
    return v


@pytest.fixture
def held(tmp_path, monkeypatch):
    """A video analysed and held for its hand scores by a watcher that then
    stopped: its row, its outputs in Processing, and the attention list."""
    import mousereach.watcher.orchestrator as o
    from mousereach.config import Paths
    import mousereach.pipeline.versions as pv
    import mousereach.watcher.repose as repose

    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path / "nas", raising=False)
    f = tmp_path / "hand.csv"
    f.write_text("video_key,pellet,score\n", encoding="utf-8")
    f.with_suffix(".json").write_text(json.dumps(
        {"last_successful_import": {"ABC_02": "2000-01-01T00:00:00"}}), encoding="utf-8")
    monkeypatch.setattr(hs, "settings", lambda: {"path": str(f), "required_projects": ["ABC"]})
    hs._CACHE.update(key=None)
    monkeypatch.setattr(rc, "answered_segments_of", lambda v, d: set())

    v = _versions()
    monkeypatch.setattr(pv, "code_stamped_versions", lambda: dict(v))
    monkeypatch.setattr(pv, "get_current_versions", lambda *a, **k: {
        "versions": {"segmenter": v["segmenter"], "reach_detector": v["reach_detector"],
                     "outcome_detector": v["outcome_detector"], "assignment": v["assignment"],
                     "dlc_scorer": SCORER}})
    monkeypatch.setattr(repose, "scorer_of", lambda h5: SCORER)

    folder = tmp_path / "Processing"
    folder.mkdir()
    outputs = {
        "_segments.json": {"segmenter_version": v["segmenter"], "boundaries": [0, 1000]},
        "_reaches.json": {"detector_version": v["reach_detector"]},
        "_pellet_outcomes.json": {"detector_version": v["outcome_detector"],
                                  "segments": [{"segment_num": 1, "outcome": "untouched"}]},
        "_reach_assignments.json": {"version": v["assignment"]},
        "_processing_manifest.json": {"dlc_model": {"dlc_scorer": SCORER}},
    }
    for sfx, doc in outputs.items():
        (folder / f"{VID}{sfx}").write_text(json.dumps(doc), encoding="utf-8")
    pose = folder / f"{VID}{SCORER}.h5"
    pose.write_bytes(b"")
    monkeypatch.setattr(o, "resolve_pose_input", lambda *a, **k: pose)
    # The pose stand-in is an empty file; the pose test is not what is tested
    # (an unusable pose of a video not filed starts it over).
    import mousereach.pipeline.pose_check as _pc
    monkeypatch.setattr(_pc, "check_pose", lambda *a, **k: (True, ""))

    db = WatcherDB(tmp_path / "local.db")
    db.register_video(video_id=VID, source_path="x")
    db.force_state(VID, "processing", reason="test fixture")
    con = sqlite3.connect(str(tmp_path / "local.db"))
    con.execute("UPDATE videos SET processing_started_at = ? WHERE video_id = ?", (STARTED, VID))
    con.commit()
    con.close()

    def watcher():
        """A freshly started processing watcher on the same machine."""
        orch = object.__new__(o.ProcessingOrchestrator)
        orch.db = db
        orch.hostname = "HOST"
        orch.processing_dir = folder
        return orch

    first = watcher()
    assert first._hand_score_precheck(VID, folder) is False      # held before the stop

    monkeypatch.setattr(o.BaseOrchestrator, "_reclaim_orphaned_work", lambda self: {})
    monkeypatch.setattr(o.ProcessingOrchestrator, "_export_missed_to_central", lambda self: 0)
    monkeypatch.setattr(o.ProcessingOrchestrator, "_set_aside_finished_copies", lambda self: 0)

    class H:
        pass
    H.folder, H.pose, H.db, H.watcher, H.versions = folder, pose, db, watcher, v
    return H


def _restarted(held):
    orch = held.watcher()
    orch._reclaim_orphaned_work()
    return orch


def _no_rerun(monkeypatch, allow=()):
    """Fail if an algorithm step runs (other than those in ``allow``)."""
    import mousereach.segmentation.core.batch as segb
    import mousereach.reach.core.batch as reachb
    import mousereach.outcomes.core.batch as outb
    import mousereach.assignment.run as assign
    import mousereach.pipeline.manifest as man
    import mousereach.pipeline.triage as tri
    import mousereach.watcher.review_gate as rg

    class Ran(Exception):
        pass

    def _must_not_run(name):
        def f(*a, **k):
            if name in allow:
                raise Ran(name)
            raise AssertionError("%s was run again for a held video" % name)
        return f
    monkeypatch.setattr(segb, "process_single", _must_not_run("segmentation"))
    monkeypatch.setattr(reachb, "process_single", _must_not_run("reach"))
    monkeypatch.setattr(outb, "process_single", _must_not_run("outcome"))
    monkeypatch.setattr(assign, "assign_reaches_for_video", _must_not_run("assignment"))
    monkeypatch.setattr(man, "create_processing_manifest", lambda **k: {})
    monkeypatch.setattr(tri, "triage_video", lambda **k: (_ for _ in ()).throw(RuntimeError("no QC")))
    monkeypatch.setattr(rg, "run_gate", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("the gate ran on a held video")))
    return Ran


def test_the_hold_is_restored_after_a_restart(held):
    orch = _restarted(held)
    assert orch._held_for_hand_scores(VID)          # still held, not counted for intake
    assert not orch._waiting_for_hand_scores(VID)   # but checked once more
    items = json.loads((held.folder.parent / "nas" / "attention" / "HOST_hand_scores.json")
                       .read_text())["items"]
    assert [i["video_id"] for i in items] == [VID]  # the list a person sees is kept


def test_a_restored_hold_with_current_outputs_reuses_its_analysis(held, monkeypatch):
    _no_rerun(monkeypatch)
    orch = _restarted(held)
    orch._run_pipeline({"id": VID, "data": {"video_id": VID, "dlc_output_path": str(held.pose)}})
    assert orch._held_for_hand_scores(VID)          # checked again, held again
    assert held.db.get_video(VID)["state"] == "processing"
    assert not orch._score_waits[VID].get("restored")


def test_a_restored_hold_remakes_outputs_older_than_the_running_code(held, monkeypatch):
    """A restart that deployed a new reach detector: segmentation is reused,
    reach detection and every step after it run again."""
    p = held.folder / f"{VID}_reaches.json"
    p.write_text(json.dumps({"detector_version": "an-older-version"}), encoding="utf-8")
    import mousereach.pipeline.versions as pv
    newer = dict(held.versions)
    monkeypatch.setattr(pv, "code_stamped_versions", lambda: dict(newer))
    monkeypatch.setattr(pv, "get_current_versions", lambda *a, **k: {
        "versions": {"dlc_scorer": SCORER},
        "compatible_versions": {}})
    Ran = _no_rerun(monkeypatch, allow=("reach",))
    orch = _restarted(held)
    with pytest.raises(Ran):                        # reach detection ran; segmentation did not
        orch._run_pipeline({"id": VID, "data": {"video_id": VID,
                                                "dlc_output_path": str(held.pose)}})


def test_a_video_that_moved_on_is_not_restored(held):
    held.db.force_state(VID, "processed", reason="test: it moved on")
    orch = _restarted(held)
    assert not orch._held_for_hand_scores(VID)


def test_a_check_that_could_not_run_is_not_restored(held, monkeypatch):
    monkeypatch.setattr(hs, "settings", lambda: (_ for _ in ()).throw(RuntimeError("unreadable")))
    first = held.watcher()
    assert first._hand_score_precheck(VID, held.folder) is False
    orch = _restarted(held)
    assert not orch._held_for_hand_scores(VID)


def test_a_restored_hold_reuses_an_output_whose_version_is_declared_compatible(held, monkeypatch):
    """A deploy that bumped the reach detector and declared the old version
    compatible: the held video's reaches are still reused."""
    import mousereach.pipeline.versions as pv
    old = held.versions["reach_detector"]
    newer = dict(held.versions, reach_detector="a-newer-version")
    monkeypatch.setattr(pv, "code_stamped_versions", lambda: dict(newer))
    monkeypatch.setattr(pv, "get_current_versions", lambda *a, **k: {
        "versions": {"segmenter": newer["segmenter"], "reach_detector": "a-newer-version",
                     "outcome_detector": newer["outcome_detector"],
                     "assignment": newer["assignment"], "dlc_scorer": SCORER},
        "compatible_versions": {"reach_detector": [old]}})
    _no_rerun(monkeypatch)
    orch = _restarted(held)
    orch._run_pipeline({"id": VID, "data": {"video_id": VID, "dlc_output_path": str(held.pose)}})
    assert orch._held_for_hand_scores(VID)


def test_a_restored_hold_of_a_review_return_resumes_where_the_return_said(held, monkeypatch):
    """A video back from triage (its row carries resume_stage) and held for its
    scores, then a restart that deployed a new segmenter not yet declared: the
    cuts the person answered against are kept, the run resumes at reach -- the
    same as a review return that was never held."""
    import mousereach.pipeline.versions as pv
    newer = dict(held.versions, segmenter="a-newer-segmenter")
    monkeypatch.setattr(pv, "code_stamped_versions", lambda: dict(newer))
    held.db.set_fields(VID, resume_stage="reach")
    Ran = _no_rerun(monkeypatch, allow=("reach",))
    orch = _restarted(held)
    with pytest.raises(Ran):                        # reach ran; segmentation did not
        orch._run_pipeline({"id": VID, "data": {"video_id": VID,
                                                "dlc_output_path": str(held.pose)}})
