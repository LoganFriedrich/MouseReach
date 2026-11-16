"""A re-run of a FILED video that fails for good, or cannot start, puts the
row back to 'archived' -- never leaves it 'failed'.

WHY: filed means done. A failed re-run changes nothing in the filed results,
but the row said 'failed' for good: every status view showed the video as
failed, and the version scan (which walks only 'archived' rows) never looked
at it again (orchestrator.ProcessingOrchestrator._restore_filed_after_failed_rerun).

Built under tmp_path with invented ids; no real folder or database.
"""
import json
from types import SimpleNamespace

import pytest

from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB

# What the row says (watcher.db.FAILED_RERUN_PREFIX), spelled out so the
# operator-facing text is pinned too.
FAILED_RERUN_PREFIX = "re-run of a filed video failed"
from mousereach.watcher.orchestrator import ProcessingOrchestrator

VID = "20990101_ABC0301_P1"
POSE = f"{VID}DLC_resnet50_TestModelshuffle1_100.h5"


@pytest.fixture
def orch(tmp_path, monkeypatch):
    root = tmp_path / "proc_root"
    proc = root / "Processing"
    proc.mkdir(parents=True)
    nas = tmp_path / "nas"
    (nas / "attention").mkdir(parents=True)
    monkeypatch.setattr(Paths, "PROCESSING_ROOT", root, raising=False)
    monkeypatch.setattr(Paths, "NAS_ROOT", nas, raising=False)
    o = object.__new__(ProcessingOrchestrator)
    o.config = SimpleNamespace(max_retries=3, max_local_pending=4)
    o.db = WatcherDB(tmp_path / "w.db")
    o.processing_dir = proc
    o.hostname = "NODE-T"
    o._score_waits = {}
    o._archive_backoff = {}
    o._get_priority_animal = lambda: None
    return o


def _filed(o, vid=VID):
    o.db.register_video(video_id=vid, source_path="x")
    o.db.force_state(vid, "processing", reason="test")
    o.db.update_state(vid, "processed")
    o.db.update_state(vid, "archiving")
    o.db.update_state(vid, "archived")
    return o.db.get_video(vid)["archived_at"]


def _working_copy(o, vid=VID):
    for n in (f"{vid}.mp4", POSE.replace(VID, vid), f"{vid}_segments.json"):
        (o.processing_dir / n).write_text("x", encoding="utf-8")


def _fail(o, times, vid=VID, scope="kinematics"):
    o.db.force_state(vid, "outdated", reason="test", reprocess_scope=scope)
    o.db.force_state(vid, "processing", reason="test re-run")
    for _ in range(times):
        o.db.mark_failed(vid, "kinematics failed: boom")


def test_a_filed_rerun_out_of_retries_goes_back_to_archived(orch):
    filed_at = _filed(orch)
    _working_copy(orch)
    _fail(orch, 3)
    orch._select_work_item()
    row = orch.db.get_video(VID)
    assert row["state"] == "archived"
    assert row["archived_at"] == filed_at                 # when it was filed, kept
    assert row["error_message"].startswith(FAILED_RERUN_PREFIX)
    assert "boom" in row["error_message"]
    assert row["error_count"] == 0                        # the next re-run has its retries
    assert row["reprocess_scope"] == "kinematics"         # scope kept
    assert not any(orch.processing_dir.iterdir())         # working copies set aside...
    aside = list((orch.processing_dir.parent / "_finished_elsewhere").rglob(f"{VID}*"))
    assert len(aside) == 3                                # ...not deleted
    listed = json.loads((Paths.NAS_ROOT / "attention" / "NODE-T_failed_reruns.json")
                        .read_text(encoding="utf-8"))["items"]
    assert [i["video_id"] for i in listed] == [VID] and listed[0]["needs_person"]


def test_a_filed_rerun_with_retries_left_is_retried_not_restored(orch):
    _filed(orch)
    _working_copy(orch)
    _fail(orch, 1)
    orch.db.set_fields(VID, last_error_at=None)
    work = orch._select_work_item()
    assert work and work["type"] == "pipeline"
    assert orch.db.get_video(VID)["state"] == "processing"


def test_a_never_filed_video_out_of_retries_stays_failed(orch):
    orch.db.register_video(video_id=VID, source_path="x")
    _working_copy(orch)
    orch.db.force_state(VID, "processing", reason="test")
    for _ in range(3):
        orch.db.mark_failed(VID, "boom")
    orch._select_work_item()
    assert orch.db.get_video(VID)["state"] == "failed"


def test_a_returned_review_that_fails_stays_failed_for_a_person(orch):
    """A review return clears the scope: its working copy holds a person's
    answers the filed results lack, so it is never set aside as 'filed'."""
    _filed(orch)
    _working_copy(orch)
    _fail(orch, 3, scope=None)
    orch._select_work_item()
    assert orch.db.get_video(VID)["state"] == "failed"
    assert (orch.processing_dir / f"{VID}.mp4").is_file()


def test_the_failed_reruns_list_is_written_when_its_folder_is_missing(orch):
    import shutil
    shutil.rmtree(Paths.NAS_ROOT / "attention", ignore_errors=True)
    _filed(orch)
    _working_copy(orch)
    _fail(orch, 3)
    orch._select_work_item()
    listed = json.loads((Paths.NAS_ROOT / "attention" / "NODE-T_failed_reruns.json")
                        .read_text(encoding="utf-8"))["items"]
    assert [i["video_id"] for i in listed] == [VID]


def test_a_restart_keeps_the_failed_rerun_message(orch):
    _filed(orch)
    _working_copy(orch)
    _fail(orch, 3)
    orch._select_work_item()
    other = "20990101_ABC0302_P1"
    _filed(orch, other)
    orch.db.set_fields(other, error_message="an old error")
    orch.db.clear_errors_on_archived()
    assert orch.db.get_video(VID)["error_message"].startswith(FAILED_RERUN_PREFIX)
    assert orch.db.get_video(other)["error_message"] is None


def test_filing_again_clears_the_message(orch):
    _filed(orch)
    _working_copy(orch)
    _fail(orch, 3)
    orch._select_work_item()
    orch.db.force_state(VID, "processed", reason="test: the next re-run finished")
    orch.db.update_state(VID, "archiving")
    orch.db.update_state(VID, "archived")
    assert orch.db.get_video(VID)["error_message"] is None
    orch._write_failed_rerun_attention()
    listed = json.loads((Paths.NAS_ROOT / "attention" / "NODE-T_failed_reruns.json")
                        .read_text(encoding="utf-8"))["items"]
    assert listed == []


def test_a_rerun_whose_copy_down_fails_goes_back_to_archived(orch, tmp_path, monkeypatch):
    """Marked 'failed', the retry ran the pipeline on the partial set the
    copy guard is there to stop."""
    import mousereach.watcher.orchestrator as o
    from mousereach.watcher import repose
    filed = tmp_path / "Analyzed" / "PROJ" / "COHORT"
    filed.mkdir(parents=True)
    for n in (f"{VID}.mp4", POSE, f"{VID}_segments.json", f"{VID}_reaches.json"):
        (filed / n).write_text("x", encoding="utf-8")
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed", raising=False)
    monkeypatch.setattr(repose, "declared_scorer", lambda: "")
    monkeypatch.setattr(o, "archived_pose_candidates", lambda *a, **k: [filed / POSE])
    monkeypatch.setattr("mousereach.archive.core.get_archive_destination", lambda vid: filed)
    orch.staging_dir = None

    def _copy(src, dest, verify=True):
        if src.name.endswith("_reaches.json"):
            return False
        dest.write_bytes(src.read_bytes())
        return True
    monkeypatch.setattr(o, "safe_copy", _copy)
    filed_at = _filed(orch)
    orch.db.force_state(VID, "outdated", reason="test", reprocess_scope="kinematics")
    orch._reprocess_video({"id": VID, "data": orch.db.get_video(VID)})
    row = orch.db.get_video(VID)
    assert row["state"] == "archived" and row["archived_at"] == filed_at
    assert "copying it down failed" in row["error_message"]
    assert not any(orch.processing_dir.iterdir())          # partial set put aside


# ---------------------------------------------------------------------------
# A video that was never filed is never put to 'archived'
# ---------------------------------------------------------------------------
# WHY: rows reach 'outdated' from states other than 'archived' (the dashboard's
# Re-run accepts 'failed' and 'unresolvable' rows; the re-pose round trip
# narrows 'unresolvable' rows to 'outdated'). Put to 'archived', such a video
# would read as filed -- passed every check -- although no check ever passed.

def _never_filed_outdated(o):
    o.db.register_video(video_id=VID, source_path="x")
    o.db.force_state(VID, "processing", reason="test: first pass")
    o.db.mark_failed(VID, "first pass failed")
    o.db.force_state(VID, "outdated", reason="test: Re-run pressed",
                     reprocess_scope="segmentation", mark_reason="redo it")


def test_a_never_filed_rerun_that_cannot_start_is_failed(orch):
    _never_filed_outdated(orch)
    # The "cannot start" branch's own call (no processing folder configured);
    # the pose search before it is not the subject here.
    orch._restore_filed_after_failed_rerun(
        VID, "could not start: PROCESSING path not configured")
    row = orch.db.get_video(VID)
    assert row["state"] == "failed"
    assert row["archived_at"] is None
    assert "PROCESSING path not configured" in row["error_message"]


def test_a_never_filed_rerun_whose_copy_down_fails_is_failed_not_archived(orch, tmp_path, monkeypatch):
    import mousereach.watcher.orchestrator as o
    from mousereach.watcher import repose
    filed = tmp_path / "Analyzed" / "PROJ" / "COHORT"
    filed.mkdir(parents=True)
    for n in (f"{VID}.mp4", POSE, f"{VID}_segments.json", f"{VID}_reaches.json"):
        (filed / n).write_text("x", encoding="utf-8")
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed", raising=False)
    monkeypatch.setattr(repose, "declared_scorer", lambda: "")
    monkeypatch.setattr(o, "archived_pose_candidates", lambda *a, **k: [filed / POSE])
    monkeypatch.setattr("mousereach.archive.core.get_archive_destination", lambda vid: filed)
    orch.staging_dir = None

    def _copy(src, dest, verify=True):
        if src.name.endswith("_reaches.json"):
            return False
        dest.write_bytes(src.read_bytes())
        return True
    monkeypatch.setattr(o, "safe_copy", _copy)
    _never_filed_outdated(orch)
    orch._reprocess_video({"id": VID, "data": orch.db.get_video(VID)})
    row = orch.db.get_video(VID)
    assert row["state"] == "failed"
    assert row["archived_at"] is None
    assert any(orch.processing_dir.iterdir())             # nothing set aside


# ---------------------------------------------------------------------------
# The version scan does not offer the same failing re-run again and again
# ---------------------------------------------------------------------------
# WHY: the restored row keeps its out-of-date manifest. Without a check the
# next scan marks it 'outdated', the re-run fails the same way through all its
# retries, it is restored, and so on for ever (reprocessor.failed_rerun_stamp).

CURRENT = {"segmenter": "2.0", "reach_detector": "3.0", "outcome_detector": "4.0",
           "assignment": "1.0", "kinematic_extractor": "5.0"}


@pytest.fixture
def scan_rig(orch, monkeypatch):
    import mousereach.config as config
    import mousereach.review.causal_review_io as crio
    monkeypatch.setattr(config, "filing_rules", lambda: {
        "ABC": {"folder": "Project_A", "cohort_folder": "{CODE}{NN}"}})
    nas = Paths.NAS_ROOT
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", nas / "Analyzed", raising=False)
    cohort = nas / "Analyzed" / "Project_A" / "ABC03"
    cohort.mkdir(parents=True)
    (nas / "pipeline_versions.json").write_text(json.dumps({"versions": CURRENT}),
                                                encoding="utf-8")
    m = {"video_stem": VID, "created_at": "2099-01-01T10:00:00",
         "pipeline_versions": dict(CURRENT, kinematic_extractor="4.0")}
    (cohort / f"{VID}_processing_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    (cohort / f"{VID}_features.json").write_text("{}", encoding="utf-8")
    (cohort / f"{VID}.mp4").write_bytes(b"x")
    monkeypatch.setattr(crio, "review_path_index", lambda stems: {})
    monkeypatch.setattr(crio, "resolve_review_path", lambda *a, **k: None)
    _filed(orch)
    _working_copy(orch)
    _fail(orch, 3)
    orch._select_work_item()                               # restored to 'archived'
    assert orch.db.get_video(VID)["state"] == "archived"

    def scan():
        from mousereach.watcher.reprocessor import ReprocessingScanner
        return ReprocessingScanner(orch.db, nas).scan(mark_outdated=True,
                                                       adopt_orphans=False)
    return type("ScanRig", (), {"nas": nas, "cohort": cohort, "scan": staticmethod(scan)})


def test_a_restored_failed_rerun_is_not_marked_again_by_the_next_scan(orch, scan_rig):
    summary = scan_rig.scan()
    assert orch.db.get_video(VID)["state"] == "archived"
    assert summary["failed_rerun_waiting"] == 1
    assert summary["failed_rerun_waiting_videos"] == [VID]
    scan_rig.scan()                                        # nor on the one after
    assert orch.db.get_video(VID)["state"] == "archived"


def test_a_declaration_change_offers_the_failed_rerun_again(orch, scan_rig):
    (scan_rig.nas / "pipeline_versions.json").write_text(
        json.dumps({"versions": dict(CURRENT, kinematic_extractor="5.1")}), encoding="utf-8")
    scan_rig.scan()
    row = orch.db.get_video(VID)
    assert row["state"] == "outdated" and row["reprocess_scope"] == "kinematics"


def test_ground_truth_saved_since_offers_the_failed_rerun_again(orch, scan_rig):
    (scan_rig.cohort / f"{VID}_unified_ground_truth.json").write_text(
        json.dumps({"last_modified_at": "2099-01-02T10:00:00"}), encoding="utf-8")
    scan_rig.scan()
    assert orch.db.get_video(VID)["state"] == "outdated"


def test_a_declaration_that_makes_the_filed_results_current_clears_the_failure(orch, scan_rig):
    (scan_rig.nas / "pipeline_versions.json").write_text(json.dumps({
        "versions": CURRENT,
        "compatible_versions": {"kinematic_extractor": ["4.0"]}}), encoding="utf-8")
    scan_rig.scan()
    row = orch.db.get_video(VID)
    assert row["state"] == "archived" and row["error_message"] is None
