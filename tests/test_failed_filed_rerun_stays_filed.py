"""A re-run of a FILED video that fails for good, or cannot start, puts the
row back to 'archived' -- never leaves it 'failed'.

WHY: filed means done. A failed re-run changes nothing in the filed results,
but the row said 'failed' for good: every status view showed the video as
failed, and the version scan (which walks only 'archived' rows) never looked
at it again (orchestrator.ProcessingOrchestrator._restore_filed_after_failed_rerun).

"Filed" is decided by what is on disk: the row was filed (archived_at) AND the
video's filed features and manifest are in its filed folder now. So a review
return or a stale-outputs gate, which wipe the re-run scope, still count; a
video mousereach-repose-videos took apart (its filed outputs moved into
Analyzed/Archive/) does not.

Built under tmp_path with invented ids; no real folder or database.
"""
import json
from pathlib import Path
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
    nas.mkdir()
    monkeypatch.setattr(Paths, "PROCESSING_ROOT", root, raising=False)
    monkeypatch.setattr(Paths, "NAS_ROOT", nas, raising=False)
    import mousereach.config as config
    monkeypatch.setattr(config, "filing_rules", lambda: {
        "ABC": {"folder": "Project_A", "cohort_folder": "{CODE}{NN}"}})
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", nas / "Analyzed", raising=False)
    o = object.__new__(ProcessingOrchestrator)
    o.config = SimpleNamespace(max_retries=3, max_local_pending=4)
    o.db = WatcherDB(tmp_path / "w.db")
    o.processing_dir = proc
    o.hostname = "NODE-T"
    o._score_waits = {}
    o._archive_backoff = {}
    o._get_priority_animal = lambda: None
    return o


def _filed_folder(vid=VID):
    from mousereach.archive.core import get_archive_destination
    return get_archive_destination(vid)


def _filed(o, vid=VID):
    """Filed: the row, and the filed features and manifest in Analyzed."""
    o.db.register_video(video_id=vid, source_path="x")
    o.db.force_state(vid, "processing", reason="test")
    o.db.update_state(vid, "processed")
    o.db.update_state(vid, "archiving")
    o.db.update_state(vid, "archived")
    d = _filed_folder(vid)
    d.mkdir(parents=True, exist_ok=True)
    for n in (f"{vid}_features.json", f"{vid}_processing_manifest.json"):
        if not (d / n).exists():
            (d / n).write_text("{}", encoding="utf-8")
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
    assert row["reprocess_scope"] is None                 # the failed round's scope ends with it
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


@pytest.fixture
def durable_store(tmp_path, monkeypatch):
    """A durable review store under tmp_path; ``rig.path`` is where this
    video's copy of record goes. ``rig.unconfigure()`` makes it a machine
    with no durable store at all."""
    import mousereach.review.causal_review_io as crio
    state = {"dir": tmp_path / "durable_reviews"}
    monkeypatch.setattr(
        crio, "durable_review_path",
        lambda stem: None if state["dir"] is None
        else state["dir"] / f"{stem}_causal_review.json")
    return type("Durable", (), {
        "path": state["dir"] / f"{VID}_causal_review.json",
        "unconfigure": staticmethod(lambda: state.update(dir=None))})


def test_a_returned_review_of_a_filed_video_that_fails_goes_back_to_archived(orch, durable_store):
    """A review return wipes the re-run scope; the filed results still stand,
    so it is not left 'failed' for good. The person's answers are in the
    durable review store (the return copied them there), so the file in the
    processing folder is a duplicate, and setting it aside loses nothing."""
    filed_at = _filed(orch)
    _working_copy(orch)
    answers = '{"reviewed_at": "2099-01-02T10:00:00"}'
    durable = durable_store.path
    durable.parent.mkdir(parents=True, exist_ok=True)
    durable.write_text(answers, encoding="utf-8")
    (orch.processing_dir / f"{VID}_causal_review.json").write_text(answers, encoding="utf-8")
    _fail(orch, 3, scope=None)                            # returned: no scope on the row
    orch.db.set_fields(VID, resume_stage="assignment", held_at_step="outcome")
    orch._select_work_item()
    row = orch.db.get_video(VID)
    assert row["state"] == "archived" and row["archived_at"] == filed_at
    assert row["error_message"].startswith(FAILED_RERUN_PREFIX)
    assert row["resume_stage"] is None                    # the next re-run starts afresh
    assert durable.read_text(encoding="utf-8") == answers  # the answers of record stay put
    aside = list((orch.processing_dir.parent / "_finished_elsewhere").rglob(
        f"{VID}_causal_review.json"))
    assert len(aside) == 1                                # the working copy moved, not deleted


def test_a_filed_rerun_whose_scope_a_stale_gate_wiped_still_goes_back_to_archived(orch):
    """A stale-outputs gate wipes the kept scope so the next pick recomputes
    the outputs; three failures after that still leave the video filed."""
    filed_at = _filed(orch)
    _working_copy(orch)
    orch.db.force_state(VID, "outdated", reason="test", reprocess_scope="kinematics")
    orch.db.force_state(VID, "processing", reason="test re-run")
    held = orch._gate_held(VID, lambda *a, **k: "stale_outputs", "clean", None, 0.0)
    assert held
    for _ in range(3):
        orch.db.mark_failed(VID, "kinematics failed: boom")
    orch._select_work_item()
    row = orch.db.get_video(VID)
    assert row["state"] == "archived" and row["archived_at"] == filed_at
    assert row["error_message"].startswith(FAILED_RERUN_PREFIX)


def _taken_apart_by_repose(o, vid=VID):
    """What mousereach-repose-videos does before a re-run: the filed pose and
    every output made from it move into Analyzed/Archive/<label>/."""
    import shutil
    d = _filed_folder(vid)
    archive = Paths.ANALYZED_OUTPUT / "Archive" / "relabel" / d.name
    archive.mkdir(parents=True)
    for f in list(d.glob(f"{vid}_*")):
        shutil.move(str(f), str(archive / f.name))


@pytest.mark.parametrize("held_first", [False, True])
def test_a_reposed_video_whose_rerun_fails_stays_failed(orch, held_first):
    """Its filed results are no longer in Analyzed, so 'archived' would read
    as filed with nothing filed. Also for a video that sat in triage (after a
    re-run of it was held) when it was re-posed."""
    _filed(orch)
    if held_first:
        orch.db.force_state(VID, "triage", reason="test: a re-run of it was held")
    _taken_apart_by_repose(orch)
    _working_copy(orch)
    orch.db.force_state(VID, "outdated", reason="test: re-posed", reprocess_scope="segmentation")
    orch.db.force_state(VID, "processing", reason="test re-run")
    for _ in range(3):
        orch.db.mark_failed(VID, "segmentation failed: boom")
    orch._select_work_item()
    row = orch.db.get_video(VID)
    assert row["state"] == "failed"
    assert not row["error_message"].startswith(FAILED_RERUN_PREFIX)
    assert (orch.processing_dir / f"{VID}.mp4").is_file()  # nothing set aside
    orch._select_work_item()                               # and it is not re-read as filed
    assert orch.db.get_video(VID)["state"] == "failed"


def test_a_reposed_video_whose_rerun_cannot_start_is_failed(orch):
    _filed(orch)
    _taken_apart_by_repose(orch)
    orch.db.force_state(VID, "outdated", reason="test: re-posed", reprocess_scope="segmentation")
    orch._restore_filed_after_failed_rerun(VID, "could not start: test")
    assert orch.db.get_video(VID)["state"] == "failed"


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


@pytest.fixture
def copy_down(orch, tmp_path, monkeypatch):
    """A filed folder holding the video's files, and a copy that fails for
    its _reaches.json (a file locked on the share) until ``rig.unlock()``."""
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
    locked = {"on": True}

    def _copy(src, dest, verify=True):
        if locked["on"] and src.name.endswith("_reaches.json"):
            return False
        dest.write_bytes(src.read_bytes())
        return True
    monkeypatch.setattr(o, "safe_copy", _copy)
    ran = []
    monkeypatch.setattr(orch, "_run_pipeline", lambda work: ran.append(work["id"]),
                        raising=False)

    def attempt():
        orch._reprocess_video({"id": VID, "data": orch.db.get_video(VID)})
        return orch.db.get_video(VID)

    def unlock():
        locked["on"] = False
    return type("CopyDown", (), {"attempt": staticmethod(attempt),
                                 "unlock": staticmethod(unlock), "ran": ran})


def test_a_copy_down_that_fails_once_keeps_the_rerun_owed(orch, copy_down):
    """One failed copy (a file locked for a moment) is not a re-run that
    failed for good: the row stays 'outdated' with no fingerprint, waits its
    hour, and nothing partial is left in the processing folder."""
    _filed(orch)
    orch.db.force_state(VID, "outdated", reason="test", reprocess_scope="kinematics")
    row = copy_down.attempt()
    assert row["state"] == "outdated"
    assert row["reprocess_scope"] == "kinematics"
    assert row["pipeline_versions_hash"] is None
    assert "copying it down failed" in row["error_message"]
    assert int(row["error_count"] or 0) == 0               # the pipeline's retries untouched
    assert not any(orch.processing_dir.iterdir())          # partial set put aside...
    aside = list((orch.processing_dir.parent / "_finished_elsewhere").rglob(f"{VID}*"))
    assert aside                                           # ...not deleted
    assert copy_down.ran == []
    # It waits its hour before the next try.
    assert not orch._rerun_start_wait_over(row)
    orch.db.set_fields(VID, last_error_at="2000-01-01T00:00:00")
    assert orch._rerun_start_wait_over(orch.db.get_video(VID))


def test_the_next_try_that_copies_starts_the_rerun_and_clears_the_message(orch, copy_down):
    _filed(orch)
    orch.db.force_state(VID, "outdated", reason="test", reprocess_scope="kinematics")
    copy_down.attempt()
    copy_down.unlock()
    row = copy_down.attempt()
    assert row["state"] == "processing"
    assert row["error_message"] is None
    assert copy_down.ran == [VID]


def test_a_filed_rerun_whose_copy_down_keeps_failing_goes_back_to_archived(orch, copy_down):
    filed_at = _filed(orch)
    orch.db.force_state(VID, "outdated", reason="test", reprocess_scope="kinematics")
    for _ in range(2):
        assert copy_down.attempt()["state"] == "outdated"
    row = copy_down.attempt()                              # the third try in a row
    assert row["state"] == "archived" and row["archived_at"] == filed_at
    assert row["error_message"].startswith(FAILED_RERUN_PREFIX)
    assert "copying it down failed" in row["error_message"]
    assert not any(orch.processing_dir.iterdir())
    assert copy_down.ran == []


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


def test_a_never_filed_rerun_whose_copy_down_fails_never_runs_on_a_partial_set(orch, copy_down):
    """A row that was never filed: the partial working set is set aside on
    every failed try, so neither the next try nor the failed-retry bucket can
    run the pipeline on it (missing human cuts would be re-cut by the
    algorithm), and it is never put to 'archived'."""
    _never_filed_outdated(orch)
    row = copy_down.attempt()
    assert row["state"] == "outdated"
    assert not any(orch.processing_dir.iterdir())
    for _ in range(2):
        row = copy_down.attempt()
    assert row["state"] == "failed"
    assert row["archived_at"] is None
    assert not any(orch.processing_dir.iterdir())          # nothing partial to retry
    assert not orch._failed_retry_due(dict(row, last_error_at=None))
    assert copy_down.ran == []


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
    nas = Paths.NAS_ROOT
    cohort = nas / "Analyzed" / "Project_A" / "ABC03"
    cohort.mkdir(parents=True, exist_ok=True)
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


def test_the_work_loop_leaves_a_rerun_that_could_not_start_until_its_hour_is_over(orch, copy_down):
    _filed(orch)
    orch.db.force_state(VID, "outdated", reason="test", reprocess_scope="kinematics")
    copy_down.attempt()
    work = orch._select_work_item()
    assert not (work and work.get("id") == VID)
    orch.db.set_fields(VID, last_error_at="2000-01-01T00:00:00")
    work = orch._select_work_item()
    assert work and work["type"] == "reprocess" and work["id"] == VID


# ---------------------------------------------------------------------------
# Setting files aside never takes the only copy of anything
# ---------------------------------------------------------------------------
# WHY: the set-aside folder is named for finished videos and operators clear
# it. A file whose only copy is in the processing folder -- a never-filed
# video's own working files, a person's re-cut of a filed video -- must stay
# where the row and the next try look for it.

def test_a_never_filed_rerun_keeps_its_own_files_when_one_copy_down_fails(orch, copy_down):
    """Re-run pressed on a failed first pass: its video, pose, a person's
    cuts and review file are in the processing folder and nowhere else, and
    the row's pose path points there. One file of its results folder fails to
    copy. Only what the try copied down is set aside."""
    from mousereach.archive.core import get_archive_destination
    _never_filed_outdated(orch)
    own = {f"{VID}.mp4": "the video", POSE: "the pose",
           f"{VID}_segments.json": '{"boundary_source": "human"}',
           f"{VID}_causal_review.json": '{"answers": 1}'}
    for name, text in own.items():
        (orch.processing_dir / name).write_text(text, encoding="utf-8")
    orch.db.set_fields(VID, dlc_output_path=str(orch.processing_dir / POSE),
                       current_path=str(orch.processing_dir / f"{VID}.mp4"))
    # Its results folder holds two earlier outputs and nothing else; the copy
    # of _reaches.json fails (copy_down).
    results = get_archive_destination(VID)
    for name in (f"{VID}.mp4", POSE, f"{VID}_segments.json"):
        (results / name).unlink()
    (results / f"{VID}_pellet_outcomes.json").write_text("filed", encoding="utf-8")

    for _ in range(3):
        row = copy_down.attempt()
        for name, text in own.items():                     # its own files: untouched
            assert (orch.processing_dir / name).read_text(encoding="utf-8") == text
        assert Path(row["dlc_output_path"]).is_file()
        assert not (orch.processing_dir / f"{VID}_pellet_outcomes.json").exists()
    assert row["state"] == "failed" and row["archived_at"] is None
    assert "no pose and no video found" not in row["error_message"]
    # What was copied down for the try was set aside, and the row says where.
    aside = orch.processing_dir.parent / "_finished_elsewhere"
    # Each try's copies are kept apart (a later try's set-aside never replaces
    # an earlier one's files), so the name can appear once per try.
    assert {p.name for p in aside.rglob(f"{VID}*") if p.is_file()} == {f"{VID}_pellet_outcomes.json"}
    assert str(orch._set_aside_folder()) in row["error_message"]
    assert copy_down.ran == []


def test_a_failed_rerun_carrying_a_persons_recut_stays_failed_with_its_files(orch):
    """A filed video was re-cut by a person in deep review and its return run
    failed for good. The re-cut exists only in the processing folder: the
    row stays 'failed' and nothing is set aside."""
    _filed(orch)
    (_filed_folder() / f"{VID}_segments.json").write_text(
        json.dumps({"boundaries": [10, 20], "boundary_source": "algorithm"}), encoding="utf-8")
    _working_copy(orch)
    recut = json.dumps({"boundaries": [12, 22], "boundary_source": "human"})
    (orch.processing_dir / f"{VID}_segments.json").write_text(recut, encoding="utf-8")
    _fail(orch, 3, scope=None)
    orch._select_work_item()
    row = orch.db.get_video(VID)
    assert row["state"] == "failed"
    assert (orch.processing_dir / f"{VID}_segments.json").read_text(encoding="utf-8") == recut
    assert not (orch.processing_dir.parent / "_finished_elsewhere").exists()
    orch._select_work_item()                               # and it is left alone after
    assert orch.db.get_video(VID)["state"] == "failed"


def test_a_persons_cuts_that_are_the_filed_ones_do_not_keep_the_row_failed(orch):
    """Human cuts copied down from the filed folder are not at risk: the
    filed folder has them."""
    _filed(orch)
    cuts = json.dumps({"boundaries": [12, 22], "boundary_source": "human"})
    (_filed_folder() / f"{VID}_segments.json").write_text(cuts, encoding="utf-8")
    _working_copy(orch)
    (orch.processing_dir / f"{VID}_segments.json").write_text(cuts, encoding="utf-8")
    _fail(orch, 3)
    orch._select_work_item()
    assert orch.db.get_video(VID)["state"] == "archived"


def test_a_restored_row_says_where_the_failed_runs_files_went(orch):
    _filed(orch)
    _working_copy(orch)
    _fail(orch, 3)
    orch._select_work_item()
    row = orch.db.get_video(VID)
    assert row["state"] == "archived"
    assert str(orch._set_aside_folder()) in row["error_message"]
    listed = json.loads((Paths.NAS_ROOT / "attention" / "NODE-T_failed_reruns.json")
                        .read_text(encoding="utf-8"))["items"]
    assert str(orch._set_aside_folder()) in listed[0]["says"]


def test_a_failed_filed_run_whose_video_is_elsewhere_keeps_its_retries(orch, tmp_path):
    """A review return can run a video whose mp4 and pose stayed outside the
    processing folder. One failure of that run is not 'working files gone':
    the row is not put back to 'archived' while it has retries left."""
    _filed(orch)
    elsewhere = tmp_path / "staging"
    elsewhere.mkdir()
    (elsewhere / f"{VID}.mp4").write_text("x", encoding="utf-8")
    (elsewhere / POSE).write_text("x", encoding="utf-8")
    _fail(orch, 1, scope=None)
    orch.db.set_fields(VID, current_path=str(elsewhere / f"{VID}.mp4"),
                       dlc_output_path=str(elsewhere / POSE))
    orch._select_work_item()
    row = orch.db.get_video(VID)
    assert row["state"] == "failed"
    assert not (row["error_message"] or "").startswith(FAILED_RERUN_PREFIX)


def test_a_failed_filed_run_with_no_files_anywhere_goes_back_to_archived(orch):
    """Nothing a retry could run on is left (no local video, and the row's
    paths point at nothing): waiting would be for ever, so it is restored."""
    _filed(orch)
    _fail(orch, 1)
    orch.db.set_fields(VID, current_path=str(orch.processing_dir / f"{VID}.mp4"),
                       dlc_output_path="")
    orch._select_work_item()
    assert orch.db.get_video(VID)["state"] == "archived"


# ---------------------------------------------------------------------------
# A person's file with no other copy keeps the failed run where a re-run finds it
# ---------------------------------------------------------------------------
# WHY: a review queue hands the run more than cuts -- ground truth, the
# deep-review clear marker, the review's answers. Until the video is filed
# again the processing folder can hold the only copy of each (ground truth
# and the marker have no second store; the review's durable copy is missing
# when no store is configured or the copy failed). Put back to 'archived',
# the row's files all go to the set-aside folder, which operators clear, and
# the next re-run copies down from the filed folder without them
# (ProcessingOrchestrator._unfiled_person_files).

PERSON_FILES = {
    "ground truth": (f"{VID}_unified_ground_truth.json", '{"segments": [1, 2, 3]}'),
    "deep-review clear marker": (f"{VID}_deep_review_cleared.json", '{"cleared_by": "a person"}'),
    "review answers": (f"{VID}_causal_review.json", '{"answers": {"3": "retrieved"}}'),
}


def _assert_left_failed_in_place(orch, name, text):
    row = orch.db.get_video(VID)
    assert row["state"] == "failed"
    assert (orch.processing_dir / name).read_text(encoding="utf-8") == text
    assert not (orch.processing_dir.parent / "_finished_elsewhere").exists()
    assert name in row["error_message"] and "no other copy" in row["error_message"]
    assert not (row["error_message"] or "").startswith(FAILED_RERUN_PREFIX)
    assert row["pipeline_versions_hash"] is None           # no "do not offer again" mark
    return row


@pytest.mark.parametrize("kind", sorted(PERSON_FILES))
def test_a_failed_rerun_carrying_a_persons_only_copy_stays_failed_with_its_files(
        orch, durable_store, kind):
    """A filed video came back from review with a file a person made, and
    the run failed for good. Nothing else holds that file: the row stays
    'failed', the file stays where a re-run finds it, the row names it."""
    name, text = PERSON_FILES[kind]
    _filed(orch)
    _working_copy(orch)
    (orch.processing_dir / name).write_text(text, encoding="utf-8")
    _fail(orch, 3, scope=None)
    orch._select_work_item()
    _assert_left_failed_in_place(orch, name, text)
    message = orch.db.get_video(VID)["error_message"]
    orch._select_work_item()                               # and it is left alone after
    row = orch.db.get_video(VID)
    assert row["state"] == "failed" and row["error_message"] == message
    assert (orch.processing_dir / f"{VID}.mp4").is_file()  # the whole working set stays


def test_review_answers_stay_in_place_on_a_machine_with_no_durable_store(orch, durable_store):
    name, text = PERSON_FILES["review answers"]
    durable_store.unconfigure()
    _filed(orch)
    _working_copy(orch)
    (orch.processing_dir / name).write_text(text, encoding="utf-8")
    _fail(orch, 3, scope=None)
    orch._select_work_item()
    _assert_left_failed_in_place(orch, name, text)


def test_review_answers_that_differ_from_the_durable_copy_stay_in_place(orch, durable_store):
    """The durable store holds an EARLIER review of this video; the answers
    in the processing folder are newer and exist nowhere else."""
    name, text = PERSON_FILES["review answers"]
    durable_store.path.parent.mkdir(parents=True)
    durable_store.path.write_text('{"answers": {}}', encoding="utf-8")
    _filed(orch)
    _working_copy(orch)
    (orch.processing_dir / name).write_text(text, encoding="utf-8")
    _fail(orch, 3, scope=None)
    orch._select_work_item()
    _assert_left_failed_in_place(orch, name, text)


@pytest.mark.parametrize("kind", ["ground truth", "deep-review clear marker"])
def test_a_persons_file_that_is_also_in_the_filed_folder_does_not_keep_the_row_failed(
        orch, durable_store, kind):
    """Ground truth copied down from the filed folder for a re-run is a
    duplicate: the filed folder has it, so the row goes back to 'archived'
    and the copy is set aside."""
    name, text = PERSON_FILES[kind]
    _filed(orch)
    (_filed_folder() / name).write_text(text, encoding="utf-8")
    _working_copy(orch)
    (orch.processing_dir / name).write_text(text, encoding="utf-8")
    _fail(orch, 3)
    orch._select_work_item()
    assert orch.db.get_video(VID)["state"] == "archived"
    assert (_filed_folder() / name).read_text(encoding="utf-8") == text
    aside = list((orch.processing_dir.parent / "_finished_elsewhere").rglob(name))
    assert len(aside) == 1


def test_ground_truth_changed_since_it_was_filed_stays_in_place(orch, durable_store):
    """The filed folder holds an older ground truth; a person changed it in
    review. The changed one is the only copy of the change."""
    name, text = PERSON_FILES["ground truth"]
    _filed(orch)
    (_filed_folder() / name).write_text('{"segments": [1]}', encoding="utf-8")
    _working_copy(orch)
    (orch.processing_dir / name).write_text(text, encoding="utf-8")
    _fail(orch, 3, scope=None)
    orch._select_work_item()
    _assert_left_failed_in_place(orch, name, text)
    assert (_filed_folder() / name).read_text(encoding="utf-8") == '{"segments": [1]}'


@pytest.mark.parametrize("kind", sorted(PERSON_FILES))
def test_a_rerun_that_cannot_start_never_sets_a_persons_only_copy_aside(
        orch, copy_down, durable_store, kind):
    """The last "could not start" try of a filed video takes the same road:
    the copies that try brought down are set aside, the person's file is
    not, and the row is 'failed', not 'archived'."""
    name, text = PERSON_FILES[kind]
    _filed(orch)
    (orch.processing_dir / name).write_text(text, encoding="utf-8")
    orch.db.force_state(VID, "outdated", reason="test", reprocess_scope="kinematics")
    for _ in range(3):
        row = copy_down.attempt()
        assert (orch.processing_dir / name).read_text(encoding="utf-8") == text
    assert row["state"] == "failed"
    assert name in row["error_message"] and "no other copy" in row["error_message"]
    assert not list((orch.processing_dir.parent / "_finished_elsewhere").rglob(name))
    assert copy_down.ran == []


def test_a_second_set_aside_the_same_day_never_replaces_the_first(orch):
    """A filed video's re-run fails for good twice in one day. Each failed
    run's files are set aside into the same dated folder; the second set
    must not replace the first set's files of the same name, and the row
    names the folder the second set really went to."""
    _filed(orch)
    for n in (f"{VID}.mp4", POSE, f"{VID}_segments.json"):
        (orch.processing_dir / n).write_text("first run", encoding="utf-8")
    _fail(orch, 3)
    orch._select_work_item()
    first = orch.db.get_video(VID)
    assert first["state"] == "archived"

    for n in (f"{VID}.mp4", POSE, f"{VID}_segments.json"):
        (orch.processing_dir / n).write_text("second run", encoding="utf-8")
    _fail(orch, 3)
    orch._select_work_item()
    second = orch.db.get_video(VID)
    assert second["state"] == "archived"

    aside = orch.processing_dir.parent / "_finished_elsewhere"
    kept = sorted(p.read_text(encoding="utf-8")
                  for p in aside.rglob(f"{VID}_segments.json"))
    assert kept == ["first run", "second run"]             # both sets kept
    assert not list(orch.processing_dir.glob(f"{VID}*"))   # nothing left behind
    # The second row's message names the folder holding the second set.
    where = [p.parent for p in aside.rglob(f"{VID}_segments.json")
             if p.read_text(encoding="utf-8") == "second run"][0]
    assert f"moved to {where})" in second["error_message"]


def test_a_set_aside_never_moves_a_file_onto_one_already_there(orch):
    """Belt and braces: told to move into a folder that already holds a file
    of the same name, the set-aside leaves the file where it is."""
    target = orch._set_aside_folder()
    target.mkdir(parents=True)
    (target / f"{VID}_segments.json").write_text("earlier", encoding="utf-8")
    (orch.processing_dir / f"{VID}_segments.json").write_text("now", encoding="utf-8")
    assert orch._set_aside_working_copies(VID, target=target) is None
    assert (target / f"{VID}_segments.json").read_text(encoding="utf-8") == "earlier"
    assert (orch.processing_dir / f"{VID}_segments.json").read_text(encoding="utf-8") == "now"
