"""Small guards around routing into review queues.

  * The archive step waits while a bundle is still being built in
    <queue>/.incoming/<stem>/ (the queue check only sees the published folder).
  * The archive step adopts a queue folder's state only when the folder holds
    this run; a husk from an earlier round does not stop the archive.
  * A run that cannot write its processing manifest sets the previous run's
    manifest aside, so it is never archived as this run's provenance.
  * The review manifest names the chosen pose, not the first file sorted.

Everything is built under tmp_path; nothing touches a real pipeline folder.
"""
import json
from unittest import mock

from mousereach.config import Paths
from mousereach.watcher.orchestrator import BaseOrchestrator, ProcessingOrchestrator

STEM = "20250101_CNT0101_P1"


def test_archive_waits_for_a_bundle_still_being_built(tmp_path, monkeypatch):
    triage = tmp_path / "triage"
    deep = tmp_path / "deep_review"
    (triage / ".incoming" / STEM).mkdir(parents=True)
    deep.mkdir()
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)

    stub = mock.Mock()
    with mock.patch("mousereach.archive.core.archive_video") as archive:
        ProcessingOrchestrator._archive_to_nas(stub, {"id": STEM})

    archive.assert_not_called()
    stub.db.force_state.assert_not_called()
    stub.db.log_step.assert_not_called()


def test_archive_still_adopts_a_published_queue_bundle(tmp_path, monkeypatch):
    triage = tmp_path / "triage"
    (triage / STEM).mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")

    stub = mock.Mock()
    ProcessingOrchestrator._archive_to_nas(stub, {"id": STEM})

    stub.db.force_state.assert_called_once()
    assert stub.db.force_state.call_args[0][:2] == (STEM, "triage")


def _husk_and_run(tmp_path, monkeypatch, routing=None):
    """A triage folder holding a lone .mp4 from an earlier round, beside this
    run's processing manifest, still in Processing. ``routing`` (optional)
    becomes the folder's routing record; its routed_at is given in seconds
    relative to this run."""
    import os
    import time
    from datetime import datetime
    triage = tmp_path / "triage"
    husk = triage / STEM
    husk.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    proc = tmp_path / "Processing"
    proc.mkdir()
    manifest = proc / f"{STEM}_processing_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    run = time.time() - 3600
    os.utime(manifest, (run, run))
    mp4 = husk / f"{STEM}.mp4"
    mp4.write_bytes(b"video")
    os.utime(mp4, (run - 86400, run - 86400))
    if routing is not None:
        rec = dict(routing)
        rec["routed_at"] = datetime.fromtimestamp(run + rec.pop("after_run_s")).isoformat()
        (husk / f"{STEM}_routing.json").write_text(json.dumps(rec), encoding="utf-8")
    stub = mock.Mock()
    stub.processing_dir = proc
    stub.staging_dir = None
    return stub


def _archives(stub):
    with mock.patch("mousereach.archive.core.archive_video",
                    return_value={"success": True, "files_moved": [],
                                  "destination": None}) as archive:
        ProcessingOrchestrator._archive_to_nas(stub, {"id": STEM})
    return archive.called


def test_archive_ignores_a_husk_from_an_earlier_round(tmp_path, monkeypatch):
    stub = _husk_and_run(tmp_path, monkeypatch)
    assert _archives(stub)
    stub.db.force_state.assert_not_called()
    assert [c.args[1] for c in stub.db.update_state.call_args_list] == ["archiving", "archived"]


def test_archive_ignores_a_husk_diverted_after_the_run(tmp_path, monkeypatch):
    # 20250808_CNT0113_P2, 2026-09-29: the husk (a lone .mp4 a return could
    # not delete) was diverted to a queue AFTER the clean re-run, so its
    # routing record is newer than the run -- but it moved only the video.
    stub = _husk_and_run(tmp_path, monkeypatch, routing={
        "after_run_s": 180, "moved_files": [f"{STEM}.mp4"], "failed_files": []})
    assert _archives(stub)
    stub.db.force_state.assert_not_called()


def test_archive_adopts_the_folder_when_this_runs_results_left(tmp_path, monkeypatch):
    # What the bench-disagreement router does: the run's results, manifest
    # included, are moved into the queue while the row still says 'processed'.
    stub = _husk_and_run(tmp_path, monkeypatch)
    (stub.processing_dir / f"{STEM}_processing_manifest.json").unlink()
    assert not _archives(stub)
    assert stub.db.force_state.call_args[0][:2] == (STEM, "triage")


def test_archive_adopts_a_route_of_this_run_that_left_the_manifest(tmp_path, monkeypatch):
    stub = _husk_and_run(tmp_path, monkeypatch, routing={
        "after_run_s": 60, "moved_files": [f"{STEM}_segments.json"],
        "failed_files": [f"{STEM}_processing_manifest.json"]})
    assert not _archives(stub)
    assert stub.db.force_state.call_args[0][:2] == (STEM, "triage")


def test_an_old_route_that_left_a_manifest_is_not_this_run(tmp_path, monkeypatch):
    stub = _husk_and_run(tmp_path, monkeypatch, routing={
        "after_run_s": -600, "moved_files": [],
        "failed_files": [f"{STEM}_processing_manifest.json"]})
    assert _archives(stub)


def test_an_unreadable_routing_record_keeps_the_cautious_behaviour(tmp_path, monkeypatch):
    stub = _husk_and_run(tmp_path, monkeypatch)
    (tmp_path / "triage" / STEM / f"{STEM}_routing.json").write_text("{not json", encoding="utf-8")
    assert not _archives(stub)
    assert stub.db.force_state.call_args[0][:2] == (STEM, "triage")


def test_a_husk_does_not_hide_a_route_still_being_built(tmp_path, monkeypatch):
    stub = _husk_and_run(tmp_path, monkeypatch)
    (tmp_path / "triage" / ".incoming" / STEM).mkdir(parents=True)
    assert not _archives(stub)
    stub.db.force_state.assert_not_called()


def test_failed_manifest_write_sets_the_old_manifest_aside(tmp_path):
    old = tmp_path / f"{STEM}_processing_manifest.json"
    old.write_text(json.dumps({"from": "previous run"}), encoding="utf-8")

    BaseOrchestrator._set_aside_stale_manifest(object(), tmp_path, STEM, RuntimeError("disk full"))

    assert not old.exists()
    aside = list((tmp_path / "_stale_manifests").glob(f"{STEM}_processing_manifest.*.json"))
    assert len(aside) == 1
    assert json.loads(aside[0].read_text(encoding="utf-8")) == {"from": "previous run"}


def test_failed_manifest_write_with_no_old_manifest_changes_nothing(tmp_path):
    BaseOrchestrator._set_aside_stale_manifest(object(), tmp_path, STEM, RuntimeError("x"))
    assert list(tmp_path.iterdir()) == []


def test_review_manifest_names_the_declared_pose_not_the_first_sorted(tmp_path, monkeypatch):
    from mousereach.pipeline import manifest as manifest_mod
    from mousereach.watcher.review_gate import _write_review_manifest

    bundle = tmp_path / STEM
    bundle.mkdir()
    (bundle / f"{STEM}.mp4").write_bytes(b"")
    first_sorted = bundle / f"{STEM}DLC_resnet101_MPSAOct27shuffle3_100000.h5"
    declared = bundle / f"{STEM}DLC_resnet50_MPSAOct27shuffle1_100000.h5"
    for p in (first_sorted, declared):
        p.write_bytes(b"pose")
    assert sorted(bundle.glob(f"{STEM}DLC*.h5"))[0] == first_sorted

    scorer = manifest_mod.extract_dlc_model_info(declared)["dlc_scorer"]
    monkeypatch.setattr(manifest_mod, "declared_dlc_scorer", lambda: scorer)

    _write_review_manifest(bundle, STEM, "test")

    written = json.loads((bundle / f"{STEM}_manifest.json").read_text(encoding="utf-8"))
    assert written["canonical_dlc_h5_path"] == str(declared)
    assert written["provenance"]["self_contained"] is True
