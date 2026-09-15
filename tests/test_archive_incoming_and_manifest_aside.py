"""Three small guards around routing into review queues.

  * The archive step waits while a bundle is still being built in
    <queue>/.incoming/<stem>/ (the queue check only sees the published folder).
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
