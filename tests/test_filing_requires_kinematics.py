"""A video is filed only with its kinematics.

Filing means done for good: what is in the filed folder is what the database
pulls. A video whose feature extraction failed -- or never ran because the
reaches or outcomes file it needs was missing -- used to go on to 'processed'
and be filed with no ``<stem>_features.json``, so it reached the filed folder
with no kinematics at all. Now it is marked failed with a reason a person
reads (mousereach-watch-status), stays in Processing, and the failed-video
retry runs it again; the archive step itself refuses a video whose features
file is missing. Built under tmp_path; no real folder or database.
"""
from pathlib import Path
from unittest import mock

import pytest

from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB
from mousereach.watcher import orchestrator as orch_mod
from mousereach.watcher.orchestrator import (
    DLCOrchestrator, ProcessingOrchestrator, NO_KINEMATICS_REASON,
)

PILLAR = "20990101_ABC9901_P1"
EASY = "20990101_ABC9902_E1"


def _row(db, vid, upto):
    db.register_video(video_id=vid, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete",
               "processing", "processed"):
        db.update_state(vid, st)
        if st == upto:
            return


@pytest.fixture
def server(tmp_path, monkeypatch):
    proc = tmp_path / "Processing"
    proc.mkdir()
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    monkeypatch.setattr(Paths, "PROCESSING", proc)
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.db.export_to_central_db = lambda vid: None
    o.processing_dir = proc
    o.staging_dir = None
    o.hostname = "NODE-T"
    o._release_claim = lambda vid: None
    return o


def _outputs(proc, vid, *, features=False, reaches=True, outcomes=True):
    (proc / f"{vid}.mp4").write_bytes(b"video")
    if reaches:
        (proc / f"{vid}_reaches.json").write_text("{}")
    if outcomes:
        (proc / f"{vid}_pellet_outcomes.json").write_text("{}")
    if features:
        (proc / f"{vid}_features.json").write_text("{}")


class _Boom:
    VERSION = "test"

    def extract(self, *a, **k):
        raise RuntimeError("extractor crashed")


def test_a_failed_extraction_is_held_as_failed_not_processed(server):
    _row(server.db, PILLAR, "processing")
    _outputs(server.processing_dir, PILLAR)
    with mock.patch("mousereach.kinematics.core.feature_extractor.FeatureExtractor", _Boom):
        server._extract_features_and_finish(PILLAR, Path("pose.h5"), False, 1.0)
    row = server.db.get_video(PILLAR)
    assert row["state"] == "failed"
    assert row["error_message"].startswith(NO_KINEMATICS_REASON)
    assert "extractor crashed" in row["error_message"]


def test_a_pillar_video_missing_its_outcomes_is_not_filed(server):
    _row(server.db, PILLAR, "processing")
    _outputs(server.processing_dir, PILLAR, outcomes=False)
    server._extract_features_and_finish(PILLAR, Path("pose.h5"), False, 1.0)
    row = server.db.get_video(PILLAR)
    assert row["state"] == "failed"
    assert "_pellet_outcomes.json" in row["error_message"]


def test_an_easy_video_without_its_session_file_is_not_filed(server):
    _row(server.db, EASY, "processing")
    _outputs(server.processing_dir, EASY, reaches=False, outcomes=False)
    server._extract_features_and_finish(EASY, Path("pose.h5"), True, 1.0)
    assert server.db.get_video(EASY)["state"] == "failed"


def test_a_written_features_file_goes_on_to_processed(server):
    _row(server.db, PILLAR, "processing")
    _outputs(server.processing_dir, PILLAR)

    class _Ok:
        VERSION = "test"

        def extract(self, *a, **k):
            return mock.Mock(n_segments=1, to_dict=lambda: {"ok": True})

    with mock.patch("mousereach.kinematics.core.feature_extractor.FeatureExtractor", _Ok), \
            mock.patch("mousereach.pipeline.manifest.record_kinematic_version"):
        server._extract_features_and_finish(PILLAR, Path("pose.h5"), False, 1.0)
    assert server.db.get_video(PILLAR)["state"] == "processed"
    assert (server.processing_dir / f"{PILLAR}_features.json").is_file()


def test_the_archive_step_refuses_a_video_without_features(server):
    _row(server.db, PILLAR, "processed")
    _outputs(server.processing_dir, PILLAR)            # no features file
    with mock.patch("mousereach.archive.core.archive_video") as archive:
        assert ProcessingOrchestrator._archive_to_nas(server, {"id": PILLAR}) is False
    archive.assert_not_called()
    row = server.db.get_video(PILLAR)
    assert row["state"] == "failed"
    assert row["error_message"].startswith(NO_KINEMATICS_REASON)
    assert (server.processing_dir / f"{PILLAR}.mp4").is_file()    # nothing moved


def test_the_local_archive_step_refuses_too(tmp_path, monkeypatch):
    proc = tmp_path / "Processing"
    proc.mkdir()
    monkeypatch.setattr(Paths, "PROCESSING", proc)
    monkeypatch.setattr(Paths, "DLC_QUEUE", tmp_path / "DLC_Queue")
    o = DLCOrchestrator.__new__(DLCOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.hostname = "NODE-T"
    _row(o.db, PILLAR, "processed")
    _outputs(proc, PILLAR)
    with mock.patch("mousereach.archive.core.archive_video") as archive:
        assert o._archive_locally_processed({"id": PILLAR}) is False
    archive.assert_not_called()
    assert o.db.get_video(PILLAR)["state"] == "failed"


def test_the_local_pipeline_uses_the_same_kinematics_step():
    """One implementation for both roles, so the rule cannot drift apart."""
    import inspect
    src = inspect.getsource(DLCOrchestrator._run_local_pipeline)
    assert "_make_features(" in src and "_hold_without_kinematics(" in src
    assert "FeatureExtractor" not in src
