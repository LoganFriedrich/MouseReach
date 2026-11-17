"""A video is not filed without the provenance manifest of its run.

WHY: a manifest that could not be written was set aside and the video filed
without one; the version scan then counts it as "no manifest" and never checks
it. And when the kinematic version could not be stamped onto the manifest the
filed manifest said kinematics never ran, which the scan reads as unrecorded
and re-runs the video from segmentation with every check.
"""
from pathlib import Path
from unittest import mock

import pytest

from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import (ProcessingOrchestrator,
                                             refuse_filing_without_features)

V = "20990101_ZZZ0101_P7"


def _row(db, upto):
    db.register_video(video_id=V, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete",
               "processing", "processed"):
        db.update_state(V, st)
        if st == upto:
            return


@pytest.fixture
def server(tmp_path, monkeypatch):
    proc = tmp_path / "Processing"
    proc.mkdir()
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    monkeypatch.setattr(Paths, "PROCESSING", proc)
    filed = tmp_path / "Analyzed" / "ZZZ01"
    filed.mkdir(parents=True)
    import mousereach.archive.core as core
    monkeypatch.setattr(core, "get_archive_destination", lambda v: filed)
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.processing_dir = proc
    o.staging_dir = None
    o.filed = filed
    for s in ("_reaches.json", "_pellet_outcomes.json"):
        (proc / f"{V}{s}").write_text("{}")
    (proc / f"{V}.mp4").write_bytes(b"v")
    return o


class _Ok:
    VERSION = "test"

    def extract(self, *a, **k):
        return mock.Mock(n_segments=1, to_dict=lambda: {"ok": True})


def test_no_manifest_of_this_run_means_no_kinematics_and_no_filing(server):
    _row(server.db, "processing")
    with mock.patch("mousereach.kinematics.core.feature_extractor.FeatureExtractor", _Ok):
        why = server._make_features(V, server.processing_dir, Path("pose.h5"), False)
    assert why and "manifest" in why


def test_a_stamp_that_could_not_be_written_holds_the_video(server):
    _row(server.db, "processing")
    (server.processing_dir / f"{V}_processing_manifest.json").write_text("{}")
    with mock.patch("mousereach.kinematics.core.feature_extractor.FeatureExtractor", _Ok), \
            mock.patch("mousereach.pipeline.manifest.record_kinematic_version",
                       return_value=False):
        why = server._make_features(V, server.processing_dir, Path("pose.h5"), False)
    assert why and "stamped" in why


def test_the_filing_step_refuses_a_video_without_its_manifest(server):
    _row(server.db, "processed")
    (server.processing_dir / f"{V}_features.json").write_text("{}")
    assert refuse_filing_without_features(server.db, V, server.processing_dir) is True
    assert server.db.get_video(V)["state"] == "failed"


def test_a_manifest_already_filed_by_an_earlier_attempt_counts(server):
    _row(server.db, "processed")
    (server.filed / f"{V}_processing_manifest.json").write_text("{}")
    (server.processing_dir / f"{V}_features.json").write_text("{}")
    assert refuse_filing_without_features(server.db, V, server.processing_dir) is False
