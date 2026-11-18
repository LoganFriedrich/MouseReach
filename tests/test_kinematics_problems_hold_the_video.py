"""Kinematics that cannot be the video's true result are not written, so the
video is held instead of filed.

WHY: the feature extractor used to PRINT a line and carry on when a person's
answers could not be applied (truth layering failed, a review copy or the
ground-truth file could not be read), when the reach assignment file could not
be read, or when a reach's extended measures failed. The features file was
then written -- with the algorithm's answers in place of the person's, or with
a reach whose extended values were all missing -- and the video was filed.
Now extract() raises (feature_extractor.KinematicsProblem), no features file is
written, and the watcher holds the video as failed with the reason. The
archive step also refuses a features file that carries a failed reach.

Built under tmp_path with invented ids; no real folder, database or
configuration.
"""
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import mousereach.review.causal_review_io as crio
from mousereach.config import Paths
from mousereach.kinematics.core.feature_extractor import FeatureExtractor
from mousereach.watcher.db import WatcherDB
from mousereach.watcher import orchestrator as orch_mod
from mousereach.watcher.orchestrator import ProcessingOrchestrator

VID = "20990101_ABC0903_P1"
POINTS = ("Nose", "BOXL", "BOXR", "SABL", "SABR", "Pellet", "Pillar",
          "RightHand", "RHLeft", "RHOut", "RHRight")


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A Processing folder with one segment and one detected reach."""
    monkeypatch.setattr(crio, "_GT_INDEX_CACHE", {})
    monkeypatch.setattr(crio, "review_dirs", lambda *a, **k: [], raising=False)
    monkeypatch.setattr(crio, "durable_review_dir", lambda: None, raising=False)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    proc = tmp_path / "Processing"
    proc.mkdir()
    monkeypatch.setattr(Paths, "PROCESSING", proc)
    n = 100
    cols, data = [], []
    for p in POINTS:
        for c, v in (("x", 100.0), ("y", 150.0), ("likelihood", 1.0)):
            cols.append(("scorer", p, c))
            data.append(np.full(n, v) + (np.arange(n) * 0.5 if c == "y" and p.startswith("R") else 0))
    pose = proc / f"{VID}DLC_x.h5"
    pd.DataFrame(np.array(data).T, columns=pd.MultiIndex.from_tuples(cols)).to_hdf(pose, key="df")
    (proc / f"{VID}_reaches.json").write_text(json.dumps(
        {"video_name": VID, "total_frames": n, "n_segments": 1,
         "segments": [{"segment_num": 1, "start_frame": 0, "end_frame": 99,
                       "ruler_pixels": 34.0, "n_reaches": 1,
                       "reaches": [{"reach_id": 1, "reach_num": 1, "start_frame": 20,
                                    "end_frame": 30, "apex_frame": 25,
                                    "duration_frames": 10}]}]}), encoding="utf-8")
    (proc / f"{VID}_pellet_outcomes.json").write_text(json.dumps(
        {"video_name": VID, "segments": [{"segment_num": 1, "outcome": "untouched",
                                          "start_frame": 0, "end_frame": 99}]}),
        encoding="utf-8")
    return proc, pose


@pytest.fixture
def server(run_dir, tmp_path):
    proc, _ = run_dir
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.db.export_to_central_db = lambda vid: None
    o.processing_dir = proc
    o.staging_dir = None
    o.hostname = "NODE-T"
    o._release_claim = lambda vid: None
    return o


def _extract(proc, pose):
    return FeatureExtractor().extract(pose, proc / f"{VID}_reaches.json",
                                      proc / f"{VID}_pellet_outcomes.json")


def test_a_clean_run_still_writes_features(run_dir):
    proc, pose = run_dir
    assert _extract(proc, pose).n_segments == 1


def test_an_unreadable_ground_truth_stops_the_kinematics(run_dir):
    proc, pose = run_dir
    (proc / f"{VID}_unified_ground_truth.json").write_text("{torn", encoding="utf-8")
    with pytest.raises(Exception) as e:
        _extract(proc, pose)
    assert "ground-truth" in str(e.value)


def test_an_unreadable_review_copy_stops_the_kinematics(run_dir, monkeypatch):
    proc, pose = run_dir
    monkeypatch.setattr(crio, "review_dirs", lambda *a, **k: [proc])
    (proc / f"{VID}_causal_review.json").write_text("{torn", encoding="utf-8")
    with pytest.raises(Exception) as e:
        _extract(proc, pose)
    assert "review copy" in str(e.value)


def test_a_failed_truth_layering_stops_the_kinematics(run_dir, monkeypatch):
    proc, pose = run_dir
    import mousereach.review.truth_resolver as tr

    def _boom(*a, **k):
        raise RuntimeError("layering broke")
    monkeypatch.setattr(tr, "resolve_truth_layers", _boom)
    with pytest.raises(Exception) as e:
        _extract(proc, pose)
    assert "layering broke" in str(e.value)


def test_an_unreadable_assignment_file_stops_the_kinematics(run_dir):
    proc, pose = run_dir
    (proc / f"{VID}_reach_assignments.json").write_text("{torn", encoding="utf-8")
    with pytest.raises(Exception) as e:
        _extract(proc, pose)
    assert "assignment" in str(e.value)


def test_a_reach_whose_measures_failed_stops_the_kinematics(run_dir, monkeypatch):
    proc, pose = run_dir

    def _boom(self, *a, **k):
        raise ValueError("measure broke")
    monkeypatch.setattr(FeatureExtractor, "_extended_features", _boom)
    with pytest.raises(Exception) as e:
        _extract(proc, pose)
    assert "measure broke" in str(e.value) and "reach 1" in str(e.value)


def test_the_kinematics_step_writes_no_features_and_gives_the_reason(server, run_dir):
    proc, pose = run_dir
    (proc / f"{VID}_unified_ground_truth.json").write_text("{torn", encoding="utf-8")
    why = server._make_features(VID, proc, pose, False)
    assert why is not None and "ground-truth" in why
    assert not (proc / f"{VID}_features.json").exists()


def test_the_review_is_stamped_applied_only_when_the_extractor_confirms_it(server, run_dir):
    proc, pose = run_dir
    review = proc / f"{VID}_causal_review.json"
    review.write_text(json.dumps({"video_name": VID, "segments": []}), encoding="utf-8")

    class _Unconfirmed:
        VERSION = "test"
        applied_ground_truth = None
        review_applied = False

        def extract(self, *a, **k):
            return mock.Mock(n_segments=1, to_dict=lambda: {"ok": True})

    with mock.patch("mousereach.kinematics.core.feature_extractor.FeatureExtractor",
                    _Unconfirmed), \
            mock.patch("mousereach.review.causal_review_io.resolve_review_path",
                       lambda vid, d: review), \
            mock.patch("mousereach.pipeline.manifest.record_kinematic_version") as rec:
        assert server._make_features(VID, proc, pose, False) is None
    assert rec.call_args.kwargs["review_path"] is None


def test_the_archive_step_refuses_a_features_file_with_a_failed_reach(server, run_dir):
    proc, _ = run_dir
    server.db.register_video(video_id=VID, source_path="x")
    (proc / f"{VID}.mp4").write_bytes(b"video")
    (proc / f"{VID}_features.json").write_text(json.dumps({
        "video_name": VID, "segments": [{"segment_num": 1, "outcome": "untouched",
                                         "reaches": [{"reach_id": 4, "extended": {
                                             "_extended_features_error": "x"}}]}]}),
        encoding="utf-8")
    assert orch_mod.refuse_filing_without_features(server.db, VID, proc) is True
    row = server.db.get_video(VID)
    assert row["state"] == "failed"
    assert "reach 4" in row["error_message"]
    assert (proc / f"{VID}.mp4").is_file()


def test_the_archive_step_passes_a_complete_features_file(server, run_dir):
    proc, _ = run_dir
    server.db.register_video(video_id=VID, source_path="x")
    (proc / f"{VID}_features.json").write_text(json.dumps({
        "video_name": VID, "segments": [{"segment_num": 1, "outcome": "untouched",
                                         "reaches": [{"reach_id": 4, "extended": {"a": 1.0}}]}]}),
        encoding="utf-8")
    assert orch_mod.refuse_filing_without_features(server.db, VID, proc) is False
