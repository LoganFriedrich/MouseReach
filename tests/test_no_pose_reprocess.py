"""A re-run that finds no pose does not fail the video.

WHY (2026-10-22): the re-run step marked a video failed whenever it found no
pose. On 2026-09-01 three videos failed that way because their bundles -- pose
included -- had been moved into triage. A missing pose is a pose to make or a
file to follow: the video is held where its pose is, or marked as needing a new
pose (which the GPU nodes are asked for), and fails only when its video is
missing too.
"""
import mousereach.watcher.orchestrator as o
from mousereach.config import Paths
from mousereach.watcher import repose
from mousereach.watcher.db import WatcherDB

V = "20250101_CNT0101_P1"


def _server(tmp_path, monkeypatch, video=True):
    triage, deep = tmp_path / "triage", tmp_path / "deep_review"
    triage.mkdir()
    deep.mkdir()
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    mp4 = tmp_path / f"{V}.mp4"
    if video:
        mp4.write_bytes(b"v")
    monkeypatch.setattr(repose, "archived_video", lambda vid: mp4 if video else None)
    server = object.__new__(o.ProcessingOrchestrator)
    server.db = WatcherDB(tmp_path / "w.db")
    server.db.register_video(video_id=V, source_path="x")
    server.db.force_state(V, "outdated", reason="test fixture")
    return server, triage, deep


def test_a_pose_in_a_review_bundle_means_held_there(tmp_path, monkeypatch):
    server, triage, deep = _server(tmp_path, monkeypatch)
    (deep / V).mkdir()
    (deep / V / f"{V}DLC_resnet101_X_100000.h5").write_bytes(b"p")
    assert server._no_pose_for_reprocess(V) == "deep_review"
    assert server.db.get_video(V)["state"] == "deep_review"


def test_an_archived_video_without_a_pose_asks_for_a_new_one(tmp_path, monkeypatch):
    server, triage, deep = _server(tmp_path, monkeypatch)
    assert server._no_pose_for_reprocess(V) == "needs_pose"
    row = server.db.get_video(V)
    assert row["state"] == "outdated"                    # parked, not failed
    assert row["reprocess_scope"] == "full"              # what publishes the request


def test_no_pose_and_no_video_is_a_failure_that_says_so(tmp_path, monkeypatch):
    server, triage, deep = _server(tmp_path, monkeypatch, video=False)
    assert server._no_pose_for_reprocess(V) == "failed"
    row = server.db.get_video(V)
    assert row["state"] == "failed"
    assert "no pose and no video" in row["error_message"]


def test_a_bundle_without_a_pose_is_not_taken_for_one(tmp_path, monkeypatch):
    server, triage, deep = _server(tmp_path, monkeypatch)
    (triage / V).mkdir()
    (triage / V / f"{V}.mp4").write_bytes(b"v")
    assert server._no_pose_for_reprocess(V) == "needs_pose"
