"""Startup recovery never adopts a video another machine has claimed for pose.

WHY: a claimed single sits in <Single_Animal>/.inflight/<holder>/ on the shared
drive, and the shared record's source_path points at it -- so the file EXISTS
for every node. Recovery used to take "the recorded file exists" as "the file is
here", adopt the holder's dlc_running row, and the startup reclaim then put it
back to dlc_queued, so a restarting GPU node posed the video a second time
(2026-10-28: each GPU restart took the 12 videos the server's CPU pose pool held).
"""
from pathlib import Path

import pytest

from mousereach.watcher import coordination, single_claim

from test_coordination_recovery_pathless import FakeLocalDB

VID = "20240101_ABC0101_P1"


@pytest.fixture
def claims(tmp_path, monkeypatch):
    root = tmp_path / "Single_Animal" / ".inflight"
    monkeypatch.setattr(single_claim, "inflight_root", lambda: root)
    return root


def _coordinator(monkeypatch, path, holder):
    c = coordination.PipelineCoordinator.__new__(coordination.PipelineCoordinator)
    monkeypatch.setattr(c, "get_all_video_states", lambda: {
        VID: {"state": "dlc_running", "hostname": holder,
              "source_path": str(path), "collage_id": None}}, raising=False)
    monkeypatch.setattr(c, "get_all_collage_states", lambda: {}, raising=False)
    monkeypatch.setattr(coordination, "locate_video_file", lambda *a, **k: Path(path))
    return c


def test_new_row_for_another_hosts_claim_is_recorded_elsewhere(claims, monkeypatch):
    path = claims / "server-cpu" / f"{VID}.mp4"
    local = FakeLocalDB({})
    stats = _coordinator(monkeypatch, path, "server-cpu").recover_local_db(local, "gpu-node")
    assert local.forced == [], "another machine's claim must not be adopted"
    assert [v for v, _ in local.unresolvable] == [VID]
    assert stats["videos_elsewhere"] == 1


def test_existing_queued_row_is_not_advanced_into_another_hosts_claim(claims, monkeypatch):
    """The rows a previous restart already adopted heal at the next restart."""
    path = claims / "server-cpu" / f"{VID}.mp4"
    local = FakeLocalDB({VID: {"state": "dlc_queued", "current_path": str(path)}})
    _coordinator(monkeypatch, path, "server-cpu").recover_local_db(local, "gpu-node")
    assert local.forced == []
    assert [v for v, _ in local.unresolvable] == [VID]


def test_this_nodes_own_claim_is_still_its_own(claims, monkeypatch):
    path = claims / "gpu-node" / f"{VID}.mp4"
    local = FakeLocalDB({})
    _coordinator(monkeypatch, path, "gpu-node").recover_local_db(local, "gpu-node")
    assert local.forced == [(VID, "dlc_running")]
    assert local.unresolvable == []


def test_a_file_outside_any_claim_folder_is_unchanged(claims, monkeypatch, tmp_path):
    path = tmp_path / "DLC_Queue" / f"{VID}.mp4"
    local = FakeLocalDB({})
    _coordinator(monkeypatch, path, "other-node").recover_local_db(local, "gpu-node")
    assert local.forced == [(VID, "dlc_running")]
