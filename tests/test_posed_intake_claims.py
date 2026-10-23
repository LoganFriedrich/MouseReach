"""Posed-video intake claims: exclusive, held while the holder runs, let go
when the video has left the posed folder.

WHY (2026-10-22): the processing server claimed a posed video by writing a
marker, waiting half a second and reading it back -- two nodes could both
believe they won -- and deleted every marker after 24 hours whatever its holder
was doing. A video held for a day in review could then be taken in by another
node. The claim is now the exclusive create the review queues use
(watcher/claim_marker.py). Each test runs two "nodes" against one shared folder
built under tmp_path.
"""
import os
import time
from unittest import mock

import pytest

from mousereach.config import Paths
from mousereach.watcher import claim_marker
from mousereach.watcher.orchestrator import ProcessingOrchestrator

V1, V2, V3 = "20250101_CNT0101_P1", "20250101_CNT0102_P1", "20250101_CNT0103_P1"


@pytest.fixture
def share(tmp_path, monkeypatch):
    nas = tmp_path / "nas"
    posed = nas / "Processing" / "Posed"
    status = nas / "Processing" / ".node_status"
    posed.mkdir(parents=True)
    status.mkdir(parents=True)
    monkeypatch.setattr(Paths, "NAS_ROOT", nas)
    return posed, status


def _node(name, posed, tmp_path, states=None):
    """A processing node; ``states`` is its own record, {video_id: state}."""
    node = object.__new__(ProcessingOrchestrator)
    node.hostname = name
    node.staging_dir = posed
    node.processing_dir = tmp_path / f"local_{name}"
    node.db = mock.Mock()
    node.states = dict(states or {})
    node.db.states_of.side_effect = lambda ids: {
        v: (node.states[v], "") for v in ids if v in node.states}
    return node


def _report(status, host, age_s=0.0):
    """A status report from ``host``, last rewritten ``age_s`` seconds ago."""
    p = status / f"{host}.json"
    p.write_text("{}", encoding="ascii")
    t = time.time() - age_s
    os.utime(p, (t, t))


def _stage(posed, vid):
    (posed / f"{vid}.mp4").write_bytes(b"v")
    (posed / f"{vid}DLC_model.h5").write_bytes(b"p")


def _age(path, seconds):
    t = time.time() - seconds
    os.utime(path, (t, t))


def test_one_of_two_nodes_wins(share, tmp_path):
    posed, status = share
    a, b = _node("NODE-A", posed, tmp_path), _node("NODE-B", posed, tmp_path)
    _report(status, "NODE-A"), _report(status, "NODE-B")
    assert a._claim_video(V1)
    assert not b._claim_video(V1)
    assert a._claim_video(V1)              # the holder is never locked out
    assert claim_marker.holder(posed / ".claims", V1, ".claimed") == "NODE-A"


def test_a_running_holders_claim_is_never_broken_for_its_age(share, tmp_path):
    posed, status = share
    a, b = _node("NODE-A", posed, tmp_path), _node("NODE-B", posed, tmp_path)
    _report(status, "NODE-A"), _report(status, "NODE-B")
    _stage(posed, V1)
    assert a._claim_video(V1)
    _age(posed / ".claims" / f"{V1}.claimed", 3 * 86400)   # held three days
    assert not b._claim_video(V1)
    b._sweep_posed_claims()
    assert claim_marker.holder(posed / ".claims", V1, ".claimed") == "NODE-A"


def test_a_stopped_holders_claim_is_broken(share, tmp_path):
    posed, status = share
    a, b = _node("NODE-A", posed, tmp_path), _node("NODE-B", posed, tmp_path)
    assert a._claim_video(V1)
    _report(status, "NODE-A", age_s=claim_marker.GONE_AFTER_S + 600)
    _report(status, "NODE-B")
    assert b._claim_video(V1)
    assert claim_marker.holder(posed / ".claims", V1, ".claimed") == "NODE-B"


def test_no_claim_folder_means_no_intake(share, tmp_path):
    posed, _ = share
    a = _node("NODE-A", posed, tmp_path)
    (posed / ".claims").write_text("not a folder")      # the folder cannot be made
    assert not a._claim_video(V1)


def test_a_declined_intake_is_no_work_and_the_video_waits_its_turn(share, tmp_path):
    posed, status = share
    a, b = _node("NODE-A", posed, tmp_path), _node("NODE-B", posed, tmp_path)
    _report(status, "NODE-A"), _report(status, "NODE-B")
    _stage(posed, V1)
    assert a._claim_video(V1)

    assert b._intake_from_staging({"id": V1, "data": {}}) is False
    b.db.log_step.assert_not_called()
    assert b._held_elsewhere_active(V1)
    assert not b._held_elsewhere_active(V2)
    b._held_elsewhere[V1] = time.monotonic() - 1        # recheck time reached
    assert not b._held_elsewhere_active(V1)


def test_own_claims_are_let_go_once_the_video_has_left_the_posed_folder(share, tmp_path):
    posed, status = share
    a = _node("NODE-A", posed, tmp_path, states={V1: "triage"})
    b = _node("NODE-B", posed, tmp_path)
    _report(status, "NODE-A"), _report(status, "NODE-B")
    _stage(posed, V1)                       # V1 still there; V2 and V3 gone
    for vid in (V1, V2):
        assert a._claim_video(vid)
    assert b._claim_video(V3)

    assert a._sweep_posed_claims() == (0, 1)
    claims = posed / ".claims"
    assert claim_marker.holder(claims, V1, ".claimed") == "NODE-A"
    assert claim_marker.holder(claims, V2, ".claimed") is None
    assert claim_marker.holder(claims, V3, ".claimed") == "NODE-B"   # never another's

    assert a._claim_video(V2)
    assert a._sweep_posed_claims() == (0, 0)  # swept at most every few minutes
    assert claim_marker.holder(claims, V2, ".claimed") == "NODE-A"


def test_archiving_lets_go_only_after_the_posed_copy_is_removed(share, tmp_path, monkeypatch):
    posed, status = share
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    a = _node("NODE-A", posed, tmp_path)
    _report(status, "NODE-A")
    _stage(posed, V1)
    assert a._claim_video(V1)

    seen = []
    real_release = a._release_claim

    def release(vid):
        seen.append(sorted(p.name for p in posed.iterdir() if p.is_file()))
        real_release(vid)
    a._release_claim = release

    ok = {"success": True, "files_moved": [], "destination": "x"}
    with mock.patch("mousereach.archive.core.archive_video", return_value=ok):
        assert ProcessingOrchestrator._archive_to_nas(a, {"id": V1}) is True
    assert seen == [[]]                       # nothing of V1 left when released
    assert claim_marker.holder(posed / ".claims", V1, ".claimed") is None


def test_a_posed_file_that_cannot_be_removed_keeps_the_claim(share, tmp_path, monkeypatch):
    posed, status = share
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    a = _node("NODE-A", posed, tmp_path)
    _report(status, "NODE-A")
    _stage(posed, V1)
    assert a._claim_video(V1)

    real_unlink = type(posed).unlink

    def stuck(self, *args, **kw):
        if self.name.endswith(".mp4"):
            raise PermissionError("in use")
        return real_unlink(self, *args, **kw)
    monkeypatch.setattr(type(posed), "unlink", stuck)

    ok = {"success": True, "files_moved": [], "destination": "x"}
    with mock.patch("mousereach.archive.core.archive_video", return_value=ok):
        assert ProcessingOrchestrator._archive_to_nas(a, {"id": V1}) is True
    assert claim_marker.holder(posed / ".claims", V1, ".claimed") == "NODE-A"


def test_videos_in_hand_without_a_claim_are_claimed(share, tmp_path):
    # Before 2026-10-22 every marker was deleted after 24 h, so posed copies of
    # videos in review had none and looked like waiting work.
    posed, status = share
    a = _node("NODE-A", posed, tmp_path,
              states={V1: "triage", V2: "dlc_complete", V3: "archived"})
    b = _node("NODE-B", posed, tmp_path)
    _report(status, "NODE-A"), _report(status, "NODE-B")
    for vid in (V1, V2, V3):
        _stage(posed, vid)
    assert a._sweep_posed_claims() == (2, 0)
    claims = posed / ".claims"
    assert claim_marker.holder(claims, V1, ".claimed") == "NODE-A"
    assert claim_marker.holder(claims, V2, ".claimed") is None     # waiting work
    assert claim_marker.holder(claims, V3, ".claimed") == "NODE-A"
    assert not b._claim_video(V1)
    assert b._claim_video(V2)


def test_a_video_waiting_again_is_let_go(share, tmp_path):
    # A lifted review hold goes back to the posed folder as 'dlc_complete' for
    # a fresh run (review_return._release_to_posed): any node may take it.
    posed, status = share
    a = _node("NODE-A", posed, tmp_path, states={V1: "deep_review"})
    _report(status, "NODE-A")
    _stage(posed, V1)
    assert a._claim_video(V1)
    assert a._sweep_posed_claims() == (0, 0)
    a.states[V1] = "dlc_complete"
    a._posed_claim_swept -= a._POSED_CLAIM_SWEEP_S + 1
    assert a._sweep_posed_claims() == (0, 1)
    assert claim_marker.holder(posed / ".claims", V1, ".claimed") is None


def test_another_nodes_claim_on_a_video_in_hand_is_left_alone(share, tmp_path):
    posed, status = share
    a = _node("NODE-A", posed, tmp_path, states={V1: "triage"})
    b = _node("NODE-B", posed, tmp_path)
    _report(status, "NODE-A"), _report(status, "NODE-B")
    _stage(posed, V1)
    assert b._claim_video(V1)
    assert a._sweep_posed_claims() == (0, 0)
    assert claim_marker.holder(posed / ".claims", V1, ".claimed") == "NODE-B"
