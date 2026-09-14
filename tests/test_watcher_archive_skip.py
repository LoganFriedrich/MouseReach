"""Watcher lookups over Analyzed never read Analyzed/Archive/ as live.

WHY: superseded outputs are filed under Analyzed/Archive/ with their ORIGINAL
names ({stem}_reaches.json, {stem}_pellet_outcomes.json, {stem}DLC_...h5,
{stem}.mp4). Every lookup here used rglob / a depth-2 glob over Analyzed, so an
archived copy was indistinguishable from the live one -- and each consequence
was silent:

  * reprocess (orchestrator._reprocess_video) re-ran a video on an old
    generation's pose (same scorer, newer mtime wins in select_pose_file);
  * the review return path (review_return._resolve_inputs) re-ran a cleared
    video on an archived mp4 / pose;
  * mousereach-route-to-queue (route_cli) wrote review flags INTO the archived
    pellet_outcomes.json and moved an archive folder into a review queue;
  * the dashboard backfill (backfill.backfill_archive) registered superseded
    videos and pointed their source path into the archive.

Each test builds a tmp Analyzed tree with a live copy and/or a same-named decoy
under Analyzed/Archive/, calls the REAL function, and asserts the decoy is
ignored while live data (including a live pose under DLC Model 4/) is found.

How each test fails on the old code is stated next to it. Decoy-only tests
fail regardless of directory listing order; live-plus-decoy tests either rely
on mtime (select_pose_file picks the newest) or on NTFS listing order
("Archive" sorts before the project folder "X").
"""
import json
import os
import time
from pathlib import Path

import pytest

import mousereach.archive.core as archive_core
import mousereach.pipeline.manifest as manifest
import mousereach.watcher.orchestrator as orch
import mousereach.watcher.repose as repose
import mousereach.watcher.review_return as rr
from mousereach.config import Paths
from mousereach.watcher.backfill import backfill_archive
from mousereach.watcher.orchestrator import ProcessingOrchestrator
from mousereach.watcher.route_cli import _find_outcomes, route_video

VID = "20240101_ABC0101_P1"
VID2 = "20240102_ABC0102_P1"
NEW = "DLC_resnet101_MPSAOct27shuffle3_100000"

SUPERSEDED = "superseded_processing_root_3.1"


def _write(path: Path, data: bytes, age_s: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if age_s:
        t = time.time() - age_s
        os.utime(path, (t, t))
    return path


@pytest.fixture
def analyzed(tmp_path, monkeypatch):
    root = tmp_path / "Analyzed"
    root.mkdir()
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", root)
    # Never let a lookup wander to the real staging area or version file.
    monkeypatch.setattr(Paths, "DLC_STAGING", None, raising=False)
    monkeypatch.setattr(manifest, "declared_dlc_scorer",
                        lambda max_age_s=None: NEW)
    return root


# ------------------------------------------------------ orchestrator reprocess

class FakeDB:
    def __init__(self):
        self.failed = []
        self.forced = []

    def mark_failed(self, video_id, msg):
        self.failed.append((video_id, msg))

    def force_state(self, video_id, state, **kw):
        self.forced.append((video_id, state, kw))


@pytest.fixture
def server(tmp_path, analyzed, monkeypatch):
    results = analyzed / "X" / "C01"
    results.mkdir(parents=True)
    _write(results / f"{VID}.mp4", b"live-video")
    monkeypatch.setattr(repose, "declared_scorer", lambda: NEW)
    monkeypatch.setattr(archive_core, "get_archive_destination",
                        lambda vid: results)
    o = object.__new__(ProcessingOrchestrator)
    o.db = FakeDB()
    o.staging_dir = None
    o.processing_dir = tmp_path / "local" / "Processing"
    o.ran = []
    o._run_pipeline = lambda work: o.ran.append(work)
    return o


def test_reprocess_uses_the_live_pose_not_a_newer_archived_one(server, analyzed):
    live = _write(analyzed / "X" / "DLC Model 4" / "C01" / f"{VID}{NEW}.h5",
                  b"live-pose", age_s=3600)
    # Same name, same scorer, NEWER: the old rglob found both and
    # select_pose_file (newest mtime among the declared scorer) chose this one.
    _write(analyzed / "Archive" / "DLC Model 4.0" / "C01" / f"{VID}{NEW}.h5",
           b"decoy-pose-archived")

    server._reprocess_video({"id": VID, "data": {}})

    assert server.db.failed == []
    assert len(server.ran) == 1
    local = server.processing_dir / f"{VID}{NEW}.h5"
    assert local.read_bytes() == live.read_bytes() == b"live-pose"
    assert server.ran[0]["data"]["dlc_output_path"] == str(local)


def test_reprocess_refuses_a_video_whose_only_pose_is_archived(server, analyzed):
    # Old code: rglob found this pose and re-ran the video on it.
    _write(analyzed / "Archive" / "DLC Model 4.0" / "C01" / f"{VID}{NEW}.h5",
           b"decoy-pose-archived")

    server._reprocess_video({"id": VID, "data": {}})

    assert server.ran == []
    assert server.db.failed and "DLC h5 not found" in server.db.failed[0][1]
    assert not (server.processing_dir / f"{VID}{NEW}.h5").exists()


# ------------------------------------------------------ review return resolver

def test_return_resolver_finds_live_inputs_and_ignores_archived(tmp_path, analyzed):
    bundle = tmp_path / "queue" / VID
    bundle.mkdir(parents=True)
    live_mp4 = _write(analyzed / "X" / "C01" / f"{VID}.mp4", b"live-video")
    live_pose = _write(analyzed / "X" / "DLC Model 4" / "C01" / f"{VID}{NEW}.h5",
                       b"live-pose", age_s=3600)
    # Decoys: the pose is newer (old code: select_pose_file chose it); the mp4
    # sorts first ("Archive" < "X" on NTFS, old code: rglob returned it first).
    _write(analyzed / "Archive" / SUPERSEDED / f"{VID}.mp4", b"decoy-video")
    _write(analyzed / "Archive" / "DLC Model 4.0" / "C01" / f"{VID}{NEW}.h5",
           b"decoy-pose")

    mp4, pose = rr._resolve_inputs(bundle, VID)

    assert mp4 == live_mp4
    assert pose == live_pose


def test_return_resolver_does_not_return_on_archive_only_inputs(tmp_path, analyzed):
    bundle = tmp_path / "queue" / VID
    bundle.mkdir(parents=True)
    # Old code: both found by rglob, so the video would have been returned to
    # Processing on superseded inputs.
    _write(analyzed / "Archive" / SUPERSEDED / f"{VID}.mp4", b"decoy-video")
    _write(analyzed / "Archive" / SUPERSEDED / f"{VID}{NEW}.h5", b"decoy-pose")

    mp4, pose = rr._resolve_inputs(bundle, VID)

    assert mp4 is None
    assert pose is None


# ------------------------------------------------------ route-to-queue CLI

def _outcomes(path: Path) -> Path:
    body = {"segments": [{"segment_num": 3, "flagged_for_review": False}]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.fixture
def queues(tmp_path, monkeypatch):
    triage = tmp_path / "Review" / "Triage"
    deep = tmp_path / "Review" / "Deep"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    return triage


@pytest.mark.parametrize("sub", [
    (SUPERSEDED,),              # depth 2: old code matched the */*/ glob
    (SUPERSEDED, "C01"),        # depth 3: old code matched the rglob fallback
])
def test_route_does_not_touch_an_archived_outcomes_file(analyzed, queues, sub):
    decoy = _outcomes(analyzed.joinpath("Archive", *sub,
                                        f"{VID}_pellet_outcomes.json"))
    before = decoy.read_bytes()

    res = route_video(VID, "triage", "bench disagreement", [3], db=None)

    assert res["routed"] is False
    assert res["error"] and "not found" in res["error"]
    assert res["flagged"] == []
    assert decoy.read_bytes() == before          # no flags written into it
    assert decoy.parent.is_dir()                 # archive folder not moved
    assert list(queues.iterdir()) == []          # nothing entered the queue


def test_find_outcomes_prefers_live_over_archived(analyzed):
    # "Archive" sorts before "X": the old */*/ glob returned the decoy first.
    _outcomes(analyzed / "Archive" / SUPERSEDED / f"{VID}_pellet_outcomes.json")
    live = _outcomes(analyzed / "X" / "C01" / f"{VID}_pellet_outcomes.json")

    assert _find_outcomes(analyzed, VID) == live


def test_find_outcomes_fallback_still_finds_deep_live_file(analyzed):
    _outcomes(analyzed / "Archive" / SUPERSEDED / "C01" / f"{VID}_pellet_outcomes.json")
    live = _outcomes(analyzed / "X" / "Sub" / "C01" / f"{VID}_pellet_outcomes.json")

    assert _find_outcomes(analyzed, VID) == live


# ------------------------------------------------------ dashboard backfill

class BackfillDB:
    def __init__(self):
        self.rows = {}

    def get_video(self, video_id):
        return self.rows.get(video_id)

    def register_video(self, video_id, source, **kw):
        self.rows[video_id] = {"source": source, **kw}

    def _now(self):
        return "now"


def test_backfill_registers_only_live_videos(analyzed):
    live_dir = analyzed / "X" / "C01"
    _write(live_dir / f"{VID}_reaches.json", b"{}")
    _write(live_dir / f"{VID}.mp4", b"live-video")
    arch = analyzed / "Archive" / SUPERSEDED
    # Old code: VID2 (archive only) was registered as a live archived video,
    # and VID's archived copy (listed first) set its source path.
    _write(arch / f"{VID}_reaches.json", b"{}")
    _write(arch / f"{VID}.mp4", b"decoy-video")
    _write(arch / f"{VID2}_reaches.json", b"{}")
    _write(arch / f"{VID2}.mp4", b"decoy-video")
    db = BackfillDB()

    result = backfill_archive(db, analyzed)

    assert result == {"new": 1, "existing": 0, "errors": 0}
    assert set(db.rows) == {VID}
    assert Path(db.rows[VID]["source"]) == live_dir / f"{VID}.mp4"
