"""The re-pose round trip over shared storage (watcher/repose.py).

WHY: a video marked outdated with scope 'full' needs a NEW pose from a GPU,
and until 2026-09-12 nothing carried it to one, nor took the new pose back
when a GPU node staged it (every discovery scan skipped ids that already
had a row). Three videos each cost one hand step on each machine. These
tests pin the three steps -- publish, consume, adopt -- and the invariants
the design critique demanded: a request is claimed by an atomic rename and
outstanding until the round trip closes, consumption is capped and never
touches locked or human-held rows, adoption narrows the reprocess scope
instead of forcing the state machine, and staging never leaves a
half-written file under its final name.
"""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.watcher.repose as repose
import mousereach.watcher.orchestrator as orch
from mousereach.watcher.coordination import _is_live_local_work
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import DLCOrchestrator

VID = "20240101_ABC0101_P1"
VID2 = "20240102_ABC0102_P1"
VID3 = "20240103_ABC0103_P1"
OLD = "DLC_resnet50_MPSAOct27shuffle1_100000"
NEW = "DLC_resnet101_MPSAOct27shuffle3_100000"


@pytest.fixture
def env(tmp_path, monkeypatch):
    nas = tmp_path / "nas"
    queue = nas / "Processing" / "Repose_Queue"
    staging = nas / "Processing" / "DLC_Complete"
    archive = nas / "Analyzed" / "X" / "C01"
    local_q = tmp_path / "local" / "DLC_Queue"
    processing = tmp_path / "local" / "Processing"
    for d in (queue, staging, archive, local_q, processing):
        d.mkdir(parents=True)
    for name in ("NAS_ROOT", "REPOSE_QUEUE", "DLC_STAGING", "DLC_QUEUE",
                 "ANALYZED_OUTPUT", "PROCESSING"):
        monkeypatch.setattr(repose.Paths, name, {
            "NAS_ROOT": nas, "REPOSE_QUEUE": queue, "DLC_STAGING": staging,
            "DLC_QUEUE": local_q, "ANALYZED_OUTPUT": nas / "Analyzed",
            "PROCESSING": processing}[name])
    monkeypatch.setattr(repose, "archive_folder", lambda vid: archive)
    for v in (VID, VID2, VID3):
        (archive / f"{v}.mp4").write_bytes(b"video-bytes-" + v.encode())
        (archive / f"{v}{OLD}.h5").write_bytes(b"old-pose")
    db = WatcherDB(db_path=tmp_path / "w.db")
    return SimpleNamespace(nas=nas, queue=queue, staging=staging, archive=archive,
                           local_q=local_q, processing=processing, db=db)


def _row(db, vid, state, **fields):
    db.register_video(video_id=vid, source_path=f"C:/x/{vid}.mp4",
                      current_path=f"C:/x/{vid}.mp4")
    if state != "discovered":
        db.force_state(vid, state, reason="test setup", **fields)
    return db.get_video(vid)


def _outdated_full(db, vid=VID):
    return _row(db, vid, "outdated", reprocess_scope="full")


def _publish(env, rows, **kw):
    kw.setdefault("hostname", "srv")
    kw.setdefault("staging_dir", env.staging)
    kw.setdefault("repose_dir", env.queue)
    kw.setdefault("declared", NEW)
    return repose.publish_pending(env.db, rows, **kw)


def _consume(env, **kw):
    kw.setdefault("hostname", "gpu")
    kw.setdefault("repose_dir", env.queue)
    kw.setdefault("declared", NEW)
    kw.setdefault("batch", 2)
    return repose.consume_requests(env.db, dlc_queue=env.local_q, **kw)


def _adopt(env, **kw):
    kw.setdefault("repose_dir", env.queue)
    kw.setdefault("declared", NEW)
    return repose.adopt_staged_reposes(env.db, env.staging, **kw)


def _steps(db, vid, step=repose.STEP):
    conn = db._get_connection()
    try:
        return [r[0] for r in conn.execute(
            "SELECT status FROM processing_log WHERE video_id=? AND step=? ORDER BY id",
            (vid, step))]
    finally:
        conn.close()


# ---------------------------------------------------------------- publish

def test_publish_writes_a_portable_request(env):
    row = _outdated_full(env.db)
    out = _publish(env, [row])
    assert out["published"] == [VID]
    body = json.loads((env.queue / f"{VID}.json").read_text())
    assert body["video_rel"] == f"Analyzed/X/C01/{VID}.mp4"
    assert body["declared_scorer"] == NEW
    assert body["requested_by"] == "srv"
    assert _steps(env.db, VID) == ["published"]


def test_publish_is_idempotent_while_queued_or_inflight(env):
    row = _outdated_full(env.db)
    _publish(env, [row])
    out = _publish(env, [row])
    assert out["published"] == [] and out["republished"] == []
    inflight = env.queue / repose.INFLIGHT / f"{VID}.json"
    inflight.parent.mkdir(exist_ok=True)
    os.replace(env.queue / f"{VID}.json", inflight)
    out = _publish(env, [row])
    assert out["inflight"] == [VID]
    assert not (env.queue / f"{VID}.json").exists()


def test_publish_skips_when_the_new_pose_is_already_staged_or_archived(env):
    row = _outdated_full(env.db)
    (env.staging / f"{VID}{NEW}.h5").write_bytes(b"new")
    assert _publish(env, [row])["staged"] == [VID]
    (env.staging / f"{VID}{NEW}.h5").unlink()
    (env.archive / f"{VID}{NEW}.h5").write_bytes(b"new")
    assert _publish(env, [row])["in_archive"] == [VID]
    assert not (env.queue / f"{VID}.json").exists()


def test_publish_rewrites_a_queued_request_when_the_declared_model_changes(env):
    row = _outdated_full(env.db)
    _publish(env, [row])
    out = _publish(env, [row], declared="DLC_other_model")
    assert out["rewritten"] == [VID]
    assert json.loads((env.queue / f"{VID}.json").read_text())["declared_scorer"] == "DLC_other_model"


def test_publish_withdraws_only_its_own_request_when_the_row_healed(env):
    row = _outdated_full(env.db)
    _publish(env, [row])
    repose.write_json_atomic(env.queue / f"{VID2}.json",
                             {"video_id": VID2, "requested_by": "other-node"})
    out = _publish(env, [])
    assert out["withdrawn"] == [VID]
    assert not (env.queue / f"{VID}.json").exists()
    assert (env.queue / f"{VID2}.json").exists()


def test_publish_returns_an_inflight_request_nobody_heartbeats(env):
    row = _outdated_full(env.db)
    infl = env.queue / repose.INFLIGHT / f"{VID}.json"
    repose.write_json_atomic(infl, {"video_id": VID, "requested_by": "srv",
                                    "consumed_by": "gpu", "declared_scorer": NEW})
    old = time.time() - repose.STALE_S - 60
    os.utime(infl, (old, old))
    out = _publish(env, [row])
    assert out["returned"] == [VID]
    assert (env.queue / f"{VID}.json").exists() and not infl.exists()


def test_publish_names_a_remote_failure_once(env):
    row = _outdated_full(env.db)
    repose.write_json_atomic(env.queue / f"{VID}.failed.json",
                             {"host": "gpu", "error": "CUDA went away"})
    latched = set()
    out = _publish(env, [row], latched=latched)
    assert out["failed_remote"] == [VID]
    assert "failed_remote" in _steps(env.db, VID)
    out = _publish(env, [row], latched=latched)
    assert out["failed_remote"] == []          # latched: said once


# ---------------------------------------------------------------- consume

def test_consume_claims_by_rename_and_requeues_an_archived_row(env):
    _row(env.db, VID, "archived")
    _publish(env, [_outdated_full_row(VID)])
    out = _consume(env)
    assert out["queued"] == 1
    row = env.db.get_video(VID)
    assert row["state"] == "dlc_queued"
    assert row["mark_reason"].startswith(repose.REASON_PREFIX)
    assert row["reprocess_scope"] is None
    assert Path(row["current_path"]) == env.local_q / f"{VID}.mp4"
    assert (env.local_q / f"{VID}.mp4").read_bytes() == (env.archive / f"{VID}.mp4").read_bytes()
    assert not (env.queue / f"{VID}.json").exists()
    infl = json.loads((env.queue / repose.INFLIGHT / f"{VID}.json").read_text())
    assert infl["consumed_by"] == "gpu"
    assert "consumed" in _steps(env.db, VID)


def _outdated_full_row(vid):
    return {"video_id": vid, "reprocess_scope": "full", "updated_at": "t"}


def test_consume_loses_the_race_cleanly(env, monkeypatch):
    _row(env.db, VID, "archived")
    _publish(env, [_outdated_full_row(VID)])
    real = os.replace

    def taken(src, dst):
        if Path(src).name == f"{VID}.json" and repose.INFLIGHT in str(dst):
            raise FileNotFoundError(src)     # another node renamed it first
        return real(src, dst)

    monkeypatch.setattr(repose.os, "replace", taken)
    out = _consume(env)
    assert out["claimed_elsewhere"] == 1 and out["queued"] == 0
    assert env.db.get_video(VID)["state"] == "archived"


def test_consume_respects_the_batch_cap(env):
    for v in (VID, VID2, VID3):
        _row(env.db, v, "archived")
    _publish(env, [_outdated_full_row(v) for v in (VID, VID2, VID3)])
    out = _consume(env, batch=2)
    assert out["queued"] == 2 and out["skipped_cap"] == 1
    assert len(list(env.queue.glob("*.json"))) == 1
    out = _consume(env, batch=2)             # two still in flight: nothing more
    assert out["queued"] == 0 and out["skipped_cap"] == 1


def test_consume_refuses_locked_and_human_held_rows(env):
    _row(env.db, VID, "crystallized")
    _row(env.db, VID2, "triage")
    _publish(env, [_outdated_full_row(VID), _outdated_full_row(VID2)])
    out = _consume(env)
    assert out["refused"] == 2 and out["queued"] == 0
    assert (env.queue / f"{VID}.json").exists() and (env.queue / f"{VID2}.json").exists()
    assert env.db.get_video(VID)["state"] == "crystallized"


def test_consume_holds_a_row_that_failed_too_often_but_retries_a_fresh_failure(env):
    _row(env.db, VID, "failed", error_count=3, error_message="boom")
    _row(env.db, VID2, "failed", error_count=1, error_message="once")
    _publish(env, [_outdated_full_row(VID), _outdated_full_row(VID2)])
    out = _consume(env)
    assert out["held_failed"] == 1 and out["queued"] == 1
    assert env.db.get_video(VID)["state"] == "failed"
    assert env.db.get_video(VID2)["state"] == "dlc_queued"
    assert "held" in _steps(env.db, VID)


def test_consume_removes_an_old_pose_that_would_be_mistaken_for_the_new_one(env):
    _row(env.db, VID, "archived")
    (env.local_q / f"{VID}{OLD}.h5").write_bytes(b"old")
    (env.local_q / f"{VID}{OLD}.csv").write_bytes(b"old")
    _publish(env, [_outdated_full_row(VID)])
    assert _consume(env)["queued"] == 1
    assert not list(env.local_q.glob(f"{VID}DLC*"))


def test_consume_completes_when_the_declared_pose_is_already_local(env):
    _row(env.db, VID, "archived")
    (env.local_q / f"{VID}{NEW}.h5").write_bytes(b"new")
    _publish(env, [_outdated_full_row(VID)])
    out = _consume(env)
    assert out["completed"] == 1 and out["queued"] == 0
    row = env.db.get_video(VID)
    assert row["state"] == "dlc_complete"
    assert Path(row["dlc_output_path"]) == env.local_q / f"{VID}{NEW}.h5"


def test_consume_closes_a_request_the_archive_already_satisfies(env):
    _row(env.db, VID, "archived")
    _publish(env, [_outdated_full_row(VID)])
    (env.archive / f"{VID}{NEW}.h5").write_bytes(b"new")
    out = _consume(env)
    assert out["satisfied"] == 1
    assert not (env.queue / f"{VID}.json").exists()
    assert not (env.queue / repose.INFLIGHT / f"{VID}.json").exists()
    assert env.db.get_video(VID)["state"] == "archived"


def test_consume_resolves_the_video_by_relative_path_when_the_absolute_one_lies(env):
    _row(env.db, VID, "archived")
    _publish(env, [_outdated_full_row(VID)])
    p = env.queue / f"{VID}.json"
    body = json.loads(p.read_text())
    body["video_path"] = "Z:\\nowhere\\" + VID + ".mp4"
    repose.write_json_atomic(p, body)
    assert _consume(env)["queued"] == 1
    assert (env.local_q / f"{VID}.mp4").exists()


def test_consume_puts_the_request_back_when_no_video_can_be_found(env, monkeypatch):
    _row(env.db, VID, "archived")
    _publish(env, [_outdated_full_row(VID)])
    monkeypatch.setattr(repose, "resolve_request_video", lambda vid, body: None)
    out = _consume(env)
    assert out["no_video"] == 1
    assert (env.queue / f"{VID}.json").exists()
    assert not (env.queue / repose.INFLIGHT / f"{VID}.json").exists()


def test_consume_registers_a_video_this_node_never_saw(env):
    _publish(env, [_outdated_full_row(VID)])
    assert _consume(env)["queued"] == 1
    row = env.db.get_video(VID)
    assert row["state"] == "dlc_queued"
    assert row["animal_id"] == "ABC0101"


def test_consume_skips_a_row_already_in_flight_here_but_heals_a_husk(env):
    _row(env.db, VID, "dlc_running", current_path=str(env.local_q / f"{VID}.mp4"))
    (env.local_q / f"{VID}.mp4").write_bytes(b"here")
    _row(env.db, VID2, "dlc_complete", current_path=str(env.local_q / f"{VID2}.mp4"))
    _publish(env, [_outdated_full_row(VID), _outdated_full_row(VID2)])
    out = _consume(env)
    assert out["in_flight"] == 1
    assert env.db.get_video(VID)["state"] == "dlc_running"
    assert env.db.get_video(VID2)["state"] == "dlc_queued"   # husk re-driven
    assert (env.queue / f"{VID}.json").exists()


def test_heartbeat_touches_only_this_nodes_live_requests(env):
    _row(env.db, VID, "dlc_running")
    infl = env.queue / repose.INFLIGHT / f"{VID}.json"
    repose.write_json_atomic(infl, {"video_id": VID, "consumed_by": "gpu"})
    old = time.time() - 7200
    os.utime(infl, (old, old))
    assert repose.heartbeat(env.db, hostname="gpu", repose_dir=env.queue) == 1
    assert time.time() - infl.stat().st_mtime < 60
    os.utime(infl, (old, old))
    assert repose.heartbeat(env.db, hostname="other", repose_dir=env.queue) == 0


def test_note_failure_writes_only_for_a_requested_video(env):
    assert repose.note_failure(VID, "gpu", "CUDA", repose_dir=env.queue) is False
    repose.write_json_atomic(env.queue / repose.INFLIGHT / f"{VID}.json", {"video_id": VID})
    assert repose.note_failure(VID, "gpu", "CUDA", repose_dir=env.queue) is True
    note = json.loads((env.queue / f"{VID}.failed.json").read_text())
    assert note["host"] == "gpu" and "CUDA" in note["error"]


# ---------------------------------------------------------------- adopt

def test_adopt_narrows_the_scope_and_records_the_staged_pose(env):
    _outdated_full(env.db)
    env.db.set_fields(VID, mark_reason="hand: please re-run")
    h5 = env.staging / f"{VID}{NEW}.h5"
    h5.write_bytes(b"new")
    repose.write_json_atomic(env.queue / repose.INFLIGHT / f"{VID}.json", {"video_id": VID})
    assert _adopt(env) == [VID]
    row = env.db.get_video(VID)
    assert row["state"] == "outdated"
    assert row["reprocess_scope"] == "segmentation"
    assert Path(row["dlc_output_path"]) == h5
    assert row["mark_reason"] == "hand: please re-run"      # a hand-mark survives
    assert not (env.queue / repose.INFLIGHT / f"{VID}.json").exists()
    assert "adopted" in _steps(env.db, VID)
    assert _adopt(env) == []                                  # idempotent


def test_adopt_ignores_an_old_model_pose_and_current_archived_rows(env):
    _outdated_full(env.db)
    (env.staging / f"{VID}{OLD}.h5").write_bytes(b"old")
    assert _adopt(env) == []
    assert env.db.get_video(VID)["reprocess_scope"] == "full"
    _row(env.db, VID2, "archived")
    (env.archive / f"{VID2}_processing_manifest.json").write_text(
        json.dumps({"dlc_model": {"dlc_scorer": NEW}}))
    (env.staging / f"{VID2}{NEW}.h5").write_bytes(b"new")
    assert _adopt(env) == []
    assert env.db.get_video(VID2)["state"] == "archived"


def test_adopt_takes_an_archived_row_whose_manifest_names_the_old_model(env):
    _row(env.db, VID, "archived")
    (env.archive / f"{VID}_processing_manifest.json").write_text(
        json.dumps({"dlc_model": {"dlc_scorer": OLD}}))
    (env.staging / f"{VID}{NEW}.h5").write_bytes(b"new")
    assert _adopt(env) == [VID]
    row = env.db.get_video(VID)
    assert row["state"] == "outdated" and row["reprocess_scope"] == "segmentation"


def test_adopt_names_a_failed_row_once_and_leaves_it(env):
    _row(env.db, VID, "failed", error_message="copy failed")
    (env.staging / f"{VID}{NEW}.h5").write_bytes(b"new")
    latched = set()
    assert _adopt(env, latched=latched) == []
    assert env.db.get_video(VID)["state"] == "failed"
    assert f"adopt_failed:{VID}" in latched


# ---------------------------------------------------------------- round trip

def test_publish_consume_adopt_closes_the_loop_without_a_second_request(env):
    srv = env.db                                        # the processing node
    gpu = WatcherDB(db_path=env.nas / "gpu.db")         # a GPU node's own ledger
    row = _outdated_full(srv)
    _row(gpu, VID, "archived")
    assert _publish(env, [row])["published"] == [VID]
    assert repose.consume_requests(gpu, dlc_queue=env.local_q, hostname="gpu",
                                   repose_dir=env.queue, declared=NEW)["queued"] == 1
    # the GPU node is posing: the publisher must not ask again
    assert _publish(env, [row])["inflight"] == [VID]
    assert not (env.queue / f"{VID}.json").exists()
    # the pose comes back through staging
    (env.staging / f"{VID}.mp4").write_bytes(b"v")
    (env.staging / f"{VID}{NEW}.h5").write_bytes(b"new")
    assert _adopt(env) == [VID]
    assert not any(env.queue.rglob("*.json"))
    assert srv.get_video(VID)["reprocess_scope"] == "segmentation"
    # and a further publish pass has nothing to ask for
    out = _publish(env, [srv.get_video(VID)])
    assert out["published"] == [] and out["staged"] == [VID]


# ---------------------------------------------------------------- stage_to_nas

class RecDB:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def rec(*a, **k):
            self.calls.append((name, a, k))
        return rec

    def called(self, name):
        return [c for c in self.calls if c[0] == name]


def _bare_dlc(env, monkeypatch):
    o = object.__new__(DLCOrchestrator)
    o.db = RecDB()
    o.hostname = "gpu"
    o.staging_dir = env.staging
    o.config = SimpleNamespace(also_process=False, dlc_gpu_device=0)
    o._sync_to_connectome = lambda *a, **k: None
    o._clear_archive_backoff = lambda vid: None
    o._note_archive_failure = lambda vid: 60
    monkeypatch.setattr(orch.Paths, "DLC_QUEUE", env.local_q)
    monkeypatch.setattr(orch.Paths, "PROCESSING", env.processing)
    monkeypatch.setattr(orch.Paths, "DLC_STAGING", env.staging)
    return o


def test_stage_to_nas_lands_files_atomically_pose_last_and_records_it(env, monkeypatch):
    o = _bare_dlc(env, monkeypatch)
    mp4 = env.local_q / f"{VID}.mp4"
    mp4.write_bytes(b"video")
    (env.local_q / f"{VID}{NEW}.h5").write_bytes(b"pose")
    order = []
    real = orch.safe_move

    def spy(src, dst):
        order.append(Path(src).name)
        assert Path(dst).name.endswith(".part")          # never the final name
        return real(src, dst)

    monkeypatch.setattr(orch, "safe_move", spy)
    o._stage_to_nas({"id": VID, "data": {"current_path": str(mp4)}})
    assert order == [f"{VID}.mp4", f"{VID}{NEW}.h5"]     # pose last
    assert sorted(p.name for p in env.staging.iterdir()) == [f"{VID}.mp4", f"{VID}{NEW}.h5"]
    assert not list(env.staging.glob("*.part"))
    (name, args, kw), = o.db.called("update_state")
    assert args[1] == "archived"
    assert Path(kw["dlc_output_path"]) == env.staging / f"{VID}{NEW}.h5"


def test_stage_to_nas_fails_the_row_when_a_file_does_not_arrive(env, monkeypatch):
    o = _bare_dlc(env, monkeypatch)
    mp4 = env.local_q / f"{VID}.mp4"
    mp4.write_bytes(b"video")
    (env.local_q / f"{VID}{NEW}.h5").write_bytes(b"pose")
    monkeypatch.setattr(orch, "safe_move",
                        lambda src, dst: False if src.name.endswith(".h5") else orch.safe_copy(src, dst))
    with pytest.raises(IOError):
        o._stage_to_nas({"id": VID, "data": {"current_path": str(mp4)}})
    assert o.db.called("mark_failed") and not o.db.called("update_state")


# ---------------------------------------------------------------- also_process

def test_local_archive_files_from_processing_cleans_both_and_closes_the_request(env, monkeypatch):
    import mousereach.archive.core as core
    o = _bare_dlc(env, monkeypatch)
    seen = {}

    def fake_archive(video_id, **kw):
        seen.update(kw)
        return {"success": True, "files_moved": ["a"]}

    monkeypatch.setattr(core, "archive_video", fake_archive)
    (env.processing / f"{VID}_segments.json").write_text("{}")
    (env.processing / f"{VID}{NEW}.h5").write_bytes(b"pose")
    (env.local_q / f"{VID}.mp4").write_bytes(b"video")
    repose.write_json_atomic(env.queue / repose.INFLIGHT / f"{VID}.json", {"video_id": VID})
    assert o._archive_locally_processed({"id": VID}) is True
    assert Path(seen["source_dir"]) == env.processing
    assert not list(env.processing.glob(f"{VID}*")) and not list(env.local_q.glob(f"{VID}*"))
    assert not (env.queue / repose.INFLIGHT / f"{VID}.json").exists()


# ---------------------------------------------------------------- recovery guard

def test_recovery_leaves_live_local_work_alone(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    assert _is_live_local_work({"state": "dlc_queued", "current_path": str(f)}) is True
    assert _is_live_local_work({"state": "dlc_queued", "current_path": str(tmp_path / "gone.mp4")}) is False
    assert _is_live_local_work({"state": "archived", "current_path": str(f)}) is False
    assert _is_live_local_work(None) is False
