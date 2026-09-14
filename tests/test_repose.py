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
    staging = nas / "Processing" / "Posed"
    archive = nas / "Analyzed" / "X" / "C01"
    local_q = tmp_path / "local" / "DLC_Queue"
    processing = tmp_path / "local" / "Processing"
    for d in (queue, staging, archive, local_q, processing):
        d.mkdir(parents=True)
    # WHY a plain FILE at the retired staging name, as the migrated share has:
    # code that hardcoded NAS_ROOT / "Processing" / "DLC_Complete" instead of
    # Paths.DLC_STAGING then raises instead of staging into a folder the test
    # happens to inspect.
    (nas / "Processing" / "DLC_Complete").write_text("retired folder", encoding="ascii")
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


def test_publish_narrows_instead_of_asking_when_the_new_pose_is_already_staged(env):
    row = _outdated_full(env.db)
    h5 = env.staging / f"{VID}{NEW}.h5"
    h5.write_bytes(b"new")
    out = _publish(env, [row])
    assert out["staged"] == [VID] and out["narrowed"] == [VID]
    assert not (env.queue / f"{VID}.json").exists()
    row = env.db.get_video(VID)
    assert row["state"] == "outdated" and row["reprocess_scope"] == "segmentation"
    assert Path(row["dlc_output_path"]) == h5


def test_publish_narrows_and_withdraws_when_the_new_pose_reached_the_archive(env):
    row = _outdated_full(env.db)
    _publish(env, [row])
    assert (env.queue / f"{VID}.json").exists()
    h5 = env.archive / f"{VID}{NEW}.h5"
    h5.write_bytes(b"new")                      # e.g. an also-process node archived it
    out = _publish(env, [row])
    assert out["narrowed"] == [VID] and out["withdrawn"] == [VID]
    assert not (env.queue / f"{VID}.json").exists()
    row = env.db.get_video(VID)
    assert row["reprocess_scope"] == "segmentation"
    assert Path(row["dlc_output_path"]) == h5


def test_publish_does_nothing_when_the_declared_model_is_unknown(env):
    row = _outdated_full(env.db)
    out = _publish(env, [row], declared="")
    assert out["unknown_model"] == [VID] and out["published"] == []
    assert not list(env.queue.glob("*.json"))


def test_publish_with_no_rows_still_sweeps(env):
    row = _outdated_full(env.db)
    _publish(env, [row])
    env.db.force_state(VID, "archived", reason="healed by hand")
    out = _publish(env, [])
    assert out["withdrawn"] == [VID]


def test_publish_closes_a_stale_claim_whose_pose_has_arrived(env):
    row = _outdated_full(env.db)
    infl = env.queue / repose.INFLIGHT / f"{VID}.json"
    repose.write_json_atomic(infl, {"video_id": VID, "requested_by": "srv",
                                    "consumed_by": "gpu", "declared_scorer": NEW})
    old = time.time() - repose.STALE_S - 60
    os.utime(infl, (old, old))
    (env.staging / f"{VID}{NEW}.h5").write_bytes(b"new")
    out = _publish(env, [row])
    assert out["closed"] == [VID] and out["returned"] == []
    assert not infl.exists() and not (env.queue / f"{VID}.json").exists()
    assert env.db.get_video(VID)["reprocess_scope"] == "segmentation"


def test_publish_costs_no_lookups_for_a_request_already_in_flight(env, monkeypatch):
    row = _outdated_full(env.db)
    repose.write_json_atomic(env.queue / repose.INFLIGHT / f"{VID}.json",
                             {"video_id": VID, "requested_by": "srv", "declared_scorer": NEW})
    calls = []
    monkeypatch.setattr(repose, "declared_pose_in_archive",
                        lambda vid, d: calls.append(vid) or None)
    monkeypatch.setattr(repose, "archived_video", lambda vid: calls.append(vid) or None)
    out = _publish(env, [row])
    assert out["inflight"] == [VID] and calls == []


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
    assert "narrowed" in _steps(env.db, VID)
    assert _adopt(env) == []                                  # idempotent


def test_adopt_leaves_a_row_narrowed_for_another_reason_alone(env):
    _row(env.db, VID, "outdated", reprocess_scope="kinematics",
         dlc_output_path=str(env.archive / f"{VID}{OLD}.h5"))
    (env.staging / f"{VID}{NEW}.h5").write_bytes(b"new")
    assert _adopt(env) == []
    row = env.db.get_video(VID)
    assert row["reprocess_scope"] == "kinematics"


def test_adopt_takes_an_unresolvable_row(env):
    _row(env.db, VID, "unresolvable")
    h5 = env.staging / f"{VID}{NEW}.h5"
    h5.write_bytes(b"new")
    assert _adopt(env) == [VID]
    row = env.db.get_video(VID)
    assert row["state"] == "outdated" and row["reprocess_scope"] == "segmentation"
    assert Path(row["dlc_output_path"]) == h5


def test_adopt_does_nothing_when_the_declared_model_is_unknown(env):
    _outdated_full(env.db)
    (env.staging / f"{VID}{NEW}.h5").write_bytes(b"new")
    assert _adopt(env, declared="") == []
    assert env.db.get_video(VID)["reprocess_scope"] == "full"


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


def test_stage_to_nas_copies_all_then_renames_pose_last_then_deletes(env, monkeypatch):
    o = _bare_dlc(env, monkeypatch)
    mp4 = env.local_q / f"{VID}.mp4"
    mp4.write_bytes(b"video")
    (env.local_q / f"{VID}{NEW}.csv").write_bytes(b"csv")
    (env.local_q / f"{VID}{NEW}.h5").write_bytes(b"pose")
    copies, renames = [], []
    real_copy = orch.safe_copy
    real_replace = Path.replace

    def spy_copy(src, dst, verify=True):
        copies.append(Path(src).name)
        assert Path(dst).name.endswith(".part")          # never the final name
        assert Path(src).exists()                        # originals untouched so far
        return real_copy(src, dst, verify=verify)

    def spy_replace(self, target):
        renames.append(Path(target).name)
        assert all((env.local_q / n).exists() for n in copies)   # still nothing deleted
        return real_replace(self, target)

    monkeypatch.setattr(orch, "safe_copy", spy_copy)
    monkeypatch.setattr(Path, "replace", spy_replace)
    o._stage_to_nas({"id": VID, "data": {"current_path": str(mp4)}})
    assert renames[-1] == f"{VID}{NEW}.h5"               # pose strictly last
    assert sorted(p.name for p in env.staging.iterdir()) == sorted(
        [f"{VID}.mp4", f"{VID}{NEW}.csv", f"{VID}{NEW}.h5"])
    assert not list(env.staging.glob("*.part"))
    assert not list(env.local_q.iterdir())                # originals gone at the end
    (name, args, kw), = o.db.called("update_state")
    assert args[1] == "archived"
    assert Path(kw["dlc_output_path"]) == env.staging / f"{VID}{NEW}.h5"


def test_stage_to_nas_fails_the_row_and_keeps_the_originals_when_a_copy_fails(env, monkeypatch):
    o = _bare_dlc(env, monkeypatch)
    mp4 = env.local_q / f"{VID}.mp4"
    mp4.write_bytes(b"video")
    h5 = env.local_q / f"{VID}{NEW}.h5"
    h5.write_bytes(b"pose")
    real_copy = orch.safe_copy
    monkeypatch.setattr(orch, "safe_copy",
                        lambda src, dst, verify=True: False if src.name.endswith(".h5")
                        else real_copy(src, dst, verify=verify))
    with pytest.raises(IOError):
        o._stage_to_nas({"id": VID, "data": {"current_path": str(mp4)}})
    assert o.db.called("mark_failed") and not o.db.called("update_state")
    assert mp4.exists() and h5.exists()                   # nothing lost
    assert not list(env.staging.iterdir())                # no final file, no .part


def test_stage_to_nas_resume_stages_a_leftover_pose_before_marking_done(env, monkeypatch):
    o = _bare_dlc(env, monkeypatch)
    staged_mp4 = env.staging / f"{VID}.mp4"
    staged_mp4.write_bytes(b"video")                      # the mp4 went; the pose did not
    h5 = env.local_q / f"{VID}{NEW}.h5"
    h5.write_bytes(b"pose")
    o._stage_to_nas({"id": VID, "data": {"current_path": str(staged_mp4)}})
    assert (env.staging / f"{VID}{NEW}.h5").exists() and not h5.exists()
    (name, args, kw), = o.db.called("force_state")
    assert args[1] == "archived"
    assert Path(kw["dlc_output_path"]) == env.staging / f"{VID}{NEW}.h5"


def test_stage_to_nas_resume_with_no_pose_anywhere_fails_the_row(env, monkeypatch):
    o = _bare_dlc(env, monkeypatch)
    staged_mp4 = env.staging / f"{VID}.mp4"
    staged_mp4.write_bytes(b"video")
    with pytest.raises(IOError):
        o._stage_to_nas({"id": VID, "data": {"current_path": str(staged_mp4)}})
    assert o.db.called("mark_failed") and not o.db.called("force_state")


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

def test_recovery_leaves_live_local_work_alone(tmp_path, monkeypatch):
    import mousereach.config as cfg
    monkeypatch.setattr(cfg.Paths, "NAS_ROOT", tmp_path / "nas")
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    assert _is_live_local_work({"state": "dlc_queued", "current_path": str(f)}) is True
    assert _is_live_local_work({"state": "dlc_queued", "current_path": str(tmp_path / "gone.mp4")}) is False
    assert _is_live_local_work({"state": "archived", "current_path": str(f)}) is False
    assert _is_live_local_work(None) is False
    # this node's own verdict about the archive is never overwritten
    assert _is_live_local_work({"state": "outdated", "current_path": None}) is True
    # a file on the shared drive is not this node's working copy
    nas_file = tmp_path / "nas" / "Analyzed" / "v.mp4"
    nas_file.parent.mkdir(parents=True)
    nas_file.write_bytes(b"x")
    assert _is_live_local_work({"state": "dlc_complete", "current_path": str(nas_file)}) is False


# ---------------------------------------------------------------- consume guards

def test_consume_takes_nothing_when_this_node_cannot_pose(env):
    _row(env.db, VID, "archived")
    _publish(env, [_outdated_full_row(VID)])
    out = _consume(env, can_pose=False)
    assert out["queued"] == 0 and (env.queue / f"{VID}.json").exists()


def test_consume_backs_off_a_request_it_could_not_resolve(env, monkeypatch):
    _row(env.db, VID, "archived")
    _publish(env, [_outdated_full_row(VID)])
    monkeypatch.setattr(repose, "resolve_request_video", lambda vid, body: None)
    retry = {}
    assert _consume(env, retry_after=retry)["no_video"] == 1
    assert retry[VID] > time.time()
    out = _consume(env, retry_after=retry)
    assert out["backoff"] == 1 and out["no_video"] == 0


def test_consume_takes_nothing_when_the_declared_model_is_unknown(env):
    _row(env.db, VID, "archived")
    _publish(env, [_outdated_full_row(VID)])
    out = _consume(env, declared="")
    assert out["unknown_model"] == 1 and out["queued"] == 0
    assert (env.queue / f"{VID}.json").exists()


def test_heartbeat_keeps_a_human_held_video_alive_but_not_a_finished_one(env):
    _row(env.db, VID, "triage")
    _row(env.db, VID2, "archived")
    for v in (VID, VID2):
        p = env.queue / repose.INFLIGHT / f"{v}.json"
        repose.write_json_atomic(p, {"video_id": v, "consumed_by": "gpu"})
        old = time.time() - 7200
        os.utime(p, (old, old))
    assert repose.heartbeat(env.db, hostname="gpu", repose_dir=env.queue) == 1
    assert time.time() - (env.queue / repose.INFLIGHT / f"{VID}.json").stat().st_mtime < 60
    assert time.time() - (env.queue / repose.INFLIGHT / f"{VID2}.json").stat().st_mtime > 3600


def test_close_request_leaves_another_nodes_claim(env):
    repose.write_json_atomic(env.queue / repose.INFLIGHT / f"{VID}.json",
                             {"video_id": VID, "consumed_by": "other-gpu"})
    assert repose.close_request(VID, env.queue, consumed_by="gpu") is False
    assert (env.queue / repose.INFLIGHT / f"{VID}.json").exists()
    assert repose.close_request(VID, env.queue, consumed_by="other-gpu") is True
    assert not (env.queue / repose.INFLIGHT / f"{VID}.json").exists()


# ---------------------------------------------------------------- DLC node paths

def test_completion_scan_waits_for_the_declared_pose_on_a_re_posed_row(env, monkeypatch):
    o = _bare_dlc(env, monkeypatch)
    o.db = env.db
    _row(env.db, VID, "dlc_queued", mark_reason=f"{repose.REASON_PREFIX} from srv",
         current_path=str(env.local_q / f"{VID}.mp4"))
    _row(env.db, VID2, "dlc_queued", current_path=str(env.local_q / f"{VID2}.mp4"))
    for v in (VID, VID2):
        (env.local_q / f"{v}.mp4").write_bytes(b"v")
        (env.local_q / f"{v}{OLD}.h5").write_bytes(b"old")
    monkeypatch.setattr(repose, "declared_scorer", lambda: NEW)
    assert o._scan_for_dlc_completions() == 1
    assert env.db.get_video(VID)["state"] == "dlc_queued"      # re-pose: still waiting
    assert env.db.get_video(VID2)["state"] == "dlc_complete"   # ordinary: old pose accepted
    (env.local_q / f"{VID}{NEW}.h5").write_bytes(b"new")
    assert o._scan_for_dlc_completions() == 1
    row = env.db.get_video(VID)
    assert row["state"] == "dlc_complete"
    assert Path(row["dlc_output_path"]) == env.local_q / f"{VID}{NEW}.h5"


def test_queue_recovery_resumes_a_re_pose_and_clears_the_old_pose(env, monkeypatch):
    o = _bare_dlc(env, monkeypatch)
    o.db = env.db
    _outdated_full(env.db)
    (env.local_q / f"{VID}.mp4").write_bytes(b"v")
    (env.local_q / f"{VID}{OLD}.h5").write_bytes(b"old")
    monkeypatch.setattr(repose, "declared_scorer", lambda: NEW)
    o._recover_local_dlc_queue()
    row = env.db.get_video(VID)
    assert row["state"] == "dlc_queued"
    assert row["mark_reason"].startswith(repose.REASON_PREFIX)
    assert not list(env.local_q.glob(f"{VID}DLC*"))


def test_dlc_run_refuses_a_re_pose_from_the_wrong_model(env, monkeypatch, tmp_path):
    import mousereach.dlc.core as dlc_core
    o = _bare_dlc(env, monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("x")
    o.config = SimpleNamespace(dlc_config_path=cfg, dlc_gpu_device=0, also_process=False)
    mp4 = env.local_q / f"{VID}.mp4"
    mp4.write_bytes(b"v")
    monkeypatch.setattr(dlc_core, "resolve_dlc_shuffle", lambda: (1, "test"))

    def fake_dlc(video_paths, config_path, output_dir, gpu, save_as_csv, shuffle):
        (Path(output_dir) / f"{VID}{OLD}.h5").write_bytes(b"old-model pose")
        return [{"status": "success"}]

    monkeypatch.setattr(dlc_core, "run_dlc_batch", fake_dlc)
    monkeypatch.setattr(repose, "declared_scorer", lambda: NEW)
    repose.write_json_atomic(env.queue / repose.INFLIGHT / f"{VID}.json", {"video_id": VID})
    with pytest.raises(RuntimeError):
        o._process_single_dlc({"id": VID, "data": {
            "current_path": str(mp4), "mark_reason": f"{repose.REASON_PREFIX} from srv"}})
    assert o.db.called("mark_failed")
    assert not any(args[1] == "dlc_complete" for _, args, _ in o.db.called("update_state"))
    assert (env.queue / f"{VID}.failed.json").exists()


class StopHere(Exception):
    pass


def test_local_pipeline_stages_the_video_beside_the_pose(env, monkeypatch):
    o = _bare_dlc(env, monkeypatch)
    mp4 = env.local_q / f"{VID}.mp4"
    mp4.write_bytes(b"video")
    h5 = env.local_q / f"{VID}{NEW}.h5"
    h5.write_bytes(b"pose")
    monkeypatch.setattr(orch, "resolve_pose_input", lambda *a, **k: h5)

    class DB(RecDB):
        def update_state(self, *a, **k):
            raise StopHere()

    o.db = DB()
    with pytest.raises(StopHere):
        o._run_local_pipeline({"id": VID, "data": {"current_path": str(mp4), "dlc_output_path": str(h5)}})
    assert (env.processing / f"{VID}.mp4").read_bytes() == b"video"
    assert (env.processing / f"{VID}{NEW}.h5").exists()


def test_local_pipeline_fails_the_row_when_the_video_is_missing(env, monkeypatch):
    o = _bare_dlc(env, monkeypatch)
    h5 = env.local_q / f"{VID}{NEW}.h5"
    h5.write_bytes(b"pose")
    monkeypatch.setattr(orch, "resolve_pose_input", lambda *a, **k: h5)
    o._run_local_pipeline({"id": VID, "data": {"current_path": str(env.local_q / "gone.mp4"),
                                               "dlc_output_path": str(h5)}})
    assert o.db.called("mark_failed") and not o.db.called("update_state")


# ---------------------------------------------------------------- scanner flags

def test_scan_full_only_marks_only_rows_that_need_a_new_pose(tmp_path, monkeypatch):
    import mousereach.pipeline.versions as versions_mod
    from mousereach.watcher.reprocessor import ReprocessingScanner
    coh = tmp_path / "Analyzed" / "Connectome" / "CNT99"
    coh.mkdir(parents=True)
    a, b = "20240101_ABC0101_P1", "20240101_ABC0102_P1"
    current = {"versions": {"dlc_scorer": NEW, "segmenter": "2.0.0", "reach_detector": "1.0.0",
                            "outcome_detector": "1.0.0", "assignment": "1.0.0",
                            "kinematic_extractor": "1.0.0"}}
    stale = {"segmenter": "1.0.0", "reach_detector": "1.0.0", "outcome_detector": "1.0.0",
             "assignment": "1.0.0", "kinematic_extractor": "1.0.0"}
    for vid, scorer in ((a, NEW), (b, OLD)):          # a: stale segmenter; b: old pose
        (coh / f"{vid}_processing_manifest.json").write_text(json.dumps(
            {"dlc_model": {"dlc_scorer": scorer}, "pipeline_versions": stale}))
        (coh / f"{vid}.mp4").write_text("x")
        (coh / f"{vid}{scorer}.h5").write_text("pose")
    monkeypatch.setattr(versions_mod, "get_current_versions", lambda root=None: current)
    monkeypatch.setattr(versions_mod, "declaration_drift", lambda cur: [])

    class DB:
        def __init__(self):
            self.forced = []
        def _get_connection(self):
            class C:
                def execute(self, *a):
                    return [(a_,) for a_ in (a, b)]
                def close(self):
                    pass
            return C()
        def get_videos_in_state(self, state):
            return [{"video_id": v, "state": "archived"} for v in (a, b)] if state == "archived" else []
        def force_state(self, video_id, state, **k):
            self.forced.append((video_id, state, k.get("reprocess_scope")))
        def get_video(self, vid):
            return {"video_id": vid, "state": "archived"}

    sc = object.__new__(ReprocessingScanner)
    sc.db = DB()
    sc.nas_root = tmp_path
    sc.archive_dir = tmp_path / "Analyzed"
    sc._manifest_cache = {}
    summary = sc.scan(mark_outdated=True, adopt_orphans=False, full_only=True)
    assert summary["adopted"] == 0
    assert [f for f in sc.db.forced if f[1] == "outdated"] == [(b, "outdated", "full")]
    assert summary["outdated_partial_unmarked"] == 1


# ---------------------------------------------------------------- retry budget

def test_a_handler_that_marked_failed_is_not_marked_again(env, monkeypatch):
    """One attempt must cost one attempt: the handler marks the row failed
    and re-raises, and the dispatch guard used to mark it a second time."""
    o = _bare_dlc(env, monkeypatch)
    o.db = env.db
    _row(env.db, VID, "dlc_queued")

    def boom(work):
        env.db.mark_failed(VID, "pose failed")
        raise RuntimeError("pose failed")

    o._process_single_dlc = boom
    o._backup_local_db = lambda: None
    assert o._dispatch_work({"type": "single_dlc", "id": VID, "data": {}}) is False
    assert env.db.get_video(VID)["error_count"] == 1


def test_a_handler_that_raised_without_marking_is_still_marked(env, monkeypatch):
    o = _bare_dlc(env, monkeypatch)
    o.db = env.db
    _row(env.db, VID, "dlc_queued")
    o._process_single_dlc = lambda work: (_ for _ in ()).throw(RuntimeError("died"))
    o._backup_local_db = lambda: None
    assert o._dispatch_work({"type": "single_dlc", "id": VID, "data": {}}) is False
    row = env.db.get_video(VID)
    assert row["state"] == "failed" and row["error_count"] == 1


def _dlc_with_config(env, monkeypatch, **over):
    o = _bare_dlc(env, monkeypatch)
    o.db = env.db
    o.config = SimpleNamespace(also_process=False, dlc_gpu_device=0,
                               dlc_config_path=None,
                               max_retries=over.pop("max_retries", 3))
    return o


def test_queue_recovery_retries_a_failed_row_whose_files_are_still_here(env, monkeypatch):
    o = _dlc_with_config(env, monkeypatch)
    monkeypatch.setattr(repose, "declared_scorer", lambda: NEW)
    _row(env.db, VID, "failed", error_count=1, error_message="stage failed")
    (env.local_q / f"{VID}.mp4").write_bytes(b"v")
    _row(env.db, VID2, "failed", error_count=1, error_message="stage failed")
    (env.local_q / f"{VID2}.mp4").write_bytes(b"v")
    (env.local_q / f"{VID2}{NEW}.h5").write_bytes(b"pose")   # pose survived
    o._recover_local_dlc_queue()
    assert env.db.get_video(VID)["state"] == "dlc_queued"    # needs a pose
    assert env.db.get_video(VID2)["state"] == "dlc_complete"  # only needs staging


def test_queue_recovery_leaves_a_row_that_failed_too_often(env, monkeypatch):
    o = _dlc_with_config(env, monkeypatch)
    monkeypatch.setattr(repose, "declared_scorer", lambda: NEW)
    _row(env.db, VID, "failed", error_count=3, error_message="stage failed")
    (env.local_q / f"{VID}.mp4").write_bytes(b"v")
    o._recover_local_dlc_queue()
    assert env.db.get_video(VID)["state"] == "failed"


# ---------------------------------------------------------------- scorer cache

def test_declared_scorer_expires_instead_of_lasting_the_process(monkeypatch):
    """A watcher started before a model change kept preferring the OLD pose
    whenever two sat side by side -- exactly what a re-pose creates."""
    import mousereach.pipeline.manifest as mf
    import mousereach.pipeline.versions as versions_mod
    seen = []

    def fake_versions(*a, **k):
        seen.append(1)
        return {"versions": {"dlc_scorer": OLD if len(seen) == 1 else NEW}}

    monkeypatch.setattr(versions_mod, "get_current_versions", fake_versions)
    monkeypatch.setattr(mf, "_DECLARED_SCORER", None)
    monkeypatch.setattr(mf, "_DECLARED_SCORER_AT", 0.0)
    assert mf.declared_dlc_scorer() == OLD
    assert mf.declared_dlc_scorer() == OLD        # still cached, one read so far
    assert len(seen) == 1
    assert repose.declared_scorer() == NEW        # the round trip always reads fresh
    assert mf.declared_dlc_scorer() == NEW        # and the shared cache is updated
