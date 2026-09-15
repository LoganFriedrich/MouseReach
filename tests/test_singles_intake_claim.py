"""Singles dropped onto the share while GPU nodes run: wait, claim, hand on.

WHY: people must be able to drop new videos into Unanalyzed/Single_Animal at
any time -- even while several GPU nodes poll it -- so recordings leave the
recording PCs at once. Two defects stood in the way:

  Half-copied  discover_new_singles registered an mp4 the moment it appeared,
               as work ready to take; a GPU node then copied (and posed)
               whatever part of the file had arrived.
  Twice-posed  every GPU node that saw a single copied it into its own queue
               and posed it: nothing in the folder said "taken".

Now a single is taken in only once it has stopped changing, and a GPU node
claims it by renaming it into .inflight/<host>/ before copying it
(watcher/single_claim.py). The claimed file stays until the video has been
handed on, then goes.

Everything lives under tmp_path with config.Paths pointed at it
(tests/conftest.py fails any test that writes into a real pipeline folder).
Host names are placeholders.
"""
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.watcher.orchestrator as orch
import mousereach.watcher.repose as repose
import mousereach.watcher.transfer as transfer
from mousereach.watcher import single_claim
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import DLCOrchestrator, ProcessingOrchestrator
from mousereach.watcher.state import WatcherStateManager

HOST = "NODE-A"
OTHER = "NODE-B"
POSE = "DLC_resnet101_MPSAOct27shuffle3_100000"
VID = "20250101_CNT0101_P1"
VID2 = "20250101_CNT0102_P1"

LAYOUT = {
    "NAS_ROOT": "share",
    "MULTI_ANIMAL_SOURCE": "share/Unanalyzed/Multi-Animal",
    "SINGLE_ANIMAL_OUTPUT": "share/Unanalyzed/Single_Animal",
    "DLC_STAGING": "share/Processing/Posed",
    "REVIEW_ROOT": "share/Processing/Review",
    "TRIAGE_REVIEW": "share/Processing/Review/triage",
    "DEEP_REVIEW": "share/Processing/Review/deep_review",
    "REPOSE_QUEUE": "share/Processing/Repose_Queue",
    "ANALYZED_OUTPUT": "share/Analyzed",
    "PROCESSING_ROOT": "node",
    "DLC_QUEUE": "node/DLC_Queue",
    "PROCESSING": "node/Processing",
}


# ----------------------------------------------------------------- fixtures

@pytest.fixture
def env(tmp_path, monkeypatch):
    dirs = {}
    for name, sub in LAYOUT.items():
        d = tmp_path / sub
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(orch.Paths, name, d, raising=False)
        dirs[name.lower()] = d
    monkeypatch.setattr(orch, "require_processing_root", lambda: dirs["processing_root"])
    monkeypatch.setattr(repose, "declared_scorer", lambda: POSE)
    return SimpleNamespace(tmp=tmp_path, mp=monkeypatch,
                           door=dirs["single_animal_output"], **dirs)


def _node(env, host):
    """A GPU node with its own database and its own local DLC queue."""
    o = object.__new__(DLCOrchestrator)
    o.db = WatcherDB(db_path=env.tmp / f"{host}.db")
    o.hostname = host
    o.queue = env.tmp / host / "DLC_Queue"
    o.queue.mkdir(parents=True)
    o.staging_dir = env.dlc_staging
    o.config = SimpleNamespace(also_process=False, dlc_gpu_device=0,
                               dlc_config_path=None, max_retries=3,
                               work_priority=None, poll_interval_seconds=30)
    o.coordinator = None
    o._sync_to_connectome = lambda *a, **k: None
    o._backup_local_db = lambda: None
    return o


def _as(env, o):
    """Act as node ``o``: Paths.DLC_QUEUE is global, so point it at o's queue."""
    env.mp.setattr(orch.Paths, "DLC_QUEUE", o.queue)
    return o


def _row(o, vid, state, path):
    o.db.register_video(video_id=vid, source_path=str(path), current_path=str(path))
    if state != "discovered":
        o.db.force_state(vid, state, reason="test setup", current_path=str(path))
    return o.db.get_video(vid)


def _adopt(env, o, vid=VID):
    _as(env, o)
    return o._adopt_single_for_dlc({"id": vid, "data": o.db.get_video(vid)})


def _drop(env, vid=VID, body=b"video"):
    p = env.door / f"{vid}.mp4"
    p.write_bytes(body)
    return p


def _age(path, seconds):
    old = time.time() - seconds
    os.utime(path, (old, old))
    return path


def _state(env, wait, quarantine=None):
    db = WatcherDB(db_path=env.tmp / "intake.db")

    def no_quarantine():
        pytest.fail("nothing may be quarantined in this test")

    config = SimpleNamespace(
        stability_wait_seconds=wait, max_retries=3,
        get_quarantine_dir=(lambda: quarantine) if quarantine else no_quarantine)
    return WatcherStateManager(db, config)


def _archive(stem):
    from mousereach.archive.core import get_archive_destination
    d = Path(get_archive_destination(stem))
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}_processing_manifest.json").write_text("{}", encoding="ascii")
    return d


# ------------------------------------------------- intake waits for a finished file

def test_a_file_still_growing_is_not_taken_in(env):
    sm = _state(env, wait=0)
    f = _drop(env, body=b"part one")

    assert sm.discover_new_singles(env.door) == [], "a first sighting never counts"
    with open(f, "ab") as fh:
        fh.write(b" and part two")
    assert sm.discover_new_singles(env.door) == [], "it grew since the last poll"
    assert sm.db.get_video(VID) is None

    assert sm.discover_new_singles(env.door) == [VID]      # unchanged across a poll
    row = sm.db.get_video(VID)
    assert row["state"] == "validated"
    assert row["current_path"] == str(f)


def test_a_stable_file_is_taken_in_only_after_the_configured_wait(env, monkeypatch):
    clock = SimpleNamespace(t=1_000_000.0)
    monkeypatch.setattr(transfer, "time", SimpleNamespace(time=lambda: clock.t))
    sm = _state(env, wait=60)
    _drop(env)

    assert sm.discover_new_singles(env.door) == []
    clock.t += 59
    assert sm.discover_new_singles(env.door) == []
    assert sm.db.get_video(VID) is None
    clock.t += 2
    assert sm.discover_new_singles(env.door) == [VID]


def test_temporary_hidden_and_claimed_files_are_never_taken_in(env):
    sm = _state(env, wait=0)
    for name in (f"{VID}.mp4.part", f"{VID}.mp4.tmp", f"{VID}.mp4.partial",
                 f"{VID}.part", f"._{VID}.mp4"):
        (env.door / name).write_bytes(b"x")
    claimed = env.door / single_claim.INFLIGHT_DIR / OTHER / f"{VID2}.mp4"
    claimed.parent.mkdir(parents=True)
    claimed.write_bytes(b"video")

    for _ in range(3):
        assert sm.discover_new_singles(env.door) == []
    assert sm.db.get_video(VID) is None
    assert sm.db.get_video(VID2) is None
    assert claimed.read_bytes() == b"video"


def test_a_misnamed_file_is_not_quarantined_while_it_may_still_be_copying(env):
    """Quarantine MOVES the file; moving it mid-copy breaks the copy."""
    sm = _state(env, wait=0, quarantine=env.tmp / "Quarantine")
    bad = env.door / "not_a_valid_name.mp4"
    bad.write_bytes(b"x")

    assert sm.discover_new_singles(env.door) == []
    assert bad.exists(), "first sighting: left where it is"
    sm.discover_new_singles(env.door)
    assert not bad.exists(), "stable: quarantined as before"


def test_a_video_given_back_to_the_folder_is_taken_in_again(env):
    sm = _state(env, wait=0)
    f = _drop(env)
    sm.db.register_video(video_id=VID, source_path=str(f), current_path=str(f))
    sm.db.force_state(VID, "unresolvable", reason="test setup",
                      error_message=f"{single_claim.LEFT_FOLDER_REASON}: {OTHER} took it")
    _drop(env, vid=VID2)
    sm.db.register_video(video_id=VID2, source_path=str(env.door / f"{VID2}.mp4"))
    sm.db.force_state(VID2, "unresolvable", reason="test setup",
                      error_message="some other reason")

    assert sm.discover_new_singles(env.door) == []          # waits like any drop
    assert sm.discover_new_singles(env.door) == [VID]
    assert sm.db.get_video(VID)["state"] == "validated"
    assert sm.db.get_video(VID2)["state"] == "unresolvable", \
        "only a row parked because the file left the folder is re-driven"


# ---------------------------------------------------------------- the claim

def test_only_one_of_two_nodes_can_claim_a_video(env):
    src = _drop(env)

    won = single_claim.claim_single(src, HOST)
    assert won == env.door / single_claim.INFLIGHT_DIR / HOST / f"{VID}.mp4"
    assert won.read_bytes() == b"video"
    assert not src.exists()

    assert single_claim.claim_single(src, OTHER) is None      # never raises
    assert not (env.door / single_claim.INFLIGHT_DIR / OTHER / f"{VID}.mp4").exists()
    assert single_claim.inflight_ids() == {VID: HOST}


def test_a_claim_refuses_outside_files_and_never_overwrites_a_claim(env):
    elsewhere = env.dlc_queue / f"{VID}.mp4"
    elsewhere.write_bytes(b"local")
    assert single_claim.claim_single(elsewhere, HOST) is None
    assert elsewhere.read_bytes() == b"local"

    held = env.door / single_claim.INFLIGHT_DIR / HOST / f"{VID}.mp4"
    held.parent.mkdir(parents=True)
    held.write_bytes(b"earlier drop")
    src = _drop(env, body=b"new drop")
    assert single_claim.claim_single(src, HOST) is None
    assert held.read_bytes() == b"earlier drop"
    assert src.read_bytes() == b"new drop"

    assert single_claim.claim_single(src, "..") is None
    assert src.exists()


def test_a_fresh_claim_of_an_old_file_is_not_handed_back(env):
    """A rename keeps the modified time; a video copied in last week would
    otherwise look like a claim nobody has touched for a day."""
    src = _age(_drop(env), 7 * 24 * 3600)
    claimed = single_claim.claim_single(src, HOST)
    assert single_claim.reclaim_stale() == []
    assert claimed.is_file()


def test_a_claim_nobody_touches_for_a_day_goes_back_to_the_folder(env):
    claimed = single_claim.claim_single(_drop(env), OTHER)
    _age(claimed, single_claim.STALE_S + 60)

    assert single_claim.reclaim_stale() == [env.door / f"{VID}.mp4"]
    assert (env.door / f"{VID}.mp4").read_bytes() == b"video"
    assert not claimed.exists()


def test_a_heartbeat_keeps_a_claim_alive(env):
    claimed = single_claim.claim_single(_drop(env), HOST)
    _age(claimed, single_claim.STALE_S + 60)

    assert single_claim.heartbeat_claims(OTHER) == 0
    assert single_claim.heartbeat_claims(HOST) == 1
    assert single_claim.reclaim_stale() == []
    assert claimed.is_file()


def test_handing_a_claim_back_never_overwrites_a_new_drop(env):
    claimed = single_claim.claim_single(_drop(env), OTHER)
    _age(claimed, single_claim.STALE_S + 60)
    _drop(env, body=b"new drop")

    assert single_claim.reclaim_stale() == []
    assert single_claim.release_single(claimed) is False
    assert claimed.read_bytes() == b"video"
    assert (env.door / f"{VID}.mp4").read_bytes() == b"new drop"


# -------------------------------------------------------- adoption on GPU nodes

def test_two_nodes_one_video_only_one_poses_it(env):
    a, b = _node(env, HOST), _node(env, OTHER)
    src = _drop(env)
    _row(a, VID, "validated", src)
    _row(b, VID, "validated", src)

    assert _adopt(env, a) is True
    assert (a.queue / f"{VID}.mp4").read_bytes() == b"video"
    assert a.db.get_video(VID)["state"] == "dlc_queued"
    claimed = single_claim.claimed_path(VID, HOST)
    assert claimed.is_file(), "kept until the video is handed on"

    assert _adopt(env, b) is False
    assert not (b.queue / f"{VID}.mp4").exists()
    after = b.db.get_video(VID)
    assert after["state"] == "unresolvable"
    assert after["error_message"].startswith(single_claim.LEFT_FOLDER_REASON)
    assert HOST in after["error_message"], "the node that has it is named"
    assert claimed.is_file(), "the loser never touches the winner's claim"


def test_a_node_that_loses_the_race_records_nothing_and_says_so_once(env, monkeypatch, caplog):
    """Both nodes saw the file; the other node's rename landed first."""
    a, b = _node(env, HOST), _node(env, OTHER)
    src = _drop(env)
    _row(b, VID, "validated", src)
    single_claim.claim_single(src, HOST)                     # node A won
    monkeypatch.setattr(orch, "locate_video_file", lambda *a, **k: src)   # B's stale view

    with caplog.at_level(logging.INFO, logger=orch.logger.name):
        assert _adopt(env, b) is False
        assert _adopt(env, b) is False
    assert b.db.get_video(VID)["state"] == "validated"
    assert not list(b.queue.iterdir())
    lost = [r for r in caplog.records if "not taken from the shared singles folder" in r.getMessage()]
    assert len(lost) == 1


def test_a_failed_copy_puts_the_video_back_for_any_node(env, monkeypatch):
    a = _node(env, HOST)
    src = _drop(env)
    _row(a, VID, "validated", src)
    monkeypatch.setattr(orch, "safe_copy", lambda s, d, verify=True: False)

    assert _adopt(env, a) is False
    assert src.read_bytes() == b"video"
    assert not single_claim.claimed_path(VID, HOST).exists()
    after = a.db.get_video(VID)
    assert after["state"] == "failed"
    assert "put back" in (after["error_message"] or "")


@pytest.mark.parametrize("where, state", [("archive", "archived"), ("triage", "triage")])
def test_a_video_that_needs_no_pose_is_never_claimed(env, where, state):
    a = _node(env, HOST)
    src = _drop(env)
    _row(a, VID, "validated", src)
    if where == "archive":
        _archive(VID)
    else:
        (env.triage_review / VID).mkdir()

    assert _adopt(env, a) is True
    assert a.db.get_video(VID)["state"] == state
    assert src.read_bytes() == b"video", "finished/held check comes before any claim"
    assert not (env.door / single_claim.INFLIGHT_DIR).exists()


def test_a_node_stopped_right_after_claiming_carries_on_from_its_claim(env):
    a = _node(env, HOST)
    src = _drop(env)
    _row(a, VID, "validated", src)
    claimed = single_claim.claim_single(src, HOST)           # then the watcher stopped

    assert _adopt(env, a) is True
    assert (a.queue / f"{VID}.mp4").read_bytes() == b"video"
    assert a.db.get_video(VID)["state"] == "dlc_queued"
    assert claimed.is_file()


# ------------------------------------------- the claimed copy goes only once handed on

def test_the_claimed_copy_is_removed_only_after_the_video_is_staged(env):
    a = _node(env, HOST)
    _row(a, VID, "validated", _drop(env))
    assert _adopt(env, a) is True
    claimed = single_claim.claimed_path(VID, HOST)
    pose = a.queue / f"{VID}{POSE}.h5"
    pose.write_bytes(b"pose")
    a.db.force_state(VID, "dlc_complete", reason="test: posed", dlc_output_path=str(pose))

    def broken(video_id, files):
        raise IOError("share went away")

    a._stage_files = broken
    with pytest.raises(IOError):
        a._stage_to_nas({"id": VID, "data": a.db.get_video(VID)})
    assert claimed.is_file(), "a stage that failed keeps the claimed copy"
    del a._stage_files

    a.db.force_state(VID, "dlc_complete", reason="test: retry")
    assert a._stage_to_nas({"id": VID, "data": a.db.get_video(VID)}) is True
    assert (env.dlc_staging / f"{VID}.mp4").read_bytes() == b"video"
    assert not claimed.exists(), "staged and confirmed: the duplicate goes"


def test_a_claimed_copy_is_kept_when_the_next_copy_cannot_be_confirmed(env):
    a = _node(env, HOST)
    claimed = single_claim.claim_single(_drop(env), HOST)
    short = env.dlc_staging / f"{VID}.mp4"
    short.write_bytes(b"vid")                                 # not the same size

    assert a._retire_claimed_single(VID, short) is False
    assert a._retire_claimed_single(VID, None) is False
    assert claimed.read_bytes() == b"video"


def test_a_claimed_copy_goes_once_a_review_bundle_holds_the_video(env):
    a = _node(env, HOST)
    claimed = single_claim.claim_single(_drop(env), HOST)
    bundle = env.deep_review / VID
    bundle.mkdir()
    (bundle / f"{VID}.mp4").write_bytes(b"video")

    assert a._retire_claim_after_hold(VID) is True
    assert not claimed.exists()


# ------------------------------------------------------------ keeping claims alive

def _fresh(path):
    return path.is_file() and time.time() - path.stat().st_mtime < 600


def test_claims_are_kept_alive_while_paused(env, monkeypatch):
    """A long recording day skips the scan; a day without a heartbeat would
    hand a video this node already copied to another node."""
    monkeypatch.setattr(repose, "heartbeat", lambda *a, **k: 0)
    a = _node(env, HOST)
    mine = _age(single_claim.claim_single(_drop(env), HOST), single_claim.STALE_S + 60)
    _row(a, VID, "dlc_queued", mine)                 # still being worked here
    theirs = _age(single_claim.claim_single(_drop(env, vid=VID2), OTHER),
                  single_claim.STALE_S + 60)

    a._while_paused()
    assert _fresh(mine)
    assert theirs.is_file(), "a paused node takes no work, so it does not sweep"


def test_the_scan_refreshes_its_own_claims_before_sweeping_stale_ones(env):
    a = _node(env, HOST)
    mine = _age(single_claim.claim_single(_drop(env), HOST), single_claim.STALE_S + 60)
    _row(a, VID, "dlc_running", mine)                # still being worked here
    theirs = _age(single_claim.claim_single(_drop(env, vid=VID2), OTHER),
                  single_claim.STALE_S + 60)

    a._single_claim_upkeep(force=True)
    assert _fresh(mine), "never counts its own claims stale after a long stop"
    assert not theirs.exists()
    assert (env.door / f"{VID2}.mp4").is_file(), "a dead node's claim is handed back"

    # The sweep is rate-limited to every few minutes.
    again = _age(single_claim.claim_single(env.door / f"{VID2}.mp4", OTHER),
                 single_claim.STALE_S + 60)
    a._single_claim_upkeep(force=True)
    assert again.is_file()


def test_the_rescue_never_moves_a_video_a_node_has_claimed(env):
    server = object.__new__(ProcessingOrchestrator)
    server.db = WatcherDB(db_path=env.tmp / "server.db")
    server.staging_dir = env.dlc_staging
    single_claim.claim_single(_drop(env), OTHER)
    settle = ProcessingOrchestrator._MISFILED_SETTLE_S + 600
    stray = env.dlc_staging / f"{VID}.mp4"
    stray.write_bytes(b"second copy")
    _age(stray, settle)
    loose = env.dlc_staging / f"{VID2}.mp4"
    loose.write_bytes(b"video 2")
    _age(loose, settle)

    assert server._rescue_misfiled_singles() == 1
    assert stray.read_bytes() == b"second copy"
    assert not (env.door / f"{VID}.mp4").exists()
    assert (env.door / f"{VID2}.mp4").read_bytes() == b"video 2", \
        "an unclaimed stray still goes to the top of the folder"
    assert single_claim.inflight_ids() == {VID: OTHER}


# ============================================================================
# Review fixes: claims that are given up, refused, duplicated, or handed on.
# Each test below fails if the call it names is removed.
# ============================================================================

VID3 = "20250101_CNT0103_P1"
VID4 = "20250101_CNT0104_P1"


def test_a_rewrite_that_keeps_the_size_is_not_taken_in(env):
    """Some copy programs set the final size first and fill the file in, so
    the modified time is compared as well as the size."""
    sm = _state(env, wait=0)
    f = _drop(env, body=b"0123456789")
    assert sm.discover_new_singles(env.door) == []
    st = f.stat()
    f.write_bytes(b"abcdefghij")                              # same size
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert sm.discover_new_singles(env.door) == [], "written to since the last poll"
    assert sm.discover_new_singles(env.door) == [VID]


class _StopScan(Exception):
    pass


def test_every_scan_refreshes_its_claims_and_sweeps_before_scanning(env):
    """Upkeep runs first in _scan_phase: without it a running node's claims go
    stale after a day and another node poses the video a second time."""
    a = _node(env, HOST)
    mine = _age(single_claim.claim_single(_drop(env), HOST), single_claim.STALE_S + 60)
    _row(a, VID, "dlc_running", mine)
    theirs = _age(single_claim.claim_single(_drop(env, vid=VID2), OTHER),
                  single_claim.STALE_S + 60)

    def scan():
        raise _StopScan()

    a.file_watcher = SimpleNamespace(scan=scan)
    with pytest.raises(_StopScan):
        a._scan_phase()
    assert _fresh(mine)
    assert not theirs.exists() and (env.door / f"{VID2}.mp4").is_file()


def test_a_claim_this_node_gave_up_on_is_not_kept_alive(env, caplog):
    """A failed or unresolvable row must not hold the only shared copy from
    every other node for as long as this node runs."""
    a = _node(env, HOST)
    working = _age(single_claim.claim_single(_drop(env), HOST), single_claim.STALE_S + 60)
    _row(a, VID, "dlc_running", working)
    given_up = _age(single_claim.claim_single(_drop(env, vid=VID2), HOST),
                    single_claim.STALE_S + 60)
    _row(a, VID2, "failed", given_up)
    no_row = _age(single_claim.claim_single(_drop(env, vid=VID3), HOST),
                  single_claim.STALE_S + 60)

    with caplog.at_level(logging.WARNING, logger=orch.logger.name):
        a._single_claim_upkeep(force=True, sweep=False)
        a._single_claim_upkeep(force=True, sweep=False)
    assert _fresh(working)
    assert not _fresh(given_up) and not _fresh(no_row)
    said = [r for r in caplog.records if "no longer refreshed" in r.getMessage()]
    assert len(said) == 2, "each abandoned claim is named once"
    assert sorted(p.name for p in single_claim.reclaim_stale()) == \
        [f"{VID2}.mp4", f"{VID3}.mp4"]


def test_a_database_error_never_hands_back_claims(env):
    a = _node(env, HOST)
    mine = _age(single_claim.claim_single(_drop(env), HOST), single_claim.STALE_S + 60)

    def broken(vid):
        raise RuntimeError("database locked")

    a.db.get_video = broken
    a._single_claim_upkeep(force=True, sweep=False)
    assert _fresh(mine)


def test_a_failed_pose_gives_the_claim_back_and_repeated_failures_stop_intake(env, monkeypatch, caplog):
    import mousereach.dlc.core as dlc_core
    import mousereach.watcher.work_priority as wp
    a = _node(env, HOST)
    cfg = env.tmp / "config.yaml"
    cfg.write_text("model", encoding="ascii")
    a.config.dlc_config_path = cfg
    a._get_priority_animal = lambda: None
    monkeypatch.setattr(dlc_core, "resolve_dlc_shuffle", lambda: (1, "test"))
    monkeypatch.setattr(dlc_core, "run_dlc_batch",
                        lambda **k: [{"status": "error", "error": "CUDA driver"}])
    monkeypatch.setattr(wp, "read_lab_priority", lambda *x, **k: None)
    wp.invalidate_lab_policy()

    with caplog.at_level(logging.WARNING, logger=orch.logger.name):
        for vid in (VID, VID2, VID3):
            _row(a, vid, "validated", _drop(env, vid=vid))
            assert _adopt(env, a, vid) is True
            with pytest.raises(RuntimeError):
                a._process_single_dlc({"id": vid, "data": a.db.get_video(vid)})
            assert a.db.get_video(vid)["state"] == "failed"
            assert (env.door / f"{vid}.mp4").read_bytes() == b"video", \
                "given back at once for a healthy node"
            assert not single_claim.claimed_path(vid, HOST).exists()

    _row(a, VID4, "validated", _drop(env, vid=VID4))
    assert a._singles_braked()
    work = a._select_work_item()
    assert work is None or work["type"] != "adopt_single", \
        "a node whose poses keep failing takes no new singles"
    assert any("pose runs in a row have failed" in r.getMessage() for r in caplog.records)
    a._note_pose_result(True)
    assert a._pose_failure_streak == 0


def test_a_claim_refused_while_the_file_is_open_parks_the_row_and_warns_later(env, monkeypatch, caplog):
    """A media player holding the file open must not starve collage cropping,
    and must be reported once it lasts."""
    import mousereach.watcher.work_priority as wp
    a = _node(env, HOST)
    a._get_priority_animal = lambda: None
    monkeypatch.setattr(wp, "read_lab_priority", lambda *x, **k: None)
    wp.invalidate_lab_policy()
    src = _drop(env)
    _row(a, VID, "validated", src)
    monkeypatch.setattr(single_claim, "claim_single", lambda s, h: None)   # refused
    clock = SimpleNamespace(t=1000.0)
    monkeypatch.setattr(orch, "time", SimpleNamespace(
        monotonic=lambda: clock.t, time=time.time, sleep=time.sleep))

    with caplog.at_level(logging.INFO, logger=orch.logger.name):
        assert a._select_work_item()["type"] == "adopt_single"
        assert _adopt(env, a) is False
        assert a._select_work_item() is None, "parked: the loop moves on to other work"
        clock.t += a._SINGLE_REFUSED_BACKOFF_S + 1
        assert a._select_work_item()["id"] == VID
        clock.t += a._SINGLE_REFUSED_WARN_S
        assert _adopt(env, a) is False
        clock.t += a._SINGLE_REFUSED_BACKOFF_S + 1
        assert _adopt(env, a) is False
    assert a.db.get_video(VID)["state"] == "validated"
    assert src.read_bytes() == b"video"
    warns = [r for r in caplog.records
             if r.levelno == logging.WARNING and "refused to be taken" in r.getMessage()]
    assert len(warns) == 1


def test_a_second_copy_of_a_video_another_node_holds_is_never_claimed(env):
    """The batch copied in again while the first copy is out being posed."""
    a, b = _node(env, HOST), _node(env, OTHER)
    _row(a, VID, "validated", _drop(env))
    assert _adopt(env, a) is True                       # NODE-A holds it
    second = _drop(env)                                 # the same name, again
    _row(b, VID, "validated", second)

    assert _adopt(env, b) is False
    row = b.db.get_video(VID)
    assert row["state"] == "unresolvable"
    assert row["error_message"].startswith(single_claim.DUPLICATE_OF_CLAIM_REASON)
    assert second.read_bytes() == b"video", "left for a person"
    assert not single_claim.claimed_path(VID, OTHER).exists()

    # Nor does the intake scan re-drive a 'left the folder' row while the
    # claim is still out: the same-named file is a second copy, not a return.
    sm = _state(env, wait=0)
    sm.db.register_video(video_id=VID, source_path=str(second), current_path=str(second))
    sm.db.force_state(VID, "unresolvable", reason="test setup",
                      error_message=f"{single_claim.LEFT_FOLDER_REASON}: {HOST} took it")
    for _ in range(3):
        assert sm.discover_new_singles(env.door) == []
    assert sm.db.get_video(VID)["state"] == "unresolvable"


def test_releasing_a_claim_another_node_already_returned_is_quiet(env, caplog):
    claimed = single_claim.claim_single(_drop(env), OTHER)
    assert single_claim.release_single(claimed) is True
    with caplog.at_level(logging.DEBUG, logger=single_claim.logger.name):
        assert single_claim.release_single(claimed) is False
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_stale_claim_blocked_by_a_new_drop_is_reported_once_a_day(env, monkeypatch, caplog):
    monkeypatch.setattr(single_claim, "_unreleasable_warned", {})
    claimed = _age(single_claim.claim_single(_drop(env), OTHER), single_claim.STALE_S + 60)
    _drop(env, body=b"new drop")
    now = time.time()
    with caplog.at_level(logging.WARNING, logger=single_claim.logger.name):
        for k in range(3):
            assert single_claim.reclaim_stale(now=now + k * 300) == []
        assert single_claim.reclaim_stale(now=now + 25 * 3600) == []
    said = [r for r in caplog.records if "cannot go back" in r.getMessage()]
    assert len(said) == 2
    assert claimed.read_bytes() == b"video"


@pytest.mark.parametrize("archived_mp4", [True, False])
def test_a_local_archive_retires_the_claimed_copy_only_against_the_archived_mp4(env, monkeypatch, archived_mp4):
    """The path every also_process GPU node takes."""
    import mousereach.archive.core as core
    a = _node(env, HOST)
    _as(env, a)
    claimed = single_claim.claim_single(_drop(env), HOST)
    _row(a, VID, "processed", env.processing / f"{VID}.mp4")
    dest = env.analyzed_output / "PROJECT_A" / "01"
    dest.mkdir(parents=True)
    if archived_mp4:
        (dest / f"{VID}.mp4").write_bytes(b"video")
    monkeypatch.setattr(core, "archive_video", lambda video_id, **k: {
        "success": True, "destination": str(dest), "files_moved": []})
    a.db.export_to_central_db = lambda *x, **k: None

    assert a._archive_locally_processed({"id": VID, "data": a.db.get_video(VID)}) is True
    assert a.db.get_video(VID)["state"] == "archived"
    assert claimed.exists() is (not archived_mp4)


def test_finishing_an_interrupted_stage_retires_the_claimed_copy(env):
    a = _node(env, HOST)
    _as(env, a)
    claimed = single_claim.claim_single(_drop(env), HOST)
    (env.dlc_staging / f"{VID}.mp4").write_bytes(b"video")    # the mp4 went over
    (a.queue / f"{VID}{POSE}.h5").write_bytes(b"pose")         # the pose did not
    _row(a, VID, "dlc_complete", a.queue / f"{VID}.mp4")

    assert a._stage_to_nas({"id": VID, "data": a.db.get_video(VID)}) is True
    assert (env.dlc_staging / f"{VID}{POSE}.h5").is_file()
    assert not claimed.exists()


@pytest.mark.parametrize("case", ["staged_here", "not_staged_here"])
def test_an_mp4_the_server_already_took_counts_only_if_this_stage_wrote_it(env, case):
    """The processing server can take a staged mp4 within seconds; then the
    stage's own verified copy stands in for the size check -- and only then."""
    a = _node(env, HOST)
    _row(a, VID, "validated", _drop(env))
    assert _adopt(env, a) is True
    claimed = single_claim.claimed_path(VID, HOST)
    pose = a.queue / f"{VID}{POSE}.h5"
    pose.write_bytes(b"pose")
    a.db.force_state(VID, "dlc_complete", reason="test: posed", dlc_output_path=str(pose))
    real = a._stage_files

    def stage_then_server_takes(video_id, files):
        names = real(video_id, files)
        (env.dlc_staging / f"{VID}.mp4").unlink()
        if case == "not_staged_here":
            names = [n for n in names if not n.endswith(".mp4")]
        return names

    a._stage_files = stage_then_server_takes
    assert a._stage_to_nas({"id": VID, "data": a.db.get_video(VID)}) is True
    assert claimed.exists() is (case == "not_staged_here")


def test_both_review_holds_in_the_local_pipeline_retire_the_claimed_copy():
    """Segmentation-failure routing and a non-clean review decision each hand
    the video to a review bundle; each must retire the claimed copy. Driving
    the whole local pipeline here would need every algorithm, so the two call
    sites are held by their source (the helper itself is tested above)."""
    import inspect
    source = inspect.getsource(DLCOrchestrator._run_local_pipeline)
    assert source.count("self._retire_claim_after_hold(video_id)") == 2
