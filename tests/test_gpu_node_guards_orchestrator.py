"""GPU-node guards in the DLC orchestrator: never pose finished or held work,
never crop without a claim, never read from the processing server's intake.

WHY: several GPU nodes share the collage intake folder and one processing
server takes posed videos in from Processing/Posed. Before more GPU nodes may
join, three defects had to go.

  Finished work  Cropping and posing never asked whether a video was already
                 finished. A node with a fresh or partial database re-cropped
                 and re-posed finished children (~14 GPU-minutes each) and
                 duplicated everything downstream. A video whose bundle sits
                 in a review queue is held for a person and is not posed
                 again either.
  Claims         Collage claims failed OPEN: a claim error, or a coordinator
                 that never started, cropped anyway; a lost claim counted as
                 progress, so the node re-picked the collage at full speed;
                 a failed crop never released its claim, blocking the collage
                 for every node.
  Posed          GPU handlers resolved files INTO Processing/Posed: copying a
                 hand-off into their own queue, running DLC with its input
                 there (DLC writes the .h5 beside its input), and marking a
                 video archived because a same-named file sat in staging.

Everything lives under tmp_path with config.Paths pointed at it
(tests/conftest.py fails any test that writes into a real pipeline folder).
Handlers run on an orchestrator built without its constructor, except in the
one test that is about the constructor.
"""
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.watcher.orchestrator as orch
import mousereach.watcher.repose as repose
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import DLCOrchestrator, ProcessingOrchestrator

HOST = "NODE-A"
OTHER = "NODE-B"
POSE = "DLC_resnet101_MPSAOct27shuffle3_100000"
OTHER_POSE = "DLC_resnet50_MPSAOct27shuffle1_100000"

# Two animals and six blank (cohort 00) slots: the cropper skips blanks, so
# this collage has exactly two expected children.
COLLAGE = ("20250101_CNT0101,CNT0102,CNT0001,CNT0002,CNT0003,CNT0004,"
           "CNT0005,CNT0006_P1.mkv")
KID1 = "20250101_CNT0101_P1"
KID2 = "20250101_CNT0102_P1"


# ----------------------------------------------------------------- fixtures

class FakeCoordinator:
    """The shared claim table, reduced to what the handler asks of it."""

    def __init__(self, claim=True, holder=OTHER, release_error=None,
                 claim_state="cropping"):
        self.claim = claim              # True, False, or an exception to raise
        self.holder = holder
        self.claim_state = claim_state  # the holder's row state when claim is False
        self.release_error = release_error
        self.update_error = None        # an exception update_collage_state raises
        self.update_result = True       # False = the claim is no longer this host's
        self.claims, self.released, self.updates = [], [], []

    def try_claim_collage(self, filename, hostname):
        self.claims.append((filename, hostname))
        if isinstance(self.claim, BaseException):
            raise self.claim
        return self.claim

    def release_collage_claim(self, filename, hostname):
        self.released.append((filename, hostname))
        if self.release_error is not None:
            raise self.release_error
        return True

    def update_collage_state(self, filename, state, **kw):
        if self.update_error is not None:
            raise self.update_error
        self.updates.append((filename, state, kw))
        return self.update_result

    def get_collage_claim(self, filename):
        return {"filename": filename, "hostname": self.holder,
                "state": self.claim_state, "claimed_at": "2025-01-01T08:00:00"}


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

    o = object.__new__(DLCOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "watcher.db")
    o.hostname = HOST
    o.working_dir = dirs["processing_root"] / "watcher_working"
    o.working_dir.mkdir()
    o.staging_dir = dirs["dlc_staging"]
    o.config = SimpleNamespace(also_process=False, dlc_gpu_device=0,
                               dlc_config_path=None, max_retries=3,
                               work_priority=None)
    o.coordinator = FakeCoordinator()
    o._sync_to_connectome = lambda *a, **k: None
    o._backup_local_db = lambda: None
    return SimpleNamespace(o=o, db=o.db, tmp=tmp_path, **dirs)


def _stable_collage(env):
    src = env.multi_animal_source / COLLAGE
    src.write_bytes(b"collage")
    env.db.register_collage(COLLAGE, str(src))
    for state in ("validated", "stable"):
        env.db.update_collage_state(COLLAGE, state)
    return {"type": "collage", "id": COLLAGE, "data": env.db.get_collage(COLLAGE)}


def _fake_crop(monkeypatch):
    """Stand in for ffmpeg: write one small file per non-blank slot."""
    import mousereach.video_prep.core.cropper as cropper
    calls = []

    def crop(input_path, output_dir, verbose=True):
        calls.append(Path(input_path).name)
        info = cropper.parse_collage_filename(Path(input_path).name)
        results = []
        for pos, aid in enumerate(info["animal_ids"], start=1):
            if cropper.is_blank_animal(aid):
                results.append({"position": pos, "animal_id": aid, "status": "skipped"})
                continue
            out = Path(output_dir) / f"{info['date']}_{aid}_{info['last_part']}.mp4"
            out.write_bytes(b"cropped " + aid.encode("ascii"))
            results.append({"position": pos, "animal_id": aid,
                            "output_path": str(out), "status": "success"})
        return results

    monkeypatch.setattr(cropper, "crop_collage", crop)
    return calls


def _archive(stem):
    """File a processing manifest where the archive keeps this video's."""
    from mousereach.archive.core import get_archive_destination
    d = Path(get_archive_destination(stem))
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}_processing_manifest.json").write_text("{}", encoding="ascii")
    return d


def _hold(env, where, stem):
    """Put a bundle folder for this video where a review hold would be."""
    from mousereach.watcher.review_routing import INCOMING_DIR_NAME
    folder = {"triage": env.triage_review / stem,
              "deep_review": env.deep_review / stem,
              "deep_incoming": env.deep_review / INCOMING_DIR_NAME / stem}[where]
    folder.mkdir(parents=True)
    return folder


def _row(env, vid, state, path):
    env.db.register_video(video_id=vid, source_path=str(path), current_path=str(path))
    if state != "discovered":
        env.db.force_state(vid, state, reason="test setup", current_path=str(path))
    return env.db.get_video(vid)


def _log(env, name, step):
    conn = env.db._get_connection()
    try:
        return [(r[0], r[1]) for r in conn.execute(
            "SELECT status, message FROM processing_log WHERE video_id=? AND step=? "
            "ORDER BY id", (name, step))]
    finally:
        conn.close()


def _dlc_ready(env, monkeypatch):
    """A DLC model configured, with inference recorded instead of run."""
    import mousereach.dlc.core as dlc_core
    cfg = env.tmp / "config.yaml"
    cfg.write_text("model", encoding="ascii")
    env.o.config.dlc_config_path = cfg
    monkeypatch.setattr(dlc_core, "resolve_dlc_shuffle", lambda: (1, "test"))
    ran = []

    def fake_dlc(video_paths, config_path, output_dir, gpu, save_as_csv, shuffle):
        ran.append(Path(video_paths[0]))
        stem = Path(video_paths[0]).stem
        (Path(output_dir) / f"{stem}{POSE}.h5").write_bytes(b"pose")
        return [{"status": "success"}]

    monkeypatch.setattr(dlc_core, "run_dlc_batch", fake_dlc)
    return ran


# ------------------------------------------ finished or held: never re-posed

def test_a_finished_child_is_recorded_and_never_queued_for_dlc(env, monkeypatch):
    calls = _fake_crop(monkeypatch)
    synced = []
    env.o._sync_to_connectome = lambda vid, state, **kw: synced.append((vid, state, kw))
    _archive(KID1)
    work = _stable_collage(env)

    assert env.o._process_collage(work) is True

    assert calls == [COLLAGE]                   # KID2 still needed the crop
    assert env.db.get_video(KID1)["state"] == "archived"
    assert not (env.dlc_queue / f"{KID1}.mp4").exists()
    assert env.db.get_video(KID2)["state"] == "dlc_queued"
    assert (env.dlc_queue / f"{KID2}.mp4").exists()
    collage = env.db.get_collage(COLLAGE)
    assert collage["state"] == "cropped"
    assert (collage["videos_created"], collage["videos_skipped"]) == (1, 7)
    # 'cropped' only while this node still holds the claim
    assert env.o.coordinator.updates == [
        (COLLAGE, "cropped", {"only_if_held_by": HOST, "singles_created": 1})]
    # the queued child is on record in the shared table from the moment it is
    # queued, so a stale-claim takeover can see that this collage has children
    assert synced == [(KID2, "dlc_queued", {"collage_id": COLLAGE,
                                            "source_path": str(env.dlc_queue / f"{KID2}.mp4")})]


@pytest.mark.parametrize("where, state", [("triage", "triage"),
                                          ("deep_incoming", "deep_review")])
def test_a_child_held_for_review_is_recorded_in_its_queue_state(env, monkeypatch,
                                                                where, state):
    _fake_crop(monkeypatch)
    _hold(env, where, KID1)

    assert env.o._process_collage(_stable_collage(env)) is True

    assert env.db.get_video(KID1)["state"] == state
    assert not (env.dlc_queue / f"{KID1}.mp4").exists()
    assert env.db.get_video(KID2)["state"] == "dlc_queued"


def test_a_child_already_in_flight_here_is_left_alone_and_not_queued_again(env, monkeypatch):
    _fake_crop(monkeypatch)
    _archive(KID1)
    queued = env.dlc_queue / f"{KID1}.mp4"
    queued.write_bytes(b"already here")
    _row(env, KID1, "dlc_running", queued)

    assert env.o._process_collage(_stable_collage(env)) is True

    assert env.db.get_video(KID1)["state"] == "dlc_running"
    assert queued.read_bytes() == b"already here"


def test_a_collage_whose_children_are_all_finished_or_held_is_not_cropped(env, monkeypatch):
    calls = _fake_crop(monkeypatch)
    _archive(KID1)
    _hold(env, "deep_review", KID2)

    assert env.o._process_collage(_stable_collage(env)) is True

    assert calls == []                          # no crop spent on it
    assert not list(env.o.working_dir.iterdir())  # not even copied over
    assert not list(env.dlc_queue.iterdir())
    assert env.db.get_video(KID1)["state"] == "archived"
    assert env.db.get_video(KID2)["state"] == "deep_review"
    assert env.db.get_collage(COLLAGE)["state"] == "cropped"
    (status, message), = _log(env, COLLAGE, "crop")
    assert status == "skipped" and "already analysed or held" in message
    assert env.o.coordinator.updates == [
        (COLLAGE, "cropped", {"only_if_held_by": HOST, "singles_created": 0})]


@pytest.mark.parametrize("where, state", [("archive", "archived"),
                                          ("triage", "triage"),
                                          ("deep_incoming", "deep_review")])
def test_dlc_refuses_a_video_that_is_finished_or_held(env, monkeypatch, where, state):
    ran = _dlc_ready(env, monkeypatch)
    mp4 = env.dlc_queue / f"{KID1}.mp4"
    mp4.write_bytes(b"video")
    row = _row(env, KID1, "dlc_queued", mp4)
    if where == "archive":
        _archive(KID1)
    else:
        _hold(env, where, KID1)

    assert env.o._process_single_dlc({"id": KID1, "data": row}) is False

    assert ran == []
    assert env.db.get_video(KID1)["state"] == state
    assert mp4.exists(), "never delete: the local copy is left for later cleanup"
    assert not list(env.dlc_queue.glob("*.h5"))


def test_a_requested_re_pose_of_an_archived_video_is_still_posed(env, monkeypatch):
    """Re-posing finished videos is the whole point of a re-pose request; the
    finished-work guard must not swallow them."""
    ran = _dlc_ready(env, monkeypatch)
    mp4 = env.dlc_queue / f"{KID1}.mp4"
    mp4.write_bytes(b"video")
    _row(env, KID1, "dlc_queued", mp4)
    env.db.set_fields(KID1, mark_reason=f"{repose.REASON_PREFIX} from NODE-S")
    _archive(KID1)

    assert env.o._process_single_dlc({"id": KID1, "data": env.db.get_video(KID1)}) is True

    assert ran == [mp4]
    assert env.db.get_video(KID1)["state"] == "dlc_complete"


def test_a_re_pose_request_for_a_video_held_in_triage_is_refused(env, monkeypatch):
    """repose.consume_requests refuses a triage row; the pose step agrees."""
    ran = _dlc_ready(env, monkeypatch)
    mp4 = env.dlc_queue / f"{KID1}.mp4"
    mp4.write_bytes(b"video")
    _row(env, KID1, "dlc_queued", mp4)
    env.db.set_fields(KID1, mark_reason=f"{repose.REASON_PREFIX} from NODE-S")
    _hold(env, "triage", KID1)

    assert env.o._process_single_dlc({"id": KID1, "data": env.db.get_video(KID1)}) is False
    assert ran == []
    assert env.db.get_video(KID1)["state"] == "triage"


def test_adopting_a_video_held_for_review_records_the_hold(env):
    src = env.single_animal_output / f"{KID1}.mp4"
    src.write_bytes(b"video")
    row = _row(env, KID1, "validated", src)
    _hold(env, "triage", KID1)

    assert env.o._adopt_single_for_dlc({"id": KID1, "data": row}) is True
    assert env.db.get_video(KID1)["state"] == "triage"
    assert not list(env.dlc_queue.iterdir())


# ------------------------------------------------------ claims fail closed

def _no_lab_policy(monkeypatch):
    """_select_work_item reads the lab's ordering policy; keep it the default."""
    import mousereach.watcher.work_priority as wp
    monkeypatch.setattr(wp, "read_lab_priority", lambda *a, **k: None)
    wp.invalidate_lab_policy()


def test_a_claim_held_elsewhere_is_skipped_for_a_while_then_asked_about_again(
        env, monkeypatch):
    """A live claim elsewhere is not parked for good: that claim can still be
    released after a failed crop, or go stale, and only a node that asks
    again can then crop the collage."""
    _no_lab_policy(monkeypatch)
    calls = _fake_crop(monkeypatch)
    coord = env.o.coordinator = FakeCoordinator(claim=False, holder=OTHER)
    work = _stable_collage(env)

    assert env.o._process_collage(work) is False

    assert calls == []
    assert env.db.get_collage(COLLAGE)["state"] == "stable"   # revisited later
    assert env.o._select_work_item() is None                  # ...but not next poll
    (status, message), = _log(env, COLLAGE, "claim")
    assert status == "skipped" and OTHER in message
    # the dispatch loop reads it as no progress, so it sleeps instead of spinning
    assert env.o._dispatch_work(work) is False

    # The recheck time passes and the holder has released it: this node crops it.
    env.o._claim_backoff[COLLAGE] = 0
    coord.claim = True
    picked = env.o._select_work_item()
    assert picked is not None and picked["id"] == COLLAGE
    assert env.o._process_collage(picked) is True
    assert calls == [COLLAGE]


def test_a_collage_already_cropped_elsewhere_is_recorded_as_cropped(env, monkeypatch):
    calls = _fake_crop(monkeypatch)
    env.o.coordinator = FakeCoordinator(claim=False, holder=OTHER, claim_state="cropped")

    assert env.o._process_collage(_stable_collage(env)) is False

    assert calls == []
    assert env.db.get_collage(COLLAGE)["state"] == "cropped"
    assert env.db.get_collages_in_state("stable") == []
    (status, message), = _log(env, COLLAGE, "claim")
    assert status == "skipped" and OTHER in message


def test_a_claim_error_warns_again_once_the_check_has_worked_in_between(
        env, monkeypatch, caplog):
    _fake_crop(monkeypatch)
    coord = env.o.coordinator = FakeCoordinator(claim=OSError("share unreachable"))
    work = _stable_collage(env)

    with caplog.at_level(logging.WARNING, logger=orch.logger.name):
        assert env.o._process_collage(work) is False     # warns
        assert env.o._process_collage(work) is False     # same outage: quiet
        coord.claim = False                              # the share answers again
        assert env.o._process_collage(work) is False
        env.o._claim_backoff.clear()
        coord.claim = OSError("database is locked")      # a new outage
        assert env.o._process_collage(work) is False     # warns again

    warned = [r for r in caplog.records if r.levelno == logging.WARNING
              and "could not check the shared claim" in r.getMessage()]
    assert len(warned) == 2


def test_a_claim_error_does_not_crop_and_warns_once(env, monkeypatch, caplog):
    calls = _fake_crop(monkeypatch)
    env.o.coordinator = FakeCoordinator(claim=OSError("share unreachable"))
    work = _stable_collage(env)

    with caplog.at_level(logging.WARNING, logger=orch.logger.name):
        assert env.o._process_collage(work) is False
        assert env.o._dispatch_work(work) is False

    assert calls == []
    assert env.db.get_collage(COLLAGE)["state"] == "stable"   # tried again next poll
    warned = [r for r in caplog.records
              if r.levelno == logging.WARNING and COLLAGE in r.getMessage()]
    assert len(warned) == 1


def test_no_coordinator_means_no_crop(env, monkeypatch, caplog):
    calls = _fake_crop(monkeypatch)
    env.o.coordinator = None
    work = _stable_collage(env)

    with caplog.at_level(logging.WARNING, logger=orch.logger.name):
        assert env.o._process_collage(work) is False
        assert env.o._process_collage(work) is False

    assert calls == []
    assert env.db.get_collage(COLLAGE)["state"] == "stable"
    warned = [r for r in caplog.records if "Collage claims are unavailable" in r.getMessage()]
    assert len(warned) == 1


def test_a_coordinator_that_fails_to_start_is_recorded_and_crops_nothing(
        env, monkeypatch, caplog):
    """The real constructor: a failed start must leave a recorded reason, not
    just a None that the collage handler used to read as 'crop anyway'."""
    import mousereach.watcher.coordination as coordination
    import mousereach.watcher.work_priority as wp
    from mousereach.config import WatcherConfig

    class Unreachable:
        def __init__(self, *a, **k):
            pass

        def ensure_tables(self):
            raise OSError("unable to open database file")

    monkeypatch.setattr(coordination, "PipelineCoordinator", Unreachable)
    monkeypatch.setattr(wp, "read_lab_priority", lambda *a, **k: None)
    wp.invalidate_lab_policy()
    calls = _fake_crop(monkeypatch)
    config = WatcherConfig({
        "enabled": True, "poll_interval_seconds": 30, "stability_wait_seconds": 60,
        "dlc_config_path": None, "dlc_gpu_device": 0, "max_retries": 3,
        "quarantine_dir": str(env.tmp / "quarantine"), "log_dir": str(env.tmp / "logs"),
    })

    node = DLCOrchestrator(config, env.db)
    assert node.coordinator is None
    assert "unable to open database file" in node._coordinator_init_error

    work = _stable_collage(env)
    with caplog.at_level(logging.WARNING, logger=orch.logger.name):
        assert node._process_collage(work) is False
        assert node._select_work_item() is None     # the collage is not even offered
    assert calls == []
    assert env.db.get_collage(COLLAGE)["state"] == "stable"

    # The share comes back. The node tries again on its own -- no restart --
    # but only once the retry interval has passed.
    class Reachable:
        def __init__(self, *a, **k):
            pass

        def ensure_tables(self):
            pass

    monkeypatch.setattr(coordination, "PipelineCoordinator", Reachable)
    assert node._select_work_item() is None         # too soon to try again
    node._coordinator_last_attempt = time.monotonic() - node._COORDINATOR_RETRY_S - 1
    with caplog.at_level(logging.INFO, logger=orch.logger.name):
        picked = node._select_work_item()
    assert picked is not None and picked["type"] == "collage"
    assert isinstance(node.coordinator, Reachable)
    assert node._coordinator_init_error is None
    assert any("available again" in r.getMessage() for r in caplog.records)


def test_a_failed_crop_releases_the_claim(env, monkeypatch):
    import mousereach.video_prep.core.cropper as cropper

    def broken(input_path, output_dir, verbose=True):
        raise RuntimeError("ffmpeg not found")

    monkeypatch.setattr(cropper, "crop_collage", broken)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        env.o._process_collage(_stable_collage(env))

    assert env.o.coordinator.released == [(COLLAGE, HOST)]
    assert env.db.get_collage(COLLAGE)["state"] == "failed"


def test_a_release_that_fails_does_not_hide_the_crop_error(env, monkeypatch):
    import mousereach.video_prep.core.cropper as cropper
    monkeypatch.setattr(cropper, "crop_collage",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ffmpeg not found")))
    env.o.coordinator = FakeCoordinator(release_error=OSError("share unreachable"))

    with pytest.raises(RuntimeError, match="ffmpeg"):
        env.o._process_collage(_stable_collage(env))
    assert env.o.coordinator.released == [(COLLAGE, HOST)]


def test_a_crop_that_fails_after_queuing_a_child_keeps_the_claim(env, monkeypatch):
    """Releasing here would let another node crop the collage again and pose
    the child already in this node's queue a second time."""
    _fake_crop(monkeypatch)
    real_copy = orch.safe_copy

    def copy(src, dst, verify=True):
        if Path(dst).parent == env.dlc_queue and Path(dst).name.startswith(KID2):
            raise OSError("disk full")
        return real_copy(src, dst, verify=verify)

    monkeypatch.setattr(orch, "safe_copy", copy)

    with pytest.raises(OSError, match="disk full"):
        env.o._process_collage(_stable_collage(env))

    assert env.o.coordinator.released == []
    assert env.db.get_video(KID1)["state"] == "dlc_queued"
    assert env.db.get_collage(COLLAGE)["state"] == "failed"


def test_a_failed_crop_keeps_the_claim_when_an_earlier_attempt_left_a_child_here(
        env, monkeypatch):
    import mousereach.video_prep.core.cropper as cropper
    monkeypatch.setattr(cropper, "crop_collage",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ffmpeg not found")))
    queued = env.dlc_queue / f"{KID1}.mp4"
    queued.write_bytes(b"cropped on an earlier attempt")
    env.db.register_video(video_id=KID1, source_path=str(queued),
                          current_path=str(queued), collage_id=COLLAGE)
    env.db.force_state(KID1, "dlc_queued", reason="test setup", current_path=str(queued))

    with pytest.raises(RuntimeError, match="ffmpeg"):
        env.o._process_collage(_stable_collage(env))

    assert env.o.coordinator.released == []


def test_a_cropped_that_does_not_reach_the_shared_table_is_sent_again(
        env, monkeypatch, caplog):
    """A claim left in 'cropping' can be taken over after a day, so a lost
    'cropped' is not a debug line any more: it is retried until it lands."""
    _fake_crop(monkeypatch)
    coord = env.o.coordinator
    coord.update_error = OSError("database is locked")

    with caplog.at_level(logging.WARNING, logger=orch.logger.name):
        assert env.o._process_collage(_stable_collage(env)) is True
    assert any("could not be marked 'cropped'" in r.getMessage() for r in caplog.records)

    env.o._retry_unsynced_collage_claims()          # still locked: kept
    assert coord.updates == [] and COLLAGE in env.o._unsynced_collage_claims
    coord.update_error = None
    env.o._retry_unsynced_collage_claims()
    assert coord.updates == [
        (COLLAGE, "cropped", {"only_if_held_by": HOST, "singles_created": 2})]
    assert not env.o._unsynced_collage_claims


def test_a_crop_whose_claim_was_taken_over_meanwhile_says_so(env, monkeypatch, caplog):
    _fake_crop(monkeypatch)
    env.o.coordinator.update_result = False          # the row is another host's now

    with caplog.at_level(logging.WARNING, logger=orch.logger.name):
        assert env.o._process_collage(_stable_collage(env)) is True
    assert any("no longer held by this node" in r.getMessage() for r in caplog.records)


# ------------------------------------------- Processing/Posed is off limits

@pytest.fixture
def posed(env):
    """Same-stem files in the processing server's intake, none on this node."""
    mp4 = env.dlc_staging / f"{KID1}.mp4"
    h5 = env.dlc_staging / f"{KID1}{POSE}.h5"
    mp4.write_bytes(b"server intake video")
    h5.write_bytes(b"server intake pose")
    return SimpleNamespace(mp4=mp4, h5=h5, names=sorted([mp4.name, h5.name]))


def test_adopt_never_takes_a_video_from_posed(env, posed):
    row = _row(env, KID1, "validated", posed.mp4)

    assert env.o._adopt_single_for_dlc({"id": KID1, "data": row}) is False

    assert not list(env.dlc_queue.iterdir())
    assert env.db.get_video(KID1)["state"] == "unresolvable"


def test_dlc_never_runs_on_a_video_in_posed(env, monkeypatch, posed):
    ran = _dlc_ready(env, monkeypatch)
    row = _row(env, KID1, "dlc_queued", posed.mp4)

    assert env.o._process_single_dlc({"id": KID1, "data": row}) is False

    assert ran == []
    assert env.db.get_video(KID1)["state"] == "unresolvable"
    assert sorted(p.name for p in env.dlc_staging.iterdir()) == posed.names


def test_local_pipeline_never_takes_its_pose_or_video_from_posed(env, posed):
    row = _row(env, KID1, "dlc_complete", posed.mp4)
    data = {**row, "dlc_output_path": str(posed.h5)}

    assert env.o._run_local_pipeline({"id": KID1, "data": data}) is False

    assert not list(env.processing.iterdir())
    assert env.db.get_video(KID1)["state"] == "failed"


def test_local_pipeline_never_takes_its_video_from_posed_when_its_pose_is_here(env, posed):
    """The pose lookup passes, so the video lookup is what is tested here."""
    pose = env.dlc_queue / f"{KID1}{POSE}.h5"
    pose.write_bytes(b"this node's pose")
    row = _row(env, KID1, "dlc_complete", posed.mp4)
    data = {**row, "dlc_output_path": str(pose)}

    assert env.o._run_local_pipeline({"id": KID1, "data": data}) is False

    assert not (env.processing / f"{KID1}.mp4").exists()
    assert env.db.get_video(KID1)["state"] == "failed"
    assert sorted(p.name for p in env.dlc_staging.iterdir()) == posed.names


@pytest.mark.parametrize("recorded", ["gone_local", "posed"])
def test_staging_never_counts_a_file_already_in_posed_as_its_own(env, posed, recorded):
    path = posed.mp4 if recorded == "posed" else env.dlc_queue / f"{KID1}.mp4"
    row = _row(env, KID1, "dlc_complete", path)

    assert env.o._stage_to_nas({"id": KID1, "data": row}) is False

    after = env.db.get_video(KID1)
    # never 'archived' on the strength of another node's files, and not
    # 'failed' either: nothing went wrong, this node just has no file for it
    assert after["state"] == "unresolvable"
    assert "already in NAS staging" in (after["error_message"] or "")
    assert sorted(p.name for p in env.dlc_staging.iterdir()) == posed.names


def test_staging_records_its_own_pose_not_one_already_in_posed(env, posed):
    # The pose already in Posed is the one a glob of staging WOULD pick: the
    # glob matches its name, select_pose_file reads the declared scorer from
    # it, and it is newer than this node's.
    foreign = env.dlc_staging / f"{KID1}DLCx{POSE}.h5"
    posed.h5.rename(foreign)
    later = time.time() + 3600
    os.utime(foreign, (later, later))
    mp4 = env.dlc_queue / f"{KID1}.mp4"
    mp4.write_bytes(b"video")
    (env.dlc_queue / f"{KID1}{POSE}.h5").write_bytes(b"this node's pose")
    row = _row(env, KID1, "dlc_complete", mp4)

    assert env.o._stage_to_nas({"id": KID1, "data": row}) is True

    after = env.db.get_video(KID1)
    assert after["state"] == "archived"
    assert Path(after["dlc_output_path"]) == env.dlc_staging / f"{KID1}{POSE}.h5"
    assert foreign.read_bytes() == b"server intake pose"      # untouched


def test_a_gpu_node_publishes_re_pose_requests_without_reading_posed(env, monkeypatch):
    seen = {}
    monkeypatch.setattr(repose, "publish_pending",
                        lambda db, rows, **kw: seen.update(kw) or {})
    env.o._publish_repose_requests()
    assert seen["staging_dir"] is False


def test_no_work_paths_report_no_progress(env):
    mp4 = env.dlc_queue / f"{KID1}.mp4"
    mp4.write_bytes(b"video")
    row = _row(env, KID1, "dlc_queued", mp4)
    import mousereach.dlc.core as dlc_core
    env.o.config.dlc_config_path = None
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dlc_core, "resolve_dlc_shuffle", lambda: (1, "test"))
        assert env.o._process_single_dlc({"id": KID1, "data": row}) is False
    assert env.db.get_video(KID1)["state"] == "dlc_queued"


def test_the_processing_server_still_takes_in_from_posed(env, posed):
    """Posed is the processing server's intake: its reads there are unchanged."""
    server = object.__new__(ProcessingOrchestrator)
    server.db = env.db
    server.processing_dir = env.processing
    server.staging_dir = env.dlc_staging
    server._claim_video = lambda video_id: True
    row = _row(env, KID1, "dlc_complete", posed.mp4)

    server._intake_from_staging({"id": KID1, "data": row})

    assert (env.processing / f"{KID1}.mp4").exists()
    assert (env.processing / f"{KID1}{POSE}.h5").exists()
    assert env.db.get_video(KID1)["state"] == "processing"
