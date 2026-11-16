"""The CPU pose pool keeps the claim of a finished pose it could not hand on,
and tries the hand-on again, waiting longer after each failure.

Such a pose used to wait in the pool's work folder until the next start while
its claim was no longer refreshed (only running poses were): a day later the
shared copy went back to the folder for any node, which posed the video again,
and the pool then handed on its own pose too. tmp_path only (the fixtures of
test_cpu_pose_pool). The tests drive the pool's own loop (run) wherever they
can, so they check what the pool does, not that a helper exists.
"""
import os

import pytest

import mousereach.watcher.cpu_pose_pool as pool_mod
import mousereach.watcher.transfer as transfer
from mousereach.watcher import single_claim

from test_cpu_pose_pool import POSE, _claimed, _local_pose, env  # noqa: F401

VID = "20240101_ABC9901_P1"


class _Finished:
    """A pose process that has just ended well."""
    def poll(self):
        return 0


class _Stop(Exception):
    pass


def _run_cycles(env, monkeypatch, n):
    """Run the pool's main loop for ``n`` cycles (no new poses started)."""
    p = env.pool
    monkeypatch.setattr(p, "recover", lambda: None, raising=False)
    monkeypatch.setattr(p, "_start_one", lambda: False, raising=False)
    monkeypatch.setattr(p, "_status", lambda: None, raising=False)
    monkeypatch.setattr(p, "_stop_requested", lambda: False, raising=False)
    calls = {"n": 0}

    def sleep(_s):
        calls["n"] += 1
        if calls["n"] >= n:
            raise _Stop()

    monkeypatch.setattr(pool_mod.time, "sleep", sleep)
    with pytest.raises(_Stop):
        p.run()


def _finished(env):
    """A pose that has just ended well, its claim in this machine's folder."""
    claimed = _claimed(env, VID)
    _local_pose(env, VID)
    env.pool.active[VID] = {"proc": _Finished(), "local": env.work / f"{VID}.mp4",
                            "started": 0.0}
    return claimed


def test_a_failed_hand_on_is_retried_in_the_running_loop(env, monkeypatch):
    claimed = _finished(env)
    real_copy = transfer.safe_copy
    attempts = {"n": 0}

    def copy_fails_twice(src, dst, verify=True):
        attempts["n"] += 1
        return False if attempts["n"] <= 2 else real_copy(src, dst, verify=verify)

    monkeypatch.setattr(transfer, "safe_copy", copy_fails_twice)
    monkeypatch.setattr(env.pool, "HAND_ON_RETRY_MAX_S", 0, raising=False)  # no wait
    beats = []
    monkeypatch.setattr(env.pool.sc, "heartbeat_claims",
                        lambda holder, only=None: beats.append(list(only or [])))

    _run_cycles(env, monkeypatch, 3)

    names = {p.name for p in env.posed.iterdir()}
    assert f"{VID}.mp4" in names and f"{VID}{POSE}.h5" in names   # handed on
    assert not claimed.exists()
    assert any(VID in b for b in beats), "the claim was not kept alive meanwhile"


def test_the_claim_of_a_pose_not_handed_on_stays_fresh(env, monkeypatch):
    """The share keeps refusing the hand-on: the claim's file is still
    refreshed every cycle (so no other node takes the video back), and the
    hand-on is tried more than once."""
    claimed = _finished(env)
    old = claimed.stat().st_mtime - 2 * 86400
    os.utime(claimed, (old, old))
    tries = {"n": 0}

    def fails(s, d, verify=True):
        tries["n"] += 1
        return False

    monkeypatch.setattr(transfer, "safe_copy", fails)
    monkeypatch.setattr(env.pool, "HAND_ON_RETRY_MAX_S", 0, raising=False)  # no wait

    _run_cycles(env, monkeypatch, 3)

    assert claimed.is_file(), "the claim was given back to the shared folder"
    assert claimed.stat().st_mtime > old + 86400, "the claim was left to go stale"
    assert tries["n"] >= 2, "the hand-on was not tried again"
    assert not any(env.posed.iterdir())
    assert (env.work / f"{VID}{POSE}.h5").is_file()


def _finished_and_unhanded(env, monkeypatch):
    """A finished pose whose hand-on failed once (the share then stays
    refusing copies)."""
    claimed = _finished(env)
    monkeypatch.setattr(transfer, "safe_copy", lambda s, d, verify=True: False)
    env.pool._reap()
    return claimed


def _no_side_folder(env):
    """Leftovers are never moved into a side folder of the work folder."""
    assert not [p for p in env.work.iterdir() if p.is_dir()]


def test_a_pose_whose_claim_went_back_is_deleted_not_handed_on(env, monkeypatch):
    """The claim went back to the shared folder meanwhile: the video is anyone's
    now. Its current copy is confirmed there, so the local copies are machine
    leftovers and are deleted -- not handed on, not moved aside."""
    claimed = _finished_and_unhanded(env, monkeypatch)
    claimed.replace(env.door / f"{VID}.mp4")             # given back to the folder
    _run_cycles(env, monkeypatch, 1)
    assert not any(env.posed.iterdir())
    assert not any(env.work.glob(f"{VID}*"))
    _no_side_folder(env)


def test_a_pose_whose_claim_another_machine_holds_is_deleted(env, monkeypatch):
    claimed = _finished_and_unhanded(env, monkeypatch)
    other = env.door / ".inflight" / "OTHER-NODE"
    other.mkdir(parents=True)
    claimed.replace(other / f"{VID}.mp4")
    _run_cycles(env, monkeypatch, 1)
    assert not any(env.posed.iterdir())
    assert not any(env.work.glob(f"{VID}*"))
    _no_side_folder(env)


def test_a_pose_whose_video_is_nowhere_else_stays_where_it_is(env, monkeypatch):
    """No current copy can be confirmed anywhere: the local video may be the
    only copy. Nothing is handed on, deleted or moved; it is looked at again."""
    claimed = _finished_and_unhanded(env, monkeypatch)
    claimed.unlink()                                     # gone, and nowhere else
    _run_cycles(env, monkeypatch, 2)
    assert not any(env.posed.iterdir())
    assert (env.work / f"{VID}.mp4").is_file()
    assert (env.work / f"{VID}{POSE}.h5").is_file()
    _no_side_folder(env)
    # Once a current copy turns up elsewhere, the leftovers go.
    (env.door / f"{VID}.mp4").write_bytes(b"video-bytes")
    _run_cycles(env, monkeypatch, 1)
    assert not any(env.work.glob(f"{VID}*"))


def test_a_bare_posed_video_is_not_a_current_copy(env, monkeypatch):
    """A video in the posed folder with no pose beside it is a hand-off that
    stopped half way (the server does not take it in): the local video and
    pose are NOT deleted on its account. Once its pose is there too, they
    are."""
    claimed = _finished_and_unhanded(env, monkeypatch)
    claimed.unlink()
    (env.posed / f"{VID}.mp4").write_bytes(b"video-bytes")   # same size, no pose
    _run_cycles(env, monkeypatch, 1)
    assert (env.work / f"{VID}.mp4").is_file()
    assert (env.work / f"{VID}{POSE}.h5").is_file()
    (env.posed / f"{VID}{POSE}.h5").write_bytes(b"pose")
    _run_cycles(env, monkeypatch, 1)
    assert not any(env.work.glob(f"{VID}*"))


def test_a_different_file_of_the_same_name_is_not_a_current_copy(env, monkeypatch):
    """A file of another size is not confirmed as this video's copy."""
    claimed = _finished_and_unhanded(env, monkeypatch)
    claimed.unlink()
    (env.door / f"{VID}.mp4").write_bytes(b"something else entirely")
    _run_cycles(env, monkeypatch, 1)
    assert (env.work / f"{VID}.mp4").is_file()


def test_a_restart_checks_local_copies_left_with_no_claim(env, monkeypatch):
    """A pose left in the work folder by a previous run, its claim taken by
    another machine meanwhile: at the next start it is checked like any
    finished pose whose claim went, not forgotten."""
    _local_pose(env, VID)
    other = env.door / ".inflight" / "OTHER-NODE"
    other.mkdir(parents=True)
    (other / f"{VID}.mp4").write_bytes(b"video-bytes")
    env.pool.recover()
    _run_cycles(env, monkeypatch, 1)
    assert not any(env.posed.iterdir())
    assert not any(env.work.glob(f"{VID}*"))
    assert (other / f"{VID}.mp4").is_file()


def test_a_restart_in_a_share_outage_never_hands_on_a_bare_video(env, monkeypatch, tmp_path):
    """The share cannot be read when the pool starts, and the work folder
    holds a video whose pose never finished. When the share is back, that
    video is NOT handed on as a bare video: nothing reaches the posed folder
    and the claimed shared copy -- the video's real file -- is kept."""
    claimed = _claimed(env, VID)
    (env.work / f"{VID}.mp4").write_bytes(b"video-bytes")     # no pose beside it
    real_door = single_claim.front_door()
    offline = tmp_path / "offline_share" / "Single_Animal"     # does not exist
    share = {"down": True}
    monkeypatch.setattr(single_claim, "front_door",
                        lambda: offline if share["down"] else real_door)

    env.pool.recover()
    share["down"] = False
    _run_cycles(env, monkeypatch, 3)

    assert not any(env.posed.iterdir())
    assert claimed.is_file()


def test_an_unreachable_share_keeps_the_finished_pose(env, monkeypatch, tmp_path):
    """Path.is_file() says False for a share that cannot be reached; that is
    not "the claim is gone". The pose stays, still waiting to be handed on."""
    _finished_and_unhanded(env, monkeypatch)
    offline = tmp_path / "offline_share" / "Single_Animal"     # does not exist
    monkeypatch.setattr(single_claim, "front_door", lambda: offline)

    _run_cycles(env, monkeypatch, 2)

    assert (env.work / f"{VID}{POSE}.h5").is_file()
    assert (env.work / f"{VID}.mp4").is_file()
    _no_side_folder(env)


def test_repeated_hand_on_failures_wait_longer_each_time(env, monkeypatch):
    """Each attempt copies a whole video; a share that stays down is not
    hammered every cycle."""
    _finished(env)
    tries = {"n": 0}

    def fails(s, d, verify=True):
        tries["n"] += 1
        return False

    monkeypatch.setattr(transfer, "safe_copy", fails)
    monkeypatch.setattr(env.pool, "_status", lambda: None, raising=False)
    clock = {"t": 1000.0}
    monkeypatch.setattr(pool_mod.time, "time", lambda: clock["t"])
    env.pool._reap()
    after_reap = tries["n"]
    env.pool._retry_hand_ons()               # first retry: at once
    assert tries["n"] > after_reap
    n1 = tries["n"]
    env.pool._retry_hand_ons()               # same moment: waits
    assert tries["n"] == n1
    clock["t"] += 61
    env.pool._retry_hand_ons()               # a minute later: tried again
    assert tries["n"] > n1
    n2 = tries["n"]
    clock["t"] += 61
    env.pool._retry_hand_ons()               # the wait has doubled
    assert tries["n"] == n2
