"""A video a GPU node cut while it had no DLC queue is never stranded.

The node publishes each cut child into its claim folder on the share and the
collage is recorded 'cropped'. With no local DLC queue nothing can be posed
there: the child's row was parked 'unresolvable' with a reason nothing ever
took again. If no other node took the video, its session was never posed --
not even after a person configured the queue. Now the claim goes back to the
top of the singles folder at once (any node with a queue may take it), and the
row is taken again here once this node has a queue -- but not before, so it
does not flip on every poll. tmp_path only (the fixtures of
test_collage_crop_failures_are_visible).
"""
from types import SimpleNamespace

import mousereach.watcher.orchestrator as orch
from mousereach.watcher import single_claim
from mousereach.watcher.state import WatcherStateManager

from test_collage_crop_failures_are_visible import (  # noqa: F401
    COLLAGE, HOST, KID1, KID2, _collage, _fake_crop, env)


def _crop_without_a_queue(env, monkeypatch):
    _fake_crop(monkeypatch)
    monkeypatch.setattr(orch.Paths, "DLC_QUEUE", None, raising=False)
    env.o._hand_out_instead_of_keeping = lambda *a, **k: False
    env.o._dispatch_work(_collage(env, "stable"))
    door = env.single_animal_output
    for kid in (KID1, KID2):
        env.o._adopt_single_for_dlc({"id": kid, "data": env.db.get_video(kid)})
    # The node's next scan touches its claims and hands back abandoned ones.
    env.o._single_claim_upkeep(force=True, sweep=False)
    return door


def _intake(env):
    sm = WatcherStateManager(env.db, SimpleNamespace(
        stability_wait_seconds=0, max_retries=3, get_quarantine_dir=lambda: None))
    door = env.single_animal_output
    sm.discover_new_singles(door)            # first sighting never counts
    return sm.discover_new_singles(door)


def test_the_cut_waits_at_the_top_of_the_singles_folder_unclaimed(env, monkeypatch):
    door = _crop_without_a_queue(env, monkeypatch)
    for kid in (KID1, KID2):
        assert (door / f"{kid}.mp4").is_file(), "any node with a queue must see it"
        assert not (door / single_claim.INFLIGHT_DIR / HOST / f"{kid}.mp4").exists()
        row = env.db.get_video(kid)
        assert row["state"] == "unresolvable"
        # The opening words older versions wrote too (single_claim.NO_DLC_QUEUE_REASON).
        assert row["error_message"].startswith("no local DLC queue is configured on this node")


def test_the_row_is_not_taken_again_while_the_node_still_has_no_queue(env, monkeypatch):
    _crop_without_a_queue(env, monkeypatch)
    assert _intake(env) == []
    assert env.db.get_video(KID1)["state"] == "unresolvable"


def test_the_row_is_taken_again_once_the_node_has_a_queue(env, monkeypatch):
    door = _crop_without_a_queue(env, monkeypatch)
    monkeypatch.setattr(orch.Paths, "DLC_QUEUE", env.dlc_queue, raising=False)
    found = _intake(env)
    assert sorted(found) == sorted([KID1, KID2]), (
        "a cut child must not wait for ever once this node can pose it")
    row = env.db.get_video(KID1)
    assert row["state"] == "validated"
    assert row["current_path"] == str(door / f"{KID1}.mp4")
    # And the ordinary adopt then queues it for pose.
    env.o._adopt_single_for_dlc({"id": KID1, "data": row})
    assert env.db.get_video(KID1)["state"] == "dlc_queued"
    assert (env.dlc_queue / f"{KID1}.mp4").is_file()


def test_a_row_another_node_holds_a_claim_on_is_not_taken_again(env, monkeypatch):
    door = _crop_without_a_queue(env, monkeypatch)
    other = door / single_claim.INFLIGHT_DIR / "NODE-B"
    other.mkdir(parents=True)
    (door / f"{KID1}.mp4").rename(other / f"{KID1}.mp4")     # NODE-B took it
    monkeypatch.setattr(orch.Paths, "DLC_QUEUE", env.dlc_queue, raising=False)
    assert KID1 not in _intake(env)
    assert env.db.get_video(KID1)["state"] == "unresolvable"
