"""A freshly cut single must be reachable by some OTHER node if this one stalls.

Cropping used to publish only the fact that a single existed -- the row went to the
shared table while the file went to this node's LOCAL queue, which no other machine
can see. So a node that stalled took its poses down with it and no other GPU could
help, which is what stranded eight videos for a week while three machines sat idle.

The rest of the design already solves this: a claim is held by a heartbeat and returned
automatically when the holder goes quiet. These pin that the crop path now joins it.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from mousereach.watcher import orchestrator as orch
from mousereach.watcher import single_claim


@pytest.fixture
def share(tmp_path, monkeypatch):
    door = tmp_path / "Unanalyzed" / "Single_Animal"
    door.mkdir(parents=True)
    monkeypatch.setattr(orch.Paths, "SINGLE_ANIMAL_OUTPUT", door, raising=False)
    monkeypatch.setattr(single_claim.Paths, "SINGLE_ANIMAL_OUTPUT", door, raising=False)
    return door


def _node(host="NODE-A"):
    o = object.__new__(orch.DLCOrchestrator)
    o.hostname = host
    return o


def _cut(tmp_path, name="20260101_ABC0101_P1.mp4"):
    local = tmp_path / "work" / name
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(b"a cut single")
    return local


def test_a_cut_single_lands_on_the_share_already_claimed(share, tmp_path):
    local = _cut(tmp_path)
    dest = _node()._publish_cropped_single(local, local.stem)

    assert dest is not None and dest.is_file(), "the artefact must reach the share"
    assert dest == single_claim.claimed_path(local.stem, "NODE-A"), (
        "it must land in THIS node's claim folder -- that placement IS the claim, "
        "so there is no window in which another node could take a video this one "
        "is already posing")
    assert not (share / local.name).exists(), (
        "it must not sit unclaimed in the front door, where a second node would "
        "pose the same video")


def test_a_stalled_node_hands_the_video_back_to_everyone(share, tmp_path):
    """The whole point: this node dies, so its heartbeat stops, so the video
    returns to the front door for any node."""
    local = _cut(tmp_path)
    dest = _node()._publish_cropped_single(local, local.stem)

    import os, time
    old = time.time() - single_claim.STALE_S - 60
    os.utime(dest, (old, old))                    # nobody has touched it since

    returned = single_claim.reclaim_stale()
    assert (share / local.name).is_file(), (
        "a video whose holder went quiet must come back to the shared folder")
    assert any(p.name == local.name for p in returned)


def test_publishing_twice_does_not_duplicate_or_overwrite(share, tmp_path):
    """A re-claimed collage re-cuts its children; the existing claim stands."""
    local = _cut(tmp_path)
    node = _node()
    first = node._publish_cropped_single(local, local.stem)
    first.write_bytes(b"the copy already being posed")
    second = node._publish_cropped_single(local, local.stem)
    assert second == first
    assert first.read_bytes() == b"the copy already being posed"


def test_a_share_that_cannot_be_written_never_stops_the_crop(tmp_path, monkeypatch):
    """No shared folder configured, or it fails: the video is still processed
    locally exactly as it was before, which is no worse than the old behaviour."""
    monkeypatch.setattr(orch.Paths, "SINGLE_ANIMAL_OUTPUT", None, raising=False)
    monkeypatch.setattr(single_claim.Paths, "SINGLE_ANIMAL_OUTPUT", None, raising=False)
    assert _node()._publish_cropped_single(_cut(tmp_path), "VID") is None


def test_a_copy_failure_is_reported_but_not_raised(share, tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "safe_copy", lambda *a, **k: False)
    assert _node()._publish_cropped_single(_cut(tmp_path), "VID") is None


def test_the_state_that_holds_the_claim_is_the_one_cropping_sets():
    """Cropping marks the child 'dlc_queued'; the heartbeat must keep claims in
    that state alive, or the video would be handed back while still being posed."""
    assert 'dlc_queued' in orch.DLCOrchestrator._CLAIM_KEEPALIVE_STATES
