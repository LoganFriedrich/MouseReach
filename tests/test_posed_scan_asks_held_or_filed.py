"""The processing server asks the shared drive before taking in a posed video it
has no row for.

A held or filed video's posed copy stays in the posed folder. A server whose
database has no row for it (rebuilt, replaced, or a row never written) used to
register that copy as new work, analyse the video again and route it into the
queue folder that already held it. Now a bundle in a review queue makes the
row that queue's state (an empty queue folder is not a hold), and a filed video with the SAME pose makes it
'archived'; a filed video with a new pose is still taken in.
tmp_path only (the fixtures of test_singles_intake_claim).
"""
import json
from types import SimpleNamespace

from mousereach.watcher.db import WatcherDB
from mousereach.watcher.state import WatcherStateManager

from test_singles_intake_claim import POSE, _archive, env  # noqa: F401

VID = "20240101_ABC9901_P1"
OTHER_POSE = "DLC_resnet50_OtherModelshuffle1_5000"


def _posed(env, pose=POSE):
    (env.dlc_staging / f"{VID}.mp4").write_bytes(b"video")
    (env.dlc_staging / f"{VID}{pose}.h5").write_bytes(b"pose")


def _scan(env):
    db = WatcherDB(db_path=env.tmp / "server.db")
    state = WatcherStateManager(db, SimpleNamespace(stability_wait_seconds=0))
    return db, state.discover_dlc_staged(env.dlc_staging)


def _filed_with(pose):
    d = _archive(VID)
    (d / f"{VID}_processing_manifest.json").write_text(
        json.dumps({"dlc_model": {"dlc_scorer": pose}}), encoding="ascii")


def test_a_video_held_in_triage_is_recorded_as_held_not_taken_in(env):
    _posed(env)
    (env.triage_review / VID).mkdir()
    (env.triage_review / VID / f"{VID}_segments.json").write_text("{}")
    db, found = _scan(env)
    assert found == []
    assert db.get_video(VID)["state"] == "triage"


def test_a_video_held_in_deep_review_is_recorded_as_held(env):
    _posed(env)
    (env.deep_review / VID).mkdir()
    (env.deep_review / VID / f"{VID}.mp4").write_bytes(b"video")
    db, found = _scan(env)
    assert found == [] and db.get_video(VID)["state"] == "deep_review"


def test_a_filed_video_with_the_same_pose_is_recorded_as_filed(env):
    _posed(env)
    _filed_with(POSE)
    db, found = _scan(env)
    assert found == [] and db.get_video(VID)["state"] == "archived"


def test_a_filed_video_with_a_new_pose_is_taken_in(env):
    _posed(env, pose=OTHER_POSE)
    _filed_with(POSE)
    db, found = _scan(env)
    assert found == [VID] and db.get_video(VID)["state"] == "dlc_complete"


def test_a_video_neither_held_nor_filed_is_taken_in(env):
    _posed(env)
    db, found = _scan(env)
    assert found == [VID] and db.get_video(VID)["state"] == "dlc_complete"


def test_an_empty_queue_folder_is_not_a_hold(env):
    """An empty or leftover folder in a review queue holds no bundle: the
    video is taken in as usual, never recorded as held (no review tool would
    list it, and the return scan retires the empty folder)."""
    _posed(env)
    (env.triage_review / VID).mkdir()
    (env.deep_review / ".incoming" / VID).mkdir(parents=True)
    db, found = _scan(env)
    assert found == [VID] and db.get_video(VID)["state"] == "dlc_complete"
