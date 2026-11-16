"""A GPU node works its DLC-queue copy of a video only while it holds the video.

A node that gave a claim back but kept its local queue copy (the give-back
could not confirm the shared copy; a failed pose; a failed hand-off) used to
pose that copy again -- at its next start, or when the row was queued by hand
-- while another node posed the shared copy. Now, at the pose start and at
startup, a DLC-queue copy with no claim of this node (and no re-pose request)
is not worked when the video is also at the top of the shared singles folder,
in another node's claim folder, or already in the posed folder. A video in no
shared place (cut here and never published, dropped straight into the queue)
is worked as before. A queue copy that is not worked because its video is
confirmed elsewhere (the other copy the same size) is a machine leftover and
is deleted -- never kept for a person, never moved aside; one whose other copy
cannot be confirmed stays where it is. tmp_path only (the fixtures of test_singles_intake_claim).
"""
import pytest

from mousereach.watcher import repose, single_claim

from test_singles_intake_claim import HOST, POSE, _as, _node, env  # noqa: F401

VID = "20240101_ABC9901_P1"


def _queued(env, a, state="dlc_queued", **fields):
    _as(env, a)
    local = a.queue / f"{VID}.mp4"
    local.write_bytes(b"video")
    a.db.register_video(VID, str(local), current_path=str(local))
    a.db.force_state(VID, state, reason="test setup", **fields)
    a._already_done_or_held = lambda vid: None
    return local


def _pose(a):
    return a._process_single_dlc({"id": VID, "data": a.db.get_video(VID)})


def test_not_posed_when_the_video_is_back_in_the_shared_folder(env):
    a = _node(env, HOST)
    local = _queued(env, a)
    (env.door / f"{VID}.mp4").write_bytes(b"video")
    assert _pose(a) is False
    row = a.db.get_video(VID)
    assert row["state"] == "unresolvable"
    assert row["error_message"].startswith(single_claim.SHARED_REASON)
    assert not local.exists()                       # the leftover is deleted
    assert (env.door / f"{VID}.mp4").is_file()      # the current copy is not


def test_not_posed_while_another_node_holds_it(env):
    a = _node(env, HOST)
    _queued(env, a)
    other = single_claim.host_dir("NODE-OTHER")
    other.mkdir(parents=True)
    (other / f"{VID}.mp4").write_bytes(b"video")
    local = a.queue / f"{VID}.mp4"
    assert _pose(a) is False
    row = a.db.get_video(VID)
    assert row["state"] == "unresolvable" and "NODE-OTHER" in row["error_message"]
    assert not local.exists() and (other / f"{VID}.mp4").is_file()


def test_not_posed_when_it_is_already_handed_on(env):
    a = _node(env, HOST)
    _queued(env, a)
    (env.dlc_staging / f"{VID}.mp4").write_bytes(b"video")
    (env.dlc_staging / f"{VID}{POSE}.h5").write_bytes(b"pose")
    assert _pose(a) is False
    assert "posed folder" in a.db.get_video(VID)["error_message"]
    assert not (a.queue / f"{VID}.mp4").exists()
    assert (env.dlc_staging / f"{VID}.mp4").is_file()


def test_a_queue_copy_whose_twin_differs_in_size_is_kept(env):
    """The copy elsewhere could not be confirmed as the same video: the
    queue copy is not worked, but it is not deleted either (it may be the
    only good copy)."""
    a = _node(env, HOST)
    local = _queued(env, a)
    (env.door / f"{VID}.mp4").write_bytes(b"a different, longer file")
    assert _pose(a) is False
    assert a.db.get_video(VID)["state"] == "unresolvable"
    assert local.is_file()
    assert not [d for d in a.queue.iterdir() if d.is_dir()]   # nothing moved aside


@pytest.mark.parametrize("case", ["own claim", "nowhere else", "re-pose request"])
def test_posed_when_this_node_holds_it(env, case):
    a = _node(env, HOST)
    fields = {}
    if case == "re-pose request":
        fields["mark_reason"] = f"{repose.REASON_PREFIX} test"
        (env.door / f"{VID}.mp4").write_bytes(b"video")
    _queued(env, a, **fields)
    if case == "own claim":
        mine = single_claim.host_dir(HOST)
        mine.mkdir(parents=True)
        (mine / f"{VID}.mp4").write_bytes(b"video")
        (env.door / f"{VID}.mp4").write_bytes(b"a second copy")
    _pose(a)        # no model is configured here, so it stops before DeepLabCut
    assert a.db.get_video(VID)["state"] == "dlc_queued"


def test_a_restart_does_not_retry_a_failed_copy_the_node_gave_back(env):
    a = _node(env, HOST)
    _queued(env, a, state="dlc_running")
    a.db.mark_failed(VID, "DLC failed: test")
    (env.door / f"{VID}.mp4").write_bytes(b"video")        # the claim went back
    a._recover_local_dlc_queue()
    row = a.db.get_video(VID)
    assert row["state"] == "unresolvable"
    assert row["error_message"].startswith(single_claim.SHARED_REASON)
    assert not (a.queue / f"{VID}.mp4").exists()


def test_a_restart_still_retries_a_failed_copy_it_holds(env):
    a = _node(env, HOST)
    _queued(env, a, state="dlc_running")
    a.db.mark_failed(VID, "DLC failed: test")
    mine = single_claim.host_dir(HOST)
    mine.mkdir(parents=True)
    (mine / f"{VID}.mp4").write_bytes(b"video")
    a._recover_local_dlc_queue()
    assert a.db.get_video(VID)["state"] == "dlc_queued"


def test_a_restart_does_not_requeue_an_unresolvable_copy_another_node_holds(env):
    a = _node(env, HOST)
    _queued(env, a, state="unresolvable")
    other = single_claim.host_dir("NODE-OTHER")
    other.mkdir(parents=True)
    (other / f"{VID}.mp4").write_bytes(b"video")
    a._recover_local_dlc_queue()
    assert a.db.get_video(VID)["state"] == "unresolvable"


def test_a_restart_finishes_a_hand_off_that_stopped_half_way(env):
    """Only the video reached the posed folder; its pose is still in this
    node's queue. The server never takes in a video without a pose, so the
    row must not be parked as 'already in the posed folder': the restart
    picks the stage up again with the pose it has."""
    a = _node(env, HOST)
    _queued(env, a, state="dlc_complete")
    (a.queue / f"{VID}{POSE}.h5").write_bytes(b"pose")
    a.db.mark_failed(VID, "staging failed: test")
    (env.dlc_staging / f"{VID}.mp4").write_bytes(b"video")     # the video only
    a._recover_local_dlc_queue()
    row = a.db.get_video(VID)
    assert row["state"] == "dlc_complete"
    assert "posed folder" not in (row["error_message"] or "")


def test_a_restart_leaves_a_failed_row_with_no_retries_left_failed(env):
    """A failed row past watcher.max_retries stays 'failed' for a person. It
    is not re-labelled 'unresolvable' (which the singles intake would set back
    to 'validated', posing it again once per restart with no limit)."""
    a = _node(env, HOST)
    _queued(env, a, state="dlc_running")
    for i in range(3):                                  # max_retries is 3
        a.db.mark_failed(VID, f"DLC failed: test {i}")
        if i < 2:
            a.db.force_state(VID, "dlc_running", reason="test")
    assert a.db.get_video(VID)["error_count"] == 3
    (env.door / f"{VID}.mp4").write_bytes(b"video")        # the claim went back
    a._recover_local_dlc_queue()
    row = a.db.get_video(VID)
    assert row["state"] == "failed"
    assert row["error_count"] == 3
