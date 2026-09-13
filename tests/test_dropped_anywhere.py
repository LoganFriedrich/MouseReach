"""A file dropped where videos go must actually get processed.

WHY: the folder layout invites it -- `Processing/Single_Animal` is where
cropped single-mouse videos live -- and dropping one there did nothing. The
watcher noticed the file, wrote it into its database as 'validated', and then
no job list ever selected that state, so it sat there for good. The database
even carried a work queue that WOULD have picked it up (state.py
get_next_work_item) and a stall detector (get_stalled_items); neither has a
single caller. A person who drops a video in the obvious place and waits is
doing nothing wrong, and must not have to know any of that.

Same for a file dropped straight into a node's own DLC queue: it was only
noticed at startup, so it waited for a restart that might never come.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.watcher.orchestrator as orch
from mousereach.watcher.orchestrator import DLCOrchestrator

VID = "20240101_ABC0101_P1"
OTHER = "20240102_ABC0102_P1"
POSE = "DLC_resnet101_MPSAOct27shuffle3_100000"


class FakeDB:
    def __init__(self, videos=None, rows=None):
        self._by_state = videos or {}
        self._rows = rows or {}
        self.moves = []
        self.registered = []
        self.unresolvable = []
        self.failed = []
        self.logged = []

    def get_videos_in_state(self, state):
        return [{**self._rows.get(v, {}), "video_id": v}
                for v in self._by_state.get(state, [])]

    def get_collages_in_state(self, state):
        return []

    def get_video(self, video_id):
        return self._rows.get(video_id)

    def register_video(self, video_id, source_path=None, current_path=None, **k):
        self.registered.append(video_id)
        self._rows[video_id] = {"video_id": video_id, "state": "discovered"}

    def update_state(self, video_id, new_state, **k):
        self.moves.append((video_id, new_state, k.get("current_path")))
        self._rows.setdefault(video_id, {})["state"] = new_state

    def force_state(self, video_id, new_state, reason=None, **k):
        self.moves.append((video_id, new_state, k.get("current_path")))
        self._rows.setdefault(video_id, {})["state"] = new_state

    def mark_unresolvable(self, video_id, reason):
        self.unresolvable.append((video_id, reason))

    def mark_failed(self, video_id, error):
        self.failed.append((video_id, error))

    def log_step(self, video_id, step, status, message=None, duration=None):
        self.logged.append((video_id, step, status))


@pytest.fixture
def node(tmp_path, monkeypatch):
    """A GPU node with a local queue and a shared singles folder of its own."""
    local = tmp_path / "local_queue"
    shared = tmp_path / "Single_Animal"
    local.mkdir()
    shared.mkdir()
    monkeypatch.setattr(orch.Paths, "DLC_QUEUE", local)
    monkeypatch.setattr(orch.Paths, "SINGLE_ANIMAL_OUTPUT", shared)

    o = object.__new__(DLCOrchestrator)
    o.db = FakeDB()
    o.hostname = "test-node"
    o.config = SimpleNamespace(also_process=False, work_priority=None,
                               max_retries=3, dlc_config_path=None)
    o._get_priority_animal = lambda: None
    o.local = local
    o.shared = shared
    return o


# ------------------------------------------------- dropped in the shared folder

def test_a_video_left_in_the_shared_folder_is_taken_on_and_queued(node):
    src = node.shared / f"{VID}.mp4"
    src.write_bytes(b"a video")
    node.db._rows[VID] = {"video_id": VID, "state": "validated",
                          "current_path": str(src)}

    assert node._adopt_single_for_dlc({"id": VID, "data": node.db._rows[VID]}) is True

    # copied onto the node, not posed where it lay: DLC writes beside its input
    dest = node.local / f"{VID}.mp4"
    assert dest.read_bytes() == b"a video"
    assert src.exists(), "the shared copy must not be moved out from under anyone"
    assert node.db.moves == [(VID, "dlc_queued", str(dest))]
    assert (VID, "adopt", "completed") in node.db.logged


def test_it_does_not_copy_again_when_the_video_is_already_here(node):
    src = node.shared / f"{VID}.mp4"
    src.write_bytes(b"a video")
    dest = node.local / f"{VID}.mp4"
    dest.write_bytes(b"a video")
    before = dest.stat().st_mtime_ns
    node.db._rows[VID] = {"video_id": VID, "state": "validated",
                          "current_path": str(src)}

    assert node._adopt_single_for_dlc({"id": VID, "data": node.db._rows[VID]}) is True
    assert dest.stat().st_mtime_ns == before
    assert node.db.moves == [(VID, "dlc_queued", str(dest))]


def test_a_video_already_in_the_archive_is_never_posed_again(node, tmp_path, monkeypatch):
    """The guard that stops a fresh node re-posing the whole corpus.

    Before 'validated' was worked at all, the shared folder was full of
    finished videos and being ignored was the only thing keeping them quiet.
    Measured on the lab's own folder: 2,271 of them, ~14 min of GPU each.
    """
    import mousereach.archive.core as core
    archive = tmp_path / "Analyzed" / "X"
    archive.mkdir(parents=True)
    (archive / f"{VID}_processing_manifest.json").write_text("{}")
    monkeypatch.setattr(core, "get_archive_destination", lambda v: archive)

    src = node.shared / f"{VID}.mp4"
    src.write_bytes(b"a video")
    node.db._rows[VID] = {"video_id": VID, "state": "validated",
                          "current_path": str(src)}

    assert node._adopt_single_for_dlc({"id": VID, "data": node.db._rows[VID]}) is True
    assert node.db.moves == [(VID, "archived", None)]      # recorded, not queued
    assert not (node.local / f"{VID}.mp4").exists()        # no GPU work created


def test_a_video_not_in_the_archive_is_still_posed(node, tmp_path, monkeypatch):
    """The guard must not swallow genuinely unprocessed videos."""
    import mousereach.archive.core as core
    empty = tmp_path / "Analyzed" / "X"
    empty.mkdir(parents=True)
    monkeypatch.setattr(core, "get_archive_destination", lambda v: empty)

    src = node.shared / f"{VID}.mp4"
    src.write_bytes(b"a video")
    node.db._rows[VID] = {"video_id": VID, "state": "validated",
                          "current_path": str(src)}

    assert node._adopt_single_for_dlc({"id": VID, "data": node.db._rows[VID]}) is True
    assert node.db.moves == [(VID, "dlc_queued", str(node.local / f"{VID}.mp4"))]


def test_a_video_that_has_since_been_removed_is_parked_not_retried_forever(node):
    node.db._rows[VID] = {"video_id": VID, "state": "validated",
                          "current_path": str(node.shared / f"{VID}.mp4")}
    assert node._adopt_single_for_dlc({"id": VID, "data": node.db._rows[VID]}) is False
    assert node.db.unresolvable and node.db.unresolvable[0][0] == VID
    assert node.db.moves == []


def test_the_work_loop_actually_selects_it(node, monkeypatch):
    """The whole defect was that nothing ever picked this state up."""
    import mousereach.watcher.work_priority as wp
    monkeypatch.setattr(wp, "read_lab_priority", lambda *a, **k: None)
    wp.invalidate_lab_policy()
    node.db._by_state = {"validated": [VID]}
    node.db._rows[VID] = {"video_id": VID, "state": "validated",
                          "animal_id": "ABC0101"}

    work = node._select_work_item()
    assert work is not None, "a video in the shared folder must be selected as work"
    assert work["type"] == "adopt_single"
    assert work["id"] == VID


# ------------------------------------------------- dropped into the local queue

def test_a_file_dropped_into_the_queue_is_picked_up_without_a_restart(node):
    (node.local / f"{VID}.mp4").write_bytes(b"v")
    assert node._adopt_untracked_queue_files() == 1
    assert node.db.registered == [VID]
    assert node.db.moves == [(VID, "dlc_queued", str(node.local / f"{VID}.mp4"))]


def test_a_file_dropped_with_its_pose_beside_it_skips_straight_past_posing(node):
    (node.local / f"{VID}.mp4").write_bytes(b"v")
    (node.local / f"{VID}{POSE}.h5").write_bytes(b"pose")
    assert node._adopt_untracked_queue_files() == 1
    assert node.db.moves == [(VID, "dlc_complete", str(node.local / f"{VID}.mp4"))]


def test_deeplabcut_by_products_are_ignored_not_quarantined(node):
    """A '..._labeled.mp4' is output, not an unprocessed video. Treating it as
    a misnamed one would file a perfectly normal artefact as a problem."""
    (node.local / f"{VID}{POSE}_labeled.mp4").write_bytes(b"overlay")
    assert node._adopt_untracked_queue_files() == 0
    assert node.db.registered == []
    assert node.db.moves == []


def test_a_video_already_being_worked_is_left_alone(node):
    (node.local / f"{VID}.mp4").write_bytes(b"v")
    node.db._rows[VID] = {"video_id": VID, "state": "dlc_running"}
    assert node._adopt_untracked_queue_files() == 0
    assert node.db.moves == []


def test_nothing_to_do_is_not_an_error(node):
    assert node._adopt_untracked_queue_files() == 0


def test_an_unreadable_queue_does_not_stop_the_cycle(node, monkeypatch):
    monkeypatch.setattr(orch.Paths, "DLC_QUEUE", node.local / "gone")
    assert node._adopt_untracked_queue_files() == 0
