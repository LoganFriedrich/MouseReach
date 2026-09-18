"""Startup recovery never advances a row whose files are not on this node.

WHY: a video another node holds appears in the shared record in a working state
(dlc_complete, processing, ...). Recovery adopts those states so a node knows what
is going on -- but adopting one for a video whose files are NOT here creates a
phantom row: the work loop picks it up, tries to stage a file that does not exist,
fails, and drops the row back to unresolvable. Unresolvable sorts BELOW the remote
state, so the next restart advances it again, for ever.

The "video not in the local database" branch already asked "are the files here?".
The "already in the local database" branch did not, which is the case seen on a
behaviour-room node on 2026-09-18: 12 pathless rows, one work slot each (about
32 s), roughly six minutes of every single startup.
"""
import pytest

from mousereach.watcher import coordination


class FakeLocalDB:
    """Only what recover_local_db touches, recording what was done to each row."""

    NO_FILE_HERE = "(no file on this node)"

    def __init__(self, rows):
        self.rows = dict(rows)
        self.forced = []
        self.unresolvable = []
        self.registered = []

    # -- reads -------------------------------------------------------------
    def get_video(self, video_id):
        return self.rows.get(video_id)

    def video_exists(self, video_id):
        return video_id in self.rows

    def collage_exists(self, filename):
        return True

    # -- writes ------------------------------------------------------------
    def force_state(self, video_id, state, reason=None, **kw):
        self.forced.append((video_id, state))
        self.rows.setdefault(video_id, {})["state"] = state

    def mark_unresolvable(self, video_id, reason):
        self.unresolvable.append((video_id, reason))
        self.rows.setdefault(video_id, {})["state"] = "unresolvable"

    def register_video(self, video_id, **kw):
        self.registered.append(video_id)
        self.rows.setdefault(video_id, {"state": "discovered"})


@pytest.fixture
def coordinator(monkeypatch):
    c = coordination.PipelineCoordinator.__new__(coordination.PipelineCoordinator)
    monkeypatch.setattr(c, "get_all_video_states", lambda: {
        "20240101_ABC0101_P1": {"state": "dlc_complete", "hostname": "other-node",
                                "source_path": None, "collage_id": None},
    }, raising=False)
    monkeypatch.setattr(c, "get_all_collage_states", lambda: {}, raising=False)
    return c


def test_a_pathless_row_is_recorded_as_elsewhere_not_advanced(coordinator, monkeypatch):
    """The bug: this row was advanced to dlc_complete on every restart."""
    monkeypatch.setattr(coordination, "locate_video_file", lambda *a, **k: None)
    local = FakeLocalDB({"20240101_ABC0101_P1": {"state": "unresolvable",
                                                 "current_path": None}})

    stats = coordinator.recover_local_db(local, "this-node")

    assert local.forced == [], "a video whose files are not here must not be advanced"
    assert local.unresolvable == [], "it already says unresolvable; do not rewrite it"
    assert stats["videos_advanced"] == 0


def test_a_pathless_row_in_another_state_is_marked_once(coordinator, monkeypatch):
    monkeypatch.setattr(coordination, "locate_video_file", lambda *a, **k: None)
    local = FakeLocalDB({"20240101_ABC0101_P1": {"state": "dlc_queued",
                                                 "current_path": None}})

    coordinator.recover_local_db(local, "this-node")

    assert local.forced == []
    assert [v for v, _ in local.unresolvable] == ["20240101_ABC0101_P1"]
    assert "other-node" in local.unresolvable[0][1]


def test_a_row_whose_file_IS_here_still_advances(coordinator, monkeypatch, tmp_path):
    """The guard must not stop real recovery: with the file present, advance."""
    here = tmp_path / "20240101_ABC0101_P1.mp4"
    here.write_bytes(b"video")
    monkeypatch.setattr(coordination, "locate_video_file", lambda *a, **k: here)
    local = FakeLocalDB({"20240101_ABC0101_P1": {"state": "dlc_queued",
                                                 "current_path": None}})

    coordinator.recover_local_db(local, "this-node")

    assert local.forced == [("20240101_ABC0101_P1", "dlc_complete")]
    assert local.unresolvable == []


def test_states_that_live_on_the_share_are_adopted_without_a_file_search(coordinator, monkeypatch):
    """archived / triage / deep_review are on the NAS: adopting them is how a node
    learns not to redo finished work, and no work-loop handler selects them."""
    monkeypatch.setattr(coordinator, "get_all_video_states", lambda: {
        "20240101_ABC0101_P1": {"state": "archived", "hostname": "other-node",
                                "source_path": None, "collage_id": None},
    }, raising=False)

    def fail_if_called(*a, **k):
        raise AssertionError("no file search for a state that lives on the share")

    monkeypatch.setattr(coordination, "locate_video_file", fail_if_called)
    local = FakeLocalDB({"20240101_ABC0101_P1": {"state": "dlc_queued",
                                                 "current_path": None}})

    coordinator.recover_local_db(local, "this-node")

    assert local.forced == [("20240101_ABC0101_P1", "archived")]
