"""A shared-record write that fails is retried, and never overwrites anything newer.

WHY (2026-10-22): a GPU node's writes to the shared record (watcher_central.db)
were best-effort -- a failure when the shared database was locked was logged at
DEBUG and dropped. The record fell behind for good: one collage's 8 singles all
existed, but only 3 were on record a day after the 10-21 lock failures. Other
nodes read that record before cutting a collage, so a missing child is a
collage another node may cut and pose a second time. Built under tmp_path with
a real coordination database.
"""
import sqlite3

from mousereach.watcher.coordination import PipelineCoordinator
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import DLCOrchestrator

V1, V2 = "20250101_CNT0101_P1", "20250101_CNT0102_P1"


def _central(tmp_path):
    return PipelineCoordinator(db_path=tmp_path / "central" / "watcher_central.db")


def _row(c, vid):
    if not c.db_path.exists():        # nothing has written the record yet
        return None
    con = sqlite3.connect(c.db_path)
    try:
        r = con.execute("select state, hostname, updated_at from pipeline_videos "
                        "where video_id=?", (vid,)).fetchone()
    finally:
        con.close()
    return r


# ----------------------------------------------------------------- coordinator

def test_a_late_write_lands_when_nothing_newer_is_on_record(tmp_path):
    c = _central(tmp_path)
    assert c.sync_video_state_unless_newer(V1, "NODE-A", "dlc_queued", "2026-10-22T10:00:00")
    assert _row(c, V1) == ("dlc_queued", "NODE-A", "2026-10-22T10:00:00")


def test_a_late_write_never_replaces_a_newer_row(tmp_path):
    c = _central(tmp_path)
    c.sync_video_state(V1, "SERVER", "archived")          # stamped now
    assert not c.sync_video_state_unless_newer(V1, "NODE-A", "dlc_complete",
                                               "2026-10-22T10:00:00")
    assert _row(c, V1)[:2] == ("archived", "SERVER")


def test_insert_if_absent_never_touches_an_existing_row(tmp_path):
    c = _central(tmp_path)
    c.sync_video_state(V1, "SERVER", "archived")
    assert not c.insert_video_state_if_absent(V1, "NODE-A", "dlc_queued", "2026-01-01T00:00:00")
    assert _row(c, V1)[:2] == ("archived", "SERVER")
    assert c.insert_video_state_if_absent(V2, "NODE-A", "dlc_queued", "2026-01-01T00:00:00")
    assert _row(c, V2) == ("dlc_queued", "NODE-A", "2026-01-01T00:00:00")


# ----------------------------------------------------------------- the node

class Flaky:
    """The real coordinator, made to fail its plain writes on demand."""

    def __init__(self, real):
        self.real, self.down = real, False

    def sync_video_state(self, *a, **k):
        if self.down:
            raise OSError("database is locked")
        return self.real.sync_video_state(*a, **k)

    def __getattr__(self, name):
        return getattr(self.real, name)


def _node(tmp_path):
    node = object.__new__(DLCOrchestrator)
    node.hostname = "NODE-A"
    node.coordinator = Flaky(_central(tmp_path))
    node.db = WatcherDB(tmp_path / "w.db")
    return node


def test_a_failed_write_is_kept_and_lands_at_the_next_scan(tmp_path):
    node = _node(tmp_path)
    node.coordinator.down = True
    node._sync_to_connectome(V1, "dlc_queued", collage_id="C.mkv")
    assert _row(node.coordinator.real, V1) is None
    assert V1 in node._failed_shared_writes()

    node.coordinator.down = False
    assert node._retry_failed_shared_writes() == 1
    assert _row(node.coordinator.real, V1)[:2] == ("dlc_queued", "NODE-A")
    assert node._failed_shared_writes() == {}


def test_a_newer_successful_write_drops_the_failed_one(tmp_path):
    node = _node(tmp_path)
    node.coordinator.down = True
    node._sync_to_connectome(V1, "dlc_queued")
    node.coordinator.down = False
    node._sync_to_connectome(V1, "dlc_complete")
    assert node._failed_shared_writes() == {}
    assert node._retry_failed_shared_writes() == 0
    assert _row(node.coordinator.real, V1)[0] == "dlc_complete"


def test_a_retry_stops_while_the_record_is_still_unavailable(tmp_path):
    node = _node(tmp_path)
    node.coordinator.down = True
    node._sync_to_connectome(V1, "dlc_queued")
    node._sync_to_connectome(V2, "dlc_queued")

    def still_down(*a, **k):
        raise OSError("database is locked")
    node.coordinator.real.sync_video_state_unless_newer = still_down
    assert node._retry_failed_shared_writes() == 0
    assert set(node._failed_shared_writes()) == {V1, V2}


def test_startup_puts_only_missing_videos_on_record(tmp_path):
    node = _node(tmp_path)
    for vid, state in ((V1, "dlc_complete"), (V2, "dlc_queued")):
        node.db.register_video(video_id=vid, source_path="x", collage_id="C.mkv")
        node.db.force_state(vid, state, reason="test fixture")
    node.coordinator.real.sync_video_state(V1, "SERVER", "archived")

    assert node._put_missing_on_shared_record() == 1
    assert _row(node.coordinator.real, V1)[:2] == ("archived", "SERVER")   # untouched
    assert _row(node.coordinator.real, V2)[:2] == ("dlc_queued", "NODE-A")
