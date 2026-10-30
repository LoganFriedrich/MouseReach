"""A GPU node's shared-record writes survive an unreachable shared database.

WHY (Lab DLC PC, 2026-10-30): the node restarted while watcher_central.db was
locked. Its connect failed, so coordinator was None; a hand-off and the next
pose start were then DROPPED (the write returned before queuing), the write that
failed just before the stop was lost with the process, and the startup fill
only ever added rows that were missing entirely -- so a video handed off at
10:59 was still on record as "posing" an hour later.
"""
from mousereach.watcher.orchestrator import DLCOrchestrator

HOST = "GPU-NODE"


class FakeCoordinator:
    def __init__(self, on_record):
        self.on_record = on_record            # {video_id: row}
        self.inserted, self.corrected, self.synced = [], [], []

    def get_all_video_states(self):
        return dict(self.on_record)

    def insert_video_states_if_absent(self, rows, chunk=200):
        self.inserted.extend(r["video_id"] for r in rows)
        return len(rows)

    def sync_video_state_unless_newer(self, video_id, hostname, state, attempted_at, **kw):
        self.corrected.append((video_id, state))
        return True

    def sync_video_state(self, video_id, hostname, state, **kw):
        self.synced.append((video_id, state))


class FakeDB:
    def __init__(self, rows):
        self.rows = rows                       # list of dicts with video_id, state, updated_at

    def get_videos_in_state(self, state):
        return [r for r in self.rows if r["state"] == state]


def _node(coordinator, rows):
    o = object.__new__(DLCOrchestrator)
    o.coordinator = coordinator
    o.hostname = HOST
    o.db = FakeDB(rows)
    return o


def test_a_write_with_no_coordinator_is_queued_not_dropped():
    o = _node(None, [])
    o._sync_to_connectome("V1", "dlc_running")
    assert o._failed_shared_writes()["V1"][0] == "dlc_running"


def test_the_fill_corrects_this_hosts_rows_that_are_behind():
    rows = [{"video_id": "V1", "state": "archived", "updated_at": "2026-10-30T11:02:21"},
            {"video_id": "V2", "state": "dlc_running", "updated_at": "2026-10-30T11:02:31"},
            {"video_id": "V3", "state": "dlc_queued", "updated_at": "2026-10-30T10:00:00"}]
    coord = FakeCoordinator({
        "V1": {"hostname": HOST, "state": "dlc_running"},        # behind: handed off since
        "V2": {"hostname": HOST, "state": "dlc_queued"},         # behind: posing since
        "V3": {"hostname": HOST, "state": "dlc_queued"},         # agrees
    })
    o = _node(coord, rows + [{"video_id": "V4", "state": "validated", "updated_at": "x"}])
    n = o._put_missing_on_shared_record()
    assert coord.inserted == ["V4"]
    assert sorted(coord.corrected) == [("V1", "archived"), ("V2", "dlc_running")]
    assert n == 3 and o._shared_fill_pending is False


def test_another_hosts_row_is_never_touched():
    rows = [{"video_id": "V1", "state": "archived", "updated_at": "t"}]
    coord = FakeCoordinator({"V1": {"hostname": "SERVER", "state": "processing"}})
    o = _node(coord, rows)
    assert o._put_missing_on_shared_record() == 0
    assert coord.corrected == [] and coord.inserted == []


def test_a_fill_that_could_not_run_is_remembered_for_the_reconnect():
    o = _node(None, [{"video_id": "V1", "state": "archived", "updated_at": "t"}])
    assert o._put_missing_on_shared_record() == 0
    assert o._shared_fill_pending is True
