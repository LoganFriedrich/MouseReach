"""A row already known to have no file here never wins a work slot.

WHY: cross-node recovery learns about videos another machine holds and registers
them with the NO_FILE_HERE placeholder. The staging bucket still selected them, and
the handler then spent the whole poll interval finding out what the row already
said -- one row per poll. Measured on a behaviour-room node on 2026-09-18: six such
rows cost 2 min 41 s of work slots after every unpause (18:14:56 -> 18:17:38, one
every ~32 s, matching poll_interval_seconds 30).

The retirement behaviour itself is correct and untouched; what is removed is paying
a poll to rediscover a fact already recorded. A row with no recorded path at all is
still selected, because locate_video_file may find its file -- that is how a node
picks up work that genuinely is here.
"""
import pytest

from mousereach.watcher.orchestrator import DLCOrchestrator


class FakeDB:
    """Only what _select_work_item reads. Empty collage/single buckets, so a pass
    that takes no video falls through every later bucket and returns None."""

    NO_FILE_HERE = "(no file on this node)"

    def __init__(self, rows):
        self._rows = rows

    def get_videos_in_state(self, state):
        return [dict(r) for r in self._rows if r["state"] == state]

    def get_collages_in_state(self, state):
        return []


class FakeConfig:
    also_process = False
    work_priority = None


def _node(rows):
    node = DLCOrchestrator.__new__(DLCOrchestrator)
    node.db = FakeDB(rows)
    node.config = FakeConfig()
    node._get_priority_animal = lambda: None
    node._pick_from_pool = lambda items, *a, **k: (items[0] if items else None)
    # The collage bucket, reached only when no video was taken: no coordination
    # failure recorded, nothing parked, nothing to retry.
    node._coordinator_init_error = None
    node._claim_backoff_active = lambda *a, **k: False
    node._retry_unsynced_collage_claims = lambda *a, **k: None
    node._repose_requests_to_take = lambda *a, **k: []
    return node


def test_a_row_marked_no_file_here_is_not_selected():
    node = _node([
        {"video_id": "20240101_ABC0101_P1", "state": "dlc_complete",
         "source_path": FakeDB.NO_FILE_HERE, "current_path": None, "animal_id": "ABC0101"},
    ])
    assert node._select_work_item() is None, "a row that says it has no file must not win a slot"


def test_a_real_row_is_still_selected(tmp_path):
    here = tmp_path / "20240101_ABC0102_P1.mp4"
    here.write_bytes(b"video")
    node = _node([
        {"video_id": "20240101_ABC0101_P1", "state": "dlc_complete",
         "source_path": FakeDB.NO_FILE_HERE, "current_path": None, "animal_id": "ABC0101"},
        {"video_id": "20240101_ABC0102_P1", "state": "dlc_complete",
         "source_path": str(here), "current_path": str(here), "animal_id": "ABC0102"},
    ])
    work = node._select_work_item()
    assert work is not None
    assert work["id"] == "20240101_ABC0102_P1", "the row with a real file is the one taken"
    assert work["type"] == "stage_to_nas"


def test_a_row_with_no_recorded_path_is_still_offered():
    """Not the same thing as NO_FILE_HERE: nobody has looked yet, and
    locate_video_file may well find the file on this node."""
    node = _node([
        {"video_id": "20240101_ABC0103_P1", "state": "dlc_complete",
         "source_path": None, "current_path": None, "animal_id": "ABC0103"},
    ])
    work = node._select_work_item()
    assert work is not None and work["id"] == "20240101_ABC0103_P1"
