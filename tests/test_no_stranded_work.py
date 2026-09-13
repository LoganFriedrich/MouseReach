"""Interrupting work must never strand it.

WHY: a node sets 'dlc_running' while it poses, 'archiving' while it files, and
'cropping' while it cuts a collage -- and NOTHING selects those states as work.
Kill the watcher at the wrong second (a stop, a reboot, a crash) and the row
stopped moving for good, silently, because the queue looked busy rather than
broken. Found 2026-09-13 when a GPU node had to be told to wait before
restarting, which is the same defect wearing a polite face: if that machine had
simply crashed instead, nobody would have been told anything.

The single-instance mutex means exactly one watcher runs per machine, so
anything still in one of those states AT STARTUP was left by a process that is
already gone. That is what makes reclaiming it a fact rather than a guess.
"""
import threading

import pytest

from mousereach.watcher.orchestrator import DLCOrchestrator, ProcessingOrchestrator


class FakeDB:
    def __init__(self, videos=None, collages=None, raises=False):
        self._videos = videos or {}
        self._collages = collages or {}
        self._raises = raises
        self.moves = []          # (kind, id, new_state, forced)
        self.logged = []

    # --- reads
    def get_videos_in_state(self, state):
        if self._raises:
            raise RuntimeError("database unavailable")
        return [{"video_id": v} for v in self._videos.get(state, [])]

    def get_collages_in_state(self, state):
        if self._raises:
            raise RuntimeError("database unavailable")
        return [{"filename": f} for f in self._collages.get(state, [])]

    # --- writes
    def update_state(self, video_id, new_state, **kw):
        self.moves.append(("video", video_id, new_state, False))

    def force_state(self, video_id, new_state, reason=None, **kw):
        assert reason, "a forced move must record why"
        self.moves.append(("video", video_id, new_state, True))

    def update_collage_state(self, filename, new_state, **kw):
        self.moves.append(("collage", filename, new_state, False))

    def force_collage_state(self, filename, new_state, **kw):
        # This one has no 'reason' parameter, unlike the video equivalent --
        # anything extra would be written as a column.
        assert not kw, "force_collage_state takes no extra fields"
        self.moves.append(("collage", filename, new_state, True))

    def log_step(self, video_id, step, status, message=None, duration=None):
        self.logged.append((video_id, step, status))


def _node(cls, **db):
    o = object.__new__(cls)
    o.db = FakeDB(**db)
    return o


# ---------------------------------------------------------------- videos

def test_a_pose_killed_half_way_goes_back_in_the_queue():
    o = _node(DLCOrchestrator, videos={"dlc_running": ["20240101_ABC0101_P1"]})
    assert o._reclaim_orphaned_work() == {"dlc_running": 1}
    # A legal transition, so no forced write is needed.
    assert o.db.moves == [("video", "20240101_ABC0101_P1", "dlc_queued", False)]
    assert ("20240101_ABC0101_P1", "reclaim", "completed") in o.db.logged


def test_an_archive_killed_half_way_is_filed_again():
    for cls in (DLCOrchestrator, ProcessingOrchestrator):
        o = _node(cls, videos={"archiving": ["20240101_ABC0102_P1"]})
        assert o._reclaim_orphaned_work() == {"archiving": 1}
        # 'archiving' -> 'processed' runs backwards, so it must be forced.
        assert o.db.moves == [("video", "20240101_ABC0102_P1", "processed", True)]


def test_a_video_mid_pipeline_is_left_alone():
    """It needs no rescue: both roles already select 'processing', and the
    pipeline reuses the stage outputs that exist rather than redoing them."""
    for cls in (DLCOrchestrator, ProcessingOrchestrator):
        o = _node(cls, videos={"processing": ["20240101_ABC0103_P1"]})
        assert o._reclaim_orphaned_work() == {}
        assert o.db.moves == []


def test_the_processing_role_never_requeues_a_pose_it_cannot_run():
    """Reclaiming 'dlc_running' on a node with no GPU would swap one dead end
    for another: nothing there ever selects 'dlc_queued'."""
    o = _node(ProcessingOrchestrator, videos={"dlc_running": ["20240101_ABC0104_P1"]})
    assert o._reclaim_orphaned_work() == {}
    assert o.db.moves == []


# ---------------------------------------------------------------- collages

def test_a_crop_killed_half_way_goes_back_to_stable():
    o = _node(DLCOrchestrator, collages={"cropping": ["20240101_collage.mp4"]})
    assert o._reclaim_orphaned_work() == {"cropping": 1}
    # 'cropping' -> 'stable' runs backwards too.
    assert o.db.moves == [("collage", "20240101_collage.mp4", "stable", True)]


def test_the_processing_role_has_no_collages_to_reclaim():
    o = _node(ProcessingOrchestrator, collages={"cropping": ["20240101_collage.mp4"]})
    assert o._reclaim_orphaned_work() == {}
    assert o.db.moves == []


# ---------------------------------------------------------------- safety

def test_everything_at_once():
    o = _node(DLCOrchestrator,
              videos={"dlc_running": ["v1", "v2"], "archiving": ["v3"],
                      "processing": ["v4"], "archived": ["v5"]},
              collages={"cropping": ["c1"], "stable": ["c2"]})
    assert o._reclaim_orphaned_work() == {"dlc_running": 2, "archiving": 1,
                                          "cropping": 1}
    moved = {m[1] for m in o.db.moves}
    assert moved == {"v1", "v2", "v3", "c1"}     # v4, v5 and c2 untouched


def test_a_node_still_starts_when_reclaiming_cannot_read_the_database():
    o = _node(DLCOrchestrator, raises=True)
    assert o._reclaim_orphaned_work() == {}      # no raise


def test_one_row_that_will_not_move_does_not_stop_the_others():
    o = _node(DLCOrchestrator, videos={"dlc_running": ["bad", "good"]})
    real = o.db.update_state

    def fussy(video_id, new_state, **kw):
        if video_id == "bad":
            raise RuntimeError("row is locked")
        return real(video_id, new_state, **kw)

    o.db.update_state = fussy
    assert o._reclaim_orphaned_work() == {"dlc_running": 1}
    assert o.db.moves == [("video", "good", "dlc_queued", False)]


def test_the_reclaim_happens_at_startup_before_any_work_is_taken():
    o = _node(DLCOrchestrator, videos={"dlc_running": ["v1"]})
    o.shutdown = lambda: None
    o._is_paused = lambda: False
    took_work = []
    o._get_next_work_item = lambda: took_work.append(1)
    stop = threading.Event()
    stop.set()                                   # exit before the first cycle
    o.run(stop)
    assert o.db.moves == [("video", "v1", "dlc_queued", False)]
    assert took_work == []                       # reclaimed first, then nothing


def test_a_reclaim_failure_never_stops_a_node_starting():
    o = _node(DLCOrchestrator)
    o.shutdown = lambda: None
    o._reclaim_orphaned_work = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    stop = threading.Event()
    stop.set()
    o.run(stop)                                  # must not raise
