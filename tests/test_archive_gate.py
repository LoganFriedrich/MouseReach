#!/usr/bin/env python3
"""Filing a finished video asks the files, not a cache -- and stops asking so often.

`Analyzed` is where a finished video is supposed to end up: the pipeline's final
output, the thing the X: backup mirrors, and the only state the version checker
looks at. Filing is the last step, and on the processing server it had not
succeeded once since February.

The reason was not the videos. `check_archive_ready` asked the pipeline index --
a cache that exists to save the dashboard a folder scan, and which is allowed to
go stale. It had. Measured 2026-08-24 on this node: 1,084 videos held in
'processed', of which 1,083 were marked approved by every stage ON DISK. The
index disagreed for the large majority: segmentation reading 'auto_review' (a
spelling only the index ever writes), and reach and outcome missing entirely for
39 of every 60 sampled. Nothing filed, so nothing reached 'archived', so the
version checker saw zero videos and had never run.

Two properties, pinned here: the verdict comes from the stage output files, and
a video that genuinely cannot be filed is not retried once a second forever
(the old loop logged 326,235 failures).
"""

import json
import pytest
from pathlib import Path

from mousereach.archive.core import check_archive_ready


@pytest.fixture
def outputs(tmp_path):
    """A finished video's three stage outputs, all approved."""
    vid = "20250624_CNT0104_P3"

    def write(stage_suffix, status):
        p = tmp_path / f"{vid}{stage_suffix}"
        p.write_text(json.dumps({} if status is None else {"validation_status": status}))
        return p

    for suffix in ("_segments.json", "_reaches.json", "_pellet_outcomes.json"):
        write(suffix, "auto_approved")
    return vid, tmp_path, write


class TestReadinessComesFromTheFiles:

    def test_all_three_approved_is_ready(self, outputs):
        vid, d, _ = outputs
        ready, status = check_archive_ready(vid, source_dir=d)
        assert ready
        assert status == {"seg": "auto_approved", "reach": "auto_approved",
                          "outcome": "auto_approved"}

    def test_human_validated_counts_too(self, outputs):
        vid, d, write = outputs
        write("_reaches.json", "validated")
        ready, _ = check_archive_ready(vid, source_dir=d)
        assert ready

    def test_a_stage_needing_review_holds_the_video(self, outputs):
        vid, d, write = outputs
        write("_reaches.json", "needs_review")
        ready, status = check_archive_ready(vid, source_dir=d)
        assert not ready
        assert status["reach"] == "needs_review"

    def test_a_missing_output_holds_the_video(self, outputs):
        vid, d, _ = outputs
        (d / f"{vid}_pellet_outcomes.json").unlink()
        ready, status = check_archive_ready(vid, source_dir=d)
        assert not ready
        assert status["outcome"] == "not_started"

    def test_a_file_with_no_verdict_holds_the_video(self, outputs):
        vid, d, write = outputs
        write("_pellet_outcomes.json", None)
        ready, _ = check_archive_ready(vid, source_dir=d)
        assert not ready

    def test_an_unreadable_file_is_not_approval(self, outputs):
        """Unreadable is not the same as fine."""
        vid, d, _ = outputs
        (d / f"{vid}_pellet_outcomes.json").write_text("{ this is not json")
        ready, status = check_archive_ready(vid, source_dir=d)
        assert not ready
        assert status["outcome"] == "unreadable"

    def test_a_stale_index_cannot_hold_an_approved_video(self, outputs, monkeypatch):
        """The regression that mattered: the cache said no, the files said yes.

        The index reports what it last scanned. If it is asked at all, a video
        whose files are approved can be held indefinitely -- which is what
        happened to 1,083 videos for six months.
        """
        vid, d, _ = outputs

        class Boom:
            def load(self):
                raise AssertionError("the readiness check must not read the index")

            def get_pipeline_status(self, _):
                raise AssertionError("the readiness check must not read the index")

        import mousereach.index as idx
        monkeypatch.setattr(idx, "PipelineIndex", Boom, raising=False)

        ready, _ = check_archive_ready(vid, source_dir=d)
        assert ready


class TestRetryBackoff:
    """A video that cannot be filed must not be retried every cycle forever."""

    @pytest.fixture
    def orch(self):
        from mousereach.watcher.orchestrator import BaseOrchestrator
        return BaseOrchestrator.__new__(BaseOrchestrator)   # no __init__ needed

    def test_first_failure_starts_a_wait(self, orch):
        delay = orch._note_archive_failure("v1")
        assert delay == 60
        assert orch._archive_backoff_active("v1")

    def test_the_wait_grows_with_repeated_failures(self, orch):
        delays = [orch._note_archive_failure("v1") for _ in range(5)]
        assert delays == [60, 300, 900, 3600, 3600], "escalates, then caps at an hour"

    def test_an_untried_video_is_not_held_back(self, orch):
        assert not orch._archive_backoff_active("never-seen")

    def test_success_forgets_the_history(self, orch):
        orch._note_archive_failure("v1")
        orch._clear_archive_backoff("v1")
        assert not orch._archive_backoff_active("v1")

    def test_the_work_picker_skips_a_backed_off_video(self):
        """Both roles filter on it, so one stuck video cannot block the queue."""
        import inspect
        from mousereach.watcher import orchestrator as o
        for fn in (o.ProcessingOrchestrator._get_next_work_item,
                   o.DLCOrchestrator._get_next_work_item):
            assert "_archive_backoff_active" in inspect.getsource(fn)


class TestArchivedStateTransition:
    """Filing a video must be able to record that it was filed.

    'processed' cannot go straight to 'archived' -- the state machine routes
    through 'archiving'. Both roles jumped it, which raised. That had never
    mattered because filing had never actually succeeded; the moment it did, the
    files moved and the row stayed in 'processed' with nothing left in Processing,
    so the next pass read 'not ready' forever. Ten videos went that way in the
    first 45 seconds.
    """

    def test_the_route_to_archived_is_legal(self):
        from mousereach.watcher.db import VIDEO_TRANSITIONS
        assert 'archived' not in VIDEO_TRANSITIONS['processed'], (
            'if this ever becomes legal, the two-hop below is merely redundant')
        assert 'archiving' in VIDEO_TRANSITIONS['processed']
        assert 'archived' in VIDEO_TRANSITIONS['archiving']

    def test_a_real_video_can_walk_it(self, tmp_path):
        from mousereach.watcher.db import WatcherDB
        db = WatcherDB(db_path=tmp_path / 'w.db')
        db.register_video(video_id='v1', source_path=str(tmp_path / 'v1.mp4'))
        for st in ('validated', 'dlc_queued', 'dlc_running', 'dlc_complete',
                   'processing', 'processed', 'archiving', 'archived'):
            db.update_state('v1', st)
        assert db.get_video('v1')['state'] == 'archived'

    def test_both_roles_hop_through_archiving(self):
        import inspect
        from mousereach.watcher import orchestrator as o
        for fn in (o.ProcessingOrchestrator._archive_to_nas,
                   o.DLCOrchestrator._archive_locally_processed):
            src = inspect.getsource(fn)
            assert "update_state(video_id, 'archiving')" in src, fn.__qualname__


class TestFilingIsRecorded:
    """Moving the files and recording the move must not come apart.

    archive_video MOVES files. If the node's row is not updated the watcher picks
    the video up again, finds nothing left in Processing, and reads 'not ready'
    forever -- correctly filed, permanently recorded as unfiled, and invisible to
    the version checker, which only looks at 'archived'. The bulk command
    (mousereach-archive) did exactly this, so draining a backlog with it would
    have stranded every video it filed.
    """

    @pytest.fixture
    def node_db(self, tmp_path, monkeypatch):
        from mousereach.watcher.db import WatcherDB
        import mousereach.config as cfg

        db_path = tmp_path / "watcher_local.db"
        db = WatcherDB(db_path=db_path)

        class FakeCfg:
            db_path = None
        FakeCfg.db_path = db_path
        monkeypatch.setattr(cfg.WatcherConfig, "load", staticmethod(lambda: FakeCfg()))
        return db

    def _make(self, db, vid, state):
        db.register_video(video_id=vid, source_path="x")
        for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete",
                   "processing", "processed"):
            db.update_state(vid, st)
            if st == state:
                return

    def test_a_filed_video_is_recorded_archived(self, node_db):
        from mousereach.archive.core import record_archived
        self._make(node_db, "v1", "processed")
        assert record_archived("v1") is True
        assert node_db.get_video("v1")["state"] == "archived"

    def test_it_is_idempotent(self, node_db):
        from mousereach.archive.core import record_archived
        self._make(node_db, "v1", "processed")
        record_archived("v1")
        assert record_archived("v1") is True
        assert node_db.get_video("v1")["state"] == "archived"

    def test_a_video_this_node_does_not_know_is_not_an_error(self, node_db):
        from mousereach.archive.core import record_archived
        assert record_archived("never_seen_00000_P1") is False

    def test_it_writes_to_the_configured_database(self, node_db, tmp_path):
        """Not the default path -- this node overrides db_path, and writing to
        the wrong file would be worse than the read-the-wrong-file bug of
        2026-08-23."""
        import inspect
        from mousereach.archive import core
        src = inspect.getsource(core.record_archived)
        assert "WatcherConfig" in src and "db_path" in src

    def test_both_bulk_paths_record(self):
        import inspect
        from mousereach.archive import core, cli
        assert "record_archived" in inspect.getsource(core.archive_all)
        assert "record_archived" in inspect.getsource(cli.main)
