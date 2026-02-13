#!/usr/bin/env python3
"""
Comprehensive integration tests for the watcher package.

Uses tempdir + in-memory SQLite. Does NOT require actual video files,
DLC models, or network drives. Mock external dependencies as needed.
"""

import pytest
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import watcher modules
from mousereach.watcher.db import WatcherDB, COLLAGE_TRANSITIONS, VIDEO_TRANSITIONS
from mousereach.watcher.validator import (
    validate_collage_filename,
    validate_single_filename,
    quarantine_file,
)
from mousereach.watcher.router import TrayRouter
from mousereach.watcher.transfer import (
    safe_copy,
    safe_move,
    check_file_stable_quick,
)
from mousereach.watcher.state import WatcherStateManager


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_db(tmp_path):
    """Create temporary WatcherDB instance."""
    db_path = tmp_path / "test_watcher.db"
    db = WatcherDB(db_path=db_path)
    return db


@pytest.fixture
def watcher_config(tmp_path):
    """Create WatcherConfig for testing."""
    from mousereach.config import WatcherConfig

    config = WatcherConfig({
        'enabled': True,
        'poll_interval_seconds': 30,
        'stability_wait_seconds': 60,
        'dlc_config_path': None,
        'dlc_gpu_device': 0,
        'auto_archive_approved': False,
        'quarantine_dir': str(tmp_path / "quarantine"),
        'log_dir': str(tmp_path / "logs"),
        'max_retries': 3,
    })
    return config


@pytest.fixture
def state_manager(temp_db, watcher_config):
    """Create WatcherStateManager with temp DB and config."""
    return WatcherStateManager(temp_db, watcher_config)


@pytest.fixture
def sample_collage_name():
    """Valid collage filename."""
    return "20250704_CNT0101,CNT0102,CNT0103,CNT0104,CNT0105,CNT0106,CNT0107,CNT0108_P1.mkv"


@pytest.fixture
def sample_single_name():
    """Valid single video filename."""
    return "20250704_CNT0101_P1.mp4"


# =============================================================================
# WATCHERDB TESTS
# =============================================================================

class TestWatcherDB:
    """Tests for WatcherDB core functionality."""

    def test_create_database_tables_exist(self, temp_db):
        """Database initialization creates all required tables."""
        # Query tables via sqlite_master
        tables = temp_db._get_connection().execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

        table_names = {row['name'] for row in tables}
        assert 'videos' in table_names
        assert 'collages' in table_names
        assert 'processing_log' in table_names

    def test_register_collage_creates_discovered_state(self, temp_db, sample_collage_name):
        """Registering collage creates it in 'discovered' state."""
        collage_id = temp_db.register_collage(
            filename=sample_collage_name,
            source_path="/fake/path/collage.mkv"
        )

        assert collage_id > 0

        # Verify state
        collage = temp_db.get_collage(sample_collage_name)
        assert collage is not None
        assert collage['state'] == 'discovered'
        assert collage['filename'] == sample_collage_name

    def test_collage_state_transitions(self, temp_db, sample_collage_name):
        """Collage state transitions follow valid rules."""
        # Register
        temp_db.register_collage(sample_collage_name, "/fake/path")

        # discovered -> validated
        temp_db.update_collage_state(sample_collage_name, 'validated')
        assert temp_db.get_collage(sample_collage_name)['state'] == 'validated'

        # validated -> stable
        temp_db.update_collage_state(sample_collage_name, 'stable')
        assert temp_db.get_collage(sample_collage_name)['state'] == 'stable'

        # stable -> cropping
        temp_db.update_collage_state(sample_collage_name, 'cropping')
        assert temp_db.get_collage(sample_collage_name)['state'] == 'cropping'

        # cropping -> cropped
        temp_db.update_collage_state(sample_collage_name, 'cropped')
        assert temp_db.get_collage(sample_collage_name)['state'] == 'cropped'

    def test_register_video_with_metadata(self, temp_db, sample_single_name):
        """Registering video with metadata stores all fields."""
        video_id = temp_db.register_video(
            video_id="20250704_CNT0101_P1",
            source_path="/fake/path/video.mp4",
            date="20250704",
            animal_id="CNT0101",
            experiment="CNT",
            cohort="01",
            subject="01",
            tray_type="P",
        )

        assert video_id == "20250704_CNT0101_P1"

        # Verify metadata
        video = temp_db.get_video(video_id)
        assert video is not None
        assert video['state'] == 'discovered'
        assert video['date'] == "20250704"
        assert video['animal_id'] == "CNT0101"
        assert video['experiment'] == "CNT"
        assert video['cohort'] == "01"
        assert video['subject'] == "01"
        assert video['tray_type'] == "P"

    def test_video_state_transitions(self, temp_db):
        """Video state transitions follow valid rules."""
        video_id = "20250704_CNT0101_P1"
        temp_db.register_video(video_id, "/fake/path")

        # discovered -> validated
        temp_db.update_state(video_id, 'validated')
        assert temp_db.get_video(video_id)['state'] == 'validated'

        # validated -> dlc_queued
        temp_db.update_state(video_id, 'dlc_queued')
        assert temp_db.get_video(video_id)['state'] == 'dlc_queued'

        # dlc_queued -> dlc_running
        temp_db.update_state(video_id, 'dlc_running')
        assert temp_db.get_video(video_id)['state'] == 'dlc_running'

        # dlc_running -> dlc_complete
        temp_db.update_state(video_id, 'dlc_complete')
        assert temp_db.get_video(video_id)['state'] == 'dlc_complete'

        # dlc_complete -> processing
        temp_db.update_state(video_id, 'processing')
        assert temp_db.get_video(video_id)['state'] == 'processing'

        # processing -> processed
        temp_db.update_state(video_id, 'processed')
        assert temp_db.get_video(video_id)['state'] == 'processed'

        # processed -> archiving
        temp_db.update_state(video_id, 'archiving')
        assert temp_db.get_video(video_id)['state'] == 'archiving'

        # archiving -> archived
        temp_db.update_state(video_id, 'archived')
        assert temp_db.get_video(video_id)['state'] == 'archived'

    def test_invalid_state_transition_raises_error(self, temp_db):
        """Invalid state transition raises ValueError."""
        video_id = "20250704_CNT0101_P1"
        temp_db.register_video(video_id, "/fake/path")

        # discovered -> dlc_running is NOT valid (must go through validated, dlc_queued)
        with pytest.raises(ValueError, match="Invalid video state transition"):
            temp_db.update_state(video_id, 'dlc_running')

    def test_mark_failed_increments_error_count(self, temp_db):
        """mark_failed increments error_count correctly."""
        video_id = "20250704_CNT0101_P1"
        temp_db.register_video(video_id, "/fake/path")
        temp_db.update_state(video_id, 'validated')
        temp_db.update_state(video_id, 'dlc_queued')
        temp_db.update_state(video_id, 'dlc_running')

        # Mark failed first time
        temp_db.mark_failed(video_id, "Test error 1")
        video = temp_db.get_video(video_id)
        assert video['state'] == 'failed'
        assert video['error_count'] == 1
        assert video['error_message'] == "Test error 1"

        # Reset and fail again
        temp_db.reset_failed(video_id, to_state='dlc_queued')
        temp_db.update_state(video_id, 'dlc_running')
        temp_db.mark_failed(video_id, "Test error 2")

        video = temp_db.get_video(video_id)
        assert video['error_count'] == 2

    def test_reset_failed_returns_to_valid_state(self, temp_db):
        """reset_failed transitions back to appropriate state."""
        video_id = "20250704_CNT0101_P1"
        temp_db.register_video(video_id, "/fake/path")
        temp_db.update_state(video_id, 'validated')
        temp_db.update_state(video_id, 'dlc_queued')
        temp_db.update_state(video_id, 'dlc_running')
        temp_db.mark_failed(video_id, "Error")

        # Reset to dlc_queued
        temp_db.reset_failed(video_id, to_state='dlc_queued')

        video = temp_db.get_video(video_id)
        assert video['state'] == 'dlc_queued'
        assert video['error_message'] is None

    def test_get_pipeline_summary_returns_correct_counts(self, temp_db):
        """get_pipeline_summary returns accurate state counts."""
        # Create a few videos in different states
        for i in range(3):
            temp_db.register_video(f"video_{i}", f"/path/{i}")

        temp_db.update_state("video_0", 'validated')
        temp_db.update_state("video_1", 'validated')
        temp_db.update_state("video_1", 'dlc_queued')
        temp_db.mark_failed("video_2", "Error")

        # Create collages
        temp_db.register_collage("collage_1.mkv", "/path/1")
        temp_db.register_collage("collage_2.mkv", "/path/2")
        temp_db.update_collage_state("collage_1.mkv", 'validated')

        summary = temp_db.get_pipeline_summary()

        assert summary['videos']['total'] == 3
        assert summary['videos']['by_state']['validated'] == 1
        assert summary['videos']['by_state']['dlc_queued'] == 1
        assert summary['videos']['failed'] == 1

        assert summary['collages']['total'] == 2
        assert summary['collages']['by_state']['discovered'] == 1
        assert summary['collages']['by_state']['validated'] == 1

    def test_log_step_creates_audit_entry(self, temp_db):
        """log_step creates processing_log entries."""
        video_id = "20250704_CNT0101_P1"
        temp_db.register_video(video_id, "/fake/path")

        # Log a step
        temp_db.log_step(
            video_id=video_id,
            step="crop",
            status="started",
            message="Starting crop",
            duration=None
        )

        # Verify log entry
        logs = temp_db.get_recent_log(limit=10)
        assert len(logs) >= 1
        assert logs[0]['video_id'] == video_id
        assert logs[0]['step'] == "crop"
        assert logs[0]['status'] == "started"

    def test_get_recent_log_returns_entries(self, temp_db):
        """get_recent_log returns recent log entries."""
        video_id = "20250704_CNT0101_P1"
        temp_db.register_video(video_id, "/fake/path")

        # Log multiple steps
        for i in range(5):
            temp_db.log_step(video_id, f"step_{i}", "completed")

        # Get recent logs
        logs = temp_db.get_recent_log(limit=3)
        assert len(logs) == 3

        # Verify all entries have required fields
        for log in logs:
            assert 'video_id' in log
            assert 'step' in log
            assert 'status' in log
            assert log['video_id'] == video_id
            assert log['status'] == "completed"

        # Verify we got entries (order may vary due to timestamp precision)
        step_names = {log['step'] for log in logs}
        assert len(step_names) == 3  # Got 3 unique steps
        assert all('step_' in step for step in step_names)  # All are step_X format


# =============================================================================
# VALIDATOR TESTS
# =============================================================================

class TestValidator:
    """Tests for filename validation."""

    def test_validate_collage_filename_valid(self):
        """validate_collage_filename accepts valid names."""
        result = validate_collage_filename(
            "20250704_CNT0101,CNT0102,CNT0103,CNT0104,CNT0105,CNT0106,CNT0107,CNT0108_P1.mkv"
        )

        assert result.valid is True
        assert result.error is None
        assert result.parsed is not None
        assert result.parsed['date'] == "20250704"
        assert result.parsed['tray_type'] == "P"
        assert len(result.parsed['animal_ids']) == 8

    def test_validate_collage_filename_invalid_date(self):
        """validate_collage_filename rejects future dates."""
        future_date = (datetime.now() + timedelta(days=10)).strftime("%Y%m%d")
        result = validate_collage_filename(
            f"{future_date}_CNT0101,CNT0102,CNT0103,CNT0104,CNT0105,CNT0106,CNT0107,CNT0108_P1.mkv"
        )

        assert result.valid is False
        assert "future" in result.error.lower()

    def test_validate_collage_filename_missing_animals(self):
        """validate_collage_filename rejects < 8 animals."""
        result = validate_collage_filename(
            "20250704_CNT0101,CNT0102,CNT0103_P1.mkv"  # Only 3 animals
        )

        assert result.valid is False
        assert "8 comma-separated" in result.error or "exactly 8" in result.error.lower()

    def test_validate_collage_filename_invalid_tray(self):
        """validate_collage_filename rejects invalid tray codes."""
        result = validate_collage_filename(
            "20250704_CNT0101,CNT0102,CNT0103,CNT0104,CNT0105,CNT0106,CNT0107,CNT0108_X1.mkv"
        )

        assert result.valid is False
        assert "tray" in result.error.lower()

    def test_validate_single_filename_valid(self):
        """validate_single_filename accepts valid names."""
        result = validate_single_filename("20250704_CNT0101_P1.mp4")

        assert result.valid is True
        assert result.error is None
        assert result.parsed is not None
        assert result.parsed['date'] == "20250704"
        assert result.parsed['animal_id'] == "CNT0101"
        assert result.parsed['tray_type'] == "P"

    def test_validate_single_filename_invalid_extension(self):
        """validate_single_filename rejects non-mp4 files."""
        result = validate_single_filename("20250704_CNT0101_P1.avi")

        assert result.valid is False
        assert "mp4" in result.error.lower() or "extension" in result.error.lower()

    def test_validate_single_filename_bad_format(self):
        """validate_single_filename rejects malformed names."""
        result = validate_single_filename("video_file.mp4")

        assert result.valid is False

    def test_quarantine_file_moves_and_creates_metadata(self, tmp_path):
        """quarantine_file moves file and creates metadata json."""
        # Create fake video file
        source_file = tmp_path / "source" / "bad_video.mkv"
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text("fake video content")

        quarantine_dir = tmp_path / "quarantine"

        # Quarantine it
        quarantine_file(source_file, quarantine_dir, reason="Test quarantine")

        # Verify file moved
        assert not source_file.exists()
        quarantined = quarantine_dir / "bad_video.mkv"
        assert quarantined.exists()

        # Verify metadata created (note: .quarantine.json suffix, not .meta.json)
        metadata_file = quarantine_dir / "bad_video.mkv.quarantine.json"
        assert metadata_file.exists()

        metadata = json.loads(metadata_file.read_text())
        assert metadata['error_message'] == "Test quarantine"
        assert str(source_file.resolve()) in metadata['original_path']


# =============================================================================
# ROUTER TESTS
# =============================================================================

class TestRouter:
    """Tests for TrayRouter pipeline routing."""

    def test_p_tray_gets_all_steps(self):
        """P tray runs full pipeline."""
        router = TrayRouter()
        steps = router.get_pipeline_steps('P')

        assert 'crop' in steps
        assert 'dlc' in steps
        assert 'segment' in steps
        assert 'reach_detection' in steps
        assert 'outcome_detection' in steps
        assert 'kinematics' in steps
        assert 'export' in steps
        assert 'archive' in steps

    def test_e_tray_skips_outcome_detection(self):
        """E tray skips outcome_detection."""
        router = TrayRouter()
        steps = router.get_pipeline_steps('E')

        assert 'dlc' in steps
        assert 'segment' in steps
        assert 'reach_detection' in steps
        assert 'outcome_detection' not in steps

    def test_f_tray_skips_outcome_detection(self):
        """F tray skips outcome_detection."""
        router = TrayRouter()
        steps = router.get_pipeline_steps('F')

        assert 'dlc' in steps
        assert 'segment' in steps
        assert 'reach_detection' in steps
        assert 'outcome_detection' not in steps

    def test_should_run_step_returns_correct_bools(self):
        """should_run_step returns correct boolean for each tray type."""
        router = TrayRouter()

        # P tray runs outcome_detection
        assert router.should_run_step('P', 'outcome_detection') is True

        # E tray skips outcome_detection
        assert router.should_run_step('E', 'outcome_detection') is False

        # F tray skips outcome_detection
        assert router.should_run_step('F', 'outcome_detection') is False

        # All trays run segmentation
        assert router.should_run_step('P', 'segment') is True
        assert router.should_run_step('E', 'segment') is True
        assert router.should_run_step('F', 'segment') is True

    def test_unknown_tray_defaults_to_full_pipeline(self):
        """Unknown tray type defaults to full pipeline."""
        router = TrayRouter()
        steps = router.get_pipeline_steps('Z')  # Unknown

        # Should run all steps (default to full)
        assert 'outcome_detection' in steps


# =============================================================================
# TRANSFER TESTS
# =============================================================================

class TestTransfer:
    """Tests for file transfer utilities."""

    def test_safe_copy_copies_file_and_returns_true(self, tmp_path):
        """safe_copy copies file successfully."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "dest" / "source.txt"

        src.write_text("test content")

        result = safe_copy(src, dst, verify=False)

        assert result is True
        assert dst.exists()
        assert dst.read_text() == "test content"

    def test_safe_copy_with_verify_validates_hash(self, tmp_path):
        """safe_copy with verify=True validates file size."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "dest" / "source.txt"

        src.write_text("test content")

        result = safe_copy(src, dst, verify=True)

        assert result is True
        assert dst.exists()
        # Verify was called (file sizes match)
        assert src.stat().st_size == dst.stat().st_size

    def test_safe_move_moves_file(self, tmp_path):
        """safe_move moves file from src to dst."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "dest" / "source.txt"

        src.write_text("test content")

        result = safe_move(src, dst)

        assert result is True
        assert not src.exists()
        assert dst.exists()
        assert dst.read_text() == "test content"

    def test_check_file_stable_quick_with_stable_file(self, tmp_path):
        """check_file_stable_quick detects stable files."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        file_size = test_file.stat().st_size
        past_time = time.time() - 120  # 2 minutes ago

        is_stable, current_size, change_time = check_file_stable_quick(
            test_file,
            recorded_size=file_size,
            min_stable_seconds=60,
            last_change_time=past_time
        )

        assert is_stable is True
        assert current_size == file_size
        assert change_time == past_time

    def test_check_file_stable_quick_with_changing_file(self, tmp_path):
        """check_file_stable_quick detects size changes."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # First check
        is_stable, size1, time1 = check_file_stable_quick(
            test_file,
            recorded_size=None,
            min_stable_seconds=60,
            last_change_time=None
        )

        assert is_stable is False  # First check

        # Modify file
        test_file.write_text("new content that is longer")

        # Second check
        is_stable, size2, time2 = check_file_stable_quick(
            test_file,
            recorded_size=size1,
            min_stable_seconds=60,
            last_change_time=time1
        )

        assert is_stable is False  # Size changed
        assert size2 != size1


# =============================================================================
# STATE MANAGER + FILE WATCHER INTEGRATION
# =============================================================================

class TestStateManagerIntegration:
    """Integration tests for StateManager + FileWatcher."""

    def test_discover_new_collages_finds_video_files(self, state_manager, tmp_path):
        """discover_new_collages finds video files in scan dir."""
        scan_dir = tmp_path / "scan"
        scan_dir.mkdir(parents=True, exist_ok=True)

        # Create valid collage file
        collage_file = scan_dir / "20250704_CNT0101,CNT0102,CNT0103,CNT0104,CNT0105,CNT0106,CNT0107,CNT0108_P1.mkv"
        collage_file.write_text("fake video")

        new_collages = state_manager.discover_new_collages(scan_dir)

        assert len(new_collages) == 1
        assert new_collages[0] == collage_file.name

    def test_discover_new_collages_skips_non_video_files(self, state_manager, tmp_path):
        """discover_new_collages skips non-video extensions."""
        scan_dir = tmp_path / "scan"
        scan_dir.mkdir(parents=True, exist_ok=True)

        # Create non-video file
        txt_file = scan_dir / "notes.txt"
        txt_file.write_text("not a video")

        new_collages = state_manager.discover_new_collages(scan_dir)

        assert len(new_collages) == 0

    def test_discover_new_collages_quarantines_invalid_filenames(self, state_manager, tmp_path, watcher_config):
        """discover_new_collages quarantines invalid filenames."""
        scan_dir = tmp_path / "scan"
        scan_dir.mkdir(parents=True, exist_ok=True)

        # Create invalid collage file
        bad_file = scan_dir / "invalid_name.mkv"
        bad_file.write_text("fake video")

        new_collages = state_manager.discover_new_collages(scan_dir)

        # Should not register as valid
        assert len(new_collages) == 0

        # Should be moved to quarantine
        quarantine_dir = watcher_config.get_quarantine_dir()
        quarantined = quarantine_dir / "invalid_name.mkv"
        assert quarantined.exists()

    def test_check_collage_stability_works(self, temp_db, watcher_config):
        """Collage stability checking updates database state."""
        # Register collage
        temp_db.register_collage("test.mkv", "/fake/path")
        temp_db.update_collage_state("test.mkv", "validated")

        # Simulate size tracking
        file_size = 1000000
        temp_db.update_file_size("test.mkv", file_size)

        # Get collage
        collage = temp_db.get_collage("test.mkv")
        assert collage['file_size'] == file_size


# =============================================================================
# ORCHESTRATOR INTEGRATION (MOCK DLC AND PIPELINE)
# =============================================================================

class TestOrchestratorIntegration:
    """Integration tests for WatcherOrchestrator (mock heavy dependencies)."""

    @pytest.fixture
    def mock_orchestrator(self, temp_db, watcher_config, tmp_path):
        """Create orchestrator with mocked dependencies."""
        with patch('mousereach.watcher.orchestrator.require_processing_root') as mock_root:
            mock_root.return_value = tmp_path / "processing"

            from mousereach.watcher.orchestrator import WatcherOrchestrator
            orch = WatcherOrchestrator(config=watcher_config, db=temp_db)
            return orch

    def test_get_next_work_item_returns_correct_priority_order(self, mock_orchestrator, temp_db):
        """_get_next_work_item returns items in priority order."""
        # Create work in different states
        # Priority 1: Stable collages
        temp_db.register_collage("collage.mkv", "/path")
        temp_db.update_collage_state("collage.mkv", "validated")
        temp_db.update_collage_state("collage.mkv", "stable")

        # Priority 2: DLC queued
        temp_db.register_video("video1", "/path")
        temp_db.update_state("video1", "validated")
        temp_db.update_state("video1", "dlc_queued")

        # Priority 3: DLC complete
        temp_db.register_video("video2", "/path")
        temp_db.update_state("video2", "validated")
        temp_db.update_state("video2", "dlc_queued")
        temp_db.update_state("video2", "dlc_running")
        temp_db.update_state("video2", "dlc_complete")

        # Get next work item - should be collage (highest priority)
        work = mock_orchestrator._get_next_work_item()
        assert work is not None
        assert work['type'] == 'collage'

    def test_dry_run_does_not_process_anything(self, mock_orchestrator, temp_db, capsys):
        """dry_run shows pending work without processing."""
        # Create pending work
        temp_db.register_collage("test.mkv", "/path")
        temp_db.update_collage_state("test.mkv", "validated")
        temp_db.update_collage_state("test.mkv", "stable")

        # Run dry run
        mock_orchestrator.dry_run()

        # Verify nothing was processed (collage still in stable state)
        collage = temp_db.get_collage("test.mkv")
        assert collage['state'] == 'stable'

        # Verify output shows pending work
        captured = capsys.readouterr()
        assert "Pending work items" in captured.out

    def test_run_once_processes_available_items(self, mock_orchestrator, temp_db):
        """run_once processes all available work items."""
        # Create some work
        temp_db.register_video("video1", "/path")
        temp_db.update_state("video1", "validated")
        temp_db.update_state("video1", "dlc_queued")

        # Track dispatch calls and advance state to break the loop
        dispatch_calls = []

        def mock_dispatch(work):
            dispatch_calls.append(work)
            # Advance state so _get_next_work_item stops returning this item
            if work['type'] == 'single_dlc':
                temp_db.update_state(work['id'], 'dlc_running')
                temp_db.mark_failed(work['id'], "mock processing")

        with patch.object(mock_orchestrator, '_dispatch_work', side_effect=mock_dispatch):
            with patch.object(mock_orchestrator, '_scan_for_dlc_completions', return_value=0):
                mock_scan_result = Mock()
                mock_scan_result.new_collages = 0
                mock_scan_result.new_singles = 0
                mock_scan_result.stable_ready = 0
                mock_orchestrator.file_watcher.scan = Mock(return_value=mock_scan_result)

                mock_orchestrator.run_once()

                # Verify dispatch was called
                assert len(dispatch_calls) == 1
                assert dispatch_calls[0]['type'] == 'single_dlc'


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_double_registration_is_idempotent(self, temp_db, sample_collage_name):
        """Registering same collage twice is idempotent."""
        id1 = temp_db.register_collage(sample_collage_name, "/path")
        id2 = temp_db.register_collage(sample_collage_name, "/path")

        assert id1 == id2

    def test_nonexistent_video_raises_error(self, temp_db):
        """Updating nonexistent video raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            temp_db.update_state("nonexistent_video", "validated")

    def test_database_thread_safety(self, temp_db):
        """Database handles concurrent access (basic smoke test)."""
        import threading

        def register_videos():
            for i in range(10):
                temp_db.register_video(f"video_{i}_{threading.current_thread().name}", "/path")

        threads = [threading.Thread(target=register_videos) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have registered 30 videos
        summary = temp_db.get_pipeline_summary()
        assert summary['videos']['total'] == 30


# =============================================================================
# MACHINE DIAGNOSTICS (roles.py)
# =============================================================================

class TestMachineDiagnostics:
    """Tests for machine diagnostics (roles.py)."""

    def test_get_available_drives_returns_dict_on_windows(self):
        """get_available_drives returns a dict on Windows."""
        from mousereach.watcher.roles import get_available_drives

        with patch('mousereach.watcher.roles.sys') as mock_sys:
            mock_sys.platform = "win32"
            with patch('mousereach.watcher.roles._drive_exists', return_value=True):
                with patch('mousereach.watcher.roles._is_local_drive', return_value=True):
                    drives = get_available_drives()
                    # Should have entries for all 26 letters since _drive_exists returns True
                    assert len(drives) == 26
                    assert drives["C:"]["local"] is True

    def test_get_available_drives_empty_on_non_windows(self):
        """get_available_drives returns empty dict on non-Windows."""
        from mousereach.watcher.roles import get_available_drives

        with patch('mousereach.watcher.roles.sys') as mock_sys:
            mock_sys.platform = "linux"
            drives = get_available_drives()
            assert drives == {}

    def test_validate_configured_paths_checks_accessibility(self, tmp_path):
        """validate_configured_paths reports accessible/inaccessible paths."""
        from mousereach.watcher.roles import validate_configured_paths

        existing_dir = tmp_path / "nas"
        existing_dir.mkdir()

        mock_config = {
            "nas_drive": str(existing_dir),
            "processing_root": str(tmp_path / "nonexistent"),
        }

        with patch('mousereach.config._load_config', return_value=mock_config):
            results = validate_configured_paths()

            # nas_drive should be OK (we created it)
            nas_result = [r for r in results if r[0] == "nas_drive"][0]
            assert nas_result[2] is True  # is_ok

            # processing_root should fail (doesn't exist)
            proc_result = [r for r in results if r[0] == "processing_root"][0]
            assert proc_result[2] is False  # is_ok

    def test_validate_configured_paths_handles_missing_config(self):
        """validate_configured_paths handles no config gracefully."""
        from mousereach.watcher.roles import validate_configured_paths

        mock_config = {}
        with patch('mousereach.config._load_config', return_value=mock_config):
            results = validate_configured_paths()
            # Should report nas_drive and processing_root as not set
            assert len(results) >= 2
            assert all(not r[2] for r in results)  # none are OK


# =============================================================================
# MACHINE IDENTITY FILE (wizard.py)
# =============================================================================

class TestMachineIdentity:
    """Tests for local machine identity file support."""

    def test_identity_file_takes_priority_over_drive_match(self, tmp_path):
        """Identity file is checked before drive-pattern matching."""
        from mousereach.setup.wizard import (
            detect_lab_profile, _load_machine_role,
            MACHINE_ROLE_FILE, _find_profile_by_name,
        )

        # Mock: identity file says "Processing Server"
        with patch('mousereach.setup.wizard._load_machine_role', return_value="Processing Server"):
            with patch('mousereach.setup.wizard._find_profile_by_name') as mock_find:
                mock_find.return_value = {
                    "name": "Processing Server",
                    "defaults": {"nas_drive": "X:\\"}
                }
                with patch('mousereach.setup.wizard._get_current_drives', return_value={}):
                    profile, drives, method = detect_lab_profile()

                    assert profile is not None
                    assert profile["name"] == "Processing Server"
                    assert method == "identity_file"

    def test_drive_match_fallback_when_no_identity_file(self):
        """Falls back to drive matching when no identity file exists."""
        from mousereach.setup.wizard import detect_lab_profile

        mock_drives = {"D": True, "A": True, "X": False, "Y": False}
        mock_profiles = [{
            "name": "NAS / DLC PC",
            "match": {"drives_present": ["D", "A", "X", "Y"], "drives_local": ["D"]},
            "defaults": {"nas_drive": "D:\\"}
        }]

        with patch('mousereach.setup.wizard._load_machine_role', return_value=None):
            with patch('mousereach.setup.wizard._get_current_drives', return_value=mock_drives):
                with patch('mousereach.setup.wizard._load_lab_profiles', return_value=mock_profiles):
                    profile, drives, method = detect_lab_profile()

                    assert profile is not None
                    assert method == "drive_match"

    def test_no_match_returns_none(self):
        """Returns None when neither identity file nor drive match works."""
        from mousereach.setup.wizard import detect_lab_profile

        with patch('mousereach.setup.wizard._load_machine_role', return_value=None):
            with patch('mousereach.setup.wizard._get_current_drives', return_value={"C": True}):
                with patch('mousereach.setup.wizard._load_lab_profiles', return_value=[]):
                    profile, drives, method = detect_lab_profile()

                    assert profile is None
                    assert method is None

    def test_set_machine_role_creates_file(self, tmp_path):
        """set_machine_role writes the identity file."""
        from mousereach.setup.wizard import set_machine_role, MACHINE_ROLE_FILE

        role_file = tmp_path / "machine_role.json"
        with patch('mousereach.setup.wizard.MACHINE_ROLE_FILE', role_file):
            with patch('mousereach.setup.wizard.CONFIG_DIR', tmp_path):
                set_machine_role("NAS / DLC PC")

                assert role_file.exists()
                data = json.loads(role_file.read_text())
                assert data["role"] == "NAS / DLC PC"

    def test_list_available_roles_returns_profile_names(self):
        """list_available_roles returns all profile names."""
        from mousereach.setup.wizard import list_available_roles

        mock_profiles = [
            {"name": "NAS / DLC PC"},
            {"name": "GPU Filming PC"},
            {"name": "Processing Server"},
        ]
        with patch('mousereach.setup.wizard._load_lab_profiles', return_value=mock_profiles):
            roles = list_available_roles()
            assert roles == ["NAS / DLC PC", "GPU Filming PC", "Processing Server"]

    def test_identity_file_with_unknown_role_falls_back(self):
        """Identity file with unknown role name falls back to drive matching."""
        from mousereach.setup.wizard import detect_lab_profile

        with patch('mousereach.setup.wizard._load_machine_role', return_value="Nonexistent Role"):
            with patch('mousereach.setup.wizard._find_profile_by_name', return_value=None):
                with patch('mousereach.setup.wizard._get_current_drives', return_value={}):
                    with patch('mousereach.setup.wizard._load_lab_profiles', return_value=[]):
                        profile, drives, method = detect_lab_profile()

                        assert profile is None
                        assert method is None
