"""
Single-machine pipeline orchestrator for the MouseReach watcher.

Coordinates: watcher scanning -> crop -> DLC -> post-DLC pipeline -> archive.
Phase 1 runs everything on one machine.
"""

import time
import socket
import logging
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime

from mousereach.watcher.db import WatcherDB
from mousereach.watcher.state import WatcherStateManager
from mousereach.watcher.watcher import FileWatcher
from mousereach.watcher.router import TrayRouter
from mousereach.watcher.transfer import safe_copy, safe_move
from mousereach.config import (
    Paths, WatcherConfig, require_processing_root, parse_tray_type,
    get_video_id, AnimalID
)

logger = logging.getLogger(__name__)


class WatcherOrchestrator:
    """
    Phase 1 single-machine orchestrator.

    Coordinates all pipeline steps on one machine:
    - Scans NAS for new collages (via FileWatcher + StateManager)
    - Crops to singles
    - Runs DLC inference automatically
    - Post-DLC pipeline (segment, reach, outcome/kinematics)
    - Archive when ready
    """

    def __init__(self, config: WatcherConfig, db: WatcherDB):
        """
        Initialize orchestrator.

        Args:
            config: WatcherConfig instance
            db: WatcherDB instance
        """
        self.config = config
        self.db = db
        self.router = TrayRouter()
        self.hostname = socket.gethostname()

        # Create state manager and file watcher
        self.state = WatcherStateManager(db, config)
        self.file_watcher = FileWatcher(config, self.state)

        # Working directory for temporary crop operations
        self.working_dir = require_processing_root() / "watcher_working"
        self.working_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"WatcherOrchestrator initialized on {self.hostname}")
        logger.info(f"Working directory: {self.working_dir}")

    def run(self, shutdown_event: threading.Event):
        """
        Main orchestrator loop: scan + process in a single thread.

        Args:
            shutdown_event: Threading event to signal shutdown
        """
        logger.info("WatcherOrchestrator starting main loop")

        while not shutdown_event.is_set():
            try:
                # Phase A: Scan for new files and check stability
                scan_result = self.file_watcher.scan()
                if scan_result.new_collages or scan_result.new_singles or scan_result.stable_ready:
                    logger.info(
                        f"Scan: {scan_result.new_collages} new collages, "
                        f"{scan_result.new_singles} new singles, "
                        f"{scan_result.stable_ready} now stable"
                    )

                # Phase B: Check for DLC completions (h5 files appearing)
                self._scan_for_dlc_completions()

                # Phase C: Process the highest-priority work item
                work = self._get_next_work_item()

                if work is None:
                    # Nothing to do - sleep and retry
                    shutdown_event.wait(timeout=self.config.poll_interval_seconds)
                    continue

                # Dispatch work to appropriate handler
                self._dispatch_work(work)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                # Sleep before retrying to avoid tight error loops
                time.sleep(5)

        self.shutdown()
        logger.info("WatcherOrchestrator stopped")

    def run_once(self):
        """
        Run one full cycle: scan + process all pending items, then exit.

        Useful for testing or cron-based scheduling.
        """
        logger.info("WatcherOrchestrator running once")

        # Scan for new files
        scan_result = self.file_watcher.scan()
        logger.info(
            f"Scan: {scan_result.new_collages} new collages, "
            f"{scan_result.new_singles} new singles, "
            f"{scan_result.stable_ready} now stable"
        )

        # Check for DLC completions
        self._scan_for_dlc_completions()

        # Process all available work items
        processed = 0
        while True:
            work = self._get_next_work_item()
            if work is None:
                break
            self._dispatch_work(work)
            processed += 1

        logger.info(f"Run-once complete: processed {processed} items")

    def dry_run(self):
        """
        Scan without processing. Shows what would be done.
        """
        logger.info("WatcherOrchestrator dry run")

        # Scan for new files
        scan_result = self.file_watcher.scan()
        print(f"\nScan results:")
        print(f"  New collages found:   {scan_result.new_collages}")
        print(f"  New singles found:    {scan_result.new_singles}")
        print(f"  Collages now stable:  {scan_result.stable_ready}")
        print(f"  Scan time:            {scan_result.scan_time_ms:.1f}ms")

        # Check for DLC completions
        dlc_completions = self._scan_for_dlc_completions()
        print(f"  DLC completions:      {dlc_completions}")

        # Show what work would be done
        print(f"\nPending work items:")
        count = 0
        # Don't actually process - just list what's available
        for state_label, state in [
            ("Collages to crop", "stable"),
            ("Videos for DLC", "dlc_queued"),
            ("Videos for pipeline", "dlc_complete"),
            ("Videos for archive", "processed"),
        ]:
            if state in ("stable",):
                items = self.db.get_collages_in_state(state)
            else:
                items = self.db.get_videos_in_state(state)
            if items:
                print(f"  {state_label}: {len(items)}")
                for item in items[:5]:
                    name = item.get('filename') or item.get('video_id')
                    print(f"    - {name}")
                if len(items) > 5:
                    print(f"    ... and {len(items) - 5} more")
                count += len(items)

        if count == 0:
            print("  (no pending work)")

        print()

    def _get_next_work_item(self):
        """
        Get next work item from database in priority order.

        Priority:
        1. Collages stable and ready to crop
        2. Videos in dlc_queued (need DLC inference)
        3. Videos with DLC complete (need post-DLC pipeline)
        4. Videos processed (ready for archive)

        Returns:
            Dict with work item details, or None if nothing ready
        """
        # Priority 1: Collages ready to crop
        collages = self.db.get_collages_in_state('stable')
        if collages:
            return {
                'type': 'collage',
                'id': collages[0]['filename'],
                'data': collages[0]
            }

        # Priority 2: Videos queued for DLC
        videos = self.db.get_videos_in_state('dlc_queued')
        if videos:
            return {
                'type': 'single_dlc',
                'id': videos[0]['video_id'],
                'data': videos[0]
            }

        # Priority 3: Videos with DLC complete, ready for pipeline
        videos = self.db.get_videos_in_state('dlc_complete')
        if videos:
            return {
                'type': 'single_pipeline',
                'id': videos[0]['video_id'],
                'data': videos[0]
            }

        # Priority 4: Videos processed and ready for archive
        if self.config.auto_archive_approved:
            videos = self.db.get_videos_in_state('processed')
            if videos:
                return {
                    'type': 'archive',
                    'id': videos[0]['video_id'],
                    'data': videos[0]
                }

        return None

    def _dispatch_work(self, work: dict):
        """
        Route work to appropriate handler.

        Args:
            work: Work item dict with 'type', 'id', 'data'
        """
        work_type = work['type']
        work_id = work['id']

        try:
            if work_type == 'collage':
                self._process_collage(work)
            elif work_type == 'single_dlc':
                self._process_single_dlc(work)
            elif work_type == 'single_pipeline':
                self._process_single_pipeline(work)
            elif work_type == 'archive':
                self._process_archive(work)
            else:
                logger.warning(f"Unknown work type: {work_type}")

        except Exception as e:
            error_msg = f"{work_type} failed: {str(e)}"
            logger.error(f"Work item {work_id} failed: {e}", exc_info=True)

            # Mark as failed in DB
            if work_type == 'collage':
                self.db.update_collage_state(work_id, 'failed', validation_error=error_msg)
            else:
                self.db.mark_failed(work_id, error_msg)

    def _scan_for_dlc_completions(self) -> int:
        """
        Scan DLC_Complete for h5 files matching videos in dlc_queued or dlc_running state.

        When DLC finishes (either via auto-DLC or manual mousereach-dlc-batch),
        h5 files appear in DLC_Complete/. This method detects them and advances
        the video state to dlc_complete.

        Returns:
            Number of DLC completions detected
        """
        dlc_complete_dir = Paths.DLC_COMPLETE if hasattr(Paths, 'DLC_COMPLETE') else None
        if not dlc_complete_dir or not dlc_complete_dir.exists():
            return 0

        count = 0

        # Check all videos waiting for DLC
        for state in ('dlc_queued', 'dlc_running'):
            videos = self.db.get_videos_in_state(state)
            for video in videos:
                video_id = video['video_id']
                # Look for DLC h5 output
                h5_files = list(dlc_complete_dir.glob(f"{video_id}DLC*.h5"))
                if h5_files:
                    dlc_path = h5_files[0]
                    logger.info(f"DLC completion detected: {video_id} -> {dlc_path.name}")

                    # Advance state
                    if state == 'dlc_queued':
                        # dlc_queued -> dlc_running -> dlc_complete
                        self.db.update_state(video_id, 'dlc_running')
                    self.db.update_state(
                        video_id, 'dlc_complete',
                        dlc_output_path=str(dlc_path),
                        current_path=str(dlc_path.parent / f"{video_id}.mp4")
                    )
                    self.db.log_step(video_id, 'dlc', 'completed', message=dlc_path.name)
                    count += 1

        return count

    def _process_collage(self, work: dict):
        """
        Process collage: crop to singles.

        Args:
            work: Work item with collage data
        """
        from mousereach.video_prep.core.cropper import crop_collage

        collage_filename = work['id']
        collage_data = work['data']

        logger.info(f"Processing collage: {collage_filename}")

        # Update state: stable -> cropping
        self.db.update_collage_state(collage_filename, 'cropping')
        self.db.log_step(collage_filename, 'crop', 'started')

        start_time = time.time()

        try:
            # Source path from DB
            source_path = Path(collage_data['source_path'])

            if not source_path.exists():
                raise FileNotFoundError(f"Collage not found: {source_path}")

            # Copy collage to working directory
            local_collage = self.working_dir / source_path.name
            logger.info(f"Copying collage to working dir: {source_path} -> {local_collage}")

            if not safe_copy(source_path, local_collage, verify=True):
                raise IOError(f"Failed to copy collage to working directory")

            # Crop collage to singles
            logger.info(f"Cropping collage: {collage_filename}")
            crop_results = crop_collage(
                input_path=local_collage,
                output_dir=self.working_dir,
                verbose=False
            )

            # Register each cropped single in DB
            videos_created = 0
            videos_skipped = 0

            for result in crop_results:
                if result['status'] == 'skipped':
                    videos_skipped += 1
                    logger.debug(f"Skipped position {result['position']}: {result.get('animal_id', '?')} (blank)")
                    continue

                if result['status'] != 'success':
                    logger.warning(f"Crop failed for position {result['position']}: {result.get('error')}")
                    continue

                # Parse metadata from filename
                output_path = Path(result['output_path'])
                video_id = get_video_id(output_path.name)

                # Parse animal ID
                animal_id = result.get('animal_id', '')
                parsed_animal = AnimalID.parse(animal_id) if animal_id else {}

                # Parse tray info
                tray_info = parse_tray_type(output_path.name)

                # Register video in DB
                self.db.register_video(
                    video_id=video_id,
                    source_path=str(output_path),
                    collage_id=collage_filename,
                    date=collage_data.get('date'),
                    animal_id=animal_id,
                    experiment=parsed_animal.get('experiment'),
                    cohort=parsed_animal.get('cohort'),
                    subject=parsed_animal.get('subject'),
                    tray_type=tray_info.get('tray_type'),
                    tray_position=result.get('position'),
                    current_path=str(output_path)
                )

                # Transition: discovered -> validated -> dlc_queued
                self.db.update_state(video_id, 'validated', current_path=str(output_path))

                # Copy single to DLC_Queue
                if Paths.DLC_QUEUE:
                    Paths.DLC_QUEUE.mkdir(parents=True, exist_ok=True)
                    dlc_queue_path = Paths.DLC_QUEUE / output_path.name
                    if safe_copy(output_path, dlc_queue_path, verify=True):
                        self.db.update_state(video_id, 'dlc_queued', current_path=str(dlc_queue_path))
                        logger.info(f"Created and queued: {video_id}")
                        videos_created += 1
                    else:
                        logger.error(f"Failed to copy {video_id} to DLC_Queue")
                else:
                    logger.error("DLC_QUEUE path not configured")

            # Update collage state: cropping -> cropped
            duration = time.time() - start_time
            self.db.update_collage_state(
                collage_filename,
                'cropped',
                videos_created=videos_created,
                videos_skipped=videos_skipped
            )
            self.db.log_step(
                collage_filename,
                'crop',
                'completed',
                message=f"Created {videos_created} singles, skipped {videos_skipped}",
                duration=duration
            )

            logger.info(f"Collage cropped: {collage_filename} ({videos_created} singles, {videos_skipped} skipped)")

            # Cleanup working directory
            local_collage.unlink(missing_ok=True)
            for result in crop_results:
                if result['status'] == 'success':
                    Path(result['output_path']).unlink(missing_ok=True)

        except Exception as e:
            duration = time.time() - start_time
            self.db.update_collage_state(collage_filename, 'failed', validation_error=str(e))
            self.db.log_step(collage_filename, 'crop', 'failed', message=str(e), duration=duration)
            raise

    def _process_single_dlc(self, work: dict):
        """
        Run DLC inference on a single video.

        Uses the DLC model configured in WatcherConfig.dlc_config_path.
        Falls back gracefully if DLC is not available.

        Args:
            work: Work item with video data
        """
        from mousereach.dlc.core import run_dlc_batch

        video_id = work['id']
        video_data = work['data']
        current_path = Path(video_data['current_path'])

        logger.info(f"Running DLC on {video_id}")

        # Validate DLC config is available
        if not self.config.dlc_config_path:
            logger.warning(
                f"DLC config not configured - {video_id} stays in dlc_queued. "
                "Run 'mousereach-setup' to set DLC model path, or run "
                "'mousereach-dlc-batch' manually."
            )
            return

        dlc_config = Path(self.config.dlc_config_path)
        if not dlc_config.exists():
            # Try appending config.yaml if a directory was given
            if dlc_config.is_dir():
                dlc_config = dlc_config / "config.yaml"
            if not dlc_config.exists():
                logger.error(f"DLC config not found: {self.config.dlc_config_path}")
                return

        # Ensure video file exists
        if not current_path.exists():
            logger.error(f"Video file not found: {current_path}")
            self.db.mark_failed(video_id, f"Video file not found: {current_path}")
            return

        # Update state: dlc_queued -> dlc_running
        self.db.update_state(video_id, 'dlc_running')
        self.db.log_step(video_id, 'dlc', 'started', message=f"GPU {self.config.dlc_gpu_device}")

        start_time = time.time()

        try:
            # Determine output directory
            dlc_output_dir = Paths.DLC_COMPLETE if hasattr(Paths, 'DLC_COMPLETE') else current_path.parent
            if dlc_output_dir:
                dlc_output_dir.mkdir(parents=True, exist_ok=True)

            # Run DLC inference
            results = run_dlc_batch(
                video_paths=[current_path],
                config_path=dlc_config,
                output_dir=dlc_output_dir,
                gpu=self.config.dlc_gpu_device,
                save_as_csv=True
            )

            duration = time.time() - start_time

            if results and results[0].get('status') == 'success':
                # Find the h5 output file
                h5_files = list(dlc_output_dir.glob(f"{video_id}DLC*.h5"))
                dlc_output = str(h5_files[0]) if h5_files else None

                # Update state: dlc_running -> dlc_complete
                self.db.update_state(
                    video_id, 'dlc_complete',
                    dlc_output_path=dlc_output,
                    current_path=str(dlc_output_dir / current_path.name)
                )
                self.db.log_step(
                    video_id, 'dlc', 'completed',
                    message=f"GPU {self.config.dlc_gpu_device}",
                    duration=duration
                )
                logger.info(f"DLC completed for {video_id} ({duration:.1f}s)")
            else:
                error_msg = results[0].get('error', 'Unknown DLC error') if results else 'No results'
                raise RuntimeError(f"DLC failed: {error_msg}")

        except Exception as e:
            duration = time.time() - start_time
            self.db.mark_failed(video_id, str(e))
            self.db.log_step(video_id, 'dlc', 'failed', message=str(e), duration=duration)
            raise

    def _process_single_pipeline(self, work: dict):
        """
        Process single video through post-DLC pipeline.

        Runs: segmentation -> reach detection -> outcome detection (tray-aware)

        Args:
            work: Work item with video data
        """
        from mousereach.pipeline.core import UnifiedPipelineProcessor

        video_id = work['id']
        video_data = work['data']

        logger.info(f"Running post-DLC pipeline for {video_id}")

        # Update state: dlc_complete -> processing
        self.db.update_state(video_id, 'processing')
        self.db.log_step(video_id, 'pipeline', 'started')

        start_time = time.time()

        try:
            # Find DLC file
            dlc_output_path = video_data.get('dlc_output_path')
            if dlc_output_path:
                dlc_path = Path(dlc_output_path)
            else:
                # Search for it
                current_path = Path(video_data['current_path'])
                dlc_files = list(current_path.parent.glob(f"{video_id}DLC*.h5"))
                if not dlc_files:
                    raise FileNotFoundError(f"No DLC file found for {video_id}")
                dlc_path = dlc_files[0]

            if not dlc_path.exists():
                raise FileNotFoundError(f"DLC file not found: {dlc_path}")

            # Parse tray type to determine if we should skip outcomes
            tray_type = video_data.get('tray_type')
            skip_outcomes = not self.router.should_run_step(tray_type or 'P', 'outcome_detection')

            if skip_outcomes:
                logger.info(f"Tray type {tray_type}: skipping outcome detection")

            # Run unified pipeline
            processor = UnifiedPipelineProcessor(
                base_dir=require_processing_root(),
                specific_files=[dlc_path],
                skip_outcomes=skip_outcomes
            )

            results = processor.run()

            # Update state: processing -> processed
            duration = time.time() - start_time

            # Build result summary
            msg_parts = []
            if hasattr(results, 'seg_processed'):
                msg_parts.append(f"Seg: {getattr(results, 'seg_auto_approved', 0)}/{results.seg_processed}")
            if hasattr(results, 'reach_processed'):
                msg_parts.append(f"Reaches: {getattr(results, 'reach_validated', 0)}/{results.reach_processed}")
            result_msg = ", ".join(msg_parts) if msg_parts else "completed"

            self.db.update_state(video_id, 'processed')
            self.db.log_step(
                video_id,
                'pipeline',
                'completed',
                message=result_msg,
                duration=duration
            )

            logger.info(f"Pipeline completed for {video_id} ({duration:.1f}s)")

            # Optional: Sync to MouseDB
            self._try_mousedb_sync(video_id)

        except Exception as e:
            duration = time.time() - start_time
            self.db.mark_failed(video_id, str(e))
            self.db.log_step(video_id, 'pipeline', 'failed', message=str(e), duration=duration)
            raise

    def _process_archive(self, work: dict):
        """
        Archive processed video to NAS.

        Only archives if auto_archive is enabled in config.

        Args:
            work: Work item with video data
        """
        video_id = work['id']
        video_data = work['data']

        logger.info(f"Archiving {video_id}")

        # Check if video is ready for archive
        try:
            from mousereach.archive.core import check_archive_ready
            if not check_archive_ready(video_id):
                logger.debug(f"Video {video_id} not ready for archive (validation incomplete)")
                return
        except ImportError:
            # Archive module may not have check_archive_ready yet
            logger.debug("Archive readiness check not available, proceeding with archive")

        # Update state: processed -> archiving
        self.db.update_state(video_id, 'archiving')
        self.db.log_step(video_id, 'archive', 'started')

        start_time = time.time()

        try:
            # Get all associated files
            current_path = Path(video_data['current_path'])
            all_files = self._get_associated_files(current_path.parent, video_id)

            # Determine archive destination
            experiment = video_data.get('experiment', 'UNKNOWN')
            if Paths.ANALYZED_OUTPUT:
                archive_dest = Paths.ANALYZED_OUTPUT / experiment
            else:
                raise ValueError("ANALYZED_OUTPUT path not configured")

            archive_dest.mkdir(parents=True, exist_ok=True)

            # Move all files to archive
            archived_files = []
            for file_path in all_files:
                dest_path = archive_dest / file_path.name
                if safe_move(file_path, dest_path):
                    archived_files.append(file_path.name)
                    logger.debug(f"Archived: {file_path.name}")
                else:
                    logger.warning(f"Failed to archive: {file_path.name}")

            # Update state: archiving -> archived
            duration = time.time() - start_time
            self.db.update_state(video_id, 'archived', current_path=str(archive_dest / current_path.name))
            self.db.log_step(
                video_id,
                'archive',
                'completed',
                message=f"Archived {len(archived_files)} files to {experiment}/",
                duration=duration
            )

            logger.info(f"Archived {video_id} to {experiment}/ ({len(archived_files)} files)")

        except Exception as e:
            duration = time.time() - start_time
            self.db.mark_failed(video_id, str(e))
            self.db.log_step(video_id, 'archive', 'failed', message=str(e), duration=duration)
            raise

    def _get_associated_files(self, directory: Path, video_id: str) -> list:
        """
        Get all files associated with a video.

        Args:
            directory: Directory to search
            video_id: Video identifier

        Returns:
            List of Path objects for associated files
        """
        files = []
        if directory.exists():
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.name.startswith(video_id):
                    files.append(file_path)
        return files

    def _try_mousedb_sync(self, video_id: str):
        """
        Try to sync to MouseDB (non-fatal if fails).

        Args:
            video_id: Video identifier
        """
        try:
            from mousereach.sync.database import sync_file_to_database

            # Find features file
            features_path = require_processing_root() / "Processing" / f"{video_id}_features.json"

            if features_path.exists():
                sync_file_to_database(features_path)
                self.db.log_step(video_id, 'mousedb_sync', 'completed')
                logger.info(f"Synced {video_id} to MouseDB")
            else:
                logger.debug(f"No features file for {video_id}, skipping MouseDB sync")

        except ImportError:
            logger.debug("mousedb not available, skipping sync")
        except Exception as e:
            logger.warning(f"MouseDB sync failed (non-fatal): {e}")
            self.db.log_step(video_id, 'mousedb_sync', 'failed', message=str(e))

    def shutdown(self):
        """Graceful shutdown."""
        logger.info("WatcherOrchestrator shutting down gracefully")
        # Cleanup any temporary files
        try:
            if self.working_dir.exists():
                for file_path in self.working_dir.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                        logger.debug(f"Cleaned up: {file_path.name}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
