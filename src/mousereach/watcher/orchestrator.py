"""
Role-aware orchestrators for the MouseReach watcher.

Two machines, two roles:

  DLCOrchestrator (mode="dlc_pc")
      NAS/DLC PC with GPU.  Scans NAS for collages -> crop -> DLC -> stage
      video+h5 to NAS DLC_Complete/ folder for the processing server.

  ProcessingOrchestrator (mode="processing_server")
      Processing server.  Watches DLC_Complete/ -> intake to local Processing/
      -> segmentation -> reach detection -> outcome detection -> archive to NAS.

Both share a BaseOrchestrator that provides the polling loop, priority animal
system, DB tracking, and graceful shutdown.
"""

import json
import os
import time
import random
import socket
import logging
import threading

# Ensure CUDA paths, cuDNN, and TF_USE_LEGACY_KERAS are set for GPU use
from mousereach.gpu import setup_gpu_env
setup_gpu_env()
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


# =============================================================================
# BASE ORCHESTRATOR
# =============================================================================

class BaseOrchestrator:
    """
    Shared infrastructure for all watcher orchestrator roles.

    Provides:
    - Polling loop (run, run_once)
    - Priority animal system
    - DB tracking and logging
    - Graceful shutdown
    """

    def __init__(self, config: WatcherConfig, db: WatcherDB):
        self.config = config
        self.db = db
        self.router = TrayRouter()
        self.hostname = socket.gethostname()

        # Create state manager and file watcher
        self.state = WatcherStateManager(db, config)
        self.file_watcher = FileWatcher(config, self.state)

        # Working directory for temporary operations
        self.working_dir = require_processing_root() / "watcher_working"
        self.working_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"{self.__class__.__name__} initialized on {self.hostname}")
        logger.info(f"Working directory: {self.working_dir}")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def _is_paused(self) -> bool:
        """Check if the watcher is paused (filming mode sentinel file exists)."""
        try:
            pause_file = require_processing_root() / "watcher_paused.flag"
            return pause_file.exists()
        except Exception:
            return False

    def run(self, shutdown_event: threading.Event):
        """Main orchestrator loop: scan + process in a single thread."""
        logger.info(f"{self.__class__.__name__} starting main loop")

        while not shutdown_event.is_set():
            try:
                # Check for pause sentinel (filming mode)
                if self._is_paused():
                    logger.info("Watcher PAUSED (filming mode) — run 'mousereach-watch-toggle' to resume.")
                    shutdown_event.wait(timeout=self.config.poll_interval_seconds)
                    continue

                # Phase A: Scan for new work
                self._scan_phase()

                # Phase B: Process the highest-priority work item
                work = self._get_next_work_item()

                if work is None:
                    shutdown_event.wait(timeout=self.config.poll_interval_seconds)
                    continue

                # Dispatch work to appropriate handler
                self._dispatch_work(work)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)

        self.shutdown()
        logger.info(f"{self.__class__.__name__} stopped")

    def run_once(self):
        """Run one full cycle: scan + process all pending items, then exit."""
        logger.info(f"{self.__class__.__name__} running once")

        self._scan_phase()

        processed = 0
        while True:
            work = self._get_next_work_item()
            if work is None:
                break
            self._dispatch_work(work)
            processed += 1

        logger.info(f"Run-once complete: processed {processed} items")

    # =========================================================================
    # ABSTRACT METHODS (subclasses must implement)
    # =========================================================================

    def _scan_phase(self):
        """Discover new work. Called once per cycle."""
        raise NotImplementedError

    def _get_next_work_item(self):
        """Return next work dict or None."""
        raise NotImplementedError

    def _dispatch_work(self, work: dict):
        """Route work to handler."""
        raise NotImplementedError

    def dry_run(self):
        """Scan without processing. Show what would be done."""
        raise NotImplementedError

    # =========================================================================
    # PRIORITY ANIMAL
    # =========================================================================

    def _get_priority_animal(self) -> Optional[str]:
        """
        Read priority animal from file, if set.

        The priority animal file is written by 'mousereach-watch-prioritize'
        and causes the watcher to prefer that animal's videos in all work queues.

        Returns:
            Animal ID string (e.g. "CNT0107") or None
        """
        priority_file = require_processing_root() / "priority_animal.json"
        if not priority_file.exists():
            return None
        try:
            with open(priority_file) as f:
                data = json.load(f)
            return data.get('animal_id')
        except Exception:
            return None

    def _matches_priority(self, item: dict, priority_animal: str,
                          animal_field: str) -> bool:
        """Check if a work item belongs to the priority animal."""
        animal_value = item.get(animal_field) or ''
        if ',' in animal_value:
            return priority_animal in animal_value.split(',')
        return animal_value == priority_animal

    def _pick_from_pool(self, items: list, priority_animal: Optional[str],
                        animal_field: str, is_collage: bool = False):
        """
        Pick a work item: priority animal first, then Pillar-first, random within tier.

        Args:
            items: All items in this work tier
            priority_animal: Animal ID to prefer, or None
            animal_field: DB field containing animal ID(s)
            is_collage: True if items are collages (filename-based pillar check)

        Returns:
            Selected item dict, or None if items is empty
        """
        if not items:
            return None

        # Split into priority animal's items and others
        if priority_animal:
            preferred = [i for i in items if self._matches_priority(i, priority_animal, animal_field)]
            if preferred:
                items = preferred  # Only pick from preferred

        # Pillar-first within selected pool
        if is_collage:
            pillar = [c for c in items if '_P' in c.get('filename', '')]
        else:
            pillar = [v for v in items if v.get('tray_type') == 'P']

        return random.choice(pillar if pillar else items)

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _get_associated_files(self, directory: Path, video_id: str) -> list:
        """Get all files associated with a video (mp4, h5, csv, etc.)."""
        files = []
        if directory.exists():
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.name.startswith(video_id):
                    files.append(file_path)
        return files

    def shutdown(self):
        """Graceful shutdown."""
        logger.info(f"{self.__class__.__name__} shutting down gracefully")
        try:
            if self.working_dir.exists():
                for file_path in self.working_dir.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                        logger.debug(f"Cleaned up: {file_path.name}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


# =============================================================================
# DLC ORCHESTRATOR (NAS / DLC PC)
# =============================================================================

class DLCOrchestrator(BaseOrchestrator):
    """
    NAS/DLC PC orchestrator.

    Processes one collage at a time:
    - Scans NAS for new collages
    - Crops to singles
    - Runs DLC inference on each single
    - Stages video+h5 back to NAS for the processing PC
    """

    def __init__(self, config: WatcherConfig, db: WatcherDB):
        super().__init__(config, db)

        # Staging directory on NAS for processing PC to pick up
        self.staging_dir = Paths.DLC_STAGING
        if self.staging_dir:
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Staging directory: {self.staging_dir}")

        # --- Cross-PC coordination via connectome.db ---
        self.coordinator = None
        try:
            from mousereach.watcher.coordination import (
                PipelineCoordinator, restore_db, backup_db
            )
            self._backup_db = backup_db
            self._restore_db = restore_db

            # Quick-restore watcher.db from NAS backup if local is empty
            nas_root = Paths.NAS_ROOT
            if nas_root:
                restore_db(db.db_path, nas_root, self.hostname)

            self.coordinator = PipelineCoordinator()
            self.coordinator.ensure_tables()
            stats = self.coordinator.recover_local_db(db, self.hostname)
            logger.info("Startup recovery from connectome.db complete")
        except Exception as e:
            logger.warning(f"Connectome DB recovery skipped: {e}")

        # Scan local DLC_Queue for orphaned files not in DB
        self._recover_local_dlc_queue()

    # =========================================================================
    # COORDINATION HELPERS
    # =========================================================================

    def _sync_to_connectome(self, video_id, state, **kwargs):
        """Best-effort sync video state to connectome.db after local DB update."""
        if not self.coordinator:
            return
        try:
            self.coordinator.sync_video_state(video_id, self.hostname, state, **kwargs)
        except Exception as e:
            logger.debug(f"Connectome sync failed (non-fatal): {e}")

    def _backup_local_db(self):
        """Best-effort backup of local watcher.db to NAS."""
        try:
            nas_root = Paths.NAS_ROOT
            if nas_root and hasattr(self, '_backup_db'):
                self._backup_db(self.db.db_path, nas_root, self.hostname)
        except Exception as e:
            logger.debug(f"DB backup failed (non-fatal): {e}")

    def _recover_local_dlc_queue(self):
        """Scan DLC_Queue for orphaned MP4s not in local DB.

        Infers state from sibling files:
          - MP4 only -> dlc_queued
          - MP4 + *DLC*.h5 -> dlc_complete
          - MP4 + h5 + pipeline JSONs -> processed
        """
        dlc_queue = Paths.DLC_QUEUE
        if not dlc_queue or not dlc_queue.exists():
            return

        recovered = 0
        for mp4 in dlc_queue.glob("*.mp4"):
            video_id = get_video_id(mp4.name)
            if not video_id:
                continue

            try:
                existing = self.db.get_video(video_id)
            except Exception:
                existing = None

            if existing is not None:
                continue  # Already tracked

            # Determine state from sibling files
            h5_files = list(dlc_queue.glob(f"{video_id}DLC*.h5"))
            json_files = list(dlc_queue.glob(f"{video_id}_*.json"))
            has_pipeline = any(
                f.name.endswith(('_segments.json', '_reaches.json', '_pellet_outcomes.json'))
                for f in json_files
            )

            if h5_files and has_pipeline:
                target_state = 'processed'
            elif h5_files:
                target_state = 'dlc_complete'
            else:
                target_state = 'dlc_queued'

            try:
                self.db.register_video(
                    video_id=video_id,
                    source_path=str(mp4),
                    current_path=str(mp4),
                )
                if target_state != 'discovered':
                    self.db.force_state(video_id, target_state)
                recovered += 1
                logger.debug(f"Recovered orphan {video_id} as {target_state}")
            except Exception as e:
                logger.debug(f"Could not recover orphan {video_id}: {e}")

        if recovered > 0:
            logger.info(f"Recovered {recovered} orphaned videos from DLC_Queue")

    # =========================================================================
    # SCAN PHASE
    # =========================================================================

    def _scan_phase(self):
        """Scan NAS for new collages/singles and check for DLC completions."""
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

    # =========================================================================
    # WORK QUEUE
    # =========================================================================

    def _get_next_work_item(self):
        """
        Get next work item — priority animal first, then Pillar-first, random.

        Priority:
        1. Stage DLC-complete videos to NAS (finish what's done)
        2. Run DLC on a queued single
        3. Crop next collage (only when nothing in-flight)
        """
        priority_animal = self._get_priority_animal()

        # Priority 1a: Archive locally processed videos (also_process mode)
        if self.config.also_process:
            processed = self.db.get_videos_in_state('processed')
            if processed:
                if priority_animal:
                    preferred = [v for v in processed if self._matches_priority(v, priority_animal, 'animal_id')]
                    pick = preferred[0] if preferred else processed[0]
                else:
                    pick = processed[0]
                return {
                    'type': 'archive_local',
                    'id': pick['video_id'],
                    'data': pick
                }

        # Priority 1b: Run local pipeline on DLC-complete videos (also_process mode)
        if self.config.also_process:
            processing = self.db.get_videos_in_state('processing')
            if processing:
                pick = self._pick_from_pool(processing, priority_animal, 'animal_id')
                return {
                    'type': 'local_pipeline',
                    'id': pick['video_id'],
                    'data': pick
                }

        # Priority 1c: Videos with DLC complete — stage to NAS or run local pipeline
        videos = self.db.get_videos_in_state('dlc_complete')
        if videos:
            # For staging, prefer priority animal but no randomization (stage ASAP)
            if priority_animal:
                preferred = [v for v in videos if self._matches_priority(v, priority_animal, 'animal_id')]
                pick = preferred[0] if preferred else videos[0]
            else:
                pick = videos[0]

            if self.config.also_process:
                return {
                    'type': 'local_pipeline',
                    'id': pick['video_id'],
                    'data': pick
                }
            return {
                'type': 'stage_to_nas',
                'id': pick['video_id'],
                'data': pick
            }

        # Priority 2: Videos queued for DLC (Pillar first, random within tier)
        videos = self.db.get_videos_in_state('dlc_queued')
        if videos:
            pick = self._pick_from_pool(videos, priority_animal, 'animal_id')
            return {
                'type': 'single_dlc',
                'id': pick['video_id'],
                'data': pick
            }

        # Priority 3: Crop next collage (Pillar first, random within tier)
        collages = self.db.get_collages_in_state('stable')
        if collages:
            pick = self._pick_from_pool(collages, priority_animal, 'animal_ids', is_collage=True)
            return {
                'type': 'collage',
                'id': pick['filename'],
                'data': pick
            }

        return None

    # =========================================================================
    # DISPATCH
    # =========================================================================

    def _dispatch_work(self, work: dict):
        """Route work to appropriate handler."""
        work_type = work['type']
        work_id = work['id']

        try:
            if work_type == 'collage':
                self._process_collage(work)
            elif work_type == 'single_dlc':
                self._process_single_dlc(work)
            elif work_type == 'stage_to_nas':
                self._stage_to_nas(work)
            elif work_type == 'local_pipeline':
                self._run_local_pipeline(work)
            elif work_type == 'archive_local':
                self._archive_locally_processed(work)
            else:
                logger.warning(f"Unknown work type: {work_type}")

            # Backup local DB to NAS after each successful work item
            self._backup_local_db()

        except Exception as e:
            error_msg = f"{work_type} failed: {str(e)}"
            logger.error(f"Work item {work_id} failed: {e}", exc_info=True)

            if work_type == 'collage':
                self.db.update_collage_state(work_id, 'failed', validation_error=error_msg)
            else:
                self.db.mark_failed(work_id, error_msg)

    # =========================================================================
    # DRY RUN
    # =========================================================================

    def dry_run(self):
        """Scan without processing. Shows what would be done."""
        logger.info("DLCOrchestrator dry run")

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

        # Show priority animal if set
        priority_animal = self._get_priority_animal()
        if priority_animal:
            print(f"\n  PRIORITY ANIMAL:      {priority_animal}")

        # Show what work would be done
        print(f"\nPending work items:")
        count = 0
        for state_label, state in [
            ("Collages to crop", "stable"),
            ("Videos for DLC", "dlc_queued"),
            ("Videos to stage to NAS", "dlc_complete"),
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

        print(f"\nStaging directory: {self.staging_dir or '(not configured)'}")
        print()

    # =========================================================================
    # DLC COMPLETION SCANNING
    # =========================================================================

    def _scan_for_dlc_completions(self) -> int:
        """
        Scan for h5 files matching videos in dlc_queued or dlc_running state.

        DLC outputs h5 files to DLC_Queue (same dir as input video).
        This detects them and advances state to dlc_complete.
        """
        dlc_queue_dir = Paths.DLC_QUEUE
        if not dlc_queue_dir or not dlc_queue_dir.exists():
            return 0

        count = 0

        for state in ('dlc_queued', 'dlc_running'):
            videos = self.db.get_videos_in_state(state)
            for video in videos:
                video_id = video['video_id']
                h5_files = list(dlc_queue_dir.glob(f"{video_id}DLC*.h5"))
                if h5_files:
                    dlc_path = h5_files[0]
                    logger.info(f"DLC completion detected: {video_id} -> {dlc_path.name}")

                    if state == 'dlc_queued':
                        self.db.update_state(video_id, 'dlc_running')
                    self.db.update_state(
                        video_id, 'dlc_complete',
                        dlc_output_path=str(dlc_path),
                        current_path=str(dlc_queue_dir / f"{video_id}.mp4")
                    )
                    self.db.log_step(video_id, 'dlc', 'completed', message=dlc_path.name)
                    count += 1

        return count

    # =========================================================================
    # COLLAGE PROCESSING
    # =========================================================================

    def _process_collage(self, work: dict):
        """Copy collage from NAS to local, crop into singles, queue for DLC."""
        from mousereach.video_prep.core.cropper import crop_collage

        collage_filename = work['id']
        collage_data = work['data']

        logger.info(f"Processing collage: {collage_filename}")

        # Cross-PC dedup: try to claim this collage before cropping
        if self.coordinator:
            try:
                if not self.coordinator.try_claim_collage(collage_filename, self.hostname):
                    logger.info(f"Collage {collage_filename} claimed by another PC, skipping")
                    return
            except Exception as e:
                logger.debug(f"Collage claim check failed (proceeding anyway): {e}")

        self.db.update_collage_state(collage_filename, 'cropping')
        self.db.log_step(collage_filename, 'crop', 'started')

        start_time = time.time()

        try:
            # Source path on NAS (D:)
            source_path = Path(collage_data['source_path'])
            if not source_path.exists():
                raise FileNotFoundError(f"Collage not found: {source_path}")

            # Copy collage to local working dir (A:)
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

            # Register each cropped single and move to DLC_Queue
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

                output_path = Path(result['output_path'])
                video_id = get_video_id(output_path.name)

                animal_id = result.get('animal_id', '')
                parsed_animal = AnimalID.parse(animal_id) if animal_id else {}
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

                self.db.update_state(video_id, 'validated', current_path=str(output_path))

                # Move single to DLC_Queue on local drive (A:)
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

            # Update collage state
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

            # Sync collage completion to connectome.db
            if self.coordinator:
                try:
                    self.coordinator.update_collage_state(
                        collage_filename, 'cropped', singles_created=videos_created
                    )
                except Exception as e:
                    logger.debug(f"Collage sync failed (non-fatal): {e}")

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

    # =========================================================================
    # DLC INFERENCE
    # =========================================================================

    def _process_single_dlc(self, work: dict):
        """Run DLC inference on a single video."""
        from mousereach.dlc.core import run_dlc_batch

        video_id = work['id']
        video_data = work['data']
        current_path = Path(video_data['current_path'])

        logger.info(f"Running DLC on {video_id}")

        if not self.config.dlc_config_path:
            logger.warning(
                f"DLC config not configured - {video_id} stays in dlc_queued. "
                "Run 'mousereach-setup' to set DLC model path."
            )
            return

        dlc_config = Path(self.config.dlc_config_path)
        if not dlc_config.exists():
            if dlc_config.is_dir():
                dlc_config = dlc_config / "config.yaml"
            if not dlc_config.exists():
                logger.error(f"DLC config not found: {self.config.dlc_config_path}")
                return

        if not current_path.exists():
            logger.error(f"Video file not found: {current_path}")
            self.db.mark_failed(video_id, f"Video file not found: {current_path}")
            return

        self.db.update_state(video_id, 'dlc_running')
        self.db.log_step(video_id, 'dlc', 'started', message=f"GPU {self.config.dlc_gpu_device}")

        start_time = time.time()

        try:
            # DLC outputs to same directory as input (DLC_Queue)
            dlc_output_dir = current_path.parent
            dlc_output_dir.mkdir(parents=True, exist_ok=True)

            results = run_dlc_batch(
                video_paths=[current_path],
                config_path=dlc_config,
                output_dir=dlc_output_dir,
                gpu=self.config.dlc_gpu_device,
                save_as_csv=True
            )

            duration = time.time() - start_time

            if results and results[0].get('status') == 'success':
                h5_files = list(dlc_output_dir.glob(f"{video_id}DLC*.h5"))
                dlc_output = str(h5_files[0]) if h5_files else None

                self.db.update_state(
                    video_id, 'dlc_complete',
                    dlc_output_path=dlc_output,
                    current_path=str(current_path)
                )
                self.db.log_step(
                    video_id, 'dlc', 'completed',
                    message=f"GPU {self.config.dlc_gpu_device}",
                    duration=duration
                )
                logger.info(f"DLC completed for {video_id} ({duration:.1f}s)")
                self._sync_to_connectome(video_id, 'dlc_complete',
                                         dlc_completed_at=datetime.now().isoformat())
            else:
                error_msg = results[0].get('error', 'Unknown DLC error') if results else 'No results'
                raise RuntimeError(f"DLC failed: {error_msg}")

        except Exception as e:
            duration = time.time() - start_time
            self.db.mark_failed(video_id, str(e))
            self.db.log_step(video_id, 'dlc', 'failed', message=str(e), duration=duration)
            raise

    # =========================================================================
    # LOCAL PIPELINE (also_process mode)
    # =========================================================================

    def _run_local_pipeline(self, work: dict):
        """Run seg/reach/outcomes locally after DLC, then archive directly.

        When also_process=True, DLC PCs run the full pipeline locally instead
        of staging to NAS for the processing server. Reuses the same pipeline
        functions as ProcessingOrchestrator._run_pipeline().
        """
        from mousereach.segmentation.core.batch import process_single as seg_single
        from mousereach.reach.core.batch import process_single as reach_single
        from mousereach.outcomes.core.batch import process_single as outcome_single
        from mousereach.pipeline.manifest import create_processing_manifest
        from mousereach.pipeline.triage import triage_video

        video_id = work['id']
        video_data = work['data']

        dlc_path = Path(video_data.get('dlc_output_path', ''))
        if not dlc_path.exists():
            dlc_queue = Paths.DLC_QUEUE
            if dlc_queue:
                h5_files = list(dlc_queue.glob(f"{video_id}DLC*.h5"))
                if h5_files:
                    dlc_path = h5_files[0]
            if not dlc_path.exists():
                self.db.mark_failed(video_id, f"DLC h5 not found for {video_id}")
                return

        processing_dir = dlc_path.parent
        logger.info(f"Running local pipeline on {video_id} (also_process mode)")

        self.db.update_state(video_id, 'processing')
        self.db.log_step(video_id, 'local_pipeline', 'started')
        pipeline_start = time.time()

        try:
            # Step 1: Segmentation
            self.db.log_step(video_id, 'segmentation', 'started')
            step_start = time.time()
            seg_result = seg_single(dlc_path)
            seg_duration = time.time() - step_start

            if seg_result.get('success', False):
                self.db.log_step(video_id, 'segmentation', 'completed',
                                message=f"boundaries={seg_result.get('n_boundaries', 0)}",
                                duration=seg_duration)
                logger.info(f"Segmentation complete: {video_id} ({seg_duration:.1f}s)")
            else:
                error = seg_result.get('error', 'segmentation failed')
                self.db.log_step(video_id, 'segmentation', 'failed', message=error, duration=seg_duration)
                self.db.mark_failed(video_id, f"Segmentation failed: {error}")
                return

            # Step 2: Reach Detection
            seg_path = processing_dir / f"{video_id}_segments.json"
            if not seg_path.exists():
                self.db.mark_failed(video_id, "Segments file not created")
                return

            self.db.log_step(video_id, 'reach_detection', 'started')
            step_start = time.time()
            reach_result = reach_single(dlc_path, seg_path)
            reach_duration = time.time() - step_start
            self.db.log_step(video_id, 'reach_detection', 'completed',
                            message=f"reaches={reach_result.get('total_reaches', 0)}",
                            duration=reach_duration)
            logger.info(f"Reach detection complete: {video_id} ({reach_duration:.1f}s)")

            # Step 3: Outcome Detection (skip for E/F trays)
            tray_info = parse_tray_type(f"{video_id}.mp4")
            tray_type = tray_info.get('tray_type', 'P')
            skip_outcomes = tray_type in ('E', 'F')

            if not skip_outcomes:
                reach_path = processing_dir / f"{video_id}_reaches.json"
                self.db.log_step(video_id, 'outcome_detection', 'started')
                step_start = time.time()
                outcome_result = outcome_single(dlc_path, seg_path, reach_path)
                outcome_duration = time.time() - step_start
                self.db.log_step(video_id, 'outcome_detection', 'completed',
                                message=f"segments={outcome_result.get('n_segments', 0)}",
                                duration=outcome_duration)
                logger.info(f"Outcome detection complete: {video_id} ({outcome_duration:.1f}s)")

            # Step 4: Feature Extraction
            if not skip_outcomes:
                reach_path = processing_dir / f"{video_id}_reaches.json"
                outcome_path = processing_dir / f"{video_id}_pellet_outcomes.json"
                if reach_path.exists() and outcome_path.exists():
                    try:
                        from mousereach.kinematics.core.feature_extractor import FeatureExtractor
                        extractor = FeatureExtractor()
                        features = extractor.extract(dlc_path, reach_path, outcome_path)
                        features_path = processing_dir / f"{video_id}_features.json"
                        with open(features_path, 'w') as f:
                            json.dump(features.to_dict(), f, indent=2)
                        logger.info(f"Feature extraction complete: {video_id}")

                        # Database sync
                        try:
                            from mousereach.sync.database import sync_file_to_database
                            sync_file_to_database(features_path)
                        except Exception as e:
                            logger.warning(f"Database sync failed for {video_id}: {e}")
                    except Exception as e:
                        logger.warning(f"Feature extraction failed for {video_id}: {e}")

            # Generate provenance manifest
            pipeline_duration = time.time() - pipeline_start
            try:
                step_timestamps = {
                    'pipeline_started_at': datetime.fromtimestamp(pipeline_start).isoformat(),
                    'pipeline_completed_at': datetime.now().isoformat(),
                }
                create_processing_manifest(
                    video_id=video_id,
                    processing_dir=processing_dir,
                    dlc_path=dlc_path,
                    step_timestamps=step_timestamps,
                )
            except Exception as e:
                logger.warning(f"Manifest creation failed for {video_id}: {e}")

            # Unified triage
            try:
                triage_result = triage_video(
                    video_id=video_id,
                    processing_dir=processing_dir,
                    h5_path=dlc_path,
                )
                for suffix in ['_segments.json', '_reaches.json', '_pellet_outcomes.json']:
                    json_path = processing_dir / f"{video_id}{suffix}"
                    if json_path.exists():
                        try:
                            with open(json_path) as f:
                                data = json.load(f)
                            data['validation_status'] = triage_result.verdict
                            data['triage_reason'] = (
                                '; '.join(f.description for f in triage_result.flags if f.severity == 'critical')
                                if triage_result.verdict == 'needs_review'
                                else 'Unified triage: all checks passed'
                            )
                            with open(json_path, 'w') as f:
                                json.dump(data, f, indent=2)
                        except Exception:
                            pass
                triage_result.save(processing_dir / f"{video_id}_triage.json")
            except Exception as e:
                logger.warning(f"Unified triage failed for {video_id}: {e}")

            self.db.update_state(video_id, 'processed')
            self.db.log_step(video_id, 'local_pipeline', 'completed',
                            message=f"All steps complete ({pipeline_duration:.1f}s total)",
                            duration=pipeline_duration)
            logger.info(f"Local pipeline complete: {video_id} ({pipeline_duration:.1f}s)")
            self._sync_to_connectome(video_id, 'processed',
                                     processed_at=datetime.now().isoformat())

        except Exception as e:
            pipeline_duration = time.time() - pipeline_start
            self.db.log_step(video_id, 'local_pipeline', 'failed', message=str(e), duration=pipeline_duration)
            self.db.mark_failed(video_id, f"Local pipeline error: {e}")
            raise

    def _archive_locally_processed(self, work: dict):
        """Archive a locally processed video directly to NAS.

        In also_process mode, results go straight to Analyzed/{project}/{cohort}/ on NAS,
        skipping the DLC_Complete staging step entirely.
        """
        from mousereach.archive.core import archive_video

        video_id = work['id']
        logger.info(f"Archiving locally processed {video_id} to NAS")
        self.db.update_state(video_id, 'archiving')
        self.db.log_step(video_id, 'archive', 'started')
        start_time = time.time()

        try:
            dlc_queue = Paths.DLC_QUEUE
            result = archive_video(
                video_id,
                dry_run=False,
                verbose=False,
                skip_ready_check=True,
                source_dir=dlc_queue,
            )
            duration = time.time() - start_time

            if result.get('success'):
                self.db.update_state(video_id, 'archived')
                self.db.log_step(video_id, 'archive', 'completed',
                                message=f"Archived {len(result.get('files_moved', []))} files",
                                duration=duration)
                logger.info(f"Archived {video_id} to NAS ({duration:.1f}s)")
                self._sync_to_connectome(video_id, 'archived',
                                         staged_at=datetime.now().isoformat())

                # Export to central DB on NAS for cross-node provenance
                self.db.export_to_central_db(video_id)

                # Clean up local DLC_Queue files
                if dlc_queue:
                    for f in self._get_associated_files(dlc_queue, video_id):
                        try:
                            f.unlink()
                        except Exception:
                            pass
            else:
                error = result.get('error', 'archive failed')
                self.db.log_step(video_id, 'archive', 'failed', message=error, duration=duration)
                logger.warning(f"Archive failed for {video_id}: {error}")

        except Exception as e:
            duration = time.time() - start_time
            self.db.log_step(video_id, 'archive', 'failed', message=str(e), duration=duration)
            logger.error(f"Archive error for {video_id}: {e}")

    # =========================================================================
    # NAS STAGING
    # =========================================================================

    def _stage_to_nas(self, work: dict):
        """
        Move DLC-complete video + h5 from local (A:) to NAS staging (D:).

        The processing PC picks these up from the NAS staging folder.
        """
        video_id = work['id']
        video_data = work['data']

        if not self.staging_dir:
            logger.error("DLC_STAGING path not configured (NAS drive not set)")
            self.db.mark_failed(video_id, "DLC_STAGING path not configured")
            return

        logger.info(f"Staging {video_id} to NAS: {self.staging_dir}")

        self.db.log_step(video_id, 'stage_to_nas', 'started')
        start_time = time.time()

        try:
            # Find all files for this video in DLC_Queue (mp4, h5, csv)
            current_path = Path(video_data['current_path'])
            source_dir = current_path.parent
            all_files = self._get_associated_files(source_dir, video_id)

            if not all_files:
                raise FileNotFoundError(f"No files found for {video_id} in {source_dir}")

            self.staging_dir.mkdir(parents=True, exist_ok=True)

            # Move each file to NAS staging
            staged_files = []
            for file_path in all_files:
                dest_path = self.staging_dir / file_path.name
                if safe_move(file_path, dest_path):
                    staged_files.append(file_path.name)
                    logger.debug(f"Staged: {file_path.name}")
                else:
                    logger.warning(f"Failed to stage: {file_path.name}")

            duration = time.time() - start_time

            # Mark as archived (done on this machine)
            self.db.update_state(
                video_id, 'archived',
                current_path=str(self.staging_dir / current_path.name)
            )
            self.db.log_step(
                video_id, 'stage_to_nas', 'completed',
                message=f"Staged {len(staged_files)} files to NAS",
                duration=duration
            )

            logger.info(f"Staged {video_id} to NAS ({len(staged_files)} files, {duration:.1f}s)")
            self._sync_to_connectome(video_id, 'archived',
                                     staged_at=datetime.now().isoformat(),
                                     nas_path=str(self.staging_dir))

        except Exception as e:
            duration = time.time() - start_time
            self.db.mark_failed(video_id, str(e))
            self.db.log_step(video_id, 'stage_to_nas', 'failed', message=str(e), duration=duration)
            raise


# =============================================================================
# PROCESSING ORCHESTRATOR (Processing Server)
# =============================================================================

class ProcessingOrchestrator(BaseOrchestrator):
    """
    Processing server orchestrator.

    Watches the NAS DLC_Complete/ staging folder for new DLC outputs,
    copies them to the local Processing/ folder, and runs the full
    analysis pipeline (segmentation -> reach detection -> outcomes).

    After processing, auto-approved videos are archived to NAS.
    """

    def __init__(self, config: WatcherConfig, db: WatcherDB):
        super().__init__(config, db)

        # Where the DLC PC stages completed files on NAS
        self.staging_dir = Paths.DLC_STAGING
        if self.staging_dir:
            logger.info(f"Watching staging directory: {self.staging_dir}")
        else:
            logger.warning("DLC_STAGING not configured -- no intake possible")

        # Local processing directory
        self.processing_dir = Paths.PROCESSING
        if self.processing_dir:
            self.processing_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Processing directory: {self.processing_dir}")

        # Reprocessing scanner (runs every N poll cycles)
        self._reprocess_scanner = None
        self._scan_cycle_count = 0
        self._reprocess_scan_interval = 10  # Every 10th poll cycle (~5 min at 30s interval)

        if Paths.NAS_ROOT:
            try:
                from mousereach.watcher.reprocessor import ReprocessingScanner
                self._reprocess_scanner = ReprocessingScanner(db, Paths.NAS_ROOT)
                logger.info("Reprocessing scanner enabled")
            except Exception as e:
                logger.warning(f"Reprocessing scanner not available: {e}")

    # =========================================================================
    # SCAN PHASE
    # =========================================================================

    def _scan_phase(self):
        """Discover new DLC outputs in the staging directory."""
        if not self.staging_dir:
            return

        # Clean up stale claims from crashed nodes
        self._cleanup_stale_claims()

        newly_found = self.state.discover_dlc_staged(self.staging_dir)
        if newly_found:
            logger.info(f"Discovered {len(newly_found)} new DLC outputs in staging")

        # Periodic reprocessing scan (every N cycles)
        self._scan_cycle_count += 1
        if (self._reprocess_scanner and
                self._scan_cycle_count % self._reprocess_scan_interval == 0):
            try:
                summary = self._reprocess_scanner.scan(mark_outdated=True)
                if summary.get('outdated', 0) > 0:
                    logger.info(
                        f"Reprocessing scan: {summary['outdated']} outdated videos found"
                    )
            except Exception as e:
                logger.warning(f"Reprocessing scan failed: {e}")

    # =========================================================================
    # WORK QUEUE
    # =========================================================================

    def _get_next_work_item(self):
        """
        Get next work item for the processing server.

        Priority:
        1. Intake: DLC-complete videos on NAS → copy to local Processing/
        2. Pipeline: Videos in Processing/ that need seg/reach/outcomes
        3. Archive: Processed videos → archive to NAS
        """
        priority_animal = self._get_priority_animal()

        # Priority 1: Intake — videos discovered in staging, not yet copied locally
        videos = self.db.get_videos_in_state('dlc_complete')
        if videos:
            # Check disk space cap
            processing_count = len(self.db.get_videos_in_state('processing'))
            if processing_count >= self.config.max_local_pending:
                logger.warning(
                    f"Local pending limit reached ({processing_count}/{self.config.max_local_pending}). "
                    "Pausing intake — review or archive pending videos."
                )
            else:
                if priority_animal:
                    preferred = [v for v in videos if self._matches_priority(v, priority_animal, 'animal_id')]
                    pick = preferred[0] if preferred else videos[0]
                else:
                    pick = videos[0]
                return {
                    'type': 'intake',
                    'id': pick['video_id'],
                    'data': pick
                }

        # Priority 2: Pipeline — run seg/reach/outcomes on locally staged videos
        videos = self.db.get_videos_in_state('processing')
        if videos:
            pick = self._pick_from_pool(videos, priority_animal, 'animal_id')
            return {
                'type': 'pipeline',
                'id': pick['video_id'],
                'data': pick
            }

        # Priority 3: Archive — move processed videos to NAS
        videos = self.db.get_videos_in_state('processed')
        if videos:
            if priority_animal:
                preferred = [v for v in videos if self._matches_priority(v, priority_animal, 'animal_id')]
                pick = preferred[0] if preferred else videos[0]
            else:
                pick = videos[0]
            return {
                'type': 'archive',
                'id': pick['video_id'],
                'data': pick
            }

        # Priority 4: Reprocess outdated videos
        outdated = self.db.get_videos_in_state('outdated')
        if outdated:
            pick = self._pick_from_pool(outdated, priority_animal, 'animal_id')
            scope = pick.get('reprocess_scope', 'full')

            if scope == 'full':
                # Needs DLC re-run — can't do it here (no CUDA), stage back to DLC queue
                self.db.force_state(pick['video_id'], 'dlc_queued')
                logger.info(f"Outdated {pick['video_id']} queued for full reprocess (DLC changed)")
                return None  # Let next cycle pick it up via DLC orchestrator
            else:
                # post_dlc: re-run seg/reach/outcomes from existing DLC output
                return {
                    'type': 'reprocess',
                    'id': pick['video_id'],
                    'data': pick
                }

        return None

    # =========================================================================
    # DISPATCH
    # =========================================================================

    def _dispatch_work(self, work: dict):
        """Route work to appropriate handler."""
        work_type = work['type']
        work_id = work['id']

        try:
            if work_type == 'intake':
                self._intake_from_staging(work)
            elif work_type == 'pipeline':
                self._run_pipeline(work)
            elif work_type == 'archive':
                self._archive_to_nas(work)
            elif work_type == 'reprocess':
                self._reprocess_video(work)
            else:
                logger.warning(f"Unknown work type: {work_type}")

        except Exception as e:
            error_msg = f"{work_type} failed: {str(e)}"
            logger.error(f"Work item {work_id} failed: {e}", exc_info=True)
            self.db.mark_failed(work_id, error_msg)

    # =========================================================================
    # REPROCESS: Re-run pipeline on outdated archived videos
    # =========================================================================

    def _reprocess_video(self, work: dict):
        """Re-run seg/reach/outcomes on an outdated archived video.

        The video's archived files are still on NAS. We copy the DLC h5
        and video to local Processing/, re-run the pipeline, and re-archive.
        """
        video_id = work['id']
        video_data = work['data']

        logger.info(f"Reprocessing outdated video {video_id} (post-DLC)")

        # Find archived files on NAS
        archive_dir = Paths.ANALYZED_OUTPUT if Paths.ANALYZED_OUTPUT else None
        if not archive_dir or not archive_dir.exists():
            self.db.mark_failed(video_id, "Archive directory not found for reprocessing")
            return

        # Search for DLC h5 in archive
        h5_files = list(archive_dir.rglob(f"{video_id}DLC*.h5"))
        if not h5_files:
            self.db.mark_failed(video_id, f"DLC h5 not found in archive for reprocessing")
            return

        source_h5 = h5_files[0]
        source_dir = source_h5.parent

        # Copy needed files to local Processing/
        if not self.processing_dir:
            self.db.mark_failed(video_id, "PROCESSING path not configured")
            return

        self.processing_dir.mkdir(parents=True, exist_ok=True)

        # Copy DLC h5 and video
        all_files = [f for f in source_dir.iterdir() if f.stem.startswith(video_id)]
        for src_file in all_files:
            dest = self.processing_dir / src_file.name
            safe_copy(src_file, dest, verify=True)

        # Transition to processing state
        local_h5 = list(self.processing_dir.glob(f"{video_id}DLC*.h5"))
        self.db.force_state(
            video_id, 'processing',
            dlc_output_path=str(local_h5[0]) if local_h5 else '',
            current_path=str(self.processing_dir / f"{video_id}.mp4")
        )

        # Run the standard pipeline (seg/reach/outcomes)
        reprocess_work = {
            'type': 'pipeline',
            'id': video_id,
            'data': {
                'video_id': video_id,
                'dlc_output_path': str(local_h5[0]) if local_h5 else '',
                'current_path': str(self.processing_dir / f"{video_id}.mp4"),
            }
        }
        self._run_pipeline(reprocess_work)

    # =========================================================================
    # DRY RUN
    # =========================================================================

    def dry_run(self):
        """Scan without processing. Shows what would be done."""
        logger.info("ProcessingOrchestrator dry run")

        # Scan staging for new files
        if self.staging_dir and self.staging_dir.exists():
            newly_found = self.state.discover_dlc_staged(self.staging_dir)
            print(f"\nStaging scan:")
            print(f"  Staging directory:  {self.staging_dir}")
            print(f"  New DLC outputs:    {len(newly_found)}")
        else:
            print(f"\nStaging scan:")
            print(f"  Staging directory:  {self.staging_dir or '(not configured)'}")
            print(f"  Status:             NOT ACCESSIBLE")
            newly_found = []

        # Show priority animal if set
        priority_animal = self._get_priority_animal()
        if priority_animal:
            print(f"\n  PRIORITY ANIMAL:    {priority_animal}")

        # Show pending work
        print(f"\nPending work items:")
        count = 0
        for state_label, state in [
            ("Videos to intake from NAS", "dlc_complete"),
            ("Videos to process (seg/reach/outcomes)", "processing"),
            ("Videos to archive to NAS", "processed"),
        ]:
            items = self.db.get_videos_in_state(state)
            if items:
                print(f"  {state_label}: {len(items)}")
                for item in items[:5]:
                    print(f"    - {item.get('video_id')}")
                if len(items) > 5:
                    print(f"    ... and {len(items) - 5} more")
                count += len(items)

        if count == 0:
            print("  (no pending work)")

        # Show local disk usage
        if self.processing_dir and self.processing_dir.exists():
            local_videos = len(list(self.processing_dir.glob("*DLC*.h5")))
            print(f"\nLocal Processing/: {local_videos} videos")
            print(f"  Max local pending: {self.config.max_local_pending}")
        print()

    # =========================================================================
    # MULTI-NODE CLAIMING
    # =========================================================================

    def _claim_video(self, video_id: str) -> bool:
        """Try to claim a video for processing. Returns True if claimed.

        Uses marker files in DLC_Complete/.claims/ to prevent multiple nodes
        from processing the same video simultaneously.
        """
        if not self.staging_dir:
            return True  # No staging = no contention

        claim_dir = self.staging_dir / ".claims"
        try:
            claim_dir.mkdir(exist_ok=True)
        except Exception:
            return True  # Can't create claims dir = single-node mode

        claim_file = claim_dir / f"{video_id}.claimed"

        # Check if already claimed by another host
        if claim_file.exists():
            try:
                content = claim_file.read_text().strip()
                claimer = content.split('\n')[0]
                if claimer != self.hostname:
                    logger.debug(f"{video_id} already claimed by {claimer}")
                    return False
                # We already claimed it (retry after crash?)
                return True
            except Exception:
                return False

        # Try to claim
        try:
            claim_file.write_text(f"{self.hostname}\n{datetime.now().isoformat()}\n")
            # Verify our claim stuck (race window check on network drives)
            time.sleep(0.5)
            content = claim_file.read_text().strip()
            if content.split('\n')[0] == self.hostname:
                logger.info(f"Claimed {video_id} for {self.hostname}")
                return True
            return False
        except Exception:
            return False

    def _release_claim(self, video_id: str):
        """Release a claim after successful archive."""
        if not self.staging_dir:
            return
        claim_file = self.staging_dir / ".claims" / f"{video_id}.claimed"
        try:
            claim_file.unlink(missing_ok=True)
        except Exception:
            pass

    def _cleanup_stale_claims(self):
        """Remove claim files older than 24 hours (crashed nodes)."""
        if not self.staging_dir:
            return
        claim_dir = self.staging_dir / ".claims"
        if not claim_dir.exists():
            return

        stale_threshold = 24 * 3600  # 24 hours
        now = time.time()

        for claim_file in claim_dir.glob("*.claimed"):
            try:
                age = now - claim_file.stat().st_mtime
                if age > stale_threshold:
                    claim_file.unlink(missing_ok=True)
                    logger.info(f"Removed stale claim: {claim_file.stem} (age: {age/3600:.1f}h)")
            except Exception:
                pass

    # =========================================================================
    # INTAKE: Copy from NAS staging to local Processing/
    # =========================================================================

    def _intake_from_staging(self, work: dict):
        """Copy DLC-complete video+h5 from NAS staging to local Processing/."""
        video_id = work['id']
        video_data = work['data']

        if not self.processing_dir:
            logger.error("PROCESSING path not configured")
            self.db.mark_failed(video_id, "PROCESSING path not configured")
            return

        # Multi-node claiming: ensure no other node is processing this video
        if not self._claim_video(video_id):
            logger.debug(f"Skipping {video_id} - claimed by another node")
            return

        logger.info(f"Intake {video_id} from staging to Processing/")

        self.db.log_step(video_id, 'intake', 'started')
        start_time = time.time()

        try:
            # Find all files in staging directory
            source_dir = Path(video_data.get('current_path', '')).parent
            if not source_dir.exists():
                # Try staging dir directly
                source_dir = self.staging_dir

            all_files = self._get_associated_files(source_dir, video_id)
            if not all_files:
                raise FileNotFoundError(f"No files found for {video_id} in {source_dir}")

            self.processing_dir.mkdir(parents=True, exist_ok=True)

            # Copy files to local Processing/
            copied_files = []
            for file_path in all_files:
                dest_path = self.processing_dir / file_path.name
                if safe_copy(file_path, dest_path, verify=True):
                    copied_files.append(file_path.name)
                    logger.debug(f"Copied: {file_path.name}")
                else:
                    logger.warning(f"Failed to copy: {file_path.name}")

            if not copied_files:
                raise IOError(f"Failed to copy any files for {video_id}")

            duration = time.time() - start_time

            # Find local DLC h5 path
            local_h5 = list(self.processing_dir.glob(f"{video_id}DLC*.h5"))
            local_mp4 = self.processing_dir / f"{video_id}.mp4"

            # Advance to processing state
            self.db.update_state(
                video_id, 'processing',
                dlc_output_path=str(local_h5[0]) if local_h5 else video_data.get('dlc_output_path'),
                current_path=str(local_mp4)
            )
            self.db.log_step(
                video_id, 'intake', 'completed',
                message=f"Copied {len(copied_files)} files to Processing/",
                duration=duration
            )

            logger.info(f"Intake complete: {video_id} ({len(copied_files)} files, {duration:.1f}s)")

        except Exception as e:
            duration = time.time() - start_time
            self.db.mark_failed(video_id, str(e))
            self.db.log_step(video_id, 'intake', 'failed', message=str(e), duration=duration)
            raise

    # =========================================================================
    # PIPELINE: Run seg -> reach -> outcomes
    # =========================================================================

    def _run_pipeline(self, work: dict):
        """Run segmentation, reach detection, and outcome detection on a video.

        After each step, runs triage to set validation_status in the output JSON.
        After all steps complete, generates a processing manifest with provenance.
        """
        from mousereach.segmentation.core.batch import process_single as seg_single, add_validation_status
        from mousereach.reach.core.batch import process_single as reach_single
        from mousereach.outcomes.core.batch import process_single as outcome_single
        from mousereach.pipeline.manifest import create_processing_manifest
        from mousereach.pipeline.triage import triage_video

        video_id = work['id']
        video_data = work['data']

        dlc_path = Path(video_data.get('dlc_output_path', ''))
        if not dlc_path.exists():
            # Try to find it in Processing/
            h5_files = list(self.processing_dir.glob(f"{video_id}DLC*.h5"))
            if h5_files:
                dlc_path = h5_files[0]
            else:
                self.db.mark_failed(video_id, f"DLC h5 not found for {video_id}")
                return

        logger.info(f"Running pipeline on {video_id}")
        pipeline_start = time.time()

        # --- Step 1: Segmentation ---
        self.db.log_step(video_id, 'segmentation', 'started')
        step_start = time.time()

        try:
            seg_result = seg_single(dlc_path)
            seg_duration = time.time() - step_start

            if seg_result.get('success', False):
                self.db.log_step(
                    video_id, 'segmentation', 'completed',
                    message=f"status={seg_result.get('status')}, boundaries={seg_result.get('n_boundaries', 0)}",
                    duration=seg_duration
                )
                logger.info(f"Segmentation complete: {video_id} ({seg_duration:.1f}s)")
            else:
                error = seg_result.get('error', 'segmentation failed')
                self.db.log_step(video_id, 'segmentation', 'failed', message=error, duration=seg_duration)
                self.db.mark_failed(video_id, f"Segmentation failed: {error}")
                return

        except Exception as e:
            seg_duration = time.time() - step_start
            self.db.log_step(video_id, 'segmentation', 'failed', message=str(e), duration=seg_duration)
            self.db.mark_failed(video_id, f"Segmentation error: {e}")
            raise

        # --- Step 2: Reach Detection ---
        seg_path = self.processing_dir / f"{video_id}_segments.json"
        if not seg_path.exists():
            self.db.mark_failed(video_id, "Segments file not created by segmentation")
            return

        self.db.log_step(video_id, 'reach_detection', 'started')
        step_start = time.time()

        try:
            reach_result = reach_single(dlc_path, seg_path)
            reach_duration = time.time() - step_start

            self.db.log_step(
                video_id, 'reach_detection', 'completed',
                message=f"reaches={reach_result.get('total_reaches', 0)}",
                duration=reach_duration
            )
            logger.info(f"Reach detection complete: {video_id} ({reach_duration:.1f}s)")

        except Exception as e:
            reach_duration = time.time() - step_start
            self.db.log_step(video_id, 'reach_detection', 'failed', message=str(e), duration=reach_duration)
            self.db.mark_failed(video_id, f"Reach detection error: {e}")
            raise

        # --- Step 3: Outcome Detection (skip for E/F trays) ---
        tray_info = parse_tray_type(f"{video_id}.mp4")
        tray_type = tray_info.get('tray_type', 'P')
        skip_outcomes = tray_type in ('E', 'F')

        if not skip_outcomes:
            reach_path = self.processing_dir / f"{video_id}_reaches.json"

            self.db.log_step(video_id, 'outcome_detection', 'started')
            step_start = time.time()

            try:
                outcome_result = outcome_single(dlc_path, seg_path, reach_path)
                outcome_duration = time.time() - step_start

                self.db.log_step(
                    video_id, 'outcome_detection', 'completed',
                    message=f"segments={outcome_result.get('n_segments', 0)}",
                    duration=outcome_duration
                )
                logger.info(f"Outcome detection complete: {video_id} ({outcome_duration:.1f}s)")

            except Exception as e:
                outcome_duration = time.time() - step_start
                self.db.log_step(video_id, 'outcome_detection', 'failed', message=str(e), duration=outcome_duration)
                self.db.mark_failed(video_id, f"Outcome detection error: {e}")
                raise
        else:
            logger.info(f"Skipping outcomes for {video_id} (tray type: {tray_type})")

        # --- Step 4: Feature Extraction (join reaches + outcomes + DLC kinematics) ---
        if not skip_outcomes:
            reach_path = self.processing_dir / f"{video_id}_reaches.json"
            outcome_path = self.processing_dir / f"{video_id}_pellet_outcomes.json"

            if reach_path.exists() and outcome_path.exists():
                self.db.log_step(video_id, 'feature_extraction', 'started')
                step_start = time.time()

                try:
                    from mousereach.kinematics.core.feature_extractor import FeatureExtractor
                    extractor = FeatureExtractor()
                    features = extractor.extract(dlc_path, reach_path, outcome_path)

                    features_path = self.processing_dir / f"{video_id}_features.json"
                    with open(features_path, 'w') as f:
                        json.dump(features.to_dict(), f, indent=2)

                    feat_duration = time.time() - step_start
                    self.db.log_step(
                        video_id, 'feature_extraction', 'completed',
                        message=f"segments={features.n_segments}",
                        duration=feat_duration
                    )
                    logger.info(f"Feature extraction complete: {video_id} ({feat_duration:.1f}s)")

                    # --- Step 5: Database sync (push features to connectome.db) ---
                    try:
                        from mousereach.sync.database import sync_file_to_database
                        synced = sync_file_to_database(features_path)
                        if synced:
                            self.db.log_step(video_id, 'db_sync', 'completed', message="Synced to connectome.db")
                            logger.info(f"Database sync complete: {video_id}")
                        else:
                            logger.debug(f"Database sync skipped for {video_id} (subject not in DB or DB unavailable)")
                    except Exception as e:
                        logger.warning(f"Database sync failed for {video_id}: {e}")

                except Exception as e:
                    feat_duration = time.time() - step_start
                    self.db.log_step(video_id, 'feature_extraction', 'failed', message=str(e), duration=feat_duration)
                    # Feature extraction failure is non-fatal — video still gets triaged/processed
                    logger.warning(f"Feature extraction failed for {video_id}: {e}")

        # --- Pipeline complete — generate provenance manifest ---
        pipeline_duration = time.time() - pipeline_start

        try:
            step_timestamps = {
                'pipeline_started_at': datetime.fromtimestamp(pipeline_start).isoformat(),
                'pipeline_completed_at': datetime.now().isoformat(),
            }
            manifest = create_processing_manifest(
                video_id=video_id,
                processing_dir=self.processing_dir,
                dlc_path=dlc_path,
                step_timestamps=step_timestamps,
            )
            logger.info(
                f"Manifest created: {video_id} "
                f"(DLC={manifest.get('dlc_model', {}).get('dlc_scorer', '?')})"
            )
        except Exception as e:
            logger.warning(f"Manifest creation failed for {video_id}: {e}")

        # --- Unified triage: run AFTER all pipeline steps complete ---
        # Evaluates DLC coherence, structural integrity, cross-step
        # consistency, and statistical outliers in one pass.
        try:
            triage_result = triage_video(
                video_id=video_id,
                processing_dir=self.processing_dir,
                h5_path=dlc_path,
            )

            # Write validation_status to each output JSON based on unified verdict
            for suffix in ['_segments.json', '_reaches.json', '_pellet_outcomes.json']:
                json_path = self.processing_dir / f"{video_id}{suffix}"
                if json_path.exists():
                    try:
                        with open(json_path) as f:
                            data = json.load(f)
                        data['validation_status'] = triage_result.verdict
                        data['triage_reason'] = (
                            '; '.join(f.description for f in triage_result.flags if f.severity == 'critical')
                            if triage_result.verdict == 'needs_review'
                            else 'Unified triage: all checks passed'
                        )
                        with open(json_path, 'w') as f:
                            json.dump(data, f, indent=2)
                    except Exception:
                        pass

            # Save triage result
            triage_result.save(self.processing_dir / f"{video_id}_triage.json")

            logger.info(
                f"Triage: {video_id} -> {triage_result.verdict} "
                f"({triage_result.n_critical} critical, {triage_result.n_warnings} warnings)"
            )
        except Exception as e:
            logger.warning(f"Unified triage failed for {video_id}: {e}")

        self.db.update_state(video_id, 'processed')
        self.db.log_step(
            video_id, 'pipeline', 'completed',
            message=f"All steps complete ({pipeline_duration:.1f}s total)",
            duration=pipeline_duration
        )
        logger.info(f"Pipeline complete: {video_id} ({pipeline_duration:.1f}s total)")

    # =========================================================================
    # ARCHIVE: Move processed videos to NAS
    # =========================================================================

    def _archive_to_nas(self, work: dict):
        """Archive processed video to NAS and clean up local copy."""
        from mousereach.archive.core import archive_video

        video_id = work['id']

        logger.info(f"Archiving {video_id} to NAS")
        self.db.log_step(video_id, 'archive', 'started')
        start_time = time.time()

        try:
            result = archive_video(video_id, dry_run=False, verbose=False)
            duration = time.time() - start_time

            if result.get('success'):
                self.db.update_state(video_id, 'archived')
                self.db.log_step(
                    video_id, 'archive', 'completed',
                    message=f"Archived {len(result.get('files_moved', []))} files to {result.get('destination')}",
                    duration=duration
                )
                logger.info(f"Archived {video_id} to NAS ({duration:.1f}s)")

                # Export to central DB on NAS for cross-node provenance
                self.db.export_to_central_db(video_id)

                # Release multi-node claim
                self._release_claim(video_id)

                # Remove from staging on NAS (the DLC PC's copy)
                if self.staging_dir:
                    staging_files = self._get_associated_files(self.staging_dir, video_id)
                    for f in staging_files:
                        try:
                            f.unlink()
                            logger.debug(f"Cleaned staging: {f.name}")
                        except Exception:
                            pass
            else:
                error = result.get('error', 'archive failed')
                # Not a fatal error — video stays in processed, will retry
                self.db.log_step(video_id, 'archive', 'failed', message=error, duration=duration)
                logger.warning(f"Archive failed for {video_id}: {error} (will retry)")

        except Exception as e:
            duration = time.time() - start_time
            self.db.log_step(video_id, 'archive', 'failed', message=str(e), duration=duration)
            logger.error(f"Archive error for {video_id}: {e}")
            # Don't mark_failed — keep in processed state for retry


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

WatcherOrchestrator = DLCOrchestrator
