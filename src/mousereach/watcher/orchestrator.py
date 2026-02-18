"""
NAS/DLC PC orchestrator for the MouseReach watcher.

This machine's job: scan NAS for collages -> crop -> DLC -> stage back to NAS.
Pipeline steps (seg/reach/outcomes) are NOT run here — the processing PC
picks up staged files from the NAS.

Flow per collage:
  1. Copy collage from NAS (D:) to local working dir (A:)
  2. Crop into single-animal videos
  3. DLC each single (using local GPU)
  4. Move video + h5 to NAS staging folder for processing PC
  5. Next collage
"""

import json
import os
import time
import random
import socket
import logging
import threading

# DLC 2.3.x requires Keras 2 APIs — tell TF to use tf-keras
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
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
    NAS/DLC PC orchestrator.

    Processes one collage at a time:
    - Scans NAS for new collages
    - Crops to singles
    - Runs DLC inference on each single
    - Stages video+h5 back to NAS for the processing PC
    """

    def __init__(self, config: WatcherConfig, db: WatcherDB):
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

        # Staging directory on NAS for processing PC to pick up
        self.staging_dir = Paths.DLC_STAGING
        if self.staging_dir:
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Staging directory: {self.staging_dir}")

        logger.info(f"WatcherOrchestrator initialized on {self.hostname}")
        logger.info(f"Working directory: {self.working_dir}")

    def run(self, shutdown_event: threading.Event):
        """Main orchestrator loop: scan + process in a single thread."""
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
                time.sleep(5)

        self.shutdown()
        logger.info("WatcherOrchestrator stopped")

    def run_once(self):
        """Run one full cycle: scan + process all pending items, then exit."""
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
        """Scan without processing. Shows what would be done."""
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
    # WORK QUEUE
    # =========================================================================

    def _get_next_work_item(self):
        """
        Get next work item — priority animal first, then Pillar-first, random within tier.

        If a priority animal is set (via mousereach-watch-prioritize),
        items belonging to that animal are preferred within each tier.

        Priority:
        1. Stage DLC-complete videos to NAS (finish what's done)
        2. Run DLC on a queued single
        3. Crop next collage (only when nothing in-flight)
        """
        priority_animal = self._get_priority_animal()

        # Priority 1: Videos with DLC complete — stage to NAS
        videos = self.db.get_videos_in_state('dlc_complete')
        if videos:
            # For staging, prefer priority animal but no randomization (stage ASAP)
            if priority_animal:
                preferred = [v for v in videos if self._matches_priority(v, priority_animal, 'animal_id')]
                pick = preferred[0] if preferred else videos[0]
            else:
                pick = videos[0]
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
            else:
                logger.warning(f"Unknown work type: {work_type}")

        except Exception as e:
            error_msg = f"{work_type} failed: {str(e)}"
            logger.error(f"Work item {work_id} failed: {e}", exc_info=True)

            if work_type == 'collage':
                self.db.update_collage_state(work_id, 'failed', validation_error=error_msg)
            else:
                self.db.mark_failed(work_id, error_msg)

    # =========================================================================
    # DLC COMPLETION SCANNING
    # =========================================================================

    def _scan_for_dlc_completions(self) -> int:
        """
        Scan for h5 files matching videos in dlc_queued or dlc_running state.

        DLC outputs h5 files to DLC_Queue (same dir as input video).
        This detects them and advances state to dlc_complete.
        """
        # Check DLC_Queue for h5 outputs (DLC writes output next to input)
        dlc_queue_dir = Paths.DLC_QUEUE
        if not dlc_queue_dir or not dlc_queue_dir.exists():
            return 0

        count = 0

        for state in ('dlc_queued', 'dlc_running'):
            videos = self.db.get_videos_in_state(state)
            for video in videos:
                video_id = video['video_id']
                # Look for DLC h5 output in DLC_Queue
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
            else:
                error_msg = results[0].get('error', 'Unknown DLC error') if results else 'No results'
                raise RuntimeError(f"DLC failed: {error_msg}")

        except Exception as e:
            duration = time.time() - start_time
            self.db.mark_failed(video_id, str(e))
            self.db.log_step(video_id, 'dlc', 'failed', message=str(e), duration=duration)
            raise

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

        except Exception as e:
            duration = time.time() - start_time
            self.db.mark_failed(video_id, str(e))
            self.db.log_step(video_id, 'stage_to_nas', 'failed', message=str(e), duration=duration)
            raise

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
        logger.info("WatcherOrchestrator shutting down gracefully")
        try:
            if self.working_dir.exists():
                for file_path in self.working_dir.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                        logger.debug(f"Cleaned up: {file_path.name}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
