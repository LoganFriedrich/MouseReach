"""
CLI entry points for the MouseReach watcher.

Commands:
    mousereach-watch           Start the automated pipeline watcher
    mousereach-watch-status    Show current pipeline state
    mousereach-watch-reprocess Reset failed videos for reprocessing
    mousereach-watch-quarantine Manage quarantined files
"""

import sys
import signal
import logging
import threading
import subprocess
import json
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path, verbose: bool = False, quiet: bool = False):
    """
    Setup logging to console and rotating file.

    Args:
        log_dir: Directory for log files
        verbose: Enable debug logging
        quiet: Suppress info logging (errors only)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "watcher.log"

    # Determine log level
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.ERROR
    else:
        level = logging.INFO

    # Format
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(name)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # File handler (rotating, 10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_file


# =============================================================================
# MAIN WATCH COMMAND
# =============================================================================

def main_watch():
    """Start the automated pipeline watcher."""
    # Parse command line arguments
    args = sys.argv[1:]
    once = '--once' in args
    dry_run = '--dry-run' in args
    verbose = '--verbose' in args
    quiet = '--quiet' in args

    # Imports here to avoid import errors when showing help
    from mousereach.config import WatcherConfig, require_processing_root, Paths
    from mousereach.watcher.db import WatcherDB

    # Load config
    try:
        config = WatcherConfig.load()
    except Exception as e:
        print(f"ERROR: Failed to load watcher configuration: {e}", file=sys.stderr)
        print("\nRun: mousereach-setup", file=sys.stderr)
        sys.exit(1)

    # Setup logging
    try:
        log_dir = config.get_log_dir()
        log_file = setup_logging(log_dir, verbose=verbose, quiet=quiet)
        logger.info(f"Logging to {log_file}")
    except Exception as e:
        print(f"ERROR: Failed to setup logging: {e}", file=sys.stderr)
        sys.exit(1)

    # Print startup banner
    # Determine mode label
    if config.mode == 'processing_server':
        mode_label = "Processing Server"
    else:
        mode_label = "DLC PC"

    print("=" * 70)
    print(f"MouseReach Watcher - {mode_label} Mode")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Mode:            {mode_label}")
    print(f"  Processing Root: {Paths.PROCESSING_ROOT}")
    print(f"  NAS Drive:       {Paths.NAS_DRIVE or '(not configured)'}")
    print(f"  Poll Interval:   {config.poll_interval_seconds}s")
    print(f"  Stability Wait:  {config.stability_wait_seconds}s")
    if config.mode == 'processing_server':
        print(f"  DLC Staging:     {Paths.DLC_STAGING or '(not configured)'}")
        print(f"  Processing Dir:  {Paths.PROCESSING or '(not configured)'}")
        print(f"  Max Local Pending: {config.max_local_pending}")
    else:
        print(f"  DLC Config:      {config.dlc_config_path or '(not configured)'}")
        print(f"  DLC GPU:         {config.dlc_gpu_device}")
    print(f"  Auto Archive:    {'Yes' if config.auto_archive_approved else 'No'}")
    print(f"  Quarantine:      {config.get_quarantine_dir()}")
    print(f"  Logs:            {log_dir}")

    # Show priority animal if set
    priority_file = Paths.PROCESSING_ROOT / "priority_animal.json" if Paths.PROCESSING_ROOT else None
    if priority_file and priority_file.exists():
        try:
            with open(priority_file) as f:
                priority_data = json.load(f)
            animal = priority_data.get('animal_id', '?')
            set_at = priority_data.get('set_at', '')[:10]
            print(f"  PRIORITY ANIMAL: {animal} (since {set_at})")
        except Exception:
            pass

    print()

    if dry_run:
        print("DRY RUN MODE - No files will be modified")
        print()
    if once:
        print("ONCE MODE - Will process pending items then exit")
        print()

    # Validate prerequisites (mode-aware)
    logger.info("Validating prerequisites...")
    problems = []

    # Check processing root exists (both modes need this)
    try:
        root = require_processing_root()
        if not root.exists():
            problems.append(f"Processing root does not exist: {root}")
    except Exception as e:
        problems.append(str(e))

    if config.mode == 'processing_server':
        # Processing server needs: DLC_STAGING accessible, Processing/ writable
        if not Paths.DLC_STAGING:
            problems.append("DLC_STAGING not configured (check NAS drive)")
        elif not Paths.DLC_STAGING.exists():
            # Try to create it
            try:
                Paths.DLC_STAGING.mkdir(parents=True, exist_ok=True)
            except Exception:
                problems.append(f"DLC_STAGING not accessible: {Paths.DLC_STAGING}")

        if not Paths.PROCESSING:
            problems.append("PROCESSING path not configured")
    else:
        # DLC PC needs: NAS drive, DLC config, ffmpeg, GPU
        if not Paths.NAS_DRIVE:
            problems.append("NAS drive not configured")
        elif not Paths.NAS_DRIVE.exists():
            problems.append(f"NAS drive does not exist: {Paths.NAS_DRIVE}")

        if config.dlc_config_path and not config.dlc_config_path.exists():
            problems.append(f"DLC config not found: {config.dlc_config_path}")

        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            problems.append("ffmpeg not found on PATH")

        # GPU validation — ensure CUDA env is set up and check availability
        try:
            from mousereach.gpu import setup_gpu_env, check_gpu
            setup_gpu_env()
            gpu_status = check_gpu()
            if gpu_status.has_any_gpu:
                gpu_name = gpu_status.nvidia_gpu_name or gpu_status.torch_gpu_name or "unknown"
                logger.info(f"GPU detected: {gpu_name}")
            else:
                problems.append(
                    "No GPU available for DLC inference. "
                    "Check NVIDIA driver, CUDA toolkit, and cuDNN installation."
                )
            for w in gpu_status.warnings:
                logger.warning(f"GPU: {w}")
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")

    if problems:
        logger.error("Prerequisite validation failed:")
        for problem in problems:
            logger.error(f"  - {problem}")
        print("\nERROR: Prerequisites not met. Run 'mousereach-setup' to configure.", file=sys.stderr)
        sys.exit(1)

    logger.info("Prerequisites OK")

    # Create database
    try:
        # Use local DB path if configured (avoids SQLite-over-SMB issues)
        db_path = config.db_path or (require_processing_root() / "watcher.db")
        db = WatcherDB(db_path)
        logger.info(f"Database initialized at {db_path}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)

    # Import and create mode-aware orchestrator
    try:
        from mousereach.watcher.orchestrator import DLCOrchestrator, ProcessingOrchestrator
    except ImportError as e:
        logger.error(f"Failed to import orchestrator: {e}")
        sys.exit(1)

    try:
        if config.mode == 'processing_server':
            orchestrator = ProcessingOrchestrator(config, db)
            logger.info("ProcessingOrchestrator initialized")
        else:
            orchestrator = DLCOrchestrator(config, db)
            logger.info("DLCOrchestrator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        sys.exit(1)

    # Setup signal handler for graceful shutdown
    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        print("\nShutdown requested - finishing current video and stopping...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run watcher
    try:
        if dry_run:
            logger.info("Running in dry-run mode - scanning only")
            orchestrator.dry_run()
        elif once:
            logger.info("Running once - processing all pending items")
            orchestrator.run_once()
            logger.info("Once mode complete")
        else:
            logger.info("Starting watcher daemon - press Ctrl+C to stop")
            orchestrator.run(shutdown_event)
            logger.info("Watcher daemon stopped")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\nShutdown complete")
    except Exception as e:
        logger.exception(f"Watcher crashed: {e}")
        sys.exit(1)

    # Print summary
    summary = db.get_pipeline_summary()
    print()
    print("=" * 70)
    print("Final Summary")
    print("=" * 70)
    print(f"Total videos: {summary['videos']['total']}")
    print(f"Total collages: {summary['collages']['total']}")
    print()
    print("Video states:")
    for state, count in sorted(summary['videos']['by_state'].items()):
        print(f"  {state:20s}: {count}")
    print()
    print("Collage states:")
    for state, count in sorted(summary['collages']['by_state'].items()):
        print(f"  {state:20s}: {count}")
    print()


# =============================================================================
# STATUS COMMAND
# =============================================================================

def main_status():
    """Show current pipeline state.

    Usage:
        mousereach-watch-status              Overall counts
        mousereach-watch-status --by-animal  Per-animal breakdown with QC
        mousereach-watch-status --json       JSON output
        mousereach-watch-status --log N      Show last N log entries
    """
    # Parse args
    args = sys.argv[1:]
    json_output = '--json' in args
    by_animal = '--by-animal' in args
    log_limit = 10

    # Check for --log N
    for i, arg in enumerate(args):
        if arg == '--log' and i + 1 < len(args):
            try:
                log_limit = int(args[i + 1])
            except ValueError:
                print(f"ERROR: Invalid --log value: {args[i + 1]}", file=sys.stderr)
                sys.exit(1)

    # Load database
    try:
        from mousereach.config import require_processing_root
        from mousereach.watcher.db import WatcherDB

        db_path = require_processing_root() / "watcher.db"
        if not db_path.exists():
            print("ERROR: Watcher database not found. Run 'mousereach-watch' first.", file=sys.stderr)
            sys.exit(1)

        db = WatcherDB(db_path)
    except Exception as e:
        print(f"ERROR: Failed to load database: {e}", file=sys.stderr)
        sys.exit(1)

    # Get summary
    summary = db.get_pipeline_summary()

    # JSON output
    if json_output:
        # Add recent log entries
        summary['recent_activity'] = db.get_recent_log(log_limit)
        print(json.dumps(summary, indent=2))
        return

    # Human-readable output
    print("=" * 70)
    print("MouseReach Watcher Status")
    print("=" * 70)
    print()

    # Show priority animal if set
    priority_file = db_path.parent / "priority_animal.json"
    if priority_file.exists():
        try:
            with open(priority_file) as f:
                priority_data = json.load(f)
            animal = priority_data.get('animal_id', '?')
            set_at = priority_data.get('set_at', '')[:10]
            print(f"  ** PRIORITY ANIMAL: {animal} (since {set_at}) **")
            print(f"     Clear with: mousereach-watch-prioritize --clear")
            print()
        except Exception:
            pass

    # Collage summary
    print("Collages:")
    collage_states = summary['collages']['by_state']
    print(f"  Total:        {summary['collages']['total']}")
    print(f"  Discovered:   {collage_states.get('discovered', 0)}")
    print(f"  Validated:    {collage_states.get('validated', 0)}")
    print(f"  Stable:       {collage_states.get('stable', 0)}")
    print(f"  Cropping:     {collage_states.get('cropping', 0)}")
    print(f"  Cropped:      {collage_states.get('cropped', 0)}")
    print(f"  Quarantined:  {collage_states.get('quarantined', 0)}")
    print(f"  Failed:       {collage_states.get('failed', 0)}")
    print()

    # Video summary
    print("Videos:")
    video_states = summary['videos']['by_state']
    print(f"  Total:        {summary['videos']['total']}")
    print(f"  Discovered:   {video_states.get('discovered', 0)}")
    print(f"  Validated:    {video_states.get('validated', 0)}")
    print(f"  DLC Queued:   {video_states.get('dlc_queued', 0)}")
    print(f"  DLC Running:  {video_states.get('dlc_running', 0)}")
    print(f"  DLC Complete: {video_states.get('dlc_complete', 0)}")
    print(f"  Processing:   {video_states.get('processing', 0)}")
    print(f"  Processed:    {video_states.get('processed', 0)}")
    print(f"  Archived:     {video_states.get('archived', 0)}")
    print(f"  Outdated:     {video_states.get('outdated', 0)}")
    print(f"  Crystallized: {video_states.get('crystallized', 0)}")
    print(f"  Quarantined:  {video_states.get('quarantined', 0)}")
    print(f"  Failed:       {video_states.get('failed', 0)}")
    print()

    # Per-animal breakdown
    if by_animal:
        animals = db.get_animal_summary()
        if animals:
            # Also scan triage results for QC breakdown
            try:
                from mousereach.config import Paths
                processing_dir = Paths.PROCESSING
            except Exception:
                processing_dir = None

            triage_by_animal = {}
            if processing_dir and processing_dir.exists():
                for triage_file in processing_dir.glob("*_triage.json"):
                    try:
                        with open(triage_file) as f:
                            td = json.load(f)
                        vid = td.get('video_id', '')
                        # Extract animal_id from video_id (YYYYMMDD_CNTxxxx_...)
                        parts = vid.split('_')
                        if len(parts) >= 2:
                            aid = parts[1]  # CNTxxxx
                            if aid not in triage_by_animal:
                                triage_by_animal[aid] = {'approved': 0, 'needs_review': 0}
                            verdict = td.get('verdict', '')
                            if verdict == 'auto_approved':
                                triage_by_animal[aid]['approved'] += 1
                            elif verdict == 'needs_review':
                                triage_by_animal[aid]['needs_review'] += 1
                    except Exception:
                        pass

            # Header
            print("Per-Animal Pipeline Status:")
            print(f"  {'Animal':<12} {'Total':>5} {'DLC':>5} {'Proc':>5} {'Done':>5} {'Arch':>5} {'Fail':>5} {'QC OK':>5} {'Review':>6}")
            print(f"  {'-'*12} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*6}")

            for a in animals:
                aid = a['animal_id']
                s = a['states']
                dlc = s.get('dlc_complete', 0)
                proc = s.get('processing', 0)
                done = s.get('processed', 0)
                arch = s.get('archived', 0)
                fail = s.get('failed', 0) + s.get('quarantined', 0)
                qc = triage_by_animal.get(aid, {})
                ok = qc.get('approved', 0)
                rev = qc.get('needs_review', 0)

                print(f"  {aid:<12} {a['total']:>5} {dlc:>5} {proc:>5} {done:>5} {arch:>5} {fail:>5} {ok:>5} {rev:>6}")

            print()

    # Recent activity
    if log_limit > 0:
        recent = db.get_recent_log(log_limit)
        if recent:
            print(f"Recent Activity (last {log_limit} entries):")
            for entry in recent:
                timestamp = entry['created_at']
                video_id = entry['video_id']
                step = entry['step']
                status = entry['status']
                message = entry.get('message', '')
                duration = entry.get('duration_seconds')

                line = f"  [{timestamp}] {video_id} - {step}: {status}"
                if message:
                    line += f" ({message})"
                if duration:
                    line += f" [{duration:.1f}s]"
                print(line)
        else:
            print("Recent Activity: (none)")
        print()


# =============================================================================
# REPROCESS COMMAND
# =============================================================================

def main_reprocess():
    """Reset failed videos for reprocessing."""
    # Parse args
    args = sys.argv[1:]
    all_failed = '--all-failed' in args
    from_step = None

    # Check for --from-step
    for i, arg in enumerate(args):
        if arg == '--from-step' and i + 1 < len(args):
            from_step = args[i + 1]

    # Get video_id if provided
    video_id = None
    for arg in args:
        if not arg.startswith('--'):
            video_id = arg
            break

    if not video_id and not all_failed:
        print("Usage: mousereach-watch-reprocess [options] [video_id]", file=sys.stderr)
        print()
        print("Options:")
        print("  --all-failed      Reset ALL failed videos")
        print("  --from-step STEP  Reprocess from a specific step")
        print()
        print("Examples:")
        print("  mousereach-watch-reprocess 20250101_CNT0101_P1")
        print("  mousereach-watch-reprocess --all-failed")
        print("  mousereach-watch-reprocess --from-step dlc_complete 20250101_CNT0101_P1")
        sys.exit(1)

    # Load database
    try:
        from mousereach.config import require_processing_root
        from mousereach.watcher.db import WatcherDB

        db_path = require_processing_root() / "watcher.db"
        if not db_path.exists():
            print("ERROR: Watcher database not found. Run 'mousereach-watch' first.", file=sys.stderr)
            sys.exit(1)

        db = WatcherDB(db_path)
    except Exception as e:
        print(f"ERROR: Failed to load database: {e}", file=sys.stderr)
        sys.exit(1)

    # Get failed videos
    if all_failed:
        failed = db.get_videos_in_state('failed')
        if not failed:
            print("No failed videos found.")
            return

        print(f"Found {len(failed)} failed videos:")
        for video in failed:
            print(f"  - {video['video_id']}: {video['error_message']}")
        print()

        # Confirm
        response = input(f"Reset all {len(failed)} videos? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

        # Reset all
        for video in failed:
            try:
                db.reset_failed(video['video_id'], to_state=from_step)
                print(f"Reset: {video['video_id']}")
            except Exception as e:
                print(f"ERROR resetting {video['video_id']}: {e}", file=sys.stderr)

        print(f"\nReset {len(failed)} videos.")

    else:
        # Reset single video
        try:
            video = db.get_video(video_id)
            if not video:
                print(f"ERROR: Video not found: {video_id}", file=sys.stderr)
                sys.exit(1)

            if video['state'] != 'failed':
                print(f"WARNING: Video {video_id} is not in failed state (current: {video['state']})")
                response = input("Reset anyway? [y/N]: ")
                if response.lower() != 'y':
                    print("Cancelled.")
                    return

            db.reset_failed(video_id, to_state=from_step)
            print(f"Reset: {video_id}")
            if from_step:
                print(f"  Will restart from: {from_step}")

        except Exception as e:
            print(f"ERROR: Failed to reset video: {e}", file=sys.stderr)
            sys.exit(1)


# =============================================================================
# QUARANTINE COMMAND
# =============================================================================

def main_quarantine():
    """Manage quarantined files."""
    # Parse args
    args = sys.argv[1:]
    list_files = '--list' in args
    release_file = None
    purge = '--purge' in args

    # Check for --release
    for i, arg in enumerate(args):
        if arg == '--release' and i + 1 < len(args):
            release_file = args[i + 1]

    if not list_files and not release_file and not purge:
        print("Usage: mousereach-watch-quarantine [options]", file=sys.stderr)
        print()
        print("Options:")
        print("  --list           Show quarantined files with reasons")
        print("  --release FILE   Release a file from quarantine (after renaming)")
        print("  --purge          Delete all quarantined files (with confirmation)")
        print()
        print("Examples:")
        print("  mousereach-watch-quarantine --list")
        print("  mousereach-watch-quarantine --release 20250101_badname.mkv")
        print("  mousereach-watch-quarantine --purge")
        sys.exit(1)

    # Load database and config
    try:
        from mousereach.config import require_processing_root, WatcherConfig
        from mousereach.watcher.db import WatcherDB

        db_path = require_processing_root() / "watcher.db"
        if not db_path.exists():
            print("ERROR: Watcher database not found. Run 'mousereach-watch' first.", file=sys.stderr)
            sys.exit(1)

        db = WatcherDB(db_path)
        config = WatcherConfig.load()
    except Exception as e:
        print(f"ERROR: Failed to load database: {e}", file=sys.stderr)
        sys.exit(1)

    quarantine_dir = config.get_quarantine_dir()

    # List quarantined files
    if list_files:
        # Get quarantined collages and videos from DB
        quarantined_collages = db.get_collages_in_state('quarantined')
        quarantined_videos = db.get_videos_in_state('quarantined')

        print("=" * 70)
        print("Quarantined Files")
        print("=" * 70)
        print()

        if quarantined_collages:
            print(f"Collages ({len(quarantined_collages)}):")
            for collage in quarantined_collages:
                print(f"  {collage['filename']}")
                print(f"    Error: {collage['validation_error']}")
                print()

        if quarantined_videos:
            print(f"Videos ({len(quarantined_videos)}):")
            for video in quarantined_videos:
                print(f"  {video['video_id']}")
                print(f"    Error: {video['error_message']}")
                print()

        if not quarantined_collages and not quarantined_videos:
            print("No quarantined files.")
        else:
            print(f"Quarantine directory: {quarantine_dir}")

        return

    # Release file from quarantine
    if release_file:
        try:
            from mousereach.watcher.validator import FileValidator
            validator = FileValidator(quarantine_dir)
            validator.release_from_quarantine(release_file)
            print(f"Released: {release_file}")
        except Exception as e:
            print(f"ERROR: Failed to release file: {e}", file=sys.stderr)
            sys.exit(1)

    # Purge all quarantined files
    if purge:
        if not quarantine_dir.exists():
            print("Quarantine directory does not exist.")
            return

        files = list(quarantine_dir.glob('*'))
        if not files:
            print("Quarantine directory is empty.")
            return

        print(f"Found {len(files)} files in quarantine:")
        for f in files:
            print(f"  - {f.name}")
        print()

        response = input(f"DELETE all {len(files)} files? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

        for f in files:
            try:
                f.unlink()
                print(f"Deleted: {f.name}")
            except Exception as e:
                print(f"ERROR deleting {f.name}: {e}", file=sys.stderr)

        print(f"\nDeleted {len(files)} files from quarantine.")


# =============================================================================
# PRIORITIZE COMMAND
# =============================================================================

def main_prioritize():
    """Set, show, or clear the priority animal for the watcher.

    When a priority animal is set, the watcher processes that animal's
    videos first within each work tier (crop, DLC, pipeline, archive).
    Other animals' videos still get processed — just after the priority
    animal's items are handled.

    Usage:
        mousereach-watch-prioritize                Show current priority
        mousereach-watch-prioritize CNT0107        Set priority to CNT0107
        mousereach-watch-prioritize --clear        Remove priority
    """
    args = sys.argv[1:]
    clear = '--clear' in args
    positional = [a for a in args if not a.startswith('--')]

    from mousereach.config import require_processing_root, AnimalID
    from datetime import datetime

    priority_file = require_processing_root() / "priority_animal.json"

    # --clear: remove priority
    if clear:
        if priority_file.exists():
            priority_file.unlink()
            print("Priority cleared. Watcher will resume normal ordering.")
        else:
            print("No priority was set.")
        return

    # No arguments: show current priority
    if not positional:
        if priority_file.exists():
            try:
                with open(priority_file) as f:
                    data = json.load(f)
                print(f"Priority animal: {data['animal_id']}")
                print(f"Set at:          {data.get('set_at', '?')}")
            except Exception as e:
                print(f"Error reading priority file: {e}")
        else:
            print("No priority animal set. Watcher uses default ordering.")
        print()
        print("Usage:")
        print("  mousereach-watch-prioritize CNT0107    Set priority")
        print("  mousereach-watch-prioritize --clear     Clear priority")
        return

    # Set priority
    animal_id = positional[0].upper()

    parsed = AnimalID.parse(animal_id)
    if not parsed.get('valid'):
        print(f"ERROR: Invalid animal ID '{animal_id}': {parsed.get('error')}", file=sys.stderr)
        sys.exit(1)

    data = {
        'animal_id': animal_id,
        'set_at': datetime.now().isoformat(),
    }
    with open(priority_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Priority set: {animal_id}")
    print()
    print(f"The watcher will now prefer {animal_id}'s videos in all work queues.")
    print(f"Other videos are still processed, just with lower priority.")
    print()
    print(f"Clear with: mousereach-watch-prioritize --clear")


# =============================================================================
# PROCESS-ANIMAL COMMAND
# =============================================================================

def _resolve_nas_dir(configured_path, subfolder):
    """Resolve NAS directory, handling the doubled '! DLC Output' config issue."""
    from mousereach.config import Paths
    if configured_path and configured_path.exists():
        return configured_path
    fallbacks = []
    if Paths.NAS_DRIVE:
        fallbacks.append(Paths.NAS_DRIVE / subfolder)
        fallbacks.append(Paths.NAS_DRIVE.parent / subfolder)
    for fb in fallbacks:
        if fb.exists():
            return fb
    return None


def main_process_animal():
    """Queue all videos for a specific animal through the pipeline.

    Searches BOTH Single_Animal (pre-cropped) AND Multi-Animal (collages)
    folders. For collages that haven't been cropped yet, registers them in
    the watcher DB so the running watcher crops and processes them.

    Usage:
        mousereach-watch-process-animal CNT0107
        mousereach-watch-process-animal CNT0107 --dry-run
        mousereach-watch-process-animal CNT0107 --tray P   # Only pillar trays
    """
    args = sys.argv[1:]
    dry_run = '--dry-run' in args
    positional = [a for a in args if not a.startswith('--')]

    # Parse tray filter
    tray_filter = None
    for i, a in enumerate(sys.argv[1:]):
        if a == '--tray' and i + 2 < len(sys.argv):
            tray_filter = sys.argv[i + 2].upper()

    if not positional:
        print("Usage: mousereach-watch-process-animal <ANIMAL_ID> [--dry-run] [--tray P|E|F]",
              file=sys.stderr)
        print()
        print("Examples:")
        print("  mousereach-watch-process-animal CNT0107")
        print("  mousereach-watch-process-animal CNT0107 --dry-run")
        print("  mousereach-watch-process-animal CNT0107 --tray P")
        print()
        print("Searches both Single_Animal and Multi-Animal folders.")
        print("Singles are copied to DLC_Queue directly.")
        print("Collages are registered so the watcher crops + processes them.")
        sys.exit(1)

    animal_id = positional[0]

    # Imports
    from mousereach.config import (
        Paths, require_processing_root, get_video_id, AnimalID
    )
    from mousereach.watcher.db import WatcherDB
    from mousereach.watcher.validator import (
        validate_single_filename, validate_collage_filename
    )
    from mousereach.watcher.transfer import safe_copy
    from mousereach.video_prep.core.cropper import parse_collage_filename

    # Validate animal ID
    parsed = AnimalID.parse(animal_id)
    if not parsed.get('valid'):
        print(f"ERROR: Invalid animal ID '{animal_id}': {parsed.get('error')}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(f"Process Animal: {animal_id}")
    print("=" * 70)
    print()

    # Resolve directories
    single_dir = _resolve_nas_dir(
        Paths.SINGLE_ANIMAL_OUTPUT, "Unanalyzed/Single_Animal"
    )
    multi_dir = _resolve_nas_dir(
        Paths.MULTI_ANIMAL_SOURCE, "Unanalyzed/Multi-Animal"
    )

    if not single_dir and not multi_dir:
        print("ERROR: No NAS directories found.", file=sys.stderr)
        print("Run 'mousereach-setup' to configure NAS_DRIVE.", file=sys.stderr)
        sys.exit(1)

    if single_dir:
        print(f"Singles:   {single_dir}")
    if multi_dir:
        print(f"Collages:  {multi_dir}")
    if tray_filter:
        print(f"Tray filter: {tray_filter} only")

    # --- Phase 1: Find pre-cropped singles ---
    singles = []
    if single_dir:
        for f in sorted(single_dir.glob(f"*_{animal_id}_*.mp4")):
            result = validate_single_filename(f.name)
            if not result.valid:
                continue
            if tray_filter and result.parsed.get('tray_type') != tray_filter:
                continue
            singles.append(f)

    single_keys = set()  # date + tray combos we already have as singles
    for f in singles:
        result = validate_single_filename(f.name)
        if result.valid:
            single_keys.add(f"{result.parsed['date']}_{result.parsed['tray_type']}")

    # --- Phase 2: Find collages containing this animal ---
    collages = []  # (collage_path, collage_filename)
    if multi_dir:
        for collage_path in sorted(multi_dir.iterdir()):
            if not collage_path.is_file():
                continue
            if animal_id not in collage_path.name:
                continue
            try:
                info = parse_collage_filename(collage_path.name)
            except (ValueError, Exception):
                continue
            if animal_id not in info['animal_ids']:
                continue

            tray_suffix = info['last_part']
            # Apply tray filter
            if tray_filter and not tray_suffix.startswith(tray_filter):
                continue

            # Skip if we already have this as a single
            key = f"{info['date']}_{tray_suffix[0]}"
            if key in single_keys:
                continue

            collages.append(collage_path)

    # --- Show summary ---
    total = len(singles) + len(collages)
    if total == 0:
        print(f"\nNo videos found for {animal_id}")
        sys.exit(0)

    # Group by date for display
    date_summary = {}
    for f in singles:
        result = validate_single_filename(f.name)
        if result.valid:
            d = result.parsed['date']
            date_summary.setdefault(d, {'singles': 0, 'collages': 0})
            date_summary[d]['singles'] += 1

    for c in collages:
        try:
            info = parse_collage_filename(c.name)
            d = info['date']
            date_summary.setdefault(d, {'singles': 0, 'collages': 0})
            date_summary[d]['collages'] += 1
        except Exception:
            pass

    print(f"\nFound {total} videos across {len(date_summary)} session dates "
          f"({len(singles)} singles + {len(collages)} collages to crop):")
    for date in sorted(date_summary.keys()):
        info = date_summary[date]
        parts = []
        if info['singles']:
            parts.append(f"{info['singles']} singles")
        if info['collages']:
            parts.append(f"{info['collages']} collages")
        print(f"  {date}: {', '.join(parts)}")

    if dry_run:
        print(f"\nDRY RUN - no files will be modified")
        sys.exit(0)

    # Confirm
    print()
    response = input(f"Queue {total} videos for processing? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Initialize database
    db_path = require_processing_root() / "watcher.db"
    db = WatcherDB(db_path)

    dlc_queue = Paths.DLC_QUEUE
    if dlc_queue:
        dlc_queue.mkdir(parents=True, exist_ok=True)

    queued_singles = 0
    queued_collages = 0
    skipped = 0
    errors = 0

    # --- Queue singles: register + copy to DLC_Queue ---
    print("\nQueuing singles...")
    for f in singles:
        video_id = get_video_id(f.name)
        existing = db.get_video(video_id)
        if existing:
            state = existing['state']
            if state in ('processed', 'archived', 'dlc_complete', 'dlc_running',
                         'dlc_queued', 'processing'):
                print(f"  SKIP {video_id} (already {state})")
                skipped += 1
                continue

        result = validate_single_filename(f.name)
        if not result.valid:
            skipped += 1
            continue

        # Register if new
        if not existing:
            db.register_video(
                video_id=video_id,
                source_path=str(f),
                date=result.parsed['date'],
                animal_id=result.parsed['animal_id'],
                experiment=result.parsed['experiment'],
                cohort=result.parsed['cohort'],
                subject=result.parsed['subject'],
                tray_type=result.parsed['tray_type'],
                current_path=str(f)
            )
            db.update_state(video_id, 'validated', current_path=str(f))

        # Copy to DLC_Queue
        if dlc_queue:
            dest = dlc_queue / f.name
            if dest.exists():
                db.update_state(video_id, 'dlc_queued', current_path=str(dest))
                queued_singles += 1
                print(f"  QUEUED {video_id} (already in DLC_Queue)")
            else:
                try:
                    if safe_copy(f, dest, verify=True):
                        db.update_state(video_id, 'dlc_queued', current_path=str(dest))
                        db.log_step(video_id, 'process_animal', 'queued',
                                    message=f"Queued by process-animal for {animal_id}")
                        queued_singles += 1
                        print(f"  QUEUED {video_id}")
                    else:
                        print(f"  ERROR {video_id} (copy failed)")
                        errors += 1
                except Exception as e:
                    print(f"  ERROR {video_id}: {e}")
                    errors += 1

    # --- Queue collages: register so watcher crops them ---
    if collages:
        print("\nRegistering collages for watcher to crop...")
        for collage_path in collages:
            filename = collage_path.name
            existing = db.get_collage(filename)

            if existing:
                state = existing['state']
                if state in ('cropped', 'archived'):
                    print(f"  SKIP {filename} (already {state})")
                    skipped += 1
                    continue
                # If discovered/validated/failed, leave it - watcher handles it
                print(f"  EXISTS {filename} ({state})")
                queued_collages += 1
                continue

            # Register new collage
            try:
                result = validate_collage_filename(filename)
                db.register_collage(filename=filename, source_path=str(collage_path))
                if result.valid and result.parsed:
                    db.update_collage_state(
                        filename=filename,
                        new_state='validated',
                        date=result.parsed['date'],
                        animal_ids=','.join(result.parsed['animal_ids']),
                        tray_suffix=f"{result.parsed['tray_type']}{result.parsed['tray_run']}"
                    )
                    # Mark as stable immediately (these files aren't being written)
                    db.update_collage_state(filename, 'stable')
                queued_collages += 1
                print(f"  REGISTERED {filename}")
            except Exception as e:
                print(f"  ERROR {filename}: {e}")
                errors += 1

    # Summary
    print()
    print("=" * 70)
    print(f"Summary for {animal_id}:")
    print(f"  Singles queued for DLC:       {queued_singles}")
    print(f"  Collages registered to crop:  {queued_collages}")
    print(f"  Already in pipeline:          {skipped}")
    print(f"  Errors:                       {errors}")
    print("=" * 70)

    if queued_singles + queued_collages > 0:
        print(f"\nThe running watcher will automatically:")
        if queued_collages > 0:
            print(f"  1. Crop {queued_collages} collages into singles")
            print(f"  2. Run DLC on all singles")
        else:
            print(f"  1. Run DLC on {queued_singles} singles")
        print(f"  {'3' if queued_collages else '2'}. Run segmentation + reach detection + outcome detection")
        print(f"\nMonitor progress: mousereach-watch-status")


# =============================================================================
# INFO / DIAGNOSTICS COMMAND
# =============================================================================

def main_info():
    """Show drives, configured paths, and watcher readiness."""
    from mousereach.watcher.roles import print_machine_info

    print_machine_info()


# =============================================================================
# TOGGLE / PAUSE / RESUME COMMANDS
# =============================================================================

def _get_pause_file():
    """Return the path to the pause sentinel file."""
    from mousereach.config import require_processing_root
    return require_processing_root() / "watcher_paused.flag"


def main_toggle():
    """Toggle the watcher between filming (paused) and processing (active) modes.

    When paused, the running watcher skips all work and waits.
    When active, normal processing resumes.

    Usage:
        mousereach-watch-toggle          Toggle current state
        mousereach-watch-toggle --status  Show current state only
    """
    args = sys.argv[1:]
    status_only = '--status' in args

    try:
        pause_file = _get_pause_file()
    except Exception as e:
        print(f"ERROR: Could not determine processing root: {e}", file=sys.stderr)
        print("Run 'mousereach-setup' to configure paths.", file=sys.stderr)
        sys.exit(1)

    currently_paused = pause_file.exists()

    if status_only:
        if currently_paused:
            print("Watcher is PAUSED (filming mode).")
            print("Run 'mousereach-watch-toggle' to resume processing.")
        else:
            print("Watcher is ACTIVE (processing mode).")
            print("Run 'mousereach-watch-toggle' to pause for filming.")
        return

    if currently_paused:
        pause_file.unlink()
        print("=" * 50)
        print("  Watcher RESUMED — processing mode active")
        print("  DLC and cropping will run during downtime.")
        print("=" * 50)
    else:
        pause_file.parent.mkdir(parents=True, exist_ok=True)
        pause_file.write_text("Watcher paused for filming.\n")
        print("=" * 50)
        print("  Watcher PAUSED -- filming mode active")
        print("  DLC processing is suspended.")
        print("  Toggle again when filming is done.")
        print("=" * 50)


# =============================================================================
# VERSION CHECK COMMAND
# =============================================================================

def main_version_check():
    """Check version compliance of archived videos.

    Compares each archived video's processing manifest against the current
    pipeline_versions.json to find outdated videos.

    Usage:
        mousereach-version-check              Show report only
        mousereach-version-check --mark       Also mark outdated videos for reprocessing
        mousereach-version-check --init       Initialize pipeline_versions.json
    """
    args = sys.argv[1:]
    mark = '--mark' in args
    init = '--init' in args

    from mousereach.config import require_processing_root, Paths
    from mousereach.watcher.db import WatcherDB

    if init:
        from mousereach.pipeline.versions import initialize_versions
        try:
            data = initialize_versions()
            print("Initialized pipeline_versions.json:")
            print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Load database
    try:
        db_path = require_processing_root() / "watcher.db"
        if not db_path.exists():
            print("ERROR: Watcher database not found. Run 'mousereach-watch' first.", file=sys.stderr)
            sys.exit(1)
        db = WatcherDB(db_path)
    except Exception as e:
        print(f"ERROR: Failed to load database: {e}", file=sys.stderr)
        sys.exit(1)

    nas_root = Paths.NAS_ROOT
    if not nas_root:
        print("ERROR: NAS_ROOT not configured", file=sys.stderr)
        sys.exit(1)

    from mousereach.watcher.reprocessor import ReprocessingScanner
    scanner = ReprocessingScanner(db, nas_root)

    if mark:
        summary = scanner.scan(mark_outdated=True)
        print(scanner.get_version_report())
        if summary['outdated'] > 0:
            print(f"\n{summary['outdated']} videos marked as 'outdated' in the database.")
            print("The watcher will automatically reprocess them.")
    else:
        print(scanner.get_version_report())
        outdated = summary = scanner.scan(mark_outdated=False)
        if summary['outdated'] > 0:
            print(f"To mark these for reprocessing, run:")
            print(f"  mousereach-version-check --mark")


# =============================================================================
# CRYSTALLIZE COMMAND
# =============================================================================

def main_crystallize():
    """Lock archived videos against reprocessing.

    Usage:
        mousereach-crystallize --cohort CNT01 --label "PNAS_2026"
        mousereach-crystallize --videos "vid1,vid2" --label "prelim"
        mousereach-crystallize --list                 Show crystallized videos
    """
    args = sys.argv[1:]

    # Parse arguments
    cohort = None
    label = "unnamed_timepoint"
    video_ids = None
    list_mode = '--list' in args

    for i, arg in enumerate(args):
        if arg == '--cohort' and i + 1 < len(args):
            cohort = args[i + 1]
        elif arg == '--label' and i + 1 < len(args):
            label = args[i + 1]
        elif arg == '--videos' and i + 1 < len(args):
            video_ids = [v.strip() for v in args[i + 1].split(',')]

    if not list_mode and not cohort and not video_ids:
        print("Usage: mousereach-crystallize [options]", file=sys.stderr)
        print()
        print("Options:")
        print("  --cohort COHORT   Crystallize all archived videos for cohort")
        print("  --videos V1,V2    Crystallize specific video IDs")
        print("  --label LABEL     Human-readable label (e.g. 'PNAS_2026_submission')")
        print("  --list            Show currently crystallized videos")
        print()
        print("Examples:")
        print("  mousereach-crystallize --cohort CNT01 --label 'PNAS_2026'")
        print("  mousereach-crystallize --videos '20250624_CNT0115_P2' --label 'test'")
        print("  mousereach-crystallize --list")
        sys.exit(1)

    # Load database
    from mousereach.config import require_processing_root
    from mousereach.watcher.db import WatcherDB

    try:
        db_path = require_processing_root() / "watcher.db"
        if not db_path.exists():
            print("ERROR: Watcher database not found.", file=sys.stderr)
            sys.exit(1)
        db = WatcherDB(db_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if list_mode:
        crystallized = db.get_videos_in_state('crystallized')
        if not crystallized:
            print("No crystallized videos.")
            return

        print("=" * 70)
        print("Crystallized Videos")
        print("=" * 70)
        print()

        # Group by label
        by_label = {}
        for v in crystallized:
            lbl = v.get('crystallized_label', '(no label)')
            by_label.setdefault(lbl, []).append(v)

        for lbl, videos in sorted(by_label.items()):
            print(f"  Label: {lbl}")
            print(f"  Crystallized by: {videos[0].get('crystallized_by', '?')}")
            print(f"  Date: {videos[0].get('crystallized_at', '?')[:10]}")
            print(f"  Videos ({len(videos)}):")
            for v in videos[:10]:
                print(f"    - {v['video_id']}")
            if len(videos) > 10:
                print(f"    ... and {len(videos) - 10} more")
            print()

        print(f"Total crystallized: {len(crystallized)}")
        return

    # Crystallize
    from mousereach.pipeline.versions import crystallize_videos
    import getpass

    try:
        username = getpass.getuser()
    except Exception:
        username = None

    count = crystallize_videos(
        db=db,
        video_ids=video_ids,
        cohort=cohort,
        label=label,
        crystallized_by=username,
    )

    print(f"Crystallized {count} videos with label '{label}'.")
    if count > 0:
        print("These videos will NOT be reprocessed when tool versions change.")
        print(f"To unlock: mousereach-uncrystallize --label '{label}'")


def main_uncrystallize():
    """Unlock crystallized videos (allows reprocessing again).

    Usage:
        mousereach-uncrystallize --label "PNAS_2026"
        mousereach-uncrystallize --videos "vid1,vid2"
    """
    args = sys.argv[1:]
    label = None
    video_ids = None

    for i, arg in enumerate(args):
        if arg == '--label' and i + 1 < len(args):
            label = args[i + 1]
        elif arg == '--videos' and i + 1 < len(args):
            video_ids = [v.strip() for v in args[i + 1].split(',')]

    if not label and not video_ids:
        print("Usage: mousereach-uncrystallize [options]", file=sys.stderr)
        print()
        print("Options:")
        print("  --label LABEL     Uncrystallize all videos with this label")
        print("  --videos V1,V2    Uncrystallize specific video IDs")
        print()
        print("Examples:")
        print("  mousereach-uncrystallize --label 'PNAS_2026'")
        sys.exit(1)

    from mousereach.config import require_processing_root
    from mousereach.watcher.db import WatcherDB
    from mousereach.pipeline.versions import uncrystallize_videos

    try:
        db_path = require_processing_root() / "watcher.db"
        db = WatcherDB(db_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    count = uncrystallize_videos(db=db, video_ids=video_ids, label=label)
    print(f"Uncrystallized {count} videos.")
    if count > 0:
        print("These videos can now be reprocessed when tool versions change.")
