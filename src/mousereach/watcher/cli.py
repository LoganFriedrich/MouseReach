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
    print("=" * 70)
    print("MouseReach Watcher - Automated Video Pipeline")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Processing Root: {Paths.PROCESSING_ROOT}")
    print(f"  NAS Drive:       {Paths.NAS_DRIVE or '(not configured)'}")
    print(f"  Poll Interval:   {config.poll_interval_seconds}s")
    print(f"  Stability Wait:  {config.stability_wait_seconds}s")
    print(f"  DLC Config:      {config.dlc_config_path or '(not configured)'}")
    print(f"  DLC GPU:         {config.dlc_gpu_device}")
    print(f"  Auto Archive:    {'Yes' if config.auto_archive_approved else 'No'}")
    print(f"  Quarantine:      {config.get_quarantine_dir()}")
    print(f"  Logs:            {log_dir}")
    print()

    if dry_run:
        print("DRY RUN MODE - No files will be modified")
        print()
    if once:
        print("ONCE MODE - Will process pending items then exit")
        print()

    # Validate prerequisites
    logger.info("Validating prerequisites...")
    problems = []

    # Check NAS is configured
    if not Paths.NAS_DRIVE:
        problems.append("NAS drive not configured")
    elif not Paths.NAS_DRIVE.exists():
        problems.append(f"NAS drive does not exist: {Paths.NAS_DRIVE}")

    # Check processing root exists
    try:
        root = require_processing_root()
        if not root.exists():
            problems.append(f"Processing root does not exist: {root}")
    except Exception as e:
        problems.append(str(e))

    # Check DLC config if configured
    if config.dlc_config_path and not config.dlc_config_path.exists():
        problems.append(f"DLC config not found: {config.dlc_config_path}")

    # Check ffmpeg is on PATH
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        problems.append("ffmpeg not found on PATH")

    if problems:
        logger.error("Prerequisite validation failed:")
        for problem in problems:
            logger.error(f"  - {problem}")
        print("\nERROR: Prerequisites not met. Run 'mousereach-setup' to configure.", file=sys.stderr)
        sys.exit(1)

    logger.info("Prerequisites OK")

    # Create database
    try:
        db_path = require_processing_root() / "watcher.db"
        db = WatcherDB(db_path)
        logger.info(f"Database initialized at {db_path}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)

    # Import orchestrator (imported here to avoid circular imports)
    try:
        from mousereach.watcher.orchestrator import WatcherOrchestrator
    except ImportError as e:
        logger.error(f"Failed to import orchestrator: {e}")
        logger.error("The watcher orchestrator module is not yet implemented.")
        sys.exit(1)

    # Create orchestrator
    try:
        orchestrator = WatcherOrchestrator(config, db)
        logger.info("Orchestrator initialized")
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
    """Show current pipeline state."""
    # Parse args
    args = sys.argv[1:]
    json_output = '--json' in args
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
    print(f"  Quarantined:  {video_states.get('quarantined', 0)}")
    print(f"  Failed:       {video_states.get('failed', 0)}")
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
# INFO / DIAGNOSTICS COMMAND
# =============================================================================

def main_info():
    """Show drives, configured paths, and watcher readiness."""
    from mousereach.watcher.roles import print_machine_info

    print_machine_info()
