"""
CLI entry points for the MouseReach watcher.

Commands:
    mousereach-watch           Start the automated pipeline watcher
    mousereach-watch-status    Show current pipeline state
    mousereach-watch-reprocess Reset failed videos for reprocessing
    mousereach-watch-quarantine Manage quarantined files
    mousereach-watch-toggle    Pause / resume the watcher by hand
    mousereach-watch-recorders Recording programs that pause the watcher
"""

import os
import re
import sys
import time
import signal
import logging
import tempfile
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


def _resolve_db_path():
    """The ONE way any command in this file picks the watcher database.

    The daemon honoured the node's config override (db_path in
    ~/.mousereach/config.json, set because SQLite over a network share loses
    writes); every other command hardcoded the fallback. On this machine that
    meant seven commands -- status, reprocess, quarantine, process-animal,
    version-check, crystallize, uncrystallize -- silently read a database last
    written in February while the daemon wrote to a different file.
    Crystallize, the brake that protects published videos from reprocessing,
    would therefore find no videos and protect nothing.

    Prints the resolved path so a wrong database is loud instead of silent.
    The choice itself lives in db_location.resolve_watcher_db_path, so commands
    outside this module (mousereach-route-to-queue) pick the same file.
    """
    from mousereach.watcher.db_location import resolve_watcher_db_path
    path = resolve_watcher_db_path()
    print(f"[watcher db] {path}")
    return path


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

def check_nas_root(nas_root, origin: str):
    """Return a problem description if NAS_ROOT is not a real pipeline root.

    ``pipeline_versions.json`` is the marker. It declares the current DLC model
    and algorithm versions, so a NAS_ROOT without one is not a pipeline root --
    it is a wrong turn that silently repoints every derived path (staging,
    Analyzed, Review, Failed, intake) and leaves version currency dead. A node
    can run for months in that state looking perfectly healthy, which is why
    this is a refuse-to-start rather than a warning.

    Args:
        nas_root: the resolved Paths.NAS_ROOT (may be None)
        origin: Paths.NAS_ROOT_ORIGIN -- 'config', 'fallback' or 'unset'

    Returns:
        A problem string, or None if the root looks like a pipeline.
    """
    if not nas_root:
        return ("nas_root not configured and no nas_drive to fall back on -- "
                "set nas_root in ~/.mousereach/config.json")
    nas_root = Path(nas_root)
    if not nas_root.exists():
        return f"NAS root does not exist: {nas_root}"
    if (nas_root / "pipeline_versions.json").exists():
        return None

    detail = (
        f"NAS root has no pipeline_versions.json: {nas_root}\n"
        f"      Everything NAS-side derives from this path -- staging, Analyzed, "
        f"Review, Failed, intake -- so output would land here too."
    )
    if origin == 'fallback':
        detail += (
            "\n      This path was NOT configured: 'nas_root' is unset, so it fell "
            "back to <nas_drive>\\! DLC Output (the pre-2026 layout).\n"
            "      Set 'nas_root' in ~/.mousereach/config.json to the pipeline root, "
            "e.g.\n"
            '        "nas_root": "<NAS>\\\\MouseReach_Pipeline"\n'
            "      or re-run mousereach-setup, which fills it in from this machine's "
            "lab profile."
        )
    else:
        detail += (
            "\n      'nas_root' points here explicitly. Either repoint it at the "
            "pipeline root, or initialize one here with "
            "mousereach-version-check --init."
        )
    return detail


def _maybe_print_help(usage: str):
    """Print usage and exit 0 when -h/--help is asked for.

    These entry points parse sys.argv by hand (deliberately -- no argparse
    rewrite of running production commands), which left them with no --help
    at all. This gate is the smallest possible fix: behavior is unchanged for
    every other invocation."""
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print(usage.strip())
        sys.exit(0)


def main_watch():
    """Start the automated pipeline watcher."""
    _maybe_print_help("""
usage: mousereach-watch [--once] [--dry-run] [--verbose] [--quiet]

Start the automated pipeline watcher daemon for this node (role and paths
from ~/.mousereach/config.json; run mousereach-setup to configure).

  --once      Run one full scan+process cycle, then exit.
  --dry-run   Show what would be processed without doing it.
  --verbose   Debug-level logging.
  --quiet     Warnings and errors only.
""")
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
    _origin = '' if Paths.NAS_ROOT_ORIGIN == 'config' else f'  [{Paths.NAS_ROOT_ORIGIN}]'
    print(f"  NAS Root:        {Paths.NAS_ROOT or '(not configured)'}{_origin}")
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

    nas_problem = check_nas_root(Paths.NAS_ROOT, Paths.NAS_ROOT_ORIGIN)
    if nas_problem:
        problems.append(nas_problem)
    else:
        # WHY refuse to start on an unmigrated share: this watcher works in the
        # stage-layout folders only. Where a retired folder name is still a real
        # folder, work waiting there would never be picked up and nothing would
        # say so. Stat only -- the folder is neither listed nor created.
        from mousereach.pipeline.pipe_structure import (
            UNMIGRATED_MESSAGE, retired_folders_present)
        for rel in retired_folders_present(Paths.NAS_ROOT):
            problems.append(f"{Paths.NAS_ROOT / rel}: {UNMIGRATED_MESSAGE}")

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
        # DLC PC needs: the shared pipeline root, DLC config, ffmpeg, GPU.
        # NAS_ROOT is what every shared path hangs off; a lab that configured
        # only nas_root (the documented setting) used to be refused here for
        # lacking the older nas_drive.
        if not Paths.NAS_ROOT:
            problems.append("NAS root not configured (nas_root in ~/.mousereach/config.json)")
        elif not Paths.NAS_ROOT.exists():
            problems.append(f"NAS root does not exist: {Paths.NAS_ROOT}")

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

    # SINGLE-INSTANCE GUARD. Two watchers on one state database is never a
    # valid configuration (interleaved work selection, double-claimed items),
    # and on 2026-09-01 a day of deploy kill/restart cycles showed how easily
    # a second instance can go unnoticed: process filters based on command
    # lines miss instances whose command line is not visible. A named mutex
    # is per-machine, costs nothing, and the OS releases it on ANY exit,
    # force-kill included. Exit code 0 on "already running": an eager
    # relauncher gets an orderly no-op, not a failure to retry.
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            _mutex = kernel32.CreateMutexW(None, True, "Global\\mousereach-watcher-singleton")
            ERROR_ALREADY_EXISTS = 183
            if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
                logger.warning("Another watcher instance already holds the "
                               "singleton mutex on this machine -- exiting.")
                print("Watcher already running on this machine; exiting.")
                sys.exit(0)
            # Keep a reference so the handle lives as long as the process.
            globals()["_watcher_singleton_mutex"] = _mutex
        except Exception as e:
            logger.warning(f"Single-instance guard unavailable ({e}); continuing")

    # Create database
    try:
        # Use local DB path if configured (avoids SQLite-over-SMB issues)
        db_path = _resolve_db_path()
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

        db_path = _resolve_db_path()
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

    # Is it paused, why, and which shared folders does it work from.
    _print_pause_and_stage_folders()

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

    # Anything sitting in a state the watcher only sets WHILE it is working
    # was interrupted -- a stop, a reboot, a crash. Say so plainly rather than
    # leaving it to look like busy work that never finishes.
    interrupted = {s: video_states.get(s, 0)
                   for s in ('dlc_running', 'archiving')
                   if video_states.get(s, 0)}
    if interrupted:
        print()
        print("  Interrupted mid-run: "
              + ", ".join(f"{n} {s}" for s, n in sorted(interrupted.items())))
        print("  The next watcher start puts these back in the queue by itself.")
        print("  Nothing is lost and nothing needs doing by hand.")
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

        # Every node, and whether its quiet is deliberate. WHY this is here: a
        # paused node and a dead node both simply stop appearing above, and a node
        # pauses whenever a recording program is open -- which is routinely left
        # open after the recording is finished. Without this, the only way to tell
        # a normal quiet machine from a dead watcher is to walk to it.
        try:
            from mousereach.watcher import node_status
            from mousereach.config import Paths
            if Paths.NAS_ROOT:
                print("Nodes:")
                print(node_status.describe(Paths.NAS_ROOT))
                print()
        except Exception as e:
            print("Nodes: (could not be read: %s)" % e)
            print()


# =============================================================================
# REPROCESS COMMAND
# =============================================================================

def main_reprocess():
    """Reset failed videos for reprocessing."""
    _maybe_print_help("""
usage: mousereach-watch-reprocess [video_id ...] [--all-failed] [--from-step STEP]

Reset failed videos so the watcher picks them up again.

  video_id ...       Specific video id(s) to reset.
  --all-failed       Reset every video currently in the failed state.
  --from-step STEP   Restart from this pipeline step instead of the beginning.
""")
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

        db_path = _resolve_db_path()
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
    _maybe_print_help("""
usage: mousereach-watch-quarantine [--list] [--release FILE] [--purge]

Manage files the watcher quarantined instead of processing.

  --list          Show what is quarantined and why.
  --release FILE  Move one file back into the normal flow.
  --purge         Permanently delete everything quarantined (confirm first).
""")
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

        db_path = _resolve_db_path()
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
    _maybe_print_help("""
usage: mousereach-watch-process-animal ANIMAL_ID [--dry-run] [--tray T]

Queue every video for one animal through the pipeline (searches both
Single_Animal pre-cropped videos and Multi-Animal collages).

  ANIMAL_ID   e.g. CNT0107
  --dry-run   Show what would be queued without queueing it.
  --tray T    Only this tray type (e.g. P for pillar).
""")
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

    # Singles a GPU node has already claimed for pose. WHY looked up: a claim
    # MOVES the video into Single_Animal/.inflight/<machine>/, out of the
    # top-level listing above. Those videos are being posed right now, so they
    # are never queued again here -- but they still count as "we have this as
    # a single", or the collage they were cut from would be sent to be cut
    # (and posed) a second time.
    claimed = {}
    if single_dir:
        try:
            from mousereach.census.runner import claimed_singles
            claimed = {vid: host for vid, host in claimed_singles(single_dir).items()
                       if f"_{animal_id}_" in f"{vid}.mp4"}
        except Exception as e:
            print(f"[!] Could not read singles claimed for pose: {_ascii(e)}",
                  file=sys.stderr)

    single_keys = set()  # date + tray combos we already have as singles
    for name in [f.name for f in singles] + [f"{vid}.mp4" for vid in claimed]:
        result = validate_single_filename(name)
        if result.valid:
            single_keys.add(f"{result.parsed['date']}_{result.parsed['tray_type']}")

    if claimed:
        print(f"\nAlready being posed by another machine ({len(claimed)}), "
              f"not queued again:")
        for vid in sorted(claimed):
            print(f"  {vid}  (claimed by {_ascii(claimed[vid])})")

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
    db_path = _resolve_db_path()
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
        if not f.is_file():
            # WHY checked again: the listing above was taken before the person
            # confirmed. In the meantime a GPU node may have claimed this single
            # (moved it into .inflight/<machine>/) and be posing it; registering
            # or copying it here would pose it twice.
            print(f"  SKIP {video_id} (no longer in the singles folder -- claimed "
                  f"for pose by another machine, or removed)")
            skipped += 1
            continue
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
    _maybe_print_help("""
usage: mousereach-watch-info

Show this machine's drives, configured mousereach paths, detected lab role,
and whether the watcher could run here. Takes no options.
""")
    from mousereach.watcher.roles import print_machine_info

    print_machine_info()


# =============================================================================
# TOGGLE / PAUSE / RESUME COMMANDS
# =============================================================================

def _get_pause_file():
    """Return the path to the pause sentinel file."""
    from mousereach.config import require_processing_root
    return require_processing_root() / "watcher_paused.flag"


def _ascii(text) -> str:
    """Printable ASCII only. WHY: a Windows console cannot encode every
    character a path, a typed program name or an exception message may
    contain, and one such character crashes the print (house rule)."""
    return str(text).encode("ascii", "replace").decode("ascii")


# Said the same way recording_guard.HAND_PAUSE_REASON says it; used only when
# that module could not be imported at all.
_HAND_PAUSE_FALLBACK = "paused by hand (watcher_paused.flag)"


def _pause_state() -> dict:
    """Why this watcher is or is not paused, decided the way the watcher
    itself decides (recording_guard.pause_reason).

    WHY one helper: the watcher pauses for two independent reasons -- the
    hand pause (watcher_paused.flag, set by mousereach-watch-toggle or the
    Pause button) and a recording program listed with
    mousereach-watch-recorders being open. If every command checked the two
    its own way, sooner or later one of them would say ACTIVE while the
    watcher sat paused, and a paused watcher otherwise looks exactly like an
    idle one.

    Never raises. Keys:
        root_error  why the hand pause could not be checked, or None
        manual      True when watcher_paused.flag exists
        names       configured recording program names
        grace       resume grace period in seconds
        recording   the recording guard's reason, or None
        overall     the reason the watcher is paused, or None = not paused
    """
    state = {'root': None, 'root_error': None, 'manual': False,
             'names': [], 'grace': 120, 'recording': None, 'overall': None}
    try:
        pause_file = _get_pause_file()
        state['root'] = pause_file.parent
        state['manual'] = pause_file.exists()
    except Exception as e:
        state['root_error'] = _ascii(e)

    try:
        from mousereach.config import WatcherConfig
        from mousereach.watcher import recording_guard as rg
        cfg = WatcherConfig.load()
        state['names'] = list(cfg.pause_while_running or [])
        state['grace'] = cfg.pause_resume_grace_seconds
        # A fresh guard has no memory of a program that closed a minute ago,
        # so from a one-off command it can report "running" but never the
        # grace countdown; the running watcher's own guard does that. scan is
        # passed explicitly so it is looked up when called, not when
        # recording_guard was imported.
        guard = rg.RecordingGuard(state['names'], grace_seconds=state['grace'],
                                  scan=rg.running_programs)
        state['recording'] = guard.reason()
        state['overall'] = rg.pause_reason(processing_root=state['root'],
                                           config=cfg, guard=guard)
    except Exception as e:
        # Fail safe, exactly like the guard: a check that could not run is
        # reported as a reason to be paused, never as "not paused". WHY:
        # recording must win, and a person told "not paused" looks no further.
        state['recording'] = f"cannot check for recording programs: {_ascii(e)}"
        state['overall'] = (_HAND_PAUSE_FALLBACK if state['manual']
                            else state['recording'])
    return state


def _recording_hint(reason: str) -> str:
    """What a person does about a recording-program pause."""
    if reason.startswith("cannot check"):
        return ("It stays paused until this check works, because recording "
                "must win (see mousereach-watch-recorders).")
    return ("Recording always wins: work starts again by itself once the "
            "recording program has been closed.")


# The shared folders a video rests in between machines and people, in the
# order a video passes through them. (label, attribute of config.Paths)
_STAGE_FOLDERS = (
    ("Cut videos waiting for a pose", "SINGLE_ANIMAL_OUTPUT"),
    ("Posed videos waiting for analysis", "DLC_STAGING"),
    ("Triage review queue", "TRIAGE_REVIEW"),
    ("Deep review queue", "DEEP_REVIEW"),
)


def _print_pause_and_stage_folders():
    """The "Pause:" line and "Stage folders:" block of mousereach-watch-status.

    WHY the pause line: a paused watcher looks exactly like an idle one -- the
    counts just stop moving -- and it can now be paused two ways, by hand or
    by a recording program being open. The line says which, in the words the
    watcher itself uses.

    WHY the folders: the operator of a GPU node had no command that showed
    which shared staging folder the node works from, so confirming a node was
    pointed at the right place meant opening config files by hand.
    """
    state = _pause_state()
    overall = state['overall']
    if overall:
        print(f"Pause:  PAUSED -- {overall}")
        if state['manual'] and state['recording']:
            print(f"        and also: {state['recording']}")
        if state['manual']:
            print("        Resume with: mousereach-watch-toggle --resume")
        else:
            print(f"        {_recording_hint(overall)}")
    else:
        print("Pause:  not paused")
    if state['root_error']:
        print(f"        (could not check the hand pause: {state['root_error']})")
    if state['names']:
        print(f"        Pauses while any of these run: "
              f"{_ascii(', '.join(state['names']))} "
              f"(resumes {state['grace']} s after they close)")
        if not overall:
            # WHY: this command cannot see the running watcher's own timer, so
            # "not paused" here can still mean "waiting out the grace period".
            print(f"        (If one closed within the last {state['grace']} s, a "
                  f"running watcher is still waiting before it starts work.)")
        print("        Change the list with: mousereach-watch-recorders")
    print()

    from mousereach.config import Paths
    print("Stage folders:")
    for label, attr in _STAGE_FOLDERS:
        path = getattr(Paths, attr, None)
        print(f"  {label} (Paths.{attr}):")
        if not path:
            print("      (not configured -- nas_root is not set; run mousereach-setup)")
            continue
        try:
            found = Path(path).is_dir()
        except OSError:
            found = False
        print(f"      {_ascii(path)}{'' if found else '   [folder not found]'}")
    print()


def _print_toggle_status():
    """mousereach-watch-toggle --status: both pause reasons, then the verdict."""
    state = _pause_state()
    if state['root_error']:
        print(f"Hand pause:          cannot check -- {state['root_error']}")
    elif state['manual']:
        print("Hand pause:          ON (watcher_paused.flag)")
    else:
        print("Hand pause:          off")

    names = _ascii(', '.join(state['names']))
    if state['recording']:
        print(f"Recording programs:  {state['recording']}")
    elif not state['names']:
        print("Recording programs:  none configured -- the watcher never pauses for one")
    else:
        print(f"Recording programs:  none running (watching for: {names})")
        print(f"                     A running watcher that has just seen one close "
              f"waits {state['grace']} s before working again.")
    print()

    if state['overall']:
        print(f"Watcher is PAUSED -- {state['overall']}.")
        if state['manual']:
            print("Run 'mousereach-watch-toggle --resume' to remove the hand pause.")
        if state['recording']:
            print(_recording_hint(state['recording']))
    else:
        print("Watcher is ACTIVE (processing mode).")
        print("Run 'mousereach-watch-toggle --pause' to pause for filming.")


def main_toggle():
    """Pause or resume the watcher by hand, or show why it is paused.

    Usage:
        mousereach-watch-toggle            Flip: paused -> active, active -> paused
        mousereach-watch-toggle --pause    Pause by hand (stays paused until --resume)
        mousereach-watch-toggle --resume   Remove the hand pause
        mousereach-watch-toggle --status   Show both pause reasons; change nothing

    Two separate things pause the watcher:
      * the hand pause set here (the file watcher_paused.flag in this
        machine's processing folder, also set by the Pause button), and
      * a recording program listed with mousereach-watch-recorders being open.
    --resume removes only the hand pause. It cannot override a recording
    program: recording always wins, so close the recording program to let
    work start again.

    --pause and --resume do exactly what they say even when run twice, so a
    person or a script never flips the watcher back by accident (the plain
    toggle cannot promise that).
    """
    args = sys.argv[1:]

    # -h/--help must never ACT. The CLI reference generator runs every command
    # with --help; before this guard, that silently toggled the LIVE watcher
    # into filming mode (it paused production twice on 2026-08-30). Unknown
    # flags likewise refuse to toggle instead of toggling by accident.
    if '-h' in args or '--help' in args:
        print(main_toggle.__doc__)
        return
    unknown = [a for a in args if a not in ('--status', '--pause', '--resume')]
    if unknown:
        print(main_toggle.__doc__)
        print(f"Unknown argument(s): {' '.join(unknown)}", file=sys.stderr)
        sys.exit(2)
    chosen = sorted(set(args))
    if len(chosen) > 1:
        # Refuse rather than guess which one was meant.
        print(main_toggle.__doc__)
        print("Choose ONE of --pause, --resume or --status.", file=sys.stderr)
        sys.exit(2)
    action = chosen[0] if chosen else None

    try:
        pause_file = _get_pause_file()
    except Exception as e:
        print(f"ERROR: Could not determine processing root: {e}", file=sys.stderr)
        print("Run 'mousereach-setup' to configure paths.", file=sys.stderr)
        sys.exit(1)

    currently_paused = pause_file.exists()

    if action == '--status':
        _print_toggle_status()
        return

    if action is None:
        # The plain toggle: flip whatever the hand pause is now.
        action = '--resume' if currently_paused else '--pause'

    if action == '--resume':
        if currently_paused:
            pause_file.unlink(missing_ok=True)
            print("=" * 50)
            print("  Watcher RESUMED -- processing mode active")
            print("  DLC and cropping will run during downtime.")
            print("=" * 50)
        else:
            print("The watcher was not paused by hand -- nothing to remove.")
        # Removing the hand pause does not beat a recording program. Say so,
        # or the operator waits for work that cannot start.
        state = _pause_state()
        if state['recording']:
            print(f"[!] Still PAUSED: {state['recording']}.")
            print(f"    {_recording_hint(state['recording'])}")
        return

    if currently_paused:
        print("The watcher is already paused by hand -- nothing changed.")
        print("Run 'mousereach-watch-toggle --resume' when filming is done.")
        return
    pause_file.parent.mkdir(parents=True, exist_ok=True)
    pause_file.write_text("Watcher paused for filming.\n")
    print("=" * 50)
    print("  Watcher PAUSED -- filming mode active")
    print("  DLC processing is suspended.")
    print("  Toggle again (or run with --resume) when filming is done.")
    print("=" * 50)


# =============================================================================
# RECORDING PROGRAMS COMMAND
# =============================================================================

_RECORDERS_USAGE = """
usage: mousereach-watch-recorders [--list] [--add NAME] [--remove NAME] [--grace SECONDS]

Show or change the recording programs that pause THIS machine's watcher.
While any listed program is open the watcher starts no new work, and a pose
in progress is stopped and posed again later, because recording always wins:
a pose can be run again, a spoiled recording cannot. A collage crop that has
already started finishes first. A running watcher picks up changes made here
by itself within about a minute.

  --list            Show the list, the grace period, and what is running now.
                    This is what happens with no options at all.
  --add NAME        Add a program. NAME is the program's name as Task Manager
                    shows it on the Details tab, for example recorder.exe.
                    Capital letters, a full path or a missing .exe still match.
  --remove NAME     Take a program off the list.
  --grace SECONDS   How long every listed program must have been closed before
                    work starts again (default 120). WHY: people often close
                    the recording program and open it again a moment later.

--add and --remove may each be given more than once. The settings are saved in
the 'watcher' section of ~/.mousereach/config.json; nothing else in that file
changes. An empty list (the default) means the watcher never pauses for a
program.

Examples:
  mousereach-watch-recorders
  mousereach-watch-recorders --add recorder.exe
  mousereach-watch-recorders --remove recorder.exe
  mousereach-watch-recorders --grace 300
"""


class _ConfigEditError(Exception):
    """The config file cannot be edited safely; nothing was written."""


def _user_config_file() -> Path:
    """The per-user config file WatcherConfig.load() reads.

    Spelled exactly as mousereach.config._load_config spells it and resolved
    when called. mousereach.config has no public helper for this path, and
    mousereach.setup.wizard.CONFIG_FILE is fixed at the moment that module is
    imported. WHY it matters: this command must write the very file the
    watcher reads, or a setting "saved" here would never reach the watcher.
    """
    from mousereach.config import config_file_path
    return config_file_path()


def _read_config_for_edit(path: Path) -> dict:
    """The whole config file, for an in-place edit. A missing file is {}.

    Raises _ConfigEditError for a file that is not a JSON object. WHY:
    mousereach.config quietly treats an unreadable file as empty; an editor
    that did the same would rewrite it holding only the watcher section and
    erase processing_root, nas_root and every other setting on the machine.
    """
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise _ConfigEditError(f"cannot read {path}: {e}")
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise _ConfigEditError(f"{path} is not valid JSON ({e})")
    if not isinstance(data, dict):
        raise _ConfigEditError(f"{path} does not hold a JSON object")
    return data


def _write_config_atomically(path: Path, data: dict):
    """Write the config through a temporary file and os.replace.

    WHY: a watcher, the GUI and the setup wizard all read this file. Writing
    it in place leaves a moment where it is half written; a reader that hits
    that moment sees no processing_root at all. os.replace swaps the complete
    new file in at once. The retries cover Windows refusing the swap while
    another program has the file open for an instant.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii (the default) keeps the file pure ASCII, so it reads back
    # identically whatever text encoding the reader assumes.
    text = json.dumps(data, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _same_program(a: str, b: str) -> bool:
    """True when two configured names match the same running program, by the
    recording guard's own rule (capitals, a pasted path and a missing .exe do
    not matter). WHY reuse it: a --remove that compared names differently
    from the guard could report "removed" while another spelling of the same
    program kept pausing the watcher."""
    from mousereach.watcher.recording_guard import clean_names
    return len(clean_names([a, b])) == 1


def main_recorders():
    """Show or change which recording programs pause this machine's watcher.

    WHY this is configuration and not code: which program a lab records with,
    and how long to wait after it closes, are that lab's own facts. They live
    in the 'watcher' section of the user's config file (pause_while_running,
    pause_resume_grace_seconds), editable here or from the watcher panel,
    so no one has to edit code or JSON by hand.
    """
    _maybe_print_help(_RECORDERS_USAGE)
    args = sys.argv[1:]

    def usage_error(message):
        print(_RECORDERS_USAGE.strip(), file=sys.stderr)
        print(file=sys.stderr)
        print(f"ERROR: {_ascii(message)}", file=sys.stderr)
        print("Nothing was changed.", file=sys.stderr)
        sys.exit(2)

    from mousereach.config import WatcherConfig
    from mousereach.watcher import recording_guard as rg

    # Every argument is checked before anything is written, so a typo in the
    # last one cannot leave a half-applied change behind.
    adds, removes, grace = [], [], None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--list':
            i += 1
            continue
        if arg not in ('--add', '--remove', '--grace'):
            usage_error(f"unknown argument: {arg}")
        if i + 1 >= len(args) or args[i + 1].startswith('--'):
            usage_error(f"{arg} needs a value")
        value = args[i + 1].strip()
        i += 2
        if arg == '--grace':
            try:
                grace = int(value)
            except ValueError:
                grace = -1
            if grace < 0:
                usage_error(f"--grace needs a whole number of seconds, 0 or more "
                            f"(got '{value}')")
        elif not rg.clean_names([value]):
            usage_error(f"{arg} needs a program name, for example recorder.exe")
        else:
            (adds if arg == '--add' else removes).append(value)

    cfg_file = _user_config_file()
    try:
        data = _read_config_for_edit(cfg_file)
        watcher = data.get('watcher')
        if watcher is None:
            watcher = {}
        if not isinstance(watcher, dict):
            raise _ConfigEditError(f"the 'watcher' section of {cfg_file} is not an object")
        current = watcher.get('pause_while_running')
        if current is not None and not isinstance(current, (str, list, tuple)):
            raise _ConfigEditError(
                f"watcher.pause_while_running in {cfg_file} is not a list of names")
    except _ConfigEditError as e:
        print(f"ERROR: {_ascii(e)}", file=sys.stderr)
        print("Nothing was changed. Fix that file (or re-run mousereach-setup) "
              "and try again.", file=sys.stderr)
        sys.exit(1)

    names = rg.clean_names(current)
    messages = []
    names_changed = False
    for name in adds:
        if any(_same_program(name, n) for n in names):
            messages.append(f"{name} is already in the list -- not added again.")
        else:
            names.append(name)
            names_changed = True
            messages.append(f"Added {name}.")
    for name in removes:
        kept = [n for n in names if not _same_program(name, n)]
        if len(kept) == len(names):
            messages.append(f"{name} was not in the list -- nothing to remove.")
        else:
            names = kept
            names_changed = True
            messages.append(f"Removed {name}.")
    grace_changed = grace is not None and watcher.get('pause_resume_grace_seconds') != grace
    if grace is not None:
        messages.append(f"Resume grace set to {grace} s." if grace_changed
                        else f"Resume grace is already {grace} s.")

    saved = False
    if names_changed or grace_changed:
        # Change only these two keys; every other watcher setting, and every
        # other section of the file, is written back exactly as it was read.
        new_watcher = dict(watcher)
        if names_changed:
            if names:
                new_watcher['pause_while_running'] = names
            else:
                # Absent, not [], so the file reads as the default again.
                new_watcher.pop('pause_while_running', None)
        if grace_changed:
            new_watcher['pause_resume_grace_seconds'] = grace
        data['watcher'] = new_watcher
        try:
            _write_config_atomically(cfg_file, data)
        except OSError as e:
            print(f"ERROR: could not save {_ascii(cfg_file)}: {_ascii(e)}", file=sys.stderr)
            print("Nothing was changed.", file=sys.stderr)
            sys.exit(1)
        watcher = new_watcher
        saved = True

    for message in messages:
        print(_ascii(message))
    if saved:
        print(f"Saved to {_ascii(cfg_file)}")
    if messages:
        print()

    # Shown through WatcherConfig so the list and grace printed are exactly
    # what the watcher will load from this file.
    view = WatcherConfig(watcher)
    names = list(view.pause_while_running or [])
    grace_now = view.pause_resume_grace_seconds
    print(f"Config file: {_ascii(cfg_file)}")
    print()
    if not names:
        print("Recording programs that pause this watcher: none")
        print("  The watcher never pauses for a recording program on this machine.")
        print("  Add one with:  mousereach-watch-recorders --add recorder.exe")
        print(f"Resume grace: {grace_now} s")
    else:
        error = None
        try:
            running = set(rg.running_programs(names))
        except ImportError:
            running, error = set(), "psutil is not installed"
        except Exception as e:
            running, error = set(), f"{type(e).__name__}: {e}"
        # Compared by the guard's own matching rule, so two spellings of one
        # program in a hand-edited file both show as running.
        running_now = [n for n in names
                       if any(_same_program(n, r) for r in running)]
        width = max(len(_ascii(n)) for n in names)
        print("Recording programs that pause this watcher:")
        for n in names:
            status = ("cannot check" if error
                      else "RUNNING now" if n in running_now else "not running")
            print(f"  {_ascii(n):<{width}}   {status}")
        print(f"Resume grace: {grace_now} s -- after the last listed program closes, "
              f"the watcher waits this long before it starts work again.")
        print()
        if error:
            print(f"Right now: cannot check for recording programs ({_ascii(error)}).")
            print("  A running watcher stays PAUSED while it cannot check, because "
                  "recording must win.")
        elif running:
            shown = [n for n in names if n in running]
            verb = "is" if len(shown) == 1 else "are"
            print(f"Right now: {_ascii(', '.join(shown))} {verb} running, so a running "
                  f"watcher is PAUSED.")
            print("  Close the recording program when you are done recording.")
        else:
            print("Right now: no listed program is running, so none of them is "
                  "pausing the watcher.")

    if saved:
        print()
        # True since the watcher re-reads these two settings when the file
        # changes (BaseOrchestrator._refresh_recording_settings).
        print("A running watcher picks up this change by itself within about a minute;")
        print("no restart is needed. A pose it started before a program was first")
        print("listed runs to its end.")


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

    # Declaration drift first: if pipeline_versions.json disagrees with what
    # the installed code stamps, every compliance number below is built on a
    # lie -- and this exact silence has burned the lab twice.
    if not init:
        try:
            from mousereach.pipeline.versions import get_current_versions, declaration_drift
            _drift = declaration_drift(get_current_versions(Paths.NAS_ROOT))
            if _drift:
                print("=" * 60)
                print("[!] DECLARATION DRIFT -- pipeline_versions.json does not match the code:")
                for stage, decl, code in _drift:
                    print("    %-20s declared %-10s code stamps %s" % (stage, decl, code))
                print("    Staleness scanning is BLIND to these stages until the declaration")
                print("    is corrected (edit pipeline_versions.json or re-run --init).")
                print("=" * 60)
        except Exception as _e:
            print(f"[!] declaration-drift check unavailable: {_e}")

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
        db_path = _resolve_db_path()
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
    _maybe_print_help("""
usage: mousereach-crystallize (--cohort C | --videos "v1,v2") --label NAME
       mousereach-crystallize --list

Lock archived videos against automatic reprocessing (for publications).

  --cohort C    Every archived video of this cohort (e.g. CNT01).
  --videos LIST Comma-separated video ids.
  --label NAME  Required label naming the lock (e.g. "PNAS_2026").
  --list        Show what is crystallized, by label.
""")
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
        db_path = _resolve_db_path()
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
    _maybe_print_help("""
usage: mousereach-uncrystallize (--label NAME | --videos "v1,v2")

Unlock crystallized videos so reprocessing can touch them again.

  --label NAME   Unlock every video crystallized under this label.
  --videos LIST  Comma-separated video ids.
""")
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
        db_path = _resolve_db_path()
        db = WatcherDB(db_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    count = uncrystallize_videos(db=db, video_ids=video_ids, label=label)
    print(f"Uncrystallized {count} videos.")
    if count > 0:
        print("These videos can now be reprocessed when tool versions change.")


# =============================================================================
# UNRESOLVABLE: videos this node has no file for
# =============================================================================

# Placeholders written by older versions of cross-node recovery. They are not
# paths; four handlers passed them to Path() anyway.
_LEGACY_PLACEHOLDERS = ('recovered', 'recovered_from_mousedb', '')

# States that mean "this node holds the files and is working on it".
_WORKING_STATES = (
    'discovered', 'validated', 'dlc_queued', 'dlc_running',
    'dlc_complete', 'processing', 'processed', 'archiving', 'outdated',
)


def _pathless_working_rows(db):
    """Rows sitting in a working state with no usable file recorded.

    These are the phantoms: videos this node was told about by another node
    through connectome.db, which carries state but no file paths. The work loop
    picks them up every cycle and there is nothing there to work on.
    """
    marker = getattr(db, 'NO_FILE_HERE', '(no file on this node)')
    out = []
    for state in _WORKING_STATES:
        for row in db.get_videos_in_state(state):
            src = (row.get('source_path') or '').strip()
            cur = (row.get('current_path') or '').strip()
            if (src in _LEGACY_PLACEHOLDERS or src == marker) and not cur:
                out.append(row)
    return out


def main_unresolvable():
    """List, sweep, or retry videos this node has no file for."""
    _maybe_print_help("""
usage: mousereach-watch-unresolvable [--list | --sweep | --retry] [--dry-run]

Handle DB rows whose video file this node cannot find anywhere.

  --list      Show the pathless rows.
  --sweep     Mark them unresolvable so they leave the work loop.
  --retry     Put previously-swept rows back into the work loop.
  --dry-run   With --sweep/--retry: show what would change, change nothing.
""")
    args = sys.argv[1:]
    do_list = '--list' in args
    do_sweep = '--sweep' in args
    do_retry = '--retry' in args
    dry_run = '--dry-run' in args

    if not (do_list or do_sweep or do_retry):
        print("Usage: mousereach-watch-unresolvable [options]", file=sys.stderr)
        print()
        print("A video is 'unresolvable' when THIS machine has no file for it.")
        print("That is not a failure -- usually another node holds it -- so it is")
        print("kept out of the work loop instead of being retried forever.")
        print()
        print("Options:")
        print("  --list      Show unresolvable videos, and any pathless rows still")
        print("              sitting in a working state (the old phantom rows)")
        print("  --sweep     Move those pathless rows to 'unresolvable' so the")
        print("              watcher stops picking them up. Nothing is deleted.")
        print("  --retry     Put back into the pipeline any unresolvable video")
        print("              whose file has since appeared on this node")
        print("  --dry-run   Show what --sweep or --retry would do, and change nothing")
        print()
        print("Examples:")
        print("  mousereach-watch-unresolvable --list")
        print("  mousereach-watch-unresolvable --sweep --dry-run")
        print("  mousereach-watch-unresolvable --sweep")
        sys.exit(1)

    from mousereach.watcher.db import WatcherDB
    from mousereach.watcher.locate import locate_video_file

    try:
        db_path = _resolve_db_path()
        if not db_path.exists():
            print("ERROR: Watcher database not found. Run 'mousereach-watch' first.",
                  file=sys.stderr)
            sys.exit(1)
        db = WatcherDB(db_path)
    except Exception as e:
        print(f"ERROR: Failed to load database: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Database: {db_path}")
    print()

    unresolvable = db.get_videos_in_state('unresolvable')
    phantoms = _pathless_working_rows(db)

    if do_list:
        print("=" * 70)
        print("Videos with no file on this node")
        print("=" * 70)
        print()
        print(f"Already marked unresolvable: {len(unresolvable)}")
        for row in unresolvable[:40]:
            print(f"  {row['video_id']:32s} {row.get('error_message') or ''}")
        if len(unresolvable) > 40:
            print(f"  ... and {len(unresolvable) - 40} more")
        print()
        print(f"Pathless rows still in a working state: {len(phantoms)}")
        for row in phantoms[:40]:
            print(f"  {row['video_id']:32s} state={row['state']:14s} "
                  f"source_path={row.get('source_path')!r}")
        if len(phantoms) > 40:
            print(f"  ... and {len(phantoms) - 40} more")
        if phantoms:
            print()
            print("Run 'mousereach-watch-unresolvable --sweep' to take these out of")
            print("the work loop. They are kept, not deleted.")
        print()

    if do_sweep:
        moved = kept = 0
        for row in phantoms:
            video_id = row['video_id']
            found = locate_video_file(video_id)
            if found is not None:
                # Not a phantom after all: the file is here, the row just never
                # recorded where. Fill the path in and leave the state alone.
                print(f"  {video_id}: file IS here ({found}) -- recording the path, "
                      f"state stays '{row['state']}'")
                if not dry_run:
                    db.force_state(video_id, row['state'],
                                   reason="sweep: recording the file this node holds",
                                   source_path=str(found), current_path=str(found))
                kept += 1
                continue
            print(f"  {video_id}: {row['state']} -> unresolvable")
            if not dry_run:
                db.mark_unresolvable(
                    video_id,
                    "swept: sat in '%s' with no file on this node" % row['state'])
            moved += 1
        print()
        print(f"{'Would move' if dry_run else 'Moved'} {moved} row(s) to unresolvable; "
              f"{kept} turned out to have a file here.")

    if do_retry:
        back = 0
        for row in unresolvable:
            video_id = row['video_id']
            found = locate_video_file(video_id)
            if found is None:
                continue
            print(f"  {video_id}: file found at {found} -> dlc_queued")
            if not dry_run:
                db.force_state(video_id, 'dlc_queued',
                               reason="file has since appeared on this node",
                               source_path=str(found), current_path=str(found),
                               error_message=None)
            back += 1
        print()
        print(f"{'Would return' if dry_run else 'Returned'} {back} video(s) to the pipeline.")
