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
    writes); every other command hardcoded the fallback. That meant seven
    commands -- status, reprocess, quarantine, process-animal,
    version-check, crystallize, uncrystallize -- silently read a database last
    written months earlier while the daemon wrote to a different file.
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
        origin: Paths.NAS_ROOT_ORIGIN -- 'config' or 'unset'

    Returns:
        A problem string, or None if the root looks like a pipeline.
    """
    if not nas_root:
        return ("nas_root is not configured -- the shared pipeline folder (the one "
                "holding Unanalyzed, Processing, Analyzed and pipeline_versions.json). "
                "Run mousereach-setup, or set nas_root in ~/.mousereach/config.json")
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
    detail += (
        "\n      'nas_root' points here. Either repoint it at the pipeline root "
        "(mousereach-setup), or initialize one here with "
        "mousereach-version-check --init."
    )
    return detail


def _maybe_print_help(usage: str):
    """Print usage and exit 0 when -h/--help is asked for; refuse, exit 2,
    when an option is given that the usage text does not list.

    These entry points parse sys.argv by hand (deliberately -- no argparse
    rewrite of running production commands), which left them with no --help
    at all, and an option they did not know was silently ignored: a typo such
    as ``mousereach-watch --dryrun`` started the REAL watcher instead of the
    preview. The usage text is the list of options each command accepts, so
    anything starting with "-" that it does not name stops the command before
    it does anything (2026-10-22)."""
    args = sys.argv[1:]
    if "-h" in args or "--help" in args:
        print(usage.strip())
        sys.exit(0)
    allowed = set(re.findall(r"(?<![\w-])(--?[A-Za-z][\w-]*)", usage))
    # An option is a dash and a LETTER: "-5" is a value (e.g. --grace -5), and
    # the command's own check gives it a better message than "unknown option".
    unknown = [a for a in args
               if re.match(r"--?[A-Za-z]", a) and a.split("=", 1)[0] not in allowed]
    if unknown:
        print(usage.strip(), file=sys.stderr)
        print(file=sys.stderr)
        print("ERROR: unknown option(s): %s" % " ".join(unknown), file=sys.stderr)
        print("Nothing was changed.", file=sys.stderr)
        sys.exit(2)


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
    # It takes no words, only the options above. A word such as "stop" or
    # "start" was silently ignored: "mousereach-watch stop" STARTED a watcher
    # (or, with one already running, said so and did nothing) -- an install
    # script used it meaning to stop one.
    # Refuse, and say how stopping actually works.
    words = [a for a in args if not a.startswith("-")]
    if words:
        print("ERROR: mousereach-watch takes no words (got: %s). Nothing was changed.\n"
              "  To stop the watcher: create watcher_stop.flag in this machine's processing\n"
              "  folder (or use the Stop button); it stops after the current item.\n"
              "  To pause it: mousereach-watch-toggle --pause. To start it: mousereach-watch."
              % " ".join(words), file=sys.stderr)
        sys.exit(2)
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

    # The watcher must be able to NAME its database before anything starts:
    # a processing server whose settings have no watcher.db_path is refused
    # here (db_location says why), before logging, the one-watcher lock or
    # any file is created -- it never builds <processing folder>/watcher.db
    # by default for the other logins' tools to disagree about.
    from mousereach.watcher.db_location import WatcherDbUnnamed, resolve_watcher_db_path
    try:
        resolve_watcher_db_path()
    except WatcherDbUnnamed as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        pass        # an unconfigured processing root is reported below

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
        mode_label = "GPU node (DLC)"

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
        print("DRY RUN MODE - read-only: nothing is created, moved or written")
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
        # A GPU node needs: the shared pipeline root, DLC config, ffmpeg, GPU.
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

    # DRY RUN: a read-only preview, before the single-instance lock, the
    # database and the watcher are built -- building the watcher runs start-up
    # recovery, and the old dry run then called the real scans, so "dry run"
    # changed the local database and touched the shared one (watcher/preview.py).
    if dry_run:
        from mousereach.watcher.preview import preview
        try:
            db_path = _resolve_db_path()
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        for line in preview(config, db_path):
            print(line)
        return

    # SINGLE-INSTANCE GUARD. Two watchers on one state database is never a
    # valid configuration (interleaved work selection, double-claimed items),
    # and a day of deploy kill/restart cycles showed how easily
    # a second instance can go unnoticed: process filters based on command
    # lines miss instances whose command line is not visible. A named mutex
    # is per-machine, costs nothing, and the OS releases it on ANY exit,
    # force-kill included. The Watcher Control panel takes the same lock
    # (watcher/singleton.py), so a watcher started there counts too. Exit
    # code 0 on "already running": an eager relauncher gets an orderly no-op,
    # not a failure to retry.
    from mousereach.watcher import singleton
    try:
        _mutex = singleton.acquire_watcher_lock()
        if _mutex is None:
            logger.warning("Another watcher instance already holds the "
                           "singleton mutex on this machine -- exiting.")
            print("Watcher already running on this machine; exiting.")
            sys.exit(0)
        # Keep a reference so the handle lives as long as the process.
        globals()["_watcher_singleton_mutex"] = _mutex
    except singleton.LockUnavailable as e:
        logger.warning(f"Single-instance guard unavailable ({e}); continuing")

    # A STOP REQUEST LEFT BEHIND. The loop exits on its first turn when
    # watcher_stop.flag exists, so a forgotten flag used to make this start
    # end at once with a single INFO line -- and a scheduled task reported
    # success. Refuse instead, loudly, naming the file and its age, with a
    # non-zero exit code. The flag is NOT deleted: the update procedure relies
    # on it keeping the watcher stopped until the person updating removes it.
    # Checked after the single-instance lock, so a second start next to a
    # running watcher that is finishing its stop says "already running".
    try:
        _left = singleton.stop_flag_left(require_processing_root() / "watcher_stop.flag")
    except Exception:
        _left = None
    if _left:
        _msg = singleton.stop_flag_message(*_left)
        logger.warning(_msg)
        print(_msg, file=sys.stderr)
        sys.exit(singleton.EXIT_STOP_FLAG_LEFT)

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
        if once:
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
    # WHY a --help gate: without one, "--help" ran the status itself, so the
    # generated CLI reference recorded this machine's live state (its videos,
    # counts and the other machines' names) instead of the usage.
    _maybe_print_help("""
usage: mousereach-watch-status [--by-animal] [--json] [--log N]

Show this node's pipeline state: counts by state; the videos held after a
failed filing attempt, the failed videos with their errors, and what the
watchers list for a person (each list with its age); recent activity; and the
status every node last reported.

  --by-animal   Per-animal breakdown with QC.
  --json        JSON output.
  --log N       Show the last N log entries (default 10).
""")
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
        summary['held_and_failed'] = _held_and_failed(db)
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
    print(f"  Recording issue: {collage_states.get('recording_issue', 0)}")
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

    _print_held_and_failed(db)

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
                print(node_status.describe(Paths.NAS_ROOT,
                                          color=node_status.terminal_color(sys.stdout)))
                print()
        except Exception as e:
            print("Nodes: (could not be read: %s)" % e)
            print()


def _held_and_failed(db) -> dict:
    """What is held, failed or listed for a person, each part read on its own.

    WHY: the counts above say how many videos sit in each state, not which ones
    are stuck or why. A video whose filing failed stays 'processed' while the
    watcher waits out a back-off kept only in its memory; a failed video's
    reason is in its row; hand-score holds and other waits are in the
    watchers' attention lists on the share. None of that was on this screen.
    Each part is read in its own try so one unreadable source never hides the
    others (and never breaks the status command); a part that cannot be read
    says so.
    """
    out = {"archive_held": [], "failed": [], "attention": [], "recording_issues": [],
           "errors": []}
    try:
        out["archive_held"] = list(db.get_videos_held_by_failed_archive())
    except Exception as e:
        out["errors"].append("filing holds could not be read: %s" % e)
    try:
        out["failed"] = [{"video_id": v.get("video_id"),
                          "error_message": v.get("error_message")}
                         for v in db.get_videos_in_state("failed")]
    except Exception as e:
        out["errors"].append("failed videos could not be read: %s" % e)
    try:
        from mousereach.config import Paths
        from mousereach.watcher import attention
        nas = getattr(Paths, "NAS_ROOT", None)
        if nas:
            ages = {}
            for item in attention.read_items(nas):
                name = item.get("listed_in")
                if name not in ages:
                    try:
                        ages[name] = time.time() - (Path(nas) / "attention" / name).stat().st_mtime
                    except OSError:
                        ages[name] = None
                out["attention"].append(dict(item, list_age_seconds=ages[name]))
    except Exception as e:
        out["errors"].append("the attention lists could not be read: %s" % e)
    # Recordings set aside as recording issues (watcher/recording_issue.py),
    # from the reason files every machine shares.
    try:
        from mousereach.config import Paths
        from mousereach.watcher.recording_issue import attention_items
        nas = getattr(Paths, "NAS_ROOT", None)
        if nas:
            out["recording_issues"] = attention_items(nas)
    except Exception as e:
        out["errors"].append("the recording-issue reason files could not be read: %s" % e)
    return out


def _print_held_and_failed(db, limit: int = 20) -> None:
    """Print _held_and_failed for a person. ASCII only."""
    from mousereach.watcher.singleton import describe_age
    data = _held_and_failed(db)

    def _some(rows, fmt):
        for r in rows[:limit]:
            print("  " + fmt(r))
        if len(rows) > limit:
            print("  ... and %d more" % (len(rows) - limit))

    held = data["archive_held"]
    if held:
        print("Held after a failed filing attempt (%d) -- the watcher tries again "
              "by itself after a wait; if the error repeats, fix its cause:" % len(held))
        _some(held, lambda r: "%s: %s (failed %s UTC)"
              % (r.get("video_id"), r.get("message") or "no error recorded",
                 r.get("failed_at") or "?"))
        print()
    failed = data["failed"]
    if failed:
        print("Failed (%d) -- each waits for a person: fix the cause, then "
              "mousereach-watch-reprocess <video_id>:" % len(failed))
        _some(failed, lambda r: "%s: %s" % (r.get("video_id"),
                                            r.get("error_message") or "no error recorded"))
        print()
    items = data["attention"]
    if items:
        print("Listed by the watchers for a person (from <NAS>/attention):")

        def _item(r):
            age = r.get("list_age_seconds")
            # A list is rewritten whenever it changes, so an old list may be
            # from a watcher that has stopped: show its age, never imply it
            # is current.
            when = ("list written %s ago" % describe_age(age)) if age is not None else "list age unknown"
            tag = "NEEDS A PERSON" if r.get("needs_person") else "waiting"
            return "[%s] %s (%s, %s)" % (tag, r.get("says") or r.get("video_id") or "?",
                                        r.get("listed_in"), when)
        _some(items, _item)
        print()
    issues = data.get("recording_issues") or []
    if issues:
        print("Set aside as RECORDING ISSUES (%d) -- failed recordings, never cut, in "
              "Processing/Review/_Problematic; list them with mousereach-recording-issue "
              "--list:" % len(issues))
        _some(issues, lambda r: ("[NEEDS A PERSON] " if r.get("needs_person") else "")
              + str(r.get("says")))
        print()
    for err in data["errors"]:
        print("  [!] " + err)
    if data["errors"]:
        print()


# =============================================================================
# REPROCESS COMMAND
# =============================================================================

def main_reprocess():
    """Reset failed videos for reprocessing."""
    _maybe_print_help("""
usage: mousereach-watch-reprocess [video_id] [--all-failed] [--from-step STEP]
       mousereach-watch-reprocess --collage COLLAGE_FILE

Reset a failed video so the watcher works it again, with its automatic
retries given back. Only a video in the 'failed' state is changed; for any
other state the command says what to do instead and exits with code 1.

  video_id           The video to reset.
  --all-failed       Reset every video currently in the failed state.
  --from-step STEP   Restart at this step: dlc_queued, dlc_complete or
                     processing ('processed' is not offered: a failed
                     video earns it again by re-running). Without it the step is read
                     from the files this machine holds (video in its DLC
                     queue: dlc_queued, or dlc_complete with a pose beside it;
                     video and pose in its Processing folder: processing;
                     a single whose copy into the DLC queue failed and
                     that is back in the shared singles folder:
                     validated).
  --collage FILE     Give ONE collage that could not be cropped its
                     automatic retries back (its file name, as listed in
                     the dashboard's red panel). Only a 'failed' collage is
                     changed. The watcher tries it again at its next scan
                     while the file is still in the intake folder; no
                     restart, and no other collage or video is affected.
                     Run it on the machine the red panel names.
""")
    # Parse args
    args = sys.argv[1:]
    if '--collage' in args:
        i = args.index('--collage')
        name = args[i + 1] if i + 1 < len(args) else None
        sys.exit(_retry_parked_collage(name))
    all_failed = '--all-failed' in args
    from_step = None

    # Check for --from-step
    for i, arg in enumerate(args):
        if arg == '--from-step' and i + 1 < len(args):
            from_step = args[i + 1]

    # Get video_id if provided. The word after --from-step is the step, not
    # the video (it used to be taken as the video id, so the documented
    # "--from-step dlc_complete <video>" looked up a video named dlc_complete).
    video_id = None
    for i, arg in enumerate(args):
        if arg.startswith('--') or (i > 0 and args[i - 1] == '--from-step'):
            continue
        video_id = arg
        break

    if not video_id and not all_failed:
        print("Usage: mousereach-watch-reprocess [options] [video_id]", file=sys.stderr)
        print()
        print("Options:")
        print("  --all-failed      Reset ALL failed videos")
        print("  --from-step STEP  Restart at this step (dlc_queued, dlc_complete,")
        print("                    processing)")
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

        # Reset all. Count only what was really reset: a row can leave
        # 'failed' between the listing above and its reset (the watcher
        # retries failed videos on its own), and such a row is not changed.
        done = 0
        for video in failed:
            vid = video['video_id']
            try:
                new_state = db.reset_failed(vid, to_state=from_step)
            except Exception as e:
                print(f"ERROR resetting {vid}: {e}", file=sys.stderr)
                continue
            if new_state:
                done += 1
                print(f"Reset: {vid} -> {new_state} (retries given back)")
            else:
                print(f"  {vid} not changed: it is no longer failed")

        print(f"\nReset {done} of {len(failed)} videos.")

    else:
        # Reset single video. Only a 'failed' row is ever changed; for any
        # other state the command says so and what to do instead, and exits 1.
        # WHY no "Reset anyway?" question any more: the reset never changed
        # such a row, yet the command printed "Reset" -- a person was told a
        # video was reset that was not.
        try:
            video = db.get_video(video_id)
            if not video:
                print(f"ERROR: Video not found: {video_id}", file=sys.stderr)
                sys.exit(1)

            new_state = None
            if video['state'] == 'failed':
                new_state = db.reset_failed(video_id, to_state=from_step)
            if not new_state:
                state = (db.get_video(video_id) or video)['state']
                print(f"{video_id} not changed: it is '{state}', not failed.")
                print(f"  {_reprocess_next_step(state)}")
                sys.exit(1)

            print(f"Reset: {video_id} -> {new_state} (retries given back)")

        except SystemExit:
            raise
        except Exception as e:
            print(f"ERROR: Failed to reset video: {e}", file=sys.stderr)
            sys.exit(1)


def _retry_parked_collage(name) -> int:
    """mousereach-watch-reprocess --collage NAME: give one failed collage its
    retries back (WatcherDB.retry_parked_collage). Returns the exit code: 0
    when it was changed, 1 when not (and says why). WHY a per-collage retry:
    the only remedy used to be raising watcher.max_retries for the whole
    machine and restarting, which retried every failed video on it too."""
    if not name:
        print("Usage: mousereach-watch-reprocess --collage COLLAGE_FILE", file=sys.stderr)
        return 1
    try:
        from mousereach.watcher.db import WatcherDB
        db_path = _resolve_db_path()
        if not db_path.exists():
            print("ERROR: Watcher database not found. Run 'mousereach-watch' first.",
                  file=sys.stderr)
            return 1
        db = WatcherDB(db_path)
        row = db.get_collage(name)
        if not row:
            print(f"ERROR: no collage named {name} on this machine. Use the file name "
                  f"the dashboard's red panel shows, on the machine it names.",
                  file=sys.stderr)
            return 1
        old = db.retry_parked_collage(name)
    except Exception as e:
        print(f"ERROR: could not give the collage its retries back: {e}", file=sys.stderr)
        return 1
    if old is None:
        state = (db.get_collage(name) or row).get('state')
        print(f"{name} not changed: it is '{state}', not failed.")
        return 1
    print(f"Retry: {name} -- retries given back (it had used {old}). The watcher "
          f"tries it again at its next scan while its file is in the intake folder.")
    return 0


def _reprocess_next_step(state: str) -> str:
    """What a person does instead, for a video mousereach-watch-reprocess did
    not reset because it is not 'failed'."""
    if state == 'archived':
        return ("A filed video is run again with the 'Re-run video' button in the "
                "dashboard (it asks why).")
    if state == 'crystallized':
        return ("It is locked for a publication; unlock it with "
                "mousereach-uncrystallize first if a re-run is really intended.")
    if state in ('triage', 'deep_review'):
        return ("It is held for a person in the review queue; it goes on by itself "
                "once the review is finished.")
    if state == 'unresolvable':
        return ("This machine has no file for it; mousereach-watch-unresolvable "
                "--list says why.")
    if state == 'quarantined':
        return "It is quarantined; see mousereach-watch-quarantine --list."
    return "It is already being worked; mousereach-watch-status shows where it is."


# =============================================================================
# QUARANTINE COMMAND
# =============================================================================

def main_quarantine():
    """Manage quarantined files."""
    _maybe_print_help("""
usage: mousereach-watch-quarantine [--list]
                                   [--release FILE --to single|collage [--as NEW_NAME]]
                                   [--purge]

Manage files the watcher quarantined instead of processing.

  --list          Show what is quarantined and why.
  --release FILE  Move one file out of quarantine, back into an intake folder.
    --to single   ... into the single-animal intake (a one-mouse video), or
    --to collage  ... into the multi-animal intake (a whole-tray recording).
    --as NEW_NAME Release it under this corrected name (default: its own name).
                  The name is checked first; an invalid name moves nothing.
  --purge         Permanently delete everything quarantined (confirm first).
""")
    # Parse args
    args = sys.argv[1:]
    list_files = '--list' in args
    purge = '--purge' in args

    def _value(flag):
        # The word after the flag; None when the flag is absent or last.
        for i, arg in enumerate(args):
            if arg == flag and i + 1 < len(args):
                return args[i + 1]
        return None

    release_file = _value('--release')
    release_to = _value('--to')
    release_as = _value('--as')

    if not list_files and not release_file and not purge:
        print("Usage: mousereach-watch-quarantine [options]", file=sys.stderr)
        print()
        print("Options:")
        print("  --list                              Show quarantined files with reasons")
        print("  --release FILE --to single|collage  Release a file back into an intake folder")
        print("      [--as NEW_NAME]                 ... under a corrected name")
        print("  --purge                             Delete all quarantined files (with confirmation)")
        print()
        print("Examples:")
        print("  mousereach-watch-quarantine --list")
        print("  mousereach-watch-quarantine --release 20250101_badname.mp4 --to single --as 20250101_ABC0101_P1.mp4")
        print("  mousereach-watch-quarantine --purge")
        sys.exit(1)

    if (release_file or release_to or release_as) and (
            not release_file or release_to not in ('single', 'collage')):
        print("ERROR: --release needs a file name and --to single or --to collage "
              "(which intake folder it goes back to). Nothing was moved.",
              file=sys.stderr)
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

    # Release file from quarantine. The same routine as the Quarantine tab:
    # the name is checked with the destination's rules before anything moves,
    # nothing in the intake folder is overwritten, and the result is checked.
    # WHY: this option imported a class that does not exist, so it failed
    # every time, and it never asked where the file should go.
    if release_file:
        from mousereach.config import Paths
        from mousereach.watcher.validator import release_from_quarantine
        dest_dir = (Paths.SINGLE_ANIMAL_OUTPUT if release_to == 'single'
                    else Paths.MULTI_ANIMAL_SOURCE)
        if not dest_dir:
            print("ERROR: the intake folder is not configured (nas_root). "
                  "Run mousereach-setup. Nothing was moved.", file=sys.stderr)
            sys.exit(1)
        ok, msg = release_from_quarantine(
            release_file, quarantine_dir, dest_dir, release_to, new_name=release_as)
        if not ok:
            print(f"ERROR: {msg}", file=sys.stderr)
            sys.exit(1)
        print(msg)

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
    # Real argument parsing so --help prints this text instead of acting
    # (generating the CLI reference runs every command with --help).
    import argparse
    ap = argparse.ArgumentParser(description=main_prioritize.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("animal", nargs="?", help="animal id to put first, e.g. CNT0107")
    ap.add_argument("--clear", action="store_true", help="remove the priority animal")
    ns = ap.parse_args()
    clear = ns.clear
    positional = [ns.animal] if ns.animal else []

    # The file and its rules live in watcher/priority_animal.py, shared with
    # the Run Pipeline tab's "Do one animal's videos first" box.
    from mousereach.watcher import priority_animal as pa

    # --clear: remove priority
    if clear:
        if pa.clear_priority():
            print("Priority cleared. Watcher will resume normal ordering.")
        else:
            print("No priority was set.")
        return

    # No arguments: show current priority
    if not positional:
        data = pa.current_priority()
        if data:
            print(f"Priority animal: {data['animal_id']}")
            print(f"Set at:          {data.get('set_at', '?')}")
        elif pa.priority_file().exists():
            print("Error reading priority file: it could not be read")
        else:
            print("No priority animal set. Watcher uses default ordering.")
        print()
        print("Usage:")
        print("  mousereach-watch-prioritize CNT0107    Set priority")
        print("  mousereach-watch-prioritize --clear     Clear priority")
        return

    # Set priority
    try:
        animal_id = pa.set_priority_animal(positional[0])
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

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


# The row states mousereach-watch-process-animal queues a single from (each
# can go to 'dlc_queued'); a video with no row is queued too. See the loop
# in main_process_animal for why this is an allow-list.
_PROCESS_ANIMAL_QUEUEABLE = ('validated', 'failed', 'unresolvable')


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
        print("Singles are claimed for this machine and copied to DLC_Queue.")
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
    import socket
    from mousereach.watcher import single_claim
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
    # Named held_for_pose, not "claimed": the queueing loop below has its own
    # "claimed" (the file this run claims), and the shared name used to hide
    # this dict there.
    held_for_pose = {}
    if single_dir:
        try:
            from mousereach.census.runner import claimed_singles
            held_for_pose = {vid: host for vid, host in claimed_singles(single_dir).items()
                             if f"_{animal_id}_" in f"{vid}.mp4"}
        except Exception as e:
            print(f"[!] Could not read singles claimed for pose: {_ascii(e)}",
                  file=sys.stderr)

    single_keys = set()  # date + tray combos we already have as singles
    for name in [f.name for f in singles] + [f"{vid}.mp4" for vid in held_for_pose]:
        result = validate_single_filename(name)
        if result.valid:
            single_keys.add(f"{result.parsed['date']}_{result.parsed['tray_type']}")

    if held_for_pose:
        print(f"\nAlready being posed by another machine ({len(held_for_pose)}), "
              f"not queued again:")
        for vid in sorted(held_for_pose):
            print(f"  {vid}  (claimed by {_ascii(held_for_pose[vid])})")

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

    # The database must already exist, and is checked BEFORE the person is
    # asked to confirm. WHY: WatcherDB() creates a missing file, so a wrong
    # path (a db_path override not yet in place, a new machine) used to queue
    # every video into an empty decoy the watcher never reads -- and the
    # videos were copied into the DLC queue with no record of them. Only
    # mousereach-watch (and the Watcher Control panel) creates the database.
    try:
        db_path = _resolve_db_path()
    except Exception as e:
        # This login cannot name the watcher database (db_location says why);
        # queueing into a guessed one is how a decoy got written.
        print(f"ERROR: {e} Nothing was queued.", file=sys.stderr)
        sys.exit(1)
    if not db_path.exists():
        print(f"ERROR: Watcher database not found at {db_path}. Run 'mousereach-watch' "
              f"once on this machine first; nothing was queued.", file=sys.stderr)
        sys.exit(1)

    # Confirm
    print()
    response = input(f"Queue {total} videos for processing? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)

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
        if video_id in held_for_pose:
            # The same video is ALSO in a node's claim folder: this file is a
            # second copy dropped in while the first is out being posed.
            # Queueing it would pose the video twice.
            print(f"  SKIP {video_id} (a copy is being posed by "
                  f"{_ascii(held_for_pose[video_id])}; this one is a second copy)")
            skipped += 1
            continue
        existing = db.get_video(video_id)
        if existing:
            state = existing['state']
            # Only a video that is waiting to be taken ('validated'), that
            # failed, or that this machine had no file for ('unresolvable') is
            # queued; each can go to 'dlc_queued'. Every other state is
            # skipped BEFORE anything is claimed. WHY an allow-list: the old
            # list of states to skip missed triage, deep_review, outdated,
            # archiving and quarantined, and for those the move to
            # 'dlc_queued' raised after the claim -- the command stopped
            # mid-list and left the video in this machine's claim folder.
            # ('crystallized', locked for a publication, is never posed again:
            # db.crystallized_refusal.)
            if state not in _PROCESS_ANIMAL_QUEUEABLE:
                print(f"  SKIP {video_id} (already {state})")
                skipped += 1
                continue
        # A publication lock record beside the filed results (set from any
        # machine; pipeline/publication_lock.py) is never posed again either,
        # whatever this machine's row says -- or with no row at all.
        from mousereach.watcher.db import crystallized_refusal
        locked = crystallized_refusal(db, video_id, "queued for a new pose")
        if locked:
            print(f"  SKIP {locked}")
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

        # Claim, then copy to DLC_Queue. WHY the claim: the watcher's own
        # intake claims a shared single (renames it into
        # Single_Animal/.inflight/<this machine>/) before copying it, so no
        # other GPU node can take the same video. This command used to copy
        # without claiming, so two nodes could pose one video. The watcher on
        # this machine keeps the claim alive and removes the claimed copy once
        # the video has moved on, exactly as for its own claims.
        if dlc_queue:
            dest = dlc_queue / f.name
            claimed = single_claim.claim_single(f, socket.gethostname())
            if claimed is None:
                print(f"  SKIP {video_id} (could not claim it in the singles folder -- "
                      f"another machine took it meanwhile, or a program has it open)")
                skipped += 1
                continue
            # Everything after the claim is in one try: whatever fails (the
            # copy, or the state change), the claim goes back to the shared
            # folder and a queue copy THIS run made is removed, so no video is
            # left held by this machine with nothing queued to pose it.
            made_copy = False
            try:
                if dest.exists():
                    db.update_state(video_id, 'dlc_queued', current_path=str(dest))
                    queued_singles += 1
                    print(f"  QUEUED {video_id} (already in DLC_Queue)")
                elif safe_copy(claimed, dest, verify=True):
                    made_copy = True
                    db.update_state(video_id, 'dlc_queued', current_path=str(dest))
                    db.log_step(video_id, 'process_animal', 'queued',
                                message=f"Queued by process-animal for {animal_id}")
                    queued_singles += 1
                    print(f"  QUEUED {video_id}")
                else:
                    single_claim.release_single(claimed)
                    print(f"  ERROR {video_id} (copy failed; given back to the "
                          f"singles folder)")
                    errors += 1
            except Exception as e:
                single_claim.release_single(claimed)
                if made_copy:
                    try:
                        dest.unlink()
                    except OSError:
                        pass
                print(f"  ERROR {video_id}: {_ascii(e)} (given back to the singles folder)")
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
    # into filming mode (it paused production). Unknown
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
                                  [--hours HH:MM-HH:MM|any] [--days DAYS|any]

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
  --hours HH:MM-HH:MM
                    When recording can happen at all, on this PC's clock, for
                    example 08:00-18:00. Outside these hours an open recording
                    program does NOT pause the watcher. WHY: the program is
                    often left open after recording, and the PC then sat idle
                    until someone closed it. "any" removes the limit.
  --days DAYS       The days recording can happen, for example Mon-Fri or
                    Mon,Wed,Fri. On other days an open program does not pause
                    the watcher. "any" removes the limit.

--add and --remove may each be given more than once. The settings are saved in
the 'watcher' section of ~/.mousereach/config.json; nothing else in that file
changes. An empty list (the default) means the watcher never pauses for a
program.

Examples:
  mousereach-watch-recorders
  mousereach-watch-recorders --add recorder.exe
  mousereach-watch-recorders --remove recorder.exe
  mousereach-watch-recorders --grace 300
  mousereach-watch-recorders --days Mon-Fri --hours 08:00-18:00
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

    Raises _ConfigEditError for a file that is not a JSON object. WHY: an
    editor that read a damaged file as empty would rewrite it holding only
    the watcher section and erase processing_root, nas_root and every other
    setting on the machine. (mousereach.config refuses a damaged file the
    same way: read_settings_file.)
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
    hours = days = None          # None = not given; "" = clear (any)
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--list':
            i += 1
            continue
        if arg not in ('--add', '--remove', '--grace', '--hours', '--days'):
            usage_error(f"unknown argument: {arg}")
        if i + 1 >= len(args) or args[i + 1].startswith('--'):
            usage_error(f"{arg} needs a value")
        value = args[i + 1].strip()
        i += 2
        if arg in ('--hours', '--days'):
            clear = value.lower() in ('any', 'none', 'off')
            if arg == '--hours':
                if not clear and rg.parse_hours(value) is None:
                    usage_error(f"--hours needs a start and an end such as 08:00-18:00, "
                                f"or 'any' (got '{value}')")
                hours = "" if clear else value
            else:
                if not clear and rg.parse_days(value) is None:
                    usage_error(f"--days needs day names such as Mon-Fri or Mon,Wed,Fri, "
                                f"or 'any' (got '{value}')")
                days = "" if clear else value
            continue
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

    hours_changed = hours is not None and (watcher.get('recording_hours') or "") != hours
    days_changed = days is not None and (watcher.get('recording_days') or "") != days
    if hours is not None:
        messages.append(("Recording hours set to %s." % hours if hours
                         else "Recording hours removed: any hour.") if hours_changed
                        else "Recording hours are already %s." % (hours or "any hour"))
    if days is not None:
        messages.append(("Recording days set to %s." % days if days
                         else "Recording days removed: every day.") if days_changed
                        else "Recording days are already %s." % (days or "every day"))

    saved = False
    if names_changed or grace_changed or hours_changed or days_changed:
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
        for key, value, changed in (('recording_hours', hours, hours_changed),
                                    ('recording_days', days, days_changed)):
            if changed:
                if value:
                    new_watcher[key] = value
                else:
                    new_watcher.pop(key, None)   # absent = the default (any time)
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
        when_guard = rg.RecordingGuard(names, recording_hours=view.recording_hours,
                                       recording_days=view.recording_days)
        if when_guard.hours is None and when_guard.days is None:
            print("Recording time: any day, any hour -- an open listed program always "
                  "pauses the watcher.")
        else:
            print("Recording time: %s, %s -- outside it an open listed program does not "
                  "pause the watcher." % (
                      rg.format_days(when_guard.days) if when_guard.days else "every day",
                      rg.format_hours(when_guard.hours) if when_guard.hours else "any hour"))
            print("  Right now it is %s recording time." % (
                "OUTSIDE" if when_guard.outside_recording_time() else "inside"))
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
    # WHY a --help gate: without one, "--help" ran a full version scan of the
    # archive (minutes), so asking how to use the command started it.
    _maybe_print_help("""
usage: mousereach-version-check [--mark] [--init]

Compare each archived video's processing manifest with the declared current
versions (pipeline_versions.json) and report which videos are out of date.

  --mark   Also mark outdated videos for reprocessing.
  --init   Create pipeline_versions.json from the installed code's versions.
""")
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
        mousereach-crystallize --cohort CNT01 --label "PAPER_2026"
        mousereach-crystallize --videos "vid1,vid2" --label "prelim"
        mousereach-crystallize --list                 Show crystallized videos
    """
    _maybe_print_help("""
usage: mousereach-crystallize (--cohort C | --videos "v1,v2") --label NAME
       mousereach-crystallize --list

Lock archived videos against automatic reprocessing (for publications).
Each locked video gets <video>_crystallized.json beside its filed results;
every machine honors it. Exit code 1 lists any video that was NOT locked.

  --cohort C    Every archived video of this cohort (e.g. CNT01).
  --videos LIST Comma-separated video ids.
  --label NAME  Required label naming the lock (e.g. "PAPER_2026").
  --list        Show what is crystallized, by label, and lock records
                set from another machine.
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

    if not list_mode:
        # Locks are set where videos are filed (pipeline/versions.py
        # publication_lock_gate says why); --list only reads, anywhere.
        from mousereach.pipeline.versions import publication_lock_gate
        allowed, note = publication_lock_gate()
        if not allowed:
            print(f"ERROR: {note}", file=sys.stderr)
            sys.exit(1)
        if note:
            print(note)

    if not list_mode and not cohort and not video_ids:
        print("Usage: mousereach-crystallize [options]", file=sys.stderr)
        print()
        print("Options:")
        print("  --cohort COHORT   Crystallize all archived videos for cohort")
        print("  --videos V1,V2    Crystallize specific video IDs")
        print("  --label LABEL     Human-readable label (e.g. 'PAPER_2026_submission')")
        print("  --list            Show currently crystallized videos")
        print()
        print("Examples:")
        print("  mousereach-crystallize --cohort CNT01 --label 'PAPER_2026'")
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
        # The lock records beside the filed results are the locks every
        # machine honors; a record with no locked row here was set from
        # another machine (or for a video this machine has no row for).
        try:
            from mousereach.pipeline.publication_lock import iter_records, read_record
            rows_here = {v['video_id'] for v in crystallized}
            extra = [(vid, read_record(p).get('label') or '?')
                     for vid, p in iter_records() if vid not in rows_here]
        except Exception as e:
            extra = []
            print(f"(the lock records beside the filed results could not be listed: {e})")
        if extra:
            print(f"Lock records with no locked row on this machine ({len(extra)}):")
            for vid, lbl in sorted(extra):
                print(f"  {vid}  (label {lbl})")
            print()
        if not crystallized:
            print("No crystallized videos in this machine's database.")
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

    failures = []
    count = crystallize_videos(
        db=db,
        video_ids=video_ids,
        cohort=cohort,
        label=label,
        crystallized_by=username,
        failures=failures,
    )

    print(f"Crystallized {count} videos with label '{label}'.")
    if count > 0:
        print("Each has a lock record (<video>_crystallized.json) beside its filed")
        print("results, which every machine honors: these videos will NOT be")
        print("reprocessed, re-filed or moved into a review queue.")
        print(f"To unlock: mousereach-uncrystallize --label '{label}'")
    if failures:
        print(f"NOT locked ({len(failures)}):", file=sys.stderr)
        for vid, why in failures:
            print(f"  {vid}: {why}", file=sys.stderr)
        sys.exit(1)


def main_uncrystallize():
    """Unlock crystallized videos (allows reprocessing again).

    Usage:
        mousereach-uncrystallize --label "PAPER_2026"
        mousereach-uncrystallize --videos "vid1,vid2"
    """
    _maybe_print_help("""
usage: mousereach-uncrystallize (--label NAME | --videos "v1,v2")

Unlock crystallized videos so reprocessing can touch them again.

  --label NAME   Unlock every video crystallized under this label.
  --videos LIST  Comma-separated video ids.
""")
    # Locks are lifted where they were set: the processing server
    # (pipeline/versions.py publication_lock_gate says why).
    from mousereach.pipeline.versions import publication_lock_gate
    allowed, note = publication_lock_gate()
    if not allowed:
        print(f"ERROR: {note}", file=sys.stderr)
        sys.exit(1)
    if note:
        print(note)

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
        print("  mousereach-uncrystallize --label 'PAPER_2026'")
        sys.exit(1)

    from mousereach.config import require_processing_root
    from mousereach.watcher.db import WatcherDB
    from mousereach.pipeline.versions import uncrystallize_videos

    try:
        db_path = _resolve_db_path()
        # Never create the database here: an unlock written to a new empty
        # file unlocks nothing, and says "Uncrystallized 0 videos" as if the
        # label had no videos.
        if not db_path.exists():
            print(f"ERROR: Watcher database not found at {db_path}.", file=sys.stderr)
            sys.exit(1)
        db = WatcherDB(db_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    failures = []
    count = uncrystallize_videos(db=db, video_ids=video_ids, label=label,
                                 failures=failures)
    print(f"Uncrystallized {count} videos.")
    if count > 0:
        print("Their lock records are removed; these videos can now be reprocessed")
        print("when tool versions change.")
    if failures:
        print(f"STILL LOCKED ({len(failures)}):", file=sys.stderr)
        for vid, why in failures:
            print(f"  {vid}: {why}", file=sys.stderr)
        sys.exit(1)


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
    through the shared database, which carries state but no file paths. The work loop
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
  --sweep     Mark them unresolvable so they leave the work loop. A row
              whose file IS here gets its path filled in instead (on a
              processing server the posed folder counts, and for a
              re-run or filing row the filed folder too; on a GPU node
              only its own working folders).
  --retry     GPU node only: queue again an unresolvable video whose file
              is now in this node's DLC queue; one at the top of the shared
              singles folder is left for the intake to claim. Any other
              copy is listed and left alone. On a processing server it
              changes nothing (its watcher takes videos back by itself).
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
        print("  --retry     GPU node only: queue again any unresolvable video whose")
        print("              file is now in this node's DLC queue; one in the shared")
        print("              singles folder is left for the intake to claim")
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
        from mousereach.config import WatcherConfig
        from mousereach.watcher import single_claim
        import socket
        try:
            mode = WatcherConfig.load().mode
        except Exception as e:
            print(f"ERROR: could not read this machine's watcher settings: {_ascii(e)}",
                  file=sys.stderr)
            sys.exit(1)
        server = mode == 'processing_server'
        moved = kept = 0
        for row in phantoms:
            video_id = row['video_id']
            # Only a file this machine works the row from counts, and that
            # depends on the machine's role (read from its watcher settings,
            # as --retry does):
            #   * GPU node: never the filed folder, the posed folder (the
            #     processing server's intake), the top of the shared singles
            #     folder or another node's claim folder. WHY: the sweep
            #     records what it finds against a WORKING state, and a working
            #     state with a filed or posed path sent a GPU node's hand-off
            #     step to move those files (it used to search everywhere).
            #   * processing server: the posed folder IS where it works a
            #     posed video from, and a filed video marked for a re-run
            #     ('outdated', or one being filed, 'archiving') is worked
            #     from its filed folder -- so those are searched, as before;
            #     the filed folder only for those two states, so no other
            #     working state is pointed at a filed copy.
            found = locate_video_file(
                video_id,
                search_archive=server and row['state'] in ('outdated', 'archiving'),
                search_staging=server)
            if found is not None:
                holder = single_claim.claim_holder(found)
                if single_claim.in_front_door(found) or (
                        holder is not None and holder != socket.gethostname()):
                    found = None
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
        _retry_unresolvable(db, unresolvable, dry_run)


def _retry_unresolvable(db, unresolvable, dry_run: bool) -> int:
    """--retry: put back an 'unresolvable' video whose file this GPU node now
    holds. Returns how many were (or would be) put back.

    WHY so narrow: it used to search everywhere (the filed folder and the
    posed folder included) and force 'dlc_queued' on any machine. On the
    processing server no work step selects 'dlc_queued', so the row was
    stranded there; on a GPU node a filed or posed copy is refused for pose
    and parked again. Now:
      * processing server: nothing is changed -- its watcher takes a posed
        video back by itself (the posed-folder scan), so it says so;
      * GPU node: only a copy in THIS node's DLC queue is queued; a copy at the
        top of the shared singles folder is left for the singles intake,
        which claims it properly (its row gets the shared-folder reason the
        intake re-drives); any other copy is listed and left alone."""
    from mousereach.config import Paths, WatcherConfig
    from mousereach.watcher import single_claim
    from mousereach.watcher.locate import locate_video_file

    try:
        mode = WatcherConfig.load().mode
    except Exception as e:
        print(f"ERROR: could not read this machine's watcher settings: {_ascii(e)}",
              file=sys.stderr)
        sys.exit(1)
    if mode == 'processing_server':
        print("This is a processing server: --retry changes nothing here.")
        print("Its watcher takes a video back by itself once the video and its pose")
        print("are in the posed folder (Processing/Posed) again; nothing needs to be")
        print("forced. 'mousereach-watch-unresolvable --list' shows why each one waits.")
        return 0

    queue = Path(Paths.DLC_QUEUE) if Paths.DLC_QUEUE else None
    keep_reasons = (single_claim.DUPLICATE_OF_CLAIM_REASON,) + single_claim.REDRIVE_REASONS
    back = shared = 0
    for row in unresolvable:
        video_id = row['video_id']
        local = queue / f"{video_id}.mp4" if queue else None
        try:
            in_queue = local is not None and local.is_file()
        except OSError:
            in_queue = False
        if in_queue:
            print(f"  {video_id}: in this node's DLC queue ({local}) -> dlc_queued")
            if not dry_run:
                db.force_state(video_id, 'dlc_queued',
                               reason="a person's retry: the file is in this node's DLC queue",
                               source_path=str(local), current_path=str(local),
                               error_message=None)
            back += 1
            continue
        found = locate_video_file(video_id, search_archive=True, search_staging=True)
        if found is None:
            continue
        if single_claim.in_front_door(found):
            reason = row.get('error_message') or ''
            if reason.startswith(keep_reasons):
                print(f"  {video_id}: in the shared singles folder; already waiting "
                      f"for the intake ({reason[:60]}) -- left as it is")
                continue
            print(f"  {video_id}: in the shared singles folder ({found}) -> left for "
                  f"the intake to claim it")
            if not dry_run:
                db.mark_unresolvable(
                    video_id, f"{single_claim.SHARED_REASON}: put back for the intake "
                              f"by a person's retry ({found})")
            shared += 1
            continue
        print(f"  {video_id}: a copy is at {found}, which is not this node's (filed, "
              f"posed for the server, or another node's claim) -- left as it is")
    print()
    print(f"{'Would return' if dry_run else 'Returned'} {back} video(s) from this "
          f"node's DLC queue to the pipeline; {shared} left for the singles intake.")
    return back + shared
