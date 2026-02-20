"""
mousereach.watcher.backup - Y: to X: backup synchronization.

Monitors Y: pipeline directories for changes and syncs to X: (Drobo NAS)
via robocopy. Runs on the processing server where Y: is local storage.

Usage:
    mousereach-backup              Start backup watcher daemon
    mousereach-backup --once       Run one sync cycle and exit
    mousereach-backup --dry-run    Show what would be synced
"""

import sys
import time
import signal
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class BackupWatcher:
    """Watch Y: for changes, sync to X: via robocopy."""

    # Directories to sync (relative to source_root)
    DEFAULT_SYNC_DIRS = [
        "Behavior/MouseReach_Pipeline",
        "Tissue/MouseBrain_Pipeline",
        "Databases",
    ]

    def __init__(self, source_root, backup_root, poll_interval=600,
                 stability_seconds=300, sync_dirs=None):
        """
        Args:
            source_root: Root directory to sync from (e.g. Y:\2_Connectome)
            backup_root: Root directory to sync to (e.g. X:\2_Connectome_Backup)
            poll_interval: Seconds between sync checks (default 10 min)
            stability_seconds: Wait this long after last change before syncing (default 5 min)
            sync_dirs: List of subdirectories to sync (relative to source_root)
        """
        self.source_root = Path(source_root)
        self.backup_root = Path(backup_root)
        self.poll_interval = poll_interval
        self.stability_seconds = stability_seconds
        self.sync_dirs = sync_dirs or self.DEFAULT_SYNC_DIRS
        self._last_sync_at = None
        self._sync_count = 0

    def run(self, shutdown_event):
        """Main loop: periodically check for changes and sync."""
        logger.info(f"BackupWatcher started: {self.source_root} -> {self.backup_root}")
        logger.info(f"Poll interval: {self.poll_interval}s, stability wait: {self.stability_seconds}s")
        logger.info(f"Sync directories: {self.sync_dirs}")

        while not shutdown_event.is_set():
            try:
                self._sync_all()
            except Exception as e:
                logger.error(f"Sync error: {e}", exc_info=True)

            shutdown_event.wait(timeout=self.poll_interval)

        logger.info(f"BackupWatcher stopped after {self._sync_count} sync cycles")

    def run_once(self):
        """Run one sync cycle and exit."""
        self._sync_all()

    def dry_run(self):
        """Show what would be synced without making changes."""
        print("=" * 70)
        print("BackupWatcher Dry Run")
        print("=" * 70)
        print()
        print(f"Source:      {self.source_root}")
        print(f"Destination: {self.backup_root}")
        print()

        for subdir in self.sync_dirs:
            src = self.source_root / subdir
            dst = self.backup_root / subdir
            if src.exists():
                # Count files
                file_count = sum(1 for _ in src.rglob('*') if _.is_file())
                print(f"  {subdir}")
                print(f"    Source:  {src} ({file_count} files)")
                print(f"    Backup:  {dst}")
                if dst.exists():
                    backup_count = sum(1 for _ in dst.rglob('*') if _.is_file())
                    print(f"    Backup files: {backup_count}")
                else:
                    print(f"    Backup:  (does not exist yet)")
            else:
                print(f"  {subdir} -- source does not exist, skipping")
            print()

    def _sync_all(self):
        """Run robocopy for each sync directory."""
        total_files = 0
        total_errors = 0

        for subdir in self.sync_dirs:
            src = self.source_root / subdir
            dst = self.backup_root / subdir

            if not src.exists():
                logger.debug(f"Source does not exist, skipping: {src}")
                continue

            files_copied, errors = self._sync_dir(src, dst)
            total_files += files_copied
            total_errors += errors

        self._last_sync_at = datetime.now()
        self._sync_count += 1

        if total_files > 0:
            logger.info(f"Sync cycle {self._sync_count}: {total_files} files copied, {total_errors} errors")
        else:
            logger.debug(f"Sync cycle {self._sync_count}: no new/changed files")

    def _sync_dir(self, src, dst):
        """Sync a single directory using robocopy.

        Returns (files_copied, errors).
        """
        try:
            result = subprocess.run([
                "robocopy",
                str(src), str(dst),
                "/E",           # Copy subdirectories including empty ones
                "/XO",          # Exclude older files (only copy newer)
                "/MT:4",        # Multi-threaded (4 threads)
                "/R:1",         # Retry once on failure
                "/W:5",         # Wait 5 seconds between retries
                "/NP",          # No progress percentage
                "/NDL",         # No directory listing
                "/NFL",         # No file listing (quiet mode)
                "/NJH",         # No job header
                "/NJS",         # No job summary
                "/XD", "__pycache__", ".git", ".claims",  # Exclude dirs
                "/XF", "*.pyc", "watcher.db-journal",     # Exclude files
            ], capture_output=True, text=True, timeout=3600)

            # Robocopy exit codes: 0=no changes, 1=files copied, 2=extra files,
            # 4=mismatches, 8=failures, 16=fatal error
            # Codes 0-7 are success; 8+ are errors
            if result.returncode >= 8:
                logger.warning(f"Robocopy error for {src}: exit code {result.returncode}")
                if result.stderr:
                    logger.warning(f"  stderr: {result.stderr.strip()}")
                return 0, 1

            # Parse output for file count (rough estimate from return code)
            files_copied = 1 if result.returncode & 1 else 0
            return files_copied, 0

        except subprocess.TimeoutExpired:
            logger.error(f"Robocopy timed out for {src}")
            return 0, 1
        except FileNotFoundError:
            logger.error("robocopy not found on PATH")
            return 0, 1
        except Exception as e:
            logger.error(f"Robocopy failed for {src}: {e}")
            return 0, 1


def main():
    """CLI entry point for mousereach-backup."""
    args = sys.argv[1:]
    once = '--once' in args
    dry_run_mode = '--dry-run' in args
    verbose = '--verbose' in args

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)-8s %(name)s - %(message)s'
    )

    # Load config
    try:
        from mousereach.config import _load_config
        config = _load_config()
        backup_config = config.get('backup', {})
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    if not backup_config.get('enabled', False):
        print("Backup watcher is not enabled in config.")
        print("Add to ~/.mousereach/config.json:")
        print('  "backup": {')
        print('    "enabled": true,')
        print('    "source_root": "Y:\\\\2_Connectome",')
        print('    "backup_root": "X:\\\\2_Connectome_Backup"')
        print('  }')
        sys.exit(1)

    source_root = backup_config.get('source_root')
    backup_root = backup_config.get('backup_root')

    if not source_root or not backup_root:
        print("ERROR: backup.source_root and backup.backup_root must be set in config", file=sys.stderr)
        sys.exit(1)

    watcher = BackupWatcher(
        source_root=source_root,
        backup_root=backup_root,
        poll_interval=backup_config.get('poll_interval_seconds', 600),
        stability_seconds=backup_config.get('stability_seconds', 300),
    )

    if dry_run_mode:
        watcher.dry_run()
        return

    if once:
        print("Running one sync cycle...")
        watcher.run_once()
        print("Done.")
        return

    # Daemon mode
    print("=" * 70)
    print("MouseReach Backup Watcher")
    print("=" * 70)
    print(f"  Source:      {source_root}")
    print(f"  Destination: {backup_root}")
    print(f"  Poll:        {backup_config.get('poll_interval_seconds', 600)}s")
    print()
    print("Press Ctrl+C to stop.")
    print()

    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        print("\nShutdown requested...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    watcher.run(shutdown_event)
    print("Backup watcher stopped.")
