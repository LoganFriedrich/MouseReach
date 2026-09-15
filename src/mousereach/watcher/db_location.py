"""Where the watcher database lives -- the ONE resolver every command shares.

WHY THIS IS ITS OWN MODULE
--------------------------
A node may point the watcher at a database other than the default through
``db_path`` in the ``watcher`` section of ~/.mousereach/config.json (set
because SQLite over a network share loses writes). The daemon honours that
override, so the override IS the live database; the default
``<processing_root>/watcher.db`` beside it is then a decoy that may still
exist, with the full schema and stale or zero rows.

Anything that opens ``WatcherDB()`` bare gets the default, not the override.
mousereach-route-to-queue did exactly that: every video it moved into a review
queue had its state written to the unused decoy, so the live database never
learned of the move and the log said "Disk and DB now disagree" for every
video. The correct resolver already existed, but inside watcher/cli.py, where
it also prints -- so other modules did not reuse it. It lives here now, with
no side effects, and cli._resolve_db_path delegates to it.

The rule, in order:
  1. the configured ``watcher.db_path`` when set -- the file the daemon writes;
  2. otherwise ``<processing_root>/watcher.db`` -- the daemon's own default.

An unreadable config counts as "no override", as it always has for the CLI
commands. An unconfigured processing root raises ConfigurationError (from
require_processing_root), whose message names the setup command.
"""
from pathlib import Path


def resolve_watcher_db_path() -> Path:
    """The watcher database this node's daemon writes to. Does not print,
    does not open or create the file."""
    from mousereach.config import WatcherConfig, require_processing_root
    try:
        override = WatcherConfig.load().db_path
    except Exception:
        override = None
    if override:
        return Path(override)
    return require_processing_root() / "watcher.db"
