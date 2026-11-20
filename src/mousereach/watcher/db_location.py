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
no side effects, and cli._resolve_db_path delegates to it. The watcher
itself (mousereach-watch and the Watcher Control panel's Start and Run Once)
uses it too, so the daemon and every tool always name the same file.

The rule, in order:
  1. the settings file cannot be read, or this login has none at all ->
     REFUSE (WatcherDbUnnamed), naming the file and the problem;
  2. the configured ``watcher.db_path`` when set -- the file the daemon writes;
  3. a machine configured as ``processing_server`` with no ``db_path`` ->
     REFUSE (WatcherDbUnnamed): the processing server's watcher database must
     be named in the settings of every login that works with it;
  4. otherwise ``<processing_root>/watcher.db`` -- the daemon's own default on
     a GPU PC (``dlc_pc``) and on a single machine (``standalone``, whose
     setup names it explicitly anyway; its posing half keeps a second file,
     ``watcher_posing.db``, that no tool reads -- watcher/standalone.py). An unconfigured processing root raises
     ConfigurationError (from require_processing_root), whose message names
     the setup command.

KNOWN LEFTOVER (rule 4). WatcherConfig defaults a missing ``watcher.mode`` to
``dlc_pc``, so a login whose settings file exists but names no mode lands
here, not in rule 3, and gets the default file without a refusal -- on the
processing server, the old decoy. Refusing a mode-less login would also stop
a GPU PC (its watcher included) whose settings never named a mode, so that
needs a decision first; until then the operator docs require every
processing-server login to carry ``watcher.mode`` = ``processing_server`` as
well as ``watcher.db_path``.

WHY REFUSE INSTEAD OF FALLING BACK (rules 1 and 3). A login with no settings
file at all gets its processing folder from the machine-wide environment
variable, and nothing about which database the watcher uses; the default file
under that folder is a guess, so it is refused like an unreadable file. Every Windows login on a
machine has its own settings file, but they share the processing folder. A
login whose settings cannot be read, or whose settings say "processing
server" without naming the database (a reviewer's login set up from a
profile, say), used to get ``<processing_root>/watcher.db`` -- on a server
whose watcher writes an overridden file, an old decoy. Commands run from that
login (crystallize, uncrystallize, process-animal, the dashboard's Reprocess
outdated) then read and wrote the decoy without a word: a publication lock
or an 'outdated' mark the watcher never sees. "Which database" cannot be
guessed, so the answer is a refusal that says what to fix.
"""
from pathlib import Path
from typing import Optional

from mousereach.settings_file import ConfigurationError


class WatcherDbUnnamed(ConfigurationError):
    """This login's settings do not say which watcher database the watcher on
    this machine uses (the settings file cannot be read or does not exist, or a processing
    server's settings name no ``watcher.db_path``). The message says what to
    fix. Distinct from "named, but the file is not there", which is a machine
    whose watcher never ran (the tools used on their own)."""


def _settings_file_name() -> str:
    try:
        from mousereach.config import config_file_path
        return str(config_file_path())
    except Exception:
        return "~/.mousereach/config.json"


def resolve_watcher_db_path() -> Path:
    """The watcher database this node's daemon writes to. Does not print,
    does not open or create the file. Raises WatcherDbUnnamed when this
    login's settings cannot say which file that is (module docstring)."""
    from mousereach.config import WatcherConfig, require_processing_root
    try:
        cfg = WatcherConfig.load()
    except Exception as e:
        raise WatcherDbUnnamed(
            f"the MouseReach settings of this login ({_settings_file_name()}) could "
            f"not be read ({e}), so which watcher database this machine uses cannot "
            f"be told. Fix the settings file (or run mousereach-setup); nothing was "
            f"read from or written to any watcher database.") from e
    if (getattr(cfg, "source_file", None) is not None
            and getattr(cfg, "source_mtime", 0) is None):
        # WatcherConfig.load() found no settings file for this login.
        raise WatcherDbUnnamed(
            f"this login has no MouseReach settings file ({_settings_file_name()}), so "
            f"which watcher database this machine uses cannot be told. Run "
            f"mousereach-setup in this login (on the processing server, set "
            f"watcher.db_path to the file the watcher writes); nothing was read from or "
            f"written to any watcher database.")
    if cfg.db_path:
        return Path(cfg.db_path)
    if (cfg.mode or "") == "processing_server":
        raise WatcherDbUnnamed(
            f"the MouseReach settings of this login ({_settings_file_name()}) say this "
            f"machine is the processing server but name no watcher database "
            f"(watcher.db_path). Every login on the processing server must name the "
            f"database its watcher writes: copy watcher.db_path from the settings of "
            f"the login that runs the watcher (mousereach-watch-status there prints it on "
            f"its first line, [watcher db]), "
            f"or set it with the Watcher Control panel's settings. On a new machine "
            f"where no watcher has run yet, set watcher.db_path to the file the watcher "
            f"should create (for example <processing folder>/watcher.db) in every login, "
            f"then start the watcher. Nothing was read "
            f"from or written to <processing folder>/watcher.db, which may be an old "
            f"copy the watcher does not use.")
    return require_processing_root() / "watcher.db"


def watcher_db_unnamed_reason() -> Optional[str]:
    """None when this login can name its watcher database (whether or not the
    file exists), else the reason it cannot. Never raises.

    For callers that treat "no watcher database" as "a machine that does not
    run the watcher" and carry on with the files alone: they must first ask
    this, because "cannot tell which database" is not "no database" -- the
    row that would refuse them may be in a database they cannot name.

    Only WatcherDbUnnamed counts. A machine with no processing folder
    configured at all (ConfigurationError from require_processing_root) runs
    no watcher, so it has no watcher database to name; callers that need the
    processing folder refuse on that themselves."""
    try:
        resolve_watcher_db_path()
    except WatcherDbUnnamed as e:
        return str(e)
    except Exception:
        return None
    return None


def open_watcher_db_if_present():
    """``(db, None)`` for this node's own watcher database, or ``(None, why)``.

    For callers outside the daemon that MOVE a video between review queues
    (the review tools' Escalate and Flag Session buttons): they record the
    move in the same database the daemon writes, so the folder the video is
    in and its database state agree. Never creates the file -- WatcherDB()
    would build an empty decoy -- and never raises: a caller that cannot open
    the database still has its own job to do, and says so to the person.
    A caller that would read "no database" as "no watcher on this machine"
    asks watcher_db_unnamed_reason() as well.
    """
    try:
        path = resolve_watcher_db_path()
    except Exception as e:
        return None, f"the watcher database could not be found ({e})"
    if not Path(path).is_file():
        return None, f"no watcher database at {path}"
    try:
        from mousereach.watcher.db import WatcherDB
        return WatcherDB(path), None
    except Exception as e:
        return None, f"the watcher database {path} could not be opened ({e})"
