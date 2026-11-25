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
  2b. no ``watcher.db_path`` in this login's settings, but the lab profile
     for THIS machine (matched by its hostname in lab_profiles.json) names
     one -> that file, with a log warning saying so (machine_watcher_db_path).
     A second person's login on the server (a reviewer set up later) has its
     own settings file and often no db_path; without this its review window
     could not record a single queue move. The login's own file is never
     edited;
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
profile, say), must not get ``<processing_root>/watcher.db`` -- on a server
whose watcher writes an overridden file, an old decoy. Commands run from that
login (crystallize, uncrystallize, process-animal, the dashboard's Reprocess
outdated) then read and wrote the decoy without a word: a publication lock
or an 'outdated' mark the watcher never sees. "Which database" cannot be
guessed, so the answer is a refusal that says what to fix -- unless the
lab profile for this very machine (matched by hostname) names it (rule 2b):
that is the lab's own record of this machine's watcher, not a guess.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

from mousereach.settings_file import ConfigurationError


# The machine's own watcher database, for a login whose settings do not name
# it (rule 2b in the module docstring). Said once per process, per file.
_MACHINE_DB_REPORTED = set()


def machine_watcher_db_path() -> Tuple[Optional[Path], Optional[str]]:
    """``(path, source)``: the watcher database the lab's profile for THIS
    machine names (``defaults.watcher.db_path`` of the profile whose
    ``match.hostname`` is this computer's name), or ``(None, None)``.

    WHY: every Windows login on a machine has its own settings file, but the
    machine has one watcher. A second person's login (a reviewer set up later)
    may have no ``watcher.db_path``, and its review window then could not
    record a single queue move in the watcher's database. The lab profiles
    file is where the lab writes down each machine's settings once; a profile
    matched by this computer's NAME is a statement about this machine, not a
    guess, so its ``db_path`` is the file the watcher here writes. A profile
    matched only by drive letters is not used: two PCs can share a layout.

    Looked for where mousereach-setup looks (MOUSEREACH_LAB_PROFILES, then
    ~/.mousereach/lab_profiles.json, then beside the setup module), and then
    as ``lab_profiles.json`` in the login's ``nas_root`` -- the shared folder
    every login has, so a login without the environment variable still finds
    the lab's file. Never raises; reads only."""
    import json
    import socket
    candidates = []
    try:
        from mousereach.setup.wizard import lab_profiles_candidates
        candidates.extend(lab_profiles_candidates())
    except Exception:
        pass
    try:
        from mousereach.config import _load_config
        nas = (_load_config() or {}).get("nas_root")
        if nas:
            candidates.append(Path(nas) / "lab_profiles.json")
    except Exception:
        pass
    try:
        host = socket.gethostname().lower()
    except Exception:
        return None, None
    for cand in candidates:
        try:
            if not Path(cand).is_file():
                continue
            with open(cand, "r", encoding="utf-8") as f:
                profiles = (json.load(f) or {}).get("profiles", [])
        except Exception:
            continue
        for prof in profiles:
            if not isinstance(prof, dict):
                continue
            wanted = (prof.get("match") or {}).get("hostname")
            names = [wanted] if isinstance(wanted, str) else list(wanted or [])
            if host not in {str(n).lower() for n in names}:
                continue
            db = ((prof.get("defaults") or {}).get("watcher") or {}).get("db_path")
            if db:
                return Path(db), (f"the lab profile '{prof.get('name', '?')}' for this "
                                  f"machine in {cand}")
        # The first profiles file that exists is the lab's; do not mix files.
        return None, None
    return None, None


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
    machine_db, source = machine_watcher_db_path()
    if machine_db is not None:
        key = str(machine_db)
        if key not in _MACHINE_DB_REPORTED:
            _MACHINE_DB_REPORTED.add(key)
            logging.getLogger(__name__).warning(
                "This login's MouseReach settings (%s) name no watcher.db_path; "
                "using this machine's watcher database %s, from %s. The login's "
                "settings file was not changed.", _settings_file_name(), machine_db, source)
        return machine_db
    if (cfg.mode or "") == "processing_server":
        raise WatcherDbUnnamed(
            f"the MouseReach settings of this login ({_settings_file_name()}) say this "
            f"machine is the processing server but name no watcher database "
            f"(watcher.db_path). Every login on the processing server must name the "
            f"database its watcher writes: copy watcher.db_path from the settings of "
            f"the login that runs the watcher (mousereach-watch-status there prints it on "
            f"its first line, [watcher db]), "
            f"or set it with the Watcher Control panel's settings, or name it once for "
            f"every login as watcher.db_path in this machine's profile in the lab "
            f"profiles file (lab_profiles.json, matched by hostname). On a new machine "
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
