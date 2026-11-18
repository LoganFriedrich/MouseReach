"""The test guard: no test may write into this machine's REAL pipeline folders.

Loaded by tests/conftest.py when pytest starts. Plain-language description for
people who run the tests: docs/TEST_GUARD.md.

WHAT IT REFUSES. From the moment it is loaded until the run ends, any attempt
to create, write, rename, move, delete, touch or open a database under a real
root fails with PermissionError("test guard: ... refused: <path>") BEFORE
anything happens on disk. That covers open() in a write mode, os.open with
create/write flags (and so tempfile.mkstemp, Path.touch, write_text), os.mkdir
(and so os.makedirs, tempfile.mkdtemp), rename/replace (either end), remove,
rmdir, utime, chmod, truncate, link, shutil copies (the destination), moves,
rmtree and sqlite3.connect. WHY refuse and not only report: a test that wrote
one row into the real shared version index failed afterwards, as designed, but
the row stayed in the live index; a failing test must leave nothing behind.
The attempt is also recorded, so a test whose code swallows the error still
fails, naming the path. A database opened read-only ("file:...?mode=ro" with
uri=True) only reads and is allowed.

WHAT IT CANNOT SEE. The guard is a Python audit hook: it sees what Python code
in this process asks for. It does NOT see writes made inside compiled
libraries that open files themselves (a video writer, HDF5, a GPU library) or
by other programs a test starts (ffmpeg, a second Python). Tests that do such
things must still point them at tmp_path; the guard cannot catch them.

WHERE THE REAL ROOTS COME FROM. They are collected once, when this module is
imported, from every place MouseReach itself would look on a real machine --
NOT only from the place the current process would look, because a test run is
normally isolated (home folder pointed at a scratch folder, MouseReach
variables unset), and an isolated run must still be protected if a test finds
its way back to the real folders:
  1. the settings file ~/.mousereach/config.json of the CURRENT home folder,
     and of the account's REAL home folder as the operating system records it
     (which a changed USERPROFILE/HOME does not move);
  2. the shared lab settings file <nas_root>/mousereach_lab_config.json named
     by either of those;
  3. every environment variable whose name starts with MOUSEREACH_, MOUSEDB_
     or MOUSEBRAIN_, or is CONNECTOME_ROOT, in ANY letter case (a machine may
     set MouseReach_PROCESSING_ROOT, which the tool reads);
  4. the same variables as Windows keeps them for the account and the
     machine (the registry), which an isolated run does not remove;
  5. what the tool's own config module resolved at import (Paths.NAS_ROOT,
     PROCESSING_ROOT, NAS_DRIVE);
  6. a list the person running the tests supplies: REACH_TEST_GUARD_ROOTS
     (folders separated by the system's path separator, ';' on Windows) and/or
     REACH_TEST_GUARD_ROOTS_FILE (a text file, one folder per line, '#'
     starts a comment).
From a settings file every absolute path value is taken: a folder as it is, a
file (a name with an extension) as its folder.

EVERY SPELLING. A path is matched whatever its spelling: letter case, forward
or back slashes, relative to the current folder, the \\\\?\\ long-path prefix,
8.3 short names (each folder level long or short), and for a mapped network
drive both the drive letter and the network (UNC) form. All spellings are
built here at import; the audit hook itself only compares strings, so it costs
nothing for the reads every import makes.

WHAT IS NOT GUARDED. The repository's own files (source, tests, docs,
__pycache__), even when the repository sits inside a real root -- its siblings
stay guarded; the system temporary folder (where pytest's tmp_path lives); and
the Python installation running the tests. An exempt folder wins only when it
lies INSIDE the real root that matched (a real root inside the temp folder is
still guarded).
"""
import json
import os
import string
import sys
from pathlib import Path

# Variables a person running the tests may set to add real roots. Not prefixed
# MOUSEREACH_: an isolated run unsets those, and its proof that nothing names
# the pipeline must stay empty.
ROOTS_ENV = "REACH_TEST_GUARD_ROOTS"
ROOTS_FILE_ENV = "REACH_TEST_GUARD_ROOTS_FILE"

# Environment variables that can name a real root, matched case-insensitively.
_ENV_PREFIXES = ("MOUSEREACH_", "MOUSEDB_", "MOUSEBRAIN_")
_ENV_NAMES = ("CONNECTOME_ROOT",)

# Audit events that change the disk, and which of their arguments is the place
# being changed. A copy's SOURCE is only read, so only its destination counts.
_ARG_POSITIONS = {
    "os.mkdir": (0,), "os.remove": (0,), "os.rmdir": (0,), "os.utime": (0,),
    "os.chmod": (0,), "os.truncate": (0,), "os.chown": (0,),
    "os.rename": (0, 1), "os.replace": (0, 1),
    "os.link": (1,), "os.symlink": (1,),
    "shutil.copyfile": (1,), "shutil.copymode": (1,), "shutil.copystat": (1,),
    "shutil.copytree": (1,), "shutil.move": (0, 1), "shutil.rmtree": (0,),
    "shutil.make_archive": (0,),
    "tempfile.mkstemp": (0,), "tempfile.mkdtemp": (0,),
}
# os.open flags that mean "this open can change the file". The audit "open"
# event of os.open carries mode None and these int flags; builtin open carries
# its mode string.
_WRITE_FLAGS = (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
                | getattr(os, "O_TEMPORARY", 0))

_WINDOWS = os.name == "nt"


# ---------------------------------------------------------------------------
# Spelling: pure string work (used by the hook)
# ---------------------------------------------------------------------------

def norm(path):
    """One comparable spelling of a path: absolute, case-folded on Windows,
    back slashes, no \\\\?\\ prefix, no trailing separator. None when it is not
    a path. Only string work: os.path.abspath joins the current folder without
    touching the disk."""
    try:
        p = os.fsdecode(os.fspath(path))
    except (TypeError, ValueError):
        return None
    if not p or "\x00" in p:
        return None
    if _WINDOWS:
        p = p.replace("/", "\\")
        low = p.lower()
        if low.startswith("\\\\?\\unc\\"):
            p = "\\\\" + p[8:]
        elif low.startswith("\\\\?\\") or low.startswith("\\\\.\\"):
            p = p[4:]
    try:
        p = os.path.normcase(os.path.abspath(p))
    except (TypeError, ValueError, OSError):
        return None
    if len(p) > 1 and p.endswith(os.sep) and not p.endswith(":" + os.sep):
        p = p.rstrip(os.sep)
    return p


def _under(n, prefix):
    """True when spelling n is prefix or lies inside it."""
    if n == prefix:
        return True
    head = prefix if prefix.endswith(os.sep) else prefix + os.sep
    return n.startswith(head)


def _sqlite_write_target(database):
    """The file a sqlite3.connect may write, or None when it cannot write one.
    A "file:" address (uri=True) is reduced to its file; one opened read-only
    (mode=ro, or immutable=1) only reads, so it is allowed like any read."""
    if isinstance(database, bytes):
        try:
            database = os.fsdecode(database)
        except (TypeError, ValueError):
            return None
    if not isinstance(database, (str, os.PathLike)):
        return None
    text = os.fspath(database)
    if text == ":memory:" or not text:
        return None
    if text.startswith("file:"):
        body, _, query = text[5:].partition("?")
        params = dict(p.partition("=")[::2] for p in query.split("&") if p)
        if params.get("mode") == "ro" or params.get("immutable") == "1":
            return None
        if params.get("mode") == "memory" or body in ("", ":memory:"):
            return None
        from urllib.parse import unquote
        body = unquote(body)
        if body.startswith("///"):
            body = body[3:]          # file:///C:/x -> C:/x
        elif body.startswith("//"):
            body = body[2:]          # file://localhost/... is not used here
            body = body.partition("/")[2] if not body.startswith("/") else body
        if _WINDOWS and len(body) > 2 and body[0] == "/" and body[2] == ":":
            body = body[1:]          # file:/C:/x -> C:/x
        return body
    return text


# ---------------------------------------------------------------------------
# Building every spelling of a root (done once, at import; may touch the disk)
# ---------------------------------------------------------------------------

def _mapped_drives():
    """{'x:': r'\\\\server\\share'} for every mapped network drive (Windows)."""
    if not _WINDOWS:
        return {}
    try:
        import ctypes
        from ctypes import wintypes
        get = ctypes.windll.mpr.WNetGetConnectionW
    except Exception:
        return {}
    found = {}
    for letter in string.ascii_uppercase:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        try:
            if get(letter + ":", buf, ctypes.byref(size)) == 0 and buf.value:
                found[letter.lower() + ":"] = buf.value
        except Exception:
            continue
    return found


def _short_long(path):
    """(short spelling, long spelling) of an existing path on Windows."""
    out = []
    if not _WINDOWS:
        return out
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        for fn in (k32.GetShortPathNameW, k32.GetLongPathNameW):
            buf = ctypes.create_unicode_buffer(32768)
            if fn(str(path), buf, 32768):
                out.append(buf.value)
    except Exception:
        pass
    return out


def _component_variants(path, limit=256):
    """Every mix of long and 8.3 short names along an existing path, so a
    folder spelled half short and half long still matches. Bounded by limit."""
    p = norm(path)
    if not p or not _WINDOWS:
        return {p} if p else set()
    drive, rest = os.path.splitdrive(p)
    parts = [x for x in rest.split(os.sep) if x]
    variants = {drive + os.sep}
    current = drive + os.sep
    for part in parts:
        current = os.path.join(current, part)
        names = {part}
        if os.path.exists(current):
            for spelled in _short_long(current):
                name = os.path.basename(spelled.rstrip("\\/"))
                if name:
                    names.add(os.path.normcase(name))
        variants = {os.path.join(v, n) for v in variants for n in names}
        if len(variants) > limit:
            variants = set(list(variants)[:limit])
    return {norm(v) for v in variants if norm(v)}


def spellings(path, mapped=None):
    """Every spelling of path: as given, real path, short/long names, and both
    the drive-letter and network form of a mapped drive. mapped is
    {'x:': r'\\\\server\\share'} (default: this machine's mapped drives)."""
    if mapped is None:
        mapped = _MAPPED
    base = norm(path)
    if not base:
        return set()
    found = {base}
    try:
        real = norm(os.path.realpath(base))
        if real:
            found.add(real)
    except (OSError, ValueError):
        pass
    for p in list(found):
        found |= _component_variants(p)
    # Mapped drives: drive letter <-> network form, in both directions.
    unc = {d: norm(u) for d, u in mapped.items() if norm(u)}
    for p in list(found):
        drive = os.path.splitdrive(p)[0].lower()
        if drive in unc:
            found.add(norm(unc[drive] + p[len(drive):]))
        for d, share in unc.items():
            if _under(p, share):
                found.add(norm(d + os.sep + p[len(share):].lstrip(os.sep)))
    return {f for f in found if f}


# ---------------------------------------------------------------------------
# Collecting the real roots
# ---------------------------------------------------------------------------

def os_home_folders():
    """The account's home folder as the operating system records it, which a
    changed USERPROFILE/HOME does not move, plus the current one."""
    homes = []
    try:
        homes.append(str(Path.home()))
    except (RuntimeError, KeyError, OSError):
        pass
    if _WINDOWS:
        try:
            import ctypes
            import uuid
            from ctypes import wintypes

            class _GUID(ctypes.Structure):
                _fields_ = [("a", wintypes.DWORD), ("b", wintypes.WORD),
                            ("c", wintypes.WORD), ("d", ctypes.c_ubyte * 8)]
            u = uuid.UUID("5E6C858F-0E22-4760-9AFE-EA3317B67173")  # FOLDERID_Profile
            g = _GUID(u.fields[0], u.fields[1], u.fields[2],
                      (ctypes.c_ubyte * 8)(*u.bytes[8:]))
            out = ctypes.c_wchar_p()
            if ctypes.windll.shell32.SHGetKnownFolderPath(
                    ctypes.byref(g), 0x8000, None, ctypes.byref(out)) == 0:
                homes.append(out.value)
                ctypes.windll.ole32.CoTaskMemFree(out)
        except Exception:
            pass
    else:
        try:
            import pwd
            homes.append(pwd.getpwuid(os.getuid()).pw_dir)
        except Exception:
            pass
    unique = []
    for h in homes:
        if h and norm(h) not in [norm(x) for x in unique]:
            unique.append(h)
    return unique


def _looks_absolute(value):
    if not isinstance(value, str) or not value.strip():
        return False
    v = value.strip()
    if _WINDOWS:
        return (len(v) >= 3 and v[1] == ":" and v[2] in "\\/") or v.startswith(("\\\\", "//"))
    return v.startswith("/")


def _as_root(value):
    """A folder for a configured value: a file (has an extension) gives its
    folder, a folder itself."""
    v = value.strip()
    return str(Path(v).parent) if Path(v).suffix else v


def paths_in_settings(data):
    """Every absolute path value anywhere in a settings dict, as a folder."""
    found = []
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif _looks_absolute(item):
            found.append(_as_root(item))
    return found


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def roots_from_settings_files(homes):
    """Real roots named by ~/.mousereach/config.json in each home folder, and
    by the shared lab settings file each one's nas_root points at."""
    roots = []
    for home in homes:
        cfg = _read_json(Path(home) / ".mousereach" / "config.json")
        roots += paths_in_settings(cfg)
        nas_root = cfg.get("nas_root")
        if isinstance(nas_root, str) and nas_root:
            roots += paths_in_settings(_read_json(Path(nas_root) / "mousereach_lab_config.json"))
    return roots


def roots_from_environment(environ):
    """Real roots named by environment variables, whatever their letter case."""
    roots = []
    for name, value in environ.items():
        upper = name.upper()
        if upper.startswith(_ENV_PREFIXES) or upper in _ENV_NAMES:
            for piece in value.split(os.pathsep):
                if _looks_absolute(piece):
                    roots.append(_as_root(piece))
    return roots


# Where Windows keeps the variables it gives every new program: the account's
# own and the machine's. An isolated run removes the tool's variables from its
# own environment, but they are still set here -- and still name real roots.
_PERSISTED_ENV_KEYS = (
    ("HKEY_CURRENT_USER", r"Environment"),
    ("HKEY_LOCAL_MACHINE", r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
)


def persisted_environment():
    """The account's and the machine's saved environment variables (Windows
    registry), as {name: value}; {} elsewhere or when unreadable. Read only."""
    found = {}
    if not _WINDOWS:
        return found
    try:
        import winreg
    except ImportError:
        return found
    for hive_name, key_path in _PERSISTED_ENV_KEYS:
        try:
            key = winreg.OpenKey(getattr(winreg, hive_name), key_path, 0, winreg.KEY_READ)
        except OSError:
            continue
        try:
            i = 0
            while True:
                try:
                    name, value, kind = winreg.EnumValue(key, i)
                except OSError:
                    break
                i += 1
                if isinstance(value, str):
                    if kind == winreg.REG_EXPAND_SZ:
                        try:
                            value = winreg.ExpandEnvironmentStrings(value)
                        except OSError:
                            pass
                    found.setdefault(name, value)
        finally:
            winreg.CloseKey(key)
    return found


def roots_supplied_by_runner(environ):
    """Real roots the person running the tests listed (see module docstring)."""
    roots = [p.strip() for p in environ.get(ROOTS_ENV, "").split(os.pathsep) if p.strip()]
    listing = environ.get(ROOTS_FILE_ENV, "").strip()
    if listing:
        try:
            with open(listing, "r", encoding="utf-8-sig") as fh:
                for line in fh:
                    line = line.split("#", 1)[0].strip()
                    if line:
                        roots.append(line)
        except OSError as exc:
            # Said, not swallowed: a person who listed roots expects them guarded.
            sys.stderr.write("test guard: cannot read %s=%s: %s\n" % (ROOTS_FILE_ENV, listing, exc))
    return roots


def roots_from_tool_config():
    """What MouseReach's own config module resolved in this process."""
    roots = []
    try:
        import mousereach.config as config
        for value in (config.Paths.NAS_ROOT, config.PROCESSING_ROOT, config.NAS_DRIVE):
            if value:
                roots.append(str(value))
    except Exception:
        pass
    return roots


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

_MAPPED = _mapped_drives()


class Guard:
    """Real roots (each a set of spellings) and exempt folders, and the hits."""

    def __init__(self):
        self.roots = []      # [(label, frozenset(spellings))]
        self.exempt = []     # [(label, frozenset(spellings))]
        self.hits = []       # [(event, path)]
        self._exempt_inside = {}   # root index -> set of exempt indexes inside it

    def _recompute(self):
        self._exempt_inside = {}
        for ri, (_, rsp) in enumerate(self.roots):
            inside = set()
            for ei, (_, esp) in enumerate(self.exempt):
                if any(_under(e, r) for e in esp for r in rsp):
                    inside.add(ei)
            self._exempt_inside[ri] = inside

    def add_root(self, path):
        sp = spellings(path)
        if sp and all(sp != existing for _, existing in self.roots):
            self.roots.append((str(path), frozenset(sp)))
            self._recompute()

    def remove_root(self, path):
        sp = frozenset(spellings(path))
        self.roots = [r for r in self.roots if r[1] != sp]
        self._recompute()

    def add_exempt(self, path):
        sp = spellings(path)
        if sp and all(sp != existing for _, existing in self.exempt):
            self.exempt.append((str(path), frozenset(sp)))
            self._recompute()

    def refused(self, path):
        """The path's spelling when a write to it must be refused, else None.
        Pure string work."""
        n = norm(path)
        if not n:
            return None
        for ri, (_, rsp) in enumerate(self.roots):
            if not any(_under(n, r) for r in rsp):
                continue
            exempt_here = any(
                any(_under(n, e) for e in self.exempt[ei][1])
                for ei in self._exempt_inside.get(ri, ()))
            if not exempt_here:
                return n
        return None

    def audit(self, event, args):
        if event == "open":
            if len(args) < 3:
                return
            mode, flags = args[1], args[2]
            if isinstance(mode, str):
                if not any(c in mode for c in "wax+"):
                    return
            elif not (isinstance(flags, int) and flags & _WRITE_FLAGS):
                return
            places = (args[0],)
        elif event == "sqlite3.connect":
            if not args:
                return
            target = _sqlite_write_target(args[0])
            if target is None:
                return
            places = (target,)
        else:
            positions = _ARG_POSITIONS.get(event)
            if positions is None:
                return
            places = tuple(args[i] for i in positions if i < len(args))
        for place in places:
            if not isinstance(place, (str, bytes, os.PathLike)):
                continue
            hit = self.refused(place)
            if hit:
                self.hits.append((event, hit))
                raise PermissionError(
                    "test guard: %s on this machine's real pipeline folder refused: %s"
                    % (event, hit))


def collect_real_roots(environ=None, homes=None, persisted=None):
    """Every real root this machine names, from every source (module docstring)."""
    environ = os.environ if environ is None else environ
    homes = os_home_folders() if homes is None else homes
    persisted = persisted_environment() if persisted is None else persisted
    return (roots_from_settings_files(homes) + roots_from_environment(environ)
            + roots_from_environment(persisted)
            + roots_supplied_by_runner(environ) + roots_from_tool_config())


def build_guard(environ=None, homes=None, repo_root=None, persisted=None):
    guard = Guard()
    for root in collect_real_roots(environ, homes, persisted):
        guard.add_root(root)
    import tempfile
    exempt = [tempfile.gettempdir(), sys.prefix, sys.base_prefix,
              sys.exec_prefix, sys.base_exec_prefix]
    if repo_root:
        exempt.append(str(repo_root))
    for path in exempt:
        guard.add_exempt(path)
    return guard


GUARD = None


def install(repo_root=None):
    """Build the guard and hook it into the interpreter (once per process).
    Installed even with no real root, so a root a test adds is still guarded."""
    global GUARD
    if GUARD is None:
        GUARD = build_guard(repo_root=repo_root)
        sys.addaudithook(lambda event, args: GUARD.audit(event, args))
    return GUARD
