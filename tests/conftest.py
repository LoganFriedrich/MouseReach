"""Suite-wide guard: no test may write into this machine's REAL pipeline folders.

WHY: tests build their own folders under tmp_path and point config.Paths at
them. A test that forgets one attribute silently acts on live lab data instead.
One did: a watcher-integration fixture built a real orchestrator, whose
constructor created the real post-pose staging folder on the shared drive. It
went unnoticed because the old folder already existed, and surfaced only when
that folder was renamed during a maintenance freeze.

HOW: an audit hook (sys.addaudithook) records every folder create, rename or
delete, every write-mode open, and every SQLite connection under the real
roots. The roots are resolved from this machine's configuration when this file
is imported, before any test patches Paths. After each test, any such event
fails that test and names the path. A folder create is recorded even when the
folder already exists -- exactly the case that hid the original bug. On a
machine with no pipeline configured there are no real roots and the guard does
nothing.
"""
import os
import sys

import pytest

_WRITE_EVENTS = frozenset({
    "os.mkdir", "os.rename", "os.replace", "os.remove", "os.rmdir",
    "shutil.copyfile", "shutil.copytree", "shutil.move", "shutil.rmtree",
    "sqlite3.connect",
    # A touch is a write too: the claimed-singles heartbeat refreshes files on
    # the share with os.utime, and a test must never refresh a real claim.
    "os.utime",
})


def _norm(path):
    try:
        return os.path.normcase(os.path.abspath(os.fsdecode(os.fspath(path))))
    except (TypeError, ValueError, OSError):
        return None


def _real_roots():
    """The live pipeline roots on this machine: the shared pipeline folder and
    the node's own processing root. Only folders that exist are guarded."""
    roots = []
    try:
        from mousereach.config import Paths
        if Paths.NAS_ROOT:
            roots.append(Paths.NAS_ROOT)
    except Exception:
        pass
    try:
        from mousereach.config import require_processing_root
        roots.append(require_processing_root())
    except Exception:
        pass
    found = []
    for root in roots:
        n = _norm(root)
        if n and os.path.isdir(n) and n not in found:
            found.append(n)
    return tuple(found)


_REAL_ROOTS = _real_roots()
_hits = []


def _inside_real_root(path):
    n = _norm(path)
    if not n:
        return None
    for root in _REAL_ROOTS:
        if n == root or n.startswith(root + os.sep):
            return n
    return None


def _audit(event, args):
    if event == "open":
        # Check the mode first: every import opens files for reading, and those
        # must stay cheap.
        if len(args) < 2 or not isinstance(args[1], str) or not any(c in args[1] for c in "wax+"):
            return
        candidates = [args[0]]
    elif event in _WRITE_EVENTS:
        candidates = list(args)
    else:
        return
    for candidate in candidates:
        if not isinstance(candidate, (str, bytes, os.PathLike)):
            continue
        hit = _inside_real_root(candidate)
        if hit:
            _hits.append((event, hit))


if _REAL_ROOTS:
    sys.addaudithook(_audit)


@pytest.fixture(autouse=True)
def _no_writes_to_real_pipeline_folders():
    start = len(_hits)
    yield
    new = _hits[start:]
    if new:
        del _hits[start:]
        shown = "\n".join(f"  {event}: {path}" for event, path in new[:10])
        more = f"\n  ... and {len(new) - 10} more" if len(new) > 10 else ""
        pytest.fail(
            "this test touched this machine's REAL pipeline folders; point "
            "config.Paths (and the processing root) at tmp_path instead:\n"
            + shown + more,
            pytrace=False)
