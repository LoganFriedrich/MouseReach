"""Suite-wide guard: no test may write into this machine's REAL pipeline folders.

WHY: tests build their own folders under tmp_path and point config.Paths at
them. A test that forgets one attribute silently acts on live lab data instead.
One did: a watcher-integration fixture built a real orchestrator, whose
constructor created the real post-pose staging folder on the shared drive. It
went unnoticed because the old folder already existed, and surfaced only when
that folder was renamed during a maintenance freeze.

HOW: an audit hook (sys.addaudithook) records every folder create, rename or
delete, every write-mode open, and every SQLite connection under the real
roots -- and, while a test is running, REFUSES it (PermissionError) before it
happens. WHY refuse, not only record: a test that wrote one row into the real
shared version index failed afterwards, as designed, but the row stayed in the
live index, where it was the newest entry; a failing test must leave nothing
behind. Code that swallows the error (best-effort bookkeeping) still has the
attempt recorded, so the test still fails. The roots are resolved from this machine's configuration when this file
is imported, before any test patches Paths. After each test, any such event
fails that test and names the path. A folder create is recorded even when the
folder already exists -- exactly the case that hid the original bug. On a
machine with no pipeline configured there are no real roots and the guard does
nothing.
"""
import os
import sys
from pathlib import Path

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
# True only while a test body (and its fixtures) runs; outside a test -- at
# import and collection -- an attempt is recorded but not refused.
_refusing = [False]


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
            if _refusing[0]:
                raise PermissionError(
                    f"test guard: {event} on this machine's real pipeline folder "
                    f"refused: {hit}")


if _REAL_ROOTS:
    sys.addaudithook(_audit)


@pytest.fixture(autouse=True)
def _no_writes_to_real_pipeline_folders():
    start = len(_hits)
    _refusing[0] = True
    try:
        yield
    finally:
        _refusing[0] = False
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


@pytest.fixture(autouse=True)
def _no_real_message_boxes(monkeypatch):
    """No test puts a real message box on this machine's screen. WHY: tests
    run watchers configured to pause for a recording program, and the watcher
    then shows its "stopping" / "safe to record" boxes (watcher/record_notice)
    -- on a machine where the suite runs while a person works, those boxes
    appeared on their screen for no recording at all. A test of the boxes
    passes its own show/close functions.

    The "is one already up?" check is answered "no" unless a test passes its
    own finder: asked of the real screen, a box left up by anything else on
    the machine made a test's notice look like a duplicate and skip it."""
    from mousereach.watcher import record_notice
    real_already_on_screen = record_notice.already_on_screen
    monkeypatch.setattr(
        record_notice, "already_on_screen",
        lambda title, find=None: (real_already_on_screen(title, find=find)
                                  if find is not None else False))
    shown = []
    monkeypatch.setattr(record_notice, "_show_windows_box",
                        lambda title, text: shown.append(title))
    monkeypatch.setattr(record_notice, "_close_windows_box", lambda title: False)
    return shown


@pytest.fixture(autouse=True)
def _no_real_hand_scores(monkeypatch):
    """No test reads this machine's hand-scores settings: the check before
    filing (watcher/hand_scores.py) would otherwise compare a test's video with
    the lab's real scores file, or hold it for scores it can never have. A test
    of the check sets hand_scores.settings itself."""
    from mousereach.watcher import hand_scores
    monkeypatch.setattr(hand_scores, "settings", lambda: {})


@pytest.fixture(autouse=True)
def _recording_check_passes(request, monkeypatch):
    """Every test's stand-in recordings pass the recording check
    (watcher/recording_issue.py) unless the test is marked
    ``real_recording_check``. WHY: tests use a few bytes as a "collage", which
    the real check rightly calls a failed recording -- and would set aside
    before the behavior the test is about could run. A test of the check
    itself is marked, and injects its own probe."""
    if request.node.get_closest_marker("real_recording_check"):
        return
    from mousereach.watcher import recording_issue
    monkeypatch.setattr(recording_issue, "assess",
                        lambda path, settings=None, **kw: recording_issue.Verdict("ok"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_recording_check: run the real recording check (no stand-in)")


# The filing layout every test sees unless it sets its own: one project folder
# level, cohort folders named code+number. Tests written before the filing
# rules were settings build their fixtures in this shape.
TEST_FILING_RULES = {
    "CNT": {"folder": "Project_A", "cohort_folder": "{CODE}{NN}"},
    "ENCR": {"folder": "Enhancer", "cohort_folder": "{CODE}{NN}"},
}


@pytest.fixture(autouse=True)
def _no_real_filing_rules(monkeypatch):
    """No test reads this machine's filing rules (the lab settings file on the
    storage drive): where a test's video is filed must not depend on the lab's
    real layout. A test of the rules themselves sets config.filing_rules."""
    import mousereach.config as config
    monkeypatch.setattr(config, "filing_rules", lambda: dict(TEST_FILING_RULES))


@pytest.fixture(autouse=True)
def _no_real_shared_indexes(monkeypatch, tmp_path_factory):
    """No test opens this machine's shared SQLite index by default path: the
    version index (pipeline/version_index.py). It finds its file from
    config.Paths when no path is given, and it is written as a side effect of
    ordinary calls (writing a processing manifest updates it). (A second one,
    the triage queue index, was removed: nothing read it.)

    WHY, when the audit guard above already watches sqlite3.connect: the guard
    reports a write AFTER it has happened, and only on a machine whose real
    roots it could resolve. A test that wrote a processing manifest without
    patching Paths.NAS_ROOT put a row for its invented video into a live
    version index before anything failed. Here the default path itself is
    turned away: whenever it would resolve outside pytest's own temporary
    folder, it resolves to a file inside it instead. A test that points
    config.Paths at tmp_path keeps the path it expects.
    """
    base = tmp_path_factory.getbasetemp().resolve()

    def _inside_temp(path):
        try:
            Path(path).resolve().relative_to(base)
            return True
        except (ValueError, OSError):
            return False

    def _guarded(module, name):
        real = module.default_index_path

        def default_index_path():
            found = real()
            if found is not None and _inside_temp(found):
                return found
            return base / "_default_indexes" / name
        monkeypatch.setattr(module, "default_index_path", default_index_path)

    import mousereach.pipeline.version_index as version_index
    _guarded(version_index, "version_index.db")
