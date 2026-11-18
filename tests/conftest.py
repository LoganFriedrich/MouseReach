"""Suite-wide fixtures. The first and most important is the test guard: no test
may write into this machine's REAL pipeline folders.

WHY: tests build their own folders under tmp_path and point config.Paths at
them. A test that forgets one attribute silently acts on live lab data instead.
One did: a watcher-integration fixture built a real orchestrator, whose
constructor created the real post-pose staging folder on the shared drive. It
went unnoticed because the old folder already existed, and surfaced only when
that folder was renamed during a maintenance freeze. Another wrote one row into
the real shared version index; the guard of the time only reported it after
the fact, so the row stayed in the live index.

HOW: tests/real_root_guard.py (loaded here, before any test or test module
runs) collects every real root this machine names -- from the settings files
at their real per-machine places, from MouseReach variables of any letter case,
and from a list the person running the tests may supply -- and installs an
audit hook that REFUSES (PermissionError) any create, write, rename, delete or
database open under them, before it happens. Each refusal is also recorded:
after each test any record fails that test and names the path, so code that
swallows the error still fails. A record made outside a test (while modules are
collected) fails the whole run. What the guard covers and how to supply roots,
for a person who does not read code: docs/TEST_GUARD.md.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_root_guard  # noqa: E402

# The repository's own files are exempt even when it sits inside a real root
# (its siblings stay guarded).
_REPO_ROOT = Path(__file__).resolve().parent.parent
GUARD = real_root_guard.install(repo_root=_REPO_ROOT)

# Records made outside any test (module import, collection); reported at the end.
_outside_tests = []
_in_test = [False]


def pytest_configure(config):
    # A --basetemp given on the command line is pytest's own temporary folder.
    basetemp = getattr(config.option, "basetemp", None)
    if basetemp:
        GUARD.add_exempt(str(basetemp))
    config.addinivalue_line(
        "markers", "real_recording_check: run the real recording check (no stand-in)")


def pytest_collectreport(report):
    _collect_outside_hits()


def _collect_outside_hits():
    if not _in_test[0] and GUARD.hits:
        _outside_tests.extend(GUARD.hits)
        del GUARD.hits[:]


def pytest_terminal_summary(terminalreporter):
    _collect_outside_hits()
    if _outside_tests:
        terminalreporter.section("test guard", sep="=", red=True)
        terminalreporter.write_line(
            "Outside any test, something tried to write into this machine's REAL "
            "pipeline folders (refused):")
        for event, path in _outside_tests[:20]:
            terminalreporter.write_line("  %s: %s" % (event, path))


def pytest_sessionfinish(session, exitstatus):
    _collect_outside_hits()
    if _outside_tests and session.exitstatus == 0:
        session.exitstatus = 1


@pytest.fixture(autouse=True)
def _no_writes_to_real_pipeline_folders():
    _collect_outside_hits()
    start = len(GUARD.hits)
    _in_test[0] = True
    try:
        yield
    finally:
        _in_test[0] = False
    new = GUARD.hits[start:]
    if new:
        del GUARD.hits[start:]
        shown = "\n".join(f"  {event}: {path}" for event, path in new[:10])
        more = f"\n  ... and {len(new) - 10} more" if len(new) > 10 else ""
        pytest.fail(
            "this test tried to write into this machine's REAL pipeline folders "
            "(the test guard refused it); point config.Paths (and the processing "
            "root) at tmp_path instead:\n" + shown + more,
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

    WHY, when the guard above already refuses sqlite3.connect under a real
    root: the guard knows only the roots it could collect, and a refused write
    still fails the test. A test that wrote a processing manifest without
    patching Paths.NAS_ROOT once put a row for its invented video into a live
    version index (back when the guard only reported). Here the default path itself is
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
