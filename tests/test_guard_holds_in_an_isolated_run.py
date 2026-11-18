"""The test guard holds in a run isolated the way test runs are isolated:
home folders moved to a scratch folder and every MouseReach variable removed.

WHY: an isolated run is exactly when the guard used to know NO real root -- it
found them only through the current home folder's settings and the tool's own
variables -- so a test that found its way back to the real folders wrote there
unhindered. And when it did know a root, it refused nothing outside a test and
only reported inside one, after the write.

Each test starts a separate Python with that isolated environment, loads
tests/conftest.py exactly as pytest does, and tries four kinds of write into a
folder under tmp_path that stands in for a real root: an open() in write mode,
an os.open create (the temporary file of an atomic write), a tempfile.mkstemp
and a folder create. Every one must be refused and the folder stay empty.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# The variables a person running the tests sets to list real roots
# (docs/TEST_GUARD.md).
ROOTS_ENV = "REACH_TEST_GUARD_ROOTS"
ROOTS_FILE_ENV = "REACH_TEST_GUARD_ROOTS_FILE"


_CHILD = r'''
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[1])
import conftest  # loads the guard exactly as pytest does
root = sys.argv[2]
outcome = {}
def attempt(name, fn):
    try:
        fn()
        outcome[name] = "written"
    except PermissionError as exc:
        outcome[name] = "refused" if "test guard" in str(exc) else "other: %s" % exc
attempt("open", lambda: open(os.path.join(root, "a.json"), "w").close())
attempt("os.open", lambda: os.close(os.open(os.path.join(root, "b.tmp"), os.O_CREAT | os.O_WRONLY)))
attempt("mkstemp", lambda: os.close(tempfile.mkstemp(dir=root)[0]))
attempt("mkdir", lambda: os.mkdir(os.path.join(root, "Processing")))
print("RESULT " + json.dumps(outcome))
'''


def _isolated_env(tmp_path):
    """The environment of an isolated run: home folders moved to a scratch
    folder and every MouseReach/mousedb/MouseBrain/CONNECTOME_ROOT variable
    removed, whatever its letter case."""
    home = tmp_path / "iso_home"
    (home / "AppData" / "Local").mkdir(parents=True)
    (home / "AppData" / "Roaming").mkdir(parents=True)
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(("MOUSEREACH_", "MOUSEDB_", "MOUSEBRAIN_"))
           and k.upper() != "CONNECTOME_ROOT"
           and k not in (ROOTS_ENV, ROOTS_FILE_ENV)}
    env.update(USERPROFILE=str(home), HOME=str(home),
               APPDATA=str(home / "AppData" / "Roaming"),
               LOCALAPPDATA=str(home / "AppData" / "Local"))
    return env


def _run_child(env, real):
    script = Path(env["HOME"]).parent / "child_probe.py"
    script.write_text(_CHILD)
    proc = subprocess.run([sys.executable, str(script), str(TESTS_DIR), str(real)],
                          env=env, capture_output=True, text=True, timeout=300)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, "child produced no result:\n" + proc.stdout + proc.stderr
    return json.loads(lines[-1][len("RESULT "):])


def _expect_all_refused(outcome, real):
    assert outcome == {k: "refused" for k in ("open", "os.open", "mkstemp", "mkdir")}, outcome
    assert sorted(p.name for p in real.iterdir()) == []


def test_an_isolated_run_still_refuses_a_root_the_runner_lists(tmp_path):
    real = tmp_path / "listed_root"
    real.mkdir()
    env = _isolated_env(tmp_path)
    env[ROOTS_ENV] = str(real)
    _expect_all_refused(_run_child(env, real), real)


def test_an_isolated_run_still_refuses_a_root_listed_in_a_file(tmp_path):
    real = tmp_path / "listed_in_file"
    real.mkdir()
    listing = tmp_path / "roots.txt"
    listing.write_text("# real roots of this machine\n%s\n" % real)
    env = _isolated_env(tmp_path)
    env[ROOTS_FILE_ENV] = str(listing)
    _expect_all_refused(_run_child(env, real), real)


def test_a_mixed_case_tool_variable_names_a_real_root(tmp_path):
    # A machine may set MouseReach_PROCESSING_ROOT (mixed case), which the
    # tool's own config reads: what it names is a real root.
    real = tmp_path / "mixed_case_root"
    real.mkdir()
    env = _isolated_env(tmp_path)
    env["MouseReach_PROCESSING_ROOT"] = str(real)
    _expect_all_refused(_run_child(env, real), real)
