"""How the test guard (tests/real_root_guard.py) decides what is a real root and
which paths under it are refused. These tests use a Guard of their own, never
the installed one, so nothing here can refuse or record a real write.

Covers: the audit events and their arguments (os.open's int flags included),
every spelling of a root (case, slashes, a mapped drive's network form), the
sources of the real roots when the home folder is redirected and the tool's
variables are unset, the repository's own files being exempt while its
siblings stay guarded, and pytest's cache never landing in the repository.
"""
import json
import os
import tempfile
from pathlib import Path

import pytest

import conftest
import real_root_guard as rg

REPO_ROOT = Path(__file__).resolve().parent.parent


def _guard(*roots, exempt=()):
    g = rg.Guard()
    for r in roots:
        g.add_root(str(r))
    for e in exempt:
        g.add_exempt(str(e))
    return g


def _event_refused(g, event, args):
    try:
        g.audit(event, args)
        return False
    except PermissionError:
        return True


def test_a_folder_create_under_a_real_root_is_refused_and_recorded(tmp_path):
    root = tmp_path / "real_share"
    g = _guard(root)
    assert _event_refused(g, "os.mkdir", (str(root / "Processing" / "Posed"), 0o777, None))
    assert [e for e, _ in g.hits] == ["os.mkdir"]


def test_write_opens_are_refused_and_read_opens_are_not(tmp_path):
    root = tmp_path / "real_share"
    g = _guard(root)
    target = str(root / "pipeline_versions.json")
    assert _event_refused(g, "open", (target, "w", 0))
    assert not _event_refused(g, "open", (target, "r", 0))
    # os.open: mode None, int flags.
    assert _event_refused(g, "open", (target + ".tmp", None, os.O_CREAT | os.O_WRONLY))
    assert _event_refused(g, "open", (target, None, os.O_RDWR))
    assert not _event_refused(g, "open", (target, None, os.O_RDONLY))


def test_a_copy_is_refused_only_at_its_destination(tmp_path):
    root = tmp_path / "real_share"
    g = _guard(root)
    assert not _event_refused(g, "shutil.copyfile", (str(root / "a"), str(tmp_path / "b")))
    assert _event_refused(g, "shutil.copyfile", (str(tmp_path / "b"), str(root / "a")))
    # a rename changes both ends
    assert _event_refused(g, "os.rename", (str(root / "a"), str(tmp_path / "b"), None, None))
    assert _event_refused(g, "os.replace", (str(tmp_path / "b"), str(root / "a"), None, None))


def test_a_database_connection_under_a_real_root_is_refused(tmp_path):
    root = tmp_path / "real_share"
    g = _guard(root)
    assert _event_refused(g, "sqlite3.connect", (str(root / "watcher.db"),))


def test_paths_outside_the_real_roots_are_ignored(tmp_path):
    root = tmp_path / "real_share"
    g = _guard(root)
    assert not _event_refused(g, "os.mkdir", (str(tmp_path / "real_share_sibling" / "x"), 0o777, None))
    assert not _event_refused(g, "os.rename", (str(tmp_path / "a"), str(tmp_path / "b"), None, None))
    assert g.hits == []


def test_case_slashes_and_relative_spellings_match(tmp_path, monkeypatch):
    root = tmp_path / "Real_Share"
    root.mkdir()
    g = _guard(root)
    assert g.refused(str(root / "x"))
    assert g.refused(str(root).replace("\\", "/") + "/x")
    monkeypatch.chdir(tmp_path)
    assert g.refused(os.path.join("Real_Share", "sub", "x"))
    if os.name == "nt":
        assert g.refused(str(root).upper() + "\\X")
        assert g.refused("\\\\?\\" + str(root) + "\\x")


@pytest.mark.skipif(os.name != "nt", reason="drive letters are Windows only")
def test_a_mapped_drive_matches_in_its_network_form_both_ways(monkeypatch):
    monkeypatch.setattr(rg, "_MAPPED", {"q:": r"\\fileserver\share"})
    g = _guard(r"Q:\pipeline")
    assert g.refused(r"\\fileserver\share\pipeline\Processing\x.json")
    assert g.refused(r"\\?\UNC\fileserver\share\pipeline\x.json")
    assert not g.refused(r"\\fileserver\share\other\x.json")
    g2 = _guard(r"\\fileserver\share\pipeline")
    assert g2.refused(r"q:/pipeline/x.json")


def test_a_drive_root_is_a_root(tmp_path):
    drive = os.path.splitdrive(str(tmp_path))[0]
    if not drive or drive.startswith("\\\\"):
        pytest.skip("no drive letter here")
    g = _guard(drive + os.sep)
    assert g.refused(drive + os.sep + "anything")


def test_the_repository_is_exempt_but_its_siblings_are_not():
    g = _guard(REPO_ROOT.parent, exempt=[REPO_ROOT])
    assert not g.refused(str(REPO_ROOT / "tests" / "__pycache__" / "x.pyc"))
    assert not g.refused(str(REPO_ROOT / "src" / "mousereach" / "x.py"))
    assert g.refused(str(REPO_ROOT.parent / (REPO_ROOT.name + "_Pipeline") / "x"))
    assert g.refused(str(REPO_ROOT.parent / "x"))


def test_an_exempt_folder_wins_only_inside_the_root(tmp_path):
    # A real root inside the temp folder is still guarded ...
    g = _guard(tmp_path / "root_in_temp", exempt=[tmp_path])
    assert g.refused(str(tmp_path / "root_in_temp" / "x"))
    # ... and a temp folder inside a real root (a whole drive) stays usable.
    g2 = _guard(tmp_path, exempt=[tmp_path / "temp"])
    assert not g2.refused(str(tmp_path / "temp" / "x"))
    assert g2.refused(str(tmp_path / "elsewhere" / "x"))


def test_the_installed_guard_exempts_this_repository_and_the_temp_folder():
    exempt = {s for _, sp in conftest.GUARD.exempt for s in sp}
    assert rg.norm(str(REPO_ROOT)) in exempt
    assert rg.norm(tempfile.gettempdir()) in exempt


# --- where the real roots come from ------------------------------------------

def test_tool_variables_of_any_letter_case_name_real_roots(tmp_path):
    a, b, c = (str(tmp_path / n) for n in ("a", "b", "c"))
    roots = rg.roots_from_environment({
        "MouseReach_PROCESSING_ROOT": a,
        "mousereach_nas_drive": b,
        "CONNECTOME_ROOT": c,
        "MOUSEREACH_LAB_PROFILES": str(tmp_path / "d" / "profiles.json"),
        "UNRELATED_ROOT": str(tmp_path / "e"),
        "MOUSEREACH_FLAG": "1",
    })
    assert set(roots) == {a, b, c, str(tmp_path / "d")}


def test_settings_files_at_a_home_folder_name_real_roots(tmp_path):
    nas = tmp_path / "shared"
    nas.mkdir()
    (nas / "mousereach_lab_config.json").write_text(json.dumps(
        {"hand_scores": {"path": str(tmp_path / "scores" / "hand.csv")},
         "filing": {"PRJ": {"folder": "Project_A"}}}))
    home = tmp_path / "home"
    (home / ".mousereach").mkdir(parents=True)
    (home / ".mousereach" / "config.json").write_text(json.dumps(
        {"nas_root": str(nas), "processing_root": str(tmp_path / "proc"),
         "watcher": {"db_path": str(tmp_path / "proc" / "watcher_local.db"), "mode": "x"}}))
    roots = rg.roots_from_settings_files([str(home)])
    assert set(roots) == {str(nas), str(tmp_path / "proc"), str(tmp_path / "scores")}


def test_the_real_home_folder_is_found_when_the_home_variables_are_moved(tmp_path, monkeypatch):
    for name in ("USERPROFILE", "HOME"):
        monkeypatch.setenv(name, str(tmp_path))
    homes = [rg.norm(h) for h in rg.os_home_folders()]
    assert rg.norm(str(tmp_path)) in homes
    assert any(h != rg.norm(str(tmp_path)) for h in homes), (
        "the account's own home folder (as the operating system records it) "
        "must be read too, so its real settings are guarded in an isolated run")


def test_the_runner_may_list_roots(tmp_path):
    listing = tmp_path / "roots.txt"
    listing.write_text("# comment\n%s\n\n%s  # trailing comment\n"
                       % (tmp_path / "r1", tmp_path / "r2"))
    roots = rg.roots_supplied_by_runner({
        rg.ROOTS_ENV: os.pathsep.join([str(tmp_path / "r0"), ""]),
        rg.ROOTS_FILE_ENV: str(listing)})
    assert roots == [str(tmp_path / "r0"), str(tmp_path / "r1"), str(tmp_path / "r2")]


def test_pytest_never_writes_its_cache_into_the_repository():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    addopts = [l for l in text.splitlines() if l.strip().startswith("addopts")]
    assert addopts and "-p no:cacheprovider" in addopts[0]
