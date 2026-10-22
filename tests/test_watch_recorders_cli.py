"""The commands a person uses to see and change why the watcher is paused.

WHY these tests: a GPU node that also records videos pauses itself while a
configured recording program runs. Three commands have to tell the same
story, and each has a way to do real harm if it is wrong:

  * mousereach-watch-recorders edits the user's config file IN PLACE. If it
    dropped any other key, the machine would lose its processing_root or
    nas_root the first time someone added a program -- and a broken file must
    never be "fixed" by overwriting it.
  * mousereach-watch-toggle --pause / --resume must do what they say even when
    run twice, and --resume must say plainly when a recording program still
    holds the watcher, or the operator waits for work that cannot start.
  * mousereach-watch-status must say the watcher is paused and why, and name
    the shared folders the node works from.

Nothing here touches this machine's real config, processing folder, process
list or database: the home folder, the processing root, the process scan, the
stage folders and the watcher database are all pointed at tmp_path or faked.
"""
import json
import pathlib
import sys

import pytest

import mousereach.config as config
import mousereach.watcher.cli as cli
import mousereach.watcher.db as watcher_db
import mousereach.watcher.recording_guard as rg


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def node(tmp_path, monkeypatch):
    """A fake machine: its own home folder, processing root and process list."""
    home = tmp_path / "home"
    (home / ".mousereach").mkdir(parents=True)
    root = tmp_path / "processing"
    root.mkdir()

    # Both the command's writer and WatcherConfig.load() resolve the config
    # file through Path.home(), so one patch points both at the fake home.
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(config, "require_processing_root", lambda: root)

    running = []

    def fake_running_programs(names):
        wanted = {r.lower() for r in running}
        return [n for n in rg.clean_names(names) if n.lower() in wanted]

    monkeypatch.setattr(rg, "running_programs", fake_running_programs)

    class Node:
        pass

    n = Node()
    n.home = home
    n.root = root
    n.flag = root / "watcher_paused.flag"
    n.config_file = home / ".mousereach" / "config.json"
    n.running = running
    return n


def _write_config(node, data):
    node.config_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_config(node):
    return json.loads(node.config_file.read_text(encoding="utf-8"))


def _run(monkeypatch, capsys, func, prog, *args):
    """Run one console command; return (exit code, stdout, stderr)."""
    monkeypatch.setattr(sys, "argv", [prog, *args])
    code = 0
    try:
        func()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    out = capsys.readouterr()
    # House rule: a Windows console cannot print non-ASCII; one character
    # crashes the command.
    out.out.encode("ascii")
    out.err.encode("ascii")
    return code, out.out, out.err


def recorders(monkeypatch, capsys, *args):
    return _run(monkeypatch, capsys, cli.main_recorders,
                "mousereach-watch-recorders", *args)


def toggle(monkeypatch, capsys, *args):
    return _run(monkeypatch, capsys, cli.main_toggle,
                "mousereach-watch-toggle", *args)


# The rest of a real config file: none of it may change when the recording
# program list is edited.
def _full_config(tmp_path):
    return {
        "processing_root": str(tmp_path / "proc"),
        "nas_root": str(tmp_path / "share"),
        "another_section": {"keep": [1, 2, 3]},
        "watcher": {
            "mode": "dlc_pc",
            "also_process": True,
            "dlc_shuffle": 3,
            "work_priority": {"projects": ["PROJECT_A", "PROJECT_B"]},
        },
    }


# ---------------------------------------------------------------- recorders

def test_add_remove_and_grace_round_trip_keeps_every_other_setting(
        node, tmp_path, monkeypatch, capsys):
    original = _full_config(tmp_path)
    _write_config(node, original)

    code, out, _ = recorders(monkeypatch, capsys, "--add", "recorder.exe")
    assert code == 0
    assert _read_config(node)["watcher"]["pause_while_running"] == ["recorder.exe"]
    assert "Added recorder.exe" in out

    # Another spelling of the same program is not a second entry: the guard
    # would match both to the one process anyway.
    code, out, _ = recorders(monkeypatch, capsys, "--add", "RECORDER")
    assert code == 0
    assert _read_config(node)["watcher"]["pause_while_running"] == ["recorder.exe"]
    assert "already in the list" in out

    code, _, _ = recorders(monkeypatch, capsys,
                           "--add", "second.exe", "--grace", "45")
    assert code == 0
    saved = _read_config(node)["watcher"]
    assert saved["pause_while_running"] == ["recorder.exe", "second.exe"]
    assert saved["pause_resume_grace_seconds"] == 45

    # The file written is the file the watcher reads.
    loaded = config.WatcherConfig.load()
    assert loaded.pause_while_running == ["recorder.exe", "second.exe"]
    assert loaded.pause_resume_grace_seconds == 45

    code, out, _ = recorders(monkeypatch, capsys, "--remove", "Recorder.EXE")
    assert code == 0
    assert _read_config(node)["watcher"]["pause_while_running"] == ["second.exe"]
    assert "Removed Recorder.EXE" in out

    code, _, _ = recorders(monkeypatch, capsys, "--remove", "second.exe")
    assert code == 0
    final = _read_config(node)
    # An emptied list is removed, so the file reads as the default again.
    assert "pause_while_running" not in final["watcher"]
    assert final["watcher"]["pause_resume_grace_seconds"] == 45

    del final["watcher"]["pause_resume_grace_seconds"]
    assert final == original, "an unrelated setting was changed or lost"
    # No temporary file is left beside the config.
    assert sorted(p.name for p in node.config_file.parent.iterdir()) == ["config.json"]


def test_list_is_the_default_and_never_writes(node, tmp_path, monkeypatch, capsys):
    _write_config(node, _full_config(tmp_path))
    before = node.config_file.read_bytes()

    for args in ((), ("--list",)):
        code, out, _ = recorders(monkeypatch, capsys, *args)
        assert code == 0
        assert "Recording programs that pause this watcher: none" in out
        assert "Resume grace: 120 s" in out
    assert node.config_file.read_bytes() == before


def test_list_on_a_machine_with_no_config_file_creates_nothing(node, monkeypatch, capsys):
    code, out, _ = recorders(monkeypatch, capsys)
    assert code == 0
    assert "none" in out
    assert not node.config_file.exists()


def test_list_says_which_program_is_running_now(node, tmp_path, monkeypatch, capsys):
    data = _full_config(tmp_path)
    data["watcher"]["pause_while_running"] = ["recorder.exe", "other.exe"]
    _write_config(node, data)
    node.running.append("recorder.exe")

    code, out, _ = recorders(monkeypatch, capsys, "--list")
    assert code == 0
    lines = out.splitlines()
    assert any("recorder.exe" in l and "RUNNING now" in l for l in lines)
    assert any("other.exe" in l and "not running" in l for l in lines)
    assert "PAUSED" in out


def test_a_check_that_cannot_run_is_reported_as_paused(node, tmp_path, monkeypatch, capsys):
    """Recording must win: "could not look" is never shown as "not running"."""
    data = _full_config(tmp_path)
    data["watcher"]["pause_while_running"] = ["recorder.exe"]
    _write_config(node, data)

    def broken(names):
        raise RuntimeError("process list unavailable")

    monkeypatch.setattr(rg, "running_programs", broken)
    code, out, _ = recorders(monkeypatch, capsys)
    assert code == 0
    assert "cannot check" in out
    assert "PAUSED" in out
    assert "not running" not in out


@pytest.mark.parametrize("args", [
    ("--grace", "soon"),
    ("--grace", "-5"),
    ("--add",),
    ("--add", "   "),
    ("--remove", "--list"),
    # A typo in the LAST argument must not leave the first one half applied.
    ("--add", "recorder.exe", "--bogus"),
])
def test_bad_arguments_change_nothing(node, tmp_path, monkeypatch, capsys, args):
    _write_config(node, _full_config(tmp_path))
    before = node.config_file.read_bytes()
    code, _, err = recorders(monkeypatch, capsys, *args)
    assert code == 2
    assert "Nothing was changed" in err
    assert node.config_file.read_bytes() == before


def test_a_broken_config_file_is_never_overwritten(node, monkeypatch, capsys):
    """Rewriting an unreadable file would erase every setting on the machine."""
    node.config_file.write_text('{ "processing_root": ', encoding="utf-8")
    before = node.config_file.read_bytes()
    code, _, err = recorders(monkeypatch, capsys, "--add", "recorder.exe")
    assert code == 1
    assert "Nothing was changed" in err
    assert node.config_file.read_bytes() == before


def test_help_changes_nothing(node, monkeypatch, capsys):
    code, out, _ = recorders(monkeypatch, capsys, "--help", "--add", "recorder.exe")
    assert code == 0
    assert "usage: mousereach-watch-recorders" in out
    assert not node.config_file.exists()


# ---------------------------------------------------------------- toggle

def test_pause_and_resume_are_explicit_and_repeatable(node, monkeypatch, capsys):
    code, out, _ = toggle(monkeypatch, capsys, "--pause")
    assert code == 0 and node.flag.exists()
    assert "PAUSED" in out

    code, out, _ = toggle(monkeypatch, capsys, "--pause")
    assert code == 0 and node.flag.exists(), "a second --pause must not resume"
    assert "already paused" in out

    code, out, _ = toggle(monkeypatch, capsys, "--resume")
    assert code == 0 and not node.flag.exists()
    assert "RESUMED" in out

    code, out, _ = toggle(monkeypatch, capsys, "--resume")
    assert code == 0 and not node.flag.exists(), "a second --resume must not pause"
    assert "nothing to remove" in out

    # The plain toggle still flips.
    toggle(monkeypatch, capsys)
    assert node.flag.exists()
    toggle(monkeypatch, capsys)
    assert not node.flag.exists()


@pytest.mark.parametrize("args", [
    ("--pause", "--resume"),
    ("--status", "--pause"),
    ("--pasue",),
])
def test_toggle_refuses_unclear_arguments(node, monkeypatch, capsys, args):
    code, _, _ = toggle(monkeypatch, capsys, *args)
    assert code == 2
    assert not node.flag.exists()


def test_toggle_help_never_acts(node, monkeypatch, capsys):
    code, out, _ = toggle(monkeypatch, capsys, "--pause", "--help")
    assert code == 0
    assert "--resume" in out
    assert not node.flag.exists()


def test_toggle_status_shows_the_hand_pause_and_the_recording_program(
        node, tmp_path, monkeypatch, capsys):
    data = _full_config(tmp_path)
    data["watcher"]["pause_while_running"] = ["recorder.exe"]
    _write_config(node, data)
    node.flag.write_text("paused")
    node.running.append("recorder.exe")

    code, out, _ = toggle(monkeypatch, capsys, "--status")
    assert code == 0
    assert "Hand pause:          ON" in out
    assert "Recording programs:  recorder.exe" in out
    assert "Watcher is PAUSED -- paused by hand" in out
    assert node.flag.exists(), "--status must change nothing"


def test_toggle_status_with_only_a_recording_program(node, tmp_path, monkeypatch, capsys):
    data = _full_config(tmp_path)
    data["watcher"]["pause_while_running"] = ["recorder.exe"]
    _write_config(node, data)
    node.running.append("recorder.exe")

    code, out, _ = toggle(monkeypatch, capsys, "--status")
    assert code == 0
    assert "Hand pause:          off" in out
    assert "Watcher is PAUSED -- recorder.exe" in out
    assert not node.flag.exists()


def test_toggle_status_with_nothing_configured_is_active(node, monkeypatch, capsys):
    code, out, _ = toggle(monkeypatch, capsys, "--status")
    assert code == 0
    assert "none configured" in out
    assert "Watcher is ACTIVE" in out


def test_resume_while_recording_says_it_is_still_paused(node, tmp_path, monkeypatch, capsys):
    """--resume removes only the hand pause; recording always wins."""
    data = _full_config(tmp_path)
    data["watcher"]["pause_while_running"] = ["recorder.exe"]
    _write_config(node, data)
    node.flag.write_text("paused")
    node.running.append("recorder.exe")

    code, out, _ = toggle(monkeypatch, capsys, "--resume")
    assert code == 0
    assert not node.flag.exists()
    assert "Still PAUSED" in out
    assert "recorder.exe" in out


# ---------------------------------------------------------------- status

class _FakeWatcherDB:
    def __init__(self, path):
        self.path = path

    def get_pipeline_summary(self):
        return {"collages": {"total": 0, "by_state": {}},
                "videos": {"total": 0, "by_state": {}}}

    def get_recent_log(self, limit):
        return []


@pytest.fixture
def status_node(node, tmp_path, monkeypatch):
    db_file = tmp_path / "watcher.db"
    db_file.write_text("")
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: db_file)
    monkeypatch.setattr(watcher_db, "WatcherDB", _FakeWatcherDB)

    share = tmp_path / "share"
    folders = {
        "SINGLE_ANIMAL_OUTPUT": share / "Unanalyzed" / "Single_Animal",
        "DLC_STAGING": share / "Processing" / "Posed",
        "TRIAGE_REVIEW": share / "Processing" / "Review" / "triage",
        "DEEP_REVIEW": share / "Processing" / "Review" / "deep_review",
    }
    for attr, path in folders.items():
        monkeypatch.setattr(config.Paths, attr, path)
    # One folder is left missing, so the "not found" note is exercised.
    for attr in ("SINGLE_ANIMAL_OUTPUT", "DLC_STAGING", "TRIAGE_REVIEW"):
        folders[attr].mkdir(parents=True)
    node.folders = folders
    return node


def status(monkeypatch, capsys, *args):
    return _run(monkeypatch, capsys, cli.main_status,
                "mousereach-watch-status", *args)


def test_status_names_the_recording_program_and_the_stage_folders(
        status_node, tmp_path, monkeypatch, capsys):
    data = _full_config(tmp_path)
    data["watcher"]["pause_while_running"] = ["recorder.exe"]
    _write_config(status_node, data)
    status_node.running.append("recorder.exe")

    code, out, _ = status(monkeypatch, capsys)
    assert code == 0
    assert "Pause:  PAUSED -- recorder.exe is running" in out
    assert "Stage folders:" in out
    for attr, path in status_node.folders.items():
        assert f"Paths.{attr}" in out
        assert str(path) in out
    missing = [l for l in out.splitlines() if str(status_node.folders["DEEP_REVIEW"]) in l]
    assert missing and "[folder not found]" in missing[0]
    staging = [l for l in out.splitlines() if str(status_node.folders["DLC_STAGING"]) in l]
    assert staging and "[folder not found]" not in staging[0]
    # The existing report is still there.
    assert "Collages:" in out and "Videos:" in out


def test_status_says_not_paused(status_node, monkeypatch, capsys):
    code, out, _ = status(monkeypatch, capsys)
    assert code == 0
    assert "Pause:  not paused" in out


def test_status_says_paused_by_hand(status_node, monkeypatch, capsys):
    status_node.flag.write_text("paused")
    code, out, _ = status(monkeypatch, capsys)
    assert code == 0
    assert "Pause:  PAUSED -- paused by hand (watcher_paused.flag)" in out
    assert "mousereach-watch-toggle --resume" in out
