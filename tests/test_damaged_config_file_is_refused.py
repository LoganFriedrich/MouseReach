"""A damaged ~/.mousereach/config.json is refused, never read as empty.

WHY: a JSON error in the file used to read as "never set up": every path was
unset, the tools told the person to run mousereach-setup, the lab's shared
settings were looked up with no nas_root (so the hand-score check was
skipped), and a save from setup rewrote the file with only the few keys it
manages -- every other setting lost.
"""
import json

import pytest

import mousereach.config as config
import mousereach.setup.wizard as wizard


@pytest.fixture
def user_file(tmp_path, monkeypatch):
    f = tmp_path / "home" / ".mousereach" / "config.json"
    f.parent.mkdir(parents=True)
    monkeypatch.setattr(config, "config_file_path", lambda: f)
    monkeypatch.setattr(config, "_LAST_GOOD_CONFIG", {}, raising=False)
    monkeypatch.setattr(config, "_CONFIG_DAMAGE_REPORTED", set(), raising=False)
    monkeypatch.setattr(wizard, "CONFIG_DIR", f.parent)
    monkeypatch.setattr(wizard, "CONFIG_FILE", f)
    return f


DAMAGED = '{\n  "nas_root": "/shared/pipeline",\n  "processing_root": "/work"\n  "watcher": {}\n}\n'


def test_a_damaged_file_never_read_cleanly_stops_with_where_and_what_to_do(user_file):
    user_file.write_text(DAMAGED, encoding="utf-8")
    with pytest.raises(Exception) as err:
        config._load_config()
    said = str(err.value)
    assert str(user_file) in said
    assert "line 4" in said and "column" in said
    said.encode("ascii")
    assert isinstance(err.value, config.ConfigurationError)


def test_a_missing_or_empty_file_is_simply_not_set_up(user_file):
    assert config._load_config() == {}
    user_file.write_text("  \n", encoding="utf-8")
    assert config._load_config() == {}


def test_a_byte_order_mark_is_accepted(user_file):
    user_file.write_bytes(b"\xef\xbb\xbf" + json.dumps({"nas_root": "/x"}).encode("utf-8"))
    assert config._load_config() == {"nas_root": "/x"}


def test_a_file_that_breaks_under_a_running_process_keeps_the_last_good_copy(user_file, caplog):
    user_file.write_text(json.dumps({"nas_root": "/shared/pipeline"}), encoding="utf-8")
    assert config._load_config()["nas_root"] == "/shared/pipeline"
    user_file.write_text(DAMAGED, encoding="utf-8")
    assert config._load_config()["nas_root"] == "/shared/pipeline"
    assert "line 4" in caplog.text


def test_a_save_from_setup_never_overwrites_a_damaged_file(user_file):
    user_file.write_text(DAMAGED, encoding="utf-8")
    with pytest.raises(Exception):
        wizard.save_config(processing_root=user_file.parent / "work")
    assert user_file.read_text(encoding="utf-8") == DAMAGED      # every setting still there


def test_the_setup_command_says_what_is_wrong_and_exits_1(user_file, monkeypatch, capsys):
    user_file.write_text(DAMAGED, encoding="utf-8")
    monkeypatch.setattr(wizard.sys, "argv", ["mousereach-setup", "--show"])
    with pytest.raises(SystemExit) as ended:
        wizard.cli_main()
    assert ended.value.code == 1
    out = capsys.readouterr().out
    assert "[FAIL]" in out and "line 4" in out
    assert user_file.read_text(encoding="utf-8") == DAMAGED


# --- The real commands, in a fresh process, with the file damaged from the start.
# WHY a subprocess: mousereach.config reads the file when it is imported, so
# only a process that imports MouseReach AFTER the damage shows what a person
# who runs the command sees. The tests above import it while the file is clean.

import os
import subprocess
import sys
from pathlib import Path

import mousereach

SRC = str(Path(mousereach.__file__).resolve().parents[1])


def _run_fresh(home, code):
    """Run ``code`` in a new Python whose home folder is ``home`` and that
    imports MouseReach from this source tree; no machine setting can leak in."""
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(("MOUSEREACH_", "MOUSEDB_", "MOUSEBRAIN_"))
           and k.upper() != "CONNECTOME_ROOT"}
    env.update(HOME=str(home), USERPROFILE=str(home), PYTHONPATH=SRC,
               PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=str(home),
                          capture_output=True, text=True, timeout=300)


@pytest.fixture
def damaged_home(tmp_path):
    home = tmp_path / "fresh_home"
    (home / ".mousereach").mkdir(parents=True)
    (home / ".mousereach" / "config.json").write_text(DAMAGED, encoding="utf-8")
    return home


def test_importing_the_package_does_not_stop_on_a_damaged_file(damaged_home):
    r = _run_fresh(damaged_home, "import mousereach, mousereach.setup.wizard; print('loaded')")
    assert r.returncode == 0, r.stderr
    assert "loaded" in r.stdout


def test_the_setup_command_run_fresh_prints_fail_not_a_traceback(damaged_home):
    # Exactly what the mousereach-setup console script does.
    r = _run_fresh(damaged_home, "import sys; sys.argv = ['mousereach-setup', '--show']\n"
                                 "from mousereach.setup.wizard import cli_main\n"
                                 "sys.exit(cli_main())")
    assert r.returncode == 1
    assert "[FAIL]" in r.stdout and "line 4" in r.stdout
    assert "Traceback" not in r.stderr
    assert (damaged_home / ".mousereach" / "config.json").read_text(encoding="utf-8") == DAMAGED


def test_the_app_launcher_run_fresh_says_fail_and_opens_the_setup_screen(damaged_home):
    r = _run_fresh(damaged_home,
                   "import mousereach.launcher as L\n"
                   "L._launch_setup_only = lambda: print('SETUP SCREEN OPENED')\n"
                   "L.launch()")
    assert r.returncode == 1
    assert "[FAIL]" in r.stdout and "line 4" in r.stdout
    assert "SETUP SCREEN OPENED" in r.stdout
    assert "Traceback" not in r.stderr


def test_the_setup_panel_reads_the_damage_without_importing_the_settings(damaged_home):
    # What the panel calls to fill its fields: it raises the same message.
    r = _run_fresh(damaged_home,
                   "from mousereach.setup.wizard import load_config\n"
                   "from mousereach.settings_file import DamagedSettingsFile\n"
                   "try:\n    load_config()\nexcept DamagedSettingsFile as e:\n"
                   "    print('PANEL SAYS', e)\n")
    assert r.returncode == 0, r.stderr
    assert "PANEL SAYS" in r.stdout and "line 4" in r.stdout


# --- A file that is not UTF-8 text is damage, said as such, never another error.

NOT_UTF8 = b'{"nas_root": "/shared/caf\xe9"}'


def test_a_user_file_that_is_not_utf8_is_refused_with_the_byte_position(user_file):
    user_file.write_bytes(NOT_UTF8)
    with pytest.raises(config.DamagedSettingsFile) as err:
        config._load_config()
    said = str(err.value)
    assert str(user_file) in said and "UTF-8" in said and "byte number 26" in said
    said.encode("ascii")


@pytest.fixture
def lab_file(tmp_path, monkeypatch, user_file):
    storage = tmp_path / "storage"
    storage.mkdir()
    user_file.write_text(json.dumps({"nas_root": str(storage)}), encoding="utf-8")
    monkeypatch.setattr(config, "_LAB_CONFIG_REPORTED", set())
    f = storage / config.LAB_CONFIG_NAME
    f.write_bytes(NOT_UTF8)
    return f


def test_a_lab_file_that_is_not_utf8_reads_as_unset_for_a_lenient_setting(lab_file):
    assert config.load_lab_config() == {}
    assert config.lab_setting("some_convenience", default="d") == "d"


def test_a_lab_file_that_is_not_utf8_stops_a_strict_setting(lab_file):
    with pytest.raises(config.DamagedSettingsFile) as err:
        config.lab_setting("hand_scores", strict=True)
    assert "UTF-8" in str(err.value)
