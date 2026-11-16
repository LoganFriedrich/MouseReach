"""The cropper imports on a machine with no MouseReach config.

Its default archive folder used to be built at import time
(Paths.ANALYZED_OUTPUT / "Multi-Animal"); with no config that is None / str, so
importing the cropper -- and mousereach.video_prep.core, and every test module
that imports either -- crashed with a TypeError that said nothing about setup.
The default folders are now looked up when a function needs one: the module
always imports, a configured machine gets exactly the folders it got before,
and an unconfigured one that relies on a default is told to run
mousereach-setup.

Each check runs in a fresh interpreter whose home folder is an empty temporary
folder, because the config is read once, when mousereach.config is imported.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import mousereach

SRC = str(Path(mousereach.__file__).resolve().parents[1])


def _isolated_env(home: Path) -> dict:
    """The current environment, with the home folder pointed at `home` and every
    MouseReach / lab setting removed, so nothing configures the child."""
    (home / "AppData" / "Local").mkdir(parents=True, exist_ok=True)
    (home / "AppData" / "Roaming").mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(("MOUSEREACH", "MOUSEDB", "MOUSEBRAIN"))
           and k.upper() != "CONNECTOME_ROOT"}
    env.update(USERPROFILE=str(home), HOME=str(home),
               APPDATA=str(home / "AppData" / "Roaming"),
               LOCALAPPDATA=str(home / "AppData" / "Local"),
               PYTHONPATH=SRC, PYTHONDONTWRITEBYTECODE="1")
    return env


def _run(code: str, home: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=_isolated_env(home), timeout=300)


def test_the_cropper_imports_with_no_config(tmp_path):
    out = _run("import mousereach.video_prep.core.cropper\n"
               "import mousereach.video_prep.core\n"
               "print('imported')\n", tmp_path)
    assert out.returncode == 0, "importing the cropper without a config failed:\n" + out.stderr
    assert out.stdout.strip() == "imported"


def test_a_default_folder_without_config_names_the_setup_command(tmp_path):
    out = _run(
        "from mousereach.config import ConfigurationError\n"
        "from mousereach.video_prep.core.cropper import archive_collages\n"
        "try:\n"
        "    archive_collages(verbose=False)\n"
        "except ConfigurationError as e:\n"
        "    print('CONFIG:' + str(e).replace(chr(10), ' '))\n", tmp_path)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("CONFIG:"), out.stdout + out.stderr
    assert "mousereach-setup" in out.stdout


def test_a_configured_machine_gets_the_same_default_folders(tmp_path):
    home = tmp_path / "home"
    nas = tmp_path / "shared"
    proc = tmp_path / "local"
    (home / ".mousereach").mkdir(parents=True)
    (home / ".mousereach" / "config.json").write_text(
        json.dumps({"nas_root": str(nas), "processing_root": str(proc)}))
    out = _run(
        "import json\n"
        "from mousereach.video_prep.core import cropper\n"
        "print(json.dumps({w: str(cropper.default_dir(w)) for w in\n"
        "    ('multi_animal', 'single_animal', 'dlc_queue', 'archive')}))\n", home)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got == {
        "multi_animal": str(nas / "Unanalyzed" / "Multi-Animal"),
        "single_animal": str(nas / "Unanalyzed" / "Single_Animal"),
        "dlc_queue": str(proc / "DLC_Queue"),
        "archive": str(nas / "Analyzed" / "Multi-Animal"),
    }
