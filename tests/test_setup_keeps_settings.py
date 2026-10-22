"""Setup never loses a setting, and an unset pipeline root stops rather than guesses.

WHY (2026-10-22): two defects compounded.
  * Saving from mousereach-setup or the in-app Setup panel rewrote
    ~/.mousereach/config.json with three keys only (processing_root, nas_drive,
    watcher). Every other setting was silently deleted -- the shared pipeline
    root (nas_root), the nightly backup's folders, the shared database path.
  * With nas_root gone, the pipeline root was GUESSED as <nas_drive>/! DLC
    Output, the layout retired in 2026. Every shared folder -- intake, posed
    videos, review queues, Analyzed -- then pointed at the old place, and the
    review tools and intake commands (which, unlike the watcher, make no check)
    would have read and written there as if nothing were wrong.
An unset nas_root now leaves every shared path unset, so each tool says
"not configured -- run mousereach-setup" instead of working in the wrong place.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import mousereach
import mousereach.setup.wizard as wizard

_PROBE = r"""
import json
from mousereach.config import Paths
print(json.dumps({k: (None if getattr(Paths, k) is None else str(getattr(Paths, k)))
                  for k in ("NAS_ROOT", "NAS_ROOT_ORIGIN", "DLC_STAGING",
                            "TRIAGE_REVIEW", "ANALYZED_OUTPUT", "SINGLE_ANIMAL_OUTPUT")}))
"""
_STRIPPED_ENV = {"HOME", "USERPROFILE", "PYTHONPATH",
                 "MOUSEREACH_NAS_DRIVE", "MOUSEREACH_PROCESSING_ROOT"}


def _paths_for(tmp_path, config):
    home = tmp_path / "home"
    (home / ".mousereach").mkdir(parents=True)
    (home / ".mousereach" / "config.json").write_text(json.dumps(config), encoding="utf-8")
    src = str(Path(mousereach.__file__).parents[1])
    env = {k: v for k, v in os.environ.items() if k.upper() not in _STRIPPED_ENV}
    env.update(USERPROFILE=str(home), HOME=str(home), PYTHONPATH=src)
    proc = subprocess.run([sys.executable, "-c", _PROBE], env=env, cwd=str(home),
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_a_drive_alone_does_not_invent_a_pipeline_root(tmp_path):
    got = _paths_for(tmp_path, {"nas_drive": str(tmp_path / "drive"),
                                "processing_root": str(tmp_path / "proc")})
    assert got["NAS_ROOT"] is None
    assert got["NAS_ROOT_ORIGIN"] == "unset"
    for k in ("DLC_STAGING", "TRIAGE_REVIEW", "ANALYZED_OUTPUT", "SINGLE_ANIMAL_OUTPUT"):
        assert got[k] is None, k


def test_a_configured_root_is_used_as_given(tmp_path):
    root = tmp_path / "share" / "Pipeline"
    got = _paths_for(tmp_path, {"nas_drive": str(tmp_path / "drive"), "nas_root": str(root)})
    assert got["NAS_ROOT"] == str(root) and got["NAS_ROOT_ORIGIN"] == "config"
    assert got["DLC_STAGING"] == str(root / "Processing" / "Posed")


def _use_config_file(monkeypatch, tmp_path, doc):
    cfg = tmp_path / ".mousereach" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(wizard, "CONFIG_DIR", cfg.parent)
    monkeypatch.setattr(wizard, "CONFIG_FILE", cfg)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    return cfg


def test_saving_keeps_every_setting_it_was_not_asked_about(tmp_path, monkeypatch):
    cfg = _use_config_file(monkeypatch, tmp_path, {
        "nas_drive": "D:/old", "nas_root": "D:/share/Pipeline",
        "processing_root": "C:/old_proc",
        "backup": {"sync_dirs": ["a", "b"]}, "central_db": "D:/share/c.db",
        "watcher": {"mode": "processing_server", "max_local_pending": 2},
    })

    wizard.save_config(nas_drive="E:/new", processing_root=Path("C:/new_proc"),
                       watcher={"poll_interval_seconds": 15})

    doc = json.loads(cfg.read_text(encoding="utf-8"))
    assert doc["nas_drive"] == "E:/new"
    assert doc["processing_root"] == str(Path("C:/new_proc"))
    assert doc["nas_root"] == "D:/share/Pipeline"             # kept
    assert doc["backup"] == {"sync_dirs": ["a", "b"]}         # kept
    assert doc["central_db"] == "D:/share/c.db"               # kept
    assert doc["watcher"] == {"mode": "processing_server", "max_local_pending": 2,
                              "poll_interval_seconds": 15}    # merged, not replaced


def test_saving_records_the_pipeline_root(tmp_path, monkeypatch):
    cfg = _use_config_file(monkeypatch, tmp_path, {"processing_root": "C:/p"})
    wizard.save_config(nas_drive=None, processing_root=Path("C:/p"),
                       nas_root=Path("D:/share/Pipeline"))
    doc = json.loads(cfg.read_text(encoding="utf-8"))
    assert doc["nas_root"] == str(Path("D:/share/Pipeline"))
    assert "nas_drive" not in doc


def test_an_unset_root_is_named_as_the_problem():
    from mousereach.watcher.cli import check_nas_root
    msg = check_nas_root(None, "unset")
    assert "nas_root" in msg and "mousereach-setup" in msg
