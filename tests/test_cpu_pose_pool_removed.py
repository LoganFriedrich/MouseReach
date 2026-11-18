"""The CPU pose pool (``mousereach-cpu-pose``) is gone, and what it left behind
is harmless.

WHY: posing on a machine's CPUs was removed on 2026-11-18 -- it did not work
well enough to keep -- so posing is done by graphics-card machines only. These
tests hold that removal in place and prove the pieces a machine that ran the
pool may still carry do no harm:

  * no ``mousereach-cpu-pose`` command and no ``cpu_pose_pool`` module;
  * a ``cpu_pose`` section left in a machine's config.json is ignored: the
    settings load exactly as without it;
  * a claim left in ``Unanalyzed/Single_Animal/.inflight/<host>-cpu/`` by the
    pool, whose status report is gone, goes back to the shared singles folder
    by the ordinary stopped-holder rule (tests/test_singles_intake_claim.py);
  * an old ``<host>-cpu`` status report still prints as one line
    (tests/test_node_status_says_what_it_is_doing.py).
"""
import importlib.util
import json
import re
from pathlib import Path

import mousereach
from mousereach import config as mr_config

REPO = Path(__file__).resolve().parent.parent


def _scripts_section() -> str:
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r"^\[project\.scripts\]\s*$(.*?)(?=^\[)", text, re.M | re.S)
    assert m, "pyproject.toml has no [project.scripts] section"
    return m.group(1)


def test_the_tests_run_against_this_checkout():
    """Every check below reads this tree; an installed copy elsewhere would
    let a removed module pass as present (or the reverse)."""
    assert Path(mousereach.__file__).resolve().is_relative_to(REPO / "src")


def test_there_is_no_cpu_pose_command():
    section = _scripts_section()
    assert "mousereach-watch =" in section          # the section was read at all
    assert "mousereach-cpu-pose" not in section
    assert "cpu_pose_pool" not in section


def test_there_is_no_cpu_pose_module():
    assert importlib.util.find_spec("mousereach.watcher.cpu_pose_pool") is None
    assert not (REPO / "src" / "mousereach" / "watcher" / "cpu_pose_pool.py").exists()


def _write_config(home: Path, data: dict) -> Path:
    path = home / ".mousereach" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_a_leftover_cpu_pose_section_is_ignored(tmp_path, monkeypatch):
    base = {
        "processing_root": str(tmp_path / "node"),
        "nas_root": str(tmp_path / "nas"),
        "watcher": {"enabled": True, "mode": "processing_server",
                    "max_local_pending": 2},
    }
    with_old = dict(base, cpu_pose={"dlc_config_path": str(tmp_path / "c.yaml"),
                                    "workers": 12, "threads": 3})

    loaded = {}
    for name, data in (("plain", base), ("with_old", with_old)):
        home = tmp_path / name
        _write_config(home, data)
        monkeypatch.setattr(mr_config, "config_file_path",
                            lambda home=home: home / ".mousereach" / "config.json")
        raw = mr_config._load_config()
        watcher = mr_config.WatcherConfig.load()
        loaded[name] = (watcher.to_dict(), {k: v for k, v in raw.items()
                                            if k != "cpu_pose"})

    assert loaded["with_old"] == loaded["plain"]
    assert loaded["with_old"][0]["mode"] == "processing_server"
