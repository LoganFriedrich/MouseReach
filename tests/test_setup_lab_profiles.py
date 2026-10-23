"""Where setup finds a lab's machine profiles, and what a profile may change.

WHY (2026-10-22): the lab's profiles file sat beside the code, git-ignored, so
no machine that pulls the code ever had one, and setup fell back to the shipped
template, whose drive patterns matched real machines and offered them a retired
folder layout. A profile's hostname rule was documented but never checked, and
only five watcher settings were ever taken from a profile, so a recording PC set
up from scratch lost its pause for the recording program. Built under tmp_path.
"""
import json

import pytest

from mousereach.setup import wizard

PROFILES = {"profiles": [{"name": "Role A", "match": {"hostname": "PC-A"},
                          "defaults": {"watcher": {"mode": "dlc_pc"}}}]}


@pytest.fixture
def places(tmp_path, monkeypatch):
    """No profiles file anywhere unless a test writes one."""
    user = tmp_path / "home" / "lab_profiles.json"
    module = tmp_path / "module" / "lab_profiles.json"
    monkeypatch.setattr(wizard, "USER_LAB_PROFILES_FILE", user)
    monkeypatch.setattr(wizard, "LAB_PROFILES_FILE", module)
    monkeypatch.delenv(wizard.LAB_PROFILES_ENV, raising=False)
    return tmp_path, user, module


def _write(path, name):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(json.dumps(PROFILES))
    data["profiles"][0]["name"] = name
    path.write_text(json.dumps(data), encoding="utf-8")


def test_the_template_is_never_used_as_a_lab_profile(places):
    assert wizard.LAB_PROFILES_EXAMPLE.is_file()
    assert wizard.lab_profiles_path() is None
    assert wizard._load_lab_profiles() == []


def test_the_environment_variable_then_home_then_beside_the_code(places, monkeypatch):
    tmp, user, module = places
    _write(module, "module")
    assert wizard._load_lab_profiles()[0]["name"] == "module"
    _write(user, "home")
    assert wizard._load_lab_profiles()[0]["name"] == "home"
    shared = tmp / "share" / "lab_profiles.json"
    _write(shared, "shared")
    monkeypatch.setenv(wizard.LAB_PROFILES_ENV, str(shared))
    assert wizard._load_lab_profiles()[0]["name"] == "shared"


def test_a_variable_naming_no_file_says_so_and_looks_elsewhere(places, monkeypatch, capsys):
    tmp, user, _ = places
    _write(user, "home")
    monkeypatch.setenv(wizard.LAB_PROFILES_ENV, str(tmp / "missing.json"))
    assert wizard._load_lab_profiles()[0]["name"] == "home"
    assert "not a file" in capsys.readouterr().out


def test_the_hostname_rule_is_checked():
    prof = {"match": {"hostname": "PC-A", "drives_present": ["C"]}}
    drives = {"C": True}
    assert wizard._match_profile(prof, drives, hostname="pc-a")
    assert not wizard._match_profile(prof, drives, hostname="PC-B")
    prof["match"]["hostname"] = ["PC-B", "PC-C"]
    assert wizard._match_profile(prof, drives, hostname="PC-C")
    assert not wizard._match_profile({"match": {"hostname": "PC-A", "drives_present": ["Z"]}},
                                     drives, hostname="PC-A")


def test_a_profile_fills_in_the_role_but_never_replaces_the_machines_settings():
    profile = {"mode": "dlc_pc", "also_process": False,
               "pause_while_running": ["recorder.exe"], "recording_days": "Mon-Fri",
               "dlc_config_path": "profile/config.yaml"}
    existing = {"also_process": True}
    got = wizard._carry_unprompted({"dlc_config_path": "asked/config.yaml"}, existing, profile)
    assert got["also_process"] is True                      # the machine's own value
    assert got["pause_while_running"] == ["recorder.exe"]   # filled in from the profile
    assert got["recording_days"] == "Mon-Fri"
    assert got["mode"] == "dlc_pc"
    assert got["dlc_config_path"] == "asked/config.yaml"    # asked for, so not overridden


def test_other_profile_sections_are_added_only_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(wizard, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(wizard, "CONFIG_FILE", tmp_path / "config.json")
    (tmp_path / "config.json").write_text(json.dumps({"backup": {"mine": 1}}), encoding="utf-8")
    wizard.save_config(sections={"backup": {"profile": 2}, "extra": {"x": 1}})
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["backup"] == {"mine": 1}
    assert saved["extra"] == {"x": 1}


def test_the_template_holds_placeholders_not_a_real_layout():
    text = wizard.LAB_PROFILES_EXAMPLE.read_text(encoding="utf-8")
    data = json.loads(text)
    assert "DLC Output" not in text
    for p in data["profiles"]:
        assert p["match"]["hostname"].startswith("<")
        assert p["defaults"]["nas_root"].startswith("<")
