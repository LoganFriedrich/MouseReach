"""Where a finished video is filed comes from the lab's filing rules, and every
lookup of a filed video finds it there, however many levels the folder has.

WHY: the project folders were written into the code (one map of codes to
folder names, per-cohort folders for two codes only). A second project of the
same grant therefore filed beside the grant instead of inside it, with no
folder per cohort, and the layout could not be changed without a code change.
The rules now live in the shared lab settings file ("filing"). Lookups that
assumed Analyzed/<project>/<cohort> was exactly two levels deep would have
lost every video filed under a two-level project folder (grant/project), so
they ask the rule instead.
"""
import json
import os

import pytest

import mousereach.config as config
from mousereach.archive.core import filed_folder

# conftest replaces config.filing_rules for every test; this is the real one.
REAL_FILING_RULES = config.__dict__["filing_rules"]

GRANT_RULES = {
    "PROJA": {"folder": "GrantA/PROJA", "cohort_folder": "{CODE}_{NN}"},
    "PROJAB": {"folder": "GrantA/PROJAB", "cohort_folder": "{CODE}_{NN}"},
    "PROJC": {"folder": "PROJC", "cohort_folder": "{CODE}"},
}


@pytest.fixture
def rules(monkeypatch):
    def use(r):
        monkeypatch.setattr(config, "filing_rules", lambda: r)
    use(GRANT_RULES)
    return use


def test_each_project_files_under_its_own_folder_and_cohort(rules):
    get = config.AnimalID.get_project_and_cohort
    assert get("PROJA0304") == ("GrantA/PROJA", "PROJA_03")
    assert get("PROJAB0216") == ("GrantA/PROJAB", "PROJAB_02")    # a code that starts like another
    assert get("PROJC0101") == ("PROJC", "PROJC")


def test_a_project_with_no_rule_files_under_its_code_as_before(rules):
    assert config.AnimalID.get_project_and_cohort("PROJD0102") == ("PROJD", "PROJD")


def test_an_id_with_no_cohort_number_gets_one_folder_not_a_broken_name(rules):
    rules({"H": {"folder": "Old", "cohort_folder": "{CODE}_{NN}"}})
    assert config.AnimalID.get_project_and_cohort("H01") == ("Old", "H")


def test_the_filed_folder_is_the_archive_destination(rules, tmp_path):
    assert filed_folder("20260101_PROJAB0216_P1", tmp_path) == \
        tmp_path / "GrantA" / "PROJAB" / "PROJAB_02"


def test_the_rules_are_read_from_the_lab_settings(monkeypatch):
    monkeypatch.setattr(config, "filing_rules", REAL_FILING_RULES)
    monkeypatch.setitem(config._FILING_CACHE, "rules", None)
    monkeypatch.setattr(config, "load_lab_config", lambda user_config=None: {"filing": GRANT_RULES})
    assert config.AnimalID.get_project_and_cohort("PROJA0304") == ("GrantA/PROJA", "PROJA_03")


@pytest.mark.parametrize("bad", [
    {"folder": "../elsewhere", "cohort_folder": "{CODE}"},
    {"folder": "C:/elsewhere", "cohort_folder": "{CODE}"},
    {"folder": "GrantA//PROJA", "cohort_folder": "{CODE}"},
    {"folder": "GrantA", "cohort_folder": "{CODE}/{NN}"},
    {"folder": "GrantA", "cohort_folder": "{CODE}_{COHORT}"},
    {"folder": "GrantA"},
    {"folder": "GrantA", "cohort_folder": "{CODE}", "extra": 1},
    "GrantA",
])
def test_a_rule_that_could_file_somewhere_unexpected_is_refused(monkeypatch, bad):
    monkeypatch.setattr(config, "filing_rules", REAL_FILING_RULES)
    monkeypatch.setitem(config._FILING_CACHE, "rules", None)
    monkeypatch.setattr(config, "load_lab_config", lambda user_config=None: {"filing": {"PROJA": bad}})
    with pytest.raises(ValueError):
        config.filing_rules()


def test_work_order_labels_include_the_grant_and_the_project(rules):
    from mousereach.watcher.work_priority import cohort_labels, project_labels
    item = {"video_id": "20260101_PROJAB0216_P1"}
    assert {"GRANTA", "PROJAB"} <= project_labels(item)
    assert "PROJA" not in project_labels(item)
    assert "PROJAB_02" in cohort_labels(item)


def _put(p, text="{}"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_the_version_scan_indexes_videos_filed_two_levels_down(rules, tmp_path):
    from mousereach.watcher.reprocessor import ReprocessingScanner
    root = tmp_path / "Analyzed"
    deep = root / "GrantA" / "PROJA" / "PROJA_03"
    shallow = root / "PROJC" / "PROJC"
    a, c = "20260101_PROJA0304_P1", "20260101_PROJC0101_P1"
    _put(deep / f"{a}_processing_manifest.json")
    _put(deep / f"{a}_features.json")
    _put(shallow / f"{c}_processing_manifest.json")
    _put(root / "Archive" / "x" / "y" / f"{a}_features.json")   # superseded: never live
    _put(root / "GrantA" / ".tmp" / "z" / f"{c}_features.json")  # dot-folder: never live
    scanner = ReprocessingScanner.__new__(ReprocessingScanner)
    scanner.archive_dir = root
    man, man_t, feat_t = scanner._index_archive()
    assert set(man) == {a, c}
    assert set(feat_t) == {a}


def test_a_moved_copy_free_bundle_still_opens(rules, tmp_path, monkeypatch):
    """The manifest still names the old folder; the video now sits where the
    filing rule says. The review tool must find it there."""
    from mousereach.review.bundle_media import resolve_bundle_pose, resolve_bundle_video
    root = tmp_path / "Analyzed"
    monkeypatch.setattr(config.Paths, "ANALYZED_OUTPUT", root)
    stem = "20260101_PROJA0304_P1"
    home = root / "GrantA" / "PROJA" / "PROJA_03"
    _put(home / f"{stem}.mp4")
    model = _put(root / "GrantA" / "PROJA" / "DLC Model 4" / "PROJA_03" / f"{stem}DLC_x.h5")
    bundle = tmp_path / "Review" / "triage" / stem
    bundle.mkdir(parents=True)
    stale = {"video_stem": stem,
             "canonical_video_path": str(tmp_path / "Old" / "PROJA03" / f"{stem}.mp4"),
             "canonical_dlc_h5_path": str(tmp_path / "Old" / "DLC Model 4" / "PROJA03" / f"{stem}DLC_x.h5")}
    assert resolve_bundle_video(stale, bundle) == home / f"{stem}.mp4"
    assert resolve_bundle_pose(stale, bundle) == model


def test_the_route_tool_finds_outcomes_in_the_filed_folder(rules, tmp_path):
    from mousereach.watcher.route_cli import _find_outcomes
    root = tmp_path / "Analyzed"
    stem = "20260101_PROJA0304_P1"
    live = _put(root / "GrantA" / "PROJA" / "PROJA_03" / f"{stem}_pellet_outcomes.json")
    _put(root / "Archive" / "GrantA" / f"{stem}_pellet_outcomes.json")
    assert _find_outcomes(root, stem) == live


def test_a_persons_own_config_cannot_move_where_videos_are_filed(monkeypatch):
    monkeypatch.setattr(config, "filing_rules", REAL_FILING_RULES)
    monkeypatch.setitem(config._FILING_CACHE, "rules", None)
    monkeypatch.setattr(config, "_load_config", lambda: {"filing": GRANT_RULES})
    monkeypatch.setattr(config, "load_lab_config", lambda user_config=None: {})
    assert config.filing_rules() == {}
