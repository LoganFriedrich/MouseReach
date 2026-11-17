"""A lab settings file that cannot be read holds the video; it never switches
the hand-score check off.

WHY: the hand-score settings usually live only in the lab's shared settings
file on the storage drive. Read leniently, a bad edit of that file -- or the
drive dropping for a moment -- read as "no hand-scores file configured", and
every video went on to filing without being compared with the bench. The same
lenient read made the filing rules fall back to a folder no lookup searches.
"""
import json

import pytest

import mousereach.config as config
import mousereach.watcher.hand_scores as hs
import mousereach.watcher.route_cli as rc
from mousereach.watcher.db import WatcherDB

# conftest replaces hand_scores.settings and config.filing_rules for every
# test; these are the real ones, taken at import.
REAL_SETTINGS = hs.settings
REAL_FILING_RULES = config.__dict__["filing_rules"]

VID = "20250709_ABC0207_P1"


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A machine whose own config.json has only nas_root, and whose
    hand-score settings come from the lab file on the storage drive."""
    storage = tmp_path / "storage"
    storage.mkdir()
    user_file = tmp_path / "home" / ".mousereach" / "config.json"
    user_file.parent.mkdir(parents=True)
    user_file.write_text(json.dumps({"nas_root": str(storage)}), encoding="utf-8")
    monkeypatch.setattr(config, "config_file_path", lambda: user_file)
    monkeypatch.setattr(config, "_LAB_CONFIG_REPORTED", set())
    monkeypatch.setattr(config, "_LAST_GOOD_CONFIG", {}, raising=False)
    monkeypatch.setattr(hs, "settings", REAL_SETTINGS)
    monkeypatch.setattr(config.Paths, "NAS_ROOT", storage, raising=False)

    scores = tmp_path / "hand.csv"
    scores.write_text("video_key,pellet,score\n%s,1,2\n" % VID, encoding="utf-8")
    scores.with_suffix(".json").write_text(json.dumps({"last_successful_import": {}}),
                                           encoding="utf-8")
    hs._CACHE.update(key=None)
    lab_file = storage / config.LAB_CONFIG_NAME
    lab_file.write_text(json.dumps({"hand_scores": {"path": str(scores),
                                                    "required_projects": ["ABC"]}}),
                        encoding="utf-8")

    folder = tmp_path / "work"
    folder.mkdir()
    (folder / f"{VID}_pellet_outcomes.json").write_text(
        json.dumps({"segments": [{"segment_num": 1, "outcome": "retrieved"}]}), encoding="utf-8")
    monkeypatch.setattr(rc, "answered_segments_of", lambda v, d: set())

    class L:
        pass
    L.storage, L.file, L.user_file, L.folder = storage, lab_file, user_file, folder
    return L


def _orch(tmp_path):
    import mousereach.watcher.orchestrator as o
    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = WatcherDB(tmp_path / "local.db")
    orch.db.register_video(video_id=VID, source_path="x")
    orch.db.force_state(VID, "processing", reason="test fixture")
    orch.hostname = "HOST"
    return orch


def _attention(lab):
    return json.loads((lab.storage / "attention" / "HOST_hand_scores.json").read_text())


def test_the_check_runs_from_the_lab_file(lab, tmp_path):
    assert hs.check(VID, lab.folder).status == hs.AGREES
    assert _orch(tmp_path)._hand_score_precheck(VID, lab.folder) is True


def test_a_lab_file_that_does_not_parse_holds_the_video_in_red(lab, tmp_path):
    lab.file.write_text('{"hand_scores": {"path": "x",}', encoding="utf-8")
    orch = _orch(tmp_path)
    assert orch._hand_score_precheck(VID, lab.folder) is False     # held, not passed
    assert orch.db.get_video(VID)["state"] == "processing"
    item = _attention(lab)["items"][0]
    assert item["kind"] == "hand_score_check_failed" and item["needs_person"] is True
    assert config.LAB_CONFIG_NAME in item["says"]
    item["says"].encode("ascii")
    # nothing was recorded as if the session had been checked
    assert "hand_scores" not in json.loads(
        (lab.folder / f"{VID}_pellet_outcomes.json").read_text())


def test_a_storage_drive_that_cannot_be_reached_holds_the_video(lab, tmp_path):
    lab.user_file.write_text(json.dumps({"nas_root": str(tmp_path / "gone")}),
                             encoding="utf-8")
    with pytest.raises(Exception):
        hs.check(VID, lab.folder)
    assert _orch(tmp_path)._hand_score_precheck(VID, lab.folder) is False


def test_a_lab_file_that_is_really_absent_still_means_not_configured(lab, tmp_path):
    lab.file.unlink()
    assert hs.check(VID, lab.folder).status == hs.NOT_CHECKED
    assert _orch(tmp_path)._hand_score_precheck(VID, lab.folder) is True


def test_a_held_video_goes_on_by_itself_once_the_file_is_fixed(lab, tmp_path):
    good = lab.file.read_text(encoding="utf-8")
    lab.file.write_text("{broken", encoding="utf-8")
    orch = _orch(tmp_path)
    assert orch._hand_score_precheck(VID, lab.folder) is False
    assert orch._waiting_for_hand_scores(VID)        # not re-checked every cycle
    lab.file.write_text(good, encoding="utf-8")
    assert not orch._waiting_for_hand_scores(VID)    # picked again
    assert orch._hand_score_precheck(VID, lab.folder) is True
    assert _attention(lab)["items"] == []


GRANT = {"ABC": {"folder": "GrantA/ABC", "cohort_folder": "{CODE}_{NN}"}}


def test_filing_keeps_its_last_good_rules_when_the_lab_file_breaks(lab, monkeypatch):
    monkeypatch.setattr(config, "filing_rules", REAL_FILING_RULES)
    monkeypatch.setitem(config._FILING_CACHE, "rules", None)
    monkeypatch.setitem(config._FILING_CACHE, "error", None)
    lab.file.write_text(json.dumps({"filing": GRANT}), encoding="utf-8")
    assert config.AnimalID.get_project_and_cohort("ABC0207") == ("GrantA/ABC", "ABC_02")
    lab.file.write_text("{broken", encoding="utf-8")
    monkeypatch.setitem(config._FILING_CACHE, "at", -1e9)          # due for a re-read
    assert config.AnimalID.get_project_and_cohort("ABC0207") == ("GrantA/ABC", "ABC_02")


def test_filing_with_no_good_rules_yet_stops_instead_of_using_a_fallback_folder(lab, monkeypatch):
    monkeypatch.setattr(config, "filing_rules", REAL_FILING_RULES)
    monkeypatch.setitem(config._FILING_CACHE, "rules", None)
    monkeypatch.setitem(config._FILING_CACHE, "error", None)
    lab.file.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        config.AnimalID.get_project_and_cohort("ABC0207")


def test_filing_with_no_good_rules_reads_a_failing_file_once_a_minute(lab, monkeypatch):
    """WHY: the work-priority sort asks for the rules for every item in every
    cycle; re-reading a damaged file (or a share that hangs) each time made a
    watcher started in that state crawl."""
    monkeypatch.setattr(config, "filing_rules", REAL_FILING_RULES)
    monkeypatch.setitem(config._FILING_CACHE, "rules", None)
    monkeypatch.setitem(config._FILING_CACHE, "error", None)
    monkeypatch.setitem(config._FILING_CACHE, "at", -1e9)
    lab.file.write_text("{broken", encoding="utf-8")
    reads = []
    real = config.load_lab_config
    monkeypatch.setattr(config, "load_lab_config",
                        lambda *a, **k: (reads.append(1), real(*a, **k))[1])
    raised = []
    for _ in range(5):
        with pytest.raises(ValueError) as e:
            config.AnimalID.get_project_and_cohort("ABC0207")
        raised.append(e.value)
    assert len(reads) == 1
    # A fresh exception each time, all saying the same: one stored instance
    # re-raised would grow its traceback on every call.
    assert len({id(x) for x in raised}) == 5
    assert len({str(x) for x in raised}) == 1
    assert all(isinstance(x, config.DamagedSettingsFile) for x in raised)
    lab.file.write_text(json.dumps({"filing": GRANT}), encoding="utf-8")
    monkeypatch.setitem(config._FILING_CACHE, "at", -1e9)          # the minute is up
    assert config.AnimalID.get_project_and_cohort("ABC0207") == ("GrantA/ABC", "ABC_02")


def test_a_video_held_while_the_lab_file_was_damaged_goes_on_in_a_lab_with_no_hand_scores(
        lab, tmp_path):
    """A lab with no hand_scores section at all: once the damaged file is
    fixed, the held video is picked again at once, not after a restart."""
    lab.file.write_text("{broken", encoding="utf-8")
    orch = _orch(tmp_path)
    assert orch._hand_score_precheck(VID, lab.folder) is False
    assert orch._waiting_for_hand_scores(VID)
    lab.file.write_text(json.dumps({"some_other_setting": 1}), encoding="utf-8")
    assert not orch._waiting_for_hand_scores(VID)    # picked again
    assert orch._hand_score_precheck(VID, lab.folder) is True
    assert _attention(lab)["items"] == []


def test_an_empty_lab_file_holds_the_video_instead_of_reading_as_no_settings(lab, tmp_path):
    """WHY: the lab file is only ever written whole, by a person who put
    settings in it. An empty one is a file caught mid-save by an editor
    (truncated, then written) or blanked by accident; read as "no settings",
    that moment switched the comparison off and the video went on unchecked."""
    lab.file.write_text("", encoding="utf-8")
    with pytest.raises(Exception):          # never "no settings, not checked"
        hs.check(VID, lab.folder)
    orch = _orch(tmp_path)
    assert orch._hand_score_precheck(VID, lab.folder) is False     # held, not passed
    item = _attention(lab)["items"][0]
    assert item["kind"] == "hand_score_check_failed" and item["needs_person"] is True
    assert "empty" in item["says"]
