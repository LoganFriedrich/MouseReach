"""The shared lab settings file on the storage drive.

WHY: a setting the whole lab shares (the hand-scores file, for one) used to be
read only from each person's own ~/.mousereach/config.json. A reviewer whose
own file lacked it silently got none of the feature: the review window's
"check your answer" box never appeared for them. Settings every PC and person
share now live once, in <nas_root>/mousereach_lab_config.json, found from
nas_root alone. A person's own config.json still wins, so nothing that worked
before changes.
"""
import json
import logging

import pytest

import mousereach.config as config
import mousereach.watcher.hand_scores as hs

# conftest replaces hand_scores.settings for every test (so no test reads this
# machine's real settings); this is the real one, taken at import.
REAL_SETTINGS = hs.settings

LAB_HAND = {"path": "<lab scores>", "required_projects": ["PROJECT_A"]}


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake home: writes the user's config.json and the lab file, both
    under tmp_path."""
    storage = tmp_path / "storage"
    storage.mkdir()
    user_file = tmp_path / "home" / ".mousereach" / "config.json"
    user_file.parent.mkdir(parents=True)
    monkeypatch.setattr(config, "config_file_path", lambda: user_file)
    monkeypatch.setattr(config, "_LAB_CONFIG_REPORTED", set())

    class H:
        lab = storage / config.LAB_CONFIG_NAME

        @staticmethod
        def user(**keys):
            keys.setdefault("nas_root", str(storage))
            user_file.write_text(json.dumps(keys), encoding="utf-8")

        @staticmethod
        def lab_file(data):
            H.lab.write_text(data if isinstance(data, str) else json.dumps(data),
                             encoding="utf-8")
    return H


def test_the_lab_file_is_found_from_nas_root_alone(home):
    home.user()
    assert config.lab_config_path() == home.lab


def test_the_users_own_value_wins(home):
    home.user(hand_scores={"path": "<mine>"})
    home.lab_file({"hand_scores": LAB_HAND})
    assert config.lab_setting("hand_scores") == {"path": "<mine>"}


def test_the_lab_file_is_used_when_the_user_lacks_the_key(home):
    home.user()
    home.lab_file({"hand_scores": LAB_HAND})
    assert config.lab_setting("hand_scores") == LAB_HAND


def test_neither_file_has_it_so_it_is_unset(home):
    home.user()
    home.lab_file({"something_else": 1})
    assert config.lab_setting("hand_scores") is None
    assert config.lab_setting("hand_scores", {}) == {}


def test_no_nas_root_means_no_lab_file_and_no_message(home, caplog):
    home.lab_file({"hand_scores": LAB_HAND})
    config.config_file_path().write_text("{}", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert config.lab_setting("hand_scores") is None
    assert not caplog.records


def test_an_unreadable_lab_file_leaves_it_unset_and_says_so_once(home, caplog):
    home.user()
    home.lab_file("{not json")
    with caplog.at_level(logging.WARNING):
        assert config.lab_setting("hand_scores") is None
        assert config.lab_setting("hand_scores") is None
    said = [r.getMessage() for r in caplog.records]
    assert len(said) == 1 and "could not be read" in said[0]
    said[0].encode("ascii")


def test_a_missing_lab_file_leaves_it_unset_and_says_so_once(home, caplog):
    home.user()
    with caplog.at_level(logging.WARNING):
        assert config.lab_setting("hand_scores") is None
        assert config.lab_setting("hand_scores") is None
    said = [r.getMessage() for r in caplog.records]
    assert len(said) == 1 and "No shared lab settings file" in said[0]


def test_a_lab_file_that_is_not_an_object_is_unreadable(home, caplog):
    home.user()
    home.lab_file("[1, 2]")
    with caplog.at_level(logging.WARNING):
        assert config.lab_setting("hand_scores") is None
    assert "could not be read" in caplog.records[0].getMessage()


def test_a_user_with_only_the_shared_keys_gets_hand_scores_from_the_lab_file(
        home, tmp_path, monkeypatch):
    """A reviewer's config.json with only the machine keys (no hand_scores)
    reads the lab's hand scores: settings() resolves, and the review window's
    lookup finds the score."""
    monkeypatch.setattr(hs, "settings", REAL_SETTINGS)
    scores = tmp_path / "hand.csv"
    scores.write_text("video_key,pellet,score\n20250709_ABC0207_P1,7,1\n", encoding="utf-8")
    home.user(nas_drive="<drive>", processing_root=str(tmp_path / "proc"),
              watcher={"mode": "processing_server"})
    home.lab_file({"hand_scores": {"path": str(scores), "required_projects": ["ABC"]}})
    hs._CACHE.update(key=None)
    assert hs.settings() == {"path": str(scores), "required_projects": ["ABC"]}
    assert hs.pellet_hand_score("20250709_ABC0207_P1", 7) == (1, "")
    assert hs.scores_file_stamp() == scores.stat().st_mtime


def test_settings_keeps_the_users_own_section(home, monkeypatch):
    monkeypatch.setattr(hs, "settings", REAL_SETTINGS)
    home.user(hand_scores={"path": "<mine>"})
    home.lab_file({"hand_scores": LAB_HAND})
    assert hs.settings() == {"path": "<mine>"}


def test_settings_is_empty_when_nothing_sets_it(home, monkeypatch):
    monkeypatch.setattr(hs, "settings", REAL_SETTINGS)
    home.user()
    home.lab_file({})
    assert hs.settings() == {}
