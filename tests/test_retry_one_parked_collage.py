"""One collage that used up its retries can be tried again on its own.

The only remedy for a collage parked out of retries used to be raising
watcher.max_retries for the whole machine and restarting the watcher, which
also gave every failed video on that machine another try. Now
``mousereach-watch-reprocess --collage NAME`` (and the "Retry a collage..."
button in Watcher Control) gives only that collage its retries back; the
collage scan retries it at its next pass, while its file is in the intake
folder. tmp_path only (the fixtures of test_collage_crop_failures_are_visible).
"""
import sys
from types import SimpleNamespace

import pytest

from mousereach.watcher import cli
from mousereach.watcher.state import WatcherStateManager

from test_collage_crop_failures_are_visible import (  # noqa: F401
    COLLAGE, FakeCoordinator, _attention, _out_of_retries, _scanning_node, env)

VID = "20250101_TST0101_P1"


def _reprocess(monkeypatch, env, *args):
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: env.db.db_path)
    monkeypatch.setattr(sys, "argv", ["mousereach-watch-reprocess", *args])
    with pytest.raises(SystemExit) as e:
        cli.main_reprocess()
    return e.value.code


def _scan(env):
    sm = WatcherStateManager(env.db, SimpleNamespace(max_retries=3))
    sm.discover_new_collages(env.multi_animal_source)
    return sm


def _a_failed_video(env):
    env.db.register_video(video_id=VID, source_path="x")
    env.db.force_state(VID, "failed", reason="test setup")
    env.db.set_fields(VID, error_count=3, error_message="pose crashed")


def test_one_parked_collage_is_tried_again_and_nothing_else_is_touched(env, monkeypatch):
    _out_of_retries(env)
    _a_failed_video(env)
    assert COLLAGE in _scan(env).parked_collages          # parked before

    assert _reprocess(monkeypatch, env, "--collage", COLLAGE) == 0

    sm = _scan(env)
    row = env.db.get_collage(COLLAGE)
    assert row["state"] == "validated", "the next scan must try the collage again"
    assert int(row["retry_count"]) == 1
    assert COLLAGE not in sm.parked_collages
    # Only that collage: a failed video on the machine keeps its count.
    video = env.db.get_video(VID)
    assert video["state"] == "failed" and int(video["error_count"]) == 3


def test_a_collage_that_is_not_failed_is_not_changed(env, monkeypatch, capsys):
    from test_collage_crop_failures_are_visible import _collage
    _collage(env, "cropped")
    assert _reprocess(monkeypatch, env, "--collage", COLLAGE) == 1
    assert env.db.get_collage(COLLAGE)["state"] == "cropped"
    assert "not failed" in capsys.readouterr().out


def test_the_red_panel_names_the_one_collage_retry(env, monkeypatch):
    _out_of_retries(env)
    env.o.coordinator = FakeCoordinator(claim_state=None)
    o = _scanning_node(env, monkeypatch)
    o._scan_phase()
    says = _attention(env, "collages")[0]["says"]
    assert "--collage" in says and "Retry a collage" in says
    assert "raise max_retries" not in says, (
        "the remedy must not be a machine-wide retry of every failed video")


# --- Watcher Control "Retry a collage..." button -------------------------------

def test_failed_collage_choices_lists_only_failed_collages_with_their_error(env):
    cw = pytest.importorskip("mousereach.watcher.control_widget")
    assert cw.failed_collage_choices(env.db) == []        # nothing failed yet
    _out_of_retries(env)
    choices = cw.failed_collage_choices(env.db)
    assert [name for name, _ in choices] == [COLLAGE]
    label = choices[0][1]
    assert "3 time(s)" in label and "crop failed: unreadable stream" in label


def test_failed_collage_choices_never_raises():
    cw = pytest.importorskip("mousereach.watcher.control_widget")

    class Broken:
        def get_collages_in_state(self, state):
            raise RuntimeError("database unreadable")
    assert cw.failed_collage_choices(Broken()) == []


def _press_retry(cw, env, monkeypatch, pick_ok=True):
    """Call the button's handler unbound, with the dialogs replaced."""
    said = []
    monkeypatch.setattr(cw, "show_info", lambda m: said.append(("info", m)))
    monkeypatch.setattr(cw, "show_error", lambda m: said.append(("error", m)))
    monkeypatch.setattr(cw.QMessageBox, "information",
                        lambda *a, **k: said.append(("box", a[-1])))
    asked = []

    def get_item(parent, title, label, items, *a, **k):
        asked.append(list(items))
        return items[0], pick_ok
    monkeypatch.setattr(cw.QInputDialog, "getItem", staticmethod(get_item))
    refreshed = []
    me = SimpleNamespace(_db=lambda: env.db, _refresh=lambda: refreshed.append(1))
    cw.WatcherControlWidget._retry_collage(me)
    return said, asked, refreshed


def test_retry_button_gives_the_picked_collage_its_retries_back(env, monkeypatch):
    cw = pytest.importorskip("mousereach.watcher.control_widget")
    _out_of_retries(env)
    said, asked, refreshed = _press_retry(cw, env, monkeypatch)
    assert len(asked) == 1 and COLLAGE in asked[0][0]
    assert int(env.db.get_collage(COLLAGE)["retry_count"]) == 0
    assert env.db.get_collage(COLLAGE)["state"] == "failed"   # the scan retries it
    assert said and said[0][0] == "info" and refreshed


def test_retry_button_declined_dialog_changes_nothing(env, monkeypatch):
    cw = pytest.importorskip("mousereach.watcher.control_widget")
    _out_of_retries(env)
    _press_retry(cw, env, monkeypatch, pick_ok=False)
    assert int(env.db.get_collage(COLLAGE)["retry_count"]) == 3


def test_retry_button_with_no_failed_collage_says_so(env, monkeypatch):
    cw = pytest.importorskip("mousereach.watcher.control_widget")
    said, asked, _ = _press_retry(cw, env, monkeypatch)
    assert not asked
    assert said and said[0][0] == "box"
