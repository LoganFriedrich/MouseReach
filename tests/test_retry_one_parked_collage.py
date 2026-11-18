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
