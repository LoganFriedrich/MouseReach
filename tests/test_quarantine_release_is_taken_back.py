"""A file released from quarantine is taken back in, and an unreadable name is
never released.

WHY: a quarantined file keeps its watcher row (state 'quarantined'), and both
intake scans skipped every file that already had a row. A file released under
its OLD name -- the Quarantine tab pre-fills it, and a name can become valid
when a validator rule is corrected -- sat in the intake folder for ever, after
the panel had said "the watcher will pick it up". And the panel moved any
name at all, so a still-invalid name was only quarantined again.

tmp_path only: the watcher database, the intake and quarantine folders are
built here; no machine configuration is read.
"""
import json
from types import SimpleNamespace

import pytest

from mousereach.watcher.db import WatcherDB
from mousereach.watcher.state import WatcherStateManager

SINGLE = "20240102_ABC9901_P1.mp4"
COLLAGE = ("20240102_ABC9901,ABC9902,ABC9903,ABC9904,"
           "ABC9905,ABC9906,ABC9907,ABC9908_P1.mkv")
BAD_SINGLE = "20240102_ABC9901.mp4"          # no tray -> invalid


@pytest.fixture
def env(tmp_path):
    intake_single = tmp_path / "Unanalyzed" / "Single_Animal"
    intake_multi = tmp_path / "Unanalyzed" / "Multi-Animal"
    quarantine = tmp_path / "Processing" / "Quarantine"
    for d in (intake_single, intake_multi, quarantine):
        d.mkdir(parents=True)
    db = WatcherDB(db_path=tmp_path / "w.db")
    config = SimpleNamespace(stability_wait_seconds=0, max_retries=3,
                             get_quarantine_dir=lambda: quarantine)
    state = WatcherStateManager(db, config)
    return SimpleNamespace(db=db, state=state, single=intake_single,
                           multi=intake_multi, quarantine=quarantine)


def _scan_singles(env, times=3):
    got = []
    for _ in range(times):        # the first sighting never counts as stable
        got += env.state.discover_new_singles(env.single)
    return got


# --- the watcher takes a released file back in --------------------------------

def test_single_released_under_its_old_name_goes_on(env):
    from mousereach.config import get_video_id
    vid = get_video_id(SINGLE)
    env.db.register_video(video_id=vid, source_path=str(env.single / SINGLE))
    env.db.mark_quarantined(vid, reason="held by an old rule", is_collage=False)
    (env.single / SINGLE).write_bytes(b"video")       # a person released it

    assert vid in _scan_singles(env)
    row = env.db.get_video(vid)
    assert row["state"] == "validated"
    assert not row.get("error_message")
    assert row["animal_id"] and row["tray_type"]
    assert (env.single / SINGLE).exists()             # left for the GPU nodes


def test_single_released_with_a_still_invalid_name_is_quarantined_again(env):
    from mousereach.config import get_video_id
    vid = get_video_id(BAD_SINGLE)
    env.db.register_video(video_id=vid, source_path=str(env.single / BAD_SINGLE))
    env.db.mark_quarantined(vid, reason="no tray", is_collage=False)
    (env.single / BAD_SINGLE).write_bytes(b"video")

    assert vid not in _scan_singles(env)
    assert not (env.single / BAD_SINGLE).exists()
    assert (env.quarantine / BAD_SINGLE).exists()
    assert (env.quarantine / (BAD_SINGLE + ".quarantine.json")).exists()
    assert env.db.get_video(vid)["state"] == "quarantined"


def test_collage_released_under_its_old_name_goes_on(env):
    env.db.register_collage(filename=COLLAGE, source_path=str(env.multi / COLLAGE))
    env.db.mark_quarantined(COLLAGE, reason="held by an old rule", is_collage=True)
    (env.multi / COLLAGE).write_bytes(b"video")

    assert COLLAGE in env.state.discover_new_collages(env.multi)
    row = env.db.get_collage(COLLAGE)
    assert row["state"] == "validated"
    assert not row.get("validation_error")
    assert row["tray_suffix"] == "P1" and row["animal_ids"].count(",") == 7


# --- the Quarantine tab refuses a name the intake cannot read -----------------

qw = pytest.importorskip("mousereach.watcher.quarantine_widget")


def _panel(env, old_name, typed_name, dest="single"):
    """A stand-in Quarantine tab with one row selected (the widget methods are
    called unbound, as the other widget tests in this suite do)."""
    (env.quarantine / old_name).write_bytes(b"video")
    (env.quarantine / (old_name + ".quarantine.json")).write_text(
        json.dumps({"error_message": "bad name"}))
    rec = {"metadata_file": str(env.quarantine / (old_name + ".quarantine.json"))}
    me = SimpleNamespace(
        _records=[rec],
        _table=SimpleNamespace(currentRow=lambda: 0,
                               item=lambda r, c: SimpleNamespace(text=lambda: typed_name)),
        _dest=SimpleNamespace(currentData=lambda: dest),
        _quarantine_dir=lambda: env.quarantine,
        _dest_dir=lambda: env.single if dest == "single" else env.multi,
        _refresh=lambda: None,
    )
    me._filename = lambda r: qw.QuarantineWidget._filename(me, r)
    return me


def test_panel_refuses_an_invalid_new_name_and_moves_nothing(env, monkeypatch):
    errors, infos = [], []
    monkeypatch.setattr(qw, "show_error", lambda m, *a, **k: errors.append(m))
    monkeypatch.setattr(qw, "show_info", lambda m, *a, **k: infos.append(m))
    me = _panel(env, "garbled.mp4", BAD_SINGLE)

    qw.QuarantineWidget._release_selected(me)

    assert (env.quarantine / "garbled.mp4").exists()
    assert (env.quarantine / "garbled.mp4.quarantine.json").exists()
    assert not (env.single / BAD_SINGLE).exists()
    assert errors and not infos


def test_panel_checks_the_chosen_destination_rules(env, monkeypatch):
    # A valid single name is not a valid collage name.
    errors = []
    monkeypatch.setattr(qw, "show_error", lambda m, *a, **k: errors.append(m))
    monkeypatch.setattr(qw, "show_info", lambda *a, **k: None)
    me = _panel(env, "garbled.mp4", SINGLE, dest="multi")

    qw.QuarantineWidget._release_selected(me)

    assert (env.quarantine / "garbled.mp4").exists()
    assert not (env.multi / SINGLE).exists()
    assert errors


def test_panel_releases_a_valid_name_and_drops_the_note(env, monkeypatch):
    monkeypatch.setattr(qw, "show_error", lambda m, *a, **k: pytest.fail(m))
    monkeypatch.setattr(qw, "show_info", lambda *a, **k: None)
    me = _panel(env, "garbled.mp4", SINGLE)

    qw.QuarantineWidget._release_selected(me)

    assert (env.single / SINGLE).exists()
    assert not (env.quarantine / "garbled.mp4").exists()
    assert not (env.quarantine / "garbled.mp4.quarantine.json").exists()


def test_panel_allows_the_unchanged_name_when_it_is_valid(env, monkeypatch):
    monkeypatch.setattr(qw, "show_error", lambda m, *a, **k: pytest.fail(m))
    monkeypatch.setattr(qw, "show_info", lambda *a, **k: None)
    me = _panel(env, SINGLE, SINGLE)

    qw.QuarantineWidget._release_selected(me)

    assert (env.single / SINGLE).exists()
