"""The collage intake scan makes no per-file request for collages on record.

WHY: every GPU node scans the shared intake folder every poll, and that folder
keeps every collage ever filmed (well over a thousand). The scan asked the
share about each file twice -- Path.is_file() on the listed name, then
is_file() again on the row's recorded path -- so it cost two network round
trips per collage. On a healthy link that was the whole ~10 s scan. When one
node's link to the share slowed to seconds per request, the same scan took
over two hours and the node stopped posing. The listing already says which
entries are files, and a recorded path that IS the listed path needs no check.

The guard counts stat calls on the intake folder's files during one scan: with
N collages on record it must not grow with N.

tmp_path only: the watcher database and the intake folder are built here; no
machine configuration is read.
"""
import os
import pathlib
from types import SimpleNamespace

import pytest

from mousereach.watcher.db import WatcherDB
from mousereach.watcher.state import WatcherStateManager

N = 60


def _name(i: int) -> str:
    a = [f"ABC{i:02d}{j:02d}" for j in range(1, 9)]
    return f"20240102_{','.join(a)}_P1.mkv"


@pytest.fixture
def env(tmp_path):
    multi = tmp_path / "Unanalyzed" / "Multi-Animal"
    quarantine = tmp_path / "Processing" / "Quarantine"
    multi.mkdir(parents=True)
    quarantine.mkdir(parents=True)
    db = WatcherDB(db_path=tmp_path / "w.db")
    config = SimpleNamespace(stability_wait_seconds=0, max_retries=3,
                             get_quarantine_dir=lambda: quarantine)
    state = WatcherStateManager(db, config)
    return SimpleNamespace(db=db, state=state, multi=multi)


def _count_stats_under(monkeypatch, folder):
    """Count os.stat / os.lstat calls on paths inside ``folder``."""
    root = os.path.normcase(str(folder))
    calls = []

    def wrap(orig):
        def f(p, *a, **k):
            try:
                s = os.path.normcase(os.fspath(p))
            except TypeError:
                s = ""
            if isinstance(s, str) and s.startswith(root + os.sep):
                calls.append(s)
            return orig(p, *a, **k)
        return f

    for name in ("stat", "lstat"):
        w = wrap(getattr(os, name))
        monkeypatch.setattr(os, name, w)
        acc = getattr(pathlib, "_NormalAccessor", None)   # Python < 3.11
        if acc is not None and hasattr(acc, name):
            monkeypatch.setattr(acc, name, staticmethod(w))
    return calls


def test_known_collages_cost_no_request_per_file(env, monkeypatch):
    for i in range(N):
        p = env.multi / _name(i)
        p.write_bytes(b"video")
        env.db.register_collage(filename=p.name, source_path=str(p))
    calls = _count_stats_under(monkeypatch, env.multi)

    assert env.state.discover_new_collages(env.multi) == []
    assert calls == [], f"{len(calls)} per-file requests for {N} known collages"


def test_a_wrong_recorded_path_is_still_corrected(env):
    p = env.multi / _name(0)
    p.write_bytes(b"video")
    env.db.register_collage(filename=p.name, source_path="SOME-HOST")
    env.state.discover_new_collages(env.multi)
    assert env.db.get_collage(p.name)["source_path"] == str(p)


def test_a_recorded_path_elsewhere_that_exists_is_kept(env, tmp_path):
    p = env.multi / _name(0)
    p.write_bytes(b"video")
    elsewhere = tmp_path / "elsewhere" / p.name
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"video")
    env.db.register_collage(filename=p.name, source_path=str(elsewhere))
    env.state.discover_new_collages(env.multi)
    assert env.db.get_collage(p.name)["source_path"] == str(elsewhere)


def test_a_folder_in_the_intake_is_not_a_collage(env):
    (env.multi / _name(1)).mkdir()                 # a folder with a video name
    (env.multi / _name(2)).write_bytes(b"video")
    got = env.state.discover_new_collages(env.multi)
    assert got == [_name(2)]
