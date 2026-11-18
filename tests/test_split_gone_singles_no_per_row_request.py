"""The pick's check for shared singles another node took asks the share
nothing per video.

WHY: every pick of every GPU node runs _split_gone_singles over its
'validated' rows. A row whose video another node had taken still points into
the shared singles folder; the folder was already listed, but the check then
asked the share about that path (is_file), and reading who holds each claim
asked twice more per claimed video. On a slow link a pick did that for every
such row, every cycle. The listings answer all of it.

The guard counts stat calls on .mp4 paths during one pick-time split: with N
taken videos it must not grow with N, and the rows must be sorted as before.

tmp_path only: Paths are patched onto a tmp layout; no machine configuration.
"""
import os
import pathlib
from types import SimpleNamespace

import pytest

from mousereach.watcher import orchestrator as orch
from mousereach.watcher import single_claim
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import DLCOrchestrator

HOST = "NODE-A"
OTHER = "NODE-B"
N = 30


@pytest.fixture
def env(tmp_path, monkeypatch):
    layout = {"SINGLE_ANIMAL_OUTPUT": "share/Unanalyzed/Single_Animal",
              "PROCESSING_ROOT": "node", "DLC_QUEUE": "node/DLC_Queue",
              "PROCESSING": "node/Processing"}
    dirs = {}
    for name, sub in layout.items():
        d = tmp_path / sub
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(orch.Paths, name, d, raising=False)
        dirs[name.lower()] = d
    o = object.__new__(DLCOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "node.db")
    o.hostname = HOST
    return SimpleNamespace(o=o, door=dirs["single_animal_output"],
                           queue=dirs["dlc_queue"], tmp=tmp_path, mp=monkeypatch)


def _vid(i):
    return f"20250101_CNT01{i:02d}_P1"


def _validated(o, vid, path):
    o.db.register_video(video_id=vid, source_path=str(path), current_path=str(path))
    o.db.force_state(vid, "validated", reason="test setup", current_path=str(path))


def _taken_by_other(env, vid):
    p = env.door / f"{vid}.mp4"
    p.write_bytes(b"video")
    single_claim.claim_single(p, OTHER)


def _count_mp4_stats(monkeypatch):
    calls = []

    def wrap(orig):
        def f(p, *a, **k):
            try:
                s = os.fspath(p)
            except TypeError:
                s = ""
            if isinstance(s, str) and s.lower().endswith(".mp4"):
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


def test_taken_videos_cost_no_request_per_row(env):
    o = env.o
    for i in range(1, N + 1):
        _taken_by_other(env, _vid(i))
        _validated(o, _vid(i), env.door / f"{_vid(i)}.mp4")
    calls = _count_mp4_stats(env.mp)
    kept = o._split_gone_singles(o.db.get_videos_in_state("validated"))
    assert kept == []
    assert calls == [], f"{len(calls)} per-video requests for {N} taken videos"
    row = o.db.get_video(_vid(1))
    assert row["state"] == "unresolvable"
    assert OTHER in row["error_message"]


def test_rows_are_still_sorted_the_same_way(env, tmp_path):
    o = env.o
    _taken_by_other(env, _vid(1))
    _validated(o, _vid(1), env.door / f"{_vid(1)}.mp4")            # taken: gone
    (env.door / f"{_vid(2)}.mp4").write_bytes(b"video")
    _validated(o, _vid(2), env.door / f"{_vid(2)}.mp4")            # still there: kept
    elsewhere = tmp_path / "elsewhere" / f"{_vid(3)}.mp4"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"video")
    _validated(o, _vid(3), elsewhere)                              # recorded elsewhere, exists: kept
    _validated(o, _vid(4), tmp_path / "nowhere" / f"{_vid(4)}.mp4")  # nowhere: gone
    kept = o._split_gone_singles(o.db.get_videos_in_state("validated"))
    assert sorted(v["video_id"] for v in kept) == [_vid(2), _vid(3)]
    assert o.db.get_video(_vid(4))["state"] == "unresolvable"
