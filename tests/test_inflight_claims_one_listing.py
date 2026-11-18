"""Reading who holds which shared single costs one listing per folder.

WHY: single_claim.inflight_claims() runs in every pick of every GPU node
(_split_gone_singles) and in several per-video handlers. It listed each
host's claim folder and then asked the share about every claimed video twice
(is_file, then stat for its modified time), plus once per host folder. On a
slow link those per-file requests were most of a pick's time. os.scandir
returns each entry's type and modified time with the listing.

The guard counts stat calls under the claim area during one read: with N
claimed videos it must not grow with N. The result must be what it was.

tmp_path only; the shared singles folder is patched onto Paths.
"""
import os
import pathlib

import pytest

from mousereach.config import Paths
from mousereach.watcher import single_claim

N = 40


@pytest.fixture
def door(tmp_path, monkeypatch):
    d = tmp_path / "Single_Animal"
    d.mkdir()
    monkeypatch.setattr(Paths, "SINGLE_ANIMAL_OUTPUT", d, raising=False)
    return d


def _count_stats_under(monkeypatch, folder):
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


def _claim(door, host, vid, mtime):
    d = door / single_claim.INFLIGHT_DIR / host
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{vid}.mp4"
    p.write_bytes(b"video")
    os.utime(p, (mtime, mtime))
    return p


def test_claims_are_read_without_a_request_per_video(door, monkeypatch):
    for i in range(N):
        _claim(door, f"HOST{i % 3}", f"20240102_ABC{i:04d}_P1", 1_700_000_000 + i)
    calls = _count_stats_under(monkeypatch, door)
    got = single_claim.inflight_claims()
    assert len(got) == N
    assert calls == [], f"{len(calls)} per-file requests for {N} claimed videos"


def test_result_is_unchanged(door):
    _claim(door, "HOSTB", "20240102_ABC0001_P1", 1_700_000_100)
    _claim(door, "HOSTA", "20240102_ABC0001_P1", 1_700_000_050)   # same video twice
    _claim(door, "HOSTA", "20240102_ABC0002_P1", 1_700_000_200)
    (door / single_claim.INFLIGHT_DIR / "HOSTA" / "notes.txt").write_text("x")
    (door / single_claim.INFLIGHT_DIR / "HOSTA" / "sub.mp4").mkdir()
    (door / single_claim.INFLIGHT_DIR / "stray.mp4").write_bytes(b"x")  # not a host folder
    got = single_claim.inflight_claims()
    # hosts are read in name order, so the first holder by name wins a duplicate
    assert got == {"20240102_ABC0001_P1": ("HOSTA", pytest.approx(1_700_000_050)),
                   "20240102_ABC0002_P1": ("HOSTA", pytest.approx(1_700_000_200))}
    assert single_claim.inflight_ids() == {"20240102_ABC0001_P1": "HOSTA",
                                           "20240102_ABC0002_P1": "HOSTA"}


def test_no_claim_area_is_empty(door):
    assert single_claim.inflight_claims() == {}
