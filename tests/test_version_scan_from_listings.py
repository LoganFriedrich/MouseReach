"""The archive's version scan finds what it needs from folder listings.

WHY: every half hour it checked every archived video separately -- its
manifest's modified time, several places its review might be, and the
review's fingerprint -- about 10 minutes a pass for 4,597 videos on 2026-10-21,
while the processing server's work loop waited. The listings must give
EXACTLY the answers the per-video checks gave.
"""
import json
import os
import time
from types import SimpleNamespace

import mousereach.pipeline.fsutil as fsutil
import mousereach.review.causal_review_io as io
import mousereach.review.staging as staging
from mousereach.watcher.reprocessor import ReprocessingScanner

SUFFIX = "_causal_review.json"


def _age(p, seconds):
    t = time.time() - seconds
    os.utime(p, (t, t))
    return p


def _review(path, age):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return _age(path, age)


def test_the_index_gives_exactly_what_the_per_video_lookup_gives(tmp_path, monkeypatch):
    pending = tmp_path / "triage"
    durable = tmp_path / "index" / "reviews"
    root = tmp_path / "Analyzed"
    pending.mkdir()
    durable.mkdir(parents=True)
    monkeypatch.setattr(staging, "DEFAULT_PENDING_DIR", pending)
    monkeypatch.setattr(staging, "DEFAULT_CONNECTOME_ROOT", root)
    monkeypatch.setattr(io, "durable_review_dir", lambda: durable)

    A, B, C, D, E, F = (f"20250101_CNT01{n:02d}_P1" for n in range(1, 7))
    G = "20220101_ASPA0401_P1"                       # not a CNT name at all
    _review(pending / A / f"{A}{SUFFIX}", 100)                      # pending only
    _review(durable / f"{B}{SUFFIX}", 100)                          # durable only
    _review(pending / C / f"{C}{SUFFIX}", 500)                      # both: durable newer
    _review(durable / f"{C}{SUFFIX}", 100)
    _review(pending / D / f"{D}{SUFFIX}", 100)                      # both: pending newer
    _review(durable / f"{D}{SUFFIX}", 500)
    # E: beside its canonical video (the video and its 4.0 pose present)
    cohort = root / "CNT01"
    cohort.mkdir(parents=True)
    (cohort / f"{E}.mp4").write_bytes(b"v")
    dlc = root / staging.DLC4_SUBDIR / "CNT01"
    dlc.mkdir(parents=True)
    (dlc / f"{E}{staging.DLC4_SCORER}.h5").write_bytes(b"p")
    _review(cohort / f"{E}{SUFFIX}", 50)
    _review(durable / f"{E}{SUFFIX}", 400)
    _review(durable / f"{G}{SUFFIX}", 100)
    (pending / F).mkdir()                                           # bundle, no review

    stems = [A, B, C, D, E, F, G]
    index = io.review_path_index(stems)
    for s in stems:
        expected = io.resolve_review_path(s)
        got = index.get(s, (None,))[0]
        assert got == expected, (s, got, expected)
    assert F not in index


def test_the_listing_walk_finds_what_the_globs_found_and_skips_the_archive(tmp_path):
    root = tmp_path / "Analyzed"
    live = root / "Connectome" / "CNT01"
    old = root / "Archive" / "2026-09-01_superseded"
    live.mkdir(parents=True)
    old.mkdir(parents=True)
    for d in (live, old):
        (d / "20250101_CNT0101_P1_processing_manifest.json").write_text("{}", encoding="utf-8")
        (d / "20250101_CNT0101_P1_features.json").write_text("{}", encoding="utf-8")
    (live / "notes.txt").write_text("x", encoding="utf-8")
    sc = object.__new__(ReprocessingScanner)
    sc.archive_dir = root
    man, man_t, feat_t = sc._index_archive()
    assert man == {"20250101_CNT0101_P1": live / "20250101_CNT0101_P1_processing_manifest.json"}
    assert set(man_t) == set(feat_t) == {"20250101_CNT0101_P1"}
    assert abs(man_t["20250101_CNT0101_P1"]
               - (live / "20250101_CNT0101_P1_processing_manifest.json").stat().st_mtime) < 1


def test_an_unchanged_review_is_fingerprinted_once(tmp_path, monkeypatch):
    review = _review(tmp_path / f"X{SUFFIX}", 10)
    calls = []
    monkeypatch.setattr(fsutil, "sha256_file", lambda p: calls.append(p) or "abc")
    sc = object.__new__(ReprocessingScanner)
    sc._review_sha_cache = {}
    st = review.stat()
    index = {"X": (review, st.st_mtime, st.st_size)}
    manifest = {"applied_review": {"sha256": "abc"}}
    assert sc._pending_review_path("X", manifest, {}, index) is None
    assert sc._pending_review_path("X", manifest, {}, index) is None
    assert len(calls) == 1
    # a video with no review in the index has nothing pending, and is not looked up
    assert sc._pending_review_path("Y", manifest, {}, index) is None
