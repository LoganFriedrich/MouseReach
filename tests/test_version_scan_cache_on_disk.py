"""The version scan's caches survive a watcher restart.

WHY: the first scan after a start re-read every manifest and review over the
share -- 623 s on 2026-10-22, with the work loop waiting -- although nearly
all of them were unchanged since the previous run. The caches now live in a
local file; every entry is still checked against the file's current listing
before use, so what changed is read again and nothing stale is trusted.
"""
import json
import os
import time

import mousereach.pipeline.fsutil as fsutil
import mousereach.pipeline.versions as versions_mod
import mousereach.review.causal_review_io as crio
import mousereach.watcher.reprocessor as rp

from test_reprocessor_archive_skip import CURRENT, NEW, OLD, STALE, StubDB

STEM = "20240101_ABC0101_P1"
STEM_B = "20240101_ABC0102_P1"


def _set_up(tmp_path, monkeypatch, stems=(STEM,)):
    coh = tmp_path / "Analyzed" / "Connectome" / "CNT01"
    coh.mkdir(parents=True)
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    index = {}
    for s in stems:
        _manifest(coh, s, NEW)
        r = reviews / f"{s}_causal_review.json"
        r.write_text("{}", encoding="utf-8")
        st = r.stat()
        index[s] = (r, st.st_mtime, st.st_size)
    monkeypatch.setattr(crio, "review_path_index",
                        lambda wanted: {s: index[s] for s in wanted if s in index})
    monkeypatch.setattr(versions_mod, "get_current_versions", lambda root=None: CURRENT)
    monkeypatch.setattr(versions_mod, "declaration_drift", lambda cur: [])
    hashed, opened = [], []
    monkeypatch.setattr(fsutil, "sha256_file", lambda p: hashed.append(p) or "abc")
    real_open = open
    monkeypatch.setattr(rp, "open", lambda p, *a, **k: opened.append(str(p))
                        or real_open(p, *a, **k), raising=False)
    return coh, hashed, opened


def _manifest(coh, stem, scorer, age=600):
    p = coh / f"{stem}_processing_manifest.json"
    p.write_text(json.dumps({"dlc_model": {"dlc_scorer": scorer},
                             "pipeline_versions": STALE,
                             "applied_review": {"sha256": "abc"}}), encoding="utf-8")
    t = time.time() - age
    os.utime(p, (t, t))
    return p


def _scan(tmp_path, stems=(STEM,)):
    db = StubDB(rows=stems, archived=stems)
    sc = rp.ReprocessingScanner(db, tmp_path, cache_path=tmp_path / "local" / "cache.json")
    return sc.scan(mark_outdated=True, adopt_orphans=False), db


def test_a_restarted_scanner_reads_nothing_unchanged(tmp_path, monkeypatch):
    (tmp_path / "local").mkdir()
    _, hashed, opened = _set_up(tmp_path, monkeypatch)

    first, _ = _scan(tmp_path)
    assert first["current"] == 1
    assert len(opened) == 1 and len(hashed) == 1
    assert (tmp_path / "local" / "cache.json").is_file()

    second, _ = _scan(tmp_path)                    # a new scanner: a restart
    assert second["current"] == 1
    assert len(opened) == 1 and len(hashed) == 1   # nothing read again


def test_a_changed_manifest_is_read_again_after_a_restart(tmp_path, monkeypatch):
    (tmp_path / "local").mkdir()
    coh, _, opened = _set_up(tmp_path, monkeypatch)
    _scan(tmp_path)
    _manifest(coh, STEM, OLD, age=60)              # rewritten: an older pose model

    summary, db = _scan(tmp_path)

    assert len(opened) == 2
    assert summary["outdated"] == 1
    assert (STEM, "outdated", "full") in db.forced


def test_a_damaged_cache_file_only_means_a_cold_scan(tmp_path, monkeypatch):
    (tmp_path / "local").mkdir()
    (tmp_path / "local" / "cache.json").write_text("{not json", encoding="utf-8")
    _, _, opened = _set_up(tmp_path, monkeypatch)

    summary, _ = _scan(tmp_path)

    assert summary["current"] == 1 and len(opened) == 1
    doc = json.loads((tmp_path / "local" / "cache.json").read_text(encoding="utf-8"))
    assert doc["format"] == rp.ReprocessingScanner._CACHE_FORMAT


def test_videos_that_left_the_archive_leave_the_cache(tmp_path, monkeypatch):
    (tmp_path / "local").mkdir()
    coh, _, _ = _set_up(tmp_path, monkeypatch, stems=(STEM, STEM_B))
    _scan(tmp_path, stems=(STEM, STEM_B))
    (coh / f"{STEM_B}_processing_manifest.json").unlink()

    _scan(tmp_path, stems=(STEM,))

    doc = json.loads((tmp_path / "local" / "cache.json").read_text(encoding="utf-8"))
    assert [m[0] for m in doc["manifests"]] == [STEM]
    assert len(doc["review_sha"]) == 1


def test_no_cache_path_keeps_the_caches_in_memory_only(tmp_path, monkeypatch):
    _set_up(tmp_path, monkeypatch)
    sc = rp.ReprocessingScanner(StubDB(rows=(STEM,), archived=(STEM,)), tmp_path)
    sc.scan(mark_outdated=True, adopt_orphans=False)
    assert not list(tmp_path.rglob("*cache*.json"))
