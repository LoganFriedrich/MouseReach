"""The collage-retirement check finds its facts from folder listings.

WHY: it read all ~4,600 processing manifests in Analyzed and looked each
offspring's review up separately every time it ran -- 14 minutes on 2026-10-21
with a slow share, while the processing server's work loop waited. The answers
must be exactly the ones the per-file checks gave.
"""
import json
import os
import time

import mousereach.pipeline.versions as versions_mod
import mousereach.review.causal_review_io as crio
from mousereach.pipeline.analyzed_tree import iter_file_entries, iter_files
from mousereach.video_prep.core import collage_provenance as cp


def _age(p, seconds):
    t = time.time() - seconds
    os.utime(p, (t, t))
    return p


def _tree(tmp_path):
    root = tmp_path / "Analyzed"
    live = root / "Connectome" / "CNT01"
    old = root / "Archive" / "2026-09-01_superseded"
    for d in (live, old, root / "Connectome" / "DLC Model 4" / "CNT01"):
        d.mkdir(parents=True)
    return root, live, old


def test_the_entry_walk_finds_exactly_the_files_the_path_walk_finds(tmp_path):
    root, live, old = _tree(tmp_path)
    for d in (live, old, root / "Connectome" / "DLC Model 4" / "CNT01"):
        (d / "a_processing_manifest.json").write_text("{}", encoding="utf-8")
        (d / "a.mp4").write_bytes(b"v")
    for pattern in ("*", "*_processing_manifest.json", "*.mp4"):
        assert sorted(e.path for e in iter_file_entries(root, pattern)) == \
            sorted(str(p) for p in iter_files(root, pattern))
    assert not any("Archive" in e.path for e in iter_file_entries(root))


def _complete(monkeypatch, root, reviews, use_index):
    monkeypatch.setattr(versions_mod, "get_current_versions", lambda nas=None: {"versions": {}})
    monkeypatch.setattr(versions_mod, "compare_manifest_to_current",
                        lambda m, c: {"is_current": bool(m.get("current"))})
    monkeypatch.setattr(crio, "resolve_review_path", lambda stem, d=None: reviews.get(stem))
    if use_index:
        monkeypatch.setattr(crio, "review_path_index", lambda stems: {
            s: (p, p.stat().st_mtime, p.stat().st_size) for s, p in reviews.items() if s in stems})
    else:
        def broken(stems):
            raise RuntimeError("index unavailable")
        monkeypatch.setattr(crio, "review_path_index", broken)
    cp._MANIFEST_CACHE.clear()
    return cp.build_complete_stems(analyzed_root=root, nas_root=root)


def test_the_listing_answers_match_the_per_file_answers(tmp_path, monkeypatch):
    root, live, _ = _tree(tmp_path)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()

    def video(stem, current, features_age=None, review_age=None):
        (live / f"{stem}_processing_manifest.json").write_text(
            json.dumps({"current": current}), encoding="utf-8")
        if features_age is not None:
            (live / f"{stem}_features.json").write_text("{}", encoding="utf-8")
            _age(live / f"{stem}_features.json", features_age)
        if review_age is not None:
            r = reviews_dir / f"{stem}_causal_review.json"
            r.write_text("{}", encoding="utf-8")
            _age(r, review_age)
            return r

    reviews = {}
    video("done", True, features_age=100)                                   # complete
    video("stale", False, features_age=100)                                 # outdated
    reviews["applied"] = video("applied", True, features_age=50, review_age=100)   # review older
    reviews["pending"] = video("pending", True, features_age=100, review_age=50)   # review newer
    reviews["no_feats"] = video("no_feats", True, review_age=50)            # review, no features

    expected = {"done", "applied"}
    assert _complete(monkeypatch, root, reviews, use_index=False) == expected
    assert _complete(monkeypatch, root, reviews, use_index=True) == expected


def test_an_unchanged_manifest_is_read_once(tmp_path, monkeypatch):
    root, live, _ = _tree(tmp_path)
    (live / "v_processing_manifest.json").write_text(json.dumps({"current": True}),
                                                     encoding="utf-8")
    _complete(monkeypatch, root, {}, use_index=True)
    reads = []
    real = json.loads
    monkeypatch.setattr(cp.json, "loads", lambda s, *a, **k: reads.append(1) or real(s, *a, **k))
    assert cp.build_complete_stems(analyzed_root=root, nas_root=root) == {"v"}
    assert reads == []
