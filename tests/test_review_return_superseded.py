"""A queue bundle of a video that has since been archived is filed away.

WHY (2026-10-22): between 2026-09-10 and 09-29 twenty videos a person had
cleared were sent back to deep review as a bare .mp4 (a file a return could
not delete while the review tool held it open; see
orchestrator.queue_folder_holds_this_run). Fifteen were cleared a second time
by hand; others sat in the queues behind bundles older than their archived
results. Only an archived video's bundle that no person wrote in qualifies,
and bundles the bench-disagreement router puts in a queue are never caught.
Everything is built under tmp_path with a real watcher database.
"""
import json
import os
import time

import mousereach.archive.core as archive_core
import mousereach.watcher.review_gate as rg
import mousereach.watcher.review_return as rr
from mousereach.watcher.db import WatcherDB

STEM = "20240101_ABC0101_P1"


def _setup(tmp_path, monkeypatch, state="archived", archived_manifest=True):
    review = tmp_path / "Review"
    triage, deep = review / "triage", review / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", review)
    analyzed = tmp_path / "Analyzed" / "ABC01"
    analyzed.mkdir(parents=True)
    monkeypatch.setattr(archive_core, "get_archive_destination", lambda vid: analyzed)
    if archived_manifest:
        m = analyzed / f"{STEM}_processing_manifest.json"
        m.write_text("{}", encoding="utf-8")
        t = time.time() - 3600
        os.utime(m, (t, t))
    monkeypatch.setattr(rg, "route_to_queue",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("routed")))
    rr._superseded_cache.clear()
    rr._status_cache.clear()
    db = WatcherDB(tmp_path / "w.db")
    db.register_video(video_id=STEM, source_path="x")
    db.force_state(STEM, state, reason="test fixture")
    return triage, deep, db


def _files(folder, names, age_s):
    folder.mkdir(parents=True, exist_ok=True)
    t = time.time() - age_s
    for n in names:
        p = folder / n
        # A sound segmentation, so the triage loop's own seg-failed rule
        # (test_review_return_stale) does not act on these bundles.
        body = ({"overall_confidence": 0.9, "boundaries": []}
                if n.endswith("_segments.json") else {})
        p.write_text(json.dumps(body), encoding="utf-8")
        os.utime(p, (t, t))
    return folder


HUSK = [f"{STEM}.mp4", f"{STEM}DLC_resnet101_X_100000.h5",
        f"{STEM}_manifest.json", f"{STEM}_routing.json"]
FULL = [f"{STEM}.mp4", f"{STEM}_segments.json", f"{STEM}_reaches.json",
        f"{STEM}_pellet_outcomes.json", f"{STEM}_manifest.json"]


def _note():
    return " ".join(p.read_text(encoding="utf-8")
                    for p in (rr.Paths.REVIEW_ROOT / "_Problematic").rglob("*_stale_retire.json"))


def test_a_results_free_husk_of_an_archived_video_leaves_deep_review(tmp_path, monkeypatch):
    # 20250808_CNT0113_P2: diverted after its clean run, so it is NEWER than
    # the archived results -- what marks it is that it holds no results.
    triage, deep, db = _setup(tmp_path, monkeypatch)
    _files(deep / STEM, HUSK, age_s=60)
    summary = rr.scan_review_queues(db, tmp_path / "proc")
    assert summary.get("superseded_retired") == 1
    assert not (deep / STEM).exists()
    assert list((tmp_path / "Review" / "_Problematic").rglob(f"{STEM}.mp4"))
    assert "holds no results" in _note()
    assert db.get_video(STEM)["state"] == "archived"


def test_a_full_bundle_older_than_the_archived_results_leaves_triage(tmp_path, monkeypatch):
    triage, deep, db = _setup(tmp_path, monkeypatch)
    _files(triage / STEM, FULL, age_s=86400)
    summary = rr.scan_review_queues(db, tmp_path / "proc")
    assert summary.get("superseded_retired") == 1
    assert not (triage / STEM).exists()
    assert "predates" in _note()


def test_a_bundle_routed_after_the_archive_stays(tmp_path, monkeypatch):
    # The bench-disagreement router's case: its routing record is written
    # after the archive.
    triage, deep, db = _setup(tmp_path, monkeypatch)
    b = _files(triage / STEM, FULL, age_s=86400)
    (b / f"{STEM}_routing.json").write_text("{}", encoding="utf-8")
    rr.scan_review_queues(db, tmp_path / "proc")
    assert (triage / STEM).is_dir()


def test_a_bundle_whose_results_left_analyzed_stays(tmp_path, monkeypatch):
    triage, deep, db = _setup(tmp_path, monkeypatch, archived_manifest=False)
    _files(triage / STEM, FULL, age_s=86400)
    rr.scan_review_queues(db, tmp_path / "proc")
    assert (triage / STEM).is_dir()


def test_a_bundle_a_person_wrote_in_stays(tmp_path, monkeypatch):
    triage, deep, db = _setup(tmp_path, monkeypatch)
    _files(deep / STEM, HUSK + [f"{STEM}_causal_review.json"], age_s=86400)
    rr.scan_review_queues(db, tmp_path / "proc")
    assert (deep / STEM).is_dir()


def test_a_held_video_keeps_its_bundle(tmp_path, monkeypatch):
    triage, deep, db = _setup(tmp_path, monkeypatch, state="deep_review")
    _files(deep / STEM, HUSK, age_s=86400)
    rr.scan_review_queues(db, tmp_path / "proc")
    assert (deep / STEM).is_dir()


def test_analyzed_is_read_once_per_bundle_and_row_change(tmp_path, monkeypatch):
    triage, deep, db = _setup(tmp_path, monkeypatch)
    _files(triage / STEM, FULL + [f"{STEM}_routing.json"], age_s=0)   # stays
    calls = []
    real = archive_core.get_archive_destination
    monkeypatch.setattr(archive_core, "get_archive_destination",
                        lambda vid: calls.append(vid) or real(vid))
    for _ in range(3):
        rr.scan_review_queues(db, tmp_path / "proc")
    assert (triage / STEM).is_dir()
    assert len(calls) == 1


def test_states_of_reads_many_rows_in_one_call(tmp_path):
    db = WatcherDB(tmp_path / "w.db")
    ids = [f"20240101_ABC01{n:02d}_P1" for n in range(1, 13)]
    for vid in ids:
        db.register_video(video_id=vid, source_path="x")
    db.force_state(ids[0], "archived", reason="test")
    got = db.states_of(ids + ["20240101_ABC0199_P1"])
    assert set(got) == set(ids)
    assert got[ids[0]][0] == "archived"
    assert db.states_of([]) == {}
