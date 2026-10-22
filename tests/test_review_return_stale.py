"""The stale-bundle guard: a queue husk must never flip a healthy video.

WHY: on 2026-09-04 two videos re-completed CLEAN (review applied, features
extracted, state 'processed') and 1.5 s later the return scan diverted their
STALE seg-failed triage husks to deep_review -- flipping the healthy videos'
state and blocking their archive. The guard checks the video's live state:
an actively-handled video's bundle is a leftover, retired to _Problematic
(never deleted), and the video is left alone.
"""
import json
import os
import time

import mousereach.watcher.review_return as rr
import mousereach.watcher.review_gate as rg


class FakeDB:
    def __init__(self, state):
        self._state = state
        self.forced = []

    def get_video(self, vid):
        return {"video_id": vid, "state": self._state}

    def force_state(self, *a, **k):
        self.forced.append((a, k))


def _make_bundle(queue_dir, stem):
    b = queue_dir / stem
    b.mkdir(parents=True)
    # seg_failed by the segmenter's own account: overall_confidence 0
    (b / f"{stem}_segments.json").write_text(
        json.dumps({"overall_confidence": 0.0, "boundaries": []}),
        encoding="utf-8")
    (b / f"{stem}_routing.json").write_text(
        json.dumps({"video_id": stem,
                    "routed_reason": "segmentation failed -- needs deep review"}),
        encoding="utf-8")
    return b


def _wire_paths(monkeypatch, tmp_path):
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    # WHY a plain FILE at the retired queue name, as the migrated share has: a
    # hardcoded REVIEW_ROOT / "flagged_for_review" join then raises here instead
    # of quietly finding these bundles and passing.
    (tmp_path / "Review" / "flagged_for_review").write_text("retired folder",
                                                            encoding="ascii")
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", tmp_path / "Review")
    return triage, deep


def test_stale_bundle_is_retired_not_diverted(tmp_path, monkeypatch):
    triage, deep = _wire_paths(monkeypatch, tmp_path)
    stem = "20240101_ABC0101_P1"
    _make_bundle(triage, stem)
    routed = []
    monkeypatch.setattr(rg, "route_to_queue",
                        lambda *a, **k: routed.append(a))

    db = FakeDB("processed")
    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert summary.get("stale_retired") == 1
    assert not routed                          # no divert attempted
    assert not db.forced                       # video state untouched
    assert not (triage / stem).exists()        # left the queue
    retired = list((tmp_path / "Review" / "_Problematic").rglob(
        f"{stem}_routing.json"))
    assert retired                             # archived, not deleted
    note = list((tmp_path / "Review" / "_Problematic").rglob(
        f"{stem}_stale_retire.json"))
    assert note and "re-handled" in note[0].read_text(encoding="utf-8")


def test_empty_dir_is_retired_not_routed(tmp_path, monkeypatch):
    """An OLD empty dir is residue, not a bundle: routing zero files while
    flipping db state is phantom action (it re-flipped a parked video the
    moment its twin side cleared, 2026-09-08). Retired regardless of state.
    Only once it is older than EMPTY_DIR_GRACE_SECONDS: a young empty dir may
    be a route still filling it (retired mid-route on 2026-09-14)."""
    triage, deep = _wire_paths(monkeypatch, tmp_path)
    stem = "20240101_ABC0103_P1"
    (triage / stem).mkdir()
    t = time.time() - 3 * rr.EMPTY_DIR_GRACE_SECONDS
    os.utime(triage / stem, (t, t))
    routed = []
    monkeypatch.setattr(rg, "route_to_queue",
                        lambda *a, **k: routed.append(a))

    db = FakeDB("triage")                      # even a non-active state
    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert summary.get("empty_retired") == 1
    assert not routed and not db.forced
    assert not (triage / stem).exists()


def test_parked_unresolvable_video_is_never_unparked(tmp_path, monkeypatch):
    """A person shelved that video with a reason; a husk must not undo it."""
    triage, deep = _wire_paths(monkeypatch, tmp_path)
    stem = "20240101_ABC0104_P1"
    _make_bundle(triage, stem)
    routed = []
    monkeypatch.setattr(rg, "route_to_queue",
                        lambda *a, **k: routed.append(a))

    db = FakeDB("unresolvable")
    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert summary.get("stale_retired") == 1
    assert not routed and not db.forced
    assert not (triage / stem).exists()


def test_genuinely_held_video_still_diverts(tmp_path, monkeypatch):
    triage, deep = _wire_paths(monkeypatch, tmp_path)
    stem = "20240101_ABC0102_P1"
    _make_bundle(triage, stem)
    routed = []
    monkeypatch.setattr(rg, "route_to_queue",
                        lambda *a, **k: routed.append((a, k)))

    db = FakeDB("triage")                      # not actively re-handled
    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert summary.get("diverted_to_deep") == 1
    assert routed                              # the designed divert ran
    assert (triage / stem).exists()            # recorder did not move it


def _split_escalation(triage, deep, stem, triage_mp4=b"video"):
    """The historical split: the full bundle in deep review, the mp4 (plus
    whatever the escalation left) still in triage."""
    d = _make_bundle(deep, stem)
    (d / f"{stem}.mp4").write_bytes(b"video")
    t = _make_bundle(triage, stem)
    (t / f"{stem}.mp4").write_bytes(triage_mp4)
    return t, d


def test_byte_identical_leftover_in_triage_is_retired(tmp_path, monkeypatch):
    """A triage copy holding only byte-identical copies of deep-review files
    carries nothing the reviewer's copy lacks: archived, never routed, and the
    deep-review bundle is untouched (20250808_CNT0113_P1, 2026-10-22)."""
    triage, deep = _wire_paths(monkeypatch, tmp_path)
    stem = "20240101_ABC0105_P1"
    t, d = _split_escalation(triage, deep, stem)
    before = sorted(p.name for p in d.iterdir())
    routed = []
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append(a))

    db = FakeDB("deep_review")
    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert summary.get("leftover_retired") == 1
    assert not routed and not db.forced
    assert not t.exists()
    assert sorted(p.name for p in d.iterdir()) == before
    archived = list((tmp_path / "Review" / "_Problematic").rglob(f"{stem}.mp4"))
    assert archived and archived[0].read_bytes() == b"video"
    note = list((tmp_path / "Review" / "_Problematic").rglob(f"{stem}_stale_retire.json"))
    assert note and "byte-identical" in note[0].read_text(encoding="utf-8")


def test_duplicate_that_differs_is_left_for_a_person(tmp_path, monkeypatch):
    """Same names and sizes but different bytes: not a leftover. It stays,
    nothing is routed, and its files are read once, not on every scan."""
    triage, deep = _wire_paths(monkeypatch, tmp_path)
    stem = "20240101_ABC0106_P1"
    t, d = _split_escalation(triage, deep, stem, triage_mp4=b"VIDEO")
    routed = []
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append(a))
    hashed = []
    real = rr._file_sha256
    monkeypatch.setattr(rr, "_file_sha256", lambda p: hashed.append(p) or real(p))

    for _ in range(2):
        summary = rr.scan_review_queues(FakeDB("deep_review"), tmp_path / "proc")
        assert not summary.get("leftover_retired")

    assert (t / f"{stem}.mp4").read_bytes() == b"VIDEO"
    assert not routed
    assert not (tmp_path / "Review" / "_Problematic").exists()
    first_scan = len(hashed)
    assert first_scan and first_scan <= 2 * 3          # one pass over the files


def test_duplicate_with_a_file_the_twin_lacks_is_left(tmp_path, monkeypatch):
    triage, deep = _wire_paths(monkeypatch, tmp_path)
    stem = "20240101_ABC0107_P1"
    t, d = _split_escalation(triage, deep, stem)
    (t / f"{stem}_causal_review.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: None)

    summary = rr.scan_review_queues(FakeDB("deep_review"), tmp_path / "proc")

    assert not summary.get("leftover_retired")
    assert (t / f"{stem}_causal_review.json").is_file()
