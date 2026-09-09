"""Cross-node claims on review-queue returns.

WHY: the review queues live ONCE on the NAS, shared by every watcher node.
With several nodes scanning the same queues, two that spot the same cleared
bundle in one pass would both start moving its files and tear the bundle in
half across machines -- the split-bundle failure class, cross-node edition.
The claim is the same write-then-verify marker pattern the DLC intake uses
(orchestrator._claim_video): claimed only at MUTATION time, never for mere
scanning; fail-open single-node; a stale marker is a crashed node's and is
broken.
"""
import json
import os
import time as _time

import mousereach.watcher.review_return as rr
import mousereach.watcher.review_gate as rg

OTHER_NODE = "SOME-OTHER-NODE"


def _wire(monkeypatch, tmp_path, no_sleep=True):
    review = tmp_path / "Review"
    triage = review / "triage"
    deep = review / "flagged_for_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", review)
    if no_sleep:
        monkeypatch.setattr(rr.time, "sleep", lambda s: None)
    return triage, deep, review


def _foreign_claim(review, stem, age_s=0):
    d = review / ".return_claims"
    d.mkdir(exist_ok=True)
    f = d / f"{stem}.claim"
    f.write_text(f"{OTHER_NODE}\n2026-09-09T00:00:00\n", encoding="utf-8")
    if age_s:
        old = _time.time() - age_s
        os.utime(f, (old, old))
    return f


# --- helper unit tests ---------------------------------------------------

def test_claim_then_release(tmp_path, monkeypatch):
    _, _, review = _wire(monkeypatch, tmp_path)
    assert rr._claim_return("stemA") is True
    f = review / ".return_claims" / "stemA.claim"
    assert f.exists()
    rr._release_return_claim("stemA")
    assert not f.exists()


def test_foreign_fresh_claim_refused(tmp_path, monkeypatch):
    _, _, review = _wire(monkeypatch, tmp_path)
    _foreign_claim(review, "stemB")
    assert rr._claim_return("stemB") is False


def test_foreign_stale_claim_broken(tmp_path, monkeypatch):
    """A claim outliving the stale window is a crashed node's leftover."""
    _, _, review = _wire(monkeypatch, tmp_path)
    f = _foreign_claim(review, "stemC", age_s=rr._RETURN_CLAIM_STALE_S + 60)
    assert rr._claim_return("stemC") is True
    assert f.read_text(encoding="utf-8").split("\n")[0] != OTHER_NODE


def test_own_claim_is_reentrant(tmp_path, monkeypatch):
    """A crash between claim and release must not deadlock the same node."""
    _, _, review = _wire(monkeypatch, tmp_path)
    assert rr._claim_return("stemD") is True
    assert rr._claim_return("stemD") is True


def test_fail_open_without_review_root(tmp_path, monkeypatch):
    """Single-node mode (no NAS root configured): claims are a no-op."""
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", None)
    assert rr._claim_return("stemE") is True
    rr._release_return_claim("stemE")   # must not raise


# --- integration-lite: the scan honours a foreign claim ------------------

class FakeDB:
    def __init__(self, state):
        self._state = state
        self.forced = []

    def get_video(self, vid):
        return {"video_id": vid, "state": self._state}

    def force_state(self, *a, **k):
        self.forced.append((a, k))


def _seg_failed_bundle(queue_dir, stem):
    b = queue_dir / stem
    b.mkdir(parents=True)
    (b / f"{stem}_segments.json").write_text(
        json.dumps({"overall_confidence": 0.0, "boundaries": []}),
        encoding="utf-8")
    (b / f"{stem}_routing.json").write_text(
        json.dumps({"video_id": stem, "routed_reason": "segmentation failed"}),
        encoding="utf-8")
    return b


def test_divert_skipped_while_other_node_holds_claim(tmp_path, monkeypatch):
    triage, deep, review = _wire(monkeypatch, tmp_path)
    stem = "20240101_ABC0105_P1"
    _seg_failed_bundle(triage, stem)
    _foreign_claim(review, stem)
    routed = []
    monkeypatch.setattr(rg, "route_to_queue",
                        lambda *a, **k: routed.append(a))

    db = FakeDB("triage")
    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert not routed                          # divert not attempted
    assert not db.forced                       # video state untouched
    assert (triage / stem).exists()            # bundle untouched
    assert summary.get("diverted_to_deep", 0) == 0


def test_retire_skipped_while_other_node_holds_claim(tmp_path, monkeypatch):
    triage, deep, review = _wire(monkeypatch, tmp_path)
    stem = "20240101_ABC0106_P1"
    _seg_failed_bundle(triage, stem)
    _foreign_claim(review, stem)
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: None)

    db = FakeDB("processed")                   # would normally stale-retire
    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert summary.get("stale_retired", 0) == 0
    assert (triage / stem).exists()            # bundle untouched


def test_return_to_processing_refused_under_foreign_claim(tmp_path, monkeypatch):
    _, _, review = _wire(monkeypatch, tmp_path)
    stem = "20240101_ABC0107_P1"
    bundle = tmp_path / "bundle" / stem
    bundle.mkdir(parents=True)
    _foreign_claim(review, stem)

    ok = rr._return_to_processing(bundle, stem, tmp_path / "proc",
                                  FakeDB("triage"), "triage_cleared")

    assert ok is False
    assert bundle.exists()                     # nothing moved


def test_claim_released_after_successful_retire(tmp_path, monkeypatch):
    """A completed mutation must not leave its marker pinning the stem."""
    triage, deep, review = _wire(monkeypatch, tmp_path)
    stem = "20240101_ABC0108_P1"
    _seg_failed_bundle(triage, stem)
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: None)

    db = FakeDB("processed")
    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert summary.get("stale_retired") == 1
    assert not (review / ".return_claims" / f"{stem}.claim").exists()
