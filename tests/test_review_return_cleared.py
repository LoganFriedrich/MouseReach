"""A human's deep-review clearance outranks the seg self-check in the RETURN
SCAN, not only in the gate.

WHY: the gate honors {stem}_deep_review_cleared.json when segmentation
self-reports failure, but the return scan re-derived seg_failed raw and
diverted the triage bundle to deep review, where the very same marker
instantly released it -- processing -> triage -> deep_review -> processing,
forever (20250708_CNT0210_P4 made 43 laps on 2026-09-09). A cleared bundle is
an ordinary triage bundle: it sits until its triaged segments are answered,
then releases.
"""
import json

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


def _wire(monkeypatch, tmp_path):
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "flagged_for_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", tmp_path / "Review")
    monkeypatch.setattr(rr.time, "sleep", lambda s: None)
    return triage, deep


def _cleared_seg_failed_bundle(queue_dir, stem, resolved: bool):
    """seg_failed by self-check, one triaged segment, deep-review cleared by a
    human; the triaged segment answered or not per ``resolved``."""
    b = queue_dir / stem
    b.mkdir(parents=True)
    (b / f"{stem}_segments.json").write_text(
        json.dumps({"overall_confidence": 0.0, "boundaries": []}),
        encoding="utf-8")
    (b / f"{stem}_pellet_outcomes.json").write_text(
        json.dumps({"segments": [{"segment_num": 3, "outcome": "triaged"}]}),
        encoding="utf-8")
    (b / f"{stem}_deep_review_cleared.json").write_text(
        json.dumps({"cleared_by": "tester"}), encoding="utf-8")
    if resolved:
        (b / f"{stem}_causal_review.json").write_text(
            json.dumps({"segments": [
                {"segment_num": 3, "answers": {"reviewed": True}}]}),
            encoding="utf-8")
    return b


def test_cleared_bundle_is_not_diverted_while_unresolved(tmp_path, monkeypatch):
    """The seg question is answered; the bundle WAITS in triage for its
    segments, instead of lapping through deep review."""
    triage, deep = _wire(monkeypatch, tmp_path)
    stem = "20240101_ABC0201_P1"
    _cleared_seg_failed_bundle(triage, stem, resolved=False)
    routed = []
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append(a))
    returned = []
    monkeypatch.setattr(rr, "_return_to_processing",
                        lambda *a, **k: returned.append(a) or True)

    db = FakeDB("triage")
    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert not routed                          # no divert
    assert not returned                        # and no premature release
    assert summary.get("diverted_to_deep", 0) == 0
    assert (triage / stem).exists()            # sits, waiting for answers


def test_cleared_bundle_releases_once_segments_are_answered(tmp_path, monkeypatch):
    triage, deep = _wire(monkeypatch, tmp_path)
    stem = "20240101_ABC0202_P1"
    _cleared_seg_failed_bundle(triage, stem, resolved=True)
    routed = []
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append(a))
    returned = []
    monkeypatch.setattr(rr, "_return_to_processing",
                        lambda bundle, stem_, pd, db_, reason: (
                            returned.append((stem_, reason)) or True))

    db = FakeDB("triage")
    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert not routed
    assert returned == [(stem, "triage_cleared")]
    assert summary.get("triage_returned") == 1


def test_cleared_husk_is_still_retired(tmp_path, monkeypatch):
    """The stale-bundle guard applies to cleared bundles too: a husk whose
    video was re-handled must be retired, never re-injected."""
    triage, deep = _wire(monkeypatch, tmp_path)
    stem = "20240101_ABC0203_P1"
    _cleared_seg_failed_bundle(triage, stem, resolved=True)
    routed = []
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append(a))
    returned = []
    monkeypatch.setattr(rr, "_return_to_processing",
                        lambda *a, **k: returned.append(a) or True)

    db = FakeDB("archived")
    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert summary.get("stale_retired") == 1
    assert not routed and not returned
    assert not (triage / stem).exists()
