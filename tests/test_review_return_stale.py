"""The stale-bundle guard: a queue husk must never flip a healthy video.

WHY: on 2026-09-04 two videos re-completed CLEAN (review applied, features
extracted, state 'processed') and 1.5 s later the return scan diverted their
STALE seg-failed triage husks to deep_review -- flipping the healthy videos'
state and blocking their archive. The guard checks the video's live state:
an actively-handled video's bundle is a leftover, retired to _Problematic
(never deleted), and the video is left alone.
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
    deep = tmp_path / "Review" / "flagged_for_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
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
