"""A quality hold the current check would not make leaves a person's queue.

WHY: the quality check demanded an outcomes file Easy and Flat trays never get,
so 469 Easy/Flat bundles sat in deep review on that false alarm (2026-10-22).
Once the check is corrected, such a hold -- made by the machine alone, with no
file a person wrote -- is re-checked, and a bundle the check now passes goes
back to the posed folder for a fresh run in its priority turn. Anything a
person touched, any other reason for the hold, and any hold the current check
still makes stays where it is.
"""
import json
from datetime import datetime

import mousereach.watcher.review_gate as rg
import mousereach.watcher.review_return as rr
from mousereach.pipeline.triage import EXPECTED_SEGMENTS_MIN

POSE = "DLC_resnet101_MPSAOct27shuffle3_100000"


class FakeDB:
    def __init__(self, fail=False):
        self.forced = []
        self.fail = fail

    def get_video(self, vid):
        return {"video_id": vid, "state": "deep_review"}

    def force_state(self, vid, state, reason=None, **kw):
        if self.fail:
            raise RuntimeError("database is locked")
        self.forced.append((vid, state, kw))

    def update_state(self, *a, **k):
        pass


def _wire(monkeypatch, tmp_path):
    review = tmp_path / "Review"
    deep, triage, posed = review / "deep_review", review / "triage", tmp_path / "Posed"
    for d in (deep, triage, posed):
        d.mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", review)
    monkeypatch.setattr(rr.Paths, "DLC_STAGING", posed, raising=False)
    monkeypatch.setattr(rr.Paths, "ANALYZED_OUTPUT", None, raising=False)
    monkeypatch.setattr(rr, "_qc_recheck_cache", {})
    monkeypatch.setattr(rr, "_return_retry_after", {})
    monkeypatch.setattr(rr, "_declared_scorer", lambda: POSE)
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: None)
    return deep, posed, review


def _bundle(deep, stem, reason="qc_needs_review",
            flags=({"category": "structural", "severity": "critical",
                    "description": "Missing outcomes file: x"},),
            outcomes=False, person=False):
    b = deep / stem
    b.mkdir()
    (b / f"{stem}.mp4").write_bytes(b"video")
    for ext in (".h5", ".csv", "_meta.pickle"):
        (b / f"{stem}{POSE}{ext}").write_bytes(b"pose")
    (b / f"{stem}_segments.json").write_text(json.dumps(
        {"boundaries": list(range(0, 1000 * EXPECTED_SEGMENTS_MIN, 1000))}), encoding="utf-8")
    (b / f"{stem}_reaches.json").write_text(json.dumps({"segments": []}), encoding="utf-8")
    if outcomes:
        (b / f"{stem}_pellet_outcomes.json").write_text(json.dumps({"segments": []}),
                                                       encoding="utf-8")
    (b / f"{stem}_triage.json").write_text(json.dumps(
        {"video_id": stem, "verdict": "needs_review", "flags": list(flags)}), encoding="utf-8")
    (b / f"{stem}_routing.json").write_text(json.dumps(
        {"video_id": stem, "routed_reason": reason}), encoding="utf-8")
    if person:
        (b / f"{stem}_causal_review.json").write_text("{}", encoding="utf-8")
    return b


def test_a_lifted_hold_goes_back_to_the_posed_folder(tmp_path, monkeypatch):
    deep, posed, review = _wire(monkeypatch, tmp_path)
    stem = "20261012_CNT0615_F4"
    b = _bundle(deep, stem)
    db = FakeDB()

    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert summary["qc_released"] == 1
    assert not b.exists()
    assert sorted(p.name for p in posed.iterdir()) == sorted(
        [f"{stem}.mp4"] + [f"{stem}{POSE}{e}" for e in (".h5", ".csv", "_meta.pickle")])
    kept = review / "_Problematic" / f"_released_qc_holds_{datetime.now():%Y-%m-%d}" / stem
    assert (kept / f"{stem}_triage.json").is_file()       # machine outputs set aside
    assert (kept / f"{stem}_release_note.json").is_file()
    assert db.forced == [(stem, "dlc_complete", {
        "current_path": str(posed / f"{stem}.mp4"),
        "dlc_output_path": str(posed / f"{stem}{POSE}.h5")})]
    assert not (tmp_path / "proc").exists()                # never straight into processing


def test_holds_that_still_stand_or_belong_to_a_person_stay(tmp_path, monkeypatch):
    deep, posed, _ = _wire(monkeypatch, tmp_path)
    kept = [
        _bundle(deep, "20261012_CNT0601_F1", person=True),                 # a person's
        _bundle(deep, "20261012_CNT0602_F1", flags=(                       # tracking hold
            {"category": "dlc_coherence", "severity": "critical",
             "description": "Ruler (SABL-SABR) highly unstable"},)),
        _bundle(deep, "20261012_CNT0603_P1"),                              # Pillar: still missing
        _bundle(deep, "20261012_CNT0604_F1", reason="segmentation_failed"),  # other reason
    ]
    db = FakeDB()

    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert summary["qc_released"] == 0
    assert all(b.exists() for b in kept) and not db.forced
    assert list(posed.iterdir()) == []


def test_nothing_in_the_posed_folder_is_replaced(tmp_path, monkeypatch):
    deep, posed, _ = _wire(monkeypatch, tmp_path)
    stem = "20261012_CNT0615_F4"
    b = _bundle(deep, stem)
    (posed / f"{stem}.mp4").write_bytes(b"another copy")
    db = FakeDB()

    assert rr.scan_review_queues(db, tmp_path / "proc")["qc_released"] == 0

    assert (posed / f"{stem}.mp4").read_bytes() == b"another copy"
    assert (b / f"{stem}.mp4").read_bytes() == b"video" and not db.forced


def test_a_release_that_cannot_finish_leaves_the_bundle_as_it_was(tmp_path, monkeypatch):
    deep, posed, review = _wire(monkeypatch, tmp_path)
    stem = "20261012_CNT0615_F4"
    b = _bundle(deep, stem)
    before = sorted(p.name for p in b.iterdir())

    assert rr.scan_review_queues(FakeDB(fail=True), tmp_path / "proc")["qc_released"] == 0

    assert sorted(p.name for p in b.iterdir()) == before
    assert list(posed.iterdir()) == []
    assert not list((review / "_Problematic").rglob("*.*"))


def test_each_bundle_is_rechecked_once_until_it_changes(tmp_path, monkeypatch):
    deep, _, _ = _wire(monkeypatch, tmp_path)
    _bundle(deep, "20261012_CNT0603_P1")          # stays held: Pillar, outcomes missing
    import mousereach.pipeline.triage as tri
    calls = []
    real = tri.triage_video
    monkeypatch.setattr(tri, "triage_video", lambda *a, **k: calls.append(a) or real(*a, **k))

    for _ in range(3):
        rr.scan_review_queues(FakeDB(), tmp_path / "proc")

    assert len(calls) == 1
