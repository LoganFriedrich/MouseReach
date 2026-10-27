"""The Escalate button sets aside any earlier deep-review clear.

Pressing "Escalate: bad segmentation -> re-segmentation" means a person has found a
new problem with the cuts. A clear from an earlier deep review used to travel
with the bundle, and the return scan -- which reads "cleared" from the marker
alone -- released the video from deep review within seconds, re-ran it on the
same cuts and put it back in triage (20250627_CNT0107_P3, 2026-10-27).
"""
import json

import mousereach.watcher.review_gate as rg
import mousereach.watcher.review_return as rr

STEM = "20240101_ABC0201_P1"


def _triage_bundle(tmp_path, monkeypatch):
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    deep.mkdir(parents=True)
    monkeypatch.setattr(rg.Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(rg.Paths, "TRIAGE_REVIEW", triage)
    b = triage / STEM
    b.mkdir(parents=True)
    (b / f"{STEM}_segments.json").write_text(json.dumps({"boundaries": []}), encoding="utf-8")
    (b / f"{STEM}_deep_review_cleared.json").write_text(
        json.dumps({"cleared_by": "tester", "cleared_at": "2026-09-08T11:45:23"}),
        encoding="utf-8")
    return b, deep


def test_escalating_sets_the_old_clear_aside(tmp_path, monkeypatch):
    b, deep = _triage_bundle(tmp_path, monkeypatch)
    bundle = rg.escalate_to_deep_review(STEM, b, reason="reviewer escalated")
    assert not (bundle / f"{STEM}_deep_review_cleared.json").exists()
    kept = list(bundle.glob(f"{STEM}_deep_review_cleared_superseded_*.json"))
    assert len(kept) == 1                                   # kept on record, not deleted
    assert json.loads(kept[0].read_text(encoding="utf-8"))["cleared_by"] == "tester"
    # and the return scan no longer reads the bundle as cleared
    assert not rr._deep_review_cleared(bundle, STEM)


def test_other_routes_keep_the_clear(tmp_path, monkeypatch):
    """Only the button voids a clear; an ordinary move keeps it."""
    b, deep = _triage_bundle(tmp_path, monkeypatch)
    bundle = rg.route_to_queue(STEM, b, deep, reason="moved", db=None)
    assert (bundle / f"{STEM}_deep_review_cleared.json").exists()


def test_no_clear_nothing_to_set_aside(tmp_path):
    assert rg.supersede_deep_review_clear(STEM, tmp_path) is None


# ---- nothing leaves the re-segmentation queue without being re-segmented ----
from mousereach.review.resegmented import resegmented_since_entry


def _queued(tmp_path, routed_at, seg_extra=None, gt_bounds=None):
    b = tmp_path / STEM
    b.mkdir(parents=True)
    (b / f"{STEM}_routing.json").write_text(json.dumps({"routed_at": routed_at}),
                                            encoding="utf-8")
    seg = {"boundaries": [10, 20, 30]}
    seg.update(seg_extra or {})
    (b / f"{STEM}_segments.json").write_text(json.dumps(seg), encoding="utf-8")
    if gt_bounds is not None:
        (b / f"{STEM}_unified_ground_truth.json").write_text(
            json.dumps({"segmentation": {"boundaries": gt_bounds}}), encoding="utf-8")
    return b


def test_machine_cuts_cannot_leave(tmp_path):
    ok, why = resegmented_since_entry(_queued(tmp_path, "2026-10-27T11:00:00"), STEM)
    assert not ok and "nobody has set its cuts by hand" in why


def test_hand_cuts_from_before_it_was_sent_do_not_count(tmp_path):
    b = _queued(tmp_path, "2026-10-27T11:00:00",
                {"boundary_source": "human", "corrected_at": "2026-08-26T13:04:56"})
    ok, why = resegmented_since_entry(b, STEM)
    assert not ok and "before it was sent here" in why


def test_hand_cuts_saved_after_it_was_sent_let_it_leave(tmp_path):
    b = _queued(tmp_path, "2026-10-27T11:00:00",
                {"boundary_source": "human", "corrected_at": "2026-10-27T12:00:00"})
    assert resegmented_since_entry(b, STEM)[0]


def test_ground_truth_counts_only_when_every_boundary_is_verified(tmp_path):
    later = "2026-10-27T12:00:00"
    some = [{"verified": True, "verified_at": later}, {"verified": False}]
    assert not resegmented_since_entry(
        _queued(tmp_path / "a", "2026-10-27T11:00:00", gt_bounds=some), STEM)[0]
    every = [{"verified": True, "verified_at": later}] * 3
    assert resegmented_since_entry(
        _queued(tmp_path / "b", "2026-10-27T11:00:00", gt_bounds=every), STEM)[0]


def test_return_scan_holds_a_cleared_video_nobody_resegmented(tmp_path, monkeypatch):
    deep = tmp_path / "Review" / "deep_review"
    deep.mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", tmp_path / "Review" / "triage")
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", tmp_path / "Review")
    b = _queued(deep, "2026-10-27T11:00:00",
                {"boundary_source": "human", "corrected_at": "2026-08-26T13:04:56"})
    (b / f"{STEM}_deep_review_cleared.json").write_text("{}", encoding="utf-8")
    returned = []
    monkeypatch.setattr(rr, "_return_to_processing",
                        lambda *a, **k: returned.append(a) or True)
    rr.scan_review_queues(None, tmp_path / "Processing")
    assert returned == []
