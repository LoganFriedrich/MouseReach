"""The review window and the return scan answer "does this bundle have
something to triage?" and "is it waiting for its segmentation to be fixed?"
with ONE rule each (triage_status.bundle_triage / triaged_in_range and
triage_status.held_for_segmentation).

WHY: each side had its own copy, and they disagreed --
  * a flagged bundle with no assignments file read "nothing to triage" in the
    window (so nobody was asked) but "flagged" to the scan (so it was never
    released);
  * an unreadable segments file read "not failed" in the window (offered for
    triage) but "failed" to the scan (moved to deep review under the reviewer),
    and a person's deep-review clear counted only on the scan side;
  * a flag on a segment that does not exist was dropped only on the release
    side: the window offered the bundle with nothing to ask, and a bundle whose
    only flags were such ghosts sat in triage for ever.
"""
import json

import pytest

import mousereach.review.causal_review_io as cio
import mousereach.watcher.review_gate as rg
import mousereach.watcher.review_return as rr
from mousereach.config import Paths
from mousereach.review.causal_review_widget import CausalReviewWidget

STEM = "20990101_XYZ0101_P1"


class Window:
    """The real window's methods on plain attributes (no display needed)."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)

    def __getattr__(self, name):
        return getattr(CausalReviewWidget, name).__get__(self)


def _window():
    return Window(_triage_only=True, _deep_review=False, _worklist=None,
                  _has_triage_cache={}, _held_logged=set(), _pool_cache=None)


class FakeDB:
    def get_video(self, vid):
        return {"video_id": vid, "state": "triage"}

    def force_state(self, *a, **k):
        pass


@pytest.fixture
def queues(tmp_path, monkeypatch):
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage, raising=False)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep, raising=False)
    monkeypatch.setattr(Paths, "REVIEW_ROOT", tmp_path / "Review", raising=False)
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(rr.time, "sleep", lambda s: None)
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "records")
    monkeypatch.setattr(cio, "gt_index", lambda refresh=False: {})
    return triage


def _bundle(queue, outcomes, segments="ok", assignments=True, cleared=False):
    b = queue / STEM
    b.mkdir(parents=True)
    (b / f"{STEM}_manifest.json").write_text(json.dumps({"video_stem": STEM}), encoding="utf-8")
    if segments == "ok":
        (b / f"{STEM}_segments.json").write_text(json.dumps(
            {"overall_confidence": 0.9, "boundaries": [100, 200, 300]}), encoding="utf-8")
    elif segments == "failed":
        (b / f"{STEM}_segments.json").write_text(json.dumps(
            {"overall_confidence": 0.0, "boundaries": [100, 200, 300]}), encoding="utf-8")
    elif segments == "unreadable":
        (b / f"{STEM}_segments.json").write_text("{not json", encoding="utf-8")
    (b / f"{STEM}_pellet_outcomes.json").write_text(json.dumps({"segments": outcomes}),
                                                    encoding="utf-8")
    if assignments:
        (b / f"{STEM}_reach_assignments.json").write_text(json.dumps({"reaches": []}),
                                                          encoding="utf-8")
    if cleared:
        (b / f"{STEM}_deep_review_cleared.json").write_text('{"cleared_by": "person-a"}',
                                                            encoding="utf-8")
    return b


FLAGGED_2 = [{"segment_num": 1, "outcome": "untouched"},
             {"segment_num": 2, "outcome": "triaged", "flagged_for_review": True}]


def _scan(monkeypatch, tmp_path):
    routed, returned = [], []
    monkeypatch.setattr(rg, "route_to_queue",
                        lambda stem, src, dest, **k: routed.append((stem, Path_(dest).name)))
    monkeypatch.setattr(rr, "_return_to_processing",
                        lambda bundle, stem, pd, db, why: returned.append((stem, why)) or True)
    rr.scan_review_queues(FakeDB(), tmp_path / "proc")
    return routed, returned


def Path_(p):
    from pathlib import Path
    return Path(p)


# --- MRR-03a: a flagged bundle with no assignments file -----------------------

def test_a_flagged_bundle_without_assignments_is_offered_in_the_window(queues):
    b = _bundle(queues, FLAGGED_2, assignments=False)
    assert _window()._bundle_eligible(b, queues.parent)


# --- MRR-03b: unreadable segments; a cleared failed segmentation -------------

def test_unreadable_segments_are_held_for_segmentation_on_both_sides(queues, tmp_path, monkeypatch):
    b = _bundle(queues, FLAGGED_2, segments="unreadable")
    assert not _window()._bundle_eligible(b, queues.parent)      # window: not offered
    routed, returned = _scan(monkeypatch, tmp_path)
    assert routed == [(STEM, "deep_review")] and not returned   # scan: to deep review


def test_a_held_bundle_is_logged_once_not_on_every_pool_scan(queues, caplog):
    """The pool is scanned on every "next video"; the skip of a bundle held
    for its segmentation is logged the first time only."""
    import logging
    b = _bundle(queues, FLAGGED_2, segments="unreadable")
    w = _window()
    with caplog.at_level(logging.INFO):
        for _ in range(3):
            assert not w._bundle_eligible(b, queues.parent)
    lines = [r for r in caplog.records if "segmentation must be fixed first" in r.getMessage()]
    assert len(lines) == 1


def test_a_cleared_failed_segmentation_with_open_pellets_is_offered_in_triage(queues, tmp_path, monkeypatch):
    b = _bundle(queues, FLAGGED_2, segments="failed", cleared=True)
    # The scan keeps it in triage, waiting for the answer ...
    routed, returned = _scan(monkeypatch, tmp_path)
    assert not routed and not returned
    # ... so the window must offer it, or nobody ever answers.
    assert _window()._bundle_eligible(b, queues.parent)


# --- MRR-03c: flags on segments that do not exist -----------------------------

def test_a_flag_on_a_nonexistent_segment_is_not_offered(queues):
    b = _bundle(queues, [{"segment_num": 1, "outcome": "untouched"},
                         {"segment_num": 9, "outcome": "triaged", "flagged_for_review": True}])
    assert not _window()._bundle_eligible(b, queues.parent)


def test_a_bundle_whose_only_flags_are_ghosts_is_returned(queues, tmp_path, monkeypatch):
    _bundle(queues, [{"segment_num": 1, "outcome": "untouched"},
                     {"segment_num": 9, "outcome": "triaged", "flagged_for_review": True}])
    routed, returned = _scan(monkeypatch, tmp_path)
    assert not routed and returned == [(STEM, "triage_cleared")]


def test_ghost_flags_do_not_keep_a_review_from_reading_complete(queues):
    from mousereach.review.causal_review_widget import bundle_review_complete
    b = _bundle(queues, [{"segment_num": 2, "outcome": "triaged", "flagged_for_review": True},
                         {"segment_num": 9, "outcome": "triaged", "flagged_for_review": True}])
    (b / f"{STEM}_causal_review.json").write_text(json.dumps({"segments": [
        {"segment_num": 2, "segment_span": {"start": 200, "end": 299},
         "human": {"outcome": "untouched"}, "answers": {},
         "reviewer": "reviewer-a", "reviewed_at": "2099-01-01T10:00:00"}]}), encoding="utf-8")
    assert bundle_review_complete(b, STEM, [b], triage_only=True)


# --- route-to-queue never makes a triage bundle with nothing to ask ----------

@pytest.fixture
def filed(tmp_path, monkeypatch):
    import mousereach.pipeline.manifest as manifest
    analyzed = tmp_path / "Analyzed"
    results = analyzed / "PROJ" / "PROJ_01"
    results.mkdir(parents=True)
    (results / f"{STEM}.mp4").write_bytes(b"video")
    (results / f"{STEM}DLC_resnet50_TestOct1shuffle1_1000.h5").write_bytes(b"pose")
    (results / f"{STEM}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": 1, "outcome": "untouched"},
        {"segment_num": 2, "outcome": "untouched"}]}), encoding="utf-8")
    (results / f"{STEM}_segments.json").write_text(json.dumps(
        {"overall_confidence": 0.9, "boundaries": [100, 200, 300]}), encoding="utf-8")
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", analyzed)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "REVIEW_ROOT", tmp_path / "Review", raising=False)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(Paths, "DLC_STAGING", None, raising=False)
    monkeypatch.setattr(manifest, "declared_dlc_scorer", lambda max_age_s=None: None)
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "records")
    monkeypatch.setattr(cio, "gt_index", lambda refresh=False: {})
    return results, triage


@pytest.mark.parametrize("segs", [None, [9]])
def test_a_triage_route_with_nothing_to_ask_is_refused_and_moves_nothing(filed, segs):
    from mousereach.watcher.route_cli import route_video
    results, triage = filed
    before = sorted(p.name for p in results.iterdir())
    res = route_video(STEM, "triage", "a check", segment_nums=segs)
    assert not res["routed"] and "nothing to ask" in (res["error"] or "")
    assert sorted(p.name for p in results.iterdir()) == before
    assert not (triage / STEM).exists()


def test_a_triage_route_naming_a_real_segment_still_goes(filed):
    from mousereach.watcher.route_cli import route_video
    results, triage = filed
    res = route_video(STEM, "triage", "a check", segment_nums=[2])
    assert res["routed"] and res["flagged"] == [2], res
