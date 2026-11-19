"""The review window builds its queue: lists the bundles, then picks the ones
that still need a person, judged on every copy of the review.

Driven through the window's own methods (_review_pool_paths, _pick_eligible,
_bundle_eligible and whatever they call), looked up on the real class at call
time -- so a method the queue path calls that no longer exists fails here
exactly as it would in the window (it once did: the window raised
AttributeError while building its queue).
"""
import json

import pytest

import mousereach.review.causal_review_io as cio
from mousereach.config import Paths
from mousereach.review.causal_review_widget import CausalReviewWidget


class Window:
    """Plain attributes, plus every method of the real window looked up on
    the class when called (no display needed for the queue path)."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)

    def __getattr__(self, name):
        return getattr(CausalReviewWidget, name).__get__(self)


ANSWERED, UNANSWERED, RECHECKED, HAS_GT, GT_OPEN = (
    "20990101_ABC9901_P1", "20990101_ABC9902_P1", "20990101_ABC9903_P1", "20990101_ABC9904_P1",
    "20990101_ABC9905_P1")


def _answer(n, start):
    return {"segment_num": n, "segment_span": {"start": start, "end": start + 99},
            "human": {"outcome": "untouched"}, "answers": {},
            "reviewer": "reviewer-a", "reviewed_at": "2099-01-01T10:00:00"}


def _placeholder(n, start):
    return {"segment_num": n, "segment_span": {"start": start, "end": start + 99},
            "human": {"outcome": None}, "answers": {"reviewed": False}}


def _bundle(queue, stem, review):
    b = queue / stem
    b.mkdir(parents=True)
    (b / f"{stem}_manifest.json").write_text(json.dumps({"video_stem": stem}), encoding="utf-8")
    (b / f"{stem}_segments.json").write_text(json.dumps(
        {"overall_confidence": 0.9, "boundaries": [100, 200, 300]}), encoding="utf-8")
    (b / f"{stem}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": 1, "outcome": "untouched"},
        {"segment_num": 2, "outcome": "triaged", "flagged_for_review": True}]}),
        encoding="utf-8")
    (b / f"{stem}_reach_assignments.json").write_text(json.dumps({"reaches": []}),
                                                      encoding="utf-8")
    (b / f"{stem}_causal_review.json").write_text(json.dumps({"segments": review}),
                                                  encoding="utf-8")
    return b


@pytest.fixture
def queue(tmp_path, monkeypatch):
    q = tmp_path / "Review" / "triage"
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", q, raising=False)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "Review" / "deep_review", raising=False)
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "records")
    monkeypatch.setattr(cio, "gt_index", lambda refresh=False: {})
    durable = tmp_path / "records" / "reviews"
    durable.mkdir(parents=True)
    _bundle(q, ANSWERED, [_placeholder(1, 100), _answer(2, 200)])
    _bundle(q, UNANSWERED, [_placeholder(1, 100), _placeholder(2, 200)])
    # Answered in the bundle's copy, reopened ("are you sure?") in the durable copy.
    _bundle(q, RECHECKED, [_placeholder(1, 100), _answer(2, 200)])
    (durable / f"{RECHECKED}_causal_review.json").write_text(json.dumps({"segments": [
        {"segment_num": 2, "segment_span": {"start": 200, "end": 299},
         "human": {"outcome": "untouched"},
         "answers": {"reviewed": False, "recheck": {"asked_at": "2099-01-05T10:00:00",
                                                    "question": "sure?"}}}]}),
        encoding="utf-8")
    # Ground truth that answers the flagged pellet: nothing for a person, the
    # return scan releases it (MRR-03d, 2026-11-19).
    b = _bundle(q, HAS_GT, [_placeholder(1, 100), _placeholder(2, 200)])
    (b / f"{HAS_GT}_unified_ground_truth.json").write_text(json.dumps({
        "segmentation": {"boundaries": [{"index": i, "frame": f}
                                        for i, f in enumerate((100, 200, 300))]},
        "outcomes": {"segments": [{"segment_num": 2, "outcome": "untouched",
                                   "determined": True}]}}), encoding="utf-8")
    # Ground truth that answers nothing: the flagged pellet is still a
    # question, so the window offers it (it used to hide every bundle with
    # ground truth, and this one was neither shown nor released).
    b = _bundle(q, GT_OPEN, [_placeholder(1, 100), _placeholder(2, 200)])
    (b / f"{GT_OPEN}_unified_ground_truth.json").write_text("{}", encoding="utf-8")
    return q


def _window(triage_only=True):
    return Window(_triage_only=triage_only, _deep_review=False, _worklist=None,
                  _has_triage_cache={}, _held_logged=set(), _pool_cache=None)


def test_each_bundle_is_judged_on_every_copy(queue):
    w = _window()
    root = queue.parent
    eligible = {b.name for b in w._review_pool_paths(queue, root) if w._bundle_eligible(b, root)}
    assert eligible == {UNANSWERED, RECHECKED, GT_OPEN}


def test_the_next_video_is_always_one_that_needs_a_person(queue):
    w = _window()
    root = queue.parent
    pool = w._review_pool_paths(queue, root)
    assert len(pool) == 5
    picked = set()
    while True:
        b, _ = w._pick_eligible(pool, root)
        if b is None:
            break
        picked.add(b.name)
        pool = [x for x in pool if x != b]
    assert picked == {UNANSWERED, RECHECKED, GT_OPEN}
