"""The review window's "is this video already reviewed?" check reads every
copy of the review, the durable one included, through the same path as the
window's restore and the release check (effective_review).

It read only the bundle's copy and the one beside the filed video. A pellet
reopened with "are you sure?" in the durable copy was then "already reviewed"
here, so the window never offered the video, while the release check held it;
and an answer only the durable copy held was missed.
"""
import json

import pytest

import mousereach.review.causal_review_io as cio
from mousereach.review.causal_review_widget import CausalReviewWidget

from types import SimpleNamespace


def bundle_review_complete(bundle, stem, dirs, triage_only):
    """The window's own check (_bundle_reviewed); self carries only the mode."""
    return CausalReviewWidget._bundle_reviewed(SimpleNamespace(_triage_only=triage_only), bundle)

STEM = "20990101_ABC9901_P1"


def _answer(n, start, when="2099-01-01T10:00:00"):
    return {"segment_num": n, "segment_span": {"start": start, "end": start + 99},
            "human": {"outcome": "untouched"}, "answers": {},
            "reviewer": "reviewer-a", "reviewed_at": when}


def _placeholder(n, start):
    return {"segment_num": n, "segment_span": {"start": start, "end": start + 99},
            "human": {"outcome": None}, "answers": {"reviewed": False}}


def _recheck(n, start):
    return {"segment_num": n, "segment_span": {"start": start, "end": start + 99},
            "human": {"outcome": "untouched"},
            "answers": {"reviewed": False,
                        "recheck": {"asked_at": "2099-01-05T10:00:00", "question": "sure?"}}}


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "records")
    durable = tmp_path / "records" / "reviews"
    durable.mkdir(parents=True)
    bundle = tmp_path / "queue" / STEM
    bundle.mkdir(parents=True)
    (bundle / f"{STEM}_segments.json").write_text(json.dumps({"boundaries": [100, 200, 300]}),
                                                  encoding="utf-8")
    return bundle, durable


def _write(folder, segments):
    (folder / f"{STEM}_causal_review.json").write_text(json.dumps({"segments": segments}),
                                                       encoding="utf-8")


def test_a_pellet_reopened_in_the_durable_copy_is_not_reviewed(site):
    bundle, durable = site
    _write(bundle, [_answer(1, 100), _answer(2, 200)])
    _write(durable, [_answer(1, 100), _recheck(2, 200)])
    assert not bundle_review_complete(bundle, STEM, [bundle], triage_only=False)


def test_an_answer_only_the_durable_copy_holds_counts(site):
    bundle, durable = site
    _write(bundle, [_answer(1, 100), _placeholder(2, 200)])
    _write(durable, [_answer(2, 200)])
    assert bundle_review_complete(bundle, STEM, [bundle], triage_only=False)


def test_triage_only_needs_the_triaged_pellets_answered_in_any_copy(site):
    bundle, durable = site
    (bundle / f"{STEM}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": 1, "outcome": "untouched"},
        {"segment_num": 2, "outcome": "triaged", "flagged_for_review": True}]}),
        encoding="utf-8")
    _write(bundle, [_placeholder(1, 100), _placeholder(2, 200)])
    assert not bundle_review_complete(bundle, STEM, [bundle], triage_only=True)
    _write(durable, [_answer(2, 200)])
    assert bundle_review_complete(bundle, STEM, [bundle], triage_only=True)
