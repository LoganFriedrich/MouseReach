"""Triage walks only the pellets still open when a video loads.

A video comes back (the nightly hand-score check) for new pellets while its
pellet file still flags the ones a person answered before; "Next" used to walk
those too and show the earlier answers (Logan, 2026-10-24).
"""
from mousereach.review.causal_review_widget import open_triaged_indices, segment_answered

SEGS = [{"segment_num": n, "outcome": "untouched"} for n in range(1, 7)]


def _rec(outcome="untouched"):
    return {"algo": {"outcome": outcome}, "human": {"outcome": "untouched"}}


def test_answered_pellets_are_not_walked():
    records = {3: _rec(), 5: _rec()}
    assert open_triaged_indices([1, 2, 3, 4, 5], SEGS, records) == [1, 3, 5]   # segments 2, 4, 6


def test_an_answer_stands_whatever_the_algorithm_says_since():
    records = {3: _rec(outcome="retrieved")}            # the algorithm's call was different then
    assert segment_answered(SEGS[2], records)
    assert open_triaged_indices([2], SEGS, records) == []


def test_nothing_answered_walks_everything():
    assert open_triaged_indices([0, 1, 2], SEGS, {}) == [0, 1, 2]
