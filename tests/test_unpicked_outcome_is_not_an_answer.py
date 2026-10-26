"""A pellet saved with no outcome picked is not an answer.

WHY (2026-10-26): the review window saves a pellet whose outcome question was
left on "-- select outcome --" with the algorithm's "triaged" still in it, and
nothing marking it open. Every check counted that as answered, so the video left
triage and was FILED with a pellet nobody decided -- 125 such records in filed
videos and 121 in the queues -- and the apply step stamped "triaged" onto the
result as a human review. One rule (is_answered) now says what an answer is, and
every check uses it.
"""
import json

import mousereach.watcher.route_cli as rc
from mousereach.config import Paths
from mousereach.review.causal_review_io import (answered_segments, apply_review_overrides,
                                                is_answered)
from mousereach.review.triage_status import resolved_segments

VID = "20250715_ABC0212_P4"


def _rec(n, outcome, **answers):
    return {"segment_num": n, "pellet_num": n,
            "segment_span": {"start": 100 * n, "end": 100 * n + 90},
            "human": {"outcome": outcome, "causal_reach": None}, "answers": answers}


def test_what_counts_as_an_answer():
    assert is_answered(_rec(1, "retrieved", triage_kind="both_uncertain", outcome_pick="retrieved"))
    assert is_answered(_rec(1, "abnormal_exception", abnormal_ranges=[[5, 9]]))
    assert is_answered(_rec(0, None, is_boundary=True))                # no pellet, no outcome needed
    assert not is_answered(_rec(1, "triaged", triage_kind="both_uncertain"))   # left on the sentinel
    assert not is_answered(_rec(1, "uncertain", triage_kind="reach_uncertain"))  # an older tool's word
    assert not is_answered(_rec(1, None, triage_kind="outcome_uncertain"))
    assert not is_answered(_rec(1, "retrieved", reviewed=False))


def test_a_left_open_pellet_keeps_the_video_in_triage():
    review = {"segments": [_rec(1, "retrieved", outcome_pick="retrieved"),
                           _rec(2, "triaged", triage_kind="both_uncertain")]}
    assert resolved_segments(review) == {1}
    current = [{"segment_num": n, "start_frame": 100 * n, "end_frame": 100 * n + 90} for n in (1, 2)]
    assert answered_segments(review, current) == {1}


def test_a_left_open_pellet_is_never_applied_as_a_human_answer():
    outcomes = {"segments": [{"segment_num": 1, "outcome": "triaged"},
                             {"segment_num": 2, "outcome": "triaged"}]}
    by_seg = {1: _rec(1, "triaged", triage_kind="both_uncertain"),
              2: _rec(2, "displaced_sa", outcome_pick="displaced_sa")}
    out = apply_review_overrides(outcomes, by_seg, reviewer="someone")["segments"]
    assert out[0]["outcome"] == "triaged" and out[0]["outcome_source"] == "algo"
    assert out[1]["outcome"] == "displaced_sa" and out[1]["outcome_source"] == "human_review"


def test_routing_sends_a_left_open_pellet_back(tmp_path, monkeypatch):
    a = tmp_path / "Analyzed" / "Connectome" / "ABC02"
    a.mkdir(parents=True)
    segs = [{"segment_num": n, "start_frame": 100 * n, "end_frame": 100 * n + 90} for n in (1, 2)]
    (a / f"{VID}_segments.json").write_text(json.dumps({"segments": segs, "boundaries": [100, 200, 300]}))
    (a / f"{VID}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": 1, "outcome": "triaged"}, {"segment_num": 2, "outcome": "untouched"}]}))
    (a / f"{VID}_causal_review.json").write_text(json.dumps({
        "video_stem": VID, "reviewer": "someone", "reviewed_at": "2026-10-24T10:00:00",
        "segments": [_rec(1, "triaged", triage_kind="both_uncertain"),
                     _rec(2, "untouched", outcome_pick="untouched")]}))
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed", raising=False)
    routed = []
    import mousereach.watcher.review_gate as rg
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append(a))
    r = rc.route_video(VID, "triage", "no outcome was picked", [1, 2])
    assert r["answered"] == [2] and r["flagged"] == [1] and r["routed"]
